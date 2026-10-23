import Semantic.Circuit.Unitary
import Semantic.Circuit.OracleXor

namespace Semantic.Circuit

noncomputable section
open scoped BigOperators

abbrev OraclePosition (d : Graph) := Fin (MultiIntroPositions.positions d).length

def failureFlag (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (p : OraclePosition d) : Bool :=
  !(Flow.Robustness.check d c ((MultiIntroPositions.positions d).get p))

def oracleRegister (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (p : OraclePosition d) (target : Bool) : Register (oracleWireCount d c R F I) :=
  restrictRegister (oracleWireCount d c R F I) (oracleState (oracleIndexWidth d) p.val target)

theorem extend_oracleRegister (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (p : OraclePosition d) (target : Bool) :
    extendRegister (oracleRegister d c R F I p target) =
      oracleState (oracleIndexWidth d) p.val target := by
  apply extend_restrict
  intro wire outside
  apply oracleState_clean (oracleIndexWidth d) p.val (oracleWireCount d c R F I) target
  · simp only [oracleWireCount]
    omega
  · exact outside

theorem oracle_registerEval (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I)
    (p : OraclePosition d) (target : Bool) :
    registerEval (oracleWireCount d c R F I) (oracle d c R F I)
      (oracleRegister d c R F I p target) =
      oracleRegister d c R F I p (target ^^ failureFlag d c p) := by
  unfold registerEval
  rw [extend_oracleRegister, oracle_xor d c R F I metadata fits]
  rfl

def oracleUnitary (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    Matrix (Register (oracleWireCount d c R F I)) (Register (oracleWireCount d c R F I)) ℂ :=
  circuitUnitary (oracleWireCount d c R F I) (oracle d c R F I) (oracle_wire_bound d c R F I)

theorem oracleUnitary_unitary (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    oracleUnitary d c R F I ∈ Matrix.unitaryGroup (Register (oracleWireCount d c R F I)) ℂ :=
  circuitUnitary_unitary _ _

def minusState (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (p : OraclePosition d) : Register (oracleWireCount d c R F I) → ℂ :=
  ((Real.sqrt 2 : ℂ)⁻¹) •
    (basisVector (oracleRegister d c R F I p false) -
      basisVector (oracleRegister d c R F I p true))

def phaseSign (failure : Bool) : ℂ := (-1 : ℂ) ^ failure.toNat


theorem oracle_minus (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I) (p : OraclePosition d) :
    (oracleUnitary d c R F I).mulVec (minusState d c R F I p) =
      phaseSign (failureFlag d c p) • minusState d c R F I p := by
  rw [minusState, Matrix.mulVec_smul, Matrix.mulVec_sub]
  simp only [oracleUnitary, circuitUnitary_basis,
    oracle_registerEval d c R F I metadata fits]
  cases failed : failureFlag d c p <;>
    simp [phaseSign, minusState, failed, sub_eq_add_neg, smul_add, smul_neg]
  abel


def encodeMinus (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (amplitude : OraclePosition d → ℂ) : Register (oracleWireCount d c R F I) → ℂ :=
  ∑ p, amplitude p • minusState d c R F I p


theorem oracle_phase (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I)
    (amplitude : OraclePosition d → ℂ) :
    (oracleUnitary d c R F I).mulVec (encodeMinus d c R F I amplitude) =
      encodeMinus d c R F I (fun p => phaseSign (failureFlag d c p) * amplitude p) := by
  unfold encodeMinus
  rw [Matrix.mulVec_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [Matrix.mulVec_smul, oracle_minus d c R F I metadata fits, smul_smul, mul_comm]


def markedProjection (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) :
    Matrix (OraclePosition d) (OraclePosition d) ℂ :=
  Matrix.diagonal (fun p => if failureFlag d c p then 1 else 0)

theorem markedProjection_idempotent (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) :
    markedProjection d c * markedProjection d c = markedProjection d c := by
  classical
  rw [markedProjection, Matrix.diagonal_mul_diagonal]
  congr 1
  funext p
  cases failureFlag d c p <;> simp

theorem markedProjection_selfAdjoint (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) :
    (markedProjection d c).conjTranspose = markedProjection d c := by
  classical
  rw [markedProjection, Matrix.diagonal_conjTranspose]
  congr 1
  funext p
  change star (if failureFlag d c p then (1 : ℂ) else 0) =
    if failureFlag d c p then (1 : ℂ) else 0
  cases failureFlag d c p <;> simp

theorem marked_reflection_apply (d : Graph) (c : RouteCertificate.CompleteProfileCertificate)
    (amplitude : OraclePosition d → ℂ) :
    (1 - (2 : ℂ) • markedProjection d c).mulVec amplitude =
      fun p => phaseSign (failureFlag d c p) * amplitude p := by
  classical
  funext p
  cases failed : failureFlag d c p <;>
    simp [Matrix.sub_mulVec, Matrix.smul_mulVec, Matrix.one_mulVec,
      markedProjection, Matrix.mulVec_diagonal, phaseSign, failed]
  ring


theorem oracle_reflection (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I)
    (amplitude : OraclePosition d → ℂ) :
    (oracleUnitary d c R F I).mulVec (encodeMinus d c R F I amplitude) =
      encodeMinus d c R F I ((1 - (2 : ℂ) • markedProjection d c).mulVec amplitude) := by
  rw [marked_reflection_apply]
  exact oracle_phase d c R F I metadata fits amplitude

theorem marked_nonempty_iff_invalid (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I) :
    (∃ p : OraclePosition d, failureFlag d c p = true) ↔
      ¬ Flow.Robustness.StrongCertificate d c := by
  rw [← oracle_marks_invalid d c R F I metadata fits]
  simp only [oracle_correct d c R F I metadata fits, Function.update_self, failureFlag]

end
end Semantic.Circuit
