import Circuit.Basic
import Mathlib.LinearAlgebra.Matrix.Permutation
import Mathlib.LinearAlgebra.UnitaryGroup

namespace Semantic.Circuit

noncomputable section


abbrev Register (N : Nat) := Fin N → Bool

def SupportBelow (N : Nat) (circuit : Circuit) : Prop :=
  ∀ wire ∈ support circuit, wire < N

def extendRegister {N : Nat} (state : Register N) : Bits :=
  fun wire => if h : wire < N then state ⟨wire, h⟩ else false

def restrictRegister (N : Nat) (state : Bits) : Register N := fun wire => state wire.val

@[simp] theorem restrict_extend {N : Nat} (state : Register N) :
    restrictRegister N (extendRegister state) = state := by
  funext wire
  simp [restrictRegister, extendRegister]

theorem extend_restrict {N : Nat} (state : Bits)
    (clean : ∀ wire, N ≤ wire → state wire = false) :
    extendRegister (restrictRegister N state) = state := by
  funext wire
  by_cases h : wire < N
  · simp [extendRegister, restrictRegister, h]
  · simp [extendRegister, h, clean wire (by omega)]

theorem supportBelow_inverse {N : Nat} {circuit : Circuit} (bounded : SupportBelow N circuit) :
    SupportBelow N (inverse circuit) := by
  intro wire member
  exact bounded wire ((support_inverse circuit wire).mp member)

def registerEval (N : Nat) (circuit : Circuit) (state : Register N) : Register N :=
  restrictRegister N (eval circuit (extendRegister state))

theorem extend_registerEval {N : Nat} (circuit : Circuit) (bounded : SupportBelow N circuit)
    (state : Register N) :
    extendRegister (registerEval N circuit state) = eval circuit (extendRegister state) := by
  apply extend_restrict
  intro wire outside
  rw [outside_support circuit _ wire (by
    intro member
    have := bounded wire member
    omega)]
  simp [extendRegister, show ¬ wire < N by omega]

theorem registerEval_inverse {N : Nat} (circuit : Circuit) (bounded : SupportBelow N circuit)
    (state : Register N) :
    registerEval N (inverse circuit) (registerEval N circuit state) = state := by
  unfold registerEval
  rw [show extendRegister (restrictRegister N (eval circuit (extendRegister state))) =
    eval circuit (extendRegister state) from extend_registerEval circuit bounded state]
  rw [inverse_undoes, restrict_extend]


def circuitPermutation (N : Nat) (circuit : Circuit) (bounded : SupportBelow N circuit) :
    Equiv.Perm (Register N) where
  toFun := registerEval N circuit
  invFun := registerEval N (inverse circuit)
  left_inv := registerEval_inverse circuit bounded
  right_inv state := by
    simpa only [inverse_inverse] using
      registerEval_inverse (inverse circuit) (supportBelow_inverse bounded) state

theorem registerEval_bijective {N : Nat} (circuit : Circuit) (bounded : SupportBelow N circuit) :
    Function.Bijective (registerEval N circuit) := (circuitPermutation N circuit bounded).bijective

theorem registerEval_append {N : Nat} (left right : Circuit) (bounded : SupportBelow N left)
    (state : Register N) :
    registerEval N (left ++ right) state = registerEval N right (registerEval N left state) := by
  unfold registerEval
  rw [show extendRegister (restrictRegister N (eval left (extendRegister state))) =
    eval left (extendRegister state) from extend_registerEval left bounded state, eval_append]


def circuitUnitary (N : Nat) (circuit : Circuit) (bounded : SupportBelow N circuit) :
    Matrix (Register N) (Register N) ℂ := by
  classical
  exact (circuitPermutation N circuit bounded)⁻¹.permMatrix ℂ

theorem circuitUnitary_unitary {N : Nat} (circuit : Circuit) (bounded : SupportBelow N circuit) :
    circuitUnitary N circuit bounded ∈ Matrix.unitaryGroup (Register N) ℂ := by
  classical
  apply Matrix.mem_unitaryGroup_iff.mpr
  change _ * (circuitUnitary N circuit bounded).conjTranspose = 1
  unfold circuitUnitary
  rw [Matrix.conjTranspose_permMatrix, ← Matrix.permMatrix_mul]
  simp

def basisVector {N : Nat} (state : Register N) : Register N → ℂ := by
  classical
  exact Pi.single state 1

theorem circuitUnitary_basis {N : Nat} (circuit : Circuit) (bounded : SupportBelow N circuit)
    (state : Register N) :
    (circuitUnitary N circuit bounded).mulVec (basisVector state) =
      basisVector (registerEval N circuit state) := by
  classical
  rw [circuitUnitary, Matrix.permMatrix_mulVec]
  funext output
  simp only [basisVector, Function.comp_apply, Pi.single_apply]
  have equal : (circuitPermutation N circuit bounded)⁻¹ output = state ↔
      output = registerEval N circuit state := by
    exact Equiv.symm_apply_eq (circuitPermutation N circuit bounded)
  simp only [equal]

theorem circuitUnitary_append {N : Nat} (left right : Circuit)
    (leftBound : SupportBelow N left) (rightBound : SupportBelow N right)
    (bothBound : SupportBelow N (left ++ right)) :
    circuitUnitary N (left ++ right) bothBound =
      circuitUnitary N right rightBound * circuitUnitary N left leftBound := by
  classical
  have composed : circuitPermutation N (left ++ right) bothBound =
      circuitPermutation N right rightBound * circuitPermutation N left leftBound := by
    apply Equiv.ext
    intro state
    exact registerEval_append left right leftBound state
  unfold circuitUnitary
  rw [composed, mul_inv_rev, Matrix.permMatrix_mul]

end
end Semantic.Circuit
