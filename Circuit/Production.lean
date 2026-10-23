import Circuit.ProductionInput

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 20000000
set_option maxRecDepth 30000

def productionExpression (d : Graph) (v : Vertex) (R F I : Nat) :
    Expr ((productionReads d v).length * (R * (1 + (F + I)))) :=
  match productionPlan (productionBasis d F) d v with
  | none => .constant false
  | some plan => orderedExpression (plan.compile (readSelections d v R F I)) (readSelections d v R F I v)

def productionCircuit (d : Graph) (v : Vertex) (R F I : Nat) : Circuit :=
  (productionExpression d v R F I).oracle

def productionTarget (d : Graph) (v : Vertex) (R F I : Nat) : Nat :=
  (productionReads d v).length * (R * (1 + (F + I))) + (productionExpression d v R F I).size

theorem productionPlan_missing {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (missing : productionPlan basis d v = none) :
    MultiIntro.multiIntroShapeB d v = false ∧
      (v.HYPOTHESIS || (predsOf d v).isEmpty) = true ∧ ¬∃ i, basis i = v.FORMULA := by
  unfold productionPlan at missing
  split at missing
  · contradiction
  · rename_i noMulti
    split at missing
    · rename_i source
      split at missing
      · contradiction
      · exact ⟨Bool.eq_false_iff.mpr noMulti, source, by assumption⟩
    · contradiction

theorem sourceRoutes_mismatch {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (routes : List FlowPair) (fits : ∀ r ∈ routes, RouteFits basis (graphWords d) r)
    (missing : ¬∃ i, basis i = v.FORMULA) : routes ≠ sourceRoutes d v := by
  intro equal
  subst routes
  cases ancestors : ancestorsInto d v with
  | nil =>
    have fitting := fits (normalizeDep [v.FORMULA], []) (by simp [sourceRoutes, ancestors])
    exact missing (fitting.2.1 v.FORMULA (by simp))
  | cons head rest =>
    have fitting := fits (normalizeDep [v.FORMULA], head.COLOURS) (by simp [sourceRoutes, ancestors])
    exact missing (fitting.2.1 v.FORMULA (by simp))

theorem productionExpression_correct (d : Graph) (v : Vertex) (R F I : Nat)
    (c : RouteCertificate.CompleteProfileCertificate) (slot : Nat) (metadata : c.graph = d)
    (fits : ProductionFits d v R F I c) :
    (productionExpression d v R F I).value (matrixState (encodedReads d v R F I c fits)) =
      Flow.Robustness.check d c (.productionRow slot v) := by
  have ownMember : v ∈ productionReads d v := by simp [productionReads]
  have capacity : ([] :: graphWords d).length ≤ 2 ^ I := (fits v ownMember).2.1
  have own := readSelections_encode d v R F I c fits v
  have priorOwn : restrictedPrior d v c v = RouteCertificate.stored c v := by
    simp [restrictedPrior, ownMember]
  rw [priorOwn] at own
  have checker : Flow.Robustness.check d c (.productionRow slot v) =
      decide (RouteCertificate.stored c v = MultiIntro.step .general d (RouteCertificate.stored c) v) := by
    simp only [Flow.Robustness.check, MultiIntroPositions.check, RouteCertificate.localExpectedFor,
      MultiIntro.kernel, metadata]
  rw [checker]
  unfold productionExpression
  cases built : productionPlan (productionBasis d F) d v with
  | none =>
    have conditions := productionPlan_missing (productionBasis d F) d v built
    have expected : MultiIntro.step .general d (RouteCertificate.stored c) v = sourceRoutes d v := by
      simp [MultiIntro.step, conditions.1, CompleteProfileKernel.tagged, LambdaFlow.stepWith,
        conditions.2.1, sourceRoutes]
    simp only [Expr.value, expected]
    exact (decide_eq_false (sourceRoutes_mismatch (productionBasis d F) d v _
      (fits v ownMember).2.2 conditions.2.2)).symm
  | some plan =>
    have compiled := plan.correct (restrictedPrior d v c) (readSelections d v R F I)
      (matrixState (encodedReads d v R F I c fits)) capacity (readSelections_encode d v R F I c fits)
    rw [productionPlan_meaning (productionBasis d F) d v plan built,
      restricted_step d v c] at compiled
    rw [orderedExpression_correct, compiled.bits, own.bits]
    simp only [routeBits_map_equal_iff (productionBasis d F) (graphWords d)
      _ _ compiled.fitting own.fitting capacity, eq_comm]




theorem production_correct (d : Graph) (v : Vertex) (listed : v ∈ d.NODES)
    (R F I : Nat) (c : RouteCertificate.CompleteProfileCertificate) (slot : Nat)
    (metadata : c.graph = d) (fits : ProductionFits d v R F I c) :
    eval (productionCircuit d v R F I) (matrixState (encodedReads d v R F I c fits)) =
      Function.update (matrixState (encodedReads d v R F I c fits))
        (productionTarget d v R F I) (Flow.Robustness.check d c (.productionRow slot v)) := by
  rw [productionCircuit, Expr.oracle_correct _ _ (matrixState_clean _),
    productionExpression_correct d v R F I c slot metadata fits]
  rfl

theorem production_ancillas_clean (d : Graph) (v : Vertex) (R F I : Nat)
    (codes : Fin (productionReads d v).length → List.Vector (Slot F I) R) (wire : Nat)
    (work : (productionReads d v).length * (R * (1 + (F + I))) ≤ wire)
    (notTarget : wire ≠ productionTarget d v R F I) :
    eval (productionCircuit d v R F I) (matrixState codes) wire = false := by
  rw [productionCircuit, (productionExpression d v R F I).oracle_restores _ (matrixState_clean _) wire notTarget]
  exact matrixState_clean codes wire work

end Semantic.Circuit
