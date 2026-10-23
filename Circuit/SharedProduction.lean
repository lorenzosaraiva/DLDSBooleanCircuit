import Circuit.SharedComparison
import Circuit.SharedPlan
import Circuit.Production

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 30000000
set_option maxRecDepth 20000

abbrev productionInputCount (d : Graph) (v : Vertex) (R F I : Nat) : Nat :=
  (productionReads d v).length * (R * (1 + (F + I)))

def productionAssemblyShared (d : Graph) (v : Vertex) (R F I : Nat) : Nat × Shared.Store :=
  let n := productionInputCount d v R F I
  let inputs := readSelections d v R F I
  match productionPlan (productionBasis d F) d v with
  | none => Shared.expression n (.constant false : Expr n) #[]
  | some plan =>
      let produced := Shared.compilePlan inputs plan #[]
      let stored := Shared.selections n (inputs v) produced.2
      Shared.ordered n produced.1 stored.1 stored.2

theorem productionAssemblyShared_bound (d : Graph) (v : Vertex) (R F I : Nat) :
    (productionAssemblyShared d v R F I).1 <
      productionInputCount d v R F I + (productionAssemblyShared d v R F I).2.size := by
  unfold productionAssemblyShared
  cases built : productionPlan (productionBasis d F) d v with
  | none => exact (Shared.expression_correct _ _ #[] (by simp) (fun _ => false)).2.1
  | some plan =>
    have source := Shared.compilePlan_bounds plan (readSelections d v R F I) #[]
    have stored := Shared.selections_correct (productionInputCount d v R F I)
      (readSelections d v R F I v) (Shared.compilePlan (readSelections d v R F I) plan #[]).2
      (by simp only [productionInputCount]; omega) (fun _ => false)
    exact (Shared.ordered_correct _ _ _ _ (fun _ => false)
      (Shared.Bounded.weaken source.2 (Nat.add_le_add_left stored.1.size_le _)) stored.2.1).2.1

def productionProgramShared (d : Graph) (v : Vertex) (R F I : Nat) :
    Shared.Program (productionInputCount d v R F I) :=
  let result := productionAssemblyShared d v R F I
  ⟨Shared.assemble _ result.2.toList, ⟨result.1, by simpa using productionAssemblyShared_bound d v R F I⟩⟩

def productionCircuitShared (d : Graph) (v : Vertex) (R F I : Nat) : Circuit :=
  (productionProgramShared d v R F I).oracle

def productionTargetShared (d : Graph) (v : Vertex) (R F I : Nat) : Nat :=
  (productionProgramShared d v R F I).target

theorem productionProgramShared_value (d : Graph) (v : Vertex) (R F I : Nat) (state : Bits) :
    (productionProgramShared d v R F I).value state =
      Shared.memory (productionInputCount d v R F I) (productionAssemblyShared d v R F I).2 state
        (productionAssemblyShared d v R F I).1 := rfl

theorem productionProgramShared_correct (d : Graph) (v : Vertex) (R F I : Nat)
    (c : RouteCertificate.CompleteProfileCertificate) (slot : Nat) (metadata : c.graph = d)
    (fits : ProductionFits d v R F I c) :
    (productionProgramShared d v R F I).value (matrixState (encodedReads d v R F I c fits)) =
      Flow.Robustness.check d c (.productionRow slot v) := by
  let state := matrixState (encodedReads d v R F I c fits)
  have capacity : ([] :: graphWords d).length ≤ 2 ^ I := (fits v (by simp [productionReads])).2.1
  have inputCorrect := readSelections_encode d v R F I c fits
  rw [productionProgramShared_value, ← productionExpression_correct d v R F I c slot metadata fits]
  unfold productionAssemblyShared productionExpression
  cases built : productionPlan (productionBasis d F) d v with
  | none =>
    exact (Shared.expression_correct _ (.constant false) #[] (by simp) state).2.2
  | some plan =>
    obtain ⟨sourceExtension, sourceBounds, sourceCorrect⟩ := Shared.compilePlan_correct plan
      (restrictedPrior d v c) (readSelections d v R F I) state capacity inputCorrect #[]
    have storedEncoded := Shared.input_encodes (productionBasis d F) (graphWords d)
      (readSelections d v R F I v) (restrictedPrior d v c v) state
      (Shared.compilePlan (readSelections d v R F I) plan #[]).2 capacity (inputCorrect v)
    obtain ⟨storedExtension, storedBounds, storedCorrect⟩ := Shared.selections_represents
      (productionInputCount d v R F I) (productionBasis d F) (graphWords d) _ _
      (Shared.compilePlan (readSelections d v R F I) plan #[]).2
      (by simp only [productionInputCount]; omega) state storedEncoded
    have sourceNow := sourceCorrect.stable storedExtension sourceBounds
    have compared := Shared.ordered_correct (productionInputCount d v R F I) _ _ _ state
      (Shared.Bounded.weaken sourceBounds (Nat.add_le_add_left storedExtension.size_le _)) storedBounds
    have old := plan.correct (restrictedPrior d v c) (readSelections d v R F I) state capacity inputCorrect
    rw [compared.2.2, sourceNow.bits, storedCorrect.bits, orderedExpression_correct, old.bits, (inputCorrect v).bits]

theorem production_correct_shared (d : Graph) (v : Vertex) (listed : v ∈ d.NODES)
    (R F I : Nat) (c : RouteCertificate.CompleteProfileCertificate) (slot : Nat)
    (metadata : c.graph = d) (fits : ProductionFits d v R F I c) :
    eval (productionCircuitShared d v R F I) (matrixState (encodedReads d v R F I c fits)) =
      Function.update (matrixState (encodedReads d v R F I c fits))
        (productionTargetShared d v R F I) (Flow.Robustness.check d c (.productionRow slot v)) := by
  rw [productionCircuitShared, Shared.Program.oracle_correct (productionProgramShared d v R F I) _ (matrixState_clean _),
    productionProgramShared_correct d v R F I c slot metadata fits]
  rfl

theorem production_ancillas_clean_shared (d : Graph) (v : Vertex) (R F I : Nat)
    (codes : Fin (productionReads d v).length → List.Vector (Slot F I) R) (wire : Nat)
    (work : productionInputCount d v R F I ≤ wire) (notTarget : wire ≠ productionTargetShared d v R F I) :
    eval (productionCircuitShared d v R F I) (matrixState codes) wire = false := by
  rw [productionCircuitShared, Shared.Program.oracle_correct (productionProgramShared d v R F I) _ (matrixState_clean _)]
  change wire ≠ (productionProgramShared d v R F I).target at notTarget
  rw [Function.update_of_ne notTarget]
  exact matrixState_clean codes wire work

end Semantic.Circuit
