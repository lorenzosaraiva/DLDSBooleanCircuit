import Infrastructure.LambdaFlowCache
import Infrastructure.CompleteLabelReference





open Semantic FlowSpec
namespace SingleIntroductionReference

def compressorLabel (edge : Deduction) : LambdaFlow.Label :=
  if edge.DEPENDENCY.isEmpty then .Lambda else .Bits edge.DEPENDENCY

def compressorOverlay (d : Graph) : LambdaFlow.LGraph := ⟨d, compressorLabel⟩

abbrev guard (d : Graph) : LambdaFlow.Guard := LambdaFlow.labelGuard (compressorOverlay d)

abbrev routes (d : Graph) (fuel : Nat) (v : Vertex) : List FlowPair :=
  LambdaFlow.lambdaFlowAt (compressorOverlay d) fuel v

def rootsB (d : Graph) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let profile := routes d (stdFuel d) v
     !profile.isEmpty && profile.all fun route => route.1.isEmpty && route.2.isEmpty)

def ReferenceVerdict (d : Graph) : Prop :=
  structuralValid d ∧ LeveledColored d ∧ rootsB d = true

def RootProfilesNonempty (d : Graph) : Prop :=
  ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
    routes d (stdFuel d) root ≠ []

abbrev table (d : Graph) : FlowTable := LambdaFlow.lambdaTable (compressorOverlay d)

def rootsFromTableB (d : Graph) (rows : FlowTable) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let profile := LambdaFlow.fromTable d (guard d) (guard d) (stdFuel d) rows v
     !profile.isEmpty && profile.all fun route => route.1.isEmpty && route.2.isEmpty)

theorem rootsFromTable_eq (d : Graph) : rootsFromTableB d (table d) = rootsB d := by
  change rootsFromTableB d
    (LambdaFlow.tableAt d (guard d) (guard d) (stdFuel d)) = rootsB d
  simp only [rootsFromTableB,
    LambdaFlow.tableAt_eq_spec, LambdaFlow.fromTable_spec]
  rfl

theorem rootsB_iff (d : Graph) : rootsB d = true ↔
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      routes d (stdFuel d) root ≠ [] ∧
      ∀ route ∈ routes d (stdFuel d) root, route.1 = [] ∧ route.2 = [] := by
  simp only [rootsB, List.all_eq_true, Bool.or_eq_true, Bool.and_eq_true,
    Bool.not_eq_true', List.isEmpty_eq_false_iff, List.isEmpty_iff]
  constructor
  · intro checked root member terminal
    exact (checked root member).resolve_left (by simp [terminal])
  · intro checked root member
    by_cases terminal : get_rule.outgoing root d = []
    · exact Or.inr (checked root member terminal)
    · exact Or.inl (by simpa using terminal)

theorem ReferenceVerdict.nonempty (d : Graph) (valid : ReferenceVerdict d) : RootProfilesNonempty d :=
  fun root member terminal => ((rootsB_iff d).mp valid.2.2 root member terminal).1

theorem guard_implies_historical (d : Graph) (dep : Dep) (edge : Deduction)
    (passed : guard d dep edge = true) : LambdaFlow.historicalGuard dep edge = true := by
  by_cases empty : edge.DEPENDENCY.isEmpty = true
  · simp [LambdaFlow.historicalGuard, depSimilarB, empty]
  · have sets : depSetEq dep edge.DEPENDENCY = true := by
      simpa [guard, LambdaFlow.labelGuard, compressorOverlay, compressorLabel,
        empty, LambdaFlow.labelAgreesB] using passed
    simp [LambdaFlow.historicalGuard, depSimilarB, sets]

#print axioms rootsFromTable_eq
#print axioms rootsB_iff
#print axioms guard_implies_historical
end SingleIntroductionReference
