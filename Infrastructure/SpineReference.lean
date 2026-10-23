import Infrastructure.DependencyGuardRedundancy

open Semantic FlowSpec SelfCertifying

namespace Spine








def LabelSound (lg : LambdaFlow.LGraph) : Prop :=
  ∀ edge ∈ lg.graph.EDGES,
    ∀ route ∈ LambdaFlow.guardFreeFlowAt lg.graph (stdFuel lg.graph) edge.START,
      ∀ tail, consume edge route.2 = some tail →
        LambdaFlow.labelAgreesB (lg.label edge) route.1 = true


def UValid (lg : LambdaFlow.LGraph) : Prop :=
  structuralValid lg.graph ∧ LeveledColored lg.graph ∧
    LambdaFlow.Def23DecoratedB lg = true ∧ LambdaFlow.guardFreeRootsB lg.graph = true




def UValidReading (lg : LambdaFlow.LGraph) : Prop :=
  structuralValid lg.graph ∧ LeveledColored lg.graph ∧
    LabelSound lg ∧ LambdaFlow.guardFreeRootsB lg.graph = true



theorem def23_implies_labelSound (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph)
    (labels : LambdaFlow.Def23DecoratedB lg = true) : LabelSound lg := by
  intro edge hedge route member tail consumed
  exact DependencyGuard.consumed_guard_at_every_fuel lg structural labels edge hedge
    (stdFuel lg.graph) route member tail consumed

theorem labelSound_at_every_fuel (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph) (labels : LabelSound lg)
    (edge : Deduction) (hedge : edge ∈ lg.graph.EDGES)
    (fuel : Nat) (route : FlowPair)
    (member : route ∈ LambdaFlow.guardFreeFlowAt lg.graph fuel edge.START)
    (tail : ColourPath) (consumed : consume edge route.2 = some tail) :
    LambdaFlow.labelGuard lg route.1 edge = true := by
  have atStandard := DependencyGuard.evaluateAt_member_at_standard lg.graph
    LambdaFlow.allowAll LambdaFlow.allowAll structural
    (structural.2.1 edge hedge).1 fuel route member
  exact labels edge hedge route atStandard tail consumed

theorem guards_redundant_of_labelSound_eq (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph) (labels : LabelSound lg) :
    ∀ fuel v, LambdaFlow.lambdaFlowAt lg fuel v =
      LambdaFlow.guardFreeFlowAt lg.graph fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
    intro v
    simp only [LambdaFlow.lambdaFlowAt, LambdaFlow.guardFreeFlowAt, LambdaFlow.evaluateAt]
    rw [show LambdaFlow.evaluateAt lg.graph (LambdaFlow.labelGuard lg)
        (LambdaFlow.labelGuard lg) fuel =
        LambdaFlow.evaluateAt lg.graph LambdaFlow.allowAll LambdaFlow.allowAll fuel
      from funext ih]
    apply DependencyGuard.step_guard_eq
    intro u finish edge hedge route hroute tail consumed
    obtain ⟨start, _, edgeMember⟩ := mem_edgesBetween hedge
    apply labelSound_at_every_fuel lg structural labels edge edgeMember fuel route
      _ tail consumed
    simpa only [start] using hroute

theorem guards_redundant_of_labelSound (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph) (_levels : LeveledColored lg.graph)
    (labels : LabelSound lg) :
    ∀ fuel v, ListSetEq (LambdaFlow.lambdaFlowAt lg fuel v)
      (LambdaFlow.guardFreeFlowAt lg.graph fuel v) := by
  intro fuel v route
  rw [guards_redundant_of_labelSound_eq lg structural labels fuel v]

theorem UValid.toReading (lg : LambdaFlow.LGraph) (valid : UValid lg) :
    UValidReading lg :=
  ⟨valid.1, valid.2.1, def23_implies_labelSound lg valid.1 valid.2.2.1, valid.2.2.2⟩

end Spine
