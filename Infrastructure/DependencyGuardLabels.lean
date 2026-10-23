import Infrastructure.DependencyGuardStability

open Semantic FlowSpec SelfCertifying

namespace DependencyGuard

set_option maxHeartbeats 1500000
set_option maxRecDepth 100000

private theorem exactLabel_agrees (label : LambdaFlow.Label) (dependency : Dep)
    (exact : Decoration.exactLabelB (LambdaFlow.toPaperLabel label) dependency = true) :
    LambdaFlow.labelAgreesB label dependency = true := by
  cases label <;> simp_all [Decoration.exactLabelB, LambdaFlow.toPaperLabel,
    LambdaFlow.labelAgreesB]





theorem paperDecorationAtB_consumed (lg : LambdaFlow.LGraph) (v : Vertex)
    (raw : List FlowPair)
    (valid : Decoration.paperDecorationAtB lg.graph
      (fun e => LambdaFlow.toPaperLabel (lg.label e)) v raw = true)
    (edge : Deduction) (outgoing : edge ∈ get_rule.outgoing v lg.graph)
    (route : FlowPair) (member : route ∈ raw) (tail : ColourPath)
    (consumed : consume edge route.2 = some tail) :
    LambdaFlow.labelGuard lg route.1 edge = true := by
  have hm : route ∈ raw.eraseDups := List.mem_eraseDups.mpr member
  have he : edge ∈ (get_rule.outgoing v lg.graph).eraseDups :=
    List.mem_eraseDups.mpr outgoing
  have hc := consume_some_colour consumed
  unfold Decoration.paperDecorationAtB at valid
  cases hrs : raw.eraseDups with
  | nil => simp [hrs] at hm
  | cons first rest =>
    cases rest with
    | nil =>
      have hr : route = first := by simpa [hrs] using hm
      subst route
      cases hos : (get_rule.outgoing v lg.graph).eraseDups with
      | nil => simp [hos] at he
      | cons out outs =>
        cases outs with
        | nil =>
          have hedge : edge = out := by simpa [hos] using he
          subst edge
          simp only [hrs, hos, Bool.and_eq_true] at valid
          exact exactLabel_agrees _ _ valid.2
        | cons next more => simp [hrs, hos] at valid
    | cons second rest =>
      simp only [hrs, Bool.and_eq_true] at valid
      have colorMember : headColour route.2 ∈
          ((first :: second :: rest).map fun r => headColour r.2).eraseDups := by
        apply List.mem_eraseDups.mpr
        exact List.mem_map.mpr ⟨route, by simpa [hrs] using hm, rfl⟩
      have checked := List.all_eq_true.mp valid.1 (headColour route.2) colorMember
      have grouped : route ∈ (first :: second :: rest).filter
          (fun r => headColour r.2 == headColour route.2) := by
        simp only [List.mem_filter, beq_self_eq_true, and_true]
        simpa [hrs] using hm
      have edgeGrouped : edge ∈ (get_rule.outgoing v lg.graph).eraseDups.filter
          (fun e => e.COLOUR == headColour route.2) := by
        simp only [List.mem_filter, beq_iff_eq]
        exact ⟨he, hc⟩
      cases hes : (get_rule.outgoing v lg.graph).eraseDups.filter
          (fun e => e.COLOUR == headColour route.2) with
      | nil => simp [hes] at edgeGrouped
      | cons out outs =>
        cases outs with
        | nil =>
          have hedge : edge = out := by simpa [hes] using edgeGrouped
          subst edge
          cases hgs : (first :: second :: rest).filter
              (fun r => headColour r.2 == headColour route.2) with
          | nil => simp [hgs] at grouped
          | cons r rs =>
            cases rs with
            | nil =>
              have same : route = r := by simpa [hgs] using grouped
              have exact : Decoration.exactLabelB
                  (LambdaFlow.toPaperLabel (lg.label out)) r.1 = true := by
                simpa only [hes, hgs] using checked
              simpa only [same] using exactLabel_agrees (lg.label out) r.1 exact
            | cons r2 more =>
              cases hl : lg.label out <;>
                simp_all [LambdaFlow.labelGuard, LambdaFlow.labelAgreesB,
                  LambdaFlow.toPaperLabel]
        | cons next more => simp [hes] at checked




theorem consumed_guard_at_every_fuel (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph)
    (labels : LambdaFlow.Def23DecoratedB lg = true)
    (edge : Deduction) (hedge : edge ∈ lg.graph.EDGES)
    (fuel : Nat) (route : FlowPair)
    (member : route ∈ LambdaFlow.guardFreeFlowAt lg.graph fuel edge.START)
    (tail : ColourPath) (consumed : consume edge route.2 = some tail) :
    LambdaFlow.labelGuard lg route.1 edge = true := by
  have sourceMember := (structural.2.1 edge hedge).1
  have atStandard := evaluateAt_member_at_standard lg.graph
    LambdaFlow.allowAll LambdaFlow.allowAll structural sourceMember fuel route member
  have outgoing : edge ∈ get_rule.outgoing edge.START lg.graph :=
    mem_outgoing_of_mem_edges_start_eq edge.START lg.graph hedge rfl
  have nonempty : (get_rule.outgoing edge.START lg.graph).isEmpty = false := by
    cases h : (get_rule.outgoing edge.START lg.graph).isEmpty with
    | false => rfl
    | true =>
      have empty := List.isEmpty_iff.mp h
      simp [empty] at outgoing
  unfold LambdaFlow.Def23DecoratedB LambdaFlow.decoratedFromB at labels
  have checked := List.all_eq_true.mp labels edge.START sourceMember
  simp only [nonempty, Bool.false_or, LambdaFlow.guardFreeTable,
    LambdaFlow.tableAt_eq_spec, LambdaFlow.fromTable_spec] at checked
  exact paperDecorationAtB_consumed lg edge.START _ checked edge outgoing route
    atStandard tail consumed

#print axioms paperDecorationAtB_consumed
#print axioms consumed_guard_at_every_fuel

end DependencyGuard
