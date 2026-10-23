import Flow.MultiIntroIncludeLocality

open Semantic FlowSpec

namespace MultiIntroInclude

set_option maxHeartbeats 3000000

theorem terminal_eligible_nil (d : Graph) (v : Vertex)
    (terminal : get_rule.outgoing v d = []) : eligibleAncestorsAtRoot d v = [] := by
  cases unique : uniqueRoot? d with
  | none => simp [eligibleAncestorsAtRoot, unique]
  | some root =>
    have inactive : ∀ ancestor, rootSeedEligibleWithPreB d root (Pre d root) v ancestor = false := by
      intro ancestor
      cases colours : ancestor.COLOURS <;>
        simp [rootSeedEligibleWithPreB, colours, terminal]
    simp [eligibleAncestorsAtRoot, unique, inactive]

theorem terminal_reseed_nil (variant : MultiIntro.Variant) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (edge : Deduction)
    (terminal : get_rule.outgoing v d = []) :
    MultiIntro.reseedEdge variant d v prior edge = [] := by
  cases formula : v.FORMULA <;>
    simp [MultiIntro.reseedEdge, formula, terminal_eligible_nil d v terminal]

theorem start_of_single_predecessor (d : Graph) (v u : Vertex)
    (predecessors : predsOf d v = [u]) (edge : Deduction)
    (incoming : edge ∈ get_rule.incoming v d) : edge.START = u := by
  simpa only [predecessors, List.mem_singleton] using
    MultiIntro.incoming_start_pred d v edge incoming

theorem incoming_iff_between (d : Graph) (v u : Vertex)
    (predecessors : predsOf d v = [u]) (edge : Deduction) :
    edge ∈ get_rule.incoming v d ↔ edge ∈ edgesBetween d u v := by
  constructor
  · intro member
    exact mem_edgesBetween_of (mem_incoming_mem_edges v d member)
      (start_of_single_predecessor d v u predecessors edge member)
      (mem_incoming_end_eq v d member)
  · intro member
    have parts : edge ∈ d.EDGES ∧ edge.START = u ∧ edge.END = v := by
      simpa only [edgesBetween, List.mem_filter, decide_eq_true_eq] using member
    rw [incoming_eq_filter, List.mem_filter]
    exact ⟨parts.1, by simpa using parts.2.2⟩

theorem singleton_of_live_multi_terminal (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair)
    (shape : MultiIntro.multiIntroShapeB d v = true)
    (terminal : get_rule.outgoing v d = [])
    (live : CompleteProfileKernel.tagged.step d prior v ≠ []) :
    ∃ u, predsOf d v = [u] := by
  obtain ⟨alpha, beta, formula, hypothesis, pairs, _, _⟩ := shape_facts d v shape
  obtain ⟨edge, incoming⟩ :=
    List.exists_mem_of_ne_nil _ (MultiIntro.shape_has_incoming d v shape)
  have predecessor := MultiIntro.incoming_start_pred d v edge incoming
  cases predecessors : predsOf d v with
  | nil => simp [predecessors] at predecessor
  | cons u rest =>
    cases rest with
    | nil => exact ⟨u, rfl⟩
    | cons w rest =>
      simp [CompleteProfileKernel.tagged, LambdaFlow.stepWith, predecessors, hypothesis,
        formula, pairs, LambdaFlow.reseedWith, terminal_eligible_nil d v terminal] at live


theorem terminal_step_membership (variant : MultiIntro.Variant) (d : Graph) (v u : Vertex)
    (prior : Vertex → List FlowPair)
    (shape : MultiIntro.multiIntroShapeB d v = true)
    (terminal : get_rule.outgoing v d = []) (predecessors : predsOf d v = [u])
    (result : FlowPair) :
    result ∈ MultiIntro.step variant d prior v ↔
      result ∈ CompleteProfileKernel.tagged.step d prior v := by
  obtain ⟨alpha, beta, formula, hypothesis, pairs, _, premiseFormulas⟩ := shape_facts d v shape
  obtain ⟨edge, incoming⟩ :=
    List.exists_mem_of_ne_nil _ (MultiIntro.shape_has_incoming d v shape)
  have betaFormula : u.FORMULA = beta := by
    simpa only [start_of_single_predecessor d v u predecessors edge incoming] using
      premiseFormulas edge incoming
  have mine : MultiIntro.step variant d prior v =
      ((get_rule.incoming v d).flatMap fun e =>
        (prior u).filterMap fun route =>
          if SingleIntroductionReference.guard d route.1 e then
            (consume e route.2).map fun tail => (depRemove route.1 alpha, tail)
          else none).eraseDups := by
    rw [MultiIntro.step, if_pos shape]
    apply congrArg List.eraseDups
    apply List.flatMap_congr
    intro e member
    simp [MultiIntro.ordinaryEdge, formula,
      start_of_single_predecessor d v u predecessors e member, betaFormula,
      terminal_reseed_nil variant d v prior e terminal]
  have reference : CompleteProfileKernel.tagged.step d prior v =
      ((prior u).flatMap fun route =>
        (edgesBetween d u v).filterMap fun e =>
          if SingleIntroductionReference.guard d route.1 e then
            (consume e route.2).map fun tail => (depRemove route.1 alpha, tail)
          else none).eraseDups := by
    simp [CompleteProfileKernel.tagged, LambdaFlow.stepWith, predecessors, hypothesis,
      formula, pairs, betaFormula, LambdaFlow.reseedWith, terminal_eligible_nil d v terminal]
  rw [mine, reference]
  simp only [List.mem_eraseDups, List.mem_flatMap, List.mem_filterMap]
  constructor
  · rintro ⟨e, member, route, routeMember, produced⟩
    exact ⟨route, routeMember, e, (incoming_iff_between d v u predecessors e).mp member, produced⟩
  · rintro ⟨route, routeMember, e, member, produced⟩
    exact ⟨e, (incoming_iff_between d v u predecessors e).mpr member, route, routeMember, produced⟩

end MultiIntroInclude
