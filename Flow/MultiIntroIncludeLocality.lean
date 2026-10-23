import Flow.MultiIntroIdentity

open Semantic FlowSpec

namespace MultiIntroInclude

set_option maxHeartbeats 3000000

theorem shape_facts (d : Graph) (v : Vertex)
    (shape : MultiIntro.multiIntroShapeB d v = true) :
    ∃ alpha beta, v.FORMULA = .implication alpha beta ∧ v.HYPOTHESIS = false ∧
      elimPairsAt d v = [] ∧ 2 ≤ (get_rule.incoming v d).length ∧
      ∀ e ∈ get_rule.incoming v d, e.START.FORMULA = beta := by
  cases formula : v.FORMULA with
  | atom symbol => simp [MultiIntro.multiIntroShapeB, formula] at shape
  | implication alpha beta =>
    have parts : v.HYPOTHESIS = false ∧
        (((elimPairsAt d v).isEmpty = true ∧ 2 ≤ (get_rule.incoming v d).length) ∧
          (get_rule.incoming v d).all (fun e => e.START.FORMULA == beta) = true) := by
      simpa only [MultiIntro.multiIntroShapeB, formula, Bool.and_eq_true,
        Bool.not_eq_true', decide_eq_true_eq] using shape
    exact ⟨alpha, beta, rfl, parts.1, List.isEmpty_iff.mp parts.2.1.1,
      parts.2.1.2, fun e member => beq_iff_eq.mp (List.all_eq_true.mp parts.2.2 e member)⟩

theorem no_multi_at_nonterminal (d : Graph) (structural : structuralValid d)
    (v : Vertex) (listed : v ∈ d.NODES) (outgoing : get_rule.outgoing v d ≠ []) :
    MultiIntro.multiIntroShapeB d v = false := by
  cases shape : MultiIntro.multiIntroShapeB d v with
  | false => rfl
  | true =>
    obtain ⟨alpha, beta, formula, hypothesis, pairs, length, _⟩ := shape_facts d v shape
    obtain elimination | introduction := structural.2.2 v listed
      (MultiIntro.shape_has_incoming d v shape) outgoing
    · exact False.elim (elimination.2.2 pairs)
    · obtain ⟨e, a, b, _, incoming, _, _, _⟩ := introduction
      simp [incoming] at length

theorem domain_of_reference_domain (d : Graph) (structural : structuralValid d) :
    MultiIntro.MIDomain d := by
  have old := CompleteProfilePositions.structuralValidB_complete d structural
  simp only [structuralValidB, Bool.and_eq_true, List.all_eq_true] at old
  simp only [MultiIntro.MIDomain, MultiIntro.miDomainB, Bool.and_eq_true,
    List.all_eq_true]
  refine ⟨⟨old.1.1, old.1.2⟩, ?_⟩
  intro v listed
  have checked := old.2 v listed
  split <;> simp_all [Bool.or_eq_true]

theorem predecessor_nonterminal (d : Graph) (v u : Vertex)
    (member : u ∈ predsOf d v) : get_rule.outgoing u d ≠ [] := by
  obtain ⟨e, incoming, start⟩ := predsOf_has_edge member
  have actual := mem_incoming_mem_edges v d incoming
  have outgoing := mem_outgoing_of_mem_edges_start_eq u d actual start
  intro empty
  simp [empty] at outgoing

theorem nonterminal_routes_identity (variant : MultiIntro.Variant) (d : Graph)
    (structural : structuralValid d) : ∀ fuel v, v ∈ d.NODES →
      get_rule.outgoing v d ≠ [] →
      MultiIntro.evaluateAt variant d fuel v = SingleIntroductionReference.routes d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v _ _; rfl
  | succ fuel ih =>
    intro v listed outgoing
    simp only [MultiIntro.evaluateAt, MultiIntro.step,
      no_multi_at_nonterminal d structural v listed outgoing, Bool.false_eq_true, if_false]
    change CompleteProfileKernel.tagged.step d (MultiIntro.evaluateAt variant d fuel) v =
      CompleteProfileKernel.tagged.step d (SingleIntroductionReference.routes d fuel) v
    apply CompleteProfileKernel.tagged.congr
    intro u predecessor
    exact ih u (mem_predsOf_mem_nodes structural predecessor)
      (predecessor_nonterminal d v u predecessor)

end MultiIntroInclude
