import Semantic.MultiIntroIncludeTerminal
import Semantic.MultiIntroCertificate

open Semantic FlowSpec

namespace MultiIntroInclude

set_option maxHeartbeats 3000000

theorem root_membership (variant : MultiIntro.Variant) (d : Graph)
    (valid : SingleIntroductionReference.ReferenceVerdict d) (root : Vertex) (listed : root ∈ d.NODES)
    (terminal : get_rule.outgoing root d = []) (result : FlowPair) :
    result ∈ MultiIntro.evaluateAt variant d (stdFuel d) root ↔
      result ∈ SingleIntroductionReference.routes d (stdFuel d) root := by
  let fuel := stdFuel d - 1
  have enough : stdFuel d = fuel + 1 := by
    dsimp [fuel]
    unfold stdFuel
    omega
  have mine : MultiIntro.evaluateAt variant d (stdFuel d) root =
      MultiIntro.step variant d (SingleIntroductionReference.routes d fuel) root := by
    conv_lhs => rw [enough, MultiIntro.evaluateAt]
    apply MultiIntro.step_congr
    intro u predecessor
    exact nonterminal_routes_identity variant d valid.1 fuel u
      (mem_predsOf_mem_nodes valid.1 predecessor)
      (predecessor_nonterminal d root u predecessor)
  have reference : SingleIntroductionReference.routes d (stdFuel d) root =
      CompleteProfileKernel.tagged.step d (SingleIntroductionReference.routes d fuel) root := by
    conv_lhs => rw [enough]
    rfl
  rw [mine, reference]
  cases shape : MultiIntro.multiIntroShapeB d root with
  | false => simp [MultiIntro.step, shape]
  | true =>
    have live : CompleteProfileKernel.tagged.step d (SingleIntroductionReference.routes d fuel) root ≠ [] := by
      rw [← reference]
      exact ((SingleIntroductionReference.rootsB_iff d).mp valid.2.2 root listed
        (by simp [terminal])).1
    obtain ⟨u, predecessors⟩ := singleton_of_live_multi_terminal d root
      (SingleIntroductionReference.routes d fuel) shape terminal live
    exact terminal_step_membership variant d root u (SingleIntroductionReference.routes d fuel)
      shape terminal predecessors result


theorem reference_included (variant : MultiIntro.Variant) (d : Graph)
    (valid : SingleIntroductionReference.ReferenceVerdict d) : MultiIntro.ValidMI variant d := by
  refine ⟨domain_of_reference_domain d valid.1, valid.2.1,
    (MultiIntro.rootsB_iff variant d).mpr ?_⟩
  intro root listed terminal
  have terminalEq := List.isEmpty_iff.mp terminal
  have old := (SingleIntroductionReference.rootsB_iff d).mp valid.2.2 root listed terminal
  have membership := root_membership variant d valid root listed terminalEq
  constructor
  · obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ old.1
    intro empty
    have present := (membership route).mpr member
    simp [empty] at present
  · intro route member
    exact old.2 route ((membership route).mp member)

theorem flowRef_implies_general (d : Graph) :
    SingleIntroductionReference.ReferenceVerdict d → MultiIntro.ValidMI .general d :=
  reference_included .general d

end MultiIntroInclude
