import Semantic.LambdaFlowModel
import Semantic.DecorationFlowMonotonicity
import Semantic.RootDischarge

open Semantic FlowSpec SelfCertifying

namespace LambdaFlow

set_option linter.unusedSimpArgs false

private theorem flat_subset {α β : Type} {xs : List α} {f g : α → List β}
    (sub : ∀ x ∈ xs, f x ⊆ g x) : xs.flatMap f ⊆ xs.flatMap g := by
  intro y member
  obtain ⟨x, hx, hy⟩ := List.mem_flatMap.mp member
  exact List.mem_flatMap.mpr ⟨x, hx, sub x hx hy⟩

private theorem reseed_guard_removal (d : Graph) (guard : Guard) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    reseedWith d guard v prior ⊆ reseedWith d allowAll v prior := by
  unfold reseedWith
  dsimp only
  split
  · exact List.Subset.refl _
  · intro route member
    rw [List.mem_eraseDups, List.mem_append] at member ⊢
    rcases member with elim | introRoute
    · left
      have sub : elimReseedWith d guard v prior (eligibleAncestorsAtRoot d v) ⊆
          elimReseedWith d allowAll v prior (eligibleAncestorsAtRoot d v) := by
        unfold elimReseedWith
        apply flat_subset
        intro pair hpair
        apply flat_subset
        intro minor hminor
        apply flat_subset
        intro major hmajor
        apply flat_subset
        intro eMinor heMinor
        apply flat_subset
        intro eMajor heMajor
        cases h1 : reseedPremiseAgreesB minor eMinor <;>
          cases h2 : reseedPremiseAgreesB major eMajor <;>
          cases h3 : guard minor.1 eMinor <;> cases h4 : guard major.1 eMajor <;>
            simp [allowAll, h1, h2, h3, h4]
      exact sub elim
    · right
      have sub : introReseedWith d guard v prior (eligibleAncestorsAtRoot d v) ⊆
          introReseedWith d allowAll v prior (eligibleAncestorsAtRoot d v) := by
        unfold introReseedWith
        split
        · split
          · apply flat_subset
            intro route hroute
            apply flat_subset
            intro edge hedge
            cases h1 : reseedZeroPremiseB route edge <;>
              cases h2 : guard route.1 edge <;> simp [allowAll, h1, h2]
          · exact List.Subset.refl _
        · exact List.Subset.refl _
      exact sub introRoute

theorem step_reseed_guard_removal (d : Graph) (guard : Guard)
    (prior : Vertex → List FlowPair) (v : Vertex) :
    stepWith d allowAll guard prior v ⊆ stepWith d allowAll allowAll prior v := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp [stepWith, source]
  · simp only [stepWith, source, Bool.false_eq_true, if_false]
    intro route member
    rw [List.mem_eraseDups, List.mem_append] at member ⊢
    rcases member with core | reseed
    · exact Or.inl core
    · exact Or.inr (reseed_guard_removal d guard v prior reseed)



theorem historical_subset_guardFree (d : Graph) : ∀ fuel v,
    flowAt d fuel v ⊆ guardFreeFlowAt d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; simp [flowAt, guardFreeFlowAt, evaluateAt]
  | succ fuel ih =>
      intro v route member
      have moved := Decoration.flowStepFrom_mono d (flowAt d fuel)
        (guardFreeFlowAt d fuel) ih v route member
      rw [← historical_step_eq] at moved
      exact step_reseed_guard_removal d historicalGuard (guardFreeFlowAt d fuel) v moved

theorem strong_formation_guardFree_roots_nonempty (d : Graph) (c : Certificate)
    (strong : StrongFormationValid d c) (root : Vertex) (member : root ∈ d.NODES)
    (terminal : get_rule.outgoing root d = []) :
    guardFreeFlowAt d (stdFuel d) root ≠ [] := by
  have old := strongFormationValid_rootsDischargedB d c strong
  have atRoot := List.all_eq_true.mp old root member
  simp only [terminal, List.isEmpty_nil, Bool.not_true, Bool.false_or,
    Bool.and_eq_true] at atRoot
  have nonempty : flowAt d (stdFuel d) root ≠ [] := by
    intro empty
    simp [empty] at atRoot
  obtain ⟨route, hroute⟩ := List.exists_mem_of_ne_nil _ nonempty
  have kept := historical_subset_guardFree d (stdFuel d) root hroute
  intro empty
  simp [empty] at kept

#print axioms step_reseed_guard_removal
#print axioms historical_subset_guardFree
#print axioms strong_formation_guardFree_roots_nonempty

end LambdaFlow
