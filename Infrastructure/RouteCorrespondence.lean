import Infrastructure.SingleIntroductionReference
import Infrastructure.LambdaGuardRemoval
import Infrastructure.DependencyGuardStability

open Semantic FlowSpec SelfCertifying
namespace SingleIntroductionReference
open LambdaFlow
set_option maxHeartbeats 2000000

def GuardLE (left right : Guard) : Prop :=
  ∀ dependency edge, left dependency edge = true → right dependency edge = true

private theorem flat_subset {α β : Type} {xs : List α} {f g : α → List β}
    (sub : ∀ x ∈ xs, f x ⊆ g x) : xs.flatMap f ⊆ xs.flatMap g := by
  intro result member
  obtain ⟨x, hx, produced⟩ := List.mem_flatMap.mp member
  exact List.mem_flatMap.mpr ⟨x, hx, sub x hx produced⟩

private theorem filter_subset {α β : Type} {xs : List α} {f g : α → Option β}
    (sub : ∀ x ∈ xs, ∀ y, f x = some y → g x = some y) :
    xs.filterMap f ⊆ xs.filterMap g := by
  intro result member
  obtain ⟨x, hx, produced⟩ := List.mem_filterMap.mp member
  exact List.mem_filterMap.mpr ⟨x, hx, sub x hx result produced⟩

private theorem option_guard {α : Type} (left right : Bool)
    (imp : left = true → right = true) (value : Option α) (result : α)
    (member : (if left then value else none) = some result) :
    (if right then value else none) = some result := by
  cases hl : left with
  | false => simp [hl] at member
  | true => simpa [hl, imp hl] using member

private theorem list_guard {α : Type} (left right : Bool)
    (imp : left = true → right = true) (value : List α) :
    (if left then value else []) ⊆ (if right then value else []) := by
  cases hl : left with
  | false => simp []
  | true => simp [imp hl]

private theorem append_subset {α : Type} {a b c e : List α}
    (left : a ⊆ c) (right : b ⊆ e) : a ++ b ⊆ c ++ e := by
  intro x member
  rcases List.mem_append.mp member with h | h
  · exact List.mem_append_left _ (left h)
  · exact List.mem_append_right _ (right h)

private theorem mixed_guards (d : Graph) (left right : Guard) (imp : GuardLE left right)
    (v : Vertex) (prior : Vertex → List FlowPair) :
    mixedWith d left v prior ⊆ mixedWith d right v prior := by
  unfold mixedWith
  split
  · apply flat_subset
    intro u hu
    split
    · apply flat_subset
      intro route member
      apply filter_subset
      intro edge he result produced
      exact option_guard _ _ (imp route.1 edge) _ _ produced
    · exact List.Subset.refl _
  · exact List.Subset.refl _

private theorem reseed_guards (d : Graph) (left right : Guard) (imp : GuardLE left right)
    (v : Vertex) (prior : Vertex → List FlowPair) :
    reseedWith d left v prior ⊆ reseedWith d right v prior := by
  unfold reseedWith
  dsimp only
  split
  · exact List.Subset.refl _
  · intro route member
    rw [List.mem_eraseDups, List.mem_append] at member ⊢
    rcases member with eliminated | introduced
    · left
      apply (show elimReseedWith d left v prior _ ⊆ elimReseedWith d right v prior _ from ?_) eliminated
      unfold elimReseedWith
      apply flat_subset
      intro pair hp
      apply flat_subset
      intro minor hm
      apply flat_subset
      intro major hj
      apply flat_subset
      intro em hem
      apply flat_subset
      intro ej hej
      apply list_guard
      intro passed
      simp only [Bool.and_eq_true] at passed ⊢
      exact ⟨⟨passed.1.1, imp _ _ passed.1.2⟩, imp _ _ passed.2⟩
    · right
      apply (show introReseedWith d left v prior _ ⊆ introReseedWith d right v prior _ from ?_) introduced
      unfold introReseedWith
      split
      · split
        · apply flat_subset
          intro premise hm
          apply flat_subset
          intro edge he
          apply list_guard
          intro passed
          simp only [Bool.and_eq_true] at passed ⊢
          exact ⟨passed.1, imp _ _ passed.2⟩
        · exact List.Subset.refl _
      · exact List.Subset.refl _

theorem stepWith_guard_mono (d : Graph) (o₁ o₂ r₁ r₂ : Guard)
    (ordinary : GuardLE o₁ o₂) (reseed : GuardLE r₁ r₂)
    (prior : Vertex → List FlowPair) (v : Vertex) :
    stepWith d o₁ r₁ prior v ⊆ stepWith d o₂ r₂ prior v := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp [stepWith, source]
  · simp only [stepWith, source, Bool.false_eq_true, if_false]
    intro route member
    rw [List.mem_eraseDups] at member ⊢
    apply append_subset _ (reseed_guards d r₁ r₂ reseed v prior) member
    by_cases pairs : (elimPairsAt d v).isEmpty = true
    · simp only [pairs, if_true]
      split
      · split
        · apply flat_subset
          intro premise hm
          apply filter_subset
          intro edge he result produced
          exact option_guard _ _ (ordinary premise.1 edge) _ _ produced
        · exact List.Subset.refl _
      · exact List.Subset.refl _
    · simp only [pairs, Bool.false_eq_true, if_false]
      apply append_subset _ (mixed_guards d o₁ o₂ ordinary v prior)
      apply flat_subset
      intro pair hp
      apply flat_subset
      intro minor hm
      apply flat_subset
      intro major hj
      apply flat_subset
      intro em hem
      apply filter_subset
      intro ej hej result produced
      apply option_guard _ _ ?_ _ _ produced
      intro passed
      simp only [Bool.and_eq_true] at passed ⊢
      exact ⟨ordinary _ _ passed.1, ordinary _ _ passed.2⟩

theorem evaluateAt_guard_mono (d : Graph) (o₁ o₂ r₁ r₂ : Guard)
    (ordinary : GuardLE o₁ o₂) (reseed : GuardLE r₁ r₂) : ∀ fuel v,
    evaluateAt d o₁ r₁ fuel v ⊆ evaluateAt d o₂ r₂ fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; exact List.Subset.refl _
  | succ fuel ih =>
      intro v route member
      have moved := DependencyGuard.stepWith_mono d o₁ r₁ _ _ ih v member
      exact stepWith_guard_mono d o₁ o₂ r₁ r₂ ordinary reseed _ v moved

theorem routes_subset_historical (d : Graph) (fuel : Nat) (v : Vertex) :
    routes d fuel v ⊆ flowAt d fuel v := by
  rw [← historical_evaluate_eq]
  exact evaluateAt_guard_mono d (guard d) allowAll (guard d) historicalGuard
    (fun _ _ _ => rfl) (guard_implies_historical d) fuel v

theorem routes_subset_guardFree (d : Graph) (fuel : Nat) (v : Vertex) :
    routes d fuel v ⊆ guardFreeFlowAt d fuel v :=
  fun _ member => historical_subset_guardFree d fuel v (routes_subset_historical d fuel v member)

#print axioms stepWith_guard_mono
#print axioms evaluateAt_guard_mono
#print axioms routes_subset_historical
#print axioms routes_subset_guardFree
end SingleIntroductionReference
