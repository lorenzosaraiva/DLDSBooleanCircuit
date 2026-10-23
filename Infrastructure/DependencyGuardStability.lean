import Infrastructure.LambdaFlowCache

open Semantic FlowSpec SelfCertifying

namespace DependencyGuard

set_option maxHeartbeats 1500000
set_option maxRecDepth 100000

private theorem bind_subset {α β : Type} {xs ys : List α}
    {f g : α → List β} (inputs : xs ⊆ ys)
    (outputs : ∀ x ∈ xs, f x ⊆ g x) : xs.flatMap f ⊆ ys.flatMap g := by
  intro result member
  obtain ⟨x, hx, produced⟩ := List.mem_flatMap.mp member
  exact List.mem_flatMap.mpr ⟨x, inputs hx, outputs x hx produced⟩

private theorem append_subset {α : Type} {a b c e : List α}
    (left : a ⊆ c) (right : b ⊆ e) : a ++ b ⊆ c ++ e := by
  intro x member
  rcases List.mem_append.mp member with h | h
  · exact List.mem_append_left _ (left h)
  · exact List.mem_append_right _ (right h)

private theorem reseed_subset (d : Graph) (guard : LambdaFlow.Guard)
    (left right : Vertex → List FlowPair) (mono : ∀ u, left u ⊆ right u) (v : Vertex) :
    LambdaFlow.reseedWith d guard v left ⊆ LambdaFlow.reseedWith d guard v right := by
  unfold LambdaFlow.reseedWith
  dsimp only
  split
  · exact List.Subset.refl _
  · intro route member
    rw [List.mem_eraseDups] at member ⊢
    apply append_subset (a := LambdaFlow.elimReseedWith d guard v left _)
      (b := LambdaFlow.introReseedWith d guard v left _) _ _ member
    · unfold LambdaFlow.elimReseedWith
      apply bind_subset (List.Subset.refl _)
      intro pair hpair
      apply bind_subset (mono pair.2)
      intro minor hminor
      apply bind_subset (mono pair.1)
      intro major hmajor
      exact List.Subset.refl _
    · unfold LambdaFlow.introReseedWith
      split
      · split
        · apply bind_subset (mono _)
          intro route hroute
          exact List.Subset.refl _
        · exact List.Subset.refl _
      · exact List.Subset.refl _

private theorem mixed_subset (d : Graph) (guard : LambdaFlow.Guard)
    (left right : Vertex → List FlowPair) (mono : ∀ u, left u ⊆ right u) (v : Vertex) :
    LambdaFlow.mixedWith d guard v left ⊆ LambdaFlow.mixedWith d guard v right := by
  unfold LambdaFlow.mixedWith
  split
  · apply bind_subset (List.Subset.refl _)
    intro u hu
    split
    · apply bind_subset (mono u)
      intro route hroute
      exact List.Subset.refl _
    · exact List.Subset.refl _
  · exact List.Subset.refl _


theorem stepWith_mono (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (left right : Vertex → List FlowPair) (mono : ∀ u, left u ⊆ right u)
    (v : Vertex) :
    LambdaFlow.stepWith d ordinary reseed left v ⊆
      LambdaFlow.stepWith d ordinary reseed right v := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp [LambdaFlow.stepWith, source]
  · simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false]
    intro route member
    rw [List.mem_eraseDups] at member ⊢
    apply append_subset _ (reseed_subset d reseed left right mono v) member
    by_cases pairs : (elimPairsAt d v).isEmpty = true
    · simp only [pairs, if_true]
      split
      · split
        · apply bind_subset (mono _)
          intro route hroute
          exact List.Subset.refl _
        · exact List.Subset.refl _
      · exact List.Subset.refl _
    · simp only [pairs, Bool.false_eq_true, if_false]
      apply append_subset _ (mixed_subset d ordinary left right mono v)
      apply bind_subset (List.Subset.refl _)
      intro pair hpair
      apply bind_subset (mono pair.2)
      intro minor hminor
      apply bind_subset (mono pair.1)
      intro major hmajor
      exact List.Subset.refl _

theorem stepWith_mono_on_preds (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (left right : Vertex → List FlowPair) (v : Vertex)
    (mono : ∀ u ∈ predsOf d v, left u ⊆ right u) :
    LambdaFlow.stepWith d ordinary reseed left v ⊆
      LambdaFlow.stepWith d ordinary reseed right v := by
  let patched : Vertex → List FlowPair := fun u => if u ∈ predsOf d v then right u else left u
  have localMono : ∀ u, left u ⊆ patched u := by
    intro u
    by_cases hu : u ∈ predsOf d v
    · simpa only [patched, hu, if_true] using mono u hu
    · simp only [patched, hu, if_false]
      exact List.Subset.refl _
  have eq := LambdaFlow.stepWith_congr_of_preds d ordinary reseed v patched right
    (fun u hu => by simp only [patched, hu, if_true])
  rw [← eq]
  exact stepWith_mono d ordinary reseed left patched localMono v

theorem stepWith_setEq (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (left right : Vertex → List FlowPair) (equal : ∀ u, ListSetEq (left u) (right u))
    (v : Vertex) :
    ListSetEq (LambdaFlow.stepWith d ordinary reseed left v)
      (LambdaFlow.stepWith d ordinary reseed right v) := by
  intro route
  exact ⟨fun h => stepWith_mono d ordinary reseed left right (fun u r => (equal u r).mp) v h,
    fun h => stepWith_mono d ordinary reseed right left (fun u r => (equal u r).mpr) v h⟩

theorem stepWith_setEq_on_preds (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (left right : Vertex → List FlowPair) (v : Vertex)
    (equal : ∀ u ∈ predsOf d v, ListSetEq (left u) (right u)) :
    ListSetEq (LambdaFlow.stepWith d ordinary reseed left v)
      (LambdaFlow.stepWith d ordinary reseed right v) := by
  let patched : Vertex → List FlowPair := fun u => if u ∈ predsOf d v then right u else left u
  have localEq : ∀ u, ListSetEq (left u) (patched u) := by
    intro u route
    by_cases hu : u ∈ predsOf d v
    · simpa only [patched, hu, if_true] using equal u hu route
    · simp only [patched, hu, if_false]
  have eq := LambdaFlow.stepWith_congr_of_preds d ordinary reseed v patched right
    (fun u hu => by simp only [patched, hu, if_true])
  rw [← eq]
  exact stepWith_setEq d ordinary reseed left patched localEq v

theorem evaluateAt_mono_fuel (d : Graph) (ordinary reseed : LambdaFlow.Guard) :
    ∀ fuel extra v, LambdaFlow.evaluateAt d ordinary reseed fuel v ⊆
      LambdaFlow.evaluateAt d ordinary reseed (fuel + extra) v := by
  have step : ∀ fuel v, LambdaFlow.evaluateAt d ordinary reseed fuel v ⊆
      LambdaFlow.evaluateAt d ordinary reseed (fuel + 1) v := by
    intro fuel
    induction fuel with
    | zero => intro v; simp [LambdaFlow.evaluateAt]
    | succ fuel ih =>
        intro v
        exact stepWith_mono d ordinary reseed _ _ ih v
  intro fuel extra
  induction extra with
  | zero => intro v; exact List.Subset.refl _
  | succ extra ih =>
      intro v route member
      exact step (fuel + extra) v (ih v member)



theorem evaluateAt_succ_stable (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (structural : structuralValid d) :
    ∀ v ∈ d.NODES, ∀ fuel, countAbove d v < fuel →
      LambdaFlow.evaluateAt d ordinary reseed (fuel + 1) v =
        LambdaFlow.evaluateAt d ordinary reseed fuel v := by
  have key : ∀ n, ∀ v ∈ d.NODES, countAbove d v = n →
      ∀ fuel, countAbove d v < fuel →
      LambdaFlow.evaluateAt d ordinary reseed (fuel + 1) v =
        LambdaFlow.evaluateAt d ordinary reseed fuel v := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro v hv hn fuel hf
      have hf1 : fuel = (fuel - 1) + 1 := by omega
      rw [LambdaFlow.evaluateAt]
      conv_rhs => rw [hf1, LambdaFlow.evaluateAt]
      apply LambdaFlow.stepWith_congr_of_preds
      intro u hu
      have humem := mem_predsOf_mem_nodes structural hu
      have hulvl := (predsOf_props structural hu).1
      have huca : countAbove d u < countAbove d v :=
        countAbove_lt_of_level_succ d humem hulvl
      have stable := ih (countAbove d u) (by omega) u humem rfl
        (fuel - 1) (by omega)
      rw [hf1]
      exact stable
  intro v hv fuel hf
  exact key (countAbove d v) v hv rfl fuel hf

theorem evaluateAt_stdFuel_pred (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (structural : structuralValid d) {v : Vertex} (member : v ∈ d.NODES) :
    LambdaFlow.evaluateAt d ordinary reseed (stdFuel d) v =
      LambdaFlow.evaluateAt d ordinary reseed (stdFuel d - 1) v := by
  have bound : countAbove d v < stdFuel d - 1 := by
    have count := countAbove_lt_length d member
    unfold stdFuel
    omega
  have stable := evaluateAt_succ_stable d ordinary reseed structural v member
    (stdFuel d - 1) bound
  have positive : stdFuel d - 1 + 1 = stdFuel d := by unfold stdFuel; omega
  simpa only [positive] using stable

theorem evaluateAt_member_at_standard (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (structural : structuralValid d) {v : Vertex} (node : v ∈ d.NODES)
    (fuel : Nat) (route : FlowPair)
    (member : route ∈ LambdaFlow.evaluateAt d ordinary reseed fuel v) :
    route ∈ LambdaFlow.evaluateAt d ordinary reseed (stdFuel d) v := by
  by_cases earlier : fuel ≤ stdFuel d
  · have later := evaluateAt_mono_fuel d ordinary reseed fuel (stdFuel d - fuel) v member
    simpa [Nat.add_sub_of_le earlier] using later
  · have stable : ∀ extra, LambdaFlow.evaluateAt d ordinary reseed (stdFuel d + extra) v =
        LambdaFlow.evaluateAt d ordinary reseed (stdFuel d) v := by
      intro extra
      induction extra with
      | zero => rfl
      | succ extra ih =>
          rw [show stdFuel d + (extra + 1) = (stdFuel d + extra) + 1 by omega,
            evaluateAt_succ_stable d ordinary reseed structural v node, ih]
          have count := countAbove_le d v
          unfold stdFuel
          omega
    have after : stdFuel d ≤ fuel := by omega
    have eq := stable (fuel - stdFuel d)
    rw [Nat.add_sub_of_le after] at eq
    simpa [eq] using member

#print axioms stepWith_mono
#print axioms stepWith_mono_on_preds
#print axioms stepWith_setEq
#print axioms stepWith_setEq_on_preds
#print axioms evaluateAt_mono_fuel
#print axioms evaluateAt_succ_stable
#print axioms evaluateAt_stdFuel_pred
#print axioms evaluateAt_member_at_standard

end DependencyGuard
