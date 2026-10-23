import Semantic.DecorationGuardSoundness

open Semantic FlowSpec SelfCertifyingProbe SelfCertifying

namespace Decoration

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

private theorem reseed_subset (d : Graph) (left right : Vertex → List FlowPair)
    (mono : ∀ u, left u ⊆ right u) (v : Vertex) :
    reseedPairs d v left ⊆ reseedPairs d v right := by
  unfold reseedPairs
  dsimp only
  split
  · exact List.Subset.refl _
  · intro route member
    rw [List.mem_eraseDups] at member ⊢
    apply append_subset (a := elimReseedPairs d v left _) (b := introReseedPairs d v left _) _ _ member
    · unfold elimReseedPairs
      apply bind_subset (List.Subset.refl _)
      intro pair hpair
      apply bind_subset (mono pair.2)
      intro minor hminor
      apply bind_subset (mono pair.1)
      intro major hmajor
      exact List.Subset.refl _
    · unfold introReseedPairs
      split
      · split
        · apply bind_subset (mono _)
          intro route hroute
          exact List.Subset.refl _
        · exact List.Subset.refl _
      · exact List.Subset.refl _

private theorem mixed_subset (d : Graph) (left right : Vertex → List FlowPair)
    (mono : ∀ u, left u ⊆ right u) (v : Vertex) :
    mixedIntroPairs d v left ⊆ mixedIntroPairs d v right := by
  unfold mixedIntroPairs
  split
  · apply bind_subset (List.Subset.refl _)
    intro u hu
    split
    · apply bind_subset (mono u)
      intro route hroute
      exact List.Subset.refl _
    · exact List.Subset.refl _
  · exact List.Subset.refl _



theorem flowStepFrom_mono (d : Graph) (left right : Vertex → List FlowPair)
    (mono : ∀ u route, route ∈ left u → route ∈ right u)
    (v : Vertex) (route : FlowPair)
    (member : route ∈ flowStepFrom d left v) :
    route ∈ flowStepFrom d right v := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simpa [flowStepFrom, source] using member
  · simp only [flowStepFrom, source, Bool.false_eq_true, if_false] at member ⊢
    rw [List.mem_eraseDups] at member ⊢
    apply append_subset _ (reseed_subset d left right mono v) member
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
      apply append_subset _ (mixed_subset d left right mono v)
      apply bind_subset (List.Subset.refl _)
      intro pair hpair
      apply bind_subset (mono pair.2)
      intro minor hminor
      apply bind_subset (mono pair.1)
      intro major hmajor
      exact List.Subset.refl _

theorem flowAt_mono_fuel (d : Graph) : ∀ fuel extra v route,
    route ∈ flowAt d fuel v → route ∈ flowAt d (fuel + extra) v := by
  have step : ∀ fuel v route,
      route ∈ flowAt d fuel v → route ∈ flowAt d (fuel + 1) v := by
    intro fuel
    induction fuel with
    | zero => simp [flowAt]
    | succ fuel ih =>
        intro v route member
        exact flowStepFrom_mono d (flowAt d fuel) (flowAt d (fuel + 1))
          ih v route member
  intro fuel extra
  induction extra with
  | zero => simp
  | succ extra ih =>
      intro v route member
      exact step (fuel + extra) v route (ih v route member)



theorem raw_edge_source_at_standard (d : Graph) (structural : structuralValid d)
    (edge : Deduction) (hedge : edge ∈ d.EDGES) (fuel : Nat)
    (route : FlowPair) (member : route ∈ flowAt d fuel edge.START) :
    route ∈ flowAt d (stdFuel d) edge.START := by
  have sourceMember := (structural.2.1 edge hedge).1
  have outgoingMember : edge ∈ get_rule.outgoing edge.START d := by
    exact mem_outgoing_of_mem_edges_start_eq edge.START d hedge rfl
  have nonterminal : get_rule.outgoing edge.START d ≠ [] := by
    intro empty
    simpa [empty] using outgoingMember
  by_cases earlier : fuel ≤ stdFuel d
  · have later := flowAt_mono_fuel d fuel (stdFuel d - fuel)
      edge.START route member
    simpa [Nat.add_sub_of_le earlier] using later
  · have stable : ∀ extra, flowAt d (stdFuel d + extra) edge.START =
        flowAt d (stdFuel d) edge.START := by
      intro extra
      induction extra with
      | zero => rfl
      | succ extra ih =>
          rw [show stdFuel d + (extra + 1) = (stdFuel d + extra) + 1 by omega,
            flowAt_succ_stable d structural edge.START sourceMember nonterminal,
            ih]
          have count := countAbove_le d edge.START
          unfold stdFuel
          omega
    have after : stdFuel d ≤ fuel := by omega
    have eq := stable (fuel - stdFuel d)
    rw [Nat.add_sub_of_le after] at eq
    simpa [eq] using member

#print axioms flowStepFrom_mono
#print axioms flowAt_mono_fuel
#print axioms raw_edge_source_at_standard

end Decoration
