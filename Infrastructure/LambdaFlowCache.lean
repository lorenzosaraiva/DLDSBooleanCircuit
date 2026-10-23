import Infrastructure.LambdaFlowModel

open Semantic FlowSpec CertificateModel

namespace LambdaFlow

set_option maxHeartbeats 1000000
set_option maxRecDepth 100000

theorem historical_evaluate_eq (d : Graph) : ∀ fuel v,
    evaluateAt d allowAll historicalGuard fuel v = flowAt d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
      intro v
      simp only [evaluateAt]
      rw [show evaluateAt d allowAll historicalGuard fuel = flowAt d fuel from funext ih]
      exact historical_step_eq d _ v

private theorem mixed_congr (d : Graph) (guard : Guard) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (agree : ∀ u ∈ predsOf d v, left u = right u) :
    mixedWith d guard v left = mixedWith d guard v right := by
  unfold mixedWith
  split
  · apply List.flatMap_congr
    intro u hu
    rw [agree u hu]
  · rfl

private theorem reseed_congr (d : Graph) (guard : Guard) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (agree : ∀ u ∈ predsOf d v, left u = right u) :
    reseedWith d guard v left = reseedWith d guard v right := by
  unfold reseedWith
  dsimp only
  split
  · rfl
  · apply congrArg List.eraseDups
    apply congrArg₂ (· ++ ·)
    · unfold elimReseedWith
      apply List.flatMap_congr
      intro pair hpair
      rcases pair with ⟨major, minor⟩
      obtain ⟨hmajor, hminor⟩ := mem_preds_of_mem_elimPairsAt hpair
      dsimp only
      rw [agree minor hminor, agree major hmajor]
    · unfold introReseedWith
      cases hp : predsOf d v with
      | nil => rfl
      | cons u rest =>
          cases rest with
          | nil =>
              have hu : u ∈ predsOf d v := by simp [hp]
              cases v.FORMULA <;> simp [agree u hu]
          | cons other rest => rfl

theorem stepWith_congr_of_preds (d : Graph) (ordinary reseed : Guard) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (agree : ∀ u ∈ predsOf d v, left u = right u) :
    stepWith d ordinary reseed left v = stepWith d ordinary reseed right v := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp [stepWith, source]
  · simp only [stepWith, source, Bool.false_eq_true, if_false]
    apply congrArg List.eraseDups
    apply congrArg₂ (· ++ ·)
    · by_cases pairs : (elimPairsAt d v).isEmpty = true
      · simp only [pairs, if_true]
        cases hp : predsOf d v with
        | nil => rfl
        | cons u rest =>
            cases rest with
            | nil =>
                have hu : u ∈ predsOf d v := by simp [hp]
                cases v.FORMULA <;> simp [agree u hu]
            | cons other rest => rfl
      · simp only [pairs, Bool.false_eq_true, if_false]
        apply congrArg₂ (· ++ ·)
        · apply List.flatMap_congr
          intro pair hpair
          rcases pair with ⟨major, minor⟩
          obtain ⟨hmajor, hminor⟩ := mem_preds_of_mem_elimPairsAt hpair
          dsimp only
          rw [agree minor hminor, agree major hmajor]
        · exact mixed_congr d ordinary v left right agree
    · exact reseed_congr d reseed v left right agree

def specificationTable (d : Graph) (ordinary reseed : Guard) (fuel : Nat) : FlowTable :=
  d.NODES.map fun v => (v, evaluateAt d ordinary reseed fuel v)

theorem fromTable_spec (d : Graph) (ordinary reseed : Guard) (fuel : Nat) (v : Vertex) :
    fromTable d ordinary reseed fuel (specificationTable d ordinary reseed fuel) v =
      evaluateAt d ordinary reseed fuel v := by
  simpa [fromTable, specificationTable, routesFromCacheTable] using
    routesFromCacheTable_exact d.NODES (evaluateAt d ordinary reseed fuel) v

theorem tableStep_spec (d : Graph) (ordinary reseed : Guard) (fuel : Nat) :
    tableStep d ordinary reseed fuel (specificationTable d ordinary reseed fuel) =
      specificationTable d ordinary reseed (fuel + 1) := by
  simp only [tableStep, specificationTable]
  apply List.map_congr_left
  intro v hv
  simp only [Prod.mk.injEq, true_and]
  change stepWith d ordinary reseed
      (fromTable d ordinary reseed fuel (specificationTable d ordinary reseed fuel)) v =
    stepWith d ordinary reseed (evaluateAt d ordinary reseed fuel) v
  rw [show fromTable d ordinary reseed fuel (specificationTable d ordinary reseed fuel) =
      evaluateAt d ordinary reseed fuel from funext (fromTable_spec d ordinary reseed fuel)]

private theorem fixed_next (d : Graph) (ordinary reseed : Guard)
    (closed : flowTableClosedB d = true) (fuel : Nat)
    (fixed : specificationTable d ordinary reseed (fuel + 1) =
      specificationTable d ordinary reseed fuel) :
    specificationTable d ordinary reseed (fuel + 2) =
      specificationTable d ordinary reseed (fuel + 1) := by
  have atNode : ∀ u ∈ d.NODES,
      evaluateAt d ordinary reseed (fuel + 1) u = evaluateAt d ordinary reseed fuel u := by
    intro u hu
    have eq := congrArg (fun table => flowTableLookup? table u) fixed
    simpa [specificationTable, CacheCorrectness.flowTableLookup?_map_of_mem d.NODES _ hu] using eq
  apply List.map_congr_left
  intro v hv
  simp only [Prod.mk.injEq, true_and]
  apply stepWith_congr_of_preds
  intro u hu
  exact atNode u (CacheCorrectness.mem_predsOf_mem_nodes_of_flowTableClosedB d closed hu)

private theorem fixed_stable (d : Graph) (ordinary reseed : Guard)
    (closed : flowTableClosedB d = true) (fuel : Nat)
    (fixed : specificationTable d ordinary reseed (fuel + 1) =
      specificationTable d ordinary reseed fuel) :
    ∀ extra, specificationTable d ordinary reseed (fuel + 1 + extra) =
      specificationTable d ordinary reseed (fuel + 1) := by
  have steps : ∀ extra, specificationTable d ordinary reseed (fuel + extra + 1) =
      specificationTable d ordinary reseed (fuel + extra) := by
    intro extra
    induction extra with
    | zero => simpa using fixed
    | succ extra ih =>
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          fixed_next d ordinary reseed closed (fuel + extra) ih
  intro extra
  induction extra with
  | zero => rfl
  | succ extra ih =>
      calc
        specificationTable d ordinary reseed (fuel + 1 + (extra + 1)) =
            specificationTable d ordinary reseed (fuel + 1 + extra) := by
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using steps (extra + 1)
        _ = specificationTable d ordinary reseed (fuel + 1) := ih

private theorem iterate_spec (d : Graph) (ordinary reseed : Guard) : ∀ remaining fuel,
    iterate d ordinary reseed remaining fuel (specificationTable d ordinary reseed fuel) =
      specificationTable d ordinary reseed (fuel + remaining) := by
  intro remaining
  induction remaining with
  | zero => intro fuel; rfl
  | succ remaining ih =>
      intro fuel
      rw [iterate, tableStep_spec]
      split
      · rename_i stopped
        have parts : flowTableClosedB d = true ∧
            specificationTable d ordinary reseed (fuel + 1) =
              specificationTable d ordinary reseed fuel := by simpa using stopped
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          (fixed_stable d ordinary reseed parts.1 fuel parts.2 remaining).symm
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ih (fuel + 1)



theorem tableAt_eq_spec (d : Graph) (ordinary reseed : Guard) (fuel : Nat) :
    tableAt d ordinary reseed fuel = specificationTable d ordinary reseed fuel := by
  unfold tableAt
  simpa only [Nat.zero_add] using iterate_spec d ordinary reseed fuel 0

theorem guardFreeRootsB_eq_raw (d : Graph) : guardFreeRootsB d =
    (d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
      (let routes := guardFreeFlowAt d (stdFuel d) v
       !routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty)) := by
  simp only [guardFreeRootsB, rootsFromTableB, guardFreeTable, tableAt_eq_spec, fromTable_spec]


theorem cached_historical_roots_eq (d : Graph) :
    (d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
      (let routes := cachedFlowAt d (flowTable d) v
       !routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty)) =
    rootsDischargedB d := by
  simp only [rootsDischargedB, CacheCorrectness.cachedFlowAt_flowTable]

#print axioms historical_evaluate_eq
#print axioms stepWith_congr_of_preds
#print axioms fromTable_spec
#print axioms tableStep_spec
#print axioms tableAt_eq_spec
#print axioms guardFreeRootsB_eq_raw
#print axioms cached_historical_roots_eq

end LambdaFlow
