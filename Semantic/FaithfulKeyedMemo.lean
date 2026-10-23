import Semantic.FaithfulKeyedSimulation

open Semantic FlowSpec

namespace FaithfulKeyed

set_option maxHeartbeats 1000000

def keyedFromTable (d : Graph) (ordinary reseed : LambdaFlow.Guard) (fuel : Nat)
    (table : KeyedEval.Table d) (v : Vertex) : KeyedEval.Carrier d :=
  match KeyedEval.lookup d table v with
  | some rows => rows
  | none => keyedAt d ordinary reseed fuel v

def keyedTableStep (d : Graph) (ordinary reseed : LambdaFlow.Guard) (fuel : Nat)
    (table : KeyedEval.Table d) : KeyedEval.Table d :=
  d.NODES.map fun v => (v, keyedStep d ordinary reseed
    (keyedFromTable d ordinary reseed fuel table) v)



def keyedIterate (d : Graph) (ordinary reseed : LambdaFlow.Guard) :
    Nat → Nat → KeyedEval.Table d → KeyedEval.Table d
  | 0, _, table => table
  | remaining + 1, fuel, table =>
      let next := keyedTableStep d ordinary reseed fuel table
      if flowTableClosedB d && next == table then next
      else keyedIterate d ordinary reseed remaining (fuel + 1) next

def keyedTableAt (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) : KeyedEval.Table d :=
  keyedIterate d ordinary reseed fuel 0 (d.NODES.map fun v => (v, []))

def keyedSpecificationTable (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) : KeyedEval.Table d :=
  d.NODES.map fun v => (v, keyedAt d ordinary reseed fuel v)

private theorem keyed_lookup_map (d : Graph) (keys : List Vertex)
    (rows : Vertex → KeyedEval.Carrier d) {v : Vertex} (member : v ∈ keys) :
    KeyedEval.lookup d (keys.map fun u => (u, rows u)) v = some (rows v) := by
  induction keys with
  | nil => simp at member
  | cons u rest ih =>
      by_cases same : u = v
      · subst v
        simp [KeyedEval.lookup]
      · have tail : v ∈ rest := by
          rcases List.mem_cons.mp member with head | tail
          · exact False.elim (same head.symm)
          · exact tail
        simpa [KeyedEval.lookup, same] using ih tail

theorem keyedFromTable_spec (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) (v : Vertex) :
    keyedFromTable d ordinary reseed fuel (keyedSpecificationTable d ordinary reseed fuel) v =
      keyedAt d ordinary reseed fuel v := by
  by_cases member : v ∈ d.NODES
  · simp [keyedFromTable, keyedSpecificationTable, keyed_lookup_map d d.NODES _ member]
  · have missing : KeyedEval.lookup d (keyedSpecificationTable d ordinary reseed fuel) v = none := by
      have notFound : (d.NODES.map fun u => (u, keyedAt d ordinary reseed fuel u)).find?
          (fun entry => entry.1 == v) = none := by
        apply List.find?_eq_none.mpr
        intro entry hentry
        obtain ⟨u, hu, rfl⟩ := List.mem_map.mp hentry
        have different : u ≠ v := by
          intro eq
          exact member (eq ▸ hu)
        simpa using different
      simp [KeyedEval.lookup, keyedSpecificationTable, notFound]
    simp [keyedFromTable, missing]

theorem keyedTableStep_spec (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) :
    keyedTableStep d ordinary reseed fuel (keyedSpecificationTable d ordinary reseed fuel) =
      keyedSpecificationTable d ordinary reseed (fuel + 1) := by
  simp only [keyedTableStep, keyedSpecificationTable]
  apply List.map_congr_left
  intro v hv
  simp only [Prod.mk.injEq, true_and]
  change keyedStep d ordinary reseed
      (keyedFromTable d ordinary reseed fuel (keyedSpecificationTable d ordinary reseed fuel)) v =
    keyedStep d ordinary reseed (keyedAt d ordinary reseed fuel) v
  rw [show keyedFromTable d ordinary reseed fuel (keyedSpecificationTable d ordinary reseed fuel) =
      keyedAt d ordinary reseed fuel from funext (keyedFromTable_spec d ordinary reseed fuel)]

theorem keyedStep_congr_of_preds (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (v : Vertex) (left right : Vertex → KeyedEval.Carrier d)
    (agree : ∀ u ∈ predsOf d v, left u = right u) :
    keyedStep d ordinary reseed left v = keyedStep d ordinary reseed right v := by
  apply congrArg (KeyedEval.merge d)
  apply LambdaFlow.stepWith_congr_of_preds
  intro u hu
  dsimp only
  rw [agree u hu]

private theorem keyed_fixed_next (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (closed : flowTableClosedB d = true) (fuel : Nat)
    (fixed : keyedSpecificationTable d ordinary reseed (fuel + 1) =
      keyedSpecificationTable d ordinary reseed fuel) :
    keyedSpecificationTable d ordinary reseed (fuel + 2) =
      keyedSpecificationTable d ordinary reseed (fuel + 1) := by
  have atNode : ∀ u ∈ d.NODES,
      keyedAt d ordinary reseed (fuel + 1) u = keyedAt d ordinary reseed fuel u := by
    intro u hu
    have eq := congrArg (fun table => KeyedEval.lookup d table u) fixed
    simpa [keyedSpecificationTable, keyed_lookup_map d d.NODES _ hu] using eq
  apply List.map_congr_left
  intro v hv
  simp only [Prod.mk.injEq, true_and]
  apply keyedStep_congr_of_preds
  intro u hu
  exact atNode u (CacheCorrectness.mem_predsOf_mem_nodes_of_flowTableClosedB d closed hu)

private theorem keyed_fixed_stable (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (closed : flowTableClosedB d = true) (fuel : Nat)
    (fixed : keyedSpecificationTable d ordinary reseed (fuel + 1) =
      keyedSpecificationTable d ordinary reseed fuel) :
    ∀ extra, keyedSpecificationTable d ordinary reseed (fuel + 1 + extra) =
      keyedSpecificationTable d ordinary reseed (fuel + 1) := by
  have steps : ∀ extra, keyedSpecificationTable d ordinary reseed (fuel + extra + 1) =
      keyedSpecificationTable d ordinary reseed (fuel + extra) := by
    intro extra
    induction extra with
    | zero => simpa using fixed
    | succ extra ih =>
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          keyed_fixed_next d ordinary reseed closed (fuel + extra) ih
  intro extra
  induction extra with
  | zero => rfl
  | succ extra ih =>
      calc
        keyedSpecificationTable d ordinary reseed (fuel + 1 + (extra + 1)) =
            keyedSpecificationTable d ordinary reseed (fuel + 1 + extra) := by
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using steps (extra + 1)
        _ = keyedSpecificationTable d ordinary reseed (fuel + 1) := ih

private theorem keyedIterate_spec (d : Graph) (ordinary reseed : LambdaFlow.Guard) :
    ∀ remaining fuel,
      keyedIterate d ordinary reseed remaining fuel (keyedSpecificationTable d ordinary reseed fuel) =
        keyedSpecificationTable d ordinary reseed (fuel + remaining) := by
  intro remaining
  induction remaining with
  | zero => intro fuel; rfl
  | succ remaining ih =>
      intro fuel
      rw [keyedIterate, keyedTableStep_spec]
      split
      · rename_i stopped
        have parts : flowTableClosedB d = true ∧
            keyedSpecificationTable d ordinary reseed (fuel + 1) =
              keyedSpecificationTable d ordinary reseed fuel := by simpa using stopped
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          (keyed_fixed_stable d ordinary reseed parts.1 fuel parts.2 remaining).symm
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ih (fuel + 1)



theorem keyedTableAt_eq_spec (d : Graph) (ordinary reseed : LambdaFlow.Guard) (fuel : Nat) :
    keyedTableAt d ordinary reseed fuel = keyedSpecificationTable d ordinary reseed fuel := by
  unfold keyedTableAt
  simpa only [Nat.zero_add] using keyedIterate_spec d ordinary reseed fuel 0

theorem keyedTableAt_lookup (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) (v : Vertex) (hv : v ∈ d.NODES) :
    KeyedEval.lookup d (keyedTableAt d ordinary reseed fuel) v =
      some (keyedAt d ordinary reseed fuel v) := by
  rw [keyedTableAt_eq_spec]
  exact keyed_lookup_map d d.NODES _ hv

theorem keyedTableAt_roots (d : Graph) (ordinary reseed : LambdaFlow.Guard) (fuel : Nat) :
    KeyedEval.rootsAcceptedB d (keyedTableAt d ordinary reseed fuel) =
      keyedRootsWithB d ordinary reseed fuel := by
  apply Bool.eq_iff_iff.mpr
  simp only [KeyedEval.rootsAcceptedB, keyedRootsWithB, List.all_eq_true]
  apply forall_congr'
  intro v
  apply forall_congr'
  intro hv
  simp [keyedTableAt_lookup d ordinary reseed fuel v hv]

theorem keyedTableAt_roots_eq_raw (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (ho : DepBlind ordinary) (hr : DepBlind reseed) (fuel : Nat) :
    KeyedEval.rootsAcceptedB d (keyedTableAt d ordinary reseed fuel) =
      rootsDischargedWithB d ordinary reseed fuel := by
  rw [keyedTableAt_roots, keyedRootsWithB_eq d ordinary reseed ho hr]

private theorem keyedFromTable_allowAll (d : Graph) (fuel : Nat)
    (table : KeyedEval.Table d) (v : Vertex) :
    keyedFromTable d LambdaFlow.allowAll LambdaFlow.allowAll fuel table v =
      LambdaKeyed.fromTable d fuel table v := by
  simp only [keyedFromTable, LambdaKeyed.fromTable, keyedAt_allowAll]
  rfl

private theorem keyedTableStep_allowAll (d : Graph) (fuel : Nat)
    (table : KeyedEval.Table d) :
    keyedTableStep d LambdaFlow.allowAll LambdaFlow.allowAll fuel table =
      LambdaKeyed.tableStep d fuel table := by
  simp only [keyedTableStep, LambdaKeyed.tableStep]
  rw [show keyedFromTable d LambdaFlow.allowAll LambdaFlow.allowAll fuel table =
    LambdaKeyed.fromTable d fuel table from funext (keyedFromTable_allowAll d fuel table)]
  rfl

private theorem keyedIterate_allowAll (d : Graph) : ∀ remaining fuel table,
    keyedIterate d LambdaFlow.allowAll LambdaFlow.allowAll remaining fuel table =
      LambdaKeyed.iterate d remaining fuel table := by
  intro remaining
  induction remaining with
  | zero => intro fuel table; rfl
  | succ remaining ih =>
      intro fuel table
      simp only [keyedIterate, LambdaKeyed.iterate, keyedTableStep_allowAll]
      split
      · rfl
      · exact ih _ _

theorem keyedTableAt_allowAll (d : Graph) :
    keyedTableAt d LambdaFlow.allowAll LambdaFlow.allowAll (stdFuel d) =
      LambdaKeyed.canonicalTable d :=
  keyedIterate_allowAll d (stdFuel d) 0 _




theorem canonicalAcceptedB_eq_guardFreeRootsB (d : Graph) :
    LambdaKeyed.canonicalAcceptedB d = LambdaFlow.guardFreeRootsB d := by
  unfold LambdaKeyed.canonicalAcceptedB
  rw [← keyedTableAt_allowAll, keyedTableAt_roots]
  exact keyed_guardFree_roots d

#print axioms keyedFromTable_spec
#print axioms keyedTableStep_spec
#print axioms keyedStep_congr_of_preds
#print axioms keyedTableAt_eq_spec
#print axioms keyedTableAt_lookup
#print axioms keyedTableAt_roots
#print axioms keyedTableAt_roots_eq_raw
#print axioms keyedTableAt_allowAll
#print axioms canonicalAcceptedB_eq_guardFreeRootsB

end FaithfulKeyed
