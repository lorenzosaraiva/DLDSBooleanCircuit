import Compression.Delivery.HCJustifiedRouteTransport






open Semantic FlowSpec OccurrenceAwareReplay CandidateMixedReplay
open RepresentedUseTransport HCJustifiedRouteTransport

namespace ContinuationPersistence

structure TransportEvent where
  transitionIndex : Nat
  owner : Nat
  before : Ancestral
  after : Ancestral
deriving Repr

structure State where
  base : CandidateState
  live : List Evidence
  transports : List TransportEvent
  retired : Nat

def initialState (graph : Graph) : State :=
  ⟨RepresentedUseTransport.initialState graph, [], [], 0⟩

structure Restoration (before after : Graph) where
  old : Evidence
  moved : Evidence
  ready : Ready before after old moved

def restoration? (selected : List Vertex) (replacement : Vertex)
    (before after : Graph) (old : Evidence) : Option (Restoration before after) :=
  let moved := old.transport selected replacement
  (ready? before after old moved).map fun proof => ⟨old, moved, proof.down⟩

def appendMissing (paths : List Ancestral) (records : List Ancestral) :
    List Ancestral :=
  records.foldl (fun current record =>
    if current.contains record then current else current ++ [record]) paths

theorem appendMissing_preserves (paths records : List Ancestral)
    (record : Ancestral) (present : record ∈ paths) :
    record ∈ appendMissing paths records := by
  unfold appendMissing
  induction records generalizing paths with
  | nil => exact present
  | cons first remaining ih =>
    simp only [List.foldl_cons]
    apply ih
    split <;> simp_all

theorem appendMissing_includes (paths records : List Ancestral)
    (record : Ancestral) (present : record ∈ records) :
    record ∈ appendMissing paths records := by
  induction records generalizing paths with
  | nil => simp at present
  | cons first remaining ih =>
    simp only [List.mem_cons] at present
    simp only [appendMissing, List.foldl_cons]
    rcases present with rfl | member
    · apply appendMissing_preserves
      split <;> simp_all
    · exact ih _ member

theorem appendMissing_no_new_record (paths records : List Ancestral)
    (record : Ancestral) (present : record ∈ appendMissing paths records) :
    record ∈ paths ∨ record ∈ records := by
  induction records generalizing paths with
  | nil => exact Or.inl present
  | cons first remaining ih =>
    simp only [appendMissing, List.foldl_cons] at present
    obtain old | new := ih _ present
    · split at old
      · exact Or.inl old
      · simp only [List.mem_append, List.mem_singleton] at old
        rcases old with old | rfl
        · exact Or.inl old
        · exact Or.inr (by simp)
    · exact Or.inr (List.mem_cons_of_mem first new)

theorem appendMissing_length_bound (paths records : List Ancestral) :
    (appendMissing paths records).length ≤ paths.length + records.length := by
  induction records generalizing paths with
  | nil => simp [appendMissing]
  | cons first remaining ih =>
    simp only [appendMissing, List.foldl_cons]
    split
    · have bound := ih paths
      simp only [appendMissing] at bound
      simp only [List.length_cons]
      omega
    · have bound := ih (paths ++ [first])
      simp only [appendMissing, List.length_append, List.length_cons,
        List.length_nil] at bound ⊢
      omega

theorem appendMissing_unchanged_of_present (paths records : List Ancestral)
    (present : ∀ record ∈ records, record ∈ paths) :
    appendMissing paths records = paths := by
  induction records generalizing paths with
  | nil => rfl
  | cons first remaining ih =>
    have firstPresent : paths.contains first = true := by
      simpa using present first (by simp)
    simp only [appendMissing, List.foldl_cons, firstPresent, ite_true]
    apply ih
    exact fun record member => present record (List.mem_cons_of_mem first member)

def step (partition : List HCClass) (transitionIndex : Nat)
    (selected : List Vertex) (state : State) : State :=
  let before := state.base.graph
  let rule := compress_ruleWithOccurrences partition selected before
  let production := compress_nodesWithPartitionCarrying partition selected before
    state.base.occurrences
  let pending := state.base.pending.map fun p =>
    p.rewriteCurrentEndpoints selected rule.1.CENTER
  let pool := pending ++ extractAt selected before production.1 rule.1.CENTER
  let activations := pool.filterMap (justifiedActivation? production.1)
  let added := (activations.map fun a => a.pending.use.record).eraseDups
  let events := added.filterMap fun record =>
    (activations.find? fun a => a.pending.use.record == record).map fun a => {
      transitionIndex := transitionIndex
      owner := a.pending.use.owner.NUMBER
      centre := a.pending.use.centre.NUMBER
      consumer := a.pending.use.consumer.NUMBER
      address := a.pending.use.address : ActivationEvent }
  let afterActivation := dlds production.1.NODES production.1.EDGES
    (production.1.PATHS ++ added)
  let restorations := state.live.filterMap
    (restoration? selected rule.1.CENTER before afterActivation)
  let records := (restorations.map fun r => r.moved.use.record).eraseDups
  let graph := dlds afterActivation.NODES afterActivation.EDGES
    (appendMissing afterActivation.PATHS records)
  let moved := state.live.map (fun old => old.transport selected rule.1.CENTER)
  let retained := moved.filter fun e =>
    graph.PATHS.contains e.use.record && e.use.currentUseB graph
  let newEvidence := activations.map Evidence.ofActivation
  {
    base := {
      graph := graph
      occurrences := production.2
      pending := pool.filter (fun p => !added.contains p.use.record)
      activations := state.base.activations ++ events }
    live := retained ++ newEvidence
    transports := state.transports ++ restorations.map (fun r => {
      transitionIndex := transitionIndex
      owner := r.old.use.owner.NUMBER
      before := r.old.use.record
      after := r.moved.use.record })
    retired := state.retired + (moved.length - retained.length) }

def runSchedule (partition : List HCClass) :
    Nat -> List (List Vertex) -> State -> State
  | _, [], state => state
  | index, selected :: remaining, state =>
    runSchedule partition (index + 1) remaining (step partition index selected state)

def candidateState (input : Graph) : State :=
  let partition := hcPartition input
  let schedule := (repeated_nodes_vertex_indexedWithPartition partition input).flatten
  runSchedule partition 0 schedule (initialState input)

end ContinuationPersistence

#print axioms ContinuationPersistence.appendMissing_preserves
#print axioms ContinuationPersistence.appendMissing_includes
#print axioms ContinuationPersistence.appendMissing_no_new_record
#print axioms ContinuationPersistence.appendMissing_length_bound
#print axioms ContinuationPersistence.appendMissing_unchanged_of_present
