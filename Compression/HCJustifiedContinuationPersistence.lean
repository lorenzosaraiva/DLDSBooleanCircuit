import Compression.Delivery.IntroductionContinuation











open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay
open RepresentedUseTransport HCJustifiedRouteTransport

namespace HCJustifiedContinuationPersistence

def originalConsumer : Use → Vertex
  | .introduction u => u.originalConsumer
  | .elimination u => u.originalConsumer
  | .completeElimination u => u.producer.originalConsumer
  | .completeIntroduction u => u.producer.originalConsumer



def OriginalUse (initial : Graph) (use : Use) : Prop :=
  use.owner ∈ initial.NODES ∧ originalConsumer use ∈ initial.NODES ∧
    ∃ outgoing ∈ initial.EDGES,
      outgoing.START = use.owner ∧ outgoing.END = originalConsumer use

instance (initial use) : Decidable (OriginalUse initial use) := by
  unfold OriginalUse
  infer_instance

def attach? (initial : Graph) (owner consumer : Vertex) (evidence : Evidence) :
    Option { e : Evidence // OriginalUse initial e.use ∧
      e.use.owner = owner ∧ originalConsumer e.use = consumer } :=
  if origin : OriginalUse initial evidence.use then
    if same : evidence.use.owner = owner ∧ originalConsumer evidence.use = consumer then
      some ⟨evidence, origin, same⟩
    else none
  else none

theorem foreign_use_rejected (initial : Graph) (owner consumer : Vertex) (evidence : Evidence)
    (foreign : evidence.use.owner ≠ owner ∨ originalConsumer evidence.use ≠ consumer) :
    attach? initial owner consumer evidence = none := by
  simp only [attach?]
  split
  · have different : ¬ (evidence.use.owner = owner ∧ originalConsumer evidence.use = consumer) := by
      rcases foreign with h | h <;> tauto
    simp [different]
  · rfl

theorem originalUse_transport (initial : Graph) (selected : List Vertex)
    (replacement : Vertex) (evidence : Evidence) (origin : OriginalUse initial evidence.use) :
    OriginalUse initial (evidence.transport selected replacement).use := by
  cases evidence with
  | mk use justified => cases use <;> exact origin

structure Registration (initial graph : Graph) where
  evidence : Evidence
  originalUse : OriginalUse initial evidence.use
  present : evidence.use.record ∈ graph.PATHS
  produced : evidence.use.sourceOutput ∈ logical graph evidence.use
  exclusive : ∀ source ∈ logical graph evidence.use,
    colourPathPrefixB source.2 evidence.use.address = true →
      source = evidence.use.sourceOutput

def register? (initial graph : Graph) (evidence : Evidence) : Option (Registration initial graph) :=
  if origin : OriginalUse initial evidence.use then
   if present : evidence.use.record ∈ graph.PATHS then
    if produced : evidence.use.sourceOutput ∈ logical graph evidence.use then
      if exclusive : ∀ source ∈ logical graph evidence.use,
          colourPathPrefixB source.2 evidence.use.address = true →
            source = evidence.use.sourceOutput then
        some ⟨evidence, origin, present, produced, exclusive⟩
      else none
    else none
  else none
  else none

theorem Registration.delivery {initial graph : Graph} (registration : Registration initial graph) :
    registration.evidence.use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto graph registration.evidence.use.centre)
      (logical graph registration.evidence.use) := by
  rw [reseedDerivedRoutes, List.mem_eraseDups, List.mem_flatMap]
  refine ⟨registration.evidence.use.record, ?_, ?_⟩
  · have present := registration.present
    simpa [ancestorsInto, Use.record] using present
  · rw [List.mem_filterMap]
    refine ⟨registration.evidence.use.sourceOutput, registration.produced, ?_⟩
    simp only [Use.record, Use.source_prefix, ite_true]
    rw [Use.source_dependency, ← Use.output_eq]

def captureElimination? (initial before after : Graph) (selected : List Vertex)
    (replacement : Vertex) (pending : EliminationUseTransport.Pending) :
    Option (Registration initial after) := do
  let complete ← HCCompleteEliminationContinuation.atConsumer? before selected pending.use
  let moved := complete.rewriteCurrentEndpoints selected replacement
  if sound : moved.producer.Sound pending.selected pending.before after (premiseFlow after) then
    register? initial after (Evidence.ofCompleteElimination moved sound)
  else none



structure Restoration (input before after : Graph) (selected : List Vertex)
    (replacement : Vertex) where
  old : Evidence
  moved : Evidence
  originalUse : OriginalUse input old.use
  actualTransport : moved = old.transport selected replacement
  ready : Ready before after old moved

def restore? (input : Graph) (selected : List Vertex) (replacement : Vertex)
    (before after : Graph) (old : Evidence) :
    Option (Restoration input before after selected replacement) :=
  if origin : OriginalUse input old.use then
    let moved := old.transport selected replacement
    (ready? before after old moved).map fun proof => ⟨old, moved, origin, rfl, proof.down⟩
  else none



theorem Restoration.contract {input before after selected replacement}
    (r : Restoration input before after selected replacement) :
    r.old.use.record ∈ before.PATHS ∧ r.moved.use.record ∉ after.PATHS ∧
    OriginalUse input r.moved.use ∧
    r.moved.use.record = collapse.rewrite_path_endpoints selected replacement r.old.use.record ∧
    r.moved.use.currentUseB after = true ∧
    r.moved.use.sourceOutput ∈ logical after r.moved.use := by
  refine ⟨r.ready.oldPresent, r.ready.absent, ?_, ?_, r.ready.survivingUse, r.ready.produced⟩
  · rw [r.actualTransport]
    exact originalUse_transport input selected replacement r.old r.originalUse
  · rw [r.actualTransport]
    exact transport_is_actual_endpoint_map selected replacement r.old

theorem restore_existing_rejected (input selected replacement before after old)
    (present : (old.transport selected replacement).use.record ∈ after.PATHS) :
    restore? input selected replacement before after old = none := by
  simp only [restore?]
  split
  · simp [ready?, present]
  · rfl

theorem restore_absent_use_rejected (input selected replacement before after old)
    (absent : (old.transport selected replacement).use.currentUseB after ≠ true) :
    restore? input selected replacement before after old = none := by
  simp only [restore?]
  split
  · simp [ready?, absent]
  · rfl

theorem restore_foreign_rejected (input selected replacement before after old)
    (foreign : ¬ OriginalUse input old.use) :
    restore? input selected replacement before after old = none := by
  simp [restore?, foreign]


export ContinuationPersistence (appendMissing appendMissing_preserves appendMissing_includes
  appendMissing_no_new_record appendMissing_length_bound appendMissing_unchanged_of_present)

theorem appendMissing_idempotent (paths records : List Ancestral) :
    appendMissing (appendMissing paths records) records = appendMissing paths records := by
  apply appendMissing_unchanged_of_present
  exact fun record member => appendMissing_includes paths records record member



theorem appendMissing_nodup (paths records : List Ancestral) (unique : paths.Nodup) :
    (appendMissing paths records).Nodup := by
  induction records generalizing paths with
  | nil => exact unique
  | cons first rest ih =>
    simp only [ContinuationPersistence.appendMissing, List.foldl_cons]
    apply ih
    split
    · exact unique
    · simpa using (List.nodup_concat paths first).mpr
        ⟨by simpa using ‹¬ paths.contains first = true›, unique⟩

structure State where
  original : Graph
  graph : Graph
  carried : List Ancestral
  elimination : List EliminationUseTransport.Pending
  introduction : List PendingUse
  live : List Evidence
  registrations : Nat := 0
  restorations : List ContinuationPersistence.TransportEvent := []
  retirements : Nat := 0

def initial (graph : Graph) : State := {
  original := graph, graph := graph, carried := [], elimination := [], introduction := [], live := [] }

structure StepResult where
  state : State
  raw : Graph
  replacement : Vertex
  registered : List Evidence
  restored : List Evidence
  retired : List Evidence

def step (partition : List HCClass) (index : Nat) (selected : List Vertex)
    (state : State) : StepResult := Id.run do
  let before := state.graph
  let rule := compress_ruleWithOccurrences partition selected before
  let production := compress_nodesWithPartitionCarrying partition selected before state.carried
  let elimination := state.elimination.filterMap
    (captureElimination? state.original before production.1 selected rule.1.CENTER)
  let introduction := state.introduction.filterMap fun pending => do
    let capture ← IntroductionContinuation.capture? before selected rule.1.CENTER production.1 pending
    register? state.original production.1 capture.evidence
  let registered := (elimination ++ introduction).map Registration.evidence
  let restorations := state.live.filterMap
    (restore? state.original selected rule.1.CENTER before production.1)
  let records := (restorations.map fun r => r.moved.use.record).eraseDups
  let graph := dlds production.1.NODES production.1.EDGES
    (appendMissing production.1.PATHS records)
  let moved := state.live.map (fun old => old.transport selected rule.1.CENTER)
  let retained := moved.filter fun e => graph.PATHS.contains e.use.record && e.use.currentUseB graph
  let retired := moved.filter fun e => !(graph.PATHS.contains e.use.record && e.use.currentUseB graph)
  let eliminationPool := state.elimination.map (fun p => p.transport selected rule.1.CENTER) ++
    EliminationUseTransport.extractAt selected before graph rule.1.CENTER
  let introductionPool := state.introduction.map (fun p => p.rewriteCurrentEndpoints selected rule.1.CENTER) ++
    RepresentedUseTransport.extractAt selected before graph rule.1.CENTER
  let next : State := {
    original := state.original,
    graph := graph, carried := production.2,
    elimination := eliminationPool.filter (fun p => p.use.currentUseB graph),
    introduction := introductionPool.filter (fun p => p.use.currentUseB graph),
    live := retained ++ registered,
    registrations := state.registrations + registered.length,
    restorations := state.restorations ++ restorations.map (fun r => {
      transitionIndex := index, owner := r.old.use.owner.NUMBER,
      before := r.old.use.record, after := r.moved.use.record }),
    retirements := state.retirements + retired.length }
  return ⟨next, production.1, rule.1.CENTER, registered,
    restorations.map (fun r => r.moved), retired⟩

theorem step_nodes_edges_carried (partition index selected state) :
    let result := step partition index selected state
    let raw := compress_nodesWithPartitionCarrying partition selected state.graph state.carried
    result.state.graph.NODES = raw.1.NODES ∧ result.state.graph.EDGES = raw.1.EDGES ∧
      result.state.carried = raw.2 := by
  simp [step]

def run (partition : List HCClass) : Nat → List (List Vertex) → State → State
  | _, [], state => state
  | index, selected :: rest, state => run partition (index + 1) rest (step partition index selected state).state

def refineWithJustifiedPersistence (input : Graph) : State :=
  let partition := hcPartition input
  run partition 1 (repeated_nodes_vertex_indexedWithPartition partition input).flatten (initial input)

end HCJustifiedContinuationPersistence
