import Compression.Delivery.EliminationUseTransport






open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay
open RepresentedUseTransport HCJustifiedRouteTransport HCCompleteEliminationContinuation

namespace EliminationContinuation

structure Pending where
  selected : List Vertex
  before : Graph
  use : CompleteRepresentedUseContinuation
  boundaryIndex : Nat

def Pending.transport (selected : List Vertex) (replacement : Vertex) (p : Pending) : Pending :=
  { p with use := p.use.rewriteCurrentEndpoints selected replacement }

structure Activation (graph : Graph) where
  pending : Pending
  sound : pending.use.producer.Sound pending.selected pending.before graph (premiseFlow graph)
  ancestry : (ancestorsInto graph pending.use.producer.centre).isEmpty = false
  exclusive : ∀ source ∈ derivedOutputs graph pending.use.producer.centre (premiseFlow graph),
    colourPathPrefixB source.2 pending.use.address = true → source = pending.use.producer.logicalOutput

def activate? (graph : Graph) (p : Pending) : Option (Activation graph) :=
  if sound : p.use.producer.Sound p.selected p.before graph (premiseFlow graph) then
    if ancestry : (ancestorsInto graph p.use.producer.centre).isEmpty = false then
      if exclusive : ∀ source ∈ derivedOutputs graph p.use.producer.centre (premiseFlow graph),
          colourPathPrefixB source.2 p.use.address = true → source = p.use.producer.logicalOutput then
        some ⟨p, sound, ancestry, exclusive⟩
      else none
    else none
  else none

theorem Activation.delivery {graph} (a : Activation graph) :
    a.pending.use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto graph a.pending.use.producer.centre ++ [a.pending.use.record])
      (derivedOutputs graph a.pending.use.producer.centre (premiseFlow graph)) :=
  (a.pending.use.producer_to_complete_delivery a.sound).2.1

structure ContinuationEvent where
  transitionIndex : Nat
  owner : Vertex
  originalConsumer : Vertex
  beforeConsumer : Vertex
  incoming : List Deduction
  outgoing : List Deduction
  direct : List Ancestral
  record : Ancestral

structure State where
  base : ContinuationPersistence.State
  waiting : List EliminationUseTransport.Pending
  pending : List Pending
  activations : List ActivationEvent
  continuations : List ContinuationEvent
  unsupported : List (Nat × Vertex)

def initialState (graph : Graph) : State := ⟨ContinuationPersistence.initialState graph, [], [], [], [], []⟩

def stepWithOrigins (partition : List HCClass) (index : Nat) (selected owners : List Vertex)
    (remaining : List (List Vertex)) (state : State) : State :=
  let before := state.base.base.graph
  let rule := compress_ruleWithOccurrences partition selected before
  let production := compress_nodesWithPartitionCarrying partition selected before state.base.base.occurrences
  let consumerReady := state.waiting.filterMap fun p =>
    (atConsumer? before selected p.use).map fun c =>
      ({ selected := p.selected, before := p.before, use := c, boundaryIndex := index } : Pending)
  let waitingMoved := (state.waiting.filter fun p =>
    !(consumerReady.any fun c => c.use.producer.owner == p.use.owner)).map
      (fun p => p.transport selected rule.1.CENTER)
  let extracted := (EliminationUseTransport.extractAt selected before production.1 rule.1.CENTER).filter
    (fun p => owners.contains p.use.owner)
  let sourcePool := waitingMoved ++ extracted
  let directReady := sourcePool.filterMap fun p => (direct? remaining p.use).map fun c =>
    ({ selected := p.selected, before := p.before, use := c, boundaryIndex := index } : Pending)
  let completePool := state.pending.map (fun p => p.transport selected rule.1.CENTER) ++
    consumerReady.map (fun p => p.transport selected rule.1.CENTER) ++ directReady
  let introPool := state.base.base.pending.map (fun p => p.rewriteCurrentEndpoints selected rule.1.CENTER) ++
    RepresentedUseTransport.extractAt selected before production.1 rule.1.CENTER
  let intro := introPool.filterMap (justifiedActivation? production.1)
  let elim := completePool.filterMap (activate? production.1)
  let added := ((intro.map fun a => a.pending.use.record) ++ (elim.map fun a => a.pending.use.record)).eraseDups
    |>.filter (fun record => !production.1.PATHS.contains record)
  let afterActivation := dlds production.1.NODES production.1.EDGES (production.1.PATHS ++ added)
  let restorations := state.base.live.filterMap
    (ContinuationPersistence.restoration? selected rule.1.CENTER before afterActivation)
  let graph := dlds afterActivation.NODES afterActivation.EDGES
    (ContinuationPersistence.appendMissing afterActivation.PATHS
      (restorations.map fun r => r.moved.use.record).eraseDups)
  let moved := state.base.live.map (fun old => old.transport selected rule.1.CENTER)
  let retained := moved.filter fun e => graph.PATHS.contains e.use.record && e.use.currentUseB graph
  let newEvidence := intro.map Evidence.ofActivation ++
    elim.map (fun a => Evidence.ofCompleteElimination a.pending.use a.sound)
  let introEvents := intro.map fun a => {
    transitionIndex := index, owner := a.pending.use.owner.NUMBER,
    centre := a.pending.use.centre.NUMBER, consumer := a.pending.use.consumer.NUMBER,
    address := a.pending.use.address : ActivationEvent }
  let elimEvents := elim.map fun a => {
    transitionIndex := index, owner := a.pending.use.producer.owner.NUMBER,
    centre := a.pending.use.producer.centre.NUMBER, consumer := a.pending.use.producer.consumer.NUMBER,
    address := a.pending.use.address : ActivationEvent }
  {
    base := {
      base := {
        graph := graph, occurrences := production.2,
        pending := introPool.filter (fun p => !added.contains p.use.record),
        activations := state.base.base.activations ++ introEvents }
      live := retained ++ newEvidence
      transports := state.base.transports ++ restorations.map (fun r => {
        transitionIndex := index, owner := r.old.use.owner.NUMBER,
        before := r.old.use.record, after := r.moved.use.record })
      retired := state.base.retired + (moved.length - retained.length) }
    waiting := sourcePool.filter (fun p =>
      !(directReady.any fun c => c.use.producer.owner == p.use.owner))
    pending := completePool.filter (fun p => !(elim.any fun a => a.pending.use.record == p.use.record))
    activations := state.activations ++ elimEvents
    continuations := state.continuations ++ consumerReady.map (fun p => {
      transitionIndex := index, owner := p.use.producer.owner,
      originalConsumer := p.use.producer.originalConsumer, beforeConsumer := p.use.producer.consumer,
      incoming := get_rule.incoming p.use.producer.consumer before,
      outgoing := get_rule.outgoing p.use.producer.consumer before,
      direct := get_rule.direct p.use.producer.consumer before, record := p.use.record })
    unsupported := state.unsupported ++ (state.waiting.filter (fun p =>
      selected.contains p.use.consumer && (atConsumer? before selected p.use).isNone)).map
        (fun p => (index, p.use.owner)) }

def step (partition : List HCClass) (index : Nat) (selected : List Vertex)
    (remaining : List (List Vertex)) (state : State) : State :=
  stepWithOrigins partition index selected selected remaining state

def runSchedule (partition : List HCClass) : Nat → List (List Vertex) → State → State
  | _, [], state => state
  | index, selected :: remaining, state =>
    runSchedule partition (index + 1) remaining (step partition index selected remaining state)

def selectionState (input : Graph) : State :=
  let partition := hcPartition input
  let schedule := (repeated_nodes_vertex_indexedWithPartition partition input).flatten
  runSchedule partition 0 schedule (initialState input)

end EliminationContinuation

#print axioms EliminationContinuation.Activation.delivery
