import Compression.Delivery.ContinuationPersistence










open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay
open RepresentedUseTransport HCJustifiedRouteTransport
open HCEliminationUse (RepresentedEliminationUse)

namespace EliminationUseTransport

structure Pending where
  selected : List Vertex
  before : Graph
  use : RepresentedEliminationUse

def Pending.transport (selected : List Vertex) (replacement : Vertex) (p : Pending) : Pending :=
  { p with use := p.use.rewriteCurrentEndpoints selected replacement }

structure Activation (graph : Graph) where
  pending : Pending
  sound : pending.use.Sound pending.selected pending.before graph (premiseFlow graph)
  ancestry : (ancestorsInto graph pending.use.centre).isEmpty = false
  absent : pending.use.record ∉ graph.PATHS
  exclusive : ∀ source ∈ derivedOutputs graph pending.use.centre (premiseFlow graph),
    colourPathPrefixB source.2 pending.use.address = true → source = pending.use.logicalOutput

def activate? (graph : Graph) (p : Pending) : Option (Activation graph) :=
  if sound : p.use.Sound p.selected p.before graph (premiseFlow graph) then
    if ancestry : (ancestorsInto graph p.use.centre).isEmpty = false then
      if absent : p.use.record ∉ graph.PATHS then
        if exclusive : ∀ source ∈ derivedOutputs graph p.use.centre (premiseFlow graph),
            colourPathPrefixB source.2 p.use.address = true → source = p.use.logicalOutput then
          some ⟨p, sound, ancestry, absent, exclusive⟩
        else none
      else none
    else none
  else none

def extractAt (selected : List Vertex) (before after : Graph) (centre : Vertex) : List Pending :=
  selected.filterMap fun owner =>
    (HCEliminationUse.producerDerivedUse? selected before after centre owner (premiseFlow after)).map
      fun extracted => ⟨selected, before, extracted.1⟩

theorem Activation.delivery {graph : Graph} (a : Activation graph) :
    a.pending.use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto graph a.pending.use.centre ++ [a.pending.use.record])
      (derivedOutputs graph a.pending.use.centre (premiseFlow graph)) :=
  (RepresentedEliminationUse.producer_to_delivery a.sound).2.2.1

structure State where
  base : ContinuationPersistence.State
  pending : List Pending
  activations : List ActivationEvent

def initialState (graph : Graph) : State := ⟨ContinuationPersistence.initialState graph, [], []⟩

def stepWithOrigins (partition : List HCClass) (index : Nat) (selected owners : List Vertex)
    (state : State) : State :=
  let before := state.base.base.graph
  let rule := compress_ruleWithOccurrences partition selected before
  let production := compress_nodesWithPartitionCarrying partition selected before state.base.base.occurrences
  let pool := state.base.base.pending.map (fun p => p.rewriteCurrentEndpoints selected rule.1.CENTER) ++
    RepresentedUseTransport.extractAt selected before production.1 rule.1.CENTER
  let intro := pool.filterMap (justifiedActivation? production.1)
  let elimPool := state.pending.map (fun p => p.transport selected rule.1.CENTER) ++
    (extractAt selected before production.1 rule.1.CENTER).filter (fun p => owners.contains p.use.owner)
  let elim := elimPool.filterMap (activate? production.1)
  let added := ((intro.map fun a => a.pending.use.record) ++
    (elim.map fun a => a.pending.use.record)).eraseDups
  let afterActivation := dlds production.1.NODES production.1.EDGES (production.1.PATHS ++ added)
  let restorations := state.base.live.filterMap
    (ContinuationPersistence.restoration? selected rule.1.CENTER before afterActivation)
  let records := (restorations.map fun r => r.moved.use.record).eraseDups
  let graph := dlds afterActivation.NODES afterActivation.EDGES
    (ContinuationPersistence.appendMissing afterActivation.PATHS records)
  let moved := state.base.live.map (fun old => old.transport selected rule.1.CENTER)
  let retained := moved.filter fun e => graph.PATHS.contains e.use.record && e.use.currentUseB graph
  let newEvidence := intro.map Evidence.ofActivation ++
    elim.map (fun a => Evidence.ofElimination a.pending.use a.sound)
  let introEvents := intro.map fun a => {
    transitionIndex := index, owner := a.pending.use.owner.NUMBER,
    centre := a.pending.use.centre.NUMBER, consumer := a.pending.use.consumer.NUMBER,
    address := a.pending.use.address : ActivationEvent }
  let elimEvents := elim.map fun a => {
    transitionIndex := index, owner := a.pending.use.owner.NUMBER,
    centre := a.pending.use.centre.NUMBER, consumer := a.pending.use.consumer.NUMBER,
    address := a.pending.use.address : ActivationEvent }
  {
    base := {
      base := {
        graph := graph
        occurrences := production.2
        pending := pool.filter (fun p => !added.contains p.use.record)
        activations := state.base.base.activations ++ introEvents }
      live := retained ++ newEvidence
      transports := state.base.transports ++ restorations.map (fun r => {
        transitionIndex := index, owner := r.old.use.owner.NUMBER,
        before := r.old.use.record, after := r.moved.use.record })
      retired := state.base.retired + (moved.length - retained.length) }
    pending := elimPool.filter (fun p => !added.contains p.use.record)
    activations := state.activations ++ elimEvents }

def step (partition : List HCClass) (index : Nat) (selected : List Vertex) (state : State) : State :=
  stepWithOrigins partition index selected selected state

def runSchedule (partition : List HCClass) : Nat → List (List Vertex) → State → State
  | _, [], state => state
  | index, selected :: remaining, state =>
    runSchedule partition (index + 1) remaining (step partition index selected state)

def selectionState (input : Graph) : State :=
  let partition := hcPartition input
  let schedule := (repeated_nodes_vertex_indexedWithPartition partition input).flatten
  runSchedule partition 0 schedule (initialState input)

end EliminationUseTransport

#print axioms EliminationUseTransport.Activation.delivery
