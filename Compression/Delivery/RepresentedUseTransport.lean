import Compression.Delivery.RoutedEliminationLocal












open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay

namespace RepresentedUseTransport

abbrev RepresentedUse := RoutedEliminationLocal.RepresentedIntroductionUse

def premiseFlow (graph : Graph) : Vertex -> List FlowPair := fun premise =>
  match occurrenceReplayAt graph (stdFuel graph) premise with
  | .ok routes => routes
  | .error _ => []

structure PendingUse where
  originSelected : List Vertex
  originBefore : Graph
  use : RepresentedUse

namespace PendingUse

def rewriteCurrentEndpoints (selected : List Vertex) (replacement : Vertex)
    (pending : PendingUse) : PendingUse :=
  { pending with
    use := pending.use.rewriteCurrentEndpoints selected replacement }

end PendingUse

structure JustifiedActivation (graph : Graph) where
  pending : PendingUse
  ancestryNonempty :
    (ancestorsInto graph pending.use.centre).isEmpty = false
  sound : pending.use.Sound pending.originSelected pending.originBefore graph
    (premiseFlow graph)
  separate : residualOccursB
    (selectionEliminationOutputs graph pending.use.centre (premiseFlow graph))
    pending.use.address = false
  recordAbsent : graph.PATHS.contains pending.use.record = false

namespace JustifiedActivation

theorem delivery {graph : Graph} (activation : JustifiedActivation graph) :
    activation.pending.use.logicalOutput ∈
        introductionSelections graph activation.pending.use.centre
          (premiseFlow graph) ∧
      activation.pending.use.logicalOutput ∈
        derivedOutputs graph activation.pending.use.centre
          (premiseFlow graph) ∧
      activation.pending.use.logicalOutput ∈ reseedDerivedRoutes
        (ancestorsInto graph activation.pending.use.centre ++
          [activation.pending.use.record])
        (derivedOutputs graph activation.pending.use.centre
          (premiseFlow graph)) ∧
      ∀ stored, consume
        (edge activation.pending.use.centre
          activation.pending.use.consumer
          activation.pending.use.owner.NUMBER stored)
        activation.pending.use.address = some activation.pending.use.tail := by
  exact RoutedEliminationLocal.producer_to_delivery activation.sound
    activation.separate

end JustifiedActivation

def justifiedActivation? (graph : Graph) (pending : PendingUse) :
    Option (JustifiedActivation graph) :=
  if ancestry : (ancestorsInto graph pending.use.centre).isEmpty = false then
    if supported : pending.use.supportedB pending.originSelected
        pending.originBefore graph (premiseFlow graph) = true then
      if separate : residualOccursB
          (selectionEliminationOutputs graph pending.use.centre
            (premiseFlow graph)) pending.use.address = false then
        if absent : graph.PATHS.contains pending.use.record = false then
          some {
            pending := pending
            ancestryNonempty := ancestry
            sound := RoutedEliminationLocal.RepresentedIntroductionUse.sound_of_supportedB
              supported
            separate := separate
            recordAbsent := absent }
        else none
      else none
    else none
  else none

def extractAt (selected : List Vertex) (before after : Graph)
    (centre : Vertex) : List PendingUse :=
  selected.filterMap fun owner =>
    match RoutedEliminationLocal.producerDerivedUse? selected before after centre owner
        (premiseFlow after) with
    | none => none
    | some evidence => some {
        originSelected := selected
        originBefore := before
        use := evidence.1 }

structure ActivationEvent where
  transitionIndex : Nat
  owner : Nat
  centre : Nat
  consumer : Nat
  address : List Nat
deriving Repr, BEq

structure SelectionState where
  graph : Graph
  occurrences : List Ancestral
  pending : List PendingUse
  activations : List ActivationEvent

def initialState (graph : Graph) : SelectionState := {
  graph := graph
  occurrences := []
  pending := []
  activations := [] }

private def eventFor? (transitionIndex : Nat)
    (activations : List (JustifiedActivation graph))
    (record : Ancestral) : Option ActivationEvent :=
  (activations.find? fun activation =>
    activation.pending.use.record == record).map fun activation => {
      transitionIndex := transitionIndex
      owner := activation.pending.use.owner.NUMBER
      centre := activation.pending.use.centre.NUMBER
      consumer := activation.pending.use.consumer.NUMBER
      address := activation.pending.use.address }

def selectionStep (partition : List HCClass) (transitionIndex : Nat)
    (selected : List Vertex) (state : SelectionState) : SelectionState :=
  let rule := compress_ruleWithOccurrences partition selected state.graph
  let base := compress_nodesWithPartitionCarrying partition selected state.graph
    state.occurrences
  let transported := state.pending.map fun pending =>
    pending.rewriteCurrentEndpoints selected rule.1.CENTER
  let originated := extractAt selected state.graph base.1 rule.1.CENTER
  let pool := transported ++ originated
  let justified := pool.filterMap (justifiedActivation? base.1)
  let added := (justified.map fun activation =>
    activation.pending.use.record).eraseDups
  let remaining := pool.filter fun pending =>
    !added.contains pending.use.record
  let events := added.filterMap (eventFor? transitionIndex justified)
  {
    graph := dlds base.1.NODES base.1.EDGES (base.1.PATHS ++ added)
    occurrences := base.2
    pending := remaining
    activations := state.activations ++ events }

def runSchedule (partition : List HCClass) :
    Nat -> List (List Vertex) -> SelectionState -> SelectionState
  | _, [], state => state
  | transitionIndex, selected :: remaining, state =>
      runSchedule partition (transitionIndex + 1) remaining
        (selectionStep partition transitionIndex selected state)

def selectionState (input : Graph) : SelectionState :=
  let partition := hcPartition input
  let schedule :=
    (repeated_nodes_vertex_indexedWithPartition partition input).flatten
  runSchedule partition 0 schedule (initialState input)



def compress_nodes_graph_withSupportedUses (input : Graph) : Graph :=
  (selectionState input).graph

end RepresentedUseTransport

#print axioms RepresentedUseTransport.JustifiedActivation.delivery
