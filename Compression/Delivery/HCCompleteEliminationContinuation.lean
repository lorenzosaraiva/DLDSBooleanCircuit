import Compression.Delivery.HCEliminationUse







open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay
open HCEliminationUse (RepresentedEliminationUse)

namespace HCCompleteEliminationContinuation

inductive Origin : RepresentedEliminationUse → Vertex → List Nat → Prop
  | direct {producer : RepresentedEliminationUse} (remaining : List (List Vertex))
      (empty : producer.tail = [])
      (unrepresented : producer.consumer = producer.originalConsumer)
      (uncollapsed : producer.consumer.COLLAPSED = false)
      (unscheduled : producer.consumer ∉ remaining.flatten) :
      Origin producer producer.traceStart []
  | captured {producer : RepresentedEliminationUse} (nonempty : producer.tail ≠ []) :
      Origin producer producer.traceStart producer.tail
  | consumerBoundary {producer : RepresentedEliminationUse} (before : Graph) (selected : List Vertex)
      (incoming outgoing : Deduction)
      (empty : producer.tail = [])
      (sameOccurrence : producer.consumer = producer.originalConsumer)
      (selectedConsumer : producer.consumer ∈ selected)
      (currentConsumer : producer.consumer ∈ before.NODES)
      (uncollapsed : producer.consumer.COLLAPSED = false)
      (derived : producer.consumer.HYPOTHESIS = false)
      (incomingExact : get_rule.incoming producer.consumer before = [incoming])
      (incomingStart : incoming.START = producer.centre)
      (incomingEnd : incoming.END = producer.consumer)
      (incomingColour : incoming.COLOUR = producer.owner.NUMBER)
      (outgoingExact : get_rule.outgoing producer.consumer before = [outgoing])
      (outgoingStart : outgoing.START = producer.consumer)
      (noDirect : get_rule.direct producer.consumer before = []) :
      Origin producer outgoing.END [producer.originalConsumer.NUMBER]
  | transport (selected : List Vertex) (replacement : Vertex) {producer start tail}
      (old : Origin producer start tail) :
      Origin (producer.rewriteCurrentEndpoints selected replacement)
        (if start ∈ selected then replacement else start) tail

structure CompleteRepresentedUseContinuation where
  producer : RepresentedEliminationUse
  traceStart : Vertex
  tail : List Nat
  origin : Origin producer traceStart tail
  matchedPrefix : colourPathPrefixB producer.address (producer.owner.NUMBER :: tail) = true

namespace CompleteRepresentedUseContinuation
def address (c : CompleteRepresentedUseContinuation) := c.producer.owner.NUMBER :: c.tail
def logicalOutput (c : CompleteRepresentedUseContinuation) : FlowPair :=
  (c.producer.logicalOutput.1, c.address)
def record (c : CompleteRepresentedUseContinuation) := path c.traceStart c.producer.centre c.address
def rewriteCurrentEndpoints (selected : List Vertex) (replacement : Vertex)
    (c : CompleteRepresentedUseContinuation) : CompleteRepresentedUseContinuation :=
  { producer := c.producer.rewriteCurrentEndpoints selected replacement
    traceStart := if c.traceStart ∈ selected then replacement else c.traceStart
    tail := c.tail
    origin := .transport selected replacement c.origin
    matchedPrefix := c.matchedPrefix }

theorem no_shortening (c : CompleteRepresentedUseContinuation) (nonempty : c.tail ≠ []) :
    c.address ≠ [c.producer.owner.NUMBER] := by
  intro equal
  exact nonempty (List.cons.inj equal).2

theorem exact_consumption (c : CompleteRepresentedUseContinuation) (stored : List Formula) :
    consume (edge c.producer.centre c.producer.consumer c.producer.owner.NUMBER stored)
      c.address = some c.tail := by simp [consume, address]

theorem no_empty_consumption (c : CompleteRepresentedUseContinuation)
    (stored : List Formula) (nonempty : c.tail ≠ []) :
    consume (edge c.producer.centre c.producer.consumer c.producer.owner.NUMBER stored)
      c.address ≠ some [] := by
  rw [c.exact_consumption]
  exact fun equal => nonempty (Option.some.inj equal)

theorem different_owners (left right : CompleteRepresentedUseContinuation)
    (different : left.producer.owner.NUMBER ≠ right.producer.owner.NUMBER) :
    left.address ≠ right.address := by
  intro equal; exact different (List.cons.inj equal).1

theorem record_rewrite (selected replacement c) :
    (rewriteCurrentEndpoints selected replacement c).record =
      collapse.rewrite_path_endpoints selected replacement c.record := by
  simp [rewriteCurrentEndpoints, record, address,
    RepresentedEliminationUse.rewriteCurrentEndpoints, collapse.rewrite_path_endpoints]

theorem prefix_delivery (existing : List Ancestral) (start centre : Vertex)
    (source : FlowPair) (address : List Nat) (derived : List FlowPair)
    (produced : source ∈ derived) (matchedPrefix : colourPathPrefixB source.2 address = true) :
    (source.1, address) ∈ reseedDerivedRoutes
      (existing ++ [path start centre address]) derived := by
  rw [reseedDerivedRoutes, List.mem_eraseDups, List.mem_flatMap]
  refine ⟨path start centre address, by simp, ?_⟩
  rw [List.mem_filterMap]
  exact ⟨source, produced, by simp [matchedPrefix]⟩

theorem producer_to_complete_delivery {selected before after flow}
    (c : CompleteRepresentedUseContinuation)
    (sound : c.producer.Sound selected before after flow) :
    c.producer.logicalOutput ∈ derivedOutputs after c.producer.centre flow ∧
    c.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto after c.producer.centre ++ [c.record])
      (derivedOutputs after c.producer.centre flow) ∧
    ∀ stored, consume (edge c.producer.centre c.producer.consumer c.producer.owner.NUMBER stored)
      c.address = some c.tail := by
  have produced := (RepresentedEliminationUse.producer_to_delivery sound).2.1
  exact ⟨produced, prefix_delivery _ _ _ _ _ _ produced c.matchedPrefix, c.exact_consumption⟩
end CompleteRepresentedUseContinuation



def direct? (remaining : List (List Vertex)) (producer : RepresentedEliminationUse) :
    Option CompleteRepresentedUseContinuation :=
  if empty : producer.tail = [] then
    if same : producer.consumer = producer.originalConsumer then
      if plain : producer.consumer.COLLAPSED = false then
        if unscheduled : producer.consumer ∉ remaining.flatten then
          some ⟨producer, producer.traceStart, [], .direct remaining empty same plain unscheduled,
            by simp [RepresentedEliminationUse.address, empty, colourPathPrefixB]⟩
        else none
      else none
    else none
  else some ⟨producer, producer.traceStart, producer.tail, .captured empty,
    RoutedEliminationLocal.prefix_self _⟩



def atConsumer? (before : Graph) (selected : List Vertex) (producer : RepresentedEliminationUse) :
    Option CompleteRepresentedUseContinuation := do
  let [incoming] := get_rule.incoming producer.consumer before | none
  let [outgoing] := get_rule.outgoing producer.consumer before | none
  if supported : producer.tail = [] ∧ producer.consumer = producer.originalConsumer ∧
      producer.consumer ∈ selected ∧ producer.consumer ∈ before.NODES ∧
      producer.consumer.COLLAPSED = false ∧ producer.consumer.HYPOTHESIS = false ∧
      get_rule.incoming producer.consumer before = [incoming] ∧
      incoming.START = producer.centre ∧ incoming.END = producer.consumer ∧
      incoming.COLOUR = producer.owner.NUMBER ∧
      get_rule.outgoing producer.consumer before = [outgoing] ∧
      outgoing.START = producer.consumer ∧ get_rule.direct producer.consumer before = [] then
    let ⟨empty, same, chosen, current, plain, derived, ins, start, finish, colour, outs, outStart, paths⟩ := supported
    some ⟨producer, outgoing.END, [producer.originalConsumer.NUMBER],
      .consumerBoundary before selected incoming outgoing empty same chosen current plain derived
        ins start finish colour outs outStart paths,
      by simp [RepresentedEliminationUse.address, empty, colourPathPrefixB]⟩
  else none

theorem consumer_boundary_generated_route (producer : RepresentedEliminationUse)
    (incoming outgoing : Deduction)
    (colour : incoming.COLOUR = producer.owner.NUMBER)
    (start : incoming.START = producer.centre) :
    pre_collapse.indirect producer.originalConsumer.NUMBER false [incoming] [outgoing] [] =
      [path outgoing.END producer.centre
        [producer.owner.NUMBER, producer.originalConsumer.NUMBER]] := by
  simp [pre_collapse.indirect, pre_collapse.indirect.create, colour, start]

theorem consumer_boundary_not_singleton (owner consumer : Nat) :
    [owner, consumer] ≠ [owner] ∧
    colourPathPrefixB [owner] [owner, consumer] = true ∧
    consume (edge producer receiver owner stored) [owner, consumer] = some [consumer] := by
  simp [colourPathPrefixB, consume]

end HCCompleteEliminationContinuation

#print axioms HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.no_shortening
#print axioms HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.exact_consumption
#print axioms HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.no_empty_consumption
#print axioms HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.different_owners
#print axioms HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.producer_to_complete_delivery
#print axioms HCCompleteEliminationContinuation.consumer_boundary_generated_route
#print axioms HCCompleteEliminationContinuation.consumer_boundary_not_singleton
