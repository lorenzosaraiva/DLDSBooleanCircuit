import Compression.Delivery.HCCompleteEliminationContinuation






open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay
open RoutedEliminationLocal (RepresentedIntroductionUse)

namespace HCCompleteIntroductionContinuation

theorem two_incoming_generated (consumer : Vertex) (first second outgoing : Deduction)
    (incoming : Deduction) (member : incoming ∈ [first, second]) :
    path outgoing.END incoming.START [incoming.COLOUR, consumer.NUMBER] ∈
      pre_collapse.indirect consumer.NUMBER false [first, second] [outgoing] [] := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;>
    simp [pre_collapse.indirect, pre_collapse.indirect.create]

structure Boundary (before : Graph) (selected : List Vertex)
    (producer : RepresentedIntroductionUse) (remote : Vertex) : Prop where
  emptySourceTail : producer.tail = []
  sameConsumer : producer.consumer = producer.originalConsumer
  selectedConsumer : producer.consumer ∈ selected
  currentConsumer : producer.consumer ∈ before.NODES
  uncollapsed : producer.consumer.COLLAPSED = false
  derived : producer.consumer.HYPOTHESIS = false
  representedPremises : ∀ incoming ∈ get_rule.incoming producer.consumer before,
    incoming.START.COLLAPSED = true ∧ incoming.COLOUR ≠ 0
  incidence : ∃ first second incoming outgoing,
    get_rule.incoming producer.consumer before = [first, second] ∧
    incoming ∈ [first, second] ∧
    incoming.START = producer.centre ∧ incoming.END = producer.consumer ∧
    incoming.COLOUR = producer.owner.NUMBER ∧
    get_rule.outgoing producer.consumer before = [outgoing] ∧
    outgoing.START = producer.consumer ∧ outgoing.END = remote ∧
    get_rule.direct producer.consumer before = []

theorem Boundary.generated {before selected producer remote}
    (b : Boundary before selected producer remote) :
    path remote producer.centre [producer.owner.NUMBER, producer.originalConsumer.NUMBER] ∈
      (pre_collapse (get_rule producer.consumer before)).INDIRECT := by
  obtain ⟨first, second, incoming, outgoing, ins, member, start, _, colour,
    outs, _, finish, direct⟩ := b.incidence
  have generated := two_incoming_generated producer.consumer first second outgoing incoming member
  have number := congrArg Vertex.NUMBER b.sameConsumer
  simpa only [pre_collapse, get_rule, b.uncollapsed, b.derived, ins, outs, direct,
    start, colour, finish, number] using generated

inductive Origin : RepresentedIntroductionUse → Vertex → List Nat → Prop
  | consumerBoundary {producer remote} (before : Graph) (selected : List Vertex)
      (boundary : Boundary before selected producer remote) :
      Origin producer remote [producer.originalConsumer.NUMBER]
  | transport (selected : List Vertex) (replacement : Vertex) {producer remote tail}
      (old : Origin producer remote tail) :
      Origin (producer.rewriteCurrentEndpoints selected replacement)
        (if remote ∈ selected then replacement else remote) tail

theorem Origin.empty_source_tail {producer remote tail}
    (origin : Origin producer remote tail) : producer.tail = [] := by
  induction origin with
  | consumerBoundary _ _ boundary => exact boundary.emptySourceTail
  | transport _ _ _ unchanged => exact unchanged

structure CompleteRepresentedIntroductionUse where
  producer : RepresentedIntroductionUse
  traceStart : Vertex
  tail : List Nat
  origin : Origin producer traceStart tail
  matchedPrefix : colourPathPrefixB producer.address (producer.owner.NUMBER :: tail) = true

namespace CompleteRepresentedIntroductionUse
def address (c : CompleteRepresentedIntroductionUse) := c.producer.owner.NUMBER :: c.tail
def sourceOutput (c : CompleteRepresentedIntroductionUse) : FlowPair := c.producer.logicalOutput
def logicalOutput (c : CompleteRepresentedIntroductionUse) : FlowPair := (c.sourceOutput.1, c.address)
def record (c : CompleteRepresentedIntroductionUse) := path c.traceStart c.producer.centre c.address
def rewriteCurrentEndpoints (selected : List Vertex) (replacement : Vertex)
    (c : CompleteRepresentedIntroductionUse) : CompleteRepresentedIntroductionUse :=
  { producer := c.producer.rewriteCurrentEndpoints selected replacement
    traceStart := if c.traceStart ∈ selected then replacement else c.traceStart
    tail := c.tail
    origin := .transport selected replacement c.origin
    matchedPrefix := c.matchedPrefix }

theorem source_correctness {selected before after flow} (c : CompleteRepresentedIntroductionUse)
    (sound : c.producer.Sound selected before after flow)
    (separate : residualOccursB (selectionEliminationOutputs after c.producer.centre flow)
      c.producer.address = false) :
    c.sourceOutput ∈ derivedOutputs after c.producer.centre flow :=
  RoutedEliminationLocal.logicalOutput_mem_derivedOutputs sound separate

theorem source_provenance {selected before after flow} (c : CompleteRepresentedIntroductionUse)
    (sound : c.producer.Sound selected before after flow) :
    c.producer.owner ∈ before.NODES ∧
    get_rule.incoming c.producer.owner before = [c.producer.originalIncoming] ∧
    c.producer.originalIncoming.START = c.producer.originalPremise ∧
    c.producer.originalIncoming.END = c.producer.owner ∧
    get_rule.outgoing c.producer.owner before = [c.producer.originalUse] ∧
    c.producer.originalUse.START = c.producer.owner ∧
    c.producer.originalUse.END = c.producer.originalConsumer :=
  ⟨sound.ownerInBefore, sound.originalIncomingExact, sound.originalIncomingStart,
    sound.originalIncomingEnd, sound.originalUseExact, sound.originalUseStart, sound.originalUseEnd⟩

theorem source_prefix (c : CompleteRepresentedIntroductionUse) :
    colourPathPrefixB c.sourceOutput.2 c.address = true := c.matchedPrefix

theorem source_address (c : CompleteRepresentedIntroductionUse) :
    c.sourceOutput.2 = [c.producer.owner.NUMBER] := by
  simp [sourceOutput, RepresentedIntroductionUse.logicalOutput,
    RepresentedIntroductionUse.address, c.origin.empty_source_tail]
theorem dependency_preserved (c : CompleteRepresentedIntroductionUse) :
    c.logicalOutput.1 = c.sourceOutput.1 := rfl
theorem no_shortening (c : CompleteRepresentedIntroductionUse) (nonempty : c.tail ≠ []) :
    c.address ≠ [c.producer.owner.NUMBER] := by
  intro equal; exact nonempty (List.cons.inj equal).2
theorem exact_consumption (c : CompleteRepresentedIntroductionUse) (stored : List Formula) :
    consume (edge c.producer.centre c.producer.consumer c.producer.owner.NUMBER stored)
      c.address = some c.tail := by simp [consume, address]
theorem different_owners (left right : CompleteRepresentedIntroductionUse)
    (different : left.producer.owner.NUMBER ≠ right.producer.owner.NUMBER) :
    left.address ≠ right.address := by
  intro equal; exact different (List.cons.inj equal).1
theorem other_owner_cannot_consume (c : CompleteRepresentedIntroductionUse) (other : Nat)
    (stored : List Formula) (different : other ≠ c.producer.owner.NUMBER) :
    consume (edge c.producer.centre c.producer.consumer other stored) c.address = none := by
  simp [consume, address, different]
theorem record_rewrite (selected replacement c) :
    (rewriteCurrentEndpoints selected replacement c).record =
      collapse.rewrite_path_endpoints selected replacement c.record := by
  simp [rewriteCurrentEndpoints, record, address,
    RepresentedIntroductionUse.rewriteCurrentEndpoints, collapse.rewrite_path_endpoints]

theorem producer_to_complete_delivery {selected before after flow}
    (c : CompleteRepresentedIntroductionUse)
    (sound : c.producer.Sound selected before after flow)
    (separate : residualOccursB (selectionEliminationOutputs after c.producer.centre flow)
      c.producer.address = false) :
    c.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto after c.producer.centre ++ [c.record])
      (derivedOutputs after c.producer.centre flow) :=
  HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.prefix_delivery
    _ _ _ _ _ _ (c.source_correctness sound separate) c.source_prefix
end CompleteRepresentedIntroductionUse





def atConsumer? (before : Graph) (selected : List Vertex) (producer : RepresentedIntroductionUse) :
    Option CompleteRepresentedIntroductionUse := do
  let [first, second] := get_rule.incoming producer.consumer before | none
  let [outgoing] := get_rule.outgoing producer.consumer before | none
  let incoming ← [first, second].find? fun e =>
    e.START == producer.centre && e.END == producer.consumer && e.COLOUR == producer.owner.NUMBER
  if supported : producer.tail = [] ∧ producer.consumer = producer.originalConsumer ∧
      producer.consumer ∈ selected ∧ producer.consumer ∈ before.NODES ∧
      producer.consumer.COLLAPSED = false ∧ producer.consumer.HYPOTHESIS = false ∧
      get_rule.incoming producer.consumer before = [first, second] ∧
      incoming ∈ [first, second] ∧ incoming.START = producer.centre ∧
      incoming.END = producer.consumer ∧ incoming.COLOUR = producer.owner.NUMBER ∧
      get_rule.outgoing producer.consumer before = [outgoing] ∧
      outgoing.START = producer.consumer ∧ get_rule.direct producer.consumer before = [] ∧
      (∀ e ∈ get_rule.incoming producer.consumer before,
        e.START.COLLAPSED = true ∧ e.COLOUR ≠ 0) then
    let ⟨empty, same, chosen, current, plain, derived, ins, member, start, finish, colour,
      outs, outStart, direct, represented⟩ := supported
    let boundary : Boundary before selected producer outgoing.END :=
      ⟨empty, same, chosen, current, plain, derived, represented,
        ⟨first, second, incoming, outgoing, ins, member, start, finish, colour,
          outs, outStart, rfl, direct⟩⟩
    some ⟨producer, outgoing.END, [producer.originalConsumer.NUMBER],
      .consumerBoundary before selected boundary,
      by simp [RepresentedIntroductionUse.address, empty, colourPathPrefixB]⟩
  else none

end HCCompleteIntroductionContinuation

#print axioms HCCompleteIntroductionContinuation.two_incoming_generated
#print axioms HCCompleteIntroductionContinuation.Boundary.generated
#print axioms HCCompleteIntroductionContinuation.Origin.empty_source_tail
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.source_correctness
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.source_provenance
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.source_prefix
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.source_address
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.dependency_preserved
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.no_shortening
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.exact_consumption
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.different_owners
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.other_owner_cannot_consume
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.record_rewrite
#print axioms HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.producer_to_complete_delivery
