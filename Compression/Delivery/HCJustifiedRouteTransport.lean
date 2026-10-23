import Compression.Delivery.HCCompleteEliminationContinuation
import Compression.Delivery.HCCompleteIntroductionContinuation












open Semantic FlowSpec OccurrenceAwareReplay CandidateMixedReplay
open RoutedEliminationLocal (RepresentedIntroductionUse)

namespace HCJustifiedRouteTransport

inductive Use where
  | introduction (use : RepresentedIntroductionUse)
  | elimination (use : HCEliminationUse.RepresentedEliminationUse)
  | completeElimination (use : HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation)
  | completeIntroduction (use : HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse)

namespace Use
def owner : Use → Vertex
  | .introduction u => u.owner
  | .elimination u => u.owner
  | .completeElimination u => u.producer.owner
  | .completeIntroduction u => u.producer.owner
def consumer : Use → Vertex
  | .introduction u => u.consumer
  | .elimination u => u.consumer
  | .completeElimination u => u.producer.consumer
  | .completeIntroduction u => u.producer.consumer
def traceStart : Use → Vertex
  | .introduction u => u.consumer
  | .elimination u => u.traceStart
  | .completeElimination u => u.traceStart
  | .completeIntroduction u => u.traceStart
def centre : Use → Vertex
  | .introduction u => u.centre
  | .elimination u => u.centre
  | .completeElimination u => u.producer.centre
  | .completeIntroduction u => u.producer.centre
def tail : Use → List Nat
  | .introduction u => u.tail
  | .elimination u => u.tail
  | .completeElimination u => u.tail
  | .completeIntroduction u => u.tail
def address (u : Use) : List Nat := u.owner.NUMBER :: u.tail
def logicalOutput : Use → FlowPair
  | .introduction u => u.logicalOutput
  | .elimination u => u.logicalOutput
  | .completeElimination u => u.logicalOutput
  | .completeIntroduction u => u.logicalOutput


def sourceOutput : Use → FlowPair
  | .introduction u => u.logicalOutput
  | .elimination u => u.logicalOutput
  | .completeElimination u => u.producer.logicalOutput
  | .completeIntroduction u => u.sourceOutput
theorem source_prefix (u : Use) : colourPathPrefixB u.sourceOutput.2 u.address = true := by
  cases u with
  | introduction u => exact RoutedEliminationLocal.prefix_self _
  | elimination u => exact RoutedEliminationLocal.prefix_self _
  | completeElimination u => exact u.matchedPrefix
  | completeIntroduction u => exact u.matchedPrefix
theorem source_dependency (u : Use) : u.sourceOutput.1 = u.logicalOutput.1 := by cases u <;> rfl
theorem output_eq (u : Use) : u.logicalOutput = (u.logicalOutput.1, u.address) := by
  cases u <;> rfl
def record (u : Use) : Ancestral := path u.traceStart u.centre u.address
def currentUseB (graph : Graph) (u : Use) : Bool :=
  graph.EDGES.any fun e => e.START == u.centre && e.END == u.consumer &&
    e.COLOUR == u.owner.NUMBER
def rewriteCurrentEndpoints (selected : List Vertex) (replacement : Vertex) : Use → Use
  | .introduction u => .introduction (u.rewriteCurrentEndpoints selected replacement)
  | .elimination u => .elimination (u.rewriteCurrentEndpoints selected replacement)
  | .completeElimination u => .completeElimination (u.rewriteCurrentEndpoints selected replacement)
  | .completeIntroduction u => .completeIntroduction (u.rewriteCurrentEndpoints selected replacement)
@[simp] theorem owner_rewrite (selected replacement u) :
    (rewriteCurrentEndpoints selected replacement u).owner = u.owner := by cases u <;> rfl
@[simp] theorem tail_rewrite (selected replacement u) :
    (rewriteCurrentEndpoints selected replacement u).tail = u.tail := by cases u <;> rfl
@[simp] theorem output_rewrite (selected replacement u) :
    (rewriteCurrentEndpoints selected replacement u).logicalOutput = u.logicalOutput := by
  cases u <;> rfl
theorem record_rewrite (selected replacement u) :
    (rewriteCurrentEndpoints selected replacement u).record =
      collapse.rewrite_path_endpoints selected replacement u.record := by
  cases u <;> simp [rewriteCurrentEndpoints, record, traceStart, centre, address,
    owner, tail, RepresentedIntroductionUse.rewriteCurrentEndpoints,
    HCEliminationUse.RepresentedEliminationUse.rewriteCurrentEndpoints,
    HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.rewriteCurrentEndpoints,
    HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.rewriteCurrentEndpoints,
    collapse.rewrite_path_endpoints]
end Use

inductive Justified : Use -> Prop
  | activation {graph : Graph} (evidence : RepresentedUseTransport.JustifiedActivation graph) :
      Justified (.introduction evidence.pending.use)
  | elimination {selected : List Vertex} {before graph : Graph}
      {use : HCEliminationUse.RepresentedEliminationUse}
      (sound : use.Sound selected before graph (RepresentedUseTransport.premiseFlow graph)) :
      Justified (.elimination use)
  | completeElimination {selected : List Vertex} {before graph : Graph}
      {use : HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation}
      (sound : use.producer.Sound selected before graph (RepresentedUseTransport.premiseFlow graph)) :
      Justified (.completeElimination use)
  | completeIntroduction {selected : List Vertex} {before graph : Graph}
      {use : HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse}
      (sound : use.producer.Sound selected before graph (RepresentedUseTransport.premiseFlow graph))
      (separate : residualOccursB
        (candidateEliminationOutputs graph use.producer.centre (RepresentedUseTransport.premiseFlow graph))
        use.producer.address = false)
      (present : use.record ∈ graph.PATHS) :
      Justified (.completeIntroduction use)
  | transport (selected : List Vertex) (replacement : Vertex) {use : Use} :
      Justified use -> Justified (use.rewriteCurrentEndpoints selected replacement)

structure Evidence where
  use : Use
  justified : Justified use

def Evidence.ofActivation {graph : Graph}
    (activation : RepresentedUseTransport.JustifiedActivation graph) : Evidence :=
  ⟨.introduction activation.pending.use, .activation activation⟩

def Evidence.ofElimination {selected : List Vertex} {before graph : Graph}
    (use : HCEliminationUse.RepresentedEliminationUse)
    (sound : use.Sound selected before graph (RepresentedUseTransport.premiseFlow graph)) : Evidence :=
  ⟨.elimination use, .elimination sound⟩

def Evidence.transport (selected : List Vertex) (replacement : Vertex)
    (evidence : Evidence) : Evidence :=
  ⟨evidence.use.rewriteCurrentEndpoints selected replacement,
    .transport selected replacement evidence.justified⟩

def Evidence.ofCompleteElimination {selected : List Vertex} {before graph : Graph}
    (use : HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation)
    (sound : use.producer.Sound selected before graph (RepresentedUseTransport.premiseFlow graph)) : Evidence :=
  ⟨.completeElimination use, .completeElimination sound⟩

def Evidence.ofCompleteIntroduction {selected : List Vertex} {before graph : Graph}
    (use : HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse)
    (sound : use.producer.Sound selected before graph (RepresentedUseTransport.premiseFlow graph))
    (separate : residualOccursB
      (candidateEliminationOutputs graph use.producer.centre (RepresentedUseTransport.premiseFlow graph))
      use.producer.address = false) (present : use.record ∈ graph.PATHS) : Evidence :=
  ⟨.completeIntroduction use, .completeIntroduction sound separate present⟩

def IntroductionSameProvenance (old current : RepresentedIntroductionUse) : Prop :=
  current.owner = old.owner ∧
  current.originalPremise = old.originalPremise ∧
  current.originalConsumer = old.originalConsumer ∧
  current.originalIncoming = old.originalIncoming ∧
  current.originalUse = old.originalUse ∧
  current.incomingColour = old.incomingColour ∧
  current.antecedent = old.antecedent ∧
  current.consequent = old.consequent ∧
  current.premiseDependency = old.premiseDependency ∧
  current.tail = old.tail

def EliminationOrigin (u : HCEliminationUse.RepresentedEliminationUse) :
    HCEliminationUse.RepresentedEliminationUse :=
  { u with
    major := u.originalMajor
    minor := u.originalMinor
    consumer := u.originalConsumer
    traceStart := u.originalTraceStart
    centre := u.owner }

def SameProvenance : Use → Use → Prop
  | .introduction old, .introduction current => IntroductionSameProvenance old current
  | .elimination old, .elimination current => EliminationOrigin current = EliminationOrigin old
  | .completeElimination old, .completeElimination current =>
      EliminationOrigin current.producer = EliminationOrigin old.producer ∧ current.tail = old.tail
  | .completeIntroduction old, .completeIntroduction current =>
      IntroductionSameProvenance old.producer current.producer ∧ current.tail = old.tail
  | _, _ => False

theorem transport_preserves_provenance (selected : List Vertex)
    (replacement : Vertex) (evidence : Evidence) :
    SameProvenance evidence.use (evidence.transport selected replacement).use := by
  cases evidence with
  | mk use proof => cases use <;>
      simp [SameProvenance, IntroductionSameProvenance, EliminationOrigin,
        Evidence.transport, Use.rewriteCurrentEndpoints,
        RepresentedIntroductionUse.rewriteCurrentEndpoints,
        HCEliminationUse.RepresentedEliminationUse.rewriteCurrentEndpoints,
        HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.rewriteCurrentEndpoints]
        <;> simp [HCCompleteIntroductionContinuation.CompleteRepresentedIntroductionUse.rewriteCurrentEndpoints,
          RepresentedIntroductionUse.rewriteCurrentEndpoints, IntroductionSameProvenance]

theorem transport_is_actual_endpoint_map (selected : List Vertex)
    (replacement : Vertex) (evidence : Evidence) :
    (evidence.transport selected replacement).use.record =
      collapse.rewrite_path_endpoints selected replacement evidence.use.record := by
  exact Use.record_rewrite
    selected replacement evidence.use

theorem transport_preserves_complete_route (selected : List Vertex)
    (replacement : Vertex) (evidence : Evidence) :
    (evidence.transport selected replacement).use.address = evidence.use.address ∧
    (evidence.transport selected replacement).use.logicalOutput =
      evidence.use.logicalOutput := by
  simp [Evidence.transport, Use.address]

theorem distinct_owners_remain_separate (selected : List Vertex)
    (replacement : Vertex) (left right : Evidence)
    (different : left.use.owner.NUMBER ≠ right.use.owner.NUMBER) :
    (left.transport selected replacement).use.record ≠
      (right.transport selected replacement).use.record := by
  intro equal
  have colours := congrArg Ancestral.COLOURS equal
  have heads := congrArg List.head? colours
  simp [Evidence.transport, Use.record, Use.address]
    at heads
  exact different heads

theorem transport_preserves_consumption (selected : List Vertex)
    (replacement : Vertex) (evidence : Evidence) (stored : List Formula) :
    let moved := (evidence.transport selected replacement).use
    consume (edge moved.centre moved.consumer moved.owner.NUMBER stored)
      moved.address = some evidence.use.tail := by
  simp [consume, Use.address, Evidence.transport]

theorem unrelated_owner_cannot_consume (selected : List Vertex)
    (replacement : Vertex) (evidence : Evidence) (other : Nat)
    (stored : List Formula) (different : other ≠ evidence.use.owner.NUMBER) :
    let moved := (evidence.transport selected replacement).use
    consume (edge moved.centre moved.consumer other stored) moved.address = none := by
  simp [consume, Use.address, Evidence.transport, different]

def logical (graph : Graph) (use : Use) : List FlowPair :=
  derivedOutputs graph use.centre (RepresentedUseTransport.premiseFlow graph)

structure Ready (before after : Graph) (old moved : Evidence) : Prop where
  oldPresent : old.use.record ∈ before.PATHS
  absent : moved.use.record ∉ after.PATHS
  centrePresent : moved.use.centre ∈ after.NODES
  consumerPresent : moved.use.consumer ∈ after.NODES
  existingAncestry : (ancestorsInto after moved.use.centre).isEmpty = false
  survivingUse : moved.use.currentUseB after = true
  produced : moved.use.sourceOutput ∈ logical after moved.use
  exclusive : ∀ source ∈ logical after moved.use,
    colourPathPrefixB source.2 moved.use.address = true ->
      source = moved.use.sourceOutput

def ready? (before after : Graph) (old moved : Evidence) :
    Option (PLift (Ready before after old moved)) :=
  if hOld : old.use.record ∈ before.PATHS then
    if hAbsent : moved.use.record ∉ after.PATHS then
      if hCentre : moved.use.centre ∈ after.NODES then
        if hConsumer : moved.use.consumer ∈ after.NODES then
          if hAncestry : (ancestorsInto after moved.use.centre).isEmpty = false then
            if hUse : moved.use.currentUseB after = true then
              if hLogical : moved.use.sourceOutput ∈ logical after moved.use then
                if hExclusive : ∀ source ∈ logical after moved.use,
                    colourPathPrefixB source.2 moved.use.address = true ->
                      source = moved.use.sourceOutput then
                  some ⟨⟨hOld, hAbsent, hCentre, hConsumer, hAncestry,
                    hUse, hLogical, hExclusive⟩⟩
                else none
              else none
            else none
          else none
        else none
      else none
    else none
  else none

theorem ready_delivery {before after : Graph} {old moved : Evidence}
    (ready : Ready before after old moved) :
    moved.use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto after moved.use.centre ++ [moved.use.record])
      (logical after moved.use) := by
  have delivered := HCCompleteEliminationContinuation.CompleteRepresentedUseContinuation.prefix_delivery
    (ancestorsInto after moved.use.centre) moved.use.traceStart moved.use.centre
    moved.use.sourceOutput moved.use.address (logical after moved.use)
    ready.produced (Use.source_prefix moved.use)
  simpa only [Use.record, Use.source_dependency, ← Use.output_eq] using delivered

theorem ready_actual_use_consumption {before after : Graph} {old moved : Evidence}
    (ready : Ready before after old moved) :
    ∃ outgoing ∈ after.EDGES,
      outgoing.START = moved.use.centre ∧ outgoing.END = moved.use.consumer ∧
      outgoing.COLOUR = moved.use.owner.NUMBER ∧
      consume outgoing moved.use.address = some moved.use.tail := by
  have edgePresent := ready.survivingUse
  simp only [Use.currentUseB, List.any_eq_true,
    Bool.and_eq_true, beq_iff_eq] at edgePresent
  obtain ⟨outgoing, member, ⟨hStart, hEnd⟩, hColour⟩ := edgePresent
  refine ⟨outgoing, member, hStart, hEnd, hColour, ?_⟩
  simp [consume, Use.address, hColour]

theorem ready_no_unrelated_reseed_output {before after : Graph}
    {old moved : Evidence} (ready : Ready before after old moved)
    {route : FlowPair}
    (new : route ∈ reseedDerivedRoutes
      (ancestorsInto after moved.use.centre ++ [moved.use.record])
      (logical after moved.use))
    (notOld : route ∉ reseedDerivedRoutes
      (ancestorsInto after moved.use.centre) (logical after moved.use)) :
    route = moved.use.logicalOutput := by
  obtain ⟨source, member, hPrefix, routeEq⟩ :=
    RoutedEliminationLocal.new_reseed_route_has_matching_source _ _ _ _ new notOld
  have same := ready.exclusive source member hPrefix
  rw [same] at routeEq
  calc
    route = (moved.use.logicalOutput.1, moved.use.address) := by
      simpa only [Use.source_dependency] using routeEq
    _ = moved.use.logicalOutput := (Use.output_eq moved.use).symm



theorem derived_kernel_unchanged_by_transport (graph : Graph) (vertex : Vertex)
    (flow : Vertex -> List FlowPair) (record : Ancestral) :
    derivedOutputs (dlds graph.NODES graph.EDGES (graph.PATHS ++ [record]))
      vertex flow = derivedOutputs graph vertex flow := by
  rfl

theorem other_centre_ancestry_unchanged (graph : Graph) (vertex : Vertex)
    (record : Ancestral) (different : record.END ≠ vertex) :
    ancestorsInto (dlds graph.NODES graph.EDGES (graph.PATHS ++ [record])) vertex =
      ancestorsInto graph vertex := by
  simp [ancestorsInto, List.filter_append, different]




theorem ready_local_step {before after : Graph} {old moved : Evidence}
    (ready : Ready before after old moved)
    (notSource : moved.use.centre.HYPOTHESIS = false)
    (hasPredecessor : (predsOf after moved.use.centre).isEmpty = false) :
    occurrenceStep
      (dlds after.NODES after.EDGES (after.PATHS ++ [moved.use.record]))
      moved.use.centre (RepresentedUseTransport.premiseFlow after) =
      .ok (reseedDerivedRoutes
        (ancestorsInto after moved.use.centre ++ [moved.use.record])
        (logical after moved.use)) := by
  have nonempty : (logical after moved.use).isEmpty = false := by
    rw [List.isEmpty_eq_false_iff]
    exact List.ne_nil_of_mem ready.produced
  have matched : (reseedDerivedRoutes
      (ancestorsInto after moved.use.centre ++ [moved.use.record])
      (logical after moved.use)).isEmpty = false := by
    rw [List.isEmpty_eq_false_iff]
    exact List.ne_nil_of_mem (ready_delivery ready)
  have ancestry :
      ancestorsInto
        (dlds after.NODES after.EDGES (after.PATHS ++ [moved.use.record]))
        moved.use.centre =
      ancestorsInto after moved.use.centre ++ [moved.use.record] := by
    simp [ancestorsInto, List.filter_append, Use.record]
  have hasAncestor : (ancestorsInto
      (dlds after.NODES after.EDGES (after.PATHS ++ [moved.use.record]))
      moved.use.centre).isEmpty = false := by
    rw [ancestry]
    simp
  have result := occurrenceStep_reseeds_prefix_matched_derived
    (dlds after.NODES after.EDGES (after.PATHS ++ [moved.use.record]))
    moved.use.centre (RepresentedUseTransport.premiseFlow after)
    hasPredecessor hasAncestor notSource nonempty
    (by simpa [ancestry, logical] using matched)
  simpa [ancestry, logical] using result

end HCJustifiedRouteTransport

#print axioms HCJustifiedRouteTransport.transport_preserves_provenance
#print axioms HCJustifiedRouteTransport.transport_is_actual_endpoint_map
#print axioms HCJustifiedRouteTransport.transport_preserves_complete_route
#print axioms HCJustifiedRouteTransport.distinct_owners_remain_separate
#print axioms HCJustifiedRouteTransport.transport_preserves_consumption
#print axioms HCJustifiedRouteTransport.unrelated_owner_cannot_consume
#print axioms HCJustifiedRouteTransport.ready_delivery
#print axioms HCJustifiedRouteTransport.ready_actual_use_consumption
#print axioms HCJustifiedRouteTransport.ready_no_unrelated_reseed_output
#print axioms HCJustifiedRouteTransport.derived_kernel_unchanged_by_transport
#print axioms HCJustifiedRouteTransport.other_centre_ancestry_unchanged
#print axioms HCJustifiedRouteTransport.ready_local_step
