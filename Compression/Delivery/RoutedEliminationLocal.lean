import Compression.Delivery.RoutedEliminationTransport
import Infrastructure.OccurrenceAwareReplay
import Infrastructure.HCSuccessorDeliverySemantics







open Semantic FlowSpec OccurrenceAwareReplay
open MixedRuleReplay HCSuccessorDeliverySemantics

namespace RoutedEliminationLocal

theorem prefix_self : (colours : List Nat) →
    colourPathPrefixB colours colours = true
  | [] => rfl
  | _ :: tail => by simp [colourPathPrefixB, prefix_self tail]

theorem delivered_full_tail (existing : List Ancestral)
    (consumer centre : Vertex) (colour : Nat) (tail : List Nat)
    (dependency : Dep) (derived : List FlowPair)
    (logical : (dependency, colour :: tail) ∈ derived) :
    (dependency, colour :: tail) ∈
      reseedDerivedRoutes
        (existing ++ [path consumer centre (colour :: tail)]) derived := by
  rw [reseedDerivedRoutes, List.mem_eraseDups, List.mem_flatMap]
  refine ⟨path consumer centre (colour :: tail), by simp, ?_⟩
  rw [List.mem_filterMap]
  refine ⟨(dependency, colour :: tail), logical, ?_⟩
  simp [prefix_self]

theorem consume_delivered_full_tail (producer consumer : Vertex)
    (colour : Nat) (tail : List Nat) (stored : List Formula) :
    consume (edge producer consumer colour stored) (colour :: tail) =
      some tail := by
  simp [consume]

theorem consume_other_owner (producer consumer : Vertex)
    (colour other : Nat) (stored : List Formula) (different : colour ≠ other) :
    consume (edge producer consumer colour stored) [other] = none := by
  simp [consume, different]

theorem existing_reseed_survives (existing : List Ancestral)
    (newRecord : Ancestral) (derived : List FlowPair) (route : FlowPair)
    (present : route ∈ reseedDerivedRoutes existing derived) :
    route ∈ reseedDerivedRoutes (existing ++ [newRecord]) derived := by
  rw [reseedDerivedRoutes, List.mem_eraseDups, List.mem_flatMap] at present ⊢
  obtain ⟨ancestor, member, matchRoute⟩ := present
  exact ⟨ancestor, List.mem_append.mpr (Or.inl member), matchRoute⟩

theorem new_reseed_route_has_matching_source
    (existing : List Ancestral) (newRecord : Ancestral)
    (derived : List FlowPair) (route : FlowPair)
    (present : route ∈ reseedDerivedRoutes
      (existing ++ [newRecord]) derived)
    (notOld : route ∉ reseedDerivedRoutes existing derived) :
    ∃ source ∈ derived,
      colourPathPrefixB source.2 newRecord.COLOURS = true ∧
        route = (source.1, newRecord.COLOURS) := by
  rw [reseedDerivedRoutes, List.mem_eraseDups,
    List.flatMap_append, List.mem_append] at present
  rcases present with old | fresh
  · exact False.elim (notOld (by simpa [reseedDerivedRoutes] using old))
  · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at fresh
    rw [List.mem_filterMap] at fresh
    obtain ⟨source, member, mapped⟩ := fresh
    by_cases hprefix : colourPathPrefixB source.2 newRecord.COLOURS = true
    · refine ⟨source, member, hprefix, ?_⟩
      simpa [hprefix] using mapped.symm
    · simp [hprefix] at mapped

theorem selected_use_not_reemitted_when_head_exists
    (later : List (List Vertex)) (nodes : List Vertex)
    (before : Graph) (rule : Neighborhood) (after : Graph)
    (owner : Vertex)
    (present : after.PATHS.any (fun record =>
      record.END == rule.CENTER &&
        record.COLOURS.head? == some owner.NUMBER) = true) :
    RoutedEliminationTransport.stableUse? later nodes before rule after owner = none := by
  simp [RoutedEliminationTransport.stableUse?, RoutedEliminationTransport.broadUse?, present]
  split <;> simp

theorem emitted_record_has_actual_owner_use
    (later : List (List Vertex)) (nodes : List Vertex)
    (before : Graph) (rule : Neighborhood) (after : Graph)
    (owner : Vertex) (record : Ancestral)
    (emitted : RoutedEliminationTransport.stableUse? later nodes before rule after
      owner = some record) :
    owner ∈ nodes ∧
    (before.NODES.filter (fun vertex =>
      vertex.NUMBER == owner.NUMBER)).length = 1 ∧
    ∃ originalUse,
      get_rule.outgoing owner before = [originalUse] ∧
      rule.OUTGOING.any (fun use =>
        use.START == rule.CENTER && use.END == originalUse.END &&
          use.COLOUR == owner.NUMBER) = true ∧
      record = path originalUse.END rule.CENTER [owner.NUMBER] := by
  unfold RoutedEliminationTransport.stableUse? at emitted
  cases hUses : get_rule.outgoing owner before with
  | nil => simp [hUses] at emitted
  | cons first rest =>
      cases rest with
      | nil =>
          simp [hUses] at emitted
          unfold RoutedEliminationTransport.broadUse? at emitted
          split at emitted <;> simp_all
      | cons second tail =>
          simp [hUses, RoutedEliminationTransport.broadUse?] at emitted

theorem emitted_head_is_structural (after : Graph)
    (centre consumer : Vertex) (colour : Nat) (stored : List Formula)
    (use : edge centre consumer colour stored ∈ after.EDGES) :
    ancestralPathHeadOKB after (path consumer centre [colour]) = true := by
  simp only [ancestralPathHeadOKB, List.any_eq_true]
  exact ⟨edge centre consumer colour stored, use, by simp⟩







structure RepresentedIntroductionUse where
  owner : Vertex
  originalPremise : Vertex
  originalConsumer : Vertex
  premise : Vertex
  consumer : Vertex
  centre : Vertex
  originalIncoming : Deduction
  originalUse : Deduction
  incomingColour : Nat
  antecedent : Formula
  consequent : Formula
  premiseDependency : Dep
  tail : List Nat
deriving Repr

namespace RepresentedIntroductionUse

def address (use : RepresentedIntroductionUse) : List Nat :=
  use.owner.NUMBER :: use.tail

def premiseResidual (use : RepresentedIntroductionUse) : List Nat :=
  use.incomingColour :: use.address

def logicalOutput (use : RepresentedIntroductionUse) : FlowPair :=
  (depRemove use.premiseDependency use.antecedent, use.address)

def record (use : RepresentedIntroductionUse) : Ancestral :=
  path use.consumer use.centre use.address




def rewriteCurrentEndpoints (nodes : List Vertex) (replacement : Vertex)
    (use : RepresentedIntroductionUse) : RepresentedIntroductionUse :=
  { use with
    premise := if use.premise ∈ nodes then replacement else use.premise
    consumer := if use.consumer ∈ nodes then replacement else use.consumer
    centre := if use.centre ∈ nodes then replacement else use.centre }

@[simp] theorem address_rewriteCurrentEndpoints (nodes : List Vertex)
    (replacement : Vertex) (use : RepresentedIntroductionUse) :
    (use.rewriteCurrentEndpoints nodes replacement).address = use.address := by
  simp [rewriteCurrentEndpoints, address]

@[simp] theorem tail_rewriteCurrentEndpoints (nodes : List Vertex)
    (replacement : Vertex) (use : RepresentedIntroductionUse) :
    (use.rewriteCurrentEndpoints nodes replacement).tail = use.tail := by
  rfl

theorem record_rewriteCurrentEndpoints (nodes : List Vertex)
    (replacement : Vertex) (use : RepresentedIntroductionUse) :
    (use.rewriteCurrentEndpoints nodes replacement).record =
      collapse.rewrite_path_endpoints nodes replacement use.record := by
  simp [rewriteCurrentEndpoints, record, address,
    collapse.rewrite_path_endpoints]

def currentIncomingB (after : Graph)
    (use : RepresentedIntroductionUse) : Bool :=
  after.EDGES.any fun incoming =>
    incoming.START == use.premise && incoming.END == use.centre &&
      incoming.COLOUR == use.incomingColour

def currentUseB (after : Graph)
    (use : RepresentedIntroductionUse) : Bool :=
  after.EDGES.any fun outgoing =>
    outgoing.START == use.centre && outgoing.END == use.consumer &&
      outgoing.COLOUR == use.owner.NUMBER





structure Sound (selected : List Vertex) (before after : Graph)
    (premiseFlow : Vertex -> List FlowPair)
    (use : RepresentedIntroductionUse) : Prop where
  selectedOwner : use.owner ∈ selected
  ownerInBefore : use.owner ∈ before.NODES
  uniqueOwnerNumber :
    (before.NODES.filter fun vertex =>
      vertex.NUMBER == use.owner.NUMBER).length = 1
  ownerUncollapsed : use.owner.COLLAPSED = false
  ownerDerived : use.owner.HYPOTHESIS = false
  ownerFormula :
    use.owner.FORMULA = Formula.implication use.antecedent use.consequent
  centreFormula :
    use.centre.FORMULA = Formula.implication use.antecedent use.consequent
  premiseFormula : use.premise.FORMULA = use.consequent
  originalIncomingExact :
    get_rule.incoming use.owner before = [use.originalIncoming]
  originalIncomingStart : use.originalIncoming.START = use.originalPremise
  originalIncomingEnd : use.originalIncoming.END = use.owner
  originalIncomingColour :
    use.originalIncoming.COLOUR = use.incomingColour
  originalUseExact : get_rule.outgoing use.owner before = [use.originalUse]
  originalUseStart : use.originalUse.START = use.owner
  originalUseEnd : use.originalUse.END = use.originalConsumer
  centreInAfter : use.centre ∈ after.NODES
  currentIncoming : currentIncomingB after use = true
  currentUse : currentUseB after use = true
  generatedTrace :
    path use.consumer use.premise use.premiseResidual ∈ after.PATHS
  premiseRoute :
    (use.premiseDependency, use.premiseResidual) ∈ premiseFlow use.premise

def supportedB (selected : List Vertex) (before after : Graph)
    (premiseFlow : Vertex -> List FlowPair)
    (use : RepresentedIntroductionUse) : Bool :=
  selected.contains use.owner &&
  before.NODES.contains use.owner &&
  (before.NODES.filter fun vertex =>
    vertex.NUMBER == use.owner.NUMBER).length == 1 &&
  !use.owner.COLLAPSED && !use.owner.HYPOTHESIS &&
  use.owner.FORMULA == Formula.implication use.antecedent use.consequent &&
  use.centre.FORMULA == Formula.implication use.antecedent use.consequent &&
  use.premise.FORMULA == use.consequent &&
  get_rule.incoming use.owner before == [use.originalIncoming] &&
  use.originalIncoming.START == use.originalPremise &&
  use.originalIncoming.END == use.owner &&
  use.originalIncoming.COLOUR == use.incomingColour &&
  get_rule.outgoing use.owner before == [use.originalUse] &&
  use.originalUse.START == use.owner &&
  use.originalUse.END == use.originalConsumer &&
  after.NODES.contains use.centre &&
  use.currentIncomingB after && use.currentUseB after &&
  after.PATHS.contains
    (path use.consumer use.premise use.premiseResidual) &&
  (premiseFlow use.premise).any fun route =>
    route.1 == use.premiseDependency &&
      route.2 == use.premiseResidual

theorem sound_of_supportedB
    {selected : List Vertex} {before after : Graph}
    {premiseFlow : Vertex -> List FlowPair}
    {use : RepresentedIntroductionUse}
    (supported : use.supportedB selected before after premiseFlow = true) :
    use.Sound selected before after premiseFlow := by
  simp only [supportedB, Bool.and_eq_true] at supported
  constructor <;> simp_all [List.any_eq_true]

end RepresentedIntroductionUse



def producerDerivedUse? (selected : List Vertex) (before after : Graph)
    (centre owner : Vertex) (premiseFlow : Vertex -> List FlowPair) :
    Option { use : RepresentedIntroductionUse //
      use.Sound selected before after premiseFlow } := by
  match owner.FORMULA, get_rule.incoming owner before,
      get_rule.outgoing owner before with
  | Formula.implication antecedent consequent, [incoming], [outgoing] =>
      match (premiseFlow incoming.START).find? fun route =>
          match route.2 with
          | incomingColour :: ownerColour :: _ =>
              incomingColour == incoming.COLOUR &&
                ownerColour == owner.NUMBER
          | _ => false with
      | none => exact none
      | some route =>
          match route.2 with
          | incomingColour :: _ :: tail =>
              let selection : RepresentedIntroductionUse := {
                owner := owner
                originalPremise := incoming.START
                originalConsumer := outgoing.END
                premise := incoming.START
                consumer := outgoing.END
                centre := centre
                originalIncoming := incoming
                originalUse := outgoing
                incomingColour := incomingColour
                antecedent := antecedent
                consequent := consequent
                premiseDependency := route.1
                tail := tail }
              if sound : selection.supportedB selected before after
                  premiseFlow = true then
                exact some ⟨selection,
                  RepresentedIntroductionUse.sound_of_supportedB sound⟩
              else exact none
          | _ => exact none
  | _, _, _ => exact none

theorem producerDerivedUse?_sound
    (selected : List Vertex) (before after : Graph)
    (centre owner : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (supported : { use : RepresentedIntroductionUse //
      use.Sound selected before after premiseFlow })
    (_extracted : producerDerivedUse? selected before after centre owner
      premiseFlow = some supported) :
    supported.1.Sound selected before after premiseFlow := by
  exact supported.2

private theorem predecessor_of_currentIncoming
    {after : Graph} {use : RepresentedIntroductionUse}
    (current : use.currentIncomingB after = true) :
    use.premise ∈ predsOf after use.centre := by
  rw [RepresentedIntroductionUse.currentIncomingB,
    List.any_eq_true] at current
  obtain ⟨incoming, incomingMember, facts⟩ := current
  have facts' : (incoming.START = use.premise ∧
      incoming.END = use.centre) ∧
      incoming.COLOUR = use.incomingColour := by
    simpa only [Bool.and_eq_true, beq_iff_eq] using facts
  obtain ⟨⟨startEq, endEq⟩, _⟩ := facts'
  rw [predsOf, List.mem_eraseDups, List.mem_map]
  refine ⟨incoming, ?_, startEq⟩
  simpa [endEq] using incomingMember

private theorem exists_currentIncoming
    {after : Graph} {use : RepresentedIntroductionUse}
    (current : use.currentIncomingB after = true) :
    ∃ incoming ∈ edgesBetween after use.premise use.centre,
      incoming.COLOUR = use.incomingColour := by
  rw [RepresentedIntroductionUse.currentIncomingB,
    List.any_eq_true] at current
  obtain ⟨incoming, incomingMember, facts⟩ := current
  have facts' : (incoming.START = use.premise ∧
      incoming.END = use.centre) ∧
      incoming.COLOUR = use.incomingColour := by
    simpa only [Bool.and_eq_true, beq_iff_eq] using facts
  obtain ⟨⟨startEq, endEq⟩, colourEq⟩ := facts'
  refine ⟨incoming, ?_, colourEq⟩
  simp [edgesBetween, incomingMember, startEq, endEq]




theorem logicalOutput_mem_introductionSelections
    {selected : List Vertex} {before after : Graph}
    {premiseFlow : Vertex -> List FlowPair}
    {use : RepresentedIntroductionUse}
    (sound : use.Sound selected before after premiseFlow) :
    use.logicalOutput ∈
      introductionSelections after use.centre premiseFlow := by
  rw [mem_introductionSelections_iff, sound.centreFormula]
  refine ⟨(use.premiseDependency, use.address), ?_, rfl⟩
  refine ⟨use.premise,
    predecessor_of_currentIncoming sound.currentIncoming,
    sound.premiseFormula, (use.premiseDependency, use.premiseResidual),
    sound.premiseRoute, ?_⟩
  obtain ⟨incoming, incomingMember, colourEq⟩ :=
    exists_currentIncoming sound.currentIncoming
  refine ⟨incoming, incomingMember, ?_, rfl⟩
  simp [RepresentedIntroductionUse.premiseResidual,
    RepresentedIntroductionUse.address, consume, colourEq]




theorem logicalOutput_mem_derivedOutputs
    {selected : List Vertex} {before after : Graph}
    {premiseFlow : Vertex -> List FlowPair}
    {use : RepresentedIntroductionUse}
    (sound : use.Sound selected before after premiseFlow)
    (separate : residualOccursB
      (selectionEliminationOutputs after use.centre premiseFlow)
      use.address = false) :
    use.logicalOutput ∈ derivedOutputs after use.centre premiseFlow := by
  apply separatedIntroduction_mem_derivedOutputs
  rw [mem_separatedIntroductionOutputs_iff]
  exact ⟨logicalOutput_mem_introductionSelections sound, separate⟩



theorem logicalOutput_mem_reseeded
    {selected : List Vertex} {before after : Graph}
    {premiseFlow : Vertex -> List FlowPair}
    {use : RepresentedIntroductionUse}
    (sound : use.Sound selected before after premiseFlow)
    (separate : residualOccursB
      (selectionEliminationOutputs after use.centre premiseFlow)
      use.address = false) :
    use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto after use.centre ++ [use.record])
      (derivedOutputs after use.centre premiseFlow) := by
  exact delivered_full_tail (ancestorsInto after use.centre)
    use.consumer use.centre use.owner.NUMBER use.tail
    (depRemove use.premiseDependency use.antecedent)
    (derivedOutputs after use.centre premiseFlow)
    (logicalOutput_mem_derivedOutputs sound separate)



theorem consume_complete_address (use : RepresentedIntroductionUse)
    (stored : List Formula) :
    consume (edge use.centre use.consumer use.owner.NUMBER stored)
      use.address = some use.tail := by
  simp [RepresentedIntroductionUse.address, consume]




theorem producer_to_delivery
    {selected : List Vertex} {before after : Graph}
    {premiseFlow : Vertex -> List FlowPair}
    {use : RepresentedIntroductionUse}
    (sound : use.Sound selected before after premiseFlow)
    (separate : residualOccursB
      (selectionEliminationOutputs after use.centre premiseFlow)
      use.address = false) :
    use.logicalOutput ∈ introductionSelections after use.centre premiseFlow ∧
    use.logicalOutput ∈ derivedOutputs after use.centre premiseFlow ∧
    use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto after use.centre ++ [use.record])
      (derivedOutputs after use.centre premiseFlow) ∧
    ∀ stored, consume
      (edge use.centre use.consumer use.owner.NUMBER stored)
      use.address = some use.tail := by
  exact ⟨logicalOutput_mem_introductionSelections sound,
    logicalOutput_mem_derivedOutputs sound separate,
    logicalOutput_mem_reseeded sound separate,
    fun stored => consume_complete_address use stored⟩



theorem existing_route_survives_supported_addition
    {after : Graph} {use : RepresentedIntroductionUse}
    {derived : List FlowPair} {route : FlowPair}
    (present : route ∈ reseedDerivedRoutes
      (ancestorsInto after use.centre) derived) :
    route ∈ reseedDerivedRoutes
      (ancestorsInto after use.centre ++ [use.record]) derived := by
  exact existing_reseed_survives _ _ _ _ present



theorem other_owner_cannot_consume (use : RepresentedIntroductionUse)
    (other : Nat) (stored : List Formula)
    (different : other ≠ use.owner.NUMBER) :
    consume (edge use.centre use.consumer other stored) use.address = none := by
  simp [RepresentedIntroductionUse.address, consume, different]



theorem complete_address_survives_endpoint_rewrite
    (nodes : List Vertex) (replacement : Vertex)
    (use : RepresentedIntroductionUse) :
    (collapse.rewrite_path_endpoints nodes replacement use.record).COLOURS =
      use.address := by
  simp [collapse.rewrite_path_endpoints, RepresentedIntroductionUse.record]

theorem complete_address_follows_rewritten_consumer
    (nodes : List Vertex) (replacement : Vertex)
    (use : RepresentedIntroductionUse)
    (consumerSelected : use.consumer ∈ nodes)
    (centreRetained : use.centre ∉ nodes) :
    collapse.rewrite_path_endpoints nodes replacement use.record =
      path replacement use.centre use.address := by
  simp [collapse.rewrite_path_endpoints, RepresentedIntroductionUse.record,
    consumerSelected, centreRetained]





def supportedStep (partition : List HCClass) (selected : List Vertex)
    (before : Graph) (occurrences : List Ancestral) (fuel : Nat) :
    Graph × List Ancestral :=
  let base := compress_nodesWithPartitionCarrying partition selected before
    occurrences
  let centre := (compress_ruleWithOccurrences partition selected before).1.CENTER
  let premiseFlow := fun premise =>
    match occurrenceReplayAt base.1 fuel premise with
    | .ok routes => routes
    | .error _ => []
  let supported := selected.filterMap fun owner =>
    producerDerivedUse? selected before base.1 centre owner premiseFlow
  let activated := supported.filter fun evidence =>
    !(ancestorsInto base.1 evidence.1.centre).isEmpty
  let added := activated.map fun evidence => evidence.1.record
  (dlds base.1.NODES base.1.EDGES (base.1.PATHS ++ added), base.2)

@[simp] theorem supportedStep_nodes (partition : List HCClass)
    (selected : List Vertex) (before : Graph)
    (occurrences : List Ancestral) (fuel : Nat) :
    (supportedStep partition selected before occurrences fuel).1.NODES =
      (compress_nodesWithPartitionCarrying partition selected before
        occurrences).1.NODES := by
  rfl

@[simp] theorem supportedStep_edges (partition : List HCClass)
    (selected : List Vertex) (before : Graph)
    (occurrences : List Ancestral) (fuel : Nat) :
    (supportedStep partition selected before occurrences fuel).1.EDGES =
      (compress_nodesWithPartitionCarrying partition selected before
        occurrences).1.EDGES := by
  rfl

@[simp] theorem supportedStep_occurrences (partition : List HCClass)
    (selected : List Vertex) (before : Graph)
    (occurrences : List Ancestral) (fuel : Nat) :
    (supportedStep partition selected before occurrences fuel).2 =
      (compress_nodesWithPartitionCarrying partition selected before
        occurrences).2 := by
  rfl

end RoutedEliminationLocal

#print axioms RoutedEliminationLocal.emitted_record_has_actual_owner_use
#print axioms RoutedEliminationLocal.delivered_full_tail
#print axioms RoutedEliminationLocal.consume_delivered_full_tail
#print axioms RoutedEliminationLocal.consume_other_owner
#print axioms RoutedEliminationLocal.existing_reseed_survives
#print axioms RoutedEliminationLocal.new_reseed_route_has_matching_source
#print axioms RoutedEliminationLocal.selected_use_not_reemitted_when_head_exists
#print axioms RoutedEliminationLocal.emitted_head_is_structural
#print axioms RoutedEliminationLocal.logicalOutput_mem_introductionSelections
#print axioms RoutedEliminationLocal.logicalOutput_mem_derivedOutputs
#print axioms RoutedEliminationLocal.logicalOutput_mem_reseeded
#print axioms RoutedEliminationLocal.producer_to_delivery
#print axioms RoutedEliminationLocal.existing_route_survives_supported_addition
#print axioms RoutedEliminationLocal.consume_complete_address
#print axioms RoutedEliminationLocal.other_owner_cannot_consume
#print axioms RoutedEliminationLocal.complete_address_survives_endpoint_rewrite
#print axioms RoutedEliminationLocal.complete_address_follows_rewritten_consumer
#print axioms RoutedEliminationLocal.RepresentedIntroductionUse.record_rewriteCurrentEndpoints
#print axioms RoutedEliminationLocal.supportedStep_nodes
#print axioms RoutedEliminationLocal.supportedStep_edges
#print axioms RoutedEliminationLocal.supportedStep_occurrences
