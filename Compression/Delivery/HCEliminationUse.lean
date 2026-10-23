import Compression.Delivery.RepresentedUseTransport






open Semantic FlowSpec OccurrenceAwareReplay MixedRuleReplay
open HCSuccessorDeliverySemantics

namespace HCEliminationUse

def continuation? (before : Graph) (owner : Vertex) (outgoing : Deduction) :
    Option (Vertex × List Nat) :=
  match get_rule.direct owner before with
  | [] => some (outgoing.END, [])
  | [trace] => match trace.COLOURS with
    | head :: tail => if head == outgoing.COLOUR then some (trace.START, tail) else none
    | [] => none
  | _ => none

structure RepresentedEliminationUse where
  owner : Vertex
  originalMajor : Vertex
  originalMinor : Vertex
  originalConsumer : Vertex
  originalTraceStart : Vertex
  originalMajorEdge : Deduction
  originalMinorEdge : Deduction
  originalUse : Deduction
  major : Vertex
  minor : Vertex
  consumer : Vertex
  traceStart : Vertex
  centre : Vertex
  majorColour : Nat
  minorColour : Nat
  majorDependency : Dep
  minorDependency : Dep
  tail : List Nat
deriving Repr

namespace RepresentedEliminationUse

def address (use : RepresentedEliminationUse) : List Nat := use.owner.NUMBER :: use.tail
def logicalOutput (use : RepresentedEliminationUse) : FlowPair :=
  (depUnion use.minorDependency use.majorDependency, use.address)
def record (use : RepresentedEliminationUse) : Ancestral :=
  path use.traceStart use.centre use.address
def rewriteCurrentEndpoints (selected : List Vertex) (replacement : Vertex)
    (use : RepresentedEliminationUse) : RepresentedEliminationUse :=
  let move := fun vertex => if vertex ∈ selected then replacement else vertex
  { use with
    major := move use.major
    minor := move use.minor
    consumer := move use.consumer
    traceStart := move use.traceStart
    centre := move use.centre }
def currentIncomingB (after : Graph) (premise centre : Vertex) (colour : Nat) : Bool :=
  after.EDGES.any fun incoming =>
    incoming.START == premise && incoming.END == centre && incoming.COLOUR == colour
def currentUseB (after : Graph) (use : RepresentedEliminationUse) : Bool :=
  currentIncomingB after use.centre use.consumer use.owner.NUMBER

def Sound (selected : List Vertex) (before after : Graph)
    (flow : Vertex → List FlowPair) (use : RepresentedEliminationUse) : Prop :=
  use.owner ∈ selected ∧ use.owner ∈ before.NODES ∧
  (before.NODES.filter fun v => v.NUMBER == use.owner.NUMBER).length = 1 ∧
  use.owner.COLLAPSED = false ∧ use.owner.HYPOTHESIS = false ∧
  (get_rule.incoming use.owner before = [use.originalMajorEdge, use.originalMinorEdge] ∨
   get_rule.incoming use.owner before = [use.originalMinorEdge, use.originalMajorEdge]) ∧
  use.originalMajorEdge.START = use.originalMajor ∧
  use.originalMinorEdge.START = use.originalMinor ∧
  use.originalMajorEdge.END = use.owner ∧ use.originalMinorEdge.END = use.owner ∧
  use.originalMajorEdge.COLOUR = use.majorColour ∧
  use.originalMinorEdge.COLOUR = use.minorColour ∧
  use.originalMajor.FORMULA = Formula.implication use.originalMinor.FORMULA use.owner.FORMULA ∧
  get_rule.outgoing use.owner before = [use.originalUse] ∧
  use.originalUse.START = use.owner ∧ use.originalUse.END = use.originalConsumer ∧
  continuation? before use.owner use.originalUse = some (use.originalTraceStart, use.tail) ∧
  use.centre ∈ after.NODES ∧
  use.major.FORMULA = Formula.implication use.minor.FORMULA use.centre.FORMULA ∧
  currentIncomingB after use.major use.centre use.majorColour = true ∧
  currentIncomingB after use.minor use.centre use.minorColour = true ∧
  use.currentUseB after = true ∧
  path use.traceStart use.major (use.majorColour :: use.address) ∈ after.PATHS ∧
  path use.traceStart use.minor (use.minorColour :: use.address) ∈ after.PATHS ∧
  (use.majorDependency, use.majorColour :: use.address) ∈ flow use.major ∧
  (use.minorDependency, use.minorColour :: use.address) ∈ flow use.minor

instance (selected : List Vertex) (before after : Graph)
    (flow : Vertex → List FlowPair) (use : RepresentedEliminationUse) :
    Decidable (use.Sound selected before after flow) := by unfold Sound; infer_instance

theorem same_owner_incidence {selected before after flow} {use : RepresentedEliminationUse}
    (sound : use.Sound selected before after flow) :
    use.originalMajorEdge.END = use.owner ∧
    use.originalMinorEdge.END = use.owner ∧ use.originalUse.START = use.owner := by
  rcases sound with ⟨_, _, _, _, _, _, _, _, hMajor, hMinor, _, _, _, _, hUse, _⟩
  exact ⟨hMajor, hMinor, hUse⟩

theorem rejects_cross_owner {selected before after flow} {use : RepresentedEliminationUse}
    (sound : use.Sound selected before after flow) (other : Vertex)
    (different : other ≠ use.owner) : use.originalMinorEdge.END ≠ other := by
  rw [(same_owner_incidence sound).2.1]
  exact Ne.symm different

theorem empty_tail_of_no_direct {selected before after flow} {use : RepresentedEliminationUse}
    (sound : use.Sound selected before after flow)
    (noDirect : get_rule.direct use.owner before = []) : use.tail = [] := by
  rcases sound with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hContinuation, _⟩
  simp only [continuation?, noDirect] at hContinuation
  exact (Prod.mk.inj (Option.some.inj hContinuation)).2.symm



theorem full_tail_of_direct {selected before after flow} {use : RepresentedEliminationUse}
    (sound : use.Sound selected before after flow) (trace : Ancestral)
    (direct : get_rule.direct use.owner before = [trace])
    (colours : trace.COLOURS = use.originalUse.COLOUR :: use.tail) :
    use.originalTraceStart = trace.START := by
  rcases sound with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hContinuation, _⟩
  simp only [continuation?, direct, colours, beq_self_eq_true, ↓reduceIte] at hContinuation
  exact (Prod.mk.inj (Option.some.inj hContinuation)).1.symm

theorem owner_addresses_separate (left right : RepresentedEliminationUse)
    (different : left.owner.NUMBER ≠ right.owner.NUMBER) : left.address ≠ right.address := by
  intro equal
  exact different (List.cons.inj equal).1

theorem other_owner_cannot_consume (use : RepresentedEliminationUse) (other : Nat)
    (stored : List Formula) (different : other ≠ use.owner.NUMBER) :
    consume (edge use.centre use.consumer other stored) use.address = none := by
  simp [consume, address, different]

theorem record_rewriteCurrentEndpoints (selected : List Vertex) (replacement : Vertex)
    (use : RepresentedEliminationUse) :
    (use.rewriteCurrentEndpoints selected replacement).record =
      collapse.rewrite_path_endpoints selected replacement use.record := by
  simp [record, address, rewriteCurrentEndpoints, collapse.rewrite_path_endpoints]

private theorem current_edge {after premise centre colour}
    (current : currentIncomingB after premise centre colour = true) :
    premise ∈ predsOf after centre ∧
    ∃ incoming ∈ edgesBetween after premise centre, incoming.COLOUR = colour := by
  simp only [currentIncomingB, List.any_eq_true, Bool.and_eq_true, beq_iff_eq] at current
  obtain ⟨incoming, member, ⟨hStart, hEnd⟩, hColour⟩ := current
  constructor
  · rw [predsOf, List.mem_eraseDups, List.mem_map]
    exact ⟨incoming, by simpa [hEnd] using member, hStart⟩
  · exact ⟨incoming, by simp [edgesBetween, member, hStart, hEnd], hColour⟩

theorem logicalOutput_mem_elimination {selected before after flow} {use : RepresentedEliminationUse}
    (sound : use.Sound selected before after flow) :
    use.logicalOutput ∈ selectionEliminationOutputs after use.centre flow := by
  rcases sound with ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    formulas, majorCurrent, minorCurrent, _, _, _, majorRoute, minorRoute⟩
  obtain ⟨majorPred, majorEdge, majorMember, majorColour⟩ := current_edge majorCurrent
  obtain ⟨minorPred, minorEdge, minorMember, minorColour⟩ := current_edge minorCurrent
  rw [mem_selectionEliminationOutputs_iff]
  refine ⟨use.major, majorPred, use.minor, minorPred, formulas,
    (use.minorDependency, use.minorColour :: use.address), minorRoute,
    (use.majorDependency, use.majorColour :: use.address), majorRoute,
    minorEdge, minorMember, majorEdge, majorMember, use.address, ?_, ?_, rfl⟩
  · simp [consume, minorColour]
  · simp [consume, majorColour]

theorem producer_to_delivery {selected before after flow} {use : RepresentedEliminationUse}
    (sound : use.Sound selected before after flow) :
    use.logicalOutput ∈ selectionEliminationOutputs after use.centre flow ∧
    use.logicalOutput ∈ derivedOutputs after use.centre flow ∧
    use.logicalOutput ∈ reseedDerivedRoutes
      (ancestorsInto after use.centre ++ [use.record])
      (derivedOutputs after use.centre flow) ∧
    ∀ stored, consume (edge use.centre use.consumer use.owner.NUMBER stored)
      use.address = some use.tail := by
  have eliminated := logicalOutput_mem_elimination sound
  have derived := elimination_mem_derivedOutputs after use.centre flow use.logicalOutput eliminated
  refine ⟨eliminated, derived, ?_, ?_⟩
  · exact RoutedEliminationLocal.delivered_full_tail _ _ _ _ _ _ _ derived
  · intro stored; simp [consume, address]

end RepresentedEliminationUse



def producerDerivedUse? (selected : List Vertex) (before after : Graph)
    (centre owner : Vertex) (flow : Vertex → List FlowPair) :
    Option { use : RepresentedEliminationUse // use.Sound selected before after flow } := do
  let [first, second] := get_rule.incoming owner before | none
  let [outgoing] := get_rule.outgoing owner before | none
  let (traceStart, tail) ← continuation? before owner outgoing
  let (majorEdge, minorEdge) :=
    if first.START.FORMULA == Formula.implication second.START.FORMULA owner.FORMULA
    then (first, second) else (second, first)
  let address := owner.NUMBER :: tail
  let majorRoute ← (flow majorEdge.START).find? (fun r => r.2 == majorEdge.COLOUR :: address)
  let minorRoute ← (flow minorEdge.START).find? (fun r => r.2 == minorEdge.COLOUR :: address)
  let use : RepresentedEliminationUse := {
    owner := owner, originalMajor := majorEdge.START, originalMinor := minorEdge.START,
    originalConsumer := outgoing.END, originalTraceStart := traceStart,
    originalMajorEdge := majorEdge, originalMinorEdge := minorEdge, originalUse := outgoing,
    major := majorEdge.START, minor := minorEdge.START, consumer := outgoing.END,
    traceStart := traceStart, centre := centre, majorColour := majorEdge.COLOUR,
    minorColour := minorEdge.COLOUR, majorDependency := majorRoute.1,
    minorDependency := minorRoute.1, tail := tail }
  if sound : use.Sound selected before after flow then some ⟨use, sound⟩ else none

end HCEliminationUse

#print axioms HCEliminationUse.RepresentedEliminationUse.same_owner_incidence
#print axioms HCEliminationUse.RepresentedEliminationUse.rejects_cross_owner
#print axioms HCEliminationUse.RepresentedEliminationUse.empty_tail_of_no_direct
#print axioms HCEliminationUse.RepresentedEliminationUse.full_tail_of_direct
#print axioms HCEliminationUse.RepresentedEliminationUse.owner_addresses_separate
#print axioms HCEliminationUse.RepresentedEliminationUse.other_owner_cannot_consume
#print axioms HCEliminationUse.RepresentedEliminationUse.record_rewriteCurrentEndpoints
#print axioms HCEliminationUse.RepresentedEliminationUse.logicalOutput_mem_elimination
#print axioms HCEliminationUse.RepresentedEliminationUse.producer_to_delivery
