import Compression.HCEliminationClassPreservation

set_option linter.unusedSimpArgs false











open Semantic FlowSpec
open MixedRuleReplay OccurrenceAwareReplay
open MixedRuleSpecification HCContextualMixedHE
open HCEliminationClassPreservation

namespace HCEliminationClassOperational


def accumulatedGraph (center : Vertex) (occurrences : List ElimOccurrence) :
    Graph :=
  ActualMixedProducerBridge.neighborhoodGraph
    (accumulatedNeighborhood center occurrences)



def expectedFlowRoutes (occurrences : List ElimOccurrence) : List FlowPair :=
  occurrences.map fun occurrence =>
    (depUnion occurrence.minorDep occurrence.majorDep, [occurrence.tag])




def expectedPremiseRoutes (occurrences : List ElimOccurrence) (v : Vertex) :
    List FlowPair :=
  occurrences.flatMap fun occurrence =>
    (if v = occurrence.major then
      [(occurrence.majorDep, [0, occurrence.tag])]
    else []) ++
    (if v = occurrence.minor then
      [(occurrence.minorDep, [0, occurrence.tag])]
    else [])




def PremiseRoutesCorrect (premiseFlow : Vertex -> List FlowPair)
    (occurrences : List ElimOccurrence) : Prop :=
  forall v, SelfCertifying.ListSetEq (premiseFlow v)
    (expectedPremiseRoutes occurrences v)

def PremiseOf (occurrence : ElimOccurrence) (v : Vertex) : Prop :=
  v = occurrence.major \/ v = occurrence.minor


theorem mem_expectedPremiseRoutes_iff
    (occurrences : List ElimOccurrence) (v : Vertex) (route : FlowPair) :
    route ∈ expectedPremiseRoutes occurrences v <->
      Exists fun occurrence =>
        occurrence ∈ occurrences /\
          ((v = occurrence.major /\
              route = (occurrence.majorDep, [0, occurrence.tag])) \/
           (v = occurrence.minor /\
              route = (occurrence.minorDep, [0, occurrence.tag]))) := by
  simp [expectedPremiseRoutes, and_or_left]

def PremisesAvoidCenter (center : Vertex)
    (occurrences : List ElimOccurrence) : Prop :=
  forall occurrence, occurrence ∈ occurrences ->
    occurrence.major ≠ center /\ occurrence.minor ≠ center

def UsesAvoidCenter (center : Vertex)
    (occurrences : List ElimOccurrence) : Prop :=
  forall occurrence, occurrence ∈ occurrences -> occurrence.use ≠ center

theorem accumulated_incoming
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences) :
    get_rule.incoming center (accumulatedGraph center occurrences) =
      occurrences.flatMap (ElimOccurrence.incoming center) := by
  simp only [accumulatedGraph,
    ActualMixedProducerBridge.neighborhoodGraph,
    accumulatedNeighborhood, List.nil_append]
  rw [FlowSpec.incoming_eq_filter]
  rw [List.filter_append]
  have incomingFilter :
      (occurrences.flatMap (ElimOccurrence.incoming center)).filter
          (fun edge => decide (edge.END = center)) =
        occurrences.flatMap (ElimOccurrence.incoming center) := by
    apply List.filter_eq_self.mpr
    intro selection selectionMember
    obtain ⟨occurrence, occurrenceMember, edgeMember⟩ :=
      List.mem_flatMap.mp selectionMember
    simp [ElimOccurrence.incoming] at edgeMember
    rcases edgeMember with rfl | rfl <;> simp
  have outgoingFilter :
      (occurrenceOutgoing center occurrences).filter
          (fun edge => decide (edge.END = center)) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro selection selectionMember
    obtain ⟨occurrence, occurrenceMember, edgeEqual⟩ :=
      List.mem_map.mp selectionMember
    subst selection
    simpa [ElimOccurrence.outgoing] using
      usesAvoid occurrence occurrenceMember
  rw [incomingFilter, outgoingFilter, List.append_nil]

theorem accumulated_no_center_ancestors
    (center : Vertex) (occurrences : List ElimOccurrence)
    (premisesAvoid : PremisesAvoidCenter center occurrences) :
    ancestorsInto (accumulatedGraph center occurrences) center = [] := by
  unfold ancestorsInto accumulatedGraph
    ActualMixedProducerBridge.neighborhoodGraph
  simp only [accumulatedNeighborhood, List.nil_append]
  induction occurrences with
  | nil => simp
  | cons occurrence rest ih =>
      have headAvoid := premisesAvoid occurrence (List.Mem.head rest)
      have tailAvoid : PremisesAvoidCenter center rest := by
        intro selection member
        exact premisesAvoid selection (List.Mem.tail occurrence member)
      simp [ElimOccurrence.indirect, headAvoid.1, headAvoid.2,
        ih tailAvoid]



theorem mem_accumulated_preds_iff
    (center u : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences) :
    u ∈ predsOf (accumulatedGraph center occurrences) center <->
      Exists fun occurrence => occurrence ∈ occurrences /\
        PremiseOf occurrence u := by
  rw [FlowSpec.predsOf_eq,
    accumulated_incoming center occurrences usesAvoid]
  simp only [List.mem_eraseDups, List.mem_map, List.mem_flatMap]
  constructor
  · rintro ⟨selection, ⟨occurrence, occurrenceMember, edgeMember⟩,
        startEqual⟩
    simp [ElimOccurrence.incoming] at edgeMember
    rcases edgeMember with rfl | rfl
    · exact ⟨occurrence, occurrenceMember,
        Or.inl startEqual.symm⟩
    · exact ⟨occurrence, occurrenceMember,
        Or.inr startEqual.symm⟩
  · rintro ⟨occurrence, occurrenceMember, major | minor⟩
    · subst u
      exact ⟨edge occurrence.major center 0 occurrence.majorStored,
        ⟨occurrence, occurrenceMember,
          by simp [ElimOccurrence.incoming]⟩, rfl⟩
    · subst u
      exact ⟨edge occurrence.minor center 0 occurrence.minorStored,
        ⟨occurrence, occurrenceMember,
          by simp [ElimOccurrence.incoming]⟩, rfl⟩



theorem accumulated_edgesBetween_colour_zero
    (center u : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (selection : Deduction)
    (member : selection ∈
      edgesBetween (accumulatedGraph center occurrences) u center) :
    selection.COLOUR = 0 := by
  have incomingMember : selection ∈
      get_rule.incoming center (accumulatedGraph center occurrences) := by
    rw [FlowSpec.incoming_eq_filter]
    unfold edgesBetween at member
    simp only [List.mem_filter] at member ⊢
    have endpoints : selection.START = u /\ selection.END = center :=
      of_decide_eq_true member.2
    exact ⟨member.1, by simp [endpoints.2]⟩
  rw [accumulated_incoming center occurrences usesAvoid] at incomingMember
  obtain ⟨occurrence, occurrenceMember, edgeMember⟩ :=
    List.mem_flatMap.mp incomingMember
  simp [ElimOccurrence.incoming] at edgeMember
  rcases edgeMember with rfl | rfl <;> rfl

theorem accumulated_major_edge_mem
    (center : Vertex) (occurrences : List ElimOccurrence)
    (occurrence : ElimOccurrence) (member : occurrence ∈ occurrences) :
    edge occurrence.major center 0 occurrence.majorStored ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.major center := by
  unfold edgesBetween accumulatedGraph
    ActualMixedProducerBridge.neighborhoodGraph accumulatedNeighborhood
  simp only [List.nil_append, List.mem_filter]
  constructor
  · apply List.mem_append_left
    exact List.mem_flatMap.mpr
      ⟨occurrence, member, by simp [ElimOccurrence.incoming]⟩
  · simp

theorem accumulated_minor_edge_mem
    (center : Vertex) (occurrences : List ElimOccurrence)
    (occurrence : ElimOccurrence) (member : occurrence ∈ occurrences) :
    edge occurrence.minor center 0 occurrence.minorStored ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.minor center := by
  unfold edgesBetween accumulatedGraph
    ActualMixedProducerBridge.neighborhoodGraph accumulatedNeighborhood
  simp only [List.nil_append, List.mem_filter]
  constructor
  · apply List.mem_append_left
    exact List.mem_flatMap.mpr
      ⟨occurrence, member, by simp [ElimOccurrence.incoming]⟩
  · simp

theorem mem_elimPairsAt_iff
    (d : Graph) (v major minor : Vertex) :
    (major, minor) ∈ elimPairsAt d v <->
      major ∈ predsOf d v /\ minor ∈ predsOf d v /\
        major.FORMULA = minor.FORMULA >> v.FORMULA := by
  unfold elimPairsAt
  simp only [List.mem_flatMap, List.mem_filterMap]
  constructor
  · rintro ⟨selectionMajor, majorMember, selectionMinor,
      minorMember, filtered⟩
    split at filtered
    · simp only [Option.some.injEq, Prod.mk.injEq] at filtered
      rcases filtered with ⟨majorEqual, minorEqual⟩
      subst selectionMajor
      subst selectionMinor
      exact ⟨majorMember, minorMember, ‹_›⟩
    · contradiction
  · rintro ⟨majorMember, minorMember, formula⟩
    exact ⟨major, majorMember, minor, minorMember, by simp [formula]⟩




theorem mem_accumulated_elimPairsAt_iff
    (center major minor : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences) :
    (major, minor) ∈
        elimPairsAt (accumulatedGraph center occurrences) center <->
      (Exists fun majorOccurrence =>
        majorOccurrence ∈ occurrences /\
          PremiseOf majorOccurrence major) /\
      (Exists fun minorOccurrence =>
        minorOccurrence ∈ occurrences /\
          PremiseOf minorOccurrence minor) /\
      major.FORMULA = minor.FORMULA >> center.FORMULA := by
  rw [mem_elimPairsAt_iff]
  rw [mem_accumulated_preds_iff center major occurrences usesAvoid]
  rw [mem_accumulated_preds_iff center minor occurrences usesAvoid]

private theorem same_occurrence_formula_selection
    (center major minor : Vertex) (occurrence : ElimOccurrence)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA)
    (majorPremise : PremiseOf occurrence major)
    (minorPremise : PremiseOf occurrence minor)
    (formula : major.FORMULA = minor.FORMULA >> center.FORMULA) :
    major = occurrence.major /\ minor = occurrence.minor := by
  rcases majorPremise with majorEq | majorEq <;>
    rcases minorPremise with minorEq | minorEq
  · subst major
    subst minor
    exact False.elim
      (FlowSpec.formula_ne_implication_self occurrence.major.FORMULA
        center.FORMULA formula)
  · exact ⟨majorEq, minorEq⟩
  · subst major
    subst minor
    exact False.elim (FlowSpec.not_mutual_implication logical formula)
  · subst major
    subst minor
    exact False.elim
      (FlowSpec.formula_ne_implication_self occurrence.minor.FORMULA
        center.FORMULA formula)




theorem accumulated_elimPair_classification
    (center major minor : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (selection : (major, minor) ∈
      elimPairsAt (accumulatedGraph center occurrences) center) :
    (Exists fun occurrence => occurrence ∈ occurrences /\
      major = occurrence.major /\ minor = occurrence.minor) \/
    (Exists fun left => Exists fun right =>
      left ∈ occurrences /\ right ∈ occurrences /\ left ≠ right /\
        PremiseOf left major /\ PremiseOf right minor) := by
  have characterized :=
    (mem_accumulated_elimPairsAt_iff center major minor occurrences
      usesAvoid).mp selection
  obtain ⟨⟨left, leftMember, leftPremise⟩,
    ⟨right, rightMember, rightPremise⟩, formula⟩ := characterized
  by_cases same : left = right
  · subst right
    have orientation := same_occurrence_formula_selection center major minor
      left (logical left leftMember) leftPremise rightPremise formula
    exact Or.inl ⟨left, leftMember, orientation⟩
  · exact Or.inr ⟨left, right, leftMember, rightMember, same,
      leftPremise, rightPremise⟩


theorem intended_elimPair_mem
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (occurrence : ElimOccurrence) (member : occurrence ∈ occurrences)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA) :
    (occurrence.major, occurrence.minor) ∈
      elimPairsAt (accumulatedGraph center occurrences) center := by
  apply (mem_accumulated_elimPairsAt_iff center occurrence.major
    occurrence.minor occurrences usesAvoid).mpr
  exact ⟨⟨occurrence, member, Or.inl rfl⟩,
    ⟨occurrence, member, Or.inr rfl⟩, logical⟩

theorem selectionEliminationOutputs_eq_pairOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair) :
    selectionEliminationOutputs d v premiseFlow =
      (elimPairsAt d v).flatMap
        (HCDerivedElimPairPreservation.eliminationPairOutputs
          d v premiseFlow) := by
  rfl

private theorem zero_pair_result_iff
    (minorEdge majorEdge : Deduction)
    (minorZero : minorEdge.COLOUR = 0)
    (majorZero : majorEdge.COLOUR = 0)
    (minorDep majorDep : Dep) (minorTag majorTag : Nat)
    (output : FlowPair) :
    (match FlowSpec.consume minorEdge [0, minorTag],
        FlowSpec.consume majorEdge [0, majorTag] with
      | some minorTail, some majorTail =>
          if minorTail = majorTail then
            some (depUnion minorDep majorDep, minorTail)
          else none
      | _, _ => none) = some output <->
      minorTag = majorTag /\
        output = (depUnion minorDep majorDep, [minorTag]) := by
  simp [FlowSpec.consume, minorZero, majorZero, eq_comm]

private theorem major_ne_minor_of_logical
    (center : Vertex) (occurrence : ElimOccurrence)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA) :
    occurrence.major ≠ occurrence.minor := by
  intro equal
  have formulaEqual := congrArg Vertex.FORMULA equal
  rw [logical] at formulaEqual
  exact FlowSpec.formula_ne_implication_self occurrence.minor.FORMULA
    center.FORMULA formulaEqual.symm




theorem selectionEliminationOutputs_sound
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (tagsInjective : TagsInjectiveOn occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (produced : output ∈ selectionEliminationOutputs
      (accumulatedGraph center occurrences) center premiseFlow) :
    output ∈ expectedFlowRoutes occurrences := by
  rw [selectionEliminationOutputs_eq_pairOutputs] at produced
  obtain ⟨pair, pairMember, pairProduced⟩ := List.mem_flatMap.mp produced
  rcases pair with ⟨major, minor⟩
  unfold HCDerivedElimPairPreservation.eliminationPairOutputs at pairProduced
  obtain ⟨minorRoute, minorRouteMember, afterMinor⟩ :=
    List.mem_flatMap.mp pairProduced
  obtain ⟨majorRoute, majorRouteMember, afterMajor⟩ :=
    List.mem_flatMap.mp afterMinor
  obtain ⟨minorEdge, minorEdgeMember, afterMinorEdge⟩ :=
    List.mem_flatMap.mp afterMajor
  obtain ⟨majorEdge, majorEdgeMember, resultEquation⟩ :=
    List.mem_filterMap.mp afterMinorEdge
  have minorExpected := (premiseCorrect minor minorRoute).mp minorRouteMember
  have majorExpected := (premiseCorrect major majorRoute).mp majorRouteMember
  rw [mem_expectedPremiseRoutes_iff] at minorExpected majorExpected
  obtain ⟨minorOccurrence, minorOccurrenceMember,
      minorOrigin⟩ := minorExpected
  obtain ⟨majorOccurrence, majorOccurrenceMember,
      majorOrigin⟩ := majorExpected
  have minorZero := accumulated_edgesBetween_colour_zero center minor
    occurrences usesAvoid minorEdge minorEdgeMember
  have majorZero := accumulated_edgesBetween_colour_zero center major
    occurrences usesAvoid majorEdge majorEdgeMember
  have pairFormula := (mem_elimPairsAt_iff
    (accumulatedGraph center occurrences) center major minor).mp pairMember |>.2.2
  rcases minorOrigin with minorOrigin | minorOrigin <;>
    rcases majorOrigin with majorOrigin | majorOrigin
  all_goals
    rcases minorOrigin with ⟨minorVertex, minorRouteEqual⟩
    rcases majorOrigin with ⟨majorVertex, majorRouteEqual⟩
    rw [minorRouteEqual, majorRouteEqual] at resultEquation
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.majorDep majorOccurrence.majorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_selection center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inl majorVertex) (Or.inl minorVertex) pairFormula
    have impossible := major_ne_minor_of_logical center minorOccurrence
      (logical minorOccurrence minorOccurrenceMember)
    exact False.elim (impossible (minorVertex.symm.trans orientation.2))
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.majorDep majorOccurrence.minorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_selection center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inr majorVertex) (Or.inl minorVertex) pairFormula
    have impossible := major_ne_minor_of_logical center minorOccurrence
      (logical minorOccurrence minorOccurrenceMember)
    exact False.elim (impossible (minorVertex.symm.trans orientation.2))
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.minorDep majorOccurrence.majorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_selection center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inl majorVertex) (Or.inr minorVertex) pairFormula
    rw [result.2]
    exact List.mem_map.mpr ⟨minorOccurrence, minorOccurrenceMember, rfl⟩
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.minorDep majorOccurrence.minorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_selection center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inr majorVertex) (Or.inr minorVertex) pairFormula
    have impossible := major_ne_minor_of_logical center minorOccurrence
      (logical minorOccurrence minorOccurrenceMember)
    exact False.elim (impossible (orientation.1.symm.trans majorVertex))



theorem selectionEliminationOutputs_complete
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (expected : output ∈ expectedFlowRoutes occurrences) :
    output ∈ selectionEliminationOutputs
      (accumulatedGraph center occurrences) center premiseFlow := by
  obtain ⟨occurrence, occurrenceMember, outputEqual⟩ :=
    List.mem_map.mp expected
  subst output
  have pairMember := intended_elimPair_mem center occurrences usesAvoid
    occurrence occurrenceMember (logical occurrence occurrenceMember)
  have majorExpected :
      (occurrence.majorDep, [0, occurrence.tag]) ∈
        expectedPremiseRoutes occurrences occurrence.major := by
    apply (mem_expectedPremiseRoutes_iff occurrences occurrence.major _).mpr
    exact ⟨occurrence, occurrenceMember, Or.inl ⟨rfl, rfl⟩⟩
  have minorExpected :
      (occurrence.minorDep, [0, occurrence.tag]) ∈
        expectedPremiseRoutes occurrences occurrence.minor := by
    apply (mem_expectedPremiseRoutes_iff occurrences occurrence.minor _).mpr
    exact ⟨occurrence, occurrenceMember, Or.inr ⟨rfl, rfl⟩⟩
  have majorRoute := (premiseCorrect occurrence.major _).mpr majorExpected
  have minorRoute := (premiseCorrect occurrence.minor _).mpr minorExpected
  let majorEdge := edge occurrence.major center 0 occurrence.majorStored
  let minorEdge := edge occurrence.minor center 0 occurrence.minorStored
  have majorEdgeMember : majorEdge ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.major center := by
    exact accumulated_major_edge_mem center occurrences occurrence
      occurrenceMember
  have minorEdgeMember : minorEdge ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.minor center := by
    exact accumulated_minor_edge_mem center occurrences occurrence
      occurrenceMember
  rw [selectionEliminationOutputs_eq_pairOutputs]
  apply List.mem_flatMap.mpr
  refine ⟨(occurrence.major, occurrence.minor), pairMember, ?_⟩
  unfold HCDerivedElimPairPreservation.eliminationPairOutputs
  apply List.mem_flatMap.mpr
  refine ⟨(occurrence.minorDep, [0, occurrence.tag]), minorRoute, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨(occurrence.majorDep, [0, occurrence.tag]), majorRoute, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨minorEdge, minorEdgeMember, ?_⟩
  apply List.mem_filterMap.mpr
  refine ⟨majorEdge, majorEdgeMember, ?_⟩
  exact (zero_pair_result_iff minorEdge majorEdge (by rfl) (by rfl)
    occurrence.minorDep occurrence.majorDep occurrence.tag occurrence.tag
    (depUnion occurrence.minorDep occurrence.majorDep,
      [occurrence.tag])).mpr ⟨rfl, rfl⟩




theorem selectionEliminationOutputs_setEq_expected
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (distinct : PairwiseDistinctTags occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (selectionEliminationOutputs
        (accumulatedGraph center occurrences) center premiseFlow)
      (expectedFlowRoutes occurrences) := by
  intro output
  constructor
  · exact selectionEliminationOutputs_sound center occurrences usesAvoid
      logical (pairwiseDistinctTags_injective occurrences distinct)
      premiseFlow premiseCorrect output
  · exact selectionEliminationOutputs_complete center occurrences usesAvoid
      logical premiseFlow premiseCorrect output

private theorem introductionSelection_has_occurrence_residual
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (member : output ∈ introductionSelections
      (accumulatedGraph center occurrences) center premiseFlow) :
    Exists fun occurrence => occurrence ∈ occurrences /\
      Exists fun dependency => output = (dependency, [occurrence.tag]) := by
  unfold introductionSelections at member
  cases formulaShape : center.FORMULA with
  | atom symbol => simp [formulaShape] at member
  | implication antecedent consequent =>
      simp only [formulaShape, List.mem_flatMap] at member
      obtain ⟨premise, premiseMember, selectionMember⟩ := member
      by_cases premiseFormula : premise.FORMULA = consequent
      · simp only [premiseFormula, ↓reduceIte] at selectionMember
        unfold selectionIntroductionOutputs at selectionMember
        rw [formulaShape] at selectionMember
        simp only [premiseFormula, ↓reduceIte] at selectionMember
        obtain ⟨premiseRoute, premiseRouteMember, afterRoute⟩ :=
          List.mem_flatMap.mp selectionMember
        obtain ⟨premiseEdge, premiseEdgeMember, resultEquation⟩ :=
          List.mem_filterMap.mp afterRoute
        have expectedPremise :=
          (premiseCorrect premise premiseRoute).mp premiseRouteMember
        rw [mem_expectedPremiseRoutes_iff] at expectedPremise
        obtain ⟨occurrence, occurrenceMember, origin⟩ := expectedPremise
        have edgeZero := accumulated_edgesBetween_colour_zero center premise
          occurrences usesAvoid premiseEdge premiseEdgeMember
        rcases origin with origin | origin
        · rcases origin with ⟨premiseEqual, routeEqual⟩
          rw [routeEqual] at resultEquation
          simp [FlowSpec.consume, edgeZero] at resultEquation
          exact ⟨occurrence, occurrenceMember,
            depRemove occurrence.majorDep antecedent, resultEquation.symm⟩
        · rcases origin with ⟨premiseEqual, routeEqual⟩
          rw [routeEqual] at resultEquation
          simp [FlowSpec.consume, edgeZero] at resultEquation
          exact ⟨occurrence, occurrenceMember,
            depRemove occurrence.minorDep antecedent, resultEquation.symm⟩
      · simp [premiseFormula] at selectionMember




theorem separatedIntroductionOutputs_eq_nil
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    separatedIntroductionOutputs (accumulatedGraph center occurrences)
      center premiseFlow = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro output outputMember
  obtain ⟨occurrence, occurrenceMember, dependency, outputEqual⟩ :=
    introductionSelection_has_occurrence_residual center occurrences usesAvoid
      premiseFlow premiseCorrect output outputMember
  have expectedMember :
      (depUnion occurrence.minorDep occurrence.majorDep,
        [occurrence.tag]) ∈ expectedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨occurrence, occurrenceMember, rfl⟩
  have eliminationMember := selectionEliminationOutputs_complete center
    occurrences usesAvoid logical premiseFlow premiseCorrect _ expectedMember
  have occurs : residualOccursB
      (selectionEliminationOutputs (accumulatedGraph center occurrences)
        center premiseFlow) output.2 = true := by
    unfold residualOccursB
    rw [List.any_eq_true]
    refine ⟨(depUnion occurrence.minorDep occurrence.majorDep,
      [occurrence.tag]), eliminationMember, ?_⟩
    simp [outputEqual]
  simp [separatedIntroductionOutputs, outputMember, occurs]

theorem derivedOutputs_setEq_expected
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (distinct : PairwiseDistinctTags occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (derivedOutputs (accumulatedGraph center occurrences)
        center premiseFlow)
      (expectedFlowRoutes occurrences) := by
  have eliminations := selectionEliminationOutputs_setEq_expected center
    occurrences usesAvoid logical distinct premiseFlow premiseCorrect
  have introductions := separatedIntroductionOutputs_eq_nil center
    occurrences usesAvoid logical premiseFlow premiseCorrect
  intro output
  rw [show derivedOutputs (accumulatedGraph center occurrences) center
      premiseFlow =
        (selectionEliminationOutputs (accumulatedGraph center occurrences)
          center premiseFlow).eraseDups by
    simp [derivedOutputs, introductions]]
  rw [List.mem_eraseDups]
  exact eliminations output





theorem accumulated_occurrenceStep
    (center : Vertex) (occurrences : List ElimOccurrence)
    (nonempty : occurrences ≠ [])
    (centerDerived : center.HYPOTHESIS = false)
    (premisesAvoid : PremisesAvoidCenter center occurrences)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (distinct : PairwiseDistinctTags occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    Exists fun outputs =>
      occurrenceStep (accumulatedGraph center occurrences) center
          premiseFlow = .ok outputs /\
        SelfCertifying.ListSetEq outputs
          (expectedFlowRoutes occurrences) := by
  have derivedSet := derivedOutputs_setEq_expected center occurrences
    usesAvoid logical distinct premiseFlow premiseCorrect
  obtain ⟨occurrence, rest, occurrenceList⟩ := List.exists_cons_of_ne_nil nonempty
  have occurrenceMember : occurrence ∈ occurrences := by
    rw [occurrenceList]
    exact List.Mem.head rest
  have predecessorMember : occurrence.major ∈
      predsOf (accumulatedGraph center occurrences) center := by
    apply (mem_accumulated_preds_iff center occurrence.major occurrences
      usesAvoid).mpr
    exact ⟨occurrence, occurrenceMember, Or.inl rfl⟩
  have predecessorsNonempty :
      (predsOf (accumulatedGraph center occurrences) center).isEmpty = false := by
    have ne : predsOf (accumulatedGraph center occurrences) center ≠ [] := by
      intro empty
      rw [empty] at predecessorMember
      exact List.not_mem_nil predecessorMember
    simpa [List.isEmpty_iff] using ne
  have noAncestors := accumulated_no_center_ancestors center occurrences
    premisesAvoid
  have expectedMember :
      (depUnion occurrence.minorDep occurrence.majorDep,
        [occurrence.tag]) ∈ expectedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨occurrence, occurrenceMember, rfl⟩
  have derivedMember :
      (depUnion occurrence.minorDep occurrence.majorDep,
        [occurrence.tag]) ∈
        derivedOutputs (accumulatedGraph center occurrences) center
          premiseFlow :=
    (derivedSet _).mpr expectedMember
  have derivedNonempty :
      (derivedOutputs (accumulatedGraph center occurrences) center
        premiseFlow).isEmpty = false := by
    have ne : derivedOutputs (accumulatedGraph center occurrences) center
        premiseFlow ≠ [] := by
      intro empty
      rw [empty] at derivedMember
      exact List.not_mem_nil derivedMember
    simpa [List.isEmpty_iff] using ne
  refine ⟨derivedOutputs (accumulatedGraph center occurrences) center
    premiseFlow, ?_, derivedSet⟩
  simp [occurrenceStep, predecessorsNonempty, noAncestors, centerDerived,
    derivedNonempty]





theorem AccumulatorInvariant.occurrenceStep
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (centerDerived : current.CENTER.HYPOTHESIS = false)
    (usesAvoid : UsesAvoidCenter current.CENTER processed)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow processed) :
    Exists fun outputs =>
      occurrenceStep
          (ActualMixedProducerBridge.neighborhoodGraph current)
          current.CENTER premiseFlow = .ok outputs /\
        SelfCertifying.ListSetEq outputs
          (expectedFlowRoutes processed) := by
  have premisesAvoid : PremisesAvoidCenter current.CENTER processed :=
    invariant.premisesAvoidCenter
  have logical : forall occurrence, occurrence ∈ processed ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> current.CENTER.FORMULA := by
    intro occurrence member
    simpa [invariant.centerFormula] using
      invariant.logicalEliminations occurrence member
  have operational := accumulated_occurrenceStep current.CENTER processed
    invariant.nonempty centerDerived premisesAvoid usesAvoid logical
    invariant.distinctTags premiseFlow premiseCorrect
  have graphEqual : accumulatedGraph current.CENTER processed =
      ActualMixedProducerBridge.neighborhoodGraph current := by
    unfold accumulatedGraph
    rw [← invariant.exactShape]
  rw [graphEqual] at operational
  exact operational



def eliminationSelectionRoutes (routes : List FlowPair) :
    List SelectionRoute :=
  routes.map fun route =>
    { production := .elimination
      dependency := route.1
      residual := { colours := route.2 } }

theorem expectedFlowRoutes_to_occurrenceRoutes
    (occurrences : List ElimOccurrence) :
    eliminationSelectionRoutes (expectedFlowRoutes occurrences) =
      occurrenceRoutes occurrences := by
  induction occurrences with
  | nil => rfl
  | cons occurrence rest ih =>
      simp [eliminationSelectionRoutes, expectedFlowRoutes,
        occurrenceRoutes, ElimOccurrence.route, ih]

theorem eliminationSelectionRoutes_congr
    {left right : List FlowPair}
    (equal : SelfCertifying.ListSetEq left right) :
    SelfCertifying.ListSetEq
      (eliminationSelectionRoutes left)
      (eliminationSelectionRoutes right) := by
  intro route
  simp only [eliminationSelectionRoutes, List.mem_map]
  constructor
  · rintro ⟨pair, pairMember, rfl⟩
    exact ⟨pair, (equal pair).mp pairMember, rfl⟩
  · rintro ⟨pair, pairMember, rfl⟩
    exact ⟨pair, (equal pair).mpr pairMember, rfl⟩

theorem boundaryDeliveries_routes_congr
    (conclusion : Formula) (edges : List Deduction)
    {left right : List SelectionRoute}
    (equal : SelfCertifying.ListSetEq left right) :
    SelfCertifying.ListSetEq
      (boundaryDeliveries conclusion left edges)
      (boundaryDeliveries conclusion right edges) := by
  unfold boundaryDeliveries
  exact SelfCertifying.flatMap_congr_list
    (fun route => edges.filterMap (observeAcross conclusion route)) equal




theorem AccumulatorInvariant.occurrenceStep_boundary
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (centerDerived : current.CENTER.HYPOTHESIS = false)
    (usesAvoid : UsesAvoidCenter current.CENTER processed)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow processed) :
    Exists fun outputs =>
      OccurrenceAwareReplay.occurrenceStep
          (ActualMixedProducerBridge.neighborhoodGraph current)
          current.CENTER premiseFlow = .ok outputs /\
        SelfCertifying.ListSetEq
          (boundaryDeliveries conclusion
            (eliminationSelectionRoutes outputs) current.OUTGOING)
          (observeEliminationBoundary dependencyFor current) := by
  obtain ⟨outputs, step, outputSet⟩ :=
    AccumulatorInvariant.occurrenceStep invariant centerDerived usesAvoid
      premiseFlow premiseCorrect
  have routeSet := eliminationSelectionRoutes_congr outputSet
  rw [expectedFlowRoutes_to_occurrenceRoutes] at routeSet
  have deliveredSet := boundaryDeliveries_routes_congr conclusion
    current.OUTGOING routeSet
  have expectedBoundary := invariant.operationalBoundary
  refine ⟨outputs, step, ?_⟩
  exact SelfCertifying.ListSetEq.trans deliveredSet
    (by simpa [expectedBoundary] using
      (SelfCertifying.ListSetEq.refl
        (observeEliminationBoundary dependencyFor current)))

end HCEliminationClassOperational
