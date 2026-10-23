import Semantic.HCEliminationClassCallerBridge











open Semantic FlowSpec
open CandidateMixedReplay OccurrenceAwareReplay
open HCEliminationClassPreservation HCEliminationClassOperational
open HCEliminationClassCallerBridge

namespace HCInterClassReplay



structure RoutedElimOccurrence where
  logical : ElimOccurrence
  majorColour : Nat
  minorColour : Nat
deriving Repr, DecidableEq

def expectedRoutedFlowRoutes
    (occurrences : List RoutedElimOccurrence) : List FlowPair :=
  occurrences.map fun routed =>
    (depUnion routed.logical.minorDep routed.logical.majorDep,
      [routed.logical.tag])

def expectedRoutedPremiseRoutes
    (occurrences : List RoutedElimOccurrence) (vertex : Vertex) :
    List FlowPair :=
  occurrences.flatMap fun routed =>
    (if vertex = routed.logical.major then
      [(routed.logical.majorDep,
        [routed.majorColour, routed.logical.tag])]
    else []) ++
    (if vertex = routed.logical.minor then
      [(routed.logical.minorDep,
        [routed.minorColour, routed.logical.tag])]
    else [])

def RoutedPremiseRoutesCorrect (premiseFlow : Vertex -> List FlowPair)
    (occurrences : List RoutedElimOccurrence) : Prop :=
  forall vertex, SelfCertifying.ListSetEq (premiseFlow vertex)
    (expectedRoutedPremiseRoutes occurrences vertex)

def RoutedPremiseOf (routed : RoutedElimOccurrence) (vertex : Vertex) : Prop :=
  vertex = routed.logical.major \/ vertex = routed.logical.minor





theorem mem_expectedRoutedPremiseRoutes_iff
    (occurrences : List RoutedElimOccurrence) (vertex : Vertex)
    (route : FlowPair) :
    route ∈ expectedRoutedPremiseRoutes occurrences vertex ↔
      ∃ routed, routed ∈ occurrences ∧
        ((vertex = routed.logical.major ∧
            route = (routed.logical.majorDep,
              [routed.majorColour, routed.logical.tag])) \/
         (vertex = routed.logical.minor ∧
            route = (routed.logical.minorDep,
              [routed.minorColour, routed.logical.tag]))) := by
  simp [expectedRoutedPremiseRoutes, and_or_left]






structure RoutedEliminationIncidence (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence) : Prop where
  predecessorSupport : ∀ vertex,
    vertex ∈ predsOf actual center ↔
      ∃ routed, routed ∈ occurrences ∧ RoutedPremiseOf routed vertex
  logical : ∀ routed, routed ∈ occurrences →
    routed.logical.major.FORMULA =
      routed.logical.minor.FORMULA >> center.FORMULA
  tagsIdentifyLogical : ∀ left, left ∈ occurrences →
    ∀ right, right ∈ occurrences →
      left.logical.tag = right.logical.tag → left.logical = right.logical
  majorEdge : ∀ routed, routed ∈ occurrences →
    ∃ actualEdge,
      actualEdge ∈ edgesBetween actual routed.logical.major center ∧
      actualEdge.COLOUR = routed.majorColour
  minorEdge : ∀ routed, routed ∈ occurrences →
    ∃ actualEdge,
      actualEdge ∈ edgesBetween actual routed.logical.minor center ∧
      actualEdge.COLOUR = routed.minorColour




def routedReplacementIncoming (center : Vertex)
    (occurrences : List RoutedElimOccurrence) : List Deduction :=
  occurrences.flatMap fun routed =>
    [edge routed.logical.major center routed.majorColour
        routed.logical.majorStored,
     edge routed.logical.minor center routed.minorColour
        routed.logical.minorStored]




structure IncomingEdgeSkeleton where
  start : Vertex
  finish : Vertex
  colour : Nat
deriving Repr, DecidableEq

def incomingEdgeSkeleton (actualEdge : Deduction) : IncomingEdgeSkeleton where
  start := actualEdge.START
  finish := actualEdge.END
  colour := actualEdge.COLOUR

def routedReplacementIncomingSkeleton (center : Vertex)
    (occurrences : List RoutedElimOccurrence) : List IncomingEdgeSkeleton :=
  (routedReplacementIncoming center occurrences).map incomingEdgeSkeleton



structure ExactRoutedReplacementIncidence (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence) : Prop where
  incomingSkeletonPerm : List.Perm
    ((get_rule.incoming center actual).map incomingEdgeSkeleton)
    (routedReplacementIncomingSkeleton center occurrences)
  logical : ∀ routed, routed ∈ occurrences →
    routed.logical.major.FORMULA =
      routed.logical.minor.FORMULA >> center.FORMULA
  tagsIdentifyLogical : ∀ left, left ∈ occurrences →
    ∀ right, right ∈ occurrences →
      left.logical.tag = right.logical.tag → left.logical = right.logical

theorem ExactRoutedReplacementIncidence.toRoutedEliminationIncidence
    {actual : Graph} {center : Vertex}
    {occurrences : List RoutedElimOccurrence}
    (incidence : ExactRoutedReplacementIncidence actual center occurrences) :
    RoutedEliminationIncidence actual center occurrences := by
  refine {
    predecessorSupport := ?_
    logical := incidence.logical
    tagsIdentifyLogical := incidence.tagsIdentifyLogical
    majorEdge := ?_
    minorEdge := ?_ }
  · intro vertex
    rw [FlowSpec.predsOf_eq, List.mem_eraseDups, List.mem_map]
    constructor
    · rintro ⟨actualEdge, actualEdgeMember, startEqual⟩
      have skeletonMember : incomingEdgeSkeleton actualEdge ∈
          routedReplacementIncomingSkeleton center occurrences := by
        exact incidence.incomingSkeletonPerm.mem_iff.mp
          (List.mem_map.mpr ⟨actualEdge, actualEdgeMember, rfl⟩)
      obtain ⟨expectedEdge, expectedEdgeMember, skeletonEqual⟩ :=
        List.mem_map.mp skeletonMember
      obtain ⟨routed, routedMember, localMember⟩ :=
        List.mem_flatMap.mp expectedEdgeMember
      rcases (by simpa using localMember) with rfl | rfl
      · have starts := congrArg IncomingEdgeSkeleton.start skeletonEqual
        exact ⟨routed, routedMember,
          Or.inl (startEqual.symm.trans starts.symm)⟩
      · have starts := congrArg IncomingEdgeSkeleton.start skeletonEqual
        exact ⟨routed, routedMember,
          Or.inr (startEqual.symm.trans starts.symm)⟩
    · rintro ⟨routed, routedMember, major | minor⟩
      · let expectedEdge := edge routed.logical.major center
          routed.majorColour routed.logical.majorStored
        have expectedMember : incomingEdgeSkeleton expectedEdge ∈
            routedReplacementIncomingSkeleton center occurrences := by
          apply List.mem_map.mpr
          exact ⟨expectedEdge, List.mem_flatMap.mpr
            ⟨routed, routedMember, by simp [expectedEdge]⟩, rfl⟩
        have expectedMember :=
          incidence.incomingSkeletonPerm.mem_iff.mpr expectedMember
        obtain ⟨actualEdge, actualEdgeMember, skeletonEqual⟩ :=
          List.mem_map.mp expectedMember
        refine ⟨actualEdge, actualEdgeMember, ?_⟩
        have starts := congrArg IncomingEdgeSkeleton.start skeletonEqual
        simpa [expectedEdge, major] using starts
      · let expectedEdge := edge routed.logical.minor center
          routed.minorColour routed.logical.minorStored
        have expectedMember : incomingEdgeSkeleton expectedEdge ∈
            routedReplacementIncomingSkeleton center occurrences := by
          apply List.mem_map.mpr
          exact ⟨expectedEdge, List.mem_flatMap.mpr
            ⟨routed, routedMember, by simp [expectedEdge]⟩, rfl⟩
        have expectedMember :=
          incidence.incomingSkeletonPerm.mem_iff.mpr expectedMember
        obtain ⟨actualEdge, actualEdgeMember, skeletonEqual⟩ :=
          List.mem_map.mp expectedMember
        refine ⟨actualEdge, actualEdgeMember, ?_⟩
        have starts := congrArg IncomingEdgeSkeleton.start skeletonEqual
        simpa [expectedEdge, minor] using starts
  · intro routed routedMember
    let expectedEdge := edge routed.logical.major center routed.majorColour
      routed.logical.majorStored
    have expectedMember : incomingEdgeSkeleton expectedEdge ∈
        routedReplacementIncomingSkeleton center occurrences := by
      apply List.mem_map.mpr
      exact ⟨expectedEdge, List.mem_flatMap.mpr
        ⟨routed, routedMember, by simp [expectedEdge]⟩, rfl⟩
    have expectedMember :=
      incidence.incomingSkeletonPerm.mem_iff.mpr expectedMember
    obtain ⟨actualEdge, actualEdgeMember, skeletonEqual⟩ :=
      List.mem_map.mp expectedMember
    refine ⟨actualEdge, ?_, ?_⟩
    rw [HorizontalCompressionInput.edgesBetween_eq_incoming]
    apply List.mem_filter.mpr
    constructor
    · exact actualEdgeMember
    · have starts := congrArg IncomingEdgeSkeleton.start skeletonEqual
      simpa [expectedEdge] using starts
    · have colours := congrArg IncomingEdgeSkeleton.colour skeletonEqual
      simpa [expectedEdge] using colours
  · intro routed routedMember
    let expectedEdge := edge routed.logical.minor center routed.minorColour
      routed.logical.minorStored
    have expectedMember : incomingEdgeSkeleton expectedEdge ∈
        routedReplacementIncomingSkeleton center occurrences := by
      apply List.mem_map.mpr
      exact ⟨expectedEdge, List.mem_flatMap.mpr
        ⟨routed, routedMember, by simp [expectedEdge]⟩, rfl⟩
    have expectedMember :=
      incidence.incomingSkeletonPerm.mem_iff.mpr expectedMember
    obtain ⟨actualEdge, actualEdgeMember, skeletonEqual⟩ :=
      List.mem_map.mp expectedMember
    refine ⟨actualEdge, ?_, ?_⟩
    rw [HorizontalCompressionInput.edgesBetween_eq_incoming]
    apply List.mem_filter.mpr
    constructor
    · exact actualEdgeMember
    · have starts := congrArg IncomingEdgeSkeleton.start skeletonEqual
      simpa [expectedEdge] using starts
    · have colours := congrArg IncomingEdgeSkeleton.colour skeletonEqual
      simpa [expectedEdge] using colours

private theorem routed_same_occurrence_formula_candidate
    (center major minor : Vertex) (occurrence : ElimOccurrence)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA)
    (majorPremise : PremiseOf occurrence major)
    (minorPremise : PremiseOf occurrence minor)
    (formula : major.FORMULA = minor.FORMULA >> center.FORMULA) :
    major = occurrence.major ∧ minor = occurrence.minor := by
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

private theorem routed_major_ne_minor
    (center : Vertex) (occurrence : ElimOccurrence)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA) :
    occurrence.major ≠ occurrence.minor := by
  intro equal
  have formulaEqual := congrArg Vertex.FORMULA equal
  rw [logical] at formulaEqual
  exact FlowSpec.formula_ne_implication_self occurrence.minor.FORMULA
    center.FORMULA formulaEqual.symm




theorem coloured_elimination_result_iff
    (minorEdge majorEdge : Deduction)
    (minorColour majorColour : Nat)
    (minorDep majorDep : Dep) (minorTag majorTag : Nat)
    (output : FlowPair) :
    (match FlowSpec.consume minorEdge [minorColour, minorTag],
        FlowSpec.consume majorEdge [majorColour, majorTag] with
      | some minorTail, some majorTail =>
          if minorTail = majorTail then
            some (depUnion minorDep majorDep, minorTail)
          else none
      | _, _ => none) = some output ↔
      minorEdge.COLOUR = minorColour ∧
      majorEdge.COLOUR = majorColour ∧
      minorTag = majorTag ∧
      output = (depUnion minorDep majorDep, [minorTag]) := by
  by_cases minorMatches : minorEdge.COLOUR = minorColour
  · by_cases majorMatches : majorEdge.COLOUR = majorColour
    · simp [FlowSpec.consume, minorMatches, majorMatches, eq_comm]
    · simp [FlowSpec.consume, minorMatches, majorMatches]
  · simp [FlowSpec.consume, minorMatches]

theorem coloured_elimination_rejects_distinct_tags
    (minorEdge majorEdge : Deduction)
    (minorColour majorColour : Nat)
    (minorDep majorDep : Dep) (minorTag majorTag : Nat)
    (minorMatches : minorEdge.COLOUR = minorColour)
    (majorMatches : majorEdge.COLOUR = majorColour)
    (differentUses : minorTag ≠ majorTag) :
    (match FlowSpec.consume minorEdge [minorColour, minorTag],
        FlowSpec.consume majorEdge [majorColour, majorTag] with
      | some minorTail, some majorTail =>
          if minorTail = majorTail then
            some (depUnion minorDep majorDep, minorTail)
          else none
      | _, _ => none) = none := by
  simp [FlowSpec.consume, minorMatches, majorMatches, differentUses]






theorem mem_reseedDerivedRoutes_iff
    (ancestors : List Ancestral) (routes : List FlowPair)
    (output : FlowPair) :
    output ∈ reseedDerivedRoutes ancestors routes ↔
      ∃ ancestor, ancestor ∈ ancestors ∧
        ∃ route, route ∈ routes ∧
          colourPathPrefixB route.2 ancestor.COLOURS = true ∧
          (route.1, ancestor.COLOURS) = output := by
  simp [reseedDerivedRoutes]





theorem candidateEliminationOutputs_sound_routed
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (produced : output ∈
      candidateEliminationOutputs actual center premiseFlow) :
    output ∈ expectedRoutedFlowRoutes occurrences := by
  rw [candidateEliminationOutputs_eq_pairOutputs] at produced
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
  rw [mem_expectedRoutedPremiseRoutes_iff] at minorExpected majorExpected
  obtain ⟨minorOwner, minorOwnerMember, minorOrigin⟩ := minorExpected
  obtain ⟨majorOwner, majorOwnerMember, majorOrigin⟩ := majorExpected
  have pairFormula :=
    (mem_elimPairsAt_iff actual center major minor).mp pairMember |>.2.2
  rcases minorOrigin with minorOrigin | minorOrigin <;>
    rcases majorOrigin with majorOrigin | majorOrigin
  all_goals
    rcases minorOrigin with ⟨minorVertex, minorRouteEqual⟩
    rcases majorOrigin with ⟨majorVertex, majorRouteEqual⟩
    rw [minorRouteEqual, majorRouteEqual] at resultEquation
  · have result := (coloured_elimination_result_iff minorEdge majorEdge
        minorOwner.majorColour majorOwner.majorColour
        minorOwner.logical.majorDep majorOwner.logical.majorDep
        minorOwner.logical.tag majorOwner.logical.tag output).mp resultEquation
    have sameLogical := incidence.tagsIdentifyLogical minorOwner
      minorOwnerMember majorOwner majorOwnerMember result.2.2.1
    rw [← sameLogical] at majorVertex result
    have orientation := routed_same_occurrence_formula_candidate center major
      minor minorOwner.logical (incidence.logical minorOwner minorOwnerMember)
      (Or.inl majorVertex) (Or.inl minorVertex) pairFormula
    have impossible := routed_major_ne_minor center minorOwner.logical
      (incidence.logical minorOwner minorOwnerMember)
    exact False.elim (impossible (minorVertex.symm.trans orientation.2))
  · have result := (coloured_elimination_result_iff minorEdge majorEdge
        minorOwner.majorColour majorOwner.minorColour
        minorOwner.logical.majorDep majorOwner.logical.minorDep
        minorOwner.logical.tag majorOwner.logical.tag output).mp resultEquation
    have sameLogical := incidence.tagsIdentifyLogical minorOwner
      minorOwnerMember majorOwner majorOwnerMember result.2.2.1
    rw [← sameLogical] at majorVertex result
    have orientation := routed_same_occurrence_formula_candidate center major
      minor minorOwner.logical (incidence.logical minorOwner minorOwnerMember)
      (Or.inr majorVertex) (Or.inl minorVertex) pairFormula
    have impossible := routed_major_ne_minor center minorOwner.logical
      (incidence.logical minorOwner minorOwnerMember)
    exact False.elim (impossible (minorVertex.symm.trans orientation.2))
  · have result := (coloured_elimination_result_iff minorEdge majorEdge
        minorOwner.minorColour majorOwner.majorColour
        minorOwner.logical.minorDep majorOwner.logical.majorDep
        minorOwner.logical.tag majorOwner.logical.tag output).mp resultEquation
    have sameLogical := incidence.tagsIdentifyLogical minorOwner
      minorOwnerMember majorOwner majorOwnerMember result.2.2.1
    rw [← sameLogical] at majorVertex result
    have orientation := routed_same_occurrence_formula_candidate center major
      minor minorOwner.logical (incidence.logical minorOwner minorOwnerMember)
      (Or.inl majorVertex) (Or.inr minorVertex) pairFormula
    rw [result.2.2.2]
    exact List.mem_map.mpr ⟨minorOwner, minorOwnerMember, rfl⟩
  · have result := (coloured_elimination_result_iff minorEdge majorEdge
        minorOwner.minorColour majorOwner.minorColour
        minorOwner.logical.minorDep majorOwner.logical.minorDep
        minorOwner.logical.tag majorOwner.logical.tag output).mp resultEquation
    have sameLogical := incidence.tagsIdentifyLogical minorOwner
      minorOwnerMember majorOwner majorOwnerMember result.2.2.1
    rw [← sameLogical] at majorVertex result
    have orientation := routed_same_occurrence_formula_candidate center major
      minor minorOwner.logical (incidence.logical minorOwner minorOwnerMember)
      (Or.inr majorVertex) (Or.inr minorVertex) pairFormula
    have impossible := routed_major_ne_minor center minorOwner.logical
      (incidence.logical minorOwner minorOwnerMember)
    exact False.elim (impossible (orientation.1.symm.trans majorVertex))




theorem candidateEliminationOutputs_complete_routed
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (expected : output ∈ expectedRoutedFlowRoutes occurrences) :
    output ∈ candidateEliminationOutputs actual center premiseFlow := by
  obtain ⟨routed, routedMember, outputEqual⟩ := List.mem_map.mp expected
  subst output
  have pairMember : (routed.logical.major, routed.logical.minor) ∈
      elimPairsAt actual center := by
    apply (mem_elimPairsAt_iff actual center routed.logical.major
      routed.logical.minor).mpr
    exact ⟨(incidence.predecessorSupport routed.logical.major).mpr
        ⟨routed, routedMember, Or.inl rfl⟩,
      (incidence.predecessorSupport routed.logical.minor).mpr
        ⟨routed, routedMember, Or.inr rfl⟩,
      incidence.logical routed routedMember⟩
  have majorExpected :
      (routed.logical.majorDep,
        [routed.majorColour, routed.logical.tag]) ∈
      expectedRoutedPremiseRoutes occurrences routed.logical.major := by
    apply (mem_expectedRoutedPremiseRoutes_iff occurrences
      routed.logical.major _).mpr
    exact ⟨routed, routedMember, Or.inl ⟨rfl, rfl⟩⟩
  have minorExpected :
      (routed.logical.minorDep,
        [routed.minorColour, routed.logical.tag]) ∈
      expectedRoutedPremiseRoutes occurrences routed.logical.minor := by
    apply (mem_expectedRoutedPremiseRoutes_iff occurrences
      routed.logical.minor _).mpr
    exact ⟨routed, routedMember, Or.inr ⟨rfl, rfl⟩⟩
  have majorRoute := (premiseCorrect routed.logical.major _).mpr majorExpected
  have minorRoute := (premiseCorrect routed.logical.minor _).mpr minorExpected
  obtain ⟨majorEdge, majorEdgeMember, majorColour⟩ :=
    incidence.majorEdge routed routedMember
  obtain ⟨minorEdge, minorEdgeMember, minorColour⟩ :=
    incidence.minorEdge routed routedMember
  rw [candidateEliminationOutputs_eq_pairOutputs]
  apply List.mem_flatMap.mpr
  refine ⟨(routed.logical.major, routed.logical.minor), pairMember, ?_⟩
  unfold HCDerivedElimPairPreservation.eliminationPairOutputs
  apply List.mem_flatMap.mpr
  refine ⟨(routed.logical.minorDep,
    [routed.minorColour, routed.logical.tag]), minorRoute, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨(routed.logical.majorDep,
    [routed.majorColour, routed.logical.tag]), majorRoute, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨minorEdge, minorEdgeMember, ?_⟩
  apply List.mem_filterMap.mpr
  refine ⟨majorEdge, majorEdgeMember, ?_⟩
  exact (coloured_elimination_result_iff minorEdge majorEdge
    routed.minorColour routed.majorColour routed.logical.minorDep
    routed.logical.majorDep routed.logical.tag routed.logical.tag
    (depUnion routed.logical.minorDep routed.logical.majorDep,
      [routed.logical.tag])).mpr
      ⟨minorColour, majorColour, rfl, rfl⟩




theorem candidateEliminationOutputs_setEq_expected_routed
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (candidateEliminationOutputs actual center premiseFlow)
      (expectedRoutedFlowRoutes occurrences) := by
  intro output
  exact ⟨candidateEliminationOutputs_sound_routed actual center occurrences
      incidence premiseFlow premiseCorrect output,
    candidateEliminationOutputs_complete_routed actual center occurrences
      incidence premiseFlow premiseCorrect output⟩



theorem candidateEliminationOutputs_setEq_expected_of_exactReplacement
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : ExactRoutedReplacementIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (candidateEliminationOutputs actual center premiseFlow)
      (expectedRoutedFlowRoutes occurrences) :=
  candidateEliminationOutputs_setEq_expected_routed actual center occurrences
    incidence.toRoutedEliminationIncidence premiseFlow premiseCorrect




def zeroRoutedOccurrence (occurrence : ElimOccurrence) :
    RoutedElimOccurrence where
  logical := occurrence
  majorColour := 0
  minorColour := 0

def zeroRoutedOccurrences (occurrences : List ElimOccurrence) :
    List RoutedElimOccurrence := occurrences.map zeroRoutedOccurrence

theorem routedReplacementIncoming_zero
    (center : Vertex) (occurrences : List ElimOccurrence) :
    routedReplacementIncoming center (zeroRoutedOccurrences occurrences) =
      occurrences.flatMap (ElimOccurrence.incoming center) := by
  induction occurrences with
  | nil => rfl
  | cons occurrence rest ih =>
      change ElimOccurrence.incoming center occurrence ++
          routedReplacementIncoming center (zeroRoutedOccurrences rest) =
        ElimOccurrence.incoming center occurrence ++
          rest.flatMap (ElimOccurrence.incoming center)
      rw [ih]

theorem accumulated_exact_routed_replacement
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : ∀ occurrence, occurrence ∈ occurrences →
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (tagsInjective : TagsInjectiveOn occurrences) :
    ExactRoutedReplacementIncidence
      (accumulatedGraph center occurrences) center
      (zeroRoutedOccurrences occurrences) := by
  refine {
    incomingSkeletonPerm := ?_
    logical := ?_
    tagsIdentifyLogical := ?_ }
  · rw [accumulated_incoming center occurrences usesAvoid]
    rw [routedReplacementIncomingSkeleton,
      routedReplacementIncoming_zero]
  · intro routed routedMember
    obtain ⟨occurrence, occurrenceMember, rfl⟩ :=
      List.mem_map.mp routedMember
    exact logical occurrence occurrenceMember
  · intro left leftMember right rightMember tagEqual
    obtain ⟨leftLogical, leftLogicalMember, rfl⟩ :=
      List.mem_map.mp leftMember
    obtain ⟨rightLogical, rightLogicalMember, rfl⟩ :=
      List.mem_map.mp rightMember
    have equal := tagsInjective leftLogical leftLogicalMember rightLogical
      rightLogicalMember tagEqual
    cases equal
    rfl




theorem candidateEliminationOutputs_setEq_expected_zeroSpecialization
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : ∀ occurrence, occurrence ∈ occurrences →
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (tagsInjective : TagsInjectiveOn occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (candidateEliminationOutputs
        (accumulatedGraph center occurrences) center premiseFlow)
      (expectedFlowRoutes occurrences) := by
  have routedCorrect : RoutedPremiseRoutesCorrect premiseFlow
      (zeroRoutedOccurrences occurrences) := by
    intro vertex
    simpa [zeroRoutedOccurrences, zeroRoutedOccurrence,
      expectedRoutedPremiseRoutes, expectedPremiseRoutes,
      List.flatMap_map] using premiseCorrect vertex
  have generalized :=
    candidateEliminationOutputs_setEq_expected_of_exactReplacement
      (accumulatedGraph center occurrences) center
      (zeroRoutedOccurrences occurrences)
      (accumulated_exact_routed_replacement center occurrences usesAvoid
        logical tagsInjective)
      premiseFlow routedCorrect
  simpa [zeroRoutedOccurrences, zeroRoutedOccurrence,
    expectedRoutedFlowRoutes, expectedFlowRoutes] using generalized

private theorem coloured_introduction_result_iff
    (actualEdge : Deduction) (colour tag : Nat)
    (dependency : Dep) (antecedent : Formula) (output : FlowPair) :
    (FlowSpec.consume actualEdge [colour, tag]).map
        (fun tail => (depRemove dependency antecedent, tail)) = some output ↔
      actualEdge.COLOUR = colour ∧
        output = (depRemove dependency antecedent, [tag]) := by
  by_cases colourMatches : actualEdge.COLOUR = colour
  · simp [FlowSpec.consume, colourMatches, eq_comm]
  · simp [FlowSpec.consume, colourMatches]

private theorem introductionCandidate_has_routed_residual
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (member : output ∈ introductionCandidates actual center premiseFlow) :
    ∃ routed, routed ∈ occurrences ∧
      ∃ dependency, output = (dependency, [routed.logical.tag]) := by
  unfold introductionCandidates at member
  cases formulaShape : center.FORMULA with
  | atom symbol => simp [formulaShape] at member
  | implication antecedent consequent =>
      simp only [formulaShape, List.mem_flatMap] at member
      obtain ⟨premise, premiseMember, candidateMember⟩ := member
      by_cases premiseFormula : premise.FORMULA = consequent
      · simp only [premiseFormula, ↓reduceIte] at candidateMember
        unfold candidateIntroductionOutputs at candidateMember
        rw [formulaShape] at candidateMember
        simp only [premiseFormula, ↓reduceIte] at candidateMember
        obtain ⟨premiseRoute, premiseRouteMember, afterRoute⟩ :=
          List.mem_flatMap.mp candidateMember
        obtain ⟨premiseEdge, premiseEdgeMember, resultEquation⟩ :=
          List.mem_filterMap.mp afterRoute
        have expectedPremise :=
          (premiseCorrect premise premiseRoute).mp premiseRouteMember
        rw [mem_expectedRoutedPremiseRoutes_iff] at expectedPremise
        obtain ⟨routed, routedMember, origin⟩ := expectedPremise
        rcases origin with origin | origin
        · rcases origin with ⟨premiseEqual, routeEqual⟩
          rw [routeEqual] at resultEquation
          have result := (coloured_introduction_result_iff premiseEdge
            routed.majorColour routed.logical.tag
            routed.logical.majorDep antecedent output).mp resultEquation
          exact ⟨routed, routedMember,
            depRemove routed.logical.majorDep antecedent, result.2⟩
        · rcases origin with ⟨premiseEqual, routeEqual⟩
          rw [routeEqual] at resultEquation
          have result := (coloured_introduction_result_iff premiseEdge
            routed.minorColour routed.logical.tag
            routed.logical.minorDep antecedent output).mp resultEquation
          exact ⟨routed, routedMember,
            depRemove routed.logical.minorDep antecedent, result.2⟩
      · simp [premiseFormula] at candidateMember





theorem separatedIntroductionOutputs_eq_nil_routed
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    separatedIntroductionOutputs actual center premiseFlow = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro output outputMember
  obtain ⟨routed, routedMember, dependency, outputEqual⟩ :=
    introductionCandidate_has_routed_residual actual center occurrences
      premiseFlow premiseCorrect output outputMember
  have expectedMember :
      (depUnion routed.logical.minorDep routed.logical.majorDep,
        [routed.logical.tag]) ∈ expectedRoutedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨routed, routedMember, rfl⟩
  have eliminationMember := candidateEliminationOutputs_complete_routed
    actual center occurrences incidence premiseFlow premiseCorrect _
    expectedMember
  have occurs : residualOccursB
      (candidateEliminationOutputs actual center premiseFlow)
        output.2 = true := by
    unfold residualOccursB
    rw [List.any_eq_true]
    refine ⟨(depUnion routed.logical.minorDep routed.logical.majorDep,
      [routed.logical.tag]), eliminationMember, ?_⟩
    simp [outputEqual]
  simp [occurs]



theorem derivedOutputs_setEq_expected_routed
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (derivedOutputs actual center premiseFlow)
      (expectedRoutedFlowRoutes occurrences) := by
  have eliminations := candidateEliminationOutputs_setEq_expected_routed
    actual center occurrences incidence premiseFlow premiseCorrect
  have introductions := separatedIntroductionOutputs_eq_nil_routed actual
    center occurrences incidence premiseFlow premiseCorrect
  intro output
  rw [show derivedOutputs actual center premiseFlow =
      (candidateEliminationOutputs actual center premiseFlow).eraseDups by
    simp [derivedOutputs, introductions]]
  rw [List.mem_eraseDups]
  exact eliminations output





theorem occurrenceStep_setEq_expected_routed
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (nonempty : occurrences ≠ [])
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (centerDerived : center.HYPOTHESIS = false)
    (noAncestors : ancestorsInto actual center = [])
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    ∃ outputs,
      occurrenceStep actual center premiseFlow = .ok outputs ∧
      SelfCertifying.ListSetEq outputs
        (expectedRoutedFlowRoutes occurrences) := by
  have derivedSet := derivedOutputs_setEq_expected_routed actual center
    occurrences incidence premiseFlow premiseCorrect
  obtain ⟨routed, rest, occurrenceList⟩ := List.exists_cons_of_ne_nil nonempty
  have routedMember : routed ∈ occurrences := by
    rw [occurrenceList]
    exact List.Mem.head rest
  have predecessorMember : routed.logical.major ∈ predsOf actual center :=
    (incidence.predecessorSupport routed.logical.major).mpr
      ⟨routed, routedMember, Or.inl rfl⟩
  have predecessorsNonempty : (predsOf actual center).isEmpty = false := by
    have ne : predsOf actual center ≠ [] := by
      intro empty
      rw [empty] at predecessorMember
      exact List.not_mem_nil predecessorMember
    simpa [List.isEmpty_iff] using ne
  have expectedMember :
      (depUnion routed.logical.minorDep routed.logical.majorDep,
        [routed.logical.tag]) ∈ expectedRoutedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨routed, routedMember, rfl⟩
  have derivedMember :
      (depUnion routed.logical.minorDep routed.logical.majorDep,
        [routed.logical.tag]) ∈ derivedOutputs actual center premiseFlow :=
    (derivedSet _).mpr expectedMember
  have derivedNonempty :
      (derivedOutputs actual center premiseFlow).isEmpty = false := by
    have ne : derivedOutputs actual center premiseFlow ≠ [] := by
      intro empty
      rw [empty] at derivedMember
      exact List.not_mem_nil derivedMember
    simpa [List.isEmpty_iff] using ne
  refine ⟨derivedOutputs actual center premiseFlow, ?_, derivedSet⟩
  simp [occurrenceStep, predecessorsNonempty, noAncestors, centerDerived,
    derivedNonempty]

theorem derivedOutputs_setEq_expected_of_exactReplacement
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (incidence : ExactRoutedReplacementIncidence actual center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    SelfCertifying.ListSetEq
      (derivedOutputs actual center premiseFlow)
      (expectedRoutedFlowRoutes occurrences) :=
  derivedOutputs_setEq_expected_routed actual center occurrences
    incidence.toRoutedEliminationIncidence premiseFlow premiseCorrect

def RoutedSourcePremiseEvidence (actual : Graph)
    (occurrences : List RoutedElimOccurrence) (_fuel : Nat)
    (vertex : Vertex) : Prop :=
  Exists fun traces : List Ancestral =>
    predsOf actual vertex = [] /\
    vertex.HYPOTHESIS = true /\
    vertex.COLLAPSED = false /\
    ancestorsInto actual vertex = traces /\
    traces ≠ [] /\
    SelfCertifying.ListSetEq
      (tracedRoutes (normalizeDep [vertex.FORMULA]) traces)
      (expectedRoutedPremiseRoutes occurrences vertex)

def RoutedLogicalPremiseEvidence (actual : Graph)
    (occurrences : List RoutedElimOccurrence) (fuel : Nat)
    (vertex : Vertex) : Prop :=
  Exists fun logicalRoutes : List FlowPair => Exists fun traces : List Ancestral =>
    DerivedLogicalReplayFrame actual fuel vertex logicalRoutes traces /\
    SelfCertifying.ListSetEq
      (reseedDerivedRoutes traces logicalRoutes)
      (expectedRoutedPremiseRoutes occurrences vertex)

inductive RoutedPremiseEvidence (actual : Graph)
    (occurrences : List RoutedElimOccurrence) (fuel : Nat)
    (vertex : Vertex) : Prop
  | source
      (evidence : RoutedSourcePremiseEvidence actual occurrences fuel vertex)
  | logicalDerived
      (evidence : RoutedLogicalPremiseEvidence actual occurrences fuel vertex)

theorem RoutedPremiseEvidence.replay
    {actual : Graph} {occurrences : List RoutedElimOccurrence}
    {fuel : Nat} {vertex : Vertex}
    (evidence : RoutedPremiseEvidence actual occurrences fuel vertex) :
    Exists fun routes =>
      occurrenceReplayAt actual (fuel + 1) vertex = .ok routes /\
      SelfCertifying.ListSetEq routes
        (expectedRoutedPremiseRoutes occurrences vertex) := by
  cases evidence with
  | source sourceEvidence =>
      rcases sourceEvidence with
        ⟨traces, noPredecessors, source, uncollapsed, actualAncestors,
          tracesNonempty, expected⟩
      refine ⟨tracedRoutes (normalizeDep [vertex.FORMULA]) traces, ?_, expected⟩
      exact occurrenceReplayAt_reseed_source actual fuel vertex traces
        noPredecessors source uncollapsed actualAncestors tracesNonempty
  | logicalDerived logicalEvidence =>
      rcases logicalEvidence with ⟨routes, traces, frame, expected⟩
      exact ⟨reseedDerivedRoutes traces routes, frame.replay, expected⟩

private theorem routesFor_eq_nil_of_not_mem_keys
    (table : List (Vertex × List FlowPair)) (vertex : Vertex)
    (absent : vertex ∉ table.map Prod.fst) :
    OccurrenceAwareReplay.routesFor table vertex = [] := by
  induction table with
  | nil => rfl
  | cons entry rest ih =>
      simp only [List.map_cons, List.mem_cons, not_or] at absent
      have headNe : entry.1 ≠ vertex := by
        intro equal
        exact absent.1 equal.symm
      simp only [OccurrenceAwareReplay.routesFor]
      simp [headNe]
      simpa [OccurrenceAwareReplay.routesFor] using ih absent.2




theorem recursiveRoutedPremiseTable_correct
    (actual : Graph) (occurrences : List RoutedElimOccurrence)
    (fuel : Nat) (predecessors : List Vertex)
    (nodup : predecessors.Nodup)
    (evidence : forall vertex, vertex ∈ predecessors ->
      RoutedPremiseEvidence actual occurrences fuel vertex)
    (support : forall vertex,
      expectedRoutedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predecessors) :
    Exists fun table =>
      predecessors.mapM (fun premise => do
        let routes <- occurrenceReplayAt actual (fuel + 1) premise
        return (premise, routes)) = .ok table /\
      table.map Prod.fst = predecessors /\
      RoutedPremiseRoutesCorrect
        (OccurrenceAwareReplay.routesFor table) occurrences := by
  have build : forall (vertices : List Vertex), vertices.Nodup ->
      (forall vertex, vertex ∈ vertices ->
        RoutedPremiseEvidence actual occurrences fuel vertex) ->
      Exists fun table =>
        vertices.mapM (fun premise => do
          let routes <- occurrenceReplayAt actual (fuel + 1) premise
          return (premise, routes)) = .ok table /\
        table.map Prod.fst = vertices /\
        (forall vertex, vertex ∈ vertices ->
          SelfCertifying.ListSetEq
            (OccurrenceAwareReplay.routesFor table vertex)
            (expectedRoutedPremiseRoutes occurrences vertex)) := by
    intro vertices verticesNodup
    induction vertices with
    | nil =>
        intro _
        exact ⟨[], rfl, rfl, by simp⟩
    | cons vertex rest ih =>
        intro allEvidence
        have nodupParts := List.nodup_cons.mp verticesNodup
        obtain ⟨routes, replay, routesCorrect⟩ :=
          (allEvidence vertex (List.Mem.head rest)).replay
        obtain ⟨tailTable, tailReplay, tailKeys, tailCorrect⟩ :=
          ih nodupParts.2 (fun candidate member =>
            allEvidence candidate (List.Mem.tail vertex member))
        refine ⟨(vertex, routes) :: tailTable, ?_, ?_, ?_⟩
        · simp only [List.mapM_cons, replay]
          rw [tailReplay]
          rfl
        · simp [tailKeys]
        · intro candidate member
          rcases List.mem_cons.mp member with equal | tailMember
          · subst candidate
            simp [OccurrenceAwareReplay.routesFor, routesCorrect]
          · have distinct : vertex ≠ candidate := by
              intro equal
              subst candidate
              exact nodupParts.1 tailMember
            simpa [OccurrenceAwareReplay.routesFor, distinct] using
              tailCorrect candidate tailMember
  obtain ⟨table, replay, keys, onSupport⟩ :=
    build predecessors nodup evidence
  refine ⟨table, replay, keys, ?_⟩
  intro vertex
  by_cases member : vertex ∈ predecessors
  · exact onSupport vertex member
  · have expectedEmpty :
        expectedRoutedPremiseRoutes occurrences vertex = [] := by
      by_contra nonempty
      exact member (support vertex nonempty)
    have absent : vertex ∉ table.map Prod.fst := by
      simpa [keys] using member
    rw [routesFor_eq_nil_of_not_mem_keys table vertex absent, expectedEmpty]
    exact SelfCertifying.ListSetEq.refl []






theorem occurrenceReplayAt_of_routed_premise_evidence
    (actual : Graph) (occurrences : List RoutedElimOccurrence)
    (fuel : Nat) (center : Vertex)
    (nodup : (predsOf actual center).Nodup)
    (evidence : forall vertex, vertex ∈ predsOf actual center ->
      RoutedPremiseEvidence actual occurrences fuel vertex)
    (support : forall vertex,
      expectedRoutedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predsOf actual center)
    (logicalKernel : forall premiseFlow,
      RoutedPremiseRoutesCorrect premiseFlow occurrences ->
        derivedOutputs actual center premiseFlow =
          expectedRoutedFlowRoutes occurrences)
    (hasPredecessor : (predsOf actual center).isEmpty = false)
    (noAncestors : ancestorsInto actual center = [])
    (derived : center.HYPOTHESIS = false)
    (logicalNonempty :
      (expectedRoutedFlowRoutes occurrences).isEmpty = false) :
    occurrenceReplayAt actual (fuel + 2) center =
      .ok (expectedRoutedFlowRoutes occurrences) := by
  obtain ⟨table, tableReplay, _, premiseCorrect⟩ :=
    recursiveRoutedPremiseTable_correct actual occurrences fuel
      (predsOf actual center) nodup evidence support
  let frame : DerivedPlainReplayFrame actual (fuel + 1) center
      (expectedRoutedFlowRoutes occurrences) := {
    tableAndLogical :=
      ⟨table, tableReplay,
        logicalKernel (OccurrenceAwareReplay.routesFor table) premiseCorrect⟩
    hasPredecessor := hasPredecessor
    noAncestors := noAncestors
    derived := derived
    logicalNonempty := logicalNonempty
  }
  simpa [Nat.add_assoc] using frame.replay





theorem occurrenceReplayAt_setEq_of_routed_premise_evidence
    (actual : Graph) (occurrences : List RoutedElimOccurrence)
    (fuel : Nat) (center : Vertex)
    (nodup : (predsOf actual center).Nodup)
    (evidence : forall vertex, vertex ∈ predsOf actual center ->
      RoutedPremiseEvidence actual occurrences fuel vertex)
    (support : forall vertex,
      expectedRoutedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predsOf actual center)
    (logicalKernel : forall premiseFlow,
      RoutedPremiseRoutesCorrect premiseFlow occurrences ->
        SelfCertifying.ListSetEq
          (derivedOutputs actual center premiseFlow)
          (expectedRoutedFlowRoutes occurrences))
    (hasPredecessor : (predsOf actual center).isEmpty = false)
    (noAncestors : ancestorsInto actual center = [])
    (derived : center.HYPOTHESIS = false)
    (expectedNonempty :
      (expectedRoutedFlowRoutes occurrences).isEmpty = false) :
    Exists fun outputs =>
      occurrenceReplayAt actual (fuel + 2) center = .ok outputs /\
      SelfCertifying.ListSetEq outputs
        (expectedRoutedFlowRoutes occurrences) := by
  obtain ⟨table, tableReplay, _, premiseCorrect⟩ :=
    recursiveRoutedPremiseTable_correct actual occurrences fuel
      (predsOf actual center) nodup evidence support
  let outputs := derivedOutputs actual center
    (OccurrenceAwareReplay.routesFor table)
  have outputsCorrect : SelfCertifying.ListSetEq outputs
      (expectedRoutedFlowRoutes occurrences) :=
    logicalKernel (OccurrenceAwareReplay.routesFor table) premiseCorrect
  have outputsNonempty : outputs.isEmpty = false := by
    rw [List.isEmpty_eq_false_iff]
    have expectedNe : expectedRoutedFlowRoutes occurrences ≠ [] := by
      simpa [List.isEmpty_iff] using expectedNonempty
    obtain ⟨route, routeMember⟩ :=
      List.exists_mem_of_ne_nil _ expectedNe
    exact List.ne_nil_of_mem ((outputsCorrect route).mpr routeMember)
  let frame : DerivedPlainReplayFrame actual (fuel + 1) center outputs := {
    tableAndLogical := ⟨table, tableReplay, rfl⟩
    hasPredecessor := hasPredecessor
    noAncestors := noAncestors
    derived := derived
    logicalNonempty := outputsNonempty
  }
  refine ⟨outputs, ?_, outputsCorrect⟩
  simpa [Nat.add_assoc] using frame.replay




theorem occurrenceReplayAt_setEq_of_routed_incidence
    (actual : Graph) (occurrences : List RoutedElimOccurrence)
    (fuel : Nat) (center : Vertex)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (nodup : (predsOf actual center).Nodup)
    (evidence : forall vertex, vertex ∈ predsOf actual center ->
      RoutedPremiseEvidence actual occurrences fuel vertex)
    (support : forall vertex,
      expectedRoutedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predsOf actual center)
    (hasPredecessor : (predsOf actual center).isEmpty = false)
    (noAncestors : ancestorsInto actual center = [])
    (derived : center.HYPOTHESIS = false)
    (expectedNonempty :
      (expectedRoutedFlowRoutes occurrences).isEmpty = false) :
    Exists fun outputs =>
      occurrenceReplayAt actual (fuel + 2) center = .ok outputs /\
      SelfCertifying.ListSetEq outputs
        (expectedRoutedFlowRoutes occurrences) := by
  apply occurrenceReplayAt_setEq_of_routed_premise_evidence actual
    occurrences fuel center nodup evidence support
  · intro premiseFlow premiseCorrect
    exact derivedOutputs_setEq_expected_routed actual center occurrences
      incidence premiseFlow premiseCorrect
  · exact hasPredecessor
  · exact noAncestors
  · exact derived
  · exact expectedNonempty



theorem occurrenceReplayAt_setEq_of_exact_routed_replacement
    (actual : Graph) (occurrences : List RoutedElimOccurrence)
    (fuel : Nat) (center : Vertex)
    (incidence : ExactRoutedReplacementIncidence actual center occurrences)
    (nodup : (predsOf actual center).Nodup)
    (evidence : forall vertex, vertex ∈ predsOf actual center ->
      RoutedPremiseEvidence actual occurrences fuel vertex)
    (support : forall vertex,
      expectedRoutedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predsOf actual center)
    (hasPredecessor : (predsOf actual center).isEmpty = false)
    (noAncestors : ancestorsInto actual center = [])
    (derived : center.HYPOTHESIS = false)
    (expectedNonempty :
      (expectedRoutedFlowRoutes occurrences).isEmpty = false) :
    Exists fun outputs =>
      occurrenceReplayAt actual (fuel + 2) center = .ok outputs /\
      SelfCertifying.ListSetEq outputs
        (expectedRoutedFlowRoutes occurrences) :=
  occurrenceReplayAt_setEq_of_routed_incidence actual occurrences fuel center
    incidence.toRoutedEliminationIncidence nodup evidence support
    hasPredecessor noAncestors derived expectedNonempty

end HCInterClassReplay
