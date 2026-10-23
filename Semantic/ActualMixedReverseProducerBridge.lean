import Semantic.ActualMixedProducerBridge
import Semantic.HCContextualMixedHI









open Semantic FlowSpec

namespace ActualMixedReverseProducerBridge

open ActualMixedProducerBridge ActualMixedHIProducerBridge

namespace HE

def producedCenter {d : Graph} (input : ActualMixedHEInput d) : Vertex :=
  collapse.center input.source input.derived

def producedMajorEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge input.major (producedCenter input) 0 input.majorStored

def producedMinorEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge input.minor (producedCenter input) 0 input.minorStored

def producedDerivedEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge (producedCenter input) input.derivedUse input.derived.NUMBER
    input.derivedStored

def producedSourceEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge (producedCenter input) input.sourceUse 0 input.sourceStored

def producedMajorTrace {d : Graph} (input : ActualMixedHEInput d) : Ancestral :=
  path input.derivedUse input.major [0, input.derived.NUMBER]

def producedMinorTrace {d : Graph} (input : ActualMixedHEInput d) : Ancestral :=
  path input.derivedUse input.minor [0, input.derived.NUMBER]




def producedNeighborhood {d : Graph} (input : ActualMixedHEInput d) :
    Neighborhood :=
  rule (producedCenter input)
    [producedMajorEdge input, producedMinorEdge input]
    [producedDerivedEdge input, producedSourceEdge input]
    [] [producedMajorTrace input, producedMinorTrace input]

def actualReverseMixedOperation {d : Graph} (input : ActualMixedHEInput d) :
    Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.source d))
    (pre_collapse (get_rule input.derived d))

theorem actualReverseMixedOperation_eq_produced
    {d : Graph} (input : ActualMixedHEInput d) :
    actualReverseMixedOperation input = (producedNeighborhood input, []) := by
  have zeroNe : 0 ≠ input.derived.NUMBER := input.positiveGeneratedColour.symm
  have incomingNe : producedMajorEdge input ≠ producedMinorEdge input := by
    intro equal
    exact input.major_ne_minor (congrArg Deduction.START equal)
  have outgoingNe : producedDerivedEdge input ≠ producedSourceEdge input := by
    intro equal
    exact input.positiveGeneratedColour (congrArg Deduction.COLOUR equal)
  have incomingBeq :
      (producedMinorEdge input == producedMajorEdge input) = false :=
    beq_eq_false_iff_ne.mpr incomingNe.symm
  have outgoingBeq :
      (producedSourceEdge input == producedDerivedEdge input) = false :=
    beq_eq_false_iff_ne.mpr outgoingNe.symm
  have incomingDedup :
      List.eraseDups [producedMajorEdge input, producedMinorEdge input] =
        [producedMajorEdge input, producedMinorEdge input] := by
    simp [List.eraseDups_cons, List.filter, incomingBeq]
  have outgoingDedup :
      List.eraseDups [producedDerivedEdge input, producedSourceEdge input] =
        [producedDerivedEdge input, producedSourceEdge input] := by
    simp [List.eraseDups_cons, List.filter, outgoingBeq]
  simp [actualReverseMixedOperation, input.derivedRule, input.sourceRule, pre_collapse, pre_collapse.outgoing, pre_collapse.direct, pre_collapse.indirect, pre_collapse.indirect.create, pre_collapse.discriminatePair, pre_collapse.outgoingRepresentativeConflictB, collapseWithOccurrences, collapse.center, collapse.rewrite_incoming, collapse.rewrite_outgoing, collapse.rewrite_direct, collapse.checkNeighborhoodPathHeads, collapse.neighborhoodPathHeadsOKB, collapse.neighborhoodPathHeadOKB, collapse.checkOutgoingRepresentatives, collapse.outgoingRepresentativesOKB, producedNeighborhood, producedCenter, producedMajorEdge, producedMinorEdge, producedSourceEdge, producedDerivedEdge, producedMajorTrace, producedMinorTrace, input.derivedNotSource, input.sourceIsHypothesis, input.derivedUncollapsed, input.sourceUncollapsed, input.positiveGeneratedColour, input.major_ne_minor, input.sourceUseSelfRepresentative, input.derivedUseSelfRepresentative, zeroNe]
  simpa [producedMajorEdge, producedMinorEdge, producedSourceEdge,
    producedDerivedEdge, producedCenter, collapse.center,
    input.derivedNotSource, input.sourceIsHypothesis] using
    And.intro incomingDedup outgoingDedup

end HE

namespace HI

def producedCenter {d : Graph} (input : ActualMixedHIInput d) : Vertex :=
  collapse.center input.source input.derived

def producedPremiseEdge {d : Graph} (input : ActualMixedHIInput d) :
    Deduction :=
  edge input.premise (producedCenter input) 0 input.premiseStored

def producedDerivedEdge {d : Graph} (input : ActualMixedHIInput d) : Deduction :=
  edge (producedCenter input) input.derivedUse input.derived.NUMBER
    input.derivedStored

def producedSourceEdge {d : Graph} (input : ActualMixedHIInput d) : Deduction :=
  edge (producedCenter input) input.sourceUse 0 input.sourceStored

def producedPremiseTrace {d : Graph} (input : ActualMixedHIInput d) :
    Ancestral :=
  path input.derivedUse input.premise [0, input.derived.NUMBER]

def producedNeighborhood {d : Graph} (input : ActualMixedHIInput d) :
    Neighborhood :=
  rule (producedCenter input)
    [producedPremiseEdge input]
    [producedDerivedEdge input, producedSourceEdge input]
    [] [producedPremiseTrace input]

def actualReverseMixedOperation {d : Graph} (input : ActualMixedHIInput d) :
    Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.source d))
    (pre_collapse (get_rule input.derived d))

theorem actualReverseMixedOperation_eq_produced
    {d : Graph} (input : ActualMixedHIInput d) :
    actualReverseMixedOperation input = (producedNeighborhood input, []) := by
  have zeroNe : 0 ≠ input.derived.NUMBER := input.positiveGeneratedColour.symm
  have outgoingNe : producedDerivedEdge input ≠ producedSourceEdge input := by
    intro equal
    exact input.positiveGeneratedColour (congrArg Deduction.COLOUR equal)
  have outgoingBeq :
      (producedSourceEdge input == producedDerivedEdge input) = false :=
    beq_eq_false_iff_ne.mpr outgoingNe.symm
  have outgoingDedup :
      List.eraseDups [producedDerivedEdge input, producedSourceEdge input] =
        [producedDerivedEdge input, producedSourceEdge input] := by
    simp [List.eraseDups_cons, List.filter, outgoingBeq]
  have incomingDedup :
      List.eraseDups [producedPremiseEdge input] =
        [producedPremiseEdge input] := by rfl
  simp [actualReverseMixedOperation, input.derivedRule, input.sourceRule, pre_collapse, pre_collapse.outgoing, pre_collapse.direct, pre_collapse.indirect, pre_collapse.indirect.create, pre_collapse.discriminatePair, pre_collapse.outgoingRepresentativeConflictB, collapseWithOccurrences, collapse.center, collapse.rewrite_incoming, collapse.rewrite_outgoing, collapse.rewrite_direct, collapse.checkNeighborhoodPathHeads, collapse.neighborhoodPathHeadsOKB, collapse.neighborhoodPathHeadOKB, collapse.checkOutgoingRepresentatives, collapse.outgoingRepresentativesOKB, producedNeighborhood, producedCenter, producedPremiseEdge, producedSourceEdge, producedDerivedEdge, producedPremiseTrace, input.derivedNotSource, input.sourceIsHypothesis, input.derivedUncollapsed, input.sourceUncollapsed, input.positiveGeneratedColour, input.sourceUseSelfRepresentative, input.derivedUseSelfRepresentative, zeroNe]
  simpa [producedPremiseEdge, producedSourceEdge, producedDerivedEdge,
    producedCenter, collapse.center, input.derivedNotSource,
    input.sourceIsHypothesis] using And.intro incomingDedup outgoingDedup

end HI

namespace Contextual

open LocalSpecificationCandidate HCContextualMixedHE HCContextualMixedHI

def heRoutes {d : Graph} (input : ActualMixedHEInput d) : List CandidateRoute :=
  [ { production := .source
      dependency := normalizeDep [input.source.FORMULA]
      residual := ⟨[]⟩ }
  , { production := .elimination
      dependency := depUnion (normalizeDep [input.minor.FORMULA])
        (normalizeDep [input.major.FORMULA])
      residual := ⟨[input.derived.NUMBER]⟩ } ]

def hePostBoundary {d : Graph} (input : ActualMixedHEInput d) :
    List BoundaryObservation :=
  boundaryDeliveries (HE.producedCenter input).FORMULA (heRoutes input)
    [HE.producedDerivedEdge input, HE.producedSourceEdge input]

theorem reverseMixedHE_boundary_preserved
    {d : Graph} (input : ActualMixedHEInput d) :
    hePostBoundary input = HCContextualMixedHE.preBoundaryDeliveries input := by
  have sameFormula : input.source.FORMULA = input.derived.FORMULA :=
    input.sameFormula
  have zeroNe : 0 ≠ input.derived.NUMBER :=
    input.positiveGeneratedColour.symm
  simp [hePostBoundary, heRoutes, boundaryDeliveries, observeAcross,
    consumeCandidate, HE.producedCenter, HE.producedDerivedEdge,
    HE.producedSourceEdge, collapse.center, sameFormula,
    input.positiveGeneratedColour, zeroNe,
    HCContextualMixedHE.preBoundaryDeliveries,
    HCContextualMixedHE.preSourceRoute,
    HCContextualMixedHE.preEliminationRoute,
    HCContextualMixedHE.preSourceEdge,
    HCContextualMixedHE.preDerivedEdge]

theorem reverseMixedHE_no_spurious_boundary_delivery
    {d : Graph} (input : ActualMixedHEInput d)
    (observation : BoundaryObservation) :
    observation ∈ hePostBoundary input ↔
      observation = HCContextualMixedHE.expectedSourceObservation input ∨
      observation = HCContextualMixedHE.expectedEliminationObservation input := by
  rw [reverseMixedHE_boundary_preserved,
    HCContextualMixedHE.preBoundaryDeliveries_eq_expected]
  simp

def hiRoutes {d : Graph} (input : ActualMixedHIInput d) : List CandidateRoute :=
  [ { production := .source
      dependency := normalizeDep [input.source.FORMULA]
      residual := ⟨[]⟩ }
  , { production := .introduction
      dependency := depRemove (normalizeDep [input.premise.FORMULA])
        input.antecedent
      residual := ⟨[input.derived.NUMBER]⟩ } ]

def hiPostBoundary {d : Graph} (input : ActualMixedHIInput d) :
    List BoundaryObservation :=
  boundaryDeliveries (HI.producedCenter input).FORMULA (hiRoutes input)
    [HI.producedDerivedEdge input, HI.producedSourceEdge input]

theorem reverseMixedHI_boundary_preserved
    {d : Graph} (input : ActualMixedHIInput d) :
    hiPostBoundary input = HCContextualMixedHI.preBoundaryDeliveries input := by
  have sameFormula : input.source.FORMULA = input.derived.FORMULA :=
    input.sameFormula
  have zeroNe : 0 ≠ input.derived.NUMBER :=
    input.positiveGeneratedColour.symm
  simp [hiPostBoundary, hiRoutes, boundaryDeliveries, observeAcross,
    consumeCandidate, HI.producedCenter, HI.producedDerivedEdge,
    HI.producedSourceEdge, collapse.center, sameFormula,
    input.derivedFormula, input.premiseFormula,
    input.positiveGeneratedColour, zeroNe,
    HCContextualMixedHI.preBoundaryDeliveries,
    HCContextualMixedHI.preSourceRoute,
    HCContextualMixedHI.preIntroductionRoute]

theorem reverseMixedHI_no_spurious_boundary_delivery
    {d : Graph} (input : ActualMixedHIInput d)
    (observation : BoundaryObservation) :
    observation ∈ hiPostBoundary input ↔
      observation = HCContextualMixedHI.expectedSourceObservation input ∨
      observation = HCContextualMixedHI.expectedIntroductionObservation input := by
  rw [reverseMixedHI_boundary_preserved,
    HCContextualMixedHI.preBoundaryDeliveries_eq_expected]
  simp

end Contextual

end ActualMixedReverseProducerBridge
