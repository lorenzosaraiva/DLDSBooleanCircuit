import Semantic.HCContextualMixedHE
import Semantic.ActualMixedHIProducerBridge








open Semantic FlowSpec LocalSpecificationCandidate
open HCContextualMixedHE ActualMixedHIProducerBridge

namespace HCContextualMixedHI

def preSourceRoute {d : Graph} (input : ActualMixedHIInput d) :
    CandidateRoute :=
  { production := .source
    dependency := normalizeDep [input.source.FORMULA]
    residual := ⟨[]⟩ }

def preIntroductionRoute {d : Graph} (input : ActualMixedHIInput d) :
    CandidateRoute :=
  { production := .introduction
    dependency := depRemove (normalizeDep [input.premise.FORMULA])
      input.antecedent
    residual := ⟨[]⟩ }

def preBoundaryDeliveries {d : Graph} (input : ActualMixedHIInput d) :
    List BoundaryObservation :=
  (observeAcross input.source.FORMULA (preSourceRoute input)
      (edge input.source input.sourceUse 0 input.sourceStored)).toList ++
    (observeAcross input.derived.FORMULA (preIntroductionRoute input)
      (edge input.derived input.derivedUse 0 input.derivedStored)).toList

def postBoundaryDeliveries {d : Graph} (input : ActualMixedHIInput d) :
    List BoundaryObservation :=
  let view := producedRuntimeView input
  boundaryDeliveries view.mixed.FORMULA (mixedHIRoutes view)
    [view.sourceEdge, view.derivedEdge]

def expectedSourceObservation {d : Graph} (input : ActualMixedHIInput d) :
    BoundaryObservation :=
  { production := .source
    conclusion := input.source.FORMULA
    use := input.sourceUse
    dependency := normalizeDep [input.source.FORMULA]
    residual := ⟨[]⟩ }

def expectedIntroductionObservation {d : Graph}
    (input : ActualMixedHIInput d) : BoundaryObservation :=
  { production := .introduction
    conclusion := input.derived.FORMULA
    use := input.derivedUse
    dependency := depRemove (normalizeDep [input.premise.FORMULA])
      input.antecedent
    residual := ⟨[]⟩ }

@[simp] theorem preBoundaryDeliveries_eq_expected
    {d : Graph} (input : ActualMixedHIInput d) :
    preBoundaryDeliveries input =
      [expectedSourceObservation input,
       expectedIntroductionObservation input] := by
  simp [preBoundaryDeliveries, observeAcross, preSourceRoute,
    preIntroductionRoute, consumeCandidate, expectedSourceObservation,
    expectedIntroductionObservation]

theorem produced_premise_prefix_transport
    {d : Graph} (input : ActualMixedHIInput d) :
    consumeCandidate (producedPremiseEdge input)
        ⟨(producedPremiseTrace input).COLOURS⟩ =
      some (mixedHIIntroductionRoute
        (producedRuntimeView input)).residual := by
  simp [producedPremiseEdge, producedPremiseTrace, producedRuntimeView,
    mixedHIIntroductionRoute, producedDerivedEdge, consumeCandidate]

theorem postBoundaryDeliveries_eq_expected
    {d : Graph} (input : ActualMixedHIInput d) :
    postBoundaryDeliveries input =
      [expectedSourceObservation input,
       expectedIntroductionObservation input] := by
  have sameFormula : input.source.FORMULA = input.derived.FORMULA :=
    input.sameFormula
  have zeroNe : 0 ≠ input.derived.NUMBER :=
    input.positiveGeneratedColour.symm
  simp [postBoundaryDeliveries, boundaryDeliveries, observeAcross,
    mixedHIRoutes, mixedHISourceRoute, mixedHIIntroductionRoute,
    consumeCandidate, producedRuntimeView, producedCenter,
    collapse.center, producedSourceEdge, producedDerivedEdge,
    expectedSourceObservation, expectedIntroductionObservation,
    sameFormula, input.positiveGeneratedColour, zeroNe]



theorem actualMixedHI_boundary_preserved
    {d : Graph} (input : ActualMixedHIInput d) :
    postBoundaryDeliveries input = preBoundaryDeliveries input := by
  rw [postBoundaryDeliveries_eq_expected,
    preBoundaryDeliveries_eq_expected]

theorem actualMixedHI_no_spurious_boundary_delivery
    {d : Graph} (input : ActualMixedHIInput d)
    (observation : BoundaryObservation) :
    observation ∈ postBoundaryDeliveries input ↔
      observation = expectedSourceObservation input ∨
      observation = expectedIntroductionObservation input := by
  rw [postBoundaryDeliveries_eq_expected]
  simp

theorem surrounding_consumer_preserved
    {d : Graph} (input : ActualMixedHIInput d)
    {R : Sort _} (consumer : List BoundaryObservation → R) :
    consumer (postBoundaryDeliveries input) =
      consumer (preBoundaryDeliveries input) := by
  rw [actualMixedHI_boundary_preserved]

structure MixedHIContextualSimulation {d : Graph}
    (input : ActualMixedHIInput d) : Prop where
  actualProducer :
    actualOrderedMixedOperation input = (producedNeighborhood input, [])
  premisePrefixTransport :
    consumeCandidate (producedPremiseEdge input)
        ⟨(producedPremiseTrace input).COLOURS⟩ =
      some (mixedHIIntroductionRoute
        (producedRuntimeView input)).residual
  exactBoundary :
    postBoundaryDeliveries input = preBoundaryDeliveries input
  noSpurious : ∀ observation,
    observation ∈ postBoundaryDeliveries input ↔
      observation = expectedSourceObservation input ∨
      observation = expectedIntroductionObservation input

theorem actualMixedHI_contextualSimulation
    {d : Graph} (input : ActualMixedHIInput d) :
    MixedHIContextualSimulation input :=
  { actualProducer := actualOrderedMixedOperation_eq_produced input
    premisePrefixTransport := produced_premise_prefix_transport input
    exactBoundary := actualMixedHI_boundary_preserved input
    noSpurious := actualMixedHI_no_spurious_boundary_delivery input }

end HCContextualMixedHI
