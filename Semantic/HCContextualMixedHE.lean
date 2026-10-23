import Semantic.ActualMixedProducerBridge























open Semantic FlowSpec
open LocalSpecificationCandidate ActualMixedProducerBridge

namespace HCContextualMixedHE




structure BoundaryObservation where
  production : MixedProduction
  conclusion : Formula
  use : Vertex
  dependency : Dep
  residual : CandidateResidual
deriving Repr, DecidableEq


def observeAcross (conclusion : Formula) (route : CandidateRoute)
    (edge : Deduction) : Option BoundaryObservation :=
  (consumeCandidate edge route.residual).map fun tail =>
    { production := route.production
      conclusion := conclusion
      use := edge.END
      dependency := route.dependency
      residual := tail }



def boundaryDeliveries (conclusion : Formula)
    (routes : List CandidateRoute) (edges : List Deduction) :
    List BoundaryObservation :=
  routes.flatMap fun route =>
    edges.filterMap fun edge => observeAcross conclusion route edge


def preSourceRoute {d : Graph} (input : ActualMixedHEInput d) :
    CandidateRoute :=
  { production := .source
    dependency := normalizeDep [input.source.FORMULA]
    residual := ⟨[]⟩ }



def preEliminationRoute {d : Graph} (input : ActualMixedHEInput d) :
    CandidateRoute :=
  { production := .elimination
    dependency := depUnion (normalizeDep [input.minor.FORMULA])
      (normalizeDep [input.major.FORMULA])
    residual := ⟨[]⟩ }

def preSourceEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge input.source input.sourceUse 0 input.sourceStored

def preDerivedEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge input.derived input.derivedUse 0 input.derivedStored



def preBoundaryDeliveries {d : Graph} (input : ActualMixedHEInput d) :
    List BoundaryObservation :=
  (observeAcross input.source.FORMULA (preSourceRoute input)
      (preSourceEdge input)).toList ++
    (observeAcross input.derived.FORMULA (preEliminationRoute input)
      (preDerivedEdge input)).toList




def postBoundaryDeliveries {d : Graph} (input : ActualMixedHEInput d) :
    List BoundaryObservation :=
  let view := producedRuntimeView input
  boundaryDeliveries view.mixed.FORMULA (candidateMixedRoutes view)
    [view.sourceEdge, view.derivedEdge]

def expectedSourceObservation {d : Graph} (input : ActualMixedHEInput d) :
    BoundaryObservation :=
  { production := .source
    conclusion := input.source.FORMULA
    use := input.sourceUse
    dependency := normalizeDep [input.source.FORMULA]
    residual := ⟨[]⟩ }

def expectedEliminationObservation {d : Graph}
    (input : ActualMixedHEInput d) : BoundaryObservation :=
  { production := .elimination
    conclusion := input.derived.FORMULA
    use := input.derivedUse
    dependency := depUnion (normalizeDep [input.minor.FORMULA])
      (normalizeDep [input.major.FORMULA])
    residual := ⟨[]⟩ }

@[simp] theorem preBoundaryDeliveries_eq_expected
    {d : Graph} (input : ActualMixedHEInput d) :
    preBoundaryDeliveries input =
      [expectedSourceObservation input,
       expectedEliminationObservation input] := by
  simp [preBoundaryDeliveries, observeAcross, preSourceRoute,
    preEliminationRoute, preSourceEdge, preDerivedEdge,
    consumeCandidate, expectedSourceObservation,
    expectedEliminationObservation]




theorem produced_premise_prefix_transport
    {d : Graph} (input : ActualMixedHEInput d) :
    consumeCandidate (producedMajorEdge input)
        ⟨(producedMajorTrace input).COLOURS⟩ =
          some (eliminationRoute (producedRuntimeView input)).residual ∧
      consumeCandidate (producedMinorEdge input)
        ⟨(producedMinorTrace input).COLOURS⟩ =
          some (eliminationRoute (producedRuntimeView input)).residual := by
  simp [producedMajorEdge, producedMinorEdge, producedMajorTrace,
    producedMinorTrace, producedRuntimeView, eliminationRoute,
    producedDerivedEdge, consumeCandidate]




theorem postBoundaryDeliveries_eq_expected
    {d : Graph} (input : ActualMixedHEInput d) :
    postBoundaryDeliveries input =
      [expectedSourceObservation input,
       expectedEliminationObservation input] := by
  have sameFormula : input.source.FORMULA = input.derived.FORMULA :=
    input.sameFormula
  have zeroNe : 0 ≠ input.derived.NUMBER :=
    input.positiveGeneratedColour.symm
  simp [postBoundaryDeliveries, boundaryDeliveries, observeAcross,
    candidateMixedRoutes, sourceRoute, eliminationRoute,
    consumeCandidate, producedRuntimeView, producedCenter,
    collapse.center, producedSourceEdge, producedDerivedEdge,
    expectedSourceObservation, expectedEliminationObservation,
    sameFormula, input.positiveGeneratedColour, zeroNe]




theorem actualMixedHE_boundary_preserved
    {d : Graph} (input : ActualMixedHEInput d) :
    postBoundaryDeliveries input = preBoundaryDeliveries input := by
  rw [postBoundaryDeliveries_eq_expected,
    preBoundaryDeliveries_eq_expected]


theorem actualMixedHE_no_spurious_boundary_delivery
    {d : Graph} (input : ActualMixedHEInput d)
    (observation : BoundaryObservation) :
    observation ∈ postBoundaryDeliveries input ↔
      observation = expectedSourceObservation input ∨
      observation = expectedEliminationObservation input := by
  rw [postBoundaryDeliveries_eq_expected]
  simp



theorem actualMixedHE_inputs_and_ancestors_represented
    {d : Graph} (input : ActualMixedHEInput d) :
    input.source ∈ d.NODES ∧ input.derived ∈ d.NODES ∧
      input.major ∈ d.NODES ∧ input.minor ∈ d.NODES ∧
      (producedNeighborhood input).INDIRECT =
        [producedMajorTrace input, producedMinorTrace input] ∧
      (producedNeighborhood input).DIRECT = [] := by
  exact ⟨input.sourceMember, input.derivedMember, input.majorMember,
    input.minorMember, rfl, rfl⟩




theorem actualMixedHE_operation_has_preserved_boundary
    {d : Graph} (input : ActualMixedHEInput d) :
    actualOrderedMixedOperation input = (producedNeighborhood input, []) ∧
      postBoundaryDeliveries input = preBoundaryDeliveries input :=
  ⟨actualOrderedMixedOperation_eq_produced input,
    actualMixedHE_boundary_preserved input⟩





theorem surrounding_consumer_preserved
    {d : Graph} (input : ActualMixedHEInput d)
    {R : Sort _} (consumer : List BoundaryObservation → R) :
    consumer (postBoundaryDeliveries input) =
      consumer (preBoundaryDeliveries input) := by
  rw [actualMixedHE_boundary_preserved]



structure MixedHEContextualSimulation {d : Graph}
    (input : ActualMixedHEInput d) : Prop where
  actualProducer :
    actualOrderedMixedOperation input = (producedNeighborhood input, [])
  premisePrefixTransport :
    consumeCandidate (producedMajorEdge input)
        ⟨(producedMajorTrace input).COLOURS⟩ =
          some (eliminationRoute (producedRuntimeView input)).residual ∧
      consumeCandidate (producedMinorEdge input)
        ⟨(producedMinorTrace input).COLOURS⟩ =
          some (eliminationRoute (producedRuntimeView input)).residual
  exactBoundary :
    postBoundaryDeliveries input = preBoundaryDeliveries input
  noSpurious : ∀ observation,
    observation ∈ postBoundaryDeliveries input ↔
      observation = expectedSourceObservation input ∨
      observation = expectedEliminationObservation input




theorem actualMixedHE_contextualSimulation
    {d : Graph} (input : ActualMixedHEInput d) :
    MixedHEContextualSimulation input :=
  { actualProducer := actualOrderedMixedOperation_eq_produced input
    premisePrefixTransport := produced_premise_prefix_transport input
    exactBoundary := actualMixedHE_boundary_preserved input
    noSpurious := actualMixedHE_no_spurious_boundary_delivery input }

end HCContextualMixedHE
