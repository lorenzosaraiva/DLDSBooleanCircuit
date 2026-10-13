import DLDS.Compression.Mixed.EliminationSource

/-!
# Contextual boundary simulation for the actual mixed HE operation

This module is the first composition layer above the restricted local
candidate.  It still covers the same first-collapse hypothesis/elimination
subclass, but it no longer stops at a list of routes at the merged vertex.
It records what crosses the transformed region's boundary and proves that the
actual producer preserves those observations.

The interface deliberately distinguishes:

* a route at a logical occurrence;
* consumption by the occurrence's particular outgoing use; and
* the value observed by an arbitrary surrounding computation after that
  consumption.

The post-merge side enumerates the Cartesian product of both candidate routes
with both outgoing edges.  The theorem therefore proves that the two rejected
cross-routings do not create extra boundary observations; this is not an
assumption hidden in a pairing supplied by the caller.
-/

open DLDS FlowSemantics
open DLDS.Compression.Mixed.LocalRules DLDS.Compression.Mixed.EliminationSource

namespace DLDS.Compression.Mixed.ContextualElimination

/-- What a surrounding graph can observe after one logical production has
crossed one outgoing use of the transformed region.  `production` is an
occurrence-role tag; it is intentionally independent of `dependency`. -/
structure BoundaryObservation where
  production : MixedProduction
  conclusion : Formula
  use : Vertex
  dependency : Dep
  residual : CandidateResidual
deriving Repr, DecidableEq

/-- Consume a candidate route across an actual boundary edge. -/
def observeAcross (conclusion : Formula) (route : CandidateRoute)
    (edge : Deduction) : Option BoundaryObservation :=
  (consumeCandidate edge route.residual).map fun tail =>
    { production := route.production
      conclusion := conclusion
      use := edge.END
      dependency := route.dependency
      residual := tail }

/-- Enumerate every route/edge combination at a shared compressed vertex.
Invalid combinations are rejected by `consumeCandidate`. -/
def boundaryDeliveries (conclusion : Formula)
    (routes : List CandidateRoute) (edges : List Deduction) :
    List BoundaryObservation :=
  routes.flatMap fun route =>
    edges.filterMap fun edge => observeAcross conclusion route edge

/-- The source occurrence before compression. -/
def preSourceRoute {d : Graph} (input : ActualMixedHEInput d) :
    CandidateRoute :=
  { production := .source
    dependency := normalizeDep [input.source.FORMULA]
    residual := ⟨[]⟩ }

/-- The elimination occurrence before compression.  Its dependency is
computed from the premise formulae, not from any stored edge payload. -/
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

/-- Before compression the productions live at different vertices, so each
route is paired only with the outgoing use of its own occurrence. -/
def preBoundaryDeliveries {d : Graph} (input : ActualMixedHEInput d) :
    List BoundaryObservation :=
  (observeAcross input.source.FORMULA (preSourceRoute input)
      (preSourceEdge input)).toList ++
    (observeAcross input.derived.FORMULA (preEliminationRoute input)
      (preDerivedEdge input)).toList

/-- After compression both productions share the generated center.  We
enumerate all four route/edge combinations rather than preselecting the two
that are expected to work. -/
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

/-- Both generated premise traces transport a source seed through the new
colour-zero premise edge to precisely the derived occurrence's residual.
This is the input side of the local simulation. -/
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

/-- Exact post-merge observations.  The proof expands the full Cartesian
route/use enumeration, so it simultaneously establishes the absence of a
source packet on the derived use and of a derived packet on the source use. -/
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

/-- The main contextual boundary equality for the restricted actual operation.
It relates observations, not internal lineage keys and not literal equality of
the two graphs. -/
theorem actualMixedHE_boundary_preserved
    {d : Graph} (input : ActualMixedHEInput d) :
    postBoundaryDeliveries input = preBoundaryDeliveries input := by
  rw [postBoundaryDeliveries_eq_expected,
    preBoundaryDeliveries_eq_expected]

/-- No additional observable route is introduced by sharing the center. -/
theorem actualMixedHE_no_spurious_boundary_delivery
    {d : Graph} (input : ActualMixedHEInput d)
    (observation : BoundaryObservation) :
    observation ∈ postBoundaryDeliveries input ↔
      observation = expectedSourceObservation input ∨
      observation = expectedEliminationObservation input := by
  rw [postBoundaryDeliveries_eq_expected]
  simp

/-- Both required input occurrences and both generated ancestor references
are explicitly present in the local output of the real producer. -/
theorem actualMixedHE_inputs_and_ancestors_represented
    {d : Graph} (input : ActualMixedHEInput d) :
    input.source ∈ d.NODES ∧ input.derived ∈ d.NODES ∧
      input.major ∈ d.NODES ∧ input.minor ∈ d.NODES ∧
      (producedNeighborhood input).INDIRECT =
        [producedMajorTrace input, producedMinorTrace input] ∧
      (producedNeighborhood input).DIRECT = [] := by
  exact ⟨input.sourceMember, input.derivedMember, input.majorMember,
    input.minorMember, rfl, rfl⟩

/-- The actual executable local call produces exactly the neighbourhood whose
boundary was used above.  This field prevents the contextual theorem from
being read as a statement about a hand-constructed replacement. -/
theorem actualMixedHE_operation_has_preserved_boundary
    {d : Graph} (input : ActualMixedHEInput d) :
    actualOrderedMixedOperation input = (producedNeighborhood input, []) ∧
      postBoundaryDeliveries input = preBoundaryDeliveries input :=
  ⟨actualOrderedMixedOperation_eq_produced input,
    actualMixedHE_boundary_preserved input⟩

/-- Any surrounding computation which depends on the transformed region only
through its delivered boundary observations receives exactly the same input.
No equality of the untouched graph, recursive evaluator, or lineage keys is
assumed. -/
theorem surrounding_consumer_preserved
    {d : Graph} (input : ActualMixedHEInput d)
    {R : Sort _} (consumer : List BoundaryObservation → R) :
    consumer (postBoundaryDeliveries input) =
      consumer (preBoundaryDeliveries input) := by
  rw [actualMixedHE_boundary_preserved]

/-- Pack the proved obligations into one named simulation predicate for later
step-by-step compression invariants. -/
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

/-- Independent input-neighbourhood
conditions for the actual `(derived, source)` operation establish its complete
boundary simulation. -/
theorem actualMixedHE_contextualSimulation
    {d : Graph} (input : ActualMixedHEInput d) :
    MixedHEContextualSimulation input :=
  { actualProducer := actualOrderedMixedOperation_eq_produced input
    premisePrefixTransport := produced_premise_prefix_transport input
    exactBoundary := actualMixedHE_boundary_preserved input
    noSpurious := actualMixedHE_no_spurious_boundary_delivery input }

end DLDS.Compression.Mixed.ContextualElimination
