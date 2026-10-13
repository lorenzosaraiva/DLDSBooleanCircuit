import DLDS.Compression.Mixed.LocalRules

set_option linter.unusedSimpArgs false

/-!
# Actual ordered producer bridge for one mixed hypothesis/elimination merge

This module reasons about the actual `pre_collapse` and
`collapseWithOccurrences` definitions.  It does not change them.  The covered
subclass has a derived implication-elimination occurrence on the left and a
hypothesis occurrence of the same formula/level on the right; both have one
colour-zero outgoing use and the elimination premises are hypotheses.
-/

open DLDS FlowSemantics
open DLDS.Compression.Mixed.LocalRules

namespace DLDS.Compression.Mixed.EliminationSource

/-- Independent input-side conditions for the supported ordered HE merge.
Dependency labels are retained only because they are fields of the historical
edges.  None is used to compute a candidate semantic dependency. -/
structure ActualMixedHEInput (d : Graph) where
  derived : Vertex
  source : Vertex
  major : Vertex
  minor : Vertex
  sourceUse : Vertex
  derivedUse : Vertex
  majorStored : List Formula
  minorStored : List Formula
  sourceStored : List Formula
  derivedStored : List Formula
  derivedMember : derived ∈ d.NODES
  sourceMember : source ∈ d.NODES
  majorMember : major ∈ d.NODES
  minorMember : minor ∈ d.NODES
  majorEdgeMember : edge major derived 0 majorStored ∈ d.EDGES
  minorEdgeMember : edge minor derived 0 minorStored ∈ d.EDGES
  sourceEdgeMember : edge source sourceUse 0 sourceStored ∈ d.EDGES
  derivedEdgeMember : edge derived derivedUse 0 derivedStored ∈ d.EDGES
  derivedNotSource : derived.HYPOTHESIS = false
  sourceIsHypothesis : source.HYPOTHESIS = true
  majorIsHypothesis : major.HYPOTHESIS = true
  minorIsHypothesis : minor.HYPOTHESIS = true
  derivedUncollapsed : derived.COLLAPSED = false
  sourceUncollapsed : source.COLLAPSED = false
  derivedPastEmpty : derived.PAST = []
  sourcePastEmpty : source.PAST = []
  sameFormula : source.FORMULA = derived.FORMULA
  sameLevel : source.LEVEL = derived.LEVEL
  majorFormula : major.FORMULA = minor.FORMULA >> derived.FORMULA
  positiveGeneratedColour : derived.NUMBER ≠ 0
  occurrencesDistinct : derived ≠ source
  major_ne_derived : major ≠ derived
  minor_ne_derived : minor ≠ derived
  major_ne_minor : major ≠ minor
  sourceUse_ne_derived : sourceUse ≠ derived
  derivedUse_ne_derived : derivedUse ≠ derived
  major_ne_center : major ≠ collapse.center derived source
  minor_ne_center : minor ≠ collapse.center derived source
  sourceUse_ne_center : sourceUse ≠ collapse.center derived source
  derivedUse_ne_center : derivedUse ≠ collapse.center derived source
  sourceUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) sourceUse sourceUse = true
  derivedUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) derivedUse derivedUse = true
  derivedRule : get_rule derived d =
    rule derived
      [edge major derived 0 majorStored, edge minor derived 0 minorStored]
      [edge derived derivedUse 0 derivedStored] [] []
  sourceRule : get_rule source d =
    rule source [] [edge source sourceUse 0 sourceStored] [] []

def producedCenter {d : Graph} (input : ActualMixedHEInput d) : Vertex :=
  collapse.center input.derived input.source

def producedMajorEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge input.major (producedCenter input) 0 input.majorStored

def producedMinorEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge input.minor (producedCenter input) 0 input.minorStored

def producedSourceEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge (producedCenter input) input.sourceUse 0 input.sourceStored

def producedDerivedEdge {d : Graph} (input : ActualMixedHEInput d) : Deduction :=
  edge (producedCenter input) input.derivedUse input.derived.NUMBER
    input.derivedStored

def producedMajorTrace {d : Graph} (input : ActualMixedHEInput d) : Ancestral :=
  path input.derivedUse input.major [0, input.derived.NUMBER]

def producedMinorTrace {d : Graph} (input : ActualMixedHEInput d) : Ancestral :=
  path input.derivedUse input.minor [0, input.derived.NUMBER]

def producedNeighborhood {d : Graph} (input : ActualMixedHEInput d) :
    Neighborhood :=
  rule (producedCenter input)
    [producedMajorEdge input, producedMinorEdge input]
    [producedSourceEdge input, producedDerivedEdge input]
    [] [producedMajorTrace input, producedMinorTrace input]

/-- The exact real ordered local producer call. -/
def actualOrderedMixedOperation {d : Graph} (input : ActualMixedHEInput d) :
    Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.derived d))
    (pre_collapse (get_rule input.source d))

/-- Turn a produced neighbourhood into the smallest graph containing exactly
its local nodes, edges, and paths.  This is a projection of the actual result,
not a separately annotated expected route set. -/
def neighborhoodGraph (n : Neighborhood) : Graph :=
  dlds ([n.CENTER] ++ n.INCOMING.map (fun e => e.START) ++
      n.OUTGOING.map (fun e => e.END)).eraseDups
    (n.INCOMING ++ n.OUTGOING)
    (n.DIRECT ++ n.INDIRECT)

theorem producedCenter_formula {d : Graph} (input : ActualMixedHEInput d) :
    (producedCenter input).FORMULA = input.derived.FORMULA := rfl

theorem producedCenter_level {d : Graph} (input : ActualMixedHEInput d) :
    (producedCenter input).LEVEL = input.derived.LEVEL := rfl

theorem producedCenter_is_mixed {d : Graph} (input : ActualMixedHEInput d) :
    (producedCenter input).HYPOTHESIS = true := by
  simp [producedCenter, collapse.center, input.derivedNotSource,
    input.sourceIsHypothesis]

/-- Exact producer calculation.  In particular, the fresh operational colour
and `[0, colour]` traces are conclusions, not hypotheses. -/
theorem actualOrderedMixedOperation_eq_produced
    {d : Graph} (input : ActualMixedHEInput d) :
    actualOrderedMixedOperation input = (producedNeighborhood input, []) := by
  have zeroNe : 0 ≠ input.derived.NUMBER := input.positiveGeneratedColour.symm
  have incomingNe : producedMajorEdge input ≠ producedMinorEdge input := by
    intro equal
    exact input.major_ne_minor (congrArg Deduction.START equal)
  have outgoingNe : producedSourceEdge input ≠ producedDerivedEdge input := by
    intro equal
    exact zeroNe (congrArg Deduction.COLOUR equal)
  have incomingBeq :
      (producedMinorEdge input == producedMajorEdge input) = false :=
    beq_eq_false_iff_ne.mpr incomingNe.symm
  have outgoingBeq :
      (producedDerivedEdge input == producedSourceEdge input) = false :=
    beq_eq_false_iff_ne.mpr outgoingNe.symm
  have incomingDedup :
      List.eraseDups [producedMajorEdge input, producedMinorEdge input] =
        [producedMajorEdge input, producedMinorEdge input] := by
    simp [List.eraseDups_cons, List.filter, incomingBeq]
  have outgoingDedup :
      List.eraseDups [producedSourceEdge input, producedDerivedEdge input] =
        [producedSourceEdge input, producedDerivedEdge input] := by
    simp [List.eraseDups_cons, List.filter, outgoingBeq]
  -- The historical operation has two layers of `pre_collapse`; the second is
  -- inert because the first result remains uncollapsed.  All generated data
  -- below reduce from the real definitions.
  simp [actualOrderedMixedOperation, input.derivedRule, input.sourceRule,
    pre_collapse, pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.indirect.create,
    pre_collapse.discriminatePair,
    pre_collapse.outgoingRepresentativeConflictB,
    collapseWithOccurrences, collapse.center, collapse.rewrite_incoming,
    collapse.rewrite_outgoing, collapse.rewrite_direct,
    collapse.checkNeighborhoodPathHeads, collapse.neighborhoodPathHeadsOKB,
    collapse.neighborhoodPathHeadOKB,
    collapse.checkOutgoingRepresentatives,
    collapse.outgoingRepresentativesOKB, producedNeighborhood,
    producedCenter, producedMajorEdge, producedMinorEdge,
    producedSourceEdge, producedDerivedEdge, producedMajorTrace,
    producedMinorTrace, input.derivedNotSource, input.sourceIsHypothesis,
    input.derivedUncollapsed, input.sourceUncollapsed,
    input.positiveGeneratedColour, input.major_ne_minor,
    input.major_ne_derived, input.minor_ne_derived,
    input.sourceUse_ne_derived, input.derivedUse_ne_derived,
    input.sourceUseSelfRepresentative,
    input.derivedUseSelfRepresentative, zeroNe]
  simpa [producedMajorEdge, producedMinorEdge, producedSourceEdge,
    producedDerivedEdge, producedCenter, collapse.center,
    input.derivedNotSource, input.sourceIsHypothesis] using
    And.intro incomingDedup outgoingDedup

def producedLocalGraph {d : Graph} (input : ActualMixedHEInput d) : Graph :=
  neighborhoodGraph (producedNeighborhood input)

/-- The local graph used by the candidate view is exactly the projection of
the real producer output. -/
theorem producedLocalGraph_eq_actualProjection
    {d : Graph} (input : ActualMixedHEInput d) :
    producedLocalGraph input =
      neighborhoodGraph (actualOrderedMixedOperation input).1 := by
  rw [actualOrderedMixedOperation_eq_produced]
  rfl

/-- The actual producer's local projection supplies every structural field of
the candidate view. -/
def producedRuntimeView {d : Graph} (input : ActualMixedHEInput d) :
    MixedRuntimeView (producedLocalGraph input) := by
  have majorNotCenter : input.major ≠ producedCenter input := by
    exact input.major_ne_center
  have minorNotCenter : input.minor ≠ producedCenter input := by
    exact input.minor_ne_center
  have sourceUseNotCenter : input.sourceUse ≠ producedCenter input := by
    exact input.sourceUse_ne_center
  have derivedUseNotCenter : input.derivedUse ≠ producedCenter input := by
    exact input.derivedUse_ne_center
  have majorNotCenter' : input.major ≠ collapse.center input.derived input.source :=
    input.major_ne_center
  have minorNotCenter' : input.minor ≠ collapse.center input.derived input.source :=
    input.minor_ne_center
  have sourceUseNotCenter' :
      input.sourceUse ≠ collapse.center input.derived input.source :=
    input.sourceUse_ne_center
  have derivedUseNotCenter' :
      input.derivedUse ≠ collapse.center input.derived input.source :=
    input.derivedUse_ne_center
  refine
    { mixed := producedCenter input
      major := input.major
      minor := input.minor
      sourceUse := input.sourceUse
      derivedUse := input.derivedUse
      majorEdge := producedMajorEdge input
      minorEdge := producedMinorEdge input
      sourceEdge := producedSourceEdge input
      derivedEdge := producedDerivedEdge input
      majorTrace := producedMajorTrace input
      minorTrace := producedMinorTrace input
      mixedMember := ?_
      majorMember := ?_
      minorMember := ?_
      majorIsHypothesis := input.majorIsHypothesis
      minorIsHypothesis := input.minorIsHypothesis
      mixedHasSourceMark := producedCenter_is_mixed input
      majorFormula := ?_
      incomingExactly := ?_
      majorEdgeShape := ?_
      minorEdgeShape := ?_
      outgoingExactly := ?_
      sourceEdgeShape := ?_
      derivedEdgeShape := ?_
      majorTraceShape := ?_
      minorTraceShape := ?_
      majorTraceMember := ?_
      minorTraceMember := ?_ }
  all_goals
    simp [producedLocalGraph, neighborhoodGraph, producedNeighborhood,
      producedCenter,
      producedMajorEdge, producedMinorEdge, producedSourceEdge,
      producedDerivedEdge, producedMajorTrace, producedMinorTrace,
      input.majorFormula, input.sameFormula,
      input.positiveGeneratedColour, input.major_ne_minor,
      input.major_ne_derived, input.minor_ne_derived,
      input.sourceUse_ne_derived, input.derivedUse_ne_derived,
      input.major_ne_center, input.minor_ne_center,
      input.sourceUse_ne_center, input.derivedUse_ne_center,
      majorNotCenter, minorNotCenter, sourceUseNotCenter,
      derivedUseNotCenter,
      majorNotCenter', minorNotCenter', sourceUseNotCenter',
      derivedUseNotCenter', input.derivedNotSource,
      input.sourceIsHypothesis,
      get_rule.incoming, get_rule.incoming.loop,
      get_rule.outgoing, get_rule.outgoing.loop]
  rfl

@[simp] theorem producedRuntimeView_mixed {d : Graph}
    (input : ActualMixedHEInput d) :
    (producedRuntimeView input).mixed = producedCenter input := rfl

@[simp] theorem producedRuntimeView_major {d : Graph}
    (input : ActualMixedHEInput d) :
    (producedRuntimeView input).major = input.major := rfl

@[simp] theorem producedRuntimeView_minor {d : Graph}
    (input : ActualMixedHEInput d) :
    (producedRuntimeView input).minor = input.minor := rfl

@[simp] theorem producedRuntimeView_sourceUse {d : Graph}
    (input : ActualMixedHEInput d) :
    (producedRuntimeView input).sourceUse = input.sourceUse := rfl

@[simp] theorem producedRuntimeView_derivedUse {d : Graph}
    (input : ActualMixedHEInput d) :
    (producedRuntimeView input).derivedUse = input.derivedUse := rfl

set_option maxHeartbeats 1000000 in
/-- The original two occurrences and their four colour-zero edges furnish the
proof-only origin witness. -/
def producedOriginWitness {d : Graph} (input : ActualMixedHEInput d) :
    MixedOriginWitness d (producedLocalGraph input)
      (producedRuntimeView input) input.source input.derived := by
  refine
    { occurrencesDistinct := input.occurrencesDistinct.symm
      sourceMember := input.sourceMember
      derivedMember := input.derivedMember
      sourceWasHypothesis := input.sourceIsHypothesis
      derivedWasNotHypothesis := input.derivedNotSource
      sourceFormula := ?_
      derivedFormula := ?_
      sameLevel := input.sameLevel
      sourceContinuation := ?_
      derivedContinuation := ?_
      majorPremise := ?_
      minorPremise := ?_ }
  · simpa [producedCenter, collapse.center] using input.sameFormula
  · simp [producedCenter, collapse.center]
  · refine ⟨edge input.source input.sourceUse 0 input.sourceStored, ?_,
      rfl, rfl, rfl⟩
    exact input.sourceEdgeMember
  · refine ⟨edge input.derived input.derivedUse 0 input.derivedStored, ?_,
      rfl, rfl, rfl⟩
    exact input.derivedEdgeMember
  · refine ⟨edge input.major input.derived 0 input.majorStored, ?_,
      rfl, rfl, rfl⟩
    exact input.majorEdgeMember
  · refine ⟨edge input.minor input.derived 0 input.minorStored, ?_,
      rfl, rfl, rfl⟩
    exact input.minorEdgeMember

set_option maxHeartbeats 1000000 in
/-- Level 2: a parameterized call to the actual ordered producer establishes
the candidate contract from independent input-neighbourhood conditions. -/
theorem actualMixedHE_establishes_candidateContract
    {d : Graph} (input : ActualMixedHEInput d) :
    LocalMixedCorrespondence d (producedLocalGraph input)
      (producedRuntimeView input) :=
  mixedMerge_local_correspondence (producedRuntimeView input)
    (producedOriginWitness input)

end DLDS.Compression.Mixed.EliminationSource
