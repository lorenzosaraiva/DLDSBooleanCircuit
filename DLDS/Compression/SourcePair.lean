import DLDS.Compression.Sequences
import Evaluator.Occurrence.Execution

set_option linter.unusedSimpArgs false

/-!
# Actual source/source compression bridge

This module covers the production-compressor case in which two top hypothesis
occurrences of one formula feed different horizontal-compression target
classes.  The actual producer discriminates the two outgoing uses.  The
successor replay then initializes one logical source dependency for each
producer-generated occurrence trace.  Stored edge dependencies play no role
in that replay theorem.

The hypotheses below are deliberately local.  This is not a claim that every
compression input has this shape.
-/

open DLDS Flow
open Evaluator.Mixed Evaluator.Occurrence

namespace DLDS.Compression.SourcePair

structure ActualSourcePairInput (d : Graph) where
  left : Vertex
  right : Vertex
  leftUse : Vertex
  rightUse : Vertex
  leftStored : List Formula
  rightStored : List Formula
  leftMember : left ∈ d.NODES
  rightMember : right ∈ d.NODES
  leftEdgeMember : edge left leftUse 0 leftStored ∈ d.EDGES
  rightEdgeMember : edge right rightUse 0 rightStored ∈ d.EDGES
  leftHypothesis : left.HYPOTHESIS = true
  rightHypothesis : right.HYPOTHESIS = true
  leftUncollapsed : left.COLLAPSED = false
  rightUncollapsed : right.COLLAPSED = false
  sameFormula : left.FORMULA = right.FORMULA
  sameLevel : left.LEVEL = right.LEVEL
  leftPositive : left.NUMBER ≠ 0
  rightPositive : right.NUMBER ≠ 0
  distinctNumbers : left.NUMBER ≠ right.NUMBER
  distinctOccurrences : left ≠ right
  leftUse_ne_center : leftUse ≠ collapse.center left right
  rightUse_ne_center : rightUse ≠ collapse.center left right
  leftUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) leftUse leftUse = true
  rightUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) rightUse rightUse = true
  differentUseRepresentatives :
    sameHCRepresentativeB (hcPartition d) leftUse rightUse = false
  leftRule : get_rule left d =
    rule left [] [edge left leftUse 0 leftStored] [] []
  rightRule : get_rule right d =
    rule right [] [edge right rightUse 0 rightStored] [] []

def producedCenter {d : Graph} (input : ActualSourcePairInput d) : Vertex :=
  collapse.center input.left input.right

def producedRightEdge {d : Graph} (input : ActualSourcePairInput d) : Deduction :=
  edge (producedCenter input) input.rightUse input.right.NUMBER input.rightStored

def producedLeftEdge {d : Graph} (input : ActualSourcePairInput d) : Deduction :=
  edge (producedCenter input) input.leftUse input.left.NUMBER input.leftStored

def producedRightTrace {d : Graph} (input : ActualSourcePairInput d) : Ancestral :=
  path input.rightUse (producedCenter input) [input.right.NUMBER]

def producedLeftTrace {d : Graph} (input : ActualSourcePairInput d) : Ancestral :=
  path input.leftUse (producedCenter input) [input.left.NUMBER]

def producedNeighborhood {d : Graph} (input : ActualSourcePairInput d) :
    Neighborhood :=
  rule (producedCenter input) []
    [producedRightEdge input, producedLeftEdge input]
    [producedRightTrace input, producedLeftTrace input] []

def actualSourcePairOperation {d : Graph} (input : ActualSourcePairInput d) :
    Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.left d))
    (pre_collapse (get_rule input.right d))

theorem actualSourcePairOperation_eq_produced
    {d : Graph} (input : ActualSourcePairInput d) :
    actualSourcePairOperation input =
      (producedNeighborhood input,
        [producedRightTrace input, producedLeftTrace input]) := by
  have outgoingNe : producedRightEdge input ≠ producedLeftEdge input := by
    intro equal
    exact input.distinctNumbers.symm (congrArg Deduction.COLOUR equal)
  have outgoingBeq :
      (producedLeftEdge input == producedRightEdge input) = false :=
    beq_eq_false_iff_ne.mpr outgoingNe.symm
  have outgoingDedup :
      List.eraseDups [producedRightEdge input, producedLeftEdge input] =
        [producedRightEdge input, producedLeftEdge input] := by
    simp [List.eraseDups_cons, List.filter, outgoingBeq]
  have reverseNumbers : input.right.NUMBER ≠ input.left.NUMBER :=
    input.distinctNumbers.symm
  simp [actualSourcePairOperation, input.leftRule, input.rightRule,
    pre_collapse, pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.discriminatePair,
    pre_collapse.outgoingRepresentativeConflictB,
    pre_collapse.discriminateHypothesis,
    collapseWithOccurrences, collapse.center, collapse.rewrite_incoming,
    collapse.rewrite_outgoing, collapse.rewrite_direct,
    collapse.rewrite_direct.rewrite_head,
    collapse.checkNeighborhoodPathHeads, collapse.neighborhoodPathHeadsOKB,
    collapse.neighborhoodPathHeadOKB,
    collapse.checkOutgoingRepresentatives,
    collapse.outgoingRepresentativesOKB, producedNeighborhood,
    producedCenter, producedRightEdge, producedLeftEdge,
    producedRightTrace, producedLeftTrace,
    input.leftHypothesis, input.rightHypothesis,
    input.leftUncollapsed, input.rightUncollapsed,
    input.leftPositive, input.rightPositive, input.distinctNumbers,
    reverseNumbers,
    input.leftUseSelfRepresentative, input.rightUseSelfRepresentative,
    input.differentUseRepresentatives]
  simpa [producedRightEdge, producedLeftEdge, producedCenter,
    collapse.center, input.leftHypothesis, input.rightHypothesis] using
    outgoingDedup

def producedLocalGraph {d : Graph} (input : ActualSourcePairInput d) : Graph :=
  DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph (producedNeighborhood input)

theorem producedLocalGraph_eq_actualProjection
    {d : Graph} (input : ActualSourcePairInput d) :
    producedLocalGraph input =
      DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph
        (actualSourcePairOperation input).1 := by
  rw [actualSourcePairOperation_eq_produced]
  rfl

theorem actualSourcePair_candidateSourceRoutes
    {d : Graph} (input : ActualSourcePairInput d) :
    candidateSourceRoutes (producedLocalGraph input) (producedCenter input) =
      [ (normalizeDep [input.left.FORMULA], [input.right.NUMBER]),
        (normalizeDep [input.left.FORMULA], [input.left.NUMBER]) ] := by
  have rightNotCenter : input.rightUse ≠ producedCenter input :=
    input.rightUse_ne_center
  have leftNotCenter : input.leftUse ≠ producedCenter input :=
    input.leftUse_ne_center
  have routeNe :
      (normalizeDep [input.left.FORMULA], [input.right.NUMBER]) ≠
        (normalizeDep [input.left.FORMULA], [input.left.NUMBER]) := by
    intro equal
    have colours := congrArg Prod.snd equal
    simp at colours
    exact input.distinctNumbers.symm colours
  have routeBeq :
      ((normalizeDep [input.left.FORMULA], [input.left.NUMBER]) ==
        (normalizeDep [input.left.FORMULA], [input.right.NUMBER])) = false :=
    beq_eq_false_iff_ne.mpr routeNe.symm
  have routeBeqRaw :
      (([input.left.FORMULA], [input.left.NUMBER]) ==
        ([input.left.FORMULA], [input.right.NUMBER])) = false := by
    simpa [normalizeDep] using routeBeq
  have routeDedup :
      List.eraseDups
        [ (normalizeDep [input.left.FORMULA], [input.right.NUMBER]),
          (normalizeDep [input.left.FORMULA], [input.left.NUMBER]) ] =
        [ (normalizeDep [input.left.FORMULA], [input.right.NUMBER]),
          (normalizeDep [input.left.FORMULA], [input.left.NUMBER]) ] := by
    simp [List.eraseDups_cons, List.filter, routeBeqRaw]
  simp [candidateSourceRoutes, Flow.ancestorsInto,
    get_rule.outgoing, get_rule.outgoing.loop,
    producedLocalGraph, DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph,
    producedNeighborhood, producedCenter, producedRightTrace,
    producedLeftTrace, producedRightEdge, producedLeftEdge,
    rightNotCenter, leftNotCenter,
    input.leftHypothesis, input.rightHypothesis,
    input.leftPositive, input.rightPositive,
    collapse.center, normalizeDep]
  simpa [normalizeDep] using routeDedup

theorem actualSourcePair_occurrenceStep
    {d : Graph} (input : ActualSourcePairInput d)
    (premiseFlow : Vertex → List FlowPair) :
    occurrenceStep (producedLocalGraph input) (producedCenter input)
        premiseFlow =
      .ok
        [ (normalizeDep [input.left.FORMULA], [input.right.NUMBER]),
          (normalizeDep [input.left.FORMULA], [input.left.NUMBER]) ] := by
  have sourceRoutes := actualSourcePair_candidateSourceRoutes input
  have noPredecessors :
      predsOf (producedLocalGraph input) (producedCenter input) = [] := by
    simp [Flow.predsOf_eq, producedLocalGraph,
      DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph, producedNeighborhood,
      producedCenter, producedRightEdge, producedLeftEdge,
      get_rule.incoming, get_rule.incoming.loop,
      input.leftUse_ne_center, input.rightUse_ne_center]
  have centerHypothesis : (producedCenter input).HYPOTHESIS = true := by
    simp [producedCenter, collapse.center, input.leftHypothesis]
  simp [occurrenceStep, noPredecessors, centerHypothesis, sourceRoutes]

/-! Each operational dependency is derived from the source formula.  In
particular, neither `leftStored` nor `rightStored` occurs in the conclusion. -/
theorem actualSourcePair_dependencies_recomputed
    {d : Graph} (input : ActualSourcePairInput d) (route : FlowPair)
    (member : route ∈
      [ (normalizeDep [input.left.FORMULA], [input.right.NUMBER]),
        (normalizeDep [input.left.FORMULA], [input.left.NUMBER]) ]) :
    route.1 = normalizeDep [input.left.FORMULA] := by
  simp [normalizeDep] at member ⊢
  rcases member with rfl | rfl <;> rfl

/-! Once a surrounding observer proves the source/source boundary equality,
the local theorem is a literal step of `compressWithOccurrences`, including
the two newly minted occurrence records. -/
theorem actualSourcePair_two_rule_trace
    {d : Graph} (input : ActualSourcePairInput d)
    (observe : Neighborhood →
      List DLDS.Compression.Mixed.ContextualElimination.BoundaryObservation)
    (preserved :
      observe (pre_collapse (get_rule input.left d)) ++
          observe (pre_collapse (get_rule input.right d)) =
        observe (producedNeighborhood input)) :
    DLDS.Compression.Sequences.ActualCompressionTrace
      (hcPartition d) observe
      [pre_collapse (get_rule input.left d),
       pre_collapse (get_rule input.right d)]
      [] (producedNeighborhood input)
      [producedRightTrace input, producedLeftTrace input] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.left d))
          (pre_collapse (get_rule input.right d)) =
        (producedNeighborhood input,
          [producedRightTrace input, producedLeftTrace input]) := by
    simpa [actualSourcePairOperation] using
      actualSourcePairOperation_eq_produced input
  apply DLDS.Compression.Sequences.ActualCompressionTrace.step
  · rw [operation]
    exact preserved
  · rw [operation]
    simpa using
      (DLDS.Compression.Sequences.ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences :=
          [producedRightTrace input, producedLeftTrace input])
        (producedNeighborhood input))

/-!
## Same representative, distinct uses

When the two uses already belong to one HC target class, the production
compressor deliberately leaves both edges at colour zero.  A single source
route then serves both uses.  This is semantic deduplication of identical
logical source productions, not loss of a dependency history.
-/

namespace SameTarget

structure Input (d : Graph) where
  left : Vertex
  right : Vertex
  leftUse : Vertex
  rightUse : Vertex
  leftStored : List Formula
  rightStored : List Formula
  leftHypothesis : left.HYPOTHESIS = true
  rightHypothesis : right.HYPOTHESIS = true
  leftUncollapsed : left.COLLAPSED = false
  rightUncollapsed : right.COLLAPSED = false
  sameFormula : left.FORMULA = right.FORMULA
  sameLevel : left.LEVEL = right.LEVEL
  distinctOccurrences : left ≠ right
  distinctUses : leftUse ≠ rightUse
  leftUse_ne_center : leftUse ≠ collapse.center left right
  rightUse_ne_center : rightUse ≠ collapse.center left right
  leftUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) leftUse leftUse = true
  rightUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) rightUse rightUse = true
  sameRepresentativeLR :
    sameHCRepresentativeB (hcPartition d) leftUse rightUse = true
  sameRepresentativeRL :
    sameHCRepresentativeB (hcPartition d) rightUse leftUse = true
  leftRule : get_rule left d =
    rule left [] [edge left leftUse 0 leftStored] [] []
  rightRule : get_rule right d =
    rule right [] [edge right rightUse 0 rightStored] [] []

def center {d : Graph} (input : Input d) : Vertex :=
  collapse.center input.left input.right

def rightEdge {d : Graph} (input : Input d) : Deduction :=
  edge (center input) input.rightUse 0 input.rightStored

def leftEdge {d : Graph} (input : Input d) : Deduction :=
  edge (center input) input.leftUse 0 input.leftStored

def neighborhood {d : Graph} (input : Input d) : Neighborhood :=
  rule (center input) [] [rightEdge input, leftEdge input] [] []

def operation {d : Graph} (input : Input d) :
    Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.left d))
    (pre_collapse (get_rule input.right d))

theorem operation_eq_produced {d : Graph} (input : Input d) :
    operation input = (neighborhood input, []) := by
  have edgeNe : rightEdge input ≠ leftEdge input := by
    intro equal
    exact input.distinctUses.symm (congrArg Deduction.END equal)
  have edgeBeq : (leftEdge input == rightEdge input) = false :=
    beq_eq_false_iff_ne.mpr edgeNe.symm
  have edgesDedup :
      List.eraseDups [rightEdge input, leftEdge input] =
        [rightEdge input, leftEdge input] := by
    simp [List.eraseDups_cons, List.filter, edgeBeq]
  simp [operation, input.leftRule, input.rightRule,
    pre_collapse, pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.discriminatePair,
    pre_collapse.outgoingRepresentativeConflictB,
    collapseWithOccurrences, collapse.center, collapse.rewrite_incoming,
    collapse.rewrite_outgoing, collapse.rewrite_direct,
    collapse.checkNeighborhoodPathHeads, collapse.neighborhoodPathHeadsOKB,
    collapse.neighborhoodPathHeadOKB,
    collapse.checkOutgoingRepresentatives,
    collapse.outgoingRepresentativesOKB,
    neighborhood, center, rightEdge, leftEdge,
    input.leftHypothesis, input.rightHypothesis,
    input.leftUncollapsed, input.rightUncollapsed,
    input.leftUseSelfRepresentative, input.rightUseSelfRepresentative,
    input.sameRepresentativeLR, input.sameRepresentativeRL]
  simpa [rightEdge, leftEdge, center, collapse.center,
    input.leftHypothesis, input.rightHypothesis] using edgesDedup

def localGraph {d : Graph} (input : Input d) : Graph :=
  DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph (neighborhood input)

theorem candidateSourceRoutes_eq_singleton {d : Graph} (input : Input d) :
    candidateSourceRoutes (localGraph input) (center input) =
      [(normalizeDep [input.left.FORMULA], [])] := by
  simp [candidateSourceRoutes, Flow.ancestorsInto,
    get_rule.outgoing, get_rule.outgoing.loop,
    localGraph, DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph,
    neighborhood, center, rightEdge, leftEdge,
    input.leftHypothesis, input.rightHypothesis,
    collapse.center]
  rfl

theorem occurrenceStep_eq_singleton
    {d : Graph} (input : Input d) (premiseFlow : Vertex → List FlowPair) :
    occurrenceStep (localGraph input) (center input) premiseFlow =
      .ok [(normalizeDep [input.left.FORMULA], [])] := by
  have sourceRoutes := candidateSourceRoutes_eq_singleton input
  have noPredecessors : predsOf (localGraph input) (center input) = [] := by
    simp [Flow.predsOf_eq, localGraph,
      DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph, neighborhood,
      center, rightEdge, leftEdge,
      get_rule.incoming, get_rule.incoming.loop,
      input.leftUse_ne_center, input.rightUse_ne_center]
  have source : (center input).HYPOTHESIS = true := by
    simp [center, collapse.center, input.leftHypothesis]
  simp [occurrenceStep, noPredecessors, source, sourceRoutes]

end SameTarget

end DLDS.Compression.SourcePair
