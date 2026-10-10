import DLDS.Compression.Mixed.EliminationSource
import DLDS.Compression.Mixed.IntroductionSource

/-!
# Partial whole-graph replay for the candidate mixed contract

This module is deliberately parallel to historical `flowAt`.  It supports the
ordinary source/elimination/introduction shapes needed by the connected mixed
fixture and one graph-recognisable mixed hypothesis/elimination shape.  Every
other shape returns an explicit error.  Stored `Deduction.DEPENDENCY` fields
are never read.
-/

open DLDS

namespace Evaluator.Mixed

open Flow

inductive CandidateReplayError where
  | fuelExhausted (vertex : Vertex)
  | nonHypothesisLeaf (vertex : Vertex)
  | unsupportedMixedShape (vertex : Vertex)
  | unsupportedReseeding (vertex : Vertex)
  | unsupportedInferenceShape (vertex : Vertex)
deriving Repr, DecidableEq

/-- Source initialisation follows the candidate edge-trace convention.  A
plain source starts with the singleton formula dependency and empty residual;
an occurrence trace supplies the residual colours.  Only a collapsed source
center may retain explicit old occurrences while also exposing a newest
colour-zero default occurrence.  An uncollapsed premise source with an
ancestor trace has no additional default occurrence. -/
def candidateSourceRoutes (d : Graph) (v : Vertex) : List FlowPair :=
  let ancestors := ancestorsInto d v
  let traced := ancestors.map fun ancestor =>
    (normalizeDep [v.FORMULA], ancestor.COLOURS)
  let hasDefaultUse :=
    (get_rule.outgoing v d).any fun edge => edge.COLOUR = 0
  if ancestors.isEmpty || (v.COLLAPSED && hasDefaultUse) then
    (traced ++ [(normalizeDep [v.FORMULA], [])]).eraseDups
  else traced.eraseDups

/-- The ordinary elimination kernel.  It recomputes dependency union and
consumes the actual premise edges. -/
def candidateEliminationOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) : List FlowPair :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    (premiseFlow minor).flatMap fun (minorDep, minorResidual) =>
      (premiseFlow major).flatMap fun (majorDep, majorResidual) =>
        (edgesBetween d minor v).flatMap fun minorEdge =>
          (edgesBetween d major v).filterMap fun majorEdge =>
            match consume minorEdge minorResidual,
                consume majorEdge majorResidual with
            | some minorTail, some majorTail =>
                if minorTail = majorTail then
                  some (depUnion minorDep majorDep, minorTail)
                else none
            | _, _ => none

/-- The ordinary one-premise implication-introduction kernel. -/
def candidateIntroductionOutputs (d : Graph) (v premise : Vertex)
    (premiseRoutes : List FlowPair) : List FlowPair :=
  match v.FORMULA with
  | Formula.implication antecedent consequent =>
      if premise.FORMULA = consequent then
        premiseRoutes.flatMap fun (dependency, residual) =>
          (edgesBetween d premise v).filterMap fun edge =>
            (consume edge residual).map fun tail =>
              (depRemove dependency antecedent, tail)
      else []
  | _ => []

/-- Runtime recognition of the supported mixed HE shape.  It uses only the
compressed graph: one elimination pair, exactly its two incoming premises,
one colour-zero source continuation, one nonzero derived continuation, and
the two producer traces `[0,k]`.  No original-tree role map and no dependency
label is consulted. -/
def mixedHERuntimeOKB (d : Graph) (v : Vertex) : Bool :=
  let predecessors := predsOf d v
  let outgoing := get_rule.outgoing v d
  let zeroOutgoing := outgoing.filter fun edge => edge.COLOUR = 0
  let colouredOutgoing := outgoing.filter fun edge => edge.COLOUR ≠ 0
  match elimPairsAt d v, zeroOutgoing, colouredOutgoing with
  | [(major, minor)], [sourceEdge], [derivedEdge] =>
      match edgesBetween d major v, edgesBetween d minor v with
      | [majorEdge], [minorEdge] =>
          v.HYPOTHESIS &&
          !predecessors.isEmpty &&
          predecessors.all (elimPremiseAtB d v) &&
          major.HYPOTHESIS && minor.HYPOTHESIS &&
          (majorEdge.COLOUR == 0) && (minorEdge.COLOUR == 0) &&
          (sourceEdge.START == v) && (derivedEdge.START == v) &&
          (derivedEdge.COLOUR != 0) &&
          (ancestorsInto d v).isEmpty &&
          d.PATHS.contains
            (path derivedEdge.END major [0, derivedEdge.COLOUR]) &&
          d.PATHS.contains
            (path derivedEdge.END minor [0, derivedEdge.COLOUR])
      | _, _ => false
  | _, _, _ => false

def routesFor (table : List (Vertex × List FlowPair)) (v : Vertex) :
    List FlowPair :=
  match table.find? fun entry => entry.1 == v with
  | some entry => entry.2
  | none => []

inductive CandidateDerivedRole where
  | introduction
  | elimination
deriving Repr, DecidableEq

/-- Runtime classification of the derived contribution at a mixed vertex.
The source contribution is orthogonal and is added by `candidateReplayAt`.
Unsupported or ambiguous shapes remain visible as `none`. -/
def candidateMixedDerivedRole? (d : Graph) (v : Vertex) :
    Option CandidateDerivedRole :=
  if mixedHERuntimeOKB d v then some .elimination
  else if DLDS.Compression.Mixed.IntroductionSource.mixedHIRuntimeOKB d v then
    some .introduction
  else none

/-- Common logical kernel for the currently supported mixed productions.
Both cases reuse the ordinary dependency computation above; neither reads a
stored edge dependency. -/
def candidateMixedDerivedOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) :
    Except CandidateReplayError (List FlowPair) :=
  match candidateMixedDerivedRole? d v with
  | some .elimination =>
      .ok (candidateEliminationOutputs d v premiseFlow)
  | some .introduction =>
      match predsOf d v with
      | [premise] =>
          .ok (candidateIntroductionOutputs d v premise
            (premiseFlow premise))
      | _ => .error (.unsupportedMixedShape v)
  | none => .error (.unsupportedMixedShape v)

/-- A visibly partial recursive replay.  Unlike historical `flowAt`, a mixed
source mark is not treated as an exclusive source: after structural runtime
recognition it contributes both the formula-derived source route and the
logically recomputed elimination routes. -/
def candidateReplayAt (d : Graph) :
    Nat → Vertex → Except CandidateReplayError (List FlowPair)
  | 0, vertex => .error (.fuelExhausted vertex)
  | fuel + 1, vertex => do
      let predecessors := predsOf d vertex
      if predecessors.isEmpty then
        if vertex.HYPOTHESIS then
          return candidateSourceRoutes d vertex
        else
          throw (.nonHypothesisLeaf vertex)
      let premiseTable ← predecessors.mapM fun premise => do
        let routes ← candidateReplayAt d fuel premise
        return (premise, routes)
      let premiseFlow := routesFor premiseTable
      if vertex.HYPOTHESIS then
        let derived ← candidateMixedDerivedOutputs d vertex premiseFlow
        return ([(normalizeDep [vertex.FORMULA], [])] ++ derived).eraseDups
      if !(ancestorsInto d vertex).isEmpty then
        throw (.unsupportedReseeding vertex)
      if !(elimPairsAt d vertex).isEmpty then
        if predecessors.all (elimPremiseAtB d vertex) then
          return (candidateEliminationOutputs d vertex premiseFlow).eraseDups
        else
          throw (.unsupportedInferenceShape vertex)
      match predecessors with
      | [premise] =>
          match vertex.FORMULA with
          | Formula.implication _ consequent =>
              if premise.FORMULA = consequent then
                return (candidateIntroductionOutputs d vertex premise
                  (premiseFlow premise)).eraseDups
              else
                throw (.unsupportedInferenceShape vertex)
          | _ => throw (.unsupportedInferenceShape vertex)
      | _ => throw (.unsupportedInferenceShape vertex)

def candidateRoots (d : Graph) : List Vertex :=
  d.NODES.filter fun vertex => (get_rule.outgoing vertex d).isEmpty

def candidateRouteClosedB (route : FlowPair) : Bool :=
  route.1.isEmpty && route.2.isEmpty

/-- Completion is checked on every raw route at every root.  Empty root sets,
empty route sets, and replay errors all reject.  No witnessability filter is
applied. -/
def candidateRootsCompleteB (d : Graph) (fuel : Nat) : Bool :=
  let roots := candidateRoots d
  !roots.isEmpty && roots.all fun root =>
    match candidateReplayAt d fuel root with
    | .error _ => false
    | .ok routes => !routes.isEmpty && routes.all candidateRouteClosedB

end Evaluator.Mixed
