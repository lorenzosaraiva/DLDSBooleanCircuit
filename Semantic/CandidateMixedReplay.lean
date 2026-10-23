import Semantic.ActualMixedProducerBridge
import Semantic.ActualMixedHIProducerBridge











open Semantic

namespace CandidateMixedReplay

open FlowSpec

inductive CandidateReplayError where
  | fuelExhausted (vertex : Vertex)
  | nonHypothesisLeaf (vertex : Vertex)
  | unsupportedMixedShape (vertex : Vertex)
  | unsupportedReseeding (vertex : Vertex)
  | unsupportedInferenceShape (vertex : Vertex)
deriving Repr, DecidableEq







def candidateSourceRoutes (d : Graph) (v : Vertex) : List FlowPair :=
  let ancestors := ancestorsInto d v
  let traced := ancestors.map fun ancestor =>
    (normalizeDep [v.FORMULA], ancestor.COLOURS)
  let hasDefaultUse :=
    (get_rule.outgoing v d).any fun edge => edge.COLOUR = 0
  if ancestors.isEmpty || (v.COLLAPSED && hasDefaultUse) then
    (traced ++ [(normalizeDep [v.FORMULA], [])]).eraseDups
  else traced.eraseDups



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




def candidateMixedDerivedRole? (d : Graph) (v : Vertex) :
    Option CandidateDerivedRole :=
  if mixedHERuntimeOKB d v then some .elimination
  else if ActualMixedHIProducerBridge.mixedHIRuntimeOKB d v then
    some .introduction
  else none




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




def candidateRootsCompleteB (d : Graph) (fuel : Nat) : Bool :=
  let roots := candidateRoots d
  !roots.isEmpty && roots.all fun root =>
    match candidateReplayAt d fuel root with
    | .error _ => false
    | .ok routes => !routes.isEmpty && routes.all candidateRouteClosedB

end CandidateMixedReplay
