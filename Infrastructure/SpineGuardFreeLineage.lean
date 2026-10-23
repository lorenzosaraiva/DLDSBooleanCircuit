import Infrastructure.FlowBranchingGridBridge
import Infrastructure.DependencyGuardStability

open Semantic FlowSpec SelfCertifying CertificateModel RouteIndexedMultiToken

namespace SpineLineage








def elimReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) : List RouteIndexedToken :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    (premiseTokens minor).flatMap fun minorToken =>
      (premiseTokens major).flatMap fun majorToken =>
        (edgesBetween d minor v).flatMap fun minorEdge =>
          (edgesBetween d major v).flatMap fun majorEdge =>
            if reseedPremiseAgreesB minorToken.flowPair minorEdge &&
                reseedPremiseAgreesB majorToken.flowPair majorEdge then
              ancestors.map fun ancestor =>
                mkToken d v
                  (.elimReseed v major minor minorEdge majorEdge ancestor
                    minorToken.routeKey majorToken.routeKey)
                  (depUnion minorToken.dependency majorToken.dependency)
                  ancestor.COLOURS
            else []

def introReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) : List RouteIndexedToken :=
  match predsOf d v, v.FORMULA with
  | [premise], Formula.implication alpha beta =>
      if premise.FORMULA == beta then
        (premiseTokens premise).flatMap fun token =>
          (edgesBetween d premise v).flatMap fun edge =>
            if reseedZeroPremiseB token.flowPair edge then
              ancestors.map fun ancestor =>
                mkToken d v (.introReseed v edge ancestor token.routeKey)
                  (depRemove token.dependency alpha) ancestor.COLOURS
            else []
      else []
  | _, _ => []



def reseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    List RouteIndexedToken :=
  let ancestors := eligibleAncestorsAtRoot d v
  if ancestors.isEmpty then []
  else elimReseedOutputs d v premiseTokens ancestors ++
    introReseedOutputs d v premiseTokens ancestors


def rawStepFrom (d : Graph)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (v : Vertex) : List RouteIndexedToken :=
  let preds := predsOf d v
  if v.HYPOTHESIS || preds.isEmpty then
    sourceOutputs d v
  else
    let elimination := elimOutputs d v premiseTokens
    let introduction := introOutputs d v premiseTokens
    let core := if (elimPairsAt d v).isEmpty then introduction
      else elimination ++ mixedIntroOutputs d v premiseTokens
    core ++ reseedOutputs d v premiseTokens


def stepFrom (d : Graph) (premiseTokens : Vertex → List RouteIndexedToken)
    (v : Vertex) : List RouteIndexedToken :=
  assignSlots v (rawStepFrom d premiseTokens v)


def evaluateAt (d : Graph) : Nat → Vertex → List RouteIndexedToken
  | 0, _ => []
  | fuel + 1, v => stepFrom d (evaluateAt d fuel) v

end SpineLineage
