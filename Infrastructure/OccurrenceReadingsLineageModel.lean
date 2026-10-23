import Infrastructure.FlowBranchingGridBridge
import Infrastructure.DependencyGuardStability
import Infrastructure.SingleIntroductionReference

open Semantic FlowSpec SelfCertifying CertificateModel RouteIndexedMultiToken

namespace TaggedLineage








def elimCombine (d : Graph) (v major minor : Vertex)
    (minorTokens majorTokens : List RouteIndexedToken) :
    List RouteIndexedToken :=
  minorTokens.flatMap fun minorToken =>
    majorTokens.flatMap fun majorToken =>
      (edgesBetween d minor v).flatMap fun minorEdge =>
        (edgesBetween d major v).filterMap fun majorEdge =>
          if SingleIntroductionReference.guard d minorToken.dependency minorEdge &&
              SingleIntroductionReference.guard d majorToken.dependency majorEdge then
            match consume minorEdge minorToken.residual,
                consume majorEdge majorToken.residual with
            | some minorTail, some majorTail =>
                if minorTail = majorTail then
                  some <| mkToken d v
                    (.elimination v major minor minorEdge majorEdge
                      minorToken.routeKey majorToken.routeKey)
                    (depUnion minorToken.dependency majorToken.dependency)
                    minorTail
                else none
            | _, _ => none
          else none


def elimOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    List RouteIndexedToken :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    elimCombine d v major minor (premiseTokens minor) (premiseTokens major)

def introOutputsFor (d : Graph) (v premise : Vertex) (alpha : Formula)
    (tokens : List RouteIndexedToken) : List RouteIndexedToken :=
  tokens.flatMap fun token =>
    (edgesBetween d premise v).filterMap fun edge =>
      if SingleIntroductionReference.guard d token.dependency edge then
        (consume edge token.residual).map fun tail =>
          mkToken d v (.introduction v edge token.routeKey)
            (depRemove token.dependency alpha) tail
      else none

def introOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    List RouteIndexedToken :=
  match predsOf d v, v.FORMULA with
  | [premise], Formula.implication alpha beta =>
      if premise.FORMULA = beta then
        introOutputsFor d v premise alpha (premiseTokens premise)
      else []
  | _, _ => []

def mixedIntroOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    List RouteIndexedToken :=
  match v.FORMULA with
  | Formula.implication alpha beta =>
      (predsOf d v).flatMap fun premise =>
        if !elimPremiseAtB d v premise && premise.FORMULA == beta then
          (premiseTokens premise).flatMap fun token =>
            (edgesBetween d premise v).filterMap fun edge =>
              if SingleIntroductionReference.guard d token.dependency edge then
                (consume edge token.residual).map fun tail =>
                  mkToken d v (.mixedIntroduction v edge token.routeKey)
                    (depRemove token.dependency alpha) tail
              else none
        else []
  | _ => []

def elimReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) : List RouteIndexedToken :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    (premiseTokens minor).flatMap fun minorToken =>
      (premiseTokens major).flatMap fun majorToken =>
        (edgesBetween d minor v).flatMap fun minorEdge =>
          (edgesBetween d major v).flatMap fun majorEdge =>
            if reseedPremiseAgreesB minorToken.flowPair minorEdge &&
                reseedPremiseAgreesB majorToken.flowPair majorEdge &&
                SingleIntroductionReference.guard d minorToken.dependency minorEdge &&
                SingleIntroductionReference.guard d majorToken.dependency majorEdge then
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
            if reseedZeroPremiseB token.flowPair edge &&
                SingleIntroductionReference.guard d token.dependency edge then
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

end TaggedLineage
