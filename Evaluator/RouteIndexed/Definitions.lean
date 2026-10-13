import Certificate.Flow.RootDischarge

open DLDS FlowSemantics Certificate.Flow Certificate.Flow.Foundation

namespace Evaluator.RouteIndexed

/-!
# Route-indexed multi-token semantics

This module is deliberately parallel to the historical grid and `MultiPath`
evaluators.  A key is an intensional derivation lineage: it never contains the
dependency which the evaluator is meant to compute.  In particular, an
elimination key records both premise occurrences and both concrete incoming
edges, so two compatible Cartesian-product choices at one compressed vertex
remain different occurrences even when their grid coordinates coincide.
-/

inductive RouteKey where
  | source (vertex : Vertex) (seed : Option Ancestral)
  | certificate (vertex : Vertex) (slot : Nat)
  | slot (vertex : Vertex) (ordinal : Nat) (lineage : RouteKey)
  | repetition (vertex : Vertex) (edge : Deduction) (parent : RouteKey)
  | introduction (vertex : Vertex) (edge : Deduction) (parent : RouteKey)
  | mixedIntroduction (vertex : Vertex) (edge : Deduction) (parent : RouteKey)
  | elimination (vertex major minor : Vertex)
      (minorEdge majorEdge : Deduction) (minorKey majorKey : RouteKey)
  | introReseed (vertex : Vertex) (edge : Deduction)
      (ancestor : Ancestral) (parent : RouteKey)
  | elimReseed (vertex major minor : Vertex)
      (minorEdge majorEdge : Deduction) (ancestor : Ancestral)
      (minorKey majorKey : RouteKey)
  deriving Repr, BEq, DecidableEq

/-- One operational occurrence.  `vertex`, `level`, and `column` are routing
metadata; dependency and residual are the semantic payload projected to Flow. -/
structure RouteIndexedToken where
  routeKey : RouteKey
  vertex : Vertex
  level : Nat
  column : Nat
  dependency : Dep
  residual : ColourPath
  deriving Repr, BEq, DecidableEq

def mkToken (d : Graph) (v : Vertex) (key : RouteKey)
    (dependency : Dep) (residual : ColourPath) : RouteIndexedToken :=
  { routeKey := key
    vertex := v
    level := v.LEVEL
    column := (buildFormulas d).idxOf v.FORMULA
    dependency := dependency
    residual := residual }

def RouteIndexedToken.flowPair (token : RouteIndexedToken) : FlowPair :=
  (token.dependency, token.residual)

def projectFlowPairs (tokens : List RouteIndexedToken) : List FlowPair :=
  tokens.map Evaluator.RouteIndexed.RouteIndexedToken.flowPair

/-- Source/hypothesis production: every actual ancestral seed gets its own
route key.  A source without an ancestral record has one distinguished seed. -/
def sourceOutputs (d : Graph) (v : Vertex) : List RouteIndexedToken :=
  let dependency := normalizeDep [v.FORMULA]
  let ancestors := ancestorsInto d v
  if ancestors.isEmpty then
    [mkToken d v (.source v none) dependency []]
  else
    ancestors.map fun ancestor =>
      mkToken d v (.source v (some ancestor)) dependency ancestor.COLOURS

/-- A standalone repetition move.  Flow has no repetition arm at a non-source
vertex, so this operation is not used by `stepFrom`; it records the operational
invariant needed by a future grid presentation of this evaluator. -/
def repetitionOutputs (d : Graph) (v : Vertex) (edge : Deduction)
    (tokens : List RouteIndexedToken) : List RouteIndexedToken :=
  tokens.map fun token =>
    mkToken d v (.repetition v edge token.routeKey)
      token.dependency token.residual

/-- One elimination occurrence.  This is the genuine compatible Cartesian
product: no premise or result dependencies are aggregated. -/
def elimCombine (d : Graph) (v major minor : Vertex)
    (minorTokens majorTokens : List RouteIndexedToken) :
    List RouteIndexedToken :=
  minorTokens.flatMap fun minorToken =>
    majorTokens.flatMap fun majorToken =>
      (edgesBetween d minor v).flatMap fun minorEdge =>
        (edgesBetween d major v).filterMap fun majorEdge =>
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

def elimOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    List RouteIndexedToken :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    elimCombine d v major minor (premiseTokens minor) (premiseTokens major)

def introOutputsFor (d : Graph) (v premise : Vertex) (alpha : Formula)
    (tokens : List RouteIndexedToken) : List RouteIndexedToken :=
  tokens.flatMap fun token =>
    (edgesBetween d premise v).filterMap fun edge =>
      (consume edge token.residual).map fun tail =>
        mkToken d v (.introduction v edge token.routeKey)
          (depRemove token.dependency alpha) tail

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
              (consume edge token.residual).map fun tail =>
                mkToken d v (.mixedIntroduction v edge token.routeKey)
                  (depRemove token.dependency alpha) tail
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
                depSimilarB minorToken.dependency minorEdge.DEPENDENCY &&
                depSimilarB majorToken.dependency majorEdge.DEPENDENCY then
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
                depSimilarB token.dependency edge.DEPENDENCY then
              ancestors.map fun ancestor =>
                mkToken d v (.introReseed v edge ancestor token.routeKey)
                  (depRemove token.dependency alpha) ancestor.COLOURS
            else []
      else []
  | _, _ => []

/-- Raw occurrence-preserving reseeding.  Unlike Flow's list representation,
we intentionally do not erase tokens which have different lineage keys. -/
def reseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    List RouteIndexedToken :=
  let ancestors := eligibleAncestorsAtRoot d v
  if ancestors.isEmpty then []
  else elimReseedOutputs d v premiseTokens ancestors ++
    introReseedOutputs d v premiseTokens ancestors

/-- Assign a stable finite occurrence slot after raw local enumeration.  The
slot is independent of dependency and also distinguishes duplicate graph-list
records which happen to carry equal structural values. -/
def assignSlotsFrom (v : Vertex) : Nat →
    List RouteIndexedToken → List RouteIndexedToken
  | _, [] => []
  | ordinal, token :: tokens =>
      { token with routeKey := .slot v ordinal token.routeKey } ::
        assignSlotsFrom v (ordinal + 1) tokens

def assignSlots (v : Vertex) (tokens : List RouteIndexedToken) :
    List RouteIndexedToken :=
  assignSlotsFrom v 0 tokens

/-- The raw route-sensitive local Flow kernel, before stable slot assignment. -/
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

/-- The route-sensitive local Flow kernel. -/
def stepFrom (d : Graph) (premiseTokens : Vertex → List RouteIndexedToken)
    (v : Vertex) : List RouteIndexedToken :=
  assignSlots v (rawStepFrom d premiseTokens v)

/-- Filtering happens after raw route production and preserves the route key. -/
def witnessableAt (d : Graph) (v : Vertex)
    (tokens : List RouteIndexedToken) : List RouteIndexedToken :=
  tokens.filter fun token => routeContinuableAtB d v token.residual

/-- Fuelled whole-graph evaluator, operationally mirroring `flowAt`. -/
def evaluateAt (d : Graph) : Nat → Vertex → List RouteIndexedToken
  | 0, _ => []
  | fuel + 1, v => stepFrom d (evaluateAt d fuel) v

def canonicalAt (d : Graph) (v : Vertex) : List RouteIndexedToken :=
  evaluateAt d (stdFuel d) v

/-- A finite certificate-side carrier.  Slots identify stored route records,
but their dependency fields are only premise inputs to the local logical
kernel; outputs are recomputed by `stepFrom`. -/
def tokensFromStored (c : Certificate) (v : Vertex) : List RouteIndexedToken :=
  (StoredRoutes c v).zipIdx.map fun (route, slot) =>
    mkToken c.graph v (.certificate v slot) route.1 route.2

def certificateLocalOutputs (c : Certificate) (v : Vertex) :
    List RouteIndexedToken :=
  stepFrom c.graph (tokensFromStored c) v

end Evaluator.RouteIndexed
