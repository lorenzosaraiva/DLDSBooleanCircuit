import Semantic.CacheCorrectness

open Semantic FlowSpec SelfCertifyingProbe SelfCertifying

namespace Decoration

structure GuardTest where
  node : Vertex
  edge : Deduction
  route : FlowPair
  kind : String

def GuardTest.passes (test : GuardTest) : Bool :=
  depSimilarB test.route.1 test.edge.DEPENDENCY





def testsAt (d : Graph) (prior : Vertex → List FlowPair) (v : Vertex) :
    List GuardTest :=
  if v.HYPOTHESIS || (predsOf d v).isEmpty ||
      (eligibleAncestorsAtRoot d v).isEmpty then []
  else
    let elimination := (elimPairsAt d v).flatMap fun (major, minor) =>
      (prior minor).flatMap fun left => (prior major).flatMap fun right =>
        (edgesBetween d minor v).flatMap fun eMinor =>
          (edgesBetween d major v).flatMap fun eMajor =>
            if reseedPremiseAgreesB left eMinor && reseedPremiseAgreesB right eMajor then
              [⟨v, eMinor, left, "elimination-minor"⟩,
               ⟨v, eMajor, right, "elimination-major"⟩]
            else []
    let introduction := match predsOf d v, v.FORMULA with
      | [u], .implication _ beta =>
          if u.FORMULA == beta then
            (prior u).flatMap fun route =>
              (edgesBetween d u v).filterMap fun edge =>
                if reseedZeroPremiseB route edge then
                  some ⟨v, edge, route, "introduction"⟩ else none
          else []
      | _, _ => []
    elimination ++ introduction

def tableRoutes (table : FlowTable) (v : Vertex) : List FlowPair :=
  (flowTableLookup? table v).getD []

def guardTests (d : Graph) (table : FlowTable) : List GuardTest :=
  d.NODES.flatMap fun v => testsAt d (tableRoutes table) v

def guardTableB (d : Graph) (table : FlowTable) : Bool :=
  (guardTests d table).all GuardTest.passes




def GuardFaithful (d : Graph) : Prop :=
  flowTableClosedB d = true ∧
    ∀ fuel, guardTableB d (CacheCorrectness.flowTableSpec d fuel) = true





def guardLoop (d : Graph) : Nat → Nat → FlowTable → Bool
  | remaining, fuel, previous =>
      guardTableB d previous &&
        (let next := flowTableStep d fuel previous
         if next == previous then true
         else match remaining with
           | 0 => false
           | rest + 1 => guardLoop d rest (fuel + 1) next)

def GuardFaithfulB (d : Graph) : Bool :=
  flowTableClosedB d && guardLoop d (stdFuel d) 0 (flowTableInitial d)

inductive PaperLabel where
  | bits (dependency : Dep)
  | lambda

def exactLabelB (label : PaperLabel) (dependency : Dep) : Bool :=
  match label with
  | .bits stored => depSetEq dependency stored
  | .lambda => false





def paperDecorationAtB (d : Graph) (labels : Deduction → PaperLabel)
    (v : Vertex) (raw : List FlowPair) : Bool :=
  let routes := raw.eraseDups
  let outgoing := (get_rule.outgoing v d).eraseDups
  match routes with
  | [] => true
  | [route] => match outgoing with
      | [outEdge] => outEdge.COLOUR == headColour route.2 && exactLabelB (labels outEdge) route.1
      | _ => false
  | _ :: _ :: _ =>
      let colours := (routes.map fun route => headColour route.2).eraseDups
      (colours.all fun colour =>
        let group := routes.filter fun route => headColour route.2 == colour
        match outgoing.filter (fun edge => edge.COLOUR == colour) with
        | [outEdge] => match group with
            | [route] => exactLabelB (labels outEdge) route.1
            | _ :: _ :: _ => match labels outEdge with
                | .lambda => true | .bits _ => false
            | [] => false
        | _ => false) &&
        outgoing.all fun edge => colours.contains edge.COLOUR

def paperDecorationB (d : Graph) (labels : Deduction → PaperLabel)
    (paperFlow : Vertex → Vertex → List FlowPair) : Bool :=
  d.NODES.all fun w => (Pre d w).all fun v =>
    paperDecorationAtB d labels v (paperFlow w v)




def localGuardFaithfulB (c : Certificate) : Bool :=
  c.graph.NODES.all fun v =>
    (eligibleAncestorsAtRoot c.graph v).isEmpty ||
      c.graph.EDGES.all fun edge =>
        !(edge.END == v) || (StoredRoutes c edge.START).all fun route =>
          !(consume edge route.2).isSome || depSimilarB route.1 edge.DEPENDENCY

def LocalGuardCompatibility (c : Certificate) : Prop :=
  ∀ v ∈ c.graph.NODES, eligibleAncestorsAtRoot c.graph v ≠ [] →
    ∀ edge ∈ c.graph.EDGES, edge.END = v →
      ∀ route ∈ StoredRoutes c edge.START, ∀ tail,
        consume edge route.2 = some tail → depSimilarB route.1 edge.DEPENDENCY = true

end Decoration
