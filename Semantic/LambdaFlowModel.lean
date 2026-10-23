import Semantic.DecorationGuardDefinitions

open Semantic FlowSpec

namespace LambdaFlow

abbrev Guard := Dep → Deduction → Bool

def allowAll : Guard := fun _ _ => true
def historicalGuard : Guard := fun dep edge => depSimilarB dep edge.DEPENDENCY

def mixedWith (d : Graph) (guard : Guard) (v : Vertex)
    (prior : Vertex → List FlowPair) : List FlowPair :=
  match v.FORMULA with
  | .implication alpha beta =>
      (predsOf d v).flatMap fun u =>
        if !elimPremiseAtB d v u && u.FORMULA == beta then
          (prior u).flatMap fun route =>
            (edgesBetween d u v).filterMap fun edge =>
              if guard route.1 edge then
                (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)
              else none
        else []
  | _ => []

def elimReseedWith (d : Graph) (guard : Guard) (v : Vertex)
    (prior : Vertex → List FlowPair) (ancestors : List Ancestral) : List FlowPair :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    (prior minor).flatMap fun left => (prior major).flatMap fun right =>
      (edgesBetween d minor v).flatMap fun eMinor =>
        (edgesBetween d major v).flatMap fun eMajor =>
          if reseedPremiseAgreesB left eMinor && reseedPremiseAgreesB right eMajor &&
              guard left.1 eMinor && guard right.1 eMajor then
            ancestors.map fun ancestor => (depUnion left.1 right.1, ancestor.COLOURS)
          else []

def introReseedWith (d : Graph) (guard : Guard) (v : Vertex)
    (prior : Vertex → List FlowPair) (ancestors : List Ancestral) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [u], .implication alpha beta =>
      if u.FORMULA == beta then
        (prior u).flatMap fun route => (edgesBetween d u v).flatMap fun edge =>
          if reseedZeroPremiseB route edge && guard route.1 edge then
            ancestors.map fun ancestor => (depRemove route.1 alpha, ancestor.COLOURS)
          else []
      else []
  | _, _ => []

def reseedWith (d : Graph) (guard : Guard) (v : Vertex)
    (prior : Vertex → List FlowPair) : List FlowPair :=
  let ancestors := eligibleAncestorsAtRoot d v
  if ancestors.isEmpty then [] else
    (elimReseedWith d guard v prior ancestors ++
      introReseedWith d guard v prior ancestors).eraseDups



def stepWith (d : Graph) (ordinary reseed : Guard)
    (prior : Vertex → List FlowPair) (v : Vertex) : List FlowPair :=
  let preds := predsOf d v
  if v.HYPOTHESIS || preds.isEmpty then
    let ancestors := ancestorsInto d v
    if ancestors.isEmpty then [(normalizeDep [v.FORMULA], [])]
    else ancestors.map fun a => (normalizeDep [v.FORMULA], a.COLOURS)
  else
    let elimination := (elimPairsAt d v).flatMap fun (major, minor) =>
      (prior minor).flatMap fun left => (prior major).flatMap fun right =>
        (edgesBetween d minor v).flatMap fun eMinor =>
          (edgesBetween d major v).filterMap fun eMajor =>
            if ordinary left.1 eMinor && ordinary right.1 eMajor then
              match consume eMinor left.2, consume eMajor right.2 with
              | some p, some q => if p = q then some (depUnion left.1 right.1, p) else none
              | _, _ => none
            else none
    let introduction := match preds, v.FORMULA with
      | [u], .implication alpha beta =>
          if u.FORMULA = beta then
            (prior u).flatMap fun route => (edgesBetween d u v).filterMap fun edge =>
              if ordinary route.1 edge then
                (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)
              else none
          else []
      | _, _ => []
    let core := if (elimPairsAt d v).isEmpty then introduction
      else elimination ++ mixedWith d ordinary v prior
    (core ++ reseedWith d reseed v prior).eraseDups

theorem historical_step_eq (d : Graph) (prior : Vertex → List FlowPair) (v : Vertex) :
    stepWith d allowAll historicalGuard prior v = flowStepFrom d prior v := by
  rfl

def evaluateAt (d : Graph) (ordinary reseed : Guard) : Nat → Vertex → List FlowPair
  | 0, _ => []
  | fuel + 1, v => stepWith d ordinary reseed (evaluateAt d ordinary reseed fuel) v



abbrev guardFreeFlowAt (d : Graph) := evaluateAt d allowAll allowAll

def fromTable (d : Graph) (ordinary reseed : Guard) (fuel : Nat)
    (table : FlowTable) (v : Vertex) : List FlowPair :=
  match flowTableLookup? table v with
  | some routes => routes
  | none => evaluateAt d ordinary reseed fuel v

def tableStep (d : Graph) (ordinary reseed : Guard) (fuel : Nat)
    (table : FlowTable) : FlowTable :=
  d.NODES.map fun v => (v, stepWith d ordinary reseed
    (fromTable d ordinary reseed fuel table) v)

def iterate (d : Graph) (ordinary reseed : Guard) : Nat → Nat → FlowTable → FlowTable
  | 0, _, table => table
  | remaining + 1, fuel, table =>
      let next := tableStep d ordinary reseed fuel table
      if flowTableClosedB d && next == table then next
      else iterate d ordinary reseed remaining (fuel + 1) next

def tableAt (d : Graph) (ordinary reseed : Guard) (fuel : Nat) : FlowTable :=
  iterate d ordinary reseed fuel 0 (flowTableInitial d)

def guardFreeTable (d : Graph) : FlowTable := tableAt d allowAll allowAll (stdFuel d)

def rootsFromTableB (d : Graph) (table : FlowTable) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let routes := fromTable d allowAll allowAll (stdFuel d) table v
     !routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty)

def guardFreeRootsB (d : Graph) : Bool := rootsFromTableB d (guardFreeTable d)

inductive Label where
  | Bits (dependency : Dep)
  | Lambda
  deriving Repr, DecidableEq

structure LGraph where
  graph : Graph
  label : Deduction → Label

def labelAgreesB (label : Label) (dependency : Dep) : Bool :=
  match label with
  | .Bits stored => depSetEq dependency stored
  | .Lambda => true

def labelGuard (lg : LGraph) : Guard := fun dep edge => labelAgreesB (lg.label edge) dep

abbrev lambdaFlowAt (lg : LGraph) := evaluateAt lg.graph (labelGuard lg) (labelGuard lg)

def lambdaTable (lg : LGraph) : FlowTable :=
  tableAt lg.graph (labelGuard lg) (labelGuard lg) (stdFuel lg.graph)

def canonicalDecorationFrom (d : Graph) (table : FlowTable) : LGraph :=
  ⟨d, fun edge =>
    let group := ((fromTable d allowAll allowAll (stdFuel d) table edge.START).eraseDups).filter
      fun route => headColour route.2 == edge.COLOUR
    match group with
    | [route] => .Bits route.1
    | _ :: _ :: _ => .Lambda
    | [] => .Bits edge.DEPENDENCY⟩

def canonicalDecoration (d : Graph) : LGraph := canonicalDecorationFrom d (guardFreeTable d)

def toPaperLabel : Label → Decoration.PaperLabel
  | .Bits dependency => .bits dependency
  | .Lambda => .lambda




def decoratedFromB (lg : LGraph) (table : FlowTable) : Bool :=
  lg.graph.NODES.all fun v => (get_rule.outgoing v lg.graph).isEmpty ||
    Decoration.paperDecorationAtB lg.graph (fun e => toPaperLabel (lg.label e)) v
      (fromTable lg.graph allowAll allowAll (stdFuel lg.graph) table v)

def Def23DecoratedB (lg : LGraph) : Bool := decoratedFromB lg (guardFreeTable lg.graph)

#print axioms historical_step_eq

end LambdaFlow
