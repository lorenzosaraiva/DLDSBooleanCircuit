import Semantic.CompleteProfileKernel
import Semantic.CompleteProfilePositions



open Semantic FlowSpec
namespace MultiIntro

inductive Variant where
  | strict
  | general
  deriving Repr, DecidableEq

def multiIntroShapeB (d : Graph) (v : Vertex) : Bool :=
  !v.HYPOTHESIS && match v.FORMULA with
    | .implication _ beta =>
        (elimPairsAt d v).isEmpty &&
        decide (2 ≤ (get_rule.incoming v d).length) &&
        (get_rule.incoming v d).all fun e => e.START.FORMULA == beta
    | _ => false

def MultiIntroShape (d : Graph) (v : Vertex) : Prop := multiIntroShapeB d v = true

def miDomainB (d : Graph) : Bool :=
  (d.EDGES.all fun e => e.START.LEVEL == e.END.LEVEL + 1) &&
  (d.EDGES.all fun e => d.NODES.contains e.START && d.NODES.contains e.END) &&
  (d.NODES.all fun v =>
    if !(get_rule.incoming v d).isEmpty && !(get_rule.outgoing v d).isEmpty then
      nodeElimShapeWideB d v || nodeIntroShapeB d v || multiIntroShapeB d v
    else true)

def MIDomain (d : Graph) : Prop := miDomainB d = true

def ordinaryEdge (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (e : Deduction) : List FlowPair :=
  match v.FORMULA with
  | .implication alpha beta =>
      if e.START.FORMULA == beta then
        (prior e.START).filterMap fun route =>
          if SingleIntroductionReference.guard d route.1 e then
            (consume e route.2).map fun tail => (depRemove route.1 alpha, tail)
          else none
      else []
  | _ => []

def reseedEdge (variant : Variant) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (e : Deduction) : List FlowPair :=
  match v.FORMULA with
  | .implication alpha beta =>
      if e.START.FORMULA == beta then
        (prior e.START).flatMap fun route =>
          if SingleIntroductionReference.guard d route.1 e &&
              (match variant with
               | .strict => reseedZeroPremiseB route e
               | .general => reseedPremiseAgreesB route e) then
            (eligibleAncestorsAtRoot d v).map fun ancestor =>
              (depRemove route.1 alpha, ancestor.COLOURS)
          else []
      else []
  | _ => []

def step (variant : Variant) (d : Graph) (prior : Vertex → List FlowPair)
    (v : Vertex) : List FlowPair :=
  if multiIntroShapeB d v then
    ((get_rule.incoming v d).flatMap fun e =>
      ordinaryEdge d v prior e ++ reseedEdge variant d v prior e).eraseDups
  else CompleteProfileKernel.tagged.step d prior v

def evaluateAt (variant : Variant) (d : Graph) : Nat → Vertex → List FlowPair
  | 0, _ => []
  | fuel+1, v => step variant d (evaluateAt variant d fuel) v

def rootsB (variant : Variant) (d : Graph) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let routes := evaluateAt variant d (stdFuel d) v
     !routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty)

def ValidMI (variant : Variant) (d : Graph) : Prop :=
  MIDomain d ∧ LeveledColored d ∧ rootsB variant d = true

abbrev ValidMIstrict := ValidMI .strict
abbrev ValidMIgeneral := ValidMI .general

end MultiIntro
