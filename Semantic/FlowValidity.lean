import Semantic.FlowModel
import Semantic.Examples



open Semantic

namespace FlowSpec


def stdFuel (d : Graph) : Nat := d.NODES.length + 1


def headColour (p : ColourPath) : Nat := p.headD 0

def edgeIsLambdaLabelled (_d : Graph) (_e : Deduction) : Bool := false


def singletonBulletB (d : Graph) (v : Vertex) (b : Dep) (p : ColourPath) : Bool :=
  match get_rule.outgoing v d with
  | [e] => e.COLOUR == headColour p && depSetEq b e.DEPENDENCY
  | _   => false


def phiBulletB (d : Graph) (v : Vertex) (F : List FlowPair) : Bool :=
  let outs := get_rule.outgoing v d
  let colours := (F.map (fun bp => headColour bp.2)).eraseDups
  colours.all (fun i =>
    match F.filter (fun bp => headColour bp.2 == i),
          outs.filter (fun e => e.COLOUR == i) with
    | [(b, _)],     [e] => depSetEq b e.DEPENDENCY
    | _ :: _ :: _,  [e] => edgeIsLambdaLabelled d e
    | _,            _   => false) &&
  outs.all (fun e => colours.contains e.COLOUR)


def flowRuleCorrectAtB (d : Graph) (v : Vertex) : Bool :=
  if (get_rule.incoming v d).isEmpty then true
  else if (get_rule.outgoing v d).isEmpty then true
  else
    match flowAt d (stdFuel d) v with
    | []            => false
    | [(b, p)]      => singletonBulletB d v b p
    | bp :: bp' :: F => phiBulletB d v (bp :: bp' :: F)

abbrev FlowRuleCorrect (d : Graph) (v : Vertex) : Prop :=
  flowRuleCorrectAtB d v = true


def rootsDischargedB (d : Graph) : Bool :=
  d.NODES.all fun r =>
    !(get_rule.outgoing r d).isEmpty ||
      (let pairs := flowAt d (stdFuel d) r
       !pairs.isEmpty && pairs.all fun bp => bp.1.isEmpty && bp.2.isEmpty)

def flowCorrectB (d : Graph) : Bool :=
  d.NODES.all (flowRuleCorrectAtB d) && rootsDischargedB d

abbrev FlowCorrect (d : Graph) : Prop :=
  flowCorrectB d = true




end FlowSpec
