import Semantic.FlowModel
import Semantic.Examples

/-! Flow-based correctness conditions corresponding to Definition 23. -/

open Semantic

namespace FlowSpec

/-- Standard fuel for the fuelled Flow machinery. -/
def stdFuel (d : Graph) : Nat := d.NODES.length + 1

/-- head(p) of a residual colour path, with the documented ε ≅ R0 default: -/
def headColour (p : ColourPath) : Nat := p.headD 0

def edgeIsLambdaLabelled (_d : Graph) (_e : Deduction) : Bool := false

/-- Def-23 singleton bullet at v for flow {(b, p)}: -/
def singletonBulletB (d : Graph) (v : Vertex) (b : Dep) (p : ColourPath) : Bool :=
  match get_rule.outgoing v d with
  | [e] => e.COLOUR == headColour p && depSetEq b e.DEPENDENCY
  | _   => false

/-- Def-23 Φᵢ bullet at v for a non-empty, non-singleton flow F: -/
def phiBulletB (d : Graph) (v : Vertex) (F : List FlowPair) : Bool :=
  let outs := get_rule.outgoing v d
  let colours := (F.map (fun bp => headColour bp.2)).eraseDups
  colours.all (fun i =>
    match F.filter (fun bp => headColour bp.2 == i),
          outs.filter (fun e => e.COLOUR == i) with
    | [(b, _)],     [e] => depSetEq b e.DEPENDENCY     -- Φᵢ singleton ⇒ L = b
    | _ :: _ :: _,  [e] => edgeIsLambdaLabelled d e    -- Φᵢ ≥ 2 ⇒ L = λ
    | _,            _   => false) &&                   -- no unique colour-i edge
  outs.all (fun e => colours.contains e.COLOUR)

/-- Literal Def-23 CorrectRuleApp at v. -/
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

/-- Root discharge via Flow (Boolean). -/
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
