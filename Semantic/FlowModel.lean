import Semantic.DLDS

/-! Executable Flow specification from Definition 22. Residual paths guide coloured
edges; an empty residual follows the default colour 0. Interior reseeding is
not represented and is excluded later by `ReseedFree`. -/

open Semantic

namespace FlowSpec

abbrev Dep        := List Formula
abbrev ColourPath := List Nat
/--  One ancestor-guided route: (dependency set, residual colour path).  -/
abbrev FlowPair   := Dep × ColourPath



def depUnion (a b : Dep) : Dep := (a ++ b).eraseDups

def depRemove (a : Dep) (φ : Formula) : Dep :=
  (a.filter (fun ψ => ψ ≠ φ)).eraseDups

def depSetEq (a b : Dep) : Bool :=
  a.all (fun x => b.any (fun y => y = x)) &&
  b.all (fun x => a.any (fun y => y = x))



/--  Deductive predecessors of `v` across ONE edge (any colour), deduplicated.  -/
def predsOf (d : Graph) (v : Vertex) : List Vertex :=
  ((d.EDGES.filter (fun e => e.END = v)).map (·.START)).eraseDups

def edgesBetween (d : Graph) (u v : Vertex) : List Deduction :=
  d.EDGES.filter (fun e => e.START = u ∧ e.END = v)

/--  The unique outgoing deduction edge of `v` with colour `c`, if any.  -/
def edgeOfColour (d : Graph) (v : Vertex) (c : Nat) : Option Deduction :=
  (d.EDGES.filter (fun e => e.START = v ∧ e.COLOUR = c)).head?

def ancestorsInto (d : Graph) (v : Vertex) : List Ancestral :=
  d.PATHS.filter (fun a => a.END = v)



def Pre (d : Graph) (w : Vertex) : List Vertex :=
  go (d.NODES.length + 1) [w] []
where
  go : Nat → List Vertex → List Vertex → List Vertex
  | 0,        _,        acc => acc
  | fuel + 1, frontier, acc =>
      let fresh := ((frontier.flatMap (predsOf d)).eraseDups).filter
        (fun u => !(acc.any (fun a => a = u)))
      if fresh.isEmpty then acc
      else go fuel fresh (acc ++ fresh)



/--
 Crossing edge `e` consumes the residual head: colour `o` consumes head `o`;
    a colour-0 edge also accepts the empty residual (default route).
-/
def consume (e : Deduction) (p : ColourPath) : Option ColourPath :=
  match p with
  | []      => if e.COLOUR = 0 then some [] else none
  | o :: p' => if e.COLOUR = o then some p' else none

def leadsTo (d : Graph) : Nat → Vertex → ColourPath → Vertex → Bool
  | 0,        _, _, _ => false
  | fuel + 1, v, p, w =>
      if v = w then p.isEmpty
      else
        match edgeOfColour d v (p.headD 0) with
        | none   => false
        | some e => leadsTo d fuel e.END p.tail w


/--  (major, minor) vertex pairs at `v`: minor `ψ`, major `ψ ⊃ l(v)`.  -/
def elimPairsAt (d : Graph) (v : Vertex) : List (Vertex × Vertex) :=
  let ps := predsOf d v
  ps.flatMap (fun major =>
    ps.filterMap (fun minor =>
      if major.FORMULA = Formula.implication minor.FORMULA v.FORMULA
      then some (major, minor) else none))

def reseedPairs (_d : Graph) (_v : Vertex) (_ruleResults : List FlowPair) :
    List FlowPair := []



/--
 All flow pairs at `v` (routes from top-nodes down to `v`, with the residual
    telling how each route continues BELOW `v`). Fuelled on levels.
-/
def flowAt (d : Graph) : Nat → Vertex → List FlowPair
  | 0, _ => []
  | fuel + 1, v =>
      let preds := predsOf d v
      if v.HYPOTHESIS || preds.isEmpty then
        let anc := ancestorsInto d v
        if anc.isEmpty then [([v.FORMULA], [])]
        else anc.map (fun a => ([v.FORMULA], a.COLOURS))
      else
        let elim : List FlowPair :=
          (elimPairsAt d v).flatMap (fun (major, minor) =>
            (flowAt d fuel minor).flatMap (fun (b₁, p₁) =>
              (flowAt d fuel major).flatMap (fun (b₂, p₂) =>
                (edgesBetween d minor v).flatMap (fun e₁ =>
                  (edgesBetween d major v).filterMap (fun e₂ =>
                    match consume e₁ p₁, consume e₂ p₂ with
                    | some q₁, some q₂ =>
                        if q₁ = q₂ then some (depUnion b₁ b₂, q₁) else none
                    | _, _ => none)))))
        let intro : List FlowPair :=
          match preds, v.FORMULA with
          | [u], Formula.implication α β =>
              if u.FORMULA = β then
                (flowAt d fuel u).flatMap (fun (b, p) =>
                  (edgesBetween d u v).filterMap (fun e =>
                    (consume e p).map (fun q => (depRemove b α, q))))
              else []
          | _, _ => []
        let core := if elim.isEmpty then intro else elim
        (core ++ reseedPairs d v core).eraseDups

/--
 `Flow(D,w)(v)` ; paper Def. 3 / source Def. 22. The generative pairs at
    `v`, restricted to routes that reach `w` exactly (residual = path from `v`
    to `w`, consumed colour-by-colour, ε defaulting to colour 0).
-/
def Flow (d : Graph) (w v : Vertex) : List FlowPair :=
  (flowAt d (d.NODES.length + 1) v).filter
    (fun bp => leadsTo d (d.NODES.length + 1) v bp.2 w)


def edgeFlowDeps (d : Graph) (e : Deduction) : List Dep :=
  ((d.NODES.flatMap (fun w =>
      (Flow d w e.START).filterMap (fun bp =>
        (consume e bp.2).map (fun _ => bp.1)))).eraseDups)

end FlowSpec
