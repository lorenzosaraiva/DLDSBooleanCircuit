import Semantic.UniversalBridge
import Semantic.FlowEdgeDepChar
import Semantic.CompressedBridge

/-! Flow-coloured admissibility and canonical routing for compressed DLDS graphs.
Carriers are selected by level and formula and follow residual colour paths. -/

open Semantic

namespace Semantic

open FlowSpec


/-- Which premise slot of w's rule the premise formula φ fills, collapsed- aware: -/
def slotForEdgeC (d : Graph) (φ : Formula) (w : Vertex) : Nat :=
  match elimPairsAt d w with
  | [(_M, S)] => if φ = S.FORMULA then 1 else 0
  | _ => slotForEdge φ w d

/-- Rule-list index of the rule applied at w, collapsed-aware: -/
def ruleIndexForNodeC? (d : Graph) (formulas : List Formula) (w : Vertex) :
    Option Nat :=
  match elimPairsAt d w with
  | [(M, _S)] =>
      match elimRulePosition? formulas w.FORMULA M.FORMULA with
      | some pos => some (introRuleCount w.FORMULA + pos)
      | none => none
  | _ => ruleIndexForNode? d formulas w

/-- Compressed input label for the edge from source formula φ into w. -/
def inputLabelForEdgeC (d : Graph) (formulas : List Formula) (φ : Formula)
    (w : Vertex) : Nat :=
  let incoming := buildIncomingMapForFormula formulas w.FORMULA
  match ruleIndexForNodeC? d formulas w with
  | some ruleIdx => inputLabelForRuleSlot incoming ruleIdx (slotForEdgeC d φ w)
  | none => 0

/-- Whether a carrier arriving at `w` supplies its minor premise. -/
def isMinorArrivalC (d : Graph) (φ : Formula) (w : Vertex) : Bool :=
  match elimPairsAt d w with
  | [(_M, S)] => decide (φ = S.FORMULA)
  | _ => false



/-- Coloured carrier chain from state (lvl, φ, p): -/
def admChainCB (d : Graph) (formulas : List Formula) :
    Nat → Formula → ColourPath → List (Nat × Nat) → Bool
  | lvl, φ, _, [] =>
      match nodeAtLevelFormula? d lvl φ with
      | none => true
      | some v => (get_rule.outgoing v d).isEmpty
  | lvl, φ, p, (t, l) :: rest =>
      match nodeAtLevelFormula? d lvl φ with
      | none => t == 0 && l == 0 && rest.all (fun s => s == (0, 0))
      | some v =>
          match get_rule.outgoing v d with
          | [] => t == 0 && l == 0 && rest.all (fun s => s == (0, 0))
          | _ :: _ =>
              match edgeOfColour d v (headColour p) with
              | none => false
              | some e =>
                  (t == formulas.idxOf e.END.FORMULA + 1) &&
                  (l == inputLabelForEdgeC d formulas φ e.END) &&
                  (if isMinorArrivalC d φ e.END && p.tail.isEmpty then
                     rest.all (fun s => s == (0, 0))
                   else admChainCB d formulas (lvl - 1) e.END.FORMULA p.tail rest)

/-- Clause (3)+(2): -/
def admHypColumnCB (d : Graph) (formulas : List Formula) (col lvl : Nat)
    (φ : Formula) (p : ColourPath) : Nat → List (Nat × Nat) → Bool
  | 0, steps => admChainCB d formulas lvl φ p steps
  | _ + 1, [] => false
  | k + 1, (t, l) :: rest =>
      (t == col + 1 && l == 0) && admHypColumnCB d formulas col lvl φ p k rest

/-- Clause (1)+(3), compressed: -/
def admColumnCB (d : Graph) (formulas : List Formula) (col : Nat)
    (steps : List (Nat × Nat)) : Bool :=
  match formulas[col]? with
  | none => steps.all (fun s => s == (0, 0))
  | some φ =>
      match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
      | none => steps.all (fun s => s == (0, 0))
      | some v =>
          let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
          (flowAt d (stdFuel d) v).any fun bp =>
            admHypColumnCB d formulas col v.LEVEL φ bp.2 (maxLvl - v.LEVEL) steps

/-- admissibleCompressedPathB ; -/
def admissibleCompressedPathB (d : Graph) (P : PathInput) : Bool :=
  let formulas := buildFormulas d
  (P.length == formulas.length) &&
  ((List.range formulas.length).all (fun col =>
    admColumnCB d formulas col (P.getD col []))) &&
  goalReceivedB d P

def routeFromFlowC (d : Graph) (formulas : List Formula) :
    Nat → Nat → Formula → ColourPath → List (Nat × Nat)
  | 0, _, _, _ => []
  | fuel + 1, lvl, φ, p =>
      match nodeAtLevelFormula? d lvl φ with
      | none => (0, 0) :: routeFromFlowC d formulas fuel lvl φ p
      | some v =>
          match get_rule.outgoing v d with
          | [] => (0, 0) :: routeFromFlowC d formulas fuel lvl φ p
          | _ :: _ =>
              match edgeOfColour d v (headColour p) with
              | none => (0, 0) :: routeFromFlowC d formulas fuel lvl φ p
              | some e =>
                  (formulas.idxOf e.END.FORMULA + 1,
                   inputLabelForEdgeC d formulas φ e.END) ::
                  (if isMinorArrivalC d φ e.END && p.tail.isEmpty then
                     List.replicate fuel (0, 0)
                   else routeFromFlowC d formulas fuel (lvl - 1) e.END.FORMULA p.tail)

def compressedPathsFromFlowWith (d : Graph) (pick : Nat → Nat) : PathInput :=
  let formulas := buildFormulas d
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  formulas.zipIdx.map fun (φ, col) =>
    match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
    | some v =>
        match (flowAt d (stdFuel d) v)[pick col]? with
        | some bp =>
            let delay := maxLvl - v.LEVEL
            List.replicate delay (col + 1, 0) ++
              routeFromFlowC d formulas (numSteps - delay) v.LEVEL φ bp.2
        | none => List.replicate numSteps (0, 0)
    | none => List.replicate numSteps (0, 0)

def compressedPathsFromFlow (d : Graph) : PathInput :=
  compressedPathsFromFlowWith d (fun _ => 0)



/-- Goal-vector-all-false on an ARBITRARY path input (the dischargedB check generalized from the canonical. -/
def dischargedOnB (d : Graph) (P : PathInput) : Bool :=
  match (get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
          P).1[goalColumn d]? with
  | none => true
  | some v => v.toList.all (fun b => !b)

/--  Replace one step of one column (garbage-path builder).  -/
def setStep (P : PathInput) (c s : Nat) (st : Nat × Nat) : PathInput :=
  P.zipIdx.map fun (colpath, ci) =>
    if ci = c then
      colpath.zipIdx.map (fun (x, si) => if si = s then st else x)
    else colpath

/--  Replace one whole column (garbage-path builder).  -/
def setCol (P : PathInput) (c : Nat) (newcol : List (Nat × Nat)) : PathInput :=
  P.zipIdx.map fun (colpath, ci) => if ci = c then newcol else colpath

def colOf (d : Graph) (φ : Formula) : Nat := (buildFormulas d).idxOf φ


namespace ExFan2

def fA   : Formula := #"A"
def fB   : Formula := #"B"
def fC   : Formula := #"C"
def fX   : Formula := #"X"
def fAB  : Formula := fA >> fB
def fBC  : Formula := fB >> fC
def fCX  : Formula := fC >> fX
def fBCX : Formula := fB >> fCX
def fI1  : Formula := fBCX >> fX
def fI2  : Formula := fBC >> fI1
def fI3  : Formula := fAB >> fI2
def fI4  : Formula := fA >> fI3

def a1  : Vertex := Vertex.node 1  7 fA   true  false []
def a2  : Vertex := Vertex.node 2  7 fA   true  false []
def ab1 : Vertex := Vertex.node 3  7 fAB  true  false []
def ab2 : Vertex := Vertex.node 4  7 fAB  true  false []
def b1  : Vertex := Vertex.node 5  6 fB   false false []
def b2  : Vertex := Vertex.node 6  6 fB   false false []
def hbc : Vertex := Vertex.node 7  6 fBC  true  false []
def hbcx: Vertex := Vertex.node 8  6 fBCX true  false []
def c   : Vertex := Vertex.node 9  5 fC   false false []
def cx  : Vertex := Vertex.node 10 5 fCX  false false []
def x   : Vertex := Vertex.node 11 4 fX   false false []
def i1  : Vertex := Vertex.node 12 3 fI1  false false []
def i2  : Vertex := Vertex.node 13 2 fI2  false false []
def i3  : Vertex := Vertex.node 14 1 fI3  false false []
def i4  : Vertex := Vertex.node 15 0 fI4  false false []

def exFan2 : Graph :=
  Graph.dlds
    [a1, a2, ab1, ab2, b1, b2, hbc, hbcx, c, cx, x, i1, i2, i3, i4]
    [Deduction.edge a1  b1 0 [fA],
     Deduction.edge ab1 b1 0 [fAB],
     Deduction.edge a2  b2 0 [fA],
     Deduction.edge ab2 b2 0 [fAB],
     Deduction.edge b1  c  0 [fA, fAB],
     Deduction.edge hbc c  0 [fBC],
     Deduction.edge b2  cx 0 [fA, fAB],
     Deduction.edge hbcx cx 0 [fBCX],
     Deduction.edge c   x  0 [fA, fAB, fBC],
     Deduction.edge cx  x  0 [fA, fAB, fBCX],
     Deduction.edge x   i1 0 [fA, fAB, fBC, fBCX],
     Deduction.edge i1  i2 0 [fA, fAB, fBC],
     Deduction.edge i2  i3 0 [fA, fAB],
     Deduction.edge i3  i4 0 [fA]]
    []

/--
 The compressed fixture: `A`, `A⊃B`, `B` collapsed (each pair → one node
    with coloured fan / ancestor addresses).
-/
def exFan2C : Graph := compress_nodes_graph exFan2

end ExFan2

open ExFan2





end Semantic
