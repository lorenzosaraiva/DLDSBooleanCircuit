import Semantic.MultiTokenModel

open scoped Classical

/-! Admissibility for multi-token paths, including matching modulo permutations
of co-located route tokens. -/

open Semantic

namespace Semantic

open FlowSpec
open ExFan2 ExFan3



/-- Slot check: -/
def slotCheckB (d : Graph) (j : Nat) (steps : List (Nat × Nat)) : Bool :=
  match (buildFormulas d)[((routesOf d).getD j (0, 0)).1]? with
  | none => steps.all (fun s => s == (0, 0))
  | some φ =>
      match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
      | none => steps.all (fun s => s == (0, 0))
      | some v =>
          match (flowAt d (stdFuel d) v)[((routesOf d).getD j (0, 0)).2]? with
          | none => steps.all (fun s => s == (0, 0))
          | some bp =>
              admHypColumnCB d (buildFormulas d) ((routesOf d).getD j (0, 0)).1
                v.LEVEL φ bp.2
                ((d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) steps

def slotColumn (d : Graph) (j : Nat) : Nat := ((routesOf d).getD j (0, 0)).1

/-- Backtracking per-column matcher: -/
def multiMatch? (check : Nat → List (Nat × Nat) → Bool) (colf : Nat → Nat)
    (i : Nat) : List (List (Nat × Nat)) → List Nat → Option (List Nat)
  | [], _ => some []
  | steps :: rest, avail =>
      avail.findSome? fun j =>
        if colf j == colf i && check j steps then
          (multiMatch? check colf (i + 1) rest (avail.erase j)).map (j :: ·)
        else none

/-- admissibleMultiPathB ; -/
def admissibleMultiPathB (d : Graph) (P : MultiPathInput) : Bool :=
  (P.length == routeCount d) &&
  (multiMatch? (slotCheckB d) (slotColumn d) 0 P
      (List.range (routeCount d))).isSome &&
  goalReceivedB d P

/-- AdmissibleMultiPath (Prop mirror, repo reflection pattern). -/
def AdmissibleMultiPath (d : Graph) (P : MultiPathInput) : Prop :=
  admissibleMultiPathB d P = true

def InvalidMultiPath (d : Graph) (P : MultiPathInput) : Prop :=
  ¬ AdmissibleMultiPath d P

/-- The M2c side condition (decidable, eval-gated below): -/
def canonicalSlotOKB (d : Graph) : Bool :=
  (List.range (routeCount d)).all fun j =>
    slotCheckB d j ((multiPathsFromFlow d).getD j [])



def swapSlots (P : MultiPathInput) (i j : Nat) : MultiPathInput :=
  setCol (setCol P i (P.getD j [])) j (P.getD i [])

def exFan2Swapped : MultiPathInput :=
  let i0 := firstRouteOfColumn exFan2C (colOf exFan2C ExFan2.fA)
  swapSlots (multiPathsFromFlow exFan2C) i0 (i0 + 1)

def exFan3Swapped : MultiPathInput :=
  let i0 := firstRouteOfColumn exFan3C (colOf exFan3C ExFan3.gA1)
  swapSlots (multiPathsFromFlow exFan3C) i0 (i0 + 1)

def exFan3TruncRoute : MultiPathInput :=
  setCol (multiPathsFromFlow exFan3C)
    (firstRouteOfColumn exFan3C (colOf exFan3C ExFan3.gA1)) []

def exFan3Duplicated : MultiPathInput :=
  let i0 := firstRouteOfColumn exFan3C (colOf exFan3C ExFan3.gA1)
  setCol (multiPathsFromFlow exFan3C) i0
    ((multiPathsFromFlow exFan3C).getD (i0 + 1) [])



end Semantic
