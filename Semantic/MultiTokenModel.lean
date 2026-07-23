import Semantic.CompressedRouting

open scoped Classical

/-! Multi-token evaluation for compressed DLDS graphs. One token is seeded per
Flow route so a collapsed node can serve an out-fan larger than its rule arity. -/

open Semantic

namespace Semantic

open FlowSpec


namespace ExFan3

def gA1 : Formula := #"A1"
def gA2 : Formula := #"A2"
def gA3 : Formula := #"A3"
def gA4 : Formula := #"A4"
def gA5 : Formula := #"A5"
def g12  : Formula := gA1 >> gA2
def g123 : Formula := gA1 >> (gA2 >> gA3)
def g23  : Formula := gA2 >> gA3
def g234 : Formula := gA2 >> (gA3 >> gA4)
def g34  : Formula := gA3 >> gA4
def g345 : Formula := gA3 >> (gA4 >> gA5)
def g45  : Formula := gA4 >> gA5
def gJ1  : Formula := g345 >> gA5
def gJ2  : Formula := g234 >> gJ1
def gJ3  : Formula := g123 >> gJ2
def gJ4  : Formula := g12  >> gJ3
def gJ5  : Formula := gA1  >> gJ4

-- level 9: hypotheses (d0's level 5, shifted so the ⊃I chain reaches 0)
def m0 : Vertex := Vertex.node 100 9 gA1  true false []
def m1 : Vertex := Vertex.node 101 9 g12  true false []
def m2 : Vertex := Vertex.node 102 9 gA1  true false []
def m3 : Vertex := Vertex.node 103 9 g123 true false []
def m4 : Vertex := Vertex.node 104 9 gA1  true false []
def m5 : Vertex := Vertex.node 105 9 g12  true false []
def m6 : Vertex := Vertex.node 106 9 gA1  true false []
def m7 : Vertex := Vertex.node 107 9 g12  true false []
def m8 : Vertex := Vertex.node 108 9 gA1  true false []
def m9 : Vertex := Vertex.node 109 9 g123 true false []
-- level 8: THREE A2 copies, TWO A2⊃A3 copies, one hypothesis
def m10 : Vertex := Vertex.node 110 8 gA2  false false []
def m11 : Vertex := Vertex.node 111 8 g23  false false []
def m12 : Vertex := Vertex.node 112 8 gA2  false false []
def m13 : Vertex := Vertex.node 113 8 g234 true  false []
def m14 : Vertex := Vertex.node 114 8 gA2  false false []
def m15 : Vertex := Vertex.node 115 8 g23  false false []
-- level 7
def m16 : Vertex := Vertex.node 116 7 gA3  false false []
def m17 : Vertex := Vertex.node 117 7 g34  false false []
def m18 : Vertex := Vertex.node 118 7 gA3  false false []
def m19 : Vertex := Vertex.node 119 7 g345 true  false []
-- level 6
def m20 : Vertex := Vertex.node 120 6 gA4  false false []
def m21 : Vertex := Vertex.node 121 6 g45  false false []
-- level 5
def m22 : Vertex := Vertex.node 122 5 gA5  false false []
-- levels 4..0: the CLOSING ⊃I chain (replaces d0's degenerate root)
def m23 : Vertex := Vertex.node 123 4 gJ1 false false []
def m24 : Vertex := Vertex.node 124 3 gJ2 false false []
def m25 : Vertex := Vertex.node 125 2 gJ3 false false []
def m26 : Vertex := Vertex.node 126 1 gJ4 false false []
def m27 : Vertex := Vertex.node 127 0 gJ5 false false []

def exFan3 : Graph :=
  Graph.dlds
    [m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,
     m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27]
    [Deduction.edge m0  m10 0 [gA1],
     Deduction.edge m1  m10 0 [g12],
     Deduction.edge m2  m11 0 [gA1],
     Deduction.edge m3  m11 0 [g123],
     Deduction.edge m4  m12 0 [gA1],
     Deduction.edge m5  m12 0 [g12],
     Deduction.edge m6  m14 0 [gA1],
     Deduction.edge m7  m14 0 [g12],
     Deduction.edge m8  m15 0 [gA1],
     Deduction.edge m9  m15 0 [g123],
     Deduction.edge m10 m16 0 [gA1, g12],
     Deduction.edge m11 m16 0 [gA1, g123],
     Deduction.edge m12 m17 0 [gA1, g12],
     Deduction.edge m13 m17 0 [g234],
     Deduction.edge m14 m18 0 [gA1, g12],
     Deduction.edge m15 m18 0 [gA1, g123],
     Deduction.edge m16 m20 0 [gA1, g12, g123],
     Deduction.edge m17 m20 0 [gA1, g12, g234],
     Deduction.edge m18 m21 0 [gA1, g12, g123],
     Deduction.edge m19 m21 0 [g345],
     Deduction.edge m20 m22 0 [gA1, g12, g123, g234],
     Deduction.edge m21 m22 0 [gA1, g12, g123, g345],
     Deduction.edge m22 m23 0 [gA1, g12, g123, g234, g345],
     Deduction.edge m23 m24 0 [gA1, g12, g123, g234],
     Deduction.edge m24 m25 0 [gA1, g12, g123],
     Deduction.edge m25 m26 0 [gA1, g12],
     Deduction.edge m26 m27 0 [gA1]]
    []

/-- The compressed witness: -/
def exFan3C : Graph := compress_nodes_graph exFan3

end ExFan3

open ExFan2 ExFan3



/-- Token multiplicity of column φ: -/
def columnMultiplicity (d : Graph) (φ : Formula) : Nat :=
  match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
  | some v => max 1 (flowAt d (stdFuel d) v).length
  | none => 1

/-- The route enumeration: -/
def routesOf (d : Graph) : List (Nat × Nat) :=
  (buildFormulas d).zipIdx.flatMap fun (φ, col) =>
    (List.range (columnMultiplicity d φ)).map (fun r => (col, r))

/--  T ; the carrier budget (Σ per-column multiplicities).  -/
def routeCount (d : Graph) : Nat := (routesOf d).length

/--  Per-column multiplicity table (diagnostic rendering).  -/
def multiplicityTable (d : Graph) : List (String × Nat) :=
  (buildFormulas d).map fun φ => (toString (repr φ), columnMultiplicity d φ)

/-- The Flow route (ancestor address) behind global data (col, r). -/
def routeAddress? (d : Graph) (col r : Nat) : Option ColourPath :=
  match (buildFormulas d)[col]? with
  | none => none
  | some φ =>
      match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
      | none => none
      | some v => ((flowAt d (stdFuel d) v)[r]?).map (·.2)


abbrev MultiPathInput := List (List (Nat × Nat))

/-- Seed ONE token per route. -/
def initialize_tokens_multi {n : Nat}
    (routes : List (Nat × Nat))
    (initial_vectors : List (List.Vector Bool n))
    (top_level : Nat) : List (Token n) :=
  routes.zipIdx.map fun ((col, _r), routeIdx) =>
    { origin_column := routeIdx
      source_column := col
      current_level := top_level
      current_column := col
      dep_vector := (initial_vectors[col]?).getD (List.Vector.replicate n false)
      input_label := 0 }

/-- nodeError minus the exact-arity count (tokens.length = arity): -/
def nodeError_multi {n : Nat}
    (node : CircuitNode n) (node_incoming : NodeIncoming)
    (tokens : List (Token n)) : Bool :=
  match tokens with
  | [] => false
  | t :: _ =>
      match decodeInputLabel node_incoming t.input_label with
      | none => true
      | some (r, _slot, _src) =>
          let reqd := node_incoming[r]!
          let arity := reqd.length
          let ruleExists := decide (r < node.rules.length)
          let labelsOK := tokens.all (fun s =>
            match decodeInputLabel node_incoming s.input_label with
            | some (r', slot', src') =>
                decide (r' = r ∧ slot' < arity ∧ s.source_column = src')
            | none => false)
          let slotsComplete := (List.range arity).all (fun i =>
            tokens.any (fun s =>
              match decodeInputLabel node_incoming s.input_label with
              | some (r', slot', src') =>
                  decide (r' = r ∧ slot' = i ∧ s.source_column = src')
              | none => false))
          ! (ruleExists && labelsOK && slotsComplete)

/-- evaluate_node with the multi arrival check; -/
def evaluate_node_multi {n : Nat}
    (node : CircuitNode n)
    (node_incoming : NodeIncoming)
    (tokens_at_node : List (Token n))
    : (List.Vector Bool n) × Bool :=
  if tokens_at_node.isEmpty then
    (List.Vector.replicate n false, false)
  else if nodeError_multi node node_incoming tokens_at_node then
    (List.Vector.replicate n false, true)
  else
    match selectedRuleIndex? node_incoming tokens_at_node with
    | none => (List.Vector.replicate n false, true)
    | some ruleIdx =>
        let available_inputs := tokens_at_node.map fun t => (t.source_column, t.dep_vector)
        let sel := ruleSelectorForIndex ruleIdx
        ((node_logic_with_routing (activate_node_from_tokens node sel).rules
            node_incoming available_inputs).fst,
         false)

/-- One grid cell of the multi evaluation: -/
def evaluate_cell_multi {n : Nat}
    (tokens : List (Token n))
    (expected : List Nat)
    (cnode : CircuitNode n)
    (node_incoming : NodeIncoming)
    (col_idx : Nat) : (List.Vector Bool n) × Bool :=
  let tokens_here := tokens.filter (·.current_column = col_idx)
  let countOK := tokens_here.length == (expected[col_idx]?.getD 0)
  let (out, err) := evaluate_node_multi cnode node_incoming tokens_here
  (out, err || !countOK)

/-- evaluate_layer with the per-column expected-count check (compiled from the graph's route structure by. -/
def evaluate_layer_multi {n : Nat}
    (layer : GridLayer n)
    (tokens : List (Token n))
    (expected : List Nat)
    : (List (List.Vector Bool n)) × Bool :=
  let results := layer.nodes.zipIdx.map fun (cnode, col_idx) =>
    evaluate_cell_multi tokens expected cnode (layer.incoming[col_idx]!) col_idx
  let outputs := results.map Prod.fst
  let errors := results.map Prod.snd
  let any_error := errors.any id
  (outputs, any_error)

/-- eval_from_level with per-layer expected profiles threaded alongside the layers; -/
def eval_from_level_multi {n : Nat}
    (paths : MultiPathInput)
    (level : Nat)
    (tokens : List (Token n))
    (remaining_layers : List (GridLayer n × List Nat))
    (accumulated_error : Bool)
    (num_levels : Nat)
    : (List (List.Vector Bool n)) × Bool :=
  match remaining_layers with
  | [] =>
      let final_outputs := (List.range n).map fun _ => List.Vector.replicate n false
      (final_outputs, accumulated_error)
  | (layer, expected) :: rest =>
      let (outputs, layer_error) := evaluate_layer_multi layer tokens expected
      match rest with
      | [] =>
          (outputs, accumulated_error || layer_error)
      | _ =>
          let new_tokens := propagate_tokens tokens paths level num_levels outputs
          eval_from_level_multi paths (level - 1) new_tokens rest
            (accumulated_error || layer_error) num_levels

/--  Multi-token analogue of `get_eval_result`.  -/
def get_eval_result_multi {n : Nat}
    (routes : List (Nat × Nat))
    (layers : List (GridLayer n))
    (expected : List (List Nat))
    (initial_vectors : List (List.Vector Bool n))
    (paths : MultiPathInput) : (List (List.Vector Bool n)) × Bool :=
  let num_levels := layers.length
  let initial_tokens := initialize_tokens_multi routes initial_vectors num_levels
  eval_from_level_multi paths (num_levels - 1) initial_tokens
    (layers.zip expected) false num_levels



/-- Canonical per-route paths: -/
def multiPathsFromFlow (d : Graph) : MultiPathInput :=
  let formulas := buildFormulas d
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  (routesOf d).map fun (col, r) =>
    match formulas[col]? with
    | none => List.replicate numSteps (0, 0)
    | some φ =>
        match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
        | none => List.replicate numSteps (0, 0)
        | some v =>
            match (flowAt d (stdFuel d) v)[r]? with
            | some bp =>
                let delay := maxLvl - v.LEVEL
                List.replicate delay (col + 1, 0) ++
                  routeFromFlowC d formulas (numSteps - delay) v.LEVEL φ bp.2
            | none => List.replicate numSteps (0, 0)

/-- Per-layer, per-column expected token counts, compiled from the graph's own route structure (the canonical route. -/
def expectedCounts (d : Graph) : List (List Nat) :=
  let numCols := (buildFormulas d).length
  let numLevels := (buildGridFromDLDS d).length
  let routes := routesOf d
  let P := multiPathsFromFlow d
  (List.range numLevels).map fun layerIdx =>
    (List.range numCols).map fun c =>
      if layerIdx = 0 then
        (routes.filter (fun rc => rc.1 == c)).length
      else
        (P.filter (fun steps =>
          match steps[layerIdx - 1]? with
          | some st => st.1 == c + 1
          | none => false)).length



/--  Multi-token evaluation result on graph `d`.  -/
def getEvalResultMultiDLDS (d : Graph) (paths : MultiPathInput) :
    (List (List.Vector Bool (buildFormulas d).length)) × Bool :=
  get_eval_result_multi (routesOf d) (buildGridFromDLDS d) (expectedCounts d)
    (initialVectorsFromDLDS d) paths

/-- Multi-token acceptance, same acceptance shape as evaluateCircuit: -/
def evaluateDLDS_multi (d : Graph) (paths : MultiPathInput)
    (goal_column : Nat) : Bool :=
  let (final_outputs, had_error) := getEvalResultMultiDLDS d paths
  if h : goal_column < final_outputs.length then
    had_error || (final_outputs.get ⟨goal_column, h⟩).toList.all (· = false)
  else true

/--  Multi-token structural-error flag on `P`.  -/
def evalErrorMultiB (d : Graph) (P : MultiPathInput) : Bool :=
  (getEvalResultMultiDLDS d P).2

/--  Goal-vector-all-false on a multi-token input (cf. `dischargedOnB`).  -/
def dischargedOnMultiB (d : Graph) (P : MultiPathInput) : Bool :=
  match (getEvalResultMultiDLDS d P).1[goalColumn d]? with
  | none => true
  | some v => v.toList.all (fun b => !b)

/-- Diagnostic row (error flag, genuinely discharged, accept). -/
def multiVerdict (d : Graph) (P : MultiPathInput) : Bool × Bool × Bool :=
  (evalErrorMultiB d P, dischargedOnMultiB d P,
   evaluateDLDS_multi d P (goalColumn d))



def gateA (d : Graph) : Bool × Bool × (Bool × Bool × Bool) :=
  (decide (multiPathsFromFlow d = pathsFromDLDS d),
   decide (getEvalResultMultiDLDS d (multiPathsFromFlow d) =
           get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
             (pathsFromDLDS d)),
   (evaluateDLDS_multi d (multiPathsFromFlow d) (goalColumn d) ==
      evaluateDLDS d (pathsFromDLDS d) (goalColumn d),
    evalErrorMultiB d (multiPathsFromFlow d),
    dischargedOnMultiB d (multiPathsFromFlow d)))

/--  Global route index of column `c`'s first route.  -/
def firstRouteOfColumn (d : Graph) (c : Nat) : Nat :=
  (routesOf d).findIdx (fun rc => rc.1 == c)

/--
 Wrong-colour multi routing: the A1 column's first token is diverted to the
    goal column at step 1 ; no edge of the collapsed `A2` kernel reaches it on
    any route.
-/
def exFan3WrongColour : MultiPathInput :=
  setStep (multiPathsFromFlow exFan3C)
    (firstRouteOfColumn exFan3C (colOf exFan3C ExFan3.gA1)) 1
    (goalColumn exFan3C + 1, 1)

/--
 Fan-starving multi routing: every route whose address threads colour
    `colour` delivers its first real step and then stops ; the coloured
    continuation below the collapsed kernel is deliberately unserved.
-/
def multiPathsStarving (d : Graph) (colour : Nat) : MultiPathInput :=
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  ((routesOf d).zip (multiPathsFromFlow d)).map fun ((col, r), full) =>
    let starved :=
      match routeAddress? d col r with
      | some p => p.contains colour
      | none => false
    if starved then
      let delay :=
        match (buildFormulas d)[col]? with
        | some φ =>
            match d.NODES.find? (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
            | some v => maxLvl - v.LEVEL
            | none => 0
        | none => 0
      full.take (delay + 1) ++ List.replicate (numSteps - delay - 1) (0, 0)
    else full

/--
 Colour 114 is the collapsed `A2` kernel's third out-colour (the m14
    copy): starving it gives the kernel only 2 of its 3 colours.
-/
def exFan3Starved : MultiPathInput := multiPathsStarving exFan3C 114


end Semantic
