import Evaluator.Compressed.Routing

open scoped Classical

/-! Multi-token evaluation for compressed DLDS graphs. One token is seeded per
Flow route so a collapsed node can serve an out-fan larger than its rule arity. -/

open DLDS

namespace DLDS

open Flow


namespace Examples.TernaryFan

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

end Examples.TernaryFan

open DLDS.Examples.BinaryFan DLDS.Examples.TernaryFan



/-- A concrete hypothesis occurrence together with one of its Flow routes. -/
abbrev SourceRoute := Vertex × FlowPair

/-- All source routes for formula `φ`, retaining the concrete hypothesis
vertex.  Formula columns remain shared, but repeated hypothesis occurrences at
different levels are no longer discarded by a first-match lookup. -/
def sourceRoutesForFormula (d : Graph) (φ : Formula) : List SourceRoute :=
  (d.NODES.filter (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ))).flatMap
    fun v => (flowAt d (stdFuel d) v).map fun bp => (v, bp)

/-- Source routes computed once per formula column.  Several canonical
operations need the same table; indexing this value avoids recomputing
`flowAt` once per route occurrence. -/
def sourceRouteTable (d : Graph) : List (List SourceRoute) :=
  (buildFormulas d).map fun φ => sourceRoutesForFormula d φ

/-- Token multiplicity of column φ.  Columns with no hypothesis retain their
single resting token; hypothesis columns get one token per concrete source
route. -/
def columnMultiplicity (d : Graph) (φ : Formula) : Nat :=
  max 1 (sourceRoutesForFormula d φ).length

/-- The route enumeration: -/
def routesOf (d : Graph) : List (Nat × Nat) :=
  (buildFormulas d).zipIdx.flatMap fun (φ, col) =>
    (List.range (columnMultiplicity d φ)).map (fun r => (col, r))

/--  T ; the carrier budget (Σ per-column multiplicities).  -/
def routeCount (d : Graph) : Nat := (routesOf d).length

/--  Per-column multiplicity table (diagnostic rendering).  -/
def multiplicityTable (d : Graph) : List (String × Nat) :=
  (buildFormulas d).map fun φ => (toString (repr φ), columnMultiplicity d φ)

/-- The concrete source occurrence and Flow pair behind `(col, r)`. -/
def sourceRoute? (d : Graph) (col r : Nat) : Option SourceRoute :=
  match (buildFormulas d)[col]? with
  | none => none
  | some φ => (sourceRoutesForFormula d φ)[r]?

lemma mem_sourceRoutesForFormula_iff (d : Graph) (φ : Formula)
    (v : Vertex) (bp : FlowPair) :
    (v, bp) ∈ sourceRoutesForFormula d φ ↔
      v ∈ d.NODES ∧ v.HYPOTHESIS = true ∧ v.FORMULA = φ ∧
        bp ∈ flowAt d (stdFuel d) v := by
  unfold sourceRoutesForFormula
  rw [List.mem_flatMap]
  constructor
  · rintro ⟨u, hu, hmap⟩
    rw [List.mem_filter] at hu
    obtain ⟨bp', hbp', hpair⟩ := List.mem_map.mp hmap
    cases hpair
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hu
    exact ⟨hu.1, hu.2.1, hu.2.2, hbp'⟩
  · rintro ⟨hv, hhyp, hformula, hbp⟩
    refine ⟨v, ?_, ?_⟩
    · rw [List.mem_filter]
      exact ⟨hv, by simp [hhyp, hformula]⟩
    · exact List.mem_map.mpr ⟨bp, hbp, rfl⟩

lemma sourceRoute?_some_props (d : Graph) {col r : Nat} {φ : Formula}
    {v : Vertex} {bp : FlowPair}
    (hformula : (buildFormulas d)[col]? = some φ)
    (hsource : sourceRoute? d col r = some (v, bp)) :
    v ∈ d.NODES ∧ v.HYPOTHESIS = true ∧ v.FORMULA = φ ∧
      bp ∈ flowAt d (stdFuel d) v := by
  unfold sourceRoute? at hsource
  rw [hformula] at hsource
  change (sourceRoutesForFormula d φ)[r]? = some (v, bp) at hsource
  have hr : r < (sourceRoutesForFormula d φ).length := by
    by_contra hnot
    rw [List.getElem?_eq_none (Nat.le_of_not_lt hnot)] at hsource
    simp at hsource
  rw [List.getElem?_eq_getElem hr, Option.some.injEq] at hsource
  apply (mem_sourceRoutesForFormula_iff d φ v bp).mp
  rw [← hsource]
  exact List.getElem_mem hr

/-- The Flow route (ancestor address) behind global data (col, r). -/
def routeAddress? (d : Graph) (col r : Nat) : Option ColourPath :=
  (sourceRoute? d col r).map fun source => source.2.2


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

/-- Tokens for one concrete rule occurrence at a merged circuit cell. -/
def tokensForRule {n : Nat} (node_incoming : NodeIncoming) (ruleIdx : Nat)
    (tokens : List (Token n)) : List (Token n) :=
  tokens.filter fun token =>
    match decodeInputLabel node_incoming token.input_label with
    | some (r, _slot, _src) => r == ruleIdx
    | none => false

/-- Every token names an existing rule.  Slot/source completeness is checked
inside each rule group by `evaluate_node_multi`. -/
def tokenRulesValidB {n : Nat} (node : CircuitNode n)
    (node_incoming : NodeIncoming) (tokens : List (Token n)) : Bool :=
  tokens.all fun token =>
    match decodeInputLabel node_incoming token.input_label with
    | some (r, _slot, _src) => decide (r < node.rules.length)
    | none => false

/-- Dependency carriers for one concrete premise slot. -/
def tokensForRuleSlot {n : Nat} (incoming : NodeIncoming) (ruleIdx slot : Nat)
    (tokens : List (Token n)) : List (Token n) :=
  tokens.filter fun token =>
    match decodeInputLabel incoming token.input_label with
    | some (r, s, source) =>
        r == ruleIdx && s == slot && token.source_column == source
    | none => false

/-- The multiplicity-aware dependency result for one rule occurrence.  Each
slot is first joined componentwise, then the rule is evaluated once.  For the
intro/elim/rep rule basis this equals the join of all one-token-per-slot
branch results, while avoiding both first-representative order dependence and
an exponential Cartesian product. -/
def ruleResultVariants {n : Nat} (node : CircuitNode n)
    (incoming : NodeIncoming) (ruleIdx : Nat) (tokens : List (Token n)) :
    List (List.Vector Bool n) :=
  let required := incoming[ruleIdx]?.getD []
  if (tokensForRule incoming ruleIdx tokens).isEmpty ||
      (List.range required.length).any fun slot =>
        (tokensForRuleSlot incoming ruleIdx slot tokens).isEmpty then []
  else
    let availableInputs := required.zipIdx.map fun ((source, _edge), slot) =>
      let vectors := (tokensForRuleSlot incoming ruleIdx slot tokens).map
        fun token => token.dep_vector
      (source, list_or vectors)
    let selector := ruleSelectorForIndex ruleIdx
    [(node_logic_with_routing
      (activate_node_from_tokens node selector).rules incoming availableInputs).1]

def ruleOccurrenceErrorB {n : Nat} (node : CircuitNode n)
    (incoming : NodeIncoming) (ruleIdx : Nat) (tokens : List (Token n)) : Bool :=
  let group := tokensForRule incoming ruleIdx tokens
  !group.isEmpty && (ruleResultVariants node incoming ruleIdx tokens).isEmpty

/-- Evaluate all simultaneously active rule occurrences at one merged cell.
Each occurrence is validated independently.  When a premise slot carries
several dependency alternatives, the alternatives in each slot are joined
componentwise before that occurrence is evaluated. -/
def evaluate_node_occurrences_multi {n : Nat}
    (node : CircuitNode n)
    (node_incoming : NodeIncoming)
    (tokens_at_node : List (Token n)) : (List.Vector Bool n) × Bool :=
  if tokens_at_node.isEmpty then
    (List.Vector.replicate n false, false)
  else
    let ruleIndices := List.range node.rules.length
    let variants := ruleIndices.flatMap fun ruleIdx =>
      ruleResultVariants node node_incoming ruleIdx tokens_at_node
    (list_or variants,
      !tokenRulesValidB node node_incoming tokens_at_node ||
        ruleIndices.any fun ruleIdx =>
          ruleOccurrenceErrorB node node_incoming ruleIdx tokens_at_node)

/-- One grid cell of the multi evaluation: -/
def evaluate_cell_multi {n : Nat}
    (tokens : List (Token n))
    (expected : List Nat)
    (cnode : CircuitNode n)
    (node_incoming : NodeIncoming)
    (col_idx : Nat) : (List.Vector Bool n) × Bool :=
  let tokens_here := tokens.filter (·.current_column = col_idx)
  let countOK := tokens_here.length == (expected[col_idx]?.getD 0)
  let (out, err) :=
    evaluate_node_occurrences_multi cnode node_incoming tokens_here
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
  let sourceTable := sourceRouteTable d
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  (routesOf d).map fun (col, r) =>
    match formulas[col]? with
    | none => List.replicate numSteps (0, 0)
    | some φ =>
        match sourceTable[col]?.bind fun routes => routes[r]? with
        | none => List.replicate numSteps (0, 0)
        | some (v, bp) =>
            let delay := maxLvl - v.LEVEL
            List.replicate delay (col + 1, 0) ++
              routeFromFlowC d formulas (numSteps - delay) v.LEVEL φ bp.2

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


/-! ### Exhausted-arrival composition

These definitions provide graph-aware propagation which can create one
carrier for each compatible locally-produced continuation route. -/

structure ContinuationPlan where
  vertex : Vertex
  route : FlowPair
  deriving BEq, DecidableEq, Repr

/-- Exact, witnessable, nonempty local continuations at the selected vertices.
Empty routes are arrival facts, not routes along which a new carrier can be
sent.  The vertex restriction is useful for testing one composition site
without constructing unrelated plans. -/
def continuationPlansForNodes (d : Graph) (enabledNodes : List Nat) :
    List ContinuationPlan :=
  ((d.NODES.filter fun v => enabledNodes.contains v.NUMBER).flatMap fun v =>
    (WitnessableRoutes d v (flowAt d (stdFuel d) v)).filterMap fun bp =>
      if bp.2.isEmpty then none else some { vertex := v, route := bp }).eraseDups

def continuationPlans (d : Graph) : List ContinuationPlan :=
  continuationPlansForNodes d (d.NODES.map fun v => v.NUMBER)

def fullPathForRoute (d : Graph) (v : Vertex) (bp : FlowPair) :
    List (Nat × Nat) :=
  let formulas := buildFormulas d
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  let col := formulas.idxOf v.FORMULA
  let delay := maxLvl - v.LEVEL
  List.replicate delay (col + 1, 0) ++
    routeFromFlowC d formulas (numSteps - delay) v.LEVEL v.FORMULA bp.2

def fullArrivalFlagsForRoute (d : Graph) (v : Vertex) (bp : FlowPair) :
    List Bool :=
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  let delay := maxLvl - v.LEVEL
  List.replicate delay false ++
    arrivalFlagsFromFlowC d (numSteps - delay) v.LEVEL v.FORMULA bp.2

/-- The evaluator reads target columns through `colEff`; arrival metadata must
therefore use the same observational equality, rather than exact list/label
equality, for the existing `ReadEquiv` theorem to remain true. -/
def pathTargetEquivalentB (left right : List (Nat × Nat)) : Bool :=
  (List.range (max left.length right.length)).all fun step =>
    colEff left step == colEff right step

/-- Select arrival metadata by the actual source path, not by its position.
This keeps co-located path permutations meaningful. -/
def sourceArrivalFlagsForPath (d : Graph) (col : Nat)
    (steps : List (Nat × Nat)) : List Bool :=
  let numSteps := (buildGridFromDLDS d).length - 1
  match (buildFormulas d)[col]? with
  | none => List.replicate numSteps false
  | some φ =>
      match (sourceRoutesForFormula d φ).find? fun source =>
          pathTargetEquivalentB (fullPathForRoute d source.1 source.2) steps with
      | none => List.replicate numSteps false
      | some source => fullArrivalFlagsForRoute d source.1 source.2

def exhaustedStopAtStepB (d : Graph) (originCol : Nat)
    (path : List (Nat × Nat)) (stepIdx : Nat) : Bool :=
  let formulas := buildFormulas d
  let numLevels := (buildGridFromDLDS d).length
  if stepIdx = 0 then false
  else
    match colEff path stepIdx, colEff path (stepIdx - 1) with
    | none, some arrivalStep =>
      let arrivalIdx := stepIdx - 1
      let sourceCol :=
        if arrivalIdx = 0 then originCol
        else
          match colEff path (arrivalIdx - 1) with
          | none => originCol
          | some prior => prior.1 - 1
      let targetCol := arrivalStep.1 - 1
      let targetLevel := numLevels - stepIdx - 1
      match formulas[sourceCol]?, formulas[targetCol]? with
      | some sourceFormula, some targetFormula =>
          match d.NODES.find? (fun v =>
              v.LEVEL == targetLevel + 1 && v.FORMULA == sourceFormula),
            d.NODES.find? (fun v =>
              v.LEVEL == targetLevel && v.FORMULA == targetFormula) with
          | some source, some target =>
              !(edgesBetween d source target).isEmpty &&
                (edgeOfColour d target 0).isNone
          | _, _ => false
      | _, _ => false
    | _, _ => false

def arrivalFlagsForPathShape (d : Graph) (originCol : Nat)
    (path : List (Nat × Nat)) : List Bool :=
  (List.range ((buildGridFromDLDS d).length - 1)).map fun stepIdx =>
    exhaustedStopAtStepB d originCol path stepIdx

def arrivalFlagsForPaths (d : Graph) (originCols : List Nat)
    (paths : MultiPathInput) : List (List Bool) :=
  originCols.zipIdx.map fun (originCol, index) =>
    arrivalFlagsForPathShape d originCol (paths.getD index [])

def initialArrivalFlags (d : Graph) (paths : MultiPathInput) : List (List Bool) :=
  arrivalFlagsForPaths d ((routesOf d).map Prod.fst) paths

def compositionPathsFromPlans (d : Graph) (plans : List ContinuationPlan) :
    MultiPathInput :=
  plans.map fun plan => fullPathForRoute d plan.vertex plan.route

def compositionArrivalFlagsFromPlans (d : Graph)
    (plans : List ContinuationPlan) : List (List Bool) :=
  let paths := compositionPathsFromPlans d plans
  let origins := plans.map fun plan => (buildFormulas d).idxOf plan.vertex.FORMULA
  arrivalFlagsForPaths d origins paths

def evaluationPathsWithComposition (d : Graph) (plans : List ContinuationPlan)
    (paths : MultiPathInput) :
    MultiPathInput :=
  paths ++ compositionPathsFromPlans d plans

def evaluationArrivalFlags (d : Graph) (plans : List ContinuationPlan)
    (paths : MultiPathInput) :
    List (List Bool) :=
  initialArrivalFlags d paths ++ compositionArrivalFlagsFromPlans d plans

/-- Vertices at which one of the supplied carriers is both an exhausted
arrival and about to stop.  A zero step after another zero step is ignored:
the token was already removed at the first stop. -/
def stoppedArrivalSites (d : Graph) (originCols : List Nat)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool)) : List Vertex :=
  let formulas := buildFormulas d
  let numLevels := (buildGridFromDLDS d).length
  let candidates := (originCols.zip (paths.zip arrivalFlags)).flatMap fun row =>
    let originCol := row.1
    let path := row.2.1
    let flags := row.2.2
    (List.range (min path.length flags.length)).filterMap fun stepIdx =>
      let step := path.getD stepIdx (0, 0)
      let currentCol? :=
        if stepIdx = 0 then some originCol
        else
          let previous := path.getD (stepIdx - 1) (0, 0)
          if previous.1 = 0 then none else some (previous.1 - 1)
      if flags.getD stepIdx false && step == (0, 0) then
        currentCol?.bind fun currentCol =>
          formulas[currentCol]?.bind fun formula =>
            d.NODES.find? fun v =>
              v.LEVEL == numLevels - stepIdx - 1 && v.FORMULA == formula
      else none
  d.NODES.filter fun v => candidates.contains v

def continuationPlansReachableGo (d : Graph) (enabledNodes : List Nat) :
    Nat → List Vertex → List Vertex → List ContinuationPlan
  | 0, _, _ => []
  | fuel + 1, frontier, seen =>
      let fresh := frontier.filter fun v =>
        enabledNodes.contains v.NUMBER && !seen.contains v
      if fresh.isEmpty then []
      else
        let newPlans := continuationPlansForNodes d (fresh.map fun v => v.NUMBER)
        let newPaths := compositionPathsFromPlans d newPlans
        let newFlags := compositionArrivalFlagsFromPlans d newPlans
        let newOrigins := newPlans.map fun plan =>
          (buildFormulas d).idxOf plan.vertex.FORMULA
        let next := stoppedArrivalSites d newOrigins newPaths newFlags
        (newPlans ++ continuationPlansReachableGo d enabledNodes fuel next
          (seen ++ fresh)).eraseDups

/-- Bounded transitive closure of continuation sites reachable from the
canonical source carriers.  Each vertex is expanded at most once and the fuel
is one more than the number of vertices. -/
def continuationPlansReachable (d : Graph) (enabledNodes : List Nat)
    (sourcePaths : MultiPathInput) : List ContinuationPlan :=
  let sourceFlags := initialArrivalFlags d sourcePaths
  let sourceOrigins := (routesOf d).map Prod.fst
  let initialSites := stoppedArrivalSites d sourceOrigins sourcePaths sourceFlags
  continuationPlansReachableGo d enabledNodes (d.NODES.length + 1)
    initialSites []

def depVectorSubsetB {n : Nat} (small large : List.Vector Bool n) : Bool :=
  (small.toList.zip large.toList).all fun bits => !bits.1 || bits.2

def tokenAtExhaustedStopB {n : Nat} (d : Graph) (paths : MultiPathInput)
    (_arrivalFlags : List (List Bool)) (stepIdx : Nat) (token : Token n) : Bool :=
  let path := paths.getD token.origin_column []
  let formulas := buildFormulas d
  let numLevels := (buildGridFromDLDS d).length
  if stepIdx = 0 then false
  else
    match colEff path stepIdx, colEff path (stepIdx - 1) with
    | none, some arrivalStep =>
      let targetLevel := numLevels - stepIdx - 1
      match formulas[token.source_column]?, formulas[token.current_column]? with
      | some sourceFormula, some targetFormula =>
          match d.NODES.find? (fun v =>
              v.LEVEL == targetLevel + 1 && v.FORMULA == sourceFormula),
            d.NODES.find? (fun v =>
              v.LEVEL == targetLevel && v.FORMULA == targetFormula) with
          | some source, some target =>
              arrivalStep.1 == token.current_column + 1 &&
                !(edgesBetween d source target).isEmpty &&
                (edgeOfColour d target 0).isNone
          | _, _ => false
      | _, _ => false
    | _, _ => false

/-- Candidate carriers created at one evaluated level.  Plans are already
exact-deduplicated; each compatible plan is emitted once per node, not once per
arriving token. -/
def compositionPlanCompatibleB (d : Graph) (paths : MultiPathInput)
    (arrivalFlags : List (List Bool)) (stepIdx : Nat)
    (layer : GridLayer (buildFormulas d).length)
    (tokens : List (Token (buildFormulas d).length))
    (plan : ContinuationPlan) : Bool :=
  let formulas := buildFormulas d
  let col := formulas.idxOf plan.vertex.FORMULA
  match layer.nodes[col]?, layer.incoming[col]? with
  | some cnode, some incoming =>
      let tokensHere := tokens.filter fun token => token.current_column == col
      tokensHere.any fun token =>
        tokenAtExhaustedStopB d paths arrivalFlags stepIdx token &&
          match decodeInputLabel incoming token.input_label with
          | none => false
          | some decoded =>
              (ruleResultVariants cnode incoming decoded.1 tokensHere).any
                fun result =>
                  depVectorSubsetB (depVector formulas plan.route.1) result
  | _, _ => false

def compositionSeedsAtLevel (d : Graph) (enabledNodes : List Nat)
    (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    (tokens : List (Token (buildFormulas d).length))
    : List (Token (buildFormulas d).length) :=
  let formulas := buildFormulas d
  let stepIdx := numLevels - level - 1
  plans.zipIdx.filterMap fun (plan, planIdx) =>
    let v := plan.vertex
    let col := formulas.idxOf v.FORMULA
    let compatible := compositionPlanCompatibleB d paths arrivalFlags stepIdx
      layer tokens plan
    if enabledNodes.contains v.NUMBER && v.LEVEL == level &&
        !(get_rule.outgoing v d).isEmpty && compatible then
      some
        { origin_column := routeCount d + planIdx
          source_column := col
          current_level := level
          current_column := col
          dep_vector := depVector formulas plan.route.1
          input_label := 0 }
    else none

/-- Advance a freshly-created continuation carrier without overwriting its
locally-produced dependency vector with the aggregate cell output. -/
def advanceCompositionSeed {n : Nat} (token : Token n) (paths : MultiPathInput)
    (level numLevels : Nat) : Option (Token n) :=
  if hPath : token.origin_column < paths.length then
    let path := paths.get ⟨token.origin_column, hPath⟩
    if hStep : level > 0 ∧ numLevels - level - 1 < path.length then
      let step := path.get ⟨numLevels - level - 1, hStep.2⟩
      if step.1 = 0 then none
      else
        some
          { origin_column := token.origin_column
            source_column := token.current_column
            current_level := level - 1
            current_column := step.1 - 1
            dep_vector := token.dep_vector
            input_label := step.2 }
    else none
  else none

/-- Advance compatible plans directly from their own canonical local path.
Unlike ordinary carriers, these origins live in the fixed suffix after the
source-route table, so their propagation is independent of any permutation of
the supplied source paths. -/
def advancedCompositionSeedsAtLevel (d : Graph) (enabledNodes : List Nat)
    (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    (tokens : List (Token (buildFormulas d).length)) :
      List (Token (buildFormulas d).length) :=
  let formulas := buildFormulas d
  let stepIdx := numLevels - level - 1
  plans.zipIdx.filterMap fun (plan, planIdx) =>
    let v := plan.vertex
    let col := formulas.idxOf v.FORMULA
    let path := fullPathForRoute d v plan.route
    if enabledNodes.contains v.NUMBER && v.LEVEL == level &&
        !(get_rule.outgoing v d).isEmpty &&
        compositionPlanCompatibleB d paths arrivalFlags stepIdx layer tokens plan
    then
      if hStep : level > 0 ∧ stepIdx < path.length then
        let step := path.get ⟨stepIdx, hStep.2⟩
        if step.1 = 0 then none
        else some
          { origin_column := routeCount d + planIdx
            source_column := col
            current_level := level - 1
            current_column := step.1 - 1
            dep_vector := depVector formulas plan.route.1
            input_label := step.2 }
      else none
    else none

def propagateTokensComposedAt (d : Graph) (enabledNodes : List Nat)
    (plans : List ContinuationPlan)
    (tokens : List (Token (buildFormulas d).length)) (paths : MultiPathInput)
    (arrivalFlags : List (List Bool)) (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    (outputs : List (List.Vector Bool (buildFormulas d).length)) :
      List (Token (buildFormulas d).length) :=
  let ordinary := propagate_tokens tokens paths level numLevels outputs
  ordinary ++ advancedCompositionSeedsAtLevel d enabledNodes plans paths
    arrivalFlags level numLevels layer tokens

def tokenCounts {n : Nat} (numCols : Nat) (tokens : List (Token n)) : List Nat :=
  (List.range numCols).map fun col =>
    (tokens.filter fun token => token.current_column == col).length

def expectedCountsComposedGo (d : Graph) (enabledNodes : List Nat)
    (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (numLevels : Nat) :
    List (GridLayer (buildFormulas d).length) → Nat →
      List (Token (buildFormulas d).length) → List (List Nat)
  | [], _, _ => []
  | layer :: rest, level, tokens =>
      let counts := tokenCounts layer.nodes.length tokens
      let evaluated := evaluate_layer_multi layer tokens counts
      counts :: expectedCountsComposedGo d enabledNodes plans paths arrivalFlags
        numLevels rest (level - 1)
        (propagateTokensComposedAt d enabledNodes plans tokens paths arrivalFlags
          level numLevels layer evaluated.1)

def expectedCountsComposedWithPlans (d : Graph) (enabledNodes : List Nat)
    (plans : List ContinuationPlan) : List (List Nat) :=
  let sourcePaths := multiPathsFromFlow d
  let paths := evaluationPathsWithComposition d plans sourcePaths
  let flags := evaluationArrivalFlags d plans sourcePaths
  let layers := buildGridFromDLDS d
  let tokens := initialize_tokens_multi (routesOf d) (initialVectorsFromDLDS d)
    layers.length
  expectedCountsComposedGo d enabledNodes plans paths flags layers.length layers
    (layers.length - 1) tokens

def expectedCountsComposedAt (d : Graph) (enabledNodes : List Nat) :
    List (List Nat) :=
  let sourcePaths := multiPathsFromFlow d
  let plans := continuationPlansReachable d enabledNodes sourcePaths
  expectedCountsComposedWithPlans d enabledNodes plans

def evalFromLevelComposedAt (d : Graph) (enabledNodes : List Nat)
    (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (numLevels : Nat) :
    Nat → List (Token (buildFormulas d).length) →
      List (GridLayer (buildFormulas d).length × List Nat) → Bool →
      List (List.Vector Bool (buildFormulas d).length) × Bool
  | _, _, [], accumulatedError =>
      ((List.range (buildFormulas d).length).map fun _ =>
          List.Vector.replicate (buildFormulas d).length false,
        accumulatedError)
  | level, tokens, (layer, expected) :: rest, accumulatedError =>
      let evaluated := evaluate_layer_multi layer tokens expected
      match rest with
      | [] => (evaluated.1, accumulatedError || evaluated.2)
      | _ =>
          evalFromLevelComposedAt d enabledNodes plans paths arrivalFlags numLevels
            (level - 1)
            (propagateTokensComposedAt d enabledNodes plans tokens paths arrivalFlags
              level numLevels layer evaluated.1)
            rest (accumulatedError || evaluated.2)

def getEvalResultMultiComposedAt (d : Graph) (enabledNodes : List Nat)
    (sourcePaths : MultiPathInput) :
    List (List.Vector Bool (buildFormulas d).length) × Bool :=
  let layers := buildGridFromDLDS d
  let plans := continuationPlansReachable d enabledNodes (multiPathsFromFlow d)
  let paths := evaluationPathsWithComposition d plans sourcePaths
  let flags := evaluationArrivalFlags d plans sourcePaths
  let tokens := initialize_tokens_multi (routesOf d) (initialVectorsFromDLDS d)
    layers.length
  evalFromLevelComposedAt d enabledNodes plans paths flags layers.length
    (layers.length - 1) tokens
    (layers.zip (expectedCountsComposedWithPlans d enabledNodes plans)) false

/-- Canonical finite representation of exactly the effective path reads made
by the evaluator.  It removes irrelevant labels after a stop and pads/truncates
every route to the grid's fixed number of propagation steps. -/
def normalizeReadPath (d : Graph) (path : List (Nat × Nat)) :
    List (Nat × Nat) :=
  (List.range ((buildGridFromDLDS d).length - 1)).map fun stepIdx =>
    (colEff path stepIdx).getD (0, 0)

def normalizeMultiPaths (d : Graph) (paths : MultiPathInput) : MultiPathInput :=
  (List.range (routeCount d)).map fun routeIdx =>
    normalizeReadPath d (paths.getD routeIdx [])



/--  Multi-token evaluation result on graph `d`.  -/
def getEvalResultMultiDLDS (d : Graph) (paths : MultiPathInput) :
    (List (List.Vector Bool (buildFormulas d).length)) × Bool :=
  getEvalResultMultiComposedAt d (d.NODES.map fun v => v.NUMBER)
    (normalizeMultiPaths d paths)

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
    (firstRouteOfColumn exFan3C (colOf exFan3C DLDS.Examples.TernaryFan.gA1)) 1
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
        match sourceRoute? d col r with
        | some (v, _) => maxLvl - v.LEVEL
        | none => 0
      full.take (delay + 1) ++ List.replicate (numSteps - delay - 1) (0, 0)
    else full

/--
 Colour 114 is the collapsed `A2` kernel's third out-colour (the m14
    copy): starving it gives the kernel only 2 of its 3 colours.
-/
def exFan3Starved : MultiPathInput := multiPathsStarving exFan3C 114


end DLDS
