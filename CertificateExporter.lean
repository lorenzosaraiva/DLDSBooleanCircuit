import Semantic.SelfCertifyingProbe
import Semantic.ThreeVertexCompressedExample
import Lean.Data.Json

open Semantic FlowSpec SelfCertifyingProbe
open Lean










namespace CertificateExporter

private def obj (fields : List (String × Json)) : Json := Json.mkObj fields
private def arr (values : List Json) : Json := Json.arr values.toArray

def formulaJson : Formula → Json
  | .atom symbol => obj [("kind", toJson "atom"), ("symbol", toJson symbol)]
  | .implication antecedent consequent =>
      obj [("kind", toJson "implication"),
        ("antecedent", formulaJson antecedent),
        ("consequent", formulaJson consequent)]

def nodeTypeJson (d : Graph) (v : Vertex) : Json :=
  match localShape d v with
  | .source => obj [("kind", toJson "source"),
      ("hypothesis", toJson v.HYPOTHESIS), ("collapsed", toJson v.COLLAPSED)]
  | .elim major minor => obj [("kind", toJson "elimination"),
      ("major_premise", toJson major.NUMBER),
      ("minor_premise", toJson minor.NUMBER),
      ("collapsed", toJson v.COLLAPSED)]
  | .elimMulti pairs => obj [("kind", toJson "multi_elimination"),
      ("pairs", arr (pairs.map fun pair =>
        obj [("major_premise", toJson pair.1.NUMBER),
          ("minor_premise", toJson pair.2.NUMBER)])),
      ("collapsed", toJson v.COLLAPSED)]
  | .intro premise alpha beta => obj [("kind", toJson "introduction"),
      ("premise", toJson premise.NUMBER),
      ("antecedent", formulaJson alpha), ("consequent", formulaJson beta),
      ("collapsed", toJson v.COLLAPSED)]
  | .invalid => obj [("kind", toJson "invalid"),
      ("hypothesis", toJson v.HYPOTHESIS), ("collapsed", toJson v.COLLAPSED)]

def nodeJson (d : Graph) (v : Vertex) : Json :=
  obj [("id", toJson v.NUMBER), ("formula", formulaJson v.FORMULA),
    ("level", toJson v.LEVEL), ("hypothesis", toJson v.HYPOTHESIS),
    ("collapsed", toJson v.COLLAPSED), ("past", toJson v.PAST),
    ("node_type", nodeTypeJson d v)]

def vertexValueJson (v : Vertex) : Json :=
  obj [("id", toJson v.NUMBER), ("formula", formulaJson v.FORMULA),
    ("level", toJson v.LEVEL), ("hypothesis", toJson v.HYPOTHESIS),
    ("collapsed", toJson v.COLLAPSED), ("past", toJson v.PAST)]

def ancestralPathJson (id : Nat) (path : Ancestral) : Json :=
  obj [("id", toJson id), ("source", toJson path.START.NUMBER),
    ("target", toJson path.END.NUMBER),
    ("source_vertex", vertexValueJson path.START),
    ("target_vertex", vertexValueJson path.END),
    ("colours", toJson path.COLOURS)]

def witnessJson (formulas : List Formula) (witness : EdgeRouteWitness) : Json :=
  obj [("edge", obj [("source", toJson witness.edge.START.NUMBER),
      ("target", toJson witness.edge.END.NUMBER),
      ("colour", toJson witness.edge.COLOUR),
      ("dependency", arr (witness.edge.DEPENDENCY.map formulaJson)),
      ("dependency_bitvector",
        toJson (depVector formulas witness.edge.DEPENDENCY).toList)]),
    ("dependency", arr (witness.dependency.map formulaJson)),
    ("dependency_bitvector",
      toJson (depVector formulas witness.dependency).toList),
    ("residual", toJson witness.residual)]

def edgeJson (c : ExpandedCertificate) (formulas : List Formula)
    (id : Nat) (edge : Deduction) : Json :=
  let witnesses := edgeWitnessesFor c edge
  let witnessRoutes := witnesses.map fun witness =>
    obj [("dependency", arr (witness.dependency.map formulaJson)),
      ("dependency_bitvector",
        toJson (depVector formulas witness.dependency).toList),
      ("residual", toJson witness.residual)]
  obj [("id", toJson id), ("source", toJson edge.START.NUMBER),
    ("target", toJson edge.END.NUMBER), ("colour", toJson edge.COLOUR),
    ("dependency", arr (edge.DEPENDENCY.map formulaJson)),
    ("dependency_bitvector", toJson (depVector formulas edge.DEPENDENCY).toList),
    ("witness_routes", arr witnessRoutes), ("witness_count", toJson witnesses.length)]

def terminalJson (formulas : List Formula) (output : TerminalOutput) : Json :=
  obj [("node_id", toJson output.node.NUMBER),
    ("dependency", arr (output.dependency.map formulaJson)),
    ("dependency_bitvector", toJson (depVector formulas output.dependency).toList),
    ("residual", toJson output.residual)]









inductive SortedIndexName where
  | edgeRecord
  | edgeSourceColour
  | nodeLevelFormula
  | witnessEdge
  deriving Repr, BEq

def SortedIndexName.name : SortedIndexName → String
  | .edgeRecord => "edge_record"
  | .edgeSourceColour => "edge_source_colour"
  | .nodeLevelFormula => "node_level_formula"
  | .witnessEdge => "witness_edge"

inductive SortedIndexMode where
  | adjacent
  | permutation
  deriving Repr, BEq

def SortedIndexMode.name : SortedIndexMode → String
  | .adjacent => "adjacent"
  | .permutation => "permutation"

structure SortedIndexTable where
  order : List Nat
  inverse : List Nat
  deriving Repr

private def natListLE : List Nat → List Nat → Bool
  | [], _ => true
  | _ :: _, [] => false
  | left :: lefts, right :: rights =>
      if left < right then true else if left == right then natListLE lefts rights else false

private def sortedOffsets (records : List α) (key : α → List Nat) : List Nat :=
  (records.zipIdx.mergeSort fun a b => natListLE (key a.1) (key b.1)).map Prod.snd

private def inverseOrder (size : Nat) (order : List Nat) : List Nat :=
  (List.range size).map fun recordIndex => order.idxOf recordIndex

private def indexTable (records : List α) (key : α → List Nat) : SortedIndexTable :=
  let order := sortedOffsets records key
  { order := order, inverse := inverseOrder records.length order }

private def indexFormulaUniverse (c : ExpandedCertificate) : List Formula :=
  (buildFormulas c.graph ++ c.graph.EDGES.flatMap (fun edge => edge.DEPENDENCY)).eraseDups

private def witnessIndexFormulaUniverse (c : ExpandedCertificate) : List Formula :=
  (indexFormulaUniverse c ++
    c.edgeRoutes.flatMap (fun witness => witness.dependency)).eraseDups

private def edgeRecordKey (c : ExpandedCertificate) (edge : Deduction) : List Nat :=
  let formulas := buildFormulas c.graph
  let extended := indexFormulaUniverse c
  let known := (depVector formulas edge.DEPENDENCY).toList.map fun bit =>
    if bit then 1 else 0
  let unknowns := edge.DEPENDENCY.filter fun formula => !(formulas.contains formula)
  let unknownKey := match unknowns with
    | [] => [0, 0]
    | formula :: _ => [1, extended.idxOf formula]
  [edge.START.NUMBER, edge.END.NUMBER, edge.COLOUR] ++ known ++ unknownKey

private def edgeSourceColourKey (edge : Deduction) : List Nat :=
  [edge.START.NUMBER, edge.COLOUR]

private def nodeLevelFormulaKey (c : ExpandedCertificate) (node : Vertex) : List Nat :=
  [node.LEVEL, (buildFormulas c.graph).idxOf node.FORMULA]

private def witnessEdgeKey
    (c : ExpandedCertificate) (witness : EdgeRouteWitness) : List Nat :=
  let formulas := buildFormulas c.graph
  let extended := witnessIndexFormulaUniverse c
  let known := (depVector formulas witness.dependency).toList.map fun bit =>
    if bit then 1 else 0
  let unknowns := witness.dependency.filter fun formula => !(formulas.contains formula)
  let unknownKey := match unknowns with
    | [] => [0, 0]
    | formula :: _ => [1, extended.idxOf formula]
  edgeRecordKey c witness.edge ++ known ++ unknownKey ++
    [witness.residual.length] ++ witness.residual

def edgeRecordIndex (c : ExpandedCertificate) : SortedIndexTable :=
  indexTable c.graph.EDGES (edgeRecordKey c)

def edgeSourceColourIndex (c : ExpandedCertificate) : SortedIndexTable :=
  indexTable c.graph.EDGES edgeSourceColourKey

def nodeLevelFormulaIndex (c : ExpandedCertificate) : SortedIndexTable :=
  indexTable c.graph.NODES (nodeLevelFormulaKey c)

def witnessEdgeIndex (c : ExpandedCertificate) : SortedIndexTable :=
  indexTable c.edgeRoutes (witnessEdgeKey c)

def sortedIndex (c : ExpandedCertificate) : SortedIndexName → SortedIndexTable
  | .edgeRecord => edgeRecordIndex c
  | .edgeSourceColour => edgeSourceColourIndex c
  | .nodeLevelFormula => nodeLevelFormulaIndex c
  | .witnessEdge => witnessEdgeIndex c

private def permutationAtB (size slot : Nat) (table : SortedIndexTable) : Bool :=
  table.order.length == size && table.inverse.length == size &&
    match table.order[slot]? with
    | none => false
    | some recordIndex =>
        recordIndex < size &&
          match table.inverse[recordIndex]? with
          | none => false
          | some inverseSlot =>
              inverseSlot == slot && table.order[inverseSlot]? == some recordIndex

private def witnessAtSlot?
    (c : ExpandedCertificate) (slot : Nat) : Option EdgeRouteWitness := do
  let recordIndex ← (witnessEdgeIndex c).order[slot]?
  c.edgeRoutes[recordIndex]?

private def edgeWitnessLocation
    (c : ExpandedCertificate) (edge : Deduction) : Option Nat :=
  (witnessEdgeIndex c).order.findIdx? fun recordIndex =>
    match c.edgeRoutes[recordIndex]? with
    | some witness => witness.edge == edge
    | none => false

private def witnessSourceBoundary
    (c : ExpandedCertificate) (source : Nat) (upper : Bool) : Nat :=
  let order := (witnessEdgeIndex c).order
  match order.findIdx? (fun recordIndex =>
      match c.edgeRoutes[recordIndex]? with
      | some witness =>
          if upper then source < witness.edge.START.NUMBER
          else source ≤ witness.edge.START.NUMBER
      | none => false) with
  | some slot => slot
  | none => order.length

private def nodeWitnessSlice
    (c : ExpandedCertificate) (node : Vertex) : Nat × Nat :=
  (witnessSourceBoundary c node.NUMBER false,
    witnessSourceBoundary c node.NUMBER true)

private def witnessMatchesEdgeAtB
    (c : ExpandedCertificate) (edge : Deduction) (slot : Nat) : Bool :=
  match witnessAtSlot? c slot with
  | some witness => witness.edge == edge
  | none => false

private def witnessIndexAdjacentAtB
    (c : ExpandedCertificate) (slot : Nat) : Bool :=
  match witnessAtSlot? c slot, witnessAtSlot? c (slot + 1) with
  | some left, some right =>
      natListLE (witnessEdgeKey c left) (witnessEdgeKey c right)
  | _, _ => false

private def witnessIndexPermutationAtB
    (c : ExpandedCertificate) (slot : Nat) : Bool :=
  let table := witnessEdgeIndex c
  permutationAtB c.edgeRoutes.length slot table &&
    match witnessAtSlot? c slot with
    | none => false
    | some witness =>
        match c.graph.NODES.find? (fun node => node == witness.edge.START) with
        | none => false
        | some source =>
            let slice := nodeWitnessSlice c source
            slice.1 ≤ slot && slot < slice.2

private def indexedWitnessCoverageAtB
    (c : ExpandedCertificate) (edge : Deduction) : Bool :=
  match edgeWitnessLocation c edge with
  | none => false
  | some slot =>
      witnessMatchesEdgeAtB c edge slot

private def witnessesInNodeSlice
    (c : ExpandedCertificate) (node : Vertex) : List EdgeRouteWitness :=
  let slice := nodeWitnessSlice c node
  (List.range (slice.2 - slice.1)).filterMap fun offset =>
    witnessAtSlot? c (slice.1 + offset)

private def indexedCoverageAtB
    (c : ExpandedCertificate) (node : Vertex) : Bool :=
  let slice := nodeWitnessSlice c node
  let witnesses := witnessesInNodeSlice c node
  let outgoing := get_rule.outgoing node c.graph
  let beforeOk := slice.1 == 0 ||
    match witnessAtSlot? c (slice.1 - 1) with
    | some witness => !(witness.edge.START == node)
    | none => true
  let afterOk := match witnessAtSlot? c slice.2 with
    | some witness => !(witness.edge.START == node)
    | none => true
  slice.1 ≤ slice.2 && slice.2 ≤ c.edgeRoutes.length && beforeOk && afterOk &&
    witnesses.all (fun witness => witness.edge.START == node) &&
    outgoing.all (fun edge =>
      let edgeWitnesses := witnesses.filter (fun witness => witness.edge == edge)
      !edgeWitnesses.isEmpty &&
        setEqB (edgeWitnesses.map fun witness =>
            (witness.dependency, witness.residual))
          ((outputsAt c node).filter fun route =>
            headColour route.2 == edge.COLOUR)) &&
    witnesses.all (fun witness => outgoing.contains witness.edge)

private def edgeRecordAdjacentAtB (c : ExpandedCertificate) (slot : Nat) : Bool :=
  let table := edgeRecordIndex c
  match table.order[slot]?, table.order[slot + 1]? with
  | some leftIndex, some rightIndex =>
      match c.graph.EDGES[leftIndex]?, c.graph.EDGES[rightIndex]? with
      | some left, some right =>
          natListLE (edgeRecordKey c left) (edgeRecordKey c right) && !(left == right)
      | _, _ => false
  | _, _ => false

private def edgeSourceColourAdjacentAtB
    (c : ExpandedCertificate) (slot : Nat) : Bool :=
  let table := edgeSourceColourIndex c
  match table.order[slot]?, table.order[slot + 1]? with
  | some leftIndex, some rightIndex =>
      match c.graph.EDGES[leftIndex]?, c.graph.EDGES[rightIndex]? with
      | some left, some right =>
          let sameKey := left.START == right.START && left.COLOUR == right.COLOUR
          natListLE (edgeSourceColourKey left) (edgeSourceColourKey right) &&
            (!sameKey || left == right)
      | _, _ => false
  | _, _ => false

private def nodeLevelFormulaAdjacentAtB
    (c : ExpandedCertificate) (slot : Nat) : Bool :=
  let table := nodeLevelFormulaIndex c
  match table.order[slot]?, table.order[slot + 1]? with
  | some leftIndex, some rightIndex =>
      match c.graph.NODES[leftIndex]?, c.graph.NODES[rightIndex]? with
      | some left, some right =>
          let sameKey := left.LEVEL == right.LEVEL && left.FORMULA == right.FORMULA
          natListLE (nodeLevelFormulaKey c left) (nodeLevelFormulaKey c right) &&
            (!sameKey || left == right)
      | _, _ => false
  | _, _ => false

def indexTableJson (table : SortedIndexTable) : Json :=
  obj [("order", toJson table.order), ("inverse", toJson table.inverse)]

private def optionNatJson : Option Nat → Json
  | none => Json.null
  | some value => toJson value

def witnessIndexJson (c : ExpandedCertificate) : Json :=
  let table := witnessEdgeIndex c
  obj [("order", toJson table.order), ("inverse", toJson table.inverse),
    ("edge_locations", arr (c.graph.EDGES.map fun edge =>
      optionNatJson (edgeWitnessLocation c edge))),
    ("node_slices", arr (c.graph.NODES.map fun node =>
      let slice := nodeWitnessSlice c node
      obj [("node_id", toJson node.NUMBER), ("start", toJson slice.1),
        ("end", toJson slice.2)]))]

inductive Check where
  | edgesNodup
  | oneEdgePerColourPerNode
  | levelFormulaUnique
  | witnessIndexIntegrity
  | witnessCoverage
  | witnessFaithful
  | structuralEdge
  | structuralNode
  | hypothesesHaveNoIncoming
  | nodeProduction
  | headCoverage
  | tailClosure
  | rootDischarge
  deriving Repr, BEq

inductive Inspection where
  | sortedIndex (index : SortedIndexName) (mode : SortedIndexMode) (slot : Nat)
  | node (node : Vertex)
  | edge (id : Nat) (edge : Deduction)
  | witness (id : Nat) (witness : EdgeRouteWitness)
  | path (id : Nat) (path : Ancestral)
  deriving Repr

structure Position where
  check : Check
  inspection : Inspection
  deriving Repr

def Check.name : Check → String
  | .edgesNodup => "EDGES.Nodup"
  | .oneEdgePerColourPerNode => "OneEdgePerColourPerNode"
  | .levelFormulaUnique => "LevelFormulaUnique"
  | .witnessIndexIntegrity => "WitnessIndexIntegrity"
  | .witnessCoverage => "WitnessCoverage"
  | .witnessFaithful => "FaithfulDecoration"
  | .structuralEdge => "StructuralEdge"
  | .structuralNode => "StructuralNode"
  | .hypothesesHaveNoIncoming => "HypothesesHaveNoIncoming"
  | .nodeProduction => "RouteProduction"
  | .headCoverage => "RouteHeadCoverage"
  | .tailClosure => "FlowTailClosure"
  | .rootDischarge => "RootDischarge"

def Check.kind : Check → String
  | .nodeProduction => "node_production"
  | .witnessCoverage => "witness_coverage"
  | .witnessFaithful => "witness_faithful"
  | .headCoverage => "head_coverage"
  | .tailClosure => "tail_closure"
  | .rootDischarge => "root_discharge"
  | .edgesNodup | .oneEdgePerColourPerNode | .levelFormulaUnique
  | .witnessIndexIntegrity => "sorted_index"
  | .structuralEdge | .structuralNode
  | .hypothesesHaveNoIncoming => "structural"

private def sortedIndexPositions (check : Check) (index : SortedIndexName)
    (size : Nat) : List Position :=
  let adjacent := (List.range (size - 1)).map fun slot =>
    { check := check,
      inspection := Inspection.sortedIndex index .adjacent slot }
  let permutation := (List.range size).map fun slot =>
    { check := check,
      inspection := Inspection.sortedIndex index .permutation slot }
  adjacent ++ permutation

def positions (c : ExpandedCertificate) : List Position :=
  let sortedIndices :=
    sortedIndexPositions .edgesNodup .edgeRecord c.graph.EDGES.length ++
    sortedIndexPositions .oneEdgePerColourPerNode .edgeSourceColour
      c.graph.EDGES.length ++
    sortedIndexPositions .levelFormulaUnique .nodeLevelFormula c.graph.NODES.length
  let witnessIndex :=
    sortedIndexPositions .witnessIndexIntegrity .witnessEdge c.edgeRoutes.length
  let witnessCoverage := c.graph.EDGES.zipIdx.map fun (e, id) =>
    { check := .witnessCoverage, inspection := Inspection.edge id e }
  let structuralEdges := c.graph.EDGES.zipIdx.map fun (e, id) =>
    { check := .structuralEdge, inspection := Inspection.edge id e }
  let witnessFaithfulness := c.edgeRoutes.zipIdx.map fun (witness, id) =>
    { check := .witnessFaithful, inspection := Inspection.witness id witness }
  let nodes := c.graph.NODES.flatMap fun v =>
    [.mk .structuralNode (Inspection.node v),
     .mk .hypothesesHaveNoIncoming (Inspection.node v),
     .mk .nodeProduction (Inspection.node v),
     .mk .headCoverage (Inspection.node v),
     .mk .tailClosure (Inspection.node v),
     .mk .rootDischarge (Inspection.node v)]
  let allPositions :=
    sortedIndices ++ witnessIndex ++ witnessCoverage ++ structuralEdges ++
      witnessFaithfulness ++ nodes
  allPositions.take
    (6 * c.graph.EDGES.length + 8 * c.graph.NODES.length +
      3 * c.edgeRoutes.length - 4)

theorem positions_length (c : ExpandedCertificate) :
    (positions c).length =
      6 * c.graph.EDGES.length + 8 * c.graph.NODES.length +
        3 * c.edgeRoutes.length - 4 := by
  unfold positions
  simp [sortedIndexPositions]
  omega


def checkPositionB (c : ExpandedCertificate) (position : Position) : Bool :=
  match position.check, position.inspection with
  | .edgesNodup, Inspection.sortedIndex .edgeRecord .adjacent slot =>
      edgeRecordAdjacentAtB c slot
  | .edgesNodup, Inspection.sortedIndex .edgeRecord .permutation slot =>
      permutationAtB c.graph.EDGES.length slot (edgeRecordIndex c)
  | .oneEdgePerColourPerNode,
      Inspection.sortedIndex .edgeSourceColour .adjacent slot =>
      edgeSourceColourAdjacentAtB c slot
  | .oneEdgePerColourPerNode,
      Inspection.sortedIndex .edgeSourceColour .permutation slot =>
      permutationAtB c.graph.EDGES.length slot (edgeSourceColourIndex c)
  | .levelFormulaUnique,
      Inspection.sortedIndex .nodeLevelFormula .adjacent slot =>
      nodeLevelFormulaAdjacentAtB c slot
  | .levelFormulaUnique,
      Inspection.sortedIndex .nodeLevelFormula .permutation slot =>
      permutationAtB c.graph.NODES.length slot (nodeLevelFormulaIndex c)
  | .witnessIndexIntegrity,
      Inspection.sortedIndex .witnessEdge .adjacent slot =>
      witnessIndexAdjacentAtB c slot
  | .witnessIndexIntegrity,
      Inspection.sortedIndex .witnessEdge .permutation slot =>
      witnessIndexPermutationAtB c slot
  | .witnessCoverage, Inspection.edge _ e => indexedWitnessCoverageAtB c e
  | .witnessFaithful, Inspection.witness _ witness =>
      witnessFaithfulAtB c witness
  | .structuralEdge, Inspection.edge _ e => structuralEdgeAtB c.graph e
  | .structuralNode, Inspection.node v => structuralNodeAtB c.graph v
  | .hypothesesHaveNoIncoming, Inspection.node v => hypSourceAtB c.graph v
  | .nodeProduction, Inspection.node v => productionAtB c v
  | .headCoverage, Inspection.node v => indexedCoverageAtB c v
  | .tailClosure, Inspection.node v => tailClosureAtB c v
  | .rootDischarge, Inspection.node v => rootDischargeAtB c v
  | _, _ => false

def inspectionJson : Inspection → Json
  | Inspection.sortedIndex index mode slot =>
      obj [("type", toJson "sorted_index"), ("index_name", toJson index.name),
        ("mode", toJson mode.name), ("slot", toJson slot)]
  | Inspection.node v => obj [("type", toJson "node"), ("node_id", toJson v.NUMBER)]
  | Inspection.edge id _ => obj [("type", toJson "edge"), ("edge_id", toJson id)]
  | Inspection.witness id witness =>
      obj [("type", toJson "edge_witness"), ("witness_id", toJson id),
        ("source", toJson witness.edge.START.NUMBER),
        ("target", toJson witness.edge.END.NUMBER)]
  | Inspection.path id p => obj [("type", toJson "ancestral_path"),
      ("path_id", toJson id), ("source", toJson p.START.NUMBER),
      ("target", toJson p.END.NUMBER)]

def positionJson (index : Nat) (position : Position) : Json :=
  obj [("index", toJson index), ("kind", toJson position.check.kind),
    ("check", toJson position.check.name),
    ("inspection", inspectionJson position.inspection)]

def resultJson (c : ExpandedCertificate) (index : Nat) (position : Position) : Json :=
  obj [("index", toJson index), ("outcome", toJson (checkPositionB c position))]

def certificateJson (name : String) (c : ExpandedCertificate) : Json :=
  let formulas := buildFormulas c.graph
  let ps := positions c
  obj [("schema_version", toJson "1.6.0"), ("fixture", toJson name),
    ("formula_universe", arr (formulas.map formulaJson)),
    ("nodes", arr (c.graph.NODES.map (nodeJson c.graph))),
    ("edges", arr (c.graph.EDGES.zipIdx.map fun (e, id) =>
      edgeJson c formulas id e)),
    ("ancestral_paths", arr (c.graph.PATHS.zipIdx.map fun (p, id) =>
      ancestralPathJson id p)),
    ("edge_route_witnesses", arr (c.edgeRoutes.map (witnessJson formulas))),
    ("terminal_records", arr (c.terminalOutputs.map (terminalJson formulas))),
    ("sorted_indices", obj [
      (SortedIndexName.edgeRecord.name, indexTableJson (edgeRecordIndex c)),
      (SortedIndexName.edgeSourceColour.name,
        indexTableJson (edgeSourceColourIndex c)),
      (SortedIndexName.nodeLevelFormula.name,
        indexTableJson (nodeLevelFormulaIndex c))]),
    ("witness_index", witnessIndexJson c),
    ("positions", arr (ps.zipIdx.map fun (position, index) =>
      positionJson index position)),
    ("results", arr (ps.zipIdx.map fun (position, index) =>
      resultJson c index position))]

private def typeSchema (name : String) : Json := obj [("type", toJson name)]
private def refSchema (name : String) : Json := obj [("$ref", toJson s!"#/$defs/{name}")]
private def arraySchema (items : Json) : Json :=
  obj [("type", toJson "array"), ("items", items)]
private def natSchema : Json :=
  obj [("type", toJson "integer"), ("minimum", toJson 0)]

def schemaJson : Json :=
  obj [("$schema", toJson "https://json-schema.org/draft/2020-12/schema"),
    ("$id", toJson "https://dlds.example/schema/expanded-certificate-1.6.0.json"),
    ("title", toJson "DLDS expanded certificate export"),
    ("type", toJson "object"),
    ("required", toJson ["schema_version", "fixture", "formula_universe", "nodes",
      "edges", "ancestral_paths", "edge_route_witnesses", "terminal_records",
      "sorted_indices", "witness_index", "positions", "results"]),
    ("properties", obj [
      ("schema_version", obj [("const", toJson "1.6.0")]),
      ("fixture", typeSchema "string"),
      ("formula_universe", arraySchema (refSchema "formula")),
      ("nodes", arraySchema (refSchema "node")),
      ("edges", arraySchema (refSchema "edge")),
      ("ancestral_paths", arraySchema (refSchema "ancestral_path")),
      ("edge_route_witnesses", arraySchema (refSchema "edge_route_witness")),
      ("terminal_records", arraySchema (refSchema "terminal")),
      ("sorted_indices", obj [("type", toJson "object"),
        ("required", toJson ["edge_record", "edge_source_colour",
          "node_level_formula"]),
        ("properties", obj [
          ("edge_record", refSchema "sorted_index_table"),
          ("edge_source_colour", refSchema "sorted_index_table"),
          ("node_level_formula", refSchema "sorted_index_table")]),
        ("additionalProperties", toJson false)]),
      ("witness_index", refSchema "witness_index"),
      ("positions", obj [("type", toJson "array"),
        ("description", toJson "Ordered local-oracle positions; index is contiguous from zero."),
        ("items", obj [("$ref", toJson "#/$defs/position")])]),
      ("results", obj [("type", toJson "array"),
        ("description", toJson "Probe Boolean outcomes aligned one-to-one with positions; true means the local check passes."),
        ("items", obj [("$ref", toJson "#/$defs/result")])])]),
    ("$defs", obj [
      ("formula", obj [("description", toJson "Recursive structured formula."),
        ("oneOf", arr [
          obj [("type", toJson "object"), ("required", toJson ["kind", "symbol"]),
            ("properties", obj [("kind", obj [("const", toJson "atom")]),
              ("symbol", typeSchema "string")]), ("additionalProperties", toJson false)],
          obj [("type", toJson "object"),
            ("required", toJson ["kind", "antecedent", "consequent"]),
            ("properties", obj [("kind", obj [("const", toJson "implication")]),
              ("antecedent", refSchema "formula"),
              ("consequent", refSchema "formula")]),
            ("additionalProperties", toJson false)]])]),
      ("node_type", obj [("type", toJson "object"), ("required", toJson ["kind"]),
        ("properties", obj [
          ("kind", obj [("enum", toJson ["source", "elimination", "introduction", "invalid"])]),
          ("hypothesis", typeSchema "boolean"), ("collapsed", typeSchema "boolean"),
          ("major_premise", natSchema), ("minor_premise", natSchema),
          ("premise", natSchema), ("antecedent", refSchema "formula"),
          ("consequent", refSchema "formula")]),
        ("additionalProperties", toJson false)]),
      ("node", obj [("type", toJson "object"),
        ("required", toJson ["id", "formula", "level", "hypothesis", "collapsed",
          "past", "node_type"]),
        ("properties", obj [("id", natSchema), ("formula", refSchema "formula"),
          ("level", natSchema), ("hypothesis", typeSchema "boolean"),
          ("collapsed", typeSchema "boolean"), ("past", arraySchema natSchema),
          ("node_type", refSchema "node_type")]),
        ("additionalProperties", toJson false)]),
      ("vertex_value", obj [("type", toJson "object"),
        ("required", toJson ["id", "formula", "level", "hypothesis", "collapsed",
          "past"]),
        ("properties", obj [("id", natSchema), ("formula", refSchema "formula"),
          ("level", natSchema), ("hypothesis", typeSchema "boolean"),
          ("collapsed", typeSchema "boolean"), ("past", arraySchema natSchema)]),
        ("additionalProperties", toJson false)]),
      ("dependency", arraySchema (refSchema "formula")),
      ("bitvector", arraySchema (typeSchema "boolean")),
      ("residual", arraySchema natSchema),
      ("edge", obj [("type", toJson "object"),
        ("required", toJson ["id", "source", "target", "colour",
          "dependency", "dependency_bitvector", "witness_routes", "witness_count"]),
        ("properties", obj [("id", natSchema), ("source", natSchema),
          ("target", natSchema), ("colour", natSchema),
          ("dependency", refSchema "dependency"),
          ("dependency_bitvector", refSchema "bitvector"),
          ("witness_routes", arraySchema (refSchema "route_witness")),
          ("witness_count", natSchema)]),
        ("description", toJson "All route-local witnesses assigned to this edge."),
        ("additionalProperties", toJson false)]),
      ("edge_value", obj [("type", toJson "object"),
        ("required", toJson ["source", "target", "colour", "dependency",
          "dependency_bitvector"]),
        ("properties", obj [("source", natSchema), ("target", natSchema),
          ("colour", natSchema), ("dependency", refSchema "dependency"),
          ("dependency_bitvector", refSchema "bitvector")]),
        ("additionalProperties", toJson false)]),
      ("edge_route_witness", obj [("type", toJson "object"),
        ("required", toJson ["edge", "dependency", "dependency_bitvector", "residual"]),
        ("properties", obj [("edge", refSchema "edge_value"),
          ("dependency", refSchema "dependency"),
          ("dependency_bitvector", refSchema "bitvector"),
          ("residual", refSchema "residual")]),
        ("additionalProperties", toJson false)]),
      ("route_witness", obj [("type", toJson "object"),
        ("required", toJson ["dependency", "dependency_bitvector", "residual"]),
        ("properties", obj [("dependency", refSchema "dependency"),
          ("dependency_bitvector", refSchema "bitvector"),
          ("residual", refSchema "residual")]),
        ("additionalProperties", toJson false)]),
      ("ancestral_path", obj [("type", toJson "object"),
        ("required", toJson ["id", "source", "target", "source_vertex",
          "target_vertex", "colours"]),
        ("properties", obj [("id", natSchema), ("source", natSchema),
          ("target", natSchema), ("source_vertex", refSchema "vertex_value"),
          ("target_vertex", refSchema "vertex_value"),
          ("colours", refSchema "residual")]),
        ("additionalProperties", toJson false)]),
      ("terminal", obj [("type", toJson "object"),
        ("required", toJson ["node_id", "dependency", "dependency_bitvector", "residual"]),
        ("properties", obj [("node_id", natSchema),
          ("dependency", refSchema "dependency"),
          ("dependency_bitvector", refSchema "bitvector"),
          ("residual", refSchema "residual")]),
        ("additionalProperties", toJson false)]),
      ("sorted_index_table", obj [("type", toJson "object"),
        ("required", toJson ["order", "inverse"]),
        ("properties", obj [("order", arraySchema natSchema),
          ("inverse", arraySchema natSchema)]),
        ("description", toJson "Dense forward record-offset permutation and its inverse."),
        ("additionalProperties", toJson false)]),
      ("node_witness_slice", obj [("type", toJson "object"),
        ("required", toJson ["node_id", "start", "end"]),
        ("properties", obj [("node_id", natSchema), ("start", natSchema),
          ("end", natSchema)]),
        ("additionalProperties", toJson false)]),
      ("witness_index", obj [("type", toJson "object"),
        ("required", toJson ["order", "inverse", "edge_locations", "node_slices"]),
        ("properties", obj [
          ("order", arraySchema natSchema), ("inverse", arraySchema natSchema),
          ("edge_locations", arraySchema (obj [("oneOf", arr [natSchema,
            typeSchema "null"])])),
          ("node_slices", arraySchema (refSchema "node_witness_slice"))]),
        ("additionalProperties", toJson false)]),
      ("inspection", obj [("oneOf", arr [
        obj [("type", toJson "object"),
          ("required", toJson ["type", "index_name", "mode", "slot"]),
          ("properties", obj [
            ("type", obj [("const", toJson "sorted_index")]),
            ("index_name", obj [("enum", toJson ["edge_record",
              "edge_source_colour", "node_level_formula", "witness_edge"])]),
            ("mode", obj [("enum", toJson ["adjacent", "permutation"])]),
            ("slot", natSchema)]),
          ("additionalProperties", toJson false)],
        obj [("type", toJson "object"), ("required", toJson ["type", "node_id"]),
          ("properties", obj [("type", obj [("const", toJson "node")]),
            ("node_id", natSchema)])],
        obj [("type", toJson "object"), ("required", toJson ["type", "edge_id"]),
          ("properties", obj [("type", obj [("const", toJson "edge")]),
            ("edge_id", natSchema)])],
        obj [("type", toJson "object"),
          ("required", toJson ["type", "witness_id", "source", "target"]),
          ("properties", obj [
            ("type", obj [("const", toJson "edge_witness")]),
            ("witness_id", natSchema), ("source", natSchema),
            ("target", natSchema)])],
        obj [("type", toJson "object"),
          ("required", toJson ["type", "path_id", "source", "target"]),
          ("properties", obj [("type", obj [("const", toJson "ancestral_path")]),
            ("path_id", natSchema), ("source", natSchema), ("target", natSchema)])]])]),
      ("position", obj [("type", toJson "object"),
        ("required", toJson ["index", "kind", "check", "inspection"]),
        ("properties", obj [("index", natSchema), ("kind", obj [("enum", toJson
          ["node_production", "witness_coverage", "head_coverage",
           "witness_faithful", "tail_closure", "root_discharge",
           "structural", "sorted_index"])]),
          ("check", typeSchema "string"), ("inspection", refSchema "inspection")]),
        ("additionalProperties", toJson false)]),
      ("result", obj [("type", toJson "object"),
        ("required", toJson ["index", "outcome"]),
        ("properties", obj [("index", natSchema),
          ("outcome", typeSchema "boolean")]),
        ("additionalProperties", toJson false)])]),
    ("additionalProperties", toJson false)]

def fixtures : List (String × ExpandedCertificate) :=
  [("3v-no-hc", generateCertificate Semantic.ThreeVertexCompressedExample.graph),
   ("endtoend-valid", eCertificate),
   ("d11-valid", d11Certificate),
   ("endtoend-altered-residual", alterFirstResidual eCertificate),
   ("endtoend-omitted-witness", omitFirstWitness eCertificate),
   ("endtoend-unfaithful-decoration", corruptDependency eCertificate),
   ("endtoend-nonempty-terminal", corruptTerminal eCertificate)]

def writeJson (path : System.FilePath) (value : Json) : IO Unit :=
  IO.FS.writeFile path (value.pretty 120 ++ "\n")

def exportAll (outputDir : System.FilePath) : IO Unit := do
  IO.FS.createDirAll outputDir
  writeJson (outputDir / "schema.json") schemaJson
  for (name, certificate) in fixtures do
    writeJson (outputDir / s!"{name}.json") (certificateJson name certificate)

def usage : String :=
  "usage: lake exe certificateExporter [--] <output-directory>"

def main (args : List String) : IO UInt32 := do
  match args with
  | [outputDir] | ["--", outputDir] =>
      exportAll (System.FilePath.mk outputDir)
      IO.println s!"wrote schema.json and {fixtures.length} fixture files to {outputDir}"
      return 0
  | _ =>
      IO.eprintln usage
      return 2

end CertificateExporter

def certificateExporterMain := CertificateExporter.main
