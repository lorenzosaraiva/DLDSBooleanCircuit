import Foundations.Graph







namespace RoutedEliminationTransport

def broadUse? (nodes : List Vertex) (before : Graph)
    (rule : Neighborhood) (after : Graph) (owner : Vertex) : Option Ancestral := do
  if !nodes.contains owner || !before.NODES.contains owner ||
      owner.COLLAPSED || owner.HYPOTHESIS ||
      (before.NODES.filter (fun v => v.NUMBER == owner.NUMBER)).length != 1 then
    none
  else
    match get_rule.outgoing owner before with
    | [originalUse] =>
        if rule.OUTGOING.any (fun use =>
            use.START == rule.CENTER && use.END == originalUse.END &&
              use.COLOUR == owner.NUMBER) &&
            !(after.PATHS.any (fun record =>
              record.END == rule.CENTER &&
                record.COLOURS.head? == some owner.NUMBER)) then
          some (path originalUse.END rule.CENTER [owner.NUMBER])
        else none
    | _ => none

def broadStep (partition : List HCClass) (nodes : List Vertex)
    (before : Graph) (occurrences : List Ancestral) :
    Graph × List Ancestral := Id.run do
  let result := compress_ruleWithOccurrences partition nodes before
  let rule := result.1
  let materialized := materializeCarriedOwners nodes before occurrences
  let rewritten := materialized.map
    (collapse.rewrite_path_endpoints nodes rule.CENTER)
  let allOccurrences := rewritten ++ result.2
  let replaced := collapse.rewrite_graph_path_endpoints nodes rule.CENTER
    (set_rule (pop_rules before (get_rules nodes before)) rule)
  let after := dlds replaced.NODES replaced.EDGES
    (ensureOccurrenceRecords replaced.PATHS allOccurrences)
  let added := nodes.filterMap (broadUse? nodes before rule after)
  return (checkAncestralPathHeads
      (dlds after.NODES after.EDGES (after.PATHS ++ added)),
    allOccurrences ++ added)

def broadCompress (before : Graph) : Graph := Id.run do
  let partition := hcPartition before
  let schedule := (repeated_nodes_vertex_indexedWithPartition partition before).flatten
  let mut current := before
  let mut occurrences : List Ancestral := []
  for nodes in schedule do
    let step := broadStep partition nodes current occurrences
    current := step.1
    occurrences := step.2
  return current










def stableUse? (later : List (List Vertex)) (nodes : List Vertex)
    (before : Graph) (rule : Neighborhood) (after : Graph)
    (owner : Vertex) : Option Ancestral := do
  let originalUse ← (get_rule.outgoing owner before).head?
  if !(after.PATHS.any (fun record => record.END == rule.CENTER)) ||
      originalUse.END.COLLAPSED ||
      (later.flatten).contains originalUse.END then none
  else broadUse? nodes before rule after owner

def selectiveStep (partition : List HCClass) (later : List (List Vertex))
    (nodes : List Vertex) (before : Graph)
    (occurrences : List Ancestral) : Graph × List Ancestral := Id.run do
  let result := compress_ruleWithOccurrences partition nodes before
  let rule := result.1
  let materialized := materializeCarriedOwners nodes before occurrences
  let rewritten := materialized.map
    (collapse.rewrite_path_endpoints nodes rule.CENTER)
  let allOccurrences := rewritten ++ result.2
  let replaced := collapse.rewrite_graph_path_endpoints nodes rule.CENTER
    (set_rule (pop_rules before (get_rules nodes before)) rule)
  let after := dlds replaced.NODES replaced.EDGES
    (ensureOccurrenceRecords replaced.PATHS allOccurrences)
  let added := nodes.filterMap (stableUse? later nodes before rule after)
  return (checkAncestralPathHeads
    (dlds after.NODES after.EDGES (after.PATHS ++ added)), allOccurrences)

def selectiveCompress (before : Graph) : Graph := Id.run do
  let partition := hcPartition before
  let mut schedule :=
    (repeated_nodes_vertex_indexedWithPartition partition before).flatten
  let mut current := before
  let mut occurrences : List Ancestral := []
  while !schedule.isEmpty do
    match schedule with
    | [] => pure ()
    | nodes :: later =>
        let step := selectiveStep partition later nodes current occurrences
        current := step.1
        occurrences := step.2
        schedule := later
  return current

end RoutedEliminationTransport
