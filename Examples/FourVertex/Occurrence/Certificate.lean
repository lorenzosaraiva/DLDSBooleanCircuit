import NonHamiltonicity.Certificate.Input
import Lean.Data.Json

/-! Lossless serialization of supplied data, with full vertex endpoint records.
No proof-body value, generator, replay, or acceptance decision is used here. -/
open Lean FlowSemantics Certificate.Occurrence Certificate.Occurrence.Positions
open NonHamiltonicity.Context NonHamiltonicity.Certificate.Input

namespace Examples.FourVertex.Occurrence.Certificate
def arrayJson (xs : List Json) : Json := .arr xs.toArray
def formulaJson : Formula -> Json
  | .atom symbol => Json.mkObj [("kind", toJson "atom"), ("symbol", toJson symbol)]
  | .implication a b => Json.mkObj [("kind", toJson "implication"),
      ("antecedent", formulaJson a), ("consequent", formulaJson b)]
def vertexJson (v : Vertex) : Json := Json.mkObj [
  ("id", toJson v.NUMBER), ("level", toJson v.LEVEL), ("formula", formulaJson v.FORMULA),
  ("hypothesis", toJson v.HYPOTHESIS), ("collapsed", toJson v.COLLAPSED), ("past", toJson v.PAST)]
def graphJson (d : _root_.Graph) : Json := Json.mkObj [
  ("nodes", arrayJson (d.NODES.map vertexJson)),
  ("edges", arrayJson (d.EDGES.map fun e => Json.mkObj [
    ("start", vertexJson e.START), ("end", vertexJson e.END), ("colour", toJson e.COLOUR),
    ("stored_dependency_crosscheck_only", arrayJson (e.DEPENDENCY.map formulaJson))])),
  ("paths", arrayJson (d.PATHS.map fun p => Json.mkObj [
    ("start", vertexJson p.START), ("end", vertexJson p.END), ("colours", toJson p.COLOURS)]))]
def routeJson (route : FlowPair) : Json := Json.mkObj [
  ("dependencies", arrayJson (route.1.map formulaJson)), ("residual", toJson route.2)]
def certificateJson (certificate : SuccessorCertificate) : Json := Json.mkObj [
  ("positions", arrayJson (certificate.positions.map fun position => Json.mkObj [
    ("node", vertexJson position.node),
    ("premise_table", arrayJson (position.premiseTable.map fun (vertex, routes) => Json.mkObj [
      ("node", vertexJson vertex), ("routes", arrayJson (routes.map routeJson))])),
    ("claimed_routes", arrayJson (position.claimedRoutes.map routeJson))]))]
def sourceGraphJson (data : SourceGraphData) : Json := Json.mkObj [
  ("n", toJson data.n), ("vertices", toJson data.vertices),
  ("edges", arrayJson (data.edges.map fun (u,v) => toJson [u,v])), ("undirected", toJson data.undirected)]
def witnessJson : SchemaWitness -> Json
  | .totality k => Json.mkObj [("kind", toJson "totality"), ("k", toJson k)]
  | .cases k vs => Json.mkObj [("kind", toJson "cases"), ("k", toJson k), ("vertices", toJson vs)]
  | .repeatVertex i j v => Json.mkObj [("kind", toJson "repeat_vertex"),
      ("i", toJson i), ("j", toJson j), ("v", toJson v)]
  | .nonedge i u v flip => Json.mkObj [("kind", toJson "nonedge"),
      ("i", toJson i), ("u", toJson u), ("v", toJson v), ("flipped", toJson flip)]
  | .closingAlias v => Json.mkObj [("kind", toJson "closing_alias"), ("v", toJson v)]
  | .invalid => Json.mkObj [("kind", toJson "invalid")]
def requestJson (request : Request) : Json := Json.mkObj [
  ("schema", toJson request.schemaVersion), ("contract", toJson request.contractVersion),
  ("source_graph", sourceGraphJson request.sourceData),
  ("context", arrayJson (request.context.map formulaJson)),
  ("schema_witnesses", arrayJson (request.witnesses.map witnessJson)),
  ("proof_graph", graphJson request.proofGraph), ("certificate", certificateJson request.certificate),
  ("root_selector", Json.mkObj [("kind", toJson "occurrence_root_index"), ("index", toJson request.rootIndex)])]
def checksJson (request : Request) : Json := Json.mkObj [
  ("source_graph", toJson (graphCheckB request.sourceData)),
  ("context", toJson (contextCheckB request.sourceData request.context request.witnesses)),
  ("production_domain", toJson (actualDomainB request.proofGraph)),
  ("root_target", toJson (rootTargetB request.proofGraph request.context request.rootIndex)),
  ("formation", toJson (successorFormationB request.proofGraph request.certificate)),
  ("accepted", toJson (requestCheckB request)),
  ("semantic_positions", toJson (semanticPositions request.proofGraph request.certificate).length)]
end Examples.FourVertex.Occurrence.Certificate
