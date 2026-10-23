import Infrastructure.OccurrenceReadingsForward
import Infrastructure.OccurrenceReadingsLineageReverse
import Infrastructure.OccurrenceReadingsLineageND

open Semantic FlowSpec SelfCertifying RouteIndexedMultiToken FlowBranchingGridBridge SpineReadings
namespace OccurrenceReadings



theorem occurrence_keyed_modulo_slots (d : Graph) (v : Vertex)
    (tree : ReadingTree) (checked : occurrenceAtCompleteB d v tree = true) :
    ∃ route fuel token, readingPayloadComplete? d tree = some route ∧
      token ∈ TaggedLineage.evaluateAt d fuel v ∧ token.flowPair = route ∧
      stripSlots token.routeKey = stripSlots tree := by
  obtain ⟨route, payload⟩ := occurrence_payload d v tree checked
  obtain ⟨fuel, token, member, pair, lineage⟩ :=
    readingPayloadComplete_keyed_modulo_slots d tree route payload
  have atVertex : vertex tree = v := by
    simp only [occurrenceAtCompleteB, Bool.and_eq_true] at checked
    simpa using checked.1.1
  exact ⟨route, fuel, token, payload, atVertex ▸ member, pair, lineage⟩



theorem keyed_occurrence (d : Graph) (fuel : Nat) (v : Vertex)
    (token : RouteIndexedToken) (member : token ∈ TaggedLineage.evaluateAt d fuel v) :
    occurrenceAtCompleteB d v token.routeKey = true ∧
      readingPayloadComplete? d token.routeKey = some token.flowPair := by
  obtain ⟨atVertex, payload⟩ :=
    OccurrenceReadingsLineageReverse.evaluateAt_payload d fuel v token member
  obtain ⟨nd, reconstructed, wellFormed, conclusion, deps⟩ :=
    TaggedLineage.routeOccurrence_tree_dependencies d fuel v token member
  change routeKeyTree? token.routeKey = some nd at reconstructed
  refine ⟨?_, payload⟩
  simp [occurrenceAtCompleteB, ndCorrectB, payload, reconstructed, atVertex, wellFormed, conclusion]


theorem occurrence_route_iff_keyed (d : Graph) (v : Vertex) (route : FlowPair) :
    (∃ tree, occurrenceAtCompleteB d v tree = true ∧ readingPayloadComplete? d tree = some route) ↔
    (∃ fuel token, token ∈ TaggedLineage.evaluateAt d fuel v ∧ token.flowPair = route) := by
  constructor
  · rintro ⟨tree, checked, payload⟩
    obtain ⟨actual, fuel, token, payload', member, pair, lineage⟩ :=
      occurrence_keyed_modulo_slots d v tree checked
    have equal : actual = route := Option.some.inj (payload'.symm.trans payload)
    exact ⟨fuel, token, member, pair.trans equal⟩
  · rintro ⟨fuel, token, member, pair⟩
    obtain ⟨checked, payload⟩ := keyed_occurrence d fuel v token member
    exact ⟨token.routeKey, checked, pair ▸ payload⟩

#print axioms occurrence_keyed_modulo_slots
#print axioms keyed_occurrence
#print axioms occurrence_route_iff_keyed
end OccurrenceReadings
