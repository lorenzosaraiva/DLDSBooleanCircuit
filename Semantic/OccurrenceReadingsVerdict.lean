import Semantic.OccurrenceReadingsCorrespondence
import Semantic.OccurrenceReadingsStability





open Semantic FlowSpec SelfCertifying RouteIndexedMultiToken SpineReadings

namespace OccurrenceReadings



theorem occurrence_route_iff_reference (d : Graph) (structural : structuralValid d)
    (v : Vertex) (node : v ∈ d.NODES) (route : FlowPair) :
    (∃ tree, occurrenceAtCompleteB d v tree = true ∧ readingPayloadComplete? d tree = some route) ↔
      route ∈ SingleIntroductionReference.routes d (stdFuel d) v := by
  constructor
  · intro reading
    obtain ⟨fuel, token, member, payload⟩ := (occurrence_route_iff_keyed d v route).mp reading
    have projected : route ∈ projectFlowPairs (TaggedLineage.evaluateAt d fuel v) :=
      List.mem_map.mpr ⟨token, member, payload⟩
    exact routes_mem_at_standard d structural node fuel route
      ((TaggedLineage.evaluateAt_project d fuel v route).mp projected)
  · intro member
    obtain ⟨token, present, payload⟩ := List.mem_map.mp
      ((TaggedLineage.evaluateAt_project d (stdFuel d) v route).mpr member)
    exact (occurrence_route_iff_keyed d v route).mpr ⟨stdFuel d, token, present, payload⟩



theorem ReferenceVerdict_occurrenceVerdictComplete (d : Graph) (valid : SingleIntroductionReference.ReferenceVerdict d) :
    occurrenceVerdictComplete d := by
  have roots := (SingleIntroductionReference.rootsB_iff d).mp valid.2.2
  constructor
  · intro root node terminal
    obtain ⟨route, present⟩ := List.exists_mem_of_ne_nil _ (roots root node terminal).1
    obtain ⟨tree, accepted, _⟩ :=
      (occurrence_route_iff_reference d valid.1 root node route).mpr present
    exact ⟨tree, accepted⟩
  · intro root node terminal tree route accepted payload
    exact (roots root node terminal).2 route
      ((occurrence_route_iff_reference d valid.1 root node route).mp
        ⟨tree, accepted, payload⟩)



theorem occurrenceVerdictComplete_rootsB (d : Graph) (structural : structuralValid d)
    (verdict : occurrenceVerdictComplete d) : SingleIntroductionReference.rootsB d = true := by
  apply (SingleIntroductionReference.rootsB_iff d).mpr
  intro root node terminal
  obtain ⟨tree, accepted⟩ := verdict.1 root node terminal
  obtain ⟨route, payload⟩ := occurrence_payload d root tree accepted
  have member := (occurrence_route_iff_reference d structural root node route).mp
    ⟨tree, accepted, payload⟩
  constructor
  · intro empty
    rw [empty] at member
    contradiction
  · intro raw present
    obtain ⟨reading, checked, pair⟩ :=
      (occurrence_route_iff_reference d structural root node raw).mpr present
    exact verdict.2 root node terminal reading raw checked pair



theorem ReferenceVerdict_iff_domain_occurrenceVerdictComplete (d : Graph) :
    SingleIntroductionReference.ReferenceVerdict d ↔
      structuralValid d ∧ LeveledColored d ∧ occurrenceVerdictComplete d := by
  constructor
  · intro valid
    exact ⟨valid.1, valid.2.1, ReferenceVerdict_occurrenceVerdictComplete d valid⟩
  · rintro ⟨structural, levels, verdict⟩
    exact ⟨structural, levels, occurrenceVerdictComplete_rootsB d structural verdict⟩

#print axioms occurrence_route_iff_reference
#print axioms ReferenceVerdict_iff_domain_occurrenceVerdictComplete
end OccurrenceReadings
