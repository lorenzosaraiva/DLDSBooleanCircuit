import Semantic.SpineReadingRouteSoundness

open Semantic FlowSpec SelfCertifying RouteIndexedMultiToken

namespace SpineReadings




theorem readingPayload_has_keyed_route (d : Graph) (tree : ReadingTree)
    (route : FlowPair) (checked : readingPayload? d tree = some route) :
    ∃ fuel token, token ∈ SpineLineage.evaluateAt d fuel (vertex tree) ∧
      token.flowPair = route := by
  obtain ⟨fuel, member⟩ := readingPayload_mem_guardFree d tree route checked
  have projected := (SpineLineage.evaluateAt_project d fuel (vertex tree) route).mpr member
  obtain ⟨token, tokenMember, projection⟩ := List.mem_map.mp projected
  exact ⟨fuel, token, tokenMember, projection⟩

#print axioms readingPayload_has_keyed_route
end SpineReadings
