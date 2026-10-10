import Robustness.JustifiedRoutes

/-! Direct fixed-index view of the existing expanded certificate records.
Only record scope is required; no missing-record coverage is used. -/
open DLDS Flow Certificate.Flow.Foundation
namespace Robustness

def expandedRows (c : ExpandedCertificate) : List (Vertex × FlowPair) :=
  c.edgeRoutes.map (fun w => (w.edge.START, (w.dependency, w.residual))) ++
  c.terminalOutputs.map (fun t => (t.node, (t.dependency, t.residual)))

def addressedExpanded (c : ExpandedCertificate)
    (ownersListed : ∀ row ∈ expandedRows c, row.1 ∈ c.graph.NODES)
    (kind : Fin (expandedRows c).length → ProducerKind)
    (premise : Fin (expandedRows c).length → Fin 2 →
      Option (Fin (expandedRows c).length)) :
    JustifiedCertificate c.graph (expandedRows c).length where
  owner slot := ((expandedRows c).get slot).1
  ownerListed slot := ownersListed _ (List.get_mem _ slot)
  route slot := ((expandedRows c).get slot).2
  kind := kind
  premise := premise

private theorem stored_mem_row (c : ExpandedCertificate) (v : Vertex)
    (route : FlowPair) (member : route ∈ Certificate.Flow.StoredRoutes c v) :
    (v, route) ∈ expandedRows c := by
  simp only [Certificate.Flow.StoredRoutes, outputsAt, List.mem_eraseDups,
    List.mem_append] at member
  rcases member with edge | terminal
  · obtain ⟨witness, hw, value⟩ := List.mem_filterMap.mp edge
    by_cases owner : witness.edge.START = v
    · simp [owner] at value
      exact List.mem_append.mpr (Or.inl (List.mem_map.mpr
        ⟨witness, hw, by simp [owner, value]⟩))
    · simp [owner] at value
  · obtain ⟨output, ho, value⟩ := List.mem_filterMap.mp terminal
    by_cases owner : output.node = v
    · simp [owner] at value
      exact List.mem_append.mpr (Or.inr (List.mem_map.mpr
        ⟨output, ho, by simp [owner, value]⟩))
    · simp [owner] at value

/-- Positive inclusion for the existing StoredRoutes accessor.
Every hypothesis is explicit: stored-owner scope, graph structure, and checks
of the supplied direct provenance. Coverage and LocalChecks are not inputs. -/
theorem expanded_stored_subset_flow (c : ExpandedCertificate)
    (ownersListed : ∀ row ∈ expandedRows c, row.1 ∈ c.graph.NODES)
    (kind : Fin (expandedRows c).length → ProducerKind)
    (premise : Fin (expandedRows c).length → Fin 2 →
      Option (Fin (expandedRows c).length))
    (structural : structuralValid c.graph)
    (checked : ∀ slot, justificationB (addressedExpanded c ownersListed kind premise) slot = true) :
    ∀ v route, route ∈ Certificate.Flow.StoredRoutes c v →
      route ∈ WitnessableRoutes c.graph v (flowAt c.graph (stdFuel c.graph) v) := by
  intro v route member
  obtain ⟨index, bound, rowEq⟩ := List.getElem_of_mem (stored_mem_row c v route member)
  let slot : Fin (expandedRows c).length := ⟨index, bound⟩
  have result := justification_sound (addressedExpanded c ownersListed kind premise)
    structural checked slot
  have eq : (expandedRows c).get slot = (v, route) := rowEq
  simpa only [addressedExpanded, eq] using result
end Robustness
