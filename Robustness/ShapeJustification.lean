import Robustness.ExpandedRoutes

/-! Positive route inclusion for fixed-length payload assignments.
Scope is explicit stored-owner membership, not coverage or route validity. -/
open DLDS FlowSemantics Certificate.Flow.Foundation CertificateSearch.Robustness
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
namespace Robustness

structure ShapeFixedJustifiedCertificate (d : Graph) (anchor : ExpandedCertificate) where
  input : ShapeFixedAssignment d anchor
  ownersListed : ∀ row ∈ expandedRows (decodeShapeFixed d anchor input),
    row.1 ∈ (decodeShapeFixed d anchor input).graph.NODES
  kind : Fin (expandedRows (decodeShapeFixed d anchor input)).length → ProducerKind
  premise : Fin (expandedRows (decodeShapeFixed d anchor input)).length → Fin 2 →
    Option (Fin (expandedRows (decodeShapeFixed d anchor input)).length)

def ShapeFixedJustifiedCertificate.addressed {d : Graph} {anchor : ExpandedCertificate}
    (c : ShapeFixedJustifiedCertificate d anchor) :=
  addressedExpanded (decodeShapeFixed d anchor c.input) c.ownersListed c.kind c.premise

/-- Shape-fixed positive inclusion, using the existing verified decoding.
The record includes only a payload assignment, owner scope and direct provenance. -/
theorem shapeFixed_stored_subset_flow (d : Graph) (anchor : ExpandedCertificate)
    (c : ShapeFixedJustifiedCertificate d anchor)
    (structural : structuralValid anchor.graph)
    (checked : ∀ slot, justificationB c.addressed slot = true) :
    ∀ v route, route ∈ Certificate.Flow.StoredRoutes (decodeShapeFixed d anchor c.input) v →
      route ∈ WitnessableRoutes anchor.graph v (flowAt anchor.graph (stdFuel anchor.graph) v) := by
  have result := expanded_stored_subset_flow (decodeShapeFixed d anchor c.input)
    c.ownersListed c.kind c.premise (by simpa using structural) checked
  simpa using result
end Robustness
