import DLDS.Compression.Mixed.ContextualElimination

/-!
# Transport lemmas for the selected HC edge-trace convention

Horizontal-compression runtime addresses in the successor model are
`EdgeColourTrace`s: one colour is present for every traversed deduction edge.
This module records the operational consequences needed by later occurrence
semantics.  It does not identify edge traces with branch addresses.  Their
only established relationship remains the endpoint-relative path theorem in
`LocalSpecificationCandidate`.
-/

open DLDS FlowSemantics DLDS.Compression.Mixed.LocalRules

namespace DLDS.Compression.EdgeTraces

def residualOfEdgeTrace (trace : EdgeColourTrace) : CandidateResidual :=
  ⟨trace.colours⟩

/-- Candidate residual consumption is exactly historical/default edge-colour
consumption after projecting the wrapper. -/
theorem consumeCandidate_colours (edge : Deduction) (residual : List Nat) :
    (consumeCandidate edge ⟨residual⟩).map DLDS.Compression.Mixed.LocalRules.CandidateResidual.colours =
      consume edge residual := by
  cases residual <;> simp [consumeCandidate, consume]

@[simp] theorem consumeCandidate_cons (edge : Deduction) (colour : Nat)
    (rest : List Nat) (colourEq : edge.COLOUR = colour) :
    consumeCandidate edge ⟨colour :: rest⟩ = some ⟨rest⟩ := by
  simp [consumeCandidate, colourEq]

@[simp] theorem consumeCandidate_empty_default (edge : Deduction)
    (isDefault : edge.COLOUR = 0) :
    consumeCandidate edge ⟨[]⟩ = some ⟨[]⟩ := by
  simp [consumeCandidate, isDefault]

/-- Pair two premise occurrences only when both concrete edges consume and
leave the same residual. -/
def candidatePairTail (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : CandidateResidual) :
    Option CandidateResidual :=
  match consumeCandidate minorEdge minorResidual,
      consumeCandidate majorEdge majorResidual with
  | some minorTail, some majorTail =>
      if minorTail = majorTail then some minorTail else none
  | _, _ => none

def flowPairTail (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : ColourPath) : Option ColourPath :=
  match consume minorEdge minorResidual, consume majorEdge majorResidual with
  | some minorTail, some majorTail =>
      if minorTail = majorTail then some minorTail else none
  | _, _ => none

/-- Compatible-premise pairing is preserved exactly by projection from the
typed candidate residual to a raw edge-colour list. -/
theorem candidatePairTail_colours
    (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : CandidateResidual) :
    (candidatePairTail minorEdge majorEdge minorResidual majorResidual).map
        DLDS.Compression.Mixed.LocalRules.CandidateResidual.colours =
      flowPairTail minorEdge majorEdge minorResidual.colours
        majorResidual.colours := by
  obtain ⟨minorColours⟩ := minorResidual
  obtain ⟨majorColours⟩ := majorResidual
  have minorProjection := consumeCandidate_colours minorEdge minorColours
  have majorProjection := consumeCandidate_colours majorEdge majorColours
  cases hminor : consumeCandidate minorEdge ⟨minorColours⟩ with
  | none =>
      have flowMinor : consume minorEdge minorColours = none := by
        simpa [hminor] using minorProjection.symm
      simp [candidatePairTail, flowPairTail, hminor, flowMinor]
  | some minorTail =>
      have flowMinor : consume minorEdge minorColours =
          some minorTail.colours := by
        simpa [hminor] using minorProjection.symm
      cases hmajor : consumeCandidate majorEdge ⟨majorColours⟩ with
      | none =>
          have flowMajor : consume majorEdge majorColours = none := by
            simpa [hmajor] using majorProjection.symm
          simp [candidatePairTail, flowPairTail, hminor, hmajor,
            flowMinor, flowMajor]
      | some majorTail =>
          have flowMajor : consume majorEdge majorColours =
              some majorTail.colours := by
            simpa [hmajor] using majorProjection.symm
          by_cases equal : minorTail = majorTail
          · subst majorTail
            simp [candidatePairTail, flowPairTail, hminor, hmajor,
              flowMinor, flowMajor]
          · have coloursNe : minorTail.colours ≠ majorTail.colours := by
              intro coloursEqual
              apply equal
              cases minorTail
              cases majorTail
              simp_all
            simp [candidatePairTail, flowPairTail, hminor, hmajor,
              flowMinor, flowMajor, equal, coloursNe]

/-- Reseeding imports exactly the stored edge trace, without interpreting it
as a branch-only address. -/
def reseedFromEdgeTrace (trace : EdgeColourTrace) : CandidateResidual :=
  ⟨trace.colours⟩

/-- A reseeded trace constructed from a witnessed graph path resolves to the
recorded endpoint.  This is the ancestor-resolution property used by the
candidate convention. -/
theorem reseeded_endpoint_of_path
    {d : Graph} {start finish : Vertex} {edges : List Deduction}
    (path : EndpointPath d finish start edges) :
    followEdgeColours d (edges.length + 1) start
        (reseedFromEdgeTrace (edgeTraceOfEdges edges)).colours =
      some finish := by
  exact endpointPath_edgeColours_decode path

def CandidateResidualComplete (residual : CandidateResidual) : Prop :=
  residual.colours = []

def FlowResidualComplete (residual : ColourPath) : Prop := residual = []

theorem completion_projection_iff (residual : CandidateResidual) :
    CandidateResidualComplete residual ↔
      FlowResidualComplete residual.colours := Iff.rfl

/-- At a terminal vertex an edge trace decodes back to that vertex exactly
when it is complete.  In particular, a nonempty residual is not silently
discarded at a root. -/
theorem terminal_decode_iff_complete
    {d : Graph} {terminal : Vertex} (fuel : Nat)
    (noOutgoing : outgoing d terminal = [])
    (residual : CandidateResidual) :
    followEdgeColours d (fuel + 1) terminal residual.colours = some terminal ↔
      CandidateResidualComplete residual := by
  have outgoingEmpty : outgoing d terminal = [] := noOutgoing
  obtain ⟨colours⟩ := residual
  cases colours with
  | nil => simp [CandidateResidualComplete, followEdgeColours]
  | cons colour rest =>
      simp [CandidateResidualComplete, followEdgeColours,
        selectedOutgoing?, outgoingEmpty]

end DLDS.Compression.EdgeTraces
