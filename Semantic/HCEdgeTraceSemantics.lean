import Semantic.HCContextualMixedHE












open Semantic FlowSpec LocalSpecificationCandidate

namespace HCEdgeTraceSemantics

def residualOfEdgeTrace (trace : EdgeColourTrace) : CandidateResidual :=
  ⟨trace.colours⟩



theorem consumeCandidate_colours (edge : Deduction) (residual : List Nat) :
    (consumeCandidate edge ⟨residual⟩).map CandidateResidual.colours =
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



theorem candidatePairTail_colours
    (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : CandidateResidual) :
    (candidatePairTail minorEdge majorEdge minorResidual majorResidual).map
        CandidateResidual.colours =
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



def reseedFromEdgeTrace (trace : EdgeColourTrace) : CandidateResidual :=
  ⟨trace.colours⟩




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

end HCEdgeTraceSemantics
