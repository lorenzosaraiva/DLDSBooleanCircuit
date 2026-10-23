import Compression.HCContextualMixedHE












open Semantic FlowSpec MixedRuleSpecification

namespace HCEdgeTraceSemantics

def residualOfEdgeTrace (trace : EdgeColourTrace) : SelectionResidual :=
  ⟨trace.colours⟩



theorem consumeSelection_colours (edge : Deduction) (residual : List Nat) :
    (consumeSelection edge ⟨residual⟩).map SelectionResidual.colours =
      consume edge residual := by
  cases residual <;> simp [consumeSelection, consume]

@[simp] theorem consumeSelection_cons (edge : Deduction) (colour : Nat)
    (rest : List Nat) (colourEq : edge.COLOUR = colour) :
    consumeSelection edge ⟨colour :: rest⟩ = some ⟨rest⟩ := by
  simp [consumeSelection, colourEq]

@[simp] theorem consumeSelection_empty_default (edge : Deduction)
    (isDefault : edge.COLOUR = 0) :
    consumeSelection edge ⟨[]⟩ = some ⟨[]⟩ := by
  simp [consumeSelection, isDefault]



def selectionPairTail (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : SelectionResidual) :
    Option SelectionResidual :=
  match consumeSelection minorEdge minorResidual,
      consumeSelection majorEdge majorResidual with
  | some minorTail, some majorTail =>
      if minorTail = majorTail then some minorTail else none
  | _, _ => none

def flowPairTail (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : ColourPath) : Option ColourPath :=
  match consume minorEdge minorResidual, consume majorEdge majorResidual with
  | some minorTail, some majorTail =>
      if minorTail = majorTail then some minorTail else none
  | _, _ => none



theorem selectionPairTail_colours
    (minorEdge majorEdge : Deduction)
    (minorResidual majorResidual : SelectionResidual) :
    (selectionPairTail minorEdge majorEdge minorResidual majorResidual).map
        SelectionResidual.colours =
      flowPairTail minorEdge majorEdge minorResidual.colours
        majorResidual.colours := by
  obtain ⟨minorColours⟩ := minorResidual
  obtain ⟨majorColours⟩ := majorResidual
  have minorProjection := consumeSelection_colours minorEdge minorColours
  have majorProjection := consumeSelection_colours majorEdge majorColours
  cases hminor : consumeSelection minorEdge ⟨minorColours⟩ with
  | none =>
      have flowMinor : consume minorEdge minorColours = none := by
        simpa [hminor] using minorProjection.symm
      simp [selectionPairTail, flowPairTail, hminor, flowMinor]
  | some minorTail =>
      have flowMinor : consume minorEdge minorColours =
          some minorTail.colours := by
        simpa [hminor] using minorProjection.symm
      cases hmajor : consumeSelection majorEdge ⟨majorColours⟩ with
      | none =>
          have flowMajor : consume majorEdge majorColours = none := by
            simpa [hmajor] using majorProjection.symm
          simp [selectionPairTail, flowPairTail, hminor, hmajor,
            flowMinor, flowMajor]
      | some majorTail =>
          have flowMajor : consume majorEdge majorColours =
              some majorTail.colours := by
            simpa [hmajor] using majorProjection.symm
          by_cases equal : minorTail = majorTail
          · subst majorTail
            simp [selectionPairTail, flowPairTail, hminor, hmajor,
              flowMinor, flowMajor]
          · have coloursNe : minorTail.colours ≠ majorTail.colours := by
              intro coloursEqual
              apply equal
              cases minorTail
              cases majorTail
              simp_all
            simp [selectionPairTail, flowPairTail, hminor, hmajor,
              flowMinor, flowMajor, equal, coloursNe]



def reseedFromEdgeTrace (trace : EdgeColourTrace) : SelectionResidual :=
  ⟨trace.colours⟩




theorem reseeded_endpoint_of_path
    {d : Graph} {start finish : Vertex} {edges : List Deduction}
    (path : EndpointPath d finish start edges) :
    followEdgeColours d (edges.length + 1) start
        (reseedFromEdgeTrace (edgeTraceOfEdges edges)).colours =
      some finish := by
  exact endpointPath_edgeColours_decode path

def SelectionResidualComplete (residual : SelectionResidual) : Prop :=
  residual.colours = []

def FlowResidualComplete (residual : ColourPath) : Prop := residual = []

theorem completion_projection_iff (residual : SelectionResidual) :
    SelectionResidualComplete residual ↔
      FlowResidualComplete residual.colours := Iff.rfl




theorem terminal_decode_iff_complete
    {d : Graph} {terminal : Vertex} (fuel : Nat)
    (noOutgoing : outgoing d terminal = [])
    (residual : SelectionResidual) :
    followEdgeColours d (fuel + 1) terminal residual.colours = some terminal ↔
      SelectionResidualComplete residual := by
  have outgoingEmpty : outgoing d terminal = [] := noOutgoing
  obtain ⟨colours⟩ := residual
  cases colours with
  | nil => simp [SelectionResidualComplete, followEdgeColours]
  | cons colour rest =>
      simp [SelectionResidualComplete, followEdgeColours,
        selectedOutgoing?, outgoingEmpty]

end HCEdgeTraceSemantics
