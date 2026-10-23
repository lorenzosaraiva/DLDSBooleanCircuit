import Compression.MixedRuleSpecification

set_option linter.unusedSimpArgs false










open Semantic FlowSpec MixedRuleSpecification

namespace ActualMixedHIProducerBridge

structure ActualMixedHIInput (d : Graph) where
  derived : Vertex
  source : Vertex
  premise : Vertex
  sourceUse : Vertex
  derivedUse : Vertex
  antecedent : Formula
  consequent : Formula
  premiseStored : List Formula
  sourceStored : List Formula
  derivedStored : List Formula
  derivedMember : derived ∈ d.NODES
  sourceMember : source ∈ d.NODES
  premiseMember : premise ∈ d.NODES
  premiseEdgeMember : edge premise derived 0 premiseStored ∈ d.EDGES
  sourceEdgeMember : edge source sourceUse 0 sourceStored ∈ d.EDGES
  derivedEdgeMember : edge derived derivedUse 0 derivedStored ∈ d.EDGES
  derivedNotSource : derived.HYPOTHESIS = false
  sourceIsHypothesis : source.HYPOTHESIS = true
  premiseIsHypothesis : premise.HYPOTHESIS = true
  derivedUncollapsed : derived.COLLAPSED = false
  sourceUncollapsed : source.COLLAPSED = false
  derivedPastEmpty : derived.PAST = []
  sourcePastEmpty : source.PAST = []
  sameFormula : source.FORMULA = derived.FORMULA
  sameLevel : source.LEVEL = derived.LEVEL
  derivedFormula : derived.FORMULA = antecedent >> consequent
  premiseFormula : premise.FORMULA = consequent
  positiveGeneratedColour : derived.NUMBER ≠ 0
  occurrencesDistinct : derived ≠ source
  premise_ne_derived : premise ≠ derived
  sourceUse_ne_derived : sourceUse ≠ derived
  derivedUse_ne_derived : derivedUse ≠ derived
  premise_ne_center : premise ≠ collapse.center derived source
  sourceUse_ne_center : sourceUse ≠ collapse.center derived source
  derivedUse_ne_center : derivedUse ≠ collapse.center derived source
  sourceUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) sourceUse sourceUse = true
  derivedUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) derivedUse derivedUse = true
  derivedRule : get_rule derived d =
    rule derived [edge premise derived 0 premiseStored]
      [edge derived derivedUse 0 derivedStored] [] []
  sourceRule : get_rule source d =
    rule source [] [edge source sourceUse 0 sourceStored] [] []

def producedCenter {d : Graph} (input : ActualMixedHIInput d) : Vertex :=
  collapse.center input.derived input.source

def producedPremiseEdge {d : Graph} (input : ActualMixedHIInput d) :
    Deduction :=
  edge input.premise (producedCenter input) 0 input.premiseStored

def producedSourceEdge {d : Graph} (input : ActualMixedHIInput d) :
    Deduction :=
  edge (producedCenter input) input.sourceUse 0 input.sourceStored

def producedDerivedEdge {d : Graph} (input : ActualMixedHIInput d) :
    Deduction :=
  edge (producedCenter input) input.derivedUse input.derived.NUMBER
    input.derivedStored

def producedPremiseTrace {d : Graph} (input : ActualMixedHIInput d) :
    Ancestral :=
  path input.derivedUse input.premise [0, input.derived.NUMBER]

def producedNeighborhood {d : Graph} (input : ActualMixedHIInput d) :
    Neighborhood :=
  rule (producedCenter input) [producedPremiseEdge input]
    [producedSourceEdge input, producedDerivedEdge input]
    [] [producedPremiseTrace input]

def actualOrderedMixedOperation {d : Graph} (input : ActualMixedHIInput d) :
    Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.derived d))
    (pre_collapse (get_rule input.source d))


theorem actualOrderedMixedOperation_eq_produced
    {d : Graph} (input : ActualMixedHIInput d) :
    actualOrderedMixedOperation input = (producedNeighborhood input, []) := by
  have zeroNe : 0 ≠ input.derived.NUMBER :=
    input.positiveGeneratedColour.symm
  have outgoingNe : producedSourceEdge input ≠ producedDerivedEdge input := by
    intro equal
    exact zeroNe (congrArg Deduction.COLOUR equal)
  have outgoingBeq :
      (producedDerivedEdge input == producedSourceEdge input) = false :=
    beq_eq_false_iff_ne.mpr outgoingNe.symm
  have outgoingDedup :
      List.eraseDups [producedSourceEdge input, producedDerivedEdge input] =
        [producedSourceEdge input, producedDerivedEdge input] := by
    simp [List.eraseDups_cons, List.filter, outgoingBeq]
  have incomingDedup :
      List.eraseDups [producedPremiseEdge input] =
        [producedPremiseEdge input] := by rfl
  simp [actualOrderedMixedOperation, input.derivedRule, input.sourceRule,
    pre_collapse, pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.indirect.create,
    pre_collapse.discriminatePair,
    pre_collapse.outgoingRepresentativeConflictB,
    collapseWithOccurrences, collapse.center, collapse.rewrite_incoming,
    collapse.rewrite_outgoing, collapse.rewrite_direct,
    collapse.checkNeighborhoodPathHeads, collapse.neighborhoodPathHeadsOKB,
    collapse.neighborhoodPathHeadOKB,
    collapse.checkOutgoingRepresentatives,
    collapse.outgoingRepresentativesOKB, producedNeighborhood,
    producedCenter, producedPremiseEdge, producedSourceEdge,
    producedDerivedEdge, producedPremiseTrace,
    input.derivedNotSource, input.sourceIsHypothesis,
    input.derivedUncollapsed, input.sourceUncollapsed,
    input.positiveGeneratedColour, input.premise_ne_derived,
    input.sourceUse_ne_derived, input.derivedUse_ne_derived,
    input.sourceUseSelfRepresentative,
    input.derivedUseSelfRepresentative, zeroNe]
  simpa [producedPremiseEdge, producedSourceEdge, producedDerivedEdge,
    producedCenter,
    collapse.center, input.derivedNotSource, input.sourceIsHypothesis] using
    And.intro incomingDedup outgoingDedup


structure MixedHIRuntimeView (d : Graph) where
  mixed : Vertex
  premise : Vertex
  sourceUse : Vertex
  derivedUse : Vertex
  antecedent : Formula
  consequent : Formula
  premiseEdge : Deduction
  sourceEdge : Deduction
  derivedEdge : Deduction
  premiseTrace : Ancestral
  mixedMember : mixed ∈ d.NODES
  premiseMember : premise ∈ d.NODES
  premiseIsHypothesis : premise.HYPOTHESIS = true
  mixedHasSourceMark : mixed.HYPOTHESIS = true
  mixedFormula : mixed.FORMULA = antecedent >> consequent
  premiseFormula : premise.FORMULA = consequent
  incomingExactly : get_rule.incoming mixed d = [premiseEdge]
  premiseEdgeShape : premiseEdge.START = premise ∧
    premiseEdge.END = mixed ∧ premiseEdge.COLOUR = 0
  outgoingExactly : get_rule.outgoing mixed d = [sourceEdge, derivedEdge]
  sourceEdgeShape : sourceEdge.START = mixed ∧
    sourceEdge.END = sourceUse ∧ sourceEdge.COLOUR = 0
  derivedEdgeShape : derivedEdge.START = mixed ∧
    derivedEdge.END = derivedUse ∧ derivedEdge.COLOUR ≠ 0
  premiseTraceShape : premiseTrace.START = derivedUse ∧
    premiseTrace.END = premise ∧
    premiseTrace.COLOURS = [0, derivedEdge.COLOUR]
  premiseTraceMember : premiseTrace ∈ d.PATHS

def neighborhoodGraph (n : Neighborhood) : Graph :=
  dlds ([n.CENTER] ++ n.INCOMING.map (fun e => e.START) ++
      n.OUTGOING.map (fun e => e.END)).eraseDups
    (n.INCOMING ++ n.OUTGOING) (n.DIRECT ++ n.INDIRECT)

def producedLocalGraph {d : Graph} (input : ActualMixedHIInput d) : Graph :=
  neighborhoodGraph (producedNeighborhood input)

def producedRuntimeView {d : Graph} (input : ActualMixedHIInput d) :
    MixedHIRuntimeView (producedLocalGraph input) := by
  have premiseNotCenter : input.premise ≠ producedCenter input :=
    input.premise_ne_center
  have sourceUseNotCenter : input.sourceUse ≠ producedCenter input :=
    input.sourceUse_ne_center
  have derivedUseNotCenter : input.derivedUse ≠ producedCenter input :=
    input.derivedUse_ne_center
  simp [producedCenter, collapse.center, input.derivedFormula,
    input.derivedNotSource, input.sourceIsHypothesis] at premiseNotCenter sourceUseNotCenter derivedUseNotCenter
  refine
    { mixed := producedCenter input
      premise := input.premise
      sourceUse := input.sourceUse
      derivedUse := input.derivedUse
      antecedent := input.antecedent
      consequent := input.consequent
      premiseEdge := producedPremiseEdge input
      sourceEdge := producedSourceEdge input
      derivedEdge := producedDerivedEdge input
      premiseTrace := producedPremiseTrace input
      mixedMember := ?_
      premiseMember := ?_
      premiseIsHypothesis := input.premiseIsHypothesis
      mixedHasSourceMark := ?_
      mixedFormula := ?_
      premiseFormula := input.premiseFormula
      incomingExactly := ?_
      premiseEdgeShape := ?_
      outgoingExactly := ?_
      sourceEdgeShape := ?_
      derivedEdgeShape := ?_
      premiseTraceShape := ?_
      premiseTraceMember := ?_ }
  all_goals
    simp [producedLocalGraph, neighborhoodGraph, producedNeighborhood,
      producedCenter, producedPremiseEdge, producedSourceEdge,
      producedDerivedEdge, producedPremiseTrace, collapse.center,
      input.derivedNotSource, input.sourceIsHypothesis,
      input.derivedFormula, input.premiseFormula,
      input.positiveGeneratedColour, input.premise_ne_derived,
      input.sourceUse_ne_derived, input.derivedUse_ne_derived,
      input.premise_ne_center, input.sourceUse_ne_center,
      input.derivedUse_ne_center,
      premiseNotCenter, sourceUseNotCenter, derivedUseNotCenter,
      get_rule.incoming, get_rule.incoming.loop,
      get_rule.outgoing, get_rule.outgoing.loop]

def mixedHISourceRoute {d : Graph} (view : MixedHIRuntimeView d) :
    SelectionRoute :=
  { production := .source
    dependency := normalizeDep [view.mixed.FORMULA]
    residual := ⟨[]⟩ }

def mixedHIIntroductionRoute {d : Graph} (view : MixedHIRuntimeView d) :
    SelectionRoute :=
  { production := .introduction
    dependency := depRemove (normalizeDep [view.premise.FORMULA])
      view.antecedent
    residual := ⟨[view.derivedEdge.COLOUR]⟩ }

def mixedHIRoutes {d : Graph} (view : MixedHIRuntimeView d) :
    List SelectionRoute :=
  [mixedHISourceRoute view, mixedHIIntroductionRoute view]

theorem mixedHI_premise_trace_tail {d : Graph}
    (view : MixedHIRuntimeView d) :
    view.premiseTrace.COLOURS =
      0 :: (mixedHIIntroductionRoute view).residual.colours := by
  simpa [mixedHIIntroductionRoute] using view.premiseTraceShape.2.2

theorem mixedHI_routes_select_disjoint_uses {d : Graph}
    (view : MixedHIRuntimeView d) :
    consumeSelection view.sourceEdge (mixedHISourceRoute view).residual =
        some ⟨[]⟩ ∧
      consumeSelection view.derivedEdge (mixedHISourceRoute view).residual =
        none ∧
      consumeSelection view.derivedEdge
          (mixedHIIntroductionRoute view).residual = some ⟨[]⟩ ∧
      consumeSelection view.sourceEdge
          (mixedHIIntroductionRoute view).residual = none := by
  have sourceColour := view.sourceEdgeShape.2.2
  have derivedColour := view.derivedEdgeShape.2.2
  have zeroNe : 0 ≠ view.derivedEdge.COLOUR :=
    fun equal => derivedColour equal.symm
  simp [mixedHISourceRoute, mixedHIIntroductionRoute, consumeSelection,
    sourceColour, derivedColour, zeroNe]



def mixedHIRuntimeOKB (d : Graph) (v : Vertex) : Bool :=
  let predecessors := predsOf d v
  let outgoing := get_rule.outgoing v d
  let zeroOutgoing := outgoing.filter fun edge => edge.COLOUR = 0
  let colouredOutgoing := outgoing.filter fun edge => edge.COLOUR ≠ 0
  match predecessors, v.FORMULA, zeroOutgoing, colouredOutgoing with
  | [premise], Formula.implication _ consequent, [sourceEdge], [derivedEdge] =>
      match edgesBetween d premise v with
      | [premiseEdge] =>
          v.HYPOTHESIS && premise.FORMULA == consequent &&
          !elimPremiseAtB d v premise &&
          (premiseEdge.COLOUR == 0) &&
          (sourceEdge.START == v) && (derivedEdge.START == v) &&
          (derivedEdge.COLOUR != 0) &&
          (ancestorsInto d v).isEmpty &&
          d.PATHS.contains
            (path derivedEdge.END premise [0, derivedEdge.COLOUR])
      | _ => false
  | _, _, _, _ => false

end ActualMixedHIProducerBridge
