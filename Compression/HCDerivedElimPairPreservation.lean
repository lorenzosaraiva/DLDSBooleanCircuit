import Compression.NDTreeHCInitialState
import Compression.ActualSourcePairBridge

set_option linter.unusedSimpArgs false















open Semantic FlowSpec
open MixedRuleReplay OccurrenceAwareReplay
open HCContextualMixedHE

namespace HCDerivedElimPairPreservation





structure Input (d : Graph) where
  left : Vertex
  right : Vertex
  leftMajor : Vertex
  leftMinor : Vertex
  rightMajor : Vertex
  rightMinor : Vertex
  leftUse : Vertex
  rightUse : Vertex
  leftMajorStored : Dep
  leftMinorStored : Dep
  rightMajorStored : Dep
  rightMinorStored : Dep
  leftUseStored : Dep
  rightUseStored : Dep
  leftMember : left ∈ d.NODES
  rightMember : right ∈ d.NODES
  leftNotSource : left.HYPOTHESIS = false
  rightNotSource : right.HYPOTHESIS = false
  leftUncollapsed : left.COLLAPSED = false
  rightUncollapsed : right.COLLAPSED = false
  sameFormula : left.FORMULA = right.FORMULA
  sameLevel : left.LEVEL = right.LEVEL
  leftMajorFormula :
    leftMajor.FORMULA = leftMinor.FORMULA >> left.FORMULA
  rightMajorFormula :
    rightMajor.FORMULA = rightMinor.FORMULA >> right.FORMULA
  crossEliminationsAbsent :
    ∀ major ∈ [rightMajor, rightMinor],
      ∀ minor ∈ [leftMajor, leftMinor],
        major.FORMULA ≠ minor.FORMULA >> left.FORMULA ∧
        minor.FORMULA ≠ major.FORMULA >> left.FORMULA
  crossPremiseFormulaDistinct :
    ∀ rightPremise ∈ [rightMajor, rightMinor],
      ∀ leftPremise ∈ [leftMajor, leftMinor],
        rightPremise.FORMULA ≠ leftPremise.FORMULA
  leftPositive : left.NUMBER ≠ 0
  rightPositive : right.NUMBER ≠ 0
  distinctNumbers : left.NUMBER ≠ right.NUMBER
  distinctCenters : left ≠ right
  rightMajor_ne_rightMinor : rightMajor ≠ rightMinor
  leftMajor_ne_leftMinor : leftMajor ≠ leftMinor
  rightMajor_ne_leftMajor : rightMajor ≠ leftMajor
  rightMajor_ne_leftMinor : rightMajor ≠ leftMinor
  rightMinor_ne_leftMajor : rightMinor ≠ leftMajor
  rightMinor_ne_leftMinor : rightMinor ≠ leftMinor
  rightMajor_ne_center : rightMajor ≠ collapse.center left right
  rightMinor_ne_center : rightMinor ≠ collapse.center left right
  leftMajor_ne_center : leftMajor ≠ collapse.center left right
  leftMinor_ne_center : leftMinor ≠ collapse.center left right
  rightUse_ne_center : rightUse ≠ collapse.center left right
  leftUse_ne_center : leftUse ≠ collapse.center left right
  rightUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) rightUse rightUse = true
  leftUseSelfRepresentative :
    sameHCRepresentativeB (hcPartition d) leftUse leftUse = true
  leftRule : get_rule left d =
    rule left
      [edge leftMajor left 0 leftMajorStored,
       edge leftMinor left 0 leftMinorStored]
      [edge left leftUse 0 leftUseStored] [] []
  rightRule : get_rule right d =
    rule right
      [edge rightMajor right 0 rightMajorStored,
       edge rightMinor right 0 rightMinorStored]
      [edge right rightUse 0 rightUseStored] [] []

def center {d : Graph} (input : Input d) : Vertex :=
  collapse.center input.left input.right

def rightMajorEdge {d : Graph} (input : Input d) : Deduction :=
  edge input.rightMajor (center input) 0 input.rightMajorStored

def rightMinorEdge {d : Graph} (input : Input d) : Deduction :=
  edge input.rightMinor (center input) 0 input.rightMinorStored

def leftMajorEdge {d : Graph} (input : Input d) : Deduction :=
  edge input.leftMajor (center input) 0 input.leftMajorStored

def leftMinorEdge {d : Graph} (input : Input d) : Deduction :=
  edge input.leftMinor (center input) 0 input.leftMinorStored

def rightOut {d : Graph} (input : Input d) : Deduction :=
  edge (center input) input.rightUse input.right.NUMBER input.rightUseStored

def leftOut {d : Graph} (input : Input d) : Deduction :=
  edge (center input) input.leftUse input.left.NUMBER input.leftUseStored

def rightMajorTrace {d : Graph} (input : Input d) : Ancestral :=
  path input.rightUse input.rightMajor [0, input.right.NUMBER]

def rightMinorTrace {d : Graph} (input : Input d) : Ancestral :=
  path input.rightUse input.rightMinor [0, input.right.NUMBER]

def leftMajorTrace {d : Graph} (input : Input d) : Ancestral :=
  path input.leftUse input.leftMajor [0, input.left.NUMBER]

def leftMinorTrace {d : Graph} (input : Input d) : Ancestral :=
  path input.leftUse input.leftMinor [0, input.left.NUMBER]

def producedNeighborhood {d : Graph} (input : Input d) : Neighborhood :=
  rule (center input)
    [rightMajorEdge input, rightMinorEdge input,
     leftMajorEdge input, leftMinorEdge input]
    [rightOut input, leftOut input]
    []
    [rightMajorTrace input, rightMinorTrace input,
     leftMajorTrace input, leftMinorTrace input]

def preparedLeft {d : Graph} (input : Input d) : Neighborhood :=
  rule input.left
    [edge input.leftMajor input.left 0 input.leftMajorStored,
     edge input.leftMinor input.left 0 input.leftMinorStored]
    [edge input.left input.leftUse input.left.NUMBER input.leftUseStored]
    []
    [path input.leftUse input.leftMajor [0, input.left.NUMBER],
     path input.leftUse input.leftMinor [0, input.left.NUMBER]]

def preparedRight {d : Graph} (input : Input d) : Neighborhood :=
  rule input.right
    [edge input.rightMajor input.right 0 input.rightMajorStored,
     edge input.rightMinor input.right 0 input.rightMinorStored]
    [edge input.right input.rightUse input.right.NUMBER input.rightUseStored]
    []
    [path input.rightUse input.rightMajor [0, input.right.NUMBER],
     path input.rightUse input.rightMinor [0, input.right.NUMBER]]

def operation {d : Graph} (input : Input d) : Neighborhood × List Ancestral :=
  collapseWithOccurrences (hcPartition d)
    (pre_collapse (get_rule input.left d))
    (pre_collapse (get_rule input.right d))

private theorem pre_collapse_left {d : Graph} (input : Input d) :
    pre_collapse (get_rule input.left d) = preparedLeft input := by
  rw [input.leftRule]
  simp [pre_collapse, pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.indirect.create,
    preparedLeft, input.leftNotSource, input.leftUncollapsed]

private theorem pre_collapse_right {d : Graph} (input : Input d) :
    pre_collapse (get_rule input.right d) = preparedRight input := by
  rw [input.rightRule]
  simp [pre_collapse, pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.indirect.create,
    preparedRight, input.rightNotSource, input.rightUncollapsed]

private theorem prepared_pair {d : Graph} (input : Input d) :
    pre_collapse.discriminatePair (hcPartition d)
      (preparedLeft input) (preparedRight input) =
        (preparedLeft input, preparedRight input) := by
  simp [pre_collapse.discriminatePair,
    pre_collapse.outgoingRepresentativeConflictB,
    preparedLeft, preparedRight, input.distinctNumbers]

private theorem incoming_nodup {d : Graph} (input : Input d) :
    List.eraseDups
      [rightMajorEdge input, rightMinorEdge input,
       leftMajorEdge input, leftMinorEdge input] =
      [rightMajorEdge input, rightMinorEdge input,
       leftMajorEdge input, leftMinorEdge input] := by
  have h21 : (rightMinorEdge input == rightMajorEdge input) = false := by
    apply beq_eq_false_iff_ne.mpr
    intro equal
    exact input.rightMajor_ne_rightMinor.symm
      (congrArg Deduction.START equal)
  have h31 : (leftMajorEdge input == rightMajorEdge input) = false := by
    apply beq_eq_false_iff_ne.mpr
    intro equal
    exact input.rightMajor_ne_leftMajor.symm
      (congrArg Deduction.START equal)
  have h41 : (leftMinorEdge input == rightMajorEdge input) = false := by
    apply beq_eq_false_iff_ne.mpr
    intro equal
    exact input.rightMajor_ne_leftMinor.symm
      (congrArg Deduction.START equal)
  have h32 : (leftMajorEdge input == rightMinorEdge input) = false := by
    apply beq_eq_false_iff_ne.mpr
    intro equal
    exact input.rightMinor_ne_leftMajor.symm
      (congrArg Deduction.START equal)
  have h42 : (leftMinorEdge input == rightMinorEdge input) = false := by
    apply beq_eq_false_iff_ne.mpr
    intro equal
    exact input.rightMinor_ne_leftMinor.symm
      (congrArg Deduction.START equal)
  have h43 : (leftMinorEdge input == leftMajorEdge input) = false := by
    apply beq_eq_false_iff_ne.mpr
    intro equal
    exact input.leftMajor_ne_leftMinor.symm
      (congrArg Deduction.START equal)
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
    h21, h31, h41, h32, h42, h43]

private theorem outgoing_nodup {d : Graph} (input : Input d) :
    List.eraseDups [rightOut input, leftOut input] =
      [rightOut input, leftOut input] := by
  have ne : rightOut input ≠ leftOut input := by
    intro equal
    exact input.distinctNumbers.symm (congrArg Deduction.COLOUR equal)
  have hbeq : (leftOut input == rightOut input) = false := by
    exact beq_eq_false_iff_ne.mpr ne.symm
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop, hbeq]

private theorem incoming_nodup_append {d : Graph} (input : Input d) :
    List.eraseDups
      ([rightMajorEdge input, rightMinorEdge input] ++
       [leftMajorEdge input, leftMinorEdge input]) =
      [rightMajorEdge input, rightMinorEdge input,
       leftMajorEdge input, leftMinorEdge input] := by
  change List.eraseDups
      [rightMajorEdge input, rightMinorEdge input,
       leftMajorEdge input, leftMinorEdge input] = _
  exact incoming_nodup input

private theorem outgoing_nodup_append {d : Graph} (input : Input d) :
    List.eraseDups ([rightOut input] ++ [leftOut input]) =
      [rightOut input, leftOut input] := by
  change List.eraseDups [rightOut input, leftOut input] = _
  exact outgoing_nodup input

private theorem produced_path_heads {d : Graph} (input : Input d) :
    collapse.neighborhoodPathHeadsOKB (producedNeighborhood input) = true := by
  simp [collapse.neighborhoodPathHeadsOKB,
    collapse.neighborhoodPathHeadOKB, producedNeighborhood,
    center, rightMajorEdge, rightMinorEdge, leftMajorEdge, leftMinorEdge,
    rightOut, leftOut, rightMajorTrace, rightMinorTrace,
    leftMajorTrace, leftMinorTrace,
    input.rightMajor_ne_center, input.rightMinor_ne_center,
    input.leftMajor_ne_center, input.leftMinor_ne_center]

private theorem produced_outgoing_representatives
    {d : Graph} (input : Input d) :
    collapse.outgoingRepresentativesOKB (hcPartition d)
      (producedNeighborhood input) = true := by
  have reverseNumbers : input.right.NUMBER ≠ input.left.NUMBER :=
    input.distinctNumbers.symm
  simp [collapse.outgoingRepresentativesOKB, producedNeighborhood,
    rightOut, leftOut, input.distinctNumbers, reverseNumbers,
    input.rightUseSelfRepresentative, input.leftUseSelfRepresentative]


theorem operation_eq_produced {d : Graph} (input : Input d) :
    operation input = (producedNeighborhood input, []) := by
  unfold operation
  rw [pre_collapse_left input, pre_collapse_right input]
  unfold collapseWithOccurrences
  rw [prepared_pair input]
  simp only [preparedLeft, preparedRight, collapse.rewrite_incoming,
    collapse.rewrite_outgoing, collapse.rewrite_direct,
    List.filter_nil, List.append_nil]
  change
    (collapse.checkOutgoingRepresentatives (hcPartition d)
        (collapse.checkNeighborhoodPathHeads
          (rule (center input)
            (List.eraseDups
              ([rightMajorEdge input, rightMinorEdge input] ++
               [leftMajorEdge input, leftMinorEdge input]))
            (List.eraseDups ([rightOut input] ++ [leftOut input]))
            []
            ([rightMajorTrace input, rightMinorTrace input] ++
             [leftMajorTrace input, leftMinorTrace input]))),
      []) = (producedNeighborhood input, [])
  rw [incoming_nodup_append input, outgoing_nodup_append input]
  change
    (collapse.checkOutgoingRepresentatives (hcPartition d)
        (collapse.checkNeighborhoodPathHeads (producedNeighborhood input)),
      []) = (producedNeighborhood input, [])
  rw [collapse.checkNeighborhoodPathHeads,
    produced_path_heads input]
  simp only [Bool.true_eq, ↓reduceIte]
  rw [collapse.checkOutgoingRepresentatives,
    produced_outgoing_representatives input]
  rfl

def localGraph {d : Graph} (input : Input d) : Graph :=
  ActualMixedProducerBridge.neighborhoodGraph (producedNeighborhood input)


def premiseFlow {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (vertex : Vertex) : List FlowPair :=
  if vertex = input.rightMajor then
    [(rightMajorDep, [0, input.right.NUMBER])]
  else if vertex = input.rightMinor then
    [(rightMinorDep, [0, input.right.NUMBER])]
  else if vertex = input.leftMajor then
    [(leftMajorDep, [0, input.left.NUMBER])]
  else if vertex = input.leftMinor then
    [(leftMinorDep, [0, input.left.NUMBER])]
  else []

def expectedRoutes {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    List FlowPair :=
  [(depUnion rightMinorDep rightMajorDep, [input.right.NUMBER]),
   (depUnion leftMinorDep leftMajorDep, [input.left.NUMBER])]

private theorem predecessor_nodup {d : Graph} (input : Input d) :
    List.eraseDups
      [input.rightMajor, input.rightMinor,
       input.leftMajor, input.leftMinor] =
      [input.rightMajor, input.rightMinor,
       input.leftMajor, input.leftMinor] := by
  have h21 : (input.rightMinor == input.rightMajor) = false :=
    beq_eq_false_iff_ne.mpr input.rightMajor_ne_rightMinor.symm
  have h31 : (input.leftMajor == input.rightMajor) = false :=
    beq_eq_false_iff_ne.mpr input.rightMajor_ne_leftMajor.symm
  have h41 : (input.leftMinor == input.rightMajor) = false :=
    beq_eq_false_iff_ne.mpr input.rightMajor_ne_leftMinor.symm
  have h32 : (input.leftMajor == input.rightMinor) = false :=
    beq_eq_false_iff_ne.mpr input.rightMinor_ne_leftMajor.symm
  have h42 : (input.leftMinor == input.rightMinor) = false :=
    beq_eq_false_iff_ne.mpr input.rightMinor_ne_leftMinor.symm
  have h43 : (input.leftMinor == input.leftMajor) = false :=
    beq_eq_false_iff_ne.mpr input.leftMajor_ne_leftMinor.symm
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
    h21, h31, h41, h32, h42, h43]

theorem local_incoming {d : Graph} (input : Input d) :
    get_rule.incoming (center input) (localGraph input) =
      [rightMajorEdge input, rightMinorEdge input,
       leftMajorEdge input, leftMinorEdge input] := by
  simp [localGraph, ActualMixedProducerBridge.neighborhoodGraph,
    producedNeighborhood, center, rightMajorEdge, rightMinorEdge,
    leftMajorEdge, leftMinorEdge, rightOut, leftOut,
    get_rule.incoming, get_rule.incoming.loop,
    input.rightUse_ne_center, input.leftUse_ne_center]

theorem local_predecessors {d : Graph} (input : Input d) :
    predsOf (localGraph input) (center input) =
      [input.rightMajor, input.rightMinor,
       input.leftMajor, input.leftMinor] := by
  rw [FlowSpec.predsOf_eq, local_incoming input]
  change List.eraseDups
      [input.rightMajor, input.rightMinor,
       input.leftMajor, input.leftMinor] = _
  exact predecessor_nodup input

private theorem local_no_center_ancestors {d : Graph} (input : Input d) :
    ancestorsInto (localGraph input) (center input) = [] := by
  simp [ancestorsInto, localGraph,
    ActualMixedProducerBridge.neighborhoodGraph, producedNeighborhood,
    center, rightMajorTrace, rightMinorTrace, leftMajorTrace, leftMinorTrace,
    input.rightMajor_ne_center, input.rightMinor_ne_center,
    input.leftMajor_ne_center, input.leftMinor_ne_center]

private theorem edges_rightMajor {d : Graph} (input : Input d) :
    edgesBetween (localGraph input) input.rightMajor (center input) =
      [rightMajorEdge input] := by
  rw [HorizontalCompressionInput.edgesBetween_eq_incoming,
    local_incoming input]
  simp [rightMajorEdge, rightMinorEdge, leftMajorEdge, leftMinorEdge,
    input.rightMajor_ne_rightMinor, Ne.symm input.rightMajor_ne_rightMinor,
    input.rightMajor_ne_leftMajor, Ne.symm input.rightMajor_ne_leftMajor,
    input.rightMajor_ne_leftMinor, Ne.symm input.rightMajor_ne_leftMinor]

private theorem edges_rightMinor {d : Graph} (input : Input d) :
    edgesBetween (localGraph input) input.rightMinor (center input) =
      [rightMinorEdge input] := by
  rw [HorizontalCompressionInput.edgesBetween_eq_incoming,
    local_incoming input]
  simp [rightMajorEdge, rightMinorEdge, leftMajorEdge, leftMinorEdge,
    input.rightMajor_ne_rightMinor, Ne.symm input.rightMajor_ne_rightMinor,
    input.rightMinor_ne_leftMajor, Ne.symm input.rightMinor_ne_leftMajor,
    input.rightMinor_ne_leftMinor, Ne.symm input.rightMinor_ne_leftMinor]

private theorem edges_leftMajor {d : Graph} (input : Input d) :
    edgesBetween (localGraph input) input.leftMajor (center input) =
      [leftMajorEdge input] := by
  rw [HorizontalCompressionInput.edgesBetween_eq_incoming,
    local_incoming input]
  simp [rightMajorEdge, rightMinorEdge, leftMajorEdge, leftMinorEdge,
    input.rightMajor_ne_leftMajor, Ne.symm input.rightMajor_ne_leftMajor,
    input.rightMinor_ne_leftMajor, Ne.symm input.rightMinor_ne_leftMajor,
    input.leftMajor_ne_leftMinor, Ne.symm input.leftMajor_ne_leftMinor]

private theorem edges_leftMinor {d : Graph} (input : Input d) :
    edgesBetween (localGraph input) input.leftMinor (center input) =
      [leftMinorEdge input] := by
  rw [HorizontalCompressionInput.edgesBetween_eq_incoming,
    local_incoming input]
  simp [rightMajorEdge, rightMinorEdge, leftMajorEdge, leftMinorEdge,
    input.rightMajor_ne_leftMinor, Ne.symm input.rightMajor_ne_leftMinor,
    input.rightMinor_ne_leftMinor, Ne.symm input.rightMinor_ne_leftMinor,
    input.leftMajor_ne_leftMinor, Ne.symm input.leftMajor_ne_leftMinor]

set_option maxRecDepth 10000 in
private theorem local_elimPairs {d : Graph} (input : Input d) :
    elimPairsAt (localGraph input) (center input) =
      [(input.rightMajor, input.rightMinor),
       (input.leftMajor, input.leftMinor)] := by
  have crossRM_LM := input.crossEliminationsAbsent input.rightMajor
    (by simp) input.leftMajor (by simp)
  have crossRM_Lm := input.crossEliminationsAbsent input.rightMajor
    (by simp) input.leftMinor (by simp)
  have crossRm_LM := input.crossEliminationsAbsent input.rightMinor
    (by simp) input.leftMajor (by simp)
  have crossRm_Lm := input.crossEliminationsAbsent input.rightMinor
    (by simp) input.leftMinor (by simp)
  have formulaRM_LM := input.crossPremiseFormulaDistinct input.rightMajor
    (by simp) input.leftMajor (by simp)
  have formulaRM_Lm := input.crossPremiseFormulaDistinct input.rightMajor
    (by simp) input.leftMinor (by simp)
  have formulaRm_LM := input.crossPremiseFormulaDistinct input.rightMinor
    (by simp) input.leftMajor (by simp)
  have formulaRm_Lm := input.crossPremiseFormulaDistinct input.rightMinor
    (by simp) input.leftMinor (by simp)
  have centerFormula : (center input).FORMULA = input.left.FORMULA := by
    simp [center, collapse.center]
  have rightFormula : input.right.FORMULA = input.left.FORMULA :=
    input.sameFormula.symm
  have rightMinorNotNested :
      input.rightMinor.FORMULA ≠
        input.rightMajor.FORMULA >> input.left.FORMULA := by
    intro equal
    have sizes := congrArg sizeOf equal
    rw [input.rightMajorFormula, rightFormula] at sizes
    simp at sizes
    omega
  have leftMinorNotNested :
      input.leftMinor.FORMULA ≠
        input.leftMajor.FORMULA >> input.left.FORMULA := by
    intro equal
    have sizes := congrArg sizeOf equal
    rw [input.leftMajorFormula] at sizes
    simp at sizes
    omega
  unfold elimPairsAt
  rw [local_predecessors input]
  simp [centerFormula, rightFormula, input.rightMajorFormula,
    input.leftMajorFormula,
    crossRM_LM.1, crossRM_LM.2, crossRM_Lm.1, crossRM_Lm.2,
    crossRm_LM.1, crossRm_LM.2, crossRm_Lm.1, crossRm_Lm.2,
    formulaRM_LM, formulaRM_Lm, formulaRm_LM, formulaRm_Lm,
    Ne.symm formulaRM_LM, Ne.symm formulaRM_Lm,
    Ne.symm formulaRm_LM, Ne.symm formulaRm_Lm,
    FlowSpec.formula_ne_implication_self,
    FlowSpec.not_mutual_implication]
  simp [crossRm_LM.1, crossRm_Lm.1, crossRM_Lm.2,
    crossRm_Lm.2, rightMinorNotNested, leftMinorNotNested,
    FlowSpec.formula_ne_implication_self]

private theorem flow_rightMajor {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    premiseFlow input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        input.rightMajor =
      [(rightMajorDep, [0, input.right.NUMBER])] := by
  simp [premiseFlow]

private theorem flow_rightMinor {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    premiseFlow input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        input.rightMinor =
      [(rightMinorDep, [0, input.right.NUMBER])] := by
  simp [premiseFlow, input.rightMajor_ne_rightMinor,
    Ne.symm input.rightMajor_ne_rightMinor]

private theorem flow_leftMajor {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    premiseFlow input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        input.leftMajor =
      [(leftMajorDep, [0, input.left.NUMBER])] := by
  simp [premiseFlow, input.rightMajor_ne_leftMajor,
    Ne.symm input.rightMajor_ne_leftMajor,
    input.rightMinor_ne_leftMajor, Ne.symm input.rightMinor_ne_leftMajor]

private theorem flow_leftMinor {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    premiseFlow input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        input.leftMinor =
      [(leftMinorDep, [0, input.left.NUMBER])] := by
  simp [premiseFlow, input.rightMajor_ne_leftMinor,
    Ne.symm input.rightMajor_ne_leftMinor,
    input.rightMinor_ne_leftMinor, Ne.symm input.rightMinor_ne_leftMinor,
    input.leftMajor_ne_leftMinor, Ne.symm input.leftMajor_ne_leftMinor]




def eliminationPairOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) (pair : Vertex × Vertex) :
    List FlowPair :=
  (premiseFlow pair.2).flatMap fun (minorDep, minorResidual) =>
    (premiseFlow pair.1).flatMap fun (majorDep, majorResidual) =>
      (edgesBetween d pair.2 v).flatMap fun minorEdge =>
        (edgesBetween d pair.1 v).filterMap fun majorEdge =>
          match consume minorEdge minorResidual,
              consume majorEdge majorResidual with
          | some minorTail, some majorTail =>
              if minorTail = majorTail then
                some (depUnion minorDep majorDep, minorTail)
              else none
          | _, _ => none

private theorem selectionEliminationOutputs_eq_pairOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex → List FlowPair) :
    selectionEliminationOutputs d v premiseFlow =
      (elimPairsAt d v).flatMap
        (eliminationPairOutputs d v premiseFlow) := by
  rfl





theorem eliminationPairOutputs_zero_edges_distinct_tags
    (d : Graph) (v major minor : Vertex)
    (premiseFlow : Vertex → List FlowPair)
    (majorDep minorDep : Dep) (majorTag minorTag : Nat)
    (majorEdge minorEdge : Deduction)
    (majorFlow : premiseFlow major = [(majorDep, [0, majorTag])])
    (minorFlow : premiseFlow minor = [(minorDep, [0, minorTag])])
    (majorEdges : edgesBetween d major v = [majorEdge])
    (minorEdges : edgesBetween d minor v = [minorEdge])
    (majorZero : majorEdge.COLOUR = 0)
    (minorZero : minorEdge.COLOUR = 0)
    (distinctTags : minorTag ≠ majorTag) :
    eliminationPairOutputs d v premiseFlow (major, minor) = [] := by
  simp [eliminationPairOutputs, majorFlow, minorFlow, majorEdges, minorEdges,
    FlowSpec.consume, majorZero, minorZero, distinctTags]

private theorem right_pair_outputs
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    eliminationPairOutputs (localGraph input) (center input)
        (premiseFlow input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep)
        (input.rightMajor, input.rightMinor) =
      [(depUnion rightMinorDep rightMajorDep, [input.right.NUMBER])] := by
  simp [eliminationPairOutputs, flow_rightMajor input,
    flow_rightMinor input, edges_rightMajor input, edges_rightMinor input,
    FlowSpec.consume, rightMajorEdge, rightMinorEdge]

private theorem left_pair_outputs
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    eliminationPairOutputs (localGraph input) (center input)
        (premiseFlow input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep)
        (input.leftMajor, input.leftMinor) =
      [(depUnion leftMinorDep leftMajorDep, [input.left.NUMBER])] := by
  simp [eliminationPairOutputs, flow_leftMajor input,
    flow_leftMinor input, edges_leftMajor input, edges_leftMinor input,
    FlowSpec.consume, leftMajorEdge, leftMinorEdge]




private theorem right_left_cross_pair_outputs
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (major : Vertex) (majorMember : major ∈ [input.rightMajor, input.rightMinor])
    (minor : Vertex) (minorMember : minor ∈ [input.leftMajor, input.leftMinor]) :
    eliminationPairOutputs (localGraph input) (center input)
        (premiseFlow input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep) (major, minor) = [] := by
  simp at majorMember minorMember
  rcases majorMember with rfl | rfl <;>
    rcases minorMember with rfl | rfl <;>
    simp [eliminationPairOutputs, flow_rightMajor input,
      flow_rightMinor input, flow_leftMajor input, flow_leftMinor input,
      edges_rightMajor input, edges_rightMinor input,
      edges_leftMajor input, edges_leftMinor input,
      FlowSpec.consume, rightMajorEdge, rightMinorEdge,
      leftMajorEdge, leftMinorEdge, input.distinctNumbers,
      Ne.symm input.distinctNumbers]

private theorem left_right_cross_pair_outputs
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (major : Vertex) (majorMember : major ∈ [input.leftMajor, input.leftMinor])
    (minor : Vertex) (minorMember : minor ∈ [input.rightMajor, input.rightMinor]) :
    eliminationPairOutputs (localGraph input) (center input)
        (premiseFlow input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep) (major, minor) = [] := by
  simp at majorMember minorMember
  rcases majorMember with rfl | rfl <;>
    rcases minorMember with rfl | rfl <;>
    simp [eliminationPairOutputs, flow_rightMajor input,
      flow_rightMinor input, flow_leftMajor input, flow_leftMinor input,
      edges_rightMajor input, edges_rightMinor input,
      edges_leftMajor input, edges_leftMinor input,
      FlowSpec.consume, rightMajorEdge, rightMinorEdge,
      leftMajorEdge, leftMinorEdge, input.distinctNumbers,
      Ne.symm input.distinctNumbers]

set_option maxRecDepth 10000 in
theorem local_occurrenceStep
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    occurrenceStep (localGraph input) (center input)
        (premiseFlow input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep) =
      .ok (expectedRoutes input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep) := by
  let graph := localGraph input
  let root := center input
  let flow := premiseFlow input rightMajorDep rightMinorDep
    leftMajorDep leftMinorDep
  have predecessors : predsOf graph root =
      [input.rightMajor, input.rightMinor,
       input.leftMajor, input.leftMinor] := local_predecessors input
  have noAncestors : ancestorsInto graph root = [] :=
    local_no_center_ancestors input
  have rootFormula : root.FORMULA = input.left.FORMULA := by
    simp [root, center, collapse.center]
  have rootNotSource : root.HYPOTHESIS = false := by
    simp [root, center, collapse.center,
      input.leftNotSource, input.rightNotSource]
  have eliminationOutputs :
      selectionEliminationOutputs graph root flow =
        expectedRoutes input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep := by
    unfold selectionEliminationOutputs
    rw [show elimPairsAt graph root =
        [(input.rightMajor, input.rightMinor),
         (input.leftMajor, input.leftMinor)] by
      exact local_elimPairs input]
    simp only [List.flatMap_cons, List.flatMap_nil]
    rw [show flow input.rightMajor =
        [(rightMajorDep, [0, input.right.NUMBER])] by
      exact flow_rightMajor input _ _ _ _]
    rw [show flow input.rightMinor =
        [(rightMinorDep, [0, input.right.NUMBER])] by
      exact flow_rightMinor input _ _ _ _]
    rw [show flow input.leftMajor =
        [(leftMajorDep, [0, input.left.NUMBER])] by
      exact flow_leftMajor input _ _ _ _]
    rw [show flow input.leftMinor =
        [(leftMinorDep, [0, input.left.NUMBER])] by
      exact flow_leftMinor input _ _ _ _]
    rw [show edgesBetween graph input.rightMajor root =
        [rightMajorEdge input] by exact edges_rightMajor input]
    rw [show edgesBetween graph input.rightMinor root =
        [rightMinorEdge input] by exact edges_rightMinor input]
    rw [show edgesBetween graph input.leftMajor root =
        [leftMajorEdge input] by exact edges_leftMajor input]
    rw [show edgesBetween graph input.leftMinor root =
        [leftMinorEdge input] by exact edges_leftMinor input]
    simp [FlowSpec.consume, rightMajorEdge, rightMinorEdge,
      leftMajorEdge, leftMinorEdge, expectedRoutes]
  have separatedIntroductions :
      separatedIntroductionOutputs graph root flow = [] := by
    have flowRM : flow input.rightMajor =
        [(rightMajorDep, [0, input.right.NUMBER])] :=
      flow_rightMajor input _ _ _ _
    have flowRm : flow input.rightMinor =
        [(rightMinorDep, [0, input.right.NUMBER])] :=
      flow_rightMinor input _ _ _ _
    have flowLM : flow input.leftMajor =
        [(leftMajorDep, [0, input.left.NUMBER])] :=
      flow_leftMajor input _ _ _ _
    have flowLm : flow input.leftMinor =
        [(leftMinorDep, [0, input.left.NUMBER])] :=
      flow_leftMinor input _ _ _ _
    have edgeRM : edgesBetween graph input.rightMajor root =
        [rightMajorEdge input] := edges_rightMajor input
    have edgeRm : edgesBetween graph input.rightMinor root =
        [rightMinorEdge input] := edges_rightMinor input
    have edgeLM : edgesBetween graph input.leftMajor root =
        [leftMajorEdge input] := edges_leftMajor input
    have edgeLm : edgesBetween graph input.leftMinor root =
        [leftMinorEdge input] := edges_leftMinor input
    unfold separatedIntroductionOutputs
    rw [eliminationOutputs]
    unfold introductionSelections
    rw [predecessors]
    cases formulaShape : root.FORMULA with
    | atom atom => simp [formulaShape]
    | implication antecedent consequent =>
        by_cases hRM : input.rightMajor.FORMULA = consequent
        · by_cases hRm : input.rightMinor.FORMULA = consequent
          · by_cases hLM : input.leftMajor.FORMULA = consequent
            · by_cases hLm : input.leftMinor.FORMULA = consequent
              · simp [formulaShape, hRM, hRm, hLM, hLm,
                  selectionIntroductionOutputs, flowRM, flowRm, flowLM,
                  flowLm, edgeRM, edgeRm, edgeLM, edgeLm,
                  FlowSpec.consume, residualOccursB, expectedRoutes,
                  input.distinctNumbers]
              · simp [formulaShape, hRM, hRm, hLM, hLm,
                  selectionIntroductionOutputs, flowRM, flowRm, flowLM,
                  flowLm, edgeRM, edgeRm, edgeLM, edgeLm,
                  FlowSpec.consume, residualOccursB, expectedRoutes,
                  input.distinctNumbers]
            · by_cases hLm : input.leftMinor.FORMULA = consequent
              <;> simp [formulaShape, hRM, hRm, hLM, hLm,
                selectionIntroductionOutputs, flowRM, flowRm, flowLM,
                flowLm, edgeRM, edgeRm, edgeLM, edgeLm,
                FlowSpec.consume, residualOccursB, expectedRoutes,
                input.distinctNumbers]
          · by_cases hLM : input.leftMajor.FORMULA = consequent
            <;> by_cases hLm : input.leftMinor.FORMULA = consequent
            <;> simp [formulaShape, hRM, hRm, hLM, hLm,
              selectionIntroductionOutputs, flowRM, flowRm, flowLM,
              flowLm, edgeRM, edgeRm, edgeLM, edgeLm,
              FlowSpec.consume, residualOccursB, expectedRoutes,
              input.distinctNumbers]
        · by_cases hRm : input.rightMinor.FORMULA = consequent
          <;> by_cases hLM : input.leftMajor.FORMULA = consequent
          <;> by_cases hLm : input.leftMinor.FORMULA = consequent
          <;> simp [formulaShape, hRM, hRm, hLM, hLm,
            selectionIntroductionOutputs, flowRM, flowRm, flowLM,
            flowLm, edgeRM, edgeRm, edgeLM, edgeLm,
          FlowSpec.consume, residualOccursB, expectedRoutes,
          input.distinctNumbers]
  have routesNodup :
      (expectedRoutes input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep).eraseDups =
      expectedRoutes input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep := by
    have routeNe :
        (depUnion rightMinorDep rightMajorDep, [input.right.NUMBER]) ≠
          (depUnion leftMinorDep leftMajorDep, [input.left.NUMBER]) := by
      intro equal
      have colours := congrArg Prod.snd equal
      simp at colours
      exact input.distinctNumbers colours.symm
    have beqFalse :
        ((depUnion leftMinorDep leftMajorDep, [input.left.NUMBER]) ==
          (depUnion rightMinorDep rightMajorDep, [input.right.NUMBER])) =
            false := beq_eq_false_iff_ne.mpr routeNe.symm
    simp [expectedRoutes, List.eraseDups, List.eraseDupsBy,
      List.eraseDupsBy.loop, beqFalse]
  change occurrenceStep graph root flow = _
  simp [occurrenceStep, predecessors, noAncestors, rootNotSource,
    derivedOutputs, eliminationOutputs, separatedIntroductions]
  rw [routesNodup]
  simp [expectedRoutes]



def rightRoute {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep : Dep) :
    MixedRuleSpecification.SelectionRoute :=
  { production := .elimination
    dependency := depUnion rightMinorDep rightMajorDep
    residual := ⟨[input.right.NUMBER]⟩ }

def leftRoute {d : Graph} (input : Input d)
    (leftMajorDep leftMinorDep : Dep) :
    MixedRuleSpecification.SelectionRoute :=
  { production := .elimination
    dependency := depUnion leftMinorDep leftMajorDep
    residual := ⟨[input.left.NUMBER]⟩ }

def rightObservation {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep : Dep) : BoundaryObservation :=
  { production := .elimination
    conclusion := input.right.FORMULA
    use := input.rightUse
    dependency := depUnion rightMinorDep rightMajorDep
    residual := ⟨[]⟩ }

def leftObservation {d : Graph} (input : Input d)
    (leftMajorDep leftMinorDep : Dep) : BoundaryObservation :=
  { production := .elimination
    conclusion := input.left.FORMULA
    use := input.leftUse
    dependency := depUnion leftMinorDep leftMajorDep
    residual := ⟨[]⟩ }



def postBoundaryDeliveries {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    List BoundaryObservation :=
  boundaryDeliveries (center input).FORMULA
    [rightRoute input rightMajorDep rightMinorDep,
     leftRoute input leftMajorDep leftMinorDep]
    [rightOut input, leftOut input]



def preBoundaryDeliveries {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    List BoundaryObservation :=
  [leftObservation input leftMajorDep leftMinorDep,
   rightObservation input rightMajorDep rightMinorDep]

theorem postBoundaryDeliveries_exact
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    postBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep =
      [rightObservation input rightMajorDep rightMinorDep,
       leftObservation input leftMajorDep leftMinorDep] := by
  have reverseNumbers : input.right.NUMBER ≠ input.left.NUMBER :=
    input.distinctNumbers.symm
  simp [postBoundaryDeliveries, boundaryDeliveries, observeAcross,
    rightRoute, leftRoute, rightOut, leftOut, center, collapse.center,
    rightObservation, leftObservation,
    MixedRuleSpecification.consumeSelection,
    input.sameFormula, input.distinctNumbers, reverseNumbers]



theorem boundary_set_preserved
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    SelfCertifying.ListSetEq
      (preBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      (postBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep) := by
  rw [postBoundaryDeliveries_exact]
  intro observation
  simp [preBoundaryDeliveries, or_comm]



theorem selection_routes_project_to_successor
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    [(rightRoute input rightMajorDep rightMinorDep).dependency,
      (leftRoute input leftMajorDep leftMinorDep).dependency] =
      [depUnion rightMinorDep rightMajorDep,
       depUnion leftMinorDep leftMajorDep] ∧
    [(rightRoute input rightMajorDep rightMinorDep).residual.colours,
      (leftRoute input leftMajorDep leftMinorDep).residual.colours] =
      [[input.right.NUMBER], [input.left.NUMBER]] := by
  constructor <;> rfl




structure StepInvariant {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) : Prop where
  actualProducer : operation input = (producedNeighborhood input, [])
  independentlyRecomputed :
    occurrenceStep (localGraph input) (center input)
        (premiseFlow input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep) =
      .ok (expectedRoutes input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
  noLostOrSpuriousBoundary :
    SelfCertifying.ListSetEq
      (preBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      (postBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)

theorem actual_step_preserves
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    StepInvariant input rightMajorDep rightMinorDep
      leftMajorDep leftMinorDep :=
  { actualProducer := operation_eq_produced input
    independentlyRecomputed := local_occurrenceStep input _ _ _ _
    noLostOrSpuriousBoundary := boundary_set_preserved input _ _ _ _ }



private theorem operational_preparedLeft_ne_preparedRight
    {d : Graph} (input : Input d) :
    preparedLeft input ≠ preparedRight input := by
  intro equal
  have centers := congrArg Neighborhood.CENTER equal
  exact input.distinctCenters centers

private theorem operational_preparedLeft_ne_produced
    {d : Graph} (input : Input d) :
    preparedLeft input ≠ producedNeighborhood input := by
  intro equal
  have collapsed := congrArg (fun n : Neighborhood => n.CENTER.COLLAPSED) equal
  simp [preparedLeft, producedNeighborhood, center, collapse.center,
    input.leftUncollapsed] at collapsed

private theorem operational_preparedRight_ne_produced
    {d : Graph} (input : Input d) :
    preparedRight input ≠ producedNeighborhood input := by
  intro equal
  have collapsed := congrArg (fun n : Neighborhood => n.CENTER.COLLAPSED) equal
  simp [preparedRight, producedNeighborhood, center, collapse.center,
    input.rightUncollapsed] at collapsed




def operationalObserve {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (neighborhood : Neighborhood) : List BoundaryObservation :=
  if neighborhood = preparedLeft input then
    [leftObservation input leftMajorDep leftMinorDep]
  else if neighborhood = preparedRight input then
    [rightObservation input rightMajorDep rightMinorDep]
  else if neighborhood = producedNeighborhood input then
    postBoundaryDeliveries input rightMajorDep rightMinorDep
      leftMajorDep leftMinorDep
  else []

theorem operationalObserve_left
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep (preparedLeft input) =
      [leftObservation input leftMajorDep leftMinorDep] := by
  simp [operationalObserve]

theorem operationalObserve_right
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep (preparedRight input) =
      [rightObservation input rightMajorDep rightMinorDep] := by
  simp [operationalObserve,
    Ne.symm (operational_preparedLeft_ne_preparedRight input)]

theorem operationalObserve_produced
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep (producedNeighborhood input) =
      postBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep := by
  simp [operationalObserve,
    Ne.symm (operational_preparedLeft_ne_produced input),
    Ne.symm (operational_preparedRight_ne_produced input)]



theorem actual_two_rule_set_trace
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    Semantic.HCCompressionSequence.ActualCompressionSetTrace
      (hcPartition d)
      (operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      [pre_collapse (get_rule input.left d),
       pre_collapse (get_rule input.right d)]
      [] (producedNeighborhood input) [] := by
  rw [pre_collapse_left input, pre_collapse_right input]
  have operationPrepared :
      collapseWithOccurrences (hcPartition d)
          (preparedLeft input) (preparedRight input) =
        (producedNeighborhood input, []) := by
    simpa [operation, pre_collapse_left input, pre_collapse_right input] using
      operation_eq_produced input
  apply Semantic.HCCompressionSequence.ActualCompressionSetTrace.step
  · rw [operationPrepared]
    rw [operationalObserve_left, operationalObserve_right,
      operationalObserve_produced]
    exact boundary_set_preserved input _ _ _ _
  · rw [operationPrepared]
    simpa using
      (Semantic.HCCompressionSequence.ActualCompressionSetTrace.single
        (partition := hcPartition d)
        (observe := operationalObserve input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep)
        (occurrences := ([] : List Ancestral))
        (producedNeighborhood input))

theorem actual_two_rule_set_trace_with_occurrences
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (occurrences : List Ancestral) :
    Semantic.HCCompressionSequence.ActualCompressionSetTrace
      (hcPartition d)
      (operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      [pre_collapse (get_rule input.left d),
       pre_collapse (get_rule input.right d)]
      occurrences (producedNeighborhood input)
      (occurrences.map
        (collapse.rewrite_path_endpoints [input.left, input.right]
          (producedNeighborhood input).CENTER)) := by
  rw [pre_collapse_left input, pre_collapse_right input]
  have operationPrepared :
      collapseWithOccurrences (hcPartition d)
          (preparedLeft input) (preparedRight input) =
        (producedNeighborhood input, []) := by
    simpa [operation, pre_collapse_left input, pre_collapse_right input] using
      operation_eq_produced input
  apply Semantic.HCCompressionSequence.ActualCompressionSetTrace.step
  · rw [operationPrepared]
    rw [operationalObserve_left, operationalObserve_right,
      operationalObserve_produced]
    exact boundary_set_preserved input _ _ _ _
  · rw [operationPrepared]
    have centers :
        [(preparedLeft input).CENTER, (preparedRight input).CENTER] =
          [input.left, input.right] := by
      rfl
    rw [centers]
    simp only [List.append_nil]
    exact Semantic.HCCompressionSequence.ActualCompressionSetTrace.single
      (partition := hcPartition d)
      (observe := operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      (occurrences := occurrences.map
        (collapse.rewrite_path_endpoints [input.left, input.right]
          (producedNeighborhood input).CENTER))
      (producedNeighborhood input)




theorem actual_two_rule_operational_boundary_preserved
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    SelfCertifying.ListSetEq
      ([pre_collapse (get_rule input.left d),
        pre_collapse (get_rule input.right d)].flatMap
          (operationalObserve input rightMajorDep rightMinorDep
            leftMajorDep leftMinorDep))
      (operationalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep (producedNeighborhood input)) :=
  (actual_two_rule_set_trace input rightMajorDep rightMinorDep
    leftMajorDep leftMinorDep).boundary_preserved




def canonicalObserve {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (neighborhood : Neighborhood) : List BoundaryObservation :=
  if neighborhood = preparedLeft input then
    [leftObservation input leftMajorDep leftMinorDep]
  else if neighborhood = preparedRight input then
    [rightObservation input rightMajorDep rightMinorDep]
  else if neighborhood = producedNeighborhood input then
    [leftObservation input leftMajorDep leftMinorDep,
     rightObservation input rightMajorDep rightMinorDep]
  else []

private theorem preparedLeft_ne_preparedRight
    {d : Graph} (input : Input d) :
    preparedLeft input ≠ preparedRight input := by
  intro equal
  have centers := congrArg Neighborhood.CENTER equal
  exact input.distinctCenters centers

private theorem preparedLeft_ne_produced
    {d : Graph} (input : Input d) :
    preparedLeft input ≠ producedNeighborhood input := by
  intro equal
  have collapsed := congrArg (fun n : Neighborhood => n.CENTER.COLLAPSED) equal
  simp [preparedLeft, producedNeighborhood, center, collapse.center,
    input.leftUncollapsed] at collapsed

private theorem preparedRight_ne_produced
    {d : Graph} (input : Input d) :
    preparedRight input ≠ producedNeighborhood input := by
  intro equal
  have collapsed := congrArg (fun n : Neighborhood => n.CENTER.COLLAPSED) equal
  simp [preparedRight, producedNeighborhood, center, collapse.center,
    input.rightUncollapsed] at collapsed

theorem canonicalObserve_left
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    canonicalObserve input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        (preparedLeft input) =
      [leftObservation input leftMajorDep leftMinorDep] := by
  simp [canonicalObserve]

theorem canonicalObserve_right
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    canonicalObserve input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        (preparedRight input) =
      [rightObservation input rightMajorDep rightMinorDep] := by
  simp [canonicalObserve, Ne.symm (preparedLeft_ne_preparedRight input)]

theorem canonicalObserve_produced
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    canonicalObserve input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
        (producedNeighborhood input) =
      [leftObservation input leftMajorDep leftMinorDep,
       rightObservation input rightMajorDep rightMinorDep] := by
  simp [canonicalObserve, Ne.symm (preparedLeft_ne_produced input),
    Ne.symm (preparedRight_ne_produced input)]

theorem canonicalObserve_before_eq_pre
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    canonicalObserve input rightMajorDep rightMinorDep leftMajorDep leftMinorDep
          (preparedLeft input) ++
        canonicalObserve input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep (preparedRight input) =
      preBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep := by
  rw [canonicalObserve_left, canonicalObserve_right]
  rfl




theorem canonicalObserve_after_setEq_post
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    SelfCertifying.ListSetEq
      (canonicalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep (producedNeighborhood input))
      (postBoundaryDeliveries input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep) := by
  rw [canonicalObserve_produced, postBoundaryDeliveries_exact]
  intro observation
  simp [or_comm]




theorem actual_two_rule_trace
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    Semantic.HCCompressionSequence.ActualCompressionTrace
      (hcPartition d)
      (canonicalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      [pre_collapse (get_rule input.left d),
       pre_collapse (get_rule input.right d)]
      [] (producedNeighborhood input) [] := by
  rw [pre_collapse_left input, pre_collapse_right input]
  have operationPrepared :
      collapseWithOccurrences (hcPartition d)
          (preparedLeft input) (preparedRight input) =
        (producedNeighborhood input, []) := by
    simpa [operation, pre_collapse_left input, pre_collapse_right input] using
      operation_eq_produced input
  apply Semantic.HCCompressionSequence.ActualCompressionTrace.step
  · rw [operationPrepared]
    simp [canonicalObserve_left, canonicalObserve_right,
      canonicalObserve_produced]
  · rw [operationPrepared]
    simpa using
      (Semantic.HCCompressionSequence.ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := canonicalObserve input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep)
        (occurrences := ([] : List Ancestral))
        (producedNeighborhood input))




theorem actual_two_rule_trace_with_occurrences
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (occurrences : List Ancestral) :
    Semantic.HCCompressionSequence.ActualCompressionTrace
      (hcPartition d)
      (canonicalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      [pre_collapse (get_rule input.left d),
       pre_collapse (get_rule input.right d)]
      occurrences (producedNeighborhood input)
      (occurrences.map
        (collapse.rewrite_path_endpoints [input.left, input.right]
          (producedNeighborhood input).CENTER)) := by
  rw [pre_collapse_left input, pre_collapse_right input]
  have operationPrepared :
      collapseWithOccurrences (hcPartition d)
          (preparedLeft input) (preparedRight input) =
        (producedNeighborhood input, []) := by
    simpa [operation, pre_collapse_left input, pre_collapse_right input] using
      operation_eq_produced input
  apply Semantic.HCCompressionSequence.ActualCompressionTrace.step
  · rw [operationPrepared]
    simp [canonicalObserve_left, canonicalObserve_right,
      canonicalObserve_produced]
  · rw [operationPrepared]
    have centers :
        [(preparedLeft input).CENTER, (preparedRight input).CENTER] =
          [input.left, input.right] := by
      rfl
    rw [centers]
    simp only [List.append_nil]
    exact Semantic.HCCompressionSequence.ActualCompressionTrace.single
      (partition := hcPartition d)
      (observe := canonicalObserve input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep)
      (occurrences := occurrences.map
        (collapse.rewrite_path_endpoints [input.left, input.right]
          (producedNeighborhood input).CENTER))
      (producedNeighborhood input)



theorem actual_two_rule_execution
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep) :
    compressWithOccurrences (hcPartition d)
      [pre_collapse (get_rule input.left d),
       pre_collapse (get_rule input.right d)] [] =
      (producedNeighborhood input, []) := by
  exact (actual_two_rule_trace input rightMajorDep rightMinorDep
    leftMajorDep leftMinorDep).refines_actual_compressor




theorem actual_two_node_graph_replacement
    {d : Graph} (input : Input d)
    (rightMajorDep rightMinorDep leftMajorDep leftMinorDep : Dep)
    (occurrences : List Ancestral) :
    compress_nodesWithPartitionCarrying (hcPartition d)
        [input.left, input.right] d occurrences =
      let materialized := materializeCarriedOwners
        [input.left, input.right] d occurrences
      let rewritten := materialized.map
        (collapse.rewrite_path_endpoints [input.left, input.right]
          (producedNeighborhood input).CENTER)
      let graph := collapse.rewrite_graph_path_endpoints
        [input.left, input.right] (producedNeighborhood input).CENTER
        (set_rule (pop_rules d (get_rules [input.left, input.right] d))
          (producedNeighborhood input))
      let graph := dlds graph.NODES graph.EDGES
        (ensureOccurrenceRecords graph.PATHS rewritten)
      (checkAncestralPathHeads graph, rewritten) := by
  have trace :
      Semantic.HCCompressionSequence.ActualCompressionTrace
        (hcPartition d)
        (canonicalObserve input rightMajorDep rightMinorDep
          leftMajorDep leftMinorDep)
        (pre_compress (get_rules [input.left, input.right] d)) []
        (producedNeighborhood input) [] := by
    simpa [get_rules, pre_compress] using
      actual_two_rule_trace input rightMajorDep rightMinorDep
        leftMajorDep leftMinorDep
  simpa using trace.refines_compress_nodesWithPartitionCarrying

end HCDerivedElimPairPreservation
