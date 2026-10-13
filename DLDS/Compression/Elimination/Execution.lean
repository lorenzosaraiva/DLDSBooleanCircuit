import DLDS.Compression.Elimination.ClassPreservation

set_option linter.unusedSimpArgs false

/-!
# Operational replay of an accumulated elimination class

This module connects the exact neighbourhood produced for an arbitrary
homogeneous elimination class to the real successor `occurrenceStep`.  The
premise table is a proof-side input contract: dependencies are computed from
the represented logical occurrences and are not read from stored deduction
payloads.
-/

open DLDS FlowSemantics
open Evaluator.Mixed Evaluator.Occurrence
open DLDS.Compression.Mixed.LocalRules DLDS.Compression.Mixed.ContextualElimination
open DLDS.Compression.Elimination.ClassPreservation

namespace DLDS.Compression.Elimination.Execution

/-- The smallest graph containing the exact accumulated producer result. -/
def accumulatedGraph (center : Vertex) (occurrences : List ElimOccurrence) :
    Graph :=
  DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph
    (accumulatedNeighborhood center occurrences)

/-- Independently specified operational result, before crossing an outgoing
occurrence-coloured edge. -/
def expectedFlowRoutes (occurrences : List ElimOccurrence) : List FlowPair :=
  occurrences.map fun occurrence =>
    (depUnion occurrence.minorDep occurrence.majorDep, [occurrence.tag])

/-- All independently specified premise routes carried at a vertex.  A
compressed premise may represent more than one occurrence, so this is a list
of all major/minor contributions rather than a singleton lookup. -/
def expectedPremiseRoutes (occurrences : List ElimOccurrence) (v : Vertex) :
    List FlowPair :=
  occurrences.flatMap fun occurrence =>
    (if v = occurrence.major then
      [(occurrence.majorDep, [0, occurrence.tag])]
    else []) ++
    (if v = occurrence.minor then
      [(occurrence.minorDep, [0, occurrence.tag])]
    else [])

/-- Legitimate caller-side induction hypothesis for this local step.  It
constrains only routes at represented premises and is independent of the
step's output. -/
def PremiseRoutesCorrect (premiseFlow : Vertex -> List FlowPair)
    (occurrences : List ElimOccurrence) : Prop :=
  forall v, Certificate.Flow.ListSetEq (premiseFlow v)
    (expectedPremiseRoutes occurrences v)

def PremiseOf (occurrence : ElimOccurrence) (v : Vertex) : Prop :=
  v = occurrence.major \/ v = occurrence.minor

/-- Membership form of the independently specified premise table. -/
theorem mem_expectedPremiseRoutes_iff
    (occurrences : List ElimOccurrence) (v : Vertex) (route : FlowPair) :
    route ∈ expectedPremiseRoutes occurrences v <->
      Exists fun occurrence =>
        occurrence ∈ occurrences /\
          ((v = occurrence.major /\
              route = (occurrence.majorDep, [0, occurrence.tag])) \/
           (v = occurrence.minor /\
              route = (occurrence.minorDep, [0, occurrence.tag]))) := by
  simp [expectedPremiseRoutes, and_or_left]

def PremisesAvoidCenter (center : Vertex)
    (occurrences : List ElimOccurrence) : Prop :=
  forall occurrence, occurrence ∈ occurrences ->
    occurrence.major ≠ center /\ occurrence.minor ≠ center

def UsesAvoidCenter (center : Vertex)
    (occurrences : List ElimOccurrence) : Prop :=
  forall occurrence, occurrence ∈ occurrences -> occurrence.use ≠ center

theorem accumulated_incoming
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences) :
    get_rule.incoming center (accumulatedGraph center occurrences) =
      occurrences.flatMap (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming center) := by
  simp only [accumulatedGraph,
    DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph,
    accumulatedNeighborhood, List.nil_append]
  rw [FlowSemantics.incoming_eq_filter]
  rw [List.filter_append]
  have incomingFilter :
      (occurrences.flatMap (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming center)).filter
          (fun edge => decide (edge.END = center)) =
        occurrences.flatMap (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming center) := by
    apply List.filter_eq_self.mpr
    intro candidate candidateMember
    obtain ⟨occurrence, occurrenceMember, edgeMember⟩ :=
      List.mem_flatMap.mp candidateMember
    simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming] at edgeMember
    rcases edgeMember with rfl | rfl <;> simp
  have outgoingFilter :
      (occurrenceOutgoing center occurrences).filter
          (fun edge => decide (edge.END = center)) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro candidate candidateMember
    obtain ⟨occurrence, occurrenceMember, edgeEqual⟩ :=
      List.mem_map.mp candidateMember
    subst candidate
    simpa [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing] using
      usesAvoid occurrence occurrenceMember
  rw [incomingFilter, outgoingFilter, List.append_nil]

theorem accumulated_no_center_ancestors
    (center : Vertex) (occurrences : List ElimOccurrence)
    (premisesAvoid : PremisesAvoidCenter center occurrences) :
    ancestorsInto (accumulatedGraph center occurrences) center = [] := by
  unfold ancestorsInto accumulatedGraph
    DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph
  simp only [accumulatedNeighborhood, List.nil_append]
  induction occurrences with
  | nil => simp
  | cons occurrence rest ih =>
      have headAvoid := premisesAvoid occurrence (List.Mem.head rest)
      have tailAvoid : PremisesAvoidCenter center rest := by
        intro candidate member
        exact premisesAvoid candidate (List.Mem.tail occurrence member)
      simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.indirect, headAvoid.1, headAvoid.2,
        ih tailAvoid]

/-- Exact predecessor carrier of the accumulated centre.  Multiplicity is
discarded by `predsOf`, but no premise vertex is lost. -/
theorem mem_accumulated_preds_iff
    (center u : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences) :
    u ∈ predsOf (accumulatedGraph center occurrences) center <->
      Exists fun occurrence => occurrence ∈ occurrences /\
        PremiseOf occurrence u := by
  rw [FlowSemantics.predsOf_eq,
    accumulated_incoming center occurrences usesAvoid]
  simp only [List.mem_eraseDups, List.mem_map, List.mem_flatMap]
  constructor
  · rintro ⟨candidate, ⟨occurrence, occurrenceMember, edgeMember⟩,
        startEqual⟩
    simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming] at edgeMember
    rcases edgeMember with rfl | rfl
    · exact ⟨occurrence, occurrenceMember,
        Or.inl startEqual.symm⟩
    · exact ⟨occurrence, occurrenceMember,
        Or.inr startEqual.symm⟩
  · rintro ⟨occurrence, occurrenceMember, major | minor⟩
    · subst u
      exact ⟨edge occurrence.major center 0 occurrence.majorStored,
        ⟨occurrence, occurrenceMember,
          by simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming]⟩, rfl⟩
    · subst u
      exact ⟨edge occurrence.minor center 0 occurrence.minorStored,
        ⟨occurrence, occurrenceMember,
          by simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming]⟩, rfl⟩

/-- Every actual edge into an accumulated centre is a colour-zero premise
edge generated by one represented occurrence. -/
theorem accumulated_edgesBetween_colour_zero
    (center u : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (candidate : Deduction)
    (member : candidate ∈
      edgesBetween (accumulatedGraph center occurrences) u center) :
    candidate.COLOUR = 0 := by
  have incomingMember : candidate ∈
      get_rule.incoming center (accumulatedGraph center occurrences) := by
    rw [FlowSemantics.incoming_eq_filter]
    unfold edgesBetween at member
    simp only [List.mem_filter] at member ⊢
    have endpoints : candidate.START = u /\ candidate.END = center :=
      of_decide_eq_true member.2
    exact ⟨member.1, by simp [endpoints.2]⟩
  rw [accumulated_incoming center occurrences usesAvoid] at incomingMember
  obtain ⟨occurrence, occurrenceMember, edgeMember⟩ :=
    List.mem_flatMap.mp incomingMember
  simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming] at edgeMember
  rcases edgeMember with rfl | rfl <;> rfl

theorem accumulated_major_edge_mem
    (center : Vertex) (occurrences : List ElimOccurrence)
    (occurrence : ElimOccurrence) (member : occurrence ∈ occurrences) :
    edge occurrence.major center 0 occurrence.majorStored ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.major center := by
  unfold edgesBetween accumulatedGraph
    DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph accumulatedNeighborhood
  simp only [List.nil_append, List.mem_filter]
  constructor
  · apply List.mem_append_left
    exact List.mem_flatMap.mpr
      ⟨occurrence, member, by simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming]⟩
  · simp

theorem accumulated_minor_edge_mem
    (center : Vertex) (occurrences : List ElimOccurrence)
    (occurrence : ElimOccurrence) (member : occurrence ∈ occurrences) :
    edge occurrence.minor center 0 occurrence.minorStored ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.minor center := by
  unfold edgesBetween accumulatedGraph
    DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph accumulatedNeighborhood
  simp only [List.nil_append, List.mem_filter]
  constructor
  · apply List.mem_append_left
    exact List.mem_flatMap.mpr
      ⟨occurrence, member, by simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming]⟩
  · simp

theorem mem_elimPairsAt_iff
    (d : Graph) (v major minor : Vertex) :
    (major, minor) ∈ elimPairsAt d v <->
      major ∈ predsOf d v /\ minor ∈ predsOf d v /\
        major.FORMULA = minor.FORMULA >> v.FORMULA := by
  unfold elimPairsAt
  simp only [List.mem_flatMap, List.mem_filterMap]
  constructor
  · rintro ⟨candidateMajor, majorMember, candidateMinor,
      minorMember, filtered⟩
    split at filtered
    · simp only [Option.some.injEq, Prod.mk.injEq] at filtered
      rcases filtered with ⟨majorEqual, minorEqual⟩
      subst candidateMajor
      subst candidateMinor
      exact ⟨majorMember, minorMember, ‹_›⟩
    · contradiction
  · rintro ⟨majorMember, minorMember, formula⟩
    exact ⟨major, majorMember, minor, minorMember, by simp [formula]⟩

/-- Complete membership characterization of the actual `elimPairsAt`
enumeration on an accumulated class.  It deliberately permits shared premise
vertices, duplicate formula labels, and formula-compatible cross members. -/
theorem mem_accumulated_elimPairsAt_iff
    (center major minor : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences) :
    (major, minor) ∈
        elimPairsAt (accumulatedGraph center occurrences) center <->
      (Exists fun majorOccurrence =>
        majorOccurrence ∈ occurrences /\
          PremiseOf majorOccurrence major) /\
      (Exists fun minorOccurrence =>
        minorOccurrence ∈ occurrences /\
          PremiseOf minorOccurrence minor) /\
      major.FORMULA = minor.FORMULA >> center.FORMULA := by
  rw [mem_elimPairsAt_iff]
  rw [mem_accumulated_preds_iff center major occurrences usesAvoid]
  rw [mem_accumulated_preds_iff center minor occurrences usesAvoid]

private theorem same_occurrence_formula_candidate
    (center major minor : Vertex) (occurrence : ElimOccurrence)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA)
    (majorPremise : PremiseOf occurrence major)
    (minorPremise : PremiseOf occurrence minor)
    (formula : major.FORMULA = minor.FORMULA >> center.FORMULA) :
    major = occurrence.major /\ minor = occurrence.minor := by
  rcases majorPremise with majorEq | majorEq <;>
    rcases minorPremise with minorEq | minorEq
  · subst major
    subst minor
    exact False.elim
      (FlowSemantics.formula_ne_implication_self occurrence.major.FORMULA
        center.FORMULA formula)
  · exact ⟨majorEq, minorEq⟩
  · subst major
    subst minor
    exact False.elim (FlowSemantics.not_mutual_implication logical formula)
  · subst major
    subst minor
    exact False.elim
      (FlowSemantics.formula_ne_implication_self occurrence.minor.FORMULA
        center.FORMULA formula)

/-- Every actual candidate is either the intended orientation of one member,
or a cross-member candidate.  This is exhaustive; no formula-separation
assumption is used. -/
theorem accumulated_elimPair_classification
    (center major minor : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (candidate : (major, minor) ∈
      elimPairsAt (accumulatedGraph center occurrences) center) :
    (Exists fun occurrence => occurrence ∈ occurrences /\
      major = occurrence.major /\ minor = occurrence.minor) \/
    (Exists fun left => Exists fun right =>
      left ∈ occurrences /\ right ∈ occurrences /\ left ≠ right /\
        PremiseOf left major /\ PremiseOf right minor) := by
  have characterized :=
    (mem_accumulated_elimPairsAt_iff center major minor occurrences
      usesAvoid).mp candidate
  obtain ⟨⟨left, leftMember, leftPremise⟩,
    ⟨right, rightMember, rightPremise⟩, formula⟩ := characterized
  by_cases same : left = right
  · subst right
    have orientation := same_occurrence_formula_candidate center major minor
      left (logical left leftMember) leftPremise rightPremise formula
    exact Or.inl ⟨left, leftMember, orientation⟩
  · exact Or.inr ⟨left, right, leftMember, rightMember, same,
      leftPremise, rightPremise⟩

/-- Every represented logical member occurs in the real enumeration. -/
theorem intended_elimPair_mem
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (occurrence : ElimOccurrence) (member : occurrence ∈ occurrences)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA) :
    (occurrence.major, occurrence.minor) ∈
      elimPairsAt (accumulatedGraph center occurrences) center := by
  apply (mem_accumulated_elimPairsAt_iff center occurrence.major
    occurrence.minor occurrences usesAvoid).mpr
  exact ⟨⟨occurrence, member, Or.inl rfl⟩,
    ⟨occurrence, member, Or.inr rfl⟩, logical⟩

theorem candidateEliminationOutputs_eq_pairOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair) :
    candidateEliminationOutputs d v premiseFlow =
      (elimPairsAt d v).flatMap
        (DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs
          d v premiseFlow) := by
  rfl

private theorem zero_pair_result_iff
    (minorEdge majorEdge : Deduction)
    (minorZero : minorEdge.COLOUR = 0)
    (majorZero : majorEdge.COLOUR = 0)
    (minorDep majorDep : Dep) (minorTag majorTag : Nat)
    (output : FlowPair) :
    (match FlowSemantics.consume minorEdge [0, minorTag],
        FlowSemantics.consume majorEdge [0, majorTag] with
      | some minorTail, some majorTail =>
          if minorTail = majorTail then
            some (depUnion minorDep majorDep, minorTail)
          else none
      | _, _ => none) = some output <->
      minorTag = majorTag /\
        output = (depUnion minorDep majorDep, [minorTag]) := by
  simp [FlowSemantics.consume, minorZero, majorZero, eq_comm]

private theorem major_ne_minor_of_logical
    (center : Vertex) (occurrence : ElimOccurrence)
    (logical : occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> center.FORMULA) :
    occurrence.major ≠ occurrence.minor := by
  intro equal
  have formulaEqual := congrArg Vertex.FORMULA equal
  rw [logical] at formulaEqual
  exact FlowSemantics.formula_ne_implication_self occurrence.minor.FORMULA
    center.FORMULA formulaEqual.symm

/-- No route produced by the full Cartesian enumeration lies outside the
independently specified family.  Cross-member route combinations have
unequal tags and therefore contribute no member. -/
theorem candidateEliminationOutputs_sound
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (tagsInjective : TagsInjectiveOn occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (produced : output ∈ candidateEliminationOutputs
      (accumulatedGraph center occurrences) center premiseFlow) :
    output ∈ expectedFlowRoutes occurrences := by
  rw [candidateEliminationOutputs_eq_pairOutputs] at produced
  obtain ⟨pair, pairMember, pairProduced⟩ := List.mem_flatMap.mp produced
  rcases pair with ⟨major, minor⟩
  unfold DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs at pairProduced
  obtain ⟨minorRoute, minorRouteMember, afterMinor⟩ :=
    List.mem_flatMap.mp pairProduced
  obtain ⟨majorRoute, majorRouteMember, afterMajor⟩ :=
    List.mem_flatMap.mp afterMinor
  obtain ⟨minorEdge, minorEdgeMember, afterMinorEdge⟩ :=
    List.mem_flatMap.mp afterMajor
  obtain ⟨majorEdge, majorEdgeMember, resultEquation⟩ :=
    List.mem_filterMap.mp afterMinorEdge
  have minorExpected := (premiseCorrect minor minorRoute).mp minorRouteMember
  have majorExpected := (premiseCorrect major majorRoute).mp majorRouteMember
  rw [mem_expectedPremiseRoutes_iff] at minorExpected majorExpected
  obtain ⟨minorOccurrence, minorOccurrenceMember,
      minorOrigin⟩ := minorExpected
  obtain ⟨majorOccurrence, majorOccurrenceMember,
      majorOrigin⟩ := majorExpected
  have minorZero := accumulated_edgesBetween_colour_zero center minor
    occurrences usesAvoid minorEdge minorEdgeMember
  have majorZero := accumulated_edgesBetween_colour_zero center major
    occurrences usesAvoid majorEdge majorEdgeMember
  have pairFormula := (mem_elimPairsAt_iff
    (accumulatedGraph center occurrences) center major minor).mp pairMember |>.2.2
  rcases minorOrigin with minorOrigin | minorOrigin <;>
    rcases majorOrigin with majorOrigin | majorOrigin
  all_goals
    rcases minorOrigin with ⟨minorVertex, minorRouteEqual⟩
    rcases majorOrigin with ⟨majorVertex, majorRouteEqual⟩
    rw [minorRouteEqual, majorRouteEqual] at resultEquation
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.majorDep majorOccurrence.majorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_candidate center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inl majorVertex) (Or.inl minorVertex) pairFormula
    have impossible := major_ne_minor_of_logical center minorOccurrence
      (logical minorOccurrence minorOccurrenceMember)
    exact False.elim (impossible (minorVertex.symm.trans orientation.2))
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.majorDep majorOccurrence.minorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_candidate center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inr majorVertex) (Or.inl minorVertex) pairFormula
    have impossible := major_ne_minor_of_logical center minorOccurrence
      (logical minorOccurrence minorOccurrenceMember)
    exact False.elim (impossible (minorVertex.symm.trans orientation.2))
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.minorDep majorOccurrence.majorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_candidate center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inl majorVertex) (Or.inr minorVertex) pairFormula
    rw [result.2]
    exact List.mem_map.mpr ⟨minorOccurrence, minorOccurrenceMember, rfl⟩
  · have result := (zero_pair_result_iff minorEdge majorEdge minorZero
        majorZero minorOccurrence.minorDep majorOccurrence.minorDep
        minorOccurrence.tag majorOccurrence.tag output).mp resultEquation
    have sameOccurrence := tagsInjective minorOccurrence
      minorOccurrenceMember majorOccurrence majorOccurrenceMember result.1
    subst majorOccurrence
    have orientation := same_occurrence_formula_candidate center major minor
      minorOccurrence (logical minorOccurrence minorOccurrenceMember)
      (Or.inr majorVertex) (Or.inr minorVertex) pairFormula
    have impossible := major_ne_minor_of_logical center minorOccurrence
      (logical minorOccurrence minorOccurrenceMember)
    exact False.elim (impossible (orientation.1.symm.trans majorVertex))

/-- Every independently specified matching member is emitted by the full
actual enumeration. -/
theorem candidateEliminationOutputs_complete
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (expected : output ∈ expectedFlowRoutes occurrences) :
    output ∈ candidateEliminationOutputs
      (accumulatedGraph center occurrences) center premiseFlow := by
  obtain ⟨occurrence, occurrenceMember, outputEqual⟩ :=
    List.mem_map.mp expected
  subst output
  have pairMember := intended_elimPair_mem center occurrences usesAvoid
    occurrence occurrenceMember (logical occurrence occurrenceMember)
  have majorExpected :
      (occurrence.majorDep, [0, occurrence.tag]) ∈
        expectedPremiseRoutes occurrences occurrence.major := by
    apply (mem_expectedPremiseRoutes_iff occurrences occurrence.major _).mpr
    exact ⟨occurrence, occurrenceMember, Or.inl ⟨rfl, rfl⟩⟩
  have minorExpected :
      (occurrence.minorDep, [0, occurrence.tag]) ∈
        expectedPremiseRoutes occurrences occurrence.minor := by
    apply (mem_expectedPremiseRoutes_iff occurrences occurrence.minor _).mpr
    exact ⟨occurrence, occurrenceMember, Or.inr ⟨rfl, rfl⟩⟩
  have majorRoute := (premiseCorrect occurrence.major _).mpr majorExpected
  have minorRoute := (premiseCorrect occurrence.minor _).mpr minorExpected
  let majorEdge := edge occurrence.major center 0 occurrence.majorStored
  let minorEdge := edge occurrence.minor center 0 occurrence.minorStored
  have majorEdgeMember : majorEdge ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.major center := by
    exact accumulated_major_edge_mem center occurrences occurrence
      occurrenceMember
  have minorEdgeMember : minorEdge ∈
      edgesBetween (accumulatedGraph center occurrences)
        occurrence.minor center := by
    exact accumulated_minor_edge_mem center occurrences occurrence
      occurrenceMember
  rw [candidateEliminationOutputs_eq_pairOutputs]
  apply List.mem_flatMap.mpr
  refine ⟨(occurrence.major, occurrence.minor), pairMember, ?_⟩
  unfold DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs
  apply List.mem_flatMap.mpr
  refine ⟨(occurrence.minorDep, [0, occurrence.tag]), minorRoute, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨(occurrence.majorDep, [0, occurrence.tag]), majorRoute, ?_⟩
  apply List.mem_flatMap.mpr
  refine ⟨minorEdge, minorEdgeMember, ?_⟩
  apply List.mem_filterMap.mpr
  refine ⟨majorEdge, majorEdgeMember, ?_⟩
  exact (zero_pair_result_iff minorEdge majorEdge (by rfl) (by rfl)
    occurrence.minorDep occurrence.majorDep occurrence.tag occurrence.tag
    (depUnion occurrence.minorDep occurrence.majorDep,
      [occurrence.tag])).mpr ⟨rfl, rfl⟩

/-- Exact set semantics of the whole actual `elimPairsAt`/Cartesian kernel.
List order and duplicate derivations are intentionally abstracted, while
route identity remains visible in the residual occurrence tag. -/
theorem candidateEliminationOutputs_setEq_expected
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (distinct : PairwiseDistinctTags occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    Certificate.Flow.ListSetEq
      (candidateEliminationOutputs
        (accumulatedGraph center occurrences) center premiseFlow)
      (expectedFlowRoutes occurrences) := by
  intro output
  constructor
  · exact candidateEliminationOutputs_sound center occurrences usesAvoid
      logical (pairwiseDistinctTags_injective occurrences distinct)
      premiseFlow premiseCorrect output
  · exact candidateEliminationOutputs_complete center occurrences usesAvoid
      logical premiseFlow premiseCorrect output

private theorem introductionCandidate_has_occurrence_residual
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences)
    (output : FlowPair)
    (member : output ∈ introductionCandidates
      (accumulatedGraph center occurrences) center premiseFlow) :
    Exists fun occurrence => occurrence ∈ occurrences /\
      Exists fun dependency => output = (dependency, [occurrence.tag]) := by
  unfold introductionCandidates at member
  cases formulaShape : center.FORMULA with
  | atom symbol => simp [formulaShape] at member
  | implication antecedent consequent =>
      simp only [formulaShape, List.mem_flatMap] at member
      obtain ⟨premise, premiseMember, candidateMember⟩ := member
      by_cases premiseFormula : premise.FORMULA = consequent
      · simp only [premiseFormula, ↓reduceIte] at candidateMember
        unfold candidateIntroductionOutputs at candidateMember
        rw [formulaShape] at candidateMember
        simp only [premiseFormula, ↓reduceIte] at candidateMember
        obtain ⟨premiseRoute, premiseRouteMember, afterRoute⟩ :=
          List.mem_flatMap.mp candidateMember
        obtain ⟨premiseEdge, premiseEdgeMember, resultEquation⟩ :=
          List.mem_filterMap.mp afterRoute
        have expectedPremise :=
          (premiseCorrect premise premiseRoute).mp premiseRouteMember
        rw [mem_expectedPremiseRoutes_iff] at expectedPremise
        obtain ⟨occurrence, occurrenceMember, origin⟩ := expectedPremise
        have edgeZero := accumulated_edgesBetween_colour_zero center premise
          occurrences usesAvoid premiseEdge premiseEdgeMember
        rcases origin with origin | origin
        · rcases origin with ⟨premiseEqual, routeEqual⟩
          rw [routeEqual] at resultEquation
          simp [FlowSemantics.consume, edgeZero] at resultEquation
          exact ⟨occurrence, occurrenceMember,
            depRemove occurrence.majorDep antecedent, resultEquation.symm⟩
        · rcases origin with ⟨premiseEqual, routeEqual⟩
          rw [routeEqual] at resultEquation
          simp [FlowSemantics.consume, edgeZero] at resultEquation
          exact ⟨occurrence, occurrenceMember,
            depRemove occurrence.minorDep antecedent, resultEquation.symm⟩
      · simp [premiseFormula] at candidateMember

/-- Every introduction candidate has a residual already represented by its
matching elimination output, so the real continuation-separation filter
removes all competing introduction productions. -/
theorem separatedIntroductionOutputs_eq_nil
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    separatedIntroductionOutputs (accumulatedGraph center occurrences)
      center premiseFlow = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro output outputMember
  obtain ⟨occurrence, occurrenceMember, dependency, outputEqual⟩ :=
    introductionCandidate_has_occurrence_residual center occurrences usesAvoid
      premiseFlow premiseCorrect output outputMember
  have expectedMember :
      (depUnion occurrence.minorDep occurrence.majorDep,
        [occurrence.tag]) ∈ expectedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨occurrence, occurrenceMember, rfl⟩
  have eliminationMember := candidateEliminationOutputs_complete center
    occurrences usesAvoid logical premiseFlow premiseCorrect _ expectedMember
  have occurs : residualOccursB
      (candidateEliminationOutputs (accumulatedGraph center occurrences)
        center premiseFlow) output.2 = true := by
    unfold residualOccursB
    rw [List.any_eq_true]
    refine ⟨(depUnion occurrence.minorDep occurrence.majorDep,
      [occurrence.tag]), eliminationMember, ?_⟩
    simp [outputEqual]
  simp [separatedIntroductionOutputs, outputMember, occurs]

theorem derivedOutputs_setEq_expected
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (distinct : PairwiseDistinctTags occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    Certificate.Flow.ListSetEq
      (derivedOutputs (accumulatedGraph center occurrences)
        center premiseFlow)
      (expectedFlowRoutes occurrences) := by
  have eliminations := candidateEliminationOutputs_setEq_expected center
    occurrences usesAvoid logical distinct premiseFlow premiseCorrect
  have introductions := separatedIntroductionOutputs_eq_nil center
    occurrences usesAvoid logical premiseFlow premiseCorrect
  intro output
  rw [show derivedOutputs (accumulatedGraph center occurrences) center
      premiseFlow =
        (candidateEliminationOutputs (accumulatedGraph center occurrences)
          center premiseFlow).eraseDups by
    simp [derivedOutputs, introductions]]
  rw [List.mem_eraseDups]
  exact eliminations output

/-- The complete local operational theorem.  It proves every guard of the
real `occurrenceStep`, including non-leaf classification, absence of
unsupported reseeding, nonempty logical production, and removal of competing
introduction outputs. -/
theorem accumulated_occurrenceStep
    (center : Vertex) (occurrences : List ElimOccurrence)
    (nonempty : occurrences ≠ [])
    (centerDerived : center.HYPOTHESIS = false)
    (premisesAvoid : PremisesAvoidCenter center occurrences)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (logical : forall occurrence, occurrence ∈ occurrences ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> center.FORMULA)
    (distinct : PairwiseDistinctTags occurrences)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow occurrences) :
    Exists fun outputs =>
      occurrenceStep (accumulatedGraph center occurrences) center
          premiseFlow = .ok outputs /\
        Certificate.Flow.ListSetEq outputs
          (expectedFlowRoutes occurrences) := by
  have derivedSet := derivedOutputs_setEq_expected center occurrences
    usesAvoid logical distinct premiseFlow premiseCorrect
  obtain ⟨occurrence, rest, occurrenceList⟩ := List.exists_cons_of_ne_nil nonempty
  have occurrenceMember : occurrence ∈ occurrences := by
    rw [occurrenceList]
    exact List.Mem.head rest
  have predecessorMember : occurrence.major ∈
      predsOf (accumulatedGraph center occurrences) center := by
    apply (mem_accumulated_preds_iff center occurrence.major occurrences
      usesAvoid).mpr
    exact ⟨occurrence, occurrenceMember, Or.inl rfl⟩
  have predecessorsNonempty :
      (predsOf (accumulatedGraph center occurrences) center).isEmpty = false := by
    have ne : predsOf (accumulatedGraph center occurrences) center ≠ [] := by
      intro empty
      rw [empty] at predecessorMember
      exact List.not_mem_nil predecessorMember
    simpa [List.isEmpty_iff] using ne
  have noAncestors := accumulated_no_center_ancestors center occurrences
    premisesAvoid
  have expectedMember :
      (depUnion occurrence.minorDep occurrence.majorDep,
        [occurrence.tag]) ∈ expectedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨occurrence, occurrenceMember, rfl⟩
  have derivedMember :
      (depUnion occurrence.minorDep occurrence.majorDep,
        [occurrence.tag]) ∈
        derivedOutputs (accumulatedGraph center occurrences) center
          premiseFlow :=
    (derivedSet _).mpr expectedMember
  have derivedNonempty :
      (derivedOutputs (accumulatedGraph center occurrences) center
        premiseFlow).isEmpty = false := by
    have ne : derivedOutputs (accumulatedGraph center occurrences) center
        premiseFlow ≠ [] := by
      intro empty
      rw [empty] at derivedMember
      exact List.not_mem_nil derivedMember
    simpa [List.isEmpty_iff] using ne
  refine ⟨derivedOutputs (accumulatedGraph center occurrences) center
    premiseFlow, ?_, derivedSet⟩
  simp [occurrenceStep, predecessorsNonempty, noAncestors, centerDerived,
    derivedNonempty]

/-- The producer-facing corollary: an actual current accumulator satisfying
`AccumulatorInvariant` runs the real local successor step successfully.  The
additional hypotheses are precisely the operational facts not yet stored in
that producer invariant. -/
theorem AccumulatorInvariant.occurrenceStep
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (centerDerived : current.CENTER.HYPOTHESIS = false)
    (usesAvoid : UsesAvoidCenter current.CENTER processed)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow processed) :
    Exists fun outputs =>
      occurrenceStep
          (DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph current)
          current.CENTER premiseFlow = .ok outputs /\
        Certificate.Flow.ListSetEq outputs
          (expectedFlowRoutes processed) := by
  have premisesAvoid : PremisesAvoidCenter current.CENTER processed :=
    invariant.premisesAvoidCenter
  have logical : forall occurrence, occurrence ∈ processed ->
      occurrence.major.FORMULA =
        occurrence.minor.FORMULA >> current.CENTER.FORMULA := by
    intro occurrence member
    simpa [invariant.centerFormula] using
      invariant.logicalEliminations occurrence member
  have operational := accumulated_occurrenceStep current.CENTER processed
    invariant.nonempty centerDerived premisesAvoid usesAvoid logical
    invariant.distinctTags premiseFlow premiseCorrect
  have graphEqual : accumulatedGraph current.CENTER processed =
      DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph current := by
    unfold accumulatedGraph
    rw [← invariant.exactShape]
  rw [graphEqual] at operational
  exact operational

/-- Boundary projection of raw successor pairs.  The production role is
fixed by this homogeneous fragment; dependency and residual are unchanged. -/
def eliminationCandidateRoutes (routes : List FlowPair) :
    List CandidateRoute :=
  routes.map fun route =>
    { production := .elimination
      dependency := route.1
      residual := { colours := route.2 } }

theorem expectedFlowRoutes_to_occurrenceRoutes
    (occurrences : List ElimOccurrence) :
    eliminationCandidateRoutes (expectedFlowRoutes occurrences) =
      occurrenceRoutes occurrences := by
  induction occurrences with
  | nil => rfl
  | cons occurrence rest ih =>
      simp [eliminationCandidateRoutes, expectedFlowRoutes,
        occurrenceRoutes, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.route, ih]

theorem eliminationCandidateRoutes_congr
    {left right : List FlowPair}
    (equal : Certificate.Flow.ListSetEq left right) :
    Certificate.Flow.ListSetEq
      (eliminationCandidateRoutes left)
      (eliminationCandidateRoutes right) := by
  intro route
  simp only [eliminationCandidateRoutes, List.mem_map]
  constructor
  · rintro ⟨pair, pairMember, rfl⟩
    exact ⟨pair, (equal pair).mp pairMember, rfl⟩
  · rintro ⟨pair, pairMember, rfl⟩
    exact ⟨pair, (equal pair).mpr pairMember, rfl⟩

theorem boundaryDeliveries_routes_congr
    (conclusion : Formula) (edges : List Deduction)
    {left right : List CandidateRoute}
    (equal : Certificate.Flow.ListSetEq left right) :
    Certificate.Flow.ListSetEq
      (boundaryDeliveries conclusion left edges)
      (boundaryDeliveries conclusion right edges) := by
  unfold boundaryDeliveries
  exact Certificate.Flow.flatMap_congr_list
    (fun route => edges.filterMap (observeAcross conclusion route)) equal

/-- Actual-step-to-boundary corollary.  Boundary observations are no longer
obtained from a separately postulated route family: they are projected from
the successful result returned by the real `occurrenceStep`. -/
theorem AccumulatorInvariant.occurrenceStep_boundary
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (centerDerived : current.CENTER.HYPOTHESIS = false)
    (usesAvoid : UsesAvoidCenter current.CENTER processed)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : PremiseRoutesCorrect premiseFlow processed) :
    Exists fun outputs =>
      Evaluator.Occurrence.occurrenceStep
          (DLDS.Compression.Mixed.EliminationSource.neighborhoodGraph current)
          current.CENTER premiseFlow = .ok outputs /\
        Certificate.Flow.ListSetEq
          (boundaryDeliveries conclusion
            (eliminationCandidateRoutes outputs) current.OUTGOING)
          (observeEliminationBoundary dependencyFor current) := by
  obtain ⟨outputs, step, outputSet⟩ :=
    DLDS.Compression.Elimination.Execution.AccumulatorInvariant.occurrenceStep invariant centerDerived usesAvoid
      premiseFlow premiseCorrect
  have routeSet := eliminationCandidateRoutes_congr outputSet
  rw [expectedFlowRoutes_to_occurrenceRoutes] at routeSet
  have deliveredSet := boundaryDeliveries_routes_congr conclusion
    current.OUTGOING routeSet
  have expectedBoundary := invariant.operationalBoundary
  refine ⟨outputs, step, ?_⟩
  exact Certificate.Flow.ListSetEq.trans deliveredSet
    (by simpa [expectedBoundary] using
      (Certificate.Flow.ListSetEq.refl
        (observeEliminationBoundary dependencyFor current)))

end DLDS.Compression.Elimination.Execution
