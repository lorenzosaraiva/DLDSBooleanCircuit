import Readings.FailurePersistence

open scoped Classical

/-!
# Prototype verifier for selected explicit dependency labels

This is deliberately separate from `rejectsMultiB`.  Its input is the finite
dependent product `Readings.Selection.ReadingData`; incoherent readings are rejected,
and every reachable/coherent rule application with a selected outgoing edge is
checked against the existing calculation `outDep`.

There is no stop constructor in `ReadingData`.  A vertex with no outgoing edge
has no output-label obligation; a vertex of outdegree one uses its unique edge;
a fan uses its actual `branchIndex` coordinate.  The error fold is monotone:
once a selected obligation fails, later checks cannot clear the error.
-/

namespace DLDS
namespace Readings.Explicit

open Flow
open Readings.Selection
open DLDS.Readings.LocalLabels

/-- One selected, active output-label obligation fails.  Shape failure and
dependency-label failure are kept in the same operational check, but the latter
is computed with the repository's `outDep`. -/
def selectedLabelErrorAtB (d : Graph)
    (reading : ReadingData (fanArities d)) (v : Vertex) : Bool :=
  selectedRuleActiveB d reading v &&
    match selectedEdge d reading v with
    | none => false
    | some selected =>
        !ruleShapeOKB v d || decide (selected.DEPENDENCY ≠ outDep v d)

/-- Accumulated error bit over a concrete list of rule applications. -/
def selectedLabelHadErrorB (d : Graph)
    (reading : ReadingData (fanArities d)) : Bool :=
  d.NODES.any (selectedLabelErrorAtB d reading)

/-- Prototype verdict.  `true` means rejection.  Unlike `rejectsMultiB`, a
structurally incoherent selector assignment is rejected rather than excluded
from the domain. -/
def rejectsReadingLabelsB (d : Graph)
    (reading : ReadingData (fanArities d)) : Bool :=
  !coherentB d reading || selectedLabelHadErrorB d reading

/-- Evidence marker for graph obligations.  Incoherent assignments are outside
the evidence event rather than being treated as graph defects.  The accumulated
local check still includes both malformed selected rule shapes and explicit
dependency-label mismatches. -/
def labelDefectB (d : Graph)
    (reading : ReadingData (fanArities d)) : Bool :=
  coherentB d reading && selectedLabelHadErrorB d reading

theorem labelDefectB_eq_true_iff (d : Graph)
    (reading : ReadingData (fanArities d)) :
    labelDefectB d reading = true ↔
      coherentB d reading = true ∧
      selectedLabelHadErrorB d reading = true := by
  simp [labelDefectB]

theorem labelDefectB_eq_false_of_incoherent (d : Graph)
    (reading : ReadingData (fanArities d))
    (hcoherent : coherentB d reading = false) :
    labelDefectB d reading = false := by
  simp [labelDefectB, hcoherent]

theorem labelDefectB_implies_rejectsReadingLabelsB (d : Graph)
    (reading : ReadingData (fanArities d))
    (hdefect : labelDefectB d reading = true) :
    rejectsReadingLabelsB d reading = true := by
  have herror : selectedLabelHadErrorB d reading = true :=
    (labelDefectB_eq_true_iff d reading).mp hdefect |>.2
  simp [rejectsReadingLabelsB, herror]

/-- Declarative correctness of every explicit output-label obligation.  This
is stated directly in terms of graph nodes, outgoing edges, rule shape,
`outDep`, and stored dependencies; it is not defined through the detector. -/
def ExplicitOutputLabelsCorrect (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, ∀ e ∈ outgoing d v,
    ruleShapeOKB v d = true ∧ e.DEPENDENCY = outDep v d

/-- Executable mirror used only for concrete finite fixtures. -/
def explicitOutputLabelsCorrectB (d : Graph) : Bool :=
  d.NODES.all fun v =>
    (outgoing d v).all fun e =>
      ruleShapeOKB v d && decide (e.DEPENDENCY = outDep v d)

theorem explicitOutputLabelsCorrectB_eq_true_iff (d : Graph) :
    explicitOutputLabelsCorrectB d = true ↔ ExplicitOutputLabelsCorrect d := by
  simp [explicitOutputLabelsCorrectB, ExplicitOutputLabelsCorrect]

/-- One declarative explicit-output obligation is violated. -/
def ExplicitOutputLabelViolation (d : Graph) (v : Vertex)
    (e : Deduction) : Prop :=
  v ∈ d.NODES ∧ e ∈ outgoing d v ∧
    (ruleShapeOKB v d = false ∨ e.DEPENDENCY ≠ outDep v d)

/-- Structural coverage of one output edge by a coherent reading.  The
definition does not assert that the edge label is correct or incorrect. -/
def ExplicitOutputCovered (d : Graph) (v : Vertex) (e : Deduction) : Prop :=
  ∃ reading : ReadingData (fanArities d),
    coherentB d reading = true ∧
    selectedRuleActiveB d reading v = true ∧
    selectedEdge d reading v = some e

/-- Coverage of all explicit output obligations.  This is intentionally a
separate hypothesis in the completeness direction. -/
def AllExplicitOutputsCovered (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, ∀ e ∈ outgoing d v, ExplicitOutputCovered d v e

theorem selectedEdge_mem_outgoing (d : Graph)
    (reading : ReadingData (fanArities d)) (v : Vertex) (e : Deduction)
    (hselected : selectedEdge d reading v = some e) :
    e ∈ outgoing d v := by
  simp only [selectedEdge] at hselected
  split at hselected
  · exact List.mem_of_getElem? hselected
  · cases hout : outgoing d v with
    | nil => simp [hout] at hselected
    | cons first rest =>
        simp [hout] at hselected
        subst e
        simp

theorem explicitOutputLabelsCorrect_iff_localRuleCorrect (d : Graph) :
    ExplicitOutputLabelsCorrect d ↔ LocalRuleCorrect d := by
  constructor
  · intro hcorrect v hv hout
    obtain ⟨e, he⟩ : ∃ e, e ∈ outgoing d v := by
      cases hlist : outgoing d v with
      | nil => exact (hout hlist).elim
      | cons first rest => exact ⟨first, by simp⟩
    have hone := hcorrect v hv e he
    exact ⟨hone.1, fun selected hselected =>
      (hcorrect v hv selected hselected).2⟩
  · intro hcorrect v hv e he
    have hout : outgoing d v ≠ [] := by
      exact List.ne_nil_of_mem he
    exact hcorrect v hv hout |>.imp_right fun hall => hall e he

theorem selectedLabelErrorAtB_eq_true_iff (d : Graph)
    (reading : ReadingData (fanArities d)) (v : Vertex) :
    selectedLabelErrorAtB d reading v = true ↔
      selectedRuleActiveB d reading v = true ∧
      ∃ selected,
        selectedEdge d reading v = some selected ∧
        (ruleShapeOKB v d = false ∨
          selected.DEPENDENCY ≠ outDep v d) := by
  unfold selectedLabelErrorAtB
  cases hactive : selectedRuleActiveB d reading v <;> simp
  cases hedge : selectedEdge d reading v <;> simp

/-- On an active, well-shaped selected application, the prototype's calculated
side is definitionally the same `outDep` used by `LocalRuleCorrect`. -/
theorem selectedLabelErrorAtB_eq_outDep_comparison
    (d : Graph) (reading : ReadingData (fanArities d)) (v : Vertex)
    (selected : Deduction)
    (hactive : selectedRuleActiveB d reading v = true)
    (hedge : selectedEdge d reading v = some selected)
    (hshape : ruleShapeOKB v d = true) :
    selectedLabelErrorAtB d reading v =
      decide (selected.DEPENDENCY ≠ outDep v d) := by
  simp [selectedLabelErrorAtB, hactive, hedge, hshape]

/-- The accumulated bit reports an actual selected local obligation, not an
unrelated global scan. -/
theorem selectedLabelHadErrorB_eq_true_iff (d : Graph)
    (reading : ReadingData (fanArities d)) :
    selectedLabelHadErrorB d reading = true ↔
      ∃ v ∈ d.NODES,
        selectedRuleActiveB d reading v = true ∧
        ∃ selected,
          selectedEdge d reading v = some selected ∧
          (ruleShapeOKB v d = false ∨
            selected.DEPENDENCY ≠ outDep v d) := by
  simp only [selectedLabelHadErrorB, List.any_eq_true,
    selectedLabelErrorAtB_eq_true_iff]

/-- Every marked reading is a sound witness of an independently stated graph
obligation violation. -/
theorem labelDefectB_sound (d : Graph)
    (reading : ReadingData (fanArities d))
    (hdefect : labelDefectB d reading = true) :
    ∃ v e, ExplicitOutputLabelViolation d v e := by
  have herror : selectedLabelHadErrorB d reading = true :=
    (labelDefectB_eq_true_iff d reading).mp hdefect |>.2
  obtain ⟨v, hv, _hactive, e, hselected, hbad⟩ :=
    (selectedLabelHadErrorB_eq_true_iff d reading).mp herror
  exact ⟨v, e, hv, selectedEdge_mem_outgoing d reading v e hselected, hbad⟩

theorem labelDefectB_implies_not_explicitOutputLabelsCorrect (d : Graph)
    (reading : ReadingData (fanArities d))
    (hdefect : labelDefectB d reading = true) :
    ¬ ExplicitOutputLabelsCorrect d := by
  obtain ⟨v, e, hv, he, hbad⟩ := labelDefectB_sound d reading hdefect
  intro hcorrect
  have hgood := hcorrect v hv e he
  rcases hbad with hshape | hdep
  · exact Bool.noConfusion (hshape.symm.trans hgood.1)
  · exact hdep hgood.2

/-- Declaratively correct explicit labels produce no false witnesses, even on
the raw selector product. -/
theorem labelDefectB_eq_false_of_explicitOutputLabelsCorrect
    (d : Graph) (hcorrect : ExplicitOutputLabelsCorrect d)
    (reading : ReadingData (fanArities d)) :
    labelDefectB d reading = false := by
  cases hdefect : labelDefectB d reading with
  | false => rfl
  | true => exact (labelDefectB_implies_not_explicitOutputLabelsCorrect
      d reading hdefect hcorrect).elim

theorem labelDefectB_implies_not_localRuleCorrect
    (d : Graph) (reading : ReadingData (fanArities d))
    (hdefect : labelDefectB d reading = true) :
    ¬ LocalRuleCorrect d := by
  rw [← explicitOutputLabelsCorrect_iff_localRuleCorrect]
  exact labelDefectB_implies_not_explicitOutputLabelsCorrect d reading hdefect

theorem labelDefectB_eq_false_of_localRuleCorrect
    (d : Graph) (hcorrect : LocalRuleCorrect d)
    (reading : ReadingData (fanArities d)) :
    labelDefectB d reading = false := by
  apply labelDefectB_eq_false_of_explicitOutputLabelsCorrect d
  exact (explicitOutputLabelsCorrect_iff_localRuleCorrect d).mpr hcorrect

/-- Coverage plus a violated explicit obligation produces a coherent marked
reading.  Coverage is structural and remains separate from the bad label. -/
theorem explicitOutputViolation_exposed
    (d : Graph) (v : Vertex) (e : Deduction)
    (hcoverage : ExplicitOutputCovered d v e)
    (hviolation : ExplicitOutputLabelViolation d v e) :
    ∃ reading, labelDefectB d reading = true := by
  obtain ⟨reading, hcoherent, hactive, hselected⟩ := hcoverage
  have hlocal : selectedLabelErrorAtB d reading v = true := by
    apply (selectedLabelErrorAtB_eq_true_iff d reading v).mpr
    exact ⟨hactive, e, hselected, hviolation.2.2⟩
  have herror : selectedLabelHadErrorB d reading = true :=
    List.any_eq_true.mpr ⟨v, hviolation.1, hlocal⟩
  exact ⟨reading, by simp [labelDefectB, hcoherent, herror]⟩

/-- Under independent coverage of all explicit outputs, declarative label
correctness is equivalent to absence of marked readings. -/
theorem explicitOutputLabelsCorrect_iff_no_labelDefect
    (d : Graph) (hcoverage : AllExplicitOutputsCovered d) :
    ExplicitOutputLabelsCorrect d ↔
      ∀ reading : ReadingData (fanArities d), labelDefectB d reading = false := by
  constructor
  · exact labelDefectB_eq_false_of_explicitOutputLabelsCorrect d
  · intro hnone v hv e he
    have hcovered := hcoverage v hv e he
    constructor
    · cases hshape : ruleShapeOKB v d with
      | true => rfl
      | false =>
          obtain ⟨reading, hmarked⟩ := explicitOutputViolation_exposed d v e
            hcovered ⟨hv, he, Or.inl hshape⟩
          exact Bool.noConfusion ((hnone reading).symm.trans hmarked)
    · by_contra hdep
      obtain ⟨reading, hmarked⟩ := explicitOutputViolation_exposed d v e
        hcovered ⟨hv, he, Or.inr hdep⟩
      exact Bool.noConfusion ((hnone reading).symm.trans hmarked)

/-- Restricting the raw marked set to coherent readings does not change it,
because coherence is part of `labelDefectB` itself. -/
theorem filter_labelDefectB_eq_filter_coherentReadings (d : Graph) :
    (readings d).filter (labelDefectB d) =
      (coherentReadings d).filter (labelDefectB d) := by
  unfold coherentReadings
  rw [List.filter_filter]
  apply List.filter_congr
  intro reading _hreading
  cases hcoherent : coherentB d reading <;>
    cases herror : selectedLabelHadErrorB d reading <;>
      simp [labelDefectB, hcoherent, herror]

/-- Exact semantics of the prototype's rejection bit. -/
theorem rejectsReadingLabelsB_eq_true_iff (d : Graph)
    (reading : ReadingData (fanArities d)) :
    rejectsReadingLabelsB d reading = true ↔
      coherentB d reading = false ∨
      ∃ v ∈ d.NODES,
        selectedRuleActiveB d reading v = true ∧
        ∃ selected,
          selectedEdge d reading v = some selected ∧
          (ruleShapeOKB v d = false ∨
            selected.DEPENDENCY ≠ outDep v d) := by
  simp [rejectsReadingLabelsB, selectedLabelHadErrorB_eq_true_iff]

/-- A selected dependency mismatch at an actual node reaches the accumulated
verdict. This pointwise correspondence concerns the separate
explicit-label reading verifier rather than for `rejectsMultiB`. -/
theorem selectedRuleApplicationMismatchB_rejectsReadingLabelsB
    (d : Graph) (reading : ReadingData (fanArities d)) (v : Vertex)
    (hv : v ∈ d.NODES)
    (hmismatch : selectedRuleApplicationMismatchB d reading v = true) :
    rejectsReadingLabelsB d reading = true := by
  have hmismatchParts :
      selectedRuleActiveB d reading v = true ∧
        selectedDecorationMismatchB d reading v = true := by
    simpa [selectedRuleApplicationMismatchB] using hmismatch
  have hlocal : selectedLabelErrorAtB d reading v = true := by
    unfold selectedLabelErrorAtB
    rw [hmismatchParts.1]
    simp only [Bool.true_and]
    unfold selectedDecorationMismatchB at hmismatchParts
    cases hedge : selectedEdge d reading v with
    | none => simp [hedge] at hmismatchParts
    | some selected =>
        simp [hedge] at hmismatchParts ⊢
        exact Or.inr hmismatchParts.2
  have haccumulated : selectedLabelHadErrorB d reading = true := by
    exact List.any_eq_true.mpr ⟨v, hv, hlocal⟩
  simp [rejectsReadingLabelsB, haccumulated]

/-- A coherent reading satisfying every selected explicit-label obligation is
not rejected spuriously by the prototype. -/
theorem rejectsReadingLabelsB_eq_false_of_selected_correct
    (d : Graph) (reading : ReadingData (fanArities d))
    (hcoherent : coherentB d reading = true)
    (hcorrect : ∀ v ∈ d.NODES,
      selectedRuleActiveB d reading v = true →
      ∀ selected, selectedEdge d reading v = some selected →
        ruleShapeOKB v d = true ∧
        selected.DEPENDENCY = outDep v d) :
    rejectsReadingLabelsB d reading = false := by
  have hnoError : selectedLabelHadErrorB d reading = false := by
    apply List.any_eq_false.mpr
    intro v hv
    unfold selectedLabelErrorAtB
    cases hactive : selectedRuleActiveB d reading v with
    | false => simp
    | true =>
        cases hedge : selectedEdge d reading v with
        | none => simp
        | some selected =>
            obtain ⟨hshape, hdep⟩ := hcorrect v hv hactive selected hedge
            simp [hshape, hdep]
  simp [rejectsReadingLabelsB, hcoherent, hnoError]

/-- Operational persistence permits an earlier change either to trigger the
global coherence rejection or to leave the selected application active. -/
theorem rejection_persists_of_incoherent_or_active
    (d : Graph) (v : Vertex) (hv : v ∈ d.NODES)
    {x y : ReadingData (fanArities d)}
    (hx : selectedRuleApplicationMismatchB d x v = true)
    (hcoordinate :
      (x.indices[branchIndex d v]?).getD 0 =
        (y.indices[branchIndex d v]?).getD 0)
    (hguard : coherentB d y = false ∨
      selectedRuleActiveB d y v = true) :
    rejectsReadingLabelsB d y = true := by
  cases hguard with
  | inl hincoherent => simp [rejectsReadingLabelsB, hincoherent]
  | inr hactive =>
      have hmismatch := selectedRuleApplicationMismatchB_persists d v hx
        hactive hcoordinate
      exact selectedRuleApplicationMismatchB_rejectsReadingLabelsB
        d y v hv hmismatch

/-- A coherent selected mismatch is evidence of a graph defect, rather than
merely a rejected selector assignment. -/
theorem selectedRuleApplicationMismatchB_marks_labelDefectB
    (d : Graph) (reading : ReadingData (fanArities d)) (v : Vertex)
    (hv : v ∈ d.NODES)
    (hcoherent : coherentB d reading = true)
    (hmismatch : selectedRuleApplicationMismatchB d reading v = true) :
    labelDefectB d reading = true := by
  have hreject := selectedRuleApplicationMismatchB_rejectsReadingLabelsB
    d reading v hv hmismatch
  have herror : selectedLabelHadErrorB d reading = true := by
    unfold rejectsReadingLabelsB at hreject
    simp [hcoherent] at hreject
    exact hreject
  simp [labelDefectB, hcoherent, herror]

/-- General conditional persistence theorem.  It deliberately keeps global
coherence and activation explicit; extracting selector coordinates sufficient
for those two conditions is a separate structural problem. -/
theorem labelDefectB_persists_of_coherent_active
    (d : Graph) (v : Vertex) (hv : v ∈ d.NODES)
    {x y : ReadingData (fanArities d)}
    (hx : selectedRuleApplicationMismatchB d x v = true)
    (hcoordinate :
      (x.indices[branchIndex d v]?).getD 0 =
        (y.indices[branchIndex d v]?).getD 0)
    (hcoherentY : coherentB d y = true)
    (hactiveY : selectedRuleActiveB d y v = true) :
    labelDefectB d y = true := by
  have hmismatchY := selectedRuleApplicationMismatchB_persists d v hx
    hactiveY hcoordinate
  exact selectedRuleApplicationMismatchB_marks_labelDefectB
    d y v hv hcoherentY hmismatchY

namespace Examples.Elimination

def A : Formula := #"A"
def B : Formula := #"B"
def X : Formula := #"X"
def L : Formula := #"L"
def Z : Formula := #"Z"
def AB : Formula := A >> B

def hA : Vertex := node 0 3 A true false []
def hAB : Vertex := node 1 3 AB true false []
def elimB : Vertex := node 2 2 B false false []
def hLater : Vertex := node 3 1 L true false []
def bypassRoot : Vertex := node 4 2 X false false []
def goodRoot : Vertex := node 5 1 X false false []
def badRoot : Vertex := node 6 1 X false false []
def laterRoot0 : Vertex := node 7 0 X false false []
def laterRoot1 : Vertex := node 8 0 X false false []

def edgeHAElim : Deduction := edge hA elimB 0 [A]
def edgeHABypass : Deduction := edge hA bypassRoot 1 [A]
def edgeHABElim : Deduction := edge hAB elimB 0 [AB]
def edgeElimGood : Deduction := edge elimB goodRoot 0 [A, AB]
def edgeElimBadControl : Deduction := edge elimB badRoot 1 [A, AB]
def edgeElimBad : Deduction := edge elimB badRoot 1 [Z]
def edgeLater0 : Deduction := edge hLater laterRoot0 0 [L]
def edgeLater1 : Deduction := edge hLater laterRoot1 1 [L]

/-- Control graph: the selected elimination label agrees with `outDep`. -/
def control : Graph := dlds
  [hA, hAB, elimB, hLater, bypassRoot, goodRoot, badRoot,
    laterRoot0, laterRoot1]
  [edgeHAElim, edgeHABypass, edgeHABElim, edgeElimGood,
    edgeElimBadControl, edgeLater0, edgeLater1]
  []

/-- Defective graph: only the dependency label of colour-1 output of the real
elimination is changed.  Nodes, topology, colours, and provenance stay fixed. -/
def defective : Graph := dlds
  control.NODES
  [edgeHAElim, edgeHABypass, edgeHABElim, edgeElimGood,
    edgeElimBad, edgeLater0, edgeLater1]
  control.PATHS

theorem fixed_data_except_tested_label :
    control.NODES = defective.NODES ∧
    (control.EDGES.map fun e => (e.START, e.END, e.COLOUR)) =
      (defective.EDGES.map fun e => (e.START, e.END, e.COLOUR)) ∧
    control.PATHS = defective.PATHS := by
  native_decide

/-- Deduction equality is record equality and therefore observes the changed
dependency, even though all routing fields agree. -/
theorem changed_dependency_changes_edge_identity :
    edgeElimBadControl ≠ edgeElimBad := by
  native_decide

theorem fan_arities_control : fanArities control = [2, 2, 2] := by
  native_decide

theorem fan_arities_defective : fanArities defective = [2, 2, 2] := by
  native_decide

theorem branch_point_numbers :
    (branchPoints defective).map (fun v => v.NUMBER) = [0, 2, 3] := by
  native_decide

theorem control_branch_point_numbers :
    (branchPoints control).map (fun v => v.NUMBER) = [0, 2, 3] := by
  native_decide

theorem raw_reading_count : (readings defective).length = 8 := by
  native_decide

theorem control_coherent_reading_count :
    (coherentReadings control).length = 4 := by
  native_decide

theorem defective_coherent_reading_count :
    (coherentReadings defective).length = 4 := by
  native_decide

theorem elim_is_real_rule : ruleShapeOKB elimB defective = true := by
  native_decide

theorem elim_branch_index : branchIndex defective elimB = 1 := by
  native_decide

theorem elim_premises :
    (get_rule.incoming elimB defective).map (fun e => e.START) = [hA, hAB] := by
  native_decide

theorem elim_calculated_dependency :
    outDep elimB defective = normalizeDep [A, AB] := by
  native_decide

theorem elim_stored_dependencies :
    (outgoing defective elimB).map (fun e => e.DEPENDENCY) =
      [normalizeDep [A, AB], [Z]] := by
  native_decide

theorem control_explicit_output_labels_correct :
    ExplicitOutputLabelsCorrect control := by
  exact (explicitOutputLabelsCorrectB_eq_true_iff control).mp (by native_decide)

theorem defective_explicit_output_labels_incorrect :
    ¬ ExplicitOutputLabelsCorrect defective := by
  intro hcorrect
  have htrue := (explicitOutputLabelsCorrectB_eq_true_iff defective).mpr hcorrect
  have hfalse : explicitOutputLabelsCorrectB defective = false := by
    native_decide
  simp [hfalse] at htrue

def readingFor (d : Graph) (hfan : fanArities d = [2, 2, 2])
    (earlier localChoice later : Fin 2) : ReadingData (fanArities d) := by
  rw [hfan]
  exact .cons earlier (.cons localChoice (.cons later .nil))

def controlX : ReadingData (fanArities control) :=
  readingFor control fan_arities_control 0 1 0

def controlGood : ReadingData (fanArities control) :=
  readingFor control fan_arities_control 0 0 0

def x : ReadingData (fanArities defective) :=
  readingFor defective fan_arities_defective 0 1 0

def changedEarlier : ReadingData (fanArities defective) :=
  readingFor defective fan_arities_defective 1 1 0

def changedLocal : ReadingData (fanArities defective) :=
  readingFor defective fan_arities_defective 0 0 0

def changedLater : ReadingData (fanArities defective) :=
  readingFor defective fan_arities_defective 0 1 1

theorem corresponding_reading_indices
    (earlier localChoice later : Fin 2) :
    (readingFor control fan_arities_control earlier localChoice later).indices =
      (readingFor defective fan_arities_defective earlier localChoice later).indices := by
  fin_cases earlier <;> fin_cases localChoice <;> fin_cases later <;>
    native_decide

/-- For the control/defect pair, coherence depends only on the preserved
topology and selected endpoints, not on the changed dependency field. -/
theorem coherence_preserved_by_tested_label_change
    (earlier localChoice later : Fin 2) :
    coherentB control
      (readingFor control fan_arities_control earlier localChoice later) =
    coherentB defective
      (readingFor defective fan_arities_defective earlier localChoice later) := by
  fin_cases earlier <;> fin_cases localChoice <;> fin_cases later <;>
    native_decide

theorem control_x_diagnostic :
    (controlX.indices, coherentB control controlX,
      selectedRuleApplicationMismatchB control controlX elimB,
      selectedLabelHadErrorB control controlX,
      rejectsReadingLabelsB control controlX) =
    ([0, 1, 0], true, false, false, false) := by
  native_decide

theorem x_diagnostic :
    (x.indices, coherentB defective x,
      selectedRuleApplicationMismatchB defective x elimB,
      selectedLabelHadErrorB defective x,
      rejectsReadingLabelsB defective x) =
    ([0, 1, 0], true, true, true, true) := by
  native_decide

theorem x_selected_application :
    selectedRuleActiveB defective x elimB = true ∧
    selectedEdge defective x elimB = some edgeElimBad := by
  native_decide

/-- Changing the earlier selector hides the original mismatch by making the
elimination locally incoherent; global rejection nevertheless persists. -/
theorem changed_earlier_diagnostic :
    (changedEarlier.indices, coherentB defective changedEarlier,
      selectedRuleApplicationMismatchB defective changedEarlier elimB,
      selectedLabelHadErrorB defective changedEarlier,
      rejectsReadingLabelsB defective changedEarlier) =
    ([1, 1, 0], false, false, false, true) := by
  native_decide

/-- Changing the tested local selector avoids the bad output and permits the
coherent reading to be accepted. -/
theorem changed_local_diagnostic :
    (changedLocal.indices, coherentB defective changedLocal,
      selectedRuleApplicationMismatchB defective changedLocal elimB,
      selectedLabelHadErrorB defective changedLocal,
      rejectsReadingLabelsB defective changedLocal) =
    ([0, 0, 0], true, false, false, false) := by
  native_decide

/-- The independent later fan coordinate is not read by the elimination
application and does not affect its rejection. -/
theorem changed_later_diagnostic :
    (changedLater.indices, coherentB defective changedLater,
      selectedRuleApplicationMismatchB defective changedLater elimB,
      selectedLabelHadErrorB defective changedLater,
      rejectsReadingLabelsB defective changedLater) =
    ([0, 1, 1], true, true, true, true) := by
  native_decide

/-- Complete assignment table for the historical rejection verifier on the
control graph.  Exactly the four incoherent assignments are rejected. -/
theorem control_assignment_table :
    (readings control).map (fun reading =>
      (reading.indices,
        coherentB control reading,
        selectedRuleActiveB control reading elimB,
        selectedRuleApplicationMismatchB control reading elimB,
        selectedLabelHadErrorB control reading,
        rejectsReadingLabelsB control reading)) =
      [([0, 0, 0], true,  true,  false, false, false),
       ([0, 0, 1], true,  true,  false, false, false),
       ([0, 1, 0], true,  true,  false, false, false),
       ([0, 1, 1], true,  true,  false, false, false),
       ([1, 0, 0], false, false, false, false, true),
       ([1, 0, 1], false, false, false, false, true),
       ([1, 1, 0], false, false, false, false, true),
       ([1, 1, 1], false, false, false, false, true)] := by
  native_decide

/-- Complete assignment table for the historical rejection verifier on the
defective graph. -/
theorem defective_assignment_table :
    (readings defective).map (fun reading =>
      (reading.indices,
        coherentB defective reading,
        selectedRuleActiveB defective reading elimB,
        selectedRuleApplicationMismatchB defective reading elimB,
        selectedLabelHadErrorB defective reading,
        rejectsReadingLabelsB defective reading)) =
      [([0, 0, 0], true,  true,  false, false, false),
       ([0, 0, 1], true,  true,  false, false, false),
       ([0, 1, 0], true,  true,  true,  true,  true),
       ([0, 1, 1], true,  true,  true,  true,  true),
       ([1, 0, 0], false, false, false, false, true),
       ([1, 0, 1], false, false, false, false, true),
       ([1, 1, 0], false, false, false, false, true),
       ([1, 1, 1], false, false, false, false, true)] := by
  native_decide

theorem changed_inputs_marker_diagnostic :
    (labelDefectB defective x,
      labelDefectB defective changedEarlier,
      labelDefectB defective changedLocal,
      labelDefectB defective changedLater) =
    (true, false, false, true) := by
  native_decide

theorem defective_bad_output_violation :
    ExplicitOutputLabelViolation defective elimB edgeElimBad := by
  refine ⟨by native_decide, by native_decide, Or.inr ?_⟩
  native_decide

theorem control_elim_outputs_covered :
    ∀ e ∈ outgoing control elimB, ExplicitOutputCovered control elimB e := by
  intro e he
  have hout : outgoing control elimB = [edgeElimGood, edgeElimBadControl] := by
    native_decide
  rw [hout] at he
  simp at he
  rcases he with he | he
  · subst e
    exact ⟨controlGood, by native_decide, by native_decide, by native_decide⟩
  · subst e
    exact ⟨controlX, by native_decide, by native_decide, by native_decide⟩

theorem defective_elim_outputs_covered :
    ∀ e ∈ outgoing defective elimB,
      ExplicitOutputCovered defective elimB e := by
  intro e he
  have hout : outgoing defective elimB = [edgeElimGood, edgeElimBad] := by
    native_decide
  rw [hout] at he
  simp at he
  rcases he with he | he
  · subst e
    exact ⟨changedLocal, by native_decide, by native_decide, by native_decide⟩
  · subst e
    exact ⟨x, by native_decide, by native_decide, by native_decide⟩

/-- The defective explicit output is exposed by a coherent reading through the
general coverage theorem, independently of the eight-row enumeration. -/
theorem defective_bad_output_is_exposed :
    ∃ reading, labelDefectB defective reading = true := by
  apply explicitOutputViolation_exposed defective elimB edgeElimBad
  · exact defective_elim_outputs_covered edgeElimBad (by native_decide)
  · exact defective_bad_output_violation

/-- Executable finite proof of the structural guard used by
`rejection_persists_of_incoherent_or_active` for every enumerated reading. -/
theorem earlier_failure_or_elim_active :
    (readings defective).all fun reading =>
      coherentB defective reading == false ||
        selectedRuleActiveB defective reading elimB := by
  native_decide

/-- Fixing only address `branchIndex defective elimB = 1` to selector value
`1` suffices for rejection.  The earlier and independent-later coordinates
remain free. -/
theorem local_bad_coordinate_suffices
    {reading : ReadingData (fanArities defective)}
    (hreading : reading ∈ readings defective)
    (hlocal : reading.indices.getD 1 0 = 1) :
    rejectsReadingLabelsB defective reading = true := by
  have hguard := List.all_eq_true.mp earlier_failure_or_elim_active
    reading hreading
  have hxMismatch :
      selectedRuleApplicationMismatchB defective x elimB = true := by
    native_decide
  have hcoordinate :
      (x.indices[branchIndex defective elimB]?).getD 0 =
        (reading.indices[branchIndex defective elimB]?).getD 0 := by
    rw [elim_branch_index]
    have hxlocal : (x.indices[1]?).getD 0 = 1 := by
      native_decide
    rw [hxlocal]
    exact hlocal.symm
  apply rejection_persists_of_incoherent_or_active defective elimB
    (by native_decide) hxMismatch hcoordinate
  simpa using hguard

theorem local_bad_cylinder_count :
    ((readings defective).filter fun reading =>
      reading.indices.getD 1 0 = 1).length = 4 := by
  native_decide

theorem local_bad_coherent_cylinder_count :
    ((coherentReadings defective).filter fun reading =>
      reading.indices.getD 1 0 = 1).length = 2 := by
  native_decide

/-- The old one-coordinate rejection cylinder is not a defect-evidence
cylinder: two of its four extensions are incoherent and therefore unmarked. -/
theorem local_bad_marker_split :
    (((readings defective).filter fun reading =>
        reading.indices.getD 1 0 = 1 ∧
          labelDefectB defective reading).length,
      ((readings defective).filter fun reading =>
        reading.indices.getD 1 0 = 1 ∧
          coherentB defective reading = false).length) = (2, 2) := by
  native_decide

theorem changedEarlier_refutes_one_coordinate_defect_cylinder :
    changedEarlier.indices.getD 1 0 = 1 ∧
    rejectsReadingLabelsB defective changedEarlier = true ∧
    labelDefectB defective changedEarlier = false := by
  native_decide

/-- Exact finite guard for the genuine defect cylinder: coordinate `0` keeps
the minor premise feeding the elimination and coordinate `1` selects its bad
output; coordinate `2` remains free. -/
theorem two_coordinate_label_defect_guard :
    (readings defective).all fun reading =>
      !((reading.indices.getD 0 0 == 0) &&
        (reading.indices.getD 1 0 == 1)) ||
        labelDefectB defective reading := by
  native_decide

theorem two_coordinates_force_labelDefect
    {reading : ReadingData (fanArities defective)}
    (hreading : reading ∈ readings defective)
    (hearlier : reading.indices.getD 0 0 = 0)
    (hlocal : reading.indices.getD 1 0 = 1) :
    labelDefectB defective reading = true := by
  have hrow := List.all_eq_true.mp two_coordinate_label_defect_guard
    reading hreading
  rw [List.getD_eq_getElem?_getD] at hearlier hlocal
  simp [hearlier, hlocal] at hrow
  exact hrow

theorem two_coordinate_defect_cylinder_count :
    ((readings defective).filter fun reading =>
      reading.indices.getD 0 0 = 0 ∧
      reading.indices.getD 1 0 = 1).length = 2 := by
  native_decide

theorem defective_rejection_count :
    ((readings defective).filter (rejectsReadingLabelsB defective)).length = 6 := by
  native_decide

theorem control_rejection_count :
    ((readings control).filter (rejectsReadingLabelsB control)).length = 4 := by
  native_decide

theorem control_label_defect_count :
    ((readings control).filter (labelDefectB control)).length = 0 := by
  native_decide

theorem defective_label_defect_count :
    ((readings defective).filter (labelDefectB defective)).length = 2 := by
  native_decide

theorem control_coherent_label_defect_count :
    ((coherentReadings control).filter (labelDefectB control)).length = 0 := by
  native_decide

theorem defective_coherent_label_defect_count :
    ((coherentReadings defective).filter
      (labelDefectB defective)).length = 2 := by
  native_decide

theorem control_exact_counts :
    ((readings control).length,
      (coherentReadings control).length,
      ((readings control).filter (rejectsReadingLabelsB control)).length,
      ((readings control).filter (labelDefectB control)).length,
      ((coherentReadings control).filter (labelDefectB control)).length) =
    (8, 4, 4, 0, 0) := by
  native_decide

theorem defective_exact_counts :
    ((readings defective).length,
      (coherentReadings defective).length,
      ((readings defective).filter
        (rejectsReadingLabelsB defective)).length,
      ((readings defective).filter (labelDefectB defective)).length,
      ((coherentReadings defective).filter
        (labelDefectB defective)).length) =
    (8, 4, 6, 2, 2) := by
  native_decide

/-- Exact relation between raw and coherence-conditioned densities in the
control fixture: `0/8 = (4/8) * (0/4)`. -/
theorem control_density_factorization :
    (0 : ℚ) / 8 = ((4 : ℚ) / 8) * ((0 : ℚ) / 4) := by
  norm_num

/-- Exact relation in the defective fixture: `2/8 = (4/8) * (2/4)`. -/
theorem defective_density_factorization :
    (2 : ℚ) / 8 = ((4 : ℚ) / 8) * ((2 : ℚ) / 4) := by
  norm_num

theorem control_coherent_rejection_count :
    ((coherentReadings control).filter
      (rejectsReadingLabelsB control)).length = 0 := by
  native_decide

theorem defective_coherent_rejection_count :
    ((coherentReadings defective).filter
      (rejectsReadingLabelsB defective)).length = 2 := by
  native_decide

/-- This small graph is not in the existing multi-token admissible domain;
the prototype result is not silently transferred to `rejectsMultiB`. -/
theorem existing_multi_marker_diagnostic :
    (canonicalSlotOKB defective,
      admissibleMultiPathB defective (multiPathsFromFlow defective),
      evalErrorMultiB defective (multiPathsFromFlow defective),
      dischargedOnMultiB defective (multiPathsFromFlow defective),
      evaluateDLDS_multi defective (multiPathsFromFlow defective)
        (goalColumn defective),
      rejectsMultiB defective (multiPathsFromFlow defective)) =
    (true, false, true, true, true, false) := by
  native_decide

end Examples.Elimination

/-! The SBCCQ-shaped fixture below has an empty
coherent-reading domain, so absence of marked inputs cannot certify
its explicit labels without coverage. -/
namespace Examples.EmptyCoherence

abbrev graph : Graph := DLDS.Readings.LocalLabels.Examples.Elimination.graph
abbrev testedNode : Vertex := Readings.Selection.Examples.ColouredCompression.a2

theorem counts :
    ((readings graph).length,
      (coherentReadings graph).length,
      ((readings graph).filter (labelDefectB graph)).length,
      ((readings graph).filter fun reading =>
        selectedDecorationMismatchB graph reading testedNode).length) =
    (8, 0, 0, 4) := by
  native_decide

theorem explicit_labels_are_not_correct :
    ¬ ExplicitOutputLabelsCorrect graph := by
  intro hcorrect
  have htrue := (explicitOutputLabelsCorrectB_eq_true_iff graph).mpr hcorrect
  have hfalse : explicitOutputLabelsCorrectB graph = false := by
    native_decide
  simp [hfalse] at htrue

theorem no_mark_does_not_certify_without_coverage :
    (∀ reading : ReadingData (fanArities graph),
      reading ∈ readings graph → labelDefectB graph reading = false) ∧
    ¬ ExplicitOutputLabelsCorrect graph := by
  constructor
  · intro reading hreading
    have hall : (readings graph).all fun candidate =>
        !labelDefectB graph candidate := by
      native_decide
    have hrow := List.all_eq_true.mp hall reading hreading
    simpa using hrow
  · exact explicit_labels_are_not_correct

end Examples.EmptyCoherence

namespace Examples.MultiTokenMarker

open DLDS.Examples.BinaryFan

def selectedEdge : Deduction :=
  exFan2C.EDGES.headD (edge DLDS.Examples.BinaryFan.a1 DLDS.Examples.BinaryFan.b1 2 [DLDS.Examples.BinaryFan.fA])

def selectedSource : Vertex := selectedEdge.START

def blankSelectedDependency (candidate : Deduction) : Deduction :=
  if candidate = selectedEdge then
    edge candidate.START candidate.END candidate.COLOUR []
  else candidate

/-- A fixed compressed DLDS with one explicit dependency label blanked. -/
def defective : Graph := dlds exFan2C.NODES
  (exFan2C.EDGES.map blankSelectedDependency) exFan2C.PATHS

def blankedSelectedEdge : Deduction :=
  edge selectedEdge.START selectedEdge.END selectedEdge.COLOUR []

theorem selected_control_edge :
    (selectedSource.NUMBER, selectedEdge.END.NUMBER, selectedEdge.COLOUR,
      selectedEdge.DEPENDENCY) = (1, 5, 2, [DLDS.Examples.BinaryFan.fA]) := by
  native_decide

theorem defective_selected_output :
    selectedSource ∈ defective.NODES ∧
    blankedSelectedEdge ∈ get_rule.outgoing selectedSource defective ∧
    get_rule.outgoing selectedSource defective ≠ [] ∧
    blankedSelectedEdge.DEPENDENCY ≠ outDep selectedSource defective := by
  native_decide

/-- Concrete refutation of the unproved implication
`not LocalRuleCorrect -> rejectsMultiB`: this graph really violates the
stored output-label obligation. -/
theorem defective_not_local_rule_correct : ¬ LocalRuleCorrect defective := by
  intro hcorrect
  obtain ⟨hnode, hedge, houtgoing, hmismatch⟩ := defective_selected_output
  have hlocal := hcorrect selectedSource hnode houtgoing
  exact hmismatch (hlocal.2 blankedSelectedEdge hedge)

theorem empty_dependency_is_flow_wildcard :
    depSimilarB [DLDS.Examples.BinaryFan.fA] [] = true := by
  native_decide

theorem unrelated_nonempty_dependency_fails_guard :
    depSimilarB [DLDS.Examples.BinaryFan.fA] [DLDS.Examples.BinaryFan.fX] = false := by
  native_decide

/-- The canonical input remains admissible, has no accumulated evaluator
error, is discharged, and is accepted despite the concrete LocalRuleCorrect
failure above. -/
theorem defective_existing_oracle_diagnostic :
    (canonicalSlotOKB defective,
      admissibleMultiPathB defective (multiPathsFromFlow defective),
      evalErrorMultiB defective (multiPathsFromFlow defective),
      dischargedOnMultiB defective (multiPathsFromFlow defective),
      evaluateDLDS_multi defective (multiPathsFromFlow defective)
        (goalColumn defective),
      rejectsMultiB defective (multiPathsFromFlow defective)) =
    (true, true, false, true, true, false) := by
  native_decide

theorem defective_multi_calculation_matches_control :
    getEvalResultMultiDLDS defective (multiPathsFromFlow defective) =
      getEvalResultMultiDLDS exFan2C (multiPathsFromFlow exFan2C) := by
  native_decide

end Examples.MultiTokenMarker

end Readings.Explicit
end DLDS
