import Readings.Selection

open scoped Classical

/-!
# Selector-local failure persistence

This file keeps two statements deliberately separate.

* `rejectsMultiB` is the Boolean phase predicate specified for the existing
  multi-token verifier: an input is marked when it is admissible and its goal
  dependency vector is not discharged.
* `selectedDecorationMismatchB` observes the dependency label of the edge
  selected by one coordinate of `Readings.Selection.ReadingData`.

There is currently no extraction from `ReadingData` to `MultiPathInput` and no
theorem saying that a selected decoration mismatch makes `rejectsMultiB` true.
The results below therefore must not be composed as if that missing bridge had
already been proved.
-/

namespace DLDS

/-! ## What the single-token evaluator can observe -/

/-- The legacy/single-token `evaluateDLDS` depends on a graph only through its
node list (formula universe and maximum level).  In particular, changing edge
topology, colours, stored dependencies, or ancestral paths while preserving
`NODES` cannot change its verdict. -/
theorem evaluateDLDS_eq_of_nodes_eq
    {left right : Graph} (hnodes : left.NODES = right.NODES)
    (paths : PathInput) (goal : Nat) :
    evaluateDLDS left paths goal = evaluateDLDS right paths goal := by
  cases left with
  | dlds leftNodes leftEdges leftPaths =>
      cases right with
      | dlds rightNodes rightEdges rightPaths =>
          simp only at hnodes
          subst rightNodes
          rfl

/-! ## Multi-token phase predicate -/

/-- Boolean phase predicate for admissible multi-paths with failed discharge.
Malformed multi-paths are outside the marked set rather than witnesses. -/
def rejectsMultiB (d : Graph) (paths : MultiPathInput) : Bool :=
  admissibleMultiPathB d paths && !dischargedOnMultiB d paths

/-- Propositional wrapper which preserves the polarity of `rejectsMultiB`. -/
def RejectsMulti (d : Graph) (paths : MultiPathInput) : Prop :=
  rejectsMultiB d paths = true

theorem rejectsMultiB_eq_true_iff (d : Graph) (paths : MultiPathInput) :
    rejectsMultiB d paths = true ↔
      AdmissibleMultiPath d paths ∧ dischargedOnMultiB d paths = false := by
  simp [rejectsMultiB, AdmissibleMultiPath]

/-- On the actual admissible multi-path domain, one rejecting witness makes
every admissible input reject.  No path coordinate is fixed: this is the
all-or-nothing consequence of `admissibleMulti_eval_canonical`, not a local
reading-cylinder theorem. -/
theorem rejectsMultiB_persists_on_admissible
    (d : Graph) (hslot : canonicalSlotOKB d = true)
    {x : MultiPathInput} (hx : rejectsMultiB d x = true)
    {y : MultiPathInput} (hy : AdmissibleMultiPath d y) :
    rejectsMultiB d y = true := by
  have hxParts := (rejectsMultiB_eq_true_iff d x).mp hx
  have hxEval : getEvalResultMultiDLDS d x =
      getEvalResultMultiDLDS d (multiPathsFromFlow d) :=
    admissibleMulti_eval_canonical d hslot hxParts.1
  have hyEval : getEvalResultMultiDLDS d y =
      getEvalResultMultiDLDS d (multiPathsFromFlow d) :=
    admissibleMulti_eval_canonical d hslot hy
  have hxy : getEvalResultMultiDLDS d x = getEvalResultMultiDLDS d y :=
    hxEval.trans hyEval.symm
  have hyNotDischarged : dischargedOnMultiB d y = false := by
    unfold dischargedOnMultiB at hxParts ⊢
    rw [← hxy]
    exact hxParts.2
  exact (rejectsMultiB_eq_true_iff d y).mpr ⟨hy, hyNotDischarged⟩

/-- Pointwise Boolean form useful for exact counting in any explicitly finite
enumeration: once an admissible rejecting witness exists, the marked subset is
exactly the admissible subset. -/
theorem rejectsMultiB_eq_admissibleMultiPathB_of_witness
    (d : Graph) (hslot : canonicalSlotOKB d = true)
    {x : MultiPathInput} (hx : rejectsMultiB d x = true) :
    ∀ y, rejectsMultiB d y = admissibleMultiPathB d y := by
  intro y
  cases hyB : admissibleMultiPathB d y with
  | false => simp [rejectsMultiB, hyB]
  | true =>
      have hy : AdmissibleMultiPath d y := hyB
      have hrejected := rejectsMultiB_persists_on_admissible d hslot hx hy
      simp [hrejected]

theorem count_rejectsMultiB_eq_admissible
    (d : Graph) (hslot : canonicalSlotOKB d = true)
    {x : MultiPathInput} (hx : rejectsMultiB d x = true)
    (samples : List MultiPathInput) :
    (samples.filter (rejectsMultiB d)).length =
    (samples.filter (admissibleMultiPathB d)).length := by
  have hpoint := rejectsMultiB_eq_admissibleMultiPathB_of_witness d hslot hx
  rw [funext hpoint]

namespace Examples.Readings.MultiToken

/-- A real rejecting point for the existing phase predicate: the canonical
`d11` multi-path is admissible, error-free, and not discharged. -/
theorem d11_canonical_slot_ok : canonicalSlotOKB d11 = true := by
  native_decide

theorem d11_canonical_rejects :
    rejectsMultiB d11 (multiPathsFromFlow d11) = true := by
  native_decide

theorem d11_canonical_diagnostic :
    multiVerdict d11 (multiPathsFromFlow d11) = (false, false, false) := by
  native_decide

/-- Vary two co-located route/token slots. -/
def d11Swapped : MultiPathInput :=
  swapSlots (multiPathsFromFlow d11) 0 1

theorem d11_swapped_admissible :
    admissibleMultiPathB d11 d11Swapped = true := by
  native_decide

theorem d11_swapped_rejects : rejectsMultiB d11 d11Swapped = true := by
  exact rejectsMultiB_persists_on_admissible d11 d11_canonical_slot_ok
    d11_canonical_rejects d11_swapped_admissible

/-- Stop the first route immediately, before its original rule arrivals. -/
def d11Stopped : MultiPathInput :=
  setCol (multiPathsFromFlow d11) 0 []

theorem d11_stopped_is_inadmissible :
    admissibleMultiPathB d11 d11Stopped = false := by
  native_decide

theorem d11_stopped_is_not_marked : rejectsMultiB d11 d11Stopped = false := by
  native_decide

/-- Concrete specialization of the exact counting statement: in every finite
sample of `d11` inputs, marked points and admissible points have equal count.
Thus the conditional density inside the admissible domain is exactly one. -/
theorem d11_count_rejects_eq_admissible (samples : List MultiPathInput) :
    (samples.filter (rejectsMultiB d11)).length =
      (samples.filter (admissibleMultiPathB d11)).length :=
  count_rejectsMultiB_eq_admissible d11 d11_canonical_slot_ok
    d11_canonical_rejects samples

end Examples.Readings.MultiToken

/-! ## One real selector coordinate and one real elimination mismatch -/

namespace Readings.LocalLabels

open Readings.Selection

/-- The selected outgoing edge exists and its stored dependency differs from
the dependency computed by the actual rule operation `outDep`. -/
def selectedDecorationMismatchB (d : Graph)
    (reading : ReadingData (fanArities d)) (v : Vertex) : Bool :=
  match selectedEdge d reading v with
  | none => false
  | some selected => decide (selected.DEPENDENCY ≠ outDep v d)

/-- `selectedEdge` reads exactly the selector at `branchIndex d v` for a fan,
and reads no selector at a node of outdegree zero or one. -/
theorem selectedEdge_eq_of_coordinate_eq
    (d : Graph) (v : Vertex)
    {x y : ReadingData (fanArities d)}
    (hcoordinate :
      (x.indices[branchIndex d v]?).getD 0 =
        (y.indices[branchIndex d v]?).getD 0) :
    selectedEdge d x v = selectedEdge d y v := by
  by_cases hfan : 2 ≤ (outgoing d v).length
  · simp [selectedEdge, hfan, hcoordinate]
  · simp [selectedEdge, hfan]

/-- Persistence of the local mismatch under agreement at the one selector
coordinate actually read by `selectedEdge`.  The graph, including both the
stored edge label and the premise labels read by `outDep`, remains fixed. -/
theorem selectedDecorationMismatchB_persists
    (d : Graph) (v : Vertex)
    {x y : ReadingData (fanArities d)}
    (hx : selectedDecorationMismatchB d x v = true)
    (hcoordinate :
      (x.indices[branchIndex d v]?).getD 0 =
        (y.indices[branchIndex d v]?).getD 0) :
    selectedDecorationMismatchB d y v = true := by
  unfold selectedDecorationMismatchB at hx ⊢
  rw [← selectedEdge_eq_of_coordinate_eq d v hcoordinate]
  exact hx

/-- Activation guard used by the local reading observation. -/
def selectedRuleActiveB (d : Graph)
    (reading : ReadingData (fanArities d)) (v : Vertex) : Bool :=
  decide (v ∈ selectedReachable d reading) && coherentAtB d reading v

/-- Operationally guarded local observation for a reading: the node is on the
selected reachable region, its elimination premises (when applicable) are
coherent for this same reading, and the selected outgoing decoration is
wrong.  This is still an intermediate observation, not `rejectsMultiB`. -/
def selectedRuleApplicationMismatchB (d : Graph)
    (reading : ReadingData (fanArities d)) (v : Vertex) : Bool :=
  selectedRuleActiveB d reading v && selectedDecorationMismatchB d reading v

/-- The generic operational observation is preserved when its activation
guard and the one edge-selector coordinate are preserved. -/
theorem selectedRuleApplicationMismatchB_persists
    (d : Graph) (v : Vertex)
    {x y : ReadingData (fanArities d)}
    (hx : selectedRuleApplicationMismatchB d x v = true)
    (hactiveY : selectedRuleActiveB d y v = true)
    (hcoordinate :
      (x.indices[branchIndex d v]?).getD 0 =
        (y.indices[branchIndex d v]?).getD 0) :
    selectedRuleApplicationMismatchB d y v = true := by
  have hxParts :
      selectedRuleActiveB d x v = true ∧
        selectedDecorationMismatchB d x v = true := by
    simpa [selectedRuleApplicationMismatchB] using hx
  have hlocal : selectedDecorationMismatchB d y v = true :=
    selectedDecorationMismatchB_persists d v hxParts.2 hcoordinate
  simp [selectedRuleApplicationMismatchB, hactiveY, hlocal]

namespace Examples.Elimination

open Readings.Selection.Examples.ColouredCompression

/-- Poison one of the two outgoing edges of the genuine elimination at `a2`.
The premise labels and every other record of the DLDS are unchanged. -/
def poisonedA2Edge : Deduction :=
  edge a2 a3ToA4 1 [A5]

def poisonA2Outgoing (candidate : Deduction) : Deduction :=
  if candidate = edgeA2A3ToA4 then poisonedA2Edge else candidate

def graph : Graph :=
  dlds Readings.Selection.Examples.ColouredCompression.graph.NODES
    (Readings.Selection.Examples.ColouredCompression.graph.EDGES.map poisonA2Outgoing)
    Readings.Selection.Examples.ColouredCompression.graph.PATHS

theorem fanArities_graph : fanArities graph = [2, 2, 2] := by
  native_decide

theorem branchPoint_numbers :
    (branchPoints graph).map (·.NUMBER) = [0, 5, 7] := by
  native_decide

theorem a2_branch_index : branchIndex graph a2 = 1 := by
  native_decide

theorem a2_is_elimination : ruleShapeOKB a2 graph = true := by
  native_decide

theorem a2_computed_dependency :
    outDep a2 graph = normalizeDep [A1, h2Formula] := by
  native_decide

theorem a2_outgoing_edges :
    outgoing graph a2 = [edgeA2A3, poisonedA2Edge] := by
  native_decide

theorem raw_reading_count : (readings graph).length = 8 := by
  native_decide

/-- The global coherence predicate accepts no reading of
this SBCCQ-shaped fixture.  Consequently the counts below are explicitly over
the raw dependent product, not over a nonempty `coherentReadings` domain. -/
theorem coherent_reading_count : (coherentReadings graph).length = 0 := by
  native_decide

/-- Complete executable table for the declared finite raw-reading enumeration.
The middle coordinate is the selector of the elimination node `a2`; changing
the first (earlier) or third (later) coordinate preserves the mismatch. -/
theorem reading_mismatch_table :
    (readings graph).map (fun reading =>
      (reading.indices, selectedDecorationMismatchB graph reading a2)) =
      [([0, 0, 0], false), ([0, 0, 1], false),
       ([0, 1, 0], true),  ([0, 1, 1], true),
       ([1, 0, 0], false), ([1, 0, 1], false),
       ([1, 1, 0], true),  ([1, 1, 1], true)] := by
  native_decide

/-- Once rule activation is included, the earlier selector matters.  The
operationally sufficient cylinder fixes coordinate `0` to `0` (both premises
feed `a2`) and coordinate `1` to `1` (the poisoned outgoing edge); coordinate
`2`, belonging to the later elimination `a3`, remains free. -/
theorem rule_application_mismatch_table :
    (readings graph).map (fun reading =>
      (reading.indices, selectedRuleApplicationMismatchB graph reading a2)) =
      [([0, 0, 0], false), ([0, 0, 1], false),
       ([0, 1, 0], true),  ([0, 1, 1], true),
       ([1, 0, 0], false), ([1, 0, 1], false),
       ([1, 1, 0], false), ([1, 1, 1], false)] := by
  native_decide

theorem rule_application_mismatch_count :
    ((readings graph).filter fun reading =>
      selectedRuleApplicationMismatchB graph reading a2).length = 2 := by
  native_decide

/-- Exact raw-product density for the operational local observation:
`2 / 8 = 1 / 4`, i.e. two independent binary coordinates fixed. -/
theorem rule_application_mismatch_ratio :
    ((readings graph).filter fun reading =>
      selectedRuleApplicationMismatchB graph reading a2).length * 4 =
        (readings graph).length := by
  native_decide

theorem exposed_reading_count :
    ((readings graph).filter fun reading =>
      selectedDecorationMismatchB graph reading a2).length = 4 := by
  native_decide

/-- The exact selector-cylinder density in the raw `ReadingData` product for
this local observation is `4 / 8 = 1 / 2`: one binary coordinate is fixed and
the other two binary coordinates are free. -/
theorem exposed_reading_ratio :
    ((readings graph).filter fun reading =>
      selectedDecorationMismatchB graph reading a2).length * 2 =
        (readings graph).length := by
  native_decide

end Examples.Elimination
end Readings.LocalLabels


end DLDS
