import Readings.Explicit.FanFamily

open scoped Classical

set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unnecessarySimpa false

/-!
# Interacting premise selectors

Two binary source fans independently route the minor and major premises to
one of two implication eliminations.  Coherence is therefore determined by
whether those choices meet at the same elimination.  A third, variable-width
fan chooses an explicit output of `e0`.
-/

namespace DLDS
namespace Readings.Explicit
namespace InteractingPremises

open Flow
open Readings.Selection
open DLDS.Readings.LocalLabels
open DLDS.Readings.Explicit.UniformElimFanFamily

def A : Formula := #"interaction_A"
def B : Formula := #"interaction_B"
def R : Formula := #"interaction_R"
def Z : Formula := #"interaction_bad"
def AB : Formula := A >> B

def hA : Vertex := node 0 3 A true false []
def hAB : Vertex := node 1 3 AB true false []
def e0 : Vertex := node 2 2 B false false []
def e1 : Vertex := node 3 2 B false false []
def root : Vertex := node 4 1 R false false []

private theorem vertex_ne_of_number_ne {u v : Vertex}
    (hnumber : u.NUMBER ≠ v.NUMBER) : u ≠ v := by
  intro huv
  exact hnumber (congrArg Vertex.NUMBER huv)

@[simp] private theorem hA_ne_hAB : hA ≠ hAB :=
  vertex_ne_of_number_ne (by simp [hA, hAB])

@[simp] private theorem hA_ne_e0 : hA ≠ e0 :=
  vertex_ne_of_number_ne (by simp [hA, e0])

@[simp] private theorem hA_ne_e1 : hA ≠ e1 :=
  vertex_ne_of_number_ne (by simp [hA, e1])

@[simp] private theorem hA_ne_root : hA ≠ root :=
  vertex_ne_of_number_ne (by simp [hA, root])

@[simp] private theorem hAB_ne_e0 : hAB ≠ e0 :=
  vertex_ne_of_number_ne (by simp [hAB, e0])

@[simp] private theorem hAB_ne_e1 : hAB ≠ e1 :=
  vertex_ne_of_number_ne (by simp [hAB, e1])

@[simp] private theorem hAB_ne_root : hAB ≠ root :=
  vertex_ne_of_number_ne (by simp [hAB, root])

@[simp] private theorem e0_ne_e1 : e0 ≠ e1 :=
  vertex_ne_of_number_ne (by simp [e0, e1])

@[simp] private theorem e0_ne_root : e0 ≠ root :=
  vertex_ne_of_number_ne (by simp [e0, root])

@[simp] private theorem e1_ne_root : e1 ≠ root :=
  vertex_ne_of_number_ne (by simp [e1, root])

@[simp] private theorem hAB_ne_hA : hAB ≠ hA := Ne.symm hA_ne_hAB
@[simp] private theorem e0_ne_hA : e0 ≠ hA := Ne.symm hA_ne_e0
@[simp] private theorem e1_ne_hA : e1 ≠ hA := Ne.symm hA_ne_e1
@[simp] private theorem root_ne_hA : root ≠ hA := Ne.symm hA_ne_root
@[simp] private theorem e0_ne_hAB : e0 ≠ hAB := Ne.symm hAB_ne_e0
@[simp] private theorem e1_ne_hAB : e1 ≠ hAB := Ne.symm hAB_ne_e1
@[simp] private theorem root_ne_hAB : root ≠ hAB := Ne.symm hAB_ne_root
@[simp] private theorem e1_ne_e0 : e1 ≠ e0 := Ne.symm e0_ne_e1
@[simp] private theorem root_ne_e0 : root ≠ e0 := Ne.symm e0_ne_root
@[simp] private theorem root_ne_e1 : root ≠ e1 := Ne.symm e1_ne_root

@[simp] private theorem hAB_beq_hA : (hAB == hA) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hA_ne_hAB)

@[simp] private theorem e0_beq_hA : (e0 == hA) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hA_ne_e0)

@[simp] private theorem e1_beq_hA : (e1 == hA) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hA_ne_e1)

@[simp] private theorem root_beq_hA : (root == hA) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hA_ne_root)

@[simp] private theorem e0_beq_hAB : (e0 == hAB) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hAB_ne_e0)

@[simp] private theorem e1_beq_hAB : (e1 == hAB) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hAB_ne_e1)

@[simp] private theorem root_beq_hAB : (root == hAB) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm hAB_ne_root)

@[simp] private theorem e1_beq_e0 : (e1 == e0) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm e0_ne_e1)

@[simp] private theorem e0_beq_e1 : (e0 == e1) = false :=
  beq_eq_false_iff_ne.mpr e0_ne_e1

@[simp] private theorem root_beq_e0 : (root == e0) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm e0_ne_root)

@[simp] private theorem root_beq_e1 : (root == e1) = false :=
  beq_eq_false_iff_ne.mpr (Ne.symm e1_ne_root)

def minorEdge (choice : Nat) : Deduction :=
  edge hA (if choice = 0 then e0 else e1) choice [A]

def majorEdge (choice : Nat) : Deduction :=
  edge hAB (if choice = 0 then e0 else e1) choice [AB]

def premiseEdges : List Deduction :=
  [minorEdge 0, minorEdge 1, majorEdge 0, majorEdge 1]

def goodDep : Dep := normalizeDep [A, AB]

def e0Output (bad : Bool) (choice : Nat) : Deduction :=
  edge e0 root choice (if bad && choice == 1 then [Z] else goodDep)

def e0Outputs (width : Nat) (bad : Bool) : List Deduction :=
  (List.range (width + 2)).map (e0Output bad)

def e1Output : Deduction := edge e1 root 0 goodDep

def graph (width : Nat) (bad : Bool) : Graph :=
  dlds [hA, hAB, e0, e1, root]
    (premiseEdges ++ e0Outputs width bad ++ [e1Output]) []

abbrev control (width : Nat) : Graph := graph width false
abbrev defective (width : Nat) : Graph := graph width true

def IsInteractingPremises (d : Graph) : Prop :=
  ∃ width bad, d = graph width bad

theorem fixed_structure_when_label_changes (width : Nat) :
    (control width).NODES = (defective width).NODES ∧
    ((control width).EDGES.map fun e => (e.START, e.END, e.COLOUR)) =
      ((defective width).EDGES.map fun e => (e.START, e.END, e.COLOUR)) ∧
    (control width).PATHS = (defective width).PATHS := by
  simp [control, defective, graph, premiseEdges, e0Outputs, e0Output]

private theorem filter_e0Outputs_start_e0 (width : Nat) (bad : Bool) :
    (e0Outputs width bad).filter (fun e => decide (e.START = e0)) =
      e0Outputs width bad := by
  apply List.filter_eq_self.mpr
  intro edge hedge
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
  simp [e0Output]

private theorem filter_e0Outputs_start_ne (width : Nat) (bad : Bool)
    (v : Vertex) (hv : v ≠ e0) :
    (e0Outputs width bad).filter (fun e => decide (e.START = v)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro edge hedge
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
  simpa [e0Output, Ne.symm hv]

theorem outgoing_hA (width : Nat) (bad : Bool) :
    outgoing (graph width bad) hA = [minorEdge 0, minorEdge 1] := by
  rw [show outgoing (graph width bad) hA =
    (graph width bad).EDGES.filter (fun e => decide (e.START = hA)) from
      outgoing_eq_filter hA _]
  simp [graph, premiseEdges, e0Outputs, e0Output, e1Output,
    minorEdge, majorEdge, hA, hAB, e0, e1, root]

theorem outgoing_hAB (width : Nat) (bad : Bool) :
    outgoing (graph width bad) hAB = [majorEdge 0, majorEdge 1] := by
  rw [show outgoing (graph width bad) hAB =
    (graph width bad).EDGES.filter (fun e => decide (e.START = hAB)) from
      outgoing_eq_filter hAB _]
  simp [graph, premiseEdges, e0Outputs, e0Output, e1Output,
    minorEdge, majorEdge, hA, hAB, e0, e1, root]

theorem outgoing_e0 (width : Nat) (bad : Bool) :
    outgoing (graph width bad) e0 = e0Outputs width bad := by
  rw [show outgoing (graph width bad) e0 =
    (graph width bad).EDGES.filter (fun e => decide (e.START = e0)) from
      outgoing_eq_filter e0 _]
  simp only [graph, List.filter_append]
  change
    (premiseEdges.filter
        (fun edge => decide (edge.START = e0)) ++
      (e0Outputs width bad).filter (fun edge => decide (edge.START = e0))) ++
      [e1Output].filter (fun edge => decide (edge.START = e0)) =
        e0Outputs width bad
  rw [filter_e0Outputs_start_e0]
  simp [premiseEdges, minorEdge, majorEdge, hA, hAB, e0, e1, e1Output]

theorem outgoing_e1 (width : Nat) (bad : Bool) :
    outgoing (graph width bad) e1 = [e1Output] := by
  rw [show outgoing (graph width bad) e1 =
    (graph width bad).EDGES.filter (fun e => decide (e.START = e1)) from
      outgoing_eq_filter e1 _]
  simp only [graph, premiseEdges, List.filter_append, List.filter_cons,
    List.filter_nil, List.append_nil]
  rw [filter_e0Outputs_start_ne width bad e1 (by simp [e0, e1])]
  simp [minorEdge, majorEdge, hA, hAB, e0, e1, e1Output]

theorem outgoing_root (width : Nat) (bad : Bool) :
    outgoing (graph width bad) root = [] := by
  rw [show outgoing (graph width bad) root =
    (graph width bad).EDGES.filter (fun e => decide (e.START = root)) from
      outgoing_eq_filter root _]
  apply List.filter_eq_nil_iff.mpr
  intro edge hedge
  simp [graph, premiseEdges, e0Outputs, e0Output, e1Output,
    minorEdge, majorEdge, hA, hAB, e0, e1, root] at hedge
  rcases hedge with rfl | rfl | rfl | rfl | ⟨i, _hi, rfl⟩ | rfl <;>
    simp [minorEdge, majorEdge, e0Output, e1Output, hA, hAB, e0, e1, root]

theorem fanNodes_graph (width : Nat) (bad : Bool) :
    (graph width bad).NODES.filter
      (fun v => decide (2 ≤ (outgoing (graph width bad) v).length)) =
      [hA, hAB, e0] := by
  change [hA, hAB, e0, e1, root].filter _ = _
  simp [outgoing_hA, outgoing_hAB, outgoing_e0, outgoing_e1,
    outgoing_root, e0Outputs]

theorem branchPoints_graph (width : Nat) (bad : Bool) :
    branchPoints (graph width bad) = [hA, hAB, e0] := by
  unfold branchPoints
  rw [fanNodes_graph]
  let cmp : Vertex → Vertex → Bool :=
    fun u v => decide (u.NUMBER ≤ v.NUMBER)
  generalize hsortedList : ([hA, hAB, e0].mergeSort cmp) = sortedList
  have hperm : sortedList.Perm [hA, hAB, e0] := by
    rw [← hsortedList]
    exact List.mergeSort_perm _ _
  have hsorted : sortedList.Pairwise (fun u v => cmp u v = true) := by
    rw [← hsortedList]
    apply List.pairwise_mergeSort
    · intro a b c hab hbc
      simp [cmp] at hab hbc ⊢
      omega
    · intro a b
      simp [cmp]
      omega
  have hlen : sortedList.length = 3 := by
    simpa using hperm.length_eq
  cases sortedList with
  | nil => simp at hlen
  | cons x xs =>
      cases xs with
      | nil => simp at hlen
      | cons y ys =>
          cases ys with
          | nil => simp at hlen
          | cons z zs =>
              cases zs with
              | cons w ws => simp at hlen
              | nil =>
                  have hx : x = hA ∨ x = hAB ∨ x = e0 := by
                    have : x ∈ [hA, hAB, e0] := hperm.mem_iff.mp (by simp)
                    simpa using this
                  have hy : y = hA ∨ y = hAB ∨ y = e0 := by
                    have : y ∈ [hA, hAB, e0] := hperm.mem_iff.mp (by simp)
                    simpa using this
                  have hz : z = hA ∨ z = hAB ∨ z = e0 := by
                    have : z ∈ [hA, hAB, e0] := hperm.mem_iff.mp (by simp)
                    simpa using this
                  have hnodup : [x, y, z].Nodup := by
                    exact hperm.nodup_iff.mpr (by
                      simp [hA, hAB, e0])
                  rcases hx with rfl | rfl | rfl <;>
                    rcases hy with rfl | rfl | rfl <;>
                    rcases hz with rfl | rfl | rfl <;>
                    simp [cmp, hA, hAB, e0] at hnodup hsorted ⊢

theorem fanArities_graph (width : Nat) (bad : Bool) :
    fanArities (graph width bad) = [2, 2, width + 2] := by
  simp [fanArities, branchPoints_graph, outgoing_hA, outgoing_hAB,
    outgoing_e0, e0Outputs]

def reading (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    ReadingData (fanArities (graph width bad)) :=
  (fanArities_graph width bad).symm ▸
    .cons minorChoice (.cons majorChoice (.cons outputChoice .nil))

theorem reading_indices (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    (reading width bad minorChoice majorChoice outputChoice).indices =
      [minorChoice.val, majorChoice.val, outputChoice.val] := by
  simp [reading, Readings.Selection.ReadingData.indices]

theorem exists_reading_choices (width : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width bad))) :
    ∃ minorChoice majorChoice : Fin 2, ∃ outputChoice : Fin (width + 2),
      r = reading width bad minorChoice majorChoice outputChoice := by
  let h := fanArities_graph width bad
  let r' : ReadingData [2, 2, width + 2] := h ▸ r
  have hdecomp :
      ∃ minorChoice majorChoice : Fin 2, ∃ outputChoice : Fin (width + 2),
        r' = .cons minorChoice (.cons majorChoice (.cons outputChoice .nil)) := by
    cases r' with
    | cons minorChoice rest =>
        cases rest with
        | cons majorChoice rest' =>
            cases rest' with
            | cons outputChoice rest'' =>
                cases rest''
                exact ⟨minorChoice, majorChoice, outputChoice, rfl⟩
  obtain ⟨minorChoice, majorChoice, outputChoice, hr'⟩ := hdecomp
  refine ⟨minorChoice, majorChoice, outputChoice, ?_⟩
  calc
    r = h.symm ▸ r' :=
      (DLDS.Readings.Explicit.UniformElimFanFamily.readingData_cast_symm_cast h r).symm
    _ = h.symm ▸
        (.cons minorChoice (.cons majorChoice (.cons outputChoice .nil))) := by
      rw [hr']
    _ = reading width bad minorChoice majorChoice outputChoice := by
      simp [reading, h]

private theorem filter_e0Outputs_end_ne (width : Nat) (bad : Bool)
    (v : Vertex) (hv : v ≠ root) :
    (e0Outputs width bad).filter (fun edge => decide (edge.END = v)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro edge hedge
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
  simpa [e0Output, Ne.symm hv]

theorem incoming_hA (width : Nat) (bad : Bool) :
    get_rule.incoming hA (graph width bad) = [] := by
  rw [incoming_eq_filter]
  apply List.filter_eq_nil_iff.mpr
  intro edge hedge
  simp [graph, premiseEdges, e0Outputs, e0Output, e1Output,
    minorEdge, majorEdge, hA, hAB, e0, e1, root] at hedge
  rcases hedge with rfl | rfl | rfl | rfl | ⟨i, _hi, rfl⟩ | rfl <;>
    simp [minorEdge, majorEdge, e0Output, e1Output, hA, hAB, e0, e1, root]

theorem incoming_hAB (width : Nat) (bad : Bool) :
    get_rule.incoming hAB (graph width bad) = [] := by
  rw [incoming_eq_filter]
  apply List.filter_eq_nil_iff.mpr
  intro edge hedge
  simp [graph, premiseEdges, e0Outputs, e0Output, e1Output,
    minorEdge, majorEdge, hA, hAB, e0, e1, root] at hedge
  rcases hedge with rfl | rfl | rfl | rfl | ⟨i, _hi, rfl⟩ | rfl <;>
    simp [minorEdge, majorEdge, e0Output, e1Output, hA, hAB, e0, e1, root]

theorem incoming_e0 (width : Nat) (bad : Bool) :
    get_rule.incoming e0 (graph width bad) = [minorEdge 0, majorEdge 0] := by
  rw [incoming_eq_filter]
  simp only [graph, premiseEdges, List.filter_append, List.filter_cons,
    List.filter_nil, List.append_nil]
  rw [filter_e0Outputs_end_ne width bad e0 (by simp [e0, root])]
  simp [minorEdge, majorEdge, e0Output, e1Output, e0, e1, root]

theorem incoming_e1 (width : Nat) (bad : Bool) :
    get_rule.incoming e1 (graph width bad) = [minorEdge 1, majorEdge 1] := by
  rw [incoming_eq_filter]
  simp only [graph, premiseEdges, List.filter_append, List.filter_cons,
    List.filter_nil, List.append_nil]
  rw [filter_e0Outputs_end_ne width bad e1 (by simp [e1, root])]
  simp [minorEdge, majorEdge, e0Output, e1Output, e0, e1, root]

theorem incoming_root (width : Nat) (bad : Bool) :
    get_rule.incoming root (graph width bad) =
      e0Outputs width bad ++ [e1Output] := by
  rw [incoming_eq_filter]
  simp only [graph, premiseEdges, List.filter_append]
  change
    premiseEdges.filter (fun edge => decide (edge.END = root)) ++
      (e0Outputs width bad).filter (fun edge => decide (edge.END = root)) ++
      [e1Output].filter (fun edge => decide (edge.END = root)) = _
  have he0 :
      (e0Outputs width bad).filter (fun edge => decide (edge.END = root)) =
        e0Outputs width bad := by
    apply List.filter_eq_self.mpr
    intro edge hedge
    obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
    simp [e0Output]
  rw [he0]
  simp [premiseEdges, minorEdge, majorEdge, e1Output, e0, e1, root]

theorem predsOf_e0 (width : Nat) (bad : Bool) :
    predsOf (graph width bad) e0 = [hA, hAB] := by
  rw [predsOf_eq, incoming_e0]
  change List.eraseDups [hA, hAB] = [hA, hAB]
  have hne : hAB ≠ hA := by
    intro h
    have := congrArg Vertex.NUMBER h
    simp [hAB, hA] at this
  have hbeq : (hAB == hA) = false := beq_eq_false_iff_ne.mpr hne
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop, hbeq]

theorem predsOf_e1 (width : Nat) (bad : Bool) :
    predsOf (graph width bad) e1 = [hA, hAB] := by
  rw [predsOf_eq, incoming_e1]
  change List.eraseDups [hA, hAB] = [hA, hAB]
  have hne : hAB ≠ hA := by
    intro h
    have := congrArg Vertex.NUMBER h
    simp [hAB, hA] at this
  have hbeq : (hAB == hA) = false := beq_eq_false_iff_ne.mpr hne
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop, hbeq]

theorem formula_of_mem_predsOf_root (width : Nat) (bad : Bool)
    {u : Vertex} (hu : u ∈ predsOf (graph width bad) root) :
    u.FORMULA = B := by
  rw [predsOf_eq, incoming_root] at hu
  have humap := List.mem_eraseDups.mp hu
  obtain ⟨edge, hedge, rfl⟩ := List.mem_map.mp humap
  rw [List.mem_append] at hedge
  rcases hedge with hedge | hedge
  · obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
    rfl
  · simp only [List.mem_singleton] at hedge
    subst edge
    rfl

theorem predsOf_hA (width : Nat) (bad : Bool) :
    predsOf (graph width bad) hA = [] := by
  rw [predsOf_eq, incoming_hA]
  rfl

theorem predsOf_hAB (width : Nat) (bad : Bool) :
    predsOf (graph width bad) hAB = [] := by
  rw [predsOf_eq, incoming_hAB]
  rfl

theorem predsOf_root_ne_nil (width : Nat) (bad : Bool) :
    predsOf (graph width bad) root ≠ [] := by
  intro hempty
  have he1 : e1 ∈ predsOf (graph width bad) root := by
    rw [predsOf_eq, incoming_root]
    apply List.mem_eraseDups.mpr
    apply List.mem_map.mpr
    exact ⟨e1Output, by simp, by simp [e1Output]⟩
  rw [hempty] at he1
  simp at he1

theorem sourceNodes_graph (width : Nat) (bad : Bool) :
    sourceNodes (graph width bad) = [hA, hAB] := by
  change [hA, hAB, e0, e1, root].filter _ = _
  simp [sourceNodes, predsOf_hA, predsOf_hAB, predsOf_e0, predsOf_e1,
    predsOf_root_ne_nil]

theorem elimPairsAt_hA (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) hA = [] := by
  simp [elimPairsAt, predsOf_eq, incoming_hA]

theorem elimPairsAt_hAB (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) hAB = [] := by
  simp [elimPairsAt, predsOf_eq, incoming_hAB]

theorem elimPairsAt_e0 (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) e0 = [(hAB, hA)] := by
  unfold elimPairsAt
  rw [predsOf_e0]
  simp [hA, hAB, e0, A, B, AB]

theorem elimPairsAt_e1 (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) e1 = [(hAB, hA)] := by
  unfold elimPairsAt
  rw [predsOf_e1]
  simp [hA, hAB, e1, A, B, AB]

theorem elimPairsAt_root (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) root = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro pair hpair
  unfold elimPairsAt at hpair
  obtain ⟨major, hmajor, hinner⟩ := List.mem_flatMap.mp hpair
  obtain ⟨minor, hminor, hselected⟩ := List.mem_filterMap.mp hinner
  split at hselected <;> rename_i hcondition
  · have hformula := formula_of_mem_predsOf_root width bad hmajor
    rw [hformula] at hcondition
    simp [B] at hcondition
  · simp at hselected

theorem branchIndex_hA (width : Nat) (bad : Bool) :
    branchIndex (graph width bad) hA = 0 := by
  simp [branchIndex, branchPoints_graph]

theorem branchIndex_hAB (width : Nat) (bad : Bool) :
    branchIndex (graph width bad) hAB = 1 := by
  simp [branchIndex, branchPoints_graph, hA, hAB]

theorem branchIndex_e0 (width : Nat) (bad : Bool) :
    branchIndex (graph width bad) e0 = 2 := by
  simp [branchIndex, branchPoints_graph, hA, hAB, e0]

theorem selectedEdge_reading_hA (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hA =
      some (minorEdge minorChoice.val) := by
  fin_cases minorChoice <;>
    simp [selectedEdge, outgoing_hA, reading_indices, branchIndex_hA]

theorem selectedEdge_reading_hAB (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hAB =
      some (majorEdge majorChoice.val) := by
  fin_cases majorChoice <;>
    simp [selectedEdge, outgoing_hAB, reading_indices, branchIndex_hAB]

theorem selectedEdge_reading_e0 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) e0 =
      some (e0Output bad outputChoice.val) := by
  simp [selectedEdge, outgoing_e0, e0Outputs, reading_indices, branchIndex_e0]

theorem selectedEdge_reading_e1 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) e1 =
      some e1Output := by
  simp [selectedEdge, outgoing_e1]

theorem selectedEdge_reading_root (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) root = none := by
  simp [selectedEdge, outgoing_root]

private def nextVertices (d : Graph)
    (r : ReadingData (fanArities d)) (frontier seen : List Vertex) :
    List Vertex :=
  frontier.filterMap (selectedEdge d r)
    |>.map (·.END)
    |>.filter (fun v => !(seen.contains v))
    |>.eraseDups

private theorem selectedReachable_go_succ (d : Graph)
    (r : ReadingData (fanArities d)) (fuel : Nat)
    (frontier seen : List Vertex) :
    selectedReachable.go d r (fuel + 1) frontier seen =
      if (nextVertices d r frontier seen).isEmpty then seen.eraseDups
      else selectedReachable.go d r fuel (nextVertices d r frontier seen)
        (seen ++ nextVertices d r frontier seen) := by
  rfl

private theorem first_next_reading (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice)
        [hA, hAB] [hA, hAB] =
      if minorChoice.val = 0 then
        if majorChoice.val = 0 then [e0] else [e0, e1]
      else
        if majorChoice.val = 0 then [e1, e0] else [e1] := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    unfold nextVertices <;>
    simp only [List.filterMap_cons, List.filterMap_nil] <;>
    rw [selectedEdge_reading_hA, selectedEdge_reading_hAB] <;>
    simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
      minorEdge, majorEdge]

private theorem second_next_e0 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice)
        [e0] [hA, hAB, e0] = [root] := by
  unfold nextVertices
  simp only [List.filterMap_cons, List.filterMap_nil]
  rw [selectedEdge_reading_e0]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
    e0Output]

private theorem second_next_e1 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice)
        [e1] [hA, hAB, e1] = [root] := by
  unfold nextVertices
  simp only [List.filterMap_cons, List.filterMap_nil]
  rw [selectedEdge_reading_e1]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
    e1Output]

private theorem second_next_e0_e1 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice)
        [e0, e1] [hA, hAB, e0, e1] = [root] := by
  unfold nextVertices
  simp only [List.filterMap_cons, List.filterMap_nil]
  rw [selectedEdge_reading_e0, selectedEdge_reading_e1]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
    e0Output, e1Output]

private theorem second_next_e1_e0 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice)
        [e1, e0] [hA, hAB, e1, e0] = [root] := by
  unfold nextVertices
  simp only [List.filterMap_cons, List.filterMap_nil]
  rw [selectedEdge_reading_e1, selectedEdge_reading_e0]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
    e0Output, e1Output]

private theorem next_root_empty (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2))
    (seen : List Vertex) :
    nextVertices (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice)
        [root] seen = [] := by
  unfold nextVertices
  simp only [List.filterMap_cons, List.filterMap_nil]
  rw [selectedEdge_reading_root]
  simp

theorem selectedReachable_reading (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedReachable (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) =
      if minorChoice.val = 0 then
        if majorChoice.val = 0 then [hA, hAB, e0, root]
        else [hA, hAB, e0, e1, root]
      else
        if majorChoice.val = 0 then [hA, hAB, e1, e0, root]
        else [hA, hAB, e1, root] := by
  fin_cases minorChoice <;> fin_cases majorChoice
  · unfold selectedReachable
    rw [sourceNodes_graph, selectedReachable_go_succ, first_next_reading]
    simp
    rw [show (graph width bad).NODES.length = 5 from rfl]
    rw [show 5 = 4 + 1 by omega, selectedReachable_go_succ, second_next_e0]
    simp only [List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte]
    rw [show 4 = 3 + 1 by omega, selectedReachable_go_succ, next_root_empty]
    simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
      hAB_beq_hA, e0_beq_hA, e0_beq_hAB, root_beq_hA,
      root_beq_hAB, root_beq_e0]
  · unfold selectedReachable
    rw [sourceNodes_graph, selectedReachable_go_succ, first_next_reading]
    simp
    rw [show (graph width bad).NODES.length = 5 from rfl]
    rw [show 5 = 4 + 1 by omega, selectedReachable_go_succ,
      second_next_e0_e1]
    simp only [List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte]
    rw [show 4 = 3 + 1 by omega, selectedReachable_go_succ, next_root_empty]
    simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
      hAB_beq_hA, e0_beq_hA, e0_beq_hAB, e1_beq_hA,
      e1_beq_hAB, e1_beq_e0, root_beq_hA, root_beq_hAB,
      root_beq_e0, root_beq_e1]
  · unfold selectedReachable
    rw [sourceNodes_graph, selectedReachable_go_succ, first_next_reading]
    simp
    rw [show (graph width bad).NODES.length = 5 from rfl]
    rw [show 5 = 4 + 1 by omega, selectedReachable_go_succ,
      second_next_e1_e0]
    simp only [List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte]
    rw [show 4 = 3 + 1 by omega, selectedReachable_go_succ, next_root_empty]
    simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
      hAB_beq_hA, e1_beq_hA, e1_beq_hAB, e0_beq_hA,
      e0_beq_hAB, e0_beq_e1, root_beq_hA, root_beq_hAB,
      root_beq_e0, root_beq_e1]
  · unfold selectedReachable
    rw [sourceNodes_graph, selectedReachable_go_succ, first_next_reading]
    simp
    rw [show (graph width bad).NODES.length = 5 from rfl]
    rw [show 5 = 4 + 1 by omega, selectedReachable_go_succ, second_next_e1]
    simp only [List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte]
    rw [show 4 = 3 + 1 by omega, selectedReachable_go_succ, next_root_empty]
    simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop,
      hAB_beq_hA, e1_beq_hA, e1_beq_hAB, root_beq_hA,
      root_beq_hAB, root_beq_e1]

theorem coherentAtB_hA (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hA = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_hA]

theorem coherentAtB_hAB (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hAB = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_hAB]

theorem coherentAtB_e0 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) e0 =
      decide (minorChoice.val = majorChoice.val) := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    unfold coherentAtB <;>
    rw [selectedReachable_reading, elimPairsAt_e0] <;>
    simp [feedsB, selectedEdge_reading_hA, selectedEdge_reading_hAB,
      minorEdge, majorEdge]

theorem coherentAtB_e1 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) e1 =
      decide (minorChoice.val = majorChoice.val) := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    unfold coherentAtB <;>
    rw [selectedReachable_reading, elimPairsAt_e1] <;>
    simp [feedsB, selectedEdge_reading_hA, selectedEdge_reading_hAB,
      minorEdge, majorEdge]

theorem coherentAtB_root (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) root = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_root]

theorem coherentB_reading (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentB (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) =
      decide (minorChoice.val = majorChoice.val) := by
  unfold coherentB
  change [hA, hAB, e0, e1, root].all _ = _
  rw [List.all_cons, coherentAtB_hA, List.all_cons, coherentAtB_hAB,
    List.all_cons, coherentAtB_e0, List.all_cons, coherentAtB_e1,
    List.all_cons, coherentAtB_root]
  fin_cases minorChoice <;> fin_cases majorChoice <;> simp

theorem coherent_iff_choices_agree (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentB (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) = true ↔
      minorChoice = majorChoice := by
  rw [coherentB_reading]
  simp [Fin.ext_iff]

theorem fixed_selector_domain_when_label_changes (width : Nat) :
    fanArities (control width) = fanArities (defective width) := by
  rw [fanArities_graph, fanArities_graph]

theorem coherence_preserved_when_label_changes (width : Nat)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    coherentB (control width)
        (reading width false minorChoice majorChoice outputChoice) =
      coherentB (defective width)
        (reading width true minorChoice majorChoice outputChoice) := by
  rw [coherentB_reading, coherentB_reading]

theorem selectedRuleActiveB_e0 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) e0 =
      decide (minorChoice.val = 0 ∧ majorChoice.val = 0) := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    simp [selectedRuleActiveB, coherentAtB_e0, selectedReachable_reading]

theorem selectedRuleActiveB_e1 (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
        (reading width bad minorChoice majorChoice outputChoice) e1 =
      decide (minorChoice.val = 1 ∧ majorChoice.val = 1) := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    simp [selectedRuleActiveB, coherentAtB_e1, selectedReachable_reading]

theorem selectedRuleActiveB_hA (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hA = true := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    simp [selectedRuleActiveB, coherentAtB_hA, selectedReachable_reading]

theorem selectedRuleActiveB_hAB (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hAB = true := by
  fin_cases minorChoice <;> fin_cases majorChoice <;>
    simp [selectedRuleActiveB, coherentAtB_hAB, selectedReachable_reading]

theorem e0_rule_shape (width : Nat) (bad : Bool) :
    ruleShapeOKB e0 (graph width bad) = true := by
  unfold ruleShapeOKB classifyRule?
  rw [incoming_e0]
  simp [premiseEdges, minorEdge, majorEdge, e0, hA, hAB, A, B, AB]

theorem e1_rule_shape (width : Nat) (bad : Bool) :
    ruleShapeOKB e1 (graph width bad) = true := by
  unfold ruleShapeOKB classifyRule?
  rw [incoming_e1]
  simp [premiseEdges, minorEdge, majorEdge, e1, hA, hAB, A, B, AB]

theorem e0_outDep (width : Nat) (bad : Bool) :
    outDep e0 (graph width bad) = goodDep := by
  unfold outDep classifyRule?
  rw [incoming_e0]
  simp [premiseEdges, minorEdge, majorEdge, goodDep, e0, hA, hAB, A, B, AB]

theorem e1_outDep (width : Nat) (bad : Bool) :
    outDep e1 (graph width bad) = goodDep := by
  unfold outDep classifyRule?
  rw [incoming_e1]
  simp [premiseEdges, minorEdge, majorEdge, goodDep, e1, hA, hAB, A, B, AB]

theorem hA_rule (width : Nat) (bad : Bool) :
    ruleShapeOKB hA (graph width bad) = true ∧
      outDep hA (graph width bad) = [A] := by
  simp [ruleShapeOKB, outDep, classifyRule?, hA]

theorem hAB_rule (width : Nat) (bad : Bool) :
    ruleShapeOKB hAB (graph width bad) = true ∧
      outDep hAB (graph width bad) = [AB] := by
  simp [ruleShapeOKB, outDep, classifyRule?, hAB]

theorem hA_outputs_covered (width : Nat) (bad : Bool) :
    ∀ edge ∈ outgoing (graph width bad) hA,
      ExplicitOutputCovered (graph width bad) hA edge := by
  intro edge hedge
  rw [outgoing_hA] at hedge
  simp at hedge
  rcases hedge with rfl | rfl
  · let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
    let r := reading width bad (0 : Fin 2) (0 : Fin 2) outputChoice
    refine ⟨r, ?_, ?_, ?_⟩
    · simp [r, coherentB_reading]
    · exact selectedRuleActiveB_hA width bad 0 0 outputChoice
    · simpa [r] using selectedEdge_reading_hA width bad 0 0 outputChoice
  · let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
    let r := reading width bad (1 : Fin 2) (1 : Fin 2) outputChoice
    refine ⟨r, ?_, ?_, ?_⟩
    · simp [r, coherentB_reading]
    · exact selectedRuleActiveB_hA width bad 1 1 outputChoice
    · simpa [r] using selectedEdge_reading_hA width bad 1 1 outputChoice

theorem hAB_outputs_covered (width : Nat) (bad : Bool) :
    ∀ edge ∈ outgoing (graph width bad) hAB,
      ExplicitOutputCovered (graph width bad) hAB edge := by
  intro edge hedge
  rw [outgoing_hAB] at hedge
  simp at hedge
  rcases hedge with rfl | rfl
  · let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
    let r := reading width bad (0 : Fin 2) (0 : Fin 2) outputChoice
    refine ⟨r, ?_, ?_, ?_⟩
    · simp [r, coherentB_reading]
    · exact selectedRuleActiveB_hAB width bad 0 0 outputChoice
    · simpa [r] using selectedEdge_reading_hAB width bad 0 0 outputChoice
  · let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
    let r := reading width bad (1 : Fin 2) (1 : Fin 2) outputChoice
    refine ⟨r, ?_, ?_, ?_⟩
    · simp [r, coherentB_reading]
    · exact selectedRuleActiveB_hAB width bad 1 1 outputChoice
    · simpa [r] using selectedEdge_reading_hAB width bad 1 1 outputChoice

theorem e0_outputs_covered (width : Nat) (bad : Bool) :
    ∀ edge ∈ outgoing (graph width bad) e0,
      ExplicitOutputCovered (graph width bad) e0 edge := by
  intro edge hedge
  rw [outgoing_e0] at hedge
  obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hedge
  have hi' : i < width + 2 := by simpa [e0Outputs] using hi
  let outputChoice : Fin (width + 2) := ⟨i, hi'⟩
  let r := reading width bad (0 : Fin 2) (0 : Fin 2) outputChoice
  refine ⟨r, ?_, ?_, ?_⟩
  · simp [r, coherentB_reading]
  · simpa [r] using selectedRuleActiveB_e0 width bad 0 0 outputChoice
  · simpa [r, outputChoice] using selectedEdge_reading_e0 width bad 0 0 outputChoice

theorem e1_output_covered (width : Nat) (bad : Bool) :
    ∀ edge ∈ outgoing (graph width bad) e1,
      ExplicitOutputCovered (graph width bad) e1 edge := by
  intro edge hedge
  rw [outgoing_e1] at hedge
  simp only [List.mem_singleton] at hedge
  subst edge
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let r := reading width bad (1 : Fin 2) (1 : Fin 2) outputChoice
  refine ⟨r, ?_, ?_, ?_⟩
  · simp [r, coherentB_reading]
  · simpa [r] using selectedRuleActiveB_e1 width bad 1 1 outputChoice
  · simpa [r] using selectedEdge_reading_e1 width bad 1 1 outputChoice

theorem allExplicitOutputsCovered_graph (width : Nat) (bad : Bool) :
    AllExplicitOutputsCovered (graph width bad) := by
  intro v hv edge hedge
  simp [graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl
  · exact hA_outputs_covered width bad edge hedge
  · exact hAB_outputs_covered width bad edge hedge
  · exact e0_outputs_covered width bad edge hedge
  · exact e1_output_covered width bad edge hedge
  · rw [outgoing_root] at hedge
    simp at hedge

theorem control_explicitOutputLabelsCorrect (width : Nat) :
    ExplicitOutputLabelsCorrect (control width) := by
  intro v hv edge hedge
  simp [control, graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl
  · rw [outgoing_hA] at hedge
    simp at hedge
    rcases hedge with rfl | rfl <;> simpa [minorEdge, hA_rule]
  · rw [outgoing_hAB] at hedge
    simp at hedge
    rcases hedge with rfl | rfl <;> simpa [majorEdge, hAB_rule]
  · rw [outgoing_e0] at hedge
    obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
    constructor
    · exact e0_rule_shape width false
    · rw [e0_outDep]
      simp [e0Output]
  · rw [outgoing_e1] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    constructor
    · exact e1_rule_shape width false
    · rw [e1_outDep]
      simp [e1Output]
  · rw [outgoing_root] at hedge
    simp at hedge

theorem control_no_labelDefect (width : Nat) :
    ∀ r : ReadingData (fanArities (control width)),
      labelDefectB (control width) r = false :=
  (explicitOutputLabelsCorrect_iff_no_labelDefect _
    (allExplicitOutputsCovered_graph width false)).mp
      (control_explicitOutputLabelsCorrect width)

theorem bad_dependency_ne_goodDep : [Z] ≠ goodDep := by
  intro h
  have hmem : A ∈ goodDep := by simp [goodDep]
  rw [← h] at hmem
  simpa [A, Z] using hmem

theorem selectedLabelErrorAtB_hA_false (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hA = false := by
  fin_cases minorChoice <;>
    simp [selectedLabelErrorAtB, selectedRuleActiveB_hA,
      selectedEdge_reading_hA, minorEdge, hA_rule]

theorem selectedLabelErrorAtB_hAB_false (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) hAB = false := by
  fin_cases majorChoice <;>
    simp [selectedLabelErrorAtB, selectedRuleActiveB_hAB,
      selectedEdge_reading_hAB, majorEdge, hAB_rule]

theorem selectedLabelErrorAtB_e0_defective (width : Nat)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (defective width)
        (reading width true minorChoice majorChoice outputChoice) e0 =
      decide (minorChoice.val = 0 ∧ majorChoice.val = 0 ∧
        outputChoice.val = 1) := by
  by_cases houtput : outputChoice.val = 1
  · fin_cases minorChoice <;> fin_cases majorChoice <;>
      simp [selectedLabelErrorAtB, selectedRuleActiveB_e0,
        selectedEdge_reading_e0, e0_rule_shape, e0_outDep, e0Output,
        houtput, bad_dependency_ne_goodDep]
  · fin_cases minorChoice <;> fin_cases majorChoice <;>
      simp [selectedLabelErrorAtB, selectedRuleActiveB_e0,
        selectedEdge_reading_e0, e0_rule_shape, e0_outDep, e0Output,
        houtput]

theorem selectedLabelErrorAtB_e1_false (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) e1 = false := by
  simp [selectedLabelErrorAtB, selectedEdge_reading_e1, e1_rule_shape,
    e1_outDep, e1Output]

theorem selectedLabelErrorAtB_root_false (width : Nat) (bad : Bool)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad minorChoice majorChoice outputChoice) root = false := by
  simp [selectedLabelErrorAtB, selectedEdge_reading_root]

theorem selectedLabelHadErrorB_defective (width : Nat)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    selectedLabelHadErrorB (defective width)
        (reading width true minorChoice majorChoice outputChoice) =
      decide (minorChoice.val = 0 ∧ majorChoice.val = 0 ∧
        outputChoice.val = 1) := by
  unfold selectedLabelHadErrorB
  change [hA, hAB, e0, e1, root].any _ = _
  simp [selectedLabelErrorAtB_hA_false, selectedLabelErrorAtB_hAB_false,
    selectedLabelErrorAtB_e0_defective, selectedLabelErrorAtB_e1_false,
    selectedLabelErrorAtB_root_false]

/-- Exact characterization of every marked reading in the defective graph. -/
theorem labelDefectB_defective_iff (width : Nat)
    (minorChoice majorChoice : Fin 2) (outputChoice : Fin (width + 2)) :
    labelDefectB (defective width)
        (reading width true minorChoice majorChoice outputChoice) = true ↔
      minorChoice.val = 0 ∧ majorChoice.val = 0 ∧ outputChoice.val = 1 := by
  rw [labelDefectB_eq_true_iff, coherentB_reading,
    selectedLabelHadErrorB_defective]
  simp only [decide_eq_true_eq]
  constructor
  · exact fun h => h.2
  · intro h
    exact ⟨by omega, h⟩

def badReference (width : Nat) :
    ReadingData (fanArities (defective width)) :=
  reading width true ⟨0, by omega⟩ ⟨0, by omega⟩ ⟨1, by omega⟩

def changedMajor (width : Nat) :
    ReadingData (fanArities (defective width)) :=
  reading width true ⟨0, by omega⟩ ⟨1, by omega⟩ ⟨1, by omega⟩

theorem badReference_indices (width : Nat) :
    (badReference width).indices = [0, 0, 1] := by
  simp [badReference, reading_indices]

theorem changedMajor_indices (width : Nat) :
    (changedMajor width).indices = [0, 1, 1] := by
  simp [changedMajor, reading_indices]

theorem badReference_marks (width : Nat) :
    labelDefectB (defective width) (badReference width) = true := by
  unfold badReference
  exact (labelDefectB_defective_iff width _ _ _).mpr (by simp)

/-- Fixing the local output selector alone does not preserve the marker: the
major-premise routing change makes the second reading incoherent. -/
theorem local_output_alone_insufficient (width : Nat) :
    (badReference width).indices.getD 2 0 =
        (changedMajor width).indices.getD 2 0 ∧
      labelDefectB (defective width) (badReference width) = true ∧
      coherentB (defective width) (changedMajor width) = false ∧
      labelDefectB (defective width) (changedMajor width) = false := by
  refine ⟨?_, badReference_marks width, ?_, ?_⟩
  · simp [badReference_indices, changedMajor_indices]
  · unfold changedMajor
    simp [coherentB_reading]
  · unfold changedMajor
    rw [Bool.eq_false_iff]
    intro hmark
    have hchoices := (labelDefectB_defective_iff width _ _ _).mp hmark
    simp at hchoices

theorem changedMajor_rejected (width : Nat) :
    rejectsReadingLabelsB (defective width) (changedMajor width) = true := by
  unfold changedMajor
  simp [rejectsReadingLabelsB, coherentB_reading]

def sufficientSupport : Finset Nat := {0, 1, 2}

def AgreesOnSupport {d : Graph} (support : Finset Nat)
    (x y : ReadingData (fanArities d)) : Prop :=
  ∀ i ∈ support, x.indices.getD i 0 = y.indices.getD i 0

theorem sufficientSupport_card : sufficientSupport.card = 3 := by
  decide

/-- Agreement on the two premise-routing coordinates and the output coordinate
derives coherence, activation and selection; none is assumed separately. -/
theorem sufficientSupport_forces_labelDefect (width : Nat)
    (y : ReadingData (fanArities (defective width)))
    (hagrees : AgreesOnSupport sufficientSupport (badReference width) y) :
    labelDefectB (defective width) y = true := by
  obtain ⟨minorChoice, majorChoice, outputChoice, rfl⟩ :=
    exists_reading_choices width true y
  have hminor := hagrees 0 (by simp [sufficientSupport])
  have hmajor := hagrees 1 (by simp [sufficientSupport])
  have houtput := hagrees 2 (by simp [sufficientSupport])
  rw [labelDefectB_defective_iff]
  simp [badReference_indices, reading_indices] at hminor hmajor houtput
  exact ⟨hminor.symm, hmajor.symm, houtput.symm⟩

theorem sufficientSupport_forces_rejection (width : Nat)
    (y : ReadingData (fanArities (defective width)))
    (hagrees : AgreesOnSupport sufficientSupport (badReference width) y) :
    rejectsReadingLabelsB (defective width) y = true :=
  labelDefectB_implies_rejectsReadingLabelsB _ _
    (sufficientSupport_forces_labelDefect width y hagrees)

theorem control_has_coherent_reading (width : Nat) :
    ∃ r : ReadingData (fanArities (control width)),
      coherentB (control width) r = true := by
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  exact ⟨reading width false 0 0 outputChoice, by simp [coherentB_reading]⟩

theorem defective_has_coherent_marked_reading (width : Nat) :
    ∃ r : ReadingData (fanArities (defective width)),
      coherentB (defective width) r = true ∧
        labelDefectB (defective width) r = true := by
  refine ⟨badReference width, ?_, badReference_marks width⟩
  unfold badReference
  simp [coherentB_reading]

theorem readings_complete (width : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width bad))) :
    r ∈ readings (graph width bad) :=
  mem_enumerateReadingData r

theorem readings_nodup (width : Nat) (bad : Bool) :
    (readings (graph width bad)).Nodup :=
  enumerateReadingData_nodup _

theorem raw_reading_count (width : Nat) (bad : Bool) :
    (readings (graph width bad)).length = 4 * (width + 2) := by
  rw [readings, enumerateReadingData_length, fanArities_graph]
  simp
  omega

def coherentCylinder (width : Nat) (bad : Bool) :
    List (ReadingData (fanArities (graph width bad))) :=
  (List.finRange 2).flatMap fun premiseChoice =>
    (List.finRange (width + 2)).map fun outputChoice =>
      reading width bad premiseChoice premiseChoice outputChoice

theorem coherentCylinder_count (width : Nat) (bad : Bool) :
    (coherentCylinder width bad).length = 2 * (width + 2) := by
  simp [coherentCylinder]
  omega

theorem coherentCylinder_nodup (width : Nat) (bad : Bool) :
    (coherentCylinder width bad).Nodup := by
  unfold coherentCylinder
  apply List.nodup_flatMap.mpr
  constructor
  · intro premiseChoice _hpremise
    exact (List.nodup_finRange (width + 2)).map (by
      intro left right hreading
      have hindices := congrArg Readings.Selection.ReadingData.indices hreading
      simp [reading_indices] at hindices
      exact Fin.ext hindices)
  · exact (List.nodup_finRange 2).imp (by
      intro left right hne candidate hleft hright
      obtain ⟨leftOutput, _hleftOutput, rfl⟩ := List.mem_map.mp hleft
      obtain ⟨rightOutput, _hrightOutput, hreading⟩ := List.mem_map.mp hright
      have hindices := congrArg Readings.Selection.ReadingData.indices hreading
      simp [reading_indices] at hindices
      exact hne (Fin.ext hindices.1.symm))

theorem mem_coherentCylinder_iff (width : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width bad))) :
    r ∈ coherentCylinder width bad ↔
      r ∈ coherentReadings (graph width bad) := by
  constructor
  · intro hr
    obtain ⟨premiseChoice, _hpremise, hinner⟩ := List.mem_flatMap.mp hr
    obtain ⟨outputChoice, _houtput, rfl⟩ := List.mem_map.mp hinner
    apply List.mem_filter.mpr
    exact ⟨readings_complete width bad _, by simp [coherentB_reading]⟩
  · intro hr
    have hparts := List.mem_filter.mp hr
    obtain ⟨minorChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width bad r
    have hagree :=
      (coherent_iff_choices_agree width bad minorChoice majorChoice outputChoice).mp
        hparts.2
    subst majorChoice
    apply List.mem_flatMap.mpr
    refine ⟨minorChoice, by simp, ?_⟩
    exact List.mem_map.mpr ⟨outputChoice, by simp, rfl⟩

theorem coherentCylinder_perm_coherentReadings (width : Nat) (bad : Bool) :
    List.Perm (coherentCylinder width bad)
      (coherentReadings (graph width bad)) := by
  apply (List.perm_ext_iff_of_nodup
    (coherentCylinder_nodup width bad)
    ((readings_nodup width bad).filter _)).mpr
  intro r
  exact mem_coherentCylinder_iff width bad r

theorem coherent_reading_count (width : Nat) (bad : Bool) :
    (coherentReadings (graph width bad)).length = 2 * (width + 2) := by
  calc
    (coherentReadings (graph width bad)).length =
        (coherentCylinder width bad).length :=
      (coherentCylinder_perm_coherentReadings width bad).length_eq.symm
    _ = 2 * (width + 2) := coherentCylinder_count width bad

def markedReadings (d : Graph) :
    List (ReadingData (fanArities d)) :=
  (readings d).filter (labelDefectB d)

theorem control_mark_count (width : Nat) :
    (markedReadings (control width)).length = 0 := by
  have hempty : markedReadings (control width) = [] := by
    apply List.filter_eq_nil_iff.mpr
    intro r _hr
    simp [control_no_labelDefect width r]
  rw [hempty]
  rfl

theorem defective_marked_perm_singleton (width : Nat) :
    List.Perm (markedReadings (defective width)) [badReference width] := by
  apply (List.perm_ext_iff_of_nodup
    ((readings_nodup width true).filter _)
    (by simp)).mpr
  intro r
  constructor
  · intro hr
    have hparts := List.mem_filter.mp hr
    obtain ⟨minorChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width true r
    have hvalues :=
      (labelDefectB_defective_iff width minorChoice majorChoice outputChoice).mp
        hparts.2
    have hminor : minorChoice = ⟨0, by omega⟩ := Fin.ext hvalues.1
    have hmajor : majorChoice = ⟨0, by omega⟩ := Fin.ext hvalues.2.1
    have houtput : outputChoice = ⟨1, by omega⟩ := Fin.ext hvalues.2.2
    simp [hminor, hmajor, houtput, badReference]
  · intro hr
    simp only [List.mem_singleton] at hr
    subst r
    exact List.mem_filter.mpr
      ⟨readings_complete width true _, badReference_marks width⟩

theorem defective_mark_count (width : Nat) :
    (markedReadings (defective width)).length = 1 := by
  simpa using (defective_marked_perm_singleton width).length_eq

/-- Raw cylinder obtained by fixing only coordinate 2 to the bad output. -/
def localOutputCylinder (width : Nat) :
    List (ReadingData (fanArities (defective width))) :=
  (List.finRange 2).flatMap fun minorChoice =>
    (List.finRange 2).map fun majorChoice =>
      reading width true minorChoice majorChoice ⟨1, by omega⟩

theorem localOutputCylinder_count (width : Nat) :
    (localOutputCylinder width).length = 4 := by
  simp [localOutputCylinder]

theorem localOutputCylinder_nodup (width : Nat) :
    (localOutputCylinder width).Nodup := by
  unfold localOutputCylinder
  apply List.nodup_flatMap.mpr
  constructor
  · intro minorChoice _hminor
    exact (List.nodup_finRange 2).map (by
      intro left right hreading
      have hindices := congrArg Readings.Selection.ReadingData.indices hreading
      simp [reading_indices] at hindices
      exact Fin.ext hindices)
  · exact (List.nodup_finRange 2).imp (by
      intro left right hne candidate hleft hright
      obtain ⟨leftMajor, _hleftMajor, rfl⟩ := List.mem_map.mp hleft
      obtain ⟨rightMajor, _hrightMajor, hreading⟩ := List.mem_map.mp hright
      have hindices := congrArg Readings.Selection.ReadingData.indices hreading
      simp [reading_indices] at hindices
      exact hne (Fin.ext hindices.1.symm))

theorem mem_localOutputCylinder_iff (width : Nat)
    (r : ReadingData (fanArities (defective width))) :
    r ∈ localOutputCylinder width ↔ r.indices.getD 2 0 = 1 := by
  constructor
  · intro hr
    obtain ⟨minorChoice, _hminor, hinner⟩ := List.mem_flatMap.mp hr
    obtain ⟨majorChoice, _hmajor, rfl⟩ := List.mem_map.mp hinner
    simp [reading_indices]
  · intro hcoordinate
    obtain ⟨minorChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width true r
    have houtput : outputChoice = ⟨1, by omega⟩ := by
      apply Fin.ext
      simpa [reading_indices] using hcoordinate
    rw [houtput]
    apply List.mem_flatMap.mpr
    exact ⟨minorChoice, by simp,
      List.mem_map.mpr ⟨majorChoice, by simp, rfl⟩⟩

def sufficientSupportCylinder (width : Nat) :
    List (ReadingData (fanArities (defective width))) := [badReference width]

theorem sufficientSupportCylinder_count (width : Nat) :
    (sufficientSupportCylinder width).length = 1 := by
  rfl

theorem sufficientSupportCylinder_nodup (width : Nat) :
    (sufficientSupportCylinder width).Nodup := by
  simp [sufficientSupportCylinder]

theorem mem_sufficientSupportCylinder_iff (width : Nat)
    (r : ReadingData (fanArities (defective width))) :
    r ∈ sufficientSupportCylinder width ↔
      AgreesOnSupport sufficientSupport (badReference width) r := by
  constructor
  · intro hr
    simp [sufficientSupportCylinder] at hr
    subst r
    intro i _hi
    rfl
  · intro hagrees
    obtain ⟨minorChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width true r
    have hminor := hagrees 0 (by simp [sufficientSupport])
    have hmajor := hagrees 1 (by simp [sufficientSupport])
    have houtput := hagrees 2 (by simp [sufficientSupport])
    simp [badReference_indices, reading_indices] at hminor hmajor houtput
    have hm : minorChoice = ⟨0, by omega⟩ := Fin.ext hminor.symm
    have hM : majorChoice = ⟨0, by omega⟩ := Fin.ext hmajor.symm
    have ho : outputChoice = ⟨1, by omega⟩ := Fin.ext houtput.symm
    simp [sufficientSupportCylinder, badReference, hm, hM, ho]

theorem localOutputCylinder_subset_readings (width : Nat) :
    ∀ r ∈ localOutputCylinder width, r ∈ readings (defective width) := by
  intro r _hr
  exact readings_complete width true r

theorem sufficientSupportCylinder_subset_readings (width : Nat) :
    ∀ r ∈ sufficientSupportCylinder width, r ∈ readings (defective width) := by
  intro r _hr
  exact readings_complete width true r

theorem sufficientSupportCylinder_all_marked (width : Nat) :
    ∀ r ∈ sufficientSupportCylinder width,
      labelDefectB (defective width) r = true := by
  intro r hr
  simp [sufficientSupportCylinder] at hr
  subst r
  exact badReference_marks width

theorem localOutputCylinder_raw_density (width : Nat) :
    ((localOutputCylinder width).length : ℚ) /
        ((readings (defective width)).length : ℚ) =
      1 / (width + 2 : ℚ) := by
  rw [localOutputCylinder_count, raw_reading_count]
  push_cast
  have hw : (width : ℚ) + 2 ≠ 0 := by positivity
  field_simp

theorem sufficientSupportCylinder_raw_density (width : Nat) :
    ((sufficientSupportCylinder width).length : ℚ) /
        ((readings (defective width)).length : ℚ) =
      1 / (4 * (width + 2 : ℚ)) := by
  rw [sufficientSupportCylinder_count, raw_reading_count]
  push_cast
  ring

/-- This equality is about all marks, not merely a lower-bound cylinder: the
exact marker characterization proves there is exactly one marked reading. -/
theorem defective_total_raw_density (width : Nat) :
    ((markedReadings (defective width)).length : ℚ) /
        ((readings (defective width)).length : ℚ) =
      1 / (4 * (width + 2 : ℚ)) := by
  rw [defective_mark_count, raw_reading_count]
  push_cast
  ring

theorem defective_conditioned_coherent_density (width : Nat) :
    ((markedReadings (defective width)).length : ℚ) /
        ((coherentReadings (defective width)).length : ℚ) =
      1 / (2 * (width + 2 : ℚ)) := by
  rw [defective_mark_count, coherent_reading_count]
  push_cast
  ring


end InteractingPremises
end Readings.Explicit
end DLDS
