import Readings.Explicit.PremiseInteraction

open scoped Classical

set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unnecessarySimpa false

/-!
# A two-stage explicit-reading composition

The producer eliminates `P` and `P → A` at `derivedA`.  Unlike the isolated
consumer fixture, the minor premise `A` is not a source: the selected output of
`derivedA` either feeds `e0`, feeds `e1`, or bypasses the consumer.  The other
consumer premise `A → B` remains a source.  The final fan at `e0` retains the
variable width used by the preceding fixtures.
-/

namespace DLDS
namespace Readings.Explicit
namespace CoveredComposition

open Flow
open Readings.Selection
open DLDS.Readings.LocalLabels
open DLDS.Readings.Explicit.UniformElimFanFamily

def P : Formula := #"composition_P"
def A : Formula := #"composition_A"
def B : Formula := #"composition_B"
def R : Formula := #"composition_R"
def Z : Formula := #"composition_bad"
def PA : Formula := P >> A
def AB : Formula := A >> B

def hP : Vertex := node 0 5 P true false []
def hPA : Vertex := node 1 5 PA true false []
def derivedA : Vertex := node 2 4 A false false []
def hAB : Vertex := node 3 4 AB true false []
def e0 : Vertex := node 4 3 B false false []
def e1 : Vertex := node 5 3 B false false []
def root : Vertex := node 6 2 R false false []

@[simp] private theorem vertex_beq_false_of_number_ne {u v : Vertex}
    (h : u.NUMBER ≠ v.NUMBER) : (u == v) = false :=
  beq_eq_false_iff_ne.mpr (by
    intro huv
    exact h (congrArg Vertex.NUMBER huv))

@[simp] private theorem hPA_beq_hP : (hPA == hP) = false :=
  vertex_beq_false_of_number_ne (by simp [hPA, hP])
@[simp] private theorem hAB_beq_hP : (hAB == hP) = false :=
  vertex_beq_false_of_number_ne (by simp [hAB, hP])
@[simp] private theorem hAB_beq_hPA : (hAB == hPA) = false :=
  vertex_beq_false_of_number_ne (by simp [hAB, hPA])
@[simp] private theorem derivedA_beq_hP : (derivedA == hP) = false :=
  vertex_beq_false_of_number_ne (by simp [derivedA, hP])
@[simp] private theorem derivedA_beq_hPA : (derivedA == hPA) = false :=
  vertex_beq_false_of_number_ne (by simp [derivedA, hPA])
@[simp] private theorem derivedA_beq_hAB : (derivedA == hAB) = false :=
  vertex_beq_false_of_number_ne (by simp [derivedA, hAB])
@[simp] private theorem hAB_beq_derivedA : (hAB == derivedA) = false :=
  vertex_beq_false_of_number_ne (by simp [hAB, derivedA])
@[simp] private theorem e0_beq_hP : (e0 == hP) = false :=
  vertex_beq_false_of_number_ne (by simp [e0, hP])
@[simp] private theorem e0_beq_hPA : (e0 == hPA) = false :=
  vertex_beq_false_of_number_ne (by simp [e0, hPA])
@[simp] private theorem e0_beq_hAB : (e0 == hAB) = false :=
  vertex_beq_false_of_number_ne (by simp [e0, hAB])
@[simp] private theorem e0_beq_derivedA : (e0 == derivedA) = false :=
  vertex_beq_false_of_number_ne (by simp [e0, derivedA])
@[simp] private theorem e1_beq_hP : (e1 == hP) = false :=
  vertex_beq_false_of_number_ne (by simp [e1, hP])
@[simp] private theorem e1_beq_hPA : (e1 == hPA) = false :=
  vertex_beq_false_of_number_ne (by simp [e1, hPA])
@[simp] private theorem e1_beq_hAB : (e1 == hAB) = false :=
  vertex_beq_false_of_number_ne (by simp [e1, hAB])
@[simp] private theorem e1_beq_derivedA : (e1 == derivedA) = false :=
  vertex_beq_false_of_number_ne (by simp [e1, derivedA])
@[simp] private theorem e1_beq_e0 : (e1 == e0) = false :=
  vertex_beq_false_of_number_ne (by simp [e1, e0])
@[simp] private theorem e0_beq_e1 : (e0 == e1) = false :=
  vertex_beq_false_of_number_ne (by simp [e0, e1])
@[simp] private theorem root_beq_hP : (root == hP) = false :=
  vertex_beq_false_of_number_ne (by simp [root, hP])
@[simp] private theorem root_beq_hPA : (root == hPA) = false :=
  vertex_beq_false_of_number_ne (by simp [root, hPA])
@[simp] private theorem root_beq_hAB : (root == hAB) = false :=
  vertex_beq_false_of_number_ne (by simp [root, hAB])
@[simp] private theorem root_beq_derivedA : (root == derivedA) = false :=
  vertex_beq_false_of_number_ne (by simp [root, derivedA])
@[simp] private theorem root_beq_e0 : (root == e0) = false :=
  vertex_beq_false_of_number_ne (by simp [root, e0])
@[simp] private theorem root_beq_e1 : (root == e1) = false :=
  vertex_beq_false_of_number_ne (by simp [root, e1])

def producerMinor : Deduction := edge hP derivedA 0 [P]
def producerMajor : Deduction := edge hPA derivedA 0 [PA]
def producerDep : Dep := normalizeDep [P, PA]

/-- Choices 0 and 1 deliver the derived premise to the corresponding consumer
application.  Choice 2 is the producer's bypass output. -/
def linkEdge (choice : Nat) : Deduction :=
  edge derivedA (if choice = 0 then e0 else if choice = 1 then e1 else root)
    choice producerDep

def linkEdges : List Deduction := [linkEdge 0, linkEdge 1, linkEdge 2]

def consumerMajorEdge (choice : Nat) : Deduction :=
  edge hAB (if choice = 0 then e0 else if choice = 1 then e1 else root)
    choice [AB]

def consumerMajorEdges : List Deduction :=
  [consumerMajorEdge 0, consumerMajorEdge 1, consumerMajorEdge 2]

def finalDep : Dep := normalizeDep (producerDep ++ [AB])

def e0Output (bad : Bool) (choice : Nat) : Deduction :=
  edge e0 root choice (if bad && choice == 1 then [Z] else finalDep)

def e0Outputs (width : Nat) (bad : Bool) : List Deduction :=
  (List.range (width + 2)).map (e0Output bad)

def e1Output : Deduction := edge e1 root 0 finalDep

def graph (width : Nat) (bad : Bool) : Graph :=
  dlds [hP, hPA, derivedA, hAB, e0, e1, root]
    ([producerMinor, producerMajor] ++ linkEdges ++ consumerMajorEdges ++
      e0Outputs width bad ++ [e1Output]) []

abbrev control (width : Nat) : Graph := graph width false
abbrev defective (width : Nat) : Graph := graph width true

theorem fixed_structure_when_label_changes (width : Nat) :
    (control width).NODES = (defective width).NODES ∧
    ((control width).EDGES.map fun e => (e.START, e.END, e.COLOUR)) =
      ((defective width).EDGES.map fun e => (e.START, e.END, e.COLOUR)) ∧
    (control width).PATHS = (defective width).PATHS := by
  simp [control, defective, graph, e0Outputs, e0Output]

theorem graph_node_count (width : Nat) (bad : Bool) :
    (graph width bad).NODES.length = 7 := rfl

theorem graph_edge_count (width : Nat) (bad : Bool) :
    (graph width bad).EDGES.length = width + 11 := by
  simp [graph, linkEdges, consumerMajorEdges, e0Outputs]

theorem every_edge_strictly_descends (width : Nat) (bad : Bool)
    (selected : Deduction) (hselected : selected ∈ (graph width bad).EDGES) :
    selected.END.LEVEL < selected.START.LEVEL := by
  simp only [graph, List.mem_append, List.mem_cons, List.mem_singleton] at hselected
  rcases hselected with ((((hproducer | hlink) | hmajor) | he0) | (he1 | hfalse))
  · rcases hproducer with rfl | rfl | hfalse
    · simp [producerMinor, hP, derivedA]
    · simp [producerMajor, hPA, derivedA]
    · simp at hfalse
  · simp [linkEdges] at hlink
    rcases hlink with rfl | rfl | rfl <;>
      simp [linkEdge, derivedA, e0, e1, root]
  · simp [consumerMajorEdges] at hmajor
    rcases hmajor with rfl | rfl | rfl <;>
      simp [consumerMajorEdge, hAB, e0, e1, root]
  · obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he0
    simp [e0Output, e0, root]
  · subst selected
    simp [e1Output, e1, root]
  · simp at hfalse

theorem every_node_reaches_root_within_two_edges (width : Nat) (bad : Bool)
    (v : Vertex) (hv : v ∈ (graph width bad).NODES) :
    v = root ∨
      (∃ e ∈ (graph width bad).EDGES, e.START = v ∧ e.END = root) ∨
      ∃ middle e₁ e₂,
        e₁ ∈ (graph width bad).EDGES ∧ e₁.START = v ∧ e₁.END = middle ∧
        e₂ ∈ (graph width bad).EDGES ∧ e₂.START = middle ∧ e₂.END = root := by
  simp [graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · right; right
    exact ⟨derivedA, producerMinor, linkEdge 2, by simp [graph, linkEdges],
      by simp [producerMinor], by simp [producerMinor],
      by simp [graph, linkEdges], by simp [linkEdge], by simp [linkEdge]⟩
  · right; right
    exact ⟨derivedA, producerMajor, linkEdge 2, by simp [graph, linkEdges],
      by simp [producerMajor], by simp [producerMajor],
      by simp [graph, linkEdges], by simp [linkEdge], by simp [linkEdge]⟩
  · right; left
    exact ⟨linkEdge 2, by simp [graph, linkEdges], by simp [linkEdge],
      by simp [linkEdge]⟩
  · right; left
    exact ⟨consumerMajorEdge 2, by simp [graph, consumerMajorEdges],
      by simp [consumerMajorEdge], by simp [consumerMajorEdge]⟩
  · right; left
    refine ⟨e0Output bad 0, ?_, by simp [e0Output], by simp [e0Output]⟩
    simp [graph, e0Outputs]
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨0, by omega, rfl⟩))))
  · right; left
    exact ⟨e1Output, by simp [graph], by simp [e1Output], by simp [e1Output]⟩
  · exact Or.inl rfl

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

theorem outgoing_hP (width : Nat) (bad : Bool) :
    outgoing (graph width bad) hP = [producerMinor] := by
  rw [show outgoing (graph width bad) hP =
    (graph width bad).EDGES.filter (fun e => decide (e.START = hP)) from
      outgoing_eq_filter hP _]
  simp [graph, linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    e0Outputs, e0Output, e1Output, producerMinor, producerMajor,
    hP, hPA, derivedA, hAB, e0, e1, root]

theorem outgoing_hPA (width : Nat) (bad : Bool) :
    outgoing (graph width bad) hPA = [producerMajor] := by
  rw [show outgoing (graph width bad) hPA =
    (graph width bad).EDGES.filter (fun e => decide (e.START = hPA)) from
      outgoing_eq_filter hPA _]
  simp [graph, linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    e0Outputs, e0Output, e1Output, producerMinor, producerMajor,
    hP, hPA, derivedA, hAB, e0, e1, root]

theorem outgoing_derivedA (width : Nat) (bad : Bool) :
    outgoing (graph width bad) derivedA =
      [linkEdge 0, linkEdge 1, linkEdge 2] := by
  rw [show outgoing (graph width bad) derivedA =
    (graph width bad).EDGES.filter (fun e => decide (e.START = derivedA)) from
      outgoing_eq_filter derivedA _]
  simp [graph, linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    e0Outputs, e0Output, e1Output, producerMinor, producerMajor,
    hP, hPA, derivedA, hAB, e0, e1, root]

theorem outgoing_hAB (width : Nat) (bad : Bool) :
    outgoing (graph width bad) hAB =
      [consumerMajorEdge 0, consumerMajorEdge 1, consumerMajorEdge 2] := by
  rw [show outgoing (graph width bad) hAB =
    (graph width bad).EDGES.filter (fun e => decide (e.START = hAB)) from
      outgoing_eq_filter hAB _]
  simp [graph, linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    e0Outputs, e0Output, e1Output, producerMinor, producerMajor,
    hP, hPA, derivedA, hAB, e0, e1, root]

theorem outgoing_e0 (width : Nat) (bad : Bool) :
    outgoing (graph width bad) e0 = e0Outputs width bad := by
  rw [show outgoing (graph width bad) e0 =
    (graph width bad).EDGES.filter (fun e => decide (e.START = e0)) from
      outgoing_eq_filter e0 _]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_start_e0]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1]

theorem outgoing_e1 (width : Nat) (bad : Bool) :
    outgoing (graph width bad) e1 = [e1Output] := by
  rw [show outgoing (graph width bad) e1 =
    (graph width bad).EDGES.filter (fun e => decide (e.START = e1)) from
      outgoing_eq_filter e1 _]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_start_ne width bad e1 (by simp [e0, e1])]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1]

theorem outgoing_root (width : Nat) (bad : Bool) :
    outgoing (graph width bad) root = [] := by
  rw [show outgoing (graph width bad) root =
    (graph width bad).EDGES.filter (fun e => decide (e.START = root)) from
      outgoing_eq_filter root _]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_start_ne width bad root (by simp [root, e0])]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem fanNodes_graph (width : Nat) (bad : Bool) :
    (graph width bad).NODES.filter
      (fun v => decide (2 ≤ (outgoing (graph width bad) v).length)) =
      [derivedA, hAB, e0] := by
  change [hP, hPA, derivedA, hAB, e0, e1, root].filter _ = _
  simp [outgoing_hP, outgoing_hPA, outgoing_derivedA, outgoing_hAB,
    outgoing_e0, outgoing_e1, outgoing_root, e0Outputs]

theorem branchPoints_graph (width : Nat) (bad : Bool) :
    branchPoints (graph width bad) = [derivedA, hAB, e0] := by
  unfold branchPoints
  rw [fanNodes_graph]
  let cmp : Vertex → Vertex → Bool :=
    fun u v => decide (u.NUMBER ≤ v.NUMBER)
  generalize hsortedList : ([derivedA, hAB, e0].mergeSort cmp) = sortedList
  have hperm : sortedList.Perm [derivedA, hAB, e0] := by
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
  have hlen : sortedList.length = 3 := by simpa using hperm.length_eq
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
                  have hx : x = derivedA ∨ x = hAB ∨ x = e0 := by
                    simpa using hperm.mem_iff.mp (by simp : x ∈ [x, y, z])
                  have hy : y = derivedA ∨ y = hAB ∨ y = e0 := by
                    simpa using hperm.mem_iff.mp (by simp : y ∈ [x, y, z])
                  have hz : z = derivedA ∨ z = hAB ∨ z = e0 := by
                    simpa using hperm.mem_iff.mp (by simp : z ∈ [x, y, z])
                  have hnodup : [x, y, z].Nodup := by
                    exact hperm.nodup_iff.mpr (by
                      simp [derivedA, hAB, e0])
                  rcases hx with rfl | rfl | rfl <;>
                    rcases hy with rfl | rfl | rfl <;>
                    rcases hz with rfl | rfl | rfl <;>
                    simp [cmp, derivedA, hAB, e0] at hnodup hsorted ⊢

theorem fanArities_graph (width : Nat) (bad : Bool) :
    fanArities (graph width bad) = [3, 3, width + 2] := by
  simp [fanArities, branchPoints_graph, outgoing_derivedA, outgoing_hAB,
    outgoing_e0, e0Outputs]

def reading (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    ReadingData (fanArities (graph width bad)) :=
  (fanArities_graph width bad).symm ▸
    .cons linkChoice (.cons majorChoice (.cons outputChoice .nil))

theorem reading_indices (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    (reading width bad linkChoice majorChoice outputChoice).indices =
      [linkChoice.val, majorChoice.val, outputChoice.val] := by
  simp [reading, Readings.Selection.ReadingData.indices]

theorem exists_reading_choices (width : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width bad))) :
    ∃ linkChoice : Fin 3, ∃ majorChoice : Fin 3,
      ∃ outputChoice : Fin (width + 2),
        r = reading width bad linkChoice majorChoice outputChoice := by
  let h := fanArities_graph width bad
  let r' : ReadingData [3, 3, width + 2] := h ▸ r
  have hdecomp : ∃ linkChoice : Fin 3, ∃ majorChoice : Fin 3,
      ∃ outputChoice : Fin (width + 2),
        r' = .cons linkChoice (.cons majorChoice (.cons outputChoice .nil)) := by
    cases r' with
    | cons linkChoice rest =>
        cases rest with
        | cons majorChoice rest' =>
            cases rest' with
            | cons outputChoice rest'' =>
                cases rest''
                exact ⟨linkChoice, majorChoice, outputChoice, rfl⟩
  obtain ⟨linkChoice, majorChoice, outputChoice, hr'⟩ := hdecomp
  refine ⟨linkChoice, majorChoice, outputChoice, ?_⟩
  calc
    r = h.symm ▸ r' :=
      (readingData_cast_symm_cast h r).symm
    _ = h.symm ▸
        (.cons linkChoice (.cons majorChoice (.cons outputChoice .nil))) := by
      rw [hr']
    _ = reading width bad linkChoice majorChoice outputChoice := by
      simp [reading, h]

private theorem filter_e0Outputs_end_ne (width : Nat) (bad : Bool)
    (v : Vertex) (hv : v ≠ root) :
    (e0Outputs width bad).filter (fun e => decide (e.END = v)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro edge hedge
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
  simpa [e0Output, Ne.symm hv]

theorem incoming_hP (width : Nat) (bad : Bool) :
    get_rule.incoming hP (graph width bad) = [] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_ne width bad hP (by simp [hP, root])]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem incoming_hPA (width : Nat) (bad : Bool) :
    get_rule.incoming hPA (graph width bad) = [] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_ne width bad hPA (by simp [hPA, root])]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem incoming_hAB (width : Nat) (bad : Bool) :
    get_rule.incoming hAB (graph width bad) = [] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_ne width bad hAB (by simp [hAB, root])]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem incoming_derivedA (width : Nat) (bad : Bool) :
    get_rule.incoming derivedA (graph width bad) =
      [producerMinor, producerMajor] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_ne width bad derivedA (by simp [derivedA, root])]
  simp [linkEdges, List.range, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem incoming_e0 (width : Nat) (bad : Bool) :
    get_rule.incoming e0 (graph width bad) =
      [linkEdge 0, consumerMajorEdge 0] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_ne width bad e0 (by simp [e0, root])]
  simp [linkEdges, List.range, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem incoming_e1 (width : Nat) (bad : Bool) :
    get_rule.incoming e1 (graph width bad) =
      [linkEdge 1, consumerMajorEdge 1] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_ne width bad e1 (by simp [e1, root])]
  simp [linkEdges, List.range, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem producer_premise_not_source (width : Nat) (bad : Bool) :
    get_rule.incoming derivedA (graph width bad) ≠ [] := by
  rw [incoming_derivedA]
  simp

private theorem filter_e0Outputs_end_root (width : Nat) (bad : Bool) :
    (e0Outputs width bad).filter (fun e => decide (e.END = root)) =
      e0Outputs width bad := by
  apply List.filter_eq_self.mpr
  intro edge hedge
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
  simp [e0Output]

theorem incoming_root (width : Nat) (bad : Bool) :
    get_rule.incoming root (graph width bad) =
      [linkEdge 2, consumerMajorEdge 2] ++ e0Outputs width bad ++ [e1Output] := by
  rw [incoming_eq_filter]
  simp only [graph, List.filter_append]
  rw [filter_e0Outputs_end_root]
  simp [linkEdges, linkEdge, consumerMajorEdges, consumerMajorEdge,
    producerMinor, producerMajor, e1Output, hP, hPA, derivedA, hAB, e0, e1,
    root]

theorem predsOf_hP (width : Nat) (bad : Bool) :
    predsOf (graph width bad) hP = [] := by
  rw [predsOf_eq, incoming_hP]
  rfl

theorem predsOf_hPA (width : Nat) (bad : Bool) :
    predsOf (graph width bad) hPA = [] := by
  rw [predsOf_eq, incoming_hPA]
  rfl

theorem predsOf_hAB (width : Nat) (bad : Bool) :
    predsOf (graph width bad) hAB = [] := by
  rw [predsOf_eq, incoming_hAB]
  rfl

theorem predsOf_derivedA (width : Nat) (bad : Bool) :
    predsOf (graph width bad) derivedA = [hP, hPA] := by
  rw [predsOf_eq, incoming_derivedA]
  change List.eraseDups [hP, hPA] = [hP, hPA]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

theorem predsOf_e0 (width : Nat) (bad : Bool) :
    predsOf (graph width bad) e0 = [derivedA, hAB] := by
  rw [predsOf_eq, incoming_e0]
  change List.eraseDups [derivedA, hAB] = [derivedA, hAB]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

theorem predsOf_e1 (width : Nat) (bad : Bool) :
    predsOf (graph width bad) e1 = [derivedA, hAB] := by
  rw [predsOf_eq, incoming_e1]
  change List.eraseDups [derivedA, hAB] = [derivedA, hAB]
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

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

/-- The substituted `A` vertex is absent from the seed list; it is reached
only through the producer's two deterministic premise edges. -/
theorem sourceNodes_graph (width : Nat) (bad : Bool) :
    sourceNodes (graph width bad) = [hP, hPA, hAB] := by
  change [hP, hPA, derivedA, hAB, e0, e1, root].filter _ = _
  simp [sourceNodes, predsOf_hP, predsOf_hPA, predsOf_derivedA, predsOf_hAB,
    predsOf_e0, predsOf_e1, predsOf_root_ne_nil]

theorem elimPairsAt_hP (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) hP = [] := by
  simp [elimPairsAt, predsOf_eq, incoming_hP]

theorem elimPairsAt_hPA (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) hPA = [] := by
  simp [elimPairsAt, predsOf_eq, incoming_hPA]

theorem elimPairsAt_hAB (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) hAB = [] := by
  simp [elimPairsAt, predsOf_eq, incoming_hAB]

theorem elimPairsAt_derivedA (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) derivedA = [(hPA, hP)] := by
  unfold elimPairsAt
  rw [predsOf_derivedA]
  simp [hP, hPA, derivedA, P, A, PA]

theorem elimPairsAt_e0 (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) e0 = [(hAB, derivedA)] := by
  unfold elimPairsAt
  rw [predsOf_e0]
  simp [derivedA, hAB, e0, A, B, AB]

theorem elimPairsAt_e1 (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) e1 = [(hAB, derivedA)] := by
  unfold elimPairsAt
  rw [predsOf_e1]
  simp [derivedA, hAB, e1, A, B, AB]

theorem formula_of_mem_predsOf_root (width : Nat) (bad : Bool)
    {u : Vertex} (hu : u ∈ predsOf (graph width bad) root) :
    u.FORMULA = A ∨ u.FORMULA = AB ∨ u.FORMULA = B := by
  rw [predsOf_eq, incoming_root] at hu
  simp only [List.mem_eraseDups, List.mem_map, List.mem_append,
    List.mem_cons, List.mem_singleton] at hu
  obtain ⟨edge, hedge, rfl⟩ := hu
  rcases hedge with ((rfl | rfl | hfalse) | hedge) | (rfl | hfalse)
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)
  · simp at hfalse
  · obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
    exact Or.inr (Or.inr rfl)
  · exact Or.inr (Or.inr rfl)
  · simp at hfalse

theorem elimPairsAt_root (width : Nat) (bad : Bool) :
    elimPairsAt (graph width bad) root = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro pair hpair
  unfold elimPairsAt at hpair
  obtain ⟨major, hmajor, hinner⟩ := List.mem_flatMap.mp hpair
  obtain ⟨minor, hminor, hselected⟩ := List.mem_filterMap.mp hinner
  split at hselected <;> rename_i hcondition
  · rcases formula_of_mem_predsOf_root width bad hmajor with hA | hAB | hB
    · rw [hA] at hcondition
      simp [A] at hcondition
    · rw [hAB] at hcondition
      simp [AB, root, B, R] at hcondition
    · rw [hB] at hcondition
      simp [B, R] at hcondition
  · simp at hselected

theorem branchIndex_derivedA (width : Nat) (bad : Bool) :
    branchIndex (graph width bad) derivedA = 0 := by
  simp [branchIndex, branchPoints_graph]

theorem branchIndex_hAB (width : Nat) (bad : Bool) :
    branchIndex (graph width bad) hAB = 1 := by
  simp [branchIndex, branchPoints_graph, derivedA, hAB]

theorem branchIndex_e0 (width : Nat) (bad : Bool) :
    branchIndex (graph width bad) e0 = 2 := by
  simp [branchIndex, branchPoints_graph, derivedA, hAB, e0]

theorem selectedEdge_reading_hP (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hP =
      some producerMinor := by
  simp [selectedEdge, outgoing_hP]

theorem selectedEdge_reading_hPA (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hPA =
      some producerMajor := by
  simp [selectedEdge, outgoing_hPA]

theorem selectedEdge_reading_derivedA (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) derivedA =
      some (linkEdge linkChoice.val) := by
  fin_cases linkChoice <;>
    simp [selectedEdge, outgoing_derivedA, reading_indices,
      branchIndex_derivedA]

theorem selectedEdge_reading_hAB (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hAB =
      some (consumerMajorEdge majorChoice.val) := by
  fin_cases majorChoice <;>
    simp [selectedEdge, outgoing_hAB, reading_indices, branchIndex_hAB]

theorem selectedEdge_reading_e0 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) e0 =
      some (e0Output bad outputChoice.val) := by
  simp [selectedEdge, outgoing_e0, e0Outputs, reading_indices,
    branchIndex_e0]

theorem selectedEdge_reading_e1 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) e1 =
      some e1Output := by
  simp [selectedEdge, outgoing_e1]

theorem selectedEdge_reading_root (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) root = none := by
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
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice)
        [hP, hPA, hAB] [hP, hPA, hAB] =
      if majorChoice.val = 0 then [derivedA, e0]
      else if majorChoice.val = 1 then [derivedA, e1]
      else [derivedA, root] := by
  fin_cases majorChoice <;>
    unfold nextVertices <;>
    simp only [List.filterMap_cons, List.filterMap_nil] <;>
    rw [selectedEdge_reading_hP, selectedEdge_reading_hPA,
      selectedEdge_reading_hAB] <;>
    simp [producerMinor, producerMajor, consumerMajorEdge,
      hP, hPA, derivedA, hAB, e0, e1, root, P, A, B, R, PA, AB,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

private theorem second_next_reading (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    nextVertices (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice)
        (if majorChoice.val = 0 then [derivedA, e0]
          else if majorChoice.val = 1 then [derivedA, e1]
          else [derivedA, root])
        ([hP, hPA, hAB] ++
          if majorChoice.val = 0 then [derivedA, e0]
          else if majorChoice.val = 1 then [derivedA, e1]
          else [derivedA, root]) =
      if linkChoice.val = 0 then
        if majorChoice.val = 0 then [root]
        else if majorChoice.val = 1 then [e0, root]
        else [e0]
      else if linkChoice.val = 1 then
        if majorChoice.val = 0 then [e1, root]
        else if majorChoice.val = 1 then [root]
        else [e1]
      else if majorChoice.val = 2 then [] else [root] := by
  fin_cases linkChoice <;> fin_cases majorChoice <;>
    norm_num at * <;>
    unfold nextVertices <;>
    simp only [List.filterMap_cons, List.filterMap_nil] <;>
    rw [selectedEdge_reading_derivedA]
  all_goals first
    | rw [selectedEdge_reading_e0]
    | rw [selectedEdge_reading_e1]
    | rw [selectedEdge_reading_root]
  all_goals
    simp [linkEdge, e0Output, e1Output, hP, hPA, derivedA, hAB, e0, e1,
      root, P, A, B, R, PA, AB, List.eraseDups, List.eraseDupsBy,
      List.eraseDupsBy.loop]

private theorem third_next_reading (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) (v : Vertex)
    (hv : v = e0 ∨ v = e1) :
    nextVertices (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice)
        [v, root] [hP, hPA, hAB, derivedA,
          (if majorChoice.val = 0 then e0 else e1), v, root] = [] := by
  rcases hv with rfl | rfl
  · unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_e0, selectedEdge_reading_root]
    simp [e0Output]
  · unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_e1, selectedEdge_reading_root]
    simp [e1Output]

/-- Exact reachable region of the composed execution.  In every case the
derived vertex is reached from `hP`/`hPA`; the consumer reached first by its
major premise is coherent only when the producer output meets it. -/
theorem selectedReachable_reading (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedReachable (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) =
      if linkChoice.val = 0 then
        if majorChoice.val = 0 then [hP, hPA, hAB, derivedA, e0, root]
        else if majorChoice.val = 1 then [hP, hPA, hAB, derivedA, e1, e0, root]
        else [hP, hPA, hAB, derivedA, root, e0]
      else if linkChoice.val = 1 then
        if majorChoice.val = 0 then [hP, hPA, hAB, derivedA, e0, e1, root]
        else if majorChoice.val = 1 then [hP, hPA, hAB, derivedA, e1, root]
        else [hP, hPA, hAB, derivedA, root, e1]
      else if majorChoice.val = 0 then [hP, hPA, hAB, derivedA, e0, root]
      else if majorChoice.val = 1 then [hP, hPA, hAB, derivedA, e1, root]
      else [hP, hPA, hAB, derivedA, root] := by
  have hs := second_next_reading width bad linkChoice majorChoice outputChoice
  have hlinkBound : linkChoice.val ≤ 2 := by omega
  have hmajorBound : majorChoice.val ≤ 2 := by omega
  interval_cases hlink : linkChoice.val <;>
    interval_cases hmajor : majorChoice.val
  all_goals
    simp only [hlink, hmajor, reduceCtorEq, ↓reduceIte] at hs
    unfold selectedReachable
    rw [sourceNodes_graph, selectedReachable_go_succ, first_next_reading]
    simp only [hlink, hmajor, reduceCtorEq, ↓reduceIte, List.isEmpty_cons,
      Bool.false_eq_true]
    rw [show (graph width bad).NODES.length = 7 from rfl]
    rw [show 7 = 6 + 1 by omega, selectedReachable_go_succ, hs]
  all_goals norm_num at *
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_root]
    simp [hP, hPA, derivedA, hAB, e0, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_e0, selectedEdge_reading_root]
    simp [e0Output, hP, hPA, derivedA, hAB, e0, e1, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_e0]
    simp [e0Output, hP, hPA, derivedA, hAB, e0, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_e1, selectedEdge_reading_root]
    simp [e1Output, hP, hPA, derivedA, hAB, e0, e1, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_root]
    simp [hP, hPA, derivedA, hAB, e1, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_e1]
    simp [e1Output, hP, hPA, derivedA, hAB, e1, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_root]
    simp [hP, hPA, derivedA, hAB, e0, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · rw [show 6 = 5 + 1 by omega, selectedReachable_go_succ]
    unfold nextVertices
    try simp only [List.filterMap_cons, List.filterMap_nil]
    rw [selectedEdge_reading_root]
    simp [hP, hPA, derivedA, hAB, e1, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]
  · simp [hP, hPA, derivedA, hAB, root,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

/-- The producer rule itself is reached independently of all nontrivial
selectors; `derivedA` is not independently seeded. -/
theorem producer_output_reachable (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    derivedA ∈ selectedReachable (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) := by
  rw [selectedReachable_reading]
  fin_cases linkChoice <;> fin_cases majorChoice <;> simp

/-- Effective delivery of the derived premise to `e0` is the producer/link
coordinate 0. -/
theorem producer_delivers_e0 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    feedsB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) derivedA e0 =
      decide (linkChoice.val = 0) := by
  fin_cases linkChoice <;>
    simp [feedsB, selectedEdge_reading_derivedA, linkEdge, e0, e1, root]

theorem producer_delivers_e1 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    feedsB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) derivedA e1 =
      decide (linkChoice.val = 1) := by
  fin_cases linkChoice <;>
    simp [feedsB, selectedEdge_reading_derivedA, linkEdge, e0, e1, root]

theorem producer_bypasses_consumer (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedEdge (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) derivedA =
        some (linkEdge 2) ↔
      linkChoice.val = 2 := by
  rw [selectedEdge_reading_derivedA]
  fin_cases linkChoice <;> simp [linkEdge]

theorem coherentAtB_hP (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hP = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_hP]

theorem coherentAtB_hPA (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hPA = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_hPA]

theorem coherentAtB_hAB (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hAB = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_hAB]

theorem coherentAtB_derivedA (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) derivedA = true := by
  unfold coherentAtB
  rw [selectedReachable_reading]
  simp only [elimPairsAt_derivedA]
  fin_cases linkChoice <;> fin_cases majorChoice <;>
    simp [feedsB, selectedEdge_reading_hP, selectedEdge_reading_hPA,
      producerMinor, producerMajor]

theorem coherentAtB_e0 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) e0 =
      decide ((linkChoice.val = 0 ∧ majorChoice.val = 0) ∨
        (linkChoice.val ≠ 0 ∧ majorChoice.val ≠ 0)) := by
  fin_cases linkChoice <;> fin_cases majorChoice <;>
    unfold coherentAtB <;>
    rw [selectedReachable_reading, elimPairsAt_e0] <;>
    simp only [feedsB, selectedEdge_reading_derivedA,
      selectedEdge_reading_hAB] <;>
    simp [linkEdge, consumerMajorEdge, hP, hPA, derivedA, hAB, e0, e1, root]

theorem coherentAtB_e1 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) e1 =
      decide ((linkChoice.val = 1 ∧ majorChoice.val = 1) ∨
        (linkChoice.val ≠ 1 ∧ majorChoice.val ≠ 1)) := by
  fin_cases linkChoice <;> fin_cases majorChoice <;>
    unfold coherentAtB <;>
    rw [selectedReachable_reading, elimPairsAt_e1] <;>
    simp only [feedsB, selectedEdge_reading_derivedA,
      selectedEdge_reading_hAB] <;>
    simp [linkEdge, consumerMajorEdge, hP, hPA, derivedA, hAB, e0, e1, root]

theorem coherentAtB_root (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) root = true := by
  unfold coherentAtB
  split <;> simp [elimPairsAt_root]

/-- Operational characterization: delivery 0, delivery 1, and the paired
bypass 2 are coherent exactly when both interacting selectors agree. -/
theorem coherentB_reading (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) =
      decide (linkChoice.val = majorChoice.val) := by
  unfold coherentB
  change [hP, hPA, derivedA, hAB, e0, e1, root].all _ = _
  rw [List.all_cons, coherentAtB_hP, List.all_cons, coherentAtB_hPA,
    List.all_cons, coherentAtB_derivedA, List.all_cons, coherentAtB_hAB,
    List.all_cons, coherentAtB_e0, List.all_cons, coherentAtB_e1,
    List.all_cons, coherentAtB_root]
  fin_cases linkChoice <;> fin_cases majorChoice <;> simp

theorem coherent_iff_delivery_matches_major (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) = true ↔
      linkChoice.val = majorChoice.val := by
  rw [coherentB_reading]
  simp

theorem control_defective_selector_structure (width : Nat) :
    fanArities (control width) = fanArities (defective width) := by
  rw [fanArities_graph, fanArities_graph]

theorem control_defective_coherence_agrees (width : Nat)
    (linkChoice majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    coherentB (control width)
        (reading width false linkChoice majorChoice outputChoice) =
      coherentB (defective width)
        (reading width true linkChoice majorChoice outputChoice) := by
  rw [coherentB_reading, coherentB_reading]

theorem selectedRuleActiveB_derivedA (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) derivedA = true := by
  simp [selectedRuleActiveB, producer_output_reachable, coherentAtB_derivedA]

theorem selectedRuleActiveB_e0 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) e0 =
      decide (linkChoice.val = 0 ∧ majorChoice.val = 0) := by
  fin_cases linkChoice <;> fin_cases majorChoice <;>
    simp only [selectedRuleActiveB, coherentAtB_e0] <;>
    rw [selectedReachable_reading] <;>
    simp [hP, hPA, derivedA, hAB, e0, e1, root]

theorem selectedRuleActiveB_e1 (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
        (reading width bad linkChoice majorChoice outputChoice) e1 =
      decide (linkChoice.val = 1 ∧ majorChoice.val = 1) := by
  fin_cases linkChoice <;> fin_cases majorChoice <;>
    simp only [selectedRuleActiveB, coherentAtB_e1] <;>
    rw [selectedReachable_reading] <;>
    simp [hP, hPA, derivedA, hAB, e0, e1, root]

theorem selectedRuleActiveB_hP (width : Nat) (bad : Bool)
    (linkChoice majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hP = true := by
  simp only [selectedRuleActiveB, coherentAtB_hP]
  rw [selectedReachable_reading]
  fin_cases linkChoice <;> fin_cases majorChoice <;> simp

theorem selectedRuleActiveB_hPA (width : Nat) (bad : Bool)
    (linkChoice majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hPA = true := by
  simp only [selectedRuleActiveB, coherentAtB_hPA]
  rw [selectedReachable_reading]
  fin_cases linkChoice <;> fin_cases majorChoice <;> simp

theorem selectedRuleActiveB_hAB (width : Nat) (bad : Bool)
    (linkChoice majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedRuleActiveB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hAB = true := by
  simp only [selectedRuleActiveB, coherentAtB_hAB]
  rw [selectedReachable_reading]
  fin_cases linkChoice <;> fin_cases majorChoice <;> simp

theorem paired_bypass_coherent (width : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2)) :
    coherentB (graph width bad)
      (reading width bad (2 : Fin 3) (2 : Fin 3) outputChoice) = true := by
  simp [coherentB_reading]

theorem producer_rule_shape (width : Nat) (bad : Bool) :
    ruleShapeOKB derivedA (graph width bad) = true := by
  unfold ruleShapeOKB classifyRule?
  rw [incoming_derivedA]
  simp [producerMinor, producerMajor, derivedA, hP, hPA, P, A, PA]

theorem producer_outDep (width : Nat) (bad : Bool) :
    outDep derivedA (graph width bad) = producerDep := by
  unfold outDep classifyRule?
  rw [incoming_derivedA]
  simp [producerMinor, producerMajor, producerDep, derivedA, hP, hPA,
    P, A, PA]

theorem e0_rule_shape (width : Nat) (bad : Bool) :
    ruleShapeOKB e0 (graph width bad) = true := by
  unfold ruleShapeOKB classifyRule?
  rw [incoming_e0]
  simp [linkEdge, consumerMajorEdge, e0, derivedA, hAB, A, B, AB]

theorem e1_rule_shape (width : Nat) (bad : Bool) :
    ruleShapeOKB e1 (graph width bad) = true := by
  unfold ruleShapeOKB classifyRule?
  rw [incoming_e1]
  simp [linkEdge, consumerMajorEdge, e1, derivedA, hAB, A, B, AB]

theorem e0_outDep (width : Nat) (bad : Bool) :
    outDep e0 (graph width bad) = finalDep := by
  unfold outDep classifyRule?
  rw [incoming_e0]
  simp [linkEdge, consumerMajorEdge, finalDep, e0, derivedA, hAB, A, B, AB]

theorem e1_outDep (width : Nat) (bad : Bool) :
    outDep e1 (graph width bad) = finalDep := by
  unfold outDep classifyRule?
  rw [incoming_e1]
  simp [linkEdge, consumerMajorEdge, finalDep, e1, derivedA, hAB, A, B, AB]

theorem hP_rule (width : Nat) (bad : Bool) :
    ruleShapeOKB hP (graph width bad) = true ∧
      outDep hP (graph width bad) = [P] := by
  simp [ruleShapeOKB, outDep, classifyRule?, hP]

theorem hPA_rule (width : Nat) (bad : Bool) :
    ruleShapeOKB hPA (graph width bad) = true ∧
      outDep hPA (graph width bad) = [PA] := by
  simp [ruleShapeOKB, outDep, classifyRule?, hPA]

theorem hAB_rule (width : Nat) (bad : Bool) :
    ruleShapeOKB hAB (graph width bad) = true ∧
      outDep hAB (graph width bad) = [AB] := by
  simp [ruleShapeOKB, outDep, classifyRule?, hAB]

/-- Reusable interface condition missing from unilateral composition: every
left choice has one coherent extension that activates and selects both the
left edge and its paired right edge. -/
structure PairedCoherentExtension (d : Graph) (ι : Type)
    (left right : Vertex) (leftEdge rightEdge : ι → Deduction) where
  extension : ι → ReadingData (fanArities d)
  coherent : ∀ i, coherentB d (extension i) = true
  left_active : ∀ i, selectedRuleActiveB d (extension i) left = true
  right_active : ∀ i, selectedRuleActiveB d (extension i) right = true
  left_selected : ∀ i, selectedEdge d (extension i) left = some (leftEdge i)
  right_selected : ∀ i, selectedEdge d (extension i) right = some (rightEdge i)

theorem PairedCoherentExtension.covers
    {d : Graph} {ι : Type} {left right : Vertex}
    {leftEdge rightEdge : ι → Deduction}
    (interface : PairedCoherentExtension d ι left right leftEdge rightEdge) :
    (∀ i, ExplicitOutputCovered d left (leftEdge i)) ∧
      (∀ i, ExplicitOutputCovered d right (rightEdge i)) := by
  constructor <;> intro i
  · exact ⟨interface.extension i, interface.coherent i,
      interface.left_active i, interface.left_selected i⟩
  · exact ⟨interface.extension i, interface.coherent i,
      interface.right_active i, interface.right_selected i⟩

def pairedInterface (width : Nat) (bad : Bool) :
    PairedCoherentExtension (graph width bad) (Fin 3) derivedA hAB
      (fun i => linkEdge i.val) (fun i => consumerMajorEdge i.val) where
  extension i := reading width bad i i ⟨0, by omega⟩
  coherent i := by simp [coherentB_reading]
  left_active i := selectedRuleActiveB_derivedA width bad i i ⟨0, by omega⟩
  right_active i := selectedRuleActiveB_hAB width bad i i ⟨0, by omega⟩
  left_selected i := selectedEdge_reading_derivedA width bad i i ⟨0, by omega⟩
  right_selected i := selectedEdge_reading_hAB width bad i i ⟨0, by omega⟩

theorem paired_outputs_covered (width : Nat) (bad : Bool) :
    (∀ i : Fin 3,
      ExplicitOutputCovered (graph width bad) derivedA (linkEdge i.val)) ∧
    (∀ i : Fin 3,
      ExplicitOutputCovered (graph width bad) hAB (consumerMajorEdge i.val)) :=
  (pairedInterface width bad).covers

theorem e0_output_covered (width : Nat) (bad : Bool) (i : Nat)
    (hi : i < width + 2) :
    ExplicitOutputCovered (graph width bad) e0 (e0Output bad i) := by
  let outputChoice : Fin (width + 2) := ⟨i, hi⟩
  let r := reading width bad (0 : Fin 3) (0 : Fin 3) outputChoice
  refine ⟨r, ?_, ?_, ?_⟩
  · simp [r, coherentB_reading]
  · simpa [r] using selectedRuleActiveB_e0 width bad 0 0 outputChoice
  · simpa [r, outputChoice] using
      selectedEdge_reading_e0 width bad 0 0 outputChoice

theorem e1_output_covered (width : Nat) (bad : Bool) :
    ExplicitOutputCovered (graph width bad) e1 e1Output := by
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let r := reading width bad (1 : Fin 3) (1 : Fin 3) outputChoice
  refine ⟨r, ?_, ?_, ?_⟩
  · simp [r, coherentB_reading]
  · simpa [r] using selectedRuleActiveB_e1 width bad 1 1 outputChoice
  · simpa [r] using selectedEdge_reading_e1 width bad 1 1 outputChoice

theorem hP_output_covered (width : Nat) (bad : Bool) :
    ExplicitOutputCovered (graph width bad) hP producerMinor := by
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let r := reading width bad (0 : Fin 3) (0 : Fin 3) outputChoice
  exact ⟨r, by simp [r, coherentB_reading],
    by simpa [r] using selectedRuleActiveB_hP width bad 0 0 outputChoice,
    by simpa [r] using selectedEdge_reading_hP width bad 0 0 outputChoice⟩

theorem hPA_output_covered (width : Nat) (bad : Bool) :
    ExplicitOutputCovered (graph width bad) hPA producerMajor := by
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let r := reading width bad (0 : Fin 3) (0 : Fin 3) outputChoice
  exact ⟨r, by simp [r, coherentB_reading],
    by simpa [r] using selectedRuleActiveB_hPA width bad 0 0 outputChoice,
    by simpa [r] using selectedEdge_reading_hPA width bad 0 0 outputChoice⟩

theorem allExplicitOutputsCovered_graph (width : Nat) (bad : Bool) :
    AllExplicitOutputsCovered (graph width bad) := by
  intro v hv edge hedge
  simp [graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · rw [outgoing_hP] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    exact hP_output_covered width bad
  · rw [outgoing_hPA] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    exact hPA_output_covered width bad
  · rw [outgoing_derivedA] at hedge
    simp at hedge
    rcases hedge with rfl | rfl | rfl
    · exact (paired_outputs_covered width bad).1 0
    · exact (paired_outputs_covered width bad).1 1
    · exact (paired_outputs_covered width bad).1 2
  · rw [outgoing_hAB] at hedge
    simp at hedge
    rcases hedge with rfl | rfl | rfl
    · exact (paired_outputs_covered width bad).2 0
    · exact (paired_outputs_covered width bad).2 1
    · exact (paired_outputs_covered width bad).2 2
  · rw [outgoing_e0] at hedge
    obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hedge
    exact e0_output_covered width bad i (by simpa using hi)
  · rw [outgoing_e1] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    exact e1_output_covered width bad
  · rw [outgoing_root] at hedge
    simp at hedge

theorem control_allExplicitOutputsCovered (width : Nat) :
    AllExplicitOutputsCovered (control width) :=
  allExplicitOutputsCovered_graph width false

theorem defective_allExplicitOutputsCovered (width : Nat) :
    AllExplicitOutputsCovered (defective width) :=
  allExplicitOutputsCovered_graph width true

theorem explicitOutputLabelsCorrect_iff_no_labelDefect_graph
    (width : Nat) (bad : Bool) :
    ExplicitOutputLabelsCorrect (graph width bad) ↔
      ∀ r : ReadingData (fanArities (graph width bad)),
        labelDefectB (graph width bad) r = false :=
  explicitOutputLabelsCorrect_iff_no_labelDefect _
    (allExplicitOutputsCovered_graph width bad)

theorem each_interface_choice_has_coherent_extension
    (width : Nat) (bad : Bool) (choice : Fin 3) :
    ∃ r : ReadingData (fanArities (graph width bad)),
      coherentB (graph width bad) r = true ∧
      selectedEdge (graph width bad) r derivedA = some (linkEdge choice.val) ∧
      selectedEdge (graph width bad) r hAB =
        some (consumerMajorEdge choice.val) := by
  exact ⟨(pairedInterface width bad).extension choice,
    (pairedInterface width bad).coherent choice,
    (pairedInterface width bad).left_selected choice,
    (pairedInterface width bad).right_selected choice⟩

theorem incompatible_interface_choices_exist (width : Nat) (bad : Bool) :
    ∃ r : ReadingData (fanArities (graph width bad)),
      coherentB (graph width bad) r = false := by
  refine ⟨reading width bad (0 : Fin 3) (1 : Fin 3) ⟨0, by omega⟩, ?_⟩
  simp [coherentB_reading]

theorem control_explicitOutputLabelsCorrect (width : Nat) :
    ExplicitOutputLabelsCorrect (control width) := by
  intro v hv edge hedge
  simp [control, graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · rw [outgoing_hP] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    simpa [producerMinor, hP_rule]
  · rw [outgoing_hPA] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    simpa [producerMajor, hPA_rule]
  · rw [outgoing_derivedA] at hedge
    simp at hedge
    rcases hedge with rfl | rfl | rfl <;>
      exact ⟨producer_rule_shape width false, by simp [linkEdge, producer_outDep]⟩
  · rw [outgoing_hAB] at hedge
    simp at hedge
    rcases hedge with rfl | rfl | rfl <;>
      simpa [consumerMajorEdge, hAB_rule]
  · rw [outgoing_e0] at hedge
    obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp hedge
    exact ⟨e0_rule_shape width false, by simp [e0Output, e0_outDep]⟩
  · rw [outgoing_e1] at hedge
    simp only [List.mem_singleton] at hedge
    subst edge
    exact ⟨e1_rule_shape width false, by simp [e1Output, e1_outDep]⟩
  · rw [outgoing_root] at hedge
    simp at hedge

/-- Soundness, not global coverage/completeness, proves absence of marks in
the declaratively correct control. -/
theorem control_no_labelDefect (width : Nat) :
    ∀ r : ReadingData (fanArities (control width)),
      labelDefectB (control width) r = false :=
  labelDefectB_eq_false_of_explicitOutputLabelsCorrect _
    (control_explicitOutputLabelsCorrect width)

theorem bad_dependency_ne_finalDep : [Z] ≠ finalDep := by
  intro h
  have hmem : P ∈ finalDep := by
    simp [finalDep, producerDep]
  rw [← h] at hmem
  simpa [P, Z] using hmem

theorem selectedLabelErrorAtB_hP_false (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hP = false := by
  simp [selectedLabelErrorAtB, selectedEdge_reading_hP, producerMinor,
    hP_rule]

theorem selectedLabelErrorAtB_hPA_false (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hPA = false := by
  simp [selectedLabelErrorAtB, selectedEdge_reading_hPA, producerMajor,
    hPA_rule]

theorem selectedLabelErrorAtB_derivedA_false (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) derivedA = false := by
  fin_cases linkChoice <;>
    simp [selectedLabelErrorAtB, selectedEdge_reading_derivedA,
      producer_rule_shape, producer_outDep, linkEdge]

theorem selectedLabelErrorAtB_hAB_false (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) hAB = false := by
  fin_cases majorChoice <;>
    simp [selectedLabelErrorAtB, selectedEdge_reading_hAB,
      consumerMajorEdge, hAB_rule]

theorem selectedLabelErrorAtB_e0_defective (width : Nat)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (defective width)
        (reading width true linkChoice majorChoice outputChoice) e0 =
      decide (linkChoice.val = 0 ∧ majorChoice.val = 0 ∧
        outputChoice.val = 1) := by
  by_cases houtput : outputChoice.val = 1
  · fin_cases linkChoice <;> fin_cases majorChoice <;>
      simp [selectedLabelErrorAtB, selectedRuleActiveB_e0,
        selectedEdge_reading_e0, e0_rule_shape, e0_outDep, e0Output,
        houtput, bad_dependency_ne_finalDep]
  · fin_cases linkChoice <;> fin_cases majorChoice <;>
      simp [selectedLabelErrorAtB, selectedRuleActiveB_e0,
        selectedEdge_reading_e0, e0_rule_shape, e0_outDep, e0Output,
        houtput]

theorem selectedLabelErrorAtB_e1_false (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) e1 = false := by
  simp [selectedLabelErrorAtB, selectedEdge_reading_e1, e1_rule_shape,
    e1_outDep, e1Output]

theorem selectedLabelErrorAtB_root_false (width : Nat) (bad : Bool)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelErrorAtB (graph width bad)
      (reading width bad linkChoice majorChoice outputChoice) root = false := by
  simp [selectedLabelErrorAtB, selectedEdge_reading_root]

theorem selectedLabelHadErrorB_defective (width : Nat)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    selectedLabelHadErrorB (defective width)
        (reading width true linkChoice majorChoice outputChoice) =
      decide (linkChoice.val = 0 ∧ majorChoice.val = 0 ∧
        outputChoice.val = 1) := by
  unfold selectedLabelHadErrorB
  change [hP, hPA, derivedA, hAB, e0, e1, root].any _ = _
  simp [selectedLabelErrorAtB_hP_false, selectedLabelErrorAtB_hPA_false,
    selectedLabelErrorAtB_derivedA_false, selectedLabelErrorAtB_hAB_false,
    selectedLabelErrorAtB_e0_defective, selectedLabelErrorAtB_e1_false,
    selectedLabelErrorAtB_root_false]

/-- Exact characterization of the coherent defect event in the composed
graph. -/
theorem labelDefectB_defective_iff (width : Nat)
    (linkChoice : Fin 3) (majorChoice : Fin 3)
    (outputChoice : Fin (width + 2)) :
    labelDefectB (defective width)
        (reading width true linkChoice majorChoice outputChoice) = true ↔
      linkChoice.val = 0 ∧ majorChoice.val = 0 ∧
        outputChoice.val = 1 := by
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

def changedProducer (width : Nat) :
    ReadingData (fanArities (defective width)) :=
  reading width true ⟨2, by omega⟩ ⟨0, by omega⟩ ⟨1, by omega⟩

def incompatibleEarlier (width : Nat) :
    ReadingData (fanArities (defective width)) :=
  reading width true ⟨0, by omega⟩ ⟨1, by omega⟩ ⟨1, by omega⟩

def compatibleOtherElimination (width : Nat) :
    ReadingData (fanArities (defective width)) :=
  reading width true ⟨1, by omega⟩ ⟨1, by omega⟩ ⟨1, by omega⟩

def compatibleBypass (width : Nat) :
    ReadingData (fanArities (defective width)) :=
  reading width true ⟨2, by omega⟩ ⟨2, by omega⟩ ⟨1, by omega⟩

theorem badReference_indices (width : Nat) :
    (badReference width).indices = [0, 0, 1] := by
  simp [badReference, reading_indices]

theorem changedProducer_indices (width : Nat) :
    (changedProducer width).indices = [2, 0, 1] := by
  simp [changedProducer, reading_indices]

theorem badReference_marks (width : Nat) :
    labelDefectB (defective width) (badReference width) = true := by
  unfold badReference
  exact (labelDefectB_defective_iff width _ _ _).mpr (by simp)

theorem defective_not_explicitOutputLabelsCorrect (width : Nat) :
    ¬ ExplicitOutputLabelsCorrect (defective width) := by
  intro hcorrect
  have hnone :=
    (explicitOutputLabelsCorrect_iff_no_labelDefect_graph width true).mp hcorrect
  have := hnone (badReference width)
  rw [badReference_marks] at this
  simp at this

/-- Keeping both final coordinates while changing the producer/link choice to
the bypass preserves the bad output index but destroys global coherence. -/
theorem changedProducer_effect (width : Nat) :
    (badReference width).indices.getD 1 0 =
        (changedProducer width).indices.getD 1 0 ∧
      (badReference width).indices.getD 2 0 =
        (changedProducer width).indices.getD 2 0 ∧
      coherentB (defective width) (changedProducer width) = false ∧
      labelDefectB (defective width) (changedProducer width) = false ∧
      rejectsReadingLabelsB (defective width) (changedProducer width) = true := by
  refine ⟨by simp [badReference_indices, changedProducer_indices],
    by simp [badReference_indices, changedProducer_indices], ?_, ?_, ?_⟩
  · simp [changedProducer, coherentB_reading]
  · apply labelDefectB_eq_false_of_incoherent
    simp [changedProducer, coherentB_reading]
  · simp [changedProducer, rejectsReadingLabelsB, coherentB_reading]

theorem three_final_bad_choice_outcomes (width : Nat) :
    labelDefectB (defective width) (badReference width) = true ∧
      (coherentB (defective width) (incompatibleEarlier width) = false ∧
        rejectsReadingLabelsB (defective width) (incompatibleEarlier width) = true) ∧
      (coherentB (defective width) (compatibleOtherElimination width) = true ∧
        labelDefectB (defective width) (compatibleOtherElimination width) = false ∧
        rejectsReadingLabelsB (defective width)
          (compatibleOtherElimination width) = false) ∧
      (coherentB (defective width) (compatibleBypass width) = true ∧
        labelDefectB (defective width) (compatibleBypass width) = false ∧
        rejectsReadingLabelsB (defective width) (compatibleBypass width) = false) := by
  refine ⟨badReference_marks width, ?_, ?_, ?_⟩
  · simp [incompatibleEarlier, coherentB_reading, rejectsReadingLabelsB]
  · simp [compatibleOtherElimination, coherentB_reading, labelDefectB,
      rejectsReadingLabelsB, selectedLabelHadErrorB_defective]
  · simp [compatibleBypass, coherentB_reading, labelDefectB,
      rejectsReadingLabelsB, selectedLabelHadErrorB_defective]

def sufficientSupport : Finset Nat := {0, 1, 2}

def AgreesOnSupport {d : Graph} (support : Finset Nat)
    (x y : ReadingData (fanArities d)) : Prop :=
  ∀ i ∈ support, x.indices.getD i 0 = y.indices.getD i 0

theorem sufficientSupport_card : sufficientSupport.card = 3 := by decide

/-- Agreement on the three original input coordinates derives the whole
operational chain: producer delivery, global coherence, final activation,
selection of the corrupted output, and the final marker. -/
theorem sufficientSupport_derives_operational_conditions (width : Nat)
    (y : ReadingData (fanArities (defective width)))
    (hagrees : AgreesOnSupport sufficientSupport (badReference width) y) :
    feedsB (defective width) y derivedA e0 = true ∧
      coherentB (defective width) y = true ∧
      selectedRuleActiveB (defective width) y e0 = true ∧
      (∃ selected, selectedEdge (defective width) y e0 = some selected ∧
        selected.DEPENDENCY = [Z]) ∧
      labelDefectB (defective width) y = true := by
  obtain ⟨linkChoice, majorChoice, outputChoice, rfl⟩ :=
    exists_reading_choices width true y
  have hlink := hagrees 0 (by simp [sufficientSupport])
  have hmajor := hagrees 1 (by simp [sufficientSupport])
  have houtput := hagrees 2 (by simp [sufficientSupport])
  simp [badReference_indices, reading_indices] at hlink hmajor houtput
  have hl : linkChoice.val = 0 := hlink.symm
  have hm : majorChoice.val = 0 := hmajor.symm
  have ho : outputChoice.val = 1 := houtput.symm
  refine ⟨?_, ?_, ?_, ?_,
    (labelDefectB_defective_iff width _ _ _).mpr ⟨hl, hm, ho⟩⟩
  · rw [producer_delivers_e0]
    simp [hl]
  · rw [coherentB_reading]
    simp [hl, hm]
  · rw [selectedRuleActiveB_e0]
    simp [hl, hm]
  · refine ⟨e0Output true outputChoice.val,
      selectedEdge_reading_e0 width true _ _ _, ?_⟩
    simp [e0Output, ho]

theorem sufficientSupport_forces_labelDefect (width : Nat)
    (y : ReadingData (fanArities (defective width)))
    (hagrees : AgreesOnSupport sufficientSupport (badReference width) y) :
    labelDefectB (defective width) y = true :=
  (sufficientSupport_derives_operational_conditions width y hagrees).2.2.2.2

theorem sufficientSupport_forces_rejection (width : Nat)
    (y : ReadingData (fanArities (defective width)))
    (hagrees : AgreesOnSupport sufficientSupport (badReference width) y) :
    rejectsReadingLabelsB (defective width) y = true :=
  labelDefectB_implies_rejectsReadingLabelsB _ _
    (sufficientSupport_forces_labelDefect width y hagrees)

theorem readings_complete (width : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width bad))) :
    r ∈ readings (graph width bad) := mem_enumerateReadingData r

theorem readings_nodup (width : Nat) (bad : Bool) :
    (readings (graph width bad)).Nodup := enumerateReadingData_nodup _

theorem raw_reading_count (width : Nat) (bad : Bool) :
    (readings (graph width bad)).length = 9 * (width + 2) := by
  rw [readings, enumerateReadingData_length, fanArities_graph]
  simp
  omega

def coherentCylinder (width : Nat) (bad : Bool) :
    List (ReadingData (fanArities (graph width bad))) :=
  (List.finRange 3).flatMap fun choice =>
    (List.finRange (width + 2)).map fun outputChoice =>
      reading width bad choice choice outputChoice

theorem coherentCylinder_count (width : Nat) (bad : Bool) :
    (coherentCylinder width bad).length = 3 * (width + 2) := by
  simp [coherentCylinder]
  omega

theorem coherentCylinder_nodup (width : Nat) (bad : Bool) :
    (coherentCylinder width bad).Nodup := by
  unfold coherentCylinder
  apply List.nodup_flatMap.mpr
  constructor
  · intro choice _hchoice
    exact (List.nodup_finRange (width + 2)).map (by
      intro left right hreading
      have hindices := congrArg Readings.Selection.ReadingData.indices hreading
      simp [reading_indices] at hindices
      exact Fin.ext hindices)
  · exact (List.nodup_finRange 3).imp (by
      intro left right hne candidate hleft hright
      obtain ⟨leftOutput, _hl, rfl⟩ := List.mem_map.mp hleft
      obtain ⟨rightOutput, _hr, hreading⟩ := List.mem_map.mp hright
      have hindices := congrArg Readings.Selection.ReadingData.indices hreading
      simp [reading_indices] at hindices
      exact hne (Fin.ext hindices.1.symm))

theorem mem_coherentCylinder_iff (width : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width bad))) :
    r ∈ coherentCylinder width bad ↔
      r ∈ coherentReadings (graph width bad) := by
  constructor
  · intro hr
    obtain ⟨choice, _hc, hinner⟩ := List.mem_flatMap.mp hr
    obtain ⟨outputChoice, _ho, rfl⟩ := List.mem_map.mp hinner
    exact List.mem_filter.mpr
      ⟨readings_complete width bad _, by simp [coherentB_reading]⟩
  · intro hr
    have hparts := List.mem_filter.mp hr
    obtain ⟨linkChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width bad r
    have hmatch :=
      (coherent_iff_delivery_matches_major width bad linkChoice majorChoice
        outputChoice).mp hparts.2
    have heq : linkChoice = majorChoice := Fin.ext hmatch
    subst majorChoice
    apply List.mem_flatMap.mpr
    refine ⟨linkChoice, by simp, ?_⟩
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
    (coherentReadings (graph width bad)).length = 3 * (width + 2) := by
  calc
    (coherentReadings (graph width bad)).length =
        (coherentCylinder width bad).length :=
      (coherentCylinder_perm_coherentReadings width bad).length_eq.symm
    _ = 3 * (width + 2) := coherentCylinder_count width bad

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
    ((readings_nodup width true).filter _) (by simp)).mpr
  intro r
  constructor
  · intro hr
    have hparts := List.mem_filter.mp hr
    obtain ⟨linkChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width true r
    have hvalues :=
      (labelDefectB_defective_iff width linkChoice majorChoice outputChoice).mp
        hparts.2
    have hl : linkChoice = ⟨0, by omega⟩ := Fin.ext hvalues.1
    have hm : majorChoice = ⟨0, by omega⟩ := Fin.ext hvalues.2.1
    have ho : outputChoice = ⟨1, by omega⟩ := Fin.ext hvalues.2.2
    simp [hl, hm, ho, badReference]
  · intro hr
    simp only [List.mem_singleton] at hr
    subst r
    exact List.mem_filter.mpr
      ⟨readings_complete width true _, badReference_marks width⟩

theorem defective_mark_count (width : Nat) :
    (markedReadings (defective width)).length = 1 := by
  simpa using (defective_marked_perm_singleton width).length_eq

def sufficientSupportCylinder (width : Nat) :
    List (ReadingData (fanArities (defective width))) := [badReference width]

theorem sufficientSupportCylinder_count (width : Nat) :
    (sufficientSupportCylinder width).length = 1 := by rfl

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
    obtain ⟨linkChoice, majorChoice, outputChoice, rfl⟩ :=
      exists_reading_choices width true r
    have hlink := hagrees 0 (by simp [sufficientSupport])
    have hmajor := hagrees 1 (by simp [sufficientSupport])
    have houtput := hagrees 2 (by simp [sufficientSupport])
    simp [badReference_indices, reading_indices] at hlink hmajor houtput
    have hl : linkChoice = ⟨0, by omega⟩ := Fin.ext hlink.symm
    have hm : majorChoice = ⟨0, by omega⟩ := Fin.ext hmajor.symm
    have ho : outputChoice = ⟨1, by omega⟩ := Fin.ext houtput.symm
    simp [sufficientSupportCylinder, badReference, hl, hm, ho]

theorem sufficientSupportCylinder_all_marked (width : Nat) :
    ∀ r ∈ sufficientSupportCylinder width,
      labelDefectB (defective width) r = true := by
  intro r hr
  simp [sufficientSupportCylinder] at hr
  subst r
  exact badReference_marks width

theorem sufficientSupportCylinder_raw_density (width : Nat) :
    ((sufficientSupportCylinder width).length : ℚ) /
        ((readings (defective width)).length : ℚ) =
      1 / (9 * (width + 2 : ℚ)) := by
  rw [sufficientSupportCylinder_count, raw_reading_count]
  push_cast
  ring

theorem defective_total_raw_density (width : Nat) :
    ((markedReadings (defective width)).length : ℚ) /
        ((readings (defective width)).length : ℚ) =
      1 / (9 * (width + 2 : ℚ)) := by
  rw [defective_mark_count, raw_reading_count]
  push_cast
  ring

theorem defective_conditioned_coherent_density (width : Nat) :
    ((markedReadings (defective width)).length : ℚ) /
        ((coherentReadings (defective width)).length : ℚ) =
      1 / (3 * (width + 2 : ℚ)) := by
  rw [defective_mark_count, coherent_reading_count]
  push_cast
  ring


end CoveredComposition
end Readings.Explicit
end DLDS
