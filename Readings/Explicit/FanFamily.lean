import Readings.Explicit.LabelValidity

open scoped Classical

set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unnecessarySimpa false

/-!
# A variable explicit-output family

This module leaves `labelDefectB` unchanged and studies a family in which a
genuine implication elimination has a variable output fan.  A second,
independent hypothesis fan supplies an arbitrary number of selector choices
which do not affect the elimination or global coherence.
-/

namespace DLDS
namespace Readings.Explicit
namespace UniformElimFanFamily

open FlowSemantics
open Readings.Selection
open DLDS.Readings.LocalLabels

/-- The repository enumerator is extensionally complete for its dependent
selector product. -/
theorem mem_enumerateReadingData {ns : List Nat} (r : ReadingData ns) :
    r ∈ enumerateReadingData ns := by
  induction r with
  | nil => simp [enumerateReadingData]
  | @cons n ns i rest ih =>
      simp only [enumerateReadingData]
      apply List.mem_flatMap.mpr
      refine ⟨i, by simp, ?_⟩
      exact List.mem_map.mpr ⟨rest, ih, rfl⟩

/-- The repository enumerator has no duplicates; hence list cardinalities below
are cardinalities of the actual `ReadingData` domain. -/
theorem enumerateReadingData_nodup :
    ∀ ns : List Nat, (enumerateReadingData ns).Nodup
  | [] => by simp [enumerateReadingData]
  | n :: ns => by
      simp only [enumerateReadingData]
      apply List.nodup_flatMap.mpr
      constructor
      · intro i hi
        exact (enumerateReadingData_nodup ns).map (by
          intro left right h
          injection h)
      · exact (List.nodup_finRange n).imp (by
          intro i j hij candidate hi hj
          obtain ⟨left, _hleft, rfl⟩ := List.mem_map.mp hi
          obtain ⟨right, _hright, hcons⟩ := List.mem_map.mp hj
          have hindices := congrArg Readings.Selection.ReadingData.indices hcons
          have hval : j.val = i.val := by
            injection hindices
          exact hij (Fin.ext hval.symm))

theorem enumerateReadingData_length :
    ∀ ns : List Nat, (enumerateReadingData ns).length = ns.prod
  | [] => by simp [enumerateReadingData]
  | n :: ns => by
      simp [enumerateReadingData, enumerateReadingData_length ns]
def A : Formula := #"family_A"
def B : Formula := #"family_B"
def L : Formula := #"family_L"
def R : Formula := #"family_R"
def Z : Formula := #"family_bad"
def AB : Formula := A >> B

def minor : Vertex := node 0 3 A true false []
def major : Vertex := node 1 3 AB true false []
def elim : Vertex := node 2 2 B false false []
def independent : Vertex := node 3 1 L true false []

def elimRoot : Vertex := node 4 1 R false false []
def independentRoot : Vertex := node 5 0 R false false []

def premiseEdges : List Deduction :=
  [edge minor elim 0 [A], edge major elim 0 [AB]]

def goodDep : Dep := normalizeDep [A, AB]

def outputEdge (bad : Bool) (i : Nat) : Deduction :=
  edge elim elimRoot i
    (if bad && i == 1 then [Z] else goodDep)

def outputEdges (width : Nat) (bad : Bool) : List Deduction :=
  (List.range (width + 2)).map (outputEdge bad)

def independentEdge (i : Nat) : Deduction :=
  edge independent independentRoot i [L]

def independentEdges (freeWidth : Nat) : List Deduction :=
  (List.range (freeWidth + 2)).map independentEdge

def graph (width freeWidth : Nat) (bad : Bool) : Graph :=
  dlds
    [minor, major, elim, independent, elimRoot, independentRoot]
    (premiseEdges ++ outputEdges width bad ++ independentEdges freeWidth)
    []

abbrev control (width freeWidth : Nat) : Graph :=
  graph width freeWidth false

abbrev defective (width freeWidth : Nat) : Graph :=
  graph width freeWidth true

/-- The structural family is the image of the explicit generator.  Its shape
mentions only incidence, levels, rules, fans and routing; `bad` may vary, so
membership is independent of output-label correctness. -/
def IsUniformElimFan (d : Graph) : Prop :=
  ∃ width freeWidth bad, d = graph width freeWidth bad

theorem control_mem_family (width freeWidth : Nat) :
    IsUniformElimFan (control width freeWidth) :=
  ⟨width, freeWidth, false, rfl⟩

theorem defective_mem_family (width freeWidth : Nat) :
    IsUniformElimFan (defective width freeWidth) :=
  ⟨width, freeWidth, true, rfl⟩

private theorem filter_outputEdges_at_elim (width : Nat) (bad : Bool) :
    (outputEdges width bad).filter (fun e => decide (e.START = elim)) =
      outputEdges width bad := by
  apply List.filter_eq_self.mpr
  intro e he
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he
  simp [outputEdge]

private theorem filter_outputEdges_at_independent (width : Nat) (bad : Bool) :
    (outputEdges width bad).filter
      (fun e => decide (e.START = independent)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro e he
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he
  simp [outputEdge, elim, independent]

private theorem filter_independentEdges_at_elim (freeWidth : Nat) :
    (independentEdges freeWidth).filter
      (fun e => decide (e.START = elim)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro e he
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he
  simp [independentEdge, elim, independent]

private theorem filter_independentEdges_at_independent (freeWidth : Nat) :
    (independentEdges freeWidth).filter
      (fun e => decide (e.START = independent)) = independentEdges freeWidth := by
  apply List.filter_eq_self.mpr
  intro e he
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he
  simp [independentEdge]

theorem fixed_structure_when_label_changes (width freeWidth : Nat) :
    (control width freeWidth).NODES = (defective width freeWidth).NODES ∧
    ((control width freeWidth).EDGES.map fun e =>
      (e.START, e.END, e.COLOUR)) =
    ((defective width freeWidth).EDGES.map fun e =>
      (e.START, e.END, e.COLOUR)) ∧
    (control width freeWidth).PATHS = (defective width freeWidth).PATHS := by
  simp [control, defective, graph, premiseEdges, outputEdges, outputEdge,
    independentEdges]

theorem outgoing_minor (width freeWidth : Nat) (bad : Bool) :
    outgoing (graph width freeWidth bad) minor = [edge minor elim 0 [A]] := by
  rw [show outgoing (graph width freeWidth bad) minor =
    (graph width freeWidth bad).EDGES.filter
      (fun e => decide (e.START = minor)) from outgoing_eq_filter minor _]
  simp [graph, premiseEdges, outputEdges, outputEdge,
    independentEdges, independentEdge, minor, major, elim, independent,
    elimRoot, independentRoot]

theorem outgoing_major (width freeWidth : Nat) (bad : Bool) :
    outgoing (graph width freeWidth bad) major = [edge major elim 0 [AB]] := by
  rw [show outgoing (graph width freeWidth bad) major =
    (graph width freeWidth bad).EDGES.filter
      (fun e => decide (e.START = major)) from outgoing_eq_filter major _]
  simp [graph, premiseEdges, outputEdges, outputEdge,
    independentEdges, independentEdge, minor, major, elim, independent,
    elimRoot, independentRoot]

theorem outgoing_elim (width freeWidth : Nat) (bad : Bool) :
    outgoing (graph width freeWidth bad) elim = outputEdges width bad := by
  rw [show outgoing (graph width freeWidth bad) elim =
    (graph width freeWidth bad).EDGES.filter
      (fun e => decide (e.START = elim)) from outgoing_eq_filter elim _]
  simp only [graph, Graph.EDGES, premiseEdges, List.filter_append]
  simp only [List.filter_cons]
  simp [minor, major, elim, independent]
  change
    (outputEdges width bad).filter (fun e => decide (e.START = elim)) ++
      (independentEdges freeWidth).filter
        (fun e => decide (e.START = elim)) = outputEdges width bad
  rw [filter_outputEdges_at_elim, filter_independentEdges_at_elim]
  simp

theorem outgoing_independent (width freeWidth : Nat) (bad : Bool) :
    outgoing (graph width freeWidth bad) independent =
      independentEdges freeWidth := by
  rw [show outgoing (graph width freeWidth bad) independent =
    (graph width freeWidth bad).EDGES.filter
      (fun e => decide (e.START = independent)) from outgoing_eq_filter independent _]
  simp only [graph, Graph.EDGES, premiseEdges, List.filter_append]
  simp only [List.filter_cons]
  simp [minor, major, elim, independent]
  change
    (outputEdges width bad).filter
        (fun e => decide (e.START = independent)) ++
      (independentEdges freeWidth).filter
        (fun e => decide (e.START = independent)) = independentEdges freeWidth
  rw [filter_outputEdges_at_independent,
    filter_independentEdges_at_independent]
  simp

theorem outgoing_elimRoot (width freeWidth : Nat) (bad : Bool) :
    outgoing (graph width freeWidth bad) elimRoot = [] := by
  rw [show outgoing (graph width freeWidth bad) elimRoot =
    (graph width freeWidth bad).EDGES.filter
      (fun e => decide (e.START = elimRoot)) from outgoing_eq_filter _ _]
  apply List.filter_eq_nil_iff.mpr
  intro e he
  simp [graph, premiseEdges, outputEdges, independentEdges, outputEdge,
    independentEdge, minor, major, elim, independent, elimRoot,
    independentRoot] at he
  rcases he with rfl | rfl | ⟨j, _hj, rfl⟩ | ⟨j, _hj, rfl⟩ <;>
    simp [minor, major, elim, independent, elimRoot]

theorem outgoing_independentRoot (width freeWidth : Nat) (bad : Bool) :
    outgoing (graph width freeWidth bad) independentRoot = [] := by
  rw [show outgoing (graph width freeWidth bad) independentRoot =
    (graph width freeWidth bad).EDGES.filter
      (fun e => decide (e.START = independentRoot)) from outgoing_eq_filter _ _]
  apply List.filter_eq_nil_iff.mpr
  intro e he
  simp [graph, premiseEdges, outputEdges, independentEdges, outputEdge,
    independentEdge, minor, major, elim, independent, elimRoot,
    independentRoot] at he
  rcases he with rfl | rfl | ⟨j, _hj, rfl⟩ | ⟨j, _hj, rfl⟩ <;>
    simp [minor, major, elim, independent, independentRoot]

theorem fanNodes_graph (width freeWidth : Nat) (bad : Bool) :
    (graph width freeWidth bad).NODES.filter
      (fun v => decide (2 ≤ (outgoing (graph width freeWidth bad) v).length)) =
      [elim, independent] := by
  change
    [minor, major, elim, independent, elimRoot, independentRoot].filter _ = _
  simp [outgoing_minor, outgoing_major, outgoing_elim,
    outgoing_independent, outgoing_elimRoot, outgoing_independentRoot,
    outputEdges, independentEdges]

theorem branchPoints_graph (width freeWidth : Nat) (bad : Bool) :
    branchPoints (graph width freeWidth bad) = [elim, independent] := by
  unfold branchPoints
  rw [fanNodes_graph]
  let cmp : Vertex → Vertex → Bool :=
    fun u v => decide (u.NUMBER ≤ v.NUMBER)
  generalize hsortedList : ([elim, independent].mergeSort cmp) = sortedList
  have hperm : sortedList.Perm [elim, independent] := by
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
  have hlen : sortedList.length = 2 := by
    simpa using hperm.length_eq
  cases sortedList with
  | nil => simp at hlen
  | cons x xs =>
      cases xs with
      | nil => simp at hlen
      | cons y ys =>
          cases ys with
          | cons z zs => simp at hlen
          | nil =>
              have hx : x = elim ∨ x = independent := by
                have : x ∈ [elim, independent] := hperm.mem_iff.mp (by simp)
                simpa using this
              have hy : y = elim ∨ y = independent := by
                have : y ∈ [elim, independent] := hperm.mem_iff.mp (by simp)
                simpa using this
              have hnodup : [x, y].Nodup := by
                exact hperm.nodup_iff.mpr (by
                  simp [elim, independent])
              rcases hx with rfl | rfl <;> rcases hy with rfl | rfl <;>
                simp [cmp, elim, independent] at hnodup hsorted ⊢

theorem fanArities_graph (width freeWidth : Nat) (bad : Bool) :
    fanArities (graph width freeWidth bad) = [width + 2, freeWidth + 2] := by
  simp [fanArities, branchPoints_graph, outgoing_elim, outgoing_independent,
    outputEdges, independentEdges]

@[simp] theorem readingData_indices_cast {ns ms : List Nat}
    (h : ns = ms) (r : ReadingData ns) :
    (h ▸ r).indices = r.indices := by
  subst h
  rfl

theorem readingData_cast_symm_cast {ns ms : List Nat}
    (h : ns = ms) (r : ReadingData ns) :
    h.symm ▸ (h ▸ r) = r := by
  cases h
  rfl

private theorem filter_outputEdges_ending_at_elim (width : Nat) (bad : Bool) :
    (outputEdges width bad).filter (fun e => decide (e.END = elim)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro e he
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he
  simp [outputEdge, elimRoot, elim]

private theorem filter_independentEdges_ending_at_elim (freeWidth : Nat) :
    (independentEdges freeWidth).filter
      (fun e => decide (e.END = elim)) = [] := by
  apply List.filter_eq_nil_iff.mpr
  intro e he
  obtain ⟨i, _hi, rfl⟩ := List.mem_map.mp he
  simp [independentEdge, independentRoot, elim]

theorem incoming_elim (width freeWidth : Nat) (bad : Bool) :
    get_rule.incoming elim (graph width freeWidth bad) = premiseEdges := by
  rw [incoming_eq_filter]
  simp only [graph, Graph.EDGES, List.filter_append]
  change premiseEdges.filter (fun e => decide (e.END = elim)) ++
    (outputEdges width bad).filter (fun e => decide (e.END = elim)) ++
    (independentEdges freeWidth).filter (fun e => decide (e.END = elim)) =
      premiseEdges
  rw [filter_outputEdges_ending_at_elim,
    filter_independentEdges_ending_at_elim]
  simp [premiseEdges]

theorem incoming_minor (width freeWidth : Nat) (bad : Bool) :
    get_rule.incoming minor (graph width freeWidth bad) = [] := by
  rw [incoming_eq_filter]
  apply List.filter_eq_nil_iff.mpr
  intro e he
  simp [graph, premiseEdges, outputEdges, outputEdge, independentEdges,
    independentEdge, minor, major, elim, independent, elimRoot,
    independentRoot] at he
  rcases he with rfl | rfl | ⟨i, _hi, rfl⟩ | ⟨i, _hi, rfl⟩ <;>
    simp [minor, elim, elimRoot, independentRoot]

theorem incoming_major (width freeWidth : Nat) (bad : Bool) :
    get_rule.incoming major (graph width freeWidth bad) = [] := by
  rw [incoming_eq_filter]
  apply List.filter_eq_nil_iff.mpr
  intro e he
  simp [graph, premiseEdges, outputEdges, outputEdge, independentEdges,
    independentEdge, minor, major, elim, independent, elimRoot,
    independentRoot] at he
  rcases he with rfl | rfl | ⟨i, _hi, rfl⟩ | ⟨i, _hi, rfl⟩ <;>
    simp [major, elim, elimRoot, independentRoot]

theorem incoming_independent (width freeWidth : Nat) (bad : Bool) :
    get_rule.incoming independent (graph width freeWidth bad) = [] := by
  rw [incoming_eq_filter]
  apply List.filter_eq_nil_iff.mpr
  intro e he
  simp [graph, premiseEdges, outputEdges, outputEdge, independentEdges,
    independentEdge, minor, major, elim, independent, elimRoot,
    independentRoot] at he
  rcases he with rfl | rfl | ⟨i, _hi, rfl⟩ | ⟨i, _hi, rfl⟩ <;>
    simp [independent, elim, elimRoot, independentRoot]

theorem predsOf_elim (width freeWidth : Nat) (bad : Bool) :
    predsOf (graph width freeWidth bad) elim = [minor, major] := by
  rw [predsOf_eq, incoming_elim]
  change List.eraseDups [minor, major] = [minor, major]
  have hne : major ≠ minor := by
    intro h
    have := congrArg Vertex.NUMBER h
    simp [major, minor] at this
  have hbeq : (major == minor) = false := beq_eq_false_iff_ne.mpr hne
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop, hbeq]

def reading (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    ReadingData (fanArities (graph width freeWidth bad)) :=
  (fanArities_graph width freeWidth bad).symm ▸
    .cons outputChoice (.cons freeChoice .nil)

theorem reading_indices (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    (reading width freeWidth bad outputChoice freeChoice).indices =
      [outputChoice.val, freeChoice.val] := by
  simp [reading, Readings.Selection.ReadingData.indices]

theorem exists_reading_choices (width freeWidth : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width freeWidth bad))) :
    ∃ outputChoice : Fin (width + 2),
      ∃ freeChoice : Fin (freeWidth + 2),
        r = reading width freeWidth bad outputChoice freeChoice := by
  let h := fanArities_graph width freeWidth bad
  let r' : ReadingData [width + 2, freeWidth + 2] := h ▸ r
  have hdecomp :
      ∃ outputChoice : Fin (width + 2),
        ∃ freeChoice : Fin (freeWidth + 2),
          r' = .cons outputChoice (.cons freeChoice .nil) := by
    cases r' with
    | cons outputChoice rest =>
        cases rest with
        | cons freeChoice rest' =>
            cases rest'
            exact ⟨outputChoice, freeChoice, rfl⟩
  obtain ⟨outputChoice, freeChoice, hr'⟩ := hdecomp
  refine ⟨outputChoice, freeChoice, ?_⟩
  calc
    r = h.symm ▸ r' := (readingData_cast_symm_cast h r).symm
    _ = h.symm ▸ (.cons outputChoice (.cons freeChoice .nil)) := by rw [hr']
    _ = reading width freeWidth bad outputChoice freeChoice := by
      simp [reading, h]

theorem elimPairsAt_minor (width freeWidth : Nat) (bad : Bool) :
    elimPairsAt (graph width freeWidth bad) minor = [] := by
  simp [elimPairsAt, predsOf, incoming_eq_filter, graph, premiseEdges,
    outputEdges, outputEdge, independentEdges, independentEdge, minor, major,
    elim, independent, elimRoot, independentRoot]

theorem elimPairsAt_major (width freeWidth : Nat) (bad : Bool) :
    elimPairsAt (graph width freeWidth bad) major = [] := by
  simp [elimPairsAt, predsOf, incoming_eq_filter, graph, premiseEdges,
    outputEdges, outputEdge, independentEdges, independentEdge, minor, major,
    elim, independent, elimRoot, independentRoot]

theorem elimPairsAt_elim (width freeWidth : Nat) (bad : Bool) :
    elimPairsAt (graph width freeWidth bad) elim = [(major, minor)] := by
  unfold elimPairsAt
  rw [predsOf_elim]
  simp [minor, major, elim, A, B, AB]

theorem elimPairsAt_independent (width freeWidth : Nat) (bad : Bool) :
    elimPairsAt (graph width freeWidth bad) independent = [] := by
  simp [elimPairsAt, predsOf, incoming_eq_filter, graph, premiseEdges,
    outputEdges, outputEdge, independentEdges, independentEdge, minor, major,
    elim, independent, elimRoot, independentRoot]

theorem elimPairsAt_elimRoot (width freeWidth : Nat) (bad : Bool) :
    elimPairsAt (graph width freeWidth bad) elimRoot = [] := by
  simp [elimPairsAt, predsOf, incoming_eq_filter, graph, premiseEdges,
    outputEdges, outputEdge, independentEdges, independentEdge, minor, major,
    elim, independent, elimRoot, independentRoot, B, R]

theorem elimPairsAt_independentRoot (width freeWidth : Nat) (bad : Bool) :
    elimPairsAt (graph width freeWidth bad) independentRoot = [] := by
  simp [elimPairsAt, predsOf, incoming_eq_filter, graph, premiseEdges,
    outputEdges, outputEdge, independentEdges, independentEdge, minor, major,
    elim, independent, elimRoot, independentRoot, L, R]

/-- Coherence does not require computing reachability when every possible
elimination pair is already fed by the selected edges. -/
theorem coherentB_of_all_elim_pairs_fed (d : Graph)
    (r : ReadingData (fanArities d))
    (hfed : ∀ w ∈ d.NODES, ∀ p q,
      elimPairsAt d w = [(p, q)] →
        feedsB d r p w = true ∧ feedsB d r q w = true) :
    coherentB d r = true := by
  apply List.all_eq_true.mpr
  intro w hw
  unfold coherentAtB
  split
  · rfl
  · cases hpairs : elimPairsAt d w with
    | nil => rfl
    | cons pair rest =>
        cases rest with
        | nil =>
            obtain ⟨p, q⟩ := pair
            obtain ⟨hp, hq⟩ := hfed w hw p q hpairs
            simp [hp, hq]
        | cons pair' rest' => rfl

theorem coherent_reading (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    coherentB (graph width freeWidth bad)
      (reading width freeWidth bad outputChoice freeChoice) = true := by
  apply coherentB_of_all_elim_pairs_fed
  intro w hw p q hpairs
  simp [graph] at hw
  rcases hw with rfl | rfl | rfl | rfl | rfl | rfl
  · rw [elimPairsAt_minor] at hpairs
    simp at hpairs
  · rw [elimPairsAt_major] at hpairs
    simp at hpairs
  · rw [elimPairsAt_elim] at hpairs
    simp at hpairs
    rcases hpairs with ⟨rfl, rfl⟩
    constructor <;>
      simp [feedsB, selectedEdge, outgoing_minor, outgoing_major,
        reading, fanArities_graph, branchPoints_graph]
  · rw [elimPairsAt_independent] at hpairs
    simp at hpairs
  · rw [elimPairsAt_elimRoot] at hpairs
    simp at hpairs
  · rw [elimPairsAt_independentRoot] at hpairs
    simp at hpairs

/-- Every inhabitant of the actual selector type is coherent.  In particular,
the second fan is a genuinely free global coordinate rather than a coherence
assumption hidden in the persistence statement. -/
theorem coherent_all (width freeWidth : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width freeWidth bad))) :
    coherentB (graph width freeWidth bad) r = true := by
  apply coherentB_of_all_elim_pairs_fed
  intro w hw p q hpairs
  simp [graph] at hw
  rcases hw with rfl | rfl | rfl | rfl | rfl | rfl
  · rw [elimPairsAt_minor] at hpairs
    simp at hpairs
  · rw [elimPairsAt_major] at hpairs
    simp at hpairs
  · rw [elimPairsAt_elim] at hpairs
    simp at hpairs
    rcases hpairs with ⟨rfl, rfl⟩
    constructor <;>
      simp [feedsB, selectedEdge, outgoing_minor, outgoing_major]
  · rw [elimPairsAt_independent] at hpairs
    simp at hpairs
  · rw [elimPairsAt_elimRoot] at hpairs
    simp at hpairs
  · rw [elimPairsAt_independentRoot] at hpairs
    simp at hpairs

theorem selectedReachable_go_preserves_seen (d : Graph)
    (r : ReadingData (fanArities d)) (v : Vertex) :
    ∀ fuel frontier seen,
      v ∈ seen → v ∈ selectedReachable.go d r fuel frontier seen := by
  intro fuel
  induction fuel with
  | zero =>
      intro frontier seen hv
      simpa [selectedReachable.go] using List.mem_eraseDups.mpr hv
  | succ fuel ih =>
      intro frontier seen hv
      unfold selectedReachable.go
      dsimp only
      split
      · exact List.mem_eraseDups.mpr hv
      · apply ih
        exact List.mem_append_left _ hv

theorem minor_mem_sourceNodes (width freeWidth : Nat) (bad : Bool) :
    minor ∈ sourceNodes (graph width freeWidth bad) := by
  apply List.mem_filter.mpr
  constructor
  · simp [graph]
  · rw [predsOf_eq, incoming_minor]
    rfl

theorem major_mem_sourceNodes (width freeWidth : Nat) (bad : Bool) :
    major ∈ sourceNodes (graph width freeWidth bad) := by
  apply List.mem_filter.mpr
  constructor
  · simp [graph]
  · rw [predsOf_eq, incoming_major]
    rfl

theorem independent_mem_sourceNodes (width freeWidth : Nat) (bad : Bool) :
    independent ∈ sourceNodes (graph width freeWidth bad) := by
  apply List.mem_filter.mpr
  constructor
  · simp [graph]
  · rw [predsOf_eq, incoming_independent]
    rfl

theorem source_mem_selectedReachable (d : Graph)
    (r : ReadingData (fanArities d)) (v : Vertex)
    (hv : v ∈ sourceNodes d) : v ∈ selectedReachable d r := by
  let sources := sourceNodes d
  let next :=
    (sources.filterMap (selectedEdge d r)
      |>.map (·.END)
      |>.filter (fun w => !(sources.contains w))
      |>.eraseDups)
  unfold selectedReachable
  change
    v ∈
      (if next.isEmpty then sources.eraseDups
       else selectedReachable.go d r d.NODES.length next (sources ++ next))
  split
  · exact List.mem_eraseDups.mpr hv
  · apply selectedReachable_go_preserves_seen
    exact List.mem_append_left next hv

theorem elim_not_mem_sourceNodes (width freeWidth : Nat) (bad : Bool) :
    elim ∉ sourceNodes (graph width freeWidth bad) := by
  intro h
  have hempty := (List.mem_filter.mp h).2
  rw [predsOf_elim] at hempty
  simp at hempty

theorem elim_mem_selectedReachable (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    elim ∈ selectedReachable (graph width freeWidth bad)
      (reading width freeWidth bad outputChoice freeChoice) := by
  let d := graph width freeWidth bad
  let r := reading width freeWidth bad outputChoice freeChoice
  let sources := sourceNodes d
  let next :=
    (sources.filterMap (selectedEdge d r)
      |>.map (·.END)
      |>.filter (fun v => !(sources.contains v))
      |>.eraseDups)
  have hselected :
      selectedEdge d r minor = some (edge minor elim 0 [A]) := by
    simp [d, selectedEdge, outgoing_minor]
  have hnext : elim ∈ next := by
    apply List.mem_eraseDups.mpr
    apply List.mem_filter.mpr
    constructor
    · apply List.mem_map.mpr
      refine ⟨edge minor elim 0 [A], ?_, rfl⟩
      apply List.mem_filterMap.mpr
      exact ⟨minor, minor_mem_sourceNodes width freeWidth bad, hselected⟩
    · simp [d, sources, elim_not_mem_sourceNodes width freeWidth bad]
  have hnextNonempty : next.isEmpty = false := by
    cases hnextList : next with
    | nil => simp [hnextList] at hnext
    | cons head tail => rfl
  unfold selectedReachable
  change
    elim ∈
      (if next.isEmpty then sources.eraseDups
       else selectedReachable.go d r d.NODES.length next (sources ++ next))
  rw [hnextNonempty]
  apply selectedReachable_go_preserves_seen
  exact List.mem_append_right sources hnext

/-- Reachability of the elimination is independent of both fan selectors:
the deterministic minor-premise edge puts it into the first frontier. -/
theorem elim_mem_selectedReachable_all (width freeWidth : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width freeWidth bad))) :
    elim ∈ selectedReachable (graph width freeWidth bad) r := by
  let d := graph width freeWidth bad
  let sources := sourceNodes d
  let next :=
    (sources.filterMap (selectedEdge d r)
      |>.map (·.END)
      |>.filter (fun v => !(sources.contains v))
      |>.eraseDups)
  have hselected :
      selectedEdge d r minor = some (edge minor elim 0 [A]) := by
    simp [d, selectedEdge, outgoing_minor]
  have hnext : elim ∈ next := by
    apply List.mem_eraseDups.mpr
    apply List.mem_filter.mpr
    constructor
    · apply List.mem_map.mpr
      refine ⟨edge minor elim 0 [A], ?_, rfl⟩
      apply List.mem_filterMap.mpr
      exact ⟨minor, minor_mem_sourceNodes width freeWidth bad, hselected⟩
    · simp [d, sources, elim_not_mem_sourceNodes width freeWidth bad]
  have hnextNonempty : next.isEmpty = false := by
    cases hnextList : next with
    | nil => simp [hnextList] at hnext
    | cons head tail => rfl
  unfold selectedReachable
  change
    elim ∈
      (if next.isEmpty then sources.eraseDups
       else selectedReachable.go d r d.NODES.length next (sources ++ next))
  rw [hnextNonempty]
  apply selectedReachable_go_preserves_seen
  exact List.mem_append_right sources hnext

theorem selectedRuleActiveB_elim_all (width freeWidth : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width freeWidth bad))) :
    selectedRuleActiveB (graph width freeWidth bad) r elim = true := by
  have hcoherent := coherent_all width freeWidth bad r
  have hlocal := List.all_eq_true.mp hcoherent elim (by simp [graph])
  simp [selectedRuleActiveB,
    elim_mem_selectedReachable_all width freeWidth bad r, hlocal]

theorem branchIndex_elim (width freeWidth : Nat) (bad : Bool) :
    branchIndex (graph width freeWidth bad) elim = 0 := by
  simp [branchIndex, branchPoints_graph]

theorem selectedEdge_reading_elim (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    selectedEdge (graph width freeWidth bad)
      (reading width freeWidth bad outputChoice freeChoice) elim =
      some (outputEdge bad outputChoice.val) := by
  simp [selectedEdge, outgoing_elim, outputEdges, reading_indices,
    branchIndex_elim]

theorem selectedRuleActiveB_reading_elim
    (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    selectedRuleActiveB (graph width freeWidth bad)
      (reading width freeWidth bad outputChoice freeChoice) elim = true := by
  have hcoherent := coherent_reading width freeWidth bad outputChoice freeChoice
  have hlocal := List.all_eq_true.mp hcoherent elim (by simp [graph])
  simp [selectedRuleActiveB,
    elim_mem_selectedReachable width freeWidth bad outputChoice freeChoice,
    hlocal]

/-- Every output of the variable elimination fan has a structurally
constructed coherent reading.  The free fan coordinate is chosen as zero,
but no label-correctness fact is used. -/
theorem elim_outputs_covered (width freeWidth : Nat) (bad : Bool) :
    ∀ e ∈ outgoing (graph width freeWidth bad) elim,
      ExplicitOutputCovered (graph width freeWidth bad) elim e := by
  intro e he
  rw [outgoing_elim] at he
  obtain ⟨i, hi, rfl⟩ := List.mem_map.mp he
  have hi' : i < width + 2 := by simpa using hi
  let outputChoice : Fin (width + 2) := ⟨i, hi'⟩
  let freeChoice : Fin (freeWidth + 2) := ⟨0, by omega⟩
  refine ⟨reading width freeWidth bad outputChoice freeChoice,
    coherent_reading width freeWidth bad outputChoice freeChoice,
    selectedRuleActiveB_reading_elim width freeWidth bad outputChoice freeChoice,
    ?_⟩
  simpa [outputChoice] using
    selectedEdge_reading_elim width freeWidth bad outputChoice freeChoice

theorem branchIndex_independent (width freeWidth : Nat) (bad : Bool) :
    branchIndex (graph width freeWidth bad) independent = 1 := by
  simp [branchIndex, branchPoints_graph, elim, independent]

theorem selectedEdge_reading_independent
    (width freeWidth : Nat) (bad : Bool)
    (outputChoice : Fin (width + 2))
    (freeChoice : Fin (freeWidth + 2)) :
    selectedEdge (graph width freeWidth bad)
      (reading width freeWidth bad outputChoice freeChoice) independent =
      some (independentEdge freeChoice.val) := by
  simp [selectedEdge, outgoing_independent, independentEdges,
    reading_indices, branchIndex_independent]

theorem selectedRuleActiveB_source_all
    (width freeWidth : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width freeWidth bad)))
    (v : Vertex) (hsource : v ∈ sourceNodes (graph width freeWidth bad))
    (hv : v ∈ (graph width freeWidth bad).NODES) :
    selectedRuleActiveB (graph width freeWidth bad) r v = true := by
  have hcoherent := coherent_all width freeWidth bad r
  have hlocal := List.all_eq_true.mp hcoherent v hv
  simp [selectedRuleActiveB, source_mem_selectedReachable _ r v hsource,
    hlocal]

theorem independent_outputs_covered
    (width freeWidth : Nat) (bad : Bool) :
    ∀ e ∈ outgoing (graph width freeWidth bad) independent,
      ExplicitOutputCovered (graph width freeWidth bad) independent e := by
  intro e he
  rw [outgoing_independent] at he
  obtain ⟨i, hi, rfl⟩ := List.mem_map.mp he
  have hi' : i < freeWidth + 2 := by simpa using hi
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let freeChoice : Fin (freeWidth + 2) := ⟨i, hi'⟩
  let r := reading width freeWidth bad outputChoice freeChoice
  refine ⟨r, coherent_all width freeWidth bad r, ?_, ?_⟩
  · exact selectedRuleActiveB_source_all width freeWidth bad r independent
      (independent_mem_sourceNodes width freeWidth bad) (by simp [graph])
  · simpa [r, freeChoice] using
      selectedEdge_reading_independent width freeWidth bad outputChoice freeChoice

theorem minor_output_covered (width freeWidth : Nat) (bad : Bool) :
    ∀ e ∈ outgoing (graph width freeWidth bad) minor,
      ExplicitOutputCovered (graph width freeWidth bad) minor e := by
  intro e he
  rw [outgoing_minor] at he
  simp only [List.mem_singleton] at he
  subst e
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let freeChoice : Fin (freeWidth + 2) := ⟨0, by omega⟩
  let r := reading width freeWidth bad outputChoice freeChoice
  refine ⟨r, coherent_all width freeWidth bad r, ?_, ?_⟩
  · exact selectedRuleActiveB_source_all width freeWidth bad r minor
      (minor_mem_sourceNodes width freeWidth bad) (by simp [graph])
  · simp [r, selectedEdge, outgoing_minor]

theorem major_output_covered (width freeWidth : Nat) (bad : Bool) :
    ∀ e ∈ outgoing (graph width freeWidth bad) major,
      ExplicitOutputCovered (graph width freeWidth bad) major e := by
  intro e he
  rw [outgoing_major] at he
  simp only [List.mem_singleton] at he
  subst e
  let outputChoice : Fin (width + 2) := ⟨0, by omega⟩
  let freeChoice : Fin (freeWidth + 2) := ⟨0, by omega⟩
  let r := reading width freeWidth bad outputChoice freeChoice
  refine ⟨r, coherent_all width freeWidth bad r, ?_, ?_⟩
  · exact selectedRuleActiveB_source_all width freeWidth bad r major
      (major_mem_sourceNodes width freeWidth bad) (by simp [graph])
  · simp [r, selectedEdge, outgoing_major]

/-- Coverage is global for the declarative property: it includes both premise
edges, every variable elimination output and every independent-fan output.
The two terminal vertices contribute no obligations. -/
theorem allExplicitOutputsCovered_graph
    (width freeWidth : Nat) (bad : Bool) :
    AllExplicitOutputsCovered (graph width freeWidth bad) := by
  intro v hv e he
  simp [graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl | rfl
  · exact minor_output_covered width freeWidth bad e he
  · exact major_output_covered width freeWidth bad e he
  · exact elim_outputs_covered width freeWidth bad e he
  · exact independent_outputs_covered width freeWidth bad e he
  · rw [outgoing_elimRoot] at he
    simp at he
  · rw [outgoing_independentRoot] at he
    simp at he

theorem elim_rule_shape (width freeWidth : Nat) (bad : Bool) :
    ruleShapeOKB elim (graph width freeWidth bad) = true := by
  unfold ruleShapeOKB classifyRule?
  rw [incoming_elim]
  simp [premiseEdges, elim, minor, major, A, B, AB]

theorem elim_outDep (width freeWidth : Nat) (bad : Bool) :
    outDep elim (graph width freeWidth bad) = goodDep := by
  unfold outDep classifyRule?
  rw [incoming_elim]
  simp [premiseEdges, goodDep, elim, minor, major, A, B, AB]

theorem minor_rule (width freeWidth : Nat) (bad : Bool) :
    ruleShapeOKB minor (graph width freeWidth bad) = true ∧
      outDep minor (graph width freeWidth bad) = [A] := by
  simp [ruleShapeOKB, outDep, classifyRule?, minor]

theorem major_rule (width freeWidth : Nat) (bad : Bool) :
    ruleShapeOKB major (graph width freeWidth bad) = true ∧
      outDep major (graph width freeWidth bad) = [AB] := by
  simp [ruleShapeOKB, outDep, classifyRule?, major]

theorem independent_rule (width freeWidth : Nat) (bad : Bool) :
    ruleShapeOKB independent (graph width freeWidth bad) = true ∧
      outDep independent (graph width freeWidth bad) = [L] := by
  simp [ruleShapeOKB, outDep, classifyRule?, independent]

theorem control_explicitOutputLabelsCorrect (width freeWidth : Nat) :
    ExplicitOutputLabelsCorrect (control width freeWidth) := by
  intro v hv e he
  simp [control, graph] at hv
  rcases hv with rfl | rfl | rfl | rfl | rfl | rfl
  · rw [outgoing_minor] at he
    simp only [List.mem_singleton] at he
    subst e
    simpa [minor_rule]
  · rw [outgoing_major] at he
    simp only [List.mem_singleton] at he
    subst e
    simpa [major_rule]
  · rw [outgoing_elim] at he
    obtain ⟨i, hi, rfl⟩ := List.mem_map.mp he
    constructor
    · exact elim_rule_shape width freeWidth false
    · rw [elim_outDep]
      simp [outputEdge]
  · rw [outgoing_independent] at he
    obtain ⟨i, hi, rfl⟩ := List.mem_map.mp he
    simpa [independentEdge, independent_rule]
  · rw [outgoing_elimRoot] at he
    simp at he
  · rw [outgoing_independentRoot] at he
    simp at he

theorem control_no_labelDefect (width freeWidth : Nat) :
    ∀ r : ReadingData (fanArities (control width freeWidth)),
      labelDefectB (control width freeWidth) r = false :=
  (explicitOutputLabelsCorrect_iff_no_labelDefect _
    (allExplicitOutputsCovered_graph width freeWidth false)).mp
      (control_explicitOutputLabelsCorrect width freeWidth)

theorem bad_dependency_ne_goodDep : [Z] ≠ goodDep := by
  intro h
  have hmem : A ∈ goodDep := by
    simp [goodDep]
  rw [← h] at hmem
  simpa [A, Z] using hmem

theorem bad_output_mem (width freeWidth : Nat) :
    outputEdge true 1 ∈ outgoing (defective width freeWidth) elim := by
  rw [outgoing_elim]
  apply List.mem_map.mpr
  refine ⟨1, ?_, rfl⟩
  simp [outputEdges]

theorem bad_output_violation (width freeWidth : Nat) :
    ExplicitOutputLabelViolation (defective width freeWidth) elim
      (outputEdge true 1) := by
  refine ⟨by simp [defective, graph], bad_output_mem width freeWidth, ?_⟩
  right
  rw [elim_outDep]
  simpa [outputEdge] using bad_dependency_ne_goodDep

theorem defective_not_explicitOutputLabelsCorrect (width freeWidth : Nat) :
    ¬ ExplicitOutputLabelsCorrect (defective width freeWidth) := by
  intro hcorrect
  have hviolation := bad_output_violation width freeWidth
  have hdep :=
    (hcorrect elim hviolation.1 (outputEdge true 1) hviolation.2.1).2
  rw [elim_outDep] at hdep
  exact bad_dependency_ne_goodDep (by simpa [outputEdge] using hdep)

def badReference (width freeWidth : Nat) :
    ReadingData (fanArities (defective width freeWidth)) :=
  reading width freeWidth true ⟨1, by omega⟩ ⟨0, by omega⟩

theorem badReference_indices (width freeWidth : Nat) :
    (badReference width freeWidth).indices = [1, 0] := by
  simp [badReference, reading_indices]

theorem badReference_mismatch (width freeWidth : Nat) :
    selectedRuleApplicationMismatchB (defective width freeWidth)
      (badReference width freeWidth) elim = true := by
  unfold selectedRuleApplicationMismatchB selectedDecorationMismatchB
  simp [badReference, selectedRuleActiveB_reading_elim,
    selectedEdge_reading_elim, outputEdge, elim_outDep,
    bad_dependency_ne_goodDep]

def badSupport : Finset Nat := {0}

def AgreesOnBadSupport {d : Graph}
    (x y : ReadingData (fanArities d)) : Prop :=
  (x.indices[0]?).getD 0 = (y.indices[0]?).getD 0

theorem badSupport_card : badSupport.card = 1 := by
  simp [badSupport]

/-- The real support is the singleton selector coordinate `0`, the fan of the
elimination.  No per-extension coherence or activation premise remains. -/
theorem bad_coordinate_forces_labelDefect
    (width freeWidth : Nat)
    (y : ReadingData (fanArities (defective width freeWidth)))
    (hcoordinate : (y.indices[0]?).getD 0 = 1) :
    labelDefectB (defective width freeWidth) y = true := by
  apply labelDefectB_persists_of_coherent_active
    (defective width freeWidth) elim (by simp [defective, graph])
    (x := badReference width freeWidth)
  · exact badReference_mismatch width freeWidth
  · simpa [branchIndex_elim, badReference_indices] using hcoordinate.symm
  · exact coherent_all width freeWidth true y
  · exact selectedRuleActiveB_elim_all width freeWidth true y

theorem agree_badSupport_forces_labelDefect
    (width freeWidth : Nat)
    (y : ReadingData (fanArities (defective width freeWidth)))
    (hagrees : AgreesOnBadSupport (badReference width freeWidth) y) :
    labelDefectB (defective width freeWidth) y = true := by
  apply bad_coordinate_forces_labelDefect
  simpa [AgreesOnBadSupport, badReference_indices] using hagrees.symm

theorem badReference_marks (width freeWidth : Nat) :
    labelDefectB (defective width freeWidth)
      (badReference width freeWidth) = true := by
  apply bad_coordinate_forces_labelDefect
  simp [badReference_indices]

theorem bad_coordinate_forces_rejection
    (width freeWidth : Nat)
    (y : ReadingData (fanArities (defective width freeWidth)))
    (hcoordinate : (y.indices[0]?).getD 0 = 1) :
    rejectsReadingLabelsB (defective width freeWidth) y = true :=
  labelDefectB_implies_rejectsReadingLabelsB _ _
    (bad_coordinate_forces_labelDefect width freeWidth y hcoordinate)

theorem readings_complete (width freeWidth : Nat) (bad : Bool)
    (r : ReadingData (fanArities (graph width freeWidth bad))) :
    r ∈ readings (graph width freeWidth bad) := by
  exact mem_enumerateReadingData r

theorem readings_nodup (width freeWidth : Nat) (bad : Bool) :
    (readings (graph width freeWidth bad)).Nodup := by
  exact enumerateReadingData_nodup _

theorem raw_reading_count (width freeWidth : Nat) (bad : Bool) :
    (readings (graph width freeWidth bad)).length =
      (width + 2) * (freeWidth + 2) := by
  rw [readings, enumerateReadingData_length, fanArities_graph]
  simp

theorem coherentReadings_eq_readings
    (width freeWidth : Nat) (bad : Bool) :
    coherentReadings (graph width freeWidth bad) =
      readings (graph width freeWidth bad) := by
  apply List.filter_eq_self.mpr
  intro r hr
  exact coherent_all width freeWidth bad r

theorem coherent_reading_count (width freeWidth : Nat) (bad : Bool) :
    (coherentReadings (graph width freeWidth bad)).length =
      (width + 2) * (freeWidth + 2) := by
  rw [coherentReadings_eq_readings, raw_reading_count]

/-- The raw selector cylinder fixing precisely coordinate zero to the defective
output. -/
def badCylinder (width freeWidth : Nat) :
    List (ReadingData (fanArities (defective width freeWidth))) :=
  (List.finRange (freeWidth + 2)).map fun freeChoice =>
    reading width freeWidth true ⟨1, by omega⟩ freeChoice

theorem badCylinder_count (width freeWidth : Nat) :
    (badCylinder width freeWidth).length = freeWidth + 2 := by
  simp [badCylinder]

theorem badCylinder_subset_readings (width freeWidth : Nat) :
    ∀ r ∈ badCylinder width freeWidth,
      r ∈ readings (defective width freeWidth) := by
  intro r hr
  exact readings_complete width freeWidth true r

theorem badCylinder_nodup (width freeWidth : Nat) :
    (badCylinder width freeWidth).Nodup := by
  apply (List.nodup_finRange (freeWidth + 2)).map
  intro left right h
  have hindices := congrArg Readings.Selection.ReadingData.indices h
  simp [reading_indices] at hindices
  exact Fin.ext hindices

theorem mem_badCylinder_iff (width freeWidth : Nat)
    (r : ReadingData (fanArities (defective width freeWidth))) :
    r ∈ badCylinder width freeWidth ↔
      (r.indices[0]?).getD 0 = 1 := by
  constructor
  · intro hr
    obtain ⟨freeChoice, _hfree, rfl⟩ := List.mem_map.mp hr
    simp [reading_indices]
  · intro hcoord
    obtain ⟨outputChoice, freeChoice, rfl⟩ :=
      exists_reading_choices width freeWidth true r
    have hval : outputChoice.val = 1 := by
      simpa [reading_indices] using hcoord
    have hout : outputChoice = (⟨1, by omega⟩ : Fin (width + 2)) :=
      Fin.ext hval
    rw [hout]
    apply List.mem_map.mpr
    exact ⟨freeChoice, by simp, rfl⟩

theorem every_badCylinder_reading_marks
    (width freeWidth : Nat)
    {r : ReadingData (fanArities (defective width freeWidth))}
    (hr : r ∈ badCylinder width freeWidth) :
    labelDefectB (defective width freeWidth) r = true := by
  obtain ⟨freeChoice, _hfree, rfl⟩ := List.mem_map.mp hr
  apply bad_coordinate_forces_labelDefect
  simp [reading_indices]

theorem node_count (width freeWidth : Nat) (bad : Bool) :
    (graph width freeWidth bad).NODES.length = 6 := by
  rfl

theorem edge_count (width freeWidth : Nat) (bad : Bool) :
    (graph width freeWidth bad).EDGES.length = width + freeWidth + 6 := by
  simp [graph, premiseEdges, outputEdges, independentEdges]
  omega

theorem fan_count (width freeWidth : Nat) (bad : Bool) :
    (branchPoints (graph width freeWidth bad)).length = 2 := by
  rw [branchPoints_graph]
  rfl

theorem correctness_iff_no_marks
    (width freeWidth : Nat) (bad : Bool) :
    ExplicitOutputLabelsCorrect (graph width freeWidth bad) ↔
      ∀ r : ReadingData (fanArities (graph width freeWidth bad)),
        labelDefectB (graph width freeWidth bad) r = false :=
  explicitOutputLabelsCorrect_iff_no_labelDefect _
    (allExplicitOutputsCovered_graph width freeWidth bad)

/-- Exact mass of the constructed raw cylinder under the uniform distribution
on the finite selector product.  This is a lower bound on the total marked
mass; no claim is made here that there are no additional marked readings. -/
theorem badCylinder_raw_density (width freeWidth : Nat) :
    ((badCylinder width freeWidth).length : ℚ) /
        ((readings (defective width freeWidth)).length : ℚ) =
      1 / (width + 2 : ℚ) := by
  rw [badCylinder_count, raw_reading_count]
  push_cast
  have hw : (width : ℚ) + 2 ≠ 0 := by positivity
  have hf : (freeWidth : ℚ) + 2 ≠ 0 := by positivity
  field_simp

/-- Conditioning on coherence costs nothing in this family because every raw
reading is coherent. -/
theorem badCylinder_coherent_density (width freeWidth : Nat) :
    ((badCylinder width freeWidth).length : ℚ) /
        ((coherentReadings (defective width freeWidth)).length : ℚ) =
      1 / (width + 2 : ℚ) := by
  rw [coherentReadings_eq_readings]
  exact badCylinder_raw_density width freeWidth


end UniformElimFanFamily
end Readings.Explicit
end DLDS
