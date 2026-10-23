import Semantic.Circuit.SharedProduction
import Semantic.MultiIntroPositions
import Mathlib.Data.List.ProdSigma

open FlowSpec
namespace Semantic.Circuit





def graphListLengths (d : Graph) : List Nat :=
  [d.NODES.length, d.EDGES.length, d.PATHS.length,
   (d.EDGES.flatMap Deduction.DEPENDENCY).length,
   (d.PATHS.flatMap Ancestral.COLOURS).length,
   ((d.NODES ++ d.EDGES.flatMap (fun e => [e.START, e.END]) ++
     d.PATHS.flatMap (fun a => [a.START, a.END])).flatMap Vertex.PAST).length,
   (d.EDGES.flatMap fun _ => d.EDGES).length,
   (RefinedEval.formulas d).length,
   (MultiIntroPositions.positions d).length]



def graphSize (d : Graph) : Nat := 1 + (graphListLengths d).sum

theorem graphSize_positive (d : Graph) : 0 < graphSize d := by
  simp [graphSize]

theorem graph_list_le (d : Graph) (n : Nat) (member : n ∈ graphListLengths d) :
    n + 1 ≤ graphSize d := by
  have h : n ≤ (graphListLengths d).sum := List.single_le_sum (fun _ _ => Nat.zero_le _) n member
  simp only [graphSize]
  omega

theorem graph_nodes_le (d : Graph) : d.NODES.length + 1 ≤ graphSize d :=
  graph_list_le d _ (by simp [graphListLengths])

theorem graph_edges_le (d : Graph) : d.EDGES.length + 1 ≤ graphSize d :=
  graph_list_le d _ (by simp [graphListLengths])

theorem graph_paths_le (d : Graph) : d.PATHS.length + 1 ≤ graphSize d :=
  graph_list_le d _ (by simp [graphListLengths])

theorem graph_pairs_le (d : Graph) : d.EDGES.length * d.EDGES.length + 1 ≤ graphSize d := by
  have h := graph_list_le d (d.EDGES.flatMap fun _ => d.EDGES).length (by simp [graphListLengths])
  simpa [List.length_flatMap, Nat.mul_comm] using h

theorem graph_positions_le (d : Graph) : (MultiIntroPositions.positions d).length + 1 ≤ graphSize d :=
  graph_list_le d _ (by simp [graphListLengths])

theorem graph_formulas_le (d : Graph) : (RefinedEval.formulas d).length + 1 ≤ graphSize d :=
  graph_list_le d _ (by simp [graphListLengths])

theorem list_member_length_le_flatMap {α β : Type} (xs : List α) (f : α → List β)
    (x : α) (member : x ∈ xs) : (f x).length ≤ (xs.flatMap f).length := by
  induction xs with
  | nil => simp at member
  | cons head rest ih =>
    simp only [List.mem_cons] at member
    rcases member with rfl | member
    · simp
    · exact le_trans (ih member) (by simp)

theorem graph_label_le (d : Graph) (e : Deduction) (member : e ∈ d.EDGES) :
    e.DEPENDENCY.length + 1 ≤ graphSize d := by
  have h := list_member_length_le_flatMap d.EDGES Deduction.DEPENDENCY e member
  have t := graph_list_le d (d.EDGES.flatMap Deduction.DEPENDENCY).length (by simp [graphListLengths])
  omega

theorem graph_word_le (d : Graph) (a : Ancestral) (member : a ∈ d.PATHS) :
    a.COLOURS.length + 1 ≤ graphSize d := by
  have h := list_member_length_le_flatMap d.PATHS Ancestral.COLOURS a member
  have t := graph_list_le d (d.PATHS.flatMap Ancestral.COLOURS).length (by simp [graphListLengths])
  omega

theorem graph_suffix_table_le (d : Graph) : ([] :: graphWords d).length ≤ graphSize d := by
  have h := graph_list_le d (d.PATHS.flatMap Ancestral.COLOURS).length (by simp [graphListLengths])
  simpa [graph_words_length, FlowSize.wordTotal, List.length_flatMap, Nat.add_comm] using h

theorem eraseDups_length_le {α : Type} [BEq α] (xs : List α) : xs.eraseDups.length ≤ xs.length := by
  cases xs with
  | nil => simp
  | cons head tail =>
    rw [List.eraseDups_cons]
    have recursive := eraseDups_length_le (tail.filter (fun x => !(x == head)))
    have filtered := List.length_filter_le (l := tail) (fun x => !(x == head))
    simpa only [List.length_cons] using Nat.add_le_add_right (recursive.trans filtered) 1
termination_by xs.length
decreasing_by have h := List.length_filter_le (l := tail) (fun x => !(x == head)); simp; omega

theorem predecessors_length_le_edges (d : Graph) (v : Vertex) :
    (predsOf d v).length ≤ d.EDGES.length := by
  unfold predsOf
  exact le_trans (eraseDups_length_le _) (by simpa using List.length_filter_le (l := d.EDGES) (fun e => e.END = v))

theorem graph_predecessors_le (d : Graph) (v : Vertex) : (predsOf d v).length + 1 ≤ graphSize d := by
  have h := predecessors_length_le_edges d v
  have t := graph_edges_le d
  omega

theorem graph_reads_le (d : Graph) (v : Vertex) : (productionReads d v).length ≤ graphSize d := by
  simpa [productionReads] using graph_predecessors_le d v

theorem graph_between_le (d : Graph) (u v : Vertex) : (edgesBetween d u v).length + 1 ≤ graphSize d := by
  have h : (edgesBetween d u v).length ≤ d.EDGES.length := List.length_filter_le ..
  have t := graph_edges_le d
  omega

theorem graph_ancestors_le (d : Graph) (v : Vertex) : (ancestorsInto d v).length + 1 ≤ graphSize d := by
  have h : (ancestorsInto d v).length ≤ d.PATHS.length := List.length_filter_le ..
  have t := graph_paths_le d
  omega

theorem graph_eligible_le (d : Graph) (v : Vertex) :
    (eligibleAncestorsAtRoot d v).length + 1 ≤ graphSize d := by
  have t := graph_ancestors_le d v
  unfold eligibleAncestorsAtRoot
  split
  · simpa using graphSize_positive d
  · rename_i root _
    dsimp only
    have h := List.length_filter_le (l := ancestorsInto d v) (rootSeedEligibleWithPreB d root (Pre d root) v)
    omega

theorem graph_seeds_le (d : Graph) (v : Vertex) : (eligibleSeeds d v).length + 1 ≤ graphSize d := by
  simpa [eligibleSeeds] using graph_eligible_le d v

theorem graph_source_le (d : Graph) (v : Vertex) : (sourceRoutes d v).length ≤ graphSize d := by
  have t := graph_ancestors_le d v
  have p := graphSize_positive d
  unfold sourceRoutes
  split <;> simp_all <;> omega

theorem sum_map_bound {α : Type} (xs : List α) (f : α → Nat) (bound : Nat)
    (each : ∀ x ∈ xs, f x ≤ bound) : (xs.map f).sum ≤ xs.length * bound := by
  induction xs with
  | nil => simp
  | cons head rest ih =>
    have h := each head (by simp)
    have t := ih (fun x hx => each x (by simp [hx]))
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    nlinarith

theorem elimination_pairs_length_le (d : Graph) (v : Vertex) :
    (elimPairsAt d v).length ≤ (predsOf d v).length * (predsOf d v).length := by
  unfold elimPairsAt
  simp only [List.length_flatMap]
  exact sum_map_bound _ _ _ (fun _ _ => List.length_filterMap_le ..)

theorem graph_elimination_le (d : Graph) (v : Vertex) : (elimPairsAt d v).length + 1 ≤ graphSize d := by
  have h := elimination_pairs_length_le d v
  have p := predecessors_length_le_edges d v
  have t := graph_pairs_le d
  have m := Nat.mul_self_le_mul_self p
  omega

end Semantic.Circuit
