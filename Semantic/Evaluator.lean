import Semantic.Routing

open scoped Classical

namespace Semantic

/-!
# Generic circuit evaluator correctness.
-/

/-- Evaluate entire layer using evaluate_node -/
def evaluate_layer {n : Nat}
  (layer : GridLayer n)
  (tokens : List (Token n))
  : (List (List.Vector Bool n)) × Bool :=

  let results := layer.nodes.zipIdx.map fun (cnode, col_idx) =>
    let tokens_here := tokens.filter (·.current_column = col_idx)
    let node_incoming := layer.incoming[col_idx]!
    evaluate_node cnode node_incoming tokens_here

  let outputs := results.map Prod.fst
  let errors := results.map Prod.snd
  let any_error := errors.any id

  (outputs, any_error)



/-- Recursive circuit evaluation --/
def eval_from_level {n : Nat}
  (paths : PathInput)
  (level : Nat)
  (tokens : List (Token n))
  (remaining_layers : List (GridLayer n))
  (accumulated_error : Bool)
  (num_levels : Nat)
  : (List (List.Vector Bool n)) × Bool :=
  match remaining_layers with
  | [] =>
      let final_outputs := (List.range n).map fun _ => List.Vector.replicate n false
      (final_outputs, accumulated_error)
  | layer :: rest =>
      let (outputs, layer_error) := evaluate_layer layer tokens
      match rest with
      | [] =>
          (outputs, accumulated_error || layer_error)
      | _ =>
          let new_tokens := propagate_tokens tokens paths level num_levels outputs
          eval_from_level paths (level - 1) new_tokens rest (accumulated_error || layer_error) num_levels

/-- Extract the evaluation result -/
def get_eval_result {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (paths : PathInput) : (List (List.Vector Bool n)) × Bool :=
  let num_levels := layers.length
  let initial_tokens := initialize_tokens initial_vectors num_levels
  eval_from_level paths (num_levels - 1) initial_tokens layers false num_levels

/-- Path is structurally invalid iff the circuit evaluation reports an error -/
def PathStructurallyInvalid {n : Nat}
  (paths : PathInput)
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) : Prop :=
  (get_eval_result layers initial_vectors paths).snd = true

/-- The selected path has no routing/XOR error in the circuit evaluator. -/
def PathHasNoRoutingError {n : Nat}
  (paths : PathInput)
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) : Prop :=
  ¬PathStructurallyInvalid paths layers initial_vectors

/-- All assumptions discharged iff goal vector is all false -/
def AllAssumptionsDischarged {n : Nat}
  (paths : PathInput)
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (goal_column : Nat) : Prop :=
  let (final_outputs, _) := get_eval_result layers initial_vectors paths
  goal_column ≥ final_outputs.length ∨
  ∃ (h : goal_column < final_outputs.length),
    ∀ i : Fin n, (final_outputs.get ⟨goal_column, h⟩).get i = false


/-- Main circuit evaluation.
    Returns true iff:
    - A structural error occurred (XOR conflict), OR
    - Goal column is out of bounds, OR
    - Goal vector has all-zero dependencies (all assumptions discharged) -/
def evaluateCircuit {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat) : Bool :=
  let (final_outputs, had_error) := get_eval_result layers initial_vectors paths
  if h : goal_column < final_outputs.length then
    let goal_vector := final_outputs.get ⟨goal_column, h⟩
    let all_discharged := goal_vector.toList.all (· = false)
    had_error || all_discharged
  else
    true

/-- Definitional unfolding of evaluateCircuit. -/
lemma evaluateCircuit_eq {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat) :
    evaluateCircuit layers initial_vectors paths goal_column =
    let (final_outputs, had_error) := get_eval_result layers initial_vectors paths
    if h : goal_column < final_outputs.length then
      let goal_vector := final_outputs.get ⟨goal_column, h⟩
      let all_discharged := goal_vector.toList.all (· = false)
      had_error || all_discharged
    else true := by
  unfold evaluateCircuit get_eval_result
  rfl



/-- Structural error in evaluation implies path is structurally invalid. -/
lemma error_implies_structurally_invalid {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (h_error : (get_eval_result layers initial_vectors paths).snd = true) :
    PathStructurallyInvalid paths layers initial_vectors :=
  h_error

/-- Path structurally invalid implies evaluation produced an error. -/
lemma structurally_invalid_implies_error {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (h_invalid : PathStructurallyInvalid paths layers initial_vectors) :
    (get_eval_result layers initial_vectors paths).snd = true :=
  h_invalid

/-- If no error and goal vector is all-false, then all assumptions are discharged. -/
lemma no_error_accept_implies_discharged {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat)
    (h_accept :
      let (final_outputs, _) := get_eval_result layers initial_vectors paths
      ∃ (h : goal_column < final_outputs.length),
        (final_outputs.get ⟨goal_column, h⟩).toList.all (· = false) = true) :
    AllAssumptionsDischarged paths layers initial_vectors goal_column := by
  unfold AllAssumptionsDischarged
  rcases h_accept with ⟨h_len, h_all⟩
  right
  refine ⟨h_len, ?_⟩
  intro i
  let vec := (get_eval_result layers initial_vectors paths).fst.get ⟨goal_column, h_len⟩
  have h_all_elems := List.all_eq_true.mp h_all
  have h_in : vec.get i ∈ vec.toList := by
    rcases vec with ⟨l, hl⟩
    simp [List.Vector.get, List.Vector.toList]
  have h_decide : decide (vec.get i = false) = true := h_all_elems (vec.get i) h_in
  simp at h_decide
  exact h_decide

/-- Operational circuit soundness: if `evaluateCircuit` accepts, then either
    the path is structurally invalid, or it has no routing error and the goal
    dependency vector is fully discharged. -/
theorem circuit_correctness {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat)
    (h_accept : evaluateCircuit layers initial_vectors paths goal_column = true) :
    PathStructurallyInvalid paths layers initial_vectors
    ∨
    (PathHasNoRoutingError paths layers initial_vectors ∧
     AllAssumptionsDischarged paths layers initial_vectors goal_column) := by
  rw [evaluateCircuit_eq] at h_accept
  cases h_eval : get_eval_result layers initial_vectors paths with
  | mk final_outputs had_error =>
  by_cases h_err : had_error
  · -- Case: had_error = true → structurally invalid
    left
    have h_err_snd : (get_eval_result layers initial_vectors paths).snd = true := by
      rw [h_eval]; exact h_err
    exact error_implies_structurally_invalid layers initial_vectors paths h_err_snd
  · -- Case: had_error = false → no routing error, then require discharge
    right
    have h_err_snd : (get_eval_result layers initial_vectors paths).snd = false := by
      rw [h_eval]; simp [h_err]
    constructor
    · -- PathHasNoRoutingError
      unfold PathHasNoRoutingError
      intro h_invalid
      have : (get_eval_result layers initial_vectors paths).snd = true :=
        structurally_invalid_implies_error layers initial_vectors paths h_invalid
      rw [h_err_snd] at this
      contradiction
    · -- AllAssumptionsDischarged
      rw [h_eval] at h_accept
      simp [h_err] at h_accept
      by_cases h_bounds : goal_column < final_outputs.length
      · -- In bounds: extract that all entries are false
        simp [h_bounds] at h_accept
        apply no_error_accept_implies_discharged layers initial_vectors paths goal_column
        rw [h_eval]
        simp
        use h_bounds
      · -- Out of bounds: vacuously true
        unfold AllAssumptionsDischarged
        rw [h_eval]
        simp
        left
        omega


/-- Operational circuit completeness: a path with no routing error and a fully
    discharged goal vector is accepted. -/

theorem circuit_completeness {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat)
    (h_valid : PathHasNoRoutingError paths layers initial_vectors)
    (h_discharged : AllAssumptionsDischarged paths layers initial_vectors goal_column) :
    evaluateCircuit layers initial_vectors paths goal_column = true := by
  unfold evaluateCircuit
  cases h_eval : get_eval_result layers initial_vectors paths with
  | mk final_outputs had_error =>
  -- Since the path has no routing error, had_error = false.
  have h_no_error : had_error = false := by
    unfold PathHasNoRoutingError PathStructurallyInvalid at h_valid
    rw [h_eval] at h_valid
    simp at h_valid
    exact h_valid
  simp only
  by_cases h_bounds : goal_column < final_outputs.length
  · -- In bounds: show all_discharged = true
    simp only [h_bounds, dite_true, h_no_error, Bool.false_or]
    unfold AllAssumptionsDischarged at h_discharged
    rw [h_eval] at h_discharged
    simp at h_discharged
    cases h_discharged with
    | inl h_ge => omega
    | inr h_all =>
      obtain ⟨h_lt, h_all_false⟩ := h_all
      have : (final_outputs.get ⟨goal_column, h_lt⟩).toList.all (· = false) = true := by
        rw [List.all_eq_true]
        intro b hb
        simp only [decide_eq_true_eq]
        by_contra h_ne
        have hb_true : b = true := by cases b <;> simp_all
        subst hb_true
        -- Get the underlying list
        obtain ⟨l, hl⟩ := final_outputs.get ⟨goal_column, h_lt⟩
        simp [List.Vector.toList] at hb
        rw [List.mem_iff_get] at hb
        obtain ⟨⟨idx, hidx⟩, hget⟩ := hb
        have idx_lt_n : idx < n := by
          have := (final_outputs[goal_column]).2
          omega
        have := h_all_false ⟨idx, idx_lt_n⟩
        simp [List.Vector.get] at this
        simp_all
      exact this
  · -- Out of bounds: vacuously true
    simp [h_bounds]

/-- Operational circuit iff: the evaluator accepts exactly when the chosen path
    is structurally invalid, or has no routing error and a fully discharged goal
    vector. -/
theorem circuit_iff {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat) :
    evaluateCircuit layers initial_vectors paths goal_column = true
    ↔
    PathStructurallyInvalid paths layers initial_vectors
    ∨
    (PathHasNoRoutingError paths layers initial_vectors ∧
     AllAssumptionsDischarged paths layers initial_vectors goal_column) := by
  constructor
  · exact circuit_correctness layers initial_vectors paths goal_column
  · intro h
    cases h with
    | inl h_invalid =>
      -- Structural error → had_error = true → had_error || _ = true
      unfold evaluateCircuit
      unfold PathStructurallyInvalid at h_invalid
      cases h_eval : get_eval_result layers initial_vectors paths with
      | mk final_outputs had_error =>
        rw [h_eval] at h_invalid
        simp at h_invalid
        simp [h_invalid]
    | inr h =>
      exact circuit_completeness layers initial_vectors paths goal_column h.1 h.2


end Semantic
