import Semantic.Core

open scoped Classical

namespace Semantic

/-!
# Boolean activation logic.
-/

/-- Check if a rule is active based on its activation bits.
    Intro and repetition rules need one input; elim rules need both inputs. -/
def is_rule_active {n : Nat} (r : Rule n) : Bool :=
  match r.activation with
  | ActivationBits.intro b      => b
  | ActivationBits.elim b1 b2   => b1 && b2
  | ActivationBits.repetition b => b

/-- XOR-based "exactly one true" checker.
    Returns true iff exactly one element of the list is true.
    This is the core conflict detection mechanism for the circuit. -/
def multiple_xor : List Bool → Bool
  | []      => false
  | [x]     => x
  | x :: xs => (x && not (List.or xs)) || (not x && multiple_xor xs)

/-- Extract activation status of all rules in a list. -/
def extract_activations {n : Nat} (rules : List (Rule n)) : List Bool :=
  rules.map is_rule_active

/-- Mask a list of bools: AND each element with a single bool. -/
def and_bool_list (b : Bool) (l : List Bool) : List Bool :=
  l.map (fun x => b && x)

/-- Bitwise OR over a list of boolean vectors.
    Used to combine outputs from multiple rules (only one should be non-zero). -/
def list_or {n : Nat} (vecs : List (List.Vector Bool n)) : List.Vector Bool n :=
  vecs.foldl (fun acc v => acc.zipWith (· || ·) v) (List.Vector.replicate n false)

/-- Apply rules with their activation masks.
    Each rule produces output only if its mask is true; otherwise zeros. -/
def apply_activations {n : Nat}
    (rules : List (Rule n))
    (masks : List Bool)
    (inputs : List (List.Vector Bool n)) : List (List.Vector Bool n) :=
  List.zipWith
    (fun r m => if m then r.combine inputs else List.Vector.replicate n false)
    rules masks

/-- Node logic: compute output dependency vector.
    1. Extract activation bits from all rules
    2. Check exactly-one-active via XOR
    3. Mask activations (all zero if XOR fails)
    4. Apply active rule's combine function
    5. OR results (only one is non-zero if valid) -/
def node_logic {n : Nat}
    (rules : List (Rule n))
    (inputs : List (List.Vector Bool n)) : List.Vector Bool n :=
  let acts  := extract_activations rules
  let xor   := multiple_xor acts
  let masks := and_bool_list xor acts
  let outs  := apply_activations rules masks inputs
  list_or outs

/-- Run a circuit node on given inputs. -/
def CircuitNode.run {n : Nat} (c : CircuitNode n)
    (inputs : List (List.Vector Bool n)) : List.Vector Bool n :=
  node_logic c.rules inputs

/-- Predicate: exactly one rule in the list is active.
    This is the semantic property that `multiple_xor` checks. -/
def exactlyOneActive {n : Nat} (rules : List (Rule n)) : Prop :=
  ∃ r, r ∈ rules ∧ is_rule_active r ∧ ∀ r', r' ∈ rules → is_rule_active r' → r' = r


/-- If a mapped list has no duplicates, the original list has no duplicates. -/
lemma nodup_of_map {α β} (f : α → β) {l : List α} :
    (l.map f).Nodup → l.Nodup := by
  induction l with
  | nil => intro _; simp
  | cons a tl ih =>
    intro h
    rcases List.nodup_cons.mp h with ⟨h_notin, h_tl⟩
    have ih' := ih h_tl
    have a_notin : a ∉ tl := by
      intro hmem
      exact h_notin (List.mem_map.mpr ⟨a, hmem, rfl⟩)
    exact List.nodup_cons.mpr ⟨a_notin, ih'⟩

@[simp]
lemma multiple_xor_cons_false (l : List Bool) :
    multiple_xor (false :: l) = multiple_xor l := by
  induction l with
  | nil => simp [multiple_xor]
  | cons b bs ih => simp [multiple_xor]

lemma multiple_xor_cons_true_aux {l : List Bool} :
    multiple_xor (true :: l) = !l.or := by
  cases l with
  | nil => simp [multiple_xor, List.or]
  | cons b bs => simp [multiple_xor]

lemma multiple_xor_cons_true {l : List Bool} :
    multiple_xor (true :: l) = true ↔ List.or l = false := by
  simp
  exact multiple_xor_cons_true_aux

lemma List.or_eq_false_iff_all_false {l : List Bool} :
    l.or = false ↔ ∀ b ∈ l, b = false := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.or, List.mem_cons, forall_eq_or_imp]
    simp [List.any]

/-- **Core equivalence theorem**: XOR over activation bits equals true
    if and only if exactly one rule is active.

    This is the key lemma connecting the Boolean circuit implementation
    to its semantic specification. -/
theorem multiple_xor_bool_iff_exactlyOneActive
    {n : ℕ} (rs : List (Rule n)) (h_nodup : rs.Nodup) :
    multiple_xor (rs.map is_rule_active) = true ↔ exactlyOneActive rs := by
  induction rs with
  | nil => simp [multiple_xor, exactlyOneActive]
  | cons r rs ih =>
    have tail_nodup : rs.Nodup := List.nodup_cons.mp h_nodup |>.2
    cases hr : is_rule_active r
    · -- r is inactive
      simp only [List.map, hr]
      rw [multiple_xor_cons_false, ih tail_nodup]
      simp only [exactlyOneActive]
      constructor
      · intro ⟨r₀, hr₀_in, h_act, h_uniq⟩
        exact ⟨r₀, List.mem_cons_of_mem _ hr₀_in, h_act, by
          intro r' hr'_mem hr'_act
          cases hr'_mem with
          | head => rw [hr] at hr'_act; contradiction
          | tail _ h_tail => exact h_uniq r' h_tail hr'_act⟩
      · intro ⟨r₀, hr₀_mem, h_act, h_uniq⟩
        cases hr₀_mem with
        | head => rw [hr] at h_act; contradiction
        | tail _ h_tail =>
          exact ⟨r₀, h_tail, h_act,
            fun r' h_in h_act' => h_uniq r' (List.mem_cons_of_mem _ h_in) h_act'⟩
    · -- r is active
      simp only [List.map, hr]
      rw [multiple_xor_cons_true]
      simp only [exactlyOneActive]
      constructor
      · intro h
        have all_false : ∀ a ∈ rs, is_rule_active a = false := by
          intro a ha
          have all_false_bools := List.or_eq_false_iff_all_false.mp h
          exact all_false_bools (is_rule_active a) (List.mem_map.mpr ⟨a, ha, rfl⟩)
        exists r
        constructor
        · exact @List.mem_cons_self _ r rs
        · constructor
          · exact hr
          · intros r' hr'_mem hr'_act
            cases hr'_mem with
            | head => rfl
            | tail _ h_tail =>
              have h_false := all_false r' h_tail
              rw [hr'_act] at h_false
              contradiction
      · intro ⟨r₁, hr₁_mem, hr₁_active, h_unique⟩
        cases hr₁_mem with
        | head =>
          simp [List.or_eq_false_iff_all_false]
          intro b hb
          by_contra h_contra
          have hb_true : is_rule_active b = true := by
            cases h_b : is_rule_active b
            · contradiction
            · rfl
          have eq_b := h_unique b (List.mem_cons_of_mem _ hb) hb_true
          let ⟨r_ne, _⟩ := List.nodup_cons.mp h_nodup
          rw [eq_b] at hb
          contradiction
        | tail _ h_tail =>
          have eq_head := h_unique r (List.Mem.head ..) hr
          have : r ∈ rs := by rwa [←eq_head] at h_tail
          let ⟨r_ne, _⟩ := List.nodup_cons.mp h_nodup
          contradiction


end Semantic
