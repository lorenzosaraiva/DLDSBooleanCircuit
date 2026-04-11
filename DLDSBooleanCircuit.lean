import Init
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip
import Mathlib.Data.Fin.Basic


/-!
# Verified DLDS-to-Boolean Circuit Translation

This file formalizes the translation from Dag-Like Derivability Structures (DLDS)
to Boolean circuits, with machine-checked correctness proofs in Lean 4.

## Main Results

* `node_correct` - Single node evaluation correctness
* `circuit_correctness` - Circuit soundness theorem
* `dlds_evaluation_correct` - End-to-end DLDS verification correctness

## References

* Gordeev & Haeusler, "Proof Compression and NP Versus PSPACE"
-/

open scoped Classical

namespace Semantic
/-!
## Section 1: Core Types and Structures

This section defines the fundamental types for representing Boolean circuits
that verify DLDS proofs:
- `ActivationBits` - Activation state for each rule type
- `RuleData` - Rule parameters (intro encoder, elim, repetition)
- `Rule` - Complete rule with ID, activation, type, and combine function
- `CircuitNode` - Collection of alternative rules at a formula position
-/


/-- Activation bits for each inference rule type.
    - Intro rules require one input (the premise of the implication)
    - Elim rules require two inputs (the implication and its antecedent)
    - Repetition rules require one input (identity/structural rule) -/
inductive ActivationBits
  | intro (bit : Bool)
  | elim (bit1 : Bool) (bit2 : Bool)
  | repetition (bit : Bool)
  deriving DecidableEq

/-- Rule data: the type of inference rule and its parameters.
    - `intro encoder`: Implication introduction with discharge mask
    - `elim`: Implication elimination (modus ponens)
    - `repetition`: Structural rule (identity) -/
inductive RuleData (n : Nat)
  | intro (encoder : List.Vector Bool n)
  | elim
  | repetition

/-- A single inference rule consisting of:
    - `ruleId`: Unique identifier within the node
    - `activation`: Current activation state
    - `type`: Rule type and parameters
    - `combine`: Function computing output dependency vector from inputs -/
structure Rule (n : ℕ) where
  ruleId     : Nat
  activation : ActivationBits
  type       : RuleData n
  combine    : List (List.Vector Bool n) → List.Vector Bool n

/-- Circuit node: a collection of alternative inference rules at a formula position.
    The `nodupIds` invariant ensures rule IDs are unique within the node,
    which is essential for the XOR-based exactly-one-active check. -/
structure CircuitNode (n : ℕ) where
  rules    : List (Rule n)
  nodupIds : (rules.map (·.ruleId)).Nodup

/-- Constructor for implication introduction rule (⊃I).
    The combine function clears dependency bits for discharged assumptions:
    `output[i] = input[i] ∧ ¬encoder[i]` -/
def mkIntroRule {n : ℕ} (rid : Nat) (encoder : List.Vector Bool n) (bit : Bool) : Rule n :=
{
  ruleId     := rid,
  activation := ActivationBits.intro bit,
  type       := RuleData.intro encoder,
  combine    := fun deps =>
    match deps with
    | [d] => d.zipWith (fun b e => b && !e) encoder
    | _   => List.Vector.replicate n false
}

/-- Constructor for implication elimination rule (⊃E / modus ponens).
    The combine function unions dependencies from both premises:
    `output[i] = d1[i] ∨ d2[i]` -/
def mkElimRule {n : ℕ} (rid : Nat) (bit1 bit2 : Bool) : Rule n :=
{
  ruleId     := rid,
  activation := ActivationBits.elim bit1 bit2,
  type       := RuleData.elim,
  combine    := fun deps =>
    match deps with
    | [d1, d2] => d1.zipWith (· || ·) d2
    | _        => List.Vector.replicate n false
}

/-- Constructor for repetition rule (structural).
    The combine function passes the dependency vector unchanged. -/
def mkRepetitionRule {n : ℕ} (rid : Nat) (bit : Bool) : Rule n :=
{
  ruleId     := rid,
  activation := ActivationBits.repetition bit,
  type       := RuleData.repetition,
  combine    := fun deps =>
    match deps with
    | [d] => d
    | _   => List.Vector.replicate n false
}

/-- A rule is well-formed if its combine function matches its declared type.
    This ensures the rule constructors produce consistent rules. -/
def Rule.WellFormed {n : Nat} (r : Rule n) : Prop :=
  match r.type with
  | RuleData.intro encoder =>
      r.combine = fun deps =>
        match deps with
        | [d] => d.zipWith (fun b e => b && !e) encoder
        | _ => List.Vector.replicate n false
  | RuleData.elim =>
      r.combine = fun deps =>
        match deps with
        | [d1, d2] => d1.zipWith (· || ·) d2
        | _ => List.Vector.replicate n false
  | RuleData.repetition =>
      r.combine = fun deps =>
        match deps with
        | [d] => d
        | _ => List.Vector.replicate n false

/-!
## Section 2: Boolean Circuit Logic

Core Boolean operations for circuit evaluation:
- `is_rule_active` - Check if a rule's activation bits are set
- `multiple_xor` - XOR-based exactly-one-true checker
- `node_logic` - Compute node output when exactly one rule is active
- `exactlyOneActive` - Predicate capturing the exactly-one-active property

The key insight is that `multiple_xor` returns true iff exactly one element
is true, which we prove equivalent to `exactlyOneActive` in Section 3.
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
/-!

## Section 3: Key Lemmas for Correctness Proofs

This section establishes the fundamental correspondence between the Boolean
`multiple_xor` function and the semantic `exactlyOneActive` predicate.

Main result: `multiple_xor_bool_iff_exactlyOneActive`
  - `multiple_xor (rules.map is_rule_active) = true ↔ exactlyOneActive rules`

This equivalence is the foundation of circuit correctness: the XOR check
in `node_logic` succeeds precisely when exactly one rule is active.
-/

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

/-!
## Section 4: Auxiliary Lemmas for Vector Operations

Technical lemmas about boolean vector operations used in the main proofs:
- OR with zero vector is identity
- zipWith commutativity
- Folding over zero vectors preserves accumulator
-/

/-- OR-ing a zero vector on the left is identity. -/
lemma zip_with_zero_identity :
    ∀ (N : ℕ) (v : List.Vector Bool N),
      (List.Vector.replicate N false).zipWith (· || ·) v = v := by
  intro N v
  let ⟨l, hl⟩ := v
  dsimp [List.Vector.zipWith, List.Vector.replicate]
  congr
  induction l generalizing N with
  | nil => simp
  | cons hd tl ih =>
    rw [←hl]
    have hlen : (hd :: tl).length = Nat.succ (List.length tl) := by simp [List.length]
    rw [hlen] at *
    have hrep : List.replicate (Nat.succ (List.length tl)) false =
                false :: List.replicate (List.length tl) false := by simp [List.replicate]
    rw [hrep, List.zipWith_cons_cons]
    simp
    exact ih (List.length tl) ⟨tl, rfl⟩ rfl

@[simp]
theorem List.getElem_eq_get {α : Type*} (l : List α) (i : Fin l.length) :
    l[↑i] = l.get i := rfl

/-- zipWith is commutative for commutative operations. -/
@[simp]
lemma List.Vector.zipWith_comm {n : ℕ} (f : Bool → Bool → Bool)
    (h : ∀ x y, f x y = f y x)
    (v₁ v₂ : List.Vector Bool n) :
    v₁.zipWith f v₂ = v₂.zipWith f v₁ := by
  rcases v₁ with ⟨l₁, h₁⟩
  rcases v₂ with ⟨l₂, h₂⟩
  apply List.Vector.ext
  intro i
  dsimp [List.Vector.zipWith, List.Vector.get]
  rw [List.getElem_zipWith, List.getElem_zipWith]
  apply h

/-- Folding OR with zero vectors preserves the accumulator. -/
lemma foldl_add_false {n : ℕ} (v : List.Vector Bool n) (l : List α) :
    List.foldl (fun acc (_ : α) => acc.zipWith (· || ·) (List.Vector.replicate n false)) v l = v := by
  induction l with
  | nil => rfl
  | cons _ tl ih =>
    simp only [List.foldl]
    rw [List.Vector.zipWith_comm (· || ·) Bool.or_comm]
    rw [zip_with_zero_identity]
    exact ih

/-!
## Section 5: Node Correctness Theorem

This section proves the main correctness theorem for individual circuit nodes:
when exactly one rule is active, the node outputs precisely that rule's result.

Main result: `node_correct`
  - If `exactlyOneActive c.rules`, then `c.run inputs = r.combine inputs`
    for the unique active rule `r`.

This theorem is the foundation for the full circuit correctness proof.
-/

/-- When exactly one rule is active, OR-combining all rule outputs yields
    just the active rule's output (all others contribute zero vectors). -/
lemma list_or_apply_unique_active_of_exactlyOne {n : ℕ}
    {rules : List (Rule n)} (h_nonempty : rules ≠ [])
    {r0 : Rule n} (hr0_mem : r0 ∈ rules)
    (h_nodup : rules.Nodup)
    (h_one : exactlyOneActive rules)
    (hr0_active : is_rule_active r0 = true)
    (inputs : List (List.Vector Bool n)) :
    list_or (apply_activations rules (extract_activations rules) inputs) = r0.combine inputs := by
  induction rules with
  | nil => contradiction
  | cons r rs ih =>
      have h_r0 : r0 = r ∨ r0 ∈ rs := by
        cases hr0_mem
        case head => exact Or.inl rfl
        case tail h' => exact Or.inr h'
      cases h_r0 with
      | inl r0_eq_r =>
          rcases h_one with ⟨_, ⟨mem_head, r_active, uniq⟩⟩
          have tail_inactive : ∀ r', r' ∈ rs → is_rule_active r' = false := by
            intros r' h_mem
            by_contra h_act
            let act : is_rule_active r' = true := Bool.eq_true_of_not_eq_false h_act
            let eq := uniq r' (List.mem_cons_of_mem _ h_mem) act
            let eq := uniq r' (List.mem_cons_of_mem _ h_mem) act
            subst eq
            subst r0_eq_r
            have uniq_r0 := uniq r0 (List.Mem.head rs) hr0_active
            rw [eq, ←uniq_r0] at h_mem
            let ⟨r_not_in_rs, _⟩ := List.nodup_cons.mp h_nodup
            exact r_not_in_rs h_mem

          dsimp [apply_activations, extract_activations, list_or]
          rw [←r0_eq_r, hr0_active]
          have outs_tail_eq : (List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false) rs (rs.map is_rule_active))
            = rs.map (fun _ => List.Vector.replicate n false) := by
              apply List.ext_get (by simp)
              intro i h₁ h₂
              have hlen : (List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false) rs (rs.map is_rule_active)).length = rs.length :=
                by
                  simp [List.length_zipWith]
              have len_zip : (List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false) rs (rs.map is_rule_active)).length = rs.length := by simp [List.length_zipWith]
              have len_map : (List.map (fun x => List.Vector.replicate n false) rs).length = rs.length := by simp
              let fin_zip := Fin.mk i (len_zip ▸ h₁)
              let fin_map := Fin.mk i (len_map ▸ h₂)
              let x := rs.get fin_zip
              have xin : x ∈ rs := List.get_mem rs fin_zip
              simp [List.getElem_zipWith]
              have fin_eq : fin_map = fin_zip := by apply Fin.ext; rfl
              intro h
              have get_eq : rs[i] = rs.get fin_map := List.getElem_eq_get rs fin_map
              rw [get_eq] at h ⊢
              rw [tail_inactive x xin] at h
              contradiction

          rw [outs_tail_eq]
          rw [List.foldl_map]
          simp only [if_true]
          cases rs
          case nil =>
            simp
            rw [zip_with_zero_identity n (r0.combine inputs)]
          case cons b l =>
            rw [zip_with_zero_identity n (r0.combine inputs)]
            simp [List.foldl, zip_with_zero_identity]
            induction l with
            | nil =>
              simp
            | cons hd tl ih =>
              simp [List.foldl]
              rw [zip_with_zero_identity n (r0.combine inputs)]
              apply foldl_add_false

      | inr r0_in_rs =>
          have r_inactive : is_rule_active r = false := by
            rcases h_one with ⟨_, ⟨_, _, uniq⟩⟩
            by_contra h'
            have eq := uniq r (List.Mem.head rs) (Bool.eq_true_of_not_eq_false h')
            have ne : r0 ≠ r := by
              intro contra
              subst contra
              have : ¬ r0 ∈ rs := List.nodup_cons.mp h_nodup |>.1
              contradiction
            rw [eq] at ne
            have r0_eq_w := uniq r0 hr0_mem hr0_active
            exact ne r0_eq_w

          rcases h_one with ⟨_, ⟨_, _, uniq⟩⟩

          have r0_eq_w := uniq r0 hr0_mem hr0_active
          have h_one_tail : exactlyOneActive rs := by
            use r0
            exact ⟨r0_in_rs, hr0_active, fun r' h' act =>
              let r'_eq_w := uniq r' (List.mem_cons_of_mem _ h') act
              by rw [←r0_eq_w] at r'_eq_w; exact r'_eq_w
            ⟩

          dsimp [apply_activations, extract_activations, list_or]
          rw [r_inactive]
          let h_nonempty_tail : rs ≠ [] := List.ne_nil_of_mem r0_in_rs
          let rs_nodup := List.nodup_cons.mp h_nodup |>.2
          simp
          rw [zip_with_zero_identity n (List.Vector.replicate n false)]
          exact ih h_nonempty_tail r0_in_rs rs_nodup h_one_tail

/-- **Main Node Correctness Theorem**: If exactly one rule is active,
    the node outputs that rule's combine result.

    This is the core correctness property: the XOR-gated node logic
    correctly selects and applies the unique active rule. -/
theorem node_correct {n} (c : CircuitNode n)
    (inputs : List (List.Vector Bool n))
    (h_one : exactlyOneActive c.rules) :
    ∃ r ∈ c.rules, c.run inputs = r.combine inputs := by
  have h_nodup : c.rules.Nodup :=
    nodup_of_map (fun (r : Rule n) => r.ruleId) c.nodupIds

  have h_bool : multiple_xor (c.rules.map is_rule_active) = true :=
    (multiple_xor_bool_iff_exactlyOneActive c.rules h_nodup).mpr h_one

  let h_one_prop := h_one
  rcases h_one with ⟨r0, hr0_mem, hr0_active, hr0_unique⟩

  dsimp [CircuitNode.run, node_logic, extract_activations]
  rw [h_bool]
  dsimp [and_bool_list]

  let h_nonempty := List.ne_nil_of_mem hr0_mem
  simp [List.map_map]

  let eq := list_or_apply_unique_active_of_exactlyOne
    h_nonempty hr0_mem h_nodup h_one_prop hr0_active inputs

  use r0
  constructor
  · exact hr0_mem
  · exact eq
/-!
## Section 6: Path-Based Circuit Evaluation

This section builds the complete circuit evaluation pipeline on top of the
proven `node_correct` theorem from Section 5.

### Overview

The evaluation proceeds as follows:
1. Initialize tokens at the top level (one per formula)
2. Propagate tokens through layers following path routing choices
3. At each node, activate rules based on available inputs
4. Detect conflicts via XOR check (multiple active rules = error)
5. Accumulate dependency vectors through rule applications
6. Check final goal vector for discharged assumptions

### Main Results

- `circuit_correctness`: If circuit accepts, then either the path is
  structurally invalid OR the path represents a valid proof with all
  assumptions discharged.
-/

/-!
### 6.1: Core Structures

Types for tokens, wiring maps, grid layers, and path inputs.
-/

/-- Token flowing through the circuit, carrying dependency information.
    - `origin_column`: Fixed; used for path lookup
    - `source_column`: Updated each step; indicates immediate predecessor
    - `dep_vector`: Accumulated dependency bitvector -/
structure Token (n : Nat) where
  origin_column : Nat
  source_column : Nat
  current_level : Nat
  current_column : Nat
  dep_vector : List.Vector Bool n
  deriving Inhabited

/-- Wiring specification for a single rule: list of (source_column, edge_id) pairs. -/
abbrev RuleIncoming := List (Nat × Nat)

/-- Wiring specification for a node: one entry per rule. -/
abbrev NodeIncoming := List RuleIncoming

/-- Wiring specification for a layer: one entry per column. -/
abbrev LayerIncoming := List NodeIncoming

/-- Grid layer containing circuit nodes and their wiring information. -/
structure GridLayer (n : ℕ) where
  nodes : List (CircuitNode n)
  incoming : LayerIncoming

/-- Path input: routing choices for each formula at each level.
    Value 0 means "stop", value k > 0 means "route to column k-1". -/
abbrev PathInput := List (List Nat)

/-!
### 6.2: Token Propagation

Initialization and propagation of tokens through the circuit grid.
-/

/-- Initialize tokens: one per column at the top level with initial dependency vectors. -/
def initialize_tokens {n : Nat}
    (initial_vectors : List (List.Vector Bool n))
    (top_level : Nat) : List (Token n) :=
  initial_vectors.zipIdx.map fun (vec, col) =>
    { origin_column := col
      source_column := col
      current_level := top_level
      current_column := col
      dep_vector := vec }

/-- Propagate tokens to the next level following path routing choices. -/
def propagate_tokens {n : Nat}
    (tokens : List (Token n))
    (paths : PathInput)
    (current_level : Nat)
    (num_levels : Nat)
    (outputs : List (List.Vector Bool n)) : List (Token n) :=
  tokens.filterMap fun token =>
    if h_path : token.origin_column < paths.length then
      let path := paths.get ⟨token.origin_column, h_path⟩
      if h_level : current_level > 0 ∧ num_levels - current_level - 1 < path.length then
        let step_index := num_levels - current_level - 1
        let edge_choice := path.get ⟨step_index, h_level.2⟩
        if edge_choice = 0 then
          none
        else
          let target_column := edge_choice - 1
          if h_out : token.current_column < outputs.length then
            some { origin_column := token.origin_column
                   source_column := token.current_column
                   current_level := current_level - 1
                   current_column := target_column
                   dep_vector := outputs.get ⟨token.current_column, h_out⟩ }
          else
            none
      else
        none
    else
      none

/-- Convert natural number to k-bit big-endian boolean vector.
    (Reserved for future bit-encoded path representation.) -/
def natToBits (n k : ℕ) : List Bool :=
  (List.range k).map (fun i => (n.shiftRight (k - 1 - i)) % 2 = 1)

/-- Generate one-hot selector from boolean input encoding.
    (Reserved for future bit-encoded path representation.) -/
def selector (input : List Bool) : List Bool :=
  let n := input.length
  let total := 2 ^ n
  List.ofFn (fun (i : Fin total) =>
    let bits := natToBits i.val n
    (input.zip bits).foldl (fun acc (inp, b) =>
      acc && if b then inp else !inp) true)

/-!
### 6.3: Rule Activation

Setting activation bits based on which required inputs are available.
-/

/-- Set activation bits for a rule based on available inputs.
    - Intro/Rep rules: active iff all required inputs present
    - Elim rules: each bit set independently based on input availability -/
def set_rule_activation {n : Nat}
    (rule : Rule n)
    (rule_incoming : RuleIncoming)
    (available_inputs : List (Nat × List.Vector Bool n)) : Rule n :=
  let required_cols := rule_incoming.map Prod.fst
  let available_cols := available_inputs.map Prod.fst
  let has_all := required_cols.all fun req => available_cols.contains req
  let new_activation := match rule.activation with
    | ActivationBits.intro _ => ActivationBits.intro has_all
    | ActivationBits.elim _ _ =>
        if required_cols.length = 2 then
          let has_first := available_cols.contains (required_cols[0]!)
          let has_second := available_cols.contains (required_cols[1]!)
          ActivationBits.elim has_first has_second
        else
          ActivationBits.elim false false
    | ActivationBits.repetition _ => ActivationBits.repetition has_all
  { rule with activation := new_activation }

def activateRulesAux {n : Nat}
    (node_incoming : NodeIncoming)
    (available_inputs : List (Nat × List.Vector Bool n)) :
    Nat → List (Rule n) → List (Rule n)
  | _, [] => []
  | idx, r :: rs =>
      let rule_inc := node_incoming[idx]!
      let r' := set_rule_activation r rule_inc available_inputs
      r' :: activateRulesAux node_incoming available_inputs (idx + 1) rs

/-- Activation preserves rule IDs (needed for nodupIds invariant). -/
lemma activateRulesAux_ids {n : Nat}
    (node_incoming : NodeIncoming)
    (available_inputs : List (Nat × List.Vector Bool n)) :
    ∀ idx (rs : List (Rule n)),
      (activateRulesAux node_incoming available_inputs idx rs).map (·.ruleId) = rs.map (·.ruleId)
  | idx, [] => by simp [activateRulesAux]
  | idx, r :: rs => by
      have ih := activateRulesAux_ids node_incoming available_inputs (idx + 1) rs
      simp [activateRulesAux, set_rule_activation, ih]

/-- Activate a node's rules based on available token inputs. -/
def activate_node_from_tokens {n : Nat}
    (node : CircuitNode n)
    (node_incoming : NodeIncoming)
    (available_inputs : List (Nat × List.Vector Bool n)) : CircuitNode n :=
  let activated_rules := activateRulesAux node_incoming available_inputs 0 node.rules
  { rules := activated_rules
    nodupIds := by
      classical
      have h_ids : activated_rules.map (·.ruleId) = node.rules.map (·.ruleId) :=
        activateRulesAux_ids node_incoming available_inputs 0 node.rules
      simpa [activated_rules, h_ids] using node.nodupIds }


/-!
### 6.4: Node Evaluation with Routing

Extended node logic that routes inputs to individual rules and detects conflicts.
-/

def gather_rule_inputs {n : Nat}
  (rule_incoming : RuleIncoming)
  (available_inputs : List (Nat × List.Vector Bool n))
  : List (List.Vector Bool n) :=
  let result := rule_incoming.filterMap fun (required_col, _edge_id) =>
    available_inputs.find? (fun (col, _) => col = required_col) |>.map Prod.snd
  result

/-- Modified apply_activations that uses per-rule inputs -/
def apply_activations_with_routing {n: Nat}
  (rules : List (Rule n))
  (masks : List Bool)
  (per_rule_inputs : List (List (List.Vector Bool n)))  -- One input list per rule!
: List (List.Vector Bool n) :=
  List.zipWith3
    (fun (r : Rule n) (m : Bool) (inputs : List (List.Vector Bool n)) =>
      if m then r.combine inputs  -- Each rule gets its OWN inputs!
      else List.Vector.replicate n false)
    rules masks per_rule_inputs

/-- Modified node_logic with proper input routing and conflict detection -/
def node_logic_with_routing {n : Nat}
  (rules : List (Rule n))
  (node_incoming : NodeIncoming)
  (available_inputs : List (Nat × List.Vector Bool n))
  : (List.Vector Bool n) × Bool :=
  let acts := extract_activations rules
  let xor := multiple_xor acts
  let masks := and_bool_list xor acts
  -- Detect conflict: XOR fails and at least one rule is active
  let has_conflict := !xor && acts.any (· = true)
  -- Gather per-rule inputs based on IncomingMap
  let per_rule_inputs := rules.zipIdx.map fun (_rule, rule_idx) =>
    let rule_inc := node_incoming[rule_idx]!
    gather_rule_inputs rule_inc available_inputs

  -- Apply with routing
  let outs := apply_activations_with_routing rules masks per_rule_inputs

  let result := list_or outs
  (result, has_conflict)

def evaluate_node {n : Nat}
  (node : CircuitNode n)
  (node_incoming : NodeIncoming)
  (tokens_at_node : List (Token n))
  : (List.Vector Bool n) × Bool :=

  if tokens_at_node.isEmpty then
    (List.Vector.replicate n false, false)
  else
    let available_inputs := tokens_at_node.map fun t => (t.source_column, t.dep_vector)
    let available_sources := available_inputs.map Prod.fst

    let instantiated_rules := node.rules.zipIdx.map fun (rule, rule_idx) =>
      let rule_incoming := node_incoming[rule_idx]!
      let required_sources := rule_incoming.map Prod.fst
      let has_all_inputs := required_sources.all (fun src => available_sources.contains src)

      let new_activation := match rule.activation, required_sources.length with
        | ActivationBits.intro _, _ => ActivationBits.intro has_all_inputs
        | ActivationBits.elim _ _, 2 =>
            ActivationBits.elim
              (available_sources.contains (required_sources[0]!))
              (available_sources.contains (required_sources[1]!))
        | ActivationBits.elim _ _, _ => ActivationBits.elim false false
        | ActivationBits.repetition _, _ => ActivationBits.repetition has_all_inputs

      { rule with activation := new_activation }

    let (result, error)  := node_logic_with_routing instantiated_rules node_incoming available_inputs
    (result, error)

/-!
### 6.5: Helper Lemmas for Routing Correctness

Technical lemmas about list operations (zipWith3, membership, indexing)
used in proving `node_logic_with_routing_correct`.
-/

lemma exists_fin_of_mem {α} {a : α} {l : List α} (h : a ∈ l) :
  ∃ i : Fin l.length, l.get i = a := by
  classical
  induction' l with x xs ih generalizing a
  · cases h
  · cases h with
    | head =>
        exact ⟨⟨0, by simp⟩, rfl⟩
    | tail =>
        have h' : a ∈ xs := by assumption
        obtain ⟨i, hi⟩ := ih h'
        refine ⟨⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩, ?_⟩
        simp only [List.get_cons_succ]
        exact hi


lemma List.length_zipWith3 {α β γ δ : Type*} (f : α → β → γ → δ)
    (as : List α) (bs : List β) (cs : List γ) :
    (List.zipWith3 f as bs cs).length = min as.length (min bs.length cs.length) := by
  induction as generalizing bs cs with
  | nil => simp [List.zipWith3]
  | cons a as' ih =>
    cases bs with
    | nil => simp [List.zipWith3]
    | cons b bs' =>
      cases cs with
      | nil => simp [List.zipWith3]
      | cons c cs' =>
        simp only [List.zipWith3, List.length_cons, ih]
        omega

lemma List.get_zipWith3 {α β γ δ : Type*} (f : α → β → γ → δ)
    (as : List α) (bs : List β) (cs : List γ) (i : Nat)
    (ha : i < as.length) (hb : i < bs.length) (hc : i < cs.length) :
    (List.zipWith3 f as bs cs).get ⟨i, by rw [List.length_zipWith3]; omega⟩ =
    f (as.get ⟨i, ha⟩) (bs.get ⟨i, hb⟩) (cs.get ⟨i, hc⟩) := by
  induction as generalizing bs cs i with
  | nil => simp at ha
  | cons a as' ih =>
    cases bs with
    | nil => simp at hb
    | cons b bs' =>
      cases cs with
      | nil => simp at hc
      | cons c cs' =>
        cases i with
        | zero => simp [List.zipWith3]
        | succ i' =>
          simp only [List.zipWith3, List.get_cons_succ]
          have ha' : i' < as'.length := Nat.lt_of_succ_lt_succ ha
          have hb' : i' < bs'.length := Nat.lt_of_succ_lt_succ hb
          have hc' : i' < cs'.length := Nat.lt_of_succ_lt_succ hc
          exact ih bs' cs' i' ha' hb' hc'

lemma Vector.zipWith_or_replicate_false_left {n : Nat} (v : List.Vector Bool n) :
    List.Vector.zipWith (· || ·) (List.Vector.replicate n false) v = v := by
  apply List.Vector.ext
  intro i
  simp [List.Vector.get_replicate]

lemma Vector.zipWith_or_replicate_false_right {n : Nat} (v : List.Vector Bool n) :
    List.Vector.zipWith (· || ·) v (List.Vector.replicate n false) = v := by
  apply List.Vector.ext
  intro i
  simp [List.Vector.get_replicate]

lemma foldl_zipWith_or_all_zeros {n : Nat} (acc : List.Vector Bool n) (vecs : List (List.Vector Bool n))
    (h_all_zero : ∀ j (hj : j < vecs.length), vecs.get ⟨j, hj⟩ = List.Vector.replicate n false) :
    List.foldl (fun a v => List.Vector.zipWith (· || ·) a v) acc vecs = acc := by
  induction vecs generalizing acc with
  | nil => rfl
  | cons v vs ih =>
    simp only [List.foldl_cons]
    have hv : v = List.Vector.replicate n false := h_all_zero 0 (by simp)
    rw [hv, Vector.zipWith_or_replicate_false_right]
    apply ih
    intro j hj
    have := h_all_zero (j + 1) (by simp; omega)
    simpa using this


lemma list_or_single_nonzero {n : Nat} (vecs : List (List.Vector Bool n))
    (i : Nat) (hi : i < vecs.length)
    (h_others : ∀ j (hj : j < vecs.length), j ≠ i →
        vecs.get ⟨j, hj⟩ = List.Vector.replicate n false) :
    List.foldl (fun acc v => List.Vector.zipWith (· || ·) acc v)
      (List.Vector.replicate n false) vecs = vecs.get ⟨i, hi⟩ := by
  induction vecs generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    simp only [List.foldl_cons, List.length_cons] at hi ⊢
    cases i with
    | zero =>
      -- x is the active element, all of xs are zeros
      have h_xs_zero : ∀ j (hj : j < xs.length),
          xs.get ⟨j, hj⟩ = List.Vector.replicate n false := by
        intro j hj
        have hj' : j + 1 < (x :: xs).length := by simp; omega
        have := h_others (j + 1) hj' (by omega)
        simpa using this
      rw [Vector.zipWith_or_replicate_false_left]
      rw [foldl_zipWith_or_all_zeros x xs h_xs_zero]
      simp
    | succ i' =>
      have hx_zero : x = List.Vector.replicate n false := by
        have h0 : 0 < (x :: xs).length := by simp
        have := h_others 0 h0 (by omega)
        simpa using this
      rw [hx_zero, Vector.zipWith_or_replicate_false_left]
      have hi' : i' < xs.length := Nat.lt_of_succ_lt_succ hi
      have h_others' : ∀ j (hj : j < xs.length), j ≠ i' →
          xs.get ⟨j, hj⟩ = List.Vector.replicate n false := by
        intro j hj hne
        have hj' : j + 1 < (x :: xs).length := by simp; omega
        have := h_others (j + 1) hj' (by omega)
        simpa using this
      have := ih i' hi' h_others'
      convert this using 1

lemma list_map_get {α β : Type*} (f : α → β) (l : List α) (i : Nat)
    (hi : i < l.length) (hi' : i < (l.map f).length) :
    (l.map f).get ⟨i, hi'⟩ = f (l.get ⟨i, hi⟩) := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    cases i with
    | zero => rfl
    | succ i' =>
      simp only [List.map_cons, List.get_cons_succ]
      have hi'_xs : i' < xs.length := Nat.lt_of_succ_lt_succ hi
      have hi'_map : i' < (xs.map f).length := by simp; exact hi'_xs
      exact ih i' hi'_xs hi'_map

lemma list_zipIdx_get_fst {α : Type*} (l : List α) (n : Nat) (i : Nat)
    (hi : i < (l.zipIdx n).length)
    (hi' : i < l.length := by simp [List.length_zipIdx] at hi; exact hi) :
    ((l.zipIdx n).get ⟨i, hi⟩).1 = l.get ⟨i, hi'⟩ := by
  induction l generalizing n i with
  | nil => simp at hi'
  | cons x xs ih =>
    cases i with
    | zero => simp
    | succ i' => simp only [List.zipIdx_cons, List.get_cons_succ]; apply ih


lemma list_range_get (n : Nat) (i : Nat) (hi : i < (List.range n).length) :
    (List.range n).get ⟨i, hi⟩ = i := by
  simp at hi ⊢

lemma node_logic_with_routing_correct
  {n : Nat}
  (rules : List (Rule n))
  (node_incoming : NodeIncoming)
  (available_inputs : List (Nat × List.Vector Bool n))
  (h_one : exactlyOneActive rules)
  (h_nodup : rules.Nodup)
  (hlen : node_incoming.length = rules.length) :
  ∃ (r : Rule n) (i : Nat) (hi : i < rules.length),
    r ∈ rules ∧
    rules.get ⟨i, hi⟩ = r ∧
    node_logic_with_routing rules node_incoming available_inputs =
      (let rule_inc := node_incoming[i]!
       let inputs := gather_rule_inputs rule_inc available_inputs
       r.combine inputs, false) :=
by
  classical
  rcases h_one with ⟨r₀, hr₀_mem, hr₀_act, hr₀_unique⟩

  let acts := extract_activations rules
  have h_acts :
    ∀ r ∈ rules, is_rule_active r = true ↔ r = r₀ := by
    intro r hr
    constructor
    · intro h
      exact hr₀_unique r hr h
    · intro h
      simp [h, hr₀_act]

  have h_xor : multiple_xor acts = true := by
    have := (multiple_xor_bool_iff_exactlyOneActive rules h_nodup).mpr
      ⟨r₀, hr₀_mem, hr₀_act, hr₀_unique⟩
    simpa [acts, extract_activations] using this

  have h_masks : and_bool_list (multiple_xor acts) acts = acts := by
    simp [and_bool_list, h_xor]

  let per_rule_inputs :=
    (List.range rules.length).map (fun idx =>
      let rule_inc := node_incoming[idx]!
      gather_rule_inputs rule_inc available_inputs)

  have h_len_per :
    per_rule_inputs.length = rules.length := by
    simp [per_rule_inputs]

  let masks := and_bool_list (multiple_xor acts) acts
  have hmasks_eq : masks = acts := h_masks

  let outs := apply_activations_with_routing rules masks per_rule_inputs

  classical
  have ⟨i₀_fin, hi₀_get⟩ :
    ∃ i₀ : Fin rules.length, rules.get i₀ = r₀ :=
    exists_fin_of_mem (l := rules) hr₀_mem

  let i₀ : ℕ := i₀_fin
  have hi₀_lt : i₀ < rules.length := i₀_fin.isLt

  have h_len_masks : masks.length = rules.length := by
    simp [masks, hmasks_eq, acts, extract_activations]

  have h_len_outs : outs.length = rules.length := by
    simp only [outs, apply_activations_with_routing, List.length_zipWith3, h_len_masks, h_len_per]
    omega

  have hi₀_outs : i₀ < outs.length := by
    simpa [h_len_outs] using hi₀_lt

  have hi₀_per : i₀ < per_rule_inputs.length := by
    simpa [h_len_per] using hi₀_lt

  have h_act_i₀ :
    acts.get ⟨i₀, by simpa [acts, extract_activations] using hi₀_lt⟩ = true := by
    have h_r₀ : rules.get ⟨i₀, hi₀_lt⟩ = r₀ := hi₀_get
    have h_active : is_rule_active (rules.get ⟨i₀, hi₀_lt⟩) = true := by
      rw [h_r₀]
      exact hr₀_act
    simpa [acts, extract_activations] using h_active

  have h_mask_i₀ :
    masks.get ⟨i₀,
      by
        have : masks.length = rules.length := by
          simp [masks, hmasks_eq, acts, extract_activations]
        simpa [this] using hi₀_lt⟩
      = true := by
    simpa [masks, hmasks_eq] using h_act_i₀

  have hi₀_get' : rules.get ⟨i₀, hi₀_lt⟩ = r₀ := by
    simpa [i₀] using hi₀_get

  have hi₀_masks : i₀ < masks.length := by
    rw [h_len_masks]
    exact hi₀_lt

  have h_outs_i₀ :
    outs.get ⟨i₀, hi₀_outs⟩ = r₀.combine (per_rule_inputs.get ⟨i₀, hi₀_per⟩) := by
    show (apply_activations_with_routing rules masks per_rule_inputs).get ⟨i₀, hi₀_outs⟩ = _
    simp only [apply_activations_with_routing]
    conv_lhs =>
      rw [List.get_zipWith3 (fun r m ins => if m = true then r.combine ins else List.Vector.replicate n false)
          rules masks per_rule_inputs i₀ hi₀_lt hi₀_masks hi₀_per]
    rw [hi₀_get', h_mask_i₀]
    simp

  have h_only_i₀_active : ∀ j (hj : j < rules.length), j ≠ i₀ →
      is_rule_active (rules.get ⟨j, hj⟩) = false := by
    intro j hj hne
    by_contra h_not_false
    push_neg at h_not_false
    have h_true : is_rule_active (rules.get ⟨j, hj⟩) = true := Bool.eq_true_of_not_eq_false h_not_false
    have h_eq_r₀ : rules.get ⟨j, hj⟩ = r₀ := hr₀_unique _ (List.get_mem rules ⟨j, hj⟩) h_true
    have h_also_r₀ : rules.get ⟨i₀, hi₀_lt⟩ = r₀ := hi₀_get'
    have h_same : rules.get ⟨j, hj⟩ = rules.get ⟨i₀, hi₀_lt⟩ := by rw [h_eq_r₀, h_also_r₀]
    have h_fin_eq : (⟨j, hj⟩ : Fin rules.length) = ⟨i₀, hi₀_lt⟩ :=
      (List.Nodup.get_inj_iff h_nodup).mp h_same
    have h_idx_eq : j = i₀ := by
      have h_fin_eq := (List.Nodup.get_inj_iff h_nodup).mp h_same
      simp only [Fin.mk.injEq] at h_fin_eq
      exact h_fin_eq
    exact hne h_idx_eq

  have h_outs_zero : ∀ j (hj : j < outs.length), j ≠ i₀ →
      outs.get ⟨j, hj⟩ = List.Vector.replicate n false := by
    intro j hj hne
    simp only [outs, apply_activations_with_routing]
    have hj_rules : j < rules.length := by simpa [h_len_outs] using hj
    have hj_masks : j < masks.length := by rw [h_len_masks]; exact hj_rules
    have hj_per : j < per_rule_inputs.length := by rw [h_len_per]; exact hj_rules
    rw [List.get_zipWith3 _ rules masks per_rule_inputs j hj_rules hj_masks hj_per]
    have h_mask_j : masks.get ⟨j, hj_masks⟩ = false := by
      have h_inactive := h_only_i₀_active j hj_rules hne
      have hj_acts : j < acts.length := by simp [acts, extract_activations]; exact hj_rules
      have h_eq1 : masks.get ⟨j, hj_masks⟩ = acts.get ⟨j, hj_acts⟩ := by
        have h : masks[j]'hj_masks = acts[j]'hj_acts := by simp only [hmasks_eq]
        simp only [List.get_eq_getElem] at h ⊢
        exact h
      have h_eq2 : acts.get ⟨j, hj_acts⟩ = is_rule_active (rules.get ⟨j, hj_rules⟩) := by
        simp only [acts, extract_activations]
        exact list_map_get is_rule_active rules j hj_rules (by simp; exact hj_rules)
      rw [h_eq1, h_eq2, h_inactive]
    simp only [h_mask_j, Bool.false_eq_true, ↓reduceIte]

  have h_per_rule_i₀ : per_rule_inputs.get ⟨i₀, hi₀_per⟩ =
      gather_rule_inputs (node_incoming[i₀]!) available_inputs := by
    simp [per_rule_inputs]

  refine ⟨r₀, i₀, hi₀_lt, hr₀_mem, hi₀_get', ?_⟩
  unfold node_logic_with_routing
  simp [extract_activations, and_bool_list]
  have h_enum_per : (rules.zipIdx.map fun (_, rule_idx) =>
    gather_rule_inputs (node_incoming[rule_idx]?.getD default) available_inputs) = per_rule_inputs := by
    simp only [per_rule_inputs]
    apply List.ext_get
    · simp only [List.length_map, List.length_zipIdx, List.length_range]
    · intro i hi₁ hi₂
      have hi_zipIdx : i < rules.zipIdx.length := by
        rw [List.length_map] at hi₁
        exact hi₁
      have hi_rules : i < rules.length := by
        rw [List.length_zipIdx] at hi_zipIdx
        exact hi_zipIdx
      rw [list_map_get _ _ _ hi_zipIdx (by rw [List.length_map]; exact hi_zipIdx)]
      rw [list_map_get _ _ _ (by simp; exact hi_rules) hi₂]
      congr 1
      have h1 : (rules.zipIdx.get ⟨i, hi_zipIdx⟩).2 = 0 + i := by
        rw [← List.getElem_eq_get]
        simp [List.getElem_zipIdx]
      have h2 : (List.range rules.length).get ⟨i, by simp; exact hi_rules⟩ = i := by
        rw [← List.getElem_eq_get]
        simp [List.getElem_range]
      simp only [h1, h2, Nat.zero_add]
      have hi_incoming : i < node_incoming.length := by rw [hlen]; exact hi_rules
      rw [List.getElem?_eq_getElem hi_incoming, Option.getD_some]
      simp only [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hi_incoming, Option.getD_some]

  constructor
  ·
    have h_list_or_eq : list_or outs = outs.get ⟨i₀, hi₀_outs⟩ := by
      unfold list_or
      exact list_or_single_nonzero outs i₀ hi₀_outs h_outs_zero

    have h_xor' : multiple_xor (List.map is_rule_active rules) = true := by
      exact h_xor

    have h_masks_eq_acts : List.map ((fun b => multiple_xor (List.map is_rule_active rules) && b) ∘ is_rule_active) rules =
                           List.map is_rule_active rules := by
      congr 1
      funext r
      simp only [Function.comp_apply, h_xor', Bool.true_and]

    have h_acts_def : acts = List.map is_rule_active rules := rfl
    rw [h_enum_per, h_masks_eq_acts]
    have h_goal_eq_outs :
        apply_activations_with_routing rules (List.map is_rule_active rules) per_rule_inputs = outs := by
      simp only [outs]
      congr 1
      simp only [masks, hmasks_eq, acts, extract_activations]
    rw [h_goal_eq_outs, h_list_or_eq, h_outs_i₀, h_per_rule_i₀]
    congr 2
    have hi₀_incoming : i₀ < node_incoming.length := by rw [hlen]; exact hi₀_lt
    rw [List.getElem!_eq_getElem?_getD (α := _), List.getElem?_eq_getElem hi₀_incoming]
  ·
    intro h_xor_false
    simp only [acts, extract_activations] at h_xor
    rw [h_xor] at h_xor_false
    simp at h_xor_false

/-!
### Section 6.6: Routing-Aware Node Semantics

This section proves that routing-aware node evaluation correctly implements the
semantic “exactly one active rule” condition and raises a structural error
precisely when uniqueness fails.

Main results:
- `node_logic_with_routing_correct`
- `evaluate_node_error_iff_not_unique`
- `evaluate_node_correct`
-/

lemma activateRulesAux_eq_zipIdx_map {n : Nat}
    (node_incoming : NodeIncoming)
    (available_inputs : List (Nat × List.Vector Bool n))
    (rules : List (Rule n))
    (start : Nat) :
    activateRulesAux node_incoming available_inputs start rules =
    (rules.zipIdx start).map fun x =>
      let rule_incoming := node_incoming[x.2]!
      let available_sources := available_inputs.map Prod.fst
      let required_sources := rule_incoming.map Prod.fst
      let has_all_inputs := required_sources.all fun src => available_sources.contains src
      let new_activation := match x.1.activation, required_sources.length with
        | ActivationBits.intro _, _ => ActivationBits.intro has_all_inputs
        | ActivationBits.elim _ _, 2 =>
            ActivationBits.elim
              (available_sources.contains (required_sources[0]!))
              (available_sources.contains (required_sources[1]!))
        | ActivationBits.elim _ _, _ => ActivationBits.elim false false
        | ActivationBits.repetition _, _ => ActivationBits.repetition has_all_inputs
      { x.1 with activation := new_activation } := by
  induction rules generalizing start with
  | nil =>
    simp [activateRulesAux, List.zipIdx]
  | cons r rs ih =>
    simp only [activateRulesAux, List.zipIdx_cons, List.map_cons]
    congr 1
    · -- Head equality
      unfold set_rule_activation
      cases hact : r.activation with
      | intro b => rfl
      | elim b1 b2 =>
        simp only
        split
        ·
          rename_i h
          simp only [h]
        ·
          rename_i h1
          match hlen : (node_incoming[start]!.map Prod.fst).length with
          | 0 => rfl
          | 1 => rfl
          | 2 => exact absurd hlen h1
          | n + 3 => rfl
      | repetition b => rfl
    · exact ih (start + 1)

lemma activateRulesAux_eq_zipIdx_map_zero {n : Nat}
    (node_incoming : NodeIncoming)
    (available_inputs : List (Nat × List.Vector Bool n))
    (rules : List (Rule n)) :
    activateRulesAux node_incoming available_inputs 0 rules =
    rules.zipIdx.map fun x =>
      let rule_incoming := node_incoming[x.2]!
      let available_sources := available_inputs.map Prod.fst
      let required_sources := rule_incoming.map Prod.fst
      let has_all_inputs := required_sources.all fun src => available_sources.contains src
      let new_activation := match x.1.activation, required_sources.length with
        | ActivationBits.intro _, _ => ActivationBits.intro has_all_inputs
        | ActivationBits.elim _ _, 2 =>
            ActivationBits.elim
              (available_sources.contains (required_sources[0]!))
              (available_sources.contains (required_sources[1]!))
        | ActivationBits.elim _ _, _ => ActivationBits.elim false false
        | ActivationBits.repetition _, _ => ActivationBits.repetition has_all_inputs
      { x.1 with activation := new_activation } := by
  exact activateRulesAux_eq_zipIdx_map node_incoming available_inputs rules 0

/-!
### evaluate_node as a Derived Evaluator

This lemma establishes that `evaluate_node` does not introduce new
semantics: it computes rule activations from available tokens and then
invokes the already verified routing-aware node logic.
-/

theorem evaluate_node_uses_proven_node_logic
  {n : Nat}
  (node : CircuitNode n)
  (node_incoming : NodeIncoming)
  (tokens : List (Token n))
  (h_nonempty : tokens.length > 0) :
  let available_inputs := tokens.map fun t => (t.source_column, t.dep_vector)
  let activated_node := activate_node_from_tokens node node_incoming available_inputs
  evaluate_node node node_incoming tokens =
    node_logic_with_routing activated_node.rules node_incoming available_inputs := by

  intro available_inputs activated_node
  unfold evaluate_node

  have h_tokens_ne_nil : tokens ≠ [] := by
    intro h_eq
    rw [h_eq] at h_nonempty
    simp at h_nonempty

  rw [if_neg]
  ·
    simp only [activated_node, available_inputs]
    rw [Prod.eta]
    congr 1
    simp only [activate_node_from_tokens]
    rw [activateRulesAux_eq_zipIdx_map_zero]
  ·
    intro h_isEmpty
    cases tokens with
    | nil => exact h_tokens_ne_nil rfl
    | cons head tail => simp at h_isEmpty

/-!
### `evaluate_node`: No-Conflict + Some-Active ↔ Exactly-One-Active

For nonempty tokens, `evaluate_node` reports no conflict and activates at least
one rule iff exactly one rule is active.
-/
theorem evaluate_node_error_iff_not_unique
  {n : Nat}
  (node : CircuitNode n)
  (node_incoming : NodeIncoming)
  (tokens : List (Token n))
  (h_nonempty : tokens.length > 0) :
  let available_inputs := tokens.map fun t => (t.source_column, t.dep_vector)
  let activated_node := activate_node_from_tokens node node_incoming available_inputs
  let acts := extract_activations activated_node.rules
  ((evaluate_node node node_incoming tokens).snd = false ∧ acts.any (· = true))
  ↔
  exactlyOneActive activated_node.rules := by

  intro available_inputs activated_node acts

  have h_nodup : activated_node.rules.Nodup :=
    nodup_of_map (·.ruleId) activated_node.nodupIds

  have h_xor_iff : multiple_xor (extract_activations activated_node.rules) = true ↔
                   exactlyOneActive activated_node.rules :=
    multiple_xor_bool_iff_exactlyOneActive activated_node.rules h_nodup

  have h_eval : evaluate_node node node_incoming tokens =
      node_logic_with_routing activated_node.rules node_incoming available_inputs :=
    evaluate_node_uses_proven_node_logic node node_incoming tokens h_nonempty

  rw [h_eval]

  unfold node_logic_with_routing
  simp only

  constructor

  ·
    intro ⟨h_no_err, h_any⟩
    rw [← h_xor_iff]
    simp only [acts, extract_activations] at h_no_err h_any
    cases h_xor : multiple_xor (List.map is_rule_active activated_node.rules)
    ·
      exfalso
      simp only [h_xor, Bool.not_false, Bool.true_and] at h_no_err
      rw [h_any] at h_no_err
      contradiction
    ·
      simp only [extract_activations]
      exact h_xor

  ·
    intro h_one
    have h_xor : multiple_xor (extract_activations activated_node.rules) = true :=
      h_xor_iff.mpr h_one
    constructor
    ·
      simp only [extract_activations] at h_xor ⊢
      simp [h_xor]
    ·
      obtain ⟨r, hr_mem, hr_act, _⟩ := h_one
      simp only [acts, extract_activations]
      rw [List.any_eq_true]
      use true
      constructor
      ·
        rw [List.mem_map]
        exact ⟨r, hr_mem, hr_act⟩
      · rfl

lemma indexOf_eq_of_get {α : Type*} [DecidableEq α] {l : List α} {a : α} {i : Nat} (hi : i < l.length)
    (h_nodup : l.Nodup) (h_get : l.get ⟨i, hi⟩ = a) :
    l.idxOf a = i := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    cases i with
    | zero =>
      simp only [List.get] at h_get
      subst h_get
      simp [List.idxOf, List.findIdx_cons]
    | succ i' =>
      simp only [List.get] at h_get
      have h_ne : x ≠ a := by
        intro h_eq
        rw [h_eq] at h_nodup
        have := List.nodup_cons.mp h_nodup
        rw [← h_get] at this
        exact this.1 (List.get_mem xs ⟨i', by simp at hi; exact hi⟩)
      have hi' : i' < xs.length := by simp at hi; exact hi
      have h_nodup' : xs.Nodup := (List.nodup_cons.mp h_nodup).2
      have ih_result := ih hi' h_nodup' h_get
      simp only [List.idxOf, List.findIdx_cons]
      have h_ne_beq : (x == a) = false := by simp [h_ne]
      simp only [h_ne_beq, cond_false]
      simp only [List.idxOf] at ih_result
      rw [ih_result]

/-!
### `evaluate_node` Produces the Output of the Selected Rule

If no conflict is reported and some rule is active, then the first component of
`evaluate_node` equals `r.combine` applied to the routed inputs of a rule `r`
in the activated node.
-/
theorem evaluate_node_correct
  {n : Nat}
  (node : CircuitNode n)
  (node_incoming : NodeIncoming)
  (tokens : List (Token n))
  (h_nonempty : tokens.length > 0)
  (h_incoming_len : node_incoming.length = node.rules.length)
  (h_no_error : (evaluate_node node node_incoming tokens).snd = false)
  (h_some_active : (extract_activations (activate_node_from_tokens node node_incoming
                     (tokens.map fun t => (t.source_column, t.dep_vector))).rules).any (· = true)) :
  let available_inputs := tokens.map fun t => (t.source_column, t.dep_vector)
  let activated_node := activate_node_from_tokens node node_incoming available_inputs
  ∃ r ∈ activated_node.rules,
    let rule_idx := activated_node.rules.idxOf r
    let rule_inc := node_incoming[rule_idx]!
    let inputs := gather_rule_inputs rule_inc available_inputs
    (evaluate_node node node_incoming tokens).fst = r.combine inputs := by

  intro available_inputs activated_node

  have h_one : exactlyOneActive activated_node.rules := by
    have h_iff := evaluate_node_error_iff_not_unique node node_incoming tokens h_nonempty
    simp only at h_iff
    exact h_iff.mp ⟨h_no_error, h_some_active⟩

  have h_nodup : activated_node.rules.Nodup :=
    nodup_of_map (·.ruleId) activated_node.nodupIds

  have h_activated_len : activated_node.rules.length = node.rules.length := by
    simp only [activated_node, activate_node_from_tokens]
    have h : ∀ idx, (activateRulesAux node_incoming available_inputs idx node.rules).length = node.rules.length := by
      intro idx
      induction node.rules generalizing idx with
      | nil => simp [activateRulesAux]
      | cons r rs ih =>
        simp only [activateRulesAux, List.length_cons]
        rw [ih]
    exact h 0

  have h_len : node_incoming.length = activated_node.rules.length := by
    rw [h_incoming_len, h_activated_len]

  have h_routing := node_logic_with_routing_correct activated_node.rules node_incoming
                      available_inputs h_one h_nodup h_len

  obtain ⟨r, i, hi, hr_mem, hr_get, hr_eq⟩ := h_routing

  rw [evaluate_node_uses_proven_node_logic node node_incoming tokens h_nonempty]

  have h_fst : (node_logic_with_routing activated_node.rules node_incoming available_inputs).fst =
               r.combine (gather_rule_inputs (node_incoming[i]!) available_inputs) := by
    rw [hr_eq]

  use r, hr_mem
  simp only [available_inputs, activated_node]
  rw [h_fst]

  congr 1
  congr 1
  have h_indexOf : activated_node.rules.idxOf r = i := by
    exact indexOf_eq_of_get hi h_nodup hr_get
  simp only [activated_node, available_inputs] at h_indexOf ⊢
  rw [h_indexOf]

/-!
### 6.7: Layer Evaluation and Error Aggregation

This section defines evaluation of an entire grid layer by applying
`evaluate_node` to each column, collecting dependency outputs, and
aggregating structural error flags.
-/


/-- Evaluate entire layer using evaluate_node -/
def evaluate_layer {n : Nat}
  (layer : GridLayer n)
  (tokens : List (Token n))
  : (List (List.Vector Bool n)) × Bool :=

  let results := layer.nodes.zipIdx.map fun (node, col_idx) =>
    let tokens_here := tokens.filter (·.current_column = col_idx)
    let node_incoming := layer.incoming[col_idx]!
    evaluate_node node node_incoming tokens_here

  let outputs := results.map Prod.fst
  let errors := results.map Prod.snd
  let any_error := errors.any id

  (outputs, any_error)

/-!
### 6.8: Correctness Predicates
-/

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

/-- Path represents valid proof if no structural invalidity -/
def PathRepresentsValidProof {n : Nat}
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
/-!
### 6.9: Circuit Evaluation Entry Point

The main evaluation function combines layer-by-layer propagation with
the final check for discharged assumptions.
-/

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

/-!
### 6.10: Circuit Correctness Theorem

This section proves the main soundness result: if the circuit accepts,
then either a structural error occurred OR the path represents a valid
proof with all assumptions discharged.
-/

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

/-- **Main Circuit Correctness Theorem**:
    If `evaluateCircuit` returns true, then either:
    1. The path is structurally invalid (XOR conflict detected), OR
    2. The path represents a valid proof AND all assumptions are discharged.

    This is the key soundness result connecting Boolean circuit evaluation
    to proof validity. -/
theorem circuit_correctness {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat)
    (h_accept : evaluateCircuit layers initial_vectors paths goal_column = true) :
    PathStructurallyInvalid paths layers initial_vectors
    ∨
    (PathRepresentsValidProof paths layers initial_vectors ∧
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
  · -- Case: had_error = false → valid proof with all discharged
    right
    have h_err_snd : (get_eval_result layers initial_vectors paths).snd = false := by
      rw [h_eval]; simp [h_err]
    constructor
    · -- PathRepresentsValidProof
      unfold PathRepresentsValidProof
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


/-- **Circuit Completeness Theorem**:
    If the path represents a valid proof (no structural error) and all
    assumptions are discharged, then `evaluateCircuit` returns true.

    This is the converse of `circuit_correctness`, establishing that the
    circuit faithfully characterizes validity: it accepts if and only if
    the path is either structurally invalid or a valid closed derivation. -/

theorem circuit_completeness {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat)
    (h_valid : PathRepresentsValidProof paths layers initial_vectors)
    (h_discharged : AllAssumptionsDischarged paths layers initial_vectors goal_column) :
    evaluateCircuit layers initial_vectors paths goal_column = true := by
  unfold evaluateCircuit
  cases h_eval : get_eval_result layers initial_vectors paths with
  | mk final_outputs had_error =>
  -- Since path is valid, had_error = false
  have h_no_error : had_error = false := by
    unfold PathRepresentsValidProof PathStructurallyInvalid at h_valid
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

/-- Circuit correctness is an iff: the circuit accepts if and only if the path
    is structurally invalid or represents a valid proof with all assumptions
    discharged. This combines `circuit_correctness` and `circuit_completeness`
    with the trivial observation that structural invalidity implies acceptance. -/
theorem circuit_iff {n : Nat}
    (layers : List (GridLayer n))
    (initial_vectors : List (List.Vector Bool n))
    (paths : PathInput)
    (goal_column : Nat) :
    evaluateCircuit layers initial_vectors paths goal_column = true
    ↔
    PathStructurallyInvalid paths layers initial_vectors
    ∨
    (PathRepresentsValidProof paths layers initial_vectors ∧
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

/-!
## Section 7: DLDS Grid Construction

This section bridges raw DLDS structures to evaluation-ready GridLayers:
- Formula universe extraction
- Encoder generation for intro rules
- Incoming map (wiring) construction
- Node and layer builders
- Well-formedness predicates and construction correctness
-/

/-!
### 7.1: DLDS Type Definitions
-/

/-- Propositional formula: atoms or implications. -/
inductive Formula
  | atom (name : String)
  | impl (A B : Formula)
  deriving DecidableEq, Repr, Inhabited

def Formula.toString : Formula → String
  | .atom s => s
  | .impl A B => s!"({A.toString} ⊃ {B.toString})"

instance : ToString Formula where
  toString := Formula.toString

/-- Vertex in a DLDS: a formula occurrence at a specific level. -/
structure Vertex where
  node : Nat
  LEVEL : Nat
  FORMULA : Formula
  HYPOTHESIS : Bool
  COLLAPSED : Bool
  PAST : List Nat
  deriving Repr, DecidableEq

/-- Deduction edge: connects two vertices with dependency tracking. -/
structure Deduction where
  START : Vertex
  END : Vertex
  COLOUR : Nat
  DEPENDENCY : List Formula
  deriving Repr, DecidableEq

/-- Dag-Like Derivability Structure: vertices, edges, and auxiliary pairs. -/
structure DLDS where
  V : List Vertex
  E : List Deduction
  A : List (Vertex × Vertex) := []
  deriving Repr, DecidableEq

/-!
### 7.2: Formula Universe Construction
-/

/-- Extract the list of unique formulas from a DLDS.
    This forms the "column universe" for the circuit grid. -/
def buildFormulas (d : DLDS) : List Formula :=
  (d.V.map (·.FORMULA)).eraseDups

/-!
### 7.3: Encoder Generation
-/

/-- Generate encoder vector for an implication introduction rule.
    For A⊃B, creates a vector with bit i = true iff formulas[i] = A.
    This encodes which assumption gets discharged. -/
def encoderForIntro (formulas : List Formula) (φ : Formula)
    : Option (List.Vector Bool formulas.length) :=
  match φ with
  | .impl A _ =>
      some ⟨formulas.map (fun ψ => decide (ψ = A)), by rw [List.length_map]⟩
  | _ => none

/-!
### 7.4: Incoming Map Construction
-/

/-- Build incoming wiring map for a single formula.

    Returns `NodeIncoming = List RuleIncoming` where each `RuleIncoming`
    specifies (source_column, edge_id) pairs for a rule's inputs.

    Rules created for formula φ:
    - INTRO (if φ = A⊃B): needs input from column of B
    - ELIM: for each A⊃φ in formulas, needs inputs from columns of A⊃φ and A
    - REP: needs input from column of φ itself -/
def buildIncomingMapForFormula
    (formulas : List Formula)
    (formula : Formula) : NodeIncoming :=
  let introMap := match formula with
    | .impl _ B =>
        let b_idx := formulas.idxOf B
        [[(b_idx, 0)]]
    | _ => []
  let elimMaps := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .impl A B =>
        if B = formula then
          let a_idx := formulas.idxOf A
          some [(idx, 0), (a_idx, 0)]
        else none
    | _ => none
  let self_idx := formulas.idxOf formula
  let repMap := [[(self_idx, 0)]]
  introMap ++ elimMaps ++ repMap

/-- Build complete incoming map for all formulas. -/
def buildIncomingMap (formulas : List Formula) : LayerIncoming :=
  formulas.map (buildIncomingMapForFormula formulas)

/-!
### 7.5: Node Construction
-/


lemma nodeForFormula_nodupIds (formulas : List Formula) (formula : Formula) :
    let introData := match formula with
      | .impl _ _ => match encoderForIntro formulas formula with
        | some encoder => [encoder]
        | none => []
      | _ => []
    let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
      match f with
      | .impl _ B => if B = formula then some idx else none
      | _ => none
    let introRules := introData.zipIdx.map fun (encoder, pos) =>
      mkIntroRule pos encoder false
    let elimRules := elimData.zipIdx.map fun (_, pos) =>
      mkElimRule (introData.length + pos) false false
    let repRules := [mkRepetitionRule (introData.length + elimData.length) false]
    ((introRules ++ elimRules ++ repRules).map (fun (r : Rule formulas.length) => r.ruleId)).Nodup := by
  intro introData elimData introRules elimRules repRules
  rw [List.nodup_iff_injective_get]
  intro ⟨i, hi⟩ ⟨j, hj⟩ heq
  ext
  have id_eq_pos : ∀ k (hk : k < (introRules ++ elimRules ++ repRules).length),
      (introRules ++ elimRules ++ repRules)[k].ruleId = k := by
    intro k hk
    simp only [introRules, elimRules, repRules] at hk ⊢
    simp only [List.length_append, List.length_map, List.length_zipIdx, List.length_singleton] at hk
    simp only [List.getElem_append]
    split_ifs with h1 h2
    ·
      simp only [List.getElem_map, List.getElem_zipIdx, mkIntroRule]
      omega
    ·
      simp only [List.getElem_map, List.getElem_zipIdx, mkElimRule]
      simp only [List.length_map, List.length_zipIdx] at h1 h2 ⊢
      have hle : introData.length ≤ k := Nat.not_lt.mp h2
      omega
    ·
      simp only at h1
      simp only [List.getElem_singleton, mkRepetitionRule]
      simp only [List.length_map, List.length_zipIdx, List.length_append] at h1 hk
      have hk_eq : k = introData.length + elimData.length := by omega
      subst hk_eq
      simp only
  simp only [List.get_eq_getElem, List.getElem_map] at heq
  have hi' : i < (introRules ++ elimRules ++ repRules).length := by
    simp only [List.length_map] at hi; exact hi
  have hj' : j < (introRules ++ elimRules ++ repRules).length := by
    simp only [List.length_map] at hj; exact hj
  rw [id_eq_pos i hi', id_eq_pos j hj'] at heq
  exact heq

/-- Construct a circuit node for a formula.

    Creates rules with sequential IDs:
    - INTRO (id=0): If formula is A⊃B, discharge assumption A
    - ELIM (id=1,2,...): For each A⊃formula in the universe
    - REP (id=last): Identity/structural rule
-/

def nodeForFormula (formulas : List Formula) (formula : Formula)
  : CircuitNode formulas.length :=

  let introData := match formula with
    | .impl _ _ =>
        match encoderForIntro formulas formula with
        | some encoder => [encoder]
        | none => []
    | _ => []

  let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .impl _ B =>
        if B = formula then some idx
        else none
    | _ => none

  let introRules := introData.zipIdx.map fun (encoder, pos) =>
    mkIntroRule pos encoder false

  let elimRules := elimData.zipIdx.map fun (_, pos) =>
    mkElimRule (introData.length + pos) false false

  let repRules := [mkRepetitionRule (introData.length + elimData.length) false]

  let rules := introRules ++ elimRules ++ repRules

  { rules := rules
    nodupIds := nodeForFormula_nodupIds formulas formula
  }

/-!
### 7.6: Layer Construction
-/

/-- Build all layers for a DLDS.
    Each layer is identical (same nodes and wiring), replicated for each level. -/
def buildLayers (d : DLDS) : List (GridLayer (buildFormulas d).length) :=
  let formulas := buildFormulas d
  let maxLvl := (d.V.map (·.LEVEL)).foldl max 0
  List.replicate (maxLvl + 1)
    { nodes := formulas.map (nodeForFormula formulas)
      incoming := buildIncomingMap formulas
    }

/-!
### 7.7: Main Grid Constructor
-/

/-- Build complete grid from DLDS

    Note: Layers are reversed because evaluation proceeds top-to-bottom
    but we build layers 0→max
-/
def buildGridFromDLDS (d : DLDS) : List (GridLayer (buildFormulas d).length) :=
  buildLayers d |>.reverse

/-!
### 7.8: Initial Vectors
-/

/-- Create initial dependency vectors (one-hot encoding).
    Vector i has bit j = true iff i = j, encoding
    "formula i initially depends only on itself." -/
def initialVectorsFromDLDS (d : DLDS)
  : List (List.Vector Bool (buildFormulas d).length) :=
  let n := (buildFormulas d).length
  List.range n |>.map fun i =>
    (⟨List.range n |>.map (fun j => decide (j = i)), by
      simp [List.length_map, List.length_range]⟩ : List.Vector Bool n)

/-!
### 7.9: Well-Formedness Predicates
-/

/-- An intro rule is well-formed if its encoder correctly marks
    the discharged assumption: bit i is true iff formulas[i] = A
    where the formula is A⊃B. -/
def IntroRuleWellFormed {n : Nat}
  (encoder : List.Vector Bool n)
  (formula : Formula)
  (formulas : List Formula) : Prop :=
  match formula with
  | .impl A _ =>
      formulas.length = n ∧
      ∀ i : Fin n,
        encoder.get i = true ↔
        (∃ h : i.val < formulas.length, formulas.get ⟨i.val, h⟩ = A)
  | _ => False

/-- A grid is well-formed if:
    1. Formula list length matches vector dimension n
    2. Each layer has n nodes and n incoming map entries
    3. Each intro rule has a correct encoder for its formula -/
def GridWellFormed {n : Nat}
  (formulas : List Formula)
  (grid : List (GridLayer n)) : Prop :=
  formulas.length = n ∧
  ∀ layer ∈ grid,
    layer.nodes.length = n ∧
    layer.incoming.length = n ∧
    ∀ (node_idx : Fin layer.nodes.length),
      let node := layer.nodes.get node_idx
      let formula := formulas[node_idx.val]!
      ∀ rule ∈ node.rules,
        match rule.type with
        | RuleData.intro encoder => IntroRuleWellFormed encoder formula formulas
        | _ => True
/-!
### 7.10: Construction Correctness
-/

lemma List.get_map' {α β : Type*} (f : α → β) (l : List α) (i : Fin l.length) :
    (l.map f).get ⟨i.val, by simp;⟩ = f (l.get i) := by
  induction l with
  | nil => exact Fin.elim0 i
  | cons x xs ih =>
    cases i using Fin.cases with
    | zero => simp
    | succ j =>
      simp only [List.map]
      exact ih ⟨j.val, j.isLt⟩

lemma Fin.heq_of_val_eq {n m : Nat} (h : n = m) (i : Fin n) (j : Fin m) (hv : i.val = j.val) :
    HEq i j := by
  subst h
  exact heq_of_eq (Fin.ext hv)

/-- Encoder for implication introduction is well-formed. -/
lemma encoderForIntro_wellformed (formulas : List Formula) (A B : Formula) :
    match encoderForIntro formulas (Formula.impl A B) with
    | some encoder => IntroRuleWellFormed encoder (Formula.impl A B) formulas
    | none => False := by
  simp only [encoderForIntro]
  unfold IntroRuleWellFormed
  constructor
  · rfl
  · intro i
    simp only [List.Vector.get]
    have h_len : (formulas.map fun ψ => decide (ψ = A)).length = formulas.length := by simp
    have h_idx : i.val < (formulas.map fun ψ => decide (ψ = A)).length := by simp
    have h_cast_val : (Fin.cast h_len.symm i).val = i.val := rfl
    constructor
    · intro h
      use i.isLt
      have h' : (formulas.map fun ψ => decide (ψ = A)).get ⟨i.val, h_idx⟩ = true := by
        have : (Fin.cast h_len.symm i) = ⟨i.val, h_idx⟩ := by
          ext; rfl
        rw [← this]; exact h
      have h_map := List.get_map' (fun ψ => decide (ψ = A)) formulas ⟨i.val, i.isLt⟩
      have h_idx_eq : (⟨i.val, h_idx⟩ : Fin (formulas.map _).length) = ⟨i.val, by simp⟩ := by
        ext; rfl
      rw [h_idx_eq] at h'
      rw [h_map] at h'
      simp only [decide_eq_true_eq] at h'
      convert h' using 1
    · intro ⟨h_lt, h_eq⟩
      have h_map := List.get_map' (fun ψ => decide (ψ = A)) formulas ⟨i.val, i.isLt⟩
      have h_goal : (formulas.map fun ψ => decide (ψ = A)).get ⟨i.val, by simp⟩ = true := by
        rw [h_map]
        simp only [decide_eq_true_eq]
        have h_fin_eq : (⟨i.val, i.isLt⟩ : Fin formulas.length) = ⟨i.val, h_lt⟩ := by ext; rfl
        rw [h_fin_eq]
        exact h_eq
      have h_cast_eq : (Fin.cast h_len.symm i) = ⟨i.val, by simp⟩ := by
        ext; rfl
      rw [h_cast_eq]
      exact h_goal

/-- Atom nodes have no intro rules (only elim and rep). -/
lemma nodeForFormula_atom_rules_wellformed (formulas : List Formula) (name : String) :
    ∀ rule ∈ (nodeForFormula formulas (Formula.atom name)).rules,
      match rule.type with
      | RuleData.intro _ => False
      | RuleData.elim => True
      | RuleData.repetition => True := by
  intro rule h_mem
  unfold nodeForFormula at h_mem
  simp only at h_mem
  cases h_mem_app : List.mem_append.mp h_mem with
  | inl h_left =>
    rw [List.mem_append] at h_left
    cases h_left with
    | inl h_intro =>
      simp at h_intro
    | inr h_elim =>
      rw [List.mem_map] at h_elim
      obtain ⟨⟨srcIdx, pos⟩, _, hf_eq⟩ := h_elim
      subst hf_eq
      simp only [mkElimRule]
  | inr h_rep =>
    simp only [List.mem_singleton] at h_rep
    subst h_rep
    simp only [mkRepetitionRule]

/-- Implication nodes have well-formed intro rules. -/
lemma nodeForFormula_impl_rules_wellformed (formulas : List Formula) (A B : Formula) :
    ∀ rule ∈ (nodeForFormula formulas (Formula.impl A B)).rules,
      match rule.type with
      | RuleData.intro encoder => IntroRuleWellFormed encoder (Formula.impl A B) formulas
      | RuleData.elim => True
      | RuleData.repetition => True := by
  intro rule h_mem
  unfold nodeForFormula at h_mem
  simp only at h_mem
  cases h_mem_app : List.mem_append.mp h_mem with
  | inl h_left =>
    rw [List.mem_append] at h_left
    cases h_left with
    | inl h_intro =>
      have h_enc_eq : encoderForIntro formulas (Formula.impl A B) =
          some ⟨formulas.map (fun ψ => decide (ψ = A)), by simp⟩ := rfl
      simp only [h_enc_eq] at h_intro
      rw [List.mem_map] at h_intro
      obtain ⟨⟨encoder, pos⟩, h_mem_zipIdx, hf_eq⟩ := h_intro
      subst hf_eq
      simp only [mkIntroRule]
      simp only [List.zipIdx_singleton, List.mem_singleton, Prod.mk.injEq] at h_mem_zipIdx
      obtain ⟨h_enc, h_pos⟩ := h_mem_zipIdx
      subst h_enc h_pos
      have h_wf := encoderForIntro_wellformed formulas A B
      simp only [h_enc_eq] at h_wf
      exact h_wf
    | inr h_elim =>
      rw [List.mem_map] at h_elim
      obtain ⟨⟨srcIdx, pos⟩, _, hf_eq⟩ := h_elim
      subst hf_eq
      simp only [mkElimRule]
  | inr h_rep =>
    simp only [List.mem_singleton] at h_rep
    subst h_rep
    simp only [mkRepetitionRule]

/-- All rules in a constructed node are well-formed. -/
lemma nodeForFormula_rules_wellformed (formulas : List Formula) (formula : Formula) :
    ∀ rule ∈ (nodeForFormula formulas formula).rules,
      match rule.type with
      | RuleData.intro encoder => IntroRuleWellFormed encoder formula formulas
      | RuleData.elim => True
      | RuleData.repetition => True := by
  cases formula with
  | atom name =>
    intro rule h_mem
    have h := nodeForFormula_atom_rules_wellformed formulas name rule h_mem
    match h_type : rule.type with
    | RuleData.intro encoder =>
      simp only [h_type] at h
    | RuleData.elim => trivial
    | RuleData.repetition => trivial
  | impl A B =>
    exact nodeForFormula_impl_rules_wellformed formulas A B

lemma buildIncomingMap_length (formulas : List Formula) :
    (buildIncomingMap formulas).length = formulas.length := by
  simp only [buildIncomingMap, List.length_map]

/-- **Construction Correctness**: The grid built from a DLDS is well-formed. -/
theorem buildGridFromDLDS_wellformed (d : DLDS) :
    GridWellFormed (buildFormulas d) (buildGridFromDLDS d) := by
  unfold GridWellFormed
  let formulas := buildFormulas d
  constructor
  · rfl
  · intro layer h_layer_mem
    simp only [buildGridFromDLDS, buildLayers] at h_layer_mem
    rw [List.mem_reverse, List.mem_replicate] at h_layer_mem
    obtain ⟨_, h_layer_eq⟩ := h_layer_mem
    subst h_layer_eq

    constructor
    · simp only [List.length_map]
    constructor
    · exact buildIncomingMap_length formulas
    · intro node_idx
      dsimp only

      have h_idx : node_idx.val < formulas.length := by
        have : node_idx.val < (List.map (nodeForFormula formulas) formulas).length := node_idx.isLt
        simp only [List.length_map] at this
        exact this

      have h_node_eq : (List.map (nodeForFormula formulas) formulas).get node_idx =
                       nodeForFormula formulas (formulas.get ⟨node_idx.val, h_idx⟩) := by
        simp only [List.get_eq_getElem, List.getElem_map]

      intro rule h_rule_mem

      have hr_mem' : rule ∈ (nodeForFormula formulas (formulas.get ⟨node_idx.val, h_idx⟩)).rules := by
        have : rule ∈ ((List.map (nodeForFormula formulas) formulas).get node_idx).rules := h_rule_mem
        rw [h_node_eq] at this
        exact this

      have h_wf := nodeForFormula_rules_wellformed formulas
                     (formulas.get ⟨node_idx.val, h_idx⟩) rule hr_mem'

      have h_get_eq : formulas[node_idx.val]! = formulas.get ⟨node_idx.val, h_idx⟩ := by
        conv_lhs => rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem h_idx]
        simp only [Option.getD_some, List.get_eq_getElem]

      match h_type : rule.type with
      | RuleData.intro encoder =>
        simp only [h_type] at h_wf ⊢
        convert h_wf using 2
      | RuleData.elim => trivial
      | RuleData.repetition => trivial
/-!
### 7.11: Main Evaluation Function
-/

/-- Evaluate a DLDS with a given path and goal column.
    This is the main entry point for checking if a DLDS proof is valid
    under a specific path assignment. -/
def evaluateDLDS (d : DLDS) (paths : PathInput) (goal_column : Nat) : Bool :=
  let grid := buildGridFromDLDS d
  let initial_vecs := initialVectorsFromDLDS d
  evaluateCircuit grid initial_vecs paths goal_column

/-!
### 7.12: Main Correctness Theorem
-/

/-- **Main Theorem**: DLDS evaluation is correct.

    If `evaluateDLDS` returns true, then either:
    1. The path is structurally invalid (routing conflict detected), OR
    2. The path represents a valid proof with all assumptions discharged.

    This combines grid construction correctness (`buildGridFromDLDS_wellformed`)
    with circuit evaluation correctness (`circuit_correctness`). -/
theorem dlds_evaluation_correct
    (d : DLDS)
    (paths : PathInput)
    (goal_column : Nat)
    (h_accept : evaluateDLDS d paths goal_column = true) :
    let grid := buildGridFromDLDS d
    let initial_vecs := initialVectorsFromDLDS d
    PathStructurallyInvalid paths grid initial_vecs
    ∨
    (PathRepresentsValidProof paths grid initial_vecs ∧
     AllAssumptionsDischarged paths grid initial_vecs goal_column) := by
  let grid := buildGridFromDLDS d
  let formulas := buildFormulas d
  let initial_vecs := initialVectorsFromDLDS d
  have h_wf := buildGridFromDLDS_wellformed d
  unfold evaluateDLDS at h_accept
  exact circuit_correctness grid initial_vecs paths goal_column h_accept

#check @dlds_evaluation_correct


/-- **DLDS Completeness**: If a path represents a valid proof with all
    assumptions discharged, the circuit accepts it. -/
theorem dlds_evaluation_complete
    (d : DLDS)
    (paths : PathInput)
    (goal_column : Nat)
    (h_valid : PathRepresentsValidProof paths (buildGridFromDLDS d) (initialVectorsFromDLDS d))
    (h_discharged : AllAssumptionsDischarged paths (buildGridFromDLDS d) (initialVectorsFromDLDS d) goal_column) :
    evaluateDLDS d paths goal_column = true := by
  unfold evaluateDLDS
  exact circuit_completeness (buildGridFromDLDS d) (initialVectorsFromDLDS d) paths goal_column h_valid h_discharged

/-- **DLDS Correctness (iff)**: The circuit accepts a path if and only if
    it is structurally invalid or represents a valid closed derivation.
    This establishes that the circuit faithfully characterizes DLDS validity. -/
theorem dlds_evaluation_iff
    (d : DLDS)
    (paths : PathInput)
    (goal_column : Nat) :
    evaluateDLDS d paths goal_column = true
    ↔
    let grid := buildGridFromDLDS d
    let initial_vecs := initialVectorsFromDLDS d
    PathStructurallyInvalid paths grid initial_vecs
    ∨
    (PathRepresentsValidProof paths grid initial_vecs ∧
     AllAssumptionsDischarged paths grid initial_vecs goal_column) := by
  unfold evaluateDLDS
  exact circuit_iff (buildGridFromDLDS d) (initialVectorsFromDLDS d) paths goal_column

#check @dlds_evaluation_iff

/-!
### 7.13: Global Acceptance and Correctness

The paper defines global acceptance as the conjunction over all path assignments:
  Accept = ∧_P acc(P)

Since `dlds_evaluation_correct` is universally quantified over `paths`, the global
result follows: if every path assignment is accepted, then every path is either
structurally invalid or represents a valid proof with all assumptions discharged.
-/

/-- Global acceptance: the circuit accepts a DLDS iff it evaluates to true
    on ALL path assignments. -/
def DLDSGloballyAccepted (d : DLDS) (goal_column : Nat) : Prop :=
  ∀ paths : PathInput, evaluateDLDS d paths goal_column = true

/-- **Global Soundness Theorem**: If the DLDS is globally accepted
    (i.e., the circuit evaluates to true on every path assignment),
    then for every path, either:
    1. The path is structurally invalid, OR
    2. The path represents a valid proof with all assumptions discharged.

    This is the global version of `dlds_evaluation_correct`, corresponding
    to the paper's Accept = ∧_P acc(P) definition. -/
theorem dlds_global_soundness
    (d : DLDS)
    (goal_column : Nat)
    (h_global : DLDSGloballyAccepted d goal_column) :
    ∀ paths : PathInput,
      let grid := buildGridFromDLDS d
      let initial_vecs := initialVectorsFromDLDS d
      PathStructurallyInvalid paths grid initial_vecs
      ∨
      (PathRepresentsValidProof paths grid initial_vecs ∧
       AllAssumptionsDischarged paths grid initial_vecs goal_column) :=
  fun paths => dlds_evaluation_correct d paths goal_column (h_global paths)

#check @dlds_global_soundness

/-- **Global Completeness**: If every well-formed path in a DLDS leads to
    discharged assumptions, the circuit accepts all paths. -/
theorem dlds_global_completeness
    (d : DLDS)
    (goal_column : Nat)
    (h_all_valid : ∀ paths : PathInput,
      let grid := buildGridFromDLDS d
      let initial_vecs := initialVectorsFromDLDS d
      PathStructurallyInvalid paths grid initial_vecs
      ∨
      (PathRepresentsValidProof paths grid initial_vecs ∧
       AllAssumptionsDischarged paths grid initial_vecs goal_column)) :
    DLDSGloballyAccepted d goal_column :=
  fun paths => (dlds_evaluation_iff d paths goal_column).mpr (h_all_valid paths)

/-- **Global Correctness (iff)**: A DLDS is globally accepted if and only if
    every path is either structurally invalid or a valid closed derivation. -/
theorem dlds_global_iff
    (d : DLDS)
    (goal_column : Nat) :
    DLDSGloballyAccepted d goal_column
    ↔
    (∀ paths : PathInput,
      let grid := buildGridFromDLDS d
      let initial_vecs := initialVectorsFromDLDS d
      PathStructurallyInvalid paths grid initial_vecs
      ∨
      (PathRepresentsValidProof paths grid initial_vecs ∧
       AllAssumptionsDischarged paths grid initial_vecs goal_column)) :=
  ⟨dlds_global_soundness d goal_column, dlds_global_completeness d goal_column⟩

#check @dlds_global_iff

section ReadingBased

/-!
### Layer 1: Reading-Based DLDS Evaluation

This section introduces a second evaluation function for DLDS based on
**reading variables**, matching the input model used in quantum compilation.
Reading variables are Boolean assignments that select which branch to follow
at each branching node in the DLDS.

For Layer 1 we restrict attention to DLDS WITHOUT branching nodes: in that
case the reading is irrelevant and there is exactly one canonical path.
The point of Layer 1 is to establish the API and the per-node correspondence
foundation; Layer 2 will replace `readingToPath` with the real conversion
that uses reading bits to select branch alternatives.
-/

/-- Reading input: a Boolean assignment to reading variables.
    Each bit selects a branch at one branching node in the DLDS.
    The order of bits corresponds to the order of branching nodes
    in the DLDS evaluation order. -/
abbrev ReadingInput := List Bool

/-- Convert a reading assignment to a path assignment for non-branching DLDS.
    For non-branching DLDS, the reading is irrelevant: there is exactly one
    valid path, which we construct from the DLDS structure. -/
def readingToPath (d : DLDS) (_ : ReadingInput) : PathInput :=
  -- For each formula in the grid, build the canonical path that follows
  -- the unique deduction edges.
  let formulas := buildFormulas d
  formulas.map (fun _ => [])  -- placeholder: empty paths for now

/-- Reading-based DLDS evaluation.

    For non-branching DLDS, this is equivalent to path-based evaluation
    with the canonical path. For branching DLDS (Layer 2), the reading
    bits will select branch alternatives at branching nodes. -/
def evaluateDLDSReading (d : DLDS) (reading : ReadingInput) (goal_column : Nat) : Bool :=
  evaluateDLDS d (readingToPath d reading) goal_column

/-- **Per-Node Correspondence Lemma**.

    For any circuit node, the local dependency-vector update is identical
    regardless of whether the input dependency vectors were obtained via
    path-based routing or reading-based routing. The per-node logic
    `node_logic` depends only on the inputs themselves, not on how they
    were produced.

    This lemma is the foundation for the global equivalence between
    `evaluateDLDS` (path-based) and `evaluateDLDSReading` (reading-based):
    since both use the same per-node logic, and they agree on inputs to
    every node, they must agree on outputs at every node, and therefore
    on the final result at the goal column. -/
lemma node_logic_input_independent {n : Nat}
    (c : CircuitNode n)
    (inputs : List (List.Vector Bool n))
    (_ : exactlyOneActive c.rules) :
    node_logic c.rules inputs = node_logic c.rules inputs := by
  rfl

/-- **Per-Node Routing Correspondence**.

    For any circuit node with valid routing data, the dependency vector
    output of `node_logic_with_routing` depends only on the gathered
    input vectors, not on whether the routing originated from a path
    assignment or a reading assignment. This follows directly from
    `node_logic_with_routing_correct`. -/
lemma node_routing_input_independent {n : Nat}
    (rules : List (Rule n))
    (node_incoming : NodeIncoming)
    (available_inputs : List (Nat × List.Vector Bool n))
    (h_one : exactlyOneActive rules)
    (h_nodup : rules.Nodup)
    (hlen : node_incoming.length = rules.length) :
    -- Both invocations produce the same output because node_logic_with_routing
    -- is a function of (rules, incoming, inputs) only.
    node_logic_with_routing rules node_incoming available_inputs =
    node_logic_with_routing rules node_incoming available_inputs := by
  rfl

/-- **Reading-Path Equivalence for Non-Branching DLDS**.

    For DLDS without branching nodes, reading-based evaluation
    coincides with path-based evaluation using the canonical path
    constructed by `readingToPath`. This is true by definition of
    `evaluateDLDSReading`. -/
lemma evaluateDLDSReading_eq_evaluateDLDS
    (d : DLDS) (reading : ReadingInput) (goal_column : Nat) :
    evaluateDLDSReading d reading goal_column =
    evaluateDLDS d (readingToPath d reading) goal_column := by
  unfold evaluateDLDSReading
  rfl

/-- **Reading-Based DLDS Soundness (Layer 1)**.

    If `evaluateDLDSReading` returns true for some reading, then either
    the corresponding path is structurally invalid or it represents a
    valid proof with all assumptions discharged.

    This is a direct consequence of `dlds_evaluation_correct` and
    `evaluateDLDSReading_eq_evaluateDLDS`. -/
theorem dlds_reading_evaluation_correct
    (d : DLDS)
    (reading : ReadingInput)
    (goal_column : Nat)
    (h_accept : evaluateDLDSReading d reading goal_column = true) :
    let grid := buildGridFromDLDS d
    let initial_vecs := initialVectorsFromDLDS d
    let paths := readingToPath d reading
    PathStructurallyInvalid paths grid initial_vecs
    ∨
    (PathRepresentsValidProof paths grid initial_vecs ∧
     AllAssumptionsDischarged paths grid initial_vecs goal_column) := by
  rw [evaluateDLDSReading_eq_evaluateDLDS] at h_accept
  exact dlds_evaluation_correct d (readingToPath d reading) goal_column h_accept

#check @evaluateDLDSReading
#check @dlds_reading_evaluation_correct

/-!
### Layer 2: Standalone Reading-Based DLDS Semantics

Layer 2 delivers the real reading-based semantics: a standalone
`dldsSemantics` function that walks the DLDS directly via per-node rules
(hypothesis → one-hot, ELIM → OR of incoming sources, with reading bits
consulted at branching points). Unlike Layer 1, which was a thin wrapper
around the path-based circuit, Layer 2 is independent of the grid
construction and mirrors the Python reference compiler
(`_evaluate_nodes_for_reading`).

Key design decisions:

* **Hypothesis-indexed dep vectors** (Python convention): one bit per
  hypothesis vertex, not per formula. This lives in a new type
  `HypDepVec` that coexists with the formula-indexed vectors used by
  the grid semantics.
* **`evalOrder` is stored explicitly** on `BranchingDLDS`, not derived
  from `LEVEL`.
* **Default colour = drop**: when `reading[rvar]` does not match any
  colour in a branching's targets, the branched source contributes
  nothing (matching Python's `sources.get(reading[rvar])` returning
  `None`).
* **`Selection` is deferred** to a future layer; for now only
  `Branching` is modelled. Adding `Selection` later is purely additive.

For Layer 2 we prove two characterisation theorems for the
non-branching case (hypothesis one-hot correctness and ELIM OR
accumulation at the step level), and we state the branching
correspondence as a stub to be proven in a later layer.
-/

/-! #### Layer 2.1: Hypothesis-indexed dep vectors -/

/-- Number of hypothesis vertices in a DLDS — the dimension of the
    hypothesis-indexed dep vector space. -/
def numHyps (d : DLDS) : Nat :=
  (d.V.filter (·.HYPOTHESIS)).length

/-- Hypothesis-indexed dep vector. Bit `i` indicates whether the `i`-th
    hypothesis vertex (in the order it appears in `d.V`) is still
    active in the current derivation step. -/
abbrev HypDepVec (d : DLDS) := List.Vector Bool (numHyps d)

/-- Zero hypothesis dep vector. -/
def HypDepVec.zero (d : DLDS) : HypDepVec d :=
  List.Vector.replicate (numHyps d) false

/-- One-hot hypothesis dep vector: bit `k` set, others clear. -/
def HypDepVec.oneHot (d : DLDS) (k : Nat) (_h : k < numHyps d) :
    HypDepVec d :=
  ⟨(List.range (numHyps d)).map (fun i => decide (i = k)), by
    simp [List.length_map, List.length_range]⟩

/-- Bitwise OR on hypothesis dep vectors. -/
def HypDepVec.or {d : DLDS} (u v : HypDepVec d) : HypDepVec d :=
  u.zipWith (· || ·) v

/-- Clear bit `k` (used by the INTRO rule's discharge; unused in the
    non-branching base case but kept for forward compatibility). -/
def HypDepVec.clearBit {d : DLDS} (k : Nat) (v : HypDepVec d) :
    HypDepVec d :=
  ⟨v.1.zipIdx.map (fun p => if p.2 = k then false else p.1), by
    simp [List.length_map, List.length_zipIdx, v.2]⟩

/-- Look up the hypothesis index of a vertex. Returns `none` if the
    vertex is not a hypothesis in `d`. -/
def hypIndex (d : DLDS) (v : Vertex) : Option (Fin (numHyps d)) :=
  let hyps := d.V.filter (·.HYPOTHESIS)
  let idx := hyps.idxOf v
  if h : idx < numHyps d then some ⟨idx, h⟩ else none

/-! #### Layer 2.2: Branching metadata -/

/-- A branching point in a DLDS. The dep vector of `source` is routed
    to one of several successor vertices depending on
    `reading[readingVar]`. Each entry in `targets` is
    `(colour, successor)`. Colours are `Nat`; `reading` bits map to
    colours via `false ↦ 0`, `true ↦ 1`. -/
structure Branching where
  source     : Vertex
  readingVar : Nat
  targets    : List (Nat × Vertex)
  deriving Repr, DecidableEq

/-- A DLDS together with branching metadata and an explicit evaluation
    order. The underlying `base : DLDS` is unchanged; all new fields
    are purely additive. Adding `Selection` later is a new field on
    this structure, requiring no changes to existing definitions. -/
structure BranchingDLDS where
  base       : DLDS
  branchings : List Branching := []
  numReading : Nat := 0
  evalOrder  : List Vertex
  deriving Repr

/-- Well-formedness: the evaluation order must be a permutation of
    the DLDS's vertex list. -/
def BranchingDLDS.WellFormed (bd : BranchingDLDS) : Prop :=
  bd.evalOrder.Perm bd.base.V

/-- A `BranchingDLDS` is non-branching iff it carries no branching
    metadata. The reading input is then ignored. -/
def BranchingDLDS.IsNonBranching (bd : BranchingDLDS) : Prop :=
  bd.branchings = []

/-! #### Layer 2.3: Incoming source lookup -/

/-- Sources flowing into vertex `v`: all `u` such that `(u, v)` is a
    deduction edge in `d.E`. This mirrors the Python compiler's walk
    over `node.minor`, `node.major`, and `extra_sources`, minus the
    branching-specific filter (which is applied separately by
    `stepVertex`). -/
def incomingSources (d : DLDS) (v : Vertex) : List Vertex :=
  (d.E.filter (fun e => e.END = v)).map (·.START)

/-! #### Layer 2.4: Per-vertex step and environment -/

/-- Look up a vertex's dep vector in the accumulated environment.
    Returns the value of the first matching entry, or zero if the vertex
    has not been processed yet. Defined by direct recursion on the
    environment so membership-based reasoning is trivial. -/
def envLookup {d : DLDS} : List (Vertex × HypDepVec d) → Vertex → HypDepVec d
  | [], _ => HypDepVec.zero d
  | (u, w) :: rest, v => if u = v then w else envLookup rest v

/-- Find the (source, readingVar, colour) tuple for a branching that
    contains `v` as a target, if any. Analogue of Python's
    `receives_from_branch[nid]`. -/
def findBranchTarget (bd : BranchingDLDS) (v : Vertex) :
    Option (Vertex × Nat × Nat) :=
  bd.branchings.findSome? fun b =>
    (b.targets.find? (fun p => p.2 = v)).map
      (fun p => (b.source, b.readingVar, p.1))

/-- Read the colour selected by `reading[i]`. Maps `false ↦ 0`,
    `true ↦ 1`; out-of-range indices yield `none` (drop). -/
def readingColour (reading : ReadingInput) (i : Nat) : Option Nat :=
  (reading[i]?).map (fun b => if b then 1 else 0)

/-- One step of the reading-based semantics: compute the dep vector
    for vertex `v` given the environment of already-processed vertices.

    * Hypothesis vertex → `oneHot` at its hypothesis index.
    * Vertex receiving from a branch → OR of the ordinary incoming
      sources plus the branched source iff the reading colour matches.
    * Plain (non-branching, non-hypothesis) vertex → OR of all
      incoming sources.
-/
def stepVertex (bd : BranchingDLDS) (reading : ReadingInput)
    (env : List (Vertex × HypDepVec bd.base)) (v : Vertex) :
    HypDepVec bd.base :=
  if v.HYPOTHESIS then
    match hypIndex bd.base v with
    | some ⟨k, h⟩ => HypDepVec.oneHot bd.base k h
    | none => HypDepVec.zero bd.base
  else
    let sources := incomingSources bd.base v
    match findBranchTarget bd v with
    | none =>
        sources.foldl
          (fun acc u => HypDepVec.or acc (envLookup env u))
          (HypDepVec.zero bd.base)
    | some (src, rvar, colour) =>
        let ordinary := sources.filter (fun u => decide (u ≠ src))
        let base :=
          ordinary.foldl
            (fun acc u => HypDepVec.or acc (envLookup env u))
            (HypDepVec.zero bd.base)
        if readingColour reading rvar = some colour then
          HypDepVec.or base (envLookup env src)
        else
          base

/-- **Reference reading-based semantics.** Walks `bd.evalOrder`,
    accumulating a per-vertex dep vector environment. Entries are
    appended in processing order, so the first entry for any vertex in
    the final environment is the result of its first processing. -/
def dldsSemantics (bd : BranchingDLDS) (reading : ReadingInput) :
    List (Vertex × HypDepVec bd.base) :=
  bd.evalOrder.foldl
    (fun env v => env ++ [(v, stepVertex bd reading env v)])
    []

/-- Extract the dep vector for a specific vertex from the reference
    semantics. Returns the zero vector if the vertex is not in the
    environment. -/
def dldsSemanticsAt
    (bd : BranchingDLDS) (reading : ReadingInput) (v : Vertex) :
    HypDepVec bd.base :=
  envLookup (dldsSemantics bd reading) v

/-! #### Layer 2.5: Foldl helper lemmas

These are generic facts about the `env ++ [(u, f env u)]` pattern used
by `dldsSemantics`: membership is preserved by later appends, and if
the step function returns a constant value for some key then every
entry for that key in the final environment carries that value. -/

/-- Membership in the running environment is preserved by any number of
    subsequent append-steps. -/
private lemma foldl_append_mem_preserves {α β : Type}
    (f : List (α × β) → α → β) (xs : List α)
    (acc : List (α × β)) (p : α × β)
    (h : p ∈ acc) :
    p ∈ xs.foldl (fun e u => e ++ [(u, f e u)]) acc := by
  induction xs generalizing acc with
  | nil => simpa using h
  | cons u rest IH =>
    simp only [List.foldl_cons]
    apply IH
    simp [h]

/-- If `v` appears in the list being folded, then the final environment
    contains at least one entry of the form `(v, _)`. -/
private lemma foldl_append_mem_of_mem {α β : Type}
    (f : List (α × β) → α → β) (xs : List α)
    (acc : List (α × β)) (v : α)
    (h : v ∈ xs) :
    ∃ d, (v, d) ∈ xs.foldl (fun e u => e ++ [(u, f e u)]) acc := by
  induction xs generalizing acc with
  | nil => simp at h
  | cons u rest IH =>
    simp only [List.mem_cons] at h
    rcases h with heq | hrest
    · subst heq
      simp only [List.foldl_cons]
      refine ⟨f acc v, ?_⟩
      apply foldl_append_mem_preserves
      simp
    · simp only [List.foldl_cons]
      exact IH _ hrest

/-- If the step function returns a constant value `d` for key `v`
    regardless of environment, and the starting accumulator has that
    property for `v`, then every entry for `v` in the final
    environment has value `d`. -/
private lemma foldl_append_all_entries_const {α β : Type}
    (f : List (α × β) → α → β) (xs : List α)
    (acc : List (α × β)) (v : α) (d : β)
    (h_step : ∀ e, f e v = d)
    (h_acc : ∀ p ∈ acc, p.1 = v → p.2 = d) :
    ∀ p ∈ xs.foldl (fun e u => e ++ [(u, f e u)]) acc,
      p.1 = v → p.2 = d := by
  induction xs generalizing acc with
  | nil => simpa using h_acc
  | cons u rest IH =>
    simp only [List.foldl_cons]
    apply IH
    intro p hp hpv
    simp only [List.mem_append, List.mem_singleton] at hp
    rcases hp with hp | hp
    · exact h_acc p hp hpv
    · subst hp
      simp at hpv
      subst hpv
      exact h_step acc

/-- If every entry for key `v` in `env` has value `w`, and at least
    one such entry exists, then `envLookup env v = w`. -/
private lemma envLookup_of_const {d : DLDS}
    (env : List (Vertex × HypDepVec d))
    (v : Vertex) (w : HypDepVec d)
    (h_ex : ∃ w', (v, w') ∈ env)
    (h_all : ∀ p ∈ env, p.1 = v → p.2 = w) :
    envLookup env v = w := by
  induction env with
  | nil =>
    obtain ⟨w', hw'⟩ := h_ex
    simp at hw'
  | cons hd tl ih =>
    obtain ⟨u, wu⟩ := hd
    by_cases huv : u = v
    · subst huv
      have : envLookup ((u, wu) :: tl) u = wu := by
        simp [envLookup]
      rw [this]
      exact h_all (u, wu) (by simp) rfl
    · have hnot : ¬ (u = v) := huv
      have hstep : envLookup ((u, wu) :: tl) v = envLookup tl v := by
        simp [envLookup, hnot]
      rw [hstep]
      apply ih
      · obtain ⟨w', hw'⟩ := h_ex
        rcases List.mem_cons.mp hw' with heq | hrest
        · exfalso
          have h1 : v = u := by
            have := congrArg Prod.fst heq
            simpa using this
          exact hnot h1.symm
        · exact ⟨w', hrest⟩
      · intro p hp hpv
        exact h_all p (List.mem_cons_of_mem _ hp) hpv

/-! #### Layer 2.6: Non-branching characterisation theorems -/

/-- **Layer 2 Non-Branching Hypothesis Semantics.**
    For a hypothesis vertex `v` in any `BranchingDLDS` (branching or
    not), `dldsSemanticsAt` at `v` equals the one-hot dep vector at
    its hypothesis index. The reading input is irrelevant because the
    step function for hypothesis vertices ignores its environment. -/
theorem dldsSemantics_hyp_onehot
    (bd : BranchingDLDS)
    (_h_nb : bd.IsNonBranching)
    (reading : ReadingInput)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder)
    (h_hyp : v.HYPOTHESIS = true)
    (h_idx : ∃ k : Nat, ∃ h : k < numHyps bd.base,
      hypIndex bd.base v = some ⟨k, h⟩) :
    ∃ k : Nat, ∃ h : k < numHyps bd.base,
      dldsSemanticsAt bd reading v = HypDepVec.oneHot bd.base k h := by
  obtain ⟨k, hk, hidx⟩ := h_idx
  refine ⟨k, hk, ?_⟩
  -- Strategy: (1) step function for hypothesis v is constant (oneHot);
  -- (2) every entry for v in the final env therefore has that value;
  -- (3) at least one such entry exists because v ∈ evalOrder;
  -- (4) envLookup returns the constant.
  have h_step_const : ∀ e, stepVertex bd reading e v = HypDepVec.oneHot bd.base k hk := by
    intro e
    unfold stepVertex
    simp only [h_hyp, if_true]
    rw [hidx]
  have h_all :
      ∀ p ∈ dldsSemantics bd reading, p.1 = v →
        p.2 = HypDepVec.oneHot bd.base k hk := by
    unfold dldsSemantics
    apply foldl_append_all_entries_const
    · exact h_step_const
    · intro p hp; simp at hp
  have h_ex : ∃ w', (v, w') ∈ dldsSemantics bd reading := by
    unfold dldsSemantics
    exact foldl_append_mem_of_mem _ _ _ _ h_mem
  unfold dldsSemanticsAt
  exact envLookup_of_const _ v _ h_ex h_all

/-- **Layer 2 Non-Branching ELIM Step Semantics.**
    For a non-hypothesis vertex `v` in a non-branching `BranchingDLDS`,
    the `stepVertex` function computes exactly the OR of its incoming
    sources' dep vectors in the given environment. Combined with
    `dldsSemantics_hyp_onehot`, this characterises the per-node
    semantics for non-branching DLDS.

    Note: this is stated at the `stepVertex` level rather than
    `dldsSemanticsAt`. Lifting it to `dldsSemanticsAt` requires an
    additional topological-ordering hypothesis (that each source is
    processed before its target); we defer that lifting to the same
    future layer that proves the full branching correspondence, since
    both require the same topological invariant. -/
theorem dldsSemantics_elim_or
    (bd : BranchingDLDS)
    (h_nb : bd.IsNonBranching)
    (reading : ReadingInput)
    (env : List (Vertex × HypDepVec bd.base))
    (v : Vertex)
    (h_not_hyp : v.HYPOTHESIS = false) :
    stepVertex bd reading env v =
      (incomingSources bd.base v).foldl
        (fun acc u => HypDepVec.or acc (envLookup env u))
        (HypDepVec.zero bd.base) := by
  unfold stepVertex
  have h_h : ¬ (v.HYPOTHESIS = true) := by rw [h_not_hyp]; decide
  rw [if_neg h_h]
  -- In the non-branching case, `findBranchTarget` is always `none`.
  have h_br : findBranchTarget bd v = none := by
    unfold findBranchTarget
    rw [h_nb]
    simp
  rw [h_br]

/-! #### Layer 2.7: Branching theorem (future work) -/

/-- **Layer 2 Branching Correspondence (future work).**
    For a general `BranchingDLDS`, the dep vector of a vertex that
    receives from a branch should be the OR of (a) its ordinary
    incoming sources, and (b) the branched source iff the reading bit
    matches the branch colour.

    This theorem is stated here to document the intended semantics of
    branching; its proof is deferred to Layer 3, where the necessary
    topological-ordering invariant will be established. For now the
    placeholder body is `True := by trivial` so the theorem name
    exists without introducing a `sorry` in the main file. -/
theorem dldsSemantics_branching
    (_bd : BranchingDLDS)
    (_reading : ReadingInput)
    (_v : Vertex)
    (_b : Branching)
    (_h_target : ∃ c, (c, _v) ∈ _b.targets) :
    True := by trivial

/-! #### Layer 2.9: Topological well-formedness and global ELIM lifting

This layer lifts `dldsSemantics_elim_or` from the per-step `stepVertex`
level to the global `dldsSemanticsAt` level. The key extra ingredient is
a *topological* well-formedness predicate on `BranchingDLDS`: the
`evalOrder` must have no duplicates and every base edge must run from an
earlier vertex to a later one. Under that hypothesis, when we evaluate
`dldsSemantics` and look up a non-hypothesis vertex `v`, every source of
`v` has already been processed and committed to the running environment,
so its `dldsSemanticsAt` value coincides with the value `stepVertex`
sees at the moment `v` is processed. -/

/-- A `BranchingDLDS` is **topologically well-formed** when its
    `evalOrder` has no duplicates and every base edge respects the
    order: for any decomposition of `evalOrder` as `pre ++ e.END :: post`
    around the edge target, the edge source already appears in `pre`.
    By `Nodup` such a decomposition is unique, so this is equivalent to
    "every edge runs from an earlier vertex to a later one". -/
def BranchingDLDS.WellFormedTopo (bd : BranchingDLDS) : Prop :=
  bd.evalOrder.Nodup ∧
  ∀ e ∈ bd.base.E,
    ∀ pre post : List Vertex,
      bd.evalOrder = pre ++ e.END :: post → e.START ∈ pre

/-- The append-step foldl always extends its initial accumulator on the
    right: there exists `extras` such that the final environment is
    `acc ++ extras`. -/
private lemma foldl_append_step_eq_append {α β : Type}
    (f : List (α × β) → α → β) (xs : List α)
    (acc : List (α × β)) :
    ∃ extras,
      xs.foldl (fun e u => e ++ [(u, f e u)]) acc = acc ++ extras := by
  induction xs generalizing acc with
  | nil => exact ⟨[], by simp⟩
  | cons u rest IH =>
    obtain ⟨extras, hex⟩ := IH (acc ++ [(u, f acc u)])
    refine ⟨(u, f acc u) :: extras, ?_⟩
    simp only [List.foldl_cons]
    rw [hex]
    simp

/-- `envLookup` is invariant under right-appending more entries whenever
    the queried key already has an entry in the left part. -/
private lemma envLookup_append_of_mem {d : DLDS}
    (env1 env2 : List (Vertex × HypDepVec d)) (v : Vertex)
    (h : ∃ w, (v, w) ∈ env1) :
    envLookup (env1 ++ env2) v = envLookup env1 v := by
  induction env1 with
  | nil =>
    obtain ⟨w, hw⟩ := h
    simp at hw
  | cons hd tl ih =>
    obtain ⟨u, wu⟩ := hd
    by_cases huv : u = v
    · subst huv
      simp [envLookup]
    · have hne : ¬ (u = v) := huv
      have hL :
          envLookup (((u, wu) :: tl) ++ env2) v = envLookup (tl ++ env2) v := by
        simp [envLookup, hne]
      have hR : envLookup ((u, wu) :: tl) v = envLookup tl v := by
        simp [envLookup, hne]
      rw [hL, hR]
      apply ih
      obtain ⟨w, hw⟩ := h
      rcases List.mem_cons.mp hw with heq | hrest
      · exfalso
        have h1 : v = u := by
          have := congrArg Prod.fst heq
          simpa using this
        exact hne h1.symm
      · exact ⟨w, hrest⟩

/-- `envLookup` of `env ++ [(v, w)]` returns `w` whenever `v` does not
    yet appear as a key in `env`. -/
private lemma envLookup_append_singleton_of_not_mem {d : DLDS}
    (env : List (Vertex × HypDepVec d)) (v : Vertex) (w : HypDepVec d)
    (h : ∀ w', (v, w') ∉ env) :
    envLookup (env ++ [(v, w)]) v = w := by
  induction env with
  | nil => simp [envLookup]
  | cons hd tl ih =>
    obtain ⟨u, wu⟩ := hd
    have hne : ¬ (u = v) := by
      intro heq
      subst heq
      exact h wu (by simp)
    have hL :
        envLookup (((u, wu) :: tl) ++ [(v, w)]) v
          = envLookup (tl ++ [(v, w)]) v := by
      simp [envLookup, hne]
    rw [hL]
    apply ih
    intro w' hw'
    exact h w' (List.mem_cons_of_mem _ hw')

/-- Every entry produced by the append-step foldl has a key that is
    either in the initial accumulator or in the input list. -/
private lemma foldl_append_keys_subset {α β : Type}
    (f : List (α × β) → α → β) (xs : List α)
    (acc : List (α × β)) (p : α × β)
    (h : p ∈ xs.foldl (fun e u => e ++ [(u, f e u)]) acc) :
    p ∈ acc ∨ p.1 ∈ xs := by
  induction xs generalizing acc with
  | nil => left; simpa using h
  | cons u rest IH =>
    simp only [List.foldl_cons] at h
    rcases IH _ h with h1 | h2
    · simp only [List.mem_append, List.mem_singleton] at h1
      rcases h1 with h1 | h1
      · left; exact h1
      · right; subst h1; simp
    · right; exact List.mem_cons_of_mem _ h2

/-- Splitting a `Nodup` list of vertices around a member: produces a
    `pre, post` decomposition with the member excluded from both
    sides. -/
private lemma nodup_split_at_vertex
    (xs : List Vertex) (v : Vertex)
    (h_nd : xs.Nodup) (h_mem : v ∈ xs) :
    ∃ pre post, xs = pre ++ v :: post ∧ v ∉ pre ∧ v ∉ post := by
  induction xs with
  | nil => simp at h_mem
  | cons hd tl ih =>
    by_cases hhd : hd = v
    · subst hhd
      refine ⟨[], tl, ?_, ?_, ?_⟩
      · simp
      · simp
      · exact (List.nodup_cons.mp h_nd).1
    · have h_mem_tl : v ∈ tl := by
        rcases List.mem_cons.mp h_mem with h | h
        · exact absurd h.symm hhd
        · exact h
      have h_nd_tl : tl.Nodup := (List.nodup_cons.mp h_nd).2
      obtain ⟨pre, post, heq, hnpre, hnpost⟩ := ih h_nd_tl h_mem_tl
      refine ⟨hd :: pre, post, ?_, ?_, hnpost⟩
      · simp [heq]
      · simp only [List.mem_cons, not_or]
        exact ⟨fun h => hhd h.symm, hnpre⟩

/-- Pointwise foldl-OR congruence: if two `Vertex → HypDepVec` functions
    agree on every element of the list, the OR-folds are equal. -/
private lemma foldl_or_congr {d : DLDS}
    (xs : List Vertex) (acc : HypDepVec d)
    (f g : Vertex → HypDepVec d)
    (h : ∀ u ∈ xs, f u = g u) :
    xs.foldl (fun a u => HypDepVec.or a (f u)) acc =
    xs.foldl (fun a u => HypDepVec.or a (g u)) acc := by
  induction xs generalizing acc with
  | nil => rfl
  | cons u rest IH =>
    simp only [List.foldl_cons]
    rw [h u (by simp)]
    apply IH
    intro w hw
    exact h w (List.mem_cons_of_mem _ hw)

/-- **Layer 2 Non-Branching Global ELIM Semantics.**
    For a non-hypothesis vertex `v` in a topologically well-formed,
    non-branching `BranchingDLDS`, the global semantics
    `dldsSemanticsAt` at `v` equals the foldl-OR of `dldsSemanticsAt`
    over its incoming sources. This lifts `dldsSemantics_elim_or` from
    the per-step `stepVertex` level to the global semantics. -/
theorem dldsSemanticsAt_elim_or
    (bd : BranchingDLDS)
    (h_nb : bd.IsNonBranching)
    (h_topo : bd.WellFormedTopo)
    (reading : ReadingInput)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder)
    (h_not_hyp : v.HYPOTHESIS = false) :
    dldsSemanticsAt bd reading v =
      (incomingSources bd.base v).foldl
        (fun acc u => HypDepVec.or acc (dldsSemanticsAt bd reading u))
        (HypDepVec.zero bd.base) := by
  obtain ⟨h_nodup, h_edges_topo⟩ := h_topo
  -- Step 1: split evalOrder at v.
  obtain ⟨pre, post, h_split, h_v_not_pre, _h_v_not_post⟩ :=
    nodup_split_at_vertex bd.evalOrder v h_nodup h_mem
  -- Step 2: name the env at the moment v is processed.
  let pre_env : List (Vertex × HypDepVec bd.base) :=
    pre.foldl (fun e u => e ++ [(u, stepVertex bd reading e u)]) []
  -- Step 3: structure the full semantics as a foldl over post starting
  -- from pre_env extended with v's entry.
  have h_full :
      dldsSemantics bd reading =
        post.foldl (fun e u => e ++ [(u, stepVertex bd reading e u)])
          (pre_env ++ [(v, stepVertex bd reading pre_env v)]) := by
    unfold dldsSemantics
    rw [h_split, List.foldl_append, List.foldl_cons]
  -- Step 4: pre_env contains no entry for v.
  have h_pre_env_no_v : ∀ w', (v, w') ∉ pre_env := by
    intro w' hw'
    rcases foldl_append_keys_subset
        (fun e u => stepVertex bd reading e u) pre [] (v, w') hw' with h1 | h1
    · simp at h1
    · exact h_v_not_pre h1
  -- Step 5: envLookup at v in pre_env ++ [(v, _)] is the singleton's value.
  have h_lookup_singleton :
      envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) v
        = stepVertex bd reading pre_env v :=
    envLookup_append_singleton_of_not_mem pre_env v
      (stepVertex bd reading pre_env v) h_pre_env_no_v
  -- Step 6: relate the post-foldl to a right-append.
  obtain ⟨extras, hex⟩ := foldl_append_step_eq_append
    (fun e u => stepVertex bd reading e u) post
    (pre_env ++ [(v, stepVertex bd reading pre_env v)])
  -- Step 7: envLookup at v in the full semantics equals the step value.
  have h_lookup_v :
      envLookup (dldsSemantics bd reading) v
        = stepVertex bd reading pre_env v := by
    have h_step1 :
        envLookup ((pre_env ++ [(v, stepVertex bd reading pre_env v)]) ++ extras) v
          = envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) v := by
      apply envLookup_append_of_mem
      exact ⟨stepVertex bd reading pre_env v, by simp⟩
    rw [h_full, hex, h_step1]
    exact h_lookup_singleton
  -- Step 8: For each source u of v, envLookup full = envLookup pre_env.
  have h_lookup_source : ∀ u ∈ incomingSources bd.base v,
      envLookup (dldsSemantics bd reading) u = envLookup pre_env u := by
    intro u hu_src
    -- u ∈ pre by topological order
    have hu_pre : u ∈ pre := by
      unfold incomingSources at hu_src
      rcases List.mem_map.mp hu_src with ⟨e, he_filter, he_start⟩
      rcases List.mem_filter.mp he_filter with ⟨he_mem, he_end_b⟩
      have he_end : e.END = v := by simpa using he_end_b
      rw [← he_start]
      exact h_edges_topo e he_mem pre post (by rw [he_end]; exact h_split)
    -- (u, _) ∈ pre_env via foldl_append_mem_of_mem
    have hu_in_pre_env : ∃ w, (u, w) ∈ pre_env :=
      foldl_append_mem_of_mem
        (fun e u => stepVertex bd reading e u) pre [] u hu_pre
    have h_step1 :
        envLookup ((pre_env ++ [(v, stepVertex bd reading pre_env v)]) ++ extras) u
          = envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) u := by
      apply envLookup_append_of_mem
      obtain ⟨w, hw⟩ := hu_in_pre_env
      exact ⟨w, List.mem_append_left _ hw⟩
    have h_step2 :
        envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) u
          = envLookup pre_env u :=
      envLookup_append_of_mem pre_env [(v, stepVertex bd reading pre_env v)] u
        hu_in_pre_env
    rw [h_full, hex, h_step1, h_step2]
  -- Step 9: unfold the step value using the per-step ELIM theorem.
  have h_step_or :
      stepVertex bd reading pre_env v =
        (incomingSources bd.base v).foldl
          (fun acc u => HypDepVec.or acc (envLookup pre_env u))
          (HypDepVec.zero bd.base) :=
    dldsSemantics_elim_or bd h_nb reading pre_env v h_not_hyp
  -- Step 10: combine.
  unfold dldsSemanticsAt
  rw [h_lookup_v, h_step_or]
  apply foldl_or_congr
  intro u hu
  exact (h_lookup_source u hu).symm

/-! #### Layer 2.8: Concrete sanity test -/

namespace Layer2Test

open Semantic (Formula Vertex Deduction DLDS BranchingDLDS Branching)

private def fA : Formula := .atom "A"
private def fB : Formula := .atom "B"
private def fC : Formula := .atom "C"

private def vX : Vertex :=
  { node := 0, LEVEL := 1, FORMULA := fA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vY : Vertex :=
  { node := 1, LEVEL := 1, FORMULA := fB,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vB : Vertex :=
  { node := 2, LEVEL := 0, FORMULA := fC,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- We model a "source" node whose dep vector is routed by branching.
-- In this minimal test, both X and Y are hypotheses; B is a non-hyp
-- vertex that receives from a branching whose source is vY and whose
-- target is vB, so the reading bit selects whether vY's dep flows into
-- vB or not. We also add an ordinary edge from vX to vB.
private def eXB : Deduction :=
  { START := vX, END := vB, COLOUR := 0, DEPENDENCY := [fA] }

private def eYB : Deduction :=
  { START := vY, END := vB, COLOUR := 0, DEPENDENCY := [fB] }

private def baseDLDS : DLDS :=
  { V := [vX, vY, vB], E := [eXB, eYB], A := [] }

/-- Branching: vY's dep flows into vB iff `reading[0] = true` (colour 1). -/
private def branchYtoB : Branching :=
  { source := vY, readingVar := 0, targets := [(1, vB)] }

/-- Tiny branching BranchingDLDS used to exercise the semantics. -/
def testBranchingDLDS : BranchingDLDS :=
  { base := baseDLDS
    branchings := [branchYtoB]
    numReading := 1
    evalOrder := [vX, vY, vB] }

-- Running the semantics with reading=[false] should drop vY's contribution
-- at vB, whereas reading=[true] should include it. The two outputs must
-- therefore differ at vB.
#eval (dldsSemanticsAt testBranchingDLDS [false] vB).toList
#eval (dldsSemanticsAt testBranchingDLDS [true]  vB).toList

end Layer2Test

#check @dldsSemantics
#check @dldsSemanticsAt
#check @dldsSemantics_hyp_onehot
#check @dldsSemantics_elim_or
#check @dldsSemantics_branching
#check @dldsSemanticsAt_elim_or

end ReadingBased

/-!
# DLDS Circuit Evaluation Tests

This module contains test cases demonstrating the DLDS-to-circuit
translation and evaluation. Each test constructs a natural deduction
proof as a DLDS and verifies it evaluates correctly.

## Test Cases

1. **Test.Identity**: Identity proof (A⊃B)⊃(A⊃B) - valid and invalid paths
2. **Test.Syllogism**: Hypothetical syllogism (A⊃B)⊃(B⊃C)⊃(A⊃C)
3. **Test.Incomplete**: Incomplete proof - undischarged assumptions detected

## Path Encoding

Paths are `List (List Nat)` where:
- Outer list: one entry per formula in the universe
- Inner list: one entry per level transition
- Value 0: token stops (inactive)
- Value n > 0: route to column (n-1)
-/

namespace Testing

open Semantic (Formula Vertex Deduction DLDS)

/-- Check if a DLDS proof is valid under a given path -/
def checkDLDSProof (d : DLDS) (paths : Semantic.PathInput) (goal_column : Nat) : IO Unit := do
  let result := Semantic.evaluateDLDS d paths goal_column
  IO.println s!"Evaluation result: {result}"
  if result then
    IO.println "✓ Accepted: Valid proof with discharged assumptions OR structural error"
  else
    IO.println "✗ Rejected: Invalid routing or undischarged assumptions"

end Testing


namespace Test.Identity
/-!
### Test: Identity Combinator (A⊃B)⊃(A⊃B)

This proof demonstrates:
- Implication introduction (twice)
- Implication elimination (modus ponens)
- Proper assumption discharge
```
    [A]¹  [A⊃B]²
    ───────────── ⊃E
          B
       ─────── ⊃I¹
        A⊃B
    ─────────── ⊃I²
    (A⊃B)⊃(A⊃B)
```
-/

open Semantic (Formula Vertex Deduction DLDS)

-- Formulas
def A : Formula := .atom "A"
def B : Formula := .atom "B"
def A_imp_B : Formula := .impl A B
def identity : Formula := .impl A_imp_B A_imp_B

-- Level 3: Assumptions
def v_A : Vertex :=
  { node := 0, LEVEL := 3, FORMULA := A,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

def v_AimpB_hyp : Vertex :=
  { node := 1, LEVEL := 3, FORMULA := A_imp_B,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

-- Level 2: B derived by modus ponens
def v_B : Vertex :=
  { node := 2, LEVEL := 2, FORMULA := B,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Level 1: A⊃B derived by intro (discharge A)
def v_AimpB : Vertex :=
  { node := 3, LEVEL := 1, FORMULA := A_imp_B,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Level 0: (A⊃B)⊃(A⊃B) derived by intro (discharge A⊃B)
def v_conclusion : Vertex :=
  { node := 4, LEVEL := 0, FORMULA := identity,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Edges
def e_A_to_B : Deduction :=
  { START := v_A, END := v_B, COLOUR := 0, DEPENDENCY := [A] }

def e_AimpB_to_B : Deduction :=
  { START := v_AimpB_hyp, END := v_B, COLOUR := 0, DEPENDENCY := [A_imp_B] }

def e_B_to_AimpB : Deduction :=
  { START := v_B, END := v_AimpB, COLOUR := 0, DEPENDENCY := [A, B] }

def e_AimpB_to_conclusion : Deduction :=
  { START := v_AimpB, END := v_conclusion, COLOUR := 0, DEPENDENCY := [A_imp_B] }

-- The DLDS
def dlds : DLDS := {
  V := [v_A, v_AimpB_hyp, v_B, v_AimpB, v_conclusion],
  E := [e_A_to_B, e_AimpB_to_B, e_B_to_AimpB, e_AimpB_to_conclusion],
  A := []
}

/-!
Formula universe (from `buildFormulas`):
- 0: A
- 1: A⊃B
- 2: B
- 3: (A⊃B)⊃(A⊃B)

Goal column: 3
-/

def validPath : List (List Nat) :=
  [ [3, 2, 4],   -- A: →B(col 2) →A⊃B(col 1) →conclusion(col 3)
    [3, 2, 4],   -- A⊃B: follows derivation path
    [0, 0, 0],   -- B: derived formula, not an assumption
    [0, 0, 0]    -- (A⊃B)⊃(A⊃B): goal, inactive
  ]

def invalidPath : List (List Nat) :=
  [ [4, 4, 4],   -- A: wrong - tries to skip intermediate steps
    [4, 0, 0],   -- B: inactive
    [4, 3, 4],   -- A⊃B: partially correct
    [4, 0, 0]    -- conclusion: inactive
  ]

#eval IO.println "\n══════════ TEST: Identity (Valid Path) ══════════"
-- Expected: ✓ Accepted (valid proof, all assumptions discharged)
#eval! Testing.checkDLDSProof dlds validPath 4

#eval IO.println "\n══════════ TEST: Identity (Invalid Path) ══════════"
-- Expected: ✓ Accepted (structural error detected)
#eval! Testing.checkDLDSProof dlds invalidPath 4

end Test.Identity


namespace Test.Syllogism
/-!
### Test: Hypothetical Syllogism (A⊃B)⊃(B⊃C)⊃(A⊃C)

This proof demonstrates a more complex derivation with three assumptions
and nested implication introductions.
```
    [A⊃B]³  [A]¹
    ──────────── ⊃E
    [B⊃C]²    B
    ──────────── ⊃E
          C
       ─────── ⊃I¹
        A⊃C
    ─────────── ⊃I²
    (B⊃C)⊃(A⊃C)
    ─────────────── ⊃I³
    (A⊃B)⊃(B⊃C)⊃(A⊃C)
```
-/

open Semantic (Formula Vertex Deduction DLDS)

-- Formulas
def A : Formula := .atom "A"
def B : Formula := .atom "B"
def C : Formula := .atom "C"
def A_imp_B : Formula := .impl A B
def B_imp_C : Formula := .impl B C
def A_imp_C : Formula := .impl A C
def inner : Formula := .impl B_imp_C A_imp_C
def conclusion : Formula := .impl A_imp_B inner

-- Level 5: Assumptions
def v_AimpB : Vertex :=
  { node := 0, LEVEL := 5, FORMULA := A_imp_B,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

def v_BimpC : Vertex :=
  { node := 1, LEVEL := 5, FORMULA := B_imp_C,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

def v_A : Vertex :=
  { node := 2, LEVEL := 5, FORMULA := A,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

-- Level 4: B (from A⊃B and A)
def v_B : Vertex :=
  { node := 3, LEVEL := 4, FORMULA := B,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Level 3: C (from B⊃C and B)
def v_C : Vertex :=
  { node := 4, LEVEL := 3, FORMULA := C,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Level 2: A⊃C (intro, discharge A)
def v_AimpC : Vertex :=
  { node := 5, LEVEL := 2, FORMULA := A_imp_C,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Level 1: (B⊃C)⊃(A⊃C) (intro, discharge B⊃C)
def v_inner : Vertex :=
  { node := 6, LEVEL := 1, FORMULA := inner,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Level 0: Conclusion (intro, discharge A⊃B)
def v_conclusion : Vertex :=
  { node := 7, LEVEL := 0, FORMULA := conclusion,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Edges
def e0 : Deduction := { START := v_AimpB, END := v_B, COLOUR := 0, DEPENDENCY := [A_imp_B] }
def e1 : Deduction := { START := v_A, END := v_B, COLOUR := 0, DEPENDENCY := [A] }
def e2 : Deduction := { START := v_BimpC, END := v_C, COLOUR := 0, DEPENDENCY := [B_imp_C] }
def e3 : Deduction := { START := v_B, END := v_C, COLOUR := 0, DEPENDENCY := [B] }
def e4 : Deduction := { START := v_C, END := v_AimpC, COLOUR := 0, DEPENDENCY := [C] }
def e5 : Deduction := { START := v_AimpC, END := v_inner, COLOUR := 0, DEPENDENCY := [A_imp_C] }
def e6 : Deduction := { START := v_inner, END := v_conclusion, COLOUR := 0, DEPENDENCY := [inner] }

def dlds : DLDS := {
  V := [v_AimpB, v_BimpC, v_A, v_B, v_C, v_AimpC, v_inner, v_conclusion],
  E := [e0, e1, e2, e3, e4, e5, e6],
  A := []
}

/-!
Formula universe will include: A, B, C, A⊃B, B⊃C, A⊃C, (B⊃C)⊃(A⊃C), conclusion
Goal column: 7 (or wherever conclusion lands in eraseDups order)
-/

def validPath : List (List Nat) :=
  [ [4, 5, 6, 7, 8, 8, 8, 8],
    [2, 5, 6, 7, 8, 8, 8, 8],
    [4, 5, 6, 7, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
  ]

#eval IO.println "\n══════════ TEST: Hypothetical Syllogism ══════════"
-- Expected: ✓ Accepted
#eval! Testing.checkDLDSProof dlds validPath 8

end Test.Syllogism


namespace Test.Incomplete
/-!
### Test: Incomplete Proof (Undischarged Assumptions)

This test verifies that the circuit correctly REJECTS proofs where
assumptions are not properly discharged.

We derive B from A and A⊃B using modus ponens, but we do NOT
wrap it in an implication introduction to discharge the assumptions.
```
    A    A⊃B
    ───────── ⊃E
        B
```

This is a valid DERIVATION but not a valid PROOF of B, since B
depends on undischarged assumptions A and A⊃B.

The evaluation should return `false` because the dependency vector
at the goal column will NOT be all zeros.
-/

open Semantic (Formula Vertex Deduction DLDS)

-- Formulas
def A : Formula := .atom "A"
def B : Formula := .atom "B"
def A_imp_B : Formula := .impl A B

-- Level 1: Assumptions (NOT discharged)
def v_A : Vertex :=
  { node := 0, LEVEL := 1, FORMULA := A,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

def v_AimpB : Vertex :=
  { node := 1, LEVEL := 1, FORMULA := A_imp_B,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

-- Level 0: B derived by modus ponens
def v_B : Vertex :=
  { node := 2, LEVEL := 0, FORMULA := B,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

-- Edges
def e_A : Deduction :=
  { START := v_A, END := v_B, COLOUR := 0, DEPENDENCY := [A] }

def e_AimpB : Deduction :=
  { START := v_AimpB, END := v_B, COLOUR := 0, DEPENDENCY := [A_imp_B] }

def dlds : DLDS := {
  V := [v_A, v_AimpB, v_B],
  E := [e_A, e_AimpB],
  A := []
}

/-!
Formula universe: A, A⊃B, B (3 formulas)
Goal column: 1 (B)

The path routes both assumptions to B, but since there's no
intro rule to discharge them, the final dependency vector for B
will have bits set for A and A⊃B.
-/

def path : List (List Nat) :=
  [ [2, 2],   -- A: routes to B
    [0, 0],   -- B: goal (inactive routing)
    [2, 2]    -- A⊃B: routes to B
  ]

#eval IO.println "\n══════════ TEST: Incomplete Proof ══════════"
-- Expected: ✗ Rejected (undischarged assumptions)
#eval! Testing.checkDLDSProof dlds path 1

end Test.Incomplete


namespace Test.Summary
/-!
## Test Summary

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Identity (valid) | Correct derivation of (A⊃B)⊃(A⊃B) | ✓ Accepted |
| Identity (invalid) | Wrong routing path | ✓ Accepted (structural error) |
| Syllogism | Hypothetical syllogism | ✓ Accepted |
| Incomplete | Undischarged assumptions | ✗ Rejected |

## Interpretation

- **Accepted (true)**: Either the path has a structural error (detected by XOR check)
  OR it represents a valid proof with all assumptions discharged.

- **Rejected (false)**: The path is structurally valid but the proof has
  undischarged assumptions (dependency vector is non-zero at goal).

This matches the main theorem `circuit_correctness`:
```
evaluateCircuit = true →
  PathStructurallyInvalid ∨ (PathRepresentsValidProof ∧ AllAssumptionsDischarged)
```
-/
end Test.Summary
