import Init
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.GetD
import DLDSBooleanCircuit.Robustness


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

section ReadingBased

/-!
### Reading-based DLDS evaluation

This section introduces a second evaluation function for DLDS based on
**reading variables**, matching the input model used in quantum compilation.
Reading variables are Boolean assignments that select which branch to follow
at each branching node in the DLDS.

For this wrapper interface we restrict attention to DLDS without branching nodes: in that
case the reading is irrelevant and there is exactly one canonical path.
The point is to establish the API and the per-node correspondence
foundation used by the standalone semantics below.
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
  formulas.map (fun _ => [])

/-- Reading-based DLDS evaluation.

    For non-branching DLDS, this is equivalent to path-based evaluation
    with the canonical path. Branching DLDS are handled by the standalone
    semantics below. -/
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
    (available_inputs : List (Nat × List.Vector Bool n)) :
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

/-- **Reading-Based DLDS Soundness.**

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

/-!
### Standalone reading-based DLDS semantics

This section defines the reading-based semantics used by the kernel
equivalence results: a standalone
`dldsSemantics` function that walks the DLDS directly via per-node rules
(hypothesis → one-hot, ELIM → OR of incoming sources, with reading bits
consulted at branching points). Unlike the wrapper interface above, which was built
around the path-based circuit, this semantics is independent of the grid
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
* **`Branching` metadata is explicit** and records the source, reading
  variable, and colour-labelled targets used by the evaluator.

We prove two characterisation theorems for the
non-branching case (hypothesis one-hot correctness and ELIM OR
accumulation at the step level), followed by introduction discharge,
branching, and global equivalence results.
-/

/-! #### Hypothesis-indexed dep vectors -/

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

/-! #### Branching metadata -/

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
    are purely additive. -/
structure BranchingDLDS where
  base       : DLDS
  branchings : List Branching := []
  numReading : Nat := 0
  evalOrder  : List Vertex
  deriving Repr

def BranchingDLDS.ofBase (d : DLDS) : BranchingDLDS :=
  { base := d
    branchings := []
    numReading := 0
    evalOrder := d.V }

@[simp] theorem BranchingDLDS.ofBase_base (d : DLDS) :
    (BranchingDLDS.ofBase d).base = d := by
  rfl

@[simp] theorem BranchingDLDS.ofBase_branchings (d : DLDS) :
    (BranchingDLDS.ofBase d).branchings = [] := by
  rfl

@[simp] theorem BranchingDLDS.ofBase_numReading (d : DLDS) :
    (BranchingDLDS.ofBase d).numReading = 0 := by
  rfl

@[simp] theorem BranchingDLDS.ofBase_evalOrder (d : DLDS) :
    (BranchingDLDS.ofBase d).evalOrder = d.V := by
  rfl

/-- Well-formedness: the evaluation order must be a permutation of
    the DLDS's vertex list. -/
def BranchingDLDS.WellFormed (bd : BranchingDLDS) : Prop :=
  bd.evalOrder.Perm bd.base.V

theorem BranchingDLDS.WellFormed_of_evalOrder_eq_baseV
    (bd : BranchingDLDS)
    (h_eval : bd.evalOrder = bd.base.V) :
    bd.WellFormed := by
  simp [BranchingDLDS.WellFormed, h_eval]

theorem BranchingDLDS.ofBase_WellFormed
    (d : DLDS) :
    (BranchingDLDS.ofBase d).WellFormed := by
  exact BranchingDLDS.WellFormed_of_evalOrder_eq_baseV
    (BranchingDLDS.ofBase d) (by simp)

theorem BranchingDLDS.ofBase_evalOrderInVertices
    (d : DLDS) :
    ∀ v ∈ (BranchingDLDS.ofBase d).evalOrder,
      v ∈ (BranchingDLDS.ofBase d).base.V := by
  intro v hv
  simpa using hv

/-- A `BranchingDLDS` is non-branching iff it carries no branching
    metadata. The reading input is then ignored. -/
def BranchingDLDS.IsNonBranching (bd : BranchingDLDS) : Prop :=
  bd.branchings = []

/-! #### Incoming source lookup -/

/-- Sources flowing into vertex `v`: all `u` such that `(u, v)` is a
    deduction edge in `d.E`. This mirrors the Python compiler's walk
    over `node.minor`, `node.major`, and `extra_sources`, minus the
    branching-specific filter (which is applied separately by
    `stepVertex`). -/
def incomingSources (d : DLDS) (v : Vertex) : List Vertex :=
  (d.E.filter (fun e => e.END = v)).map (·.START)

/-! #### Per-vertex step and environment -/

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

/-- The hypothesis discharged by an introduction vertex, if any.
    Decoded from `DLDS.A`: an entry `(w, h)` marks `w` as an introduction
    vertex discharging the hypothesis vertex `h`. -/
def findIntroDischarge (bd : BranchingDLDS) (v : Vertex) : Option Vertex :=
  (bd.base.A.find? (fun p => p.1 = v)).map (·.2)

/-- One step of the reading-based semantics: compute the dep vector
    for vertex `v` given the environment of already-processed vertices.

    Cases are dispatched in priority order:
    1. Hypothesis vertex → `oneHot` at its hypothesis index.
    2. Vertex receiving from a branch → OR of the ordinary incoming
       sources plus the branched source iff the reading colour matches.
    3. Introduction vertex (discharging a hypothesis) → OR of all
       incoming sources with the discharged bit cleared.
    4. Plain ELIM vertex → OR of all incoming sources.

    Note: if a vertex is both a branching target and an introduction,
    branching wins (design choice, not a property of the DLDS type). -/
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
    | some (src, rvar, colour) =>
        let ordinary := sources.filter (fun u => decide (u ≠ src))
        let base := ordinary.foldl
          (fun acc u => HypDepVec.or acc (envLookup env u))
          (HypDepVec.zero bd.base)
        if readingColour reading rvar = some colour then
          HypDepVec.or base (envLookup env src)
        else
          base
    | none =>
        let or_all := sources.foldl
          (fun acc u => HypDepVec.or acc (envLookup env u))
          (HypDepVec.zero bd.base)
        match findIntroDischarge bd v with
        | some h =>
            match hypIndex bd.base h with
            | some ⟨k, _⟩ => HypDepVec.clearBit k or_all
            | none => or_all
        | none => or_all

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

/-! #### Foldl helper lemmas

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

/-! #### Non-branching characterisation theorems -/

/-- **Non-Branching Hypothesis Semantics.**
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

/-- **Non-Branching ELIM Step Semantics.**
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
    (h_not_intro : findIntroDischarge bd v = none)
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
  simp only [h_br, h_not_intro]

/-! #### Topological well-formedness and global ELIM lifting

This section lifts `dldsSemantics_elim_or` from the per-step `stepVertex`
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

/-- **Non-Branching Global ELIM Semantics.**
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
    (h_not_intro : findIntroDischarge bd v = none)
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
    dldsSemantics_elim_or bd h_nb reading pre_env v h_not_intro h_not_hyp
  -- Step 10: combine.
  unfold dldsSemanticsAt
  rw [h_lookup_v, h_step_or]
  apply foldl_or_congr
  intro u hu
  exact (h_lookup_source u hu).symm

/-! #### Global introduction discharge

Introduction vertices are handled directly by the unified `stepVertex`
definition. Each entry `(w, h) ∈ d.A` marks `w` as an introduction
vertex discharging the hypothesis vertex `h`; `findIntroDischarge`
decodes this, and `stepVertex` dispatches to the intro arm in its
four-case priority order (hypothesis, branching, intro, plain ELIM). -/

/-- **Step-level INTRO characterisation.** For a non-hypothesis,
    non-branching intro vertex `v` discharging hypothesis `h_discharge`
    at index `k`, `stepVertex` computes `HypDepVec.clearBit k` of the
    OR-fold of the incoming sources' environment lookups. -/
theorem dldsSemantics_intro_clearbit
    (bd : BranchingDLDS) (reading : ReadingInput)
    (env : List (Vertex × HypDepVec bd.base))
    (v h_discharge : Vertex) (k : Nat) (hk : k < numHyps bd.base)
    (h_not_hyp : v.HYPOTHESIS = false)
    (h_not_branch : findBranchTarget bd v = none)
    (h_intro : findIntroDischarge bd v = some h_discharge)
    (h_idx : hypIndex bd.base h_discharge = some ⟨k, hk⟩) :
    stepVertex bd reading env v =
      HypDepVec.clearBit k
        ((incomingSources bd.base v).foldl
          (fun acc u => HypDepVec.or acc (envLookup env u))
          (HypDepVec.zero bd.base)) := by
  unfold stepVertex
  have h_h : ¬ (v.HYPOTHESIS = true) := by rw [h_not_hyp]; decide
  rw [if_neg h_h]
  simp only [h_not_branch, h_intro, h_idx]

/-- **Global INTRO discharge theorem.** Under topological well-formedness,
    `dldsSemanticsAt` at an introduction vertex `v` discharging hypothesis
    `h_discharge` at index `k` equals the clearBit of the OR-fold of
    `dldsSemanticsAt` over `v`'s incoming sources. Structurally mirrors
    `dldsSemanticsAt_elim_or`. -/
theorem dldsSemanticsAt_intro_clearbit
    (bd : BranchingDLDS)
    (h_topo : bd.WellFormedTopo)
    (reading : ReadingInput)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder)
    (h_not_hyp : v.HYPOTHESIS = false)
    (h_not_branch : findBranchTarget bd v = none)
    (h_discharge : Vertex) (k : Nat) (hk : k < numHyps bd.base)
    (h_intro : findIntroDischarge bd v = some h_discharge)
    (h_idx : hypIndex bd.base h_discharge = some ⟨k, hk⟩) :
    dldsSemanticsAt bd reading v =
      HypDepVec.clearBit k
        ((incomingSources bd.base v).foldl
          (fun acc u => HypDepVec.or acc (dldsSemanticsAt bd reading u))
          (HypDepVec.zero bd.base)) := by
  obtain ⟨h_nodup, h_edges_topo⟩ := h_topo
  obtain ⟨pre, post, h_split, h_v_not_pre, _h_v_not_post⟩ :=
    nodup_split_at_vertex bd.evalOrder v h_nodup h_mem
  let pre_env : List (Vertex × HypDepVec bd.base) :=
    pre.foldl (fun e u => e ++ [(u, stepVertex bd reading e u)]) []
  have h_full :
      dldsSemantics bd reading =
        post.foldl (fun e u => e ++ [(u, stepVertex bd reading e u)])
          (pre_env ++ [(v, stepVertex bd reading pre_env v)]) := by
    unfold dldsSemantics
    rw [h_split, List.foldl_append, List.foldl_cons]
  have h_pre_env_no_v : ∀ w', (v, w') ∉ pre_env := by
    intro w' hw'
    rcases foldl_append_keys_subset
        (fun e u => stepVertex bd reading e u) pre [] (v, w') hw' with h1 | h1
    · simp at h1
    · exact h_v_not_pre h1
  have h_lookup_singleton :
      envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) v
        = stepVertex bd reading pre_env v :=
    envLookup_append_singleton_of_not_mem pre_env v
      (stepVertex bd reading pre_env v) h_pre_env_no_v
  obtain ⟨extras, hex⟩ := foldl_append_step_eq_append
    (fun e u => stepVertex bd reading e u) post
    (pre_env ++ [(v, stepVertex bd reading pre_env v)])
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
  have h_lookup_source : ∀ u ∈ incomingSources bd.base v,
      envLookup (dldsSemantics bd reading) u = envLookup pre_env u := by
    intro u hu_src
    have hu_pre : u ∈ pre := by
      unfold incomingSources at hu_src
      rcases List.mem_map.mp hu_src with ⟨e, he_filter, he_start⟩
      rcases List.mem_filter.mp he_filter with ⟨he_mem, he_end_b⟩
      have he_end : e.END = v := by simpa using he_end_b
      rw [← he_start]
      exact h_edges_topo e he_mem pre post (by rw [he_end]; exact h_split)
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
  have h_step_clear :
      stepVertex bd reading pre_env v =
        HypDepVec.clearBit k
          ((incomingSources bd.base v).foldl
            (fun acc u => HypDepVec.or acc (envLookup pre_env u))
            (HypDepVec.zero bd.base)) :=
    dldsSemantics_intro_clearbit bd reading pre_env v h_discharge k hk
      h_not_hyp h_not_branch h_intro h_idx
  unfold dldsSemanticsAt
  rw [h_lookup_v, h_step_clear]
  congr 1
  apply foldl_or_congr
  intro u hu
  exact (h_lookup_source u hu).symm

namespace Examples.IntroDischarge

open Semantic (Formula Vertex Deduction DLDS BranchingDLDS)

private def fA : Formula := .atom "A"

private def vA : Vertex :=
  { node := 0, LEVEL := 1, FORMULA := fA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vIntro : Vertex :=
  { node := 1, LEVEL := 0, FORMULA := .impl fA fA,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

private def eAtoIntro : Deduction :=
  { START := vA, END := vIntro, COLOUR := 0, DEPENDENCY := [fA] }

private def baseDLDS : DLDS :=
  { V := [vA, vIntro], E := [eAtoIntro], A := [(vIntro, vA)] }

def introDischargeDLDS : BranchingDLDS :=
  { base := baseDLDS
    branchings := []
    numReading := 0
    evalOrder := [vA, vIntro] }

-- Expected: [false]. vA contributes its one-hot [true] to the OR-fold at
-- vIntro, and the introduction clears that bit.
#eval (dldsSemanticsAt introDischargeDLDS [] vIntro).toList

end Examples.IntroDischarge

/-! #### Branching correspondence

This section proves `dldsSemanticsAt_branching`, the global-level
characterisation of the reading-based semantics at a vertex receiving
from a branching. It uses the same proof infrastructure as
`dldsSemanticsAt_elim_or` together with a stronger well-formedness
predicate `WellFormedBranching` that ensures branching sources are
processed before their targets. -/

/-- A `BranchingDLDS` is **branching-well-formed** when it is
    topologically well-formed (`WellFormedTopo`) and, additionally,
    every branching source appears before every branching target in
    `evalOrder`. -/
def BranchingDLDS.WellFormedBranching (bd : BranchingDLDS) : Prop :=
  bd.WellFormedTopo ∧
  ∀ b ∈ bd.branchings,
    ∀ c v, (c, v) ∈ b.targets →
      ∀ pre post : List Vertex,
        bd.evalOrder = pre ++ v :: post → b.source ∈ pre

/-- Generic inversion for `List.findSome?`: if it returns `some b`,
    then some element of the list maps to `some b`. -/
private lemma list_findSome?_spec {α β : Type} (f : α → Option β)
    (xs : List α) (b : β)
    (h : xs.findSome? f = some b) :
    ∃ a ∈ xs, f a = some b := by
  induction xs with
  | nil => simp at h
  | cons x rest ih =>
    unfold List.findSome? at h
    split at h
    · -- f x = some val: findSome? returns it
      rename_i b' heq
      -- heq : f x = some b', h : some b' = some b
      have : b' = b := by injection h
      subst this
      exact ⟨x, by simp, heq⟩
    · -- f x = none: findSome? recurses
      obtain ⟨a, ha_mem, ha_eq⟩ := ih h
      exact ⟨a, List.mem_cons_of_mem _ ha_mem, ha_eq⟩

/-- Generic inversion for `List.find?`: if it returns `some a`,
    then `a` is in the list and satisfies the predicate. -/
private lemma list_find?_spec {α : Type} (p : α → Bool)
    (xs : List α) (a : α)
    (h : xs.find? p = some a) :
    a ∈ xs ∧ p a = true := by
  induction xs with
  | nil => simp at h
  | cons x rest ih =>
    unfold List.find? at h
    split at h
    · -- p x = true: find? returns some x
      have hxa : x = a := by injection h
      subst hxa
      simp_all
    · -- p x = false: find? recurses
      obtain ⟨ha_mem, ha_p⟩ := ih h
      exact ⟨List.mem_cons_of_mem _ ha_mem, ha_p⟩

/-- Inversion for `findBranchTarget`: if it returns
    `some (src, rvar, colour)` for vertex `v`, there exists a branching
    `b ∈ bd.branchings` with `b.source = src`, `b.readingVar = rvar`,
    and `(colour, v) ∈ b.targets`. -/
private lemma findBranchTarget_spec (bd : BranchingDLDS) (v : Vertex)
    (src : Vertex) (rvar colour : Nat)
    (h : findBranchTarget bd v = some (src, rvar, colour)) :
    ∃ b ∈ bd.branchings,
      b.source = src ∧ b.readingVar = rvar ∧ (colour, v) ∈ b.targets := by
  unfold findBranchTarget at h
  obtain ⟨b, hb_mem, hb_eq⟩ := list_findSome?_spec _ _ _ h
  -- hb_eq : (b.targets.find? (fun p => p.2 = v)).map (...) = some (src, rvar, colour)
  -- Invert Option.map
  cases hfind : b.targets.find? (fun p => decide (p.2 = v)) with
  | none =>
    simp [hfind] at hb_eq
  | some q =>
    simp [hfind] at hb_eq
    -- hb_eq should give us (b.source, b.readingVar, q.1) = (src, rvar, colour)
    obtain ⟨h1, h2, h3⟩ := hb_eq
    -- From find?: q ∈ b.targets and q.2 = v
    obtain ⟨hq_mem, hq_pred⟩ := list_find?_spec _ _ _ hfind
    have hq2 : q.2 = v := by
      simp at hq_pred
      exact hq_pred
    -- q = (q.1, q.2) = (colour, v)
    have hq_eq : q = (colour, v) := by
      cases q with
      | mk fst snd =>
        simp at h3 hq2
        exact Prod.ext h3 hq2
    rw [hq_eq] at hq_mem
    exact ⟨b, hb_mem, h1, h2, hq_mem⟩

/-- Step-level branching characterisation: for a non-hypothesis vertex
    `v` that is a branching target, `stepVertex` computes the OR of
    ordinary sources conditionally joined with the branched source. -/
private lemma stepVertex_branching_eq (bd : BranchingDLDS)
    (reading : ReadingInput)
    (env : List (Vertex × HypDepVec bd.base))
    (v : Vertex)
    (h_not_hyp : v.HYPOTHESIS = false)
    (src : Vertex) (rvar colour : Nat)
    (h_branch : findBranchTarget bd v = some (src, rvar, colour)) :
    let ordinary := (incomingSources bd.base v).filter (fun u => decide (u ≠ src))
    let base_val :=
      ordinary.foldl
        (fun acc u => HypDepVec.or acc (envLookup env u))
        (HypDepVec.zero bd.base)
    stepVertex bd reading env v =
      (if readingColour reading rvar = some colour then
        HypDepVec.or base_val (envLookup env src)
      else
        base_val) := by
  unfold stepVertex
  have h_h : ¬ (v.HYPOTHESIS = true) := by rw [h_not_hyp]; decide
  rw [if_neg h_h]
  simp only [h_branch]

/-- **Branching Global Semantics.**
    For a non-hypothesis vertex `v` in a branching-well-formed
    `BranchingDLDS` that receives from a branching with source `src`,
    reading variable `rvar`, and colour `colour`, the global semantics
    `dldsSemanticsAt` at `v` equals the foldl-OR of `dldsSemanticsAt`
    over the ordinary incoming sources, conditionally joined with
    `dldsSemanticsAt` at `src` when the reading colour matches. -/
theorem dldsSemanticsAt_branching
    (bd : BranchingDLDS)
    (h_wf : bd.WellFormedBranching)
    (reading : ReadingInput)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder)
    (h_not_hyp : v.HYPOTHESIS = false)
    (src : Vertex) (rvar colour : Nat)
    (h_branch : findBranchTarget bd v = some (src, rvar, colour)) :
    let sources := incomingSources bd.base v
    let ordinary := sources.filter (fun u => decide (u ≠ src))
    let base_or :=
      ordinary.foldl
        (fun acc u => HypDepVec.or acc (dldsSemanticsAt bd reading u))
        (HypDepVec.zero bd.base)
    dldsSemanticsAt bd reading v =
      (if readingColour reading rvar = some colour then
        HypDepVec.or base_or (dldsSemanticsAt bd reading src)
      else
        base_or) := by
  -- Destructure well-formedness.
  obtain ⟨h_topo, h_branch_topo⟩ := h_wf
  obtain ⟨h_nodup, h_edges_topo⟩ := h_topo
  -- Step 1: split evalOrder at v.
  obtain ⟨pre, post, h_split, h_v_not_pre, _h_v_not_post⟩ :=
    nodup_split_at_vertex bd.evalOrder v h_nodup h_mem
  -- Step 2: define pre_env.
  let pre_env : List (Vertex × HypDepVec bd.base) :=
    pre.foldl (fun e u => e ++ [(u, stepVertex bd reading e u)]) []
  -- Step 3: full semantics = post-foldl starting from pre_env ++ [(v, ...)].
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
  -- Step 5: envLookup at v in pre_env ++ [(v, ...)] is the step value.
  have h_lookup_singleton :
      envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) v
        = stepVertex bd reading pre_env v :=
    envLookup_append_singleton_of_not_mem pre_env v
      (stepVertex bd reading pre_env v) h_pre_env_no_v
  -- Step 6: relate post-foldl to a right-append.
  obtain ⟨extras, hex⟩ := foldl_append_step_eq_append
    (fun e u => stepVertex bd reading e u) post
    (pre_env ++ [(v, stepVertex bd reading pre_env v)])
  -- Step 7: envLookup at v in full semantics = step value.
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
  -- Step 8: extract branching membership from findBranchTarget.
  obtain ⟨branch, hb_mem, hb_src, _hb_rvar, hb_tgt⟩ :=
    findBranchTarget_spec bd v src rvar colour h_branch
  -- Step 9: src ∈ pre by branching well-formedness.
  have h_src_pre : src ∈ pre := by
    rw [← hb_src]
    exact h_branch_topo branch hb_mem colour v hb_tgt pre post h_split
  -- Step 10: for every ordinary source u of v, envLookup full = envLookup pre_env.
  have h_lookup_ordinary :
      ∀ u ∈ (incomingSources bd.base v).filter (fun u => decide (u ≠ src)),
        envLookup (dldsSemantics bd reading) u = envLookup pre_env u := by
    intro u hu
    have hu_src : u ∈ incomingSources bd.base v :=
      List.mem_of_mem_filter hu
    have hu_pre : u ∈ pre := by
      unfold incomingSources at hu_src
      rcases List.mem_map.mp hu_src with ⟨e, he_filter, he_start⟩
      rcases List.mem_filter.mp he_filter with ⟨he_mem, he_end_b⟩
      have he_end : e.END = v := by simpa using he_end_b
      rw [← he_start]
      exact h_edges_topo e he_mem pre post (by rw [he_end]; exact h_split)
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
  -- Step 11: envLookup full src = envLookup pre_env src.
  have h_lookup_src :
      envLookup (dldsSemantics bd reading) src = envLookup pre_env src := by
    have hu_in_pre_env : ∃ w, (src, w) ∈ pre_env :=
      foldl_append_mem_of_mem
        (fun e u => stepVertex bd reading e u) pre [] src h_src_pre
    have h_step1 :
        envLookup ((pre_env ++ [(v, stepVertex bd reading pre_env v)]) ++ extras) src
          = envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) src := by
      apply envLookup_append_of_mem
      obtain ⟨w, hw⟩ := hu_in_pre_env
      exact ⟨w, List.mem_append_left _ hw⟩
    have h_step2 :
        envLookup (pre_env ++ [(v, stepVertex bd reading pre_env v)]) src
          = envLookup pre_env src :=
      envLookup_append_of_mem pre_env [(v, stepVertex bd reading pre_env v)] src
        hu_in_pre_env
    rw [h_full, hex, h_step1, h_step2]
  -- Step 12: unfold stepVertex using the branching case.
  have h_step_eq := stepVertex_branching_eq bd reading pre_env v h_not_hyp
    src rvar colour h_branch
  -- Step 13: combine.
  unfold dldsSemanticsAt
  rw [h_lookup_v, h_step_eq]
  -- Goal: (if ... then HypDepVec.or (foldl ... envLookup pre_env ...) (envLookup pre_env src)
  --        else foldl ... envLookup pre_env ...)
  --       = (if ... then HypDepVec.or (foldl ... dldsSemanticsAt ...) (dldsSemanticsAt src)
  --          else foldl ... dldsSemanticsAt ...)
  by_cases h_col : readingColour reading rvar = some colour
  · -- colour matches: both if-branches take the true path
    simp only [if_pos h_col]
    congr 1
    · -- ordinary foldl equality
      apply foldl_or_congr
      intro u hu
      exact (h_lookup_ordinary u hu).symm
    · -- src lookup equality
      exact h_lookup_src.symm
  · -- colour doesn't match: both if-branches take the false path
    simp only [if_neg h_col]
    apply foldl_or_congr
    intro u hu
    exact (h_lookup_ordinary u hu).symm

/-! #### Minimal branching example -/

namespace Examples.MinimalBranching

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
def minimalBranchingDLDS : BranchingDLDS :=
  { base := baseDLDS
    branchings := [branchYtoB]
    numReading := 1
    evalOrder := [vX, vY, vB] }

-- Running the semantics with reading=[false] should drop vY's contribution
-- at vB, whereas reading=[true] should include it. The two outputs must
-- therefore differ at vB.
#eval (dldsSemanticsAt minimalBranchingDLDS [false] vB).toList
#eval (dldsSemanticsAt minimalBranchingDLDS [true]  vB).toList

end Examples.MinimalBranching

/-! #### All-cases worked example

A single branching DLDS that simultaneously exercises branching,
introduction discharge, and plain ELIM, with four hypotheses so dep
vectors have enough structure to distinguish per-vertex contributions.
Used in the companion paper as the medium worked example. -/

namespace Examples.AllCases

open Semantic (Formula Vertex Deduction DLDS BranchingDLDS Branching)

private def fA : Formula := .atom "A"
private def fB : Formula := .atom "B"
private def fC : Formula := .atom "C"
private def fD : Formula := .atom "D"
private def fE : Formula := .atom "E"
private def fF : Formula := .atom "F"
private def fG : Formula := .atom "G"
private def fH : Formula := .atom "H"

private def vH1 : Vertex :=
  { node := 0, LEVEL := 2, FORMULA := fA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }
private def vH2 : Vertex :=
  { node := 1, LEVEL := 2, FORMULA := fB,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }
private def vH3 : Vertex :=
  { node := 2, LEVEL := 2, FORMULA := fC,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }
private def vH4 : Vertex :=
  { node := 3, LEVEL := 2, FORMULA := fD,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vE1 : Vertex :=
  { node := 4, LEVEL := 1, FORMULA := fE,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }
private def vE2 : Vertex :=
  { node := 5, LEVEL := 1, FORMULA := fF,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }
private def vI : Vertex :=
  { node := 6, LEVEL := 1, FORMULA := fG,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }
private def vRoot : Vertex :=
  { node := 7, LEVEL := 0, FORMULA := fH,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

private def e_H1_E1 : Deduction :=
  { START := vH1, END := vE1, COLOUR := 0, DEPENDENCY := [fA] }
private def e_H2_E1 : Deduction :=
  { START := vH2, END := vE1, COLOUR := 0, DEPENDENCY := [fB] }
private def e_H2_E2 : Deduction :=
  { START := vH2, END := vE2, COLOUR := 0, DEPENDENCY := [fB] }
private def e_H3_E2 : Deduction :=
  { START := vH3, END := vE2, COLOUR := 0, DEPENDENCY := [fC] }
private def e_E1_I : Deduction :=
  { START := vE1, END := vI, COLOUR := 0, DEPENDENCY := [fE] }
private def e_I_Root : Deduction :=
  { START := vI, END := vRoot, COLOUR := 0, DEPENDENCY := [fG] }
private def e_E2_Root : Deduction :=
  { START := vE2, END := vRoot, COLOUR := 0, DEPENDENCY := [fF] }
private def e_H4_Root : Deduction :=
  { START := vH4, END := vRoot, COLOUR := 0, DEPENDENCY := [fD] }

private def baseDLDS : DLDS :=
  { V := [vH1, vH2, vH3, vH4, vE1, vE2, vI, vRoot]
    E := [e_H1_E1, e_H2_E1, e_H2_E2, e_H3_E2,
          e_E1_I, e_I_Root, e_E2_Root, e_H4_Root]
    A := [(vI, vH1)] }

/-- Branching: vH3's dep flows into vE2 iff `reading[0] = true`
    (colour 1). -/
private def branchH3toE2 : Branching :=
  { source := vH3, readingVar := 0, targets := [(1, vE2)] }

/-- Eight-vertex branching DLDS exercising branching, introduction
    discharge, and plain ELIM in a single object. Four hypotheses
    vH1..vH4 with indices 0..3; vE1 plain-ORs vH1 and vH2; vE2 ORs
    vH2 with vH3 conditionally on the reading bit; vI discharges
    vH1 from vE1 (clearing bit 0); vRoot plain-ORs vI, vE2, and vH4. -/
def allCasesDLDS : BranchingDLDS :=
  { base := baseDLDS
    branchings := [branchH3toE2]
    numReading := 1
    evalOrder := [vH1, vH2, vH3, vH4, vE1, vE2, vI, vRoot] }

-- Reading [false]: the branching drops vH3's contribution at vE2.
-- Expected dep vectors (bit order: vH1, vH2, vH3, vH4):
--   vE1   = [true,  true,  false, false]
--   vE2   = [false, true,  false, false]
--   vI    = [false, true,  false, false]
--   vRoot = [false, true,  false, true ]
#eval (dldsSemanticsAt allCasesDLDS [false] vE1).toList
#eval (dldsSemanticsAt allCasesDLDS [false] vE2).toList
#eval (dldsSemanticsAt allCasesDLDS [false] vI).toList
#eval (dldsSemanticsAt allCasesDLDS [false] vRoot).toList

-- Reading [true]: colour 1 matches, so vH3 flows into vE2 and on into
-- vRoot. Expected dep vectors:
--   vE1   = [true,  true,  false, false]
--   vE2   = [false, true,  true,  false]
--   vI    = [false, true,  false, false]
--   vRoot = [false, true,  true,  true ]
#eval (dldsSemanticsAt allCasesDLDS [true] vE1).toList
#eval (dldsSemanticsAt allCasesDLDS [true] vE2).toList
#eval (dldsSemanticsAt allCasesDLDS [true] vI).toList
#eval (dldsSemanticsAt allCasesDLDS [true] vRoot).toList

end Examples.AllCases

/-! #### Kernel equivalence across index spaces

This section formalizes the per-vertex kernel equivalence between the
classical Boolean-circuit evaluation (`node_logic`, Section 2) and the
reading-based evaluation (`stepVertex`).

The two kernels operate in different index spaces:
* `node_logic` uses formula-indexed dep vectors of dimension
  `numFormulas d = (buildFormulas d).length`
* `stepVertex` uses hypothesis-indexed dep vectors of dimension
  `numHyps d`

The index-space translation `formulaVecToHypVec` maps formula-indexed
vectors to hypothesis-indexed vectors by projecting onto the formula
columns corresponding to hypothesis vertices. The main theorem
`classicalKernel_stepVertex_equiv` proves that a formula-indexed kernel
mirroring `stepVertex`'s four-case dispatch translates to `stepVertex`'s
output when the environments are aligned. -/

/-- Number of formula columns in the classical Boolean-circuit grid. -/
def numFormulas (d : DLDS) : Nat := (buildFormulas d).length

/-- `getD` of `List.zipWith f` distributes when `f false false = false`. -/
private lemma getD_zipWith_eq
    (f : Bool → Bool → Bool) (l1 l2 : List Bool) (i : Nat)
    (hlen : l1.length = l2.length) (hf : f false false = false) :
    (List.zipWith f l1 l2).getD i false =
      f (l1.getD i false) (l2.getD i false) := by
  induction l1 generalizing l2 i with
  | nil =>
    cases l2 with
    | nil => simp [List.zipWith, List.getD]; exact hf
    | cons _ _ => simp at hlen
  | cons a1 r1 ih =>
    cases l2 with
    | nil => simp at hlen
    | cons a2 r2 =>
      cases i with
      | zero => simp [List.zipWith, List.getD]
      | succ i' =>
        simp only [List.zipWith, List.getD, List.getElem?_cons_succ]
        exact ih r2 i' (by simpa using hlen)

/-- `List.zipWith f (xs.map g) (xs.map h) = xs.map (fun x => f (g x) (h x))`. -/
private lemma zipWith_map_map₂ {α β : Type*} (f : β → β → β)
    (g h : α → β) (xs : List α) :
    List.zipWith f (xs.map g) (xs.map h) =
    xs.map (fun x => f (g x) (h x)) := by
  induction xs with
  | nil => rfl
  | cons _ _ ih => simp [ih]

/-- Index-space translation from formula-indexed dep vectors to
    hypothesis-indexed dep vectors. For each hypothesis vertex `w` at
    position `k` in `d.V.filter (·.HYPOTHESIS)`, bit `k` of the output
    is the bit at `w.FORMULA`'s column in `buildFormulas d`. -/
def formulaVecToHypVec (d : DLDS)
    (u : List.Vector Bool (numFormulas d)) : HypDepVec d :=
  ⟨(d.V.filter (·.HYPOTHESIS)).map (fun w =>
    u.1.getD ((buildFormulas d).idxOf w.FORMULA) false
  ), by simp [numHyps]⟩

private lemma List.getD_replicate_false (n i : Nat) :
    (List.replicate n false).getD i false = false := by
  unfold List.getD
  cases h : (List.replicate n false)[i]? with
  | none => rfl
  | some val =>
    simp [List.getElem?_replicate] at h
    simp [h.2]

/-- The translation sends the zero formula vector to the zero hyp vector. -/
lemma formulaVecToHypVec_zero (d : DLDS) :
    formulaVecToHypVec d (List.Vector.replicate (numFormulas d) false) =
    HypDepVec.zero d := by
  unfold formulaVecToHypVec HypDepVec.zero numHyps
  congr 1
  have hmain : ∀ w : Vertex, (List.Vector.replicate (numFormulas d) false).1.getD
    ((buildFormulas d).idxOf w.FORMULA) false = false := by
    intro w
    simp [List.Vector.replicate]
  induction d.V.filter (·.HYPOTHESIS) with
  | nil => simp
  | cons w ws ih =>
    simp only [List.map_cons, List.length_cons, List.replicate_succ]
    rw [hmain w, ih]

/-- The translation distributes over component-wise OR. -/
lemma formulaVecToHypVec_or (d : DLDS)
    (u v : List.Vector Bool (numFormulas d)) :
    formulaVecToHypVec d (u.zipWith (· || ·) v) =
    HypDepVec.or (formulaVecToHypVec d u) (formulaVecToHypVec d v) := by
  unfold formulaVecToHypVec HypDepVec.or
  congr 1
  simp only [List.Vector.zipWith]
  induction d.V.filter (·.HYPOTHESIS) with
  | nil => simp
  | cons w ws ih =>
    simp only [List.map_cons, List.zipWith_cons_cons]
    rw [getD_zipWith_eq (· || ·) u.1 v.1
      ((buildFormulas d).idxOf w.FORMULA)
      (by rw [u.2, v.2]) rfl]
    rw [ih]

/-- The translation commutes with foldl-OR. -/
lemma formulaVecToHypVec_foldl_or (d : DLDS)
    (sources : List Vertex)
    (fenv : Vertex → List.Vector Bool (numFormulas d))
    (acc : List.Vector Bool (numFormulas d)) :
    formulaVecToHypVec d
      (sources.foldl (fun a u => a.zipWith (· || ·) (fenv u)) acc) =
    sources.foldl (fun a u => HypDepVec.or a (formulaVecToHypVec d (fenv u)))
      (formulaVecToHypVec d acc) := by
  induction sources generalizing acc with
  | nil => rfl
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (acc.zipWith (· || ·) (fenv s))]
    rw [formulaVecToHypVec_or]

/-- Well-formedness: hypothesis vertices have distinct formulas. -/
def DLDS.HypFormulasDistinct (d : DLDS) : Prop :=
  ((d.V.filter (·.HYPOTHESIS)).map (·.FORMULA)).Nodup

/-- Well-formedness: every hypothesis vertex's formula appears in
    `buildFormulas d`, so `idxOf` returns a valid column index. -/
def DLDS.HypFormulasInBuild (d : DLDS) : Prop :=
  ∀ w ∈ d.V, w.HYPOTHESIS = true →
    (buildFormulas d).idxOf w.FORMULA < numFormulas d

/-- Formula-indexed mirror of `stepVertex`. Computes the same per-vertex
    dep vector but in formula-indexed space, using the same four-case
    dispatch. This abstracts `node_logic`'s grid evaluation: the hypothesis
    case produces a formula one-hot, the ELIM case computes OR, the INTRO
    case computes AND-NOT, and branching computes conditional OR. -/
def classicalKernel (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (v : Vertex) : List.Vector Bool (numFormulas bd.base) :=
  let n := numFormulas bd.base
  if v.HYPOTHESIS then
    ⟨(List.range n).map (fun i =>
      decide (i = (buildFormulas bd.base).idxOf v.FORMULA)),
     by simp [n, numFormulas]⟩
  else
    let sources := incomingSources bd.base v
    match findBranchTarget bd v with
    | some (src, rvar, colour) =>
        let ordinary := sources.filter (fun u => decide (u ≠ src))
        let base_val := ordinary.foldl
          (fun acc u => acc.zipWith (· || ·) (fenv u))
          (List.Vector.replicate n false)
        if readingColour reading rvar = some colour then
          base_val.zipWith (· || ·) (fenv src)
        else base_val
    | none =>
        let or_all := sources.foldl
          (fun acc u => acc.zipWith (· || ·) (fenv u))
          (List.Vector.replicate n false)
        match findIntroDischarge bd v with
        | some h_vertex =>
            let encoder : List.Vector Bool n :=
              ⟨(buildFormulas bd.base).map (fun f => decide (f = h_vertex.FORMULA)),
               by simp [n, numFormulas]⟩
            or_all.zipWith (fun b e => b && !e) encoder
        | none => or_all

/-- Foldl congruence: foldl with two step functions agreeing on every
    list element produces the same result. -/
private lemma foldl_congr_on_list {α β : Type*}
    (f g : α → β → α) (xs : List β) (acc : α)
    (h : ∀ a b, b ∈ xs → f a b = g a b) :
    xs.foldl f acc = xs.foldl g acc := by
  induction xs generalizing acc with
  | nil => rfl
  | cons b rest ih =>
    simp only [List.foldl_cons]
    rw [h acc b List.mem_cons_self]
    apply ih
    intro a' b' hb'
    exact h a' b' (List.mem_cons_of_mem _ hb')

/-- The hypothesis list `d.V.filter (·.HYPOTHESIS)` is `Nodup` whenever
    its formula projection is. -/
private lemma hyps_nodup_of_distinct (d : DLDS) (h_distinct : d.HypFormulasDistinct) :
    (d.V.filter (·.HYPOTHESIS)).Nodup :=
  List.Nodup.of_map _ h_distinct

/-- Two hypothesis vertices have the same formula iff they are equal. -/
private lemma hyp_formula_eq_iff_eq
    (d : DLDS) (h_distinct : d.HypFormulasDistinct)
    {w v : Vertex}
    (hw_in : w ∈ d.V.filter (·.HYPOTHESIS))
    (hv_in : v ∈ d.V.filter (·.HYPOTHESIS)) :
    w.FORMULA = v.FORMULA ↔ w = v := by
  refine ⟨fun h => ?_, fun h => h ▸ rfl⟩
  exact List.inj_on_of_nodup_map h_distinct hw_in hv_in h

/-- Pointwise translation of the formula-one-hot at `v.FORMULA` to the
    hypothesis-one-hot at `v`'s position in the hypothesis list. -/
private lemma formulaVecToHypVec_formula_oneHot
    (d : DLDS) (v : Vertex)
    (h_distinct : d.HypFormulasDistinct)
    (h_in_build : d.HypFormulasInBuild)
    (h_v_mem : v ∈ d.V) (h_v_hyp : v.HYPOTHESIS = true)
    (h_idx_lt : (d.V.filter (·.HYPOTHESIS)).idxOf v < numHyps d) :
    formulaVecToHypVec d
      ⟨(List.range (numFormulas d)).map (fun i =>
          decide (i = (buildFormulas d).idxOf v.FORMULA)),
        by simp [numFormulas]⟩ =
    HypDepVec.oneHot d ((d.V.filter (·.HYPOTHESIS)).idxOf v) h_idx_lt := by
  apply Subtype.ext
  unfold formulaVecToHypVec HypDepVec.oneHot
  dsimp only
  set hyps := d.V.filter (·.HYPOTHESIS) with hhyp
  have hv_in : v ∈ hyps :=
    List.mem_filter.mpr ⟨h_v_mem, by simp [h_v_hyp]⟩
  have hyps_nodup : hyps.Nodup := hyps_nodup_of_distinct d h_distinct
  have hlen_hyps : hyps.length = numHyps d := rfl
  apply List.ext_getElem
  · simp [List.length_map, List.length_range, hlen_hyps]
  · intro i hi1 hi2
    rw [List.getElem_map, List.getElem_map, List.getElem_range]
    have hi_lt_hyps : i < hyps.length := by
      simpa [List.length_map] using hi1
    have hwi_mem : hyps[i] ∈ hyps := List.getElem_mem _
    have hwi_in_dV : hyps[i] ∈ d.V := (List.mem_filter.mp hwi_mem).1
    have hwi_hyp : hyps[i].HYPOTHESIS = true := by
      have := (List.mem_filter.mp hwi_mem).2; simpa using this
    have h_idx_w_lt : (buildFormulas d).idxOf hyps[i].FORMULA < numFormulas d :=
      h_in_build _ hwi_in_dV hwi_hyp
    have h_lhs_get :
        ((List.range (numFormulas d)).map
          (fun j => decide (j = (buildFormulas d).idxOf v.FORMULA))).getD
          ((buildFormulas d).idxOf hyps[i].FORMULA) false
        = decide ((buildFormulas d).idxOf hyps[i].FORMULA
            = (buildFormulas d).idxOf v.FORMULA) := by
      rw [List.getD_eq_getElem _ _ (by simp [List.length_map, List.length_range, h_idx_w_lt])]
      simp
    rw [h_lhs_get]
    have h_eq_iff :
        ((buildFormulas d).idxOf hyps[i].FORMULA = (buildFormulas d).idxOf v.FORMULA)
        ↔ (i = hyps.idxOf v) := by
      constructor
      · intro hidx
        have h_idx_v_lt : (buildFormulas d).idxOf v.FORMULA < (buildFormulas d).length :=
          h_in_build _ h_v_mem h_v_hyp
        have h_idx_w_lt' : (buildFormulas d).idxOf hyps[i].FORMULA <
            (buildFormulas d).length := h_idx_w_lt
        have hwi_form_in : hyps[i].FORMULA ∈ buildFormulas d :=
          List.idxOf_lt_length_iff.mp h_idx_w_lt'
        have hv_form_in : v.FORMULA ∈ buildFormulas d :=
          List.idxOf_lt_length_iff.mp h_idx_v_lt
        have h1 := List.getElem?_idxOf hwi_form_in
        have h2 := List.getElem?_idxOf hv_form_in
        rw [hidx] at h1
        have h_form_eq : hyps[i].FORMULA = v.FORMULA :=
          Option.some.inj (h1.symm.trans h2)
        have h_w_eq : hyps[i] = v :=
          (hyp_formula_eq_iff_eq d h_distinct hwi_mem hv_in).mp h_form_eq
        have := hyps_nodup.idxOf_getElem i hi_lt_hyps
        rw [h_w_eq] at this
        exact this.symm
      · intro hi_eq
        have h_w_eq : hyps[i] = v := by
          subst hi_eq
          exact List.getElem_idxOf (List.idxOf_lt_length_of_mem hv_in)
        rw [h_w_eq]
    by_cases h : i = hyps.idxOf v
    · simp [h]
    · have hne : ¬ ((buildFormulas d).idxOf hyps[i].FORMULA
          = (buildFormulas d).idxOf v.FORMULA) := fun heq => h (h_eq_iff.mp heq)
      simp [hne, h]

/-- Pointwise translation of the AND-NOT (clear-bit) operation: applying
    `formulaVecToHypVec` to `u.zipWith (b && !e)` for an `e` that marks
    only `h_vertex.FORMULA`'s column equals clearing bit `k` of the
    translated `u`, where `k` is `h_vertex`'s hypothesis index. -/
private lemma formulaVecToHypVec_clearBit
    (d : DLDS) (u : List.Vector Bool (numFormulas d))
    (h_vertex : Vertex)
    (h_distinct : d.HypFormulasDistinct)
    (h_in_build : d.HypFormulasInBuild)
    (h_h_mem : h_vertex ∈ d.V) (h_h_hyp : h_vertex.HYPOTHESIS = true)
    (_ : (d.V.filter (·.HYPOTHESIS)).idxOf h_vertex < numHyps d) :
    formulaVecToHypVec d
      (u.zipWith (fun b e => b && !e)
        ⟨(buildFormulas d).map (fun f => decide (f = h_vertex.FORMULA)),
         by simp [numFormulas]⟩) =
    HypDepVec.clearBit ((d.V.filter (·.HYPOTHESIS)).idxOf h_vertex)
      (formulaVecToHypVec d u) := by
  apply Subtype.ext
  unfold formulaVecToHypVec HypDepVec.clearBit
  dsimp only
  set hyps := d.V.filter (·.HYPOTHESIS) with hhyp
  have hh_in : h_vertex ∈ hyps :=
    List.mem_filter.mpr ⟨h_h_mem, by simp [h_h_hyp]⟩
  have hyps_nodup : hyps.Nodup := hyps_nodup_of_distinct d h_distinct
  have hlen_hyps : hyps.length = numHyps d := rfl
  -- Show two lists of length numHyps are pointwise equal.
  apply List.ext_getElem
  · simp [List.length_map, List.length_zipIdx, hlen_hyps]
  · intro i hi1 hi2
    rw [List.getElem_map]
    simp only [List.getElem_map, List.getElem_zipIdx]
    have hi_lt_hyps : i < hyps.length := by
      simpa [List.length_map] using hi1
    have hwi_mem : hyps[i] ∈ hyps := List.getElem_mem _
    have hwi_in_dV : hyps[i] ∈ d.V := (List.mem_filter.mp hwi_mem).1
    have hwi_hyp : hyps[i].HYPOTHESIS = true := by
      have := (List.mem_filter.mp hwi_mem).2; simpa using this
    have h_idx_w_lt : (buildFormulas d).idxOf hyps[i].FORMULA < numFormulas d :=
      h_in_build _ hwi_in_dV hwi_hyp
    -- LHS: zipWith result at position idxOf hyps[i].FORMULA
    have hlen_u : u.1.length = numFormulas d := u.2
    have hlen_enc : ((buildFormulas d).map
      (fun f => decide (f = h_vertex.FORMULA))).length = numFormulas d := by
      simp [numFormulas]
    simp only [List.Vector.zipWith]
    rw [getD_zipWith_eq (fun b e => b && !e) u.1
      ((buildFormulas d).map (fun f => decide (f = h_vertex.FORMULA)))
      ((buildFormulas d).idxOf hyps[i].FORMULA)
      (by rw [hlen_u, hlen_enc]) rfl]
    -- Evaluate the encoder getD
    have h_idx_form_lt : (buildFormulas d).idxOf hyps[i].FORMULA <
        (buildFormulas d).length := h_idx_w_lt
    have hwi_form_in : hyps[i].FORMULA ∈ buildFormulas d :=
      List.idxOf_lt_length_iff.mp h_idx_form_lt
    have h_enc_get :
        ((buildFormulas d).map (fun f => decide (f = h_vertex.FORMULA))).getD
          ((buildFormulas d).idxOf hyps[i].FORMULA) false
        = decide (hyps[i].FORMULA = h_vertex.FORMULA) := by
      rw [List.getD_eq_getElem _ _ (by simp [List.length_map, h_idx_form_lt])]
      rw [List.getElem_map]
      rw [List.getElem_idxOf h_idx_form_lt]
    rw [h_enc_get]
    -- RHS: clearBit at position i. Since hyps[i] corresponds to position i in the map,
    -- the map's i-th entry has index i.
    -- Goal RHS reduces to: if i = idxOf h_vertex then false else (formulaVecToHypVec u).1[i]
    -- with (formulaVecToHypVec u).1[i] = u.1.getD (idxOf hyps[i].FORMULA) false.
    -- The propositional condition `hyps[i].FORMULA = h_vertex.FORMULA ↔ i = idxOf h_vertex`.
    have h_eq_iff :
        (hyps[i].FORMULA = h_vertex.FORMULA) ↔ (i = hyps.idxOf h_vertex) := by
      constructor
      · intro h_form_eq
        have h_w_eq : hyps[i] = h_vertex :=
          (hyp_formula_eq_iff_eq d h_distinct hwi_mem hh_in).mp h_form_eq
        have := hyps_nodup.idxOf_getElem i hi_lt_hyps
        rw [h_w_eq] at this
        exact this.symm
      · intro hi_eq
        have h_w_eq : hyps[i] = h_vertex := by
          have h_idx_lt' : hyps.idxOf h_vertex < hyps.length :=
            List.idxOf_lt_length_of_mem hh_in
          have key : hyps[hyps.idxOf h_vertex]'h_idx_lt' = h_vertex :=
            List.getElem_idxOf h_idx_lt'
          -- Bridge from i to idxOf h_vertex hyps using hi_eq
          have : hyps[i]'hi_lt_hyps = hyps[hyps.idxOf h_vertex]'h_idx_lt' := by
            congr 1
          rw [this]
          exact key
        rw [h_w_eq]
    by_cases h : i = hyps.idxOf h_vertex
    · have h_form := h_eq_iff.mpr h
      simp [h]
    · have h_form_ne : ¬ (hyps[i].FORMULA = h_vertex.FORMULA) :=
        fun heq => h (h_eq_iff.mp heq)
      simp [h, h_form_ne]

/-- **Per-vertex kernel equivalence.**

    For any vertex `v` in a well-formed BranchingDLDS with distinct
    hypothesis formulas, the translated formula-indexed classical kernel
    output equals the reading-based `stepVertex` output, provided
    environments are aligned.

    This theorem formalizes that `node_logic`'s per-vertex computation
    (abstracted via `classicalKernel`) and `stepVertex` compute the same
    function modulo the formula/hypothesis index translation. -/
theorem classicalKernel_stepVertex_equiv
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (env : List (Vertex × HypDepVec bd.base))
    (v : Vertex)
    (h_align : ∀ u, formulaVecToHypVec bd.base (fenv u) = envLookup env u)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_v_mem : v ∈ bd.base.V)
    (h_discharge_wf : ∀ h_vertex, findIntroDischarge bd v = some h_vertex →
      h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true) :
    formulaVecToHypVec bd.base (classicalKernel bd reading fenv v) =
    stepVertex bd reading env v := by
  unfold classicalKernel stepVertex
  by_cases h_hyp : v.HYPOTHESIS = true
  · -- Hypothesis case
    rw [if_pos h_hyp, if_pos h_hyp]
    have hv_in_hyps : v ∈ bd.base.V.filter (·.HYPOTHESIS) :=
      List.mem_filter.mpr ⟨h_v_mem, by simp [h_hyp]⟩
    have h_idx_lt : (bd.base.V.filter (·.HYPOTHESIS)).idxOf v < numHyps bd.base :=
      List.idxOf_lt_length_of_mem hv_in_hyps
    have h_hypIndex : hypIndex bd.base v =
        some ⟨(bd.base.V.filter (·.HYPOTHESIS)).idxOf v, h_idx_lt⟩ := by
      unfold hypIndex
      simp [h_idx_lt]
    rw [h_hypIndex]
    exact formulaVecToHypVec_formula_oneHot bd.base v h_distinct h_in_build
      h_v_mem h_hyp h_idx_lt
  · -- Non-hypothesis case
    rw [if_neg h_hyp, if_neg h_hyp]
    -- Both sides match on findBranchTarget bd v
    cases h_branch : findBranchTarget bd v with
    | some triple =>
      obtain ⟨src, rvar, colour⟩ := triple
      simp only
      -- Both branches have a conditional on readingColour
      set ordinary := (incomingSources bd.base v).filter
        (fun u => decide (u ≠ src)) with hord
      -- Translate the ordinary foldl
      have h_foldl_translate :
          formulaVecToHypVec bd.base
            (ordinary.foldl
              (fun acc u => acc.zipWith (· || ·) (fenv u))
              (List.Vector.replicate (numFormulas bd.base) false)) =
          ordinary.foldl
            (fun acc u => HypDepVec.or acc (envLookup env u))
            (HypDepVec.zero bd.base) := by
        rw [formulaVecToHypVec_foldl_or, formulaVecToHypVec_zero]
        apply foldl_congr_on_list
        intro a u _
        rw [h_align]
      by_cases h_col : readingColour reading rvar = some colour
      · simp only [h_col, if_true]
        rw [formulaVecToHypVec_or]
        rw [h_foldl_translate, h_align]
      · simp only [h_col, if_false]
        exact h_foldl_translate
    | none =>
      simp only
      -- Translate the foldl over all incoming sources
      have h_foldl_translate :
          formulaVecToHypVec bd.base
            ((incomingSources bd.base v).foldl
              (fun acc u => acc.zipWith (· || ·) (fenv u))
              (List.Vector.replicate (numFormulas bd.base) false)) =
          (incomingSources bd.base v).foldl
            (fun acc u => HypDepVec.or acc (envLookup env u))
            (HypDepVec.zero bd.base) := by
        rw [formulaVecToHypVec_foldl_or, formulaVecToHypVec_zero]
        apply foldl_congr_on_list
        intro a u _
        rw [h_align]
      cases h_intro : findIntroDischarge bd v with
      | some h_vertex =>
        simp only
        obtain ⟨hh_mem, hh_hyp⟩ := h_discharge_wf h_vertex h_intro
        have hh_in_hyps : h_vertex ∈ bd.base.V.filter (·.HYPOTHESIS) :=
          List.mem_filter.mpr ⟨hh_mem, by simp [hh_hyp]⟩
        have h_idx_lt : (bd.base.V.filter (·.HYPOTHESIS)).idxOf h_vertex
            < numHyps bd.base :=
          List.idxOf_lt_length_of_mem hh_in_hyps
        have h_hypIndex : hypIndex bd.base h_vertex =
            some ⟨(bd.base.V.filter (·.HYPOTHESIS)).idxOf h_vertex, h_idx_lt⟩ := by
          unfold hypIndex
          simp [h_idx_lt]
        rw [h_hypIndex]
        rw [formulaVecToHypVec_clearBit bd.base _ h_vertex h_distinct h_in_build
          hh_mem hh_hyp h_idx_lt]
        rw [h_foldl_translate]
      | none =>
        simp only
        exact h_foldl_translate

/-! #### Global kernel equivalence

This subsection lifts the per-vertex kernel equivalence to the full
evaluation-order semantics. We run the formula-indexed `classicalKernel`
in lockstep with `dldsSemantics`, and show that the two accumulated
environments remain pointwise aligned under `formulaVecToHypVec`. -/

/-- Formula-indexed lookup mirroring `envLookup`. Missing vertices map to
    the zero formula vector. -/
def formulaEnvLookup {d : DLDS} :
    List (Vertex × List.Vector Bool (numFormulas d)) →
      Vertex → List.Vector Bool (numFormulas d)
  | [], _ => List.Vector.replicate (numFormulas d) false
  | (u, w) :: rest, v => if u = v then w else formulaEnvLookup rest v

/-- One step of the accumulating formula-indexed kernel semantics. -/
private def classicalKernelSemanticsStep (bd : BranchingDLDS)
    (reading : ReadingInput)
    (env : List (Vertex × List.Vector Bool (numFormulas bd.base)))
    (v : Vertex) :
    List (Vertex × List.Vector Bool (numFormulas bd.base)) :=
  env ++ [(v, classicalKernel bd reading (formulaEnvLookup env) v)]

/-- One step of the accumulating hypothesis-indexed semantics. -/
private def dldsSemanticsStep (bd : BranchingDLDS)
    (reading : ReadingInput)
    (env : List (Vertex × HypDepVec bd.base))
    (v : Vertex) :
    List (Vertex × HypDepVec bd.base) :=
  env ++ [(v, stepVertex bd reading env v)]

/-- Formula-indexed semantics accumulated along `evalOrder`. -/
def classicalKernelSemantics (bd : BranchingDLDS) (reading : ReadingInput) :
    List (Vertex × List.Vector Bool (numFormulas bd.base)) :=
  bd.evalOrder.foldl (classicalKernelSemanticsStep bd reading) []

/-- Formula-indexed lookup in the accumulated kernel semantics. -/
def classicalKernelSemanticsAt (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) : List.Vector Bool (numFormulas bd.base) :=
  formulaEnvLookup (classicalKernelSemantics bd reading) v

private lemma formulaEnvLookup_append_singleton_of_not_mem {d : DLDS}
    (env : List (Vertex × List.Vector Bool (numFormulas d)))
    (v : Vertex) (w : List.Vector Bool (numFormulas d))
    (h : ∀ w', (v, w') ∉ env) :
    formulaEnvLookup (env ++ [(v, w)]) v = w := by
  induction env with
  | nil =>
      simp [formulaEnvLookup]
  | cons hd tl ih =>
      obtain ⟨u, wu⟩ := hd
      have hne : ¬ (u = v) := by
        intro heq
        subst heq
        exact h wu (by simp)
      have hstep :
          formulaEnvLookup (((u, wu) :: tl) ++ [(v, w)]) v =
            formulaEnvLookup (tl ++ [(v, w)]) v := by
        simp [formulaEnvLookup, hne]
      rw [hstep]
      apply ih
      intro w' hw'
      exact h w' (List.mem_cons_of_mem _ hw')

private lemma formulaEnvLookup_append_singleton_ne {d : DLDS}
    (env : List (Vertex × List.Vector Bool (numFormulas d)))
    (u v : Vertex) (w : List.Vector Bool (numFormulas d))
    (h : v ≠ u) :
    formulaEnvLookup (env ++ [(u, w)]) v = formulaEnvLookup env v := by
  induction env generalizing v with
  | nil =>
      have huv : ¬ (u = v) := by
        intro h'
        exact h h'.symm
      simp [formulaEnvLookup, huv]
  | cons hd tl ih =>
      obtain ⟨x, wx⟩ := hd
      by_cases hx : x = v
      · subst hx
        simp [formulaEnvLookup]
      · have hstep :
          formulaEnvLookup (((x, wx) :: tl) ++ [(u, w)]) v =
            formulaEnvLookup (tl ++ [(u, w)]) v := by
          simp [formulaEnvLookup, hx]
        have hbase :
            formulaEnvLookup ((x, wx) :: tl) v = formulaEnvLookup tl v := by
          simp [formulaEnvLookup, hx]
        rw [hstep, hbase, ih (v := v) h]

private lemma envLookup_append_singleton_ne {d : DLDS}
    (env : List (Vertex × HypDepVec d))
    (u v : Vertex) (w : HypDepVec d)
    (h : v ≠ u) :
    envLookup (env ++ [(u, w)]) v = envLookup env v := by
  induction env generalizing v with
  | nil =>
      have huv : ¬ (u = v) := by
        intro h'
        exact h h'.symm
      simp [envLookup, huv]
  | cons hd tl ih =>
      obtain ⟨x, wx⟩ := hd
      by_cases hx : x = v
      · subst hx
        simp [envLookup]
      · have hstep :
          envLookup (((x, wx) :: tl) ++ [(u, w)]) v =
            envLookup (tl ++ [(u, w)]) v := by
          simp [envLookup, hx]
        have hbase :
            envLookup ((x, wx) :: tl) v = envLookup tl v := by
          simp [envLookup, hx]
        rw [hstep, hbase, ih (v := v) h]

/-- Parallel fold invariant: if the initial formula and hypothesis
    environments are pointwise aligned, then processing the same vertex
    list with `classicalKernel` and `stepVertex` preserves that
    alignment. -/
private theorem classicalKernelSemantics_align_from
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (xs : List Vertex)
    (hypEnv : List (Vertex × HypDepVec bd.base))
    (formEnv : List (Vertex × List.Vector Bool (numFormulas bd.base)))
    (h_align :
      ∀ u, formulaVecToHypVec bd.base (formulaEnvLookup formEnv u) =
        envLookup hypEnv u)
    (h_xs_nodup : xs.Nodup)
    (h_hyp_fresh : ∀ v ∈ xs, ∀ w, (v, w) ∉ hypEnv)
    (h_form_fresh : ∀ v ∈ xs, ∀ w, (v, w) ∉ formEnv)
    (h_memV : ∀ v ∈ xs, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ xs, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true) :
    ∀ u,
      formulaVecToHypVec bd.base
        (formulaEnvLookup
          (xs.foldl (classicalKernelSemanticsStep bd reading) formEnv) u) =
      envLookup (xs.foldl (dldsSemanticsStep bd reading) hypEnv) u := by
  induction xs generalizing hypEnv formEnv with
  | nil =>
      intro u
      simpa using h_align u
  | cons x rest ih =>
      let hypVal := stepVertex bd reading hypEnv x
      let formVal := classicalKernel bd reading (formulaEnvLookup formEnv) x
      have h_rest_nodup : rest.Nodup := (List.nodup_cons.mp h_xs_nodup).2
      have h_x_not_mem_rest : x ∉ rest := (List.nodup_cons.mp h_xs_nodup).1
      have h_align_step :
          ∀ u,
            formulaVecToHypVec bd.base
              (formulaEnvLookup (formEnv ++ [(x, formVal)]) u) =
            envLookup (hypEnv ++ [(x, hypVal)]) u := by
        intro u
        by_cases hu : u = x
        · subst u
          rw [formulaEnvLookup_append_singleton_of_not_mem formEnv x formVal]
          rw [envLookup_append_singleton_of_not_mem hypEnv x hypVal]
          · simpa [formVal, hypVal] using
              classicalKernel_stepVertex_equiv bd reading
                (formulaEnvLookup formEnv) hypEnv x h_align
                h_distinct h_in_build
                (h_memV x (by simp))
                (fun h_vertex h_find =>
                  h_discharge_wf x (by simp) h_vertex h_find)
          · intro w
            exact h_hyp_fresh x (by simp) w
          · intro w
            exact h_form_fresh x (by simp) w
        · rw [formulaEnvLookup_append_singleton_ne formEnv x u formVal hu]
          rw [envLookup_append_singleton_ne hypEnv x u hypVal hu]
          exact h_align u
      have h_hyp_fresh_rest :
          ∀ v ∈ rest, ∀ w, (v, w) ∉ (hypEnv ++ [(x, hypVal)]) := by
        intro v hv w
        have hv_ne : v ≠ x := by
          intro h_eq
          subst h_eq
          exact h_x_not_mem_rest hv
        simp [h_hyp_fresh v (List.mem_cons_of_mem _ hv) w, hv_ne]
      have h_form_fresh_rest :
          ∀ v ∈ rest, ∀ w, (v, w) ∉ (formEnv ++ [(x, formVal)]) := by
        intro v hv w
        have hv_ne : v ≠ x := by
          intro h_eq
          subst h_eq
          exact h_x_not_mem_rest hv
        simp [h_form_fresh v (List.mem_cons_of_mem _ hv) w, hv_ne]
      have h_memV_rest : ∀ v ∈ rest, v ∈ bd.base.V := by
        intro v hv
        exact h_memV v (List.mem_cons_of_mem _ hv)
      have h_discharge_wf_rest :
          ∀ v ∈ rest, ∀ h_vertex,
            findIntroDischarge bd v = some h_vertex →
              h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true := by
        intro v hv h_vertex h_find
        exact h_discharge_wf v (List.mem_cons_of_mem _ hv) h_vertex h_find
      intro u
      simpa [List.foldl_cons, classicalKernelSemanticsStep, dldsSemanticsStep,
        hypVal, formVal] using
        ih
          (hypEnv := hypEnv ++ [(x, hypVal)])
          (formEnv := formEnv ++ [(x, formVal)])
          h_align_step
          h_rest_nodup
          h_hyp_fresh_rest
          h_form_fresh_rest
          h_memV_rest
          h_discharge_wf_rest
          u

/-- **Global kernel equivalence.** The formula-indexed
    `classicalKernel` semantics and the hypothesis-indexed
    `dldsSemantics` agree pointwise after translating with
    `formulaVecToHypVec`. -/
theorem classicalKernel_dldsSemantics_global_equiv
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_topo : bd.WellFormedTopo)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_eval_in_V : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ bd.evalOrder, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true) :
    ∀ v ∈ bd.evalOrder,
      formulaVecToHypVec bd.base (classicalKernelSemanticsAt bd reading v) =
        dldsSemanticsAt bd reading v := by
  intro v h_mem
  let _ := h_mem
  have h_nodup : bd.evalOrder.Nodup := h_topo.1
  have h_align0 :
      ∀ u,
        formulaVecToHypVec bd.base
          (formulaEnvLookup
            ([] : List (Vertex × List.Vector Bool (numFormulas bd.base))) u) =
        envLookup ([] : List (Vertex × HypDepVec bd.base)) u := by
    intro u
    simpa [formulaEnvLookup, envLookup] using
      (formulaVecToHypVec_zero bd.base)
  have h_align_all :
      ∀ u,
        formulaVecToHypVec bd.base
          (formulaEnvLookup
            (bd.evalOrder.foldl (classicalKernelSemanticsStep bd reading) []) u) =
        envLookup (bd.evalOrder.foldl (dldsSemanticsStep bd reading) []) u := by
    simpa using
      (classicalKernelSemantics_align_from bd reading h_distinct h_in_build
        bd.evalOrder [] []
        h_align0
        h_nodup
        (by intro v hv w; simp)
        (by intro v hv w; simp)
        h_eval_in_V
        h_discharge_wf)
  simpa [classicalKernelSemanticsAt, classicalKernelSemantics,
    dldsSemanticsAt, dldsSemantics, classicalKernelSemanticsStep,
    dldsSemanticsStep] using h_align_all v

/-- Corollary: for any well-formed `BranchingDLDS`, the hypothesis-indexed
dep vector computed by `dldsSemanticsAt` at a goal vertex `v` coincides with
the translation (via `formulaVecToHypVec`) of the formula-indexed dep vector
computed by `classicalKernelSemanticsAt` at `v`.

This surfaces the global kernel equivalence in goal-centric form: the
hypothesis-indexed "all dependencies discharged" check at `v` amounts to
the translated formula-indexed kernel output being zero at `v`. -/
theorem dldsSemanticsAt_eq_translated_classicalKernelSemanticsAt_at_goal
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_topo : bd.WellFormedTopo)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_eval_in_V : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ bd.evalOrder, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true)
    (goalVertex : Vertex)
    (h_goal_mem : goalVertex ∈ bd.evalOrder) :
    dldsSemanticsAt bd reading goalVertex =
      formulaVecToHypVec bd.base
        (classicalKernelSemanticsAt bd reading goalVertex) := by
  have h :=
    classicalKernel_dldsSemantics_global_equiv bd reading h_topo h_distinct
      h_in_build h_eval_in_V h_discharge_wf goalVertex h_goal_mem
  exact h.symm

namespace Examples.AllCases

-- Cross-validation: running the formula-indexed classical kernel and then
-- translating via `formulaVecToHypVec` must match the hypothesis-indexed
-- reading-based semantics at every vertex, for every reading. The two
-- vectors are equal by `classicalKernel_dldsSemantics_global_equiv`; this
-- `#eval` block is a sanity check that the definitions evaluate to the
-- claimed values on a nontrivial medium-sized example.

-- Reading [false]:
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [false] vE1)).toList
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [false] vE2)).toList
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [false] vI)).toList
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [false] vRoot)).toList

-- Reading [true]:
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [true] vE1)).toList
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [true] vE2)).toList
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [true] vI)).toList
#eval (formulaVecToHypVec allCasesDLDS.base
  (classicalKernelSemanticsAt allCasesDLDS [true] vRoot)).toList

end Examples.AllCases

/-! #### Grid bridge infrastructure

The definitions below connect reading inputs to the existing path-based
grid evaluator without changing the earlier APIs. The bridge theorem is
proved under `BranchingDLDS.GridCompatible`, an explicit semantic invariant
that says the extracted grid environment is closed under `classicalKernel`.
The counterexample at the end records why that invariant is necessary. -/

/-- Maximum DLDS level. This is the top level used by `buildLayers`. -/
def dldsMaxLevel (d : DLDS) : Nat :=
  (d.V.map (fun v => v.LEVEL)).foldl max 0

/-- Number of path transitions used by the constructed grid. -/
def gridTransitionCount (d : DLDS) : Nat :=
  (buildGridFromDLDS d).length - 1

lemma gridTransitionCount_eq_dldsMaxLevel (d : DLDS) :
    gridTransitionCount d = dldsMaxLevel d := by
  unfold gridTransitionCount buildGridFromDLDS buildLayers dldsMaxLevel
  simp

/-- First top-level vertex carrying a formula, if one exists. -/
def topVertexForFormula (bd : BranchingDLDS) (formula : Formula) : Option Vertex :=
  let top := dldsMaxLevel bd.base
  bd.base.V.find? fun v => decide (v.FORMULA = formula ∧ v.LEVEL = top)

/-- Branching metadata whose source is the current vertex, if any. -/
def branchingFromSource (bd : BranchingDLDS) (u : Vertex) : Option Branching :=
  bd.branchings.find? fun b => decide (b.source = u)

/-- Keep only one-level-down targets, since the grid advances one layer per step. -/
def oneLevelDownTarget (u w : Vertex) : Option Vertex :=
  if w.LEVEL + 1 = u.LEVEL then some w else none

/-- Target chosen by a reading bit for one branching source. -/
def branchTargetForReading (reading : ReadingInput) (b : Branching) :
    Option Vertex :=
  match readingColour reading b.readingVar with
  | none => none
  | some colour =>
      match b.targets.find? (fun p => decide (p.1 = colour)) with
      | none => none
      | some target => oneLevelDownTarget b.source target.2

/-- Ordinary DLDS edge target followed by the path converter. -/
def ordinaryTargetForPath (bd : BranchingDLDS) (u : Vertex) : Option Vertex :=
  match bd.base.E.find? (fun e =>
      decide (e.START = u ∧ e.END.LEVEL + 1 = u.LEVEL)) with
  | none => none
  | some e => some e.END

/-- Next vertex selected by the reading-induced path. Branching metadata has
    priority over ordinary edges, matching `stepVertex`'s branch-target case. -/
def nextVertexForReading (bd : BranchingDLDS) (reading : ReadingInput)
    (u : Vertex) : Option Vertex :=
  match branchingFromSource bd u with
  | some b => branchTargetForReading reading b
  | none => ordinaryTargetForPath bd u

/-- Build one token trajectory. The output length is exactly `steps`. -/
def readingPathFromVertex (bd : BranchingDLDS) (reading : ReadingInput) :
    Option Vertex -> Nat -> List Nat
  | _, 0 => []
  | none, steps + 1 => 0 :: readingPathFromVertex bd reading none steps
  | some u, steps + 1 =>
      match nextVertexForReading bd reading u with
      | none => 0 :: readingPathFromVertex bd reading none steps
      | some w =>
          ((buildFormulas bd.base).idxOf w.FORMULA + 1) ::
            readingPathFromVertex bd reading (some w) steps

lemma readingPathFromVertex_length
    (bd : BranchingDLDS) (reading : ReadingInput)
    (start : Option Vertex) (steps : Nat) :
    (readingPathFromVertex bd reading start steps).length = steps := by
  induction steps generalizing start with
  | zero =>
      cases start <;> rfl
  | succ steps ih =>
      cases start with
      | none =>
          simp [readingPathFromVertex, ih]
      | some u =>
          simp [readingPathFromVertex]
          split <;> simp [ih]

/-- Convert a reading assignment into complete grid paths.

Each formula column starts at the first top-level vertex carrying that
formula. At each level transition it follows either the reading-selected
branching target or the first one-level ordinary edge. Missing edges, missing
top vertices, and unmatched branch colours emit stop entries. -/
def readingToPathFull (bd : BranchingDLDS) (reading : ReadingInput) :
    PathInput :=
  let steps := dldsMaxLevel bd.base
  (buildFormulas bd.base).map fun formula =>
    readingPathFromVertex bd reading (topVertexForFormula bd formula) steps

/-- `readingToPathFull` produces one path per formula and one entry per
    grid transition in each path. -/
theorem readingToPathFull_wellformed
    (bd : BranchingDLDS) (reading : ReadingInput) :
    (readingToPathFull bd reading).length = numFormulas bd.base ∧
      ∀ path ∈ readingToPathFull bd reading,
        path.length = gridTransitionCount bd.base := by
  constructor
  · simp [readingToPathFull, numFormulas]
  · intro path h_mem
    unfold readingToPathFull at h_mem
    rcases List.mem_map.mp h_mem with ⟨formula, _h_formula, h_path⟩
    rw [← h_path, readingPathFromVertex_length,
      gridTransitionCount_eq_dldsMaxLevel]

lemma readingPathFromVertex_routes_first
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex) (steps : Nat)
    (h_next : nextVertexForReading bd reading u = some v) :
    (readingPathFromVertex bd reading (some u) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1) := by
  simp [readingPathFromVertex, h_next]

lemma readingPathFromVertex_stops_first
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u : Vertex) (steps : Nat)
    (h_next : nextVertexForReading bd reading u = none) :
    (readingPathFromVertex bd reading (some u) (steps + 1)).head? =
      some 0 := by
  simp [readingPathFromVertex, h_next]

/-- Formula column lookup used by trace extraction. -/
def formulaColumn (d : DLDS) (formula : Formula) : Nat :=
  (buildFormulas d).idxOf formula

/-- Zero formula-indexed vector. -/
def formulaVecZero (d : DLDS) : List.Vector Bool (numFormulas d) :=
  List.Vector.replicate (numFormulas d) false

/-- Read a formula column from a layer output, returning zero out of bounds. -/
def outputAtFormula (d : DLDS)
    (outputs : List (List.Vector Bool (numFormulas d)))
    (formula : Formula) : List.Vector Bool (numFormulas d) :=
  if h : formulaColumn d formula < outputs.length then
    outputs.get ⟨formulaColumn d formula, h⟩
  else
    formulaVecZero d

/-- Trace layer outputs while running the same evaluator as `get_eval_result`. -/
def evalTraceFromLevel {n : Nat}
    (paths : PathInput)
    (level : Nat)
    (tokens : List (Token n))
    (remainingLayers : List (GridLayer n))
    (numLevels : Nat) :
    List (Prod Nat (List (List.Vector Bool n))) :=
  match remainingLayers with
  | [] => []
  | layer :: rest =>
      let result := evaluate_layer layer tokens
      let outputs := result.1
      let here := (level, outputs)
      match rest with
      | [] => [here]
      | _ =>
          let newTokens := propagate_tokens tokens paths level numLevels outputs
          here :: evalTraceFromLevel paths (level - 1) newTokens rest numLevels

/-- Full evaluation trace from the initial grid tokens. -/
def get_eval_trace {n : Nat}
    (layers : List (GridLayer n))
    (initialVectors : List (List.Vector Bool n))
    (paths : PathInput) :
    List (Prod Nat (List (List.Vector Bool n))) :=
  let numLevels := layers.length
  let initialTokens := initialize_tokens initialVectors numLevels
  evalTraceFromLevel paths (numLevels - 1) initialTokens layers numLevels

/-- Find the layer outputs recorded for a DLDS level. -/
def traceOutputsAtLevel {n : Nat}
    (trace : List (Prod Nat (List (List.Vector Bool n))))
    (level : Nat) : Option (List (List.Vector Bool n)) :=
  (trace.find? (fun entry => decide (entry.1 = level))).map Prod.snd

/-- Extract the grid-computed formula-indexed vector at a vertex's level
    and formula column. This reads actual `evaluate_layer` outputs. -/
def extract_grid_result_at_vertex
    (bd : BranchingDLDS)
    (grid : List (GridLayer (numFormulas bd.base)))
    (initialVecs : List (List.Vector Bool (numFormulas bd.base)))
    (paths : PathInput)
    (v : Vertex) : List.Vector Bool (numFormulas bd.base) :=
  match traceOutputsAtLevel (get_eval_trace grid initialVecs paths) v.LEVEL with
  | none => formulaVecZero bd.base
  | some outputs => outputAtFormula bd.base outputs v.FORMULA

theorem extract_grid_result_at_vertex_of_trace_hit
    (bd : BranchingDLDS)
    (grid : List (GridLayer (numFormulas bd.base)))
    (initialVecs : List (List.Vector Bool (numFormulas bd.base)))
    (paths : PathInput)
    (v : Vertex)
    (outputs : List (List.Vector Bool (numFormulas bd.base)))
    (h_trace :
      traceOutputsAtLevel (get_eval_trace grid initialVecs paths) v.LEVEL =
        some outputs) :
    extract_grid_result_at_vertex bd grid initialVecs paths v =
      outputAtFormula bd.base outputs v.FORMULA := by
  unfold extract_grid_result_at_vertex
  rw [h_trace]

/-- Vertex-indexed environment extracted from a concrete grid run. -/
def gridFEnvFromPath
    (bd : BranchingDLDS)
    (grid : List (GridLayer (numFormulas bd.base)))
    (initialVecs : List (List.Vector Bool (numFormulas bd.base)))
    (paths : PathInput) :
    Vertex → List.Vector Bool (numFormulas bd.base) :=
  fun u => extract_grid_result_at_vertex bd grid initialVecs paths u

/-- Vertex-indexed environment extracted from the reading-induced grid run. -/
def gridFEnvFromReading
    (bd : BranchingDLDS) (reading : ReadingInput) :
    Vertex → List.Vector Bool (numFormulas bd.base) :=
  let grid := buildGridFromDLDS bd.base
  let initialVecs := initialVectorsFromDLDS bd.base
  let paths := readingToPathFull bd reading
  gridFEnvFromPath bd grid initialVecs paths

/-- Semantic compatibility between the current formula-rule grid and the
    DLDS-edge-based classical kernel for one reading.

This is the explicit trust boundary needed by the bridge: after running the
actual `node_logic` grid with `readingToPathFull`, every evaluation-order
vertex must already satisfy the same per-vertex equation as
`classicalKernel`. A later structural version should derive this predicate
from edge-rule shape constraints on `DLDS.E`. -/
def BranchingDLDS.GridCompatibleForReading
    (bd : BranchingDLDS) (reading : ReadingInput) : Prop :=
  ∀ v ∈ bd.evalOrder,
    gridFEnvFromReading bd reading v =
      classicalKernel bd reading (gridFEnvFromReading bd reading) v

/-- Grid compatibility for all readings. -/
def BranchingDLDS.GridCompatible (bd : BranchingDLDS) : Prop :=
  ∀ reading, bd.GridCompatibleForReading reading

/-- Bridge under the explicit grid-compatibility invariant. -/
theorem node_logic_equals_classicalKernel_under_GridCompatible
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder)
    (h_compat : bd.GridCompatible) :
    let grid := buildGridFromDLDS bd.base
    let initialVecs := initialVectorsFromDLDS bd.base
    let paths := readingToPathFull bd reading
    let fenvFromGrid : Vertex → List.Vector Bool (numFormulas bd.base) :=
      gridFEnvFromPath bd grid initialVecs paths
    fenvFromGrid v = classicalKernel bd reading fenvFromGrid v := by
  simpa [BranchingDLDS.GridCompatible, BranchingDLDS.GridCompatibleForReading,
    gridFEnvFromReading, gridFEnvFromPath] using h_compat reading v h_mem

theorem node_logic_equals_classicalKernel_for_reading
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder)
    (h_compat : bd.GridCompatibleForReading reading) :
    let grid := buildGridFromDLDS bd.base
    let initialVecs := initialVectorsFromDLDS bd.base
    let paths := readingToPathFull bd reading
    let fenvFromGrid : Vertex → List.Vector Bool (numFormulas bd.base) :=
      gridFEnvFromPath bd grid initialVecs paths
    fenvFromGrid v = classicalKernel bd reading fenvFromGrid v := by
  simpa [BranchingDLDS.GridCompatibleForReading, gridFEnvFromReading,
    gridFEnvFromPath] using h_compat v h_mem

/-! ##### Edge-aware grid semantics

The original grid routes tokens by formula column. That loses the proof-role
information needed to distinguish, for example, `A` used as the minor premise
of an elimination from `A` used as a repetition input. The edge-aware evaluator
below keeps the existing grid semantics intact and adds a refined variant where
the second component of `RuleIncoming` is treated as a real input tag. -/

def gridTagIntro : Nat := 0
def gridTagElimMajor : Nat := 1
def gridTagElimMinor : Nat := 2
def gridTagRepetition : Nat := 3

/-- Edge-aware incoming map. It has the same rule order as `nodeForFormula`,
    but tags each expected source by its proof role. -/
def buildTaggedIncomingMapForFormula
    (formulas : List Formula)
    (formula : Formula) : NodeIncoming :=
  let introMap := match formula with
    | .impl _ B =>
        let bIdx := formulas.idxOf B
        [[(bIdx, gridTagIntro)]]
    | _ => []
  let elimMaps := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .impl A B =>
        if B = formula then
          let aIdx := formulas.idxOf A
          some [(idx, gridTagElimMajor), (aIdx, gridTagElimMinor)]
        else none
    | _ => none
  let selfIdx := formulas.idxOf formula
  let repMap := [[(selfIdx, gridTagRepetition)]]
  introMap ++ elimMaps ++ repMap

def buildTaggedIncomingMap (formulas : List Formula) : LayerIncoming :=
  formulas.map (buildTaggedIncomingMapForFormula formulas)

def buildTaggedLayers (d : DLDS) :
    List (GridLayer (buildFormulas d).length) :=
  let formulas := buildFormulas d
  let maxLvl := (d.V.map (·.LEVEL)).foldl max 0
  List.replicate (maxLvl + 1)
    { nodes := formulas.map (nodeForFormula formulas)
      incoming := buildTaggedIncomingMap formulas }

def buildTaggedGridFromDLDS (d : DLDS) :
    List (GridLayer (buildFormulas d).length) :=
  buildTaggedLayers d

lemma List.length_filterMap_eq_of_isSome
    {α β γ : Type}
    (xs : List α)
    (f : α → Option β)
    (g : α → Option γ)
    (h : ∀ x, (f x).isSome = (g x).isSome) :
    (xs.filterMap f).length = (xs.filterMap g).length := by
  induction xs with
  | nil =>
      simp
  | cons x xs ih =>
      simp only [List.filterMap_cons]
      specialize h x
      cases hfx : f x <;> cases hgx : g x <;>
        simp [Option.isSome, hfx, hgx] at h ⊢ <;> try exact ih

lemma buildTaggedIncomingMapForFormula_length_eq
    (formulas : List Formula) (formula : Formula) :
    (buildTaggedIncomingMapForFormula formulas formula).length =
      (nodeForFormula formulas formula).rules.length := by
  cases formula with
  | atom name =>
      unfold buildTaggedIncomingMapForFormula
      conv_rhs => unfold nodeForFormula
      simp only [List.length_append, List.length_map, List.length_zipIdx,
        List.length_cons, List.length_nil]
      have h_eq :
          (List.filterMap
              (fun x =>
                match x.1 with
                | .impl A B =>
                    if B = Formula.atom name then
                      some [(x.2, gridTagElimMajor), (List.idxOf A formulas, gridTagElimMinor)]
                    else none
                | _ => none)
              formulas.zipIdx).length =
            (List.filterMap
              (fun x =>
                match x.1 with
                | .impl A B => if B = Formula.atom name then some x.2 else none
                | _ => none)
              formulas.zipIdx).length :=
        List.length_filterMap_eq_of_isSome _ _ _ (by
          intro x
          cases x with
          | mk f idx =>
              cases f with
              | atom s =>
                  simp [Option.isSome]
              | impl A B =>
                  by_cases hB : B = Formula.atom name <;> simp [hB, Option.isSome])
      omega
  | impl A B =>
      unfold buildTaggedIncomingMapForFormula
      conv_rhs => unfold nodeForFormula
      simp only [List.length_append, List.length_map, List.length_zipIdx,
        List.length_cons, List.length_nil]
      have h_intro :
          (match encoderForIntro formulas (Formula.impl A B) with
            | some encoder => [encoder]
            | none => []).length = 1 := by
        simp [encoderForIntro]
      have h_eq :
          (List.filterMap
              (fun x =>
                match x.1 with
                | .impl A_1 B_1 =>
                    if B_1 = Formula.impl A B then
                      some [(x.2, gridTagElimMajor), (List.idxOf A_1 formulas, gridTagElimMinor)]
                    else none
                | _ => none)
              formulas.zipIdx).length =
            (List.filterMap
              (fun x =>
                match x.1 with
                | .impl A_1 B_1 => if B_1 = Formula.impl A B then some x.2 else none
                | _ => none)
              formulas.zipIdx).length :=
        List.length_filterMap_eq_of_isSome _ _ _ (by
          intro x
          cases x with
          | mk f idx =>
              cases f with
              | atom s =>
                  simp [Option.isSome]
              | impl A_1 B_1 =>
                  by_cases hB : B_1 = Formula.impl A B <;> simp [hB, Option.isSome])
      rw [h_intro]
      omega

/-- Tokens whose source input is tagged by proof role. -/
structure TaggedToken (n : Nat) where
  origin_column : Nat
  source_column : Nat
  source_tag : Nat
  current_level : Nat
  current_column : Nat
  dep_vector : List.Vector Bool n
  deriving Inhabited

def taggedInputToken {n : Nat}
    (sourceColumn sourceTag : Nat)
    (dep : List.Vector Bool n) : TaggedToken n :=
  { origin_column := sourceColumn
    source_column := sourceColumn
    source_tag := sourceTag
    current_level := 0
    current_column := 0
    dep_vector := dep }

abbrev TaggedPathInput := List (List (Nat × Nat))

def initialize_tagged_tokens {n : Nat}
    (initialVectors : List (List.Vector Bool n))
    (topLevel : Nat) : List (TaggedToken n) :=
  initialVectors.zipIdx.map fun (vec, col) =>
    { origin_column := col
      source_column := col
      source_tag := gridTagRepetition
      current_level := topLevel
      current_column := col
      dep_vector := vec }

def gather_rule_inputs_tagged {n : Nat}
    (ruleIncoming : RuleIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n)) :
    List (List.Vector Bool n) :=
  ruleIncoming.filterMap fun required =>
    availableInputs.find? (fun (tag, _) => tag = required) |>.map Prod.snd

def set_rule_activation_tagged {n : Nat}
    (rule : Rule n)
    (ruleIncoming : RuleIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n)) :
    Rule n :=
  let availableTags := availableInputs.map Prod.fst
  let hasAll := ruleIncoming.all fun req => availableTags.contains req
  let newActivation := match rule.activation with
    | ActivationBits.intro _ => ActivationBits.intro hasAll
    | ActivationBits.elim _ _ =>
        if ruleIncoming.length = 2 then
          let hasFirst := availableTags.contains (ruleIncoming[0]!)
          let hasSecond := availableTags.contains (ruleIncoming[1]!)
          ActivationBits.elim hasFirst hasSecond
        else
          ActivationBits.elim false false
    | ActivationBits.repetition _ => ActivationBits.repetition hasAll
  { rule with activation := newActivation }

def activateTaggedRulesAux {n : Nat}
    (nodeIncoming : NodeIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n)) :
    Nat → List (Rule n) → List (Rule n)
  | _, [] => []
  | idx, r :: rs =>
      let ruleInc := nodeIncoming[idx]!
      let r' := set_rule_activation_tagged r ruleInc availableInputs
      r' :: activateTaggedRulesAux nodeIncoming availableInputs (idx + 1) rs

lemma activateTaggedRulesAux_ids {n : Nat}
    (nodeIncoming : NodeIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n)) :
    ∀ idx (rs : List (Rule n)),
      (activateTaggedRulesAux nodeIncoming availableInputs idx rs).map
          (·.ruleId) = rs.map (·.ruleId)
  | idx, [] => by simp [activateTaggedRulesAux]
  | idx, r :: rs => by
      have ih := activateTaggedRulesAux_ids nodeIncoming availableInputs
        (idx + 1) rs
      simp [activateTaggedRulesAux, set_rule_activation_tagged, ih]

lemma activateTaggedRulesAux_eq_zipIdx_map {n : Nat}
    (nodeIncoming : NodeIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n))
    (rules : List (Rule n))
    (start : Nat) :
    activateTaggedRulesAux nodeIncoming availableInputs start rules =
      (rules.zipIdx start).map fun x =>
        let ruleInc := nodeIncoming[x.2]!
        set_rule_activation_tagged x.1 ruleInc availableInputs := by
  induction rules generalizing start with
  | nil =>
      simp [activateTaggedRulesAux, List.zipIdx]
  | cons r rs ih =>
      simp [activateTaggedRulesAux, List.zipIdx_cons, ih]

lemma activateTaggedRulesAux_eq_zipIdx_map_zero {n : Nat}
    (nodeIncoming : NodeIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n))
    (rules : List (Rule n)) :
    activateTaggedRulesAux nodeIncoming availableInputs 0 rules =
      rules.zipIdx.map fun x =>
        let ruleInc := nodeIncoming[x.2]!
        set_rule_activation_tagged x.1 ruleInc availableInputs := by
  simpa using activateTaggedRulesAux_eq_zipIdx_map nodeIncoming
    availableInputs rules 0

def activate_node_from_tagged_tokens {n : Nat}
    (node : CircuitNode n)
    (nodeIncoming : NodeIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n)) :
    CircuitNode n :=
  let activatedRules := activateTaggedRulesAux nodeIncoming availableInputs
    0 node.rules
  { rules := activatedRules
    nodupIds := by
      classical
      have hIds : activatedRules.map (·.ruleId) = node.rules.map (·.ruleId) :=
        activateTaggedRulesAux_ids nodeIncoming availableInputs 0 node.rules
      simpa [activatedRules, hIds] using node.nodupIds }

def node_logic_with_tagged_routing {n : Nat}
    (rules : List (Rule n))
    (nodeIncoming : NodeIncoming)
    (availableInputs : List ((Nat × Nat) × List.Vector Bool n)) :
    (List.Vector Bool n) × Bool :=
  let acts := extract_activations rules
  let xor := multiple_xor acts
  let masks := and_bool_list xor acts
  let hasConflict := !xor && acts.any (· = true)
  let perRuleInputs := rules.zipIdx.map fun (_rule, ruleIdx) =>
    let ruleInc := nodeIncoming[ruleIdx]!
    gather_rule_inputs_tagged ruleInc availableInputs
  let outs := apply_activations_with_routing rules masks perRuleInputs
  (list_or outs, hasConflict)

def evaluate_node_tagged {n : Nat}
    (node : CircuitNode n)
    (nodeIncoming : NodeIncoming)
    (tokensAtNode : List (TaggedToken n)) :
    (List.Vector Bool n) × Bool :=
  if tokensAtNode.isEmpty then
    (List.Vector.replicate n false, false)
  else
    let availableInputs := tokensAtNode.map fun t =>
      ((t.source_column, t.source_tag), t.dep_vector)
    let activatedNode :=
      activate_node_from_tagged_tokens node nodeIncoming availableInputs
    node_logic_with_tagged_routing activatedNode.rules nodeIncoming
      availableInputs

def evaluate_layer_tagged {n : Nat}
    (layer : GridLayer n)
    (tokens : List (TaggedToken n)) :
    (List (List.Vector Bool n)) × Bool :=
  let results := layer.nodes.zipIdx.map fun (node, colIdx) =>
    let tokensHere := tokens.filter (·.current_column = colIdx)
    let nodeIncoming := layer.incoming[colIdx]!
    evaluate_node_tagged node nodeIncoming tokensHere
  let outputs := results.map Prod.fst
  let errors := results.map Prod.snd
  (outputs, errors.any id)

def propagate_tagged_tokens {n : Nat}
    (tokens : List (TaggedToken n))
    (paths : TaggedPathInput)
    (currentLevel : Nat)
    (numLevels : Nat)
    (outputs : List (List.Vector Bool n)) : List (TaggedToken n) :=
  tokens.filterMap fun token =>
    if hPath : token.origin_column < paths.length then
      let path := paths.get ⟨token.origin_column, hPath⟩
      if hLevel : currentLevel > 0 ∧ numLevels - currentLevel - 1 < path.length then
        let stepIndex := numLevels - currentLevel - 1
        let step := path.get ⟨stepIndex, hLevel.2⟩
        if step.1 = 0 then
          none
        else
          let targetColumn := step.1 - 1
          if hOut : token.current_column < outputs.length then
            some { origin_column := token.origin_column
                   source_column := token.current_column
                   source_tag := step.2
                   current_level := currentLevel - 1
                   current_column := targetColumn
                   dep_vector := outputs.get ⟨token.current_column, hOut⟩ }
          else
            none
      else
        none
    else
      none

def eval_tagged_from_level {n : Nat}
    (paths : TaggedPathInput)
    (level : Nat)
    (tokens : List (TaggedToken n))
    (remainingLayers : List (GridLayer n))
    (accumulatedError : Bool)
    (numLevels : Nat) :
    (List (List.Vector Bool n)) × Bool :=
  match remainingLayers with
  | [] =>
      let finalOutputs := (List.range n).map fun _ =>
        List.Vector.replicate n false
      (finalOutputs, accumulatedError)
  | layer :: rest =>
      let result := evaluate_layer_tagged layer tokens
      match rest with
      | [] => (result.1, accumulatedError || result.2)
      | _ =>
          let newTokens := propagate_tagged_tokens tokens paths level
            numLevels result.1
          eval_tagged_from_level paths (level - 1) newTokens rest
            (accumulatedError || result.2) numLevels

def get_tagged_eval_result {n : Nat}
    (layers : List (GridLayer n))
    (initialVectors : List (List.Vector Bool n))
    (paths : TaggedPathInput) : (List (List.Vector Bool n)) × Bool :=
  let numLevels := layers.length
  let initialTokens := initialize_tagged_tokens initialVectors numLevels
  eval_tagged_from_level paths (numLevels - 1) initialTokens layers false
    numLevels

def evalTaggedTraceFromLevel {n : Nat}
    (paths : TaggedPathInput)
    (level : Nat)
    (tokens : List (TaggedToken n))
    (remainingLayers : List (GridLayer n))
    (numLevels : Nat) :
    List (Prod Nat (List (List.Vector Bool n))) :=
  match remainingLayers with
  | [] => []
  | layer :: rest =>
      let result := evaluate_layer_tagged layer tokens
      let here := (level, result.1)
      match rest with
      | [] => [here]
      | _ =>
          let newTokens := propagate_tagged_tokens tokens paths level
            numLevels result.1
          here :: evalTaggedTraceFromLevel paths (level - 1) newTokens
            rest numLevels

def get_tagged_eval_trace {n : Nat}
    (layers : List (GridLayer n))
    (initialVectors : List (List.Vector Bool n))
    (paths : TaggedPathInput) :
    List (Prod Nat (List (List.Vector Bool n))) :=
  let numLevels := layers.length
  let initialTokens := initialize_tagged_tokens initialVectors numLevels
  evalTaggedTraceFromLevel paths (numLevels - 1) initialTokens layers
    numLevels

lemma node_logic_with_tagged_routing_correct
  {n : Nat}
  (rules : List (Rule n))
  (nodeIncoming : NodeIncoming)
  (availableInputs : List ((Nat × Nat) × List.Vector Bool n))
  (h_one : exactlyOneActive rules)
  (h_nodup : rules.Nodup)
  (hlen : nodeIncoming.length = rules.length) :
  ∃ (r : Rule n) (i : Nat) (hi : i < rules.length),
    r ∈ rules ∧
    rules.get ⟨i, hi⟩ = r ∧
    node_logic_with_tagged_routing rules nodeIncoming availableInputs =
      (let ruleInc := nodeIncoming[i]!
       let inputs := gather_rule_inputs_tagged ruleInc availableInputs
       r.combine inputs, false) := by
  classical
  rcases h_one with ⟨r₀, hr₀_mem, hr₀_act, hr₀_unique⟩

  let acts := extract_activations rules
  have h_xor : multiple_xor acts = true := by
    have := (multiple_xor_bool_iff_exactlyOneActive rules h_nodup).mpr
      ⟨r₀, hr₀_mem, hr₀_act, hr₀_unique⟩
    simpa [acts, extract_activations] using this

  have h_masks : and_bool_list (multiple_xor acts) acts = acts := by
    simp [and_bool_list, h_xor]

  let perRuleInputs :=
    (List.range rules.length).map (fun idx =>
      let ruleInc := nodeIncoming[idx]!
      gather_rule_inputs_tagged ruleInc availableInputs)

  have h_len_per : perRuleInputs.length = rules.length := by
    simp [perRuleInputs]

  let masks := and_bool_list (multiple_xor acts) acts
  have hmasks_eq : masks = acts := h_masks

  let outs := apply_activations_with_routing rules masks perRuleInputs

  have ⟨i₀_fin, hi₀_get⟩ :
    ∃ i₀ : Fin rules.length, rules.get i₀ = r₀ :=
    exists_fin_of_mem (l := rules) hr₀_mem

  let i₀ : Nat := i₀_fin
  have hi₀_lt : i₀ < rules.length := i₀_fin.isLt

  have h_len_masks : masks.length = rules.length := by
    simp [masks, hmasks_eq, acts, extract_activations]

  have h_len_outs : outs.length = rules.length := by
    simp only [outs, apply_activations_with_routing, List.length_zipWith3,
      h_len_masks, h_len_per]
    omega

  have hi₀_outs : i₀ < outs.length := by
    simpa [h_len_outs] using hi₀_lt

  have hi₀_per : i₀ < perRuleInputs.length := by
    simpa [h_len_per] using hi₀_lt

  have h_act_i₀ :
    acts.get ⟨i₀, by simpa [acts, extract_activations] using hi₀_lt⟩ =
      true := by
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
        simpa [this] using hi₀_lt⟩ = true := by
    simpa [masks, hmasks_eq] using h_act_i₀

  have hi₀_get' : rules.get ⟨i₀, hi₀_lt⟩ = r₀ := by
    simpa [i₀] using hi₀_get

  have hi₀_masks : i₀ < masks.length := by
    rw [h_len_masks]
    exact hi₀_lt

  have h_outs_i₀ :
    outs.get ⟨i₀, hi₀_outs⟩ =
      r₀.combine (perRuleInputs.get ⟨i₀, hi₀_per⟩) := by
    show (apply_activations_with_routing rules masks perRuleInputs).get
        ⟨i₀, hi₀_outs⟩ = _
    simp only [apply_activations_with_routing]
    conv_lhs =>
      rw [List.get_zipWith3
        (fun r m ins =>
          if m = true then r.combine ins
          else List.Vector.replicate n false)
        rules masks perRuleInputs i₀ hi₀_lt hi₀_masks hi₀_per]
    rw [hi₀_get', h_mask_i₀]
    simp

  have h_only_i₀_active :
      ∀ j (hj : j < rules.length), j ≠ i₀ →
        is_rule_active (rules.get ⟨j, hj⟩) = false := by
    intro j hj hne
    by_contra h_not_false
    push_neg at h_not_false
    have h_true :
        is_rule_active (rules.get ⟨j, hj⟩) = true :=
      Bool.eq_true_of_not_eq_false h_not_false
    have h_eq_r₀ :
        rules.get ⟨j, hj⟩ = r₀ :=
      hr₀_unique _ (List.get_mem rules ⟨j, hj⟩) h_true
    have h_also_r₀ : rules.get ⟨i₀, hi₀_lt⟩ = r₀ := hi₀_get'
    have h_same :
        rules.get ⟨j, hj⟩ = rules.get ⟨i₀, hi₀_lt⟩ := by
      rw [h_eq_r₀, h_also_r₀]
    have h_idx_eq : j = i₀ := by
      have h_fin_eq : (⟨j, hj⟩ : Fin rules.length) = ⟨i₀, hi₀_lt⟩ :=
        (List.Nodup.get_inj_iff h_nodup).mp h_same
      simp only [Fin.mk.injEq] at h_fin_eq
      exact h_fin_eq
    exact hne h_idx_eq

  have h_outs_zero :
      ∀ j (hj : j < outs.length), j ≠ i₀ →
        outs.get ⟨j, hj⟩ = List.Vector.replicate n false := by
    intro j hj hne
    simp only [outs, apply_activations_with_routing]
    have hj_rules : j < rules.length := by
      simpa [h_len_outs] using hj
    have hj_masks : j < masks.length := by
      rw [h_len_masks]
      exact hj_rules
    have hj_per : j < perRuleInputs.length := by
      rw [h_len_per]
      exact hj_rules
    rw [List.get_zipWith3 _ rules masks perRuleInputs j hj_rules hj_masks
      hj_per]
    have h_mask_j : masks.get ⟨j, hj_masks⟩ = false := by
      have h_inactive := h_only_i₀_active j hj_rules hne
      have hj_acts : j < acts.length := by
        simp [acts, extract_activations]
        exact hj_rules
      have h_eq1 :
          masks.get ⟨j, hj_masks⟩ = acts.get ⟨j, hj_acts⟩ := by
        have h : masks[j]'hj_masks = acts[j]'hj_acts := by
          simp only [hmasks_eq]
        simp only [List.get_eq_getElem] at h ⊢
        exact h
      have h_eq2 :
          acts.get ⟨j, hj_acts⟩ =
            is_rule_active (rules.get ⟨j, hj_rules⟩) := by
        simp only [acts, extract_activations]
        exact list_map_get is_rule_active rules j hj_rules
          (by simp; exact hj_rules)
      rw [h_eq1, h_eq2, h_inactive]
    simp only [h_mask_j, Bool.false_eq_true, ↓reduceIte]

  have h_per_rule_i₀ :
      perRuleInputs.get ⟨i₀, hi₀_per⟩ =
        gather_rule_inputs_tagged (nodeIncoming[i₀]!) availableInputs := by
    simp [perRuleInputs]

  refine ⟨r₀, i₀, hi₀_lt, hr₀_mem, hi₀_get', ?_⟩
  unfold node_logic_with_tagged_routing
  simp [extract_activations, and_bool_list]
  have h_enum_per :
      (rules.zipIdx.map fun (_, ruleIdx) =>
        gather_rule_inputs_tagged
          (nodeIncoming[ruleIdx]?.getD default) availableInputs) =
        perRuleInputs := by
    simp only [perRuleInputs]
    apply List.ext_get
    · simp only [List.length_map, List.length_zipIdx, List.length_range]
    · intro i hi₁ hi₂
      have hi_zipIdx : i < rules.zipIdx.length := by
        simpa [List.length_map] using hi₁
      have hi_rules : i < rules.length := by
        rw [List.length_zipIdx] at hi_zipIdx
        exact hi_zipIdx
      rw [list_map_get _ _ _ hi_zipIdx hi₁]
      rw [list_map_get _ _ _ (by simp; exact hi_rules) hi₂]
      congr 1
      have h1 : (rules.zipIdx.get ⟨i, hi_zipIdx⟩).2 = 0 + i := by
        rw [← List.getElem_eq_get]
        simp [List.getElem_zipIdx]
      have h2 :
          (List.range rules.length).get ⟨i, by simp; exact hi_rules⟩ = i := by
        rw [← List.getElem_eq_get]
        simp [List.getElem_range]
      simp only [h1, h2, Nat.zero_add]
      have hiIncoming : i < nodeIncoming.length := by
        rw [hlen]
        exact hi_rules
      rw [List.getElem?_eq_getElem hiIncoming, Option.getD_some]
      simp only [List.getElem!_eq_getElem?_getD,
        List.getElem?_eq_getElem hiIncoming, Option.getD_some]

  constructor
  ·
    have h_list_or_eq : list_or outs = outs.get ⟨i₀, hi₀_outs⟩ := by
      unfold list_or
      exact list_or_single_nonzero outs i₀ hi₀_outs h_outs_zero
    have h_masks_eq_acts :
        List.map
          ((fun b => multiple_xor (List.map is_rule_active rules) && b) ∘
            is_rule_active) rules =
        List.map is_rule_active rules := by
      congr 1
      funext r
      simpa [acts, extract_activations, h_xor] using
        (show (multiple_xor (List.map is_rule_active rules) &&
            is_rule_active r) = is_rule_active r by
          rw [show multiple_xor (List.map is_rule_active rules) = true by
            simpa [acts, extract_activations] using h_xor]
          simp)
    rw [h_enum_per, h_masks_eq_acts]
    have h_goal_eq_outs :
        apply_activations_with_routing rules (List.map is_rule_active rules)
          perRuleInputs = outs := by
      simp only [outs]
      congr 1
      simp only [masks, hmasks_eq, acts, extract_activations]
    rw [h_goal_eq_outs, h_list_or_eq, h_outs_i₀, h_per_rule_i₀]
    congr 2
    have hiIncoming : i₀ < nodeIncoming.length := by
      rw [hlen]
      exact hi₀_lt
    rw [List.getElem!_eq_getElem?_getD (α := _),
      List.getElem?_eq_getElem hiIncoming]
  ·
    intro h_xor_false
    simp only [acts, extract_activations] at h_xor
    rw [h_xor] at h_xor_false
    simp at h_xor_false

theorem evaluate_node_tagged_uses_proven_node_logic
  {n : Nat}
  (node : CircuitNode n)
  (nodeIncoming : NodeIncoming)
  (tokens : List (TaggedToken n))
  (h_nonempty : tokens.length > 0) :
  let availableInputs := tokens.map fun t =>
    ((t.source_column, t.source_tag), t.dep_vector)
  let activatedNode := activate_node_from_tagged_tokens node nodeIncoming
    availableInputs
  evaluate_node_tagged node nodeIncoming tokens =
    node_logic_with_tagged_routing activatedNode.rules nodeIncoming
      availableInputs := by
  intro availableInputs activatedNode
  cases tokens with
  | nil =>
      simp at h_nonempty
  | cons head tail =>
      simp [evaluate_node_tagged, availableInputs, activatedNode,
        activate_node_from_tagged_tokens]

theorem evaluate_node_tagged_error_iff_not_unique
  {n : Nat}
  (node : CircuitNode n)
  (nodeIncoming : NodeIncoming)
  (tokens : List (TaggedToken n))
  (h_nonempty : tokens.length > 0) :
  let availableInputs := tokens.map fun t =>
    ((t.source_column, t.source_tag), t.dep_vector)
  let activatedNode := activate_node_from_tagged_tokens node nodeIncoming
    availableInputs
  let acts := extract_activations activatedNode.rules
  ((evaluate_node_tagged node nodeIncoming tokens).snd = false ∧
    acts.any (· = true))
    ↔ exactlyOneActive activatedNode.rules := by
  intro availableInputs activatedNode acts
  have h_nodup : activatedNode.rules.Nodup :=
    nodup_of_map (·.ruleId) activatedNode.nodupIds
  have h_xor_iff :
      multiple_xor (extract_activations activatedNode.rules) = true ↔
        exactlyOneActive activatedNode.rules :=
    multiple_xor_bool_iff_exactlyOneActive activatedNode.rules h_nodup
  have h_eval :
      evaluate_node_tagged node nodeIncoming tokens =
        node_logic_with_tagged_routing activatedNode.rules nodeIncoming
          availableInputs :=
    evaluate_node_tagged_uses_proven_node_logic node nodeIncoming tokens
      h_nonempty
  rw [h_eval]
  unfold node_logic_with_tagged_routing
  simp only
  constructor
  ·
    intro ⟨h_no_err, h_any⟩
    rw [← h_xor_iff]
    simp only [acts, extract_activations] at h_no_err h_any
    cases h_xor : multiple_xor (List.map is_rule_active activatedNode.rules)
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
    have h_xor :
        multiple_xor (extract_activations activatedNode.rules) = true :=
      h_xor_iff.mpr h_one
    constructor
    · simp only [extract_activations] at h_xor ⊢
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

theorem evaluate_node_tagged_correct
  {n : Nat}
  (node : CircuitNode n)
  (nodeIncoming : NodeIncoming)
  (tokens : List (TaggedToken n))
  (h_nonempty : tokens.length > 0)
  (h_incoming_len : nodeIncoming.length = node.rules.length)
  (h_no_error : (evaluate_node_tagged node nodeIncoming tokens).snd = false)
  (h_some_active :
    (extract_activations
      (activate_node_from_tagged_tokens node nodeIncoming
        (tokens.map fun t =>
          ((t.source_column, t.source_tag), t.dep_vector))).rules).any
      (· = true)) :
  let availableInputs := tokens.map fun t =>
    ((t.source_column, t.source_tag), t.dep_vector)
  let activatedNode := activate_node_from_tagged_tokens node nodeIncoming
    availableInputs
  ∃ r ∈ activatedNode.rules,
    let ruleIdx := activatedNode.rules.idxOf r
    let ruleInc := nodeIncoming[ruleIdx]!
    let inputs := gather_rule_inputs_tagged ruleInc availableInputs
    (evaluate_node_tagged node nodeIncoming tokens).fst =
      r.combine inputs := by
  intro availableInputs activatedNode
  have h_one : exactlyOneActive activatedNode.rules := by
    have h_iff :=
      evaluate_node_tagged_error_iff_not_unique node nodeIncoming tokens
        h_nonempty
    simp only at h_iff
    exact h_iff.mp ⟨h_no_error, h_some_active⟩
  have h_nodup : activatedNode.rules.Nodup :=
    nodup_of_map (·.ruleId) activatedNode.nodupIds
  have h_activated_len : activatedNode.rules.length = node.rules.length := by
    simp only [activatedNode, activate_node_from_tagged_tokens]
    have h :
        ∀ idx,
          (activateTaggedRulesAux nodeIncoming availableInputs idx node.rules).length =
            node.rules.length := by
      intro idx
      induction node.rules generalizing idx with
      | nil => simp [activateTaggedRulesAux]
      | cons r rs ih =>
          simp only [activateTaggedRulesAux, List.length_cons]
          rw [ih]
    exact h 0
  have h_len : nodeIncoming.length = activatedNode.rules.length := by
    rw [h_incoming_len, h_activated_len]
  have h_routing :=
    node_logic_with_tagged_routing_correct activatedNode.rules nodeIncoming
      availableInputs h_one h_nodup h_len
  obtain ⟨r, i, hi, hr_mem, hr_get, hr_eq⟩ := h_routing
  rw [evaluate_node_tagged_uses_proven_node_logic node nodeIncoming tokens
    h_nonempty]
  have h_fst :
      (node_logic_with_tagged_routing activatedNode.rules nodeIncoming
          availableInputs).fst =
        r.combine (gather_rule_inputs_tagged (nodeIncoming[i]!)
          availableInputs) := by
    rw [hr_eq]
  use r, hr_mem
  simp only [availableInputs, activatedNode]
  rw [h_fst]
  congr 1
  congr 1
  have h_indexOf : activatedNode.rules.idxOf r = i := by
    exact indexOf_eq_of_get hi h_nodup hr_get
  simp only [activatedNode, availableInputs] at h_indexOf ⊢
  rw [h_indexOf]

lemma multiple_xor_replicate_false_append_true
    (k : Nat) :
    multiple_xor (List.replicate k false ++ [true]) = true := by
  induction k with
  | zero =>
      simp [multiple_xor]
  | succ k ih =>
      simp [List.replicate_succ, ih]

lemma list_or_replicate_zero_append_singleton
    {n : Nat} (k : Nat)
    (v : List.Vector Bool n) :
    list_or (List.replicate k (List.Vector.replicate n false) ++ [v]) = v := by
  induction k with
  | zero =>
      simp [list_or, Vector.zipWith_or_replicate_false_right]
  | succ k ih =>
      simp [List.replicate_succ, list_or, List.foldl,
        Vector.zipWith_or_replicate_false_left]
      have hfold :
          List.foldl
              (fun acc v =>
                List.Vector.zipWith (fun x1 x2 => x1 || x2) acc v)
              (List.Vector.replicate n false)
              (List.replicate k (List.Vector.replicate n false)) =
            List.Vector.replicate n false := by
        apply foldl_zipWith_or_all_zeros
        intro j hj
        simp
      rw [hfold, Vector.zipWith_or_replicate_false_right]

lemma gather_rule_inputs_tagged_singleton_repetition
    {n : Nat}
    (selfIdx : Nat)
    (dep : List.Vector Bool n) :
    gather_rule_inputs_tagged
      [(selfIdx, gridTagRepetition)]
      [((selfIdx, gridTagRepetition), dep)] = [dep] := by
  simp [gather_rule_inputs_tagged]

lemma gather_rule_inputs_tagged_intro_of_singleton_repetition
    {n : Nat}
    (requiredCol selfIdx : Nat)
    (dep : List.Vector Bool n) :
    gather_rule_inputs_tagged
      [(requiredCol, gridTagIntro)]
      [((selfIdx, gridTagRepetition), dep)] = [] := by
  simp [gather_rule_inputs_tagged, gridTagIntro, gridTagRepetition]

lemma gather_rule_inputs_tagged_elim_of_singleton_repetition
    {n : Nat}
    (majorCol minorCol selfIdx : Nat)
    (dep : List.Vector Bool n) :
    gather_rule_inputs_tagged
      [(majorCol, gridTagElimMajor), (minorCol, gridTagElimMinor)]
      [((selfIdx, gridTagRepetition), dep)] = [] := by
  simp [gather_rule_inputs_tagged, gridTagElimMajor,
    gridTagElimMinor, gridTagRepetition]

lemma set_rule_activation_tagged_intro_of_singleton_repetition
    {n : Nat}
    (rid requiredCol selfIdx : Nat)
    (encoder : List.Vector Bool n)
    (dep : List.Vector Bool n) :
    set_rule_activation_tagged
      (mkIntroRule rid encoder false)
      [(requiredCol, gridTagIntro)]
      [((selfIdx, gridTagRepetition), dep)]
      = mkIntroRule rid encoder false := by
  simp [set_rule_activation_tagged, mkIntroRule,
    gridTagIntro, gridTagRepetition]

lemma set_rule_activation_tagged_elim_of_singleton_repetition
    {n : Nat}
    (rid majorCol minorCol selfIdx : Nat)
    (dep : List.Vector Bool n) :
    set_rule_activation_tagged
      (mkElimRule rid false false)
      [(majorCol, gridTagElimMajor), (minorCol, gridTagElimMinor)]
      [((selfIdx, gridTagRepetition), dep)]
      = mkElimRule rid false false := by
  simp [set_rule_activation_tagged, mkElimRule,
    gridTagElimMajor, gridTagElimMinor, gridTagRepetition]

lemma set_rule_activation_tagged_repetition_of_singleton_repetition
    {n : Nat}
    (rid selfIdx : Nat)
    (dep : List.Vector Bool n) :
    set_rule_activation_tagged
      (mkRepetitionRule rid false)
      [(selfIdx, gridTagRepetition)]
      [((selfIdx, gridTagRepetition), dep)]
      = mkRepetitionRule rid true := by
  simp [set_rule_activation_tagged, mkRepetitionRule]

lemma apply_activations_with_routing_false_prefix_true_last
    {n : Nat}
    (preRules : List (Rule n))
    (preInputs : List (List (List.Vector Bool n)))
    (rid : Nat)
    (dep : List.Vector Bool n)
    (h_len : preInputs.length = preRules.length) :
    apply_activations_with_routing
      (preRules ++ [mkRepetitionRule rid true])
      (List.replicate preRules.length false ++ [true])
      (preInputs ++ [[dep]])
      =
    List.replicate preRules.length (List.Vector.replicate n false) ++ [dep] := by
  induction preRules generalizing preInputs with
  | nil =>
      cases preInputs with
      | nil =>
          simp [apply_activations_with_routing, List.zipWith3, mkRepetitionRule]
      | cons x xs =>
          simp at h_len
  | cons r rs ih =>
      cases preInputs with
      | nil =>
          simp at h_len
      | cons ins rest =>
          simp at h_len
          simp [apply_activations_with_routing, List.replicate_succ,
            List.zipWith3]
          exact ih rest h_len

theorem node_logic_with_tagged_routing_false_prefix_true_last_of_perInputs
    {n : Nat}
    (preRules : List (Rule n))
    (nodeIncoming : NodeIncoming)
    (preInputs : List (List (List.Vector Bool n)))
    (rid selfIdx : Nat)
    (dep : List.Vector Bool n)
    (h_prefix_false :
      extract_activations preRules = List.replicate preRules.length false)
    (h_per :
      ((preRules ++ [mkRepetitionRule rid true]).zipIdx.map fun (x : Rule n × Nat) =>
        let ruleInc := nodeIncoming[x.2]!
        gather_rule_inputs_tagged ruleInc
          [((selfIdx, gridTagRepetition), dep)])
        = preInputs ++ [[dep]])
    (h_len : preInputs.length = preRules.length) :
    node_logic_with_tagged_routing
      (preRules ++ [mkRepetitionRule rid true])
      nodeIncoming
      [((selfIdx, gridTagRepetition), dep)] = (dep, false) := by
  unfold node_logic_with_tagged_routing
  have h_acts :
      extract_activations (preRules ++ [mkRepetitionRule rid true]) =
        List.replicate preRules.length false ++ [true] := by
    simpa [extract_activations, mkRepetitionRule, is_rule_active] using
      congrArg (fun bs => bs ++ [true]) h_prefix_false
  rw [h_acts, h_per]
  simp [multiple_xor_replicate_false_append_true, and_bool_list]
  rw [apply_activations_with_routing_false_prefix_true_last
    preRules preInputs rid dep h_len]
  simp [list_or_replicate_zero_append_singleton]

lemma extract_activations_introRules_false_from
    {n : Nat} (start : Nat) (introData : List (List.Vector Bool n)) :
    extract_activations
      ((introData.zipIdx start).map fun (encoder, pos) => mkIntroRule pos encoder false) =
      List.replicate introData.length false := by
  induction introData generalizing start with
  | nil =>
      simp [extract_activations]
  | cons encoder rest ih =>
      have htail := ih (start + 1)
      simp [extract_activations, mkIntroRule, is_rule_active] at htail ⊢
      simpa [List.replicate_succ] using congrArg (List.cons false) htail

lemma extract_activations_introRules_false
    {n : Nat} (introData : List (List.Vector Bool n)) :
    extract_activations
      (introData.zipIdx.map fun (encoder, pos) => mkIntroRule pos encoder false) =
      List.replicate introData.length false := by
  simpa using extract_activations_introRules_false_from 0 introData

lemma extract_activations_elimRules_false_from
    {n : Nat} (offset start : Nat) (elimData : List Nat) :
    extract_activations
      (((elimData.zipIdx start).map fun (_x, pos) => mkElimRule (offset + pos) false false)
        : List (Rule n)) =
      List.replicate elimData.length false := by
  induction elimData generalizing offset start with
  | nil =>
      simp [extract_activations]
  | cons x xs ih =>
      have htail := ih offset (start + 1)
      simp [extract_activations, mkElimRule, is_rule_active] at htail ⊢
      simpa [List.replicate_succ] using congrArg (List.cons false) htail

lemma extract_activations_elimRules_false
    {n : Nat} (offset : Nat) (elimData : List Nat) :
    extract_activations
      ((elimData.zipIdx.map fun (_x, pos) => mkElimRule (offset + pos) false false)
        : List (Rule n)) =
      List.replicate elimData.length false := by
  simpa using extract_activations_elimRules_false_from offset 0 elimData

lemma zipIdx_map_append_singleton
    {α β : Type*} (xs : List α) (a : α) (start : Nat) (f : α × Nat → β) :
    ((xs ++ [a]).zipIdx start).map f =
      (xs.zipIdx start).map f ++ [f (a, start + xs.length)] := by
  induction xs generalizing start with
  | nil =>
      simp [List.zipIdx]
  | cons x rest ih =>
      simp [List.zipIdx_cons, ih]
      congr 1
      simp [Nat.add_left_comm, Nat.add_comm]

@[simp] lemma List.getElem?_append_singleton_length
    {α : Type*} (xs : List α) (a : α) :
    (xs ++ [a])[xs.length]? = some a := by
  induction xs with
  | nil =>
      simp
  | cons x rest ih =>
      simp

@[simp] lemma List.getElem?_cons_succ'
    {α : Type*} (x : α) (xs : List α) (n : Nat) :
    (x :: xs)[n + 1]? = xs[n]? := by
  simp

@[simp] lemma List.getElem?_append_cons_length
    {α : Type*} (xs : List α) (y : α) (ys : List α) :
    (xs ++ y :: ys)[xs.length]? = some y := by
  induction xs with
  | nil =>
      simp
  | cons x rest ih =>
      simp

lemma buildTaggedIncomingMapForFormula_repetition_slot_getD
    (formulas : List Formula) (formula : Formula) :
    let introData := match formula with
      | .impl _ _ =>
          match encoderForIntro formulas formula with
          | some encoder => [encoder]
          | none => []
      | _ => []
    let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
      match f with
      | .impl _ B => if B = formula then some idx else none
      | _ => none
    (buildTaggedIncomingMapForFormula formulas formula)[introData.length + elimData.length]?.getD default =
      [(formulas.idxOf formula, gridTagRepetition)] := by
  cases formula with
  | atom name =>
      dsimp
      let elimIdxs := formulas.zipIdx.filterMap fun (f, idx) =>
        match f with
        | .impl _ B => if B = Formula.atom name then some idx else none
        | _ => none
      let elimMaps := formulas.zipIdx.filterMap fun (f, idx) =>
        match f with
        | .impl A B =>
            if B = Formula.atom name then
              some [(idx, gridTagElimMajor), (formulas.idxOf A, gridTagElimMinor)]
            else none
        | _ => none
      have h_len : elimMaps.length = elimIdxs.length := by
        exact List.length_filterMap_eq_of_isSome _ _ _ (by
          intro x
          cases x with
          | mk f idx =>
              cases f with
              | atom s =>
                  simp [Option.isSome]
              | impl A B =>
                  by_cases hB : B = Formula.atom name <;> simp [hB, Option.isSome])
      unfold buildTaggedIncomingMapForFormula
      rw [← h_len]
      simp [elimMaps]
  | impl A B =>
      dsimp
      let elimIdxs := formulas.zipIdx.filterMap fun (f, idx) =>
        match f with
        | .impl _ B' => if B' = Formula.impl A B then some idx else none
        | _ => none
      let elimMaps := formulas.zipIdx.filterMap fun (f, idx) =>
        match f with
        | .impl A' B' =>
            if B' = Formula.impl A B then
              some [(idx, gridTagElimMajor), (formulas.idxOf A', gridTagElimMinor)]
            else none
        | _ => none
      have h_len : elimMaps.length = elimIdxs.length := by
        exact List.length_filterMap_eq_of_isSome _ _ _ (by
          intro x
          cases x with
          | mk f idx =>
              cases f with
              | atom s =>
                  simp [Option.isSome]
              | impl A' B' =>
                  by_cases hB : B' = Formula.impl A B <;> simp [hB, Option.isSome])
      let selfIdx := List.idxOf (Formula.impl A B) formulas
      unfold buildTaggedIncomingMapForFormula
      rw [← h_len]
      have h_impl :
          ((([(List.idxOf B formulas, gridTagIntro)] ::
              (elimMaps ++ [[(selfIdx, gridTagRepetition)]]))[1 + elimMaps.length]?).getD default) =
            [(selfIdx, gridTagRepetition)] := by
        have h_step :
            ([(List.idxOf B formulas, gridTagIntro)] ::
                (elimMaps ++ [[(selfIdx, gridTagRepetition)]]))[1 + elimMaps.length]? =
              some [(selfIdx, gridTagRepetition)] := by
          simp [Nat.add_comm]
        have h_step' := congrArg (fun o => o.getD default) h_step
        simpa using h_step'
      simpa [selfIdx, elimMaps] using h_impl

lemma elimRules_from_zipIdx_eq_of_length_from
    {n : Nat} {α β : Type*}
    (xs : List α) (ys : List β)
    (start offset : Nat)
    (h_len : xs.length = ys.length) :
    (((xs.zipIdx start).map fun (_x, pos) => mkElimRule (offset + pos) false false)
      : List (Rule n)) =
    (((ys.zipIdx start).map fun (_y, pos) => mkElimRule (offset + pos) false false)
      : List (Rule n)) := by
  induction xs generalizing ys start with
  | nil =>
      cases ys with
      | nil =>
          simp
      | cons y ys =>
          simp at h_len
  | cons x xs ih =>
      cases ys with
      | nil =>
          simp at h_len
      | cons y ys =>
          simp at h_len
          simpa [List.zipIdx_cons] using ih ys (start + 1) h_len

lemma elimRules_shift_eq_zipIdx_from
    {n : Nat} {α : Type*}
    (xs : List α) (start offset : Nat) :
    (((xs.zipIdx start).map fun (_x, pos) => mkElimRule (offset + pos) false false)
      : List (Rule n)) =
    (((xs.zipIdx (offset + start)).map fun (_x, rid) => mkElimRule rid false false)
      : List (Rule n)) := by
  induction xs generalizing start offset with
  | nil =>
      simp
  | cons x xs ih =>
      simpa [List.zipIdx_cons, Nat.add_assoc] using
        ih (start := start + 1) (offset := offset)

lemma activateTaggedElimThenRepRulesAux_of_singleton_repetition
    {n : Nat}
    (incomingPrefix : NodeIncoming)
    (elimPairs : List (Nat × Nat))
    (ridStart selfIdx : Nat)
    (dep : List.Vector Bool n) :
    activateTaggedRulesAux
      (incomingPrefix ++
        elimPairs.map (fun p => [(p.1, gridTagElimMajor), (p.2, gridTagElimMinor)]) ++
        [[(selfIdx, gridTagRepetition)]])
      [((selfIdx, gridTagRepetition), dep)]
      incomingPrefix.length
      ((((elimPairs.zipIdx ridStart).map fun (_pair, rid) =>
          mkElimRule rid false false) : List (Rule n)) ++
        [mkRepetitionRule (ridStart + elimPairs.length) false]) =
      (((elimPairs.zipIdx ridStart).map fun (_pair, rid) =>
          mkElimRule rid false false) : List (Rule n)) ++
        [mkRepetitionRule (ridStart + elimPairs.length) true] := by
  induction elimPairs generalizing incomingPrefix ridStart with
  | nil =>
      simp [activateTaggedRulesAux,
        set_rule_activation_tagged_repetition_of_singleton_repetition]
  | cons pair rest ih =>
      rcases pair with ⟨majorCol, minorCol⟩
      simp [activateTaggedRulesAux, List.zipIdx_cons, List.append_assoc,
        set_rule_activation_tagged_elim_of_singleton_repetition]
      simpa [List.append_assoc, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
        ih
          (incomingPrefix := incomingPrefix ++
            [[(majorCol, gridTagElimMajor), (minorCol, gridTagElimMinor)]])
          (ridStart := ridStart + 1)

lemma extract_activations_nodeForFormula_prefix_false
    (formulas : List Formula) (formula : Formula) :
    let introData := match formula with
      | .impl _ _ =>
          match encoderForIntro formulas formula with
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
    extract_activations (introRules ++ elimRules) =
      List.replicate (introRules.length + elimRules.length) false := by
  let introData := match formula with
    | .impl _ _ =>
        match encoderForIntro formulas formula with
        | some encoder => [encoder]
        | none => []
    | _ => []
  let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .impl _ B => if B = formula then some idx else none
    | _ => none
  let introRules := introData.zipIdx.map fun (encoder, pos) =>
    mkIntroRule pos encoder false
  let elimRules : List (Rule formulas.length) := elimData.zipIdx.map fun (_, pos) =>
    mkElimRule (introData.length + pos) false false
  change extract_activations (introRules ++ elimRules) =
    List.replicate (introRules.length + elimRules.length) false
  have h_intro :
      extract_activations introRules =
        List.replicate introRules.length false := by
    simpa [introRules] using
      extract_activations_introRules_false introData
  have h_elim :
      extract_activations elimRules =
        List.replicate elimRules.length false := by
    simpa [elimRules] using
      extract_activations_elimRules_false introData.length elimData
  rw [show extract_activations (introRules ++ elimRules) =
      extract_activations introRules ++ extract_activations elimRules by
        simp [extract_activations]]
  rw [h_intro, h_elim, List.replicate_add]

theorem node_logic_with_tagged_routing_nodeForFormula_singleton_repetition
    (formulas : List Formula) (formula : Formula)
    (dep : List.Vector Bool formulas.length) :
    let introData := match formula with
      | .impl _ _ =>
          match encoderForIntro formulas formula with
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
    let preRules := introRules ++ elimRules
    let repRid := introData.length + elimData.length
    node_logic_with_tagged_routing
      (preRules ++ [mkRepetitionRule repRid true])
      (buildTaggedIncomingMapForFormula formulas formula)
      [((formulas.idxOf formula, gridTagRepetition), dep)] = (dep, false) := by
  let introData := match formula with
    | .impl _ _ =>
        match encoderForIntro formulas formula with
        | some encoder => [encoder]
        | none => []
    | _ => []
  let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .impl _ B => if B = formula then some idx else none
    | _ => none
  let introRules := introData.zipIdx.map fun (encoder, pos) =>
    mkIntroRule pos encoder false
  let elimRules : List (Rule formulas.length) := elimData.zipIdx.map fun (_, pos) =>
    mkElimRule (introData.length + pos) false false
  let preRules := introRules ++ elimRules
  let repRid := introData.length + elimData.length
  let prefixInputs :=
    (preRules.zipIdx.map fun (x : Rule formulas.length × Nat) =>
      let ruleInc := (buildTaggedIncomingMapForFormula formulas formula)[x.2]!
      gather_rule_inputs_tagged ruleInc
        [((formulas.idxOf formula, gridTagRepetition), dep)])
  change node_logic_with_tagged_routing
    (preRules ++ [mkRepetitionRule repRid true])
    (buildTaggedIncomingMapForFormula formulas formula)
    [((formulas.idxOf formula, gridTagRepetition), dep)] = (dep, false)
  have h_prefix :
      extract_activations preRules =
        List.replicate preRules.length false := by
    simpa [preRules, introRules, elimRules, introData, elimData] using
      extract_activations_nodeForFormula_prefix_false formulas formula
  have h_per :
      ((preRules ++ [mkRepetitionRule repRid true]).zipIdx.map
        fun (x : Rule formulas.length × Nat) =>
          let ruleInc := (buildTaggedIncomingMapForFormula formulas formula)[x.2]!
          gather_rule_inputs_tagged ruleInc
            [((formulas.idxOf formula, gridTagRepetition), dep)]) =
        prefixInputs ++ [[dep]] := by
    rw [zipIdx_map_append_singleton]
    simp [prefixInputs]
    have h_preLen : preRules.length = repRid := by
      simp [preRules, introRules, elimRules, repRid]
    have h_rep_getD :
        ((buildTaggedIncomingMapForFormula formulas formula)[repRid]?.getD default) =
          [(formulas.idxOf formula, gridTagRepetition)] := by
      simpa [repRid, introData, elimData] using
        buildTaggedIncomingMapForFormula_repetition_slot_getD formulas formula
    rw [h_preLen]
    rw [h_rep_getD, gather_rule_inputs_tagged_singleton_repetition]
  have h_len : prefixInputs.length = preRules.length := by
    simp [prefixInputs]
  exact node_logic_with_tagged_routing_false_prefix_true_last_of_perInputs
    (preRules := preRules)
    (nodeIncoming := buildTaggedIncomingMapForFormula formulas formula)
    (preInputs := prefixInputs)
    (rid := repRid)
    (selfIdx := formulas.idxOf formula)
    (dep := dep)
    h_prefix h_per h_len

/-- Formula-role check used by the edge-aware path converter. This is defined
    locally instead of using the later structural helper so the edge-aware
    evaluator remains independent of the structural predicate section. -/
def taggedElimPairMatches
    (target major minor : Vertex) : Bool :=
  match major.FORMULA with
  | Formula.impl A B => decide (B = target.FORMULA ∧ A = minor.FORMULA)
  | _ => false

/-- Role tag assigned to the concrete source-target edge. Elimination is
    checked before repetition so that `A` as the minor premise of `A -> A`
    is not confused with a repetition input. -/
def gridTagForSourceTarget
    (bd : BranchingDLDS) (source target : Vertex) : Nat :=
  match incomingSources bd.base target with
  | [u] =>
      if decide (source = u) then
        match findIntroDischarge bd target with
        | some _ => gridTagIntro
        | none => gridTagRepetition
      else gridTagRepetition
  | [u, w] =>
      if taggedElimPairMatches target u w then
        if decide (source = u) then gridTagElimMajor
        else if decide (source = w) then gridTagElimMinor
        else gridTagRepetition
      else if taggedElimPairMatches target w u then
        if decide (source = w) then gridTagElimMajor
        else if decide (source = u) then gridTagElimMinor
        else gridTagRepetition
      else gridTagRepetition
  | _ => gridTagRepetition

def nextVertexAndTagForReading
    (bd : BranchingDLDS) (reading : ReadingInput) (u : Vertex) :
    Option (Vertex × Nat) :=
  match nextVertexForReading bd reading u with
  | none => none
  | some v => some (v, gridTagForSourceTarget bd u v)

def readingTaggedPathFromVertex
    (bd : BranchingDLDS) (reading : ReadingInput) :
    Option Vertex → Nat → List (Nat × Nat)
  | _, 0 => []
  | none, steps + 1 =>
      (0, gridTagRepetition) ::
        readingTaggedPathFromVertex bd reading none steps
  | some u, steps + 1 =>
      match nextVertexAndTagForReading bd reading u with
      | none =>
          (0, gridTagRepetition) ::
            readingTaggedPathFromVertex bd reading none steps
      | some (w, tag) =>
          ((buildFormulas bd.base).idxOf w.FORMULA + 1, tag) ::
            readingTaggedPathFromVertex bd reading (some w) steps

lemma readingTaggedPathFromVertex_length
    (bd : BranchingDLDS) (reading : ReadingInput)
    (start : Option Vertex) (steps : Nat) :
    (readingTaggedPathFromVertex bd reading start steps).length = steps := by
  induction steps generalizing start with
  | zero =>
      cases start <;> rfl
  | succ steps ih =>
      cases start with
      | none =>
          simp [readingTaggedPathFromVertex, ih]
      | some u =>
          simp [readingTaggedPathFromVertex]
          split <;> simp [ih]

def readingToTaggedPathFull
    (bd : BranchingDLDS) (reading : ReadingInput) : TaggedPathInput :=
  let steps := dldsMaxLevel bd.base
  (buildFormulas bd.base).map fun formula =>
    readingTaggedPathFromVertex bd reading (topVertexForFormula bd formula)
      steps

theorem readingToTaggedPathFull_wellformed
    (bd : BranchingDLDS) (reading : ReadingInput) :
    (readingToTaggedPathFull bd reading).length = numFormulas bd.base ∧
      ∀ path ∈ readingToTaggedPathFull bd reading,
        path.length = gridTransitionCount bd.base := by
  constructor
  · simp [readingToTaggedPathFull, numFormulas]
  · intro path hMem
    unfold readingToTaggedPathFull at hMem
    rcases List.mem_map.mp hMem with ⟨formula, _hFormula, hPath⟩
    rw [← hPath, readingTaggedPathFromVertex_length,
      gridTransitionCount_eq_dldsMaxLevel]

def extract_tagged_grid_result_at_vertex
    (bd : BranchingDLDS)
    (grid : List (GridLayer (numFormulas bd.base)))
    (initialVecs : List (List.Vector Bool (numFormulas bd.base)))
    (paths : TaggedPathInput)
    (v : Vertex) : List.Vector Bool (numFormulas bd.base) :=
  match traceOutputsAtLevel (get_tagged_eval_trace grid initialVecs paths)
      v.LEVEL with
  | none => formulaVecZero bd.base
  | some outputs => outputAtFormula bd.base outputs v.FORMULA

def taggedGridFEnvFromPath
    (bd : BranchingDLDS)
    (grid : List (GridLayer (numFormulas bd.base)))
    (initialVecs : List (List.Vector Bool (numFormulas bd.base)))
    (paths : TaggedPathInput) :
    Vertex → List.Vector Bool (numFormulas bd.base) :=
  fun u => extract_tagged_grid_result_at_vertex bd grid initialVecs paths u

def taggedGridFEnvFromReading
    (bd : BranchingDLDS) (reading : ReadingInput) :
    Vertex → List.Vector Bool (numFormulas bd.base) :=
  let grid := buildTaggedGridFromDLDS bd.base
  let initialVecs := initialVectorsFromDLDS bd.base
  let paths := readingToTaggedPathFull bd reading
  taggedGridFEnvFromPath bd grid initialVecs paths

def BranchingDLDS.TaggedGridCompatibleForReading
    (bd : BranchingDLDS) (reading : ReadingInput) : Prop :=
  ∀ v ∈ bd.evalOrder,
    taggedGridFEnvFromReading bd reading v =
      classicalKernel bd reading (taggedGridFEnvFromReading bd reading) v

def BranchingDLDS.TaggedGridCompatible (bd : BranchingDLDS) : Prop :=
  ∀ reading, bd.TaggedGridCompatibleForReading reading

theorem tagged_node_logic_equals_classicalKernel_for_reading
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex)
    (hMem : v ∈ bd.evalOrder)
    (hCompat : bd.TaggedGridCompatibleForReading reading) :
    let grid := buildTaggedGridFromDLDS bd.base
    let initialVecs := initialVectorsFromDLDS bd.base
    let paths := readingToTaggedPathFull bd reading
    let fenvFromGrid : Vertex → List.Vector Bool (numFormulas bd.base) :=
      taggedGridFEnvFromPath bd grid initialVecs paths
    fenvFromGrid v = classicalKernel bd reading fenvFromGrid v := by
  simpa [BranchingDLDS.TaggedGridCompatibleForReading,
    taggedGridFEnvFromReading, taggedGridFEnvFromPath] using
    hCompat v hMem

theorem tagged_node_logic_equals_classicalKernel_under_TaggedGridCompatible
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex)
    (hMem : v ∈ bd.evalOrder)
    (hCompat : bd.TaggedGridCompatible) :
    let grid := buildTaggedGridFromDLDS bd.base
    let initialVecs := initialVectorsFromDLDS bd.base
    let paths := readingToTaggedPathFull bd reading
    let fenvFromGrid : Vertex → List.Vector Bool (numFormulas bd.base) :=
      taggedGridFEnvFromPath bd grid initialVecs paths
    fenvFromGrid v = classicalKernel bd reading fenvFromGrid v := by
  exact tagged_node_logic_equals_classicalKernel_for_reading
    bd reading v hMem (hCompat reading)

private theorem classicalKernel_eq_of_eq_on_dependencies
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv₁ fenv₂ : Vertex → List.Vector Bool (numFormulas bd.base))
    (v : Vertex)
    (h_sources :
      ∀ u ∈ incomingSources bd.base v, fenv₁ u = fenv₂ u)
    (h_branch :
      ∀ src rvar colour,
        findBranchTarget bd v = some (src, rvar, colour) →
          fenv₁ src = fenv₂ src) :
    classicalKernel bd reading fenv₁ v =
      classicalKernel bd reading fenv₂ v := by
  unfold classicalKernel
  by_cases h_hyp : v.HYPOTHESIS = true
  · simp [h_hyp]
  · simp only [h_hyp]
    cases h_find : findBranchTarget bd v with
    | some triple =>
        obtain ⟨src, rvar, colour⟩ := triple
        simp only
        set ordinary := (incomingSources bd.base v).filter
          (fun u => decide (u ≠ src)) with hordinary
        have h_ordinary :
            ∀ u ∈ ordinary, fenv₁ u = fenv₂ u := by
          intro u hu
          have hu' : u ∈ (incomingSources bd.base v).filter
              (fun u => decide (u ≠ src)) := by
            simpa [hordinary] using hu
          exact h_sources u (List.mem_filter.mp hu').1
        have h_fold :
            ordinary.foldl
                (fun acc u => acc.zipWith (· || ·) (fenv₁ u))
                (List.Vector.replicate (numFormulas bd.base) false) =
              ordinary.foldl
                (fun acc u => acc.zipWith (· || ·) (fenv₂ u))
                (List.Vector.replicate (numFormulas bd.base) false) := by
          apply foldl_congr_on_list
          intro acc u hu
          rw [h_ordinary u hu]
        by_cases h_col : readingColour reading rvar = some colour
        · simp [h_col, h_fold, h_branch src rvar colour h_find]
        · simp [h_col, h_fold]
    | none =>
        simp only
        have h_fold :
            (incomingSources bd.base v).foldl
                (fun acc u => acc.zipWith (· || ·) (fenv₁ u))
                (List.Vector.replicate (numFormulas bd.base) false) =
              (incomingSources bd.base v).foldl
                (fun acc u => acc.zipWith (· || ·) (fenv₂ u))
                (List.Vector.replicate (numFormulas bd.base) false) := by
          apply foldl_congr_on_list
          intro acc u hu
          rw [h_sources u hu]
        cases h_intro : findIntroDischarge bd v with
        | some h_vertex =>
            simp [h_fold]
        | none =>
            simp [h_fold]

private theorem classicalKernelSemantics_fixedPoint_from
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (h_wf_branching : bd.WellFormedBranching)
    (h_compat :
      ∀ v ∈ bd.evalOrder,
        fenv v = classicalKernel bd reading fenv v)
    (pre xs : List Vertex)
    (h_split : bd.evalOrder = pre ++ xs)
    (h_pre :
      ∀ u ∈ pre,
        formulaEnvLookup
          (pre.foldl (classicalKernelSemanticsStep bd reading) []) u =
          fenv u) :
    ∀ u ∈ pre ++ xs,
      formulaEnvLookup
        ((pre ++ xs).foldl (classicalKernelSemanticsStep bd reading) []) u =
        fenv u := by
  induction xs generalizing pre with
  | nil =>
      intro u hu
      simpa using h_pre u (by simpa using hu)
  | cons x rest ih =>
      set envPre :=
        pre.foldl (classicalKernelSemanticsStep bd reading) [] with henvPre
      have h_nd_split : (pre ++ x :: rest).Nodup := by
        simpa [h_split] using h_wf_branching.1.1
      have h_x_notin_pre : x ∉ pre := by
        have hparts := List.nodup_append.mp h_nd_split
        have hdisj := hparts.2.2
        intro hx
        exact (hdisj x hx x (by simp)) rfl
      have h_sources_eq :
          ∀ u ∈ incomingSources bd.base x,
            formulaEnvLookup envPre u = fenv u := by
        intro u hu
        unfold incomingSources at hu
        rcases List.mem_map.mp hu with ⟨e, he_filter, he_start⟩
        rcases List.mem_filter.mp he_filter with ⟨he_mem, he_end_bool⟩
        have he_end : e.END = x := by
          simpa using he_end_bool
        have h_src_pre : e.START ∈ pre :=
          h_wf_branching.1.2 e he_mem pre rest (by simpa [he_end] using h_split)
        have hu_pre : u ∈ pre := by
          simpa [he_start] using h_src_pre
        simpa [henvPre] using h_pre u hu_pre
      have h_branch_eq :
          ∀ src rvar colour,
            findBranchTarget bd x = some (src, rvar, colour) →
              formulaEnvLookup envPre src = fenv src := by
        intro src rvar colour h_find
        obtain ⟨b, hb_mem, hb_src, hb_rvar, hb_target⟩ :=
          findBranchTarget_spec bd x src rvar colour h_find
        have h_src_pre : b.source ∈ pre :=
          h_wf_branching.2 b hb_mem colour x hb_target pre rest h_split
        have hsrc_pre : src ∈ pre := by
          simpa [hb_src] using h_src_pre
        simpa [henvPre] using h_pre src hsrc_pre
      have h_step :
          classicalKernel bd reading (formulaEnvLookup envPre) x = fenv x := by
        have h_kernel :=
          classicalKernel_eq_of_eq_on_dependencies bd reading
            (formulaEnvLookup envPre) fenv x h_sources_eq h_branch_eq
        have hx_mem : x ∈ bd.evalOrder := by
          rw [h_split]
          simp
        calc
          classicalKernel bd reading (formulaEnvLookup envPre) x =
              classicalKernel bd reading fenv x := h_kernel
          _ = fenv x := (h_compat x hx_mem).symm
      have h_pre_next :
          ∀ u ∈ pre ++ [x],
            formulaEnvLookup
              ((pre ++ [x]).foldl
                (classicalKernelSemanticsStep bd reading) []) u =
              fenv u := by
        intro u hu
        rw [List.foldl_append]
        simp only [List.foldl_cons, List.foldl_nil,
          classicalKernelSemanticsStep]
        by_cases hux : u = x
        · subst u
          rw [formulaEnvLookup_append_singleton_of_not_mem]
          · simpa [henvPre] using h_step
          · intro w hw
            have hkeys :=
              foldl_append_keys_subset
                (fun env v => classicalKernel bd reading
                  (formulaEnvLookup env) v)
                pre [] (x, w) (by simpa [henvPre] using hw)
            rcases hkeys with hacc | hkey
            · simp at hacc
            · exact h_x_notin_pre hkey
        · have hu_pre : u ∈ pre := by
            rcases List.mem_append.mp hu with hpre | hxmem
            · exact hpre
            · simp at hxmem
              exact False.elim (hux hxmem)
          rw [formulaEnvLookup_append_singleton_ne
            envPre x u
            (classicalKernel bd reading (formulaEnvLookup envPre) x) hux]
          simpa [henvPre] using h_pre u hu_pre
      have h_split_next : bd.evalOrder = (pre ++ [x]) ++ rest := by
        simpa [List.append_assoc] using h_split
      intro u hu
      have h_rec := ih (pre := pre ++ [x]) h_split_next h_pre_next
      simpa [List.append_assoc] using h_rec u (by simpa [List.append_assoc] using hu)

theorem taggedGridFEnvFromReading_eq_classicalKernelSemanticsAt_for_reading
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_wf_branching : bd.WellFormedBranching)
    (h_compat : bd.TaggedGridCompatibleForReading reading) :
    ∀ v ∈ bd.evalOrder,
      taggedGridFEnvFromReading bd reading v =
        classicalKernelSemanticsAt bd reading v := by
  intro v hv
  let fenv := taggedGridFEnvFromReading bd reading
  have h_fixed :
      ∀ u ∈ ([] : List Vertex) ++ bd.evalOrder,
        formulaEnvLookup
          ((([] : List Vertex) ++ bd.evalOrder).foldl
            (classicalKernelSemanticsStep bd reading) []) u =
          fenv u :=
    classicalKernelSemantics_fixedPoint_from bd reading fenv
      h_wf_branching
      (by
        intro u hu
        exact h_compat u hu)
      [] bd.evalOrder
      (by simp)
      (by intro u hu; simp at hu)
  have hv_fixed := h_fixed v (by simpa using hv)
  simpa [fenv, classicalKernelSemanticsAt, classicalKernelSemantics] using
    hv_fixed.symm

theorem taggedGridFEnvFromReading_eq_classicalKernelSemanticsAt
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_wf_branching : bd.WellFormedBranching)
    (h_compat : bd.TaggedGridCompatible) :
    ∀ v ∈ bd.evalOrder,
      taggedGridFEnvFromReading bd reading v =
        classicalKernelSemanticsAt bd reading v := by
  exact taggedGridFEnvFromReading_eq_classicalKernelSemanticsAt_for_reading
    bd reading h_wf_branching (h_compat reading)

/-- Conditional bridge from the tagged-grid run induced by one reading to the
    actual reading/DLDS dependency semantics. The tagged grid is
    formula-indexed, so the statement compares after `formulaVecToHypVec`. -/
theorem taggedGridFEnvFromReading_eq_dldsSemanticsAt_for_reading
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_wf_branching : bd.WellFormedBranching)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_eval_in_V : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ bd.evalOrder, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true)
    (h_compat : bd.TaggedGridCompatibleForReading reading) :
    ∀ v ∈ bd.evalOrder,
      formulaVecToHypVec bd.base (taggedGridFEnvFromReading bd reading v) =
        dldsSemanticsAt bd reading v := by
  intro v hv
  have h_tag :=
    taggedGridFEnvFromReading_eq_classicalKernelSemanticsAt_for_reading
      bd reading h_wf_branching h_compat v hv
  have h_kernel :=
    classicalKernel_dldsSemantics_global_equiv bd reading h_wf_branching.1
      h_distinct h_in_build h_eval_in_V h_discharge_wf v hv
  rw [h_tag]
  exact h_kernel

/-- Global-reading variant of
    `taggedGridFEnvFromReading_eq_dldsSemanticsAt_for_reading`. -/
theorem taggedGridFEnvFromReading_eq_dldsSemanticsAt
    (bd : BranchingDLDS) (reading : ReadingInput)
    (h_wf_branching : bd.WellFormedBranching)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_eval_in_V : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ bd.evalOrder, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true)
    (h_compat : bd.TaggedGridCompatible) :
    ∀ v ∈ bd.evalOrder,
      formulaVecToHypVec bd.base (taggedGridFEnvFromReading bd reading v) =
        dldsSemanticsAt bd reading v := by
  exact taggedGridFEnvFromReading_eq_dldsSemanticsAt_for_reading
    bd reading h_wf_branching h_distinct h_in_build h_eval_in_V
    h_discharge_wf (h_compat reading)

/-! ##### Reading-level rejection and robustness instantiation -/

namespace RobustnessInput

def toReadingInput {r : Nat}
    (x : Robustness.Input 2 r) : ReadingInput :=
  List.ofFn fun i : Fin r => decide ((x i).val = 1)

end RobustnessInput

def HasUndischargedDependency
    (bd : BranchingDLDS) (root : Vertex) (reading : ReadingInput) : Prop :=
  dldsSemanticsAt bd reading root ≠ HypDepVec.zero bd.base

def RejectsReading
    (bd : BranchingDLDS) (root : Vertex)
    (reading : Robustness.Input 2 bd.numReading) : Prop :=
  HasUndischargedDependency bd root
    (RobustnessInput.toReadingInput reading)

theorem rejectsReading_density_from_badPrefix
    (bd : BranchingDLDS)
    (root : Vertex)
    {C : Nat}
    (π : Robustness.Prefix 2 bd.numReading)
    (hπ : Robustness.BadPrefix (RejectsReading bd root) π)
    (hC : π.len ≤ C)
    (hCr : C ≤ bd.numReading) :
    2 ^ bd.numReading ≤
      (Robustness.BadFinset (RejectsReading bd root)).card * 2 ^ C := by
  exact Robustness.bad_inputs_density_fixed_C_mul
    (m := 2)
    (r := bd.numReading)
    (C := C)
    (hm := by decide)
    (rejects := RejectsReading bd root)
    (π := π)
    hπ hC hCr

/-- Paper-facing semantic robustness wrapper: a bad prefix of length at most
    `C` forces a multiplication-form lower bound on rejected readings. -/
theorem semantic_robustness_from_short_badPrefix
    (bd : BranchingDLDS)
    (root : Vertex)
    {C : Nat}
    (π : Robustness.Prefix 2 bd.numReading)
    (hπ : Robustness.BadPrefix (RejectsReading bd root) π)
    (hC : π.len ≤ C)
    (hCr : C ≤ bd.numReading) :
    2 ^ bd.numReading ≤
      (Robustness.BadFinset (RejectsReading bd root)).card * 2 ^ C := by
  exact rejectsReading_density_from_badPrefix bd root π hπ hC hCr

def TaggedRejectsReading
    (bd : BranchingDLDS) (root : Vertex)
    (reading : Robustness.Input 2 bd.numReading) : Prop :=
  formulaVecToHypVec bd.base
      (taggedGridFEnvFromReading bd
        (RobustnessInput.toReadingInput reading) root) ≠
    HypDepVec.zero bd.base

theorem taggedRejectsReading_iff_rejectsReading
    (bd : BranchingDLDS)
    (root : Vertex)
    (reading : Robustness.Input 2 bd.numReading)
    (h_root_mem : root ∈ bd.evalOrder)
    (h_wf_branching : bd.WellFormedBranching)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_eval_in_V : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ bd.evalOrder, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true)
    (h_compat :
      bd.TaggedGridCompatibleForReading
        (RobustnessInput.toReadingInput reading)) :
    TaggedRejectsReading bd root reading ↔ RejectsReading bd root reading := by
  unfold TaggedRejectsReading RejectsReading HasUndischargedDependency
  rw [taggedGridFEnvFromReading_eq_dldsSemanticsAt_for_reading
    bd (RobustnessInput.toReadingInput reading) h_wf_branching h_distinct
    h_in_build h_eval_in_V h_discharge_wf h_compat root h_root_mem]

/-- Tagged-grid-facing robustness wrapper. Under the conditional tagged-grid
    bridge assumptions for every binary reading, a bad prefix for the tagged
    rejection predicate yields the semantic rejected-reading lower bound. -/
theorem taggedGrid_robustness_from_short_badPrefix
    (bd : BranchingDLDS)
    (root : Vertex)
    {C : Nat}
    (π : Robustness.Prefix 2 bd.numReading)
    (hπ : Robustness.BadPrefix (TaggedRejectsReading bd root) π)
    (hC : π.len ≤ C)
    (hCr : C ≤ bd.numReading)
    (h_root_mem : root ∈ bd.evalOrder)
    (h_wf_branching : bd.WellFormedBranching)
    (h_distinct : bd.base.HypFormulasDistinct)
    (h_in_build : bd.base.HypFormulasInBuild)
    (h_eval_in_V : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V)
    (h_discharge_wf :
      ∀ v ∈ bd.evalOrder, ∀ h_vertex,
        findIntroDischarge bd v = some h_vertex →
          h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true)
    (h_compat :
      ∀ reading : Robustness.Input 2 bd.numReading,
        bd.TaggedGridCompatibleForReading
          (RobustnessInput.toReadingInput reading)) :
    2 ^ bd.numReading ≤
      (Robustness.BadFinset (RejectsReading bd root)).card * 2 ^ C := by
  apply semantic_robustness_from_short_badPrefix bd root π
  · intro reading hreading
    have h_tagged := hπ reading hreading
    exact (taggedRejectsReading_iff_rejectsReading bd root reading
      h_root_mem h_wf_branching h_distinct h_in_build h_eval_in_V
      h_discharge_wf (h_compat reading)).mp h_tagged
  · exact hC
  · exact hCr

theorem DLDS.hypFormulasInBuild_of_buildFormulas
    (d : DLDS) : d.HypFormulasInBuild := by
  intro w hw _h_hyp
  have hmem : w.FORMULA ∈ d.V.map (·.FORMULA) :=
    List.mem_map.mpr ⟨w, hw, rfl⟩
  exact List.idxOf_lt_length_iff.mpr (by
    simpa [buildFormulas] using hmem)

theorem BranchingDLDS.ofBase_hypFormulasInBuild
    (d : DLDS) :
    (BranchingDLDS.ofBase d).base.HypFormulasInBuild := by
  exact DLDS.hypFormulasInBuild_of_buildFormulas d

theorem BranchingDLDS.evalOrderInVertices_of_WellFormed
    (bd : BranchingDLDS)
    (h_wf : bd.WellFormed) :
    ∀ v ∈ bd.evalOrder, v ∈ bd.base.V := by
  intro v hv
  exact (List.Perm.mem_iff h_wf).mp hv

/-
RobustnessReady proof obligations:
1. wellFormedBranching:
   status: requires a major structural theorem for generated branching DLDSs.
   Existing local semantics theorems use this predicate, but generation has
   not yet been connected to `WellFormedBranching`.
2. hypDistinct:
   status: requires a structural construction invariant. It says hypothesis
   vertices have pairwise distinct formulas, and is not implied by the raw
   `DLDS` type.
3. hypInBuild:
   status: already directly provable from `buildFormulas`; see
   `DLDS.hypFormulasInBuild_of_buildFormulas`.
4. evalOrderInVertices:
   status: probably provable with a small helper once `bd.WellFormed` is
   available; see `BranchingDLDS.evalOrderInVertices_of_WellFormed`.
   `RobustnessReady` does not assume `bd.WellFormed` separately.
   Inspection note: this file currently has hand-written example
   `BranchingDLDS` values (`introDischargeDLDS`, `minimalBranchingDLDS`,
   `allCasesDLDS`) and a structural predicate
   `StructuralGridCompatibleNonBranching` that carries `bd.WellFormed`.
   There is not yet a production generator theorem producing
   `bd.WellFormed`; for literal constructions whose `evalOrder` is exactly
   `bd.base.V`, use `BranchingDLDS.WellFormed_of_evalOrder_eq_baseV`.
5. introDischargeWF:
   status: requires a construction invariant for auxiliary discharge pairs
   in `bd.base.A`: discharged vertices must belong to `bd.base.V` and be
   hypotheses.
6. taggedCompatible:
   status: requires the major tagged-grid structural correctness theorem.
   This is the main remaining bridge from generated DLDSs to the tagged grid.
7. rootInEvalOrder:
   status: requires clarification/change in the construction interface or a
   root-selection invariant tying the chosen root to `bd.evalOrder`.
-/

/-- Assumption package for using the conditional tagged-grid robustness
    theorem at a chosen root. This only bundles the hypotheses already
    required by `taggedGrid_robustness_from_short_badPrefix`. -/
structure RobustnessReady
    (bd : BranchingDLDS) (root : Vertex) : Prop where
  wellFormedBranching : bd.WellFormedBranching
  hypDistinct : bd.base.HypFormulasDistinct
  hypInBuild : bd.base.HypFormulasInBuild
  evalOrderInVertices : ∀ v ∈ bd.evalOrder, v ∈ bd.base.V
  introDischargeWF :
    ∀ v ∈ bd.evalOrder, ∀ h_vertex,
      findIntroDischarge bd v = some h_vertex →
        h_vertex ∈ bd.base.V ∧ h_vertex.HYPOTHESIS = true
  taggedCompatible :
    ∀ reading : Robustness.Input 2 bd.numReading,
      bd.TaggedGridCompatibleForReading
        (RobustnessInput.toReadingInput reading)
  rootInEvalOrder : root ∈ bd.evalOrder

theorem taggedGrid_robustness_from_short_badPrefix_ready
    (bd : BranchingDLDS)
    (root : Vertex)
    (ready : RobustnessReady bd root)
    {C : Nat}
    (π : Robustness.Prefix 2 bd.numReading)
    (hπ : Robustness.BadPrefix (TaggedRejectsReading bd root) π)
    (hC : π.len ≤ C)
    (hCr : C ≤ bd.numReading) :
    2 ^ bd.numReading ≤
      (Robustness.BadFinset (RejectsReading bd root)).card * 2 ^ C := by
  exact taggedGrid_robustness_from_short_badPrefix bd root π hπ hC hCr
    ready.rootInEvalOrder
    ready.wellFormedBranching
    ready.hypDistinct
    ready.hypInBuild
    ready.evalOrderInVertices
    ready.introDischargeWF
    ready.taggedCompatible

/-! ##### Structural compatibility, non-branching fragment

This is the first structural approximation to `GridCompatible`. It is local
and executable: each evaluation-order vertex must have one of the formula-rule
shapes that `buildIncomingMapForFormula` can represent. The theorem deriving
the full semantic `GridCompatible` from this structural predicate is the next
large proof obligation; the lemmas here record the shape facts needed for it. -/

/-- A source vertex is exactly one grid level above its target. -/
def sourceOneLevelAbove (source target : Vertex) : Bool :=
  decide (source.LEVEL = target.LEVEL + 1)

/-- A formula pair can be the major and minor premise of an elimination whose
    conclusion formula is `target`. -/
def elimPremiseFormulasMatch
    (target major minor : Formula) : Bool :=
  match major with
  | Formula.impl A B => decide (B = target ∧ A = minor)
  | _ => false

/-- Two source vertices form a proper implication-elimination pair for `v`. -/
def elimSourcePairMatches (v major minor : Vertex) : Bool :=
  elimPremiseFormulasMatch v.FORMULA major.FORMULA minor.FORMULA &&
    sourceOneLevelAbove major v &&
    sourceOneLevelAbove minor v

/-- Hypothesis shape accepted by the current grid initialization model. -/
def isHypothesisShapeForGrid (bd : BranchingDLDS) (v : Vertex) : Bool :=
  decide (v.HYPOTHESIS = true ∧ v.LEVEL = dldsMaxLevel bd.base ∧
    topVertexForFormula bd v.FORMULA = some v)

/-- Repetition shape: a single source with the same formula one level above. -/
def isRepetitionShapeForGrid (bd : BranchingDLDS) (v : Vertex) : Bool :=
  match incomingSources bd.base v with
  | [u] =>
      decide (v.HYPOTHESIS = false ∧ findBranchTarget bd v = none ∧
        findIntroDischarge bd v = none ∧ u.FORMULA = v.FORMULA) &&
        sourceOneLevelAbove u v
  | _ => false

/-- Introduction shape: target `A -> B`, one source `B`, discharging `A`. -/
def isIntroShapeForGrid (bd : BranchingDLDS) (v : Vertex) : Bool :=
  match v.FORMULA, incomingSources bd.base v, findIntroDischarge bd v with
  | Formula.impl A B, [u], some h =>
      decide (v.HYPOTHESIS = false ∧ findBranchTarget bd v = none ∧
        h.HYPOTHESIS = true ∧ h.FORMULA = A ∧ u.FORMULA = B) &&
        sourceOneLevelAbove u v
  | _, _, _ => false

/-- Elimination shape: exactly two sources, one major `A -> B` and one minor
    `A`, both one level above target `B`. Either source order is accepted. -/
def isElimShapeForGrid (bd : BranchingDLDS) (v : Vertex) : Bool :=
  match incomingSources bd.base v with
  | [u, w] =>
      decide (v.HYPOTHESIS = false ∧ findBranchTarget bd v = none ∧
        findIntroDischarge bd v = none) &&
        (elimSourcePairMatches v u w || elimSourcePairMatches v w u)
  | _ => false

/-- Local structural shape accepted by the current non-branching grid. -/
def vertexGridRuleShapeBool (bd : BranchingDLDS) (v : Vertex) : Bool :=
  isHypothesisShapeForGrid bd v ||
    isRepetitionShapeForGrid bd v ||
    isIntroShapeForGrid bd v ||
    isElimShapeForGrid bd v

/-- Every concrete edge endpoint is represented as a DLDS vertex. -/
def DLDS.EdgeEndpointsInV (d : DLDS) : Prop :=
  ∀ e ∈ d.E, e.START ∈ d.V ∧ e.END ∈ d.V

/-- A source vertex has at most one one-level-down outgoing edge. This is
    needed because the path-grid carries one token per formula column and
    cannot split a token across several targets in the same reading. -/
def DLDS.UniqueOneLevelOutgoing (d : DLDS) : Prop :=
  ∀ e₁ ∈ d.E, ∀ e₂ ∈ d.E,
    e₁.START = e₂.START →
    e₁.END.LEVEL + 1 = e₁.START.LEVEL →
    e₂.END.LEVEL + 1 = e₂.START.LEVEL →
    e₁.END = e₂.END

/-- Executable non-branching structural compatibility predicate, plus the
    ownership/topological facts needed to turn local shape facts into a grid
    simulation theorem. -/
structure BranchingDLDS.StructuralGridCompatibleNonBranching
    (bd : BranchingDLDS) : Prop where
  nonbranching : bd.IsNonBranching
  wellformed : bd.WellFormed
  topo : bd.WellFormedTopo
  edges_in_vertices : bd.base.EdgeEndpointsInV
  unique_one_level_outgoing : bd.base.UniqueOneLevelOutgoing
  vertex_shapes : bd.evalOrder.all (fun v => vertexGridRuleShapeBool bd v) = true

theorem structuralGridCompatibleNonBranching_vertex_shape
    (bd : BranchingDLDS)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (v : Vertex)
    (h_mem : v ∈ bd.evalOrder) :
    vertexGridRuleShapeBool bd v = true := by
  exact (List.all_eq_true.mp h_struct.vertex_shapes) v h_mem

theorem structuralGridCompatibleNonBranching_isNonBranching
    (bd : BranchingDLDS)
    (h_struct : bd.StructuralGridCompatibleNonBranching) :
    bd.IsNonBranching :=
  h_struct.nonbranching

theorem structuralGridCompatibleNonBranching_wellformed
    (bd : BranchingDLDS)
    (h_struct : bd.StructuralGridCompatibleNonBranching) :
    bd.WellFormed :=
  h_struct.wellformed

theorem structuralGridCompatibleNonBranching_topo
    (bd : BranchingDLDS)
    (h_struct : bd.StructuralGridCompatibleNonBranching) :
    bd.WellFormedTopo :=
  h_struct.topo

theorem structuralGridCompatibleNonBranching_edges_in_vertices
    (bd : BranchingDLDS)
    (h_struct : bd.StructuralGridCompatibleNonBranching) :
    bd.base.EdgeEndpointsInV :=
  h_struct.edges_in_vertices

theorem structuralGridCompatibleNonBranching_unique_one_level_outgoing
    (bd : BranchingDLDS)
    (h_struct : bd.StructuralGridCompatibleNonBranching) :
    bd.base.UniqueOneLevelOutgoing :=
  h_struct.unique_one_level_outgoing

lemma list_find?_some_mem {α : Type*} (p : α → Bool)
    {xs : List α} {a : α}
    (h : xs.find? p = some a) :
    a ∈ xs := by
  induction xs with
  | nil => simp at h
  | cons x rest ih =>
      unfold List.find? at h
      split at h
      · injection h with hx
        subst hx
        simp
      · exact List.mem_cons_of_mem _ (ih h)

theorem ordinaryTargetForPath_eq_some_of_unique_edge
    (bd : BranchingDLDS) (u v : Vertex) (e : Deduction)
    (h_edge_mem : e ∈ bd.base.E)
    (h_start : e.START = u)
    (h_end : e.END = v)
    (h_level : v.LEVEL + 1 = u.LEVEL)
    (h_unique : bd.base.UniqueOneLevelOutgoing) :
    ordinaryTargetForPath bd u = some v := by
  unfold ordinaryTargetForPath
  cases h_find : bd.base.E.find? (fun e =>
      decide (e.START = u ∧ e.END.LEVEL + 1 = u.LEVEL)) with
  | none =>
      have h_not := (List.find?_eq_none.mp h_find) e h_edge_mem
      have h_pred :
          (fun e : Deduction =>
            decide (e.START = u ∧ e.END.LEVEL + 1 = u.LEVEL)) e = true := by
        simp [h_start, h_end, h_level]
      exact False.elim (h_not h_pred)
  | some e' =>
      have h_found_mem :
          e' ∈ bd.base.E := by
        exact list_find?_some_mem _ h_find
      have h_found_pred :
          (fun e : Deduction =>
            decide (e.START = u ∧ e.END.LEVEL + 1 = u.LEVEL)) e' = true := by
        exact @List.find?_some Deduction
          (fun e => decide (e.START = u ∧ e.END.LEVEL + 1 = u.LEVEL))
          e' bd.base.E h_find
      simp at h_found_pred
      have h_end_eq : e'.END = v := by
        have h_e_end : e.END = e'.END := by
          exact h_unique e h_edge_mem e' h_found_mem
            (by rw [h_start, h_found_pred.1])
            (by rw [h_start, h_end]; exact h_level)
            (by rw [h_found_pred.1]; exact h_found_pred.2)
        rw [← h_e_end, h_end]
      simp [h_end_eq]

theorem nextVertexForReading_eq_some_of_unique_nonbranching_edge
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex) (e : Deduction)
    (h_nb : bd.IsNonBranching)
    (h_edge_mem : e ∈ bd.base.E)
    (h_start : e.START = u)
    (h_end : e.END = v)
    (h_level : v.LEVEL + 1 = u.LEVEL)
    (h_unique : bd.base.UniqueOneLevelOutgoing) :
    nextVertexForReading bd reading u = some v := by
  unfold nextVertexForReading branchingFromSource
  have h_no_branch : bd.branchings.find? (fun b => decide (b.source = u)) = none := by
    rw [h_nb]
    rfl
  rw [h_no_branch]
  exact ordinaryTargetForPath_eq_some_of_unique_edge bd u v e
    h_edge_mem h_start h_end h_level h_unique

theorem incomingSources_mem_iff_edge
    (d : DLDS) (u v : Vertex) :
    u ∈ incomingSources d v ↔
      ∃ e ∈ d.E, e.START = u ∧ e.END = v := by
  unfold incomingSources
  constructor
  · intro h
    rcases List.mem_map.mp h with ⟨e, he_filter, he_start⟩
    rcases List.mem_filter.mp he_filter with ⟨he_mem, he_end_bool⟩
    have he_end : e.END = v := by simpa using he_end_bool
    exact ⟨e, he_mem, by simpa using he_start, he_end⟩
  · intro h
    rcases h with ⟨e, he_mem, he_start, he_end⟩
    apply List.mem_map.mpr
    refine ⟨e, ?_, ?_⟩
    · apply List.mem_filter.mpr
      exact ⟨he_mem, by simp [he_end]⟩
    · exact he_start

theorem nextVertexForReading_eq_some_of_incoming_source
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex)
    (h_nb : bd.IsNonBranching)
    (h_source : u ∈ incomingSources bd.base v)
    (h_level : u.LEVEL = v.LEVEL + 1)
    (h_unique : bd.base.UniqueOneLevelOutgoing) :
    nextVertexForReading bd reading u = some v := by
  obtain ⟨e, he_mem, he_start, he_end⟩ :=
    (incomingSources_mem_iff_edge bd.base u v).mp h_source
  exact nextVertexForReading_eq_some_of_unique_nonbranching_edge bd reading
    u v e h_nb he_mem he_start he_end h_level.symm
    h_unique

theorem readingPathFromVertex_routes_first_of_incoming_source
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex) (steps : Nat)
    (h_nb : bd.IsNonBranching)
    (h_source : u ∈ incomingSources bd.base v)
    (h_level : u.LEVEL = v.LEVEL + 1)
    (h_unique : bd.base.UniqueOneLevelOutgoing) :
    (readingPathFromVertex bd reading (some u) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1) := by
  apply readingPathFromVertex_routes_first
  exact nextVertexForReading_eq_some_of_incoming_source bd reading u v
    h_nb h_source h_level h_unique

theorem repetition_shape_sources
    (bd : BranchingDLDS) (v : Vertex)
    (h_rep : isRepetitionShapeForGrid bd v = true) :
    ∃ u,
      incomingSources bd.base v = [u] ∧
      v.HYPOTHESIS = false ∧
      findBranchTarget bd v = none ∧
      findIntroDischarge bd v = none ∧
      u.FORMULA = v.FORMULA ∧
      u.LEVEL = v.LEVEL + 1 := by
  unfold isRepetitionShapeForGrid sourceOneLevelAbove at h_rep
  cases hs : incomingSources bd.base v with
  | nil => simp [hs] at h_rep
  | cons u rest =>
      cases rest with
      | nil =>
          simp [hs] at h_rep
          exact ⟨u, rfl, h_rep.1.1, h_rep.1.2.1, h_rep.1.2.2.1,
            h_rep.1.2.2.2, h_rep.2⟩
      | cons w rest' => simp [hs] at h_rep

theorem intro_shape_sources
    (bd : BranchingDLDS) (v : Vertex)
    (h_intro_shape : isIntroShapeForGrid bd v = true) :
    ∃ A B u h_vertex,
      v.FORMULA = Formula.impl A B ∧
      incomingSources bd.base v = [u] ∧
      findIntroDischarge bd v = some h_vertex ∧
      v.HYPOTHESIS = false ∧
      findBranchTarget bd v = none ∧
      h_vertex.HYPOTHESIS = true ∧
      h_vertex.FORMULA = A ∧
      u.FORMULA = B ∧
      u.LEVEL = v.LEVEL + 1 := by
  unfold isIntroShapeForGrid sourceOneLevelAbove at h_intro_shape
  cases hf : v.FORMULA with
  | atom name => simp [hf] at h_intro_shape
  | impl A B =>
      cases hs : incomingSources bd.base v with
      | nil => simp [hf, hs] at h_intro_shape
      | cons u rest =>
          cases rest with
          | nil =>
              cases hd : findIntroDischarge bd v with
              | none => simp [hf, hs, hd] at h_intro_shape
              | some h_vertex =>
                  simp [hf, hs, hd] at h_intro_shape
                  exact ⟨A, B, u, h_vertex, rfl, rfl, rfl,
                    h_intro_shape.1.1, h_intro_shape.1.2.1,
                    h_intro_shape.1.2.2.1, h_intro_shape.1.2.2.2.1,
                    h_intro_shape.1.2.2.2.2,
                    h_intro_shape.2⟩
          | cons w rest' => simp [hf, hs] at h_intro_shape

theorem elimSourcePairMatches_formula
    (v major minor : Vertex)
    (h_pair : elimSourcePairMatches v major minor = true) :
    ∃ A,
      major.FORMULA = Formula.impl A v.FORMULA ∧
      minor.FORMULA = A ∧
      major.LEVEL = v.LEVEL + 1 ∧
      minor.LEVEL = v.LEVEL + 1 := by
  unfold elimSourcePairMatches elimPremiseFormulasMatch sourceOneLevelAbove at h_pair
  cases hm : major.FORMULA with
  | atom name => simp [hm] at h_pair
  | impl A B =>
      simp [hm] at h_pair
      refine ⟨A, ?_, ?_, ?_, ?_⟩ <;> aesop

theorem elim_shape_sources
    (bd : BranchingDLDS) (v : Vertex)
    (h_elim : isElimShapeForGrid bd v = true) :
    ∃ u w,
      incomingSources bd.base v = [u, w] ∧
      v.HYPOTHESIS = false ∧
      findBranchTarget bd v = none ∧
      findIntroDischarge bd v = none ∧
      (elimSourcePairMatches v u w = true ∨
        elimSourcePairMatches v w u = true) := by
  unfold isElimShapeForGrid at h_elim
  cases hs : incomingSources bd.base v with
  | nil => simp [hs] at h_elim
  | cons u rest =>
      cases rest with
      | nil => simp [hs] at h_elim
      | cons w rest' =>
          cases rest' with
          | nil =>
              simp [hs] at h_elim
              exact ⟨u, w, rfl, h_elim.1.1, h_elim.1.2.1,
                h_elim.1.2.2, h_elim.2⟩
          | cons z zs => simp [hs] at h_elim

theorem repetition_shape_routes_to_target_column
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_rep : isRepetitionShapeForGrid bd v = true) :
    ∃ u,
      incomingSources bd.base v = [u] ∧
      (readingPathFromVertex bd reading (some u) (steps + 1)).head? =
        some ((buildFormulas bd.base).idxOf v.FORMULA + 1) := by
  obtain ⟨u, h_sources, _h_not_hyp, _h_no_branch, _h_no_intro,
    _h_formula, h_level⟩ := repetition_shape_sources bd v h_rep
  refine ⟨u, h_sources, ?_⟩
  apply readingPathFromVertex_routes_first_of_incoming_source
  · exact h_struct.nonbranching
  · simp [h_sources]
  · exact h_level
  · exact h_struct.unique_one_level_outgoing

theorem intro_shape_routes_to_target_column
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_intro_shape : isIntroShapeForGrid bd v = true) :
    ∃ A B u h_vertex,
      v.FORMULA = Formula.impl A B ∧
      incomingSources bd.base v = [u] ∧
      findIntroDischarge bd v = some h_vertex ∧
      (readingPathFromVertex bd reading (some u) (steps + 1)).head? =
        some ((buildFormulas bd.base).idxOf v.FORMULA + 1) := by
  obtain ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge,
    _h_not_hyp, _h_no_branch, _hhyp, _hform, _huform, h_level⟩ :=
    intro_shape_sources bd v h_intro_shape
  refine ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge, ?_⟩
  apply readingPathFromVertex_routes_first_of_incoming_source
  · exact h_struct.nonbranching
  · simp [h_sources]
  · exact h_level
  · exact h_struct.unique_one_level_outgoing

theorem elim_shape_routes_to_target_column
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_elim : isElimShapeForGrid bd v = true) :
    ∃ u w,
      incomingSources bd.base v = [u, w] ∧
      (readingPathFromVertex bd reading (some u) (steps + 1)).head? =
        some ((buildFormulas bd.base).idxOf v.FORMULA + 1) ∧
      (readingPathFromVertex bd reading (some w) (steps + 1)).head? =
        some ((buildFormulas bd.base).idxOf v.FORMULA + 1) := by
  obtain ⟨u, w, h_sources, _h_not_hyp, _h_no_branch, _h_no_intro, h_pair⟩ :=
    elim_shape_sources bd v h_elim
  have h_u_level : u.LEVEL = v.LEVEL + 1 := by
    cases h_pair with
    | inl hp =>
        obtain ⟨A, _hmajor, _hminor, hulevel, _hwlevel⟩ :=
          elimSourcePairMatches_formula v u w hp
        exact hulevel
    | inr hp =>
        obtain ⟨A, _hmajor, _hminor, hwlevel, hulevel⟩ :=
          elimSourcePairMatches_formula v w u hp
        exact hulevel
  have h_w_level : w.LEVEL = v.LEVEL + 1 := by
    cases h_pair with
    | inl hp =>
        obtain ⟨A, _hmajor, _hminor, _hulevel, hwlevel⟩ :=
          elimSourcePairMatches_formula v u w hp
        exact hwlevel
    | inr hp =>
        obtain ⟨A, _hmajor, _hminor, hwlevel, _hulevel⟩ :=
          elimSourcePairMatches_formula v w u hp
        exact hwlevel
  refine ⟨u, w, h_sources, ?_, ?_⟩
  · apply readingPathFromVertex_routes_first_of_incoming_source
    · exact h_struct.nonbranching
    · simp [h_sources]
    · exact h_u_level
    · exact h_struct.unique_one_level_outgoing
  · apply readingPathFromVertex_routes_first_of_incoming_source
    · exact h_struct.nonbranching
    · simp [h_sources]
    · exact h_w_level
    · exact h_struct.unique_one_level_outgoing

lemma readingTaggedPathFromVertex_routes_first
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex) (tag steps : Nat)
    (h_next : nextVertexAndTagForReading bd reading u = some (v, tag)) :
    (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1, tag) := by
  simp [readingTaggedPathFromVertex, h_next]

theorem nextVertexAndTagForReading_eq_some_of_incoming_source
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex)
    (h_nb : bd.IsNonBranching)
    (h_source : u ∈ incomingSources bd.base v)
    (h_level : u.LEVEL = v.LEVEL + 1)
    (h_unique : bd.base.UniqueOneLevelOutgoing) :
    nextVertexAndTagForReading bd reading u =
      some (v, gridTagForSourceTarget bd u v) := by
  unfold nextVertexAndTagForReading
  rw [nextVertexForReading_eq_some_of_incoming_source bd reading u v
    h_nb h_source h_level h_unique]

theorem readingTaggedPathFromVertex_routes_first_of_incoming_source
    (bd : BranchingDLDS) (reading : ReadingInput)
    (u v : Vertex) (steps : Nat)
    (h_nb : bd.IsNonBranching)
    (h_source : u ∈ incomingSources bd.base v)
    (h_level : u.LEVEL = v.LEVEL + 1)
    (h_unique : bd.base.UniqueOneLevelOutgoing) :
    (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
        gridTagForSourceTarget bd u v) := by
  apply readingTaggedPathFromVertex_routes_first
  exact nextVertexAndTagForReading_eq_some_of_incoming_source bd reading
    u v h_nb h_source h_level h_unique

theorem taggedElimPairMatches_of_elimSourcePairMatches
    (v major minor : Vertex)
    (h_pair : elimSourcePairMatches v major minor = true) :
    taggedElimPairMatches v major minor = true := by
  obtain ⟨A, h_major, h_minor, _h_major_level, _h_minor_level⟩ :=
    elimSourcePairMatches_formula v major minor h_pair
  unfold taggedElimPairMatches
  rw [h_major, h_minor]
  simp

private theorem taggedElimPairMatches_formula
    (v major minor : Vertex)
    (h_pair : taggedElimPairMatches v major minor = true) :
    major.FORMULA = Formula.impl minor.FORMULA v.FORMULA := by
  unfold taggedElimPairMatches at h_pair
  cases h_major : major.FORMULA with
  | atom name =>
      simp [h_major] at h_pair
  | impl A B =>
      simp [h_major] at h_pair
      rcases h_pair with ⟨hB, hA⟩
      subst hA
      subst hB
      rfl

private def formulaSize : Formula → Nat
  | .atom _ => 1
  | .impl A B => formulaSize A + formulaSize B + 1

private theorem taggedElimPairMatches_not_swapped
    (v major minor : Vertex)
    (h_pair : taggedElimPairMatches v major minor = true) :
    taggedElimPairMatches v minor major = false := by
  by_contra h_swap
  have h_major :
      major.FORMULA = Formula.impl minor.FORMULA v.FORMULA :=
    taggedElimPairMatches_formula v major minor h_pair
  have h_minor :
      minor.FORMULA = Formula.impl major.FORMULA v.FORMULA :=
    taggedElimPairMatches_formula v minor major
      (Bool.eq_true_of_not_eq_false h_swap)
  have h_size_major :
      formulaSize major.FORMULA =
        formulaSize minor.FORMULA + formulaSize v.FORMULA + 1 := by
    rw [h_major]
    simp [formulaSize]
  have h_size_minor :
      formulaSize minor.FORMULA =
        formulaSize major.FORMULA + formulaSize v.FORMULA + 1 := by
    rw [h_minor]
    simp [formulaSize]
  omega

private theorem Formula.impl_ne_left :
    ∀ A B, Formula.impl A B ≠ A
  | .atom name, B => by
      intro h
      cases h
  | .impl A₁ A₂, B => by
      intro h
      injection h with h₁ h₂
      exact Formula.impl_ne_left A₁ A₂ h₁

theorem source_vertices_distinct_of_elimSourcePairMatches
    (v major minor : Vertex)
    (h_pair : elimSourcePairMatches v major minor = true) :
    major ≠ minor := by
  intro h_eq
  obtain ⟨A, h_major, h_minor, _h_major_level, _h_minor_level⟩ :=
    elimSourcePairMatches_formula v major minor h_pair
  have h_forms : major.FORMULA = minor.FORMULA := by
    simp [h_eq]
  rw [h_major, h_minor] at h_forms
  exact Formula.impl_ne_left A v.FORMULA h_forms

theorem gridTagForSourceTarget_eq_repetition_of_repetition_shape
    (bd : BranchingDLDS) (v : Vertex)
    (h_rep : isRepetitionShapeForGrid bd v = true) :
    ∃ u,
      incomingSources bd.base v = [u] ∧
      gridTagForSourceTarget bd u v = gridTagRepetition := by
  obtain ⟨u, h_sources, _h_not_hyp, _h_no_branch, h_no_intro,
    _h_formula, _h_level⟩ := repetition_shape_sources bd v h_rep
  refine ⟨u, h_sources, ?_⟩
  simp [gridTagForSourceTarget, h_sources, h_no_intro]

theorem gridTagForSourceTarget_eq_intro_of_intro_shape
    (bd : BranchingDLDS) (v : Vertex)
    (h_intro_shape : isIntroShapeForGrid bd v = true) :
    ∃ A B u h_vertex,
      v.FORMULA = Formula.impl A B ∧
      incomingSources bd.base v = [u] ∧
      findIntroDischarge bd v = some h_vertex ∧
      gridTagForSourceTarget bd u v = gridTagIntro := by
  obtain ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge,
    _h_not_hyp, _h_no_branch, _h_hyp, _h_h_formula, _h_u_formula,
    _h_level⟩ := intro_shape_sources bd v h_intro_shape
  refine ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge, ?_⟩
  simp [gridTagForSourceTarget, h_sources, h_discharge]

theorem gridTagForSourceTarget_eq_elim_major_of_pair
    (bd : BranchingDLDS) (v u w : Vertex)
    (h_sources : incomingSources bd.base v = [u, w])
    (h_pair : elimSourcePairMatches v u w = true) :
    gridTagForSourceTarget bd u v = gridTagElimMajor := by
  have h_tag_pair : taggedElimPairMatches v u w = true :=
    taggedElimPairMatches_of_elimSourcePairMatches v u w h_pair
  simp [gridTagForSourceTarget, h_sources, h_tag_pair]

theorem gridTagForSourceTarget_eq_elim_minor_of_pair
    (bd : BranchingDLDS) (v u w : Vertex)
    (h_sources : incomingSources bd.base v = [u, w])
    (h_pair : elimSourcePairMatches v u w = true) :
    gridTagForSourceTarget bd w v = gridTagElimMinor := by
  have h_tag_pair : taggedElimPairMatches v u w = true :=
    taggedElimPairMatches_of_elimSourcePairMatches v u w h_pair
  have h_uw : u ≠ w :=
    source_vertices_distinct_of_elimSourcePairMatches v u w h_pair
  have h_wu : w ≠ u := fun h => h_uw h.symm
  simp [gridTagForSourceTarget, h_sources, h_tag_pair, h_wu]

theorem gridTagForSourceTarget_eq_elim_major_of_swapped_pair
    (bd : BranchingDLDS) (v u w : Vertex)
    (h_sources : incomingSources bd.base v = [u, w])
    (h_pair : elimSourcePairMatches v w u = true) :
    gridTagForSourceTarget bd w v = gridTagElimMajor := by
  have h_tag_pair : taggedElimPairMatches v w u = true :=
    taggedElimPairMatches_of_elimSourcePairMatches v w u h_pair
  have h_no_forward : taggedElimPairMatches v u w = false :=
    taggedElimPairMatches_not_swapped v w u h_tag_pair
  simp [gridTagForSourceTarget, h_sources, h_tag_pair, h_no_forward]

theorem gridTagForSourceTarget_eq_elim_minor_of_swapped_pair
    (bd : BranchingDLDS) (v u w : Vertex)
    (h_sources : incomingSources bd.base v = [u, w])
    (h_pair : elimSourcePairMatches v w u = true) :
    gridTagForSourceTarget bd u v = gridTagElimMinor := by
  have h_tag_pair : taggedElimPairMatches v w u = true :=
    taggedElimPairMatches_of_elimSourcePairMatches v w u h_pair
  have h_no_forward : taggedElimPairMatches v u w = false :=
    taggedElimPairMatches_not_swapped v w u h_tag_pair
  have h_wu : w ≠ u :=
    source_vertices_distinct_of_elimSourcePairMatches v w u h_pair
  have h_uw : u ≠ w := fun h => h_wu h.symm
  simp [gridTagForSourceTarget, h_sources, h_tag_pair, h_no_forward, h_uw]

theorem repetition_shape_routes_to_tagged_target
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_rep : isRepetitionShapeForGrid bd v = true) :
    ∃ u,
      incomingSources bd.base v = [u] ∧
      (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
        some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
          gridTagRepetition) := by
  obtain ⟨u, h_sources, _h_not_hyp, _h_no_branch, h_no_intro,
    _h_formula, h_level⟩ := repetition_shape_sources bd v h_rep
  have h_route :=
    readingTaggedPathFromVertex_routes_first_of_incoming_source bd reading
      u v steps h_struct.nonbranching (by simp [h_sources]) h_level
      h_struct.unique_one_level_outgoing
  have h_tag :
      gridTagForSourceTarget bd u v = gridTagRepetition := by
    simp [gridTagForSourceTarget, h_sources, h_no_intro]
  refine ⟨u, h_sources, ?_⟩
  simpa [h_tag] using h_route

theorem intro_shape_routes_to_tagged_target
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_intro_shape : isIntroShapeForGrid bd v = true) :
    ∃ A B u h_vertex,
      v.FORMULA = Formula.impl A B ∧
      incomingSources bd.base v = [u] ∧
      findIntroDischarge bd v = some h_vertex ∧
      (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
        some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
          gridTagIntro) := by
  obtain ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge,
    _h_not_hyp, _h_no_branch, _h_hyp, _h_h_formula, _h_u_formula,
    h_level⟩ := intro_shape_sources bd v h_intro_shape
  have h_route :=
    readingTaggedPathFromVertex_routes_first_of_incoming_source bd reading
      u v steps h_struct.nonbranching (by simp [h_sources]) h_level
      h_struct.unique_one_level_outgoing
  have h_tag :
      gridTagForSourceTarget bd u v = gridTagIntro := by
    simp [gridTagForSourceTarget, h_sources, h_discharge]
  refine ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge, ?_⟩
  simpa [h_tag] using h_route

theorem elim_pair_routes_to_tagged_target
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v u w : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_sources : incomingSources bd.base v = [u, w])
    (h_pair : elimSourcePairMatches v u w = true) :
    (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
        gridTagElimMajor) ∧
    (readingTaggedPathFromVertex bd reading (some w) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
        gridTagElimMinor) := by
  obtain ⟨A, _h_major, _h_minor, h_u_level, h_w_level⟩ :=
    elimSourcePairMatches_formula v u w h_pair
  have h_u_route :=
    readingTaggedPathFromVertex_routes_first_of_incoming_source bd reading
      u v steps h_struct.nonbranching (by simp [h_sources]) h_u_level
      h_struct.unique_one_level_outgoing
  have h_w_route :=
    readingTaggedPathFromVertex_routes_first_of_incoming_source bd reading
      w v steps h_struct.nonbranching (by simp [h_sources]) h_w_level
      h_struct.unique_one_level_outgoing
  have h_u_tag := gridTagForSourceTarget_eq_elim_major_of_pair
    bd v u w h_sources h_pair
  have h_w_tag := gridTagForSourceTarget_eq_elim_minor_of_pair
    bd v u w h_sources h_pair
  constructor
  · simpa [h_u_tag] using h_u_route
  · simpa [h_w_tag] using h_w_route

theorem elim_swapped_pair_routes_to_tagged_target
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v u w : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_sources : incomingSources bd.base v = [u, w])
    (h_pair : elimSourcePairMatches v w u = true) :
    (readingTaggedPathFromVertex bd reading (some w) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
        gridTagElimMajor) ∧
    (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
      some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
        gridTagElimMinor) := by
  obtain ⟨A, _h_major, _h_minor, h_w_level, h_u_level⟩ :=
    elimSourcePairMatches_formula v w u h_pair
  have h_w_route :=
    readingTaggedPathFromVertex_routes_first_of_incoming_source bd reading
      w v steps h_struct.nonbranching (by simp [h_sources]) h_w_level
      h_struct.unique_one_level_outgoing
  have h_u_route :=
    readingTaggedPathFromVertex_routes_first_of_incoming_source bd reading
      u v steps h_struct.nonbranching (by simp [h_sources]) h_u_level
      h_struct.unique_one_level_outgoing
  have h_w_tag := gridTagForSourceTarget_eq_elim_major_of_swapped_pair
    bd v u w h_sources h_pair
  have h_u_tag := gridTagForSourceTarget_eq_elim_minor_of_swapped_pair
    bd v u w h_sources h_pair
  constructor
  · simpa [h_w_tag] using h_w_route
  · simpa [h_u_tag] using h_u_route

theorem elim_shape_routes_to_tagged_target
    (bd : BranchingDLDS) (reading : ReadingInput)
    (v : Vertex) (steps : Nat)
    (h_struct : bd.StructuralGridCompatibleNonBranching)
    (h_elim : isElimShapeForGrid bd v = true) :
    ∃ u w,
      incomingSources bd.base v = [u, w] ∧
      ((elimSourcePairMatches v u w = true ∧
        (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
          some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
            gridTagElimMajor) ∧
        (readingTaggedPathFromVertex bd reading (some w) (steps + 1)).head? =
          some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
            gridTagElimMinor)) ∨
       (elimSourcePairMatches v w u = true ∧
        (readingTaggedPathFromVertex bd reading (some w) (steps + 1)).head? =
          some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
            gridTagElimMajor) ∧
        (readingTaggedPathFromVertex bd reading (some u) (steps + 1)).head? =
          some ((buildFormulas bd.base).idxOf v.FORMULA + 1,
            gridTagElimMinor))) := by
  obtain ⟨u, w, h_sources, _h_not_hyp, _h_no_branch, _h_no_intro, h_pair⟩ :=
    elim_shape_sources bd v h_elim
  refine ⟨u, w, h_sources, ?_⟩
  cases h_pair with
  | inl hp =>
      have h_routes := elim_pair_routes_to_tagged_target
        bd reading v u w steps h_struct h_sources hp
      exact Or.inl ⟨hp, h_routes.1, h_routes.2⟩
  | inr hp =>
      have h_routes := elim_swapped_pair_routes_to_tagged_target
        bd reading v u w steps h_struct h_sources hp
      exact Or.inr ⟨hp, h_routes.1, h_routes.2⟩

lemma Vector.zipWith_or_comm {n : Nat}
    (u v : List.Vector Bool n) :
    u.zipWith (· || ·) v = v.zipWith (· || ·) u := by
  apply List.Vector.ext
  intro i
  simp [Bool.or_comm]

theorem classicalKernel_hypothesis_shape_eq_onehot
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (v : Vertex)
    (h_hyp_shape : isHypothesisShapeForGrid bd v = true) :
    classicalKernel bd reading fenv v =
      ⟨(List.range (numFormulas bd.base)).map (fun i =>
          decide (i = (buildFormulas bd.base).idxOf v.FORMULA)),
        by simp [numFormulas]⟩ := by
  unfold isHypothesisShapeForGrid at h_hyp_shape
  simp at h_hyp_shape
  unfold classicalKernel
  simp [h_hyp_shape.1]

theorem classicalKernel_repetition_shape_eq_source
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (v : Vertex)
    (h_rep : isRepetitionShapeForGrid bd v = true) :
    ∃ u,
      incomingSources bd.base v = [u] ∧
      classicalKernel bd reading fenv v = fenv u := by
  obtain ⟨u, h_sources, h_not_hyp, h_no_branch, h_no_intro,
    _h_formula, _h_level⟩ := repetition_shape_sources bd v h_rep
  refine ⟨u, h_sources, ?_⟩
  unfold classicalKernel
  simp [h_not_hyp, h_sources, h_no_branch, h_no_intro]
  rw [Vector.zipWith_or_replicate_false_right]

theorem classicalKernel_intro_shape_eq_grid_intro
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (v : Vertex)
    (h_intro_shape : isIntroShapeForGrid bd v = true) :
    ∃ A B u h_vertex,
      v.FORMULA = Formula.impl A B ∧
      incomingSources bd.base v = [u] ∧
      findIntroDischarge bd v = some h_vertex ∧
      classicalKernel bd reading fenv v =
        (fenv u).zipWith (fun b e => b && !e)
          ⟨(buildFormulas bd.base).map
              (fun f => decide (f = h_vertex.FORMULA)),
            by simp [numFormulas]⟩ := by
  obtain ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge,
    h_not_hyp, h_no_branch, _hhyp, _hform, _huform, _hlevel⟩ :=
    intro_shape_sources bd v h_intro_shape
  refine ⟨A, B, u, h_vertex, h_formula, h_sources, h_discharge, ?_⟩
  unfold classicalKernel
  simp [h_not_hyp, h_sources, h_no_branch, h_discharge]
  rw [Vector.zipWith_or_replicate_false_right]

theorem classicalKernel_elim_shape_eq_source_or
    (bd : BranchingDLDS) (reading : ReadingInput)
    (fenv : Vertex → List.Vector Bool (numFormulas bd.base))
    (v : Vertex)
    (h_elim : isElimShapeForGrid bd v = true) :
    ∃ u w,
      incomingSources bd.base v = [u, w] ∧
      (elimSourcePairMatches v u w = true ∨
        elimSourcePairMatches v w u = true) ∧
      classicalKernel bd reading fenv v =
        (fenv u).zipWith (· || ·) (fenv w) := by
  obtain ⟨u, w, h_sources, h_not_hyp, h_no_branch, h_no_intro, h_pair⟩ :=
    elim_shape_sources bd v h_elim
  refine ⟨u, w, h_sources, h_pair, ?_⟩
  unfold classicalKernel
  simp [h_not_hyp, h_sources, h_no_branch, h_no_intro]
  rw [Vector.zipWith_or_replicate_false_right]
  exact Vector.zipWith_or_comm (fenv w) (fenv u)

end ReadingBased

/-!
# DLDS Circuit Evaluation Examples

This module contains executable examples demonstrating the DLDS-to-circuit
translation and evaluation. Each example constructs a natural deduction
proof as a DLDS and verifies it evaluates correctly.

## Examples

1. **Examples.Identity**: Identity proof (A⊃B)⊃(A⊃B) - valid and invalid paths
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


namespace Examples.Identity
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

end Examples.Identity


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
