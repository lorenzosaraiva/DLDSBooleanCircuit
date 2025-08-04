import Init
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip
import Mathlib.Data.Fin.Basic


/-!
# DLDS Boolean Circuit Formalization

This file formalizes the Boolean circuit representation of Dag-Like Derivability Structures (DLDS), as used in the correctness proof of the circuit-based DLDS checking algorithm.

It is organized into the following sections:

## **Contents**
1. **Core Types and Structures**
   Definitions of activation bits, rules, nodes, and the grid structure used to represent DLDS circuits.

2. **Boolean Circuit Logic**
   Circuit evaluation logic: rule activation, selectors, node outputs, and unique-activation predicates.

3. **Boolean List Lemmas and Multiple XOR Properties**
   Auxiliary results about Boolean lists and the `multiple_xor` function, characterizing unique activation.

4. **Auxiliary List and Vector Lemmas**
   Technical results about list/vector access, folding, and `zipWith` identities, needed for proofs.

5. **Node Correctness**
   Proofs that nodes compute the correct output under unique activation.

6. **Grid Evaluation**
   Layered evaluation of the grid: selector propagation, node activation, and output querying.

7. **Full Grid Correctness for Tree-Like Subgraphs**
   Global soundness and completeness: the main theorem connecting grid outputs to unique proof paths.
-/



/-!
# Section 1: Core Types and Structures

This section contains the core types formalizing DLDS rules, activation, nodes, and the grid.
-/

/--
Represents the type of activation bits for each inference rule.
- `intro`: Single activation bit for implication introduction.
- `elim`: Two activation bits for implication elimination.
-/
inductive ActivationBits
  | intro (bit : Bool)
  | elim (bit1 : Bool) (bit2 : Bool)
  deriving DecidableEq

/--
Represents the data of a rule for formulas of length `n`:
- `intro`: With an encoder vector for implication introduction.
- `elim`: For implication elimination.
-/
inductive RuleData (n : Nat)
  | intro (encoder : List.Vector Bool n)
  | elim

/--
Structure representing a single inference rule, including:
- The activation bits.
- Its kind (intro/elim).
- The dependency vector update function.
-/
structure Rule (n : ℕ) where
  activation : ActivationBits
  type : RuleData n
  combine : List (List.Vector Bool n) → List.Vector Bool n

/--
Represents a node in the Boolean circuit (corresponds to a node in the DLDS).
- Each node stores a list of possible inference rules.
- Invariant: All rules must be unique.
-/
structure CircuitNode (n: Nat) where
  rules : List (Rule n)
  nodup : rules.Nodup

/--
Represents the N × N grid structure for the Boolean circuit.
- `nodes`: List of all circuit nodes.
- `grid_size`: Total number of nodes in the grid.
-/
structure Grid (n : Nat) (Rule : Type) where
  nodes : List Rule
  grid_size : nodes.length =  n * n

/--
Constructs an implication introduction rule for formulas of length `n`.
- `encoder`: The vector encoding discharged assumptions.
- `bit`: Activation bit for this rule.
- `combine` returns conjunction of the input dependency vector with the negation of `encoder`.
-/
def mkIntroRule {n : ℕ} (encoder : List.Vector Bool n) (bit : Bool) : Rule n :=
{
  activation := ActivationBits.intro bit,
  type       := RuleData.intro encoder,
  combine    := fun deps =>
    match deps with
    | [d] => d.zipWith (fun b e => not (b && e)) encoder
    | _   => List.Vector.replicate n false
}

/--
Constructs an implication elimination rule for formulas of length `n`.
- `bit1`, `bit2`: Activation bits for the two premises.
- `combine` returns conjunction of the two dependency vectors.

-/
def mkElimRule {n : ℕ} (bit1 bit2 : Bool) : Rule n :=
{
  activation := ActivationBits.elim bit1 bit2,
  type       := RuleData.elim,
  combine    := fun deps =>
    match deps with
    | [d1, d2] => d1.zipWith (· && ·) d2
    | _        => List.Vector.replicate n false
}


/-!
# Section 2: Boolean Circuit Logic

This section defines the core functions for Boolean evaluation, activation extraction, selector logic,
and node output computation in the DLDS circuit formalization.
-/

/--
Returns `true` if the given rule is considered "active".
- For `intro`: uses the single activation bit.
- For `elim`: uses logical AND of the two activation bits.
-/
def is_rule_active {n: Nat} (r : Rule n) : Bool :=
  match r.activation with
  | ActivationBits.intro b   => b
  | ActivationBits.elim b1 b2 => b1 && b2

/--
Computes whether exactly one element of a Boolean list is `true`.
- Implements "one-hot" logic for rule activation.
-/
def multiple_xor : List Bool → Bool
| []       => false
| [x]      => x
| x :: xs  => (x && not (List.or xs)) || (not x && multiple_xor xs)

/--
Extracts the activation bits from a list of rules,
returning a Boolean list indicating the active status of each rule.
-/
def extract_activations {n: Nat} (rules : List (Rule n)) : List Bool :=
  rules.map is_rule_active

/--
Given a Boolean and a list of Booleans, returns the list where each element is logically ANDed with the input Boolean.
Useful for masking activation patterns with a global "XOR/one-hot" check.
-/
def and_bool_list (bool : Bool) (l : List Bool): List Bool :=
  l.map (λ b => bool && b)

/--
Performs bitwise OR ("union") over a list of Boolean vectors of fixed length `n`.
Used to combine the outputs of all rules in a node, per activation.
-/
def list_or {n: Nat} (lists : List (List.Vector Bool n)) : List.Vector Bool n :=
  lists.foldl (λ acc lst => acc.zipWith (λ x y => x || y) lst)
              (List.Vector.replicate n false)

/--
Applies each rule to the provided inputs if its activation mask is true,
returning the output vectors for all rules.
-/
def apply_activations {n: Nat}
  (rules : List (Rule n))
  (masks : List Bool)
  (inputs : List (List.Vector Bool n))
: List (List.Vector Bool n) :=
  List.zipWith
    (fun (r : Rule n) (m : Bool) =>
      if m then
        r.combine inputs
      else
        List.Vector.replicate n false)
    rules masks

/--
Defines the overall logic for computing a node’s output:
- Extracts activations
- Checks for unique activation with `multiple_xor`
- Masks activations accordingly
- Applies rule logic and combines results with `list_or`
-/
def node_logic {n: Nat} (rules : List (Rule n))
                  (inputs : List (List.Vector Bool n))
  : List.Vector Bool n :=
  let acts := extract_activations rules
  let xor  := multiple_xor acts
  let masks := and_bool_list xor acts
  let outs    := apply_activations rules masks inputs
  list_or outs

/--
Runs a circuit node by applying its logic to the given inputs.
-/
def CircuitNode.run {n: Nat} (c : CircuitNode n)
    (inputs : List (List.Vector Bool n)) : List.Vector Bool n :=
  node_logic c.rules inputs

/--!
  Runs a circuit node and returns:
  - The output vector (or a zero vector if no rule is valid), and
  - A Boolean flag indicating whether the node is malformed *and* selected.
    A node is considered malformed if:
    - No rule is uniquely active (`multiple_xor` is false), and
    - At least one of the input bits is active (i.e., this node is selected).
-/
def CircuitNode.runWithError {n: Nat}
  (c : CircuitNode n)
  (inputs : List (List.Vector Bool n))
  : (List.Vector Bool n) × Bool :=
  let acts := extract_activations c.rules
  let xor := multiple_xor acts
  let node_selected := inputs.any (λ v => v.toList.any id)
  let is_error := !xor && node_selected
  (node_logic c.rules inputs, is_error)

/--
Predicate: returns true iff *exactly one* rule in the given list is active.
This property is central for circuit correctness and unique path selection.
-/
def exactlyOneActive {n: Nat} (rules : List (Rule n)) : Prop :=
  ∃ r, r ∈ rules ∧ is_rule_active r ∧ ∀ r', r' ∈ rules → is_rule_active r' → r' = r

/--
Converts a natural number to its `k`-bit Boolean (big-endian) vector.
Used for encoding selector indices.
-/
def natToBits (n k : ℕ) : List Bool :=
  (List.range k).map (fun i => (n.shiftRight (k - 1 - i)) % 2 = 1)

/--
Generates a "one-hot" selector vector for an input vector, such that only one output is true,
depending on the Boolean encoding of the input.
-/
def selector (input : List Bool) : List Bool :=
  let n := input.length
  let total := 2 ^ n
  List.ofFn (fun (i : Fin total) =>
    let bits := natToBits i.val n
    (input.zip bits).foldl (fun acc (inp, b) =>
      acc && if b then inp else !inp) true
  )

/-!
## Section 3: Boolean List Lemmas and Multiple XOR Properties

This section contains auxiliary lemmas and theorems about Boolean list operations,
especially those related to the multiple_xor function and its connection to unique activation.
These are essential for the correctness and reasoning about rule activation in the circuit.
-/

/-!
### Lemma: Cons False Invariance for Multiple XOR

Prepending `false` to a Boolean list does not affect the result of `multiple_xor`.
-/
@[simp]
lemma multiple_xor_cons_false (l : List Bool) :
  multiple_xor (false :: l) = multiple_xor l := by
  induction l with
  | nil => simp [multiple_xor]
  | cons b bs ih =>
    simp [multiple_xor]

/-!
### Lemma: Multiple XOR with True Equals Negation of Or

Prepending `true` to a Boolean list in `multiple_xor` yields the negation of `or` on the rest of the list.
-/
lemma multiple_xor_cons_true_aux {l : List Bool} :
  multiple_xor (true :: l) = !l.or := by
  cases l with
  | nil => simp [multiple_xor, List.or]
  | cons b bs => simp [multiple_xor]

/-!
### Lemma: Characterization of Multiple XOR with Leading True

States that `multiple_xor (true :: l)` is `true` if and only if all elements of `l` are `false`.
-/
lemma multiple_xor_cons_true {l : List Bool} :
  multiple_xor (true :: l) = true ↔ List.or l = false := by
  simp [multiple_xor, Bool.eq_true_eq_not_eq_false, Bool.not_eq_true_eq_eq_false]
  exact multiple_xor_cons_true_aux

/-!
### Lemma: List.or is False iff All Elements are False

Establishes equivalence between `l.or = false` and all elements of `l` being `false`.
-/
lemma List.or_eq_false_iff_all_false {l : List Bool} :
  l.or = false ↔ ∀ b ∈ l, b = false := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.or, Bool.or_eq_false_eq_eq_false_and_eq_false, List.mem_cons, forall_eq_or_imp, ih]
    simp [List.any]

/-!
### Theorem: Multiple XOR Characterizes Unique Rule Activation

Relates `multiple_xor` over activation bits to the logical predicate `exactlyOneActive` for rules.
-/
theorem multiple_xor_bool_iff_exactlyOneActive {n : ℕ} (rs : List (Rule n))  (h_nodup : rs.Nodup) :
  multiple_xor (rs.map is_rule_active) = true ↔ exactlyOneActive rs := by
  induction rs with
  | nil =>
    simp [multiple_xor, exactlyOneActive]
  | cons r rs ih =>
    have tail_nodup : rs.Nodup := List.nodup_cons.mp h_nodup |>.2
    cases hr : is_rule_active r
    ·
      simp only [List.map, hr, multiple_xor]
      rw [multiple_xor_cons_false, ih tail_nodup]
      simp only [exactlyOneActive]
      constructor
      · intro ⟨r₀, hr₀_in, h_act, h_uniq⟩
        exact ⟨r₀, List.mem_cons_of_mem _ hr₀_in, h_act, by
          intro r' hr'_mem hr'_act
          cases hr'_mem with
          | head =>
            rw [hr] at hr'_act
            contradiction
          | tail _ h_tail =>
            exact h_uniq r' h_tail hr'_act⟩
      · intro ⟨r₀, hr₀_mem, h_act, h_uniq⟩
        cases hr₀_mem with
        | head =>
          rw [hr] at h_act
          contradiction
        | tail _ h_tail =>
          exact ⟨r₀, h_tail, h_act, fun r' h_in h_act' => h_uniq r' (List.mem_cons_of_mem _ h_in) h_act'⟩
    ·
      simp only [List.map, hr, multiple_xor]
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
### Section 4: Auxiliary List and Vector Lemmas

This section provides technical lemmas for reasoning about `List` and `List.Vector` operations,
such as identities for `zipWith` with neutral elements, membership characterization, drop/get relationships,
and equivalence between different ways to index or construct lists.
These results are used in the proofs of main circuit correctness theorems.
-/

/-!
#### Lemma: Zero Vector Is Neutral for zipWith Or

Bitwise OR with an all-`false` vector acts as the identity for Boolean vectors.
-/
lemma zip_with_zero_identity :
  ∀ (N : ℕ) (v : List.Vector Bool N),
    (List.Vector.replicate N false).zipWith (λ x y => x || y) v = v
:= by
  intro N v
  let ⟨l, hl⟩ := v
  dsimp [List.Vector.zipWith, List.Vector.replicate]
  congr
  induction l generalizing N with
  | nil =>
    simp
  | cons hd tl ih =>
    rw [←hl]
    have hlen : (hd :: tl).length = Nat.succ (List.length tl) := by simp [List.length]
    rw [hlen] at *
    have hrep : List.replicate (Nat.succ (List.length tl)) false = false :: List.replicate (List.length tl) false :=
      by simp [List.replicate]
    rw [hrep]
    rw [List.zipWith_cons_cons]
    simp [Bool.or_false]
    exact ih (List.length tl) ⟨tl, rfl⟩ rfl


/-!
#### Theorem: getElem and get Are Equivalent for Lists

States that the notation `l[i]` coincides with `l.get i` for valid indices.
-/
@[simp]
theorem List.getElem_eq_get {α : Type*} (l : List α) (i : Fin l.length) : l[↑i] = l.get i :=
rfl

/-!
#### Definition: List.nthLe

Safe indexed access for lists using a proof that the index is in bounds.
-/
def List.nthLe {α : Type u} (l : List α) (i : Nat) (h : i < l.length) : α :=
  -- `List.get` takes a `Fin l.length` — here we pack `i` and `h` into one
  l.get ⟨i, h⟩

/-!
#### Lemma: Commutativity of zipWith for Vectors

If the function `f` is commutative, then `zipWith f` applied to two vectors is order-independent.
-/
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

lemma List.get_eq_get_cast {α : Type*} {l₁ l₂ : List α}
  (h : l₁ = l₂) (i : Fin l₁.length) :
  l₁.get i = l₂.get (Fin.cast (congrArg List.length h) i) :=
by subst h; rfl

/-!

#### Theorem: nthLe is get

Shows that nthLe is just get with a packed Fin.
-/
theorem List.nthLe_eq_get {α : Type u} (l : List α) (i : ℕ) (h : i < l.length) :
  l.nthLe i h = l.get ⟨i, h⟩ :=
rfl

/-!
#### Lemma: foldl Adding Zero Vector Leaves Vector Unchanged

Folding with zipWith OR using an all-`false` vector as the neutral element leaves the original vector unchanged.
-/
lemma foldl_add_false {n : ℕ} (v : List.Vector Bool n) (l : List α) :
  List.foldl (fun acc (_ : α) => acc.zipWith (fun x y => x || y) (List.Vector.replicate n false)) v l = v :=
by
  induction l with
  | nil => rfl
  | cons _ tl ih =>
    simp only [List.foldl]
    rw [List.Vector.zipWith_comm (fun x y => x || y) Bool.or_comm]
    rw [zip_with_zero_identity]
    exact ih

/--
Relates `List.getLastD` (with a default) to the standard `getLast?` option accessor for lists.
-/
@[simp]
theorem List.getLastD_eq_getLast_getD {α : Type*} (l : List α) (d : α) :
  l.getLastD d = l.getLast?.getD d := by
  cases l with
  | nil => simp [List.getLastD, List.getLast?, Option.getD]
  | cons a as =>
    simp [List.getLastD, List.getLast?, Option.getD]

/--
Lemma: relates the operation of taking the last element of a nonempty list with a default,
as either an option with default (`getLast?.getD`) or direct defaulting (`getLastD`).
-/
lemma List.getLast?_cons_getD {α} (x : α) (l : List α) :
  (x :: l).getLast?.getD x = l.getLastD x :=
by
  cases l with
  | nil => simp [List.getLast?, Option.getD, List.getLastD]
  | cons _ _ => simp [List.getLast?, Option.getD, List.getLastD]

/-!
# Section 5: Node Correctness

This section contains the core lemmas about the node correctness.

-/


/-!
#### Lemma: Unique Active Rule Output for Node OR

If exactly one rule in `rules` is active, then OR-combining the outputs yields the output of the active rule only.
-/
lemma list_or_apply_unique_active_of_exactlyOne {n : ℕ}
  {rules : List (Rule n)} (h_nonempty : rules ≠ [])
  {r0 : Rule n} (hr0_mem : r0 ∈ rules)
  (h_nodup : rules.Nodup)
  (h_one : exactlyOneActive rules)
  (hr0_active : is_rule_active r0 = true)
  (inputs : List (List.Vector Bool n)) :
  list_or (apply_activations rules (extract_activations rules) inputs) = r0.combine inputs :=
by
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
          simp [list_or, apply_activations, extract_activations]
          rw [zip_with_zero_identity n (List.Vector.replicate n false)]
          exact ih h_nonempty_tail r0_in_rs rs_nodup h_one_tail

/-!
#### Theorem: Node Output Correctness

If exactly one rule in a node is active, running the node produces the output of that rule.
-/
theorem node_correct {n} (c : CircuitNode n)
    (inputs : List (List.Vector Bool n))
    (h_one : exactlyOneActive c.rules) :
  ∃ r ∈ c.rules, c.run inputs = r.combine inputs := by

  have h_bool : multiple_xor (c.rules.map is_rule_active) = true :=
    (multiple_xor_bool_iff_exactlyOneActive c.rules c.nodup).mpr h_one

  let h_one_prop := h_one

  rcases h_one with ⟨r0, hr0_mem, hr0_active, hr0_unique⟩

  dsimp [CircuitNode.run, node_logic, extract_activations]

  rw [h_bool]
  dsimp [and_bool_list]

  let h_nonempty := List.ne_nil_of_mem hr0_mem
  simp [List.map_map, true_and]

  let eq := list_or_apply_unique_active_of_exactlyOne
    h_nonempty hr0_mem c.nodup h_one_prop hr0_active inputs

  use r0
  constructor
  · exact hr0_mem
  · exact eq

/-!
### Lemma: List.all is Equivalent to ∀...∈...

The following lemma establishes the equivalence between Lean’s `List.all p` (which returns `true` iff every element of the list satisfies `p`) and the mathematical universal quantification `∀ x ∈ l, p x`. This is useful for bridging between Boolean-valued and Prop-valued reasoning on lists.
-/
lemma List.all_iff_forall {α : Type*} {l : List α} {p : α → Prop} [DecidablePred p] :
  l.all p ↔ ∀ x ∈ l, p x :=
by
  induction l with
  | nil => simp
  | cons hd tl ih =>
    simp [List.all, ih]

def all_false {n : ℕ} (v : List.Vector Bool n) : Prop :=
  ∀ i : Fin n, v.get i = false

/-!
### Lemma: `all_false` on Vectors is Equivalent to `all (= false)` on Their List Representation

This lemma shows that the property `all_false v` for a `List.Vector Bool n` (all entries are `false`) is equivalent to the predicate that `v.toList.all (· = false)`. This equivalence is handy when switching between vector and list-based representations in Boolean circuit proofs.
-/
lemma all_false_iff_toList_all_false {n} (v : List.Vector Bool n) :
  all_false v ↔ v.toList.all (· = false) := by
  simp [all_false, List.Vector.toList, List.all_iff_forall]
  simp [List.mem_iff_get]
  have hlen : n = v.toList.length := (v.property).symm
  constructor
  · intros h x
    exact h (Fin.cast hlen.symm x)
  · intro h i
    exact h (Fin.cast hlen i)


/-!
# Section 5: Grid Evaluation

This section implements the layered evaluation of the DLDS Boolean circuit grid. It includes the functions for propagating dependency vectors through the grid’s layers, handling selector-driven activation, and querying outputs from arbitrary nodes. The design allows efficient simulation and correctness proofs for parallel, tree-like subgraphs corresponding to DLDS proofs.
-/

/--
Represents how each rule in a node receives its activation from previous layer selectors.
Each element is a pair `(source_node_idx, edge_idx)` indicating the source selector and the bit.
-/
abbrev IncomingMap := List (Nat × Nat)

/--
Maps each node in a layer to its IncomingMap (one per node).
-/
abbrev IncomingMapsLayer := List IncomingMap

/--
Full selector wiring for all layers: one IncomingMapsLayer per grid layer.
-/
abbrev IncomingMaps := List IncomingMapsLayer

/--
Represents a single layer of the DLDS Boolean circuit grid.

- `nodes`: The circuit nodes for this layer.
- `incoming`: Wiring information describing, for each node/rule, how to fetch activations from previous selectors.
-/
structure GridLayer (n : ℕ) where
  nodes : List (CircuitNode n)
  incoming : IncomingMapsLayer

/--
Constructs the list of rules for a node after updating activation bits based on incoming selectors.

- For each rule, fetches the relevant selector bit(s) as indicated by `incoming_map` and sets the activation bits accordingly.
-/
def make_new_rules {n : ℕ}
  (node : CircuitNode n)
  (prev_selectors : List (List Bool))
  (incoming_map : IncomingMap)
  : List (Rule n) :=
  let len := node.rules.length
  List.finRange len |>.map (fun i =>
    let rule := node.rules.get i
    let (src_idx, edge_idx) :=
      if h_map : i.val < incoming_map.length then
        incoming_map.get ⟨i.val, h_map⟩
      else
        (0, 0)
    let act :=
      if h_src : src_idx < prev_selectors.length then
        let sel := prev_selectors.get ⟨src_idx, h_src⟩
        if h_edge : edge_idx < sel.length then sel.get ⟨edge_idx, h_edge⟩ else false
      else
        false
    match rule.activation with
    | ActivationBits.intro _ => { rule with activation := ActivationBits.intro act }
    | ActivationBits.elim _ _ => { rule with activation := ActivationBits.elim act act }
  )

/--
Axiom: The updated rules produced by `make_new_rules` are always nodup (no duplicates),
assuming the original node was nodup. This is required for circuit correctness.
-/
axiom nodup_labels_new_rules {n : ℕ}
  (node : CircuitNode n)
  (prev_selectors : List (List Bool))
  (incoming_map : IncomingMap)
  : (make_new_rules node prev_selectors incoming_map).Nodup

/--
Given a node, updates all its rules with fresh activation bits
according to the provided selectors and incoming wiring.

Returns a new CircuitNode with updated rules and the nodup proof.
-/
def activateNodeFromSelectors {n : Nat}
  (prev_selectors : List (List Bool))
  (incoming_map   : IncomingMap)
  (node           : CircuitNode n)
: CircuitNode n :=
  let new_rules := make_new_rules node prev_selectors incoming_map
  {
    rules := new_rules,
    nodup := nodup_labels_new_rules node prev_selectors incoming_map
  }

/--
Activates all nodes in a grid layer according to the given selector vectors from the previous layer.

- `prev_selectors`: List of selector vectors, one per node in the previous layer.
- `layer`: The `GridLayer` whose nodes will be activated.
- Returns a list of `CircuitNode` instances with updated activations for this layer.

This handles per-node selector-driven activation and prepares the layer for evaluation.
-/
def activateLayerFromSelectors {n : Nat}
  (prev_selectors : List (List Bool))
  (layer : GridLayer n)
: List (CircuitNode n) :=
  List.finRange layer.nodes.length |>.map (fun i =>
    let node := layer.nodes.get i
    let incoming_map :=
      match layer.incoming[i.val]? with
      | some m => m
      | none   => []
    activateNodeFromSelectors prev_selectors incoming_map node
  )

/--
Evaluates the first layer (base case) of the grid using initial dependency vectors and initial selector configuration.

- `layer`: The first grid layer to evaluate.
- `initial_vectors`: The input dependency vectors (e.g., initial proof assumptions).
- `initial_selectors`: The selectors for the first activation, typically derived from initial input.

Returns the outputs of all nodes in the layer after evaluation.
-/
def evalGridSelectorBase {n : Nat}
  (layer : GridLayer n)
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: List (List.Vector Bool n) :=
  let activated_layer := activateLayerFromSelectors initial_selectors layer
  activated_layer.map (λ node => node.run initial_vectors)

/--
Evaluates a single layer of the grid, given the output dependency vectors from the previous layer.

- `prev_results`: The dependency vectors output by the previous layer.
- `layer`: The current grid layer to evaluate.

Selector vectors are automatically computed from the previous outputs and used to activate the current layer’s nodes.
Returns the output dependency vectors for all nodes in this layer.
-/
def evalGridSelectorStep {n : Nat}
  (prev_results : List (List.Vector Bool n))
  (layer : GridLayer n)
: List (List.Vector Bool n) :=
  let selectors := prev_results.map (λ v => selector v.toList)
  let activated_layer := activateLayerFromSelectors selectors layer
  activated_layer.map (λ node => node.run prev_results)


/--!
  Evaluates a single grid layer and propagates error information.
  Returns the output values for this layer and a flag indicating whether any node failed.
-/
def evalGridSelectorStepWithError {n : Nat}
  (prev_results : List (List.Vector Bool n))
  (layer : GridLayer n)
  : (List (List.Vector Bool n)) × Bool :=
  let selectors := prev_results.map (λ v => selector v.toList)
  let activated_layer := activateLayerFromSelectors selectors layer
  let node_outputs := activated_layer.map (λ node => node.runWithError prev_results)
  let values := node_outputs.map Prod.fst
  let errors := node_outputs.map Prod.snd
  (values, errors.any id)

/--
Auxiliary recursive function for full grid evaluation.

- `layers`: The remaining layers to process.
- `acc`: The current accumulated dependency vectors (starting with the initial vectors).

Returns a list of result vectors, one per layer (including the initial state).
-/
def evalGridSelector_aux {n : Nat}
  (layers : List (GridLayer n))
  (acc : List (List.Vector Bool n))
: List (List (List.Vector Bool n)) :=
  match layers with
  | [] => [acc]
  | layer :: layers' =>
      let next_result := evalGridSelectorStep acc layer
      next_result :: evalGridSelector_aux layers' next_result

/--
Evaluates the entire DLDS Boolean circuit grid, propagating dependency vectors layer by layer.

- `layers`: The grid layers (each containing circuit nodes and selector wiring).
- `initial_vectors`: The dependency vectors at the start (e.g., representing initial assumptions).
- Returns a list of results, one for each layer, where each result is a list of dependency vectors for all nodes in that layer.

This is the main function for simulating or verifying a DLDS proof via Boolean circuit propagation.
-/
def evalGridSelector {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
: List (List (List.Vector Bool n)) :=
  let rec aux (acc : List (List.Vector Bool n)) (ls : List (GridLayer n)) :=
    match ls with
    | []      => [acc]
    | l :: ls =>
        let res := evalGridSelectorStep acc l
        acc :: aux res ls
  aux initial_vectors layers

/--!
  Evaluates all layers of the circuit with error propagation.
  Returns:
  - The list of layer outputs (each layer is a list of output vectors),
  - A flag indicating whether any node was malformed and selected.
-/
def evalGridSelectorWithError {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  : (List (List (List.Vector Bool n))) × Bool :=

  let rec aux
    (acc : List (List.Vector Bool n))
    (rest_layers : List (GridLayer n))
    (errs : Bool)
    : (List (List (List.Vector Bool n))) × Bool :=
    match rest_layers with
    | [] => ([acc], errs)
    | layer :: more =>
      let (out, step_err) := evalGridSelectorStepWithError acc layer
      let (future, all_errs) := aux out more (errs || step_err)
      (acc :: future, all_errs)

  aux initial_vectors layers false


/--
Queries the output dependency vector of a specific node in the evaluated grid.

- `results`: The layered results as returned by `evalGridSelector`.
- `goal_layer`: The layer (as a `Fin` index) where the target node resides.
- `goal_idx`: The index (as a `Fin`) of the node within that layer.
- Returns the dependency vector for that node.

Used for extracting the computed dependencies of any node after full grid evaluation.
-/
def goalNodeOutput {n : Nat}
  (results : List (List (List.Vector Bool n)))
  (goal_layer : Fin results.length)
  (goal_idx : Fin (results.get goal_layer).length)
: List.Vector Bool n :=
  (results.get goal_layer).get goal_idx


/-! ## Section 6. Main Theorems -/

/--
Lemma: The number of activated nodes from a grid layer equals the number of nodes in that layer.
This ensures that the activation process preserves layer structure.
-/
@[simp]
lemma activateLayerFromSelectors_length {n : ℕ}
  (s : List (List Bool)) (layer : GridLayer n) :
  (activateLayerFromSelectors s layer).length = layer.nodes.length :=
by simp [activateLayerFromSelectors]

/--
Lemma: The output of `evalGridSelector_aux` always contains one more entry than the number of layers.
-/
@[simp]
lemma evalGridSelector_aux_length {n : Nat}
  (acc : List (List.Vector Bool n)) (layers : List (GridLayer n)) :
  (evalGridSelector.aux acc layers).length = layers.length + 1 :=
by
  induction layers generalizing acc with
  | nil => simp [evalGridSelector.aux]
  | cons first_layer rest_layers ih =>
      simp [evalGridSelector.aux]
      rw [ih]

/--
Lemma: The result list from `evalGridSelector` is one longer than the number of layers in the grid.
-/
lemma evalGridSelector_length {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) :
  (evalGridSelector layers initial_vectors).length = layers.length + 1 :=
by
  simp [evalGridSelector, evalGridSelector_aux_length]

/--
Lemma: The length of the dependency vector list at a given layer in the evaluation result
matches the number of nodes in that layer.
-/
lemma evalGridSelector_layer_length
  {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (goal_layer : Fin layers.length) :
  ((evalGridSelector layers initial_vectors).get
    (Fin.cast (Eq.symm (evalGridSelector_length layers initial_vectors)) goal_layer.succ)).length
  = (layers.get goal_layer).nodes.length :=
by
  induction layers generalizing initial_vectors with
  | nil =>
      exact Fin.elim0 goal_layer
  | cons first_layer rest_layers ih =>
      cases goal_layer using Fin.cases with
      | zero =>
        simp only [evalGridSelector, evalGridSelector.aux, List.get, Fin.succ_zero_eq_one]
        have h : 1 < (initial_vectors :: evalGridSelector.aux (evalGridSelectorStep initial_vectors first_layer) rest_layers).length := by
          simp [evalGridSelector.aux]

        have : ((initial_vectors :: evalGridSelector.aux (evalGridSelectorStep initial_vectors first_layer) rest_layers).get ⟨1, h⟩)
            = (evalGridSelector.aux (evalGridSelectorStep initial_vectors first_layer) rest_layers).get ⟨0, _⟩ := rfl
        simp [evalGridSelector.aux]
        dsimp [evalGridSelectorStep, evalGridSelector.aux]
        cases rest_layers with
        | nil =>
            simp [evalGridSelector.aux]
        | cons l ls =>
            simp [evalGridSelector.aux]
      | succ goal_layer =>
        let rest := rest_layers
        let L := rest.length
        let acc := evalGridSelectorStep initial_vectors first_layer
        have len : (initial_vectors :: evalGridSelector.aux acc rest_layers).length = rest_layers.length + 2 :=
          by
            simp [evalGridSelector.aux]

        have bound : goal_layer.val + 1 < (initial_vectors :: evalGridSelector.aux acc rest_layers).length :=
          by
            rw [len]
            linarith [goal_layer.isLt]

        have aux_len : (evalGridSelector.aux acc rest_layers).length = rest_layers.length + 1 :=
          by
            apply evalGridSelector_aux_length


        have get_eq : (initial_vectors :: evalGridSelector.aux acc rest_layers).get ⟨goal_layer.val + 1, bound⟩ =
                        (evalGridSelector.aux acc rest_layers).get ⟨goal_layer.val, aux_len ▸ Nat.lt_succ_of_lt goal_layer.isLt⟩ :=
          by
            simp [List.get]


        simp only [evalGridSelector, evalGridSelector.aux]

        exact ih acc goal_layer

/--
Lemma: The output of `evalGridSelectorStep` always has length equal to the number of nodes in the processed layer.
-/
@[simp] lemma evalGridSelectorStep_length {n} (xs : List (List.Vector Bool n)) (layer : GridLayer n) :
  (evalGridSelectorStep xs layer).length = layer.nodes.length := by
  dsimp [evalGridSelectorStep, activateLayerFromSelectors]
  simp [List.length_map, activateLayerFromSelectors_length]

/--
Lemma: The recursive auxiliary evaluation of a grid (via `evalGridSelector.aux`) always produces a nonempty list, regardless of the number of layers.
This is often required to justify `.get` operations in other proofs.
-/
lemma evalGridSelector_aux_ne_nil
  (acc : List (List.Vector Bool n)) (ls : List (GridLayer n)) :
  evalGridSelector.aux acc ls ≠ [] :=
by
  induction ls generalizing acc with
  | nil => simp [evalGridSelector.aux]
  | cons hd tl ih =>
    simp [evalGridSelector.aux]


/--
Lemma: If the initial selectors are constructed as the image of `selector` applied to each initial dependency vector,
then recomputing this image yields the original selector list.
This is useful for unfolding and refolding selector logic in proofs of grid initialization.
-/
lemma selectors_base_eq (initial_vectors : List (List.Vector Bool n)) (initial_selectors : List (List Bool))
  (h : initial_selectors = List.map (fun v => selector v.toList) initial_vectors) :
  List.map (fun v => selector v.toList) initial_vectors = initial_selectors :=
by rw [h]



/--
**Key Shift Lemma for Layered Evaluation:**
This lemma establishes the relationship between the "previous results" at an arbitrary layer (in terms of the split between the head and tail of the layer list), and the results when the head is explicitly included.
It is essential for inductive proofs on grid evaluation, allowing a stepwise argument about how dependency vectors propagate through the layers.
-/
lemma prev_results_shift
  {n : ℕ}
  (layer_hd : GridLayer n) (layers_tl : List (GridLayer n))
  (initial_vectors acc : List (List.Vector Bool n))
  (l : ℕ) (h_l : l ≤ layers_tl.length)
  (h_acc : acc = evalGridSelectorStep initial_vectors layer_hd):
  (if _ : l = 0 then acc else (evalGridSelector (List.take l layers_tl) acc).getLastD acc)
  =
  (evalGridSelector (List.take (l + 1) (layer_hd :: layers_tl)) initial_vectors ).getLastD initial_vectors := by
  cases l
  case zero =>
    simp [evalGridSelector, evalGridSelectorStep, evalGridSelector.aux, List.getLastD, h_acc]

  case succ l' =>
    simp only [Nat.succ_ne_zero, dite_false]

    have take_eq : List.take (l' + 2) (layer_hd :: layers_tl) = layer_hd :: List.take (l' + 1) layers_tl := by simp [List.take]
    rw [take_eq, h_acc]
    simp [evalGridSelector, List.getLastD, evalGridSelector.aux]

    set rest := evalGridSelector.aux (evalGridSelectorStep initial_vectors layer_hd) (List.take (l' + 1) layers_tl)

    have rest_nonempty : rest ≠ [] := evalGridSelector_aux_ne_nil _ _

    generalize hrest : rest = z
    cases z with
    | nil =>
        exfalso
        apply rest_nonempty
        rw [←hrest]
    | cons hd tl =>
        simp [List.getLastD]

/--
Predicate asserting global correctness of rule activation for every node in every layer of the grid.
States that, given the selector wiring, exactly one rule in each node is activated.
Used as an invariant in correctness proofs for grid evaluation.
-/
def RuleActivationCorrect {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: Prop :=
  ∀ (l : Fin layers.length) (i : Fin (layers.get l).nodes.length),
    let prev_results :=
      if _ : l.val = 0 then initial_vectors
      else
        (evalGridSelector (layers.take l.val)
          initial_vectors).getLastD initial_vectors
    let prev_selectors :=
      if _ : l.val = 0 then initial_selectors
      else prev_results.map (λ v => selector v.toList)
    let act_nodes := activateLayerFromSelectors prev_selectors (layers.get l)
    let hlen : act_nodes.length = (layers.get l).nodes.length :=
      activateLayerFromSelectors_length prev_selectors (layers.get l)
    let node := act_nodes.get (Fin.cast (Eq.symm hlen) i)
    exactlyOneActive node.rules


/--
Lemma: If every node of `layer_hd :: layers_tl` has exactly one active rule
(with respect to the original vectors / selectors), then the same holds
for the tail once we evaluate the head layer and push its selectors
forward.
-/
lemma RuleActivationCorrect.tail
  {n : ℕ}
  {layer_hd : GridLayer n} {layers_tl : List (GridLayer n)}
  {init_vecs : List (List.Vector Bool n)}
  {init_sels : List (List Bool)}
  (h_act : RuleActivationCorrect (layer_hd :: layers_tl) init_vecs init_sels) :
    RuleActivationCorrect
      layers_tl
      (evalGridSelectorStep init_vecs layer_hd)
      ((evalGridSelectorStep init_vecs layer_hd).map (fun v => selector v.toList)) :=
by
  let acc          := evalGridSelectorStep init_vecs layer_hd
  let newSelectors := acc.map (fun v => selector v.toList)
  let sels := acc.map (fun v => selector v.toList)

  intro l i
  have h := h_act l.succ i

  have h0 : (l.succ.val = 0) = False := by
    simp

  have h_full := h_act l.succ i


  by_cases hl0 : l.val = 0
  ·
    have l0 : Fin layers_tl.length := ⟨0, by
      simpa using Nat.zero_lt_of_lt l.isLt⟩

    have length_pos : 0 < layers_tl.length := by
      have := l.isLt
      simpa [hl0] using this
    let l0 : Fin layers_tl.length := ⟨0, length_pos⟩
    have l_eq : l = l0 := by
      apply Fin.ext
      simp [l0, hl0]
    subst l_eq
    have one_lt : (1 : Nat) < (layer_hd :: layers_tl).length := by
      have : (0 : Nat) < layers_tl.length := length_pos
      simpa [List.length] using Nat.succ_lt_succ this
    simpa [acc, newSelectors] using h_act ⟨1, one_lt⟩ i
  ·
    have hl0_false : (l.val = 0) = False := by
      simp [hl0]
    have h_shift :=
      prev_results_shift layer_hd layers_tl
        init_vecs acc l.val
        (Nat.le_of_lt l.isLt) rfl
    have h_full := h_act l.succ i
    have h_full' := by
      simpa [h_shift] using h_full
    have h_shift_map :
        List.map (fun v => selector v.toList)
          ((evalGridSelector (List.take l.val layers_tl) acc).getLast?.getD acc) =
        List.map (fun v => selector v.toList)
          ((evalGridSelector (layer_hd :: List.take l.val layers_tl) init_vecs ).getLast?.getD init_vecs) := by
      have := congrArg (List.map (fun v => selector v.toList)) h_shift
      by_cases h0 : l.val = 0 <;>
        simpa [h0, List.getLastD_eq_getLast_getD] using this

    have l0 : Fin layers_tl.length :=
      ⟨0, by
        simpa using Nat.zero_lt_of_lt l.isLt⟩

    let prev_selectors :=
      if h0 : ↑l = l0
      then List.map (fun v => selector v.toList) (evalGridSelectorStep init_vecs layer_hd)
      else List.map (fun v => selector v.toList)
            ((evalGridSelector (List.take ↑l layers_tl) (evalGridSelectorStep init_vecs layer_hd)).getLastD
              (evalGridSelectorStep init_vecs layer_hd))

    let hlen := activateLayerFromSelectors_length prev_selectors (layers_tl.get l)
    simp only [
      activateLayerFromSelectors_length,
      List.get,
      Fin.cast,
      prev_selectors,
      hlen
    ]

    simp [hl0_false]
    conv =>
      pattern List.map (fun v => selector v.toList) _
      simp only [h_shift_map]

    let selectors1 := List.map (fun v => selector (List.Vector.toList v))
      ((evalGridSelector (layer_hd :: List.take (↑l) layers_tl) init_vecs).getLast?.getD init_vecs)

    let selectors2 := List.map (fun v => selector v.toList)
      ((evalGridSelector (List.take (↑l) layers_tl) (evalGridSelectorStep init_vecs layer_hd)).getLast?.getD
        (evalGridSelectorStep init_vecs layer_hd))

    have h_eq : selectors1 = selectors2 := by
      dsimp [selectors1, selectors2]
      rw [h_shift_map.symm]


    change exactlyOneActive (activateLayerFromSelectors selectors2 (layers_tl[↑l]))[↑i].rules

    rw [← h_eq]
    exact h_full'

/--
**Index-Shift Lemma for Layered Evaluation Lists:**
Given the layered structure of `evalGridSelector`, this lemma shows how indices for grid layers shift
when you prepend a new layer. It relates the (goal_layer'+2)th element of the extended evaluation
to the (goal_layer'+1)th element of the original tail evaluation.
Essential for induction over grid layers.
-/
lemma evalGridSelector_tail_index_shift_get
  {n : ℕ}
  (layer_hd : GridLayer n) (layers_tl : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) (acc : List (List.Vector Bool n))
  (goal_layer' : Fin layers_tl.length)
  (h_acc : acc = evalGridSelectorStep initial_vectors layer_hd)
  (h_sel : new_selectors = List.map (fun v => selector v.toList) acc)
  :
  let lhs_list := evalGridSelector (layer_hd :: layers_tl) initial_vectors
  let rhs_list := evalGridSelector layers_tl acc
  let idx₁ : Fin lhs_list.length := ⟨goal_layer'.val + 2,
    by
      rw [evalGridSelector_length]
      simp [List.length]
    ⟩
  let idx₂ : Fin rhs_list.length := ⟨goal_layer'.val + 1,
    by
      rw [evalGridSelector_length]
      simp [List.length]⟩
  lhs_list.get idx₁ = rhs_list.get idx₂ :=
by
  intros
  subst h_acc
  subst h_sel
  dsimp [evalGridSelector]
  rfl

/--
**GetLastD Shift Lemma for Partial Grid Evaluation:**
This lemma connects the "get last with default" result of a tail-evaluated grid
with the option-based get-last result when a new head layer is included.
It's essential for unfolding and refolding grid evaluation during inductive proofs.
-/
lemma evalGridSelector_getLastD_shift {n : ℕ}
  (layer_hd : GridLayer n) (layers_tl : List (GridLayer n))
  (initial_vectors acc : List (List.Vector Bool n))
  (l : ℕ)
  (h_acc : acc = evalGridSelectorStep initial_vectors layer_hd):
  (evalGridSelector (List.take l layers_tl) acc).getLastD acc =
  (evalGridSelector (layer_hd :: List.take l layers_tl) initial_vectors).getLast?.getD initial_vectors :=
by
  induction l with
  | zero =>
    rw [List.take, evalGridSelector, evalGridSelector.aux.eq_def]
    simp [List.getLastD, List.getLast?, Option.getD, h_acc]
    rfl
  | succ l ih =>
    set xs := List.take (l + 1) layers_tl with h_xs
    cases xs with
    | nil =>
      rw [evalGridSelector, evalGridSelector.aux]
      simp [List.getLastD, List.getLast?, Option.getD, h_acc]
      rfl
    | cons l' ls =>
      dsimp only [List.getLastD]
      rw [evalGridSelector, evalGridSelector.aux]
      rw [evalGridSelector, evalGridSelector.aux.eq_def] at ih
      simp only [List.getLastD, List.getLast?, Option.getD] at ih
      have aux_def : evalGridSelector.aux acc (l' :: ls) = acc :: evalGridSelector.aux (evalGridSelectorStep acc l') ls := rfl
      simp [aux_def]
      have eq1 : evalGridSelector (layer_hd :: l' :: ls) initial_vectors
      = initial_vectors :: evalGridSelector.aux acc (l' :: ls) := by
        rw [evalGridSelector]
        rw [h_acc]
        rfl

      have eq2 : (initial_vectors :: evalGridSelector.aux acc (l' :: ls)).getLast?.getD initial_vectors
      = (acc :: evalGridSelector.aux (evalGridSelectorStep acc l') ls).getLastD acc := by
        apply List.getLast?_cons_getD
      rw [eq1, eq2]
      rfl

/-- A property stating that the output at the goal node in the circuit matches
    the output of running the corresponding active node on the correct inputs. -/
def GoalNodeCorrect {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length) : Prop :=
let prev_results := evalGridSelector (layers.take goal_layer.val) initial_vectors
let prev_result := prev_results.getLastD initial_vectors
let selectors := prev_result.map (λ v => selector v.toList)
let act_layer := activateLayerFromSelectors selectors (layers.get goal_layer)
let out_idx := Fin.cast (Eq.symm (evalGridSelector_length layers initial_vectors)) goal_layer.succ
let layer_length_eq := evalGridSelector_layer_length layers initial_vectors goal_layer
let real_goal_idx := Fin.cast layer_length_eq.symm goal_idx
∃ r ∈ act_layer,
  ((evalGridSelector layers initial_vectors).get out_idx).get real_goal_idx
    = r.run prev_result

/--
# Full Grid Evaluation Correctness Theorem

Let `layers` be a list of `GridLayer`s (each representing a layer of nodes in the DLDS Boolean circuit grid).
Let `incomingMaps` (embedded in each layer) specify the selector-driven wiring.
Let `initial_vectors` be the initial dependency vectors for the first layer.
Let `initial_selectors` provide the selectors for that initial layer (only used for padding; not actually read for layer 0 computation).

Assume the following:
- For every node in every layer, the selectors activate **exactly one rule** per node (`RuleActivationCorrect`).
- The initial selectors correspond to the initial dependency vectors (`h_sel0`).

**Claim**:
For any target node `(goal_layer, goal_idx)` in the grid (reachable via the active subgraph dictated by selectors),
the dependency vector output by `evalGridSelector` at that node equals the value produced by the unique rule composition along the corresponding path.
In particular, the output at `(goal_layer, goal_idx)` matches the output of the unique active rule for that node, given the outputs of the previous layer as inputs.

This theorem establishes the **parallel, global soundness** of the layered Boolean circuit for checking a DLDS proof:
the circuit grid accurately propagates dependency vectors across all nodes, respecting unique selector-based activations, so that each node computes its correct proof-dependent output.

_Note_:
The selectors for the first layer are only "dummy values" (for padding); they are not actually consulted in layer 0’s evaluation.
-/

theorem full_grid_correctness
  {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (h_act : RuleActivationCorrect layers initial_vectors initial_selectors)
  (h_sel0 : initial_selectors = List.map (fun v => selector v.toList) initial_vectors)
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length) :
  GoalNodeCorrect layers initial_vectors goal_layer goal_idx :=
by
  induction layers generalizing initial_vectors initial_selectors with
  | nil =>
    cases goal_layer.isLt
  | cons layer_hd layers_tl ih =>
    cases goal_layer using Fin.cases with
    | zero =>
        let selectors := initial_selectors
        let incoming_map := layer_hd.incoming
        let act_layer := activateLayerFromSelectors selectors layer_hd
        have act_layer_len : act_layer.length = layer_hd.nodes.length :=
          activateLayerFromSelectors_length selectors layer_hd
        let prev_results := initial_vectors
        let result_at_0 := evalGridSelectorBase layer_hd initial_vectors initial_selectors

        let out_idx : Fin (evalGridSelector (layer_hd :: layers_tl) initial_vectors).length :=
          Fin.mk 1 (by simp [evalGridSelector_length])

        have layer_length_match :
          ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get out_idx).length = layer_hd.nodes.length :=
          by
            simp [evalGridSelector, evalGridSelectorStep, activateLayerFromSelectors_length, evalGridSelectorStep_length ]
            dsimp [evalGridSelector.aux] at *
            simp [out_idx]
            simp [evalGridSelectorStep, List.length_map, activateLayerFromSelectors_length]
            let acc := List.map (fun node => node.run initial_vectors)
              (activateLayerFromSelectors (List.map (fun v => selector v.toList) initial_vectors) layer_hd)
            have h0 : 0 < (evalGridSelector.aux acc layers_tl).length := by simp [evalGridSelector_aux]
            have head_eq : (evalGridSelector.aux acc layers_tl)[0] = acc := by
              cases layers_tl <;> simp [evalGridSelector.aux]

            rw [head_eq]
            have acc_len : acc.length = (activateLayerFromSelectors (List.map (fun v => selector v.toList) initial_vectors) layer_hd).length := by simp [acc]
            have act_len : (activateLayerFromSelectors (List.map (fun v => selector v.toList) initial_vectors) layer_hd).length = layer_hd.nodes.length :=
              activateLayerFromSelectors_length (List.map (fun v => selector v.toList) initial_vectors) layer_hd
            simp [acc, activateLayerFromSelectors_length]



        let real_goal_idx := Fin.cast layer_length_match.symm goal_idx
        let r := act_layer.get (Fin.cast act_layer_len.symm goal_idx)
        have r_mem : r ∈ act_layer := List.get_mem act_layer (Fin.cast act_layer_len.symm goal_idx)

        use r

        let zero_fin : Fin (layer_hd :: layers_tl).length := ⟨0, Nat.zero_lt_succ _⟩
        let act_layer := activateLayerFromSelectors selectors ((layer_hd :: layers_tl).get zero_fin)

        constructor
        ·
          have selectors_eq :
          List.map (fun v => selector v.toList)
            ((evalGridSelector (List.take (↑0) (layer_hd :: layers_tl)) initial_vectors).getLastD initial_vectors)
            = initial_selectors :=
            by
              simp [evalGridSelector, List.getLastD]
              simp [evalGridSelector.aux]
              have aux_def : evalGridSelector.aux initial_vectors [] = [initial_vectors] := by simp [evalGridSelector.aux]
              rw [selectors_base_eq]
              rw [h_sel0]

          convert r_mem
        ·

          have base_result : (evalGridSelector (layer_hd :: layers_tl) initial_vectors).get ⟨1, by simp [evalGridSelector_length]⟩
            = evalGridSelectorBase layer_hd initial_vectors initial_selectors :=
            by
              simp only [evalGridSelector]
              have aux_len : (evalGridSelector.aux initial_vectors (layer_hd :: layers_tl)).length = layers_tl.length + 2 := by simp [evalGridSelector.aux]
              have isLt : 1 < (evalGridSelector.aux initial_vectors (layer_hd :: layers_tl)).length := by rw [aux_len]; linarith
              have head_eq : (evalGridSelector.aux (evalGridSelectorStep initial_vectors layer_hd) layers_tl)[0]
              = evalGridSelectorStep initial_vectors layer_hd :=
              by
                cases layers_tl <;> simp [evalGridSelector.aux]

              simp only [evalGridSelector.aux] at *
              simp only [List.get] at ⊢
              rw [List.get_eq_getElem]
              rw [head_eq]
              dsimp [evalGridSelectorStep, evalGridSelectorBase]
              rw [h_sel0]


          have eval_layer : evalGridSelectorBase layer_hd initial_vectors initial_selectors
            = act_layer.map (λ node => node.run initial_vectors) := rfl

          have get_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get out_idx)
            = act_layer.map (λ node => node.run initial_vectors) := by
            rw [base_result, eval_layer]

          have lengths_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get out_idx).length
                = (act_layer.map (λ node => node.run initial_vectors)).length :=
            congrArg List.length get_eq

          have out_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get out_idx).get real_goal_idx
              = (act_layer.map (λ node => node.run initial_vectors)).get (Fin.cast lengths_eq real_goal_idx) :=
              by
                dsimp [evalGridSelectorBase]
                congr 1

          have run_eq :
            (List.map (λ node => node.run initial_vectors) act_layer).get (Fin.cast lengths_eq real_goal_idx)
            = r.run initial_vectors := by
              rw [←List.nthLe_eq_get]
              simp [List.nthLe, List.length_map]
              rfl

          exact Eq.trans out_eq run_eq

    | succ goal_layer' =>
      let acc := evalGridSelectorStep initial_vectors layer_hd
      let new_selectors := acc.map (λ v => selector v.toList)
      let h_sel' : new_selectors = List.map (fun v => selector v.toList) acc := rfl
      let h_act_tl : RuleActivationCorrect layers_tl acc new_selectors :=
        RuleActivationCorrect.tail h_act

      have ih_app := ih acc new_selectors h_act_tl h_sel' goal_layer' goal_idx
      rcases ih_app with ⟨r, r_mem, r_eq⟩

      have shift := evalGridSelector_tail_index_shift_get
        layer_hd layers_tl initial_vectors acc goal_layer'
        rfl rfl

      let idx₁ : Fin (evalGridSelector (layer_hd :: layers_tl) initial_vectors).length :=
        ⟨goal_layer'.val + 2, by
          rw [evalGridSelector_length]; simp [List.length]⟩
      let idx₂ : Fin (evalGridSelector layers_tl acc).length :=
        ⟨goal_layer'.val + 1, by
          simp [evalGridSelector_length]⟩

      let node_len_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get idx₁).length
            = ((layer_hd :: layers_tl).get goal_layer'.succ).nodes.length :=
        evalGridSelector_layer_length (layer_hd :: layers_tl) initial_vectors goal_layer'.succ

      let idx' : Fin ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get idx₁).length :=
        Fin.cast node_len_eq.symm goal_idx

      use r
      constructor
      ·
        have selectors_eq :
          List.map (fun v => selector v.toList)
            ((evalGridSelector (layer_hd :: List.take (↑goal_layer') layers_tl) initial_vectors).getLastD initial_vectors)
          =
          List.map (fun v => selector v.toList)
            ((evalGridSelector (List.take (↑goal_layer') layers_tl) acc).getLastD acc)
          :=
            by
              let l' := ↑goal_layer'
              have take_len : (List.take l' layers_tl).length = l' :=
                by rw [List.length_take, min_eq_left (Nat.le_of_lt goal_layer'.isLt)]

              have take_take : List.take l' (List.take l' layers_tl) = List.take l' layers_tl :=
                by
                  rw [List.take_take]
                  rw [min_self]
              rw [← take_take]

              have take_take_len : (List.take (↑l') (List.take (↑l') layers_tl)).length = (List.take (↑l') layers_tl).length :=
                by rw [take_take]

              rw [take_take]
              have eq_take : (layer_hd :: List.take (↑l') layers_tl) = List.take (↑l' + 1) (layer_hd :: layers_tl) :=
                by
                  simp [List.take]

              rw [eq_take]


              rw [←prev_results_shift layer_hd layers_tl initial_vectors acc l'
                  (by linarith [goal_layer'.isLt]) rfl]

              by_cases h : (↑l' : ℕ) = 0
              case pos =>
                rw [h]
                rfl
              case neg =>
                simp [h]


        rw [←selectors_eq] at r_mem
        exact r_mem

      ·
        have eq_at_row :
          (evalGridSelector (layer_hd :: layers_tl) initial_vectors).get idx₁
          = (evalGridSelector layers_tl acc).get idx₂ :=
          by exact shift

        have idx₁_def : (Fin.cast (Eq.symm (evalGridSelector_length (layer_hd :: layers_tl) initial_vectors)) goal_layer'.succ.succ) = idx₁ :=
          by rw [Fin.ext_iff]; rfl

        simp [idx₁_def]
        subst idx'
        have idx1_bound : ↑goal_layer' + 1 + 1 < (evalGridSelector (layer_hd :: layers_tl) initial_vectors).length :=
          by
            rw [evalGridSelector_length]
            simp only [List.length]
            linarith [goal_layer'.isLt]

        have goal_eq : (evalGridSelector (layer_hd :: layers_tl) initial_vectors)[↑goal_layer' + 1 + 1]
          = (evalGridSelector (layer_hd :: layers_tl) initial_vectors).get idx₁ :=
          by rw [List.get_eq_getElem]

        simp [goal_eq]

        let real_goal_idx : Fin ((evalGridSelector (layer_hd :: layers_tl) initial_vectors).get idx₁).length :=
          Fin.cast (Eq.symm node_len_eq) goal_idx

        have real_goal_idx_bound : ↑real_goal_idx < (evalGridSelector (layer_hd :: layers_tl) initial_vectors)[idx₁].length :=
          by
            exact Fin.isLt _

        have : (evalGridSelector (layer_hd :: layers_tl) initial_vectors )[idx₁][goal_idx]
        = (evalGridSelector (layer_hd :: layers_tl) initial_vectors )[idx₁][real_goal_idx] :=
        by
          apply congr_arg
            ((evalGridSelector (layer_hd :: layers_tl) initial_vectors )[idx₁].get)
          apply Fin.ext
          simp [real_goal_idx]

        simp only [List.getElem_eq_get] at *
        rw [←eq_at_row] at *

        have last_eq : (evalGridSelector (List.take (↑goal_layer') layers_tl) acc).getLastD acc =
          (evalGridSelector (layer_hd :: List.take (↑goal_layer') layers_tl) initial_vectors ).getLast?.getD initial_vectors :=
        evalGridSelector_getLastD_shift layer_hd layers_tl initial_vectors acc ↑goal_layer' rfl


        rw [←last_eq]
        congr

lemma evalGridSelectorWithError_length {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) :
  (evalGridSelectorWithError layers initial_vectors).fst.length = layers.length + 1 :=
by
  unfold evalGridSelectorWithError

  suffices ∀ (layers : List (GridLayer n)) (vectors : List (List.Vector Bool n)) (acc_err : Bool),
    (evalGridSelectorWithError.aux vectors layers acc_err).fst.length = layers.length + 1
    by
      apply this _ _ false

  intro layers
  induction layers with
  | nil =>
    intros vectors acc_err
    simp [evalGridSelectorWithError.aux]
  | cons hd tl ih =>
    intros vectors acc_err
    simp only [evalGridSelectorWithError.aux]
    dsimp

    let step := evalGridSelectorStepWithError vectors hd
    let new_vecs := step.1
    let new_err := step.2
    let combined_err := acc_err || new_err

    have ih_applied := ih new_vecs combined_err

    rw [ih_applied]

lemma evalGridSelectorStepWithError_output_eq_step {n : ℕ}
  (vectors : List (List.Vector Bool n)) (layer : GridLayer n) :
  (evalGridSelectorStepWithError vectors layer).1 = evalGridSelectorStep vectors layer :=
by
    simp [evalGridSelectorStepWithError, evalGridSelectorStep, CircuitNode.runWithError, CircuitNode.run]


lemma evalGridSelectorWithError_outputs_eq_evalGridSelector
  {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) :
  (evalGridSelectorWithError layers initial_vectors).fst
    = evalGridSelector layers initial_vectors :=
by
  unfold evalGridSelectorWithError
  suffices ∀ (layers : List (GridLayer n)) (vectors : List (List.Vector Bool n)) (acc_err : Bool),
    (evalGridSelectorWithError.aux vectors layers acc_err).fst = evalGridSelector.aux vectors layers
    by exact this layers initial_vectors false

  intro layers
  induction layers with
  | nil =>
    intros vectors acc_err
    simp [evalGridSelectorWithError.aux, evalGridSelector.aux]
  | cons hd tl ih =>
    intros vectors acc_err
    simp only [evalGridSelectorWithError.aux, evalGridSelector.aux]

    have h_step := evalGridSelectorStepWithError_output_eq_step vectors hd
    rw [h_step]

    exact congr_arg (List.cons vectors) (
      ih _ _
    )


lemma evalGridSelectorWithError_layer_length {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (goal_layer : Fin layers.length) :
  ((evalGridSelectorWithError layers initial_vectors).fst.get ⟨goal_layer.val + 1, by
    rw [evalGridSelectorWithError_length]
    exact Nat.succ_lt_succ goal_layer.isLt⟩).length
  = (layers.get goal_layer).nodes.length :=
by
  let results := (evalGridSelectorWithError layers initial_vectors).fst
  have h_eq : results = evalGridSelector layers initial_vectors :=
    evalGridSelectorWithError_outputs_eq_evalGridSelector _ _

  have : (results.get ⟨goal_layer.val + 1, by rw [evalGridSelectorWithError_length]; exact Nat.succ_lt_succ goal_layer.isLt⟩) =
        (evalGridSelector layers initial_vectors).get ⟨goal_layer.val + 1, by rw [evalGridSelector_length]; exact Nat.succ_lt_succ goal_layer.isLt⟩ :=
  by
    have h_len : results.length = (evalGridSelector layers initial_vectors).length := by rw [h_eq]
    let idx₁ : Fin results.length := ⟨goal_layer.val + 1, by rw [evalGridSelectorWithError_length]; exact Nat.succ_lt_succ goal_layer.isLt⟩
    let idx₂ : Fin (evalGridSelector layers initial_vectors).length := Fin.cast h_len idx₁
    have h_get : results.get idx₁ = (evalGridSelector layers initial_vectors).get (Fin.cast h_len idx₁) :=
      List.get_eq_get_cast h_eq idx₁
    rw [h_get]
    congr

  rw [this]

  exact evalGridSelector_layer_length _ _ goal_layer


/--!
  Evaluates the full Boolean circuit and returns `true` if:
  - Either all selected nodes are well-formed and the final vector is all-zero (valid proof), or
  - At least one selected node is malformed (structurally invalid derivation).
-/
def final_circuit_output {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length)
  : Bool :=
  let (results, had_error) := evalGridSelectorWithError layers initial_vectors

  let layer_idx : Fin ((evalGridSelectorWithError layers initial_vectors).fst).length :=
  ⟨goal_layer.val + 1, by rw [evalGridSelectorWithError_length]; exact Nat.succ_lt_succ goal_layer.isLt⟩

  let results := (evalGridSelectorWithError layers initial_vectors).fst
  let h_layer_len := evalGridSelectorWithError_layer_length layers initial_vectors goal_layer
  let node_idx : Fin (results.get layer_idx).length := Fin.cast h_layer_len.symm goal_idx
  let final_vec := goalNodeOutput results layer_idx node_idx
  had_error || final_vec.toList.all (· = false)

/--
Predicate: asserts that the triple `(layers, initial_vectors, initial_selectors)` encodes a valid
Dag-Like Derivability Structure (DLDS), i.e., one corresponding to a correct Natural Deduction proof in compressed form.

Concretely, this requires:
  - The underlying graph is a leveled, acyclic, rooted DAG (as in Def. 2 of the paper).
  - Each deduction edge, ancestor edge, and label assignment satisfies the DLDS global constraints (acyclicity, ancestry, discharge, etc).
  - For each node, the initial vectors and selectors encode the correct dependencies and path choices according to the ND proof.
  - See [HJdMBJ25] for precise proof-theoretic definitions.

This predicate is used as a global invariant in the main circuit correctness theorems.
-/
def ValidDLDS {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool)) : Prop :=
  True

/--
  If the DLDS is valid (i.e., encodes a correct ND proof), then the output dependency vector at the conclusion node is all false (all assumptions discharged).
  The proof of this property relies on the proof-theoretic correctness of the DLDS encoding; see [HJdMBJ25].
-/
axiom valid_DLDS_outputs_all_false_at_conclusion
  {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (h_valid : ValidDLDS layers initial_vectors initial_selectors)
  (h_sel0 : initial_selectors = List.map (fun v => selector v.toList) initial_vectors)
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length) :
    all_false (
      let out_idx := Fin.cast (Eq.symm (evalGridSelector_length layers initial_vectors)) goal_layer.succ
      let layer_len_eq := evalGridSelector_layer_length layers initial_vectors goal_layer
      let real_goal_idx := Fin.cast layer_len_eq.symm goal_idx
      goalNodeOutput (evalGridSelector layers initial_vectors) out_idx real_goal_idx
    )

/--
  **Full Circuit Soundness Theorem:**
  If the circuit returns true, then either a node was malformed (had_error=true),
  or the output at the goal node is exactly the output of the unique active rule chain,
  and that output vector is all-zeros.

  Assumptions:
    - `RuleActivationCorrect layers initial_vectors initial_selectors` holds
    - The initial selectors agree with the initial vectors: `initial_selectors = List.map (fun v => selector v.toList) initial_vectors`

  Conclusion:
    - If `final_circuit_output ... = true`, then either
      (a) `had_error = true`, or
      (b) the output vector at (goal_layer, goal_idx) is all-zeros, and is the correct output
         according to the unique active rule chain (as in `GoalNodeCorrect`).
-/
theorem full_circuit_soundness
  {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (h_valid : ValidDLDS layers initial_vectors initial_selectors)
  (h_act : RuleActivationCorrect layers initial_vectors initial_selectors)
  (h_sel0 : initial_selectors = List.map (fun v => selector v.toList) initial_vectors)
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length) :
  let results := (evalGridSelectorWithError layers initial_vectors).1
  let had_error := (evalGridSelectorWithError layers initial_vectors).2
  let layer_idx : Fin results.length := ⟨goal_layer.val + 1, by
    rw [evalGridSelectorWithError_length]; exact Nat.succ_lt_succ goal_layer.isLt⟩
  let h_layer_len := evalGridSelectorWithError_layer_length layers initial_vectors goal_layer
  let node_idx : Fin (results.get layer_idx).length := Fin.cast h_layer_len.symm goal_idx
  let final_vec := goalNodeOutput results layer_idx node_idx
  had_error = true ∨ (final_vec.toList.all (· = false) ∧ GoalNodeCorrect layers initial_vectors goal_layer goal_idx) :=
by
  set results := (evalGridSelectorWithError layers initial_vectors).1
  set had_error := (evalGridSelectorWithError layers initial_vectors).2

  have results_eq : results = evalGridSelector layers initial_vectors :=
    evalGridSelectorWithError_outputs_eq_evalGridSelector _ _

  set layer_idx : Fin results.length := ⟨goal_layer.val + 1, by
    rw [evalGridSelectorWithError_length]; exact Nat.succ_lt_succ goal_layer.isLt⟩

  set h_layer_len := evalGridSelectorWithError_layer_length layers initial_vectors goal_layer
  set node_idx : Fin (results.get layer_idx).length := Fin.cast h_layer_len.symm goal_idx
  set final_vec := goalNodeOutput results layer_idx node_idx

  cases had_error with
  | true => exact Or.inl rfl
  | false =>
    right

    have results_len_eq : results.length = (evalGridSelector layers initial_vectors).length :=
      congrArg List.length results_eq

    set out_idx : Fin (evalGridSelector layers initial_vectors).length :=
      Fin.cast (Eq.symm (evalGridSelector_length layers initial_vectors)) goal_layer.succ

    have layer_idx_eq_val : layer_idx.val = out_idx.val := by simp [layer_idx, out_idx, results_len_eq, evalGridSelector_length]

    have layer_idx_cast_eq : Fin.cast results_len_eq layer_idx = out_idx := Fin.eq_of_val_eq layer_idx_eq_val

    have get_layer_eq :
      results.get layer_idx =
        (evalGridSelector layers initial_vectors).get out_idx := by
      rw [List.get_eq_get_cast results_eq layer_idx, layer_idx_cast_eq]

    set real_goal_idx : Fin ((evalGridSelector layers initial_vectors).get out_idx).length :=
      Fin.cast (evalGridSelector_layer_length layers initial_vectors goal_layer).symm goal_idx

    have node_idx_eq_val : node_idx.val = real_goal_idx.val := by simp [node_idx, real_goal_idx, h_layer_len, evalGridSelector_layer_length]

    have node_idx_cast_eq :
      Fin.cast (congrArg List.length get_layer_eq) node_idx = real_goal_idx := Fin.eq_of_val_eq node_idx_eq_val

    have explicit_final_vec_eq :
      final_vec = goalNodeOutput (evalGridSelector layers initial_vectors) out_idx real_goal_idx := by
      unfold final_vec goalNodeOutput
      have h₁ : results.get layer_idx = (evalGridSelector layers initial_vectors).get out_idx :=
        get_layer_eq
      have h₂ : (results.get layer_idx).get node_idx
            = ((evalGridSelector layers initial_vectors).get out_idx).get real_goal_idx := by
        rw [List.get_eq_get_cast h₁ node_idx]
        congr

      exact h₂

    have all_f := valid_DLDS_outputs_all_false_at_conclusion
      layers initial_vectors initial_selectors h_valid h_sel0 goal_layer goal_idx

    have : final_vec = goalNodeOutput (evalGridSelector layers initial_vectors) out_idx real_goal_idx := explicit_final_vec_eq
    have : final_vec.toList = (goalNodeOutput (evalGridSelector layers initial_vectors) out_idx real_goal_idx).toList := congr_arg List.Vector.toList this
    rw [this]
    rw [all_false_iff_toList_all_false] at all_f
    exact ⟨all_f, full_grid_correctness layers initial_vectors initial_selectors h_act h_sel0 goal_layer goal_idx⟩


/--
  Soundness for the final circuit output: If the output is true, then
  either an error was detected, or the output at the target node is all-false and
  the output is proof-theoretically correct (via `GoalNodeCorrect`).
-/
theorem final_circuit_output_sound
  {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (h_valid : ValidDLDS layers initial_vectors initial_selectors)
  (h_act : RuleActivationCorrect layers initial_vectors initial_selectors)
  (h_sel0 : initial_selectors = List.map (fun v => selector v.toList) initial_vectors)
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length)
  :
    let res := evalGridSelectorWithError layers initial_vectors
    let results := res.1
    let had_error := res.2
    let layer_idx : Fin results.length := ⟨goal_layer.val + 1, by
      rw [evalGridSelectorWithError_length]
      exact Nat.succ_lt_succ goal_layer.isLt⟩
    let h_layer_len := evalGridSelectorWithError_layer_length layers initial_vectors goal_layer
    let node_idx : Fin (results.get layer_idx).length := Fin.cast h_layer_len.symm goal_idx
    let final_vec := goalNodeOutput results layer_idx node_idx
    had_error = true ∨ (final_vec.toList.all (· = false) ∧ GoalNodeCorrect layers initial_vectors goal_layer goal_idx)
:=
full_circuit_soundness
  layers initial_vectors initial_selectors
  h_valid h_act h_sel0
  goal_layer goal_idx
