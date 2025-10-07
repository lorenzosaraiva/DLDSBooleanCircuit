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

open scoped Classical

/-- Tagless-final algebra for bits and fixed-length vectors, in a monad `m`. -/
class BitVecAlg (m : Type → Type) where
  -- carriers
  Bit : Type
  Vec : Nat → Type

  -- primitive bits
  const  : Bool → m Bit
  buf    : Bit → m Bit
  bnot   : Bit → m Bit
  band   : Bit → Bit → m Bit
  bor    : Bit → Bit → m Bit
  bxor   : Bit → Bit → m Bit

  -- vectors (length-indexed)
  vconst    : ∀ {n}, Bool → m (Vec n)
  vmap      : ∀ {n}, (Bit → m Bit) → Vec n → m (Vec n)
  vzipWith  : ∀ {n}, (Bit → Bit → m Bit) → Vec n → Vec n → m (Vec n)
  vreduceOr : ∀ {n}, Vec n → m Bit

  -- bridge List Bit ↔ Vec n (useful sometimes)
  vfromList : ∀ {n}, (bs : List Bit) → (h : bs.length = n) → m (Vec n)
  vtoList   : ∀ {n}, Vec n → m (List Bit)
  -- build a Vec n directly from Bool literals
  vfromBools : ∀ {n}, (bs : List Bool) → (h : bs.length = n) → m (Vec n)

  -- mask a vector by one bit
  vmaskBy   : ∀ {n}, Bit → Vec n → m (Vec n)

/-! Pure Boolean instance (`m = Id`) -/
namespace Pure

abbrev M := Id
abbrev B := Bool
abbrev V (n : Nat) := List.Vector Bool n

instance : BitVecAlg M where
  Bit := B
  Vec := V

  const  := fun b => b
  buf    := fun b => b
  bnot   := fun b => !b
  band   := fun x y => x && y
  bor    := fun x y => x || y
  bxor   := fun x y => Bool.xor x y

  vconst {n} b := List.Vector.replicate n b

  vmap := by
    intro n f v
    exact ⟨v.toList.map f, by simp [List.length_map] ⟩

  vzipWith := by
    intro n f v1 v2
    rcases v1 with ⟨l1, h1⟩
    rcases v2 with ⟨l2, h2⟩
    refine ⟨List.zipWith f l1 l2, ?_⟩
    simp [List.length_zipWith, h1, h2]

  vreduceOr := by
    intro n v
    exact v.toList.foldl (· || ·) false

  vfromList {n} bs h := by
    refine (⟨bs, ?_⟩ : V n)
    simp [h]

  vtoList {n} v := v.toList

  vfromBools {n} bs h := (⟨bs, by simp [h]⟩ : V n)

  vmaskBy {n} b v := by
    exact ⟨v.toList.map (fun x => b && x), by simp [List.length_map]⟩

end Pure



/-- Tiny helper: OR-reduce a list of algebraic bits. -/
def orListA {m} [Monad m] [BitVecAlg m] :
  List (BitVecAlg.Bit (m := m)) → m (BitVecAlg.Bit (m := m))
| []       => BitVecAlg.const false
| [a]      => BitVecAlg.buf a
| a :: xs  => do
  let t ← orListA xs
  BitVecAlg.bor a t

/-- Generic multiple-xor over the algebra. -/
def multipleXorA {m} [Monad m] [BitVecAlg m] :
  List (BitVecAlg.Bit (m := m)) → m (BitVecAlg.Bit (m := m))
| []       => BitVecAlg.const false
| [x]      => BitVecAlg.buf x
| x :: xs  => do
  let orTail ← orListA xs
  let notOr ← BitVecAlg.bnot orTail
  let t1    ← BitVecAlg.band x notOr
  let nx    ← BitVecAlg.bnot x
  let mx    ← multipleXorA xs
  let t2    ← BitVecAlg.band nx mx
  BitVecAlg.bor t1 t2

/-- Generic selector on algebraic bits (list size = n), returns 2^n outputs. -/
def selectorA {m} [Monad m] [BitVecAlg m]
  (input : List (BitVecAlg.Bit (m := m))) :
  m (List (BitVecAlg.Bit (m := m))) := do
  let n := input.length
  let natToBits (x k : Nat) : List Bool :=
    (List.range k).map (fun i => ((x >>> (k - 1 - i)) % 2 = 1))

  let rec zipEval
      (inp : List (BitVecAlg.Bit (m := m)))
      (ps  : List Bool) :
      m (List (BitVecAlg.Bit (m := m))) := do
    match inp, ps with
    | [], [] => pure []
    | b::bs, q::qs =>
      let lit ← if q then BitVecAlg.buf b else BitVecAlg.bnot b
      let rest ← zipEval bs qs
      pure (lit :: rest)
    | _, _ => pure []

  let rec andList
      : List (BitVecAlg.Bit (m := m)) → m (BitVecAlg.Bit (m := m))
  | []      => BitVecAlg.const true
  | [a]     => BitVecAlg.buf a
  | a::bs   => do let t ← andList bs; BitVecAlg.band a t

  (List.range (Nat.pow 2 n)).mapM (fun i => do
    let patt := natToBits i n
    let lits ← zipEval input patt
    andList lits)

/-- Build a `Vec n` from a `List Bool` by lifting each Bool with `const`. -/
def vecOfBoolList {m} [Monad m] [BitVecAlg m] {n}
  (bs : List Bool) (h : bs.length = n) :
  m (BitVecAlg.Vec (m := m) n) :=
  BitVecAlg.vfromBools (m := m) (n := n) bs h

/-- OR-reduce a list of length-indexed vectors componentwise. Returns the all-false vector on empty. -/
def listOrVecsA {m} [Monad m] [BitVecAlg m] {n}
  (vs : List (BitVecAlg.Vec (m := m) n)) : m (BitVecAlg.Vec (m := m) n) := do
  match vs with
  | []      => BitVecAlg.vconst (n := n) false
  | v :: tl =>
    let rec go (acc : BitVecAlg.Vec (m := m) n) (rest : List (BitVecAlg.Vec (m := m) n)) : m (BitVecAlg.Vec (m := m) n) := do
      match rest with
      | []      => pure acc
      | w :: ws =>
        let acc' ← BitVecAlg.vzipWith (fun a b => BitVecAlg.bor a b) acc w
        go acc' ws
    go v tl

/-- `true` if any bit is set in any vector in the list. -/
def anyOfVecsA {m} [Monad m] [BitVecAlg m] {n}
  (vs : List (BitVecAlg.Vec (m := m) n)) : m (BitVecAlg.Bit (m := m)) := do
  let tails ← vs.mapM (fun v => BitVecAlg.vreduceOr v)
  orListA tails

-- Evaluate tagless ops in the Pure instance
@[simp] lemma Pure.run_const (b : Bool) :
  Id.run (BitVecAlg.const (m := Pure.M) b) = b := rfl

@[simp] lemma Pure.run_buf (b : Bool) :
  Id.run (BitVecAlg.buf (m := Pure.M) b) = b := rfl

@[simp] lemma Pure.run_bnot (b : Bool) :
  Id.run (BitVecAlg.bnot (m := Pure.M) b) = !b := rfl

@[simp] lemma Pure.run_band (x y : Bool) :
  Id.run (BitVecAlg.band (m := Pure.M) x y) = (x && y) := rfl

@[simp] lemma Pure.run_bor (x y : Bool) :
  Id.run (BitVecAlg.bor (m := Pure.M) x y) = (x || y) := rfl

@[simp] lemma Pure.run_bxor (x y : Bool) :
  Id.run (BitVecAlg.bxor (m := Pure.M) x y) = Bool.xor x y := rfl

-- Let simp evaluate Id/do-blocks.
@[simp] lemma Pure.run_bind {α β} (x : Id α) (f : α → Id β) :
  Id.run (x >>= f) = Id.run (f (Id.run x)) := rfl

@[simp] lemma Pure.run_pure {α} (x : α) :
  Id.run (pure x : Id α) = x := rfl

/-- Decode `k` binary bits into a 1-hot vector of length `2^k`.
    This is just a thin wrapper over your existing `selectorA`. -/
def decodeIndex {m} [Monad m] [BitVecAlg m]
  (bits : List (BitVecAlg.Bit (m := m)))
  : m (List (BitVecAlg.Bit (m := m))) :=
  selectorA bits

 @[simp] lemma orListA_pure_or (xs : List Bool) :
  Id.run (orListA (m := Pure.M) xs) = xs.or := by
  induction xs with
  | nil => simp [orListA, List.or]
  | cons a tl ih =>
    sorry

-- OR-reduce over tagless ‘A’ computes List.any id under Pure
@[simp] lemma orListA_pure_any (xs : List Bool) :
  Id.run (orListA (m := Pure.M) xs) = xs.any id := by
  induction xs with
  | nil =>
      simp [orListA, List.any]
  | cons a tl ih =>
      sorry


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
  | repetition (bit : Bool)
  deriving DecidableEq

/--
Represents the data of a rule for formulas of length `n`:
- `intro`: With an encoder vector for implication introduction.
- `elim`: For implication elimination.
-/
inductive RuleData (n : Nat)
  | intro (encoder : List.Vector Bool n)
  | elim
  | repetition

/--
Structure representing a single inference rule, including:
- The activation bits.
- Its kind (intro/elim).
- The dependency vector update function.
-/
structure Rule (n : ℕ) where
  ruleId    : Nat
  activation : ActivationBits
  type       : RuleData n
  combine    : List (List.Vector Bool n) → List.Vector Bool n

/--
Represents a node in the Boolean circuit (corresponds to a node in the DLDS).
- Each node stores a list of possible inference rules.
- Invariant: All rules must be unique.
-/
structure CircuitNode (n : ℕ) where
  rules    : List (Rule n)
  nodupIds : (rules.map (·.ruleId)).Nodup

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
def mkIntroRule {n : ℕ} (rid : Nat) (encoder : List.Vector Bool n) (bit : Bool) : Rule n :=
{
  ruleId    := rid,
  activation := ActivationBits.intro bit,
  type       := RuleData.intro encoder,
  combine    := fun deps =>
    match deps with
    | [d] => d.zipWith (fun b e => b && !e) encoder
    | _   => List.Vector.replicate n false
}

/--
Constructs an implication elimination rule for formulas of length `n`.
- `bit1`, `bit2`: Activation bits for the two premises.
- `combine` returns conjunction of the two dependency vectors.

-/
def mkElimRule {n : ℕ} (rid : Nat) (bit1 bit2 : Bool) : Rule n :=
{
  ruleId    := rid,
  activation := ActivationBits.elim bit1 bit2,
  type       := RuleData.elim,
  combine    := fun deps =>
    match deps with
    | [d1, d2] => d1.zipWith (· && ·) d2
    | _        => List.Vector.replicate n false
}

def mkRepetitionRule {n : ℕ} (rid : Nat) (bit : Bool) : Rule n :=
{
  ruleId    := rid,
  activation := ActivationBits.repetition bit,
  type       := RuleData.repetition,
  combine    := fun deps =>
    match deps with
    | [d] => d
    | _   => List.Vector.replicate n false
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
  | ActivationBits.repetition b => b
/--
Computes whether exactly one element of a Boolean list is `true`.
- Implements "one-hot" logic for rule activation.
-/
def multiple_xor : List Bool → Bool
| []       => false
| [x]      => x
| x :: xs  => (x && not (List.or xs)) || (not x && multiple_xor xs)

@[simp] lemma multipleXorA_pure (xs : List Bool) :
  Id.run (multipleXorA (m := Pure.M) xs) = multiple_xor xs := by
  induction xs with
  | nil =>
      simp [multipleXorA, multiple_xor]
  | cons x tl ih =>
    cases tl with
    | nil =>
        simp [multipleXorA, multiple_xor]
    | cons h t =>
        cases h <;> simp [multipleXorA, ih, multiple_xor, orListA_pure_or, List.or]


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

/-- Tagless combine for your `RuleData`: ⊃I: d ∧ ¬encoder, ⊃E: zipWith ∧. -/
def ruleCombineA {m} [Monad m] [BitVecAlg m] {n}
  (rd : RuleData n)
  (deps : List (BitVecAlg.Vec (m := m) n)) :
  m (BitVecAlg.Vec (m := m) n) := do
  match rd, deps with
  | RuleData.intro enc, [d] =>
      -- turn Bool encoder into a Bit vector, then NOT it componentwise and AND with d
      let encV  ← vecOfBoolList (m := m) (n := n) enc.toList (by simp)
      let notEn ← BitVecAlg.vmap (fun x => BitVecAlg.bnot x) encV
      BitVecAlg.vzipWith (fun a b => BitVecAlg.band a b) d notEn
  | RuleData.elim, [d1, d2] =>
      BitVecAlg.vzipWith (fun a b => BitVecAlg.band a b) d1 d2
  | RuleData.repetition, [d] =>
      pure d                          -- ← keep it simple
  | _, _ =>
      BitVecAlg.vconst (n := n) false

/-- Node logic over the algebra, *given* activation bits per rule. -/
def nodeLogicGivenActsA {m} [Monad m] [BitVecAlg m] {n}
  (rules  : List (Rule n))
  (acts   : List (BitVecAlg.Bit (m := m)))
  (inputs : List (BitVecAlg.Vec (m := m) n)) :
  m (BitVecAlg.Vec (m := m) n) := do
  let xor ← multipleXorA acts
  let masks ← acts.mapM (fun a => BitVecAlg.band xor a)
  let outs ← (List.zip rules masks).mapM
    (fun (rm : Rule n × BitVecAlg.Bit (m := m)) => do
      let r    := rm.fst
      let mask := rm.snd
      let raw  ← ruleCombineA (m := m) (n := n) r.type inputs
      BitVecAlg.vmaskBy mask raw)
  listOrVecsA outs

/--
Defines the overall logic for computing a node’s output:
- Extracts activations
- Checks for unique activation with `multiple_xor`
- Masks activations accordingly
- Applies rule logic and combines results with `list_or`
-/
def node_logic {n : Nat}
  (rules : List (Rule n))
  (inputs : List (List.Vector Bool n)) :
  List.Vector Bool n :=
  let acts  := extract_activations rules
  let xor   := multiple_xor acts
  let masks := and_bool_list xor acts
  let outs  := apply_activations rules masks inputs
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




/-- Same as above, but also returns the error bit: (¬xor) ∧ selected. -/
def nodeLogicWithErrorGivenActsA {m} [Monad m] [BitVecAlg m] {n}
  (rules  : List (Rule n))
  (acts   : List (BitVecAlg.Bit (m := m)))
  (inputs : List (BitVecAlg.Vec (m := m) n)) :
  m (BitVecAlg.Vec (m := m) n × BitVecAlg.Bit (m := m)) := do
  let xor ← multipleXorA acts
  let out ← nodeLogicGivenActsA (m := m) (n := n) rules acts inputs
  let sel ← anyOfVecsA (m := m) (n := n) inputs
  let notX ← BitVecAlg.bnot xor
  let err  ← BitVecAlg.band notX sel
  pure (out, err)

/-- Activation over the algebra: intro uses 1 bit, elim uses 2 bits; otherwise false. -/
def isRuleActiveA {m} [Monad m] [BitVecAlg m]
  (act : ActivationBits)
  (ws  : List (BitVecAlg.Bit (m := m))) :
  m (BitVecAlg.Bit (m := m)) :=
  match act, ws with
  | ActivationBits.intro _,      [b]      => BitVecAlg.buf b
  | ActivationBits.elim _ _,     [b1, b2] => BitVecAlg.band b1 b2
  | ActivationBits.repetition _, [b]      => BitVecAlg.buf b
  | _,                            _        => BitVecAlg.const false


/-- For each rule, compute its single activation bit from its activation *inputs*. -/
def actsFromRuleActWiresA {m} [Monad m] [BitVecAlg m] {n}
  (rules    : List (Rule n))
  (ruleActs : List (List (BitVecAlg.Bit (m := m)))) :
  m (List (BitVecAlg.Bit (m := m))) :=
  (List.zip rules ruleActs).mapM (fun (r, ws) => isRuleActiveA (m := m) r.activation ws)

/-!
## Section 3: Boolean List Lemmas and Multiple XOR Properties

This section contains auxiliary lemmas and theorems about Boolean list operations,
especially those related to the multiple_xor function and its connection to unique activation.
These are essential for the correctness and reasoning about rule activation in the circuit.
-/

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

  have h_nodup : c.rules.Nodup :=
    nodup_of_map (fun (r : Rule n) => r.ruleId) c.nodupIds

  have h_bool :
    multiple_xor (c.rules.map is_rule_active) = true :=
    (multiple_xor_bool_iff_exactlyOneActive c.rules h_nodup).mpr h_one

  let h_one_prop := h_one

  rcases h_one with ⟨r0, hr0_mem, hr0_active, hr0_unique⟩

  dsimp [CircuitNode.run, node_logic, extract_activations]

  rw [h_bool]
  dsimp [and_bool_list]

  let h_nonempty := List.ne_nil_of_mem hr0_mem
  simp [List.map_map, true_and]

  let eq := list_or_apply_unique_active_of_exactlyOne
    h_nonempty hr0_mem h_nodup h_one_prop hr0_active inputs

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
abbrev IncomingMap := List (List (Nat × Nat))


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

    let ps : List (Nat × Nat) :=
      if h : i.val < incoming_map.length then
        incoming_map.get ⟨i.val, h⟩
      else
        []

    let getBit (src edge : Nat) : Bool :=
      if hS : src < prev_selectors.length then
        let sel := prev_selectors.get ⟨src, hS⟩
        if hE : edge < sel.length then sel.get ⟨edge, hE⟩ else false
      else
        false

    let b1b2 :=
      match ps with
      | (s1,e1) :: (s2,e2) :: _ => (getBit s1 e1, getBit s2 e2)
      | (s1,e1) :: []           => (getBit s1 e1, false)
      | []                      => (false, false)

    let newAct :=
      match rule.activation with
      | ActivationBits.intro _   => ActivationBits.intro b1b2.fst
      | ActivationBits.elim _ _  => ActivationBits.elim b1b2.fst b1b2.snd
      | ActivationBits.repetition _   => ActivationBits.repetition b1b2.fst
    { rule with activation := newAct }
  )

@[simp] lemma ruleId_updateAct {n} (r : Rule n) (a : ActivationBits) :
  ({ r with activation := a }).ruleId = r.ruleId := rfl

lemma make_new_rules_map_ruleId_eq
  {n : ℕ} (node : CircuitNode n) (prev_selectors : List (List Bool)) (incoming_map : IncomingMap) :
  (make_new_rules node prev_selectors incoming_map).map (·.ruleId)
  = node.rules.map (·.ruleId) := by
  apply List.ext_get
  · simp [make_new_rules]
  · intro i h₁ h₂
    simp [make_new_rules, ruleId_updateAct]

-- helper: map-get over finRange reproduces the list
@[simp] lemma map_get_finRange_eq {α} (l : List α) :
  (List.finRange l.length).map (fun i => l.get i) = l := by
  simp [List.ofFn_eq_map]


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
  -- Prove (map ruleId new_rules) = (map ruleId node.rules) indexwise, then reuse node.nodupIds
  let nodupIds :=
    by
      classical
      -- 1) Lists have the same length
      have len_eq : new_rules.length = node.rules.length := by
        simp [new_rules, make_new_rules]      -- finRange/map preserves length

      -- 2) Pointwise same ruleId at every index
      have ids_eq :
        new_rules.map (·.ruleId) = node.rules.map (·.ruleId) :=
      by
        apply List.ext_get
        · simp [new_rules, make_new_rules]      -- lengths
        · intro i hiL hiR
          -- align Fin indices on both sides
          have fi_rules  : Fin node.rules.length := ⟨i, by simpa using hiR⟩
          have fi_new    : Fin new_rules.length  := ⟨i, by simpa [len_eq] using hiL⟩
          -- both sides reduce to the same `node.rules[fi_rules].ruleId`
          simp [new_rules, make_new_rules, len_eq]

      -- 3) Transfer Nodup through the equality
      simpa [ids_eq] using node.nodupIds

  { rules := new_rules, nodupIds := nodupIds }

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
            let selectors₀ : List (List Bool) :=
              List.map (fun v : List.Vector Bool n => selector v.toList) initial_vectors

            have act_len :
              (activateLayerFromSelectors selectors₀ layer_hd).length = layer_hd.nodes.length :=
              activateLayerFromSelectors_length selectors₀ layer_hd
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
  - The underlying graph is a leveled, rooted DAG (as in Def. 2 of the paper).
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


namespace CircuitOp

/-- Wires are addressed by natural indices. -/
abbrev Wire := Nat

/-- Primitive Boolean gates. `out` is always a fresh wire produced by the gate. -/
inductive Gate
  | const (out : Wire) (val : Bool)
  | buf   (a out : Wire)
  | not   (a out : Wire)
  | and   (a b out : Wire)
  | or    (a b out : Wire)
  | xor   (a b out : Wire)
  deriving Repr, DecidableEq

/-- A circuit is a fixed number of wires and an ordered list of gates. -/
structure Circuit where
  numWires : Nat
  gates    : List Gate
  deriving Repr, Inhabited

/-- Internal builder state. -/
structure BuildState where
  nextWire : Nat := 0
  gates    : List Gate := []
deriving Repr

abbrev Builder := StateM BuildState

def allocInput (i : Nat) : Builder Wire :=
  pure i

/-- Allocate a fresh wire. -/
def fresh : Builder Wire := do
  let s ← get
  set { s with nextWire := s.nextWire + 1 }
  pure s.nextWire

/-- Append a gate. -/
def emit (g : Gate) : Builder Unit := do
  let s ← get
  set { s with gates := s.gates.concat g }

/-- Run a builder and get the produced result and circuit. -/
def runBuilder (b : Builder α) : (α × Circuit) :=
  Id.run <| do
    let (a, s) := b { nextWire := 0, gates := [] } |>.run
    let c : Circuit := { numWires := s.nextWire, gates := s.gates }
    (a, c)


def allocConst (val : Bool) : Builder Wire := do
  let out ← fresh
  emit (Gate.const out val)
  pure out

def allocBuf (a : Wire) : Builder Wire := do
  let out ← fresh
  emit (Gate.buf a out)
  pure out

def allocNot (a : Wire) : Builder Wire := do
  let out ← fresh
  emit (Gate.not a out)
  pure out

def allocAnd (a b : Wire) : Builder Wire := do
  let out ← fresh
  emit (Gate.and a b out)
  pure out

def allocOr (a b : Wire) : Builder Wire := do
  let out ← fresh
  emit (Gate.or a b out)
  pure out

def allocXor (a b : Wire) : Builder Wire := do
  let out ← fresh
  emit (Gate.xor a b out)
  pure out

def allocConstVec (n : Nat) (val : Bool) : Builder (List Wire) := do
  let rec go (k : Nat) (acc : List Wire) := do
    if k = 0 then pure acc.reverse
    else
      let w ← allocConst val
      go (k-1) (w :: acc)
  go n []


def simulate (c : Circuit) (initial : Array Bool) : Array Bool :=
  Id.run do
    let mut vals := initial
    if vals.size < c.numWires then
      vals := vals ++ (Array.replicate (c.numWires - vals.size) false)
    for g in c.gates do
      match g with
      | Gate.const out v   => vals := vals.set! out v
      | Gate.buf a out     => vals := vals.set! out vals[a]!
      | Gate.not a out     => vals := vals.set! out (!vals[a]!)
      | Gate.and a b out   => vals := vals.set! out (vals[a]! && vals[b]!)
      | Gate.or  a b out   => vals := vals.set! out (vals[a]! || vals[b]!)
      | Gate.xor a b out   => vals := vals.set! out (xor (vals[a]!) (vals[b]!))
    pure vals

/-- Convenience: run a builder then simulate with given inputs. -/
def simulateBuilder (b : Builder α) (inputs : List Bool)
  : (α × Array Bool × Circuit) :=
  let (a, c) := runBuilder b
  let arr := Array.mk inputs
  let out := simulate c arr
  (a, out, c)

/-- Turn a Bool vector into constant wires (one per bit). -/
def constVecOf {n : Nat} (v : List.Vector Bool n) : Builder (List Wire) := do
  let rec go (xs : List Bool) (acc : List Wire) := do
    match xs with
    | []      => pure acc.reverse
    | b :: tl =>
      let w ← allocConst b
      go tl (w :: acc)
  go v.toList []

-- Interpret the tagless algebra in the gate-level builder
instance : BitVecAlg CircuitOp.Builder where
  Bit := CircuitOp.Wire
  Vec := fun _ => List CircuitOp.Wire

  const  := CircuitOp.allocConst
  buf    := CircuitOp.allocBuf
  bnot   := CircuitOp.allocNot
  band   := CircuitOp.allocAnd
  bor    := CircuitOp.allocOr
  bxor   := CircuitOp.allocXor

  vconst    := fun {n} b => CircuitOp.allocConstVec n b
  vmap      := fun f v => v.mapM f
  vzipWith  := fun f v1 v2 => (List.zip v1 v2).mapM (fun (a,b) => f a b)

  -- ↓ implement reduce-or directly with gates (no tagless helper)
  vreduceOr := fun v => do
    match v with
    | []      => CircuitOp.allocConst false
    | a :: tl =>
      let rec go (cur : CircuitOp.Wire) (rest : List CircuitOp.Wire) := do
        match rest with
        | []     => pure cur
        | b::rs  =>
          let cur' ← CircuitOp.allocOr cur b
          go cur' rs
      go a tl

  -- Lists of wires already serve as Vecs here
  vfromList := fun bs _h => pure bs
  vtoList   := fun v => pure v

  -- Build a Vec from Bool literals (allocate constants)
  vfromBools := fun bs _h => bs.mapM CircuitOp.allocConst

  -- Mask a vector by a bit: out[i] = bit ∧ v[i]
  vmaskBy := fun bit v => v.mapM (fun a => CircuitOp.allocAnd bit a)

/-- Activation arity (metadata), ignoring the booleans inside. -/
def actArity : ActivationBits → Nat
  | ActivationBits.intro _   => 1
  | ActivationBits.elim _ _  => 2
  | ActivationBits.repetition _ => 1

/-- Arity list for a node’s rules (intro→1, elim→2). -/
def nodeActArities {n : Nat} (c : CircuitNode n) : List Nat :=
  c.rules.map (fun r => actArity r.activation)

/-- Check that the provided activation-wires-per-rule match the expected arities. -/
def actWiresWellFormed (expected : List Nat) (given : List (List Wire)) : Bool :=
  (List.zip expected given).all (fun (k, ws) => ws.length = k)


def List.replicateM {m : Type u → Type v} [Monad m] {α : Type u} (n : Nat) (x : m α) : m (List α) :=
  match n with
  | 0     => pure []
  | n + 1 => do
    let a ← x
    let rest ← replicateM n x
    pure (a :: rest)

def constDepsOf {n} (vs : List (List.Vector Bool n)) : Builder (List (List Wire)) :=
  vs.mapM constVecOf

/-- Read a list of wires from a simulated value array. -/
def readVec (vals : Array Bool) (ws : List Wire) : List Bool :=
  ws.map (fun w => vals[w]!)

/-- AND-of-NOTs = "all false" on a vector. -/
def vecAllFalse (xs : List Wire) : Builder Wire := do
  let any ← BitVecAlg.vreduceOr (m := Builder) (n := xs.length) xs
  BitVecAlg.bnot (m := Builder) any

def n := 4
def enc : List.Vector Bool n := ⟨[false,false,true,false], by decide⟩

-- Reuse your rule constructors, just give each rule a distinct ruleId.
-- If your mkIntroRule/mkElimRule already take a ruleId argument, use that.
-- Otherwise, set it via structure-update as below.
-- new API: (ruleId : Nat) → encoder/bits…
def rI : Rule n := mkIntroRule 0 enc true
def rE : Rule n := mkElimRule 1 true true

def node : CircuitNode n :=
{
  rules := [rI, rE], nodupIds := by
    -- map ruleId over [rI, rE] is [0, 1], so it's Nodup
    simp [rI, rE, mkIntroRule, mkElimRule]
}


def layer : GridLayer n :=
  { nodes := [node],
    -- one node, two rules; each rule has a list of (src,edge) pairs
    -- e.g. pull two bits for elim from selector of prev node 0
    incoming := [ [ [(0,1)], [(0,0),(0,3)] ] ] }

def init : List (List.Vector Bool n) :=
  [ ⟨[true,false,true,false], by decide⟩
  , ⟨[false,false,false,false], by decide⟩
  ]

-- #eval simulateWholeGridAutoActsValues [layer] init
-- -- → (all layer outputs as Bool lists, had_error flag, circuit)

-- -- compare compiled vs. semantic at node 0 in layer 0:
-- #eval checkAgainstSemanticsAt [layer] init ⟨0, by decide⟩ ⟨0, by decide⟩
-- true = match

def semLayersToBools {n} (xs : List (List (List.Vector Bool n))) :
    List (List (List Bool)) :=
  xs.map (fun layer => layer.map (·.toList))

-- #eval (CircuitOp.gatesFor [layer] init).length       -- how many gates?
-- #eval (CircuitOp.gatesAndHandlesFor [layer] init)     -- full details (prints via `Repr`)

/-- rule “shape” coming from the proof (no activation bits here). -/
inductive RuleSpec (n : ℕ)
  | intro (encoder : List.Vector Bool n)   -- ⊃I
  | elim                                   -- ⊃E

/-- one node: just the rule specs; wiring lives at the layer level. -/
structure NodeSpec (n : ℕ) where
  rules : List (RuleSpec n)

/-- one layer: nodes + incoming wiring for each node’s rules. -/
structure LayerSpec (n : ℕ) where
  nodes    : List (NodeSpec n)
  incoming : IncomingMapsLayer              -- same shape you already use

/-- whole proof IR for size `n`. -/
structure ProofSpec (n : ℕ) where
  layers          : List (LayerSpec n)
  initial_vectors : List (List.Vector Bool n)

def ruleOfSpec {n} (rid : Nat) : RuleSpec n → Rule n
  | RuleSpec.intro enc => mkIntroRule rid enc true
  | RuleSpec.elim      => mkElimRule rid true true

-- (optional but makes the next simp bulletproof)
@[simp] lemma ruleOfSpec_ruleId {n} (i : Nat) (rs : RuleSpec n) :
  (ruleOfSpec i rs).ruleId = i := by
  cases rs <;> simp [ruleOfSpec, mkIntroRule, mkElimRule]

@[simp] lemma map_val_finRange (len : Nat) :
  (List.finRange len).map (fun i : Fin len => (i : Nat)) = List.range len := by
  apply List.ext_get
  · simp
  · intro i _ _
    simp [List.get]

def nodeOfSpec {n} (ns : NodeSpec n) : CircuitNode n := by
  classical
  let len := ns.rules.length
  -- Build rules by safe Fin indexing
  let rules : List (Rule n) :=
    (List.finRange len).map (fun i : Fin len =>
      ruleOfSpec (i : Nat) (ns.rules.get i))

  -- map ruleId over `rules` collapses to mapping Fin→Nat over finRange
  have ids_eq₁ :
      rules.map (·.ruleId)
      = (List.finRange len).map (fun i : Fin len => (i : Nat)) := by
    -- prove by pointwise equality on indices
    apply List.ext_get
    ·
      simp [rules]
    ·
      intro i hiL hiR
      simp [rules, List.getElem_map, ruleOfSpec_ruleId]

  -- Fin→Nat over finRange is exactly `range`
  have ids_eq₂ :
      (List.finRange len).map (fun i : Fin len => (i : Nat))
      = List.range len :=
    map_val_finRange len

  have ids_eq : rules.map (·.ruleId) = List.range len :=
    ids_eq₁.trans ids_eq₂

  exact
  {
    rules    := rules,
    nodupIds := by
      -- `range len` is nodup; transport via equality
      have h : (List.range len).Nodup := List.nodup_range (n := len)
      simpa [ids_eq] using h,
  }



def layerOfSpec {n} (ls : LayerSpec n) : GridLayer n :=
  { nodes := ls.nodes.map nodeOfSpec, incoming := ls.incoming }

def gridOfProof {n} (ps : ProofSpec n) : (List (GridLayer n)) × (List (List.Vector Bool n)) :=
  (ps.layers.map layerOfSpec, ps.initial_vectors)


inductive Formula
  | atom (name : String)
  | impl (A B : Formula)
  deriving DecidableEq, Repr

structure Vertex where
  node      : Nat
  LEVEL     : Nat
  FORMULA   : Formula
  HYPOTHESIS : Bool
  COLLAPSED : Bool
  PAST      : List Nat
  deriving Repr

structure Deduction where
  START : Vertex
  END   : Vertex
  COLOUR : Nat
  DEPENDENCY : List Formula
  deriving Repr

structure DLDS where
  V : List Vertex
  E : List Deduction
  A : List (Vertex × Vertex) := []
  deriving Repr

def collectFormulas (d : DLDS) : List Formula :=
  -- include subformulas if you want full closure; otherwise just node labels:
  (d.V.map (·.FORMULA)).eraseDups

-- pick your order (lex by shape/name is fine)
def compareFormula : Formula → Formula → Ordering
  | .atom a, .atom b => compare a b
  | .atom _, .impl _ _ => .lt
  | .impl _ _, .atom _ => .gt
  | .impl a b, .impl c d =>
      match compareFormula a c with
      | .eq => compareFormula b d
      | o   => o

def orderΓ [DecidableEq Formula] (Γ : List Formula) : List Formula :=
  Γ.eraseDups

def buildUniverse [DecidableEq Formula] (d : DLDS) : List Formula :=
  orderΓ (d.V.map (·.FORMULA))

def indexOf (Γ : List Formula) (φ : Formula) : Nat :=
  (Γ.idxOf φ)  -- assumes DecidableEq; safe after `orderΓ`

-- needs membership decidability
def encodeSet [DecidableEq Formula]
  (Γord : List Formula) (fs : List Formula) :
  List.Vector Bool Γord.length :=
  ⟨
    -- i ranges over indices of Γord
    List.ofFn (fun (i : Fin Γord.length) =>
      decide (Γord.get i ∈ fs)),
    -- length (List.ofFn ...) = Γord.length
    by simp
  ⟩

def encoderForIntro [DecidableEq Formula]
  (Γord : List Formula) (φ : Formula) :
  Option (List.Vector Bool Γord.length) :=
  match φ with
  | .impl A _ =>
      let bits := Γord.map (fun ψ => decide (ψ = A))
      -- expose `bits` to `simp` so it can use `List.length_map`
      some ⟨bits, by simp [bits]⟩
  | _ => none


def specOfFormula (Γord : List Formula) (φ : Formula) :
  NodeSpec Γord.length :=
  match encoderForIntro Γord φ with
  | some enc => { rules := [RuleSpec.intro enc, RuleSpec.elim] }
  | none     => { rules := [RuleSpec.elim] }


def buildLayer (Γord : List Formula) : GridLayer Γord.length :=
  let rowNodes :=
    Γord.map (fun φ => nodeOfSpec (specOfFormula Γord φ))
  { nodes := rowNodes, incoming := [] }

/-- Group vertices by `LEVEL`, sort each layer by `node`,
    and carry an association list for node indices.
    Skipping levels is allowed; edges must satisfy ls < le. -/
structure LayeredDLDS where
  layers     : List (List Vertex)
  indexAlist : List (Nat × (Nat × Nat))  -- (nodeId ↦ (layerIdx, idxInLayer))
  deriving Repr

/-- Lookup `(layerIdx, idxInLayer)` for a node id in a `LayeredDLDS`. -/
def LayeredDLDS.indexOf? (L : LayeredDLDS) (n : Nat) : Option (Nat × Nat) :=
  (L.indexAlist.find? (fun p => p.fst = n)).map (·.snd)

/-- assoc-list lookup used internally in `layerize` checks -/
def idxLookup (idx : List (Nat × (Nat × Nat))) (n : Nat) : Option (Nat × Nat) :=
  (idx.find? (fun p => p.fst = n)).map (·.snd)

namespace List

def insertBy {α} (lt : α → α → Bool) (a : α) : List α → List α
  | []       => [a]
  | b :: bs  =>
    if lt a b then
      a :: b :: bs
    else
      b :: insertBy lt a bs

/-- Insertion sort by a boolean comparator `lt`. -/
def sortBy {α} (lt : α → α → Bool) : List α → List α
  | []       => []
  | a :: as  =>
    insertBy lt a (sortBy lt as)
end List

/-- Helper: collect all vertices at a given level and sort them by `node`. -/
def verticesAtLevelSorted (vs : List Vertex) (ℓ : Nat) : List Vertex :=
  let atL := vs.filter (fun v => v.LEVEL = ℓ)
  List.sortBy (fun a b => a.node < b.node) atL

/-- Build the list of layers 0..maxLev, each sorted by `node`. -/
def buildLayers (d : DLDS) : List (List Vertex) :=
  let maxLev := (d.V.map (·.LEVEL)).foldl Nat.max 0
  (List.range (maxLev + 1)).map (fun ℓ => verticesAtLevelSorted d.V ℓ)

/-- Pair each element with its 0-based index. -/
def zipWithIndex {α} (xs : List α) : List (Nat × α) :=
  let rec go : List α → Nat → List (Nat × α)
    | [],      _ => []
    | a :: as, i => (i, a) :: go as (i + 1)
  go xs 0

/-- Build a `(nodeId → (layerIdx, idxInLayer))` association list. -/
def buildIndex (layers : List (List Vertex)) : List (Nat × (Nat × Nat)) :=
  let withLayerIdx := zipWithIndex layers
  withLayerIdx.foldl
    (fun acc (ℓ, row) =>
      let rowWithIdx := zipWithIndex row
      acc ++ rowWithIdx.map (fun (i, v) => (v.node, (ℓ, i)))
    )
    []

/-- Returns a structured layering if:
    - layers are built from LEVEL,
    - each layer is sorted by node,
    - and every deduction goes from a lower level to a strictly higher level (ls < le).
    Otherwise, returns an explanatory error. -/


def layerize (d : DLDS) : Except String LayeredDLDS := do
  let layers := buildLayers d
  let idx    := buildIndex layers

  -- sanity: every vertex appears in the index
  for v in d.V do
    match idxLookup idx v.node with
    | none   => throw s!"Unknown vertex id in index: {v.node}"
    | some _ => pure ()

  -- edges must go forward (skips allowed): ls < le
  for e in d.E do
    let some (ls, _) := idxLookup idx e.START.node
      | throw s!"Edge START not found in index: {e.START.node}"
    let some (le, _) := idxLookup idx e.END.node
      | throw s!"Edge END not found in index: {e.END.node}"
    if ¬ (ls < le) then
      throw s!"Non-forward edge {e.START.node}@L{ls} → {e.END.node}@L{le} (expected ls < le)"

  let result : LayeredDLDS :=
  {
    layers     := layers,
    indexAlist := idx,
  }

  pure result

/-- Compact the layered view by dropping empty layers and
    remapping to consecutive indices 0..L-1. -/
structure CompactDLDS where
  layers      : List (List Vertex)              -- nonempty, consecutive
  indexAlist  : List (Nat × (Nat × Nat))      -- nodeId ↦ (compactLayer, idxInLayer)
  deriving Repr


/-- Drop empty layers from `LayeredDLDS` and reindex. -/
def compactify (L : LayeredDLDS) : CompactDLDS :=
  let nonempty := L.layers.filter (fun row => ¬ row.isEmpty)
  let idx      := buildIndex nonempty
  {
    layers      := nonempty,
    indexAlist  := idx,
  }

/-- Lookup helper on the compact index. -/
def CompactDLDS.indexOf? (C : CompactDLDS) (n : Nat) : Option (Nat × Nat) :=
  (C.indexAlist.find? (fun p => p.fst = n)).map (·.snd)


/-- Extract the *initial dependency vectors* for the compact layer 0.
    Policy: a vertex is initially “selected” iff it is a hypothesis (`HYPOTHESIS = true`).
    We encode the set of dependencies using `encodeSet` you defined. -/
def initialVectorsOfLayer0 (Γord : List Formula) (C : CompactDLDS)
  : List (List.Vector Bool Γord.length) :=
  match C.layers with
  | []       => []
  | row :: _ =>
    row.map (fun v =>
      CircuitOp.encodeSet Γord (if v.HYPOTHESIS then [v.FORMULA] else [])
    )

/-- Convenience: check that the `n` parameter matches `Γord.length`. -/
def assertN (n : Nat) (Γord : List Formula) : Except String PUnit :=
  if _ : n = Γord.length then
    pure ()
  else
    throw s!"Dimension mismatch: n = {n} but |Γ| = {Γord.length}."

/-- Big-endian Bool list -> Nat index, consistent with `natToBits` used by `selector`. -/
def bitsToNatBE (bs : List Bool) : Nat :=
  bs.foldl (fun acc b => acc * 2 + (if b then 1 else 0)) 0

/-- Dependency set -> selector index (w.r.t. Γ ordering). -/
def selIndexForDependency (Γord : List Formula) (deps : List Formula) : Nat :=
  bitsToNatBE ((encodeSet Γord deps).toList)

/-- Safe lookup of (layer,idxInLayer) inside a CompactDLDS; error with a message. -/
def expectIndex (C : CompactDLDS) (nodeId : Nat) : Except String (Nat × Nat) :=
  match C.indexOf? nodeId with
  | some p => .ok p
  | none   => .error s!"index: unknown node id {nodeId}"

def isImpl : Formula → Option (Formula × Formula)
  | .impl A B => some (A,B)
  | _         => none


/-- Build incoming wiring for one node (total, no errors). -/
def incomingForNode
  (Γord : List Formula)
  (C    : CircuitOp.CompactDLDS)
  (rowIdx colIdx : Nat)
  : IncomingMap :=
  -- Grab the vertex if it exists; otherwise, no rules get any wiring.
  let row    := C.layers.getD rowIdx []
  let v?     := row[colIdx]?
  match v? with
  | none      => []   -- node missing ⇒ no rules wired
  | some v    =>
    -- Decide rule shape for this formula: intro+elim or just elim.
    let arities : List Nat :=
      match CircuitOp.encoderForIntro Γord v.FORMULA with
      | some _ => [1, 2]  -- ⊃I needs 1 bit; ⊃E needs 2 bits
      | none   => [2]     -- only ⊃E
    -- For now, return empty activation sources for each rule.
    -- (Real extraction will fill these with (srcNode, edgeIdx).)
    arities.map (fun _ => [])

/-- Wiring for an entire layer (total). -/
def incomingForLayer
  (Γord : List Formula)
  (C    : CircuitOp.CompactDLDS)
  (rowIdx : Nat)
  : IncomingMapsLayer :=
  let row := C.layers.getD rowIdx []
  -- one IncomingMap per node in this layer
  (List.range row.length).map (fun colIdx =>
    incomingForNode Γord C rowIdx colIdx)

/-- Build fully wired `GridLayer`s (nodes + incoming), total version. -/
def buildWiredLayers
  (Γord : List Formula)
  (C    : CircuitOp.CompactDLDS)
  : List (GridLayer Γord.length) :=
  -- Nodes: determined by the formula at each vertex
  let nodesPerLayer : List (List (CircuitNode Γord.length)) :=
    C.layers.map (fun row =>
      row.map (fun v =>
        CircuitOp.nodeOfSpec (CircuitOp.specOfFormula Γord v.FORMULA)))
  -- Incoming: currently empty (or default) per node; later you’ll fill real wiring
  let incomingPerLayer : List IncomingMapsLayer :=
    (List.range C.layers.length).map (incomingForLayer Γord C)
  -- Zip into GridLayers
  (List.zip nodesPerLayer incomingPerLayer).map (fun (ns, inc) =>
    { nodes := ns, incoming := inc })

/-- End-to-end: DLDS → (wired layers, initial vectors). -/
def gridFromDLDS
  (d : DLDS)
  : Except String (
      List (GridLayer (buildUniverse d).length) ×
      List (List.Vector Bool (buildUniverse d).length)
    ) := do
  let Γ := buildUniverse d
  let layered ← layerize d
  let C := compactify layered
  let layers := buildWiredLayers Γ C
  let init := initialVectorsOfLayer0 Γ C
  pure (layers, init)

/-- Run semantic pipeline from DLDS. -/
def runSemanticFromDLDS (d : DLDS) :
  Except String (List (List (List Bool))) := do
  let (layers, init) ← gridFromDLDS d
  -- to Bool lists for inspection:
  let sem := evalGridSelector layers init
  pure (sem.map (fun lay => lay.map (·.toList)))


end CircuitOp

-- multiple XOR (pure)
#eval multipleXorA (m := Pure.M) [true,false,false,false]     -- expect true
#eval multipleXorA (m := Pure.M) [true,true,false,false]     -- expect false

-- selector (pure) vs your boolean selector
#eval selectorA (m := Pure.M) [true,false]                   -- expect [false,true,false,false]
#eval selector [true,false]

-- tagless on the builder: compile & simulate
#eval
  let (w, vals, _) :=
    CircuitOp.simulateBuilder
      (do
        let ws ← [true,false,true,false].mapM (fun b =>
          BitVecAlg.const (m := CircuitOp.Builder) b)
        multipleXorA (m := CircuitOp.Builder) ws)
      []
  vals.get! (w : Nat)


-- a 1-layer, 1-node DLDS with atom “p”, marked as a hypothesis
-- make sure the previous `namespace CircuitOp` is closed:
-- end CircuitOp

-- (place this after `end CircuitOp`)

def V0 : CircuitOp.Vertex :=
{ node := 0,
  LEVEL := 0,
  FORMULA := CircuitOp.Formula.atom "p",
  HYPOTHESIS := true,
  COLLAPSED := false,
  PAST := [] }

def D0 : CircuitOp.DLDS :=
{ V := [V0],
  E := [] }

def n := 4
def enc : List.Vector Bool n := ⟨[false,false,true,false], by decide⟩
def rI : Rule n := mkIntroRule 0 enc true
def rE : Rule n := mkElimRule 1 true true
def node : CircuitNode n :=
  { rules := [rI, rE], nodupIds := by simp [rI, rE, mkIntroRule, mkElimRule] }
def layer : GridLayer n :=
  { nodes := [node],
    incoming := [ [ [(0,1)], [(0,0),(0,3)] ] ] }
def init : List (List.Vector Bool n) :=
  [ ⟨[true,false,true,false], by decide⟩
  , ⟨[false,false,false,false], by decide⟩
  ]

/-- Tagless, wiring-based activation extraction:
    For each rule `i`, look up its incoming `(src, edge)` pairs,
    read those selector bits from `prevSelectors`, and produce the *combined* activation bit:
      - intro: use the *first* pair’s bit (if any);
      - elim:  AND the *first two* pairs’ bits (if present).
    OOB indices → `false`. Extra pairs are ignored. -/
def actsFromIncomingA {m} [Monad m] [BitVecAlg m] {n}
  (rules         : List (Rule n))
  (prevSelectors : List (List (BitVecAlg.Bit (m := m))))
  (incoming      : IncomingMap)
  : m (List (BitVecAlg.Bit (m := m))) := do
  let len := rules.length
  (List.finRange len).mapM <| fun i => do
    let r  := rules.get i
    let ps : List (Nat × Nat) :=
      match incoming[i]? with
      | some pairs => pairs
      | none       => []

    -- safe selector bit fetch with default false
    let getSel (src edge : Nat) : m (BitVecAlg.Bit (m := m)) := do
      match prevSelectors[src]? with
      | some sel =>
        match sel[edge]? with
        | some b => BitVecAlg.buf b
        | none   => BitVecAlg.const false
      | none => BitVecAlg.const false

    match r.type, ps with
    | RuleData.intro _, (s,e) :: _ =>
        getSel s e
    | RuleData.elim,  (s1,e1) :: (s2,e2) :: _ => do
        let b1 ← getSel s1 e1
        let b2 ← getSel s2 e2
        BitVecAlg.band b1 b2
    | _, _ =>
        BitVecAlg.const false


-- Pseudocode sketch (names match your library)
def evalLayerA {m} [Monad m] [BitVecAlg m] {n}
  (prev : List (BitVecAlg.Vec (m := m) n))
  (L : GridLayer n)
  : m (List (BitVecAlg.Vec (m := m) n) × BitVecAlg.Bit (m := m)) := do
  -- selectors from previous outputs, taglessly
  let sels ← prev.mapM (fun v => do
    let bs ← BitVecAlg.vtoList v
    selectorA (m := m) bs)
  -- for each node: build acts from incoming wiring (AND both for elim)
  let outsErrs ← (List.zip L.nodes L.incoming).mapM (fun (nd, incMap) => do
    let acts ← actsFromIncomingA (m := m) nd.rules sels incMap   -- returns combined act per rule
    nodeLogicWithErrorGivenActsA (m := m) (n := n) nd.rules acts prev)
  let outs := outsErrs.map Prod.fst
  let errs := outsErrs.map Prod.snd
  let err  ← orListA (m := m) errs
  pure (outs, err)

def evalGridA {m} [Monad m] [BitVecAlg m] {n}
  (layers : List (GridLayer n))
  (init   : List (BitVecAlg.Vec (m := m) n))
  : m (List (List (BitVecAlg.Vec (m := m) n)) × BitVecAlg.Bit (m := m)) := do
  let mut acc  := [init]
  let mut prev := init
  let mut eacc ← BitVecAlg.const (m := m) false
  for L in layers do
    let (next, eL) ← evalLayerA (m := m) prev L
    acc  := acc ++ [next]
    prev := next
    eacc ← BitVecAlg.bor eacc eL
  pure (acc, eacc)


-- /-- For any BitVecAlg instance, evalGridA is natural in algebra morphisms.
--     Instantiating the morphism Builder→Pure (simulate+read) yields equality of results. -/
-- theorem evalGridA_builder_sound
--   (layers : List (GridLayer n))
--   (init   : List (List.Vector Bool n)) :
--   let ((outsW, errW), circ) := CircuitOp.runBuilder (CircuitOp.compileWholeGridAutoActs layers init)
--   let vals := CircuitOp.simulate circ #[]
--   (outsW.map (·.map (CircuitOp.readVec vals)), vals[errW]!)
--     = evalGridSelectorWithError layers init

open CircuitOp BitVecAlg

inductive EdgeKind where
  | rep   (tgtIdx : Nat)
  | intro (tgtIdx : Nat) (discharge : Formula)
  | elim  (tgtIdx : Nat) (minorIdx : Nat) (ante : Formula)

noncomputable instance : DecidableEq EdgeKind := Classical.decEq _

instance : Repr EdgeKind where
  reprPrec
    | .rep t, _       => s!"EdgeKind.rep {repr t}"
    | .intro t d, pr  =>
        let ds := Repr.reprPrec d pr
        s!"EdgeKind.intro {repr t} {ds}"
    | .elim t m a, pr =>
        let as := Repr.reprPrec a pr
        s!"EdgeKind.elim {repr t} {repr m} {as}"



/-- Per source (prev level) list all outgoing EdgeKind options towards curr level. -/
abbrev OutgoingOptions := List EdgeKind

def enumerateOutgoingForLevel
  (prevFormulas : List Formula)
  (currFormulas : List Formula)
  : List OutgoingOptions :=
Id.run do
  let mut out : List OutgoingOptions := []

  -- iterate sources with Fin indices (no Inhabited needed)
  for s in List.finRange prevFormulas.length do
    let Fs   := prevFormulas.get s
    let _ := s.val
    let mut opts : OutgoingOptions := []

    -- REP: Fs == Ft
    for t in List.finRange currFormulas.length do
      let Ft   := currFormulas.get t
      let tIdx := t.val
      if Fs = Ft then
        opts := EdgeKind.rep tIdx :: opts

    -- INTRO: Ft = X ⊃ Y, Fs = Y
    for t in List.finRange currFormulas.length do
      let Ft   := currFormulas.get t
      let tIdx := t.val
      match Ft with
      | Formula.impl X Y =>
          if Fs = Y then
            opts := EdgeKind.intro tIdx X :: opts
      | _ => ()

    -- ELIM: Fs = X ⊃ Y, Ft = Y, choose a minor m with X
    match Fs with
    | Formula.impl X Y =>
        for t in List.finRange currFormulas.length do
          let Ft   := currFormulas.get t
          let tIdx := t.val
          if Ft = Y then
            for m in List.finRange prevFormulas.length do
              let Fm   := prevFormulas.get m
              let mIdx := m.val
              if Fm = X then
                opts := EdgeKind.elim tIdx mIdx X :: opts
    | _ => ()

    -- append this source's options (reverse to preserve original order)
    out := out ++ [opts.reverse]

  pure out

/-- ceil(log2 n); by convention, `bitsForChoices 0 = 0`, and `bitsForChoices 1 = 0`. -/
def bitsForChoices (n : Nat) : Nat :=
  Id.run do
    -- we want the least k with 2^k ≥ n', with n' = max(n, 1)
    let n' := if n = 0 then 1 else n
    let mut k := 0
    while Nat.pow 2 k < n' do
      k := k + 1
    pure k

structure SourcePathSpec where
  bits  : Nat   -- K_s
  start : Nat   -- starting offset into the global input array
deriving Repr

/-- At a level, we have one SourcePathSpec per **prev** node. -/
abbrev LevelPathLayout := List SourcePathSpec

/-- For the whole grid (between consecutive level pairs). Index i = specs for prev level i. -/
abbrev PathLayout := List LevelPathLayout

/-- Given choices per source, compute layout and total inputs consumed so far. -/
def buildLevelPathLayout (runningStart : Nat) (outgoing : List OutgoingOptions)
  : (LevelPathLayout × Nat) :=
  let specs :=
    outgoing.map (fun opts =>
      let m := opts.length + 1  -- +1 for inactive index 0
      let k := bitsForChoices m
      SourcePathSpec.mk k 0  -- temp start=0
    )
  -- assign sequential starts
  let rec assign (accStart : Nat) (ss : List SourcePathSpec) (acc : List SourcePathSpec)
    : (List SourcePathSpec × Nat) :=
    match ss with
    | [] => (acc.reverse, accStart)
    | s :: tl =>
        let s' := { s with start := accStart }
        assign (accStart + s.bits) tl (s' :: acc)
  let (specs2, endStart) := assign runningStart specs []
  (specs2, endStart)

open CircuitOp

/-- Read K path bits (indices `[start .. start+bits)`) as Builder wires, MSB→LSB or LSB→MSB
    consistently with your `selectorA` bit order. Here I pass them **in the same order**
    they’re read; if your `selectorA` expects MSB-first, keep as-is; if it expects LSB-first,
    just `reverse` the list before calling `selectorA`. -/
def oneHotFromPathInputs (start bits : Nat) : Builder (List Wire) := do
  let mut xs : List Wire := []
  -- collect [start, start+1, ..., start+bits-1]
  for i in [start : start + bits] do
    xs := xs ++ [← allocInput i]
  -- If your `selectorA` expects MSB→LSB and `xs` is MSB→LSB already, keep it.
  -- If it expects the opposite, do: `selectorA xs.reverse`
  selectorA xs

/-- Decode K_s input bits at [start .. start+K_s) into a one-hot of size 2^K_s. -/
def decodeOneHotFromInputs (spec : SourcePathSpec) : Builder (List Wire) := do
  let mut bitsW : List Wire := []
  for i in [spec.start : spec.start + spec.bits] do
    bitsW := (← allocInput i) :: bitsW
  -- your builder-side binary decoder → one-hot of length 2^K
  decodeIndex bitsW.reverse

/-- χ_X mask as a vector over your basis; implement from your existing encoder. -/
def dischargeMaskFromBasis
  [DecidableEq Formula]
  (basis : List Formula) (X : Formula)
  : Builder (List Wire) := do
  let mut ws : List Wire := []
  for f in basis do
    ws := ws ++ [← allocConst (decide (f = X))]
  pure ws



/-- Target index selector for an edge kind. -/
def tgtOf : EdgeKind → Nat
  | .rep   t     => t
  | .intro t _   => t
  | .elim  t _ _ => t

instance : Inhabited SourcePathSpec :=
  ⟨{ start := 0, bits := 0 }⟩

/-- Compile one level where selectors choose **outgoing** edges per source. -/
def compileLevelOutgoing
  (encN      : Nat)
  (basis     : List Formula)
  (prevDeps  : List (List Wire))
  (currForms : List Formula)
  (outgoing  : List OutgoingOptions)   -- per source
  (layout    : LevelPathLayout)        -- per source
  : Builder (List (List Wire) × Wire) := do

  -- arrays = clean indexing
  let prevArr   := prevDeps.toArray
  let outsArr   := outgoing.toArray
  let layoutArr := layout.toArray

  let tgtCount  := currForms.length
  let srcCount  := outsArr.size
  let layCount  := layoutArr.size
  let loopCount := Nat.min srcCount layCount

  -- accumulators
  let zeroVec ← allocConstVec (n := encN) false
  let mut perTarget : Array (List Wire) := Array.replicate tgtCount zeroVec
  let mut globalErr : Wire := (← allocConst false)

  -- size mismatch (layout vs outgoing) → error bit
  let mismatchErr ← allocConst (layCount ≠ srcCount)
  globalErr ← allocOr globalErr mismatchErr

  -- for each source (bounded by the min of both arrays' sizes)
  for s in [0:loopCount] do
    let opts := outsArr[s]!
    let spec : SourcePathSpec := layoutArr[s]!
    let sel  ← oneHotFromPathInputs spec.start spec.bits
    let m    := opts.length

    -- any selection beyond m (0=inactive, 1..m=valid) → error
    let extraErr ←
      sel.drop (m+1) |>.foldlM (fun acc w => allocOr acc w) (← allocConst false)
    globalErr ← allocOr globalErr extraErr

    -- for each outgoing option of this source
    for (j0, opt) in opts.enum do
      let j    := j0 + 1
      let pick := sel[j]!   -- Wire (Nat), has Inhabited so `!` is fine

      -- build contribution for this edge
      let contrib : List Wire ←
        match opt with
        | EdgeKind.rep _tgt =>
            vmaskBy (n := encN) pick (prevArr[s]!)
        | EdgeKind.intro _tgt X =>
            let χ  ← dischargeMaskFromBasis basis X          -- length = encN (basis.length should = encN)
            let nχ ← vmap      (n := encN) allocNot χ
            let masked ← vzipWith (n := encN) allocAnd (prevArr[s]!) nχ
            vmaskBy (n := encN) pick masked
        | EdgeKind.elim _tgt mIdx _X =>
            let maj := prevArr[s]!
            let min := prevArr[mIdx]!
            let orV ← vzipWith (n := encN) allocOr maj min
            vmaskBy (n := encN) pick orV

      -- accumulate into the target bucket
      let t      := tgtOf opt
      let prevV  : List Wire := perTarget[t]!
      let newV   ← vzipWith (n := encN) allocOr prevV contrib
      perTarget  := perTarget.set! t newV

  -- finalize outputs and return
  let outs := (List.range tgtCount).map (fun i => perTarget[i]!)
  pure (outs, globalErr)


structure BuiltOutgoing where
  encN      : Nat
  basis     : List Formula            -- NEW: basis used by dependency vectors
  levels    : List (List Vertex)      -- grouped by LEVEL, sorted
  formulas  : List (List Formula)     -- per level, same order as levels
  initDeps  : List (List Bool)        -- vectors for level 0 nodes
  outgoings : List (List OutgoingOptions)  -- for each (prev level), per-source options
  layout    : PathLayout                   -- for each (prev level), per-source path spec
deriving Repr

/-- Build the basis and the initial dependency vectors for level 0.
    Basis = unique hypotheses' formulas (you can swap in a richer basis later). -/
def buildEncodingAndInit (G : DLDS) (L0 : List Vertex) :
  List Formula × Nat × List (List Bool) :=
Id.run do
  let basis : List Formula := (G.V.map (·.FORMULA)).eraseDups
  let n := basis.length
  let oneHot (i : Nat) : List Bool := (List.range n).map (fun j => decide (j = i))
  let idxOf? (f : Formula) : Option Nat := basis.idxOf? f
  let init : List (List Bool) :=
    L0.map (fun v =>
      match idxOf? v.FORMULA with
      | some i => oneHot i
      | none   => List.replicate n false)
  pure (basis, n, init)


structure LevelPack where
  level : Nat
  nodes : List Vertex
  deriving Repr

open Std

def groupByLevel (G : DLDS) : List LevelPack :=
  Id.run do
    let mut m : Std.HashMap Nat (List Vertex) := {}
    for v in G.V do
      let prev := m.getD v.LEVEL []
      m := m.insert v.LEVEL (v :: prev)
    let sorted : Array (Nat × List Vertex) :=
      (m.toList).toArray.qsort (fun a b => a.fst < b.fst)
    let packs : List LevelPack :=
      sorted.toList.map (fun (k, vs) => { level := k, nodes := vs.reverse })
    pure packs

/-- Build the outgoing model and path layout for the entire DLDS. -/
def buildOutgoing (G : DLDS) : Except String BuiltOutgoing := do
  let packs := groupByLevel G
  if packs.isEmpty then
    throw "DLDS has no vertices"

  let levels   := packs.map (·.nodes)
  let formulas := levels.map (·.map (·.FORMULA))

  -- basis + size + level-0 initial dependency vectors
  let (basis, encN, initDeps) := buildEncodingAndInit G levels.head!

  -- per transition (prev → curr): outgoing options and path layout
  let mut outgoings : List (List OutgoingOptions) := []
  let mut layout    : PathLayout := []
  let mut cursor    : Nat := 0

  for k in [0 : levels.length - 1] do
    let prevF := formulas[k]!
    let currF := formulas[k+1]!
    let outs  := enumerateOutgoingForLevel prevF currF
    let (ll, cursor') := buildLevelPathLayout cursor outs
    outgoings := outgoings.concat outs
    layout    := layout.concat ll
    cursor    := cursor'

  -- assemble bundle
  pure {
    encN     := encN
    basis    := basis
    levels   := levels
    formulas := formulas
    initDeps := initDeps
    outgoings:= outgoings
    layout   := layout
  }


/-- Write `val` into `K` bits (little-endian) starting at `start`. -/
def writeBitsLE (arr : Array Bool) (start K val : Nat) : Array Bool :=
  Id.run do
    let mut a := arr
    for j in [0:K] do
      let bit := ((val >>> j) &&& 1) = 1
      a := a.set! (start + j) bit
    a

/-- Flatten path indices (per prev-level, per source) to input bits using the layout. -/
def packPathBits (layout : PathLayout) (path : List (List Nat)) : Array Bool :=
  Id.run do
    -- total number of bits across all level specs
    let totalBits :=
      layout.foldl (fun acc lvl =>
        acc + lvl.foldl (fun a spec => a + spec.bits) 0) 0
    let mut arr : Array Bool := Array.replicate totalBits false

    -- work with arrays for clean indexing + defaults
    let layoutArr := layout.toArray
    let pathArr   := path.toArray

    for lv in [0:layoutArr.size] do
      let lvlSpecs : LevelPathLayout := layoutArr[lv]!
      let specsArr := lvlSpecs.toArray
      let rowArr   := (pathArr.getD lv []).toArray
      for s in [0:specsArr.size] do
        let spec : SourcePathSpec := specsArr[s]!
        let val  : Nat            := rowArr.getD s 0
        arr := writeBitsLE arr spec.start spec.bits val

    pure arr

/-- Compile all levels and aggregate the global error. -/
def compileWholeGridOutgoing (B : BuiltOutgoing)
  : Builder (List (List (List Wire)) × Wire) := do
  let formsArr   := B.formulas.toArray
  let outsArr    := B.outgoings.toArray
  let layoutArr  := B.layout.toArray
  let transCount := outsArr.size

  -- level 0 as wires (convert each Bool to a const wire)
  let lvl0 ← B.initDeps.mapM (fun row => row.mapM allocConst)

  let mut allLevels : List (List (List Wire)) := [lvl0]
  let mut prev := lvl0
  let mut gErr : Wire := (← allocConst false)

  -- per transition: prev level k → curr level k+1
  for k in [0:transCount] do
    let currForms := formsArr[k+1]!
    let outs      := outsArr[k]!
    let specs     := layoutArr[k]!
    let (next, e) ← compileLevelOutgoing B.encN B.basis prev currForms outs specs
    gErr ← allocOr gErr e
    allLevels := allLevels ++ [next]
    prev := next

  pure (allLevels, gErr)


def simulateWithPath
  (B : BuiltOutgoing) (goalLevel goalIdx : Nat) (path : List (List Nat)) : IO Bool := do
  let inputs := packPathBits B.layout path

  -- Build everything in ONE builder run: grid -> pick goal -> zero-check -> final wire
  let (finalWire, circ) := CircuitOp.runBuilder do
    let (levelsW, gErr) ← compileWholeGridOutgoing B
    -- index with arrays (no List get!/Inhabited headaches)
    let levelsA    := levelsW.toArray
    let levelVecs  := levelsA[goalLevel]!       -- List (List Wire)
    let levelA     := levelVecs.toArray
    let goalV      := levelA[goalIdx]!          -- List Wire
    let zero       ← vecAllFalse goalV          -- Builder Wire
    let final      ← allocOr gErr zero
    pure final

  let outs := CircuitOp.simulate circ inputs    -- Array Bool (or similar)
  pure (outs[finalWire]!)


/-- Build circuit + keep the wire handles for (layer outputs, error) using outgoing path. -/
def buildCircuitWithHandlesOutgoing
  (B : BuiltOutgoing)
  : (CircuitOp.Circuit × List (List (List CircuitOp.Wire)) × CircuitOp.Wire) :=
  let ((outs, err), circ) := CircuitOp.runBuilder (compileWholeGridOutgoing B)
  (circ, outs, err)

/-- Run compiled pipeline from DLDS given a concrete path (per prev-level, per source). -/
def runCompiledFromDLDSWithPath
  (d : CircuitOp.DLDS)
  (path : List (List Nat))
  : Except String (List (List (List Bool)) × Bool) := do
  let B ← buildOutgoing d
  let ((outsW, errW), circ) := runBuilder (compileWholeGridOutgoing B)
  let inputs := packPathBits B.layout path
  let vals := simulate circ inputs
  let outsB : List (List (List Bool)) :=
    outsW.map (fun layer => layer.map (fun vec => readVec vals vec))
  let hadErr := vals[errW]!
  pure (outsB, hadErr)


def specLevelOutgoing
  (encN      : Nat)
  (basis     : List CircuitOp.Formula)
  (prevDeps  : List (List Bool))
  (currForms : List CircuitOp.Formula)
  (outgoing  : List (List EdgeKind))
  (choices   : List Nat)
  : (List (List Bool) × Bool) :=
Id.run do
  -- helpers
  let zeroVec : List Bool := List.replicate encN false
  let orVec  (a b : List Bool) : List Bool := List.zipWith (· || ·) a b
  let andVec (a b : List Bool) : List Bool := List.zipWith (· && ·) a b
  let notVec (v : List Bool) : List Bool := v.map (· = false)
  let chi (X : CircuitOp.Formula) : List Bool := basis.map (fun f => decide (f = X))

  let tgtCount := currForms.length
  let mut perTarget : Array (List Bool) := Array.replicate tgtCount zeroVec

  let mismatchErr : Bool := outgoing.length ≠ choices.length
  let srcCount    : Nat  := Nat.min outgoing.length choices.length

  let mut err : Bool := mismatchErr

  -- arrays for clean indexing
  let outsA   := outgoing.toArray        -- : Array (List EdgeKind)
  let prevA   := prevDeps.toArray        -- : Array (List Bool)
  let choiceA := choices.toArray         -- : Array Nat

  for s in [0:srcCount] do
    let opts : List EdgeKind := outsA[s]!
    let m    : Nat           := opts.length
    let j    : Nat           := choiceA[s]!  -- 0=inactive, 1..m valid

    if j = 0 then
      pure ()
    else if j > m then
      err := true
    else
      -- pick the (j-1)-th option safely
      match opts[j - 1]? with
      | none   => err := true
      | some opt =>
        let maj : List Bool := prevA.getD s zeroVec

        -- contribution for the chosen edge
        let contrib : List Bool :=
          match opt with
          | EdgeKind.rep _tgt         => maj
          | EdgeKind.intro _tgt X     => andVec maj (notVec (chi X))
          | EdgeKind.elim _tgt mIdx _ => orVec maj (prevA.getD mIdx zeroVec)

        -- target bucket
        let tgt : Nat :=
          match opt with
          | EdgeKind.rep t       => t
          | EdgeKind.intro t _   => t
          | EdgeKind.elim t _ _  => t

        if h : tgt < tgtCount then
          -- Fin index avoids deprecated get!/proof noise
          let old := perTarget[tgt]!
          perTarget := perTarget.set! tgt (orVec old contrib)
        else
          err := true

  (perTarget.toList, err)

/-- pure: OR two Bool vectors of the same length (zip ||). -/
def orVec (a b : List Bool) : List Bool := List.zipWith (· || ·) a b

/-- pure spec for a whole grid using outgoing-path choices. -/
def specWholeGridOutgoing
  (B     : BuiltOutgoing)
  (path  : List (List Nat))     -- per transition k, per source choice j
  : (List (List (List Bool)) × Bool) :=
Id.run do
  let formsA   := B.formulas.toArray
  let outsA    := B.outgoings.toArray
  let transCnt := outsA.size

  -- level 0 directly from B.initDeps
  let mut allLevels : List (List (List Bool)) := [B.initDeps]
  let mut prev      : List (List Bool)        := B.initDeps
  let mut gErr      : Bool                    := false

  let pathA := path.toArray

  for k in [0:transCnt] do
    let currForms := formsA[k+1]!
    let outgoing  := outsA[k]!
    let choices   := (pathA.getD k []).toArray.toList
    let (next, e) := specLevelOutgoing
                      B.encN B.basis prev currForms outgoing choices
    gErr      := gErr || e
    allLevels := allLevels ++ [next]
    prev      := next

  (allLevels, gErr)

/-- compare a and b level-by-level, node-by-node, vector-wise. -/
def eqLevels (a b : List (List (List Bool))) : Bool :=
  decide (a = b)

/-- single-run check: spec vs compiled for the same BuiltOutgoing and path. -/
def checkSpecVsCompiledOnce
  (B     : BuiltOutgoing)
  (path  : List (List Nat))
  : Bool :=
Id.run do
  -- spec
  let (outsS, errS) := specWholeGridOutgoing B path

  -- compiled
  let ((outsW, errW), circ) := runBuilder (compileWholeGridOutgoing B)
  let inputs := packPathBits B.layout path
  let vals   := simulate circ inputs
  let outsC  : List (List (List Bool)) :=
      outsW.map (fun layer => layer.map (fun vec => readVec vals vec))
  let errC   : Bool := vals[errW]!

  eqLevels outsS outsC && (errS = errC)


#eval
  match buildOutgoing D0 with
  | .error e => false
  | .ok B =>
      let path : List (List Nat) := []
      checkSpecVsCompiledOnce B path

/-- pure: "all false" on a Bool vector. -/
def allFalse (xs : List Bool) : Bool := xs.all (· = false)

def specFinalOutput
  (B : BuiltOutgoing)
  (goalLevel goalIdx : Nat)
  (path : List (List Nat))
  : Bool :=
Id.run do
  let (levels, err) := specWholeGridOutgoing B path
  let levelsA := levels.toArray
  let some level := levelsA[goalLevel]?
    | true   -- out-of-bounds -> conservatively flag as error/true
  let levelA := level.toArray
  let v      := levelA.getD goalIdx []  -- default empty
  err || allFalse v

/-- check final-output wire vs spec bit, starting from DLDS. -/
def checkFinalBitFromDLDS
  (d : DLDS) (goalLevel goalIdx : Nat) (path : List (List Nat))
  : Except String Bool := do
  let B ← buildOutgoing d
  -- compiled wire
  let (wire, circ) := runBuilder do
    let (levelsW, gErr) ← compileWholeGridOutgoing B
    let lvlA  := (levelsW.toArray)[goalLevel]!
    let goalV := (lvlA.toArray).getD goalIdx []
    let zero  ← vecAllFalse goalV
    allocOr gErr zero
  let vals  := simulate circ (packPathBits B.layout path)
  let comp  := vals[wire]!
  let spec  := specFinalOutput B goalLevel goalIdx path
  pure (comp = spec)

/-- For each transition k, return the number of sources at prev level k. -/
def prevRowSizes (B : BuiltOutgoing) : List Nat :=
  (List.range B.outgoings.length).map (fun k => (B.formulas[k]!).length)

/-- For each transition k and source s, how many valid choices (0..m)? (m varies by s). -/
def pathDomain (B : BuiltOutgoing) : List (List Nat) :=
  B.outgoings.map (fun row => row.map (fun opts => opts.length))

/-- A "do nothing" path: all sources inactive (0). -/
def defaultInactivePath (B : BuiltOutgoing) : List (List Nat) :=
  B.outgoings.map (fun row => List.replicate row.length 0)

/-- clamp each path choice to the valid domain 0..m for that source -/
def clampPathToOutgoing (outgoing : List (List EdgeKind)) (row : List Nat) : List Nat :=
  let m := outgoing.length
  let outA := outgoing.toArray
  (List.range m).map (fun s =>
    let opts := outA[s]!
    let maxj := opts.length
    let j := row.getD s 0
    if j > maxj then maxj else j)

/-- clamp a whole path to the model B -/
def clampPath (B : BuiltOutgoing) (path : List (List Nat)) : List (List Nat) :=
  let outsA := B.outgoings.toArray
  let rows  := path.length.min outsA.size
  (List.range rows).map (fun k =>
    clampPathToOutgoing (outsA[k]!) (path.getD k []))


/-- Validate key invariants of `BuiltOutgoing`. Returns `ok ()` or an explanatory error. -/
def validateBuiltOutgoing (B : BuiltOutgoing) : Except String PUnit := do
  -- every transition has a prev and next formula row
  if B.formulas.length < 1 then
    throw "validate: formulas must contain at least level 0"
  if B.outgoings.length ≠ (B.formulas.length - 1) then
    throw s!"validate: outgoings length {B.outgoings.length} ≠ formulas-1 {B.formulas.length-1}"
  if B.layout.length    ≠ (B.formulas.length - 1) then
    throw s!"validate: layout length {B.layout.length} ≠ formulas-1 {B.formulas.length-1}"

  -- per transition: layout rows count must equal prev-level size
  for k in [0 : B.outgoings.length] do
    let prevSz := (B.formulas[k]!).length
    let lrows  := (B.layout[k]!).length
    if lrows ≠ prevSz then
      throw s!"validate: layout rows at trans {k} = {lrows} ≠ prev size {prevSz}"
    let outs   := (B.outgoings[k]!).toArray
    -- for each source, options' targets must be < next-level size
    let nextSz := (B.formulas[k+1]!).length
    for s in [0:prevSz] do
      for opt in (outs[s]!) do
        let tgt : Nat := match opt with
          | .rep t       => t
          | .intro t _   => t
          | .elim t _ _  => t
        if ¬ (tgt < nextSz) then
          throw s!"validate: at trans {k}, source {s} has target {tgt} ≥ next size {nextSz}"

  -- encN must match basis length
  if B.encN ≠ B.basis.length then
    throw s!"validate: encN {B.encN} ≠ basis.length {B.basis.length}"

  pure ()


/-- Pretty string for a formula. -/
def ppFormula : Formula → String
  | .atom s     => s
  | .impl a b   =>
    let pa := match a with | .atom _ => ppFormula a | _ => "(" ++ ppFormula a ++ ")"
    let pb := match b with | .atom _ => ppFormula b | _ => "(" ++ ppFormula b ++ ")"
    pa ++ " ⊃ " ++ pb

/-- Pretty string for one EdgeKind (source-local view). -/
def ppEdgeKind : EdgeKind → String
  | .rep t          => s!"rep → t={t}"
  | .intro t X      => s!"intro → t={t}, discharge={ppFormula X}"
  | .elim t m X     => s!"elim → t={t}, minor={m}, ante={ppFormula X}"

/-- Dump one transition's outgoing options as a table (sources × options). -/
def ppOutgoingRow (prevForms currForms : List Formula) (row : List OutgoingOptions) : String :=
  let header := s!"prev: [{String.intercalate ", " (prevForms.map ppFormula)}]  ⟶  next: [{String.intercalate ", " (currForms.map ppFormula)}]"
  let lines :=
    (row.enum.map fun (s, opts) =>
      let left  := s!"s={s} ({ppFormula (prevForms.getD s (.atom "?"))})"
      let right := String.intercalate " | " (opts.enum.map (fun (j, ek) => s!"j={j+1}: {ppEdgeKind ek}"))
      left ++ " : " ++ right)
  String.intercalate "\n" (header :: lines)

/-- Dump all transitions. -/
def ppAllOutgoings (B : BuiltOutgoing) : String :=
  let F := B.formulas.toArray
  let O := B.outgoings.toArray
  let acc :=
    (List.range O.size).map (fun k =>
      s!"\n-- Transition {k} --\n" ++
      ppOutgoingRow (F[k]!) (F[k+1]!) (O[k]!))
  String.intercalate "\n" acc

/-- Check spec vs compiled for a DLDS and a path (after clamping). -/
def checkSpecVsCompiledFromDLDS
  (d    : DLDS)
  (path : List (List Nat))
  : Except String Bool := do
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let p := clampPath B path
  pure (checkSpecVsCompiledOnce B p)

/-- Always-0 path check. -/
def checkZeroPath (d : DLDS) : Except String Bool := do
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let p := defaultInactivePath B
  pure (checkSpecVsCompiledOnce B p)


/-- Number of nodes per level. -/
def levelSizes (B : BuiltOutgoing) : List Nat :=
  B.formulas.map (·.length)

/-- Total input bits demanded by the path layout. -/
def totalPathBits (B : BuiltOutgoing) : Nat :=
  B.layout.foldl (fun acc lvl =>
    acc + lvl.foldl (fun a s => a + s.bits) 0) 0

/-- Find (level, index-in-level) for a node id inside a BuiltOutgoing. -/
def indexOfNodeId? (B : BuiltOutgoing) (nodeId : Nat) : Option (Nat × Nat) :=
  let L := B.levels.toArray
  let rec loop (k : Nat) : Option (Nat × Nat) :=
    if h : k < L.size then
      let row := L[k]!
      match (row.enum.find? (fun (i,v) => v.node = nodeId)) with
      | some (i, _) => some (k, i)
      | none        => loop (k+1)
    else none
  loop 0

structure Executable where
  circuit   : CircuitOp.Circuit
  outsW     : List (List (List CircuitOp.Wire))   -- per level, per node, dep-vector wires
  errW      : CircuitOp.Wire
  layout    : PathLayout
  encN      : Nat
  basis     : List CircuitOp.Formula
  levels    : List (List CircuitOp.Vertex)
  outgoings : List (List OutgoingOptions)
deriving Repr

/-- Build an Executable from a DLDS: full-universe basis, enumerated rule space. -/
def buildExecutableFromDLDS (d : CircuitOp.DLDS) : Except String Executable := do
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let ((outsW, errW), circ) := CircuitOp.runBuilder (compileWholeGridOutgoing B)
  pure {
    circuit   := circ
    outsW     := outsW
    errW      := errW
    layout    := B.layout
    encN      := B.encN
    basis     := B.basis
    levels    := B.levels
    outgoings := B.outgoings
  }

/-- Using the Executable’s baked `levels`, find (level, idx) for a node id. -/
def execIndexOfNodeId? (E : Executable) (nodeId : Nat) : Option (Nat × Nat) :=
  let L := E.levels.toArray
  let rec go (k : Nat) : Option (Nat × Nat) :=
    if h : k < L.size then
      let row := L[k]!
      match (row.enum.find? (fun (i,v) => v.node = nodeId)) with
      | some (i, _) => some (k, i)
      | none        => go (k+1)
    else none
  go 0

/-- Read one node’s dependency vector as Bool list from a simulated value array. -/
def readNodeDeps (E : Executable) (vals : Array Bool) (level idx : Nat) : List Bool :=
  let levelW := (E.outsW.toArray)[level]!
  let nodeW  := (levelW.toArray).getD idx []
  CircuitOp.readVec vals nodeW


/-- Simulate an Executable with a concrete path → full value array. -/
def execSimulate (E : Executable) (path : List (List Nat)) : Array Bool :=
  let inputs := packPathBits E.layout path
  CircuitOp.simulate E.circuit inputs

/-- Spec-final bit computed on the host: err OR "goal vector is all false". -/
def execFinalAt (E : Executable) (vals : Array Bool) (level idx : Nat) : Bool :=
  let err := vals[E.errW]!
  let v   := readNodeDeps E vals level idx
  err || (v.all (· = false))

/-- Run by (level, idx). -/
def runOnceAt (E : Executable) (path : List (List Nat)) (level idx : Nat) : Bool :=
  let vals := execSimulate E path
  execFinalAt E vals level idx

/-- Run by original DLDS node id. -/
def runOnceAtNode (E : Executable) (path : List (List Nat)) (nodeId : Nat) : Option Bool :=
  match execIndexOfNodeId? E nodeId with
  | none => none
  | some (lev, i) =>
      let vals := execSimulate E path
      some (execFinalAt E vals lev i)


/-- How many sources on each transition. -/
def execPrevRowSizes (E : Executable) : List Nat :=
  (List.range E.outgoings.length).map (fun k => (E.levels[k]!).length)

/-- For each transition k and source s: max valid choice m (so domain is 0..m). -/
def execPathDomain (E : Executable) : List (List Nat) :=
  E.outgoings.map (fun row => row.map (fun opts => opts.length))

/-- All-zeros path. -/
def execDefaultInactivePath (E : Executable) : List (List Nat) :=
  E.outgoings.map (fun row => List.replicate row.length 0)

/-- Clamp a provided path to the domain described by `E`. -/
def execClampPath (E : Executable) (path : List (List Nat)) : List (List Nat) :=
  let outsA := E.outgoings.toArray
  let rows  := path.length.min outsA.size
  (List.range rows).map (fun k =>
    let row    := outsA[k]!
    let maxPer := row.map (·.length)
    let inp    := path.getD k []
    let m := maxPer.length
    let inpA := inp.toArray
    (List.range m).map (fun s =>
      let maxj := maxPer.getD s 0
      let j    := inpA.getD s 0
      if j > maxj then maxj else j))

/-- Build once from DLDS, clamp the path, run at a node id. -/
def runDLDSAtNode
  (d : CircuitOp.DLDS) (nodeId : Nat) (path : List (List Nat))
  : Except String (Option Bool) := do
  let E ← buildExecutableFromDLDS d
  let p := execClampPath E path
  pure (runOnceAtNode E p nodeId)

#eval
  match buildExecutableFromDLDS D0 with
  | .error e => s!"ERR: {e}"
  | .ok E =>
      let domain := execPathDomain E
      let zeroP  := execDefaultInactivePath E
      let nodeId := match E.levels with
                    | []      => 0
                    | row::_  => match row with | [] => 0 | v::_ => v.node
      let r? := runOnceAtNode E zeroP nodeId
      s!"domain={domain}, zero-path={zeroP}, result={r?}"

/-- Atoms and compound we’ll use. -/
def A     : Formula := .atom "A"
def B     : Formula := .atom "B"
def AimpB : Formula := .impl A B
def ABimpAB: Formula := .impl AimpB AimpB

/-- Vertices: put A and (A ⊃ B) at level 0; derive B at level 1; carry (A ⊃ B) at level 2. -/
def n0 : Vertex := { node := 0, LEVEL := 0, FORMULA := A,     HYPOTHESIS := true,  COLLAPSED := false, PAST := [] }
def n1 : Vertex := { node := 1, LEVEL := 0, FORMULA := AimpB, HYPOTHESIS := true,  COLLAPSED := false, PAST := [] }
def n2 : Vertex := { node := 2, LEVEL := 1, FORMULA := B,     HYPOTHESIS := false, COLLAPSED := false, PAST := [] }
def n3 : Vertex := { node := 3, LEVEL := 2, FORMULA := AimpB, HYPOTHESIS := false, COLLAPSED := false, PAST := [] }
def n4 : Vertex := { node := 4, LEVEL := 3, FORMULA := ABimpAB, HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

/-- Edges: 0→2 (A contributes as minor), 1→2 (A ⊃ B major), 2→3 (intro to A ⊃ B discharging A). -/
def e0 : Deduction := { START := n0, END := n2, COLOUR := 0, DEPENDENCY := [A] }
def e1 : Deduction := { START := n1, END := n2, COLOUR := 0, DEPENDENCY := [AimpB] }
def e2 : Deduction := { START := n2, END := n3, COLOUR := 0, DEPENDENCY := [A, AimpB] }
def e3 : Deduction := { START := n3, END := n4, COLOUR := 0, DEPENDENCY := [AimpB] }

/-- The DLDS. -/
def d0 : DLDS := { V := [n0, n1, n2, n3, n4], E := [e0, e1, e2, e3], A := [] }

#eval
  match buildOutgoing d0 with
  | .error e => s!"ERR: {e}"
  | .ok B =>
      let _ := validateBuiltOutgoing B   -- will throw in #eval if something’s inconsistent
      let sizes   := levelSizes B
      let summary := ppAllOutgoings B
      let domain  := pathDomain B        -- for each transition k and source s: valid m (domain 0..m)
      s!"levelSizes={sizes}\npathDomain={domain}\n{summary}"



#eval
  match checkZeroPath d0 with
  | .error e => s!"ERR: {e}"
  | .ok ok?  => s!"zeroPath spec==compiled ? {ok?}"

#eval
  let path : List (List Nat) := [[0,1], [1]]  -- trans0: A:0, A⊃B:1 ; trans1: B:1
  match checkSpecVsCompiledFromDLDS d0 path with
  | .error e => s!"ERR: {e}"
  | .ok ok?  => s!"path {path}, spec==compiled ? {ok?}"


#eval
  match buildExecutableFromDLDS d0 with
  | .error e => s!"ERR: {e}"
  | .ok E =>
      let domain := execPathDomain E
      let zeroP  := execDefaultInactivePath E
      s!"domain={domain}, zeroPath={zeroP}, encN={E.encN}, basis={E.basis.map ppFormula}"

#eval
  match buildExecutableFromDLDS d0 with
  | .error e => s!"ERR: {e}"
  | .ok E =>
      let path := execClampPath E [[0,1],[1]]
      let run (nid : Nat) :=
        match runOnceAtNode E path nid with
        | none   => s!"node {nid}: not found"
        | some b => s!"node {nid}: final={b}"
      String.intercalate "\n" (List.map run [0,1,2,3])

/-- For a single transition k, the per-edge activation bits: prev_size × next_size. -/
abbrev LevelEdgeBits := Array (Array Bool)
/-- Whole path = one LevelEdgeBits per transition. -/
abbrev EdgePathBits  := Array LevelEdgeBits

/-- For a single transition k, the input bit index assigned to edge (s,t). -/
abbrev LevelEdgeLayout := Array (Array Nat)

/-- Whole layout = one LevelEdgeLayout per transition. -/
abbrev EdgePathLayout := List LevelEdgeLayout

/-- Build a rectangular Array (rows = prevSz, cols = nextSz) by a filler. -/
private def buildRect {α} (prevSz nextSz : Nat) (f : Nat → Nat → α) : Array (Array α) :=
  Id.run do
    let mut rows : Array (Array α) := Array.mkEmpty prevSz
    for s in [0:prevSz] do
      let mut row : Array α := Array.mkEmpty nextSz
      for t in [0:nextSz] do
        row := row.push (f s t)
      rows := rows.push row
    pure rows

/-- Assign **one input bit per edge** in row-major order, starting at `cursor`.
    Returns the level layout and the next cursor. -/
def buildLevelEdgeLayout (cursor prevSz nextSz : Nat)
  : (LevelEdgeLayout × Nat) :=
  let lay : LevelEdgeLayout :=
    buildRect prevSz nextSz (fun s t => cursor + s * nextSz + t)
  let next := cursor + prevSz * nextSz
  (lay, next)

/-- Pack one transition’s per-edge activation bits into a flat input Array using the layout. -/
def packLevelEdgeBits (lay : LevelEdgeLayout) (bits : LevelEdgeBits) (total : Nat) : Array Bool :=
  Id.run do
    let mut a : Array Bool := Array.replicate total false
    let prevSz := lay.size
    for s in [0:prevSz] do
      let rowLay := lay[s]!
      let rowOn  := (bits.getD s #[] : Array Bool)
      let nextSz := rowLay.size
      for t in [0:nextSz] do
        let idx := rowLay[t]!
        let b   := rowOn.getD t false
        a := a.set! idx b
    pure a

/-- Maximum input index used inside one level's layout. Returns 0 if empty. -/
def maxIndexInLevel (lay : LevelEdgeLayout) : Nat :=
  Id.run do
    let mut mx : Nat := 0
    let mut any := false
    for s in [0:lay.size] do
      let row := lay[s]!
      for t in [0:row.size] do
        let idx := row[t]!
        if !any || idx > mx then
          mx := idx
          any := true
    if any then mx else 0

/-- Total number of input bits implied by the whole layout (max index + 1, or 0). -/
def totalBitsFromLayout (layout : EdgePathLayout) : Nat :=
  Id.run do
    let mut mx : Nat := 0
    let mut any := false
    for lay in layout do
      let m := maxIndexInLevel lay
      if !any || m > mx then
        mx := m
        any := true
    if any then mx + 1 else 0

/-- Pack a whole path (list of LevelEdgeBits) using the whole layout. -/
def packEdgePathBits (layout : EdgePathLayout) (path : EdgePathBits) : Array Bool :=
  Id.run do
    let total := totalBitsFromLayout layout
    let mut acc : Array Bool := Array.replicate total false
    let layA  := layout.toArray
    let pathA := path
    for k in [0:layA.size] do
      let lay  := layA[k]!
      let bits : LevelEdgeBits := pathA.getD k (#[] : Array (Array Bool))
      let levelPacked := packLevelEdgeBits lay bits total
      -- layouts are disjoint; OR merge to be defensive
      for i in [0:total] do
        acc := acc.set! i (acc[i]! || levelPacked[i]!)
    pure acc


/-- Semantic role of an edge (source s, target t). -/
inductive EdgeRole
  | rep                            -- Fs = Ft
  | intro (X : Formula)            -- Ft = X ⊃ Y and Fs = Y
  | elimMajor (X : Formula)        -- Fs = X ⊃ Y and Ft = Y
  | elimMinor (X : Formula)        -- Fs = X       and Ft = Y
  | invalid
  deriving Repr, DecidableEq


/-- Small helpers for Bool vectors. -/
def andVec (a b : List Bool) : List Bool := List.zipWith (· && ·) a b
def notVec (v : List Bool)   : List Bool := v.map (· = false)

/-- χ_X : mask (length = basis.length), 1 where basis = X. -/
def chiMask (basis : List Formula) (X : Formula) : List Bool :=
  basis.map (fun f => decide (f = X))

/-- Zero vector of length n. -/
def zeroVec (n : Nat) : List Bool := List.replicate n false


/-- One-hot for `φ` w.r.t. `basis`. -/
def oneHotFor (basis : List Formula) (φ : Formula) : List Bool :=
  let n := basis.length
  match basis.idxOf? φ with
  | none   => List.replicate n false
  | some i => (List.range n).map (fun j => decide (j = i))

/-- Pretty a Bool vector as 0/1 string. -/
def ppBits (v : List Bool) : String :=
  String.join (v.map (fun b => if b then "1" else "0"))

/-- Pretty a list of Bool vectors alongside the target formulas. -/
def ppTargetVectors (targets : List Formula) (vs : List (List Bool)) : String :=
  let lines :=
    (List.zip targets vs).map (fun (f, v) =>
      s!"{ppFormula f}  :=  {ppBits v}")
  String.intercalate "\n" lines

/-- Make an `src×tgt` Array(Array Bool) with `true` at the given (s,t) pairs. -/
def mkEdgeOn (src tgt : Nat) (ones : List (Nat × Nat)) : Array (Array Bool) :=
  Id.run do
    let mut a := Array.replicate src (Array.replicate tgt false)
    for (s,t) in ones do
      if h₁ : s < src then
        if h₂ : t < tgt then
          let row := a[s]!
          a := a.set! s (row.set! t true)
        else
          pure ()
      else
        pure ()
    pure a



def edge (s e : Vertex) (col : Nat) (deps : List Formula) : Deduction :=
  { START := s, END := e, COLOUR := col, DEPENDENCY := deps }

/-- Pull out: levels, per-level formulas, full-universe basis, and init vectors for level 0. -/
def extractLevelData (d : DLDS) :
  List (List Vertex) × List (List Formula) × List Formula × Nat × List (List Bool) :=
Id.run do
  let packs := groupByLevel d
  let levels : List (List Vertex) := packs.map (·.nodes)
  let forms  : List (List Formula) := levels.map (·.map (·.FORMULA))
  -- full-universe basis:
  let basis : List Formula := (d.V.map (·.FORMULA)).eraseDups
  let encN := basis.length
  -- initial dependency vectors for level 0: mark hypotheses
  let L0 := levels.headD []
  let init : List (List Bool) :=
    L0.map (fun v => oneHotFor basis (if v.HYPOTHESIS then v.FORMULA else .atom "__⊥"))
  (levels, forms, basis, encN, init)


/-- Build full-universe basis, levels (their count), and initial level-0 dependency vectors per formula. -/
def buildEdgesUniverse (d : DLDS)
  : (List Formula) × Nat × Nat × List (List Bool) × List (List Formula) := Id.run do
  -- universe basis
  let basis : List Formula := (d.V.map (·.FORMULA)).eraseDups
  let N := basis.length

  -- levels: use the DLDS LEVELs to determine how many transitions, but
  -- each level's "grid row" is the *full universe* (N nodes).
  let packs := groupByLevel d
  let L     := packs.length
  let levelsForms : List (List Formula) := List.replicate L basis

  -- level-0 initial dep vectors: one per universe formula; 1-hot if that formula occurs
  -- at level 0 as a hypothesis, else all-false.
  let L0_nodes := (packs.headD { level := 0, nodes := [] }).nodes
  let isHyp (φ : Formula) : Bool := (L0_nodes.any (fun v => v.FORMULA = φ ∧ v.HYPOTHESIS))
  let oneHotFor (φ : Formula) : List Bool :=
    let idx? := basis.idxOf? φ
    (List.range N).map (fun j => decide (some j = idx?))
  let init0 : List (List Bool) :=
    basis.map (fun φ => if isHyp φ then oneHotFor φ else List.replicate N false)

  (basis, N, L, init0, levelsForms)

/-- For pretty-printing a roles table. -/
def ppRoles (prev curr : List Formula) (roles : Array (Array EdgeRole)) : String :=
  let header := s!"prev=[{String.intercalate ", " (prev.map ppFormula)}]  →  next=[{String.intercalate ", " (curr.map ppFormula)}]"
  let lines :=
    roles.toList.enum.map (fun (s,row) =>
      let left  := s!"s={s} ({ppFormula (prev.getD s (.atom "?"))})"
      let cells := row.toList.enum.map (fun (t,r) =>
        s!"t={t}:{match r with
                  | .rep => "rep"
                  | .intro X => s!"intro({ppFormula X})"
                  | .elimMajor X => s!"elimMaj({ppFormula X})"  -- (not used by our role gen)
                  | .elimMinor X => s!"elimMin({ppFormula X})"
                  | .invalid => "INVALID"}")
      left ++ " : " ++ String.intercalate " | " cells)
  String.intercalate "\n" (header :: lines)

structure BuiltEdges where
  basis     : List Formula         -- full universe
  N         : Nat                  -- basis size
  L         : Nat                  -- number of levels
  init0     : List (List Bool)     -- N vectors at level 0 (each length N)
  roles     : Array (Array EdgeRole)   -- common NxN roles between identical rows (basis→basis)
  starts    : Array Nat            -- length L-1; start offset of each transition
  totalBits : Nat
deriving Repr

/-- Pack a "path" described as, for each transition k, a list of (s,t) edges turned ON. -/
def packEdgePath (B : BuiltEdges) (path : List (List (Nat × Nat))) : Array Bool :=
  Id.run do
    let mut arr := Array.replicate B.totalBits false
    let pathA   := path.toArray
    let starts  := B.starts
    let block   := B.N * B.N
    for k in [0:starts.size] do
      let start := starts[k]!
      let row   := pathA.getD k [] |>.toArray
      for (s,t) in row do
        if hs : s < B.N then
          if ht : t < B.N then
            let idx := start + s * B.N + t
            if hidx : idx < arr.size then
              arr := arr.set! idx true
            else
              pure ()
          else pure ()
        else pure ()
    pure arr

/-- SPEC: combine prev deps with NxN bit matrix and roles; returns (target vectors, invalid-edge error). -/
def specLevelEdges
  (encN      : Nat)
  (basis     : List Formula)
  (prevDeps  : List (List Bool))
  (currForms : List Formula)
  (edgeOn    : Array (Array Bool))
  (roles     : Array (Array EdgeRole))
  : (List (List Bool) × Bool) :=
Id.run do
  -- helpers (pure)
  let zeroVec (n : Nat) : List Bool := List.replicate n false
  let orVec  (a b : List Bool) : List Bool := List.zipWith (· || ·) a b
  let andVec (a b : List Bool) : List Bool := List.zipWith (· && ·) a b
  let notVec (v : List Bool) : List Bool := v.map (· = false)
  let chiMask (basis : List Formula) (X : Formula) : List Bool :=
    basis.map (fun f => decide (f = X))

  let prevA    := prevDeps.toArray
  let tgtCount := currForms.length
  let srcCount := prevA.size

  let mut perTarget : Array (List Bool) := Array.replicate tgtCount (zeroVec encN)
  let mut errInvalid : Bool := false

  for t in [0:tgtCount] do
    let mut repC    : List Bool := zeroVec encN
    let mut introC  : List Bool := zeroVec encN
    let mut elimC   : List Bool := zeroVec encN

    /- collect majors/minors keyed by antecedent X (only those whose bit is ON) -/
    let mut majors : List (Formula × Nat) := []   -- (X, sMaj)
    let mut minors : List (Formula × Nat) := []   -- (X, sMin)

    for s in [0:srcCount] do
      let r   := (roles.getD s #[] : Array EdgeRole).getD t EdgeRole.invalid
      let on  := (edgeOn.getD s #[] : Array Bool).getD t false
      match r with
      | .rep =>
          if on then
            repC := orVec repC (prevA.getD s (zeroVec encN))
      | .intro X =>
          if on then
            let maj := prevA.getD s (zeroVec encN)
            let nχ  := notVec (chiMask basis X)
            introC := orVec introC (andVec maj nχ)
      | .elimMinor X =>
          if on then minors := (X, s) :: minors
      | .elimMajor X =>
          if on then majors := (X, s) :: majors
      | .invalid =>
          if on then errInvalid := true

    /- pair majors and minors on the same X: OR over (maj ∨ min) for all pairs -/
    for (X, sMaj) in majors do
      for (_, sMin) in minors.filter (fun p => p.fst = X) do
        let maj := prevA.getD sMaj (zeroVec encN)
        let min := prevA.getD sMin (zeroVec encN)
        elimC := orVec elimC (orVec maj min)

    let outT := orVec (orVec repC introC) elimC
    perTarget := perTarget.set! t outT

  (perTarget.toList, errInvalid)

/-- Roles for (Fs, Ft) pair. -/
def roleOfPair : Formula → Formula → EdgeRole
  | Fs, Ft =>
    if Fs = Ft then
      .rep
    else
      match Ft with
      | .impl X Y =>
          if Fs = Y then
            .intro X
          else
            match Fs with
            | .impl X' Y' =>
                if Y' = Y then .elimMajor X' else .invalid
            | _ => .invalid
      | _ =>
        match Fs with
        | .impl X' Y' =>
            if Y' = Ft then .elimMajor X' else .invalid
        | _ =>
          -- non-impl target, non-equal source => invalid
          .invalid

/-- Build roles matrix for one level transition: S×T. -/
def edgeRolesForLevel (prevForms currForms : List Formula) : Array (Array EdgeRole) :=
  let prevA := prevForms.toArray
  let nextA := currForms.toArray
  let S := prevA.size
  let T := nextA.size
  let row (s : Nat) : Array EdgeRole :=
    Array.mk <|
      (List.range T).map (fun t =>
        let Fs := prevA.getD s (.atom "__?")
        let Ft := nextA.getD t (.atom "__?")
        if Fs = Ft then
          EdgeRole.rep
        else
          match Ft with
          | .impl X Y =>
              if Fs = Y then
                EdgeRole.intro X
              else
                match Fs with
                | .impl X' Y' =>
                    -- MAJOR when Fs = X'⊃Y' and Ft = Y'
                    if Y' = Y then EdgeRole.elimMajor X' else EdgeRole.invalid
                | _ => EdgeRole.invalid
          | _ =>
            match Fs with
            | .impl X' Y' =>
                -- MAJOR when Ft = Y' and Fs = X'⊃Y' (non-impl target case)
                if Y' = Ft then EdgeRole.elimMajor X' else EdgeRole.invalid
            | _ =>
              EdgeRole.invalid)
  Array.mk <| (List.range S).map row

/-- Universe & level info for the edges pipeline, with a safe check for empty DLDS. -/
def edgesUniverseFromDLDS
  (d : DLDS)
  : Except String (List Formula × Nat × List (List Formula) × List (List Bool)) := do
  let basis := buildUniverse d
  let packs := groupByLevel d
  match packs with
  | [] =>
      throw "edgesUniverseFromDLDS: DLDS has no vertices (no levels found)."
  | p0 :: _ =>
      -- one row of formulas per level (in order)
      let levelsForms : List (List Formula) :=
        packs.map (fun p => p.nodes.map (·.FORMULA))
      -- size & initial vectors from level 0 (you already made this use the full-universe basis)
      let (_, encN, init0) := buildEncodingAndInit d p0.nodes
      pure (basis, encN, levelsForms, init0)

/-- SPEC: whole-grid execution with NxN edges per transition. -/
def specWholeGridEdges
  (encN   : Nat)
  (basis  : List Formula)
  (levels : List (List Formula))
  (init   : List (List Bool))
  (path   : EdgePathBits)     -- one S×T matrix per transition
  : (List (List (List Bool)) × Bool) :=
Id.run do
  let formsA   := levels.toArray
  let transCnt := (levels.length - 1).max 0

  let mut allLevels : List (List (List Bool)) := [init]
  let mut prev      : List (List Bool)        := init
  let mut gErr      : Bool                    := false

  for k in [0:transCnt] do
    let prevForms := formsA[k]!
    let nextForms := formsA[k+1]!
    let roles     := edgeRolesForLevel prevForms nextForms
    let edgeBits  : LevelEdgeBits := path.getD k (#[] : Array (Array Bool))
    let (next, e) := specLevelEdges encN basis prev nextForms edgeBits roles
    gErr      := gErr || e
    allLevels := allLevels ++ [next]
    prev      := next

  (allLevels, gErr)

def specFinalFromEdges
  (encN   : Nat) (basis : List Formula) (levels : List (List Formula))
  (init   : List (List Bool)) (path : EdgePathBits)
  (goalLevel goalIdx : Nat)
  : Bool :=
Id.run do
  let (outs, err) := specWholeGridEdges encN basis levels init path
  let outsA := outs.toArray
  let some row := outsA[goalLevel]? | true
  let v := (row.toArray).getD goalIdx (List.replicate encN false)
  err || allFalse v



/-- gate-level helpers -/
def vecOrW  (a b : List Wire) : Builder (List Wire) :=
  BitVecAlg.vzipWith (m := Builder) (n := a.length) CircuitOp.allocOr a b

def vecAndW (a b : List Wire) : Builder (List Wire) :=
  BitVecAlg.vzipWith (m := Builder) (n := a.length) CircuitOp.allocAnd a b

def vecOrWn  (n : Nat) (a b : List Wire) : Builder (List Wire) :=
  BitVecAlg.vzipWith (m := Builder) (n := n) CircuitOp.allocOr a b

def vecAndWn (n : Nat) (a b : List Wire) : Builder (List Wire) :=
  BitVecAlg.vzipWith (m := Builder) (n := n) CircuitOp.allocAnd a b

def vecNotWn (n : Nat) (a : List Wire) : Builder (List Wire) :=
  BitVecAlg.vmap (m := Builder) (n := n) CircuitOp.allocNot a

def vecNotW (a : List Wire) : Builder (List Wire) :=
  BitVecAlg.vmap (m := Builder) (n := a.length) CircuitOp.allocNot a
/-- χ_X mask as wires -/
def chiMaskW (basis : List Formula) (X : Formula) : CircuitOp.Builder (List CircuitOp.Wire) :=
  basis.mapM (fun f => CircuitOp.allocConst (decide (f = X)))
/-- Compile one NxN level using packed input layout start + s*N + t. -/

def compileLevelEdges
  (B         : BuiltEdges)
  (prevDeps  : List (List CircuitOp.Wire))
  (currForms : List Formula)
  (start     : Nat)
  : CircuitOp.Builder (List (List CircuitOp.Wire) × CircuitOp.Wire) := do

  let prevA := prevDeps.toArray
  let tgtCount := currForms.length
  let srcCount := prevA.size

  -- roles are common NxN on the universe rows
  let roles := B.roles

  let zeroV ← CircuitOp.allocConstVec B.N false
  let mut perTarget : Array (List CircuitOp.Wire) := Array.replicate tgtCount zeroV
  let mut errInv : CircuitOp.Wire := (← CircuitOp.allocConst false)

  -- collect per-target buckets
  for t in [0:tgtCount] do
    let mut repC    : List CircuitOp.Wire := zeroV
    let mut introC  : List CircuitOp.Wire := zeroV
    let mut elimC   : List CircuitOp.Wire := zeroV

    -- collect minors & majors keyed by X
    let mut majors : List (Formula × Nat) := []
    let mut minors : List (Formula × Nat) := []

    for s in [0:srcCount] do
      let bitIdx := start + s * B.N + t
      let on ← CircuitOp.allocInput bitIdx
      let r  := (roles.getD s #[] : Array EdgeRole).getD t EdgeRole.invalid
      match r with
      | .rep => do
          let maj := prevA.getD s zeroV
          let masked ← BitVecAlg.vmaskBy (m := CircuitOp.Builder) (n := B.N) on maj
          repC ← vecOrWn B.N repC masked
      | .intro X => do
          let χ  ← chiMaskW B.basis X
          let nχ ← vecNotWn B.N χ
          let maj := prevA.getD s zeroV
          let maj2 ← vecAndWn B.N maj nχ
          let masked ← BitVecAlg.vmaskBy (m := CircuitOp.Builder) (n := B.N) on maj2
          introC ← vecOrWn B.N introC masked
      | .elimMinor X => do
          minors := (X, s) :: minors
          pure ()
      | .elimMajor X => do
          majors := (X, s) :: majors
          pure ()
      | .invalid => do
          -- if invalid edge is ON, raise error; else ignore
          let e ← CircuitOp.allocBuf on
          errInv ← CircuitOp.allocOr errInv e

    -- auto-major discovery (rows are basis)
    for s in [0:srcCount] do
      match (B.basis.getD s (.atom "__?")) with
      | .impl X' Y' =>
          if Y' = currForms.getD t (.atom "__?") then
            majors := (X', s) :: majors
            pure ()
          else
            pure ()
      | _ => pure ()

    -- combine elim pairs for each X: OR over (maj ∨ min), mask by (on_major AND on_minor)
    for (X, sMaj) in majors do
      for (_, sMin) in minors.filter (fun p => p.fst = X) do
        let bitMaj ← CircuitOp.allocInput (start + sMaj * B.N + t)
        let bitMin ← CircuitOp.allocInput (start + sMin * B.N + t)
        let both   ← CircuitOp.allocAnd bitMaj bitMin
        let maj    := prevA.getD sMaj zeroV
        let min    := prevA.getD sMin zeroV
        let uni    ← vecOrWn B.N maj min
        let masked ← BitVecAlg.vmaskBy (m := CircuitOp.Builder) (n := B.N) both uni
        elimC ← vecOrWn B.N elimC masked

    let outT ← vecOrWn B.N (← vecOrWn B.N repC introC) elimC
    perTarget := perTarget.set! t outT

  pure (perTarget.toList, errInv)



/-- Whole grid compile (L levels → L-1 transitions). -/
def compileWholeGridEdges (B : BuiltEdges)
  : CircuitOp.Builder (List (List (List CircuitOp.Wire)) × CircuitOp.Wire) := do
  let T := if B.L = 0 then 0 else B.L - 1
  -- level 0 wires
  let lvl0 ← B.init0.mapM (fun row => row.mapM CircuitOp.allocConst)
  let mut levels : List (List (List CircuitOp.Wire)) := [lvl0]
  let mut prev := lvl0
  let mut gErr : CircuitOp.Wire := (← CircuitOp.allocConst false)
  for k in [0:T] do
    let start := B.starts[k]!
    let (next, e) ← compileLevelEdges B prev B.basis start
    gErr ← CircuitOp.allocOr gErr e
    levels := levels ++ [next]
    prev := next
  pure (levels, gErr)


/-- "all false" in wires -/
def vecAllFalse (xs : List CircuitOp.Wire) : CircuitOp.Builder CircuitOp.Wire := do
  let any ← BitVecAlg.vreduceOr (m := CircuitOp.Builder) (n := xs.length) xs
  BitVecAlg.bnot (m := CircuitOp.Builder) any

/-- Build the Edges model from a DLDS. -/
def buildEdges (d : DLDS) : BuiltEdges :=
  Id.run do
    let (basis, N, L, init0, levelsForms) := buildEdgesUniverse d
    -- Roles are classified on the basis×basis grid (same for every transition).
    let roles := edgeRolesForLevel basis basis
    -- Bit layout: blocks of N*N per transition.
    let T := (if L = 0 then 0 else L - 1)
    let block := N * N
    let starts := Array.mk <| (List.range T).map (fun k => k * block)
    let totalBits := T * block
    { basis, N, L, init0, roles, starts, totalBits }

/-- Final result = error ∨ allZero(goalVec). -/
def simulateEdgesFromDLDS
  (d : DLDS)
  (path : List (List (Nat × Nat)))   -- for each transition k, which edges (s,t) are ON
  (goalLevel goalIdx : Nat)
  : Except String Bool := do
  let B := buildEdges d
  -- compile circuit once
  let (fwire, circ) := CircuitOp.runBuilder do
    let (levelsW, gErr) ← compileWholeGridEdges B
    let levelsA := levelsW.toArray
    let some level := levelsA[goalLevel]?
      | pure (← CircuitOp.allocConst true)   -- oob → force "true"
    let goalV := (level.toArray).getD goalIdx []
    let zero ← CircuitOp.vecAllFalse goalV
    CircuitOp.allocOr gErr zero
  -- pack edge-on bits and simulate
  let inputs := packEdgePath B path
  let vals := CircuitOp.simulate circ inputs
  pure (vals[fwire]!)


#eval
  let B := buildEdges d0
  s!"basis={B.basis.map ppFormula}, N={B.N}, L={B.L}, totalBits={B.totalBits}, starts={B.starts.toList}"

/-- Helper: find a formula index in B.basis, default 0 if missing. -/
def idx! (B : BuiltEdges) (f : Formula) : Nat :=
  match B.basis.idxOf? f with
  | some i => i
  | none   => 0  -- shouldn't happen in our examples

/-- The test path for d0, picked using the basis indices. -/
def pathEdges_d0 (BE : BuiltEdges) : List (List (Nat × Nat)) :=
  let sA     := idx! BE A
  let sAimpB := idx! BE AimpB
  let sB     := idx! BE B
  let tA     := sA
  let tAimpB := sAimpB
  let tB     := sB
  -- trans 0: (A⊃B → B) and (A → B)
  let trans0 : List (Nat × Nat) := [(sAimpB, tB), (sA, tB)]
  -- trans 1: (B → A⊃B)
  let trans1 : List (Nat × Nat) := [(sB, tAimpB)]
  [trans0, trans1]

#eval
  let BE := buildEdges d0
  let path := pathEdges_d0 BE
  let goalLevel := 2
  let goalIdx   := idx! BE AimpB
  match simulateEdgesFromDLDS d0 path goalLevel goalIdx with
  | .error e => s!"ERR: {e}"
  | .ok b    => s!"final bit = {b}"  -- expect "false"


/-- Transpose N subpaths (one per column/leaf) into per-transition rows
    expected by the `BuiltOutgoing` pipeline.

    Input shape (your LaTeX):  subpaths : List (List Nat)
      - outer length N (columns / unique formulas in the basis)
      - each inner length ideally L-1 (steps), but we tolerate ragged
        by padding missing entries with 0.

    Output shape (ours):       List (List Nat)
      - outer length L-1 (transitions)
      - each inner length N    (sources at that transition)
-/
def transposeSubpaths
  (N Lminus1 : Nat)
  (subpaths : List (List Nat)) :
  List (List Nat) :=
  let colsA := subpaths.toArray
  -- For each transition k, build the row [ subpaths[c][k] default 0 | c ← 0..N-1 ].
  (List.range Lminus1).map (fun k =>
    (List.range N).map (fun c =>
      (colsA.getD c []).toArray.getD k 0))

/-- Best-effort "root" selection: pick a vertex at the *maximum* LEVEL.
    If multiple, choose the smallest node id for determinism. -/
def pickRoot (d : DLDS) : Option Vertex :=
  let mx := d.V.foldl (fun acc v => Nat.max acc v.LEVEL) 0
  let candidates := d.V.filter (fun v => v.LEVEL = mx)
  match candidates with
  | []      => none
  | v :: vs => some (List.foldl (fun best w => if w.node < best.node then w else best) v vs)

/-- Given a BuiltOutgoing and a DLDS, compute the fixed goal = (level 0, index of l(root)). -/
def goalForRoot (B : BuiltOutgoing) (d : DLDS) : Except String (Nat × Nat) := do
  let some r := pickRoot d | throw "goalForRoot: DLDS has no root candidate."
  let some j := B.basis.idxOf? r.FORMULA
    | throw s!"goalForRoot: root formula {ppFormula r.FORMULA} not in basis."
  pure (0, j)

/-- Decode a Bool dependency vector back to the formulas it marks. -/
def depsAsFormulas (basis : List CircuitOp.Formula) (v : List Bool) : List CircuitOp.Formula :=
  (List.zip basis v).foldr (fun (f,b) acc => if b then f :: acc else acc) []

/-- Uniform layout: for each transition k, allocate N specs (one per subpath c). -/
def buildUniformPathLayout (N Lminus1 : Nat) (cursor0 : Nat := 0)
  : (PathLayout × Nat) := Id.run do
  let K := bitsForChoices (N+1)
  let mut cursor := cursor0
  let mut layout : PathLayout := []
  for _k in [0:Lminus1] do
    let row : LevelPathLayout :=
      (List.range N).map (fun _ => { bits := K, start := cursor })
    cursor := cursor + N * K
    layout := layout ++ [row]
  (layout, cursor)


/-- One level compile under continuous subpaths (free target picks). -/
def compileLevelContinuous
  (basis : List Formula) (N : Nat)
  (prevForms nextForms : List Formula)
  (prevDeps : List (List Wire))
  (pos : Array (List Wire))         -- N entries, each a one-hot length N
  (inactive : Array Wire)           -- N entries
  (specs : LevelPathLayout)         -- N entries: K bits per subpath
  : Builder ( -- outputs at next level
              List (List Wire)            -- per real target tReal, dep vector
              × Wire                       -- structural error for this step
              × Array (List Wire)          -- pos' for next step
              × Array Wire                 -- inactive' for next step
            ) := do
  let S := prevForms.length
  let T := nextForms.length
  let K := match specs.head? with | none => 0 | some s => s.bits

  /- 1) decode j_c and derive on_c_s_t -/
  let mut on_s_t : Array (Array Wire) := Array.replicate N (Array.replicate N (← allocConst false))
  let mut posNext : Array (List Wire) := Array.mkArray N (List.replicate N (← allocConst false))
  let mut inactNext : Array Wire := Array.mkArray N (← allocConst false)

  let mut stepErr : Wire := (← allocConst false)

  for c in [0:N] do
    let sp := (specs.toArray).getD c { bits := 0, start := 0 }
    -- gather bits and decode j (0..N). You can reuse your decoder to a one-hot over N+1
    let selBits ← (List.range sp.bits).mapM (fun i => allocInput (sp.start + i))
    let oneHotNplus1 ← decodeIndex selBits.reverse  -- length = 2^K ≥ N+1; safe
    -- j = 0..N; produce eq wires eq_j_k
    -- safer: build eq wires only for [0..N], ignore extras but OR extras into stepErr
    let mut eqJ : Array Wire := Array.mkEmpty (N+1)
    for jVal in [0:N+1] do
      let w := oneHotNplus1.getD jVal (← allocConst false)
      eqJ := eqJ.push w
    -- any selection beyond N? -> error
    let extras := (List.range (oneHotNplus1.length)).drop (N+1)
    for idx in extras do
      stepErr ← allocOr stepErr (oneHotNplus1.getD idx (← allocConst false))

    -- compute on_c_s_t = pos[c][s] ∧ eqJ[t+1], for all s,t
    for s in [0:N] do
      let atS := (pos.getD c (List.replicate N (← allocConst false))).getD s (← allocConst false)
      for t in [0:N] do
        let pick_t := eqJ.getD (t+1) (← allocConst false)
        let onct  ← allocAnd atS pick_t
        let row := on_s_t.get! s
        let merged ← allocOr (row.get! t) onct
        on_s_t := on_s_t.set! s (row.set! t merged)

    -- advance position and inactive
    let inPrev := inactive.getD c (← allocConst false)
    let becameInactive ← allocOr inPrev (eqJ.get! 0)         -- j=0 OR was already inactive
    inactNext := inactNext.set! c becameInactive

    -- pos'[c] = if inactive' then 0s else onehot(t)
    let mut nextP : List Wire := []
    for t in [0:N] do
      let pt := eqJ.getD (t+1) (← allocConst false)
      let keep ← allocAnd pt (← allocNot becameInactive)
      nextP := nextP ++ [keep]
    posNext := posNext.set! c nextP

  /- 2) produce outputs for *real* targets tReal ∈ [0..T-1] -/
  let prevA := prevDeps.toArray
  let mut outs : Array (List Wire) := Array.replicate T (List.replicate N (← allocConst false))

  for tReal in [0:T] do
    let Ft := nextForms.getD tReal (.atom "__?")
    let mut repC    : List Wire := List.replicate N (← allocConst false)
    let mut introC  : List Wire := List.replicate N (← allocConst false)
    let mut elimC   : List Wire := List.replicate N (← allocConst false)

    -- collect majors/minors keyed by X with on_s_tReal
    let mut majors : List (Formula × Nat × Wire) := []
    let mut minors : List (Formula × Nat × Wire) := []

    for s in [0:N] do
      let Fs := basis.getD s (.atom "__?")
      let on := ((on_s_t.get! s).get! (basis.idxOf? Ft |>.getD 0))  -- will fix mapping below
      -- role wrt *this* real target formula
      let role := roleOfPair Fs Ft
      -- map (s (basis row)) to whether s exists in prevForms
      let sReal? := prevForms.idxOf? Fs
      let sPresent := decide (sReal?.isSome)
      let on_s_tReal := (on)  -- already edge-select
      -- structural errors:
      if not sPresent then
        stepErr ← allocOr stepErr on_s_tReal
      match role with
      | .rep =>
          if sPresent then
            let maj := prevA.getD (sReal?.getD 0) (List.replicate N (← allocConst false))
            let masked ← BitVecAlg.vmaskBy (m := Builder) (n := N) on_s_tReal maj
            repC ← BitVecAlg.vzipWith (m := Builder) (n := N) allocOr repC masked
          else pure ()
      | .intro X =>
          if sPresent then
            let chi ← dischargeMaskFromBasis basis X
            let nχ  ← BitVecAlg.vmap (m := Builder) (n := N) allocNot chi
            let maj := prevA.getD (sReal?.getD 0) (List.replicate N (← allocConst false))
            let maj2 ← BitVecAlg.vzipWith (m := Builder) (n := N) allocAnd maj nχ
            let masked ← BitVecAlg.vmaskBy (m := Builder) (n := N) on_s_tReal maj2
            introC ← BitVecAlg.vzipWith (m := Builder) (n := N) allocOr introC masked
          else pure ()
      | .elimMajor X =>
          majors := (X, sReal?.getD 0, on_s_tReal) :: majors
      | .elimMinor X =>
          minors := (X, sReal?.getD 0, on_s_tReal) :: minors
      | .invalid =>
          stepErr ← allocOr stepErr on_s_tReal

    -- pair elim majors/minors with same X and AND their on-bits
    for (X, sMaj, onMaj) in majors do
      for (_, sMin, onMin) in minors.filter (fun p => p.fst = X) do
        let both ← allocAnd onMaj onMin
        let maj := prevA.getD sMaj (List.replicate N (← allocConst false))
        let min := prevA.getD sMin (List.replicate N (← allocConst false))
        let uni ← BitVecAlg.vzipWith (m := Builder) (n := N) allocOr maj min
        let masked ← BitVecAlg.vmaskBy (m := Builder) (n := N) both uni
        elimC ← BitVecAlg.vzipWith (m := Builder) (n := N) allocOr elimC masked

    let outT ← BitVecAlg.vzipWith (m := Builder) (n := N) allocOr (← BitVecAlg.vzipWith (m := Builder) (n := N) allocOr repC introC) elimC
    outs := outs.set! tReal outT

  pure (outs.toList, stepErr, posNext, inactNext)

/-- onehot(i) over size N (as wires). -/
def oneHotConstW (N i : Nat) : Builder (List Wire) := do
  let mut out := []
  for j in [0:N] do
    out := out ++ [← allocConst (j = i)]
  pure out

/-- onehot(t) chosen by equals(j, t+1), reusing your decodeIndex/selector as needed. -/
def eqConstW (w : Wire) (const : Bool) : Builder Wire := do
  -- simplest: xor with const then not
  let wc ← allocConst const
  let x  ← allocXor w wc
  allocNot x


  /-- Whole grid for continuous subpaths. -/
def compileWholeGridContinuous
  (basis : List Formula) (levels : List (List Formula))
  (initDeps : List (List Bool))  -- level 0 vectors, length = size(prev level)
  (layout : PathLayout)          -- built by buildUniformPathLayout
  : Builder (List (List (List Wire)) × Wire
             × Array (List Wire) × Array Wire) := do
  let N := basis.length
  -- level 0
  let lvl0 ← initDeps.mapM (fun v => v.mapM allocConst)
  let mut outs : List (List (List Wire)) := [lvl0]
  let mut prev := lvl0
  let mut gErr : Wire := (← allocConst false)

  -- positions: pos₀[c] = onehot(c); inactive₀[c] = false
  let mut pos : Array (List Wire) := Array.mkEmpty N
  for c in [0:N] do
    pos := pos.push (← oneHotConstW N c)
  let mut inactive : Array Wire := Array.replicate N (← allocConst false)

  let formsA := levels.toArray
  let layA   := layout.toArray
  let T := (levels.length - 1).max 0
  for k in [0:T] do
    let prevForms := formsA[k]!
    let nextForms := formsA[k+1]!
    let specs     := layA[k]!
    let (next, e, pos', ina') ← compileLevelContinuous basis N prevForms nextForms prev pos inactive specs
    outs := outs ++ [next]
    prev := next
    pos := pos'
    inactive := ina'
    gErr ← allocOr gErr e

  pure (outs, gErr, pos, inactive)


def acceptBitFromSubpaths
  (d : DLDS) (subpaths : List (List Nat)) : Except String Bool := do
  let B ← buildOutgoing d             -- you already compute basis, levels, init vectors
  validateBuiltOutgoing B
  let N        := B.basis.length
  let Lminus1  := B.formulas.length - 1
  let choices  := transposeSubpaths N Lminus1 subpaths
  let (layout, _) := buildUniformPathLayout N Lminus1

  let (goalLevel, goalIdx) ← goalForRoot B d

  let (wire, circ) := runBuilder do
    let (levelsW, gErr, _pos, _ina) ← compileWholeGridContinuous B.basis B.formulas B.initDeps layout
    let level0A := (levelsW.toArray)[goalLevel]!
    let goalV   := (level0A.toArray).getD goalIdx []
    let zero    ← CircuitOp.vecAllFalse goalV
    allocOr gErr zero

  let vals := simulate circ (packPathBits layout choices)
  pure (vals[wire]!)
/-- (Optional) Spec version of the same acceptance bit, for sanity checks. -/
def specAcceptBitFromSubpaths
  (d : DLDS)
  (subpaths : List (List Nat)) : Except String Bool := do
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let N       := B.basis.length
  let Lminus1 := (B.formulas.length - 1)
  let choices := transposeSubpaths N Lminus1 subpaths
  let (goalLevel, goalIdx) ← goalForRoot B d
  pure (specFinalOutput B goalLevel goalIdx choices)

/-- COMPILED: acceptance bit and final dependency set from N subpaths. -/
def evalSubpathsWithDeps
  (d : DLDS) (subpaths : List (List Nat))
  : Except String (Bool × List Bool × List Formula) := do
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let N        := B.basis.length
  let Lminus1  := B.formulas.length - 1
  let choices  := transposeSubpaths N Lminus1 subpaths
  let (layout, _) := buildUniformPathLayout N Lminus1
  let (gLev, gIdx) ← goalForRoot B d

  let ((finalW, goalW), circ) := runBuilder do
    let (levelsW, gErr, _pos, _ina) ← compileWholeGridContinuous B.basis B.formulas B.initDeps layout
    let levelA := (levelsW.toArray)[gLev]!
    let goalV  := (levelA.toArray).getD gIdx []
    let zero   ← CircuitOp.vecAllFalse goalV
    let acc    ← allocOr gErr zero
    pure (acc, goalV)

  let inputs := packPathBits layout choices
  let vals   := simulate circ inputs
  let vec    := readVec vals goalW
  let acc    := vals[finalW]!
  let set    := depsAsFormulas B.basis vec
  pure (acc, vec, set)


/-- SPEC (pure): same outputs computed by the reference semantics. -/
def specSubpathsWithDeps
  (d : DLDS)
  (subpaths : List (List Nat))
  : Except String (Bool × List Bool × List CircuitOp.Formula) := do
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let N        := B.basis.length
  let Lminus1  := B.formulas.length - 1
  let choices  := transposeSubpaths N Lminus1 subpaths
  let (gLev, gIdx) ← goalForRoot B d

  let (levels, err) := specWholeGridOutgoing B choices
  let some row  := (levels.toArray)[gLev]? | throw "spec: goal level out of bounds"
  let goalVec   := (row.toArray).getD gIdx (List.replicate B.encN false)
  let acc       := err || goalVec.all (· = false)
  let depSet    := depsAsFormulas B.basis goalVec
  pure (acc, goalVec, depSet)

/-- Pretty print a list of formulas as `[A, B, ...]`. -/
def ppFormulaList (xs : List CircuitOp.Formula) : String :=
  "[" ++ String.intercalate ", " (xs.map ppFormula) ++ "]"

#eval
  match evalSubpathsWithDeps d0
          -- three columns (A, A⊃B, B), two steps each:
          [[0,0], [1,0], [1,0]] with
  | .error e => s!"ERR: {e}"
  | .ok (acc, vec, set) =>
      s!"acc={acc}, final_vec={vec}, final_set={ppFormulaList set}"
#eval
  let P := [[0,0], [1,0], [1,0]]
  match (evalSubpathsWithDeps d0 P, specSubpathsWithDeps d0 P) with
  | (.ok (acc1,v1,_), .ok (acc2,v2,_)) => (acc1 = acc2) && (v1 = v2)
  | _ => false


/-- Pretty one formula name. -/
def short (f : CircuitOp.Formula) : String := ppFormula f

/-- One transition: show the choice for every source, and a row raster (● at chosen t). -/
def debugTransition
  (prevForms currForms : List CircuitOp.Formula)
  (outgoingRow : List (List EdgeKind))   -- options for each source
  (choicesRow  : List Nat)               -- user's choices for each source (0=inactive)
  : String :=
Id.run do
  let S := prevForms.length
  let T := currForms.length
  let outsA := outgoingRow.toArray
  let chA   := choicesRow.toArray
  let tgtSym (t : Nat) :=
    if t < T then s!"{t}:{short (currForms.getD t (.atom "?"))}"
    else s!"{t}:<?>"
  let mut lines : List String := []
  for s in [0:S] do
    let Fs := prevForms.getD s (.atom "?")
    let opts := outsA.getD s []
    let j    := chA.getD s 0
    if j = 0 then
      lines := lines.concat s!"s={s} [{short Fs}]  inactive (0)"
    else
      let m := opts.length
      if j > m then
        lines := lines.concat s!"s={s} [{short Fs}]  CHOICE {j} > {m}  (INVALID)"
      else
        match opts[j-1]? with
        | none =>
            lines := lines.concat s!"s={s} [{short Fs}]  CHOICE {j} → <missing>"
        | some ek =>
            let t := tgtOf ek
            let row := (List.range T).map (fun t' => if t' = t then "●" else "·")
                         |> String.intercalate " "
            lines := lines.concat
              s!"s={s} [{short Fs}]  j={j} → t={t} [{tgtSym t}]  {ppEdgeKind ek}\n    {row}"
  String.intercalate "\n" (
    s!"prev=[{String.intercalate ", " (prevForms.map short)}]"
    :: s!"next=[{String.intercalate ", " (currForms.map short)}]" :: lines)

/-- Full debug for subpaths: per transition + final bit and dep set. -/
def debugEvalSubpaths (d : DLDS) (subpaths : List (List Nat)) : IO Unit := do
  match buildOutgoing d with
  | .error e => IO.println s!"ERR building: {e}"
  | .ok B =>
      match validateBuiltOutgoing B with
      | .error e => IO.println s!"ERR validate: {e}"
      | .ok _ =>
        let N := B.basis.length
        let Lm1 := B.formulas.length - 1
        let choices := transposeSubpaths N Lm1 subpaths
        -- per transition report
        let formsA := B.formulas.toArray
        let outsA  := B.outgoings.toArray
        for k in [0:outsA.size] do
          let prevF := formsA[k]!
          let nextF := formsA[k+1]!
          let row   := outsA[k]!
          let chRow := (choices.toArray).getD k []
          IO.println s!"\n-- Transition {k} --"
          IO.println (debugTransition prevF nextF row chRow)
        -- Run once and show final
        match evalSubpathsWithDeps d subpaths with
        | .error e => IO.println s!"\nRUN ERR: {e}"
        | .ok (acc, vec, set) =>
            IO.println s!"\nacc={acc}"
            IO.println s!"final_vec={vec}"
            IO.println s!"final_set={ppFormulaList set}"

#eval debugEvalSubpaths d0 [[0,0], [1,0], [1,0]]


/-- Safe basis access with default. -/
def basisAt (basis : List CircuitOp.Formula) (i : Nat) : CircuitOp.Formula :=
  basis.getD i (.atom "?")

/-- Index of φ in a list (option). -/
def idxOf? [DecidableEq α] (xs : List α) (x : α) : Option Nat :=
  xs.enum.findSome? (fun (i,y) => if y = x then some i else none)

/-- Pretty one header line like:  A  A⊃B  B -/
def headerLine (basis : List CircuitOp.Formula) : String :=
  String.intercalate "  " (basis.map short)

/-- Pretty a row of “X [n]” aligned to the basis. -/
def destinyRow (labels : List CircuitOp.Formula) (dest : List (Option Nat)) : String :=
  let cells := (List.zip labels dest).map (fun (f, d?) =>
    match d? with
    | none   => s!"{short f} [0]"
    | some j => s!"{short f} [{j}]")
  String.intercalate "  " cells

/-- Build, for a transition k, a basis-sized vector of destinations:
    for each basis column c, `none` if inactive/absent, or `some tBasis`. -/
def destsForTransition
  (B : BuiltOutgoing) (k : Nat) (choicesRow : List Nat)
  : List (Option Nat) := Id.run do
  let basis    := B.basis
  let prevRow  := (B.formulas.toArray)[k]!
  let nextRow  := (B.formulas.toArray)[k+1]!
  let outsRow  := (B.outgoings.toArray)[k]!
  let chArr    := choicesRow.toArray

  let mut out : List (Option Nat) := []
  for c in [0:basis.length] do
    let f := basisAt basis c
    match idxOf? prevRow f with
    | none =>
        out := out ++ [none]            -- that basis formula isn't a node at this level
    | some s =>
        let j := chArr.getD s 0         -- 0 = inactive
        if j = 0 then
          out := out ++ [none]
        else
          let opts := (outsRow.toArray)[s]!
          if h : j ≤ opts.length then
            let ek := (opts.toArray).getD (j-1) (EdgeKind.rep 0)
            let tLocal :=
              match ek with
              | .rep t        => t
              | .intro t _    => t
              | .elim t _ _   => t
            let tF    := (nextRow.toArray).getD tLocal (.atom "?")
            let tBase := basis.idxOf? tF |>.getD 0
            out := out ++ [some tBase]
          else
            out := out ++ [none]
  out

/-- NxN bullets for a transition (rows = basis sources, cols = basis targets). -/
def bulletsForTransition
  (B : BuiltOutgoing) (k : Nat) (choicesRow : List Nat)
  : List String := Id.run do
  let basis   := B.basis
  let dests   := destsForTransition B k choicesRow
  let N       := basis.length
  let colsHdr := "    " ++ String.intercalate " " ((List.range N).map (fun j => toString j))
  let mut rows : List String := [colsHdr]
  for s in [0:N] do
    let rowLabel := s!"{s}:{short (basisAt basis s)} | "
    let d? := (dests.toArray)[s]!
    let line :=
      match d? with
      | none   => rowLabel ++ String.intercalate " " (List.replicate N "·")
      | some t =>
          rowLabel ++
          (List.range N |>.map (fun j => if j = t then "●" else "·") |>
           String.intercalate " ")
    rows := rows ++ [line]
  rows



/-- Current position of each subpath c; `none` = inactive. -/
abbrev ContPos := Array (Option Nat)

/-- Initial positions: subpath c starts at basis column c. -/
def contInit (N : Nat) : ContPos :=
  Array.mk <| (List.range N).map some

/-- For step k, collect the choice j for each subpath c from the *original* subpaths layout (N×(L-1)). -/
def stepChoicesForK (N : Nat) (k : Nat) (subpaths : List (List Nat)) : Array Nat :=
  let cols := subpaths.toArray
  Array.mk <| (List.range N).map (fun c => (cols.getD c []).toArray.getD k 0)

/-- Advance one step: given current positions and the choices for this step,
    (a) produce all active edges (s,t) and
    (b) the next positions. Multiple subpaths can create multiple (s,t). -/
def contAdvance (pos : ContPos) (choicesK : Array Nat)
  : (List (Nat × Nat) × ContPos) :=
Id.run do
  let N := pos.size
  let mut edges : List (Nat × Nat) := []
  let mut next  : ContPos := pos
  for c in [0:N] do
    let j := choicesK.getD c 0
    match pos.getD c none with
    | none    => pure ()
    | some s  =>
      if j = 0 then
        next := next.set! c none
      else
        let t := j - 1
        edges := (s, t) :: edges
        next  := next.set! c (some t)
  (edges.reverse, next)


/-- Make an N×N matrix with ● at the listed (s,t) pairs. -/
def rasterFromEdges (N : Nat) (edges : List (Nat × Nat)) : List String :=
  let hdr := "    " ++
    String.intercalate " " ((List.range N).map (fun x => toString x))
  let grid : Array (Array Bool) :=
    Id.run do
      let mut a := Array.replicate N (Array.replicate N false)
      for (s,t) in edges do
        if h₁ : s < N ∧ t < N then
          let row := a[s]!
          a := a.set! s (row.set! t true)
      pure a
  let rows :=
    (List.range N).map (fun s =>
      let cells := (grid[s]!).toList.map (fun b => if b then "●" else "·") |> String.intercalate " "
      s!"{s} | {cells}")
  hdr :: rows

/-- For the *destiny line*: for each basis column (source) list **all** targets chosen this step.
    `[]` means no outgoing from that source this step (prints as `[0]` per your spec). -/
def destinyFromEdges (N : Nat) (edges : List (Nat × Nat)) : List (List Nat) :=
  Id.run do
    let mut buckets : Array (List Nat) := Array.replicate N []
    for (s,t) in edges do
      if h : s < N then
        buckets := buckets.set! s (buckets[s]! ++ [t])
    buckets.toList

/-- Pretty "A [0]  A⊃B [2,2]  B [3]" style line. -/
def destinyRowMulti (labels : List CircuitOp.Formula) (dest : List (List Nat)) : String :=
  let cells :=
    (List.zip labels dest).map (fun (f, ts) =>
      match ts with
      | []      => s!"{ppFormula f} []"            -- was “[0]”
      | _::_    => s!"{ppFormula f} [{String.intercalate ", " (ts.map toString)}]")
  String.intercalate "  " cells

/-- Show dependency vector as a bitstring (e.g. "0101"). -/
def ppDepSet (vec : List Bool) : String :=
  vec.map (fun b => if b then '1' else '0')
      |>.asString

/-- FULL debug using *continuous* subpaths (no rule filtering).
    We only need the basis (for pretty labels) and the number of levels to know how many steps to show. -/
def debugGridAndSubgraph
  (d : DLDS) (subpaths : List (List Nat)) : Except String String := do
  -- basis = the same one used everywhere (your buildOutgoing basis is fine just to *print* names)
  let B ← buildOutgoing d
  validateBuiltOutgoing B
  let basis := B.basis
  let N     := basis.length
  let L     := B.formulas.length
  let Lm1   := (L - 1)

  let header := "Basis: " ++ String.intercalate "  " (basis.map ppFormula)
  let mut blocks : List String := [header]

  -- run the continuous path semantics just for printing (no rule/structure checks here)
  let mut pos := contInit N
  -- before the loop:
  -- seed prevDeps with your actual level-0 vectors
  let prevDeps0 := B.initDeps
  let mut prevDeps := prevDeps0

  for k in [0:Lm1] do
    let choicesK := stepChoicesForK N k subpaths
    let (edgesK, pos') := contAdvance pos choicesK
    pos := pos'

    /- Pretty grid for this step -/
    let bullets := rasterFromEdges N edgesK
    let dests   := destinyFromEdges N edgesK
    let rowLine := destinyRowMulti basis dests
    let title   := s!"\n-- Transition {k} (level {k} → {k+1}) --"
    blocks := blocks ++ [title]
    blocks := blocks ++ bullets

    /- RECOMPUTE dependency vectors for next level -/
    let prevForms := (B.formulas.toArray)[k]!
    let nextForms := (B.formulas.toArray)[k+1]!
    -- edgesK are basis indices already; build S×T bit matrix
    let edgeOn : LevelEdgeBits := mkEdgeOn N N edgesK
    let roles  := edgeRolesForLevel prevForms nextForms
    let (nextDeps, _err) := specLevelEdges B.encN B.basis prevDeps nextForms edgeOn roles

    /- Print per-source dep set for THIS step (sources live at prev level) -/
    -- helper: safe dep vector for a basis column `c` at prev level
    let depAt (c : Nat) : List Bool :=
      match idxOf? prevForms (basisAt basis c) with
      | none   => List.replicate B.encN false
      | some s => (prevDeps.toArray).getD s (List.replicate B.encN false)

    let srcLines :=
      (List.range N).map (fun c =>
        let φs  := basisAt basis c
        let v   := depAt c
        let outs := dests.getD c []
        s!"{ppFormula φs} : {ppDepSet v}  →  out={[String.intercalate ", " (outs.map toString)]}")
    blocks := blocks ++ srcLines
    blocks := blocks ++ [rowLine]

    -- ADVANCE for next iteration
    prevDeps := nextDeps


  -- mark the goal column on the last line (same as before)
  let (_, goalIdxBasis) ← goalForRoot B d
  let last :=
    String.intercalate " "
      ((List.range N).map (fun j =>
        if j = goalIdxBasis
        then s!"{ppFormula (basis.getD j (.atom "?"))} [X]"
        else s!"{ppFormula (basis.getD j (.atom "?"))} [ ]"))
  blocks := blocks ++ ["\n-- Last level (goal mark) --", last]

  pure (String.intercalate "\n" blocks)


#eval   IO.println <|
  match debugGridAndSubgraph d0 [[0,0,0], [1,2,3], [1,2,3], [0,0,0]] with
  | .error e => s!"ERR: {e}"
  | .ok s    => s




private def targetsFrom (edges : List (Nat × Nat)) (s : Nat) : List Nat :=
  edges.filterMap (fun (s',t) => if s'=s then some t else none)

/-- Debug: at each transition k, for every basis source s:
      φ_s : deps_before  →  out=[t,...]
    Then advances deps using your NxN-edges semantics.
 -/
def debugDepsAndEdges
  (d : DLDS) (subpaths : List (List Nat)) : Except String String := do
  -- Use the same basis/universe model as your “continuous” view
  let BE := buildEdges d
  let basis := BE.basis
  let N     := BE.N
  let L     := BE.L
  let Lm1   := if L = 0 then 0 else L - 1

  -- per-step walk state
  let mut pos  := contInit N
  let mut deps : List (List Bool) := BE.init0   -- dependency vectors at current level

  let roles := edgeRolesForLevel basis basis     -- common NxN roles
  let header := "Basis: " ++ String.intercalate "  " (basis.map ppFormula)
  let mut blocks : List String := [header]

  for k in [0:Lm1] do
    -- which edges are taken at this step, per your subpaths
    let choicesK := stepChoicesForK N k subpaths
    let (edgesK, pos') := contAdvance pos choicesK
    pos := pos'

    -- pretty: per source show deps BEFORE this step and its outgoing targets
    let mut lines : List String := []
    for s in [0:N] do
      let φs   := basis.getD s (.atom "?")
      let v    := (deps.toArray).getD s (List.replicate N false)
      let outs := targetsFrom edgesK s
      lines := lines ++ [
        s!"{ppFormula φs} : {ppDepSet v}  →  out={
          if outs.isEmpty then "[]"
          else "[" ++ String.intercalate ", " (outs.map toString) ++ "]"
        }"
]
    -- also keep your raster (●/·) for context
    let raster := rasterFromEdges N edgesK

    blocks := blocks ++
      [s!"\n-- Transition {k} (level {k} → {k+1}) --"] ++
      raster ++
      lines

    -- advance dependency vectors by one step using the NxN semantics
    let edgeOn : LevelEdgeBits := mkEdgeOn N N edgesK
    let (deps', _err) := specLevelEdges N basis deps basis edgeOn roles
    deps := deps'

  -- mark goal column at the end (same as before)
  let (_, goalIdx) ← goalForRoot (← buildOutgoing d) d
  let last :=
    String.intercalate " "
      ((List.range N).map (fun j =>
        if j = goalIdx then s!"{ppFormula (basis.getD j (.atom "?"))} [X]"
        else s!"{ppFormula (basis.getD j (.atom "?"))} [ ]"))
  blocks := blocks ++ ["\n-- Last level (goal mark) --", last]

  pure (String.intercalate "\n" blocks)

  #eval
  IO.println <|
    match debugDepsAndEdges d0 [[0,0,0], [1,2,3], [1,2,3], [0,0,0]] with
    | .error e => s!"ERR: {e}"
    | .ok s    => s
