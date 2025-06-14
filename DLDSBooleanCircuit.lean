import Init
import Mathlib.Data.List.Basic  
import Mathlib.Tactic           
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip
import Mathlib.Data.Fin.Basic



-- Define an enum to distinguish rule types
inductive RuleType
  | intro  -- 1 activation bit
  | elim   -- 2 activation bits
  deriving DecidableEq, Repr

inductive RuleData (n : Nat)
  | intro (encoder : List.Vector Bool n)
  | elim

-- Provide equality comparison for RuleType
instance : BEq RuleType where
  beq x y :=
    match x, y with
    | RuleType.intro, RuleType.intro => true
    | RuleType.elim, RuleType.elim => true
    | _, _ => false

-- Structure for activation bits
inductive ActivationBits
  | intro (bit : Bool)
  | elim (bit1 : Bool) (bit2 : Bool)
  deriving DecidableEq

@[ext]
structure Rule (n : ℕ) where
  activation : ActivationBits
  kind       : RuleData n
  combine    : List (List.Vector Bool n) → List.Vector Bool n

structure GraphPath (n l : ℕ) where
  idxs : List (Fin (n * n))
  len  : idxs.length = l

structure Grid (n : Nat) (Rule : Type) where
  nodes : List Rule
  grid_size : nodes.length =  n * n

def mkIntroRule {n : ℕ} (encoder : List.Vector Bool n) (bit : Bool) : Rule n :=
{
  activation := ActivationBits.intro bit,
  kind       := RuleData.intro encoder,
  combine    := fun deps =>
    match deps with
    | [d] => d.zipWith (fun b e => not (b && e)) encoder
    | _   => List.Vector.replicate n false
}


def mkElimRule {n : ℕ} (bit1 bit2 : Bool) : Rule n :=
{
  activation := ActivationBits.elim bit1 bit2,
  kind       := RuleData.elim,
  combine    := fun deps =>
    match deps with
    | [d1, d2] => d1.zipWith (· && ·) d2
    | _        => List.Vector.replicate n false
}

def is_rule_active {n: Nat} (r : Rule n) : Bool :=
  match r.activation with
  | ActivationBits.intro b   => b
  | ActivationBits.elim b1 b2 => b1 && b2

def multiple_xor : List Bool → Bool
| []       => false
| [x]      => x
| x :: xs  => (x && not (List.or xs)) || (not x && multiple_xor xs)

def extract_activations {n: Nat} (rules : List (Rule n)) : List Bool :=
  rules.map is_rule_active

-- Perform AND operation between a Boolean an a List
def and_bool_list (bool : Bool) (l : List Bool): List Bool :=
  l.map (λ b => bool && b)

def list_or {n: Nat} (lists : List (List.Vector Bool n)) : List.Vector Bool n :=
  lists.foldl (λ acc lst => acc.zipWith (λ x y => x || y) lst)
              (List.Vector.replicate n false)

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

def node_logic {n: Nat} (rules : List (Rule n))
                  (inputs : List (List.Vector Bool n))
  : List.Vector Bool n :=
  let acts := extract_activations rules
  let xor  := multiple_xor acts
  let masks := and_bool_list xor acts
  let outs    := apply_activations rules masks inputs
  list_or outs

structure CircuitNode (n: Nat) where
  rules : List (Rule n)
  pairwise : rules.Pairwise (· ≠ ·)

def mkCircuitNode {n} (rs : List (Rule n)) (h : rs.Pairwise (· ≠ ·)) : CircuitNode n :=
  { rules := rs, pairwise := h }

def CircuitNode.run {n: Nat} (c : CircuitNode n)
    (inputs : List (List.Vector Bool n)) : List.Vector Bool n :=
  node_logic c.rules inputs


def exactlyOneActive {n: Nat} (rules : List (Rule n)) : Prop :=
  ∃ r, r ∈ rules ∧ is_rule_active r ∧ ∀ r', r' ∈ rules → is_rule_active r' → r' = r


def natToBits (n k : ℕ) : List Bool :=
  (List.range k).map (fun i => (n.shiftRight (k - 1 - i)) % 2 = 1)


def selector (input : List Bool) : List Bool :=
  let n := input.length
  let total := 2 ^ n
  List.ofFn (fun (i : Fin total) =>
    let bits := natToBits i.val n
    (input.zip bits).foldl (fun acc (inp, b) =>
      acc && if b then inp else !inp) true
  )


@[simp]
lemma multiple_xor_cons_false (l : List Bool) :
  multiple_xor (false :: l) = multiple_xor l := by
  induction l with
  | nil => simp [multiple_xor]
  | cons b bs ih =>
    simp [multiple_xor]

lemma multiple_xor_cons_true_aux {l : List Bool} :
  multiple_xor (true :: l) = !l.or := by
  cases l with
  | nil => simp [multiple_xor, List.or]
  | cons b bs => simp [multiple_xor]

lemma multiple_xor_cons_true {l : List Bool} :
  multiple_xor (true :: l) = true ↔ List.or l = false := by
  simp [multiple_xor, Bool.eq_true_eq_not_eq_false, Bool.not_eq_true_eq_eq_false]
  exact multiple_xor_cons_true_aux

lemma bool_eq_false_of_ne_true : ∀ b : Bool, b ≠ true → b = false := by
  intro b h
  cases b
  · rfl
  · contradiction

lemma List.or_eq_false_iff_all_false {l : List Bool} :
  l.or = false ↔ ∀ b ∈ l, b = false := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.or, Bool.or_eq_false_eq_eq_false_and_eq_false, List.mem_cons, forall_eq_or_imp, ih]
    simp [List.any]

theorem multiple_xor_bool_iff_exactlyOneActive {n : ℕ} (rs : List (Rule n)) (h_pairwise : rs.Pairwise (· ≠ ·)) :
  multiple_xor (rs.map is_rule_active) = true ↔ exactlyOneActive rs := by
  induction rs with
  | nil =>
    simp [multiple_xor, exactlyOneActive]
  | cons r rs ih =>
    have tail_pairwise : rs.Pairwise (· ≠ ·) := h_pairwise.tail
    cases hr : is_rule_active r
    · -- Case: r is inactive
      simp only [List.map, hr, multiple_xor]
      rw [multiple_xor_cons_false, ih tail_pairwise]
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
    · -- Case: r is active
      simp only [List.map, hr, multiple_xor]
      rw [multiple_xor_cons_true]
      simp only [exactlyOneActive]
      constructor
      · intro h
        -- or (rs.map is_rule_active) = false means all others are inactive
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
          -- if the active rule is r itself
          simp [List.or_eq_false_iff_all_false]
          intro b hb
          by_contra h_contra
          have hb_true : is_rule_active b = true := by
            cases h_b : is_rule_active b
            · contradiction -- trivial since h_contra assumes it's not false
            · rfl          
          have eq_b := h_unique b (List.mem_cons_of_mem _ hb) hb_true
          let ⟨r_ne, _⟩ := List.pairwise_cons.mp h_pairwise
          rw [eq_b] at hb
          have r_ne_self := r_ne r hb
          exact r_ne_self rfl
        | tail _ h_tail =>
          have eq_head := h_unique r (List.Mem.head ..) hr
          let ⟨r_ne, _⟩ := List.pairwise_cons.mp h_pairwise
          exact False.elim (r_ne r₁ h_tail eq_head)



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


lemma mem_zipWith_map {α β γ : Type} (f : α → β → γ) (l : List α) (g : α → β)
  (x : γ) (hx : x ∈ List.zipWith f l (l.map g)) : ∃ (r : α), r ∈ l ∧ x = f r (g r) :=
by
  induction l with
  | nil =>
      simp at hx
  | cons a as ih =>
      simp [List.zipWith, List.map] at hx
      cases hx with
      | inl h1 =>
          exists a
          constructor
          · simp
          · rw [h1]
      | inr h₂ =>
          obtain ⟨r, hr_in, h_eq⟩ := ih h₂
          exists r
          constructor
          · exact (List.mem_cons_of_mem a hr_in)
          · exact h_eq


/--
If exactly one rule in `rules` is active,
then OR‑folding the result of `apply_activations rules masks inputs`
yields exactly that rule’s `combine inputs`.
-/

lemma list_or_apply_unique_active_of_exactlyOne {n : ℕ}
  {rules : List (Rule n)} (h_nonempty : rules ≠ [])
  {r0 : Rule n} (hr0_mem : r0 ∈ rules)
  (h_one      : exactlyOneActive rules)
  (hr0_active : is_rule_active r0 = true)
  (inputs     : List (List.Vector Bool n)) :
  list_or (apply_activations rules (extract_activations rules) inputs)
    = r0.combine inputs := by
  -- abbreviations
  -- abbreviations from before
  let masks := extract_activations rules
  let outs  := apply_activations rules masks inputs
  

  -- show masks.length = rules.length
  have masks_eq : masks.length = rules.length := by
    -- `extract_activations = List.map is_rule_active`
    dsimp [masks, extract_activations]
    simp [List.length_map]

  -- now outs.length = min rules.length masks.length,
  -- and since rules.length ≤ masks.length, `min … = rules.length`.
  have h_len : outs.length = rules.length := by
    dsimp [outs, apply_activations]
    -- length_zipWith: length (zipWith _ l1 l₂) = min l1.length l₂.length
    rw [List.length_zipWith]
    -- rewrite masks.length to rules.length
    rw [masks_eq]
    -- now we have `min rules.length rules.length = rules.length`
    apply Nat.min_self

  -- prove every element of `outs` is either zero or exactly `r0.combine inputs`
  have h_all : ∀ x ∈ outs, x = List.Vector.replicate n false ∨ x = r0.combine inputs := by
    let outs := List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false) rules (rules.map is_rule_active)
    intros x hx
    have outs_eq : apply_activations rules masks inputs =
      List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false)
        rules (List.map is_rule_active rules) := by
      dsimp [apply_activations, masks, extract_activations]
      
    have hx' : x ∈ List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false)
      rules (List.map is_rule_active rules) := by
      rw [←outs_eq]; exact hx

    have ⟨r, hr_in, x_eq⟩ := mem_zipWith_map (fun r m => if m then r.combine inputs else List.Vector.replicate n false) rules is_rule_active x hx'

    obtain ⟨r₁, hr₁_mem, hr₁_active, h_unique⟩ := h_one

    have r0_eq_r₁ : r0 = r₁ := by
      apply h_unique r0 hr0_mem hr0_active
    subst r0_eq_r₁

    by_cases h : r = r0
    · subst h
      rw [x_eq, hr0_active]
      right; rfl

    · have h_inactive : is_rule_active r = false := by
        by_contra contra
        have hr_act : is_rule_active r = true :=
          Bool.eq_true_of_not_eq_false contra
        have eq_r0 := h_unique r hr_in hr_act
        exact h eq_r0
      rw [x_eq, h_inactive]
      left; rfl
  
  let idx := outs.idxOf (r0.combine inputs)


  have in_outs : r0.combine inputs ∈ outs := by
    -- exists in outs because h_all + exactlyOneActive
    obtain ⟨i, hi, get_eq⟩ := List.get_of_mem hr0_mem
    have acts_len : (List.map is_rule_active rules).length = rules.length := by simp
    let j : Fin rules.length := i
    let k : Fin (List.map is_rule_active rules).length := acts_len ▸ j
    let outs := List.zipWith (fun r m => if m then r.combine inputs else List.Vector.replicate n false)
                         rules (rules.map is_rule_active)

    have get_idx : outs.get (h_len ▸ i) = (rules.get i).combine inputs := by
      dsimp [outs, apply_activations, masks, extract_activations]
      -- Now get the value at position i, using the definition of apply_activations
      rw [List.getElem_zipWith, List.getElem_map]
      have : rules.get (h_len ▸ i) = rules.get i := by
        sorry
      sorry
        --List.get_eq_get_of_eq _ _ (by rw [h_len])
      -- rw [this]
      -- exact if_pos hr0_active
        
      -- rules.get i = rules.get i, so fine
      -- (rules.map is_rule_active).get i = is_rule_active (rules.get i)

      -- rfl

    have mem : (rules.get i).combine inputs ∈ outs := by
  -- This index is valid since (h_len ▸ i).isLt = i.isLt (because lengths are equal)
      have idx_lt : (h_len ▸ i).val < outs.length := (h_len ▸ i).isLt
      rw [←get_idx]
      exact List.get_mem outs (h_len ▸ i)

    exact mem
   
  
  have : list_or outs = r0.combine inputs := by
    sorry

  exact this
      
  
theorem node_correct {n} (c : CircuitNode n)
    (inputs : List (List.Vector Bool n))
    (h_one : exactlyOneActive c.rules) :
  ∃ r ∈ c.rules, c.run inputs = r.combine inputs := by

  -- 0) immediately derive the Bool‐onehot:
  have h_bool : multiple_xor (c.rules.map is_rule_active) = true :=
    (multiple_xor_bool_iff_exactlyOneActive c.rules c.pairwise).mpr h_one
  
  let h_one_prop := h_one

  -- 1) now unpack the unique‐active Prop
  rcases h_one with ⟨r0, hr0_mem, hr0_active, hr0_unique⟩

  -- 2) simplify the let-chain, *including* extract_activations
  dsimp [CircuitNode.run, node_logic, extract_activations]

  -- 3) now use simp, telling it both `h_bool` *and* `and_bool_list`
  --    this unfolds `and_bool_list true xs = xs` and rewrites the one-hot test
  -- 3) Rewrite away the Bool test and the mask
  rw [h_bool]                   -- replaces multiple_xor … with true
  dsimp [and_bool_list]         -- uses and_bool_list true xs = xs
  -- 1) replace the Boolean test with `true`
  -- 2) unfold the definition of `and_bool_list` so `and_bool_list true xs = xs`

    -- 4) now c.rules ≠ [] follows from hr0_mem
  let h_nonempty := List.ne_nil_of_mem hr0_mem
  simp [List.map_map, true_and]

  let eq := list_or_apply_unique_active_of_exactlyOne
    h_nonempty hr0_mem h_one_prop hr0_active inputs

  use r0
  constructor
  · exact hr0_mem
  · exact eq



-- Function to count occurrences of true in a boolean lis

def exactlyOne (l : List Bool) : Bool :=
  match l with
  | []      => false
  | [x]     => x
  | x :: xs => (x && not (List.or xs)) || (not x && exactlyOne xs)


def exactlyOneIndexTrue (l : List Bool) : Prop :=
  ∃ (i : Nat),
    i < l.length ∧
    l[i]? = some true ∧
    ∀ j, j < l.length → l[j]? = some true → j = i


def initial_inputs {n} : List (List.Vector Bool n) :=
  List.replicate n (List.Vector.replicate n false)

namespace List

/--
`nthLe l i h` returns the `i`th element of `l`, given a proof `h : i < l.length`.
-/
def nthLe {α : Type u} (l : List α) (i : Nat) (h : i < l.length) : α :=
  -- `List.get` takes a `Fin l.length` — here we pack `i` and `h` into one
  l.get ⟨i, h⟩


def init {α : Type u} : List α → List α
| []        => []
| [_]       => []              -- if there’s exactly one element, drop it
| x :: y :: xs => x :: init (y :: xs)

end List


def evalStep (grid : Grid n (CircuitNode n))
             (inputs : List (List.Vector Bool n))
             (sel : Fin n)
  : List (List.Vector Bool n) :=
  ((grid.nodes.drop (sel.val * n)).take n).map (·.run inputs)

def evalStepOne (grid   : Grid n (CircuitNode n))
                (state  : List.Vector Bool n)
                (sel    : Fin (n * n))
  : List.Vector Bool n := 
  let node := grid.nodes.nthLe sel.val (by simp [grid.grid_size])
  -- wrap the single state vector into a singleton list,
  -- because CircuitNode.run expects List (List.Vector Bool n)
  node.run [state]

def evalGrid {n L : ℕ}
  (initial : List.Vector Bool n)
  (grid    : Grid n (CircuitNode n))
  (p       : GraphPath n L)
: List.Vector Bool n :=
  p.idxs.foldl (fun st sel => evalStepOne grid st sel) initial

@[simp] lemma nthLe_cons_succ {α} (a : α) (l : List α) (i h1 h₂) :
  (a :: l).nthLe (i+1) h₂ = l.nthLe i h1 := rfl

theorem List.nthLe_eq_get {α : Type u} (l : List α) (i : ℕ) (h : i < l.length) :
  l.nthLe i h = l.get ⟨i, h⟩ :=
rfl

@[simp]
theorem List.nthLe_cons_zero {α : Type*} (x : α) (xs : List α) (h : 0 < (x :: xs).length) :
  (x :: xs).nthLe 0 h = x :=
rfl

theorem grid_correct {n L : ℕ}
  (grid : Grid n (CircuitNode n))
  (h_act : ∀ i hi,
    exactlyOneActive (grid.nodes.nthLe i hi).rules)
  (initial : List.Vector Bool n)
  (p : GraphPath n L)
: ∃ (vs : List (List.Vector Bool n)) (vs_len : vs.length = L + 1),
    vs.length = L + 1 ∧
    -- head = initial:
    vs.nthLe 0 (by simp [vs_len]) = initial ∧
    -- last = evalGrid initial grid p:
    vs.nthLe L (by simp [vs_len])
      = evalGrid initial grid p ∧
    -- each step applies exactly one node:
    ∀ (i : Fin L),
      let inputs := vs.nthLe i.val
        (by simpa [vs_len] using Nat.lt_succ_of_lt i.isLt);
      let sel := p.idxs.nthLe i.val (by simp [p.len]);
      vs.nthLe (i.val + 1) (by simp [vs_len])
        = (grid.nodes.nthLe sel.val (by simp [grid.grid_size])).run [inputs] :=
by
induction L generalizing initial with
  | zero =>
      rcases p with ⟨idxs, idxs_len⟩
      have : idxs = [] := by simpa using idxs_len
      subst this
      exists [initial]
      simp [evalGrid, List.nthLe_eq_get]
  | succ L' ih =>
    -- split the path into its list-of-selectors
    rcases p with ⟨idxs, idxs_len⟩
    -- now idxs_len : idxs.length = L'+1
    -- break idxs into head :: tail
    cases idxs with
    | nil =>
      -- impossible, since idxs.length = L'+1 > 0
      simp at idxs_len
    | cons sel rest =>
      -- from idxs_len : rest.length + 1 = L'+1, we get
      have rest_len : rest.length = L' := by simpa [List.length_cons] using idxs_len

      -- 1) pick the one node at index sel.val
      let c := grid.nodes.nthLe sel.val (by simp [grid.grid_size])

      -- 2) by hypothesis that node has exactly one active rule
      have hi_sel : sel.val < grid.nodes.length := by simp [grid.grid_size] 
      have h_one := h_act sel.val hi_sel

      -- 3) apply your existing `node_correct`
      rcases node_correct c [initial] h_one with ⟨r, hr_mem, hr_eq⟩

      -- `c.run [initial] = r.combine [initial]`
      let next := r.combine [initial]

      -- 4) recurse on the tail with `next` as the new initial
      let p' : GraphPath n L' := ⟨rest, rest_len⟩

      rcases ih next p' with ⟨vs', vs'_len, h_head, h_last, h_steps⟩

      -- 5) stitch together the full list of states
      let vs := initial :: vs'
      have vs_len : List.length vs = (L' + 1) + 1 := by
        -- `List.length_cons` : ∀ {α} {a l}, List.length (a :: l) = l.length + 1
        rw [List.length_cons, vs'_len]

      use vs, vs_len
      
            -- 1) length
      constructor
      -- 1) proves A : vs.length = L' + 1 + 1
      · dsimp [vs]; simp [vs'_len]

      -- 2) proves B : vs.nthLe 0 _ = initial
      constructor
      · rfl

      -- 3) proves C : vs.nthLe (L'+1) _ = evalGrid initial grid (sel :: rest)
      constructor
      · -- Final state correctness
        simp [evalGrid, evalStepOne, hr_eq, h_last]
        rw [nthLe_cons_succ]
        rw [h_steps.1]
        dsimp [evalGrid]
        dsimp [evalStepOne]
        rw [hr_eq]

      
      · -- Intermediate correctness at each step
        intro i
        cases i using Fin.cases with
        | zero =>
          -- First step explicitly
          simp [evalStepOne, hr_eq]
          simp [vs, h_last, vs_len, evalStepOne]
          rw [nthLe_cons_succ]
          rw [List.nthLe]
          have h_get : vs'.get ⟨0, by simp [vs'_len]⟩ = next := by rw [← List.nthLe_eq_get, h_last]
          rw [h_get, hr_eq]

        | succ i' =>
          -- Subsequent steps by induction hypothesis
          simp [vs]
          rw [nthLe_cons_succ]    -- first one succeeds
          exact h_steps.right i'

  -- define function that changes only activation
def set_activation (n : ℕ) (r : Rule n) (a : ActivationBits) : Rule n :=
  { r with activation := a }

lemma set_activation_injective {n} {r : Rule n} : Function.Injective (set_activation n r) := by
  intros a₁ a₂ h_eq
  simp [set_activation] at h_eq
  exact h_eq

lemma set_activation_injective_full {n : ℕ} :
  ∀ {r₁ r₂ : Rule n} {a₁ a₂ : ActivationBits},
    set_activation n r₁ a₁ = set_activation n r₂ a₂ →
      r₁.kind = r₂.kind ∧ r₁.combine = r₂.combine ∧ a₁ = a₂ := by
  intros r₁ r₂ a₁ a₂ h_eq
  simp [set_activation] at h_eq
  sorry


/-- Type alias: for each rule in a node, where to read activation from (previous layer) -/
abbrev IncomingMap := List (Nat × Nat)
/-- For all nodes in a layer -/
abbrev IncomingMapsLayer := List IncomingMap
/-- For all layers -/
abbrev IncomingMaps := List IncomingMapsLayer

structure GridLayer (n : ℕ) where
  nodes : List (CircuitNode n)
  incoming : IncomingMapsLayer

def GridLayers (n : ℕ) := List (GridLayer n)

def dummyIncomingMapsLayer (num_nodes : Nat) : IncomingMapsLayer :=
  List.replicate num_nodes []



def activateNodeFromSelectors {n : Nat}
  (prev_selectors : List (List Bool))
  (incoming_map   : IncomingMap)
  (node           : CircuitNode n)
: CircuitNode n :=
let len := node.rules.length
let new_rules := List.finRange len |>.map (fun i =>
  let rule := node.rules.get i -- i : Fin len
  let (src_idx, edge_idx) :=
    if h_map : i.val < incoming_map.length then
      incoming_map.get ⟨i.val, h_map⟩
    else
      (0, 0) 
  let act :=
    if h_src : src_idx < prev_selectors.length then
      let sel := prev_selectors.get ⟨src_idx, h_src⟩
      if h_edge : edge_idx < sel.length then
        sel.get ⟨edge_idx, h_edge⟩
      else
        false
    else
      false
  match rule.activation with
  | ActivationBits.intro _ =>
      { rule with activation := ActivationBits.intro act }
  | ActivationBits.elim _ _ =>
      { rule with activation := ActivationBits.elim act act }
)
let h_pairwise : new_rules.Pairwise (· ≠ ·) := by
  -- same as before; activation doesn't affect .Pairwise since rules remain structurally the same
  let old_rules := node.rules
  have h_old : old_rules.Pairwise (· ≠ ·) := node.pairwise


  -- Prove that if r1 ≠ r2, then set_activation r1 a1 ≠ set_activation r2 a2
  have h_inj : ∀ r1 r2 a1 a2, r1 ≠ r2 → set_activation n r1 a1 ≠ set_activation n r2 a2 := by
    intros r1 r2 a1 a2 h_ne
    intro h_eq
    simp only [set_activation] at h_eq

    rcases set_activation_injective_full h_eq with ⟨h_kind, h_comb, h_act⟩
    
    -- Use `ext` to show r1 = r2 under the equal fields
    have : r1 = r2 := by
      ext
      · exact h_act   -- activation
      · exact h_kind         -- kind
      · rw [h_comb]          -- combine

    -- Contradiction with h_ne
    exact h_ne this

  have h_pairwise : new_rules.Pairwise (· ≠ ·) := by
    simp [List.pairwise_map]
    exact node.pairwise.imp (h_inj _ _ _ _)




{ rules := new_rules, pairwise := h_pairwise }



/-- Activates all nodes in a layer via selector wiring -/
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

def evalGridSelectorBase {n : Nat}
  (layer : GridLayer n)
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: List (List.Vector Bool n) :=
  let activated_layer := activateLayerFromSelectors initial_selectors layer
  activated_layer.map (λ node => node.run initial_vectors)

def evalGridSelectorStep {n : Nat}
  (prev_results : List (List.Vector Bool n))
  (layer : GridLayer n)
: List (List.Vector Bool n) :=
  let selectors := prev_results.map (λ v => selector v.toList)
  let activated_layer := activateLayerFromSelectors selectors layer
  activated_layer.map (λ node => node.run prev_results)

def evalGridSelector_aux {n : Nat}
  (layers : List (GridLayer n))
  (acc : List (List.Vector Bool n))
: List (List (List.Vector Bool n)) :=
  match layers with
  | [] => [acc]
  | layer :: layers' =>
      let next_result := evalGridSelectorStep acc layer
      next_result :: evalGridSelector_aux layers' next_result


def evalGridSelector {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: List (List (List.Vector Bool n)) :=
  let rec aux (acc : List (List.Vector Bool n)) (ls : List (GridLayer n)) :=
    match ls with
    | []      => [acc]
    | l :: ls => 
        let res := evalGridSelectorStep acc l
        acc :: aux res ls
  aux initial_vectors layers





/-- Query the output dependency vector of a goal node after grid evaluation -/
def goalNodeOutput {n : Nat}
  (results : List (List (List.Vector Bool n)))
  (goal_layer : Fin results.length)
  (goal_idx : Fin (results.get goal_layer).length)
: List.Vector Bool n :=
  (results.get goal_layer).get goal_idx


/-! ## 5. Full Grid Correctness for Parallel/Tree-Like Subgraphs -/

/--
Let `layers` be a grid (list of layers, each a list of circuit nodes).
Let `incomingMaps` describe selector-driven wiring.
Let `initial_vectors` give the initial dependency vectors at layer 0.
Let `initial_selectors` describe the initial activation selectors.

Suppose that for each active node at each layer (as determined by selectors/subgraph),
the selectors induce exactlyOneActive in its rules.

Then for every node `(l, i)` that is reachable via some path from an initial node
(along the active subgraph), the output dependency vector at `(l, i)` computed by
`evalGridSelector` equals the unique composition of rule applications along that path.

This expresses full, parallel/global correctness of the Boolean circuit for DLDS proof checking.

The selectors supplied for the first layer (layer 0) are ignored and act as padding.
They are not read or used during the evaluation of the initial state; only the initial dependency
vectors are relevant for the output at layer 0.


-/

@[simp]
theorem List.getD_cons {α : Type*} (x : α) (xs : List α) (n : Nat) (d : α) :
  (x :: xs).getD n d = if n = 0 then x else xs.getD (n - 1) d := by
  cases n <;> simp [List.getD]

/-- `List.getD_zero`: getD at zero returns the head of the list, or default for nil. -/
@[simp]
theorem List.getD_zero {α : Type*} (l : List α) (d : α) :
  l.getD 0 d = l.headD d := by
  cases l <;> simp [List.getD, List.headD]


  
lemma List.get?_map {α β : Type} (l : List α) (f : α → β) (i : Nat) :
  (l.map f)[i]? = (l[i]?).map f := by
  induction l generalizing i with
  | nil =>
    simp [List.get?, List.map]
  | cons hd tl ih =>
    cases i with
    | zero => simp [List.get?, List.map]
    | succ i' =>
      simp [List.get?, List.map, ih]

lemma List.get?_append_right {α} (l₁ l₂ : List α) (i : Nat) (h : i ≥ l₁.length) :
  (l₁ ++ l₂)[i]? = l₂[i - l₁.length]? := by
  induction l₁ generalizing i with
  | nil => simp
  | cons _ tl ih =>
    cases i with
    | zero => simp at h
    | succ i' =>
      simp [ih _ (Nat.le_of_succ_le_succ h)]

@[simp]
theorem List.getLastD_eq_getLast_getD {α : Type*} (l : List α) (d : α) :
  l.getLastD d = l.getLast?.getD d := by
  cases l with
  | nil => simp [List.getLastD, List.getLast?, Option.getD]
  | cons a as =>
    simp [List.getLastD, List.getLast?, Option.getD]

lemma List.get?_singleton_zero {α} (x : α) : [x][0]? = some x := by simp

def dummyRule {n : Nat} : Rule n :=
  {
    activation := ActivationBits.intro true,
    kind := RuleData.intro (List.Vector.replicate n true),
    combine := fun _ => List.Vector.replicate n false
  }

def dummyNode (n : Nat) : CircuitNode n :=
  {
    rules := [dummyRule],
    pairwise := by
      simp [List.Pairwise, List.Pairwise]
      -- with just one element, this is trivial
  }

lemma getD_zero_eq_match {α : Type*} (l : List α) (d : α) :
  l.getD 0 d = (match l[0]? with | some x => x | none => d) := by
  cases l <;> simp [List.getD, List.get?, Option.getD]


@[simp]
lemma List.getD_singleton {α} (x : α) (n : Nat) (d : α) :
  ([x] : List α).getD n d = if n = 0 then x else d :=
by
  cases n
  · simp [List.getD] -- n = 0
  · simp [List.getD] -- n = n'+1

lemma List.getD_singleton_succ {α} (x d : α) (n : ℕ) :
  ([x] : List α).getD (n + 1) d = d :=
by simp [List.getD]

@[simp]
lemma List.getD_zero_eq_get {α : Type*} (l : List α) (d : α) (h : 0 < l.length) :
  l.getD 0 d = (l[0]?).getD d :=
by simp [List.getD, List.get?, Option.getD]


@[simp]
lemma activateLayerFromSelectors_length {n : ℕ}
  (s : List (List Bool)) (layer : GridLayer n) :
  (activateLayerFromSelectors s layer).length = layer.nodes.length :=
by simp [activateLayerFromSelectors]


@[simp]
lemma option_getD_some {α} (a : α) (d : α) : (some a).getD d = a := by simp [Option.getD]
@[simp]
lemma option_getD_none {α} (d : α) : (none : Option α).getD d = d := by simp [Option.getD]
@[simp]
lemma list_singleton_getD {α} (x : α) (d : α) : ([x] : List α)[0]?.getD d = x := by simp



/--
Predicate asserting that for every node (l, i) in the grid,
the selectors induce exactly one active rule in that node.
-/
def RuleActivationCorrect {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: Prop :=
  ∀ (l : Fin layers.length) (i : Fin (layers.get l).nodes.length),
    let prev_results :=
      if h0 : l.val = 0 then initial_vectors
      else
        (evalGridSelector (layers.take l.val)
          initial_vectors initial_selectors).getLastD initial_vectors
    let prev_selectors :=
      if h0 : l.val = 0 then initial_selectors
      else prev_results.map (λ v => selector v.toList)
    let act_nodes := activateLayerFromSelectors prev_selectors (layers.get l)
    let hlen : act_nodes.length = (layers.get l).nodes.length :=
      activateLayerFromSelectors_length prev_selectors (layers.get l)
    let node := act_nodes.get (Fin.cast (Eq.symm hlen) i)
    exactlyOneActive node.rules


lemma evalGridSelector_base_succ {n : Nat}
  (first_layer : GridLayer n)
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool)) :
  (evalGridSelector [first_layer] initial_vectors initial_selectors).get? 1 =
    some (activateLayerFromSelectors (initial_vectors.map (λ v => selector v.toList)) first_layer |>.map (λ node => node.run initial_vectors)) :=
by
  simp [evalGridSelector]
  rfl


@[simp] lemma List.getLastD_singleton {α} (x d : α) : ([x] : List α).getLastD d = x := rfl

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

lemma evalGridSelector_length {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool)) :
  (evalGridSelector layers initial_vectors initial_selectors).length = layers.length + 1 :=
by
  simp [evalGridSelector, evalGridSelector_aux_length]


lemma evalGridSelector_aux_get0 {n : ℕ}
  (layers : List (GridLayer n)) (acc : List (List.Vector Bool n))
  (h : 0 < (evalGridSelector_aux layers acc).length) :
  (evalGridSelector_aux layers acc).get ⟨0, h⟩ =
    match layers with
    | [] => acc
    | l :: _ => evalGridSelectorStep acc l :=
by
  cases layers
  · simp [evalGridSelector_aux]
  · simp [evalGridSelector_aux]


lemma evalGridSelector_layer_length
  {n : ℕ}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (goal_layer : Fin layers.length) :
  ((evalGridSelector layers initial_vectors initial_selectors).get
    (Fin.cast (Eq.symm (evalGridSelector_length layers initial_vectors initial_selectors)) goal_layer.succ)).length
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

@[simp] lemma evalGridSelectorStep_length {n} (xs : List (List.Vector Bool n)) (layer : GridLayer n) :
  (evalGridSelectorStep xs layer).length = layer.nodes.length := by
  dsimp [evalGridSelectorStep, activateLayerFromSelectors]
  simp [List.length_map, activateLayerFromSelectors_length]

lemma selectors_base_eq (initial_vectors : List (List.Vector Bool n)) (initial_selectors : List (List Bool))
  (h : initial_selectors = List.map (fun v => selector v.toList) initial_vectors) :
  List.map (fun v => selector v.toList) initial_vectors = initial_selectors :=
by rw [h]


lemma evalGridSelector_cons_succ_get
  {n : ℕ}
  (layer_hd : GridLayer n)
  (layers_tl : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (goal_layer' : Fin layers_tl.length)
:
  (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get
      ⟨goal_layer'.val + 1, by
        have h := goal_layer'.isLt
        simp [evalGridSelector_length] at *
        have bound : goal_layer'.val + 1 < layers_tl.length + 2 :=
          by linarith [goal_layer'.isLt]
        exact Nat.lt_of_succ_lt_succ bound
      ⟩
  =
  (evalGridSelector layers_tl
    (evalGridSelectorStep initial_vectors layer_hd)
    (List.map (fun v => selector v.toList) (evalGridSelectorStep initial_vectors layer_hd))
  ).get (Fin.cast (by simp [evalGridSelector_length]) goal_layer'.castSucc)

:=
by
  rfl

lemma match_getLast_of_ne_nil_two_args {α : Type*} (xs : List α) (a₀ : α) (h : xs ≠ []) :
  (match xs, a₀ with
   | [], a₀ => a₀
   | a :: as, _ => (a :: as).getLast h)
  = xs.getLast h :=
by cases xs <;> simp [List.getLast]; contradiction

lemma match_getLast_of_ne_nil_three_args {α : Type*} (xs : List α) (acc : α) (h : xs ≠ []) :
    (match xs, acc, h with
     | [], a₀, _ => a₀
     | a :: as, _, _ => (a :: as).getLast (by simp)) = xs.getLast h :=
by
  cases xs
  · contradiction
  · simp [List.getLast]

lemma match_getLast_of_ne_nil_two_args2 {α : Type u} (l : List α) (x : α) (h : l ≠ []) :
  (match l, x, h with
   | [], a₀, h => a₀
   | a :: as, x, h => (a :: as).getLast h) = l.getLast h :=
by
  cases l
  · contradiction
  · rfl


lemma getLast_eq_after_cons {α : Type*} (xs : List α) (x : α) (h : xs ≠ []) :
    (match xs, x with
     | [], a₀ => a₀
     | a :: as, _ => (a :: as).getLast (by simp)) =
    (x :: xs).getLast (by simp) :=
by
  cases xs with
  | nil => contradiction
  | cons a as => simp [List.getLast]




@[simp]
lemma evalGridSelector_getLastD_eq_aux {n}
  (layers : List (GridLayer n)) (initial_vectors : List (List.Vector Bool n)) :
  (evalGridSelector layers initial_vectors initial_selectors).getLastD initial_vectors
  =
  (evalGridSelector.aux initial_vectors layers).getLast?.getD initial_vectors :=
by
  simp [evalGridSelector]

lemma List.getLastD_eq_getLast_of_ne_nil {α} (xs : List α) (d : α) (h : xs ≠ []) : 
    xs.getLastD d = xs.getLast h := by
  cases xs with
  | nil => contradiction
  | cons a as => 
    cases as with
    | nil => simp [List.getLastD, List.getLast]
    | cons b bs => 
      simp [List.getLastD, List.getLast]



lemma prev_results_shift
  {n : ℕ}
  (layer_hd : GridLayer n) (layers_tl : List (GridLayer n))
  (initial_vectors acc : List (List.Vector Bool n))
  (initial_selectors new_selectors : List (List Bool))
  (l : ℕ) (h_l : l ≤ layers_tl.length)
  (h_acc : acc = evalGridSelectorStep initial_vectors layer_hd)
  (h_sel : new_selectors = List.map (fun v => selector v.toList) acc)
  (h_sel0 : initial_selectors = List.map (fun v => selector v.toList) initial_vectors) :
  (if h0 : l = 0 then acc else (evalGridSelector (List.take l layers_tl) acc new_selectors).getLastD acc)
  =
  (evalGridSelector (List.take (l + 1) (layer_hd :: layers_tl)) initial_vectors initial_selectors).getLastD initial_vectors :=
by
  cases l
  case zero =>
    -- l = 0
    simp [evalGridSelector, evalGridSelectorStep, evalGridSelector.aux, List.getLastD]
    rw [h_acc]
    rfl
  case succ l' =>
    -- l = l'+1 > 0
    rw [Nat.add_comm, List.take]
    -- Now use definition of evalGridSelector
    rw [evalGridSelector]
    rw [h_acc]
    -- getLastD on (x :: xs) is xs.getLastD acc
    rw [List.getLastD.eq_def]
    sorry
    -- cases l'
    -- case zero =>
    --   simp only [Nat.add_zero] at *
    --   simp only [if_neg (by decide)]
    --   rw [evalGridSelector, h_acc, List.getLastD]
    --   rfl
    -- case succ l'' =>
    --   -- l' + 1 ≥ 1, so 1 + l' ≠ 0
    --   rw [if_neg (by decide)]
    --   rw [evalGridSelector, h_acc, List.getLastD]
    --   rfl


/--
If every node of `layer_hd :: layers_tl` has exactly one active rule
(with respect to the original vectors / selectors), then the same holds
for the tail once we evaluate the head layer and push its selectors
forward.
-/
lemma RuleActivationCorrect.tail
  {n : ℕ}
  {layer_hd : GridLayer n} {layers_tl : List (GridLayer n)}
  {init_vecs : List (List.Vector Bool n)}
  {init_sels : List (List Bool)}
  (h_sel0  : init_sels = List.map (fun v => selector v.toList) init_vecs) 
  (h_act : RuleActivationCorrect (layer_hd :: layers_tl) init_vecs init_sels) :
    RuleActivationCorrect
      layers_tl
      (evalGridSelectorStep init_vecs layer_hd)
      ((evalGridSelectorStep init_vecs layer_hd).map (fun v => selector v.toList)) :=
by
  -- abbreviations just to shorten formulas
  let acc          := evalGridSelectorStep init_vecs layer_hd
  let newSelectors := acc.map (fun v => selector v.toList)
  let sels := acc.map (fun v => selector v.toList)


  -- goal after unfolding the `let`s
  intro l i
  -- reuse the fact we already know on the *full* list,
  -- indexed by `Fin.succ l`
  have h := h_act l.succ i

  have h0 : (l.succ.val = 0) = False := by
    -- `l.succ.val` is `l.val + 1`
    simp

    -- fact for the full list, at index `succ l`
  have h_full := h_act l.succ i


  by_cases hl0 : l.val = 0
  ----------------------------------------------------------------
  -- CASE 1 :  l.val = 0  ---------------------------------------
  ----------------------------------------------------------------
  · -- turn it into an equality of Fins
    have l0 : Fin layers_tl.length := ⟨0, by
  -- `l.isLt` already tells us `layers_tl.length > 0`
  -- so `0 < layers_tl.length`
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
    ----------------------------------------------------------------
  -- CASE 2 :  l.val ≠ 0  ---------------------------------------
  ----------------------------------------------------------------
  · -- here  (l.val = 0) = False
    
    have hl0_false : (l.val = 0) = False := by
      simp [hl0]
    have h_shift :=
      prev_results_shift layer_hd layers_tl
        init_vecs acc init_sels sels l.val
        (Nat.le_of_lt l.isLt) rfl rfl h_sel0
    have h_full := h_act l.succ i
    have h_full' := by
      simpa [h_shift] using h_full
    have h_shift_map :
        List.map (fun v => selector v.toList)
          ((evalGridSelector (List.take l.val layers_tl) acc sels).getLast?.getD acc) =
        List.map (fun v => selector v.toList)
          ((evalGridSelector (layer_hd :: List.take l.val layers_tl) init_vecs init_sels).getLast?.getD init_vecs) := by
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
            ((evalGridSelector (List.take ↑l layers_tl) (evalGridSelectorStep init_vecs layer_hd)
              (List.map (fun v => selector v.toList) (evalGridSelectorStep init_vecs layer_hd))).getLastD
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
      ((evalGridSelector (layer_hd :: List.take (↑l) layers_tl) init_vecs init_sels).getLast?.getD init_vecs)

    let selectors2 := List.map (fun v => selector v.toList)
      ((evalGridSelector (List.take (↑l) layers_tl) (evalGridSelectorStep init_vecs layer_hd)
        (List.map (fun v => selector v.toList) (evalGridSelectorStep init_vecs layer_hd))).getLast?.getD
        (evalGridSelectorStep init_vecs layer_hd))

    have h_eq : selectors1 = selectors2 := by
      dsimp [selectors1, selectors2]
      rw [h_shift_map.symm]

      
    change exactlyOneActive (activateLayerFromSelectors selectors2 (layers_tl[↑l]))[↑i].rules

    rw [← h_eq]
    exact h_full'



lemma evalGridSelector_index_eq_get {iv is} {l : ℕ}
  (h : l < (evalGridSelector gs iv is).length) :
  (evalGridSelector gs iv is)[l] = (evalGridSelector gs iv is).get ⟨l, h⟩ := rfl


lemma match_getLast_of_ne_nil {α : Type*} (xs : List α) (acc : α) (h : xs ≠ []) :
    (match match xs with
        | [] => none
        | a :: as => some ((a :: as).getLast (by simp))
     with
     | some x => x
     | none => acc) = xs.getLast h :=
by
  cases xs with
  | nil => contradiction
  | cons a as => simp



@[simp]
lemma Fin.cast_cast {n₁ n₂ n₃ : ℕ} (h₁ : n₁ = n₂) (h₂ : n₂ = n₃) (i : Fin n₁) :
    Fin.cast h₂ (Fin.cast h₁ i) = Fin.cast (h₁.trans h₂) i :=
by cases h₁; cases h₂; rfl

@[simp] lemma Fin.cast_eq {n : ℕ} (h : n = n) (i : Fin n) : Fin.cast h i = i := by cases h; rfl

lemma List.getLast?_cons_getD {α} (x : α) (l : List α) :
  (x :: l).getLast?.getD x = l.getLastD x :=
by
  cases l with
  | nil => simp [List.getLast?, Option.getD, List.getLastD]
  | cons _ _ => simp [List.getLast?, Option.getD, List.getLastD]

lemma evalGridSelector_tail_index_shift_get
  {n : ℕ}
  (layer_hd : GridLayer n) (layers_tl : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n)) (acc : List (List.Vector Bool n))
  (initial_selectors new_selectors : List (List Bool))
  (goal_layer' : Fin layers_tl.length)
  (h_acc : acc = evalGridSelectorStep initial_vectors layer_hd)
  (h_sel : new_selectors = List.map (fun v => selector v.toList) acc)
  :
  let lhs_list := evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors
  let rhs_list := evalGridSelector layers_tl acc new_selectors
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

lemma evalGridSelector_getLastD_shift {n : ℕ}
  (layer_hd : GridLayer n) (layers_tl : List (GridLayer n))
  (initial_vectors acc : List (List.Vector Bool n))
  (initial_selectors new_selectors : List (List Bool))
  (l : ℕ)
  (h_acc : acc = evalGridSelectorStep initial_vectors layer_hd):
  (evalGridSelector (List.take l layers_tl) acc new_selectors).getLastD acc =
  (evalGridSelector (layer_hd :: List.take l layers_tl) initial_vectors initial_selectors).getLast?.getD initial_vectors :=
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
      have eq1 : evalGridSelector (layer_hd :: l' :: ls) initial_vectors initial_selectors
      = initial_vectors :: evalGridSelector.aux acc (l' :: ls) := by
        rw [evalGridSelector]
        rw [h_acc]
        rfl

      have eq2 : (initial_vectors :: evalGridSelector.aux acc (l' :: ls)).getLast?.getD initial_vectors
      = (acc :: evalGridSelector.aux (evalGridSelectorStep acc l') ls).getLastD acc := by
        apply List.getLast?_cons_getD
      rw [eq1, eq2]
      rfl

     

@[simp]
theorem List.getElem_eq_get {α : Type*} (l : List α) (i : Fin l.length) : l[↑i] = l.get i := rfl


theorem full_grid_correctness
  {n : Nat}
  (layers : List (GridLayer n))
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (h_act : RuleActivationCorrect layers initial_vectors initial_selectors)
  (h_sel0 : initial_selectors = List.map (fun v => selector v.toList) initial_vectors)
  (goal_layer : Fin layers.length)
  (goal_idx : Fin (layers.get goal_layer).nodes.length) :
    let prev_results := evalGridSelector (layers.take goal_layer.val) initial_vectors initial_selectors
    let prev_result := prev_results.getLastD initial_vectors
    let selectors := prev_result.map (λ v => selector v.toList)
    let act_layer := activateLayerFromSelectors selectors (layers.get goal_layer)
    let out_idx := Fin.cast (Eq.symm (evalGridSelector_length layers initial_vectors initial_selectors)) goal_layer.succ
    let layer_length_eq := evalGridSelector_layer_length layers initial_vectors initial_selectors goal_layer
    let real_goal_idx := Fin.cast layer_length_eq.symm goal_idx

    ∃ r ∈ act_layer,
      ((evalGridSelector layers initial_vectors initial_selectors).get out_idx).get real_goal_idx
        = r.run prev_result :=
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

        let out_idx : Fin (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).length :=
          Fin.mk 1 (by simp [evalGridSelector_length])

        have layer_length_match :
          ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get out_idx).length = layer_hd.nodes.length :=
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
                -- All these are definitional:


        constructor
        ·   
          have selectors_eq : 
          List.map (fun v => selector v.toList)
            ((evalGridSelector (List.take (↑0) (layer_hd :: layers_tl)) initial_vectors initial_selectors).getLastD initial_vectors)
            = initial_selectors := 
            by 
              simp [evalGridSelector, List.getLastD]
              simp [evalGridSelector.aux]
              have aux_def : evalGridSelector.aux initial_vectors [] = [initial_vectors] := by simp [evalGridSelector.aux]
              rw [selectors_base_eq]
              rw [h_sel0]
          
          convert r_mem
        ·       

          have base_result : (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get ⟨1, by simp [evalGridSelector_length]⟩
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

          have get_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get out_idx)
            = act_layer.map (λ node => node.run initial_vectors) := by
            rw [base_result, eval_layer]

          have lengths_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get out_idx).length
                = (act_layer.map (λ node => node.run initial_vectors)).length :=
            congrArg List.length get_eq       

          have out_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get out_idx).get real_goal_idx
              = (act_layer.map (λ node => node.run initial_vectors)).get (Fin.cast lengths_eq real_goal_idx) := 
              by
                dsimp [evalGridSelectorBase]
                rw [h_sel0] 
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
        RuleActivationCorrect.tail h_sel0 h_act

      -- Inductive hypothesis
      have ih_app := ih acc new_selectors h_act_tl h_sel' goal_layer' goal_idx
      rcases ih_app with ⟨r, r_mem, r_eq⟩

      -- Use your tail index shift lemma to adjust indices
      have shift := evalGridSelector_tail_index_shift_get
        layer_hd layers_tl initial_vectors acc initial_selectors new_selectors goal_layer'
        rfl rfl 

      -- Set up indices: goal_layer'.succ.succ = ↑goal_layer' + 2
      let idx₁ : Fin (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).length :=
        ⟨goal_layer'.val + 2, by
          rw [evalGridSelector_length]; simp [List.length]⟩
      let idx₂ : Fin (evalGridSelector layers_tl acc new_selectors).length :=
        ⟨goal_layer'.val + 1, by
          simp [evalGridSelector_length]⟩

      -- The lengths of the "row" at the indices
      let node_len_eq : ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get idx₁).length
            = ((layer_hd :: layers_tl).get goal_layer'.succ).nodes.length :=
        evalGridSelector_layer_length (layer_hd :: layers_tl) initial_vectors initial_selectors goal_layer'.succ

      let idx' : Fin ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get idx₁).length :=
        Fin.cast node_len_eq.symm goal_idx

      use r
      constructor
      ·
        have selectors_eq :
          List.map (fun v => selector v.toList)
            ((evalGridSelector (layer_hd :: List.take (↑goal_layer') layers_tl) initial_vectors initial_selectors).getLastD initial_vectors)
          =
          List.map (fun v => selector v.toList)
            ((evalGridSelector (List.take (↑goal_layer') layers_tl) acc new_selectors).getLastD acc)
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

              
              rw [←prev_results_shift layer_hd layers_tl initial_vectors acc initial_selectors new_selectors l'
                  (by linarith [goal_layer'.isLt]) rfl rfl h_sel0]
              
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
          (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get idx₁
          = (evalGridSelector layers_tl acc new_selectors).get idx₂ :=
          by exact shift

        have idx₁_def : (Fin.cast (Eq.symm (evalGridSelector_length (layer_hd :: layers_tl) initial_vectors initial_selectors)) goal_layer'.succ.succ) = idx₁ :=
          by rw [Fin.ext_iff]; rfl

        simp [idx₁_def]
        subst idx'
        have idx1_bound : ↑goal_layer' + 1 + 1 < (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).length :=
          by
            rw [evalGridSelector_length]
            simp only [List.length]
            linarith [goal_layer'.isLt]

        have goal_eq : (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors)[↑goal_layer' + 1 + 1]
          = (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get idx₁ :=
          by rw [List.get_eq_getElem]

        simp [goal_eq]

        let real_goal_idx : Fin ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors).get idx₁).length :=
          Fin.cast (Eq.symm node_len_eq) goal_idx

        have real_goal_idx_bound : ↑real_goal_idx < (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors)[idx₁].length :=
          by
            exact Fin.isLt _

        have : (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors)[idx₁][goal_idx]
        = (evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors)[idx₁][real_goal_idx] :=
        by
          apply congr_arg
            ((evalGridSelector (layer_hd :: layers_tl) initial_vectors initial_selectors)[idx₁].get)
          apply Fin.ext
          simp [real_goal_idx]

        simp only [List.getElem_eq_get] at * 
        rw [←eq_at_row] at *

        have last_eq : (evalGridSelector (List.take (↑goal_layer') layers_tl) acc new_selectors).getLastD acc =
          (evalGridSelector (layer_hd :: List.take (↑goal_layer') layers_tl) initial_vectors initial_selectors).getLast?.getD initial_vectors :=
        evalGridSelector_getLastD_shift layer_hd layers_tl initial_vectors acc initial_selectors new_selectors ↑goal_layer' rfl
        
        
        rw [←last_eq]
        congr


        

      





