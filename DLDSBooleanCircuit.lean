import Init
import Mathlib.Data.List.Basic  
import Mathlib.Tactic           
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.GetD
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip


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
  

lemma List.getD_mem {α : Type u} (l : List α) (n : Nat) (d : α) (h : n < l.length) :
    l.getD n d ∈ l :=
by
  rw [l.getD_eq_getElem d h]
  apply List.getElem_mem



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



/-- Type alias: for each rule in a node, where to read activation from (previous layer) -/
abbrev IncomingMap := List (Nat × Nat)
/-- For all nodes in a layer -/
abbrev IncomingMapsLayer := List IncomingMap
/-- For all layers -/
abbrev IncomingMaps := List IncomingMapsLayer

/-- Activates the rules of a node by wiring from previous selectors, according to incoming_map -/
def activateNodeFromSelectors {n : Nat}
  (prev_selectors : List (List Bool))
  (incoming_map   : IncomingMap)
  (node           : CircuitNode n)
: CircuitNode n :=
let new_rules :=
  node.rules.enum.map (fun ⟨i, rule⟩ =>
    let (src_idx, edge_idx) := incoming_map.getD i (0, 0)
    let act := (prev_selectors.getD src_idx []).getD edge_idx false
    match rule.activation with
    | ActivationBits.intro _ =>
        { rule with activation := ActivationBits.intro act }
    | ActivationBits.elim _ _ =>
        { rule with activation := ActivationBits.elim act act }
  );
-- The following lemma: structure update that only changes activation preserves distinctness.
let h_pairwise : new_rules.Pairwise (· ≠ ·) :=
  by
    sorry
    -- We'll use the fact that the map f : Rule n → Rule n that only changes activation is injective.
    -- So pairwise distinctness is preserved under injective maps.
  --   let f := fun (i : Nat) (rule : Rule n) =>
  --     let (src_idx, edge_idx) := incoming_map.getD i (0, 0)
  --     let act := (prev_selectors.getD src_idx []).getD edge_idx false
  --     match rule.activation with
  --     | ActivationBits.intro _ => { rule with activation := ActivationBits.intro act }
  --     | ActivationBits.elim _ _ => { rule with activation := ActivationBits.elim act act };
  --   have inj_f : ∀ i₁ i₂ r₁ r₂, r₁ ≠ r₂ → f i₁ r₁ ≠ f i₂ r₂ :=
  --     by
  --       intros i₁ i₂ r₁ r₂ hne contra
  --       -- If two rules are equal after this update, they must have been the same originally.
  --       -- Structure equality: if fields except activation are equal, and original rules were distinct, then these can't be equal unless r₁ = r₂.
  --       have : r₁ = r₂ := by
  --         -- Compare all fields except activation.
  --         cases r₁; cases r₂; simp at contra; congr
  --         case activation => skip -- activation may differ
  --         case kind => assumption
  --         case combine => assumption
  --       contradiction;
  --   -- Use induction on the rules list to transport pairwise
  --   revert new_rules
  --   generalize h_rules : node.rules.enum.toList = enum_rules
  --   intro new_rules
  --   induction enum_rules with
  --   | nil => simp [List.Pairwise.nil]
  --   | cons (i₁, r₁) rest ih =>
  --     simp only [List.map, List.Pairwise.cons]
  --     split
  --     · exact ih _
  --     · intros (i₂, r₂) h_mem hne
  --       exact inj_f i₁ i₂ r₁ r₂ hne
  -- ;
{ rules := new_rules, pairwise := h_pairwise }


/-- Activates all nodes in a layer via selector wiring -/
def activateLayerFromSelectors {n : Nat}
  (prev_selectors : List (List Bool))
  (incoming_maps  : IncomingMapsLayer)
  (nodes          : List (CircuitNode n))
: List (CircuitNode n) :=
  nodes.enum.map (fun ⟨i, node⟩ =>
    activateNodeFromSelectors prev_selectors (incoming_maps.getD i []) node)

/-- Evaluate the full grid, propagating dependency vectors and selector outputs layer-by-layer. -/
def evalGridSelector {n : Nat}
  (layers         : List (List (CircuitNode n)))
  (incomingMaps   : IncomingMaps)
  (initial_vectors: List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: List (List (List.Vector Bool n)) :=
let rec eval (vecs : List (List.Vector Bool n)) (sels : List (List Bool))
             (ls  : List (List (CircuitNode n)))
             (ims : IncomingMaps) (acc : List (List (List.Vector Bool n))) :=
  match ls, ims with
  | [], [] => (acc.reverse)
  | (nodes :: rest), (maps :: rest_maps) =>
      let act_nodes := activateLayerFromSelectors sels maps nodes
      let outs := act_nodes.map (λ node => node.run vecs)
      let next_selectors := outs.map (λ v => selector v.toList)
      eval outs next_selectors rest rest_maps (outs :: acc)
  | _, _ => acc.reverse -- Defensive fallback
eval initial_vectors initial_selectors layers incomingMaps [initial_vectors]

/-- Query the output dependency vector of a goal node after grid evaluation -/
def goalNodeOutput {n : Nat}
  (results : List (List (List.Vector Bool n)))
  (goal_layer : Nat) (goal_idx : Nat)
: List.Vector Bool n :=
  (results.getD goal_layer []).getD goal_idx (List.Vector.replicate n false)


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


/-- `List.getD_get`: getD at a valid index is the same as get. -/
@[simp]
theorem List.getD_get {α : Type*} (l : List α) (i : Fin l.length) (d : α) :
  l.getD i d = l.get i := by
  sorry
  -- induction l generalizing i with
  -- | nil => cases i
  -- | cons x xs ih =>
  --   cases i with
  --   | mk 0 _   => simp [List.getD, List.get]
  --   | mk (n+1) h =>
  --     have : n < xs.length := Nat.lt_of_succ_lt_succ h
  --     simp [List.getD, List.get, ih ⟨n, this⟩ d]

theorem List.getD_eq_get (l : List α) (i : Nat) (d : α) (h : i < l.length) :
  l.getD i d = l.get ⟨i, h⟩ := by 
  sorry
  
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
  sorry
  -- unfold List.getLastD List.getLast?
  -- -- This unfolds getLastD to getLast?.getD
  -- rfl

lemma List.get?_singleton_zero {α} (x : α) : [x][0]? = some x := by simp

@[simp] lemma List.getLastD_eq_getLast?_getD {α : Type*} (l : List α) (d : α) :
  l.getLastD d = (l.getLast?).getD d := by
  sorry
  -- cases l.reverse <;> simp [List.getLastD, List.getLast?, Option.getD]

lemma nat_min_le_left (a b : Nat) : min a b ≤ a := by
  sorry
  -- cases Nat.le_total a b with
  -- | inl h => simp [min, if_pos h]
  -- | inr h => simp [min, if_neg (Nat.not_le_of_gt h)]; exact Nat.le_refl a


lemma List.length_evalGridSelector {n : Nat} 
  (layers : List (List (CircuitNode n))) 
  (incoming : IncomingMaps)
  (iv : List (List.Vector Bool n)) 
  (is : List (List Bool)) :
  (evalGridSelector layers incoming iv is).length = min layers.length incoming.length := by
  sorry
  -- induction layers generalizing incoming with
  -- | nil => simp [evalGridSelector]
  -- | cons _ tl ih =>
  --   cases incoming with
  --   | nil => simp [evalGridSelector]
  --   | cons _ tlin =>
  --     simp [evalGridSelector, ih]

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

lemma length_evalGridSelector_eval {n : Nat}
  (layers : List (List (CircuitNode n)))
  (incoming : IncomingMaps)
  (iv : List (List.Vector Bool n))
  (is : List (List Bool))
  (acc : List (List (List.Vector Bool n))) :
  (evalGridSelector.eval iv is layers incoming acc).length = min layers.length incoming.length + acc.length := by
  induction layers generalizing incoming acc with
  | nil =>
    cases incoming
    · simp [evalGridSelector, evalGridSelector.eval]
    · simp [evalGridSelector, evalGridSelector.eval]
  | cons l ls ih =>
    cases incoming
    · simp [evalGridSelector, evalGridSelector.eval]
    · simp [evalGridSelector, evalGridSelector.eval]
      sorry
      -- apply ih

lemma getD_zero_eq_match {α : Type*} (l : List α) (d : α) :
  l.getD 0 d = (match l[0]? with | some x => x | none => d) := by
  cases l <;> simp [List.getD, List.get?, Option.getD]

lemma evalGridSelector_length {n : Nat}
  (layers : List (List (CircuitNode n)))
  (incomingMaps : IncomingMaps)
  (initial_vectors: List (List.Vector Bool n))
  (initial_selectors : List (List Bool)) :
  (evalGridSelector layers incomingMaps initial_vectors initial_selectors).length
  = min layers.length incomingMaps.length + 1 := by
  sorry
  -- induction layers generalizing incomingMaps with
  -- | nil =>
  --   cases incomingMaps
  --   · simp [evalGridSelector]
  --   · simp [evalGridSelector]
  -- | cons hd tl ih =>
  --   cases incomingMaps
  --   · simp [evalGridSelector]
  --   · simp [evalGridSelector]
  --     rw [ih]
  --     simp


/--
Predicate asserting that for every node (l, i) in the grid,
the selectors induce exactly one active rule in that node.
-/
def RuleActivationCorrect {n : ℕ}
  (layers : List (List (CircuitNode n)))
  (incomingMaps : IncomingMaps)
  (initial_vectors : List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
: Prop :=
  ∀ (l i : ℕ),
    l < layers.length →
    i < (layers.getD l []).length →
    let prev_results :=
      if l = 0 then initial_vectors
      else
        (evalGridSelector (layers.take l) (incomingMaps.take l)
          initial_vectors initial_selectors).getLastD initial_vectors
    let prev_selectors :=
      if l = 0 then initial_selectors
      else prev_results.map (λ v => selector v.toList)
    let node := (activateLayerFromSelectors prev_selectors
                  (incomingMaps.getD l []) (layers.getD l []))
                |>.getD i { rules := [], pairwise := by simp }
    exactlyOneActive node.rules


/--
The output of the grid at (goal_layer, goal_idx) matches the run of the corresponding active node.
-/
def GoalNodeCorrect {n : ℕ}
  (layers         : List (List (CircuitNode n)))
  (incomingMaps   : IncomingMaps)
  (initial_vectors: List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (goal_layer goal_idx : ℕ)
: Prop :=
  let inputs :=
    if goal_layer = 0 then initial_vectors
    else
      (evalGridSelector (layers.take goal_layer) (incomingMaps.take goal_layer)
        initial_vectors initial_selectors).getLastD initial_vectors

  let selectors :=
    if goal_layer = 0 then initial_selectors
    else inputs.map (λ v => selector v.toList)

  let nodes := activateLayerFromSelectors selectors
                  (incomingMaps.getD goal_layer [])
                  (layers.getD goal_layer [])

  goal_idx < nodes.length ∧
  ∃ (r : CircuitNode n), r ∈ nodes ∧
    (List.getD (List.getD (evalGridSelector layers incomingMaps initial_vectors initial_selectors) goal_layer []) goal_idx (List.Vector.replicate n false)
    = r.run inputs)


theorem full_grid_correctness
  {n : Nat}
  (layers         : List (List (CircuitNode n)))
  (incomingMaps   : IncomingMaps)
  (initial_vectors: List (List.Vector Bool n))
  (initial_selectors : List (List Bool))
  (h_act : RuleActivationCorrect layers incomingMaps initial_vectors initial_selectors)
  (goal_layer goal_idx : Nat)
  (hl : goal_layer < layers.length)
  (hi : goal_idx < (layers.getD goal_layer []).length)
  (hl_maps : goal_layer < incomingMaps.length)
  :
    ∃ r ∈ (activateLayerFromSelectors
             (if goal_layer = 0 then initial_selectors
              else (evalGridSelector (layers.take goal_layer) (incomingMaps.take goal_layer)
                      initial_vectors initial_selectors).getLastD initial_vectors
                  |>.map (λ v => selector v.toList))
             (incomingMaps.getD goal_layer []) (layers.getD goal_layer [])),
      List.getD (List.getD (evalGridSelector layers incomingMaps initial_vectors initial_selectors) goal_layer []) goal_idx (List.Vector.replicate n false)
      = r.run
          (if goal_layer = 0 then initial_vectors
           else (evalGridSelector (layers.take goal_layer) (incomingMaps.take goal_layer)
                   initial_vectors initial_selectors).getLastD initial_vectors)
  :=
by
  induction goal_layer generalizing initial_vectors initial_selectors incomingMaps layers with
  | zero =>
    let nodes := activateLayerFromSelectors initial_selectors (incomingMaps.getD 0 []) (layers.getD 0 [])
    have hlen : nodes.length = (layers.getD 0 []).length := by
      unfold nodes activateLayerFromSelectors; simp
    use nodes.get ⟨goal_idx, hlen.symm ▸ hi⟩, List.get_mem nodes ⟨goal_idx, hlen.symm ▸ hi⟩

    have h_nodes_len : goal_idx < nodes.length := by rw [hlen]; exact hi

    have h1' : exactlyOneActive (nodes.get ⟨goal_idx, h_nodes_len⟩).rules := by
      rw [←hlen] at hi 
      have h1 := h_act 0 goal_idx hl (hlen.symm ▸ hi)
      dsimp [nodes] at h1
      convert h1
      rw [List.getD_get nodes ⟨goal_idx, h_nodes_len⟩ { rules := [], pairwise := by simp }]

    have eval0 : evalGridSelector layers incomingMaps initial_vectors initial_selectors =
      [nodes.map (λ node => node.run initial_vectors)] := by
      dsimp [evalGridSelector, evalGridSelector.eval, nodes, activateLayerFromSelectors]
      simp
      simp [List.getD, List.get?, Option.getD] at *
      simp only [List.getD_zero, List.get?, Option.getD] at nodes ⊢
      ext x
      dsimp
      cases x
      case zero => sorry
      case succ n => sorry

    
    rw [eval0]
    simp only [List.getD, List.getD_cons, List.getD_zero]
    have h_option_get : nodes[goal_idx]? = some nodes[goal_idx] := by
      simp [List.get?_eq_get, h_nodes_len]

    have simplified_goal : ([nodes.map (fun node => node.run initial_vectors)][0]?.getD []) = nodes.map (fun node => node.run initial_vectors) := by rfl
    rw [simplified_goal]
    rw [List.get?_map, h_option_get]
    simp
  | succ l ih =>
    let prev_results :=
      evalGridSelector (layers.take l.succ) (incomingMaps.take l.succ)
        initial_vectors initial_selectors
    let prev_outs := prev_results.getLastD initial_vectors
    let prev_selectors := prev_outs.map (λ v => selector v.toList)
    let nodes := activateLayerFromSelectors prev_selectors (incomingMaps.getD l.succ []) (layers.getD l.succ [])
    have h1 := h_act (l + 1) goal_idx hl hi
    have h_nodes_len : nodes.length = (layers.getD (l+1) []).length := by
      unfold nodes activateLayerFromSelectors; simp
      
    use nodes.get ⟨goal_idx, h_nodes_len ▸ hi⟩, List.get_mem nodes ⟨goal_idx, h_nodes_len ▸ hi⟩

    have eval_succ : evalGridSelector layers incomingMaps initial_vectors initial_selectors =
        evalGridSelector (layers.take (l+1)) (incomingMaps.take (l+1)) initial_vectors initial_selectors ++
          [nodes.map (λ node => node.run prev_outs)] := by
      cases layers with
      | nil => simp at hl
      | cons hd tl =>
        cases incomingMaps with
        | nil =>  
          simp
          exfalso
          exact absurd hl_maps (Nat.not_lt_zero _)
        | cons map_hd map_tl =>
          simp [evalGridSelector]
          dsimp [evalGridSelector, evalGridSelector.eval]
          let cur_outs := List.map (fun node => node.run initial_vectors) (activateLayerFromSelectors initial_selectors map_hd hd)
          let cur_selectors := List.map (fun v => selector v.toList) cur_outs
          sorry

    
          
    rw [eval_succ]

    simp [List.getD_append_right]
    have goal_layer_eq : l + 1 - min (l + 1) layers.length = 0 := by 
      rw [min_eq_left]
      simp
      exact Nat.le_of_lt hl
      
    rw [List.get?_append_right]

    simp [List.length_evalGridSelector, min_eq_left_of_lt hl]
    have goal_layer_eq' : l + 1 - min (l + 1) (List.length incomingMaps) = 0 := by
      rw [min_eq_left]
      simp
      exact Nat.le_of_lt hl_maps

    rw [goal_layer_eq']  
    rw [List.get?_singleton_zero]
    simp [List.getD]
    
    have h_option_get : nodes[goal_idx]? = some nodes[goal_idx] := by
      simp [List.get?_eq_get, hi]

    simp [List.get?_map, h_option_get]
    


    let cur_prev_outs := (evalGridSelector (List.take (l + 1) layers) (List.take (l + 1) incomingMaps) initial_vectors    initial_selectors).getLastD initial_vectors

    have prev_outs_eq : cur_prev_outs = (evalGridSelector (List.take (l + 1) layers) (List.take (l + 1) incomingMaps) initial_vectors initial_selectors).getLast?.getD initial_vectors :=
      List.getLastD_eq_getLast_getD _ _

    let prev_outs := cur_prev_outs

    change nodes[goal_idx].run ((evalGridSelector (List.take (l + 1) layers) (List.take (l + 1) incomingMaps) initial_vectors initial_selectors).getLastD initial_vectors) =
       nodes[goal_idx].run ((evalGridSelector (List.take (l + 1) layers) (List.take (l + 1) incomingMaps) initial_vectors initial_selectors).getLast?.getD initial_vectors)


    rw [List.getLastD_eq_getLast_getD]
    rw [ge_iff_le]
    rw [List.length_evalGridSelector]
    rw [List.length_take, List.length_take]
    simp






    


    




