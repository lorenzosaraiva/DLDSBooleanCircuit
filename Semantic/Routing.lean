import Semantic.NodeCorrectness

open scoped Classical

namespace Semantic

/-!
# Token routing and node evaluation.
-/

/-- Token flowing through the circuit, carrying Definition 3 dependency data.
    - `origin_column`: Fixed; used for path lookup
    - `source_column`: Updated each step; indicates immediate predecessor
    - `dep_vector`: accumulated dependency bitstring `b⃗` of Definition 3 -/
structure Token (n : Nat) where
  origin_column : Nat
  source_column : Nat
  current_level : Nat
  current_column : Nat
  /-- Dependency bitstring `b⃗` carried by this token, in the formula-column basis. -/
  dep_vector : List.Vector Bool n
  input_label : Nat   -- destination input wire: 0 = repetition/self, k+1 = k-th non-rep wire
  deriving Inhabited

/-- Input label carried by a nonempty token group; malformed/free paths use the first. -/
def tokenInputLabel {n : Nat} (tokens : List (Token n)) : Nat :=
  (tokens.head?.map (·.input_label)).getD 0

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
    Each step is `(targetColumn, inputLabel)`: `targetColumn = 0` stops;
    `targetColumn = k+1` routes to column `k`; `inputLabel` names the
    destination node's input wire. -/
abbrev PathInput := List (List (Nat × Nat))



/-- Initialize tokens: one per column at the top level with initial dependency vectors. -/
def initialize_tokens {n : Nat}
    (initial_vectors : List (List.Vector Bool n))
    (top_level : Nat) : List (Token n) :=
  initial_vectors.zipIdx.map fun (vec, col) =>
    { origin_column := col
      source_column := col
      current_level := top_level
      current_column := col
      dep_vector := vec
      input_label := 0 }   -- top-layer resting tokens enter the rep/self wire

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
        -- `.1` = target column; `.2` = destination input label.
        if edge_choice.1 = 0 then
          none
        else
          let target_column := edge_choice.1 - 1
          if h_out : token.current_column < outputs.length then
            some { origin_column := token.origin_column
                   source_column := token.current_column
                   current_level := current_level - 1
                   current_column := target_column
                   dep_vector := outputs.get ⟨token.current_column, h_out⟩
                   input_label := edge_choice.2 }
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



/-- Set ALL of a rule's activation bits to `b` (one-hot on/off), preserving its
    type and `ruleId`. -/
def setActivation {n : Nat} (r : Rule n) (b : Bool) : Rule n :=
  { r with activation := match r.activation with
    | ActivationBits.intro _      => ActivationBits.intro b
    | ActivationBits.elim _ _      => ActivationBits.elim b b
    | ActivationBits.repetition _ => ActivationBits.repetition b }

/-- One-hot selector test: is the rule at index `i` (out of `rules_len`) the one
    named by `sel`? Encoding: `0` = none, `1` = rep (the last rule),
    `k+2` = rule index `k`. -/
def selectorActive (rules_len sel i : Nat) : Bool :=
  match sel with
  | 0 => false
  | 1 => decide (i = rules_len - 1)
  | Nat.succ (Nat.succ k) => decide (i = k)

/-- Selector-driven one-hot activation (index-threaded fold). Replaces the old
    availability-based `set_rule_activation`; the activation no longer reads
    `available_sources` at all — exactly the rule named by `sel` is set active. -/
def activateRulesAux {n : Nat} (sel rules_len : Nat) :
    Nat → List (Rule n) → List (Rule n)
  | _, [] => []
  | idx, r :: rs =>
      setActivation r (selectorActive rules_len sel idx)
        :: activateRulesAux sel rules_len (idx + 1) rs

/-- Activation preserves rule IDs (needed for nodupIds invariant). -/
lemma activateRulesAux_ids {n : Nat} (sel rules_len : Nat) :
    ∀ idx (rs : List (Rule n)),
      (activateRulesAux sel rules_len idx rs).map (·.ruleId) = rs.map (·.ruleId)
  | idx, [] => by simp [activateRulesAux]
  | idx, r :: rs => by
      have ih := activateRulesAux_ids sel rules_len (idx + 1) rs
      simp [activateRulesAux, setActivation, ih]

lemma activateRulesAux_length {n : Nat} (sel rules_len : Nat) :
    ∀ idx (rs : List (Rule n)),
      (activateRulesAux sel rules_len idx rs).length = rs.length
  | idx, [] => by simp [activateRulesAux]
  | idx, _ :: rs => by
      simp [activateRulesAux, activateRulesAux_length sel rules_len (idx + 1) rs]

/-- Activate a node's rules from an internally inferred rule selector (one-hot). -/
def activate_node_from_tokens {n : Nat}
    (node : CircuitNode n) (sel : Nat) : CircuitNode n :=
  let activated_rules := activateRulesAux sel node.rules.length 0 node.rules
  { rules := activated_rules
    nodupIds := by
      have h_ids : activated_rules.map (·.ruleId) = node.rules.map (·.ruleId) :=
        activateRulesAux_ids sel node.rules.length 0 node.rules
      simpa [activated_rules, h_ids] using node.nodupIds }




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

/-- Flatten all non-repetition incoming wires into `(ruleIdx, slotIdx, sourceColumn)`.
    The repetition/self wire is reserved as input label `0`; flattened labels are
    therefore `k+1` for the `k`-th entry of this list. -/
def flattenIncomingAux : Nat → NodeIncoming → List (Nat × Nat × Nat)
  | _, [] => []
  | _, [_rep] => []
  | ruleIdx, inc :: rest =>
      inc.zipIdx.map (fun ((src, _edge), slotIdx) => (ruleIdx, slotIdx, src)) ++
        flattenIncomingAux (ruleIdx + 1) rest

def flattenIncoming (node_incoming : NodeIncoming) : List (Nat × Nat × Nat) :=
  flattenIncomingAux 0 node_incoming

def listGet? {α : Type*} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, Nat.succ k => listGet? xs k

/-- Decode a destination input label into `(ruleIdx, slotIdx, sourceColumn)`.
    Label `0` is always the repetition/self input (the last rule's first slot).
    Labels `k+1` address the flattened non-repetition incoming wires. -/
def decodeInputLabel (node_incoming : NodeIncoming) (label : Nat) :
    Option (Nat × Nat × Nat) :=
  match label with
  | 0 =>
      if 0 < node_incoming.length then
        let ruleIdx := node_incoming.length - 1
        match node_incoming[ruleIdx]! with
        | [] => none
        | (src, _edge) :: _ => some (ruleIdx, 0, src)
      else none
  | Nat.succ k =>
      listGet? (flattenIncoming node_incoming) k

/-- Encode `(ruleIdx, slotIdx)` back to an input label. For the final repetition
    rule this returns `0`; otherwise it returns the flattened non-repetition
    label. Invalid pairs also fall back to `0`, making them dead-end unless they
    really are the repetition/self wire at the destination. -/
def inputLabelForRuleSlotAux (targetRule targetSlot ruleIdx offset : Nat) :
    NodeIncoming → Nat
  | [] => 0
  | [_rep] => 0
  | inc :: rest =>
      if ruleIdx = targetRule then
        if targetSlot < inc.length then offset + targetSlot + 1 else 0
      else
        inputLabelForRuleSlotAux targetRule targetSlot (ruleIdx + 1)
          (offset + inc.length) rest

def inputLabelForRuleSlot (node_incoming : NodeIncoming) (ruleIdx slotIdx : Nat) : Nat :=
  if ruleIdx + 1 = node_incoming.length then 0
  else inputLabelForRuleSlotAux ruleIdx slotIdx 0 0 node_incoming

/-- Sum of the arities of rules before index `r`. In the flattened encoding this
    is the global slot offset of rule `r`; the final repetition rule contributes
    nothing to `flattenIncoming`, but its prefix is still well-defined. -/
def incomingPrefixLen : NodeIncoming → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | inc :: rest, r + 1 => inc.length + incomingPrefixLen rest r

lemma listGet?_append_left {α : Type*} (xs ys : List α) {i : Nat}
    (hi : i < xs.length) :
    listGet? (xs ++ ys) i = listGet? xs i := by
  induction xs generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
      cases i with
      | zero => simp [listGet?]
      | succ i =>
          simp only [List.length_cons] at hi
          simp [listGet?, ih (Nat.lt_of_succ_lt_succ hi)]

lemma listGet?_append_right {α : Type*} (xs ys : List α) (i : Nat) :
    listGet? (xs ++ ys) (xs.length + i) = listGet? ys i := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [listGet?, ih, Nat.succ_add]

lemma listGet?_map_zipIdx_source_start
    (inc : RuleIncoming) (ruleIdx start slotIdx : Nat)
    (hs : slotIdx < inc.length) :
    listGet?
        ((inc.zipIdx start).map
          (fun ((src, _edge), slotIdx) => (ruleIdx, slotIdx, src)))
        slotIdx =
      some (ruleIdx, start + slotIdx, (inc.get ⟨slotIdx, hs⟩).1) := by
  induction inc generalizing start slotIdx with
  | nil => simp at hs
  | cons x xs ih =>
      cases slotIdx with
      | zero =>
          cases x
          simp [listGet?]
      | succ slotIdx =>
          simp only [List.length_cons] at hs
          simp [List.zipIdx_cons, listGet?]
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
            using ih (start + 1) slotIdx (Nat.lt_of_succ_lt_succ hs)

lemma listGet?_map_zipIdx_source
    (inc : RuleIncoming) (ruleIdx slotIdx : Nat)
    (hs : slotIdx < inc.length) :
    listGet?
        (inc.zipIdx.map
          (fun ((src, _edge), slotIdx) => (ruleIdx, slotIdx, src)))
        slotIdx =
      some (ruleIdx, slotIdx, (inc.get ⟨slotIdx, hs⟩).1) := by
  simpa using listGet?_map_zipIdx_source_start inc ruleIdx 0 slotIdx hs

lemma flattenIncomingAux_get
    (ni : NodeIncoming) (start r s : Nat)
    (hr : r + 1 < ni.length)
    (hs : s < (ni.get ⟨r, Nat.lt_of_succ_lt hr⟩).length) :
    listGet? (flattenIncomingAux start ni) (incomingPrefixLen ni r + s) =
      some (start + r, s,
        ((ni.get ⟨r, Nat.lt_of_succ_lt hr⟩).get ⟨s, hs⟩).1) := by
  induction ni generalizing start r with
  | nil => simp at hr
  | cons inc rest ih =>
      cases r with
      | zero =>
          cases rest with
          | nil => simp at hr
          | cons rep rest' =>
              simp only [incomingPrefixLen, Nat.zero_add]
              unfold flattenIncomingAux
              rw [listGet?_append_left]
              · exact listGet?_map_zipIdx_source inc start s hs
              · simpa using hs
      | succ r =>
          simp only [List.length_cons] at hr
          have hr_rest : r + 1 < rest.length := Nat.lt_of_succ_lt_succ hr
          have hslot : s < (rest.get ⟨r, Nat.lt_of_succ_lt hr_rest⟩).length := by
            simpa using hs
          cases rest with
          | nil => simp at hr_rest
          | cons rep rest' =>
              simp only [incomingPrefixLen, List.get_cons_succ]
              unfold flattenIncomingAux
              have happ := listGet?_append_right
                (inc.zipIdx.map
                  (fun ((src, _edge), slotIdx) => (start, slotIdx, src)))
                (flattenIncomingAux (start + 1) (rep :: rest'))
                (incomingPrefixLen (rep :: rest') r + s)
              simp only [List.length_map, List.length_zipIdx] at happ
              rw [show inc.length + incomingPrefixLen (rep :: rest') r + s =
                    inc.length + (incomingPrefixLen (rep :: rest') r + s) by omega]
              rw [happ]
              simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
                using ih (start + 1) r hr_rest hslot

lemma listGet?_mem {α : Type*} :
    ∀ {xs : List α} {i : Nat} {x : α}, listGet? xs i = some x → x ∈ xs
  | [], _, _, h => by simp [listGet?] at h
  | y :: ys, 0, x, h => by
      simp [listGet?] at h
      subst h
      simp
  | y :: ys, Nat.succ i, x, h => by
      simp [listGet?] at h
      exact List.mem_cons_of_mem y (listGet?_mem h)

lemma fst_mem_of_mem_zipIdx {α : Type*} :
    ∀ {xs : List α} {start : Nat} {x : α} {idx : Nat},
      (x, idx) ∈ xs.zipIdx start → x ∈ xs
  | [], _, _, _, h => by simp at h
  | y :: ys, start, x, idx, h => by
      simp only [List.zipIdx_cons, List.mem_cons] at h
      cases h with
      | inl hhead =>
          cases hhead
          simp
      | inr htail =>
          exact List.mem_cons_of_mem y (fst_mem_of_mem_zipIdx htail)

lemma mem_flattenIncomingAux_source
    (ni : NodeIncoming) (start : Nat) {r slot src : Nat}
    (hmem : (r, slot, src) ∈ flattenIncomingAux start ni) :
    start ≤ r ∧
      ∃ inc, ni[r - start]? = some inc ∧ src ∈ inc.map Prod.fst := by
  induction ni generalizing start r slot src with
  | nil =>
      simp [flattenIncomingAux] at hmem
  | cons inc rest ih =>
      cases rest with
      | nil =>
          simp [flattenIncomingAux] at hmem
      | cons rep rest' =>
          simp only [flattenIncomingAux, List.mem_append] at hmem
          cases hmem with
          | inl hhead =>
              rw [List.mem_map] at hhead
              obtain ⟨item, hzip, hitem⟩ := hhead
              rcases item with ⟨⟨src0, edge0⟩, slot0⟩
              simp at hitem
              rcases hitem with ⟨hr, hslot, hsrc⟩
              subst r
              subst slot
              subst src
              have hzip_info := List.mem_zipIdx hzip
              rcases hzip_info with ⟨i, hi, hget⟩
              constructor
              · omega
              · refine ⟨inc, ?_, ?_⟩
                · simp
                · rw [List.mem_map]
                  refine ⟨(src0, edge0), ?_, rfl⟩
                  exact fst_mem_of_mem_zipIdx hzip
          | inr htail =>
              have ht := ih (start + 1) htail
              rcases ht with ⟨hle, incTail, hentry, hsrc⟩
              constructor
              · omega
              · refine ⟨incTail, ?_, hsrc⟩
                have hsub : r - start = Nat.succ (r - (start + 1)) := by
                  omega
                rw [hsub]
                simpa using hentry

lemma mem_flattenIncomingAux_rule_bound
    (ni : NodeIncoming) (start : Nat) {r slot src : Nat}
    (hmem : (r, slot, src) ∈ flattenIncomingAux start ni) :
    r + 1 < start + ni.length := by
  induction ni generalizing start r slot src with
  | nil =>
      simp [flattenIncomingAux] at hmem
  | cons inc rest ih =>
      cases rest with
      | nil =>
          simp [flattenIncomingAux] at hmem
      | cons rep rest' =>
          simp only [flattenIncomingAux, List.mem_append] at hmem
          cases hmem with
          | inl hhead =>
              rw [List.mem_map] at hhead
              obtain ⟨item, _hzip, hitem⟩ := hhead
              rcases item with ⟨⟨src0, edge0⟩, slot0⟩
              simp at hitem
              rcases hitem with ⟨hr, _hslot, _hsrc⟩
              subst r
              simp
          | inr htail =>
              have htailBound := ih (start + 1) htail
              simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using htailBound

lemma decodeInputLabel_succ_source_mem
    {ni : NodeIncoming} {k ruleIdx slot src : Nat}
    (hdec : decodeInputLabel ni (Nat.succ k) = some (ruleIdx, slot, src)) :
    ∃ inc, ni[ruleIdx]? = some inc ∧ src ∈ inc.map Prod.fst := by
  unfold decodeInputLabel flattenIncoming at hdec
  have hflat : (ruleIdx, slot, src) ∈ flattenIncomingAux 0 ni :=
    listGet?_mem hdec
  have hsrc := mem_flattenIncomingAux_source ni 0 hflat
  rcases hsrc with ⟨_, inc, hentry, hmem⟩
  refine ⟨inc, ?_, hmem⟩
  simpa using hentry

lemma decodeInputLabel_succ_nonrep
    {ni : NodeIncoming} {k ruleIdx slot src : Nat}
    (hdec : decodeInputLabel ni (Nat.succ k) = some (ruleIdx, slot, src)) :
    ruleIdx + 1 ≠ ni.length := by
  intro hlast
  unfold decodeInputLabel flattenIncoming at hdec
  have hflat : (ruleIdx, slot, src) ∈ flattenIncomingAux 0 ni :=
    listGet?_mem hdec
  have hruleLt : ruleIdx + 1 < 0 + ni.length :=
    mem_flattenIncomingAux_rule_bound ni 0 hflat
  omega

lemma inputLabelForRuleSlotAux_eq_prefix
    (ni : NodeIncoming) (start offset r s : Nat)
    (hr : r + 1 < ni.length)
    (hs : s < (ni.get ⟨r, Nat.lt_of_succ_lt hr⟩).length) :
    inputLabelForRuleSlotAux (start + r) s start offset ni =
      offset + incomingPrefixLen ni r + s + 1 := by
  induction ni generalizing start offset r with
  | nil => simp at hr
  | cons inc rest ih =>
      cases r with
      | zero =>
          cases rest with
          | nil => simp at hr
          | cons rep rest' =>
              simp only [incomingPrefixLen]
              have hs' : s < inc.length := by simpa using hs
              simp [inputLabelForRuleSlotAux, hs']
      | succ r =>
          simp only [List.length_cons] at hr
          have hr_rest : r + 1 < rest.length := Nat.lt_of_succ_lt_succ hr
          have hslot : s < (rest.get ⟨r, Nat.lt_of_succ_lt hr_rest⟩).length := by
            simpa using hs
          cases rest with
          | nil => simp at hr_rest
          | cons rep rest' =>
              have hne : ¬ start = start + (r + 1) := by omega
              have htarget : start + (r + 1) = (start + 1) + r := by omega
              simp only [incomingPrefixLen]
              simp [inputLabelForRuleSlotAux]
              rw [htarget, ih (start + 1) (offset + inc.length) r hr_rest hslot]
              omega

lemma inputLabelForRuleSlot_decode_roundtrip_nonrep
    (ni : NodeIncoming) (r s : Nat)
    (hr : r + 1 < ni.length)
    (hs : s < (ni.get ⟨r, Nat.lt_of_succ_lt hr⟩).length) :
    decodeInputLabel ni (inputLabelForRuleSlot ni r s) =
      some (r, s, ((ni.get ⟨r, Nat.lt_of_succ_lt hr⟩).get ⟨s, hs⟩).1) := by
  have haux := inputLabelForRuleSlotAux_eq_prefix ni 0 0 r s hr hs
  have haux' :
      inputLabelForRuleSlotAux r s 0 0 ni =
        incomingPrefixLen ni r + s + 1 := by
    simpa [Nat.zero_add] using haux
  have hnotlast : ¬ r + 1 = ni.length := by omega
  have hlabel : incomingPrefixLen ni r + s + 1 =
      Nat.succ (incomingPrefixLen ni r + s) := by omega
  unfold inputLabelForRuleSlot
  rw [if_neg hnotlast, haux', hlabel]
  unfold decodeInputLabel flattenIncoming
  simpa [Nat.zero_add] using flattenIncomingAux_get ni 0 r s hr hs

lemma decode_repetition_head
    (inc : RuleIncoming) (ruleIdx : Nat)
    (hhead : 0 < inc.length) :
    (match inc with
      | [] => none
      | (src, _edge) :: _ => some (ruleIdx, 0, src)) =
      some (ruleIdx, 0, (inc.get ⟨0, hhead⟩).1) := by
  cases inc with
  | nil => simp at hhead
  | cons p ps =>
      cases p
      simp

lemma inputLabelForRuleSlot_decode_roundtrip_rep
    (ni : NodeIncoming) (r : Nat)
    (hr : r + 1 = ni.length)
    (hidx : r < ni.length)
    (hs : 0 < (ni.get ⟨r, hidx⟩).length) :
    decodeInputLabel ni (inputLabelForRuleSlot ni r 0) =
      some (r, 0, ((ni.get ⟨r, hidx⟩).get ⟨0, hs⟩).1) := by
  have hpos : 0 < ni.length := by omega
  have hidx : ni.length - 1 = r := by omega
  subst r
  unfold inputLabelForRuleSlot
  rw [if_pos (by omega)]
  unfold decodeInputLabel
  rw [if_pos hpos]
  simp only
  rw [List.getElem!_eq_getElem?_getD]
  rw [List.getElem?_eq_getElem hidx]
  simp only [Option.getD_some]
  simpa [List.get_eq_getElem] using
    decode_repetition_head (ni.get ⟨ni.length - 1, hidx⟩) (ni.length - 1) hs

def ruleSelectorForIndex (ruleIdx : Nat) : Nat :=
  ruleIdx + 2

/-- Pure-lookup error condition for a node, semantics (c): arrivals are an error
    unless their input labels decode to one rule, each required slot is filled
    exactly once, and every token's `source_column` matches the static incoming
    wire named by its label. -/
def nodeError {n : Nat}
    (node : CircuitNode n) (node_incoming : NodeIncoming)
    (tokens : List (Token n)) : Bool :=
  match tokens with
  | [] => false
  | t :: _ =>
      match decodeInputLabel node_incoming t.input_label with
      | none => true
      | some (r, _slot, _src) =>
          let reqd := node_incoming[r]!
          let arity := reqd.length
          let ruleExists := decide (r < node.rules.length)
          let labelsOK := tokens.all (fun s =>
            match decodeInputLabel node_incoming s.input_label with
            | some (r', slot', src') =>
                decide (r' = r ∧ slot' < arity ∧ s.source_column = src')
            | none => false)
          let countOK := decide (tokens.length = arity)
          let slotsComplete := (List.range arity).all (fun i =>
            tokens.any (fun s =>
              match decodeInputLabel node_incoming s.input_label with
              | some (r', slot', src') =>
                  decide (r' = r ∧ slot' = i ∧ s.source_column = src')
              | none => false))
          ! (ruleExists && labelsOK && countOK && slotsComplete)

def selectedRuleIndex? {n : Nat}
    (node_incoming : NodeIncoming) (tokens : List (Token n)) : Option Nat :=
  match tokens with
  | [] => none
  | t :: _ => (decodeInputLabel node_incoming t.input_label).map (fun x => x.1)

/-- Local positive form of the pure-lookup detector: if a nonempty token group
    decodes to one existing rule, has exactly that rule's arity, every token names
    a valid slot/source for that rule, and every required slot is present, then
    `nodeError` is false. This is the evaluator-local fact needed by the tree
    routing proof; it does not claim arbitrary paths are error-free. -/
lemma nodeError_false_of_exact_rule_slots {n : Nat}
    (node : CircuitNode n) (node_incoming : NodeIncoming)
    (t : Token n) (ts : List (Token n))
    (r slot src : Nat)
    (hhead : decodeInputLabel node_incoming t.input_label = some (r, slot, src))
    (hrule : r < node.rules.length)
    (hlabels :
      ∀ s ∈ t :: ts,
        ∃ slot' src',
          decodeInputLabel node_incoming s.input_label = some (r, slot', src') ∧
          slot' < (node_incoming[r]?.getD default).length ∧
          s.source_column = src')
    (hcount : (t :: ts).length = (node_incoming[r]?.getD default).length)
    (hcomplete :
      ∀ i, i < (node_incoming[r]?.getD default).length →
        ∃ s ∈ t :: ts, ∃ src',
          decodeInputLabel node_incoming s.input_label = some (r, i, src') ∧
          s.source_column = src') :
    nodeError node node_incoming (t :: ts) = false := by
  classical
  unfold nodeError
  simp only [hhead]
  have hruleBool : decide (r < node.rules.length) = true := by
    simp [hrule]
  have hcountBool :
      decide ((t :: ts).length = (node_incoming[r]?.getD default).length) = true := by
    simp [hcount]
  have hlabelsOK :
      (List.all (t :: ts) fun s =>
        match decodeInputLabel node_incoming s.input_label with
        | some (r', slot', src') =>
            decide (r' = r ∧ slot' < (node_incoming[r]?.getD default).length ∧
              s.source_column = src')
        | none => false) = true := by
    rw [List.all_eq_true]
    intro s hs
    obtain ⟨slot', src', hdec, hslot, hsrc⟩ := hlabels s hs
    simp [hdec, hslot, hsrc]
  have hslotsComplete :
      (List.range (node_incoming[r]?.getD default).length).all (fun i =>
        (t :: ts).any (fun s =>
          match decodeInputLabel node_incoming s.input_label with
          | some (r', slot', src') =>
              decide (r' = r ∧ slot' = i ∧ s.source_column = src')
          | none => false)) = true := by
    rw [List.all_eq_true]
    intro i hi
    have hi_lt : i < (node_incoming[r]?.getD default).length :=
      List.mem_range.mp hi
    obtain ⟨s, hs, src', hdec, hsrc⟩ := hcomplete i hi_lt
    rw [List.any_eq_true]
    refine ⟨s, hs, ?_⟩
    simp [hdec, hsrc]
  have hlabelsProp :
      (match decodeInputLabel node_incoming t.input_label with
          | some (r', slot', src') =>
              decide (r' = r) &&
                (decide (slot' < (node_incoming[r]?.getD default).length) &&
                  decide (t.source_column = src'))
          | none => false) = true ∧
      ∀ x ∈ ts,
        (match decodeInputLabel node_incoming x.input_label with
          | some (r', slot', src') =>
              decide (r' = r) &&
                (decide (slot' < (node_incoming[r]?.getD default).length) &&
                  decide (x.source_column = src'))
          | none => false) = true := by
    constructor
    · obtain ⟨slot', src', hdec, hslot, hsrc⟩ :=
        hlabels t (List.Mem.head ts)
      simp [hdec, hslot, hsrc]
    · intro x hx
      obtain ⟨slot', src', hdec, hslot, hsrc⟩ :=
        hlabels x (List.mem_cons_of_mem t hx)
      simp [hdec, hslot, hsrc]
  have hcountProp : ts.length + 1 = (node_incoming[r]?.getD default).length := by
    simpa using hcount
  have hslotsProp :
      ∀ x < (node_incoming[r]?.getD default).length,
        (match decodeInputLabel node_incoming t.input_label with
          | some (r', slot', src') =>
              decide (r' = r) &&
                (decide (slot' = x) && decide (t.source_column = src'))
          | none => false) = true ∨
        ∃ x_1, x_1 ∈ ts ∧
          ((match decodeInputLabel node_incoming x_1.input_label with
            | some (r', slot', src') =>
                decide (r' = r) &&
                  (decide (slot' = x) && decide (x_1.source_column = src'))
            | none => false) = true) := by
    intro x hx
    obtain ⟨s, hs, src', hdec, hsrc⟩ := hcomplete x hx
    cases hs with
    | head =>
        left
        simp [hdec, hsrc]
    | tail _ htail =>
        right
        refine ⟨s, htail, ?_⟩
        simp [hdec, hsrc]
  simp only [List.getElem!_eq_getElem?_getD]
  rw [hlabelsOK, hcountBool, hslotsComplete]
  simp [hruleBool]

def evaluate_node {n : Nat}
  (node : CircuitNode n)
  (node_incoming : NodeIncoming)
  (tokens_at_node : List (Token n))
  : (List.Vector Bool n) × Bool :=

  if tokens_at_node.isEmpty then
    (List.Vector.replicate n false, false)
  else if nodeError node node_incoming tokens_at_node then
    -- semantics (c): dead-end or incoherent labels ⇒ structural error
    (List.Vector.replicate n false, true)
  else
    -- coherent: fire the uniquely named rule (one-hot) and route its slot-ordered
    -- inputs by `node_incoming`; the error flag is governed by `nodeError`, not by
    -- input availability, and not by the head token alone.
    match selectedRuleIndex? node_incoming tokens_at_node with
    | none => (List.Vector.replicate n false, true)
    | some ruleIdx =>
        let available_inputs := tokens_at_node.map fun t => (t.source_column, t.dep_vector)
        let sel := ruleSelectorForIndex ruleIdx
        ((node_logic_with_routing (activate_node_from_tokens node sel).rules node_incoming available_inputs).fst,
         false)

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



lemma activateRulesAux_eq_zipIdx_map {n : Nat} (sel rules_len : Nat)
    (rules : List (Rule n)) (start : Nat) :
    activateRulesAux sel rules_len start rules =
    (rules.zipIdx start).map (fun x => setActivation x.1 (selectorActive rules_len sel x.2)) := by
  induction rules generalizing start with
  | nil => simp [activateRulesAux, List.zipIdx]
  | cons r rs ih =>
    simp only [activateRulesAux, List.zipIdx_cons, List.map_cons]
    rw [ih]

lemma activateRulesAux_eq_zipIdx_map_zero {n : Nat} (sel rules_len : Nat)
    (rules : List (Rule n)) :
    activateRulesAux sel rules_len 0 rules =
    rules.zipIdx.map (fun x => setActivation x.1 (selectorActive rules_len sel x.2)) :=
  activateRulesAux_eq_zipIdx_map sel rules_len rules 0




lemma is_rule_active_setActivation {n : Nat} (r : Rule n) (b : Bool) :
    is_rule_active (setActivation r b) = b := by
  rcases r with ⟨ruleId, activation, type, combine⟩
  cases activation <;> simp [setActivation, is_rule_active]

lemma selectorActive_true_unique {rules_len sel i j : Nat}
    (hi : selectorActive rules_len sel i = true)
    (hj : selectorActive rules_len sel j = true) :
    i = j := by
  cases sel with
  | zero =>
      simp [selectorActive] at hi
  | succ sel' =>
      cases sel' with
      | zero =>
          simp [selectorActive] at hi hj
          omega
      | succ k =>
          simp [selectorActive] at hi hj
          omega

lemma zipIdx_pair_eq_of_snd_eq {α : Type*} {l : List α} {start : Nat}
    {x y : α × Nat}
    (hx : x ∈ l.zipIdx start) (hy : y ∈ l.zipIdx start)
    (h_snd : x.2 = y.2) :
    x = y := by
  rw [List.mem_iff_get] at hx hy
  obtain ⟨⟨i, hi⟩, hx_get⟩ := hx
  obtain ⟨⟨j, hj⟩, hy_get⟩ := hy
  have hx_get' : x = (l.zipIdx start).get ⟨i, hi⟩ := hx_get.symm
  have hy_get' : y = (l.zipIdx start).get ⟨j, hj⟩ := hy_get.symm
  have hx_snd : x.2 = start + i := by
    rw [hx_get']
    rw [List.get_eq_getElem]
    simp [List.getElem_zipIdx]
  have hy_snd : y.2 = start + j := by
    rw [hy_get']
    rw [List.get_eq_getElem]
    simp [List.getElem_zipIdx]
  have hij : i = j := by omega
  have hfin : (⟨i, hi⟩ : Fin (l.zipIdx start).length) = ⟨j, hj⟩ := by
    ext
    exact hij
  rw [hx_get', hy_get', hfin]

lemma activateRulesAux_active_unique {n : Nat} (sel rules_len start : Nat)
    (rules : List (Rule n)) :
    ∀ r₁ ∈ activateRulesAux sel rules_len start rules,
      is_rule_active r₁ = true →
    ∀ r₂ ∈ activateRulesAux sel rules_len start rules,
      is_rule_active r₂ = true → r₂ = r₁ := by
  intro r₁ hr₁ hact₁ r₂ hr₂ hact₂
  rw [activateRulesAux_eq_zipIdx_map] at hr₁ hr₂
  rw [List.mem_map] at hr₁ hr₂
  obtain ⟨x₁, hx₁, rfl⟩ := hr₁
  obtain ⟨x₂, hx₂, rfl⟩ := hr₂
  simp only [is_rule_active_setActivation] at hact₁ hact₂
  have hidx : x₂.2 = x₁.2 := selectorActive_true_unique hact₂ hact₁
  have hpair : x₂ = x₁ := zipIdx_pair_eq_of_snd_eq hx₂ hx₁ hidx
  rw [hpair]

lemma activate_node_exactlyOne_of_any {n : Nat} (node : CircuitNode n) (sel : Nat)
    (h_any : (extract_activations (activate_node_from_tokens node sel).rules).any (· = true) = true) :
    exactlyOneActive (activate_node_from_tokens node sel).rules := by
  rw [List.any_eq_true] at h_any
  obtain ⟨b, hb_mem, hb_true⟩ := h_any
  simp at hb_true
  subst b
  simp only [extract_activations] at hb_mem
  rw [List.mem_map] at hb_mem
  obtain ⟨r, hr_mem, hr_act⟩ := hb_mem
  refine ⟨r, hr_mem, hr_act, ?_⟩
  intro r' hr'_mem hr'_act
  exact activateRulesAux_active_unique sel node.rules.length 0 node.rules
    r (by simpa [activate_node_from_tokens] using hr_mem) hr_act
    r' (by simpa [activate_node_from_tokens] using hr'_mem) hr'_act

lemma activateRulesAux_get {n : Nat} (sel rules_len start : Nat) :
    forall (rules : List (Rule n)) (i : Nat)
      (hi : i < rules.length)
      (hi' : i < (activateRulesAux sel rules_len start rules).length),
      (activateRulesAux sel rules_len start rules).get ⟨i, hi'⟩ =
        setActivation (rules.get ⟨i, hi⟩)
          (selectorActive rules_len sel (start + i)) := by
  intro rules
  induction rules generalizing start with
  | nil =>
      intro i hi _
      simp at hi
  | cons r rs ih =>
      intro i hi hi'
      cases i with
      | zero =>
          simp [activateRulesAux]
      | succ i =>
          have hi_rs : i < rs.length := by
            simpa using Nat.lt_of_succ_lt_succ hi
          have hi'_rs :
              i < (activateRulesAux sel rules_len (start + 1) rs).length := by
            simpa [activateRulesAux] using Nat.lt_of_succ_lt_succ hi'
          have h := ih (start + 1) i hi_rs hi'_rs
          simpa [activateRulesAux, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h

lemma activate_node_from_tokens_get {n : Nat}
    (node : CircuitNode n) (sel idx : Nat)
    (hidx : idx < node.rules.length)
    (hidx' : idx < (activate_node_from_tokens node sel).rules.length) :
    (activate_node_from_tokens node sel).rules.get ⟨idx, hidx'⟩ =
      setActivation (node.rules.get ⟨idx, hidx⟩)
        (selectorActive node.rules.length sel idx) := by
  unfold activate_node_from_tokens at hidx' ⊢
  simpa [Nat.zero_add] using
    activateRulesAux_get sel node.rules.length 0 node.rules idx hidx hidx'

lemma activate_node_from_tokens_selected_active {n : Nat}
    (node : CircuitNode n) (sel idx : Nat)
    (hidx : idx < node.rules.length)
    (hsel : selectorActive node.rules.length sel idx = true) :
    is_rule_active
      ((activate_node_from_tokens node sel).rules.get
        ⟨idx, by
          simpa [activate_node_from_tokens, activateRulesAux_length] using hidx⟩) = true := by
  rw [activate_node_from_tokens_get node sel idx hidx]
  simp [is_rule_active_setActivation, hsel]

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


end Semantic
