import Evaluator.MultiToken.Admissibility

open scoped Classical

/-!
# Multi-token congruence and canonical reduction


The multi evaluator cannot distinguish two `MultiPathInput`s related by a
permutation of co-located routes' subpaths (`ColocPerm` ; equal multisets of
(origin column, subpath) pairs). Evaluation reads a token only through
  its key `tkey t = (source_column, current_column, input_label,
    dep_vector)` ; everything `evaluate_layer_multi` looks at ; and
  its own subpath ; the only use of `origin_column` is the path lookup in
    `propagate_tokens`.
Thus the whole run is a function of the multiset of `(key, subpath)` pairs
(`attachK`): propagation factors through `stepK` on those pairs
(`propagate_map_attachK`, via `propagate_tokens_effStep`), and a
`List.Perm` of them is carried through the level induction. The two
a-priori order-sensitive reads are neutralized:
  `nodeError_multi` / `selectedRuleIndex?` consult the HEAD token, but
    mixed-rule or undecodable arrivals error under EITHER head
    (`groupOK_false_of_mixed` / `groupOK_false_of_undecodable`) and
    single-rule arrivals make the head irrelevant (`groupOK_perm`);
  `gather_rule_inputs`' `find?` takes the FIRST token of a source column,
    but `dep_vector` is a function of `source_column` on every reachable
    token list (`DepFunctional` ; seeds read `initial_vectors[col]`,
    `propagate_tokens` sets `dep := outputs[source]`), so any representative
    is the same (`find?_perm_of_fst`).


Within a route the coloured clause-2 chain is determined: at the
`(level,formula)`-selected node the head-colour edge is a function of the
state (`edgeOfColour`), so two valid chains for the SAME route agree on every
effective read (`chainC_colEff_unique`, the per-route analogue of the
unique-edge argument `chain_colEff_unique`, with `(lvl, residual)` threaded).
Across co-located routes the only freedom is the matching extracted from
`multiMatch?` (`multiMatch?_perm` / `multiMatch?_valid`) ; that matching IS
the permutation of the statement. Canonical slot-validity
(`canonicalSlotOKB d`) is the decidable Path-B side condition (eval-gated
true on the examples). Combining this reduction with the multiset and
read-equivalence congruences, every admissible multi-path evaluates identically to the canonical one
(`admissibleMulti_eval_canonical` / `admissibleMulti_accept_canonical`).

The compressed universal bridge is stated in `MultiTokenBridge`.
-/

open DLDS

namespace DLDS

open FlowSemantics



/--
 The co-location pairs of a multi-path input: the multiset of
    (origin column, subpath) pairs is everything the run can see of it
    (`tokenMultiset_congr`).
-/
def colocPairs (d : Graph) (P : MultiPathInput) : List (Nat × List (Nat × Nat)) :=
  (List.range (routeCount d)).map fun i => (slotColumn d i, P.getD i [])

/--  `P'` is `P` with the subpaths of co-located routes permuted.  -/
def ColocPerm (d : Graph) (P P' : MultiPathInput) : Prop :=
  (colocPairs d P).Perm (colocPairs d P')



/--
 Everything node evaluation reads of a token:
    `(source_column, current_column, input_label, dep_vector)`.
-/
def tkey {n : Nat} (t : Token n) : Nat × Nat × Nat × List.Vector Bool n :=
  (t.source_column, t.current_column, t.input_label, t.dep_vector)

/--
 Everything the WHOLE run reads of a token: its key plus its own subpath
    (`origin_column` is only the path-lookup index).
-/
def attachK {n : Nat} (P : MultiPathInput) (t : Token n) :
    (Nat × Nat × Nat × List.Vector Bool n) × List (Nat × Nat) :=
  (tkey t, P.getD t.origin_column [])

/--
 `dep_vector` is a function of `source_column` ; true of the seeds and of
    every propagation output; the reason first-match input gathering cannot
    distinguish co-located tokens.
-/
def DepFunctional {n : Nat} (T : List (Token n)) : Prop :=
  ∃ f : Nat → List.Vector Bool n, ∀ t ∈ T, t.dep_vector = f t.source_column

lemma initialize_tokens_multi_depFunctional {n : Nat}
    (routes : List (Nat × Nat)) (iv : List (List.Vector Bool n)) (top : Nat) :
    DepFunctional (initialize_tokens_multi routes iv top) := by
  refine ⟨fun c => (iv[c]?).getD (List.Vector.replicate n false), ?_⟩
  intro t ht
  unfold initialize_tokens_multi at ht
  rw [List.mem_map] at ht
  obtain ⟨⟨⟨col, r⟩, idx⟩, _, rfl⟩ := ht
  rfl

lemma effStep_eq_colEff_getD (P : PathInput) (c s : Nat) :
    effStep P c s = colEff (P.getD c []) s := by
  by_cases hc : c < P.length
  · exact effStep_colEff hc s
  · rw [effStep_ge_col hc]
    have hnil : P.getD c [] = [] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (Nat.le_of_not_lt hc)]
      rfl
    rw [hnil]
    rfl

lemma propagate_tokens_depFunctional {n : Nat} (T : List (Token n))
    (P : MultiPathInput) (lvl num : Nat) (outs : List (List.Vector Bool n)) :
    DepFunctional (propagate_tokens T P lvl num outs) := by
  refine ⟨fun c => (outs[c]?).getD (List.Vector.replicate n false), ?_⟩
  intro t ht
  rw [propagate_tokens_effStep, List.mem_filterMap] at ht
  obtain ⟨s, _, hs⟩ := ht
  by_cases hl : lvl > 0
  · rw [if_pos hl] at hs
    cases hes : effStep P s.origin_column (num - lvl - 1) with
    | none => rw [hes] at hs; simp at hs
    | some st =>
        rw [hes] at hs
        dsimp only at hs
        by_cases hout : s.current_column < outs.length
        · rw [dif_pos hout] at hs
          obtain rfl := Option.some.inj hs
          show outs.get ⟨s.current_column, hout⟩ = _
          rw [List.get_eq_getElem]
          simp [List.getElem?_eq_getElem hout]
        · rw [dif_neg hout] at hs; simp at hs
  · rw [if_neg hl] at hs; simp at hs

def stepK {n : Nat} (lvl num_levels : Nat) (outs : List (List.Vector Bool n)) :
    ((Nat × Nat × Nat × List.Vector Bool n) × List (Nat × Nat)) →
      Option ((Nat × Nat × Nat × List.Vector Bool n) × List (Nat × Nat))
  | ((_, cur, _, _), pth) =>
      if lvl > 0 then
        match colEff pth (num_levels - lvl - 1) with
        | none => none
        | some st =>
            if h : cur < outs.length then
              some ((cur, st.1 - 1, st.2, outs.get ⟨cur, h⟩), pth)
            else none
      else none

lemma propagate_map_attachK {n : Nat} (T : List (Token n)) (P : MultiPathInput)
    (lvl num : Nat) (outs : List (List.Vector Bool n)) :
    (propagate_tokens T P lvl num outs).map (attachK P) =
      (T.map (attachK P)).filterMap (stepK lvl num outs) := by
  rw [propagate_tokens_effStep, List.map_filterMap, List.filterMap_map]
  apply List.filterMap_congr
  intro t _
  simp only [Function.comp]
  by_cases hl : lvl > 0
  · rw [if_pos hl, effStep_eq_colEff_getD, List.getD_eq_getElem?_getD]
    show _ = stepK lvl num outs (attachK P t)
    cases hcE : colEff ((P[t.origin_column]?).getD []) (num - lvl - 1) with
    | none => simp [stepK, attachK, tkey, hl, hcE]
    | some st =>
        by_cases hout : t.current_column < outs.length
        · simp [stepK, attachK, tkey, hl, hcE, hout]
        · simp [stepK, attachK, tkey, hl, hcE, hout]
  · rw [if_neg hl]
    show _ = stepK lvl num outs (attachK P t)
    simp [stepK, hl]



lemma tkey_transfer {n : Nat} {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    ∀ s' ∈ T', ∃ s ∈ T, tkey s = tkey s' := by
  intro s' hs'
  have hm : tkey s' ∈ T.map tkey :=
    h.mem_iff.mpr (List.mem_map.mpr ⟨s', hs', rfl⟩)
  rw [List.mem_map] at hm
  obtain ⟨s, hs, heq⟩ := hm
  exact ⟨s, hs, heq⟩

lemma attachK_transfer {n : Nat} {P P' : MultiPathInput}
    {T T' : List (Token n)}
    (h : (T.map (attachK P)).Perm (T'.map (attachK P'))) :
    ∀ s' ∈ T', ∃ s ∈ T, attachK P s = attachK P' s' := by
  intro s' hs'
  have hm : attachK P' s' ∈ T.map (attachK P) :=
    h.mem_iff.mpr (List.mem_map.mpr ⟨s', hs', rfl⟩)
  rw [List.mem_map] at hm
  obtain ⟨s, hs, heq⟩ := hm
  exact ⟨s, hs, heq⟩

lemma tkey_eq_fields {n : Nat} {s t : Token n} (h : tkey s = tkey t) :
    s.source_column = t.source_column ∧ s.current_column = t.current_column ∧
    s.input_label = t.input_label ∧ s.dep_vector = t.dep_vector := by
  simp only [tkey, Prod.mk.injEq] at h
  exact ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

lemma tokenAtExhaustedStopB_congr {n : Nat} (d : Graph)
    {P P' : MultiPathInput} (flags flags' : List (List Bool)) (stepIdx : Nat)
    {s t : Token n} (hk : tkey s = tkey t)
    (hp : P.getD s.origin_column [] = P'.getD t.origin_column []) :
    tokenAtExhaustedStopB d P flags stepIdx s =
      tokenAtExhaustedStopB d P' flags' stepIdx t := by
  have hf := tkey_eq_fields hk
  unfold tokenAtExhaustedStopB
  simp only
  rw [hp, hf.1, hf.2.1]

/--  The head-independent core of `nodeError_multi` for a fixed rule `r`.  -/
def groupOK {n : Nat} (node : CircuitNode n) (ni : NodeIncoming) (r : Nat)
    (tokens : List (Token n)) : Bool :=
  decide (r < node.rules.length) &&
  (tokens.all (fun s =>
    match decodeInputLabel ni s.input_label with
    | some (r', slot', src') =>
        decide (r' = r ∧ slot' < (ni[r]!).length ∧ s.source_column = src')
    | none => false)) &&
  ((List.range (ni[r]!).length).all (fun i =>
    tokens.any (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' = i ∧ s.source_column = src')
      | none => false)))

lemma nodeError_multi_eq_head {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) (t : Token n) (ts : List (Token n)) :
    nodeError_multi node ni (t :: ts) =
      match decodeInputLabel ni t.input_label with
      | none => true
      | some (r, _, _) => ! groupOK node ni r (t :: ts) := rfl

lemma groupOK_false_of_undecodable {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) (r : Nat) {T : List (Token n)} {s : Token n}
    (hs : s ∈ T) (hdec : decodeInputLabel ni s.input_label = none) :
    groupOK node ni r T = false := by
  unfold groupOK
  have hfail : (T.all (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' < (ni[r]!).length ∧ s.source_column = src')
      | none => false)) = false := by
    rw [Bool.eq_false_iff]
    intro hb
    have hx := (List.all_eq_true.mp hb) s hs
    rw [hdec] at hx
    simp at hx
  rw [hfail]
  simp

lemma groupOK_false_of_mixed {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) {r r' : Nat} (hne : r' ≠ r)
    {T : List (Token n)} {s : Token n} (hs : s ∈ T)
    {slot src : Nat}
    (hdec : decodeInputLabel ni s.input_label = some (r', slot, src)) :
    groupOK node ni r T = false := by
  unfold groupOK
  have hfail : (T.all (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' < (ni[r]!).length ∧ s.source_column = src')
      | none => false)) = false := by
    rw [Bool.eq_false_iff]
    intro hb
    have hx := (List.all_eq_true.mp hb) s hs
    rw [hdec] at hx
    exact hne (of_decide_eq_true hx).1
  rw [hfail]
  simp

lemma groupOK_perm {n : Nat} (node : CircuitNode n) (ni : NodeIncoming)
    (r : Nat) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    groupOK node ni r T = groupOK node ni r T' := by
  unfold groupOK
  have hall : (T.all (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' < (ni[r]!).length ∧ s.source_column = src')
      | none => false)) = (T'.all (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' < (ni[r]!).length ∧ s.source_column = src')
      | none => false)) := by
    rw [Bool.eq_iff_iff, List.all_eq_true, List.all_eq_true]
    constructor
    · intro hall s' hs'
      obtain ⟨s, hsT, hkey⟩ := tkey_transfer h s' hs'
      have hf := tkey_eq_fields hkey
      have hx := hall s hsT
      rw [hf.2.2.1, hf.1] at hx
      exact hx
    · intro hall s hsT
      obtain ⟨s', hs', hkey⟩ := tkey_transfer h.symm s hsT
      have hf := tkey_eq_fields hkey
      have hx := hall s' hs'
      rw [hf.2.2.1, hf.1] at hx
      exact hx
  have hany : ∀ i : Nat, (T.any (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' = i ∧ s.source_column = src')
      | none => false)) = (T'.any (fun s =>
      match decodeInputLabel ni s.input_label with
      | some (r', slot', src') =>
          decide (r' = r ∧ slot' = i ∧ s.source_column = src')
      | none => false)) := by
    intro i
    rw [Bool.eq_iff_iff, List.any_eq_true, List.any_eq_true]
    constructor
    · rintro ⟨s, hsT, hx⟩
      obtain ⟨s', hs', hkey⟩ := tkey_transfer h.symm s hsT
      have hf := tkey_eq_fields hkey
      refine ⟨s', hs', ?_⟩
      rw [hf.2.2.1, hf.1]
      exact hx
    · rintro ⟨s', hs', hx⟩
      obtain ⟨s, hsT, hkey⟩ := tkey_transfer h s' hs'
      have hf := tkey_eq_fields hkey
      refine ⟨s, hsT, ?_⟩
      rw [← hf.2.2.1, ← hf.1] at hx
      exact hx
  have hslots : ((List.range (ni[r]!).length).all (fun i =>
      T.any (fun s =>
        match decodeInputLabel ni s.input_label with
        | some (r', slot', src') =>
            decide (r' = r ∧ slot' = i ∧ s.source_column = src')
        | none => false))) = ((List.range (ni[r]!).length).all (fun i =>
      T'.any (fun s =>
        match decodeInputLabel ni s.input_label with
        | some (r', slot', src') =>
            decide (r' = r ∧ slot' = i ∧ s.source_column = src')
        | none => false))) := by
    rw [Bool.eq_iff_iff, List.all_eq_true, List.all_eq_true]
    constructor
    · intro hx i hi; rw [← hany i]; exact hx i hi
    · intro hx i hi; rw [hany i]; exact hx i hi
  rw [hall, hslots]

/--
 **`nodeError_multi` is key-multiset invariant** ; the head choice is
    irrelevant: mixed-rule or undecodable arrivals error under either head,
    single-rule arrivals reduce to the head-independent `groupOK`.
-/
lemma nodeError_multi_perm {n : Nat} (node : CircuitNode n) (ni : NodeIncoming)
    {T T' : List (Token n)} (h : (T.map tkey).Perm (T'.map tkey)) :
    nodeError_multi node ni T = nodeError_multi node ni T' := by
  have hlen : T.length = T'.length := by
    have := h.length_eq; simpa using this
  cases T with
  | nil =>
      cases T' with
      | nil => rfl
      | cons a b => simp at hlen
  | cons t ts =>
      cases T' with
      | nil => simp at hlen
      | cons t' ts' =>
          rw [nodeError_multi_eq_head, nodeError_multi_eq_head]
          cases hdec : decodeInputLabel ni t.input_label with
          | none =>
              cases hdec' : decodeInputLabel ni t'.input_label with
              | none => rfl
              | some rss' =>
                  obtain ⟨r', sl', sr'⟩ := rss'
                  obtain ⟨s', hs', hkey⟩ :=
                    tkey_transfer h.symm t (List.mem_cons_self ..)
                  have hf := tkey_eq_fields hkey
                  show true = ! groupOK node ni r' (t' :: ts')
                  rw [groupOK_false_of_undecodable node ni r' hs'
                    (by rw [hf.2.2.1, hdec])]
                  rfl
          | some rss =>
              obtain ⟨r, sl, sr⟩ := rss
              cases hdec' : decodeInputLabel ni t'.input_label with
              | none =>
                  obtain ⟨s, hsT, hkey⟩ :=
                    tkey_transfer h t' (List.mem_cons_self ..)
                  have hf := tkey_eq_fields hkey
                  show (! groupOK node ni r (t :: ts)) = true
                  rw [groupOK_false_of_undecodable node ni r hsT
                    (by rw [hf.2.2.1, hdec'])]
                  rfl
              | some rss' =>
                  obtain ⟨r', sl', sr'⟩ := rss'
                  by_cases hrr : r' = r
                  · subst hrr
                    show (! groupOK node ni r' (t :: ts)) =
                      (! groupOK node ni r' (t' :: ts'))
                    rw [groupOK_perm node ni r' h]
                  · obtain ⟨s, hsT, hkey⟩ :=
                      tkey_transfer h t' (List.mem_cons_self ..)
                    have hf := tkey_eq_fields hkey
                    show (! groupOK node ni r (t :: ts)) =
                      (! groupOK node ni r' (t' :: ts'))
                    rw [groupOK_false_of_mixed node ni hrr hsT
                      (by rw [hf.2.2.1, hdec'])]
                    obtain ⟨s'', hs'', hkey''⟩ :=
                      tkey_transfer h.symm t (List.mem_cons_self ..)
                    have hf'' := tkey_eq_fields hkey''
                    rw [groupOK_false_of_mixed node ni
                      (fun hh => hrr hh.symm) hs''
                      (by rw [hf''.2.2.1, hdec])]

/--
 Under a coherent (no-error) arrival, the selected rule is the same for
    any key-permuted token list.
-/
lemma selectedRuleIndex?_perm {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey))
    (hne : nodeError_multi node ni T = false) :
    selectedRuleIndex? ni T = selectedRuleIndex? ni T' := by
  have hlen : T.length = T'.length := by
    have := h.length_eq; simpa using this
  cases T with
  | nil =>
      cases T' with
      | nil => rfl
      | cons a b => simp at hlen
  | cons t ts =>
      cases T' with
      | nil => simp at hlen
      | cons t' ts' =>
          cases hdec : decodeInputLabel ni t.input_label with
          | none =>
              rw [nodeError_multi_eq_head, hdec] at hne
              simp at hne
          | some rss =>
              obtain ⟨r, sl, sr⟩ := rss
              rw [nodeError_multi_eq_head, hdec] at hne
              have hgOK : groupOK node ni r (t :: ts) = true := by
                simpa using hne
              unfold groupOK at hgOK
              rw [Bool.and_eq_true, Bool.and_eq_true] at hgOK
              obtain ⟨⟨_, hlab⟩, _⟩ := hgOK
              obtain ⟨s, hsT, hkey⟩ := tkey_transfer h t' (List.mem_cons_self ..)
              have hf := tkey_eq_fields hkey
              have hx := (List.all_eq_true.mp hlab) s hsT
              cases hdecs : decodeInputLabel ni s.input_label with
              | none => rw [hdecs] at hx; simp at hx
              | some rss2 =>
                  obtain ⟨r2, sl2, sr2⟩ := rss2
                  rw [hdecs] at hx
                  have hr2 : r2 = r := (of_decide_eq_true hx).1
                  show (decodeInputLabel ni t.input_label).map (·.1) =
                    (decodeInputLabel ni t'.input_label).map (·.1)
                  rw [hdec, ← hf.2.2.1, hdecs, hr2]
                  rfl

lemma find?_perm_of_fst {n : Nat} {A A' : List (Nat × List.Vector Bool n)}
    (hperm : A.Perm A')
    (hfun : ∀ a ∈ A, ∀ b ∈ A, a.1 = b.1 → a.2 = b.2)
    (q : Nat × List.Vector Bool n → Bool)
    (huniq : ∀ a b, q a = true → q b = true → a.1 = b.1) :
    A.find? q = A'.find? q := by
  cases hA : A.find? q with
  | none =>
      rw [List.find?_eq_none] at hA
      symm
      rw [List.find?_eq_none]
      intro x hx
      exact hA x (hperm.mem_iff.mpr hx)
  | some a =>
      have haA : a ∈ A := List.mem_of_find?_eq_some hA
      have hqa : q a = true := List.find?_some hA
      have hsome : (A'.find? q).isSome := by
        rw [List.find?_isSome]
        exact ⟨a, hperm.mem_iff.mp haA, hqa⟩
      obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp hsome
      have hbA' : b ∈ A' := List.mem_of_find?_eq_some hb
      have hqb : q b = true := List.find?_some hb
      have hbA : b ∈ A := hperm.mem_iff.mpr hbA'
      have hfst : a.1 = b.1 := huniq a b hqa hqb
      have hsnd : a.2 = b.2 := hfun a haA b hbA hfst
      rw [hb]
      exact congrArg some (Prod.ext hfst hsnd)

/--
 `gather_rule_inputs` is invariant under available-input permutation with
    column-functional values.
-/
lemma gather_rule_inputs_perm {n : Nat}
    {A A' : List (Nat × List.Vector Bool n)} (hperm : A.Perm A')
    (hfun : ∀ a ∈ A, ∀ b ∈ A, a.1 = b.1 → a.2 = b.2)
    (inc : RuleIncoming) :
    gather_rule_inputs inc A = gather_rule_inputs inc A' := by
  unfold gather_rule_inputs
  apply List.filterMap_congr
  intro rce _
  obtain ⟨rc, eid⟩ := rce
  have hfind : A.find? (fun p => p.1 = rc) = A'.find? (fun p => p.1 = rc) := by
    apply find?_perm_of_fst hperm hfun
    intro a b ha hb
    have ha' : a.1 = rc := of_decide_eq_true ha
    have hb' : b.1 = rc := of_decide_eq_true hb
    rw [ha', hb']
  show ((A.find? fun p => p.1 = rc).map Prod.snd) =
    ((A'.find? fun p => p.1 = rc).map Prod.snd)
  rw [hfind]

lemma node_logic_with_routing_perm {n : Nat} (rules : List (Rule n))
    (ni : NodeIncoming) {A A' : List (Nat × List.Vector Bool n)}
    (hperm : A.Perm A')
    (hfun : ∀ a ∈ A, ∀ b ∈ A, a.1 = b.1 → a.2 = b.2) :
    node_logic_with_routing rules ni A = node_logic_with_routing rules ni A' := by
  unfold node_logic_with_routing
  have hg : (rules.zipIdx.map fun (_rule, rule_idx) =>
      gather_rule_inputs (ni[rule_idx]!) A) =
    (rules.zipIdx.map fun (_rule, rule_idx) =>
      gather_rule_inputs (ni[rule_idx]!) A') := by
    apply List.map_congr_left
    intro x _
    exact gather_rule_inputs_perm hperm hfun _
  rw [hg]

lemma evaluate_node_multi_perm {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey))
    (hf : ∃ f : Nat → List.Vector Bool n,
      ∀ t ∈ T, t.dep_vector = f t.source_column) :
    evaluate_node_multi node ni T = evaluate_node_multi node ni T' := by
  have hlen : T.length = T'.length := by
    have := h.length_eq; simpa using this
  unfold evaluate_node_multi
  by_cases hemp : T.isEmpty
  · have hemp' : T'.isEmpty = true := by
      cases T' with
      | nil => rfl
      | cons a b =>
          cases T with
          | nil => simp at hlen
          | cons c d => simp at hemp
    rw [if_pos hemp, if_pos hemp']
  · have hemp' : ¬ T'.isEmpty = true := by
      cases T' with
      | nil =>
          cases T with
          | nil => exact absurd rfl hemp
          | cons c d => simp at hlen
      | cons a b => simp
    rw [if_neg hemp, if_neg hemp']
    rw [nodeError_multi_perm node ni h]
    by_cases herr : nodeError_multi node ni T' = true
    · rw [if_pos herr, if_pos herr]
    · rw [if_neg herr, if_neg herr]
      have hnerr : nodeError_multi node ni T = false := by
        rw [nodeError_multi_perm node ni h]
        exact Bool.eq_false_iff.mpr herr
      rw [selectedRuleIndex?_perm node ni h hnerr]
      cases hsel : selectedRuleIndex? ni T' with
      | none => rfl
      | some ruleIdx =>
          have hA : (T.map fun t => (t.source_column, t.dep_vector)).Perm
              (T'.map fun t => (t.source_column, t.dep_vector)) := by
            have hh := h.map (fun k => (k.1, k.2.2.2))
            rw [List.map_map, List.map_map] at hh
            exact hh
          have hAfun : ∀ a ∈ (T.map fun t => (t.source_column, t.dep_vector)),
              ∀ b ∈ (T.map fun t => (t.source_column, t.dep_vector)),
              a.1 = b.1 → a.2 = b.2 := by
            obtain ⟨f, hff⟩ := hf
            intro a ha b hb hab
            rw [List.mem_map] at ha hb
            obtain ⟨sa, hsa, rfl⟩ := ha
            obtain ⟨sb, hsb, rfl⟩ := hb
            show sa.dep_vector = sb.dep_vector
            rw [hff sa hsa, hff sb hsb]
            simp only at hab
            rw [hab]
          exact congrArg (fun res => (res.1, false))
            (node_logic_with_routing_perm
              (activate_node_from_tokens node
                (ruleSelectorForIndex ruleIdx)).rules ni hA hAfun)

lemma filter_rule_map_tkey {n : Nat} (ni : NodeIncoming) (r : Nat)
    (S : List (Token n)) :
    (tokensForRule ni r S).map tkey =
      (S.map tkey).filter (fun k =>
        match decodeInputLabel ni k.2.2.1 with
        | some (r', _slot, _src) => r' == r
        | none => false) := by
  induction S with
  | nil => rfl
  | cons a as ih =>
      unfold tokensForRule
      simp only [List.filter_cons, List.map_cons]
      cases hdec : decodeInputLabel ni a.input_label with
      | none =>
          simp [tkey, hdec, tokensForRule] at ih ⊢
          exact ih
      | some rss =>
          obtain ⟨r', slot, src⟩ := rss
          by_cases hrr : r' = r
          · simp [tkey, hdec, hrr, tokensForRule] at ih ⊢
            exact ih
          · simp [tkey, hdec, hrr, tokensForRule] at ih ⊢
            exact ih

lemma filter_rule_slot_map_tkey {n : Nat} (ni : NodeIncoming) (r slot : Nat)
    (S : List (Token n)) :
    (tokensForRuleSlot ni r slot S).map tkey =
      (S.map tkey).filter (fun k =>
        match decodeInputLabel ni k.2.2.1 with
        | some (r', slot', source) =>
            r' == r && slot' == slot && k.1 == source
        | none => false) := by
  induction S with
  | nil => rfl
  | cons a as ih =>
      unfold tokensForRuleSlot
      simp only [List.filter_cons, List.map_cons]
      cases hdec : decodeInputLabel ni a.input_label with
      | none =>
          simp [tkey, hdec, tokensForRuleSlot] at ih ⊢
          exact ih
      | some rss =>
          obtain ⟨r', slot', source⟩ := rss
          by_cases hmatch :
              (r' == r && slot' == slot && a.source_column == source) = true
          · simp [tkey, hdec, hmatch, tokensForRuleSlot] at ih ⊢
            exact ih
          · simp [tkey, hdec, hmatch, tokensForRuleSlot] at ih ⊢
            exact ih

lemma list_or_perm {n : Nat} {xs ys : List (List.Vector Bool n)}
    (h : xs.Perm ys) : list_or xs = list_or ys := by
  unfold list_or
  let op := fun (acc v : List.Vector Bool n) => acc.zipWith (· || ·) v
  letI : RightCommutative op :=
    ⟨by
      intro base left right
      apply List.Vector.ext
      intro i
      simp [op, Bool.or_assoc, Bool.or_comm, Bool.or_left_comm]⟩
  change List.foldl op (List.Vector.replicate n false) xs =
    List.foldl op (List.Vector.replicate n false) ys
  apply h.foldl_eq

lemma tokensForRule_perm {n : Nat} (ni : NodeIncoming) (r : Nat)
    {T T' : List (Token n)} (h : (T.map tkey).Perm (T'.map tkey)) :
    ((tokensForRule ni r T).map tkey).Perm
      ((tokensForRule ni r T').map tkey) := by
  rw [filter_rule_map_tkey, filter_rule_map_tkey]
  exact h.filter _

lemma tokensForRuleSlot_perm {n : Nat} (ni : NodeIncoming) (r slot : Nat)
    {T T' : List (Token n)} (h : (T.map tkey).Perm (T'.map tkey)) :
    ((tokensForRuleSlot ni r slot T).map tkey).Perm
      ((tokensForRuleSlot ni r slot T').map tkey) := by
  rw [filter_rule_slot_map_tkey, filter_rule_slot_map_tkey]
  exact h.filter _

lemma slot_list_or_eq {n : Nat} (ni : NodeIncoming) (r slot : Nat)
    {T T' : List (Token n)} (h : (T.map tkey).Perm (T'.map tkey)) :
    list_or ((tokensForRuleSlot ni r slot T).map (·.dep_vector)) =
      list_or ((tokensForRuleSlot ni r slot T').map (·.dep_vector)) := by
  apply list_or_perm
  have hp := (tokensForRuleSlot_perm ni r slot h).map fun k => k.2.2.2
  simpa [Function.comp_def, tkey] using hp

lemma ruleResultVariants_perm {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) (r : Nat) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    ruleResultVariants node ni r T = ruleResultVariants node ni r T' := by
  let required := ni[r]?.getD []
  have hgroup : (tokensForRule ni r T).isEmpty =
      (tokensForRule ni r T').isEmpty := by
    have hp := tokensForRule_perm ni r h
    have hlen := hp.length_eq
    cases hT : tokensForRule ni r T <;>
      cases hT' : tokensForRule ni r T' <;> simp_all
  have hslot : ∀ slot,
      (tokensForRuleSlot ni r slot T).isEmpty =
        (tokensForRuleSlot ni r slot T').isEmpty := by
    intro slot
    have hp := tokensForRuleSlot_perm ni r slot h
    have hlen := hp.length_eq
    cases hT : tokensForRuleSlot ni r slot T <;>
      cases hT' : tokensForRuleSlot ni r slot T' <;> simp_all
  have hany : ∀ slots : List Nat,
      slots.any (fun slot => (tokensForRuleSlot ni r slot T).isEmpty) =
        slots.any (fun slot => (tokensForRuleSlot ni r slot T').isEmpty) := by
    intro slots
    induction slots with
    | nil => rfl
    | cons slot rest ih => simp only [List.any_cons, hslot slot, ih]
  have havailable :
      required.zipIdx.map (fun x =>
        let source := x.1.1
        let slot := x.2
        (source, list_or ((tokensForRuleSlot ni r slot T).map
          (·.dep_vector)))) =
      required.zipIdx.map (fun x =>
        let source := x.1.1
        let slot := x.2
        (source, list_or ((tokensForRuleSlot ni r slot T').map
          (·.dep_vector)))) := by
    apply List.map_congr_left
    intro x _
    obtain ⟨⟨source, edge⟩, slot⟩ := x
    simp only
    rw [slot_list_or_eq ni r slot h]
  unfold ruleResultVariants
  change (if (tokensForRule ni r T).isEmpty ||
        (List.range required.length).any
          (fun slot => (tokensForRuleSlot ni r slot T).isEmpty) then []
      else _) = _
  rw [hgroup, hany (List.range required.length), havailable]

lemma filter_colb_map_tkey {n : Nat} (col : Nat) (S : List (Token n)) :
    (S.filter (fun t => t.current_column == col)).map tkey =
      (S.map tkey).filter (fun k => k.2.1 == col) := by
  induction S with
  | nil => rfl
  | cons a as ih =>
      by_cases hc : a.current_column = col
      · simp [tkey, hc, ih]
      · simp [tkey, hc, ih]

lemma compositionPlanCompatibleB_perm (d : Graph) {P P' : MultiPathInput}
    (flags flags' : List (List Bool)) (stepIdx : Nat)
    (layer : GridLayer (buildFormulas d).length)
    {T T' : List (Token (buildFormulas d).length)}
    (h : (T.map (attachK P)).Perm (T'.map (attachK P')))
    (plan : ContinuationPlan) :
    compositionPlanCompatibleB d P flags stepIdx layer T plan =
      compositionPlanCompatibleB d P' flags' stepIdx layer T' plan := by
  have hkey : (T.map tkey).Perm (T'.map tkey) := by
    have hm := h.map Prod.fst
    simpa [List.map_map, Function.comp_def, attachK] using hm
  unfold compositionPlanCompatibleB
  dsimp only
  let col := (buildFormulas d).idxOf plan.vertex.FORMULA
  cases hn : layer.nodes[col]? with
  | none => simp []
  | some cnode =>
      cases hi : layer.incoming[col]? with
      | none => simp []
      | some incoming =>
          simp only []
          let TH := T.filter fun token => token.current_column == col
          let TH' := T'.filter fun token => token.current_column == col
          have hhere : (TH.map tkey).Perm (TH'.map tkey) := by
            dsimp only [TH, TH']
            rw [filter_colb_map_tkey, filter_colb_map_tkey]
            exact hkey.filter _
          rw [Bool.eq_iff_iff, List.any_eq_true, List.any_eq_true]
          constructor
          · rintro ⟨s, hs, hpred⟩
            have hsT : s ∈ T := List.mem_of_mem_filter hs
            obtain ⟨t, htT, hatt⟩ := attachK_transfer h.symm s hsT
            have hparts : tkey t = tkey s ∧
                P'.getD t.origin_column [] = P.getD s.origin_column [] := by
              simpa [attachK] using hatt
            have hf := tkey_eq_fields hparts.1
            have ht : t ∈ TH' := by
              apply List.mem_filter.mpr
              refine ⟨htT, ?_⟩
              simpa [TH, TH', col, hf.2.1] using (List.mem_filter.mp hs).2
            refine ⟨t, ht, ?_⟩
            have harr := tokenAtExhaustedStopB_congr d flags' flags stepIdx
              hparts.1 hparts.2
            have hpredEq :
                (tokenAtExhaustedStopB d P flags stepIdx s &&
                  match decodeInputLabel incoming s.input_label with
                  | none => false
                  | some decoded =>
                      (ruleResultVariants cnode incoming decoded.1 TH).any
                        fun result => depVectorSubsetB
                          (depVector (buildFormulas d) plan.route.1) result) =
                (tokenAtExhaustedStopB d P' flags' stepIdx t &&
                  match decodeInputLabel incoming t.input_label with
                  | none => false
                  | some decoded =>
                      (ruleResultVariants cnode incoming decoded.1 TH').any
                        fun result => depVectorSubsetB
                          (depVector (buildFormulas d) plan.route.1) result) := by
              rw [← harr, hf.2.2.1]
              cases hd : decodeInputLabel incoming s.input_label with
              | none => simp []
              | some decoded =>
                  simp only []
                  rw [ruleResultVariants_perm cnode incoming decoded.1 hhere]
            change (tokenAtExhaustedStopB d P' flags' stepIdx t &&
              match decodeInputLabel incoming t.input_label with
              | none => false
              | some decoded =>
                  (ruleResultVariants cnode incoming decoded.1 TH').any
                    fun result => depVectorSubsetB
                      (depVector (buildFormulas d) plan.route.1) result) = true
            rw [← hpredEq]
            exact hpred
          · rintro ⟨t, ht, hpred⟩
            have htT : t ∈ T' := List.mem_of_mem_filter ht
            obtain ⟨s, hsT, hatt⟩ := attachK_transfer h t htT
            have hparts : tkey s = tkey t ∧
                P.getD s.origin_column [] = P'.getD t.origin_column [] := by
              simpa [attachK] using hatt
            have hf := tkey_eq_fields hparts.1
            have hs : s ∈ TH := by
              apply List.mem_filter.mpr
              refine ⟨hsT, ?_⟩
              simpa [TH, TH', col, hf.2.1] using (List.mem_filter.mp ht).2
            refine ⟨s, hs, ?_⟩
            have harr := tokenAtExhaustedStopB_congr d flags flags' stepIdx
              hparts.1 hparts.2
            have hpredEq :
                (tokenAtExhaustedStopB d P flags stepIdx s &&
                  match decodeInputLabel incoming s.input_label with
                  | none => false
                  | some decoded =>
                      (ruleResultVariants cnode incoming decoded.1 TH).any
                        fun result => depVectorSubsetB
                          (depVector (buildFormulas d) plan.route.1) result) =
                (tokenAtExhaustedStopB d P' flags' stepIdx t &&
                  match decodeInputLabel incoming t.input_label with
                  | none => false
                  | some decoded =>
                      (ruleResultVariants cnode incoming decoded.1 TH').any
                        fun result => depVectorSubsetB
                          (depVector (buildFormulas d) plan.route.1) result) := by
              rw [harr, hf.2.2.1]
              cases hd : decodeInputLabel incoming t.input_label with
              | none => simp []
              | some decoded =>
                  simp only []
                  rw [ruleResultVariants_perm cnode incoming decoded.1 hhere]
            change (tokenAtExhaustedStopB d P flags stepIdx s &&
              match decodeInputLabel incoming s.input_label with
              | none => false
              | some decoded =>
                  (ruleResultVariants cnode incoming decoded.1 TH).any
                    fun result => depVectorSubsetB
                      (depVector (buildFormulas d) plan.route.1) result) = true
            rw [hpredEq]
            exact hpred

lemma advancedCompositionSeedsAtLevel_perm (d : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    {P P' : MultiPathInput} (flags flags' : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    {T T' : List (Token (buildFormulas d).length)}
    (h : (T.map (attachK P)).Perm (T'.map (attachK P'))) :
    advancedCompositionSeedsAtLevel d enabledNodes plans P flags level numLevels
        layer T =
      advancedCompositionSeedsAtLevel d enabledNodes plans P' flags' level
        numLevels layer T' := by
  unfold advancedCompositionSeedsAtLevel
  apply List.filterMap_congr
  intro item hitem
  obtain ⟨plan, planIdx⟩ := item
  dsimp only
  rw [compositionPlanCompatibleB_perm d flags flags'
    (numLevels - level - 1) layer h plan]

lemma mem_advancedCompositionSeeds_origin (d : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (P : MultiPathInput) (flags : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    (T : List (Token (buildFormulas d).length))
    {seed : Token (buildFormulas d).length}
    (hseed : seed ∈ advancedCompositionSeedsAtLevel d enabledNodes plans P flags
      level numLevels layer T) :
    ∃ planIdx, seed.origin_column = routeCount d + planIdx := by
  unfold advancedCompositionSeedsAtLevel at hseed
  rw [List.mem_filterMap] at hseed
  obtain ⟨⟨plan, planIdx⟩, hmem, heq⟩ := hseed
  dsimp only at heq
  split at heq <;> simp_all
  obtain ⟨hstep, hnonzero, hrecord⟩ := heq
  subst seed
  exact ⟨planIdx, rfl⟩

lemma evaluationPaths_getD_plan (d : Graph) (plans : List ContinuationPlan)
    (P : MultiPathInput) (hP : P.length = routeCount d) (planIdx : Nat) :
    (evaluationPathsWithComposition d plans P).getD
        (routeCount d + planIdx) [] =
      (compositionPathsFromPlans d plans).getD planIdx [] := by
  unfold evaluationPathsWithComposition
  rw [List.getD_eq_getElem?_getD]
  rw [List.getElem?_append_right]
  · simp [hP]
  · omega

lemma advancedCompositionSeeds_map_attachK_eq (d : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    {S S' : MultiPathInput} (hS : S.length = routeCount d)
    (hS' : S'.length = routeCount d) (flags flags' : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    {T T' : List (Token (buildFormulas d).length)}
    (h : (T.map (attachK (evaluationPathsWithComposition d plans S))).Perm
      (T'.map (attachK (evaluationPathsWithComposition d plans S')))) :
    (advancedCompositionSeedsAtLevel d enabledNodes plans
        (evaluationPathsWithComposition d plans S) flags level
        numLevels layer T).map
          (attachK (evaluationPathsWithComposition d plans S)) =
      (advancedCompositionSeedsAtLevel d enabledNodes plans
        (evaluationPathsWithComposition d plans S') flags' level
        numLevels layer T').map
          (attachK (evaluationPathsWithComposition d plans S')) := by
  have hseeds := advancedCompositionSeedsAtLevel_perm d enabledNodes plans
    flags flags' level numLevels layer h
  rw [← hseeds]
  apply List.map_congr_left
  intro seed hs
  unfold attachK
  congr 1
  obtain ⟨planIdx, horigin⟩ := mem_advancedCompositionSeeds_origin d
    enabledNodes plans (evaluationPathsWithComposition d plans S) flags level
      numLevels layer T hs
  rw [horigin, evaluationPaths_getD_plan d plans S hS planIdx,
    evaluationPaths_getD_plan d plans S' hS' planIdx]

lemma propagateTokensComposedAt_perm (d : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    {S S' : MultiPathInput} (hS : S.length = routeCount d)
    (hS' : S'.length = routeCount d) (flags flags' : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas d).length)
    {T T' : List (Token (buildFormulas d).length)}
    (outputs : List (List.Vector Bool (buildFormulas d).length))
    (h : (T.map (attachK (evaluationPathsWithComposition d plans S))).Perm
      (T'.map (attachK (evaluationPathsWithComposition d plans S')))) :
    ((propagateTokensComposedAt d enabledNodes plans T
        (evaluationPathsWithComposition d plans S) flags level numLevels layer
        outputs).map
      (attachK (evaluationPathsWithComposition d plans S))).Perm
    ((propagateTokensComposedAt d enabledNodes plans T'
        (evaluationPathsWithComposition d plans S') flags' level numLevels
        layer outputs).map
      (attachK (evaluationPathsWithComposition d plans S'))) := by
  unfold propagateTokensComposedAt
  simp only [List.map_append]
  apply List.Perm.append
  · rw [propagate_map_attachK, propagate_map_attachK]
    exact h.filterMap _
  · rw [advancedCompositionSeeds_map_attachK_eq d enabledNodes plans hS hS'
      flags flags' level numLevels layer h]

lemma tokenRulesValidB_perm {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    tokenRulesValidB node ni T = tokenRulesValidB node ni T' := by
  unfold tokenRulesValidB
  rw [Bool.eq_iff_iff, List.all_eq_true, List.all_eq_true]
  constructor
  · intro hall t' ht'
    obtain ⟨t, ht, hkey⟩ := tkey_transfer h t' ht'
    have hf := tkey_eq_fields hkey
    have hx := hall t ht
    rw [hf.2.2.1] at hx
    exact hx
  · intro hall t ht
    obtain ⟨t', ht', hkey⟩ := tkey_transfer h.symm t ht
    have hf := tkey_eq_fields hkey
    have hx := hall t' ht'
    rw [hf.2.2.1] at hx
    exact hx

lemma ruleOccurrenceErrorB_perm {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) (r : Nat) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    ruleOccurrenceErrorB node ni r T = ruleOccurrenceErrorB node ni r T' := by
  simp only [ruleOccurrenceErrorB]
  have hp := tokensForRule_perm ni r h
  have hlen := hp.length_eq
  have hempty : (tokensForRule ni r T).isEmpty =
      (tokensForRule ni r T').isEmpty := by
    cases hT : tokensForRule ni r T <;>
      cases hT' : tokensForRule ni r T' <;> simp_all
  rw [hempty, ruleResultVariants_perm node ni r h]

lemma evaluate_node_occurrences_multi_perm {n : Nat} (node : CircuitNode n)
    (ni : NodeIncoming) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    evaluate_node_occurrences_multi node ni T =
      evaluate_node_occurrences_multi node ni T' := by
  have hlen : T.length = T'.length := by
    have := h.length_eq
    simpa using this
  have hempty : T.isEmpty = T'.isEmpty := by
    cases T <;> cases T' <;> simp_all
  unfold evaluate_node_occurrences_multi
  rw [hempty]
  by_cases hT' : T'.isEmpty
  · rw [if_pos hT', if_pos hT']
  · rw [if_neg hT', if_neg hT']
    have hvariants :
        (List.range node.rules.length).flatMap
            (fun r => ruleResultVariants node ni r T) =
          (List.range node.rules.length).flatMap
            (fun r => ruleResultVariants node ni r T') := by
      induction List.range node.rules.length with
      | nil => rfl
      | cons r rest ih =>
          simp only [List.flatMap_cons]
          rw [ruleResultVariants_perm node ni r h, ih]
    have herrors :
        (List.range node.rules.length).any
            (fun r => ruleOccurrenceErrorB node ni r T) =
          (List.range node.rules.length).any
            (fun r => ruleOccurrenceErrorB node ni r T') := by
      induction List.range node.rules.length with
      | nil => rfl
      | cons r rest ih =>
          simp only [List.any_cons, ruleOccurrenceErrorB_perm node ni r h, ih]
    simp only
    rw [hvariants, tokenRulesValidB_perm node ni h, herrors]

lemma filter_col_map_tkey {n : Nat} (col : Nat) (S : List (Token n)) :
    (S.filter (fun t => t.current_column = col)).map tkey =
      (S.map tkey).filter (fun k => k.2.1 = col) := by
  induction S with
  | nil => rfl
  | cons a as ih =>
      by_cases hc : a.current_column = col
      · simp [tkey, hc, ih]
      · simp [tkey, hc, ih]

/--
 **Layer evaluation is key-multiset invariant.** Includes the per-column
    expected-count check (lengths are permutation-invariant).
-/
lemma evaluate_layer_multi_perm {n : Nat} (layer : GridLayer n)
    (expected : List Nat) {T T' : List (Token n)}
    (h : (T.map tkey).Perm (T'.map tkey)) :
    evaluate_layer_multi layer T expected = evaluate_layer_multi layer T' expected := by
  unfold evaluate_layer_multi
  have hres : (layer.nodes.zipIdx.map fun (cnode, col_idx) =>
      evaluate_cell_multi T expected cnode (layer.incoming[col_idx]!) col_idx) =
    (layer.nodes.zipIdx.map fun (cnode, col_idx) =>
      evaluate_cell_multi T' expected cnode (layer.incoming[col_idx]!) col_idx) := by
    apply List.map_congr_left
    intro x _
    obtain ⟨cnode, col⟩ := x
    show evaluate_cell_multi T expected cnode (layer.incoming[col]!) col =
      evaluate_cell_multi T' expected cnode (layer.incoming[col]!) col
    have hfp : ((T.filter (fun t => t.current_column = col)).map tkey).Perm
        ((T'.filter (fun t => t.current_column = col)).map tkey) := by
      rw [filter_col_map_tkey, filter_col_map_tkey]
      exact h.filter _
    have hflen : (T.filter (fun t => t.current_column = col)).length =
        (T'.filter (fun t => t.current_column = col)).length := by
      have := hfp.length_eq
      simpa using this
    have hnode := evaluate_node_occurrences_multi_perm cnode
      (layer.incoming[col]!) hfp
    unfold evaluate_cell_multi
    simp only [hnode, hflen]
  simp only [hres]

lemma evalFromLevelComposedAt_perm (d : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    {S S' : MultiPathInput} (hS : S.length = routeCount d)
    (hS' : S'.length = routeCount d) (flags flags' : List (List Bool))
    (numLevels : Nat) :
    ∀ (rem : List (GridLayer (buildFormulas d).length × List Nat))
      (level : Nat) (T T' : List (Token (buildFormulas d).length))
      (err : Bool),
      (T.map (attachK (evaluationPathsWithComposition d plans S))).Perm
        (T'.map (attachK (evaluationPathsWithComposition d plans S'))) →
      evalFromLevelComposedAt d enabledNodes plans
          (evaluationPathsWithComposition d plans S) flags numLevels level T rem
          err =
        evalFromLevelComposedAt d enabledNodes plans
          (evaluationPathsWithComposition d plans S') flags' numLevels level T'
          rem err := by
  intro rem
  induction rem with
  | nil => intro level T T' err h; rfl
  | cons le rest ih =>
      obtain ⟨layer, expected⟩ := le
      intro level T T' err h
      have hkey : (T.map tkey).Perm (T'.map tkey) := by
        have hm := h.map Prod.fst
        simpa [List.map_map, Function.comp_def, attachK] using hm
      have hlayer := evaluate_layer_multi_perm layer expected hkey
      cases rest with
      | nil =>
          simp only [evalFromLevelComposedAt]
          rw [hlayer]
      | cons next tail =>
          simp only [evalFromLevelComposedAt]
          rw [← hlayer]
          cases hel : evaluate_layer_multi layer T expected with
          | mk outputs layerError =>
              exact ih (level - 1) _ _ _
                (propagateTokensComposedAt_perm d enabledNodes plans hS hS'
                  flags flags' level numLevels layer outputs h)



/--
 The layered multi evaluation is invariant under any transformation
    preserving the multiset of `(key, subpath)` pairs.
-/
lemma eval_from_level_multi_perm {n : Nat} {P P' : MultiPathInput}
    (num_levels : Nat) :
    ∀ (rem : List (GridLayer n × List Nat)) (level : Nat)
      (T T' : List (Token n)) (err : Bool),
      ((T.map (attachK P)).Perm (T'.map (attachK P'))) →
      eval_from_level_multi P level T rem err num_levels =
        eval_from_level_multi P' level T' rem err num_levels := by
  intro rem
  induction rem with
  | nil => intro level T T' err _; rfl
  | cons le rest ih =>
      obtain ⟨layer, expected⟩ := le
      intro level T T' err hperm
      have hkey : (T.map tkey).Perm (T'.map tkey) := by
        have hh := hperm.map Prod.fst
        rw [List.map_map, List.map_map] at hh
        exact hh
      have hlayer := evaluate_layer_multi_perm layer expected hkey
      rw [eval_from_level_multi]
      conv_rhs => rw [eval_from_level_multi]
      rw [← hlayer]
      cases hel : evaluate_layer_multi layer T expected with
      | mk outs lerr =>
          cases rest with
          | nil => rfl
          | cons l2 r2 =>
              have hpp : ((propagate_tokens T P level num_levels outs).map
                    (attachK P)).Perm
                  ((propagate_tokens T' P' level num_levels outs).map
                    (attachK P')) := by
                rw [propagate_map_attachK, propagate_map_attachK]
                exact hperm.filterMap _
              exact ih (level - 1) _ _ _ hpp

lemma init_map_attachK (d : Graph) (P : MultiPathInput) (L : Nat) :
    ((initialize_tokens_multi (routesOf d) (initialVectorsFromDLDS d) L).map
        (attachK P)) =
      (colocPairs d P).map (fun cp =>
        ((cp.1, cp.1, 0,
          ((initialVectorsFromDLDS d)[cp.1]?).getD
            (List.Vector.replicate (buildFormulas d).length false)), cp.2)) := by
  unfold initialize_tokens_multi colocPairs routeCount
  rw [List.map_map, List.map_map]
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp only [List.length_map, List.length_zipIdx] at h1
    have hgd : (routesOf d).getD i (0, 0) = (routesOf d)[i] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1]
      rfl
    cases hri : (routesOf d)[i] with
    | mk col r =>
        simp only [List.getElem_map, List.getElem_zipIdx, List.getElem_range,
          Function.comp, attachK, tkey, slotColumn, hgd, hri, Nat.zero_add]

lemma evaluationPaths_getD_source (d : Graph) (plans : List ContinuationPlan)
    (S : MultiPathInput) (i : Nat) (hi : i < S.length) :
    (evaluationPathsWithComposition d plans S).getD i [] = S.getD i [] := by
  unfold evaluationPathsWithComposition
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_append_left hi]

lemma colocPairs_evaluationPaths (d : Graph) (plans : List ContinuationPlan)
    (S : MultiPathInput) (hS : S.length = routeCount d) :
    colocPairs d (evaluationPathsWithComposition d plans S) = colocPairs d S := by
  unfold colocPairs
  apply List.map_congr_left
  intro i hi
  simp only [List.mem_range] at hi
  rw [evaluationPaths_getD_source d plans S i (by simpa [hS] using hi)]

lemma init_map_attachK_evaluationPaths (d : Graph)
    (plans : List ContinuationPlan) (S : MultiPathInput)
    (hS : S.length = routeCount d) (L : Nat) :
    ((initialize_tokens_multi (routesOf d) (initialVectorsFromDLDS d) L).map
        (attachK (evaluationPathsWithComposition d plans S))) =
      (colocPairs d S).map (fun cp =>
        ((cp.1, cp.1, 0,
          ((initialVectorsFromDLDS d)[cp.1]?).getD
            (List.Vector.replicate (buildFormulas d).length false)), cp.2)) := by
  rw [init_map_attachK, colocPairs_evaluationPaths d plans S hS]

lemma getEvalResultMultiComposedAt_perm (d : Graph)
    (enabledNodes : List Nat) {S S' : MultiPathInput}
    (hS : S.length = routeCount d) (hS' : S'.length = routeCount d)
    (hperm : ColocPerm d S S') :
    getEvalResultMultiComposedAt d enabledNodes S =
      getEvalResultMultiComposedAt d enabledNodes S' := by
  unfold getEvalResultMultiComposedAt
  let plans := continuationPlansReachable d enabledNodes (multiPathsFromFlow d)
  let layers := buildGridFromDLDS d
  apply evalFromLevelComposedAt_perm d enabledNodes plans hS hS'
  rw [init_map_attachK_evaluationPaths d plans S hS,
    init_map_attachK_evaluationPaths d plans S' hS']
  exact hperm.map _

lemma normalizeMultiPaths_length (d : Graph) (P : MultiPathInput) :
    (normalizeMultiPaths d P).length = routeCount d := by
  simp [normalizeMultiPaths]

lemma normalizeMultiPaths_getD (d : Graph) (P : MultiPathInput)
    (i : Nat) (hi : i < routeCount d) :
    (normalizeMultiPaths d P).getD i [] =
      normalizeReadPath d (P.getD i []) := by
  unfold normalizeMultiPaths
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by simpa using hi)]
  simp

lemma colocPairs_normalizeMultiPaths (d : Graph) (P : MultiPathInput) :
    colocPairs d (normalizeMultiPaths d P) =
      (colocPairs d P).map fun cp => (cp.1, normalizeReadPath d cp.2) := by
  unfold colocPairs
  rw [List.map_map]
  apply List.map_congr_left
  intro i hi
  simp only [List.mem_range] at hi
  rw [normalizeMultiPaths_getD d P i hi]
  rfl

lemma ColocPerm_normalizeMultiPaths (d : Graph) {P P' : MultiPathInput}
    (hperm : ColocPerm d P P') :
    ColocPerm d (normalizeMultiPaths d P) (normalizeMultiPaths d P') := by
  unfold ColocPerm at hperm ⊢
  rw [colocPairs_normalizeMultiPaths, colocPairs_normalizeMultiPaths]
  exact hperm.map _

lemma normalizeMultiPaths_eq_of_ReadEquiv (d : Graph) {P Q : MultiPathInput}
    (hRE : ReadEquiv P Q) : normalizeMultiPaths d P = normalizeMultiPaths d Q := by
  unfold normalizeMultiPaths
  apply List.map_congr_left
  intro routeIdx hroute
  unfold normalizeReadPath
  apply List.map_congr_left
  intro stepIdx hstep
  have heq := hRE routeIdx stepIdx
  rw [effStep_eq_colEff_getD, effStep_eq_colEff_getD] at heq
  rw [heq]

/--
 **M2b ; the token-multiset congruence.** Two multi-path inputs related by
    a permutation of co-located routes' subpaths (equal multisets of (origin
    column, subpath) pairs) evaluate BIT-IDENTICALLY.
-/
theorem tokenMultiset_congr (d : Graph) {P P' : MultiPathInput}
    (hperm : ColocPerm d P P') :
    getEvalResultMultiDLDS d P = getEvalResultMultiDLDS d P' := by
  unfold getEvalResultMultiDLDS
  apply getEvalResultMultiComposedAt_perm
  · exact normalizeMultiPaths_length d P
  · exact normalizeMultiPaths_length d P'
  · exact ColocPerm_normalizeMultiPaths d hperm

/--  Acceptance (error flag + discharge) is co-location-permutation invariant.  -/
theorem evaluateDLDS_multi_coloc_congr (d : Graph) {P P' : MultiPathInput}
    (hperm : ColocPerm d P P') (g : Nat) :
    evaluateDLDS_multi d P g = evaluateDLDS_multi d P' g := by
  unfold evaluateDLDS_multi
  rw [tokenMultiset_congr d hperm]



lemma eval_from_level_multi_readequiv {n : Nat} {P Q : MultiPathInput}
    (hRE : ReadEquiv P Q) (num_levels : Nat) :
    ∀ (rem : List (GridLayer n × List Nat)) (level : Nat)
      (T : List (Token n)) (err : Bool),
      eval_from_level_multi P level T rem err num_levels =
        eval_from_level_multi Q level T rem err num_levels := by
  intro rem
  induction rem with
  | nil => intro level T err; rfl
  | cons le rest ih =>
      obtain ⟨layer, expected⟩ := le
      intro level T err
      rw [eval_from_level_multi]
      conv_rhs => rw [eval_from_level_multi]
      cases hel : evaluate_layer_multi layer T expected with
      | mk outs lerr =>
          cases rest with
          | nil => rfl
          | cons l2 r2 =>
              simp only [propagate_tokens_congr hRE]
              exact ih (level - 1) _ _

theorem getEvalResultMultiDLDS_congr (d : Graph) {P Q : MultiPathInput}
    (hRE : ReadEquiv P Q) :
    getEvalResultMultiDLDS d P = getEvalResultMultiDLDS d Q := by
  unfold getEvalResultMultiDLDS
  rw [normalizeMultiPaths_eq_of_ReadEquiv d hRE]


lemma admChainCursor_eq_initial_of_not_allowed (d : Graph)
    (formulas : List Formula) (lvl : Nat) (φ : Formula)
    (cursor : ResidualCursor)
    (hstop : exhaustedStopAllowedB d lvl φ cursor = false) :
    ∀ chain,
      admChainCursorCB d formulas lvl φ cursor chain =
        admChainCB d formulas lvl φ cursor.residual chain := by
  intro chain
  cases cursor with
  | mk residual arrived =>
      cases chain with
      | nil =>
          simp only [admChainCB, admChainCursorCB, hstop, exhaustedStopAllowedB_initial]
      | cons step rest =>
          obtain ⟨target, label⟩ := step
          simp only [admChainCB, admChainCursorCB, hstop, exhaustedStopAllowedB_initial]
          simp [DLDS.ResidualCursor.initial, DLDS.ResidualCursor.afterEdge]

lemma admChainCursor_all_stops_of_allowed (d : Graph)
    (formulas : List Formula) (lvl : Nat) (φ : Formula)
    (cursor : ResidualCursor)
    (hstop : exhaustedStopAllowedB d lvl φ cursor = true) :
    ∀ chain, admChainCursorCB d formulas lvl φ cursor chain = true →
      ∀ x ∈ chain, x = (0, 0) := by
  intro chain hadm x hx
  cases cursor with
  | mk residual arrived =>
      cases residual with
      | cons colour tail =>
          simp [exhaustedStopAllowedB] at hstop
      | nil =>
          cases chain with
          | nil => cases hx
          | cons step rest =>
              obtain ⟨target, label⟩ := step
              let stopped := target == 0 && label == 0 &&
                rest.all (fun s => s == (0, 0))
              by_cases hs : stopped = true
              · have hparts : (target = 0 ∧ label = 0) ∧
                    ∀ x ∈ rest, x = (0, 0) := by
                  simpa [stopped, Bool.and_eq_true, List.all_eq_true] using hs
                cases hx with
                | head =>
                    exact Prod.ext hparts.1.1 hparts.1.2
                | tail _ hxr => exact hparts.2 x hxr
              · simp [admChainCursorCB, hstop, hs, stopped] at hadm



lemma terminal_chainC_all_stops (d : Graph) (formulas : List Formula)
    (lvl : Nat) (φ : Formula) (p : ColourPath)
    (hterm : admChainCB d formulas lvl φ p [] = true) :
    ∀ chain, admChainCB d formulas lvl φ p chain = true →
      ∀ x ∈ chain, x = (0, 0) := by
  intro chain
  cases chain with
  | nil => intro _ x hx; cases hx
  | cons st rest =>
      obtain ⟨t, l⟩ := st
      intro hch x hx
      cases hfind : nodeAtLevelFormula? d lvl φ with
      | none =>
          simp [admChainCB, admChainCursorCB, hfind] at hch
          obtain ⟨⟨ht, hl⟩, hrest⟩ := hch
          cases hx with
          | head => exact Prod.ext (by simpa using ht) (by simpa using hl)
          | tail _ hxr =>
              obtain ⟨a, b⟩ := x
              exact Prod.ext (hrest a b hxr).1 (hrest a b hxr).2
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              simp [admChainCB, admChainCursorCB, hfind, hout] at hch
              obtain ⟨⟨ht, hl⟩, hrest⟩ := hch
              cases hx with
              | head => exact Prod.ext (by simpa using ht) (by simpa using hl)
              | tail _ hxr =>
                  obtain ⟨a, b⟩ := x
                  exact Prod.ext (hrest a b hxr).1 (hrest a b hxr).2
          | cons e es =>
              exfalso
              simp [admChainCB, admChainCursorCB, DLDS.ResidualCursor.initial,
                exhaustedStopAllowedB, hfind, hout] at hterm

lemma chainC_colEff_unique (d : Graph) (formulas : List Formula) :
    ∀ (chain1 : List (Nat × Nat)) (lvl : Nat) (φ : Formula) (p : ColourPath)
      (chain2 : List (Nat × Nat)),
      admChainCB d formulas lvl φ p chain1 = true →
      admChainCB d formulas lvl φ p chain2 = true →
      ∀ s, colEff chain1 s = colEff chain2 s := by
  intro chain1
  induction chain1 with
  | nil =>
      intro lvl φ p chain2 h1 h2 s
      rw [colEff_all_stops (fun _ hx => by cases hx)]
      exact (colEff_all_stops
        (terminal_chainC_all_stops d formulas lvl φ p h1 chain2 h2) s).symm
  | cons st1 rest1 ih =>
      intro lvl φ p chain2 h1 h2 s
      obtain ⟨t1, l1⟩ := st1
      cases hfind : nodeAtLevelFormula? d lvl φ with
      | none =>
          have hterm : admChainCB d formulas lvl φ p [] = true := by
            simp [admChainCB, admChainCursorCB, hfind]
          rw [colEff_all_stops
              (terminal_chainC_all_stops d formulas lvl φ p hterm _ h1),
            colEff_all_stops
              (terminal_chainC_all_stops d formulas lvl φ p hterm _ h2)]
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              have hterm : admChainCB d formulas lvl φ p [] = true := by
                simp [admChainCB, admChainCursorCB, hfind, hout]
              rw [colEff_all_stops
                  (terminal_chainC_all_stops d formulas lvl φ p hterm _ h1),
                colEff_all_stops
                  (terminal_chainC_all_stops d formulas lvl φ p hterm _ h2)]
          | cons e0 es =>
              cases hedge : edgeOfColour d v (headColour p) with
              | none =>
                  exfalso
                  simp [admChainCB, admChainCursorCB, DLDS.ResidualCursor.initial,
                    exhaustedStopAllowedB, hfind, hout, hedge] at h1
              | some e =>
                  cases chain2 with
                  | nil =>
                      exfalso
                      simp [admChainCB, admChainCursorCB, DLDS.ResidualCursor.initial,
                        exhaustedStopAllowedB, hfind, hout] at h2
                  | cons st2 rest2 =>
                      obtain ⟨t2, l2⟩ := st2
                      simp [admChainCB, admChainCursorCB, DLDS.ResidualCursor.initial,
                        exhaustedStopAllowedB, hfind, hout, hedge] at h1 h2
                      obtain ⟨⟨ht1, hl1⟩, htail1⟩ := h1
                      obtain ⟨⟨ht2, hl2⟩, htail2⟩ := h2
                      have hst1 : (t1, l1) =
                          (formulas.idxOf e.END.FORMULA + 1,
                           inputLabelForArrivalC d formulas e.START e.END) :=
                        Prod.ext (by simpa using ht1) (by simpa using hl1)
                      have hst2 : (t2, l2) =
                          (formulas.idxOf e.END.FORMULA + 1,
                           inputLabelForArrivalC d formulas e.START e.END) :=
                        Prod.ext (by simpa using ht2) (by simpa using hl2)
                      cases s with
                      | zero =>
                          simp only [colEff, List.getElem?_cons_zero, hst1, hst2]
                      | succ s' =>
                          show colEff rest1 s' = colEff rest2 s'
                          by_cases hmin :
                              isMinorArrivalVertexC d e.START e.END = true ∧
                                p.tail = []
                          · simp only [if_pos hmin] at htail1 htail2
                            rw [colEff_all_stops (fun x hx => by
                                  obtain ⟨a, b⟩ := x
                                  exact Prod.ext (htail1 a b hx).1
                                    (htail1 a b hx).2),
                              colEff_all_stops (fun x hx => by
                                  obtain ⟨a, b⟩ := x
                                  exact Prod.ext (htail2 a b hx).1
                                    (htail2 a b hx).2)]
                          · simp only [if_neg hmin] at htail1 htail2
                            let next :=
                              ({ residual := p,
                                  exhaustedArrival := false } : ResidualCursor).afterEdge
                            by_cases hstop : exhaustedStopAllowedB d (lvl - 1)
                                e.END.FORMULA next = true
                            · rw [colEff_all_stops
                                  (admChainCursor_all_stops_of_allowed d formulas
                                    (lvl - 1) e.END.FORMULA next hstop rest1 htail1),
                                colEff_all_stops
                                  (admChainCursor_all_stops_of_allowed d formulas
                                    (lvl - 1) e.END.FORMULA next hstop rest2 htail2)]
                            · have hstop' : exhaustedStopAllowedB d (lvl - 1)
                                  e.END.FORMULA next = false :=
                                Bool.eq_false_of_not_eq_true hstop
                              rw [admChainCursor_eq_initial_of_not_allowed d formulas
                                  (lvl - 1) e.END.FORMULA next hstop' rest1] at htail1
                              rw [admChainCursor_eq_initial_of_not_allowed d formulas
                                  (lvl - 1) e.END.FORMULA next hstop' rest2] at htail2
                              exact ih (lvl - 1) e.END.FORMULA p.tail rest2
                                htail1 htail2 s'

lemma admHypColumnC_colEff_unique (d : Graph) (formulas : List Formula)
    (col lvl : Nat) (φ : Formula) (p : ColourPath) :
    ∀ (delay : Nat) (steps1 steps2 : List (Nat × Nat)),
      admHypColumnCB d formulas col lvl φ p delay steps1 = true →
      admHypColumnCB d formulas col lvl φ p delay steps2 = true →
      ∀ s, colEff steps1 s = colEff steps2 s := by
  intro delay
  induction delay with
  | zero =>
      intro steps1 steps2 h1 h2 s
      exact chainC_colEff_unique d formulas steps1 lvl φ p steps2 h1 h2 s
  | succ k ih =>
      intro steps1 steps2 h1 h2 s
      cases steps1 with
      | nil => simp [admHypColumnCB] at h1
      | cons st1 r1 =>
          cases steps2 with
          | nil => simp [admHypColumnCB] at h2
          | cons st2 r2 =>
              obtain ⟨t1, l1⟩ := st1
              obtain ⟨t2, l2⟩ := st2
              simp only [admHypColumnCB, Bool.and_eq_true] at h1 h2
              obtain ⟨hh1, ht1⟩ := h1
              obtain ⟨hh2, ht2⟩ := h2
              have hst1 : (t1, l1) = (col + 1, 0) :=
                Prod.ext (by simpa using hh1.1) (by simpa using hh1.2)
              have hst2 : (t2, l2) = (col + 1, 0) :=
                Prod.ext (by simpa using hh2.1) (by simpa using hh2.2)
              cases s with
              | zero => simp only [colEff, List.getElem?_cons_zero, hst1, hst2]
              | succ s' =>
                  show colEff r1 s' = colEff r2 s'
                  exact ih r1 r2 ht1 ht2 s'

lemma slotCheck_colEff_unique (d : Graph) (j : Nat)
    {steps1 steps2 : List (Nat × Nat)}
    (h1 : slotCheckB d j steps1 = true) (h2 : slotCheckB d j steps2 = true) :
    ∀ s, colEff steps1 s = colEff steps2 s := by
  intro s
  cases hfc : (buildFormulas d)[((routesOf d).getD j (0, 0)).1]? with
  | none =>
      simp only [slotCheckB, hfc] at h1 h2
      rw [colEff_all_stops (allStops_of h1), colEff_all_stops (allStops_of h2)]
  | some φ =>
      cases hsource : sourceRoute? d ((routesOf d).getD j (0, 0)).1
          ((routesOf d).getD j (0, 0)).2 with
      | none =>
          simp only [slotCheckB, hfc, hsource] at h1 h2
          rw [colEff_all_stops (allStops_of h1),
            colEff_all_stops (allStops_of h2)]
      | some source =>
          obtain ⟨v, bp⟩ := source
          simp only [slotCheckB, hfc, hsource] at h1 h2
          exact admHypColumnC_colEff_unique d (buildFormulas d)
            ((routesOf d).getD j (0, 0)).1
            v.LEVEL φ bp.2 _ steps1 steps2 h1 h2 s



private lemma findSome?_eq_some_ex {α β : Type _} {l : List α}
    {f : α → Option β} {b : β}
    (h : l.findSome? f = some b) : ∃ a ∈ l, f a = some b := by
  induction l with
  | nil => simp [List.findSome?] at h
  | cons x xs ih =>
      rw [List.findSome?_cons] at h
      cases hfx : f x with
      | some y =>
          rw [hfx] at h
          exact ⟨x, List.mem_cons_self .., by rw [hfx]; exact h⟩
      | none =>
          rw [hfx] at h
          obtain ⟨a, ha, hfa⟩ := ih h
          exact ⟨a, List.mem_cons_of_mem x ha, hfa⟩

lemma multiMatch?_length (check : Nat → List (Nat × Nat) → Bool)
    (colf : Nat → Nat) :
    ∀ (subs : List (List (Nat × Nat))) (i : Nat) (avail σ : List Nat),
      multiMatch? check colf i subs avail = some σ → σ.length = subs.length := by
  intro subs
  induction subs with
  | nil =>
      intro i avail σ h
      simp only [multiMatch?, Option.some.injEq] at h
      rw [← h]
      rfl
  | cons steps rest ih =>
      intro i avail σ h
      simp only [multiMatch?] at h
      obtain ⟨j, hjmem, hj⟩ := findSome?_eq_some_ex h
      by_cases hcond : (colf j == colf i && check j steps) = true
      · rw [if_pos hcond] at hj
        cases hrec : multiMatch? check colf (i + 1) rest (avail.erase j) with
        | none => rw [hrec] at hj; simp at hj
        | some σ' =>
            rw [hrec] at hj
            have hj2 : j :: σ' = σ := by simpa using hj
            rw [← hj2]
            simp [ih (i + 1) (avail.erase j) σ' hrec]
      · rw [if_neg hcond] at hj
        simp at hj

lemma multiMatch?_perm (check : Nat → List (Nat × Nat) → Bool)
    (colf : Nat → Nat) :
    ∀ (subs : List (List (Nat × Nat))) (i : Nat) (avail σ : List Nat),
      multiMatch? check colf i subs avail = some σ →
      subs.length = avail.length → σ.Perm avail := by
  intro subs
  induction subs with
  | nil =>
      intro i avail σ h hlen
      simp only [multiMatch?, Option.some.injEq] at h
      have havail : avail = [] := by
        cases avail with
        | nil => rfl
        | cons a b => simp at hlen
      rw [← h, havail]
  | cons steps rest ih =>
      intro i avail σ h hlen
      simp only [multiMatch?] at h
      obtain ⟨j, hjmem, hj⟩ := findSome?_eq_some_ex h
      by_cases hcond : (colf j == colf i && check j steps) = true
      · rw [if_pos hcond] at hj
        cases hrec : multiMatch? check colf (i + 1) rest (avail.erase j) with
        | none => rw [hrec] at hj; simp at hj
        | some σ' =>
            rw [hrec] at hj
            have hj2 : j :: σ' = σ := by simpa using hj
            have hlen' : rest.length = (avail.erase j).length := by
              rw [List.length_erase_of_mem hjmem]
              simp only [List.length_cons] at hlen
              omega
            have hperm' := ih (i + 1) (avail.erase j) σ' hrec hlen'
            rw [← hj2]
            exact (hperm'.cons j).trans (List.perm_cons_erase hjmem).symm
      · rw [if_neg hcond] at hj
        simp at hj

lemma multiMatch?_valid (check : Nat → List (Nat × Nat) → Bool)
    (colf : Nat → Nat) :
    ∀ (subs : List (List (Nat × Nat))) (i : Nat) (avail σ : List Nat),
      multiMatch? check colf i subs avail = some σ →
      ∀ k (hk : k < σ.length) (hk2 : k < subs.length),
        colf (σ[k]) = colf (i + k) ∧ check (σ[k]) (subs[k]) = true := by
  intro subs
  induction subs with
  | nil =>
      intro i avail σ h k hk hk2
      simp at hk2
  | cons steps rest ih =>
      intro i avail σ h k hk hk2
      simp only [multiMatch?] at h
      obtain ⟨j, hjmem, hj⟩ := findSome?_eq_some_ex h
      by_cases hcond : (colf j == colf i && check j steps) = true
      · rw [if_pos hcond] at hj
        cases hrec : multiMatch? check colf (i + 1) rest (avail.erase j) with
        | none => rw [hrec] at hj; simp at hj
        | some σ' =>
            rw [hrec] at hj
            have hj2 : j :: σ' = σ := by simpa using hj
            rw [Bool.and_eq_true] at hcond
            subst hj2
            cases k with
            | zero =>
                refine ⟨?_, by simpa using hcond.2⟩
                simp only [Nat.add_zero, List.getElem_cons_zero]
                have hcol := hcond.1
                simp only [beq_iff_eq] at hcol
                exact hcol
            | succ k' =>
                have hk' : k' < σ'.length := by simpa using hk
                have hk2' : k' < rest.length := by simpa using hk2
                have hrec' := ih (i + 1) (avail.erase j) σ' hrec k' hk' hk2'
                simp only [List.getElem_cons_succ]
                refine ⟨?_, hrec'.2⟩
                rw [hrec'.1]
                congr 1
                omega
      · rw [if_neg hcond] at hj
        simp at hj



lemma multiPathsFromFlow_length (d : Graph) :
    (multiPathsFromFlow d).length = routeCount d := by
  unfold multiPathsFromFlow routeCount
  simp

private lemma perm_range_getElem_idxOf {σ : List Nat} {T : Nat}
    (hperm : σ.Perm (List.range T)) (hnodup : σ.Nodup) {c : Nat} (hc : c < T) :
    ∃ hlt : σ.idxOf c < σ.length, σ[σ.idxOf c]'hlt = c := by
  have hcmem : c ∈ σ := hperm.mem_iff.mpr (List.mem_range.mpr hc)
  obtain ⟨k, hk, hgetk⟩ := List.mem_iff_getElem.mp hcmem
  have hkidx : σ.idxOf c = k :=
    indexOf_eq_of_get hk hnodup (by simpa [List.get_eq_getElem] using hgetk)
  refine ⟨by rw [hkidx]; exact hk, ?_⟩
  simp only [hkidx]
  exact hgetk

/--
 Every admissible multi-path is, after a
    permutation of co-located routes' subpaths, read-equivalent to the
    canonical `multiPathsFromFlow`.
-/
def AdmissibleMultiReducesToCanonical (d : Graph) : Prop :=
  ∀ P, AdmissibleMultiPath d P →
    ∃ P', ColocPerm d P P' ∧ ReadEquiv P' (multiPathsFromFlow d)

/--
 Under the decidable canonical-slot-validity side
    condition `canonicalSlotOKB`, eval-gated true on all fixtures): the
    matcher's assignment IS the permutation; within each route the coloured
    chain is forced (`slotCheck_colEff_unique`).
-/
theorem admissibleMultiReducesToCanonical (d : Graph)
    (hcanon : canonicalSlotOKB d = true) :
    AdmissibleMultiReducesToCanonical d := by
  intro P hadm
  have hadmB : admissibleMultiPathB d P = true := hadm
  unfold admissibleMultiPathB at hadmB
  rw [Bool.and_eq_true, Bool.and_eq_true] at hadmB
  obtain ⟨⟨hlen, hmatch⟩, _hgoal⟩ := hadmB
  have hlenEq : P.length = routeCount d := by simpa using hlen
  rw [Option.isSome_iff_exists] at hmatch
  obtain ⟨σ, hσ⟩ := hmatch
  have hσlen : σ.length = P.length :=
    multiMatch?_length (slotCheckB d) (slotColumn d) P 0 _ σ hσ
  have hσperm : σ.Perm (List.range (routeCount d)) :=
    multiMatch?_perm (slotCheckB d) (slotColumn d) P 0 _ σ hσ
      (by rw [hlenEq, List.length_range])
  have hσnodup : σ.Nodup := hσperm.nodup_iff.mpr List.nodup_range
  have hσvalid := multiMatch?_valid (slotCheckB d) (slotColumn d) P 0 _ σ hσ
  have hcanonAt : ∀ j ∈ List.range (routeCount d),
      slotCheckB d j ((multiPathsFromFlow d).getD j []) = true :=
    fun j hj => (List.all_eq_true.mp hcanon) j hj
  refine ⟨(List.range (routeCount d)).map
      (fun j => P.getD (σ.idxOf j) []), ?_, ?_⟩
  ·
    unfold ColocPerm
    have hpairsP' : colocPairs d ((List.range (routeCount d)).map
        (fun j => P.getD (σ.idxOf j) [])) =
      (List.range (routeCount d)).map
        (fun j => (slotColumn d j, P.getD (σ.idxOf j) [])) := by
      unfold colocPairs
      apply List.map_congr_left
      intro j hj
      have hj' : j < routeCount d := List.mem_range.mp hj
      have hgd : ((List.range (routeCount d)).map
          (fun j => P.getD (σ.idxOf j) [])).getD j [] =
          P.getD (σ.idxOf j) [] := by
        rw [List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem (by simpa using hj')]
        simp
      rw [hgd]
    have hpairsP : colocPairs d P =
        σ.map (fun j => (slotColumn d j, P.getD (σ.idxOf j) [])) := by
      unfold colocPairs
      apply List.ext_getElem
      · simp only [List.length_map, List.length_range]
        omega
      · intro k h1 h2
        simp only [List.getElem_map, List.getElem_range]
        have hkσ : k < σ.length := by simpa using h2
        have hkP : k < P.length := by omega
        have hidxk : σ.idxOf (σ[k]'hkσ) = k :=
          indexOf_eq_of_get hkσ hσnodup (by simp [List.get_eq_getElem])
        have hcolk : slotColumn d (σ[k]'hkσ) = slotColumn d k := by
          have hval := (hσvalid k hkσ (by omega)).1
          simpa using hval
        rw [hidxk, hcolk]
    rw [hpairsP, hpairsP']
    exact hσperm.map _
  ·
    intro c s
    by_cases hc : c < routeCount d
    · rw [effStep_colEff (P := (List.range (routeCount d)).map
          (fun j => P.getD (σ.idxOf j) [])) (by simpa using hc),
        effStep_colEff (P := multiPathsFromFlow d)
          (by rw [multiPathsFromFlow_length]; exact hc)]
      have hP'c : ((List.range (routeCount d)).map
          (fun j => P.getD (σ.idxOf j) [])).getD c [] =
          P.getD (σ.idxOf c) [] := by
        rw [List.getD_eq_getElem?_getD,
          List.getElem?_eq_getElem (by simpa using hc)]
        simp
      rw [hP'c]
      obtain ⟨hlt, hgc⟩ := perm_range_getElem_idxOf hσperm hσnodup hc
      have hkP : σ.idxOf c < P.length := by omega
      have hvalid := (hσvalid (σ.idxOf c) hlt (by omega)).2
      rw [hgc] at hvalid
      have hPgetD : P.getD (σ.idxOf c) [] = P[σ.idxOf c]'hkP := by
        rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hkP]
        rfl
      rw [hPgetD]
      exact slotCheck_colEff_unique d c hvalid
        (hcanonAt c (List.mem_range.mpr hc)) s
    · rw [effStep_ge_col (by simpa using hc),
        effStep_ge_col (by rw [multiPathsFromFlow_length]; exact hc)]



/--
 Every admissible multi-path evaluates identically to the canonical
    (M2c reduction + M2b multiset congruence + read-equivalence congruence).
-/
theorem admissibleMulti_eval_canonical (d : Graph)
    (hcanon : canonicalSlotOKB d = true) {P : MultiPathInput}
    (hadm : AdmissibleMultiPath d P) :
    getEvalResultMultiDLDS d P =
      getEvalResultMultiDLDS d (multiPathsFromFlow d) := by
  obtain ⟨P', hcp, hre⟩ := admissibleMultiReducesToCanonical d hcanon P hadm
  rw [tokenMultiset_congr d hcp]
  exact getEvalResultMultiDLDS_congr d hre

/--  Acceptance form of the combined corollary.  -/
theorem admissibleMulti_accept_canonical (d : Graph)
    (hcanon : canonicalSlotOKB d = true) {P : MultiPathInput}
    (hadm : AdmissibleMultiPath d P) (g : Nat) :
    evaluateDLDS_multi d P g =
      evaluateDLDS_multi d (multiPathsFromFlow d) g := by
  unfold evaluateDLDS_multi
  rw [admissibleMulti_eval_canonical d hcanon hadm]




end DLDS
