import Semantic.DLDS

open scoped Classical

namespace Semantic

/-! # Simple-tree bridge to genuine circuit acceptance. -/

lemma coherent_base_layer (formulas : List Formula)
    (vecs : List (List.Vector Bool formulas.length)) (tl : Nat)
    (hvecs : vecs.length = formulas.length)
    (hnodup : formulas.Nodup) :
    TokensCoherentAtLayer
      { nodes := formulas.map (nodeForFormula formulas)
        incoming := buildIncomingMap formulas }
      (initialize_tokens vecs tl) := by
  intro col hNode hIncoming
  have hcolF : col < formulas.length := by
    simpa using hNode
  have hcolV : col < vecs.length := by omega
  rw [init_tokens_filter_singleton vecs tl col]
  simp [hcolV]
  let formula := formulas[col]'hcolF
  let node := nodeForFormula formulas formula
  change TokensMatchOneRule node ((buildIncomingMap formulas)[col]'hIncoming)
    [initTokenAt vecs tl col hcolV]
  have hincDef :
      ((buildIncomingMap formulas)[col]'hIncoming) =
        buildIncomingMapForFormula formulas formula := by
    simp [formula, buildIncomingMap]
  rw [hincDef]
  let incoming := buildIncomingMapForFormula formulas formula
  change TokensMatchOneRule node incoming [initTokenAt vecs tl col hcolV]
  unfold TokensMatchOneRule
  let r := incoming.length - 1
  have hIncomingPos : 0 < incoming.length := by
    exact buildIncomingMapForFormula_length_pos formulas formula
  have hrIncoming : r < incoming.length := by
    dsimp [r]
    omega
  have hrule : r < node.rules.length := by
    dsimp [node, incoming, r]
    rw [← incoming_rules_aligned_length formulas formula]
    omega
  have hidxOf : formulas.idxOf formula = col := by
    exact indexOf_eq_of_get hcolF hnodup rfl
  have hdec : decodeInputLabel incoming 0 = some (r, 0, col) := by
    dsimp [incoming, r]
    simpa [formula, hidxOf] using
      decodeInputLabel_zero_buildIncomingMapForFormula formulas formula
  have harity : ((incoming[r]?.getD default).length = 1) := by
    dsimp [r]
    simpa [incoming] using buildIncomingMapForFormula_last_rep_length formulas formula
  refine ⟨r, 0, col, hdec, hrule, ?_, ?_, ?_⟩
  · intro s hs
    simp only [List.mem_singleton] at hs
    subst s
    refine ⟨0, col, hdec, ?_, rfl⟩
    rw [harity]
    omega
  · simp [harity]
  · intro i hi
    have hi0 : i = 0 := by
      rw [harity] at hi
      omega
    subst i
    refine ⟨initTokenAt vecs tl col hcolV, by simp, col, ?_, rfl⟩
    exact hdec

lemma initialVectorsFromDLDS_length (d : Graph) :
    (initialVectorsFromDLDS d).length = (buildFormulas d).length := by
  unfold initialVectorsFromDLDS
  simp

private lemma eraseDupsBy_loop_nodup {α : Type*} [BEq α] [LawfulBEq α] :
    ∀ (xs acc : List α), acc.Nodup →
      (List.eraseDupsBy.loop (fun x y : α => x == y) xs acc).Nodup
  | [], acc, hacc => by
      rw [List.eraseDupsBy.loop.eq_1]
      exact List.nodup_reverse.mpr hacc
  | a :: xs, acc, hacc => by
      rw [List.eraseDupsBy.loop.eq_2]
      cases h : acc.any (fun y => a == y)
      · apply eraseDupsBy_loop_nodup xs (a :: acc)
        exact List.nodup_cons.mpr ⟨by
          intro hmem
          have hany : acc.any (fun y => a == y) = true := by
            rw [List.any_eq_true]
            exact ⟨a, hmem, by simp⟩
          rw [h] at hany
          contradiction, hacc⟩
      · exact eraseDupsBy_loop_nodup xs acc hacc

lemma List.eraseDups_nodup' {α : Type*} [BEq α] [LawfulBEq α] (l : List α) :
    l.eraseDups.Nodup := by
  rw [List.eraseDups.eq_1]
  exact eraseDupsBy_loop_nodup l [] (by simp)

lemma buildFormulas_nodup (d : Graph) :
    (buildFormulas d).Nodup := by
  unfold buildFormulas
  exact List.eraseDups_nodup' _

lemma buildGridFromDLDS_getD_zero (d : Graph) :
    ((buildGridFromDLDS d).getD 0 { nodes := [], incoming := [] }) =
      { nodes := (buildFormulas d).map (nodeForFormula (buildFormulas d))
        incoming := buildIncomingMap (buildFormulas d) } := by
  unfold buildGridFromDLDS buildLayers
  simp

lemma coherent_base (d : Graph) (_hvalid : ValidDLDS d) :
    TokensCoherentAtLayer
      ((buildGridFromDLDS d).getD 0 { nodes := [], incoming := [] })
      (initialize_tokens (initialVectorsFromDLDS d) (buildGridFromDLDS d).length) := by
  rw [buildGridFromDLDS_getD_zero]
  exact coherent_base_layer (buildFormulas d) (initialVectorsFromDLDS d)
    (buildGridFromDLDS d).length
    (initialVectorsFromDLDS_length d)
    (buildFormulas_nodup d)

lemma coherent_base_wellRouted (d : Graph) (hvalid : ValidDLDS d) :
    TokensWellRoutedAtLayer d 0
      ((buildGridFromDLDS d).getD 0 { nodes := [], incoming := [] })
      (initialize_tokens (initialVectorsFromDLDS d) (buildGridFromDLDS d).length) := by
  constructor
  · intro t ht
    rcases init_token_at_col (initialVectorsFromDLDS d) (buildGridFromDLDS d).length t |>.mp ht with
      ⟨col, hcol, rfl⟩
    have hcolF : col < (buildFormulas d).length := by
      simpa [initialVectorsFromDLDS_length d] using hcol
    constructor
    · exact hcolF
    · have hpathLen : col < (pathsFromDLDS d).length := by
        simpa [pathsFromDLDS] using hcolF
      simp [routeStateAfter, initTokenAt, hpathLen]
  · exact coherent_base d hvalid

lemma coherent_base_exact (d : Graph) (hvalid : ValidDLDS d) :
    TokensExactAtLayer d 0
      ((buildGridFromDLDS d).getD 0 { nodes := [], incoming := [] })
      (initialize_tokens (initialVectorsFromDLDS d) (buildGridFromDLDS d).length) := by
  constructor
  · exact coherent_base_wellRouted d hvalid
  · intro origin horigin
    have hpath : origin < (pathsFromDLDS d).length := by
      simpa [pathsFromDLDS] using horigin
    simp [routeStateAfter, hpath]
    let vecs := initialVectorsFromDLDS d
    let tl := (buildGridFromDLDS d).length
    have hvec : origin < vecs.length := by
      simpa [vecs, initialVectorsFromDLDS_length d] using horigin
    let t0 := initTokenAt vecs tl origin hvec
    refine ⟨t0, ?_, rfl, ?_⟩
    · rw [init_token_at_col]
      exact ⟨origin, hvec, rfl⟩
    · intro s hs hsorigin
      rcases (init_token_at_col vecs tl s).mp hs with ⟨col, hcol, rfl⟩
      dsimp [initTokenAt] at hsorigin ⊢
      subst hsorigin
      rfl

lemma tokensPresence_step (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (i level : Nat)
    (hj : i + 1 < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (hp : TokensPresenceAtDepth d i tokens)
    (hwr : ∀ t ∈ tokens, TokenWellRoutedAtDepth d i t)
    (outs : List (List.Vector Bool (buildFormulas d).length))
    (houts : outs.length = (buildFormulas d).length) :
    TokensPresenceAtDepth d (i + 1)
      (propagate_tokens tokens (pathsFromDLDS d) level
        (buildGridFromDLDS d).length outs) := by
  intro origin horigin
  cases hnext : routeStateAfter (pathsFromDLDS d) origin (i + 1) with
  | none =>
      intro t' ht' horigin'
      unfold propagate_tokens at ht'
      rw [List.mem_filterMap] at ht'
      obtain ⟨t, ht_mem, ht_some⟩ := ht'
      dsimp only at ht_some
      split at ht_some
      · rename_i h_path
        split at ht_some
        · rename_i h_level
          split at ht_some
          · simp at ht_some
          · rename_i h_stop
            split at ht_some
            · rename_i h_out
              rw [Option.some.injEq] at ht_some
              subst t'
              have hstate := (hwr t ht_mem).2
              have hstepEq : (buildGridFromDLDS d).length - level - 1 = i := by
                omega
              have hstepBound :
                  i < ((pathsFromDLDS d).get ⟨t.origin_column, h_path⟩).length := by
                simpa [hstepEq] using h_level.2
              have hstepBoundElem :
                  i < ((pathsFromDLDS d)[t.origin_column]'h_path).length := by
                simpa [List.get_eq_getElem] using hstepBound
              have hnext_t :=
                routeStateAfter_get_eq (pathsFromDLDS d)
                  (origin := t.origin_column) (depth := i)
                  (current := t.current_column) (source := t.source_column)
                  (label := t.input_label) h_path hstate hstepBoundElem
              have hstop_i :
                  ¬ (((pathsFromDLDS d)[t.origin_column]'h_path)[i]'hstepBoundElem).1 = 0 := by
                simpa [List.get_eq_getElem, hstepEq] using h_stop
              have hsome :
                  routeStateAfter (pathsFromDLDS d) t.origin_column (i + 1) =
                    some
                      ((((pathsFromDLDS d)[t.origin_column]'h_path)[i]'hstepBoundElem).1 - 1,
                        t.current_column,
                        (((pathsFromDLDS d)[t.origin_column]'h_path)[i]'hstepBoundElem).2) := by
                rw [hnext_t]
                simp [hstop_i]
              have hto : t.origin_column = origin := by
                simpa using horigin'
              have hnext_t_origin :
                  routeStateAfter (pathsFromDLDS d) t.origin_column (i + 1) = none := by
                simpa [hto] using hnext
              rw [hnext_t_origin] at hsome
              contradiction
            · simp at ht_some
        · simp at ht_some
      · simp at ht_some
  | some st =>
      rcases st with ⟨current', source', label'⟩
      obtain ⟨current, source, label, steps, target, inputLabel,
        hprev, hsteps, hstep, htarget, _htriple⟩ :=
        routeStateAfter_live_succ (pathsFromDLDS d) hnext
      have hprev_presence := hp origin horigin
      rw [hprev] at hprev_presence
      obtain ⟨t, ht_mem, htorigin, htuniq⟩ := hprev_presence
      have htw := hwr t ht_mem
      have hpath : t.origin_column < (pathsFromDLDS d).length := by
        rw [htorigin]
        simpa [pathsFromDLDS] using horigin
      have hstate_t : routeStateAfter (pathsFromDLDS d) t.origin_column i =
          some (t.current_column, t.source_column, t.input_label) := htw.2
      have hstepsGet :
          (pathsFromDLDS d).get ⟨t.origin_column, hpath⟩ = steps :=
        path_get_of_origin_eq (pathsFromDLDS d) htorigin hsteps hpath
      subst steps
      have hstepEq : (buildGridFromDLDS d).length - level - 1 = i := by
        omega
      have hstepLt : i < ((pathsFromDLDS d).get ⟨t.origin_column, hpath⟩).length :=
        getElem?_some_lt hstep
      have hstepLtEval :
          (buildGridFromDLDS d).length - level - 1 <
            ((pathsFromDLDS d).get ⟨t.origin_column, hpath⟩).length := by
        simpa [hstepEq] using hstepLt
      have hstepGetEval :
          ((pathsFromDLDS d).get ⟨t.origin_column, hpath⟩).get
            ⟨(buildGridFromDLDS d).length - level - 1, hstepLtEval⟩ =
            (target, inputLabel) :=
        path_step_get_of_step_eq hstep hstepEq hstepLtEval
      have hlevel : level > 0 ∧
          (buildGridFromDLDS d).length - level - 1 <
            ((pathsFromDLDS d).get ⟨t.origin_column, hpath⟩).length := by
        constructor
        · omega
        · exact hstepLtEval
      have hstopEval :
          ¬ (((pathsFromDLDS d).get ⟨t.origin_column, hpath⟩).get
            ⟨(buildGridFromDLDS d).length - level - 1, hstepLtEval⟩).1 = 0 := by
        rw [hstepGetEval]
        exact htarget
      have hcurrent_lt : t.current_column < (buildFormulas d).length := by
        exact routeStateAfter_pathsFromDLDS_current_lt d htree hvalid horigin
          (by simpa [htorigin] using hstate_t)
      have hout : t.current_column < outs.length := by
        simpa [houts] using hcurrent_lt
      let step :=
        ((pathsFromDLDS d).get ⟨t.origin_column, hpath⟩).get
          ⟨(buildGridFromDLDS d).length - level - 1, hstepLtEval⟩
      let tnext : Token (buildFormulas d).length :=
        { origin_column := t.origin_column
          source_column := t.current_column
          current_level := level - 1
          current_column := step.1 - 1
          dep_vector := outs.get ⟨t.current_column, hout⟩
          input_label := step.2 }
      have ht_next_mem :
          tnext ∈ propagate_tokens tokens (pathsFromDLDS d) level
            (buildGridFromDLDS d).length outs := by
        unfold propagate_tokens
        rw [List.mem_filterMap]
        refine ⟨t, ht_mem, ?_⟩
        dsimp only
        simp only [hpath, ↓reduceDIte, hlevel, hstopEval, hout]
        rfl
      refine ⟨tnext, ht_next_mem, ?_, ?_⟩
      · simp [tnext, htorigin]
      · intro s hs hsorigin
        unfold propagate_tokens at hs
        rw [List.mem_filterMap] at hs
        obtain ⟨q, hq_mem, hq_some⟩ := hs
        dsimp only at hq_some
        split at hq_some
        · rename_i hq_path
          split at hq_some
          · rename_i hq_level
            split at hq_some
            · simp at hq_some
            · rename_i hq_stop
              split at hq_some
              · rename_i hq_out
                rw [Option.some.injEq] at hq_some
                subst s
                have hqorigin : q.origin_column = origin := by
                  simpa using hsorigin
                have hqeq : q = t := htuniq q hq_mem hqorigin
                subst hqeq
                -- Both successors come from the same token and step.
                -- The remaining equalities are proof-irrelevant bounds.
                simp only [tnext, step]
              · simp at hq_some
          · simp at hq_some
        · simp at hq_some

lemma coherent_step_wellRouted (d : Graph) (_htree : IsSimpleTreeDLDS d) (_hvalid : ValidDLDS d)
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (hwell : TokensWellRoutedAtLayer d i ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens) :
    let outs := (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1
    ∀ (_hj : i + 1 < (buildGridFromDLDS d).length),
      ∀ t' ∈ (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length outs),
        TokenWellRoutedAtDepth d (i + 1) t' := by
  dsimp only
  intro _hj t' ht'
  unfold propagate_tokens at ht'
  rw [List.mem_filterMap] at ht'
  obtain ⟨t, ht_mem, ht_some⟩ := ht'
  dsimp only at ht_some
  split at ht_some
  · rename_i h_path
    split at ht_some
    · rename_i h_level
      split at ht_some
      · simp at ht_some
      · rename_i h_stop
        split at ht_some
        · rename_i h_out
          rw [Option.some.injEq] at ht_some
          subst t'
          rcases hwell.1 t ht_mem with ⟨horigin, hstate⟩
          constructor
          · exact horigin
          · have hstepEq : (buildGridFromDLDS d).length - level - 1 = i := by
              omega
            have hstepBound :
                i < ((pathsFromDLDS d).get ⟨t.origin_column, h_path⟩).length := by
              simpa [hstepEq] using h_level.2
            have hstepBoundElem :
                i < ((pathsFromDLDS d)[t.origin_column]'h_path).length := by
              simpa [List.get_eq_getElem] using hstepBound
            have hnext :=
              routeStateAfter_get_eq (pathsFromDLDS d)
                (origin := t.origin_column) (depth := i)
                (current := t.current_column) (source := t.source_column)
                (label := t.input_label) h_path hstate hstepBoundElem
            rw [hnext]
            have hstop_i :
                ¬ (((pathsFromDLDS d)[t.origin_column]'h_path)[i]'hstepBoundElem).1 = 0 := by
              simpa [List.get_eq_getElem, hstepEq] using h_stop
            simp [hstop_i, hstepEq]
        · simp at ht_some
    · simp at ht_some
  · simp at ht_some

lemma coherent_step_routeAligned
    (d : Graph) (_htree : IsSimpleTreeDLDS d) (_hvalid : ValidDLDS d)
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (halign : TokensRouteAlignedAtDepth d i tokens) :
    let outs := (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1
    ∀ (_hj : i + 1 < (buildGridFromDLDS d).length),
      ∀ t' ∈ (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length outs),
        TokenWellRoutedAtDepth d (i + 1) t' := by
  dsimp only
  intro _hj t' ht'
  unfold propagate_tokens at ht'
  rw [List.mem_filterMap] at ht'
  obtain ⟨t, ht_mem, ht_some⟩ := ht'
  dsimp only at ht_some
  split at ht_some
  · rename_i h_path
    split at ht_some
    · rename_i h_level
      split at ht_some
      · simp at ht_some
      · rename_i h_stop
        split at ht_some
        · rename_i h_out
          rw [Option.some.injEq] at ht_some
          subst t'
          rcases halign t ht_mem with ⟨horigin, hstate⟩
          constructor
          · exact horigin
          · have hstepEq : (buildGridFromDLDS d).length - level - 1 = i := by
              omega
            have hstepBound :
                i < ((pathsFromDLDS d).get ⟨t.origin_column, h_path⟩).length := by
              simpa [hstepEq] using h_level.2
            have hstepBoundElem :
                i < ((pathsFromDLDS d)[t.origin_column]'h_path).length := by
              simpa [List.get_eq_getElem] using hstepBound
            have hnext :=
              routeStateAfter_get_eq (pathsFromDLDS d)
                (origin := t.origin_column) (depth := i)
                (current := t.current_column) (source := t.source_column)
                (label := t.input_label) h_path hstate hstepBoundElem
            rw [hnext]
            have hstop_i :
                ¬ (((pathsFromDLDS d)[t.origin_column]'h_path)[i]'hstepBoundElem).1 = 0 := by
              simpa [List.get_eq_getElem, hstepEq] using h_stop
            simp [hstop_i, hstepEq]
        · simp at ht_some
    · simp at ht_some
  · simp at ht_some



/--
 Boolean check: at every trace layer, each column's token group satisfies
    `tokensMatchOneRuleB` (empty ⟹ true; non-empty ⟹ one rule, count=arity, all slots).
-/
def uniqueTokenPerSlotB (d : Graph) : Bool :=
  let layers := buildGridFromDLDS d
  let traces := tokenTraceDLDS d
  (traces.zipIdx 0).all fun ⟨tokens, depth⟩ =>
    match layers[depth]? with
    | none => true
    | some layer => tokensCoherentAtLayerB layer tokens

lemma tokensMatchOneRuleB_implies_Prop {n : Nat}
    (node : CircuitNode n) (inc : NodeIncoming) (toks : List (Token n))
    (h : tokensMatchOneRuleB node inc toks = true) :
    TokensMatchOneRule node inc toks := by
  cases toks with
  | nil => trivial
  | cons t ts =>
      simp only [tokensMatchOneRuleB] at h
      show ∃ r slot src, _
      cases hdec : decodeInputLabel inc t.input_label with
      | none => rw [hdec] at h; exact absurd h (by simp)
      | some rss =>
          obtain ⟨r, slot0, src0⟩ := rss
          rw [hdec] at h
          simp only [Bool.and_eq_true, decide_eq_true_eq] at h
          obtain ⟨⟨⟨hrule, hall⟩, hlen⟩, hany⟩ := h
          refine ⟨r, slot0, src0, rfl, hrule, ?_, hlen, ?_⟩
          · intro s hs
            rw [List.all_eq_true] at hall
            have hcheck := hall s hs
            cases hdecs : decodeInputLabel inc s.input_label with
            | none => rw [hdecs] at hcheck; exact absurd hcheck (by simp)
            | some rss' =>
                obtain ⟨r', slot', src'⟩ := rss'
                rw [hdecs] at hcheck
                simp only [decide_eq_true_eq] at hcheck
                obtain ⟨hr', hslot', hsrc'⟩ := hcheck
                subst hr'
                exact ⟨slot', src', rfl, hslot', hsrc'⟩
          · intro i hi
            rw [List.all_eq_true] at hany
            have hcheck := hany i (List.mem_range.mpr hi)
            rw [List.any_eq_true] at hcheck
            obtain ⟨s, hs, hsbool⟩ := hcheck
            cases hdecs : decodeInputLabel inc s.input_label with
            | none => rw [hdecs] at hsbool; exact absurd hsbool (by simp)
            | some rss' =>
                obtain ⟨r', slot', src'⟩ := rss'
                rw [hdecs] at hsbool
                simp only [decide_eq_true_eq] at hsbool
                obtain ⟨hr', hslot', hsrc'⟩ := hsbool
                subst hr'; subst hslot'
                exact ⟨s, hs, src', hdecs, hsrc'⟩

lemma tokensCoherentAtLayerB_implies_Prop {n : Nat}
    (layer : GridLayer n) (toks : List (Token n))
    (h : tokensCoherentAtLayerB layer toks = true) :
    TokensCoherentAtLayer layer toks := by
  intro col hN hI
  unfold tokensCoherentAtLayerB at h
  rw [List.all_eq_true] at h
  have hcol := h col (List.mem_range.mpr hN)
  rw [List.getElem?_eq_getElem hN, List.getElem?_eq_getElem hI] at hcol
  simp only at hcol
  have hP := tokensMatchOneRuleB_implies_Prop _ _ _ hcol
  simpa [List.get_eq_getElem] using hP

lemma tokensMatchOneRule_Prop_implies_B {n : Nat}
    (node : CircuitNode n) (inc : NodeIncoming) (toks : List (Token n))
    (h : TokensMatchOneRule node inc toks) :
    tokensMatchOneRuleB node inc toks = true := by
  cases toks with
  | nil => rfl
  | cons t ts =>
      unfold TokensMatchOneRule at h
      obtain ⟨r, slot, src, hdec, hrule, hall, hlen, hslots⟩ := h
      simp only [tokensMatchOneRuleB, hdec]
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      refine ⟨⟨⟨hrule, ?_⟩, hlen⟩, ?_⟩
      · rw [List.all_eq_true]
        intro s hs
        obtain ⟨slot', src', hdec_s, hslot', hsrc'⟩ := hall s hs
        rw [hdec_s]
        simp [hslot', hsrc']
      · rw [List.all_eq_true]
        intro i hiMem
        have hi : i < (inc[r]?.getD default).length := List.mem_range.mp hiMem
        obtain ⟨s, hs, src', hdec_s, hsrc'⟩ := hslots i hi
        rw [List.any_eq_true]
        refine ⟨s, hs, ?_⟩
        rw [hdec_s]
        simp [hsrc']

lemma tokensCoherentAtLayer_Prop_implies_B {n : Nat}
    (layer : GridLayer n) (toks : List (Token n))
    (hlen : layer.incoming.length = layer.nodes.length)
    (h : TokensCoherentAtLayer layer toks) :
    tokensCoherentAtLayerB layer toks = true := by
  unfold tokensCoherentAtLayerB
  rw [List.all_eq_true]
  intro col hmem
  have hN : col < layer.nodes.length := List.mem_range.mp hmem
  cases hIopt : layer.incoming[col]? with
  | none =>
      have hI : col < layer.incoming.length := by
        rwa [hlen]
      rw [List.getElem?_eq_getElem hI] at hIopt
      simp at hIopt
  | some incoming =>
      have hI : col < layer.incoming.length := getElem?_some_lt hIopt
      have hincoming :
          layer.incoming.get ⟨col, hI⟩ = incoming := by
        have hsome :
            layer.incoming[col]? = some (layer.incoming.get ⟨col, hI⟩) := by
          simp [List.get_eq_getElem]
        rw [hsome] at hIopt
        exact Option.some.inj hIopt
      have hmatch :
          TokensMatchOneRule (layer.nodes.get ⟨col, hN⟩) incoming
            (toks.filter (fun t => t.current_column = col)) := by
        rw [← hincoming]
        exact h col hN hI
      have hB := tokensMatchOneRule_Prop_implies_B _ _ _ hmatch
      simpa [List.getElem?_eq_getElem hN, hIopt, List.get_eq_getElem] using hB

/--
 The concrete token list at descent depth `k`: the `k`-fold `propagate_tokens`
    iteration starting from `initialize_tokens`, threading each layer's
    `evaluate_layer` output exactly as `eval_from_level`/`tokenTraceAux` do. This is
    the actual token stream the evaluator carries; the cascade threads it so the
    decidable route-coherence certificate `routeCoherentB` applies layer by layer.
-/
def descentTokens (d : Graph) : Nat → List (Token (buildFormulas d).length)
  | 0 => initialize_tokens (initialVectorsFromDLDS d) (buildGridFromDLDS d).length
  | k + 1 =>
      propagate_tokens (descentTokens d k) (pathsFromDLDS d)
        ((buildGridFromDLDS d).length - 1 - k) (buildGridFromDLDS d).length
        (evaluate_layer ((buildGridFromDLDS d).getD k { nodes := [], incoming := [] })
          (descentTokens d k)).1

lemma descent_route_aligned_presence
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d) :
    ∀ depth, depth < (buildGridFromDLDS d).length →
      TokensRouteAlignedAtDepth d depth (descentTokens d depth) ∧
      TokensPresenceAtDepth d depth (descentTokens d depth) := by
  intro depth hd
  induction depth with
  | zero =>
      have hbase := coherent_base_exact d hvalid
      constructor
      · simpa [descentTokens] using hbase.1.1
      · simpa [descentTokens] using hbase.2
  | succ depth ih =>
      have hi : depth < (buildGridFromDLDS d).length := by omega
      have hj : depth + 1 < (buildGridFromDLDS d).length := hd
      obtain ⟨halign, hpres⟩ := ih hi
      let level := (buildGridFromDLDS d).length - 1 - depth
      have hgetD_i :
          (buildGridFromDLDS d).get ⟨depth, hi⟩ =
            (buildGridFromDLDS d).getD depth { nodes := [], incoming := [] } := by
        rw [List.get_eq_getElem]
        exact List.getElem_eq_getD _
      have hstepTok :
          descentTokens d (depth + 1) =
            propagate_tokens (descentTokens d depth) (pathsFromDLDS d) level
              (buildGridFromDLDS d).length
              (evaluate_layer ((buildGridFromDLDS d).get ⟨depth, hi⟩)
                (descentTokens d depth)).1 := by
        dsimp [descentTokens, level]
        rw [← hgetD_i]
        rw [List.get_eq_getElem]
      constructor
      · intro t ht
        have hraw :=
          coherent_step_routeAligned d htree hvalid depth level hi (by rfl)
            (descentTokens d depth) halign hj
        exact hraw t (by simpa [hstepTok] using ht)
      · have houts_len :
            (evaluate_layer ((buildGridFromDLDS d).get ⟨depth, hi⟩)
              (descentTokens d depth)).1.length =
              (buildFormulas d).length := by
          have := evaluate_layer_outputs_length
            ((buildGridFromDLDS d).get ⟨depth, hi⟩) (descentTokens d depth)
          rwa [buildGridFromDLDS_get_nodes_length] at this
        have hpres_next :=
          tokensPresence_step d htree hvalid depth level hj (by rfl)
            (descentTokens d depth) hpres halign
            (evaluate_layer ((buildGridFromDLDS d).get ⟨depth, hi⟩)
              (descentTokens d depth)).1 houts_len
        simpa [hstepTok] using hpres_next
/--
 **Decidable singleton Flow-condition certificate.** At every descent depth,
    each column's token group covers its selected rule's slots exactly once.
    Scope: simple-tree singleton routing only; no colour fan-out, λ-edges, or
    collapsed-node branching from the full Definition 3 setting.
-/
def routeCoherentB (d : Graph) : Bool :=
  (List.range (buildGridFromDLDS d).length).all (fun depth =>
    tokensCoherentAtLayerB ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
      (descentTokens d depth))

lemma routeCoherentB_layer (d : Graph) (hcert : routeCoherentB d = true)
    (depth : Nat) (hd : depth < (buildGridFromDLDS d).length) :
    tokensCoherentAtLayerB ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
      (descentTokens d depth) = true := by
  unfold routeCoherentB at hcert
  rw [List.all_eq_true] at hcert
  exact hcert depth (List.mem_range.mpr hd)

lemma routeCoherentB_of_layers (d : Graph)
    (h :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        tokensCoherentAtLayerB
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth) = true) :
    routeCoherentB d = true := by
  unfold routeCoherentB
  rw [List.all_eq_true]
  intro depth hmem
  exact h depth (List.mem_range.mp hmem)

lemma routeCoherentB_of_layer_props (d : Graph)
    (h :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth)) :
    routeCoherentB d = true := by
  apply routeCoherentB_of_layers
  intro depth hd
  have hlen :
      ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] }).incoming.length =
        ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] }).nodes.length := by
    have hgetD :
        (buildGridFromDLDS d).getD depth { nodes := [], incoming := [] } =
          (buildGridFromDLDS d).get ⟨depth, hd⟩ := by
      rw [List.get_eq_getElem]
      exact (List.getElem_eq_getD _).symm
    have hmem : (buildGridFromDLDS d).getD depth { nodes := [], incoming := [] } ∈
        buildGridFromDLDS d := by
      rw [hgetD]
      exact List.get_mem _ _
    have hwf := (buildGridFromDLDS_wellformed d).2 _ hmem
    exact hwf.2.1.trans hwf.1.symm
  exact tokensCoherentAtLayer_Prop_implies_B _ _ hlen (h depth hd)

lemma descent_coherent_base (d : Graph) (hvalid : ValidDLDS d) :
    TokensCoherentAtLayer
      ((buildGridFromDLDS d).getD 0 { nodes := [], incoming := [] })
      (descentTokens d 0) := by
  simpa [descentTokens] using (coherent_base_exact d hvalid).coherent

/--
 Induction assembly for `descent_coherent`: once the structural one-step
    coherence theorem is proved, all descent layers are coherent.
-/
lemma descent_coherent_of_step (d : Graph) (hvalid : ValidDLDS d)
    (hstep :
      ∀ depth,
        depth + 1 < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth) →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD (depth + 1) { nodes := [], incoming := [] })
          (descentTokens d (depth + 1))) :
    ∀ depth, depth < (buildGridFromDLDS d).length →
      TokensCoherentAtLayer
        ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
        (descentTokens d depth) := by
  intro depth hd
  induction depth with
  | zero =>
      exact descent_coherent_base d hvalid
  | succ depth ih =>
      exact hstep depth hd (ih (by omega))

/--
 **Per-slot uniqueness** (coherence propagation step), discharged from the
    decidable certificate `routeCoherentB`.

    Given the descent tokens at depth `i` (`htok`), one `propagate_tokens` step lands
    exactly the descent tokens at depth `i+1` (definitional), whose layer coherence
    is asserted by the certificate and transported `Bool → Prop` by
    `tokensCoherentAtLayerB_implies_Prop`.
-/
lemma unique_token_per_slot (d : Graph) (hcert : routeCoherentB d = true)
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (htok : tokens = descentTokens d i)
    (hj : i + 1 < (buildGridFromDLDS d).length) :
    TokensCoherentAtLayer
      ((buildGridFromDLDS d).get ⟨i + 1, hj⟩)
      (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length
        (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1) := by
  have hgetD_i : (buildGridFromDLDS d).get ⟨i, hi⟩ =
      (buildGridFromDLDS d).getD i { nodes := [], incoming := [] } := by
    rw [List.get_eq_getElem]; exact List.getElem_eq_getD _
  have hstep : propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length
      (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1 = descentTokens d (i + 1) := by
    subst htok
    rw [hgetD_i]
    rw [show level = (buildGridFromDLDS d).length - 1 - i from hlvl]
    rfl
  rw [hstep]
  have hB := routeCoherentB_layer d hcert (i + 1) hj
  have hgetD_j : (buildGridFromDLDS d).getD (i + 1) { nodes := [], incoming := [] } =
      (buildGridFromDLDS d).get ⟨i + 1, hj⟩ := by
    rw [List.get_eq_getElem]; exact (List.getElem_eq_getD _).symm
  rw [hgetD_j] at hB
  exact tokensCoherentAtLayerB_implies_Prop _ _ hB

/--
 Certificate-free per-slot step, assuming the structural descent-coherence
    theorem directly rather than `routeCoherentB`. This is the downstream shape
    used once `descent_coherent` is proved.
-/
lemma unique_token_per_slot_of_descent_coherent (d : Graph)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth))
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (htok : tokens = descentTokens d i)
    (hj : i + 1 < (buildGridFromDLDS d).length) :
    TokensCoherentAtLayer
      ((buildGridFromDLDS d).get ⟨i + 1, hj⟩)
      (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length
        (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1) := by
  have hgetD_i : (buildGridFromDLDS d).get ⟨i, hi⟩ =
      (buildGridFromDLDS d).getD i { nodes := [], incoming := [] } := by
    rw [List.get_eq_getElem]; exact List.getElem_eq_getD _
  have hstep : propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length
      (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1 = descentTokens d (i + 1) := by
    subst htok
    rw [hgetD_i]
    rw [show level = (buildGridFromDLDS d).length - 1 - i from hlvl]
    rfl
  rw [hstep]
  have hgetD_j : (buildGridFromDLDS d).getD (i + 1) { nodes := [], incoming := [] } =
      (buildGridFromDLDS d).get ⟨i + 1, hj⟩ := by
    rw [List.get_eq_getElem]; exact (List.getElem_eq_getD _).symm
  rw [← hgetD_j]
  exact hdesc (i + 1) hj

/--
 Exact step: composes (a) well-routed half, (b) per-slot coherence from the
    certificate, (c) exact presence.
-/
lemma coherent_step_exact (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true)
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (htok : tokens = descentTokens d i)
    (hexact : TokensExactAtLayer d i ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens) :
    let outs := (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1
    ∀ (hj : i + 1 < (buildGridFromDLDS d).length),
      TokensExactAtLayer d (i + 1)
        ((buildGridFromDLDS d).get ⟨i + 1, hj⟩)
        (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length outs) := by
  dsimp only
  intro hj
  refine ⟨⟨?_, ?_⟩, ?_⟩
  ·
    exact coherent_step_wellRouted d htree hvalid i level hi hlvl tokens hexact.1 hj
  ·
    exact unique_token_per_slot d hcert i level hi hlvl tokens htok hj
  ·
    have houts_len : (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1.length =
        (buildFormulas d).length := by
      have := evaluate_layer_outputs_length ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens
      rwa [buildGridFromDLDS_get_nodes_length] at this
    exact tokensPresence_step d htree hvalid i level hj hlvl tokens hexact.2 hexact.1.1 _ houts_len

/--  Exact step with the certificate replaced by structural descent coherence.  -/
lemma coherent_step_exact_of_descent_coherent
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth))
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (htok : tokens = descentTokens d i)
    (hexact : TokensExactAtLayer d i ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens) :
    let outs := (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1
    ∀ (hj : i + 1 < (buildGridFromDLDS d).length),
      TokensExactAtLayer d (i + 1)
        ((buildGridFromDLDS d).get ⟨i + 1, hj⟩)
        (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length outs) := by
  dsimp only
  intro hj
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · exact coherent_step_wellRouted d htree hvalid i level hi hlvl tokens hexact.1 hj
  · exact unique_token_per_slot_of_descent_coherent d hdesc
      i level hi hlvl tokens htok hj
  · have houts_len : (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1.length =
        (buildFormulas d).length := by
      have := evaluate_layer_outputs_length ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens
      rwa [buildGridFromDLDS_get_nodes_length] at this
    exact tokensPresence_step d htree hvalid i level hj hlvl tokens hexact.2 hexact.1.1 _ houts_len

/--  Exact propagation step parameterized by next-layer coherence.  -/
lemma coherent_step_exact_of_next_coherence
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (i level : Nat) (hi : i < (buildGridFromDLDS d).length)
    (hlvl : level = (buildGridFromDLDS d).length - 1 - i)
    (tokens : List (Token (buildFormulas d).length))
    (_htok : tokens = descentTokens d i)
    (hexact : TokensExactAtLayer d i ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens)
    (hj : i + 1 < (buildGridFromDLDS d).length)
    (hnextCoh :
      TokensCoherentAtLayer
        ((buildGridFromDLDS d).get ⟨i + 1, hj⟩)
        (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length
          (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1)) :
    let outs := (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1
    TokensExactAtLayer d (i + 1)
      ((buildGridFromDLDS d).get ⟨i + 1, hj⟩)
      (propagate_tokens tokens (pathsFromDLDS d) level (buildGridFromDLDS d).length outs) := by
  dsimp only
  refine ⟨⟨?_, hnextCoh⟩, ?_⟩
  · exact coherent_step_wellRouted d htree hvalid i level hi hlvl tokens hexact.1 hj
  · have houts_len : (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).1.length =
        (buildFormulas d).length := by
      have := evaluate_layer_outputs_length ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens
      rwa [buildGridFromDLDS_get_nodes_length] at this
    exact tokensPresence_step d htree hvalid i level hj hlvl tokens hexact.2 hexact.1.1 _ houts_len

-- A formula cannot equal an implication that has it as the left sub-formula.
private lemma formula_implication_ne_left (a b : Formula) : Formula.implication a b ≠ a := by
  intro h
  induction a generalizing b with
  | atom _ => exact Formula.noConfusion h
  | implication a1 a2 ih1 _ =>
      exact ih1 a2 (Formula.implication.inj h).1

/--
 Route fact for the major-carrier invariant: at a destination `⊃E`, the
    major premise is not the minor, so the carrier tail follows the conclusion
    route from the elimination node.
-/
lemma routeFrom_major_tail_continues
    (d : Graph) (formulas : List Formula) (fuel : Nat)
    (φ : Formula) (v : Vertex) (e : Deduction) (es : List Deduction)
    (major minor : Deduction)
    (hfind : d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some v)
    (hout : get_rule.outgoing v d = e :: es)
    (hclass : classifyRule? e.END d = some (DLDSRuleClass.elim major minor))
    (hmajor : φ = major.START.FORMULA) :
    routeFrom d formulas (Nat.succ fuel) φ =
      (formulas.idxOf e.END.FORMULA + 1, inputLabelForEdge d formulas φ e.END) ::
        routeFrom d formulas fuel e.END.FORMULA := by
  apply routeFrom_nonminor_tail_continues d formulas fuel φ v e es hfind hout
  simp [hclass]
  intro hminorEq
  have hmaj_shape :
      major.START.FORMULA = Formula.implication minor.START.FORMULA e.END.FORMULA :=
    classifyRule?_elim_major_formula_eq_minor hclass
  rw [hmajor, hmaj_shape] at hminorEq
  exact formula_implication_ne_left minor.START.FORMULA e.END.FORMULA hminorEq

/--
 At an `⊃E` node, the major premise uses slot 0 and the minor premise uses
    slot 1.
-/
lemma slotForEdge_major_minor_distinct (d : Graph)
    (w : Vertex) (major minor : Deduction)
    (hclass : classifyRule? w d = some (DLDSRuleClass.elim major minor)) :
    slotForEdge major.START.FORMULA w d = 0 ∧
    slotForEdge minor.START.FORMULA w d = 1 := by
  obtain ⟨_, A, hmaj⟩ := classifyRule?_elim_major_mem_incoming hclass
  have hminor : minor.START.FORMULA = A := by
    unfold classifyRule? at hclass
    by_cases hhyp : w.HYPOTHESIS = true
    · simp [hhyp] at hclass
    · simp [hhyp] at hclass
      cases hinc : get_rule.incoming w d with
      | nil => simp [hinc] at hclass
      | cons e es =>
          cases es with
          | nil => simp [hinc] at hclass
          | cons e2 es2 =>
              cases es2 with
              | nil =>
                  simp [hinc] at hclass
                  split_ifs at hclass with h1 h2
                  · obtain ⟨rfl, rfl⟩ := DLDSRuleClass.elim.inj (Option.some.inj hclass)
                    rw [hmaj] at h1; exact (Formula.implication.inj h1).1.symm
                  · obtain ⟨rfl, rfl⟩ := DLDSRuleClass.elim.inj (Option.some.inj hclass)
                    rw [hmaj] at h2; exact (Formula.implication.inj h2).1.symm
              | cons _ _ => simp [hinc] at hclass
  constructor
  ·
    simp only [slotForEdge, hclass]
    rw [hmaj, hminor]
    exact if_neg (formula_implication_ne_left A w.FORMULA)
  ·
    simp only [slotForEdge, hclass, hminor, ↓reduceIte]

lemma inputLabelForEdge_decodes_intro_premise
    (d : Graph) (w : Vertex) (p : Deduction) (ruleIdx : Nat)
    (hclass : classifyRule? w d = some (DLDSRuleClass.intro p))
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some ruleIdx) :
    decodeInputLabel
      (buildIncomingMapForFormula (buildFormulas d) w.FORMULA)
      (inputLabelForEdge d (buildFormulas d) p.START.FORMULA w) =
        some (ruleIdx, 0, (buildFormulas d).idxOf p.START.FORMULA) := by
  apply inputLabelForEdge_decodes_of_classified_slot
      (d := d) (formulas := buildFormulas d) (φ := p.START.FORMULA)
      (w := w) (ruleIdx := ruleIdx) (slot := 0)
      (srcs := [(buildFormulas d).idxOf p.START.FORMULA])
  · exact hsel
  · exact dlds_incoming_matches_rule_premises d w ruleIdx hsel
  · simp [classifiedRuleSourceColumns?, hclass]
  · rfl
  · simp [slotForEdge, hclass]

lemma inputLabelForEdge_decodes_elim_major
    (d : Graph) (w : Vertex) (major minor : Deduction) (ruleIdx : Nat)
    (hclass : classifyRule? w d = some (DLDSRuleClass.elim major minor))
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some ruleIdx) :
    decodeInputLabel
      (buildIncomingMapForFormula (buildFormulas d) w.FORMULA)
      (inputLabelForEdge d (buildFormulas d) major.START.FORMULA w) =
        some (ruleIdx, 0, (buildFormulas d).idxOf major.START.FORMULA) := by
  have hslots := slotForEdge_major_minor_distinct d w major minor hclass
  apply inputLabelForEdge_decodes_of_classified_slot
      (d := d) (formulas := buildFormulas d) (φ := major.START.FORMULA)
      (w := w) (ruleIdx := ruleIdx) (slot := 0)
      (srcs := [(buildFormulas d).idxOf major.START.FORMULA,
        (buildFormulas d).idxOf minor.START.FORMULA])
  · exact hsel
  · exact dlds_incoming_matches_rule_premises d w ruleIdx hsel
  · simp [classifiedRuleSourceColumns?, hclass]
  · rfl
  · exact hslots.1

lemma inputLabelForEdge_decodes_elim_minor
    (d : Graph) (w : Vertex) (major minor : Deduction) (ruleIdx : Nat)
    (hclass : classifyRule? w d = some (DLDSRuleClass.elim major minor))
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some ruleIdx) :
    decodeInputLabel
      (buildIncomingMapForFormula (buildFormulas d) w.FORMULA)
      (inputLabelForEdge d (buildFormulas d) minor.START.FORMULA w) =
        some (ruleIdx, 1, (buildFormulas d).idxOf minor.START.FORMULA) := by
  have hslots := slotForEdge_major_minor_distinct d w major minor hclass
  apply inputLabelForEdge_decodes_of_classified_slot
      (d := d) (formulas := buildFormulas d) (φ := minor.START.FORMULA)
      (w := w) (ruleIdx := ruleIdx) (slot := 1)
      (srcs := [(buildFormulas d).idxOf major.START.FORMULA,
        (buildFormulas d).idxOf minor.START.FORMULA])
  · exact hsel
  · exact dlds_incoming_matches_rule_premises d w ruleIdx hsel
  · simp [classifiedRuleSourceColumns?, hclass]
  · rfl
  · exact hslots.2

/--  Generic projection: `TokensExactAlong → TokensMatchAlong`.  -/
lemma exactAlong_implies_matchAlong_aux (d : Graph) (nl : Nat) :
    ∀ (layers : List (GridLayer (buildFormulas d).length)) (depth level : Nat)
      (tokens : List (Token (buildFormulas d).length)),
      TokensExactAlong d nl depth level tokens layers →
      TokensMatchAlong (pathsFromDLDS d) nl level tokens layers := by
  intro layers
  induction layers with
  | nil => intro depth level tokens _; trivial
  | cons layer rest ih =>
      intro depth level tokens hexact
      obtain ⟨hlayer, hrest⟩ := hexact
      exact ⟨hlayer.coherent, ih (depth + 1) (level - 1) _ hrest⟩

private lemma List.tail_drop_eq_drop_succ {α : Type*} (xs : List α) (i : Nat) :
    (xs.drop i).tail = xs.drop (i + 1) := by
  induction xs generalizing i with
  | nil =>
      cases i <;> simp
  | cons x xs ih =>
      cases i with
      | zero => simp
      | succ i =>
          simp [Nat.add_assoc]

private lemma List.get_zero_eq_getD_zero {α : Type*}
    (xs : List α) (default : α) (h : 0 < xs.length) :
    xs.get ⟨0, h⟩ = xs.getD 0 default := by
  cases xs with
  | nil => simp at h
  | cons x xs => rfl

/--  Exactness along the evaluator descent.  -/
lemma tokens_exact_along (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true) :
    TokensExactAlong d ((buildGridFromDLDS d).length)
      0
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  let grid := buildGridFromDLDS d
  let nl := grid.length
  have hbase :
      TokensExactAtLayer d 0
        (grid.getD 0 { nodes := [], incoming := [] })
        (initialize_tokens (initialVectorsFromDLDS d) nl) := by
    simpa [grid, nl] using coherent_base_exact d hvalid
  have hstep :
      ∀ (idx level : Nat) (hi : idx < grid.length)
        (tokens : List (Token (buildFormulas d).length)),
        level = grid.length - 1 - idx →
        tokens = descentTokens d idx →
        TokensExactAtLayer d idx (grid.get ⟨idx, hi⟩) tokens →
        TokensExactAlong d grid.length idx level tokens (grid.drop idx) := by
    intro idx
    induction hrem : grid.drop idx generalizing idx with
    | nil =>
        intro level hi tokens hlvl htok hexact
        simp [TokensExactAlong]
    | cons layer rest ih =>
        intro level hi tokens hlvl htok hexact
        have hhead : layer = grid.get ⟨idx, hi⟩ := by
          have hget? : (grid.drop idx)[0]? = grid[idx]? := by
            simp
          have hidx? : grid[idx]? = some (grid.get ⟨idx, hi⟩) := by
            simp [List.get_eq_getElem]
          have hhead? : (grid.drop idx)[0]? = some layer := by
            simp [hrem]
          rw [hget?, hidx?] at hhead?
          exact (Option.some.inj hhead?).symm
        subst layer
        simp [TokensExactAlong]
        refine ⟨hexact, ?_⟩
        by_cases hj : idx + 1 < grid.length
        · have hnextExact :=
            coherent_step_exact d htree hvalid hcert idx level hi hlvl tokens htok hexact hj
          have hdrop_next : rest = grid.drop (idx + 1) := by
            have htail : (grid.drop idx).tail = rest := by simp [hrem]
            exact htail.symm.trans (List.tail_drop_eq_drop_succ grid idx)
          have hnext_tok :
              propagate_tokens tokens (pathsFromDLDS d) level grid.length
                (evaluate_layer (grid.get ⟨idx, hi⟩) tokens).1 = descentTokens d (idx + 1) := by
            subst htok
            have hgetD : grid.get ⟨idx, hi⟩ =
                grid.getD idx { nodes := [], incoming := [] } := by
              rw [List.get_eq_getElem]; exact List.getElem_eq_getD _
            rw [hgetD, show level = grid.length - 1 - idx from hlvl]
            rfl
          exact ih (idx + 1) hdrop_next.symm (level - 1) hj
            (propagate_tokens tokens (pathsFromDLDS d) level grid.length
              (evaluate_layer (grid.get ⟨idx, hi⟩) tokens).1)
            (by omega) hnext_tok hnextExact
        · have hdrop_end : rest = [] := by
            have hidx_last : idx + 1 = grid.length := by omega
            have hdrop_succ_empty : grid.drop (idx + 1) = [] := by
              simp [hidx_last]
            have htail : (grid.drop idx).tail = rest := by simp [hrem]
            exact htail.symm.trans
              ((List.tail_drop_eq_drop_succ grid idx).trans hdrop_succ_empty)
          simp [hdrop_end, TokensExactAlong]
  have hgrid_nonempty : 0 < grid.length := by
    unfold grid buildGridFromDLDS buildLayers
    simp
  have hheadExact :
      TokensExactAtLayer d 0 (grid.get ⟨0, hgrid_nonempty⟩)
        (initialize_tokens (initialVectorsFromDLDS d) nl) := by
    have hgetD :
        grid.get ⟨0, hgrid_nonempty⟩ =
          grid.getD 0 { nodes := [], incoming := [] } := by
      exact List.get_zero_eq_getD_zero grid { nodes := [], incoming := [] } hgrid_nonempty
    rw [hgetD]
    exact hbase
  have htok0 :
      initialize_tokens (initialVectorsFromDLDS d) nl = descentTokens d 0 := by
    rfl
  simpa [grid, nl] using
    hstep 0 (nl - 1) hgrid_nonempty
      (initialize_tokens (initialVectorsFromDLDS d) nl) rfl htok0 hheadExact

/--
 `tokens_exact_along` with the route-coherence certificate derived from the
    structural descent-coherence theorem, not assumed.
-/
lemma tokens_exact_along_of_descent_coherent (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth)) :
    TokensExactAlong d ((buildGridFromDLDS d).length)
      0
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  exact tokens_exact_along d htree hvalid (routeCoherentB_of_layer_props d hdesc)

/--  Exactness implies the node-match invariant.  -/
lemma tokens_exact_implies_match_along (d : Graph) :
    TokensExactAlong d ((buildGridFromDLDS d).length)
      0
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) →
    TokensMatchAlong (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  intro h
  exact exactAlong_implies_matchAlong_aux d _ (buildGridFromDLDS d) 0 _ _ h

/--
 Along the evaluator descent induced by `pathsFromDLDS d`, every nonempty
    token group currently present at a single grid column matches one decoded
    rule and fills that rule's slots exactly once; empty columns are permitted.
    This intentionally covers padding/repetition layers as well as logical DLDS
    node levels, since repeated tokens are still evaluated by grid nodes.

    Assembly: `coherent_base` gives layer 0; `coherent_step` (+ induction)
    carries coherence down every layer; wrapping into `TokensMatchAlong`'s
    recursive shape closes the goal.
-/
lemma tokens_at_node_match_one_rule (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true) :
    TokensMatchAlong (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) :=
  tokens_exact_implies_match_along d (tokens_exact_along d htree hvalid hcert)

lemma tokens_at_node_match_one_rule_of_descent_coherent (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth)) :
    TokensMatchAlong (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) :=
  tokens_exact_implies_match_along d
    (tokens_exact_along_of_descent_coherent d htree hvalid hdesc)

/--
 Bridge 1: `nodeError = false` on a token group implies the `evaluate_node` error
    flag is also `false`.  Follows because the `else if nodeError` branch is bypassed
    and `selectedRuleIndex?` succeeds (decode can't fail when nodeError is false).
-/
private lemma evaluate_node_snd_false_of_noError {n : Nat}
    (cnode : CircuitNode n) (incoming : NodeIncoming) (toks : List (Token n))
    (h : nodeError cnode incoming toks = false) :
    (evaluate_node cnode incoming toks).2 = false := by
  simp only [evaluate_node]
  cases hemp : toks.isEmpty
  ·
    simp only [Bool.false_eq_true, ↓reduceIte, h]
    suffices hs : ∃ r, selectedRuleIndex? incoming toks = some r by
      obtain ⟨r, hr⟩ := hs; simp [hr]
    cases toks with
    | nil => simp at hemp
    | cons t ts =>
        simp only [selectedRuleIndex?]
        unfold nodeError at h
        simp at h
        cases hdec : decodeInputLabel incoming t.input_label with
        | none => simp [hdec] at h
        | some triple => exact ⟨triple.1, by simp⟩
  · simp   -- isEmpty = true → snd = false immediately

/--
 Bridge 2: when all columns of a DLDS grid layer have `nodeError = false`,
    `evaluate_layer` reports no error.
-/
private lemma evaluate_layer_snd_false_of_noError_DLDS (d : Graph)
    (i : Nat) (hi : i < (buildGridFromDLDS d).length)
    (tokens : List (Token (buildFormulas d).length))
    (hnoerr : ∀ col (hN : col < ((buildGridFromDLDS d).get ⟨i, hi⟩).nodes.length)
              (hI : col < ((buildGridFromDLDS d).get ⟨i, hi⟩).incoming.length),
      nodeError (((buildGridFromDLDS d).get ⟨i, hi⟩).nodes.get ⟨col, hN⟩)
        (((buildGridFromDLDS d).get ⟨i, hi⟩).incoming.get ⟨col, hI⟩)
        (tokens.filter (fun t => t.current_column = col)) = false) :
    (evaluate_layer ((buildGridFromDLDS d).get ⟨i, hi⟩) tokens).2 = false := by
  simp only [evaluate_layer, List.any_eq_false]
  intro b hb
  rw [List.mem_map] at hb
  obtain ⟨res, hres_mem, rfl⟩ := hb
  simp only [List.mem_map] at hres_mem
  obtain ⟨⟨cnode, col_idx⟩, hzip, rfl⟩ := hres_mem
  simp
  have hzip_info := List.mem_zipIdx hzip
  obtain ⟨_, hcol_N_raw, helem⟩ := hzip_info
  simp only [Nat.zero_add] at hcol_N_raw
  have hcol_N : col_idx < ((buildGridFromDLDS d).get ⟨i, hi⟩).nodes.length := hcol_N_raw
  have hcol_I : col_idx < ((buildGridFromDLDS d).get ⟨i, hi⟩).incoming.length := by
    rw [buildGridFromDLDS_get_incoming_length]; rwa [buildGridFromDLDS_get_nodes_length] at hcol_N
  have hcnode : cnode = ((buildGridFromDLDS d).get ⟨i, hi⟩).nodes.get ⟨col_idx, hcol_N⟩ := by
    simp only [List.get_eq_getElem]
    simpa [Nat.sub_zero, List.getElem_eq_get] using helem
  have hnodeErr := hnoerr col_idx hcol_N hcol_I
  have hnodeErr' :
      nodeError cnode
        ((((buildGridFromDLDS d).get ⟨i, hi⟩).incoming[col_idx]?).getD default)
        (tokens.filter (fun t => t.current_column = col_idx)) = false := by
    have hincoming_getD :
        ((((buildGridFromDLDS d).get ⟨i, hi⟩).incoming[col_idx]?).getD default) =
          ((buildGridFromDLDS d).get ⟨i, hi⟩).incoming.get ⟨col_idx, hcol_I⟩ := by
      rw [List.getElem?_eq_getElem hcol_I]
      rfl
    rw [hcnode]
    rw [hincoming_getD]
    exact hnodeErr
  have hev := evaluate_node_snd_false_of_noError cnode
    ((((buildGridFromDLDS d).get ⟨i, hi⟩).incoming[col_idx]?).getD default)
    (tokens.filter (fun t => t.current_column = col_idx)) hnodeErr'
  simpa [List.getElem!_eq_getElem?_getD, Bool.not_eq_true] using hev

/--
 General layer version of the previous bridge, parameterized by the only
    shape fact `evaluate_layer` needs: every node column has an incoming entry.
-/
private lemma evaluate_layer_snd_false_of_noError {n : Nat}
    (layer : GridLayer n)
    (tokens : List (Token n))
    (hlen : layer.incoming.length = layer.nodes.length)
    (hnoerr : ∀ col (hN : col < layer.nodes.length) (hI : col < layer.incoming.length),
      nodeError (layer.nodes.get ⟨col, hN⟩)
        (layer.incoming.get ⟨col, hI⟩)
        (tokens.filter (fun t => t.current_column = col)) = false) :
    (evaluate_layer layer tokens).2 = false := by
  simp only [evaluate_layer, List.any_eq_false]
  intro b hb
  rw [List.mem_map] at hb
  obtain ⟨res, hres_mem, rfl⟩ := hb
  simp only [List.mem_map] at hres_mem
  obtain ⟨⟨cnode, col_idx⟩, hzip, rfl⟩ := hres_mem
  simp
  have hzip_info := List.mem_zipIdx hzip
  obtain ⟨_, hcol_N_raw, helem⟩ := hzip_info
  simp only [Nat.zero_add] at hcol_N_raw
  have hcol_N : col_idx < layer.nodes.length := hcol_N_raw
  have hcol_I : col_idx < layer.incoming.length := by
    rwa [hlen]
  have hcnode : cnode = layer.nodes.get ⟨col_idx, hcol_N⟩ := by
    simp only [List.get_eq_getElem]
    simpa [Nat.sub_zero, List.getElem_eq_get] using helem
  have hnodeErr := hnoerr col_idx hcol_N hcol_I
  have hnodeErr' :
      nodeError cnode ((layer.incoming[col_idx]?).getD default)
        (tokens.filter (fun t => t.current_column = col_idx)) = false := by
    have hincoming_getD :
        ((layer.incoming[col_idx]?).getD default) =
          layer.incoming.get ⟨col_idx, hcol_I⟩ := by
      rw [List.getElem?_eq_getElem hcol_I]
      rfl
    rw [hcnode]
    rw [hincoming_getD]
    exact hnodeErr
  have hev := evaluate_node_snd_false_of_noError cnode
    ((layer.incoming[col_idx]?).getD default)
    (tokens.filter (fun t => t.current_column = col_idx)) hnodeErr'
  simpa [List.getElem!_eq_getElem?_getD, Bool.not_eq_true] using hev

/--
 General: `NodeErrorFalseAlong` follows from `TokensMatchAlong` by
    `coherentLayer_implies_noError` at each step.
-/
private lemma NodeErrorFalseAlong_of_TokensMatchAlong {n : Nat}
    (paths : PathInput) (nl : Nat) :
    ∀ level tokens (layers : List (GridLayer n)),
    TokensMatchAlong paths nl level tokens layers →
    NodeErrorFalseAlong paths nl level tokens layers := by
  intro level tokens layers hmatch
  induction layers generalizing level tokens with
  | nil => trivial
  | cons layer rest ih =>
      obtain ⟨hcoh, hrest⟩ := hmatch
      exact ⟨coherentLayer_implies_noError layer tokens hcoh, ih _ _ hrest⟩

/--
 Bridge 3: `TokensMatchAlong` ⟹ `NoLayerError`.
    Connection: `TokensCoherentAtLayer` → `nodeError = false` (`coherentLayer_implies_noError`)
    → `evaluate_layer.snd = false` (`evaluate_layer_snd_false_of_noError`, which needs the
    per-layer `incoming.length = nodes.length` hypothesis `hlen` to align `incoming[col]!`
    with `.get ⟨col, _⟩`). Induction mirrors the `NoLayerError` recursion.
-/
private lemma NoLayerError_of_TokensMatchAlong_DLDS (d : Graph) :
    ∀ (level : Nat) (tokens : List (Token (buildFormulas d).length))
      (layers : List (GridLayer (buildFormulas d).length)),
    (∀ layer ∈ layers, layer.incoming.length = layer.nodes.length) →
    TokensMatchAlong (pathsFromDLDS d) ((buildGridFromDLDS d).length) level tokens layers →
    NoLayerError (pathsFromDLDS d) ((buildGridFromDLDS d).length) level tokens layers := by
  intro level tokens layers hlen hmatch
  induction layers generalizing level tokens with
  | nil => trivial
  | cons layer rest ih =>
      obtain ⟨hcoh, hrest⟩ := hmatch
      have hhead_len : layer.incoming.length = layer.nodes.length := by
        exact hlen layer (by simp)
      have htail_len : ∀ l ∈ rest, l.incoming.length = l.nodes.length := by
        intro l hl
        exact hlen l (by simp [hl])
      refine ⟨?_, ih _ _ htail_len hrest⟩
      exact evaluate_layer_snd_false_of_noError layer tokens hhead_len
        (coherentLayer_implies_noError layer tokens hcoh)

/--  Combining the bridges: `TokensMatchAlong` ⟹ `NodeErrorFalseAlong ∧ NoLayerError`.  -/
lemma nodeError_false_along_descent (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true) :
    NodeErrorFalseAlong (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) ∧
    NoLayerError (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  have hmatch := tokens_at_node_match_one_rule d htree hvalid hcert
  have hlen : ∀ layer ∈ buildGridFromDLDS d, layer.incoming.length = layer.nodes.length := by
    intro layer hmem
    have hwf := (buildGridFromDLDS_wellformed d).2 layer hmem
    exact hwf.2.1.trans hwf.1.symm
  exact ⟨NodeErrorFalseAlong_of_TokensMatchAlong _ _ _ _ _ hmatch,
         NoLayerError_of_TokensMatchAlong_DLDS d _ _ _ hlen hmatch⟩

lemma nodeError_false_along_descent_of_descent_coherent (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth)) :
    NodeErrorFalseAlong (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) ∧
    NoLayerError (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  exact nodeError_false_along_descent d htree hvalid
    (routeCoherentB_of_layer_props d hdesc)

/--
 **Per-evaluation no-conflict**. With semantics (c) the
    detector is real again, so this is no longer a one-line consequence of
    one-hot activation.
-/
lemma tree_routing_unique (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true) :
    NoLayerError (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  exact (nodeError_false_along_descent d htree hvalid hcert).2

lemma tree_routing_unique_of_descent_coherent (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth)) :
    NoLayerError (pathsFromDLDS d) ((buildGridFromDLDS d).length)
      ((buildGridFromDLDS d).length - 1)
      (initialize_tokens (initialVectorsFromDLDS d) ((buildGridFromDLDS d).length))
      (buildGridFromDLDS d) := by
  exact (nodeError_false_along_descent_of_descent_coherent d htree hvalid hdesc).2

/--
 **Structural sublemma**: the DLDS-derived path on a simple tree produces no
    routing conflict, so `had_error = false`. Reduced to `tree_routing_unique` via
    the glue lemma.
-/
lemma no_routing_error (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true) :
    PathHasNoRoutingError (pathsFromDLDS d) (buildGridFromDLDS d)
      (initialVectorsFromDLDS d) := by
  unfold PathHasNoRoutingError PathStructurallyInvalid
  have hfalse :
      (get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
      (pathsFromDLDS d)).snd = false := by
    unfold get_eval_result
    apply eval_from_level_snd_false_of_NoLayerError
    · rfl
    · exact tree_routing_unique d htree hvalid hcert
  rw [hfalse]; simp

lemma no_routing_error_of_descent_coherent (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth)) :
    PathHasNoRoutingError (pathsFromDLDS d) (buildGridFromDLDS d)
      (initialVectorsFromDLDS d) := by
  exact no_routing_error d htree hvalid (routeCoherentB_of_layer_props d hdesc)

/--  Executable check that the evaluated goal vector is all-false.  -/
def dischargedB (d : Graph) : Bool :=
  match (get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
          (pathsFromDLDS d)).1[goalColumn d]? with
  | none => true
  | some v => v.toList.all (fun b => !b)

/--  The executable discharge check implies semantic discharge.  -/
lemma discharge (d : Graph) (_htree : IsTreeDLDS d) (_hvalid : ValidDLDS d)
    (hdis : dischargedB d = true) :
    AllAssumptionsDischarged (pathsFromDLDS d) (buildGridFromDLDS d)
      (initialVectorsFromDLDS d) (goalColumn d) := by
  unfold dischargedB at hdis
  by_cases hg : goalColumn d <
      (get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d) (pathsFromDLDS d)).1.length
  · refine Or.inr ⟨hg, fun i => ?_⟩
    have hsome :
        (get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d) (pathsFromDLDS d)).1[goalColumn d]?
          = some ((get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
              (pathsFromDLDS d)).1.get ⟨goalColumn d, hg⟩) := by
      simp [List.get_eq_getElem]
    rw [hsome] at hdis
    rw [List.all_eq_true] at hdis
    have hmem : ((get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
        (pathsFromDLDS d)).1.get ⟨goalColumn d, hg⟩).get i ∈
        ((get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d)
        (pathsFromDLDS d)).1.get ⟨goalColumn d, hg⟩).toList := by
      exact List.Vector.get_mem i _
    have hb := hdis _ hmem
    simpa using hb
  · exact Or.inl (not_lt.mp hg)

/--
 **Tree bridge (forward)**: a valid tree DLDS yields genuine circuit acceptance
    of its extracted path. In particular this implies
    `evaluateDLDS d (pathsFromDLDS d) (goalColumn d) = true` (via
    `dlds_evaluation_complete`), i.e. the DLDS-derived path is a genuine,
    non-erroring witness of the paper's global Accept ; not a vacuous / conflict
    acceptance.
-/
theorem tree_bridge_forward (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hcert : routeCoherentB d = true) (hdis : dischargedB d = true) :
    GenuinelyAccepts d (pathsFromDLDS d) (goalColumn d) :=
  ⟨no_routing_error d htree hvalid hcert, discharge d htree.1 hvalid hdis⟩

/--
 Forward bridge with route coherence supplied structurally (`hdesc`) rather
    than by the `routeCoherentB` certificate.  The discharge certificate remains
    intentionally unchanged for the later discharge phase.
-/
theorem tree_bridge_forward_of_descent_coherent
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdesc :
      ∀ depth, depth < (buildGridFromDLDS d).length →
        TokensCoherentAtLayer
          ((buildGridFromDLDS d).getD depth { nodes := [], incoming := [] })
          (descentTokens d depth))
    (hdis : dischargedB d = true) :
    GenuinelyAccepts d (pathsFromDLDS d) (goalColumn d) :=
  ⟨no_routing_error_of_descent_coherent d htree hvalid hdesc,
    discharge d htree.1 hvalid hdis⟩

/-!
The forward bridge `tree_bridge_forward` currently uses two decidable
certificates: `routeCoherentB d = true` (per-slot carrier coherence) and
`dischargedB d = true` (goal vector all-false).
-/


/--  A hypothesis column is its own principal carrier.  -/
lemma principalCarrierForSourceColumn?_hyp (d : Graph)
    (col : Nat) (v : Vertex)
    (hsrc : sourceNodeAtColumn? d col = some v)
    (hcol : (buildFormulas d)[col]? = some v.FORMULA)
    (hhyp : classifyRule? v d = some DLDSRuleClass.hypothesis) :
    principalCarrierForSourceColumn? d col = some col := by
  unfold principalCarrierForSourceColumn?
  rw [hsrc]
  show principalCarrierColumn? d v = some col
  unfold principalCarrierColumn?
  rw [principalCarrierFormula?_hyp d ((d.NODES.map (·.LEVEL)).foldl max 0) v hhyp]
  simp only [Option.map_some]
  have hlt : col < (buildFormulas d).length := getElem?_some_lt hcol
  have hget : (buildFormulas d).get ⟨col, hlt⟩ = v.FORMULA := by
    rw [List.get_eq_getElem]
    exact (List.getElem?_eq_some_iff.mp hcol).2
  have hidx : (buildFormulas d).idxOf v.FORMULA = col :=
    indexOf_eq_of_get hlt (buildFormulas_nodup d) hget
  rw [hidx]

private lemma replicate_zero_get_nonzero_false
    {n i target label : Nat}
    (hget : (List.replicate n (0, 0) : List (Nat × Nat))[i]? =
      some (target, label))
    (htarget : target ≠ 0) : False := by
  rw [List.getElem?_replicate] at hget
  split at hget
  · have htarget0 : target = 0 := by
      exact (congrArg Prod.fst (Option.some.inj hget)).symm
    exact htarget htarget0
  · simp at hget

private lemma getElem?_append_right_sub {α : Type*}
    (xs ys : List α) {i : Nat} (h : xs.length ≤ i) :
    (xs ++ ys)[i]? = ys[i - xs.length]? := by
  induction xs generalizing i with
  | nil =>
      simp
  | cons x xs ih =>
      cases i with
      | zero =>
          simp at h
      | succ i =>
          have hle : xs.length ≤ i := by
            simpa [Nat.succ_le_succ_iff] using h
          simpa [Nat.succ_sub_succ_eq_sub] using ih hle

lemma getElem?_replicate_append_tail_at {α : Type*}
    (x : α) (xs : List α) (delay j : Nat) :
    (List.replicate delay x ++ xs)[delay + j]? = xs[j]? := by
  have hle : (List.replicate delay x).length ≤ delay + j := by
    simp
  have hright :=
    getElem?_append_right_sub (List.replicate delay x) xs hle
  simpa using hright

lemma pathsFromDLDS_step_principal_zero (d : Graph)
    {origin col src lbl : Nat}
    (hstep :
      routeStateAfter (pathsFromDLDS d) origin 1 = some (col, src, lbl)) :
    principalCarrierForSourceColumn? d src = some origin := by
  obtain ⟨current, prevSource, prevLabel, steps, target, inputLabel,
    hprev, hsteps, hstep0, htarget, htriple⟩ :=
    routeStateAfter_live_succ (pathsFromDLDS d) (by simpa using hstep)
  cases htriple
  by_cases hpath : origin < (pathsFromDLDS d).length
  · simp [routeStateAfter, hpath] at hprev
    rcases hprev with ⟨rfl, _hprevSource, _hprevLabel⟩
    have horiginF : origin < (buildFormulas d).length := by
      simpa [pathsFromDLDS_length] using hpath
    let φ := (buildFormulas d).get ⟨origin, horiginF⟩
    have hφ : (buildFormulas d)[origin]? = some φ := by
      exact List.getElem?_eq_getElem horiginF
    have hentry := pathsFromDLDS_get?_of_formula d hφ
    rw [hentry] at hsteps
    cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
    | none =>
        have hsteps_eq :
            List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) = steps := by
          simpa [hfind] using Option.some.inj hsteps
        exact False.elim
          (replicate_zero_get_nonzero_false
            (by simpa [← hsteps_eq] using hstep0) htarget)
    | some v =>
        have hvφ : v.FORMULA = φ := by
          exact of_decide_eq_true
            (List.find?_some
              (p := fun u => decide (u.FORMULA = φ))
              (l := d.NODES) hfind)
        by_cases hhyp : v.HYPOTHESIS = true
        · have hsrc :
              sourceNodeAtColumn? d origin = some v := by
            unfold sourceNodeAtColumn?
            have hcol : (buildFormulas d)[origin]? = some v.FORMULA := by
              simpa [hvφ] using hφ
            rw [hcol]
            simpa [hvφ] using hfind
          have hcol : (buildFormulas d)[origin]? = some v.FORMULA := by
            simpa [hvφ] using hφ
          have hclass : classifyRule? v d = some DLDSRuleClass.hypothesis := by
            simp [classifyRule?, hhyp]
          exact principalCarrierForSourceColumn?_hyp d origin v hsrc hcol hclass
        · have hsteps_eq : steps =
              List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) := by
            have hrev :
                List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) = steps := by
              simpa [hfind, hhyp] using Option.some.inj hsteps
            exact hrev.symm
          exact False.elim
            (replicate_zero_get_nonzero_false
              (by simpa [hsteps_eq] using hstep0) htarget)
  · simp [routeStateAfter, hpath] at hprev

/--
 General delay/self-loop replay: a path whose entry begins with `D` copies of
    the self-loop step `(origin+1, 0)` keeps the route resting at its own column
    through the first `D` depths.
-/
lemma routeStateAfter_replicate_prefix_self
    (paths : PathInput) (origin D : Nat) (rest : List (Nat × Nat))
    (horigin : origin < paths.length)
    (hentry : paths[origin]? = some (List.replicate D (origin + 1, 0) ++ rest)) :
    ∀ k, k ≤ D → routeStateAfter paths origin k = some (origin, origin, 0) := by
  intro k
  induction k with
  | zero =>
      intro _
      simp [routeStateAfter, horigin]
  | succ k ih =>
      intro hk
      have hkD : k < D := by omega
      have hprev := ih (by omega)
      have hstepk :
          (List.replicate D (origin + 1, 0) ++ rest)[k]? = some (origin + 1, 0) := by
        rw [List.getElem?_append_left (by simpa using hkD), List.getElem?_replicate]
        simp [hkD]
      rw [show k + 1 = Nat.succ k from rfl]
      simp only [routeStateAfter, hprev, hentry, hstepk]
      simp

/--
 `principalCarrierFormula?` is monotone in fuel for `some` results: extra fuel
    never changes a settled principal-carrier formula. Used to reconcile the
    `d.NODES.length + 1` fuel of `principalCarrierColumn?` across one spine step.
-/
lemma principalCarrierFormula?_fuel_mono (d : Graph) :
    ∀ (fuel : Nat) (v : Vertex) (φ : Formula),
      principalCarrierFormula? d fuel v = some φ →
      principalCarrierFormula? d (fuel + 1) v = some φ := by
  intro fuel
  induction fuel with
  | zero => intro v φ h; simp [principalCarrierFormula?] at h
  | succ fuel ih =>
      intro v φ h
      cases hc : classifyRule? v d with
      | none => simp [principalCarrierFormula?, hc] at h
      | some cls =>
          cases cls with
          | hypothesis =>
              rw [principalCarrierFormula?_hyp d fuel v hc] at h
              rw [principalCarrierFormula?_hyp d (Nat.succ fuel) v hc]
              exact h
          | intro p =>
              rw [principalCarrierFormula?_intro d fuel v p hc] at h
              rw [principalCarrierFormula?_intro d (Nat.succ fuel) v p hc]
              exact ih p.START φ h
          | elim major minor =>
              rw [principalCarrierFormula?_elim d fuel v major minor hc] at h
              rw [principalCarrierFormula?_elim d (Nat.succ fuel) v major minor hc]
              exact ih major.START φ h

lemma foldl_max_mono_acc (xs : List Nat) {a b : Nat} (hab : a ≤ b) :
    xs.foldl max a ≤ xs.foldl max b := by
  induction xs generalizing a b with
  | nil =>
      simpa using hab
  | cons x xs ih =>
      simp
      exact ih (by omega)

lemma le_foldl_max_of_le_acc (xs : List Nat) {a b : Nat} (hab : a ≤ b) :
    a ≤ xs.foldl max b := by
  induction xs generalizing a b with
  | nil =>
      simpa using hab
  | cons x xs ih =>
      simp
      exact ih (by omega)

lemma level_le_foldl_max_of_mem :
    ∀ (nodes : List Vertex) (v : Vertex),
      v ∈ nodes → v.LEVEL ≤ (nodes.map (·.LEVEL)).foldl max 0
  | [], v, hv => by
      simp at hv
  | u :: nodes, v, hv => by
      simp at hv
      cases hv with
      | inl huv =>
          subst huv
          exact le_foldl_max_of_le_acc (nodes.map (·.LEVEL)) (le_max_right 0 v.LEVEL)
      | inr hvn =>
          have ih := level_le_foldl_max_of_mem nodes v hvn
          exact le_trans ih
            (foldl_max_mono_acc (nodes.map (·.LEVEL)) (Nat.zero_le _))

lemma node_level_le_maxLvl (d : Graph) {v : Vertex} (hv : v ∈ d.NODES) :
    v.LEVEL ≤ (d.NODES.map (·.LEVEL)).foldl max 0 :=
  level_le_foldl_max_of_mem d.NODES v hv

lemma intro_carrier_child_mem_level (d : Graph) (hvalid : ValidDLDS d)
    {v : Vertex} {p : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p)) :
    p.START ∈ d.NODES ∧ p.START.LEVEL = v.LEVEL + 1 := by
  have hpinc : p ∈ get_rule.incoming v d :=
    classifyRule?_intro_mem_incoming hclass
  have hpedge : p ∈ d.EDGES := mem_incoming_mem_edges v d hpinc
  have hpstart : p.START ∈ d.NODES := (hvalid.hygiene.2.1 hpedge).1
  exact ⟨hpstart, incoming_start_level_of_valid d hvalid hpinc⟩

lemma elim_carrier_child_mem_level (d : Graph) (hvalid : ValidDLDS d)
    {v : Vertex} {major minor : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor)) :
    major.START ∈ d.NODES ∧ major.START.LEVEL = v.LEVEL + 1 := by
  have hmajor_inc := (classifyRule?_elim_major_mem_incoming hclass).1
  have hedge : major ∈ d.EDGES := mem_incoming_mem_edges v d hmajor_inc
  have hstart : major.START ∈ d.NODES := (hvalid.hygiene.2.1 hedge).1
  exact ⟨hstart, incoming_start_level_of_valid d hvalid hmajor_inc⟩

lemma principalCarrierFormula?_fuel_stable_above (d : Graph) (hvalid : ValidDLDS d) :
    ∀ m, ∀ v ∈ d.NODES, ∀ k,
      (d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL ≤ m →
      m < k →
      principalCarrierFormula? d k v =
        principalCarrierFormula? d (k + 1) v := by
  intro m
  induction m with
  | zero =>
      intro v hv k hmeasure hk
      cases k with
      | zero => omega
      | succ k' =>
          have hmax_le_v :
              (d.NODES.map (·.LEVEL)).foldl max 0 ≤ v.LEVEL := by
            omega
          have hv_le_max := node_level_le_maxLvl d hv
          have hlevel_eq :
              (d.NODES.map (·.LEVEL)).foldl max 0 = v.LEVEL := by
            exact le_antisymm hmax_le_v hv_le_max
          have hshape : RuleShapeOK v d := ruleShapeOK_of_valid_node d hvalid hv
          unfold RuleShapeOK ruleShapeOKB at hshape
          cases hclass : classifyRule? v d with
          | none =>
              simp [principalCarrierFormula?, hclass]
          | some cls =>
              cases cls with
              | hypothesis =>
                  simp [principalCarrierFormula?, hclass]
              | intro p =>
                  have hp := intro_carrier_child_mem_level d hvalid hclass
                  have hp_le_max := node_level_le_maxLvl d hp.1
                  omega
              | elim major minor =>
                  have hmaj := elim_carrier_child_mem_level d hvalid hclass
                  have hmaj_le_max := node_level_le_maxLvl d hmaj.1
                  omega
  | succ m ih =>
      intro v hv k hmeasure hk
      cases k with
      | zero => omega
      | succ k' =>
          have hshape : RuleShapeOK v d := ruleShapeOK_of_valid_node d hvalid hv
          unfold RuleShapeOK ruleShapeOKB at hshape
          cases hclass : classifyRule? v d with
          | none =>
              simp [principalCarrierFormula?, hclass]
          | some cls =>
              cases cls with
              | hypothesis =>
                  simp [principalCarrierFormula?, hclass]
              | intro p =>
                  have hp := intro_carrier_child_mem_level d hvalid hclass
                  have hchild_measure :
                      (d.NODES.map (·.LEVEL)).foldl max 0 - p.START.LEVEL ≤ m := by
                    omega
                  have hk' : m < k' := by omega
                  have hchild :=
                    ih p.START hp.1 k' hchild_measure hk'
                  rw [principalCarrierFormula?_intro d k' v p hclass]
                  rw [principalCarrierFormula?_intro d (k' + 1) v p hclass]
                  exact hchild
              | elim major minor =>
                  have hmaj := elim_carrier_child_mem_level d hvalid hclass
                  have hchild_measure :
                      (d.NODES.map (·.LEVEL)).foldl max 0 - major.START.LEVEL ≤ m := by
                    omega
                  have hk' : m < k' := by omega
                  have hchild :=
                    ih major.START hmaj.1 k' hchild_measure hk'
                  rw [principalCarrierFormula?_elim d k' v major minor hclass]
                  rw [principalCarrierFormula?_elim d (k' + 1) v major minor hclass]
                  exact hchild

lemma principalCarrierFormula?_fuel_stable_at_max (d : Graph) (hvalid : ValidDLDS d)
    {v : Vertex} (hv : v ∈ d.NODES) :
    principalCarrierFormula? d ((d.NODES.map (·.LEVEL)).foldl max 0 + 1) v =
      principalCarrierFormula? d (((d.NODES.map (·.LEVEL)).foldl max 0 + 1) + 1) v := by
  apply principalCarrierFormula?_fuel_stable_above d hvalid
      ((d.NODES.map (·.LEVEL)).foldl max 0) v hv
  · exact Nat.sub_le _ _
  · omega

lemma principalCarrierFormula?_fuel_stable_at_max_pos (d : Graph) (hvalid : ValidDLDS d)
    {v : Vertex} (hv : v ∈ d.NODES) (hpos : 0 < v.LEVEL) :
    principalCarrierFormula? d ((d.NODES.map (·.LEVEL)).foldl max 0) v =
      principalCarrierFormula? d (((d.NODES.map (·.LEVEL)).foldl max 0) + 1) v := by
  have hv_le_max := node_level_le_maxLvl d hv
  apply principalCarrierFormula?_fuel_stable_above d hvalid
      ((d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) v hv
  · rfl
  · omega

lemma principalCarrierColumn?_intro_of_stable (d : Graph)
    {v : Vertex} {p : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p))
    (hstable :
      principalCarrierFormula? d ((d.NODES.map (·.LEVEL)).foldl max 0) p.START =
        principalCarrierFormula? d (((d.NODES.map (·.LEVEL)).foldl max 0) + 1) p.START) :
    principalCarrierColumn? d v = principalCarrierColumn? d p.START := by
  unfold principalCarrierColumn?
  rw [principalCarrierFormula?_intro d ((d.NODES.map (·.LEVEL)).foldl max 0) v p hclass]
  rw [hstable]

lemma principalCarrierColumn?_elim_of_stable (d : Graph)
    {v : Vertex} {major minor : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor))
    (hstable :
      principalCarrierFormula? d ((d.NODES.map (·.LEVEL)).foldl max 0) major.START =
        principalCarrierFormula? d (((d.NODES.map (·.LEVEL)).foldl max 0) + 1) major.START) :
    principalCarrierColumn? d v = principalCarrierColumn? d major.START := by
  unfold principalCarrierColumn?
  rw [principalCarrierFormula?_elim d ((d.NODES.map (·.LEVEL)).foldl max 0) v major minor hclass]
  rw [hstable]

lemma classifyRule?_hypothesis_hyp {d : Graph} {v : Vertex}
    (hclass : classifyRule? v d = some DLDSRuleClass.hypothesis) :
    v.HYPOTHESIS = true := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · exact hhyp
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil => simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil =>
            simp [hinc] at hclass
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                by_cases h12 :
                    e.START.FORMULA =
                      Formula.implication e2.START.FORMULA v.FORMULA
                · simp [h12] at hclass
                ·
                  by_cases h21 :
                      e2.START.FORMULA =
                        Formula.implication e.START.FORMULA v.FORMULA
                  · by_cases hfirst :
                        e.START.FORMULA =
                          Formula.implication e2.START.FORMULA v.FORMULA
                    · exact False.elim (h12 hfirst)
                    · simp only [hfirst, if_false] at hclass
                      simp only [h21, if_true] at hclass
                      cases hclass
                  · simp [h12, h21] at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass

lemma hypothesis_carrier_reaches_source_column (d : Graph)
    {col : Nat} {v : Vertex}
    (hsrc : sourceNodeAtColumn? d col = some v)
    (hcol : (buildFormulas d)[col]? = some v.FORMULA)
    (hhyp : classifyRule? v d = some DLDSRuleClass.hypothesis) :
    routeStateAfter (pathsFromDLDS d) col
      ((d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) = some (col, col, 0) := by
  have hcolLt : col < (buildFormulas d).length := getElem?_some_lt hcol
  have hpath : col < (pathsFromDLDS d).length := by
    simpa [pathsFromDLDS_length] using hcolLt
  have hget : (buildFormulas d).get ⟨col, hcolLt⟩ = v.FORMULA := by
    exact (List.getElem?_eq_some_iff.mp hcol).2
  have hidx : (buildFormulas d).idxOf v.FORMULA = col :=
    indexOf_eq_of_get hcolLt (buildFormulas_nodup d) hget
  have hentryRaw := pathsFromDLDS_get?_of_formula d hcol
  unfold sourceNodeAtColumn? at hsrc
  rw [hcol] at hsrc
  have hvhyp : v.HYPOTHESIS = true := classifyRule?_hypothesis_hyp hhyp
  let D := (d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL
  let rest := routeFrom d (buildFormulas d) ((buildGridFromDLDS d).length - 1 - D) v.FORMULA
  have hentry :
      (pathsFromDLDS d)[col]? =
        some (List.replicate D (col + 1, 0) ++ rest) := by
    simpa [D, rest, hsrc, hvhyp, hidx] using hentryRaw
  exact routeStateAfter_replicate_prefix_self
    (pathsFromDLDS d) col D rest hpath hentry D (by simp)

lemma principal_carrier_reaches_node_hyp (d : Graph)
    (hinj : InjFormulas d)
    {v : Vertex} (hv : v ∈ d.NODES)
    {origin : Nat}
    (hhyp : classifyRule? v d = some DLDSRuleClass.hypothesis)
    (hpc : principalCarrierColumn? d v = some origin) :
    routeStateAfter (pathsFromDLDS d) origin
      ((d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) =
        some ((buildFormulas d).idxOf v.FORMULA,
          (buildFormulas d).idxOf v.FORMULA, 0) := by
  let col := (buildFormulas d).idxOf v.FORMULA
  have hmemF : v.FORMULA ∈ buildFormulas d := by
    unfold buildFormulas
    exact List.mem_eraseDups.mpr (List.mem_map.mpr ⟨v, hv, rfl⟩)
  have hcolLt : col < (buildFormulas d).length :=
    List.idxOf_lt_length_of_mem hmemF
  have hcol : (buildFormulas d)[col]? = some v.FORMULA := by
    rw [List.getElem?_eq_getElem hcolLt]
    exact congrArg some (List.getElem_idxOf hcolLt)
  have hsrc : sourceNodeAtColumn? d col = some v := by
    simpa [col] using sourceNodeAtColumn?_idxOf_of_mem d hinj hv
  have hreach := hypothesis_carrier_reaches_source_column d hsrc hcol hhyp
  have hpc_col : principalCarrierColumn? d v = some col := by
    unfold principalCarrierColumn?
    rw [principalCarrierFormula?_hyp d ((d.NODES.map (·.LEVEL)).foldl max 0) v hhyp]
    rfl
  have horigin : origin = col := Option.some.inj (hpc.symm.trans hpc_col)
  subst origin
  simpa [col] using hreach

lemma principal_carrier_reaches_source_column_hyp (d : Graph)
    {col : Nat} {v : Vertex}
    (hsrc : sourceNodeAtColumn? d col = some v)
    (hcol : (buildFormulas d)[col]? = some v.FORMULA)
    {origin : Nat}
    (hhyp : classifyRule? v d = some DLDSRuleClass.hypothesis)
    (hpc : principalCarrierForSourceColumn? d col = some origin) :
    routeStateAfter (pathsFromDLDS d) origin
      ((d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) = some (col, col, 0) := by
  have hpc_col : principalCarrierForSourceColumn? d col = some col :=
    principalCarrierForSourceColumn?_hyp d col v hsrc hcol hhyp
  have horigin : origin = col := Option.some.inj (hpc.symm.trans hpc_col)
  subst origin
  exact hypothesis_carrier_reaches_source_column d hsrc hcol hhyp

lemma principal_carrier_column_intro_eq_premise (d : Graph)
    (hvalid : ValidDLDS d)
    {v : Vertex} {p : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p)) :
    principalCarrierColumn? d v = principalCarrierColumn? d p.START := by
  have hp := intro_carrier_child_mem_level d hvalid hclass
  have hpos : 0 < p.START.LEVEL := by
    rw [hp.2]
    exact Nat.succ_pos _
  exact principalCarrierColumn?_intro_of_stable d hclass
    (principalCarrierFormula?_fuel_stable_at_max_pos d hvalid hp.1 hpos)

lemma principal_carrier_column_elim_eq_major (d : Graph)
    (hvalid : ValidDLDS d)
    {v : Vertex} {major minor : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor)) :
    principalCarrierColumn? d v = principalCarrierColumn? d major.START := by
  have hmajor := elim_carrier_child_mem_level d hvalid hclass
  have hpos : 0 < major.START.LEVEL := by
    rw [hmajor.2]
    exact Nat.succ_pos _
  exact principalCarrierColumn?_elim_of_stable d hclass
    (principalCarrierFormula?_fuel_stable_at_max_pos d hvalid hmajor.1 hpos)

lemma simpleTree_outgoing_eq_singleton (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {v : Vertex} {e : Deduction}
    (heout : e ∈ get_rule.outgoing v d) :
    get_rule.outgoing v d = [e] := by
  have hedge : e ∈ d.EDGES := mem_outgoing_mem_edges v d heout
  have hstart : e.START = v := mem_outgoing_start_eq v d heout
  have hv : v ∈ d.NODES := by
    simpa [hstart] using (hvalid.hygiene.2.1 hedge).1
  have hlen := htree.1.1 v hv
  cases hout : get_rule.outgoing v d with
  | nil =>
      rw [hout] at heout
      simp at heout
  | cons a rest =>
      cases rest with
      | nil =>
          rw [hout] at heout
          simp at heout
          subst heout
          rfl
      | cons b rest =>
          rw [hout] at hlen
          simp at hlen

lemma routeFrom_head_along_nonminor_outgoing
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {fuel src dst : Nat} {v w : Vertex} {e : Deduction}
    (hsrc : sourceNodeAtColumn? d src = some v)
    (hdst : sourceNodeAtColumn? d dst = some w)
    (heout : e ∈ get_rule.outgoing v d)
    (_hstart : e.START = v)
    (hend : e.END = w)
    (hnonminor :
      (match classifyRule? w d with
       | some (DLDSRuleClass.elim _ minor) =>
           decide (v.FORMULA = minor.START.FORMULA)
       | _ => false) = false) :
    ∃ rest,
      routeFrom d (buildFormulas d) (fuel + 1) v.FORMULA =
        (dst + 1, inputLabelForEdge d (buildFormulas d) v.FORMULA w) :: rest := by
  have hv : v ∈ d.NODES := sourceNodeAtColumn?_mem d hsrc
  have hfind : d.NODES.find? (fun u => decide (u.FORMULA = v.FORMULA)) = some v :=
    find_node_by_formula_eq_of_inj d htree.2 hv
  have hout : get_rule.outgoing v d = [e] :=
    simpleTree_outgoing_eq_singleton d htree hvalid heout
  have hdstφ : (buildFormulas d)[dst]? = some w.FORMULA :=
    sourceNodeAtColumn?_eq_implies_formula_at_column d hdst
  have hdstLt : dst < (buildFormulas d).length := getElem?_some_lt hdstφ
  have hdstGet : (buildFormulas d).get ⟨dst, hdstLt⟩ = w.FORMULA :=
    (List.getElem?_eq_some_iff.mp hdstφ).2
  have hidxw : (buildFormulas d).idxOf w.FORMULA = dst :=
    indexOf_eq_of_get hdstLt (buildFormulas_nodup d) hdstGet
  refine ⟨routeFrom d (buildFormulas d) fuel w.FORMULA, ?_⟩
  have hnonminor' :
      (match classifyRule? e.END d with
       | some (DLDSRuleClass.elim _ minor) =>
           decide (v.FORMULA = minor.START.FORMULA)
       | _ => false) = false := by
    simpa [hend] using hnonminor
  have hroute :=
    routeFrom_nonminor_tail_continues d (buildFormulas d) fuel
      v.FORMULA v e [] hfind hout hnonminor'
  simpa [hend, hidxw] using hroute

lemma routeFrom_head_along_nonminor_outgoing_get?
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {fuel src dst : Nat} {v w : Vertex} {e : Deduction}
    (hsrc : sourceNodeAtColumn? d src = some v)
    (hdst : sourceNodeAtColumn? d dst = some w)
    (heout : e ∈ get_rule.outgoing v d)
    (hstart : e.START = v)
    (hend : e.END = w)
    (hnonminor :
      (match classifyRule? w d with
       | some (DLDSRuleClass.elim _ minor) =>
           decide (v.FORMULA = minor.START.FORMULA)
       | _ => false) = false) :
    (routeFrom d (buildFormulas d) (fuel + 1) v.FORMULA)[0]? =
      some (dst + 1, inputLabelForEdge d (buildFormulas d) v.FORMULA w) := by
  obtain ⟨rest, hroute⟩ :=
    routeFrom_head_along_nonminor_outgoing d htree hvalid
      hsrc hdst heout hstart hend hnonminor
  rw [hroute]
  rfl

lemma principalCarrierFormula?_some_is_hypothesis_formula
    (d : Graph) (hvalid : ValidDLDS d)
    {fuel : Nat} {v : Vertex} {φ : Formula}
    (hv : v ∈ d.NODES)
    (hpc : principalCarrierFormula? d fuel v = some φ) :
    ∃ hvtx,
      hvtx ∈ d.NODES ∧
      hvtx.HYPOTHESIS = true ∧
      hvtx.FORMULA = φ := by
  induction fuel generalizing v φ with
  | zero =>
      simp [principalCarrierFormula?] at hpc
  | succ fuel ih =>
      cases hclass : classifyRule? v d with
      | none =>
          simp [principalCarrierFormula?, hclass] at hpc
      | some cls =>
          cases cls with
          | hypothesis =>
              simp [principalCarrierFormula?, hclass] at hpc
              subst φ
              exact ⟨v, hv, classifyRule?_hypothesis_hyp hclass, rfl⟩
          | intro p =>
              rw [principalCarrierFormula?_intro d fuel v p hclass] at hpc
              have hp := intro_carrier_child_mem_level d hvalid hclass
              exact ih hp.1 hpc
          | elim major minor =>
              rw [principalCarrierFormula?_elim d fuel v major minor hclass] at hpc
              have hmajor := elim_carrier_child_mem_level d hvalid hclass
              exact ih hmajor.1 hpc

lemma find_hypothesis_by_principal_formula
    (d : Graph) (htree : IsSimpleTreeDLDS d)
    {hvtx : Vertex} {φ : Formula}
    (hhmem : hvtx ∈ d.NODES)
    (hhφ : hvtx.FORMULA = φ) :
    d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some hvtx := by
  have hfind := find_node_by_formula_eq_of_inj d htree.2 hhmem
  simpa [hhφ] using hfind

lemma getElem?_eq_some_of_idxOf_eq
    (d : Graph) {φ : Formula} {origin : Nat}
    (hmem : φ ∈ buildFormulas d)
    (hidx : (buildFormulas d).idxOf φ = origin) :
    (buildFormulas d)[origin]? = some φ := by
  subst origin
  have hlt : (buildFormulas d).idxOf φ < (buildFormulas d).length :=
    List.idxOf_lt_length_of_mem hmem
  rw [List.getElem?_eq_getElem hlt]
  exact congrArg some (List.getElem_idxOf hlt)

lemma pathsFromDLDS_principal_origin_entry_some_hyp
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {src origin : Nat} {v : Vertex}
    (hsrc : sourceNodeAtColumn? d src = some v)
    (hprincipal : principalCarrierForSourceColumn? d src = some origin) :
    ∃ φ hv delay rest,
      (buildFormulas d)[origin]? = some φ ∧
      d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some hv ∧
      hv.HYPOTHESIS = true ∧
      delay = (((d.NODES.map (·.LEVEL)).foldl max 0) - hv.LEVEL) ∧
      rest =
        routeFrom d (buildFormulas d)
          ((buildGridFromDLDS d).length - 1 - delay)
          φ ∧
      (pathsFromDLDS d)[origin]? =
        some (List.replicate delay ((buildFormulas d).idxOf φ + 1, 0) ++ rest) := by
  have hv : v ∈ d.NODES := sourceNodeAtColumn?_mem d hsrc
  unfold principalCarrierForSourceColumn? at hprincipal
  rw [hsrc] at hprincipal
  unfold principalCarrierColumn? at hprincipal
  cases hpcf :
      principalCarrierFormula? d (((d.NODES.map (·.LEVEL)).foldl max 0) + 1) v with
  | none =>
      simp [hpcf] at hprincipal
  | some φ =>
      have hidx : (buildFormulas d).idxOf φ = origin := by
        simp [hpcf] at hprincipal
        exact hprincipal
      obtain ⟨hvtx, hhmem, hhhyp, hhφ⟩ :=
        principalCarrierFormula?_some_is_hypothesis_formula d hvalid hv hpcf
      have hfind :
          d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some hvtx :=
        find_hypothesis_by_principal_formula d htree hhmem hhφ
      have hmemF : φ ∈ buildFormulas d := by
        unfold buildFormulas
        exact List.mem_eraseDups.mpr
          (List.mem_map.mpr ⟨hvtx, hhmem, hhφ⟩)
      have hφorigin : (buildFormulas d)[origin]? = some φ :=
        getElem?_eq_some_of_idxOf_eq d hmemF hidx
      have hentryRaw := pathsFromDLDS_get?_of_formula d hφorigin
      let delay := ((d.NODES.map (·.LEVEL)).foldl max 0) - hvtx.LEVEL
      let rest :=
        routeFrom d (buildFormulas d)
          ((buildGridFromDLDS d).length - 1 - delay)
          φ
      refine ⟨φ, hvtx, delay, rest, hφorigin, hfind, hhhyp, rfl, rfl, ?_⟩
      rw [hfind] at hentryRaw
      dsimp only at hentryRaw
      rw [hhhyp] at hentryRaw
      exact hentryRaw

lemma pathsFromDLDS_principal_origin_entry
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {src origin : Nat} {v : Vertex}
    (hsrc : sourceNodeAtColumn? d src = some v)
    (hprincipal : principalCarrierForSourceColumn? d src = some origin) :
    ∃ φ hv,
      (buildFormulas d)[origin]? = some φ ∧
      d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some hv ∧
      hv.HYPOTHESIS = true ∧
      (pathsFromDLDS d)[origin]? =
        some
          (List.replicate
            (((d.NODES.map (·.LEVEL)).foldl max 0) - hv.LEVEL)
            ((buildFormulas d).idxOf φ + 1, 0)
          ++
          routeFrom d (buildFormulas d)
            ((buildGridFromDLDS d).length - 1 -
              (((d.NODES.map (·.LEVEL)).foldl max 0) - hv.LEVEL))
            φ) := by
  obtain ⟨φ, hv, delay, rest, hφ, hfind, hhyp, hdelay, hrest, hentry⟩ :=
    pathsFromDLDS_principal_origin_entry_some_hyp d htree hvalid hsrc hprincipal
  refine ⟨φ, hv, hφ, hfind, hhyp, ?_⟩
  subst delay
  subst rest
  exact hentry

lemma routeStateAfter_succ_of_step_lookup
    {paths : PathInput}
    {origin depth current source label target inputLabel : Nat}
    (hstate :
      routeStateAfter paths origin depth = some (current, source, label))
    (hstep :
      paths[origin]? >>= (fun steps => steps[depth]?) =
        some (target, inputLabel))
    (hnz : target ≠ 0) :
    routeStateAfter paths origin (depth + 1) =
      some (target - 1, current, inputLabel) := by
  rw [show depth + 1 = Nat.succ depth by rfl]
  unfold routeStateAfter
  rw [hstate]
  cases hpaths : paths[origin]? with
  | none =>
      simp [hpaths] at hstep
  | some steps =>
      cases hlookup : steps[depth]? with
      | none =>
          simp [hpaths, hlookup] at hstep
      | some step =>
          rcases step with ⟨target', inputLabel'⟩
          have hstep' : some (target', inputLabel') = some (target, inputLabel) := by
            simpa [hpaths, hlookup] using hstep
          have htarget : target' = target := congrArg Prod.fst (Option.some.inj hstep')
          have hlabel : inputLabel' = inputLabel := congrArg Prod.snd (Option.some.inj hstep')
          subst target'
          subst inputLabel'
          simp [hlookup, hnz]

lemma routeStateAfter_succ_of_step_lookup_dst
    {paths : PathInput}
    {origin depth current source label dst inputLabel : Nat}
    (hstate :
      routeStateAfter paths origin depth = some (current, source, label))
    (hstep :
      paths[origin]? >>= (fun steps => steps[depth]?) =
        some (dst + 1, inputLabel)) :
    routeStateAfter paths origin (depth + 1) =
      some (dst, current, inputLabel) := by
  have hsucc :=
    routeStateAfter_succ_of_step_lookup (paths := paths) hstate hstep
      (by omega)
  have hsub : (dst + 1) - 1 = dst := by omega
  simpa [hsub] using hsucc

lemma principal_transfer_step_of_stable (d : Graph)
    (hinj : InjFormulas d) (hvalid : ValidDLDS d)
    (hstable :
      ∀ u ∈ d.NODES, 0 < u.LEVEL →
        principalCarrierFormula? d ((d.NODES.map (·.LEVEL)).foldl max 0) u =
          principalCarrierFormula? d (((d.NODES.map (·.LEVEL)).foldl max 0) + 1) u)
    {v w : Vertex} {e : Deduction} {origin : Nat}
    (hv : v ∈ d.NODES) (hw : w ∈ d.NODES)
    (heout : e ∈ get_rule.outgoing v d)
    (hend : e.END = w)
    (hnonminor :
      (match classifyRule? w d with
       | some (DLDSRuleClass.elim _ minor) =>
           decide (v.FORMULA = minor.START.FORMULA)
       | _ => false) = false)
    (hprev :
      principalCarrierForSourceColumn? d
        ((buildFormulas d).idxOf v.FORMULA) = some origin) :
    principalCarrierForSourceColumn? d
        ((buildFormulas d).idxOf w.FORMULA) = some origin := by
  have hstart : e.START = v := mem_outgoing_start_eq v d heout
  have hein : e ∈ get_rule.incoming w d := by
    have h := outgoing_mem_incoming_end v d heout
    simpa [hend] using h
  rw [principalCarrierForSourceColumn?_idxOf_of_mem d hinj hw]
  rw [principalCarrierForSourceColumn?_idxOf_of_mem d hinj hv] at hprev
  have hshape : RuleShapeOK w d := ruleShapeOK_of_valid_node d hvalid hw
  unfold RuleShapeOK ruleShapeOKB at hshape
  cases hclass : classifyRule? w d with
  | none =>
      simp [hclass] at hshape
  | some cls =>
      cases cls with
      | hypothesis =>
          have hhyp : w.HYPOTHESIS = true := classifyRule?_hypothesis_hyp hclass
          have hno := hvalid.hypNoIncoming w hw hhyp
          rw [hno] at hein
          simp at hein
      | intro p =>
          have hep : e = p := classifyRule?_intro_incoming_eq hclass hein
          have hpstart : p.START = v := by
            rw [← hep]
            exact hstart
          have hp_pos : 0 < p.START.LEVEL := by
            have hvlevel := outgoing_end_level_of_valid d hvalid heout
            rw [hpstart]
            rw [hvlevel]
            exact Nat.succ_pos _
          have hcol :
              principalCarrierColumn? d w = principalCarrierColumn? d p.START :=
            principalCarrierColumn?_intro_of_stable d hclass
              (hstable p.START (by simpa [hpstart] using hv) hp_pos)
          rw [hcol, hpstart]
          exact hprev
      | elim major minor =>
          have he_or :
              e = major ∨ e = minor :=
            classifyRule?_elim_incoming_eq_major_or_minor hclass hein
          cases he_or with
          | inl hemajor =>
              have hmajstart : major.START = v := by
                rw [← hemajor]
                exact hstart
              have hmaj_pos : 0 < major.START.LEVEL := by
                have hvlevel := outgoing_end_level_of_valid d hvalid heout
                rw [hmajstart]
                rw [hvlevel]
                exact Nat.succ_pos _
              have hcol :
                  principalCarrierColumn? d w =
                    principalCarrierColumn? d major.START :=
                principalCarrierColumn?_elim_of_stable d hclass
                  (hstable major.START (by simpa [hmajstart] using hv) hmaj_pos)
              rw [hcol, hmajstart]
              exact hprev
          | inr heminor =>
              have hvminor : v.FORMULA = minor.START.FORMULA := by
                rw [← hstart, heminor]
              rw [hclass] at hnonminor
              simp [hvminor] at hnonminor

lemma principal_transfer_step (d : Graph)
    (hinj : InjFormulas d) (hvalid : ValidDLDS d)
    {v w : Vertex} {e : Deduction} {origin : Nat}
    (hv : v ∈ d.NODES) (hw : w ∈ d.NODES)
    (heout : e ∈ get_rule.outgoing v d)
    (hend : e.END = w)
    (hnonminor :
      (match classifyRule? w d with
       | some (DLDSRuleClass.elim _ minor) =>
           decide (v.FORMULA = minor.START.FORMULA)
       | _ => false) = false)
    (hprev :
      principalCarrierForSourceColumn? d
        ((buildFormulas d).idxOf v.FORMULA) = some origin) :
    principalCarrierForSourceColumn? d
        ((buildFormulas d).idxOf w.FORMULA) = some origin := by
  exact principal_transfer_step_of_stable d hinj hvalid
    (fun u hu hpos =>
      principalCarrierFormula?_fuel_stable_at_max_pos d hvalid hu hpos)
    hv hw heout hend hnonminor hprev

private lemma routeFrom_minor_match_tail_replicate
    (d : Graph) (formulas : List Formula) (fuel : Nat)
    (phi : Formula) (v : Vertex) (e : Deduction) (es : List Deduction)
    (hfind : d.NODES.find? (fun u => decide (u.FORMULA = phi)) = some v)
    (hout : get_rule.outgoing v d = e :: es)
    (hminor :
      (match classifyRule? e.END d with
       | some (DLDSRuleClass.elim _ minor) =>
           decide (phi = minor.START.FORMULA)
       | _ => false) = true) :
    routeFrom d formulas (Nat.succ fuel) phi =
      (formulas.idxOf e.END.FORMULA + 1, inputLabelForEdge d formulas phi e.END) ::
        List.replicate fuel (0, 0) := by
  cases hclass : classifyRule? e.END d with
  | none =>
      simp [hclass] at hminor
  | some cls =>
      cases cls with
      | hypothesis =>
          simp [hclass] at hminor
      | intro p =>
          simp [hclass] at hminor
      | elim major minor =>
          have hphi : phi = minor.START.FORMULA := by
            exact of_decide_eq_true (by simpa [hclass] using hminor)
          exact routeFrom_minor_tail_replicate d formulas fuel phi v e es major minor
            hfind hout hclass hphi

lemma routeFrom_preserves_principal_to_live_target (d : Graph)
    (hinj : InjFormulas d) (hvalid : ValidDLDS d) :
    ∀ (fuel : Nat) (φ : Formula) (j src col prevLbl lbl origin : Nat),
      (routeFrom d (buildFormulas d) fuel φ)[j]? = some (src + 1, prevLbl) →
      (routeFrom d (buildFormulas d) fuel φ)[j + 1]? = some (col + 1, lbl) →
      principalCarrierForSourceColumn? d ((buildFormulas d).idxOf φ) = some origin →
      principalCarrierForSourceColumn? d src = some origin := by
  intro fuel
  induction fuel with
  | zero =>
      intro φ j src col prevLbl lbl origin hstep _ _
      simp [routeFrom] at hstep
  | succ fuel ih =>
      intro φ j src col prevLbl lbl origin hstep hnext hprincipal
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          cases j with
          | zero =>
              simp [routeFrom, hfind] at hstep
          | succ j =>
              have hstepTail :
                  (routeFrom d (buildFormulas d) fuel φ)[j]? =
                    some (src + 1, prevLbl) := by
                simpa [routeFrom, hfind] using hstep
              have hnextTail :
                  (routeFrom d (buildFormulas d) fuel φ)[j + 1]? =
                    some (col + 1, lbl) := by
                simpa [routeFrom, hfind, Nat.succ_eq_add_one, Nat.add_assoc] using hnext
              exact ih φ j src col prevLbl lbl origin hstepTail hnextTail hprincipal
      | some v =>
          have hvφ : v.FORMULA = φ := by
            exact of_decide_eq_true
              (List.find?_some
                (p := fun u => decide (u.FORMULA = φ))
                (l := d.NODES) hfind)
          cases hout : get_rule.outgoing v d with
          | nil =>
              cases j with
              | zero =>
                  simp [routeFrom, hfind, hout] at hstep
              | succ j =>
                  have hstepTail :
                      (routeFrom d (buildFormulas d) fuel φ)[j]? =
                        some (src + 1, prevLbl) := by
                    simpa [routeFrom, hfind, hout] using hstep
                  have hnextTail :
                      (routeFrom d (buildFormulas d) fuel φ)[j + 1]? =
                        some (col + 1, lbl) := by
                    simpa [routeFrom, hfind, hout, Nat.succ_eq_add_one, Nat.add_assoc] using hnext
                  exact ih φ j src col prevLbl lbl origin hstepTail hnextTail hprincipal
          | cons e es =>
              have heout : e ∈ get_rule.outgoing v d := by
                rw [hout]
                simp
              have hedge : e ∈ d.EDGES := mem_outgoing_mem_edges v d heout
              have hstart : e.START = v := mem_outgoing_start_eq v d heout
              have hv : v ∈ d.NODES := by
                simpa [hstart] using (hvalid.hygiene.2.1 hedge).1
              have hw : e.END ∈ d.NODES := (hvalid.hygiene.2.1 hedge).2
              have hprincipal_v :
                  principalCarrierForSourceColumn? d
                    ((buildFormulas d).idxOf v.FORMULA) = some origin := by
                simpa [hvφ] using hprincipal
              cases j with
              | zero =>
                  have hsrc :
                      src = (buildFormulas d).idxOf e.END.FORMULA := by
                    have hhead :
                        some ((buildFormulas d).idxOf e.END.FORMULA + 1,
                          inputLabelForEdge d (buildFormulas d) φ e.END) =
                          some (src + 1, prevLbl) := by
                      simpa [routeFrom, hfind, hout] using hstep
                    have hfst := congrArg Prod.fst (Option.some.inj hhead)
                    simpa [routeFrom, hfind, hout] using hfst.symm
                  have hnotminor :
                      (match classifyRule? e.END d with
                       | some (DLDSRuleClass.elim _ minor) =>
                           decide (φ = minor.START.FORMULA)
                       | _ => false) = false := by
                    by_cases hminor :
                        (match classifyRule? e.END d with
                         | some (DLDSRuleClass.elim _ minor) =>
                             decide (φ = minor.START.FORMULA)
                         | _ => false) = true
                    · exfalso
                      have hr :=
                        routeFrom_minor_match_tail_replicate d (buildFormulas d) fuel
                          φ v e es hfind hout hminor
                      have hnextRep :
                          (List.replicate fuel (0, 0) : List (Nat × Nat))[0]? =
                            some (col + 1, lbl) := by
                        have hnext' := hnext
                        rw [hr] at hnext'
                        simpa using hnext'
                      exact replicate_zero_get_nonzero_false
                        hnextRep
                        (by omega)
                    · cases hval :
                        (match classifyRule? e.END d with
                         | some (DLDSRuleClass.elim _ minor) =>
                             decide (φ = minor.START.FORMULA)
                         | _ => false) <;> simp [hval] at hminor ⊢
                  have hnotminor_v :
                      (match classifyRule? e.END d with
                       | some (DLDSRuleClass.elim _ minor) =>
                           decide (v.FORMULA = minor.START.FORMULA)
                       | _ => false) = false := by
                    simpa [hvφ] using hnotminor
                  have htransfer :=
                    principal_transfer_step d hinj hvalid hv hw heout rfl hnotminor_v
                      hprincipal_v
                  simpa [hsrc] using htransfer
              | succ j =>
                  have hnotminor :
                      (match classifyRule? e.END d with
                       | some (DLDSRuleClass.elim _ minor) =>
                           decide (φ = minor.START.FORMULA)
                       | _ => false) = false := by
                    by_cases hminor :
                        (match classifyRule? e.END d with
                         | some (DLDSRuleClass.elim _ minor) =>
                             decide (φ = minor.START.FORMULA)
                         | _ => false) = true
                    · exfalso
                      have hr :=
                        routeFrom_minor_match_tail_replicate d (buildFormulas d) fuel
                          φ v e es hfind hout hminor
                      have hstepRep :
                          (List.replicate fuel (0, 0) : List (Nat × Nat))[j]? =
                            some (src + 1, prevLbl) := by
                        have hstep' := hstep
                        rw [hr] at hstep'
                        simpa using hstep'
                      exact replicate_zero_get_nonzero_false
                        hstepRep
                        (by omega)
                    · cases hval :
                        (match classifyRule? e.END d with
                         | some (DLDSRuleClass.elim _ minor) =>
                             decide (φ = minor.START.FORMULA)
                         | _ => false) <;> simp [hval] at hminor ⊢
                  have hnotminor_v :
                      (match classifyRule? e.END d with
                       | some (DLDSRuleClass.elim _ minor) =>
                           decide (v.FORMULA = minor.START.FORMULA)
                       | _ => false) = false := by
                    simpa [hvφ] using hnotminor
                  have hprincipal_w :
                      principalCarrierForSourceColumn? d
                        ((buildFormulas d).idxOf e.END.FORMULA) = some origin :=
                    principal_transfer_step d hinj hvalid hv hw heout rfl hnotminor_v
                      hprincipal_v
                  have hstepTail :
                      (routeFrom d (buildFormulas d) fuel e.END.FORMULA)[j]? =
                        some (src + 1, prevLbl) := by
                    have hr :=
                      routeFrom_nonminor_tail_continues d (buildFormulas d) fuel
                        φ v e es hfind hout hnotminor
                    have hstep' := hstep
                    rw [hr] at hstep'
                    simpa using hstep'
                  have hnextTail :
                      (routeFrom d (buildFormulas d) fuel e.END.FORMULA)[j + 1]? =
                        some (col + 1, lbl) := by
                    have hr :=
                      routeFrom_nonminor_tail_continues d (buildFormulas d) fuel
                        φ v e es hfind hout hnotminor
                    have hnext' := hnext
                    rw [hr] at hnext'
                    simpa [Nat.succ_eq_add_one, Nat.add_assoc] using hnext'
                  exact ih e.END.FORMULA j src col prevLbl lbl origin
                    hstepTail hnextTail hprincipal_w

lemma carrier_preservation_step (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth origin prevSource src col lbl prevLbl : Nat}
    (h1 : routeStateAfter (pathsFromDLDS d) origin (depth + 1) =
            some (src, prevSource, prevLbl))
    (h2 : routeStateAfter (pathsFromDLDS d) origin (depth + 2) = some (col, src, lbl))
    (h3 : principalCarrierForSourceColumn? d prevSource = some origin) :
    principalCarrierForSourceColumn? d src = some origin := by
  obtain ⟨_cur0, _src0, _lbl0, steps, target, inputLabel,
    _hprev0, hsteps, hstep, htarget, htriple⟩ :=
    routeStateAfter_live_succ (pathsFromDLDS d) h1
  cases htriple
  obtain ⟨_cur1, _src1, _lbl1, steps2, target2, inputLabel2,
    _hprev1, hsteps2, hstepNext, htarget2, htriple2⟩ :=
    routeStateAfter_live_succ (pathsFromDLDS d) h2
  cases htriple2
  have hsteps2_eq : steps2 = steps := by
    rw [hsteps] at hsteps2
    exact (Option.some.inj hsteps2).symm
  subst steps2
  have htarget_succ : target = target - 1 + 1 := by omega
  have htarget2_succ : target2 = target2 - 1 + 1 := by omega
  have horiginPath : origin < (pathsFromDLDS d).length :=
    getElem?_some_lt hsteps
  have horiginF : origin < (buildFormulas d).length := by
    simpa [pathsFromDLDS_length] using horiginPath
  let φ := (buildFormulas d).get ⟨origin, horiginF⟩
  have hφ : (buildFormulas d)[origin]? = some φ :=
    List.getElem?_eq_getElem horiginF
  have hentry := pathsFromDLDS_get?_of_formula d hφ
  cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
  | none =>
      have hsteps_eq : steps =
          List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) := by
        rw [hentry] at hsteps
        exact (Option.some.inj (by simpa [hfind] using hsteps)).symm
      exact False.elim
        (replicate_zero_get_nonzero_false
          (by simpa [hsteps_eq] using hstep) htarget)
  | some v =>
      have hvφ : v.FORMULA = φ := by
        exact of_decide_eq_true
          (List.find?_some
            (p := fun u => decide (u.FORMULA = φ))
            (l := d.NODES) hfind)
      by_cases hhyp : v.HYPOTHESIS = true
      · let D := (d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL
        let rest :=
          routeFrom d (buildFormulas d)
            ((buildGridFromDLDS d).length - 1 - D) φ
        have hsteps_eq : steps =
            List.replicate D ((buildFormulas d).idxOf φ + 1, 0) ++ rest := by
          rw [hentry] at hsteps
          exact (Option.some.inj (by simpa [hfind, hhyp, D, rest] using hsteps)).symm
        have hidx_origin : (buildFormulas d).idxOf φ = origin := by
          have hget : (buildFormulas d).get ⟨origin, horiginF⟩ = φ := rfl
          exact indexOf_eq_of_get horiginF (buildFormulas_nodup d) hget
        have hentry_path :
            (pathsFromDLDS d)[origin]? =
              some (List.replicate D (origin + 1, 0) ++ rest) := by
          rw [hsteps]
          simp [hsteps_eq, hidx_origin]
        by_cases hdelay : depth < D
        · have hself :=
            routeStateAfter_replicate_prefix_self
              (pathsFromDLDS d) origin D rest horiginPath hentry_path
              (depth + 1) (by omega)
          rw [h1] at hself
          have hEq := Option.some.inj hself
          have hcur : target - 1 = origin := congrArg (fun p : Nat × Nat × Nat => p.1) hEq
          have hprevEq : prevSource = origin :=
            congrArg (fun p : Nat × Nat × Nat => p.2.1) hEq
          rw [hprevEq] at h3
          rw [hcur]
          exact h3
        · have hge : D ≤ depth := by omega
          have hstepTail :
              rest[depth - D]? = some (target - 1 + 1, prevLbl) := by
            have hright :=
              getElem?_append_right_sub
                (List.replicate D (origin + 1, 0)) rest (i := depth)
                (by simpa [List.length_replicate] using hge)
            rw [hsteps_eq] at hstep
            rw [hidx_origin] at hstep
            rw [hright] at hstep
            rw [← htarget_succ]
            simpa [List.length_replicate] using hstep
          have hstepNextTail :
              rest[(depth - D) + 1]? = some (target2 - 1 + 1, lbl) := by
            have hright :=
              getElem?_append_right_sub
                (List.replicate D (origin + 1, 0)) rest (i := depth + 1)
                (by simp [List.length_replicate]; omega)
            rw [hsteps_eq] at hstepNext
            rw [hidx_origin] at hstepNext
            rw [hright] at hstepNext
            have hidx : depth + 1 - D = (depth - D) + 1 := by omega
            rw [← hidx, ← htarget2_succ]
            simpa [List.length_replicate] using hstepNext
          have hsrcNode :
              sourceNodeAtColumn? d origin = some v := by
            unfold sourceNodeAtColumn?
            have hcol : (buildFormulas d)[origin]? = some v.FORMULA := by
              simpa [hvφ] using hφ
            rw [hcol]
            simpa [hvφ] using hfind
          have hcolv : (buildFormulas d)[origin]? = some v.FORMULA := by
            simpa [hvφ] using hφ
          have hclass : classifyRule? v d = some DLDSRuleClass.hypothesis := by
            simp [classifyRule?, hhyp]
          have hprincipal_start :
              principalCarrierForSourceColumn? d ((buildFormulas d).idxOf φ) =
                some origin := by
            have hh :=
              principalCarrierForSourceColumn?_hyp d origin v hsrcNode hcolv hclass
            simpa [hidx_origin] using hh
          have htail :=
            routeFrom_preserves_principal_to_live_target d htree.2 hvalid
              ((buildGridFromDLDS d).length - 1 - D) φ (depth - D)
              (target - 1) (target2 - 1) prevLbl lbl origin
              hstepTail hstepNextTail hprincipal_start
          exact htail
      · have hsteps_eq : steps =
            List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) := by
          rw [hentry] at hsteps
          exact (Option.some.inj (by simpa [hfind, hhyp] using hsteps)).symm
        exact False.elim
          (replicate_zero_get_nonzero_false
            (by simpa [hsteps_eq] using hstep) htarget)

/--
 Every positive-depth live step is carried by the principal carrier of its
    recorded source column.
-/
lemma pathsFromDLDS_step_principal (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d) :
    ∀ (depth origin col src lbl : Nat),
      routeStateAfter (pathsFromDLDS d) origin (depth + 1) = some (col, src, lbl) →
      principalCarrierForSourceColumn? d src = some origin := by
  intro depth
  induction depth with
  | zero =>
      intro origin col src lbl hstep
      exact pathsFromDLDS_step_principal_zero d hstep
  | succ depth ih =>
      intro origin col src lbl hstep
      obtain ⟨prevSource, prevLbl, hprev⟩ :=
        routeStateAfter_live_succ_prev_source (pathsFromDLDS d) hstep
      have hPrevPrincipal : principalCarrierForSourceColumn? d prevSource = some origin :=
        ih origin src prevSource prevLbl hprev
      exact carrier_preservation_step d htree hvalid hprev hstep hPrevPrincipal

/--
 Positive-depth arrival form of `pathsFromDLDS_step_principal`.

This is the route-level source/principal fact needed by the no-extra-carrier
argument: once a token has taken at least one live step, its recorded
`source_column` is carried by exactly that origin.
-/
lemma arriving_state_positive_principal_source (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth origin col src lbl : Nat}
    (hpos : 0 < depth)
    (hstate :
      routeStateAfter (pathsFromDLDS d) origin depth = some (col, src, lbl)) :
    principalCarrierForSourceColumn? d src = some origin := by
  cases depth with
  | zero =>
      omega
  | succ depth =>
      exact pathsFromDLDS_step_principal d htree hvalid
        depth origin col src lbl hstate



/--
 If vertex `v` has an outgoing edge `e` to vertex `w`, and `φ = v.FORMULA`,
    then `inputLabelForEdge d formulas φ w` decodes to source `idxOf φ` with the
    rule index pinned to `w`'s `ruleIndexForNode?` (the rule applied AT column `w`).
    Exposing the rule index is what makes `decodedRuleAtColumn?` unique per column.
-/
private lemma inputLabelForEdge_decodes_to_src (d : Graph) (hvalid : ValidDLDS d)
    {v w : Vertex} {e : Deduction} {φ : Formula} {ruleIdx : Nat}
    (_hv : v ∈ d.NODES) (hw : w ∈ d.NODES)
    (hvφ : v.FORMULA = φ)
    (heout : e ∈ get_rule.outgoing v d)
    (hend : e.END = w)
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some ruleIdx) :
    ∃ slot,
      decodeInputLabel
        (buildIncomingMapForFormula (buildFormulas d) w.FORMULA)
        (inputLabelForEdge d (buildFormulas d) φ w) =
          some (ruleIdx, slot, (buildFormulas d).idxOf φ) := by
  have hein : e ∈ get_rule.incoming w d := by
    rw [← hend]; exact outgoing_mem_incoming_end v d heout
  cases hclass : classifyRule? w d with
  | none =>
      simp [ruleIndexForNode?, hclass] at hsel
  | some cls =>
      cases cls with
      | hypothesis =>
          have hhyp : w.HYPOTHESIS = true := classifyRule?_hypothesis_hyp hclass
          have hno := hvalid.hypNoIncoming w hw hhyp
          simp [hno] at hein
      | intro p =>
          have hep : e = p := classifyRule?_intro_incoming_eq hclass hein
          have hstart : e.START = v := mem_outgoing_start_eq v d heout
          have hpφ : p.START.FORMULA = φ := by
            rw [← hep, hstart]; exact hvφ
          refine ⟨0, ?_⟩
          rw [← hpφ]
          exact inputLabelForEdge_decodes_intro_premise d w p ruleIdx hclass hsel
      | elim major minor =>
          have hstart : e.START = v := mem_outgoing_start_eq v d heout
          have hmor := classifyRule?_elim_incoming_eq_major_or_minor hclass hein
          cases hmor with
          | inl hemajor =>
              have hφmaj : major.START.FORMULA = φ := by
                rw [← hemajor, hstart]; exact hvφ
              refine ⟨0, ?_⟩
              rw [← hφmaj]
              exact inputLabelForEdge_decodes_elim_major d w major minor ruleIdx hclass hsel
          | inr heminor =>
              have hφmin : minor.START.FORMULA = φ := by
                rw [← heminor, hstart]; exact hvφ
              refine ⟨1, ?_⟩
              rw [← hφmin]
              exact inputLabelForEdge_decodes_elim_minor d w major minor ruleIdx hclass hsel

/--
 First step of `routeFrom` from `φ`: its label decodes to source = `idxOf φ`,
    with the rule index pinned to the landing node's `ruleIndexForNode?` (the node
    at `col`). Exposes the landing node `w` so callers can identify the column's
    rule uniformly across arrivals.
-/
private lemma routeFrom_step0_label_decode_src (d : Graph) (hvalid : ValidDLDS d)
    (hinj : InjFormulas d) :
    ∀ (fuel : Nat) (φ : Formula) (col lbl : Nat),
      (routeFrom d (buildFormulas d) fuel φ)[0]? = some (col + 1, lbl) →
      ∃ ruleIdx slot w,
        sourceNodeAtColumn? d col = some w ∧
        ruleIndexForNode? d (buildFormulas d) w = some ruleIdx ∧
        decodeInputLabel
          (buildIncomingMapForFormula (buildFormulas d)
            ((buildFormulas d).getD col default))
          lbl = some (ruleIdx, slot, (buildFormulas d).idxOf φ) := by
  intro fuel
  induction fuel with
  | zero =>
      intro φ col lbl hstep
      simp [routeFrom] at hstep
  | succ fuel _ih =>
      intro φ col lbl hstep
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          simp only [routeFrom, hfind, List.getElem?_cons_zero] at hstep
          simp at hstep
      | some v =>
          have hvφ : v.FORMULA = φ :=
            of_decide_eq_true (List.find?_some (p := fun u => decide (u.FORMULA = φ)) (l := d.NODES) hfind)
          cases hout : get_rule.outgoing v d with
          | nil =>
              simp only [routeFrom, hfind, hout, List.getElem?_cons_zero] at hstep
              simp at hstep
          | cons e es =>
              simp only [routeFrom, hfind, hout, List.getElem?_cons_zero] at hstep
              have hcol_eq : col = (buildFormulas d).idxOf e.END.FORMULA := by
                have := congrArg Prod.fst (Option.some.inj hstep)
                simp at this
                omega
              have hlbl_eq : lbl = inputLabelForEdge d (buildFormulas d) φ e.END := by
                have := congrArg Prod.snd (Option.some.inj hstep)
                simpa using this.symm
              subst hcol_eq
              subst hlbl_eq
              have hv : v ∈ d.NODES := find?_some_mem hfind
              have heout : e ∈ get_rule.outgoing v d := by rw [hout]; simp
              have hw : e.END ∈ d.NODES :=
                (hvalid.hygiene.2.1 (mem_outgoing_mem_edges v d heout)).2
              have hsrcNode : sourceNodeAtColumn? d ((buildFormulas d).idxOf e.END.FORMULA) =
                  some e.END := sourceNodeAtColumn?_idxOf_of_mem d hinj hw
              obtain ⟨ruleIdx, hsel⟩ := ruleIndexForNode?_isSome_of_valid_node d hvalid hw
              obtain ⟨slot, hdec⟩ :=
                inputLabelForEdge_decodes_to_src d hvalid hv hw hvφ heout rfl hsel
              refine ⟨ruleIdx, slot, e.END, hsrcNode, hsel, ?_⟩
              have hgetD : (buildFormulas d).getD ((buildFormulas d).idxOf e.END.FORMULA) default =
                  e.END.FORMULA := by
                have hmemF : e.END.FORMULA ∈ buildFormulas d := by
                  unfold buildFormulas
                  exact List.mem_eraseDups.mpr (List.mem_map.mpr ⟨e.END, hw, rfl⟩)
                have hidxLt : (buildFormulas d).idxOf e.END.FORMULA < (buildFormulas d).length :=
                  List.idxOf_lt_length_of_mem hmemF
                simp [List.getD, List.getElem?_eq_getElem hidxLt, List.getElem_idxOf hidxLt]
              rw [hgetD]
              exact hdec

/--
 Consecutive steps of `routeFrom`: label at step `j+1` decodes to source =
    target of step `j`, with rule index pinned to the landing node at `col`.
-/
private lemma routeFrom_consec_label_decode_src (d : Graph) (hvalid : ValidDLDS d)
    (hinj : InjFormulas d) :
    ∀ (fuel : Nat) (φ : Formula) (j src col prevLbl lbl : Nat),
      (routeFrom d (buildFormulas d) fuel φ)[j]? = some (src + 1, prevLbl) →
      (routeFrom d (buildFormulas d) fuel φ)[j + 1]? = some (col + 1, lbl) →
      ∃ ruleIdx slot w,
        sourceNodeAtColumn? d col = some w ∧
        ruleIndexForNode? d (buildFormulas d) w = some ruleIdx ∧
        decodeInputLabel
          (buildIncomingMapForFormula (buildFormulas d)
            ((buildFormulas d).getD col default))
          lbl = some (ruleIdx, slot, src) := by
  intro fuel
  induction fuel with
  | zero =>
      intro φ j src col prevLbl lbl hstep _
      simp [routeFrom] at hstep
  | succ fuel ih =>
      intro φ j src col prevLbl lbl hstep hnext
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          cases j with
          | zero =>
              simp only [routeFrom, hfind, List.getElem?_cons_zero] at hstep
              simp at hstep
          | succ j =>
              have hstep' : (routeFrom d (buildFormulas d) fuel φ)[j]? = some (src + 1, prevLbl) := by
                simpa [routeFrom, hfind, Nat.succ_eq_add_one] using hstep
              have hnext' : (routeFrom d (buildFormulas d) fuel φ)[j + 1]? = some (col + 1, lbl) := by
                simpa [routeFrom, hfind, Nat.succ_eq_add_one, Nat.add_assoc] using hnext
              exact ih φ j src col prevLbl lbl hstep' hnext'
      | some v =>
          have hvφ : v.FORMULA = φ :=
            of_decide_eq_true (List.find?_some (p := fun u => decide (u.FORMULA = φ)) (l := d.NODES) hfind)
          cases hout : get_rule.outgoing v d with
          | nil =>
              cases j with
              | zero =>
                  simp only [routeFrom, hfind, hout, List.getElem?_cons_zero] at hstep
                  simp at hstep
              | succ j =>
                  have hstep' : (routeFrom d (buildFormulas d) fuel φ)[j]? = some (src + 1, prevLbl) := by
                    simpa [routeFrom, hfind, hout, Nat.succ_eq_add_one] using hstep
                  have hnext' : (routeFrom d (buildFormulas d) fuel φ)[j + 1]? = some (col + 1, lbl) := by
                    simpa [routeFrom, hfind, hout, Nat.succ_eq_add_one, Nat.add_assoc] using hnext
                  exact ih φ j src col prevLbl lbl hstep' hnext'
          | cons e es =>
              by_cases hminor : (match classifyRule? e.END d with
                  | some (DLDSRuleClass.elim _ minor) => decide (φ = minor.START.FORMULA)
                  | _ => false) = true
              ·
                have hr :=
                  routeFrom_minor_match_tail_replicate d (buildFormulas d) fuel
                    φ v e es hfind hout hminor
                cases j with
                | zero =>
                    have : (List.replicate fuel (0, 0))[0]? = some (col + 1, lbl) := by
                      have hnext' := hnext
                      rw [hr] at hnext'
                      simpa using hnext'
                    rw [List.getElem?_replicate] at this
                    split at this <;> simp at this
                | succ j =>
                    have : (List.replicate fuel (0, 0))[j]? = some (src + 1, prevLbl) := by
                      have hstep' := hstep
                      rw [hr] at hstep'
                      simpa using hstep'
                    rw [List.getElem?_replicate] at this
                    split at this <;> simp at this
              ·
                have hnotminor :
                    (match classifyRule? e.END d with
                    | some (DLDSRuleClass.elim _ minor) => decide (φ = minor.START.FORMULA)
                    | _ => false) = false := by
                  cases hval :
                      (match classifyRule? e.END d with
                      | some (DLDSRuleClass.elim _ minor) => decide (φ = minor.START.FORMULA)
                      | _ => false) <;> simp [hval] at hminor ⊢
                have hr :=
                  routeFrom_nonminor_tail_continues d (buildFormulas d) fuel
                    φ v e es hfind hout hnotminor
                cases j with
                | zero =>
                    have hsrc : src = (buildFormulas d).idxOf e.END.FORMULA := by
                      have hhead : some ((buildFormulas d).idxOf e.END.FORMULA + 1,
                            inputLabelForEdge d (buildFormulas d) φ e.END) =
                          some (src + 1, prevLbl) := by
                        rw [hr] at hstep
                        simpa using hstep
                      have heq := congrArg Prod.fst (Option.some.inj hhead)
                      simp at heq
                      omega
                    have hnext0 : (routeFrom d (buildFormulas d) fuel e.END.FORMULA)[0]? =
                        some (col + 1, lbl) := by
                      have hnext' := hnext
                      rw [hr] at hnext'
                      simpa [Nat.succ_eq_add_one] using hnext'
                    have hstep0 :=
                      routeFrom_step0_label_decode_src d hvalid hinj fuel e.END.FORMULA col lbl hnext0
                    rw [hsrc]
                    exact hstep0
                | succ j =>
                    have hstep' : (routeFrom d (buildFormulas d) fuel e.END.FORMULA)[j]? =
                        some (src + 1, prevLbl) := by
                      have hstep'' := hstep
                      rw [hr] at hstep''
                      simpa [Nat.succ_eq_add_one] using hstep''
                    have hnext' : (routeFrom d (buildFormulas d) fuel e.END.FORMULA)[j + 1]? =
                        some (col + 1, lbl) := by
                      have hnext'' := hnext
                      rw [hr] at hnext''
                      simpa [Nat.succ_eq_add_one, Nat.add_assoc] using hnext''
                    exact ih e.END.FORMULA j src col prevLbl lbl hstep' hnext'

/--
 **G2 core ; `route_label_decodes_node`.**  At any live route state at
    `depth > 0`, the carried label decodes at column `col` to a triple whose rule
    index is pinned to the rule applied at the column's node (`ruleIndexForNode?`)
    and whose source is exactly the recorded `src`. Exposing the landing node `w`
    is what makes `decodedRuleAtColumn?` uniform across all arrivals at a column.
-/
lemma route_label_decodes_node (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth origin col src lbl : Nat} {φ : Formula}
    (hpos : 0 < depth)
    (hstate : routeStateAfter (pathsFromDLDS d) origin depth = some (col, src, lbl))
    (hcol : (buildFormulas d)[col]? = some φ) :
    ∃ ruleIdx slot w,
      sourceNodeAtColumn? d col = some w ∧
      ruleIndexForNode? d (buildFormulas d) w = some ruleIdx ∧
      decodeInputLabel (buildIncomingMapForFormula (buildFormulas d) φ) lbl =
        some (ruleIdx, slot, src) := by
  obtain ⟨n, rfl⟩ : ∃ n, depth = n + 1 := ⟨depth - 1, by omega⟩
  obtain ⟨cur_n, _src_n, _lbl_n, steps, target_n, inputLabel_n,
    hprev, hsteps, hstep_n, htarget_n, htriple⟩ :=
    routeStateAfter_live_succ (pathsFromDLDS d) hstate
  simp only [Prod.mk.injEq] at htriple
  obtain ⟨hcol_eq, hsrc_eq, hlbl_eq⟩ := htriple
  have hstep_col : steps[n]? = some (col + 1, lbl) := by
    rw [hstep_n]; simp only [Option.some.injEq, Prod.mk.injEq]
    exact ⟨by omega, hlbl_eq.symm⟩
  have horiginPath : origin < (pathsFromDLDS d).length := getElem?_some_lt hsteps
  have horiginF : origin < (buildFormulas d).length := by
    simpa [pathsFromDLDS_length] using horiginPath
  let φ_origin := (buildFormulas d).get ⟨origin, horiginF⟩
  have hφ_origin : (buildFormulas d)[origin]? = some φ_origin :=
    List.getElem?_eq_getElem horiginF
  have hidx_origin : (buildFormulas d).idxOf φ_origin = origin :=
    indexOf_eq_of_get horiginF (buildFormulas_nodup d) rfl
  have hentry := pathsFromDLDS_get?_of_formula d hφ_origin
  have hgetD_col : (buildFormulas d).getD col default = φ := by simp [List.getD, hcol]
  cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ_origin)) with
  | none =>
      have hsteps_eq : steps = List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) := by
        rw [hentry] at hsteps
        exact (Option.some.inj (by simpa [hfind] using hsteps)).symm
      exact False.elim (replicate_zero_get_nonzero_false
        (by simpa [hsteps_eq] using hstep_col) (by omega))
  | some v =>
      have hvφ : v.FORMULA = φ_origin := of_decide_eq_true
        (List.find?_some (p := fun u => decide (u.FORMULA = φ_origin)) (l := d.NODES) hfind)
      have hv : v ∈ d.NODES := find?_some_mem hfind
      by_cases hhyp : v.HYPOTHESIS = true
      · let D := (d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL
        let rest := routeFrom d (buildFormulas d)
            ((buildGridFromDLDS d).length - 1 - D) φ_origin
        have hsteps_eq : steps =
            List.replicate D ((buildFormulas d).idxOf φ_origin + 1, 0) ++ rest := by
          rw [hentry] at hsteps
          exact (Option.some.inj (by simpa [hfind, hhyp, D, rest] using hsteps)).symm
        have hsteps_origin : steps = List.replicate D (origin + 1, 0) ++ rest := by
          simp [hsteps_eq, hidx_origin]
        have hentry_path : (pathsFromDLDS d)[origin]? =
            some (List.replicate D (origin + 1, 0) ++ rest) := by
          rw [hsteps]; simp [hsteps_eq, hidx_origin]
        by_cases hdelay : n < D
        ·
          have hstep_delay : steps[n]? = some (origin + 1, 0) := by
            rw [hsteps_origin, List.getElem?_append_left (by simpa using hdelay),
                List.getElem?_replicate]; simp [hdelay]
          rw [hstep_delay] at hstep_col
          simp only [Option.some.injEq, Prod.mk.injEq] at hstep_col
          have hcol_orig : col = origin := by omega
          have hlbl_zero : lbl = 0 := hstep_col.2.symm
          have hsrc_delay : cur_n = origin := by
            have hself := routeStateAfter_replicate_prefix_self
              (pathsFromDLDS d) origin D rest horiginPath hentry_path n (by omega)
            rw [hprev] at hself
            exact (congrArg (fun p : Nat × Nat × Nat => p.1) (Option.some.inj hself))
          have hφφ : φ = φ_origin := by
            have : (buildFormulas d)[col]? = some φ_origin := by rw [hcol_orig]; exact hφ_origin
            exact Option.some.inj (hcol.symm.trans this)
          have hidxv : (buildFormulas d).idxOf v.FORMULA = col := by
            rw [hvφ, hidx_origin]; exact hcol_orig.symm
          have hsrcNode : sourceNodeAtColumn? d col = some v := by
            rw [← hidxv]; exact sourceNodeAtColumn?_idxOf_of_mem d htree.2 hv
          have hclass : classifyRule? v d = some DLDSRuleClass.hypothesis := by
            simp [classifyRule?, hhyp]
          have hincLen : 0 < (buildIncomingMapForFormula (buildFormulas d) φ).length := by
            unfold buildIncomingMapForFormula; cases φ <;> simp
          have hvφ' : v.FORMULA = φ := hvφ.trans hφφ.symm
          have hsel : ruleIndexForNode? d (buildFormulas d) v =
              some ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) := by
            unfold ruleIndexForNode?
            rw [hclass, hvφ']
            rw [if_pos hincLen]
          have hdec0 := decodeInputLabel_zero_buildIncomingMapForFormula (buildFormulas d) φ
          have hidxφ : (buildFormulas d).idxOf φ = col := by
            rw [hφφ, hidx_origin]; exact hcol_orig.symm
          refine ⟨(buildIncomingMapForFormula (buildFormulas d) φ).length - 1, 0, v,
            hsrcNode, hsel, ?_⟩
          rw [hlbl_zero, hdec0, hidxφ]
          have : src = col := by rw [hsrc_eq, hsrc_delay, hcol_orig]
          rw [this]
        ·
          have hge : D ≤ n := by omega
          have hstep_rest : rest[n - D]? = some (col + 1, lbl) := by
            have hright := getElem?_append_right_sub
              (List.replicate D (origin + 1, 0)) rest (i := n)
              (by simpa [List.length_replicate] using hge)
            rw [hsteps_origin] at hstep_col
            rw [hright] at hstep_col
            simpa [List.length_replicate] using hstep_col
          by_cases hn_D : n = D
          ·
            have hrest0 : rest[0]? = some (col + 1, lbl) := by
              simpa [hn_D, Nat.sub_self] using hstep_rest
            have hsrc_orig : cur_n = origin := by
              by_cases hD0 : D = 0
              · have hn0 : n = 0 := by rw [hn_D]; exact hD0
                rw [hn0] at hprev
                simp only [routeStateAfter, horiginPath, ↓reduceIte,
                  Option.some.injEq, Prod.mk.injEq] at hprev
                exact hprev.1.symm
              · have hself := routeStateAfter_replicate_prefix_self
                  (pathsFromDLDS d) origin D rest horiginPath hentry_path n (by omega)
                rw [hprev] at hself
                exact (congrArg (fun p : Nat × Nat × Nat => p.1) (Option.some.inj hself))
            obtain ⟨ruleIdx', slot', w', hsrcNode', hsel', hdec'⟩ :=
              routeFrom_step0_label_decode_src d hvalid htree.2 _ φ_origin col lbl hrest0
            rw [hgetD_col] at hdec'
            refine ⟨ruleIdx', slot', w', hsrcNode', hsel', ?_⟩
            have : (buildFormulas d).idxOf φ_origin = src := by
              rw [hidx_origin, ← hsrc_orig, ← hsrc_eq]
            rw [this] at hdec'; exact hdec'
          ·
            have hn_gt_D : D < n := by omega
            have hn_pos : 0 < n := by omega
            obtain ⟨cur_nm1, _src_nm1, _lbl_nm1, steps2, target_nm1, inputLabel_nm1,
              hprevprev, hsteps2, hstep_nm1, htarget_nm1, htriple2⟩ :=
              routeStateAfter_live_succ (pathsFromDLDS d)
                (show routeStateAfter (pathsFromDLDS d) origin ((n - 1) + 1) =
                    some (cur_n, _src_n, _lbl_n) from
                  by rwa [Nat.sub_add_cancel hn_pos])
            have hcur_n_eq : cur_n = target_nm1 - 1 := by
              simp only [Prod.mk.injEq] at htriple2; exact htriple2.1
            have htarget_nm1_succ : target_nm1 = cur_n + 1 := by omega
            have hsteps2_eq : steps2 = steps := by
              rw [hsteps] at hsteps2; exact (Option.some.inj hsteps2).symm
            subst steps2
            have hstep_nm1_col : steps[n - 1]? = some (cur_n + 1, inputLabel_nm1) := by
              rw [hstep_nm1]; simp [htarget_nm1_succ]
            have hstep_nm1_rest : rest[n - D - 1]? = some (cur_n + 1, inputLabel_nm1) := by
              have hright := getElem?_append_right_sub
                (List.replicate D (origin + 1, 0)) rest (i := n - 1)
                (by simp only [List.length_replicate]; omega)
              rw [hsteps_origin] at hstep_nm1_col
              rw [hright] at hstep_nm1_col
              simp only [List.length_replicate] at hstep_nm1_col
              convert hstep_nm1_col using 2; omega
            obtain ⟨ruleIdx', slot', w', hsrcNode', hsel', hdec'⟩ :=
              routeFrom_consec_label_decode_src d hvalid htree.2 _ φ_origin
                (n - D - 1) cur_n col inputLabel_nm1 lbl
                hstep_nm1_rest (by rwa [show n - D - 1 + 1 = n - D from by omega])
            rw [hgetD_col] at hdec'
            refine ⟨ruleIdx', slot', w', hsrcNode', hsel', ?_⟩
            rw [hsrc_eq]; exact hdec'
      ·
        have hsteps_eq : steps = List.replicate ((buildGridFromDLDS d).length - 1) (0, 0) := by
          rw [hentry] at hsteps
          exact (Option.some.inj (by simpa [hfind, hhyp] using hsteps)).symm
        exact False.elim (replicate_zero_get_nonzero_false
          (by simpa [hsteps_eq] using hstep_col) (by omega))

/--
 **G2 ; `route_label_source_agree`.**  At any live route state, if the label
    `lbl` decodes at column `col` to `(ruleIdx, slot, decodedSrc)`, then
    `decodedSrc = src`. Depth 0 is the rep self-token; depth > 0 reuses
    `route_label_decodes_node`.
-/
lemma route_label_source_agree (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth origin col src lbl ruleIdx slot decodedSrc : Nat} {φ : Formula}
    (hstate : routeStateAfter (pathsFromDLDS d) origin depth = some (col, src, lbl))
    (hcol : (buildFormulas d)[col]? = some φ)
    (hdec : decodeInputLabel (buildIncomingMapForFormula (buildFormulas d) φ) lbl =
              some (ruleIdx, slot, decodedSrc)) :
    decodedSrc = src := by
  cases depth with
  | zero =>
      simp only [routeStateAfter] at hstate
      by_cases hlt : origin < (pathsFromDLDS d).length
      · simp only [hlt, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hstate
        obtain ⟨hcol_eq, hsrc_eq, hlbl_eq⟩ := hstate
        rw [← hcol_eq] at hcol
        rw [← hlbl_eq] at hdec
        have hdec0 := decodeInputLabel_zero_buildIncomingMapForFormula (buildFormulas d) φ
        rw [hdec0] at hdec
        have hdecodedSrc : decodedSrc = (buildFormulas d).idxOf φ := by
          have heq := Option.some.inj hdec
          simp only [Prod.mk.injEq] at heq
          exact heq.2.2.symm
        have hcol_lt : origin < (buildFormulas d).length := getElem?_some_lt hcol
        have hget : (buildFormulas d).get ⟨origin, hcol_lt⟩ = φ :=
          (List.getElem?_eq_some_iff.mp hcol).2
        have hidx : (buildFormulas d).idxOf φ = origin :=
          indexOf_eq_of_get hcol_lt (buildFormulas_nodup d) hget
        rw [hdecodedSrc, hidx]; exact hsrc_eq
      · simp [hlt] at hstate
  | succ n =>
      obtain ⟨ruleIdx', slot', w', _hw', _hsel', hdec'⟩ :=
        route_label_decodes_node d htree hvalid (by omega) hstate hcol
      have heq := Option.some.inj (hdec.symm.trans hdec')
      simp only [Prod.mk.injEq] at heq
      exact heq.2.2

/--
 For a non-hypothesis node `w`, the applied rule index is never the final
    (repetition) index `length - 1`: intro sits at 0 with `length ≥ 2`, and elim
    sits at an entry of width 2 whereas the repetition entry has width 1.
-/
lemma ruleIndexForNode?_succ_ne_length_of_nonhyp (d : Graph) {w : Vertex} {r : Nat}
    (hnhyp : classifyRule? w d ≠ some DLDSRuleClass.hypothesis)
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some r) :
    r + 1 ≠ (buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length := by
  unfold ruleIndexForNode? at hsel
  cases hclass : classifyRule? w d with
  | none => rw [hclass] at hsel; simp at hsel
  | some cls =>
      cases cls with
      | hypothesis => exact absurd hclass hnhyp
      | intro p =>
          simp only [hclass] at hsel
          obtain ⟨A, B, hform⟩ := classifyRule?_intro_formula_implication hclass
          rw [hform] at hsel
          simp only [Option.some.injEq] at hsel
          subst hsel
          rw [hform]
          simp only [buildIncomingMapForFormula, List.length_append, List.length_cons,
            List.length_nil]
          omega
      | elim major minor =>
          simp only [hclass] at hsel
          cases hpos : elimRulePosition? (buildFormulas d) w.FORMULA major.START.FORMULA with
          | none => rw [hpos] at hsel; simp at hsel
          | some pos =>
              rw [hpos] at hsel
              simp only [Option.some.injEq] at hsel
              subst hsel
              intro hcontra
              have hentry2 := elimRulePosition?_indexes_elim_entry
                (buildFormulas d) w.FORMULA major.START.FORMULA pos hpos
              have hrep1 := buildIncomingMapForFormula_last_rep_length (buildFormulas d) w.FORMULA
              have hidxeq : introRuleCount w.FORMULA + pos =
                  (buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length - 1 := by omega
              rw [hidxeq] at hentry2
              rw [hentry2] at hrep1
              exact absurd hrep1 (by decide)

/--
 If a node's applied rule index is the final (repetition) index, the node is a
    hypothesis. Contrapositive of `ruleIndexForNode?_succ_ne_length_of_nonhyp`.
-/
lemma classifyRule?_hyp_of_ruleIndex_last (d : Graph) {w : Vertex}
    (hsel : ruleIndexForNode? d (buildFormulas d) w =
      some ((buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length - 1)) :
    classifyRule? w d = some DLDSRuleClass.hypothesis := by
  by_contra hnhyp
  have hne := ruleIndexForNode?_succ_ne_length_of_nonhyp d hnhyp hsel
  have hpos : 0 < (buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length := by
    unfold buildIncomingMapForFormula; cases w.FORMULA <;> simp
  exact hne (by omega)

/--
 **Rule uniqueness per column.** At `depth > 0`, every origin arriving at `col`
    pins `decodedRuleAtColumn? d depth col` to the rule index of the column's node.
-/
lemma decodedRuleAtColumn?_eq_node (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth col origin ruleIdx : Nat} {wv : Vertex}
    (hpos : 0 < depth)
    (hmem : origin ∈ arrivingOriginsAt d depth col)
    (hw : sourceNodeAtColumn? d col = some wv)
    (hsel : ruleIndexForNode? d (buildFormulas d) wv = some ruleIdx) :
    decodedRuleAtColumn? d depth col = some ruleIdx := by
  have horigin := mem_arrivingOriginsAt_origin_lt hmem
  obtain ⟨src, lbl, hstate⟩ := mem_arrivingOriginsAt_state hmem
  have hpred_origin :
      (fun o => match routeStateAfter (pathsFromDLDS d) o depth with
        | some (current, _, _) => current == col
        | none => false) origin = true := by
    simp only; rw [hstate]; simp
  rcases hdc : decodedRuleAtColumn? d depth col with _ | r
  · -- The origin supplies a successful witness for `find?`.
    exfalso
    rcases hfs : (List.range (buildFormulas d).length).find? (fun o =>
        match routeStateAfter (pathsFromDLDS d) o depth with
        | some (current, _, _) => current == col
        | none => false) with _ | origin0
    · exact absurd hpred_origin
        (List.find?_eq_none.mp hfs origin (List.mem_range.mpr horigin))
    · have hpred0 := List.find?_some hfs
      have horigin0 : origin0 < (buildFormulas d).length :=
        List.mem_range.mp (find?_some_mem hfs)
      rcases hstate0 : routeStateAfter (pathsFromDLDS d) origin0 depth with _ | ⟨cur0, src0, lbl0⟩
      · rw [hstate0] at hpred0; simp at hpred0
      · have hcur0 : cur0 = col := by
          rw [hstate0] at hpred0; simpa [BEq.beq, decide_eq_true_eq] using hpred0
        rw [hcur0] at hstate0
        have hcol_lt : col < (buildFormulas d).length :=
          routeStateAfter_pathsFromDLDS_current_lt d htree hvalid horigin0 hstate0
        obtain ⟨φ, hφ⟩ : ∃ φ, (buildFormulas d)[col]? = some φ :=
          ⟨_, List.getElem?_eq_getElem hcol_lt⟩
        obtain ⟨ruleIdx0, slot0, w0, _hw0, _hsel0, hdec0⟩ :=
          route_label_decodes_node d htree hvalid hpos hstate0 hφ
        have hsomeDec : decodedRuleAtColumn? d depth col = some ruleIdx0 := by
          unfold decodedRuleAtColumn?
          change
            (match (List.range (buildFormulas d).length).find? (fun o =>
              match routeStateAfter (pathsFromDLDS d) o depth with
              | some (current, _, _) => current == col
              | none => false) with
            | none => none
            | some origin =>
                match routeStateAfter (pathsFromDLDS d) origin depth with
                | none => none
                | some (_, _, label) =>
                    match (buildFormulas d)[col]? with
                    | none => none
                    | some φ =>
                        match decodeInputLabel
                            (buildIncomingMapForFormula (buildFormulas d) φ) label with
                        | some (ruleIdx, _, _) => some ruleIdx
                        | none => none) = some ruleIdx0
          rw [hfs]
          simp [hstate0, hφ, hdec0]
        rw [hsomeDec] at hdc
        contradiction
  ·
    obtain ⟨o0, s0, l0, φ0, slot0, src0, ho0lt, hstate0, hφ0, hdec0⟩ :=
      decodedRuleAtColumn?_some_route hdc
    obtain ⟨r', slot', w', hw', hsel', hdec'⟩ :=
      route_label_decodes_node d htree hvalid hpos hstate0 hφ0
    have hrr' : r = r' := by
      have hh := hdec0.symm.trans hdec'
      simp only [Option.some.injEq, Prod.mk.injEq] at hh
      exact hh.1
    have hw'eq : w' = wv := Option.some.inj (hw'.symm.trans hw)
    have hr'rule : r' = ruleIdx := by
      rw [hw'eq] at hsel'; exact Option.some.inj (hsel'.symm.trans hsel)
    rw [hrr', hr'rule]

/--
 At `depth > 0`, every origin arriving
    at `col` lies in the expected carrier list of the column's decoded rule.
    Non-rep slots: the decoded source is a real premise column and `origin` is its
    principal carrier (`mem_expectedCarrierOriginsForRule?_of_nonrep_source`). Rep
    slot: the column is a hypothesis carrying itself, so `origin = col ∈ [col]`.
-/
lemma no_extra_origin (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth col origin ruleIdx : Nat} {expected : List Nat}
    (hpos : 0 < depth)
    (hdec : decodedRuleAtColumn? d depth col = some ruleIdx)
    (hexp : expectedCarrierOriginsForRule? d col ruleIdx = some expected)
    (hmem : origin ∈ arrivingOriginsAt d depth col) :
    origin ∈ expected := by
  have horigin := mem_arrivingOriginsAt_origin_lt hmem
  obtain ⟨src, lbl, hstate⟩ := mem_arrivingOriginsAt_state hmem
  have hcol_lt : col < (buildFormulas d).length :=
    routeStateAfter_pathsFromDLDS_current_lt d htree hvalid horigin hstate
  obtain ⟨φ, hφ⟩ : ∃ φ, (buildFormulas d)[col]? = some φ :=
    ⟨_, List.getElem?_eq_getElem hcol_lt⟩
  obtain ⟨ruleIdx', slot, w, hw, hsel, hdecμ⟩ :=
    route_label_decodes_node d htree hvalid hpos hstate hφ
  have hdec_eq : decodedRuleAtColumn? d depth col = some ruleIdx' :=
    decodedRuleAtColumn?_eq_node d htree hvalid hpos hmem hw hsel
  have hridx : ruleIdx' = ruleIdx := Option.some.inj (hdec_eq.symm.trans hdec)
  subst hridx
  have hprincipal : principalCarrierForSourceColumn? d src = some origin :=
    arriving_state_positive_principal_source d htree hvalid hpos hstate
  have hwφ : w.FORMULA = φ := by
    have hw' := hw
    simp only [sourceNodeAtColumn?, hφ] at hw'
    exact of_decide_eq_true
      (List.find?_some (p := fun v => decide (v.FORMULA = φ)) (l := d.NODES) hw')
  cases lbl with
  | succ k =>
      obtain ⟨inc, hentry, hsrc_mem⟩ := decodeInputLabel_succ_source_mem hdecμ
      have hnonrep := decodeInputLabel_succ_nonrep hdecμ
      exact mem_expectedCarrierOriginsForRule?_of_nonrep_source d hφ hentry hnonrep
        hsrc_mem hprincipal hexp
  | zero =>
      rw [decodeInputLabel_zero_buildIncomingMapForFormula] at hdecμ
      have heq := Option.some.inj hdecμ
      simp only [Prod.mk.injEq] at heq
      obtain ⟨hr_last, _, hsrc_idx⟩ := heq
      have hidxφ : (buildFormulas d).idxOf φ = col :=
        indexOf_eq_of_get hcol_lt (buildFormulas_nodup d) ((List.getElem?_eq_some_iff.mp hφ).2)
      have hsrc_col : src = col := by rw [← hsrc_idx, hidxφ]
      have hsel_last : ruleIndexForNode? d (buildFormulas d) w =
          some ((buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length - 1) := by
        rw [hwφ, hr_last]; exact hsel
      have hhyp := classifyRule?_hyp_of_ruleIndex_last d hsel_last
      have hcolw : (buildFormulas d)[col]? = some w.FORMULA := by rw [hwφ]; exact hφ
      have hprincipal_col : principalCarrierForSourceColumn? d col = some col :=
        principalCarrierForSourceColumn?_hyp d col w hw hcolw hhyp
      have horigin_col : origin = col := by
        rw [hsrc_col] at hprincipal
        exact (Option.some.inj (hprincipal.symm.trans hprincipal_col))
      have hrep := expectedCarrierOriginsForRule?_rep d hφ
      rw [← hr_last] at hexp
      have hexp_eq : expected = [col] := Option.some.inj (hexp.symm.trans hrep)
      rw [hexp_eq, horigin_col]; simp

lemma carrier_arrival_from_expected_rep (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {depth col origin ruleIdx : Nat} {expected : List Nat}
    (hpos : 0 < depth)
    (hdec : decodedRuleAtColumn? d depth col = some ruleIdx)
    (hexp : expectedCarrierOriginsForRule? d col ruleIdx = some expected)
    (hexp_eq : expected = [col])
    (hmem : origin ∈ expected) :
    origin ∈ arrivingOriginsAt d depth col := by
  obtain ⟨origin0, source0, label0, _φ, _slot, _src,
    horigin0, hstate0, _hφ, _hlabel⟩ :=
    decodedRuleAtColumn?_some_route hdec
  have harr0 : origin0 ∈ arrivingOriginsAt d depth col :=
    arrivingOriginsAt_mem_of_state horigin0 hstate0
  have horigin0_exp : origin0 ∈ expected :=
    no_extra_origin d htree hvalid hpos hdec hexp harr0
  have horigin0_col : origin0 = col := by
    rw [hexp_eq] at horigin0_exp
    simpa using horigin0_exp
  have horigin_col : origin = col := by
    rw [hexp_eq] at hmem
    simpa using hmem
  subst origin
  rwa [horigin0_col] at harr0


#print axioms tree_bridge_forward
#print axioms tree_bridge_forward_of_descent_coherent
end Semantic
