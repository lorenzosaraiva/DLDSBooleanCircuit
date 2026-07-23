import Semantic.FlowCollapsedProof
import Semantic.FlowTreeProof
set_option linter.unusedSimpArgs false

/-!
# FlowEdgeDepChar ; the strengthened route-edge dependency characterization

This file derives the route-edge dependency characterization without assuming
the conclusion `b = premiseDepUnion`:
`routeEdgeDepChar`, proved by `countAbove` induction, where the per-premise
dependency equality is supplied by the induction hypothesis.

The four vetted construction invariants are carried as hypotheses (their
construction proofs from greedy DGTD / Def-11 are deferred):
  `OneEdgePerColourPerNode`   [out-side uniqueness]   ; reused
  `RouteFanUnique`            [in-side residual-head] ; reused
  `FaithfulDecoration`        [Def-11 rule-cased `sourceDepSet`]
  `RouteHeadCoverage`         [existence counterpart to (i)]
plus the structural Def-23 conjuncts (`structuralValid`: leveling, hygiene,
per-kernel-node ⊃E/⊃I shape), `ReseedFree`, and:
  `d.EDGES.Nodup`             [List-vs-set artifact: Def-12 edge sets]
Preservation of edge `Nodup` by `compress_nodes_graph` is external to this
development.

`flowRuleCorrect_collapsed` is derived as a COROLLARY.
-/

open Semantic

namespace FlowSpec

/-!
`sourceDepSet d v` is the dependency Def-11 stamps on `v`'s outgoing edge,
matching `outDep`'s casing ; EXCEPT the ⊃E arm is routed through `elimPairsAt`
(so it fires at collapsed nodes, where `classifyRule?` returns `none`). On the
⊃E arm it is exactly `premiseDepUnion d v`.
-/
def sourceDepSet (d : Graph) (v : Vertex) : Dep :=
  if (get_rule.incoming v d).isEmpty || v.HYPOTHESIS then [v.FORMULA]
  else match elimPairsAt d v with
    | [(_M, _S)] => premiseDepUnion d v
    | _ =>
      match classifyRule? v d with
      | some (DLDSRuleClass.intro p) =>
          match antecedent? v.FORMULA with
          | some α => depRemove p.DEPENDENCY α
          | none   => p.DEPENDENCY
      | _ => []

/--
 **FaithfulDecoration** (generalized to all nodes): every outgoing edge of
    `v` carries the Def-11 source label `sourceDepSet d v`.
-/
def FaithfulDecoration (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, ∀ e ∈ get_rule.outgoing v d, e.DEPENDENCY = sourceDepSet d v

/--
 **RouteHeadCoverage** (4th invariant): at every node with an outgoing edge,
    the route head colours and the outgoing edge colours coincide ; every route's
    head colour is realized by an outgoing edge (⊃ item-1 existence) AND every
    outgoing edge's colour is some route's head (⊃ item-2: `Φᵢ = ∅ ⇒ no colour-i
    edge`). The second direction is eval-true on d11' 10/11/16; it is the
    construction's "no spurious edge" guarantee.
-/
def RouteHeadCoverage (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] →
    (∀ r ∈ flowAt d (stdFuel d) v,
        ∃ e ∈ get_rule.outgoing v d, e.COLOUR = headColour r.2) ∧
    (∀ e ∈ get_rule.outgoing v d,
        ∃ r ∈ flowAt d (stdFuel d) v, headColour r.2 = e.COLOUR)



/--
 `v` applies ⊃E: a unique major/minor premise pair and a non-empty premise
    set.
-/
def NodeElimShape (d : Graph) (v : Vertex) : Prop :=
  ∃ M S, v.HYPOTHESIS = false ∧ (predsOf d v).isEmpty = false ∧
    elimPairsAt d v = [(M, S)]

/--
 `v` applies ⊃I: a single premise `u` proving the consequent of an
    implication `v.FORMULA = α ⊃ β`, with no ⊃E pairing.
-/
def NodeIntroShape (d : Graph) (v : Vertex) : Prop :=
  ∃ (p : Deduction) (α β : Formula), v.HYPOTHESIS = false ∧
    get_rule.incoming v d = [p] ∧
    v.FORMULA = Formula.implication α β ∧ p.START.FORMULA = β ∧
    elimPairsAt d v = []

/--
 **structuralValid**: the genuine structural conjuncts only ; leveling
    (premises one level up), hygiene (edge endpoints are nodes), and the pure
    ⊃E/⊃I rule shape at each kernel node with an outgoing edge. NO list-model
    edge nodup artifact, NO Φᵢ-singleton, NO per-colour uniqueness, NO
    dependency-equality clause.
-/
def structuralValid (d : Graph) : Prop :=
  (∀ e ∈ d.EDGES, e.START.LEVEL = e.END.LEVEL + 1) ∧
  (∀ e ∈ d.EDGES, e.START ∈ d.NODES ∧ e.END ∈ d.NODES) ∧
  (∀ v ∈ d.NODES, get_rule.incoming v d ≠ [] → get_rule.outgoing v d ≠ [] →
     NodeElimShape d v ∨ NodeIntroShape d v)

/-!     are congruences for it)  -/

lemma mem_depUnion {a b : Dep} {x : Formula} :
    x ∈ depUnion a b ↔ x ∈ a ∨ x ∈ b := by
  unfold depUnion
  rw [List.mem_eraseDups, List.mem_append]

lemma mem_depRemove {a : Dep} {α x : Formula} :
    x ∈ depRemove a α ↔ x ∈ a ∧ x ≠ α := by
  unfold depRemove
  rw [List.mem_eraseDups, List.mem_filter]
  constructor
  · rintro ⟨hx, hne⟩
    exact ⟨hx, by simpa using hne⟩
  · rintro ⟨hx, hne⟩
    exact ⟨hx, by simpa using hne⟩

/--  `depSetEq a b` reflects mutual membership.  -/
lemma depSetEq_iff {a b : Dep} :
    depSetEq a b = true ↔ (∀ x ∈ a, x ∈ b) ∧ (∀ x ∈ b, x ∈ a) := by
  unfold depSetEq
  rw [Bool.and_eq_true, List.all_eq_true, List.all_eq_true]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun x hx => ?_, fun x hx => ?_⟩
    · have := h1 x hx
      rw [List.any_eq_true] at this
      obtain ⟨y, hy, hyx⟩ := this
      have : y = x := by simpa using hyx
      exact this ▸ hy
    · have := h2 x hx
      rw [List.any_eq_true] at this
      obtain ⟨y, hy, hyx⟩ := this
      have : y = x := by simpa using hyx
      exact this ▸ hy
  · rintro ⟨h1, h2⟩
    refine ⟨fun x hx => ?_, fun x hx => ?_⟩
    · rw [List.any_eq_true]
      exact ⟨x, h1 x hx, by simp⟩
    · rw [List.any_eq_true]
      exact ⟨x, h2 x hx, by simp⟩

lemma depSetEq_depUnion_congr {a a' b b' : Dep}
    (ha : depSetEq a a' = true) (hb : depSetEq b b' = true) :
    depSetEq (depUnion a b) (depUnion a' b') = true := by
  rw [depSetEq_iff] at ha hb ⊢
  obtain ⟨ha1, ha2⟩ := ha
  obtain ⟨hb1, hb2⟩ := hb
  constructor
  · intro x hx
    rw [mem_depUnion] at hx ⊢
    rcases hx with h | h
    · exact Or.inl (ha1 x h)
    · exact Or.inr (hb1 x h)
  · intro x hx
    rw [mem_depUnion] at hx ⊢
    rcases hx with h | h
    · exact Or.inl (ha2 x h)
    · exact Or.inr (hb2 x h)

lemma depSetEq_depRemove_congr {a a' : Dep} {α : Formula}
    (ha : depSetEq a a' = true) :
    depSetEq (depRemove a α) (depRemove a' α) = true := by
  rw [depSetEq_iff] at ha ⊢
  obtain ⟨ha1, ha2⟩ := ha
  constructor
  · intro x hx
    rw [mem_depRemove] at hx ⊢
    exact ⟨ha1 x hx.1, hx.2⟩
  · intro x hx
    rw [mem_depRemove] at hx ⊢
    exact ⟨ha2 x hx.1, hx.2⟩

/--  A crossing edge that consumes a residual has the residual's head colour.  -/
lemma consume_some_colour {e : Deduction} {p q : ColourPath}
    (h : consume e p = some q) : e.COLOUR = headColour p := by
  unfold consume at h
  unfold headColour
  cases p with
  | nil =>
      by_cases hc : e.COLOUR = 0
      · simp [hc]
      · simp [hc] at h
  | cons o p' =>
      by_cases hc : e.COLOUR = o
      · simp [hc]
      · simp [hc] at h



lemma mem_edgesBetween {d : Graph} {u v : Vertex} {e : Deduction}
    (h : e ∈ edgesBetween d u v) : e.START = u ∧ e.END = v ∧ e ∈ d.EDGES := by
  unfold edgesBetween at h
  rw [List.mem_filter] at h
  obtain ⟨hmem, hp⟩ := h
  have hp' := of_decide_eq_true hp
  exact ⟨hp'.1, hp'.2, hmem⟩

lemma mem_outgoing_of_edgesBetween {d : Graph} {u v : Vertex} {e : Deduction}
    (h : e ∈ edgesBetween d u v) : e ∈ get_rule.outgoing u d := by
  obtain ⟨hs, _, hmem⟩ := mem_edgesBetween h
  rw [outgoing_eq_filter, List.mem_filter]
  exact ⟨hmem, decide_eq_true hs⟩

lemma edgesBetween_ne_nil_of_mem {d : Graph} {u v : Vertex} {e : Deduction}
    (h : e ∈ edgesBetween d u v) : edgesBetween d u v ≠ [] := by
  intro hnil; rw [hnil] at h; exact List.not_mem_nil h



lemma preds_cons2_of_elimPairsAt {d : Graph} {v M S : Vertex}
    (h : elimPairsAt d v = [(M, S)]) :
    ∃ a b rest, predsOf d v = a :: b :: rest := by
  unfold elimPairsAt at h
  cases hp : predsOf d v with
  | nil => rw [hp] at h; simp at h
  | cons a tl =>
      cases htl : tl with
      | nil =>
          rw [hp, htl] at h
          simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil,
            List.filterMap_cons, List.filterMap_nil] at h
          rw [if_neg (formula_ne_implication_self a.FORMULA v.FORMULA)] at h
          simp at h
      | cons b rest => exact ⟨a, b, rest, rfl⟩

lemma ite_isEmpty_nil_eraseDups {α : Type _} [BEq α] (X : List α) :
    (if X.isEmpty then ([] : List α) else X).eraseDups = X.eraseDups := by
  cases h : X.isEmpty with
  | true => simp [h, List.isEmpty_iff.mp h]
  | false => simp [h]



lemma flowAt_kernel_elim_eq (d : Graph) {v M S : Vertex} (fuel : Nat)
    (hhyp : v.HYPOTHESIS = false)
    (hpredsne : (predsOf d v).isEmpty = false)
    (hpairs : elimPairsAt d v = [(M, S)]) :
    flowAt d (fuel + 1) v =
      (elimCombine d v M S (flowAt d fuel S) (flowAt d fuel M)).eraseDups := by
  obtain ⟨a, b, rest, hcons⟩ := preds_cons2_of_elimPairsAt hpairs
  unfold elimCombine
  rw [flowAt, hhyp]
  simp only [Bool.false_or, hpredsne, Bool.false_eq_true, if_false]
  rw [hpairs, hcons]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil,
    reduceCtorEq, reseedPairs]
  exact ite_isEmpty_nil_eraseDups _



lemma flowAt_kernel_intro_eq (d : Graph) {v u : Vertex} {α β : Formula}
    (fuel : Nat)
    (hhyp : v.HYPOTHESIS = false)
    (hpreds : predsOf d v = [u])
    (hform : v.FORMULA = Formula.implication α β)
    (hu : u.FORMULA = β)
    (helim : elimPairsAt d v = []) :
    flowAt d (fuel + 1) v =
      ((flowAt d fuel u).flatMap (fun bp =>
        (edgesBetween d u v).filterMap (fun e =>
          (consume e bp.2).map (fun q => (depRemove bp.1 α, q))))).eraseDups := by
  have hpredsne : (predsOf d v).isEmpty = false := by rw [hpreds]; rfl
  rw [flowAt, hhyp]
  simp only [Bool.false_or, hpredsne, Bool.false_eq_true, if_false]
  rw [helim, hpreds, hform]
  simp only [List.flatMap_nil, List.isEmpty_nil, if_true, reseedPairs,
    List.append_nil, hu]



lemma incoming_isEmpty_false_of_preds {d : Graph} {v : Vertex}
    (h : (predsOf d v).isEmpty = false) : (get_rule.incoming v d).isEmpty = false := by
  rw [predsOf_eq] at h
  cases hi : get_rule.incoming v d with
  | nil => rw [hi] at h; simp at h
  | cons e es => rfl

lemma incoming_isEmpty_of_preds {d : Graph} {v : Vertex}
    (h : (predsOf d v).isEmpty = true) : (get_rule.incoming v d).isEmpty = true := by
  rw [predsOf_eq] at h
  cases hi : get_rule.incoming v d with
  | nil => rfl
  | cons e es =>
      rw [hi] at h
      rw [List.isEmpty_iff] at h
      have hmem : e.START ∈ ((e :: es).map (fun x => x.START)).eraseDups :=
        List.mem_eraseDups.mpr (by simp)
      rw [h] at hmem
      simp at hmem

lemma sourceDepSet_top {d : Graph} {v : Vertex}
    (hguard : ((get_rule.incoming v d).isEmpty || v.HYPOTHESIS) = true) :
    sourceDepSet d v = [v.FORMULA] := by
  unfold sourceDepSet; rw [if_pos hguard]

lemma sourceDepSet_elim {d : Graph} {v M S : Vertex}
    (hinc : (get_rule.incoming v d).isEmpty = false)
    (hhyp : v.HYPOTHESIS = false)
    (hpairs : elimPairsAt d v = [(M, S)]) :
    sourceDepSet d v = premiseDepUnion d v := by
  unfold sourceDepSet
  rw [if_neg (by simp [hinc, hhyp]), hpairs]

lemma sourceDepSet_intro {d : Graph} {v : Vertex} {p : Deduction} {α β : Formula}
    (hhyp : v.HYPOTHESIS = false)
    (hincoming : get_rule.incoming v d = [p])
    (hform : v.FORMULA = Formula.implication α β)
    (hpβ : p.START.FORMULA = β)
    (helim : elimPairsAt d v = []) :
    sourceDepSet d v = depRemove p.DEPENDENCY α := by
  have hinc : (get_rule.incoming v d).isEmpty = false := by rw [hincoming]; rfl
  have hclass : classifyRule? v d = some (DLDSRuleClass.intro p) := by
    unfold classifyRule?
    simp only [hhyp, hincoming, hform, consequent?, hpβ, Bool.false_eq_true,
      if_false, if_true, reduceIte]
  unfold sourceDepSet
  rw [if_neg (by simp [hinc, hhyp]), helim, hclass, hform]
  simp only [antecedent?]



lemma premiseDepUnion_eq {d : Graph} {v M S : Vertex}
    (hfaithful : FaithfulDecoration d)
    (hSnode : S ∈ d.NODES) (hMnode : M ∈ d.NODES)
    (hpairs : elimPairsAt d v = [(M, S)])
    (hSne : edgesBetween d S v ≠ []) (hMne : edgesBetween d M v ≠ []) :
    premiseDepUnion d v = depUnion (sourceDepSet d S) (sourceDepSet d M) := by
  unfold premiseDepUnion
  obtain ⟨eS, tlS, hS⟩ := List.exists_cons_of_ne_nil hSne
  obtain ⟨eM, tlM, hM⟩ := List.exists_cons_of_ne_nil hMne
  have heS : eS ∈ edgesBetween d S v := by rw [hS]; exact List.mem_cons_self ..
  have heM : eM ∈ edgesBetween d M v := by rw [hM]; exact List.mem_cons_self ..
  have h1 : eS.DEPENDENCY = sourceDepSet d S :=
    hfaithful S hSnode eS (mem_outgoing_of_edgesBetween heS)
  have h2 : eM.DEPENDENCY = sourceDepSet d M :=
    hfaithful M hMnode eM (mem_outgoing_of_edgesBetween heM)
  simp only [hpairs, hS, hM, h1, h2]



theorem routeEdgeDepChar_aux (d : Graph) (hstruct : structuralValid d)
    (honepercol : OneEdgePerColourPerNode d) (hroutefan : RouteFanUnique d)
    (hfaithful : FaithfulDecoration d) (hcoverage : RouteHeadCoverage d)
    (hreseed : ReseedFree d) :
    ∀ fuel : Nat, ∀ v : Vertex, v ∈ d.NODES →
      get_rule.outgoing v d ≠ [] → countAbove d v < fuel →
        ∀ r ∈ flowAt d fuel v, ∀ e ∈ get_rule.outgoing v d,
          e.COLOUR = headColour r.2 → depSetEq r.1 e.DEPENDENCY = true := by
  obtain ⟨hlevel, hhygiene, hshape⟩ := hstruct
  intro fuel
  induction fuel with
  | zero => intro v _ _ hcount _ _ _ _ _; omega
  | succ fuel ih =>
    intro v hv hout hcount r hr e he _hcol
    have hedep : e.DEPENDENCY = sourceDepSet d v := hfaithful v hv e he
    by_cases htop : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
    ·
      have hguard : ((get_rule.incoming v d).isEmpty || v.HYPOTHESIS) = true := by
        rcases Bool.or_eq_true _ _ |>.mp htop with hh | hp
        · rw [hh]; simp
        · rw [incoming_isEmpty_of_preds hp]; simp
      have hrfst : r.1 = [v.FORMULA] := by
        rw [flowAt] at hr
        simp only [htop, if_true] at hr
        by_cases hanc : (ancestorsInto d v).isEmpty = true
        · rw [if_pos hanc] at hr
          rw [List.mem_singleton] at hr; rw [hr]
        · rw [if_neg hanc] at hr
          rw [List.mem_map] at hr
          obtain ⟨a, _, ha⟩ := hr; rw [← ha]
      rw [hrfst, hedep, sourceDepSet_top hguard]
      exact depSetEq_self _
    ·
      have hhyp : v.HYPOTHESIS = false := by
        cases hh : v.HYPOTHESIS with
        | false => rfl
        | true => exact absurd (by rw [hh]; simp) htop
      have hpredsne : (predsOf d v).isEmpty = false := by
        cases hp : (predsOf d v).isEmpty with
        | false => rfl
        | true => exact absurd (by rw [hp]; simp [hhyp]) htop
      have hinc : (get_rule.incoming v d).isEmpty = false :=
        incoming_isEmpty_false_of_preds hpredsne
      have hkernel : get_rule.incoming v d ≠ [] := by
        intro h; rw [h] at hinc; simp at hinc
      rcases hshape v hv hkernel hout with helimsh | hintrosh
      ·
        obtain ⟨M, S, _, _, hpairs⟩ := helimsh
        rw [flowAt_kernel_elim_eq d fuel hhyp hpredsne hpairs, List.mem_eraseDups] at hr
        unfold elimCombine at hr
        rw [List.mem_flatMap] at hr; obtain ⟨bp₁, hbp₁, hr⟩ := hr
        rw [List.mem_flatMap] at hr; obtain ⟨bp₂, hbp₂, hr⟩ := hr
        rw [List.mem_flatMap] at hr; obtain ⟨e₁, he₁, hr⟩ := hr
        rw [List.mem_filterMap] at hr; obtain ⟨e₂, he₂, hr⟩ := hr
        cases hc1 : consume e₁ bp₁.2 with
        | none => simp [hc1] at hr
        | some q₁ =>
          cases hc2 : consume e₂ bp₂.2 with
          | none => simp [hc1, hc2] at hr
          | some q₂ =>
            simp only [hc1, hc2] at hr
            by_cases hq : q₁ = q₂
            · rw [if_pos hq] at hr
              have hrval : r = (depUnion bp₁.1 bp₂.1, q₁) := by
                rw [Option.some_inj] at hr; exact hr.symm
              obtain ⟨hS1, hS2, hSe⟩ := mem_edgesBetween he₁
              obtain ⟨hM1, hM2, hMe⟩ := mem_edgesBetween he₂
              have hSnode : S ∈ d.NODES := hS1 ▸ (hhygiene e₁ hSe).1
              have hMnode : M ∈ d.NODES := hM1 ▸ (hhygiene e₂ hMe).1
              have hSout : e₁ ∈ get_rule.outgoing S d := mem_outgoing_of_edgesBetween he₁
              have hMout : e₂ ∈ get_rule.outgoing M d := mem_outgoing_of_edgesBetween he₂
              have hSoutne : get_rule.outgoing S d ≠ [] := by
                intro h; rw [h] at hSout; exact List.not_mem_nil hSout
              have hMoutne : get_rule.outgoing M d ≠ [] := by
                intro h; rw [h] at hMout; exact List.not_mem_nil hMout
              have hSlvl : S.LEVEL = v.LEVEL + 1 := by
                have := hlevel e₁ hSe; rw [hS1, hS2] at this; exact this
              have hMlvl : M.LEVEL = v.LEVEL + 1 := by
                have := hlevel e₂ hMe; rw [hM1, hM2] at this; exact this
              have hSca : countAbove d S < fuel := by
                have := countAbove_lt_of_level_succ d hSnode hSlvl; omega
              have hMca : countAbove d M < fuel := by
                have := countAbove_lt_of_level_succ d hMnode hMlvl; omega
              have hcolS : e₁.COLOUR = headColour bp₁.2 := consume_some_colour hc1
              have hcolM : e₂.COLOUR = headColour bp₂.2 := consume_some_colour hc2
              have hdep1 : depSetEq bp₁.1 (sourceDepSet d S) = true := by
                rw [← hfaithful S hSnode e₁ hSout]
                exact ih S hSnode hSoutne hSca bp₁ hbp₁ e₁ hSout hcolS
              have hdep2 : depSetEq bp₂.1 (sourceDepSet d M) = true := by
                rw [← hfaithful M hMnode e₂ hMout]
                exact ih M hMnode hMoutne hMca bp₂ hbp₂ e₂ hMout hcolM
              have hSne : edgesBetween d S v ≠ [] := edgesBetween_ne_nil_of_mem he₁
              have hMne : edgesBetween d M v ≠ [] := edgesBetween_ne_nil_of_mem he₂
              have hsrc : sourceDepSet d v = depUnion (sourceDepSet d S) (sourceDepSet d M) := by
                rw [sourceDepSet_elim hinc hhyp hpairs,
                  premiseDepUnion_eq hfaithful hSnode hMnode hpairs hSne hMne]
              rw [hrval, hedep, hsrc]
              exact depSetEq_depUnion_congr hdep1 hdep2
            · rw [if_neg hq] at hr; exact absurd hr (by simp)
      ·
        obtain ⟨p, α, β, _, hincoming, hform, hpβ, helim⟩ := hintrosh
        have hpreds : predsOf d v = [p.START] := by
          rw [predsOf_eq, hincoming]; rfl
        rw [flowAt_kernel_intro_eq d fuel hhyp hpreds hform hpβ helim,
          List.mem_eraseDups] at hr
        rw [List.mem_flatMap] at hr; obtain ⟨bp, hbp, hr⟩ := hr
        rw [List.mem_filterMap] at hr; obtain ⟨e', he', hr⟩ := hr
        cases hc : consume e' bp.2 with
        | none => simp [hc] at hr
        | some q =>
          simp only [hc, Option.map_some, Option.some.injEq] at hr
          have hrval : r = (depRemove bp.1 α, q) := hr.symm
          obtain ⟨hu1, hu2, hue⟩ := mem_edgesBetween he'
          have hunode : p.START ∈ d.NODES := hu1 ▸ (hhygiene e' hue).1
          have huout : e' ∈ get_rule.outgoing p.START d := mem_outgoing_of_edgesBetween he'
          have huoutne : get_rule.outgoing p.START d ≠ [] := by
            intro h; rw [h] at huout; exact List.not_mem_nil huout
          have hulvl : p.START.LEVEL = v.LEVEL + 1 := by
            have := hlevel e' hue; rw [hu1, hu2] at this; exact this
          have huca : countAbove d p.START < fuel := by
            have := countAbove_lt_of_level_succ d hunode hulvl; omega
          have hcolu : e'.COLOUR = headColour bp.2 := consume_some_colour hc
          have hdepu : depSetEq bp.1 (sourceDepSet d p.START) = true := by
            rw [← hfaithful p.START hunode e' huout]
            exact ih p.START hunode huoutne huca bp hbp e' huout hcolu
          have hsrc : sourceDepSet d v = depRemove (sourceDepSet d p.START) α := by
            rw [sourceDepSet_intro hhyp hincoming hform hpβ helim]
            congr 1
            exact hfaithful p.START hunode p
              (mem_outgoing_of_mem_edges_start_eq p.START d
                (mem_incoming_mem_edges v d (by rw [hincoming]; exact List.mem_cons_self ..)) rfl)
          rw [hrval, hedep, hsrc]
          exact depSetEq_depRemove_congr hdepu

/--
 **routeEdgeDepChar** (prompt statement order): for every node with an
    outgoing edge and any fuel exceeding `countAbove`, each route's dependency
    set matches every outgoing edge whose colour is the route's head.
-/
theorem routeEdgeDepChar (d : Graph) (hstruct : structuralValid d)
    (honepercol : OneEdgePerColourPerNode d) (hroutefan : RouteFanUnique d)
    (hfaithful : FaithfulDecoration d) (hcoverage : RouteHeadCoverage d)
    (hreseed : ReseedFree d) :
    ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] →
      ∀ fuel, countAbove d v < fuel →
        ∀ r ∈ flowAt d fuel v, ∀ e ∈ get_rule.outgoing v d,
          e.COLOUR = headColour r.2 → depSetEq r.1 e.DEPENDENCY = true := by
  intro v hv hout fuel hcount
  exact routeEdgeDepChar_aux d hstruct honepercol hroutefan hfaithful hcoverage
    hreseed fuel v hv hout hcount



lemma nodup_allEq_eq_singleton {α : Type _} {l : List α} {a : α}
    (hnodup : l.Nodup) (hmem : a ∈ l) (hall : ∀ x ∈ l, x = a) : l = [a] := by
  cases l with
  | nil => simp at hmem
  | cons x xs =>
      have hxa : x = a := hall x (List.mem_cons_self ..)
      have hxsnil : xs = [] := by
        cases xs with
        | nil => rfl
        | cons y ys =>
            exfalso
            rw [List.nodup_cons] at hnodup
            have hya : y = a := hall y (by simp)
            exact hnodup.1 (by rw [hxa, ← hya]; exact List.mem_cons_self ..)
      rw [hxsnil, hxa]

private lemma eraseDupsBy_loop_nodup {α : Type _} [BEq α] [LawfulBEq α] :
    ∀ (xs acc : List α), acc.Nodup →
      (List.eraseDupsBy.loop (fun x y : α => x == y) xs acc).Nodup
  | [], acc, hacc => by
      rw [List.eraseDupsBy.loop.eq_1]; exact List.nodup_reverse.mpr hacc
  | a :: xs, acc, hacc => by
      rw [List.eraseDupsBy.loop.eq_2]
      cases h : acc.any (fun y => a == y)
      · apply eraseDupsBy_loop_nodup xs (a :: acc)
        refine List.nodup_cons.mpr ⟨?_, hacc⟩
        intro hmem
        have hany : acc.any (fun y => a == y) = true := by
          rw [List.any_eq_true]; exact ⟨a, hmem, by simp⟩
        rw [h] at hany; contradiction
      · exact eraseDupsBy_loop_nodup xs acc hacc

lemma eraseDups_nodup {α : Type _} [BEq α] [LawfulBEq α] (l : List α) :
    (l.eraseDups).Nodup := by
  rw [List.eraseDups.eq_1]
  exact eraseDupsBy_loop_nodup l [] (by simp)

/--  The flow at a kernel node is deduplicated (hence `Nodup`).  -/
lemma flowAt_kernel_nodup (d : Graph) {v : Vertex}
    (hshape : ∀ w ∈ d.NODES, get_rule.incoming w d ≠ [] →
      get_rule.outgoing w d ≠ [] → NodeElimShape d w ∨ NodeIntroShape d w)
    (hv : v ∈ d.NODES) (hkernel : get_rule.incoming v d ≠ [])
    (hout : get_rule.outgoing v d ≠ []) :
    (flowAt d (stdFuel d) v).Nodup := by
  have hstd : stdFuel d = (stdFuel d - 1) + 1 := by
    unfold stdFuel; omega
  rcases hshape v hv hkernel hout with
    ⟨M, S, hhyp, hpredsne, hpairs⟩ | ⟨p, α, β, hhyp, hinc, hform, hpβ, helim⟩
  · rw [hstd, flowAt_kernel_elim_eq d (stdFuel d - 1) hhyp hpredsne hpairs]
    exact eraseDups_nodup _
  · have hpreds : predsOf d v = [p.START] := by rw [predsOf_eq, hinc]; rfl
    rw [hstd, flowAt_kernel_intro_eq d (stdFuel d - 1) hhyp hpreds hform hpβ helim]
    exact eraseDups_nodup _

/--  One outgoing colour-`i` edge: uniqueness up to equality + `Nodup`.  -/
lemma allEq_colour_filter_singleton {L : List Deduction} {i : Nat} {e : Deduction}
    (hnodup : L.Nodup)
    (huniq : ∀ e1 ∈ L, ∀ e2 ∈ L, e1.COLOUR = e2.COLOUR → e1 = e2)
    (hemem : e ∈ L) (hecol : e.COLOUR = i) :
    L.filter (fun e => e.COLOUR == i) = [e] := by
  apply nodup_allEq_eq_singleton (List.Nodup.filter _ hnodup)
  · rw [List.mem_filter]; exact ⟨hemem, by rw [hecol]; simp⟩
  · intro x hx
    rw [List.mem_filter] at hx
    have hxcol : x.COLOUR = i := by simpa using hx.2
    exact huniq x hx.1 e hemem (by rw [hxcol, hecol])

/--  One route with head colour `i`: uniqueness up to equality + `Nodup`.  -/
lemma allEq_head_filter_singleton {F : List FlowPair} {i : Nat} {r : FlowPair}
    (hnodup : F.Nodup)
    (huniq : ∀ r1 ∈ F, ∀ r2 ∈ F, r1.2.headD 0 = r2.2.headD 0 → r1 = r2)
    (hrmem : r ∈ F) (hrhead : headColour r.2 = i) :
    F.filter (fun bp => headColour bp.2 == i) = [r] := by
  apply nodup_allEq_eq_singleton (List.Nodup.filter _ hnodup)
  · rw [List.mem_filter]; exact ⟨hrmem, by rw [hrhead]; simp⟩
  · intro x hx
    rw [List.mem_filter] at hx
    have hxhead : headColour x.2 = i := by simpa using hx.2
    exact huniq x hx.1 r hrmem (hxhead.trans hrhead.symm)

/--
 `flowRuleCorrect_collapsed` works at any kernel node; the
    collapsed nodes being the multi-route `Φᵢ` instance). The literal Def-23
    CorrectRuleApp predicate holds, derived from `routeEdgeDepChar` (label
    correctness), `RouteFanUnique` (Φᵢ ≤ 1), `RouteHeadCoverage` (Φᵢ ≥ 1 / edge
    exists / no spurious edge) and `OneEdgePerColourPerNode` (edge ≤ 1).

    `hEdgesNodup` is a list-vs-set construction artifact: paper `E_D^i` is a set
    (Def 12), so legal HC output has no duplicate edges; the construction proof
    that `compress_nodes_graph` output has `Nodup` edges is deferred.
-/
theorem flowRuleCorrect_collapsed (d : Graph)
    (hstruct : structuralValid d) (hEdgesNodup : d.EDGES.Nodup)
    (honepercol : OneEdgePerColourPerNode d)
    (hroutefan : RouteFanUnique d) (hfaithful : FaithfulDecoration d)
    (hcoverage : RouteHeadCoverage d) (hreseed : ReseedFree d)
    {v : Vertex} (hv : v ∈ d.NODES) (hkernel : get_rule.incoming v d ≠ []) :
    FlowRuleCorrect d v := by
  unfold FlowRuleCorrect flowRuleCorrectAtB
  have hincEmpty : (get_rule.incoming v d).isEmpty = false := by
    cases h : get_rule.incoming v d with
    | nil => exact absurd h hkernel
    | cons e es => rfl
  rw [hincEmpty]
  by_cases houtEmpty : (get_rule.outgoing v d).isEmpty
  · rw [if_pos houtEmpty]; simp
  · rw [if_neg houtEmpty]
    have hout : get_rule.outgoing v d ≠ [] := by
      intro h; rw [h] at houtEmpty; exact houtEmpty rfl
    have hshape := hstruct.2.2
    have houtnodup : (get_rule.outgoing v d).Nodup := by
      rw [outgoing_eq_filter]; exact List.Nodup.filter _ hEdgesNodup
    have hFnodup := flowAt_kernel_nodup d hshape hv hkernel hout
    have hca : countAbove d v < stdFuel d := by
      have := countAbove_le d v; unfold stdFuel; omega
    have hRE := routeEdgeDepChar d hstruct honepercol hroutefan hfaithful
      hcoverage hreseed v hv hout (stdFuel d) hca
    obtain ⟨hcov1, hcov2⟩ := hcoverage v hv hout
    have huniqE : ∀ e1 ∈ get_rule.outgoing v d, ∀ e2 ∈ get_rule.outgoing v d,
        e1.COLOUR = e2.COLOUR → e1 = e2 := by
      intro e1 h1 e2 h2 hh
      exact honepercol e1 e2 (mem_outgoing_mem_edges v d h1)
        (mem_outgoing_mem_edges v d h2)
        ⟨by rw [mem_outgoing_start_eq v d h1, mem_outgoing_start_eq v d h2], hh⟩
    cases hF : flowAt d (stdFuel d) v with
    | nil =>
        exfalso
        obtain ⟨e0, he0⟩ := List.exists_mem_of_ne_nil _ hout
        obtain ⟨r, hrmem, _⟩ := hcov2 e0 he0
        rw [hF] at hrmem; exact List.not_mem_nil hrmem
    | cons r0 rs =>
        cases rs with
        | nil =>
            rcases r0 with ⟨b, p⟩
            show singletonBulletB d v b p = true
            have hr0mem : (b, p) ∈ flowAt d (stdFuel d) v := by rw [hF]; simp
            have hallcol : ∀ x ∈ get_rule.outgoing v d, x.COLOUR = headColour p := by
              intro x hx
              obtain ⟨r, hrmem, hrhead⟩ := hcov2 x hx
              rw [hF, List.mem_singleton] at hrmem
              rw [← hrhead, hrmem]
            obtain ⟨e0, he0⟩ := List.exists_mem_of_ne_nil _ hout
            have houts : get_rule.outgoing v d = [e0] := by
              apply nodup_allEq_eq_singleton houtnodup he0
              intro x hx
              exact huniqE x hx e0 he0 (by rw [hallcol x hx, hallcol e0 he0])
            unfold singletonBulletB
            rw [houts]
            have hcole0 : e0.COLOUR = headColour p := hallcol e0 he0
            have hdep := hRE (b, p) hr0mem e0 he0 hcole0
            simp only [Bool.and_eq_true, beq_iff_eq]
            exact ⟨hcole0, hdep⟩
        | cons r1 rs' =>
            show phiBulletB d v (r0 :: r1 :: rs') = true
            have hFnodup' : (r0 :: r1 :: rs').Nodup := hF ▸ hFnodup
            have huniqR : ∀ a ∈ (r0 :: r1 :: rs'), ∀ b ∈ (r0 :: r1 :: rs'),
                a.2.headD 0 = b.2.headD 0 → a = b :=
              fun a ha b hb hh => hroutefan v hv a b (hF ▸ ha) (hF ▸ hb) hh
            unfold phiBulletB
            rw [Bool.and_eq_true]
            refine ⟨?_, ?_⟩
            · rw [List.all_eq_true]
              intro i hi
              rw [List.mem_eraseDups, List.mem_map] at hi
              obtain ⟨r, hrmemF, hrhead⟩ := hi
              obtain ⟨ed, hedmem, hedcol⟩ := hcov1 r (hF ▸ hrmemF)
              rw [hrhead] at hedcol
              have hffilt := allEq_head_filter_singleton hFnodup' huniqR hrmemF hrhead
              have hofilt := allEq_colour_filter_singleton houtnodup huniqE hedmem hedcol
              rcases r with ⟨b, p⟩
              rw [hffilt, hofilt]
              have hdep := hRE (b, p) (hF ▸ hrmemF) ed hedmem (by
                show ed.COLOUR = headColour p
                rw [hedcol]; exact hrhead.symm)
              simpa using hdep
            · rw [List.all_eq_true]
              intro e he
              obtain ⟨r, hrmem, hrhead⟩ := hcov2 e he
              rw [List.contains_eq_mem, decide_eq_true_eq, List.mem_eraseDups,
                List.mem_map]
              exact ⟨r, hF ▸ hrmem, hrhead⟩

/-!     FlowCollapsedProof header ; facts go through `#eval`)  -/

def faithfulNewAtB (d : Graph) (v : Vertex) : Bool :=
  (get_rule.outgoing v d).all fun e => decide (e.DEPENDENCY = sourceDepSet d v)

def routeHeadCoverageAtB (d : Graph) (v : Vertex) : Bool :=
  ((flowAt d (stdFuel d) v).all fun r =>
    (get_rule.outgoing v d).any fun e => e.COLOUR == headColour r.2) &&
  ((get_rule.outgoing v d).all fun e =>
    (flowAt d (stdFuel d) v).any fun r => headColour r.2 == e.COLOUR)

def routeEdgeDepCharAtB (d : Graph) (v : Vertex) : Bool :=
  (flowAt d (stdFuel d) v).all fun r =>
    (get_rule.outgoing v d).all fun e =>
      if e.COLOUR == headColour r.2 then depSetEq r.1 e.DEPENDENCY else true

/-!  generalized FaithfulDecoration (rule-cased sourceDepSet) over ALL nodes  -/
/-!  RouteHeadCoverage (both directions) at the collapsed kernels  -/
/-!  routeEdgeDepChar predicate eval-true at ALL nodes of the fixtures  -/
/-!  literal Def-23 CorrectRuleApp at the collapsed kernels (the corollary)  -/


#print axioms flowRuleCorrect_collapsed
end FlowSpec
