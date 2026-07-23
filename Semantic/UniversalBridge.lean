import Semantic.TreeBridge
import Semantic.Examples

/-! Universal acceptance for simple-tree DLDS graphs. Admissible paths may differ
from the canonical path only by evaluator-inert terminal stops. -/

open Semantic

namespace Semantic



/--
 Clause (2)+(4): an active carrier trace from current formula `φ`. Each
    real-event step `(t,l)` must hit the unique outgoing edge `e` of the current
    node: target column `t = idxOf(e.END.FORMULA)+1` and label
    `l = inputLabelForEdge … φ e.END`. A minor-premise delivery to an `⊃E` node
    stops (its merged output is carried by the major); reaching a root (no
    outgoing) stops; trailing steps must then be `(0,0)`.
-/
def admChainB (d : Graph) (formulas : List Formula) :
    Formula → List (Nat × Nat) → Bool
  | φ, [] =>
      -- an empty chain is legitimate ONLY if the carrier has reached a node with
      -- no outgoing edge (the root); a node with an outgoing edge MUST follow it
      -- (clause 2), so truncating a hypothesis's delivery is inadmissible.
      match d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none => true
      | some v => (get_rule.outgoing v d).isEmpty
  | φ, (t, l) :: rest =>
      match d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none => t == 0 && l == 0 && rest.all (fun s => s == (0, 0))
      | some v =>
          match get_rule.outgoing v d with
          | [] => t == 0 && l == 0 && rest.all (fun s => s == (0, 0))
          | e :: _ =>
              let isMinor : Bool :=
                match classifyRule? e.END d with
                | some (DLDSRuleClass.elim _ minor) => decide (φ = minor.START.FORMULA)
                | _ => false
              (t == formulas.idxOf e.END.FORMULA + 1) &&
              (l == inputLabelForEdge d formulas φ e.END) &&
              (if isMinor then rest.all (fun s => s == (0, 0))
               else admChainB d formulas e.END.FORMULA rest)

/--
 Clause (3)+(2): a hypothesis column carries EXACTLY `delay` legitimate TOP
    padding steps `(col+1, 0)` (repetition at its own column ; "down to its
    first-use level"), then its carrier trace. The exact count is what forces the
    canonical path (a looser "≥ 0 paddings" admits level-shifted variants).
-/
def admHypColumnB (d : Graph) (formulas : List Formula) (col : Nat) (φ : Formula) :
    Nat → List (Nat × Nat) → Bool
  | 0, steps => admChainB d formulas φ steps
  | _ + 1, [] => false
  | k + 1, (t, l) :: rest =>
      (t == col + 1 && l == 0) && admHypColumnB d formulas col φ k rest

/--
 Clause (1)+(3): a column's path is admissible ; a hypothesis column carries a
    `delay`-padded carrier trace (`delay = maxLevel − level`, its first-use
    level); a derived column is inactive (all stops).
-/
def admColumnB (d : Graph) (formulas : List Formula) (col : Nat)
    (steps : List (Nat × Nat)) : Bool :=
  match formulas[col]? with
  | none => steps.all (fun s => s == (0, 0))
  | some φ =>
      match d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none => steps.all (fun s => s == (0, 0))
      | some v =>
          if v.HYPOTHESIS then
            let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
            admHypColumnB d formulas col φ (maxLvl - v.LEVEL) steps
          else steps.all (fun s => s == (0, 0))

/--
 Clause (5): some carrier delivers the goal to the goal column via a real
    derivation event (label ≠ 0).
-/
def goalReceivedB (d : Graph) (P : PathInput) : Bool :=
  P.any (fun colpath => colpath.any (fun s => s.1 == goalColumn d + 1 && s.2 != 0))

/--
 **admissibleDLDSPathB** ; Bool mirror: the path has one column per formula
    (structural well-formedness ; `PathInput` is formula-column indexed; this is
    NOT clause-3/4 "fill to depth", within-column trailing stops stay free), every
    column is admissible (clauses 1–4), and the goal column receives the goal via
    a real event (clause 5).
-/
def admissibleDLDSPathB (d : Graph) (P : PathInput) : Bool :=
  let formulas := buildFormulas d
  (P.length == formulas.length) &&
  ((List.range formulas.length).all (fun col =>
    admColumnB d formulas col (P.getD col []))) &&
  goalReceivedB d P

/--  **AdmissibleDLDSPath** (Prop mirror, repo's boolean-reflection pattern).  -/
def AdmissibleDLDSPath (d : Graph) (P : PathInput) : Prop :=
  admissibleDLDSPathB d P = true

/--  Faithful `Invalid`: not an admissible carrier assignment.  -/
def InvalidDLDSPath (d : Graph) (P : PathInput) : Prop :=
  ¬ AdmissibleDLDSPath d P


def badGoalSelf : PathInput := [[(0, 0)], [(2, 1)]]

def skipRealRule : PathInput := [[(0, 0)], [(0, 0)]]

def nonHypFakeEvent : PathInput := [[(2, 1)], [(1, 1)]]

def exE : Graph :=
  let a   : Vertex := Vertex.node 1 4 (#"A") true false []
  let ab  : Vertex := Vertex.node 2 4 (#"A" >> #"B") true false []
  let b   : Vertex := Vertex.node 3 3 (#"B") false false []
  let abb : Vertex := Vertex.node 4 2 ((#"A" >> #"B") >> #"B") false false []
  let rt  : Vertex := Vertex.node 5 1 (#"A" >> ((#"A" >> #"B") >> #"B")) false false []
  Graph.dlds [a, ab, b, abb, rt]
    [Deduction.edge a   b   0 [#"A"],
     Deduction.edge ab  b   0 [#"A" >> #"B"],
     Deduction.edge b   abb 0 [#"A", #"A" >> #"B"],
     Deduction.edge abb rt  0 [#"A"]] []

def exE_truncMinor : PathInput :=
  (List.range (pathsFromDLDS exE).length).map
    (fun c => if c == 0 then [] else (pathsFromDLDS exE).getD c [])



/--
 The only information `propagate_tokens` reads about a column's path at a step:
    `none` if the step is absent or a `(0,0)` stop, else the real move `(t,l)`.
-/
def effStep (P : PathInput) (c s : Nat) : Option (Nat × Nat) :=
  match P[c]? with
  | none => none
  | some pcol =>
      match pcol[s]? with
      | none => none
      | some stp => if stp.1 = 0 then none else some stp

/--
 Two path inputs are read-equivalent iff the evaluator cannot tell them
    apart: every `(column, step)` read yields the same effective move.
-/
def ReadEquiv (P Q : PathInput) : Prop := ∀ c s, effStep P c s = effStep Q c s

lemma effStep_lt {P : PathInput} {c s : Nat} (hc : c < P.length)
    (hs : s < (P[c]'hc).length) :
    effStep P c s =
      (if ((P[c]'hc)[s]'hs).1 = 0 then none else some ((P[c]'hc)[s]'hs)) := by
  simp only [effStep, List.getElem?_eq_getElem hc, List.getElem?_eq_getElem hs]

lemma effStep_ge_col {P : PathInput} {c s : Nat} (hc : ¬ c < P.length) :
    effStep P c s = none := by
  simp only [effStep, List.getElem?_eq_none (Nat.le_of_not_lt hc)]

lemma effStep_ge_step {P : PathInput} {c s : Nat} (hc : c < P.length)
    (hs : ¬ s < (P[c]'hc).length) :
    effStep P c s = none := by
  simp only [effStep, List.getElem?_eq_getElem hc,
    List.getElem?_eq_none (Nat.le_of_not_lt hs)]

lemma propagate_tokens_effStep {n : Nat} (P : PathInput) (tokens : List (Token n))
    (current_level num_levels : Nat) (outputs : List (List.Vector Bool n)) :
    propagate_tokens tokens P current_level num_levels outputs =
      tokens.filterMap (fun token =>
        if current_level > 0 then
          match effStep P token.origin_column (num_levels - current_level - 1) with
          | none => none
          | some st =>
              if h_out : token.current_column < outputs.length then
                some { origin_column := token.origin_column
                       source_column := token.current_column
                       current_level := current_level - 1
                       current_column := st.1 - 1
                       dep_vector := outputs.get ⟨token.current_column, h_out⟩
                       input_label := st.2 }
              else none
        else none) := by
  unfold propagate_tokens
  apply List.filterMap_congr
  intro token _
  by_cases hcl : current_level > 0
  · rw [if_pos hcl]
    by_cases hc : token.origin_column < P.length
    · rw [dif_pos hc]
      simp only [List.get_eq_getElem]
      by_cases hlt : num_levels - current_level - 1 < (P[token.origin_column]'hc).length
      · rw [dif_pos (And.intro hcl hlt), effStep_lt hc hlt]
        split <;> rfl
      · rw [dif_neg (by intro h; exact hlt h.2), effStep_ge_step hc hlt]
    · rw [dif_neg hc, effStep_ge_col hc]
  · rw [if_neg hcl]
    by_cases hc : token.origin_column < P.length
    · rw [dif_pos hc]
      simp only [List.get_eq_getElem]
      rw [dif_neg (by intro h; exact hcl h.1)]
    · rw [dif_neg hc]

/--  Read-equivalent inputs give equal token propagation.  -/
lemma propagate_tokens_congr {n : Nat} {P Q : PathInput} (hRE : ReadEquiv P Q)
    (tokens : List (Token n)) (current_level num_levels : Nat)
    (outputs : List (List.Vector Bool n)) :
    propagate_tokens tokens P current_level num_levels outputs =
      propagate_tokens tokens Q current_level num_levels outputs := by
  rw [propagate_tokens_effStep, propagate_tokens_effStep]
  apply List.filterMap_congr
  intro token _
  rw [hRE token.origin_column (num_levels - current_level - 1)]

lemma eval_from_level_congr {n : Nat} {P Q : PathInput} (hRE : ReadEquiv P Q)
    (num_levels : Nat) :
    ∀ (layers : List (GridLayer n)) (level : Nat) (tokens : List (Token n))
      (err : Bool),
      eval_from_level P level tokens layers err num_levels =
        eval_from_level Q level tokens layers err num_levels := by
  intro layers
  induction layers with
  | nil => intro level tokens err; rfl
  | cons layer rest ih =>
      intro level tokens err
      cases rest with
      | nil => simp only [eval_from_level]
      | cons l2 r2 =>
          rw [eval_from_level]
          conv_rhs => rw [eval_from_level]
          cases hel : evaluate_layer layer tokens with
          | mk outs le =>
              simp only [propagate_tokens_congr hRE]
              exact ih (level - 1) _ _

lemma get_eval_result_congr {n : Nat} {P Q : PathInput} (hRE : ReadEquiv P Q)
    (layers : List (GridLayer n)) (iv : List (List.Vector Bool n)) :
    get_eval_result layers iv P = get_eval_result layers iv Q := by
  unfold get_eval_result
  exact eval_from_level_congr hRE layers.length layers (layers.length - 1)
    (initialize_tokens iv layers.length) false

/--  Read-equivalent path inputs are accepted/discharged identically.  -/
lemma evaluateDLDS_congr {d : Graph} {P Q : PathInput} (g : Nat)
    (hRE : ReadEquiv P Q) :
    evaluateDLDS d P g = evaluateDLDS d Q g := by
  unfold evaluateDLDS evaluateCircuit
  dsimp only
  rw [get_eval_result_congr hRE]

/--
 `Discharged d P g` ; the goal column's evaluated dependency vector is all
    false (every assumption discharged) on path input `P`.
-/
def Discharged (d : Graph) (P : PathInput) (g : Nat) : Prop :=
  AllAssumptionsDischarged P (buildGridFromDLDS d) (initialVectorsFromDLDS d) g

lemma Discharged_congr {d : Graph} {P Q : PathInput} (g : Nat) (hRE : ReadEquiv P Q) :
    Discharged d P g = Discharged d Q g := by
  unfold Discharged AllAssumptionsDischarged
  rw [get_eval_result_congr hRE]


/--  `effStep` for a single column.  -/
def colEff (col : List (Nat × Nat)) (s : Nat) : Option (Nat × Nat) :=
  match col[s]? with
  | none => none
  | some st => if st.1 = 0 then none else some st

lemma effStep_colEff {P : PathInput} {c : Nat} (h : c < P.length) (s : Nat) :
    effStep P c s = colEff (P.getD c []) s := by
  have hg : P.getD c [] = P[c]'h := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]
  unfold effStep colEff
  rw [List.getElem?_eq_getElem h, hg]

lemma colEff_all_stops {col : List (Nat × Nat)} (h : ∀ x ∈ col, x = (0, 0))
    (s : Nat) : colEff col s = none := by
  unfold colEff
  cases hs : col[s]? with
  | none => rfl
  | some st =>
      have hmem : st ∈ col := List.mem_of_getElem? hs
      rw [h st hmem]; simp

/--
 A chain validated by `admChainB` at a TERMINAL formula (no outgoing edge, or
    not found) is entirely stops, so its `colEff` is `none`.
-/
lemma terminal_chain_all_stops (d : Graph) (formulas : List Formula) (φ : Formula)
    (hterm : admChainB d formulas φ [] = true) :
    ∀ chain, admChainB d formulas φ chain = true → ∀ x ∈ chain, x = (0, 0) := by
  intro chain
  cases chain with
  | nil => intro _ x hx; cases hx
  | cons st rest =>
      intro hch x hx
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          simp only [admChainB, hfind] at hch
          rw [Bool.and_eq_true, Bool.and_eq_true] at hch
          obtain ⟨⟨ht, hl⟩, hrest⟩ := hch
          cases hx with
          | head => exact Prod.ext (by simpa using ht) (by simpa using hl)
          | tail _ hxr => exact by simpa using (List.all_eq_true.mp hrest) x hxr
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              simp only [admChainB, hfind, hout] at hch
              rw [Bool.and_eq_true, Bool.and_eq_true] at hch
              obtain ⟨⟨ht, hl⟩, hrest⟩ := hch
              cases hx with
              | head => exact Prod.ext (by simpa using ht) (by simpa using hl)
              | tail _ hxr => exact by simpa using (List.all_eq_true.mp hrest) x hxr
          | cons e es => simp [admChainB, hfind, hout] at hterm

lemma terminal_chain_colEff (d : Graph) (formulas : List Formula) (φ : Formula)
    (hterm : admChainB d formulas φ [] = true)
    (chain : List (Nat × Nat)) (hch : admChainB d formulas φ chain = true) (s : Nat) :
    colEff chain s = none :=
  colEff_all_stops (terminal_chain_all_stops d formulas φ hterm chain hch) s

/--
 **Two `admChainB`-valid chains from the same formula agree on `colEff`.** The
    unique-outgoing-edge check forces identical real events; root/minor stops and
    trailing differences are all `colEff = none`. No fuel/`routeFrom` needed.
-/
lemma chain_colEff_unique (d : Graph) (formulas : List Formula) :
    ∀ (chain1 : List (Nat × Nat)) (φ : Formula) (chain2 : List (Nat × Nat)),
      admChainB d formulas φ chain1 = true →
      admChainB d formulas φ chain2 = true →
      ∀ s, colEff chain1 s = colEff chain2 s := by
  intro chain1
  induction chain1 with
  | nil =>
      intro φ chain2 h1 h2 s
      rw [colEff_all_stops (fun _ hx => by cases hx)]
      exact (terminal_chain_colEff d formulas φ h1 chain2 h2 s).symm
  | cons st1 rest1 ih =>
      intro φ chain2 h1 h2 s
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          have hterm : admChainB d formulas φ [] = true := by
            simp only [admChainB, hfind]
          rw [terminal_chain_colEff d formulas φ hterm _ h1 s,
            terminal_chain_colEff d formulas φ hterm _ h2 s]
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              have hterm : admChainB d formulas φ [] = true := by
                simp only [admChainB, hfind, hout, List.isEmpty_nil]
              rw [terminal_chain_colEff d formulas φ hterm _ h1 s,
                terminal_chain_colEff d formulas φ hterm _ h2 s]
          | cons e es =>
              cases chain2 with
              | nil =>
                  exfalso; simp [admChainB, hfind, hout] at h2
              | cons st2 rest2 =>
                  simp only [admChainB, hfind, hout] at h1 h2
                  set isMin : Bool :=
                    (match classifyRule? e.END d with
                      | some (DLDSRuleClass.elim _ minor) =>
                          decide (φ = minor.START.FORMULA)
                      | _ => false) with hmin
                  rw [Bool.and_eq_true, Bool.and_eq_true] at h1 h2
                  obtain ⟨⟨ht1, hl1⟩, htail1⟩ := h1
                  obtain ⟨⟨ht2, hl2⟩, htail2⟩ := h2
                  have hst1 : st1 =
                      (formulas.idxOf e.END.FORMULA + 1,
                       inputLabelForEdge d formulas φ e.END) :=
                    Prod.ext (by simpa using ht1) (by simpa using hl1)
                  have hst2 : st2 =
                      (formulas.idxOf e.END.FORMULA + 1,
                       inputLabelForEdge d formulas φ e.END) :=
                    Prod.ext (by simpa using ht2) (by simpa using hl2)
                  cases s with
                  | zero =>
                      simp only [colEff, List.getElem?_cons_zero, hst1, hst2]
                  | succ s' =>
                      show colEff rest1 s' = colEff rest2 s'
                      by_cases hIsMin : isMin = true
                      · rw [if_pos hIsMin] at htail1 htail2
                        rw [colEff_all_stops (fun x hx => by
                              simpa using (List.all_eq_true.mp htail1) x hx),
                          colEff_all_stops (fun x hx => by
                              simpa using (List.all_eq_true.mp htail2) x hx)]
                      · rw [if_neg hIsMin] at htail1 htail2
                        exact ih e.END.FORMULA rest2 htail1 htail2 s'

lemma allStops_of {steps : List (Nat × Nat)}
    (h : steps.all (fun s => s == (0, 0)) = true) : ∀ x ∈ steps, x = (0, 0) :=
  fun x hx => by simpa using (List.all_eq_true.mp h) x hx

/--
 Two `admHypColumnB`-valid columns (same `col`, `φ`, `delay`) agree on `colEff`:
    the `delay` paddings are forced identical, the chains agree by
    `chain_colEff_unique`.
-/
lemma admHypColumn_colEff_unique (d : Graph) (formulas : List Formula)
    (col : Nat) (φ : Formula) :
    ∀ (delay : Nat) (steps1 steps2 : List (Nat × Nat)),
      admHypColumnB d formulas col φ delay steps1 = true →
      admHypColumnB d formulas col φ delay steps2 = true →
      ∀ s, colEff steps1 s = colEff steps2 s := by
  intro delay
  induction delay with
  | zero =>
      intro steps1 steps2 h1 h2 s
      exact chain_colEff_unique d formulas steps1 φ steps2 h1 h2 s
  | succ k ih =>
      intro steps1 steps2 h1 h2 s
      cases steps1 with
      | nil => simp [admHypColumnB] at h1
      | cons st1 r1 =>
          cases steps2 with
          | nil => simp [admHypColumnB] at h2
          | cons st2 r2 =>
              simp only [admHypColumnB, Bool.and_eq_true] at h1 h2
              obtain ⟨hh1, ht1⟩ := h1
              obtain ⟨hh2, ht2⟩ := h2
              have hst1 : st1 = (col + 1, 0) :=
                Prod.ext (by simpa using hh1.1) (by simpa using hh1.2)
              have hst2 : st2 = (col + 1, 0) :=
                Prod.ext (by simpa using hh2.1) (by simpa using hh2.2)
              cases s with
              | zero => simp only [colEff, List.getElem?_cons_zero, hst1, hst2]
              | succ s' =>
                  show colEff r1 s' = colEff r2 s'
                  exact ih r1 r2 ht1 ht2 s'

/--  Two `admColumnB`-valid columns at the same column index agree on `colEff`.  -/
lemma admColumn_colEff_unique (d : Graph) (formulas : List Formula) (c : Nat)
    (steps1 steps2 : List (Nat × Nat))
    (h1 : admColumnB d formulas c steps1 = true)
    (h2 : admColumnB d formulas c steps2 = true) :
    ∀ s, colEff steps1 s = colEff steps2 s := by
  intro s
  cases hfc : formulas[c]? with
  | none =>
      simp only [admColumnB, hfc] at h1 h2
      rw [colEff_all_stops (allStops_of h1), colEff_all_stops (allStops_of h2)]
  | some φ =>
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          simp only [admColumnB, hfc, hfind] at h1 h2
          rw [colEff_all_stops (allStops_of h1), colEff_all_stops (allStops_of h2)]
      | some v =>
          by_cases hhyp : v.HYPOTHESIS = true
          · simp only [admColumnB, hfc, hfind, hhyp, if_true] at h1 h2
            exact admHypColumn_colEff_unique d formulas c φ _ steps1 steps2 h1 h2 s
          · have hhf : v.HYPOTHESIS = false := by
              cases hb : v.HYPOTHESIS with
              | true => exact absurd hb hhyp
              | false => rfl
            simp only [admColumnB, hfc, hfind, hhf, Bool.false_eq_true, if_false] at h1 h2
            rw [colEff_all_stops (allStops_of h1), colEff_all_stops (allStops_of h2)]


lemma level_zero_terminal (d : Graph)
    (hlev : ∀ e ∈ d.EDGES, e.START.LEVEL = e.END.LEVEL + 1)
    {v : Vertex} (hzero : v.LEVEL = 0) : get_rule.outgoing v d = [] := by
  cases hout : get_rule.outgoing v d with
  | nil => rfl
  | cons e es =>
      exfalso
      have he : e ∈ get_rule.outgoing v d := by rw [hout]; exact List.mem_cons_self ..
      have heE : e ∈ d.EDGES := mem_outgoing_mem_edges v d he
      have hs : e.START = v := mem_outgoing_start_eq v d he
      have hl := hlev e heE
      rw [hs, hzero] at hl
      omega

lemma replicate_all_stops (k : Nat) :
    (List.replicate k (0, 0)).all (fun s => s == (0, 0)) = true := by
  rw [List.all_eq_true]
  intro x hx
  simp [List.eq_of_mem_replicate hx]

lemma routeFrom_terminal_stops (d : Graph) (formulas : List Formula) (φ : Formula)
    (hterm : ∀ v, d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some v →
      get_rule.outgoing v d = []) :
    ∀ fuel, (routeFrom d formulas fuel φ).all (fun s => s == (0, 0)) = true := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ f ih =>
      cases hfind : d.NODES.find? (fun u => decide (u.FORMULA = φ)) with
      | none =>
          simpa only [routeFrom, hfind, List.all_cons, beq_self_eq_true,
            Bool.true_and] using ih
      | some v =>
          simpa only [routeFrom, hfind, hterm v hfind, List.all_cons,
            beq_self_eq_true, Bool.true_and] using ih

lemma admChainB_cons_reduce (d : Graph) (formulas : List Formula) (φ : Formula)
    (t l : Nat) (rest : List (Nat × Nat)) {v : Vertex} {e : Deduction}
    {es : List Deduction}
    (hfind : d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some v)
    (hout : get_rule.outgoing v d = e :: es) :
    admChainB d formulas φ ((t, l) :: rest) =
      ((t == formulas.idxOf e.END.FORMULA + 1 &&
        l == inputLabelForEdge d formulas φ e.END) &&
       (if (match classifyRule? e.END d with
            | some (DLDSRuleClass.elim _ minor) => decide (φ = minor.START.FORMULA)
            | _ => false) then rest.all (fun s => s == (0, 0))
        else admChainB d formulas e.END.FORMULA rest)) := by
  conv_lhs => rw [admChainB]
  simp only [hfind, hout]

lemma routeFrom_admChainB (d : Graph) (formulas : List Formula)
    (hlev : ∀ e ∈ d.EDGES, e.START.LEVEL = e.END.LEVEL + 1)
    (hhyg : ∀ e ∈ d.EDGES, e.START ∈ d.NODES ∧ e.END ∈ d.NODES)
    (hinj : ∀ u ∈ d.NODES, ∀ w ∈ d.NODES, u.FORMULA = w.FORMULA → u = w) :
    ∀ (fuel : Nat) (φ : Formula),
      (∀ v ∈ d.NODES, v.FORMULA = φ → v.LEVEL ≤ fuel) →
      admChainB d formulas φ (routeFrom d formulas fuel φ) = true := by
  intro fuel
  induction fuel with
  | zero =>
      intro φ hcond
      show admChainB d formulas φ [] = true
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none => simp only [admChainB, hfind]
      | some v =>
          have hvmem : v ∈ d.NODES := find?_some_mem hfind
          have hvform : v.FORMULA = φ := of_decide_eq_true (by have h := List.find?_some hfind; simpa using h)
          have hzero : v.LEVEL = 0 := Nat.le_zero.mp (hcond v hvmem hvform)
          simp only [admChainB, hfind, level_zero_terminal d hlev hzero,
            List.isEmpty_nil]
  | succ f ih =>
      intro φ hcond
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          have htail := routeFrom_terminal_stops d formulas φ
            (fun w hw => by simp [hfind] at hw) f
          simp only [routeFrom, hfind, admChainB, List.all_cons, beq_self_eq_true,
            Bool.true_and, htail, Bool.and_self]
      | some v =>
          have hvmem : v ∈ d.NODES := find?_some_mem hfind
          have hvform : v.FORMULA = φ := of_decide_eq_true (by have h := List.find?_some hfind; simpa using h)
          cases hout : get_rule.outgoing v d with
          | nil =>
              have htail := routeFrom_terminal_stops d formulas φ
                (fun w hw => by rw [hfind] at hw; cases hw; exact hout) f
              simp only [routeFrom, hfind, hout, admChainB, htail, beq_self_eq_true,
                Bool.true_and, Bool.and_self]
          | cons e es =>
              have he : e ∈ get_rule.outgoing v d := by
                rw [hout]; exact List.mem_cons_self ..
              have heE : e ∈ d.EDGES := mem_outgoing_mem_edges v d he
              have hsv : e.START = v := mem_outgoing_start_eq v d he
              have hEmem : e.END ∈ d.NODES := (hhyg e heE).2
              have hlevel : v.LEVEL = e.END.LEVEL + 1 := by
                have := hlev e heE; rw [hsv] at this; exact this
              have hcondE : ∀ w ∈ d.NODES, w.FORMULA = e.END.FORMULA → w.LEVEL ≤ f := by
                intro w hw hwf
                have hweq : w = e.END := hinj w hw e.END hEmem hwf
                have hvle := hcond v hvmem hvform
                rw [hweq]; omega
              simp only [routeFrom, hfind, hout]
              rw [admChainB_cons_reduce d formulas φ _ _ _ hfind hout]
              simp only [beq_self_eq_true, Bool.true_and, Bool.and_true]
              cases hcls : classifyRule? e.END d with
              | none =>
                  simp only [hcls]
                  exact ih e.END.FORMULA hcondE
              | some cls =>
                  cases cls with
                  | hypothesis =>
                      simp only [hcls]; exact ih e.END.FORMULA hcondE
                  | intro p =>
                      simp only [hcls]; exact ih e.END.FORMULA hcondE
                  | elim major minor =>
                      by_cases hφ : φ = minor.START.FORMULA
                      · simp only [hcls, hφ, decide_true, if_true]
                        exact replicate_all_stops f
                      · simp only [hcls, decide_eq_false hφ, Bool.false_eq_true,
                          if_false]
                        exact ih e.END.FORMULA hcondE



lemma foldl_max_ge_init : ∀ (l : List Nat) (init : Nat), init ≤ l.foldl max init := by
  intro l
  induction l with
  | nil => intro init; simp
  | cons a t ih => intro init; exact le_trans (le_max_left init a) (ih (max init a))

lemma foldl_max_ge : ∀ (l : List Nat) (init x : Nat), x ∈ l → x ≤ l.foldl max init := by
  intro l
  induction l with
  | nil => intro init x hx; cases hx
  | cons a t ih =>
      intro init x hx
      cases hx with
      | head => exact le_trans (le_max_right init a) (foldl_max_ge_init t (max init a))
      | tail _ hxt => exact ih (max init a) x hxt

lemma level_le_maxLvl (d : Graph) {v : Vertex} (hv : v ∈ d.NODES) :
    v.LEVEL ≤ (d.NODES.map (·.LEVEL)).foldl max 0 :=
  foldl_max_ge _ 0 v.LEVEL (List.mem_map.mpr ⟨v, hv, rfl⟩)

lemma numSteps_eq_maxLvl (d : Graph) :
    (buildGridFromDLDS d).length - 1 = (d.NODES.map (·.LEVEL)).foldl max 0 := by
  unfold buildGridFromDLDS buildLayers
  simp [List.length_reverse, List.length_replicate]

lemma admHypColumnB_paddings (d : Graph) (formulas : List Formula) (col : Nat)
    (φ : Formula) :
    ∀ (delay : Nat) (chain : List (Nat × Nat)),
      admHypColumnB d formulas col φ delay
        (List.replicate delay (col + 1, 0) ++ chain) =
      admChainB d formulas φ chain := by
  intro delay
  induction delay with
  | zero => intro chain; rfl
  | succ k ih =>
      intro chain
      rw [List.replicate_succ, List.cons_append, admHypColumnB]
      simp only [beq_self_eq_true, Bool.and_self, Bool.true_and]
      exact ih chain



lemma canonical_column_admissible (d : Graph) (htree : IsSimpleTreeDLDS d)
    (hvalid : ValidDLDS d) (c : Nat) (hc : c < (buildFormulas d).length) :
    admColumnB d (buildFormulas d) c ((pathsFromDLDS d).getD c []) = true := by
  have hlev : ∀ e ∈ d.EDGES, e.START.LEVEL = e.END.LEVEL + 1 := hvalid.leveledColored
  have hhyg : ∀ e ∈ d.EDGES, e.START ∈ d.NODES ∧ e.END ∈ d.NODES :=
    fun e he => hvalid.hygiene.2.1 he
  have hinj : ∀ u ∈ d.NODES, ∀ w ∈ d.NODES, u.FORMULA = w.FORMULA → u = w := htree.2
  set formulas := buildFormulas d with hform
  have hφ : formulas[c]? = some (formulas[c]'hc) := List.getElem?_eq_getElem hc
  set φ := formulas[c]'hc with hφdef
  have hnd : formulas.Nodup := by rw [hform]; exact buildFormulas_nodup_bridge d
  have hidx : formulas.idxOf φ = c := by
    rw [hφdef]; exact List.Nodup.idxOf_getElem hnd c hc
  have hns : (buildGridFromDLDS d).length - 1 = (d.NODES.map (·.LEVEL)).foldl max 0 :=
    numSteps_eq_maxLvl d
  rw [List.getD_eq_getElem?_getD, pathsFromDLDS_get?_of_formula d hφ, Option.getD_some]
  unfold admColumnB
  rw [hφ]
  cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
  | none => simp only [hfind]; exact replicate_all_stops _
  | some v =>
      simp only [hfind]
      have hvmem : v ∈ d.NODES := find?_some_mem hfind
      have hvform : v.FORMULA = φ := of_decide_eq_true (by
        have h := List.find?_some hfind; simpa using h)
      by_cases hhyp : v.HYPOTHESIS
      · simp only [hhyp, if_true]
        rw [hidx, hns]
        have hvle : v.LEVEL ≤ (d.NODES.map (·.LEVEL)).foldl max 0 := level_le_maxLvl d hvmem
        have hfuel : (d.NODES.map (·.LEVEL)).foldl max 0 -
            ((d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) = v.LEVEL := by omega
        rw [hfuel]
        rw [admHypColumnB_paddings]
        apply routeFrom_admChainB d formulas hlev hhyg hinj
        intro w hw hwf
        have hweq : w = v := hinj w hw v hvmem (by rw [hwf, hvform])
        rw [hweq]
      · simp only [hhyp, Bool.false_eq_true, if_false]
        exact replicate_all_stops _

def AdmissibleReducesToCanonical (d : Graph) : Prop :=
  ∀ P, AdmissibleDLDSPath d P → ReadEquiv P (pathsFromDLDS d)

theorem admissibleReducesToCanonical (d : Graph) (htree : IsSimpleTreeDLDS d)
    (hvalid : ValidDLDS d) : AdmissibleReducesToCanonical d := by
  intro P hadm c s
  have hadmB : admissibleDLDSPathB d P = true := hadm
  unfold admissibleDLDSPathB at hadmB
  rw [Bool.and_eq_true, Bool.and_eq_true] at hadmB
  obtain ⟨⟨hlen, hcols⟩, _⟩ := hadmB
  have hlenEq : P.length = (buildFormulas d).length := by simpa using hlen
  by_cases hc : c < (buildFormulas d).length
  · have hcP : c < P.length := by rw [hlenEq]; exact hc
    have hcC : c < (pathsFromDLDS d).length := by rw [pathsFromDLDS_length]; exact hc
    rw [effStep_colEff hcP, effStep_colEff hcC]
    have hPcol : admColumnB d (buildFormulas d) c (P.getD c []) = true :=
      (List.all_eq_true.mp hcols) c (List.mem_range.mpr hc)
    have hCcol := canonical_column_admissible d htree hvalid c hc
    exact admColumn_colEff_unique d (buildFormulas d) c _ _ hPcol hCcol s
  · rw [effStep_ge_col (by rw [hlenEq]; exact hc),
      effStep_ge_col (by rw [pathsFromDLDS_length]; exact hc)]

/--
 On a valid simple tree
    with a discharged goal, EVERY path input is either inadmissible (the faithful
    `Invalid = ¬AdmissibleDLDSPath`) or discharges the goal. The admissible case
    is reduced to the canonical path by read-equivalence
    (`admissibleReducesToCanonical`, now a proved theorem) and then discharged by
    the existing `discharge` lemma ; so the universal `Accept` holds, with no path
    slipping through a vacuous acceptance.
-/
theorem simpleTree_universally_accepted
    (d : Graph) (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    (hdis : dischargedB d = true) :
    ∀ P, ¬ AdmissibleDLDSPath d P ∨ Discharged d P (goalColumn d) := by
  intro P
  by_cases hadm : AdmissibleDLDSPath d P
  · right
    have hRE : ReadEquiv P (pathsFromDLDS d) :=
      admissibleReducesToCanonical d htree hvalid P hadm
    have hcanon : Discharged d (pathsFromDLDS d) (goalColumn d) :=
      discharge d htree.1 hvalid hdis
    rw [Discharged_congr (goalColumn d) hRE]
    exact hcanon
  · left; exact hadm



end Semantic
