import Semantic.Evaluator
import HorizontalCompressionEXEC

open scoped Classical

namespace Semantic

/-!
# DLDS extraction, validity predicates, and structural helpers.
-/

/-- Extract the list of unique formulas from a DLDS.
    This forms the "column universe" for the circuit grid. -/
def buildFormulas (d : Graph) : List Formula :=
  (d.NODES.map (·.FORMULA)).eraseDups



/-- Generate encoder vector for an implication introduction rule.
    For A⊃B, creates a vector with bit i = true iff formulas[i] = A.
    This encodes which assumption gets discharged. -/
def encoderForIntro (formulas : List Formula) (φ : Formula)
    : Option (List.Vector Bool formulas.length) :=
  match φ with
  | .implication A _ =>
      some ⟨formulas.map (fun ψ => decide (ψ = A)), by rw [List.length_map]⟩
  | _ => none



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
    | .implication _ B =>
        let b_idx := formulas.idxOf B
        [[(b_idx, 0)]]
    | _ => []
  let elimMaps := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .implication A B =>
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




lemma nodeForFormula_nodupIds (formulas : List Formula) (formula : Formula) :
    let introData := match formula with
      | .implication _ _ => match encoderForIntro formulas formula with
        | some encoder => [encoder]
        | none => []
      | _ => []
    let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
      match f with
      | .implication _ B => if B = formula then some idx else none
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
    | .implication _ _ =>
        match encoderForIntro formulas formula with
        | some encoder => [encoder]
        | none => []
    | _ => []

  let elimData := formulas.zipIdx.filterMap fun (f, idx) =>
    match f with
    | .implication _ B =>
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



/-- Build all layers for a DLDS.
    Each layer is identical (same nodes and wiring), replicated for each level. -/
def buildLayers (d : Graph) : List (GridLayer (buildFormulas d).length) :=
  let formulas := buildFormulas d
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  List.replicate (maxLvl + 1)
    { nodes := formulas.map (nodeForFormula formulas)
      incoming := buildIncomingMap formulas
    }



/-- Build complete grid from DLDS

    Note: Layers are reversed because evaluation proceeds top-to-bottom
    but we build layers 0→max
-/
def buildGridFromDLDS (d : Graph) : List (GridLayer (buildFormulas d).length) :=
  buildLayers d |>.reverse



/-- Flow top-node base case of Definition 3: one-hot dependency bitstrings
    `(b⃗_{l(v)}, ε)` for the formula columns before any residual path is followed. -/
def initialVectorsFromDLDS (d : Graph)
  : List (List.Vector Bool (buildFormulas d).length) :=
  let n := (buildFormulas d).length
  List.range n |>.map fun i =>
    (⟨List.range n |>.map (fun j => decide (j = i)), by
      simp [List.length_map, List.length_range]⟩ : List.Vector Bool n)

/-- Boolean encoding of a DLDS dependency set as the Definition 3 dependency
    bitstring `b⃗`; bit `i` is true exactly when `formulas[i]` is listed. -/
def depVector (formulas : List Formula) (deps : List Formula) :
    List.Vector Bool formulas.length :=
  ⟨formulas.map (fun φ => decide (φ ∈ deps)), by simp⟩

lemma depVector_get (formulas deps : List Formula) (i : Fin formulas.length) :
    (depVector formulas deps).get i = decide (formulas.get i ∈ deps) := by
  rcases i with ⟨i, hi⟩
  simp [depVector, List.Vector.get]

lemma depVector_nil_get (formulas : List Formula) (i : Fin formulas.length) :
    (depVector formulas []).get i = false := by
  simp [depVector_get]

lemma depVector_nil_eq_replicate (formulas : List Formula) :
    depVector formulas [] = List.Vector.replicate formulas.length false := by
  apply List.Vector.ext
  intro i
  simp [depVector_nil_get, List.Vector.get_replicate]

lemma empty_dep_to_vector_eq_false_vector (d : Graph) :
    ∀ i : Fin (buildFormulas d).length,
      (depVector (buildFormulas d) []).get i = false := by
  intro i
  exact depVector_nil_get (buildFormulas d) i

lemma depVector_removeAll_singleton_eq_zipWith_encoder
    (formulas : List Formula) (deps : List Formula) (A : Formula) :
    depVector formulas (List.eraseDups (List.removeAll deps [A])) =
      (depVector formulas deps).zipWith
        (fun b e => b && !e)
        (depVector formulas [A]) := by
  apply List.Vector.ext
  intro i
  dsimp [depVector, List.Vector.get, List.Vector.zipWith]
  rw [List.getElem_zipWith]
  simp only [List.getElem_map]
  by_cases hdep : formulas.get i ∈ deps
  · by_cases hA : formulas.get i = A
    · simp [List.mem_eraseDups, List.removeAll, List.mem_filter]
    · have hnotA : ¬ formulas.get i ∈ [A] := by
        simpa [List.mem_singleton] using hA
      simp [List.mem_eraseDups, List.removeAll, List.mem_filter]
  · simp [List.mem_eraseDups, List.removeAll, List.mem_filter]

lemma depVector_difference_singleton_eq_zipWith_encoder
    (formulas : List Formula) (deps : List Formula) (A : Formula) :
    depVector formulas (deps − [A]) =
      (depVector formulas deps).zipWith
        (fun b e => b && !e)
        (depVector formulas [A]) := by
  exact depVector_removeAll_singleton_eq_zipWith_encoder formulas deps A

lemma depVector_eraseDups_append_eq_zipWith_or
    (formulas : List Formula) (xs ys : List Formula) :
    depVector formulas (List.eraseDups (xs ++ ys)) =
      (depVector formulas xs).zipWith (· || ·) (depVector formulas ys) := by
  apply List.Vector.ext
  intro i
  dsimp [depVector, List.Vector.get, List.Vector.zipWith]
  rw [List.getElem_zipWith]
  simp only [List.getElem_map]
  by_cases hx : formulas.get i ∈ xs
  · simp [List.mem_eraseDups, List.mem_append]
  · by_cases hy : formulas.get i ∈ ys
    · simp [List.mem_eraseDups, List.mem_append]
    · simp [List.mem_eraseDups, List.mem_append]

lemma depVector_singleton_get (formulas : List Formula) (A : Formula)
    (i : Fin formulas.length) :
    (depVector formulas [A]).get i = decide (formulas.get i = A) := by
  simp [depVector_get]

lemma depVector_repetition_eq (formulas deps : List Formula) :
    depVector formulas deps = depVector formulas deps := rfl



/-- An intro rule is well-formed if its encoder correctly marks
    the discharged assumption: bit i is true iff formulas[i] = A
    where the formula is A⊃B. -/
def IntroRuleWellFormed {n : Nat}
  (encoder : List.Vector Bool n)
  (formula : Formula)
  (formulas : List Formula) : Prop :=
  match formula with
  | .implication A _ =>
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
    match encoderForIntro formulas (Formula.implication A B) with
    | some encoder => IntroRuleWellFormed encoder (Formula.implication A B) formulas
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
    ∀ rule ∈ (nodeForFormula formulas (Formula.implication A B)).rules,
      match rule.type with
      | RuleData.intro encoder => IntroRuleWellFormed encoder (Formula.implication A B) formulas
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
      have h_enc_eq : encoderForIntro formulas (Formula.implication A B) =
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
  | implication A B =>
    exact nodeForFormula_impl_rules_wellformed formulas A B

lemma buildIncomingMap_length (formulas : List Formula) :
    (buildIncomingMap formulas).length = formulas.length := by
  simp only [buildIncomingMap, List.length_map]

/-- **Construction Correctness**: The grid built from a DLDS is well-formed. -/
theorem buildGridFromDLDS_wellformed (d : Graph) :
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


/-- Evaluate a DLDS graph with a chosen path assignment and goal column. -/
def evaluateDLDS (d : Graph) (paths : PathInput) (goal_column : Nat) : Bool :=
  let grid := buildGridFromDLDS d
  let initial_vecs := initialVectorsFromDLDS d
  evaluateCircuit grid initial_vecs paths goal_column



/-- Consequent of an implication, else `none`. Used to match the ⊃I premise. -/
def consequent? : Formula → Option Formula
  | Formula.implication _ β => some β
  | _                       => none

/-- Antecedent of an implication, else `none`. The assumption discharged by ⊃I. -/
def antecedent? : Formula → Option Formula
  | Formula.implication α _ => some α
  | _                       => none

/-- Shared classifier for the natural-deduction rule applied at a DLDS node. -/
inductive DLDSRuleClass where
  | hypothesis
  | intro (premise : Deduction)
  | elim (major minor : Deduction)
  deriving Repr

/-- Classify the actual rule at `v` from `get_rule.incoming v d`. The elim case
    stores `(major, minor)` in formula-matched order. -/
def classifyRule? (v : Vertex) (d : Graph) : Option DLDSRuleClass :=
  let inc := get_rule.incoming v d
  if v.HYPOTHESIS = true then
    some DLDSRuleClass.hypothesis
  else
    match inc with
    | [p] =>
        if consequent? v.FORMULA = some p.START.FORMULA then
          some (DLDSRuleClass.intro p)
        else
          none
    | [e₁, e₂] =>
        if e₁.START.FORMULA = Formula.implication e₂.START.FORMULA v.FORMULA then
          some (DLDSRuleClass.elim e₁ e₂)
        else if e₂.START.FORMULA = Formula.implication e₁.START.FORMULA v.FORMULA then
          some (DLDSRuleClass.elim e₂ e₁)
        else
          none
    | _ => none

def introRuleCount (formula : Formula) : Nat :=
  match formula with
  | Formula.implication _ _ => 1
  | _ => 0

/-- Position, among the elimination rules for `target`, of the major premise
    formula `majorFormula`, using exactly `nodeForFormula`'s elim ordering. -/
def elimRulePosition? (formulas : List Formula) (target majorFormula : Formula) : Option Nat :=
  let rec loop (pos : Nat) : List Formula → Option Nat
    | [] => none
    | f :: fs =>
        match f with
        | Formula.implication _ B =>
            if B = target then
              if f = majorFormula then some pos else loop (pos + 1) fs
            else
              loop pos fs
        | _ => loop pos fs
  loop 0 formulas

/-- Rule-list index of the actual rule applied at `v`, using the same ordering as
    `nodeForFormula` / `buildIncomingMapForFormula`. -/
def ruleIndexForNode? (d : Graph) (formulas : List Formula) (v : Vertex) : Option Nat :=
  match classifyRule? v d with
  | none => none
  | some DLDSRuleClass.hypothesis =>
      let incoming := buildIncomingMapForFormula formulas v.FORMULA
      if incoming.length > 0 then some (incoming.length - 1) else none
  | some (DLDSRuleClass.intro _) =>
      match v.FORMULA with
      | Formula.implication _ _ => some 0
      | _ => none
  | some (DLDSRuleClass.elim major _) =>
      match elimRulePosition? formulas v.FORMULA major.START.FORMULA with
      | some pos => some (introRuleCount v.FORMULA + pos)
      | none => none

lemma elim_filterMap_lengths_eq (formulas : List Formula) (target : Formula) :
    ∀ xs : List (Formula × Nat),
      (xs.filterMap (fun x =>
        match x.1 with
        | Formula.implication A B =>
            if B = target then some [(x.2, 0), (formulas.idxOf A, 0)] else none
        | _ => none)).length =
      (xs.filterMap (fun x =>
        match x.1 with
        | Formula.implication _ B =>
            if B = target then some x.2 else none
        | _ => none)).length
  | [] => by simp
  | (f, idx) :: xs => by
      cases f with
      | atom name =>
          simp [elim_filterMap_lengths_eq formulas target xs]
      | implication A B =>
          by_cases hB : B = target
          · simp [hB, elim_filterMap_lengths_eq formulas target xs]
          · simp [hB, elim_filterMap_lengths_eq formulas target xs]

lemma incoming_rules_aligned_length (formulas : List Formula) (formula : Formula) :
    (buildIncomingMapForFormula formulas formula).length =
      (nodeForFormula formulas formula).rules.length := by
  unfold buildIncomingMapForFormula nodeForFormula
  cases formula with
  | atom name =>
      simp [elim_filterMap_lengths_eq formulas (Formula.atom name) formulas.zipIdx]
  | implication A B =>
      simp [encoderForIntro,
        elim_filterMap_lengths_eq formulas (Formula.implication A B) formulas.zipIdx]

lemma incoming_rules_aligned (formulas : List Formula) (formula : Formula) :
    (buildIncomingMapForFormula formulas formula).length =
      (nodeForFormula formulas formula).rules.length :=
  incoming_rules_aligned_length formulas formula

lemma elimRulePosition_loop_ge_and_indexes_elim_entry
    (formulas : List Formula) (target major : Formula) :
    ∀ (fs : List Formula) (offset start found : Nat),
      elimRulePosition?.loop target major start fs = some found →
      start ≤ found ∧
      (((fs.zipIdx offset).filterMap (fun x =>
        match x.1 with
        | Formula.implication A B =>
            if B = target then some [(x.2, 0), (formulas.idxOf A, 0)] else none
        | _ => none))[found - start]?.getD default).length = 2
  | [], offset, start, found, h => by
      simp [elimRulePosition?.loop] at h
  | f :: fs, offset, start, found, h => by
      cases f with
      | atom name =>
          simp [elimRulePosition?.loop] at h
          exact elimRulePosition_loop_ge_and_indexes_elim_entry
            formulas target major fs (offset + 1) start found h
      | implication A B =>
          by_cases hB : B = target
          · by_cases hf : Formula.implication A B = major
            · subst hf
              simp [elimRulePosition?.loop, hB] at h
              subst h
              simp [hB]
            · have hneq : ¬ Formula.implication A target = major := by
                intro hmajor
                exact hf (by simpa [hB] using hmajor)
              simp [elimRulePosition?.loop, hB, hneq] at h
              have ih := elimRulePosition_loop_ge_and_indexes_elim_entry
                formulas target major fs (offset + 1) (start + 1) found h
              constructor
              · omega
              · rcases ih with ⟨hge, hidx⟩
                have hsub : found - start = Nat.succ (found - (start + 1)) := by
                  omega
                simp [hB, hsub, hidx]
          · simp [elimRulePosition?.loop, hB] at h
            have ih := elimRulePosition_loop_ge_and_indexes_elim_entry
              formulas target major fs (offset + 1) start found h
            rcases ih with ⟨hge, hidx⟩
            exact ⟨hge, by simpa [hB] using hidx⟩

lemma getElem?_getD_length_two_lt (xs : List RuleIncoming) (i : Nat)
    (h : (xs[i]?.getD default).length = 2) :
    i < xs.length := by
  by_contra hlt
  have hnone : xs[i]? = none := by
    rw [List.getElem?_eq_none_iff]
    omega
  have hdef : (default : RuleIncoming) = [] := rfl
  rw [hnone] at h
  rw [hdef] at h
  simp at h

lemma getElem?_getD_length_pos_lt (xs : List RuleIncoming) (i : Nat)
    (h : 0 < (xs[i]?.getD default).length) :
    i < xs.length := by
  by_contra hlt
  have hnone : xs[i]? = none := by
    rw [List.getElem?_eq_none_iff]
    omega
  have hdef : (default : RuleIncoming) = [] := rfl
  rw [hnone] at h
  rw [hdef] at h
  simp at h

lemma getD_append_left_length_two (xs ys : List RuleIncoming) (i : Nat)
    (h : (xs[i]?.getD default).length = 2) :
    (((xs ++ ys)[i]?.getD default).length = 2) := by
  have hlt : i < xs.length := getElem?_getD_length_two_lt xs i h
  rw [List.getElem?_append_left hlt]
  exact h

lemma elimRulePosition?_indexes_elim_entry
    (formulas : List Formula) (target major : Formula) (pos : Nat)
    (hpos : elimRulePosition? formulas target major = some pos) :
    (((buildIncomingMapForFormula formulas target)[introRuleCount target + pos]?.getD default).length = 2) := by
  unfold elimRulePosition? at hpos
  have h :=
    elimRulePosition_loop_ge_and_indexes_elim_entry
      formulas target major formulas 0 0 pos hpos
  rcases h with ⟨_, hidx⟩
  unfold buildIncomingMapForFormula introRuleCount
  cases target with
  | atom name =>
      let elimMaps :=
        (formulas.zipIdx.filterMap (fun x =>
          match x.1 with
          | Formula.implication A B =>
              if B = Formula.atom name then some [(x.2, 0), (formulas.idxOf A, 0)] else none
          | _ => none))
      have hidx0 : (elimMaps[pos]?.getD default).length = 2 := by
        simpa [elimMaps] using hidx
      simpa [buildIncomingMapForFormula, introRuleCount, elimMaps] using
        getD_append_left_length_two elimMaps
          [[(formulas.idxOf (Formula.atom name), 0)]] pos hidx0
  | implication A B =>
      let elimMaps :=
        (formulas.zipIdx.filterMap (fun x =>
          match x.1 with
          | Formula.implication A_1 B_1 =>
              if B_1 = Formula.implication A B then
                some [(x.2, 0), (formulas.idxOf A_1, 0)]
              else none
          | _ => none))
      have hidx0 : (elimMaps[pos]?.getD default).length = 2 := by
        simpa [elimMaps] using hidx
      have happ := getD_append_left_length_two elimMaps
        [[(formulas.idxOf (Formula.implication A B), 0)]] pos hidx0
      have hsucc : 1 + pos = Nat.succ pos := by omega
      rw [hsucc]
      simpa [buildIncomingMapForFormula, introRuleCount, elimMaps] using happ

private lemma eraseDupsBy_loop_nodup_bridge {α : Type*} [BEq α] [LawfulBEq α] :
    ∀ (xs acc : List α), acc.Nodup →
      (List.eraseDupsBy.loop (fun x y : α => x == y) xs acc).Nodup
  | [], acc, hacc => by
      rw [List.eraseDupsBy.loop.eq_1]
      exact List.nodup_reverse.mpr hacc
  | a :: xs, acc, hacc => by
      rw [List.eraseDupsBy.loop.eq_2]
      cases h : acc.any (fun y => a == y)
      · apply eraseDupsBy_loop_nodup_bridge xs (a :: acc)
        exact List.nodup_cons.mpr ⟨by
          intro hmem
          have hany : acc.any (fun y => a == y) = true := by
            rw [List.any_eq_true]
            exact ⟨a, hmem, by simp⟩
          rw [h] at hany
          contradiction, hacc⟩
      · exact eraseDupsBy_loop_nodup_bridge xs acc hacc

lemma buildFormulas_nodup_bridge (d : Graph) :
    (buildFormulas d).Nodup := by
  unfold buildFormulas
  rw [List.eraseDups.eq_1]
  exact eraseDupsBy_loop_nodup_bridge (d.NODES.map (·.FORMULA)) [] (by simp)

private lemma idxOf_head_after_prefix {Alpha : Type*} [DecidableEq Alpha]
    (pref suffix : List Alpha) (x : Alpha)
    (hnodup : (pref ++ x :: suffix).Nodup) :
    (pref ++ x :: suffix).idxOf x = pref.length := by
  have hidx : pref.length < (pref ++ x :: suffix).length := by simp
  apply indexOf_eq_of_get (l := pref ++ x :: suffix) (i := pref.length) hidx hnodup
  rw [List.get_eq_getElem]
  simp

lemma elimRulePosition_loop_entry_sources_aux
    (target A : Formula) :
    ∀ (pref fs : List Formula) (start found : Nat),
      (pref ++ fs).Nodup →
      elimRulePosition?.loop target (Formula.implication A target) start fs = some found →
      (((fs.zipIdx pref.length).filterMap (fun x =>
        match x.1 with
        | Formula.implication A' B =>
            if B = target then some [(x.2, 0), ((pref ++ fs).idxOf A', 0)] else none
        | _ => none))[found - start]?).map (fun inc => inc.map Prod.fst) =
          some [(pref ++ fs).idxOf (Formula.implication A target),
            (pref ++ fs).idxOf A]
    := by
  intro pref fs
  induction fs generalizing pref with
  | nil =>
      intro start found _ h
      simp [elimRulePosition?.loop] at h
  | cons f fs ih =>
      intro start found hnodup h
      cases f with
      | atom name =>
          simp [elimRulePosition?.loop] at h
          have ih' := ih (pref ++ [Formula.atom name]) start found
            (by simpa [List.append_assoc] using hnodup) h
          simpa [List.zipIdx_cons, List.append_assoc] using ih'
      | implication A' B =>
          by_cases hBtarget : B = target
          · by_cases hf : Formula.implication A' B = Formula.implication A target
            · cases hf
              simp [elimRulePosition?.loop] at h
              subst h
              have hidx :
                  (pref ++ Formula.implication A target :: fs).idxOf
                      (Formula.implication A target) = pref.length :=
                idxOf_head_after_prefix pref fs (Formula.implication A target) hnodup
              simp [hidx]
            · have hneq : ¬ Formula.implication A' target = Formula.implication A target := by
                intro hmaj
                exact hf (by simpa [hBtarget] using hmaj)
              simp [elimRulePosition?.loop, hBtarget, hneq] at h
              have ih' := ih (pref ++ [Formula.implication A' B]) (start + 1) found
                (by simpa [List.append_assoc] using hnodup) h
              have hsub : found - start = Nat.succ (found - (start + 1)) := by
                have hge := elimRulePosition_loop_ge_and_indexes_elim_entry
                  (pref ++ Formula.implication A' B :: fs)
                  target (Formula.implication A target) fs (pref.length + 1)
                  (start + 1) found h
                omega
              simpa [hBtarget, hsub, List.append_assoc] using ih'
          · simp [elimRulePosition?.loop, hBtarget] at h
            have ih' := ih (pref ++ [Formula.implication A' B]) start found
              (by simpa [List.append_assoc] using hnodup) h
            simpa [hBtarget, List.append_assoc] using ih'

lemma elimRulePosition?_entry_sources_imp
    (d : Graph) (target A : Formula) (pos : Nat)
    (hpos : elimRulePosition? (buildFormulas d) target
      (Formula.implication A target) = some pos) :
    Option.map (fun inc => inc.map Prod.fst)
      ((buildIncomingMapForFormula (buildFormulas d) target)[introRuleCount target + pos]?) =
      some [(buildFormulas d).idxOf (Formula.implication A target),
        (buildFormulas d).idxOf A] := by
  let formulas := buildFormulas d
  unfold elimRulePosition? at hpos
  have hsrc := elimRulePosition_loop_entry_sources_aux target A
    ([] : List Formula) formulas 0 pos (by simpa [formulas] using buildFormulas_nodup_bridge d)
    hpos
  unfold buildIncomingMapForFormula introRuleCount
  cases target with
  | atom name =>
      let elimMaps :=
        (formulas.zipIdx.filterMap (fun x =>
          match x.1 with
          | Formula.implication A_1 B =>
              if B = Formula.atom name then some [(x.2, 0), (formulas.idxOf A_1, 0)] else none
          | _ => none))
      have hsrc0 :
          (elimMaps[pos]?).map (fun inc => inc.map Prod.fst) =
            some [formulas.idxOf (Formula.implication A (Formula.atom name)),
              formulas.idxOf A] := by
        simpa [elimMaps, formulas] using hsrc
      have hlt : pos < elimMaps.length := by
        by_contra hnot
        have hnone : elimMaps[pos]? = none := by
          rw [List.getElem?_eq_none_iff]
          omega
        rw [hnone] at hsrc0
        simp at hsrc0
      simpa [buildIncomingMapForFormula, introRuleCount, elimMaps, formulas,
        List.getElem?_append_left hlt] using hsrc0
  | implication A0 B0 =>
      let elimMaps :=
        (formulas.zipIdx.filterMap (fun x =>
          match x.1 with
          | Formula.implication A_1 B_1 =>
              if B_1 = Formula.implication A0 B0 then
                some [(x.2, 0), (formulas.idxOf A_1, 0)]
              else none
          | _ => none))
      have hsrc0 :
          (elimMaps[pos]?).map (fun inc => inc.map Prod.fst) =
            some [formulas.idxOf (Formula.implication A (Formula.implication A0 B0)),
              formulas.idxOf A] := by
        simpa [elimMaps, formulas] using hsrc
      have hlt : pos < elimMaps.length := by
        by_contra hnot
        have hnone : elimMaps[pos]? = none := by
          rw [List.getElem?_eq_none_iff]
          omega
        rw [hnone] at hsrc0
        simp at hsrc0
      have hsucc : 1 + pos = Nat.succ pos := by omega
      rw [hsucc]
      simpa [buildIncomingMapForFormula, introRuleCount, elimMaps, formulas,
        List.getElem?_append_left hlt] using hsrc0

lemma elimRulePosition_loop_isSome_of_mem
    (target majorFormula : Formula) :
    ∀ (fs : List Formula) (pos : Nat),
      majorFormula ∈ fs →
      (∃ A, majorFormula = Formula.implication A target) →
      ∃ found, elimRulePosition?.loop target majorFormula pos fs = some found
  | [], _, hmem, _ => by simp at hmem
  | f :: fs, pos, hmem, hshape => by
      cases hmem with
      | head =>
        obtain ⟨A, hmajor⟩ := hshape
        rw [hmajor]
        simp [elimRulePosition?.loop]
      | tail _ htail =>
        cases f with
        | atom name =>
            simp [elimRulePosition?.loop]
            exact elimRulePosition_loop_isSome_of_mem target majorFormula fs pos htail hshape
        | implication A B =>
            by_cases hB : B = target
            · by_cases hf : Formula.implication A B = majorFormula
              · subst hf
                exact ⟨pos, by simp [elimRulePosition?.loop, hB]⟩
              · have hneq : ¬ Formula.implication A target = majorFormula := by
                  intro h
                  exact hf (by simpa [hB] using h)
                simp [elimRulePosition?.loop, hB, hneq]
                exact elimRulePosition_loop_isSome_of_mem target majorFormula fs (pos + 1) htail hshape
            · simp [elimRulePosition?.loop, hB]
              exact elimRulePosition_loop_isSome_of_mem target majorFormula fs pos htail hshape

lemma elimRulePosition?_isSome_of_mem
    (formulas : List Formula) (target majorFormula : Formula)
    (hmem : majorFormula ∈ formulas)
    (hshape : ∃ A, majorFormula = Formula.implication A target) :
    ∃ pos, elimRulePosition? formulas target majorFormula = some pos := by
  unfold elimRulePosition?
  exact elimRulePosition_loop_isSome_of_mem target majorFormula formulas 0 hmem hshape

lemma mem_incoming_loop_mem_edges (v : Vertex) {e : Deduction} :
    ∀ edges : List Deduction,
      e ∈ get_rule.incoming.loop v edges → e ∈ edges
  | [], he => by
      rw [get_rule.incoming.loop.eq_1] at he
      simp at he
  | ed :: edges, he => by
      rw [get_rule.incoming.loop.eq_2] at he
      by_cases hend : ed.END = v
      · simp [hend] at he
        exact he.elim (fun h => by subst h; exact List.Mem.head edges)
          (fun h => List.mem_cons_of_mem ed
            (mem_incoming_loop_mem_edges v edges h))
      · simp [hend] at he
        exact List.mem_cons_of_mem ed
          (mem_incoming_loop_mem_edges v edges he)

lemma mem_incoming_loop_end_eq (v : Vertex) {e : Deduction} :
    ∀ edges : List Deduction,
      e ∈ get_rule.incoming.loop v edges → e.END = v
  | [], he => by
      rw [get_rule.incoming.loop.eq_1] at he
      simp at he
  | ed :: edges, he => by
      rw [get_rule.incoming.loop.eq_2] at he
      by_cases hend : ed.END = v
      · simp [hend] at he
        exact he.elim (fun h => by subst h; exact hend)
          (fun h => mem_incoming_loop_end_eq v edges h)
      · simp [hend] at he
        exact mem_incoming_loop_end_eq v edges he

lemma mem_incoming_mem_edges (v : Vertex) (d : Graph) {e : Deduction}
    (he : e ∈ get_rule.incoming v d) : e ∈ d.EDGES := by
  rw [get_rule.incoming.eq_1] at he
  exact mem_incoming_loop_mem_edges v d.EDGES he

lemma mem_incoming_end_eq (v : Vertex) (d : Graph) {e : Deduction}
    (he : e ∈ get_rule.incoming v d) : e.END = v := by
  rw [get_rule.incoming.eq_1] at he
  exact mem_incoming_loop_end_eq v d.EDGES he

lemma mem_incoming_loop_of_mem_edges_end_eq (v : Vertex) {e : Deduction} :
    ∀ edges : List Deduction,
      e ∈ edges → e.END = v → e ∈ get_rule.incoming.loop v edges
  | [], hmem, _ => by simp at hmem
  | ed :: edges, hmem, hend => by
      rw [get_rule.incoming.loop.eq_2]
      by_cases heq : ed.END = v
      · simp [heq]
        cases hmem with
        | head =>
            simp
        | tail _ htail =>
            exact Or.inr (mem_incoming_loop_of_mem_edges_end_eq v edges htail hend)
      · simp [heq]
        cases hmem with
        | head =>
            exact False.elim (heq hend)
        | tail _ htail =>
            exact mem_incoming_loop_of_mem_edges_end_eq v edges htail hend

lemma mem_incoming_of_mem_edges_end_eq (v : Vertex) (d : Graph) {e : Deduction}
    (hedge : e ∈ d.EDGES) (hend : e.END = v) :
    e ∈ get_rule.incoming v d := by
  rw [get_rule.incoming.eq_1]
  exact mem_incoming_loop_of_mem_edges_end_eq v d.EDGES hedge hend

lemma mem_outgoing_loop_mem_edges (v : Vertex) {e : Deduction} :
    ∀ edges : List Deduction,
      e ∈ get_rule.outgoing.loop v edges → e ∈ edges
  | [], he => by
      rw [get_rule.outgoing.loop.eq_1] at he
      simp at he
  | ed :: edges, he => by
      rw [get_rule.outgoing.loop.eq_2] at he
      by_cases hstart : ed.START = v
      · simp [hstart] at he
        exact he.elim (fun h => by subst h; exact List.Mem.head edges)
          (fun h => List.mem_cons_of_mem ed
            (mem_outgoing_loop_mem_edges v edges h))
      · simp [hstart] at he
        exact List.mem_cons_of_mem ed
          (mem_outgoing_loop_mem_edges v edges he)

lemma mem_outgoing_loop_start_eq (v : Vertex) {e : Deduction} :
    ∀ edges : List Deduction,
      e ∈ get_rule.outgoing.loop v edges → e.START = v
  | [], he => by
      rw [get_rule.outgoing.loop.eq_1] at he
      simp at he
  | ed :: edges, he => by
      rw [get_rule.outgoing.loop.eq_2] at he
      by_cases hstart : ed.START = v
      · simp [hstart] at he
        exact he.elim (fun h => by subst h; exact hstart)
          (fun h => mem_outgoing_loop_start_eq v edges h)
      · simp [hstart] at he
        exact mem_outgoing_loop_start_eq v edges he

lemma mem_outgoing_mem_edges (v : Vertex) (d : Graph) {e : Deduction}
    (he : e ∈ get_rule.outgoing v d) : e ∈ d.EDGES := by
  rw [get_rule.outgoing.eq_1] at he
  exact mem_outgoing_loop_mem_edges v d.EDGES he

lemma mem_outgoing_start_eq (v : Vertex) (d : Graph) {e : Deduction}
    (he : e ∈ get_rule.outgoing v d) : e.START = v := by
  rw [get_rule.outgoing.eq_1] at he
  exact mem_outgoing_loop_start_eq v d.EDGES he

lemma mem_outgoing_loop_of_mem_edges_start_eq (v : Vertex) {e : Deduction} :
    ∀ edges : List Deduction,
      e ∈ edges → e.START = v → e ∈ get_rule.outgoing.loop v edges
  | [], hmem, _ => by simp at hmem
  | ed :: edges, hmem, hstart => by
      rw [get_rule.outgoing.loop.eq_2]
      by_cases heq : ed.START = v
      · simp [heq]
        cases hmem with
        | head =>
            simp
        | tail _ htail =>
            exact Or.inr (mem_outgoing_loop_of_mem_edges_start_eq v edges htail hstart)
      · simp [heq]
        cases hmem with
        | head =>
            exact False.elim (heq hstart)
        | tail _ htail =>
            exact mem_outgoing_loop_of_mem_edges_start_eq v edges htail hstart

lemma mem_outgoing_of_mem_edges_start_eq (v : Vertex) (d : Graph) {e : Deduction}
    (hedge : e ∈ d.EDGES) (hstart : e.START = v) :
    e ∈ get_rule.outgoing v d := by
  rw [get_rule.outgoing.eq_1]
  exact mem_outgoing_loop_of_mem_edges_start_eq v d.EDGES hedge hstart

lemma outgoing_mem_incoming_end (v : Vertex) (d : Graph) {e : Deduction}
    (he : e ∈ get_rule.outgoing v d) :
    e ∈ get_rule.incoming e.END d := by
  exact mem_incoming_of_mem_edges_end_eq e.END d
    (mem_outgoing_mem_edges v d he) rfl

lemma incoming_mem_outgoing_start (v : Vertex) (d : Graph) {e : Deduction}
    (he : e ∈ get_rule.incoming v d) :
    e ∈ get_rule.outgoing e.START d := by
  exact mem_outgoing_of_mem_edges_start_eq e.START d
    (mem_incoming_mem_edges v d he) rfl

lemma classifyRule?_intro_formula_implication {d : Graph} {v : Vertex} {p : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p)) :
    ∃ A B, v.FORMULA = Formula.implication A B := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil => simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil =>
            simp [hinc] at hclass
            obtain ⟨hcons, _⟩ := hclass
            cases hform : v.FORMULA with
            | atom name =>
                rw [hform] at hcons
                simp [consequent?] at hcons
            | implication A B => exact ⟨A, B, rfl⟩
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                split at hclass <;> simp at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass

lemma classifyRule?_elim_major_mem_incoming {d : Graph} {v : Vertex}
    {major minor : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor)) :
    major ∈ get_rule.incoming v d ∧
      ∃ A, major.START.FORMULA = Formula.implication A v.FORMULA := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil => simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil => simp [hinc] at hclass
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                by_cases h12 : e.START.FORMULA = Formula.implication e2.START.FORMULA v.FORMULA
                · simp [h12] at hclass
                  obtain ⟨hmajor, _⟩ := hclass
                  have hmem : major ∈ [e, e2] := by
                    simp [← hmajor]
                  have hshape : major.START.FORMULA =
                      Formula.implication e2.START.FORMULA v.FORMULA := by
                    simpa [← hmajor] using h12
                  exact ⟨hmem, ⟨e2.START.FORMULA, hshape⟩⟩
                · simp [h12] at hclass
                  by_cases h21 : e2.START.FORMULA = Formula.implication e.START.FORMULA v.FORMULA
                  · simp [h21] at hclass
                    obtain ⟨hmajor, _⟩ := hclass
                    have hmem : major ∈ [e, e2] := by
                      simp [← hmajor]
                    have hshape : major.START.FORMULA =
                        Formula.implication e.START.FORMULA v.FORMULA := by
                      simpa [← hmajor] using h21
                    exact ⟨hmem, ⟨e.START.FORMULA, hshape⟩⟩
                  · simp [h21] at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass

lemma classifyRule?_elim_major_formula_eq_minor {d : Graph} {v : Vertex}
    {major minor : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor)) :
    major.START.FORMULA = Formula.implication minor.START.FORMULA v.FORMULA := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil => simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil => simp [hinc] at hclass
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                by_cases h12 : e.START.FORMULA = Formula.implication e2.START.FORMULA v.FORMULA
                · simp [h12] at hclass
                  obtain ⟨hmajor, hminor⟩ := hclass
                  subst hmajor
                  subst hminor
                  exact h12
                · simp [h12] at hclass
                  by_cases h21 : e2.START.FORMULA = Formula.implication e.START.FORMULA v.FORMULA
                  · simp [h21] at hclass
                    obtain ⟨hmajor, hminor⟩ := hclass
                    subst hmajor
                    subst hminor
                    exact h21
                  · simp [h21] at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass

lemma classifyRule?_intro_incoming_eq {d : Graph} {v : Vertex} {p e : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p))
    (he : e ∈ get_rule.incoming v d) :
    e = p := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil =>
        simp [hinc] at he
    | cons e1 es =>
        cases es with
        | nil =>
            simp [hinc] at hclass
            simp [hinc] at he
            obtain ⟨_, hp⟩ := hclass
            exact he.trans hp
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                split at hclass <;> simp at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass

lemma classifyRule?_intro_mem_incoming {d : Graph} {v : Vertex} {p : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p)) :
    p ∈ get_rule.incoming v d := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil =>
        simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil =>
            simp [hinc] at hclass
            obtain ⟨_, hp⟩ := hclass
            simp [← hp]
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                split at hclass <;> simp at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass

lemma classifyRule?_elim_incoming_eq_major_or_minor {d : Graph} {v : Vertex}
    {major minor e : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor))
    (he : e ∈ get_rule.incoming v d) :
    e = major ∨ e = minor := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil =>
        simp [hinc] at he
    | cons e1 es =>
        cases es with
        | nil =>
            simp [hinc] at hclass
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                simp [hinc] at he
                by_cases h12 :
                    e1.START.FORMULA =
                      Formula.implication e2.START.FORMULA v.FORMULA
                · simp [h12] at hclass
                  obtain ⟨hmajor, hminor⟩ := hclass
                  subst hmajor
                  subst hminor
                  cases he with
                  | inl he1 =>
                      left
                      exact he1
                  | inr he2 =>
                      right
                      exact he2
                · simp [h12] at hclass
                  by_cases h21 :
                      e2.START.FORMULA =
                        Formula.implication e1.START.FORMULA v.FORMULA
                  · simp [h21] at hclass
                    obtain ⟨hmajor, hminor⟩ := hclass
                    subst hmajor
                    subst hminor
                    cases he with
                    | inl he1 =>
                        right
                        exact he1
                    | inr he2 =>
                        left
                        exact he2
                  · simp [h21] at hclass
            | cons e3 es3 =>
                simp [hinc] at hclass



/-- Which premise slot of `w`'s actual rule the premise formula `φ` fills:
    minor of an elimination -> slot 1, everything else (major / intro premise) -> 0. -/
def slotForEdge (φ : Formula) (w : Vertex) (d : Graph) : Nat :=
  match classifyRule? w d with
  | some (DLDSRuleClass.elim _ minor) => if φ = minor.START.FORMULA then 1 else 0
  | _ => 0

/-- Input label for the edge from a source formula `φ` into destination node `w`.
    The path carries this label only; the destination node decodes it back to a
    `(rule, slot)` incoming wire. -/
def inputLabelForEdge (d : Graph) (formulas : List Formula) (φ : Formula) (w : Vertex) : Nat :=
  let incoming := buildIncomingMapForFormula formulas w.FORMULA
  match ruleIndexForNode? d formulas w with
  | some ruleIdx => inputLabelForRuleSlot incoming ruleIdx (slotForEdge φ w d)
  | none => 0

/-- Descending carrier route (one inner `paths` list) for a live carrier whose
    current formula is `φ`.

    At an elimination node, the minor-premise carrier delivers its token and stops.
    The major-premise carrier is the representative carrier for the merged output;
    after `evaluate_node` computes the output vector at the elimination column,
    `propagate_tokens` overwrites that carrier's dependency vector with the merged
    result and routes it onward. -/
def routeFrom (d : Graph) (formulas : List Formula) : Nat → Formula → List (Nat × Nat)
  | 0, _ => []
  | Nat.succ fuel, φ =>
      match d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none => (0, 0) :: routeFrom d formulas fuel φ
      | some v =>
          match get_rule.outgoing v d with
          | [] => (0, 0) :: routeFrom d formulas fuel φ
          | (e :: _) =>
              (formulas.idxOf e.END.FORMULA + 1, inputLabelForEdge d formulas φ e.END) ::
              (if (match classifyRule? e.END d with
                    | some (DLDSRuleClass.elim _ minor) =>
                        decide (φ = minor.START.FORMULA)
                    | _ => false)
               then List.replicate fuel (0, 0)
               else routeFrom d formulas fuel e.END.FORMULA)

/-- Route fact for the minor-stop invariant: if the next edge lands at an `⊃E`
    node and the current carrier is the minor premise, the head step delivers the
    token to that `⊃E` and every remaining step is a stop. -/
lemma routeFrom_minor_tail_replicate
    (d : Graph) (formulas : List Formula) (fuel : Nat)
    (φ : Formula) (v : Vertex) (e : Deduction) (es : List Deduction)
    (major minor : Deduction)
    (hfind : d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some v)
    (hout : get_rule.outgoing v d = e :: es)
    (hclass : classifyRule? e.END d = some (DLDSRuleClass.elim major minor))
    (hminor : φ = minor.START.FORMULA) :
    routeFrom d formulas (Nat.succ fuel) φ =
      (formulas.idxOf e.END.FORMULA + 1, inputLabelForEdge d formulas φ e.END) ::
        List.replicate fuel (0, 0) := by
  subst φ
  simp [routeFrom, hfind, hout, hclass]

/-- Route fact for the carrier invariant: if the next edge is not the minor
    premise of its destination elimination, the route tail continues from the
    destination formula.  The major premise of an `⊃E` is the important case. -/
lemma routeFrom_nonminor_tail_continues
    (d : Graph) (formulas : List Formula) (fuel : Nat)
    (φ : Formula) (v : Vertex) (e : Deduction) (es : List Deduction)
    (hfind : d.NODES.find? (fun u => decide (u.FORMULA = φ)) = some v)
    (hout : get_rule.outgoing v d = e :: es)
    (hnotminor :
      (match classifyRule? e.END d with
        | some (DLDSRuleClass.elim _ minor) => decide (φ = minor.START.FORMULA)
        | _ => false) = false) :
    routeFrom d formulas (Nat.succ fuel) φ =
      (formulas.idxOf e.END.FORMULA + 1, inputLabelForEdge d formulas φ e.END) ::
        routeFrom d formulas fuel e.END.FORMULA := by
  simp [routeFrom, hfind, hout, hnotminor]

/-- Goal column = index in `buildFormulas d` of the root's FORMULA, where the root
    is the unique node with no outgoing deduction edge. -/
def goalColumn (d : Graph) : Nat :=
  match d.NODES.find? (fun v => (get_rule.outgoing v d).isEmpty) with
  | some r => (buildFormulas d).idxOf r.FORMULA
  | none => 0

/-- Singleton/simple-tree residual path `p` of Definition 3, extracted from the
    actual deduction edges. The grid is formula/level-shaped, but this input path
    is where the real DLDS edge structure is read; no colour fan-out/λ/collapse.

    The universal grid still has one initial token per formula column.  For the
    DLDS-derived path, only hypothesis columns are live carrier starts; derived
    formula columns stop immediately until an upstream carrier reaches them and
    receives the node's computed output vector from the evaluator. -/
def pathsFromDLDS (d : Graph) : PathInput :=
  let formulas := buildFormulas d
  let numSteps := (buildGridFromDLDS d).length - 1
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  formulas.map (fun φ =>
    match d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
    | some v =>
        if v.HYPOTHESIS then
          let delay := maxLvl - v.LEVEL
          List.replicate delay (formulas.idxOf φ + 1, 0) ++
            routeFrom d formulas (numSteps - delay) φ
        else List.replicate numSteps (0, 0)
    | none => List.replicate numSteps (0, 0))

lemma pathsFromDLDS_length (d : Graph) :
    (pathsFromDLDS d).length = (buildFormulas d).length := by
  simp [pathsFromDLDS]

lemma pathsFromDLDS_get?_of_formula (d : Graph) {origin : Nat} {φ : Formula}
    (hφ : (buildFormulas d)[origin]? = some φ) :
    (pathsFromDLDS d)[origin]? =
      some
        (match d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
        | some v =>
            if v.HYPOTHESIS then
              let delay := (d.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL
              List.replicate delay ((buildFormulas d).idxOf φ + 1, 0) ++
                routeFrom d (buildFormulas d)
                  ((buildGridFromDLDS d).length - 1 - delay) φ
            else
              List.replicate ((buildGridFromDLDS d).length - 1) (0, 0)
        | none =>
            List.replicate ((buildGridFromDLDS d).length - 1) (0, 0)) := by
  simp [pathsFromDLDS, List.getElem?_map, hφ]



/-- Operational DLDS evaluator soundness. `evaluateDLDS` accepts only when the
    chosen path is structurally invalid, or has no routing error and the goal
    dependency vector is fully discharged. This is not a characterization of
    full compressed DLDS validity. -/
theorem dlds_evaluation_correct
    (d : Graph)
    (paths : PathInput)
    (goal_column : Nat)
    (h_accept : evaluateDLDS d paths goal_column = true) :
    let grid := buildGridFromDLDS d
    let initial_vecs := initialVectorsFromDLDS d
    PathStructurallyInvalid paths grid initial_vecs
    ∨
    (PathHasNoRoutingError paths grid initial_vecs ∧
     AllAssumptionsDischarged paths grid initial_vecs goal_column) := by
  let grid := buildGridFromDLDS d
  let formulas := buildFormulas d
  let initial_vecs := initialVectorsFromDLDS d
  have h_wf := buildGridFromDLDS_wellformed d
  unfold evaluateDLDS at h_accept
  exact circuit_correctness grid initial_vecs paths goal_column h_accept



/-- Operational DLDS evaluator completeness for a selected path with no routing
    error and a fully discharged goal vector. -/
theorem dlds_evaluation_complete
    (d : Graph)
    (paths : PathInput)
    (goal_column : Nat)
    (h_valid : PathHasNoRoutingError paths (buildGridFromDLDS d) (initialVectorsFromDLDS d))
    (h_discharged : AllAssumptionsDischarged paths (buildGridFromDLDS d) (initialVectorsFromDLDS d) goal_column) :
    evaluateDLDS d paths goal_column = true := by
  unfold evaluateDLDS
  exact circuit_completeness (buildGridFromDLDS d) (initialVectorsFromDLDS d) paths goal_column h_valid h_discharged

/-- Operational DLDS evaluator iff: `evaluateDLDS` accepts exactly when the
    chosen path is structurally invalid, or has no routing error and the goal
    dependency vector is fully discharged. -/
theorem dlds_evaluation_iff
    (d : Graph)
    (paths : PathInput)
    (goal_column : Nat) :
    evaluateDLDS d paths goal_column = true
    ↔
    let grid := buildGridFromDLDS d
    let initial_vecs := initialVectorsFromDLDS d
    PathStructurallyInvalid paths grid initial_vecs
    ∨
    (PathHasNoRoutingError paths grid initial_vecs ∧
     AllAssumptionsDischarged paths grid initial_vecs goal_column) := by
  unfold evaluateDLDS
  exact circuit_iff (buildGridFromDLDS d) (initialVectorsFromDLDS d) paths goal_column




/-- `GenuinelyAccepts` rules out the neutral acceptance of structurally invalid
paths. It is the predicate used in the simple-tree bridge, and does not by
itself express full compressed DLDS validity. -/
def GenuinelyAccepts (d : Graph) (paths : PathInput) (g : Nat) : Prop :=
  let grid := buildGridFromDLDS d
  let iv := initialVectorsFromDLDS d
  PathHasNoRoutingError paths grid iv ∧ AllAssumptionsDischarged paths grid iv g

-- (The two genuine-acceptance witnesses are proved after `exClosedAA`/`exClosedABA`
-- are defined, in the non-vacuity section below.)



/-- Global acceptance: the circuit accepts a DLDS iff it evaluates to true
    on ALL path assignments. -/
def DLDSGloballyAccepted (d : Graph) (goal_column : Nat) : Prop :=
  ∀ paths : PathInput, evaluateDLDS d paths goal_column = true

/-- Global operational soundness: if every path assignment is accepted, then
    every path is structurally invalid, or has no routing error and a fully
    discharged goal vector. -/
theorem dlds_global_soundness
    (d : Graph)
    (goal_column : Nat)
    (h_global : DLDSGloballyAccepted d goal_column) :
    ∀ paths : PathInput,
      let grid := buildGridFromDLDS d
      let initial_vecs := initialVectorsFromDLDS d
      PathStructurallyInvalid paths grid initial_vecs
      ∨
      (PathHasNoRoutingError paths grid initial_vecs ∧
       AllAssumptionsDischarged paths grid initial_vecs goal_column) :=
  fun paths => dlds_evaluation_correct d paths goal_column (h_global paths)


/-- Global operational completeness for the evaluator predicate. -/
theorem dlds_global_completeness
    (d : Graph)
    (goal_column : Nat)
    (h_all_valid : ∀ paths : PathInput,
      let grid := buildGridFromDLDS d
      let initial_vecs := initialVectorsFromDLDS d
      PathStructurallyInvalid paths grid initial_vecs
      ∨
      (PathHasNoRoutingError paths grid initial_vecs ∧
       AllAssumptionsDischarged paths grid initial_vecs goal_column)) :
    DLDSGloballyAccepted d goal_column :=
  fun paths => (dlds_evaluation_iff d paths goal_column).mpr (h_all_valid paths)

/-- Global operational iff for `DLDSGloballyAccepted`. -/
theorem dlds_global_iff
    (d : Graph)
    (goal_column : Nat) :
    DLDSGloballyAccepted d goal_column
    ↔
    (∀ paths : PathInput,
      let grid := buildGridFromDLDS d
      let initial_vecs := initialVectorsFromDLDS d
      PathStructurallyInvalid paths grid initial_vecs
      ∨
      (PathHasNoRoutingError paths grid initial_vecs ∧
       AllAssumptionsDischarged paths grid initial_vecs goal_column)) :=
  ⟨dlds_global_soundness d goal_column, dlds_global_completeness d goal_column⟩


/-!
## Section: Valid-DLDS Predicate (Definition 23)

This section defines `ValidDLDS`, the validity predicate of Definition 23
(Haeusler/Barros/Brasil-Filho). The local CorrectRuleApp condition is our own
direct transcription (`LocalRuleCorrect`, `RootDischarged`) reading the
natural-deduction rule off each node's incoming/outgoing deduction edges; it does
NOT call Robinson's `type0/1/2/3` neighborhood predicates. The remaining
conjuncts are the global structural conditions (Leveled-Colored, Simplicity,
Ancestor-Simplicity) plus basic graph hygiene.

`ValidDLDS` is purely a property of the DLDS graph; it does not mention the
Boolean circuit at all.
-/

/-- One colour-`c` step: extend `reached` with the END of every colour-`c`
    deduction edge whose START is already reached. -/
def colour_one_step (EDGES : List Deduction) (c : Nat) (reached : List Vertex) : List Vertex :=
  EDGES.foldr
    (fun e acc => if (e.COLOUR = c) ∧ (e.START ∈ reached) then e.END :: acc else acc)
    reached

/-- Iterate `colour_one_step` `fuel` times to approximate the colour-`c`
    reachability closure of `reached`. -/
def colour_closure : Nat → List Deduction → Nat → List Vertex → List Vertex
  | 0,        _,     _, reached => reached
  | fuel + 1, EDGES, c, reached => colour_closure fuel EDGES c (colour_one_step EDGES c reached)

/-- Vertices reachable from `v` by one-or-more colour-`c` deduction edges.
    `EDGES.length` steps suffice to expose any cycle through `v`. -/
def colour_descendants (EDGES : List Deduction) (c : Nat) (v : Vertex) : List Vertex :=
  let seeds := EDGES.filterMap
    (fun e => if (e.COLOUR = c) ∧ (e.START = v) then some e.END else none)
  colour_closure EDGES.length EDGES c seeds

/-- **Def-23 Color-Acyclicity**: for each colour, the colour-`i` deduction
    subgraph is acyclic. A cycle in colour `i` would make some edge's START
    reachable from itself through colour-`i` edges, so we forbid exactly that. -/
def ColorAcyclicity (d : Graph) : Prop :=
  ∀ e ∈ d.EDGES, e.START ∉ colour_descendants d.EDGES e.COLOUR e.START

def colorAcyclicityB (d : Graph) : Bool :=
  d.EDGES.all fun e =>
    decide (e.START ∉ colour_descendants d.EDGES e.COLOUR e.START)

/-- **Def-23 Leveled-Colored**: every deduction edge drops exactly one level,
    i.e. `START.LEVEL = END.LEVEL + 1`. -/
def LeveledColored (d : Graph) : Prop :=
  ∀ e ∈ d.EDGES, e.START.LEVEL = e.END.LEVEL + 1

/-- **Def-23 Simplicity**: at most one deduction edge per `(START, END, COLOUR)`. -/
def Simplicity (d : Graph) : Prop :=
  ∀ e₁ ∈ d.EDGES, ∀ e₂ ∈ d.EDGES,
    (e₁.START = e₂.START ∧ e₁.END = e₂.END ∧ e₁.COLOUR = e₂.COLOUR) → e₁ = e₂

/-- **Def-23 Ancestor-Simplicity**: the ancestral `PATHS` list has no two
    distinct entries sharing the same `(START, END, COLOURS)` triple.

    NOTE: keying on `(START, END, COLOURS)` (rather than `(START, END)`) is what
    a real compressed DLDS satisfies — compression produces several coloured
    ancestral paths between the same endpoints. This may be *weaker* than the
    canonical Ancestor-Simplicity stated over the ancestral-edge relation `E_A`;
    revisit once the bridge fixes the intended strength. -/
def AncestorSimplicity (d : Graph) : Prop :=
  ∀ p₁ ∈ d.PATHS, ∀ p₂ ∈ d.PATHS,
    (p₁.START = p₂.START ∧ p₁.END = p₂.END ∧ p₁.COLOURS = p₂.COLOURS) → p₁ = p₂

/-- **Def-23 Ancestor-Edges**: ancestral edges go upward in level. -/
def AncestorEdges (d : Graph) : Prop :=
  ∀ p ∈ d.PATHS, p.START.LEVEL < p.END.LEVEL

def ancestorEdgesB (d : Graph) : Bool :=
  d.PATHS.all fun p => decide (p.START.LEVEL < p.END.LEVEL)

/-- Drop the optional leading `0` used by Robinson's colour-path convention
    before checking the positive colour payload. -/
def ancestorBackwayPayload (colours : List Nat) : List Nat :=
  match colours with
  | [] => []
  | c :: rest => if c = 0 then rest else c :: rest

/-- Def-19 / Algorithm 2 relative address lookup.

    Starting from `origin`, consume `γ` left-to-right. At each step, if the
    current node has a unique outgoing deduction edge, follow it. Otherwise,
    follow the unique outgoing edge whose colour is the current head of `γ`.
    If neither choice is unique, the address is invalid.

    The fuel is `d.NODES.length + 1`: a valid acyclic address cannot pass
    through more graph vertices than this, while the recursion also strictly
    consumes `γ` at each successful step. -/
def relativeAddress (d : Graph) (origin : Vertex) (γ : List Nat) : Option Vertex :=
  loop (d.NODES.length + 1) origin γ
where
  loop : Nat → Vertex → List Nat → Option Vertex
  | _, b, [] => some b
  | 0, _, _ :: _ => none
  | fuel + 1, b, colour :: rest =>
      match get_rule.outgoing b d with
      | [g] => loop fuel g.END rest
      | outs =>
          match outs.filter (fun e => e.COLOUR = colour) with
          | [g] => loop fuel g.END rest
          | _ => none

def relativeAddressB (d : Graph) (origin : Vertex) (γ : List Nat)
    (target : Vertex) : Bool :=
  relativeAddress d origin γ == some target

/-- **Def-23 Ancestor-Backway-Information**: each ancestor edge label is the
    genuine relative address from the edge's endpoint back to its startpoint.

    This is Def-19 / Algorithm 2 from the source: for each ancestral edge
    `p : START → END`, running the relative-address lookup from `p.END` with
    `p.COLOURS` must return `p.START`. On tree fixtures `PATHS = []`, so the
    condition is vacuous. -/
def AncestorBackwayInformation (d : Graph) : Prop :=
  ∀ p ∈ d.PATHS, relativeAddress d p.END p.COLOURS = some p.START

def checkNumbersB (xs : List Nat) : Bool :=
  !xs.isEmpty && xs.all fun n => decide (n > 0)

def ancestorBackwayInformationB (d : Graph) : Bool :=
  d.PATHS.all fun p => relativeAddressB d p.END p.COLOURS p.START

/-- **Def-23 Non-Nested-Ancestor-Edges**: no two ancestral intervals are
    properly nested by their endpoint levels. -/
def NonNestedAncestorEdges (d : Graph) : Prop :=
  ∀ p₁ ∈ d.PATHS, ∀ p₂ ∈ d.PATHS,
    ¬ (p₁.START.LEVEL < p₂.START.LEVEL ∧ p₂.END.LEVEL < p₁.END.LEVEL) ∧
    ¬ (p₂.START.LEVEL < p₁.START.LEVEL ∧ p₁.END.LEVEL < p₂.END.LEVEL)

def nonNestedAncestorEdgesB (d : Graph) : Bool :=
  d.PATHS.all fun p₁ =>
    d.PATHS.all fun p₂ =>
      !(decide (p₁.START.LEVEL < p₂.START.LEVEL ∧ p₂.END.LEVEL < p₁.END.LEVEL)) &&
      !(decide (p₂.START.LEVEL < p₁.START.LEVEL ∧ p₁.END.LEVEL < p₂.END.LEVEL))



/-- The dependency a correct natural-deduction rule at `v` emits, computed from
    `inc := get_rule.incoming v d` (reading `DEPENDENCY` regardless of `COLOUR`):

    * `v.HYPOTHESIS = true`     ⟶ `[v.FORMULA]` (top formula / assumption);
    * `inc = [p]` (⊃I)          ⟶ `p.DEPENDENCY − [α]` where `v.FORMULA = α >> β`
                                  (the antecedent `α` is discharged);
    * `inc = [e₁, e₂]` (⊃E)     ⟶ `minor.DEPENDENCY ∪ major.DEPENDENCY`, where the
                                  major premise is the edge whose `START.FORMULA`
                                  is `ψ >> v.FORMULA` and the minor the one whose
                                  `START.FORMULA` is `ψ` (matched by FORMULA, not
                                  position);
    * otherwise                 ⟶ `[]` (ill-formed; gated false by `RuleShapeOK`).

    Set-ops `∪`, `−`, `#` are Robinson's `eraseDups`-based List operations. -/
def outDep (v : Vertex) (d : Graph) : List Formula :=
  match classifyRule? v d with
  | some DLDSRuleClass.hypothesis => [v.FORMULA]
  | some (DLDSRuleClass.intro p) =>
      match antecedent? v.FORMULA with
      | some α => p.DEPENDENCY − [α]
      | none   => p.DEPENDENCY
  | some (DLDSRuleClass.elim major minor) =>
      List.eraseDups (minor.DEPENDENCY ++ major.DEPENDENCY)
  | none => []

/-- Boolean shape test mirroring the `outDep` cases: hypothesis; one-premise ⊃I
    whose premise proves the consequent of `v.FORMULA`; two-premise ⊃E whose two
    premises are `ψ >> v.FORMULA` and `ψ` (matched by formula). -/
def ruleShapeOKB (v : Vertex) (d : Graph) : Bool :=
  (classifyRule? v d).isSome

/-- The natural-deduction rule shape at `v` is well-formed. -/
abbrev RuleShapeOK (v : Vertex) (d : Graph) : Prop := ruleShapeOKB v d = true

lemma ruleIndexForNode?_isSome_of_ruleShapeOK
    (d : Graph) (w : Vertex)
    (hcheck : check_dlds d)
    (hshape : RuleShapeOK w d) :
    ∃ ruleIdx, ruleIndexForNode? d (buildFormulas d) w = some ruleIdx := by
  unfold RuleShapeOK ruleShapeOKB at hshape
  cases hclass : classifyRule? w d with
  | none => simp [hclass] at hshape
  | some cls =>
      cases cls with
      | hypothesis =>
          unfold ruleIndexForNode?
          rw [hclass]
          have hlen : 0 < (buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length := by
            unfold buildIncomingMapForFormula
            cases w.FORMULA <;> simp
          exact ⟨(buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length - 1,
            by simp [hlen]⟩
      | intro p =>
          obtain ⟨A, B, hform⟩ := classifyRule?_intro_formula_implication hclass
          unfold ruleIndexForNode?
          rw [hclass, hform]
          exact ⟨0, rfl⟩
      | elim major minor =>
          obtain ⟨hmajor_in, hmajor_shape⟩ := classifyRule?_elim_major_mem_incoming hclass
          have hmajor_edge : major ∈ d.EDGES := mem_incoming_mem_edges w d hmajor_in
          have hmajor_node : major.START ∈ d.NODES := (hcheck.2.1 hmajor_edge).1
          have hmajor_formula_mem : major.START.FORMULA ∈ buildFormulas d := by
            unfold buildFormulas
            exact List.mem_eraseDups.mpr
              (List.mem_map.mpr ⟨major.START, hmajor_node, rfl⟩)
          obtain ⟨pos, hpos⟩ :=
            elimRulePosition?_isSome_of_mem (buildFormulas d) w.FORMULA
              major.START.FORMULA hmajor_formula_mem hmajor_shape
          refine ⟨introRuleCount w.FORMULA + pos, ?_⟩
          simp [ruleIndexForNode?, hclass, hpos]

lemma slotForEdge_in_range (d : Graph) (formulas : List Formula)
    (φ : Formula) (w : Vertex) (ruleIdx : Nat)
    (hsel : ruleIndexForNode? d formulas w = some ruleIdx) :
    slotForEdge φ w d <
      ((buildIncomingMapForFormula formulas w.FORMULA)[ruleIdx]?.getD default).length := by
  unfold ruleIndexForNode? at hsel
  cases hclass : classifyRule? w d with
  | none =>
      simp [hclass] at hsel
  | some cls =>
      cases cls with
      | hypothesis =>
          simp [hclass] at hsel
          have hlen : 0 < (buildIncomingMapForFormula formulas w.FORMULA).length := by
            unfold buildIncomingMapForFormula
            cases w.FORMULA <;> simp
          simp [hlen] at hsel
          subst ruleIdx
          unfold slotForEdge
          rw [hclass]
          unfold buildIncomingMapForFormula
          cases w.FORMULA <;> simp
      | intro p =>
          simp [hclass] at hsel
          cases hform : w.FORMULA with
          | atom name =>
              simp [hform] at hsel
          | implication A B =>
              simp [hform] at hsel
              subst ruleIdx
              simp [slotForEdge, hclass, buildIncomingMapForFormula]
      | elim major minor =>
          simp [hclass] at hsel
          cases hpos : elimRulePosition? formulas w.FORMULA major.START.FORMULA with
          | none =>
              simp [hpos] at hsel
          | some pos =>
              simp [hpos] at hsel
              subst ruleIdx
              have harity := elimRulePosition?_indexes_elim_entry
                formulas w.FORMULA major.START.FORMULA pos hpos
              unfold slotForEdge
              rw [hclass]
              by_cases hφ : φ = minor.START.FORMULA
              · simp [hφ, harity]
              · simp [hφ, harity]

/-- Source columns of the DLDS premises of `w`'s classified rule, in slot order.
    Hypotheses use the repetition/self wire. -/
def classifiedRuleSourceColumns? (d : Graph) (formulas : List Formula)
    (w : Vertex) : Option (List Nat) :=
  match classifyRule? w d with
  | none => none
  | some DLDSRuleClass.hypothesis =>
      some [formulas.idxOf w.FORMULA]
  | some (DLDSRuleClass.intro p) =>
      some [formulas.idxOf p.START.FORMULA]
  | some (DLDSRuleClass.elim major minor) =>
      some [formulas.idxOf major.START.FORMULA, formulas.idxOf minor.START.FORMULA]

def incomingRuleSourceColumns? (formulas : List Formula) (formula : Formula)
    (ruleIdx : Nat) : Option (List Nat) :=
  ((buildIncomingMapForFormula formulas formula)[ruleIdx]?).map
    (fun inc => inc.map Prod.fst)

/-- Graph/grid bridge: the selected incoming-map entry for `w`'s rule has exactly
    the DLDS premise source columns in slot order. -/
lemma dlds_incoming_matches_rule_premises
    (d : Graph) (w : Vertex) (ruleIdx : Nat)
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some ruleIdx) :
    incomingRuleSourceColumns? (buildFormulas d) w.FORMULA ruleIdx =
      classifiedRuleSourceColumns? d (buildFormulas d) w := by
  unfold incomingRuleSourceColumns? classifiedRuleSourceColumns?
  unfold ruleIndexForNode? at hsel
  cases hclass : classifyRule? w d with
  | none =>
      simp [hclass] at hsel
  | some cls =>
      cases cls with
      | hypothesis =>
          simp [hclass] at hsel
          have hlen : 0 < (buildIncomingMapForFormula (buildFormulas d) w.FORMULA).length := by
            unfold buildIncomingMapForFormula
            cases w.FORMULA <;> simp
          simp [hlen] at hsel
          subst ruleIdx
          unfold buildIncomingMapForFormula
          cases w.FORMULA <;> simp
      | intro p =>
          simp [hclass] at hsel
          cases hform : w.FORMULA with
          | atom name =>
              simp [hform] at hsel
          | implication A B =>
              simp [hform] at hsel
              subst ruleIdx
              have hp : p.START.FORMULA = B := by
                unfold classifyRule? at hclass
                by_cases hhyp : w.HYPOTHESIS = true
                · simp [hhyp] at hclass
                · simp [hhyp] at hclass
                  cases hinc : get_rule.incoming w d with
                  | nil =>
                      simp [hinc] at hclass
                  | cons e es =>
                      cases es with
                      | nil =>
                          simp [hinc, hform] at hclass
                          rcases hclass with ⟨hcons, hp_eq⟩
                          subst hp_eq
                          simpa [consequent?] using hcons.symm
                      | cons e2 es2 =>
                          cases es2 with
                          | nil =>
                              simp [hinc, hform] at hclass
                              split at hclass <;> simp at hclass
                          | cons e3 es3 =>
                              simp [hinc] at hclass
              simp [buildIncomingMapForFormula, hp]
      | elim major minor =>
          simp [hclass] at hsel
          cases hpos : elimRulePosition? (buildFormulas d) w.FORMULA major.START.FORMULA with
          | none =>
              simp [hpos] at hsel
          | some pos =>
              simp [hpos] at hsel
              subst ruleIdx
              have hmajor :
                  major.START.FORMULA =
                    Formula.implication minor.START.FORMULA w.FORMULA :=
                classifyRule?_elim_major_formula_eq_minor hclass
              have hpos' :
                  elimRulePosition? (buildFormulas d) w.FORMULA
                    (Formula.implication minor.START.FORMULA w.FORMULA) = some pos := by
                simpa [hmajor] using hpos
              have hsrc := elimRulePosition?_entry_sources_imp
                d w.FORMULA minor.START.FORMULA pos hpos'
              simpa [incomingRuleSourceColumns?, classifiedRuleSourceColumns?, hclass, hmajor]
                using hsrc

lemma inputLabelForEdge_decodes_of_ruleIndex
    (d : Graph) (formulas : List Formula) (φ : Formula) (w : Vertex)
    (ruleIdx : Nat)
    (hsel : ruleIndexForNode? d formulas w = some ruleIdx) :
    ∃ slot src,
      decodeInputLabel
        (buildIncomingMapForFormula formulas w.FORMULA)
        (inputLabelForEdge d formulas φ w) = some (ruleIdx, slot, src) ∧
      ruleIdx < (nodeForFormula formulas w.FORMULA).rules.length ∧
      slot <
        ((buildIncomingMapForFormula formulas w.FORMULA)[ruleIdx]?.getD default).length := by
  let incoming := buildIncomingMapForFormula formulas w.FORMULA
  have hslot := slotForEdge_in_range d formulas φ w ruleIdx hsel
  have hpos : 0 < (incoming[ruleIdx]?.getD default).length := by
    dsimp [incoming] at hslot ⊢
    omega
  have hidxIncoming : ruleIdx < incoming.length :=
    getElem?_getD_length_pos_lt incoming ruleIdx hpos
  have hidxRules : ruleIdx < (nodeForFormula formulas w.FORMULA).rules.length := by
    rw [← incoming_rules_aligned_length formulas w.FORMULA]
    exact hidxIncoming
  have hslotGet :
      slotForEdge φ w d < (incoming.get ⟨ruleIdx, hidxIncoming⟩).length := by
    dsimp [incoming] at hslot ⊢
    rw [List.getElem?_eq_getElem hidxIncoming, Option.getD_some] at hslot
    exact hslot
  unfold inputLabelForEdge
  rw [hsel]
  by_cases hlast : ruleIdx + 1 = incoming.length
  · have hs0 : 0 < (incoming.get ⟨ruleIdx, hidxIncoming⟩).length := by
      exact Nat.lt_of_le_of_lt (Nat.zero_le _) hslotGet
    refine ⟨0, (incoming.get ⟨ruleIdx, hidxIncoming⟩).get ⟨0, hs0⟩ |>.1, ?_, hidxRules, ?_⟩
    · have hlabel :
          inputLabelForRuleSlot incoming ruleIdx (slotForEdge φ w d) =
            inputLabelForRuleSlot incoming ruleIdx 0 := by
        simp [inputLabelForRuleSlot, hlast]
      change
        decodeInputLabel incoming
          (inputLabelForRuleSlot incoming ruleIdx (slotForEdge φ w d)) =
          some (ruleIdx, 0, ((incoming.get ⟨ruleIdx, hidxIncoming⟩).get ⟨0, hs0⟩).1)
      rw [hlabel]
      simpa [incoming] using
        inputLabelForRuleSlot_decode_roundtrip_rep incoming ruleIdx hlast hidxIncoming hs0
    · dsimp [incoming]
      rw [List.getElem?_eq_getElem hidxIncoming, Option.getD_some]
      exact hs0
  · have hnonrep : ruleIdx + 1 < incoming.length := by omega
    refine ⟨slotForEdge φ w d,
      (incoming.get ⟨ruleIdx, Nat.lt_of_succ_lt hnonrep⟩).get
        ⟨slotForEdge φ w d, ?_⟩ |>.1, ?_, hidxRules, ?_⟩
    · exact hslotGet
    · simpa [incoming] using
        inputLabelForRuleSlot_decode_roundtrip_nonrep incoming ruleIdx
          (slotForEdge φ w d) hnonrep hslotGet
    · dsimp [incoming]
      rw [List.getElem?_eq_getElem hidxIncoming, Option.getD_some]
      exact hslotGet

/-- **Def-23 CorrectRuleApp (local).** Every non-root node (`get_rule.outgoing`
    non-empty) applies a well-formed rule, and every outgoing edge carries the
    dependency that rule emits — `DEPENDENCY` read regardless of `COLOUR`. -/
def LocalRuleCorrect (d : Graph) : Prop :=
  ∀ v ∈ d.NODES,
    (get_rule.outgoing v d ≠ []) →
      RuleShapeOK v d ∧ (∀ e ∈ get_rule.outgoing v d, e.DEPENDENCY = outDep v d)

/-- **Def-23 closed-derivation.** Every root (`get_rule.outgoing` empty) has a
    well-formed rule shape and the rule computed over its incoming edges emits
    the empty dependency, i.e. all assumptions are discharged. -/
def RootDischarged (d : Graph) : Prop :=
  ∀ r ∈ d.NODES, (get_rule.outgoing r d = []) →
    RuleShapeOK r d ∧ outDep r d = []

/-- **Leaf/kernel partition.** Hypothesis nodes are leaves of the deduction
    graph: they may feed later rule applications through outgoing edges, but no
    deduction rule may feed into them. This closes the faithfulness gap where a
    node marked `HYPOTHESIS=true` was classified as a leaf while still receiving
    incoming carriers in the grid. -/
def HypothesesHaveNoIncoming (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, v.HYPOTHESIS = true → get_rule.incoming v d = []

/-- **Definition 4 — Valid DLDS.** Named transcription of the paper validity
    conditions. Graph hygiene, color/level/simplicity/rule/discharge are full;
    ancestor-path conditions are present and vacuous on the simple-tree scope
    (`PATHS = []`), while `hypNoIncoming` records the leaf/kernel partition. -/
structure ValidDLDS (d : Graph) : Prop where
  -- Def-23: basic graph hygiene (unique node numbering, edges/paths attached to nodes)
  hygiene : check_dlds d
  -- Def-23: Leveled-Colored
  leveledColored : LeveledColored d
  -- Def-23: Simplicity
  simplicity : Simplicity d
  -- Def-23: Ancestor-Simplicity
  ancestorSimplicity : AncestorSimplicity d
  -- Leaf/kernel partition: hypotheses are deduction leaves (no incoming edges)
  hypNoIncoming : HypothesesHaveNoIncoming d
  -- Def-23: CorrectRuleApp (local natural-deduction rule correctness)
  localRuleCorrect : LocalRuleCorrect d
  -- Def-23: closed derivation (root dependencies discharged)
  rootDischarged : RootDischarged d
  -- Def-23: Color-Acyclicity
  colorAcyclicity : ColorAcyclicity d
  -- Def-23: Ancestor-Edges
  ancestorEdges : AncestorEdges d
  -- Def-23: Ancestor-Backway-Information
  ancestorBackwayInformation : AncestorBackwayInformation d
  -- Def-23: Non-Nested-Ancestor-Edges
  nonNestedAncestorEdges : NonNestedAncestorEdges d



/-- Scope predicate: a *tree* DLDS — every node has at most one outgoing deduction
    edge, no node is collapsed, there are no ancestral paths, and all deduction
    edges carry the default colour `0`. Nonzero colours are introduced by
    collapse, so the uncollapsed tree fragment uses only colour `0`. The
    compressed case will relax these conditions. -/
def IsTreeDLDS (d : Graph) : Prop :=
  (∀ v ∈ d.NODES, (get_rule.outgoing v d).length ≤ 1)
  ∧ (∀ v ∈ d.NODES, v.COLLAPSED = false)
  ∧ d.PATHS = []
  ∧ (∀ e ∈ d.EDGES, e.COLOUR = 0)

/-- Injective formula labeling: distinct nodes carry distinct formulas. This is
    the precondition that makes `buildFormulas`' column-per-formula model faithful
    to a tree — without it the dedup merges multiple same-formula nodes (with
    different outgoing edges) into one column and `pathsFromDLDS` becomes lossy
    (confirmed counterexample last turn). -/
def InjFormulas (d : Graph) : Prop :=
  ∀ u ∈ d.NODES, ∀ v ∈ d.NODES, u.FORMULA = v.FORMULA → u = v

/-- A *simple* tree DLDS: a tree whose formula labeling is injective. This is the
    fragment on which the column-based routing of `pathsFromDLDS` is faithful, so
    the forward bridge can be proved here. (Excludes formula repetition, e.g.
    contraction — that needs the compressed colour/ancestral machinery later.) -/
def IsSimpleTreeDLDS (d : Graph) : Prop :=
  IsTreeDLDS d ∧ InjFormulas d

/-- `NoLayerError paths num_levels level tokens layers` mirrors `eval_from_level`'s
    own recursion: every layer evaluated along the descent reports no XOR conflict
    (`(evaluate_layer …).2 = false`), with tokens threaded by `propagate_tokens`
    exactly as the evaluator threads them. -/
def NoLayerError {n : Nat} (paths : PathInput) (num_levels : Nat) :
    Nat → List (Token n) → List (GridLayer n) → Prop
  | _, _, [] => True
  | level, tokens, (layer :: rest) =>
      (evaluate_layer layer tokens).2 = false ∧
      NoLayerError paths num_levels (level - 1)
        (propagate_tokens tokens paths level num_levels (evaluate_layer layer tokens).1) rest

/-- **Glue lemma** (general evaluator fact, tree-agnostic): if no layer along the
    descent conflicts, the accumulated error flag of `eval_from_level` stays
    `false`. Proved by induction on the layer list, threading `acc = false`. -/
lemma eval_from_level_snd_false_of_NoLayerError {n : Nat}
    (paths : PathInput) (num_levels : Nat) :
    ∀ (layers : List (GridLayer n)) (level : Nat) (tokens : List (Token n)) (acc : Bool),
      acc = false →
      NoLayerError paths num_levels level tokens layers →
      (eval_from_level paths level tokens layers acc num_levels).2 = false := by
  intro layers
  induction layers with
  | nil =>
      intro level tokens acc hacc _
      simp only [eval_from_level]
      exact hacc
  | cons layer rest ih =>
      intro level tokens acc hacc hnle
      obtain ⟨hhead, htail⟩ := hnle
      rw [eval_from_level]
      cases hel : evaluate_layer layer tokens with
      | mk outputs layer_error =>
        have hle0 : layer_error = false := by
          have h := hhead; rw [hel] at h; exact h
        cases rest with
        | nil => simp [hacc, hle0]
        | cons l2 r2 =>
            simp only []
            apply ih
            · simp [hacc, hle0]
            · have h := htail; rw [hel] at h; exact h

/-- Each node of a tree DLDS has at most one outgoing deduction edge. Clean
    projection of `IsTreeDLDS`'s first conjunct for downstream lemmas. -/
lemma tree_one_outgoing (d : Graph) (htree : IsTreeDLDS d) :
    ∀ v ∈ d.NODES, (get_rule.outgoing v d).length ≤ 1 :=
  htree.1

/-- **Column ↔ node correspondence.** Under `InjFormulas`, the map
    `node ↦ idxOf node.FORMULA in buildFormulas d` is injective on `d.NODES`:
    distinct nodes land in distinct columns. With membership giving column
    validity, this is the bijection that makes "the node at column `k`"
    well-defined — the property whose absence broke the routing last turn. -/
lemma column_node_bij (d : Graph) (hinj : InjFormulas d) :
    ∀ u ∈ d.NODES, ∀ v ∈ d.NODES,
      (buildFormulas d).idxOf u.FORMULA = (buildFormulas d).idxOf v.FORMULA → u = v := by
  intro u hu v hv hcol
  apply hinj u hu v hv
  have hmu : u.FORMULA ∈ buildFormulas d := by
    unfold buildFormulas; exact List.mem_eraseDups.mpr (List.mem_map.mpr ⟨u, hu, rfl⟩)
  have hmv : v.FORMULA ∈ buildFormulas d := by
    unfold buildFormulas; exact List.mem_eraseDups.mpr (List.mem_map.mpr ⟨v, hv, rfl⟩)
  have eu : (buildFormulas d)[(buildFormulas d).idxOf u.FORMULA]'(List.idxOf_lt_length_of_mem hmu)
      = u.FORMULA := List.getElem_idxOf (List.idxOf_lt_length_of_mem hmu)
  have ev : (buildFormulas d)[(buildFormulas d).idxOf v.FORMULA]'(List.idxOf_lt_length_of_mem hmv)
      = v.FORMULA := List.getElem_idxOf (List.idxOf_lt_length_of_mem hmv)
  rw [← eu, ← ev]; congr 1

lemma find_node_by_formula_eq_of_inj_list (nodes : List Vertex)
    (hinj : ∀ u ∈ nodes, ∀ v ∈ nodes, u.FORMULA = v.FORMULA → u = v)
    {v : Vertex} (hv : v ∈ nodes) :
    nodes.find? (fun u => decide (u.FORMULA = v.FORMULA)) = some v := by
  induction nodes with
  | nil =>
      simp at hv
  | cons u us ih =>
      by_cases huf : u.FORMULA = v.FORMULA
      · have huv : u = v := hinj u (@List.mem_cons_self Vertex u us) v hv huf
        simp [List.find?, huv]
      · have hv_us : v ∈ us := by
          cases hv with
          | head => contradiction
          | tail _ htail => exact htail
        have hinj_us : ∀ x ∈ us, ∀ y ∈ us, x.FORMULA = y.FORMULA → x = y := by
          intro x hx y hy hxy
          exact hinj x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy) hxy
        simp [List.find?, huf, ih hinj_us hv_us]

/-- Under injective formula labels, the formula-indexed path lookup used by
    `routeFrom` recovers the unique DLDS node carrying that formula. -/
lemma find_node_by_formula_eq_of_inj (d : Graph) (hinj : InjFormulas d)
    {v : Vertex} (hv : v ∈ d.NODES) :
    d.NODES.find? (fun u => decide (u.FORMULA = v.FORMULA)) = some v :=
  find_node_by_formula_eq_of_inj_list d.NODES hinj hv

/-- Token-threading auxiliary: every token produced by `initialize_tokens` has its
    `source_column`, `current_column`, and `origin_column` all equal (its own
    column). Base case for the token-dynamics invariant. -/
lemma initialize_source_eq {n : Nat} (vecs : List (List.Vector Bool n)) (tl : Nat) :
    ∀ t ∈ initialize_tokens vecs tl,
      t.source_column = t.current_column ∧ t.current_column = t.origin_column := by
  intro t ht
  unfold initialize_tokens at ht
  rw [List.mem_map] at ht
  obtain ⟨⟨vec, col⟩, _, ht'⟩ := ht
  subst ht'
  exact ⟨rfl, rfl⟩

/-- Token-threading auxiliary: every token produced by `propagate_tokens` carries
    `source_column = (originating token).current_column` and preserves
    `origin_column`. This is the step fact the token-dynamics induction needs:
    a token's recorded source is exactly the column it came from. -/
lemma propagate_source_eq {n : Nat} (tokens : List (Token n)) (paths : PathInput)
    (cl nl : Nat) (outs : List (List.Vector Bool n)) :
    ∀ t' ∈ propagate_tokens tokens paths cl nl outs,
      ∃ t ∈ tokens, t'.source_column = t.current_column ∧ t'.origin_column = t.origin_column := by
  intro t' ht'
  unfold propagate_tokens at ht'
  rw [List.mem_filterMap] at ht'
  obtain ⟨t, ht_mem, ht_eq⟩ := ht'
  refine ⟨t, ht_mem, ?_⟩
  dsimp only at ht_eq
  split at ht_eq
  · split at ht_eq
    · split at ht_eq
      · simp at ht_eq
      · split at ht_eq
        · rw [Option.some.injEq] at ht_eq; subst ht_eq; exact ⟨rfl, rfl⟩
        · simp at ht_eq
    · simp at ht_eq
  · simp at ht_eq

lemma ruleShapeOK_of_valid_node (d : Graph) (hvalid : ValidDLDS d)
    {w : Vertex} (hw : w ∈ d.NODES) :
    RuleShapeOK w d := by
  by_cases hout : get_rule.outgoing w d = []
  · exact (hvalid.rootDischarged w hw hout).1
  · exact (hvalid.localRuleCorrect w hw hout).1

/-- Every valid DLDS node has a selected grid-rule index for its actual
    natural-deduction rule. This factors the repeated `RuleShapeOK` plumbing
    out of the route-coherence proof. -/
lemma ruleIndexForNode?_isSome_of_valid_node
    (d : Graph) (hvalid : ValidDLDS d)
    {w : Vertex} (hw : w ∈ d.NODES) :
    ∃ ruleIdx, ruleIndexForNode? d (buildFormulas d) w = some ruleIdx := by
  exact ruleIndexForNode?_isSome_of_ruleShapeOK d w hvalid.hygiene
    (ruleShapeOK_of_valid_node d hvalid hw)

/-- The selected rule's source-column list exists for every valid node. -/
lemma classifiedRuleSourceColumns?_isSome_of_valid_node
    (d : Graph) (hvalid : ValidDLDS d)
    {w : Vertex} (hw : w ∈ d.NODES) :
    ∃ srcs, classifiedRuleSourceColumns? d (buildFormulas d) w = some srcs := by
  have hshape := ruleShapeOK_of_valid_node d hvalid hw
  unfold RuleShapeOK ruleShapeOKB at hshape
  cases hclass : classifyRule? w d with
  | none =>
      simp [hclass] at hshape
  | some cls =>
      cases cls <;> simp [classifiedRuleSourceColumns?, hclass]

/-- The selected incoming-map entry for every valid node is exactly its
    classified source columns. -/
lemma incomingRuleSourceColumns?_of_valid_node
    (d : Graph) (_hvalid : ValidDLDS d)
    {w : Vertex} (_hw : w ∈ d.NODES)
    {ruleIdx : Nat}
    (hsel : ruleIndexForNode? d (buildFormulas d) w = some ruleIdx) :
    incomingRuleSourceColumns? (buildFormulas d) w.FORMULA ruleIdx =
      classifiedRuleSourceColumns? d (buildFormulas d) w :=
  dlds_incoming_matches_rule_premises d w ruleIdx hsel

lemma incoming_start_level_of_valid (d : Graph) (hvalid : ValidDLDS d)
    {v : Vertex} {e : Deduction}
    (he : e ∈ get_rule.incoming v d) :
    e.START.LEVEL = v.LEVEL + 1 := by
  have hedge : e ∈ d.EDGES := mem_incoming_mem_edges v d he
  have hend : e.END = v := mem_incoming_end_eq v d he
  have hlev := hvalid.leveledColored e hedge
  simpa [hend] using hlev

lemma outgoing_end_level_of_valid (d : Graph) (hvalid : ValidDLDS d)
    {v : Vertex} {e : Deduction}
    (he : e ∈ get_rule.outgoing v d) :
    v.LEVEL = e.END.LEVEL + 1 := by
  have hedge : e ∈ d.EDGES := mem_outgoing_mem_edges v d he
  have hstart : e.START = v := mem_outgoing_start_eq v d he
  have hlev := hvalid.leveledColored e hedge
  simpa [hstart] using hlev

/-- The length of `routeFrom` always equals the fuel. -/
lemma routeFrom_length (d : Graph) (formulas : List Formula) :
    ∀ (fuel : Nat) (φ : Formula),
      (routeFrom d formulas fuel φ).length = fuel := by
  intro fuel
  induction fuel with
  | zero => intro φ; simp [routeFrom]
  | succ fuel ih =>
    intro φ
    cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
    | none => simp [routeFrom, hfind, ih]
    | some v =>
      cases hout : get_rule.outgoing v d with
      | nil => simp [routeFrom, hfind, hout, ih]
      | cons e es =>
        simp only [routeFrom, hfind, hout, List.length_cons]
        cases (match classifyRule? e.END d with
          | some (DLDSRuleClass.elim major minor) =>
              decide (φ = minor.START.FORMULA)
          | _ => false) with
        | false => simp [ih]
        | true  => simp

lemma routeFrom_label_coherent (d : Graph) (hvalid : ValidDLDS d) :
    ∀ (fuel : Nat) (φ : Formula) (step target label : Nat)
      (hStep : step < (routeFrom d (buildFormulas d) fuel φ).length),
      (routeFrom d (buildFormulas d) fuel φ).get ⟨step, hStep⟩ = (target, label) →
      target ≠ 0 →
      ∃ hTarget : target - 1 < (buildFormulas d).length,
        ∃ w ∈ d.NODES,
        w.FORMULA = (buildFormulas d).get ⟨target - 1, hTarget⟩ ∧
        ∃ ruleIdx slot src,
          ruleIndexForNode? d (buildFormulas d) w = some ruleIdx ∧
          decodeInputLabel
            (buildIncomingMapForFormula (buildFormulas d) w.FORMULA)
            label = some (ruleIdx, slot, src) ∧
          ruleIdx <
            (nodeForFormula (buildFormulas d) w.FORMULA).rules.length ∧
          slot <
            ((buildIncomingMapForFormula (buildFormulas d) w.FORMULA)[ruleIdx]?.getD default).length := by
  intro fuel
  induction fuel with
  | zero =>
      intro φ step target label hStep _ _
      simp [routeFrom] at hStep
  | succ fuel ih =>
      intro φ step target label hStep hget htarget
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          cases step with
          | zero =>
              have htarget0 : target = 0 := by
                have hfst := congrArg Prod.fst hget
                simpa [routeFrom, hfind] using hfst.symm
              exact False.elim (htarget htarget0)
          | succ step =>
              have hStepTail : step < (routeFrom d (buildFormulas d) fuel φ).length := by
                simpa [routeFrom, hfind] using hStep
              have hgetTail :
                  (routeFrom d (buildFormulas d) fuel φ).get ⟨step, hStepTail⟩ =
                    (target, label) := by
                simpa [routeFrom, hfind] using hget
              exact ih φ step target label hStepTail hgetTail htarget
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              cases step with
              | zero =>
                  have htarget0 : target = 0 := by
                    have hfst := congrArg Prod.fst hget
                    simpa [routeFrom, hfind, hout] using hfst.symm
                  exact False.elim (htarget htarget0)
              | succ step =>
                  have hStepTail : step < (routeFrom d (buildFormulas d) fuel φ).length := by
                    simpa [routeFrom, hfind, hout] using hStep
                  have hgetTail :
                      (routeFrom d (buildFormulas d) fuel φ).get ⟨step, hStepTail⟩ =
                        (target, label) := by
                    simpa [routeFrom, hfind, hout] using hget
                  exact ih φ step target label hStepTail hgetTail htarget
          | cons e es =>
              cases step with
              | zero =>
                  have htarget_eq :
                      target = (buildFormulas d).idxOf e.END.FORMULA + 1 := by
                    have hfst := congrArg Prod.fst hget
                    simpa [routeFrom, hfind, hout] using hfst.symm
                  have hlabel_eq :
                      label = inputLabelForEdge d (buildFormulas d) φ e.END := by
                    have hsnd := congrArg Prod.snd hget
                    simpa [routeFrom, hfind, hout] using hsnd.symm
                  subst target
                  subst label
                  have he_out : e ∈ get_rule.outgoing v d := by
                    rw [hout]
                    simp
                  have he_edge : e ∈ d.EDGES := mem_outgoing_mem_edges v d he_out
                  have hend_node : e.END ∈ d.NODES := (hvalid.hygiene.2.1 he_edge).2
                  have hend_formula_mem : e.END.FORMULA ∈ buildFormulas d := by
                    unfold buildFormulas
                    exact List.mem_eraseDups.mpr
                      (List.mem_map.mpr ⟨e.END, hend_node, rfl⟩)
                  have hidx_lt :
                      (buildFormulas d).idxOf e.END.FORMULA < (buildFormulas d).length :=
                    List.idxOf_lt_length_of_mem hend_formula_mem
                  have hTarget :
                      ((buildFormulas d).idxOf e.END.FORMULA + 1) - 1 <
                        (buildFormulas d).length := by
                    simpa using hidx_lt
                  have hformula :
                      e.END.FORMULA =
                        (buildFormulas d).get
                          ⟨((buildFormulas d).idxOf e.END.FORMULA + 1) - 1, hTarget⟩ := by
                    have hgetIdx :
                        (buildFormulas d).get
                          ⟨(buildFormulas d).idxOf e.END.FORMULA, hidx_lt⟩ =
                            e.END.FORMULA := by
                      rw [List.get_eq_getElem]
                      exact List.getElem_idxOf hidx_lt
                    simp
                  have hshape : RuleShapeOK e.END d :=
                    ruleShapeOK_of_valid_node d hvalid hend_node
                  obtain ⟨ruleIdx, hsel⟩ :=
                    ruleIndexForNode?_isSome_of_ruleShapeOK d e.END hvalid.hygiene hshape
                  obtain ⟨slot, src, hdec, hidxRule, hslot⟩ :=
                    inputLabelForEdge_decodes_of_ruleIndex d (buildFormulas d)
                      φ e.END ruleIdx hsel
                  refine ⟨hTarget, e.END, hend_node, hformula, ruleIdx, slot, src, hsel, ?_, hidxRule, hslot⟩
                  exact hdec
              | succ step =>
                  have hstep_lt_fuel : step < fuel := by
                    have hlen := routeFrom_length d (buildFormulas d) (Nat.succ fuel) φ
                    rw [hlen] at hStep; omega
                  cases hb : (match classifyRule? e.END d with
                    | some (DLDSRuleClass.elim major minor) =>
                        decide (φ = minor.START.FORMULA)
                    | _ => (false : Bool)) with
                  | false =>
                    -- Not minor: tail = routeFrom fuel e.END.FORMULA
                    have hStepTail : step < (routeFrom d (buildFormulas d) fuel e.END.FORMULA).length := by
                      rwa [routeFrom_length]
                    -- hexpand in get form so ▸ works directly
                    have hexpand : (routeFrom d (buildFormulas d) (Nat.succ fuel) φ).get ⟨step + 1, hStep⟩ =
                        (routeFrom d (buildFormulas d) fuel e.END.FORMULA).get ⟨step, hStepTail⟩ := by
                      simp only [List.get_eq_getElem, routeFrom, hfind, hout, hb,
                                 List.getElem_cons_succ]
                      simp
                    exact ih e.END.FORMULA step target label hStepTail (hexpand ▸ hget) htarget
                  | true =>
                    -- Minor: tail = replicate (0,0); every element is (0,0), target = 0
                    exfalso; apply htarget
                    have hStepTail : step < (List.replicate fuel (0, 0) : List (Nat × Nat)).length := by
                      simp [hstep_lt_fuel]
                    have hexpand : (routeFrom d (buildFormulas d) (Nat.succ fuel) φ).get ⟨step + 1, hStep⟩ =
                        (List.replicate fuel (0, 0) : List (Nat × Nat)).get ⟨step, hStepTail⟩ := by
                      simp only [List.get_eq_getElem, routeFrom, hfind, hout, hb,
                                 List.getElem_cons_succ]
                      simp
                    have hgetTail := hexpand ▸ hget
                    have hval : (List.replicate fuel (0, 0) : List (Nat × Nat)).get ⟨step, hStepTail⟩ =
                        (0, 0) := by simp [List.get_eq_getElem, List.getElem_replicate]
                    rw [hval] at hgetTail
                    exact (Prod.mk.inj hgetTail).1.symm

/-- Local coherence predicate for one evaluated node: the token group is either
    empty, or all arrivals decode to one existing rule, have the rule's arity,
    match the static source column for their decoded slots, and cover every slot.
    This is exactly the invariant intended to feed
    `nodeError_false_of_exact_rule_slots`. -/
def TokensMatchOneRule {n : Nat}
    (node : CircuitNode n) (node_incoming : NodeIncoming)
    (tokens : List (Token n)) : Prop :=
  match tokens with
  | [] => True
  | t :: ts =>
      ∃ r slot src,
        decodeInputLabel node_incoming t.input_label = some (r, slot, src) ∧
        r < node.rules.length ∧
        (∀ s ∈ t :: ts,
          ∃ slot' src',
            decodeInputLabel node_incoming s.input_label = some (r, slot', src') ∧
            slot' < (node_incoming[r]?.getD default).length ∧
            s.source_column = src') ∧
        (t :: ts).length = (node_incoming[r]?.getD default).length ∧
        (∀ i, i < (node_incoming[r]?.getD default).length →
          ∃ s ∈ t :: ts, ∃ src',
            decodeInputLabel node_incoming s.input_label = some (r, i, src') ∧
            s.source_column = src')

/-- Recursive token-coherence invariant along the exact descent threaded by
    `propagate_tokens` in `NoLayerError`. -/
def TokensMatchAlong {n : Nat} (paths : PathInput) (num_levels : Nat) :
    Nat → List (Token n) → List (GridLayer n) → Prop
  | _, _, [] => True
  | level, tokens, (layer :: rest) =>
      (∀ col (hNode : col < layer.nodes.length) (hIncoming : col < layer.incoming.length),
        TokensMatchOneRule
          (layer.nodes.get ⟨col, hNode⟩)
          (layer.incoming.get ⟨col, hIncoming⟩)
          (tokens.filter (fun t => t.current_column = col))) ∧
      TokensMatchAlong paths num_levels (level - 1)
        (propagate_tokens tokens paths level num_levels (evaluate_layer layer tokens).1) rest

/-- Recursive node-error invariant along the exact evaluator descent. This is
    the proof-level bridge from token coherence to `NoLayerError`. -/
def NodeErrorFalseAlong {n : Nat} (paths : PathInput) (num_levels : Nat) :
    Nat → List (Token n) → List (GridLayer n) → Prop
  | _, _, [] => True
  | level, tokens, (layer :: rest) =>
      (∀ col (hNode : col < layer.nodes.length) (hIncoming : col < layer.incoming.length),
        nodeError
          (layer.nodes.get ⟨col, hNode⟩)
          (layer.incoming.get ⟨col, hIncoming⟩)
          (tokens.filter (fun t => t.current_column = col)) = false) ∧
      NodeErrorFalseAlong paths num_levels (level - 1)
        (propagate_tokens tokens paths level num_levels (evaluate_layer layer tokens).1) rest

/-- Every non-stopping target that appears in a `routeFrom` tail is a valid
    formula column. This is the only part of the old label-coherence statement
    needed to keep replayed token columns in bounds. -/
lemma routeFrom_mem_target_lt (d : Graph) (hvalid : ValidDLDS d) :
    ∀ (fuel : Nat) (φ : Formula) (target label : Nat),
      (target, label) ∈ routeFrom d (buildFormulas d) fuel φ →
      target ≠ 0 →
      target - 1 < (buildFormulas d).length := by
  intro fuel
  induction fuel with
  | zero =>
      intro φ target label hmem _
      simp [routeFrom] at hmem
  | succ fuel ih =>
      intro φ target label hmem htarget
      cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
      | none =>
          simp [routeFrom, hfind] at hmem
          cases hmem with
          | inl hhead =>
              exact False.elim (htarget hhead.1)
          | inr htail =>
              exact ih φ target label htail htarget
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              simp [routeFrom, hfind, hout] at hmem
              cases hmem with
              | inl hhead =>
                  exact False.elim (htarget hhead.1)
              | inr htail =>
                  exact ih φ target label htail htarget
          | cons e es =>
              simp only [routeFrom, hfind, hout, List.mem_cons] at hmem
              cases hmem with
              | inl hhead =>
                  have htarget_eq :
                      target = (buildFormulas d).idxOf e.END.FORMULA + 1 :=
                    (Prod.mk.inj hhead).1
                  subst target
                  have he_out : e ∈ get_rule.outgoing v d := by
                    rw [hout]; simp
                  have he_edge : e ∈ d.EDGES := mem_outgoing_mem_edges v d he_out
                  have hend_node : e.END ∈ d.NODES := (hvalid.hygiene.2.1 he_edge).2
                  have hend_formula_mem : e.END.FORMULA ∈ buildFormulas d := by
                    unfold buildFormulas
                    exact List.mem_eraseDups.mpr
                      (List.mem_map.mpr ⟨e.END, hend_node, rfl⟩)
                  have hidx_lt :
                      (buildFormulas d).idxOf e.END.FORMULA < (buildFormulas d).length :=
                    List.idxOf_lt_length_of_mem hend_formula_mem
                  simpa using hidx_lt
              | inr htail =>
                  cases hb : (match classifyRule? e.END d with
                    | some (DLDSRuleClass.elim major minor) =>
                        decide (φ = minor.START.FORMULA)
                    | _ => (false : Bool)) with
                  | false =>
                      simp [hb] at htail
                      exact ih e.END.FORMULA target label htail htarget
                  | true =>
                      simp [hb] at htail
                      have hzero : target = 0 := by
                        exact htail.2.1
                      exact False.elim (htarget hzero)

/-- Every non-stopping target that appears anywhere in `pathsFromDLDS` is a valid
    formula column. Delayed hypothesis self-routes target their own column;
    route tails use `routeFrom_mem_target_lt`; stopped/padded routes are zero. -/
lemma pathsFromDLDS_mem_target_lt (d : Graph) (hvalid : ValidDLDS d)
    {inner : List (Nat × Nat)} (hinner : inner ∈ pathsFromDLDS d)
    {target label : Nat} (hmem : (target, label) ∈ inner) (htarget : target ≠ 0) :
    target - 1 < (buildFormulas d).length := by
  unfold pathsFromDLDS at hinner
  rw [List.mem_map] at hinner
  obtain ⟨φ, hφmem, hinner_eq⟩ := hinner
  subst inner
  cases hfind : d.NODES.find? (fun v => decide (v.FORMULA = φ)) with
  | none =>
      simp [hfind] at hmem
      have hzero : target = 0 := by
        exact hmem.2.1
      exact False.elim (htarget hzero)
  | some v =>
      by_cases hhyp : v.HYPOTHESIS
      · simp [hfind, hhyp] at hmem
        cases hmem with
        | inl hdelay =>
            have htarget_eq : target = (buildFormulas d).idxOf φ + 1 := by
              exact hdelay.2.1
            subst target
            have hidx_lt : (buildFormulas d).idxOf φ < (buildFormulas d).length :=
              List.idxOf_lt_length_of_mem hφmem
            simpa using hidx_lt
        | inr htail =>
            exact routeFrom_mem_target_lt d hvalid _ φ target label htail htarget
      · simp [hfind, hhyp] at hmem
        have hzero : target = 0 := by
          exact hmem.2.1
        exact False.elim (htarget hzero)

/-- **Singleton Flow-condition layer predicate.** In the simple-tree/no-colour
    fan-out/no-λ/no-collapse case of Definition 3, every column's token group is
    either empty or covers one existing rule's slots exactly once.

    This is exactly the per-column `TokensMatchOneRule` condition, so
    `TokensCoherentAtLayer layer tokens → nodeError_false_of_exact_rule_slots`
    holds for every column definitionally — lemma 3 below becomes trivial.

    `pathsFromDLDS exClosedABA = [[(2,1),(3,1),(0,0)], [(3,1),(0,0),(0,0)], [(0,0),(0,0),(0,0)]]`
    Layer 0 tokens [(0,0),(1,0),(2,0)] — rep wire, one per col;
    Layer 1 tokens [(1,1),(2,1)] — both carrying label 1 (rep at their col);
    Layer 2 tokens [(2,1)] — single rep token at col 2;
    Layer 3 tokens [] — empty after root is reached. -/
def TokensCoherentAtLayer {n : Nat}
    (layer : GridLayer n) (tokens : List (Token n)) : Prop :=
  ∀ col (hNode : col < layer.nodes.length) (hIncoming : col < layer.incoming.length),
    TokensMatchOneRule
      (layer.nodes.get ⟨col, hNode⟩)
      (layer.incoming.get ⟨col, hIncoming⟩)
      (tokens.filter (fun (t : Token n) => t.current_column = col))

/-- Replay the path followed by one origin token for `depth` propagation steps.
    The returned triple is `(current_column, source_column, input_label)` at the
    layer reached after that many steps. `none` means the token has stopped. -/
def routeStateAfter (paths : PathInput) (origin : Nat) : Nat → Option (Nat × Nat × Nat)
  | 0 =>
      if origin < paths.length then some (origin, origin, 0) else none
  | depth + 1 =>
      match routeStateAfter paths origin depth with
      | none => none
      | some (current, _source, _label) =>
          match paths[origin]? with
          | none => none
          | some steps =>
              match steps[depth]? with
              | none => none
              | some (target, inputLabel) =>
                  if target = 0 then none
                  else some (target - 1, current, inputLabel)

/-- One-step unfolding of `routeStateAfter` in the same bounded-`get` spelling
    used by `propagate_tokens`. -/
lemma routeStateAfter_get_eq
    (paths : PathInput) {origin depth current source label : Nat}
    (horigin : origin < paths.length)
    (hstate : routeStateAfter paths origin depth = some (current, source, label))
    (hidx : depth < (paths[origin]'horigin).length) :
    routeStateAfter paths origin (depth + 1) =
      (let step := (paths[origin]'horigin)[depth]'hidx
       if step.1 = 0 then none else some (step.1 - 1, current, step.2)) := by
  rw [show depth + 1 = Nat.succ depth by rfl]
  simp only [routeStateAfter, hstate]
  simp [List.getElem?_eq_getElem horigin, List.getElem?_eq_getElem hidx]

lemma routeStateAfter_live_succ
    (paths : PathInput) {origin depth current' source' label' : Nat}
    (hlive : routeStateAfter paths origin (depth + 1) =
      some (current', source', label')) :
    ∃ current source label steps target inputLabel,
      routeStateAfter paths origin depth = some (current, source, label) ∧
      paths[origin]? = some steps ∧
      steps[depth]? = some (target, inputLabel) ∧
      target ≠ 0 ∧
      (current', source', label') = (target - 1, current, inputLabel) := by
  rw [show depth + 1 = Nat.succ depth by rfl] at hlive
  unfold routeStateAfter at hlive
  cases hstate : routeStateAfter paths origin depth with
  | none =>
      simp [hstate] at hlive
  | some st =>
      rcases st with ⟨current, source, label⟩
      cases hsteps : paths[origin]? with
      | none =>
          simp [hstate, hsteps] at hlive
      | some steps =>
          cases hstep : steps[depth]? with
          | none =>
              simp [hstate, hsteps, hstep] at hlive
          | some step =>
              rcases step with ⟨target, inputLabel⟩
              by_cases hstop : target = 0
              · simp [hstate, hsteps, hstep, hstop] at hlive
              · simp [hstate, hsteps, hstep, hstop] at hlive
                rcases hlive with ⟨hcurrent, hsource, hlabel⟩
                subst current'
                subst source'
                subst label'
                exact ⟨current, source, label, steps, target, inputLabel,
                  rfl, rfl, hstep, hstop, rfl⟩

lemma routeStateAfter_live_succ_prev_source
    (paths : PathInput) {origin depth current' source' label' : Nat}
    (hlive : routeStateAfter paths origin (depth + 1) =
      some (current', source', label')) :
    ∃ prevSource prevLabel,
      routeStateAfter paths origin depth = some (source', prevSource, prevLabel) := by
  obtain ⟨current, prevSource, prevLabel, _steps, _target, _inputLabel,
    hprev, _hsteps, _hstep, _htarget, htriple⟩ :=
    routeStateAfter_live_succ paths hlive
  cases htriple
  exact ⟨prevSource, prevLabel, hprev⟩

lemma origin_lt_paths_length_of_routeStateAfter_some
    {paths : PathInput} {origin depth current source label : Nat}
    (hstate : routeStateAfter paths origin depth = some (current, source, label)) :
    origin < paths.length := by
  induction depth generalizing current source label with
  | zero =>
      unfold routeStateAfter at hstate
      by_cases hlt : origin < paths.length
      · exact hlt
      · simp [hlt] at hstate
  | succ depth ih =>
      obtain ⟨prevCurrent, prevSource, prevLabel, _steps, _target, _inputLabel,
        hprev, _hsteps, _hstep, _htarget, _htriple⟩ :=
        routeStateAfter_live_succ paths hstate
      exact ih hprev

lemma origin_lt_buildFormulas_length_of_routeStateAfter_pathsFromDLDS_some
    (d : Graph) {origin depth current source label : Nat}
    (hstate :
      routeStateAfter (pathsFromDLDS d) origin depth = some (current, source, label)) :
    origin < (buildFormulas d).length := by
  have hpath := origin_lt_paths_length_of_routeStateAfter_some hstate
  simpa [pathsFromDLDS_length] using hpath

/-- Principal carrier formula for a DLDS node, following the minor-stops /
    major-carries semantics:

    * a hypothesis carries itself;
    * an `⊃I` node is carried by its sole premise carrier;
    * an `⊃E` node is carried by its major-premise carrier.

    The fuel is a structural guard for malformed or cyclic inputs. In valid
    simple trees, levels decrease along incoming edges, so `d.NODES.length + 1`
    is sufficient. -/
def principalCarrierFormula? (d : Graph) : Nat → Vertex → Option Formula
  | 0, _ => none
  | Nat.succ fuel, v =>
      match classifyRule? v d with
      | some DLDSRuleClass.hypothesis => some v.FORMULA
      | some (DLDSRuleClass.intro p) => principalCarrierFormula? d fuel p.START
      | some (DLDSRuleClass.elim major _) => principalCarrierFormula? d fuel major.START
      | none => none

def principalCarrierColumn? (d : Graph) (v : Vertex) : Option Nat :=
  (principalCarrierFormula? d ((d.NODES.map (·.LEVEL)).foldl max 0 + 1) v).map
    (fun φ => (buildFormulas d).idxOf φ)

/-- Diagnostic for the principal-carrier invariant: at the descent depth
    corresponding to `v.LEVEL`, the principal carrier for `v` is occupying
    `v`'s formula column. -/
def principalCarrierAtNodeB (d : Graph) (v : Vertex) : Bool :=
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  let depth := maxLvl - v.LEVEL
  match principalCarrierColumn? d v with
  | none => false
  | some origin =>
      match routeStateAfter (pathsFromDLDS d) origin depth with
      | some (current, _, _) => current == (buildFormulas d).idxOf v.FORMULA
      | none => false

def principalCarriersB (d : Graph) : Bool :=
  d.NODES.all (principalCarrierAtNodeB d)



def sourceNodeAtColumn? (d : Graph) (col : Nat) : Option Vertex :=
  match (buildFormulas d)[col]? with
  | none => none
  | some φ => d.NODES.find? (fun v => decide (v.FORMULA = φ))

def principalCarrierForSourceColumn? (d : Graph) (src : Nat) : Option Nat :=
  match sourceNodeAtColumn? d src with
  | none => some src
  | some v => principalCarrierColumn? d v

lemma sourceNodeAtColumn?_eq_implies_formula_at_column
    (d : Graph) {col : Nat} {v : Vertex}
    (hsrc : sourceNodeAtColumn? d col = some v) :
    (buildFormulas d)[col]? = some v.FORMULA := by
  unfold sourceNodeAtColumn? at hsrc
  cases hφ : (buildFormulas d)[col]? with
  | none =>
      simp [hφ] at hsrc
  | some φ =>
      rw [hφ] at hsrc
      have hvφ : v.FORMULA = φ := by
        exact of_decide_eq_true
          (List.find?_some
            (p := fun u => decide (u.FORMULA = φ))
            (l := d.NODES) hsrc)
      simp [hvφ]

lemma sourceNodeAtColumn?_idxOf_of_mem (d : Graph) (hinj : InjFormulas d)
    {v : Vertex} (hv : v ∈ d.NODES) :
    sourceNodeAtColumn? d ((buildFormulas d).idxOf v.FORMULA) = some v := by
  unfold sourceNodeAtColumn?
  have hmemF : v.FORMULA ∈ buildFormulas d := by
    unfold buildFormulas
    exact List.mem_eraseDups.mpr (List.mem_map.mpr ⟨v, hv, rfl⟩)
  have hidxLt : (buildFormulas d).idxOf v.FORMULA < (buildFormulas d).length :=
    List.idxOf_lt_length_of_mem hmemF
  have hget? :
      (buildFormulas d)[(buildFormulas d).idxOf v.FORMULA]? =
        some v.FORMULA := by
    rw [List.getElem?_eq_getElem hidxLt]
    exact congrArg some (List.getElem_idxOf hidxLt)
  rw [hget?]
  exact find_node_by_formula_eq_of_inj d hinj hv

lemma principalCarrierForSourceColumn?_idxOf_of_mem (d : Graph) (hinj : InjFormulas d)
    {v : Vertex} (hv : v ∈ d.NODES) :
    principalCarrierForSourceColumn? d ((buildFormulas d).idxOf v.FORMULA) =
      principalCarrierColumn? d v := by
  unfold principalCarrierForSourceColumn?
  rw [sourceNodeAtColumn?_idxOf_of_mem d hinj hv]

lemma carrier_source_step_of_prev_principal
    (d : Graph) {depth origin col src lbl : Nat}
    (hprevPrincipal :
      ∀ prevSource prevLabel,
        routeStateAfter (pathsFromDLDS d) origin depth =
          some (src, prevSource, prevLabel) →
        principalCarrierForSourceColumn? d src = some origin)
    (hstep :
      routeStateAfter (pathsFromDLDS d) origin (depth + 1) =
        some (col, src, lbl)) :
    principalCarrierForSourceColumn? d src = some origin := by
  obtain ⟨prevSource, prevLabel, hprev⟩ :=
    routeStateAfter_live_succ_prev_source (pathsFromDLDS d) hstep
  exact hprevPrincipal prevSource prevLabel hprev

def expectedCarrierOriginsForRule? (d : Graph) (col ruleIdx : Nat) : Option (List Nat) := do
  let φ <- (buildFormulas d)[col]?
  let incoming := buildIncomingMapForFormula (buildFormulas d) φ
  let inc <- incoming[ruleIdx]?
  if ruleIdx + 1 = incoming.length then
    some [col]
  else
    inc.map Prod.fst |>.mapM (principalCarrierForSourceColumn? d)

def arrivingOriginsAt (d : Graph) (depth col : Nat) : List Nat :=
  let n := (buildFormulas d).length
  (List.range n).filter fun origin =>
    match routeStateAfter (pathsFromDLDS d) origin depth with
    | some (current, _, _) => current == col
    | none => false

def decodedRuleAtColumn? (d : Graph) (depth col : Nat) : Option Nat :=
  let n := (buildFormulas d).length
  match (List.range n).find? (fun origin =>
      match routeStateAfter (pathsFromDLDS d) origin depth with
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
          match decodeInputLabel (buildIncomingMapForFormula (buildFormulas d) φ) label with
          | some (ruleIdx, _, _) => some ruleIdx
          | none => none

def insertNat (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x <= y then x :: y :: ys else y :: insertNat x ys

def sortNats : List Nat → List Nat
  | [] => []
  | x :: xs => insertNat x (sortNats xs)

lemma insertNat_ne_nil (x : Nat) (xs : List Nat) : insertNat x xs ≠ [] := by
  cases xs with
  | nil => simp [insertNat]
  | cons y ys =>
      by_cases h : x <= y
      · simp [insertNat, h]
      · simp [insertNat, h]

lemma insertNat_perm_cons (x : Nat) :
    ∀ xs : List Nat, (insertNat x xs).Perm (x :: xs)
  | [] => by
      simp [insertNat]
  | y :: ys => by
      by_cases h : x <= y
      · simp [insertNat, h]
      · simpa [insertNat, h] using
          (List.Perm.cons y (insertNat_perm_cons x ys)).trans
            (List.Perm.swap x y ys)

lemma sortNats_perm (xs : List Nat) : (sortNats xs).Perm xs := by
  induction xs with
  | nil =>
      simp [sortNats]
  | cons x xs ih =>
      exact (insertNat_perm_cons x (sortNats xs)).trans
        (List.Perm.cons x ih)

lemma insertNat_comm (a b : Nat) : ∀ xs : List Nat,
    insertNat a (insertNat b xs) = insertNat b (insertNat a xs) := by
  intro xs
  induction xs with
  | nil =>
      by_cases hab : a <= b <;> by_cases hba : b <= a <;>
        simp [insertNat, hab, hba] <;> omega
  | cons x xs ih =>
      by_cases hbx : b <= x <;>
      by_cases hax : a <= x <;>
      by_cases hab : a <= b <;>
      by_cases hba : b <= a <;>
        simp [insertNat, hbx, hax, hab, hba, ih] <;> omega

lemma sortNats_eq_of_perm {xs ys : List Nat} (hperm : xs.Perm ys) :
    sortNats xs = sortNats ys := by
  induction hperm with
  | nil =>
      rfl
  | cons x _p ih =>
      simp [sortNats, ih]
  | swap x y l =>
      simp [sortNats, insertNat_comm]
  | trans _ _ ih1 ih2 =>
      exact ih1.trans ih2

lemma perm_cons_erase_of_mem {x : Nat} :
    ∀ {xs : List Nat}, x ∈ xs → xs.Perm (x :: xs.erase x)
  | [], h => by
      simp at h
  | y :: ys, h => by
      by_cases hxy : x = y
      · subst y
        simp
      · have htail : x ∈ ys := by
          simpa [hxy] using h
        have ih := perm_cons_erase_of_mem (x := x) (xs := ys) htail
        have hne : y ≠ x := by
          intro hyx
          exact hxy hyx.symm
        have hbeq : (y == x) = false := by
          simp [BEq.beq, hne]
        simpa [List.erase, hbeq] using
          (List.Perm.cons y ih).trans (List.Perm.swap x y (ys.erase x))

lemma perm_of_subset_subset_nodup
    (xs ys : List Nat)
    (hxs : xs.Nodup) (hys : ys.Nodup)
    (hxy : ∀ x, x ∈ xs → x ∈ ys)
    (hyx : ∀ x, x ∈ ys → x ∈ xs) :
    xs.Perm ys := by
  induction xs generalizing ys with
  | nil =>
      cases ys with
      | nil => exact List.Perm.nil
      | cons y ys =>
          have hy : y ∈ ([] : List Nat) := hyx y (by simp)
          simp at hy
  | cons z zs ih =>
      have hz_notin : z ∉ zs := (List.nodup_cons.mp hxs).1
      have hzs_tail : zs.Nodup := (List.nodup_cons.mp hxs).2
      have hzys : z ∈ ys := hxy z (by simp)
      have hys_perm : ys.Perm (z :: ys.erase z) := perm_cons_erase_of_mem hzys
      have hys_erase_nodup : (ys.erase z).Nodup := List.Nodup.erase z hys
      have hz_notin_erase : z ∉ ys.erase z :=
        (List.nodup_cons.mp ((List.Perm.nodup_iff hys_perm).mp hys)).1
      have hxy_tail : ∀ a, a ∈ zs → a ∈ ys.erase z := by
        intro a ha
        have hays : a ∈ ys := hxy a (by simp [ha])
        have haz : a ≠ z := by
          intro h
          subst h
          exact hz_notin ha
        exact (List.mem_erase_of_ne haz).2 hays
      have hyx_tail : ∀ a, a ∈ ys.erase z → a ∈ zs := by
        intro a ha
        have hmem_cons : a ∈ z :: zs := hyx a (List.mem_of_mem_erase ha)
        have haz : a ≠ z := by
          intro h
          subst h
          exact hz_notin_erase ha
        simpa [haz] using hmem_cons
      exact (List.Perm.cons z
          (ih (ys.erase z) hzs_tail hys_erase_nodup hxy_tail hyx_tail)).trans
        hys_perm.symm

lemma sortNats_eq_of_subset_subset_nodup
    (xs ys : List Nat)
    (hxs : xs.Nodup) (hys : ys.Nodup)
    (hxy : ∀ x, x ∈ xs → x ∈ ys)
    (hyx : ∀ x, x ∈ ys → x ∈ xs) :
    sortNats xs = sortNats ys :=
  sortNats_eq_of_perm (perm_of_subset_subset_nodup xs ys hxs hys hxy hyx)

lemma sortNats_eq_nil_iff (xs : List Nat) : sortNats xs = [] ↔ xs = [] := by
  constructor
  · intro h
    cases xs with
    | nil => rfl
    | cons x xs =>
        simp [sortNats] at h
        exact False.elim ((insertNat_ne_nil x (sortNats xs)) h)
  · intro h
    subst h
    rfl

lemma sortNats_isEmpty_eq_true_iff (xs : List Nat) :
    (sortNats xs).isEmpty = true ↔ xs = [] := by
  rw [List.isEmpty_iff]
  exact sortNats_eq_nil_iff xs

lemma find?_some_mem {α : Type*} {xs : List α} {p : α → Bool} {x : α}
    (h : xs.find? p = some x) : x ∈ xs := by
  induction xs with
  | nil =>
      simp at h
  | cons y ys ih =>
      by_cases hp : p y = true
      · simp [List.find?, hp] at h
        subst x
        simp
      · have hpFalse : p y = false := by
          cases hy : p y <;> simp [hy] at hp ⊢
        simp [List.find?, hpFalse] at h
        simp [ih h]

lemma sourceNodeAtColumn?_mem (d : Graph) {col : Nat} {v : Vertex}
    (hsrc : sourceNodeAtColumn? d col = some v) :
    v ∈ d.NODES := by
  unfold sourceNodeAtColumn? at hsrc
  cases hφ : (buildFormulas d)[col]? with
  | none =>
      simp [hφ] at hsrc
  | some φ =>
      rw [hφ] at hsrc
      exact find?_some_mem hsrc

def placementCoherentAtColumnB (d : Graph) (depth col : Nat) : Bool :=
  let arrivals := sortNats (arrivingOriginsAt d depth col)
  if arrivals.isEmpty then true
  else
    match decodedRuleAtColumn? d depth col with
    | none => false
    | some ruleIdx =>
        match expectedCarrierOriginsForRule? d col ruleIdx with
        | none => false
        | some expected => decide (sortNats expected = arrivals)

def placementCoherentB (d : Graph) : Bool :=
  (List.range (buildGridFromDLDS d).length).all fun depth =>
    (List.range (buildFormulas d).length).all fun col =>
      placementCoherentAtColumnB d depth col

/-- Membership in `arrivingOriginsAt` is exactly bounded origin membership plus
    a live route state whose current column is the queried column. -/
lemma mem_arrivingOriginsAt_iff (d : Graph) (depth col origin : Nat) :
    origin ∈ arrivingOriginsAt d depth col ↔
      origin < (buildFormulas d).length ∧
        ∃ source label,
          routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label) := by
  unfold arrivingOriginsAt
  rw [List.mem_filter]
  constructor
  · intro h
    rcases h with ⟨hrange, hroute⟩
    have horigin : origin < (buildFormulas d).length := List.mem_range.mp hrange
    constructor
    · exact horigin
    · cases hstate : routeStateAfter (pathsFromDLDS d) origin depth with
      | none =>
          simp [hstate] at hroute
      | some st =>
          rcases st with ⟨current, source, label⟩
          have hcurrent : current = col := by
            simpa [hstate, BEq.beq, decide_eq_true_eq] using hroute
          exact ⟨source, label, by simp [hcurrent]⟩
  · intro h
    rcases h with ⟨horigin, source, label, hstate⟩
    constructor
    · exact List.mem_range.mpr horigin
    · simp [hstate]

lemma mem_arrivingOriginsAt_origin_lt {d : Graph} {depth col origin : Nat}
    (h : origin ∈ arrivingOriginsAt d depth col) :
    origin < (buildFormulas d).length :=
  (mem_arrivingOriginsAt_iff d depth col origin).mp h |>.1

lemma mem_arrivingOriginsAt_state {d : Graph} {depth col origin : Nat}
    (h : origin ∈ arrivingOriginsAt d depth col) :
    ∃ source label,
      routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label) :=
  (mem_arrivingOriginsAt_iff d depth col origin).mp h |>.2

lemma arrivingOriginsAt_mem_of_state {d : Graph} {depth col origin source label : Nat}
    (horigin : origin < (buildFormulas d).length)
    (hstate : routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label)) :
    origin ∈ arrivingOriginsAt d depth col :=
  (mem_arrivingOriginsAt_iff d depth col origin).mpr ⟨horigin, source, label, hstate⟩

lemma arrivingOriginsAt_nodup (d : Graph) (depth col : Nat) :
    (arrivingOriginsAt d depth col).Nodup := by
  unfold arrivingOriginsAt
  exact List.Nodup.sublist List.filter_sublist List.nodup_range

lemma filter_range_eq_singleton (n col : Nat) (hcol : col < n) :
    (List.range n).filter (fun origin => origin == col) = [col] := by
  induction n with
  | zero =>
      omega
  | succ n ih =>
      by_cases hcn : col = n
      · subst col
        have hnotin : ∀ x, x ∈ List.range n → (x == n) = false := by
          intro x hx
          have hxlt : x < n := List.mem_range.mp hx
          simp [BEq.beq, decide_eq_false_iff_not]
          omega
        have hfilter_nil : (List.range n).filter (fun origin => origin == n) = [] := by
          apply List.eq_nil_iff_forall_not_mem.mpr
          intro x hx
          rw [List.mem_filter] at hx
          have hfalse := hnotin x hx.1
          rw [hfalse] at hx
          simp at hx
        simp [List.range_succ, hfilter_nil]
      · have hcol_lt_n : col < n := by omega
        have hne : (n == col) = false := by
          simp [BEq.beq, decide_eq_false_iff_not]
          omega
        simp [List.range_succ, hne, ih hcol_lt_n]

lemma find?_range_beq_eq (n col : Nat) (hcol : col < n) :
    (List.range n).find? (fun origin => origin == col) = some col := by
  induction n with
  | zero =>
      omega
  | succ n ih =>
      by_cases hcn : col = n
      · subst col
        have hfind_none : (List.range n).find? (fun origin => origin == n) = none := by
          rw [List.find?_eq_none]
          intro x hx
          have hxlt : x < n := List.mem_range.mp hx
          simp [BEq.beq, decide_eq_true_eq]
          omega
        simp [List.range_succ, hfind_none]
      · have hcol_lt_n : col < n := by omega
        have hne : (n == col) = false := by
          simp [BEq.beq, decide_eq_false_iff_not]
          omega
        simp [List.range_succ, hne, ih hcol_lt_n]

lemma arrivingOriginsAt_zero_eq_singleton (d : Graph) {col : Nat}
    (hcol : col < (buildFormulas d).length) :
    arrivingOriginsAt d 0 col = [col] := by
  unfold arrivingOriginsAt
  have hcongr :
      List.filter
          (fun origin =>
            match routeStateAfter (pathsFromDLDS d) origin 0 with
            | some (current, _, _) => current == col
            | none => false)
          (List.range (buildFormulas d).length) =
        List.filter (fun origin => origin == col)
          (List.range (buildFormulas d).length) := by
    apply List.filter_congr
    intro origin horigin
    have horigin_lt : origin < (buildFormulas d).length := List.mem_range.mp horigin
    have hpath : origin < (pathsFromDLDS d).length := by
      simpa [pathsFromDLDS] using horigin_lt
    simp [routeStateAfter, hpath]
  rw [hcongr]
  exact filter_range_eq_singleton (buildFormulas d).length col hcol

lemma placementCoherentAtColumnB_of_no_arrivals
    (d : Graph) (depth col : Nat)
    (h : arrivingOriginsAt d depth col = []) :
    placementCoherentAtColumnB d depth col = true := by
  unfold placementCoherentAtColumnB
  have hempty : (sortNats (arrivingOriginsAt d depth col)).isEmpty = true := by
    rw [sortNats_isEmpty_eq_true_iff]
    exact h
  simp [hempty]

lemma placementCoherentAtColumnB_of_expected_carriers
    (d : Graph) (depth col ruleIdx : Nat) (expected : List Nat)
    (harr : arrivingOriginsAt d depth col ≠ [])
    (hdec : decodedRuleAtColumn? d depth col = some ruleIdx)
    (hexp : expectedCarrierOriginsForRule? d col ruleIdx = some expected)
    (heq : sortNats expected = sortNats (arrivingOriginsAt d depth col)) :
    placementCoherentAtColumnB d depth col = true := by
  unfold placementCoherentAtColumnB
  have hnotEmpty : (sortNats (arrivingOriginsAt d depth col)).isEmpty = false := by
    cases hsort : sortNats (arrivingOriginsAt d depth col) with
    | nil =>
        have hnil : arrivingOriginsAt d depth col = [] :=
          (sortNats_eq_nil_iff (arrivingOriginsAt d depth col)).mp hsort
        exact False.elim (harr hnil)
      | cons _ _ => rfl
  simp [hnotEmpty, hdec, hexp, heq]

lemma placementCoherentAtColumnB_of_subsets
    (d : Graph) (depth col ruleIdx : Nat) (expected : List Nat)
    (harr : arrivingOriginsAt d depth col ≠ [])
    (hdec : decodedRuleAtColumn? d depth col = some ruleIdx)
    (hexp : expectedCarrierOriginsForRule? d col ruleIdx = some expected)
    (hexp_nodup : expected.Nodup)
    (hexp_to_arr :
      ∀ origin, origin ∈ expected → origin ∈ arrivingOriginsAt d depth col)
    (harr_to_exp :
      ∀ origin, origin ∈ arrivingOriginsAt d depth col → origin ∈ expected) :
    placementCoherentAtColumnB d depth col = true := by
  apply placementCoherentAtColumnB_of_expected_carriers d depth col ruleIdx expected
    harr hdec hexp
  exact sortNats_eq_of_subset_subset_nodup expected (arrivingOriginsAt d depth col)
    hexp_nodup (arrivingOriginsAt_nodup d depth col) hexp_to_arr harr_to_exp

lemma arrivals_eq_expected_of_subset_parts
    (d : Graph) (depth col : Nat) (expected : List Nat)
    (hexp_nodup : expected.Nodup)
    (hexp_to_arr :
      ∀ origin, origin ∈ expected → origin ∈ arrivingOriginsAt d depth col)
    (harr_to_exp :
      ∀ origin, origin ∈ arrivingOriginsAt d depth col → origin ∈ expected) :
    sortNats expected = sortNats (arrivingOriginsAt d depth col) :=
  sortNats_eq_of_subset_subset_nodup expected (arrivingOriginsAt d depth col)
    hexp_nodup (arrivingOriginsAt_nodup d depth col) hexp_to_arr harr_to_exp

lemma mem_of_list_mapM_some {α β : Type*} {f : α → Option β}
    {xs : List α} {ys : List β} {x : α} {y : β}
    (hmap : xs.mapM f = some ys)
    (hx : x ∈ xs) (hf : f x = some y) :
    y ∈ ys := by
  induction xs generalizing ys with
  | nil =>
      simp at hx
  | cons a rest ih =>
      simp only [List.mapM_cons] at hmap
      cases hfa : f a with
      | none =>
          simp [hfa] at hmap
      | some ya =>
          simp [hfa] at hmap
          obtain ⟨ysRest, hrest, hys⟩ := Option.bind_eq_some_iff.mp hmap
          cases hx with
          | head =>
              have hya : ya = y := Option.some.inj (hfa.symm.trans hf)
              have hysEq : ys = ya :: ysRest := (Option.some.inj hys).symm
              rw [hysEq, hya]
              simp
          | tail _ hxrest =>
              have hysEq : ys = ya :: ysRest := (Option.some.inj hys).symm
              rw [hysEq]
              exact List.mem_cons_of_mem _ (ih hrest hxrest)

lemma exists_of_mem_list_mapM_some {α β : Type*} {f : α → Option β}
    {xs : List α} {ys : List β} {y : β}
    (hmap : xs.mapM f = some ys)
    (hy : y ∈ ys) :
    ∃ x, x ∈ xs ∧ f x = some y := by
  induction xs generalizing ys with
  | nil =>
      simp at hmap
      subst ys
      simp at hy
  | cons a rest ih =>
      simp only [List.mapM_cons] at hmap
      cases hfa : f a with
      | none =>
          simp [hfa] at hmap
      | some ya =>
          simp [hfa] at hmap
          obtain ⟨ysRest, hrest, hys⟩ := Option.bind_eq_some_iff.mp hmap
          have hysEq : ys = ya :: ysRest := (Option.some.inj hys).symm
          rw [hysEq] at hy
          cases hy with
          | head =>
              exact ⟨a, by simp, by simpa using hfa⟩
          | tail _ hyrest =>
              obtain ⟨x, hx, hfx⟩ := ih hrest hyrest
              exact ⟨x, by simp [hx], hfx⟩

lemma mem_expectedCarrierOriginsForRule?_of_nonrep_source
    (d : Graph) {col ruleIdx src origin : Nat} {φ : Formula}
    {inc : RuleIncoming} {expected : List Nat}
    (hφ : (buildFormulas d)[col]? = some φ)
    (hentry :
      (buildIncomingMapForFormula (buildFormulas d) φ)[ruleIdx]? = some inc)
    (hnonrep : ruleIdx + 1 ≠
      (buildIncomingMapForFormula (buildFormulas d) φ).length)
    (hsrc : src ∈ inc.map Prod.fst)
    (hprincipal : principalCarrierForSourceColumn? d src = some origin)
    (hexp : expectedCarrierOriginsForRule? d col ruleIdx = some expected) :
    origin ∈ expected := by
  unfold expectedCarrierOriginsForRule? at hexp
  rw [hφ] at hexp
  simp [hentry, hnonrep] at hexp
  rw [List.mem_map] at hsrc
  obtain ⟨pair, hpair, hsrc_eq⟩ := hsrc
  subst hsrc_eq
  exact mem_of_list_mapM_some hexp hpair hprincipal

lemma source_of_mem_expectedCarrierOriginsForRule?_nonrep
    (d : Graph) {col ruleIdx origin : Nat} {φ : Formula}
    {inc : RuleIncoming} {expected : List Nat}
    (hφ : (buildFormulas d)[col]? = some φ)
    (hentry :
      (buildIncomingMapForFormula (buildFormulas d) φ)[ruleIdx]? = some inc)
    (hnonrep : ruleIdx + 1 ≠
      (buildIncomingMapForFormula (buildFormulas d) φ).length)
    (hexp : expectedCarrierOriginsForRule? d col ruleIdx = some expected)
    (hmem : origin ∈ expected) :
    ∃ src edgeId,
      (src, edgeId) ∈ inc ∧
        principalCarrierForSourceColumn? d src = some origin := by
  unfold expectedCarrierOriginsForRule? at hexp
  rw [hφ] at hexp
  simp [hentry, hnonrep] at hexp
  obtain ⟨pair, hpair, hprincipal⟩ :=
    exists_of_mem_list_mapM_some hexp hmem
  exact ⟨pair.1, pair.2, hpair, by simpa using hprincipal⟩

lemma expectedCarrierOriginsForRule?_rep
    (d : Graph) {col : Nat} {φ : Formula}
    (hφ : (buildFormulas d)[col]? = some φ) :
    expectedCarrierOriginsForRule? d col
        ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) =
      some [col] := by
  have hlen : 0 < (buildIncomingMapForFormula (buildFormulas d) φ).length := by
    unfold buildIncomingMapForFormula
    cases φ <;> simp
  have hidx : ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) <
      (buildIncomingMapForFormula (buildFormulas d) φ).length := by
    omega
  have hlast :
      ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) + 1 =
        (buildIncomingMapForFormula (buildFormulas d) φ).length := by
    omega
  simp [expectedCarrierOriginsForRule?, hφ, List.getElem?_eq_getElem hidx, hlast]

lemma find?_congr_on {α : Type*} {xs : List α} {p q : α → Bool}
    (h : ∀ x, x ∈ xs → p x = q x) :
    xs.find? p = xs.find? q := by
  induction xs with
  | nil =>
      rfl
  | cons x xs ih =>
      have hx : p x = q x := h x (by simp)
      have htail : ∀ y, y ∈ xs → p y = q y := by
        intro y hy
        exact h y (by simp [hy])
      simp [List.find?, hx, ih htail]

lemma exists_state_of_arrivingOriginsAt_ne_nil
    {d : Graph} {depth col : Nat}
    (h : arrivingOriginsAt d depth col ≠ []) :
    ∃ origin source label,
      origin < (buildFormulas d).length ∧
      routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label) := by
  cases harr : arrivingOriginsAt d depth col with
  | nil =>
      exact False.elim (h harr)
  | cons origin rest =>
      have hmem : origin ∈ arrivingOriginsAt d depth col := by
        rw [harr]
        simp
      obtain ⟨horigin, source, label, hstate⟩ :=
        (mem_arrivingOriginsAt_iff d depth col origin).mp hmem
      exact ⟨origin, source, label, horigin, hstate⟩

lemma decodedRuleAtColumn?_some_route
    {d : Graph} {depth col ruleIdx : Nat}
    (hdec : decodedRuleAtColumn? d depth col = some ruleIdx) :
    ∃ origin source label φ slot src,
      origin < (buildFormulas d).length ∧
      routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label) ∧
      (buildFormulas d)[col]? = some φ ∧
      decodeInputLabel (buildIncomingMapForFormula (buildFormulas d) φ) label =
        some (ruleIdx, slot, src) := by
  unfold decodedRuleAtColumn? at hdec
  cases hfind : (List.range (buildFormulas d).length).find? (fun origin =>
      match routeStateAfter (pathsFromDLDS d) origin depth with
      | some (current, _, _) => current == col
      | none => false) with
  | none =>
      simp [hfind] at hdec
  | some origin =>
      have horigin : origin < (buildFormulas d).length := by
        exact List.mem_range.mp (find?_some_mem hfind)
      have hpred := List.find?_some hfind
      cases hstate : routeStateAfter (pathsFromDLDS d) origin depth with
      | none =>
          simp [hfind, hstate] at hdec
      | some st =>
          rcases st with ⟨current, source, label⟩
          have hcurrent : current = col := by
            simpa [hstate, BEq.beq, decide_eq_true_eq] using hpred
          cases hφ : (buildFormulas d)[col]? with
          | none =>
              simp [hfind, hstate, hφ] at hdec
          | some φ =>
              cases hlabel :
                  decodeInputLabel (buildIncomingMapForFormula (buildFormulas d) φ) label with
              | none =>
                  simp [hfind, hstate, hφ, hlabel] at hdec
              | some triple =>
                  rcases triple with ⟨r, slot, src⟩
                  simp [hfind, hstate, hφ, hlabel] at hdec
                  subst r
                  exact ⟨origin, source, label, φ, slot, src, horigin,
                    by simpa [hcurrent] using hstate,
                    by simp, hlabel⟩

lemma placementCoherentAtColumnB_of_placementCoherentB
    (d : Graph) (hplace : placementCoherentB d = true)
    {depth col : Nat}
    (hdepth : depth ∈ List.range (buildGridFromDLDS d).length)
    (hcol : col ∈ List.range (buildFormulas d).length) :
    placementCoherentAtColumnB d depth col = true := by
  unfold placementCoherentB at hplace
  rw [List.all_eq_true] at hplace
  have hdepth_all := hplace depth hdepth
  rw [List.all_eq_true] at hdepth_all
  exact hdepth_all col hcol

lemma principalCarrierFormula?_hyp (d : Graph) (fuel : Nat) (v : Vertex)
    (hclass : classifyRule? v d = some DLDSRuleClass.hypothesis) :
    principalCarrierFormula? d (Nat.succ fuel) v = some v.FORMULA := by
  simp [principalCarrierFormula?, hclass]

lemma principalCarrierFormula?_intro (d : Graph) (fuel : Nat)
    (v : Vertex) (p : Deduction)
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p)) :
    principalCarrierFormula? d (Nat.succ fuel) v =
      principalCarrierFormula? d fuel p.START := by
  simp [principalCarrierFormula?, hclass]

lemma principalCarrierFormula?_elim (d : Graph) (fuel : Nat)
    (v : Vertex) (major minor : Deduction)
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor)) :
    principalCarrierFormula? d (Nat.succ fuel) v =
      principalCarrierFormula? d fuel major.START := by
  simp [principalCarrierFormula?, hclass]

lemma getElem?_some_lt {α : Type*} {xs : List α} {i : Nat} {x : α}
    (h : xs[i]? = some x) : i < xs.length := by
  exact (List.getElem?_eq_some_iff.mp h).1

lemma getElem?_map_fst_eq_some {xs : List (Nat × Nat)} {i src : Nat}
    (h : (xs.map Prod.fst)[i]? = some src) :
    ∃ edgeId, xs[i]? = some (src, edgeId) := by
  induction xs generalizing i with
  | nil =>
      simp at h
  | cons p xs ih =>
      cases i with
      | zero =>
          cases p with
          | mk a b =>
              simp at h
              subst h
              exact ⟨b, rfl⟩
      | succ i =>
          simp at h
          simpa using h

lemma path_get_of_origin_eq {n : Nat}
    (paths : PathInput) {t : Token n} {origin : Nat} {steps : List (Nat × Nat)}
    (horigin : t.origin_column = origin)
    (hsteps : paths[origin]? = some steps)
    (hpath : t.origin_column < paths.length) :
    paths.get ⟨t.origin_column, hpath⟩ = steps := by
  rw [← Option.some.injEq]
  have hsome :
      paths[t.origin_column]? =
        some (paths.get ⟨t.origin_column, hpath⟩) := by
    simp [List.get_eq_getElem]
  rw [← hsome]
  rw [horigin]
  exact hsteps

lemma path_step_get_of_step_eq {α : Type*}
    {steps : List α} {evalIdx i : Nat} {x : α}
    (hstep : steps[i]? = some x)
    (hidx : evalIdx = i)
    (hbound : evalIdx < steps.length) :
    steps.get ⟨evalIdx, hbound⟩ = x := by
  rw [← Option.some.injEq]
  have hsome :
      steps[evalIdx]? = some (steps.get ⟨evalIdx, hbound⟩) := by
    simp [List.get_eq_getElem]
  rw [← hsome]
  rw [hidx]
  exact hstep

lemma routeStateAfter_pathsFromDLDS_current_lt
    (d : Graph) (_htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d)
    {origin depth current source label : Nat}
    (horigin : origin < (buildFormulas d).length)
    (hstate : routeStateAfter (pathsFromDLDS d) origin depth =
      some (current, source, label)) :
    current < (buildFormulas d).length := by
  induction depth generalizing current source label with
  | zero =>
      unfold routeStateAfter at hstate
      have hpath : origin < (pathsFromDLDS d).length := by
        simpa [pathsFromDLDS] using horigin
      simp [hpath] at hstate
      rcases hstate with ⟨rfl, rfl, rfl⟩
      exact horigin
  | succ depth ih =>
      obtain ⟨prev, prevSrc, prevLbl, steps, target, inputLabel,
        hprev, hsteps, hstep, htarget, htriple⟩ :=
        routeStateAfter_live_succ (pathsFromDLDS d) hstate
      rcases htriple with ⟨rfl, rfl, rfl⟩
      have hOriginPath : origin < (pathsFromDLDS d).length := by
        simpa [pathsFromDLDS] using horigin
      have hstepsGet : (pathsFromDLDS d).get ⟨origin, hOriginPath⟩ = steps := by
        rw [← Option.some.injEq]
        simpa [List.getElem?_eq_getElem hOriginPath] using hsteps
      subst steps
      have hstepLt : depth < ((pathsFromDLDS d).get ⟨origin, hOriginPath⟩).length := by
        exact getElem?_some_lt hstep
      have hstepGet :
          ((pathsFromDLDS d).get ⟨origin, hOriginPath⟩).get ⟨depth, hstepLt⟩ =
            (target, label) := by
        rw [List.get_eq_getElem]
        exact (List.getElem?_eq_some_iff.mp hstep).2
      have hinner_mem :
          (pathsFromDLDS d).get ⟨origin, hOriginPath⟩ ∈ pathsFromDLDS d :=
        List.get_mem _ _
      have hentry_mem :
          (target, label) ∈ (pathsFromDLDS d).get ⟨origin, hOriginPath⟩ := by
        rw [← hstepGet]
        exact List.get_mem _ _
      exact pathsFromDLDS_mem_target_lt d hvalid hinner_mem hentry_mem htarget

lemma evaluate_layer_outputs_length {n : Nat}
    (layer : GridLayer n) (tokens : List (Token n)) :
    (evaluate_layer layer tokens).1.length = layer.nodes.length := by
  unfold evaluate_layer
  simp [List.length_zipIdx]

lemma buildGridFromDLDS_get_nodes_length (d : Graph)
    (i : Nat) (hi : i < (buildGridFromDLDS d).length) :
    (((buildGridFromDLDS d).get ⟨i, hi⟩).nodes.length =
      (buildFormulas d).length) := by
  unfold buildGridFromDLDS buildLayers
  simp

lemma buildGridFromDLDS_get_incoming_length (d : Graph)
    (i : Nat) (hi : i < (buildGridFromDLDS d).length) :
    (((buildGridFromDLDS d).get ⟨i, hi⟩).incoming.length =
      (buildFormulas d).length) := by
  unfold buildGridFromDLDS buildLayers
  simp [buildIncomingMap_length]

/-- Token-level strengthened route invariant. It pins down all routing fields
    from the origin column and the number of propagation steps already taken. -/
def TokenWellRoutedAtDepth {n : Nat} (d : Graph) (depth : Nat) (t : Token n) : Prop :=
  t.origin_column < (buildFormulas d).length ∧
  routeStateAfter (pathsFromDLDS d) t.origin_column depth =
    some (t.current_column, t.source_column, t.input_label)

/-- Strengthened layer invariant: every token is the genuine token obtained by
    replaying `pathsFromDLDS` to this layer depth, and the old per-column
    rule-group coherence also holds. -/
def TokensWellRoutedAtLayer {n : Nat}
    (d : Graph) (depth : Nat) (layer : GridLayer n) (tokens : List (Token n)) : Prop :=
  (∀ t ∈ tokens, TokenWellRoutedAtDepth d depth t) ∧
  TokensCoherentAtLayer layer tokens

/-- Raw route agreement only, without per-column slot coherence.  This is the
    non-circular premise needed for the future placement-to-token bridge. -/
def TokensRouteAlignedAtDepth {n : Nat}
    (d : Graph) (depth : Nat) (tokens : List (Token n)) : Prop :=
  ∀ t ∈ tokens, TokenWellRoutedAtDepth d depth t

lemma routeAligned_token_mem_arrivingOriginsAt {n : Nat}
    {d : Graph} {depth : Nat} {tokens : List (Token n)} {t : Token n}
    (halign : TokensRouteAlignedAtDepth d depth tokens)
    (ht : t ∈ tokens) :
    t.origin_column ∈ arrivingOriginsAt d depth t.current_column := by
  have htwr := halign t ht
  exact arrivingOriginsAt_mem_of_state htwr.1 htwr.2

lemma TokensWellRoutedAtLayer.coherent {n : Nat}
    {d : Graph} {depth : Nat} {layer : GridLayer n} {tokens : List (Token n)}
    (h : TokensWellRoutedAtLayer d depth layer tokens) :
    TokensCoherentAtLayer layer tokens :=
  h.2

lemma TokensWellRoutedAtLayer.routeAligned {n : Nat}
    {d : Graph} {depth : Nat} {layer : GridLayer n} {tokens : List (Token n)}
    (h : TokensWellRoutedAtLayer d depth layer tokens) :
    TokensRouteAlignedAtDepth d depth tokens :=
  h.1

/-- Presence/exactness clause: at a layer depth, every live route has exactly
    one token with that origin, and every stopped route has no token. This closes
    the "dropped premise" gap left by `TokensWellRoutedAtLayer`. -/
def TokensPresenceAtDepth {n : Nat}
    (d : Graph) (depth : Nat) (tokens : List (Token n)) : Prop :=
  ∀ origin, origin < (buildFormulas d).length →
    match routeStateAfter (pathsFromDLDS d) origin depth with
    | none =>
        ∀ t ∈ tokens, t.origin_column ≠ origin
    | some _ =>
        ∃ t ∈ tokens,
          t.origin_column = origin ∧
          ∀ s ∈ tokens, s.origin_column = origin → s = t

lemma presence_aligned_token_of_arrivingOrigin {n : Nat}
    {d : Graph} {depth col origin source label : Nat}
    {tokens : List (Token n)}
    (halign : TokensRouteAlignedAtDepth d depth tokens)
    (hpres : TokensPresenceAtDepth d depth tokens)
    (hstate :
      routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label))
    (horigin : origin < (buildFormulas d).length) :
    ∃ t ∈ tokens,
      t.origin_column = origin ∧
      t.current_column = col ∧
      t.source_column = source ∧
      t.input_label = label ∧
      ∀ s ∈ tokens, s.origin_column = origin → s = t := by
  have hpres_origin := hpres origin horigin
  rw [hstate] at hpres_origin
  obtain ⟨t, ht, ht_origin, huniq⟩ := hpres_origin
  have htwr := halign t ht
  have htstate :
      routeStateAfter (pathsFromDLDS d) origin depth =
        some (t.current_column, t.source_column, t.input_label) := by
    simpa [ht_origin] using htwr.2
  have hsame :
      (t.current_column, t.source_column, t.input_label) = (col, source, label) := by
    exact Option.some.inj (htstate.symm.trans hstate)
  cases hsame
  exact ⟨t, ht, ht_origin, rfl, rfl, rfl, huniq⟩

lemma presence_aligned_token_of_arrivingOriginsAt {n : Nat}
    {d : Graph} {depth col origin : Nat} {tokens : List (Token n)}
    (halign : TokensRouteAlignedAtDepth d depth tokens)
    (hpres : TokensPresenceAtDepth d depth tokens)
    (hmem : origin ∈ arrivingOriginsAt d depth col) :
    ∃ source label t,
      t ∈ tokens ∧
      t.origin_column = origin ∧
      t.current_column = col ∧
      t.source_column = source ∧
      t.input_label = label ∧
      routeStateAfter (pathsFromDLDS d) origin depth = some (col, source, label) ∧
      ∀ s ∈ tokens, s.origin_column = origin → s = t := by
  have horigin := mem_arrivingOriginsAt_origin_lt hmem
  obtain ⟨source, label, hstate⟩ := mem_arrivingOriginsAt_state hmem
  obtain ⟨t, ht, ht_origin, hcur, hsrc, hlbl, huniq⟩ :=
    presence_aligned_token_of_arrivingOrigin
      halign hpres hstate horigin
  exact ⟨source, label, t, ht, ht_origin, hcur, hsrc, hlbl, hstate, huniq⟩

def tokenOriginsAtColumn {n : Nat} (tokens : List (Token n)) (col : Nat) : List Nat :=
  (tokens.filter (fun t => t.current_column = col)).map (fun t => t.origin_column)

lemma tokenOriginsAtColumn_subset_arrivingOriginsAt {n : Nat}
    {d : Graph} {depth col origin : Nat} {tokens : List (Token n)}
    (halign : TokensRouteAlignedAtDepth d depth tokens)
    (hmem : origin ∈ tokenOriginsAtColumn tokens col) :
    origin ∈ arrivingOriginsAt d depth col := by
  unfold tokenOriginsAtColumn at hmem
  rw [List.mem_map] at hmem
  obtain ⟨t, htfilter, horigin⟩ := hmem
  rw [List.mem_filter] at htfilter
  have harr := routeAligned_token_mem_arrivingOriginsAt halign htfilter.1
  rw [← horigin]
  have hcur : t.current_column = col := by
    exact of_decide_eq_true htfilter.2
  simpa [hcur] using harr

lemma arrivingOriginsAt_subset_tokenOriginsAtColumn {n : Nat}
    {d : Graph} {depth col origin : Nat} {tokens : List (Token n)}
    (halign : TokensRouteAlignedAtDepth d depth tokens)
    (hpres : TokensPresenceAtDepth d depth tokens)
    (hmem : origin ∈ arrivingOriginsAt d depth col) :
    origin ∈ tokenOriginsAtColumn tokens col := by
  obtain ⟨source, label, t, ht, ht_origin, hcur, _hsrc, _hlbl, _hstate, _huniq⟩ :=
    presence_aligned_token_of_arrivingOriginsAt
      halign hpres hmem
  unfold tokenOriginsAtColumn
  rw [List.mem_map]
  refine ⟨t, ?_, ?_⟩
  · rw [List.mem_filter]
    exact ⟨ht, by simpa using (decide_eq_true hcur)⟩
  · exact ht_origin

/-- Exact layer invariant: old well-routed/coherent invariant plus exact
    presence of the live origin tokens. -/
def TokensExactAtLayer {n : Nat}
    (d : Graph) (depth : Nat) (layer : GridLayer n) (tokens : List (Token n)) : Prop :=
  TokensWellRoutedAtLayer d depth layer tokens ∧
  TokensPresenceAtDepth d depth tokens

lemma TokensExactAtLayer.coherent {n : Nat}
    {d : Graph} {depth : Nat} {layer : GridLayer n} {tokens : List (Token n)}
    (h : TokensExactAtLayer d depth layer tokens) :
    TokensCoherentAtLayer layer tokens :=
  h.1.coherent

/-- Recursive strengthened invariant along the evaluator descent. -/
def TokensWellRoutedAlong {n : Nat} (d : Graph) (num_levels : Nat) :
    Nat → Nat → List (Token n) → List (GridLayer n) → Prop
  | _, _, _, [] => True
  | depth, level, tokens, (layer :: rest) =>
      TokensWellRoutedAtLayer d depth layer tokens ∧
      TokensWellRoutedAlong d num_levels (depth + 1) (level - 1)
        (propagate_tokens tokens (pathsFromDLDS d) level num_levels
          (evaluate_layer layer tokens).1)
        rest

/-- Recursive exact invariant along the evaluator descent: every layer is
    well-routed/coherent and has exactly the live origin tokens. -/
def TokensExactAlong {n : Nat} (d : Graph) (num_levels : Nat) :
    Nat → Nat → List (Token n) → List (GridLayer n) → Prop
  | _, _, _, [] => True
  | depth, level, tokens, (layer :: rest) =>
      TokensExactAtLayer d depth layer tokens ∧
      TokensExactAlong d num_levels (depth + 1) (level - 1)
        (propagate_tokens tokens (pathsFromDLDS d) level num_levels
          (evaluate_layer layer tokens).1)
        rest

def tokensMatchOneRuleB {n : Nat}
    (node : CircuitNode n) (node_incoming : NodeIncoming)
    (tokens : List (Token n)) : Bool :=
  match tokens with
  | [] => true
  | t :: ts =>
      match decodeInputLabel node_incoming t.input_label with
      | none => false
      | some (r, _slot, _src) =>
          let arity := (node_incoming[r]?.getD default).length
          decide (r < node.rules.length) &&
          (t :: ts).all (fun s =>
            match decodeInputLabel node_incoming s.input_label with
            | some (r', slot', src') =>
                decide (r' = r ∧ slot' < arity ∧ s.source_column = src')
            | none => false) &&
          decide ((t :: ts).length = arity) &&
          (List.range arity).all (fun i =>
            (t :: ts).any (fun s =>
              match decodeInputLabel node_incoming s.input_label with
              | some (r', slot', src') =>
                  decide (r' = r ∧ slot' = i ∧ s.source_column = src')
              | none => false))

def tokensCoherentAtLayerB {n : Nat}
    (layer : GridLayer n) (tokens : List (Token n)) : Bool :=
  (List.range layer.nodes.length).all (fun col =>
    match layer.nodes[col]?, layer.incoming[col]? with
    | some cnode, some incoming =>
        tokensMatchOneRuleB cnode incoming
          (tokens.filter (fun t => t.current_column = col))
    | _, _ => false)

def tokenWellRoutedAtDepthB {n : Nat} (d : Graph) (depth : Nat) (t : Token n) : Bool :=
  decide (t.origin_column < (buildFormulas d).length) &&
  match routeStateAfter (pathsFromDLDS d) t.origin_column depth with
  | some (current, source, label) =>
      decide (t.current_column = current ∧
        t.source_column = source ∧
        t.input_label = label)
  | none => false

def tokensWellRoutedAtLayerB {n : Nat}
    (d : Graph) (depth : Nat) (layer : GridLayer n) (tokens : List (Token n)) : Bool :=
  tokens.all (tokenWellRoutedAtDepthB d depth) &&
  tokensCoherentAtLayerB layer tokens

def originCount {n : Nat} (origin : Nat) (tokens : List (Token n)) : Nat :=
  (tokens.filter (fun t => t.origin_column = origin)).length

def tokensPresenceAtDepthB {n : Nat}
    (d : Graph) (depth : Nat) (tokens : List (Token n)) : Bool :=
  (List.range (buildFormulas d).length).all (fun origin =>
    match routeStateAfter (pathsFromDLDS d) origin depth with
    | none => decide (originCount origin tokens = 0)
    | some _ => decide (originCount origin tokens = 1))

def tokensExactAtLayerB {n : Nat}
    (d : Graph) (depth : Nat) (layer : GridLayer n) (tokens : List (Token n)) : Bool :=
  tokensWellRoutedAtLayerB d depth layer tokens &&
  tokensPresenceAtDepthB d depth tokens

def tokenTraceAux {n : Nat}
    (paths : PathInput) (level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (num_levels : Nat) : List (List (Token n)) :=
  match layers with
  | [] => []
  | layer :: rest =>
      tokens ::
        match rest with
        | [] => []
        | _ :: _ =>
            let outs := (evaluate_layer layer tokens).1
            tokenTraceAux paths (level - 1)
              (propagate_tokens tokens paths level num_levels outs)
              rest num_levels

def tokenTraceDLDS (d : Graph) : List (List (Token (buildFormulas d).length)) :=
  let layers := buildGridFromDLDS d
  let num_levels := layers.length
  tokenTraceAux (pathsFromDLDS d) (num_levels - 1)
    (initialize_tokens (initialVectorsFromDLDS d) num_levels)
    layers num_levels

/-- `TokensCoherentAtLayer` implies `nodeError = false` at every column.
    Unpacks `TokensMatchOneRule`'s existentials and applies
    `nodeError_false_of_exact_rule_slots`. -/
lemma coherentLayer_implies_noError {n : Nat}
    (layer : GridLayer n) (tokens : List (Token n))
    (hcoh : TokensCoherentAtLayer layer tokens) :
    ∀ col (hNode : col < layer.nodes.length) (hIncoming : col < layer.incoming.length),
      nodeError
        (layer.nodes.get ⟨col, hNode⟩)
        (layer.incoming.get ⟨col, hIncoming⟩)
        (tokens.filter (fun t => t.current_column = col)) = false := by
  intro col hNode hIncoming
  have hmatch := hcoh col hNode hIncoming
  -- Introduce a helper that works on a generic list variable, avoiding the
  -- `generalize`-with-wrong-predicate pitfall; then instantiate at the filter.
  suffices h : ∀ (toks : List (Token n)),
      toks = tokens.filter (fun t => decide (t.current_column = col)) →
      TokensMatchOneRule (layer.nodes.get ⟨col, hNode⟩)
        (layer.incoming.get ⟨col, hIncoming⟩) toks →
      nodeError (layer.nodes.get ⟨col, hNode⟩)
        (layer.incoming.get ⟨col, hIncoming⟩) toks = false from
    h _ rfl hmatch
  intro toks _ htmatch
  cases toks with
  | nil => rfl
  | cons t ts =>
      obtain ⟨r, _s, _src, hhead, hrule, hlabels, hcount, hcomplete⟩ := htmatch
      exact nodeError_false_of_exact_rule_slots _ _ _ _ _ _ _ hhead hrule hlabels hcount hcomplete

def initTokenAt {n : Nat} (vecs : List (List.Vector Bool n))
    (tl col : Nat) (h : col < vecs.length) : Token n :=
  { origin_column := col
    source_column := col
    current_level := tl
    current_column := col
    dep_vector := vecs.get ⟨col, h⟩
    input_label := 0 }

def initTokensFrom {n : Nat} (vecs : List (List.Vector Bool n))
    (tl start : Nat) : List (Token n) :=
  vecs.zipIdx start |>.map fun (vec, col) =>
    { origin_column := col
      source_column := col
      current_level := tl
      current_column := col
      dep_vector := vec
      input_label := 0 }

lemma init_token_at_col {n : Nat} (vecs : List (List.Vector Bool n))
    (tl : Nat) (t : Token n) :
    t ∈ initialize_tokens vecs tl ↔
      ∃ col, ∃ h : col < vecs.length, t = initTokenAt vecs tl col h := by
  constructor
  · intro ht
    unfold initialize_tokens at ht
    rw [List.mem_map] at ht
    obtain ⟨⟨vec, col⟩, hzip, ht⟩ := ht
    have hz := List.mem_zipIdx hzip
    rcases hz with ⟨_, hlt, hvec⟩
    have hcol : col < vecs.length := by simpa using hlt
    refine ⟨col, hcol, ?_⟩
    subst ht
    have hvec' : vec = vecs.get ⟨col, hcol⟩ := by
      rw [List.get_eq_getElem]
      simpa [Nat.sub_zero] using hvec
    subst hvec'
    rfl
  · intro ht
    rcases ht with ⟨col, hcol, rfl⟩
    unfold initialize_tokens
    rw [List.mem_map]
    refine ⟨(vecs.get ⟨col, hcol⟩, col), ?_, rfl⟩
    rw [List.mem_iff_get]
    refine ⟨⟨col, by simpa [List.length_zipIdx] using hcol⟩, ?_⟩
    rw [List.get_eq_getElem]
    simp [List.getElem_zipIdx]

lemma initTokensFrom_filter_eq {n : Nat}
    (vecs : List (List.Vector Bool n)) (tl start col : Nat) :
    (initTokensFrom vecs tl start).filter (fun t => t.current_column = col) =
      if _h : start ≤ col ∧ col < start + vecs.length then
        [{ origin_column := col
           source_column := col
           current_level := tl
           current_column := col
           dep_vector := (vecs[col - start]?.getD default)
           input_label := 0 }]
      else [] := by
  induction vecs generalizing start col with
  | nil =>
      simp [initTokensFrom]
  | cons vec vecs ih =>
      unfold initTokensFrom
      simp only [List.zipIdx_cons, List.map_cons, List.filter_cons]
      by_cases hcol : col = start
      · subst col
        have hhead : decide (start = start) = true := by simp
        rw [hhead]
        have htail_false : ¬ (start + 1 ≤ start ∧ start < start + 1 + vecs.length) := by
          omega
        change
          { origin_column := start, source_column := start, current_level := tl,
            current_column := start, dep_vector := vec, input_label := 0 } ::
            (initTokensFrom vecs tl (start + 1)).filter
              (fun t => decide (t.current_column = start)) =
            if h : start ≤ start ∧ start < start + (vec :: vecs).length then
              [{ origin_column := start, source_column := start, current_level := tl,
                 current_column := start, dep_vector := ((vec :: vecs)[start - start]?.getD default),
                 input_label := 0 }]
            else []
        rw [ih (start + 1) start]
        simp
      · have hhead : decide (start = col) = false := by
          simp [show ¬ start = col by intro hs; exact hcol hs.symm]
        rw [hhead]
        change
          (initTokensFrom vecs tl (start + 1)).filter
              (fun t => decide (t.current_column = col)) =
            if h : start ≤ col ∧ col < start + (vec :: vecs).length then
              [{ origin_column := col, source_column := col, current_level := tl,
                 current_column := col, dep_vector := ((vec :: vecs)[col - start]?.getD default),
                 input_label := 0 }]
            else []
        rw [ih (start + 1) col]
        by_cases htail : start + 1 ≤ col ∧ col < start + 1 + vecs.length
        · have hwhole : start ≤ col ∧ col < start + (vec :: vecs).length := by
            simp
            omega
          have hidx :
              (vecs[col - (start + 1)]?.getD default) =
                ((vec :: vecs)[col - start]?.getD default) := by
            have hsub : col - start = Nat.succ (col - (start + 1)) := by omega
            rw [hsub]
            rfl
          have hlt : col < start + (vecs.length + 1) := by omega
          simp [htail, hwhole, hidx, hlt]
        · have htail' : ¬ (start < col ∧ col < start + 1 + vecs.length) := by
            omega
          have hwhole' : ¬ (start ≤ col ∧ col < start + (vecs.length + 1)) := by
            omega
          simp [htail', hwhole']

lemma init_tokens_filter_singleton {n : Nat}
    (vecs : List (List.Vector Bool n)) (tl col : Nat) :
    (initialize_tokens vecs tl).filter (fun t => t.current_column = col) =
      if h : col < vecs.length then [initTokenAt vecs tl col h] else [] := by
  have hinit : initialize_tokens vecs tl = initTokensFrom vecs tl 0 := by
    rfl
  rw [hinit, initTokensFrom_filter_eq]
  by_cases hcol : col < vecs.length
  · have hcond : (0 ≤ col ∧ col < 0 + vecs.length) := by omega
    simp [hcol, hcond, initTokenAt]
  · have hcond : ¬ (0 ≤ col ∧ col < 0 + vecs.length) := by omega
    simp [hcol]

lemma buildIncomingMapForFormula_last_rep_getD
    (formulas : List Formula) (formula : Formula) :
    ((buildIncomingMapForFormula formulas formula)[
        (buildIncomingMapForFormula formulas formula).length - 1]?.getD default) =
      [(formulas.idxOf formula, 0)] := by
  cases formula with
  | atom name =>
      simp [buildIncomingMapForFormula]
  | implication A B =>
      simp [buildIncomingMapForFormula]

lemma decodeInputLabel_zero_buildIncomingMapForFormula
    (formulas : List Formula) (formula : Formula) :
    decodeInputLabel (buildIncomingMapForFormula formulas formula) 0 =
      some ((buildIncomingMapForFormula formulas formula).length - 1, 0,
        formulas.idxOf formula) := by
  have hlast := buildIncomingMapForFormula_last_rep_getD formulas formula
  have hlen : 0 < (buildIncomingMapForFormula formulas formula).length := by
    unfold buildIncomingMapForFormula
    cases formula <;> simp
  unfold decodeInputLabel
  rw [if_pos hlen]
  simp only
  rw [List.getElem!_eq_getElem?_getD]
  rw [hlast]

lemma decodedRuleAtColumn?_zero
    (d : Graph) {col : Nat} {φ : Formula}
    (hφ : (buildFormulas d)[col]? = some φ) :
    decodedRuleAtColumn? d 0 col =
      some ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) := by
  have hcol : col < (buildFormulas d).length := getElem?_some_lt hφ
  unfold decodedRuleAtColumn?
  have hfind :
      (List.range (buildFormulas d).length).find? (fun origin =>
        match routeStateAfter (pathsFromDLDS d) origin 0 with
        | some (current, _, _) => current == col
        | none => false) = some col := by
    have hcongr :
        (List.range (buildFormulas d).length).find? (fun origin =>
          match routeStateAfter (pathsFromDLDS d) origin 0 with
          | some (current, _, _) => current == col
          | none => false) =
        (List.range (buildFormulas d).length).find? (fun origin => origin == col) := by
      apply find?_congr_on
      intro origin horigin
      have horigin_lt : origin < (buildFormulas d).length := List.mem_range.mp horigin
      have hpath : origin < (pathsFromDLDS d).length := by
        simpa [pathsFromDLDS] using horigin_lt
      simp [routeStateAfter, hpath]
    rw [hcongr]
    exact find?_range_beq_eq (buildFormulas d).length col hcol
  have hstate :
      routeStateAfter (pathsFromDLDS d) col 0 = some (col, col, 0) := by
    have hpath : col < (pathsFromDLDS d).length := by
      simpa [pathsFromDLDS] using hcol
    simp [routeStateAfter, hpath]
  simp [hfind, hstate, hφ, decodeInputLabel_zero_buildIncomingMapForFormula]

lemma placementCoherentAtColumnB_zero
    (d : Graph) {col : Nat}
    (hcol : col < (buildFormulas d).length) :
    placementCoherentAtColumnB d 0 col = true := by
  let φ := (buildFormulas d).get ⟨col, hcol⟩
  have hφ : (buildFormulas d)[col]? = some φ := by
    simp [φ, List.get_eq_getElem]
  have harr : arrivingOriginsAt d 0 col = [col] :=
    arrivingOriginsAt_zero_eq_singleton d hcol
  have hdec :
      decodedRuleAtColumn? d 0 col =
        some ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) :=
    decodedRuleAtColumn?_zero d hφ
  have hexp :
      expectedCarrierOriginsForRule? d col
          ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) =
        some [col] :=
    expectedCarrierOriginsForRule?_rep d hφ
  exact placementCoherentAtColumnB_of_expected_carriers d 0 col
    ((buildIncomingMapForFormula (buildFormulas d) φ).length - 1) [col]
    (by simp [harr]) hdec hexp (by simp [harr, sortNats])

lemma placementCoherentAtDepth_zero (d : Graph) :
    ∀ col, col < (buildFormulas d).length →
      placementCoherentAtColumnB d 0 col = true := by
  intro col hcol
  exact placementCoherentAtColumnB_zero d hcol

lemma buildIncomingMapForFormula_last_rep_length
    (formulas : List Formula) (formula : Formula) :
    (((buildIncomingMapForFormula formulas formula)[
      (buildIncomingMapForFormula formulas formula).length - 1]?.getD default).length = 1) := by
  have hlast := buildIncomingMapForFormula_last_rep_getD formulas formula
  simpa using congrArg List.length hlast

/-- Exact label/source form for the selected DLDS rule: if slot `slot` of the
    selected incoming-map entry names source column `idxOf φ`, and
    `slotForEdge` chooses that slot, then the encoded path label decodes back to
    precisely `(ruleIdx, slot, idxOf φ)`. -/
lemma inputLabelForEdge_decodes_of_rule_slot_source
    (d : Graph) (formulas : List Formula) (φ : Formula) (w : Vertex)
    (ruleIdx edgeId : Nat)
    (hsel : ruleIndexForNode? d formulas w = some ruleIdx)
    (hsrc :
      (((buildIncomingMapForFormula formulas w.FORMULA)[ruleIdx]?.getD default)[slotForEdge φ w d]? =
        some (formulas.idxOf φ, edgeId))) :
    decodeInputLabel
      (buildIncomingMapForFormula formulas w.FORMULA)
      (inputLabelForEdge d formulas φ w) =
        some (ruleIdx, slotForEdge φ w d, formulas.idxOf φ) := by
  let incoming := buildIncomingMapForFormula formulas w.FORMULA
  let slot := slotForEdge φ w d
  have hslotRange : slot < (incoming[ruleIdx]?.getD default).length :=
    getElem?_some_lt hsrc
  have hidxIncoming : ruleIdx < incoming.length :=
    getElem?_getD_length_pos_lt incoming ruleIdx (by
      dsimp [incoming] at hslotRange ⊢
      omega)
  unfold inputLabelForEdge
  rw [hsel]
  by_cases hlast : ruleIdx + 1 = incoming.length
  · have hidxLast : ruleIdx = incoming.length - 1 := by omega
    have hentryLen : (incoming.get ⟨ruleIdx, hidxIncoming⟩).length = 1 := by
      have hlastLen := buildIncomingMapForFormula_last_rep_length formulas w.FORMULA
      have hlastLen' : ((incoming[ruleIdx]?.getD default).length = 1) := by
        dsimp [incoming] at hlastLen ⊢
        simpa [hidxLast] using hlastLen
      rw [List.getElem?_eq_getElem hidxIncoming, Option.getD_some] at hlastLen'
      exact hlastLen'
    have hslotRange' : slot < (incoming.get ⟨ruleIdx, hidxIncoming⟩).length := by
      dsimp [incoming] at hslotRange ⊢
      rw [List.getElem?_eq_getElem hidxIncoming, Option.getD_some] at hslotRange
      exact hslotRange
    have hslot0 : slot = 0 := by
      rw [hentryLen] at hslotRange'
      omega
    subst slot
    have hsrc' :
        (incoming.get ⟨ruleIdx, hidxIncoming⟩)[0]? =
          some (formulas.idxOf φ, edgeId) := by
      dsimp [incoming] at hsrc ⊢
      rw [hslot0] at hsrc
      rw [List.getElem?_eq_getElem hidxIncoming, Option.getD_some] at hsrc
      exact hsrc
    have hhead : 0 < (incoming.get ⟨ruleIdx, hidxIncoming⟩).length := by
      rw [hentryLen]
      omega
    have hpair :
        (incoming.get ⟨ruleIdx, hidxIncoming⟩).get ⟨0, hhead⟩ =
          (formulas.idxOf φ, edgeId) := by
      exact (List.getElem?_eq_some_iff.mp hsrc').2
    have hsrcFst :
        ((incoming.get ⟨ruleIdx, hidxIncoming⟩).get ⟨0, hhead⟩).1 =
          formulas.idxOf φ := by
      rw [hpair]
    have hdec :=
      inputLabelForRuleSlot_decode_roundtrip_rep incoming ruleIdx
        hlast hidxIncoming hhead
    rw [hslot0]
    change
      decodeInputLabel incoming (inputLabelForRuleSlot incoming ruleIdx 0) =
        some (ruleIdx, 0, formulas.idxOf φ)
    rw [hsrcFst] at hdec
    exact hdec
  · have hnonrep : ruleIdx + 1 < incoming.length := by omega
    have hidxNonrep : ruleIdx < incoming.length := Nat.lt_of_succ_lt hnonrep
    have hsrc' :
        (incoming.get ⟨ruleIdx, hidxNonrep⟩)[slot]? =
          some (formulas.idxOf φ, edgeId) := by
      dsimp [incoming] at hsrc ⊢
      rw [List.getElem?_eq_getElem hidxNonrep, Option.getD_some] at hsrc
      exact hsrc
    have hslotGet : slot < (incoming.get ⟨ruleIdx, hidxNonrep⟩).length :=
      getElem?_some_lt hsrc'
    have hpair :
        (incoming.get ⟨ruleIdx, hidxNonrep⟩).get ⟨slot, hslotGet⟩ =
          (formulas.idxOf φ, edgeId) := by
      exact (List.getElem?_eq_some_iff.mp hsrc').2
    have hsrcFst :
        ((incoming.get ⟨ruleIdx, hidxNonrep⟩).get ⟨slot, hslotGet⟩).1 =
          formulas.idxOf φ := by
      rw [hpair]
    have hdec :=
      inputLabelForRuleSlot_decode_roundtrip_nonrep incoming ruleIdx slot
        hnonrep hslotGet
    change
      decodeInputLabel incoming (inputLabelForRuleSlot incoming ruleIdx slot) =
        some (ruleIdx, slot, formulas.idxOf φ)
    rw [hsrcFst] at hdec
    exact hdec

/-- If the graph/grid bridge says that selected rule `ruleIdx` has source list
    `srcs`, then every source appearing in `srcs` is backed by an actual pair in
    the selected incoming-map entry. -/
lemma selectedIncoming_slot_source_pair
    (d : Graph) (formulas : List Formula) (w : Vertex)
    (ruleIdx slot src : Nat)
    (_hsel : ruleIndexForNode? d formulas w = some ruleIdx)
    (hbridge :
      incomingRuleSourceColumns? formulas w.FORMULA ruleIdx =
        classifiedRuleSourceColumns? d formulas w)
    {srcs : List Nat}
    (hsrcs : classifiedRuleSourceColumns? d formulas w = some srcs)
    (hslot : srcs[slot]? = some src) :
    ∃ edgeId,
      (((buildIncomingMapForFormula formulas w.FORMULA)[ruleIdx]?.getD default)[slot]? =
        some (src, edgeId)) := by
  unfold incomingRuleSourceColumns? at hbridge
  rw [hsrcs] at hbridge
  cases hentry : (buildIncomingMapForFormula formulas w.FORMULA)[ruleIdx]? with
  | none =>
      rw [hentry] at hbridge
      simp at hbridge
  | some inc =>
      rw [hentry] at hbridge
      simp only [Option.map_some, Option.some.injEq] at hbridge
      have hslotMap : (inc.map Prod.fst)[slot]? = some src := by
        rw [hbridge]
        exact hslot
      obtain ⟨edgeId, hpair⟩ := getElem?_map_fst_eq_some hslotMap
      refine ⟨edgeId, ?_⟩
      simpa using hpair

/-- Combined selected-slot decode lemma: if a classified slot for `w` names the
    source column of formula `φ`, and `slotForEdge` chooses that slot, then the
    path label for the edge into `w` decodes to the selected rule, that slot, and
    that source. -/
lemma inputLabelForEdge_decodes_of_classified_slot
    (d : Graph) (formulas : List Formula) (φ : Formula) (w : Vertex)
    (ruleIdx slot : Nat)
    {srcs : List Nat}
    (hsel : ruleIndexForNode? d formulas w = some ruleIdx)
    (hbridge :
      incomingRuleSourceColumns? formulas w.FORMULA ruleIdx =
        classifiedRuleSourceColumns? d formulas w)
    (hsrcs : classifiedRuleSourceColumns? d formulas w = some srcs)
    (hslot : srcs[slot]? = some (formulas.idxOf φ))
    (hslotFor : slotForEdge φ w d = slot) :
    decodeInputLabel
      (buildIncomingMapForFormula formulas w.FORMULA)
      (inputLabelForEdge d formulas φ w) =
        some (ruleIdx, slot, formulas.idxOf φ) := by
  obtain ⟨edgeId, hpair⟩ :=
    selectedIncoming_slot_source_pair d formulas w ruleIdx slot
      (formulas.idxOf φ) hsel hbridge hsrcs hslot
  have hpairFor :
      (((buildIncomingMapForFormula formulas w.FORMULA)[ruleIdx]?.getD default)[slotForEdge φ w d]? =
        some (formulas.idxOf φ, edgeId)) := by
    simpa [hslotFor] using hpair
  have hdec :=
    inputLabelForEdge_decodes_of_rule_slot_source d formulas φ w ruleIdx edgeId
      hsel hpairFor
  simpa [hslotFor] using hdec

lemma buildIncomingMapForFormula_length_pos
    (formulas : List Formula) (formula : Formula) :
    0 < (buildIncomingMapForFormula formulas formula).length := by
  unfold buildIncomingMapForFormula
  cases formula <;> simp



#print axioms dlds_global_iff
end Semantic
