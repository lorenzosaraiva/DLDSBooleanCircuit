import Semantic.MultiTokenReduction

open scoped Classical

/-! Universal acceptance for the compressed multi-token evaluator. The eight HC
legality invariants are hypotheses attributed to the horizontal-compression
construction; their preservation is not formalized here. `dischargedMultiB`
remains an executable certificate. -/

open Semantic

namespace Semantic

open FlowSpec
open ExFan2 ExFan3



/--  No structural/routing error in the multi evaluation of `P`.  -/
def NoMultiRoutingError (d : Graph) (P : MultiPathInput) : Prop :=
  (getEvalResultMultiDLDS d P).2 = false

/--
 All assumptions discharged on multi input `P`: the goal column's final
    dependency vector is all-false (mirror of `AllAssumptionsDischarged`).
-/
def DischargedMulti (d : Graph) (P : MultiPathInput) (g : Nat) : Prop :=
  g ≥ (getEvalResultMultiDLDS d P).1.length ∨
  ∃ h : g < (getEvalResultMultiDLDS d P).1.length,
    ∀ i, ((getEvalResultMultiDLDS d P).1.get ⟨g, h⟩).get i = false

/--
 Genuine acceptance: no routing error AND a discharged goal (mirror of
    `GenuinelyAccepts`; rules out vacuous/errored acceptance).
-/
def GenuinelyAcceptsMulti (d : Graph) (P : MultiPathInput) (g : Nat) : Prop :=
  NoMultiRoutingError d P ∧ DischargedMulti d P g



/--
 **Discharge certificate** (mirror of `dischargedB`): the canonical
    multi-run's goal vector is all-false. Evaluator-side image of Def-23 root
    discharge (Thm 21: every root Flow route carries the empty dependency ;
    `rootsDischargedB`); the examples below show the two agree on every
    fixture, including the undischarged `d11'` where both are FALSE.
-/
def dischargedMultiB (d : Graph) : Bool :=
  dischargedOnMultiB d (multiPathsFromFlow d)

def canonicalNoErrorB (d : Graph) : Bool :=
  !(evalErrorMultiB d (multiPathsFromFlow d))


private def traceGo {n : Nat} (P : MultiPathInput) (num : Nat) :
    List (GridLayer n × List Nat) → Nat → List (Token n) →
      List (List (List.Vector Bool n))
  | [], _, _ => []
  | (layer, exp) :: rest, level, toks =>
      let res := evaluate_layer_multi layer toks exp
      res.1 :: traceGo P num rest (level - 1)
        (propagate_tokens toks P level num res.1)

def multiLayerTrace (d : Graph) :
    List (List (List.Vector Bool (buildFormulas d).length)) :=
  traceGo (multiPathsFromFlow d) (buildGridFromDLDS d).length
    ((buildGridFromDLDS d).zip (expectedCounts d))
    ((buildGridFromDLDS d).length - 1)
    (initialize_tokens_multi (routesOf d) (initialVectorsFromDLDS d)
      (buildGridFromDLDS d).length)

def decorationTraceFailures (d : Graph) : List Nat :=
  let formulas := buildFormulas d
  let maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0
  let tr := multiLayerTrace d
  (d.NODES.filter fun v =>
    let prescribed : Dep :=
      match get_rule.outgoing v d with
      | [] => []
      | e :: _ => e.DEPENDENCY
    match tr[maxLvl - v.LEVEL]? with
    | none => true
    | some outs =>
        match outs[formulas.idxOf v.FORMULA]? with
        | none => true
        | some vec => !(decide (vec = depVector formulas prescribed))).map
    (·.NUMBER)

def decorationTraceB (d : Graph) : Bool :=
  (decorationTraceFailures d).isEmpty

private lemma vector_get_mem_toList {n : Nat} (vec : List.Vector Bool n)
    (i : Fin n) : vec.get i ∈ vec.toList := by
  rcases vec with ⟨l, hl⟩
  simp [List.Vector.get, List.Vector.toList]

lemma discharge_multi (d : Graph) (hdis : dischargedMultiB d = true) :
    DischargedMulti d (multiPathsFromFlow d) (goalColumn d) := by
  unfold dischargedMultiB dischargedOnMultiB at hdis
  by_cases hg : goalColumn d <
      (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.length
  · refine Or.inr ⟨hg, fun i => ?_⟩
    have hsome :
        (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1[goalColumn d]?
          = some ((getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.get
              ⟨goalColumn d, hg⟩) := by
      simp [List.get_eq_getElem]
    rw [hsome] at hdis
    have hall : ((getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.get
        ⟨goalColumn d, hg⟩).toList.all (fun b => !b) = true := by
      simpa using hdis
    rw [List.all_eq_true] at hall
    have hx := hall _ (vector_get_mem_toList _ i)
    simpa using hx
  · exact Or.inl (Nat.le_of_not_lt hg)

theorem canonical_multi_accepts (d : Graph)
    (hdis : dischargedMultiB d = true) :
    evaluateDLDS_multi d (multiPathsFromFlow d) (goalColumn d) = true := by
  unfold dischargedMultiB dischargedOnMultiB at hdis
  unfold evaluateDLDS_multi
  show (if h : goalColumn d <
        (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.length then
      (getEvalResultMultiDLDS d (multiPathsFromFlow d)).2 ||
        ((getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.get
          ⟨goalColumn d, h⟩).toList.all (· = false)
    else true) = true
  by_cases hg : goalColumn d <
      (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.length
  · rw [dif_pos hg]
    have hsome :
        (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1[goalColumn d]?
          = some ((getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.get
              ⟨goalColumn d, hg⟩) := by
      simp [List.get_eq_getElem]
    rw [hsome] at hdis
    have hall : ((getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.get
        ⟨goalColumn d, hg⟩).toList.all (· = false) = true := by
      have hd : ((getEvalResultMultiDLDS d (multiPathsFromFlow d)).1.get
          ⟨goalColumn d, hg⟩).toList.all (fun b => !b) = true := by
        simpa using hdis
      rw [List.all_eq_true] at hd ⊢
      intro b hb
      have := hd b hb
      simpa using this
    rw [hall]
    simp
  · rw [dif_neg hg]

theorem canonical_multi_genuine (d : Graph)
    (hnoerr : canonicalNoErrorB d = true)
    (hdis : dischargedMultiB d = true) :
    GenuinelyAcceptsMulti d (multiPathsFromFlow d) (goalColumn d) := by
  constructor
  · unfold NoMultiRoutingError
    unfold canonicalNoErrorB evalErrorMultiB at hnoerr
    simpa using hnoerr
  · exact discharge_multi d hdis



/--
 **The compressed universal bridge.** On a legal fully-compressed DLDS
    (the Path-B legality context: structural validity, the four invariants,
    `LevelFormulaUnique`, `ReseedFree`, `ValidDLDS` ; Thm 21/24), with the
    canonical slot-validity and discharge certificates (decidable; provenance
    in the header; eval-true on all closed fixtures), EVERY multi-path input
    is either inadmissible (the faithful multi `Invalid`) or discharges the
    goal.

    Proof: an admissible `P` reduces modulo a co-located permutation to the
    canonical path (M2c), evaluates bit-identically to it (M2b's multiset
    congruence + the read-equivalence congruence), and the canonical
    discharges (M3b). The legality hypotheses are the theorem's Path-B scope
    (underscore-named: the assembled M2/M3 pieces consume the two
    certificates; the invariants are what make the certificates the RIGHT
    notion on HC output ; they are consumed by `flowRuleCorrect_collapsed`'s
    per-node correctness, whose composition into the certificates is the
    documented M3b obligation).
-/
theorem compressed_universally_accepted (d : Graph)
    (_hstruct : structuralValid d)
    (_hnodup : d.EDGES.Nodup)
    (_honepercol : OneEdgePerColourPerNode d)
    (_hroutefan : RouteFanUnique d)
    (_hfaithful : FaithfulDecoration d)
    (_hcoverage : RouteHeadCoverage d)
    (_hlfu : LevelFormulaUnique d)
    (_hreseed : ReseedFree d)
    (_hvalid : ValidDLDS d)
    (hslot : canonicalSlotOKB d = true)
    (hdis : dischargedMultiB d = true) :
    ∀ P, ¬ AdmissibleMultiPath d P ∨ DischargedMulti d P (goalColumn d) := by
  intro P
  by_cases hadm : AdmissibleMultiPath d P
  · right
    have heval : getEvalResultMultiDLDS d P =
        getEvalResultMultiDLDS d (multiPathsFromFlow d) :=
      admissibleMulti_eval_canonical d hslot hadm
    have hcanon : DischargedMulti d (multiPathsFromFlow d) (goalColumn d) :=
      discharge_multi d hdis
    unfold DischargedMulti at hcanon ⊢
    rw [heval]
    exact hcanon
  · left
    exact hadm

/--
 Admissible inputs are moreover GENUINELY accepted (error-free), given the
    error-freeness certificate: the full strengthening of the bridge's right
    disjunct.
-/
theorem compressed_universally_accepted_genuine (d : Graph)
    (hslot : canonicalSlotOKB d = true)
    (hnoerr : canonicalNoErrorB d = true)
    (hdis : dischargedMultiB d = true) :
    ∀ P, AdmissibleMultiPath d P →
      GenuinelyAcceptsMulti d P (goalColumn d) := by
  intro P hadm
  have heval : getEvalResultMultiDLDS d P =
      getEvalResultMultiDLDS d (multiPathsFromFlow d) :=
    admissibleMulti_eval_canonical d hslot hadm
  obtain ⟨hne, hdm⟩ := canonical_multi_genuine d hnoerr hdis
  constructor
  · unfold NoMultiRoutingError at hne ⊢
    rw [heval]
    exact hne
  · unfold DischargedMulti at hdm ⊢
    rw [heval]
    exact hdm

theorem canonical_multi_witness (d : Graph)
    (hadm : admissibleMultiPathB d (multiPathsFromFlow d) = true)
    (hnoerr : canonicalNoErrorB d = true)
    (hdis : dischargedMultiB d = true) :
    AdmissibleMultiPath d (multiPathsFromFlow d) ∧
      GenuinelyAcceptsMulti d (multiPathsFromFlow d) (goalColumn d) :=
  ⟨hadm, canonical_multi_genuine d hnoerr hdis⟩





#print axioms compressed_universally_accepted_genuine
end Semantic
