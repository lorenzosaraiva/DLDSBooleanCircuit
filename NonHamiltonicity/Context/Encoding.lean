import NonHamiltonicity.Context.Soundness
import Init.Data.Nat.ToString

/-! Checked data bridge for the independently proposed
`hermann-positional-context/1.0`. Source graphs here are combinatorial graphs,
never the global ND/DLDS `Graph`. Schema witnesses are untrusted data. -/
namespace NonHamiltonicity.Context
open DLDS ProofSearch.Grid

def decimalValue (digits : List Char) : Nat :=
  digits.foldl (fun n c => 10 * n + (c.toNat - '0'.toNat)) 0

theorem decimalValue_append_digit (digits : List Char) (c : Char) :
    decimalValue (digits ++ [c]) = 10 * decimalValue digits + (c.toNat - '0'.toNat) := by
  simp [decimalValue, List.foldl_append]

theorem digitChar_value (n : Nat) (bound : n < 10) :
    (Nat.digitChar n).toNat - '0'.toNat = n := by
  interval_cases n <;> decide

theorem decimalValue_toDigits (n : Nat) : decimalValue (Nat.toDigits 10 n) = n := by
  induction n using Nat.strong_induction_on with
  | h n ih =>
      rw [Nat.toDigits_eq_if (by omega)]
      split
      · rename_i small
        simpa only [decimalValue, List.foldl_cons, List.foldl_nil, Nat.mul_zero, Nat.zero_add]
          using digitChar_value n small
      · rename_i large
        rw [decimalValue_append_digit, ih (n / 10) (Nat.div_lt_self (by omega) (by omega)),
          digitChar_value (n % 10) (Nat.mod_lt _ (by omega))]
        omega

def splitDigits : List Char → List Char × List Char
  | [] => ([], [])
  | c :: cs => if c.isDigit then
      let (digits, rest) := splitDigits cs
      (c :: digits, rest)
    else ([], c :: cs)

theorem splitDigits_append (digits : List Char) (rest : List Char)
    (allDigits : ∀ c ∈ digits, c.isDigit = true)
    (stop : splitDigits rest = ([], rest)) :
    splitDigits (digits ++ rest) = (digits, rest) := by
  induction digits with
  | nil => exact stop
  | cons c cs ih =>
      simp only [List.cons_append, splitDigits, allDigits c (by simp), ite_true]
      rw [ih (fun d mem => allDigits d (by simp [mem]))]

def encodeTag : Tag → String
  | .result => "q"
  | .position i v => "X" ++ toString i ++ "v" ++ toString v
  | .cases k => "ORX" ++ toString k

def decodeChars : List Char → Option Tag
  | ['q'] => some .result
  | 'X' :: cs =>
      let (digits, rest) := splitDigits cs
      match rest with
      | 'v' :: vs => some (.position (decimalValue digits) (decimalValue vs))
      | _ => none
  | 'O' :: 'R' :: 'X' :: ds => some (.cases (decimalValue ds))
  | _ => none

/- The re-encoding guard rejects leading zeros, alternate spellings and unknown
names. Zero tags exist in the abstract Names domain, but schema bounds reject
them from every admissible formula. -/
def decodeTag (s : String) : Option Tag := do
  let tag ← decodeChars s.toList
  if encodeTag tag = s then some tag else none

theorem decodeChars_encodeTag (tag : Tag) :
    decodeChars (encodeTag tag).toList = some tag := by
  cases tag with
  | result => rfl
  | cases k =>
      simp [encodeTag, String.toList_append, Nat.repr_eq_ofList_toDigits,
        decodeChars, decimalValue_toDigits]
  | position i v =>
      have split := splitDigits_append (Nat.toDigits 10 i) ('v' :: Nat.toDigits 10 v)
        (fun c mem => Nat.isDigit_of_mem_toDigits (by omega) (by omega) mem) (by rfl)
      simp [encodeTag, String.toList_append, Nat.repr_eq_ofList_toDigits,
        decodeChars, List.append_assoc, split, decimalValue_toDigits]

theorem decodeTag_encodeTag (tag : Tag) : decodeTag (encodeTag tag) = some tag := by
  simp [decodeTag, decodeChars_encodeTag]

theorem encodeTag_injective : Function.Injective encodeTag := by
  intro a b same
  have := congrArg decodeTag same
  simpa only [decodeTag_encodeTag, Option.some.injEq] using this

theorem position_names_injective (i v j u : Nat)
    (same : encodeTag (.position i v) = encodeTag (.position j u)) : i = j ∧ v = u := by
  simpa only [NonHamiltonicity.Context.Tag.position.injEq] using encodeTag_injective same

theorem case_names_injective (k l : Nat)
    (same : encodeTag (.cases k) = encodeTag (.cases l)) : k = l := by
  simpa only [NonHamiltonicity.Context.Tag.cases.injEq] using encodeTag_injective same

theorem result_position_disjoint (i v : Nat) : encodeTag .result ≠ encodeTag (.position i v) := by
  intro same
  cases encodeTag_injective same

theorem result_case_disjoint (k : Nat) : encodeTag .result ≠ encodeTag (.cases k) := by
  intro same
  cases encodeTag_injective same

theorem position_case_disjoint (i v k : Nat) : encodeTag (.position i v) ≠ encodeTag (.cases k) := by
  intro same
  cases encodeTag_injective same

def canonicalNames : Names := ⟨encodeTag, encodeTag_injective⟩

structure SourceGraphData where
  n : Nat
  vertices : List Nat
  edges : List (Nat × Nat)
  undirected : Bool
  deriving DecidableEq, Repr

def normalizedEdge (e : Nat × Nat) : Nat × Nat := (min e.1 e.2, max e.1 e.2)

def SourceGraphValid (data : SourceGraphData) : Prop :=
  3 ≤ data.n ∧ data.vertices = List.range' 1 data.n ∧ data.undirected = true ∧
  (∀ e ∈ data.edges, 1 ≤ e.1 ∧ e.1 ≤ data.n ∧ 1 ≤ e.2 ∧ e.2 ≤ data.n ∧ e.1 ≠ e.2) ∧
  (data.edges.map normalizedEdge).Nodup

instance (data : SourceGraphData) : Decidable (SourceGraphValid data) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

def graphCheckB (data : SourceGraphData) : Bool := decide (SourceGraphValid data)

theorem graphCheckB_sound (data : SourceGraphData) (checked : graphCheckB data = true) :
    SourceGraphValid data := of_decide_eq_true checked

def sourceEdge (data : SourceGraphData) (u v : Nat) : Prop :=
  (u, v) ∈ data.edges ∨ (v, u) ∈ data.edges

instance (data : SourceGraphData) (u v : Nat) : Decidable (sourceEdge data u v) :=
  inferInstanceAs (Decidable (_ ∨ _))

def formalGraph (data : SourceGraphData) (valid : graphCheckB data = true) : Graph where
  n := data.n
  atLeastThree := (of_decide_eq_true valid).1
  edge := sourceEdge data
  symmetric := by
    intro u v h
    exact h.elim Or.inr Or.inl
  irreflexive := by
    intro v h
    have member : (v, v) ∈ data.edges := h.elim id id
    exact ((of_decide_eq_true valid).2.2.2.1 (v, v) member).2.2.2.2 rfl

theorem formalGraph_edge_iff (data : SourceGraphData) (valid : graphCheckB data = true)
    (u v : Nat) : (formalGraph data valid).edge u v ↔
      (u, v) ∈ data.edges ∨ (v, u) ∈ data.edges := Iff.rfl

theorem formalGraph_edge_bounds (data : SourceGraphData) (valid : graphCheckB data = true)
    (u v : Nat) (edge : (formalGraph data valid).edge u v) :
    1 ≤ u ∧ u ≤ data.n ∧ 1 ≤ v ∧ v ≤ data.n := by
  have bounds := (graphCheckB_sound data valid).2.2.2.1
  rcases edge with forward | backward
  · exact ⟨(bounds (u,v) forward).1, (bounds (u,v) forward).2.1,
      (bounds (u,v) forward).2.2.1, (bounds (u,v) forward).2.2.2.1⟩
  · exact ⟨(bounds (v,u) backward).2.2.1, (bounds (v,u) backward).2.2.2.1,
      (bounds (v,u) backward).1, (bounds (v,u) backward).2.1⟩

/- Witnesses name exactly the original schemas, including both nonedge orders.
Full case alternatives are checked by List.Perm, without enumerating permutations. -/
inductive SchemaWitness where
  | totality (k : Nat)
  | cases (k : Nat) (vertices : List Nat)
  | repeatVertex (i j v : Nat)
  | nonedge (i u v : Nat) (flipped : Bool)
  | closingAlias (v : Nat)
  | invalid
  deriving DecidableEq, Repr

def checkedFormula (data : SourceGraphData) : SchemaWitness → Formula
  | .totality k => orx canonicalNames k
  | .cases k vs => casesFormula canonicalNames k vs
  | .repeatVertex i j v => x canonicalNames i v >> (x canonicalNames j v >> q canonicalNames)
  | .nonedge i u v false => x canonicalNames i u >> (x canonicalNames (i+1) v >> q canonicalNames)
  | .nonedge i u v true => x canonicalNames (i+1) v >> (x canonicalNames i u >> q canonicalNames)
  | .closingAlias v => x canonicalNames 1 v >> x canonicalNames (data.n + 1) v
  | .invalid => q canonicalNames

def WitnessValid (data : SourceGraphData) : SchemaWitness → Prop
  | .totality k => 1 ≤ k ∧ k ≤ data.n
  | .cases k vs => 1 ≤ k ∧ k ≤ data.n ∧ vs.Perm (List.range' 1 data.n)
  | .repeatVertex i j v => 1 ≤ i ∧ i ≤ data.n ∧ 1 ≤ j ∧ j ≤ data.n ∧
      1 ≤ v ∧ v ≤ data.n ∧ i ≠ j
  | .nonedge i u v _ => 1 ≤ i ∧ i ≤ data.n ∧ 1 ≤ u ∧ u ≤ data.n ∧
      1 ≤ v ∧ v ≤ data.n ∧ ¬sourceEdge data u v
  | .closingAlias v => 1 ≤ v ∧ v ≤ data.n
  | .invalid => False

instance (data : SourceGraphData) (w : SchemaWitness) : Decidable (WitnessValid data w) := by
  cases w <;> unfold WitnessValid <;> infer_instance

/- Structural AST equality avoids the casts in the shared Formula.decEq during
finite kernel reduction. Its equivalence is proved, not assumed. -/
def formulaEqB : Formula → Formula → Bool
  | .atom a, .atom b => a == b
  | .implication a b, .implication c d => formulaEqB a c && formulaEqB b d
  | _, _ => false

@[simp] theorem formulaEqB_iff (a b : Formula) : formulaEqB a b = true ↔ a = b := by
  induction a generalizing b with
  | atom s => cases b <;> simp [formulaEqB]
  | implication a c ia ic => cases b <;> simp [formulaEqB, ia, ic]

theorem formulaBEq_eq_formulaEqB (a b : Formula) : (a == b) = formulaEqB a b := by
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq, formulaEqB_iff]

def formulaCheckB (data : SourceGraphData) (formula : Formula) (w : SchemaWitness) : Bool :=
  formulaEqB formula (checkedFormula data w) && decide (WitnessValid data w)

theorem formulaCheckB_sound (data : SourceGraphData) (valid : graphCheckB data = true)
    (formula : Formula) (w : SchemaWitness) (checked : formulaCheckB data formula w = true) :
    Admissible (formalGraph data valid) canonicalNames formula := by
  have both := Bool.and_eq_true_iff.mp checked
  have same := (formulaEqB_iff _ _).mp both.1
  have conditions : WitnessValid data w := of_decide_eq_true both.2
  subst formula
  cases w with
  | totality k => exact .totality k conditions.1 conditions.2
  | cases k vs => exact .cases k vs conditions.1 conditions.2.1 conditions.2.2
  | repeatVertex i j v =>
      exact .repevertexAt i j v conditions.1 conditions.2.1 conditions.2.2.1
        conditions.2.2.2.1 conditions.2.2.2.2.1 conditions.2.2.2.2.2.1 conditions.2.2.2.2.2.2
  | nonedge i u v flip =>
      cases flip
      · exact .nonedge i u v conditions.1 conditions.2.1 conditions.2.2.1
          conditions.2.2.2.1 conditions.2.2.2.2.1 conditions.2.2.2.2.2.1 conditions.2.2.2.2.2.2
      · exact .nonedgeFlipped i u v conditions.1 conditions.2.1 conditions.2.2.1
          conditions.2.2.2.1 conditions.2.2.2.2.1 conditions.2.2.2.2.2.1 conditions.2.2.2.2.2.2
  | closingAlias v => exact .closingAlias v conditions.1 conditions.2
  | invalid => exact False.elim conditions

def contextCheckB (data : SourceGraphData) : List Formula → List SchemaWitness → Bool
  | [], [] => true
  | f :: fs, w :: ws => formulaCheckB data f w && contextCheckB data fs ws
  | _, _ => false

theorem contextCheckB_sound (data : SourceGraphData) (valid : graphCheckB data = true)
    (context : List Formula) (witnesses : List SchemaWitness)
    (checked : contextCheckB data context witnesses = true) :
    ContextAdmissible (formalGraph data valid) canonicalNames context := by
  induction context generalizing witnesses with
  | nil => intro f mem; simp at mem
  | cons f fs ih =>
      cases witnesses with
      | nil => simp [contextCheckB] at checked
      | cons w ws =>
          have both : formulaCheckB data f w = true ∧ contextCheckB data fs ws = true := by
            simpa only [contextCheckB, Bool.and_eq_true] using checked
          intro a mem
          rcases List.mem_cons.mp mem with same | later
          · subst a; exact formulaCheckB_sound data valid f w both.1
          · exact ih ws both.2 a later

theorem checked_body_rules_out_cycle (data : SourceGraphData)
    (valid : graphCheckB data = true) (body : NDTree) (witnesses : List SchemaWitness)
    (contextChecked : contextCheckB data body.dependencies witnesses = true)
    (wellFormed : body.wellFormedB = true) (conclusion : body.conclusion = .atom "q") :
    ¬HasHamiltonianCycle (formalGraph data valid) :=
  imported_body_rules_out_cycle (formalGraph data valid) canonicalNames body wellFormed conclusion
    (contextCheckB_sound data valid body.dependencies witnesses contextChecked)

/- These structural negative facts hold for every injective Names instance,
independently of the witness recognizer or either concrete graph. -/
theorem bare_position_not_admissible (g : Graph) (names : Names) (i v : Nat) :
    ¬Admissible g names (x names i v) := by
  have exclude : ∀ f, Admissible g names f → f ≠ x names i v := by
    intro f admitted
    cases admitted with
    | totality k lo hi =>
        intro same
        have tags := names.injective (Formula.atom.inj same)
        cases tags
    | cases k vs lo hi complete => cases vs <;> simp [casesFormula, chain, x]
    | repevertexAt => simp [x]
    | nonedge => simp [x]
    | nonedgeFlipped => simp [x]
    | closingAlias => simp [x]
  intro admitted
  exact exclude _ admitted rfl

theorem same_position_obstruction_not_admissible (g : Graph) (names : Names) (i v : Nat) :
    ¬Admissible g names (x names i v >> (x names i v >> q names)) := by
  have exclude : ∀ f, Admissible g names f → f ≠ (x names i v >> (x names i v >> q names)) := by
    intro f admitted
    cases admitted with
    | totality => simp [orx]
    | cases k vs lo hi complete => cases vs <;> simp [casesFormula, chain, x, orx, q]
    | repevertexAt a b c alo ahi blo bhi clo chi different =>
        intro same
        have outer := Formula.implication.inj same
        have inner := Formula.implication.inj outer.2
        have ea := (NonHamiltonicity.Context.Tag.position.inj (names.injective (Formula.atom.inj outer.1))).1
        have eb := (NonHamiltonicity.Context.Tag.position.inj (names.injective (Formula.atom.inj inner.1))).1
        exact different (ea.trans eb.symm)
    | nonedge a u w alo ahi ulo uhi wlo whi missing =>
        intro same
        have outer := Formula.implication.inj same
        have inner := Formula.implication.inj outer.2
        have ea := (NonHamiltonicity.Context.Tag.position.inj (names.injective (Formula.atom.inj outer.1))).1
        have eb := (NonHamiltonicity.Context.Tag.position.inj (names.injective (Formula.atom.inj inner.1))).1
        omega
    | nonedgeFlipped a u w alo ahi ulo uhi wlo whi missing =>
        intro same
        have outer := Formula.implication.inj same
        have inner := Formula.implication.inj outer.2
        have ea := (NonHamiltonicity.Context.Tag.position.inj (names.injective (Formula.atom.inj outer.1))).1
        have eb := (NonHamiltonicity.Context.Tag.position.inj (names.injective (Formula.atom.inj inner.1))).1
        omega
    | closingAlias => simp [x]
  intro admitted
  exact exclude _ admitted rfl

end NonHamiltonicity.Context
