import ProofSearch.BranchingGrid.Completeness

/-! General schema adequacy, with an explicit injective atom encoding.
This does not prove refinement of the Python JSON/string classifier. -/
namespace NonHamiltonicity.Context
open DLDS ProofSearch.Grid

inductive Tag where
  | result
  | position (i v : Nat)
  | cases (k : Nat)
  deriving DecidableEq

structure Names where
  encode : Tag -> String
  injective : Function.Injective encode

structure Graph where
  n : Nat
  atLeastThree : 3 <= n
  edge : Nat -> Nat -> Prop
  symmetric : forall u v, edge u v -> edge v u
  irreflexive : forall v, Not (edge v v)

structure Cycle (g : Graph) where
  vertexAt : Nat -> Nat
  bounds : forall i, 1 <= i -> i <= g.n -> 1 <= vertexAt i /\ vertexAt i <= g.n
  distinct : forall i j, 1 <= i -> i <= g.n -> 1 <= j -> j <= g.n -> vertexAt i = vertexAt j -> i = j
  covers : forall v, 1 <= v -> v <= g.n -> exists i, 1 <= i /\ i <= g.n /\ vertexAt i = v
  closingEquality : vertexAt (g.n + 1) = vertexAt 1
  steps : forall i, 1 <= i -> i <= g.n -> g.edge (vertexAt i) (vertexAt (i + 1))

def HasHamiltonianCycle (g : Graph) : Prop := Nonempty (Cycle g)
def q (names : Names) : Formula := .atom (names.encode .result)
def x (names : Names) (i v : Nat) : Formula := .atom (names.encode (.position i v))
def orx (names : Names) (k : Nat) : Formula := .atom (names.encode (.cases k))

def chain : List Formula -> Formula -> Formula
  | [], tail => tail
  | a :: rest, tail => a >> chain rest tail

def casesFormula (names : Names) (k : Nat) (vertices : List Nat) : Formula :=
  chain (vertices.map fun v => x names k v >> q names) (orx names k >> q names)

inductive Admissible (g : Graph) (names : Names) : Formula -> Prop where
  | totality (k) (lo : 1 <= k) (hi : k <= g.n) : Admissible g names (orx names k)
  | cases (k vertices) (lo : 1 <= k) (hi : k <= g.n)
      (complete : vertices.Perm (List.range' 1 g.n)) : Admissible g names (casesFormula names k vertices)
  | repevertexAt (i j v) (ilo : 1 <= i) (ihi : i <= g.n) (jlo : 1 <= j) (jhi : j <= g.n)
      (vlo : 1 <= v) (vhi : v <= g.n) (different : Not (i = j)) :
      Admissible g names (x names i v >> (x names j v >> q names))
  | nonedge (i u v) (ilo : 1 <= i) (ihi : i <= g.n)
      (ulo : 1 <= u) (uhi : u <= g.n) (vlo : 1 <= v) (vhi : v <= g.n)
      (missing : Not (g.edge u v)) :
      Admissible g names (x names i u >> (x names (i + 1) v >> q names))
  | nonedgeFlipped (i u v) (ilo : 1 <= i) (ihi : i <= g.n)
      (ulo : 1 <= u) (uhi : u <= g.n) (vlo : 1 <= v) (vhi : v <= g.n)
      (missing : Not (g.edge u v)) :
      Admissible g names (x names (i + 1) v >> (x names i u >> q names))
  | closingAlias (v) (vlo : 1 <= v) (vhi : v <= g.n) :
      Admissible g names (x names 1 v >> x names (g.n + 1) v)

def ContextAdmissible (g : Graph) (names : Names) (context : List Formula) : Prop :=
  forall f, f ∈ context -> Admissible g names f

def Truth (value : String -> Prop) : Formula -> Prop
  | .atom s => value s
  | .implication a b => Truth value a -> Truth value b

def tagTruth {g : Graph} (cycle : Cycle g) : Tag -> Prop
  | .result => False
  | .position i v => cycle.vertexAt i = v
  | .cases _ => True

def valuation {g : Graph} (names : Names) (cycle : Cycle g) (s : String) : Prop :=
  exists t, names.encode t = s /\ tagTruth cycle t

theorem valuation_encode {g : Graph} (names : Names) (cycle : Cycle g) (t : Tag) :
    valuation names cycle (names.encode t) <-> tagTruth cycle t := by
  constructor
  · rintro ⟨other, equal, truth⟩
    have same := names.injective equal
    subst other
    exact truth
  · intro truth
    exact ⟨t, rfl, truth⟩

theorem chain_true_of_false_member (value : String -> Prop) (premises : List Formula)
    (tail f : Formula) (member : f ∈ premises) (falsePremise : Not (Truth value f)) :
    Truth value (chain premises tail) := by
  induction premises with
  | nil => simp at member
  | cons a rest ih =>
      change Truth value a -> Truth value (chain rest tail)
      intro trueA
      rcases List.mem_cons.mp member with same | later
      · subst f
        exact False.elim (falsePremise trueA)
      · exact ih later

theorem admissible_true {g : Graph} (names : Names) (cycle : Cycle g) {f : Formula}
    (admitted : Admissible g names f) : Truth (valuation names cycle) f := by
  induction admitted with
  | totality k lo hi =>
      exact (valuation_encode names cycle (.cases k)).mpr True.intro
  | cases k vertices lo hi complete =>
      let chosen := cycle.vertexAt k
      have bounds := cycle.bounds k lo hi
      have inRange : chosen ∈ List.range' 1 g.n := by
        simp only [List.mem_range'_1]
        dsimp [chosen]
        omega
      have member : chosen ∈ vertices := complete.mem_iff.mpr inRange
      apply chain_true_of_false_member (valuation names cycle) _ _ (x names k chosen >> q names)
      · exact List.mem_map.mpr ⟨chosen, member, rfl⟩
      · intro continuation
        have selected : Truth (valuation names cycle) (x names k chosen) :=
          (valuation_encode names cycle (.position k chosen)).mpr rfl
        have falseQ := continuation selected
        exact (valuation_encode names cycle .result).mp falseQ
  | repevertexAt i j v ilo ihi jlo jhi vlo vhi different =>
      intro hi hj
      have ei := (valuation_encode names cycle (.position i v)).mp hi
      have ej := (valuation_encode names cycle (.position j v)).mp hj
      exact False.elim (different (cycle.distinct i j ilo ihi jlo jhi (ei.trans ej.symm)))
  | nonedge i u v ilo ihi ulo uhi vlo vhi missing =>
      intro hu hv
      have eu := (valuation_encode names cycle (.position i u)).mp hu
      have ev := (valuation_encode names cycle (.position (i + 1) v)).mp hv
      have edge := cycle.steps i ilo ihi
      rw [eu, ev] at edge
      exact False.elim (missing edge)
  | nonedgeFlipped i u v ilo ihi ulo uhi vlo vhi missing =>
      intro hv hu
      have eu := (valuation_encode names cycle (.position i u)).mp hu
      have ev := (valuation_encode names cycle (.position (i + 1) v)).mp hv
      have edge := cycle.steps i ilo ihi
      rw [eu, ev] at edge
      exact False.elim (missing edge)
  | closingAlias v vlo vhi =>
      intro first
      apply (valuation_encode names cycle (.position (g.n + 1) v)).mpr
      change cycle.vertexAt (g.n + 1) = v
      rw [cycle.closingEquality]
      exact (valuation_encode names cycle (.position 1 v)).mp first

theorem derives_truth (value : String -> Prop) {context : List Formula} {f : Formula}
    (derivation : Derives context f) (assumptions : forall a, a ∈ context -> Truth value a) :
    Truth value f := by
  induction derivation with
  | hyp member => exact assumptions _ member
  | intro premise ih =>
      intro alpha
      apply ih
      intro a member
      rcases List.mem_cons.mp member with same | later
      · subst a; exact alpha
      · exact assumptions a later
  | elim major minor ihMajor ihMinor => exact (ihMajor assumptions) (ihMinor assumptions)

theorem context_derivation_rules_out_cycle (g : Graph) (names : Names) (context : List Formula)
    (admitted : ContextAdmissible g names context) (derivation : Derives context (q names)) :
    Not (HasHamiltonianCycle g) := by
  rintro ⟨cycle⟩
  have truth := derives_truth (valuation names cycle) derivation
    (fun a member => admissible_true names cycle (admitted a member))
  exact (valuation_encode names cycle .result).mp truth

theorem imported_body_rules_out_cycle (g : Graph) (names : Names) (body : NDTree)
    (wellFormed : body.wellFormedB = true) (conclusion : body.conclusion = q names)
    (admitted : ContextAdmissible g names body.dependencies) : Not (HasHamiltonianCycle g) := by
  apply context_derivation_rules_out_cycle g names body.dependencies admitted
  simpa [conclusion] using body.derives_dependencies wellFormed


end NonHamiltonicity.Context
