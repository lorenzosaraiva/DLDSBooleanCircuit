import Infrastructure.SingleIntroductionReference
import Infrastructure.FaithfulFormulaSupport
import Infrastructure.FaithfulKeyedEncoding



open Semantic FlowSpec SelfCertifying
namespace RouteGrid

def basisGraph (d : Graph) (v : Vertex) : Graph := { d with NODES := v :: d.NODES }

structure Token (d : Graph) (v : Vertex) where
  dependency : KeyedEval.Vector (basisGraph d v)
  residual : ColourPath
  deriving BEq, DecidableEq, ReflBEq, LawfulBEq

def encode (d : Graph) (v : Vertex) (route : FlowPair) : Token d v :=
  ⟨depVector (RefinedEval.formulas (basisGraph d v)) route.1, route.2⟩

def decode (d : Graph) (v : Vertex) (token : Token d v) : FlowPair :=
  (normalizeDep (KeyedEval.decode (basisGraph d v) token.dependency), token.residual)

def pack (d : Graph) (v : Vertex) (routes : List FlowPair) : List (Token d v) :=
  (routes.map (encode d v)).eraseDups

def project (d : Graph) (v : Vertex) (tokens : List (Token d v)) : List FlowPair :=
  tokens.map (decode d v)

theorem basis_mem (d : Graph) (v : Vertex) (formula : Formula) :
    formula ∈ RefinedEval.formulas (basisGraph d v) ↔
      formula = v.FORMULA ∨ formula ∈ RefinedEval.formulas d := by
  simp [RefinedEval.formulas, basisGraph]

theorem basis_same_on_nodes (d : Graph) (v : Vertex) (member : v ∈ d.NODES) :
    RefinedEval.formulas (basisGraph d v) = RefinedEval.formulas d := by
  apply canonicalDep_eq_of_depSetEq (normalizeDep_canonical _) (normalizeDep_canonical _)
  apply (depSetEq_eq_true_iff _ _).mpr
  intro formula
  change formula ∈ RefinedEval.formulas (basisGraph d v) ↔ formula ∈ RefinedEval.formulas d
  rw [basis_mem]
  constructor
  · rintro (rfl | present)
    · exact FaithfulKeyed.node_formula_supported d v member
    · exact present
  · exact Or.inr

theorem reference_supported (d : Graph) (fuel : Nat) (v : Vertex) :
    ∀ route ∈ SingleIntroductionReference.routes d fuel v, ∀ formula ∈ route.1,
      formula ∈ RefinedEval.formulas (basisGraph d v) := by
  cases fuel with
  | zero => simp [SingleIntroductionReference.routes, LambdaFlow.evaluateAt]
  | succ fuel =>
    apply FaithfulKeyed.stepWith_supported d (SingleIntroductionReference.guard d)
      (SingleIntroductionReference.guard d) v _ _ ((basis_mem d v v.FORMULA).mpr (Or.inl rfl))
    intro premise predecessor route routeMember formula formulaMember
    apply (basis_mem d v formula).mpr
    exact Or.inr (FaithfulKeyed.evaluateAt_supported d (SingleIntroductionReference.guard d)
      (SingleIntroductionReference.guard d) fuel premise
      (FaithfulKeyed.predecessor_formula_supported d v premise predecessor)
      route routeMember formula formulaMember)

theorem decode_encode (d : Graph) (v : Vertex) (route : FlowPair)
    (canonical : CanonicalDep route.1)
    (supported : ∀ formula ∈ route.1, formula ∈ RefinedEval.formulas (basisGraph d v)) :
    decode d v (encode d v route) = route := by
  apply Prod.ext
  · apply canonicalDep_eq_of_depSetEq (normalizeDep_canonical _) canonical
    apply (depSetEq_eq_true_iff _ _).mpr
    intro formula
    change formula ∈ normalizeDep (KeyedEval.decode (basisGraph d v)
        (depVector (RefinedEval.formulas (basisGraph d v)) route.1)) ↔ formula ∈ route.1
    rw [mem_normalizeDep, LambdaKeyed.decode_depVector_mem]
    exact ⟨And.right, fun present => ⟨supported formula present, present⟩⟩
  · rfl

theorem project_pack (d : Graph) (v : Vertex) (routes : List FlowPair)
    (faithful : ∀ route ∈ routes, decode d v (encode d v route) = route) :
    ListSetEq (project d v (pack d v routes)) routes := by
  intro route
  simp only [project, pack, List.mem_map, List.mem_eraseDups]
  constructor
  · rintro ⟨token, ⟨source, member, rfl⟩, equality⟩
    rw [faithful source member] at equality
    exact equality ▸ member
  · intro member
    exact ⟨encode d v route, ⟨route, member, rfl⟩, faithful route member⟩

theorem encode_injective_on_routes (d : Graph) (v : Vertex) (left right : FlowPair)
    (leftCanonical : CanonicalDep left.1) (rightCanonical : CanonicalDep right.1)
    (leftSupported : ∀ f ∈ left.1, f ∈ RefinedEval.formulas (basisGraph d v))
    (rightSupported : ∀ f ∈ right.1, f ∈ RefinedEval.formulas (basisGraph d v))
    (same : encode d v left = encode d v right) : left = right := by
  have h := congrArg (decode d v) same
  simpa only [decode_encode d v left leftCanonical leftSupported,
    decode_encode d v right rightCanonical rightSupported] using h

theorem decode_dependency_empty_iff (d : Graph) (v : Vertex) (token : Token d v) :
    (decode d v token).1 = [] ↔ token.dependency.toList.all (!·) = true := by
  change normalizeDep (KeyedEval.decode (basisGraph d v) token.dependency) = [] ↔ _
  rw [FlowBranchingGridBridge.normalizeDep_eq_nil_iff]
  exact FaithfulKeyed.decode_eq_nil_iff _ _

def cellClosedB (d : Graph) (v : Vertex) (tokens : List (Token d v)) : Bool :=
  !tokens.isEmpty && tokens.all fun token =>
    token.dependency.toList.all (!·) && token.residual.isEmpty

theorem cellClosed_iff (d : Graph) (v : Vertex) (tokens : List (Token d v)) :
    cellClosedB d v tokens = true ↔ FaithfulKeyed.ClosedRoutes (project d v tokens) := by
  rw [FaithfulKeyed.closedRoutes_iff_bool]
  simp only [cellClosedB, project, List.isEmpty_map, List.all_map, Function.comp_def,
    Bool.and_eq_true, List.all_eq_true, List.isEmpty_iff]
  constructor
  · rintro ⟨nonempty, closed⟩
    refine ⟨nonempty, ?_⟩
    intro token member
    exact ⟨(decode_dependency_empty_iff d v token).mpr (List.all_eq_true.mpr (closed token member).1),
      (closed token member).2⟩
  · rintro ⟨nonempty, closed⟩
    refine ⟨nonempty, ?_⟩
    intro token member
    exact ⟨List.all_eq_true.mp ((decode_dependency_empty_iff d v token).mp (closed token member).1),
      (closed token member).2⟩

#print axioms decode_encode
#print axioms reference_supported
#print axioms project_pack
#print axioms cellClosed_iff
end RouteGrid
