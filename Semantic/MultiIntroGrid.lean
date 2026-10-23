import Semantic.MultiIntroGridSupport
import Semantic.RouteGrid



open Semantic FlowSpec SelfCertifying RouteGrid
namespace MultiIntroGrid
open MultiIntro

def step (variant : Variant) (d : Graph) (prior : (v : Vertex) → List (Token d v)) (v : Vertex) :
    List (Token d v) :=
  pack d v (MultiIntro.step variant d
    (fun u => project d u (prior u)) v)

def evaluateAt (variant : Variant) (d : Graph) : (fuel : Nat) → (v : Vertex) → List (Token d v)
  | 0, _ => []
  | fuel+1, v => step variant d (evaluateAt variant d fuel) v

def acceptedB (variant : Variant) (d : Graph) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    cellClosedB d v (evaluateAt variant d (stdFuel d) v)

theorem representable (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex)
    (route : FlowPair) (member : route ∈ MultiIntro.evaluateAt variant d fuel v) :
    decode d v (encode d v route) = route :=
  decode_encode d v route
    (MultiIntro.routes_canonical variant d fuel v route member)
    (MultiIntro.routes_basis_supported variant d fuel v route member)

theorem evaluateAt_setEq_routes (variant : Variant) (d : Graph) : ∀ fuel v,
    ListSetEq (project d v (evaluateAt variant d fuel v)) (MultiIntro.evaluateAt variant d fuel v) := by
  intro fuel
  induction fuel with
  | zero => intro v; exact ListSetEq.refl []
  | succ fuel ih =>
    intro v
    let produced := MultiIntro.step variant d
      (fun u => project d u (evaluateAt variant d fuel u)) v
    have localEq : ListSetEq produced (MultiIntro.evaluateAt variant d (fuel+1) v) :=
      MultiIntro.step_setEq variant d _ _ ih v
    change ListSetEq (project d v (pack d v produced)) _
    exact (project_pack d v produced (fun route member =>
      representable variant d (fuel+1) v route ((localEq route).mp member))).trans localEq

theorem token_mem_iff (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex) (token : Token d v) :
    token ∈ evaluateAt variant d fuel v ↔
      ∃ route ∈ MultiIntro.evaluateAt variant d fuel v, encode d v route = token := by
  cases fuel with
  | zero => simp [evaluateAt, MultiIntro.evaluateAt]
  | succ fuel =>
    have localEq := MultiIntro.step_setEq variant d
      (fun u => project d u (evaluateAt variant d fuel u)) (MultiIntro.evaluateAt variant d fuel)
      (evaluateAt_setEq_routes variant d fuel) v
    simp only [evaluateAt, step, pack, List.mem_eraseDups, List.mem_map]
    constructor
    · rintro ⟨route, member, same⟩
      exact ⟨route, (localEq route).mp member, same⟩
    · rintro ⟨route, member, same⟩
      exact ⟨route, (localEq route).mpr member, same⟩

theorem token_roundtrip (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex) (token : Token d v)
    (member : token ∈ evaluateAt variant d fuel v) : encode d v (decode d v token) = token := by
  obtain ⟨route, present, rfl⟩ := (token_mem_iff variant d fuel v token).mp member
  rw [representable variant d fuel v route present]

theorem distinct_tokens_distinct_routes (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex)
    (left right : Token d v) (leftMember : left ∈ evaluateAt variant d fuel v)
    (rightMember : right ∈ evaluateAt variant d fuel v)
    (same : decode d v left = decode d v right) : left = right := by
  rw [← token_roundtrip variant d fuel v left leftMember, ← token_roundtrip variant d fuel v right rightMember]
  exact congrArg (encode d v) same

theorem cellClosed_evaluate_eq (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex) :
    cellClosedB d v (evaluateAt variant d fuel v) =
      (!(MultiIntro.evaluateAt variant d fuel v).isEmpty &&
        (MultiIntro.evaluateAt variant d fuel v).all fun route => route.1.isEmpty && route.2.isEmpty) := by
  apply Bool.eq_iff_iff.mpr
  rw [cellClosed_iff, ← FaithfulKeyed.closedRoutes_iff_bool]
  exact (FaithfulKeyed.SummaryEq.of_mem_iff (evaluateAt_setEq_routes variant d fuel v)).closed_iff

theorem acceptedB_eq_rootsB (variant : Variant) (d : Graph) :
    acceptedB variant d = MultiIntro.rootsB variant d := by
  simp only [acceptedB, MultiIntro.rootsB, cellClosed_evaluate_eq]

theorem valid_iff_grid (variant : Variant) (d : Graph) : MultiIntro.ValidMI variant d ↔
    MultiIntro.MIDomain d ∧ LeveledColored d ∧ acceptedB variant d = true := by
  rw [acceptedB_eq_rootsB]
  rfl

#print axioms evaluateAt_setEq_routes
#print axioms token_mem_iff
#print axioms valid_iff_grid
end MultiIntroGrid
