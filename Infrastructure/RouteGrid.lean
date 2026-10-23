import Infrastructure.RouteEncoding
import Infrastructure.RouteCorrespondence
import Infrastructure.SpineNDSoundness



open Semantic FlowSpec SelfCertifying
namespace RouteGrid



def step (d : Graph) (prior : (v : Vertex) → List (Token d v)) (v : Vertex) :
    List (Token d v) :=
  pack d v (LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
    (fun u => project d u (prior u)) v)

def evaluateAt (d : Graph) : (fuel : Nat) → (v : Vertex) → List (Token d v)
  | 0, _ => []
  | fuel + 1, v => step d (evaluateAt d fuel) v

def acceptedB (d : Graph) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    cellClosedB d v (evaluateAt d (stdFuel d) v)

theorem reference_representable (d : Graph) (fuel : Nat) (v : Vertex)
    (route : FlowPair) (member : route ∈ SingleIntroductionReference.routes d fuel v) :
    decode d v (encode d v route) = route :=
  decode_encode d v route
    (SpineND.guardFreeFlowAt_dependency_canonical d fuel v route
      (SingleIntroductionReference.routes_subset_guardFree d fuel v member))
    (reference_supported d fuel v route member)



theorem evaluateAt_setEq_reference (d : Graph) : ∀ fuel v,
    ListSetEq (project d v (evaluateAt d fuel v)) (SingleIntroductionReference.routes d fuel v) := by
  intro fuel
  induction fuel with
  | zero => intro v; exact ListSetEq.refl []
  | succ fuel ih =>
    intro v
    let produced := LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
      (fun u => project d u (evaluateAt d fuel u)) v
    have localEq : ListSetEq produced (SingleIntroductionReference.routes d (fuel+1) v) :=
      DependencyGuard.stepWith_setEq d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
        _ _ ih v
    change ListSetEq (project d v (pack d v produced)) _
    exact (project_pack d v produced (fun route member =>
      reference_representable d (fuel+1) v route ((localEq route).mp member))).trans localEq



theorem token_mem_iff (d : Graph) (fuel : Nat) (v : Vertex) (token : Token d v) :
    token ∈ evaluateAt d fuel v ↔
      ∃ route ∈ SingleIntroductionReference.routes d fuel v, encode d v route = token := by
  cases fuel with
  | zero => simp [evaluateAt, SingleIntroductionReference.routes, LambdaFlow.evaluateAt]
  | succ fuel =>
    have localEq := DependencyGuard.stepWith_setEq d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
      (fun u => project d u (evaluateAt d fuel u)) (SingleIntroductionReference.routes d fuel)
      (evaluateAt_setEq_reference d fuel) v
    simp only [evaluateAt, step, pack, List.mem_eraseDups, List.mem_map]
    constructor
    · rintro ⟨route, member, same⟩
      exact ⟨route, (localEq route).mp member, same⟩
    · rintro ⟨route, member, same⟩
      exact ⟨route, (localEq route).mpr member, same⟩

theorem token_roundtrip (d : Graph) (fuel : Nat) (v : Vertex) (token : Token d v)
    (member : token ∈ evaluateAt d fuel v) : encode d v (decode d v token) = token := by
  obtain ⟨route, present, rfl⟩ := (token_mem_iff d fuel v token).mp member
  rw [reference_representable d fuel v route present]

theorem distinct_tokens_distinct_routes (d : Graph) (fuel : Nat) (v : Vertex)
    (left right : Token d v) (leftMember : left ∈ evaluateAt d fuel v)
    (rightMember : right ∈ evaluateAt d fuel v)
    (same : decode d v left = decode d v right) : left = right := by
  rw [← token_roundtrip d fuel v left leftMember, ← token_roundtrip d fuel v right rightMember]
  exact congrArg (encode d v) same

private theorem nodup_eraseDups {α : Type} [BEq α] [LawfulBEq α] :
    ∀ values : List α, values.eraseDups.Nodup
  | [] => by simp
  | value :: rest => by
    rw [List.eraseDups_cons]
    constructor
    · intro selection member equal
      rw [List.mem_eraseDups, List.mem_filter] at member
      subst selection
      simp at member
    · exact nodup_eraseDups (rest.filter fun selection => !selection == value)
termination_by values => values.length
decreasing_by simpa using List.length_filter_le (fun selection => !selection == value) rest

theorem evaluateAt_nodup (d : Graph) (fuel : Nat) (v : Vertex) :
    (evaluateAt d fuel v).Nodup := by
  cases fuel with
  | zero => simp [evaluateAt]
  | succ fuel => exact nodup_eraseDups _

theorem project_evaluateAt_nodup (d : Graph) (fuel : Nat) (v : Vertex) :
    (project d v (evaluateAt d fuel v)).Nodup := by
  exact List.Nodup.map_on (fun left hl right hr same =>
    distinct_tokens_distinct_routes d fuel v left right hl hr same) (evaluateAt_nodup d fuel v)



theorem token_count_eq_distinct_routes (d : Graph) (fuel : Nat) (v : Vertex) :
    (evaluateAt d fuel v).length = (SingleIntroductionReference.routes d fuel v).eraseDups.length := by
  have permutation : (project d v (evaluateAt d fuel v)).Perm
      (SingleIntroductionReference.routes d fuel v).eraseDups := by
    apply (List.perm_ext_iff_of_nodup (project_evaluateAt_nodup d fuel v) (nodup_eraseDups _)).mpr
    intro route
    simpa using evaluateAt_setEq_reference d fuel v route
  simpa [project] using permutation.length_eq

theorem cellClosed_evaluate_eq (d : Graph) (fuel : Nat) (v : Vertex) :
    cellClosedB d v (evaluateAt d fuel v) =
      (!(SingleIntroductionReference.routes d fuel v).isEmpty &&
        (SingleIntroductionReference.routes d fuel v).all fun route => route.1.isEmpty && route.2.isEmpty) := by
  apply Bool.eq_iff_iff.mpr
  rw [cellClosed_iff, ← FaithfulKeyed.closedRoutes_iff_bool]
  exact (FaithfulKeyed.SummaryEq.of_mem_iff (evaluateAt_setEq_reference d fuel v)).closed_iff


theorem acceptedB_eq_rootsB (d : Graph) : acceptedB d = SingleIntroductionReference.rootsB d := by
  simp only [acceptedB, SingleIntroductionReference.rootsB, cellClosed_evaluate_eq]

theorem ReferenceVerdict_iff_grid (d : Graph) : SingleIntroductionReference.ReferenceVerdict d ↔
    structuralValid d ∧ LeveledColored d ∧ acceptedB d = true := by
  rw [acceptedB_eq_rootsB]
  rfl

#print axioms evaluateAt_setEq_reference
#print axioms token_mem_iff
#print axioms distinct_tokens_distinct_routes
#print axioms token_count_eq_distinct_routes
#print axioms acceptedB_eq_rootsB
#print axioms ReferenceVerdict_iff_grid
end RouteGrid
