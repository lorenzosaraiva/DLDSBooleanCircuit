import Flow.MultiIntroSoundness
import Infrastructure.RouteGrid

open Semantic FlowSpec SelfCertifying
namespace MultiIntro
set_option maxHeartbeats 3000000

theorem ordinaryEdge_dependency (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (e : Deduction) (r : FlowPair) (member : r ∈ ordinaryEdge d v prior e) :
    ∃ input ∈ prior e.START, ∃ alpha, r.1 = depRemove input.1 alpha := by
  cases formula : v.FORMULA with
  | atom symbol => simp [ordinaryEdge, formula] at member
  | implication alpha beta =>
    simp only [ordinaryEdge, formula] at member
    split at member
    · obtain ⟨input, hi, produced⟩ := List.mem_filterMap.mp member
      split at produced
      · obtain ⟨word, _, eq⟩ := Option.map_eq_some_iff.mp produced
        cases eq
        exact ⟨input, hi, alpha, rfl⟩
      · contradiction
    · contradiction

theorem reseedEdge_dependency (variant : Variant) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (e : Deduction) (r : FlowPair)
    (member : r ∈ reseedEdge variant d v prior e) :
    ∃ input ∈ prior e.START, ∃ alpha, r.1 = depRemove input.1 alpha := by
  cases formula : v.FORMULA with
  | atom symbol => simp [reseedEdge, formula] at member
  | implication alpha beta =>
    simp only [reseedEdge, formula] at member
    split at member
    · obtain ⟨input, hi, produced⟩ := List.mem_flatMap.mp member
      split_ifs at produced
      · obtain ⟨ancestor, _, eq⟩ := List.mem_map.mp produced
        subst r
        exact ⟨input, hi, alpha, rfl⟩
      · contradiction
    · contradiction

theorem step_supported (variant : Variant) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (fs : List Formula) (hv : v.FORMULA ∈ fs)
    (hp : ∀ u ∈ predsOf d v, FaithfulKeyed.RoutesSupported fs (prior u)) :
    FaithfulKeyed.RoutesSupported fs (step variant d prior v) := by
  intro r member f bit
  unfold step at member
  split at member
  · rw [List.mem_eraseDups] at member
    obtain ⟨e, he, produced⟩ := List.mem_flatMap.mp member
    have witness : ∃ input ∈ prior e.START, ∃ alpha, r.1 = depRemove input.1 alpha := by
      rcases List.mem_append.mp produced with ordinary | reseed
      · exact ordinaryEdge_dependency d v prior e r ordinary
      · exact reseedEdge_dependency variant d v prior e r reseed
    obtain ⟨input, hi, alpha, same⟩ := witness
    rw [same] at bit
    exact hp e.START (incoming_start_pred d v e he) input hi f (mem_depRemove.mp bit).1
  · exact FaithfulKeyed.stepWith_supported d (SingleIntroductionReference.guard d)
      (SingleIntroductionReference.guard d) v prior fs hv hp r member f bit

theorem evaluateAt_supported (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex)
    (hv : v.FORMULA ∈ RefinedEval.formulas d) :
    FaithfulKeyed.RoutesSupported (RefinedEval.formulas d) (evaluateAt variant d fuel v) := by
  induction fuel generalizing v with
  | zero => simp [evaluateAt, FaithfulKeyed.RoutesSupported]
  | succ fuel ih =>
    exact step_supported variant d v _ _ hv (fun u hu =>
      ih u (FaithfulKeyed.predecessor_formula_supported d v u hu))

theorem routes_basis_supported (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex) :
    ∀ r ∈ evaluateAt variant d fuel v, ∀ f ∈ r.1,
      f ∈ RefinedEval.formulas (RouteGrid.basisGraph d v) := by
  cases fuel with
  | zero => simp [evaluateAt]
  | succ fuel =>
    apply step_supported variant d v _ _ ((RouteGrid.basis_mem d v v.FORMULA).mpr (Or.inl rfl))
    intro u hu r hr f hf
    exact (RouteGrid.basis_mem d v f).mpr (Or.inr
      (evaluateAt_supported variant d fuel u
        (FaithfulKeyed.predecessor_formula_supported d v u hu) r hr f hf))

theorem routes_canonical (variant : Variant) (d : Graph) (fuel : Nat) (v : Vertex)
    (r : FlowPair) (member : r ∈ evaluateAt variant d fuel v) : CanonicalDep r.1 := by
  obtain ⟨tree, _, _, same⟩ := evaluateAt_tree_dependencies variant d fuel v r member
  rw [← same]
  exact normalizeDep_canonical _

theorem ordinaryEdge_setEq (d : Graph) (v : Vertex) (left right : Vertex → List FlowPair)
    (same : ∀ u, ListSetEq (left u) (right u)) (e : Deduction) :
    ListSetEq (ordinaryEdge d v left e) (ordinaryEdge d v right e) := by
  cases formula : v.FORMULA with
  | atom symbol => simp [ordinaryEdge, formula, ListSetEq]
  | implication alpha beta =>
    unfold ordinaryEdge
    simp only [formula]
    split
    · intro r
      simp only [List.mem_filterMap]
      exact exists_congr fun input => and_congr (same e.START input) Iff.rfl
    · exact ListSetEq.refl _

theorem reseedEdge_setEq (variant : Variant) (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair) (same : ∀ u, ListSetEq (left u) (right u))
    (e : Deduction) : ListSetEq (reseedEdge variant d v left e) (reseedEdge variant d v right e) := by
  cases formula : v.FORMULA with
  | atom symbol => simp [reseedEdge, formula, ListSetEq]
  | implication alpha beta =>
    unfold reseedEdge
    simp only [formula]
    split
    · intro r
      simp only [List.mem_flatMap]
      exact exists_congr fun input => and_congr (same e.START input) Iff.rfl
    · exact ListSetEq.refl _

theorem step_setEq (variant : Variant) (d : Graph) (left right : Vertex → List FlowPair)
    (same : ∀ u, ListSetEq (left u) (right u)) (v : Vertex) :
    ListSetEq (step variant d left v) (step variant d right v) := by
  unfold step
  split
  · intro r
    simp only [List.mem_eraseDups, List.mem_flatMap, List.mem_append]
    apply exists_congr
    intro e
    exact and_congr Iff.rfl (or_congr (ordinaryEdge_setEq d v left right same e r)
      (reseedEdge_setEq variant d v left right same e r))
  · exact DependencyGuard.stepWith_setEq d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) left right same v

end MultiIntro
