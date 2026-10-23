import Infrastructure.SingleIntroductionReference
import Infrastructure.SpineNDSoundness

open Semantic FlowSpec GridCertificateSoundness FlowBranchingGridBridge
namespace MultiIntroReferenceND

set_option maxHeartbeats 1000000

def RouteND (v : Vertex) (route : FlowPair) : Prop :=
  ∃ tree : NDTree, tree.wellFormedB = true ∧ tree.conclusion = v.FORMULA ∧
    normalizeDep tree.dependencies = route.1

private theorem source_nd (v : Vertex) (residual : ColourPath) :
    RouteND v (normalizeDep [v.FORMULA], residual) := by
  exact ⟨.hyp v.FORMULA, rfl, rfl, rfl⟩

private theorem intro_nd (v u : Vertex) (route : FlowPair)
    (alpha beta : Formula) (shape : v.FORMULA = alpha >> beta)
    (premise : u.FORMULA = beta) (correct : RouteND u route)
    (residual : ColourPath) : RouteND v (depRemove route.1 alpha, residual) := by
  obtain ⟨tree, wellFormed, conclusion, dependency⟩ := correct
  refine ⟨.intro alpha beta tree, ?_, ?_, ?_⟩
  · simp [NDTree.wellFormedB, wellFormed, conclusion, premise]
  · exact shape.symm
  · rw [normalize_dependencies_intro, dependency]

private theorem elim_nd (d : Graph) (v major minor : Vertex)
    (pair : (major, minor) ∈ elimPairsAt d v) (left right : FlowPair)
    (minorND : RouteND minor left) (majorND : RouteND major right)
    (residual : ColourPath) : RouteND v (depUnion left.1 right.1, residual) := by
  obtain ⟨minorTree, minorWF, minorFormula, minorDep⟩ := minorND
  obtain ⟨majorTree, majorWF, majorFormula, majorDep⟩ := majorND
  have shape := elimPairsAt_formula d v major minor pair
  refine ⟨.elim minor.FORMULA v.FORMULA majorTree minorTree, ?_, rfl, ?_⟩
  · simp [NDTree.wellFormedB, minorWF, majorWF, minorFormula, majorFormula, shape]
  · rw [normalize_dependencies_elim, minorDep, majorDep]

private theorem elimReseed_nd (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (ancestors : List Ancestral)
    (correct : ∀ u route, route ∈ prior u → RouteND u route) :
    ∀ route, route ∈ LambdaFlow.elimReseedWith d (SingleIntroductionReference.guard d) v prior ancestors → RouteND v route := by
  intro route member
  simp only [LambdaFlow.elimReseedWith, List.mem_flatMap] at member
  obtain ⟨⟨major, minor⟩, pair, left, hl, right, hr, eMinor, _, eMajor, _, h⟩ := member
  split at h
  · obtain ⟨ancestor, _, produced⟩ := List.mem_map.mp h
    subst route
    exact elim_nd d v major minor pair left right (correct minor left hl)
      (correct major right hr) _
  · contradiction

private theorem introReseed_nd (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (ancestors : List Ancestral)
    (correct : ∀ u route, route ∈ prior u → RouteND u route) :
    ∀ route, route ∈ LambdaFlow.introReseedWith d (SingleIntroductionReference.guard d) v prior ancestors → RouteND v route := by
  intro route member
  unfold LambdaFlow.introReseedWith at member
  cases hp : predsOf d v with
  | nil => simp [hp] at member
  | cons u rest =>
    cases rest with
    | cons other rest => simp [hp] at member
    | nil =>
      cases hv : v.FORMULA with
      | atom symbol => simp [hp, hv] at member
      | implication alpha beta =>
        simp only [hp, hv] at member
        split at member
        · rename_i hu
          simp only [List.mem_flatMap] at member
          obtain ⟨input, hi, edge, _, h⟩ := member
          split at h
          · obtain ⟨ancestor, _, produced⟩ := List.mem_map.mp h
            subst route
            exact intro_nd v u input alpha beta hv (by simpa using hu)
              (correct u input hi) _
          · contradiction
        · contradiction

private theorem reseed_nd (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair)
    (correct : ∀ u route, route ∈ prior u → RouteND u route) :
    ∀ route, route ∈ LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v prior → RouteND v route := by
  intro route member
  simp only [LambdaFlow.reseedWith] at member
  split at member
  · contradiction
  · rw [List.mem_eraseDups, List.mem_append] at member
    rcases member with h | h
    · exact elimReseed_nd d v prior _ correct route h
    · exact introReseed_nd d v prior _ correct route h

private theorem mixed_nd (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair)
    (correct : ∀ u route, route ∈ prior u → RouteND u route) :
    ∀ route, route ∈ LambdaFlow.mixedWith d (SingleIntroductionReference.guard d) v prior →
      RouteND v route := by
  intro route member
  unfold LambdaFlow.mixedWith at member
  cases hv : v.FORMULA with
  | atom symbol => simp [hv] at member
  | implication alpha beta =>
    simp only [hv, List.mem_flatMap] at member
    obtain ⟨u, _, h⟩ := member
    split at h
    · rename_i hu
      obtain ⟨input, hi, h⟩ := List.mem_flatMap.mp h
      obtain ⟨edge, _, produced⟩ := List.mem_filterMap.mp h
      split at produced
      · obtain ⟨tail, _, eq⟩ := Option.map_eq_some_iff.mp produced
        cases eq
        have parts : (!elimPremiseAtB d v u) = true ∧ u.FORMULA = beta := by
          simpa only [Bool.and_eq_true, beq_iff_eq] using hu
        exact intro_nd v u input alpha beta hv parts.2 (correct u input hi) tail
      · contradiction
    · contradiction

theorem step_nd (d : Graph)
    (prior : Vertex → List FlowPair)
    (correct : ∀ u route, route ∈ prior u → RouteND u route) (v : Vertex) :
    ∀ route, route ∈ LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) prior v → RouteND v route := by
  intro route member
  simp only [LambdaFlow.stepWith] at member
  split at member
  · split at member
    · simp only [List.mem_singleton] at member
      subst route
      exact source_nd v []
    · obtain ⟨ancestor, _, eq⟩ := List.mem_map.mp member
      subst route
      exact source_nd v _
  · rw [List.mem_eraseDups, List.mem_append] at member
    rcases member with core | ancestral
    · split at core
      · cases hp : predsOf d v with
        | nil => simp [hp] at core
        | cons u rest =>
          cases rest with
          | cons other rest => simp [hp] at core
          | nil =>
            cases hv : v.FORMULA with
            | atom symbol => simp [hp, hv] at core
            | implication alpha beta =>
              simp only [hp, hv] at core
              split at core
              · rename_i hu
                obtain ⟨input, hi, h⟩ := List.mem_flatMap.mp core
                obtain ⟨edge, _, produced⟩ := List.mem_filterMap.mp h
                split at produced
                · obtain ⟨tail, _, eq⟩ := Option.map_eq_some_iff.mp produced
                  cases eq
                  exact intro_nd v u input alpha beta hv hu (correct u input hi) tail
                · contradiction
              · contradiction
      · rw [List.mem_append] at core
        rcases core with elimination | mixed
        · simp only [List.mem_flatMap] at elimination
          obtain ⟨⟨major, minor⟩, pair, left, hl, right, hr, eMinor, _, h⟩ := elimination
          obtain ⟨eMajor, _, produced⟩ := List.mem_filterMap.mp h
          split at produced
          · split at produced
            · split at produced
              · cases produced
                exact elim_nd d v major minor pair left right
                  (correct minor left hl) (correct major right hr) _
              · contradiction
            · contradiction
          · contradiction
        · exact mixed_nd d v prior correct route mixed
    · exact reseed_nd d v prior correct route ancestral


#print axioms step_nd
end MultiIntroReferenceND
