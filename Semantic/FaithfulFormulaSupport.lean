import Semantic.LambdaFlowCache
import Semantic.LambdaKeyedEncoding

open Semantic FlowSpec

namespace FaithfulKeyed

set_option maxHeartbeats 2000000

def RoutesSupported (fs : List Formula) (routes : List FlowPair) : Prop :=
  ∀ route ∈ routes, ∀ formula ∈ route.1, formula ∈ fs

theorem node_formula_supported (d : Graph) (v : Vertex) (member : v ∈ d.NODES) :
    v.FORMULA ∈ RefinedEval.formulas d := by
  simp only [RefinedEval.formulas, mem_normalizeDep, List.mem_append]
  exact Or.inl (List.mem_map.mpr ⟨v, member, rfl⟩)

theorem predecessor_formula_supported (d : Graph) (v u : Vertex)
    (member : u ∈ predsOf d v) : u.FORMULA ∈ RefinedEval.formulas d := by
  obtain ⟨edge, hedge, hstart⟩ := predsOf_has_edge member
  have hedges := mem_incoming_mem_edges v d hedge
  simp only [RefinedEval.formulas, mem_normalizeDep, List.mem_append]
  apply Or.inr
  apply List.mem_flatMap.mpr
  exact ⟨edge, hedges, by simp [hstart]⟩

private theorem supported_nil (fs : List Formula) : RoutesSupported fs [] := by
  simp [RoutesSupported]

private theorem supported_flatMap {α : Type} {fs : List Formula}
    {xs : List α} {f : α → List FlowPair}
    (supported : ∀ x ∈ xs, RoutesSupported fs (f x)) :
    RoutesSupported fs (xs.flatMap f) := by
  intro route member formula bit
  obtain ⟨x, hx, hr⟩ := List.mem_flatMap.mp member
  exact supported x hx route hr formula bit

private theorem supported_append {fs : List Formula} {left right : List FlowPair}
    (hl : RoutesSupported fs left) (hr : RoutesSupported fs right) :
    RoutesSupported fs (left ++ right) := by
  intro route member formula bit
  rcases List.mem_append.mp member with member | member
  · exact hl route member formula bit
  · exact hr route member formula bit

private theorem supported_eraseDups {fs : List Formula} {routes : List FlowPair}
    (supported : RoutesSupported fs routes) : RoutesSupported fs routes.eraseDups := by
  intro route member formula bit
  exact supported route (List.mem_eraseDups.mp member) formula bit

private theorem supported_remove {fs deps : List Formula} {alpha : Formula}
    (supported : ∀ f ∈ deps, f ∈ fs) : ∀ f ∈ depRemove deps alpha, f ∈ fs := by
  intro f hf
  simp only [depRemove, mem_normalizeDep, List.removeAll, List.mem_filter] at hf
  exact supported f hf.1

private theorem supported_union {fs left right : List Formula}
    (hl : ∀ f ∈ left, f ∈ fs) (hr : ∀ f ∈ right, f ∈ fs) :
    ∀ f ∈ depUnion left right, f ∈ fs := by
  intro f hf
  simp only [depUnion, mem_normalizeDep, List.mem_append] at hf
  exact hf.elim (hl f) (hr f)

private theorem supported_map_dep {α : Type} {fs deps : List Formula}
    {xs : List α} {residual : α → ColourPath}
    (supported : ∀ f ∈ deps, f ∈ fs) :
    RoutesSupported fs (xs.map fun x => (deps, residual x)) := by
  intro route member formula bit
  obtain ⟨x, _, rfl⟩ := List.mem_map.mp member
  exact supported formula bit

private theorem supported_option_dep {fs deps : List Formula} {p : Option ColourPath}
    (supported : ∀ f ∈ deps, f ∈ fs) :
    ∀ route, p.map (fun tail => (deps, tail)) = some route →
      ∀ f ∈ route.1, f ∈ fs := by
  cases p <;> simp_all

private theorem supported_filterMap {α : Type} {fs : List Formula}
    {xs : List α} {f : α → Option FlowPair}
    (supported : ∀ x ∈ xs, ∀ route, f x = some route →
      ∀ formula ∈ route.1, formula ∈ fs) :
    RoutesSupported fs (xs.filterMap f) := by
  intro route member formula bit
  obtain ⟨x, hx, output⟩ := List.mem_filterMap.mp member
  exact supported x hx route output formula bit

private theorem mixed_supported (d : Graph) (guard : LambdaFlow.Guard)
    (v : Vertex) (prior : Vertex → List FlowPair) (fs : List Formula)
    (hprior : ∀ u ∈ predsOf d v, RoutesSupported fs (prior u)) :
    RoutesSupported fs (LambdaFlow.mixedWith d guard v prior) := by
  unfold LambdaFlow.mixedWith
  split
  · apply supported_flatMap
    intro u hu
    split
    · apply supported_flatMap
      intro route hr
      apply supported_filterMap
      intro edge _ result output
      split at output
      · exact supported_option_dep (supported_remove (hprior u hu route hr)) result output
      · simp at output
    · exact supported_nil fs
  · exact supported_nil fs

private theorem elimReseed_supported (d : Graph) (guard : LambdaFlow.Guard)
    (v : Vertex) (prior : Vertex → List FlowPair) (ancestors : List Ancestral)
    (fs : List Formula)
    (hprior : ∀ u ∈ predsOf d v, RoutesSupported fs (prior u)) :
    RoutesSupported fs (LambdaFlow.elimReseedWith d guard v prior ancestors) := by
  unfold LambdaFlow.elimReseedWith
  apply supported_flatMap
  intro pair hp
  rcases pair with ⟨major, minor⟩
  obtain ⟨hmaj, hmin⟩ := mem_preds_of_mem_elimPairsAt hp
  apply supported_flatMap
  intro left hl
  apply supported_flatMap
  intro right hr
  apply supported_flatMap
  intro eMinor _
  apply supported_flatMap
  intro eMajor _
  split
  · exact supported_map_dep (supported_union (hprior minor hmin left hl)
      (hprior major hmaj right hr))
  · exact supported_nil fs

private theorem introReseed_supported (d : Graph) (guard : LambdaFlow.Guard)
    (v : Vertex) (prior : Vertex → List FlowPair) (ancestors : List Ancestral)
    (fs : List Formula)
    (hprior : ∀ u ∈ predsOf d v, RoutesSupported fs (prior u)) :
    RoutesSupported fs (LambdaFlow.introReseedWith d guard v prior ancestors) := by
  unfold LambdaFlow.introReseedWith
  split
  · rename_i u alpha beta hpred hformula
    have hu : u ∈ predsOf d v := by simp [hpred]
    split
    · apply supported_flatMap
      intro route hr
      apply supported_flatMap
      intro edge _
      split
      · exact supported_map_dep (supported_remove (hprior u hu route hr))
      · exact supported_nil fs
    · exact supported_nil fs
  · exact supported_nil fs

private theorem reseed_supported (d : Graph) (guard : LambdaFlow.Guard)
    (v : Vertex) (prior : Vertex → List FlowPair) (fs : List Formula)
    (hprior : ∀ u ∈ predsOf d v, RoutesSupported fs (prior u)) :
    RoutesSupported fs (LambdaFlow.reseedWith d guard v prior) := by
  unfold LambdaFlow.reseedWith
  dsimp only
  split
  · exact supported_nil fs
  · apply supported_eraseDups
    exact supported_append (elimReseed_supported d guard v prior _ fs hprior)
      (introReseed_supported d guard v prior _ fs hprior)

theorem stepWith_supported (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (v : Vertex) (prior : Vertex → List FlowPair) (fs : List Formula)
    (hvertex : v.FORMULA ∈ fs)
    (hprior : ∀ u ∈ predsOf d v, RoutesSupported fs (prior u)) :
    RoutesSupported fs (LambdaFlow.stepWith d ordinary reseed prior v) := by
  unfold LambdaFlow.stepWith
  dsimp only
  split
  · split
    · simpa [RoutesSupported] using hvertex
    · apply supported_map_dep
      simpa using hvertex
  · apply supported_eraseDups
    apply supported_append
    · split
      · split
        · rename_i u alpha beta hpred hformula
          have hu : u ∈ predsOf d v := by simp [hpred]
          split
          · apply supported_flatMap
            intro route hr
            apply supported_filterMap
            intro edge _ result output
            split at output
            · exact supported_option_dep (supported_remove (hprior u hu route hr)) result output
            · simp at output
          · exact supported_nil fs
        · exact supported_nil fs
      · apply supported_append
        · apply supported_flatMap
          intro pair hp
          rcases pair with ⟨major, minor⟩
          obtain ⟨hmaj, hmin⟩ := mem_preds_of_mem_elimPairsAt hp
          apply supported_flatMap
          intro left hl
          apply supported_flatMap
          intro right hr
          apply supported_flatMap
          intro eMinor _
          apply supported_filterMap
          intro eMajor _ result output
          have hs := supported_union (hprior minor hmin left hl) (hprior major hmaj right hr)
          split at output
          · split at output
            · split at output
              · cases output
                exact hs
              · simp at output
            · simp at output
          · simp at output
        · exact mixed_supported d ordinary v prior fs hprior
    · exact reseed_supported d reseed v prior fs hprior



theorem evaluateAt_supported (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) (v : Vertex) (hv : v.FORMULA ∈ RefinedEval.formulas d) :
    ∀ route ∈ LambdaFlow.evaluateAt d ordinary reseed fuel v,
      ∀ formula ∈ route.1, formula ∈ RefinedEval.formulas d := by
  induction fuel generalizing v with
  | zero => simp [LambdaFlow.evaluateAt]
  | succ fuel ih =>
      apply stepWith_supported d ordinary reseed v _ _ hv
      intro u hu
      exact ih u (predecessor_formula_supported d v u hu)

theorem evaluateAt_node_supported (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) (v : Vertex) (hv : v ∈ d.NODES) :
    ∀ route ∈ LambdaFlow.evaluateAt d ordinary reseed fuel v,
      ∀ formula ∈ route.1, formula ∈ RefinedEval.formulas d :=
  evaluateAt_supported d ordinary reseed fuel v (node_formula_supported d v hv)

#print axioms node_formula_supported
#print axioms predecessor_formula_supported
#print axioms stepWith_supported
#print axioms evaluateAt_supported
#print axioms evaluateAt_node_supported

end FaithfulKeyed
