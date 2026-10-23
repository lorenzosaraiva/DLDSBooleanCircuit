import Flow.MultiIntroLocality
import Flow.MultiIntroReferenceND

open Semantic FlowSpec GridCertificateSoundness FlowBranchingGridBridge
namespace MultiIntro
set_option maxHeartbeats 2000000

abbrev RouteND := MultiIntroReferenceND.RouteND

theorem intro_nd (v u : Vertex) (input : FlowPair) (alpha beta : Formula)
    (formula : v.FORMULA = alpha >> beta) (premise : u.FORMULA = beta)
    (correct : RouteND u input) (word : ColourPath) :
    RouteND v (depRemove input.1 alpha, word) := by
  obtain ⟨tree, wf, conclusion, dependency⟩ := correct
  refine ⟨.intro alpha beta tree, ?_, formula.symm, ?_⟩
  · simp [NDTree.wellFormedB, wf, conclusion, premise]
  · rw [normalize_dependencies_intro, dependency]

theorem ordinaryEdge_nd (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (e : Deduction) (correct : ∀ u r, r ∈ prior u → RouteND u r) :
    ∀ r, r ∈ ordinaryEdge d v prior e → RouteND v r := by
  intro r member
  cases formula : v.FORMULA with
  | atom symbol => simp [ordinaryEdge, formula] at member
  | implication alpha beta =>
    simp only [ordinaryEdge, formula] at member
    split at member
    · rename_i premise
      obtain ⟨input, hi, produced⟩ := List.mem_filterMap.mp member
      split at produced
      · obtain ⟨word, _, eq⟩ := Option.map_eq_some_iff.mp produced
        cases eq
        exact intro_nd v e.START input alpha beta formula (by simpa using premise)
          (correct e.START input hi) word
      · contradiction
    · contradiction

theorem reseedEdge_nd (variant : Variant) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) (e : Deduction)
    (correct : ∀ u r, r ∈ prior u → RouteND u r) :
    ∀ r, r ∈ reseedEdge variant d v prior e → RouteND v r := by
  intro r member
  cases formula : v.FORMULA with
  | atom symbol => simp [reseedEdge, formula] at member
  | implication alpha beta =>
    simp only [reseedEdge, formula] at member
    split at member
    · rename_i premise
      obtain ⟨input, hi, produced⟩ := List.mem_flatMap.mp member
      split_ifs at produced
      · obtain ⟨ancestor, _, eq⟩ := List.mem_map.mp produced
        subst r
        exact intro_nd v e.START input alpha beta formula (by simpa using premise)
          (correct e.START input hi) _
      · contradiction
    · contradiction

theorem step_nd (variant : Variant) (d : Graph) (prior : Vertex → List FlowPair)
    (correct : ∀ u r, r ∈ prior u → RouteND u r) (v : Vertex) :
    ∀ r, r ∈ step variant d prior v → RouteND v r := by
  intro r member
  unfold step at member
  split at member
  · rw [List.mem_eraseDups] at member
    obtain ⟨e, _, produced⟩ := List.mem_flatMap.mp member
    rcases List.mem_append.mp produced with ordinary | reseed
    · exact ordinaryEdge_nd d v prior e correct r ordinary
    · exact reseedEdge_nd variant d v prior e correct r reseed
  · exact MultiIntroReferenceND.step_nd d prior correct v r member

theorem evaluateAt_tree_dependencies (variant : Variant) (d : Graph) :
    ∀ fuel v r, r ∈ evaluateAt variant d fuel v → RouteND v r := by
  intro fuel
  induction fuel with
  | zero => intro v r member; simp [evaluateAt] at member
  | succ fuel ih => exact fun v r member => step_nd variant d _ ih v r member

theorem rootsB_iff (variant : Variant) (d : Graph) : rootsB variant d = true ↔
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      evaluateAt variant d (stdFuel d) root ≠ [] ∧
      ∀ r ∈ evaluateAt variant d (stdFuel d) root, r.1 = [] ∧ r.2 = [] := by
  rw [← roots_kernel_eq]
  simpa only [kernel_evaluate] using CompleteProfileKernel.rootsB_iff (kernel variant) d

theorem valid_roots_derivable (variant : Variant) (d : Graph)
    (valid : ValidMI variant d) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true → Derives [] root.FORMULA := by
  intro root listed terminal
  obtain ⟨nonempty, closed⟩ := (rootsB_iff variant d).mp valid.2.2 root listed terminal
  obtain ⟨r, member⟩ := List.exists_mem_of_ne_nil _ nonempty
  obtain ⟨tree, wf, conclusion, dependency⟩ :=
    evaluateAt_tree_dependencies variant d (stdFuel d) root r member
  exact extracted_certificate_sound wf conclusion
    ((normalizeDep_eq_nil_iff _).mp (dependency.trans (closed r member).1))

end MultiIntro
