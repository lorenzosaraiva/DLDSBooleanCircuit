import DLDSBooleanCircuit

open scoped Classical

namespace Semantic
namespace ReadingBased

/-- In the non-branching structural fragment, a non-hypothesis atom with one
    incoming source and no intro annotation can only be a repetition node.
    This rules out the malformed direct `A -> B` atom edge as a compatible
    grid vertex. -/
theorem atom_single_source_shape_forces_repetition_formula
    (bd : BranchingDLDS) (v u : Vertex) (name : String)
    (h_shape : vertexGridRuleShapeBool bd v = true)
    (h_formula : v.FORMULA = Formula.atom name)
    (h_not_hyp : v.HYPOTHESIS = false)
    (h_no_branch : findBranchTarget bd v = none)
    (h_no_intro : findIntroDischarge bd v = none)
    (h_sources : incomingSources bd.base v = [u]) :
    u.FORMULA = v.FORMULA := by
  unfold vertexGridRuleShapeBool isHypothesisShapeForGrid
    isRepetitionShapeForGrid isIntroShapeForGrid isElimShapeForGrid
    elimSourcePairMatches elimPremiseFormulasMatch sourceOneLevelAbove at h_shape
  rw [h_formula] at h_shape
  rw [h_formula]
  simp [h_not_hyp, h_no_branch, h_no_intro, h_sources] at h_shape
  exact h_shape.1

/-! A checked obstruction to the requested bridge.

This tiny DLDS has an arbitrary atom-to-atom edge `A -> B`. The
reading-based kernel follows `DLDS.E`, so `B` depends on `A`. The grid
node for atom `B` has no elimination rule for an arbitrary atom source, so
the actual grid output at `B` is zero. Thus the unrestricted bridge theorem
is false for the current grid construction. -/

namespace GridBridgeCounterexample

private def fA : Formula := .atom "A"
private def fB : Formula := .atom "B"

private def vA : Vertex :=
  { node := 0, LEVEL := 1, FORMULA := fA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vB : Vertex :=
  { node := 1, LEVEL := 0, FORMULA := fB,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

private def eAB : Deduction :=
  { START := vA, END := vB, COLOUR := 0, DEPENDENCY := [fA] }

private def base : DLDS :=
  { V := [vA, vB], E := [eAB], A := [] }

private def bd : BranchingDLDS :=
  { base := base, branchings := [], numReading := 0, evalOrder := [vA, vB] }

theorem atom_edge_grid_not_classicalKernel :
    Not (gridFEnvFromReading bd [] vB =
      classicalKernel bd [] (gridFEnvFromReading bd []) vB) := by
  native_decide

theorem atom_edge_not_GridCompatibleForReading :
    Not (bd.GridCompatibleForReading []) := by
  intro h
  exact atom_edge_grid_not_classicalKernel (h vB (by simp [bd]))

theorem atom_edge_not_StructuralGridCompatibleNonBranching :
    Not bd.StructuralGridCompatibleNonBranching := by
  intro h
  have h_shape : vertexGridRuleShapeBool bd vB = true :=
    structuralGridCompatibleNonBranching_vertex_shape bd h vB (by simp [bd])
  have h_forced : vA.FORMULA = vB.FORMULA :=
    atom_single_source_shape_forces_repetition_formula bd vB vA "B"
      h_shape
      (by simp [vB, fB])
      (by simp [vB])
      (by simp [findBranchTarget, bd])
      (by simp [findIntroDischarge, bd, base])
      (by simp [incomingSources, bd, base, eAB])
  simp [vA, vB, fA, fB] at h_forced

end GridBridgeCounterexample

/-! A second checked obstruction to the structural non-branching bridge.

The executable shape predicate accepts this DLDS as an implication-elimination
vertex, but the formula-grid node has both the elimination rule and the
repetition rule active at the conclusion column `A`. This exposes the next
needed invariant: structural compatibility must rule out rule-activation
overlap, not only malformed edge formulas. -/

namespace GridBridgeElimOverlapCounterexample

private def fA : Formula := .atom "A"
private def fAA : Formula := .impl fA fA

private def vMajor : Vertex :=
  { node := 0, LEVEL := 1, FORMULA := fAA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vMinor : Vertex :=
  { node := 1, LEVEL := 1, FORMULA := fA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vTarget : Vertex :=
  { node := 2, LEVEL := 0, FORMULA := fA,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

private def eMajor : Deduction :=
  { START := vMajor, END := vTarget, COLOUR := 0, DEPENDENCY := [fAA] }

private def eMinor : Deduction :=
  { START := vMinor, END := vTarget, COLOUR := 0, DEPENDENCY := [fA] }

private def base : DLDS :=
  { V := [vMajor, vMinor, vTarget], E := [eMajor, eMinor], A := [] }

private def bd : BranchingDLDS :=
  { base := base, branchings := [], numReading := 0,
    evalOrder := [vMajor, vMinor, vTarget] }

theorem elim_overlap_grid_not_classicalKernel :
    Not (gridFEnvFromReading bd [] vTarget =
      classicalKernel bd [] (gridFEnvFromReading bd []) vTarget) := by
  native_decide

theorem elim_overlap_not_GridCompatibleForReading :
    Not (bd.GridCompatibleForReading []) := by
  intro h
  exact elim_overlap_grid_not_classicalKernel (h vTarget (by simp [bd]))

theorem elim_overlap_taggedGrid_no_error :
    (get_tagged_eval_result (buildTaggedGridFromDLDS bd.base)
      (initialVectorsFromDLDS bd.base) (readingToTaggedPathFull bd [])).snd =
        false := by
  native_decide

theorem elim_overlap_taggedGrid_equals_classicalKernel :
    taggedGridFEnvFromReading bd [] vTarget =
      classicalKernel bd [] (taggedGridFEnvFromReading bd []) vTarget := by
  native_decide

theorem elim_overlap_TaggedGridCompatibleForReading :
    bd.TaggedGridCompatibleForReading [] := by
  intro v hv
  simp [bd] at hv
  rcases hv with rfl | rfl | rfl
  · native_decide
  · native_decide
  · native_decide

private lemma vMajor_mem_pre_of_target_split
    (pre post : List Vertex)
    (h : bd.evalOrder = pre ++ vTarget :: post) :
    vMajor ∈ pre := by
  cases pre with
  | nil =>
      simp [bd, vMajor, vTarget] at h
  | cons p ps =>
      simp [bd] at h
      simpa [List.mem_cons] using (Or.inl h.1 : vMajor = p ∨ vMajor ∈ ps)

private lemma vMinor_mem_pre_of_target_split
    (pre post : List Vertex)
    (h : bd.evalOrder = pre ++ vTarget :: post) :
    vMinor ∈ pre := by
  cases pre with
  | nil =>
      simp [bd, vMajor, vMinor, vTarget] at h
  | cons p ps =>
      cases ps with
      | nil =>
          simp [bd, vMajor, vMinor, vTarget] at h
      | cons q qs =>
          simp [bd] at h
          simpa [List.mem_cons] using
            (Or.inr (Or.inl h.2.1) :
              vMinor = p ∨ vMinor = q ∨ vMinor ∈ qs)

theorem elim_overlap_is_StructuralGridCompatibleNonBranching :
    bd.StructuralGridCompatibleNonBranching := by
  refine
    { nonbranching := by rfl
      wellformed := by
        unfold BranchingDLDS.WellFormed bd base
        exact List.Perm.refl _
      topo := ?_
      edges_in_vertices := ?_
      unique_one_level_outgoing := ?_
      vertex_shapes := by native_decide }
  · unfold BranchingDLDS.WellFormedTopo
    constructor
    · native_decide
    · intro e he pre post hsplit
      simp [bd, base] at he
      rcases he with rfl | rfl
      · exact vMajor_mem_pre_of_target_split pre post hsplit
      · exact vMinor_mem_pre_of_target_split pre post hsplit
  · intro e he
    change e ∈ [eMajor, eMinor] at he
    simp at he
    rcases he with rfl | rfl
    ·
      simp [bd, base, eMajor, vMajor, vMinor, vTarget]
    ·
      simp [bd, base, eMinor, vMajor, vMinor, vTarget]
  · intro e1 he1 e2 he2 h_start h_level1 h_level2
    change e1 ∈ [eMajor, eMinor] at he1
    change e2 ∈ [eMajor, eMinor] at he2
    simp at he1 he2
    rcases he1 with rfl | rfl
    · rcases he2 with rfl | rfl
      · rfl
      · have h_false : False := by
          simp [eMajor, eMinor, vMajor, vMinor] at h_start
        exact False.elim h_false
    · rcases he2 with rfl | rfl
      · have h_false : False := by
          simp [eMajor, eMinor, vMajor, vMinor] at h_start
        exact False.elim h_false
      · rfl

theorem structuralGridCompatibleNonBranching_not_sufficient :
    Not (forall (bd : BranchingDLDS) reading,
        bd.StructuralGridCompatibleNonBranching ->
          bd.GridCompatibleForReading reading) := by
  intro h
  exact elim_overlap_not_GridCompatibleForReading
    (h bd [] elim_overlap_is_StructuralGridCompatibleNonBranching)

theorem taggedGrid_strictly_improves_formulaGrid :
    Exists (fun bd : BranchingDLDS =>
      bd.StructuralGridCompatibleNonBranching /\
      bd.TaggedGridCompatibleForReading [] /\
      Not (bd.GridCompatibleForReading [])) := by
  exact Exists.intro bd
    (And.intro elim_overlap_is_StructuralGridCompatibleNonBranching
      (And.intro elim_overlap_TaggedGridCompatibleForReading
        elim_overlap_not_GridCompatibleForReading))

end GridBridgeElimOverlapCounterexample

end ReadingBased
end Semantic
