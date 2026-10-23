import Semantic.FlowProofSearch

open Semantic GridCertificateSoundness Semantic.BranchingGrid
open FlowBranchingGridBridge

namespace Flow

theorem search_complete (goal : Formula) :
    Derives [] goal ↔
      ∃ (fs : List Formula) (height K : Nat)
        (assignment : OccurrenceAssignment fs.length height K),
        occurrenceMarkedB fs goal height K assignment = true := by
  constructor
  · exact (derivable_iff_exists_occurrenceMarked goal).mp
  · rintro ⟨fs, height, K, assignment, marked⟩
    exact marked_sound fs goal height K assignment marked

theorem search_complete_dimensions (goal : Formula) (derivation : Derives [] goal) :
    ∃ tree : NDTree,
      tree.wellFormedB = true ∧ tree.conclusion = goal ∧ tree.dependencies = [] ∧
      occurrenceMarkedB tree.formulas goal tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree (treeFormulaCoverage tree)) = true := by
  obtain ⟨tree, wellFormed, conclusion, support⟩ := derives_has_finite_tree derivation
  have closed : tree.dependencies = [] := by
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro formula member
    have impossible := support formula member
    simp at impossible
  refine ⟨tree, wellFormed, conclusion, closed, ?_⟩
  rw [← conclusion]
  exact occurrenceMarked_complete tree.formulas tree (treeFormulaCoverage tree) wellFormed closed

end Flow
