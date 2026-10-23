import Flow.Core
import Flow.MultiIntroSoundness
import Infrastructure.ReferenceProofSearch
import Search.GridAcceptance

open Semantic FlowSpec GridCertificateSoundness Semantic.BranchingGrid
open FlowBranchingGridBridge

namespace Flow


theorem terminal_gridWitness (d : Graph) (valid : _root_.Flow d)
    (root : Vertex) (node : root ∈ d.NODES)
    (terminal : (get_rule.outgoing root d).isEmpty = true) :
    ∃ tree : NDTree,
      tree.wellFormedB = true ∧ tree.conclusion = root.FORMULA ∧
      tree.dependencies = [] ∧
      occurrenceMarkedB tree.formulas root.FORMULA tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree (treeFormulaCoverage tree)) = true := by
  have checked := (MultiIntro.rootsB_iff .general d).mp valid.2.2 root node terminal
  obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ checked.1
  obtain ⟨tree, wellFormed, conclusion, dependency⟩ :=
    MultiIntro.evaluateAt_tree_dependencies .general d (stdFuel d) root route member
  have closed : tree.dependencies = [] :=
    (normalizeDep_eq_nil_iff _).mp (dependency.trans (checked.2 route member).1)
  refine ⟨tree, wellFormed, conclusion, closed, ?_⟩
  rw [← conclusion]
  exact occurrenceMarked_complete tree.formulas tree (treeFormulaCoverage tree) wellFormed closed

theorem marked_sound (fs : List Formula) (goal : Formula) (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K)
    (marked : occurrenceMarkedB fs goal height K assignment = true) :
    Derives [] goal :=
  occurrenceMarked_sound fs goal height K assignment marked

theorem marked_extracts (fs : List Formula) (goal : Formula) (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K)
    (marked : occurrenceMarkedB fs goal height K assignment = true) :
    ∃ tree,
      extractOccurrenceGoalCone fs goal height K assignment = some tree ∧
        tree.wellFormedB = true ∧ tree.conclusion = goal ∧
        tree.dependencies = [] ∧ tree.height ≤ height :=
  occurrenceMarked_extracts fs goal height K assignment marked


theorem separation {d₁ d₂ : Graph}
    (formulaUniverse : buildFormulas d₁ = buildFormulas d₂)
    (height : GridAcceptance.levelBound d₁ = GridAcceptance.levelBound d₂) (goal : Formula) :
    ¬ ∃ paths : PathInput,
      GridAcceptance.Marked (buildFormulas d₁) goal (GridAcceptance.levelBound d₁) paths ≠
        GridAcceptance.Marked (buildFormulas d₂) goal (GridAcceptance.levelBound d₂) paths :=
  GridAcceptance.no_marked_path_distinguishes formulaUniverse height goal

end Flow
