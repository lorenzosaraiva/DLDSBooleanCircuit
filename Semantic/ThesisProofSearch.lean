import Semantic.RouteCorrespondence
import Semantic.FlowBranchingGridBridge






open Semantic FlowSpec RouteIndexedMultiToken RouteIndexedFastEvaluator
open GridCertificateSoundness Semantic.BranchingGrid FlowBranchingGridBridge

namespace ThesisProofSearch


theorem ReferenceVerdict_terminal_gridWitness (d : Graph) (valid : SingleIntroductionReference.ReferenceVerdict d)
    (root : Vertex) (node : root ∈ d.NODES)
    (terminal : (get_rule.outgoing root d).isEmpty = true) :
    ∃ tree : NDTree,
      tree.wellFormedB = true ∧ tree.conclusion = root.FORMULA ∧
      tree.dependencies = [] ∧
      occurrenceMarkedB tree.formulas root.FORMULA tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree (treeFormulaCoverage tree)) = true := by
  have checked := (SingleIntroductionReference.rootsB_iff d).mp valid.2.2 root node terminal
  obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ checked.1
  have closed := checked.2 route member
  have oldMember := SingleIntroductionReference.routes_subset_historical d (stdFuel d) root member
  obtain ⟨token, tree, present, payload, rebuilt, wellFormed, conclusion, marked⟩ :=
    flowRootPair_to_gridWitness d root route oldMember closed
  obtain ⟨same, sameRebuilt, _, _, dependency⟩ :=
    routeOccurrence_tree_dependencies d root token present
  have equal : same = tree := Option.some.inj (sameRebuilt.symm.trans rebuilt)
  subst same
  have emptyDependency : token.dependency = [] := by
    have := congrArg Prod.fst payload
    exact this.trans closed.1
  exact ⟨tree, wellFormed, conclusion,
    (normalizeDep_eq_nil_iff tree.dependencies).mp (dependency.trans emptyDependency), marked⟩

#print axioms ReferenceVerdict_terminal_gridWitness
end ThesisProofSearch
