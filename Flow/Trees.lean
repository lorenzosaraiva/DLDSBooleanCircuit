import Flow.Core
import Infrastructure.NDTreeOccurrenceReplay

open Semantic FlowSpec GridCertificateSoundness
open HorizontalCompressionInput FlowBranchingGridBridge

namespace Flow.Tree

set_option maxHeartbeats 3000000
set_option maxRecDepth 100000

abbrev step (d : Graph) :=
  LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)

@[simp] theorem guard_own_label (d : Graph) (deduction : Deduction) :
    SingleIntroductionReference.guard d deduction.DEPENDENCY deduction = true := by
  by_cases empty : deduction.DEPENDENCY.isEmpty = true
  · simp [SingleIntroductionReference.guard, LambdaFlow.labelGuard, SingleIntroductionReference.compressorOverlay,
      SingleIntroductionReference.compressorLabel, empty, LambdaFlow.labelAgreesB]
  · simp [SingleIntroductionReference.guard, LambdaFlow.labelGuard, SingleIntroductionReference.compressorOverlay,
      SingleIntroductionReference.compressorLabel, empty, LambdaFlow.labelAgreesB, depSetEq_self]

@[simp] theorem guard_edge (d : Graph) (start finish : Vertex) (colour : Nat) (dep : Dep) :
    SingleIntroductionReference.guard d dep (edge start finish colour dep) = true :=
  guard_own_label d _

theorem step_incoming (left right : Graph) (vertex : Vertex)
    (incoming : get_rule.incoming vertex left = get_rule.incoming vertex right)
    (leftPaths : left.PATHS = []) (rightPaths : right.PATHS = [])
    (prior : Vertex → List FlowPair) : step left prior vertex = step right prior vertex := by
  have predecessors : predsOf left vertex = predsOf right vertex := by
    simp only [FlowSpec.predsOf_eq, incoming]
  have between : ∀ start, edgesBetween left start vertex = edgesBetween right start vertex := by
    intro start
    simp only [edgesBetween_eq_incoming, incoming]
  have eliminations : elimPairsAt left vertex = elimPairsAt right vertex := by
    simp only [elimPairsAt, predecessors]
  have premise : ∀ start, elimPremiseAtB left vertex start = elimPremiseAtB right vertex start := by
    intro start
    simp only [elimPremiseAtB, eliminations]
  have guards : SingleIntroductionReference.guard left = SingleIntroductionReference.guard right := rfl
  have leftAncestors : ancestorsInto left vertex = [] := by simp [ancestorsInto, leftPaths]
  have rightAncestors : ancestorsInto right vertex = [] := by simp [ancestorsInto, rightPaths]
  have leftEligible : eligibleAncestorsAtRoot left vertex = [] := by
    unfold eligibleAncestorsAtRoot
    split <;> simp [leftAncestors]
  have rightEligible : eligibleAncestorsAtRoot right vertex = [] := by
    unfold eligibleAncestorsAtRoot
    split <;> simp [rightAncestors]
  simp only [step, LambdaFlow.stepWith, LambdaFlow.mixedWith, LambdaFlow.reseedWith,
    leftAncestors, rightAncestors, leftEligible, rightEligible, List.isEmpty_nil,
    predecessors, eliminations, guards, premise]
  simp_rw [between]
  simp

@[simp] theorem hyp_step
    (formula : Formula) (level next : Nat)
    (premiseFlow : Vertex → List FlowPair) :
    step (encodedGraphAt (.hyp formula) level next) premiseFlow
        (encodeNDTreeAt (.hyp formula) level next).root =
      [(normalizeDep [formula], [])] := by
  simp [step, LambdaFlow.stepWith, encodedGraphAt, encodeNDTreeAt,
    predsOf, ancestorsInto,
    List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

theorem intro_step
    (alpha beta : Formula) (premise : NDTree) (level next : Nat)
    (premiseConclusion : premise.conclusion = beta)
    (premiseFlow : Vertex → List FlowPair)
    (atPremise : premiseFlow
        (encodeNDTreeAt premise (level + 1) (next + 1)).root =
      [(normalizeDep premise.dependencies, [])]) :
    step (encodedGraphAt (.intro alpha beta premise) level next)
        premiseFlow
        (encodeNDTreeAt (.intro alpha beta premise) level next).root =
      [(normalizeDep (NDTree.intro alpha beta premise).dependencies, [])] := by
  rw [FlowBranchingGridBridge.normalize_dependencies_intro]
  let graph := encodedGraphAt (.intro alpha beta premise) level next
  let root := (encodeNDTreeAt (.intro alpha beta premise) level next).root
  let child := (encodeNDTreeAt premise (level + 1) (next + 1)).root
  have predecessors : predsOf graph root = [child] := by
    exact encoded_intro_root_preds alpha beta premise level next
  have incoming : get_rule.incoming root graph =
      [edge child root 0 (normalizeDep premise.dependencies)] := by
    exact encodeNDTreeAt_intro_root_incoming alpha beta premise level next
  have childFormula : child.FORMULA = beta := by
    simp [child, premiseConclusion]
  have rootFormula : root.FORMULA = alpha >> beta := by
    simp [root, encodeNDTreeAt]
  have rootHypothesis : root.HYPOTHESIS = false := by
    rfl
  have childEdges : edgesBetween graph child root =
      [edge child root 0 (normalizeDep premise.dependencies)] := by
    rw [edgesBetween_eq_incoming, incoming]
    simp
  have eliminations : elimPairsAt graph root = [] := by
    simp [elimPairsAt, predecessors, childFormula, rootFormula,
      FlowSpec.formula_ne_implication_self]
  have noAncestors : ancestorsInto graph root = [] := by
    rfl
  have childRoutes : premiseFlow child =
      [(normalizeDep premise.dependencies, [])] := by
    exact atPremise
  have noEligible : eligibleAncestorsAtRoot graph root = [] := by
    unfold eligibleAncestorsAtRoot
    split <;> simp [noAncestors]
  change step graph premiseFlow root = _
  simp [step, LambdaFlow.stepWith, LambdaFlow.reseedWith,
    noEligible, predecessors, eliminations,
    rootFormula, childFormula,
    childEdges, childRoutes, FlowSpec.consume,
    rootHypothesis, List.eraseDups, List.eraseDupsBy,
    List.eraseDupsBy.loop]

theorem elim_step
    (alpha beta : Formula) (major minor : NDTree) (level next : Nat)
    (majorConclusion : major.conclusion = alpha >> beta)
    (minorConclusion : minor.conclusion = alpha)
    (premiseFlow : Vertex → List FlowPair)
    (atMajor : premiseFlow
        (encodeNDTreeAt major (level + 1) (next + 1)).root =
      [(normalizeDep major.dependencies, [])])
    (atMinor : premiseFlow
        (encodeNDTreeAt minor (level + 1)
          (encodeNDTreeAt major (level + 1) (next + 1)).next).root =
      [(normalizeDep minor.dependencies, [])]) :
    step (encodedGraphAt (.elim alpha beta major minor) level next)
        premiseFlow
        (encodeNDTreeAt (.elim alpha beta major minor) level next).root =
      [(normalizeDep (NDTree.elim alpha beta major minor).dependencies, [])] := by
  rw [FlowBranchingGridBridge.normalize_dependencies_elim]
  let graph := encodedGraphAt (.elim alpha beta major minor) level next
  let root := (encodeNDTreeAt (.elim alpha beta major minor) level next).root
  let majorRoot := (encodeNDTreeAt major (level + 1) (next + 1)).root
  let minorRoot := (encodeNDTreeAt minor (level + 1)
    (encodeNDTreeAt major (level + 1) (next + 1)).next).root
  have rootsNe : majorRoot ≠ minorRoot := by
    exact encoded_elim_roots_ne major minor level next
  have predecessors : predsOf graph root = [majorRoot, minorRoot] := by
    exact encoded_elim_root_preds alpha beta major minor level next
  have incoming : get_rule.incoming root graph =
      [edge majorRoot root 0 (normalizeDep major.dependencies),
       edge minorRoot root 0 (normalizeDep minor.dependencies)] := by
    exact encodeNDTreeAt_elim_root_incoming alpha beta major minor level next
  have majorFormula : majorRoot.FORMULA = alpha >> beta := by
    simp [majorRoot, majorConclusion]
  have minorFormula : minorRoot.FORMULA = alpha := by
    simp [minorRoot, minorConclusion]
  have rootFormula : root.FORMULA = beta := by
    simp [root, encodeNDTreeAt]
  have rootHypothesis : root.HYPOTHESIS = false := by
    rfl
  have majorEdges : edgesBetween graph majorRoot root =
      [edge majorRoot root 0 (normalizeDep major.dependencies)] := by
    rw [edgesBetween_eq_incoming, incoming]
    simp [Ne.symm rootsNe]
  have minorEdges : edgesBetween graph minorRoot root =
      [edge minorRoot root 0 (normalizeDep minor.dependencies)] := by
    rw [edgesBetween_eq_incoming, incoming]
    simp [rootsNe]
  have eliminations : elimPairsAt graph root = [(majorRoot, minorRoot)] := by
    simp [elimPairsAt, predecessors, majorFormula, minorFormula, rootFormula, FlowSpec.formula_ne_implication_self, FlowSpec.not_mutual_implication]
  have noAncestors : ancestorsInto graph root = [] := by
    rfl
  have majorRoutes : premiseFlow majorRoot =
      [(normalizeDep major.dependencies, [])] := by
    exact atMajor
  have minorRoutes : premiseFlow minorRoot =
      [(normalizeDep minor.dependencies, [])] := by
    exact atMinor
  have noEligible : eligibleAncestorsAtRoot graph root = [] := by
    unfold eligibleAncestorsAtRoot
    split <;> simp [noAncestors]
  change step graph premiseFlow root = _
  cases beta <;>
    simp [step, LambdaFlow.stepWith, LambdaFlow.mixedWith, LambdaFlow.reseedWith,
      noEligible, predecessors, eliminations, rootHypothesis,
      majorFormula, minorFormula, rootFormula, majorEdges, minorEdges,
      majorRoutes, minorRoutes, elimPremiseAtB, FlowSpec.consume,
      List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

theorem routes_in_context
    (tree : NDTree) (level next : Nat) (graph : Graph)
    (wellFormed : tree.wellFormedB = true)
    (noPaths : graph.PATHS = [])
    (incomingPreserved : ∀ vertex ∈
        (encodeNDTreeAt tree level next).nodes,
      get_rule.incoming vertex graph =
        get_rule.incoming vertex (encodedGraphAt tree level next))
    {fuel : Nat} (fuelSufficient : tree.height < fuel) :
    SingleIntroductionReference.routes graph fuel
        (encodeNDTreeAt tree level next).root =
      [(normalizeDep tree.dependencies, [])] := by
  induction tree generalizing level next fuel with
  | hyp formula =>
      cases fuel with
      | zero => simp [NDTree.height] at fuelSufficient
      | succ fuel =>
          let root := (encodeNDTreeAt (.hyp formula) level next).root
          have rootIncoming : get_rule.incoming root graph =
              get_rule.incoming root
                (encodedGraphAt (.hyp formula) level next) := by
            exact incomingPreserved root (by simp [root, encodeNDTreeAt])
          have predecessors : predsOf graph root = [] := by
            calc
              predsOf graph root = predsOf
                  (encodedGraphAt (.hyp formula) level next) root := by
                    simp only [FlowSpec.predsOf_eq, rootIncoming]
              _ = [] := by rfl
          let emptyFlow : Vertex → List FlowPair := SingleIntroductionReference.routes graph fuel
          have stepLocal := hyp_step formula level next
            emptyFlow
          have stepContext : step graph emptyFlow root =
              [(normalizeDep [formula], [])] :=
            (step_incoming graph
              (encodedGraphAt (.hyp formula) level next) root rootIncoming
              noPaths rfl emptyFlow).trans stepLocal
          simpa [SingleIntroductionReference.routes, LambdaFlow.lambdaFlowAt, LambdaFlow.evaluateAt, step, root, predecessors, emptyFlow,
            NDTree.dependencies] using
            stepContext
  | intro alpha beta premise ih =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true,
        beq_iff_eq] at wellFormed
      cases fuel with
      | zero => simp [NDTree.height] at fuelSufficient
      | succ fuel =>
          let root :=
            (encodeNDTreeAt (.intro alpha beta premise) level next).root
          let child :=
            (encodeNDTreeAt premise (level + 1) (next + 1)).root
          have childContext : ∀ vertex ∈
              (encodeNDTreeAt premise (level + 1) (next + 1)).nodes,
              get_rule.incoming vertex graph =
                get_rule.incoming vertex
                  (encodedGraphAt premise (level + 1) (next + 1)) := by
            intro vertex member
            exact (incomingPreserved vertex (by
              simp only [encodeNDTreeAt, List.mem_cons]
              exact Or.inr member)).trans
                (encodeNDTreeAt_intro_child_incoming alpha beta premise
                  level next member)
          have childFuel : premise.height < fuel := by
            simp only [NDTree.height] at fuelSufficient
            omega
          have childReplay := ih (level + 1) (next + 1)
            wellFormed.1 childContext childFuel
          have childReplayAt : SingleIntroductionReference.routes graph fuel child =
              [(normalizeDep premise.dependencies, [])] := by
            exact childReplay
          have rootIncoming : get_rule.incoming root graph =
              get_rule.incoming root
                (encodedGraphAt (.intro alpha beta premise) level next) := by
            exact incomingPreserved root (by simp [root, encodeNDTreeAt])
          have predecessors : predsOf graph root = [child] := by
            calc
              predsOf graph root = predsOf
                  (encodedGraphAt (.intro alpha beta premise) level next)
                  root := by
                    simp only [FlowSpec.predsOf_eq, rootIncoming]
              _ = [child] := encoded_intro_root_preds alpha beta premise
                level next
          let expected : List FlowPair :=
            [(normalizeDep premise.dependencies, [])]
          let premiseFlow : Vertex → List FlowPair := SingleIntroductionReference.routes graph fuel
          have atChild : premiseFlow child = expected := by
            exact childReplayAt
          have stepLocal : step
              (encodedGraphAt (.intro alpha beta premise) level next)
              premiseFlow root =
              [(normalizeDep
                (NDTree.intro alpha beta premise).dependencies, [])] := by
            exact intro_step alpha beta premise level next
              wellFormed.2 premiseFlow atChild
          have stepContext : step graph premiseFlow root =
              [(normalizeDep
                (NDTree.intro alpha beta premise).dependencies, [])] :=
            (step_incoming graph
              (encodedGraphAt (.intro alpha beta premise) level next)
              root rootIncoming noPaths rfl premiseFlow).trans stepLocal
          change SingleIntroductionReference.routes graph (fuel + 1) root = _
          exact stepContext
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true,
        beq_iff_eq] at wellFormed
      cases fuel with
      | zero => simp [NDTree.height] at fuelSufficient
      | succ fuel =>
          let root :=
            (encodeNDTreeAt (.elim alpha beta major minor) level next).root
          let majorRoot :=
            (encodeNDTreeAt major (level + 1) (next + 1)).root
          let minorRoot :=
            (encodeNDTreeAt minor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next).root
          have majorContext : ∀ vertex ∈
              (encodeNDTreeAt major (level + 1) (next + 1)).nodes,
              get_rule.incoming vertex graph =
                get_rule.incoming vertex
                  (encodedGraphAt major (level + 1) (next + 1)) := by
            intro vertex member
            exact (incomingPreserved vertex (by
              simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
              exact Or.inr (Or.inl member))).trans
                (encodeNDTreeAt_elim_major_incoming alpha beta major minor
                  level next member)
          have minorContext : ∀ vertex ∈
              (encodeNDTreeAt minor (level + 1)
                (encodeNDTreeAt major (level + 1) (next + 1)).next).nodes,
              get_rule.incoming vertex graph =
                get_rule.incoming vertex
                  (encodedGraphAt minor (level + 1)
                    (encodeNDTreeAt major (level + 1) (next + 1)).next) := by
            intro vertex member
            exact (incomingPreserved vertex (by
              simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
              exact Or.inr (Or.inr member))).trans
                (encodeNDTreeAt_elim_minor_incoming alpha beta major minor
                  level next member)
          have majorFuel : major.height < fuel := by
            simp only [NDTree.height] at fuelSufficient
            omega
          have minorFuel : minor.height < fuel := by
            simp only [NDTree.height] at fuelSufficient
            omega
          have majorReplay := ihMajor (level + 1) (next + 1)
            wellFormed.1.1.1 majorContext majorFuel
          have minorReplay := ihMinor (level + 1)
            (encodeNDTreeAt major (level + 1) (next + 1)).next
            wellFormed.1.1.2 minorContext minorFuel
          have majorReplayAt : SingleIntroductionReference.routes graph fuel majorRoot =
              [(normalizeDep major.dependencies, [])] := by
            exact majorReplay
          have minorReplayAt : SingleIntroductionReference.routes graph fuel minorRoot =
              [(normalizeDep minor.dependencies, [])] := by
            exact minorReplay
          have rootIncoming : get_rule.incoming root graph =
              get_rule.incoming root
                (encodedGraphAt (.elim alpha beta major minor) level next) := by
            exact incomingPreserved root (by simp [root, encodeNDTreeAt])
          have predecessors : predsOf graph root =
              [majorRoot, minorRoot] := by
            calc
              predsOf graph root = predsOf
                  (encodedGraphAt (.elim alpha beta major minor) level next)
                  root := by
                    simp only [FlowSpec.predsOf_eq, rootIncoming]
              _ = [majorRoot, minorRoot] := encoded_elim_root_preds
                alpha beta major minor level next
          let majorExpected : List FlowPair :=
            [(normalizeDep major.dependencies, [])]
          let minorExpected : List FlowPair :=
            [(normalizeDep minor.dependencies, [])]
          let premiseFlow : Vertex → List FlowPair := SingleIntroductionReference.routes graph fuel
          have rootsNe : majorRoot ≠ minorRoot := by
            exact encoded_elim_roots_ne major minor level next
          have atMajor : premiseFlow majorRoot = majorExpected := by
            exact majorReplayAt
          have atMinor : premiseFlow minorRoot = minorExpected := by
            exact minorReplayAt
          have stepLocal : step
              (encodedGraphAt (.elim alpha beta major minor) level next)
              premiseFlow root =
              [(normalizeDep
                (NDTree.elim alpha beta major minor).dependencies, [])] := by
            exact elim_step alpha beta major minor level next
              wellFormed.1.2 wellFormed.2 premiseFlow atMajor atMinor
          have stepContext : step graph premiseFlow root =
              [(normalizeDep
                (NDTree.elim alpha beta major minor).dependencies, [])] :=
            (step_incoming graph
              (encodedGraphAt (.elim alpha beta major minor) level next)
              root rootIncoming noPaths rfl premiseFlow).trans stepLocal
          change SingleIntroductionReference.routes graph (fuel + 1) root = _
          exact stepContext

theorem root_routes (tree : NDTree) (wellFormed : tree.wellFormedB = true) :
    SingleIntroductionReference.routes (ndTreeToGraph tree) (stdFuel (ndTreeToGraph tree))
      (encodeNDTreeAt tree 0 1).root = [(normalizeDep tree.dependencies, [])] := by
  apply routes_in_context tree 0 1 (ndTreeToGraph tree) wellFormed rfl
  · intro vertex member
    rfl
  · have height := Semantic.BranchingGrid.NDTree.height_le_nodeCount tree
    change tree.height < (ndTreeToGraph tree).NODES.length + 1
    rw [show (ndTreeToGraph tree).NODES.length = tree.nodeCount from encodeNDTreeAt_nodes_length tree 0 1]
    omega

theorem reference_accepted (tree : NDTree) (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) : SingleIntroductionReference.ReferenceVerdict (ndTreeToGraph tree) := by
  refine ⟨ValidDLDSDecomposition.validDLDS_structuralValid
      (ndTreeToGraph_validDLDS tree wellFormed closed),
    ndTreeToGraph_leveledColored tree, (SingleIntroductionReference.rootsB_iff _).mpr ?_⟩
  intro root listed terminal
  have equal := (ndTreeToGraph_uniqueRoot tree).2 root listed (List.isEmpty_iff.mp terminal)
  subst root
  rw [root_routes tree wellFormed, closed, normalizeDep_nil]
  simp

end Flow.Tree

namespace Flow

theorem tree_accepted (tree : GridCertificateSoundness.NDTree)
    (wellFormed : tree.wellFormedB = true) (closed : tree.dependencies = []) :
    _root_.Flow (HorizontalCompressionInput.ndTreeToGraph tree) :=
  of_reference _ (Tree.reference_accepted tree wellFormed closed)

end Flow
