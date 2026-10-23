import Compression.NDTreePreReachability
import Infrastructure.OccurrenceAwareReplay










open Semantic FlowSpec
open GridCertificateSoundness ValidDLDSDecomposition
open MixedRuleReplay OccurrenceAwareReplay

namespace HorizontalCompressionInput

lemma edgesBetween_eq_incoming (d : Graph) (start finish : Vertex) :
    edgesBetween d start finish =
      (get_rule.incoming finish d).filter fun deduction =>
        decide (deduction.START = start) := by
  unfold edgesBetween
  rw [FlowSpec.incoming_eq_filter, List.filter_filter]
  apply List.filter_congr
  intro deduction member
  simp

lemma occurrenceStep_congr_of_incoming_paths_empty
    (left right : Graph) (vertex : Vertex)
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right)
    (leftPaths : left.PATHS = []) (rightPaths : right.PATHS = [])
    (premiseFlow : Vertex → List FlowPair) :
    occurrenceStep left vertex premiseFlow =
      occurrenceStep right vertex premiseFlow := by
  have predecessors : predsOf left vertex = predsOf right vertex := by
    simp only [FlowSpec.predsOf_eq, incoming]
  have between : ∀ start,
      edgesBetween left start vertex = edgesBetween right start vertex := by
    intro start
    simp only [edgesBetween_eq_incoming, incoming]
  have eliminations : elimPairsAt left vertex = elimPairsAt right vertex := by
    simp only [elimPairsAt, predecessors]
  have eliminationOutputs :
      selectionEliminationOutputs left vertex premiseFlow =
        selectionEliminationOutputs right vertex premiseFlow := by
    unfold selectionEliminationOutputs
    rw [eliminations]
    simp_rw [between]
  have introductions : introductionSelections left vertex premiseFlow =
      introductionSelections right vertex premiseFlow := by
    unfold introductionSelections selectionIntroductionOutputs
    rw [predecessors]
    simp_rw [between]
  have separated : separatedIntroductionOutputs left vertex premiseFlow =
      separatedIntroductionOutputs right vertex premiseFlow := by
    unfold separatedIntroductionOutputs
    rw [eliminationOutputs, introductions]
  have derived : derivedOutputs left vertex premiseFlow =
      derivedOutputs right vertex premiseFlow := by
    simp only [derivedOutputs, eliminationOutputs, separated]
  have leftAncestors : ancestorsInto left vertex = [] := by
    simp [ancestorsInto, leftPaths]
  have rightAncestors : ancestorsInto right vertex = [] := by
    simp [ancestorsInto, rightPaths]
  have sources : selectionSourceRoutes left vertex =
      selectionSourceRoutes right vertex := by
    simp [selectionSourceRoutes, leftAncestors, rightAncestors]
  simp only [occurrenceStep, predecessors, leftAncestors, rightAncestors,
    derived, sources]

@[simp] theorem encoded_intro_root_preds
    (alpha beta : Formula) (premise : NDTree) (level next : Nat) :
    predsOf (encodedGraphAt (.intro alpha beta premise) level next)
        (encodeNDTreeAt (.intro alpha beta premise) level next).root =
      [(encodeNDTreeAt premise (level + 1) (next + 1)).root] := by
  change predsOf
      (dlds (encodeNDTreeAt (.intro alpha beta premise) level next).nodes
        (encodeNDTreeAt (.intro alpha beta premise) level next).edges []) _ = _
  rw [FlowSpec.predsOf_eq, encodeNDTreeAt_intro_root_incoming]
  rfl

theorem encoded_elim_roots_ne
    (major minor : NDTree) (level next : Nat) :
    (encodeNDTreeAt major (level + 1) (next + 1)).root ≠
      (encodeNDTreeAt minor (level + 1)
        (encodeNDTreeAt major (level + 1) (next + 1)).next).root := by
  intro equal
  have numberEqual := congrArg Vertex.NUMBER equal
  simp [encodeNDTreeAt_next] at numberEqual
  have positive := major.nodeCount_pos
  omega

@[simp] theorem encoded_elim_root_preds
    (alpha beta : Formula) (major minor : NDTree) (level next : Nat) :
    predsOf (encodedGraphAt (.elim alpha beta major minor) level next)
        (encodeNDTreeAt (.elim alpha beta major minor) level next).root =
      [(encodeNDTreeAt major (level + 1) (next + 1)).root,
       (encodeNDTreeAt minor (level + 1)
        (encodeNDTreeAt major (level + 1) (next + 1)).next).root] := by
  change predsOf
      (dlds (encodeNDTreeAt (.elim alpha beta major minor) level next).nodes
        (encodeNDTreeAt (.elim alpha beta major minor) level next).edges []) _ = _
  rw [FlowSpec.predsOf_eq, encodeNDTreeAt_elim_root_incoming]
  exact FlowSpec.eraseDups_pair (encoded_elim_roots_ne major minor level next)

@[simp] theorem encoded_hyp_occurrenceStep
    (formula : Formula) (level next : Nat)
    (premiseFlow : Vertex → List FlowPair) :
    occurrenceStep (encodedGraphAt (.hyp formula) level next)
        (encodeNDTreeAt (.hyp formula) level next).root premiseFlow =
      .ok [(normalizeDep [formula], [])] := by
  simp [occurrenceStep, encodedGraphAt, encodeNDTreeAt,
    predsOf, selectionSourceRoutes, ancestorsInto,
    List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop]

theorem encoded_intro_occurrenceStep
    (alpha beta : Formula) (premise : NDTree) (level next : Nat)
    (premiseConclusion : premise.conclusion = beta)
    (premiseFlow : Vertex → List FlowPair)
    (atPremise : premiseFlow
        (encodeNDTreeAt premise (level + 1) (next + 1)).root =
      [(normalizeDep premise.dependencies, [])]) :
    occurrenceStep (encodedGraphAt (.intro alpha beta premise) level next)
        (encodeNDTreeAt (.intro alpha beta premise) level next).root
        premiseFlow =
      .ok [(normalizeDep (NDTree.intro alpha beta premise).dependencies, [])] := by
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
  change occurrenceStep graph root premiseFlow = _
  simp [occurrenceStep, predecessors, noAncestors, derivedOutputs,
    selectionEliminationOutputs, eliminations,
    separatedIntroductionOutputs, introductionSelections,
    rootFormula, childFormula, selectionIntroductionOutputs,
    childEdges, childRoutes, FlowSpec.consume, residualOccursB,
    rootHypothesis, List.eraseDups, List.eraseDupsBy,
    List.eraseDupsBy.loop]

theorem encoded_elim_occurrenceStep
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
    occurrenceStep (encodedGraphAt (.elim alpha beta major minor) level next)
        (encodeNDTreeAt (.elim alpha beta major minor) level next).root
        premiseFlow =
      .ok [(normalizeDep (NDTree.elim alpha beta major minor).dependencies, [])] := by
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
  have eliminationOutputs :
      selectionEliminationOutputs graph root premiseFlow =
        [(depUnion (normalizeDep minor.dependencies)
          (normalizeDep major.dependencies), [])] := by
    simp [selectionEliminationOutputs, eliminations, majorEdges, minorEdges,
      majorRoutes, minorRoutes, FlowSpec.consume]
  have separatedIntroductions :
      separatedIntroductionOutputs graph root premiseFlow = [] := by
    unfold separatedIntroductionOutputs
    rw [eliminationOutputs]
    cases beta <;>
      simp [introductionSelections, predecessors, rootFormula,
        majorFormula, minorFormula, selectionIntroductionOutputs,
        majorEdges, minorEdges, majorRoutes, minorRoutes,
        FlowSpec.consume, residualOccursB]
  change occurrenceStep graph root premiseFlow = _
  simp [occurrenceStep, predecessors, noAncestors, derivedOutputs,
    eliminationOutputs, separatedIntroductions,
    rootHypothesis, List.eraseDups, List.eraseDupsBy,
    List.eraseDupsBy.loop]







theorem encodeNDTreeAt_occurrenceReplayAt_in_context
    (tree : NDTree) (level next : Nat) (graph : Graph)
    (wellFormed : tree.wellFormedB = true)
    (noPaths : graph.PATHS = [])
    (incomingPreserved : ∀ vertex ∈
        (encodeNDTreeAt tree level next).nodes,
      get_rule.incoming vertex graph =
        get_rule.incoming vertex (encodedGraphAt tree level next))
    {fuel : Nat} (fuelSufficient : tree.height < fuel) :
    occurrenceReplayAt graph fuel
        (encodeNDTreeAt tree level next).root =
      .ok [(normalizeDep tree.dependencies, [])] := by
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
          let emptyFlow : Vertex → List FlowPair :=
            OccurrenceAwareReplay.routesFor []
          have stepLocal := encoded_hyp_occurrenceStep formula level next
            emptyFlow
          have stepContext : occurrenceStep graph root emptyFlow =
              .ok [(normalizeDep [formula], [])] :=
            (occurrenceStep_congr_of_incoming_paths_empty graph
              (encodedGraphAt (.hyp formula) level next) root rootIncoming
              noPaths rfl emptyFlow).trans stepLocal
          simpa [occurrenceReplayAt, root, predecessors, emptyFlow,
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
          have childReplayAt : occurrenceReplayAt graph fuel child =
              .ok [(normalizeDep premise.dependencies, [])] := by
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
          let premiseFlow : Vertex → List FlowPair :=
            OccurrenceAwareReplay.routesFor [(child, expected)]
          have atChild : premiseFlow child = expected := by
            simp [premiseFlow, OccurrenceAwareReplay.routesFor]
          have stepLocal : occurrenceStep
              (encodedGraphAt (.intro alpha beta premise) level next)
              root premiseFlow =
              .ok [(normalizeDep
                (NDTree.intro alpha beta premise).dependencies, [])] := by
            exact encoded_intro_occurrenceStep alpha beta premise level next
              wellFormed.2 premiseFlow atChild
          have stepContext : occurrenceStep graph root premiseFlow =
              .ok [(normalizeDep
                (NDTree.intro alpha beta premise).dependencies, [])] :=
            (occurrenceStep_congr_of_incoming_paths_empty graph
              (encodedGraphAt (.intro alpha beta premise) level next)
              root rootIncoming noPaths rfl premiseFlow).trans stepLocal
          change occurrenceReplayAt graph (fuel + 1) root = _
          rw [occurrenceReplayAt, predecessors]
          simpa [childReplayAt, premiseFlow, expected] using stepContext
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
          have majorReplayAt : occurrenceReplayAt graph fuel majorRoot =
              .ok [(normalizeDep major.dependencies, [])] := by
            exact majorReplay
          have minorReplayAt : occurrenceReplayAt graph fuel minorRoot =
              .ok [(normalizeDep minor.dependencies, [])] := by
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
          let premiseFlow : Vertex → List FlowPair :=
            OccurrenceAwareReplay.routesFor
              [(majorRoot, majorExpected), (minorRoot, minorExpected)]
          have rootsNe : majorRoot ≠ minorRoot := by
            exact encoded_elim_roots_ne major minor level next
          have atMajor : premiseFlow majorRoot = majorExpected := by
            simp [premiseFlow, OccurrenceAwareReplay.routesFor]
          have atMinor : premiseFlow minorRoot = minorExpected := by
            simp [premiseFlow, OccurrenceAwareReplay.routesFor, rootsNe]
          have stepLocal : occurrenceStep
              (encodedGraphAt (.elim alpha beta major minor) level next)
              root premiseFlow =
              .ok [(normalizeDep
                (NDTree.elim alpha beta major minor).dependencies, [])] := by
            exact encoded_elim_occurrenceStep alpha beta major minor level next
              wellFormed.1.2 wellFormed.2 premiseFlow atMajor atMinor
          have stepContext : occurrenceStep graph root premiseFlow =
              .ok [(normalizeDep
                (NDTree.elim alpha beta major minor).dependencies, [])] :=
            (occurrenceStep_congr_of_incoming_paths_empty graph
              (encodedGraphAt (.elim alpha beta major minor) level next)
              root rootIncoming noPaths rfl premiseFlow).trans stepLocal
          change occurrenceReplayAt graph (fuel + 1) root = _
          rw [occurrenceReplayAt, predecessors]
          simpa [majorReplayAt, minorReplayAt, premiseFlow,
            majorExpected, minorExpected] using stepContext




theorem encodeNDTreeAt_occurrenceReplayAt
    (tree : NDTree) (level next : Nat)
    (wellFormed : tree.wellFormedB = true)
    {fuel : Nat} (fuelSufficient : tree.height < fuel) :
    occurrenceReplayAt (encodedGraphAt tree level next) fuel
        (encodeNDTreeAt tree level next).root =
      .ok [(normalizeDep tree.dependencies, [])] := by
  apply encodeNDTreeAt_occurrenceReplayAt_in_context tree level next
    (encodedGraphAt tree level next) wellFormed rfl
  · intro vertex member
    rfl
  · exact fuelSufficient

def ndTreeSourceMark : NDTree → Bool
  | .hyp _ => true
  | .intro _ _ _ => false
  | .elim _ _ _ _ => false

@[simp] theorem encodeNDTreeAt_root_hypothesis
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root.HYPOTHESIS =
      ndTreeSourceMark tree := by
  cases tree <;> rfl



def InitialOccurrenceReplayAt
    (graph : Graph) (fuel : Nat) (vertex : Vertex) : Prop :=
  ∃ subtree level next,
    vertex = (encodeNDTreeAt subtree level next).root ∧
    vertex.FORMULA = subtree.conclusion ∧
    vertex.HYPOTHESIS = ndTreeSourceMark subtree ∧
    occurrenceReplayAt graph fuel vertex =
      .ok [(normalizeDep subtree.dependencies, [])]




theorem encodeNDTreeAt_allOccurrencesReplay_in_context
    (tree : NDTree) (level next : Nat) (graph : Graph)
    (wellFormed : tree.wellFormedB = true)
    (noPaths : graph.PATHS = [])
    (incomingPreserved : ∀ vertex ∈
        (encodeNDTreeAt tree level next).nodes,
      get_rule.incoming vertex graph =
        get_rule.incoming vertex (encodedGraphAt tree level next))
    {fuel : Nat} (fuelSufficient : tree.height < fuel) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      InitialOccurrenceReplayAt graph fuel vertex := by
  induction tree generalizing level next with
  | hyp formula =>
      intro vertex member
      simp [encodeNDTreeAt] at member
      subst vertex
      refine ⟨.hyp formula, level, next, rfl, ?_, ?_, ?_⟩
      · exact encodeNDTreeAt_root_formula (.hyp formula) level next
      · exact encodeNDTreeAt_root_hypothesis (.hyp formula) level next
      · exact encodeNDTreeAt_occurrenceReplayAt_in_context
          (.hyp formula) level next graph wellFormed noPaths
          incomingPreserved fuelSufficient
  | intro alpha beta premise ih =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true,
        beq_iff_eq] at wellFormed
      intro vertex member
      simp only [encodeNDTreeAt, List.mem_cons] at member
      rcases member with rfl | childMember
      · refine ⟨.intro alpha beta premise, level, next, rfl, ?_, ?_, ?_⟩
        · exact encodeNDTreeAt_root_formula (.intro alpha beta premise)
            level next
        · exact encodeNDTreeAt_root_hypothesis (.intro alpha beta premise)
            level next
        · exact encodeNDTreeAt_occurrenceReplayAt_in_context
            (.intro alpha beta premise) level next graph
            (by simpa [NDTree.wellFormedB] using wellFormed)
            noPaths incomingPreserved fuelSufficient
      · have childContext : ∀ child ∈
            (encodeNDTreeAt premise (level + 1) (next + 1)).nodes,
            get_rule.incoming child graph =
              get_rule.incoming child
                (encodedGraphAt premise (level + 1) (next + 1)) := by
          intro child childIn
          exact (incomingPreserved child (by
            simp only [encodeNDTreeAt, List.mem_cons]
            exact Or.inr childIn)).trans
              (encodeNDTreeAt_intro_child_incoming alpha beta premise
                level next childIn)
        apply ih (level + 1) (next + 1) wellFormed.1 childContext
        · simp only [NDTree.height] at fuelSufficient
          omega
        · exact childMember
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true,
        beq_iff_eq] at wellFormed
      intro vertex member
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append] at member
      rcases member with rfl | majorMember | minorMember
      · refine ⟨.elim alpha beta major minor, level, next, rfl, ?_, ?_, ?_⟩
        · exact encodeNDTreeAt_root_formula (.elim alpha beta major minor)
            level next
        · exact encodeNDTreeAt_root_hypothesis (.elim alpha beta major minor)
            level next
        · exact encodeNDTreeAt_occurrenceReplayAt_in_context
            (.elim alpha beta major minor) level next graph
            (by simpa [NDTree.wellFormedB] using wellFormed)
            noPaths incomingPreserved fuelSufficient
      · have majorContext : ∀ child ∈
            (encodeNDTreeAt major (level + 1) (next + 1)).nodes,
            get_rule.incoming child graph =
              get_rule.incoming child
                (encodedGraphAt major (level + 1) (next + 1)) := by
          intro child childIn
          exact (incomingPreserved child (by
            simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
            exact Or.inr (Or.inl childIn))).trans
              (encodeNDTreeAt_elim_major_incoming alpha beta major minor
                level next childIn)
        apply ihMajor (level + 1) (next + 1) wellFormed.1.1.1
          majorContext
        · simp only [NDTree.height] at fuelSufficient
          omega
        · exact majorMember
      · have minorContext : ∀ child ∈
            (encodeNDTreeAt minor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next).nodes,
            get_rule.incoming child graph =
              get_rule.incoming child
                (encodedGraphAt minor (level + 1)
                  (encodeNDTreeAt major (level + 1) (next + 1)).next) := by
          intro child childIn
          exact (incomingPreserved child (by
            simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
            exact Or.inr (Or.inr childIn))).trans
              (encodeNDTreeAt_elim_minor_incoming alpha beta major minor
                level next childIn)
        apply ihMinor (level + 1)
          (encodeNDTreeAt major (level + 1) (next + 1)).next
          wellFormed.1.1.2 minorContext
        · simp only [NDTree.height] at fuelSufficient
          omega
        · exact minorMember

theorem ndTreeToGraph_occurrenceReplayAt
    (tree : NDTree) (wellFormed : tree.wellFormedB = true) :
    occurrenceReplayAt (ndTreeToGraph tree)
        ((ndTreeToGraph tree).NODES.length + 1)
        (encodeNDTreeAt tree 0 1).root =
      .ok [(normalizeDep tree.dependencies, [])] := by
  have fuelSufficient : tree.height <
      (ndTreeToGraph tree).NODES.length + 1 := by
    rw [show (ndTreeToGraph tree).NODES =
        (encodeNDTreeAt tree 0 1).nodes by rfl,
      encodeNDTreeAt_nodes_length]
    have := Semantic.BranchingGrid.NDTree.height_le_nodeCount tree
    omega
  exact encodeNDTreeAt_occurrenceReplayAt tree 0 1 wellFormed
    fuelSufficient

theorem ndTreeToGraph_allOccurrencesReplay
    (tree : NDTree) (wellFormed : tree.wellFormedB = true) :
    ∀ vertex ∈ (ndTreeToGraph tree).NODES,
      InitialOccurrenceReplayAt (ndTreeToGraph tree)
        ((ndTreeToGraph tree).NODES.length + 1) vertex := by
  have fuelSufficient : tree.height <
      (ndTreeToGraph tree).NODES.length + 1 := by
    rw [show (ndTreeToGraph tree).NODES =
        (encodeNDTreeAt tree 0 1).nodes by rfl,
      encodeNDTreeAt_nodes_length]
    have := Semantic.BranchingGrid.NDTree.height_le_nodeCount tree
    omega
  apply encodeNDTreeAt_allOccurrencesReplay_in_context tree 0 1
    (ndTreeToGraph tree) wellFormed rfl
  · intro vertex member
    rfl
  · exact fuelSufficient

private theorem list_eq_singleton_of_nodup_mem_all_eq
    {α : Type} {items : List α} {item : α}
    (nodup : items.Nodup) (member : item ∈ items)
    (allEqual : ∀ value ∈ items, value = item) :
    items = [item] := by
  cases items with
  | nil => contradiction
  | cons head tail =>
      have headEqual := allEqual head (List.mem_cons_self)
      subst head
      have itemNotTail := (List.nodup_cons.mp nodup).1
      have tailEmpty : tail = [] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro value valueMember
        have valueEqual := allEqual value (List.mem_cons_of_mem _ valueMember)
        exact itemNotTail (valueEqual ▸ valueMember)
      simp [tailEmpty]

theorem encodeNDTreeAt_nodes_nodup
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).nodes.Nodup := by
  apply List.Nodup.of_map Vertex.NUMBER
  exact encodeNDTreeAt_node_numbers_nodup tree level next

theorem ndTreeToGraph_occurrenceRoots (tree : NDTree) :
    occurrenceRoots (ndTreeToGraph tree) =
      [(encodeNDTreeAt tree 0 1).root] := by
  let root := (encodeNDTreeAt tree 0 1).root
  have unique := ndTreeToGraph_uniqueRoot tree
  have rootsNodup : (occurrenceRoots (ndTreeToGraph tree)).Nodup := by
    exact (encodeNDTreeAt_nodes_nodup tree 0 1).filter _
  have rootMember : root ∈ occurrenceRoots (ndTreeToGraph tree) := by
    rw [occurrenceRoots, List.mem_filter]
    exact ⟨unique.1.1, by
      simpa [List.isEmpty_iff] using unique.1.2⟩
  have allEqual : ∀ vertex ∈ occurrenceRoots (ndTreeToGraph tree),
      vertex = root := by
    intro vertex member
    have filtered := List.mem_filter.mp member
    have terminal : get_rule.outgoing vertex (ndTreeToGraph tree) = [] := by
      simpa [List.isEmpty_iff] using filtered.2
    exact unique.2 vertex filtered.1 terminal
  exact list_eq_singleton_of_nodup_mem_all_eq rootsNodup rootMember allEqual



theorem ndTreeToGraph_occurrenceRootsComplete
    (tree : NDTree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    occurrenceRootsCompleteB (ndTreeToGraph tree)
      ((ndTreeToGraph tree).NODES.length + 1) = true := by
  unfold occurrenceRootsCompleteB
  rw [ndTreeToGraph_occurrenceRoots tree]
  simp only [List.isEmpty_cons, Bool.not_false, List.all_cons,
    List.all_nil, Bool.and_true, Bool.true_and]
  rw [ndTreeToGraph_occurrenceReplayAt tree wellFormed]
  simp [closed, routeClosedB]

end HorizontalCompressionInput
