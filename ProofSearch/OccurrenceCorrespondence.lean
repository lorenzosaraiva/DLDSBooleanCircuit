import Evaluator.Occurrence.Relation
import ProofSearch.FlowCorrespondence

/-!
# Successor reference routes as Branching Grid V2 witnesses

This bridge is graph-relative and occurrence-sensitive.  It first proves that
every successful route in the independent successor reference has an ordinary
well-formed `NDTree` with the same conclusion and normalized dependency.  It
then invokes the existing constructive Branching Grid V2 compiler.

The Grid does not inspect residual routing.  Residual completion therefore
remains an explicit successor-side conjunct.
-/

open DLDS Flow
open Evaluator.Mixed Evaluator.Occurrence
open Evaluator.Occurrence.LocalRules Evaluator.Occurrence.Relation
open DLDS.ProofSearch.BranchingGrid ProofSearch.Grid

namespace ProofSearch.OccurrenceCorrespondence

def ReferenceRouteTree (vertex : Vertex) (route : FlowPair) : Prop :=
  ∃ tree : NDTree,
    tree.wellFormedB = true ∧
    tree.conclusion = vertex.FORMULA ∧
    normalizeDep tree.dependencies = route.1

def ReferenceResultTreesCorrect (vertex : Vertex) : ReferenceResult -> Prop
  | .error _ => True
  | .success routes =>
      ∀ route, routes route -> ReferenceRouteTree vertex route

theorem sourceProduction_tree
    (d : Graph) (vertex : Vertex) (route : FlowPair)
    (production : SourceProduction d vertex route) :
    ReferenceRouteTree vertex route := by
  cases production with
  | traced ancestor member =>
      exact ⟨.hyp vertex.FORMULA, rfl, rfl, by simp [ProofSearch.Grid.NDTree.dependencies]⟩
  | default permitted =>
      exact ⟨.hyp vertex.FORMULA, rfl, rfl, by simp [ProofSearch.Grid.NDTree.dependencies]⟩

theorem relIntroductionProduction_tree
    (d : Graph) (vertex : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate)
    (premiseCorrect : ∀ premise route,
      premiseRoutes premise route -> ReferenceRouteTree premise route)
    (route : FlowPair)
    (production : RelIntroductionProduction d vertex premiseRoutes route) :
    ReferenceRouteTree vertex route := by
  cases production with
  | intro antecedent consequent premise dependency residual tail edge
      formula premiseMember premiseFormula routeMember edgeMember consumed =>
      obtain ⟨tree, wellFormed, conclusion, dependencyCorrect⟩ :=
        premiseCorrect premise (dependency, residual) routeMember
      refine ⟨.intro antecedent consequent tree, ?_, ?_, ?_⟩
      · simp [ProofSearch.Grid.NDTree.wellFormedB, wellFormed, conclusion, premiseFormula]
      · simpa [ProofSearch.Grid.NDTree.conclusion] using formula.symm
      · rw [ProofSearch.FlowCorrespondence.normalize_dependencies_intro,
          dependencyCorrect]

theorem relEliminationProduction_tree
    (d : Graph) (vertex : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate)
    (premiseCorrect : ∀ premise route,
      premiseRoutes premise route -> ReferenceRouteTree premise route)
    (route : FlowPair)
    (production : RelEliminationProduction d vertex premiseRoutes route) :
    ReferenceRouteTree vertex route := by
  cases production with
  | elim major minor minorDep majorDep minorResidual majorResidual tail
      minorEdge majorEdge pairMember minorRoute majorRoute minorEdgeMember
      majorEdgeMember minorConsumed majorConsumed =>
      obtain ⟨minorTree, minorWellFormed, minorConclusion,
        minorDependency⟩ :=
          premiseCorrect minor (minorDep, minorResidual) minorRoute
      obtain ⟨majorTree, majorWellFormed, majorConclusion,
        majorDependency⟩ :=
          premiseCorrect major (majorDep, majorResidual) majorRoute
      have majorFormula := ProofSearch.FlowCorrespondence.elimPairsAt_formula
        d vertex major minor pairMember
      refine ⟨.elim minor.FORMULA vertex.FORMULA majorTree minorTree,
        ?_, rfl, ?_⟩
      · simp [ProofSearch.Grid.NDTree.wellFormedB, minorWellFormed, majorWellFormed,
          minorConclusion, majorConclusion, majorFormula]
      · rw [ProofSearch.FlowCorrespondence.normalize_dependencies_elim,
          minorDependency, majorDependency]

theorem logicalProduction_tree
    (d : Graph) (vertex : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate)
    (premiseCorrect : ∀ premise route,
      premiseRoutes premise route -> ReferenceRouteTree premise route)
    (route : FlowPair)
    (production : LogicalProduction d vertex premiseRoutes route) :
    ReferenceRouteTree vertex route := by
  cases production with
  | elimination eliminationProduction =>
      exact relEliminationProduction_tree d vertex premiseRoutes
        premiseCorrect route eliminationProduction
  | introduction introductionProduction separated =>
      exact relIntroductionProduction_tree d vertex premiseRoutes
        premiseCorrect route introductionProduction

theorem reseededProduction_tree
    (d : Graph) (vertex : Vertex) (logical : RoutePredicate)
    (logicalCorrect : ∀ route, logical route -> ReferenceRouteTree vertex route)
    (route : FlowPair) (production : ReseededProduction d vertex logical route) :
    ReferenceRouteTree vertex route := by
  cases production with
  | reseed dependency residual ancestor logicalRoute ancestorMember prefixes =>
      exact logicalCorrect (dependency, residual) logicalRoute

theorem referenceStep_trees_correct
    (d : Graph) (vertex : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate)
    (premiseCorrect : ∀ premise route,
      premiseRoutes premise route -> ReferenceRouteTree premise route) :
    ReferenceResultTreesCorrect vertex
      (referenceStep d vertex premiseRoutes) := by
  classical
  unfold referenceStep
  split
  · split
    · intro route production
      exact sourceProduction_tree d vertex route production
    · trivial
  · let logical := LogicalProduction d vertex premiseRoutes
    have logicalCorrect : ∀ route, logical route ->
        ReferenceRouteTree vertex route :=
      fun route production => logicalProduction_tree d vertex premiseRoutes
        premiseCorrect route production
    split
    · change ReferenceResultTreesCorrect vertex
        (if _ : ∃ route, logical route then
          let reseeded := ReseededProduction d vertex logical
          if _ : ∃ route, reseeded route then .success reseeded
          else .error (.unsupportedReseeding vertex)
        else .error (.noLogicalProduction vertex))
      split
      · let reseeded := ReseededProduction d vertex logical
        change ReferenceResultTreesCorrect vertex
          (if _ : ∃ route, reseeded route then .success reseeded
          else .error (.unsupportedReseeding vertex))
        split
        · intro route production
          exact reseededProduction_tree d vertex logical logicalCorrect route
            production
        · trivial
      · trivial
    · split
      · change ReferenceResultTreesCorrect vertex
          (if _ : ∃ route, logical route then
            .success (fun route =>
              SourceProduction d vertex route ∨ logical route)
          else .error (.mixedProductionLost vertex))
        split
        · intro route production
          rcases production with source | derived
          · exact sourceProduction_tree d vertex route source
          · exact logicalCorrect route derived
        · trivial
      · change ReferenceResultTreesCorrect vertex
          (if _ : ∃ route, logical route then .success logical
          else .error (.noLogicalProduction vertex))
        split
        · exact logicalCorrect
        · trivial

inductive ReferencePremiseTableTreesCorrect :
    ReferencePremiseTable -> Prop
  | nil : ReferencePremiseTableTreesCorrect []
  | cons (vertex : Vertex) (routes : RoutePredicate)
      (tail : ReferencePremiseTable)
      (headCorrect : ∀ route, routes route -> ReferenceRouteTree vertex route)
      (tailCorrect : ReferencePremiseTableTreesCorrect tail) :
      ReferencePremiseTableTreesCorrect ((vertex, routes) :: tail)

theorem ReferencePremiseTableTreesCorrect.routesFor
    {table : ReferencePremiseTable}
    (correct : ReferencePremiseTableTreesCorrect table) :
    ∀ vertex route, referenceRoutesFor table vertex route ->
      ReferenceRouteTree vertex route := by
  intro vertex route member
  induction correct with
  | nil => simp [referenceRoutesFor] at member
  | cons headVertex routes tail headCorrect tailCorrect ih =>
      simp only [referenceRoutesFor, List.find?_cons] at member
      cases headEq : (headVertex == vertex) with
      | false =>
          simp [headEq] at member
          exact ih member
      | true =>
          have vertexEq : headVertex = vertex := by simpa using headEq
          simp [headEq] at member
          simpa [vertexEq] using headCorrect route member

theorem referencePremises_trees_correct
    (replay : Vertex -> ReferenceResult)
    (pointwise : ∀ vertex, ReferenceResultTreesCorrect vertex (replay vertex)) :
    ∀ premises,
      match referencePremises replay premises with
      | .error _ => True
      | .ok table => ReferencePremiseTableTreesCorrect table := by
  intro premises
  induction premises with
  | nil => exact .nil
  | cons premise premises ih =>
      cases replayEq : replay premise with
      | error reason => simp [referencePremises, replayEq]
      | success routes =>
          have headCorrect : ∀ route, routes route ->
              ReferenceRouteTree premise route := by
            simpa [ReferenceResultTreesCorrect, replayEq] using
              pointwise premise
          cases tailEq : referencePremises replay premises with
          | error reason => simp [referencePremises, replayEq, tailEq]
          | ok table =>
              have tailCorrect : ReferencePremiseTableTreesCorrect table := by
                simpa [tailEq] using ih
              simpa [referencePremises, replayEq, tailEq] using
                ProofSearch.OccurrenceCorrespondence.ReferencePremiseTableTreesCorrect.cons premise routes table
                  headCorrect tailCorrect

/-- Every route admitted by the fuelled independent successor reference has
an ordinary well-formed ND tree with exactly its normalized dependency. -/
theorem referenceReplayAt_tree_correct
    (d : Graph) (fuel : Nat) (vertex : Vertex) :
    ReferenceResultTreesCorrect vertex (referenceReplayAt d fuel vertex) := by
  induction fuel generalizing vertex with
  | zero => simp [referenceReplayAt, ReferenceResultTreesCorrect]
  | succ fuel ih =>
      simp only [referenceReplayAt]
      have premiseCorrect := referencePremises_trees_correct
        (referenceReplayAt d fuel) ih (predsOf d vertex)
      cases tableEq : referencePremises (referenceReplayAt d fuel)
          (predsOf d vertex) with
      | error reason => simp [ReferenceResultTreesCorrect]
      | ok table =>
          have tableCorrect : ReferencePremiseTableTreesCorrect table := by
            simpa [tableEq] using premiseCorrect
          simpa [tableEq] using referenceStep_trees_correct d vertex
            (referenceRoutesFor table) tableCorrect.routesFor

/-- The graph-relative Grid witness for one successor route.  The ND witness
is explicitly related to the route dependency; residual completion remains a
separate conjunct because the Grid does not check routing. -/
def SuccessorRouteGridCompleted (vertex : Vertex) (route : FlowPair) : Prop :=
  route.2 = [] ∧
    ∃ tree : NDTree,
      tree.wellFormedB = true ∧
      tree.conclusion = vertex.FORMULA ∧
      normalizeDep tree.dependencies = route.1 ∧
      occurrenceMarkedB tree.formulas vertex.FORMULA tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree
          (ProofSearch.FlowCorrespondence.treeFormulaCoverage tree)) = true

theorem completedReferenceRoute_iff_grid
    (vertex : Vertex) (route : FlowPair)
    (treeCorrect : ReferenceRouteTree vertex route) :
    (route.1 = [] ∧ route.2 = []) ↔
      SuccessorRouteGridCompleted vertex route := by
  obtain ⟨tree, wellFormed, conclusion, dependency⟩ := treeCorrect
  constructor
  · rintro ⟨dependencyEmpty, residualEmpty⟩
    have normalizedEmpty : normalizeDep tree.dependencies = [] := by
      simpa [dependencyEmpty] using dependency
    have treeClosed : tree.dependencies = [] :=
      (ProofSearch.FlowCorrespondence.normalizeDep_eq_nil_iff tree.dependencies).mp
        normalizedEmpty
    refine ⟨residualEmpty, tree, wellFormed, conclusion, dependency, ?_⟩
    simpa [conclusion] using occurrenceMarked_complete tree.formulas tree
      (ProofSearch.FlowCorrespondence.treeFormulaCoverage tree) wellFormed treeClosed
  · rintro ⟨residualEmpty, candidate, candidateWellFormed,
      candidateConclusion, candidateDependency, marked⟩
    have candidateClosed : candidate.dependencies = [] := by
      rw [← ProofSearch.FlowCorrespondence.encodeCertifiedNDTree_marked_iff_closed
        candidate.formulas candidate
        (ProofSearch.FlowCorrespondence.treeFormulaCoverage candidate)
        candidateWellFormed]
      simpa [candidateConclusion] using marked
    constructor
    · simpa [candidateClosed] using candidateDependency.symm
    · exact residualEmpty

def ReferenceCompiledGridValid (d : Graph) (fuel : Nat) : Prop :=
  (occurrenceRoots d).isEmpty = false ∧
    ∀ root, root ∈ occurrenceRoots d ->
      ∃ routes,
        referenceReplayAt d fuel root = .success routes ∧
        (∃ route, routes route) ∧
        ∀ route, routes route -> SuccessorRouteGridCompleted root route

theorem referenceRootsComplete_iff_compiledGridValid
    (d : Graph) (fuel : Nat) :
    ReferenceRootsComplete d fuel ↔
      ReferenceCompiledGridValid d fuel := by
  constructor
  · rintro ⟨rootsNonempty, rootsComplete⟩
    refine ⟨rootsNonempty, ?_⟩
    intro root rootMember
    have complete := rootsComplete root rootMember
    cases replayEq : referenceReplayAt d fuel root with
    | error reason => simp [ReferenceResultComplete, replayEq] at complete
    | success routes =>
        have complete' : (∃ route, routes route) ∧
            ∀ route, routes route -> routeClosedB route = true := by
          simpa [ReferenceResultComplete, replayEq] using complete
        refine ⟨routes, rfl, complete'.1, ?_⟩
        intro route member
        have routeTree := referenceReplayAt_tree_correct d fuel root
        have routeTree' : ReferenceRouteTree root route := by
          rw [replayEq] at routeTree
          exact routeTree route member
        apply (completedReferenceRoute_iff_grid root route routeTree').mp
        have closed := complete'.2 route member
        simpa [routeClosedB] using closed
  · rintro ⟨rootsNonempty, rootsValid⟩
    refine ⟨rootsNonempty, ?_⟩
    intro root rootMember
    obtain ⟨routes, replayEq, routesNonempty, routesValid⟩ :=
      rootsValid root rootMember
    simp only [replayEq, ReferenceResultComplete]
    refine ⟨routesNonempty, ?_⟩
    intro route member
    have routeTree := referenceReplayAt_tree_correct d fuel root
    have routeTree' : ReferenceRouteTree root route := by
      rw [replayEq] at routeTree
      exact routeTree route member
    have closed := (completedReferenceRoute_iff_grid root route routeTree').mpr
      (routesValid route member)
    simpa [routeClosedB] using closed

/-- Connected fixed-graph endpoint: the actual successor root Boolean is
equivalent to existence of the graph-relative family of marked Branching Grid
V2 witnesses, with routing completion retained explicitly. -/
theorem occurrenceRootsCompleteB_iff_compiledGridValid
    (d : Graph) (fuel : Nat) :
    occurrenceRootsCompleteB d fuel = true ↔
      ReferenceCompiledGridValid d fuel :=
  (occurrenceRootsCompleteB_iff_reference d fuel).trans
    (referenceRootsComplete_iff_compiledGridValid d fuel)

end ProofSearch.OccurrenceCorrespondence
