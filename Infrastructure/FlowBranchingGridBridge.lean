import Infrastructure.RouteIndexedFastEvaluator
import Flow.Grid.BranchingGridCompleteness

open Semantic FlowSpec SelfCertifying CertificateModel
open RouteIndexedMultiToken RouteIndexedFastEvaluator
open Semantic.BranchingGrid GridCertificateSoundness

namespace FlowBranchingGridBridge
















def routeKeyTree? : RouteKey -> Option NDTree
  | .source vertex _ => some (.hyp vertex.FORMULA)
  | .certificate _ _ => none
  | .slot _ _ lineage => routeKeyTree? lineage
  | .repetition _ _ parent => routeKeyTree? parent
  | .introduction vertex _ parent =>
      match vertex.FORMULA, routeKeyTree? parent with
      | .implication alpha beta, some premise =>
          some (.intro alpha beta premise)
      | _, _ => none
  | .mixedIntroduction vertex _ parent =>
      match vertex.FORMULA, routeKeyTree? parent with
      | .implication alpha beta, some premise =>
          some (.intro alpha beta premise)
      | _, _ => none
  | .elimination vertex _ minor _ _ minorKey majorKey =>
      match routeKeyTree? minorKey, routeKeyTree? majorKey with
      | some minorTree, some majorTree =>
          some (.elim minor.FORMULA vertex.FORMULA majorTree minorTree)
      | _, _ => none
  | .introReseed vertex _ _ parent =>
      match vertex.FORMULA, routeKeyTree? parent with
      | .implication alpha beta, some premise =>
          some (.intro alpha beta premise)
      | _, _ => none
  | .elimReseed vertex _ minor _ _ _ minorKey majorKey =>
      match routeKeyTree? minorKey, routeKeyTree? majorKey with
      | some minorTree, some majorTree =>
          some (.elim minor.FORMULA vertex.FORMULA majorTree minorTree)
      | _, _ => none

def routeTokenTree? (token : RouteIndexedToken) : Option NDTree :=
  routeKeyTree? token.routeKey



def routeKeyLogicalSize : RouteKey -> Nat
  | .source _ _ => 1
  | .certificate _ _ => 0
  | .slot _ _ lineage => routeKeyLogicalSize lineage
  | .repetition _ _ parent => routeKeyLogicalSize parent
  | .introduction _ _ parent => routeKeyLogicalSize parent + 1
  | .mixedIntroduction _ _ parent => routeKeyLogicalSize parent + 1
  | .elimination _ _ _ _ _ minorKey majorKey =>
      routeKeyLogicalSize majorKey + routeKeyLogicalSize minorKey + 1
  | .introReseed _ _ _ parent => routeKeyLogicalSize parent + 1
  | .elimReseed _ _ _ _ _ _ minorKey majorKey =>
      routeKeyLogicalSize majorKey + routeKeyLogicalSize minorKey + 1

theorem routeKeyTree_nodeCount {key : RouteKey} {tree : NDTree}
    (reconstructed : routeKeyTree? key = some tree) :
    tree.nodeCount = routeKeyLogicalSize key := by
  induction key generalizing tree with
  | source vertex seed =>
      simp [routeKeyTree?] at reconstructed
      subst tree
      rfl
  | certificate vertex slot => simp [routeKeyTree?] at reconstructed
  | slot vertex ordinal lineage ih =>
      exact ih reconstructed
  | repetition vertex edge parent ih =>
      exact ih reconstructed
  | introduction vertex edge parent ih =>
      cases formula : vertex.FORMULA <;>
        cases parentTree : routeKeyTree? parent <;>
          simp [routeKeyTree?, formula, parentTree] at reconstructed
      subst tree
      simp only [NDTree.nodeCount, ih parentTree]
      rw [routeKeyLogicalSize]
  | mixedIntroduction vertex edge parent ih =>
      cases formula : vertex.FORMULA <;>
        cases parentTree : routeKeyTree? parent <;>
          simp [routeKeyTree?, formula, parentTree] at reconstructed
      subst tree
      simp only [NDTree.nodeCount, ih parentTree]
      rw [routeKeyLogicalSize]
  | elimination vertex major minor minorEdge majorEdge minorKey majorKey
      ihMinor ihMajor =>
      cases minorTree : routeKeyTree? minorKey <;>
        cases majorTree : routeKeyTree? majorKey <;>
          simp [routeKeyTree?, minorTree, majorTree]
            at reconstructed
      subst tree
      simp only [NDTree.nodeCount, ihMinor minorTree, ihMajor majorTree]
      rw [routeKeyLogicalSize]
  | introReseed vertex edge ancestor parent ih =>
      cases formula : vertex.FORMULA <;>
        cases parentTree : routeKeyTree? parent <;>
          simp [routeKeyTree?, formula, parentTree] at reconstructed
      subst tree
      simp only [NDTree.nodeCount, ih parentTree]
      rw [routeKeyLogicalSize]
  | elimReseed vertex major minor minorEdge majorEdge ancestor minorKey majorKey
      ihMinor ihMajor =>
      cases minorTree : routeKeyTree? minorKey <;>
        cases majorTree : routeKeyTree? majorKey <;>
          simp [routeKeyTree?, minorTree, majorTree]
            at reconstructed
      subst tree
      simp only [NDTree.nodeCount, ihMinor minorTree, ihMajor majorTree]
      rw [routeKeyLogicalSize]




def RouteOccurrenceTreeCorrect (token : RouteIndexedToken) : Prop :=
  exists tree,
    routeTokenTree? token = some tree /\
    tree.wellFormedB = true /\
    tree.conclusion = token.vertex.FORMULA /\
    normalizeDep tree.dependencies = token.dependency

lemma normalizeDep_eq_of_membership {left right : Dep}
    (hleft : CanonicalDep left) (hright : CanonicalDep right)
    (hmembership : forall formula, formula ∈ left ↔ formula ∈ right) :
    left = right := by
  apply canonicalDep_eq_of_depSetEq hleft hright
  rw [depSetEq_eq_true_iff]
  exact hmembership

lemma mem_depUnion_iff {left right : Dep} {formula : Formula} :
    formula ∈ depUnion left right ↔ formula ∈ left ∨ formula ∈ right := by
  simp [depUnion]

lemma mem_depRemove_iff {dependency : Dep} {alpha formula : Formula} :
    formula ∈ depRemove dependency alpha ↔
      formula ∈ dependency ∧ formula ≠ alpha := by
  simp [depRemove, List.removeAll]

lemma normalize_dependencies_intro (alpha beta : Formula) (tree : NDTree) :
    normalizeDep (NDTree.intro alpha beta tree).dependencies =
      depRemove (normalizeDep tree.dependencies) alpha := by
  apply normalizeDep_eq_of_membership
  · exact normalizeDep_canonical _
  · exact depRemove_canonical _ _
  · intro formula
    simp [NDTree.dependencies, mem_depRemove_iff]

lemma normalize_dependencies_elim (alpha beta : Formula)
    (major minor : NDTree) :
    normalizeDep (NDTree.elim alpha beta major minor).dependencies =
      depUnion (normalizeDep minor.dependencies)
        (normalizeDep major.dependencies) := by
  apply normalizeDep_eq_of_membership
  · exact normalizeDep_canonical _
  · exact depUnion_canonical _ _
  · intro formula
    simp [NDTree.dependencies, mem_depUnion_iff, or_comm]

lemma routeOccurrenceTreeCorrect_source (d : Graph) (vertex : Vertex)
    (seed : Option Ancestral) (residual : ColourPath) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex (.source vertex seed)
        (normalizeDep [vertex.FORMULA]) residual) := by
  refine ⟨.hyp vertex.FORMULA, rfl, rfl, rfl, ?_⟩
  simp [NDTree.dependencies, RouteIndexedMultiToken.mkToken]

lemma routeOccurrenceTreeCorrect_slot (vertex : Vertex) (ordinal : Nat)
    (token : RouteIndexedToken)
    (correct : RouteOccurrenceTreeCorrect token) :
    RouteOccurrenceTreeCorrect
      { token with routeKey := .slot vertex ordinal token.routeKey } := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ := correct
  exact ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩

def RouteOccurrenceAtCorrect (vertex : Vertex)
    (token : RouteIndexedToken) : Prop :=
  token.vertex = vertex ∧ RouteOccurrenceTreeCorrect token

lemma routeOccurrenceTreeCorrect_repetition (d : Graph) (vertex : Vertex)
    (edge : Deduction) (token : RouteIndexedToken)
    (vertexFormula : token.vertex.FORMULA = vertex.FORMULA)
    (correct : RouteOccurrenceTreeCorrect token) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex (.repetition vertex edge token.routeKey)
        token.dependency token.residual) := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ := correct
  refine ⟨tree, reconstructed, wellFormed, ?_, ?_⟩
  · simpa [RouteIndexedMultiToken.mkToken] using conclusion.trans vertexFormula
  · simpa [RouteIndexedMultiToken.mkToken] using dependency

lemma routeOccurrenceTreeCorrect_introduction
    (d : Graph) (vertex : Vertex) (edge : Deduction)
    (token : RouteIndexedToken) (alpha beta : Formula)
    (vertexFormula : vertex.FORMULA = alpha >> beta)
    (premiseFormula : token.vertex.FORMULA = beta)
    (correct : RouteOccurrenceTreeCorrect token)
    (residual : ColourPath) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex (.introduction vertex edge token.routeKey)
        (depRemove token.dependency alpha) residual) := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ := correct
  change routeKeyTree? token.routeKey = some tree at reconstructed
  refine ⟨.intro alpha beta tree, ?_, ?_, ?_, ?_⟩
  · simp [routeTokenTree?, routeKeyTree?, RouteIndexedMultiToken.mkToken,
      vertexFormula, reconstructed]
  · simp [NDTree.wellFormedB, wellFormed, conclusion, premiseFormula]
  · simp [NDTree.conclusion, RouteIndexedMultiToken.mkToken, vertexFormula]
  · rw [normalize_dependencies_intro, dependency]
    simp [RouteIndexedMultiToken.mkToken]

lemma routeOccurrenceTreeCorrect_mixedIntroduction
    (d : Graph) (vertex : Vertex) (edge : Deduction)
    (token : RouteIndexedToken) (alpha beta : Formula)
    (vertexFormula : vertex.FORMULA = alpha >> beta)
    (premiseFormula : token.vertex.FORMULA = beta)
    (correct : RouteOccurrenceTreeCorrect token)
    (residual : ColourPath) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex (.mixedIntroduction vertex edge token.routeKey)
        (depRemove token.dependency alpha) residual) := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ := correct
  change routeKeyTree? token.routeKey = some tree at reconstructed
  refine ⟨.intro alpha beta tree, ?_, ?_, ?_, ?_⟩
  · simp [routeTokenTree?, routeKeyTree?, RouteIndexedMultiToken.mkToken,
      vertexFormula, reconstructed]
  · simp [NDTree.wellFormedB, wellFormed, conclusion, premiseFormula]
  · simp [NDTree.conclusion, RouteIndexedMultiToken.mkToken, vertexFormula]
  · rw [normalize_dependencies_intro, dependency]
    simp [RouteIndexedMultiToken.mkToken]

lemma routeOccurrenceTreeCorrect_introReseed
    (d : Graph) (vertex : Vertex) (edge : Deduction)
    (ancestor : Ancestral) (token : RouteIndexedToken)
    (alpha beta : Formula)
    (vertexFormula : vertex.FORMULA = alpha >> beta)
    (premiseFormula : token.vertex.FORMULA = beta)
    (correct : RouteOccurrenceTreeCorrect token) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex
        (.introReseed vertex edge ancestor token.routeKey)
        (depRemove token.dependency alpha) ancestor.COLOURS) := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ := correct
  change routeKeyTree? token.routeKey = some tree at reconstructed
  refine ⟨.intro alpha beta tree, ?_, ?_, ?_, ?_⟩
  · simp [routeTokenTree?, routeKeyTree?, RouteIndexedMultiToken.mkToken,
      vertexFormula, reconstructed]
  · simp [NDTree.wellFormedB, wellFormed, conclusion, premiseFormula]
  · simp [NDTree.conclusion, RouteIndexedMultiToken.mkToken, vertexFormula]
  · rw [normalize_dependencies_intro, dependency]
    simp [RouteIndexedMultiToken.mkToken]

lemma routeOccurrenceTreeCorrect_elimination
    (d : Graph) (vertex major minor : Vertex)
    (minorEdge majorEdge : Deduction)
    (minorToken majorToken : RouteIndexedToken)
    (majorFormula : major.FORMULA = minor.FORMULA >> vertex.FORMULA)
    (minorAt : minorToken.vertex = minor)
    (majorAt : majorToken.vertex = major)
    (minorCorrect : RouteOccurrenceTreeCorrect minorToken)
    (majorCorrect : RouteOccurrenceTreeCorrect majorToken)
    (residual : ColourPath) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex
        (.elimination vertex major minor minorEdge majorEdge
          minorToken.routeKey majorToken.routeKey)
        (depUnion minorToken.dependency majorToken.dependency) residual) := by
  obtain ⟨minorTree, minorReconstructed, minorWellFormed,
    minorConclusion, minorDependency⟩ := minorCorrect
  obtain ⟨majorTree, majorReconstructed, majorWellFormed,
    majorConclusion, majorDependency⟩ := majorCorrect
  change routeKeyTree? minorToken.routeKey = some minorTree at minorReconstructed
  change routeKeyTree? majorToken.routeKey = some majorTree at majorReconstructed
  refine ⟨.elim minor.FORMULA vertex.FORMULA majorTree minorTree,
    ?_, ?_, ?_, ?_⟩
  · simp [routeTokenTree?, routeKeyTree?, RouteIndexedMultiToken.mkToken,
      minorReconstructed, majorReconstructed]
  · simp [NDTree.wellFormedB, minorWellFormed, majorWellFormed,
      minorConclusion, majorConclusion, minorAt, majorAt, majorFormula]
  · rfl
  · rw [normalize_dependencies_elim, minorDependency, majorDependency]
    simp [RouteIndexedMultiToken.mkToken]

lemma routeOccurrenceTreeCorrect_elimReseed
    (d : Graph) (vertex major minor : Vertex)
    (minorEdge majorEdge : Deduction) (ancestor : Ancestral)
    (minorToken majorToken : RouteIndexedToken)
    (majorFormula : major.FORMULA = minor.FORMULA >> vertex.FORMULA)
    (minorAt : minorToken.vertex = minor)
    (majorAt : majorToken.vertex = major)
    (minorCorrect : RouteOccurrenceTreeCorrect minorToken)
    (majorCorrect : RouteOccurrenceTreeCorrect majorToken) :
    RouteOccurrenceTreeCorrect
      (mkToken d vertex
        (.elimReseed vertex major minor minorEdge majorEdge ancestor
          minorToken.routeKey majorToken.routeKey)
        (depUnion minorToken.dependency majorToken.dependency)
        ancestor.COLOURS) := by
  obtain ⟨minorTree, minorReconstructed, minorWellFormed,
    minorConclusion, minorDependency⟩ := minorCorrect
  obtain ⟨majorTree, majorReconstructed, majorWellFormed,
    majorConclusion, majorDependency⟩ := majorCorrect
  change routeKeyTree? minorToken.routeKey = some minorTree at minorReconstructed
  change routeKeyTree? majorToken.routeKey = some majorTree at majorReconstructed
  refine ⟨.elim minor.FORMULA vertex.FORMULA majorTree minorTree,
    ?_, ?_, ?_, ?_⟩
  · simp [routeTokenTree?, routeKeyTree?, RouteIndexedMultiToken.mkToken,
      minorReconstructed, majorReconstructed]
  · simp [NDTree.wellFormedB, minorWellFormed, majorWellFormed,
      minorConclusion, majorConclusion, minorAt, majorAt, majorFormula]
  · rfl
  · rw [normalize_dependencies_elim, minorDependency, majorDependency]
    simp [RouteIndexedMultiToken.mkToken]

lemma sourceOutputs_correct (d : Graph) (vertex : Vertex)
    (token : RouteIndexedToken) (member : token ∈ sourceOutputs d vertex) :
    RouteOccurrenceAtCorrect vertex token := by
  simp only [sourceOutputs] at member
  split at member
  · simp only [List.mem_singleton] at member
    subst token
    exact ⟨rfl, routeOccurrenceTreeCorrect_source d vertex none []⟩
  · rw [List.mem_map] at member
    obtain ⟨ancestor, _ancestorMember, rfl⟩ := member
    exact ⟨rfl,
      routeOccurrenceTreeCorrect_source d vertex (some ancestor)
        ancestor.COLOURS⟩

lemma assignSlotsFrom_correct (vertex : Vertex) :
    forall (tokens : List RouteIndexedToken) (ordinal : Nat),
      (forall token, token ∈ tokens -> RouteOccurrenceAtCorrect vertex token) ->
      forall token, token ∈ assignSlotsFrom vertex ordinal tokens ->
        RouteOccurrenceAtCorrect vertex token := by
  intro tokens
  induction tokens with
  | nil => simp [assignSlotsFrom]
  | cons head tail ih =>
      intro ordinal allCorrect token member
      simp only [assignSlotsFrom, List.mem_cons] at member
      rcases member with rfl | inTail
      · have headCorrect := allCorrect head (by simp)
        constructor
        · exact headCorrect.1
        · exact routeOccurrenceTreeCorrect_slot vertex ordinal head
            headCorrect.2
      · apply ih (ordinal + 1)
        · intro selection selectionMember
          exact allCorrect selection (by simp [selectionMember])
        · exact inTail

lemma assignSlots_correct (vertex : Vertex) (tokens : List RouteIndexedToken)
    (allCorrect : forall token, token ∈ tokens ->
      RouteOccurrenceAtCorrect vertex token) :
    forall token, token ∈ assignSlots vertex tokens ->
      RouteOccurrenceAtCorrect vertex token := by
  exact assignSlotsFrom_correct vertex tokens 0 allCorrect

lemma elimCombine_correct (d : Graph) (vertex major minor : Vertex)
    (minorTokens majorTokens : List RouteIndexedToken)
    (majorFormula : major.FORMULA = minor.FORMULA >> vertex.FORMULA)
    (minorCorrect : forall token, token ∈ minorTokens ->
      RouteOccurrenceAtCorrect minor token)
    (majorCorrect : forall token, token ∈ majorTokens ->
      RouteOccurrenceAtCorrect major token) :
    forall token, token ∈ RouteIndexedMultiToken.elimCombine d vertex major minor
      minorTokens majorTokens -> RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [RouteIndexedMultiToken.elimCombine, List.mem_flatMap] at member
  obtain ⟨minorToken, minorMember, majorWitness⟩ := member
  obtain ⟨majorToken, majorMember, minorEdgeWitness⟩ := majorWitness
  obtain ⟨minorEdge, minorEdgeMember, majorEdgeWitness⟩ := minorEdgeWitness
  rw [List.mem_filterMap] at majorEdgeWitness
  obtain ⟨majorEdge, majorEdgeMember, produced⟩ := majorEdgeWitness
  split at produced <;> try contradiction
  next minorTail minorConsume =>
    split at produced <;> try contradiction
    next majorTail majorConsume =>
      simp only [Option.some.injEq] at produced
      subst token
      have minorInvariant := minorCorrect minorToken minorMember
      have majorInvariant := majorCorrect majorToken majorMember
      exact ⟨rfl,
        routeOccurrenceTreeCorrect_elimination d vertex major minor
          minorEdge majorEdge minorToken majorToken majorFormula
          minorInvariant.1 majorInvariant.1 minorInvariant.2 majorInvariant.2
          []⟩

lemma elimOutputs_correct (d : Graph) (vertex : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token) :
    forall token, token ∈ RouteIndexedMultiToken.elimOutputs d vertex premiseTokens ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [RouteIndexedMultiToken.elimOutputs, List.mem_flatMap] at member
  obtain ⟨pair, pairMember, tokenMember⟩ := member
  rcases pair with ⟨major, minor⟩
  have majorFormula :
      major.FORMULA = minor.FORMULA >> vertex.FORMULA := by
    simp only [elimPairsAt, List.mem_flatMap] at pairMember
    obtain ⟨major', _majorPred, minorWitness⟩ := pairMember
    rw [List.mem_filterMap] at minorWitness
    obtain ⟨minor', _minorPred, produced⟩ := minorWitness
    split at produced <;> simp_all
  exact elimCombine_correct d vertex major minor
    (premiseTokens minor) (premiseTokens major) majorFormula
    (premiseCorrect minor) (premiseCorrect major) token tokenMember

lemma introOutputsFor_correct (d : Graph) (vertex premise : Vertex)
    (alpha beta : Formula) (tokens : List RouteIndexedToken)
    (vertexFormula : vertex.FORMULA = alpha >> beta)
    (premiseFormula : premise.FORMULA = beta)
    (tokensCorrect : forall token, token ∈ tokens ->
      RouteOccurrenceAtCorrect premise token) :
    forall token, token ∈ introOutputsFor d vertex premise alpha tokens ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [introOutputsFor, List.mem_flatMap] at member
  obtain ⟨premiseToken, premiseMember, edgeWitness⟩ := member
  rw [List.mem_filterMap] at edgeWitness
  obtain ⟨edge, _edgeMember, produced⟩ := edgeWitness
  cases consumed : consume edge premiseToken.residual with
  | none => simp [consumed] at produced
  | some tail =>
    simp only [consumed, Option.map_some, Option.some.injEq] at produced
    subst token
    have premiseInvariant := tokensCorrect premiseToken premiseMember
    exact ⟨rfl,
      routeOccurrenceTreeCorrect_introduction d vertex edge premiseToken
        alpha beta vertexFormula (by simpa [premiseInvariant.1] using premiseFormula)
        premiseInvariant.2 []⟩

lemma introOutputs_correct (d : Graph) (vertex : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token) :
    forall token, token ∈ introOutputs d vertex premiseTokens ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  unfold introOutputs at member
  cases predecessors : predsOf d vertex with
  | nil => simp [predecessors] at member
  | cons premise rest =>
      cases rest with
      | nil =>
          cases formula : vertex.FORMULA with
          | atom symbol => simp [predecessors, formula] at member
          | implication alpha beta =>
              simp only [predecessors, formula] at member
              split at member
              · next premiseFormula =>
                  exact introOutputsFor_correct d vertex premise alpha beta
                    (premiseTokens premise) formula premiseFormula
                    (premiseCorrect premise) token member
              · contradiction
      | cons second rest => simp [predecessors] at member

lemma mixedIntroOutputs_correct (d : Graph) (vertex : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token) :
    forall token, token ∈ mixedIntroOutputs d vertex premiseTokens ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  unfold mixedIntroOutputs at member
  cases formula : vertex.FORMULA with
  | atom symbol => simp [formula] at member
  | implication alpha beta =>
      simp only [formula, List.mem_flatMap] at member
      obtain ⟨premise, _premiseMember, tokenWitness⟩ := member
      split at tokenWitness
      · next guard =>
          have premiseFormula : premise.FORMULA = beta := by
            simp only [Bool.and_eq_true, beq_iff_eq] at guard
            exact guard.2
          simp only [List.mem_flatMap] at tokenWitness
          obtain ⟨premiseToken, premiseMember, edgeWitness⟩ := tokenWitness
          rw [List.mem_filterMap] at edgeWitness
          obtain ⟨edge, _edgeMember, produced⟩ := edgeWitness
          cases consumed : consume edge premiseToken.residual with
          | none => simp [consumed] at produced
          | some tail =>
              simp only [consumed, Option.map_some, Option.some.injEq] at produced
              subst token
              have premiseInvariant :=
                premiseCorrect premise premiseToken premiseMember
              exact ⟨rfl,
                routeOccurrenceTreeCorrect_mixedIntroduction d vertex edge
                  premiseToken alpha beta formula
                  (by simpa [premiseInvariant.1] using premiseFormula)
                  premiseInvariant.2 []⟩
      · contradiction

lemma elimPairsAt_formula (d : Graph) (vertex major minor : Vertex)
    (member : (major, minor) ∈ elimPairsAt d vertex) :
    major.FORMULA = minor.FORMULA >> vertex.FORMULA := by
  simp only [elimPairsAt, List.mem_flatMap] at member
  obtain ⟨major', _majorPred, minorWitness⟩ := member
  rw [List.mem_filterMap] at minorWitness
  obtain ⟨minor', _minorPred, produced⟩ := minorWitness
  split at produced <;> simp_all

lemma elimReseedOutputs_correct (d : Graph) (vertex : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (ancestors : List Ancestral)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token) :
    forall token,
      token ∈ elimReseedOutputs d vertex premiseTokens ancestors ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [elimReseedOutputs, List.mem_flatMap] at member
  obtain ⟨pair, pairMember, minorWitness⟩ := member
  rcases pair with ⟨major, minor⟩
  obtain ⟨minorToken, minorMember, majorWitness⟩ := minorWitness
  obtain ⟨majorToken, majorMember, minorEdgeWitness⟩ := majorWitness
  obtain ⟨minorEdge, _minorEdgeMember, majorEdgeWitness⟩ := minorEdgeWitness
  obtain ⟨majorEdge, _majorEdgeMember, ancestorWitness⟩ := majorEdgeWitness
  split at ancestorWitness
  · next guard =>
      rw [List.mem_map] at ancestorWitness
      obtain ⟨ancestor, _ancestorMember, produced⟩ := ancestorWitness
      subst token
      have minorInvariant := premiseCorrect minor minorToken minorMember
      have majorInvariant := premiseCorrect major majorToken majorMember
      exact ⟨rfl,
        routeOccurrenceTreeCorrect_elimReseed d vertex major minor
          minorEdge majorEdge ancestor minorToken majorToken
          (elimPairsAt_formula d vertex major minor pairMember)
          minorInvariant.1 majorInvariant.1 minorInvariant.2 majorInvariant.2⟩
  · contradiction

lemma introReseedOutputs_correct (d : Graph) (vertex : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (ancestors : List Ancestral)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token) :
    forall token,
      token ∈ introReseedOutputs d vertex premiseTokens ancestors ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  unfold introReseedOutputs at member
  cases predecessors : predsOf d vertex with
  | nil => simp [predecessors] at member
  | cons premise rest =>
      cases rest with
      | nil =>
          cases formula : vertex.FORMULA with
          | atom symbol => simp [predecessors, formula] at member
          | implication alpha beta =>
              simp only [predecessors, formula] at member
              split at member
              · next premiseFormula =>
                  simp only [List.mem_flatMap] at member
                  obtain ⟨premiseToken, premiseMember, edgeWitness⟩ := member
                  obtain ⟨edge, _edgeMember, ancestorWitness⟩ := edgeWitness
                  split at ancestorWitness
                  · rw [List.mem_map] at ancestorWitness
                    obtain ⟨ancestor, _ancestorMember, produced⟩ := ancestorWitness
                    subst token
                    have premiseInvariant :=
                      premiseCorrect premise premiseToken premiseMember
                    exact ⟨rfl,
                      routeOccurrenceTreeCorrect_introReseed d vertex edge
                        ancestor premiseToken alpha beta formula
                        (by simpa [premiseInvariant.1] using premiseFormula)
                        premiseInvariant.2⟩
                  · contradiction
              · contradiction
      | cons second rest => simp [predecessors] at member

lemma reseedOutputs_correct (d : Graph) (vertex : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token) :
    forall token, token ∈ reseedOutputs d vertex premiseTokens ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [reseedOutputs] at member
  split at member
  · contradiction
  · rw [List.mem_append] at member
    rcases member with elimination | introduction
    · exact elimReseedOutputs_correct d vertex premiseTokens
        (eligibleAncestorsAtRoot d vertex) premiseCorrect token elimination
    · exact introReseedOutputs_correct d vertex premiseTokens
        (eligibleAncestorsAtRoot d vertex) premiseCorrect token introduction

lemma rawStepFrom_correct (d : Graph)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token)
    (vertex : Vertex) :
    forall token, token ∈ rawStepFrom d premiseTokens vertex ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [rawStepFrom] at member
  split at member
  · exact sourceOutputs_correct d vertex token member
  · rw [List.mem_append] at member
    rcases member with coreMember | reseedMember
    · split at coreMember
      · exact introOutputs_correct d vertex premiseTokens premiseCorrect
          token coreMember
      · rw [List.mem_append] at coreMember
        rcases coreMember with elimination | introduction
        · exact elimOutputs_correct d vertex premiseTokens premiseCorrect
            token elimination
        · exact mixedIntroOutputs_correct d vertex premiseTokens premiseCorrect
            token introduction
    · exact reseedOutputs_correct d vertex premiseTokens premiseCorrect
        token reseedMember

lemma stepFrom_correct (d : Graph)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseCorrect : forall premise token,
      token ∈ premiseTokens premise -> RouteOccurrenceAtCorrect premise token)
    (vertex : Vertex) :
    forall token, token ∈ stepFrom d premiseTokens vertex ->
      RouteOccurrenceAtCorrect vertex token := by
  exact assignSlots_correct vertex (rawStepFrom d premiseTokens vertex)
    (rawStepFrom_correct d premiseTokens premiseCorrect vertex)




theorem routeIndexedAt_tree_correct (d : Graph) :
    forall fuel vertex token,
      token ∈ evaluateAt d fuel vertex ->
      RouteOccurrenceAtCorrect vertex token := by
  intro fuel
  induction fuel with
  | zero =>
      intro vertex token member
      simp [evaluateAt] at member
  | succ fuel ih =>
      intro vertex token member
      exact stepFrom_correct d (evaluateAt d fuel)
        (fun premise selection selectionMember =>
          ih premise selection selectionMember)
        vertex token member

theorem routeIndexedCanonical_tree_correct (d : Graph)
    (vertex : Vertex) (token : RouteIndexedToken)
    (member : token ∈ canonicalAt d vertex) :
    RouteOccurrenceAtCorrect vertex token :=
  routeIndexedAt_tree_correct d (stdFuel d) vertex token member

theorem routeIndexedCanonicalFast_tree_correct (d : Graph)
    (vertex : Vertex) (token : RouteIndexedToken)
    (member : token ∈ routeIndexedCanonicalFast d vertex) :
    RouteOccurrenceAtCorrect vertex token := by
  rw [routeIndexedCanonicalFast_eq_canonical] at member
  exact routeIndexedCanonical_tree_correct d vertex token member


theorem routeOccurrence_tree_dependencies (d : Graph)
    (vertex : Vertex) (token : RouteIndexedToken)
    (member : token ∈ routeIndexedCanonicalFast d vertex) :
    exists tree,
      routeTokenTree? token = some tree /\
      tree.wellFormedB = true /\
      tree.conclusion = vertex.FORMULA /\
      normalizeDep tree.dependencies = token.dependency := by
  have correct := routeIndexedCanonicalFast_tree_correct d vertex token member
  obtain ⟨atVertex, tree, reconstructed, wellFormed, conclusion, dependency⟩ :=
    correct
  exact ⟨tree, reconstructed, wellFormed,
    by simpa [atVertex] using conclusion, dependency⟩

lemma normalizeDep_eq_nil_iff (dependency : Dep) :
    normalizeDep dependency = [] ↔ dependency = [] := by
  constructor
  · intro normalized
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro formula member
    have normalizedMember : formula ∈ normalizeDep dependency :=
      mem_normalizeDep.mpr member
    simp [normalized] at normalizedMember
  · rintro rfl
    exact normalizeDep_nil





theorem encodeCertifiedNDTree_marked_iff_closed
    (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree)
    (wellFormed : tree.wellFormedB = true) :
    occurrenceMarkedB fs tree.conclusion tree.nodeCount
      (fs.length + tree.nodeCount)
      (encodeCertifiedNDTree fs tree covered) = true ↔
    tree.dependencies = [] := by
  constructor
  · intro marked
    let root := allocatedRoot fs tree.nodeCount 0 tree.nodeCount
      tree.nodeCount_pos (by simp)
    let column := formulaColumnOf fs covered.conclusion
    let result := evalOccurrence fs tree.nodeCount
      (fs.length + tree.nodeCount) (encodeCertifiedNDTree fs tree covered)
    have spec : CompilerRunSpec fs tree.nodeCount 0 tree
        (initialState fs (fs.length + tree.nodeCount) tree.nodeCount)
        root column result := by
      simpa [root, column, result] using
        encodeCertifiedNDTree_correct fs tree covered wellFormed
    have marked' :
        (!result.error &&
          (acceptedToken? fs tree.conclusion tree.nodeCount result.state).isSome) =
          true := by
      simpa [occurrenceMarkedB, result] using marked
    have foundSome :
        (acceptedToken? fs tree.conclusion tree.nodeCount result.state).isSome =
          true := by
      simp only [Bool.and_eq_true] at marked'
      exact marked'.2
    cases found : acceptedToken? fs tree.conclusion tree.nodeCount result.state with
    | none => simp [found] at foundSome
    | some accepted =>
        have acceptedSound := acceptedToken_sound fs tree.conclusion
          tree.nodeCount result.state accepted found
        have acceptedMember : accepted ∈ liveTokens result.state :=
          List.mem_of_find?_eq_some found
        simp only [liveTokens, List.mem_filterMap] at acceptedMember
        obtain ⟨carrier, _carrierMember, atCarrier⟩ := acceptedMember
        by_cases inPrivate : InAllocatedInterval fs 0 tree.nodeCount carrier
        · by_cases isRoot : carrier = root
          · subst carrier
            obtain ⟨rootLevel, rootAt⟩ := spec.root
            rw [rootAt] at atCarrier
            have tokenEq :
                BranchingGrid.mkToken fs root column rootLevel tree = accepted :=
              Option.some.inj atCarrier
            have treeEq : tree = accepted.tree := by
              simpa [BranchingGrid.mkToken] using
                congrArg BranchToken.tree tokenEq
            simpa [treeEq] using acceptedSound.2.2.1
          · have emptyAt := spec.cleared carrier inPrivate isRoot
            rw [emptyAt] at atCarrier
            contradiction
        · have carrierBelow : carrier.val < fs.length := by
            have carrierBound := carrier.isLt
            simp only [InAllocatedInterval] at inPrivate
            omega
          let sourceColumn : Fin fs.length :=
            ⟨carrier.val, carrierBelow⟩
          have sourceAt : CarriesAt fs
              (initialState fs (fs.length + tree.nodeCount) tree.nodeCount)
              carrier sourceColumn (.hyp (fs.get sourceColumn)) := by
            refine ⟨tree.nodeCount, ?_⟩
            simp [initialState, sourceColumn, carrierBelow,
              BranchingGrid.mkToken]
          have sourcePreserved := (spec.frame carrier inPrivate).1
            sourceColumn (.hyp (fs.get sourceColumn)) sourceAt
          obtain ⟨sourceLevel, finalSourceAt⟩ := sourcePreserved
          rw [atCarrier] at finalSourceAt
          have tokenEq :
              BranchingGrid.mkToken fs carrier sourceColumn sourceLevel
                (.hyp (fs.get sourceColumn)) = accepted :=
            Option.some.inj finalSourceAt.symm
          have acceptedTree : accepted.tree = .hyp (fs.get sourceColumn) := by
            simpa [BranchingGrid.mkToken] using
              (congrArg BranchToken.tree tokenEq).symm
          have emptyDependencies := acceptedSound.2.2.1
          rw [acceptedTree] at emptyDependencies
          simp [NDTree.dependencies] at emptyDependencies
  · intro closed
    exact occurrenceMarked_complete fs tree covered wellFormed closed

def treeFormulaCoverage (tree : NDTree) : FormulaCovered tree.formulas tree :=
  fun _ member => member




def FlowRouteGridCompleted (vertex : Vertex)
    (token : RouteIndexedToken) : Prop :=
  token.residual = [] ∧
    exists tree,
      routeTokenTree? token = some tree ∧
      tree.conclusion = vertex.FORMULA ∧
      occurrenceMarkedB tree.formulas vertex.FORMULA tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree
          (treeFormulaCoverage tree)) = true



theorem completedFlowOccurrence_iff_grid
    (d : Graph) (vertex : Vertex) (token : RouteIndexedToken)
    (member : token ∈ routeIndexedCanonicalFast d vertex) :
    (token.dependency = [] ∧ token.residual = []) ↔
      FlowRouteGridCompleted vertex token := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ :=
    routeOccurrence_tree_dependencies d vertex token member
  constructor
  · rintro ⟨tokenClosed, residualClosed⟩
    have normalizedClosed : normalizeDep tree.dependencies = [] := by
      simpa [tokenClosed] using dependency
    have treeClosed : tree.dependencies = [] :=
      (normalizeDep_eq_nil_iff tree.dependencies).mp normalizedClosed
    refine ⟨residualClosed, tree, reconstructed, conclusion, ?_⟩
    rw [← conclusion]
    exact occurrenceMarked_complete tree.formulas tree
      (treeFormulaCoverage tree) wellFormed treeClosed
  · rintro ⟨residualClosed, selection, selectionReconstructed,
      selectionConclusion, marked⟩
    have selectionEq : selection = tree := by
      rw [reconstructed] at selectionReconstructed
      exact (Option.some.inj selectionReconstructed).symm
    subst selection
    have treeClosed : tree.dependencies = [] := by
      rw [← encodeCertifiedNDTree_marked_iff_closed tree.formulas tree
        (treeFormulaCoverage tree) wellFormed]
      simpa [conclusion] using marked
    constructor
    · simpa [treeClosed] using dependency.symm
    · exact residualClosed

theorem completedRouteOccurrence_to_closedND
    (d : Graph) (root : Vertex) (token : RouteIndexedToken)
    (member : token ∈ routeIndexedCanonicalFast d root)
    (completed : token.dependency = [] ∧ token.residual = []) :
    exists tree,
      routeTokenTree? token = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = root.FORMULA ∧
      tree.dependencies = [] ∧
      Derives [] root.FORMULA := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, dependency⟩ :=
    routeOccurrence_tree_dependencies d root token member
  have normalizedClosed : normalizeDep tree.dependencies = [] := by
    simpa [completed.1] using dependency
  have treeClosed :=
    (normalizeDep_eq_nil_iff tree.dependencies).mp normalizedClosed
  exact ⟨tree, reconstructed, wellFormed, conclusion, treeClosed,
    extracted_certificate_sound wellFormed conclusion treeClosed⟩



theorem routeRootOccurrence_to_markedGrid
    (d : Graph) (root : Vertex) (token : RouteIndexedToken)
    (member : token ∈ routeIndexedCanonicalFast d root)
    (completed : token.dependency = [] ∧ token.residual = []) :
    exists tree,
      routeTokenTree? token = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = root.FORMULA ∧
      occurrenceMarkedB tree.formulas root.FORMULA tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree
          (treeFormulaCoverage tree)) = true := by
  obtain ⟨tree, reconstructed, wellFormed, conclusion, _dependency⟩ :=
    routeOccurrence_tree_dependencies d root token member
  have grid := (completedFlowOccurrence_iff_grid d root token member).mp completed
  obtain ⟨_residual, selection, selectionReconstructed,
    selectionConclusion, marked⟩ := grid
  rw [reconstructed] at selectionReconstructed
  have selectionEq := (Option.some.inj selectionReconstructed).symm
  subst selection
  exact ⟨tree, reconstructed, wellFormed, conclusion, marked⟩

def RouteIndexedCanonicalFastDischarged (d : Graph) : Prop :=
  forall root, root ∈ d.NODES ->
    (get_rule.outgoing root d).isEmpty = true ->
      routeIndexedCanonicalFast d root ≠ [] ∧
      forall token, token ∈ routeIndexedCanonicalFast d root ->
        token.dependency = [] ∧ token.residual = []




theorem rootsDischargedB_iff_routeIndexedCanonicalFastDischarged (d : Graph) :
    rootsDischargedB d = true ↔ RouteIndexedCanonicalFastDischarged d := by
  unfold rootsDischargedB RouteIndexedCanonicalFastDischarged
  rw [List.all_eq_true]
  constructor
  · intro roots root rootMember rootIs
    have rootCheck := roots root rootMember
    simp only [rootIs, Bool.not_true, Bool.false_or,
      Bool.and_eq_true, List.all_eq_true] at rootCheck
    have correspondence := routeIndexedCanonicalFast_setEq_flowAt d root
    constructor
    · intro tokensEmpty
      have projectedEmpty :
          projectFlowPairs (routeIndexedCanonicalFast d root) = [] := by
        simp [tokensEmpty, projectFlowPairs]
      have flowEmpty : flowAt d (stdFuel d) root = [] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro route routeMember
        have projectedMember := (correspondence route).mpr routeMember
        simp [projectedEmpty] at projectedMember
      have flowNonempty : flowAt d (stdFuel d) root ≠ [] := by
        simpa [List.isEmpty_iff] using rootCheck.1
      exact flowNonempty flowEmpty
    · intro token tokenMember
      have projectedMember : token.flowPair ∈
          projectFlowPairs (routeIndexedCanonicalFast d root) := by
        exact List.mem_map.mpr ⟨token, tokenMember, rfl⟩
      have flowMember := (correspondence token.flowPair).mp projectedMember
      have empty := rootCheck.2 token.flowPair flowMember
      simpa [RouteIndexedToken.flowPair, List.isEmpty_iff] using empty
  · intro routed root rootMember
    by_cases rootIs : (get_rule.outgoing root d).isEmpty = true
    · simp only [rootIs, Bool.not_true, Bool.false_or,
        Bool.and_eq_true, List.all_eq_true]
      have routeCheck := routed root rootMember rootIs
      have correspondence := routeIndexedCanonicalFast_setEq_flowAt d root
      constructor
      · have flowNonempty : flowAt d (stdFuel d) root ≠ [] := by
          intro flowEmpty
          apply routeCheck.1
          apply List.eq_nil_iff_forall_not_mem.mpr
          intro token tokenMember
          have projectedMember : token.flowPair ∈
              projectFlowPairs (routeIndexedCanonicalFast d root) :=
            List.mem_map.mpr ⟨token, tokenMember, rfl⟩
          have flowMember := (correspondence token.flowPair).mp projectedMember
          simp [flowEmpty] at flowMember
        simpa [List.isEmpty_iff] using flowNonempty
      · intro route flowMember
        have projectedMember := (correspondence route).mpr flowMember
        change route ∈
          (routeIndexedCanonicalFast d root).map RouteIndexedToken.flowPair
          at projectedMember
        rw [List.mem_map] at projectedMember
        obtain ⟨token, tokenMember, tokenProjection⟩ := projectedMember
        have empty := routeCheck.2 token tokenMember
        subst route
        simp [RouteIndexedToken.flowPair, empty]
    · have rootNotEmpty : (get_rule.outgoing root d).isEmpty = false :=
        Bool.eq_false_of_not_eq_true rootIs
      simp [rootNotEmpty]



def FlowCompiledGridValid (d : Graph) : Prop :=
  forall root, root ∈ d.NODES ->
    (get_rule.outgoing root d).isEmpty = true ->
      routeIndexedCanonicalFast d root ≠ [] ∧
      forall token, token ∈ routeIndexedCanonicalFast d root ->
        FlowRouteGridCompleted root token

theorem routeIndexedDischarged_iff_flowCompiledGridValid (d : Graph) :
    RouteIndexedCanonicalFastDischarged d ↔ FlowCompiledGridValid d := by
  constructor
  · intro discharged root rootMember rootIs
    have rootCheck := discharged root rootMember rootIs
    refine ⟨rootCheck.1, ?_⟩
    intro token tokenMember
    exact (completedFlowOccurrence_iff_grid d root token tokenMember).mp
      (rootCheck.2 token tokenMember)
  · intro compiled root rootMember rootIs
    have rootCheck := compiled root rootMember rootIs
    refine ⟨rootCheck.1, ?_⟩
    intro token tokenMember
    exact (completedFlowOccurrence_iff_grid d root token tokenMember).mpr
      (rootCheck.2 token tokenMember)


theorem rootsDischargedB_iff_flowCompiledGridValid (d : Graph) :
    rootsDischargedB d = true ↔ FlowCompiledGridValid d :=
  (rootsDischargedB_iff_routeIndexedCanonicalFastDischarged d).trans
    (routeIndexedDischarged_iff_flowCompiledGridValid d)

theorem flowCompiledGridValid_of_strongFormation
    (d : Graph) (certificate : Certificate)
    (valid : StrongFormationValid d certificate) :
    FlowCompiledGridValid d :=
  (rootsDischargedB_iff_flowCompiledGridValid d).mp
    (strongFormationValid_rootsDischargedB d certificate valid)




theorem flowRootPair_to_gridWitness
    (d : Graph) (root : Vertex) (route : FlowPair)
    (member : route ∈ flowAt d (stdFuel d) root)
    (completed : route.1 = [] ∧ route.2 = []) :
    exists token tree,
      token ∈ routeIndexedCanonicalFast d root ∧
      token.flowPair = route ∧
      routeTokenTree? token = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = root.FORMULA ∧
      occurrenceMarkedB tree.formulas root.FORMULA tree.nodeCount
        (tree.formulas.length + tree.nodeCount)
        (encodeCertifiedNDTree tree.formulas tree
          (treeFormulaCoverage tree)) = true := by
  have correspondence := routeIndexedCanonicalFast_setEq_flowAt d root
  have projectedMember := (correspondence route).mpr member
  change route ∈
    (routeIndexedCanonicalFast d root).map RouteIndexedToken.flowPair
    at projectedMember
  rw [List.mem_map] at projectedMember
  obtain ⟨token, tokenMember, projection⟩ := projectedMember
  have tokenCompleted : token.dependency = [] ∧ token.residual = [] := by
    rw [← projection] at completed
    simpa [RouteIndexedToken.flowPair] using completed
  obtain ⟨tree, reconstructed, wellFormed, conclusion, marked⟩ :=
    routeRootOccurrence_to_markedGrid d root token tokenMember tokenCompleted
  exact ⟨token, tree, tokenMember, projection, reconstructed,
    wellFormed, conclusion, marked⟩


end FlowBranchingGridBridge
