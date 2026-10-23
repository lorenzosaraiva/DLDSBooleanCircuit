import Semantic.OccurrenceReadingsLineageProjection

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe RouteIndexedMultiToken
open FlowBranchingGridBridge GridCertificateSoundness

namespace TaggedLineage






lemma elimCombine_correct (d : Graph) (vertex major minor : Vertex)
    (minorTokens majorTokens : List RouteIndexedToken)
    (majorFormula : major.FORMULA = minor.FORMULA >> vertex.FORMULA)
    (minorCorrect : forall token, token ∈ minorTokens ->
      RouteOccurrenceAtCorrect minor token)
    (majorCorrect : forall token, token ∈ majorTokens ->
      RouteOccurrenceAtCorrect major token) :
    forall token, token ∈ TaggedLineage.elimCombine d vertex major minor
      minorTokens majorTokens -> RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [TaggedLineage.elimCombine, List.mem_flatMap] at member
  obtain ⟨minorToken, minorMember, majorWitness⟩ := member
  obtain ⟨majorToken, majorMember, minorEdgeWitness⟩ := majorWitness
  obtain ⟨minorEdge, minorEdgeMember, majorEdgeWitness⟩ := minorEdgeWitness
  rw [List.mem_filterMap] at majorEdgeWitness
  obtain ⟨majorEdge, majorEdgeMember, produced⟩ := majorEdgeWitness
  split at produced <;> try contradiction
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
    forall token, token ∈ TaggedLineage.elimOutputs d vertex premiseTokens ->
      RouteOccurrenceAtCorrect vertex token := by
  intro token member
  simp only [TaggedLineage.elimOutputs, List.mem_flatMap] at member
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
  split at produced <;> try contradiction
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
          split at produced <;> try contradiction
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




theorem evaluateAt_tree_correct (d : Graph) :
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
        (fun premise candidate candidateMember =>
          ih premise candidate candidateMember)
        vertex token member



theorem routeOccurrence_tree_dependencies (d : Graph) (fuel : Nat)
    (vertex : Vertex) (token : RouteIndexedToken)
    (member : token ∈ evaluateAt d fuel vertex) :
    ∃ tree, routeTokenTree? token = some tree ∧ tree.wellFormedB = true ∧
      tree.conclusion = vertex.FORMULA ∧
      normalizeDep tree.dependencies = token.dependency := by
  obtain ⟨atVertex, tree, reconstructed, wellFormed, conclusion, dependency⟩ :=
    evaluateAt_tree_correct d fuel vertex token member
  exact ⟨tree, reconstructed, wellFormed,
    by simpa [atVertex] using conclusion, dependency⟩
#print axioms evaluateAt_tree_correct
#print axioms routeOccurrence_tree_dependencies
end TaggedLineage
