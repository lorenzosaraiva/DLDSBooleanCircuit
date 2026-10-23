import Infrastructure.SpineGuardFreeProjection
import Infrastructure.OccurrenceCertificateSoundness

open Semantic FlowSpec SelfCertifying CertificateModel RouteIndexedMultiToken
open FlowBranchingGridBridge GridCertificateSoundness

namespace SpineLineage






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
        (fun premise selection selectionMember =>
          ih premise selection selectionMember)
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
end SpineLineage

namespace SpineND




theorem guardFreeFlowAt_tree_dependencies (d : Graph) (fuel : Nat)
    (v : Vertex) (route : FlowPair)
    (member : route ∈ LambdaFlow.guardFreeFlowAt d fuel v) :
    ∃ tree : NDTree, tree.wellFormedB = true ∧
      tree.conclusion = v.FORMULA ∧ normalizeDep tree.dependencies = route.1 := by
  have projected := (SpineLineage.evaluateAt_project d fuel v route).mpr member
  obtain ⟨token, tokenMember, projection⟩ := List.mem_map.mp projected
  obtain ⟨atVertex, tree, _reconstructed, wellFormed, conclusion, dependency⟩ :=
    SpineLineage.evaluateAt_tree_correct d fuel v token tokenMember
  refine ⟨tree, wellFormed, ?_, ?_⟩
  · simpa [atVertex] using conclusion
  · have depProjection := congrArg Prod.fst projection
    exact dependency.trans depProjection


theorem guardFreeFlowAt_dependency_canonical (d : Graph) (fuel : Nat)
    (v : Vertex) (route : FlowPair)
    (member : route ∈ LambdaFlow.guardFreeFlowAt d fuel v) :
    CanonicalDep route.1 := by
  obtain ⟨tree, _wellFormed, _conclusion, dependency⟩ :=
    guardFreeFlowAt_tree_dependencies d fuel v route member
  rw [← dependency]
  exact normalizeDep_canonical _


theorem guardFreeFlowAt_closed_derivable (d : Graph) (fuel : Nat)
    (v : Vertex) (route : FlowPair)
    (member : route ∈ LambdaFlow.guardFreeFlowAt d fuel v)
    (closed : route.1 = []) : Derives [] v.FORMULA := by
  obtain ⟨tree, wellFormed, conclusion, dependency⟩ :=
    guardFreeFlowAt_tree_dependencies d fuel v route member
  have treeClosed := (normalizeDep_eq_nil_iff tree.dependencies).mp
    (dependency.trans closed)
  exact extracted_certificate_sound wellFormed conclusion treeClosed


theorem rootsDischargedB_roots_derivable (d : Graph)
    (verdict : rootsDischargedB d = true) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA := by
  intro root rootMember rootIs
  obtain ⟨nonempty, completed⟩ :=
    (rootsDischargedB_iff_routeIndexedCanonicalFastDischarged d).mp verdict
      root rootMember rootIs
  obtain ⟨token, member⟩ := List.exists_mem_of_ne_nil _ nonempty
  obtain ⟨_tree, _reconstructed, _wellFormed, _conclusion, _closed, derivation⟩ :=
    completedRouteOccurrence_to_closedND d root token member (completed token member)
  exact derivation


theorem guardFreeRootsB_roots_derivable (d : Graph)
    (verdict : LambdaFlow.guardFreeRootsB d = true) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA := by
  intro root rootMember rootIs
  rw [LambdaFlow.guardFreeRootsB_eq_raw, List.all_eq_true] at verdict
  have checked := verdict root rootMember
  simp only [rootIs, Bool.not_true, Bool.false_or, Bool.and_eq_true,
    List.all_eq_true] at checked
  have nonempty : LambdaFlow.guardFreeFlowAt d (stdFuel d) root ≠ [] := by
    simpa using checked.1
  obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ nonempty
  have closed : route.1 = [] := by
    have check := checked.2 route member
    simpa [List.isEmpty_iff] using check.1
  exact guardFreeFlowAt_closed_derivable d (stdFuel d) root route member closed


theorem strongFormationValidOccurrence_roots_derivable (d : Graph)
    (c : OccurrenceCertificate.OccurrenceCertificate)
    (valid : OccurrenceCertificate.StrongFormationValidOccurrence d c) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA :=
  guardFreeRootsB_roots_derivable d
    (OccurrenceCertificate.strongFormationValidOccurrence_guardFreeRootsB d c valid)



theorem keyedAccepted_roots_derivable (d : Graph)
    (verdict : LambdaKeyed.canonicalAcceptedB d = true) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA := by
  apply guardFreeRootsB_roots_derivable d
  simpa only [FaithfulKeyed.canonicalAcceptedB_eq_guardFreeRootsB] using verdict

#print axioms SpineLineage.evaluateAt_tree_correct
#print axioms guardFreeFlowAt_tree_dependencies
#print axioms rootsDischargedB_roots_derivable
#print axioms guardFreeRootsB_roots_derivable
#print axioms strongFormationValidOccurrence_roots_derivable
#print axioms keyedAccepted_roots_derivable
end SpineND
