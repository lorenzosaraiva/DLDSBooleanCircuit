import Semantic.MultiIntroReadingLineage
import Semantic.OccurrenceReadingsLineageND

open Semantic FlowSpec RouteIndexedMultiToken FlowBranchingGridBridge
open GridCertificateSoundness
namespace MultiIntroReadingLineage

theorem ordinary_correct (d : Graph) (v : Vertex) (F : Vertex → List RouteIndexedToken)
    (correct : ∀ u token, token ∈ F u → RouteOccurrenceAtCorrect u token)
    (edge : Deduction) (token : RouteIndexedToken) (member : token ∈ ordinary d v F edge) :
    RouteOccurrenceAtCorrect v token := by
  cases formula : v.FORMULA with
  | atom symbol => simp [ordinary, formula] at member
  | implication alpha beta =>
    simp only [ordinary, formula] at member
    split at member
    · rename_i premise
      obtain ⟨child, childMember, produced⟩ := List.mem_filterMap.mp member
      split at produced <;> try contradiction
      cases consumed : consume edge child.residual with
      | none => simp [consumed] at produced
      | some tail =>
        simp only [consumed, Option.map_some, Option.some.injEq] at produced
        subst token
        have childCorrect := correct edge.START child childMember
        exact ⟨rfl, routeOccurrenceTreeCorrect_introduction d v edge child alpha beta formula
          (by simpa [childCorrect.1] using premise) childCorrect.2 tail⟩
    · contradiction

theorem reseed_correct (variant : MultiIntro.Variant) (d : Graph) (v : Vertex)
    (F : Vertex → List RouteIndexedToken)
    (correct : ∀ u token, token ∈ F u → RouteOccurrenceAtCorrect u token)
    (edge : Deduction) (token : RouteIndexedToken) (member : token ∈ reseed variant d v F edge) :
    RouteOccurrenceAtCorrect v token := by
  cases formula : v.FORMULA with
  | atom symbol => simp [reseed, formula] at member
  | implication alpha beta =>
    simp only [reseed, formula] at member
    split at member
    · rename_i premise
      obtain ⟨child, childMember, produced⟩ := List.mem_flatMap.mp member
      split_ifs at produced
      · obtain ⟨ancestor, _, equal⟩ := List.mem_map.mp produced
        subst token
        have childCorrect := correct edge.START child childMember
        exact ⟨rfl, routeOccurrenceTreeCorrect_introReseed d v edge ancestor child alpha beta formula
          (by simpa [childCorrect.1] using premise) childCorrect.2⟩
      · contradiction
    · contradiction

theorem raw_correct (variant : MultiIntro.Variant) (d : Graph)
    (F : Vertex → List RouteIndexedToken)
    (correct : ∀ u token, token ∈ F u → RouteOccurrenceAtCorrect u token)
    (v : Vertex) (token : RouteIndexedToken) (member : token ∈ rawStepFrom variant d F v) :
    RouteOccurrenceAtCorrect v token := by
  unfold rawStepFrom at member
  split at member
  · obtain ⟨edge, _, produced⟩ := List.mem_flatMap.mp member
    rcases List.mem_append.mp produced with direct | ancestral
    · exact ordinary_correct d v F correct edge token direct
    · exact reseed_correct variant d v F correct edge token ancestral
  · exact TaggedLineage.rawStepFrom_correct d F correct v token member

theorem step_correct (variant : MultiIntro.Variant) (d : Graph)
    (F : Vertex → List RouteIndexedToken)
    (correct : ∀ u token, token ∈ F u → RouteOccurrenceAtCorrect u token)
    (v : Vertex) : ∀ token, token ∈ stepFrom variant d F v → RouteOccurrenceAtCorrect v token := by
  exact assignSlots_correct v (rawStepFrom variant d F v) (raw_correct variant d F correct v)

theorem evaluateAt_tree_correct (variant : MultiIntro.Variant) (d : Graph) :
    ∀ fuel v token, token ∈ evaluateAt variant d fuel v → RouteOccurrenceAtCorrect v token := by
  intro fuel
  induction fuel with
  | zero => intro v token member; simp [evaluateAt] at member
  | succ fuel ih =>
    intro v token member
    exact step_correct variant d (evaluateAt variant d fuel) ih v token member

theorem route_tree (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (token : RouteIndexedToken) (member : token ∈ evaluateAt variant d fuel v) :
    ∃ tree, routeKeyTree? token.routeKey = some tree ∧ tree.wellFormedB = true ∧
      tree.conclusion = v.FORMULA ∧ normalizeDep tree.dependencies = token.dependency := by
  obtain ⟨atVertex, tree, reconstructed, wellFormed, conclusion, dependency⟩ :=
    evaluateAt_tree_correct variant d fuel v token member
  exact ⟨tree, reconstructed, wellFormed, by simpa [atVertex] using conclusion, dependency⟩

#print axioms evaluateAt_tree_correct
#print axioms route_tree
end MultiIntroReadingLineage
