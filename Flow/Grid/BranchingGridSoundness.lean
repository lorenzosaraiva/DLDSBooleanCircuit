import Flow.Grid.BranchingGrid

open scoped Classical

namespace Semantic
namespace BranchingGrid

open GridCertificateSoundness

theorem dependencySupportedB_sound (fs : List Formula) (tree : NDTree)
    (supported : dependencySupportedB fs tree = true) :
    forall formula, formula ∈ tree.dependencies -> formula ∈ fs := by
  intro formula member
  have hall := List.all_eq_true.mp supported
  have contained := hall formula member
  simpa using contained

theorem tokenValidB_sound (fs : List Formula) {K : Nat}
    (token : BranchToken fs.length K)
    (valid : tokenValidB fs token = true) :
    And (token.tree.wellFormedB = true)
      (And (fs.get token.formulaColumn = token.tree.conclusion)
        (And (token.dependency = depVector fs token.tree.dependencies)
          (forall formula, formula ∈ token.tree.dependencies -> formula ∈ fs))) := by
  simp only [tokenValidB, Bool.and_eq_true, beq_iff_eq] at valid
  exact And.intro valid.1.1.1
    (And.intro valid.1.1.2
      (And.intro valid.1.2
        (dependencySupportedB_sound fs token.tree valid.2)))

theorem tokenAcceptsB_sound (fs : List Formula) (goal : Formula)
    (height : Nat) {K : Nat} (token : BranchToken fs.length K)
    (accepts : tokenAcceptsB fs goal height token = true) :
    And (token.tree.wellFormedB = true)
      (And (token.tree.conclusion = goal)
        (And (token.tree.dependencies = [])
          (token.tree.height <= height))) := by
  simp only [tokenAcceptsB, Bool.and_eq_true, beq_iff_eq,
    decide_eq_true_eq] at accepts
  have valid := tokenValidB_sound fs token accepts.1.1.1
  have dependencyEq := valid.2.2.1
  have support := valid.2.2.2
  have empty : token.tree.dependencies = [] :=
    dependencies_empty_of_depVector_all_false fs token.tree.dependencies
      support (by simpa [dependencyEq] using accepts.1.2)
  exact And.intro valid.1
    (And.intro accepts.1.1.2 (And.intro empty accepts.2))

theorem acceptedToken_sound (fs : List Formula) (goal : Formula)
    (height : Nat) {K : Nat} (state : BranchState fs.length K)
    (token : BranchToken fs.length K)
    (found : acceptedToken? fs goal height state = some token) :
    And (token.tree.wellFormedB = true)
      (And (token.tree.conclusion = goal)
        (And (token.tree.dependencies = [])
          (token.tree.height <= height))) := by
  have accepts : tokenAcceptsB fs goal height token = true :=
    List.find?_some found
  exact tokenAcceptsB_sound fs goal height token accepts




theorem occurrenceMarked_extracts
    (fs : List Formula) (goal : Formula) (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K)
    (marked : occurrenceMarkedB fs goal height K assignment = true) :
    Exists fun tree =>
      And (extractOccurrenceGoalCone fs goal height K assignment = some tree)
        (And (tree.wellFormedB = true)
          (And (tree.conclusion = goal)
            (And (tree.dependencies = []) (tree.height <= height)))) := by
  let result := evalOccurrence fs height K assignment
  have marked' :
      And ((!result.error) = true)
        ((acceptedToken? fs goal height result.state).isSome = true) := by
    simpa [occurrenceMarkedB, result, Bool.and_eq_true] using marked
  cases found : acceptedToken? fs goal height result.state with
  | none => simp [found] at marked'
  | some token =>
      have sound := acceptedToken_sound fs goal height result.state token found
      refine Exists.intro token.tree ?_
      refine And.intro ?_ sound
      simp [extractOccurrenceGoalCone, result, found]

theorem occurrenceMarked_sound
    (fs : List Formula) (goal : Formula) (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K)
    (marked : occurrenceMarkedB fs goal height K assignment = true) :
    Derives [] goal := by
  obtain ⟨tree, _extracted, wellFormed, conclusion, dependencies, _height⟩ :=
    occurrenceMarked_extracts fs goal height K assignment marked
  exact extracted_certificate_sound wellFormed conclusion dependencies



theorem branchingGridMarked_sound
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height K : Nat)
    (assignment : BranchingAssignment fs height K)
    (marked : branchingGridMarkedB fs goal height K assignment = true) :
    Derives [] goal := by
  cases assignment with
  | inl old =>
      exact GridCertificateSoundness.gridMarked_sound
        fs hfs goal hgoal height old.paths marked
  | inr occurrence =>
      exact occurrenceMarked_sound fs goal height K occurrence marked


theorem branchingGridMarked_extracts
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height K : Nat)
    (assignment : BranchingAssignment fs height K)
    (marked : branchingGridMarkedB fs goal height K assignment = true) :
    Exists fun tree =>
      And (extractBranchingGoalCone fs goal height K assignment = some tree)
        (And (tree.wellFormedB = true)
          (And (tree.conclusion = goal)
            (And (tree.dependencies = []) (tree.height <= height)))) := by
  cases assignment with
  | inl old =>
      obtain ⟨tree, extracted, wellFormed, conclusion, dependencies, treeHeight⟩ :=
        GridCertificateSoundness.gridMarked_extracts
          fs hfs goal hgoal height old.paths marked
      exact ⟨tree, extracted, wellFormed, conclusion, dependencies, treeHeight⟩
  | inr occurrence =>
      exact occurrenceMarked_extracts fs goal height K occurrence marked


end BranchingGrid
end Semantic
