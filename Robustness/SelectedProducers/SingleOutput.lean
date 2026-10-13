import Robustness.SelectedProducers.Selected

/-! Every selected branch recomputes at most one output, including bad indices. -/
open DLDS FlowSemantics
namespace Robustness.SelectedProducers

theorem pick_thin {α : Type} (xs : List α) (index : Nat) :
    (pick xs index).length ≤ 1 := by
  cases choice : xs[index]? <;> simp [pick, choice]

private theorem thin_bind {α β : Type} {xs : List α} {f : α → List β}
    (thin : xs.length ≤ 1) (images : ∀ x ∈ xs, (f x).length ≤ 1) :
    (xs.flatMap f).length ≤ 1 := by
  cases xs with
  | nil => simp
  | cons x rest =>
    have empty : rest = [] := by
      have zero : rest.length = 0 := by simp only [List.length_cons] at thin; omega
      simpa using zero
    subst rest
    simpa using images x (by simp)

private theorem thin_map {α β : Type} {xs : List α}
    (thin : xs.length ≤ 1) (f : α → β) : (xs.map f).length ≤ 1 := by
  simpa using thin

private theorem thin_filterMap {α β : Type} {xs : List α}
    (thin : xs.length ≤ 1) (f : α → Option β) : (xs.filterMap f).length ≤ 1 :=
  (List.length_filterMap_le f xs).trans thin

set_option maxHeartbeats 2000000 in
theorem selectedRows_atMostOne (d : Graph) (P : Vertex → List FlowPair)
    (v : Vertex) (kind : SelectedKind) :
    (selectedRows d P v kind).length ≤ 1 := by
  cases kind <;> simp only [selectedRows, chosenElim, chosenIntro, chosenMixed,
    chosenElimReseed, chosenIntroReseed]
  all_goals
    repeat' first
    | exact thin_map (pick_thin _ _) _
    | exact thin_filterMap (pick_thin _ _) _
    | apply thin_bind (pick_thin _ _); intro x hx
    | split
    | simp only [List.length_nil, List.length_cons]; omega

end Robustness.SelectedProducers
