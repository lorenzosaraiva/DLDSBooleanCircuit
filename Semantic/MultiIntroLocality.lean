import Semantic.MultiIntroModel

open Semantic FlowSpec
namespace MultiIntro
set_option maxHeartbeats 2000000

theorem incoming_start_pred (d : Graph) (v : Vertex) (e : Deduction)
    (member : e ∈ get_rule.incoming v d) : e.START ∈ predsOf d v := by
  unfold predsOf
  rw [List.mem_eraseDups, List.mem_map]
  exact ⟨e, List.mem_filter.mpr ⟨mem_incoming_mem_edges v d member,
    by simp [mem_incoming_end_eq v d member]⟩, rfl⟩

theorem step_congr (variant : Variant) (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (agree : ∀ u ∈ predsOf d v, left u = right u) :
    step variant d left v = step variant d right v := by
  unfold step
  split
  · apply congrArg List.eraseDups
    apply List.flatMap_congr
    intro e member
    have same := agree e.START (incoming_start_pred d v e member)
    simp only [ordinaryEdge, reseedEdge, same]
  · exact CompleteProfileKernel.tagged.congr d v left right agree

def kernel (variant : Variant) : CompleteProfileKernel.Kernel where
  step := step variant
  congr := step_congr variant

theorem kernel_evaluate (variant : Variant) (d : Graph) : ∀ fuel v,
    CompleteProfileKernel.evaluateAt (kernel variant) d fuel v = evaluateAt variant d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
    intro v
    change step variant d (CompleteProfileKernel.evaluateAt (kernel variant) d fuel) v =
      step variant d (evaluateAt variant d fuel) v
    rw [funext ih]

theorem roots_kernel_eq (variant : Variant) (d : Graph) :
    CompleteProfileKernel.rootsB (kernel variant) d = rootsB variant d := by
  simp only [CompleteProfileKernel.rootsB, rootsB, kernel_evaluate]

end MultiIntro
