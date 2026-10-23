import Semantic.Circuit.GraphPlanBudget
import Semantic.Circuit.SharedResources

open FlowSpec
namespace Semantic.Circuit
open Shared
set_option maxHeartbeats 50000000

def productionAssignmentBound (R F I g : Nat) : Nat :=
  allocationUnit g F I * (4 * productionMass R g + 20 * productionMass R g ^ 2) +
    orderedWorkBound (4 * productionMass R g) R (1 + (F + I))

def productionWireBound (R F I g : Nat) : Nat :=
  g * (R * (1 + (F + I))) + productionAssignmentBound R F I g + 1

def productionGateBound (R F I g : Nat) : Nat := 4 * productionAssignmentBound R F I g + 1

theorem readCandidates_length (d : Graph) (v : Vertex) (R F I : Nat) (u : Vertex) :
    (readCandidates d v R F I u).length = if u ∈ productionReads d v then R else 0 := by
  unfold readCandidates
  split <;> simp_all [profileCandidates]

theorem readCandidates_zeroWork (d : Graph) (v : Vertex) (R F I : Nat) (u : Vertex)
    (entry : Candidate (productionInputCount d v R F I) (1 + (F + I)))
    (member : entry ∈ readCandidates d v R F I u) : ZeroWork entry := by
  unfold readCandidates at member
  split at member
  · obtain ⟨slot, _, rfl⟩ := List.mem_map.mp member
    exact ⟨rfl, fun _ => rfl⟩
  · simp at member

theorem readCandidates_work (d : Graph) (v : Vertex) (R F I : Nat) (u : Vertex) (code : Store) :
    (candidates (productionInputCount d v R F I) (readCandidates d v R F I u) code).2.size = code.size := by
  rw [candidates_work]
  have each : (readCandidates d v R F I u).map candidateWork =
      List.replicate (readCandidates d v R F I u).length 0 := by
    apply List.map_eq_replicate_iff.mpr
    intro entry member
    have z := readCandidates_zeroWork d v R F I u entry member
    simp [candidateWork, z.1, z.2]
  rw [each]; simp

theorem production_assignments_le (d : Graph) (v : Vertex) (R F I : Nat) :
    (productionAssemblyShared d v R F I).2.size ≤ productionAssignmentBound R F I (graphSize d) := by
  unfold productionAssemblyShared
  cases built : productionPlan (productionBasis d F) d v with
  | none =>
    simp only [expression, emit, Array.size_push, Array.size_empty]
    have positive : 0 < (R + 1) * (2 * R + 1) := by positivity
    dsimp [productionAssignmentBound, orderedWorkBound]
    omega
  | some plan =>
    let inputs := readCandidates d v R F I
    let source := compilePlan inputs plan #[]
    let stored := candidates (productionInputCount d v R F I) (inputs v) source.2
    have budget := production_budget (productionBasis d F) d v R plan built
    have monotone := planBudget_mono plan (fun u => (inputs u).length) (fun _ => R)
      (by intro u; simp only [inputs, readCandidates_length]; split <;> omega)
    have allocated := compilePlan_allocation plan inputs (graphSize d) budget.2.2 (graph_suffix_table_le d)
      (readCandidates_zeroWork d v R F I) #[]
    have sourceLength : source.1.length ≤ 4 * productionMass R (graphSize d) := by
      rw [allocated.1]; exact monotone.1.trans budget.1
    have sourceWork : source.2.size ≤ allocationUnit (graphSize d) F I *
        (4 * productionMass R (graphSize d) + 20 * productionMass R (graphSize d) ^ 2) := by
      have h := Nat.mul_le_mul_right (allocationUnit (graphSize d) F I) (monotone.2.trans budget.2.1)
      have a := allocated.2
      simp only [Array.size_empty, Nat.zero_add] at a
      exact a.trans (by simpa [Nat.mul_comm] using h)
    have storedLength : stored.1.length = R := by
      simp only [stored, candidates_length, inputs]
      rw [readCandidates_length, if_pos (by simp [productionReads])]
    have storedWork : stored.2.size = source.2.size := readCandidates_work d v R F I v source.2
    have sourceBounds := compilePlan_bounds plan inputs #[]
    have storedBounds := candidates_correct (productionInputCount d v R F I) (inputs v) source.2
      (Nat.le_add_right (productionInputCount d v R F I) source.2.size) (fun _ => false)
    have ordered := orderedTable_work (productionInputCount d v R F I) source.1 stored.1 stored.2
      (Bounded.weaken sourceBounds.2 (Nat.add_le_add_left storedWork.ge _)) storedBounds.2.1
    change (orderedTable (productionInputCount d v R F I) source.1 stored.1 stored.2).2.size ≤ _
    conv_rhs at ordered => rw [storedLength, storedWork]
    have table : orderedWorkBound source.1.length R (1 + (F + I)) ≤
        orderedWorkBound (4 * productionMass R (graphSize d)) R (1 + (F + I)) := by
      unfold orderedWorkBound
      gcongr
    exact ordered.trans (by dsimp [productionAssignmentBound]; omega)

theorem production_polynomial_size (d : Graph) (v : Vertex) (R F I : Nat) :
    (productionResourcesShared d v R F I).1 ≤ productionWireBound R F I (graphSize d) ∧
    (productionResourcesShared d v R F I).2 ≤ productionGateBound R F I (graphSize d) := by
  have assignments := production_assignments_le d v R F I
  have inputs := Nat.mul_le_mul_right (R * (1 + (F + I))) (graph_reads_le d v)
  have gates := resourceCount_bound (productionInputCount d v R F I) (productionAssemblyShared d v R F I).2
  constructor
  · simp only [productionResourcesShared, resourceCount, productionWireBound]
    change (productionReads d v).length * (R * (1 + (F + I))) + _ + 1 ≤ _
    omega
  · change (resourceCount _ _).2 ≤ _
    dsimp only [productionGateBound]
    omega

end Semantic.Circuit
