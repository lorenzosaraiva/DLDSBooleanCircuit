import Semantic.Circuit.RouteAllocation
import Semantic.Circuit.SharedBounds

open FlowSpec
namespace Semantic.Circuit.Shared
set_option maxHeartbeats 50000000
set_option maxRecDepth 100000




def planBudget {F : Nat} {d : Graph} {basis : Fin F → Formula} (inputs : Vertex → Nat) :
    ProductionPlan d basis → Nat × Nat
  | .input v => (inputs v, 0)
  | .constants routes _ => (routes.length, routes.length)
  | .unary edges _ seeds _ child =>
      let source := planBudget inputs child
      let count := source.1 * (edges.length * (seedModes seeds).length)
      (count, source.2 + count)
  | .binary minors majors seeds left right =>
      let a := planBudget inputs left
      let b := planBudget inputs right
      let count := a.1 * b.1 * (minors.length * majors.length * (seedModes seeds).length)
      (count, a.2 + b.2 + count)
  | .append left right =>
      let a := planBudget inputs left
      let b := planBudget inputs right
      (a.1 + b.1, a.2 + b.2)
  | .dedup child =>
      let source := planBudget inputs child
      (source.1, source.2 + source.1 * source.1)

def PlanLabels {F : Nat} {d : Graph} {basis : Fin F → Formula} (g : Nat) :
    ProductionPlan d basis → Prop
  | .input _ => True
  | .constants _ _ => True
  | .unary edges _ _ _ child => (∀ e ∈ edges, e.DEPENDENCY.length ≤ g) ∧ PlanLabels g child
  | .binary minors majors _ left right =>
      (∀ e ∈ minors, e.DEPENDENCY.length ≤ g) ∧ (∀ e ∈ majors, e.DEPENDENCY.length ≤ g) ∧
      PlanLabels g left ∧ PlanLabels g right
  | .append left right => PlanLabels g left ∧ PlanLabels g right
  | .dedup child => PlanLabels g child

def allocationUnit (g F I : Nat) : Nat := routeAllocation g F I + 9 * (1 + (F + I)) + 4

theorem candidates_work_le {scope width : Nat} (base : Nat) (entries : List (Candidate scope width))
    (code : Store) (cost : Nat) (each : ∀ e ∈ entries, candidateWork e ≤ cost) :
    (candidates base entries code).2.size ≤ code.size + entries.length * cost := by
  rw [candidates_work]
  exact Nat.add_le_add_left (sum_map_bound _ _ _ each) _

theorem unaryEmission_length {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (edges : List Deduction) (formula : Formula) (seeds : Option (List (SeedWord d))) (zero : Bool)
    (e : Candidate n (1 + (F + I))) :
    (unaryEmission basis d edges formula seeds zero e).length = edges.length * (seedModes seeds).length := by
  simp [unaryEmission, List.length_flatMap]

theorem binaryEmission_length {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minors majors : List Deduction) (seeds : Option (List (SeedWord d)))
    (left right : Candidate n (1 + (F + I))) :
    (binaryEmission basis d minors majors seeds left right).length =
      minors.length * majors.length * (seedModes seeds).length := by
  simp [binaryEmission, List.length_flatMap, Nat.mul_assoc]

theorem flatMap_length_constant {α β : Type} (xs : List α) (f : α → List β) (k : Nat)
    (lengths : ∀ x ∈ xs, (f x).length = k) : (xs.flatMap f).length = xs.length * k := by
  simp only [List.length_flatMap]
  rw [List.map_eq_replicate_iff.mpr lengths, List.sum_replicate_nat]

theorem compilePlan_allocation {n F I : Nat} {d : Graph} {basis : Fin F → Formula}
    (plan : ProductionPlan d basis) (inputs : Vertex → List (Candidate n (1 + (F + I))))
    (g : Nat) (labels : PlanLabels g plan) (table : ([] :: graphWords d).length ≤ g)
    (zero : ∀ v e, e ∈ inputs v → ZeroWork e) (code : Store) :
    (compilePlan inputs plan code).1.length = (planBudget (fun v => (inputs v).length) plan).1 ∧
      (compilePlan inputs plan code).2.size ≤ code.size +
        (planBudget (fun v => (inputs v).length) plan).2 * allocationUnit g F I := by
  let cost := allocationUnit g F I
  have routeCost : routeAllocation g F I ≤ cost := by dsimp [cost, allocationUnit]; omega
  have constantCost : 2 + F + I ≤ cost := by
    dsimp [cost, allocationUnit, routeAllocation]; omega
  have flagCost : 9 * (1 + (F + I)) + 4 ≤ cost := by simp [cost, allocationUnit]
  induction plan generalizing code with
  | input v =>
    refine ⟨candidates_length .., ?_⟩
    simp only [compilePlan, candidates_work, planBudget, Nat.zero_mul, Nat.add_zero]
    have each : (inputs v).map candidateWork = List.replicate (inputs v).length 0 := by
      apply List.map_eq_replicate_iff.mpr
      intro e he
      have z := zero v e he
      simp [candidateWork, z.1, z.2]
    rw [each]; simp
  | constants routes fitting =>
    refine ⟨by simp [compilePlan, candidates_length, planBudget], ?_⟩
    exact (candidates_work_le n _ code cost (by
      intro e he; obtain ⟨r, _, rfl⟩ := List.mem_map.mp he
      rw [constantCandidate_work]; exact constantCost)).trans (by simp [planBudget, cost])
  | unary edges formula seeds only child ih =>
    let source := compilePlan inputs child code
    have h := ih labels.2 code
    have bounds := (compilePlan_bounds child inputs code).2
    let entries := (source.1.map fun e => e.expressions (n + source.2.size)).flatMap
      (unaryEmission basis d edges formula seeds only)
    have length : entries.length = source.1.length * (edges.length * (seedModes seeds).length) := by
      rw [flatMap_length_constant _ _ _ (by intro e _; exact unaryEmission_length ..)]
      simp
    have each : ∀ e ∈ entries, candidateWork e ≤ cost := by
      intro e he
      obtain ⟨entry, hm, edge, heEdge, mode, _, rfl⟩ :=
        (by simpa [entries, unaryEmission, List.mem_flatMap, List.mem_map] using he)
      exact (unaryCandidate_work_le basis d edge formula mode _ _ g
        (registers_zeroWork entry _ (bounds entry hm)) (labels.1 edge heEdge) table).trans routeCost
    have workBound := candidates_work_le n entries source.2 cost each
    change (candidates n entries source.2).1.length = _ ∧ (candidates n entries source.2).2.size ≤ _
    rw [candidates_length, length]
    dsimp only [planBudget]
    refine ⟨by rw [h.1], ?_⟩
    rw [length, h.1] at workBound
    have hs : source.2.size ≤ code.size + (planBudget (fun v => (inputs v).length) child).2 * cost := h.2
    dsimp only [cost] at *
    nlinarith [hs]
  | binary minors majors seeds left right hl hr =>
    let a := compilePlan inputs left code
    let b := compilePlan inputs right a.2
    have ha := hl labels.2.2.1 code
    have hb := hr labels.2.2.2 a.2
    have ab := compilePlan_bounds left inputs code
    have bb := compilePlan_bounds right inputs a.2
    have leftBounds := Bounded.weaken ab.2 (Nat.add_le_add_left bb.1.size_le n)
    let entries := (a.1.map fun e => e.expressions (n + b.2.size)).flatMap fun x =>
      (b.1.map fun e => e.expressions (n + b.2.size)).flatMap (binaryEmission basis d minors majors seeds x)
    have length : entries.length = a.1.length * b.1.length *
        (minors.length * majors.length * (seedModes seeds).length) := by
      simp [entries, List.length_flatMap, binaryEmission_length, Function.comp_def, Nat.mul_assoc]
    have each : ∀ e ∈ entries, candidateWork e ≤ cost := by
      intro e he
      obtain ⟨x, hx, y, hy, minor, hm, major, hg, mode, _, rfl⟩ :=
        (by simpa [entries, binaryEmission, List.mem_flatMap, List.mem_map] using he)
      exact (binaryCandidate_work_le basis d minor major mode _ _ g
        (registers_zeroWork x _ (leftBounds x hx)) (registers_zeroWork y _ (bb.2 y hy))
        (labels.1 minor hm) (labels.2.1 major hg) table).trans routeCost
    have w := candidates_work_le n entries b.2 cost each
    change (candidates n entries b.2).1.length = _ ∧ (candidates n entries b.2).2.size ≤ _
    rw [candidates_length, length]
    dsimp only [planBudget]
    refine ⟨by rw [ha.1, hb.1], ?_⟩
    rw [length, ha.1, hb.1] at w
    nlinarith [ha.2, hb.2]
  | append left right hl hr =>
    have a := hl labels.1 code
    have b := hr labels.2 (compilePlan inputs left code).2
    simp only [compilePlan, planBudget, List.length_append]
    exact ⟨by rw [a.1, b.1], by nlinarith [a.2, b.2]⟩
  | dedup child ih =>
    have h := ih labels code
    have w := firstFlags_work n (compilePlan inputs child code).1.length
      (compilePlan inputs child code).1 (compilePlan inputs child code).2 (compilePlan_bounds child inputs code).2
    simp only [compilePlan, planBudget, firstFlags_length]
    refine ⟨h.1, ?_⟩
    rw [h.1]
    rw [h.1] at w
    have m := Nat.mul_le_mul_left
      ((planBudget (fun v => (inputs v).length) child).1 * (planBudget (fun v => (inputs v).length) child).1) flagCost
    dsimp only [cost] at m
    nlinarith [h.2]

end Semantic.Circuit.Shared
