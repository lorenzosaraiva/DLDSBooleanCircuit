import Circuit.GraphSize
import Circuit.SharedLookupSize

open FlowSpec
namespace Semantic.Circuit.Shared
open scoped BigOperators
set_option maxHeartbeats 30000000

def ZeroWork {n width : Nat} (e : Selection n width) : Prop :=
  work e.live = 0 ∧ ∀ i, work (e.payload i) = 0

theorem registers_zeroWork {width : Nat} (e : Registers width) (scope : Nat)
    (bounded : e.Bounded scope) : ZeroWork (e.expressions scope) :=
  ⟨work_reference _ _ bounded.1, fun i => work_reference _ _ (bounded.2 i)⟩

def lookupAllocation (g I : Nat) : Nat := g * (11 * I + 7) + 1
def guardAllocation (g F : Nat) : Nat := 3 + g * (2 + 5 * F) + 2 * F
def routeAllocation (g F I : Nat) : Nat :=
  20 + 2 * guardAllocation g F + 2 * lookupAllocation g I + 4 * F +
    I * (5 * lookupAllocation g I + 9) + (1 + (F + I))

theorem formulaPresent_work_le {n F : Nat} (basis : Fin F → Formula) (mask : Fin F → Expr n)
    (formula : Formula) (zero : ∀ i, work (mask i) = 0) :
    work (formulaPresent basis mask formula) ≤ 1 + 5 * F := by
  rw [formulaPresent, work_any]
  simp only [List.length_map, List.length_finRange, List.map_map, Function.comp_def]
  have h : ((List.finRange F).map (fun i => work (if basis i = formula then mask i else .constant false))).sum ≤ F := by
    have t := sum_map_bound (List.finRange F)
      (fun i => work (if basis i = formula then mask i else .constant false)) 1
      (by intro i _; dsimp only; split <;> simp [work, zero])
    simpa using t
  omega

theorem dependencyGuard_work_le {n F : Nat} (basis : Fin F → Formula) (mask : Fin F → Expr n)
    (edge : Deduction) (g : Nat) (label : edge.DEPENDENCY.length ≤ g)
    (zero : ∀ i, work (mask i) = 0) :
    work (dependencyGuard basis mask edge) ≤ guardAllocation g F := by
  unfold dependencyGuard
  split
  · simp only [work]; dsimp [guardAllocation]; omega
  · simp only [work, work_all, List.length_map, List.length_finRange, List.map_map, Function.comp_def]
    have a := sum_map_bound edge.DEPENDENCY (fun f => work (formulaPresent basis mask f))
      (1 + 5 * F) (fun _ _ => formulaPresent_work_le basis mask _ zero)
    have b := sum_map_bound (List.finRange F)
      (fun i => work (if basis i ∈ edge.DEPENDENCY then Expr.constant true else (mask i).neg)) 1
      (by intro i _; dsimp only; split <;> simp [work, zero])
    simp only [List.length_finRange, Nat.mul_one] at b
    have m := Nat.mul_le_mul_right (2 + 5 * F) label
    dsimp [guardAllocation]
    nlinarith

theorem consumeBit_work {n I : Nat} (input : Fin I → Expr n) (words : List ColourPath)
    (edge : Deduction) (bit : Fin I) (zero : ∀ i, work (input i) = 0) :
    work (consumeBit input words edge bit) = (1 + words.length) * (11 * I + 7) + 1 := by
  simpa [consumeBit, Nat.add_comm] using indexedLookup_work input
    (([] :: words).map fun word =>
      (BitVec.ofNat I (((consume edge word).map (([] :: words).idxOf ·)).getD 0)).getLsbD bit.val) zero

theorem fieldEqual_work_le {n width : Nat} (left right : Fin width → Expr n) (a b : Nat)
    (hl : ∀ i, work (left i) ≤ a) (hr : ∀ i, work (right i) ≤ b) :
    work (fieldEqual left right) ≤ 1 + width * (2 * a + 2 * b + 9) := by
  simp only [fieldEqual, work_all, List.length_map, List.length_finRange, List.map_map, Function.comp_def]
  have h := sum_map_bound (List.finRange width)
    (fun i => work ((left i).equivalent (right i))) (2 * a + 2 * b + 8)
    (by intro i _; dsimp only; rw [work_equivalent]; have x := hl i; have y := hr i; omega)
  simp only [List.length_finRange] at h
  nlinarith

theorem packSelection_work {n F I : Nat} (live : Expr n) (mask : Fin F → Expr n)
    (residual : Fin I → Expr n) :
    selectionWork (packSelection live mask residual) =
      work live + 1 + (∑ i, work (mask i)) + ∑ i, work (residual i) := by
  simp [selectionWork, packSelection, Fin.sum_univ_add, work]
  omega

theorem constantSelection_work {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (route : FlowPair) : selectionWork (constantSelection (n := n) (I := I) basis words route) = 2 + F + I := by
  simp [selectionWork, constantSelection, work]
  omega

theorem unarySelection_work_le {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (edge : Deduction) (formula : Formula) (reseed : Option ColourPath) (extra : Bool)
    (e : Selection n (1 + (F + I))) (g : Nat) (zero : ZeroWork e)
    (label : edge.DEPENDENCY.length ≤ g) (table : ([] :: graphWords d).length ≤ g) :
    selectionWork (unarySelection basis d edge formula reseed extra e) ≤ routeAllocation g F I := by
  have mz : ∀ i, work (selectionMask e i) = 0 := fun i => zero.2 _
  have rz : ∀ i, work (selectionResidual e i) = 0 := fun i => zero.2 _
  have guard := dependencyGuard_work_le basis (selectionMask e) edge g label mz
  have lookup : (1 + (graphWords d).length) * (11 * I + 7) + 1 ≤ lookupAllocation g I := by
    have t : 1 + (graphWords d).length ≤ g := by simpa only [List.length_cons, Nat.add_comm] using table
    exact Nat.add_le_add_right (Nat.mul_le_mul_right (11 * I + 7) t) 1
  have valid := consumeValid_work (selectionResidual e) (graphWords d) edge rz
  have mask : (∑ i, work (maskClear basis formula (selectionMask e) i)) ≤ F := by
    have h : ∀ i, work (maskClear basis formula (selectionMask e) i) ≤ 1 := by
      intro i; unfold maskClear; split <;> simp [work, mz]
    simpa using Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => h i)
  let fields : Fin I → Expr n := match reseed with
      | none => consumeBit (selectionResidual e) (graphWords d) edge
      | some word => fun i => .constant (BitVec.ofNat I (([] :: graphWords d).idxOf word))[i.val]
  have residualBound : (∑ i, work (fields i)) ≤ I * lookupAllocation g I := by
    apply le_trans (Finset.sum_le_sum (s := Finset.univ) (g := fun _ => lookupAllocation g I) ?_) (by simp)
    intro i _
    cases reseed <;> dsimp only [fields]
    · rw [consumeBit_work _ _ _ _ rz]; exact lookup
    · simp [work, lookupAllocation]
  unfold unarySelection
  rw [packSelection_work]
  change work _ + 1 + (∑ i, work (maskClear basis formula (selectionMask e) i)) +
    (∑ i, work (fields i)) ≤ _
  simp only [work, zero.1, valid]
  dsimp [routeAllocation]
  nlinarith

theorem binarySelection_work_le {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minor major : Deduction) (reseed : Option ColourPath)
    (left right : Selection n (1 + (F + I))) (g : Nat) (hl : ZeroWork left) (hr : ZeroWork right)
    (lm : minor.DEPENDENCY.length ≤ g) (lg : major.DEPENDENCY.length ≤ g)
    (table : ([] :: graphWords d).length ≤ g) :
    selectionWork (binarySelection basis d minor major reseed left right) ≤ routeAllocation g F I := by
  have ml : ∀ i, work (selectionMask left i) = 0 := fun i => hl.2 _
  have mr : ∀ i, work (selectionMask right i) = 0 := fun i => hr.2 _
  have rl : ∀ i, work (selectionResidual left i) = 0 := fun i => hl.2 _
  have rr : ∀ i, work (selectionResidual right i) = 0 := fun i => hr.2 _
  have gl := dependencyGuard_work_le basis (selectionMask left) minor g lm ml
  have gr := dependencyGuard_work_le basis (selectionMask right) major g lg mr
  have lookup : (1 + (graphWords d).length) * (11 * I + 7) + 1 ≤ lookupAllocation g I := by
    have t : 1 + (graphWords d).length ≤ g := by simpa only [List.length_cons, Nat.add_comm] using table
    exact Nat.add_le_add_right (Nat.mul_le_mul_right (11 * I + 7) t) 1
  have vl := consumeValid_work (selectionResidual left) (graphWords d) minor rl
  have vr := consumeValid_work (selectionResidual right) (graphWords d) major rr
  have bl : ∀ i, work (consumeBit (selectionResidual left) (graphWords d) minor i) ≤ lookupAllocation g I := by
    intro i; rw [consumeBit_work _ _ _ _ rl]; exact lookup
  have br : ∀ i, work (consumeBit (selectionResidual right) (graphWords d) major i) ≤ lookupAllocation g I := by
    intro i; rw [consumeBit_work _ _ _ _ rr]; exact lookup
  have agreement : work (binaryAgreement d minor major reseed left right) ≤
      1 + I * (4 * lookupAllocation g I + 9) := by
    cases reseed
    · exact (fieldEqual_work_le _ _ _ _ bl br).trans (by ring_nf; rfl)
    · simp [binaryAgreement, work]
  have mask : (∑ i, work (maskUnion (selectionMask left) (selectionMask right) i)) = 4 * F := by
    simp [maskUnion, work_disj, ml, mr, Nat.mul_comm]
  let fields : Fin I → Expr n := match reseed with
      | none => consumeBit (selectionResidual left) (graphWords d) minor
      | some word => fun i => .constant ((BitVec.ofNat I (([] :: graphWords d).idxOf word)).getLsbD i.val)
  have residualBound : (∑ i, work (fields i)) ≤ I * lookupAllocation g I := by
    apply le_trans (Finset.sum_le_sum (s := Finset.univ) (g := fun _ => lookupAllocation g I) ?_) (by simp)
    intro i _
    cases reseed <;> dsimp only [fields]
    · exact bl i
    · simp [work, lookupAllocation]
  unfold binarySelection
  rw [packSelection_work, mask, work_all]
  simp only [List.length_cons, List.length_nil, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, hl.1, hr.1, vl, vr]
  cases reseed <;> simp only [fields, work, consumeBit_work _ _ _ _ rl, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul] at residualBound ⊢
  all_goals dsimp [routeAllocation]; nlinarith

end Semantic.Circuit.Shared
