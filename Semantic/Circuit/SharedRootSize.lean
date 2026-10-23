import Semantic.Circuit.SharedLookupSize
import Semantic.Circuit.SharedResources

namespace Semantic.Circuit

set_option maxHeartbeats 20000000

theorem root_zero_field_work {n k : Nat} (wires : Fin k → Fin n) :
    Shared.work (equalConstant wires (fun _ => false)) = 2 * k + 1 := by
  simp only [equalConstant, Bool.false_eq_true, ite_false, Shared.work_all,
    List.length_map, List.length_finRange, List.map_map, Function.comp_def]
  have same : (List.finRange k).map (fun i => Shared.work (Expr.neg (.bit (wires i)))) = List.replicate k 1 := by
    have each : ∀ i ∈ List.finRange k, Shared.work (Expr.neg (.bit (wires i))) = 1 := fun _ _ => rfl
    simpa only [List.length_finRange] using List.map_eq_replicate_iff.mpr each
  rw [same, List.sum_replicate_nat]
  omega

theorem root_slot_work (R F I : Nat) (slot : Fin R) :
    Shared.work (slotExpression R F I slot) = 2 * F + 2 * I + 8 := by
  simp only [slotExpression, Shared.work_disj, Shared.work, root_zero_field_work]
  omega

theorem root_expression_work (R F I : Nat) :
    Shared.work (rootExpression R F I) = 3 + R * (2 * F + 2 * I + 13) := by
  simp only [rootExpression, Shared.work, Shared.work_any, Shared.work_all,
    List.length_map, List.length_finRange, List.map_map, Function.comp_def]
  have slots : (List.finRange R).map (fun slot => Shared.work (slotExpression R F I slot)) =
      List.replicate R (2 * F + 2 * I + 8) := by
    simpa only [List.length_finRange] using
      List.map_eq_replicate_iff.mpr (fun (i : Fin R) (_ : i ∈ List.finRange R) => root_slot_work R F I i)
  rw [slots]
  simp only [List.map_const', List.length_finRange, List.sum_replicate_nat]
  ring

theorem root_polynomial_size (R F I : Nat) :
    (rootResourceCountShared R F I).1 = R * (1 + (F + I)) + (3 + R * (2 * F + 2 * I + 13)) + 1 ∧
    (rootResourceCountShared R F I).2 ≤ 4 * (3 + R * (2 * F + 2 * I + 13)) + 1 := by
  have allocated : (Shared.expression (R * (1 + (F + I))) (rootExpression R F I) #[]).2.size =
      3 + R * (2 * F + 2 * I + 13) := by
    rw [Shared.expression_work, root_expression_work]
    simp
  constructor
  · simp only [rootResourceCountShared, Shared.resourceCount, allocated]
  · have bound := Shared.resourceCount_bound (R * (1 + (F + I)))
      (Shared.expression (R * (1 + (F + I))) (rootExpression R F I) #[]).2
    rw [allocated] at bound
    exact bound

end Semantic.Circuit
