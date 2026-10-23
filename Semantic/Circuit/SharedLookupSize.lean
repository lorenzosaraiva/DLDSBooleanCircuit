import Semantic.Circuit.SharedCost

namespace Semantic.Circuit.Shared

set_option maxHeartbeats 20000000

theorem work_any {n : Nat} (entries : List (Expr n)) :
    work (Expr.any entries) = 1 + 4 * entries.length + (entries.map work).sum := by
  induction entries with
  | nil => rfl
  | cons head rest ih =>
    change work (head.disj (Expr.any rest)) = _
    rw [work_disj, ih]
    simp only [List.length_cons, List.map_cons, List.sum_cons]
    omega

theorem fieldEqual_constant_work {n I : Nat} (input : Fin I → Expr n)
    (constant : Fin I → Bool) (inputWork : ∀ i, work (input i) = 0) :
    work (fieldEqual input (fun i => .constant (constant i))) = 11 * I + 1 := by
  simp only [fieldEqual, work_all, List.length_map, List.length_finRange, List.map_map, Function.comp_def]
  have rows : (List.finRange I).map
      (fun i => work ((input i).equivalent (.constant (constant i)))) = List.replicate I 10 := by
    have each : ∀ i ∈ List.finRange I, work ((input i).equivalent (.constant (constant i))) = 10 := by
      intro i _
      simp only [work_equivalent, inputWork, work]
    simpa only [List.length_finRange] using List.map_eq_replicate_iff.mpr each
  rw [rows, List.sum_replicate_nat]
  omega

theorem lookupBit_work {n I : Nat} (input : Fin I → Expr n) (table : List (BitVec I × Bool))
    (inputWork : ∀ i, work (input i) = 0) :
    work (lookupBit input table) = table.length * (11 * I + 7) + 1 := by
  simp only [lookupBit, work_any, List.length_map, List.map_map, Function.comp_def]
  have rows : table.map (fun row => work ((fieldEqual input
      (fun i => .constant (row.1.getLsbD i.val))).conj (.constant row.2))) =
      List.replicate table.length (11 * I + 3) := by
    apply List.map_eq_replicate_iff.mpr
    intro row _
    simp only [work, fieldEqual_constant_work input _ inputWork]
  rw [rows, List.sum_replicate_nat]
  ring

theorem indexedLookup_work {n I : Nat} (input : Fin I → Expr n) (table : List Bool)
    (inputWork : ∀ i, work (input i) = 0) :
    work (indexedLookup input table) = table.length * (11 * I + 7) + 1 := by
  simpa [indexedLookup] using lookupBit_work input
    ((List.finRange table.length).map fun i => (BitVec.ofNat I i.val, table.get i)) inputWork

theorem consumeValid_work {n I : Nat} (input : Fin I → Expr n)
    (words : List FlowSpec.ColourPath) (edge : Deduction) (inputWork : ∀ i, work (input i) = 0) :
    work (consumeValid input words edge) = (1 + words.length) * (11 * I + 7) + 1 := by
  simpa [consumeValid, List.length_map, Nat.add_comm] using
    indexedLookup_work input (([] :: words).map fun word => (FlowSpec.consume edge word).isSome) inputWork



theorem full_capacity_lookup_steps (I : Nat) :
    (expression I (indexedLookup (fun i : Fin I => Expr.bit i) (List.replicate (2 ^ I) false)) #[]).2.size =
      2 ^ I * (11 * I + 7) + 1 := by
  rw [expression_work, indexedLookup_work _ _ (fun _ => rfl)]
  simp

end Semantic.Circuit.Shared
