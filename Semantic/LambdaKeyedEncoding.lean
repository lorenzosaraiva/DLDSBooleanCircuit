import Semantic.LambdaKeyedEvaluator

open Semantic FlowSpec

namespace LambdaKeyed

private theorem decoded_map_mem (formulas deps : List Formula) (formula : Formula) :
    formula ∈ ((formulas.zip (formulas.map fun f => decide (f ∈ deps))).filterMap
      fun (f, bit) => if bit then some f else none) ↔
      formula ∈ formulas ∧ formula ∈ deps := by
  induction formulas with
  | nil => simp
  | cons f rest ih =>
      by_cases present : f ∈ deps
      · simp only [List.map_cons, List.zip_cons_cons, List.filterMap_cons]
        simp [present, ih]
        aesop
      · simp only [List.map_cons, List.zip_cons_cons, List.filterMap_cons]
        simp [present, ih]
        aesop

theorem decode_depVector_mem (d : Graph) (deps : Dep) (formula : Formula) :
    formula ∈ KeyedEval.decode d (depVector (RefinedEval.formulas d) deps) ↔
      formula ∈ RefinedEval.formulas d ∧ formula ∈ deps := by
  simpa [KeyedEval.decode, depVector] using decoded_map_mem (RefinedEval.formulas d) deps formula

theorem decode_merge_mem (d : Graph) (routes : List FlowPair) (row : KeyedEval.Row d)
    (member : row ∈ KeyedEval.merge d routes) (formula : Formula) :
    formula ∈ KeyedEval.decode d row.2 ↔
      formula ∈ RefinedEval.formulas d ∧ ∃ route ∈ routes,
        route.2 = row.1 ∧ formula ∈ route.1 := by
  obtain ⟨residual, _, rfl⟩ := List.mem_map.mp member
  simp only [decode_depVector_mem, List.mem_flatMap, List.mem_filter, beq_iff_eq]
  aesop




theorem decode_merge_setEq (d : Graph) (routes : List FlowPair) (row : KeyedEval.Row d)
    (member : row ∈ KeyedEval.merge d routes)
    (supported : ∀ route ∈ routes, ∀ formula ∈ route.1,
      formula ∈ RefinedEval.formulas d) :
    depSetEq (KeyedEval.decode d row.2)
      ((routes.filter fun route => route.2 == row.1).flatMap Prod.fst) = true := by
  apply (depSetEq_eq_true_iff _ _).mpr
  intro formula
  rw [decode_merge_mem d routes row member]
  simp only [List.mem_flatMap, List.mem_filter, beq_iff_eq]
  constructor
  · rintro ⟨_, route, hroute, same, bit⟩
    exact ⟨route, ⟨hroute, same⟩, bit⟩
  · rintro ⟨route, ⟨hroute, same⟩, bit⟩
    exact ⟨supported route hroute formula bit, route, hroute, same, bit⟩

#print axioms decode_depVector_mem
#print axioms decode_merge_mem
#print axioms decode_merge_setEq

end LambdaKeyed
