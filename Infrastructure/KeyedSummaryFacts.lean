import Infrastructure.ResidualKeyedEvaluator

open Semantic FlowSpec

namespace KeyedEval




theorem keys_merge (d : Graph) (routes : List FlowPair) :
    (merge d routes).map Prod.fst = (routes.map Prod.snd).eraseDups := by
  simp [merge, List.map_map, Function.comp_def]

theorem merge_keys_nodup (d : Graph) (routes : List FlowPair) :
    ((merge d routes).map Prod.fst).Nodup := by
  rw [keys_merge]
  exact FlowSpec.eraseDups_nodup _



theorem merge_bit_iff (d : Graph) (routes : List FlowPair) (row : Row d)
    (member : row ∈ merge d routes)
    (index : Fin (RefinedEval.formulas d).length) :
    row.2.get index = true ↔ ∃ route ∈ routes,
      route.2 = row.1 ∧ (RefinedEval.formulas d).get index ∈ route.1 := by
  obtain ⟨residual, _, rfl⟩ := List.mem_map.mp member
  simp only [depVector_get, decide_eq_true_eq, List.mem_flatMap, List.mem_filter,
    beq_iff_eq]
  aesop

theorem merge_key_unique (d : Graph) (routes : List FlowPair) (a b : Row d)
    (ha : a ∈ merge d routes) (hb : b ∈ merge d routes)
    (same : a.1 = b.1) : a = b := by
  obtain ⟨left, _, rfl⟩ := List.mem_map.mp ha
  obtain ⟨right, _, rfl⟩ := List.mem_map.mp hb
  dsimp at same
  subst right
  rfl

#print axioms keys_merge
#print axioms merge_keys_nodup
#print axioms merge_bit_iff
#print axioms merge_key_unique

end KeyedEval
