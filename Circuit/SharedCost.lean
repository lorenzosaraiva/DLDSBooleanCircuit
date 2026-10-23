import Circuit.SharedFlags
import Circuit.SharedOrdered

namespace Semantic.Circuit.Shared

open scoped BigOperators
set_option maxHeartbeats 20000000

def work {n : Nat} : Expr n → Nat
  | .bit _ => 0
  | .constant _ => 1
  | .neg e => work e + 1
  | .conj a b => work a + work b + 1

theorem expression_work {n : Nat} (base : Nat) (e : Expr n) (code : Store) :
    (expression base e code).2.size = code.size + work e := by
  induction e generalizing code with
  | bit wire => rfl
  | constant b => simp [expression, emit, work]
  | neg e ih => simp only [expression, emit, Array.size_push, ih, work]; omega
  | conj a b ha hb => simp only [expression, emit, Array.size_push, hb, ha, work]; omega

theorem work_disj {n : Nat} (a b : Expr n) : work (a.disj b) = work a + work b + 4 := by
  simp [Expr.disj, work]
  omega

theorem work_equivalent {n : Nat} (a b : Expr n) : work (a.equivalent b) = 2 * work a + 2 * work b + 8 := by
  simp only [Expr.equivalent, work_disj, work]
  omega

theorem work_choose {n : Nat} (a b c : Expr n) : work (a.choose b c) = 2 * work a + work b + work c + 7 := by
  simp only [Expr.choose, work_disj, work]
  omega

theorem work_all {n : Nat} (entries : List (Expr n)) :
    work (Expr.all entries) = 1 + entries.length + (entries.map work).sum := by
  induction entries with
  | nil => rfl
  | cons head rest ih =>
    change work head + work (Expr.all rest) + 1 = _
    rw [ih]
    simp only [List.length_cons, List.map_cons, List.sum_cons]
    omega

theorem work_reference (scope wire : Nat) (bounded : wire < scope) : work (reference scope wire) = 0 := by
  simp [reference, bounded, work]

theorem vector_work {scope : Nat} (base width : Nat) (fields : Fin width → Expr scope) (code : Store) :
    (vector base width fields code).2.size = code.size + ∑ i, work (fields i) := by
  induction width generalizing code with
  | zero => simp [vector]
  | succ width ih =>
    simp only [vector, ih, expression_work, Fin.sum_univ_succ]
    omega

def selectionWork {scope width : Nat} (entry : Selection scope width) : Nat :=
  work entry.live + ∑ i, work (entry.payload i)

theorem selection_work {scope width : Nat} (base : Nat) (entry : Selection scope width) (code : Store) :
    (selection base entry code).2.size = code.size + selectionWork entry := by
  simp only [selection, vector_work, expression_work, selectionWork]
  omega

theorem selections_work {scope width : Nat} (base : Nat) (entries : List (Selection scope width)) (code : Store) :
    (selections base entries code).2.size = code.size + (entries.map selectionWork).sum := by
  induction entries generalizing code with
  | nil => simp [selections]
  | cons head tail ih => simp only [selections, ih, selection_work, List.map_cons, List.sum_cons]; omega

theorem fieldEqual_work {scope width : Nat} (left right : Registers width)
    (leftBound : left.Bounded scope) (rightBound : right.Bounded scope) :
    work (fieldEqual (left.expressions scope).payload (right.expressions scope).payload) = 9 * width + 1 := by
  simp only [fieldEqual, work_all, List.length_map, List.length_finRange, List.map_map]
  have each : (List.finRange width).map
      (fun i => work (((left.expressions scope).payload i).equivalent ((right.expressions scope).payload i))) =
      List.replicate width 8 := by
    apply List.ext_getElem
    · simp
    · intro i hi hj
      simp only [List.getElem_map, List.getElem_replicate]
      rw [work_equivalent]
      simp only [Registers.expressions]
      rw [work_reference _ _ (leftBound.2 _), work_reference _ _ (rightBound.2 _)]
  change 1 + width + ((List.finRange width).map
    (fun i => work (((left.expressions scope).payload i).equivalent ((right.expressions scope).payload i)))).sum = _
  rw [each, List.sum_replicate_nat]
  omega

end Semantic.Circuit.Shared
