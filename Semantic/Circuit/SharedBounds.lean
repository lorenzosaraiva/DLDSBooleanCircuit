import Semantic.Circuit.SharedCost
import Semantic.Circuit.SharedComparison

namespace Semantic.Circuit.Shared

open scoped BigOperators
set_option maxHeartbeats 30000000

theorem suppress_work {scope width : Nat} (pivot entry : Registers width)
    (pivotBound : pivot.Bounded scope) (entryBound : entry.Bounded scope) :
    candidateWork (suppress (pivot.expressions scope) (entry.expressions scope)) = 9 * width + 4 := by
  simp only [candidateWork, suppress, work]
  rw [fieldEqual_work pivot entry pivotBound entryBound]
  have p : work (pivot.expressions scope).live = 0 := work_reference _ _ pivotBound.1
  have e : work (entry.expressions scope).live = 0 := work_reference _ _ entryBound.1
  have fields : ∀ i, work ((entry.expressions scope).payload i) = 0 := fun i => work_reference _ _ (entryBound.2 i)
  simp only [p, e, fields, Finset.sum_const_zero]
  omega

theorem suppressRegisters_work {width : Nat} (base : Nat) (pivot : Registers width)
    (entries : List (Registers width)) (code : Store)
    (pivotBound : pivot.Bounded (base + code.size)) (bounded : Bounded entries (base + code.size)) :
    (suppressRegisters base pivot entries code).2.size = code.size + entries.length * (9 * width + 4) := by
  simp only [suppressRegisters, candidates_work, List.map_map, Function.comp_def]
  have same := List.map_eq_replicate_iff.mpr (fun e member => suppress_work pivot e pivotBound (bounded e member))
  rw [same, List.sum_replicate_nat]

theorem firstFlags_length {width : Nat} (base fuel : Nat) (entries : List (Registers width)) (code : Store) :
    (firstFlags base fuel entries code).1.length = entries.length := by
  induction fuel generalizing entries code with
  | zero => rfl
  | succ fuel ih => cases entries <;> simp [firstFlags, ih]

theorem firstFlags_work {width : Nat} (base fuel : Nat) (entries : List (Registers width))
    (code : Store) (bounded : Bounded entries (base + code.size)) :
    (firstFlags base fuel entries code).2.size ≤ code.size + fuel * entries.length * (9 * width + 4) := by
  induction fuel generalizing entries code with
  | zero => simp [firstFlags]
  | succ fuel ih =>
    cases entries with
    | nil => simp [firstFlags]
    | cons head tail =>
      have hb := bounded head (by simp)
      have tb : Bounded tail (base + code.size) := fun e h => bounded e (by simp [h])
      have next := (suppressRegisters_correct base head tail code (fun _ => false) hb tb).2.1
      have recursiveBound := ih (suppressRegisters base head tail code).1 (suppressRegisters base head tail code).2 next
      rw [suppressRegisters_length, suppressRegisters_work base head tail code hb tb] at recursiveBound
      simp only [firstFlags, List.length_cons]
      nlinarith

theorem emptyExpression_work {width : Nat} (scope : Nat) (entries : List (Registers width))
    (bounded : Bounded entries scope) :
    work (emptyExpression scope entries) = 2 * entries.length + 1 := by
  simp only [emptyExpression, work_all, List.length_map, List.map_map, Function.comp_def]
  have same : entries.map (fun e => work ((reference scope e.live).neg)) = List.replicate entries.length 1 := by
    apply List.map_eq_replicate_iff.mpr
    intro e member
    simp only [work, work_reference _ _ (bounded e member).1]
  rw [same, List.sum_replicate_nat]
  omega

theorem cellExpression_work {width : Nat} (scope : Nat) (head other : Registers width)
    (diagonal right below : Nat) (headBound : head.Bounded scope) (otherBound : other.Bounded scope)
    (diagonalBound : diagonal < scope) (rightBound : right < scope) (belowBound : below < scope) :
    work (cellExpression scope head other diagonal right below) = 9 * width + 16 := by
  simp only [cellExpression, work_choose, work]
  rw [fieldEqual_work head other headBound otherBound]
  rw [work_reference _ _ headBound.1, work_reference _ _ otherBound.1,
    work_reference _ _ diagonalBound, work_reference _ _ rightBound, work_reference _ _ belowBound]
  omega

theorem orderedRow_work {width : Nat} (base : Nat) (head : Registers width)
    (right : List (Registers width)) (below : Fin (right.length + 1) → Nat) (last : Nat) (code : Store)
    (headBound : head.Bounded (base + code.size)) (rightBounds : Bounded right (base + code.size))
    (belowBounds : WireBounds below (base + code.size)) (lastBound : last < base + code.size) :
    (orderedRow base head right below last code).2.size = code.size + right.length * (9 * width + 16) := by
  induction right generalizing code with
  | nil => simp [orderedRow]
  | cons other tail ih =>
    have tailBounds : Bounded tail (base + code.size) := fun e h => rightBounds e (by simp [h])
    have recursiveBound := ih (fun i => below i.succ) code headBound tailBounds (fun i => belowBounds i.succ) lastBound
    have tailCorrect := orderedRow_correct base head tail (fun i => below i.succ) last code
      (fun _ => false) headBound tailBounds (fun i => belowBounds i.succ) lastBound
    have grow := Nat.add_le_add_left tailCorrect.1.size_le base
    simp only [orderedRow, expression_work, List.length_cons]
    rw [cellExpression_work _ _ _ _ _ _ (Registers.weaken head headBound grow)
      (Registers.weaken other (rightBounds other (by simp)) grow)
      (lt_of_lt_of_le (belowBounds 1) grow) (tailCorrect.2.1 0) (lt_of_lt_of_le (belowBounds 0) grow), recursiveBound]
    ring

def orderedWorkBound (candidates stored width : Nat) : Nat :=
  (stored + 1) * (2 * stored + 1) + candidates * (2 * candidates + 1 + stored * (9 * width + 16))

theorem orderedTable_work {width : Nat} (base : Nat) (left right : List (Registers width))
    (code : Store) (leftBounds : Bounded left (base + code.size)) (rightBounds : Bounded right (base + code.size)) :
    (orderedTable base left right code).2.size ≤ code.size + orderedWorkBound left.length right.length width := by
  induction left generalizing code with
  | nil =>
    simp only [orderedTable, vector_work, orderedWorkBound, List.length_nil, Nat.zero_mul, Nat.add_zero]
    have bound : ∀ i : Fin (right.length + 1),
        work (emptyExpression (base + code.size) (right.drop i.val)) ≤ 2 * right.length + 1 := by
      intro i
      rw [emptyExpression_work _ _ (fun e h => rightBounds e (List.mem_of_mem_drop h))]
      simp only [List.length_drop]
      omega
    have sumBound := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => bound i)
    simpa using Nat.add_le_add_left sumBound code.size
  | cons head tail ih =>
    have tailBounds : Bounded tail (base + code.size) := fun e h => leftBounds e (by simp [h])
    have recursiveBound := ih code tailBounds rightBounds
    have belowCorrect := orderedTable_correct base tail right code (fun _ => false) tailBounds rightBounds
    have leftBelow := Bounded.weaken leftBounds (Nat.add_le_add_left belowCorrect.1.size_le base)
    have lastCorrect := expression_correct base
      (emptyExpression (base + (orderedTable base tail right code).2.size) (head :: tail))
      (orderedTable base tail right code).2 (by omega) (fun _ => false)
    have preceding := belowCorrect.1.trans lastCorrect.1
    have row := orderedRow_work base head right (orderedTable base tail right code).1 _ _
      (Registers.weaken head (leftBounds head (by simp)) (Nat.add_le_add_left preceding.size_le base))
      (Bounded.weaken rightBounds (Nat.add_le_add_left preceding.size_le base))
      (fun i => lt_of_lt_of_le (belowCorrect.2.1 i) (Nat.add_le_add_left lastCorrect.1.size_le base)) lastCorrect.2.1
    dsimp only [orderedTable]
    rw [row, expression_work, emptyExpression_work _ _ leftBelow]
    simp only [orderedWorkBound, List.length_cons] at *
    nlinarith

end Semantic.Circuit.Shared
