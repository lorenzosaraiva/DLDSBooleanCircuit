import Circuit.SharedOrdered

namespace Semantic.Circuit.Shared

set_option maxHeartbeats 30000000
set_option maxRecDepth 6000

theorem rowTruth_selected {α : Type} [DecidableEq α] (head : Bool × α) (rest right : List (Bool × α))
    (below : Fin (right.length + 1) → Bool) (last : Bool)
    (belowCorrect : ∀ i, below i = decide (selected rest = selected (right.drop i.val)))
    (lastCorrect : last = decide (selected (head :: rest) = [])) (i : Fin (right.length + 1)) :
    rowTruth head right below last i = decide (selected (head :: rest) = selected (right.drop i.val)) := by
  induction right with
  | nil => simpa [rowTruth] using lastCorrect
  | cons other tail ih =>
    have next : ∀ j : Fin (tail.length + 1), below j.succ = decide (selected rest = selected (tail.drop j.val)) := by
      intro j
      simpa using belowCorrect j.succ
    refine Fin.cases ?_ (fun j => ?_) i
    · simp only [rowTruth, Fin.cases_zero]
      rw [belowCorrect 0, belowCorrect 1, ih _ next 0]
      by_cases a : head.1 = true <;> by_cases b : other.1 = true <;>
        simp [selected_cons, a, b]
    · simpa [rowTruth] using ih _ next j

def emptyExpression {width : Nat} (scope : Nat) (entries : List (Registers width)) : Expr scope :=
  Expr.all (entries.map fun e => (reference scope e.live).neg)

theorem emptyExpression_value {width : Nat} (scope : Nat) (entries : List (Registers width))
    (state : Bits) (bounded : Bounded entries scope) :
    (emptyExpression scope entries).value state = decide (selected (values entries state) = []) := by
  have same : orderedExpression ([] : List (Selection scope width))
      (entries.map fun e => e.expressions scope) = emptyExpression scope entries := by
    rw [orderedExpression]
    simp only [emptyExpression, List.map_map, Function.comp_def, Registers.expressions]
  rw [← same, orderedExpression_correct, expressions_values entries scope state bounded]
  simp only [List.map_nil, selected_nil, eq_comm]
  rfl

def orderedTable {width : Nat} (base : Nat) : (left right : List (Registers width)) → Store →
    (Fin (right.length + 1) → Nat) × Store
  | [], right, code => vector base (right.length + 1)
      (fun i => emptyExpression (base + code.size) (right.drop i.val)) code
  | head :: tail, right, code =>
      let below := orderedTable base tail right code
      let last := expression base (emptyExpression (base + below.2.size) (head :: tail)) below.2
      orderedRow base head right below.1 last.1 last.2

theorem orderedTable_correct {width : Nat} (base : Nat) (left right : List (Registers width))
    (code : Store) (state : Bits) (leftBounds : Bounded left (base + code.size))
    (rightBounds : Bounded right (base + code.size)) :
    let result := orderedTable base left right code
    Extends code result.2 ∧ WireBounds result.1 (base + result.2.size) ∧
      ∀ i, memory base result.2 state (result.1 i) =
        decide (selected (values left (memory base code state)) =
          selected (values (right.drop i.val) (memory base code state))) := by
  induction left generalizing code with
  | nil =>
    dsimp only [orderedTable]
    obtain ⟨extension, bounds, same⟩ := vector_correct base (right.length + 1)
      (fun i => emptyExpression (base + code.size) (right.drop i.val)) code (by omega) state
    refine ⟨extension, bounds, ?_⟩
    intro i
    rw [same i, emptyExpression_value _ _ _ (fun e member => rightBounds e (List.mem_of_mem_drop member))]
    apply Bool.eq_iff_iff.mpr
    simp only [decide_eq_true_eq, values, List.map_nil, selected_nil]
    exact eq_comm
  | cons head tail ih =>
    obtain ⟨belowExtension, belowBounds, belowValue⟩ := ih code
      (fun e h => leftBounds e (by simp [h])) rightBounds
    have leftBelow := Bounded.weaken leftBounds (Nat.add_le_add_left belowExtension.size_le base)
    have rightBelow := Bounded.weaken rightBounds (Nat.add_le_add_left belowExtension.size_le base)
    obtain ⟨lastExtension, lastBound, lastValue⟩ := expression_correct base
      (emptyExpression (base + (orderedTable base tail right code).2.size) (head :: tail))
      (orderedTable base tail right code).2 (by omega) state
    have preceding := belowExtension.trans lastExtension
    have headLast := Registers.weaken head (leftBounds head (by simp))
      (Nat.add_le_add_left preceding.size_le base)
    have rightLast := Bounded.weaken rightBounds (Nat.add_le_add_left preceding.size_le base)
    have belowLast : WireBounds (orderedTable base tail right code).1
        (base + (expression base (emptyExpression (base + (orderedTable base tail right code).2.size)
          (head :: tail)) (orderedTable base tail right code).2).2.size) :=
      fun i => lt_of_lt_of_le (belowBounds i) (Nat.add_le_add_left lastExtension.size_le base)
    obtain ⟨rowExtension, rowBounds, rowValue⟩ := orderedRow_correct base head right
      (orderedTable base tail right code).1 _ _ state headLast rightLast belowLast lastBound
    refine ⟨preceding.trans rowExtension, rowBounds, ?_⟩
    intro i
    dsimp only [orderedTable]
    rw [congrFun rowValue i]
    rw [← values_stable (head :: tail) preceding base state leftBounds,
      ← values_stable (right.drop i.val) preceding base state
        (fun e h => rightBounds e (List.mem_of_mem_drop h))]
    let lastResult := expression base
      (emptyExpression (base + (orderedTable base tail right code).2.size) (head :: tail))
      (orderedTable base tail right code).2
    let current := memory base lastResult.2 state
    have rowSemantic := rowTruth_selected (head.value current) (values tail current) (values right current)
      (fun j => current ((orderedTable base tail right code).1 (Fin.cast (by simp [values]) j)))
      (current lastResult.1) ?_ ?_ (Fin.cast (by simp [values]) i)
    · simpa only [values, List.map_cons, List.map_drop, Fin.val_cast] using rowSemantic
    · intro j
      dsimp only [current, lastResult]
      rw [lastExtension.stable base state _ (belowBounds _), belowValue]
      simp only [values_stable tail preceding base state (fun e h => leftBounds e (by simp [h]))]
      have frame := values_stable (right.drop j.val) preceding base state
        (fun e h => rightBounds e (List.mem_of_mem_drop h))
      simpa only [values, List.map_drop, Fin.val_cast] using congrArg (fun r =>
        decide (selected (values tail (memory base code state)) = selected r)) frame.symm
    · dsimp only [current, lastResult]
      change _ = decide (selected (values (head :: tail) (memory base _ state)) = [])
      rw [lastValue, emptyExpression_value _ _ _ leftBelow]
      rw [values_stable (head :: tail) lastExtension base state
        (Bounded.weaken leftBounds (Nat.add_le_add_left belowExtension.size_le base))]

def ordered {width : Nat} (base : Nat) (left right : List (Registers width)) (code : Store) : Nat × Store :=
  let table := orderedTable base left right code
  (table.1 0, table.2)

theorem ordered_correct {width : Nat} (base : Nat) (left right : List (Registers width))
    (code : Store) (state : Bits) (leftBounds : Bounded left (base + code.size))
    (rightBounds : Bounded right (base + code.size)) :
    let result := ordered base left right code
    Extends code result.2 ∧ result.1 < base + result.2.size ∧
      memory base result.2 state result.1 =
        decide (selected (values left (memory base code state)) = selected (values right (memory base code state))) := by
  obtain ⟨extension, bounds, same⟩ := orderedTable_correct base left right code state leftBounds rightBounds
  exact ⟨extension, bounds 0, by simpa using same 0⟩

end Semantic.Circuit.Shared
