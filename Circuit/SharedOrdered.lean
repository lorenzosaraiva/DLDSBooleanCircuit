import Circuit.SharedFlags

namespace Semantic.Circuit.Shared

set_option maxHeartbeats 30000000
set_option maxRecDepth 100000

def WireBounds {width : Nat} (wires : Fin width → Nat) (scope : Nat) : Prop := ∀ i, wires i < scope

theorem wire_values_stable {width : Nat} (wires : Fin width → Nat) {before after : Store}
    (extension : Extends before after) (base : Nat) (state : Bits)
    (bounded : WireBounds wires (base + before.size)) :
    (fun i => memory base after state (wires i)) = fun i => memory base before state (wires i) := by
  funext i
  exact extension.stable base state _ (bounded i)

def cellExpression {width : Nat} (scope : Nat) (head other : Registers width)
    (diagonal right below : Nat) : Expr scope :=
  (reference scope head.live).choose
    ((reference scope other.live).choose
      ((fieldEqual (head.expressions scope).payload (other.expressions scope).payload).conj
        (reference scope diagonal))
      (reference scope right))
    (reference scope below)

theorem cellExpression_value {width : Nat} (scope : Nat) (head other : Registers width)
    (diagonal right below : Nat) (state : Bits) (headBound : head.Bounded scope)
    (otherBound : other.Bounded scope) (diagBound : diagonal < scope)
    (rightBound : right < scope) (belowBound : below < scope) :
    (cellExpression scope head other diagonal right below).value state =
      if (head.value state).1 then
        if (other.value state).1 then decide ((head.value state).2 = (other.value state).2) && state diagonal
        else state right
      else state below := by
  simp only [cellExpression, Expr.value_choose, Expr.value, fieldEqual_value]
  rw [reference_value _ _ _ headBound.1, reference_value _ _ _ otherBound.1,
    reference_value _ _ _ diagBound, reference_value _ _ _ rightBound, reference_value _ _ _ belowBound]
  congr 2
  simp only [Registers.expressions, reference_value _ _ _ (headBound.2 _),
    reference_value _ _ _ (otherBound.2 _), Registers.value]
  rfl

def rowTruth {α : Type} [DecidableEq α] (head : Bool × α) :
    (right : List (Bool × α)) → (Fin (right.length + 1) → Bool) → Bool → Fin (right.length + 1) → Bool
  | [], _, last => fun _ => last
  | other :: tail, below, last =>
      let rest := rowTruth head tail (fun i => below i.succ) last
      Fin.cases
        (if head.1 then if other.1 then decide (head.2 = other.2) && below 1 else rest 0 else below 0)
        rest

def orderedRow {width : Nat} (base : Nat) (head : Registers width) :
    (right : List (Registers width)) → (Fin (right.length + 1) → Nat) → Nat → Store →
      (Fin (right.length + 1) → Nat) × Store
  | [], _, last, code => (fun _ => last, code)
  | other :: tail, below, last, code =>
      let rest := orderedRow base head tail (fun i => below i.succ) last code
      let cell := expression base (cellExpression (base + rest.2.size) head other (below 1) (rest.1 0) (below 0)) rest.2
      (Fin.cases cell.1 rest.1, cell.2)

theorem orderedRow_correct {width : Nat} (base : Nat) (head : Registers width)
    (right : List (Registers width)) (below : Fin (right.length + 1) → Nat) (last : Nat)
    (code : Store) (state : Bits) (headBound : head.Bounded (base + code.size))
    (rightBounds : Bounded right (base + code.size)) (belowBounds : WireBounds below (base + code.size))
    (lastBound : last < base + code.size) :
    let result := orderedRow base head right below last code
    Extends code result.2 ∧ WireBounds result.1 (base + result.2.size) ∧
      (fun i => memory base result.2 state (result.1 i)) =
        fun i => rowTruth (head.value (memory base code state)) (values right (memory base code state))
          (fun i => memory base code state (below (Fin.cast (by simp [values]) i)))
          (memory base code state last) (Fin.cast (by simp [values]) i) := by
  induction right generalizing code with
  | nil =>
    refine ⟨Extends.refl _, fun _ => lastBound, ?_⟩
    rfl
  | cons other tail ih =>
    obtain ⟨tailExtension, tailBounds, tailValue⟩ := ih (fun i => below i.succ) code
      headBound (fun e h => rightBounds e (by simp [h])) (fun i => belowBounds i.succ) lastBound
    have grow := Nat.add_le_add_left tailExtension.size_le base
    have hBound := Registers.weaken head headBound grow
    have oBound := Registers.weaken other (rightBounds other (by simp)) grow
    have bBound : ∀ i, below i < base + (orderedRow base head tail (fun i => below i.succ) last code).2.size :=
      fun i => lt_of_lt_of_le (belowBounds i) grow
    obtain ⟨cellExtension, cellBound, cellValue⟩ := expression_correct base
      (cellExpression (base + (orderedRow base head tail (fun i => below i.succ) last code).2.size)
        head other (below 1) ((orderedRow base head tail (fun i => below i.succ) last code).1 0) (below 0))
      (orderedRow base head tail (fun i => below i.succ) last code).2 (by omega) state
    refine ⟨tailExtension.trans cellExtension, ?_, ?_⟩
    · intro i
      refine Fin.cases ?_ (fun j => ?_) i
      · exact cellBound
      · exact lt_of_lt_of_le (tailBounds j) (Nat.add_le_add_left cellExtension.size_le base)
    · funext i
      refine Fin.cases ?_ (fun j => ?_) i
      · dsimp only [orderedRow, Fin.cases_zero]
        rw [cellValue, cellExpression_value _ _ _ _ _ _ _ hBound oBound (bBound 1) (tailBounds 0) (bBound 0)]
        rw [head.stable tailExtension base state headBound,
          other.stable tailExtension base state (rightBounds other (by simp)),
          tailExtension.stable base state (below 1) (belowBounds 1),
          tailExtension.stable base state (below 0) (belowBounds 0)]
        have tailHead := congrFun tailValue 0
        rw [tailHead]
        rfl
      · exact (cellExtension.stable base state _ (tailBounds j)).trans (congrFun tailValue j)

end Semantic.Circuit.Shared
