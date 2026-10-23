import Semantic.Circuit.Basic

namespace Semantic.Circuit

set_option maxHeartbeats 3000000

inductive Expr (inputs : Nat) where
  | bit (wire : Fin inputs)
  | constant (value : Bool)
  | neg (value : Expr inputs)
  | conj (left right : Expr inputs)

def Expr.value {n : Nat} : Expr n → Bits → Bool
  | .bit wire, state => state wire.val
  | .constant value, _ => value
  | .neg e, state => !(e.value state)
  | .conj a b, state => a.value state && b.value state

def Expr.size {n : Nat} : Expr n → Nat
  | .bit _ | .constant _ => 1
  | .neg e => 1 + e.size
  | .conj a b => 1 + a.size + b.size

theorem Expr.size_positive {n : Nat} (e : Expr n) : 0 < e.size := by cases e <;> simp [Expr.size] <;> omega

theorem Expr.value_congr {n : Nat} (e : Expr n) (left right : Bits)
    (same : ∀ wire : Fin n, left wire.val = right wire.val) : e.value left = e.value right := by
  induction e with
  | bit wire => exact same wire
  | constant _ => rfl
  | neg e ih => exact congrArg Bool.not ih
  | conj a b ha hb => simp only [Expr.value, ha, hb]


def Expr.compile {n : Nat} (e : Expr n) (base : Nat) (space : n ≤ base) : Circuit :=
  match e with
  | .bit wire => [Gate.cnot wire.val base (by have := wire.isLt; omega)]
  | .constant value => if value then [Gate.x base] else []
  | .neg child =>
      child.compile (base + 1) (by omega) ++
        [Gate.cnot (base + 1) base (by omega), Gate.x base]
  | .conj left right =>
      left.compile (base + 1) (by omega) ++
      right.compile (base + 1 + left.size) (by omega) ++
      [Gate.toffoli (base + 1) (base + 1 + left.size) base
        (by have := left.size_positive; omega) (by omega) (by have := left.size_positive; omega)]

theorem Expr.compile_bounds {n : Nat} (e : Expr n) (base : Nat) (space : n ≤ base) :
    ∀ gate ∈ e.compile base space, base ≤ gate.target ∧
      ∀ wire ∈ gate.support, wire < base + e.size := by
  induction e generalizing base with
  | bit wire =>
    intro gate member
    simp only [Expr.compile, List.mem_singleton] at member
    subst gate
    constructor
    · exact Nat.le_refl _
    · intro q member
      simp only [Gate.support, Gate.cnot, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at member
      have := wire.isLt
      rcases member with rfl | rfl <;> simp only [Expr.size] <;> omega
  | constant bit =>
    intro gate member
    cases bit with
    | false => simp [Expr.compile] at member
    | true =>
      have equal : gate = Gate.x base := by simpa [Expr.compile] using member
      subst gate
      simp [Gate.support, Gate.x, Expr.size]
  | neg e ih =>
    intro gate member
    simp only [Expr.compile, List.mem_append, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at member
    rcases member with child | rfl | rfl
    · obtain ⟨lower, upper⟩ := ih (base+1) (by omega) gate child
      refine ⟨by omega, fun q hq => ?_⟩
      have := upper q hq
      simp only [Expr.size]
      omega
    · refine ⟨Nat.le_refl _, ?_⟩
      intro q member
      have := e.size_positive
      simp only [Gate.support, Gate.cnot, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl <;> simp only [Expr.size] <;> omega
    · simp [Gate.support, Gate.x, Expr.size]
  | conj a b ha hb =>
    intro gate member
    simp only [Expr.compile, List.mem_append, List.mem_singleton] at member
    rcases member with (child | child) | rfl
    · obtain ⟨lower, upper⟩ := ha (base+1) (by omega) gate child
      refine ⟨by omega, fun q hq => ?_⟩
      have := upper q hq
      simp only [Expr.size]
      omega
    · obtain ⟨lower, upper⟩ := hb (base+1+a.size) (by omega) gate child
      refine ⟨by omega, fun q hq => ?_⟩
      have := upper q hq
      simp only [Expr.size]
      omega
    · refine ⟨Nat.le_refl _, ?_⟩
      intro q member
      have := a.size_positive
      have := b.size_positive
      simp only [Gate.support, Gate.toffoli, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl <;> simp only [Expr.size] <;> omega

theorem eval_fixed_targets (circuit : Circuit) (state : Bits) (wire : Nat)
    (different : ∀ gate ∈ circuit, wire ≠ gate.target) : eval circuit state wire = state wire := by
  induction circuit generalizing state with
  | nil => rfl
  | cons gate rest ih =>
    exact (ih (gate.eval state) (fun g hg => different g (by simp [hg]))).trans
      (gate.eval_other state wire (different gate (by simp)))

theorem Expr.compile_below {n : Nat} (e : Expr n) (base : Nat) (space : n ≤ base)
    (state : Bits) (wire : Nat) (below : wire < base) :
    eval (e.compile base space) state wire = state wire := by
  apply eval_fixed_targets
  intro gate member
  have := (e.compile_bounds base space gate member).1
  omega

theorem Expr.compile_above {n : Nat} (e : Expr n) (base : Nat) (space : n ≤ base)
    (state : Bits) (wire : Nat) (above : base + e.size ≤ wire) :
    eval (e.compile base space) state wire = state wire := by
  apply outside_support
  intro member
  obtain ⟨gate, member, contained⟩ := List.mem_flatMap.mp member
  have := (e.compile_bounds base space gate member).2 wire contained
  omega

theorem Expr.compile_correct {n : Nat} (e : Expr n) (base : Nat) (space : n ≤ base)
    (state : Bits) (clean : ∀ wire, base ≤ wire → state wire = false) :
    eval (e.compile base space) state base = e.value state := by
  induction e generalizing base state with
  | bit wire => simp [Expr.compile, eval, Gate.eval, Gate.enabled, clean base (by omega), Expr.value, Gate.cnot]
  | constant bit => cases bit <;> simp [Expr.compile, eval, Gate.eval, Gate.enabled, Gate.x, Expr.value, clean base (by omega)]
  | neg e ih =>
    have output := ih (base+1) (by omega) state (fun q hq => clean q (by omega))
    have untouched := e.compile_below (base+1) (by omega) state base (by omega)
    simp only [Expr.compile, eval_append, eval]
    simp [Gate.eval, Gate.enabled, Gate.x, Gate.cnot, output, untouched, Expr.value,
      clean base (by omega), show base + 1 ≠ base by omega]
  | conj a b ha hb =>
    let middle := eval (a.compile (base+1) (by omega)) state
    have cleanB : ∀ q, base+1+a.size ≤ q → middle q = false := by
      intro q hq
      rw [show middle q = state q from a.compile_above _ _ state q hq]
      exact clean q (by omega)
    have aOutput := ha (base+1) (by omega) state (fun q hq => clean q (by omega))
    have bOutput := hb (base+1+a.size) (by omega) middle cleanB
    have bValue : b.value middle = b.value state := b.value_congr _ _ (fun wire =>
      a.compile_below _ _ state wire.val (by have := wire.isLt; omega))
    have leftKeep := b.compile_below (base+1+a.size) (by omega) middle (base+1)
      (by have := a.size_positive; omega)
    have rootKeep := b.compile_below (base+1+a.size) (by omega) middle base (by omega)
    have rootFirst := a.compile_below (base+1) (by omega) state base (by omega)
    simp only [Expr.compile, eval_append, eval]
    change (Gate.toffoli (base+1) (base+1+a.size) base _ _ _).eval
      (eval (b.compile (base+1+a.size) _) middle) base = _
    simp only [Gate.eval, Function.update_self, Gate.enabled, Gate.toffoli, List.all_cons, List.all_nil, Bool.and_true]
    rw [rootKeep, show middle base = state base from rootFirst, clean base (by omega), Bool.false_xor, leftKeep,
      show middle (base+1) = a.value state from aOutput, bOutput, bValue]
    rfl

end Semantic.Circuit
