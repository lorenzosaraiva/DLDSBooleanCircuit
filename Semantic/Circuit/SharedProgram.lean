import Semantic.Circuit.Gadgets

namespace Semantic.Circuit
namespace Shared

set_option maxHeartbeats 12000000
set_option maxRecDepth 30000


inductive Step (available : Nat) where
  | constant (value : Bool)
  | copy (source : Fin available)
  | neg (source : Fin available)
  | conj (left right : Fin available)

def Step.value {n : Nat} : Step n → Bits → Bool
  | .constant b, _ => b
  | .copy a, state => state a.val
  | .neg a, state => !(state a.val)
  | .conj a b, state => state a.val && state b.val

def Step.compile {n : Nat} : Step n → Circuit
  | .constant b => if b then [Gate.x n] else []
  | .copy a => [Gate.cnot a.val n (by have := a.isLt; omega)]
  | .neg a => [Gate.cnot a.val n (by have := a.isLt; omega), Gate.x n]
  | .conj a b => if same : a.val = b.val then
      [Gate.cnot a.val n (by have := a.isLt; omega)]
    else [Gate.toffoli a.val b.val n same (by have := a.isLt; omega) (by have := b.isLt; omega)]

theorem Step.correct {n : Nat} (step : Step n) (state : Bits) (zero : state n = false) :
    eval step.compile state = Function.update state n (step.value state) := by
  cases step with
  | constant b => cases b <;> simp [Step.compile, Step.value, eval, Gate.eval, Gate.enabled, Gate.x, zero]
  | copy a => simp [Step.compile, Step.value, eval, Gate.eval, Gate.enabled, Gate.cnot, zero]
  | neg a => simp [Step.compile, Step.value, eval, Gate.eval, Gate.enabled, Gate.cnot, Gate.x, zero]
  | conj a b =>
    by_cases same : a.val = b.val
    · simp [Step.compile, Step.value, eval, Gate.eval, Gate.enabled, Gate.cnot, zero, same]
    · simp [Step.compile, Step.value, eval, Gate.eval, Gate.enabled, Gate.toffoli, zero, same]

theorem Step.bounds {n : Nat} (step : Step n) :
    ∀ gate ∈ step.compile, gate.target = n ∧ ∀ wire ∈ gate.support, wire < n + 1 := by
  cases step with
  | constant b => cases b <;> simp [Step.compile, Gate.x, Gate.support]
  | copy a =>
    intro gate member
    have bound := a.isLt
    simp only [Step.compile, List.mem_singleton] at member
    subst gate
    simp [Gate.cnot, Gate.support] <;> omega
  | neg a =>
    intro gate member
    have bound := a.isLt
    simp only [Step.compile, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl <;> simp [Gate.cnot, Gate.x, Gate.support] <;> omega
  | conj a b =>
    intro gate member
    have left := a.isLt
    have right := b.isLt
    by_cases same : a.val = b.val
    · simp only [Step.compile, dif_pos same, List.mem_singleton] at member
      subst gate
      simp [Gate.cnot, Gate.support] <;> omega
    · simp only [Step.compile, dif_neg same, List.mem_singleton] at member
      subst gate
      simp [Gate.toffoli, Gate.support] <;> omega

theorem Step.gate_bound {n : Nat} (step : Step n) : step.compile.length ≤ 2 := by
  cases step with
  | constant b => cases b <;> simp [Step.compile]
  | copy a => simp [Step.compile]
  | neg a => simp [Step.compile]
  | conj a b => by_cases h : a.val = b.val <;> simp [Step.compile, h]



inductive Sequence : Nat → Type where
  | nil (n : Nat) : Sequence n
  | cons {n : Nat} (head : Step n) (tail : Sequence (n + 1)) : Sequence n

def Sequence.length {n : Nat} : Sequence n → Nat
  | .nil _ => 0
  | .cons _ tail => 1 + tail.length

def Sequence.compile {n : Nat} : Sequence n → Circuit
  | .nil _ => []
  | .cons head tail => head.compile ++ tail.compile

def Sequence.run {n : Nat} : Sequence n → Bits → Bits
  | .nil _, state => state
  | .cons head tail, state => tail.run (Function.update state n (head.value state))

theorem Sequence.preserves_before {n : Nat} (code : Sequence n) (state : Bits) (wire : Nat)
    (lower : wire < n) : code.run state wire = state wire := by
  induction code generalizing state wire with
  | nil => rfl
  | @cons n head tail ih =>
    rw [Sequence.run, ih _ wire (by omega)]
    simp [show wire ≠ n by omega]

theorem Sequence.above {n : Nat} (code : Sequence n) (state : Bits) (wire : Nat)
    (upper : n + code.length ≤ wire) : code.run state wire = state wire := by
  induction code generalizing state wire with
  | nil => rfl
  | @cons n head tail ih =>
    simp only [Sequence.length] at upper
    rw [Sequence.run, ih _ wire (by omega)]
    simp [show wire ≠ n by omega]

theorem Sequence.correct {n : Nat} (code : Sequence n) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) : eval code.compile state = code.run state := by
  induction code generalizing state with
  | nil => rfl
  | @cons n head tail ih =>
    rw [Sequence.compile, eval_append, head.correct state (clean n (by omega)),
      ih (Function.update state n (head.value state))]
    · rfl
    · intro wire upper
      simp [show wire ≠ n by omega, clean wire (by omega)]

theorem Sequence.bounds {n : Nat} (code : Sequence n) :
    ∀ gate ∈ code.compile, n ≤ gate.target ∧ ∀ wire ∈ gate.support, wire < n + code.length := by
  induction code with
  | nil => simp [Sequence.compile]
  | @cons n head tail ih =>
    intro gate member
    simp only [Sequence.compile, List.mem_append] at member
    rcases member with first | later
    · obtain ⟨rfl, bound⟩ := head.bounds gate first
      refine ⟨by omega, ?_⟩
      intro wire contained
      have := bound wire contained
      simp only [Sequence.length]
      omega
    · obtain ⟨target, bound⟩ := ih gate later
      refine ⟨by omega, ?_⟩
      intro wire contained
      have := bound wire contained
      simp only [Sequence.length]
      omega

def Sequence.valueAt {n : Nat} : Sequence n → Bits → Nat → Bool
  | .nil _, _, _ => false
  | .cons head _, state, 0 => head.value state
  | .cons head tail, state, index + 1 =>
      tail.valueAt (Function.update state n (head.value state)) index

theorem Sequence.step_values {n : Nat} (code : Sequence n) (state : Bits) (index : Nat)
    (inside : index < code.length) : code.run state (n + index) = code.valueAt state index := by
  induction code generalizing state index with
  | nil => simp [Sequence.length] at inside
  | @cons n head tail ih =>
    cases index with
    | zero =>
      simp only [Sequence.run, Sequence.valueAt, Nat.add_zero]
      rw [tail.preserves_before _ n (by omega)]
      simp
    | succ index =>
      simp only [Sequence.length] at inside
      simpa [Sequence.run, Sequence.valueAt, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        ih (Function.update state n (head.value state)) index (by omega)

theorem Sequence.gate_bound {n : Nat} (code : Sequence n) : code.compile.length ≤ 2 * code.length := by
  induction code with
  | nil => simp [Sequence.compile, Sequence.length]
  | cons head tail ih =>
    have := head.gate_bound
    simp only [Sequence.compile, List.length_append, Sequence.length]
    omega

structure Program (inputs : Nat) where
  code : Sequence inputs
  output : Fin (inputs + code.length)

def Program.wires {n : Nat} (program : Program n) : Nat := n + program.code.length + 1

def Program.target {n : Nat} (program : Program n) : Nat := n + program.code.length

def Program.value {n : Nat} (program : Program n) (state : Bits) : Bool :=
  program.code.run state program.output.val

def Program.oracle {n : Nat} (program : Program n) : Circuit :=
  program.code.compile ++ [Gate.cnot program.output.val program.target (by have := program.output.isLt; simp [Program.target]; omega)] ++
    inverse program.code.compile

theorem Program.oracle_correct {n : Nat} (program : Program n) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval program.oracle state = Function.update state program.target (program.value state) := by
  have fresh : program.target ∉ support program.code.compile := by
    intro member
    obtain ⟨gate, contained, own⟩ := List.mem_flatMap.mp member
    have := (program.code.bounds gate contained).2 _ own
    simp only [Program.target] at *
    omega
  rw [Program.oracle, compute_copy_uncompute _ _ _ _ _ fresh,
    clean _ (by simp [Program.target]), Bool.false_xor, program.code.correct state clean]
  rfl

theorem Program.wire_count {n : Nat} (program : Program n) :
    program.wires = n + program.code.length + 1 := rfl

theorem Program.gate_bound {n : Nat} (program : Program n) :
    program.oracle.length ≤ 4 * program.code.length + 1 := by
  have := program.code.gate_bound
  simp [Program.oracle, inverse]
  omega

end Shared
end Semantic.Circuit
