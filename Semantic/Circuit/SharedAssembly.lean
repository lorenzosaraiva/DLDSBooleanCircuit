import Semantic.Circuit.SharedProgram

namespace Semantic.Circuit.Shared

set_option maxHeartbeats 16000000
set_option maxRecDepth 100000



inductive Instruction where
  | constant (value : Bool)
  | copy (source : Nat)
  | neg (source : Nat)
  | conj (left right : Nat)

def Instruction.resolve (bound : Nat) : Instruction → Step bound
  | .constant b => .constant b
  | .copy a => if h : a < bound then .copy ⟨a, h⟩ else .constant false
  | .neg a => if h : a < bound then .neg ⟨a, h⟩ else .constant true
  | .conj a b => if ha : a < bound then
      if hb : b < bound then .conj ⟨a, ha⟩ ⟨b, hb⟩ else .constant false
    else .constant false

def assemble (base : Nat) : List Instruction → Sequence base
  | [] => .nil base
  | head :: tail => .cons (head.resolve base) (assemble (base + 1) tail)

@[simp] theorem assemble_length (base : Nat) (values : List Instruction) :
    (assemble base values).length = values.length := by
  induction values generalizing base <;> simp [assemble, Sequence.length, *, Nat.add_comm]

def run (base : Nat) (values : List Instruction) (state : Bits) : Bits := (assemble base values).run state

@[simp] theorem run_nil (base : Nat) (state : Bits) : run base [] state = state := rfl

theorem run_cons (base : Nat) (head : Instruction) (tail : List Instruction) (state : Bits) :
    run base (head :: tail) state =
      run (base + 1) tail (Function.update state base ((head.resolve base).value state)) := rfl

theorem run_append (base : Nat) (left right : List Instruction) (state : Bits) :
    run base (left ++ right) state = run (base + left.length) right (run base left state) := by
  induction left generalizing base state with
  | nil => simp
  | cons head tail ih =>
    simp only [List.cons_append, run_cons, ih, List.length_cons]
    congr 1 <;> omega

abbrev Store := Array Instruction

def memory (base : Nat) (code : Store) (state : Bits) : Bits := run base code.toList state

def Extends (before after : Store) : Prop := before.toList <+: after.toList

theorem Extends.refl (code : Store) : Extends code code := List.prefix_refl _

theorem Extends.trans {a b c : Store} (ab : Extends a b) (bc : Extends b c) : Extends a c := List.IsPrefix.trans ab bc

theorem Extends.size_le {a b : Store} (extension : Extends a b) : a.size ≤ b.size := by
  simpa using extension.length_le

theorem Extends.stable {a b : Store} (extension : Extends a b) (base : Nat)
    (state : Bits) (wire : Nat) (bound : wire < base + a.size) :
    memory base b state wire = memory base a state wire := by
  obtain ⟨tail, equal⟩ := extension
  unfold memory
  rw [← equal, run_append]
  exact (assemble (base + a.toList.length) tail).preserves_before _ wire (by simpa using bound)

theorem Extends.expression {a b : Store} (extension : Extends a b) (base : Nat)
    (state : Bits) {scope : Nat} (e : Expr scope) (bound : scope ≤ base + a.size) :
    e.value (memory base b state) = e.value (memory base a state) := by
  apply Expr.value_congr
  intro wire
  exact extension.stable base state wire.val (by have := wire.isLt; omega)

def emit (base : Nat) (op : Instruction) (code : Store) : Nat × Store :=
  (base + code.size, code.push op)

theorem emit_extends (base : Nat) (op : Instruction) (code : Store) :
    Extends code (emit base op code).2 := by simp [Extends, emit]

theorem memory_push (base : Nat) (code : Store) (op : Instruction) (state : Bits) :
    memory base (code.push op) state =
      Function.update (memory base code state) (base + code.size)
        ((op.resolve (base + code.size)).value (memory base code state)) := by
  simp only [memory, Array.toList_push, run_append, run_cons, run_nil,
    Array.length_toList, Nat.add_zero]

def expression {scope : Nat} (base : Nat) : Expr scope → Store → Nat × Store
  | .bit wire, code => (wire.val, code)
  | .constant b, code => emit base (.constant b) code
  | .neg child, code =>
      let result := expression base child code
      emit base (.neg result.1) result.2
  | .conj left right, code =>
      let first := expression base left code
      let second := expression base right first.2
      emit base (.conj first.1 second.1) second.2

theorem expression_correct {scope : Nat} (base : Nat) (e : Expr scope) (code : Store)
    (space : scope ≤ base + code.size) (state : Bits) :
    let result := expression base e code
    Extends code result.2 ∧ result.1 < base + result.2.size ∧
      memory base result.2 state result.1 = e.value (memory base code state) := by
  induction e generalizing code with
  | bit wire =>
    refine ⟨Extends.refl _, ?_, rfl⟩
    have := wire.isLt
    exact lt_of_lt_of_le this space
  | constant b =>
    refine ⟨emit_extends base (.constant b) code, by simp [expression, emit], ?_⟩
    simp [expression, emit, memory_push, Instruction.resolve, Step.value, Expr.value]
  | neg e ih =>
    obtain ⟨extension, bound, value⟩ := ih code space
    refine ⟨extension.trans (emit_extends base _ _), ?_, ?_⟩
    · simp [expression, emit]
    · simp only [expression, emit, memory_push, Function.update_self]
      simp only [Instruction.resolve, dif_pos bound, Step.value, Expr.value]
      rw [value]
  | conj a b ha hb =>
    obtain ⟨firstExtension, firstBound, firstValue⟩ := ha code space
    have nextSpace : scope ≤ base + (expression base a code).2.size := by
      have := firstExtension.size_le
      omega
    obtain ⟨secondExtension, secondBound, secondValue⟩ := hb (expression base a code).2 nextSpace
    have firstBoundLater : (expression base a code).1 <
        base + (expression base b (expression base a code).2).2.size := by
      have := secondExtension.size_le
      omega
    refine ⟨(firstExtension.trans secondExtension).trans (emit_extends base _ _), ?_, ?_⟩
    · simp [expression, emit]
    · simp only [expression, emit, memory_push, Function.update_self]
      simp only [Instruction.resolve, dif_pos firstBoundLater, dif_pos secondBound, Step.value, Expr.value]
      rw [secondExtension.stable base state _ firstBound, firstValue, secondValue,
        firstExtension.expression base state b space]

theorem expression_steps {scope : Nat} (base : Nat) (e : Expr scope) (code : Store) :
    (expression base e code).2.size ≤ code.size + e.size := by
  induction e generalizing code with
  | bit wire => simp [expression, Expr.size]
  | constant b => simp [expression, emit, Expr.size]
  | neg e ih =>
    have := ih code
    simp only [expression, emit, Array.size_push, Expr.size]
    omega
  | conj a b ha hb =>
    have first := ha code
    have second := hb (expression base a code).2
    simp only [expression, emit, Array.size_push, Expr.size]
    omega

def expressionProgram {n : Nat} (e : Expr n) : Program n :=
  let result := expression n e #[]
  ⟨assemble n result.2.toList, ⟨result.1, by
    simpa using (expression_correct n e #[] (by simp) (fun _ => false)).2.1⟩⟩

theorem expressionProgram_value {n : Nat} (e : Expr n) (state : Bits) :
    (expressionProgram e).value state = e.value state := by
  exact (expression_correct n e #[] (by simp) state).2.2

theorem expression_oracle {n : Nat} (e : Expr n) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (expressionProgram e).oracle state =
      Function.update state (expressionProgram e).target (e.value state) := by
  rw [Program.oracle_correct _ state clean, expressionProgram_value]

end Semantic.Circuit.Shared
