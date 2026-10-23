import Mathlib.Data.List.Basic
import Mathlib.Tactic

namespace Semantic.Circuit

abbrev Bits := Nat → Bool

theorem all_congr_mem {α : Type} (values : List α) (left right : α → Bool)
    (agree : ∀ x ∈ values, left x = right x) : values.all left = values.all right := by
  induction values with
  | nil => rfl
  | cons head rest ih =>
    simp only [List.all_cons]
    rw [agree head (by simp), ih (fun x member => agree x (by simp [member]))]



structure Gate where
  controls : List Nat
  target : Nat
  fresh : target ∉ controls
  distinct : controls.Nodup

def Gate.x (target : Nat) : Gate := ⟨[], target, by simp, by simp⟩
def Gate.cnot (control target : Nat) (fresh : target ≠ control) : Gate :=
  ⟨[control], target, by simpa, by simp⟩
def Gate.toffoli (left right target : Nat) (different : left ≠ right)
    (leftFresh : target ≠ left) (rightFresh : target ≠ right) : Gate :=
  ⟨[left, right], target, by simp_all, by simp_all⟩
def Gate.mcx (controls : List Nat) (target : Nat) (fresh : target ∉ controls)
    (distinct : controls.Nodup) : Gate := ⟨controls, target, fresh, distinct⟩

def Gate.support (gate : Gate) : List Nat := gate.target :: gate.controls
def Gate.enabled (gate : Gate) (state : Bits) : Bool := gate.controls.all state
def Gate.eval (gate : Gate) (state : Bits) : Bits :=
  Function.update state gate.target (state gate.target ^^ gate.enabled state)

theorem Gate.eval_other (gate : Gate) (state : Bits) (wire : Nat) (different : wire ≠ gate.target) :
    gate.eval state wire = state wire := by simp [Gate.eval, different]

theorem Gate.eval_target (gate : Gate) (state : Bits) :
    gate.eval state gate.target = (state gate.target ^^ gate.enabled state) := by simp [Gate.eval]

theorem Gate.enabled_eval (gate : Gate) (state : Bits) :
    gate.enabled (gate.eval state) = gate.enabled state := by
  apply all_congr_mem
  intro wire member
  apply gate.eval_other
  intro equal
  subst wire
  exact gate.fresh member

theorem Gate.involution (gate : Gate) (state : Bits) : gate.eval (gate.eval state) = state := by
  funext wire
  by_cases same : wire = gate.target
  · subst wire
    rw [Gate.eval_target, Gate.enabled_eval, Gate.eval_target]
    simp [Bool.xor_assoc]
  · simp [Gate.eval_other, same]

theorem Gate.outside_support (gate : Gate) (state : Bits) (wire : Nat)
    (outside : wire ∉ gate.support) : gate.eval state wire = state wire :=
  gate.eval_other state wire ((by simpa [Gate.support] using outside :
    wire ≠ gate.target ∧ wire ∉ gate.controls).1)

abbrev Circuit := List Gate

def eval : Circuit → Bits → Bits
  | [], state => state
  | gate :: rest, state => eval rest (gate.eval state)

def inverse (circuit : Circuit) : Circuit := circuit.reverse
def support (circuit : Circuit) : List Nat := circuit.flatMap Gate.support

theorem eval_append (left right : Circuit) (state : Bits) :
    eval (left ++ right) state = eval right (eval left state) := by
  induction left generalizing state with
  | nil => rfl
  | cons gate rest ih => exact ih (gate.eval state)

theorem inverse_undoes (circuit : Circuit) (state : Bits) :
    eval (inverse circuit) (eval circuit state) = state := by
  induction circuit generalizing state with
  | nil => rfl
  | cons gate rest ih =>
    simp only [inverse, List.reverse_cons, eval_append, eval]
    rw [show eval (List.reverse rest) (eval rest (gate.eval state)) = gate.eval state from ih _]
    exact gate.involution state

theorem inverse_inverse (circuit : Circuit) : inverse (inverse circuit) = circuit := List.reverse_reverse _

theorem outside_support (circuit : Circuit) (state : Bits) (wire : Nat)
    (outside : wire ∉ support circuit) : eval circuit state wire = state wire := by
  induction circuit generalizing state with
  | nil => rfl
  | cons gate rest ih =>
    have absent : wire ∉ gate.support ∧ wire ∉ support rest := by simpa [support] using outside
    exact (ih (gate.eval state) absent.2).trans (gate.outside_support state wire absent.1)

theorem Gate.eval_update (gate : Gate) (state : Bits) (wire : Nat) (bit : Bool)
    (outside : wire ∉ gate.support) :
    gate.eval (Function.update state wire bit) = Function.update (gate.eval state) wire bit := by
  have absent : wire ≠ gate.target ∧ wire ∉ gate.controls := by simpa [Gate.support] using outside
  have targetNe : wire ≠ gate.target := absent.1
  have controls : wire ∉ gate.controls := absent.2
  have enabled : gate.enabled (Function.update state wire bit) = gate.enabled state := by
    apply all_congr_mem
    intro control member
    have different : control ≠ wire := by rintro rfl; exact controls member
    simp [different]
  funext query
  by_cases own : query = wire
  · subst query; simp [Gate.eval, targetNe]
  · by_cases target : query = gate.target
    · subst query; simp [Gate.eval, targetNe, Ne.symm targetNe, enabled]
    · simp [Gate.eval, own, target]

theorem eval_update (circuit : Circuit) (state : Bits) (wire : Nat) (bit : Bool)
    (outside : wire ∉ support circuit) :
    eval circuit (Function.update state wire bit) = Function.update (eval circuit state) wire bit := by
  induction circuit generalizing state with
  | nil => rfl
  | cons gate rest ih =>
    have absent : wire ∉ gate.support ∧ wire ∉ support rest := by simpa [support] using outside
    simp only [eval, gate.eval_update state wire bit absent.1]
    exact ih (gate.eval state) absent.2

theorem support_inverse (circuit : Circuit) (wire : Nat) :
    wire ∈ support (inverse circuit) ↔ wire ∈ support circuit := by
  simp [support, inverse, List.mem_flatMap]

end Semantic.Circuit
