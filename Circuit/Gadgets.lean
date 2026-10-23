import Circuit.Expression

namespace Semantic.Circuit

set_option maxHeartbeats 3000000



theorem conjugate_gate (circuit : Circuit) (gate : Gate) (state : Bits)
    (fresh : gate.target ∉ support circuit) :
    eval (circuit ++ [gate] ++ inverse circuit) state =
      Function.update state gate.target
        (state gate.target ^^ gate.enabled (eval circuit state)) := by
  simp only [eval_append, eval]
  unfold Gate.eval
  rw [eval_update _ _ _ _ (by simpa only [support_inverse] using fresh), inverse_undoes]
  rw [outside_support circuit state gate.target fresh]

theorem compute_copy_uncompute (circuit : Circuit) (source target : Nat) (state : Bits)
    (separate : target ≠ source) (fresh : target ∉ support circuit) :
    eval (circuit ++ [Gate.cnot source target separate] ++ inverse circuit) state =
      Function.update state target (state target ^^ eval circuit state source) := by
  simpa only [Gate.enabled, Gate.cnot, List.all_cons, List.all_nil, Bool.and_true] using
    conjugate_gate circuit (Gate.cnot source target separate) state fresh

theorem compute_copy_uncompute_clean (circuit : Circuit) (source target : Nat)
    (f : Bits → Bool) (state : Bits) (separate : target ≠ source)
    (fresh : target ∉ support circuit) (zero : state target = false)
    (computes : eval circuit state source = f state) :
    eval (circuit ++ [Gate.cnot source target separate] ++ inverse circuit) state =
      Function.update state target (f state) := by
  rw [compute_copy_uncompute _ _ _ _ separate fresh, zero, Bool.false_xor, computes]

def Expr.oracle {n : Nat} (e : Expr n) : Circuit :=
  e.compile n (Nat.le_refl n) ++
    [Gate.cnot n (n + e.size) (by have := e.size_positive; omega)] ++
    inverse (e.compile n (Nat.le_refl n))

theorem Expr.oracle_correct {n : Nat} (e : Expr n) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval e.oracle state = Function.update state (n + e.size) (e.value state) := by
  have fresh : n + e.size ∉ support (e.compile n (Nat.le_refl n)) := by
    intro member
    obtain ⟨gate, member, contained⟩ := List.mem_flatMap.mp member
    have := (e.compile_bounds n (Nat.le_refl n) gate member).2 _ contained
    omega
  rw [Expr.oracle, compute_copy_uncompute _ _ _ _ _ fresh,
    clean _ (by omega), Bool.false_xor, e.compile_correct n _ state clean]

theorem Expr.oracle_restores {n : Nat} (e : Expr n) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) (wire : Nat)
    (different : wire ≠ n + e.size) : eval e.oracle state wire = state wire := by
  rw [e.oracle_correct state clean]
  simp [different]

def Expr.disj {n : Nat} (left right : Expr n) : Expr n := .neg (.conj (.neg left) (.neg right))
def Expr.all {n : Nat} (values : List (Expr n)) : Expr n := values.foldr Expr.conj (.constant true)
def Expr.any {n : Nat} (values : List (Expr n)) : Expr n := values.foldr Expr.disj (.constant false)

theorem Expr.value_disj {n : Nat} (left right : Expr n) (state : Bits) :
    (left.disj right).value state = (left.value state || right.value state) := by
  simp [Expr.disj, Expr.value, Bool.not_and]

theorem Expr.value_all {n : Nat} (values : List (Expr n)) (state : Bits) :
    (Expr.all values).value state = values.all (fun e => e.value state) := by
  induction values with
  | nil => rfl
  | cons head rest ih =>
    change (head.value state && (Expr.all rest).value state) = _
    rw [ih]
    rfl

theorem Expr.value_any {n : Nat} (values : List (Expr n)) (state : Bits) :
    (Expr.any values).value state = values.any (fun e => e.value state) := by
  induction values with
  | nil => rfl
  | cons head rest ih =>
    change (head.disj (Expr.any rest)).value state = _
    rw [Expr.value_disj, ih]
    rfl

def equalConstant {n k : Nat} (wires : Fin k → Fin n) (constant : Fin k → Bool) : Expr n :=
  Expr.all ((List.finRange k).map fun i => if constant i then .bit (wires i) else .neg (.bit (wires i)))

theorem equalConstant_value {n k : Nat} (wires : Fin k → Fin n) (constant : Fin k → Bool) (state : Bits) :
    (equalConstant wires constant).value state =
      (List.finRange k).all (fun i => state (wires i).val == constant i) := by
  rw [equalConstant, Expr.value_all, List.all_map]
  apply all_congr_mem
  intro i _
  cases h : constant i <;> simp [h, Expr.value]

theorem equality_gadget {n k : Nat} (wires : Fin k → Fin n) (constant : Fin k → Bool)
    (state : Bits) (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (equalConstant wires constant).oracle state =
      Function.update state (n + (equalConstant wires constant).size)
        ((List.finRange k).all (fun i => state (wires i).val == constant i)) := by
  rw [Expr.oracle_correct _ state clean, equalConstant_value]

theorem and_gadget {n : Nat} (wires : List (Fin n)) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (Expr.all (wires.map Expr.bit)).oracle state =
      Function.update state (n + (Expr.all (wires.map Expr.bit)).size)
        (wires.all (fun wire => state wire.val)) := by
  rw [Expr.oracle_correct _ state clean, Expr.value_all]
  simp only [List.all_map, Function.comp_def, Expr.value]

theorem or_gadget {n : Nat} (wires : List (Fin n)) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (Expr.any (wires.map Expr.bit)).oracle state =
      Function.update state (n + (Expr.any (wires.map Expr.bit)).size)
        (wires.any (fun wire => state wire.val)) := by
  rw [Expr.oracle_correct _ state clean, Expr.value_any]
  simp only [List.any_map, Function.comp_def, Expr.value]

theorem and_gate (controls : List Nat) (target : Nat) (fresh : target ∉ controls)
    (distinct : controls.Nodup) (state : Bits) :
    eval [Gate.mcx controls target fresh distinct] state =
      Function.update state target (state target ^^ controls.all state) := rfl

end Semantic.Circuit
