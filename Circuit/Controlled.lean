import Circuit.Gadgets

namespace Semantic.Circuit

def Gate.controlled (control : Nat) (gate : Gate) : Gate :=
  if h : control ∈ gate.support then gate else
    Gate.mcx (control :: gate.controls) gate.target
      (by simp only [List.mem_cons, not_or]; exact ⟨by intro e; exact h (by simp [Gate.support, e]), gate.fresh⟩)
      (List.nodup_cons.mpr ⟨by intro e; exact h (by simp [Gate.support, e]), gate.distinct⟩)

def controlled (control : Nat) (circuit : Circuit) : Circuit :=
  circuit.map (Gate.controlled control)

theorem Gate.controlled_eval (control : Nat) (gate : Gate) (state : Bits)
    (outside : control ∉ gate.support) :
    (gate.controlled control).eval state = if state control then gate.eval state else state := by
  simp only [Gate.controlled, dif_neg outside, Gate.mcx, Gate.eval, Gate.enabled,
    List.all_cons]
  cases h : state control <;> simp []

theorem controlled_eval (control : Nat) (circuit : Circuit) (state : Bits)
    (outside : control ∉ support circuit) :
    eval (controlled control circuit) state = if state control then eval circuit state else state := by
  induction circuit generalizing state with
  | nil => simp [controlled, eval]
  | cons gate rest ih =>
    have absent : control ∉ gate.support ∧ control ∉ support rest := by
      simpa [support] using outside
    simp only [controlled, List.map_cons, eval]
    rw [Gate.controlled_eval _ _ _ absent.1]
    cases h : state control
    · simpa [h] using ih state absent.2
    · rw [if_pos (by simp), ← controlled, ih _ absent.2,
        gate.outside_support state control absent.1]
      simp [h]

theorem controlled_false (control : Nat) (circuit : Circuit) (state : Bits)
    (outside : control ∉ support circuit) (zero : state control = false) :
    eval (controlled control circuit) state = state := by rw [controlled_eval _ _ _ outside, zero]; rfl

theorem controlled_true (control : Nat) (circuit : Circuit) (state : Bits)
    (outside : control ∉ support circuit) (one : state control = true) :
    eval (controlled control circuit) state = eval circuit state := by rw [controlled_eval _ _ _ outside, one]; rfl

theorem controlled_clean (control : Nat) (circuit : Circuit) (state : Bits) (wire : Nat)
    (outside : control ∉ support circuit) (zero : state wire = false)
    (restored : eval circuit state wire = false) :
    eval (controlled control circuit) state wire = false := by
  rw [controlled_eval _ _ _ outside]
  cases state control <;> assumption

theorem controlled_length (control : Nat) (circuit : Circuit) :
    (controlled control circuit).length = circuit.length := List.length_map _

def Gate.rename (f : Nat → Nat) (injective : Function.Injective f) (gate : Gate) : Gate :=
  ⟨gate.controls.map f, f gate.target,
    by simpa only [List.mem_map, injective.eq_iff, exists_eq_right] using gate.fresh,
    gate.distinct.map injective⟩

def rename (f : Nat → Nat) (injective : Function.Injective f) (circuit : Circuit) : Circuit :=
  circuit.map (Gate.rename f injective)

theorem Gate.rename_view (f : Nat → Nat) (injective : Function.Injective f)
    (gate : Gate) (state : Bits) :
    (gate.rename f injective).eval state ∘ f = gate.eval (state ∘ f) := by
  funext wire
  by_cases h : wire = gate.target
  · subst wire; simp [Gate.eval, Gate.rename, Gate.enabled, List.all_map, Function.comp_def]
  · have different := (injective.ne h)
    simp [Gate.eval, Gate.rename, different, h, Function.comp_def]

theorem rename_view (f : Nat → Nat) (injective : Function.Injective f)
    (circuit : Circuit) (state : Bits) :
    eval (rename f injective circuit) state ∘ f = eval circuit (state ∘ f) := by
  induction circuit generalizing state with
  | nil => rfl
  | cons gate rest ih =>
    simp only [rename, List.map_cons, eval]
    rw [← rename, ih, Gate.rename_view]

theorem rename_update (f : Nat → Nat) (injective : Function.Injective f)
    (circuit : Circuit) (state : Bits) (target : Nat) (answer : Bool)
    (correct : eval circuit (state ∘ f) = Function.update (state ∘ f) target answer) :
    eval (rename f injective circuit) state = Function.update state (f target) answer := by
  funext wire
  by_cases image : ∃ original, f original = wire
  · obtain ⟨original, rfl⟩ := image
    have view := congrFun (rename_view f injective circuit state) original
    rw [correct] at view
    simpa only [Function.comp_apply, Function.update_apply, injective.eq_iff] using view
  · have absent : wire ∉ support (rename f injective circuit) := by
      simp only [support, rename, List.mem_flatMap, List.mem_map, Gate.support, Gate.rename]
      rintro ⟨gate, ⟨original, _, rfl⟩, member⟩
      rcases List.mem_cons.mp member with h | h
      · exact image ⟨original.target, h.symm⟩
      · obtain ⟨x, _, hx⟩ := List.mem_map.mp h
        exact image ⟨x, hx⟩
    rw [outside_support _ _ _ absent, Function.update_of_ne (by intro h; exact image ⟨target, h.symm⟩)]

end Semantic.Circuit
