import Semantic.Circuit.Gadgets

namespace Semantic.Circuit

def loadWires (control : Nat) (targets : List Nat) (separate : control ∉ targets) : Circuit :=
  targets.attach.map fun target => Gate.cnot control target.val
    (by intro equal; exact separate (equal ▸ target.property))

theorem loadWires_eval (control : Nat) (targets : List Nat) (separate : control ∉ targets)
    (distinct : targets.Nodup) (state : Bits) (wire : Nat) :
    eval (loadWires control targets separate) state wire =
      if wire ∈ targets then state wire ^^ state control else state wire := by
  induction targets generalizing state with
  | nil => rfl
  | cons head tail ih =>
    have headNe : head ≠ control := by intro e; exact separate (by simp [e])
    have tailSep : control ∉ tail := fun h => separate (by simp [h])
    have nd := List.nodup_cons.mp distinct
    have expansion : loadWires control (head :: tail) separate =
        Gate.cnot control head headNe :: loadWires control tail tailSep := by
      simp [loadWires, List.attach_cons, List.map_pmap]
    rw [expansion]
    change eval (loadWires control tail tailSep) ((Gate.cnot control head headNe).eval state) wire = _
    rw [ih tailSep nd.2]
    have controlFixed : (Gate.cnot control head headNe).eval state control = state control := by
      apply Gate.eval_other
      exact Ne.symm headNe
    rw [controlFixed]
    by_cases own : wire = head
    · subst wire
      simp [nd.1, Gate.eval, Gate.enabled, Gate.cnot]
    · simp [own, Gate.eval_other, Gate.cnot]

def loadingTargets (base : Nat) (data : List Bool) : List Nat :=
  ((List.finRange data.length).filter fun i => data.get i).map fun i => base + i.val

theorem loadingTargets_nodup (base : Nat) (data : List Bool) :
    (loadingTargets base data).Nodup := by
  apply List.Nodup.map
  · intro a b h
    apply Fin.ext
    change base + a.val = base + b.val at h
    omega
  · exact (List.nodup_finRange _).filter _

theorem loadingTargets_mem (base : Nat) (data : List Bool) (wire : Nat) :
    wire ∈ loadingTargets base data ↔
      ∃ i : Fin data.length, wire = base + i.val ∧ data.get i = true := by
  simp only [loadingTargets, List.mem_map, List.mem_filter, List.mem_finRange, true_and]
  constructor
  · rintro ⟨i, active, equal⟩; exact ⟨i, equal.symm, active⟩
  · rintro ⟨i, equal, active⟩; exact ⟨i, active, equal.symm⟩

theorem loadingTargets_separate (control base : Nat) (data : List Bool) (before : control < base) :
    control ∉ loadingTargets base data := by
  rw [loadingTargets_mem]
  rintro ⟨i, equal, _⟩
  omega

def loadData (control base : Nat) (data : List Bool) (before : control < base) : Circuit :=
  loadWires control (loadingTargets base data) (loadingTargets_separate control base data before)

theorem loadData_value (control base : Nat) (data : List Bool) (before : control < base)
    (state : Bits) (i : Fin data.length) (selected : state control = true)
    (zero : state (base + i.val) = false) :
    eval (loadData control base data before) state (base + i.val) = data.get i := by
  rw [loadData, loadWires_eval _ _ _ (loadingTargets_nodup _ _), selected, zero]
  have member : base + i.val ∈ loadingTargets base data ↔ data.get i = true := by
    rw [loadingTargets_mem]
    constructor
    · rintro ⟨j, equal, active⟩
      have same : i = j := Fin.ext (by omega)
      simpa [same] using active
    · intro active; exact ⟨i, rfl, active⟩
  simp only [member]
  cases h : data.get i <;> simp [h]

theorem loadData_inactive (control base : Nat) (data : List Bool) (before : control < base)
    (state : Bits) (inactive : state control = false) :
    eval (loadData control base data before) state = state := by
  funext wire
  rw [loadData, loadWires_eval _ _ _ (loadingTargets_nodup _ _), inactive]
  split <;> simp

theorem unloadData_restores (control base : Nat) (data : List Bool) (before : control < base)
    (state : Bits) :
    eval (inverse (loadData control base data before))
      (eval (loadData control base data before) state) = state := inverse_undoes _ _

theorem loadData_outside (control base : Nat) (data : List Bool) (before : control < base)
    (state : Bits) (wire : Nat) (outside : wire < base ∨ base + data.length ≤ wire) :
    eval (loadData control base data before) state wire = state wire := by
  rw [loadData, loadWires_eval _ _ _ (loadingTargets_nodup _ _)]
  have absent : wire ∉ loadingTargets base data := by
    rw [loadingTargets_mem]
    rintro ⟨i, equal, _⟩
    rcases outside with h | h <;> have := i.isLt <;> omega
  simp [absent]

theorem loadData_view (control base count : Nat) (input : Bits) (before : control < base)
    (state : Bits) (selected : state control = true)
    (clean : ∀ wire, base ≤ wire → state wire = false)
    (inputClean : ∀ wire, count ≤ wire → input wire = false) :
    eval (loadData control base ((List.finRange count).map fun i => input i.val) before) state ∘
      (fun wire => base + wire) = input := by
  funext wire
  dsimp only [Function.comp_apply]
  by_cases h : wire < count
  · let i : Fin ((List.finRange count).map fun i => input i.val).length := ⟨wire, by simpa using h⟩
    have correct := loadData_value control base _ before state i selected (clean _ (by omega))
    simpa [i, List.get_eq_getElem] using correct
  · rw [loadData_outside _ _ _ _ _ _ (Or.inr (by simp; omega)), clean _ (by omega), inputClean _ (by omega)]

end Semantic.Circuit
