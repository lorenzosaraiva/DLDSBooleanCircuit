import Semantic.Circuit.IndexFlag
import Semantic.Circuit.PositionInputs

namespace Semantic.Circuit

theorem Shared.Program.support_bound {inputs : Nat} (program : Shared.Program inputs)
    (wire : Nat) (member : wire ∈ support program.oracle) : wire < program.wires := by
  have computation : ∀ wire ∈ support program.code.compile, wire < inputs + program.code.length := by
    intro wire member
    obtain ⟨gate, contains, own⟩ := List.mem_flatMap.mp member
    exact (program.code.bounds gate contains).2 wire own
  simp only [Shared.Program.oracle, support, List.flatMap_append, List.mem_append] at member
  rcases member with (first | middle) | last
  · have := computation wire first
    simp only [Shared.Program.wires]
    omega
  · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, Gate.support,
      Gate.cnot, List.mem_cons, List.not_mem_nil, or_false] at middle
    rcases middle with rfl | rfl
    · simp [Shared.Program.target, Shared.Program.wires]
    · have := program.output.isLt
      simp only [Shared.Program.wires]
      omega
  · have last' : wire ∈ support program.code.compile := (support_inverse _ _).mp last
    have := computation wire last'
    simp only [Shared.Program.wires]
    omega

theorem rename_support (f : Nat → Nat) (injective : Function.Injective f)
    (circuit : Circuit) (wire : Nat) :
    wire ∈ support (rename f injective circuit) ↔ ∃ source ∈ support circuit, f source = wire := by
  have equality : support (rename f injective circuit) = (support circuit).map f := by
    induction circuit with
    | nil => rfl
    | cons gate tail ih =>
      change (gate.rename f injective).support ++ support (rename f injective tail) =
        (gate.support ++ support tail).map f
      rw [List.map_append, ih]
      rfl
  rw [equality, List.mem_map]

theorem controlled_support (control : Nat) (circuit : Circuit) (wire : Nat)
    (member : wire ∈ support (controlled control circuit)) :
    wire = control ∨ wire ∈ support circuit := by
  simp only [controlled, support, List.mem_flatMap, List.mem_map] at member
  obtain ⟨_, ⟨gate, contains, rfl⟩, own⟩ := member
  unfold Gate.controlled at own
  split at own
  · exact Or.inr (List.mem_flatMap.mpr ⟨gate, contains, own⟩)
  · simp only [Gate.mcx, Gate.support, List.mem_cons] at own
    rcases own with equal | equal | member
    · exact Or.inr (List.mem_flatMap.mpr ⟨gate, contains, by simp [Gate.support, equal]⟩)
    · exact Or.inl equal
    · exact Or.inr (List.mem_flatMap.mpr ⟨gate, contains, by simp [Gate.support, member]⟩)

theorem loadData_support (control base : Nat) (data : List Bool) (before : control < base)
    (wire : Nat) (member : wire ∈ support (loadData control base data before)) :
    wire = control ∨ base ≤ wire := by
  simp only [loadData, loadWires, support, List.mem_flatMap, List.mem_map] at member
  obtain ⟨_, ⟨target, _, rfl⟩, own⟩ := member
  simp only [Gate.support, Gate.cnot, List.mem_cons, List.not_mem_nil, or_false] at own
  rcases own with equal | equal
  · have bound := (loadingTargets_mem base data target.val).mp target.property
    obtain ⟨i, hi, _⟩ := bound
    exact Or.inr (by omega)
  · exact Or.inl equal

theorem controlled_control_fixed (control : Nat) (circuit : Circuit) (state : Bits)
    (outside : control ∉ support circuit) :
    eval (controlled control circuit) state control = state control := by
  rw [controlled_eval _ _ _ outside]
  split
  · exact outside_support circuit state control outside
  · rfl

def oracleState (width value : Nat) (target : Bool) : Bits :=
  Function.update (indexState width value) width target

theorem oracleState_inputs (width value : Nat) (target : Bool) (wire : Nat) (inside : wire < width) :
    oracleState width value target wire = indexState width value wire := by
  simp [oracleState, show wire ≠ width by omega]

theorem oracleState_clean (width value base : Nat) (target : Bool) (space : width < base)
    (wire : Nat) (work : base ≤ wire) : oracleState width value target wire = false := by
  simp [oracleState, indexState, show wire ≠ width by omega, show ¬ wire < width by omega]

end Semantic.Circuit
