import Circuit.SharedAssembly
import Circuit.Root

namespace Semantic.Circuit

def Shared.allProgram {n : Nat} (wires : List (Fin n)) : Shared.Program n :=
  Shared.expressionProgram (Expr.all (wires.map Expr.bit))

def Shared.anyProgram {n : Nat} (wires : List (Fin n)) : Shared.Program n :=
  Shared.expressionProgram (Expr.any (wires.map Expr.bit))

theorem Shared.allProgram_correct {n : Nat} (wires : List (Fin n)) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (Shared.allProgram wires).oracle state =
      Function.update state (Shared.allProgram wires).target (wires.all fun i => state i) := by
  rw [allProgram, expression_oracle _ state clean, Expr.value_all]
  simp [Function.comp_def, Expr.value]

theorem Shared.anyProgram_correct {n : Nat} (wires : List (Fin n)) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (Shared.anyProgram wires).oracle state =
      Function.update state (Shared.anyProgram wires).target (wires.any fun i => state i) := by
  rw [anyProgram, expression_oracle _ state clean, Expr.value_any]
  simp [Function.comp_def, Expr.value]

def rootProgramShared (R F I : Nat) : Shared.Program (R * (1 + (F + I))) :=
  Shared.expressionProgram (rootExpression R F I)

def rootCircuitShared (R F I : Nat) : Circuit := (rootProgramShared R F I).oracle

def rootTargetShared (R F I : Nat) : Nat := (rootProgramShared R F I).target

def rootResourcesShared (R F I : Nat) : Nat × Nat :=
  ((rootProgramShared R F I).wires, (rootProgramShared R F I).oracle.length)

theorem root_correct_shared (d : Graph) (certificate : RouteCertificate.CompleteProfileCertificate)
    (vertex : Vertex) (positionSlot : Nat) {F : Nat} (R I : Nat)
    (basis : Fin F → Formula) (words : List FlowSpec.ColourPath)
    (fits : ProfileFits R I basis words (RouteCertificate.stored certificate vertex)) :
    eval (rootCircuitShared R F I)
      (inputState (encodeProfile R I basis words (RouteCertificate.stored certificate vertex) fits.1)) =
      Function.update
        (inputState (encodeProfile R I basis words (RouteCertificate.stored certificate vertex) fits.1))
        (rootTargetShared R F I) (_root_.Flow.Robustness.check d certificate (.rootDischarge positionSlot vertex)) := by
  rw [rootCircuitShared, rootProgramShared, Shared.expression_oracle _ _ (inputState_clean _),
    rootExpression_value, encoded_check R I basis words _ fits]
  rfl

theorem root_ancillas_clean_shared {R F I : Nat} (code : List.Vector (Slot F I) R) (wire : Nat)
    (workspace : R * (1 + (F + I)) ≤ wire) (notTarget : wire ≠ rootTargetShared R F I) :
    eval (rootCircuitShared R F I) (inputState code) wire = false := by
  rw [rootCircuitShared, Shared.Program.oracle_correct _ _ (inputState_clean _)]
  change wire ≠ (rootProgramShared R F I).target at notTarget
  rw [Function.update_of_ne notTarget]
  exact inputState_clean code wire workspace

end Semantic.Circuit
