import Circuit.SharedAssembly
import Robustness.FlowRobustness

namespace Semantic.Circuit

inductive MetadataKind where
  | graphIdentity
  | rowKeyOrder
  | structuralDomain

def MetadataKind.position : MetadataKind → MultiIntroPositions.Position
  | .graphIdentity => .graphIdentity
  | .rowKeyOrder => .rowKeyOrder
  | .structuralDomain => .structuralDomain



def metadataProgram (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (kind : MetadataKind) : Shared.Program 0 :=
  Shared.expressionProgram (.constant (Flow.Robustness.check d c kind.position))

def metadataCircuit (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (kind : MetadataKind) : Circuit :=
  (metadataProgram d c kind).oracle

theorem metadata_correct (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (kind : MetadataKind)
    (state : Bits) (clean : ∀ wire, state wire = false) :
    eval (metadataCircuit d c kind) state = Function.update state
      (metadataProgram d c kind).target (Flow.Robustness.check d c kind.position) := by
  rw [metadataCircuit, Shared.Program.oracle_correct _ _ (fun wire _ => clean wire),
    metadataProgram, Shared.expressionProgram_value]
  rfl

theorem metadata_wires (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (kind : MetadataKind) :
    (metadataProgram d c kind).wires = 2 := rfl

end Semantic.Circuit
