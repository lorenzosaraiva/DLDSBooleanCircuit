import Instances.TwoK2.Formation

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace Instances.TwoK2
open HamiltonianContext
def metadataB : Bool := graphCheckB sourceData && contextCheckB sourceData context witnesses && NonHamiltonicity.rootTargetB proofGraph context 0
theorem metadata_checked : metadataB = true := by native_decide
private theorem metadata_parts : graphCheckB sourceData = true ∧ contextCheckB sourceData context witnesses = true ∧ NonHamiltonicity.rootTargetB proofGraph context 0 = true := by simpa [metadataB, Bool.and_assoc] using metadata_checked
theorem source_checked : graphCheckB sourceData = true := metadata_parts.1
def obligation : CertificateApplication.Obligation where
  schemaVersion := "hermann-certificate-v3-request/1.0"
  contractVersion := "hermann-positional-context/1.0"
  sourceData := sourceData
  context := context
  witnesses := witnesses
  proofGraph := proofGraph
  certificate := certificate
  rootIndex := 0
theorem obligation_checked : CertificateApplication.obligationCheckB obligation = true := by
  simp [CertificateApplication.obligationCheckB, obligation, source_checked, metadata_parts.2.1, metadata_parts.2.2, formation_checked]
theorem no_hamiltonian_cycle : ¬HasHamiltonianCycle (formalGraph sourceData source_checked) := by
  obtain ⟨valid, noCycle⟩ := CertificateApplication.checked_obligation_sound obligation obligation_checked
  exact noCycle
#print axioms no_hamiltonian_cycle
end Instances.TwoK2
