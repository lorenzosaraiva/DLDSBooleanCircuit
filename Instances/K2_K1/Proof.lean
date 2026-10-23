import Instances.K2_K1.Formation

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace Instances.K2_K1
open HermannContext
def metadataB : Bool := graphCheckB sourceData && contextCheckB sourceData context witnesses && NonHamiltonicity.rootTargetB proofGraph context 0
theorem metadata_checked : metadataB = true := by native_decide
private theorem metadata_parts : graphCheckB sourceData = true ∧ contextCheckB sourceData context witnesses = true ∧ NonHamiltonicity.rootTargetB proofGraph context 0 = true := by simpa [metadataB, Bool.and_assoc] using metadata_checked
theorem source_checked : graphCheckB sourceData = true := metadata_parts.1
def request : CertificateApplication.Request where
  schemaVersion := "hermann-certificate-v3-request/1.0"
  contractVersion := "hermann-positional-context/1.0"
  sourceData := sourceData
  context := context
  witnesses := witnesses
  proofGraph := proofGraph
  certificate := certificate
  rootIndex := 0
theorem request_checked : CertificateApplication.requestCheckB request = true := by
  simp [CertificateApplication.requestCheckB, request, source_checked, metadata_parts.2.1, metadata_parts.2.2, formation_checked]
theorem no_hamiltonian_cycle : ¬HasHamiltonianCycle (formalGraph sourceData source_checked) := by
  obtain ⟨valid, noCycle⟩ := CertificateApplication.checked_request_sound request request_checked
  exact noCycle
#print axioms no_hamiltonian_cycle
end Instances.K2_K1
