import Instances.K3_K1.CertificateData
import Flow.Certificate.MultiIntro
import NonHamiltonicity.OccurrenceAcceptance

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace Instances.K3_K1
open Instances.K3_K1 HamiltonianContext

theorem formation_checked : MultiIntro.strongCertificateB .general proofGraph certificate = true := by native_decide
theorem valid : MultiIntro.ValidMI .general proofGraph :=
  MultiIntro.formation_valid .general proofGraph certificate
    (MultiIntro.strongCertificateB_sound .general proofGraph certificate formation_checked)
def metadataB : Bool := graphCheckB sourceData && contextCheckB sourceData context witnesses && NonHamiltonicity.rootTargetB proofGraph context 0
theorem metadata_checked : metadataB = true := by native_decide
private theorem metadata_parts : graphCheckB sourceData = true ∧ contextCheckB sourceData context witnesses = true ∧ NonHamiltonicity.rootTargetB proofGraph context 0 = true := by
  simpa [metadataB, Bool.and_assoc] using metadata_checked
theorem source_checked : graphCheckB sourceData = true := metadata_parts.1
theorem no_hamiltonian_cycle : ¬HasHamiltonianCycle (formalGraph sourceData source_checked) := by
  obtain ⟨root, listed, terminal, target⟩ := NonHamiltonicity.rootTargetB_sound proofGraph context 0 metadata_parts.2.2
  exact MultiIntro.checked_certificate_rules_out_hamiltonian_cycle .general
    (formalGraph sourceData source_checked) canonicalNames context
    (contextCheckB_sound sourceData source_checked context witnesses metadata_parts.2.1)
    proofGraph certificate formation_checked root listed terminal target
end Instances.K3_K1
