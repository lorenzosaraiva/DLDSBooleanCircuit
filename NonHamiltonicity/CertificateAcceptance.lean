import NonHamiltonicity.CertificateApplication
import NonHamiltonicity.OccurrenceAcceptance

open Semantic HamiltonianContext

namespace CertificateApplication



structure Obligation where
  schemaVersion : String
  contractVersion : String
  sourceData : SourceGraphData
  context : List Formula
  witnesses : List SchemaWitness
  proofGraph : _root_.Graph
  certificate : RouteCertificate.CompleteProfileCertificate
  rootIndex : Nat

def obligationCheckB (obligation : Obligation) : Bool :=
  (obligation.schemaVersion == "hermann-certificate-v3-request/1.0") &&
  (obligation.contractVersion == "hermann-positional-context/1.0") &&
  graphCheckB obligation.sourceData &&
  contextCheckB obligation.sourceData obligation.context obligation.witnesses &&
  NonHamiltonicity.rootTargetB obligation.proofGraph obligation.context obligation.rootIndex &&
  RouteCertificate.strongFormationValidCompleteB obligation.proofGraph obligation.certificate

theorem checked_certificate_rules_out_cycle
    (sourceData : SourceGraphData) (valid : graphCheckB sourceData = true)
    (context : List Formula) (witnesses : List SchemaWitness)
    (contextChecked : contextCheckB sourceData context witnesses = true)
    (proofGraph : _root_.Graph) (certificate : RouteCertificate.CompleteProfileCertificate)
    (rootIndex : Nat)
    (rootChecked : NonHamiltonicity.rootTargetB proofGraph context rootIndex = true)
    (formation : RouteCertificate.strongFormationValidCompleteB proofGraph certificate = true) :
    ¬HasHamiltonianCycle (formalGraph sourceData valid) := by
  obtain ⟨root, member, terminal, target⟩ :=
    NonHamiltonicity.rootTargetB_sound proofGraph context rootIndex rootChecked
  exact checked_certificate_rules_out_hamiltonian_cycle
    (formalGraph sourceData valid) canonicalNames context
    (contextCheckB_sound sourceData valid context witnesses contextChecked)
    proofGraph certificate formation root member terminal target

theorem checked_obligation_sound (obligation : Obligation) (accepted : obligationCheckB obligation = true) :
    ∃ valid : graphCheckB obligation.sourceData = true,
      ¬HasHamiltonianCycle (formalGraph obligation.sourceData valid) := by
  have parts :
      (obligation.schemaVersion == "hermann-certificate-v3-request/1.0") = true ∧
      (obligation.contractVersion == "hermann-positional-context/1.0") = true ∧
      graphCheckB obligation.sourceData = true ∧
      contextCheckB obligation.sourceData obligation.context obligation.witnesses = true ∧
      NonHamiltonicity.rootTargetB obligation.proofGraph obligation.context obligation.rootIndex = true ∧
      RouteCertificate.strongFormationValidCompleteB obligation.proofGraph obligation.certificate = true := by
    simpa [obligationCheckB, Bool.and_assoc] using accepted
  exact ⟨parts.2.2.1, checked_certificate_rules_out_cycle obligation.sourceData parts.2.2.1
    obligation.context obligation.witnesses parts.2.2.2.1 obligation.proofGraph obligation.certificate
    obligation.rootIndex parts.2.2.2.2.1 parts.2.2.2.2.2⟩

#print axioms checked_certificate_rules_out_cycle
#print axioms checked_obligation_sound
end CertificateApplication
