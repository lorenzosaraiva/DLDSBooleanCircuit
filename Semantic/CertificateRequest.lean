import Semantic.CertificateApplication
import Semantic.NonHamiltonicityRequest

open Semantic HermannContext

namespace CertificateApplication



structure Request where
  schemaVersion : String
  contractVersion : String
  sourceData : SourceGraphData
  context : List Formula
  witnesses : List SchemaWitness
  proofGraph : _root_.Graph
  certificate : RouteCertificate.CompleteProfileCertificate
  rootIndex : Nat

def requestCheckB (request : Request) : Bool :=
  (request.schemaVersion == "hermann-certificate-v3-request/1.0") &&
  (request.contractVersion == "hermann-positional-context/1.0") &&
  graphCheckB request.sourceData &&
  contextCheckB request.sourceData request.context request.witnesses &&
  NonHamiltonicity.rootTargetB request.proofGraph request.context request.rootIndex &&
  RouteCertificate.strongFormationValidCompleteB request.proofGraph request.certificate

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

theorem checked_request_sound (request : Request) (accepted : requestCheckB request = true) :
    ∃ valid : graphCheckB request.sourceData = true,
      ¬HasHamiltonianCycle (formalGraph request.sourceData valid) := by
  have parts :
      (request.schemaVersion == "hermann-certificate-v3-request/1.0") = true ∧
      (request.contractVersion == "hermann-positional-context/1.0") = true ∧
      graphCheckB request.sourceData = true ∧
      contextCheckB request.sourceData request.context request.witnesses = true ∧
      NonHamiltonicity.rootTargetB request.proofGraph request.context request.rootIndex = true ∧
      RouteCertificate.strongFormationValidCompleteB request.proofGraph request.certificate = true := by
    simpa [requestCheckB, Bool.and_assoc] using accepted
  exact ⟨parts.2.2.1, checked_certificate_rules_out_cycle request.sourceData parts.2.2.1
    request.context request.witnesses parts.2.2.2.1 request.proofGraph request.certificate
    request.rootIndex parts.2.2.2.2.1 parts.2.2.2.2.2⟩

#print axioms checked_certificate_rules_out_cycle
#print axioms checked_request_sound
end CertificateApplication
