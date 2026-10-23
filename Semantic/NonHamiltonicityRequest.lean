import Semantic.NonHamiltonicity
import Semantic.HermannContextBridge

open Semantic HermannContext

namespace NonHamiltonicity


def roots (proofGraph : _root_.Graph) : List Vertex :=
  proofGraph.NODES.filter fun root => (get_rule.outgoing root proofGraph).isEmpty

def rootTargetB (proofGraph : _root_.Graph) (context : List Formula) (rootIndex : Nat) : Bool :=
  match (roots proofGraph)[rootIndex]? with
  | none => false
  | some root => formulaEqB root.FORMULA (chain context (q canonicalNames))

theorem rootTargetB_sound (proofGraph : _root_.Graph) (context : List Formula) (rootIndex : Nat)
    (checked : rootTargetB proofGraph context rootIndex = true) :
    ∃ root, root ∈ proofGraph.NODES ∧ (get_rule.outgoing root proofGraph).isEmpty = true ∧
      root.FORMULA = chain context (q canonicalNames) := by
  unfold rootTargetB at checked
  cases selected : (roots proofGraph)[rootIndex]? with
  | none => simp [selected] at checked
  | some root =>
      have member := List.mem_of_getElem? selected
      have rootParts : root ∈ proofGraph.NODES ∧
          (get_rule.outgoing root proofGraph).isEmpty = true := List.mem_filter.mp member
      exact ⟨root, rootParts.1, rootParts.2,
        (formulaEqB_iff _ _).mp (by simpa [selected] using checked)⟩



structure Request where
  schemaVersion : String
  contractVersion : String
  sourceData : SourceGraphData
  context : List Formula
  witnesses : List SchemaWitness
  proofGraph : _root_.Graph
  certificate : OccurrenceCertificate.OccurrenceCertificate
  rootIndex : Nat



def requestCheckB (request : Request) : Bool :=
  (request.schemaVersion == "hermann-certificate-v2-request/1.0") &&
  (request.contractVersion == "hermann-positional-context/1.0") &&
  graphCheckB request.sourceData &&
  contextCheckB request.sourceData request.context request.witnesses &&
  rootTargetB request.proofGraph request.context request.rootIndex &&
  OccurrenceCertificate.strongFormationValidOccurrenceB request.proofGraph request.certificate

theorem checked_certificate_rules_out_cycle
    (sourceData : SourceGraphData) (valid : graphCheckB sourceData = true)
    (context : List Formula) (witnesses : List SchemaWitness)
    (contextChecked : contextCheckB sourceData context witnesses = true)
    (proofGraph : _root_.Graph) (certificate : OccurrenceCertificate.OccurrenceCertificate)
    (rootIndex : Nat) (rootChecked : rootTargetB proofGraph context rootIndex = true)
    (formation : OccurrenceCertificate.strongFormationValidOccurrenceB proofGraph certificate = true) :
    ¬HasHamiltonianCycle (formalGraph sourceData valid) := by
  obtain ⟨root, member, terminal, target⟩ := rootTargetB_sound proofGraph context rootIndex rootChecked
  exact checked_certificate_rules_out_hamiltonian_cycle
    (formalGraph sourceData valid) canonicalNames context
    (contextCheckB_sound sourceData valid context witnesses contextChecked)
    proofGraph certificate formation root member terminal target

theorem checked_request_sound (request : Request) (accepted : requestCheckB request = true) :
    ∃ valid : graphCheckB request.sourceData = true,
      ¬HasHamiltonianCycle (formalGraph request.sourceData valid) := by
  have parts :
      (request.schemaVersion == "hermann-certificate-v2-request/1.0") = true ∧
      (request.contractVersion == "hermann-positional-context/1.0") = true ∧
      graphCheckB request.sourceData = true ∧
      contextCheckB request.sourceData request.context request.witnesses = true ∧
      rootTargetB request.proofGraph request.context request.rootIndex = true ∧
      OccurrenceCertificate.strongFormationValidOccurrenceB request.proofGraph request.certificate = true := by
    simpa [requestCheckB, Bool.and_assoc] using accepted
  exact ⟨parts.2.2.1, checked_certificate_rules_out_cycle request.sourceData parts.2.2.1
    request.context request.witnesses parts.2.2.2.1 request.proofGraph request.certificate
    request.rootIndex parts.2.2.2.2.1 parts.2.2.2.2.2⟩

#print axioms rootTargetB_sound
#print axioms checked_certificate_rules_out_cycle
#print axioms checked_request_sound
end NonHamiltonicity
