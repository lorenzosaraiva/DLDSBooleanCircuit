import NonHamiltonicity.Soundness
import NonHamiltonicity.HamiltonianContextBridge

open Semantic HamiltonianContext

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



structure Obligation where
  schemaVersion : String
  contractVersion : String
  sourceData : SourceGraphData
  context : List Formula
  witnesses : List SchemaWitness
  proofGraph : _root_.Graph
  certificate : OccurrenceCertificate.OccurrenceCertificate
  rootIndex : Nat



def obligationCheckB (obligation : Obligation) : Bool :=
  (obligation.schemaVersion == "hermann-certificate-v2-request/1.0") &&
  (obligation.contractVersion == "hermann-positional-context/1.0") &&
  graphCheckB obligation.sourceData &&
  contextCheckB obligation.sourceData obligation.context obligation.witnesses &&
  rootTargetB obligation.proofGraph obligation.context obligation.rootIndex &&
  OccurrenceCertificate.strongFormationValidOccurrenceB obligation.proofGraph obligation.certificate

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

theorem checked_obligation_sound (obligation : Obligation) (accepted : obligationCheckB obligation = true) :
    ∃ valid : graphCheckB obligation.sourceData = true,
      ¬HasHamiltonianCycle (formalGraph obligation.sourceData valid) := by
  have parts :
      (obligation.schemaVersion == "hermann-certificate-v2-request/1.0") = true ∧
      (obligation.contractVersion == "hermann-positional-context/1.0") = true ∧
      graphCheckB obligation.sourceData = true ∧
      contextCheckB obligation.sourceData obligation.context obligation.witnesses = true ∧
      rootTargetB obligation.proofGraph obligation.context obligation.rootIndex = true ∧
      OccurrenceCertificate.strongFormationValidOccurrenceB obligation.proofGraph obligation.certificate = true := by
    simpa [obligationCheckB, Bool.and_assoc] using accepted
  exact ⟨parts.2.2.1, checked_certificate_rules_out_cycle obligation.sourceData parts.2.2.1
    obligation.context obligation.witnesses parts.2.2.2.1 obligation.proofGraph obligation.certificate
    obligation.rootIndex parts.2.2.2.2.1 parts.2.2.2.2.2⟩

#print axioms rootTargetB_sound
#print axioms checked_certificate_rules_out_cycle
#print axioms checked_obligation_sound
end NonHamiltonicity
