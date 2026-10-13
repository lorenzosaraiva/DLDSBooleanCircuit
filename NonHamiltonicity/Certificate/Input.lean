import NonHamiltonicity.Certificate.Soundness
import NonHamiltonicity.Context.Encoding

/-! Thin checks of supplied certificate/statement data. Root selectors are
zero-based indices into the exact ordered occurrenceRoots list, never NUMBER.
The unchanged local formation check is the certificate authority. -/
open NonHamiltonicity.Context Evaluator.Occurrence Evaluator.Occurrence.Fuel
open Certificate.Occurrence.Positions

namespace NonHamiltonicity.Certificate.Input

def actualDomainB (proofGraph : _root_.Graph) : Bool :=
  proofGraph.NODES.all fun vertex => (FlowSemantics.predsOf proofGraph vertex).all fun premise =>
    proofGraph.NODES.contains premise && decide (premise.LEVEL = vertex.LEVEL + 1)

theorem actualDomainB_sound (proofGraph : _root_.Graph)
    (checked : actualDomainB proofGraph = true) : ActualPredecessorDomain proofGraph := by
  intro vertex member premise predecessor
  have row := List.all_eq_true.mp checked vertex member
  have entry := List.all_eq_true.mp row premise predecessor
  simpa using entry

def rootTargetB (proofGraph : _root_.Graph) (context : List Formula) (rootIndex : Nat) : Bool :=
  match (occurrenceRoots proofGraph)[rootIndex]? with
  | none => false
  | some root => formulaEqB root.FORMULA (chain context (q canonicalNames))

theorem rootTargetB_sound (proofGraph : _root_.Graph) (context : List Formula) (rootIndex : Nat)
    (checked : rootTargetB proofGraph context rootIndex = true) :
    ∃ root, root ∈ occurrenceRoots proofGraph ∧
      root.FORMULA = chain context (q canonicalNames) := by
  unfold rootTargetB at checked
  cases selected : (occurrenceRoots proofGraph)[rootIndex]? with
  | none => simp [selected] at checked
  | some root =>
      exact ⟨root, List.mem_of_getElem? selected,
        (formulaEqB_iff _ _).mp (by simpa [selected] using checked)⟩

structure Request where
  schemaVersion : String
  contractVersion : String
  sourceData : SourceGraphData
  context : List Formula
  witnesses : List SchemaWitness
  proofGraph : _root_.Graph
  certificate : SuccessorCertificate
  rootIndex : Nat

def requestCheckB (request : Request) : Bool :=
  (request.schemaVersion == "hermann-certificate-request/1.0") &&
  (request.contractVersion == "hermann-positional-context/1.0") &&
  graphCheckB request.sourceData &&
  contextCheckB request.sourceData request.context request.witnesses &&
  actualDomainB request.proofGraph &&
  rootTargetB request.proofGraph request.context request.rootIndex &&
  successorFormationB request.proofGraph request.certificate

theorem checked_certificate_rules_out_cycle
    (sourceData : SourceGraphData) (valid : graphCheckB sourceData = true)
    (context : List Formula) (witnesses : List SchemaWitness)
    (contextChecked : contextCheckB sourceData context witnesses = true)
    (proofGraph : _root_.Graph) (certificate : SuccessorCertificate)
    (domainChecked : actualDomainB proofGraph = true)
    (rootIndex : Nat) (rootChecked : rootTargetB proofGraph context rootIndex = true)
    (formation : successorFormationB proofGraph certificate = true) :
    ¬HasHamiltonianCycle (formalGraph sourceData valid) := by
  obtain ⟨root, member, target⟩ := rootTargetB_sound proofGraph context rootIndex rootChecked
  exact NonHamiltonicity.Certificate.Soundness.certificate_rules_out_hamiltonian_cycle
    (formalGraph sourceData valid) canonicalNames context
    (contextCheckB_sound sourceData valid context witnesses contextChecked)
    proofGraph certificate (actualDomainB_sound proofGraph domainChecked) formation root member target

theorem checked_request_sound (request : Request) (accepted : requestCheckB request = true) :
    ∃ valid : graphCheckB request.sourceData = true,
      ¬HasHamiltonianCycle (formalGraph request.sourceData valid) := by
  have parts :
      (request.schemaVersion == "hermann-certificate-request/1.0") = true ∧
      (request.contractVersion == "hermann-positional-context/1.0") = true ∧
      graphCheckB request.sourceData = true ∧
      contextCheckB request.sourceData request.context request.witnesses = true ∧
      actualDomainB request.proofGraph = true ∧
      rootTargetB request.proofGraph request.context request.rootIndex = true ∧
      successorFormationB request.proofGraph request.certificate = true := by
    simpa [requestCheckB, Bool.and_assoc] using accepted
  exact ⟨parts.2.2.1, checked_certificate_rules_out_cycle request.sourceData parts.2.2.1
    request.context request.witnesses parts.2.2.2.1 request.proofGraph request.certificate
    parts.2.2.2.2.1 request.rootIndex parts.2.2.2.2.2.1 parts.2.2.2.2.2.2⟩

end NonHamiltonicity.Certificate.Input
