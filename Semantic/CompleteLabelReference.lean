import Semantic.SpineCertificateLabels
import Semantic.OccurrenceCertificateCompleteness




open Semantic FlowSpec SelfCertifying SelfCertifyingProbe OccurrenceCertificate

namespace CompleteLabelReference

abbrev RefValid := Spine.UValidReading



def LabelSoundCheck (c : OccurrenceCertificate) (labels : Deduction → LambdaFlow.Label) : Prop :=
  ∀ edge ∈ c.graph.EDGES, ∀ witness ∈ edgeWitnessesFor c edge,
    LambdaFlow.labelAgreesB (labels edge) witness.dependency = true


def labelSoundCheckB (c : OccurrenceCertificate) (labels : Deduction → LambdaFlow.Label) : Bool :=
  c.graph.EDGES.all fun edge => (edgeWitnessesFor c edge).all fun witness =>
    LambdaFlow.labelAgreesB (labels edge) witness.dependency

theorem labelSoundCheckB_iff (c : OccurrenceCertificate) (labels : Deduction → LambdaFlow.Label) :
    labelSoundCheckB c labels = true ↔ LabelSoundCheck c labels := by
  simp [labelSoundCheckB, LabelSoundCheck, List.all_eq_true]



theorem localLabelCheck_implies_labelSoundCheck (d : Graph) (c : OccurrenceCertificate)
    (valid : StrongFormationValidOccurrence d c) (labels : Deduction → LambdaFlow.Label)
    (checked : LocalLabelCheck c labels) : LabelSoundCheck c labels := by
  intro edge hedge witness member
  have graphEq := valid.2.1.1
  have structural : structuralValid d := valid.2.1.2.2.2.2.2.1
  have edgeMember : edge ∈ d.EDGES := by simpa [graphEq] using hedge
  have source : edge.START ∈ c.graph.NODES := by
    simpa [graphEq] using (structural.2.1 edge edgeMember).1
  have outgoing := mem_outgoing_of_mem_edges_start_eq edge.START c.graph hedge rfl
  have covered := (valid.2.1.2.2.1 edge.START source edge outgoing).2
  have projected := (covered (witness.dependency, witness.residual)).mp
    (List.mem_map.mpr ⟨witness, member, rfl⟩)
  obtain ⟨stored, head⟩ := List.mem_filter.mp projected
  exact LocalLabelCheck.agrees c labels checked edge hedge
    (witness.dependency, witness.residual) stored (by simpa using head)

theorem labelSoundCheck_implies_labelSound (lg : LambdaFlow.LGraph) (c : OccurrenceCertificate)
    (valid : StrongFormationValidOccurrence lg.graph c) (labels : LabelSoundCheck c lg.label) :
    Spine.LabelSound lg := by
  intro edge hedge route member tail consumed
  have graphEq := valid.2.1.1
  have structural : structuralValid lg.graph := valid.2.1.2.2.2.2.2.1
  have source := (structural.2.1 edge hedge).1
  have outgoing := mem_outgoing_of_mem_edges_start_eq edge.START lg.graph hedge rfl
  have continuable := routeContinuableAtB_of_consume_outgoing lg.graph edge.START
    edge route.2 tail outgoing consumed
  have stored := (storedRoutes_setEq_guardFree lg.graph c valid.1 valid.2.1
    edge.START source route).mpr (List.mem_filter.mpr ⟨member, continuable⟩)
  have covered := (valid.2.1.2.2.1 edge.START (by simpa [graphEq] using source)
    edge (by simpa [graphEq] using outgoing)).2
  have projected := (covered route).mpr
    (List.mem_filter.mpr ⟨stored, by simpa using (consume_some_colour consumed).symm⟩)
  obtain ⟨witness, witnessMember, pairEq⟩ := List.mem_map.mp projected
  have agrees := labels edge (by simpa [graphEq] using hedge) witness witnessMember
  have depEq := congrArg Prod.fst pairEq
  change witness.dependency = route.1 at depEq
  rw [← depEq]
  exact agrees


theorem certificate_refValid (lg : LambdaFlow.LGraph) (c : OccurrenceCertificate)
    (valid : StrongFormationValidOccurrence lg.graph c) (labels : LabelSoundCheck c lg.label) :
    RefValid lg := by
  have structural : structuralValid lg.graph := valid.2.1.2.2.2.2.2.1
  exact ⟨structural, structural.1, labelSoundCheck_implies_labelSound lg c valid labels,
    strongFormationValidOccurrence_guardFreeRootsB lg.graph c valid⟩



theorem generated_labelSoundCheck (lg : LambdaFlow.LGraph) (labels : Spine.LabelSound lg) :
    LabelSoundCheck (generateOccurrenceCertificate lg.graph) lg.label := by
  intro edge _ witness member
  obtain ⟨edgeMember, _, routeMember, head⟩ :=
    (mem_generated_edgeWitnessesFor_iff lg.graph edge witness).mp member
  exact labels edge edgeMember (witness.dependency, witness.residual) routeMember
    witness.residual.tail (consume_headColour head.symm)



theorem refValid_generated_certificate (lg : LambdaFlow.LGraph)
    (valid : RefValid lg) (suitable : SuitableGraphOccurrence lg.graph) :
    StrongFormationValidOccurrence lg.graph (generateOccurrenceCertificate lg.graph) ∧
      LabelSoundCheck (generateOccurrenceCertificate lg.graph) lg.label :=
  ⟨generateOccurrenceCertificate_strongFormation lg.graph suitable valid.2.2.2,
    generated_labelSoundCheck lg valid.2.2.1⟩

theorem refValid_exists_certificate (lg : LambdaFlow.LGraph)
    (valid : RefValid lg) (suitable : SuitableGraphOccurrence lg.graph) :
    ∃ c, StrongFormationValidOccurrence lg.graph c ∧ LabelSoundCheck c lg.label :=
  ⟨generateOccurrenceCertificate lg.graph, refValid_generated_certificate lg valid suitable⟩

theorem uValid_refValid (lg : LambdaFlow.LGraph) (valid : Spine.UValid lg) :
    RefValid lg := Spine.UValid.toReading lg valid

#print axioms certificate_refValid
#print axioms refValid_generated_certificate
end CompleteLabelReference
