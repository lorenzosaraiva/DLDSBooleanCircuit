import Semantic.SpineReference
import Semantic.OccurrenceCertificateLabels
import Semantic.OccurrenceCertificateSoundness

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe OccurrenceCertificate

namespace Spine




theorem strong_localLabel_implies_labelSound (lg : LambdaFlow.LGraph)
    (c : OccurrenceCertificate) (valid : StrongFormationValidOccurrence lg.graph c)
    (labels : LocalLabelCheck c lg.label) : LabelSound lg := by
  intro edge hedge route member tail consumed
  have graphEq := valid.2.1.1
  have structural := valid.2.1.2.2.2.2.2.1
  have sourceMember := (structural.2.1 edge hedge).1
  have outgoing := mem_outgoing_of_mem_edges_start_eq edge.START lg.graph hedge rfl
  have continuable := routeContinuableAtB_of_consume_outgoing lg.graph edge.START
    edge route.2 tail outgoing consumed
  have witnessable : route ∈ WitnessableRoutes lg.graph edge.START
      (LambdaFlow.guardFreeFlowAt lg.graph (stdFuel lg.graph) edge.START) :=
    List.mem_filter.mpr ⟨member, continuable⟩
  have stored := (storedRoutes_setEq_guardFree lg.graph c valid.1 valid.2.1
    edge.START sourceMember route).mpr witnessable
  exact LocalLabelCheck.agrees c lg.label labels edge (by simpa [graphEq] using hedge)
    route stored (consume_some_colour consumed).symm



theorem strong_localLabel_implies_UValidReading (lg : LambdaFlow.LGraph)
    (c : OccurrenceCertificate) (valid : StrongFormationValidOccurrence lg.graph c)
    (labels : LocalLabelCheck c lg.label) : UValidReading lg := by
  have structural : structuralValid lg.graph := valid.2.1.2.2.2.2.2.1
  exact ⟨structural, structural.1,
    strong_localLabel_implies_labelSound lg c valid labels,
    strongFormationValidOccurrence_guardFreeRootsB lg.graph c valid⟩

end Spine
