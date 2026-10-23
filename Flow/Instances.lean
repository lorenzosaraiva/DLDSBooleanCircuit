import Flow.Core
import Instances.E3.PipelineEquality
import Instances.E3.Proof
import Instances.K2_K1.PipelineEquality
import Instances.K2_K1.Proof
import Instances.P3.PipelineEquality
import Instances.P3.Proof
import Instances.E4.PipelineEquality
import Instances.E4.Proof
import Instances.K2_2K1.PipelineEquality
import Instances.K2_2K1.Proof
import Instances.TwoK2.PipelineEquality
import Instances.TwoK2.Proof
import Instances.P3_K1.PipelineEquality
import Instances.P3_K1.Proof
import Instances.P4.PipelineEquality
import Instances.P4.Proof
import Instances.Star.PipelineEquality
import Instances.Star.Proof
import Instances.K3_K1.PipelineEquality

namespace Flow.Instances

namespace E3

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.E3.input).graph := by
  rw [Instances.E3.pipeline_equality]
  exact Flow.of_reference _ Instances.E3.flowRef

end E3

namespace K2_K1

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.K2_K1.input).graph := by
  rw [Instances.K2_K1.pipeline_equality]
  exact Flow.of_reference _ Instances.K2_K1.flowRef

end K2_K1

namespace P3

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.P3.input).graph := by
  rw [Instances.P3.pipeline_equality]
  exact Flow.of_reference _ Instances.P3.flowRef

end P3

namespace E4

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.E4.input).graph := by
  rw [Instances.E4.pipeline_equality]
  exact Flow.of_reference _ Instances.E4.flowRef

end E4

namespace K2_2K1

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.K2_2K1.input).graph := by
  rw [Instances.K2_2K1.pipeline_equality]
  exact Flow.of_reference _ Instances.K2_2K1.flowRef

end K2_2K1

namespace TwoK2

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.TwoK2.input).graph := by
  rw [Instances.TwoK2.pipeline_equality]
  exact Flow.of_reference _ Instances.TwoK2.flowRef

end TwoK2

namespace P3_K1

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.P3_K1.input).graph := by
  rw [Instances.P3_K1.pipeline_equality]
  exact Flow.of_reference _ Instances.P3_K1.flowRef

end P3_K1

namespace P4

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.P4.input).graph := by
  rw [Instances.P4.pipeline_equality]
  exact Flow.of_reference _ Instances.P4.flowRef

end P4

namespace Star

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.Star.input).graph := by
  rw [Instances.Star.pipeline_equality]
  exact Flow.of_reference _ Instances.Star.flowRef

end Star

namespace K3_K1

theorem pipeline_flow :
    _root_.Flow
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence Instances.K3_K1.input).graph :=
  Instances.K3_K1.pipeline_valid

end K3_K1

end Flow.Instances
