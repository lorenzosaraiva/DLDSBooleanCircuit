import Instances.K3_K1.Data
import Compression.HCJustifiedContinuationPersistence
import Instances.K3_K1.Proof

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

namespace Instances.K3_K1


theorem pipeline_equality_checked :
    decide ((HCJustifiedContinuationPersistence.refineWithJustifiedPersistence input).graph =
      proofGraph) = true := by native_decide

theorem pipeline_equality :
    (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence input).graph =
      proofGraph :=
  of_decide_eq_true pipeline_equality_checked


theorem pipeline_valid :
    MultiIntro.ValidMI .general
      (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence input).graph := by
  rw [pipeline_equality]
  exact Instances.K3_K1.valid

end Instances.K3_K1
