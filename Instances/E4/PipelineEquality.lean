import Instances.E4.Data
import Compression.HCJustifiedContinuationPersistence
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace Instances.E4
theorem pipeline_equality_checked : decide ((HCJustifiedContinuationPersistence.refineWithJustifiedPersistence input).graph = proofGraph) = true := by native_decide
theorem pipeline_equality : (HCJustifiedContinuationPersistence.refineWithJustifiedPersistence input).graph = proofGraph := of_decide_eq_true pipeline_equality_checked
#print axioms pipeline_equality
end Instances.E4
