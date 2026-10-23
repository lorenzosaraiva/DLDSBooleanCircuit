import Instances.K2_2K1.Certificate
import NonHamiltonicity.CertificateAcceptance
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace Instances.K2_2K1
theorem formation_checked : RouteCertificate.strongFormationValidCompleteB proofGraph certificate = true := by native_decide
theorem flowRef : SingleIntroductionReference.ReferenceVerdict proofGraph := RouteCertificate.strongFormationValidCompleteB_ReferenceVerdict proofGraph certificate formation_checked
end Instances.K2_2K1
