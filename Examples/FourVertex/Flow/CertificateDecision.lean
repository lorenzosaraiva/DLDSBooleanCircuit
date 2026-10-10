import Examples.FourVertex.Flow.Certificate

open DLDS

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.FourVertex.Flow.CertificateDecision

theorem certificateReport_eq_expected :
    DLDS.Examples.FourVertex.Flow.Certificate.certificateReport =
    { positions := 307858
      witnesses := 8755
      terminals := 1
      failuresByCheck := []
      failureCount := 0
      expanded := true
      localChecks := true } := by
  native_decide

end DLDS.Examples.FourVertex.Flow.CertificateDecision
