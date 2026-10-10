import Examples.FourVertex.Flow.Acceptance

open DLDS

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.FourVertex.Flow.CachedDischarge

open DLDS.Examples.FourVertex.Flow.Acceptance

/-- Cached discharge decision for the fixed historical closed 4v fixture. -/
theorem cached_true :
    DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB graph = true := by
  native_decide

end DLDS.Examples.FourVertex.Flow.CachedDischarge
