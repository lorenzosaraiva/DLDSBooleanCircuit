import Examples.FourVertex.Flow.Certificate

open DLDS

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.FourVertex.Flow.Slots

theorem slot_true :
    canonicalSlotOKB DLDS.Examples.FourVertex.Flow.Certificate.graph = true := by
  native_decide

end DLDS.Examples.FourVertex.Flow.Slots
