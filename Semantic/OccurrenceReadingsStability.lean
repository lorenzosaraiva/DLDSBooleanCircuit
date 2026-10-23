import Semantic.SingleIntroductionReference
import Semantic.DependencyGuardStability




open Semantic FlowSpec

namespace OccurrenceReadings



theorem routes_mem_at_standard (d : Graph) (structural : structuralValid d)
    {v : Vertex} (node : v ∈ d.NODES) (fuel : Nat) (route : FlowPair)
    (member : route ∈ SingleIntroductionReference.routes d fuel v) :
    route ∈ SingleIntroductionReference.routes d (stdFuel d) v := by
  exact DependencyGuard.evaluateAt_member_at_standard d
    (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) structural node fuel route member

#print axioms routes_mem_at_standard
end OccurrenceReadings
