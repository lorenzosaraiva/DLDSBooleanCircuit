import Examples.FourVertex.Flow.CachedDischarge

open DLDS

namespace DLDS.Examples.FourVertex.Flow.EndToEnd

open DLDS.Examples.FourVertex.Flow.Acceptance DLDS.Examples.FourVertex.Flow.CachedDischarge

/-- Universal acceptance of the fixed historical closed 4v fixture. -/
theorem closed_4v_self_certifying :
    ∀ P, ¬ AdmissibleMultiPath graph P ∨
      DischargedMulti graph P (goalColumn graph) :=
  closed_4v_self_certifying_of_cached cached_true


end DLDS.Examples.FourVertex.Flow.EndToEnd
