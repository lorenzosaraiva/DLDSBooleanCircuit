import Infrastructure.OccurrenceReadingsVerdict
import Infrastructure.RouteGrid
import Infrastructure.RouteSoundness
import Infrastructure.RouteCertificateCompleteness























open Semantic FlowSpec GridCertificateSoundness

namespace ReferenceMain



theorem main (d : Graph) :
    (SingleIntroductionReference.ReferenceVerdict d ↔
      structuralValid d ∧ LeveledColored d ∧ RouteGrid.acceptedB d = true) ∧
    (SingleIntroductionReference.ReferenceVerdict d ↔
      ∃ c, RouteCertificate.StrongFormationValidComplete d c) ∧
    (SingleIntroductionReference.ReferenceVerdict d ↔
      structuralValid d ∧ LeveledColored d ∧ OccurrenceReadings.occurrenceVerdictComplete d) ∧
    (SingleIntroductionReference.ReferenceVerdict d →
      ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
        Derives [] root.FORMULA) :=
  ⟨RouteGrid.ReferenceVerdict_iff_grid d,
    RouteCertificate.ReferenceVerdict_iff_exists_complete_profile_certificate d,
    OccurrenceReadings.ReferenceVerdict_iff_domain_occurrenceVerdictComplete d,
    SingleIntroductionReference.ReferenceVerdict.roots_derivable d⟩

#print axioms main
end ReferenceMain
