import Flow.Readings.MultiIntroReadingEquivalence
import Flow.Certificate.Positions
import Flow.Grid.MultiIntroGrid

open Semantic FlowSpec GridCertificateSoundness
namespace MultiIntroMain

theorem main (variant : MultiIntro.Variant) (d : Graph) :
    (MultiIntro.ValidMI variant d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ MultiIntroGrid.acceptedB variant d = true) ∧
    (MultiIntro.ValidMI variant d ↔
      ∃ c, MultiIntro.StrongCertificate variant d c) ∧
    (MultiIntro.ValidMI variant d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ MultiIntroReadings.verdict variant d) ∧
    (MultiIntro.ValidMI variant d →
      ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
        Derives [] root.FORMULA) :=
  ⟨MultiIntroGrid.valid_iff_grid variant d,
    MultiIntro.valid_iff_exists_certificate variant d,
    MultiIntroReadings.valid_iff_readings variant d,
    MultiIntro.valid_roots_derivable variant d⟩

theorem extras (variant : MultiIntro.Variant) (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) :
    (MultiIntro.StrongCertificate variant d c ↔
      ∀ position ∈ MultiIntroPositions.positions d, ¬MultiIntroPositions.FailedAt variant d c position) ∧
    (¬MultiIntro.StrongCertificate variant d c →
      ∃ position ∈ MultiIntroPositions.positions d, MultiIntroPositions.FailedAt variant d c position) ∧
    (∀ slot v, MultiIntroPositions.readBound c (.productionRow slot v) ≤
      1 + (predsOf c.graph v).length) ∧
    ((MultiIntroPositions.positions d).length = d.NODES.length +
      (d.NODES.filter fun v => (get_rule.outgoing v d).isEmpty).length + 3) :=
  ⟨MultiIntroPositions.formation_iff_no_failed_position variant d c,
    MultiIntroPositions.invalid_has_failed_position variant d c,
    MultiIntroPositions.production_read_bound c,
    MultiIntroPositions.position_count d⟩

theorem checked_certificate_view (variant : MultiIntro.Variant) (d : Graph) :
    MultiIntro.ValidMI variant d ↔ ∃ c, MultiIntro.strongCertificateB variant d c = true := by
  rw [(main variant d).2.1]
  exact exists_congr fun c => (MultiIntroPositions.checked_iff variant d c).symm

#print axioms main
#print axioms extras
#print axioms checked_certificate_view
end MultiIntroMain
