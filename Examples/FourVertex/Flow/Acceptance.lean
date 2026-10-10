import Examples.FourVertex.Flow.CertificateDecision
import Examples.FourVertex.Flow.Slots
import Certificate.Flow.Cache

open DLDS Flow Certificate.Flow Certificate.Flow.Foundation

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.FourVertex.Flow.Acceptance

def graph : Graph := DLDS.Examples.FourVertex.Flow.Certificate.graph
def certificate : ExpandedCertificate := DLDS.Examples.FourVertex.Flow.Certificate.certificate

theorem expanded_true : expandedCertB graph certificate = true := by
  have h := congrArg
    DLDS.Certificate.Flow.IntroductionSuffix.CertificateReport.expanded
    DLDS.Examples.FourVertex.Flow.CertificateDecision.certificateReport_eq_expected
  simpa [graph, certificate,
    DLDS.Examples.FourVertex.Flow.Certificate.certificateReport,
    DLDS.Certificate.Flow.IntroductionSuffix.certificateReport] using h

theorem localChecks_true : localChecksB graph certificate = true := by
  have h := congrArg
    DLDS.Certificate.Flow.IntroductionSuffix.CertificateReport.localChecks
    DLDS.Examples.FourVertex.Flow.CertificateDecision.certificateReport_eq_expected
  simpa [graph, certificate,
    DLDS.Examples.FourVertex.Flow.Certificate.certificateReport,
    DLDS.Certificate.Flow.IntroductionSuffix.certificateReport] using h

theorem structural_valid : structuralValid graph := by
  have hexp := expandedCertB_sound graph certificate expanded_true
  have hlocal := localChecksB_sound graph certificate hexp localChecks_true
  exact (retained_invariants_of_local graph certificate hlocal).1

theorem slot_true : canonicalSlotOKB graph = true := by
  exact DLDS.Examples.FourVertex.Flow.Slots.slot_true

theorem discharged_true_of_cached
    (hcached :
      DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB graph = true) :
    dischargedMultiB graph = true := by
  apply DLDS.Certificate.Flow.Cache.dischargedMultiB_of_cached graph
  exact hcached

/-- Composition form of the closed 4v bridge.  Its sole remaining executable
fixture fact is the cached discharge Boolean. -/
theorem closed_4v_self_certifying_of_cached
    (hcached :
      DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB graph = true) :
    ∀ P, ¬ AdmissibleMultiPath graph P ∨
      DischargedMulti graph P (goalColumn graph) := by
  exact compressed_universally_accepted_certificateB graph certificate
    expanded_true localChecks_true slot_true
    (discharged_true_of_cached hcached)


end DLDS.Examples.FourVertex.Flow.Acceptance
