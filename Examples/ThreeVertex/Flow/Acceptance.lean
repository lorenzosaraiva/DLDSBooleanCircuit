import Examples.ThreeVertex.Flow.Certificate

open DLDS Flow Certificate.Flow Certificate.Flow.Foundation

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.ThreeVertex.Flow.Acceptance

def graph : Graph := DLDS.Examples.ThreeVertex.Flow.Certificate.graph
def certificate : ExpandedCertificate := DLDS.Examples.ThreeVertex.Flow.Certificate.certificate

structure CursorInvariantReport where
  routeHeadCoverage : Bool
  flowTailClosure : Bool
  deriving BEq, DecidableEq, Repr

def cursorInvariantReport : CursorInvariantReport where
  routeHeadCoverage := cursorRouteHeadCoverageB graph
  flowTailClosure := cursorFlowTailClosureB graph

def expectedCursorInvariantReport : CursorInvariantReport where
  routeHeadCoverage := true
  flowTailClosure := true

theorem cursorInvariantReport_eq_expected :
    cursorInvariantReport = expectedCursorInvariantReport := by
  native_decide

theorem cursorRouteHeadCoverage_true :
    cursorRouteHeadCoverageB graph = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Acceptance.CursorInvariantReport.routeHeadCoverage
    cursorInvariantReport_eq_expected
  simpa [cursorInvariantReport, expectedCursorInvariantReport] using h

theorem cursorFlowTailClosure_true :
    cursorFlowTailClosureB graph = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Acceptance.CursorInvariantReport.flowTailClosure
    cursorInvariantReport_eq_expected
  simpa [cursorInvariantReport, expectedCursorInvariantReport] using h

theorem expanded_true : expandedCertB graph certificate = true := by
  exact DLDS.Examples.ThreeVertex.Flow.Certificate.expanded_true

theorem localChecks_true : localChecksB graph certificate = true := by
  exact DLDS.Examples.ThreeVertex.Flow.Certificate.localChecks_true

theorem discharged_true : dischargedMultiB graph = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Certificate.RawReport.discharged
    DLDS.Examples.ThreeVertex.Flow.Certificate.rawReport_eq_expected
  simpa [graph, DLDS.Examples.ThreeVertex.Flow.Certificate.rawReport,
    DLDS.Examples.ThreeVertex.Flow.Certificate.expectedRawReport] using h

theorem slot_true : canonicalSlotOKB graph = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Certificate.RawReport.slotOK
    DLDS.Examples.ThreeVertex.Flow.Certificate.rawReport_eq_expected
  simpa [graph, DLDS.Examples.ThreeVertex.Flow.Certificate.rawReport,
    DLDS.Examples.ThreeVertex.Flow.Certificate.expectedRawReport] using h

theorem noError_true : canonicalNoErrorB graph = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Certificate.RawReport.noError
    DLDS.Examples.ThreeVertex.Flow.Certificate.rawReport_eq_expected
  simpa [graph, DLDS.Examples.ThreeVertex.Flow.Certificate.rawReport,
    DLDS.Examples.ThreeVertex.Flow.Certificate.expectedRawReport] using h

/-- On this fixture, local certificate checks imply the exact invariant
package consumed by the cursor-aware universal bridge. -/
theorem bridge_invariants_of_local (hloc : LocalChecks graph certificate) :
    structuralValid graph ∧ graph.EDGES.Nodup ∧
      OneEdgePerColourPerNode graph ∧
      CursorRouteHeadCoverage graph ∧ LevelFormulaUnique graph ∧
      HypothesesHaveNoIncoming graph ∧ CursorFlowTailClosure graph := by
  obtain ⟨hstruct, hnodup, hone, hlfu, hhyp⟩ :=
    retained_invariants_of_local graph certificate hloc
  exact ⟨hstruct, hnodup, hone,
    cursorRouteHeadCoverageB_sound graph cursorRouteHeadCoverage_true,
    hlfu, hhyp,
    cursorFlowTailClosureB_sound graph cursorFlowTailClosure_true⟩

theorem bridge_invariant_package :
    structuralValid graph ∧ graph.EDGES.Nodup ∧
      OneEdgePerColourPerNode graph ∧
      CursorRouteHeadCoverage graph ∧ LevelFormulaUnique graph ∧
      HypothesesHaveNoIncoming graph ∧ CursorFlowTailClosure graph := by
  have hexp := expandedCertB_sound graph certificate expanded_true
  have hloc := localChecksB_sound graph certificate hexp localChecks_true
  exact bridge_invariants_of_local hloc

theorem closed_3v_self_certifying :
    ∀ P, ¬ AdmissibleMultiPath graph P ∨
      DischargedMulti graph P (goalColumn graph) := by
  exact compressed_universally_accepted_certificateB graph certificate
    expanded_true localChecks_true slot_true discharged_true


end DLDS.Examples.ThreeVertex.Flow.Acceptance
