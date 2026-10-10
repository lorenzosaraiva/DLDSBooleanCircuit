import Examples.ThreeVertex.Flow.Closed
import Certificate.Flow.Soundness

open DLDS Flow Certificate.Flow.Foundation Certificate.Flow

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.ThreeVertex.Flow.Certificate

def graph : Graph := DLDS.Examples.ThreeVertex.Flow.Closed.graph

def openCertificate : ExpandedCertificate :=
  generateCertificate DLDS.Examples.ThreeVertex.Flow.Open.graph

def openRootRoutes : List FlowPair :=
  openCertificate.terminalOutputs.filterMap fun output =>
    if output.node == DLDS.Examples.ThreeVertex.Flow.Open.v127 then
      some (output.dependency, output.residual)
    else none

def shiftedOpenWitnesses : List EdgeRouteWitness :=
  openCertificate.edgeRoutes.map fun witness =>
    { witness with
      edge := shiftDeduction DLDS.Examples.ThreeVertex.Flow.Closed.suffixLength witness.edge }

def advanceIntro (edge : Deduction) (routes : List FlowPair) : List FlowPair :=
  match edge.END.FORMULA with
  | Formula.implication alpha _ =>
      (routes.filterMap fun route =>
        (consume edge route.2).map fun tail =>
          (depRemove route.1 alpha, tail)).eraseDups
  | _ => []

def suffixCertificateData : List Deduction → List FlowPair →
    List EdgeRouteWitness × List FlowPair
  | [], routes => ([], routes)
  | current :: restEdges, routes =>
      let witnesses := routes.filter
        (fun route => headColour route.2 == current.COLOUR) |>.map fun route =>
          { edge := current, dependency := route.1, residual := route.2 }
      let rest := suffixCertificateData restEdges (advanceIntro current routes)
      (witnesses ++ rest.1, rest.2)

def suffixData : List EdgeRouteWitness × List FlowPair :=
  suffixCertificateData DLDS.Examples.ThreeVertex.Flow.Closed.suffixEdges openRootRoutes

/-- Incremental generation is extensionally the ordinary certificate
generation: the shifted open part reuses its generated witnesses, and each
new one-premise introduction advances those terminal routes once.  This avoids
recomputing the full 3v Flow recursion separately for all 31 suffix nodes. -/
def certificate : ExpandedCertificate :=
  let openCert := generateCertificate DLDS.Examples.ThreeVertex.Flow.Open.graph
  let rootRoutes := openCert.terminalOutputs.filterMap fun output =>
    if output.node == DLDS.Examples.ThreeVertex.Flow.Open.v127 then
      some (output.dependency, output.residual)
    else none
  let shifted := openCert.edgeRoutes.map fun witness =>
    { witness with
      edge := shiftDeduction DLDS.Examples.ThreeVertex.Flow.Closed.suffixLength witness.edge }
  let suffix := suffixCertificateData DLDS.Examples.ThreeVertex.Flow.Closed.suffixEdges rootRoutes
  { graph
    edgeRoutes := shifted ++ suffix.1
    terminalOutputs := suffix.2.map fun route =>
      { node := DLDS.Examples.ThreeVertex.Flow.Closed.suffixVertex DLDS.Examples.ThreeVertex.Flow.Closed.suffixLength
        dependency := route.1
        residual := route.2 } }

def checkHistogram : List (String × Nat) :=
  let names := (failuresMemo certificate).map (·.check)
  names.eraseDups.map fun name => (name, (names.filter (· == name)).length)

def compressionKeys : List (Nat × Formula) :=
  graph.NODES.map fun vertex => (vertex.LEVEL, vertex.FORMULA)

structure ConstructionReport where
  suffixLength : Nat
  openNodes : Nat
  openEdges : Nat
  closedNodes : Nat
  closedEdges : Nat
  paths : Nat
  compressionKeyNodup : Bool
  deriving BEq, DecidableEq, Repr

structure CertificateReport where
  failuresByCheck : List (String × Nat)
  failureCount : Nat
  expanded : Bool
  localChecks : Bool
  deriving BEq, DecidableEq, Repr

structure RawReport where
  routeCount : Nat
  slotOK : Bool
  discharged : Bool
  noError : Bool
  deriving BEq, DecidableEq, Repr

def constructionReport : ConstructionReport where
  suffixLength := DLDS.Examples.ThreeVertex.Flow.Closed.suffixLength
  openNodes := DLDS.Examples.ThreeVertex.Flow.Open.graph.NODES.length
  openEdges := DLDS.Examples.ThreeVertex.Flow.Open.graph.EDGES.length
  closedNodes := graph.NODES.length
  closedEdges := graph.EDGES.length
  paths := graph.PATHS.length
  compressionKeyNodup := decide compressionKeys.Nodup

def certificateReport : CertificateReport :=
  let cert := certificate
  let cache := buildCertificateCache cert
  let found := failuresCached cert cache
  { failuresByCheck :=
      let names := found.map (·.check)
      names.eraseDups.map fun name =>
        (name, (names.filter (· == name)).length)
    failureCount := found.length
    expanded := expandedCertCachedB graph cert cache
    localChecks := localChecksCachedB graph cert cache }

def rawReport : RawReport where
  routeCount := DLDS.routeCount graph
  slotOK := canonicalSlotOKB graph
  discharged := dischargedMultiB graph
  noError := canonicalNoErrorB graph

def expectedConstructionReport : ConstructionReport where
  suffixLength := 31
  openNodes := 97
  openEdges := 113
  closedNodes := 128
  closedEdges := 144
  paths := 41
  compressionKeyNodup := true

def expectedCertificateReport : CertificateReport where
  failuresByCheck := []
  failureCount := 0
  expanded := true
  localChecks := true

def expectedRawReport : RawReport where
  routeCount := 107
  slotOK := true
  discharged := true
  noError := true

theorem constructionReport_eq_expected :
    constructionReport = expectedConstructionReport := by
  native_decide

theorem certificateReport_eq_expected :
    certificateReport = expectedCertificateReport := by
  native_decide

theorem rawReport_eq_expected : rawReport = expectedRawReport := by
  native_decide

theorem expanded_true : expandedCertB graph certificate = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Certificate.CertificateReport.expanded certificateReport_eq_expected
  simpa [certificateReport, expectedCertificateReport] using h

theorem localChecks_true : localChecksB graph certificate = true := by
  have h := congrArg DLDS.Examples.ThreeVertex.Flow.Certificate.CertificateReport.localChecks certificateReport_eq_expected
  simpa [certificateReport, expectedCertificateReport] using h

theorem invariant_package :
    structuralValid graph ∧ graph.EDGES.Nodup ∧
      OneEdgePerColourPerNode graph ∧
      WitnessableRouteHeadCoverage graph ∧ LevelFormulaUnique graph ∧
      HypothesesHaveNoIncoming graph := by
  have hexp := expandedCertB_sound graph certificate expanded_true
  have hloc := localChecksB_sound graph certificate hexp localChecks_true
  exact invariants_of_expanded_certificate graph certificate hexp hloc

end DLDS.Examples.ThreeVertex.Flow.Certificate
