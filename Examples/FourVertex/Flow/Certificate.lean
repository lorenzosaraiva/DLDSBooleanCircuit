import Examples.FourVertex.Flow.Closed
import Certificate.Flow.IntroductionSuffix

open DLDS FlowSemantics Certificate.Flow.Foundation Certificate.Flow

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Examples.FourVertex.Flow.Certificate

def graph : Graph := DLDS.Examples.FourVertex.Flow.Closed.graph

def openCertificate : ExpandedCertificate :=
  generateCertificate DLDS.Examples.FourVertex.Flow.Open.graph

def openRootRoutes : List FlowPair :=
  openCertificate.terminalOutputs.filterMap fun output =>
    if output.node == DLDS.Examples.FourVertex.Flow.Open.v325 then
      some (output.dependency, output.residual)
    else none

def shiftedOpenWitnesses : List EdgeRouteWitness :=
  openCertificate.edgeRoutes.map fun witness =>
    { witness with
      edge := shiftDeduction DLDS.Examples.FourVertex.Flow.Closed.suffixLength witness.edge }

def suffixData : List EdgeRouteWitness × List FlowPair :=
  DLDS.Certificate.Flow.IntroductionSuffix.suffixCertificateData
    DLDS.Examples.FourVertex.Flow.Closed.suffixEdges openRootRoutes

def certificate : ExpandedCertificate :=
  let openCert := generateCertificate DLDS.Examples.FourVertex.Flow.Open.graph
  let rootRoutes := openCert.terminalOutputs.filterMap fun output =>
    if output.node == DLDS.Examples.FourVertex.Flow.Open.v325 then
      some (output.dependency, output.residual)
    else none
  let shifted := openCert.edgeRoutes.map fun witness =>
    { witness with
      edge := shiftDeduction DLDS.Examples.FourVertex.Flow.Closed.suffixLength witness.edge }
  let suffix := DLDS.Certificate.Flow.IntroductionSuffix.suffixCertificateData
    DLDS.Examples.FourVertex.Flow.Closed.suffixEdges rootRoutes
  { graph
    edgeRoutes := shifted ++ suffix.1
    terminalOutputs := suffix.2.map fun route =>
      { node := DLDS.Examples.FourVertex.Flow.Closed.suffixVertex DLDS.Examples.FourVertex.Flow.Closed.suffixLength
        dependency := route.1
        residual := route.2 } }

def certificateReport : DLDS.Certificate.Flow.IntroductionSuffix.CertificateReport :=
  DLDS.Certificate.Flow.IntroductionSuffix.certificateReport graph certificate

end DLDS.Examples.FourVertex.Flow.Certificate
