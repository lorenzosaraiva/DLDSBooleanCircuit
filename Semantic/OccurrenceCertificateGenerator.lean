import Semantic.OccurrenceCertificateModel

open Semantic FlowSpec SelfCertifyingProbe SelfCertifying

namespace OccurrenceCertificate



def generateOccurrenceCertificateFromTable (d : Graph) (table : FlowTable) : OccurrenceCertificate :=
  let routes := LambdaFlow.fromTable d LambdaFlow.allowAll LambdaFlow.allowAll
    (stdFuel d) table
  { graph := d
    edgeRoutes := d.EDGES.flatMap (witnessForEdgeFrom routes)
    terminalOutputs := d.NODES.flatMap fun v =>
      if (get_rule.outgoing v d).isEmpty then
        (routes v).map fun route =>
          { node := v, dependency := route.1, residual := route.2 }
      else [] }

def generateOccurrenceCertificate (d : Graph) : OccurrenceCertificate :=
  generateOccurrenceCertificateFromTable d (LambdaFlow.guardFreeTable d)

theorem generateOccurrenceCertificate_eq_raw (d : Graph) :
    generateOccurrenceCertificate d =
      { graph := d
        edgeRoutes := d.EDGES.flatMap
          (witnessForEdgeFrom (LambdaFlow.guardFreeFlowAt d (stdFuel d)))
        terminalOutputs := d.NODES.flatMap fun v =>
          if (get_rule.outgoing v d).isEmpty then
            (LambdaFlow.guardFreeFlowAt d (stdFuel d) v).map fun route =>
              { node := v, dependency := route.1, residual := route.2 }
          else [] } := by
  simp only [generateOccurrenceCertificate, generateOccurrenceCertificateFromTable,
    LambdaFlow.guardFreeTable, LambdaFlow.tableAt_eq_spec,
    LambdaFlow.fromTable_spec, LambdaFlow.guardFreeFlowAt]
  rw [show LambdaFlow.fromTable d LambdaFlow.allowAll LambdaFlow.allowAll
      (stdFuel d) (LambdaFlow.specificationTable d LambdaFlow.allowAll
        LambdaFlow.allowAll (stdFuel d)) =
      LambdaFlow.evaluateAt d LambdaFlow.allowAll LambdaFlow.allowAll (stdFuel d)
    from funext (LambdaFlow.fromTable_spec d LambdaFlow.allowAll
      LambdaFlow.allowAll (stdFuel d))]

@[simp] theorem generateOccurrenceCertificate_graph (d : Graph) :
    (generateOccurrenceCertificate d).graph = d := rfl

#print axioms generateOccurrenceCertificate_eq_raw

end OccurrenceCertificate
