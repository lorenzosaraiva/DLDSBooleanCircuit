import Examples.Flow.ClosedImplication
import Examples.FourVertex.Flow.Open

/-! The closed version of the imported `4v_no_hc` derivation. -/

open DLDS

namespace DLDS.Examples.FourVertex.Flow.Closed

def hypotheses : List Formula :=
  ((DLDS.Examples.FourVertex.Flow.Open.graph.NODES.filter (·.HYPOTHESIS)).map
    (·.FORMULA)).eraseDups

def suffixLength : Nat := hypotheses.length

def shiftedOpen : Graph :=
  dlds (DLDS.Examples.FourVertex.Flow.Open.graph.NODES.map (shiftVertex suffixLength))
       (DLDS.Examples.FourVertex.Flow.Open.graph.EDGES.map (shiftDeduction suffixLength))
       (DLDS.Examples.FourVertex.Flow.Open.graph.PATHS.map (shiftAncestral suffixLength))

def suffixFormula (step : Nat) : Formula :=
  (hypotheses.drop (suffixLength - step)).foldr Formula.implication
    DLDS.Examples.FourVertex.Flow.Open.v325.FORMULA

def suffixVertex (step : Nat) : Vertex :=
  if step = 0 then
    shiftVertex suffixLength DLDS.Examples.FourVertex.Flow.Open.v325
  else
    node (DLDS.Examples.FourVertex.Flow.Open.v325.NUMBER + step)
      (suffixLength - step) (suffixFormula step) false false []

def suffixNodes : List Vertex :=
  (List.range suffixLength).map fun index => suffixVertex (index + 1)

def suffixEdges : List Deduction :=
  (List.range suffixLength).map fun index =>
    edge (suffixVertex index) (suffixVertex (index + 1)) 0
      (normalizeDep (hypotheses.take (suffixLength - index)))

def extended : Graph :=
  dlds (shiftedOpen.NODES ++ suffixNodes)
       (shiftedOpen.EDGES ++ suffixEdges)
       shiftedOpen.PATHS

/-- Run HC over the extended derivation.  The measurement module checks that
both the input and output candidate sets are empty. -/
def graph : Graph := compress_nodes_graph extended

structure ConstructionReport where
  openNodes : Nat
  openEdges : Nat
  closedNodes : Nat
  closedEdges : Nat
  candidatesBefore : Bool
  candidatesAfter : Bool
  compressionUnchanged : Bool
  deriving BEq, DecidableEq, Repr

def constructionReport : ConstructionReport where
  openNodes := DLDS.Examples.FourVertex.Flow.Open.graph.NODES.length
  openEdges := DLDS.Examples.FourVertex.Flow.Open.graph.EDGES.length
  closedNodes := graph.NODES.length
  closedEdges := graph.EDGES.length
  candidatesBefore := repeated_nodes_vertex extended == []
  candidatesAfter := repeated_nodes_vertex graph == []
  compressionUnchanged := graph == extended

def expectedConstructionReport : ConstructionReport where
  openNodes := 216
  openEdges := 278
  closedNodes := 272
  closedEdges := 334
  candidatesBefore := true
  candidatesAfter := true
  compressionUnchanged := true

theorem constructionReport_eq_expected :
    constructionReport = expectedConstructionReport := by
  native_decide

end DLDS.Examples.FourVertex.Flow.Closed
