import Examples.Flow.ClosedImplication
import Examples.ThreeVertex.Flow.Open

/-!
The closed version of the imported `3v_no_hc` derivation.

`Pipeline3vNoHC.graph` ends in an open derivation of `q`.  As in the
`d0closed` construction in `Examples.Flow.ClosedImplication`, the whole open graph is
shifted upward and a chain of implication-introduction rules is appended.
The chain discharges the root dependencies from right to left.
-/

open DLDS

namespace DLDS.Examples.ThreeVertex.Flow.Closed

/-- The open context `alpha_G`: every distinct formula carried by a hypothesis
node, in graph order.  This is the direct analogue of `hs` in
`Examples.Flow.ClosedImplication`.  Reading only the final elimination edges is insufficient
for this fixture because their route-specific decorations omit three source
hypotheses that raw Flow still carries to the root. -/
def hypotheses : List Formula :=
  ((DLDS.Examples.ThreeVertex.Flow.Open.graph.NODES.filter (·.HYPOTHESIS)).map
    (·.FORMULA)).eraseDups

def suffixLength : Nat := hypotheses.length

def shiftedOpen : Graph :=
  dlds (DLDS.Examples.ThreeVertex.Flow.Open.graph.NODES.map (shiftVertex suffixLength))
       (DLDS.Examples.ThreeVertex.Flow.Open.graph.EDGES.map (shiftDeduction suffixLength))
       (DLDS.Examples.ThreeVertex.Flow.Open.graph.PATHS.map (shiftAncestral suffixLength))

/-- Formula after `step` hypotheses have been discharged. -/
def suffixFormula (step : Nat) : Formula :=
  (hypotheses.drop (suffixLength - step)).foldr Formula.implication
    DLDS.Examples.ThreeVertex.Flow.Open.v127.FORMULA

/-- `step = 0` is the shifted open root; positive steps are the appended
introduction nodes. -/
def suffixVertex (step : Nat) : Vertex :=
  if step = 0 then
    shiftVertex suffixLength DLDS.Examples.ThreeVertex.Flow.Open.v127
  else
    node (DLDS.Examples.ThreeVertex.Flow.Open.v127.NUMBER + step)
      (suffixLength - step) (suffixFormula step) false false []

def suffixNodes : List Vertex :=
  (List.range suffixLength).map fun index => suffixVertex (index + 1)

/-- The edge at `index` carries the context immediately before its
antecedent is discharged. -/
def suffixEdges : List Deduction :=
  (List.range suffixLength).map fun index =>
    edge (suffixVertex index) (suffixVertex (index + 1)) 0
      (normalizeDep (hypotheses.take (suffixLength - index)))

def extended : Graph :=
  dlds (shiftedOpen.NODES ++ suffixNodes)
       (shiftedOpen.EDGES ++ suffixEdges)
       shiftedOpen.PATHS

/-- The suffix introduces one fresh level per node, while the imported open
graph is already at its HC fixpoint.  `extended` is therefore the compressed
graph; the certificate module checks that its level/formula compression keys
are duplicate-free. -/
def graph : Graph := extended

end DLDS.Examples.ThreeVertex.Flow.Closed
