import DLDS.All

/-!
# Historical Readings definitions

Despite the `Probe` filename, this core module defines `ReadingData` and the
finite selector space used by the historical Readings development. No theorem
in the repository identifies this family with successor occurrence replay.
-/

open DLDS
open Flow

namespace Readings.Selection

def outgoing (d : Graph) (v : Vertex) : List Deduction :=
  get_rule.outgoing v d

def branchPoints (d : Graph) : List Vertex :=
  (d.NODES.filter fun v => 2 ≤ (outgoing d v).length).mergeSort
    (fun u v => u.NUMBER ≤ v.NUMBER)

inductive ReadingData : List Nat → Type
  | nil : ReadingData []
  | cons {n : Nat} {ns : List Nat} :
      Fin n → ReadingData ns → ReadingData (n :: ns)
  deriving Repr

def fanArities (d : Graph) : List Nat :=
  (branchPoints d).map fun v => (outgoing d v).length

def enumerateReadingData : (ns : List Nat) → List (ReadingData ns)
  | [] => [.nil]
  | n :: ns =>
      (List.finRange n).flatMap fun i =>
        (enumerateReadingData ns).map fun rest => .cons i rest

def readings (d : Graph) : List (ReadingData (fanArities d)) :=
  enumerateReadingData (fanArities d)

def ReadingData.indices : {ns : List Nat} → ReadingData ns → List Nat
  | [], .nil => []
  | _ :: _, .cons i rest => i.val :: rest.indices

def branchIndex (d : Graph) (v : Vertex) : Nat :=
  (branchPoints d).idxOf v

def selectedEdge (d : Graph) (r : ReadingData (fanArities d))
    (v : Vertex) : Option Deduction :=
  let es := outgoing d v
  if 2 ≤ es.length then
    es[(r.indices.getD (branchIndex d v) 0)]?
  else
    es.head?

def selectedResidual (d : Graph) (r : ReadingData (fanArities d)) :
    Nat → Vertex → ColourPath
  | 0, _ => []
  | fuel + 1, v =>
      match selectedEdge d r v with
      | none => []
      | some e => e.COLOUR :: selectedResidual d r fuel e.END

def normalizedResidual (d : Graph) (r : ReadingData (fanArities d)) :
    Nat → Vertex → ColourPath
  | 0, _ => []
  | fuel + 1, v =>
      match selectedEdge d r v with
      | none => []
      | some e =>
          let rest := normalizedResidual d r fuel e.END
          if 2 ≤ (outgoing d v).length then e.COLOUR :: rest else rest

def sourceNodes (d : Graph) : List Vertex :=
  d.NODES.filter fun v => (predsOf d v).isEmpty

def selectedReachable (d : Graph) (r : ReadingData (fanArities d)) :
    List Vertex :=
  go (d.NODES.length + 1) (sourceNodes d) (sourceNodes d)
where
  go : Nat → List Vertex → List Vertex → List Vertex
    | 0, _, seen => seen.eraseDups
    | fuel + 1, frontier, seen =>
        let next := frontier.filterMap (selectedEdge d r)
          |>.map (·.END)
          |>.filter (fun v => !(seen.contains v))
          |>.eraseDups
        if next.isEmpty then seen.eraseDups
        else go fuel next (seen ++ next)

def feedsB (d : Graph) (r : ReadingData (fanArities d))
    (u w : Vertex) : Bool :=
  match selectedEdge d r u with
  | none => false
  | some e => decide (e.END = w)

def coherentAtB (d : Graph) (r : ReadingData (fanArities d))
    (w : Vertex) : Bool :=
  if w ∉ selectedReachable d r then true
  else
    match elimPairsAt d w with
    | [(major, minor)] => feedsB d r major w && feedsB d r minor w
    | _ => true

def coherentB (d : Graph) (r : ReadingData (fanArities d)) : Bool :=
  d.NODES.all (coherentAtB d r)

def coherentReadings (d : Graph) : List (ReadingData (fanArities d)) :=
  (readings d).filter (coherentB d)

def incoherentReadings (d : Graph) : List (ReadingData (fanArities d)) :=
  (readings d).filter fun r => !(coherentB d r)

structure P1Failure where
  reading : List Nat
  node : Nat
  predicted : ColourPath
  actual : List FlowPair
  deriving Repr

def p1FailureForReading (d : Graph)
    (r : ReadingData (fanArities d)) : Option P1Failure :=
  (selectedReachable d r).findSome? fun v =>
    let predicted := selectedResidual d r (d.NODES.length + 1) v
    let actual := flowAt d (stdFuel d) v
    if actual.any fun route => route.2 == predicted then none
    else some {
      reading := r.indices
      node := v.NUMBER
      predicted := predicted
      actual := actual
    }

def p1Failure (d : Graph) : Option P1Failure :=
  (readings d).findSome? (p1FailureForReading d)

def p1NormalizedFailureForReading (d : Graph)
    (r : ReadingData (fanArities d)) : Option P1Failure :=
  (selectedReachable d r).findSome? fun v =>
    let predicted := normalizedResidual d r (d.NODES.length + 1) v
    let actual := flowAt d (stdFuel d) v
    if actual.any fun route => route.2 == predicted then none
    else some {
      reading := r.indices
      node := v.NUMBER
      predicted := predicted
      actual := actual
    }

def p1NormalizedFailureAmong (d : Graph)
    (rs : List (ReadingData (fanArities d))) : Option P1Failure :=
  rs.findSome? (p1NormalizedFailureForReading d)

def normalizedResidualsAmong (d : Graph)
    (rs : List (ReadingData (fanArities d))) (v : Vertex) :
    List ColourPath :=
  rs.filterMap fun r =>
    if v ∈ selectedReachable d r then
      some (normalizedResidual d r (d.NODES.length + 1) v)
    else none

def producedResiduals (d : Graph) (v : Vertex) : List ColourPath :=
  (readings d).filterMap fun r =>
    if v ∈ selectedReachable d r then
      some (selectedResidual d r (d.NODES.length + 1) v)
    else none

def p2Table (d : Graph) : List (Nat × Nat × Nat) :=
  d.NODES.map fun v =>
    (v.NUMBER,
      (flowAt d (stdFuel d) v).length,
      (producedResiduals d v).eraseDups.length)

structure P3Failure where
  reading : List Nat
  node : Nat
  major : Nat
  minor : Nat
  majorResidual : ColourPath
  minorResidual : ColourPath
  deriving Repr

def p3FailureAt (d : Graph) (r : ReadingData (fanArities d))
    (w : Vertex) : Option P3Failure :=
  match elimPairsAt d w with
  | [(major, minor)] =>
      if w ∉ selectedReachable d r then none
      else
        let majorResidual := selectedResidual d r (d.NODES.length + 1) major
        let minorResidual := selectedResidual d r (d.NODES.length + 1) minor
        match selectedEdge d r major, selectedEdge d r minor with
        | some majorEdge, some minorEdge =>
            match consume majorEdge majorResidual, consume minorEdge minorResidual with
            | some majorTail, some minorTail =>
                if majorEdge.END = w && minorEdge.END = w &&
                    majorTail = minorTail then none
                else some {
                  reading := r.indices
                  node := w.NUMBER
                  major := major.NUMBER
                  minor := minor.NUMBER
                  majorResidual := majorResidual
                  minorResidual := minorResidual
                }
            | _, _ => some {
                reading := r.indices
                node := w.NUMBER
                major := major.NUMBER
                minor := minor.NUMBER
                majorResidual := majorResidual
                minorResidual := minorResidual
              }
        | _, _ => some {
            reading := r.indices
            node := w.NUMBER
            major := major.NUMBER
            minor := minor.NUMBER
            majorResidual := majorResidual
            minorResidual := minorResidual
          }
  | _ => none

def p3FailureForReading (d : Graph)
    (r : ReadingData (fanArities d)) : Option P3Failure :=
  d.NODES.findSome? (p3FailureAt d r)

def p3Failure (d : Graph) : Option P3Failure :=
  (readings d).findSome? (p3FailureForReading d)

def p4Table (d : Graph) : List (Nat × Nat × Nat) :=
  d.NODES.map fun v =>
    let rs := readings d
    let reachableCount := (rs.filter fun r => v ∈ selectedReachable d r).length
    (v.NUMBER, reachableCount, (producedResiduals d v).eraseDups.length)

def halfFeedsB (d : Graph) (r : ReadingData (fanArities d))
    (w : Vertex) : Bool :=
  if w ∉ selectedReachable d r then false
  else
    match elimPairsAt d w with
    | [(major, minor)] => xor (feedsB d r major w) (feedsB d r minor w)
    | _ => false

def p5Table (d : Graph) : List (Nat × Nat) :=
  d.NODES.filterMap fun w =>
    match elimPairsAt d w with
    | [_] =>
        some (w.NUMBER, ((readings d).filter fun r => halfFeedsB d r w).length)
    | _ => none

def normalizedTailAfterStep (d : Graph)
    (r : ReadingData (fanArities d)) (v : Vertex) : ColourPath :=
  let residual := normalizedResidual d r (d.NODES.length + 1) v
  if 2 ≤ (outgoing d v).length then residual.tail else residual

def flowEqualTailAtB (d : Graph) (r : ReadingData (fanArities d))
    (w : Vertex) : Bool :=
  if w ∉ selectedReachable d r then true
  else
    match elimPairsAt d w with
    | [(major, minor)] =>
        normalizedTailAfterStep d r major =
          normalizedTailAfterStep d r minor
    | _ => true

def flowEqualTailB (d : Graph) (r : ReadingData (fanArities d)) : Bool :=
  d.NODES.all (flowEqualTailAtB d r)

structure P6Disagreement where
  reading : List Nat
  structural : Bool
  equalTail : Bool
  deriving Repr

def p6StructuralNotFlow (d : Graph) : Option P6Disagreement :=
  (readings d).findSome? fun r =>
    if coherentB d r && !(flowEqualTailB d r) then
      some { reading := r.indices, structural := true, equalTail := false }
    else none

def p6FlowNotStructural (d : Graph) : Option P6Disagreement :=
  (readings d).findSome? fun r =>
    if flowEqualTailB d r && !(coherentB d r) then
      some { reading := r.indices, structural := false, equalTail := true }
    else none

structure P7Row where
  node : Nat
  routeCount : Nat
  coveredCount : Nat
  missingResiduals : List ColourPath
  deriving Repr

def p7Table (d : Graph) : List P7Row :=
  d.NODES.map fun v =>
    let actual := flowAt d (stdFuel d) v
    let produced :=
      (normalizedResidualsAmong d (coherentReadings d) v).eraseDups
    let missing := actual.filterMap fun route =>
      if route.2 ∈ produced then none else some route.2
    {
      node := v.NUMBER
      routeCount := actual.length
      coveredCount := actual.length - missing.length
      missingResiduals := missing
    }

structure P8Report where
  coherentCount : Nat
  root : Nat
  rootResiduals : List ColourPath
  distinctProfileCount : Nat
  profilesSeparate : Bool
  deriving Repr

def residualProfile (d : Graph) (r : ReadingData (fanArities d)) :
    List (Nat × ColourPath) :=
  d.NODES.map fun v =>
    (v.NUMBER, normalizedResidual d r (d.NODES.length + 1) v)

def selectedRoot (d : Graph) : Option Vertex :=
  d.NODES.find? fun v => (outgoing d v).isEmpty

def p8Report (d : Graph) : Option P8Report :=
  match selectedRoot d with
  | none => none
  | some root =>
      let coherent := coherentReadings d
      let rootResiduals :=
        (coherent.map fun r =>
          normalizedResidual d r (d.NODES.length + 1) root).eraseDups
      let profiles := (coherent.map (residualProfile d)).eraseDups
      some {
        coherentCount := coherent.length
        root := root.NUMBER
        rootResiduals := rootResiduals
        distinctProfileCount := profiles.length
        profilesSeparate := profiles.length = coherent.length
      }

namespace Examples.ColouredCompression

def A1 : Formula := #"A1"
def A2 : Formula := #"A2"
def A3 : Formula := #"A3"
def A4 : Formula := #"A4"
def A5 : Formula := #"A5"

def h1Formula : Formula := A1
def h2Formula : Formula := A1 >> A2
def h3Formula : Formula := A1 >> (A2 >> A3)
def h4Formula : Formula := A2 >> (A3 >> A4)
def h5Formula : Formula := A3 >> (A4 >> A5)

def h1 : Vertex := node 0 4 h1Formula true false []
def h2 : Vertex := node 1 4 h2Formula true false []
def h3 : Vertex := node 2 4 h3Formula true false []
def h4 : Vertex := node 3 3 h4Formula true false []
def h5 : Vertex := node 4 2 h5Formula true false []
def a2 : Vertex := node 5 3 A2 false false []
def a2ToA3 : Vertex := node 6 3 (A2 >> A3) false false []
def a3 : Vertex := node 7 2 A3 false false []
def a3ToA4 : Vertex := node 8 2 (A3 >> A4) false false []
def a4 : Vertex := node 9 1 A4 false false []
def a4ToA5 : Vertex := node 10 1 (A4 >> A5) false false []
def root : Vertex := node 11 0 A5 false false []

def edgeH1A2 : Deduction := edge h1 a2 0 [h1Formula]
def edgeH1A2ToA3 : Deduction := edge h1 a2ToA3 1 [h1Formula]
def edgeH2A2 : Deduction := edge h2 a2 0 [h2Formula]
def edgeH3A2ToA3 : Deduction := edge h3 a2ToA3 0 [h3Formula]
def edgeA2A3 : Deduction := edge a2 a3 0 [A1, h2Formula]
def edgeA2A3ToA4 : Deduction := edge a2 a3ToA4 1 [A1, h2Formula]
def edgeA2ToA3A3 : Deduction := edge a2ToA3 a3 0 [A1, h3Formula]
def edgeH4A3ToA4 : Deduction := edge h4 a3ToA4 0 [h4Formula]
def edgeA3A4 : Deduction := edge a3 a4 0 [A1, h2Formula, h3Formula]
def edgeA3A4ToA5 : Deduction := edge a3 a4ToA5 1 [A1, h2Formula, h3Formula]
def edgeA3ToA4A4 : Deduction := edge a3ToA4 a4 0 [A1, h2Formula, h4Formula]
def edgeH5A4ToA5 : Deduction := edge h5 a4ToA5 0 [h5Formula]
def edgeA4Root : Deduction :=
  edge a4 root 0 [A1, h2Formula, h3Formula, h4Formula]
def edgeA4ToA5Root : Deduction :=
  edge a4ToA5 root 0 [A1, h2Formula, h3Formula, h5Formula]

def graph : Graph :=
  dlds
    [h1, h2, h3, h4, h5, a2, a2ToA3, a3, a3ToA4, a4, a4ToA5, root]
    [edgeH1A2, edgeH1A2ToA3, edgeH2A2, edgeH3A2ToA3,
      edgeA2A3, edgeA2A3ToA4, edgeA2ToA3A3, edgeH4A3ToA4,
      edgeA3A4, edgeA3A4ToA5, edgeA3ToA4A4, edgeH5A4ToA5,
      edgeA4Root, edgeA4ToA5Root]
    []

abbrev PaperDep := List Bool

def zero : PaperDep := List.replicate 5 false

def oneHot (i : Nat) : PaperDep :=
  (List.range 5).map fun j => decide (i = j)

def depOr (a b : PaperDep) : PaperDep :=
  List.zipWith (· || ·) a b

def lookupDep : List (Vertex × PaperDep) → Vertex → PaperDep
  | [], _ => zero
  | (u, dep) :: rest, v => if u = v then dep else lookupDep rest v

def hypothesisIndex (v : Vertex) : Option Nat :=
  if v = h1 then some 0
  else if v = h2 then some 1
  else if v = h3 then some 2
  else if v = h4 then some 3
  else if v = h5 then some 4
  else none

def edgeActiveB (r : ReadingData (fanArities graph))
    (e : Deduction) : Bool :=
  match selectedEdge graph r e.START with
  | none => false
  | some selected => decide (selected = e)

def paperStep (r : ReadingData (fanArities graph))
    (env : List (Vertex × PaperDep)) (v : Vertex) : PaperDep :=
  match hypothesisIndex v with
  | some i => oneHot i
  | none =>
      (get_rule.incoming v graph).foldl
        (fun acc e =>
          if edgeActiveB r e then depOr acc (lookupDep env e.START)
          else acc)
        zero

def paperOrder : List Vertex :=
  [h1, h2, h3, h4, h5, a2, a2ToA3, a3, a3ToA4, a4, a4ToA5, root]

def paperSemantics (r : ReadingData (fanArities graph)) :
    List (Vertex × PaperDep) :=
  paperOrder.foldl (fun env v => env ++ [(v, paperStep r env v)]) []

def paperRootDep (r : ReadingData (fanArities graph)) : PaperDep :=
  lookupDep (paperSemantics r) root

def allOnes : PaperDep := List.replicate 5 true

def paperTable : List (List Nat × PaperDep × Bool) :=
  (readings graph).map fun r =>
    let dep := paperRootDep r
    (r.indices, dep, dep = allOnes)

def fanTable : List (Nat × Nat) :=
  graph.NODES.map fun v => (v.NUMBER, (outgoing graph v).length)

def flowTable : List (Nat × List FlowPair) :=
  graph.NODES.map fun v => (v.NUMBER, flowAt graph (stdFuel graph) v)

end Examples.ColouredCompression


end Readings.Selection
