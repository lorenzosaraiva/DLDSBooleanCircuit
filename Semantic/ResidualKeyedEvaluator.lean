import Semantic.RefinedOccurrenceEvaluator

open Semantic FlowSpec SelfCertifyingProbe

namespace KeyedEval

abbrev Vector (d : Graph) := List.Vector Bool (RefinedEval.formulas d).length
abbrev Row (d : Graph) := ColourPath × Vector d
abbrev Carrier (d : Graph) := List (Row d)
abbrev Table (d : Graph) := List (Vertex × Carrier d)

def decode (d : Graph) (vector : Vector d) : Dep :=
  (RefinedEval.formulas d |>.zip vector.toList).filterMap fun (formula, bit) =>
    if bit then some formula else none




def merge (d : Graph) (routes : List FlowPair) : Carrier d :=
  (routes.map Prod.snd).eraseDups.map fun residual =>
    (residual, depVector (RefinedEval.formulas d)
      ((routes.filter fun route => route.2 == residual).flatMap Prod.fst))

def expand (d : Graph) (rows : Carrier d) : List FlowPair :=
  rows.map fun row => (decode d row.2, row.1)




def step (d : Graph) (prior : Vertex → Carrier d) (v : Vertex) : Carrier d :=
  merge d (flowStepFrom d (fun u => expand d (prior u)) v)

def evaluateAt (d : Graph) : Nat → Vertex → Carrier d
  | 0, _ => []
  | fuel + 1, v => step d (evaluateAt d fuel) v

def keyCount (d : Graph) (fuel : Nat) (v : Vertex) : Nat :=
  (evaluateAt d fuel v).length

def lookup (d : Graph) (table : Table d) (v : Vertex) : Option (Carrier d) :=
  (table.find? fun entry => entry.1 == v).map Prod.snd

def fromTable (d : Graph) (fuel : Nat) (table : Table d) (v : Vertex) : Carrier d :=
  match lookup d table v with
  | some rows => rows
  | none => evaluateAt d fuel v

def tableStep (d : Graph) (fuel : Nat) (table : Table d) : Table d :=
  d.NODES.map fun v => (v, step d (fromTable d fuel table) v)

def iterate (d : Graph) (mayStop : Bool) : Nat → Nat → Table d → Table d
  | 0, _, previous => previous
  | remaining + 1, fuel, previous =>
      let next := tableStep d fuel previous
      if mayStop && next == previous then next
      else iterate d mayStop remaining (fuel + 1) next

def canonicalTable (d : Graph) : Table d :=
  iterate d (flowTableClosedB d) (stdFuel d) 0 (d.NODES.map fun v => (v, []))




def rootsAcceptedB (d : Graph) (table : Table d) : Bool :=
  d.NODES.all fun v =>
    !(get_rule.outgoing v d).isEmpty ||
      (let rows := (lookup d table v).getD []
       !rows.isEmpty && rows.all fun row =>
         row.1.isEmpty && row.2.toList.all (!·))

def canonicalAcceptedB (d : Graph) : Bool := rootsAcceptedB d (canonicalTable d)

def maxKeys (d : Graph) (table : Table d) : Nat :=
  (table.map fun entry => entry.2.length).foldl max 0

def maxLevelKeys (d : Graph) (table : Table d) : Nat :=
  ((RefinedEval.levels d).map fun level =>
    ((table.filter fun entry => entry.1.LEVEL == level).map fun entry =>
      entry.2.length).sum).foldl max 0

end KeyedEval
