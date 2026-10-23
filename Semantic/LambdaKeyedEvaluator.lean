import Semantic.LambdaFlowModel
import Semantic.KeyedSummaryFacts

open Semantic FlowSpec

namespace LambdaKeyed

abbrev Carrier := KeyedEval.Carrier
abbrev Table := KeyedEval.Table

def step (d : Graph) (prior : Vertex → Carrier d) (v : Vertex) : Carrier d :=
  KeyedEval.merge d (LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll
    (fun u => KeyedEval.expand d (prior u)) v)

def evaluateAt (d : Graph) : Nat → Vertex → Carrier d
  | 0, _ => []
  | fuel + 1, v => step d (evaluateAt d fuel) v

def fromTable (d : Graph) (fuel : Nat) (table : Table d) (v : Vertex) : Carrier d :=
  match KeyedEval.lookup d table v with
  | some rows => rows
  | none => evaluateAt d fuel v

def tableStep (d : Graph) (fuel : Nat) (table : Table d) : Table d :=
  d.NODES.map fun v => (v, step d (fromTable d fuel table) v)

def iterate (d : Graph) : Nat → Nat → Table d → Table d
  | 0, _, table => table
  | remaining + 1, fuel, table =>
      let next := tableStep d fuel table
      if flowTableClosedB d && next == table then next
      else iterate d remaining (fuel + 1) next

def canonicalTable (d : Graph) : Table d :=
  iterate d (stdFuel d) 0 (d.NODES.map fun v => (v, []))

def canonicalAcceptedB (d : Graph) : Bool :=
  KeyedEval.rootsAcceptedB d (canonicalTable d)

end LambdaKeyed
