import Semantic.RouteIndexedMultiTokenModel

open Semantic FlowSpec SelfCertifyingProbe

namespace RefinedEval



def formulas (d : Graph) : List Formula :=
  normalizeDep (d.NODES.map (·.FORMULA) ++
    d.EDGES.flatMap fun edge => [edge.START.FORMULA, edge.END.FORMULA])

structure Occurrence (width : Nat) where
  owner : Vertex
  vector : List.Vector Bool width
  residual : ColourPath
  deriving BEq, DecidableEq

abbrev Token (d : Graph) := Occurrence (formulas d).length
abbrev Table (d : Graph) := List (Vertex × List (Token d))

def decode (d : Graph) (vector : List.Vector Bool (formulas d).length) : Dep :=
  normalizeDep ((formulas d).zipIdx.filterMap fun (formula, index) =>
    if vector.toList[index]?.getD false then some formula else none)

def encode (d : Graph) (owner : Vertex) (dependency : Dep) (residual : ColourPath) :
    Token d := ⟨owner, depVector (formulas d) dependency, residual⟩

def inputs (d : Graph) (owner : Vertex) (tokens : List (Token d)) :
    List RouteIndexedMultiToken.RouteIndexedToken :=
  tokens.zipIdx.map fun (token, slot) =>
    RouteIndexedMultiToken.mkToken d owner (.certificate owner slot)
      (decode d token.vector) token.residual





def step (d : Graph) (prior : Vertex → List (Token d)) (v : Vertex) : List (Token d) :=
  ((RouteIndexedMultiToken.stepFrom d (fun u => inputs d u (prior u)) v).map
    fun token => encode d v token.dependency token.residual).eraseDups

def evaluateAt (d : Graph) : Nat → Vertex → List (Token d)
  | 0, _ => []
  | fuel + 1, v => step d (evaluateAt d fuel) v

def lookup (d : Graph) (table : Table d) (v : Vertex) : Option (List (Token d)) :=
  (table.find? fun entry => entry.1 == v).map Prod.snd

def fromTable (d : Graph) (fuel : Nat) (table : Table d) (v : Vertex) : List (Token d) :=
  match lookup d table v with
  | some tokens => tokens
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



def budget (d : Graph) : Nat :=
  ((buildFormulas d).map fun formula =>
    max 1 (((d.NODES.filter fun v => v.HYPOTHESIS && v.FORMULA == formula).flatMap
      fun v => RouteIndexedMultiToken.sourceOutputs d v).length)).sum

def atLevel (d : Graph) (table : Table d) (level : Nat) : List (Token d) :=
  ((d.NODES.filter fun v => v.LEVEL == level).flatMap
    fun v => (lookup d table v).getD []).eraseDups

def levels (d : Graph) : List Nat := (d.NODES.map (·.LEVEL)).eraseDups

def capacityB (d : Graph) (table : Table d) : Bool :=
  (levels d).all fun level => decide ((atLevel d table level).length ≤ budget d)


def frame (d : Graph) (table : Table d) (level : Nat) : List (Option (Token d)) :=
  let selected := (atLevel d table level).take (budget d)
  selected.map some ++ List.replicate (budget d - selected.length) none

def rootsAcceptedB (d : Graph) (table : Table d) : Bool :=
  d.NODES.all fun v =>
    !(get_rule.outgoing v d).isEmpty ||
      (let tokens := (lookup d table v).getD []
       !tokens.isEmpty && tokens.all fun token =>
         token.vector.toList.all (!·) && token.residual.isEmpty)

def boundedAcceptedB (d : Graph) (table : Table d) : Bool :=
  capacityB d table && rootsAcceptedB d table

def refinedCanonicalAcceptanceB (d : Graph) : Bool :=
  boundedAcceptedB d (canonicalTable d)

end RefinedEval
