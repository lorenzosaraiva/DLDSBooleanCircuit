import Semantic

open Semantic
open FlowSpec

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace Semantic

def shiftVertex (k : Nat) (v : Vertex) : Vertex :=
  node v.NUMBER (v.LEVEL + k) v.FORMULA v.HYPOTHESIS v.COLLAPSED v.PAST

def shiftDeduction (k : Nat) (e : Deduction) : Deduction :=
  edge (shiftVertex k e.START) (shiftVertex k e.END) e.COLOUR e.DEPENDENCY

def shiftAncestral (k : Nat) (p : Ancestral) : Ancestral :=
  path (shiftVertex k p.START) (shiftVertex k p.END) p.COLOURS

def shiftedD0 : Graph :=
  dlds (d0.NODES.map (shiftVertex 5))
       (d0.EDGES.map (shiftDeduction 5))
       (d0.PATHS.map (shiftAncestral 5))

def h1 : Formula := #"A1"
def h2 : Formula := #"A1" >> #"A2"
def h3 : Formula := #"A1" >> (#"A2" >> #"A3")
def h4 : Formula := #"A2" >> (#"A3" >> #"A4")
def h5 : Formula := #"A3" >> (#"A4" >> #"A5")
def c0 : Formula := #"A4" >> #"A5"

def q23 : Vertex := shiftVertex 5 n23
def q24 : Vertex := node 24 4 (h5 >> c0) false false []
def q25 : Vertex := node 25 3 (h4 >> (h5 >> c0)) false false []
def q26 : Vertex := node 26 2 (h3 >> (h4 >> (h5 >> c0))) false false []
def q27 : Vertex := node 27 1 (h2 >> (h3 >> (h4 >> (h5 >> c0)))) false false []
def q28 : Vertex := node 28 0 (h1 >> (h2 >> (h3 >> (h4 >> (h5 >> c0))))) false false []

def hs : List Formula := [h1, h2, h3, h4, h5]

def introEdges : List Deduction :=
  [edge q23 q24 0 hs,
   edge q24 q25 0 [h1, h2, h3, h4],
   edge q25 q26 0 [h1, h2, h3],
   edge q26 q27 0 [h1, h2],
   edge q27 q28 0 [h1]]

def d0closed : Graph :=
  dlds (shiftedD0.NODES ++ [q24, q25, q26, q27, q28])
       (shiftedD0.EDGES ++ introEdges)
       shiftedD0.PATHS

def e : Graph := compress_nodes_graph d0closed

def structuralValidB (d : Graph) : Bool :=
  (d.EDGES.all fun x => x.START.LEVEL == x.END.LEVEL + 1) &&
  (d.EDGES.all fun x => d.NODES.contains x.START && d.NODES.contains x.END) &&
  (d.NODES.all fun v =>
    if !(get_rule.incoming v d).isEmpty && !(get_rule.outgoing v d).isEmpty then
      nodeElimShapeB d v || nodeIntroShapeB d v
    else true)

def nodupB (d : Graph) : Bool := decide d.EDGES.Nodup

theorem nodupB_sound (d : Graph) : nodupB d = true → d.EDGES.Nodup := by
  simp [nodupB]

def oneEdgeB (d : Graph) : Bool :=
  d.EDGES.all fun x =>
    d.EDGES.all fun y =>
      if x.START == y.START && x.COLOUR == y.COLOUR then decide (x = y) else true

def routeFanB (d : Graph) : Bool := d.NODES.all (routeFanUniqueAtB d)
def faithfulB (d : Graph) : Bool := d.NODES.all (faithfulNewAtB d)
def coverageB (d : Graph) : Bool :=
  d.NODES.all fun v =>
    (get_rule.outgoing v d).isEmpty || routeHeadCoverageAtB d v
def hypNoIncomingB (d : Graph) : Bool :=
  d.NODES.all fun v =>
    if v.HYPOTHESIS then (get_rule.incoming v d).isEmpty else true

def invariantTable (d : Graph) : List (String × Bool) :=
  [("structuralValid", structuralValidB d),
   ("EDGES.Nodup", nodupB d),
   ("OneEdgePerColourPerNode", oneEdgeB d),
   ("RouteFanUnique", routeFanB d),
   ("FaithfulDecoration", faithfulB d),
   ("RouteHeadCoverage", coverageB d),
   ("LevelFormulaUnique", levelFormulaUniqueB d),
   ("HypothesesHaveNoIncoming", hypNoIncomingB d),
   ("ReseedFree", reseedFreeB d),
   ("FlowTailClosure", flowTailClosureB d),
   ("dischargedMultiB", dischargedMultiB d)]

theorem nodeElimShapeB_sound (d : Graph) (v : Vertex) :
    nodeElimShapeB d v = true → NodeElimShape d v := by
  simp [nodeElimShapeB, NodeElimShape] <;> aesop

theorem nodeIntroShapeB_sound (d : Graph) (v : Vertex) :
    nodeIntroShapeB d v = true → NodeIntroShape d v := by
  simp [nodeIntroShapeB, NodeIntroShape] <;> aesop

theorem structuralValidB_sound (d : Graph) :
    structuralValidB d = true → structuralValid d := by
  classical
  simp [structuralValidB, structuralValid]
  intro hlev hend hshape
  refine ⟨hlev, hend, ?_⟩
  intro v hv hin hout
  rcases hshape v hv with hterm | helim | hintro
  · rcases hterm with hin' | hout'
    · exact (hin hin').elim
    · exact (hout hout').elim
  · exact Or.inl (nodeElimShapeB_sound d v helim)
  · exact Or.inr (nodeIntroShapeB_sound d v hintro)

def failedNodes (d : Graph) :=
  (d.NODES.filterMap fun v =>
      if !(get_rule.incoming v d).isEmpty && !(get_rule.outgoing v d).isEmpty &&
          !(nodeElimShapeB d v || nodeIntroShapeB d v) then some v.NUMBER else none,
   d.NODES.filterMap fun v => if faithfulNewAtB d v then none else some v.NUMBER,
   d.NODES.filterMap fun v =>
      if !(get_rule.outgoing v d).isEmpty && !routeHeadCoverageAtB d v then
        some v.NUMBER else none,
   d.NODES.filterMap fun v => if flowTailClosureAtB d v then none else some v.NUMBER)

theorem oneEdgeB_sound (d : Graph) :
    oneEdgeB d = true → OneEdgePerColourPerNode d := by
  classical
  simp [oneEdgeB, OneEdgePerColourPerNode]
  intro h e1 e2 he1 he2 hs hc
  rcases h e1 he1 e2 he2 with hdiff | heq
  · rcases hdiff with hs' | hc'
    · exact (hs' hs).elim
    · exact (hc' hc).elim
  · exact heq

theorem routeFanB_sound (d : Graph) :
    routeFanB d = true → RouteFanUnique d := by
  classical
  simp [routeFanB, routeFanUniqueAtB, RouteFanUnique]
  intro h v hv a b a' b' hab hab' hh
  rcases h v hv a b hab a' b' hab' with hh' | heq
  · exact (hh' hh).elim
  · exact heq

theorem faithfulB_sound (d : Graph) :
    faithfulB d = true → FaithfulDecoration d := by
  simp [faithfulB, faithfulNewAtB, FaithfulDecoration]

theorem coverageB_sound (d : Graph) :
    coverageB d = true → RouteHeadCoverage d := by
  classical
  simp [coverageB, routeHeadCoverageAtB, RouteHeadCoverage]
  intro h v hv hout
  rcases h v hv with hout' | hcov
  · exact (hout hout').elim
  · exact hcov

theorem levelFormulaUniqueB_sound (d : Graph) :
    levelFormulaUniqueB d = true → LevelFormulaUnique d := by
  classical
  simp [levelFormulaUniqueB, LevelFormulaUnique]
  intro h v1 hv1 v2 hv2 hl hf
  rcases h v1 hv1 v2 hv2 with hl' | hf' | heq
  · exact (hl' hl).elim
  · exact (hf' hf).elim
  · exact heq

theorem hypNoIncomingB_sound (d : Graph) :
    hypNoIncomingB d = true → HypothesesHaveNoIncoming d := by
  classical
  simp [hypNoIncomingB, HypothesesHaveNoIncoming]
  intro h v hv hh
  exact (h v hv).resolve_left (by simpa [hh])

theorem reseedFreeB_sound (d : Graph) :
    reseedFreeB d = true → ReseedFree d := by
  simp [reseedFreeB, ReseedFree]

theorem flowTailClosureB_sound (d : Graph) :
    flowTailClosureB d = true → FlowTailClosure d := by
  classical
  simp [flowTailClosureB, flowTailClosureAtB, FlowTailClosure]
  intro h w hw M S hp
  have hw' := h w hw
  simp [hp] at hw'
  rcases hw' with ⟨⟨hpred, hSM⟩, hMS⟩
  refine ⟨hpred, ?_, ?_⟩
  · intro a b hb eS heS q hc
    have hx := hSM a b hb eS heS
    simp [hc] at hx
    exact hx
  · intro a b hb eM heM q hc
    have hx := hMS a b hb eM heM
    simp [hc] at hx
    exact hx

theorem compressed_universally_accepted_exec (d : Graph)
    (hstruct : structuralValidB d = true)
    (hnodup : nodupB d = true)
    (honepercol : oneEdgeB d = true)
    (hroutefan : routeFanB d = true)
    (hfaithful : faithfulB d = true)
    (hcoverage : coverageB d = true)
    (hlfu : levelFormulaUniqueB d = true)
    (hhypNoInc : hypNoIncomingB d = true)
    (hreseed : reseedFreeB d = true)
    (hftc : flowTailClosureB d = true)
    (hdis : dischargedMultiB d = true) :
    ∀ P, ¬ AdmissibleMultiPath d P ∨
      DischargedMulti d P (goalColumn d) :=
  compressed_universally_accepted' d
    (structuralValidB_sound d hstruct)
    (nodupB_sound d hnodup)
    (oneEdgeB_sound d honepercol)
    (routeFanB_sound d hroutefan)
    (faithfulB_sound d hfaithful)
    (coverageB_sound d hcoverage)
    (levelFormulaUniqueB_sound d hlfu)
    (hypNoIncomingB_sound d hhypNoInc)
    (reseedFreeB_sound d hreseed)
    (flowTailClosureB_sound d hftc)
    hdis

theorem end_to_end :
    ∀ P, ¬ AdmissibleMultiPath e P ∨
      DischargedMulti e P (goalColumn e) := by
  apply compressed_universally_accepted_exec
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide
  · native_decide

end Semantic
