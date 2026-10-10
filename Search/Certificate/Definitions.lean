import Certificate.Encoding.Export
import Certificate.Flow.Soundness

open DLDS Flow Certificate.Flow.Foundation

/-!
# Expanded-certificate search problem

This module packages the existing self-certifying checker as a finite search
problem.  It contains no search algorithm and no quantum semantics: `n` is
only the size of the finite query domain.
-/

namespace Search.Certificate

abbrev Certificate := Certificate.Flow.Certificate

/-- Number of searchable local-check positions. -/
def numPositions (d : Graph) (c : Certificate) : Nat :=
  6 * d.EDGES.length + 8 * d.NODES.length + c.edgeRoutes.length - 3

def sortedBlock (size : Nat) (check : Bool) : List Bool :=
  List.replicate (size - 1) check ++ List.replicate size check

def rawPositionChecks (d : Graph) (c : Certificate) : List Bool :=
  let sortedIndices :=
    sortedBlock d.EDGES.length (nodupB d) ++
    sortedBlock d.EDGES.length (oneEdgeB d) ++
    sortedBlock d.NODES.length (levelFormulaUniqueB d)
  let witnessCoverage := d.EDGES.map (witnessCoverageAtB c)
  let structuralEdges := d.EDGES.map (structuralEdgeAtB d)
  let witnessFaithfulness := c.edgeRoutes.map (witnessFaithfulAtB c)
  let nodes := d.NODES.flatMap fun v =>
    [structuralNodeAtB d v,
     hypSourceAtB d v,
     productionAtB c v,
     coverageAtB c v,
     tailClosureAtB c v,
     rootDischargeAtB c v]
  sortedIndices ++ witnessCoverage ++ structuralEdges ++
    witnessFaithfulness ++ nodes

/-- Boolean outcomes in the same family order as the exported position list.

The sorted indices are computed deterministically by the exporter, so every
position in such a block carries the corresponding reflected invariant. -/
def positionChecks (d : Graph) (c : Certificate) : List Bool :=
  (rawPositionChecks d c).take (numPositions d c)

/-! `ExpandedCert` is also a finite conjunction.  Keep its five classes as
separate searchable positions rather than folding them into one global bit. -/

/-- Search positions for graph identity, edge witness coverage, witness scope,
root terminal coverage, and terminal-output scope, in that order. -/
def expandedPositionChecks (d : Graph) (c : Certificate) : List Bool :=
  [decide (c.graph = d)] ++
  d.EDGES.map (witnessCoverageAtB c) ++
  c.edgeRoutes.map (fun witness => d.EDGES.contains witness.edge) ++
  d.NODES.map (fun vertex =>
    if (get_rule.outgoing vertex d).isEmpty then
      !(c.terminalOutputs.filter fun output => output.node == vertex).isEmpty
    else true) ++
  c.terminalOutputs.map (fun output =>
    d.NODES.contains output.node &&
      (get_rule.outgoing output.node d).isEmpty)

/-- Number of searchable `ExpandedCert` component positions. -/
def expandedNumPositions (d : Graph) (c : Certificate) : Nat :=
  (expandedPositionChecks d c).length

/-- Total number of formation positions, including `ExpandedCert` and the
existing proof-specific local checks. -/
def totalNumPositions (d : Graph) (c : Certificate) : Nat :=
  expandedNumPositions d c + numPositions d c

/-- Full certificate formation validity, including scoping/coverage. -/
def FormationValid (d : Graph) (c : Certificate) : Prop :=
  Certificate.Flow.ExpandedCert d c ∧ Certificate.Flow.LocalChecks d c

theorem expandedPositionChecks_all (d : Graph) (c : Certificate) :
    (expandedPositionChecks d c).all id =
      Certificate.Flow.expandedCertB d c := by
  unfold expandedPositionChecks Certificate.Flow.expandedCertB
  simp only [List.all_append, List.all_cons, List.all_nil, id_eq,
    List.all_map, Bool.and_true]
  unfold Certificate.Flow.Foundation.witnessCoverageAtB
  simp [Bool.and_assoc]

/-- Exported finite list of executable position outcomes. -/
def exportedPositions (d : Graph) (c : Certificate) : List Bool :=
  positionChecks d c

theorem positionChecks_length (d : Graph) (c : Certificate) :
    (positionChecks d c).length = numPositions d c := by
  unfold positionChecks rawPositionChecks numPositions
  simp [sortedBlock]
  omega

/-- The proposition asserting that the cited bound is the exported-list size. -/
def ExportedPositionCount (d : Graph) (c : Certificate) : Prop :=
  numPositions d c = (exportedPositions d c).length

theorem exportedPositionCount (d : Graph) (c : Certificate) :
    ExportedPositionCount d c := by
  simpa [ExportedPositionCount, exportedPositions] using
    (positionChecks_length d c).symm

theorem numPositions_eq_positions_length (d : Graph) (c : Certificate) :
    numPositions d c = (exportedPositions d c).length :=
  exportedPositionCount d c

theorem positionChecks_eq_raw (d : Graph) (c : Certificate)
    (hedges : 0 < d.EDGES.length) (hnodes : 0 < d.NODES.length) :
    positionChecks d c = rawPositionChecks d c := by
  unfold positionChecks
  apply List.take_of_length_le
  unfold rawPositionChecks numPositions
  simp [sortedBlock]
  omega

def listedCheckPositionB (d : Graph) (c : Certificate) (i : Nat) : Bool :=
  (positionChecks d c)[i]?.getD true

/-- Indexed executable position predicate. Out-of-range indices are padding
positions and pass. The empty-edge/node branch preserves the same reflected
predicate for degenerate finite graphs, where the three `size - 1` index
blocks overlap under truncated natural-number subtraction. -/
def checkPositionB (d : Graph) (c : Certificate) (i : Nat) : Bool :=
  if d.EDGES.isEmpty || d.NODES.isEmpty then
    Certificate.Flow.localChecksB d c
  else
    listedCheckPositionB d c i

/-- A position is marked exactly when its local check fails. -/
def markedAt (d : Graph) (c : Certificate) (i : Nat) : Bool :=
  !checkPositionB d c i

/-- Propositional counterpart of `markedAt`. -/
def MarkedAt (d : Graph) (c : Certificate) (i : Nat) : Prop :=
  checkPositionB d c i = false

/-- Total formation oracle.  Expanded-certificate component positions form
the prefix; existing local-check positions retain their original indices in
the suffix. -/
def totalCheckPositionB (d : Graph) (c : Certificate)
    (position : Nat) : Bool :=
  if inExpanded : position < expandedNumPositions d c then
    (expandedPositionChecks d c)[position]
  else
    checkPositionB d c (position - expandedNumPositions d c)

/-- A failed total formation position. -/
def TotalMarkedAt (d : Graph) (c : Certificate) (position : Nat) : Prop :=
  totalCheckPositionB d c position = false

theorem markedAt_reflect (d : Graph) (c : Certificate) (i : Nat) :
    markedAt d c i = true ↔ MarkedAt d c i := by
  simp [markedAt, MarkedAt]

private theorem setEqB_complete {α : Type} [BEq α] [LawfulBEq α]
    {xs ys : List α} (h : Certificate.Flow.ListSetEq xs ys) :
    setEqB xs ys = true := by
  simp only [setEqB, Bool.and_eq_true, List.all_eq_true,
    List.contains_eq_mem]
  exact ⟨fun x hx => by simpa using h x |>.mp hx,
    fun y hy => by simpa using h y |>.mpr hy⟩

private theorem coverageAtB_complete (d : Graph) (c : Certificate)
    (hexp : Certificate.Flow.ExpandedCert d c)
    (hcoverage : Certificate.Flow.LocalCoverage c) {v : Vertex}
    (hv : v ∈ d.NODES) : coverageAtB c v = true := by
  have hvc : v ∈ c.graph.NODES := by simpa [hexp.1] using hv
  unfold coverageAtB
  simp only [Bool.and_eq_true, List.all_eq_true]
  constructor
  · intro edge hedge
    have hlocal := hcoverage v hvc edge hedge
    constructor
    · cases hs : edgeWitnessesFor c edge with
      | nil => exact (hlocal.1 hs).elim
      | cons witness rest => simp []
    · simpa [Certificate.Flow.StoredRoutes] using setEqB_complete hlocal.2
  · intro witness hw
    rw [List.mem_filter] at hw
    have hwGraph : witness.edge ∈ d.EDGES := hexp.2.2.1 witness hw.1
    have hstart : witness.edge.START = v := by
      simpa using hw.2
    simpa using mem_outgoing_of_mem_edges_start_eq v c.graph
      (by simpa [hexp.1] using hwGraph) hstart

private theorem structuralValidB_complete (d : Graph)
    (h : structuralValid d) : structuralValidB d = true := by
  classical
  simp only [structuralValidB, Bool.and_eq_true, List.all_eq_true]
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro edge hedge
    simpa using h.1 edge hedge
  · intro edge hedge
    have endpoints := h.2.1 edge hedge
    simpa using endpoints
  · intro v hv
    by_cases hin : get_rule.incoming v d = []
    · simp [hin]
    by_cases hout : get_rule.outgoing v d = []
    · simp [hout]
    obtain helim | hintro := h.2.2 v hv hin hout
    · obtain ⟨hhyp, hpreds, hpairs⟩ := helim
      have he : nodeElimShapeWideB d v = true := by
        simp [nodeElimShapeWideB, hhyp, hpreds, hpairs]
      simp [he]
    · obtain ⟨premise, alpha, beta, hhyp, hincoming, hformula,
        hpremise, hpairs⟩ := hintro
      have hi : nodeIntroShapeB d v = true := by
        simp [nodeIntroShapeB, hhyp, hincoming, hformula, hpremise, hpairs]
      simp [hi]

private theorem oneEdgeB_complete (d : Graph)
    (h : OneEdgePerColourPerNode d) : oneEdgeB d = true := by
  classical
  simp only [oneEdgeB, List.all_eq_true]
  intro e₁ he₁ e₂ he₂
  by_cases hs : e₁.START = e₂.START
  · by_cases hc : e₁.COLOUR = e₂.COLOUR
    · simp [h e₁ e₂ he₁ he₂ ⟨hs, hc⟩]
    · simp [hs, hc]
  · simp [hs]

private theorem elimPremiseAtB_complete (d : Graph) (w u : Vertex)
    (h : ElimPremiseAt d w u) : elimPremiseAtB d w u = true := by
  obtain ⟨major, minor, hpair, hside⟩ := h
  unfold elimPremiseAtB
  rw [List.any_eq_true]
  refine ⟨(major, minor), hpair, ?_⟩
  rcases hside with rfl | rfl <;> simp

private theorem certificateCarrierAccountedAtB_complete
    (c : Certificate) (w u : Vertex)
    (h : Certificate.Flow.LocalCarrierAccountedAt c w u) :
    certificateCarrierAccountedAtB c w u = true := by
  obtain ⟨alpha, hformula, haccounted⟩ := h
  unfold certificateCarrierAccountedAtB
  rw [hformula]
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true]
  refine ⟨True.intro, ?_⟩
  intro tail htail
  rw [List.any_eq_true]
  obtain ⟨dependency, hroute⟩ := haccounted tail htail
  exact ⟨(dependency, tail), hroute, by simp⟩

private theorem localChecksB_complete (d : Graph) (c : Certificate)
    (hexp : Certificate.Flow.ExpandedCert d c)
    (h : Certificate.Flow.LocalChecks d c) :
    Certificate.Flow.localChecksB d c = true := by
  unfold Certificate.Flow.localChecksB
  simp only [Bool.and_eq_true]
  refine ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩,
    ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · exact decide_eq_true h.1
  · rw [List.all_eq_true]
    intro v hv
    exact h.2.1 v (by simpa [h.1] using hv)
  · rw [List.all_eq_true]
    intro v hv
    exact coverageAtB_complete d c hexp h.2.2.1 hv
  · rw [List.all_eq_true]
    intro v hv
    unfold Certificate.Flow.Foundation.tailClosureAtB
    cases hpairs : elimPairsAt c.graph v with
    | nil => rfl
    | cons pair pairs =>
      have htail := h.2.2.2.1 v (by simpa [h.1] using hv)
        (by simp [hpairs])
      simp only [Bool.and_eq_true, List.all_eq_true]
      refine ⟨?_, ?_⟩
      · intro u hu
        rcases htail.1 u hu with helim | hcarrier
        · rw [Bool.or_eq_true]
          exact Or.inl (elimPremiseAtB_complete c.graph v u helim)
        · rw [Bool.or_eq_true]
          exact Or.inr (certificateCarrierAccountedAtB_complete c v u hcarrier)
      · intro pair hpair
        rcases pair with ⟨major, minor⟩
        exact htail.2 major minor (by simpa [hpairs] using hpair)
  · rw [List.all_eq_true]
    intro v hv
    by_cases hterminal : (get_rule.outgoing v c.graph).isEmpty = true
    · obtain ⟨⟨output, hout, hnode, houtdep, houtres⟩, hshape, hdep⟩ :=
        h.2.2.2.2.1 v (by simpa [h.1] using hv) hterminal
      simp [rootDischargeAtB, hterminal, List.any_eq_true]
      exact ⟨output, hout, hnode, by simp [houtdep, houtres,
        h.2.1 v (by simpa [h.1] using hv), hshape, hdep]⟩
    · have hne : (get_rule.outgoing v c.graph).isEmpty = false := by
        exact Bool.eq_false_of_not_eq_true hterminal
      simp [rootDischargeAtB, hne]
  · exact structuralValidB_complete d h.2.2.2.2.2.1
  · simpa [nodupB] using h.2.2.2.2.2.2.1
  · exact oneEdgeB_complete d h.2.2.2.2.2.2.2.1
  · rw [List.all_eq_true]
    intro witness hwitness
    exact h.2.2.2.2.2.2.2.2.1 witness hwitness
  · unfold levelFormulaUniqueB
    rw [List.all_eq_true]
    intro v₁ hv₁
    rw [List.all_eq_true]
    intro v₂ hv₂
    by_cases hl : v₁.LEVEL = v₂.LEVEL
    · by_cases hf : v₁.FORMULA = v₂.FORMULA
      · simp [h.2.2.2.2.2.2.2.2.2.1 v₁ hv₁ v₂ hv₂ hl hf]
      · simp [hl, hf]
    · simp [hl]
  · unfold hypNoIncomingB
    rw [List.all_eq_true]
    intro v hv
    by_cases hh : v.HYPOTHESIS = true
    · simpa [hh] using h.2.2.2.2.2.2.2.2.2.2 v hv hh
    · simp [hh]

private theorem all_positionChecks_iff_localChecksB
    (d : Graph) (c : Certificate)
    (hexp : Certificate.Flow.ExpandedCert d c)
    (hedges : 0 < d.EDGES.length) (hnodes : 0 < d.NODES.length) :
    (positionChecks d c).all id = true ↔
      Certificate.Flow.localChecksB d c = true := by
  rw [positionChecks_eq_raw d c hedges hnodes]
  have hedgesne : d.EDGES ≠ [] := List.ne_nil_of_length_pos hedges
  have hnodesne : d.NODES ≠ [] := List.ne_nil_of_length_pos hnodes
  constructor
  · intro hall
    simp [rawPositionChecks, sortedBlock, hedgesne, hnodesne] at hall
    rcases hall with
      ⟨hnodup, hone, hlevel, _hwitness, hstructural,
        hfaithful, hnode⟩
    have hstruct : structuralValidB d = true := by
      simp only [structuralValidB, Bool.and_eq_true, List.all_eq_true]
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · intro edge hedge
        have hs := hstructural edge hedge
        simp [structuralEdgeAtB] at hs
        simpa using hs.1.1
      · intro edge hedge
        have hs := hstructural edge hedge
        simp [structuralEdgeAtB] at hs
        exact ⟨by simpa using hs.1.2, by simpa using hs.2⟩
      · intro v hv
        simpa [structuralNodeAtB] using (hnode v hv).1
    have hhyp : hypNoIncomingB d = true := by
      unfold hypNoIncomingB
      rw [List.all_eq_true]
      exact fun v hv => (hnode v hv).2.1
    unfold Certificate.Flow.localChecksB
    simp only [Bool.and_eq_true, List.all_eq_true]
    exact ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨decide_eq_true hexp.1,
      fun v hv => (hnode v hv).2.2.1⟩,
      fun v hv => (hnode v hv).2.2.2.1⟩,
      fun v hv => (hnode v hv).2.2.2.2.1⟩,
      fun v hv => (hnode v hv).2.2.2.2.2⟩,
      hstruct⟩, hnodup⟩, hone⟩, hfaithful⟩, hlevel⟩, hhyp⟩
  · intro hlocal
    have hprop := Certificate.Flow.localChecksB_sound d c hexp hlocal
    have hcomplete := localChecksB_complete d c hexp hprop
    simp only [Certificate.Flow.localChecksB, Bool.and_eq_true,
      List.all_eq_true] at hcomplete
    rcases hcomplete with ⟨hleft, hhyp⟩
    rcases hleft with ⟨hleft, hlevel⟩
    rcases hleft with ⟨hleft, hfaithful⟩
    rcases hleft with ⟨hleft, hone⟩
    rcases hleft with ⟨hleft, hnodup⟩
    rcases hleft with ⟨hleft, hstruct⟩
    rcases hleft with ⟨hleft, hroot⟩
    rcases hleft with ⟨hleft, htail⟩
    rcases hleft with ⟨hleft, hcoverage⟩
    rcases hleft with ⟨_hgraph, hproduction⟩
    have hwitness : ∀ edge ∈ d.EDGES, witnessCoverageAtB c edge = true := by
      intro edge hedge
      have hlen := hexp.2.1 edge hedge
      unfold witnessCoverageAtB
      cases hs : edgeWitnessesFor c edge with
      | nil => simp [hs] at hlen
      | cons witness rest => simp []
    have hstructural : ∀ edge ∈ d.EDGES,
        structuralEdgeAtB d edge = true := by
      simp [structuralValidB, List.all_eq_true] at hstruct
      intro edge hedge
      have hlevel := hstruct.1.1 edge hedge
      have hendpoints := hstruct.1.2 edge hedge
      simp [structuralEdgeAtB, hlevel, hendpoints.1, hendpoints.2]
    have hstructuralNode : ∀ v ∈ d.NODES,
        structuralNodeAtB d v = true := by
      simp [structuralValidB, List.all_eq_true] at hstruct
      intro v hv
      simpa [structuralNodeAtB] using hstruct.2 v hv
    have hfaithful' : ∀ witness ∈ c.edgeRoutes,
        witnessFaithfulAtB c witness = true := by
      simpa [List.all_eq_true] using hfaithful
    have hhyp' : ∀ v ∈ d.NODES, hypSourceAtB d v = true := by
      simpa [hypNoIncomingB, hypSourceAtB, List.all_eq_true] using hhyp
    simp [rawPositionChecks, sortedBlock, hedgesne, hnodesne]
    exact ⟨hnodup, hone, hlevel, hwitness, hstructural, hfaithful',
      fun v hv => ⟨hstructuralNode v hv, hhyp' v hv,
        hproduction v hv, hcoverage v hv, htail v hv, hroot v hv⟩⟩

private theorem no_marked_iff_all_positionChecks
    (d : Graph) (c : Certificate)
    (hedges : 0 < d.EDGES.length) (hnodes : 0 < d.NODES.length) :
    (∀ i < numPositions d c, ¬ MarkedAt d c i) ↔
      (positionChecks d c).all id = true := by
  have hedgeEmpty : d.EDGES.isEmpty = false := by
    simp [List.ne_nil_of_length_pos hedges]
  have hnodeEmpty : d.NODES.isEmpty = false := by
    simp [List.ne_nil_of_length_pos hnodes]
  rw [List.all_eq_true, List.forall_mem_iff_getElem]
  constructor
  · intro h i hi
    cases hb : (positionChecks d c)[i] with
    | false =>
        exfalso
        apply h i (by simpa [positionChecks_length] using hi)
        simp [MarkedAt, checkPositionB, listedCheckPositionB, hedgeEmpty,
          hnodeEmpty, hi, hb]
    | true => rfl
  · intro h i hi hmarked
    have hi' : i < (positionChecks d c).length := by
      simpa [positionChecks_length] using hi
    have hpass := h i hi'
    have hpass' : (positionChecks d c)[i] = true := by simpa using hpass
    have hmarked' : (positionChecks d c)[i] = false := by
      simpa [MarkedAt, checkPositionB, listedCheckPositionB, hedgeEmpty,
        hnodeEmpty, hi'] using hmarked
    exact Bool.noConfusion (hpass'.symm.trans hmarked')

/-- All queried positions pass exactly when all semantic local checks hold.

`ExpandedCert` supplies the graph/certificate identity and total witness
coverage that are preconditions of the local-check soundness theorem. -/
theorem no_marked_iff_localChecks (d : Graph) (c : Certificate)
    (hexp : Certificate.Flow.ExpandedCert d c) :
    (∀ i < numPositions d c, ¬ MarkedAt d c i) ↔
      Certificate.Flow.LocalChecks d c := by
  have hboolProp : Certificate.Flow.localChecksB d c = true ↔
      Certificate.Flow.LocalChecks d c :=
    ⟨Certificate.Flow.localChecksB_sound d c hexp,
      localChecksB_complete d c hexp⟩
  by_cases hedges : 0 < d.EDGES.length
  · by_cases hnodes : 0 < d.NODES.length
    · exact (no_marked_iff_all_positionChecks d c hedges hnodes).trans
        ((all_positionChecks_iff_localChecksB d c hexp hedges hnodes).trans
          hboolProp)
    · have hnodes0 : d.NODES = [] := by
        exact List.eq_nil_of_length_eq_zero (Nat.eq_zero_of_not_pos hnodes)
      have hdeg : d.EDGES.isEmpty || d.NODES.isEmpty = true := by
        simp [hnodes0]
      have hnpos : 0 < numPositions d c := by
        unfold numPositions
        simp [hnodes0] at ⊢
        omega
      constructor
      · intro h
        apply hboolProp.mp
        by_contra hfalse
        have hzero := h 0 hnpos
        apply hzero
        simp [MarkedAt, checkPositionB, hnodes0,
          Bool.eq_false_of_not_eq_true hfalse]
      · intro hloc i hi hmarked
        have htrue := hboolProp.mpr hloc
        simp [MarkedAt, checkPositionB, hnodes0, htrue] at hmarked
  · have hedges0 : d.EDGES = [] := by
      exact List.eq_nil_of_length_eq_zero (Nat.eq_zero_of_not_pos hedges)
    have hdeg : d.EDGES.isEmpty || d.NODES.isEmpty = true := by
      simp [hedges0]
    by_cases hnpos : 0 < numPositions d c
    · constructor
      · intro h
        apply hboolProp.mp
        by_contra hfalse
        have hzero := h 0 hnpos
        apply hzero
        simp [MarkedAt, checkPositionB, hedges0,
          Bool.eq_false_of_not_eq_true hfalse]
      · intro hloc i hi hmarked
        have htrue := hboolProp.mpr hloc
        simp [MarkedAt, checkPositionB, hedges0, htrue] at hmarked
    · have hnzero : d.NODES = [] := by
        have hlength : d.NODES.length = 0 := by
          unfold numPositions at hnpos
          simp [hedges0] at hnpos
          omega
        exact List.eq_nil_of_length_eq_zero hlength
      have hlocalB : Certificate.Flow.localChecksB d c = true := by
        have hwzero : c.edgeRoutes = [] := by
          apply List.eq_nil_iff_forall_not_mem.mpr
          intro witness hwitness
          have hedge := hexp.2.2.1 witness hwitness
          simp [hedges0] at hedge
        simp [Certificate.Flow.localChecksB, hexp.1, hedges0, hnzero, hwzero,
          structuralValidB, nodupB, oneEdgeB,
          levelFormulaUniqueB, hypNoIncomingB,
          get_rule.incoming, get_rule.incoming.loop]
      constructor
      · intro _
        exact hboolProp.mp hlocalB
      · intro _ i hi
        have hn : numPositions d c = 0 := Nat.eq_zero_of_not_pos hnpos
        rw [hn] at hi
        exact (Nat.not_lt_zero i hi).elim

theorem rootDischarged_of_localChecks (d : Graph) (c : Certificate)
    (hloc : Certificate.Flow.LocalChecks d c) : RootDischarged d := by
  intro v hv hroot
  have hroot' : (get_rule.outgoing v c.graph).isEmpty = true := by
    simp [hloc.1, hroot]
  have h := hloc.2.2.2.2.1 v (by simpa [hloc.1] using hv) hroot'
  exact ⟨by simpa [hloc.1] using h.2.1, by simpa [hloc.1] using h.2.2⟩

/-- Search correctness, accepting direction: no marked position yields the
retained formation invariants, including semantic root discharge.  The newer
occurrence-aware `LocalTailClosure` is deliberately not strengthened here to
the older equality-only `WitnessableFlowTailClosure`. -/
theorem no_marked_implies_valid (d : Graph) (c : Certificate) :
    Certificate.Flow.ExpandedCert d c →
    (∀ i < numPositions d c, ¬ MarkedAt d c i) →
    (structuralValid d ∧ d.EDGES.Nodup ∧ OneEdgePerColourPerNode d ∧
      Certificate.Flow.WitnessableRouteHeadCoverage d ∧ LevelFormulaUnique d ∧
      HypothesesHaveNoIncoming d) ∧ RootDischarged d := by
  intro hexp hno
  have hloc := (no_marked_iff_localChecks d c hexp).mp hno
  exact ⟨Certificate.Flow.invariants_of_expanded_certificate d c hexp hloc,
    rootDischarged_of_localChecks d c hloc⟩

/-- Search correctness, rejecting direction: failure of the certified validity
package exposes a marked position. -/
theorem invalid_implies_marked (d : Graph) (c : Certificate) :
    Certificate.Flow.ExpandedCert d c →
    ¬ ((structuralValid d ∧ d.EDGES.Nodup ∧ OneEdgePerColourPerNode d ∧
      Certificate.Flow.WitnessableRouteHeadCoverage d ∧ LevelFormulaUnique d ∧
      HypothesesHaveNoIncoming d) ∧ RootDischarged d) →
    ∃ i < numPositions d c, MarkedAt d c i := by
  intro hexp hinvalid
  by_contra hnone
  apply hinvalid
  apply no_marked_implies_valid d c hexp
  intro i hi hmarked
  exact hnone ⟨i, hi, hmarked⟩

/-- Exact rejecting direction for the current certificate-formation predicate.
Under the scoping/coverage assumptions in `ExpandedCert`, failure of
`LocalChecks` is witnessed by one of the proof-specific local positions. -/
theorem formation_invalid_implies_marked (d : Graph) (c : Certificate)
    (hexp : Certificate.Flow.ExpandedCert d c)
    (hinvalid : ¬ Certificate.Flow.LocalChecks d c) :
    ∃ i < numPositions d c, MarkedAt d c i := by
  by_contra hnone
  apply hinvalid
  apply (no_marked_iff_localChecks d c hexp).mp
  intro i hi hmarked
  exact hnone ⟨i, hi, hmarked⟩

private lemma exists_false_position {checks : List Bool}
    (someFailure : checks.all id = false) :
    ∃ position, ∃ inBounds : position < checks.length,
      checks[position] = false := by
  by_contra noFailure
  have allTrue : checks.all id = true := by
    rw [List.all_eq_true, List.forall_mem_iff_getElem]
    intro position inBounds
    cases valueEq : checks[position] with
    | false =>
        exact False.elim <| noFailure
          ⟨position, inBounds, valueEq⟩
    | true => rfl
  simp_all

/-- Every failure of the complete formation predicate has a searchable failed
position.  Unlike `formation_invalid_implies_marked`, this theorem has no
`ExpandedCert` precondition: failures of that finite conjunction occupy the
prefix of the total position space. -/
theorem total_formation_invalid_implies_marked (d : Graph)
    (c : Certificate) (invalid : ¬ FormationValid d c) :
    ∃ position, position < totalNumPositions d c ∧
      TotalMarkedAt d c position := by
  by_cases expanded : Certificate.Flow.ExpandedCert d c
  · have localInvalid : ¬ Certificate.Flow.LocalChecks d c := by
      intro hlocal
      apply invalid
      exact ⟨expanded, hlocal⟩
    obtain ⟨position, position_lt, marked⟩ :=
      formation_invalid_implies_marked d c expanded localInvalid
    refine ⟨expandedNumPositions d c + position, ?_, ?_⟩
    · simp [totalNumPositions]
      omega
    · have notExpanded :
          ¬ expandedNumPositions d c + position <
            expandedNumPositions d c := by omega
      unfold TotalMarkedAt totalCheckPositionB
      rw [dif_neg notExpanded]
      simpa [MarkedAt] using marked
  · have expandedB : Certificate.Flow.expandedCertB d c = false := by
      cases valueEq : Certificate.Flow.expandedCertB d c with
      | false => rfl
      | true =>
          exact False.elim <| expanded
            (Certificate.Flow.expandedCertB_sound d c valueEq)
    have allFalse : (expandedPositionChecks d c).all id = false := by
      rw [expandedPositionChecks_all]
      exact expandedB
    obtain ⟨position, position_lt, failed⟩ :=
      exists_false_position allFalse
    refine ⟨position, ?_, ?_⟩
    · simp [totalNumPositions, expandedNumPositions]
      omega
    · unfold TotalMarkedAt totalCheckPositionB
      rw [dif_pos (by simpa [expandedNumPositions] using position_lt)]
      exact failed

def markedCount (d : Graph) (c : Certificate) : Nat :=
  (exportedPositions d c).count false

def fixtureReports : List (String × Nat × Nat) :=
  Certificate.Encoding.Export.fixtures.map fun fixture =>
    (fixture.1, numPositions fixture.2.graph fixture.2,
      markedCount fixture.2.graph fixture.2)

/-- Executable EndToEnd-pattern report: fixture name, `n`, and marked count. -/
theorem fixtureReports_native : fixtureReports =
    [("3v-no-hc", 1607, 1),
     ("endtoend-valid", 337, 0),
     ("d11-valid", 262, 1),
     ("endtoend-altered-residual", 337, 3),
     ("endtoend-omitted-witness", 336, 5),
     ("endtoend-unfaithful-decoration", 337, 4),
     ("endtoend-nonempty-terminal", 337, 2)] := by
  native_decide

end Search.Certificate
