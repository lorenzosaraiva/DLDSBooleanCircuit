import Robustness.Certificate.Support
import Certificate.Flow.Positions

/-!
# Canonical finite encoding of verifier-visible certificate data

The certificate checker reads an embedded graph, route witnesses, and terminal
outputs.  The types behind those fields contain unbounded naturals, formulas,
and lists, so there is no single finite type containing every Lean
`ExpandedCertificate`.  For an anchor certificate, however, its record and
list shapes and the scalar values occurring in it are finite.

This file gives the corresponding canonical *bounded field encoding*.  Each
coordinate is one scalar field or one entry of a verifier-visible list.  The
outer record counts are bounded by the anchor and have explicit presence
coordinates; each inner list has its anchor length plus one explicit spare
slot.  An alphabet value is one tagged scalar value occurring in these slots
(plus a reserved Boolean value, which also makes the alphabet nonempty).
Thus the decoder represents fieldwise substitutions, record omission, and one
extra dependency/residual/metadata entry within those explicit bounds, and it
exactly round-trips every anchor certificate.
-/

open DLDS FlowSemantics Certificate.Flow.Foundation

namespace CertificateSearch
namespace Robustness

inductive CanonicalDeductionOccurrence where
  | graphEdge (index : Nat)
  | routeWitnessEdge (index : Nat)
  deriving Repr, DecidableEq

inductive CanonicalVertexOccurrence where
  | graphNode (index : Nat)
  | deductionStart (occurrence : CanonicalDeductionOccurrence)
  | deductionEnd (occurrence : CanonicalDeductionOccurrence)
  | ancestralStart (index : Nat)
  | ancestralEnd (index : Nat)
  | terminalNode (index : Nat)
  deriving Repr, DecidableEq

/-- A field address in precisely the data traversed by the certificate
verifier.  Formula values are kept as formula-valued scalars; naturals and
Booleans are scalar fields.  Every entry of every variable-width list gets its
own coordinate. -/
inductive CanonicalRawCoord where
  | graphNodePresent (index : Nat)
  | graphEdgePresent (index : Nat)
  | ancestralPresent (index : Nat)
  | witnessPresent (index : Nat)
  | terminalPresent (index : Nat)
  | vertexNumber (vertex : CanonicalVertexOccurrence)
  | vertexLevel (vertex : CanonicalVertexOccurrence)
  | vertexFormula (vertex : CanonicalVertexOccurrence)
  | vertexHypothesis (vertex : CanonicalVertexOccurrence)
  | vertexCollapsed (vertex : CanonicalVertexOccurrence)
  | vertexPastPresent (vertex : CanonicalVertexOccurrence) (index : Nat)
  | vertexPast (vertex : CanonicalVertexOccurrence) (index : Nat)
  | deductionColour (edge : CanonicalDeductionOccurrence)
  | deductionDependencyPresent (edge : CanonicalDeductionOccurrence)
      (index : Nat)
  | deductionDependency (edge : CanonicalDeductionOccurrence) (index : Nat)
  | ancestralColourPresent (pathIndex entryIndex : Nat)
  | ancestralColour (pathIndex entryIndex : Nat)
  | witnessDependencyPresent (witnessIndex entryIndex : Nat)
  | witnessDependency (witnessIndex entryIndex : Nat)
  | witnessResidualPresent (witnessIndex entryIndex : Nat)
  | witnessResidual (witnessIndex entryIndex : Nat)
  | terminalDependencyPresent (terminalIndex entryIndex : Nat)
  | terminalDependency (terminalIndex entryIndex : Nat)
  | terminalResidualPresent (terminalIndex entryIndex : Nat)
  | terminalResidual (terminalIndex entryIndex : Nat)
  deriving Repr, DecidableEq

inductive CanonicalAtom where
  | natural (value : Nat)
  | formula (value : Formula)
  | flag (value : Bool)
  deriving Repr, DecidableEq

/-- Coarse ownership of a verifier-visible field.  This classification is
syntactic: it is computed from the coordinate constructor, not by testing the
checker on all assignments. -/
inductive CanonicalFieldRegion where
  | graph
  | witness
  | terminal
  deriving Repr, DecidableEq

private def deductionOccurrenceRegion : CanonicalDeductionOccurrence →
    CanonicalFieldRegion
  | .graphEdge _ => .graph
  | .routeWitnessEdge _ => .witness

private def vertexOccurrenceRegion : CanonicalVertexOccurrence →
    CanonicalFieldRegion
  | .graphNode _ => .graph
  | .deductionStart occurrence => deductionOccurrenceRegion occurrence
  | .deductionEnd occurrence => deductionOccurrenceRegion occurrence
  | .ancestralStart _ => .graph
  | .ancestralEnd _ => .graph
  | .terminalNode _ => .terminal

/-- Region ownership can be read directly from an address. -/
def canonicalRawCoordRegion : CanonicalRawCoord → CanonicalFieldRegion
  | .graphNodePresent _ => .graph
  | .graphEdgePresent _ => .graph
  | .ancestralPresent _ => .graph
  | .witnessPresent _ => .witness
  | .terminalPresent _ => .terminal
  | .vertexNumber occurrence => vertexOccurrenceRegion occurrence
  | .vertexLevel occurrence => vertexOccurrenceRegion occurrence
  | .vertexFormula occurrence => vertexOccurrenceRegion occurrence
  | .vertexHypothesis occurrence => vertexOccurrenceRegion occurrence
  | .vertexCollapsed occurrence => vertexOccurrenceRegion occurrence
  | .vertexPastPresent occurrence _ => vertexOccurrenceRegion occurrence
  | .vertexPast occurrence _ => vertexOccurrenceRegion occurrence
  | .deductionColour occurrence => deductionOccurrenceRegion occurrence
  | .deductionDependencyPresent occurrence _ =>
      deductionOccurrenceRegion occurrence
  | .deductionDependency occurrence _ => deductionOccurrenceRegion occurrence
  | .ancestralColourPresent _ _ => .graph
  | .ancestralColour _ _ => .graph
  | .witnessDependencyPresent _ _ => .witness
  | .witnessDependency _ _ => .witness
  | .witnessResidualPresent _ _ => .witness
  | .witnessResidual _ _ => .witness
  | .terminalDependencyPresent _ _ => .terminal
  | .terminalDependency _ _ => .terminal
  | .terminalResidualPresent _ _ => .terminal
  | .terminalResidual _ _ => .terminal

abbrev CanonicalSlot := CanonicalRawCoord × CanonicalAtom

/-- One explicit spare slot gives each inner verifier-visible list an anchor
bound of `length + 1`.  Existing entries are active; the spare entry is
inactive at the anchor and can represent the corruption fixtures which append
one dependency or residual item. -/
def boundedEntries {α : Type} (values : List α) (fallback : α) :
    List (Bool × α) :=
  values.map (fun value => (true, value)) ++ [(false, fallback)]

@[simp] theorem boundedEntries_length {α : Type}
    (values : List α) (fallback : α) :
    (boundedEntries values fallback).length = values.length + 1 := by
  simp [boundedEntries]

theorem filterMap_zipIdx_presence {α : Type}
    (entries : List (Bool × α)) (start : Nat := 0) :
    (entries.zipIdx start).filterMap
        (fun entry => if entry.1.1 = true then some entry.1.2 else none) =
      entries.filterMap
        (fun entry => if entry.1 = true then some entry.2 else none) := by
  induction entries generalizing start with
  | nil => rfl
  | cons entry rest induction =>
      rcases entry with ⟨present, value⟩
      cases present <;> simp [List.zipIdx, induction (start + 1)]

@[simp] theorem filterMap_boundedEntries_zipIdx {α : Type}
    (values : List α) (fallback : α) (start : Nat := 0) :
    ((boundedEntries values fallback).zipIdx start).filterMap
        (fun entry => if entry.1.1 = true then some entry.1.2 else none) =
      values := by
  rw [filterMap_zipIdx_presence (start := start)]
  simp [boundedEntries]

def vertexSlots (occurrence : CanonicalVertexOccurrence)
    (vertex : Vertex) : List CanonicalSlot :=
  [(.vertexNumber occurrence, .natural vertex.NUMBER),
   (.vertexLevel occurrence, .natural vertex.LEVEL),
   (.vertexFormula occurrence, .formula vertex.FORMULA),
   (.vertexHypothesis occurrence, .flag vertex.HYPOTHESIS),
   (.vertexCollapsed occurrence, .flag vertex.COLLAPSED)] ++
  (boundedEntries vertex.PAST 0).zipIdx.flatMap fun (entry, index) =>
    [(.vertexPastPresent occurrence index, .flag entry.1),
     (.vertexPast occurrence index, .natural entry.2)]

def deductionSlots (occurrence : CanonicalDeductionOccurrence)
    (edge : Deduction) : List CanonicalSlot :=
  vertexSlots (.deductionStart occurrence) edge.START ++
  vertexSlots (.deductionEnd occurrence) edge.END ++
  [(.deductionColour occurrence, .natural edge.COLOUR)] ++
  (boundedEntries edge.DEPENDENCY (#"BAD")).zipIdx.flatMap fun (entry, index) =>
    [(.deductionDependencyPresent occurrence index, .flag entry.1),
     (.deductionDependency occurrence index, .formula entry.2)]

def ancestralSlots (pathIndex : Nat) (path : Ancestral) :
    List CanonicalSlot :=
  vertexSlots (.ancestralStart pathIndex) path.START ++
  vertexSlots (.ancestralEnd pathIndex) path.END ++
  (boundedEntries path.COLOURS 999).zipIdx.flatMap fun (entry, index) =>
    [(.ancestralColourPresent pathIndex index, .flag entry.1),
     (.ancestralColour pathIndex index, .natural entry.2)]

def witnessSlots (witnessIndex : Nat)
    (witness : EdgeRouteWitness) : List CanonicalSlot :=
  deductionSlots (.routeWitnessEdge witnessIndex) witness.edge ++
  (boundedEntries witness.dependency (#"BAD")).zipIdx.flatMap
      (fun (entry, index) =>
        [(.witnessDependencyPresent witnessIndex index, .flag entry.1),
         (.witnessDependency witnessIndex index, .formula entry.2)]) ++
  (boundedEntries witness.residual 999).zipIdx.flatMap fun (entry, index) =>
    [(.witnessResidualPresent witnessIndex index, .flag entry.1),
     (.witnessResidual witnessIndex index, .natural entry.2)]

def terminalSlots (terminalIndex : Nat)
    (output : TerminalOutput) : List CanonicalSlot :=
  vertexSlots (.terminalNode terminalIndex) output.node ++
  (boundedEntries output.dependency (#"BAD")).zipIdx.flatMap
      (fun (entry, index) =>
        [(.terminalDependencyPresent terminalIndex index, .flag entry.1),
         (.terminalDependency terminalIndex index, .formula entry.2)]) ++
  (boundedEntries output.residual 999).zipIdx.flatMap fun (entry, index) =>
    [(.terminalResidualPresent terminalIndex index, .flag entry.1),
     (.terminalResidual terminalIndex index, .natural entry.2)]

private theorem vertexSlots_owned (occurrence : CanonicalVertexOccurrence)
    (vertex : Vertex) (region : CanonicalFieldRegion)
    (owned : vertexOccurrenceRegion occurrence = region) :
    ∀ slot ∈ vertexSlots occurrence vertex,
      canonicalRawCoordRegion slot.1 = region := by
  intro slot member
  simp [vertexSlots, boundedEntries] at member
  aesop

private theorem deductionSlots_owned
    (occurrence : CanonicalDeductionOccurrence) (edge : Deduction)
    (region : CanonicalFieldRegion)
    (owned : deductionOccurrenceRegion occurrence = region) :
    ∀ slot ∈ deductionSlots occurrence edge,
      canonicalRawCoordRegion slot.1 = region := by
  intro slot member
  simp [deductionSlots, vertexSlots, boundedEntries] at member
  aesop

private theorem ancestralSlots_owned (pathIndex : Nat) (path : Ancestral) :
    ∀ slot ∈ ancestralSlots pathIndex path,
      canonicalRawCoordRegion slot.1 = .graph := by
  intro slot member
  simp [ancestralSlots, vertexSlots, boundedEntries] at member
  aesop

private theorem witnessSlots_owned (witnessIndex : Nat)
    (witness : EdgeRouteWitness) :
    ∀ slot ∈ witnessSlots witnessIndex witness,
      canonicalRawCoordRegion slot.1 = .witness := by
  intro slot member
  simp [witnessSlots, deductionSlots, vertexSlots, boundedEntries] at member
  aesop

private theorem terminalSlots_owned (terminalIndex : Nat)
    (output : TerminalOutput) :
    ∀ slot ∈ terminalSlots terminalIndex output,
      canonicalRawCoordRegion slot.1 = .terminal := by
  intro slot member
  simp [terminalSlots, vertexSlots, boundedEntries] at member
  aesop

/-- Graph-record portion of the canonical field table. -/
def canonicalGraphSlots (c : Certificate) : List CanonicalSlot :=
  (c.graph.NODES.zipIdx.flatMap fun (vertex, index) =>
      [(.graphNodePresent index, .flag true)] ++
        vertexSlots (.graphNode index) vertex) ++
  (c.graph.EDGES.zipIdx.flatMap fun (deduction, index) =>
      [(.graphEdgePresent index, .flag true)] ++
        deductionSlots (.graphEdge index) deduction) ++
  (c.graph.PATHS.zipIdx.flatMap fun (ancestral, index) =>
      [(.ancestralPresent index, .flag true)] ++
        ancestralSlots index ancestral)

/-- Route-witness portion of the canonical field table. -/
def canonicalWitnessSlots (c : Certificate) : List CanonicalSlot :=
  (c.edgeRoutes.zipIdx.flatMap fun (witness, index) =>
      [(.witnessPresent index, .flag true)] ++ witnessSlots index witness)

/-- Terminal-output portion of the canonical field table. -/
def canonicalTerminalSlots (c : Certificate) : List CanonicalSlot :=
  (c.terminalOutputs.zipIdx.flatMap fun (output, index) =>
      [(.terminalPresent index, .flag true)] ++ terminalSlots index output)

theorem canonicalGraphSlots_owned (c : Certificate) :
    ∀ slot ∈ canonicalGraphSlots c,
      canonicalRawCoordRegion slot.1 = .graph := by
  have nodeOwned : ∀ index vertex slot,
      slot ∈ vertexSlots (.graphNode index) vertex →
        canonicalRawCoordRegion slot.1 = .graph := by
    intro index vertex slot member
    exact vertexSlots_owned (.graphNode index) vertex .graph rfl slot member
  have edgeOwned : ∀ index edge slot,
      slot ∈ deductionSlots (.graphEdge index) edge →
        canonicalRawCoordRegion slot.1 = .graph := by
    intro index edge slot member
    exact deductionSlots_owned (.graphEdge index) edge .graph rfl slot member
  intro slot member
  simp [canonicalGraphSlots] at member
  aesop (add safe forward [nodeOwned, edgeOwned, ancestralSlots_owned])

private theorem canonicalWitnessSlots_owned (c : Certificate) :
    ∀ slot ∈ canonicalWitnessSlots c,
      canonicalRawCoordRegion slot.1 = .witness := by
  have witnessOwned : ∀ index witness slot,
      slot ∈ witnessSlots index witness →
        canonicalRawCoordRegion slot.1 = .witness := by
    intro index witness slot member
    exact witnessSlots_owned index witness slot member
  intro slot member
  simp [canonicalWitnessSlots] at member
  aesop (add safe forward witnessOwned)

private theorem canonicalTerminalSlots_owned (c : Certificate) :
    ∀ slot ∈ canonicalTerminalSlots c,
      canonicalRawCoordRegion slot.1 = .terminal := by
  have terminalOwned : ∀ index output slot,
      slot ∈ terminalSlots index output →
        canonicalRawCoordRegion slot.1 = .terminal := by
    intro index output slot member
    exact terminalSlots_owned index output slot member
  intro slot member
  simp [canonicalTerminalSlots] at member
  aesop (add safe forward terminalOwned)

/-- Complete flat field table for an anchor certificate.  The derived
permutation/index tables in `CertificateExporter` are intentionally absent:
they are recomputed views, not fields read from the certificate record by
`totalCheckPositionB`. -/
def canonicalSlots (c : Certificate) : List CanonicalSlot :=
  canonicalGraphSlots c ++ canonicalWitnessSlots c ++ canonicalTerminalSlots c

private theorem canonicalSlot_mem_graph_of_region (c : Certificate)
    (slot : CanonicalSlot) (member : slot ∈ canonicalSlots c)
    (region : canonicalRawCoordRegion slot.1 = .graph) :
    slot ∈ canonicalGraphSlots c := by
  simp only [canonicalSlots, List.mem_append] at member
  rcases member with (graphMember | witnessMember) | terminalMember
  · exact graphMember
  · have owned := canonicalWitnessSlots_owned c slot witnessMember
    rw [region] at owned
    contradiction
  · have owned := canonicalTerminalSlots_owned c slot terminalMember
    rw [region] at owned
    contradiction

private theorem canonicalSlot_mem_witness_of_region (c : Certificate)
    (slot : CanonicalSlot) (member : slot ∈ canonicalSlots c)
    (region : canonicalRawCoordRegion slot.1 = .witness) :
    slot ∈ canonicalWitnessSlots c := by
  simp only [canonicalSlots, List.mem_append] at member
  rcases member with (graphMember | witnessMember) | terminalMember
  · have owned := canonicalGraphSlots_owned c slot graphMember
    rw [region] at owned
    contradiction
  · exact witnessMember
  · have owned := canonicalTerminalSlots_owned c slot terminalMember
    rw [region] at owned
    contradiction

private theorem canonicalSlot_mem_terminal_of_region (c : Certificate)
    (slot : CanonicalSlot) (member : slot ∈ canonicalSlots c)
    (region : canonicalRawCoordRegion slot.1 = .terminal) :
    slot ∈ canonicalTerminalSlots c := by
  simp only [canonicalSlots, List.mem_append] at member
  rcases member with (graphMember | witnessMember) | terminalMember
  · have owned := canonicalGraphSlots_owned c slot graphMember
    rw [region] at owned
    contradiction
  · have owned := canonicalWitnessSlots_owned c slot witnessMember
    rw [region] at owned
    contradiction
  · exact terminalMember

def canonicalSlotSet (c : Certificate) : Finset CanonicalSlot :=
  (canonicalSlots c).toFinset

/-- Coordinates are anchor field addresses paired with their anchor value.
Keeping the anchor value in the finite key makes decoding executable without
any choice of an occurrence among equal scalar values. -/
abbrev CanonicalCoord (_d : Graph) (c : Certificate) :=
  { slot : CanonicalSlot // slot ∈ canonicalSlotSet c }

def canonicalAtomSet (c : Certificate) : Finset CanonicalAtom :=
  insert (.flag false) ((canonicalSlots c).map Prod.snd).toFinset

abbrev CanonicalAlphabet (_d : Graph) (c : Certificate) :=
  { atom : CanonicalAtom // atom ∈ canonicalAtomSet c }

/-- All canonical coordinates owned by one syntactic certificate component. -/
def canonicalRegionReadSet (d : Graph) (c : Certificate)
    (region : CanonicalFieldRegion) : Finset (CanonicalCoord d c) :=
  Finset.univ.filter fun coordinate =>
    canonicalRawCoordRegion coordinate.1.1 = region

instance canonicalAlphabetNonempty (d : Graph) (c : Certificate) :
    Nonempty (CanonicalAlphabet d c) :=
  ⟨⟨.flag false, by simp [canonicalAtomSet]⟩⟩

/-- The canonical encoding selects the anchor scalar stored at each field
coordinate. -/
def encodeCertificate (d : Graph) (c : Certificate) :
    Assignment (CanonicalCoord d c) (CanonicalAlphabet d c) :=
  fun coordinate =>
    ⟨coordinate.1.2, by
      have slotMem : coordinate.1 ∈ canonicalSlots c := by
        have member := coordinate.2
        unfold canonicalSlotSet at member
        exact List.mem_toFinset.mp member
      simp only [canonicalAtomSet, Finset.mem_insert, List.mem_toFinset]
      exact Or.inr (List.mem_map.mpr
        ⟨coordinate.1, slotMem, rfl⟩)⟩

def readCanonicalAtom (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (coordinate : CanonicalRawCoord) (fallback : CanonicalAtom) :
    CanonicalAtom :=
  if member : (coordinate, fallback) ∈ canonicalSlotSet c then
    (input ⟨(coordinate, fallback), member⟩).1
  else fallback

def readNatural (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (coordinate : CanonicalRawCoord) (fallback : Nat) : Nat :=
  match readCanonicalAtom d c input coordinate (.natural fallback) with
  | .natural value => value
  | _ => fallback

def readFormula (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (coordinate : CanonicalRawCoord) (fallback : Formula) : Formula :=
  match readCanonicalAtom d c input coordinate (.formula fallback) with
  | .formula value => value
  | _ => fallback

def readFlag (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (coordinate : CanonicalRawCoord) (fallback : Bool) : Bool :=
  match readCanonicalAtom d c input coordinate (.flag fallback) with
  | .flag value => value
  | _ => fallback

private theorem readCanonicalAtom_region_congr (d : Graph)
    (c : Certificate) (region : CanonicalFieldRegion)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c region) left right)
    (coordinate : CanonicalRawCoord) (fallback : CanonicalAtom)
    (owned : canonicalRawCoordRegion coordinate = region) :
    readCanonicalAtom d c left coordinate fallback =
      readCanonicalAtom d c right coordinate fallback := by
  unfold readCanonicalAtom
  split
  · apply congrArg Subtype.val
    symm
    apply agree
    simp [canonicalRegionReadSet, owned]
  · rfl

private theorem readNatural_region_congr (d : Graph) (c : Certificate)
    (region : CanonicalFieldRegion)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c region) left right)
    (coordinate : CanonicalRawCoord) (fallback : Nat)
    (owned : canonicalRawCoordRegion coordinate = region) :
    readNatural d c left coordinate fallback =
      readNatural d c right coordinate fallback := by
  unfold readNatural
  rw [readCanonicalAtom_region_congr d c region left right agree coordinate
    (.natural fallback) owned]

private theorem readFormula_region_congr (d : Graph) (c : Certificate)
    (region : CanonicalFieldRegion)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c region) left right)
    (coordinate : CanonicalRawCoord) (fallback : Formula)
    (owned : canonicalRawCoordRegion coordinate = region) :
    readFormula d c left coordinate fallback =
      readFormula d c right coordinate fallback := by
  unfold readFormula
  rw [readCanonicalAtom_region_congr d c region left right agree coordinate
    (.formula fallback) owned]

private theorem readFlag_region_congr (d : Graph) (c : Certificate)
    (region : CanonicalFieldRegion)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c region) left right)
    (coordinate : CanonicalRawCoord) (fallback : Bool)
    (owned : canonicalRawCoordRegion coordinate = region) :
    readFlag d c left coordinate fallback =
      readFlag d c right coordinate fallback := by
  unfold readFlag
  rw [readCanonicalAtom_region_congr d c region left right agree coordinate
    (.flag fallback) owned]

@[simp] private theorem readCanonicalAtom_encode (d : Graph)
    (c : Certificate) (coordinate : CanonicalRawCoord)
    (fallback : CanonicalAtom) :
    readCanonicalAtom d c (encodeCertificate d c) coordinate fallback =
      fallback := by
  unfold readCanonicalAtom
  split <;> rfl

@[simp] private theorem readNatural_encode (d : Graph) (c : Certificate)
    (coordinate : CanonicalRawCoord) (fallback : Nat) :
    readNatural d c (encodeCertificate d c) coordinate fallback = fallback := by
  simp [readNatural]

@[simp] private theorem readFormula_encode (d : Graph) (c : Certificate)
    (coordinate : CanonicalRawCoord) (fallback : Formula) :
    readFormula d c (encodeCertificate d c) coordinate fallback = fallback := by
  simp [readFormula]

@[simp] private theorem readFlag_encode (d : Graph) (c : Certificate)
    (coordinate : CanonicalRawCoord) (fallback : Bool) :
    readFlag d c (encodeCertificate d c) coordinate fallback = fallback := by
  simp [readFlag]

def decodeVertex (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (occurrence : CanonicalVertexOccurrence) (vertex : Vertex) : Vertex :=
  { NUMBER := readNatural d c input (.vertexNumber occurrence) vertex.NUMBER
    LEVEL := readNatural d c input (.vertexLevel occurrence) vertex.LEVEL
    FORMULA := readFormula d c input (.vertexFormula occurrence) vertex.FORMULA
    HYPOTHESIS := readFlag d c input (.vertexHypothesis occurrence)
      vertex.HYPOTHESIS
    COLLAPSED := readFlag d c input (.vertexCollapsed occurrence)
      vertex.COLLAPSED
    PAST := (boundedEntries vertex.PAST 0).zipIdx.filterMap fun (entry, index) =>
      if readFlag d c input (.vertexPastPresent occurrence index) entry.1 then
        some (readNatural d c input (.vertexPast occurrence index) entry.2)
      else none }

def decodeDeduction (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (occurrence : CanonicalDeductionOccurrence) (edge : Deduction) : Deduction :=
  { START := decodeVertex d c input (.deductionStart occurrence) edge.START
    END := decodeVertex d c input (.deductionEnd occurrence) edge.END
    COLOUR := readNatural d c input (.deductionColour occurrence) edge.COLOUR
    DEPENDENCY := (boundedEntries edge.DEPENDENCY (#"BAD")).zipIdx.filterMap
        fun (entry, index) =>
      if readFlag d c input
          (.deductionDependencyPresent occurrence index) entry.1 then
        some (readFormula d c input
          (.deductionDependency occurrence index) entry.2)
      else none }

private def decodeAncestral (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (pathIndex : Nat) (path : Ancestral) : Ancestral :=
  { START := decodeVertex d c input (.ancestralStart pathIndex) path.START
    END := decodeVertex d c input (.ancestralEnd pathIndex) path.END
    COLOURS := (boundedEntries path.COLOURS 999).zipIdx.filterMap
        fun (entry, index) =>
      if readFlag d c input (.ancestralColourPresent pathIndex index) entry.1 then
        some (readNatural d c input (.ancestralColour pathIndex index) entry.2)
      else none }

def decodeWitness (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (witnessIndex : Nat) (witness : EdgeRouteWitness) : EdgeRouteWitness :=
  { edge := decodeDeduction d c input (.routeWitnessEdge witnessIndex)
      witness.edge
    dependency := (boundedEntries witness.dependency (#"BAD")).zipIdx.filterMap
        fun (entry, index) =>
      if readFlag d c input
          (.witnessDependencyPresent witnessIndex index) entry.1 then
        some (readFormula d c input
          (.witnessDependency witnessIndex index) entry.2)
      else none
    residual := (boundedEntries witness.residual 999).zipIdx.filterMap
        fun (entry, index) =>
      if readFlag d c input (.witnessResidualPresent witnessIndex index)
          entry.1 then
        some (readNatural d c input (.witnessResidual witnessIndex index) entry.2)
      else none }

def decodeTerminal (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (terminalIndex : Nat) (output : TerminalOutput) : TerminalOutput :=
  { node := decodeVertex d c input (.terminalNode terminalIndex) output.node
    dependency := (boundedEntries output.dependency (#"BAD")).zipIdx.filterMap
        fun (entry, index) =>
      if readFlag d c input
          (.terminalDependencyPresent terminalIndex index) entry.1 then
        some (readFormula d c input
          (.terminalDependency terminalIndex index) entry.2)
      else none
    residual := (boundedEntries output.residual 999).zipIdx.filterMap
        fun (entry, index) =>
      if readFlag d c input (.terminalResidualPresent terminalIndex index)
          entry.1 then
        some (readNatural d c input (.terminalResidual terminalIndex index) entry.2)
      else none }

private theorem decodeVertex_region_congr (d : Graph) (c : Certificate)
    (region : CanonicalFieldRegion)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c region) left right)
    (occurrence : CanonicalVertexOccurrence) (vertex : Vertex)
    (owned : vertexOccurrenceRegion occurrence = region) :
    decodeVertex d c left occurrence vertex =
      decodeVertex d c right occurrence vertex := by
  unfold decodeVertex
  congr 1
  · exact readNatural_region_congr d c region left right agree
      (.vertexNumber occurrence) vertex.NUMBER owned
  · exact readNatural_region_congr d c region left right agree
      (.vertexLevel occurrence) vertex.LEVEL owned
  · exact readFormula_region_congr d c region left right agree
      (.vertexFormula occurrence) vertex.FORMULA owned
  · exact readFlag_region_congr d c region left right agree
      (.vertexHypothesis occurrence) vertex.HYPOTHESIS owned
  · exact readFlag_region_congr d c region left right agree
      (.vertexCollapsed occurrence) vertex.COLLAPSED owned
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c region left right agree
      (.vertexPastPresent occurrence index) anchorPresent owned]
    by_cases present :
        readFlag d c right (.vertexPastPresent occurrence index)
          anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readNatural_region_congr d c region left right agree
        (.vertexPast occurrence index) anchorValue owned
    · simp [present]

private theorem decodeDeduction_region_congr (d : Graph) (c : Certificate)
    (region : CanonicalFieldRegion)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c region) left right)
    (occurrence : CanonicalDeductionOccurrence) (edge : Deduction)
    (owned : deductionOccurrenceRegion occurrence = region) :
    decodeDeduction d c left occurrence edge =
      decodeDeduction d c right occurrence edge := by
  unfold decodeDeduction
  congr 1
  · exact decodeVertex_region_congr d c region left right agree
      (.deductionStart occurrence) edge.START owned
  · exact decodeVertex_region_congr d c region left right agree
      (.deductionEnd occurrence) edge.END owned
  · exact readNatural_region_congr d c region left right agree
      (.deductionColour occurrence) edge.COLOUR owned
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c region left right agree
      (.deductionDependencyPresent occurrence index) anchorPresent owned]
    by_cases present : readFlag d c right
        (.deductionDependencyPresent occurrence index) anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readFormula_region_congr d c region left right agree
        (.deductionDependency occurrence index) anchorValue owned
    · simp [present]

private theorem decodeAncestral_graph_congr (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c .graph) left right)
    (pathIndex : Nat) (path : Ancestral) :
    decodeAncestral d c left pathIndex path =
      decodeAncestral d c right pathIndex path := by
  unfold decodeAncestral
  congr 1
  · exact decodeVertex_region_congr d c .graph left right agree
      (.ancestralStart pathIndex) path.START rfl
  · exact decodeVertex_region_congr d c .graph left right agree
      (.ancestralEnd pathIndex) path.END rfl
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c .graph left right agree
      (.ancestralColourPresent pathIndex index) anchorPresent rfl]
    by_cases present : readFlag d c right
        (.ancestralColourPresent pathIndex index) anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readNatural_region_congr d c .graph left right agree
        (.ancestralColour pathIndex index) anchorValue rfl
    · simp [present]

private theorem decodeWitness_congr (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c .witness) left right)
    (witnessIndex : Nat) (witness : EdgeRouteWitness) :
    decodeWitness d c left witnessIndex witness =
      decodeWitness d c right witnessIndex witness := by
  unfold decodeWitness
  congr 1
  · exact decodeDeduction_region_congr d c .witness left right agree
      (.routeWitnessEdge witnessIndex) witness.edge rfl
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c .witness left right agree
      (.witnessDependencyPresent witnessIndex index) anchorPresent rfl]
    by_cases present : readFlag d c right
        (.witnessDependencyPresent witnessIndex index) anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readFormula_region_congr d c .witness left right agree
        (.witnessDependency witnessIndex index) anchorValue rfl
    · simp [present]
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c .witness left right agree
      (.witnessResidualPresent witnessIndex index) anchorPresent rfl]
    by_cases present : readFlag d c right
        (.witnessResidualPresent witnessIndex index) anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readNatural_region_congr d c .witness left right agree
        (.witnessResidual witnessIndex index) anchorValue rfl
    · simp [present]

private theorem decodeTerminal_congr (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c .terminal) left right)
    (terminalIndex : Nat) (output : TerminalOutput) :
    decodeTerminal d c left terminalIndex output =
      decodeTerminal d c right terminalIndex output := by
  unfold decodeTerminal
  congr 1
  · exact decodeVertex_region_congr d c .terminal left right agree
      (.terminalNode terminalIndex) output.node rfl
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c .terminal left right agree
      (.terminalDependencyPresent terminalIndex index) anchorPresent rfl]
    by_cases present : readFlag d c right
        (.terminalDependencyPresent terminalIndex index) anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readFormula_region_congr d c .terminal left right agree
        (.terminalDependency terminalIndex index) anchorValue rfl
    · simp [present]
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, index⟩
    simp only
    rw [readFlag_region_congr d c .terminal left right agree
      (.terminalResidualPresent terminalIndex index) anchorPresent rfl]
    by_cases present : readFlag d c right
        (.terminalResidualPresent terminalIndex index) anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readNatural_region_congr d c .terminal left right agree
        (.terminalResidual terminalIndex index) anchorValue rfl
    · simp [present]

/-- Decoder for all verifier-visible fields.  Record counts are bounded by the
anchor and each inner-list bound is the anchor length plus one. -/
def decodeCertificate (d : Graph) (c : Certificate)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c)) :
    Certificate :=
  { graph :=
      { NODES := c.graph.NODES.zipIdx.filterMap fun (vertex, index) =>
          if readFlag d c input (.graphNodePresent index) true then
            some (decodeVertex d c input (.graphNode index) vertex)
          else none
        EDGES := c.graph.EDGES.zipIdx.filterMap fun (deduction, index) =>
          if readFlag d c input (.graphEdgePresent index) true then
            some (decodeDeduction d c input (.graphEdge index) deduction)
          else none
        PATHS := c.graph.PATHS.zipIdx.filterMap fun (ancestral, index) =>
          if readFlag d c input (.ancestralPresent index) true then
            some (decodeAncestral d c input index ancestral)
          else none }
    edgeRoutes := c.edgeRoutes.zipIdx.filterMap fun (witness, index) =>
      if readFlag d c input (.witnessPresent index) true then
        some (decodeWitness d c input index witness)
      else none
    terminalOutputs := c.terminalOutputs.zipIdx.filterMap fun (output, index) =>
      if readFlag d c input (.terminalPresent index) true then
        some (decodeTerminal d c input index output)
      else none }

theorem decodeCertificate_graph_congr (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c .graph) left right) :
    (decodeCertificate d c left).graph =
      (decodeCertificate d c right).graph := by
  simp only [decodeCertificate]
  congr 1
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨vertex, index⟩
    simp only
    rw [readFlag_region_congr d c .graph left right agree
      (.graphNodePresent index) true rfl]
    by_cases present : readFlag d c right (.graphNodePresent index) true = true
    · simp only [present, if_true, Option.some.injEq]
      exact decodeVertex_region_congr d c .graph left right agree
        (.graphNode index) vertex rfl
    · simp [present]
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨edge, index⟩
    simp only
    rw [readFlag_region_congr d c .graph left right agree
      (.graphEdgePresent index) true rfl]
    by_cases present : readFlag d c right (.graphEdgePresent index) true = true
    · simp only [present, if_true, Option.some.injEq]
      exact decodeDeduction_region_congr d c .graph left right agree
        (.graphEdge index) edge rfl
    · simp [present]
  · apply List.filterMap_congr
    intro entry entryMem
    rcases entry with ⟨path, index⟩
    simp only
    rw [readFlag_region_congr d c .graph left right agree
      (.ancestralPresent index) true rfl]
    by_cases present : readFlag d c right (.ancestralPresent index) true = true
    · simp only [present, if_true, Option.some.injEq]
      exact decodeAncestral_graph_congr d c left right agree index path
    · simp [present]

theorem decodeCertificate_edgeRoutes_congr (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c .witness) left right) :
    (decodeCertificate d c left).edgeRoutes =
      (decodeCertificate d c right).edgeRoutes := by
  unfold decodeCertificate
  apply List.filterMap_congr
  intro entry entryMem
  rcases entry with ⟨witness, index⟩
  simp only
  rw [readFlag_region_congr d c .witness left right agree
    (.witnessPresent index) true rfl]
  by_cases present : readFlag d c right (.witnessPresent index) true = true
  · simp only [present, if_true, Option.some.injEq]
    exact decodeWitness_congr d c left right agree index witness
  · simp [present]

theorem decodeCertificate_terminalOutputs_congr (d : Graph)
    (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalRegionReadSet d c .terminal) left right) :
    (decodeCertificate d c left).terminalOutputs =
      (decodeCertificate d c right).terminalOutputs := by
  unfold decodeCertificate
  apply List.filterMap_congr
  intro entry entryMem
  rcases entry with ⟨output, index⟩
  simp only
  rw [readFlag_region_congr d c .terminal left right agree
    (.terminalPresent index) true rfl]
  by_cases present : readFlag d c right (.terminalPresent index) true = true
  · simp only [present, if_true, Option.some.injEq]
    exact decodeTerminal_congr d c left right agree index output
  · simp [present]

@[simp] private theorem decodeVertex_encode (d : Graph) (c : Certificate)
    (occurrence : CanonicalVertexOccurrence) (vertex : Vertex) :
    decodeVertex d c (encodeCertificate d c) occurrence vertex = vertex := by
  cases vertex
  simp [decodeVertex]

@[simp] private theorem decodeDeduction_encode (d : Graph) (c : Certificate)
    (occurrence : CanonicalDeductionOccurrence) (edge : Deduction) :
    decodeDeduction d c (encodeCertificate d c) occurrence edge = edge := by
  cases edge
  simp [decodeDeduction]

@[simp] private theorem decodeAncestral_encode (d : Graph) (c : Certificate)
    (pathIndex : Nat) (path : Ancestral) :
    decodeAncestral d c (encodeCertificate d c) pathIndex path = path := by
  cases path
  simp [decodeAncestral]

@[simp] private theorem decodeWitness_encode (d : Graph) (c : Certificate)
    (witnessIndex : Nat) (witness : EdgeRouteWitness) :
    decodeWitness d c (encodeCertificate d c) witnessIndex witness = witness := by
  cases witness
  simp [decodeWitness]

@[simp] private theorem decodeTerminal_encode (d : Graph) (c : Certificate)
    (terminalIndex : Nat) (output : TerminalOutput) :
    decodeTerminal d c (encodeCertificate d c) terminalIndex output = output := by
  cases output
  simp [decodeTerminal]

/-- Exact round trip at every anchor; no formation assumption is needed. -/
theorem decodeCertificate_encodeCertificate (d : Graph) (c : Certificate) :
    decodeCertificate d c (encodeCertificate d c) = c := by
  cases c with
  | mk graph edgeRoutes terminalOutputs =>
      cases graph
      simp [decodeCertificate]

/-- Hence the bounded representation is observationally exact for every
total verifier position. -/
theorem totalCheckPositionB_decode_encode (d : Graph) (c : Certificate)
    (position : Nat) :
    totalCheckPositionB d
        (decodeCertificate d c (encodeCertificate d c)) position =
      totalCheckPositionB d c position := by
  rw [decodeCertificate_encodeCertificate]

/-! ## Structurally computed read sets

The numeric total-position API has one unavoidable source of non-locality:
decoded record-presence bits can change the lengths of the witness and
terminal blocks, and hence can change which check a later numeric index names.
The prefix below is the part whose class is stable under every bounded decode.
-/

/-- Length of the total-position prefix which remains in `ExpandedCert` even
when every optional witness and terminal record is absent. -/
def canonicalStableExpandedPrefix (d : Graph) : Nat :=
  1 + d.EDGES.length + d.NODES.length

/-- A syntactic read set for every total numeric position.

* position zero reads only the embedded graph;
* the remaining class-stable expanded prefix reads witness and terminal data;
* later indices conservatively read the whole encoding because presence bits
  can move the numeric index between expanded and local-check classes.
-/
def canonicalSyntacticReadSet (d : Graph) (c : Certificate)
    (position : Nat) : Finset (CanonicalCoord d c) :=
  if position = 0 then
    canonicalRegionReadSet d c .graph
  else if position < canonicalStableExpandedPrefix d then
    canonicalRegionReadSet d c .witness ∪
      canonicalRegionReadSet d c .terminal
  else Finset.univ

private theorem expandedNumPositions_stable_lower_bound (d : Graph)
    (c : Certificate) :
    canonicalStableExpandedPrefix d ≤ expandedNumPositions d c := by
  simp [canonicalStableExpandedPrefix, expandedNumPositions,
    expandedPositionChecks]
  omega

private theorem totalCheckPositionB_zero_congr (d : Graph)
    (left right : Certificate) (graphEq : left.graph = right.graph) :
    totalCheckPositionB d left 0 = totalCheckPositionB d right 0 := by
  have leftPositive : 0 < expandedNumPositions d left := by
    have := expandedNumPositions_stable_lower_bound d left
    unfold canonicalStableExpandedPrefix at this
    omega
  have rightPositive : 0 < expandedNumPositions d right := by
    have := expandedNumPositions_stable_lower_bound d right
    unfold canonicalStableExpandedPrefix at this
    omega
  unfold totalCheckPositionB
  rw [dif_pos leftPositive, dif_pos rightPositive]
  simp [expandedPositionChecks, graphEq]

private theorem totalCheckPositionB_stable_prefix_congr (d : Graph)
    (left right : Certificate) (position : Nat)
    (positive : 0 < position)
    (stable : position < canonicalStableExpandedPrefix d)
    (routesEq : left.edgeRoutes = right.edgeRoutes)
    (terminalsEq : left.terminalOutputs = right.terminalOutputs) :
    totalCheckPositionB d left position =
      totalCheckPositionB d right position := by
  have leftExpanded : position < expandedNumPositions d left :=
    stable.trans_le (expandedNumPositions_stable_lower_bound d left)
  have rightExpanded : position < expandedNumPositions d right :=
    stable.trans_le (expandedNumPositions_stable_lower_bound d right)
  unfold totalCheckPositionB
  rw [dif_pos leftExpanded, dif_pos rightExpanded]
  cases left with
  | mk leftGraph leftRoutes leftTerminals =>
      cases right with
      | mk rightGraph rightRoutes rightTerminals =>
          simp only at routesEq terminalsEq
          subst rightRoutes
          subst rightTerminals
          cases position with
          | zero => omega
          | succ index =>
              have sameCoverage :
                  witnessCoverageAtB
                      { graph := leftGraph, edgeRoutes := leftRoutes,
                        terminalOutputs := leftTerminals } =
                    witnessCoverageAtB
                      { graph := rightGraph, edgeRoutes := leftRoutes,
                        terminalOutputs := leftTerminals } := rfl
              simp only [expandedPositionChecks]
              simp only [sameCoverage, List.singleton_append]
              rfl

/-- Agreement on the structurally computed read set fixes the total
formation-failure oracle for arbitrary bounded encodings. -/
theorem canonicalSyntacticReadSet_agreement (d : Graph) (c : Certificate)
    (position : Nat)
    (anchor other : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalSyntacticReadSet d c position)
      anchor other) :
    totalFailureB d (decodeCertificate d c) position other =
      totalFailureB d (decodeCertificate d c) position anchor := by
  by_cases zero : position = 0
  · subst position
    have graphAgree : AgreeOn (canonicalRegionReadSet d c .graph)
        anchor other := by
      simpa [canonicalSyntacticReadSet] using agree
    unfold totalFailureB
    rw [totalCheckPositionB_zero_congr d _ _
      (decodeCertificate_graph_congr d c other anchor
        (fun coordinate member => (graphAgree coordinate member).symm))]
  · by_cases stable : position < canonicalStableExpandedPrefix d
    · have regionAgree : AgreeOn
          (canonicalRegionReadSet d c .witness ∪
            canonicalRegionReadSet d c .terminal) anchor other := by
        simpa [canonicalSyntacticReadSet, zero, stable] using agree
      have witnessAgree : AgreeOn (canonicalRegionReadSet d c .witness)
          anchor other := by
        intro coordinate member
        exact regionAgree coordinate (Finset.mem_union_left _ member)
      have terminalAgree : AgreeOn (canonicalRegionReadSet d c .terminal)
          anchor other := by
        intro coordinate member
        exact regionAgree coordinate (Finset.mem_union_right _ member)
      unfold totalFailureB
      rw [totalCheckPositionB_stable_prefix_congr d _ _ position
        (Nat.pos_of_ne_zero zero) stable
        (decodeCertificate_edgeRoutes_congr d c other anchor
          (fun coordinate member => (witnessAgree coordinate member).symm))
        (decodeCertificate_terminalOutputs_congr d c other anchor
          (fun coordinate member => (terminalAgree coordinate member).symm))]
    · have inputEq : other = anchor := by
        funext coordinate
        exact agree coordinate (by
          simp [canonicalSyntacticReadSet, zero, stable])
      rw [inputEq]

def canonicalSupportOfPosition (d : Graph) (c : Certificate)
    (position : Nat) : Finset (CanonicalCoord d c) :=
  supportOfPosition d (decodeCertificate d c) position

def canonicalSupportOfTotalPosition (d : Graph) (c : Certificate)
    (position : Nat) : Finset (CanonicalCoord d c) :=
  supportOfTotalPosition d (decodeCertificate d c) position

/-- The extensional inclusion-minimal semantic support is contained in the
structurally computed read set. -/
theorem canonicalSupportOfTotalPosition_subset_syntactic (d : Graph)
    (c : Certificate) (position : Nat) :
    canonicalSupportOfTotalPosition d c position ⊆
      canonicalSyntacticReadSet d c position := by
  exact supportOfOracle_minimal
    (totalFailureB d (decodeCertificate d c) position)
    (canonicalSyntacticReadSet d c position)
    (canonicalSyntacticReadSet_agreement d c position)

theorem canonicalSupportOfPosition_agreement (d : Graph) (c : Certificate)
    (position : Nat)
    (other : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalSupportOfPosition d c position)
      (encodeCertificate d c) other) :
    markedAt d (decodeCertificate d c other) position = markedAt d c position := by
  have fixed := supportOfPosition_agreement d (decodeCertificate d c) position
    (encodeCertificate d c) other agree
  simpa [canonicalSupportOfPosition, localFailureB,
    decodeCertificate_encodeCertificate] using fixed

theorem canonicalSupportOfTotalPosition_agreement (d : Graph)
    (c : Certificate) (position : Nat)
    (other : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalSupportOfTotalPosition d c position)
      (encodeCertificate d c) other) :
    totalFailureB d (decodeCertificate d c) position other =
      !totalCheckPositionB d c position := by
  have fixed := supportOfTotalPosition_agreement d (decodeCertificate d c)
    position (encodeCertificate d c) other agree
  simpa [canonicalSupportOfTotalPosition, totalFailureB,
    decodeCertificate_encodeCertificate] using fixed

/-! ## Explicit structural field budget -/

def vertexFieldWidth (vertex : Vertex) : Nat :=
  5 + 2 * (vertex.PAST.length + 1)

def deductionFieldWidth (edge : Deduction) : Nat :=
  vertexFieldWidth edge.START + vertexFieldWidth edge.END + 1 +
    2 * (edge.DEPENDENCY.length + 1)

def ancestralFieldWidth (path : Ancestral) : Nat :=
  vertexFieldWidth path.START + vertexFieldWidth path.END +
    2 * (path.COLOURS.length + 1)

def witnessFieldWidth (witness : EdgeRouteWitness) : Nat :=
  deductionFieldWidth witness.edge + 2 * (witness.dependency.length + 1) +
    2 * (witness.residual.length + 1)

def terminalFieldWidth (output : TerminalOutput) : Nat :=
  vertexFieldWidth output.node + 2 * (output.dependency.length + 1) +
    2 * (output.residual.length + 1)

@[simp] private theorem sum_map_zipIdx_fst {α : Type}
    (values : List α) (width : α → Nat) (start : Nat := 0) :
    ((values.zipIdx start).map fun entry => width entry.1).sum =
      (values.map width).sum := by
  induction values generalizing start with
  | nil => rfl
  | cons value rest induction =>
      simp [List.zipIdx, induction]

@[simp] private theorem vertexSlots_length
    (occurrence : CanonicalVertexOccurrence) (vertex : Vertex) :
    (vertexSlots occurrence vertex).length = vertexFieldWidth vertex := by
  simp [vertexSlots, vertexFieldWidth]
  omega

@[simp] private theorem deductionSlots_length
    (occurrence : CanonicalDeductionOccurrence) (deduction : Deduction) :
    (deductionSlots occurrence deduction).length =
      deductionFieldWidth deduction := by
  simp [deductionSlots, deductionFieldWidth]
  omega

@[simp] private theorem ancestralSlots_length (index : Nat)
    (ancestral : Ancestral) :
    (ancestralSlots index ancestral).length =
      ancestralFieldWidth ancestral := by
  simp [ancestralSlots, ancestralFieldWidth]
  omega

@[simp] private theorem witnessSlots_length (index : Nat)
    (witness : EdgeRouteWitness) :
    (witnessSlots index witness).length = witnessFieldWidth witness := by
  simp [witnessSlots, witnessFieldWidth]
  omega

@[simp] private theorem terminalSlots_length (index : Nat)
    (output : TerminalOutput) :
    (terminalSlots index output).length = terminalFieldWidth output := by
  simp [terminalSlots, terminalFieldWidth]
  omega

/-- Structural field budget of the embedded graph region. -/
def graphFieldBudget (c : Certificate) : Nat :=
  (c.graph.NODES.map fun vertex => 1 + vertexFieldWidth vertex).sum +
  (c.graph.EDGES.map fun edge => 1 + deductionFieldWidth edge).sum +
  (c.graph.PATHS.map fun path => 1 + ancestralFieldWidth path).sum

/-- Structural field budget of the route-witness region. -/
def witnessFieldBudget (c : Certificate) : Nat :=
  (c.edgeRoutes.map fun witness => 1 + witnessFieldWidth witness).sum

/-- Structural field budget of the terminal-output region. -/
def terminalFieldBudget (c : Certificate) : Nat :=
  (c.terminalOutputs.map fun output => 1 + terminalFieldWidth output).sum

/-- Sum of all verifier-visible scalar slots.  This makes the contributions
of graph records, dependency widths, and residual widths explicit. -/
def certificateFieldBudget (c : Certificate) : Nat :=
  graphFieldBudget c + witnessFieldBudget c + terminalFieldBudget c

theorem canonicalGraphSlots_length (c : Certificate) :
    (canonicalGraphSlots c).length = graphFieldBudget c := by
  simp [canonicalGraphSlots, graphFieldBudget, List.length_flatMap]
  omega

theorem canonicalWitnessSlots_length (c : Certificate) :
    (canonicalWitnessSlots c).length = witnessFieldBudget c := by
  simp [canonicalWitnessSlots, witnessFieldBudget, List.length_flatMap]
  omega

theorem canonicalTerminalSlots_length (c : Certificate) :
    (canonicalTerminalSlots c).length = terminalFieldBudget c := by
  simp [canonicalTerminalSlots, terminalFieldBudget, List.length_flatMap]
  omega

theorem canonicalSlots_length (c : Certificate) :
    (canonicalSlots c).length = certificateFieldBudget c := by
  simp [canonicalSlots, canonicalGraphSlots_length,
    canonicalWitnessSlots_length, canonicalTerminalSlots_length,
    certificateFieldBudget]
  omega

theorem canonicalGraphReadSet_card_le_graphFieldBudget (d : Graph)
    (c : Certificate) :
    (canonicalRegionReadSet d c .graph).card ≤ graphFieldBudget c := by
  calc
    (canonicalRegionReadSet d c .graph).card ≤
        (canonicalGraphSlots c).toFinset.card := by
      apply Finset.card_le_card_of_injOn
        (fun coordinate : CanonicalCoord d c => coordinate.1)
      · intro coordinate member
        apply List.mem_toFinset.mpr
        apply canonicalSlot_mem_graph_of_region c coordinate.1
        · apply List.mem_toFinset.mp
          exact coordinate.2
        · exact (Finset.mem_filter.mp member).2
      · intro left leftMem right rightMem equality
        exact Subtype.ext equality
    _ ≤ (canonicalGraphSlots c).length := List.toFinset_card_le _
    _ = graphFieldBudget c := canonicalGraphSlots_length c

theorem canonicalWitnessReadSet_card_le_witnessFieldBudget (d : Graph)
    (c : Certificate) :
    (canonicalRegionReadSet d c .witness).card ≤ witnessFieldBudget c := by
  calc
    (canonicalRegionReadSet d c .witness).card ≤
        (canonicalWitnessSlots c).toFinset.card := by
      apply Finset.card_le_card_of_injOn
        (fun coordinate : CanonicalCoord d c => coordinate.1)
      · intro coordinate member
        apply List.mem_toFinset.mpr
        apply canonicalSlot_mem_witness_of_region c coordinate.1
        · apply List.mem_toFinset.mp
          exact coordinate.2
        · exact (Finset.mem_filter.mp member).2
      · intro left leftMem right rightMem equality
        exact Subtype.ext equality
    _ ≤ (canonicalWitnessSlots c).length := List.toFinset_card_le _
    _ = witnessFieldBudget c := canonicalWitnessSlots_length c

theorem canonicalTerminalReadSet_card_le_terminalFieldBudget (d : Graph)
    (c : Certificate) :
    (canonicalRegionReadSet d c .terminal).card ≤ terminalFieldBudget c := by
  calc
    (canonicalRegionReadSet d c .terminal).card ≤
        (canonicalTerminalSlots c).toFinset.card := by
      apply Finset.card_le_card_of_injOn
        (fun coordinate : CanonicalCoord d c => coordinate.1)
      · intro coordinate member
        apply List.mem_toFinset.mpr
        apply canonicalSlot_mem_terminal_of_region c coordinate.1
        · apply List.mem_toFinset.mp
          exact coordinate.2
        · exact (Finset.mem_filter.mp member).2
      · intro left leftMem right rightMem equality
        exact Subtype.ext equality
    _ ≤ (canonicalTerminalSlots c).length := List.toFinset_card_le _
    _ = terminalFieldBudget c := canonicalTerminalSlots_length c

/-- Position-specific structural bound corresponding exactly to the three
branches of `canonicalSyntacticReadSet`. -/
def canonicalPositionReadBound (d : Graph) (c : Certificate)
    (position : Nat) : Nat :=
  if position = 0 then graphFieldBudget c
  else if position < canonicalStableExpandedPrefix d then
    witnessFieldBudget c + terminalFieldBudget c
  else certificateFieldBudget c

theorem canonicalSyntacticReadSet_card_le_positionReadBound (d : Graph)
    (c : Certificate) (position : Nat) :
    (canonicalSyntacticReadSet d c position).card ≤
      canonicalPositionReadBound d c position := by
  by_cases zero : position = 0
  · simp [canonicalSyntacticReadSet, canonicalPositionReadBound, zero,
      canonicalGraphReadSet_card_le_graphFieldBudget]
  · by_cases stable : position < canonicalStableExpandedPrefix d
    · simp only [canonicalSyntacticReadSet, canonicalPositionReadBound,
        zero, stable, if_false, if_true]
      exact (Finset.card_union_le _ _).trans
        (Nat.add_le_add
          (canonicalWitnessReadSet_card_le_witnessFieldBudget d c)
          (canonicalTerminalReadSet_card_le_terminalFieldBudget d c))
    · simp only [canonicalSyntacticReadSet, canonicalPositionReadBound,
        zero, stable, if_false]
      rw [Finset.card_univ, Fintype.card_coe]
      exact (List.toFinset_card_le (canonicalSlots c)).trans_eq
        (canonicalSlots_length c)

theorem canonicalSupportOfTotalPosition_card_le_positionReadBound (d : Graph)
    (c : Certificate) (position : Nat) :
    (canonicalSupportOfTotalPosition d c position).card ≤
      canonicalPositionReadBound d c position :=
  (Finset.card_le_card
    (canonicalSupportOfTotalPosition_subset_syntactic d c position)).trans
    (canonicalSyntacticReadSet_card_le_positionReadBound d c position)

theorem canonicalCoordinate_card_le_fieldBudget (d : Graph)
    (c : Certificate) :
    Fintype.card (CanonicalCoord d c) ≤ certificateFieldBudget c := by
  rw [Fintype.card_coe]
  exact (List.toFinset_card_le (canonicalSlots c)).trans_eq
    (canonicalSlots_length c)

theorem canonicalAlphabet_card_le_fieldBudget_add_one (d : Graph)
    (c : Certificate) :
    Fintype.card (CanonicalAlphabet d c) ≤ certificateFieldBudget c + 1 := by
  rw [Fintype.card_coe]
  unfold canonicalAtomSet
  calc
    (insert (CertificateSearch.Robustness.CanonicalAtom.flag false)
        ((canonicalSlots c).map Prod.snd).toFinset).card ≤
        ((canonicalSlots c).map Prod.snd).toFinset.card + 1 :=
      Finset.card_insert_le _ _
    _ ≤ certificateFieldBudget c + 1 :=
      Nat.add_le_add_right
        ((List.toFinset_card_le ((canonicalSlots c).map Prod.snd)).trans
          (by simp [canonicalSlots_length])) 1

theorem canonicalSupportOfPosition_card_le_fieldBudget (d : Graph)
    (c : Certificate) (position : Nat) :
    (canonicalSupportOfPosition d c position).card ≤
      certificateFieldBudget c := by
  exact (supportOfPosition_card_le d (decodeCertificate d c) position).trans
    (canonicalCoordinate_card_le_fieldBudget d c)

theorem canonicalSupportOfTotalPosition_card_le_fieldBudget (d : Graph)
    (c : Certificate) (position : Nat) :
    (canonicalSupportOfTotalPosition d c position).card ≤
      certificateFieldBudget c := by
  exact (supportOfTotalPosition_card_le d (decodeCertificate d c) position).trans
    (canonicalCoordinate_card_le_fieldBudget d c)

theorem canonicalSupportOfPosition_denominator_le_fieldBudget (d : Graph)
    (c : Certificate) (position : Nat) :
    Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfPosition d c position).card ≤
      (certificateFieldBudget c + 1) ^ certificateFieldBudget c := by
  calc
    Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfPosition d c position).card ≤
      (certificateFieldBudget c + 1) ^
        (canonicalSupportOfPosition d c position).card :=
      Nat.pow_le_pow_left
        (canonicalAlphabet_card_le_fieldBudget_add_one d c) _
    _ ≤ (certificateFieldBudget c + 1) ^ certificateFieldBudget c :=
      Nat.pow_le_pow_right (by omega)
        (canonicalSupportOfPosition_card_le_fieldBudget d c position)

theorem canonicalSupportOfTotalPosition_denominator_le_fieldBudget
    (d : Graph) (c : Certificate) (position : Nat) :
    Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfTotalPosition d c position).card ≤
      (certificateFieldBudget c + 1) ^ certificateFieldBudget c := by
  calc
    Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfTotalPosition d c position).card ≤
      (certificateFieldBudget c + 1) ^
        (canonicalSupportOfTotalPosition d c position).card :=
      Nat.pow_le_pow_left
        (canonicalAlphabet_card_le_fieldBudget_add_one d c) _
    _ ≤ (certificateFieldBudget c + 1) ^ certificateFieldBudget c :=
      Nat.pow_le_pow_right (by omega)
        (canonicalSupportOfTotalPosition_card_le_fieldBudget d c position)

def canonicalMaxTotalSupport (d : Graph) (c : Certificate) : Nat :=
  (Finset.range (totalNumPositions d c)).fold max 0 fun position =>
    (canonicalSupportOfTotalPosition d c position).card

theorem canonicalMaxTotalSupport_le_fieldBudget (d : Graph)
    (c : Certificate) :
    canonicalMaxTotalSupport d c ≤ certificateFieldBudget c := by
  unfold canonicalMaxTotalSupport
  rw [Finset.fold_max_le]
  exact ⟨Nat.zero_le _, fun position _ =>
    canonicalSupportOfTotalPosition_card_le_fieldBudget d c position⟩

/-! ## Decoder-free local robustness theorem -/

noncomputable def canonicalMalformationDensity (d : Graph) (c : Certificate)
    (position : Nat) : ℚ :=
  malformationDensity d (decodeCertificate d c) position

theorem canonicalMarkedAt_density_lower_bound (d : Graph) (c : Certificate)
    {position : Nat} (position_lt : position < numPositions d c)
    (failure : MarkedAt d c position) :
    (1 : ℚ) /
        Fintype.card (CanonicalAlphabet d c) ^
          (canonicalSupportOfPosition d c position).card ≤
      canonicalMalformationDensity d c position := by
  let witness := automaticMalformationWitness
    (d := d) (decode := decodeCertificate d c)
    (anchor := encodeCertificate d c) position
    (by simpa [decodeCertificate_encodeCertificate] using position_lt)
    (by simpa [decodeCertificate_encodeCertificate] using failure)
  simpa [canonicalMalformationDensity, canonicalSupportOfPosition] using
    (malformationDensity_lower_bound witness)

theorem canonical_formation_invalid_amplification (d : Graph)
    (c : Certificate) (expanded : Certificate.Flow.ExpandedCert d c)
    (invalid : ¬ Certificate.Flow.LocalChecks d c) :
    ∃ position, position < numPositions d c ∧ MarkedAt d c position ∧
      (canonicalSupportOfPosition d c position).card ≤
        certificateFieldBudget c ∧
      (1 : ℚ) /
          Fintype.card (CanonicalAlphabet d c) ^
            (canonicalSupportOfPosition d c position).card ≤
        canonicalMalformationDensity d c position := by
  obtain ⟨position, position_lt, failure⟩ :=
    formation_invalid_implies_localMalformation d c expanded invalid
  exact ⟨position, position_lt, failure,
    canonicalSupportOfPosition_card_le_fieldBudget d c position,
    canonicalMarkedAt_density_lower_bound d c position_lt failure⟩

/-! ## Total-formation density in the canonical encoding -/

abbrev CanonicalTotalOrbit (d : Graph) (c : Certificate)
    (position : Nat) :=
  { other : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c) //
    AgreeOn (canonicalSupportOfTotalPosition d c position)
      (encodeCertificate d c) other }

abbrev CanonicalTotalMalformationAssignments (d : Graph) (c : Certificate)
    (position : Nat) :=
  { input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c) //
    totalFailureB d (decodeCertificate d c) position input = true }

noncomputable instance canonicalTotalMalformationAssignmentsFintype
    (d : Graph) (c : Certificate) (position : Nat) :
    Fintype (CanonicalTotalMalformationAssignments d c position) :=
  Fintype.ofFinite _

def canonicalTotalOrbitEquiv (d : Graph) (c : Certificate)
    (position : Nat) :
    CanonicalTotalOrbit d c position ≃
      (↥(canonicalSupportOfTotalPosition d c position)ᶜ →
        CanonicalAlphabet d c) where
  toFun other coordinate := other.1 coordinate.1
  invFun free :=
    ⟨fun coordinate =>
        if member : coordinate ∈
            canonicalSupportOfTotalPosition d c position then
          encodeCertificate d c coordinate
        else free ⟨coordinate, by simpa using member⟩,
      by
        intro coordinate member
        simp [member]⟩
  left_inv other := by
    apply Subtype.ext
    funext coordinate
    by_cases member : coordinate ∈
        canonicalSupportOfTotalPosition d c position
    · simp [member, other.2 coordinate member]
    · simp [member]
  right_inv free := by
    funext coordinate
    have outside : ¬coordinate.1 ∈
        canonicalSupportOfTotalPosition d c position :=
      Finset.mem_compl.mp coordinate.2
    simp [outside]

noncomputable instance canonicalTotalOrbitFintype
    (d : Graph) (c : Certificate) (position : Nat) :
    Fintype (CanonicalTotalOrbit d c position) :=
  Fintype.ofEquiv
    (↥(canonicalSupportOfTotalPosition d c position)ᶜ →
      CanonicalAlphabet d c)
    (canonicalTotalOrbitEquiv d c position).symm

theorem canonicalTotalOrbit_card (d : Graph) (c : Certificate)
    (position : Nat) :
    Fintype.card (CanonicalTotalOrbit d c position) =
      Fintype.card (CanonicalAlphabet d c) ^
        (Fintype.card (CanonicalCoord d c) -
          (canonicalSupportOfTotalPosition d c position).card) := by
  calc
    Fintype.card (CanonicalTotalOrbit d c position) =
        Fintype.card
          (↥(canonicalSupportOfTotalPosition d c position)ᶜ →
            CanonicalAlphabet d c) :=
      Fintype.card_congr (canonicalTotalOrbitEquiv d c position)
    _ = Fintype.card (CanonicalAlphabet d c) ^
          Fintype.card
            ↥(canonicalSupportOfTotalPosition d c position)ᶜ :=
      Fintype.card_fun
    _ = Fintype.card (CanonicalAlphabet d c) ^
          (canonicalSupportOfTotalPosition d c position)ᶜ.card := by
      congr 1
      exact Fintype.card_coe _
    _ = Fintype.card (CanonicalAlphabet d c) ^
        (Fintype.card (CanonicalCoord d c) -
          (canonicalSupportOfTotalPosition d c position).card) := by
      congr 1
      exact Finset.card_compl
        (canonicalSupportOfTotalPosition d c position)

private theorem canonicalTotalFailure_of_orbit (d : Graph)
    (c : Certificate) {position : Nat} (failure : TotalMarkedAt d c position)
    (other : CanonicalTotalOrbit d c position) :
    totalFailureB d (decodeCertificate d c) position other.1 = true := by
  have fixed := canonicalSupportOfTotalPosition_agreement d c position
    other.1 other.2
  rw [fixed]
  simpa [TotalMarkedAt] using failure

private def canonicalTotalOrbitToAssignments (d : Graph) (c : Certificate)
    {position : Nat} (failure : TotalMarkedAt d c position) :
    CanonicalTotalOrbit d c position →
      CanonicalTotalMalformationAssignments d c position :=
  fun other => ⟨other.1,
    canonicalTotalFailure_of_orbit d c failure other⟩

private theorem canonicalTotalOrbitToAssignments_injective (d : Graph)
    (c : Certificate) {position : Nat}
    (failure : TotalMarkedAt d c position) :
    Function.Injective
      (canonicalTotalOrbitToAssignments d c failure) := by
  intro left right equal
  apply Subtype.ext
  exact congrArg
    (fun bad : CanonicalTotalMalformationAssignments d c position => bad.1)
    equal

theorem canonicalTotalMalformationAssignments_lower_bound (d : Graph)
    (c : Certificate) {position : Nat} (failure : TotalMarkedAt d c position) :
    Fintype.card (CanonicalAlphabet d c) ^
        (Fintype.card (CanonicalCoord d c) -
          (canonicalSupportOfTotalPosition d c position).card) ≤
      Fintype.card (CanonicalTotalMalformationAssignments d c position) := by
  rw [← canonicalTotalOrbit_card d c position]
  exact Fintype.card_le_of_injective
    (canonicalTotalOrbitToAssignments d c failure)
    (canonicalTotalOrbitToAssignments_injective d c failure)

noncomputable def canonicalTotalMalformationDensity (d : Graph)
    (c : Certificate) (position : Nat) : ℚ :=
  Fintype.card (CanonicalTotalMalformationAssignments d c position) /
    Fintype.card (CanonicalAlphabet d c) ^
      Fintype.card (CanonicalCoord d c)

theorem canonicalTotalMarkedAt_density_lower_bound (d : Graph)
    (c : Certificate) {position : Nat} (failure : TotalMarkedAt d c position) :
    (1 : ℚ) /
        Fintype.card (CanonicalAlphabet d c) ^
          (canonicalSupportOfTotalPosition d c position).card ≤
      canonicalTotalMalformationDensity d c position := by
  have alphabetPositive : 0 < Fintype.card (CanonicalAlphabet d c) :=
    Fintype.card_pos
  have supportLe :
      (canonicalSupportOfTotalPosition d c position).card ≤
        Fintype.card (CanonicalCoord d c) := Finset.card_le_univ _
  have lower := canonicalTotalMalformationAssignments_lower_bound d c failure
  have cross :
      Fintype.card (CanonicalAlphabet d c) ^
          Fintype.card (CanonicalCoord d c) ≤
        Fintype.card (CanonicalAlphabet d c) ^
            (canonicalSupportOfTotalPosition d c position).card *
          Fintype.card
            (CanonicalTotalMalformationAssignments d c position) := by
    calc
      Fintype.card (CanonicalAlphabet d c) ^
          Fintype.card (CanonicalCoord d c) =
          Fintype.card (CanonicalAlphabet d c) ^
              (canonicalSupportOfTotalPosition d c position).card *
            Fintype.card (CanonicalAlphabet d c) ^
              (Fintype.card (CanonicalCoord d c) -
                (canonicalSupportOfTotalPosition d c position).card) := by
        rw [← pow_add, Nat.add_sub_of_le supportLe]
      _ ≤ Fintype.card (CanonicalAlphabet d c) ^
            (canonicalSupportOfTotalPosition d c position).card *
          Fintype.card
            (CanonicalTotalMalformationAssignments d c position) :=
        Nat.mul_le_mul_left _ lower
  have cross' :
      (canonicalAtomSet c).card ^ (canonicalSlotSet c).card ≤
        (canonicalAtomSet c).card ^
            (canonicalSupportOfTotalPosition d c position).card *
          Fintype.card
            (CanonicalTotalMalformationAssignments d c position) := by
    simpa only [Fintype.card_coe] using cross
  have leftPositive :
      (0 : ℚ) < Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfTotalPosition d c position).card := by
    positivity
  have rightPositive :
      (0 : ℚ) < Fintype.card (CanonicalAlphabet d c) ^
        Fintype.card (CanonicalCoord d c) := by
    positivity
  unfold canonicalTotalMalformationDensity
  apply (div_le_div_iff₀ leftPositive rightPositive).2
  norm_num
  exact_mod_cast (show
    (canonicalAtomSet c).card ^ (canonicalSlotSet c).card ≤
      Fintype.card (CanonicalTotalMalformationAssignments d c position) *
        (canonicalAtomSet c).card ^
          (canonicalSupportOfTotalPosition d c position).card by
    simpa [Nat.mul_comm] using cross')

theorem canonicalTotalMarkedAt_positionReadBound_density_lower_bound
    (d : Graph) (c : Certificate) {position : Nat}
    (failure : TotalMarkedAt d c position) :
    (1 : ℚ) /
        Fintype.card (CanonicalAlphabet d c) ^
          canonicalPositionReadBound d c position ≤
      canonicalTotalMalformationDensity d c position := by
  have exactLower := canonicalTotalMarkedAt_density_lower_bound d c failure
  have supportLe :=
    canonicalSupportOfTotalPosition_card_le_positionReadBound d c position
  have alphabetOne :
      (1 : ℚ) ≤ Fintype.card (CanonicalAlphabet d c) := by
    exact_mod_cast (Fintype.card_pos : 0 <
      Fintype.card (CanonicalAlphabet d c))
  have denominatorLe :
      (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
          (canonicalSupportOfTotalPosition d c position).card ≤
        (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
          canonicalPositionReadBound d c position :=
    pow_le_pow_right₀ alphabetOne supportLe
  have positive :
      (0 : ℚ) < (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
        (canonicalSupportOfTotalPosition d c position).card := by
    positivity
  exact (one_div_le_one_div_of_le positive denominatorLe).trans exactLower

theorem canonicalTotalMarkedAt_fieldBudget_density_lower_bound (d : Graph)
    (c : Certificate) {position : Nat} (failure : TotalMarkedAt d c position) :
    (1 : ℚ) /
        (certificateFieldBudget c + 1) ^ certificateFieldBudget c ≤
      canonicalTotalMalformationDensity d c position := by
  have exactLower := canonicalTotalMarkedAt_density_lower_bound d c failure
  have denominatorNat :=
    canonicalSupportOfTotalPosition_denominator_le_fieldBudget d c position
  have denominatorRat :
      (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
          (canonicalSupportOfTotalPosition d c position).card ≤
        (certificateFieldBudget c + 1 : ℚ) ^ certificateFieldBudget c := by
    exact_mod_cast denominatorNat
  have positive :
      (0 : ℚ) < (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
        (canonicalSupportOfTotalPosition d c position).card := by
    positivity
  exact (one_div_le_one_div_of_le positive denominatorRat).trans exactLower

/-- Decoder-free symbolic robustness for complete formation invalidity. -/
theorem total_formation_invalid_canonical_amplification (d : Graph)
    (c : Certificate) (invalid : ¬ FormationValid d c) :
    ∃ position, position < totalNumPositions d c ∧
      TotalMarkedAt d c position ∧
      (canonicalSupportOfTotalPosition d c position).card ≤
        certificateFieldBudget c ∧
      (1 : ℚ) /
          Fintype.card (CanonicalAlphabet d c) ^
            (canonicalSupportOfTotalPosition d c position).card ≤
        canonicalTotalMalformationDensity d c position ∧
      (1 : ℚ) /
          (certificateFieldBudget c + 1) ^ certificateFieldBudget c ≤
        canonicalTotalMalformationDensity d c position := by
  obtain ⟨position, position_lt, failure⟩ :=
    CertificateSearch.total_formation_invalid_implies_marked d c invalid
  exact ⟨position, position_lt, failure,
    canonicalSupportOfTotalPosition_card_le_fieldBudget d c position,
    canonicalTotalMarkedAt_density_lower_bound d c failure,
    canonicalTotalMarkedAt_fieldBudget_density_lower_bound d c failure⟩

/-- Complete formation invalidity with the new position-specific syntactic
read bound, requiring no externally supplied witness. -/
theorem total_formation_invalid_syntactic_amplification (d : Graph)
    (c : Certificate) (invalid : ¬ FormationValid d c) :
    ∃ position, position < totalNumPositions d c ∧
      TotalMarkedAt d c position ∧
      (canonicalSupportOfTotalPosition d c position).card ≤
        canonicalPositionReadBound d c position ∧
      (1 : ℚ) /
          Fintype.card (CanonicalAlphabet d c) ^
            canonicalPositionReadBound d c position ≤
        canonicalTotalMalformationDensity d c position := by
  obtain ⟨position, position_lt, failure⟩ :=
    CertificateSearch.total_formation_invalid_implies_marked d c invalid
  exact ⟨position, position_lt, failure,
    canonicalSupportOfTotalPosition_card_le_positionReadBound d c position,
    canonicalTotalMarkedAt_positionReadBound_density_lower_bound d c failure⟩

/-! ## Root-discharge boundary -/

/-- Existing local root discharge plus the missing universal condition on
every stored root route.  This is deliberately separate from
`LocalRootDischarge`, whose certificate check is existential. -/
def StrongLocalRootDischarge (c : Certificate) : Prop :=
  Certificate.Flow.LocalRootDischarge c ∧
  ∀ vertex ∈ c.graph.NODES,
    (get_rule.outgoing vertex c.graph).isEmpty = true →
    ∀ route ∈ flowAt c.graph (stdFuel c.graph) vertex,
      route.1 = [] ∧ route.2 = []

theorem strongLocalRootDischarge_local (c : Certificate)
    (strong : StrongLocalRootDischarge c) :
    Certificate.Flow.LocalRootDischarge c := strong.1

theorem strongLocalRootDischarge_rootsDischargedB (d : Graph)
    (c : Certificate) (expanded : Certificate.Flow.ExpandedCert d c)
    (hlocal : Certificate.Flow.LocalChecks d c)
    (strong : StrongLocalRootDischarge c) : rootsDischargedB d = true := by
  unfold rootsDischargedB
  rw [List.all_eq_true]
  intro vertex vertexMem
  by_cases root : (get_rule.outgoing vertex d).isEmpty = true
  · simp only [root, Bool.not_true, Bool.false_or]
    have vertexMemC : vertex ∈ c.graph.NODES := by
      simpa [expanded.1] using vertexMem
    have rootC : (get_rule.outgoing vertex c.graph).isEmpty = true := by
      simpa [expanded.1] using root
    obtain ⟨output, outputMem, outputNode, outputDep, outputResidual⟩ :=
      (strong.1 vertex vertexMemC rootC).1
    rw [Bool.and_eq_true, List.all_eq_true]
    constructor
    · have emptyRouteMem : ([], []) ∈ flowAt d (stdFuel d) vertex :=
        localChecks_root_has_empty_flowRoute d c expanded hlocal vertexMem root
      have flowNonempty : flowAt d (stdFuel d) vertex ≠ [] := by
        intro flowEmpty
        simp [flowEmpty] at emptyRouteMem
      simp [flowNonempty]
    · intro route routeMem
      have routeMemC : route ∈ flowAt c.graph (stdFuel c.graph) vertex := by
        simpa [expanded.1] using routeMem
      obtain ⟨dependencyEmpty, residualEmpty⟩ :=
        strong.2 vertex vertexMemC rootC route routeMemC
      simp [dependencyEmpty, residualEmpty]
  · have nonroot : (get_rule.outgoing vertex d).isEmpty = false :=
      Bool.eq_false_of_not_eq_true root
    simp [nonroot]

/-- Exact missing evaluator/Flow refinement.  A true dependency bit in the
canonical evaluator's goal vector must be supported by a dependency formula
on some raw Flow route at a graph root.  No theorem establishing this
property is currently present in the evaluator development. -/
def CanonicalGoalBitsSupportedByRootFlow (d : Graph) : Prop :=
  ∀ (goalVector : List.Vector Bool (buildFormulas d).length),
    (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1[goalColumn d]? =
        some goalVector →
    ∀ index, goalVector.get index = true →
      ∃ root ∈ d.NODES,
        (get_rule.outgoing root d).isEmpty = true ∧
        ∃ route ∈ flowAt d (stdFuel d) root,
          (buildFormulas d).get index ∈ route.1

theorem dischargedMultiB_of_rootsDischargedB
    (d : Graph) (supported : CanonicalGoalBitsSupportedByRootFlow d)
    (roots : rootsDischargedB d = true) : dischargedMultiB d = true := by
  unfold dischargedMultiB dischargedOnMultiB
  cases goalResult :
      (getEvalResultMultiDLDS d (multiPathsFromFlow d)).1[goalColumn d]? with
  | none => rfl
  | some goalVector =>
      rw [List.all_eq_true]
      intro bit bitMem
      have bitFalse : bit = false := by
        by_contra bitNotFalse
        have bitTrue : bit = true := by
          cases bit <;> simp_all
        obtain ⟨listIndex, indexValue⟩ := List.mem_iff_get.mp bitMem
        let index : Fin (buildFormulas d).length :=
          ⟨listIndex, by simpa using listIndex.isLt⟩
        have indexValue' : goalVector.get index = bit := by
          simpa [index, List.Vector.get, List.Vector.toList] using indexValue
        have indexTrue : goalVector.get index = true := by
          rw [indexValue', bitTrue]
        obtain ⟨root, rootMem, rootTerminal, route, routeMem,
            formulaMem⟩ := supported goalVector goalResult index indexTrue
        unfold rootsDischargedB at roots
        rw [List.all_eq_true] at roots
        have rootCheck := roots root rootMem
        simp only [rootTerminal, Bool.not_true, Bool.false_or] at rootCheck
        rw [Bool.and_eq_true, List.all_eq_true] at rootCheck
        have emptyRoute := rootCheck.2 route routeMem
        rw [Bool.and_eq_true] at emptyRoute
        have dependencyEmpty : route.1 = [] := by
          simpa using emptyRoute.1
        simp [dependencyEmpty] at formulaMem
      simp [bitFalse]

/-! ### Scope of the canonical provenance condition

`CanonicalGoalBitsSupportedByRootFlow` is not an unconditional property of an
arbitrary `Graph`.  The following three-vertex fixture isolates the obstruction.
The ordinary introduction branch discharges `A` at the unique root, while a
malformed self-loop on a hypothesis with the root formula feeds a repetition
occurrence into the same final formula cell.  Thus raw root Flow is discharged,
but the formula-merged evaluator cell still contains the hypothesis bit.

The fixture deliberately violates `structuralValid` and
`HypothesesHaveNoIncoming`; it does not contradict a bridge restricted by
`LocalChecks`. -/

namespace CanonicalGoalFlowCounterexample

def assumption : Formula := .atom "canonical_goal_flow_counterexample_A"
def goal : Formula := .implication assumption assumption

def premise : Vertex := Vertex.node 1 1 assumption true false []
def root : Vertex := Vertex.node 2 0 goal false false []
def loopingHypothesis : Vertex := Vertex.node 3 1 goal true false []

def introEdge : Deduction := Deduction.edge premise root 0 [assumption]
def loopEdge : Deduction :=
  Deduction.edge loopingHypothesis loopingHypothesis 0 [goal]

def graph : Graph :=
  Graph.dlds [premise, root, loopingHypothesis] [introEdge, loopEdge] []

def canonicalGoalVector : List.Vector Bool (buildFormulas graph).length :=
  ⟨[false, true], by native_decide⟩

theorem root_flow_exact :
    flowAt graph (stdFuel graph) root = [([], [])] := by
  native_decide

theorem canonical_goal_result_exact :
    (getEvalResultMultiDLDS graph (multiPathsFromFlow graph)).1[goalColumn graph]? =
      some canonicalGoalVector := by
  native_decide

theorem roots_discharged : rootsDischargedB graph = true := by
  native_decide

theorem canonical_slot_ok : canonicalSlotOKB graph = true := by
  native_decide

theorem canonical_paths_exact :
    multiPathsFromFlow graph = [[(2, 1)], [(2, 0)]] := by
  native_decide

theorem canonical_no_error : canonicalNoErrorB graph = true := by
  native_decide

theorem canonical_not_discharged : dischargedMultiB graph = false := by
  native_decide

theorem structural_invalid : structuralValidB graph = false := by
  native_decide

theorem hypotheses_have_incoming : hypNoIncomingB graph = false := by
  native_decide

theorem canonical_goal_bits_not_supported :
    ¬ CanonicalGoalBitsSupportedByRootFlow graph := by
  intro supported
  let index : Fin (buildFormulas graph).length := ⟨1, by native_decide⟩
  have indexTrue : canonicalGoalVector.get index = true := by native_decide
  have noRootSupport : ¬ ∃ candidate ∈ graph.NODES,
      (get_rule.outgoing candidate graph).isEmpty = true ∧
      ∃ route ∈ flowAt graph (stdFuel graph) candidate,
        (buildFormulas graph).get index ∈ route.1 := by
    native_decide
  exact noRootSupport
    (supported canonicalGoalVector canonical_goal_result_exact index indexTrue)

end CanonicalGoalFlowCounterexample

theorem dischargedMultiB_of_strongLocalRootDischarge (d : Graph)
    (c : Certificate) (expanded : Certificate.Flow.ExpandedCert d c)
    (hlocal : Certificate.Flow.LocalChecks d c)
    (strong : StrongLocalRootDischarge c)
    (supported : CanonicalGoalBitsSupportedByRootFlow d) :
    dischargedMultiB d = true :=
  dischargedMultiB_of_rootsDischargedB d supported
    (strongLocalRootDischarge_rootsDischargedB d c expanded hlocal strong)

theorem compressed_universally_accepted_from_strong_formation
    (d : Graph) (c : Certificate)
    (expanded : Certificate.Flow.ExpandedCert d c)
    (hlocal : Certificate.Flow.LocalChecks d c)
    (strong : StrongLocalRootDischarge c)
    (supported : CanonicalGoalBitsSupportedByRootFlow d)
    (slot : canonicalSlotOKB d = true) :
    ∀ paths, ¬ AdmissibleMultiPath d paths ∨
      DischargedMulti d paths (goalColumn d) := by
  intro paths
  by_cases admissible : AdmissibleMultiPath d paths
  · right
    have evaluation : getEvalResultMultiDLDS d paths =
        getEvalResultMultiDLDS d (multiPathsFromFlow d) :=
      admissibleMulti_eval_canonical d slot admissible
    have canonical : DischargedMulti d (multiPathsFromFlow d)
        (goalColumn d) :=
      discharge_multi d
        (dischargedMultiB_of_strongLocalRootDischarge d c expanded hlocal
          strong supported)
    unfold DischargedMulti at canonical ⊢
    rw [evaluation]
    exact canonical
  · exact Or.inl admissible

/-! ## Stable semantic-position read sets -/

def canonicalAllRegionReadSet (d : Graph) (c : Certificate) :
    Finset (CanonicalCoord d c) :=
  (canonicalRegionReadSet d c .graph : Finset (CanonicalCoord d c)) ∪
    ((canonicalRegionReadSet d c .witness : Finset (CanonicalCoord d c)) ∪
      (canonicalRegionReadSet d c .terminal : Finset (CanonicalCoord d c)))

/-- Structurally computed coordinates read by one stable semantic check.
Checks whose data is entirely in the external graph `d` have empty support.
The broad branch is an explicit union of the record regions: unrestricted
field substitutions can make any certificate record relevant to a node. -/
def canonicalSemanticReadSet (d : Graph) (c : Certificate) :
    VerificationPosition → Finset (CanonicalCoord d c)
  | .graphIdentity => canonicalRegionReadSet d c .graph
  | .expandedEdgeWitnessCoverage _ _
  | .localEdgeWitnessCoverage _ _ => canonicalRegionReadSet d c .witness
  | .expandedRootTerminalCoverage _ _ =>
      canonicalRegionReadSet d c .terminal
  | .expandedWitnessScope _ _
  | .expandedTerminalScope _ _ => ∅
  | .edgeNodupPrefix _
  | .edgeNodupMain _
  | .oneEdgePrefix _
  | .oneEdgeMain _
  | .levelFormulaUniquePrefix _
  | .levelFormulaUniqueMain _
  | .structuralEdge _ _
  | .structuralNode _ _
  | .hypothesisSource _ _ => ∅
  | .witnessFaithful _ _
  | .nodeProduction _ _
  | .routeCoverage _ _
  | .tailClosure _ _
  | .rootDischarge _ _
  | .degenerateLocal _ => canonicalAllRegionReadSet d c

private theorem agreeOn_graph_of_allRegions (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalAllRegionReadSet d c) left right) :
    AgreeOn (canonicalRegionReadSet d c .graph) left right := by
  intro coordinate member
  apply agree coordinate
  exact Finset.mem_union_left _ member

private theorem agreeOn_witness_of_allRegions (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalAllRegionReadSet d c) left right) :
    AgreeOn (canonicalRegionReadSet d c .witness) left right := by
  intro coordinate member
  apply agree coordinate
  exact Finset.mem_union_right _ (Finset.mem_union_left _ member)

private theorem agreeOn_terminal_of_allRegions (d : Graph) (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalAllRegionReadSet d c) left right) :
    AgreeOn (canonicalRegionReadSet d c .terminal) left right := by
  intro coordinate member
  apply agree coordinate
  exact Finset.mem_union_right _ (Finset.mem_union_right _ member)

private theorem decodeCertificate_allRegions_congr (d : Graph)
    (c : Certificate)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalAllRegionReadSet d c) left right) :
    decodeCertificate d c left = decodeCertificate d c right := by
  have graphEq := decodeCertificate_graph_congr d c left right
    (agreeOn_graph_of_allRegions d c left right agree)
  have routesEq := decodeCertificate_edgeRoutes_congr d c left right
    (agreeOn_witness_of_allRegions d c left right agree)
  have terminalsEq := decodeCertificate_terminalOutputs_congr d c left right
    (agreeOn_terminal_of_allRegions d c left right agree)
  cases leftDecoded : decodeCertificate d c left with
  | mk leftGraph leftRoutes leftTerminals =>
      cases rightDecoded : decodeCertificate d c right with
      | mk rightGraph rightRoutes rightTerminals =>
          simp only [leftDecoded, rightDecoded] at graphEq routesEq terminalsEq
          subst rightGraph
          subst rightRoutes
          subst rightTerminals
          rfl

private theorem witnessCoverageAtB_congr (left right : Certificate)
    (routesEq : left.edgeRoutes = right.edgeRoutes) :
    witnessCoverageAtB left = witnessCoverageAtB right := by
  funext edge
  unfold witnessCoverageAtB edgeWitnessesFor
  rw [routesEq]

/-- Agreement on the semantic read set fixes the actual semantic checker for
all bounded canonical assignments. -/
theorem canonicalSemanticReadSet_agreement (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (left right : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c))
    (agree : AgreeOn (canonicalSemanticReadSet d c position) left right) :
    checkSemanticPositionB d (decodeCertificate d c left) position =
      checkSemanticPositionB d (decodeCertificate d c right) position := by
  cases position with
  | graphIdentity =>
      have graphAgree : AgreeOn (canonicalRegionReadSet d c .graph)
          left right := by simpa [canonicalSemanticReadSet] using agree
      simp only [checkSemanticPositionB]
      rw [decodeCertificate_graph_congr d c left right graphAgree]
  | expandedEdgeWitnessCoverage slot deduction =>
      have witnessAgree : AgreeOn (canonicalRegionReadSet d c .witness)
          left right := by simpa [canonicalSemanticReadSet] using agree
      simp only [checkSemanticPositionB]
      rw [witnessCoverageAtB_congr _ _
        (decodeCertificate_edgeRoutes_congr d c left right witnessAgree)]
  | expandedWitnessScope slot witness => rfl
  | expandedRootTerminalCoverage slot vertex =>
      have terminalAgree : AgreeOn (canonicalRegionReadSet d c .terminal)
          left right := by simpa [canonicalSemanticReadSet] using agree
      simp only [checkSemanticPositionB]
      rw [decodeCertificate_terminalOutputs_congr d c left right terminalAgree]
  | expandedTerminalScope slot output => rfl
  | edgeNodupPrefix slot => rfl
  | edgeNodupMain slot => rfl
  | oneEdgePrefix slot => rfl
  | oneEdgeMain slot => rfl
  | levelFormulaUniquePrefix slot => rfl
  | levelFormulaUniqueMain slot => rfl
  | localEdgeWitnessCoverage slot deduction =>
      have witnessAgree : AgreeOn (canonicalRegionReadSet d c .witness)
          left right := by simpa [canonicalSemanticReadSet] using agree
      simp only [checkSemanticPositionB]
      rw [witnessCoverageAtB_congr _ _
        (decodeCertificate_edgeRoutes_congr d c left right witnessAgree)]
  | structuralEdge slot deduction => rfl
  | witnessFaithful slot witness =>
      have allAgree : AgreeOn (canonicalAllRegionReadSet d c) left right := by
        simpa [canonicalSemanticReadSet] using agree
      rw [decodeCertificate_allRegions_congr d c left right allAgree]
  | structuralNode slot vertex => rfl
  | hypothesisSource slot vertex => rfl
  | nodeProduction slot vertex =>
      have allAgree : AgreeOn (canonicalAllRegionReadSet d c) left right := by
        simpa [canonicalSemanticReadSet] using agree
      rw [decodeCertificate_allRegions_congr d c left right allAgree]
  | routeCoverage slot vertex =>
      have allAgree : AgreeOn (canonicalAllRegionReadSet d c) left right := by
        simpa [canonicalSemanticReadSet] using agree
      rw [decodeCertificate_allRegions_congr d c left right allAgree]
  | tailClosure slot vertex =>
      have allAgree : AgreeOn (canonicalAllRegionReadSet d c) left right := by
        simpa [canonicalSemanticReadSet] using agree
      rw [decodeCertificate_allRegions_congr d c left right allAgree]
  | rootDischarge slot vertex =>
      have allAgree : AgreeOn (canonicalAllRegionReadSet d c) left right := by
        simpa [canonicalSemanticReadSet] using agree
      rw [decodeCertificate_allRegions_congr d c left right allAgree]
  | degenerateLocal slot =>
      have allAgree : AgreeOn (canonicalAllRegionReadSet d c) left right := by
        simpa [canonicalSemanticReadSet] using agree
      rw [decodeCertificate_allRegions_congr d c left right allAgree]

def canonicalSemanticFailureB (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c)) : Bool :=
  !checkSemanticPositionB d (decodeCertificate d c input) position

def canonicalSupportOfSemanticPosition (d : Graph) (c : Certificate)
    (position : VerificationPosition) : Finset (CanonicalCoord d c) :=
  supportOfOracle (canonicalSemanticFailureB d c position)

theorem canonicalSupportOfSemanticPosition_subset_readSet (d : Graph)
    (c : Certificate) (position : VerificationPosition) :
    canonicalSupportOfSemanticPosition d c position ⊆
      canonicalSemanticReadSet d c position := by
  apply supportOfOracle_minimal
  intro left right agree
  unfold canonicalSemanticFailureB
  rw [canonicalSemanticReadSet_agreement d c position left right agree]

/-- The graph parameter preserves the uniform semantic-check interface. -/
def canonicalSemanticReadBound (_d : Graph) (c : Certificate) :
    VerificationPosition → Nat
  | .graphIdentity => graphFieldBudget c
  | .expandedEdgeWitnessCoverage _ _
  | .localEdgeWitnessCoverage _ _ => witnessFieldBudget c
  | .expandedRootTerminalCoverage _ _ => terminalFieldBudget c
  | .expandedWitnessScope _ _
  | .expandedTerminalScope _ _ => 0
  | .edgeNodupPrefix _
  | .edgeNodupMain _
  | .oneEdgePrefix _
  | .oneEdgeMain _
  | .levelFormulaUniquePrefix _
  | .levelFormulaUniqueMain _
  | .structuralEdge _ _
  | .structuralNode _ _
  | .hypothesisSource _ _ => 0
  | .witnessFaithful _ _
  | .nodeProduction _ _
  | .routeCoverage _ _
  | .tailClosure _ _
  | .rootDischarge _ _
  | .degenerateLocal _ => certificateFieldBudget c

theorem canonicalAllRegionReadSet_card_le_fieldBudget (d : Graph)
    (c : Certificate) :
    (canonicalAllRegionReadSet d c).card ≤ certificateFieldBudget c := by
  unfold canonicalAllRegionReadSet certificateFieldBudget
  calc
    ((canonicalRegionReadSet d c .graph : Finset (CanonicalCoord d c)) ∪
        ((canonicalRegionReadSet d c .witness : Finset (CanonicalCoord d c)) ∪
          (canonicalRegionReadSet d c .terminal :
            Finset (CanonicalCoord d c)))).card ≤
        (canonicalRegionReadSet d c .graph).card +
          ((canonicalRegionReadSet d c .witness :
              Finset (CanonicalCoord d c)) ∪
            (canonicalRegionReadSet d c .terminal :
              Finset (CanonicalCoord d c))).card :=
      Finset.card_union_le _ _
    _ ≤ graphFieldBudget c +
          (witnessFieldBudget c + terminalFieldBudget c) := by
      apply Nat.add_le_add
      · exact canonicalGraphReadSet_card_le_graphFieldBudget d c
      · exact (Finset.card_union_le _ _).trans
          (Nat.add_le_add
            (canonicalWitnessReadSet_card_le_witnessFieldBudget d c)
            (canonicalTerminalReadSet_card_le_terminalFieldBudget d c))
    _ = graphFieldBudget c + witnessFieldBudget c + terminalFieldBudget c := by
      omega

theorem canonicalSemanticReadSet_card_le (d : Graph) (c : Certificate)
    (position : VerificationPosition) :
    (canonicalSemanticReadSet d c position).card ≤
      canonicalSemanticReadBound d c position := by
  cases position <;>
    simp [canonicalSemanticReadSet, canonicalSemanticReadBound,
      canonicalGraphReadSet_card_le_graphFieldBudget,
      canonicalWitnessReadSet_card_le_witnessFieldBudget,
      canonicalTerminalReadSet_card_le_terminalFieldBudget,
      canonicalAllRegionReadSet_card_le_fieldBudget]

theorem canonicalSupportOfSemanticPosition_card_le (d : Graph)
    (c : Certificate) (position : VerificationPosition) :
    (canonicalSupportOfSemanticPosition d c position).card ≤
      canonicalSemanticReadBound d c position :=
  (Finset.card_le_card
    (canonicalSupportOfSemanticPosition_subset_readSet d c position)).trans
    (canonicalSemanticReadSet_card_le d c position)

/-! Position-family statements are kept explicit so downstream bounds do not
have to unfold the semantic dispatcher. -/

theorem graphIdentity_readSet_card_le (d : Graph) (c : Certificate) :
    (canonicalSemanticReadSet d c .graphIdentity).card ≤
      graphFieldBudget c :=
  canonicalSemanticReadSet_card_le d c .graphIdentity

theorem edgeWitnessCoverage_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (deduction : Deduction) :
    (canonicalSemanticReadSet d c
      (.expandedEdgeWitnessCoverage slot deduction)).card ≤
        witnessFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem localEdgeWitnessCoverage_readSet_card_le (d : Graph)
    (c : Certificate) (slot : Nat) (deduction : Deduction) :
    (canonicalSemanticReadSet d c
      (.localEdgeWitnessCoverage slot deduction)).card ≤
        witnessFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem expandedWitnessScope_readSet_card_eq_zero (d : Graph)
    (c : Certificate) (slot : Nat) (witness : EdgeRouteWitness) :
    (canonicalSemanticReadSet d c
      (.expandedWitnessScope slot witness)).card = 0 := by
  rfl

theorem rootTerminalCoverage_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c
      (.expandedRootTerminalCoverage slot vertex)).card ≤
        terminalFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem expandedTerminalScope_readSet_card_eq_zero (d : Graph)
    (c : Certificate) (slot : Nat) (output : TerminalOutput) :
    (canonicalSemanticReadSet d c
      (.expandedTerminalScope slot output)).card = 0 := by
  rfl

theorem structuralEdge_readSet_card_eq_zero (d : Graph) (c : Certificate)
    (slot : Nat) (deduction : Deduction) :
    (canonicalSemanticReadSet d c (.structuralEdge slot deduction)).card = 0 := by
  rfl

theorem structuralNode_readSet_card_eq_zero (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.structuralNode slot vertex)).card = 0 := by
  rfl

theorem hypothesisSource_readSet_card_eq_zero (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.hypothesisSource slot vertex)).card = 0 := by
  rfl

theorem witnessFaithful_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (witness : EdgeRouteWitness) :
    (canonicalSemanticReadSet d c (.witnessFaithful slot witness)).card ≤
      certificateFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem nodeProduction_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.nodeProduction slot vertex)).card ≤
      certificateFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem routeCoverage_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.routeCoverage slot vertex)).card ≤
      certificateFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem tailClosure_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.tailClosure slot vertex)).card ≤
      certificateFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

theorem rootDischarge_readSet_card_le (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.rootDischarge slot vertex)).card ≤
      certificateFieldBudget c :=
  canonicalSemanticReadSet_card_le d c _

/-- Largest record width among stored route witnesses. -/
def maxWitnessRecordWidth (c : Certificate) : Nat :=
  (c.edgeRoutes.map fun witness => 1 + witnessFieldWidth witness).foldr max 0

theorem sum_le_length_mul_foldr_max (values : List Nat) :
    values.sum ≤ values.length * values.foldr max 0 := by
  induction values with
  | nil => simp
  | cons value rest inductionHypothesis =>
      have valueBound : value ≤ max value (rest.foldr max 0) :=
        Nat.le_max_left _ _
      have restMaxBound : rest.foldr max 0 ≤
          max value (rest.foldr max 0) := Nat.le_max_right _ _
      calc
        (value :: rest).sum = value + rest.sum := by simp
        _ ≤ max value (rest.foldr max 0) +
            rest.length * max value (rest.foldr max 0) :=
          Nat.add_le_add valueBound
            (inductionHypothesis.trans
              (Nat.mul_le_mul_left rest.length restMaxBound))
        _ = (value :: rest).length *
            (value :: rest).foldr max 0 := by
          simp [Nat.succ_mul, Nat.add_comm]

/-- The witness-region budget exposes both witness multiplicity and the
largest dependency/residual-bearing record width. -/
theorem witnessFieldBudget_le_witnessCount_mul_maxWidth (c : Certificate) :
    witnessFieldBudget c ≤
      c.edgeRoutes.length * maxWitnessRecordWidth c := by
  unfold witnessFieldBudget maxWitnessRecordWidth
  simpa using sum_le_length_mul_foldr_max
    (c.edgeRoutes.map fun witness => 1 + witnessFieldWidth witness)

theorem edgeWitnessMultiplicity_le_total (c : Certificate)
    (deduction : Deduction) :
    (edgeWitnessesFor c deduction).length ≤ c.edgeRoutes.length := by
  exact List.length_filter_le _ _

/-- A whole-certificate bound with flat witness-store multiplicity explicit. -/
def multiplicityExposedFieldBound (c : Certificate) : Nat :=
  graphFieldBudget c +
    c.edgeRoutes.length * maxWitnessRecordWidth c +
      terminalFieldBudget c

theorem certificateFieldBudget_le_multiplicityExposedFieldBound
    (c : Certificate) :
    certificateFieldBudget c <= multiplicityExposedFieldBound c := by
  have witnessBound :=
    witnessFieldBudget_le_witnessCount_mul_maxWidth c
  unfold certificateFieldBudget multiplicityExposedFieldBound
  omega

/-- A fixed-edge query must retain every mutable witness record, since a free
assignment can change any record's edge key to the queried edge. -/
theorem edgeWitnessCoverage_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat) (deduction : Deduction) :
    (canonicalSemanticReadSet d c
      (.expandedEdgeWitnessCoverage slot deduction)).card <=
        c.edgeRoutes.length * maxWitnessRecordWidth c :=
  (edgeWitnessCoverage_readSet_card_le d c slot deduction).trans
    (witnessFieldBudget_le_witnessCount_mul_maxWidth c)

theorem localEdgeWitnessCoverage_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat) (deduction : Deduction) :
    (canonicalSemanticReadSet d c
      (.localEdgeWitnessCoverage slot deduction)).card <=
        c.edgeRoutes.length * maxWitnessRecordWidth c :=
  (localEdgeWitnessCoverage_readSet_card_le d c slot deduction).trans
    (witnessFieldBudget_le_witnessCount_mul_maxWidth c)

theorem witnessFaithful_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat)
    (witness : EdgeRouteWitness) :
    (canonicalSemanticReadSet d c (.witnessFaithful slot witness)).card <=
      multiplicityExposedFieldBound c :=
  (witnessFaithful_readSet_card_le d c slot witness).trans
    (certificateFieldBudget_le_multiplicityExposedFieldBound c)

theorem nodeProduction_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.nodeProduction slot vertex)).card <=
      multiplicityExposedFieldBound c :=
  (nodeProduction_readSet_card_le d c slot vertex).trans
    (certificateFieldBudget_le_multiplicityExposedFieldBound c)

theorem routeCoverage_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.routeCoverage slot vertex)).card <=
      multiplicityExposedFieldBound c :=
  (routeCoverage_readSet_card_le d c slot vertex).trans
    (certificateFieldBudget_le_multiplicityExposedFieldBound c)

theorem tailClosure_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.tailClosure slot vertex)).card <=
      multiplicityExposedFieldBound c :=
  (tailClosure_readSet_card_le d c slot vertex).trans
    (certificateFieldBudget_le_multiplicityExposedFieldBound c)

theorem rootDischarge_readSet_card_le_multiplicity
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (canonicalSemanticReadSet d c (.rootDischarge slot vertex)).card <=
      multiplicityExposedFieldBound c :=
  (rootDischarge_readSet_card_le d c slot vertex).trans
    (certificateFieldBudget_le_multiplicityExposedFieldBound c)

/-- Maximum of the proved structural bound over the finite semantic
serialization. -/
def maxSemanticSupportBound (d : Graph) (c : Certificate) : Nat :=
  (semanticPositions d c).toFinset.fold max 0 fun position =>
    canonicalSemanticReadBound d c position

theorem canonicalSemanticReadBound_le_max (d : Graph) (c : Certificate)
    {position : VerificationPosition}
    (member : position ∈ semanticPositions d c) :
    canonicalSemanticReadBound d c position ≤ maxSemanticSupportBound d c := by
  rw [maxSemanticSupportBound, Finset.le_fold_max]
  exact Or.inr ⟨position, List.mem_toFinset.mpr member, le_rfl⟩

theorem canonicalSupportOfSemanticPosition_card_le_max (d : Graph)
    (c : Certificate) {position : VerificationPosition}
    (member : position ∈ semanticPositions d c) :
    (canonicalSupportOfSemanticPosition d c position).card ≤
      maxSemanticSupportBound d c :=
  (canonicalSupportOfSemanticPosition_card_le d c position).trans
    (canonicalSemanticReadBound_le_max d c member)

theorem maxSemanticSupportBound_le_fieldBudget (d : Graph)
    (c : Certificate) :
    maxSemanticSupportBound d c ≤ certificateFieldBudget c := by
  unfold maxSemanticSupportBound
  rw [Finset.fold_max_le]
  constructor
  · omega
  · intro position _
    cases position <;>
      simp [canonicalSemanticReadBound, certificateFieldBudget] <;> omega

theorem maxSemanticSupportBound_le_multiplicityExposed (d : Graph)
    (c : Certificate) :
    maxSemanticSupportBound d c <= multiplicityExposedFieldBound c :=
  (maxSemanticSupportBound_le_fieldBudget d c).trans
    (certificateFieldBudget_le_multiplicityExposedFieldBound c)

/-! ## Density for stable semantic failures

This is the certificate-specific specialization of the existing finite-product
orbit argument.  It is stated here because the older `MalformationWitness`
stores a numeric verifier offset and therefore cannot express a stable
semantic position across decoded list-length changes. -/

abbrev CanonicalSemanticOrbit (d : Graph) (c : Certificate)
    (position : VerificationPosition) :=
  { other : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c) //
    AgreeOn (canonicalSupportOfSemanticPosition d c position)
      (encodeCertificate d c) other }

abbrev CanonicalSemanticFailureAssignments (d : Graph) (c : Certificate)
    (position : VerificationPosition) :=
  { input : Assignment (CanonicalCoord d c) (CanonicalAlphabet d c) //
    canonicalSemanticFailureB d c position input = true }

noncomputable instance canonicalSemanticOrbitFintype (d : Graph)
    (c : Certificate) (position : VerificationPosition) :
    Fintype (CanonicalSemanticOrbit d c position) := Fintype.ofFinite _

noncomputable instance canonicalSemanticFailureAssignmentsFintype
    (d : Graph) (c : Certificate) (position : VerificationPosition) :
    Fintype (CanonicalSemanticFailureAssignments d c position) :=
  Fintype.ofFinite _

def canonicalSemanticOrbitEquiv (d : Graph) (c : Certificate)
    (position : VerificationPosition) :
    CanonicalSemanticOrbit d c position ≃
      (↥(canonicalSupportOfSemanticPosition d c position)ᶜ →
        CanonicalAlphabet d c) where
  toFun other coordinate := other.1 coordinate.1
  invFun free :=
    ⟨fun coordinate =>
        if member : coordinate ∈
            canonicalSupportOfSemanticPosition d c position then
          encodeCertificate d c coordinate
        else free ⟨coordinate, by simpa using member⟩,
      by
        intro coordinate member
        simp [member]⟩
  left_inv other := by
    apply Subtype.ext
    funext coordinate
    by_cases member : coordinate ∈
        canonicalSupportOfSemanticPosition d c position
    · simp [member, other.2 coordinate member]
    · simp [member]
  right_inv free := by
    funext coordinate
    have outside : coordinate.1 ∉
        canonicalSupportOfSemanticPosition d c position :=
      Finset.mem_compl.mp coordinate.2
    simp [outside]

theorem canonicalSemanticOrbit_card (d : Graph) (c : Certificate)
    (position : VerificationPosition) :
    Fintype.card (CanonicalSemanticOrbit d c position) =
      Fintype.card (CanonicalAlphabet d c) ^
        (Fintype.card (CanonicalCoord d c) -
          (canonicalSupportOfSemanticPosition d c position).card) := by
  calc
    Fintype.card (CanonicalSemanticOrbit d c position) =
        Fintype.card
          (↥(canonicalSupportOfSemanticPosition d c position)ᶜ →
            CanonicalAlphabet d c) :=
      Fintype.card_congr (canonicalSemanticOrbitEquiv d c position)
    _ = Fintype.card (CanonicalAlphabet d c) ^
          Fintype.card
            ↥(canonicalSupportOfSemanticPosition d c position)ᶜ :=
      Fintype.card_fun
    _ = Fintype.card (CanonicalAlphabet d c) ^
          (canonicalSupportOfSemanticPosition d c position)ᶜ.card := by
      congr 1
      exact Fintype.card_coe _
    _ = Fintype.card (CanonicalAlphabet d c) ^
        (Fintype.card (CanonicalCoord d c) -
          (canonicalSupportOfSemanticPosition d c position).card) := by
      congr 1
      exact Finset.card_compl _

def canonicalSemanticOrbitToFailures (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    CanonicalSemanticOrbit d c position →
      CanonicalSemanticFailureAssignments d c position := by
  intro other
  refine ⟨other.1, ?_⟩
  have invariant := supportOfOracle_agreement
    (canonicalSemanticFailureB d c position)
    (encodeCertificate d c) other.1 other.2
  rw [invariant]
  unfold canonicalSemanticFailureB
  rw [decodeCertificate_encodeCertificate]
  unfold SemanticMarkedAt at failure
  rw [failure]
  rfl

theorem canonicalSemanticOrbitToFailures_injective (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    Function.Injective
      (canonicalSemanticOrbitToFailures d c position failure) := by
  intro left right equality
  apply Subtype.ext
  simpa [canonicalSemanticOrbitToFailures] using
    congrArg Subtype.val equality

theorem canonicalSemanticFailureAssignments_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    Fintype.card (CanonicalAlphabet d c) ^
        (Fintype.card (CanonicalCoord d c) -
          (canonicalSupportOfSemanticPosition d c position).card) ≤
      Fintype.card (CanonicalSemanticFailureAssignments d c position) := by
  rw [← canonicalSemanticOrbit_card d c position]
  exact Fintype.card_le_of_injective
    (canonicalSemanticOrbitToFailures d c position failure)
    (canonicalSemanticOrbitToFailures_injective d c position failure)

noncomputable def canonicalSemanticMalformationDensity (d : Graph)
    (c : Certificate) (position : VerificationPosition) : ℚ :=
  Fintype.card (CanonicalSemanticFailureAssignments d c position) /
    Fintype.card (CanonicalAlphabet d c) ^
      Fintype.card (CanonicalCoord d c)

theorem canonicalSemanticMarked_density_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    (1 : ℚ) / Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfSemanticPosition d c position).card ≤
      canonicalSemanticMalformationDensity d c position := by
  have alphabetPositive : 0 < Fintype.card (CanonicalAlphabet d c) :=
    Fintype.card_pos
  have supportLe :
      (canonicalSupportOfSemanticPosition d c position).card ≤
        Fintype.card (CanonicalCoord d c) := Finset.card_le_univ _
  have lower := canonicalSemanticFailureAssignments_lower_bound
    d c position failure
  have cross :
      Fintype.card (CanonicalAlphabet d c) ^
          Fintype.card (CanonicalCoord d c) ≤
        Fintype.card (CanonicalAlphabet d c) ^
            (canonicalSupportOfSemanticPosition d c position).card *
          Fintype.card
            (CanonicalSemanticFailureAssignments d c position) := by
    calc
      Fintype.card (CanonicalAlphabet d c) ^
          Fintype.card (CanonicalCoord d c) =
        Fintype.card (CanonicalAlphabet d c) ^
            (canonicalSupportOfSemanticPosition d c position).card *
          Fintype.card (CanonicalAlphabet d c) ^
            (Fintype.card (CanonicalCoord d c) -
              (canonicalSupportOfSemanticPosition d c position).card) := by
        rw [← pow_add, Nat.add_sub_of_le supportLe]
      _ ≤ _ := Nat.mul_le_mul_left _ lower
  unfold canonicalSemanticMalformationDensity
  apply (div_le_div_iff₀ (by positivity :
      (0 : ℚ) < Fintype.card (CanonicalAlphabet d c) ^
        (canonicalSupportOfSemanticPosition d c position).card)
    (by positivity : (0 : ℚ) < Fintype.card (CanonicalAlphabet d c) ^
      Fintype.card (CanonicalCoord d c))).2
  norm_num
  exact_mod_cast (by simpa [Nat.mul_comm] using cross)

theorem canonicalSemanticMarked_bound_density_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    (1 : ℚ) / Fintype.card (CanonicalAlphabet d c) ^
        canonicalSemanticReadBound d c position ≤
      canonicalSemanticMalformationDensity d c position := by
  have exactLower :=
    canonicalSemanticMarked_density_lower_bound d c position failure
  have supportLe := canonicalSupportOfSemanticPosition_card_le d c position
  have alphabetOne :
      (1 : ℚ) ≤ Fintype.card (CanonicalAlphabet d c) := by
    exact_mod_cast (Fintype.card_pos :
      0 < Fintype.card (CanonicalAlphabet d c))
  have denominatorLe :
      (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
          (canonicalSupportOfSemanticPosition d c position).card ≤
        (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
          canonicalSemanticReadBound d c position :=
    pow_le_pow_right₀ alphabetOne supportLe
  have positive :
      (0 : ℚ) < (Fintype.card (CanonicalAlphabet d c) : ℚ) ^
        (canonicalSupportOfSemanticPosition d c position).card := by
    positivity
  exact (one_div_le_one_div_of_le positive denominatorLe).trans exactLower

/-- Complete formation invalidity yields an unconditional exact density bound
for a genuine stable semantic failure. -/
theorem formation_invalid_semantic_amplification (d : Graph)
    (c : Certificate) (invalid : ¬ FormationValid d c) :
    ∃ position, position ∈ semanticPositions d c ∧
      SemanticMarkedAt d c position ∧
      (canonicalSupportOfSemanticPosition d c position).card ≤
        canonicalSemanticReadBound d c position ∧
      canonicalSemanticReadBound d c position ≤
        maxSemanticSupportBound d c ∧
      (1 : ℚ) / Fintype.card (CanonicalAlphabet d c) ^
          canonicalSemanticReadBound d c position ≤
        canonicalSemanticMalformationDensity d c position := by
  obtain ⟨position, member, failure⟩ :=
    formation_invalid_implies_semanticMarked d c invalid
  exact ⟨position, member, failure,
    canonicalSupportOfSemanticPosition_card_le d c position,
    by
      unfold maxSemanticSupportBound
      rw [Finset.le_fold_max]
      exact Or.inr ⟨position, List.mem_toFinset.mpr member, le_rfl⟩,
    canonicalSemanticMarked_bound_density_lower_bound d c position failure⟩

/-- Complete formation invalidity with the flat-store multiplicity obstruction
substituted directly into the density exponent. -/
theorem formation_invalid_semantic_multiplicity_amplification
    (d : Graph) (c : Certificate) (invalid : Not (FormationValid d c)) :
    Exists fun position =>
      And (List.Mem position (semanticPositions d c))
        (And (SemanticMarkedAt d c position)
          (And
            ((canonicalSupportOfSemanticPosition d c position).card <=
              multiplicityExposedFieldBound c)
            ((1 : Rat) / Fintype.card (CanonicalAlphabet d c) ^
                multiplicityExposedFieldBound c <=
              canonicalSemanticMalformationDensity d c position))) := by
  have result := formation_invalid_semantic_amplification d c invalid
  let position := Classical.choose result
  have facts := Classical.choose_spec result
  have exponentBound : canonicalSemanticReadBound d c position <=
      multiplicityExposedFieldBound c :=
    facts.2.2.2.1.trans
      (maxSemanticSupportBound_le_multiplicityExposed d c)
  have supportExposed := facts.2.2.1.trans exponentBound
  have alphabetPositive : 0 < Fintype.card (CanonicalAlphabet d c) :=
    Fintype.card_pos
  have denominatorBoundNat :
      Fintype.card (CanonicalAlphabet d c) ^
          canonicalSemanticReadBound d c position <=
        Fintype.card (CanonicalAlphabet d c) ^
          multiplicityExposedFieldBound c :=
    Nat.pow_le_pow_right alphabetPositive exponentBound
  have denominatorBound :
      (Fintype.card (CanonicalAlphabet d c) : Rat) ^
          canonicalSemanticReadBound d c position <=
        (Fintype.card (CanonicalAlphabet d c) : Rat) ^
          multiplicityExposedFieldBound c := by
    exact_mod_cast denominatorBoundNat
  have positive :
      (0 : Rat) < (Fintype.card (CanonicalAlphabet d c) : Rat) ^
        canonicalSemanticReadBound d c position := by
    positivity
  refine Exists.intro position ?_
  exact And.intro facts.1
    (And.intro facts.2.1
      (And.intro supportExposed
        ((one_div_le_one_div_of_le positive denominatorBound).trans
          facts.2.2.2.2)))


end Robustness
end CertificateSearch
