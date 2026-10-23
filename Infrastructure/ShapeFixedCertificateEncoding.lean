import Infrastructure.CanonicalCertificateEncoding










open Semantic FlowSpec CertificateModel

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace SearchProblem
namespace CertificateRobustness




def shapeFixedPayloadRaw : CanonicalRawCoord → Bool
  | .witnessDependency _ _ => true
  | .witnessResidual _ _ => true
  | .terminalDependency _ _ => true
  | .terminalResidual _ _ => true
  | _ => false



abbrev ShapeFixedCoord (d : Graph) (c : Certificate) :=
  { coordinate : CanonicalCoord d c //
    shapeFixedPayloadRaw coordinate.1.1 = true }


abbrev ShapeFixedAlphabet (d : Graph) (c : Certificate) :=
  CanonicalAlphabet d c

abbrev ShapeFixedAssignment (d : Graph) (c : Certificate) :=
  Assignment (ShapeFixedCoord d c) (ShapeFixedAlphabet d c)



def shapeFixedToCanonical (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) :
    Assignment (CanonicalCoord d c) (CanonicalAlphabet d c) :=
  fun coordinate =>
    if payload : shapeFixedPayloadRaw coordinate.1.1 = true then
      input ⟨coordinate, payload⟩
    else encodeCertificate d c coordinate

theorem shapeFixedToCanonical_injective (d : Graph) (c : Certificate) :
    Function.Injective (shapeFixedToCanonical d c) := by
  intro left right equality
  funext coordinate
  have atCoordinate := congrFun equality coordinate.1
  simpa [shapeFixedToCanonical, coordinate.2] using atCoordinate


def encodeShapeFixed (d : Graph) (c : Certificate) :
    ShapeFixedAssignment d c :=
  fun coordinate => encodeCertificate d c coordinate.1



def decodeShapeFixed (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) : Certificate :=
  decodeCertificate d c (shapeFixedToCanonical d c input)

@[simp] theorem shapeFixedToCanonical_encodeShapeFixed (d : Graph)
    (c : Certificate) :
    shapeFixedToCanonical d c (encodeShapeFixed d c) =
      encodeCertificate d c := by
  funext coordinate
  by_cases payload : shapeFixedPayloadRaw coordinate.1.1 = true
  · simp [shapeFixedToCanonical, payload, encodeShapeFixed]
  · simp [shapeFixedToCanonical, payload]

@[simp] theorem decodeShapeFixed_encodeShapeFixed (d : Graph)
    (c : Certificate) :
    decodeShapeFixed d c (encodeShapeFixed d c) = c := by
  unfold decodeShapeFixed
  rw [shapeFixedToCanonical_encodeShapeFixed]
  exact decodeCertificate_encodeCertificate d c


def canonicalStructuralReadSet (d : Graph) (c : Certificate) :
    Finset (CanonicalCoord d c) :=
  Finset.univ.filter fun coordinate =>
    shapeFixedPayloadRaw coordinate.1.1 = false



theorem shapeFixedToCanonical_preserves_structural_coordinates
    (d : Graph) (c : Certificate) (input : ShapeFixedAssignment d c) :
    AgreeOn (canonicalStructuralReadSet d c)
      (encodeCertificate d c) (shapeFixedToCanonical d c input) := by
  intro coordinate member
  have structural : shapeFixedPayloadRaw coordinate.1.1 = false := by
    simpa [canonicalStructuralReadSet] using member
  simp [shapeFixedToCanonical, structural]

theorem payloadRaw_not_graph {coordinate : CanonicalRawCoord}
    (payload : shapeFixedPayloadRaw coordinate = true) :
    canonicalRawCoordRegion coordinate ≠ .graph := by
  cases coordinate <;>
    simp_all [shapeFixedPayloadRaw, canonicalRawCoordRegion]



@[simp] theorem decodeShapeFixed_graph (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).graph = c.graph := by
  unfold decodeShapeFixed
  have agree : AgreeOn (canonicalRegionReadSet d c .graph)
      (shapeFixedToCanonical d c input) (encodeCertificate d c) := by
    intro coordinate member
    have graphOwned : canonicalRawCoordRegion coordinate.1.1 = .graph := by
      simpa [canonicalRegionReadSet] using member
    have structural : shapeFixedPayloadRaw coordinate.1.1 = false := by
      by_contra notFalse
      have payload : shapeFixedPayloadRaw coordinate.1.1 = true := by
        cases value : shapeFixedPayloadRaw coordinate.1.1 <;> simp_all
      exact payloadRaw_not_graph payload graphOwned
    simp [shapeFixedToCanonical, structural]
  have congruence := decodeCertificate_graph_congr d c
    (shapeFixedToCanonical d c input) (encodeCertificate d c) agree
  rw [decodeCertificate_encodeCertificate] at congruence
  exact congruence



theorem decodeCertificate_shapeFixedToCanonical (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) :
    decodeCertificate d c (shapeFixedToCanonical d c input) =
      decodeShapeFixed d c input := rfl

@[simp] theorem readCanonicalAtom_shapeFixed_structural (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c)
    (coordinate : CanonicalRawCoord) (fallback : CanonicalAtom)
    (structural : shapeFixedPayloadRaw coordinate = false) :
    readCanonicalAtom d c (shapeFixedToCanonical d c input)
      coordinate fallback = fallback := by
  unfold readCanonicalAtom
  split
  · simp [shapeFixedToCanonical, structural, encodeCertificate]
  · rfl

@[simp] theorem readFlag_shapeFixed_structural (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c)
    (coordinate : CanonicalRawCoord) (fallback : Bool)
    (structural : shapeFixedPayloadRaw coordinate = false) :
    readFlag d c (shapeFixedToCanonical d c input) coordinate fallback =
      fallback := by
  simp [readFlag, structural]

@[simp] theorem readNatural_shapeFixed_structural (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c)
    (coordinate : CanonicalRawCoord) (fallback : Nat)
    (structural : shapeFixedPayloadRaw coordinate = false) :
    readNatural d c (shapeFixedToCanonical d c input) coordinate fallback =
      fallback := by
  simp [readNatural, structural]

@[simp] theorem readFormula_shapeFixed_structural (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c)
    (coordinate : CanonicalRawCoord) (fallback : Formula)
    (structural : shapeFixedPayloadRaw coordinate = false) :
    readFormula d c (shapeFixedToCanonical d c input) coordinate fallback =
      fallback := by
  simp [readFormula, structural]

@[simp] theorem decodeVertex_shapeFixed (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c)
    (occurrence : CanonicalVertexOccurrence) (vertex : Vertex) :
    decodeVertex d c (shapeFixedToCanonical d c input) occurrence vertex =
      vertex := by
  unfold decodeVertex
  congr 1 <;> simp [shapeFixedPayloadRaw]

@[simp] theorem decodeDeduction_shapeFixed (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c)
    (occurrence : CanonicalDeductionOccurrence) (edge : Deduction) :
    decodeDeduction d c (shapeFixedToCanonical d c input) occurrence edge =
      edge := by
  unfold decodeDeduction
  congr 1 <;> simp [shapeFixedPayloadRaw]

@[simp] theorem decodeWitness_shapeFixed_edge (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) (index : Nat)
    (witness : EdgeRouteWitness) :
    (decodeWitness d c (shapeFixedToCanonical d c input) index witness).edge =
      witness.edge := by
  simp [decodeWitness]

@[simp] theorem decodeTerminal_shapeFixed_node (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) (index : Nat)
    (output : TerminalOutput) :
    (decodeTerminal d c (shapeFixedToCanonical d c input) index output).node =
      output.node := by
  simp [decodeTerminal]

theorem decodeShapeFixed_edgeRoutes_eq (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).edgeRoutes =
      c.edgeRoutes.zipIdx.map fun entry =>
        decodeWitness d c (shapeFixedToCanonical d c input)
          entry.2 entry.1 := by
  simp [decodeShapeFixed, decodeCertificate, shapeFixedPayloadRaw]

theorem decodeShapeFixed_terminalOutputs_eq (d : Graph) (c : Certificate)
    (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).terminalOutputs =
      c.terminalOutputs.zipIdx.map fun entry =>
        decodeTerminal d c (shapeFixedToCanonical d c input)
          entry.2 entry.1 := by
  simp [decodeShapeFixed, decodeCertificate, shapeFixedPayloadRaw]

@[simp] theorem decodeShapeFixed_witnessOwners (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).edgeRoutes.map (·.edge) =
      c.edgeRoutes.map (·.edge) := by
  rw [decodeShapeFixed_edgeRoutes_eq]
  have loop : ∀ (records : List EdgeRouteWitness) (start : Nat),
      (records.zipIdx start).map (fun entry => entry.1.edge) =
        records.map (·.edge) := by
    intro records start
    induction records generalizing start with
    | nil => rfl
    | cons record records induction =>
        simp [List.zipIdx, induction (start := start + 1)]
  rw [List.map_map]
  change (c.edgeRoutes.zipIdx.map fun entry =>
      (decodeWitness d c (shapeFixedToCanonical d c input)
        entry.2 entry.1).edge) = c.edgeRoutes.map (·.edge)
  simpa only [decodeWitness_shapeFixed_edge] using loop c.edgeRoutes 0

@[simp] theorem decodeShapeFixed_terminalOwners (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).terminalOutputs.map (·.node) =
      c.terminalOutputs.map (·.node) := by
  rw [decodeShapeFixed_terminalOutputs_eq]
  have loop : ∀ (records : List TerminalOutput) (start : Nat),
      (records.zipIdx start).map (fun entry => entry.1.node) =
        records.map (·.node) := by
    intro records start
    induction records generalizing start with
    | nil => rfl
    | cons record records induction =>
        simp [List.zipIdx, induction (start := start + 1)]
  rw [List.map_map]
  change (c.terminalOutputs.zipIdx.map fun entry =>
      (decodeTerminal d c (shapeFixedToCanonical d c input)
        entry.2 entry.1).node) = c.terminalOutputs.map (·.node)
  simpa only [decodeTerminal_shapeFixed_node] using loop c.terminalOutputs 0

@[simp] theorem decodeShapeFixed_edgeRoutes_length (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).edgeRoutes.length = c.edgeRoutes.length := by
  simp [decodeShapeFixed, decodeCertificate, shapeFixedPayloadRaw]

@[simp] theorem decodeShapeFixed_terminalOutputs_length (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c) :
    (decodeShapeFixed d c input).terminalOutputs.length =
      c.terminalOutputs.length := by
  simp [decodeShapeFixed, decodeCertificate, shapeFixedPayloadRaw]



def shapeFixedRawIndex : CanonicalRawCoord → Option (Bool × Nat)
  | .witnessDependency index _ => some (false, index)
  | .witnessResidual index _ => some (false, index)
  | .terminalDependency index _ => some (true, index)
  | .terminalResidual index _ => some (true, index)
  | _ => none

def witnessPayloadOwnerRelevantB (c : Certificate) (vertices : List Vertex)
    (index : Nat) : Bool :=
  match c.edgeRoutes[index]? with
  | some witness => vertices.contains witness.edge.START
  | none => false

def terminalPayloadOwnerRelevantB (c : Certificate) (vertices : List Vertex)
    (index : Nat) : Bool :=
  match c.terminalOutputs[index]? with
  | some output => vertices.contains output.node
  | none => false

def shapeFixedRawRelevantAtVerticesB (c : Certificate)
    (vertices : List Vertex) (coordinate : CanonicalRawCoord) : Bool :=
  match shapeFixedRawIndex coordinate with
  | some (false, index) => witnessPayloadOwnerRelevantB c vertices index
  | some (true, index) => terminalPayloadOwnerRelevantB c vertices index
  | none => false

def shapeFixedCoordRelevantAtVerticesB (c : Certificate)
    (vertices : List Vertex) (coordinate : ShapeFixedCoord d c) : Bool :=
  shapeFixedRawRelevantAtVerticesB c vertices coordinate.1.1.1



def shapeFixedVertexReadSet (d : Graph) (c : Certificate)
    (vertices : List Vertex) : Finset (ShapeFixedCoord d c) :=
  Finset.univ.filter fun coordinate =>
    shapeFixedCoordRelevantAtVerticesB c vertices coordinate

theorem shapeFixedVertexReadSet_mono (d : Graph) (c : Certificate)
    {small large : List Vertex}
    (subset : ∀ vertex ∈ small, vertex ∈ large) :
    shapeFixedVertexReadSet d c small ⊆
      shapeFixedVertexReadSet d c large := by
  intro coordinate member
  simp only [shapeFixedVertexReadSet, Finset.mem_filter,
    Finset.mem_univ, true_and] at member ⊢
  unfold shapeFixedCoordRelevantAtVerticesB shapeFixedRawRelevantAtVerticesB at member ⊢
  cases owner : shapeFixedRawIndex coordinate.1.1.1 with
  | none => simp [owner] at member
  | some tagged =>
      rcases tagged with ⟨terminal, index⟩
      cases terminal
      · simp only [owner] at member ⊢
        unfold witnessPayloadOwnerRelevantB at member ⊢
        cases getWitness : c.edgeRoutes[index]? with
        | none => simp [getWitness] at member
        | some witness =>
            simp only [getWitness] at member ⊢
            simpa using subset witness.edge.START (by simpa using member)
      · simp only [owner] at member ⊢
        unfold terminalPayloadOwnerRelevantB at member ⊢
        cases getOutput : c.terminalOutputs[index]? with
        | none => simp [getOutput] at member
        | some output =>
            simp only [getOutput] at member ⊢
            simpa using subset output.node (by simpa using member)

theorem agreeOn_shapeFixedVertexReadSet_mono
    (d : Graph) (c : Certificate) {small large : List Vertex}
    (subset : ∀ vertex ∈ small, vertex ∈ large)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c large) left right) :
    AgreeOn (shapeFixedVertexReadSet d c small) left right := by
  intro coordinate member
  exact agree coordinate (shapeFixedVertexReadSet_mono d c subset member)

theorem readCanonicalAtom_shapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (coordinate : CanonicalRawCoord) (fallback : CanonicalAtom)
    (payload : shapeFixedPayloadRaw coordinate = true)
    (relevant : match shapeFixedRawIndex coordinate with
      | some (false, index) => witnessPayloadOwnerRelevantB c vertices index
      | some (true, index) => terminalPayloadOwnerRelevantB c vertices index
      | none => false) :
    readCanonicalAtom d c (shapeFixedToCanonical d c left)
        coordinate fallback =
      readCanonicalAtom d c (shapeFixedToCanonical d c right)
        coordinate fallback := by
  unfold readCanonicalAtom
  split
  · rename_i member
    let canonical : CanonicalCoord d c := ⟨(coordinate, fallback), member⟩
    let fixed : ShapeFixedCoord d c := ⟨canonical, payload⟩
    have fixedMember : fixed ∈ shapeFixedVertexReadSet d c vertices := by
      simp only [shapeFixedVertexReadSet, Finset.mem_filter,
        Finset.mem_univ, true_and]
      unfold shapeFixedCoordRelevantAtVerticesB shapeFixedRawRelevantAtVerticesB
      simp only [fixed, canonical]
      cases owner : shapeFixedRawIndex coordinate with
      | none => simp [owner] at relevant
      | some tagged =>
          rcases tagged with ⟨terminal, index⟩
          cases terminal <;> simp_all
    have equality := agree fixed fixedMember
    have valueEquality := congrArg Subtype.val equality
    simpa [shapeFixedToCanonical, payload, fixed, canonical] using
      valueEquality.symm
  · rfl

theorem readFormula_shapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (coordinate : CanonicalRawCoord) (fallback : Formula)
    (payload : shapeFixedPayloadRaw coordinate = true)
    (relevant : match shapeFixedRawIndex coordinate with
      | some (false, index) => witnessPayloadOwnerRelevantB c vertices index
      | some (true, index) => terminalPayloadOwnerRelevantB c vertices index
      | none => false) :
    readFormula d c (shapeFixedToCanonical d c left) coordinate fallback =
      readFormula d c (shapeFixedToCanonical d c right) coordinate fallback := by
  unfold readFormula
  rw [readCanonicalAtom_shapeFixed_congr d c vertices left right agree
    coordinate (.formula fallback) payload relevant]

theorem readNatural_shapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (coordinate : CanonicalRawCoord) (fallback : Nat)
    (payload : shapeFixedPayloadRaw coordinate = true)
    (relevant : match shapeFixedRawIndex coordinate with
      | some (false, index) => witnessPayloadOwnerRelevantB c vertices index
      | some (true, index) => terminalPayloadOwnerRelevantB c vertices index
      | none => false) :
    readNatural d c (shapeFixedToCanonical d c left) coordinate fallback =
      readNatural d c (shapeFixedToCanonical d c right) coordinate fallback := by
  unfold readNatural
  rw [readCanonicalAtom_shapeFixed_congr d c vertices left right agree
    coordinate (.natural fallback) payload relevant]

theorem decodeWitness_shapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (index : Nat) (witness : EdgeRouteWitness)
    (relevant : witnessPayloadOwnerRelevantB c vertices index = true) :
    decodeWitness d c (shapeFixedToCanonical d c left) index witness =
      decodeWitness d c (shapeFixedToCanonical d c right) index witness := by
  unfold decodeWitness
  congr 1
  · apply List.filterMap_congr
    intro entry _
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, entryIndex⟩
    simp only
    rw [readFlag_shapeFixed_structural d c left
      (.witnessDependencyPresent index entryIndex) anchorPresent (by rfl)]
    rw [readFlag_shapeFixed_structural d c right
      (.witnessDependencyPresent index entryIndex) anchorPresent (by rfl)]
    by_cases present : anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readFormula_shapeFixed_congr d c vertices left right agree
        (.witnessDependency index entryIndex) anchorValue (by rfl)
        (by simpa [shapeFixedRawIndex] using relevant)
    · simp [present]
  · apply List.filterMap_congr
    intro entry _
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, entryIndex⟩
    simp only
    rw [readFlag_shapeFixed_structural d c left
      (.witnessResidualPresent index entryIndex) anchorPresent (by rfl)]
    rw [readFlag_shapeFixed_structural d c right
      (.witnessResidualPresent index entryIndex) anchorPresent (by rfl)]
    by_cases present : anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readNatural_shapeFixed_congr d c vertices left right agree
        (.witnessResidual index entryIndex) anchorValue (by rfl)
        (by simpa [shapeFixedRawIndex] using relevant)
    · simp [present]

theorem decodeTerminal_shapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (index : Nat) (output : TerminalOutput)
    (relevant : terminalPayloadOwnerRelevantB c vertices index = true) :
    decodeTerminal d c (shapeFixedToCanonical d c left) index output =
      decodeTerminal d c (shapeFixedToCanonical d c right) index output := by
  unfold decodeTerminal
  congr 1
  · apply List.filterMap_congr
    intro entry _
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, entryIndex⟩
    simp only
    rw [readFlag_shapeFixed_structural d c left
      (.terminalDependencyPresent index entryIndex) anchorPresent (by rfl)]
    rw [readFlag_shapeFixed_structural d c right
      (.terminalDependencyPresent index entryIndex) anchorPresent (by rfl)]
    by_cases present : anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readFormula_shapeFixed_congr d c vertices left right agree
        (.terminalDependency index entryIndex) anchorValue (by rfl)
        (by simpa [shapeFixedRawIndex] using relevant)
    · simp [present]
  · apply List.filterMap_congr
    intro entry _
    rcases entry with ⟨⟨anchorPresent, anchorValue⟩, entryIndex⟩
    simp only
    rw [readFlag_shapeFixed_structural d c left
      (.terminalResidualPresent index entryIndex) anchorPresent (by rfl)]
    rw [readFlag_shapeFixed_structural d c right
      (.terminalResidualPresent index entryIndex) anchorPresent (by rfl)]
    by_cases present : anchorPresent = true
    · simp only [present, if_true, Option.some.injEq]
      exact readNatural_shapeFixed_congr d c vertices left right agree
        (.terminalResidual index entryIndex) anchorValue (by rfl)
        (by simpa [shapeFixedRawIndex] using relevant)
    · simp [present]

theorem witnessPayloadOwnerRelevantB_of_mem_zipIdx
    (c : Certificate) (vertices : List Vertex)
    (witness : EdgeRouteWitness) (index : Nat)
    (member : (witness, index) ∈ c.edgeRoutes.zipIdx)
    (owner : vertices.contains witness.edge.START = true) :
    witnessPayloadOwnerRelevantB c vertices index = true := by
  have facts := List.mem_zipIdx member
  simp only [Nat.zero_le, true_and, zero_add, Nat.sub_zero] at facts
  have getValue : c.edgeRoutes[index]? = some witness := by
    rw [List.getElem?_eq_getElem facts.1]
    exact congrArg some facts.2.symm
  unfold witnessPayloadOwnerRelevantB
  rw [getValue]
  exact owner

theorem terminalPayloadOwnerRelevantB_of_mem_zipIdx
    (c : Certificate) (vertices : List Vertex)
    (output : TerminalOutput) (index : Nat)
    (member : (output, index) ∈ c.terminalOutputs.zipIdx)
    (owner : vertices.contains output.node = true) :
    terminalPayloadOwnerRelevantB c vertices index = true := by
  have facts := List.mem_zipIdx member
  simp only [Nat.zero_le, true_and, zero_add, Nat.sub_zero] at facts
  have getValue : c.terminalOutputs[index]? = some output := by
    rw [List.getElem?_eq_getElem facts.1]
    exact congrArg some facts.2.symm
  unfold terminalPayloadOwnerRelevantB
  rw [getValue]
  exact owner



theorem outputsAt_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (vertex : Vertex) (vertexMember : vertex ∈ vertices) :
    outputsAt (decodeShapeFixed d c left) vertex =
      outputsAt (decodeShapeFixed d c right) vertex := by
  have edgeOutputs :
      ((decodeShapeFixed d c left).edgeRoutes.filterMap fun witness =>
          if witness.edge.START == vertex then
            some (witness.dependency, witness.residual)
          else none) =
        ((decodeShapeFixed d c right).edgeRoutes.filterMap fun witness =>
          if witness.edge.START == vertex then
            some (witness.dependency, witness.residual)
          else none) := by
    rw [decodeShapeFixed_edgeRoutes_eq, decodeShapeFixed_edgeRoutes_eq]
    rw [List.filterMap_map, List.filterMap_map]
    apply List.filterMap_congr
    intro entry member
    rcases entry with ⟨witness, index⟩
    simp only [Function.comp_apply, decodeWitness_shapeFixed_edge]
    by_cases ownerEq : witness.edge.START = vertex
    · have owner : vertices.contains witness.edge.START = true := by
        simp [ownerEq, vertexMember]
      have relevant := witnessPayloadOwnerRelevantB_of_mem_zipIdx
        c vertices witness index member owner
      rw [decodeWitness_shapeFixed_congr d c vertices left right agree
        index witness relevant]
    · simp [ownerEq]
  have terminalOutputs :
      ((decodeShapeFixed d c left).terminalOutputs.filterMap fun output =>
          if output.node == vertex then
            some (output.dependency, output.residual)
          else none) =
        ((decodeShapeFixed d c right).terminalOutputs.filterMap fun output =>
          if output.node == vertex then
            some (output.dependency, output.residual)
          else none) := by
    rw [decodeShapeFixed_terminalOutputs_eq,
      decodeShapeFixed_terminalOutputs_eq]
    rw [List.filterMap_map, List.filterMap_map]
    apply List.filterMap_congr
    intro entry member
    rcases entry with ⟨output, index⟩
    simp only [Function.comp_apply, decodeTerminal_shapeFixed_node]
    by_cases ownerEq : output.node = vertex
    · have owner : vertices.contains output.node = true := by
        simp [ownerEq, vertexMember]
      have relevant := terminalPayloadOwnerRelevantB_of_mem_zipIdx
        c vertices output index member owner
      rw [decodeTerminal_shapeFixed_congr d c vertices left right agree
        index output relevant]
    · simp [ownerEq]
  unfold outputsAt
  rw [edgeOutputs, terminalOutputs]

def localExpectedReadVertices (graph : Graph) (vertex : Vertex) : List Vertex :=
  match localShape graph vertex with
  | .source => predsOf graph vertex
  | .elim major minor => major :: minor :: predsOf graph vertex
  | .elimMulti pairs =>
      pairs.flatMap (fun pair => [pair.1, pair.2]) ++ predsOf graph vertex
  | .intro premise _ _ => premise :: predsOf graph vertex
  | .invalid => predsOf graph vertex

theorem preds_mem_localExpectedReadVertices (graph : Graph) (vertex u : Vertex)
    (member : u ∈ predsOf graph vertex) :
    u ∈ localExpectedReadVertices graph vertex := by
  unfold localExpectedReadVertices
  cases localShape graph vertex <;> simp [member]

theorem localExpected_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertex : Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn
      (shapeFixedVertexReadSet d c
        (localExpectedReadVertices c.graph vertex)) left right) :
    localExpected (decodeShapeFixed d c left) vertex =
      localExpected (decodeShapeFixed d c right) vertex := by
  let vertices := localExpectedReadVertices c.graph vertex
  have outputAgree : ∀ u ∈ vertices,
      outputsAt (decodeShapeFixed d c left) u =
        outputsAt (decodeShapeFixed d c right) u := by
    intro u member
    exact outputsAt_decodeShapeFixed_congr d c vertices left right agree u member
  have predAgree : ∀ u ∈ predsOf c.graph vertex,
      outputsAt (decodeShapeFixed d c left) u =
        outputsAt (decodeShapeFixed d c right) u := by
    intro u member
    exact outputAgree u (preds_mem_localExpectedReadVertices
      c.graph vertex u member)
  unfold localExpected
  rw [show (decodeShapeFixed d c left).graph = c.graph by simp,
    show (decodeShapeFixed d c right).graph = c.graph by simp]
  cases shape : localShape c.graph vertex with
  | source => simp []
  | invalid => simp []
  | elim major minor =>
      have majorMember : major ∈ vertices := by
        simp [vertices, localExpectedReadVertices, shape]
      have minorMember : minor ∈ vertices := by
        simp [vertices, localExpectedReadVertices, shape]
      simp only []
      unfold elimExpected mixedIntroExpected reseedExpected
      simp only [decodeShapeFixed_graph]
      rw [outputAgree minor minorMember, outputAgree major majorMember,
        mixedIntroPairs_congr_of_preds c.graph vertex _ _ predAgree,
        reseedPairs_congr_of_preds c.graph vertex _ _ predAgree]
  | elimMulti pairs =>
      have pairMember : ∀ pair ∈ pairs,
          pair.1 ∈ vertices ∧ pair.2 ∈ vertices := by
        intro pair member
        constructor
        · simp only [vertices, localExpectedReadVertices, shape,
            List.mem_append]
          apply Or.inl
          rw [List.mem_flatMap]
          exact ⟨pair, member, by simp⟩
        · simp only [vertices, localExpectedReadVertices, shape,
            List.mem_append]
          apply Or.inl
          rw [List.mem_flatMap]
          exact ⟨pair, member, by simp⟩
      simp only []
      unfold elimExpectedMany mixedIntroExpected reseedExpected
      simp only [decodeShapeFixed_graph]
      have elimEq :
          (pairs.flatMap fun pair =>
            elimCombine c.graph vertex pair.1 pair.2
              (outputsAt (decodeShapeFixed d c left) pair.2)
              (outputsAt (decodeShapeFixed d c left) pair.1)) =
          (pairs.flatMap fun pair =>
            elimCombine c.graph vertex pair.1 pair.2
              (outputsAt (decodeShapeFixed d c right) pair.2)
              (outputsAt (decodeShapeFixed d c right) pair.1)) := by
        apply List.flatMap_congr
        intro pair member
        rcases pair with ⟨major, minor⟩
        rw [outputAgree minor (pairMember (major, minor) member).2,
          outputAgree major (pairMember (major, minor) member).1]
      have mixedEq := mixedIntroPairs_congr_of_preds c.graph vertex _ _ predAgree
      have reseedEq := reseedPairs_congr_of_preds c.graph vertex _ _ predAgree
      rw [elimEq, mixedEq, reseedEq]
  | intro premise alpha beta =>
      have premiseMember : premise ∈ vertices := by
        simp [vertices, localExpectedReadVertices, shape]
      simp only []
      unfold introExpected reseedExpected
      simp only [decodeShapeFixed_graph]
      rw [outputAgree premise premiseMember,
        reseedPairs_congr_of_preds c.graph vertex _ _ predAgree]

def productionReadVertices (graph : Graph) (vertex : Vertex) : List Vertex :=
  vertex :: localExpectedReadVertices graph vertex

theorem localExpectedReadVertices_subset_productionReadVertices
    (graph : Graph) (vertex u : Vertex)
    (member : u ∈ localExpectedReadVertices graph vertex) :
    u ∈ productionReadVertices graph vertex := by
  simp [productionReadVertices, member]

theorem productionAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertex : Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn
      (shapeFixedVertexReadSet d c (productionReadVertices c.graph vertex))
        left right) :
    productionAtB (decodeShapeFixed d c left) vertex =
      productionAtB (decodeShapeFixed d c right) vertex := by
  have localAgree : AgreeOn
      (shapeFixedVertexReadSet d c
        (localExpectedReadVertices c.graph vertex)) left right :=
    agreeOn_shapeFixedVertexReadSet_mono d c
      (localExpectedReadVertices_subset_productionReadVertices c.graph vertex)
      left right agree
  have expectedEq := localExpected_decodeShapeFixed_congr
    d c vertex left right localAgree
  have outputEq := outputsAt_decodeShapeFixed_congr d c
    (productionReadVertices c.graph vertex) left right agree vertex
    (by simp [productionReadVertices])
  unfold productionAtB witnessableExpected
  simp only [decodeShapeFixed_graph]
  rw [outputEq, expectedEq]

theorem witnessFaithfulAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (witness : EdgeRouteWitness)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn
      (shapeFixedVertexReadSet d c
        (localExpectedReadVertices c.graph witness.edge.START)) left right) :
    witnessFaithfulAtB (decodeShapeFixed d c left) witness =
      witnessFaithfulAtB (decodeShapeFixed d c right) witness := by
  unfold witnessFaithfulAtB
  rw [localExpected_decodeShapeFixed_congr d c witness.edge.START
    left right agree]

theorem edgeWitnessesFor_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (edge : Deduction) (startMember : edge.START ∈ vertices) :
    edgeWitnessesFor (decodeShapeFixed d c left) edge =
      edgeWitnessesFor (decodeShapeFixed d c right) edge := by
  unfold edgeWitnessesFor
  rw [decodeShapeFixed_edgeRoutes_eq, decodeShapeFixed_edgeRoutes_eq]
  rw [List.filter_map, List.filter_map]
  apply List.map_congr_left
  intro entry member
  rcases entry with ⟨witness, index⟩
  simp only [] at member ⊢
  have originalMember : (witness, index) ∈ c.edgeRoutes.zipIdx :=
    (List.mem_filter.mp member).1
  have ownerEq : witness.edge = edge := by
    simpa using (List.mem_filter.mp member).2
  have owner : vertices.contains witness.edge.START = true := by
    simp [ownerEq, startMember]
  exact decodeWitness_shapeFixed_congr d c vertices left right agree
    index witness (witnessPayloadOwnerRelevantB_of_mem_zipIdx
      c vertices witness index originalMember owner)

theorem terminalOutputsFor_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (vertex : Vertex) (memberVertex : vertex ∈ vertices) :
    (decodeShapeFixed d c left).terminalOutputs.filter
        (fun output => output.node == vertex) =
      (decodeShapeFixed d c right).terminalOutputs.filter
        (fun output => output.node == vertex) := by
  rw [decodeShapeFixed_terminalOutputs_eq,
    decodeShapeFixed_terminalOutputs_eq]
  rw [List.filter_map, List.filter_map]
  apply List.map_congr_left
  intro entry member
  rcases entry with ⟨output, index⟩
  simp only [] at member ⊢
  have originalMember : (output, index) ∈ c.terminalOutputs.zipIdx :=
    (List.mem_filter.mp member).1
  have ownerEq : output.node = vertex := by
    simpa using (List.mem_filter.mp member).2
  have owner : vertices.contains output.node = true := by
    simp [ownerEq, memberVertex]
  exact decodeTerminal_shapeFixed_congr d c vertices left right agree
    index output (terminalPayloadOwnerRelevantB_of_mem_zipIdx
      c vertices output index originalMember owner)

theorem edgeRoutesFrom_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (vertex : Vertex) (memberVertex : vertex ∈ vertices) :
    (decodeShapeFixed d c left).edgeRoutes.filter
        (fun witness => witness.edge.START == vertex) =
      (decodeShapeFixed d c right).edgeRoutes.filter
        (fun witness => witness.edge.START == vertex) := by
  rw [decodeShapeFixed_edgeRoutes_eq, decodeShapeFixed_edgeRoutes_eq]
  rw [List.filter_map, List.filter_map]
  apply List.map_congr_left
  intro entry member
  rcases entry with ⟨witness, index⟩
  simp only at member ⊢
  have originalMember : (witness, index) ∈ c.edgeRoutes.zipIdx :=
    (List.mem_filter.mp member).1
  have ownerEq : witness.edge.START = vertex := by
    simpa using (List.mem_filter.mp member).2
  have owner : vertices.contains witness.edge.START = true := by
    simp [ownerEq, memberVertex]
  exact decodeWitness_shapeFixed_congr d c vertices left right agree
    index witness (witnessPayloadOwnerRelevantB_of_mem_zipIdx
      c vertices witness index originalMember owner)

theorem coverageAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertex : Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c [vertex]) left right) :
    coverageAtB (decodeShapeFixed d c left) vertex =
      coverageAtB (decodeShapeFixed d c right) vertex := by
  have outputEq := outputsAt_decodeShapeFixed_congr d c [vertex]
    left right agree vertex (by simp)
  have routesEq := edgeRoutesFrom_decodeShapeFixed_congr d c [vertex]
    left right agree vertex (by simp)
  have outgoingEq :
      (get_rule.outgoing vertex c.graph).all (fun edge =>
        let witnesses := edgeWitnessesFor (decodeShapeFixed d c left) edge
        !witnesses.isEmpty &&
          setEqB (witnesses.map fun witness =>
              (witness.dependency, witness.residual))
            ((outputsAt (decodeShapeFixed d c left) vertex).filter fun route =>
              headColour route.2 == edge.COLOUR)) =
      (get_rule.outgoing vertex c.graph).all (fun edge =>
        let witnesses := edgeWitnessesFor (decodeShapeFixed d c right) edge
        !witnesses.isEmpty &&
          setEqB (witnesses.map fun witness =>
              (witness.dependency, witness.residual))
            ((outputsAt (decodeShapeFixed d c right) vertex).filter fun route =>
              headColour route.2 == edge.COLOUR)) := by
    let leftCheck := fun edge : Deduction =>
      let witnesses := edgeWitnessesFor (decodeShapeFixed d c left) edge
      !witnesses.isEmpty &&
        setEqB (witnesses.map fun witness =>
            (witness.dependency, witness.residual))
          ((outputsAt (decodeShapeFixed d c left) vertex).filter fun route =>
            headColour route.2 == edge.COLOUR)
    let rightCheck := fun edge : Deduction =>
      let witnesses := edgeWitnessesFor (decodeShapeFixed d c right) edge
      !witnesses.isEmpty &&
        setEqB (witnesses.map fun witness =>
            (witness.dependency, witness.residual))
          ((outputsAt (decodeShapeFixed d c right) vertex).filter fun route =>
            headColour route.2 == edge.COLOUR)
    have pointwise : ∀ edge ∈ get_rule.outgoing vertex c.graph,
        leftCheck edge = rightCheck edge := by
      intro edge member
      have start : edge.START = vertex :=
        mem_outgoing_start_eq vertex c.graph member
      have edgeEq := edgeWitnessesFor_decodeShapeFixed_congr d c [vertex]
        left right agree edge (by simp [start])
      simp only [leftCheck, rightCheck]
      rw [edgeEq, outputEq]
    have allCongr : ∀ records : List Deduction,
        (∀ edge ∈ records, leftCheck edge = rightCheck edge) →
        records.all leftCheck = records.all rightCheck := by
      intro records hypothesis
      induction records with
      | nil => rfl
      | cons edge records induction =>
          simp only [List.all_cons]
          rw [hypothesis edge (by simp)]
          rw [induction (by
            intro item itemMember
            exact hypothesis item (by simp [itemMember]))]
    exact allCongr (get_rule.outgoing vertex c.graph) pointwise
  unfold coverageAtB
  simp only [decodeShapeFixed_graph]
  rw [outgoingEq, routesEq]

theorem listAll_congr_of_mem {α : Type} (items : List α)
    (left right : α → Bool)
    (agree : ∀ item ∈ items, left item = right item) :
    items.all left = items.all right := by
  induction items with
  | nil => rfl
  | cons item items induction =>
      simp only [List.all_cons]
      rw [agree item (by simp)]
      rw [induction (by
        intro other member
        exact agree other (by simp [member]))]

theorem listAny_congr_of_mem {α : Type} (items : List α)
    (left right : α → Bool)
    (agree : ∀ item ∈ items, left item = right item) :
    items.any left = items.any right := by
  induction items with
  | nil => rfl
  | cons item items induction =>
      simp only [List.any_cons]
      rw [agree item (by simp)]
      rw [induction (by
        intro other member
        exact agree other (by simp [member]))]

theorem continuableConsumedWitnessTails_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (start finish : Vertex) (startMember : start ∈ vertices) :
    continuableConsumedWitnessTails (decodeShapeFixed d c left) start finish =
      continuableConsumedWitnessTails (decodeShapeFixed d c right)
        start finish := by
  unfold continuableConsumedWitnessTails consumedWitnessTails
  simp only [decodeShapeFixed_graph]
  have consumedEq :
      ((edgesBetween c.graph start finish).flatMap fun edge =>
        (edgeWitnessesFor (decodeShapeFixed d c left) edge).filterMap
          fun witness => consume edge witness.residual) =
      ((edgesBetween c.graph start finish).flatMap fun edge =>
        (edgeWitnessesFor (decodeShapeFixed d c right) edge).filterMap
          fun witness => consume edge witness.residual) := by
    apply List.flatMap_congr
    intro edge member
    have endpoints := mem_edgesBetween member
    rw [edgeWitnessesFor_decodeShapeFixed_congr d c vertices left right
      agree edge (by simpa [endpoints.1] using startMember)]
  rw [consumedEq]

def tailDirectionB (c : Certificate)
    (witnesses : Deduction → List EdgeRouteWitness)
    (w first second : Vertex) (reverseReseed : Bool) : Bool :=
  (edgesBetween c.graph first w).all fun firstEdge =>
    (witnesses firstEdge).all fun firstWitness =>
      match consume firstEdge firstWitness.residual with
      | none => true
      | some firstTail =>
          !routeContinuableAtB c.graph w firstTail ||
            (edgesBetween c.graph second w).any fun secondEdge =>
              (witnesses secondEdge).any fun secondWitness =>
                consume secondEdge secondWitness.residual == some firstTail ||
                  if reverseReseed then
                    reseedPairSupportsB c.graph w
                      (secondWitness.dependency, secondWitness.residual) secondEdge
                      (firstWitness.dependency, firstWitness.residual) firstEdge
                  else
                    reseedPairSupportsB c.graph w
                      (firstWitness.dependency, firstWitness.residual) firstEdge
                      (secondWitness.dependency, secondWitness.residual) secondEdge

theorem tailDirectionB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (w first second : Vertex)
    (reverseReseed : Bool)
    (firstMember : first ∈ vertices) (secondMember : second ∈ vertices) :
    tailDirectionB (decodeShapeFixed d c left)
        (edgeWitnessesFor (decodeShapeFixed d c left)) w first second
          reverseReseed =
      tailDirectionB (decodeShapeFixed d c right)
        (edgeWitnessesFor (decodeShapeFixed d c right)) w first second
          reverseReseed := by
  unfold tailDirectionB
  simp only [decodeShapeFixed_graph]
  apply listAll_congr_of_mem
  intro firstEdge firstEdgeMember
  have firstEndpoints : firstEdge.START = first :=
    (mem_edgesBetween firstEdgeMember).1
  have firstWitnessEq := edgeWitnessesFor_decodeShapeFixed_congr
    d c vertices left right agree firstEdge
      (by simpa [firstEndpoints] using firstMember)
  rw [firstWitnessEq]
  apply listAll_congr_of_mem
  intro firstWitness _
  cases consumed : consume firstEdge firstWitness.residual with
  | none => simp []
  | some firstTail =>
      simp only []
      congr 1
      apply listAny_congr_of_mem
      intro secondEdge secondEdgeMember
      have secondEndpoints : secondEdge.START = second :=
        (mem_edgesBetween secondEdgeMember).1
      rw [edgeWitnessesFor_decodeShapeFixed_congr d c vertices left right
        agree secondEdge (by simpa [secondEndpoints] using secondMember)]

theorem certificatePairTailClosureAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertices : List Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedVertexReadSet d c vertices) left right)
    (w major minor : Vertex)
    (majorMember : major ∈ vertices) (minorMember : minor ∈ vertices) :
    certificatePairTailClosureAtB (decodeShapeFixed d c left) w major minor =
      certificatePairTailClosureAtB (decodeShapeFixed d c right)
        w major minor := by
  have minorDirection := tailDirectionB_decodeShapeFixed_congr
    d c vertices left right agree w minor major false minorMember majorMember
  have majorDirection := tailDirectionB_decodeShapeFixed_congr
    d c vertices left right agree w major minor true majorMember minorMember
  have combined := congrArg₂ (fun first second : Bool => first && second)
    minorDirection majorDirection
  simpa [certificatePairTailClosureAtB, certificatePairTailClosureFromB,
    tailDirectionB] using combined

theorem certificateCarrierAccountedAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (w u : Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn
      (shapeFixedVertexReadSet d c (productionReadVertices c.graph w))
        left right)
    (uMember : u ∈ productionReadVertices c.graph w) :
    certificateCarrierAccountedAtB (decodeShapeFixed d c left) w u =
      certificateCarrierAccountedAtB (decodeShapeFixed d c right) w u := by
  have tailsEq := continuableConsumedWitnessTails_decodeShapeFixed_congr
    d c (productionReadVertices c.graph w) left right agree u w uMember
  have localAgree : AgreeOn
      (shapeFixedVertexReadSet d c (localExpectedReadVertices c.graph w))
        left right := agreeOn_shapeFixedVertexReadSet_mono d c
          (localExpectedReadVertices_subset_productionReadVertices c.graph w)
          left right agree
  have expectedEq := localExpected_decodeShapeFixed_congr
    d c w left right localAgree
  unfold certificateCarrierAccountedAtB witnessableExpected
  simp only [decodeShapeFixed_graph]
  rw [tailsEq, expectedEq]

theorem tailClosureAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (w : Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn
      (shapeFixedVertexReadSet d c (productionReadVertices c.graph w))
        left right) :
    CertificateModel.tailClosureAtB (decodeShapeFixed d c left) w =
      CertificateModel.tailClosureAtB (decodeShapeFixed d c right) w := by
  unfold CertificateModel.tailClosureAtB
  simp only [decodeShapeFixed_graph]
  cases pairsEq : elimPairsAt c.graph w with
  | nil => simp
  | cons pair pairs =>
      simp only
      congr 1
      · apply listAll_congr_of_mem
        intro u uMember
        by_cases premise : elimPremiseAtB c.graph w u = true
        · simp [premise]
        · simp only [premise, Bool.false_or]
          have localMember : u ∈ productionReadVertices c.graph w := by
            apply localExpectedReadVertices_subset_productionReadVertices
            apply preds_mem_localExpectedReadVertices
            exact uMember
          exact certificateCarrierAccountedAtB_decodeShapeFixed_congr
            d c w u left right agree localMember
      · apply listAll_congr_of_mem
        intro current currentMember
        rcases current with ⟨major, minor⟩
        have pairIn : (major, minor) ∈ elimPairsAt c.graph w := by
          rw [pairsEq]
          exact currentMember
        obtain ⟨majorPred, minorPred⟩ :=
          mem_preds_of_mem_elimPairsAt pairIn
        exact certificatePairTailClosureAtB_decodeShapeFixed_congr
          d c (productionReadVertices c.graph w) left right agree w major minor
          (localExpectedReadVertices_subset_productionReadVertices c.graph w
            major (preds_mem_localExpectedReadVertices c.graph w major majorPred))
          (localExpectedReadVertices_subset_productionReadVertices c.graph w
            minor (preds_mem_localExpectedReadVertices c.graph w minor minorPred))

theorem rootDischargeAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (vertex : Vertex)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn
      (shapeFixedVertexReadSet d c (productionReadVertices c.graph vertex))
        left right) :
    rootDischargeAtB (decodeShapeFixed d c left) vertex =
      rootDischargeAtB (decodeShapeFixed d c right) vertex := by
  have terminalAgree : AgreeOn
      (shapeFixedVertexReadSet d c [vertex]) left right :=
    agreeOn_shapeFixedVertexReadSet_mono d c (by
      intro item member
      have equality : item = vertex := by simpa using member
      subst item
      simp [productionReadVertices]) left right agree
  have terminalEq := terminalOutputsFor_decodeShapeFixed_congr d c [vertex]
    left right terminalAgree vertex (by simp)
  have productionEq := productionAtB_decodeShapeFixed_congr
    d c vertex left right agree
  unfold rootDischargeAtB
  simp only [decodeShapeFixed_graph]
  rw [terminalEq, productionEq]

theorem witnessCoverageAtB_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (edge : Deduction)
    (left right : ShapeFixedAssignment d c) :
    witnessCoverageAtB (decodeShapeFixed d c left) edge =
      witnessCoverageAtB (decodeShapeFixed d c right) edge := by
  unfold witnessCoverageAtB edgeWitnessesFor
  rw [decodeShapeFixed_edgeRoutes_eq, decodeShapeFixed_edgeRoutes_eq]
  rw [List.filter_map, List.filter_map]
  have leftPredicate :
      ((fun witness : EdgeRouteWitness => witness.edge == edge) ∘
          fun entry => decodeWitness d c (shapeFixedToCanonical d c left)
            entry.2 entry.1) =
        fun entry : EdgeRouteWitness × Nat => entry.1.edge == edge := by
    funext entry
    rcases entry with ⟨witness, index⟩
    simp
  have rightPredicate :
      ((fun witness : EdgeRouteWitness => witness.edge == edge) ∘
          fun entry => decodeWitness d c (shapeFixedToCanonical d c right)
            entry.2 entry.1) =
        fun entry : EdgeRouteWitness × Nat => entry.1.edge == edge := by
    funext entry
    rcases entry with ⟨witness, index⟩
    simp
  rw [leftPredicate, rightPredicate]
  simp

theorem rootTerminalCoverage_decodeShapeFixed_congr
    (d : Graph) (c : Certificate) (root : Vertex)
    (left right : ShapeFixedAssignment d c) :
    ((decodeShapeFixed d c left).terminalOutputs.filter fun output =>
        output.node == root).isEmpty =
      ((decodeShapeFixed d c right).terminalOutputs.filter fun output =>
        output.node == root).isEmpty := by
  rw [decodeShapeFixed_terminalOutputs_eq,
    decodeShapeFixed_terminalOutputs_eq]
  rw [List.filter_map, List.filter_map]
  have leftPredicate :
      ((fun output : TerminalOutput => output.node == root) ∘
          fun entry => decodeTerminal d c (shapeFixedToCanonical d c left)
            entry.2 entry.1) =
        fun entry : TerminalOutput × Nat => entry.1.node == root := by
    funext entry
    rcases entry with ⟨output, index⟩
    simp
  have rightPredicate :
      ((fun output : TerminalOutput => output.node == root) ∘
          fun entry => decodeTerminal d c (shapeFixedToCanonical d c right)
            entry.2 entry.1) =
        fun entry : TerminalOutput × Nat => entry.1.node == root := by
    funext entry
    rcases entry with ⟨output, index⟩
    simp
  rw [leftPredicate, rightPredicate]
  simp



def shapeFixedPositionReadVertices (c : Certificate) :
    VerificationPosition → List Vertex
  | .witnessFaithful _ witness =>
      localExpectedReadVertices c.graph witness.edge.START
  | .nodeProduction _ vertex => productionReadVertices c.graph vertex
  | .routeCoverage _ vertex => [vertex]
  | .tailClosure _ vertex => productionReadVertices c.graph vertex
  | .rootDischarge _ vertex => productionReadVertices c.graph vertex
  | _ => []




def shapeFixedSemanticReadSet (d : Graph) (c : Certificate) :
    VerificationPosition → Finset (ShapeFixedCoord d c)
  | .degenerateLocal _ => Finset.univ
  | position => shapeFixedVertexReadSet d c
      (shapeFixedPositionReadVertices c position)

def shapeFixedSemanticFailureB (d : Graph) (c : Certificate)
    (position : VerificationPosition) :
    ShapeFixedAssignment d c → Bool :=
  fun input => !checkSemanticPositionB d (decodeShapeFixed d c input) position

theorem shapeFixedSemanticReadSet_agreement
    (d : Graph) (c : Certificate) (position : VerificationPosition)
    (left right : ShapeFixedAssignment d c)
    (agree : AgreeOn (shapeFixedSemanticReadSet d c position) left right) :
    checkSemanticPositionB d (decodeShapeFixed d c left) position =
      checkSemanticPositionB d (decodeShapeFixed d c right) position := by
  cases position with
  | graphIdentity => simp [checkSemanticPositionB]
  | expandedEdgeWitnessCoverage slot edge =>
      exact witnessCoverageAtB_decodeShapeFixed_congr d c edge left right
  | expandedWitnessScope slot witness => rfl
  | expandedRootTerminalCoverage slot root =>
      change
        (if (get_rule.outgoing root d).isEmpty then
          !((decodeShapeFixed d c left).terminalOutputs.filter fun output =>
            output.node == root).isEmpty
        else true) =
        (if (get_rule.outgoing root d).isEmpty then
          !((decodeShapeFixed d c right).terminalOutputs.filter fun output =>
            output.node == root).isEmpty
        else true)
      by_cases terminal : (get_rule.outgoing root d).isEmpty = true
      · simp only [terminal, if_true]
        rw [rootTerminalCoverage_decodeShapeFixed_congr d c root left right]
      · simp [terminal]
  | expandedTerminalScope slot output => rfl
  | edgeNodupPrefix occurrence => rfl
  | edgeNodupMain occurrence => rfl
  | oneEdgePrefix occurrence => rfl
  | oneEdgeMain occurrence => rfl
  | levelFormulaUniquePrefix occurrence => rfl
  | levelFormulaUniqueMain occurrence => rfl
  | localEdgeWitnessCoverage slot edge =>
      exact witnessCoverageAtB_decodeShapeFixed_congr d c edge left right
  | structuralEdge slot edge => rfl
  | witnessFaithful slot witness =>
      exact witnessFaithfulAtB_decodeShapeFixed_congr d c witness left right
        (by simpa [shapeFixedSemanticReadSet,
          shapeFixedPositionReadVertices] using agree)
  | structuralNode slot vertex => rfl
  | hypothesisSource slot vertex => rfl
  | nodeProduction slot vertex =>
      exact productionAtB_decodeShapeFixed_congr d c vertex left right
        (by simpa [shapeFixedSemanticReadSet,
          shapeFixedPositionReadVertices] using agree)
  | routeCoverage slot vertex =>
      exact coverageAtB_decodeShapeFixed_congr d c vertex left right
        (by simpa [shapeFixedSemanticReadSet,
          shapeFixedPositionReadVertices] using agree)
  | tailClosure slot vertex =>
      exact tailClosureAtB_decodeShapeFixed_congr d c vertex left right
        (by simpa [shapeFixedSemanticReadSet,
          shapeFixedPositionReadVertices] using agree)
  | rootDischarge slot vertex =>
      exact rootDischargeAtB_decodeShapeFixed_congr d c vertex left right
        (by simpa [shapeFixedSemanticReadSet,
          shapeFixedPositionReadVertices] using agree)
  | degenerateLocal occurrence =>
      have equalInputs : left = right := by
        funext coordinate
        exact (agree coordinate (by
          simp [shapeFixedSemanticReadSet])).symm
      subst right
      rfl


def shapeFixedSupportOfSemanticPosition (d : Graph) (c : Certificate)
    (position : VerificationPosition) : Finset (ShapeFixedCoord d c) :=
  supportOfOracle (shapeFixedSemanticFailureB d c position)

theorem shapeFixedSupportOfSemanticPosition_subset_readSet
    (d : Graph) (c : Certificate) (position : VerificationPosition) :
    shapeFixedSupportOfSemanticPosition d c position ⊆
      shapeFixedSemanticReadSet d c position := by
  apply supportOfOracle_minimal
  intro left right agree
  unfold shapeFixedSemanticFailureB
  rw [shapeFixedSemanticReadSet_agreement d c position left right agree]

def localWitnessMultiplicity (c : Certificate) (vertices : List Vertex) : Nat :=
  (c.edgeRoutes.filter fun witness =>
    vertices.contains witness.edge.START).length

def localTerminalMultiplicity (c : Certificate) (vertices : List Vertex) : Nat :=
  (c.terminalOutputs.filter fun output =>
    vertices.contains output.node).length

def witnessPayloadWidth (witness : EdgeRouteWitness) : Nat :=
  witness.dependency.length + witness.residual.length + 2

def terminalPayloadWidth (output : TerminalOutput) : Nat :=
  output.dependency.length + output.residual.length + 2

def localMaxWitnessPayloadWidth (c : Certificate)
    (vertices : List Vertex) : Nat :=
  ((c.edgeRoutes.filter fun witness =>
      vertices.contains witness.edge.START).map witnessPayloadWidth).foldr max 0

def localMaxTerminalPayloadWidth (c : Certificate)
    (vertices : List Vertex) : Nat :=
  ((c.terminalOutputs.filter fun output =>
      vertices.contains output.node).map terminalPayloadWidth).foldr max 0



def localPayloadRecordBound (c : Certificate) (vertices : List Vertex) : Nat :=
  ((c.edgeRoutes.filter fun witness =>
      vertices.contains witness.edge.START).map witnessPayloadWidth).sum +
    ((c.terminalOutputs.filter fun output =>
      vertices.contains output.node).map terminalPayloadWidth).sum

theorem localPayloadRecordBound_le_multiplicity_mul_width
    (c : Certificate) (vertices : List Vertex) :
    localPayloadRecordBound c vertices ≤
      localWitnessMultiplicity c vertices *
          localMaxWitnessPayloadWidth c vertices +
        localTerminalMultiplicity c vertices *
          localMaxTerminalPayloadWidth c vertices := by
  have witnessBound := sum_le_length_mul_foldr_max
    ((c.edgeRoutes.filter fun witness =>
      vertices.contains witness.edge.START).map witnessPayloadWidth)
  have terminalBound := sum_le_length_mul_foldr_max
    ((c.terminalOutputs.filter fun output =>
      vertices.contains output.node).map terminalPayloadWidth)
  unfold localPayloadRecordBound localWitnessMultiplicity
    localTerminalMultiplicity localMaxWitnessPayloadWidth
    localMaxTerminalPayloadWidth at *
  simpa using Nat.add_le_add witnessBound terminalBound




def localPayloadCoordinateCount (d : Graph) (c : Certificate)
    (vertices : List Vertex) : Nat :=
  (shapeFixedVertexReadSet d c vertices).card




def shapeFixedLocalSlotList (c : Certificate) (vertices : List Vertex) :
    List CanonicalSlot :=
  (canonicalSlots c).filter fun slot =>
    shapeFixedRawRelevantAtVerticesB c vertices slot.1

def localPayloadSlotBound (c : Certificate) (vertices : List Vertex) : Nat :=
  (shapeFixedLocalSlotList c vertices).length

private theorem shapeFixedRawRelevant_false_of_graphRegion
    (c : Certificate) (vertices : List Vertex)
    (coordinate : CanonicalRawCoord)
    (region : canonicalRawCoordRegion coordinate = .graph) :
    shapeFixedRawRelevantAtVerticesB c vertices coordinate = false := by
  cases coordinate <;>
    simp_all [canonicalRawCoordRegion, shapeFixedRawRelevantAtVerticesB,
      shapeFixedRawIndex]

private theorem filter_eq_nil_of_all_false {alpha : Type}
    (predicate : alpha → Bool) (values : List alpha)
    (allFalse : ∀ value ∈ values, predicate value = false) :
    values.filter predicate = [] := by
  induction values with
  | nil => rfl
  | cons head tail ih =>
      have headFalse := allFalse head (by simp)
      have tailFalse : ∀ value ∈ tail, predicate value = false := by
        intro value member
        exact allFalse value (by simp [member])
      simp [headFalse, ih tailFalse]

private theorem filteredCanonicalGraphSlots_nil
    (c : Certificate) (vertices : List Vertex) :
    (canonicalGraphSlots c).filter (fun slot =>
      shapeFixedRawRelevantAtVerticesB c vertices slot.1) = [] := by
  apply filter_eq_nil_of_all_false
  intro slot member
  exact shapeFixedRawRelevant_false_of_graphRegion c vertices slot.1
    (canonicalGraphSlots_owned c slot member)

private theorem filteredWitnessRecord_length
    (c : Certificate) (vertices : List Vertex)
    (entry : EdgeRouteWitness × Nat)
    (member : entry ∈ c.edgeRoutes.zipIdx) :
    ((([(CanonicalRawCoord.witnessPresent entry.2,
          CanonicalAtom.flag true)] : List CanonicalSlot) ++
        witnessSlots entry.2 entry.1).filter fun slot =>
          shapeFixedRawRelevantAtVerticesB c vertices slot.1).length =
      if vertices.contains entry.1.edge.START then
        witnessPayloadWidth entry.1 else 0 := by
  have facts := List.mem_zipIdx member
  have bound : entry.2 < c.edgeRoutes.length := by simpa using facts.2.1
  have value : c.edgeRoutes[entry.2] = entry.1 := by
    simpa using facts.2.2.symm
  have getWitness : c.edgeRoutes[entry.2]? = some entry.1 := by
    rw [List.getElem?_eq_getElem bound, value]
  simp only [witnessSlots, deductionSlots, vertexSlots, boundedEntries,
    List.filter_append, List.length_append]
  simp only [List.filter_flatMap, List.length_flatMap]
  simp only [List.filter_cons, List.filter_nil]
  simp [shapeFixedRawRelevantAtVerticesB, shapeFixedRawIndex,
    witnessPayloadOwnerRelevantB, witnessPayloadWidth, getWitness]
  by_cases relevant : entry.1.edge.START ∈ vertices <;>
    simp [relevant] ; omega

private theorem filteredTerminalRecord_length
    (c : Certificate) (vertices : List Vertex)
    (entry : TerminalOutput × Nat)
    (member : entry ∈ c.terminalOutputs.zipIdx) :
    ((([(CanonicalRawCoord.terminalPresent entry.2,
          CanonicalAtom.flag true)] : List CanonicalSlot) ++
        terminalSlots entry.2 entry.1).filter fun slot =>
          shapeFixedRawRelevantAtVerticesB c vertices slot.1).length =
      if vertices.contains entry.1.node then
        terminalPayloadWidth entry.1 else 0 := by
  have facts := List.mem_zipIdx member
  have bound : entry.2 < c.terminalOutputs.length := by
    simpa using facts.2.1
  have value : c.terminalOutputs[entry.2] = entry.1 := by
    simpa using facts.2.2.symm
  have getOutput : c.terminalOutputs[entry.2]? = some entry.1 := by
    rw [List.getElem?_eq_getElem bound, value]
  simp only [terminalSlots, vertexSlots, boundedEntries,
    List.filter_append, List.length_append]
  simp only [List.filter_flatMap, List.length_flatMap]
  simp only [List.filter_cons, List.filter_nil]
  simp [shapeFixedRawRelevantAtVerticesB, shapeFixedRawIndex,
    terminalPayloadOwnerRelevantB, terminalPayloadWidth, getOutput]
  by_cases relevant : entry.1.node ∈ vertices <;>
    simp [relevant] ; omega

private theorem sum_zipIdx_if_eq_filter_sum {alpha : Type}
    (records : List alpha) (predicate : alpha → Bool)
    (width : alpha → Nat) :
    (records.zipIdx.map fun entry =>
      if predicate entry.1 then width entry.1 else 0).sum =
      ((records.filter predicate).map width).sum := by
  suffices ∀ start,
      ((records.zipIdx start).map fun entry =>
        if predicate entry.1 then width entry.1 else 0).sum =
        ((records.filter predicate).map width).sum from this 0
  intro start
  induction records generalizing start with
  | nil => simp
  | cons head tail ih =>
      by_cases relevant : predicate head <;>
        simp [List.zipIdx, ih, relevant]

theorem localPayloadSlotBound_eq_recordBound
    (c : Certificate) (vertices : List Vertex) :
    localPayloadSlotBound c vertices =
      localPayloadRecordBound c vertices := by
  have witnessPart :
      ((canonicalWitnessSlots c).filter fun slot =>
        shapeFixedRawRelevantAtVerticesB c vertices slot.1).length =
        ((c.edgeRoutes.filter fun witness =>
          vertices.contains witness.edge.START).map
            witnessPayloadWidth).sum := by
    unfold canonicalWitnessSlots
    rw [List.filter_flatMap, List.length_flatMap]
    calc
      (List.map
          (fun entry =>
            (List.filter
              (fun slot => shapeFixedRawRelevantAtVerticesB c vertices slot.1)
              ([(CanonicalRawCoord.witnessPresent entry.2,
                  CanonicalAtom.flag true)] ++
                witnessSlots entry.2 entry.1)).length)
          c.edgeRoutes.zipIdx).sum =
          (c.edgeRoutes.zipIdx.map fun entry =>
            if vertices.contains entry.1.edge.START then
              witnessPayloadWidth entry.1 else 0).sum := by
        congr 1
        apply List.map_congr_left
        intro entry member
        exact filteredWitnessRecord_length c vertices entry member
      _ = _ := sum_zipIdx_if_eq_filter_sum c.edgeRoutes
        (fun witness => vertices.contains witness.edge.START)
        witnessPayloadWidth
  have terminalPart :
      ((canonicalTerminalSlots c).filter fun slot =>
        shapeFixedRawRelevantAtVerticesB c vertices slot.1).length =
        ((c.terminalOutputs.filter fun output =>
          vertices.contains output.node).map terminalPayloadWidth).sum := by
    unfold canonicalTerminalSlots
    rw [List.filter_flatMap, List.length_flatMap]
    calc
      (List.map
          (fun entry =>
            (List.filter
              (fun slot => shapeFixedRawRelevantAtVerticesB c vertices slot.1)
              ([(CanonicalRawCoord.terminalPresent entry.2,
                  CanonicalAtom.flag true)] ++
                terminalSlots entry.2 entry.1)).length)
          c.terminalOutputs.zipIdx).sum =
          (c.terminalOutputs.zipIdx.map fun entry =>
            if vertices.contains entry.1.node then
              terminalPayloadWidth entry.1 else 0).sum := by
        congr 1
        apply List.map_congr_left
        intro entry member
        exact filteredTerminalRecord_length c vertices entry member
      _ = _ := sum_zipIdx_if_eq_filter_sum c.terminalOutputs
        (fun output => vertices.contains output.node)
        terminalPayloadWidth
  unfold localPayloadSlotBound shapeFixedLocalSlotList
  rw [canonicalSlots, List.filter_append, List.filter_append,
    filteredCanonicalGraphSlots_nil]
  simp only [List.nil_append, List.length_append]
  rw [witnessPart, terminalPart]
  simp [localPayloadRecordBound]

theorem shapeFixedRawRelevant_payload (c : Certificate)
    (vertices : List Vertex) (coordinate : CanonicalRawCoord)
    (relevant : shapeFixedRawRelevantAtVerticesB c vertices coordinate = true) :
    shapeFixedPayloadRaw coordinate = true := by
  cases coordinate <;>
    simp_all [shapeFixedRawRelevantAtVerticesB, shapeFixedRawIndex,
      shapeFixedPayloadRaw]

theorem localPayloadCoordinateCount_le_slotBound
    (d : Graph) (c : Certificate) (vertices : List Vertex) :
    localPayloadCoordinateCount d c vertices ≤
      localPayloadSlotBound c vertices := by
  let forget : ShapeFixedCoord d c → CanonicalSlot := fun coordinate =>
    coordinate.1.1
  have injective : Function.Injective forget := by
    intro left right equality
    apply Subtype.ext
    apply Subtype.ext
    exact equality
  have imageEq :
      Finset.image forget (shapeFixedVertexReadSet d c vertices) =
        (shapeFixedLocalSlotList c vertices).toFinset := by
    ext slot
    constructor
    · intro member
      rcases Finset.mem_image.mp member with ⟨coordinate, coordinateMember,
        equality⟩
      have relevant : shapeFixedRawRelevantAtVerticesB c vertices
          coordinate.1.1.1 = true := by
        simpa [shapeFixedVertexReadSet,
          shapeFixedCoordRelevantAtVerticesB] using coordinateMember
      have slotMember : coordinate.1.1 ∈ canonicalSlots c := by
        exact List.mem_toFinset.mp coordinate.1.2
      subst slot
      exact List.mem_toFinset.mpr (List.mem_filter.mpr
        ⟨slotMember, relevant⟩)
    · intro member
      have filtered := List.mem_filter.mp (List.mem_toFinset.mp member)
      have canonicalMember : slot ∈ canonicalSlotSet c :=
        List.mem_toFinset.mpr filtered.1
      have payload := shapeFixedRawRelevant_payload c vertices slot.1 filtered.2
      let canonical : CanonicalCoord d c := ⟨slot, canonicalMember⟩
      let coordinate : ShapeFixedCoord d c := ⟨canonical, payload⟩
      apply Finset.mem_image.mpr
      refine ⟨coordinate, ?_, rfl⟩
      simpa [shapeFixedVertexReadSet, shapeFixedCoordRelevantAtVerticesB,
        coordinate, canonical] using filtered.2
  unfold localPayloadCoordinateCount localPayloadSlotBound
  calc
    (shapeFixedVertexReadSet d c vertices).card =
        (Finset.image forget
          (shapeFixedVertexReadSet d c vertices)).card :=
      (Finset.card_image_of_injective _ injective).symm
    _ = (shapeFixedLocalSlotList c vertices).toFinset.card :=
      congrArg Finset.card imageEq
    _ ≤ (shapeFixedLocalSlotList c vertices).length :=
      List.toFinset_card_le _

def shapeFixedSemanticReadBound (d : Graph) (c : Certificate)
    (position : VerificationPosition) : Nat :=
  (shapeFixedSemanticReadSet d c position).card

theorem shapeFixedSupport_card_le_readBound
    (d : Graph) (c : Certificate) (position : VerificationPosition) :
    (shapeFixedSupportOfSemanticPosition d c position).card ≤
      shapeFixedSemanticReadBound d c position := by
  exact Finset.card_le_card
    (shapeFixedSupportOfSemanticPosition_subset_readSet d c position)

@[simp] theorem shapeFixedVertexReadSet_empty (d : Graph) (c : Certificate) :
    shapeFixedVertexReadSet d c [] = ∅ := by
  ext coordinate
  simp only [shapeFixedVertexReadSet, Finset.mem_filter, Finset.mem_univ,
    true_and, Finset.notMem_empty, iff_false]
  unfold shapeFixedCoordRelevantAtVerticesB shapeFixedRawRelevantAtVerticesB
  cases owner : shapeFixedRawIndex coordinate.1.1.1 with
  | none => simp []
  | some tagged =>
      rcases tagged with ⟨terminal, index⟩
      cases terminal
      · simp only []
        unfold witnessPayloadOwnerRelevantB
        split <;> simp
      · simp only []
        unfold terminalPayloadOwnerRelevantB
        split <;> simp

inductive ShapeFixedFailureClass where
  | structural
  | payload
  | degenerateGlobal
  deriving Repr, BEq, DecidableEq



def shapeFixedFailureClass : VerificationPosition → ShapeFixedFailureClass
  | .witnessFaithful _ _
  | .nodeProduction _ _
  | .routeCoverage _ _
  | .tailClosure _ _
  | .rootDischarge _ _ => .payload
  | .degenerateLocal _ => .degenerateGlobal
  | _ => .structural

theorem shapeFixedStructural_readSet_eq_empty
    (d : Graph) (c : Certificate) (position : VerificationPosition)
    (structural : shapeFixedFailureClass position = .structural) :
    shapeFixedSemanticReadSet d c position = ∅ := by
  cases position <;>
    simp_all [shapeFixedFailureClass, shapeFixedSemanticReadSet,
      shapeFixedPositionReadVertices]

theorem shapeFixedStructural_support_eq_empty
    (d : Graph) (c : Certificate) (position : VerificationPosition)
    (structural : shapeFixedFailureClass position = .structural) :
    shapeFixedSupportOfSemanticPosition d c position = ∅ := by
  ext coordinate
  constructor
  · intro member
    have impossible := shapeFixedSupportOfSemanticPosition_subset_readSet
      d c position member
    rw [shapeFixedStructural_readSet_eq_empty d c position structural]
      at impossible
    simp at impossible
  · simp

theorem shapeFixedReadSet_card_le_coordinate_card
    (d : Graph) (c : Certificate) (position : VerificationPosition) :
    (shapeFixedSemanticReadSet d c position).card ≤
      Fintype.card (ShapeFixedCoord d c) :=
  Finset.card_le_univ _

theorem shapeFixedCoordinate_card_le_canonicalCoordinate_card
    (d : Graph) (c : Certificate) :
    Fintype.card (ShapeFixedCoord d c) ≤
      Fintype.card (CanonicalCoord d c) :=
  Fintype.card_subtype_le _

theorem shapeFixedSemanticReadBound_le_canonical
    (d : Graph) (c : Certificate) (position : VerificationPosition) :
    shapeFixedSemanticReadBound d c position ≤
      canonicalSemanticReadBound d c position := by
  have global : shapeFixedSemanticReadBound d c position ≤
      certificateFieldBudget c :=
    (shapeFixedReadSet_card_le_coordinate_card d c position).trans
      ((shapeFixedCoordinate_card_le_canonicalCoordinate_card d c).trans
        (canonicalCoordinate_card_le_fieldBudget d c))
  cases position <;>
    simp [shapeFixedSemanticReadBound, shapeFixedSemanticReadSet,
      shapeFixedPositionReadVertices, canonicalSemanticReadBound] at * <;>
    assumption

theorem witnessFaithful_shapeFixedSupport_card_le
    (d : Graph) (c : Certificate) (slot : Nat)
    (witness : EdgeRouteWitness) :
    (shapeFixedSupportOfSemanticPosition d c
      (.witnessFaithful slot witness)).card ≤
      localPayloadCoordinateCount d c
        (localExpectedReadVertices c.graph witness.edge.START) :=
  shapeFixedSupport_card_le_readBound d c (.witnessFaithful slot witness)

theorem witnessFaithful_shapeFixedSupport_card_le_localSlots
    (d : Graph) (c : Certificate) (slot : Nat)
    (witness : EdgeRouteWitness) :
    (shapeFixedSupportOfSemanticPosition d c
      (.witnessFaithful slot witness)).card ≤
      localPayloadSlotBound c
        (localExpectedReadVertices c.graph witness.edge.START) :=
  (witnessFaithful_shapeFixedSupport_card_le d c slot witness).trans
    (localPayloadCoordinateCount_le_slotBound d c _)

theorem witnessFaithful_shapeFixedSupport_card_le_recordBound
    (d : Graph) (c : Certificate) (slot : Nat)
    (witness : EdgeRouteWitness) :
    (shapeFixedSupportOfSemanticPosition d c
      (.witnessFaithful slot witness)).card ≤
      localPayloadRecordBound c
        (localExpectedReadVertices c.graph witness.edge.START) := by
  rw [← localPayloadSlotBound_eq_recordBound]
  exact witnessFaithful_shapeFixedSupport_card_le_localSlots d c slot witness

theorem nodeProduction_shapeFixedSupport_card_le
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.nodeProduction slot vertex)).card ≤
      localPayloadCoordinateCount d c
        (productionReadVertices c.graph vertex) :=
  shapeFixedSupport_card_le_readBound d c (.nodeProduction slot vertex)

theorem nodeProduction_shapeFixedSupport_card_le_localSlots
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.nodeProduction slot vertex)).card ≤
      localPayloadSlotBound c
        (productionReadVertices c.graph vertex) :=
  (nodeProduction_shapeFixedSupport_card_le d c slot vertex).trans
    (localPayloadCoordinateCount_le_slotBound d c _)

theorem nodeProduction_shapeFixedSupport_card_le_recordBound
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.nodeProduction slot vertex)).card ≤
      localPayloadRecordBound c
        (productionReadVertices c.graph vertex) := by
  rw [← localPayloadSlotBound_eq_recordBound]
  exact nodeProduction_shapeFixedSupport_card_le_localSlots d c slot vertex

theorem routeCoverage_shapeFixedSupport_card_le
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.routeCoverage slot vertex)).card ≤
      localPayloadCoordinateCount d c [vertex] :=
  shapeFixedSupport_card_le_readBound d c (.routeCoverage slot vertex)

theorem routeCoverage_shapeFixedSupport_card_le_localSlots
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.routeCoverage slot vertex)).card ≤
      localPayloadSlotBound c [vertex] :=
  (routeCoverage_shapeFixedSupport_card_le d c slot vertex).trans
    (localPayloadCoordinateCount_le_slotBound d c _)

theorem routeCoverage_shapeFixedSupport_card_le_recordBound
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.routeCoverage slot vertex)).card ≤
      localPayloadRecordBound c [vertex] := by
  rw [← localPayloadSlotBound_eq_recordBound]
  exact routeCoverage_shapeFixedSupport_card_le_localSlots d c slot vertex

theorem tailClosure_shapeFixedSupport_card_le
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.tailClosure slot vertex)).card ≤
      localPayloadCoordinateCount d c
        (productionReadVertices c.graph vertex) :=
  shapeFixedSupport_card_le_readBound d c (.tailClosure slot vertex)

theorem tailClosure_shapeFixedSupport_card_le_localSlots
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.tailClosure slot vertex)).card ≤
      localPayloadSlotBound c
        (productionReadVertices c.graph vertex) :=
  (tailClosure_shapeFixedSupport_card_le d c slot vertex).trans
    (localPayloadCoordinateCount_le_slotBound d c _)

theorem tailClosure_shapeFixedSupport_card_le_recordBound
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.tailClosure slot vertex)).card ≤
      localPayloadRecordBound c
        (productionReadVertices c.graph vertex) := by
  rw [← localPayloadSlotBound_eq_recordBound]
  exact tailClosure_shapeFixedSupport_card_le_localSlots d c slot vertex

theorem rootDischarge_shapeFixedSupport_card_le
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.rootDischarge slot vertex)).card ≤
      localPayloadCoordinateCount d c
        (productionReadVertices c.graph vertex) :=
  shapeFixedSupport_card_le_readBound d c (.rootDischarge slot vertex)

theorem rootDischarge_shapeFixedSupport_card_le_localSlots
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.rootDischarge slot vertex)).card ≤
      localPayloadSlotBound c
        (productionReadVertices c.graph vertex) :=
  (rootDischarge_shapeFixedSupport_card_le d c slot vertex).trans
    (localPayloadCoordinateCount_le_slotBound d c _)

theorem rootDischarge_shapeFixedSupport_card_le_recordBound
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex) :
    (shapeFixedSupportOfSemanticPosition d c
      (.rootDischarge slot vertex)).card ≤
      localPayloadRecordBound c
        (productionReadVertices c.graph vertex) := by
  rw [← localPayloadSlotBound_eq_recordBound]
  exact rootDischarge_shapeFixedSupport_card_le_localSlots d c slot vertex



abbrev ShapeFixedSemanticOrbit (d : Graph) (c : Certificate)
    (position : VerificationPosition) :=
  { other : ShapeFixedAssignment d c //
    AgreeOn (shapeFixedSupportOfSemanticPosition d c position)
      (encodeShapeFixed d c) other }

abbrev ShapeFixedSemanticFailureAssignments (d : Graph) (c : Certificate)
    (position : VerificationPosition) :=
  { input : ShapeFixedAssignment d c //
    shapeFixedSemanticFailureB d c position input = true }

noncomputable instance shapeFixedSemanticOrbitFintype (d : Graph)
    (c : Certificate) (position : VerificationPosition) :
    Fintype (ShapeFixedSemanticOrbit d c position) := Fintype.ofFinite _

noncomputable instance shapeFixedSemanticFailureAssignmentsFintype
    (d : Graph) (c : Certificate) (position : VerificationPosition) :
    Fintype (ShapeFixedSemanticFailureAssignments d c position) :=
  Fintype.ofFinite _

def shapeFixedSemanticOrbitEquiv (d : Graph) (c : Certificate)
    (position : VerificationPosition) :
    ShapeFixedSemanticOrbit d c position ≃
      (↥(shapeFixedSupportOfSemanticPosition d c position)ᶜ →
        ShapeFixedAlphabet d c) where
  toFun other coordinate := other.1 coordinate.1
  invFun free :=
    ⟨fun coordinate =>
        if member : coordinate ∈
            shapeFixedSupportOfSemanticPosition d c position then
          encodeShapeFixed d c coordinate
        else free ⟨coordinate, by simpa using member⟩,
      by
        intro coordinate member
        simp [member]⟩
  left_inv other := by
    apply Subtype.ext
    funext coordinate
    by_cases member : coordinate ∈
        shapeFixedSupportOfSemanticPosition d c position
    · simp [member, other.2 coordinate member]
    · simp [member]
  right_inv free := by
    funext coordinate
    have outside : coordinate.1 ∉
        shapeFixedSupportOfSemanticPosition d c position :=
      Finset.mem_compl.mp coordinate.2
    simp [outside]

theorem shapeFixedSemanticOrbit_card (d : Graph) (c : Certificate)
    (position : VerificationPosition) :
    Fintype.card (ShapeFixedSemanticOrbit d c position) =
      Fintype.card (ShapeFixedAlphabet d c) ^
        (Fintype.card (ShapeFixedCoord d c) -
          (shapeFixedSupportOfSemanticPosition d c position).card) := by
  calc
    Fintype.card (ShapeFixedSemanticOrbit d c position) =
        Fintype.card
          (↥(shapeFixedSupportOfSemanticPosition d c position)ᶜ →
            ShapeFixedAlphabet d c) :=
      Fintype.card_congr (shapeFixedSemanticOrbitEquiv d c position)
    _ = Fintype.card (ShapeFixedAlphabet d c) ^
          Fintype.card
            ↥(shapeFixedSupportOfSemanticPosition d c position)ᶜ :=
      Fintype.card_fun
    _ = Fintype.card (ShapeFixedAlphabet d c) ^
          (shapeFixedSupportOfSemanticPosition d c position)ᶜ.card := by
      congr 1
      exact Fintype.card_coe _
    _ = Fintype.card (ShapeFixedAlphabet d c) ^
        (Fintype.card (ShapeFixedCoord d c) -
          (shapeFixedSupportOfSemanticPosition d c position).card) := by
      congr 1
      exact Finset.card_compl _

def shapeFixedSemanticOrbitToFailures (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    ShapeFixedSemanticOrbit d c position →
      ShapeFixedSemanticFailureAssignments d c position := by
  intro other
  refine ⟨other.1, ?_⟩
  have invariant := supportOfOracle_agreement
    (shapeFixedSemanticFailureB d c position)
    (encodeShapeFixed d c) other.1 other.2
  rw [invariant]
  unfold shapeFixedSemanticFailureB
  rw [decodeShapeFixed_encodeShapeFixed]
  unfold SemanticMarkedAt at failure
  rw [failure]
  rfl

theorem shapeFixedSemanticOrbitToFailures_injective (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    Function.Injective
      (shapeFixedSemanticOrbitToFailures d c position failure) := by
  intro left right equality
  apply Subtype.ext
  simpa [shapeFixedSemanticOrbitToFailures] using
    congrArg Subtype.val equality

theorem shapeFixedSemanticFailureAssignments_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    Fintype.card (ShapeFixedAlphabet d c) ^
        (Fintype.card (ShapeFixedCoord d c) -
          (shapeFixedSupportOfSemanticPosition d c position).card) ≤
      Fintype.card (ShapeFixedSemanticFailureAssignments d c position) := by
  rw [← shapeFixedSemanticOrbit_card d c position]
  exact Fintype.card_le_of_injective
    (shapeFixedSemanticOrbitToFailures d c position failure)
    (shapeFixedSemanticOrbitToFailures_injective d c position failure)

noncomputable def shapeFixedSemanticMalformationDensity (d : Graph)
    (c : Certificate) (position : VerificationPosition) : ℚ :=
  Fintype.card (ShapeFixedSemanticFailureAssignments d c position) /
    Fintype.card (ShapeFixedAlphabet d c) ^
      Fintype.card (ShapeFixedCoord d c)

theorem shapeFixedSemanticMarked_density_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    (1 : ℚ) / Fintype.card (ShapeFixedAlphabet d c) ^
        (shapeFixedSupportOfSemanticPosition d c position).card ≤
      shapeFixedSemanticMalformationDensity d c position := by
  have alphabetPositive : 0 < Fintype.card (ShapeFixedAlphabet d c) :=
    Fintype.card_pos
  have supportLe :
      (shapeFixedSupportOfSemanticPosition d c position).card ≤
        Fintype.card (ShapeFixedCoord d c) := Finset.card_le_univ _
  have lower := shapeFixedSemanticFailureAssignments_lower_bound
    d c position failure
  have cross :
      Fintype.card (ShapeFixedAlphabet d c) ^
          Fintype.card (ShapeFixedCoord d c) ≤
        Fintype.card (ShapeFixedAlphabet d c) ^
            (shapeFixedSupportOfSemanticPosition d c position).card *
          Fintype.card
            (ShapeFixedSemanticFailureAssignments d c position) := by
    calc
      Fintype.card (ShapeFixedAlphabet d c) ^
          Fintype.card (ShapeFixedCoord d c) =
        Fintype.card (ShapeFixedAlphabet d c) ^
            (shapeFixedSupportOfSemanticPosition d c position).card *
          Fintype.card (ShapeFixedAlphabet d c) ^
            (Fintype.card (ShapeFixedCoord d c) -
              (shapeFixedSupportOfSemanticPosition d c position).card) := by
        rw [← pow_add, Nat.add_sub_of_le supportLe]
      _ ≤ _ := Nat.mul_le_mul_left _ lower
  unfold shapeFixedSemanticMalformationDensity
  apply (div_le_div_iff₀ (by positivity :
      (0 : ℚ) < Fintype.card (ShapeFixedAlphabet d c) ^
        (shapeFixedSupportOfSemanticPosition d c position).card)
    (by positivity : (0 : ℚ) < Fintype.card (ShapeFixedAlphabet d c) ^
      Fintype.card (ShapeFixedCoord d c))).2
  norm_num
  exact_mod_cast (by simpa [Nat.mul_comm] using cross)

theorem shapeFixedSemanticMarked_bound_density_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    (1 : ℚ) / Fintype.card (ShapeFixedAlphabet d c) ^
        shapeFixedSemanticReadBound d c position ≤
      shapeFixedSemanticMalformationDensity d c position := by
  have exactLower := shapeFixedSemanticMarked_density_lower_bound
    d c position failure
  have supportLe := shapeFixedSupport_card_le_readBound d c position
  have alphabetOne :
      (1 : ℚ) ≤ Fintype.card (ShapeFixedAlphabet d c) := by
    exact_mod_cast (Fintype.card_pos :
      0 < Fintype.card (ShapeFixedAlphabet d c))
  have denominatorLe :
      (Fintype.card (ShapeFixedAlphabet d c) : ℚ) ^
          (shapeFixedSupportOfSemanticPosition d c position).card ≤
        (Fintype.card (ShapeFixedAlphabet d c) : ℚ) ^
          shapeFixedSemanticReadBound d c position :=
    pow_le_pow_right₀ alphabetOne supportLe
  have positive :
      (0 : ℚ) < (Fintype.card (ShapeFixedAlphabet d c) : ℚ) ^
        (shapeFixedSupportOfSemanticPosition d c position).card := by
    positivity
  exact (one_div_le_one_div_of_le positive denominatorLe).trans exactLower





theorem formation_invalid_shapeFixed_amplification (d : Graph)
    (c : Certificate) (invalid : ¬ FormationValid d c) :
    ∃ position, position ∈ semanticPositions d c ∧
      SemanticMarkedAt d c position ∧
      (shapeFixedSupportOfSemanticPosition d c position).card ≤
        shapeFixedSemanticReadBound d c position ∧
      (1 : ℚ) / Fintype.card (ShapeFixedAlphabet d c) ^
          shapeFixedSemanticReadBound d c position ≤
        shapeFixedSemanticMalformationDensity d c position := by
  obtain ⟨position, member, failure⟩ :=
    formation_invalid_implies_semanticMarked d c invalid
  exact ⟨position, member, failure,
    shapeFixedSupport_card_le_readBound d c position,
    shapeFixedSemanticMarked_bound_density_lower_bound
      d c position failure⟩


end CertificateRobustness
end SearchProblem
