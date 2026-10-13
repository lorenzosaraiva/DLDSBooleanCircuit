import Search.Certificate.Definitions
import Mathlib.Data.List.Sigma
import Certificate.Flow.RootDischarge

set_option maxHeartbeats 1000000

/-!
# Stable semantic certificate-verifier positions

The legacy search oracle serializes checks as natural-number offsets.  Those
offsets are useful to exported search code, but a witness-list length change
moves every later block.  `VerificationPosition` names the check family and
its local occurrence directly.  Natural-number positions are retained below
only as the serialization of `semanticPositions`.
-/

open DLDS FlowSemantics Certificate.Flow.Foundation

namespace CertificateSearch

/-- Stable logical names for every component of the complete formation
checker.  Occurrence numbers distinguish repeated graph records without
making later families depend on the lengths of earlier families. -/
inductive VerificationPosition where
  | graphIdentity
  | expandedEdgeWitnessCoverage (edgeSlot : Nat) (edge : Deduction)
  | expandedWitnessScope (witnessSlot : Nat) (witness : EdgeRouteWitness)
  | expandedRootTerminalCoverage (rootSlot : Nat) (root : Vertex)
  | expandedTerminalScope (terminalSlot : Nat) (output : TerminalOutput)
  | edgeNodupPrefix (occurrence : Nat)
  | edgeNodupMain (occurrence : Nat)
  | oneEdgePrefix (occurrence : Nat)
  | oneEdgeMain (occurrence : Nat)
  | levelFormulaUniquePrefix (occurrence : Nat)
  | levelFormulaUniqueMain (occurrence : Nat)
  | localEdgeWitnessCoverage (edgeSlot : Nat) (edge : Deduction)
  | structuralEdge (edgeSlot : Nat) (edge : Deduction)
  | witnessFaithful (witnessSlot : Nat) (witness : EdgeRouteWitness)
  | structuralNode (nodeSlot : Nat) (node : Vertex)
  | hypothesisSource (nodeSlot : Nat) (node : Vertex)
  | nodeProduction (nodeSlot : Nat) (node : Vertex)
  | routeCoverage (nodeSlot : Nat) (node : Vertex)
  | tailClosure (nodeSlot : Nat) (node : Vertex)
  | rootDischarge (nodeSlot : Nat) (node : Vertex)
  /-- The old oracle deliberately repeats the complete local checker on
  degenerate graphs, because truncated arithmetic makes its nominal blocks
  overlap. -/
  | degenerateLocal (occurrence : Nat)
  deriving Repr, DecidableEq

/-- The verifier, stated directly by semantic check family. -/
def checkSemanticPositionB (d : Graph) (c : Certificate) :
    VerificationPosition → Bool
  | .graphIdentity => decide (c.graph = d)
  | .expandedEdgeWitnessCoverage _ deduction => witnessCoverageAtB c deduction
  | .expandedWitnessScope _ witness => d.EDGES.contains witness.edge
  | .expandedRootTerminalCoverage _ vertex =>
      if (get_rule.outgoing vertex d).isEmpty then
        !(c.terminalOutputs.filter fun output => output.node == vertex).isEmpty
      else true
  | .expandedTerminalScope _ output =>
      d.NODES.contains output.node && (get_rule.outgoing output.node d).isEmpty
  | .edgeNodupPrefix _ => nodupB d
  | .edgeNodupMain _ => nodupB d
  | .oneEdgePrefix _ => oneEdgeB d
  | .oneEdgeMain _ => oneEdgeB d
  | .levelFormulaUniquePrefix _ => levelFormulaUniqueB d
  | .levelFormulaUniqueMain _ => levelFormulaUniqueB d
  | .localEdgeWitnessCoverage _ deduction => witnessCoverageAtB c deduction
  | .structuralEdge _ deduction => structuralEdgeAtB d deduction
  | .witnessFaithful _ witness => witnessFaithfulAtB c witness
  | .structuralNode _ vertex => structuralNodeAtB d vertex
  | .hypothesisSource _ vertex => hypSourceAtB d vertex
  | .nodeProduction _ vertex => productionAtB c vertex
  | .routeCoverage _ vertex => coverageAtB c vertex
  | .tailClosure _ vertex => tailClosureAtB c vertex
  | .rootDischarge _ vertex => rootDischargeAtB c vertex
  | .degenerateLocal _ => Certificate.Flow.localChecksB d c

private def slotMap (size : Nat)
    (position : Nat → VerificationPosition) : List VerificationPosition :=
  (List.range size).map position

private def indexedMap {α : Type} (values : List α)
    (position : Nat → α → VerificationPosition) : List VerificationPosition :=
  values.zipIdx.map fun entry => position entry.2 entry.1

@[simp] private theorem indexedMap_length {α : Type} (values : List α)
    (position : Nat → α → VerificationPosition) :
    (indexedMap values position).length = values.length := by
  simp [indexedMap]

private def repeatedInvariantPositions (size : Nat)
    (prefixPosition mainPosition : Nat → VerificationPosition) :
    List VerificationPosition :=
  slotMap (size - 1) prefixPosition ++ slotMap size mainPosition

/-- Semantic names for the five classes making up `ExpandedCert`. -/
def expandedSemanticPositions (d : Graph) (c : Certificate) :
    List VerificationPosition :=
  [.graphIdentity] ++
  indexedMap d.EDGES .expandedEdgeWitnessCoverage ++
  indexedMap c.edgeRoutes .expandedWitnessScope ++
  indexedMap d.NODES .expandedRootTerminalCoverage ++
  indexedMap c.terminalOutputs .expandedTerminalScope

private def rawLocalSemanticPositions (d : Graph) (c : Certificate) :
    List VerificationPosition :=
  repeatedInvariantPositions d.EDGES.length
      .edgeNodupPrefix .edgeNodupMain ++
  repeatedInvariantPositions d.EDGES.length
      .oneEdgePrefix .oneEdgeMain ++
  repeatedInvariantPositions d.NODES.length
      .levelFormulaUniquePrefix .levelFormulaUniqueMain ++
  indexedMap d.EDGES .localEdgeWitnessCoverage ++
  indexedMap d.EDGES .structuralEdge ++
  indexedMap c.edgeRoutes .witnessFaithful ++
  d.NODES.zipIdx.flatMap fun entry =>
    let node := entry.1
    let slot := entry.2
    [.structuralNode slot node,
     .hypothesisSource slot node,
     .nodeProduction slot node,
     .routeCoverage slot node,
     .tailClosure slot node,
     .rootDischarge slot node]

/-- Semantic local positions.  The exceptional branch exactly records the
legacy behavior on empty-edge or empty-node graphs. -/
def localSemanticPositions (d : Graph) (c : Certificate) :
    List VerificationPosition :=
  if d.EDGES.isEmpty || d.NODES.isEmpty then
    slotMap (numPositions d c) .degenerateLocal
  else
    rawLocalSemanticPositions d c

/-- Duplicate-free finite serialization of every complete-formation check. -/
def semanticPositions (d : Graph) (c : Certificate) :
    List VerificationPosition :=
  expandedSemanticPositions d c ++ localSemanticPositions d c

/-- Failure at one stable semantic position. -/
def SemanticMarkedAt (d : Graph) (c : Certificate)
    (position : VerificationPosition) : Prop :=
  checkSemanticPositionB d c position = false

private theorem map_range_getElem?_getD {α β : Type} (xs : List α)
    (f : α → β) (fallback : β) :
    (List.range xs.length).map
        (fun i => (xs[i]?).map f |>.getD fallback) = xs.map f := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map, List.getElem?_cons_zero, Option.map_some,
        Option.getD_some]
      rw [List.cons.injEq]
      exact ⟨rfl, by
        simpa only [Function.comp_apply, List.getElem?_cons_succ] using ih⟩

@[simp] private theorem map_zipIdx_fst_apply {α β : Type} (xs : List α)
    (f : α → β) :
    xs.zipIdx.map (fun entry => f entry.1) = xs.map f := by
  simpa only [List.map_map, Function.comp_apply] using
    congrArg (List.map f) (List.zipIdx_map_fst 0 xs)

theorem expandedSemanticPositionChecks (d : Graph) (c : Certificate) :
    (expandedSemanticPositions d c).map (checkSemanticPositionB d c) =
      expandedPositionChecks d c := by
  unfold expandedSemanticPositions expandedPositionChecks indexedMap
  simp only [List.map_append, List.map_cons, List.map_nil, List.map_map,
    Function.comp_def, checkSemanticPositionB]
  rw [map_zipIdx_fst_apply d.EDGES (witnessCoverageAtB c)]
  rw [map_zipIdx_fst_apply c.edgeRoutes
    (fun witness => d.EDGES.contains witness.edge)]
  rw [map_zipIdx_fst_apply d.NODES (fun vertex =>
    if (get_rule.outgoing vertex d).isEmpty then
      !(c.terminalOutputs.filter fun output => output.node == vertex).isEmpty
    else true)]
  rw [map_zipIdx_fst_apply c.terminalOutputs (fun output =>
    d.NODES.contains output.node &&
      (get_rule.outgoing output.node d).isEmpty)]

theorem expandedSemanticPositions_length (d : Graph) (c : Certificate) :
    (expandedSemanticPositions d c).length = expandedNumPositions d c := by
  simp [expandedSemanticPositions, expandedNumPositions,
    expandedPositionChecks]

theorem localSemanticPositions_length (d : Graph) (c : Certificate) :
    (localSemanticPositions d c).length = numPositions d c := by
  unfold localSemanticPositions
  split <;> rename_i branch
  · simp [slotMap]
  · have nonempty : d.EDGES ≠ [] ∧ d.NODES ≠ [] := by
      simpa using branch
    have edgesPositive : 0 < d.EDGES.length := by
      cases edges : d.EDGES with
      | nil => exact (nonempty.1 edges).elim
      | cons head tail => simp []
    have nodesPositive : 0 < d.NODES.length := by
      cases nodes : d.NODES with
      | nil => exact (nonempty.2 nodes).elim
      | cons head tail => simp []
    simp [rawLocalSemanticPositions, repeatedInvariantPositions,
      slotMap, numPositions, indexedMap]
    omega

theorem semanticPositions_length (d : Graph) (c : Certificate) :
    (semanticPositions d c).length = totalNumPositions d c := by
  simp [semanticPositions, totalNumPositions,
    expandedSemanticPositions_length, localSemanticPositions_length]

private theorem flatMap_zipIdx_fst_apply {α β : Type} (xs : List α)
    (f : α → List β) (start : Nat := 0) :
    (xs.zipIdx start).flatMap (fun entry => f entry.1) = xs.flatMap f := by
  induction xs generalizing start with
  | nil => rfl
  | cons value rest inductionHypothesis =>
      simp [List.zipIdx, inductionHypothesis]

private theorem map_nodeSemanticPositions (d : Graph) (c : Certificate) :
    (d.NODES.zipIdx.flatMap fun entry =>
      [.structuralNode entry.2 entry.1,
       .hypothesisSource entry.2 entry.1,
       .nodeProduction entry.2 entry.1,
       .routeCoverage entry.2 entry.1,
       .tailClosure entry.2 entry.1,
       .rootDischarge entry.2 entry.1]).map (checkSemanticPositionB d c) =
      d.NODES.flatMap fun vertex =>
        [structuralNodeAtB d vertex,
         hypSourceAtB d vertex,
         productionAtB c vertex,
         coverageAtB c vertex,
         tailClosureAtB c vertex,
         rootDischargeAtB c vertex] := by
  rw [List.map_flatMap]
  simpa [checkSemanticPositionB] using
    flatMap_zipIdx_fst_apply d.NODES (fun vertex =>
      [structuralNodeAtB d vertex,
       hypSourceAtB d vertex,
       productionAtB c vertex,
       coverageAtB c vertex,
       tailClosureAtB c vertex,
       rootDischargeAtB c vertex])

/-- The semantic local serialization computes exactly the legacy local
oracle list on ordinary graphs and its documented repeated-global-check list
on degenerate graphs. -/
theorem localSemanticPositionChecks (d : Graph) (c : Certificate) :
    (localSemanticPositions d c).map (checkSemanticPositionB d c) =
      if d.EDGES.isEmpty || d.NODES.isEmpty then
        List.replicate (numPositions d c) (Certificate.Flow.localChecksB d c)
      else positionChecks d c := by
  by_cases degenerate : (d.EDGES.isEmpty || d.NODES.isEmpty)
  · simp only [localSemanticPositions, degenerate, if_true, slotMap,
      List.map_map]
    change (List.range (numPositions d c)).map
        (fun _ => Certificate.Flow.localChecksB d c) =
      List.replicate (numPositions d c) (Certificate.Flow.localChecksB d c)
    simp
  · have nonempty : d.EDGES ≠ [] ∧ d.NODES ≠ [] := by
      simpa using degenerate
    have edgesPositive : 0 < d.EDGES.length := by
      cases edges : d.EDGES with
      | nil => exact (nonempty.1 edges).elim
      | cons => simp
    have nodesPositive : 0 < d.NODES.length := by
      cases nodes : d.NODES with
      | nil => exact (nonempty.2 nodes).elim
      | cons => simp
    rw [show localSemanticPositions d c = rawLocalSemanticPositions d c by
      simp [localSemanticPositions, degenerate]]
    rw [show (if d.EDGES.isEmpty || d.NODES.isEmpty then
        List.replicate (numPositions d c) (Certificate.Flow.localChecksB d c)
      else positionChecks d c) = positionChecks d c by simp [degenerate]]
    rw [positionChecks_eq_raw d c edgesPositive nodesPositive]
    simp only [rawLocalSemanticPositions, List.map_append,
        repeatedInvariantPositions]
    rw [map_nodeSemanticPositions d c]
    simp only [slotMap, indexedMap, List.map_map,
      Function.comp_def, checkSemanticPositionB]
    simp [rawPositionChecks, sortedBlock]

/-- All serialized semantic outcomes, before taking individual indices. -/
def semanticPositionChecks (d : Graph) (c : Certificate) : List Bool :=
  (semanticPositions d c).map (checkSemanticPositionB d c)

theorem semanticPositionChecks_eq_legacy (d : Graph) (c : Certificate) :
    semanticPositionChecks d c =
      expandedPositionChecks d c ++
        (if d.EDGES.isEmpty || d.NODES.isEmpty then
          List.replicate (numPositions d c)
            (Certificate.Flow.localChecksB d c)
        else positionChecks d c) := by
  simp [semanticPositionChecks, semanticPositions,
    expandedSemanticPositionChecks, localSemanticPositionChecks]

private theorem checkPositionB_eq_semanticLocalGetD (d : Graph)
    (c : Certificate) (index : Nat) (inBounds : index < numPositions d c) :
    checkPositionB d c index =
      ((localSemanticPositions d c).map
        (checkSemanticPositionB d c))[index]?.getD true := by
  by_cases degenerate : (d.EDGES.isEmpty || d.NODES.isEmpty)
  · rw [localSemanticPositionChecks]
    simp [checkPositionB, degenerate, inBounds]
  · rw [localSemanticPositionChecks]
    simp [checkPositionB, listedCheckPositionB, degenerate]

/-- The numeric total checker is precisely lookup in the semantic
serialization.  This holds even for out-of-range padding indices. -/
theorem totalCheckPositionB_eq_semanticGetD (d : Graph) (c : Certificate)
    (index : Nat) (inBounds : index < totalNumPositions d c) :
    totalCheckPositionB d c index =
      (semanticPositionChecks d c)[index]?.getD true := by
  unfold totalCheckPositionB semanticPositionChecks semanticPositions
  rw [List.map_append, expandedSemanticPositionChecks]
  by_cases expanded : index < expandedNumPositions d c
  · rw [dif_pos expanded, List.getElem?_append,
      if_pos (by simpa [expandedSemanticPositions_length] using expanded)]
    rw [List.getElem?_eq_getElem (by
      simpa [expandedNumPositions] using expanded)]
    simp
  · rw [dif_neg expanded, List.getElem?_append,
      if_neg (by simpa [expandedSemanticPositions_length] using expanded)]
    have localInBounds :
        index - expandedNumPositions d c < numPositions d c := by
      unfold totalNumPositions at inBounds
      omega
    have localLookup := checkPositionB_eq_semanticLocalGetD d c
      (index - expandedNumPositions d c) localInBounds
    simpa [expandedNumPositions] using localLookup

theorem checkSemanticPositionB_eq_totalCheckPositionB (d : Graph)
    (c : Certificate) (index : Nat)
    (inBounds : index < (semanticPositions d c).length) :
    checkSemanticPositionB d c (semanticPositions d c)[index] =
      totalCheckPositionB d c index := by
  rw [totalCheckPositionB_eq_semanticGetD d c index (by
    simpa [semanticPositions_length] using inBounds)]
  simp [semanticPositionChecks, inBounds]

private theorem slotMap_nodup (size : Nat)
    (position : Nat → VerificationPosition)
    (injective : Function.Injective position) :
    (slotMap size position).Nodup := by
  exact List.nodup_range.map injective

private def verificationPositionKey : VerificationPosition → Nat × Nat
  | .graphIdentity => (0, 0)
  | .expandedEdgeWitnessCoverage slot _ => (1, slot)
  | .expandedWitnessScope slot _ => (2, slot)
  | .expandedRootTerminalCoverage slot _ => (3, slot)
  | .expandedTerminalScope slot _ => (4, slot)
  | .edgeNodupPrefix slot => (5, slot)
  | .edgeNodupMain slot => (6, slot)
  | .oneEdgePrefix slot => (7, slot)
  | .oneEdgeMain slot => (8, slot)
  | .levelFormulaUniquePrefix slot => (9, slot)
  | .levelFormulaUniqueMain slot => (10, slot)
  | .localEdgeWitnessCoverage slot _ => (11, slot)
  | .structuralEdge slot _ => (12, slot)
  | .witnessFaithful slot _ => (13, slot)
  | .structuralNode slot _ => (14, slot)
  | .hypothesisSource slot _ => (15, slot)
  | .nodeProduction slot _ => (16, slot)
  | .routeCoverage slot _ => (17, slot)
  | .tailClosure slot _ => (18, slot)
  | .rootDischarge slot _ => (19, slot)
  | .degenerateLocal slot => (20, slot)

@[simp] private theorem taggedRange_nodup (tag size : Nat) :
    ((List.range size).map fun slot => (tag, slot)).Nodup := by
  apply List.Nodup.map
  · intro left right equality
    simpa using congrArg Prod.snd equality
  · exact List.nodup_range

@[simp] private theorem map_zipIdx_snd_apply {α β : Type} (xs : List α)
    (f : Nat → β) :
    xs.zipIdx.map (fun entry => f entry.2) = (List.range xs.length).map f := by
  simpa only [List.map_map, Function.comp_apply, ← List.range_eq_range'] using
    congrArg (List.map f) (List.zipIdx_map_snd 0 xs)

private theorem flatMap_zipIdx_snd_apply {α β : Type} (xs : List α)
    (f : Nat → List β) :
    xs.zipIdx.flatMap (fun entry => f entry.2) =
      (List.range xs.length).flatMap f := by
  rw [← List.flatMap_map]
  simpa only [List.map_map, Function.comp_apply, ← List.range_eq_range'] using
    congrArg (List.flatMap f) (List.zipIdx_map_snd 0 xs)

private theorem nodeKeys_nodup (size : Nat) :
    ((List.range size).flatMap fun slot =>
      [(14, slot), (15, slot), (16, slot),
       (17, slot), (18, slot), (19, slot)]).Nodup := by
  rw [List.nodup_flatMap]
  constructor
  · intro slot member
    simp
  · apply List.nodup_range.imp
    intro leftIndex rightIndex indicesDiffer
    simp only [Function.onFun]
    rw [List.disjoint_left]
    intro position leftMember rightMember
    simp at leftMember rightMember
    rcases leftMember with rfl | rfl | rfl | rfl | rfl | rfl <;>
      rcases rightMember with equality | equality | equality | equality |
        equality | equality <;>
      cases equality <;> exact indicesDiffer rfl

private theorem mappedNodePositions_nodup (nodes : List Vertex) :
    ((nodes.zipIdx.flatMap fun entry =>
      [CertificateSearch.VerificationPosition.structuralNode entry.2 entry.1,
       CertificateSearch.VerificationPosition.hypothesisSource entry.2 entry.1,
       CertificateSearch.VerificationPosition.nodeProduction entry.2 entry.1,
       CertificateSearch.VerificationPosition.routeCoverage entry.2 entry.1,
       CertificateSearch.VerificationPosition.tailClosure entry.2 entry.1,
       CertificateSearch.VerificationPosition.rootDischarge entry.2 entry.1]).map
        verificationPositionKey).Nodup := by
  rw [List.map_flatMap]
  simp only [verificationPositionKey, List.map_cons, List.map_nil]
  have keySerialization := flatMap_zipIdx_snd_apply nodes (fun slot =>
    [(14, slot), (15, slot), (16, slot),
     (17, slot), (18, slot), (19, slot)])
  rw [keySerialization]
  exact nodeKeys_nodup nodes.length

private def taggedBlock (tag size : Nat) : List (Nat × Nat) :=
  (List.range size).map fun slot => (tag, slot)

private theorem taggedBlock_disjoint_of_tag_ne {leftTag rightTag leftSize rightSize : Nat}
    (tagsDiffer : leftTag ≠ rightTag) :
    (taggedBlock leftTag leftSize).Disjoint
      (taggedBlock rightTag rightSize) := by
  rw [List.disjoint_left]
  intro key leftMember rightMember
  simp [taggedBlock] at leftMember rightMember
  obtain ⟨leftIndex, _, rfl⟩ := leftMember
  obtain ⟨rightIndex, _, equality⟩ := rightMember
  exact tagsDiffer (congrArg Prod.fst equality).symm

private theorem fst_eq_tag_of_mem_taggedBlock {tag size : Nat}
    {key : Nat × Nat} (member : key ∈ taggedBlock tag size) :
    key.1 = tag := by
  simp [taggedBlock] at member
  obtain ⟨index, _, rfl⟩ := member
  rfl

private theorem taggedBlocks_nodup :
    ∀ specifications : List (Nat × Nat),
      (specifications.map Prod.fst).Nodup →
      (specifications.flatMap fun specification =>
        taggedBlock specification.1 specification.2).Nodup
  | [], _ => by simp
  | specification :: rest, distinctTags => by
      rw [List.map_cons, List.nodup_cons] at distinctTags
      rw [List.flatMap_cons, List.nodup_append]
      refine ⟨taggedRange_nodup specification.1 specification.2,
        taggedBlocks_nodup rest distinctTags.2, ?_⟩
      intro key keyInHead otherKey keyInRest keysEqual
      simp only [List.mem_flatMap] at keyInRest
      obtain ⟨other, otherMember, keyInOther⟩ := keyInRest
      have tagsDiffer : specification.1 ≠ other.1 := by
        intro tagsEqual
        apply distinctTags.1
        exact List.mem_map.mpr ⟨other, otherMember, tagsEqual.symm⟩
      subst otherKey
      exact List.disjoint_left.mp
        (taggedBlock_disjoint_of_tag_ne tagsDiffer) keyInHead keyInOther

private theorem mappedExpandedPositions_eq_taggedBlocks (d : Graph)
    (c : Certificate) :
    (expandedSemanticPositions d c).map verificationPositionKey =
      ([(0, 1), (1, d.EDGES.length), (2, c.edgeRoutes.length),
        (3, d.NODES.length), (4, c.terminalOutputs.length)]).flatMap
          fun specification => taggedBlock specification.1 specification.2 := by
  simp [expandedSemanticPositions, indexedMap, taggedBlock,
    verificationPositionKey, List.map_map, Function.comp_def,
    map_zipIdx_snd_apply]

private theorem mappedExpandedPositions_nodup (d : Graph) (c : Certificate) :
    ((expandedSemanticPositions d c).map verificationPositionKey).Nodup := by
  rw [mappedExpandedPositions_eq_taggedBlocks d c]
  apply taggedBlocks_nodup
  simp

private theorem mappedExpandedPosition_tag_le (d : Graph) (c : Certificate)
    {key : Nat × Nat}
    (member : key ∈
      (expandedSemanticPositions d c).map verificationPositionKey) :
    key.1 ≤ 4 := by
  rw [mappedExpandedPositions_eq_taggedBlocks d c] at member
  simp [taggedBlock] at member
  rcases member with h | h | h | h | h <;>
    obtain ⟨_, _, rfl⟩ := h <;> omega

private theorem mappedLocalPosition_tag_ge (d : Graph) (c : Certificate)
    {key : Nat × Nat}
    (member : key ∈
      (localSemanticPositions d c).map verificationPositionKey) :
    5 ≤ key.1 := by
  obtain ⟨position, positionMember, keyEquality⟩ := List.mem_map.mp member
  rw [← keyEquality]
  have localOnly :
      (localSemanticPositions d c).all (fun candidate =>
        match candidate with
        | .graphIdentity
        | .expandedEdgeWitnessCoverage _ _
        | .expandedWitnessScope _ _
        | .expandedRootTerminalCoverage _ _
        | .expandedTerminalScope _ _ => false
        | _ => true) = true := by
    unfold localSemanticPositions
    split <;>
      simp [rawLocalSemanticPositions, repeatedInvariantPositions,
        slotMap, indexedMap]
  have classified := (List.all_eq_true.mp localOnly) position positionMember
  cases position <;> simp [verificationPositionKey] at classified ⊢

private def prefixSemanticPositions (d : Graph) (c : Certificate) :
    List VerificationPosition :=
  slotMap (d.EDGES.length - 1) .edgeNodupPrefix ++
      slotMap d.EDGES.length .edgeNodupMain ++
      slotMap (d.EDGES.length - 1) .oneEdgePrefix ++
      slotMap d.EDGES.length .oneEdgeMain ++
      slotMap (d.NODES.length - 1) .levelFormulaUniquePrefix ++
      slotMap d.NODES.length .levelFormulaUniqueMain ++
      indexedMap d.EDGES .localEdgeWitnessCoverage ++
      indexedMap d.EDGES .structuralEdge ++
      indexedMap c.edgeRoutes .witnessFaithful

private def prefixPositionSpecifications (d : Graph) (c : Certificate) :
    List (Nat × Nat) :=
  [(5, d.EDGES.length - 1), (6, d.EDGES.length),
   (7, d.EDGES.length - 1), (8, d.EDGES.length),
   (9, d.NODES.length - 1), (10, d.NODES.length),
   (11, d.EDGES.length), (12, d.EDGES.length),
   (13, c.edgeRoutes.length)]

private theorem mappedPrefixPositions_eq_taggedBlocks (d : Graph)
    (c : Certificate) :
    (prefixSemanticPositions d c).map verificationPositionKey =
      (prefixPositionSpecifications d c).flatMap fun specification =>
        taggedBlock specification.1 specification.2 := by
  simp [prefixSemanticPositions, prefixPositionSpecifications,
    slotMap, indexedMap, taggedBlock, verificationPositionKey,
    List.map_map, Function.comp_def, map_zipIdx_snd_apply]

private theorem mappedPrefixPositions_nodup (d : Graph) (c : Certificate) :
    ((prefixSemanticPositions d c).map verificationPositionKey).Nodup := by
  rw [mappedPrefixPositions_eq_taggedBlocks d c]
  apply taggedBlocks_nodup
  simp [prefixPositionSpecifications]

private theorem mappedLocalPositions_nodup (d : Graph) (c : Certificate) :
    ((localSemanticPositions d c).map verificationPositionKey).Nodup := by
  unfold localSemanticPositions
  split
  · simp [slotMap, verificationPositionKey, Function.comp_def]
  · have nodeNodup := mappedNodePositions_nodup d.NODES
    let prefixPositions : List VerificationPosition :=
      prefixSemanticPositions d c
    let nodePositions : List VerificationPosition :=
      d.NODES.zipIdx.flatMap fun entry =>
        [.structuralNode entry.2 entry.1,
         .hypothesisSource entry.2 entry.1,
         .nodeProduction entry.2 entry.1,
         .routeCoverage entry.2 entry.1,
         .tailClosure entry.2 entry.1,
         .rootDischarge entry.2 entry.1]
    have serialization : rawLocalSemanticPositions d c =
        prefixPositions ++ nodePositions := by
      simp [rawLocalSemanticPositions, repeatedInvariantPositions,
        prefixPositions, prefixSemanticPositions, nodePositions,
        List.append_assoc]
    rw [serialization, List.map_append]
    have prefixNodup :
        (prefixPositions.map verificationPositionKey).Nodup := by
      exact mappedPrefixPositions_nodup d c
    apply prefixNodup.append nodeNodup
    rw [List.disjoint_left]
    intro key prefixMember nodeMember
    have prefixTag : key.1 ≤ 13 := by
      have taggedMember := prefixMember
      dsimp [prefixPositions] at taggedMember
      rw [mappedPrefixPositions_eq_taggedBlocks d c] at taggedMember
      obtain ⟨specification, specificationMember, keyMember⟩ :=
        List.mem_flatMap.mp taggedMember
      have tagBound : specification.1 ≤ 13 := by
        change specification ∈
          [(5, d.EDGES.length - 1), (6, d.EDGES.length),
           (7, d.EDGES.length - 1), (8, d.EDGES.length),
           (9, d.NODES.length - 1), (10, d.NODES.length),
           (11, d.EDGES.length), (12, d.EDGES.length),
           (13, c.edgeRoutes.length)] at specificationMember
        simp at specificationMember
        rcases specificationMember with rfl | rfl | rfl | rfl | rfl |
          rfl | rfl | rfl | rfl <;> omega
      rw [fst_eq_tag_of_mem_taggedBlock keyMember]
      exact tagBound
    have nodeTag : 14 ≤ key.1 := by
      obtain ⟨position, positionMember, keyEquality⟩ :=
        List.mem_map.mp nodeMember
      rw [← keyEquality]
      simp only [List.mem_flatMap] at positionMember
      obtain ⟨entry, _, positionMember⟩ := positionMember
      simp at positionMember
      rcases positionMember with rfl | rfl | rfl | rfl | rfl | rfl <;>
        simp [verificationPositionKey]
    omega

/-- The semantic serialization never aliases two logical checks.  In
particular duplicate graph values remain distinct through their occurrence
number, while distinct check families have distinct constructors. -/
theorem semanticPositions_nodup (d : Graph) (c : Certificate) :
    (semanticPositions d c).Nodup := by
  apply List.Nodup.of_map verificationPositionKey
  unfold semanticPositions
  rw [List.map_append]
  apply (mappedExpandedPositions_nodup d c).append
    (mappedLocalPositions_nodup d c)
  rw [List.disjoint_left]
  intro key expandedMember localMember
  have upper := mappedExpandedPosition_tag_le d c expandedMember
  have lower := mappedLocalPosition_tag_ge d c localMember
  omega

/-- Numeric serialization of one stable position. -/
def semanticPositionIndex (d : Graph) (c : Certificate)
    (position : VerificationPosition) : Nat :=
  (semanticPositions d c).idxOf position

theorem semanticPositionIndex_lt (d : Graph) (c : Certificate)
    {position : VerificationPosition}
    (member : position ∈ semanticPositions d c) :
    semanticPositionIndex d c position < (semanticPositions d c).length := by
  simpa [semanticPositionIndex] using
    (List.idxOf_lt_length_iff.mpr member)

/-- Semantic-to-numeric checker agreement. -/
theorem checkSemanticPositionB_eq_totalCheckPositionB_index (d : Graph)
    (c : Certificate) {position : VerificationPosition}
    (member : position ∈ semanticPositions d c) :
    checkSemanticPositionB d c position =
      totalCheckPositionB d c (semanticPositionIndex d c position) := by
  have inBounds := semanticPositionIndex_lt d c member
  have found :
      (semanticPositions d c)[semanticPositionIndex d c position] =
        position := List.getElem_idxOf inBounds
  conv_lhs => rw [← found]
  exact checkSemanticPositionB_eq_totalCheckPositionB d c _ inBounds

/-- The semantic serialization and numeric valid positions are inverse. -/
theorem semanticPositionIndex_getElem (d : Graph) (c : Certificate)
    (index : Nat) (inBounds : index < (semanticPositions d c).length) :
    semanticPositionIndex d c (semanticPositions d c)[index] = index := by
  exact List.Nodup.idxOf_getElem (semanticPositions_nodup d c) index inBounds

private theorem expandedCertB_complete (d : Graph) (c : Certificate)
    (expanded : Certificate.Flow.ExpandedCert d c) :
    Certificate.Flow.expandedCertB d c = true := by
  rcases expanded with
    ⟨graphIdentity, witnessCoverage, witnessScope, rootCoverage,
      terminalScope⟩
  unfold Certificate.Flow.expandedCertB
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq,
    List.contains_eq_mem]
  refine ⟨⟨⟨⟨graphIdentity, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · intro edge edgeMember
    simpa [witnessCoverageAtB] using witnessCoverage edge edgeMember
  · exact witnessScope
  · intro vertex vertexMember
    by_cases terminal : (get_rule.outgoing vertex d).isEmpty = true
    · simp [terminal, rootCoverage vertex vertexMember terminal]
    · simp [Bool.eq_false_of_not_eq_true terminal]
  · exact terminalScope

private theorem totalCheckPositionB_true_of_formationValid (d : Graph)
    (c : Certificate) (valid : FormationValid d c) (index : Nat)
    (inBounds : index < totalNumPositions d c) :
    totalCheckPositionB d c index = true := by
  by_cases expandedIndex : index < expandedNumPositions d c
  · unfold totalCheckPositionB
    rw [dif_pos expandedIndex]
    have allExpanded : (expandedPositionChecks d c).all id = true := by
      rw [expandedPositionChecks_all]
      exact expandedCertB_complete d c valid.1
    rw [List.all_eq_true] at allExpanded
    exact allExpanded _ (List.getElem_mem (by
      simpa [expandedNumPositions] using expandedIndex))
  · have localIndexBound :
        index - expandedNumPositions d c < numPositions d c := by
      unfold totalNumPositions at inBounds
      omega
    have notMarked :=
      (no_marked_iff_localChecks d c valid.1).mpr valid.2
        (index - expandedNumPositions d c) localIndexBound
    unfold totalCheckPositionB
    rw [dif_neg expandedIndex]
    cases checkValue : checkPositionB d c
        (index - expandedNumPositions d c) with
    | false =>
        exact False.elim (notMarked (by simpa [MarkedAt] using checkValue))
    | true => rfl

/-- Every complete-formation failure has a stable semantic witness. -/
theorem formation_invalid_implies_semanticMarked (d : Graph)
    (c : Certificate) (invalid : ¬ FormationValid d c) :
    ∃ position, position ∈ semanticPositions d c ∧
      SemanticMarkedAt d c position := by
  obtain ⟨index, inBounds, marked⟩ :=
    total_formation_invalid_implies_marked d c invalid
  have semanticBound : index < (semanticPositions d c).length := by
    simpa [semanticPositions_length] using inBounds
  refine ⟨(semanticPositions d c)[index],
    List.getElem_mem semanticBound, ?_⟩
  unfold SemanticMarkedAt
  unfold TotalMarkedAt at marked
  rw [checkSemanticPositionB_eq_totalCheckPositionB d c index semanticBound]
  exact marked

/-- Complete formation is exactly the absence of failed semantic positions. -/
theorem formationValid_iff_no_semanticMarked (d : Graph)
    (c : Certificate) :
    FormationValid d c ↔
      ∀ position ∈ semanticPositions d c,
        ¬ SemanticMarkedAt d c position := by
  constructor
  · intro valid position member marked
    have indexBound := semanticPositionIndex_lt d c member
    have checkerAgreement :=
      checkSemanticPositionB_eq_totalCheckPositionB_index d c member
    have passes := totalCheckPositionB_true_of_formationValid d c valid
      (semanticPositionIndex d c position) (by
        simpa [semanticPositions_length] using indexBound)
    unfold SemanticMarkedAt at marked
    rw [checkerAgreement, passes] at marked
    contradiction
  · intro noMarked
    by_contra invalid
    obtain ⟨position, member, marked⟩ :=
      formation_invalid_implies_semanticMarked d c invalid
    exact noMarked position member marked

/-!
The original semantic positions are complete for `FormationValid`.  Universal
raw-Flow root discharge needs one additional, genuinely certificate-local
position per certificate node.  Keeping it in a wrapper type preserves the
stable legacy serialization above.
-/

inductive StrongVerificationPosition where
  | base (position : VerificationPosition)
  | universalRootDischarge (nodeSlot : Nat) (node : Vertex)
  deriving Repr, DecidableEq

def checkStrongSemanticPositionB (d : Graph) (c : Certificate) :
    StrongVerificationPosition → Bool
  | .base position => checkSemanticPositionB d c position
  | .universalRootDischarge _ vertex =>
      Certificate.Flow.universalLocalRootDischargeAtB c vertex

def strongRootSemanticPositions (c : Certificate) :
    List StrongVerificationPosition :=
  c.graph.NODES.zipIdx.map fun entry =>
    .universalRootDischarge entry.2 entry.1

def strongSemanticPositions (d : Graph) (c : Certificate) :
    List StrongVerificationPosition :=
  (semanticPositions d c).map .base ++ strongRootSemanticPositions c

def StrongSemanticMarkedAt (d : Graph) (c : Certificate)
    (position : StrongVerificationPosition) : Prop :=
  checkStrongSemanticPositionB d c position = false

/-- Strong formation, including the new universal local root obligation, is
exactly the absence of a failed strong semantic position. -/
theorem strongFormationValid_iff_no_strongSemanticMarked (d : Graph)
    (c : Certificate) :
    Certificate.Flow.StrongFormationValid d c ↔
      ∀ position ∈ strongSemanticPositions d c,
        ¬ StrongSemanticMarkedAt d c position := by
  constructor
  · rintro ⟨expanded, hlocal, universal⟩ position member marked
    rw [strongSemanticPositions, List.mem_append] at member
    rcases member with baseMember | rootMember
    · obtain ⟨base, baseMember', baseEq⟩ := List.mem_map.mp baseMember
      subst position
      exact (formationValid_iff_no_semanticMarked d c).mp
        ⟨expanded, hlocal⟩ base baseMember' marked
    · obtain ⟨entry, entryMember, entryEq⟩ :=
        List.mem_map.mp rootMember
      subst position
      rcases entry with ⟨node, slot⟩
      obtain ⟨slotBound, nodeEq⟩ := List.mem_zipIdx' entryMember
      have nodeMember : node ∈ c.graph.NODES := by
        rw [nodeEq]
        exact List.getElem_mem slotBound
      have passed :
          Certificate.Flow.universalLocalRootDischargeAtB c node = true :=
        (Certificate.Flow.universalLocalRootDischargeAtB_eq_true_iff c node).mpr
          (universal node nodeMember)
      unfold StrongSemanticMarkedAt at marked
      simp only [checkStrongSemanticPositionB] at marked
      rw [passed] at marked
      contradiction
  · intro noMarked
    have formation : FormationValid d c :=
      (formationValid_iff_no_semanticMarked d c).mpr (by
        intro position member marked
        exact noMarked (.base position) (by
          rw [strongSemanticPositions, List.mem_append]
          exact Or.inl (List.mem_map.mpr ⟨position, member, rfl⟩)) (by
            simpa [StrongSemanticMarkedAt, checkStrongSemanticPositionB]
              using marked))
    have universal : Certificate.Flow.UniversalLocalRootDischarge c := by
      intro node nodeMember
      apply (Certificate.Flow.universalLocalRootDischargeAtB_eq_true_iff
        c node).mp
      cases checked :
          Certificate.Flow.universalLocalRootDischargeAtB c node with
      | true => rfl
      | false =>
          obtain ⟨slot, slotBound, nodeAtSlot⟩ :=
            (List.mem_iff_getElem.mp nodeMember)
          have zipMember : (node, slot) ∈ c.graph.NODES.zipIdx := by
            rw [List.mem_iff_getElem]
            refine ⟨slot, by simpa using slotBound, ?_⟩
            rw [List.getElem_zipIdx]
            simp [nodeAtSlot]
          have positionMember :
              CertificateSearch.StrongVerificationPosition.universalRootDischarge slot node ∈
                strongSemanticPositions d c := by
            rw [strongSemanticPositions, List.mem_append]
            exact Or.inr (List.mem_map.mpr
              ⟨(node, slot), zipMember, rfl⟩)
          have marked : StrongSemanticMarkedAt d c
              (.universalRootDischarge slot node) := by
            simpa [StrongSemanticMarkedAt, checkStrongSemanticPositionB]
              using checked
          exact False.elim
            (noMarked (.universalRootDischarge slot node)
              positionMember marked)
    exact ⟨formation.1, formation.2, universal⟩

/-- Every failure of strong formation has a finite, stable semantic witness. -/
theorem strongFormation_invalid_implies_strongSemanticMarked (d : Graph)
    (c : Certificate)
    (invalid : ¬ Certificate.Flow.StrongFormationValid d c) :
    ∃ position, position ∈ strongSemanticPositions d c ∧
      StrongSemanticMarkedAt d c position := by
  by_contra noWitness
  apply invalid
  apply (strongFormationValid_iff_no_strongSemanticMarked d c).mpr
  intro position member marked
  exact noWitness ⟨position, member, marked⟩


end CertificateSearch
