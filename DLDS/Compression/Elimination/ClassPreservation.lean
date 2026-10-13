import DLDS.Compression.Elimination.PairPreservation
import DLDS.Compression.Sequences

/-!
# Repeated homogeneous elimination-class preservation

This module extends the two-member elimination/elimination result to the
left-associated state produced while one homogeneous class is reduced. The
proof-facing occurrence data below is not runtime input: it names the original
logical occurrences and supplies independently computed premise dependencies.
The production compressor continues to operate only on `Neighborhood` and
`Ancestral` records.
-/

open DLDS FlowSemantics
open DLDS.Compression.Mixed.LocalRules DLDS.Compression.Mixed.ContextualElimination
open DLDS.Compression.Sequences

namespace DLDS.Compression.Elimination.ClassPreservation

/-- Proof-side description of one original implication-elimination
occurrence. Stored edge payloads and independently computed premise
dependencies are deliberately separate fields. -/
structure ElimOccurrence where
  original : Vertex
  major : Vertex
  minor : Vertex
  use : Vertex
  majorStored : Dep
  minorStored : Dep
  useStored : Dep
  majorDep : Dep
  minorDep : Dep
deriving Repr, DecidableEq

def ElimOccurrence.tag (occurrence : ElimOccurrence) : Nat :=
  occurrence.original.NUMBER

def ElimOccurrence.route (occurrence : ElimOccurrence) : CandidateRoute :=
  { production := .elimination
    dependency := depUnion occurrence.minorDep occurrence.majorDep
    residual := { colours := [occurrence.tag] } }

def ElimOccurrence.outgoing (center : Vertex)
    (occurrence : ElimOccurrence) : Deduction :=
  edge center occurrence.use occurrence.tag occurrence.useStored

def ElimOccurrence.observation (conclusion : Formula)
    (occurrence : ElimOccurrence) : BoundaryObservation :=
  { production := .elimination
    conclusion := conclusion
    use := occurrence.use
    dependency := depUnion occurrence.minorDep occurrence.majorDep
    residual := { colours := [] } }

def occurrenceRoutes (occurrences : List ElimOccurrence) :
    List CandidateRoute :=
  occurrences.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.route

def occurrenceOutgoing (center : Vertex)
    (occurrences : List ElimOccurrence) : List Deduction :=
  occurrences.map (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing center)

def occurrenceObservations (conclusion : Formula)
    (occurrences : List ElimOccurrence) : List BoundaryObservation :=
  occurrences.map (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.observation conclusion)

def ElimOccurrence.incoming (center : Vertex)
    (occurrence : ElimOccurrence) : List Deduction :=
  [edge occurrence.major center 0 occurrence.majorStored,
   edge occurrence.minor center 0 occurrence.minorStored]

def ElimOccurrence.indirect (occurrence : ElimOccurrence) : List Ancestral :=
  [path occurrence.use occurrence.major [0, occurrence.tag],
   path occurrence.use occurrence.minor [0, occurrence.tag]]

/-- The exact `pre_collapse` shape of one ordinary uncollapsed elimination
member. -/
def ElimOccurrence.prepared (occurrence : ElimOccurrence) : Neighborhood :=
  rule occurrence.original
    (occurrence.incoming occurrence.original)
    [occurrence.outgoing occurrence.original]
    [] occurrence.indirect

/-- Exact accumulated neighbourhood shape, in the producer's newest-first
enumeration order. -/
def accumulatedNeighborhood (center : Vertex)
    (occurrences : List ElimOccurrence) : Neighborhood :=
  rule center
    (occurrences.flatMap (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming center))
    (occurrenceOutgoing center occurrences)
    [] (occurrences.flatMap DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.indirect)

@[simp] theorem rewrite_incoming_occurrence
    (newCenter oldCenter : Vertex) (occurrence : ElimOccurrence) :
    collapse.rewrite_incoming newCenter
        (occurrence.incoming oldCenter) =
      occurrence.incoming newCenter := by
  simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming, collapse.rewrite_incoming]

@[simp] theorem rewrite_incoming_occurrences
    (newCenter oldCenter : Vertex) (occurrences : List ElimOccurrence) :
    collapse.rewrite_incoming newCenter
        (occurrences.flatMap (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming oldCenter)) =
      occurrences.flatMap (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming newCenter) := by
  induction occurrences with
  | nil => simp [collapse.rewrite_incoming]
  | cons occurrence rest ih =>
      simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming, collapse.rewrite_incoming, ih]

@[simp] theorem rewrite_outgoing_occurrence
    (newCenter oldCenter : Vertex) (occurrence : ElimOccurrence) :
    collapse.rewrite_outgoing newCenter
        [occurrence.outgoing oldCenter] =
      [occurrence.outgoing newCenter] := by
  simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing, collapse.rewrite_outgoing]

@[simp] theorem rewrite_outgoing_occurrences
    (newCenter oldCenter : Vertex) (occurrences : List ElimOccurrence) :
    collapse.rewrite_outgoing newCenter
        (occurrenceOutgoing oldCenter occurrences) =
      occurrenceOutgoing newCenter occurrences := by
  induction occurrences with
  | nil => simp [occurrenceOutgoing, collapse.rewrite_outgoing]
  | cons occurrence rest ih =>
      unfold occurrenceOutgoing at ih
      unfold occurrenceOutgoing
      simp only [List.map_cons, collapse.rewrite_outgoing]
      congr 1

private theorem eraseDups_eq_self_of_nodup
    {alpha : Type} [BEq alpha] [LawfulBEq alpha]
    (values : List alpha) (nodup : values.Nodup) :
    values.eraseDups = values := by
  induction values with
  | nil => rfl
  | cons value rest ih =>
      have valueAbsent : Not (List.Mem value rest) :=
        (List.nodup_cons.mp nodup).1
      have restNodup : rest.Nodup := (List.nodup_cons.mp nodup).2
      rw [List.eraseDups_cons]
      have filtered : rest.filter (fun candidate => !candidate == value) =
          rest := by
        apply List.filter_eq_self.mpr
        intro candidate member
        have different : Not (candidate = value) := by
          intro equal
          apply valueAbsent
          simpa [equal] using member
        simp [different]
      rw [filtered, ih restNodup]

/-- Structural facts needed by the raw producer equation.  These facts speak
only about the candidate neighbourhood assembled from current records.  Later
class invariants derive them from occurrence freshness and incidence. -/
structure ProducerStepConditions
    (partition : List HCClass) (currentCenter : Vertex)
    (processed : List ElimOccurrence) (next : ElimOccurrence) : Prop where
  noRepresentativeConflict :
    pre_collapse.outgoingRepresentativeConflictB partition
      (accumulatedNeighborhood currentCenter processed) next.prepared = false
  incomingNodup :
    (next.incoming (collapse.center currentCenter next.original) ++
      processed.flatMap
        (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming
          (collapse.center currentCenter next.original))).Nodup
  outgoingNodup :
    (next.outgoing (collapse.center currentCenter next.original) ::
      occurrenceOutgoing (collapse.center currentCenter next.original)
        processed).Nodup
  pathHeadsOK :
    collapse.neighborhoodPathHeadsOKB
      (accumulatedNeighborhood
        (collapse.center currentCenter next.original)
        (next :: processed)) = true
  outgoingRepresentativesOK :
    collapse.outgoingRepresentativesOKB partition
      (accumulatedNeighborhood
        (collapse.center currentCenter next.original)
        (next :: processed)) = true

/-- Exact producer equation for an already-accumulated elimination prefix
and the next prepared elimination. No semantic boundary equality is assumed. -/
theorem collapse_accumulated_with_next_eq
    (partition : List HCClass) (currentCenter : Vertex)
    (processed : List ElimOccurrence) (next : ElimOccurrence)
    (conditions : ProducerStepConditions partition currentCenter
      processed next) :
    collapseWithOccurrences partition
        (accumulatedNeighborhood currentCenter processed) next.prepared =
      (accumulatedNeighborhood
        (collapse.center currentCenter next.original)
        (next :: processed), []) := by
  have incomingErase :
      List.eraseDups
        (next.incoming (collapse.center currentCenter next.original) ++
          processed.flatMap
            (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming
              (collapse.center currentCenter next.original))) =
        next.incoming (collapse.center currentCenter next.original) ++
          processed.flatMap
            (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming
              (collapse.center currentCenter next.original)) :=
    eraseDups_eq_self_of_nodup _ conditions.incomingNodup
  have outgoingErase :
      List.eraseDups
        (next.outgoing (collapse.center currentCenter next.original) ::
          occurrenceOutgoing (collapse.center currentCenter next.original)
            processed) =
        next.outgoing (collapse.center currentCenter next.original) ::
          occurrenceOutgoing (collapse.center currentCenter next.original)
            processed :=
    eraseDups_eq_self_of_nodup _ conditions.outgoingNodup
  have preparedPair :
      pre_collapse.discriminatePair partition
          (accumulatedNeighborhood currentCenter processed) next.prepared =
        (accumulatedNeighborhood currentCenter processed, next.prepared) := by
    simp [pre_collapse.discriminatePair,
      conditions.noRepresentativeConflict]
  let newCenter := collapse.center currentCenter next.original
  let candidate :=
    rule newCenter
      (List.eraseDups
        (collapse.rewrite_incoming newCenter next.prepared.INCOMING ++
          collapse.rewrite_incoming newCenter
            (accumulatedNeighborhood currentCenter processed).INCOMING))
      (List.eraseDups
        (collapse.rewrite_outgoing newCenter next.prepared.OUTGOING ++
          collapse.rewrite_outgoing newCenter
            (accumulatedNeighborhood currentCenter processed).OUTGOING))
      (collapse.rewrite_direct newCenter next.prepared.OUTGOING
          next.prepared.DIRECT ++
        collapse.rewrite_direct newCenter
          (accumulatedNeighborhood currentCenter processed).OUTGOING
          (accumulatedNeighborhood currentCenter processed).DIRECT)
      (next.prepared.INDIRECT ++
        (accumulatedNeighborhood currentCenter processed).INDIRECT)
  have actualEq :
      collapseWithOccurrences partition
          (accumulatedNeighborhood currentCenter processed) next.prepared =
        (collapse.checkOutgoingRepresentatives partition
          (collapse.checkNeighborhoodPathHeads candidate), []) := by
    unfold collapseWithOccurrences
    rw [preparedPair]
    rfl
  rw [actualEq]
  have candidateEq :
      candidate =
      rule (collapse.center currentCenter next.original)
          (List.eraseDups
            (collapse.rewrite_incoming
                (collapse.center currentCenter next.original)
                next.prepared.INCOMING ++
              collapse.rewrite_incoming
                (collapse.center currentCenter next.original)
                (accumulatedNeighborhood currentCenter processed).INCOMING))
          (List.eraseDups
            (collapse.rewrite_outgoing
                (collapse.center currentCenter next.original)
                next.prepared.OUTGOING ++
              collapse.rewrite_outgoing
                (collapse.center currentCenter next.original)
                (accumulatedNeighborhood currentCenter processed).OUTGOING))
          (collapse.rewrite_direct
              (collapse.center currentCenter next.original)
              next.prepared.OUTGOING next.prepared.DIRECT ++
            collapse.rewrite_direct
              (collapse.center currentCenter next.original)
              (accumulatedNeighborhood currentCenter processed).OUTGOING
              (accumulatedNeighborhood currentCenter processed).DIRECT)
          (next.prepared.INDIRECT ++
            (accumulatedNeighborhood currentCenter processed).INDIRECT) := by
    rfl
  rw [candidateEq]
  have candidateShape :
      rule (collapse.center currentCenter next.original)
          (List.eraseDups
            (collapse.rewrite_incoming
                (collapse.center currentCenter next.original)
                next.prepared.INCOMING ++
              collapse.rewrite_incoming
                (collapse.center currentCenter next.original)
                (accumulatedNeighborhood currentCenter processed).INCOMING))
          (List.eraseDups
            (collapse.rewrite_outgoing
                (collapse.center currentCenter next.original)
                next.prepared.OUTGOING ++
              collapse.rewrite_outgoing
                (collapse.center currentCenter next.original)
                (accumulatedNeighborhood currentCenter processed).OUTGOING))
          (collapse.rewrite_direct
              (collapse.center currentCenter next.original)
              next.prepared.OUTGOING next.prepared.DIRECT ++
            collapse.rewrite_direct
              (collapse.center currentCenter next.original)
              (accumulatedNeighborhood currentCenter processed).OUTGOING
              (accumulatedNeighborhood currentCenter processed).DIRECT)
          (next.prepared.INDIRECT ++
            (accumulatedNeighborhood currentCenter processed).INDIRECT) =
        accumulatedNeighborhood
          (collapse.center currentCenter next.original)
          (next :: processed) := by
    simp [accumulatedNeighborhood, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared,
      incomingErase, outgoingErase, collapse.rewrite_direct]
    rfl
  rw [candidateShape]
  have headsChecked :
      collapse.checkNeighborhoodPathHeads
          (accumulatedNeighborhood
            (collapse.center currentCenter next.original)
            (next :: processed)) =
        accumulatedNeighborhood
          (collapse.center currentCenter next.original)
          (next :: processed) := by
    simp [collapse.checkNeighborhoodPathHeads, conditions.pathHeadsOK]
  rw [headsChecked]
  have representativesChecked :
      collapse.checkOutgoingRepresentatives partition
          (accumulatedNeighborhood
            (collapse.center currentCenter next.original)
            (next :: processed)) =
        accumulatedNeighborhood
          (collapse.center currentCenter next.original)
          (next :: processed) := by
    simp [collapse.checkOutgoingRepresentatives,
      conditions.outgoingRepresentativesOK]
  rw [representativesChecked]

/-- Pairwise freshness of the occurrence colours retained by the actual
compressor. Formulae and dependency vectors play no role in this relation. -/
def PairwiseDistinctTags (occurrences : List ElimOccurrence) : Prop :=
  occurrences.Pairwise fun left right => Not (left.tag = right.tag)

theorem PairwiseDistinctTags.tail
    {occurrence : ElimOccurrence} {rest : List ElimOccurrence}
    (distinct : PairwiseDistinctTags (occurrence :: rest)) :
    PairwiseDistinctTags rest :=
  (List.pairwise_cons.mp distinct).2

theorem occurrenceOutgoing_nodup
    (center : Vertex) (occurrences : List ElimOccurrence)
    (distinct : PairwiseDistinctTags occurrences) :
    (occurrenceOutgoing center occurrences).Nodup := by
  induction occurrences with
  | nil => simp [occurrenceOutgoing]
  | cons occurrence rest ih =>
      have headDistinct : forall candidate, List.Mem candidate rest ->
          Not (occurrence.tag = candidate.tag) := by
        intro candidate member
        exact (List.pairwise_cons.mp distinct).1 candidate member
      have headAbsent : Not (List.Mem (occurrence.outgoing center)
          (occurrenceOutgoing center rest)) := by
        intro member
        obtain existsCandidate := List.mem_map.mp member
        cases existsCandidate with
        | intro candidate conjunction =>
            cases conjunction with
            | intro candidateMember edgeEqual =>
                have colourEqual := congrArg Deduction.COLOUR edgeEqual
                have tagEqual : candidate.tag = occurrence.tag := by
                  simpa [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing] using colourEqual
                exact headDistinct candidate candidateMember tagEqual.symm
      rw [show occurrenceOutgoing center (occurrence :: rest) =
          occurrence.outgoing center :: occurrenceOutgoing center rest by rfl]
      exact List.nodup_cons.mpr
        (And.intro headAbsent (ih distinct.tail))

/-- Every generated indirect record has an actual colour-zero incoming edge
with the same original premise endpoint. This is derived from the accumulated
shape, not supplied as a successful post-check. -/
theorem accumulated_pathHeadsOK
    (center : Vertex) (occurrences : List ElimOccurrence)
    (premisesAvoidCenter : forall occurrence,
      List.Mem occurrence occurrences ->
      Not (occurrence.major = center) /\
        Not (occurrence.minor = center)) :
    collapse.neighborhoodPathHeadsOKB
      (accumulatedNeighborhood center occurrences) = true := by
  unfold collapse.neighborhoodPathHeadsOKB
  rw [List.all_eq_true]
  intro ancestral ancestralMember
  simp only [accumulatedNeighborhood, List.nil_append] at ancestralMember
  obtain occurrenceExists := List.mem_flatMap.mp ancestralMember
  cases occurrenceExists with
  | intro occurrence occurrenceData =>
      cases occurrenceData with
      | intro occurrenceMember pathMember =>
          have avoids := premisesAvoidCenter occurrence occurrenceMember
          simp only [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.indirect, List.mem_cons,
            List.not_mem_nil, or_false] at pathMember
          cases pathMember with
          | inl pathEqual =>
              subst ancestral
              simp [collapse.neighborhoodPathHeadOKB,
                accumulatedNeighborhood, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming,
                avoids.1]
              exact Exists.intro occurrence
                (And.intro occurrenceMember (Or.inl rfl))
          | inr pathEqual =>
              subst ancestral
              simp [collapse.neighborhoodPathHeadOKB,
                accumulatedNeighborhood, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming,
                avoids.2]
              exact Exists.intro occurrence
                (And.intro occurrenceMember (Or.inr rfl))

def TagsInjectiveOn (occurrences : List ElimOccurrence) : Prop :=
  forall left, List.Mem left occurrences ->
    forall right, List.Mem right occurrences ->
      left.tag = right.tag -> left = right

theorem pairwiseDistinctTags_injective
    (occurrences : List ElimOccurrence)
    (distinct : PairwiseDistinctTags occurrences) :
    TagsInjectiveOn occurrences := by
  induction occurrences with
  | nil =>
      intro left impossible
      cases impossible
  | cons head tail ih =>
      have headDistinct := (List.pairwise_cons.mp distinct).1
      have tailDistinct := (List.pairwise_cons.mp distinct).2
      intro left leftMember right rightMember tagEqual
      cases leftMember with
      | head =>
          cases rightMember with
          | head => rfl
          | tail _ rightTail =>
              exact False.elim (headDistinct right rightTail tagEqual)
      | tail _ leftTail =>
          cases rightMember with
          | head =>
              exact False.elim (headDistinct left leftTail tagEqual.symm)
          | tail _ rightTail =>
              exact ih tailDistinct left leftTail right rightTail tagEqual

/-- The producer's outgoing-representative guard follows from occurrence-tag
injectivity and the ordinary self-representative fact for every use. -/
theorem accumulated_outgoingRepresentativesOK
    (partition : List HCClass) (center : Vertex)
    (occurrences : List ElimOccurrence)
    (tagsInjective : TagsInjectiveOn occurrences)
    (usesRepresented : forall occurrence,
      List.Mem occurrence occurrences ->
      sameHCRepresentativeB partition occurrence.use occurrence.use = true) :
    collapse.outgoingRepresentativesOKB partition
      (accumulatedNeighborhood center occurrences) = true := by
  unfold collapse.outgoingRepresentativesOKB
  rw [List.all_eq_true]
  intro leftEdge leftEdgeMember
  rw [List.all_eq_true]
  intro rightEdge rightEdgeMember
  have leftExists := List.mem_map.mp leftEdgeMember
  have rightExists := List.mem_map.mp rightEdgeMember
  cases leftExists with
  | intro left leftData =>
      cases leftData with
      | intro leftMember leftEqual =>
          cases rightExists with
          | intro right rightData =>
              cases rightData with
              | intro rightMember rightEqual =>
                  subst leftEdge
                  subst rightEdge
                  by_cases sameTag : left.tag = right.tag
                  case pos =>
                    have sameOccurrence := tagsInjective left leftMember right
                      rightMember sameTag
                    subst right
                    simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing,
                      usesRepresented left leftMember]
                  case neg =>
                    simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing, sameTag]

/-- A fresh occurrence tag makes the pair-discrimination branch inert. The
claim is independent of formulae and dependency payloads. -/
theorem noRepresentativeConflict_of_fresh_tag
    (partition : List HCClass) (currentCenter : Vertex)
    (processed : List ElimOccurrence) (next : ElimOccurrence)
    (fresh : forall occurrence, List.Mem occurrence processed ->
      Not (occurrence.tag = next.tag)) :
    pre_collapse.outgoingRepresentativeConflictB partition
      (accumulatedNeighborhood currentCenter processed) next.prepared =
        false := by
  unfold pre_collapse.outgoingRepresentativeConflictB
  simp only [accumulatedNeighborhood, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared,
    occurrenceOutgoing, List.any_map, List.any_cons, List.any_nil,
    Bool.or_false]
  rw [List.any_eq_false]
  intro occurrence occurrenceMember
  have different := fresh occurrence occurrenceMember
  simp [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing, different]

def DependencyTableCorrect (dependencyFor : Nat -> Dep)
    (occurrences : List ElimOccurrence) : Prop :=
  forall occurrence, List.Mem occurrence occurrences ->
    dependencyFor occurrence.tag =
      depUnion occurrence.minorDep occurrence.majorDep

/-- A uniform proof-only observer for every intermediate state of one class.
It reads the exact produced outgoing records, while obtaining dependencies
from the independent logical table rather than `Deduction.DEPENDENCY`. -/
def observeEliminationBoundary (dependencyFor : Nat -> Dep)
    (neighborhood : Neighborhood) : List BoundaryObservation :=
  neighborhood.OUTGOING.map fun outgoing =>
    { production := .elimination
      conclusion := neighborhood.CENTER.FORMULA
      use := outgoing.END
      dependency := dependencyFor outgoing.COLOUR
      residual := { colours := [] } }

theorem observeEliminationBoundary_accumulated
    (dependencyFor : Nat -> Dep) (center : Vertex)
    (occurrences : List ElimOccurrence)
    (correct : DependencyTableCorrect dependencyFor occurrences) :
    observeEliminationBoundary dependencyFor
        (accumulatedNeighborhood center occurrences) =
      occurrenceObservations center.FORMULA occurrences := by
  induction occurrences with
  | nil => simp [observeEliminationBoundary, accumulatedNeighborhood,
      occurrenceOutgoing, occurrenceObservations]
  | cons occurrence rest ih =>
      have headCorrect := correct occurrence (List.Mem.head rest)
      have tailCorrect : DependencyTableCorrect dependencyFor rest := by
        intro candidate member
        exact correct candidate (List.Mem.tail occurrence member)
      simp [observeEliminationBoundary, accumulatedNeighborhood,
        occurrenceOutgoing, occurrenceObservations,
        DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.observation,
        headCorrect]
      exact tailCorrect

theorem observeEliminationBoundary_prepared
    (dependencyFor : Nat -> Dep) (occurrence : ElimOccurrence)
    (correct : dependencyFor occurrence.tag =
      depUnion occurrence.minorDep occurrence.majorDep) :
    observeEliminationBoundary dependencyFor occurrence.prepared =
      [occurrence.observation occurrence.original.FORMULA] := by
  simp [observeEliminationBoundary, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared,
    DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.observation, correct]

/-- Primitive structural compatibility of the next original occurrence with
the accumulated prefix. No producer result or boundary-preservation proof is
a field. -/
structure CompatibleExtension
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (currentCenter : Vertex) (processed : List ElimOccurrence)
    (next : ElimOccurrence) : Prop where
  distinctTags : PairwiseDistinctTags (next :: processed)
  incomingNodup :
    (next.incoming (collapse.center currentCenter next.original) ++
      processed.flatMap
        (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming
          (collapse.center currentCenter next.original))).Nodup
  premisesAvoidNewCenter : forall occurrence,
    List.Mem occurrence (next :: processed) ->
    Not (occurrence.major = collapse.center currentCenter next.original) /\
      Not (occurrence.minor = collapse.center currentCenter next.original)
  usesRepresented : forall occurrence,
    List.Mem occurrence (next :: processed) ->
    sameHCRepresentativeB partition occurrence.use occurrence.use = true
  dependencyCorrect :
    DependencyTableCorrect dependencyFor (next :: processed)
  sameConclusion : next.original.FORMULA = currentCenter.FORMULA
  sameLevel : next.original.LEVEL = currentCenter.LEVEL
  logicalEliminations : forall occurrence,
    List.Mem occurrence (next :: processed) ->
    occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> currentCenter.FORMULA

/-- The static compatibility fields derive every guard and deduplication fact
used by the real producer. -/
theorem producerStepConditions_of_compatible
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (currentCenter : Vertex) (processed : List ElimOccurrence)
    (next : ElimOccurrence)
    (compatible : CompatibleExtension partition dependencyFor
      currentCenter processed next) :
    ProducerStepConditions partition currentCenter processed next := by
  have fresh : forall occurrence, List.Mem occurrence processed ->
      Not (occurrence.tag = next.tag) := by
    intro occurrence member
    exact Ne.symm
      ((List.pairwise_cons.mp compatible.distinctTags).1 occurrence member)
  exact
    { noRepresentativeConflict :=
        noRepresentativeConflict_of_fresh_tag partition currentCenter
          processed next fresh
      incomingNodup := compatible.incomingNodup
      outgoingNodup :=
        occurrenceOutgoing_nodup
          (collapse.center currentCenter next.original)
          (next :: processed) compatible.distinctTags
      pathHeadsOK :=
        accumulated_pathHeadsOK
          (collapse.center currentCenter next.original)
          (next :: processed) compatible.premisesAvoidNewCenter
      outgoingRepresentativesOK :=
        accumulated_outgoingRepresentativesOK partition
          (collapse.center currentCenter next.original)
          (next :: processed)
          (pairwiseDistinctTags_injective _ compatible.distinctTags)
          compatible.usesRepresented }

/-- A current accumulated state relative to its represented original
elimination occurrences. The shape fields are exact operational records; the
dependency table is logical and independent of stored edge labels. -/
structure AccumulatorInvariant
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (current : Neighborhood)
    (processed : List ElimOccurrence) : Prop where
  nonempty : Not (processed = [])
  exactShape :
    current = accumulatedNeighborhood current.CENTER processed
  distinctTags : PairwiseDistinctTags processed
  tagsInjective : TagsInjectiveOn processed
  incomingNodup :
    (processed.flatMap
      (DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming current.CENTER)).Nodup
  premisesAvoidCenter : forall occurrence,
    List.Mem occurrence processed ->
    Not (occurrence.major = current.CENTER) /\
      Not (occurrence.minor = current.CENTER)
  usesRepresented : forall occurrence,
    List.Mem occurrence processed ->
    sameHCRepresentativeB partition occurrence.use occurrence.use = true
  dependencyCorrect : DependencyTableCorrect dependencyFor processed
  centerFormula : current.CENTER.FORMULA = conclusion
  logicalEliminations : forall occurrence,
    List.Mem occurrence processed ->
    occurrence.major.FORMULA =
      occurrence.minor.FORMULA >> conclusion

/-- The boundary observation of one real producer step is preserved as a
set.  The only permutation is the producer's newest-first accumulation
order.  Dependencies in both sides come from `dependencyFor`. -/
theorem observeEliminationBoundary_step_setEq
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (currentCenter : Vertex) (processed : List ElimOccurrence)
    (next : ElimOccurrence)
    (compatible : CompatibleExtension partition dependencyFor
      currentCenter processed next) :
    Certificate.Flow.ListSetEq
      (observeEliminationBoundary dependencyFor
          (accumulatedNeighborhood currentCenter processed) ++
        observeEliminationBoundary dependencyFor next.prepared)
      (observeEliminationBoundary dependencyFor
        (accumulatedNeighborhood
          (collapse.center currentCenter next.original)
          (next :: processed))) := by
  have oldCorrect : DependencyTableCorrect dependencyFor processed := by
    intro occurrence member
    exact compatible.dependencyCorrect occurrence
      (List.Mem.tail next member)
  have nextCorrect : dependencyFor next.tag =
      depUnion next.minorDep next.majorDep :=
    compatible.dependencyCorrect next (List.Mem.head processed)
  rw [observeEliminationBoundary_accumulated dependencyFor currentCenter
    processed oldCorrect]
  rw [observeEliminationBoundary_prepared dependencyFor next nextCorrect]
  rw [observeEliminationBoundary_accumulated dependencyFor
    (collapse.center currentCenter next.original) (next :: processed)
    compatible.dependencyCorrect]
  intro observation
  simp [occurrenceObservations, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.observation,
    collapse.center, compatible.sameConclusion, or_comm]

/-- One actual selected elimination step preserves the focused accumulator
invariant.  Its operation equation is about the production function itself;
no preservation assertion is hidden in the step data. -/
theorem accumulatorInvariant_step
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (current : Neighborhood)
    (processed : List ElimOccurrence) (next : ElimOccurrence)
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (compatible : CompatibleExtension partition dependencyFor
      current.CENTER processed next) :
    let nextState := accumulatedNeighborhood
      (collapse.center current.CENTER next.original) (next :: processed)
    collapseWithOccurrences partition current next.prepared =
        (nextState, []) /\
      AccumulatorInvariant partition dependencyFor conclusion
        nextState (next :: processed) := by
  let nextState := accumulatedNeighborhood
    (collapse.center current.CENTER next.original) (next :: processed)
  have conditions := producerStepConditions_of_compatible partition
    dependencyFor current.CENTER processed next compatible
  have operation : collapseWithOccurrences partition current next.prepared =
      (nextState, []) := by
    rw [invariant.exactShape]
    exact collapse_accumulated_with_next_eq partition current.CENTER
      processed next conditions
  refine And.intro operation ?_
  refine
    { nonempty := by simp
      exactShape := by rfl
      distinctTags := compatible.distinctTags
      tagsInjective :=
        pairwiseDistinctTags_injective _ compatible.distinctTags
      incomingNodup := ?_
      premisesAvoidCenter := ?_
      usesRepresented := compatible.usesRepresented
      dependencyCorrect := compatible.dependencyCorrect
      centerFormula := ?_
      logicalEliminations := ?_ }
  · simpa [nextState, accumulatedNeighborhood] using
      compatible.incomingNodup
  · simpa [nextState, accumulatedNeighborhood] using
      compatible.premisesAvoidNewCenter
  · simpa [nextState, accumulatedNeighborhood, collapse.center] using
      invariant.centerFormula
  · intro occurrence member
    have logical := compatible.logicalEliminations occurrence member
    simpa [invariant.centerFormula] using logical

/-- Structural schedule for the remaining members of one homogeneous
elimination class.  Constructors carry only the compatibility facts from
which the producer guards are derived; they do not carry step preservation. -/
inductive CompatibleEliminationTail
    (partition : List HCClass) (dependencyFor : Nat -> Dep) :
    Vertex -> List ElimOccurrence -> List ElimOccurrence -> Prop where
  | nil (currentCenter : Vertex) (processed : List ElimOccurrence) :
      CompatibleEliminationTail partition dependencyFor currentCenter
        processed []
  | cons (currentCenter : Vertex) (processed : List ElimOccurrence)
      (next : ElimOccurrence) (rest : List ElimOccurrence)
      (compatible : CompatibleExtension partition dependencyFor
        currentCenter processed next)
      (tail : CompatibleEliminationTail partition dependencyFor
        (collapse.center currentCenter next.original) (next :: processed)
        rest) :
      CompatibleEliminationTail partition dependencyFor currentCenter
        processed (next :: rest)

/-- Static facts which initialize the one-member accumulator. -/
structure InitialEliminationMember
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (first : ElimOccurrence) : Prop where
  incomingNodup : (first.incoming first.original).Nodup
  premisesAvoidCenter :
    Not (first.major = first.original) /\
      Not (first.minor = first.original)
  useRepresented :
    sameHCRepresentativeB partition first.use first.use = true
  dependencyCorrect : dependencyFor first.tag =
    depUnion first.minorDep first.majorDep
  centerFormula : first.original.FORMULA = conclusion
  logicalElimination : first.major.FORMULA =
    first.minor.FORMULA >> conclusion

theorem initialEliminationMember_invariant
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (first : ElimOccurrence)
    (initial : InitialEliminationMember partition dependencyFor
      conclusion first) :
    AccumulatorInvariant partition dependencyFor conclusion first.prepared
      [first] := by
  refine
    { nonempty := by simp
      exactShape := by rfl
      distinctTags := by simp [PairwiseDistinctTags]
      tagsInjective := pairwiseDistinctTags_injective _ (by
        simp [PairwiseDistinctTags])
      incomingNodup := ?_
      premisesAvoidCenter := ?_
      usesRepresented := ?_
      dependencyCorrect := ?_
      centerFormula := initial.centerFormula
      logicalEliminations := ?_ }
  · simpa [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared, accumulatedNeighborhood] using
      initial.incomingNodup
  · intro occurrence member
    cases member with
    | head => simpa [DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared] using initial.premisesAvoidCenter
    | tail _ impossible => cases impossible
  · intro occurrence member
    cases member with
    | head => exact initial.useRepresented
    | tail _ impossible => cases impossible
  · intro occurrence member
    cases member with
    | head => exact initial.dependencyCorrect
    | tail _ impossible => cases impossible
  · intro occurrence member
    cases member with
    | head => exact initial.logicalElimination
    | tail _ impossible => cases impossible

/-- Any finite compatible tail constructs the real left-associated
`ActualCompressionSetTrace`.  The final represented prefix is exactly the
reverse of the consumed schedule followed by the old prefix, matching the
producer's newest-first record order. -/
theorem compatibleEliminationTail_actualTrace
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (current : Neighborhood)
    (processed remaining : List ElimOccurrence)
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (schedule : CompatibleEliminationTail partition dependencyFor
      current.CENTER processed remaining) :
    Exists fun final : Neighborhood => Exists fun finalProcessed : List ElimOccurrence =>
      ActualCompressionSetTrace partition
          (observeEliminationBoundary dependencyFor)
          (current :: remaining.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared) [] final [] /\
        AccumulatorInvariant partition dependencyFor conclusion
          final finalProcessed /\
        finalProcessed = remaining.reverse ++ processed := by
  induction remaining generalizing current processed with
  | nil =>
      cases schedule with
      | nil =>
          exact ⟨current, processed,
            DLDS.Compression.Sequences.ActualCompressionSetTrace.single current, invariant, by simp⟩
  | cons next rest ih =>
      cases schedule with
      | cons _ _ _ _ compatible tail =>
          let nextState := accumulatedNeighborhood
            (collapse.center current.CENTER next.original)
            (next :: processed)
          have step := accumulatorInvariant_step partition dependencyFor
            conclusion current processed next invariant compatible
          have operation : collapseWithOccurrences partition current
              next.prepared = (nextState, []) := by
            exact step.1
          have nextInvariant : AccumulatorInvariant partition dependencyFor
              conclusion nextState (next :: processed) := step.2
          obtain ⟨final, finalProcessed, tailTrace, finalInvariant,
              finalProcessedEq⟩ :=
            ih nextState (next :: processed) nextInvariant tail
          have boundary : Certificate.Flow.ListSetEq
              (observeEliminationBoundary dependencyFor current ++
                observeEliminationBoundary dependencyFor next.prepared)
              (observeEliminationBoundary dependencyFor
                (collapseWithOccurrences partition current next.prepared).1) := by
            rw [invariant.exactShape] at operation ⊢
            rw [operation]
            exact observeEliminationBoundary_step_setEq partition dependencyFor
              current.CENTER processed next compatible
          refine ⟨final, finalProcessed, ?_, finalInvariant, ?_⟩
          · apply DLDS.Compression.Sequences.ActualCompressionSetTrace.step current next.prepared
              (rest.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared) boundary
            simpa [operation] using tailTrace
          · simpa [List.reverse_cons, List.append_assoc] using finalProcessedEq

/-- Exact endpoint transport applied by the inner recursion to an already
carried occurrence accumulator. -/
def rewriteAccumulatorAlongEliminations :
    Vertex -> List ElimOccurrence -> List Ancestral -> List Ancestral
  | _, [], occurrences => occurrences
  | currentCenter, next :: rest, occurrences =>
      let nextCenter := collapse.center currentCenter next.original
      let rewritten := occurrences.map
        (collapse.rewrite_path_endpoints [currentCenter, next.original]
          nextCenter)
      rewriteAccumulatorAlongEliminations nextCenter rest rewritten

/-- The same arbitrary-length trace with a nonempty carried occurrence
accumulator.  Every endpoint rewrite is definitionally the one performed by
`compressWithOccurrences`; pure elimination steps add no fabricated paths. -/
theorem compatibleEliminationTail_actualTrace_withOccurrences
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (current : Neighborhood)
    (processed remaining : List ElimOccurrence)
    (occurrences : List Ancestral)
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (schedule : CompatibleEliminationTail partition dependencyFor
      current.CENTER processed remaining) :
    Exists fun final : Neighborhood => Exists fun finalProcessed : List ElimOccurrence =>
      ActualCompressionSetTrace partition
          (observeEliminationBoundary dependencyFor)
          (current :: remaining.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared) occurrences final
          (rewriteAccumulatorAlongEliminations current.CENTER remaining
            occurrences) /\
        AccumulatorInvariant partition dependencyFor conclusion
          final finalProcessed /\
        finalProcessed = remaining.reverse ++ processed := by
  induction remaining generalizing current processed occurrences with
  | nil =>
      cases schedule with
      | nil =>
          exact ⟨current, processed,
            DLDS.Compression.Sequences.ActualCompressionSetTrace.single current, invariant, by simp⟩
  | cons next rest ih =>
      cases schedule with
      | cons _ _ _ _ compatible tail =>
          let nextState := accumulatedNeighborhood
            (collapse.center current.CENTER next.original)
            (next :: processed)
          let rewritten := occurrences.map
            (collapse.rewrite_path_endpoints
              [current.CENTER, next.original] nextState.CENTER)
          have step := accumulatorInvariant_step partition dependencyFor
            conclusion current processed next invariant compatible
          have operation : collapseWithOccurrences partition current
              next.prepared = (nextState, []) := step.1
          have nextInvariant : AccumulatorInvariant partition dependencyFor
              conclusion nextState (next :: processed) := step.2
          obtain ⟨final, finalProcessed, tailTrace, finalInvariant,
              finalProcessedEq⟩ :=
            ih nextState (next :: processed) rewritten nextInvariant tail
          have boundary : Certificate.Flow.ListSetEq
              (observeEliminationBoundary dependencyFor current ++
                observeEliminationBoundary dependencyFor next.prepared)
              (observeEliminationBoundary dependencyFor
                (collapseWithOccurrences partition current next.prepared).1) := by
            rw [invariant.exactShape] at operation ⊢
            rw [operation]
            exact observeEliminationBoundary_step_setEq partition dependencyFor
              current.CENTER processed next compatible
          refine ⟨final, finalProcessed, ?_, finalInvariant, ?_⟩
          · apply DLDS.Compression.Sequences.ActualCompressionSetTrace.step current next.prepared
              (rest.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared) boundary
            rw [operation]
            simpa [rewritten, nextState, rewriteAccumulatorAlongEliminations]
              using tailTrace
          · simpa [List.reverse_cons, List.append_assoc] using finalProcessedEq

/-- Arbitrary nonempty homogeneous elimination classes therefore have a
constructed production trace, starting from the first actual prepared
member. -/
theorem eliminationClass_actualTrace
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (first : ElimOccurrence)
    (rest : List ElimOccurrence)
    (initial : InitialEliminationMember partition dependencyFor
      conclusion first)
    (schedule : CompatibleEliminationTail partition dependencyFor
      first.original [first] rest) :
    Exists fun final : Neighborhood => Exists fun finalProcessed : List ElimOccurrence =>
      ActualCompressionSetTrace partition
          (observeEliminationBoundary dependencyFor)
          ((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared) [] final [] /\
        AccumulatorInvariant partition dependencyFor conclusion
          final finalProcessed /\
        finalProcessed = (first :: rest).reverse := by
  have initialInvariant := initialEliminationMember_invariant partition
    dependencyFor conclusion first initial
  obtain ⟨final, finalProcessed, trace, finalInvariant,
      finalProcessedEq⟩ :=
    compatibleEliminationTail_actualTrace partition dependencyFor conclusion
      first.prepared [first] rest initialInvariant schedule
  exact ⟨final, finalProcessed, by simpa using trace, finalInvariant, by
    simpa [List.reverse_cons, List.append_assoc] using finalProcessedEq⟩

/-- Executable and semantic consequences of the constructed finite trace.
The occurrence accumulator remains empty because pure derived eliminations
mint no direct occurrences. -/
theorem eliminationClass_actualExecution
    (partition : List HCClass) (dependencyFor : Nat -> Dep)
    (conclusion : Formula) (first : ElimOccurrence)
    (rest : List ElimOccurrence)
    (initial : InitialEliminationMember partition dependencyFor
      conclusion first)
    (schedule : CompatibleEliminationTail partition dependencyFor
      first.original [first] rest) :
    Exists fun final : Neighborhood => Exists fun finalProcessed : List ElimOccurrence =>
      compressWithOccurrences partition
          ((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared) [] = (final, []) /\
        Certificate.Flow.ListSetEq
          (((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared).flatMap
            (observeEliminationBoundary dependencyFor))
          (observeEliminationBoundary dependencyFor final) /\
        AccumulatorInvariant partition dependencyFor conclusion
          final finalProcessed /\
        finalProcessed = (first :: rest).reverse := by
  obtain ⟨final, finalProcessed, trace, invariant, processedEq⟩ :=
    eliminationClass_actualTrace partition dependencyFor conclusion first rest
      initial schedule
  exact ⟨final, finalProcessed, trace.refines_actual_compressor,
    trace.boundary_preserved, invariant, processedEq⟩

/-- Facts read from an actual graph which characterize an ordinary,
uncollapsed, derived implication-elimination member before `pre_collapse`.
The original graph's unrelated ancestral records are intentionally absent:
the producer constructs this member's indirect records from incidence. -/
structure ActualGraphEliminationMember (d : Graph)
    (occurrence : ElimOccurrence) : Prop where
  nodePresent : List.Mem occurrence.original d.NODES
  uncollapsed : occurrence.original.COLLAPSED = false
  derived : occurrence.original.HYPOTHESIS = false
  incoming : get_rule.incoming occurrence.original d =
    occurrence.incoming occurrence.original
  outgoing : get_rule.outgoing occurrence.original d =
    [edge occurrence.original occurrence.use 0 occurrence.useStored]
  direct : get_rule.direct occurrence.original d = []

/-- The production frontend prepares an actual graph member into exactly the
occurrence neighbourhood used by the class theorem. -/
theorem ActualGraphEliminationMember.pre_collapse_eq_prepared
    {d : Graph} {occurrence : ElimOccurrence}
    (actual : ActualGraphEliminationMember d occurrence) :
    pre_collapse (get_rule occurrence.original d) = occurrence.prepared := by
  unfold get_rule
  simp only [pre_collapse, actual.uncollapsed, actual.derived,
    actual.incoming, actual.outgoing, actual.direct]
  simp [pre_collapse.outgoing, pre_collapse.direct,
    pre_collapse.indirect, pre_collapse.indirect.create,
    DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared,
    DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.incoming, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing,
    DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.indirect, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.tag]

/-- Componentwise graph membership establishes the exact list consumed by
the real class compressor (`get_rules` followed by `pre_compress`). -/
theorem pre_compress_get_rules_elimination_members
    (d : Graph) (members : List ElimOccurrence)
    (actual : forall occurrence, List.Mem occurrence members ->
      ActualGraphEliminationMember d occurrence) :
    pre_compress (get_rules (members.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original) d) =
      members.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.prepared := by
  induction members with
  | nil => rfl
  | cons occurrence rest ih =>
      have headActual := actual occurrence (List.Mem.head rest)
      have tailActual : forall candidate, List.Mem candidate rest ->
          ActualGraphEliminationMember d candidate := by
        intro candidate member
        exact actual candidate (List.Mem.tail occurrence member)
      simp only [List.map_cons, get_rules, pre_compress]
      rw [headActual.pre_collapse_eq_prepared, ih tailActual]

/-- A class selected by the production outer schedule, together with the
componentwise graph facts which make each member an ordinary elimination. -/
structure ActualScheduledEliminationClass (d : Graph)
    (members : List ElimOccurrence) : Prop where
  scheduled : (repeated_nodes_vertex d).flatten.contains
    (members.map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original) = true
  graphMembers : forall occurrence, List.Mem occurrence members ->
    ActualGraphEliminationMember d occurrence

/-- The constructed trace is a trace for the actual `get_rules` and
`pre_compress` input selected from the graph, not merely for a hand-written
list of neighbourhoods. -/
theorem actualGraphEliminationClass_trace
    (d : Graph) (dependencyFor : Nat -> Dep) (conclusion : Formula)
    (first : ElimOccurrence) (rest : List ElimOccurrence)
    (selected : ActualScheduledEliminationClass d (first :: rest))
    (initial : InitialEliminationMember (hcPartition d) dependencyFor
      conclusion first)
    (schedule : CompatibleEliminationTail (hcPartition d) dependencyFor
      first.original [first] rest) :
    Exists fun final : Neighborhood => Exists fun finalProcessed : List ElimOccurrence =>
      ActualCompressionSetTrace (hcPartition d)
          (observeEliminationBoundary dependencyFor)
          (pre_compress
            (get_rules ((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original) d))
          [] final [] /\
        AccumulatorInvariant (hcPartition d) dependencyFor conclusion
          final finalProcessed /\
        finalProcessed = (first :: rest).reverse := by
  obtain ⟨final, finalProcessed, trace, invariant, processedEq⟩ :=
    eliminationClass_actualTrace (hcPartition d) dependencyFor conclusion
      first rest initial schedule
  have prepared := pre_compress_get_rules_elimination_members d
    (first :: rest) selected.graphMembers
  refine ⟨final, finalProcessed, ?_, invariant, processedEq⟩
  rw [prepared]
  exact trace

/-- Exact class-level execution plus independently observed boundary
preservation for a production-selected graph class. -/
theorem actualGraphEliminationClass_execution
    (d : Graph) (dependencyFor : Nat -> Dep) (conclusion : Formula)
    (first : ElimOccurrence) (rest : List ElimOccurrence)
    (selected : ActualScheduledEliminationClass d (first :: rest))
    (initial : InitialEliminationMember (hcPartition d) dependencyFor
      conclusion first)
    (schedule : CompatibleEliminationTail (hcPartition d) dependencyFor
      first.original [first] rest) :
    Exists fun final : Neighborhood => Exists fun finalProcessed : List ElimOccurrence =>
      compress_ruleWithOccurrences (hcPartition d)
          ((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original) d = (final, []) /\
        Certificate.Flow.ListSetEq
          ((pre_compress
            (get_rules ((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original) d)).flatMap
              (observeEliminationBoundary dependencyFor))
          (observeEliminationBoundary dependencyFor final) /\
        AccumulatorInvariant (hcPartition d) dependencyFor conclusion
          final finalProcessed /\
        finalProcessed = (first :: rest).reverse := by
  obtain ⟨final, finalProcessed, trace, invariant, processedEq⟩ :=
    actualGraphEliminationClass_trace d dependencyFor conclusion first rest
      selected initial schedule
  exact ⟨final, finalProcessed,
    trace.refines_compress_ruleWithOccurrences,
    trace.boundary_preserved, invariant, processedEq⟩

/-- Exact graph replacement and external occurrence-reference transport for
the covered class.  This is the production `pop_rules`/`set_rule` boundary;
it deliberately does not claim preservation for later outer-loop classes. -/
theorem actualGraphEliminationClass_graphReplacement
    (d : Graph) (dependencyFor : Nat -> Dep) (conclusion : Formula)
    (first : ElimOccurrence) (rest : List ElimOccurrence)
    (selected : ActualScheduledEliminationClass d (first :: rest))
    (initial : InitialEliminationMember (hcPartition d) dependencyFor
      conclusion first)
    (schedule : CompatibleEliminationTail (hcPartition d) dependencyFor
      first.original [first] rest)
    (occurrences : List Ancestral) :
    Exists fun final : Neighborhood =>
      compress_nodesWithPartitionCarrying (hcPartition d)
          ((first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original) d occurrences =
        let nodes := (first :: rest).map DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.original
        let materialized := materializeCarriedOwners nodes d occurrences
        let rewritten := materialized.map
          (collapse.rewrite_path_endpoints nodes final.CENTER)
        let graph := collapse.rewrite_graph_path_endpoints nodes final.CENTER
          (set_rule (pop_rules d (get_rules nodes d)) final)
        let graph := dlds graph.NODES graph.EDGES
          (ensureOccurrenceRecords graph.PATHS rewritten)
        (checkAncestralPathHeads graph, rewritten) := by
  obtain ⟨final, finalProcessed, trace, invariant, processedEq⟩ :=
    actualGraphEliminationClass_trace d dependencyFor conclusion first rest
      selected initial schedule
  refine ⟨final, ?_⟩
  simpa using trace.refines_compress_nodesWithPartitionCarrying

@[simp] theorem observeAcross_matching_occurrence
    (conclusion : Formula) (center : Vertex)
    (occurrence : ElimOccurrence) :
    observeAcross conclusion occurrence.route
        (occurrence.outgoing center) =
      some (occurrence.observation conclusion) := by
  simp [observeAcross, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.route, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing,
    DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.observation, consumeCandidate, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.tag]

@[simp] theorem observeAcross_distinct_occurrences
    (conclusion : Formula) (center : Vertex)
    (routeOccurrence edgeOccurrence : ElimOccurrence)
    (distinct : Not (routeOccurrence.tag = edgeOccurrence.tag)) :
    observeAcross conclusion routeOccurrence.route
        (edgeOccurrence.outgoing center) = none := by
  have reverseDistinct : Not (edgeOccurrence.tag = routeOccurrence.tag) :=
    Ne.symm distinct
  simp [observeAcross, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.route, DLDS.Compression.Elimination.ClassPreservation.ElimOccurrence.outgoing,
    consumeCandidate, reverseDistinct]

private theorem oneRoute_crossEdges_empty
    (conclusion : Formula) (center : Vertex)
    (routeOccurrence : ElimOccurrence)
    (edgeOccurrences : List ElimOccurrence)
    (distinct : forall edgeOccurrence,
      List.Mem edgeOccurrence edgeOccurrences ->
      Not (routeOccurrence.tag = edgeOccurrence.tag)) :
    (occurrenceOutgoing center edgeOccurrences).filterMap
        (observeAcross conclusion routeOccurrence.route) = [] := by
  induction edgeOccurrences with
  | nil => simp [occurrenceOutgoing]
  | cons edgeOccurrence rest ih =>
      have headDistinct := distinct edgeOccurrence (List.Mem.head rest)
      have tailDistinct : forall candidate, List.Mem candidate rest ->
          Not (routeOccurrence.tag = candidate.tag) := by
        intro candidate member
        exact distinct candidate (List.Mem.tail edgeOccurrence member)
      simp only [occurrenceOutgoing, List.map_cons, List.filterMap_cons]
      rw [observeAcross_distinct_occurrences conclusion center
        routeOccurrence edgeOccurrence headDistinct]
      exact ih tailDistinct

private theorem crossRoutes_oneEdge_empty
    (conclusion : Formula) (center : Vertex)
    (routeOccurrences : List ElimOccurrence)
    (edgeOccurrence : ElimOccurrence)
    (distinct : forall routeOccurrence,
      List.Mem routeOccurrence routeOccurrences ->
      Not (routeOccurrence.tag = edgeOccurrence.tag)) :
    ((occurrenceRoutes routeOccurrences).flatMap fun route =>
        [edgeOccurrence.outgoing center].filterMap
          (observeAcross conclusion route)) = [] := by
  induction routeOccurrences with
  | nil => simp [occurrenceRoutes]
  | cons routeOccurrence rest ih =>
      have headDistinct := distinct routeOccurrence (List.Mem.head rest)
      have tailDistinct : forall candidate, List.Mem candidate rest ->
          Not (candidate.tag = edgeOccurrence.tag) := by
        intro candidate member
        exact distinct candidate (List.Mem.tail routeOccurrence member)
      simp only [occurrenceRoutes, List.map_cons, List.flatMap_cons]
      rw [show [edgeOccurrence.outgoing center].filterMap
          (observeAcross conclusion routeOccurrence.route) = [] by
        simp [observeAcross_distinct_occurrences conclusion center
          routeOccurrence edgeOccurrence headDistinct]]
      simp only [List.nil_append]
      exact ih tailDistinct

private theorem crossRoutes_consEdge
    (conclusion : Formula) (center : Vertex)
    (routeOccurrences : List ElimOccurrence)
    (edgeOccurrence : ElimOccurrence)
    (remainingEdges : List Deduction)
    (distinct : forall routeOccurrence,
      List.Mem routeOccurrence routeOccurrences ->
      Not (routeOccurrence.tag = edgeOccurrence.tag)) :
    ((occurrenceRoutes routeOccurrences).flatMap fun route =>
        (edgeOccurrence.outgoing center :: remainingEdges).filterMap
          (observeAcross conclusion route)) =
      ((occurrenceRoutes routeOccurrences).flatMap fun route =>
        remainingEdges.filterMap (observeAcross conclusion route)) := by
  induction routeOccurrences with
  | nil => simp [occurrenceRoutes]
  | cons routeOccurrence rest ih =>
      have headDistinct := distinct routeOccurrence (List.Mem.head rest)
      have tailDistinct : forall candidate, List.Mem candidate rest ->
          Not (candidate.tag = edgeOccurrence.tag) := by
        intro candidate member
        exact distinct candidate (List.Mem.tail routeOccurrence member)
      simp only [occurrenceRoutes, List.map_cons, List.flatMap_cons,
        List.filterMap_cons]
      rw [observeAcross_distinct_occurrences conclusion center
        routeOccurrence edgeOccurrence headDistinct]
      simp only
      congr 1
      exact ih tailDistinct

/-- The route-sensitive boundary observation is compositional for an
arbitrary finite family. Every independently recomputed route crosses its
matching occurrence-coloured use, all cross combinations are rejected, and
the exact occurrence order and multiplicity are retained. -/
theorem boundaryDeliveries_occurrence_family
    (conclusion : Formula) (center : Vertex)
    (occurrences : List ElimOccurrence)
    (distinct : PairwiseDistinctTags occurrences) :
    boundaryDeliveries conclusion (occurrenceRoutes occurrences)
        (occurrenceOutgoing center occurrences) =
      occurrenceObservations conclusion occurrences := by
  induction occurrences with
  | nil => simp [boundaryDeliveries, occurrenceRoutes,
      occurrenceOutgoing, occurrenceObservations]
  | cons occurrence rest ih =>
      have headDistinct : forall candidate, List.Mem candidate rest ->
          Not (occurrence.tag = candidate.tag) := by
        intro candidate member
        exact (List.pairwise_cons.mp distinct).1 candidate member
      have reverseDistinct : forall candidate, List.Mem candidate rest ->
          Not (candidate.tag = occurrence.tag) := by
        intro candidate member
        exact Ne.symm (headDistinct candidate member)
      have tailDistinct : PairwiseDistinctTags rest :=
        (List.pairwise_cons.mp distinct).2
      rw [show occurrenceRoutes (occurrence :: rest) =
          occurrence.route :: occurrenceRoutes rest by rfl]
      rw [show occurrenceOutgoing center (occurrence :: rest) =
          occurrence.outgoing center :: occurrenceOutgoing center rest by rfl]
      unfold boundaryDeliveries
      rw [List.flatMap_cons]
      rw [List.filterMap_cons]
      rw [observeAcross_matching_occurrence]
      simp only
      rw [oneRoute_crossEdges_empty conclusion center occurrence rest
        headDistinct]
      rw [crossRoutes_consEdge conclusion center rest occurrence
        (occurrenceOutgoing center rest) reverseDistinct]
      change [occurrence.observation conclusion] ++
          boundaryDeliveries conclusion (occurrenceRoutes rest)
            (occurrenceOutgoing center rest) = _
      rw [ih tailDistinct]
      rfl

/-- A matching major/minor occurrence with colour-zero incoming edges emits
the independently recomputed union and retains exactly its occurrence tag as
the continuation. -/
theorem matchingEliminationPairOutputs_zero_edges
    (d : Graph) (v major minor : Vertex)
    (premiseFlow : Vertex -> List FlowPair)
    (majorDep minorDep : Dep) (tag : Nat)
    (majorEdge minorEdge : Deduction)
    (majorFlow : premiseFlow major = [(majorDep, [0, tag])])
    (minorFlow : premiseFlow minor = [(minorDep, [0, tag])])
    (majorEdges : edgesBetween d major v = [majorEdge])
    (minorEdges : edgesBetween d minor v = [minorEdge])
    (majorZero : majorEdge.COLOUR = 0)
    (minorZero : minorEdge.COLOUR = 0) :
    DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs d v premiseFlow
        (major, minor) =
      [(depUnion minorDep majorDep, [tag])] := by
  simp [DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs,
    majorFlow, minorFlow, majorEdges, minorEdges, FlowSemantics.consume,
    majorZero, minorZero]

/-- For any two distinct members of an arbitrary processed family, the
logical Cartesian elimination kernel rejects their cross occurrence when
their premise routes retain the respective tags.  This is the family-level
reuse of the EE pair theorem; formulae may be identical. -/
theorem family_crossEliminationPairOutputs_empty
    (occurrences : List ElimOccurrence)
    (distinct : PairwiseDistinctTags occurrences)
    (left right : ElimOccurrence)
    (leftMember : List.Mem left occurrences)
    (rightMember : List.Mem right occurrences)
    (different : Not (left = right))
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (majorEdge minorEdge : Deduction)
    (majorFlow : premiseFlow left.major =
      [(left.majorDep, [0, left.tag])])
    (minorFlow : premiseFlow right.minor =
      [(right.minorDep, [0, right.tag])])
    (majorEdges : edgesBetween d left.major v = [majorEdge])
    (minorEdges : edgesBetween d right.minor v = [minorEdge])
    (majorZero : majorEdge.COLOUR = 0)
    (minorZero : minorEdge.COLOUR = 0) :
    DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs d v premiseFlow
        (left.major, right.minor) = [] := by
  have injective := pairwiseDistinctTags_injective occurrences distinct
  have tagDifferent : Not (right.tag = left.tag) := by
    intro tagEqual
    have same := injective right rightMember left leftMember tagEqual
    exact different same.symm
  exact
    DLDS.Compression.Elimination.PairPreservation.eliminationPairOutputs_zero_edges_distinct_tags
      d v left.major right.minor premiseFlow left.majorDep right.minorDep
      left.tag right.tag majorEdge minorEdge majorFlow minorFlow majorEdges
      minorEdges majorZero minorZero tagDifferent

/-- On an accumulated state the uniform edge observer agrees exactly with
the explicit route-sensitive Cartesian boundary evaluator. -/
theorem routeBoundary_eq_observeEliminationBoundary
    (dependencyFor : Nat -> Dep) (center : Vertex)
    (occurrences : List ElimOccurrence)
    (distinct : PairwiseDistinctTags occurrences)
    (correct : DependencyTableCorrect dependencyFor occurrences) :
    boundaryDeliveries center.FORMULA (occurrenceRoutes occurrences)
        (accumulatedNeighborhood center occurrences).OUTGOING =
      observeEliminationBoundary dependencyFor
        (accumulatedNeighborhood center occurrences) := by
  rw [show (accumulatedNeighborhood center occurrences).OUTGOING =
      occurrenceOutgoing center occurrences by rfl]
  rw [boundaryDeliveries_occurrence_family center.FORMULA center
    occurrences distinct]
  exact (observeEliminationBoundary_accumulated dependencyFor center
    occurrences correct).symm

/-- The invariant's operational boundary is the independently recomputed
route boundary.  This uses the exact current producer state, not stored
dependency labels. -/
theorem AccumulatorInvariant.operationalBoundary
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed) :
    boundaryDeliveries conclusion (occurrenceRoutes processed)
        current.OUTGOING =
      observeEliminationBoundary dependencyFor current := by
  rw [invariant.exactShape]
  rw [← invariant.centerFormula]
  simpa [accumulatedNeighborhood] using
    routeBoundary_eq_observeEliminationBoundary dependencyFor
      current.CENTER processed invariant.distinctTags invariant.dependencyCorrect

/-- Membership equality of class observations transports to membership
equality of any predicate or consumer which depends only on a complete
`BoundaryObservation`. This is the precise congruence supplied by the set
trace; it does not identify the underlying operational neighbourhoods. -/
theorem ListSetEq_observation_any_congr
    {left right : List BoundaryObservation}
    (equal : Certificate.Flow.ListSetEq left right)
    (predicate : BoundaryObservation -> Prop) :
    (Exists fun observation =>
        List.Mem observation left /\ predicate observation) <->
      Exists fun observation =>
        List.Mem observation right /\ predicate observation := by
  constructor
  case mp =>
    intro existsLeft
    cases existsLeft with
    | intro observation conjunction =>
        cases conjunction with
        | intro member property =>
            exact Exists.intro observation
              (And.intro ((equal observation).mp member) property)
  case mpr =>
    intro existsRight
    cases existsRight with
    | intro observation conjunction =>
        cases conjunction with
        | intro member property =>
            exact Exists.intro observation
              (And.intro ((equal observation).mpr member) property)

end DLDS.Compression.Elimination.ClassPreservation
