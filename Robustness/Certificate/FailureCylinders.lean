import Certificate.Encoding.FixedShape

/-!
# One-sided failure-cylinder robustness

This file separates a support which determines a Boolean oracle in both
directions from the weaker object needed by a robustness argument: a partial
assignment which forces one anchored failure.  It also records the dependent
(coordinate-specific) alphabet version of the counting argument.
-/

open DLDS FlowSemantics Certificate.Flow.Foundation

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace CertificateSearch
namespace Robustness

deriving instance DecidableEq for LocalRuleShape

/-! ## Homogeneous one-sided witnesses -/

/-- A partial assignment which forces the anchored Boolean failure.  Unlike
`MalformationWitness.localized`, this has no converse and need not determine
the oracle on successful assignments. -/
structure FailureWitness {Coordinate Alphabet : Type}
    [DecidableEq Coordinate]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet) where
  support : Finset Coordinate
  anchorFailure : oracle anchor = true
  forcesFailure : ∀ other, AgreeOn support anchor other → oracle other = true

def failureWitnessSize {Coordinate Alphabet : Type}
    [DecidableEq Coordinate]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) : Nat :=
  witness.support.card

/-- The cylinder of all full assignments extending the witness's anchored
partial assignment. -/
abbrev FailureCylinder {Coordinate Alphabet : Type}
    [DecidableEq Coordinate]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :=
  { other : Assignment Coordinate Alphabet //
    AgreeOn witness.support anchor other }

abbrev OracleFailureAssignments {Coordinate Alphabet : Type}
    (oracle : Assignment Coordinate Alphabet → Bool) :=
  { input : Assignment Coordinate Alphabet // oracle input = true }

def failureCylinderEquiv {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :
    FailureCylinder witness ≃ (↥witness.supportᶜ → Alphabet) where
  toFun other coordinate := other.1 coordinate.1
  invFun free :=
    ⟨fun coordinate =>
        if member : coordinate ∈ witness.support then
          anchor coordinate
        else
          free ⟨coordinate, by simpa using member⟩,
      by
        intro coordinate member
        simp [member]⟩
  left_inv other := by
    apply Subtype.ext
    funext coordinate
    by_cases member : coordinate ∈ witness.support
    · simp [member, other.2 coordinate member]
    · simp [member]
  right_inv free := by
    funext coordinate
    have outside : coordinate.1 ∉ witness.support :=
      Finset.mem_compl.mp coordinate.2
    simp [outside]

noncomputable instance failureCylinderFintype
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [Fintype Alphabet] [DecidableEq Coordinate]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    {witness : FailureWitness oracle anchor} :
    Fintype (FailureCylinder witness) :=
  Fintype.ofEquiv (↥witness.supportᶜ → Alphabet)
    (failureCylinderEquiv witness).symm

noncomputable instance oracleFailureAssignmentsFintype
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [Fintype Alphabet]
    {oracle : Assignment Coordinate Alphabet → Bool} :
    Fintype (OracleFailureAssignments oracle) :=
  Fintype.ofFinite _

/-- Exact cardinality of a one-sided failure cylinder. -/
theorem failureCylinder_card {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :
    Fintype.card (FailureCylinder witness) =
      Fintype.card Alphabet ^
        (Fintype.card Coordinate - witness.support.card) := by
  calc
    Fintype.card (FailureCylinder witness) =
        Fintype.card (↥witness.supportᶜ → Alphabet) :=
      Fintype.card_congr (failureCylinderEquiv witness)
    _ = Fintype.card Alphabet ^ Fintype.card ↥witness.supportᶜ :=
      Fintype.card_fun
    _ = Fintype.card Alphabet ^ witness.supportᶜ.card := by
      rw [Fintype.card_coe]
    _ = Fintype.card Alphabet ^
        (Fintype.card Coordinate - witness.support.card) := by
      congr 1
      exact Finset.card_compl witness.support

def failureCylinderToAssignments {Coordinate Alphabet : Type}
    [DecidableEq Coordinate]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :
    FailureCylinder witness → OracleFailureAssignments oracle :=
  fun other => ⟨other.1, witness.forcesFailure other.1 other.2⟩

theorem failureCylinderToAssignments_injective
    {Coordinate Alphabet : Type} [DecidableEq Coordinate]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :
    Function.Injective (failureCylinderToAssignments witness) := by
  intro left right equality
  apply Subtype.ext
  exact congrArg
    (fun bad : OracleFailureAssignments oracle => bad.1) equality

/-- Every one-sided cylinder injects into the assignments exposing the same
failure.  Only `forcesFailure`, not two-sided oracle invariance, is used. -/
theorem failureAssignments_lower_bound {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :
    Fintype.card Alphabet ^
        (Fintype.card Coordinate - witness.support.card) ≤
      Fintype.card (OracleFailureAssignments oracle) := by
  rw [← failureCylinder_card witness]
  exact Fintype.card_le_of_injective
    (failureCylinderToAssignments witness)
    (failureCylinderToAssignments_injective witness)

noncomputable def oracleFailureDensity {Coordinate Alphabet : Type}
    [Fintype Coordinate] [Fintype Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool) : ℚ :=
  Fintype.card (OracleFailureAssignments oracle) /
    Fintype.card Alphabet ^ Fintype.card Coordinate

theorem failureDensity_lower_bound {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet] [Nonempty Alphabet]
    {oracle : Assignment Coordinate Alphabet → Bool}
    {anchor : Assignment Coordinate Alphabet}
    (witness : FailureWitness oracle anchor) :
    (1 : ℚ) / Fintype.card Alphabet ^ witness.support.card ≤
      oracleFailureDensity oracle := by
  have alphabetPositive : 0 < Fintype.card Alphabet := Fintype.card_pos
  have supportLe : witness.support.card ≤ Fintype.card Coordinate :=
    Finset.card_le_univ _
  have lower := failureAssignments_lower_bound witness
  have cross :
      Fintype.card Alphabet ^ Fintype.card Coordinate ≤
        Fintype.card Alphabet ^ witness.support.card *
          Fintype.card (OracleFailureAssignments oracle) := by
    calc
      Fintype.card Alphabet ^ Fintype.card Coordinate =
          Fintype.card Alphabet ^ witness.support.card *
            Fintype.card Alphabet ^
              (Fintype.card Coordinate - witness.support.card) := by
        rw [← pow_add, Nat.add_sub_of_le supportLe]
      _ ≤ _ := Nat.mul_le_mul_left _ lower
  unfold oracleFailureDensity
  apply (div_le_div_iff₀ (by positivity :
      (0 : ℚ) < Fintype.card Alphabet ^ witness.support.card)
    (by positivity :
      (0 : ℚ) < Fintype.card Alphabet ^ Fintype.card Coordinate)).2
  norm_num
  exact_mod_cast (by simpa [Nat.mul_comm] using cross)

/-- Forget the unnecessary two-sided part of the repository's earlier
`MalformationWitness`. -/
def FailureWitness.ofMalformationWitness
    {Coordinate Alphabet : Type} [DecidableEq Coordinate]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    FailureWitness (localFailureB d decode witness.position) anchor where
  support := witness.support
  anchorFailure := witness.failure
  forcesFailure := by
    intro other agree
    rw [witness.localized other agree]
    exact witness.failure

/-! ## Automatic minimum one-sided support -/

def ForcesFailure {Coordinate Alphabet : Type} [DecidableEq Coordinate]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (support : Finset Coordinate) : Prop :=
  ∀ other, AgreeOn support anchor other → oracle other = true

theorem forcesFailure_univ {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true) :
    ForcesFailure oracle anchor Finset.univ := by
  intro other agree
  have equality : other = anchor := by
    funext coordinate
    exact agree coordinate (Finset.mem_univ coordinate)
  simpa [equality] using failure

private theorem failureSupportSize_exists {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true) :
    ∃ size, ∃ support : Finset Coordinate,
      support.card = size ∧ ForcesFailure oracle anchor support :=
  ⟨Fintype.card Coordinate, Finset.univ,
    Finset.card_univ, forcesFailure_univ oracle anchor failure⟩

/-- Least number of homogeneous coordinates which force this particular
anchored failure.  This is one-sided and can be strictly smaller than
`supportOfOracle` or any complete syntactic read set. -/
noncomputable def minimumFailureWitnessSize {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true) : Nat := by
  classical
  exact Nat.find (failureSupportSize_exists oracle anchor failure)

noncomputable def minimumFailureSupport {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true) : Finset Coordinate := by
  classical
  exact Classical.choose
    (Nat.find_spec (failureSupportSize_exists oracle anchor failure))

theorem minimumFailureSupport_spec {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true) :
    (minimumFailureSupport oracle anchor failure).card =
        minimumFailureWitnessSize oracle anchor failure ∧
      ForcesFailure oracle anchor
        (minimumFailureSupport oracle anchor failure) := by
  classical
  exact Classical.choose_spec
    (Nat.find_spec (failureSupportSize_exists oracle anchor failure))

noncomputable def minimumFailureWitness {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true) : FailureWitness oracle anchor where
  support := minimumFailureSupport oracle anchor failure
  anchorFailure := failure
  forcesFailure := (minimumFailureSupport_spec oracle anchor failure).2

theorem minimumFailureWitness_minimal {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor : Assignment Coordinate Alphabet)
    (failure : oracle anchor = true)
    (support : Finset Coordinate)
    (forces : ForcesFailure oracle anchor support) :
    failureWitnessSize (minimumFailureWitness oracle anchor failure) ≤
      support.card := by
  classical
  unfold failureWitnessSize minimumFailureWitness
  rw [(minimumFailureSupport_spec oracle anchor failure).1]
  exact Nat.find_min' (failureSupportSize_exists oracle anchor failure)
    ⟨support, rfl, forces⟩

/-! ## Shape-fixed semantic instantiation -/

abbrev ShapeFixedFailureWitness (d : Graph) (c : Certificate)
    (position : VerificationPosition) :=
  FailureWitness (shapeFixedSemanticFailureB d c position)
    (encodeShapeFixed d c)

abbrev ShapeFixedFailureCylinder (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (witness : ShapeFixedFailureWitness d c position) :=
  FailureCylinder witness

def shapeFixedFailureWitnessFromReadSet (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    ShapeFixedFailureWitness d c position where
  support := shapeFixedSemanticReadSet d c position
  anchorFailure := by
    unfold shapeFixedSemanticFailureB
    rw [decodeShapeFixed_encodeShapeFixed]
    unfold SemanticMarkedAt at failure
    simp [failure]
  forcesFailure := by
    intro other agree
    unfold shapeFixedSemanticFailureB
    have invariant := shapeFixedSemanticReadSet_agreement d c position
      (encodeShapeFixed d c) other agree
    rw [← invariant, decodeShapeFixed_encodeShapeFixed]
    unfold SemanticMarkedAt at failure
    simp [failure]

noncomputable def minimumShapeFixedFailureWitness (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    ShapeFixedFailureWitness d c position :=
  minimumFailureWitness (shapeFixedSemanticFailureB d c position)
    (encodeShapeFixed d c) (by
      unfold shapeFixedSemanticFailureB
      rw [decodeShapeFixed_encodeShapeFixed]
      unfold SemanticMarkedAt at failure
      simp [failure])

theorem minimumShapeFixedFailureWitness_size_le (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position)
    (witness : ShapeFixedFailureWitness d c position) :
    failureWitnessSize
        (minimumShapeFixedFailureWitness d c position failure) ≤
      failureWitnessSize witness := by
  unfold minimumShapeFixedFailureWitness
  exact minimumFailureWitness_minimal
    (shapeFixedSemanticFailureB d c position) (encodeShapeFixed d c)
    _ witness.support witness.forcesFailure

/-- Every actual failed semantic check automatically has a minimum one-sided
failure witness.  The bound is by the two-sided read set, but the chosen
support is the least forcing support, not that read set. -/
theorem semanticMarked_has_minimumFailureWitness (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    ∃ witness : ShapeFixedFailureWitness d c position,
      failureWitnessSize witness =
          minimumFailureWitnessSize
            (shapeFixedSemanticFailureB d c position)
            (encodeShapeFixed d c) witness.anchorFailure ∧
        failureWitnessSize witness ≤
          shapeFixedSemanticReadBound d c position := by
  let witness := minimumShapeFixedFailureWitness d c position failure
  refine ⟨witness, ?_, ?_⟩
  · exact (minimumFailureSupport_spec
      (shapeFixedSemanticFailureB d c position)
      (encodeShapeFixed d c) witness.anchorFailure).1
  · exact minimumFailureWitness_minimal
      (shapeFixedSemanticFailureB d c position)
      (encodeShapeFixed d c) witness.anchorFailure
      (shapeFixedSemanticReadSet d c position)
      (shapeFixedFailureWitnessFromReadSet d c position failure).forcesFailure

theorem shapeFixedFailureCylinder_card (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (witness : ShapeFixedFailureWitness d c position) :
    Fintype.card (ShapeFixedFailureCylinder d c position witness) =
      Fintype.card (ShapeFixedAlphabet d c) ^
        (Fintype.card (ShapeFixedCoord d c) -
          failureWitnessSize witness) := by
  simpa [failureWitnessSize] using failureCylinder_card witness

theorem shapeFixedFailureWitness_density_lower_bound (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (witness : ShapeFixedFailureWitness d c position) :
    (1 : ℚ) / Fintype.card (ShapeFixedAlphabet d c) ^
        failureWitnessSize witness ≤
      shapeFixedSemanticMalformationDensity d c position := by
  simpa only [oracleFailureDensity, shapeFixedSemanticMalformationDensity,
    failureWitnessSize] using failureDensity_lower_bound witness

/-- A failed semantic position determines an automatically chosen least
one-sided failure cylinder.  The read-set bound is retained only as an upper
bound on the least forcing support; the cylinder itself is defined by the
one-sided failure property. -/
theorem semanticMarked_has_badCylinder (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (failure : SemanticMarkedAt d c position) :
    ∃ witness : ShapeFixedFailureWitness d c position,
      failureWitnessSize witness =
          minimumFailureWitnessSize
            (shapeFixedSemanticFailureB d c position)
            (encodeShapeFixed d c) witness.anchorFailure ∧
      failureWitnessSize witness ≤
          shapeFixedSemanticReadBound d c position ∧
      Fintype.card (ShapeFixedFailureCylinder d c position witness) =
          Fintype.card (ShapeFixedAlphabet d c) ^
            (Fintype.card (ShapeFixedCoord d c) -
              failureWitnessSize witness) ∧
      (1 : ℚ) / Fintype.card (ShapeFixedAlphabet d c) ^
          failureWitnessSize witness ≤
            shapeFixedSemanticMalformationDensity d c position := by
  obtain ⟨witness, minimum, bounded⟩ :=
    semanticMarked_has_minimumFailureWitness d c position failure
  exact ⟨witness, minimum, bounded,
    shapeFixedFailureCylinder_card d c position witness,
    shapeFixedFailureWitness_density_lower_bound d c position witness⟩

/-- Failure-cylinder bound in the fixed-shape certificate space:
failure of complete certificate formation produces a genuine semantic
position and a least finite partial assignment whose whole cylinder fails
that same position. -/
theorem formation_invalid_has_badCylinder (d : Graph) (c : Certificate)
    (invalid : ¬ FormationValid d c) :
    ∃ position, position ∈ semanticPositions d c ∧
      ∃ witness : ShapeFixedFailureWitness d c position,
        failureWitnessSize witness =
            minimumFailureWitnessSize
              (shapeFixedSemanticFailureB d c position)
              (encodeShapeFixed d c) witness.anchorFailure ∧
        failureWitnessSize witness ≤
            shapeFixedSemanticReadBound d c position ∧
        Fintype.card (ShapeFixedFailureCylinder d c position witness) =
            Fintype.card (ShapeFixedAlphabet d c) ^
              (Fintype.card (ShapeFixedCoord d c) -
                failureWitnessSize witness) ∧
        (1 : ℚ) / Fintype.card (ShapeFixedAlphabet d c) ^
            failureWitnessSize witness ≤
              shapeFixedSemanticMalformationDensity d c position := by
  obtain ⟨position, member, failure⟩ :=
    formation_invalid_implies_semanticMarked d c invalid
  obtain ⟨witness, minimum, bounded, cardinality, density⟩ :=
    semanticMarked_has_badCylinder d c position failure
  exact ⟨position, member, witness, minimum, bounded, cardinality, density⟩

/-- In the fixed-shape space, every failing structural position has the empty
payload witness and hence its entire assignment space is a failure cylinder. -/
def structuralShapeFixedFailureWitness (d : Graph) (c : Certificate)
    (position : VerificationPosition)
    (structural : shapeFixedFailureClass position = .structural)
    (failure : SemanticMarkedAt d c position) :
    ShapeFixedFailureWitness d c position where
  support := ∅
  anchorFailure := by
    unfold shapeFixedSemanticFailureB
    rw [decodeShapeFixed_encodeShapeFixed]
    unfold SemanticMarkedAt at failure
    simp [failure]
  forcesFailure := by
    intro other _
    have readEmpty := shapeFixedStructural_readSet_eq_empty
      d c position structural
    have agree : AgreeOn (shapeFixedSemanticReadSet d c position)
        (encodeShapeFixed d c) other := by
      rw [readEmpty]
      simp [AgreeOn]
    exact (shapeFixedFailureWitnessFromReadSet
      d c position failure).forcesFailure other agree

@[simp] theorem structuralShapeFixedFailureWitness_size
    (d : Graph) (c : Certificate) (position : VerificationPosition)
    (structural : shapeFixedFailureClass position = .structural)
    (failure : SemanticMarkedAt d c position) :
    failureWitnessSize
      (structuralShapeFixedFailureWitness d c position structural failure) = 0 :=
  rfl

theorem structuralShapeFixedFailure_density_one (d : Graph)
    (c : Certificate) (position : VerificationPosition)
    (structural : shapeFixedFailureClass position = .structural)
    (failure : SemanticMarkedAt d c position) :
    (1 : ℚ) ≤ shapeFixedSemanticMalformationDensity d c position := by
  simpa using shapeFixedFailureWitness_density_lower_bound d c position
    (structuralShapeFixedFailureWitness d c position structural failure)

/-! ## Payload-list shape preservation -/

private theorem filterMap_boundedEntries_value_length
    {α β : Type} (values : List α) (fallback : α)
    (read : Nat → α → β) :
    ((boundedEntries values fallback).zipIdx.filterMap fun entry =>
      if entry.1.1 = true then
        some (read entry.2 entry.1.2)
      else none).length = values.length := by
  have loop : ∀ (entries : List (Bool × α)) (start : Nat),
      ((entries.zipIdx start).filterMap fun entry =>
        if entry.1.1 = true then
          some (read entry.2 entry.1.2)
        else none).length =
        (entries.filter fun entry => entry.1).length := by
    intro entries start
    induction entries generalizing start with
    | nil => rfl
    | cons entry entries induction =>
        rcases entry with ⟨present, value⟩
        cases present <;>
          simp [List.zipIdx, induction (start := start + 1)]
  rw [loop]
  simp only [boundedEntries, List.filter_append, List.filter_map]
  simp

@[simp] theorem decodeWitness_shapeFixed_dependency_length
    (d : Graph) (c : Certificate) (input : ShapeFixedAssignment d c)
    (index : Nat) (witness : EdgeRouteWitness) :
    (decodeWitness d c (shapeFixedToCanonical d c input)
      index witness).dependency.length = witness.dependency.length := by
  simp only [decodeWitness]
  simp only [readFlag_shapeFixed_structural d c input
    (.witnessDependencyPresent index _) _ (by rfl)]
  exact filterMap_boundedEntries_value_length witness.dependency #"BAD"
    (fun entryIndex anchorValue =>
      readFormula d c (shapeFixedToCanonical d c input)
        (.witnessDependency index entryIndex) anchorValue)

@[simp] theorem decodeWitness_shapeFixed_residual_length
    (d : Graph) (c : Certificate) (input : ShapeFixedAssignment d c)
    (index : Nat) (witness : EdgeRouteWitness) :
    (decodeWitness d c (shapeFixedToCanonical d c input)
      index witness).residual.length = witness.residual.length := by
  simp only [decodeWitness]
  simp only [readFlag_shapeFixed_structural d c input
    (.witnessResidualPresent index _) _ (by rfl)]
  exact filterMap_boundedEntries_value_length witness.residual 999
    (fun entryIndex anchorValue =>
      readNatural d c (shapeFixedToCanonical d c input)
        (.witnessResidual index entryIndex) anchorValue)

@[simp] theorem decodeTerminal_shapeFixed_dependency_length
    (d : Graph) (c : Certificate) (input : ShapeFixedAssignment d c)
    (index : Nat) (output : TerminalOutput) :
    (decodeTerminal d c (shapeFixedToCanonical d c input)
      index output).dependency.length = output.dependency.length := by
  simp only [decodeTerminal]
  simp only [readFlag_shapeFixed_structural d c input
    (.terminalDependencyPresent index _) _ (by rfl)]
  exact filterMap_boundedEntries_value_length output.dependency #"BAD"
    (fun entryIndex anchorValue =>
      readFormula d c (shapeFixedToCanonical d c input)
        (.terminalDependency index entryIndex) anchorValue)

@[simp] theorem decodeTerminal_shapeFixed_residual_length
    (d : Graph) (c : Certificate) (input : ShapeFixedAssignment d c)
    (index : Nat) (output : TerminalOutput) :
    (decodeTerminal d c (shapeFixedToCanonical d c input)
      index output).residual.length = output.residual.length := by
  simp only [decodeTerminal]
  simp only [readFlag_shapeFixed_structural d c input
    (.terminalResidualPresent index _) _ (by rfl)]
  exact filterMap_boundedEntries_value_length output.residual 999
    (fun entryIndex anchorValue =>
      readNatural d c (shapeFixedToCanonical d c input)
        (.terminalResidual index entryIndex) anchorValue)

def terminalPayloadLengths (c : Certificate) : List (Nat × Nat × Vertex) :=
  c.terminalOutputs.map fun output =>
    (output.dependency.length, output.residual.length, output.node)

theorem terminalPayloadLengths_decodeShapeFixed (d : Graph)
    (c : Certificate) (input : ShapeFixedAssignment d c) :
    terminalPayloadLengths (decodeShapeFixed d c input) =
      terminalPayloadLengths c := by
  unfold terminalPayloadLengths
  rw [decodeShapeFixed_terminalOutputs_eq]
  calc
    List.map
        (fun output =>
          (output.dependency.length, output.residual.length, output.node))
        (List.map
          (fun entry => decodeTerminal d c (shapeFixedToCanonical d c input)
            entry.2 entry.1) c.terminalOutputs.zipIdx) =
      List.map
        (fun entry =>
          (entry.1.dependency.length, entry.1.residual.length, entry.1.node))
        c.terminalOutputs.zipIdx := by
          rw [List.map_map]
          apply List.map_congr_left
          intro entry member
          rcases entry with ⟨output, index⟩
          simp
    _ = List.map
        (fun output =>
          (output.dependency.length, output.residual.length, output.node))
        c.terminalOutputs := by
          simpa only [List.map_map, Function.comp_apply] using
            congrArg
              (List.map fun output =>
                (output.dependency.length, output.residual.length,
                  output.node))
              (List.zipIdx_map_fst 0 c.terminalOutputs)

/-- A root-terminal failure mode determined purely by frozen list shape: every
terminal record owned by the root has at least one active dependency or
residual entry. -/
def RootTerminalsBlockedByShape (c : Certificate) (vertex : Vertex) : Prop :=
  c.terminalOutputs.all (fun output =>
    output.node != vertex ||
      !(output.dependency.isEmpty && output.residual.isEmpty)) = true

theorem rootDischargeAtB_false_of_terminal_shape
    (d : Graph) (c : Certificate) (vertex : Vertex)
    (root : (get_rule.outgoing vertex c.graph).isEmpty = true)
    (blocked : RootTerminalsBlockedByShape c vertex)
    (input : ShapeFixedAssignment d c) :
    rootDischargeAtB (decodeShapeFixed d c input) vertex = false := by
  unfold rootDischargeAtB
  simp only [decodeShapeFixed_graph, root, if_true]
  apply List.any_eq_false.mpr
  intro decoded member
  rw [Bool.not_eq_true]
  have decodedMember := (List.mem_filter.mp member).1
  have decodedOwner : decoded.node = vertex := by
    simpa using (List.mem_filter.mp member).2
  rw [decodeShapeFixed_terminalOutputs_eq] at decodedMember
  obtain ⟨entry, entryMember, decodedEq⟩ :=
    List.mem_map.mp decodedMember
  rcases entry with ⟨output, index⟩
  subst decoded
  have zipFacts := List.mem_zipIdx entryMember
  simp only [Nat.zero_le, true_and, zero_add, Nat.sub_zero] at zipFacts
  have outputMember : output ∈ c.terminalOutputs := by
    rw [zipFacts.2]
    exact List.getElem_mem zipFacts.1
  have outputOwner : output.node = vertex := by
    simpa using decodedOwner
  have block := (List.all_eq_true.mp blocked) output outputMember
  have payloadNonempty :
      output.dependency ≠ [] ∨ output.residual ≠ [] := by
    simp [outputOwner] at block
    simpa using block
  rcases payloadNonempty with dependencyNonempty | residualNonempty
  · have decodedDependencyNonempty :
        (decodeTerminal d c (shapeFixedToCanonical d c input)
          index output).dependency ≠ [] := by
      intro empty
      have zero :
          (decodeTerminal d c (shapeFixedToCanonical d c input)
            index output).dependency.length = 0 := by simp [empty]
      rw [decodeTerminal_shapeFixed_dependency_length] at zero
      exact dependencyNonempty (List.length_eq_zero_iff.mp zero)
    simp [decodedDependencyNonempty]
  · have decodedResidualNonempty :
        (decodeTerminal d c (shapeFixedToCanonical d c input)
          index output).residual ≠ [] := by
      intro empty
      have zero :
          (decodeTerminal d c (shapeFixedToCanonical d c input)
            index output).residual.length = 0 := by simp [empty]
      rw [decodeTerminal_shapeFixed_residual_length] at zero
      exact residualNonempty (List.length_eq_zero_iff.mp zero)
    simp [decodedResidualNonempty]

/-- The corresponding one-sided witness fixes no payload coordinates. -/
def rootTerminalShapeFailureWitness (d : Graph) (c : Certificate)
    (slot : Nat) (vertex : Vertex)
    (root : (get_rule.outgoing vertex c.graph).isEmpty = true)
    (blocked : RootTerminalsBlockedByShape c vertex) :
    ShapeFixedFailureWitness d c (.rootDischarge slot vertex) where
  support := ∅
  anchorFailure := by
    have failed := rootDischargeAtB_false_of_terminal_shape
      d c vertex root blocked (encodeShapeFixed d c)
    rw [decodeShapeFixed_encodeShapeFixed] at failed
    unfold shapeFixedSemanticFailureB checkSemanticPositionB
    simp [failed]
  forcesFailure := by
    intro other _
    have failed := rootDischargeAtB_false_of_terminal_shape
      d c vertex root blocked other
    unfold shapeFixedSemanticFailureB checkSemanticPositionB
    simp [failed]

@[simp] theorem rootTerminalShapeFailureWitness_size
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex)
    (root : (get_rule.outgoing vertex c.graph).isEmpty = true)
    (blocked : RootTerminalsBlockedByShape c vertex) :
    failureWitnessSize
      (rootTerminalShapeFailureWitness d c slot vertex root blocked) = 0 :=
  rfl

/-- Shape-frozen source-production obstruction carried by one stored witness whose
frozen residual length cannot occur in any graph-determined source output.
Payload values in every other record remain free.  In fact the offending
record's own values remain free too: only its already-frozen list length is
used. -/
theorem sourceProductionAtB_false_of_residual_length
    (d : Graph) (c : Certificate) (vertex : Vertex)
    (source : localShape c.graph vertex = .source)
    (witness : EdgeRouteWitness) (index : Nat)
    (member : (witness, index) ∈ c.edgeRoutes.zipIdx)
    (owner : witness.edge.START = vertex)
    (unexpectedLength :
      (witnessableExpected c vertex).all (fun route =>
        route.2.length != witness.residual.length) = true)
    (input : ShapeFixedAssignment d c) :
    productionAtB (decodeShapeFixed d c input) vertex = false := by
  let decodedWitness := decodeWitness d c
    (shapeFixedToCanonical d c input) index witness
  have decodedWitnessMember :
      decodedWitness ∈ (decodeShapeFixed d c input).edgeRoutes := by
    rw [decodeShapeFixed_edgeRoutes_eq]
    exact List.mem_map.mpr ⟨(witness, index), member, rfl⟩
  have decodedOwner : decodedWitness.edge.START = vertex := by
    simpa [decodedWitness] using owner
  have storedMember :
      (decodedWitness.dependency, decodedWitness.residual) ∈
        outputsAt (decodeShapeFixed d c input) vertex := by
    unfold outputsAt
    rw [List.mem_eraseDups]
    apply List.mem_append_left
    rw [List.mem_filterMap]
    exact ⟨decodedWitness, decodedWitnessMember, by simp [decodedOwner]⟩
  have expectedEq :
      witnessableExpected (decodeShapeFixed d c input) vertex =
        witnessableExpected c vertex := by
    unfold witnessableExpected localExpected
    simp only [decodeShapeFixed_graph, source]
  have storedNotExpected :
      (decodedWitness.dependency, decodedWitness.residual) ∉
        witnessableExpected (decodeShapeFixed d c input) vertex := by
    rw [expectedEq]
    intro expectedMember
    have mismatch := (List.all_eq_true.mp unexpectedLength)
      (decodedWitness.dependency, decodedWitness.residual) expectedMember
    simp only [bne_iff_ne] at mismatch
    apply mismatch
    simp [decodedWitness]
  have containsFalse :
      (witnessableExpected (decodeShapeFixed d c input) vertex).contains
          (decodedWitness.dependency, decodedWitness.residual) = false := by
    simpa using storedNotExpected
  have leftAllFalse :
      (outputsAt (decodeShapeFixed d c input) vertex).all
          (fun route =>
            (witnessableExpected (decodeShapeFixed d c input) vertex).contains
              route) = false := by
    rw [List.all_eq_false]
    exact ⟨(decodedWitness.dependency, decodedWitness.residual), storedMember,
      by simpa using storedNotExpected⟩
  unfold productionAtB
  simp only [decodeShapeFixed_graph, source]
  unfold setEqB
  rw [leftAllFalse]
  rfl

def sourceResidualLengthFailureWitness
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex)
    (source : localShape c.graph vertex = .source)
    (witness : EdgeRouteWitness) (index : Nat)
    (member : (witness, index) ∈ c.edgeRoutes.zipIdx)
    (owner : witness.edge.START = vertex)
    (unexpectedLength :
      (witnessableExpected c vertex).all (fun route =>
        route.2.length != witness.residual.length) = true) :
    ShapeFixedFailureWitness d c (.nodeProduction slot vertex) where
  support := ∅
  anchorFailure := by
    have failed := sourceProductionAtB_false_of_residual_length
      d c vertex source witness index member owner unexpectedLength
      (encodeShapeFixed d c)
    rw [decodeShapeFixed_encodeShapeFixed] at failed
    unfold shapeFixedSemanticFailureB checkSemanticPositionB
    simp [failed]
  forcesFailure := by
    intro other _
    have failed := sourceProductionAtB_false_of_residual_length
      d c vertex source witness index member owner unexpectedLength other
    unfold shapeFixedSemanticFailureB checkSemanticPositionB
    simp [failed]

@[simp] theorem sourceResidualLengthFailureWitness_size
    (d : Graph) (c : Certificate) (slot : Nat) (vertex : Vertex)
    (source : localShape c.graph vertex = .source)
    (witness : EdgeRouteWitness) (index : Nat)
    (member : (witness, index) ∈ c.edgeRoutes.zipIdx)
    (owner : witness.edge.START = vertex)
    (unexpectedLength :
      (witnessableExpected c vertex).all (fun route =>
        route.2.length != witness.residual.length) = true) :
    failureWitnessSize
      (sourceResidualLengthFailureWitness d c slot vertex source
        witness index member owner unexpectedLength) = 0 := rfl

/-! ## Failure-mode classification

The classification below is deliberately syntactic/logical.  The five
payload-reading families contain universal or absence tests, so no theorem
below silently promotes the entire family to a constant-size class.
-/

inductive FailureObstructionClass where
  | smallPositive
  | absenceOrUniversal
  | structuralFrozen
  | mixed
  | degenerateGlobal
  deriving Repr, BEq, DecidableEq

def verificationFamilyObstructionClass :
    VerificationPosition → FailureObstructionClass
  | .witnessFaithful _ _ => .absenceOrUniversal
  | .nodeProduction _ _ => .absenceOrUniversal
  | .routeCoverage _ _ => .absenceOrUniversal
  | .tailClosure _ _ => .mixed
  | .rootDischarge _ _ => .absenceOrUniversal
  | .degenerateLocal _ => .degenerateGlobal
  | _ => .structuralFrozen

/-- A concrete, checker-specific positive obstruction.  This predicate is
useful for modes such as an immutable malformed record or a single bad value:
fixing one payload coordinate is enough to keep the same position false. -/
def SingleCoordinateFailureObstruction (d : Graph) (c : Certificate)
    (position : VerificationPosition) (coordinate : ShapeFixedCoord d c) :
    Prop :=
  SemanticMarkedAt d c position ∧
    ∀ other : ShapeFixedAssignment d c,
      other coordinate = encodeShapeFixed d c coordinate →
        shapeFixedSemanticFailureB d c position other = true

def singleCoordinateFailureWitness (d : Graph) (c : Certificate)
    (position : VerificationPosition) (coordinate : ShapeFixedCoord d c)
    (obstruction : SingleCoordinateFailureObstruction
      d c position coordinate) : ShapeFixedFailureWitness d c position where
  support := {coordinate}
  anchorFailure := by
    have failure := obstruction.1
    unfold SemanticMarkedAt at failure
    unfold shapeFixedSemanticFailureB
    rw [decodeShapeFixed_encodeShapeFixed]
    simp [failure]
  forcesFailure := by
    intro other agree
    exact obstruction.2 other (agree coordinate (by simp))

@[simp] theorem singleCoordinateFailureWitness_size
    (d : Graph) (c : Certificate) (position : VerificationPosition)
    (coordinate : ShapeFixedCoord d c)
    (obstruction : SingleCoordinateFailureObstruction
      d c position coordinate) :
    failureWitnessSize
      (singleCoordinateFailureWitness d c position coordinate obstruction) = 1 :=
  Finset.card_singleton coordinate

/-! ## Coordinate-specific alphabets -/

abbrev DependentAssignment (Coordinate : Type)
    (Alphabet : Coordinate → Type) := ∀ coordinate, Alphabet coordinate

def DependentAgreeOn {Coordinate : Type} [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type} (support : Finset Coordinate)
    (anchor other : DependentAssignment Coordinate Alphabet) : Prop :=
  ∀ coordinate ∈ support, other coordinate = anchor coordinate

structure DependentFailureWitness {Coordinate : Type}
    [DecidableEq Coordinate] {Alphabet : Coordinate → Type}
    (oracle : DependentAssignment Coordinate Alphabet → Bool)
    (anchor : DependentAssignment Coordinate Alphabet) where
  support : Finset Coordinate
  anchorFailure : oracle anchor = true
  forcesFailure : ∀ other,
    DependentAgreeOn support anchor other → oracle other = true

abbrev DependentFailureCylinder {Coordinate : Type}
    [DecidableEq Coordinate] {Alphabet : Coordinate → Type}
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :=
  { other : DependentAssignment Coordinate Alphabet //
    DependentAgreeOn witness.support anchor other }

abbrev DependentFailureAssignments {Coordinate : Type}
    {Alphabet : Coordinate → Type}
    (oracle : DependentAssignment Coordinate Alphabet → Bool) :=
  { input : DependentAssignment Coordinate Alphabet // oracle input = true }

def dependentFailureCylinderEquiv {Coordinate : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type}
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :
    DependentFailureCylinder witness ≃
      (∀ coordinate : ↥witness.supportᶜ, Alphabet coordinate.1) where
  toFun other coordinate := other.1 coordinate.1
  invFun free :=
    ⟨fun coordinate =>
        if member : coordinate ∈ witness.support then
          anchor coordinate
        else
          free ⟨coordinate, by simpa using member⟩,
      by
        intro coordinate member
        simp [member]⟩
  left_inv other := by
    apply Subtype.ext
    funext coordinate
    by_cases member : coordinate ∈ witness.support
    · simp [member, other.2 coordinate member]
    · simp [member]
  right_inv free := by
    funext coordinate
    have outside : coordinate.1 ∉ witness.support :=
      Finset.mem_compl.mp coordinate.2
    simp [outside]

noncomputable instance dependentFailureCylinderFintype
    {Coordinate : Type} [Fintype Coordinate] [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type} [∀ coordinate, Fintype (Alphabet coordinate)]
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    {witness : DependentFailureWitness oracle anchor} :
    Fintype (DependentFailureCylinder witness) :=
  Fintype.ofEquiv
    (∀ coordinate : ↥witness.supportᶜ, Alphabet coordinate.1)
    (dependentFailureCylinderEquiv witness).symm

noncomputable instance dependentFailureAssignmentsFintype
    {Coordinate : Type} [Fintype Coordinate]
    {Alphabet : Coordinate → Type} [∀ coordinate, Fintype (Alphabet coordinate)]
    {oracle : DependentAssignment Coordinate Alphabet → Bool} :
    Fintype (DependentFailureAssignments oracle) := Fintype.ofFinite _

/-- Exact product-of-coordinate-alphabets cylinder count. -/
theorem dependentFailureCylinder_card {Coordinate : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type}
    [∀ coordinate, Fintype (Alphabet coordinate)]
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :
    Fintype.card (DependentFailureCylinder witness) =
      ∏ coordinate : ↥witness.supportᶜ,
        Fintype.card (Alphabet coordinate.1) := by
  calc
    Fintype.card (DependentFailureCylinder witness) =
        Fintype.card
          (∀ coordinate : ↥witness.supportᶜ, Alphabet coordinate.1) :=
      Fintype.card_congr (dependentFailureCylinderEquiv witness)
    _ = _ := Fintype.card_pi

def dependentFailureCylinderToAssignments {Coordinate : Type}
    [DecidableEq Coordinate] {Alphabet : Coordinate → Type}
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :
    DependentFailureCylinder witness →
      DependentFailureAssignments oracle :=
  fun other => ⟨other.1, witness.forcesFailure other.1 other.2⟩

theorem dependentFailureCylinderToAssignments_injective
    {Coordinate : Type} [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type}
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :
    Function.Injective
      (dependentFailureCylinderToAssignments witness) := by
  intro left right equality
  apply Subtype.ext
  exact congrArg
    (fun bad : DependentFailureAssignments oracle => bad.1) equality

theorem dependentFailureAssignments_lower_bound {Coordinate : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type}
    [∀ coordinate, Fintype (Alphabet coordinate)]
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :
    (∏ coordinate : ↥witness.supportᶜ,
        Fintype.card (Alphabet coordinate.1)) ≤
      Fintype.card (DependentFailureAssignments oracle) := by
  rw [← dependentFailureCylinder_card witness]
  exact Fintype.card_le_of_injective
    (dependentFailureCylinderToAssignments witness)
    (dependentFailureCylinderToAssignments_injective witness)

noncomputable def dependentFailureDensity {Coordinate : Type}
    [Fintype Coordinate] {Alphabet : Coordinate → Type}
    [∀ coordinate, Fintype (Alphabet coordinate)]
    (oracle : DependentAssignment Coordinate Alphabet → Bool) : ℚ :=
  Fintype.card (DependentFailureAssignments oracle) /
    ∏ coordinate, Fintype.card (Alphabet coordinate)

private theorem subtype_compl_product_eq_finset_compl_product
    {Coordinate : Type} [Fintype Coordinate] [DecidableEq Coordinate]
    (support : Finset Coordinate) (f : Coordinate → Nat) :
    (∏ coordinate : ↥supportᶜ, f coordinate.1) =
      ∏ coordinate ∈ supportᶜ, f coordinate := by
  exact Finset.prod_coe_sort supportᶜ f

/-- Coordinate-specific density bound.  A fixed dependency bit is charged
`2`, while a residual or formula cell is charged only its own alphabet size;
no global maximum alphabet is imposed by the theorem. -/
theorem dependentFailureDensity_lower_bound {Coordinate : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    {Alphabet : Coordinate → Type}
    [∀ coordinate, Fintype (Alphabet coordinate)]
    [∀ coordinate, Nonempty (Alphabet coordinate)]
    {oracle : DependentAssignment Coordinate Alphabet → Bool}
    {anchor : DependentAssignment Coordinate Alphabet}
    (witness : DependentFailureWitness oracle anchor) :
    (1 : ℚ) /
        (∏ coordinate ∈ witness.support,
          Fintype.card (Alphabet coordinate)) ≤
      dependentFailureDensity oracle := by
  let fixed : Nat := ∏ coordinate ∈ witness.support,
    Fintype.card (Alphabet coordinate)
  let free : Nat := ∏ coordinate ∈ witness.supportᶜ,
    Fintype.card (Alphabet coordinate)
  have fixedPositive : 0 < fixed := by
    unfold fixed
    positivity
  have totalPositive : 0 <
      (∏ coordinate, Fintype.card (Alphabet coordinate)) := by
    positivity
  have lowerRaw := dependentFailureAssignments_lower_bound witness
  have lower : free ≤
      Fintype.card (DependentFailureAssignments oracle) := by
    have productEq := subtype_compl_product_eq_finset_compl_product
      witness.support (fun coordinate => Fintype.card (Alphabet coordinate))
    rw [productEq] at lowerRaw
    exact lowerRaw
  have partition : fixed * free =
      ∏ coordinate, Fintype.card (Alphabet coordinate) := by
    unfold fixed free
    exact Finset.prod_mul_prod_compl witness.support _
  unfold dependentFailureDensity
  apply (div_le_div_iff₀ (by exact_mod_cast fixedPositive)
    (by exact_mod_cast totalPositive)).2
  norm_num
  exact_mod_cast (show
    (∏ coordinate, Fintype.card (Alphabet coordinate)) ≤
      Fintype.card (DependentFailureAssignments oracle) * fixed by
    rw [← partition]
    simpa [Nat.mul_comm] using Nat.mul_le_mul_left fixed lower)


end Robustness
end CertificateSearch
