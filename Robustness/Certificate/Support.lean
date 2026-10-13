import Search.Certificate.Definitions
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Update

/-!
# Robustness of proof-specific certificate malformations

`SpaceACounting` concerns the separate proof-search `PathInput` carrier.  This
file does not identify a rejected path assignment with a malformed DLDS.
Instead it states the analogous finite-support theorem for the actual expanded-
certificate oracle, `CertificateSearch.markedAt`.  For every supplied finite
certificate decoder it computes the inclusion-minimal essential-coordinate
support and proves that agreement there fixes the real verifier result.

`CanonicalCertificateEncoding` instantiates this layer with an
anchor-bounded field encoding of all verifier-visible certificate data.  This
file remains representation-parametric so that the orbit/cardinality results
can also be reused by other finite encodings.
-/

open DLDS FlowSemantics Certificate.Flow.Foundation

namespace CertificateSearch
namespace Robustness

/-- A finite encoding of variants of one supplied certificate. -/
abbrev Assignment (Coordinate Alphabet : Type) := Coordinate → Alphabet

/-- Two encodings agree on all coordinates used by a local malformation
witness. -/
def AgreeOn {Coordinate Alphabet : Type} [DecidableEq Coordinate]
    (support : Finset Coordinate)
    (anchor other : Assignment Coordinate Alphabet) : Prop :=
  ∀ coordinate ∈ support, other coordinate = anchor coordinate

/-- The existing proof-specific local-failure oracle, transported through a
finite certificate encoding. -/
def localFailureB {Coordinate Alphabet : Type} (d : Graph)
    (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (input : Assignment Coordinate Alphabet) : Bool :=
  markedAt d (decode input) position

/-! ## Automatically extracted finite supports

For a fixed finite decoder, a coordinate belongs to the support exactly when
changing only that coordinate can change the real `markedAt` oracle.  Thus the
definition is extensional (and executable), not a manually asserted read set.
It also makes no assumptions about how a decoder packages certificate fields.
-/

/-- Executable test that a coordinate is essential to a Boolean oracle. -/
def coordinateAffectsB {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (coordinate : Coordinate) : Bool :=
  decide (∃ input value,
    oracle (Function.update input coordinate value) != oracle input)

theorem coordinateAffectsB_eq_true {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (coordinate : Coordinate) :
    coordinateAffectsB oracle coordinate = true ↔
      ∃ input value,
        oracle (Function.update input coordinate value) != oracle input := by
  simp [coordinateAffectsB]

/-- The inclusion-minimal semantic support of a Boolean oracle over a finite
product encoding: precisely the coordinates which can affect its result. -/
def supportOfOracle {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool) : Finset Coordinate :=
  Finset.univ.filter fun coordinate =>
    coordinateAffectsB oracle coordinate = true

theorem mem_supportOfOracle_iff {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (coordinate : Coordinate) :
    coordinate ∈ supportOfOracle oracle ↔
      ∃ input value,
        oracle (Function.update input coordinate value) != oracle input := by
  simp [supportOfOracle, coordinateAffectsB_eq_true]

/-- Automatically computed support of one position of the actual certificate
verifier, transported through a finite certificate decoder. -/
def supportOfPosition {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) : Finset Coordinate :=
  supportOfOracle (localFailureB d decode position)

/-- Failure bit for the total formation oracle, including the searchable
`ExpandedCert` prefix. -/
def totalFailureB {Coordinate Alphabet : Type} (d : Graph)
    (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (input : Assignment Coordinate Alphabet) : Bool :=
  !totalCheckPositionB d (decode input) position

/-- Automatic essential-coordinate support for a total formation position. -/
def supportOfTotalPosition {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) : Finset Coordinate :=
  supportOfOracle (totalFailureB d decode position)

theorem mem_supportOfPosition_iff {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (coordinate : Coordinate) :
    coordinate ∈ supportOfPosition d decode position ↔
      ∃ input value,
        localFailureB d decode position
            (Function.update input coordinate value) !=
          localFailureB d decode position input := by
  exact mem_supportOfOracle_iff (localFailureB d decode position) coordinate

theorem oracle_update_of_not_mem_support {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    {coordinate : Coordinate} (input : Assignment Coordinate Alphabet)
    (value : Alphabet) (outside : coordinate ∉ supportOfOracle oracle) :
    oracle (Function.update input coordinate value) = oracle input := by
  by_contra changed
  apply outside
  simp only [supportOfOracle, Finset.mem_filter, Finset.mem_univ, true_and]
  exact (coordinateAffectsB_eq_true oracle coordinate).2
    ⟨input, value, by simpa using changed⟩

private theorem oracle_updateFinset_of_disjoint_support
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor other : Assignment Coordinate Alphabet)
    (coordinates : Finset Coordinate)
    (outside : Disjoint coordinates (supportOfOracle oracle)) :
    oracle (Function.updateFinset anchor coordinates
      (fun coordinate => other coordinate.1)) = oracle anchor := by
  induction coordinates using Finset.induction_on with
  | empty => simp
  | @insert coordinate coordinates notMem ih =>
      have coordinateOutside : coordinate ∉ supportOfOracle oracle :=
        Finset.disjoint_left.mp outside (Finset.mem_insert_self _ _)
      have restOutside : Disjoint coordinates (supportOfOracle oracle) :=
        Finset.disjoint_left.mpr fun item itemMem supportMem =>
          Finset.disjoint_left.mp outside
            (Finset.mem_insert_of_mem itemMem) supportMem
      let updated := Function.updateFinset anchor coordinates
        (fun item => other item.1)
      have updateEq :
          Function.updateFinset anchor (insert coordinate coordinates)
              (fun item => other item.1) =
            Function.update updated coordinate (other coordinate) := by
        funext item
        by_cases itemEq : item = coordinate
        · subst itemEq
          simp [Function.updateFinset, updated]
        · by_cases itemMem : item ∈ coordinates
          · simp [Function.updateFinset, updated, itemEq, itemMem]
          · simp [Function.updateFinset, updated, itemEq, itemMem]
      rw [updateEq, oracle_update_of_not_mem_support oracle updated
        (other coordinate) coordinateOutside]
      exact ih restOutside

/-- Agreement on the automatically extracted support fixes the oracle value.
-/
theorem supportOfOracle_agreement {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (anchor other : Assignment Coordinate Alphabet)
    (agree : AgreeOn (supportOfOracle oracle) anchor other) :
    oracle other = oracle anchor := by
  let complement := (supportOfOracle oracle)ᶜ
  have outside : Disjoint complement (supportOfOracle oracle) :=
    disjoint_compl_left
  have updatedEq : Function.updateFinset anchor complement
        (fun coordinate => other coordinate.1) = other := by
    funext coordinate
    by_cases member : coordinate ∈ supportOfOracle oracle
    · have notComplement : coordinate ∉ complement := by
        simp [complement, member]
      simp [Function.updateFinset, notComplement,
        agree coordinate member]
    · have inComplement : coordinate ∈ complement := by
        simp [complement, member]
      simp [Function.updateFinset, inComplement]
  rw [← updatedEq]
  exact oracle_updateFinset_of_disjoint_support oracle anchor other
    complement outside

/-- Inclusion-minimality: every set which determines the oracle for all
encodings contains the automatically extracted support. -/
theorem supportOfOracle_minimal {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (oracle : Assignment Coordinate Alphabet → Bool)
    (support : Finset Coordinate)
    (sufficient : ∀ anchor other,
      AgreeOn support anchor other → oracle other = oracle anchor) :
    supportOfOracle oracle ⊆ support := by
  intro coordinate essential
  obtain ⟨input, value, changed⟩ :=
    (mem_supportOfOracle_iff oracle coordinate).mp essential
  by_contra outside
  have agree : AgreeOn support input
      (Function.update input coordinate value) := by
    intro item item_mem
    have item_ne : item ≠ coordinate :=
      fun item_eq => outside (item_eq ▸ item_mem)
    rw [Function.update_of_ne item_ne]
  have unchanged := sufficient input
    (Function.update input coordinate value) agree
  have changed' : oracle (Function.update input coordinate value) ≠
      oracle input := by simpa using changed
  exact changed' unchanged

/-- The real local-failure bit is invariant under agreement on its automatic
support. -/
theorem supportOfPosition_agreement {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (anchor other : Assignment Coordinate Alphabet)
    (agree : AgreeOn (supportOfPosition d decode position) anchor other) :
    localFailureB d decode position other =
      localFailureB d decode position anchor := by
  exact supportOfOracle_agreement (localFailureB d decode position)
    anchor other agree

theorem supportOfPosition_minimal {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (support : Finset Coordinate)
    (sufficient : ∀ anchor other,
      AgreeOn support anchor other →
        localFailureB d decode position other =
          localFailureB d decode position anchor) :
    supportOfPosition d decode position ⊆ support := by
  exact supportOfOracle_minimal (localFailureB d decode position)
    support sufficient

theorem supportOfTotalPosition_agreement {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (anchor other : Assignment Coordinate Alphabet)
    (agree : AgreeOn (supportOfTotalPosition d decode position) anchor other) :
    totalFailureB d decode position other =
      totalFailureB d decode position anchor := by
  exact supportOfOracle_agreement (totalFailureB d decode position)
    anchor other agree

/-- No automatically extracted position support is larger than the coordinate
carrier of the chosen finite encoding. -/
theorem supportOfPosition_card_le {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) :
    (supportOfPosition d decode position).card ≤ Fintype.card Coordinate := by
  exact Finset.card_le_univ _

theorem supportOfTotalPosition_card_le {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) :
    (supportOfTotalPosition d decode position).card ≤
      Fintype.card Coordinate := by
  exact Finset.card_le_univ _

/-- Evidence that one genuine `MarkedAt` failure is determined by a finite set
of coordinates of a certificate encoding.  This packages locality; it does not
assume that arbitrary `gridMarkedB = false` paths are malformed certificates. -/
structure MalformationWitness {Coordinate Alphabet : Type}
    [DecidableEq Coordinate] (d : Graph)
    (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet) where
  position : Nat
  position_lt : position < numPositions d (decode anchor)
  support : Finset Coordinate
  failure : localFailureB d decode position anchor = true
  localized : ∀ other, AgreeOn support anchor other →
    localFailureB d decode position other =
      localFailureB d decode position anchor

/-- Canonical witness obtained from the actual verifier and the automatic
essential-coordinate support. -/
def automaticMalformationWitness {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (position : Nat) (position_lt : position < numPositions d (decode anchor))
    (failure : MarkedAt d (decode anchor) position) :
    MalformationWitness d decode anchor where
  position := position
  position_lt := position_lt
  support := supportOfPosition d decode position
  failure := by
    simpa [localFailureB] using
      (markedAt_reflect d (decode anchor) position).2 failure
  localized := fun other agree =>
    supportOfPosition_agreement d decode position anchor other agree

/-- Maximum automatic support among the positions of the anchor certificate. -/
def maxLocalSupport {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet) : Nat :=
  (Finset.range (numPositions d (decode anchor))).fold max 0 fun position =>
    (supportOfPosition d decode position).card

/-- Every valid search position is bounded by the exact per-certificate
maximum support. -/
theorem supportOfPosition_card_le_maxLocalSupport
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet) {position : Nat}
    (position_lt : position < numPositions d (decode anchor)) :
    (supportOfPosition d decode position).card ≤
      maxLocalSupport d decode anchor := by
  rw [maxLocalSupport, Finset.le_fold_max]
  exact Or.inr ⟨position, Finset.mem_range.mpr position_lt, le_rfl⟩

theorem maxLocalSupport_le_coordinate_card {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet) :
    maxLocalSupport d decode anchor ≤ Fintype.card Coordinate := by
  unfold maxLocalSupport
  rw [Finset.fold_max_le]
  constructor
  · exact Nat.zero_le _
  · intro position _
    exact supportOfPosition_card_le d decode position

/-- Encodings obtained by varying every coordinate outside the witness
support. -/
abbrev MalformationOrbit {Coordinate Alphabet : Type}
    [DecidableEq Coordinate] {d : Graph}
    {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :=
  { other : Assignment Coordinate Alphabet //
    AgreeOn witness.support anchor other }

/-- Encodings whose decoded certificate exposes the same proof-specific local
malformation at `position`. -/
abbrev MalformationAssignments {Coordinate Alphabet : Type} (d : Graph)
    (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) :=
  { input : Assignment Coordinate Alphabet //
    localFailureB d decode position input = true }

noncomputable instance malformationAssignmentsFintype
    {Coordinate Alphabet : Type} [Fintype Coordinate] [Fintype Alphabet]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {position : Nat} :
    Fintype (MalformationAssignments d decode position) :=
  Fintype.ofFinite _

/-- LEVEL 1: changing coordinates outside a supplied support cannot change the
corresponding proof-specific local-failure bit. -/
theorem localFailure_unread_invariant {Coordinate Alphabet : Type}
    [DecidableEq Coordinate] {d : Graph}
    {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor)
    {other : Assignment Coordinate Alphabet}
    (agree : AgreeOn witness.support anchor other) :
    localFailureB d decode witness.position other =
      localFailureB d decode witness.position anchor :=
  witness.localized other agree

theorem localFailure_of_mem_orbit {Coordinate Alphabet : Type}
    [DecidableEq Coordinate] {d : Graph}
    {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor)
    (other : MalformationOrbit witness) :
    localFailureB d decode witness.position other.1 = true := by
  rw [localFailure_unread_invariant witness other.2]
  exact witness.failure

/-- The malformation orbit is equivalent to an arbitrary assignment on the
complement of the witness support. -/
def malformationOrbitEquiv {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    MalformationOrbit witness ≃ (↥witness.supportᶜ → Alphabet) where
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
    have notMember : ¬ coordinate.1 ∈ witness.support :=
      Finset.mem_compl.mp coordinate.2
    simp [notMember]

noncomputable instance malformationOrbitFintype
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [Fintype Alphabet] [DecidableEq Coordinate]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    {witness : MalformationWitness d decode anchor} :
    Fintype (MalformationOrbit witness) :=
  Fintype.ofEquiv (↥witness.supportᶜ → Alphabet)
    (malformationOrbitEquiv witness).symm

/-- LEVEL 2: if the finite input carrier has `T` coordinates, the alphabet has
`A` values, and the witness fixes `r` coordinates, its orbit has exactly
`A^(T-r)` members. -/
theorem malformationOrbit_card {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    Fintype.card (MalformationOrbit witness) =
      Fintype.card Alphabet ^
        (Fintype.card Coordinate - witness.support.card) := by
  calc
    Fintype.card (MalformationOrbit witness) =
        Fintype.card (↥witness.supportᶜ → Alphabet) :=
      Fintype.card_congr (malformationOrbitEquiv witness)
    _ = Fintype.card Alphabet ^ Fintype.card (↥witness.supportᶜ) :=
      Fintype.card_fun
    _ = Fintype.card Alphabet ^ witness.supportᶜ.card := by
      rw [Fintype.card_coe]
    _ = Fintype.card Alphabet ^
        (Fintype.card Coordinate - witness.support.card) := by
      congr 1
      exact Finset.card_compl witness.support

/-- Every member of a support orbit is an assignment exposing the same
proof-specific certificate malformation. -/
def malformationOrbitToAssignments {Coordinate Alphabet : Type}
    [DecidableEq Coordinate] {d : Graph}
    {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    MalformationOrbit witness →
      MalformationAssignments d decode witness.position :=
  fun other => ⟨other.1, localFailure_of_mem_orbit witness other⟩

lemma malformationOrbitToAssignments_injective
    {Coordinate Alphabet : Type} [DecidableEq Coordinate]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    Function.Injective (malformationOrbitToAssignments witness) := by
  intro left right equal
  apply Subtype.ext
  exact congrArg
    (fun bad : MalformationAssignments d decode witness.position => bad.1)
    equal

/-- LEVEL 3: one supported local malformation contributes at least its full
`A^(T-r)` orbit to the set of inputs exposing that same malformation. -/
theorem malformationAssignments_lower_bound
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    Fintype.card Alphabet ^
        (Fintype.card Coordinate - witness.support.card) ≤
      Fintype.card
        (MalformationAssignments d decode witness.position) := by
  rw [← malformationOrbit_card witness]
  exact Fintype.card_le_of_injective
    (malformationOrbitToAssignments witness)
    (malformationOrbitToAssignments_injective witness)

/-- Total size of the finite encoding carrier. -/
theorem assignment_card (Coordinate Alphabet : Type)
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet] :
    Fintype.card (Assignment Coordinate Alphabet) =
      Fintype.card Alphabet ^ Fintype.card Coordinate := by
  simp [Assignment]

/-- Natural-number form of the exact density consequence
`density(bad) ≥ A^(-r)`. -/
theorem malformationDensity_cross_bound
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    Fintype.card Alphabet ^ Fintype.card Coordinate ≤
      Fintype.card Alphabet ^ witness.support.card *
        Fintype.card
          (MalformationAssignments d decode witness.position) := by
  have supportLe : witness.support.card ≤ Fintype.card Coordinate := by
    exact Finset.card_le_univ _
  have lower := malformationAssignments_lower_bound witness
  calc
    Fintype.card Alphabet ^ Fintype.card Coordinate =
        Fintype.card Alphabet ^ witness.support.card *
          Fintype.card Alphabet ^
            (Fintype.card Coordinate - witness.support.card) := by
      rw [← pow_add, Nat.add_sub_of_le supportLe]
    _ ≤ Fintype.card Alphabet ^ witness.support.card *
        Fintype.card
          (MalformationAssignments d decode witness.position) :=
      Nat.mul_le_mul_left _ lower

noncomputable def malformationDensity {Coordinate Alphabet : Type}
    [Fintype Coordinate] [Fintype Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) : ℚ :=
  Fintype.card (MalformationAssignments d decode position) /
    Fintype.card Alphabet ^ Fintype.card Coordinate

/-- Rational presentation of the supported density bound.  This is exact in
terms of `A` and `r`; it is not labelled inverse-polynomial because the project
does not yet bound `r` as a function of certificate size. -/
theorem malformationDensity_lower_bound
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet] [Nonempty Alphabet]
    {d : Graph} {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :
    (1 : ℚ) / Fintype.card Alphabet ^ witness.support.card ≤
      malformationDensity d decode witness.position := by
  have alphabetPositive : 0 < Fintype.card Alphabet := Fintype.card_pos
  have leftDenominatorPositive :
      (0 : ℚ) < Fintype.card Alphabet ^ witness.support.card := by
    positivity
  have rightDenominatorPositive :
      (0 : ℚ) < Fintype.card Alphabet ^ Fintype.card Coordinate := by
    positivity
  unfold malformationDensity
  apply (div_le_div_iff₀ leftDenominatorPositive rightDenominatorPositive).2
  norm_num
  have cross := malformationDensity_cross_bound witness
  exact_mod_cast (show
    Fintype.card Alphabet ^ Fintype.card Coordinate ≤
      Fintype.card
          (MalformationAssignments d decode witness.position) *
        Fintype.card Alphabet ^ witness.support.card by
    simpa [Nat.mul_comm] using cross)

/-- LEVEL 4 at the exact current scope: an expanded certificate which fails
`LocalChecks` has a proof-specific local malformation. -/
theorem formation_invalid_implies_localMalformation (d : Graph)
    (certificate : Certificate)
    (expanded : Certificate.Flow.ExpandedCert d certificate)
    (invalid : ¬ Certificate.Flow.LocalChecks d certificate) :
    ∃ position < numPositions d certificate,
      MarkedAt d certificate position :=
  CertificateSearch.formation_invalid_implies_marked d certificate expanded invalid

/-- An expanded, locally invalid decoded certificate automatically supplies a
supported genuine malformation.  The conclusion records both the exact
support exponent and the per-certificate maximum-support density bound; no
external `MalformationWitness` or locality premise is required. -/
theorem formation_invalid_automatic_amplification
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet] [Nonempty Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet)
    (expanded : Certificate.Flow.ExpandedCert d (decode anchor))
    (invalid : ¬ Certificate.Flow.LocalChecks d (decode anchor)) :
    ∃ position, position < numPositions d (decode anchor) ∧
      MarkedAt d (decode anchor) position ∧
      (supportOfPosition d decode position).card ≤
        maxLocalSupport d decode anchor ∧
      Fintype.card Alphabet ^
          (Fintype.card Coordinate -
            (supportOfPosition d decode position).card) ≤
        Fintype.card (MalformationAssignments d decode position) ∧
      (1 : ℚ) / Fintype.card Alphabet ^
          (supportOfPosition d decode position).card ≤
        malformationDensity d decode position ∧
      (1 : ℚ) / Fintype.card Alphabet ^
          maxLocalSupport d decode anchor ≤
        malformationDensity d decode position := by
  obtain ⟨position, position_lt, marked⟩ :=
    formation_invalid_implies_localMalformation d (decode anchor)
      expanded invalid
  let witness := automaticMalformationWitness (decode := decode)
    position position_lt marked
  have supportEq : witness.support = supportOfPosition d decode position := rfl
  have supportLe := supportOfPosition_card_le_maxLocalSupport
    d decode anchor position_lt
  have orbitLower := malformationAssignments_lower_bound witness
  have densityLower := malformationDensity_lower_bound witness
  have alphabetPositive : 0 < Fintype.card Alphabet := Fintype.card_pos
  have powerLeNat :
      Fintype.card Alphabet ^ (supportOfPosition d decode position).card ≤
        Fintype.card Alphabet ^ maxLocalSupport d decode anchor :=
    Nat.pow_le_pow_right alphabetPositive supportLe
  have powerLeRat :
      (Fintype.card Alphabet : ℚ) ^
          (supportOfPosition d decode position).card ≤
        (Fintype.card Alphabet : ℚ) ^
          maxLocalSupport d decode anchor := by
    exact_mod_cast powerLeNat
  have supportPowerPositive :
      (0 : ℚ) < (Fintype.card Alphabet : ℚ) ^
        (supportOfPosition d decode position).card := by
    positivity
  have maxDensityLower :
      (1 : ℚ) / Fintype.card Alphabet ^
          maxLocalSupport d decode anchor ≤
        malformationDensity d decode position := by
    exact (one_div_le_one_div_of_le supportPowerPositive powerLeRat).trans
      (by simpa [supportEq] using densityLower)
  refine ⟨position, position_lt, marked, supportLe, ?_, ?_, maxDensityLower⟩
  · simpa [supportEq] using orbitLower
  · simpa [supportEq] using densityLower

/-- Legacy conditional composition retained for callers which already possess
a syntactic support.  `formation_invalid_automatic_amplification` removes this
premise by using the computed essential-coordinate support. -/
theorem formation_invalid_amplification
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet)
    (expanded : Certificate.Flow.ExpandedCert d (decode anchor))
    (invalid : ¬ Certificate.Flow.LocalChecks d (decode anchor))
    (locality : ∀ position,
      position < numPositions d (decode anchor) →
      MarkedAt d (decode anchor) position →
      ∃ support : Finset Coordinate, ∀ other,
        AgreeOn support anchor other →
        localFailureB d decode position other =
          localFailureB d decode position anchor) :
    ∃ witness : MalformationWitness d decode anchor,
      Fintype.card Alphabet ^
          (Fintype.card Coordinate - witness.support.card) ≤
        Fintype.card
          (MalformationAssignments d decode witness.position) := by
  obtain ⟨position, position_lt, marked⟩ :=
    formation_invalid_implies_localMalformation d (decode anchor)
      expanded invalid
  obtain ⟨support, localized⟩ := locality position position_lt marked
  have failure : localFailureB d decode position anchor = true :=
    (markedAt_reflect d (decode anchor) position).2 marked
  let witness : MalformationWitness d decode anchor :=
    { position := position
      position_lt := position_lt
      support := support
      failure := failure
      localized := localized }
  exact ⟨witness, malformationAssignments_lower_bound witness⟩

/-! ## Exact discharge information supplied by local checking

The existing root checker is existential.  Combining it with the proved
stored-route/Flow equality therefore yields an empty Flow route at every root,
but not the universal empty-route condition used by `rootsDischargedB` and not
the evaluator equality needed for `dischargedMultiB`.
-/

theorem localChecks_root_has_empty_witnessableRoute (d : Graph)
    (c : Certificate) (expanded : Certificate.Flow.ExpandedCert d c)
    (hlocal : Certificate.Flow.LocalChecks d c)
    {vertex : Vertex} (vertex_mem : vertex ∈ d.NODES)
    (root : (get_rule.outgoing vertex d).isEmpty = true) :
    ([], []) ∈ WitnessableRoutes d vertex
      (flowAt d (stdFuel d) vertex) := by
  have vertex_mem_c : vertex ∈ c.graph.NODES := by
    simpa [hlocal.1] using vertex_mem
  have root_c : (get_rule.outgoing vertex c.graph).isEmpty = true := by
    simpa [hlocal.1] using root
  obtain ⟨output, output_mem, output_node, output_dep, output_residual⟩ :=
    (hlocal.2.2.2.2.1 vertex vertex_mem_c root_c).1
  have stored : ([], []) ∈ Certificate.Flow.StoredRoutes c vertex := by
    unfold Certificate.Flow.StoredRoutes outputsAt
    rw [List.mem_eraseDups]
    apply List.mem_append_right
    rw [List.mem_filterMap]
    exact ⟨output, output_mem,
      by simp [output_node, output_dep, output_residual]⟩
  exact (Certificate.Flow.storedRoutes_setEq_flow d c expanded hlocal
    vertex vertex_mem ([], [])).mp stored

theorem localChecks_root_has_empty_flowRoute (d : Graph)
    (c : Certificate) (expanded : Certificate.Flow.ExpandedCert d c)
    (hlocal : Certificate.Flow.LocalChecks d c)
    {vertex : Vertex} (vertex_mem : vertex ∈ d.NODES)
    (root : (get_rule.outgoing vertex d).isEmpty = true) :
    ([], []) ∈ flowAt d (stdFuel d) vertex := by
  have witnessable := localChecks_root_has_empty_witnessableRoute d c
    expanded hlocal vertex_mem root
  unfold WitnessableRoutes at witnessable
  rw [List.mem_filter] at witnessable
  exact witnessable.1

/-- Concrete Boolean form of the quantifier gap: existence of an empty root
route does not imply that every root route is empty. -/
theorem rootDischarge_existential_universal_gap :
    ([(([], []) : FlowPair), ([#"BAD"], [])].any fun pair =>
        pair.1.isEmpty && pair.2.isEmpty) = true ∧
      ([(([], []) : FlowPair), ([#"BAD"], [])].all fun pair =>
        pair.1.isEmpty && pair.2.isEmpty) = false := by
  native_decide

/-! ## Executable terminal-corruption/orbit fixture -/

abbrev FixtureCoordinate := Fin 3

def terminalCorruptionAnchor : Assignment FixtureCoordinate Bool :=
  fun coordinate => decide (coordinate = 0)

/-- A finite family of genuine expanded certificates.  Coordinate `0`
corrupts exactly the first terminal dependency.  Coordinates `1` and `2`
alter an unrelated residual and reorder witnesses, respectively. -/
def terminalCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let residualVariant :=
    if input 1 then alterFirstResidual eCertificate else eCertificate
  let orderVariant :=
    if input 2 then
      { residualVariant with edgeRoutes := residualVariant.edgeRoutes.reverse }
    else residualVariant
  if input 0 then corruptTerminal orderVariant else orderVariant

def terminalCorruptionWitness : MalformationWitness DLDS.e
    terminalCorruptionFamily terminalCorruptionAnchor where
  position := 336
  position_lt := by native_decide
  support := {0}
  failure := by native_decide
  localized := by
    intro other agree
    have zeroEq := agree 0 (by simp)
    have zeroTrue : other 0 = true := by
      simpa [terminalCorruptionAnchor] using zeroEq
    cases oneEq : other 1 <;> cases twoEq : other 2 <;>
      simp [localFailureB, terminalCorruptionFamily, zeroTrue, oneEq, twoEq,
        terminalCorruptionAnchor] <;> native_decide

/-- The unmodified supplied certificate passes all formation checks. -/
theorem endToEndCertificate_all_local_checks :
    Certificate.Flow.localChecksB DLDS.e eCertificate = true := by
  native_decide

/-- The anchor changes only the terminal dependency and fails the root position
of the proof-specific search oracle. -/
theorem terminalCorruption_root_position_fails :
    terminalCorruptionFamily terminalCorruptionAnchor =
        corruptTerminal eCertificate ∧
      checkPositionB DLDS.e
        (terminalCorruptionFamily terminalCorruptionAnchor) 336 = false := by
  constructor
  · simp [terminalCorruptionFamily, terminalCorruptionAnchor]
  · native_decide

theorem terminalCorruption_expanded_but_local_invalid :
    Certificate.Flow.expandedCertB DLDS.e
        (terminalCorruptionFamily terminalCorruptionAnchor) = true ∧
      Certificate.Flow.localChecksB DLDS.e
        (terminalCorruptionFamily terminalCorruptionAnchor) = false := by
  native_decide

theorem terminalCorruption_exact_local_failures :
    checkPositionB DLDS.e
        (terminalCorruptionFamily terminalCorruptionAnchor) 333 = false ∧
      checkPositionB DLDS.e
        (terminalCorruptionFamily terminalCorruptionAnchor) 336 = false ∧
      markedCount DLDS.e
        (terminalCorruptionFamily terminalCorruptionAnchor) = 2 := by
  native_decide

theorem terminalCorruption_support_is_one_cell :
    terminalCorruptionWitness.support = {0} ∧
      terminalCorruptionWitness.support.card = 1 := by
  native_decide

/-- Exhaustive four-member orbit test: every variation outside coordinate `0`
continues to expose the same root malformation. -/
theorem terminalCorruption_orbit_exposes_root
    (other : MalformationOrbit terminalCorruptionWitness) :
    localFailureB DLDS.e terminalCorruptionFamily 336 other.1 = true :=
  localFailure_of_mem_orbit terminalCorruptionWitness other

theorem terminalCorruption_orbit_card :
    Fintype.card (MalformationOrbit terminalCorruptionWitness) = 4 := by
  rw [malformationOrbit_card]
  native_decide

theorem terminalCorruption_bad_assignments_lower_bound :
    4 ≤ Fintype.card
      (MalformationAssignments DLDS.e terminalCorruptionFamily 336) := by
  simpa using
    (malformationAssignments_lower_bound terminalCorruptionWitness)

abbrev FixtureBits := Bool × Bool × Bool

def fixtureAssignment (bits : FixtureBits) : Assignment FixtureCoordinate Bool :=
  fun coordinate =>
    match coordinate.val with
    | 0 => bits.1
    | 1 => bits.2.1
    | _ => bits.2.2

def fixtureBitInputs : List FixtureBits :=
  [(false, false, false), (false, false, true),
   (false, true, false), (false, true, true),
   (true, false, false), (true, false, true),
   (true, true, false), (true, true, true)]

/-- Executable census `(all inputs, support orbit, failing orbit members)`.
This exhausts the eight three-bit encodings; precisely four fix coordinate `0`,
and all four expose root position `336`. -/
def terminalCorruptionOrbitCensus : Nat × Nat × Nat :=
  let orbit := fixtureBitInputs.filter fun bits =>
    fixtureAssignment bits 0 == terminalCorruptionAnchor 0
  let failing := orbit.filter fun bits =>
    localFailureB DLDS.e terminalCorruptionFamily 336
      (fixtureAssignment bits)
  (fixtureBitInputs.length, orbit.length, failing.length)

theorem terminalCorruptionOrbitCensus_native :
    terminalCorruptionOrbitCensus = (8, 4, 4) := by
  native_decide

/-! ## Automatic-support corruption matrix

Each family below has one active certificate-field mutation coordinate and two
free coordinates.  The latter deliberately decode to no certificate change,
so the computed essential support -- rather than a hand-written support --
must discover the singleton active coordinate.
-/

/-- Corrupt only the dependency stored on the first route witness. -/
def corruptFirstWitnessDependency (c : Certificate) : Certificate :=
  match c.edgeRoutes with
  | [] => c
  | witness :: rest =>
      let badWitness : EdgeRouteWitness :=
        { edge := witness.edge
          dependency := normalizeDep (witness.dependency ++ [#"BAD"])
          residual := witness.residual }
      { c with edgeRoutes := badWitness :: rest }

/-- Corrupt the dependency produced by the last implication-introduction
witness (the final witness of the `eCertificate` fixture). -/
def corruptLastIntroWitnessDependency (c : Certificate) : Certificate :=
  match c.edgeRoutes.reverse with
  | [] => c
  | witness :: rest =>
      let badWitness : EdgeRouteWitness :=
        { edge := witness.edge
          dependency := normalizeDep (witness.dependency ++ [#"BAD"])
          residual := witness.residual }
      { c with edgeRoutes := (badWitness :: rest).reverse }

def isolatedCorruptionFamily (corrupt : Certificate → Certificate)
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  if input 0 then corrupt eCertificate else eCertificate

/-- Dependency corruption with two genuine, non-interfering certificate
variations: route order and the terminal dependency. -/
def dependencyCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let ordered := if input 1 then
    { eCertificate with edgeRoutes := eCertificate.edgeRoutes.reverse }
  else eCertificate
  let terminal := if input 2 then corruptTerminal ordered else ordered
  if input 0 then corruptFirstWitnessDependency terminal else terminal

/-- Final-introduction dependency corruption with unrelated first-residual and
route-order variations. -/
def introductionCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let introduced := if input 0 then
    corruptLastIntroWitnessDependency eCertificate
  else eCertificate
  let residual := if input 1 then alterFirstResidual introduced else introduced
  if input 2 then
    { residual with edgeRoutes := residual.edgeRoutes.reverse }
  else residual

/-- First-residual corruption with route-order and terminal variations. -/
def residualCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let residual := if input 0 then alterFirstResidual eCertificate else eCertificate
  let ordered := if input 1 then
    { residual with edgeRoutes := residual.edgeRoutes.reverse }
  else residual
  if input 2 then corruptTerminal ordered else ordered

/-- Missing first witness with route-order and terminal variations. -/
def coverageCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let covered := if input 0 then omitFirstWitness eCertificate else eCertificate
  let ordered := if input 1 then
    { covered with edgeRoutes := covered.edgeRoutes.reverse }
  else covered
  if input 2 then corruptTerminal ordered else ordered

structure AutomaticCorruptionReport where
  name : String
  failurePosition : Nat
  supportSize : Nat
  predictedOrbitSize : Nat
  orbitSize : Nat
  failingOrbitSize : Nat
  markedPositions : Nat
  totalPositions : Nat
  deriving Repr, BEq, DecidableEq

def fixtureAgreesOnSupportB (support : Finset FixtureCoordinate)
    (input : Assignment FixtureCoordinate Bool) : Bool :=
  (!decide (0 ∈ support) ||
      input 0 == terminalCorruptionAnchor 0) &&
    (!decide (1 ∈ support) ||
      input 1 == terminalCorruptionAnchor 1) &&
    (!decide (2 ∈ support) ||
      input 2 == terminalCorruptionAnchor 2)

def automaticCorruptionReport (name : String)
    (family : Assignment FixtureCoordinate Bool → Certificate)
    (position : Nat) :
    AutomaticCorruptionReport :=
  let support := supportOfPosition DLDS.e family position
  let orbit := fixtureBitInputs.filter fun bits =>
    fixtureAgreesOnSupportB support (fixtureAssignment bits)
  let failing := orbit.filter fun bits =>
    localFailureB DLDS.e family position (fixtureAssignment bits)
  let corrupted := family terminalCorruptionAnchor
  { name
    failurePosition := position
    supportSize := support.card
    predictedOrbitSize := 2 ^ (Fintype.card FixtureCoordinate - support.card)
    orbitSize := orbit.length
    failingOrbitSize := failing.length
    markedPositions := markedCount DLDS.e corrupted
    totalPositions := numPositions DLDS.e corrupted }

def automaticCorruptionReports : List AutomaticCorruptionReport :=
  [automaticCorruptionReport "wrong-dependency-vector"
      dependencyCorruptionFamily 201,
   automaticCorruptionReport "wrong-introduction-discharge"
      introductionCorruptionFamily 327,
   automaticCorruptionReport "wrong-route-residual"
      residualCorruptionFamily 231,
   automaticCorruptionReport "wrong-terminal-root-dependency"
      terminalCorruptionFamily 336,
   automaticCorruptionReport "missing-edge-witness-coverage"
      coverageCorruptionFamily 145]

/-- Five independent corruption classes.  In every case the automatically
computed support has one bit, its predicted four-member orbit is exact, and
all four members retain the same local failure.  The last two fields are the
separate marked-position census for the one corrupted certificate. -/
theorem automaticCorruptionReports_native : automaticCorruptionReports =
    [{ name := "wrong-dependency-vector", failurePosition := 201,
       supportSize := 1, predictedOrbitSize := 4, orbitSize := 4,
       failingOrbitSize := 4, markedPositions := 4, totalPositions := 337 },
     { name := "wrong-introduction-discharge", failurePosition := 327,
       supportSize := 1, predictedOrbitSize := 4, orbitSize := 4,
       failingOrbitSize := 4, markedPositions := 4, totalPositions := 337 },
     { name := "wrong-route-residual", failurePosition := 231,
       supportSize := 1, predictedOrbitSize := 4, orbitSize := 4,
       failingOrbitSize := 4, markedPositions := 3, totalPositions := 337 },
     { name := "wrong-terminal-root-dependency", failurePosition := 336,
       supportSize := 1, predictedOrbitSize := 4, orbitSize := 4,
       failingOrbitSize := 4, markedPositions := 2, totalPositions := 337 },
     { name := "missing-edge-witness-coverage", failurePosition := 145,
       supportSize := 1, predictedOrbitSize := 4, orbitSize := 4,
       failingOrbitSize := 4, markedPositions := 5, totalPositions := 336 }] := by
  native_decide

/-- The omitted-witness fixture is also caught by the new `ExpandedCert`
prefix.  Position `1` is the first per-edge witness-coverage position, and its
automatic total-oracle support is the active corruption bit. -/
theorem missingWitness_total_formation_fixture :
    TotalMarkedAt DLDS.e
        (isolatedCorruptionFamily omitFirstWitness terminalCorruptionAnchor) 1 ∧
      (supportOfTotalPosition DLDS.e
        (isolatedCorruptionFamily omitFirstWitness) 1).card = 1 ∧
      totalNumPositions DLDS.e
        (isolatedCorruptionFamily omitFirstWitness terminalCorruptionAnchor) =
          411 := by
  constructor
  · change totalCheckPositionB DLDS.e
      (isolatedCorruptionFamily omitFirstWitness terminalCorruptionAnchor) 1 =
        false
    native_decide
  · native_decide


end Robustness
end CertificateSearch
