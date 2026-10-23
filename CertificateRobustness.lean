import SearchProblem
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Update

















open Semantic FlowSpec SelfCertifyingProbe

namespace SearchProblem
namespace CertificateRobustness


abbrev Assignment (Coordinate Alphabet : Type) := Coordinate → Alphabet



def AgreeOn {Coordinate Alphabet : Type} [DecidableEq Coordinate]
    (support : Finset Coordinate)
    (anchor other : Assignment Coordinate Alphabet) : Prop :=
  ∀ coordinate ∈ support, other coordinate = anchor coordinate



def localFailureB {Coordinate Alphabet : Type} (d : Graph)
    (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (input : Assignment Coordinate Alphabet) : Bool :=
  markedAt d (decode input) position










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



def supportOfPosition {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) : Finset Coordinate :=
  supportOfOracle (localFailureB d decode position)



def totalFailureB {Coordinate Alphabet : Type} (d : Graph)
    (decode : Assignment Coordinate Alphabet → Certificate)
    (position : Nat) (input : Assignment Coordinate Alphabet) : Bool :=
  !totalCheckPositionB d (decode input) position


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


def maxLocalSupport {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet) : Nat :=
  (Finset.range (numPositions d (decode anchor))).fold max 0 fun position =>
    (supportOfPosition d decode position).card



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



abbrev MalformationOrbit {Coordinate Alphabet : Type}
    [DecidableEq Coordinate] {d : Graph}
    {decode : Assignment Coordinate Alphabet → Certificate}
    {anchor : Assignment Coordinate Alphabet}
    (witness : MalformationWitness d decode anchor) :=
  { other : Assignment Coordinate Alphabet //
    AgreeOn witness.support anchor other }



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


theorem assignment_card (Coordinate Alphabet : Type)
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet] :
    Fintype.card (Assignment Coordinate Alphabet) =
      Fintype.card Alphabet ^ Fintype.card Coordinate := by
  simp [Assignment]



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



theorem formation_invalid_implies_localMalformation (d : Graph)
    (certificate : Certificate)
    (expanded : SelfCertifying.ExpandedCert d certificate)
    (invalid : ¬ SelfCertifying.LocalChecks d certificate) :
    ∃ position < numPositions d certificate,
      MarkedAt d certificate position :=
  SearchProblem.formation_invalid_implies_marked d certificate expanded invalid





theorem formation_invalid_automatic_amplification
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet] [Nonempty Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet)
    (expanded : SelfCertifying.ExpandedCert d (decode anchor))
    (invalid : ¬ SelfCertifying.LocalChecks d (decode anchor)) :
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




theorem formation_invalid_amplification
    {Coordinate Alphabet : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    [Fintype Alphabet] [DecidableEq Alphabet]
    (d : Graph) (decode : Assignment Coordinate Alphabet → Certificate)
    (anchor : Assignment Coordinate Alphabet)
    (expanded : SelfCertifying.ExpandedCert d (decode anchor))
    (invalid : ¬ SelfCertifying.LocalChecks d (decode anchor))
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









theorem localChecks_root_has_empty_witnessableRoute (d : Graph)
    (c : Certificate) (expanded : SelfCertifying.ExpandedCert d c)
    (hlocal : SelfCertifying.LocalChecks d c)
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
  have stored : ([], []) ∈ SelfCertifying.StoredRoutes c vertex := by
    unfold SelfCertifying.StoredRoutes outputsAt
    rw [List.mem_eraseDups]
    apply List.mem_append_right
    rw [List.mem_filterMap]
    exact ⟨output, output_mem,
      by simp [output_node, output_dep, output_residual]⟩
  exact (SelfCertifying.storedRoutes_setEq_flow d c expanded hlocal
    vertex vertex_mem ([], [])).mp stored

theorem localChecks_root_has_empty_flowRoute (d : Graph)
    (c : Certificate) (expanded : SelfCertifying.ExpandedCert d c)
    (hlocal : SelfCertifying.LocalChecks d c)
    {vertex : Vertex} (vertex_mem : vertex ∈ d.NODES)
    (root : (get_rule.outgoing vertex d).isEmpty = true) :
    ([], []) ∈ flowAt d (stdFuel d) vertex := by
  have witnessable := localChecks_root_has_empty_witnessableRoute d c
    expanded hlocal vertex_mem root
  unfold WitnessableRoutes at witnessable
  rw [List.mem_filter] at witnessable
  exact witnessable.1



theorem rootDischarge_existential_universal_gap :
    ([(([], []) : FlowPair), ([#"BAD"], [])].any fun pair =>
        pair.1.isEmpty && pair.2.isEmpty) = true ∧
      ([(([], []) : FlowPair), ([#"BAD"], [])].all fun pair =>
        pair.1.isEmpty && pair.2.isEmpty) = false := by
  native_decide



abbrev FixtureCoordinate := Fin 3

def terminalCorruptionAnchor : Assignment FixtureCoordinate Bool :=
  fun coordinate => decide (coordinate = 0)




def terminalCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let residualVariant :=
    if input 1 then alterFirstResidual eCertificate else eCertificate
  let orderVariant :=
    if input 2 then
      { residualVariant with edgeRoutes := residualVariant.edgeRoutes.reverse }
    else residualVariant
  if input 0 then corruptTerminal orderVariant else orderVariant

def terminalCorruptionWitness : MalformationWitness Semantic.e
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


theorem endToEndCertificate_all_local_checks :
    SelfCertifying.localChecksB Semantic.e eCertificate = true := by
  native_decide



theorem terminalCorruption_root_position_fails :
    terminalCorruptionFamily terminalCorruptionAnchor =
        corruptTerminal eCertificate ∧
      checkPositionB Semantic.e
        (terminalCorruptionFamily terminalCorruptionAnchor) 336 = false := by
  constructor
  · simp [terminalCorruptionFamily, terminalCorruptionAnchor]
  · native_decide

theorem terminalCorruption_expanded_but_local_invalid :
    SelfCertifying.expandedCertB Semantic.e
        (terminalCorruptionFamily terminalCorruptionAnchor) = true ∧
      SelfCertifying.localChecksB Semantic.e
        (terminalCorruptionFamily terminalCorruptionAnchor) = false := by
  native_decide

theorem terminalCorruption_exact_local_failures :
    checkPositionB Semantic.e
        (terminalCorruptionFamily terminalCorruptionAnchor) 333 = false ∧
      checkPositionB Semantic.e
        (terminalCorruptionFamily terminalCorruptionAnchor) 336 = false ∧
      markedCount Semantic.e
        (terminalCorruptionFamily terminalCorruptionAnchor) = 2 := by
  native_decide

theorem terminalCorruption_support_is_one_cell :
    terminalCorruptionWitness.support = {0} ∧
      terminalCorruptionWitness.support.card = 1 := by
  native_decide



theorem terminalCorruption_orbit_exposes_root
    (other : MalformationOrbit terminalCorruptionWitness) :
    localFailureB Semantic.e terminalCorruptionFamily 336 other.1 = true :=
  localFailure_of_mem_orbit terminalCorruptionWitness other

theorem terminalCorruption_orbit_card :
    Fintype.card (MalformationOrbit terminalCorruptionWitness) = 4 := by
  rw [malformationOrbit_card]
  native_decide

theorem terminalCorruption_bad_assignments_lower_bound :
    4 ≤ Fintype.card
      (MalformationAssignments Semantic.e terminalCorruptionFamily 336) := by
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




def terminalCorruptionOrbitCensus : Nat × Nat × Nat :=
  let orbit := fixtureBitInputs.filter fun bits =>
    fixtureAssignment bits 0 == terminalCorruptionAnchor 0
  let failing := orbit.filter fun bits =>
    localFailureB Semantic.e terminalCorruptionFamily 336
      (fixtureAssignment bits)
  (fixtureBitInputs.length, orbit.length, failing.length)

theorem terminalCorruptionOrbitCensus_native :
    terminalCorruptionOrbitCensus = (8, 4, 4) := by
  native_decide










def corruptFirstWitnessDependency (c : Certificate) : Certificate :=
  match c.edgeRoutes with
  | [] => c
  | witness :: rest =>
      let badWitness : EdgeRouteWitness :=
        { edge := witness.edge
          dependency := normalizeDep (witness.dependency ++ [#"BAD"])
          residual := witness.residual }
      { c with edgeRoutes := badWitness :: rest }



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



def dependencyCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let ordered := if input 1 then
    { eCertificate with edgeRoutes := eCertificate.edgeRoutes.reverse }
  else eCertificate
  let terminal := if input 2 then corruptTerminal ordered else ordered
  if input 0 then corruptFirstWitnessDependency terminal else terminal



def introductionCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let introduced := if input 0 then
    corruptLastIntroWitnessDependency eCertificate
  else eCertificate
  let residual := if input 1 then alterFirstResidual introduced else introduced
  if input 2 then
    { residual with edgeRoutes := residual.edgeRoutes.reverse }
  else residual


def residualCorruptionFamily
    (input : Assignment FixtureCoordinate Bool) : Certificate :=
  let residual := if input 0 then alterFirstResidual eCertificate else eCertificate
  let ordered := if input 1 then
    { residual with edgeRoutes := residual.edgeRoutes.reverse }
  else residual
  if input 2 then corruptTerminal ordered else ordered


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
  let support := supportOfPosition Semantic.e family position
  let orbit := fixtureBitInputs.filter fun bits =>
    fixtureAgreesOnSupportB support (fixtureAssignment bits)
  let failing := orbit.filter fun bits =>
    localFailureB Semantic.e family position (fixtureAssignment bits)
  let corrupted := family terminalCorruptionAnchor
  { name
    failurePosition := position
    supportSize := support.card
    predictedOrbitSize := 2 ^ (Fintype.card FixtureCoordinate - support.card)
    orbitSize := orbit.length
    failingOrbitSize := failing.length
    markedPositions := markedCount Semantic.e corrupted
    totalPositions := numPositions Semantic.e corrupted }

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




theorem missingWitness_total_formation_fixture :
    TotalMarkedAt Semantic.e
        (isolatedCorruptionFamily omitFirstWitness terminalCorruptionAnchor) 1 ∧
      (supportOfTotalPosition Semantic.e
        (isolatedCorruptionFamily omitFirstWitness) 1).card = 1 ∧
      totalNumPositions Semantic.e
        (isolatedCorruptionFamily omitFirstWitness terminalCorruptionAnchor) =
          411 := by
  constructor
  · change totalCheckPositionB Semantic.e
      (isolatedCorruptionFamily omitFirstWitness terminalCorruptionAnchor) 1 =
        false
    native_decide
  · native_decide


end CertificateRobustness
end SearchProblem
