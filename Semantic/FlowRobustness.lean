import Semantic.Flow
import Semantic.MultiIntroPositions
import Semantic.CompleteProfileDefects

open Semantic FlowSpec RouteCertificate CompleteProfileRobustness
open SearchProblem.CertificateRobustness

namespace Flow.Robustness

abbrev check := MultiIntroPositions.check .general
abbrev FailedAt := MultiIntroPositions.FailedAt .general
abbrev StrongCertificate := MultiIntro.StrongCertificate .general

def readSet (c : CompleteProfileCertificate) (position : MultiIntroPositions.Position) :=
  rowReadSet c (MultiIntroPositions.readVertices c position)

def readBound (c : CompleteProfileCertificate) (position : MultiIntroPositions.Position) : Nat :=
  (readSet c position).card

theorem read_bound_le (c : CompleteProfileCertificate) (position : MultiIntroPositions.Position) :
    readBound c position ≤ (MultiIntroPositions.readVertices c position).length :=
  rowReadSet_card_le c _

theorem production_read_bound (c : CompleteProfileCertificate) (slot : Nat) (v : Vertex) :
    readBound c (.productionRow slot v) ≤ 1 + (predsOf c.graph v).length := by
  simpa [MultiIntroPositions.readVertices, Nat.add_comm] using
    read_bound_le c (.productionRow slot v)

theorem root_read_bound (c : CompleteProfileCertificate) (slot : Nat) (v : Vertex) :
    readBound c (.rootDischarge slot v) ≤ 1 := by
  simpa [MultiIntroPositions.readVertices] using read_bound_le c (.rootDischarge slot v)

def failureOracle (d : Graph) (c : CompleteProfileCertificate) (position : MultiIntroPositions.Position)
    (assignment : RowAssignment c) : Bool :=
  !(check d (decodeRows c assignment) position)

theorem check_anchor (d : Graph) (c : CompleteProfileCertificate) (position : MultiIntroPositions.Position) :
    check d (decodeRows c (rowAnchor c)) position =
      check d c position := by
  apply MultiIntroPositions.check_congr .general
  · rfl
  · exact decodeRows_keys c _
  · intro v _
    exact stored_decodeRows_anchor c v

theorem check_agreement (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position) (left right : RowAssignment c)
    (agree : AgreeOn (readSet c position) left right) :
    check d (decodeRows c left) position =
      check d (decodeRows c right) position := by
  apply MultiIntroPositions.check_congr .general
  · rfl
  · rw [decodeRows_keys, decodeRows_keys]
  · have readVertices : MultiIntroPositions.readVertices (decodeRows c left) position =
        MultiIntroPositions.readVertices c position := by cases position <;> rfl
    rw [readVertices]
    exact stored_decodeRows_agree c _ left right agree

def positionFailureWitness (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position) (failed : FailedAt d c position) :
    FailureWitness (failureOracle d c position) (rowAnchor c) where
  support := readSet c position
  anchorFailure := by
    unfold failureOracle
    rw [check_anchor]
    simp [FailedAt, MultiIntroPositions.FailedAt] at failed
    simp [failed]
  forcesFailure := by
    intro other agree
    unfold failureOracle
    rw [← check_agreement d c position (rowAnchor c) other agree, check_anchor]
    simp [FailedAt, MultiIntroPositions.FailedAt] at failed
    simp [failed]



theorem failure_persists (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position) (failed : FailedAt d c position)
    (other : RowAssignment c)
    (agree : AgreeOn (readSet c position) (rowAnchor c) other) :
    FailedAt d (decodeRows c other) position := by
  have forced := (positionFailureWitness d c position failed).forcesFailure other agree
  simpa [failureOracle, FailedAt] using forced

noncomputable def failureDensity (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position) : ℚ :=
  oracleFailureDensity (failureOracle d c position)

theorem failed_position_density (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position) (failed : FailedAt d c position) :
    (1 : ℚ) / Fintype.card (RowAlphabet c) ^ readBound c position ≤
      failureDensity d c position := by
  exact failureDensity_lower_bound (positionFailureWitness d c position failed)

theorem failed_position_cylinder_card (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position) (failed : FailedAt d c position) :
    Fintype.card (FailureCylinder (positionFailureWitness d c position failed)) =
      Fintype.card (RowAlphabet c) ^
        (Fintype.card (RowCoordinate c) - readBound c position) := by
  exact failureCylinder_card (positionFailureWitness d c position failed)



theorem formation_invalid_semantic_amplification (d : Graph) (c : CompleteProfileCertificate)
    (invalid : ¬ StrongCertificate d c) :
    ∃ position ∈ MultiIntroPositions.positions d, FailedAt d c position ∧
      readBound c position ≤ (MultiIntroPositions.readVertices c position).length ∧
      (1 : ℚ) / Fintype.card (RowAlphabet c) ^ readBound c position ≤
        failureDensity d c position := by
  obtain ⟨position, member, failed⟩ := MultiIntroPositions.invalid_has_failed_position .general d c invalid
  exact ⟨position, member, failed, read_bound_le c position,
    failed_position_density d c position failed⟩



theorem formation_invalid_has_badCylinder (d : Graph) (c : CompleteProfileCertificate)
    (invalid : ¬ StrongCertificate d c) :
    ∃ position ∈ MultiIntroPositions.positions d,
      ∃ witness : FailureWitness (failureOracle d c position) (rowAnchor c),
        witness.support = readSet c position ∧
        Fintype.card (FailureCylinder witness) =
          Fintype.card (RowAlphabet c) ^
            (Fintype.card (RowCoordinate c) - readBound c position) ∧
        (1 : ℚ) / Fintype.card (RowAlphabet c) ^ readBound c position ≤
          failureDensity d c position := by
  obtain ⟨position, member, failed⟩ := MultiIntroPositions.invalid_has_failed_position .general d c invalid
  exact ⟨position, member, positionFailureWitness d c position failed, rfl,
    failed_position_cylinder_card d c position failed,
    failed_position_density d c position failed⟩


theorem decodeRows_injective (c : CompleteProfileCertificate) : Function.Injective (decodeRows c) :=
  CompleteProfileRobustness.decodeRows_injective c

abbrev PositiveRootDefect := CompleteProfileRobustness.PositiveRootDefectComplete
abbrev AbsentRootDefect := CompleteProfileRobustness.AbsentRootDefectComplete

theorem root_failure_positive_or_absence (d : Graph) (c : CompleteProfileCertificate)
    (slot : Nat) (v : Vertex) :
    FailedAt d c (.rootDischarge slot v) ↔
      PositiveRootDefect c v ∨ AbsentRootDefect c v :=
  CompleteProfileRobustness.root_failure_positive_or_absence d c slot v

def ExtraProductionRoute (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  ∃ route ∈ stored c v, route ∉ localExpectedFor (MultiIntro.kernel .general) c v

def MissingProductionRoute (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  ∃ route ∈ localExpectedFor (MultiIntro.kernel .general) c v, route ∉ stored c v

def OrderedProductionDefect (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  stored c v ≠ localExpectedFor (MultiIntro.kernel .general) c v ∧
    SelfCertifying.ListSetEq (stored c v) (localExpectedFor (MultiIntro.kernel .general) c v)

theorem production_failure_cases (d : Graph) (c : CompleteProfileCertificate)
    (slot : Nat) (v : Vertex) :
    FailedAt d c (.productionRow slot v) ↔
      ExtraProductionRoute c v ∨ MissingProductionRoute c v ∨ OrderedProductionDefect c v := by
  classical
  change (decide (stored c v = localExpectedFor (MultiIntro.kernel .general) c v) = false) ↔ _
  rw [decide_eq_false_iff_not]
  constructor
  · intro different
    by_cases extra : ExtraProductionRoute c v
    · exact Or.inl extra
    by_cases missing : MissingProductionRoute c v
    · exact Or.inr (Or.inl missing)
    refine Or.inr (Or.inr ⟨different, ?_⟩)
    intro route
    constructor
    · intro present
      by_contra absent
      exact extra ⟨route, present, absent⟩
    · intro present
      by_contra absent
      exact missing ⟨route, present, absent⟩
  · rintro (⟨route, present, absent⟩ | ⟨route, present, absent⟩ | ⟨different, _⟩)
    · intro equality
      exact absent (equality ▸ present)
    · intro equality
      exact absent (equality.symm ▸ present)
    · exact different

theorem duplicate_profile_boundary (route : FlowPair) :
    [route, route] ≠ [route] ∧ SelfCertifying.ListSetEq [route, route] [route] ∧
      ¬ (∃ value ∈ [route, route], value ∉ [route]) ∧
      ¬ (∃ value ∈ [route], value ∉ [route, route]) :=
  CompleteProfileRobustness.duplicate_profile_boundary route

theorem metadata_failure_density_one (d : Graph) (c : CompleteProfileCertificate)
    (position : MultiIntroPositions.Position)
    (metadata : MultiIntroPositions.readVertices c position = [])
    (failed : FailedAt d c position) : (1 : ℚ) ≤ failureDensity d c position := by
  have bound := read_bound_le c position
  rw [metadata] at bound
  have zero : readBound c position = 0 := by simpa using bound
  simpa [zero] using failed_position_density d c position failed

theorem root_defect_failure_persists (d : Graph) (c : CompleteProfileCertificate)
    (slot : Nat) (v : Vertex)
    (defect : PositiveRootDefect c v ∨ AbsentRootDefect c v)
    (other : RowAssignment c)
    (agree : AgreeOn (readSet c (.rootDischarge slot v)) (rowAnchor c) other) :
    FailedAt d (decodeRows c other) (.rootDischarge slot v) := by
  exact failure_persists d c _
    ((root_failure_positive_or_absence d c slot v).mpr defect) other agree

end Flow.Robustness
