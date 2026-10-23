import Semantic.CompleteProfilePositions
import Semantic.CompleteProfileMutation










open Semantic FlowSpec RouteCertificate CompleteProfilePositions
open SearchProblem.CertificateRobustness

namespace CompleteProfileRobustness

def positionReadSetComplete (c : CompleteProfileCertificate) (position : VerificationPositionComplete) :=
  rowReadSet c (positionReadVertices c position)

def positionReadBoundComplete (c : CompleteProfileCertificate) (position : VerificationPositionComplete) : Nat :=
  (positionReadSetComplete c position).card

theorem positionReadBoundComplete_le (c : CompleteProfileCertificate) (position : VerificationPositionComplete) :
    positionReadBoundComplete c position ≤ (positionReadVertices c position).length :=
  rowReadSet_card_le c _

theorem production_read_bound (c : CompleteProfileCertificate) (slot : Nat) (v : Vertex) :
    positionReadBoundComplete c (.productionRow slot v) ≤ 1 + (predsOf c.graph v).length := by
  simpa [positionReadVertices, Nat.add_comm] using
    positionReadBoundComplete_le c (.productionRow slot v)

theorem root_read_bound (c : CompleteProfileCertificate) (slot : Nat) (v : Vertex) :
    positionReadBoundComplete c (.rootDischarge slot v) ≤ 1 := by
  simpa [positionReadVertices] using positionReadBoundComplete_le c (.rootDischarge slot v)

def failureOracleComplete (d : Graph) (c : CompleteProfileCertificate) (position : VerificationPositionComplete)
    (assignment : RowAssignment c) : Bool :=
  !(checkPositionCompleteB d (decodeRows c assignment) position)

theorem check_anchor (d : Graph) (c : CompleteProfileCertificate) (position : VerificationPositionComplete) :
    checkPositionCompleteB d (decodeRows c (rowAnchor c)) position =
      checkPositionCompleteB d c position := by
  apply checkPositionCompleteB_congr
  · rfl
  · exact decodeRows_keys c _
  · intro v _
    exact stored_decodeRows_anchor c v

theorem check_agreement (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) (left right : RowAssignment c)
    (agree : AgreeOn (positionReadSetComplete c position) left right) :
    checkPositionCompleteB d (decodeRows c left) position =
      checkPositionCompleteB d (decodeRows c right) position := by
  apply checkPositionCompleteB_congr
  · rfl
  · rw [decodeRows_keys, decodeRows_keys]
  · have readVertices : positionReadVertices (decodeRows c left) position =
        positionReadVertices c position := by cases position <;> rfl
    rw [readVertices]
    exact stored_decodeRows_agree c _ left right agree

def positionFailureWitnessComplete (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) (failed : FailedAtComplete d c position) :
    FailureWitness (failureOracleComplete d c position) (rowAnchor c) where
  support := positionReadSetComplete c position
  anchorFailure := by
    unfold failureOracleComplete
    rw [check_anchor]
    simp [FailedAtComplete] at failed
    simp [failed]
  forcesFailure := by
    intro other agree
    unfold failureOracleComplete
    rw [← check_agreement d c position (rowAnchor c) other agree, check_anchor]
    simp [FailedAtComplete] at failed
    simp [failed]



theorem failure_persists (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) (failed : FailedAtComplete d c position)
    (other : RowAssignment c)
    (agree : AgreeOn (positionReadSetComplete c position) (rowAnchor c) other) :
    FailedAtComplete d (decodeRows c other) position := by
  have forced := (positionFailureWitnessComplete d c position failed).forcesFailure other agree
  simpa [failureOracleComplete, FailedAtComplete] using forced

noncomputable def failureDensityComplete (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) : ℚ :=
  oracleFailureDensity (failureOracleComplete d c position)

theorem failed_position_density (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) (failed : FailedAtComplete d c position) :
    (1 : ℚ) / Fintype.card (RowAlphabet c) ^ positionReadBoundComplete c position ≤
      failureDensityComplete d c position := by
  exact failureDensity_lower_bound (positionFailureWitnessComplete d c position failed)

theorem failed_position_cylinder_card (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) (failed : FailedAtComplete d c position) :
    Fintype.card (FailureCylinder (positionFailureWitnessComplete d c position failed)) =
      Fintype.card (RowAlphabet c) ^
        (Fintype.card (RowCoordinate c) - positionReadBoundComplete c position) := by
  exact failureCylinder_card (positionFailureWitnessComplete d c position failed)



theorem formation_invalid_semantic_amplification (d : Graph) (c : CompleteProfileCertificate)
    (invalid : ¬ StrongFormationValidComplete d c) :
    ∃ position ∈ positionsComplete d, FailedAtComplete d c position ∧
      positionReadBoundComplete c position ≤ (positionReadVertices c position).length ∧
      (1 : ℚ) / Fintype.card (RowAlphabet c) ^ positionReadBoundComplete c position ≤
        failureDensityComplete d c position := by
  obtain ⟨position, member, failed⟩ := invalid_has_failed_position d c invalid
  exact ⟨position, member, failed, positionReadBoundComplete_le c position,
    failed_position_density d c position failed⟩



theorem formation_invalid_has_badCylinder (d : Graph) (c : CompleteProfileCertificate)
    (invalid : ¬ StrongFormationValidComplete d c) :
    ∃ position ∈ positionsComplete d,
      ∃ witness : FailureWitness (failureOracleComplete d c position) (rowAnchor c),
        witness.support = positionReadSetComplete c position ∧
        Fintype.card (FailureCylinder witness) =
          Fintype.card (RowAlphabet c) ^
            (Fintype.card (RowCoordinate c) - positionReadBoundComplete c position) ∧
        (1 : ℚ) / Fintype.card (RowAlphabet c) ^ positionReadBoundComplete c position ≤
          failureDensityComplete d c position := by
  obtain ⟨position, member, failed⟩ := invalid_has_failed_position d c invalid
  exact ⟨position, member, positionFailureWitnessComplete d c position failed, rfl,
    failed_position_cylinder_card d c position failed,
    failed_position_density d c position failed⟩

#print axioms formation_invalid_semantic_amplification
#print axioms formation_invalid_has_badCylinder
#print axioms failure_persists
end CompleteProfileRobustness
