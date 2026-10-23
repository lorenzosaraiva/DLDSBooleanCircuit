import Robustness.CompleteProfileRobustness









open Semantic FlowSpec SelfCertifying RouteCertificate CompleteProfilePositions

namespace CompleteProfileRobustness



theorem decodeRows_injective (c : CompleteProfileCertificate) : Function.Injective (decodeRows c) := by
  intro left right equal
  funext key
  apply Subtype.ext
  have profiles := congrArg (fun certificate => stored certificate key.val) equal
  simpa [stored_decodeRows, assignedProfile, key.property] using profiles

def PositiveRootDefectComplete (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  ∃ route ∈ stored c v, route.1 ≠ [] ∨ route.2 ≠ []

def AbsentRootDefectComplete (c : CompleteProfileCertificate) (v : Vertex) : Prop := stored c v = []

theorem root_failure_positive_or_absence (d : Graph) (c : CompleteProfileCertificate)
    (slot : Nat) (v : Vertex) :
    FailedAtComplete d c (.rootDischarge slot v) ↔
      PositiveRootDefectComplete c v ∨ AbsentRootDefectComplete c v := by
  simp only [FailedAtComplete, checkPositionCompleteB, rootDischargeAtCompleteB,
    Bool.and_eq_false_iff, Bool.not_eq_false', List.isEmpty_iff,
    List.all_eq_false, PositiveRootDefectComplete, AbsentRootDefectComplete]
  simpa only [Bool.and_eq_true, List.isEmpty_iff, not_and_or] using
    (or_comm (a := stored c v = [])
      (b := ∃ route ∈ stored c v, route.1 ≠ [] ∨ route.2 ≠ []))

def ExtraProductionRouteComplete (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  ∃ route ∈ stored c v, route ∉ localExpectedComplete c v

def MissingProductionRouteComplete (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  ∃ route ∈ localExpectedComplete c v, route ∉ stored c v

def OrderedProductionDefectComplete (c : CompleteProfileCertificate) (v : Vertex) : Prop :=
  stored c v ≠ localExpectedComplete c v ∧
    ListSetEq (stored c v) (localExpectedComplete c v)



theorem production_failure_cases (d : Graph) (c : CompleteProfileCertificate)
    (slot : Nat) (v : Vertex) :
    FailedAtComplete d c (.productionRow slot v) ↔
      ExtraProductionRouteComplete c v ∨ MissingProductionRouteComplete c v ∨
        OrderedProductionDefectComplete c v := by
  classical
  change (decide (stored c v = localExpectedComplete c v) = false) ↔ _
  rw [decide_eq_false_iff_not]
  constructor
  · intro different
    by_cases extra : ExtraProductionRouteComplete c v
    · exact Or.inl extra
    by_cases missing : MissingProductionRouteComplete c v
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
    [route, route] ≠ [route] ∧ ListSetEq [route, route] [route] ∧
      ¬ (∃ value ∈ [route, route], value ∉ [route]) ∧
      ¬ (∃ value ∈ [route], value ∉ [route, route]) := by
  simp [ListSetEq]

theorem metadata_failure_density_one (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) (metadata : positionReadVertices c position = [])
    (failed : FailedAtComplete d c position) :
    (1 : ℚ) ≤ failureDensityComplete d c position := by
  have bound := positionReadBoundComplete_le c position
  rw [metadata] at bound
  have zero : positionReadBoundComplete c position = 0 := by simpa using bound
  simpa [zero] using failed_position_density d c position failed



theorem root_defect_failure_persists (d : Graph) (c : CompleteProfileCertificate)
    (slot : Nat) (v : Vertex)
    (defect : PositiveRootDefectComplete c v ∨ AbsentRootDefectComplete c v)
    (other : RowAssignment c)
    (agree : SearchProblem.CertificateRobustness.AgreeOn
      (positionReadSetComplete c (.rootDischarge slot v)) (rowAnchor c) other) :
    FailedAtComplete d (decodeRows c other) (.rootDischarge slot v) := by
  exact failure_persists d c _
    ((root_failure_positive_or_absence d c slot v).mpr defect) other agree

#print axioms root_failure_positive_or_absence
#print axioms production_failure_cases
#print axioms root_defect_failure_persists
end CompleteProfileRobustness
