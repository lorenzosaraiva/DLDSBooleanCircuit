import Infrastructure.DecorationGuardDefinitions

open Semantic FlowSpec CertificateModel SelfCertifying

namespace Decoration

private theorem fixed_guards (d : Graph) (closed : flowTableClosedB d = true)
    (fuel : Nat)
    (fixed : CacheCorrectness.flowTableSpec d (fuel + 1) =
      CacheCorrectness.flowTableSpec d fuel)
    (good : guardTableB d (CacheCorrectness.flowTableSpec d fuel) = true) :
    ∀ extra, guardTableB d (CacheCorrectness.flowTableSpec d (fuel + extra)) = true := by
  intro extra
  cases extra with
  | zero => simpa using good
  | succ extra =>
      have stable := CacheCorrectness.flowTableSpec_stable_of_fixed d closed fixed extra
      rw [show fuel + (extra + 1) = fuel + 1 + extra by omega, stable, fixed]
      exact good

private theorem guardLoop_sound (d : Graph) (closed : flowTableClosedB d = true) :
    ∀ remaining fuel,
      guardLoop d remaining fuel (CacheCorrectness.flowTableSpec d fuel) = true →
      ∀ extra, guardTableB d (CacheCorrectness.flowTableSpec d (fuel + extra)) = true := by
  intro remaining
  induction remaining with
  | zero =>
      intro fuel checked
      rw [guardLoop, CacheCorrectness.flowTableStep_spec] at checked
      simp only [Bool.and_eq_true] at checked
      obtain ⟨good, rest⟩ := checked
      split at rest
      · rename_i stopped
        have fixed : CacheCorrectness.flowTableSpec d (fuel + 1) =
            CacheCorrectness.flowTableSpec d fuel := by simpa using stopped
        exact fixed_guards d closed fuel fixed good
      · contradiction
  | succ remaining ih =>
      intro fuel checked
      rw [guardLoop, CacheCorrectness.flowTableStep_spec] at checked
      simp only [Bool.and_eq_true] at checked
      obtain ⟨good, rest⟩ := checked
      split at rest
      · rename_i stopped
        have fixed : CacheCorrectness.flowTableSpec d (fuel + 1) =
            CacheCorrectness.flowTableSpec d fuel := by simpa using stopped
        exact fixed_guards d closed fuel fixed good
      · intro extra
        cases extra with
        | zero => simpa using good
        | succ extra =>
            have later := ih (fuel + 1) rest extra
            simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using later



theorem GuardFaithfulB_sound (d : Graph) (checked : GuardFaithfulB d = true) :
    GuardFaithful d := by
  simp only [GuardFaithfulB, Bool.and_eq_true] at checked
  refine ⟨checked.1, ?_⟩
  have initial : flowTableInitial d = CacheCorrectness.flowTableSpec d 0 := rfl
  rw [initial] at checked
  intro fuel
  simpa using guardLoop_sound d checked.1 (stdFuel d) 0 checked.2 fuel

theorem localGuardFaithfulB_sound (c : Certificate)
    (checked : localGuardFaithfulB c = true) : LocalGuardCompatibility c := by
  intro v hv ancestors edge hedge finish route hroute tail consumed
  unfold localGuardFaithfulB at checked
  have atNode := (List.all_eq_true.mp checked) v hv
  have nonempty : (eligibleAncestorsAtRoot c.graph v).isEmpty = false := by
    cases paths : eligibleAncestorsAtRoot c.graph v with
    | nil => exact False.elim (ancestors paths)
    | cons a rest => rfl
  simp only [nonempty, Bool.false_or] at atNode
  have atEdge := (List.all_eq_true.mp atNode) edge hedge
  have same : (edge.END == v) = true := by simpa using finish
  simp only [same, Bool.not_true, Bool.false_or] at atEdge
  have atRoute := (List.all_eq_true.mp atEdge) route hroute
  simpa [consumed] using atRoute





theorem local_guard_consumed_raw (d : Graph) (c : Certificate)
    (expanded : ExpandedCert d c) (checked : LocalChecks d c)
    (guard : localGuardFaithfulB c = true)
    (v : Vertex) (member : v ∈ d.NODES)
    (ancestors : eligibleAncestorsAtRoot d v ≠ [])
    (edge : Deduction) (hedge : edge ∈ d.EDGES) (finish : edge.END = v)
    (route : FlowPair) (hroute : route ∈ flowAt d (stdFuel d) edge.START)
    (tail : ColourPath) (consumed : consume edge route.2 = some tail) :
    depSimilarB route.1 edge.DEPENDENCY = true := by
  have graphEq := checked.1
  subst d
  have structural := (retained_invariants_of_local c.graph c checked).1
  have sourceMember := (structural.2.1 edge hedge).1
  have between : edge ∈ edgesBetween c.graph edge.START v := by
    simp [edgesBetween, hedge, finish]
  have continuable := routeContinuableAtB_of_consume_edgesBetween
    c.graph edge.START v edge route.2 tail between consumed
  have stored := (storedRoutes_setEq_flow c.graph c expanded checked
    edge.START sourceMember route).mpr (List.mem_filter.mpr ⟨hroute, continuable⟩)
  exact localGuardFaithfulB_sound c guard v member ancestors edge hedge finish
    route stored tail consumed

#print axioms GuardFaithfulB_sound
#print axioms localGuardFaithfulB_sound
#print axioms local_guard_consumed_raw

end Decoration
