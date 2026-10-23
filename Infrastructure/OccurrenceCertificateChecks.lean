import Infrastructure.OccurrenceCertificateModel

open Semantic FlowSpec CertificateModel SelfCertifying

namespace OccurrenceCertificate




lemma productionAtOccurrenceB_setEq (c : Certificate) {v : Vertex}
    (h : productionAtOccurrenceB c v = true) :
    ListSetEq (StoredRoutes c v) (witnessableExpectedOccurrence c v) := by
  unfold productionAtOccurrenceB at h
  cases hs : localShape c.graph v <;> simp [hs] at h
  · exact setEqB_sound h
  · exact setEqB_sound h
  · exact setEqB_sound h
  · exact setEqB_sound h


lemma localProduction2_of_all (c : Certificate)
    (h : c.graph.NODES.all (productionAtOccurrenceB c) = true) : LocalProductionOccurrence c := by
  rw [List.all_eq_true] at h
  exact h

lemma localWitnessFaithful2_of_all (c : Certificate)
    (h : c.edgeRoutes.all (witnessFaithfulAtOccurrenceB c) = true) :
    LocalWitnessFaithfulOccurrence c := by
  rw [List.all_eq_true] at h
  exact h


lemma tailClosureAtOccurrenceB_sound (c : Certificate) {v : Vertex}
    (h : tailClosureAtOccurrenceB c v = true) :
    elimPairsAt c.graph v ≠ [] →
      (∀ u ∈ predsOf c.graph v,
        ElimPremiseAt c.graph v u ∨ LocalCarrierAccountedAtOccurrence c v u) ∧
      (∀ major minor, (major, minor) ∈ elimPairsAt c.graph v →
        certificatePairTailClosureAtB c v major minor = true) := by
  intro hpairsNe
  unfold tailClosureAtOccurrenceB at h
  cases hpairs : elimPairsAt c.graph v with
  | nil => exact (hpairsNe hpairs).elim
  | cons pair pairs =>
    rw [hpairs] at h
    simp only [Bool.and_eq_true, List.all_eq_true] at h
    refine ⟨?_, ?_⟩
    · intro u hu
      have hu' := h.1 u hu
      rw [Bool.or_eq_true] at hu'
      rcases hu' with helim | hcarrier
      · exact Or.inl (elimPremiseAtB_sound c.graph v u helim)
      · right
        unfold certificateCarrierAccountedAtOccurrenceB at hcarrier
        cases hformula : v.FORMULA with
        | atom name => simp [hformula] at hcarrier
        | implication alpha beta =>
            simp only [hformula, Bool.and_eq_true, List.all_eq_true] at hcarrier
            have huFormula : u.FORMULA = beta := by simpa using hcarrier.1
            refine ⟨alpha, by simp [hformula, huFormula], ?_⟩
            intro tail htail
            have hresult := hcarrier.2 tail htail
            rw [List.any_eq_true] at hresult
            obtain ⟨result, hresultMem, hresidual⟩ := hresult
            rcases result with ⟨dependency, residual⟩
            have hresidual' : residual = tail := by simpa using hresidual
            subst tail
            exact ⟨dependency, hresultMem⟩
    · intro major minor hpair
      apply h.2 (major, minor)
      simpa [hpairs] using hpair

lemma rootDischargeAtOccurrenceB_sound (c : Certificate) {v : Vertex}
    (h : rootDischargeAtOccurrenceB c v = true)
    (hterminal : (get_rule.outgoing v c.graph).isEmpty = true) :
    (∃ output, output ∈ c.terminalOutputs ∧ output.node = v ∧
      output.dependency = [] ∧ output.residual = []) ∧
      RuleShapeOK v c.graph ∧ outDep v c.graph = [] := by
  unfold rootDischargeAtOccurrenceB at h
  simp only [hterminal, if_true] at h
  rw [List.any_eq_true] at h
  obtain ⟨output, houtput, hchecks⟩ := h
  rw [List.mem_filter] at houtput
  obtain ⟨houtput, hnode⟩ := houtput
  simp only [Bool.and_eq_true] at hchecks
  refine ⟨⟨output, houtput, by simpa using hnode, ?_, ?_⟩, ?_, ?_⟩
  · exact List.isEmpty_iff.mp hchecks.1.1.1.1
  · exact List.isEmpty_iff.mp hchecks.1.1.1.2
  · exact hchecks.1.2
  · exact of_decide_eq_true hchecks.2


theorem localChecksOccurrenceB_sound (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (h : localChecksOccurrenceB d c = true) :
    LocalChecksOccurrence d c := by
  have _expandedDomain : ExpandedCert d c := hexp
  simp only [localChecksOccurrenceB, Bool.and_eq_true] at h
  rcases h with ⟨h, hhyp⟩
  rcases h with ⟨h, hlevelFormula⟩
  rcases h with ⟨h, hfaithful⟩
  rcases h with ⟨h, hone⟩
  rcases h with ⟨h, hnodup⟩
  rcases h with ⟨h, hstruct⟩
  rcases h with ⟨h, hroot⟩
  rcases h with ⟨h, htail⟩
  rcases h with ⟨h, hcoverage⟩
  rcases h with ⟨hgraph, hproduction⟩
  have hgraph' : c.graph = d := of_decide_eq_true hgraph
  have hproduction' : LocalProductionOccurrence c := by
    intro v hv
    rw [hgraph'] at hv
    rw [List.all_eq_true] at hproduction
    exact hproduction v hv
  have hcoverage' : LocalCoverage c := by
    rw [List.all_eq_true] at hcoverage
    intro v hv
    exact coverageAtB_sound c
      (hcoverage v (by simpa [hgraph'] using hv))
  have htail' : LocalTailClosureOccurrence c := by
    rw [List.all_eq_true] at htail
    intro v hv
    exact tailClosureAtOccurrenceB_sound c (htail v (by simpa [hgraph'] using hv))
  have hroot' : LocalRootDischarge c := by
    rw [List.all_eq_true] at hroot
    intro v hv hterminal
    have hcheck := hroot v (by simpa [hgraph'] using hv)
    exact rootDischargeAtOccurrenceB_sound c hcheck hterminal
  have hfaithful' : LocalWitnessFaithfulOccurrence c :=
    localWitnessFaithful2_of_all c hfaithful
  exact ⟨hgraph', hproduction', hcoverage', htail', hroot',
    structuralValidB_sound d hstruct, nodupB_sound d hnodup,
    oneEdgeB_sound d hone, hfaithful',
    levelFormulaUniqueB_sound d hlevelFormula,
    hypNoIncomingB_sound d hhyp⟩


theorem universalLocalRootDischargeAtOccurrenceB_iff
    (c : Certificate) (vertex : Vertex) :
    universalLocalRootDischargeAtOccurrenceB c vertex = true ↔
      UniversalLocalRootDischargeAtOccurrence c vertex := by
  by_cases root : (get_rule.outgoing vertex c.graph).isEmpty = true
  · rw [show universalLocalRootDischargeAtOccurrenceB c vertex =
        ((StoredRoutes c vertex).all rootRouteEmptyB &&
          (localExpectedOccurrence c vertex).all fun route => route.2.isEmpty) by
      simp [universalLocalRootDischargeAtOccurrenceB, root]]
    simp only [Bool.and_eq_true, List.all_eq_true]
    constructor
    · rintro ⟨stored, expected⟩
      intro _
      constructor
      · intro route member
        exact (rootRouteEmptyB_eq_true_iff route).mp (stored route member)
      · intro route member
        exact List.isEmpty_iff.mp (expected route member)
    · intro hlocal
      obtain ⟨stored, expected⟩ := hlocal root
      constructor
      · intro route member
        exact (rootRouteEmptyB_eq_true_iff route).mpr (stored route member)
      · intro route member
        exact List.isEmpty_iff.mpr (expected route member)
  · simp [universalLocalRootDischargeAtOccurrenceB,
      UniversalLocalRootDischargeAtOccurrence, root]

theorem universalLocalRootDischargeOccurrenceB_iff (c : Certificate) :
    universalLocalRootDischargeOccurrenceB c = true ↔
      UniversalLocalRootDischargeOccurrence c := by
  unfold universalLocalRootDischargeOccurrenceB UniversalLocalRootDischargeOccurrence
  rw [List.all_eq_true]
  constructor
  · intro checks vertex member
    exact (universalLocalRootDischargeAtOccurrenceB_iff c vertex).mp
      (checks vertex member)
  · intro hlocal vertex member
    exact (universalLocalRootDischargeAtOccurrenceB_iff c vertex).mpr
      (hlocal vertex member)


theorem strongFormationValidOccurrenceB_sound (d : Graph) (c : OccurrenceCertificate)
    (checked : strongFormationValidOccurrenceB d c = true) : StrongFormationValidOccurrence d c := by
  simp only [strongFormationValidOccurrenceB, Bool.and_eq_true] at checked
  have expanded := expandedCertB_sound d c checked.1.1
  exact ⟨expanded, localChecksOccurrenceB_sound d c expanded checked.1.2,
    (universalLocalRootDischargeOccurrenceB_iff c).mp checked.2⟩

#print axioms productionAtOccurrenceB_setEq
#print axioms localChecksOccurrenceB_sound
#print axioms universalLocalRootDischargeOccurrenceB_iff
#print axioms strongFormationValidOccurrenceB_sound

end OccurrenceCertificate
