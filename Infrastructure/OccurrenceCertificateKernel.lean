import Infrastructure.OccurrenceCertificateModel

open Semantic FlowSpec CertificateModel SelfCertifying

namespace OccurrenceCertificate





lemma elimReseed2_project_inputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) (ancestors : List Ancestral) :
    ListSetEq
      (LambdaFlow.elimReseedWith d LambdaFlow.allowAll v (fun u => WitnessableRoutes d u (F u)) ancestors)
      (LambdaFlow.elimReseedWith d LambdaFlow.allowAll v F ancestors) := by
  intro result
  unfold LambdaFlow.elimReseedWith WitnessableRoutes
  simp only [List.mem_flatMap, List.mem_filter]
  constructor
  · rintro ⟨pair, hpair, minorRoute, ⟨hminor, _⟩,
      majorRoute, ⟨hmajor, _⟩, hrest⟩
    exact ⟨pair, hpair, minorRoute, hminor, majorRoute, hmajor, hrest⟩
  · rintro ⟨pair, hpair, minorRoute, hminor, majorRoute, hmajor,
      minorEdge, hminorEdge, majorEdge, hmajorEdge, hlast⟩
    rcases pair with ⟨major, minor⟩
    split at hlast
    next hcondition =>
      simp only [Bool.and_eq_true] at hcondition
      refine ⟨(major, minor), hpair, minorRoute, ⟨hminor, ?_⟩,
        majorRoute, ⟨hmajor, ?_⟩, minorEdge, hminorEdge,
        majorEdge, hmajorEdge, ?_⟩
      · exact routeContinuable_of_reseedAgree d minor v minorRoute minorEdge
          hminorEdge hcondition.1.1.1
      · exact routeContinuable_of_reseedAgree d major v majorRoute majorEdge
          hmajorEdge hcondition.1.1.2
      · simpa [hcondition] using hlast
    next hcondition => simp at hlast

lemma introReseed2_project_inputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) (ancestors : List Ancestral) :
    ListSetEq
      (LambdaFlow.introReseedWith d LambdaFlow.allowAll v (fun u => WitnessableRoutes d u (F u)) ancestors)
      (LambdaFlow.introReseedWith d LambdaFlow.allowAll v F ancestors) := by
  unfold LambdaFlow.introReseedWith
  cases hp : predsOf d v with
  | nil => exact ListSetEq.refl _
  | cons premise rest =>
      cases rest with
      | cons premise' rest => exact ListSetEq.refl _
      | nil =>
        cases hformula : v.FORMULA with
        | atom => exact ListSetEq.refl _
        | implication alpha beta =>
          by_cases hbeta : (premise.FORMULA == beta) = true
          · simp only [hbeta, if_true]
            intro result
            unfold WitnessableRoutes
            simp only [List.mem_flatMap, List.mem_filter]
            constructor
            · rintro ⟨route, ⟨hroute, _⟩, hrest⟩
              exact ⟨route, hroute, hrest⟩
            · rintro ⟨route, hroute, edge, hedge, hlast⟩
              split at hlast
              next hcondition =>
                simp only [Bool.and_eq_true] at hcondition
                refine ⟨route, ⟨hroute, ?_⟩, edge, hedge, ?_⟩
                · unfold reseedZeroPremiseB at hcondition
                  simp only [Bool.and_eq_true] at hcondition
                  exact routeContinuable_of_reseedAgree d premise v route edge
                    hedge hcondition.1.2
                · simpa [hcondition] using hlast
              next hcondition => simp at hlast
          · have hbeta' : (premise.FORMULA == beta) = false :=
              Bool.eq_false_of_not_eq_true hbeta
            simp only [hbeta', Bool.false_eq_true, if_false]
            exact ListSetEq.refl []

lemma reseed2_project_inputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) :
    ListSetEq
      (LambdaFlow.reseedWith d LambdaFlow.allowAll v (fun u => WitnessableRoutes d u (F u)))
      (LambdaFlow.reseedWith d LambdaFlow.allowAll v F) := by
  unfold LambdaFlow.reseedWith
  generalize hanc : eligibleAncestorsAtRoot d v = ancestors
  cases hempty : ancestors.isEmpty with
  | true =>
      simp only [hempty, if_true]
      exact ListSetEq.refl []
  | false =>
      simp only [hempty, Bool.false_eq_true, if_false]
      apply eraseDups_congr
      intro result
      simp only [List.mem_append]
      exact or_congr
        (elimReseed2_project_inputs d v F ancestors result)
        (introReseed2_project_inputs d v F ancestors result)


theorem step2_project_inputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) :
    ListSetEq
      (LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll
        (fun u => WitnessableRoutes d u (F u)) v)
      (LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll F v) := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp [LambdaFlow.stepWith, source, ListSetEq.refl]
  · simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false]
    apply eraseDups_congr
    apply append_congr
    · by_cases pairs : (elimPairsAt d v).isEmpty = true
      · simp only [pairs, if_true]
        cases hp : predsOf d v with
        | nil => exact ListSetEq.refl _
        | cons u rest =>
          cases rest with
          | cons other rest => exact ListSetEq.refl _
          | nil =>
            cases hf : v.FORMULA with
            | atom => exact ListSetEq.refl _
            | implication alpha beta =>
              simp only []
              by_cases form : u.FORMULA = beta
              · simp only [form, if_true]
                exact introProjectInputsRaw d v u alpha (F u)
              · simp only [form, if_false]
                exact ListSetEq.refl _
      · simp only [pairs, Bool.false_eq_true, if_false]
        apply append_congr
        · apply flatMap_congr_fun
          intro pair
          rcases pair with ⟨major, minor⟩
          exact elimCombine_project_inputs d v major minor (F minor) (F major)
        · exact mixedIntroProjectInputs d v F
    · exact reseed2_project_inputs d v F

theorem localExpectedOccurrence_setEq_step (c : OccurrenceCertificate) (v : Vertex)
    (structural : structuralValid c.graph)
    (validShape : localShape c.graph v ≠ .invalid) :
    ListSetEq (localExpectedOccurrence c v)
      (LambdaFlow.stepWith c.graph LambdaFlow.allowAll LambdaFlow.allowAll
        (outputsAt c) v) := by
  cases hs : localShape c.graph v with
  | source =>
    have guard := localShape_source_inv hs
    simp [localExpectedOccurrence, hs, LambdaFlow.stepWith, guard, sourceExpected,
      ListSetEq]
  | elim major minor =>
    obtain ⟨guard, pairs⟩ := localShape_elim_raw_inv hs
    intro route
    simp [localExpectedOccurrence, hs, LambdaFlow.stepWith, guard, pairs,
      elimExpected, mixedIntroExpected, reseedExpectedOccurrence, LambdaFlow.mixedWith,
      mixedIntroPairs, elimCombine, LambdaFlow.allowAll, List.mem_eraseDups,
      List.mem_append, or_assoc]
    rfl
  | elimMulti pairs =>
    obtain ⟨guard, pairEq, pairNe⟩ := localShape_elimMulti_raw_inv hs
    have pairEmpty : pairs.isEmpty = false := by simpa using pairNe
    intro route
    simp [localExpectedOccurrence, hs, LambdaFlow.stepWith, guard, pairEq, pairEmpty,
      elimExpectedMany, mixedIntroExpected, reseedExpectedOccurrence, LambdaFlow.mixedWith,
      mixedIntroPairs, elimCombine, LambdaFlow.allowAll, List.mem_eraseDups,
      List.mem_append, or_assoc]
    rfl
  | intro premise alpha beta =>
    obtain ⟨guard, _, _, _⟩ := localShape_intro_raw_inv hs
    obtain ⟨preds, formula, premiseFormula, _, _, _, pairs⟩ :=
      localShape_intro_inv structural hs
    obtain ⟨hyp, _⟩ := guard_false_parts guard
    intro route
    simp [localExpectedOccurrence, hs, LambdaFlow.stepWith, hyp, pairs, preds,
      formula, premiseFormula, introExpected, reseedExpectedOccurrence, LambdaFlow.allowAll,
      List.mem_eraseDups, List.mem_append]
  | invalid => exact False.elim (validShape hs)

#print axioms elimReseed2_project_inputs
#print axioms introReseed2_project_inputs
#print axioms reseed2_project_inputs
#print axioms step2_project_inputs
#print axioms localExpectedOccurrence_setEq_step

end OccurrenceCertificate
