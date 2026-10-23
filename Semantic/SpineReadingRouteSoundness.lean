import Semantic.SpineReadingModel
import Semantic.SpineNDSoundness

open Semantic FlowSpec SelfCertifying RouteIndexedMultiToken

namespace SpineReadings






private theorem intro_step (d : Graph) (F : Vertex → List FlowPair)
    (v u : Vertex) (e : Deduction) (p : FlowPair) (alpha beta : Formula)
    (tail : ColourPath) (source : sourceB d v = false)
    (pairs : (elimPairsAt d v).isEmpty = true) (preds : predsOf d v = [u])
    (formula : v.FORMULA = alpha >> beta) (premise : u.FORMULA = beta)
    (edge : e ∈ edgesBetween d u v) (member : p ∈ F u)
    (consumed : consume e p.2 = some tail) :
    (depRemove p.1 alpha, tail) ∈
      LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll F v := by
  simp only [sourceB] at source
  simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false,
    pairs, if_true, List.mem_eraseDups, List.mem_append]
  apply Or.inl
  simp only [preds, formula, premise, if_true, List.mem_flatMap, List.mem_filterMap]
  refine ⟨p, member, e, edge, ?_⟩
  simp [LambdaFlow.allowAll, consumed]

private theorem mixed_step (d : Graph) (F : Vertex → List FlowPair)
    (v u : Vertex) (e : Deduction) (p : FlowPair) (alpha beta : Formula)
    (tail : ColourPath) (source : sourceB d v = false)
    (pairs : (elimPairsAt d v).isEmpty = false) (pred : u ∈ predsOf d v)
    (formula : v.FORMULA = alpha >> beta) (premise : u.FORMULA = beta)
    (unused : elimPremiseAtB d v u = false)
    (edge : e ∈ edgesBetween d u v) (member : p ∈ F u)
    (consumed : consume e p.2 = some tail) :
    (depRemove p.1 alpha, tail) ∈
      LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll F v := by
  simp only [sourceB] at source
  simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false,
    pairs, List.mem_eraseDups, List.mem_append]
  apply Or.inl ∘ Or.inr
  simp only [LambdaFlow.mixedWith, formula, List.mem_flatMap]
  refine ⟨u, pred, ?_⟩
  simp only [unused, premise, beq_self_eq_true, Bool.not_false,
    Bool.true_and, if_true, List.mem_flatMap, List.mem_filterMap]
  refine ⟨p, member, e, edge, ?_⟩
  simp [LambdaFlow.allowAll, consumed]

private theorem elim_step (d : Graph) (F : Vertex → List FlowPair)
    (v major minor : Vertex) (em eM : Deduction) (pm pM : FlowPair)
    (tail : ColourPath) (source : sourceB d v = false)
    (pair : (major, minor) ∈ elimPairsAt d v)
    (minorEdge : em ∈ edgesBetween d minor v)
    (majorEdge : eM ∈ edgesBetween d major v)
    (minorMember : pm ∈ F minor) (majorMember : pM ∈ F major)
    (minorConsume : consume em pm.2 = some tail)
    (majorConsume : consume eM pM.2 = some tail) :
    (depUnion pm.1 pM.1, tail) ∈
      LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll F v := by
  have pairs : (elimPairsAt d v).isEmpty = false := by
    cases h : elimPairsAt d v with
    | nil => simp [h] at pair
    | cons head rest => rfl
  simp only [sourceB] at source
  simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false,
    pairs, List.mem_eraseDups, List.mem_append]
  apply Or.inl ∘ Or.inl
  simp only [List.mem_flatMap, List.mem_filterMap]
  refine ⟨(major, minor), pair, pm, minorMember, pM, majorMember,
    em, minorEdge, eM, majorEdge, ?_⟩
  simp [LambdaFlow.allowAll, minorConsume, majorConsume]

private theorem intro_reseed_step (d : Graph) (F : Vertex → List FlowPair)
    (v u : Vertex) (e : Deduction) (a : Ancestral) (p : FlowPair)
    (alpha beta : Formula) (source : sourceB d v = false)
    (preds : predsOf d v = [u]) (formula : v.FORMULA = alpha >> beta)
    (premise : u.FORMULA = beta) (edge : e ∈ edgesBetween d u v)
    (member : p ∈ F u) (zero : reseedZeroPremiseB p e = true)
    (ancestor : a ∈ eligibleAncestorsAtRoot d v) :
    (depRemove p.1 alpha, a.COLOURS) ∈
      LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll F v := by
  have ancestors : (eligibleAncestorsAtRoot d v).isEmpty = false := by
    cases h : eligibleAncestorsAtRoot d v with
    | nil => simp [h] at ancestor
    | cons head rest => rfl
  simp only [sourceB] at source
  simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false,
    List.mem_eraseDups, List.mem_append]
  apply Or.inr
  simp only [LambdaFlow.reseedWith, ancestors, Bool.false_eq_true, if_false,
    List.mem_eraseDups, List.mem_append]
  apply Or.inr
  simp only [LambdaFlow.introReseedWith, preds, formula, premise,
    beq_self_eq_true, if_true, List.mem_flatMap]
  refine ⟨p, member, e, edge, ?_⟩
  simp only [zero, LambdaFlow.allowAll, Bool.true_and, if_true, List.mem_map]
  exact ⟨a, ancestor, rfl⟩

private theorem elim_reseed_step (d : Graph) (F : Vertex → List FlowPair)
    (v major minor : Vertex) (em eM : Deduction) (a : Ancestral) (pm pM : FlowPair)
    (source : sourceB d v = false) (pair : (major, minor) ∈ elimPairsAt d v)
    (minorEdge : em ∈ edgesBetween d minor v) (majorEdge : eM ∈ edgesBetween d major v)
    (minorMember : pm ∈ F minor) (majorMember : pM ∈ F major)
    (minorAgrees : reseedPremiseAgreesB pm em = true)
    (majorAgrees : reseedPremiseAgreesB pM eM = true)
    (ancestor : a ∈ eligibleAncestorsAtRoot d v) :
    (depUnion pm.1 pM.1, a.COLOURS) ∈
      LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll F v := by
  have ancestors : (eligibleAncestorsAtRoot d v).isEmpty = false := by
    cases h : eligibleAncestorsAtRoot d v with
    | nil => simp [h] at ancestor
    | cons head rest => rfl
  simp only [sourceB] at source
  simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false,
    List.mem_eraseDups, List.mem_append]
  apply Or.inr
  simp only [LambdaFlow.reseedWith, ancestors, Bool.false_eq_true, if_false,
    List.mem_eraseDups, List.mem_append]
  apply Or.inl
  simp only [LambdaFlow.elimReseedWith, List.mem_flatMap]
  refine ⟨(major, minor), pair, pm, minorMember, pM, majorMember,
    em, minorEdge, eM, majorEdge, ?_⟩
  simp only [minorAgrees, majorAgrees, LambdaFlow.allowAll,
    Bool.true_and, if_true, List.mem_map]
  exact ⟨a, ancestor, rfl⟩




theorem readingPayload_mem_guardFree (d : Graph) (tree : ReadingTree)
    (route : FlowPair) (checked : readingPayload? d tree = some route) :
    ∃ fuel, route ∈ LambdaFlow.guardFreeFlowAt d fuel (vertex tree) := by
  induction tree generalizing route with
  | source v seed =>
      simp only [readingPayload?] at checked
      split at checked
      · rename_i source
        simp only [sourceB] at source
        cases seed with
        | none =>
            dsimp only at checked
            split at checked
            · rename_i ancestors
              simp only [Option.some.injEq] at checked
              subst route
              refine ⟨1, ?_⟩
              simp [vertex, LambdaFlow.evaluateAt, LambdaFlow.stepWith, source,
                ancestors, normalizeDep_singleton]
            · contradiction
        | some a =>
            dsimp only at checked
            split at checked
            · rename_i hasAncestor
              have ancestor : a ∈ ancestorsInto d v := by simpa using hasAncestor
              have ancestors : (ancestorsInto d v).isEmpty = false := by
                cases h : ancestorsInto d v with
                | nil => simp [h] at ancestor
                | cons head rest => rfl
              simp only [Option.some.injEq] at checked
              subst route
              refine ⟨1, ?_⟩
              simp only [vertex, LambdaFlow.evaluateAt, LambdaFlow.stepWith,
                source, if_true, ancestors, Bool.false_eq_true, if_false,
                normalizeDep_singleton, List.mem_map]
              exact ⟨a, ancestor, rfl⟩
            · contradiction
      · contradiction
  | certificate v ordinal => simp [readingPayload?] at checked
  | repetition v e parent ih => simp [readingPayload?] at checked
  | slot v ordinal parent ih =>
      simp only [readingPayload?] at checked
      split at checked
      · rename_i same
        have vertexEq : v = vertex parent := by simpa using same
        simpa only [vertex, vertexEq] using ih route checked
      · contradiction
  | introduction v e parent ih =>
      cases child : readingPayload? d parent with
      | none => simp [readingPayload?, child] at checked
      | some p =>
          obtain ⟨fuel, member⟩ := ih p child
          cases formula : v.FORMULA with
          | atom symbol => simp [readingPayload?, child, formula] at checked
          | implication alpha beta =>
              simp only [readingPayload?, child, formula, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨⟨⟨source, pairs⟩, preds⟩, premise⟩, edge⟩ := guard
                have edge' : e ∈ edgesBetween d (vertex parent) v := by
                  simpa [edgeB] using edge
                cases consumed : consume e p.2 with
                | none => simp [consumed] at checked
                | some tail =>
                    simp only [consumed, Option.bind_eq_bind, Option.bind_some] at checked
                    injection checked with checked
                    subst route
                    exact ⟨fuel + 1, intro_step d _ v (vertex parent) e p alpha beta
                      tail source pairs preds formula premise edge' member consumed⟩
              · contradiction
  | mixedIntroduction v e parent ih =>
      cases child : readingPayload? d parent with
      | none => simp [readingPayload?, child] at checked
      | some p =>
          obtain ⟨fuel, member⟩ := ih p child
          cases formula : v.FORMULA with
          | atom symbol => simp [readingPayload?, child, formula] at checked
          | implication alpha beta =>
              simp only [readingPayload?, child, formula, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨⟨⟨⟨source, pairs⟩, pred⟩, unused⟩, premise⟩, edge⟩ := guard
                have pred' : vertex parent ∈ predsOf d v := by simpa using pred
                have edge' : e ∈ edgesBetween d (vertex parent) v := by
                  simpa [edgeB] using edge
                cases consumed : consume e p.2 with
                | none => simp [consumed] at checked
                | some tail =>
                    simp only [consumed, Option.bind_eq_bind, Option.bind_some] at checked
                    injection checked with checked
                    subst route
                    exact ⟨fuel + 1, mixed_step d _ v (vertex parent) e p alpha beta
                      tail source pairs pred' formula premise unused edge' member consumed⟩
              · contradiction
  | elimination v major minor em eM km kM ihm ihM =>
      cases childm : readingPayload? d km with
      | none => simp [readingPayload?, childm] at checked
      | some pm =>
          cases childM : readingPayload? d kM with
          | none => simp [readingPayload?, childm, childM] at checked
          | some pM =>
              simp only [readingPayload?, childm, childM, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨⟨⟨⟨source, minorAt⟩, majorAt⟩, pair⟩, edgeMinor⟩, edgeMajor⟩ := guard
                have pair' : (major, minor) ∈ elimPairsAt d v := by simpa using pair
                have em' : em ∈ edgesBetween d minor v := by simpa [edgeB] using edgeMinor
                have eM' : eM ∈ edgesBetween d major v := by simpa [edgeB] using edgeMajor
                obtain ⟨fm, hm⟩ := ihm pm childm
                obtain ⟨fM, hM⟩ := ihM pM childM
                have hm' := DependencyGuard.evaluateAt_mono_fuel d LambdaFlow.allowAll
                  LambdaFlow.allowAll fm fM (vertex km) hm
                have hM' := DependencyGuard.evaluateAt_mono_fuel d LambdaFlow.allowAll
                  LambdaFlow.allowAll fM fm (vertex kM) hM
                rw [minorAt] at hm'
                rw [majorAt, Nat.add_comm fM fm] at hM'
                cases cm : consume em pm.2 with
                | none => simp [cm] at checked
                | some tm =>
                    cases cM : consume eM pM.2 with
                    | none => simp [cm, cM] at checked
                    | some tM =>
                        simp only [cm, cM, Option.bind_eq_bind, Option.bind_some] at checked
                        split at checked
                        · rename_i tails
                          have tailEq : tm = tM := by simpa using tails
                          subst tM
                          simp only [Option.some.injEq] at checked
                          subst route
                          exact ⟨fm + fM + 1, elim_step d _ v major minor em eM pm pM
                            tm source pair' em' eM' hm' hM' cm cM⟩
                        · contradiction
              · contradiction
  | introReseed v e a parent ih =>
      cases child : readingPayload? d parent with
      | none => simp [readingPayload?, child] at checked
      | some p =>
          obtain ⟨fuel, member⟩ := ih p child
          cases formula : v.FORMULA with
          | atom symbol => simp [readingPayload?, child, formula] at checked
          | implication alpha beta =>
              simp only [readingPayload?, child, formula, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨⟨⟨⟨source, preds⟩, premise⟩, edge⟩, zero⟩, ancestor⟩ := guard
                have edge' : e ∈ edgesBetween d (vertex parent) v := by
                  simpa [edgeB] using edge
                have ancestor' : a ∈ eligibleAncestorsAtRoot d v := by simpa using ancestor
                simp only [Option.some.injEq] at checked
                subst route
                exact ⟨fuel + 1, intro_reseed_step d _ v (vertex parent) e a p alpha beta
                  source preds formula premise edge' member zero ancestor'⟩
              · contradiction
  | elimReseed v major minor em eM a km kM ihm ihM =>
      cases childm : readingPayload? d km with
      | none => simp [readingPayload?, childm] at checked
      | some pm =>
          cases childM : readingPayload? d kM with
          | none => simp [readingPayload?, childm, childM] at checked
          | some pM =>
              simp only [readingPayload?, childm, childM, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨⟨⟨⟨⟨⟨⟨source, minorAt⟩, majorAt⟩, pair⟩, edgeMinor⟩,
                  edgeMajor⟩, minorAgrees⟩, majorAgrees⟩, ancestor⟩ := guard
                have pair' : (major, minor) ∈ elimPairsAt d v := by simpa using pair
                have em' : em ∈ edgesBetween d minor v := by simpa [edgeB] using edgeMinor
                have eM' : eM ∈ edgesBetween d major v := by simpa [edgeB] using edgeMajor
                have ancestor' : a ∈ eligibleAncestorsAtRoot d v := by simpa using ancestor
                obtain ⟨fm, hm⟩ := ihm pm childm
                obtain ⟨fM, hM⟩ := ihM pM childM
                have hm' := DependencyGuard.evaluateAt_mono_fuel d LambdaFlow.allowAll
                  LambdaFlow.allowAll fm fM (vertex km) hm
                have hM' := DependencyGuard.evaluateAt_mono_fuel d LambdaFlow.allowAll
                  LambdaFlow.allowAll fM fm (vertex kM) hM
                rw [minorAt] at hm'
                rw [majorAt, Nat.add_comm fM fm] at hM'
                simp only [Option.some.injEq] at checked
                subst route
                exact ⟨fm + fM + 1, elim_reseed_step d _ v major minor em eM a pm pM
                  source pair' em' eM' hm' hM' minorAgrees majorAgrees ancestor'⟩
              · contradiction

#print axioms readingPayload_mem_guardFree
end SpineReadings
