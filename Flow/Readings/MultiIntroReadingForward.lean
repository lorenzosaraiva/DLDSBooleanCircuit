import Flow.Readings.MultiIntroReadingLineage
import Infrastructure.OccurrenceReadingsLineageModel
import Infrastructure.SpineReadingSlotQuotient

open Semantic FlowSpec SelfCertifying RouteIndexedMultiToken SpineReadings
open MultiIntroReadingLineage
namespace MultiIntroReadings




private theorem assignSlots_witness (v : Vertex) (tokens : List RouteIndexedToken)
    (token : RouteIndexedToken) (member : token ∈ tokens) (ordinal : Nat) :
    ∃ result, result ∈ assignSlotsFrom v ordinal tokens ∧
      result.flowPair = token.flowPair ∧ stripSlots result.routeKey = stripSlots token.routeKey := by
  induction tokens generalizing ordinal with
  | nil => simp at member
  | cons head rest ih =>
      rcases List.mem_cons.mp member with equal | below
      · subst token
        exact ⟨{ head with routeKey := .slot v ordinal head.routeKey },
          List.mem_cons_self, rfl, rfl⟩
      · obtain ⟨result, present, payload, lineage⟩ := ih below (ordinal + 1)
        exact ⟨result, List.mem_cons_of_mem _ present, payload, lineage⟩

private theorem realize_raw (variant : MultiIntro.Variant) (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v : Vertex) (tree : ReadingTree) (route : FlowPair)
    (token : RouteIndexedToken) (member : token ∈ MultiIntroReadingLineage.rawStepFrom variant d F v)
    (payload : token.flowPair = route)
    (lineage : stripSlots token.routeKey = stripSlots tree) :
    ∃ result, result ∈ MultiIntroReadingLineage.stepFrom variant d F v ∧
      result.flowPair = route ∧ stripSlots result.routeKey = stripSlots tree := by
  obtain ⟨result, present, pair, key⟩ := assignSlots_witness v _ token member 0
  exact ⟨result, present, pair.trans payload, key.trans lineage⟩

private theorem intro_raw (variant : MultiIntro.Variant) (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v u : Vertex) (e : Deduction) (token : RouteIndexedToken) (alpha beta : Formula)
    (tail : ColourPath) (source : sourceB d v = false)
    (pairs : (elimPairsAt d v).isEmpty = true) (choice : (MultiIntro.multiIntroShapeB d v || predsOf d v == [u]) = true)
    (formula : v.FORMULA = alpha >> beta) (premise : u.FORMULA = beta)
    (edge : e ∈ edgesBetween d u v) (member : token ∈ F u)
    (consumed : consume e token.residual = some tail)
    (tagged : SingleIntroductionReference.guard d token.dependency e = true) :
    mkToken d v (.introduction v e token.routeKey)
      (depRemove token.dependency alpha) tail ∈ MultiIntroReadingLineage.rawStepFrom variant d F v := by
  by_cases multi : MultiIntro.multiIntroShapeB d v = true
  · have edgeStart : e.START = u := by
      have parts := List.mem_filter.mp edge
      exact (of_decide_eq_true parts.2).1
    simp only [MultiIntroReadingLineage.rawStepFrom, multi, if_true,
      List.mem_flatMap, List.mem_append]
    refine ⟨e, edge_incoming d u v e edge, Or.inl ?_⟩
    simp only [ordinary, formula, edgeStart, premise, beq_self_eq_true,
      if_true, List.mem_filterMap]
    exact ⟨token, member, by simp [tagged, consumed]⟩
  · have absent : MultiIntro.multiIntroShapeB d v = false := Bool.eq_false_of_not_eq_true multi
    have preds : predsOf d v = [u] := by simpa [absent] using choice
    simp only [MultiIntroReadingLineage.rawStepFrom, absent, Bool.false_eq_true, if_false]

    simp only [sourceB] at source
    simp only [TaggedLineage.rawStepFrom, source, Bool.false_eq_true, if_false,
      pairs, if_true, List.mem_append]
    apply Or.inl
    simp only [TaggedLineage.introOutputs, preds, formula, premise, if_true,
      TaggedLineage.introOutputsFor, List.mem_flatMap, List.mem_filterMap]
    refine ⟨token, member, e, edge, ?_⟩
    simp [tagged, consumed]
private theorem mixed_raw (variant : MultiIntro.Variant) (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v u : Vertex) (e : Deduction) (token : RouteIndexedToken) (alpha beta : Formula)
    (tail : ColourPath) (source : sourceB d v = false)
    (pairs : (elimPairsAt d v).isEmpty = false) (pred : u ∈ predsOf d v)
    (formula : v.FORMULA = alpha >> beta) (premise : u.FORMULA = beta)
    (unused : elimPremiseAtB d v u = false)
    (edge : e ∈ edgesBetween d u v) (member : token ∈ F u)
    (consumed : consume e token.residual = some tail)
    (tagged : SingleIntroductionReference.guard d token.dependency e = true) :
    mkToken d v (.mixedIntroduction v e token.routeKey)
      (depRemove token.dependency alpha) tail ∈ MultiIntroReadingLineage.rawStepFrom variant d F v := by
  simp only [MultiIntroReadingLineage.rawStepFrom, not_multi_of_pairs d v pairs,
    Bool.false_eq_true, if_false]
  simp only [sourceB] at source
  simp only [TaggedLineage.rawStepFrom, source, Bool.false_eq_true, if_false,
    pairs, List.mem_append]
  apply Or.inl ∘ Or.inr
  simp only [TaggedLineage.mixedIntroOutputs, formula, List.mem_flatMap]
  refine ⟨u, pred, ?_⟩
  simp only [unused, premise, beq_self_eq_true, Bool.not_false,
    Bool.true_and, if_true, List.mem_flatMap, List.mem_filterMap]
  refine ⟨token, member, e, edge, ?_⟩
  simp [tagged, consumed]

private theorem elim_raw (variant : MultiIntro.Variant) (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v major minor : Vertex) (em eM : Deduction) (tm tM : RouteIndexedToken)
    (tail : ColourPath) (source : sourceB d v = false)
    (pair : (major, minor) ∈ elimPairsAt d v)
    (minorEdge : em ∈ edgesBetween d minor v) (majorEdge : eM ∈ edgesBetween d major v)
    (minorMember : tm ∈ F minor) (majorMember : tM ∈ F major)
    (minorConsume : consume em tm.residual = some tail)
    (majorConsume : consume eM tM.residual = some tail)
    (taggedm : SingleIntroductionReference.guard d tm.dependency em = true)
    (taggedM : SingleIntroductionReference.guard d tM.dependency eM = true) :
    mkToken d v (.elimination v major minor em eM tm.routeKey tM.routeKey)
      (depUnion tm.dependency tM.dependency) tail ∈ MultiIntroReadingLineage.rawStepFrom variant d F v := by
  have pairs : (elimPairsAt d v).isEmpty = false := by
    cases h : elimPairsAt d v with
    | nil => simp [h] at pair
    | cons head rest => rfl
  simp only [MultiIntroReadingLineage.rawStepFrom, not_multi_of_pairs d v pairs,
    Bool.false_eq_true, if_false]
  simp only [sourceB] at source
  simp only [TaggedLineage.rawStepFrom, source, Bool.false_eq_true, if_false,
    pairs, List.mem_append]
  apply Or.inl ∘ Or.inl
  simp only [TaggedLineage.elimOutputs, TaggedLineage.elimCombine, List.mem_flatMap, List.mem_filterMap]
  refine ⟨(major, minor), pair, tm, minorMember, tM, majorMember,
    em, minorEdge, eM, majorEdge, ?_⟩
  simp [taggedm, taggedM, minorConsume, majorConsume]

private theorem intro_reseed_raw (variant : MultiIntro.Variant) (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v u : Vertex) (e : Deduction) (a : Ancestral) (token : RouteIndexedToken)
    (alpha beta : Formula) (source : sourceB d v = false)
    (choice : (MultiIntro.multiIntroShapeB d v || predsOf d v == [u]) = true) (formula : v.FORMULA = alpha >> beta)
    (premise : u.FORMULA = beta) (edge : e ∈ edgesBetween d u v)
    (member : token ∈ F u) (tested : (if MultiIntro.multiIntroShapeB d v then reseedTest variant token.flowPair e
      else reseedZeroPremiseB token.flowPair e) = true)
    (ancestor : a ∈ eligibleAncestorsAtRoot d v)
    (tagged : SingleIntroductionReference.guard d token.dependency e = true) :
    mkToken d v (.introReseed v e a token.routeKey)
      (depRemove token.dependency alpha) a.COLOURS ∈ MultiIntroReadingLineage.rawStepFrom variant d F v := by
  by_cases multi : MultiIntro.multiIntroShapeB d v = true
  · have edgeStart : e.START = u := by
      have parts := List.mem_filter.mp edge
      exact (of_decide_eq_true parts.2).1
    have selected : reseedTest variant token.flowPair e = true := by simpa [multi] using tested
    simp only [MultiIntroReadingLineage.rawStepFrom, multi, if_true,
      List.mem_flatMap, List.mem_append]
    refine ⟨e, edge_incoming d u v e edge, Or.inr ?_⟩
    simp only [reseed, formula, edgeStart, premise, beq_self_eq_true,
      if_true, List.mem_flatMap]
    refine ⟨token, member, ?_⟩
    simp only [tagged, selected, Bool.true_and, if_true, List.mem_map]
    exact ⟨a, ancestor, rfl⟩
  · have absent : MultiIntro.multiIntroShapeB d v = false := Bool.eq_false_of_not_eq_true multi
    have preds : predsOf d v = [u] := by simpa [absent] using choice
    have zero : reseedZeroPremiseB token.flowPair e = true := by simpa [absent] using tested
    simp only [MultiIntroReadingLineage.rawStepFrom, absent, Bool.false_eq_true, if_false]

    have ancestors : (eligibleAncestorsAtRoot d v).isEmpty = false := by
      cases h : eligibleAncestorsAtRoot d v with
      | nil => simp [h] at ancestor
      | cons head rest => rfl
    simp only [sourceB] at source
    simp only [TaggedLineage.rawStepFrom, source, Bool.false_eq_true, if_false, List.mem_append]
    apply Or.inr
    simp only [TaggedLineage.reseedOutputs, ancestors, Bool.false_eq_true, if_false, List.mem_append]
    apply Or.inr
    simp only [TaggedLineage.introReseedOutputs, preds, formula, premise,
      beq_self_eq_true, if_true, List.mem_flatMap]
    refine ⟨token, member, e, edge, ?_⟩
    simp only [zero, tagged, Bool.true_and, if_true, List.mem_map]
    exact ⟨a, ancestor, rfl⟩
private theorem elim_reseed_raw (variant : MultiIntro.Variant) (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v major minor : Vertex) (em eM : Deduction) (a : Ancestral) (tm tM : RouteIndexedToken)
    (source : sourceB d v = false) (pair : (major, minor) ∈ elimPairsAt d v)
    (minorEdge : em ∈ edgesBetween d minor v) (majorEdge : eM ∈ edgesBetween d major v)
    (minorMember : tm ∈ F minor) (majorMember : tM ∈ F major)
    (minorAgrees : reseedPremiseAgreesB tm.flowPair em = true)
    (majorAgrees : reseedPremiseAgreesB tM.flowPair eM = true)
    (ancestor : a ∈ eligibleAncestorsAtRoot d v)
    (taggedm : SingleIntroductionReference.guard d tm.dependency em = true)
    (taggedM : SingleIntroductionReference.guard d tM.dependency eM = true) :
    mkToken d v (.elimReseed v major minor em eM a tm.routeKey tM.routeKey)
      (depUnion tm.dependency tM.dependency) a.COLOURS ∈ MultiIntroReadingLineage.rawStepFrom variant d F v := by
  have pairs : (elimPairsAt d v).isEmpty = false := by
    cases h : elimPairsAt d v with
    | nil => simp [h] at pair
    | cons head rest => rfl
  have ancestors : (eligibleAncestorsAtRoot d v).isEmpty = false := by
    cases h : eligibleAncestorsAtRoot d v with
    | nil => simp [h] at ancestor
    | cons head rest => rfl
  simp only [MultiIntroReadingLineage.rawStepFrom, not_multi_of_pairs d v pairs,
    Bool.false_eq_true, if_false]
  simp only [sourceB] at source
  simp only [TaggedLineage.rawStepFrom, source, Bool.false_eq_true, if_false, List.mem_append]
  apply Or.inr
  simp only [TaggedLineage.reseedOutputs, ancestors, Bool.false_eq_true, if_false, List.mem_append]
  apply Or.inl
  simp only [TaggedLineage.elimReseedOutputs, List.mem_flatMap]
  refine ⟨(major, minor), pair, tm, minorMember, tM, majorMember,
    em, minorEdge, eM, majorEdge, ?_⟩
  simp only [minorAgrees, majorAgrees, taggedm, taggedM, Bool.true_and, if_true, List.mem_map]
  exact ⟨a, ancestor, rfl⟩





theorem payload_eventually_keyed (variant : MultiIntro.Variant) (d : Graph) (tree : ReadingTree)
    (route : FlowPair) (checked : payload? variant d tree = some route) :
    ∃ threshold, ∀ fuel, threshold ≤ fuel →
      ∃ token, token ∈ MultiIntroReadingLineage.evaluateAt variant d fuel (vertex tree) ∧
        token.flowPair = route ∧ stripSlots token.routeKey = stripSlots tree := by
  induction tree generalizing route with
  | source v seed =>
      simp only [payload?] at checked
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
              intro fuel enough
              cases fuel with
              | zero => omega
              | succ fuel =>
                  apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _
                    (mkToken d v (.source v none) (normalizeDep [v.FORMULA]) [])
                  · simp [MultiIntroReadingLineage.rawStepFrom,
                    not_multi_of_source d v (by simpa [sourceB] using source),
                    TaggedLineage.rawStepFrom, source, sourceOutputs, ancestors]
                  · simp [mkToken, RouteIndexedToken.flowPair, normalizeDep_singleton]
                  · rfl
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
              intro fuel enough
              cases fuel with
              | zero => omega
              | succ fuel =>
                  apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _
                    (mkToken d v (.source v (some a)) (normalizeDep [v.FORMULA]) a.COLOURS)
                  · simp only [MultiIntroReadingLineage.rawStepFrom,
                      not_multi_of_source d v (by simpa [sourceB] using source), Bool.false_eq_true, if_false,
                      TaggedLineage.rawStepFrom, source, if_true, sourceOutputs,
                      ancestors, Bool.false_eq_true, if_false, List.mem_map]
                    exact ⟨a, ancestor, rfl⟩
                  · simp [mkToken, RouteIndexedToken.flowPair, normalizeDep_singleton]
                  · rfl
            · contradiction
      · contradiction
  | certificate v ordinal => simp [payload?] at checked
  | repetition v e parent ih => simp [payload?] at checked
  | slot v ordinal parent ih =>
      simp only [payload?] at checked
      split at checked
      · rename_i same
        have vertexEq : v = vertex parent := by simpa using same
        simpa only [vertex, vertexEq, stripSlots] using ih route checked
      · contradiction
  | introduction v e parent ih =>
      cases child : payload? variant d parent with
      | none => simp [payload?, child] at checked
      | some p =>
          obtain ⟨threshold, eventual⟩ := ih p child
          cases formula : v.FORMULA with
          | atom symbol => simp [payload?, child, formula] at checked
          | implication alpha beta =>
              simp only [payload?, child, formula, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨guard, tagged⟩ := guard
                obtain ⟨⟨⟨⟨source, pairs⟩, preds⟩, premise⟩, edge⟩ := guard
                have edge' : e ∈ edgesBetween d (vertex parent) v := by simpa [edgeB] using edge
                cases consumed : consume e p.2 with
                | none => simp [consumed] at checked
                | some tail =>
                    simp only [consumed, Option.bind_some] at checked
                    injection checked with checked
                    subst route
                    refine ⟨threshold + 1, ?_⟩
                    intro fuel enough
                    cases fuel with
                    | zero => omega
                    | succ fuel =>
                        obtain ⟨token, member, payload, lineage⟩ := eventual fuel (by omega)
                        have dep : token.dependency = p.1 := congrArg Prod.fst payload
                        have residual : token.residual = p.2 := congrArg Prod.snd payload
                        apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _ _
                          (intro_raw variant d _ v (vertex parent) e token alpha beta tail
                            source pairs preds formula premise edge' member (by simpa [residual] using consumed)
                            (by simpa [dep] using tagged))
                        · simp [mkToken, RouteIndexedToken.flowPair, dep]
                        · simp [mkToken, stripSlots, lineage]
              · contradiction
  | mixedIntroduction v e parent ih =>
      cases child : payload? variant d parent with
      | none => simp [payload?, child] at checked
      | some p =>
          obtain ⟨threshold, eventual⟩ := ih p child
          cases formula : v.FORMULA with
          | atom symbol => simp [payload?, child, formula] at checked
          | implication alpha beta =>
              simp only [payload?, child, formula, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨guard, tagged⟩ := guard
                obtain ⟨⟨⟨⟨⟨source, pairs⟩, pred⟩, unused⟩, premise⟩, edge⟩ := guard
                have pred' : vertex parent ∈ predsOf d v := by simpa using pred
                have edge' : e ∈ edgesBetween d (vertex parent) v := by simpa [edgeB] using edge
                cases consumed : consume e p.2 with
                | none => simp [consumed] at checked
                | some tail =>
                    simp only [consumed, Option.bind_some] at checked
                    injection checked with checked
                    subst route
                    refine ⟨threshold + 1, ?_⟩
                    intro fuel enough
                    cases fuel with
                    | zero => omega
                    | succ fuel =>
                        obtain ⟨token, member, payload, lineage⟩ := eventual fuel (by omega)
                        have dep : token.dependency = p.1 := congrArg Prod.fst payload
                        have residual : token.residual = p.2 := congrArg Prod.snd payload
                        apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _ _
                          (mixed_raw variant d _ v (vertex parent) e token alpha beta tail
                            source pairs pred' formula premise unused edge' member
                            (by simpa [residual] using consumed) (by simpa [dep] using tagged))
                        · simp [mkToken, RouteIndexedToken.flowPair, dep]
                        · simp [mkToken, stripSlots, lineage]
              · contradiction
  | elimination v major minor em eM km kM ihm ihM =>
      cases childm : payload? variant d km with
      | none => simp [payload?, childm] at checked
      | some pm =>
          cases childM : payload? variant d kM with
          | none => simp [payload?, childm, childM] at checked
          | some pM =>
              simp only [payload?, childm, childM, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨guard, taggedm⟩, taggedM⟩ := guard
                obtain ⟨⟨⟨⟨⟨source, minorAt⟩, majorAt⟩, pair⟩, edgeMinor⟩, edgeMajor⟩ := guard
                have pair' : (major, minor) ∈ elimPairsAt d v := by simpa using pair
                have em' : em ∈ edgesBetween d minor v := by simpa [edgeB] using edgeMinor
                have eM' : eM ∈ edgesBetween d major v := by simpa [edgeB] using edgeMajor
                obtain ⟨fm, eventualm⟩ := ihm pm childm
                obtain ⟨fM, eventualM⟩ := ihM pM childM
                cases cm : consume em pm.2 with
                | none => simp [cm] at checked
                | some tm =>
                    cases cM : consume eM pM.2 with
                    | none => simp [cm, cM] at checked
                    | some tM =>
                        simp only [cm, cM, Option.bind_some] at checked
                        split at checked
                        · rename_i tails
                          have tailEq : tm = tM := by simpa using tails
                          subst tM
                          simp only [Option.some.injEq] at checked
                          subst route
                          refine ⟨fm + fM + 1, ?_⟩
                          intro fuel enough
                          cases fuel with
                          | zero => omega
                          | succ fuel =>
                              obtain ⟨tokenm, memberm, payloadm, lineagem⟩ := eventualm fuel (by omega)
                              obtain ⟨tokenM, memberM, payloadM, lineageM⟩ := eventualM fuel (by omega)
                              rw [minorAt] at memberm
                              rw [majorAt] at memberM
                              have depm : tokenm.dependency = pm.1 := congrArg Prod.fst payloadm
                              have depM : tokenM.dependency = pM.1 := congrArg Prod.fst payloadM
                              have resm : tokenm.residual = pm.2 := congrArg Prod.snd payloadm
                              have resM : tokenM.residual = pM.2 := congrArg Prod.snd payloadM
                              apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _ _
                                (elim_raw variant d _ v major minor em eM tokenm tokenM tm source pair'
                                  em' eM' memberm memberM (by simpa [resm] using cm) (by simpa [resM] using cM)
                                  (by simpa [depm] using taggedm) (by simpa [depM] using taggedM))
                              · simp [mkToken, RouteIndexedToken.flowPair, depm, depM]
                              · simp [mkToken, stripSlots, lineagem, lineageM]
                        · contradiction
              · contradiction
  | introReseed v e a parent ih =>
      cases child : payload? variant d parent with
      | none => simp [payload?, child] at checked
      | some p =>
          obtain ⟨threshold, eventual⟩ := ih p child
          cases formula : v.FORMULA with
          | atom symbol => simp [payload?, child, formula] at checked
          | implication alpha beta =>
              simp only [payload?, child, formula, Option.bind_eq_bind, Option.bind_some] at checked
              generalize selectedValue : (if MultiIntro.multiIntroShapeB d v then
                reseedTest variant p e else reseedZeroPremiseB p e) = selected at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨guard, tagged⟩ := guard
                obtain ⟨⟨⟨⟨⟨source, preds⟩, premise⟩, edge⟩, zero⟩, ancestor⟩ := guard
                have zero : (if MultiIntro.multiIntroShapeB d v then
                    reseedTest variant p e else reseedZeroPremiseB p e) = true := selectedValue.trans zero
                have edge' : e ∈ edgesBetween d (vertex parent) v := by simpa [edgeB] using edge
                have ancestor' : a ∈ eligibleAncestorsAtRoot d v := by simpa using ancestor
                simp only [Option.some.injEq] at checked
                subst route
                refine ⟨threshold + 1, ?_⟩
                intro fuel enough
                cases fuel with
                | zero => omega
                | succ fuel =>
                    obtain ⟨token, member, payload, lineage⟩ := eventual fuel (by omega)
                    have dep : token.dependency = p.1 := congrArg Prod.fst payload
                    apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _ _
                      (intro_reseed_raw variant d _ v (vertex parent) e a token alpha beta source
                        preds formula premise edge' member (by simpa [payload] using zero) ancestor' (by simpa [dep] using tagged))
                    · simp [mkToken, RouteIndexedToken.flowPair, dep]
                    · simp [mkToken, stripSlots, lineage]
              · contradiction
  | elimReseed v major minor em eM a km kM ihm ihM =>
      cases childm : payload? variant d km with
      | none => simp [payload?, childm] at checked
      | some pm =>
          cases childM : payload? variant d kM with
          | none => simp [payload?, childm, childM] at checked
          | some pM =>
              simp only [payload?, childm, childM, Option.bind_eq_bind, Option.bind_some] at checked
              split at checked
              · rename_i guard
                simp only [Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq] at guard
                obtain ⟨⟨guard, taggedm⟩, taggedM⟩ := guard
                obtain ⟨⟨⟨⟨⟨⟨⟨⟨source, minorAt⟩, majorAt⟩, pair⟩, edgeMinor⟩,
                  edgeMajor⟩, minorAgrees⟩, majorAgrees⟩, ancestor⟩ := guard
                have pair' : (major, minor) ∈ elimPairsAt d v := by simpa using pair
                have em' : em ∈ edgesBetween d minor v := by simpa [edgeB] using edgeMinor
                have eM' : eM ∈ edgesBetween d major v := by simpa [edgeB] using edgeMajor
                have ancestor' : a ∈ eligibleAncestorsAtRoot d v := by simpa using ancestor
                obtain ⟨fm, eventualm⟩ := ihm pm childm
                obtain ⟨fM, eventualM⟩ := ihM pM childM
                simp only [Option.some.injEq] at checked
                subst route
                refine ⟨fm + fM + 1, ?_⟩
                intro fuel enough
                cases fuel with
                | zero => omega
                | succ fuel =>
                    obtain ⟨tokenm, memberm, payloadm, lineagem⟩ := eventualm fuel (by omega)
                    obtain ⟨tokenM, memberM, payloadM, lineageM⟩ := eventualM fuel (by omega)
                    rw [minorAt] at memberm
                    rw [majorAt] at memberM
                    have depm : tokenm.dependency = pm.1 := congrArg Prod.fst payloadm
                    have depM : tokenM.dependency = pM.1 := congrArg Prod.fst payloadM
                    apply realize_raw variant d (MultiIntroReadingLineage.evaluateAt variant d fuel) v _ _ _
                      (elim_reseed_raw variant d _ v major minor em eM a tokenm tokenM source pair'
                        em' eM' memberm memberM (by simpa [payloadm] using minorAgrees)
                        (by simpa [payloadM] using majorAgrees) ancestor'
                        (by simpa [depm] using taggedm) (by simpa [depM] using taggedM))
                    · simp [mkToken, RouteIndexedToken.flowPair, depm, depM]
                    · simp [mkToken, stripSlots, lineagem, lineageM]
              · contradiction


theorem payload_keyed_modulo_slots (variant : MultiIntro.Variant) (d : Graph) (tree : ReadingTree)
    (route : FlowPair) (checked : payload? variant d tree = some route) :
    ∃ fuel token, token ∈ MultiIntroReadingLineage.evaluateAt variant d fuel (vertex tree) ∧
      token.flowPair = route ∧ stripSlots token.routeKey = stripSlots tree := by
  obtain ⟨threshold, eventual⟩ := payload_eventually_keyed variant d tree route checked
  obtain ⟨token, member, payload, lineage⟩ := eventual threshold (Nat.le_refl _)
  exact ⟨threshold, token, member, payload, lineage⟩

#print axioms payload_eventually_keyed
#print axioms payload_keyed_modulo_slots
end MultiIntroReadings
