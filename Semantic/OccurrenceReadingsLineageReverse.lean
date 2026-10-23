import Semantic.OccurrenceReadingsLineageModel
import Semantic.OccurrenceReadingsModel

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe RouteIndexedMultiToken SpineReadings OccurrenceReadings

namespace OccurrenceReadingsLineageReverse





private theorem source_checked (d : Graph) (v : Vertex)
    (source : sourceB d v = true) (token : RouteIndexedToken)
    (member : token ∈ sourceOutputs d v) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  simp only [sourceOutputs] at member
  split at member
  · rename_i ancestors
    simp only [List.mem_singleton] at member
    subst token
    simp [vertex, mkToken, readingPayloadComplete?, source, ancestors,
      RouteIndexedToken.flowPair, normalizeDep_singleton]
  · rename_i ancestors
    obtain ⟨ancestor, present, produced⟩ := List.mem_map.mp member
    subst token
    simp [vertex, mkToken, readingPayloadComplete?, source, present,
      RouteIndexedToken.flowPair, normalizeDep_singleton]

private theorem intro_checked (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v : Vertex) (source : sourceB d v = false)
    (pairs : (elimPairsAt d v).isEmpty = true)
    (previous : ∀ u token, token ∈ F u →
      vertex token.routeKey = u ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (token : RouteIndexedToken) (member : token ∈ TaggedLineage.introOutputs d v F) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  unfold TaggedLineage.introOutputs at member
  cases preds : predsOf d v with
  | nil => simp [preds] at member
  | cons u rest =>
      cases rest with
      | cons w rest => simp [preds] at member
      | nil =>
          cases formula : v.FORMULA with
          | atom s => simp [preds, formula] at member
          | implication alpha beta =>
              simp only [preds, formula] at member
              split at member
              · rename_i premise
                simp only [TaggedLineage.introOutputsFor, List.mem_flatMap, List.mem_filterMap] at member
                obtain ⟨child, childMem, e, edgeMem, produced⟩ := member
                split at produced <;> try contradiction
                rename_i crossed
                cases consumed : consume e child.residual with
                | none => simp [consumed] at produced
                | some tail =>
                    simp only [consumed, Option.map_some, Option.some.injEq] at produced
                    subst token
                    obtain ⟨childAt, childChecked⟩ := previous u child childMem
                    constructor
                    · rfl
                    · simp [mkToken, readingPayloadComplete?, childChecked, childAt, formula,
                        source, pairs, preds, premise, edgeB, edgeMem,
                        RouteIndexedToken.flowPair, consumed, crossed]
              · contradiction

private theorem mixed_checked (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v : Vertex) (source : sourceB d v = false)
    (pairs : (elimPairsAt d v).isEmpty = false)
    (previous : ∀ u token, token ∈ F u →
      vertex token.routeKey = u ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (token : RouteIndexedToken) (member : token ∈ TaggedLineage.mixedIntroOutputs d v F) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  unfold TaggedLineage.mixedIntroOutputs at member
  cases formula : v.FORMULA with
  | atom s => simp [formula] at member
  | implication alpha beta =>
      simp only [formula, List.mem_flatMap] at member
      obtain ⟨u, predecessor, outputs⟩ := member
      split at outputs
      · rename_i guard
        have parts : elimPremiseAtB d v u = false ∧ u.FORMULA = beta := by simpa using guard
        simp only [List.mem_flatMap, List.mem_filterMap] at outputs
        obtain ⟨child, childMem, e, edgeMem, produced⟩ := outputs
        split at produced <;> try contradiction
        rename_i crossed
        cases consumed : consume e child.residual with
        | none => simp [consumed] at produced
        | some tail =>
            simp only [consumed, Option.map_some, Option.some.injEq] at produced
            subst token
            obtain ⟨childAt, childChecked⟩ := previous u child childMem
            constructor
            · rfl
            · simp [mkToken, readingPayloadComplete?, childChecked, childAt, formula,
                source, pairs, predecessor, parts.1, parts.2, edgeB, edgeMem,
                RouteIndexedToken.flowPair, consumed, crossed]
      · contradiction

private theorem elim_checked (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v : Vertex) (source : sourceB d v = false)
    (previous : ∀ u token, token ∈ F u →
      vertex token.routeKey = u ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (token : RouteIndexedToken) (member : token ∈ TaggedLineage.elimOutputs d v F) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  simp only [TaggedLineage.elimOutputs, TaggedLineage.elimCombine,
    List.mem_flatMap, List.mem_filterMap] at member
  obtain ⟨⟨major, minor⟩, pair, childm, memm, childM, memM, em, edgeMnr, eM, edgeMjr, produced⟩ := member
  split at produced <;> try contradiction
  rename_i crossed
  have guards : SingleIntroductionReference.guard d childm.dependency em = true /\
      SingleIntroductionReference.guard d childM.dependency eM = true := by
    simpa only [Bool.and_eq_true] using crossed
  cases cm : consume em childm.residual with
  | none => simp [cm] at produced
  | some tm =>
      cases cM : consume eM childM.residual with
      | none => simp [cm, cM] at produced
      | some tM =>
          simp only [cm, cM] at produced
          split at produced
          · rename_i sameTail
            simp only [Option.some.injEq] at produced
            subst token
            obtain ⟨minorAt, minorChecked⟩ := previous minor childm memm
            obtain ⟨majorAt, majorChecked⟩ := previous major childM memM
            constructor
            · rfl
            · simp [mkToken, readingPayloadComplete?, minorChecked, majorChecked,
                minorAt, majorAt, source, pair, edgeB, edgeMnr, edgeMjr,
                RouteIndexedToken.flowPair, cm, cM, sameTail, guards.1, guards.2]
          · contradiction

private theorem intro_reseed_checked (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v : Vertex) (source : sourceB d v = false)
    (previous : ∀ u token, token ∈ F u →
      vertex token.routeKey = u ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (token : RouteIndexedToken)
    (member : token ∈ TaggedLineage.introReseedOutputs d v F (eligibleAncestorsAtRoot d v)) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  unfold TaggedLineage.introReseedOutputs at member
  cases preds : predsOf d v with
  | nil => simp [preds] at member
  | cons u rest =>
      cases rest with
      | cons w rest => simp [preds] at member
      | nil =>
          cases formula : v.FORMULA with
          | atom s => simp [preds, formula] at member
          | implication alpha beta =>
              simp only [preds, formula] at member
              split at member
              · rename_i premise
                have premiseEq : u.FORMULA = beta := by simpa using premise
                simp only [List.mem_flatMap] at member
                obtain ⟨child, childMem, e, edgeMem, produced⟩ := member
                split at produced
                · rename_i zero
                  obtain ⟨ancestor, ancestorMem, same⟩ := List.mem_map.mp produced
                  subst token
                  obtain ⟨childAt, childChecked⟩ := previous u child childMem
                  constructor
                  · rfl
                  · simpa [mkToken, readingPayloadComplete?, childChecked, childAt, formula,
                      source, preds, premiseEq, edgeB, edgeMem, ancestorMem,
                      RouteIndexedToken.flowPair] using zero
                · contradiction
              · contradiction

private theorem elim_reseed_checked (d : Graph) (F : Vertex → List RouteIndexedToken)
    (v : Vertex) (source : sourceB d v = false)
    (previous : ∀ u token, token ∈ F u →
      vertex token.routeKey = u ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (token : RouteIndexedToken)
    (member : token ∈ TaggedLineage.elimReseedOutputs d v F (eligibleAncestorsAtRoot d v)) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  simp only [TaggedLineage.elimReseedOutputs, List.mem_flatMap] at member
  obtain ⟨⟨major, minor⟩, pair, childm, memm, childM, memM, em, edgeMnr, eM, edgeMjr, produced⟩ := member
  split at produced
  · rename_i guard
    have parts : reseedPremiseAgreesB childm.flowPair em = true ∧
        reseedPremiseAgreesB childM.flowPair eM = true := by
      have all := guard
      simp only [Bool.and_eq_true] at all
      exact all.1.1
    obtain ⟨ancestor, ancestorMem, same⟩ := List.mem_map.mp produced
    subst token
    obtain ⟨minorAt, minorChecked⟩ := previous minor childm memm
    obtain ⟨majorAt, majorChecked⟩ := previous major childM memM
    constructor
    · rfl
    · simpa [mkToken, readingPayloadComplete?, minorChecked, majorChecked, minorAt, majorAt,
        source, pair, edgeB, edgeMnr, edgeMjr, ancestorMem,
        RouteIndexedToken.flowPair, Bool.and_assoc] using guard
  · contradiction

private theorem raw_checked (d : Graph) (F : Vertex → List RouteIndexedToken)
    (previous : ∀ u token, token ∈ F u →
      vertex token.routeKey = u ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (v : Vertex) (token : RouteIndexedToken)
    (member : token ∈ TaggedLineage.rawStepFrom d F v) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  simp only [TaggedLineage.rawStepFrom] at member
  split at member
  · rename_i source
    exact source_checked d v source token member
  · rename_i notSource
    have source : sourceB d v = false := by simpa [sourceB] using notSource
    rcases List.mem_append.mp member with core | reseed
    · split at core
      · rename_i pairs
        exact intro_checked d F v source pairs previous token core
      · rename_i nonempty
        have pairs : (elimPairsAt d v).isEmpty = false := by simpa using nonempty
        rcases List.mem_append.mp core with elimination | mixed
        · exact elim_checked d F v source previous token elimination
        · exact mixed_checked d F v source pairs previous token mixed
    · simp only [TaggedLineage.reseedOutputs] at reseed
      split at reseed
      · contradiction
      · rcases List.mem_append.mp reseed with elimination | introduction
        · exact elim_reseed_checked d F v source previous token elimination
        · exact intro_reseed_checked d F v source previous token introduction

private theorem slots_checked (d : Graph) (v : Vertex) (tokens : List RouteIndexedToken)
    (previous : ∀ token, token ∈ tokens →
      vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair)
    (ordinal : Nat) (token : RouteIndexedToken)
    (member : token ∈ assignSlotsFrom v ordinal tokens) :
    vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  induction tokens generalizing ordinal with
  | nil => simp [assignSlotsFrom] at member
  | cons head rest ih =>
      simp only [assignSlotsFrom, List.mem_cons] at member
      rcases member with rfl | below
      · have checked := previous head List.mem_cons_self
        constructor
        · rfl
        · simpa [readingPayloadComplete?, checked.1, RouteIndexedToken.flowPair] using checked.2
      · apply ih (fun child present => previous child (List.mem_cons_of_mem _ present))
          (ordinal + 1) below




theorem evaluateAt_payload (d : Graph) : ∀ fuel v token,
    token ∈ TaggedLineage.evaluateAt d fuel v →
      vertex token.routeKey = v ∧ readingPayloadComplete? d token.routeKey = some token.flowPair := by
  intro fuel
  induction fuel with
  | zero => intro v token member; simp [TaggedLineage.evaluateAt] at member
  | succ fuel ih =>
      intro v token member
      exact slots_checked d v (TaggedLineage.rawStepFrom d (TaggedLineage.evaluateAt d fuel) v)
        (fun child present => raw_checked d (TaggedLineage.evaluateAt d fuel) ih v child present)
        0 token member

#print axioms evaluateAt_payload
end OccurrenceReadingsLineageReverse
