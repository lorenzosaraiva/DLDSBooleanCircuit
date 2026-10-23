import Infrastructure.HCSuccessorFlowSpecification
















open Semantic FlowSpec
open MixedRuleReplay OccurrenceAwareReplay
open HCSuccessorFlowSpecification

namespace HCSuccessorFixedGraphReference

abbrev RoutePredicate := FlowPair -> Prop



inductive RelIntroductionProduction (d : Graph) (v : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate) : FlowPair -> Prop
  | intro (antecedent consequent : Formula) (premise : Vertex)
      (dependency : Dep) (residual tail : ColourPath) (edge : Deduction)
      (formula : v.FORMULA = antecedent >> consequent)
      (premiseMember : premise ∈ predsOf d v)
      (premiseFormula : premise.FORMULA = consequent)
      (routeMember : premiseRoutes premise (dependency, residual))
      (edgeMember : edge ∈ edgesBetween d premise v)
      (consumed : consume edge residual = some tail) :
      RelIntroductionProduction d v premiseRoutes
        (depRemove dependency antecedent, tail)



inductive RelEliminationProduction (d : Graph) (v : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate) : FlowPair -> Prop
  | elim (major minor : Vertex)
      (minorDep majorDep : Dep)
      (minorResidual majorResidual tail : ColourPath)
      (minorEdge majorEdge : Deduction)
      (pairMember : (major, minor) ∈ elimPairsAt d v)
      (minorRoute : premiseRoutes minor (minorDep, minorResidual))
      (majorRoute : premiseRoutes major (majorDep, majorResidual))
      (minorEdgeMember : minorEdge ∈ edgesBetween d minor v)
      (majorEdgeMember : majorEdge ∈ edgesBetween d major v)
      (minorConsumed : consume minorEdge minorResidual = some tail)
      (majorConsumed : consume majorEdge majorResidual = some tail) :
      RelEliminationProduction d v premiseRoutes
        (depUnion minorDep majorDep, tail)




inductive LogicalProduction (d : Graph) (v : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate) : FlowPair -> Prop
  | elimination (route : FlowPair)
      (production : RelEliminationProduction d v premiseRoutes route) :
      LogicalProduction d v premiseRoutes route
  | introduction (route : FlowPair)
      (production : RelIntroductionProduction d v premiseRoutes route)
      (separated : ∀ eliminationRoute,
        RelEliminationProduction d v premiseRoutes eliminationRoute ->
          eliminationRoute.2 ≠ route.2) :
      LogicalProduction d v premiseRoutes route

theorem introductionProduction_iff_rel
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair) :
    IntroductionProduction d v premiseFlow route ↔
      RelIntroductionProduction d v
        (fun premise selection => selection ∈ premiseFlow premise) route := by
  constructor <;> intro production
  · cases production with
    | intro antecedent consequent premise dependency residual tail edge
        formula premiseMember premiseFormula routeMember edgeMember consumed =>
        exact .intro antecedent consequent premise dependency residual tail edge
          formula premiseMember premiseFormula routeMember edgeMember consumed
  · cases production with
    | intro antecedent consequent premise dependency residual tail edge
        formula premiseMember premiseFormula routeMember edgeMember consumed =>
        exact .intro antecedent consequent premise dependency residual tail edge
          formula premiseMember premiseFormula routeMember edgeMember consumed

theorem eliminationProduction_iff_rel
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair) :
    EliminationProduction d v premiseFlow route ↔
      RelEliminationProduction d v
        (fun premise selection => selection ∈ premiseFlow premise) route := by
  constructor <;> intro production
  · cases production with
    | elim major minor minorDep majorDep minorResidual majorResidual tail
        minorEdge majorEdge pairMember minorRoute majorRoute minorEdgeMember
        majorEdgeMember minorConsumed majorConsumed =>
        exact .elim major minor minorDep majorDep minorResidual majorResidual
          tail minorEdge majorEdge pairMember minorRoute majorRoute
          minorEdgeMember majorEdgeMember minorConsumed majorConsumed
  · cases production with
    | elim major minor minorDep majorDep minorResidual majorResidual tail
        minorEdge majorEdge pairMember minorRoute majorRoute minorEdgeMember
        majorEdgeMember minorConsumed majorConsumed =>
        exact .elim major minor minorDep majorDep minorResidual majorResidual
          tail minorEdge majorEdge pairMember minorRoute majorRoute
          minorEdgeMember majorEdgeMember minorConsumed majorConsumed

theorem residualOccursB_eq_false_iff
    (routes : List FlowPair) (residual : ColourPath) :
    residualOccursB routes residual = false ↔
      ∀ route, route ∈ routes -> route.2 ≠ residual := by
  simp [residualOccursB]


theorem derivedOutputs_iff_logicalProduction
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair) :
    route ∈ derivedOutputs d v premiseFlow ↔
      LogicalProduction d v
        (fun premise selection => selection ∈ premiseFlow premise) route := by
  rw [derivedOutputs, List.mem_eraseDups, List.mem_append]
  constructor
  · intro member
    rcases member with elimination | introduction
    · exact .elimination route
        ((eliminationProduction_iff_rel d v premiseFlow route).mp
          ((selectionEliminationOutputs_iff_eliminationProduction
            d v premiseFlow route).mp elimination))
    · have separated :=
        (mem_separatedIntroductionOutputs_iff d v premiseFlow route).mp
          introduction
      apply LogicalProduction.introduction route
      · exact (introductionProduction_iff_rel d v premiseFlow route).mp
          ((introductionSelections_iff_introductionProduction
            d v premiseFlow route).mp separated.1)
      · intro eliminationRoute eliminationProduction sameResidual
        have eliminationMember : eliminationRoute ∈
            selectionEliminationOutputs d v premiseFlow :=
          (selectionEliminationOutputs_iff_eliminationProduction
            d v premiseFlow eliminationRoute).mpr
              ((eliminationProduction_iff_rel d v premiseFlow
                eliminationRoute).mpr eliminationProduction)
        have absent := (residualOccursB_eq_false_iff
          (selectionEliminationOutputs d v premiseFlow) route.2).mp separated.2
        exact absent eliminationRoute eliminationMember sameResidual
  · intro production
    cases production with
    | elimination production =>
        exact Or.inl
          ((selectionEliminationOutputs_iff_eliminationProduction
            d v premiseFlow route).mpr
              ((eliminationProduction_iff_rel d v premiseFlow route).mpr
                production))
    | introduction production separated =>
        apply Or.inr
        apply (mem_separatedIntroductionOutputs_iff d v premiseFlow route).mpr
        constructor
        · exact (introductionSelections_iff_introductionProduction
            d v premiseFlow route).mpr
              ((introductionProduction_iff_rel d v premiseFlow route).mpr
                production)
        · apply (residualOccursB_eq_false_iff
            (selectionEliminationOutputs d v premiseFlow) route.2).mpr
          intro eliminationRoute eliminationMember sameResidual
          apply separated eliminationRoute
          · exact (eliminationProduction_iff_rel d v premiseFlow
              eliminationRoute).mp
                ((selectionEliminationOutputs_iff_eliminationProduction
                  d v premiseFlow eliminationRoute).mp eliminationMember)
          · exact sameResidual

inductive ReferenceResult where
  | error (reason : OccurrenceReplayError)
  | success (routes : RoutePredicate)




def MatchesReference
    (actual : Except OccurrenceReplayError (List FlowPair))
    (reference : ReferenceResult) : Prop :=
  match actual, reference with
  | .error actualError, .error referenceError => actualError = referenceError
  | .ok routes, .success routePredicate =>
      ∀ route, route ∈ routes ↔ routePredicate route
  | _, _ => False




noncomputable def referenceStep (d : Graph) (v : Vertex)
    (premiseRoutes : Vertex -> RoutePredicate) : ReferenceResult := by
  classical
  exact
    if (predsOf d v).isEmpty then
      if v.HYPOTHESIS then
        .success (SourceProduction d v)
      else .error (.nonHypothesisLeaf v)
    else
      let logical := LogicalProduction d v premiseRoutes
      if !(ancestorsInto d v).isEmpty && !v.HYPOTHESIS then
        if _ : ∃ route, logical route then
          let reseeded := ReseededProduction d v logical
          if _ : ∃ route, reseeded route then
            .success reseeded
          else .error (.unsupportedReseeding v)
        else .error (.noLogicalProduction v)
      else if v.HYPOTHESIS then
        if _ : ∃ route, logical route then
          .success (fun route => SourceProduction d v route ∨ logical route)
        else .error (.mixedProductionLost v)
      else if _ : ∃ route, logical route then
        .success logical
      else .error (.noLogicalProduction v)

theorem exists_logicalProduction_iff
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair) :
    (∃ route, LogicalProduction d v
      (fun premise selection => selection ∈ premiseFlow premise) route) ↔
      (derivedOutputs d v premiseFlow).isEmpty = false := by
  rw [List.isEmpty_eq_false_iff]
  constructor
  · rintro ⟨route, production⟩
    exact List.ne_nil_of_mem
      ((derivedOutputs_iff_logicalProduction d v premiseFlow route).mpr
        production)
  · intro nonempty
    obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ nonempty
    exact ⟨route,
      (derivedOutputs_iff_logicalProduction d v premiseFlow route).mp member⟩

theorem exists_reseededProduction_iff
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair) :
    (∃ route, ReseededProduction d v
      (LogicalProduction d v
        (fun premise selection => selection ∈ premiseFlow premise)) route) ↔
      (reseedDerivedRoutes (ancestorsInto d v)
        (derivedOutputs d v premiseFlow)).isEmpty = false := by
  rw [List.isEmpty_eq_false_iff]
  constructor
  · rintro ⟨route, production⟩
    apply List.ne_nil_of_mem
    apply (reseedDerivedRoutes_iff_reseededProduction d v
      (derivedOutputs d v premiseFlow) route).mpr
    cases production with
    | reseed dependency residual ancestor logicalRoute ancestorMember prefixes =>
        exact .reseed dependency residual ancestor
          ((derivedOutputs_iff_logicalProduction d v premiseFlow
            (dependency, residual)).mpr logicalRoute)
          ancestorMember prefixes
  · intro nonempty
    obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ nonempty
    have production := (reseedDerivedRoutes_iff_reseededProduction d v
      (derivedOutputs d v premiseFlow) route).mp member
    cases production with
    | reseed dependency residual ancestor logicalRoute ancestorMember prefixes =>
        exact ⟨(dependency, ancestor.COLOURS), .reseed dependency residual
          ancestor
          ((derivedOutputs_iff_logicalProduction d v premiseFlow
            (dependency, residual)).mp logicalRoute)
          ancestorMember prefixes⟩



theorem occurrenceStep_matches_reference
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair) :
    MatchesReference (occurrenceStep d v premiseFlow)
      (referenceStep d v
        (fun premise route => route ∈ premiseFlow premise)) := by
  classical
  let logical := LogicalProduction d v
    (fun premise route => route ∈ premiseFlow premise)
  have logicalExists :
      (∃ route, logical route) ↔
        (derivedOutputs d v premiseFlow).isEmpty = false :=
    exists_logicalProduction_iff d v premiseFlow
  by_cases leaf : (predsOf d v).isEmpty = true
  · by_cases source : v.HYPOTHESIS = true
    · simp only [occurrenceStep, referenceStep, leaf, source, ↓reduceIte,
        MatchesReference]
      exact selectionSourceRoutes_iff_sourceProduction d v
    · have notSource : v.HYPOTHESIS = false := Bool.eq_false_iff.mpr source
      simp [occurrenceStep, referenceStep, leaf, notSource, MatchesReference]
  · have notLeaf : (predsOf d v).isEmpty = false :=
      Bool.eq_false_iff.mpr leaf
    by_cases reseed :
        (!(ancestorsInto d v).isEmpty && !v.HYPOTHESIS) = true
    · by_cases hasLogical : ∃ route, logical route
      · have derivedNonempty := logicalExists.mp hasLogical
        dsimp [logical] at hasLogical
        let reseeded := ReseededProduction d v logical
        have reseededExists :
            (∃ route, reseeded route) ↔
              (reseedDerivedRoutes (ancestorsInto d v)
                (derivedOutputs d v premiseFlow)).isEmpty = false :=
          exists_reseededProduction_iff d v premiseFlow
        by_cases hasReseeded : ∃ route, reseeded route
        · have executableNonempty := reseededExists.mp hasReseeded
          dsimp [reseeded, logical] at hasReseeded
          simp only [occurrenceStep, referenceStep, notLeaf, reseed,
            derivedNonempty, hasLogical, executableNonempty, hasReseeded,
            ↓reduceIte, ↓reduceDIte, MatchesReference]
          intro route
          exact (reseedDerivedRoutes_iff_reseededProduction d v
            (derivedOutputs d v premiseFlow) route).trans <| by
              constructor <;> intro production
              · cases production with
                | reseed dependency residual ancestor logicalRoute
                    ancestorMember prefixes =>
                    exact .reseed dependency residual ancestor
                      ((derivedOutputs_iff_logicalProduction d v premiseFlow
                        (dependency, residual)).mp logicalRoute)
                      ancestorMember prefixes
              · cases production with
                | reseed dependency residual ancestor logicalRoute
                    ancestorMember prefixes =>
                    exact .reseed dependency residual ancestor
                      ((derivedOutputs_iff_logicalProduction d v premiseFlow
                        (dependency, residual)).mpr logicalRoute)
                      ancestorMember prefixes
        · have noReseeded := hasReseeded
          dsimp [reseeded, logical] at noReseeded
          have executableIsEmpty :
              (reseedDerivedRoutes (ancestorsInto d v)
                (derivedOutputs d v premiseFlow)).isEmpty = true := by
            cases value : (reseedDerivedRoutes (ancestorsInto d v)
                (derivedOutputs d v premiseFlow)).isEmpty with
            | false => exact False.elim (hasReseeded (reseededExists.mpr value))
            | true => rfl
          simp [occurrenceStep, referenceStep, notLeaf, reseed,
            derivedNonempty, hasLogical, noReseeded, executableIsEmpty,
            MatchesReference]
      · have noLogical := hasLogical
        dsimp [logical] at noLogical
        have derivedIsEmpty :
            (derivedOutputs d v premiseFlow).isEmpty = true := by
          cases value : (derivedOutputs d v premiseFlow).isEmpty with
          | false => exact False.elim (hasLogical (logicalExists.mpr value))
          | true => rfl
        simp [occurrenceStep, referenceStep, notLeaf, reseed, noLogical,
          derivedIsEmpty, MatchesReference]
    · have notReseed :
          (!(ancestorsInto d v).isEmpty && !v.HYPOTHESIS) = false :=
        Bool.eq_false_iff.mpr reseed
      by_cases source : v.HYPOTHESIS = true
      · by_cases hasLogical : ∃ route, logical route
        · have derivedNonempty := logicalExists.mp hasLogical
          dsimp [logical] at hasLogical
          simp only [occurrenceStep, referenceStep, notLeaf, source, derivedNonempty, hasLogical, Bool.not_true, Bool.and_false, Bool.false_eq_true, ↓reduceIte, ↓reduceDIte, MatchesReference]
          intro route
          rw [List.mem_eraseDups, List.mem_append]
          constructor
          · rintro (sourceRoute | derived)
            · exact Or.inl
                ((selectionSourceRoutes_iff_sourceProduction d v route).mp
                  sourceRoute)
            · exact Or.inr
                ((derivedOutputs_iff_logicalProduction d v premiseFlow
                  route).mp derived)
          · rintro (sourceRoute | derived)
            · exact Or.inl
                ((selectionSourceRoutes_iff_sourceProduction d v route).mpr
                  sourceRoute)
            · exact Or.inr
                ((derivedOutputs_iff_logicalProduction d v premiseFlow
                  route).mpr derived)
        · have noLogical := hasLogical
          dsimp [logical] at noLogical
          have derivedIsEmpty :
              (derivedOutputs d v premiseFlow).isEmpty = true := by
            cases value : (derivedOutputs d v premiseFlow).isEmpty with
            | false => exact False.elim (hasLogical (logicalExists.mpr value))
            | true => rfl
          simp [occurrenceStep, referenceStep, notLeaf, source, noLogical, derivedIsEmpty, MatchesReference]
      · have notSource : v.HYPOTHESIS = false := Bool.eq_false_iff.mpr source
        have ancestorsEmpty : (ancestorsInto d v).isEmpty = true := by
          simpa [notSource] using notReseed
        by_cases hasLogical : ∃ route, logical route
        · have derivedNonempty := logicalExists.mp hasLogical
          dsimp [logical] at hasLogical
          simp only [occurrenceStep, referenceStep, notLeaf, notSource, ancestorsEmpty, derivedNonempty, hasLogical, ↓reduceDIte, MatchesReference]
          exact derivedOutputs_iff_logicalProduction d v premiseFlow
        · have noLogical := hasLogical
          dsimp [logical] at noLogical
          have derivedIsEmpty :
              (derivedOutputs d v premiseFlow).isEmpty = true := by
            cases value : (derivedOutputs d v premiseFlow).isEmpty with
            | false => exact False.elim (hasLogical (logicalExists.mpr value))
            | true => rfl
          simp [occurrenceStep, referenceStep, notLeaf, notSource, ancestorsEmpty, noLogical, derivedIsEmpty, MatchesReference]

abbrev ReferencePremiseTable := List (Vertex × RoutePredicate)



def referenceRoutesFor (table : ReferencePremiseTable) (v : Vertex) :
    RoutePredicate :=
  match table.find? fun entry => entry.1 == v with
  | some entry => entry.2
  | none => fun _ => False




inductive PremiseTableMatches :
    List (Vertex × List FlowPair) -> ReferencePremiseTable -> Prop
  | nil : PremiseTableMatches [] []
  | cons (vertex : Vertex) (actualRoutes : List FlowPair)
      (referenceRoutes : RoutePredicate)
      (actualTail : List (Vertex × List FlowPair))
      (referenceTail : ReferencePremiseTable)
      (head : ∀ route, route ∈ actualRoutes ↔ referenceRoutes route)
      (tail : PremiseTableMatches actualTail referenceTail) :
      PremiseTableMatches
        ((vertex, actualRoutes) :: actualTail)
        ((vertex, referenceRoutes) :: referenceTail)

theorem PremiseTableMatches.routesFor
    {actual : List (Vertex × List FlowPair)}
    {reference : ReferencePremiseTable}
    (hmatches : PremiseTableMatches actual reference) :
    ∀ vertex route,
      route ∈ OccurrenceAwareReplay.routesFor actual vertex ↔
        referenceRoutesFor reference vertex route := by
  intro vertex route
  induction hmatches with
  | nil => simp [OccurrenceAwareReplay.routesFor, referenceRoutesFor]
  | cons headVertex actualRoutes referenceRoutes actualTail referenceTail
      headMatches tailMatches ih =>
      simp only [OccurrenceAwareReplay.routesFor, referenceRoutesFor,
        List.find?_cons]
      cases headEq : (headVertex == vertex) with
      | false => simpa [headEq] using ih
      | true => simpa [headEq] using headMatches route

def actualPremises
    (replay : Vertex -> Except OccurrenceReplayError (List FlowPair)) :
    List Vertex -> Except OccurrenceReplayError
      (List (Vertex × List FlowPair))
  | [] => .ok []
  | premise :: premises =>
      match replay premise with
      | .error reason => .error reason
      | .ok routes =>
          match actualPremises replay premises with
          | .error reason => .error reason
          | .ok table => .ok ((premise, routes) :: table)



def referencePremises (replay : Vertex -> ReferenceResult) :
    List Vertex -> Except OccurrenceReplayError ReferencePremiseTable
  | [] => .ok []
  | premise :: premises =>
      match replay premise with
      | .error reason => .error reason
      | .success routes =>
          match referencePremises replay premises with
          | .error reason => .error reason
          | .ok table => .ok ((premise, routes) :: table)

def PremiseResultsMatch
    (actual : Except OccurrenceReplayError
      (List (Vertex × List FlowPair)))
    (reference : Except OccurrenceReplayError ReferencePremiseTable) : Prop :=
  match actual, reference with
  | .error actualError, .error referenceError => actualError = referenceError
  | .ok actualTable, .ok referenceTable =>
      PremiseTableMatches actualTable referenceTable
  | _, _ => False

theorem actualPremises_eq_mapM
    (replay : Vertex -> Except OccurrenceReplayError (List FlowPair))
    (premises : List Vertex) :
    actualPremises replay premises =
      premises.mapM fun premise => do
        let routes <- replay premise
        return (premise, routes) := by
  induction premises with
  | nil => rfl
  | cons premise premises ih =>
      simp only [actualPremises, List.mapM_cons]
      rw [← ih]
      cases replay premise <;> cases actualPremises replay premises <;> rfl

theorem premiseTraversal_matches
    (actualReplay : Vertex ->
      Except OccurrenceReplayError (List FlowPair))
    (referenceReplay : Vertex -> ReferenceResult)
    (pointwise : ∀ vertex,
      MatchesReference (actualReplay vertex) (referenceReplay vertex))
    (premises : List Vertex) :
    PremiseResultsMatch
      (actualPremises actualReplay premises)
      (referencePremises referenceReplay premises) := by
  induction premises with
  | nil => exact PremiseTableMatches.nil
  | cons premise premises ih =>
      have headMatches := pointwise premise
      cases actualEq : actualReplay premise with
      | error actualError =>
          cases referenceEq : referenceReplay premise with
          | error referenceError =>
              have errorEq : actualError = referenceError := by
                simpa [MatchesReference, actualEq, referenceEq] using headMatches
              simpa [actualPremises, referencePremises, actualEq, referenceEq,
                PremiseResultsMatch] using errorEq
          | success referenceRoutes =>
              simp [MatchesReference, actualEq, referenceEq] at headMatches
      | ok actualRoutes =>
          cases referenceEq : referenceReplay premise with
          | error referenceError =>
              simp [MatchesReference, actualEq, referenceEq] at headMatches
          | success referenceRoutes =>
              have routeMatches : ∀ route,
                  route ∈ actualRoutes ↔ referenceRoutes route := by
                simpa [MatchesReference, actualEq, referenceEq] using headMatches
              cases actualTailEq : actualPremises actualReplay premises with
              | error actualError =>
                  cases referenceTailEq :
                      referencePremises referenceReplay premises with
                  | error referenceError =>
                      have errorEq : actualError = referenceError := by
                        simpa [PremiseResultsMatch, actualTailEq,
                          referenceTailEq] using ih
                      simpa [actualPremises, referencePremises, actualEq,
                        referenceEq, actualTailEq, referenceTailEq,
                        PremiseResultsMatch] using errorEq
                  | ok referenceTable =>
                      simp [PremiseResultsMatch, actualTailEq,
                        referenceTailEq] at ih
              | ok actualTable =>
                  cases referenceTailEq :
                      referencePremises referenceReplay premises with
                  | error referenceError =>
                      simp [PremiseResultsMatch, actualTailEq,
                        referenceTailEq] at ih
                  | ok referenceTable =>
                      have tailMatches :
                          PremiseTableMatches actualTable referenceTable := by
                        simpa [PremiseResultsMatch, actualTailEq,
                          referenceTailEq] using ih
                      simpa [actualPremises, referencePremises, actualEq,
                        referenceEq, actualTailEq, referenceTailEq,
                        PremiseResultsMatch] using
                          PremiseTableMatches.cons premise actualRoutes
                            referenceRoutes actualTable referenceTable
                            routeMatches tailMatches

theorem referenceStep_congr
    (d : Graph) (v : Vertex)
    (left right : Vertex -> RoutePredicate)
    (same : ∀ vertex route, left vertex route ↔ right vertex route) :
    referenceStep d v left = referenceStep d v right := by
  have predicatesEqual : left = right := by
    funext vertex route
    exact propext (same vertex route)
  rw [predicatesEqual]




noncomputable def referenceReplayAt (d : Graph) :
    Nat -> Vertex -> ReferenceResult
  | 0, vertex => .error (.fuelExhausted vertex)
  | fuel + 1, vertex =>
      match referencePremises (referenceReplayAt d fuel)
          (predsOf d vertex) with
      | .error reason => .error reason
      | .ok premiseTable =>
          referenceStep d vertex (referenceRoutesFor premiseTable)

theorem occurrenceReplayAt_succ
    (d : Graph) (fuel : Nat) (vertex : Vertex) :
    occurrenceReplayAt d (fuel + 1) vertex =
      match actualPremises (occurrenceReplayAt d fuel)
          (predsOf d vertex) with
      | .error reason => .error reason
      | .ok premiseTable =>
          occurrenceStep d vertex
            (OccurrenceAwareReplay.routesFor premiseTable) := by
  simp only [occurrenceReplayAt]
  rw [actualPremises_eq_mapM]
  cases List.mapM
      (fun premise => do
        let routes ← occurrenceReplayAt d fuel premise
        return (premise, routes))
      (predsOf d vertex) <;> rfl




theorem occurrenceReplayAt_matches_reference
    (d : Graph) (fuel : Nat) (vertex : Vertex) :
    MatchesReference (occurrenceReplayAt d fuel vertex)
      (referenceReplayAt d fuel vertex) := by
  induction fuel generalizing vertex with
  | zero => simp [occurrenceReplayAt, referenceReplayAt, MatchesReference]
  | succ fuel ih =>
      rw [occurrenceReplayAt_succ]
      simp only [referenceReplayAt]
      have traversed := premiseTraversal_matches
        (occurrenceReplayAt d fuel) (referenceReplayAt d fuel) ih
        (predsOf d vertex)
      cases actualEq : actualPremises (occurrenceReplayAt d fuel)
          (predsOf d vertex) with
      | error actualError =>
          cases referenceEq : referencePremises (referenceReplayAt d fuel)
              (predsOf d vertex) with
          | error referenceError =>
              have errorEq : actualError = referenceError := by
                simpa [PremiseResultsMatch, actualEq, referenceEq] using
                  traversed
              simpa [actualEq, referenceEq, MatchesReference] using errorEq
          | ok referenceTable =>
              simp [PremiseResultsMatch, actualEq, referenceEq] at traversed
      | ok actualTable =>
          cases referenceEq : referencePremises (referenceReplayAt d fuel)
              (predsOf d vertex) with
          | error referenceError =>
              simp [PremiseResultsMatch, actualEq, referenceEq] at traversed
          | ok referenceTable =>
              have tablesMatch :
                  PremiseTableMatches actualTable referenceTable := by
                simpa [PremiseResultsMatch, actualEq, referenceEq] using
                  traversed
              have localMatch := occurrenceStep_matches_reference d vertex
                (OccurrenceAwareReplay.routesFor actualTable)
              have stepEq : referenceStep d vertex
                    (fun premise route =>
                      route ∈ OccurrenceAwareReplay.routesFor actualTable
                        premise) =
                  referenceStep d vertex
                    (referenceRoutesFor referenceTable) :=
                referenceStep_congr d vertex _ _ tablesMatch.routesFor
              simpa [actualEq, referenceEq, stepEq] using localMatch




def ReferenceResultComplete : ReferenceResult -> Prop
  | .error _ => False
  | .success routes =>
      (∃ route, routes route) ∧
        ∀ route, routes route -> routeClosedB route = true

def executableResultCompleteB
    (result : Except OccurrenceReplayError (List FlowPair)) : Bool :=
  match result with
  | .error _ => false
  | .ok routes => !routes.isEmpty && routes.all routeClosedB

theorem executableResultComplete_iff_reference
    (actual : Except OccurrenceReplayError (List FlowPair))
    (reference : ReferenceResult)
    (hmatches : MatchesReference actual reference) :
    executableResultCompleteB actual = true ↔
      ReferenceResultComplete reference := by
  cases actual with
  | error actualError =>
      cases reference with
      | error referenceError => simp [executableResultCompleteB,
          ReferenceResultComplete]
      | success routes => simp [MatchesReference] at hmatches
  | ok actualRoutes =>
      cases reference with
      | error referenceError => simp [MatchesReference] at hmatches
      | success referenceRoutes =>
          have routeIff : ∀ route,
              route ∈ actualRoutes ↔ referenceRoutes route := by
            simpa [MatchesReference] using hmatches
          constructor
          · intro complete
            have parts : !actualRoutes.isEmpty = true ∧
                actualRoutes.all routeClosedB = true := by
              simpa [executableResultCompleteB] using complete
            have nonempty : actualRoutes ≠ [] := by
              simpa [executableResultCompleteB, List.isEmpty_eq_false_iff]
                using parts.1
            obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ nonempty
            constructor
            · exact ⟨route, (routeIff route).mp member⟩
            · intro selection selectionReference
              have selectionMember := (routeIff selection).mpr
                selectionReference
              exact List.all_eq_true.mp parts.2 selection selectionMember
          · rintro ⟨⟨route, routeReference⟩, allClosed⟩
            have nonempty : !actualRoutes.isEmpty = true := by
              have actualNonempty : actualRoutes.isEmpty = false := by
                rw [List.isEmpty_eq_false_iff]
                exact List.ne_nil_of_mem
                  ((routeIff route).mpr routeReference)
              simp [actualNonempty]
            have closed : actualRoutes.all routeClosedB = true := by
              apply List.all_eq_true.mpr
              intro selection selectionMember
              exact allClosed selection ((routeIff selection).mp selectionMember)
            simpa [executableResultCompleteB] using And.intro nonempty closed


def ReferenceRootsComplete (d : Graph) (fuel : Nat) : Prop :=
  (occurrenceRoots d).isEmpty = false ∧
    ∀ root, root ∈ occurrenceRoots d ->
      ReferenceResultComplete (referenceReplayAt d fuel root)




theorem occurrenceRootsCompleteB_iff_reference
    (d : Graph) (fuel : Nat) :
    occurrenceRootsCompleteB d fuel = true ↔
      ReferenceRootsComplete d fuel := by
  unfold occurrenceRootsCompleteB ReferenceRootsComplete
  rw [Bool.and_eq_true]
  constructor
  · rintro ⟨rootsNonempty, rootsComplete⟩
    constructor
    · simpa using rootsNonempty
    · intro root rootMember
      have actualComplete := List.all_eq_true.mp rootsComplete root rootMember
      exact (executableResultComplete_iff_reference
        (occurrenceReplayAt d fuel root) (referenceReplayAt d fuel root)
        (occurrenceReplayAt_matches_reference d fuel root)).mp actualComplete
  · rintro ⟨rootsNonempty, rootsComplete⟩
    constructor
    · simpa using rootsNonempty
    · apply List.all_eq_true.mpr
      intro root rootMember
      exact (executableResultComplete_iff_reference
        (occurrenceReplayAt d fuel root) (referenceReplayAt d fuel root)
        (occurrenceReplayAt_matches_reference d fuel root)).mpr
          (rootsComplete root rootMember)

end HCSuccessorFixedGraphReference
