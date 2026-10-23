import Semantic.OccurrenceReadingsLineageModel

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe RouteIndexedMultiToken

namespace TaggedLineage

set_option maxHeartbeats 0





lemma project_elimReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) :
    ListSetEq (projectFlowPairs (elimReseedOutputs d v premiseTokens ancestors))
      (LambdaFlow.elimReseedWith d (SingleIntroductionReference.guard d) v
        (fun u => projectFlowPairs (premiseTokens u)) ancestors) := by
  have projection :
      projectFlowPairs (elimReseedOutputs d v premiseTokens ancestors) =
        LambdaFlow.elimReseedWith d (SingleIntroductionReference.guard d) v
          (fun u => projectFlowPairs (premiseTokens u)) ancestors := by
    simp only [projectFlowPairs, elimReseedOutputs, LambdaFlow.elimReseedWith, Bool.and_true,
      List.map_flatMap, List.flatMap_map,
      RouteIndexedToken.flowPair]
    congr 1
    funext pair
    rcases pair with ⟨major, minor⟩
    congr 1
    funext minorToken
    congr 1
    funext majorToken
    congr 1
    funext minorEdge
    congr 1
    funext majorEdge
    split <;> simp [List.map_map, Function.comp_def, mkToken,
      RouteIndexedToken.flowPair]
  rw [projection]
  exact ListSetEq.refl _

lemma project_introReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) :
    ListSetEq (projectFlowPairs (introReseedOutputs d v premiseTokens ancestors))
      (LambdaFlow.introReseedWith d (SingleIntroductionReference.guard d) v
        (fun u => projectFlowPairs (premiseTokens u)) ancestors) := by
  intro route
  simp [introReseedOutputs, LambdaFlow.introReseedWith, projectFlowPairs,
    RouteIndexedToken.flowPair] ; aesop

lemma project_reseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (reseedOutputs d v premiseTokens))
      (LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v (fun u => projectFlowPairs (premiseTokens u))) := by
  cases hempty : (eligibleAncestorsAtRoot d v).isEmpty with
  | true =>
      simpa [reseedOutputs, LambdaFlow.reseedWith, hempty] using
        (ListSetEq.refl ([] : List FlowPair))
  | false =>
    simp only [reseedOutputs, LambdaFlow.reseedWith, hempty, Bool.false_eq_true,
      if_false]
    apply ListSetEq.trans
      (show ListSetEq
        (projectFlowPairs
          (elimReseedOutputs d v premiseTokens (eligibleAncestorsAtRoot d v) ++
           introReseedOutputs d v premiseTokens (eligibleAncestorsAtRoot d v)))
        (LambdaFlow.elimReseedWith d (SingleIntroductionReference.guard d) v
          (fun u => projectFlowPairs (premiseTokens u))
            (eligibleAncestorsAtRoot d v) ++
         LambdaFlow.introReseedWith d (SingleIntroductionReference.guard d) v
          (fun u => projectFlowPairs (premiseTokens u))
            (eligibleAncestorsAtRoot d v)) from by
          rw [projectFlowPairs, List.map_append]
          exact append_congr
            (project_elimReseedOutputs d v premiseTokens _)
            (project_introReseedOutputs d v premiseTokens _))
    intro route
    simp

private def flowElimCombine (d : Graph) (v major minor : Vertex)
    (minorRoutes majorRoutes : List FlowPair) : List FlowPair :=
  minorRoutes.flatMap fun left => majorRoutes.flatMap fun right =>
    (edgesBetween d minor v).flatMap fun minorEdge =>
      (edgesBetween d major v).filterMap fun majorEdge =>
        if SingleIntroductionReference.guard d left.1 minorEdge &&
            SingleIntroductionReference.guard d right.1 majorEdge then
          match consume minorEdge left.2, consume majorEdge right.2 with
          | some p, some q =>
              if p = q then some (depUnion left.1 right.1, p) else none
          | _, _ => none
        else none

private def flowElimFrom (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    flowElimCombine d v major minor (premiseFlow minor) (premiseFlow major)

private def flowIntroFrom (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [premise], Formula.implication alpha beta =>
      if premise.FORMULA = beta then
        (premiseFlow premise).flatMap fun route =>
          (edgesBetween d premise v).filterMap fun edge =>
            if SingleIntroductionReference.guard d route.1 edge then
              (consume edge route.2).map fun tail =>
                (depRemove route.1 alpha, tail)
            else none
      else []
  | _, _ => []

private lemma project_elimCombine (d : Graph) (v major minor : Vertex)
    (minorTokens majorTokens : List RouteIndexedToken) :
    projectFlowPairs (elimCombine d v major minor minorTokens majorTokens) =
      flowElimCombine d v major minor
        (projectFlowPairs minorTokens) (projectFlowPairs majorTokens) := by
  simp only [projectFlowPairs, elimCombine, flowElimCombine,
    List.map_flatMap, List.flatMap_map, List.map_filterMap,
    RouteIndexedToken.flowPair]
  congr 1
  funext minorToken
  congr 1
  funext majorToken
  congr 1
  funext minorEdge
  congr 1
  funext majorEdge
  split
  next guard =>
    cases hminor : consume minorEdge minorToken.residual <;>
      cases hmajor : consume majorEdge majorToken.residual <;>
        simp [mkToken, RouteIndexedToken.flowPair]
  next guard => simp

private lemma project_elimOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (elimOutputs d v premiseTokens))
      (flowElimFrom d v (fun u => projectFlowPairs (premiseTokens u))) := by
  have projection : projectFlowPairs (elimOutputs d v premiseTokens) =
      flowElimFrom d v (fun u => projectFlowPairs (premiseTokens u)) := by
    simp only [projectFlowPairs, elimOutputs, flowElimFrom, List.map_flatMap]
    congr 1
    funext pair
    exact project_elimCombine d v pair.1 pair.2 _ _
  rw [projection]
  exact ListSetEq.refl _

private lemma project_introOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (introOutputs d v premiseTokens))
      (flowIntroFrom d v (fun u => projectFlowPairs (premiseTokens u))) := by
  intro route
  simp [introOutputs, introOutputsFor, flowIntroFrom, projectFlowPairs,
    RouteIndexedToken.flowPair]; aesop

private lemma project_mixedIntroOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex -> List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (mixedIntroOutputs d v premiseTokens))
      (LambdaFlow.mixedWith d (SingleIntroductionReference.guard d) v
        (fun u => projectFlowPairs (premiseTokens u))) := by
  intro route
  simp [mixedIntroOutputs, LambdaFlow.mixedWith, projectFlowPairs,
    RouteIndexedToken.flowPair]; aesop

private lemma flowStepFrom_as_parts (d : Graph)
    (premiseFlow : Vertex → List FlowPair) (v : Vertex) :
    LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) premiseFlow v =
      if v.HYPOTHESIS || (predsOf d v).isEmpty then
        let ancestors := ancestorsInto d v
        if ancestors.isEmpty then [(normalizeDep [v.FORMULA], [])]
        else ancestors.map fun ancestor =>
          (normalizeDep [v.FORMULA], ancestor.COLOURS)
      else
        let core := if (elimPairsAt d v).isEmpty then
          flowIntroFrom d v premiseFlow
        else flowElimFrom d v premiseFlow ++ LambdaFlow.mixedWith d (SingleIntroductionReference.guard d) v premiseFlow
        (core ++ LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v premiseFlow).eraseDups := by
  rfl



theorem project_stepFrom_self (d : Graph)
    (premiseTokens : Vertex → List RouteIndexedToken) (v : Vertex) :
    ListSetEq (projectFlowPairs (stepFrom d premiseTokens v))
      (LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) (fun u => projectFlowPairs (premiseTokens u)) v) := by
  rw [stepFrom, project_assignSlots]
  rw [flowStepFrom_as_parts]
  by_cases hguard :
      (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp only [rawStepFrom, hguard, if_true]
    rw [project_sourceOutputs]
    exact ListSetEq.refl _
  · have hguardFalse :
        (v.HYPOTHESIS || (predsOf d v).isEmpty) = false :=
      Bool.eq_false_of_not_eq_true hguard
    simp only [rawStepFrom, hguardFalse, Bool.false_eq_true, if_false]
    by_cases hpairs : (elimPairsAt d v).isEmpty = true
    · simp only [hpairs, if_true]
      apply ListSetEq.trans
        (show ListSetEq
          (projectFlowPairs
            (introOutputs d v premiseTokens ++
              reseedOutputs d v premiseTokens))
          (flowIntroFrom d v
              (fun u => projectFlowPairs (premiseTokens u)) ++
            LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v
              (fun u => projectFlowPairs (premiseTokens u))) from by
            rw [projectFlowPairs, List.map_append]
            exact append_congr
              (project_introOutputs d v premiseTokens)
              (project_reseedOutputs d v premiseTokens))
      intro route
      simp
    · have hpairsFalse : (elimPairsAt d v).isEmpty = false :=
        Bool.eq_false_of_not_eq_true hpairs
      simp only [hpairsFalse, Bool.false_eq_true, if_false]
      apply ListSetEq.trans
        (show ListSetEq
          (projectFlowPairs
            ((elimOutputs d v premiseTokens ++
                mixedIntroOutputs d v premiseTokens) ++
              reseedOutputs d v premiseTokens))
          ((flowElimFrom d v
                (fun u => projectFlowPairs (premiseTokens u)) ++
              LambdaFlow.mixedWith d (SingleIntroductionReference.guard d) v
                (fun u => projectFlowPairs (premiseTokens u))) ++
            LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v
              (fun u => projectFlowPairs (premiseTokens u))) from by
            rw [projectFlowPairs, List.map_append, List.map_append]
            exact append_congr
              (append_congr
                (project_elimOutputs d v premiseTokens)
                (project_mixedIntroOutputs d v premiseTokens))
              (project_reseedOutputs d v premiseTokens))
      intro route
      simp


theorem evaluateAt_project (d : Graph) : ∀ fuel v,
    ListSetEq (projectFlowPairs (evaluateAt d fuel v))
      (SingleIntroductionReference.routes d fuel v) := by
  intro fuel
  induction fuel with
  | zero => intro v; exact ListSetEq.refl []
  | succ fuel ih =>
      intro v
      exact (project_stepFrom_self d (evaluateAt d fuel) v).trans
        (DependencyGuard.stepWith_setEq d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
          _ _ ih v)

#print axioms evaluateAt_project
end TaggedLineage
