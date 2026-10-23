import Semantic.SpineGuardFreeLineage

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe RouteIndexedMultiToken

namespace SpineLineage

set_option maxHeartbeats 0





lemma project_elimReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) :
    ListSetEq (projectFlowPairs (elimReseedOutputs d v premiseTokens ancestors))
      (LambdaFlow.elimReseedWith d LambdaFlow.allowAll v
        (fun u => projectFlowPairs (premiseTokens u)) ancestors) := by
  have projection :
      projectFlowPairs (elimReseedOutputs d v premiseTokens ancestors) =
        LambdaFlow.elimReseedWith d LambdaFlow.allowAll v
          (fun u => projectFlowPairs (premiseTokens u)) ancestors := by
    simp only [projectFlowPairs, elimReseedOutputs, LambdaFlow.elimReseedWith, LambdaFlow.allowAll, Bool.and_true,
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
      (LambdaFlow.introReseedWith d LambdaFlow.allowAll v
        (fun u => projectFlowPairs (premiseTokens u)) ancestors) := by
  intro route
  simp [introReseedOutputs, LambdaFlow.introReseedWith, LambdaFlow.allowAll, projectFlowPairs,
    RouteIndexedToken.flowPair] ; aesop

lemma project_reseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (reseedOutputs d v premiseTokens))
      (LambdaFlow.reseedWith d LambdaFlow.allowAll v (fun u => projectFlowPairs (premiseTokens u))) := by
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
        (LambdaFlow.elimReseedWith d LambdaFlow.allowAll v
          (fun u => projectFlowPairs (premiseTokens u))
            (eligibleAncestorsAtRoot d v) ++
         LambdaFlow.introReseedWith d LambdaFlow.allowAll v
          (fun u => projectFlowPairs (premiseTokens u))
            (eligibleAncestorsAtRoot d v)) from by
          rw [projectFlowPairs, List.map_append]
          exact append_congr
            (project_elimReseedOutputs d v premiseTokens _)
            (project_introReseedOutputs d v premiseTokens _))
    intro route
    simp

private def flowElimFrom (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) : List FlowPair :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    FlowSpec.elimCombine d v major minor (premiseFlow minor) (premiseFlow major)

private def flowIntroFrom (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [premise], Formula.implication alpha beta =>
      if premise.FORMULA = beta then
        (premiseFlow premise).flatMap fun route =>
          (edgesBetween d premise v).filterMap fun edge =>
            (consume edge route.2).map fun tail =>
              (depRemove route.1 alpha, tail)
      else []
  | _, _ => []

private lemma flowStepFrom_as_parts (d : Graph)
    (premiseFlow : Vertex → List FlowPair) (v : Vertex) :
    LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll premiseFlow v =
      if v.HYPOTHESIS || (predsOf d v).isEmpty then
        let ancestors := ancestorsInto d v
        if ancestors.isEmpty then [(normalizeDep [v.FORMULA], [])]
        else ancestors.map fun ancestor =>
          (normalizeDep [v.FORMULA], ancestor.COLOURS)
      else
        let core := if (elimPairsAt d v).isEmpty then
          flowIntroFrom d v premiseFlow
        else flowElimFrom d v premiseFlow ++ mixedIntroPairs d v premiseFlow
        (core ++ LambdaFlow.reseedWith d LambdaFlow.allowAll v premiseFlow).eraseDups := by
  rfl



theorem project_stepFrom_self (d : Graph)
    (premiseTokens : Vertex → List RouteIndexedToken) (v : Vertex) :
    ListSetEq (projectFlowPairs (stepFrom d premiseTokens v))
      (LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll (fun u => projectFlowPairs (premiseTokens u)) v) := by
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
            LambdaFlow.reseedWith d LambdaFlow.allowAll v
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
              mixedIntroPairs d v
                (fun u => projectFlowPairs (premiseTokens u))) ++
            LambdaFlow.reseedWith d LambdaFlow.allowAll v
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
      (LambdaFlow.guardFreeFlowAt d fuel v) := by
  intro fuel
  induction fuel with
  | zero => intro v; exact ListSetEq.refl []
  | succ fuel ih =>
      intro v
      exact (project_stepFrom_self d (evaluateAt d fuel) v).trans
        (DependencyGuard.stepWith_setEq d LambdaFlow.allowAll LambdaFlow.allowAll
          _ _ ih v)

#print axioms evaluateAt_project
end SpineLineage
