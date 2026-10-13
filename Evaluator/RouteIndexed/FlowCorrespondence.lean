import Evaluator.RouteIndexed.Definitions

open DLDS FlowSemantics Certificate.Flow Certificate.Flow.Foundation

namespace Evaluator.RouteIndexed

set_option maxHeartbeats 0
set_option maxRecDepth 100000

@[simp] lemma flowPair_dependency (token : RouteIndexedToken) :
    token.flowPair.1 = token.dependency := rfl

@[simp] lemma flowPair_residual (token : RouteIndexedToken) :
    token.flowPair.2 = token.residual := rfl

@[simp] lemma flowPair_mkToken (d : Graph) (v : Vertex) (key : RouteKey)
    (dependency : Dep) (residual : ColourPath) :
    (mkToken d v key dependency residual).flowPair =
      (dependency, residual) := rfl

lemma project_sourceOutputs (d : Graph) (v : Vertex) :
    projectFlowPairs (sourceOutputs d v) =
      (let ancestors := ancestorsInto d v
       if ancestors.isEmpty then [(normalizeDep [v.FORMULA], [])]
       else ancestors.map fun ancestor =>
         (normalizeDep [v.FORMULA], ancestor.COLOURS)) := by
  by_cases h : ancestorsInto d v = []
  · simp [sourceOutputs, projectFlowPairs, h]
  · simp [sourceOutputs, projectFlowPairs, h, List.map_map,
      Function.comp_def]

lemma project_repetitionOutputs (d : Graph) (v : Vertex) (edge : Deduction)
    (tokens : List RouteIndexedToken) :
    projectFlowPairs (repetitionOutputs d v edge tokens) =
      projectFlowPairs tokens := by
  simp [repetitionOutputs, projectFlowPairs, Evaluator.RouteIndexed.RouteIndexedToken.flowPair, mkToken]

lemma project_elimCombine (d : Graph) (v major minor : Vertex)
    (minorTokens majorTokens : List RouteIndexedToken) :
    ListSetEq
      (projectFlowPairs (elimCombine d v major minor minorTokens majorTokens))
      (FlowSemantics.elimCombine d v major minor
        (projectFlowPairs minorTokens) (projectFlowPairs majorTokens)) := by
  have projection :
      projectFlowPairs (elimCombine d v major minor minorTokens majorTokens) =
        FlowSemantics.elimCombine d v major minor
          (projectFlowPairs minorTokens) (projectFlowPairs majorTokens) := by
    simp only [projectFlowPairs, Evaluator.RouteIndexed.elimCombine,
      FlowSemantics.elimCombine, List.map_flatMap, List.flatMap_map,
      List.map_filterMap, Evaluator.RouteIndexed.RouteIndexedToken.flowPair]
    congr 1
    funext minorToken
    congr 1
    funext majorToken
    congr 1
    funext minorEdge
    congr 1
    funext majorEdge
    cases hminor : consume minorEdge minorToken.residual <;>
      cases hmajor : consume majorEdge majorToken.residual <;>
        simp [mkToken, Evaluator.RouteIndexed.RouteIndexedToken.flowPair]
  rw [projection]
  exact Certificate.Flow.ListSetEq.refl _

lemma project_elimOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (elimOutputs d v premiseTokens))
      ((elimPairsAt d v).flatMap fun (major, minor) =>
        FlowSemantics.elimCombine d v major minor
          (projectFlowPairs (premiseTokens minor))
          (projectFlowPairs (premiseTokens major))) := by
  intro route
  simp [elimOutputs, FlowSemantics.elimCombine,
    Evaluator.RouteIndexed.elimCombine, projectFlowPairs,
    Evaluator.RouteIndexed.RouteIndexedToken.flowPair, mkToken] ;
    aesop (config := { maxRuleApplications := 1000 })

lemma project_introOutputsFor (d : Graph) (v premise : Vertex)
    (alpha : Formula) (tokens : List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (introOutputsFor d v premise alpha tokens))
      ((projectFlowPairs tokens).flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail =>
            (depRemove route.1 alpha, tail)) := by
  intro route
  simp [introOutputsFor, projectFlowPairs, List.map_flatMap,
    Evaluator.RouteIndexed.RouteIndexedToken.flowPair] ; aesop

lemma project_introOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (introOutputs d v premiseTokens))
      (match predsOf d v, v.FORMULA with
       | [premise], Formula.implication alpha beta =>
           if premise.FORMULA = beta then
             (projectFlowPairs (premiseTokens premise)).flatMap fun route =>
               (edgesBetween d premise v).filterMap fun edge =>
                 (consume edge route.2).map fun tail =>
                   (depRemove route.1 alpha, tail)
           else []
       | _, _ => []) := by
  intro route
  simp [introOutputs, projectFlowPairs, introOutputsFor,
    Evaluator.RouteIndexed.RouteIndexedToken.flowPair] ; aesop

lemma project_mixedIntroOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (mixedIntroOutputs d v premiseTokens))
      (mixedIntroPairs d v
        (fun u => projectFlowPairs (premiseTokens u))) := by
  intro route
  simp [mixedIntroOutputs, mixedIntroPairs, projectFlowPairs,
    Evaluator.RouteIndexed.RouteIndexedToken.flowPair] ; aesop

lemma project_elimReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) :
    ListSetEq (projectFlowPairs (elimReseedOutputs d v premiseTokens ancestors))
      (elimReseedPairs d v
        (fun u => projectFlowPairs (premiseTokens u)) ancestors) := by
  have projection :
      projectFlowPairs (elimReseedOutputs d v premiseTokens ancestors) =
        elimReseedPairs d v
          (fun u => projectFlowPairs (premiseTokens u)) ancestors := by
    simp only [projectFlowPairs, elimReseedOutputs, elimReseedPairs,
      List.map_flatMap, List.flatMap_map,
      Evaluator.RouteIndexed.RouteIndexedToken.flowPair]
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
      Evaluator.RouteIndexed.RouteIndexedToken.flowPair]
  rw [projection]
  exact Certificate.Flow.ListSetEq.refl _

lemma project_introReseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (ancestors : List Ancestral) :
    ListSetEq (projectFlowPairs (introReseedOutputs d v premiseTokens ancestors))
      (introReseedPairs d v
        (fun u => projectFlowPairs (premiseTokens u)) ancestors) := by
  intro route
  simp [introReseedOutputs, introReseedPairs, projectFlowPairs,
    Evaluator.RouteIndexed.RouteIndexedToken.flowPair] ; aesop

lemma project_reseedOutputs (d : Graph) (v : Vertex)
    (premiseTokens : Vertex → List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (reseedOutputs d v premiseTokens))
      (reseedPairs d v (fun u => projectFlowPairs (premiseTokens u))) := by
  cases hempty : (eligibleAncestorsAtRoot d v).isEmpty with
  | true =>
      simpa [reseedOutputs, reseedPairs, hempty] using
        (Certificate.Flow.ListSetEq.refl ([] : List FlowPair))
  | false =>
    simp only [reseedOutputs, reseedPairs, hempty, Bool.false_eq_true,
      if_false]
    apply Certificate.Flow.ListSetEq.trans
      (show ListSetEq
        (projectFlowPairs
          (elimReseedOutputs d v premiseTokens (eligibleAncestorsAtRoot d v) ++
           introReseedOutputs d v premiseTokens (eligibleAncestorsAtRoot d v)))
        (elimReseedPairs d v
          (fun u => projectFlowPairs (premiseTokens u))
            (eligibleAncestorsAtRoot d v) ++
         introReseedPairs d v
          (fun u => projectFlowPairs (premiseTokens u))
            (eligibleAncestorsAtRoot d v)) from by
          rw [projectFlowPairs, List.map_append]
          exact append_congr
            (project_elimReseedOutputs d v premiseTokens _)
            (project_introReseedOutputs d v premiseTokens _))
    intro route
    simp

lemma project_witnessableAt (d : Graph) (v : Vertex)
    (tokens : List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (witnessableAt d v tokens))
      (WitnessableRoutes d v (projectFlowPairs tokens)) := by
  intro route
  unfold projectFlowPairs witnessableAt WitnessableRoutes
  simp only [List.mem_map, List.mem_filter]
  constructor
  · rintro ⟨token, ⟨token_mem, continuable⟩, rfl⟩
    exact ⟨⟨token, token_mem, rfl⟩, continuable⟩
  · rintro ⟨⟨token, token_mem, rfl⟩, continuable⟩
    exact ⟨token, ⟨token_mem, continuable⟩, rfl⟩

lemma project_assignSlotsFrom (v : Vertex) :
    ∀ ordinal tokens,
      projectFlowPairs (assignSlotsFrom v ordinal tokens) =
        projectFlowPairs tokens := by
  intro ordinal tokens
  induction tokens generalizing ordinal with
  | nil => rfl
  | cons token tokens ih =>
      rw [assignSlotsFrom]
      change token.flowPair ::
          projectFlowPairs (assignSlotsFrom v (ordinal + 1) tokens) =
        token.flowPair :: projectFlowPairs tokens
      rw [ih]

lemma project_assignSlots (v : Vertex) (tokens : List RouteIndexedToken) :
    projectFlowPairs (assignSlots v tokens) = projectFlowPairs tokens :=
  project_assignSlotsFrom v 0 tokens

private def flowElimFrom (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) : List FlowPair :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    FlowSemantics.elimCombine d v major minor (premiseFlow minor) (premiseFlow major)

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
    flowStepFrom d premiseFlow v =
      if v.HYPOTHESIS || (predsOf d v).isEmpty then
        let ancestors := ancestorsInto d v
        if ancestors.isEmpty then [(normalizeDep [v.FORMULA], [])]
        else ancestors.map fun ancestor =>
          (normalizeDep [v.FORMULA], ancestor.COLOURS)
      else
        let core := if (elimPairsAt d v).isEmpty then
          flowIntroFrom d v premiseFlow
        else flowElimFrom d v premiseFlow ++ mixedIntroPairs d v premiseFlow
        (core ++ reseedPairs d v premiseFlow).eraseDups := by
  rfl

/-- First local correctness theorem: the operational route-keyed kernel and
the raw Flow local kernel have exactly the same projected set of outputs. -/
theorem project_stepFrom_self (d : Graph)
    (premiseTokens : Vertex → List RouteIndexedToken) (v : Vertex) :
    ListSetEq (projectFlowPairs (stepFrom d premiseTokens v))
      (flowStepFrom d (fun u => projectFlowPairs (premiseTokens u)) v) := by
  rw [stepFrom, project_assignSlots]
  rw [flowStepFrom_as_parts]
  by_cases hguard :
      (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp only [rawStepFrom, hguard, if_true]
    rw [project_sourceOutputs]
    exact Certificate.Flow.ListSetEq.refl _
  · have hguardFalse :
        (v.HYPOTHESIS || (predsOf d v).isEmpty) = false :=
      Bool.eq_false_of_not_eq_true hguard
    simp only [rawStepFrom, hguardFalse, Bool.false_eq_true, if_false]
    by_cases hpairs : (elimPairsAt d v).isEmpty = true
    · simp only [hpairs, if_true]
      apply Certificate.Flow.ListSetEq.trans
        (show ListSetEq
          (projectFlowPairs
            (introOutputs d v premiseTokens ++
              reseedOutputs d v premiseTokens))
          (flowIntroFrom d v
              (fun u => projectFlowPairs (premiseTokens u)) ++
            reseedPairs d v
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
      apply Certificate.Flow.ListSetEq.trans
        (show ListSetEq
          (projectFlowPairs
            ((elimOutputs d v premiseTokens ++
                mixedIntroOutputs d v premiseTokens) ++
              reseedOutputs d v premiseTokens))
          ((flowElimFrom d v
                (fun u => projectFlowPairs (premiseTokens u)) ++
              mixedIntroPairs d v
                (fun u => projectFlowPairs (premiseTokens u))) ++
            reseedPairs d v
              (fun u => projectFlowPairs (premiseTokens u))) from by
            rw [projectFlowPairs, List.map_append, List.map_append]
            exact append_congr
              (append_congr
                (project_elimOutputs d v premiseTokens)
                (project_mixedIntroOutputs d v premiseTokens))
              (project_reseedOutputs d v premiseTokens))
      intro route
      simp

private lemma flowElimFrom_congr (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hagrees : ∀ u, ListSetEq (left u) (right u)) :
    ListSetEq (flowElimFrom d v left) (flowElimFrom d v right) := by
  unfold flowElimFrom
  apply flatMap_congr_fun
  intro pair
  exact elimCombine_congr_raw d v pair.1 pair.2
    (hagrees pair.2) (hagrees pair.1)

private lemma flowIntroFrom_congr (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hagrees : ∀ u, ListSetEq (left u) (right u)) :
    ListSetEq (flowIntroFrom d v left) (flowIntroFrom d v right) := by
  cases hp : predsOf d v with
  | nil =>
      simp only [flowIntroFrom, hp]
      exact Certificate.Flow.ListSetEq.refl []
  | cons premise rest =>
      cases rest with
      | cons next rest =>
          simp only [flowIntroFrom, hp]
          exact Certificate.Flow.ListSetEq.refl []
      | nil =>
          cases hf : v.FORMULA with
          | atom symbol =>
              simp only [flowIntroFrom, hp, hf]
              exact Certificate.Flow.ListSetEq.refl []
          | implication alpha beta =>
              by_cases heq : premise.FORMULA = beta
              · simp only [flowIntroFrom, hp, hf, heq, if_true]
                exact flatMap_congr_list _ (hagrees premise)
              · simp only [flowIntroFrom, hp, hf, heq, if_false]
                exact Certificate.Flow.ListSetEq.refl []

lemma flowStepFrom_congr_set (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hagrees : ∀ u, ListSetEq (left u) (right u)) :
    ListSetEq (flowStepFrom d left v) (flowStepFrom d right v) := by
  rw [flowStepFrom_as_parts, flowStepFrom_as_parts]
  by_cases hguard :
      (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp only [hguard, if_true]
    exact Certificate.Flow.ListSetEq.refl _
  · have hguardFalse :
        (v.HYPOTHESIS || (predsOf d v).isEmpty) = false :=
      Bool.eq_false_of_not_eq_true hguard
    simp only [hguardFalse, Bool.false_eq_true, if_false]
    by_cases hpairs : (elimPairsAt d v).isEmpty = true
    · simp only [hpairs, if_true]
      apply eraseDups_congr
      exact append_congr (flowIntroFrom_congr d v left right hagrees)
        (reseedPairs_congr_set_of_preds d v left right fun u _ => hagrees u)
    · have hpairsFalse : (elimPairsAt d v).isEmpty = false :=
        Bool.eq_false_of_not_eq_true hpairs
      simp only [hpairsFalse, Bool.false_eq_true, if_false]
      apply eraseDups_congr
      exact append_congr
        (append_congr (flowElimFrom_congr d v left right hagrees)
          (mixedIntroPairs_congr_set_of_preds d v left right
            fun u _ => hagrees u))
        (reseedPairs_congr_set_of_preds d v left right
          fun u _ => hagrees u)

/-- Local correctness under a semantic correspondence hypothesis on every
premise carrier. -/
theorem routeIndexedLocalOutputs_setEq_flowStepFrom (d : Graph)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (premiseFlow : Vertex → List FlowPair) (v : Vertex)
    (hagrees : ∀ u,
      ListSetEq (projectFlowPairs (premiseTokens u)) (premiseFlow u)) :
    ListSetEq (projectFlowPairs (stepFrom d premiseTokens v))
      (flowStepFrom d premiseFlow v) :=
  (project_stepFrom_self d premiseTokens v).trans
    (flowStepFrom_congr_set d v _ _ hagrees)

/-- Filtering is a separate post-production phase. -/
theorem routeIndexedWitnessableLocal_setEq (d : Graph)
    (premiseTokens : Vertex → List RouteIndexedToken)
    (premiseFlow : Vertex → List FlowPair) (v : Vertex)
    (hagrees : ∀ u,
      ListSetEq (projectFlowPairs (premiseTokens u)) (premiseFlow u)) :
    ListSetEq
      (projectFlowPairs (witnessableAt d v (stepFrom d premiseTokens v)))
      (WitnessableRoutes d v (flowStepFrom d premiseFlow v)) :=
  (project_witnessableAt d v _).trans
    (witnessableRoutes_congr d v
      (routeIndexedLocalOutputs_setEq_flowStepFrom d premiseTokens
        premiseFlow v hagrees))

theorem tokensFromStored_project (c : Certificate) (v : Vertex) :
    ListSetEq (projectFlowPairs (tokensFromStored c v)) (StoredRoutes c v) := by
  have projected : projectFlowPairs (tokensFromStored c v) = StoredRoutes c v := by
    unfold projectFlowPairs tokensFromStored
    rw [List.map_map]
    change (StoredRoutes c v).zipIdx.map Prod.fst = StoredRoutes c v
    exact List.zipIdx_map_fst 0 (StoredRoutes c v)
  rw [projected]
  exact Certificate.Flow.ListSetEq.refl _

/-- Certificate records provide a finite premise carrier, but this local
evaluator recomputes the conclusion dependency with the logical kernel rather
than returning the certificate's conclusion record. -/
theorem certificateLocalOutputs_setEq_flowStepFrom
    (c : Certificate) (v : Vertex) :
    ListSetEq (projectFlowPairs (certificateLocalOutputs c v))
      (flowStepFrom c.graph (StoredRoutes c) v) :=
  routeIndexedLocalOutputs_setEq_flowStepFrom c.graph
    (tokensFromStored c) (StoredRoutes c) v (tokensFromStored_project c)

/-- General whole-graph correspondence, for every fuel and every vertex. -/
theorem routeIndexedAt_setEq_flowAt (d : Graph) :
    ∀ fuel v,
      ListSetEq (projectFlowPairs (evaluateAt d fuel v))
        (flowAt d fuel v) := by
  intro fuel
  induction fuel with
  | zero =>
      intro v
      exact Certificate.Flow.ListSetEq.refl []
  | succ fuel ih =>
      intro v
      exact routeIndexedLocalOutputs_setEq_flowStepFrom d
        (evaluateAt d fuel) (flowAt d fuel) v ih

theorem routeIndexedCanonical_setEq_flowAt (d : Graph) (v : Vertex) :
    ListSetEq (projectFlowPairs (canonicalAt d v))
      (flowAt d (stdFuel d) v) :=
  routeIndexedAt_setEq_flowAt d (stdFuel d) v

theorem routeIndexedCanonicalWitnessable_setEq (d : Graph) (v : Vertex) :
    ListSetEq (projectFlowPairs (witnessableAt d v (canonicalAt d v)))
      (WitnessableRoutes d v (flowAt d (stdFuel d) v)) :=
  (project_witnessableAt d v _).trans
    (witnessableRoutes_congr d v
      (routeIndexedCanonical_setEq_flowAt d v))

/-- Propositional discharge for the route-sensitive evaluator.  Every root has
at least one operational occurrence and every root occurrence projects to an
empty dependency and empty residual. -/
def RouteIndexedCanonicalDischarged (d : Graph) : Prop :=
  ∀ root ∈ d.NODES,
    (get_rule.outgoing root d).isEmpty = true →
      projectFlowPairs (canonicalAt d root) ≠ [] ∧
      ∀ route ∈ projectFlowPairs (canonicalAt d root),
        route.1 = [] ∧ route.2 = []

theorem routeIndexedCanonicalDischarged_of_rootsDischargedB (d : Graph)
    (discharged : rootsDischargedB d = true) :
    RouteIndexedCanonicalDischarged d := by
  unfold rootsDischargedB at discharged
  rw [List.all_eq_true] at discharged
  intro root root_mem root_is_root
  have root_check := discharged root root_mem
  simp only [root_is_root, Bool.not_true, Bool.false_or,
    Bool.and_eq_true, List.all_eq_true] at root_check
  have correspondence := routeIndexedCanonical_setEq_flowAt d root
  constructor
  · intro projected_empty
    have flow_empty : flowAt d (stdFuel d) root = [] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro route route_mem
      have projected_mem := (correspondence route).mpr route_mem
      simp [projected_empty] at projected_mem
    have flow_nonempty : flowAt d (stdFuel d) root ≠ [] := by
      simpa [List.isEmpty_iff] using root_check.1
    exact flow_nonempty flow_empty
  · intro route route_mem
    have flow_mem := (correspondence route).mp route_mem
    have route_check := root_check.2 route flow_mem
    simpa [List.isEmpty_iff] using route_check

/-- The desired certificate-to-new-evaluator bridge.  No provenance side
condition is needed: route identity is preserved by construction. -/
theorem routeIndexedCanonicalDischarged_of_strongFormation
    (d : Graph) (c : Certificate) (valid : StrongFormationValid d c) :
    RouteIndexedCanonicalDischarged d :=
  routeIndexedCanonicalDischarged_of_rootsDischargedB d
    (strongFormationValid_rootsDischargedB d c valid)


end Evaluator.RouteIndexed
