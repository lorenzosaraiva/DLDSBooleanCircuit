import Infrastructure.SelfCertifying

open Semantic FlowSpec CertificateModel SelfCertifying

namespace Semantic.PipelineClosedCertificateSupport

def advanceIntro (edge : Deduction) (routes : List FlowPair) : List FlowPair :=
  match edge.END.FORMULA with
  | Formula.implication alpha _ =>
      (routes.filterMap fun route =>
        (consume edge route.2).map fun tail =>
          (depRemove route.1 alpha, tail)).eraseDups
  | _ => []

def suffixCertificateData : List Deduction → List FlowPair →
    List EdgeRouteWitness × List FlowPair
  | [], routes => ([], routes)
  | current :: restEdges, routes =>
      let witnesses := routes.filter
        (fun route => headColour route.2 == current.COLOUR) |>.map fun route =>
          { edge := current, dependency := route.1, residual := route.2 }
      let rest := suffixCertificateData restEdges (advanceIntro current routes)
      (witnesses ++ rest.1, rest.2)

def checkHistogram (certificate : ExpandedCertificate) : List (String × Nat) :=
  let names := (failuresMemo certificate).map (·.check)
  names.eraseDups.map fun name => (name, (names.filter (· == name)).length)

def checkHistogramFrom (found : List Failure) : List (String × Nat) :=
  let names := found.map (·.check)
  names.eraseDups.map fun name => (name, (names.filter (· == name)).length)

structure CertificateReport where
  positions : Nat
  witnesses : Nat
  terminals : Nat
  failuresByCheck : List (String × Nat)
  failureCount : Nat
  expanded : Bool
  localChecks : Bool
  deriving BEq, DecidableEq, Repr

def certificateReport (graph : Graph) (certificate : ExpandedCertificate) :
    CertificateReport :=
  let cache := buildCertificateCache certificate
  let found := failuresCached certificate cache
  { positions := positionCount certificate
    witnesses := certificate.edgeRoutes.length
    terminals := certificate.terminalOutputs.length
    failuresByCheck := checkHistogramFrom found
    failureCount := found.length
    expanded := expandedCertCachedB graph certificate cache
    localChecks := localChecksCachedB graph certificate cache }

structure FlowRouteReport where
  totalRoutes : Nat
  zeroFlowNodes : Nat
  maximumNode : Nat
  maximumRoutes : Nat
  focusNode : Option Nat
  focusRoutes : Option Nat
  deriving BEq, DecidableEq, Repr



def flowRouteReport (graph : Graph) (focusNode : Option Nat := none) :
    FlowRouteReport :=
  let counts := (flowTable graph).map fun entry =>
    (entry.1.NUMBER, entry.2.length)
  let maximum := counts.foldl (fun current selection =>
    if current.2 < selection.2 then selection else current) (0, 0)
  { totalRoutes := (counts.map (·.2)).sum
    zeroFlowNodes := (counts.filter fun entry => entry.2 = 0).length
    maximumNode := maximum.1
    maximumRoutes := maximum.2
    focusNode
    focusRoutes := focusNode.bind fun number =>
      (counts.find? fun entry => entry.1 = number).map (·.2) }

structure RawCursorReport where
  routeCount : Nat
  slotOK : Bool
  discharged : Bool
  noError : Bool
  cursorRouteHeadCoverage : Bool
  cursorFlowTailClosure : Bool
  deriving BEq, DecidableEq, Repr

structure RawCursorCache where
  flow : FlowTable
  formulas : List Formula
  sourceRoutes : List (List SourceRoute)
  routes : List (Nat × Nat)
  paths : MultiPathInput

def rawSourceRouteTable (graph : Graph) (flow : FlowTable) :
    List (List SourceRoute) :=
  let routesAt := cachedFlowAt graph flow
  let formulas := buildFormulas graph
  formulas.map fun formula =>
    (graph.NODES.filter fun v =>
      v.HYPOTHESIS && decide (v.FORMULA = formula)).flatMap fun v =>
        (routesAt v).map fun route => (v, route)

def rawRoutes (formulas : List Formula)
    (sourceRoutes : List (List SourceRoute)) : List (Nat × Nat) :=
  formulas.zipIdx.flatMap fun (_, column) =>
    (List.range (max 1 (sourceRoutes[column]?.getD []).length)).map fun index =>
      (column, index)

def rawPaths (graph : Graph) (formulas : List Formula)
    (sourceRoutes : List (List SourceRoute)) (routes : List (Nat × Nat)) :
    MultiPathInput :=
  let numSteps := (buildGridFromDLDS graph).length - 1
  let maxLevel := (graph.NODES.map (·.LEVEL)).foldl max 0
  routes.map fun (column, index) =>
    match formulas[column]? with
    | none => List.replicate numSteps (0, 0)
    | some formula =>
        match sourceRoutes[column]?.bind fun entries => entries[index]? with
        | none => List.replicate numSteps (0, 0)
        | some (v, route) =>
            let delay := maxLevel - v.LEVEL
            List.replicate delay (column + 1, 0) ++
              routeFromFlowC graph formulas (numSteps - delay) v.LEVEL
                formula route.2

def buildRawCursorCache (graph : Graph) : RawCursorCache :=
  let flow := flowTable graph
  let formulas := buildFormulas graph
  let sourceRoutes := rawSourceRouteTable graph flow
  let routes := rawRoutes formulas sourceRoutes
  let paths := rawPaths graph formulas sourceRoutes routes
  { flow, formulas, sourceRoutes, routes, paths }

def continuationPlansForNodesCached (graph : Graph) (cache : RawCursorCache)
    (enabledNodes : List Nat) : List ContinuationPlan :=
  ((graph.NODES.filter fun v => enabledNodes.contains v.NUMBER).flatMap fun v =>
    (WitnessableRoutes graph v (cachedFlowAt graph cache.flow v)).filterMap fun route =>
      if route.2.isEmpty then none else some { vertex := v, route }).eraseDups

def continuationPlansReachableGoCached (graph : Graph) (cache : RawCursorCache)
    (enabledNodes : List Nat) :
    Nat → List Vertex → List Vertex → List ContinuationPlan
  | 0, _, _ => []
  | fuel + 1, frontier, seen =>
      let fresh := frontier.filter fun v =>
        enabledNodes.contains v.NUMBER && !seen.contains v
      if fresh.isEmpty then []
      else
        let newPlans := continuationPlansForNodesCached graph cache
          (fresh.map fun v => v.NUMBER)
        let newPaths := compositionPathsFromPlans graph newPlans
        let newFlags := compositionArrivalFlagsFromPlans graph newPlans
        let newOrigins := newPlans.map fun plan =>
          cache.formulas.idxOf plan.vertex.FORMULA
        let next := stoppedArrivalSites graph newOrigins newPaths newFlags
        (newPlans ++ continuationPlansReachableGoCached graph cache enabledNodes fuel next
          (seen ++ fresh)).eraseDups

def continuationPlansReachableCached (graph : Graph) (cache : RawCursorCache)
    (enabledNodes : List Nat) (sourcePaths : MultiPathInput) :
    List ContinuationPlan :=
  let sourceFlags := arrivalFlagsForPaths graph (cache.routes.map Prod.fst) sourcePaths
  let initialSites := stoppedArrivalSites graph (cache.routes.map Prod.fst)
    sourcePaths sourceFlags
  continuationPlansReachableGoCached graph cache enabledNodes
    (graph.NODES.length + 1) initialSites []

def advancedCompositionSeedsAtLevelCached (graph : Graph)
    (cache : RawCursorCache) (enabledNodes : List Nat)
    (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas graph).length)
    (tokens : List (Token (buildFormulas graph).length)) :
      List (Token (buildFormulas graph).length) :=
  let stepIdx := numLevels - level - 1
  plans.zipIdx.filterMap fun (plan, planIdx) =>
    let v := plan.vertex
    let col := cache.formulas.idxOf v.FORMULA
    let path := fullPathForRoute graph v plan.route
    if enabledNodes.contains v.NUMBER && v.LEVEL == level &&
        !(get_rule.outgoing v graph).isEmpty &&
        compositionPlanCompatibleB graph paths arrivalFlags stepIdx layer tokens plan
    then
      if hStep : level > 0 ∧ stepIdx < path.length then
        let step := path.get ⟨stepIdx, hStep.2⟩
        if step.1 = 0 then none
        else some
          { origin_column := cache.routes.length + planIdx
            source_column := col
            current_level := level - 1
            current_column := step.1 - 1
            dep_vector := depVector (buildFormulas graph) plan.route.1
            input_label := step.2 }
      else none
    else none

def propagateTokensComposedAtCached (graph : Graph) (cache : RawCursorCache)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (tokens : List (Token (buildFormulas graph).length)) (paths : MultiPathInput)
    (arrivalFlags : List (List Bool)) (level numLevels : Nat)
    (layer : GridLayer (buildFormulas graph).length)
    (outputs : List (List.Vector Bool (buildFormulas graph).length)) :
      List (Token (buildFormulas graph).length) :=
  let ordinary := propagate_tokens tokens paths level numLevels outputs
  ordinary ++ advancedCompositionSeedsAtLevelCached graph cache enabledNodes plans paths
    arrivalFlags level numLevels layer tokens

def expectedCountsComposedGoCached (graph : Graph) (cache : RawCursorCache)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (numLevels : Nat) :
    List (GridLayer (buildFormulas graph).length) → Nat →
      List (Token (buildFormulas graph).length) → List (List Nat)
  | [], _, _ => []
  | layer :: rest, level, tokens =>
      let counts := tokenCounts layer.nodes.length tokens
      let evaluated := evaluate_layer_multi layer tokens counts
      counts :: expectedCountsComposedGoCached graph cache enabledNodes plans paths
        arrivalFlags numLevels rest (level - 1)
        (propagateTokensComposedAtCached graph cache enabledNodes plans tokens paths
          arrivalFlags level numLevels layer evaluated.1)

def evalFromLevelComposedAtCached (graph : Graph) (cache : RawCursorCache)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (numLevels : Nat) :
    Nat → List (Token (buildFormulas graph).length) →
      List (GridLayer (buildFormulas graph).length × List Nat) → Bool →
      List (List.Vector Bool (buildFormulas graph).length) × Bool
  | _, _, [], accumulatedError =>
      ((List.range (buildFormulas graph).length).map fun _ =>
          List.Vector.replicate (buildFormulas graph).length false,
        accumulatedError)
  | level, tokens, (layer, expected) :: rest, accumulatedError =>
      let evaluated := evaluate_layer_multi layer tokens expected
      match rest with
      | [] => (evaluated.1, accumulatedError || evaluated.2)
      | _ =>
          evalFromLevelComposedAtCached graph cache enabledNodes plans paths
            arrivalFlags numLevels (level - 1)
            (propagateTokensComposedAtCached graph cache enabledNodes plans tokens paths
              arrivalFlags level numLevels layer evaluated.1)
            rest (accumulatedError || evaluated.2)

def getEvalResultMultiDLDSCached (graph : Graph) (cache : RawCursorCache) :
    List (List.Vector Bool (buildFormulas graph).length) × Bool :=
  let enabledNodes := graph.NODES.map fun v => v.NUMBER
  let sourcePaths := (List.range cache.routes.length).map fun routeIndex =>
    normalizeReadPath graph (cache.paths.getD routeIndex [])
  let layers := buildGridFromDLDS graph
  let plans := continuationPlansReachableCached graph cache enabledNodes cache.paths
  let paths := evaluationPathsWithComposition graph plans sourcePaths
  let flags := arrivalFlagsForPaths graph (cache.routes.map Prod.fst) sourcePaths ++
    compositionArrivalFlagsFromPlans graph plans
  let expectedPaths := evaluationPathsWithComposition graph plans cache.paths
  let expectedFlags := arrivalFlagsForPaths graph (cache.routes.map Prod.fst) cache.paths ++
    compositionArrivalFlagsFromPlans graph plans
  let tokens := initialize_tokens_multi cache.routes (initialVectorsFromDLDS graph)
    layers.length
  let expected := expectedCountsComposedGoCached graph cache enabledNodes plans
    expectedPaths expectedFlags layers.length layers (layers.length - 1) tokens
  evalFromLevelComposedAtCached graph cache enabledNodes plans paths flags layers.length
    (layers.length - 1) tokens (layers.zip expected) false

def slotCheckCachedB (graph : Graph) (cache : RawCursorCache)
    (slot : Nat) (steps : List (Nat × Nat)) : Bool :=
  let routeIndex := cache.routes.getD slot (0, 0)
  match cache.formulas[routeIndex.1]? with
  | none => steps.all fun step => step == (0, 0)
  | some formula =>
      match cache.sourceRoutes[routeIndex.1]?.bind fun routes =>
          routes[routeIndex.2]? with
      | none => steps.all fun step => step == (0, 0)
      | some (v, route) =>
          admHypColumnCB graph cache.formulas routeIndex.1 v.LEVEL formula route.2
            ((graph.NODES.map (·.LEVEL)).foldl max 0 - v.LEVEL) steps

def cursorRouteHeadCoverageAtCachedB (graph : Graph)
    (cache : RawCursorCache) (v : Vertex) : Bool :=
  let outgoing := get_rule.outgoing v graph
  let routes := cachedFlowAt graph cache.flow v
  (routes.all fun route =>
      exhaustedFlowArrivalAtB graph v route ||
        outgoing.any fun edge => edge.COLOUR == headColour route.2) &&
  (outgoing.all fun edge =>
    routes.any fun route => headColour route.2 == edge.COLOUR)

def cursorPairFlowTailClosureAtCachedB (graph : Graph)
    (cache : RawCursorCache) (w major minor : Vertex) : Bool :=
  let routes := cachedFlowAt graph cache.flow
  ((routes minor).all fun route =>
    (edgesBetween graph minor w).all fun edge =>
      match consume edge route.2 with
      | none => true
      | some tail =>
          exhaustedConsumedTailAtB graph w tail ||
            (routes major).any fun partner =>
              (edgesBetween graph major w).any fun partnerEdge =>
                consume partnerEdge partner.2 == some tail ||
                  reseedPairSupportsB graph w route edge partner partnerEdge) &&
  ((routes major).all fun route =>
    (edgesBetween graph major w).all fun edge =>
      match consume edge route.2 with
      | none => true
      | some tail =>
          exhaustedConsumedTailAtB graph w tail ||
            (routes minor).any fun partner =>
              (edgesBetween graph minor w).any fun partnerEdge =>
                consume partnerEdge partner.2 == some tail ||
                  reseedPairSupportsB graph w partner partnerEdge route edge)

def cursorCarrierAccountedAtCachedB (graph : Graph)
    (cache : RawCursorCache) (w u : Vertex) : Bool :=
  match w.FORMULA with
  | Formula.implication _ beta =>
      let routes := cachedFlowAt graph cache.flow
      u.FORMULA == beta &&
        (routes u).all fun route =>
          (edgesBetween graph u w).all fun edge =>
            match consume edge route.2 with
            | none => true
            | some tail =>
                exhaustedConsumedTailAtB graph w tail ||
                  (routes w).any fun result => result.2 == tail
  | _ => false

def cursorFlowTailClosureAtCachedB (graph : Graph)
    (cache : RawCursorCache) (w : Vertex) : Bool :=
  match elimPairsAt graph w with
  | [] => true
  | pairs =>
      (predsOf graph w).all (fun u =>
        elimPremiseAtB graph w u ||
          cursorCarrierAccountedAtCachedB graph cache w u) &&
      pairs.all fun (major, minor) =>
        cursorPairFlowTailClosureAtCachedB graph cache w major minor

def dischargedFromRawCursorCacheB (graph : Graph) (cache : RawCursorCache) : Bool :=
  let evaluation := getEvalResultMultiDLDSCached graph cache
  match evaluation.1[goalColumn graph]? with
  | none => true
  | some vector => vector.toList.all fun bit => !bit

def cachedDischargedMultiB (graph : Graph) : Bool :=
  dischargedFromRawCursorCacheB graph (buildRawCursorCache graph)



def rawCursorReport (graph : Graph) : RawCursorReport :=
  let cache := buildRawCursorCache graph
  { routeCount := cache.routes.length
    slotOK := (List.range cache.routes.length).all fun slot =>
      slotCheckCachedB graph cache slot (cache.paths.getD slot [])
    discharged := dischargedFromRawCursorCacheB graph cache
    noError := !(getEvalResultMultiDLDSCached graph cache).2
    cursorRouteHeadCoverage := graph.NODES.all fun v =>
      (get_rule.outgoing v graph).isEmpty ||
        cursorRouteHeadCoverageAtCachedB graph cache v
    cursorFlowTailClosure := graph.NODES.all
      (cursorFlowTailClosureAtCachedB graph cache) }

end Semantic.PipelineClosedCertificateSupport
