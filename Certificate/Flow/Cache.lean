import Certificate.Flow.IntroductionSuffix

open DLDS Flow Certificate.Flow.Foundation

set_option maxHeartbeats 0
set_option maxRecDepth 100000

namespace DLDS.Certificate.Flow.Cache

/-- The specification table at one recursive `flowAt` fuel layer. -/
def flowTableSpec (graph : Graph) (fuel : Nat) : FlowTable :=
  graph.NODES.map fun vertex => (vertex, flowAt graph fuel vertex)

lemma flowFromTable_spec (graph : Graph) (fuel : Nat) (vertex : Vertex) :
    flowFromTable graph fuel (flowTableSpec graph fuel) vertex =
      flowAt graph fuel vertex := by
  simpa [flowTableSpec, flowFromTable, routesFromCacheTable] using
    (routesFromCacheTable_exact graph.NODES (flowAt graph fuel) vertex)

lemma flowStepFrom_flowAt (graph : Graph) (fuel : Nat) (vertex : Vertex) :
    flowStepFrom graph (flowAt graph fuel) vertex =
      flowAt graph (fuel + 1) vertex := by
  rfl

lemma flowTableStep_spec (graph : Graph) (fuel : Nat) :
    flowTableStep graph fuel (flowTableSpec graph fuel) =
      flowTableSpec graph (fuel + 1) := by
  simp only [flowTableStep, flowTableSpec]
  apply List.map_congr_left
  intro vertex hvertex
  simp only [Prod.mk.injEq, true_and]
  change flowStepFrom graph
      (flowFromTable graph fuel (flowTableSpec graph fuel)) vertex =
    flowAt graph (fuel + 1) vertex
  rw [show flowFromTable graph fuel (flowTableSpec graph fuel) =
      flowAt graph fuel from funext (flowFromTable_spec graph fuel)]
  exact flowStepFrom_flowAt graph fuel vertex

lemma flowStepFrom_congr_of_preds (graph : Graph) (vertex : Vertex)
    (left right : Vertex → List FlowPair)
    (hagrees : ∀ premise ∈ predsOf graph vertex,
      left premise = right premise) :
    flowStepFrom graph left vertex = flowStepFrom graph right vertex := by
  by_cases hguard :
      (vertex.HYPOTHESIS || (predsOf graph vertex).isEmpty) = true
  · simp [flowStepFrom, hguard]
  · simp only [flowStepFrom, hguard, Bool.false_eq_true, if_false]
    apply congrArg List.eraseDups
    apply congrArg₂ (fun xs ys => xs ++ ys)
    · by_cases hpairs : (elimPairsAt graph vertex).isEmpty = true
      · simp only [hpairs, if_true]
        cases hpreds : predsOf graph vertex with
        | nil => rfl
        | cons premise rest =>
            cases rest with
            | nil =>
                have hpremise : premise ∈ predsOf graph vertex := by
                  simp [hpreds]
                cases vertex.FORMULA <;>
                  simp [hagrees premise hpremise]
            | cons other tail => simp
      · simp only [hpairs, Bool.false_eq_true, if_false]
        apply congrArg₂ (fun xs ys => xs ++ ys)
        · apply List.flatMap_congr
          intro pair hpair
          rcases pair with ⟨major, minor⟩
          obtain ⟨hmajor, hminor⟩ :=
            mem_preds_of_mem_elimPairsAt hpair
          rw [hagrees minor hminor, hagrees major hmajor]
        · exact mixedIntroPairs_congr_of_preds graph vertex left right hagrees
    · exact reseedPairs_congr_of_preds graph vertex left right hagrees

lemma flowTableLookup?_map_of_mem (keys : List Vertex)
    (routes : Vertex → List FlowPair) {vertex : Vertex}
    (hvertex : vertex ∈ keys) :
    flowTableLookup? (keys.map fun key => (key, routes key)) vertex =
      some (routes vertex) := by
  induction keys with
  | nil => simp at hvertex
  | cons key keys ih =>
      by_cases hkey : key = vertex
      · subst vertex
        simp [flowTableLookup?]
      · have htail : vertex ∈ keys := by
          rcases List.mem_cons.mp hvertex with hhead | htail
          · exact False.elim (hkey hhead.symm)
          · exact htail
        have hbeq : (key == vertex) = false := by simp [hkey]
        simp only [List.map_cons, flowTableLookup?, List.find?_cons, hbeq]
        change flowTableLookup? (keys.map fun key => (key, routes key)) vertex =
          some (routes vertex)
        exact ih htail

lemma flowAt_eq_of_flowTableSpec_eq (graph : Graph) {fuel : Nat}
    (htable : flowTableSpec graph (fuel + 1) = flowTableSpec graph fuel)
    {vertex : Vertex} (hvertex : vertex ∈ graph.NODES) :
    flowAt graph (fuel + 1) vertex = flowAt graph fuel vertex := by
  have hlookup := congrArg (fun table => flowTableLookup? table vertex) htable
  simpa [flowTableSpec, flowTableLookup?_map_of_mem graph.NODES
    (flowAt graph (fuel + 1)) hvertex,
    flowTableLookup?_map_of_mem graph.NODES (flowAt graph fuel) hvertex]
    using hlookup

lemma mem_predsOf_mem_nodes_of_flowTableClosedB (graph : Graph)
    (hclosed : flowTableClosedB graph = true) {vertex premise : Vertex}
    (hpremise : premise ∈ predsOf graph vertex) : premise ∈ graph.NODES := by
  obtain ⟨edge, hincoming, hstart⟩ := predsOf_has_edge hpremise
  have hedge : edge ∈ graph.EDGES :=
    mem_incoming_mem_edges vertex graph hincoming
  unfold flowTableClosedB at hclosed
  rw [List.all_eq_true] at hclosed
  have hendpoints := hclosed edge hedge
  simp only [Bool.and_eq_true] at hendpoints
  rw [← hstart]
  simpa using hendpoints.1

lemma flowTableSpec_fixed_succ (graph : Graph)
    (hclosed : flowTableClosedB graph = true) {fuel : Nat}
    (htable : flowTableSpec graph (fuel + 1) = flowTableSpec graph fuel) :
    flowTableSpec graph (fuel + 2) = flowTableSpec graph (fuel + 1) := by
  simp only [flowTableSpec]
  apply List.map_congr_left
  intro vertex hvertex
  simp only [Prod.mk.injEq, true_and]
  rw [← flowStepFrom_flowAt graph (fuel + 1) vertex,
    ← flowStepFrom_flowAt graph fuel vertex]
  apply flowStepFrom_congr_of_preds
  intro premise hpremise
  exact flowAt_eq_of_flowTableSpec_eq graph htable
    (mem_predsOf_mem_nodes_of_flowTableClosedB graph hclosed hpremise)

lemma flowTableSpec_stable_of_fixed (graph : Graph)
    (hclosed : flowTableClosedB graph = true) {fuel : Nat}
    (htable : flowTableSpec graph (fuel + 1) = flowTableSpec graph fuel) :
    ∀ extra, flowTableSpec graph (fuel + 1 + extra) =
      flowTableSpec graph (fuel + 1) := by
  have hsteps : ∀ extra,
      flowTableSpec graph (fuel + extra + 1) =
        flowTableSpec graph (fuel + extra) := by
    intro extra
    induction extra with
    | zero => simpa using htable
    | succ extra ih =>
        have hnext := flowTableSpec_fixed_succ graph hclosed ih
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hnext
  intro extra
  induction extra with
  | zero => rfl
  | succ extra ih =>
      calc
        flowTableSpec graph (fuel + 1 + (extra + 1)) =
            flowTableSpec graph (fuel + 1 + extra) := by
              simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
                hsteps (extra + 1)
        _ = flowTableSpec graph (fuel + 1) := ih

lemma flowTableInitial_eq_spec_zero (graph : Graph) :
    flowTableInitial graph = flowTableSpec graph 0 := by
  rfl

lemma flowTableClosedB_of_structuralValid (graph : Graph)
    (hstruct : structuralValid graph) : flowTableClosedB graph = true := by
  unfold flowTableClosedB
  rw [List.all_eq_true]
  intro edge hedge
  obtain ⟨hstart, hend⟩ := hstruct.2.1 edge hedge
  simp [hstart, hend]

lemma flowTableIterate_spec (graph : Graph)
    (hclosed : flowTableClosedB graph = true) : ∀ remaining fuel,
    flowTableIterate graph true remaining fuel (flowTableSpec graph fuel) =
      flowTableSpec graph (fuel + remaining) := by
  intro remaining
  induction remaining with
  | zero => intro fuel; rfl
  | succ remaining ih =>
      intro fuel
      rw [flowTableIterate, flowTableStep_spec]
      split <;> rename_i hstop
      · have hfixed : flowTableSpec graph (fuel + 1) =
            flowTableSpec graph fuel := by
          have hparts := hstop
          simp only [Bool.true_and, Bool.and_eq_true, beq_iff_eq] at hparts
          exact hparts.2
        have hstable := flowTableSpec_stable_of_fixed graph hclosed hfixed remaining
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hstable.symm
      · have hrest := ih (fuel + 1)
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hrest

lemma flowTableIterate_spec_noStop (graph : Graph) : ∀ remaining fuel,
    flowTableIterate graph false remaining fuel (flowTableSpec graph fuel) =
      flowTableSpec graph (fuel + remaining) := by
  intro remaining
  induction remaining with
  | zero => intro fuel; rfl
  | succ remaining ih =>
      intro fuel
      rw [flowTableIterate, flowTableStep_spec]
      have hrest := ih (fuel + 1)
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hrest

theorem flowTable_eq_spec (graph : Graph) :
    flowTable graph = flowTableSpec graph (stdFuel graph) := by
  unfold flowTable flowTableAt
  rw [flowTableInitial_eq_spec_zero]
  cases hclosed : flowTableClosedB graph with
  | false =>
      simpa [hclosed, stdFuel] using
        flowTableIterate_spec_noStop graph (graph.NODES.length + 1) 0
  | true =>
      simpa [hclosed, stdFuel] using
        flowTableIterate_spec graph hclosed (graph.NODES.length + 1) 0

/-- The executable table is a memoized presentation of the recursive Flow
specification at standard fuel. -/
theorem cachedFlowAt_flowTable (graph : Graph)
    (vertex : Vertex) :
    cachedFlowAt graph (flowTable graph) vertex =
      flowAt graph (stdFuel graph) vertex := by
  unfold cachedFlowAt
  rw [flowTable_eq_spec graph]
  exact flowFromTable_spec graph (stdFuel graph) vertex

theorem rawSourceRouteTable_flowTable (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.rawSourceRouteTable graph
        (flowTable graph) = sourceRouteTable graph := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.rawSourceRouteTable
    sourceRouteTable sourceRoutesForFormula
  apply List.map_congr_left
  intro formula hformula
  apply List.flatMap_congr
  intro vertex hvertex
  rw [cachedFlowAt_flowTable graph vertex]

theorem rawRoutes_sourceRouteTable (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.rawRoutes
        (buildFormulas graph) (sourceRouteTable graph) = routesOf graph := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.rawRoutes routesOf
    columnMultiplicity
  apply List.flatMap_congr
  intro entry hentry
  rcases entry with ⟨formula, column⟩
  have hzip := List.mem_zipIdx hentry
  have hcolumn : column < (buildFormulas graph).length := by
    simpa using hzip.2.1
  have hformula : formula = (buildFormulas graph)[column] := by
    simpa using hzip.2.2
  have hsource : (sourceRouteTable graph)[column]? =
      some (sourceRoutesForFormula graph formula) := by
    unfold sourceRouteTable
    rw [List.getElem?_map, List.getElem?_eq_getElem hcolumn]
    simp [hformula]
  dsimp
  rw [hsource]
  simp

theorem rawPaths_canonical (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.rawPaths graph
        (buildFormulas graph) (sourceRouteTable graph) (routesOf graph) =
      multiPathsFromFlow graph := by
  rfl

def canonicalRawCursorCache (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.RawCursorCache where
  flow := flowTable graph
  formulas := buildFormulas graph
  sourceRoutes := sourceRouteTable graph
  routes := routesOf graph
  paths := multiPathsFromFlow graph

theorem buildRawCursorCache_eq_canonical (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.buildRawCursorCache graph =
      canonicalRawCursorCache graph := by
  dsimp [DLDS.Certificate.Flow.IntroductionSuffix.buildRawCursorCache,
    canonicalRawCursorCache]
  rw [rawSourceRouteTable_flowTable graph,
    rawRoutes_sourceRouteTable graph, rawPaths_canonical graph]

theorem continuationPlansForNodesCached_canonical (graph : Graph)
    (enabledNodes : List Nat) :
    DLDS.Certificate.Flow.IntroductionSuffix.continuationPlansForNodesCached
        graph (canonicalRawCursorCache graph) enabledNodes =
      continuationPlansForNodes graph enabledNodes := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.continuationPlansForNodesCached
    continuationPlansForNodes canonicalRawCursorCache
  apply congrArg List.eraseDups
  apply List.flatMap_congr
  intro vertex hvertex
  rw [cachedFlowAt_flowTable graph vertex]

theorem continuationPlansReachableGoCached_canonical (graph : Graph)
    (enabledNodes : List Nat) :
    ∀ fuel frontier seen,
      DLDS.Certificate.Flow.IntroductionSuffix.continuationPlansReachableGoCached
          graph (canonicalRawCursorCache graph) enabledNodes fuel frontier seen =
        continuationPlansReachableGo graph enabledNodes fuel frontier seen := by
  intro fuel
  induction fuel with
  | zero => intro frontier seen; rfl
  | succ fuel ih =>
      intro frontier seen
      simp only [
        DLDS.Certificate.Flow.IntroductionSuffix.continuationPlansReachableGoCached,
        continuationPlansReachableGo]
      split
      · rfl
      · rw [continuationPlansForNodesCached_canonical graph]
        rw [ih]
        rfl

theorem continuationPlansReachableCached_canonical (graph : Graph)
    (enabledNodes : List Nat)
    (sourcePaths : MultiPathInput) :
    DLDS.Certificate.Flow.IntroductionSuffix.continuationPlansReachableCached
        graph (canonicalRawCursorCache graph) enabledNodes sourcePaths =
      continuationPlansReachable graph enabledNodes sourcePaths := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.continuationPlansReachableCached
    continuationPlansReachable initialArrivalFlags
  rw [continuationPlansReachableGoCached_canonical graph]
  rfl

theorem advancedCompositionSeedsAtLevelCached_canonical (graph : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas graph).length)
    (tokens : List (Token (buildFormulas graph).length)) :
    DLDS.Certificate.Flow.IntroductionSuffix.advancedCompositionSeedsAtLevelCached
        graph (canonicalRawCursorCache graph) enabledNodes plans paths arrivalFlags
        level numLevels layer tokens =
      advancedCompositionSeedsAtLevel graph enabledNodes plans paths arrivalFlags
        level numLevels layer tokens := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.advancedCompositionSeedsAtLevelCached
    advancedCompositionSeedsAtLevel canonicalRawCursorCache routeCount
  rfl

theorem propagateTokensComposedAtCached_canonical (graph : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (tokens : List (Token (buildFormulas graph).length))
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (level numLevels : Nat)
    (layer : GridLayer (buildFormulas graph).length)
    (outputs : List (List.Vector Bool (buildFormulas graph).length)) :
    DLDS.Certificate.Flow.IntroductionSuffix.propagateTokensComposedAtCached
        graph (canonicalRawCursorCache graph) enabledNodes plans tokens paths
        arrivalFlags level numLevels layer outputs =
      propagateTokensComposedAt graph enabledNodes plans tokens paths arrivalFlags
        level numLevels layer outputs := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.propagateTokensComposedAtCached
    propagateTokensComposedAt
  rw [advancedCompositionSeedsAtLevelCached_canonical]

theorem expectedCountsComposedGoCached_canonical (graph : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (numLevels : Nat) (layers : List (GridLayer (buildFormulas graph).length))
    (level : Nat) (tokens : List (Token (buildFormulas graph).length)) :
    DLDS.Certificate.Flow.IntroductionSuffix.expectedCountsComposedGoCached
        graph (canonicalRawCursorCache graph) enabledNodes plans paths arrivalFlags
        numLevels layers level tokens =
      expectedCountsComposedGo graph enabledNodes plans paths arrivalFlags
        numLevels layers level tokens := by
  induction layers generalizing level tokens with
  | nil => rfl
  | cons layer rest ih =>
      simp only [
        DLDS.Certificate.Flow.IntroductionSuffix.expectedCountsComposedGoCached,
        expectedCountsComposedGo]
      rw [propagateTokensComposedAtCached_canonical]
      rw [ih]

theorem evalFromLevelComposedAtCached_canonical (graph : Graph)
    (enabledNodes : List Nat) (plans : List ContinuationPlan)
    (paths : MultiPathInput) (arrivalFlags : List (List Bool))
    (numLevels level : Nat)
    (tokens : List (Token (buildFormulas graph).length))
    (layers : List (GridLayer (buildFormulas graph).length × List Nat))
    (accumulatedError : Bool) :
    DLDS.Certificate.Flow.IntroductionSuffix.evalFromLevelComposedAtCached
        graph (canonicalRawCursorCache graph) enabledNodes plans paths arrivalFlags
        numLevels level tokens layers accumulatedError =
      evalFromLevelComposedAt graph enabledNodes plans paths arrivalFlags
        numLevels level tokens layers accumulatedError := by
  induction layers generalizing level tokens accumulatedError with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨layer, expected⟩
      simp only [
        DLDS.Certificate.Flow.IntroductionSuffix.evalFromLevelComposedAtCached,
        evalFromLevelComposedAt]
      cases rest with
      | nil => rfl
      | cons next tail =>
          rw [propagateTokensComposedAtCached_canonical]
          rw [ih]

/-- The cached evaluator computes the canonical multipath evaluation exactly. -/
theorem getEvalResultMultiDLDSCached_canonical (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.getEvalResultMultiDLDSCached
        graph (canonicalRawCursorCache graph) =
      getEvalResultMultiDLDS graph (multiPathsFromFlow graph) := by
  simp only [
    DLDS.Certificate.Flow.IntroductionSuffix.getEvalResultMultiDLDSCached,
    getEvalResultMultiDLDS, getEvalResultMultiComposedAt,
    normalizeMultiPaths, routeCount,
    continuationPlansReachableCached_canonical graph,
    expectedCountsComposedWithPlans, evaluationArrivalFlags,
    initialArrivalFlags,
    expectedCountsComposedGoCached_canonical,
    evalFromLevelComposedAtCached_canonical]
  rfl

/-- End-to-end cache correctness for the executable cache builder. -/
theorem getEvalResultMultiDLDSCached_buildRawCursorCache (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.getEvalResultMultiDLDSCached
        graph
        (DLDS.Certificate.Flow.IntroductionSuffix.buildRawCursorCache graph) =
      getEvalResultMultiDLDS graph (multiPathsFromFlow graph) := by
  rw [buildRawCursorCache_eq_canonical graph]
  exact getEvalResultMultiDLDSCached_canonical graph

theorem cachedDischargedMultiB_eq_dischargedMultiB (graph : Graph) :
    DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB graph =
      dischargedMultiB graph := by
  unfold DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB
    DLDS.Certificate.Flow.IntroductionSuffix.dischargedFromRawCursorCacheB
    dischargedMultiB dischargedOnMultiB
  rw [getEvalResultMultiDLDSCached_buildRawCursorCache graph]
  rfl

theorem rawCursorReport_discharged_eq_dischargedMultiB (graph : Graph) :
    (DLDS.Certificate.Flow.IntroductionSuffix.rawCursorReport graph).discharged =
      dischargedMultiB graph := by
  calc
    (DLDS.Certificate.Flow.IntroductionSuffix.rawCursorReport graph).discharged =
        DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB graph := by
          rfl
    _ = dischargedMultiB graph :=
      cachedDischargedMultiB_eq_dischargedMultiB graph

theorem dischargedMultiB_of_cached (graph : Graph)
    (hcached :
      DLDS.Certificate.Flow.IntroductionSuffix.cachedDischargedMultiB graph =
        true) : dischargedMultiB graph = true := by
  rw [cachedDischargedMultiB_eq_dischargedMultiB graph] at hcached
  exact hcached

theorem dischargedMultiB_of_rawCursorReport (graph : Graph)
    (hcached :
      (DLDS.Certificate.Flow.IntroductionSuffix.rawCursorReport graph).discharged =
        true) : dischargedMultiB graph = true := by
  rw [rawCursorReport_discharged_eq_dischargedMultiB graph] at hcached
  exact hcached

end DLDS.Certificate.Flow.Cache
