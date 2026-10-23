import Infrastructure.EndToEnd









open Semantic FlowSpec

namespace CertificateModel

structure EdgeRouteWitness where
  edge : Deduction
  dependency : Dep
  residual : ColourPath
  deriving Repr, BEq, DecidableEq

structure TerminalOutput where
  node : Vertex
  dependency : Dep
  residual : ColourPath
  deriving Repr, BEq, DecidableEq

structure ExpandedCertificate where
  graph : Graph
  edgeRoutes : List EdgeRouteWitness
  terminalOutputs : List TerminalOutput
  deriving Repr

structure Failure where
  check : String
  position : String
  deriving Repr, BEq

def routesAt (d : Graph) (v : Vertex) : List FlowPair :=
  flowAt d (stdFuel d) v

def routesAtCached (d : Graph) (table : FlowTable) (v : Vertex) : List FlowPair :=
  cachedFlowAt d table v

def witnessForEdgeFrom (routes : Vertex → List FlowPair)
    (edge : Deduction) : List EdgeRouteWitness :=
  (routes edge.START).filter
    (fun route => headColour route.2 == edge.COLOUR) |>.map
      (fun route => { edge, dependency := route.1, residual := route.2 })

def witnessForEdge? (d : Graph) (edge : Deduction) : List EdgeRouteWitness :=
  witnessForEdgeFrom (routesAt d) edge

def generateCertificateFromFlowTable (d : Graph) (table : FlowTable) :
    ExpandedCertificate :=
  let routes := routesAtCached d table
  { graph := d
    edgeRoutes := d.EDGES.flatMap (witnessForEdgeFrom routes)
    terminalOutputs := d.NODES.flatMap fun v =>
      if (get_rule.outgoing v d).isEmpty then
        (routes v).map fun route =>
          { node := v, dependency := route.1, residual := route.2 }
      else [] }

def generateCertificate (d : Graph) : ExpandedCertificate :=
  let table := flowTable d
  generateCertificateFromFlowTable d table

def edgeWitnessesFor (c : ExpandedCertificate) (edge : Deduction) :
    List EdgeRouteWitness :=
  c.edgeRoutes.filter fun witness => witness.edge == edge


def witnessCoverageAtB (c : ExpandedCertificate) (edge : Deduction) : Bool :=
  !(edgeWitnessesFor c edge).isEmpty

def outputsAt (c : ExpandedCertificate) (v : Vertex) : List FlowPair :=
  let edgeOutputs := c.edgeRoutes.filterMap fun witness =>
    if witness.edge.START == v then
      some (witness.dependency, witness.residual)
    else none
  let terminalOutputs := c.terminalOutputs.filterMap fun output =>
    if output.node == v then some (output.dependency, output.residual) else none
  (edgeOutputs ++ terminalOutputs).eraseDups

def setEqB {α : Type} [BEq α] (xs ys : List α) : Bool :=
  xs.all (fun x => ys.contains x) && ys.all (fun y => xs.contains y)

def sourceExpected (d : Graph) (v : Vertex) : List FlowPair :=
  let ancestors := ancestorsInto d v
  if ancestors.isEmpty then [([v.FORMULA], [])]
  else ancestors.map fun path => ([v.FORMULA], path.COLOURS)

def elimExpected (c : ExpandedCertificate) (w major minor : Vertex) :
    List FlowPair :=
  elimCombine c.graph w major minor (outputsAt c minor) (outputsAt c major)
    |>.eraseDups

def elimExpectedMany (c : ExpandedCertificate) (w : Vertex)
    (pairs : List (Vertex × Vertex)) : List FlowPair :=
  (pairs.flatMap fun (major, minor) =>
    elimCombine c.graph w major minor (outputsAt c minor) (outputsAt c major))
    |>.eraseDups

def introExpected (c : ExpandedCertificate) (w premise : Vertex)
    (alpha : Formula) : List FlowPair :=
  ((outputsAt c premise).flatMap fun route =>
    (edgesBetween c.graph premise w).filterMap fun edge =>
          (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)).eraseDups

def mixedIntroExpected (c : ExpandedCertificate) (w : Vertex) : List FlowPair :=
  mixedIntroPairs c.graph w (outputsAt c) |>.eraseDups

def reseedExpected (c : ExpandedCertificate) (v : Vertex) : List FlowPair :=
  reseedPairs c.graph v (outputsAt c)

inductive LocalRuleShape where
  | source
  | elim (major minor : Vertex)
  | elimMulti (pairs : List (Vertex × Vertex))
  | intro (premise : Vertex) (alpha beta : Formula)
  | invalid
  deriving Repr, BEq

@[simp] lemma LocalRuleShape.elimMulti_ne_elim (pairs : List (Vertex × Vertex))
    (major minor : Vertex) :
    LocalRuleShape.elimMulti pairs ≠ LocalRuleShape.elim major minor := by
  intro impossible
  cases impossible

@[simp] lemma LocalRuleShape.elimMulti_ne_intro (pairs : List (Vertex × Vertex))
    (premise : Vertex) (alpha beta : Formula) :
    LocalRuleShape.elimMulti pairs ≠
      LocalRuleShape.intro premise alpha beta := by
  intro impossible
  cases impossible

def localShape (d : Graph) (v : Vertex) : LocalRuleShape :=
  if v.HYPOTHESIS || (predsOf d v).isEmpty then .source
  else
    match elimPairsAt d v with
    | [(major, minor)] => .elim major minor
    | [] =>
        match predsOf d v, v.FORMULA with
        | [premise], Formula.implication alpha beta =>
            if premise.FORMULA == beta then .intro premise alpha beta
            else .invalid
        | _, _ => .invalid
    | pairs => .elimMulti pairs

def localExpected (c : ExpandedCertificate) (v : Vertex) : List FlowPair :=
  match localShape c.graph v with
  | .source => sourceExpected c.graph v
  | .elim major minor =>
      (elimExpected c v major minor ++ mixedIntroExpected c v ++
        reseedExpected c v).eraseDups
  | .elimMulti pairs =>
      (elimExpectedMany c v pairs ++ mixedIntroExpected c v ++
        reseedExpected c v).eraseDups
  | .intro premise alpha _ =>
      (introExpected c v premise alpha ++ reseedExpected c v).eraseDups
  | .invalid => []




def localExpectedFromOutputs (c : ExpandedCertificate)
    (stored : Vertex → List FlowPair) (v : Vertex) : List FlowPair :=
  match localShape c.graph v with
  | .source => sourceExpected c.graph v
  | .elim major minor =>
      (elimCombine c.graph v major minor (stored minor) (stored major) ++
        mixedIntroPairs c.graph v stored ++ reseedPairs c.graph v stored).eraseDups
  | .elimMulti pairs =>
      ((pairs.flatMap fun (major, minor) =>
          elimCombine c.graph v major minor (stored minor) (stored major)) ++
        mixedIntroPairs c.graph v stored ++ reseedPairs c.graph v stored).eraseDups
  | .intro premise alpha _ =>
      (((stored premise).flatMap fun route =>
          (edgesBetween c.graph premise v).filterMap fun edge =>
            (consume edge route.2).map fun tail =>
              (depRemove route.1 alpha, tail)) ++
        reseedPairs c.graph v stored).eraseDups
  | .invalid => []



def witnessableExpected (c : ExpandedCertificate) (v : Vertex) : List FlowPair :=
  WitnessableRoutes c.graph v (localExpected c v)







def witnessFaithfulAtB (c : ExpandedCertificate)
    (witness : EdgeRouteWitness) : Bool :=
  let matchingRoutes := (localExpected c witness.edge.START).filter fun route =>
    route.2 == witness.residual
  matchingRoutes.isEmpty || matchingRoutes.any fun route =>
    depSetEq route.1 witness.dependency

def productionAtB (c : ExpandedCertificate) (v : Vertex) : Bool :=
  match localShape c.graph v with
  | .invalid => false
  | _ => setEqB (outputsAt c v) (witnessableExpected c v)

def coverageAtB (c : ExpandedCertificate) (v : Vertex) : Bool :=
  let outgoing := get_rule.outgoing v c.graph
  outgoing.all (fun edge =>
    let witnesses := edgeWitnessesFor c edge
    !witnesses.isEmpty &&
      setEqB (witnesses.map fun witness =>
          (witness.dependency, witness.residual))
        ((outputsAt c v).filter fun route =>
          headColour route.2 == edge.COLOUR)) &&
  (c.edgeRoutes.filter fun witness => witness.edge.START == v).all
    (fun witness => outgoing.contains witness.edge)

def consumedWitnessTails (c : ExpandedCertificate) (start finish : Vertex) :
    List ColourPath :=
  ((edgesBetween c.graph start finish).flatMap fun edge =>
    (edgeWitnessesFor c edge).filterMap fun witness =>
      consume edge witness.residual).eraseDups




def continuableConsumedWitnessTails (c : ExpandedCertificate)
    (start finish : Vertex) : List ColourPath :=
  (consumedWitnessTails c start finish).filter fun tail =>
    routeContinuableAtB c.graph finish tail

abbrev EdgeWitnessTable := List (Deduction × List EdgeRouteWitness)
abbrev TailTable := List ((Vertex × Vertex) × List ColourPath)

structure CertificateCache where
  edgeWitnesses : EdgeWitnessTable
  outputs : FlowTable
  localExpectations : FlowTable
  witnessableExpectations : FlowTable
  continuableTails : TailTable

def edgeWitnessTableLookup? (table : EdgeWitnessTable) (edge : Deduction) :
    Option (List EdgeRouteWitness) :=
  (table.find? fun entry => entry.1 == edge).map (fun entry => entry.2)

def tailTableLookup? (table : TailTable) (start finish : Vertex) :
    Option (List ColourPath) :=
  (table.find? fun entry => entry.1 == (start, finish)).map (fun entry => entry.2)

def routesFromCacheTable (table : FlowTable) (fallback : Vertex → List FlowPair)
    (v : Vertex) : List FlowPair :=
  match flowTableLookup? table v with
  | some routes => routes
  | none => fallback v

def witnessesFromCacheTable (table : EdgeWitnessTable)
    (fallback : Deduction → List EdgeRouteWitness) (edge : Deduction) :
    List EdgeRouteWitness :=
  match edgeWitnessTableLookup? table edge with
  | some witnesses => witnesses
  | none => fallback edge

def tailsFromCacheTable (table : TailTable)
    (fallback : Vertex → Vertex → List ColourPath) (start finish : Vertex) :
    List ColourPath :=
  match tailTableLookup? table start finish with
  | some tails => tails
  | none => fallback start finish

def consumedWitnessTailsFrom (c : ExpandedCertificate)
    (witnesses : Deduction → List EdgeRouteWitness) (start finish : Vertex) :
    List ColourPath :=
  ((edgesBetween c.graph start finish).flatMap fun edge =>
    (witnesses edge).filterMap fun witness =>
      consume edge witness.residual).eraseDups

def continuableConsumedWitnessTailsFrom (c : ExpandedCertificate)
    (witnesses : Deduction → List EdgeRouteWitness) (start finish : Vertex) :
    List ColourPath :=
  (consumedWitnessTailsFrom c witnesses start finish).filter fun tail =>
    routeContinuableAtB c.graph finish tail




def buildCertificateCache (c : ExpandedCertificate) : CertificateCache :=
  let edgeKeys := c.graph.EDGES.eraseDups
  let edgeWitnesses : EdgeWitnessTable := edgeKeys.map fun edge =>
    (edge, edgeWitnessesFor c edge)
  let witnesses := witnessesFromCacheTable edgeWitnesses (edgeWitnessesFor c)
  let outputs : FlowTable := c.graph.NODES.map fun v => (v, outputsAt c v)
  let localExpectations : FlowTable := c.graph.NODES.map fun v =>
    (v, localExpected c v)
  let localRoutes := routesFromCacheTable localExpectations (localExpected c)
  let witnessableExpectations : FlowTable := c.graph.NODES.map fun v =>
    (v, WitnessableRoutes c.graph v (localRoutes v))
  let pairKeys := (c.graph.EDGES.map fun edge =>
    (edge.START, edge.END)).eraseDups
  let continuableTails : TailTable := pairKeys.map fun pair =>
    (pair, continuableConsumedWitnessTailsFrom c witnesses pair.1 pair.2)
  { edgeWitnesses, outputs, localExpectations, witnessableExpectations,
    continuableTails }

def cachedEdgeWitnessesFor (c : ExpandedCertificate) (cache : CertificateCache)
    (edge : Deduction) : List EdgeRouteWitness :=
  witnessesFromCacheTable cache.edgeWitnesses (edgeWitnessesFor c) edge

def cachedOutputsAt (c : ExpandedCertificate) (cache : CertificateCache)
    (v : Vertex) : List FlowPair :=
  routesFromCacheTable cache.outputs (outputsAt c) v

def cachedLocalExpected (c : ExpandedCertificate) (cache : CertificateCache)
    (v : Vertex) : List FlowPair :=
  routesFromCacheTable cache.localExpectations (localExpected c) v

def cachedWitnessableExpected (c : ExpandedCertificate) (cache : CertificateCache)
    (v : Vertex) : List FlowPair :=
  routesFromCacheTable cache.witnessableExpectations (witnessableExpected c) v

def cachedContinuableConsumedWitnessTails (c : ExpandedCertificate)
    (cache : CertificateCache) (start finish : Vertex) : List ColourPath :=
  tailsFromCacheTable cache.continuableTails
    (continuableConsumedWitnessTails c) start finish

lemma routesFromCacheTable_exact (keys : List Vertex)
    (fallback : Vertex → List FlowPair) (v : Vertex) :
    routesFromCacheTable (keys.map fun key => (key, fallback key)) fallback v =
      fallback v := by
  induction keys with
  | nil => simp [routesFromCacheTable, flowTableLookup?]
  | cons key keys ih =>
      by_cases h : key = v
      · subst v
        simp [routesFromCacheTable, flowTableLookup?]
      · have hbeq : (key == v) = false := by simp [h]
        simp only [List.map_cons, routesFromCacheTable, flowTableLookup?,
          List.find?_cons, hbeq, Option.map]
        change routesFromCacheTable
          (keys.map fun key => (key, fallback key)) fallback v = fallback v
        exact ih

lemma witnessesFromCacheTable_exact (keys : List Deduction)
    (fallback : Deduction → List EdgeRouteWitness) (edge : Deduction) :
    witnessesFromCacheTable
        (keys.map fun key => (key, fallback key)) fallback edge =
      fallback edge := by
  induction keys with
  | nil => simp [witnessesFromCacheTable, edgeWitnessTableLookup?]
  | cons key keys ih =>
      by_cases h : key = edge
      · subst edge
        simp [witnessesFromCacheTable, edgeWitnessTableLookup?]
      · have hbeq : (key == edge) = false := by simp [h]
        simp only [List.map_cons, witnessesFromCacheTable,
          edgeWitnessTableLookup?, List.find?_cons, hbeq, Option.map]
        change witnessesFromCacheTable
          (keys.map fun key => (key, fallback key)) fallback edge = fallback edge
        exact ih

lemma tailsFromCacheTable_exact (keys : List (Vertex × Vertex))
    (fallback : Vertex → Vertex → List ColourPath) (start finish : Vertex) :
    tailsFromCacheTable
        (keys.map fun key => (key, fallback key.1 key.2)) fallback start finish =
      fallback start finish := by
  induction keys with
  | nil => simp [tailsFromCacheTable, tailTableLookup?]
  | cons key keys ih =>
      by_cases h : key = (start, finish)
      · subst key
        simp [tailsFromCacheTable, tailTableLookup?]
      · have hbeq : (key == (start, finish)) = false := by simp [h]
        simp only [List.map_cons, tailsFromCacheTable, tailTableLookup?,
          List.find?_cons, hbeq, Option.map]
        change tailsFromCacheTable
          (keys.map fun key => (key, fallback key.1 key.2)) fallback
            start finish = fallback start finish
        exact ih

@[simp] theorem cachedEdgeWitnessesFor_buildCertificateCache
    (c : ExpandedCertificate) (edge : Deduction) :
    cachedEdgeWitnessesFor c (buildCertificateCache c) edge =
      edgeWitnessesFor c edge := by
  simp [cachedEdgeWitnessesFor, buildCertificateCache,
    witnessesFromCacheTable_exact]

@[simp] theorem cachedOutputsAt_buildCertificateCache
    (c : ExpandedCertificate) (v : Vertex) :
    cachedOutputsAt c (buildCertificateCache c) v = outputsAt c v := by
  simp [cachedOutputsAt, buildCertificateCache, routesFromCacheTable_exact]

@[simp] theorem cachedLocalExpected_buildCertificateCache
    (c : ExpandedCertificate) (v : Vertex) :
    cachedLocalExpected c (buildCertificateCache c) v = localExpected c v := by
  simp [cachedLocalExpected, buildCertificateCache, routesFromCacheTable_exact]

@[simp] theorem cachedWitnessableExpected_buildCertificateCache
    (c : ExpandedCertificate) (v : Vertex) :
    cachedWitnessableExpected c (buildCertificateCache c) v =
      witnessableExpected c v := by
  simp only [cachedWitnessableExpected, buildCertificateCache]
  have hlocal : routesFromCacheTable
      (c.graph.NODES.map fun key => (key, localExpected c key))
        (localExpected c) = localExpected c := by
    funext key
    exact routesFromCacheTable_exact c.graph.NODES (localExpected c) key
  rw [hlocal]
  change routesFromCacheTable
    (c.graph.NODES.map fun key => (key, witnessableExpected c key))
      (witnessableExpected c) v = witnessableExpected c v
  exact routesFromCacheTable_exact c.graph.NODES (witnessableExpected c) v

@[simp] theorem cachedContinuableConsumedWitnessTails_buildCertificateCache
    (c : ExpandedCertificate) (start finish : Vertex) :
    cachedContinuableConsumedWitnessTails c (buildCertificateCache c)
        start finish =
      continuableConsumedWitnessTails c start finish := by
  simp only [cachedContinuableConsumedWitnessTails, buildCertificateCache]
  have hw : witnessesFromCacheTable
      (c.graph.EDGES.eraseDups.map fun edge => (edge, edgeWitnessesFor c edge))
        (edgeWitnessesFor c) = edgeWitnessesFor c := by
    funext edge
    exact witnessesFromCacheTable_exact c.graph.EDGES.eraseDups
      (edgeWitnessesFor c) edge
  rw [hw]
  change tailsFromCacheTable
      (((c.graph.EDGES.map fun edge => (edge.START, edge.END)).eraseDups).map
        fun pair => (pair, continuableConsumedWitnessTails c pair.1 pair.2))
      (continuableConsumedWitnessTails c) start finish =
        continuableConsumedWitnessTails c start finish
  exact tailsFromCacheTable_exact
    (c.graph.EDGES.map fun edge => (edge.START, edge.END)).eraseDups
    (continuableConsumedWitnessTails c) start finish




def certificateCarrierAccountedAtB (c : ExpandedCertificate)
    (w u : Vertex) : Bool :=
  match w.FORMULA with
  | Formula.implication _ beta =>
      u.FORMULA == beta &&
        (continuableConsumedWitnessTails c u w).all fun tail =>
          (witnessableExpected c w).any fun result => result.2 == tail
  | _ => false




def certificatePairTailClosureFromB (c : ExpandedCertificate)
    (witnesses : Deduction → List EdgeRouteWitness)
    (w major minor : Vertex) : Bool :=
  ((edgesBetween c.graph minor w).all fun minorEdge =>
    (witnesses minorEdge).all fun minorWitness =>
      match consume minorEdge minorWitness.residual with
      | none => true
      | some minorTail =>
          !routeContinuableAtB c.graph w minorTail ||
            (edgesBetween c.graph major w).any fun majorEdge =>
              (witnesses majorEdge).any fun majorWitness =>
                consume majorEdge majorWitness.residual == some minorTail ||
                  reseedPairSupportsB c.graph w
                    (minorWitness.dependency, minorWitness.residual) minorEdge
                    (majorWitness.dependency, majorWitness.residual) majorEdge) &&
  ((edgesBetween c.graph major w).all fun majorEdge =>
    (witnesses majorEdge).all fun majorWitness =>
      match consume majorEdge majorWitness.residual with
      | none => true
      | some majorTail =>
          !routeContinuableAtB c.graph w majorTail ||
            (edgesBetween c.graph minor w).any fun minorEdge =>
              (witnesses minorEdge).any fun minorWitness =>
                consume minorEdge minorWitness.residual == some majorTail ||
                  reseedPairSupportsB c.graph w
                    (minorWitness.dependency, minorWitness.residual) minorEdge
                    (majorWitness.dependency, majorWitness.residual) majorEdge)

def certificatePairTailClosureAtB (c : ExpandedCertificate)
    (w major minor : Vertex) : Bool :=
  certificatePairTailClosureFromB c (edgeWitnessesFor c) w major minor

def tailClosureAtB (c : ExpandedCertificate) (w : Vertex) : Bool :=
  match elimPairsAt c.graph w with
  | [] => true
  | pairs =>
      (predsOf c.graph w).all (fun u =>
        elimPremiseAtB c.graph w u || certificateCarrierAccountedAtB c w u) &&
      pairs.all fun (major, minor) =>
        certificatePairTailClosureAtB c w major minor

def rootDischargeAtB (c : ExpandedCertificate) (v : Vertex) : Bool :=
  if (get_rule.outgoing v c.graph).isEmpty then
    (c.terminalOutputs.filter fun output => output.node == v).any fun output =>
      output.dependency.isEmpty && output.residual.isEmpty &&
      productionAtB c v && ruleShapeOKB v c.graph &&
      decide (outDep v c.graph = [])
  else true

def witnessFaithfulAtCachedB (c : ExpandedCertificate) (cache : CertificateCache)
    (witness : EdgeRouteWitness) : Bool :=
  let matchingRoutes := (cachedLocalExpected c cache witness.edge.START).filter
    fun route => route.2 == witness.residual
  matchingRoutes.isEmpty || matchingRoutes.any fun route =>
    depSetEq route.1 witness.dependency

def productionAtCachedB (c : ExpandedCertificate) (cache : CertificateCache)
    (v : Vertex) : Bool :=
  match localShape c.graph v with
  | .invalid => false
  | _ => setEqB (cachedOutputsAt c cache v)
      (cachedWitnessableExpected c cache v)

def coverageAtCachedB (c : ExpandedCertificate) (cache : CertificateCache)
    (v : Vertex) : Bool :=
  let outgoing := get_rule.outgoing v c.graph
  outgoing.all (fun edge =>
    let witnesses := cachedEdgeWitnessesFor c cache edge
    !witnesses.isEmpty &&
      setEqB (witnesses.map fun witness =>
          (witness.dependency, witness.residual))
        ((cachedOutputsAt c cache v).filter fun route =>
          headColour route.2 == edge.COLOUR)) &&
  (c.edgeRoutes.filter fun witness => witness.edge.START == v).all
    (fun witness => outgoing.contains witness.edge)

def certificateCarrierAccountedAtCachedB (c : ExpandedCertificate)
    (cache : CertificateCache) (w u : Vertex) : Bool :=
  match w.FORMULA with
  | Formula.implication _ beta =>
      u.FORMULA == beta &&
        (cachedContinuableConsumedWitnessTails c cache u w).all fun tail =>
          (cachedWitnessableExpected c cache w).any fun result => result.2 == tail
  | _ => false

def tailClosureAtCachedB (c : ExpandedCertificate) (cache : CertificateCache)
    (w : Vertex) : Bool :=
  match elimPairsAt c.graph w with
  | [] => true
  | pairs =>
      (predsOf c.graph w).all (fun u =>
        elimPremiseAtB c.graph w u ||
          certificateCarrierAccountedAtCachedB c cache w u) &&
      pairs.all fun (major, minor) =>
        certificatePairTailClosureFromB c
          (cachedEdgeWitnessesFor c cache) w major minor

def rootDischargeAtCachedB (c : ExpandedCertificate) (cache : CertificateCache)
    (v : Vertex) : Bool :=
  if (get_rule.outgoing v c.graph).isEmpty then
    (c.terminalOutputs.filter fun output => output.node == v).any fun output =>
      output.dependency.isEmpty && output.residual.isEmpty &&
      productionAtCachedB c cache v && ruleShapeOKB v c.graph &&
      decide (outDep v c.graph = [])
  else true

def expandedCertCachedB (d : Graph) (c : ExpandedCertificate)
    (cache : CertificateCache) : Bool :=
  decide (c.graph = d) &&
  d.EDGES.all (fun edge => !(cachedEdgeWitnessesFor c cache edge).isEmpty) &&
  c.edgeRoutes.all (fun witness => d.EDGES.contains witness.edge) &&
  d.NODES.all (fun v =>
    if (get_rule.outgoing v d).isEmpty then
      !(c.terminalOutputs.filter fun output => output.node == v).isEmpty
    else true) &&
  c.terminalOutputs.all (fun output =>
    d.NODES.contains output.node &&
      (get_rule.outgoing output.node d).isEmpty)

@[simp] theorem witnessFaithfulAtCachedB_buildCertificateCache
    (c : ExpandedCertificate) (witness : EdgeRouteWitness) :
    witnessFaithfulAtCachedB c (buildCertificateCache c) witness =
      witnessFaithfulAtB c witness := by
  simp [witnessFaithfulAtCachedB, witnessFaithfulAtB]

@[simp] theorem productionAtCachedB_buildCertificateCache
    (c : ExpandedCertificate) (v : Vertex) :
    productionAtCachedB c (buildCertificateCache c) v = productionAtB c v := by
  simp [productionAtCachedB, productionAtB]

@[simp] theorem coverageAtCachedB_buildCertificateCache
    (c : ExpandedCertificate) (v : Vertex) :
    coverageAtCachedB c (buildCertificateCache c) v = coverageAtB c v := by
  simp [coverageAtCachedB, coverageAtB]

@[simp] theorem certificateCarrierAccountedAtCachedB_buildCertificateCache
    (c : ExpandedCertificate) (w u : Vertex) :
    certificateCarrierAccountedAtCachedB c (buildCertificateCache c) w u =
      certificateCarrierAccountedAtB c w u := by
  simp [certificateCarrierAccountedAtCachedB, certificateCarrierAccountedAtB]

@[simp] theorem tailClosureAtCachedB_buildCertificateCache
    (c : ExpandedCertificate) (w : Vertex) :
    tailClosureAtCachedB c (buildCertificateCache c) w = tailClosureAtB c w := by
  simp [tailClosureAtCachedB, tailClosureAtB,
    certificatePairTailClosureAtB, certificatePairTailClosureFromB]

@[simp] theorem rootDischargeAtCachedB_buildCertificateCache
    (c : ExpandedCertificate) (v : Vertex) :
    rootDischargeAtCachedB c (buildCertificateCache c) v = rootDischargeAtB c v := by
  simp [rootDischargeAtCachedB, rootDischargeAtB]

def structuralEdgeAtB (d : Graph) (edge : Deduction) : Bool :=
  edge.START.LEVEL == edge.END.LEVEL + 1 &&
    d.NODES.contains edge.START && d.NODES.contains edge.END

def structuralNodeAtB (d : Graph) (v : Vertex) : Bool :=
  if !(get_rule.incoming v d).isEmpty && !(get_rule.outgoing v d).isEmpty then
    nodeElimShapeWideB d v || nodeIntroShapeB d v
  else true

def hypSourceAtB (d : Graph) (v : Vertex) : Bool :=
  if v.HYPOTHESIS then (get_rule.incoming v d).isEmpty else true

def edgeCoverageFailures (c : ExpandedCertificate) : List Failure :=
  let graphMissing := c.graph.EDGES.filterMap fun edge =>
    if witnessCoverageAtB c edge then none
    else
      let witnesses := edgeWitnessesFor c edge
      let start := edge.START.NUMBER
      let finish := edge.END.NUMBER
      let count := witnesses.length
      some (Failure.mk "WitnessCoverage" s!"edge {start}->{finish}: count {count}")
  let extra := c.edgeRoutes.filterMap fun witness =>
    if c.graph.EDGES.contains witness.edge then none
    else
      let start := witness.edge.START.NUMBER
      let finish := witness.edge.END.NUMBER
      some (Failure.mk "WitnessCoverage" s!"extra edge {start}->{finish}")
  graphMissing ++ extra

def edgeCoverageFailuresCached (c : ExpandedCertificate)
    (cache : CertificateCache) : List Failure :=
  let graphMissing := c.graph.EDGES.filterMap fun edge =>
    if !(cachedEdgeWitnessesFor c cache edge).isEmpty then none
    else
      let witnesses := cachedEdgeWitnessesFor c cache edge
      let start := edge.START.NUMBER
      let finish := edge.END.NUMBER
      let count := witnesses.length
      some (Failure.mk "WitnessCoverage" s!"edge {start}->{finish}: count {count}")
  let extra := c.edgeRoutes.filterMap fun witness =>
    if c.graph.EDGES.contains witness.edge then none
    else
      let start := witness.edge.START.NUMBER
      let finish := witness.edge.END.NUMBER
      some (Failure.mk "WitnessCoverage" s!"extra edge {start}->{finish}")
  graphMissing ++ extra

def failures (c : ExpandedCertificate) : List Failure :=
  let d := c.graph
  let indexFailures :=
    (if Semantic.nodupB d then [] else
      [{ check := "EDGES.Nodup", position := "edge-record index" }]) ++
    (if Semantic.oneEdgeB d then [] else
      [{ check := "OneEdgePerColourPerNode", position := "source-colour index" }]) ++
    (if levelFormulaUniqueB d then [] else
      [{ check := "LevelFormulaUnique", position := "level-formula index" }])
  let edgeFailures := d.EDGES.flatMap fun edge =>
    let start := edge.START.NUMBER
    let finish := edge.END.NUMBER
    let pos := s!"edge {start}->{finish}"
    (if structuralEdgeAtB d edge then [] else
      [{ check := "StructuralEdge", position := pos }])
  let witnessFailures := c.edgeRoutes.zipIdx.filterMap fun (witness, index) =>
    if witnessFaithfulAtB c witness then none
    else
      some (Failure.mk "FaithfulDecoration"
        s!"witness {index} at node {witness.edge.START.NUMBER}")
  let nodeFailures := d.NODES.flatMap fun v =>
    let pos := s!"node {v.NUMBER}"
    (if structuralNodeAtB d v then [] else
      [{ check := "StructuralNode", position := pos }]) ++
    (if hypSourceAtB d v then [] else
      [{ check := "HypothesesHaveNoIncoming", position := pos }]) ++
    (if productionAtB c v then [] else
      [{ check := "RouteProduction", position := pos }]) ++
    (if coverageAtB c v then [] else
      [{ check := "RouteHeadCoverage", position := pos }]) ++
    (if tailClosureAtB c v then [] else
      [{ check := "FlowTailClosure", position := pos }]) ++
    (if rootDischargeAtB c v then [] else
      [{ check := "RootDischarge", position := pos }])
  indexFailures ++ edgeCoverageFailures c ++ edgeFailures ++
    witnessFailures ++ nodeFailures

def failuresCached (c : ExpandedCertificate) (cache : CertificateCache) :
    List Failure :=
  let d := c.graph
  let indexFailures :=
    (if Semantic.nodupB d then [] else
      [{ check := "EDGES.Nodup", position := "edge-record index" }]) ++
    (if Semantic.oneEdgeB d then [] else
      [{ check := "OneEdgePerColourPerNode", position := "source-colour index" }]) ++
    (if levelFormulaUniqueB d then [] else
      [{ check := "LevelFormulaUnique", position := "level-formula index" }])
  let edgeFailures := d.EDGES.flatMap fun edge =>
    let start := edge.START.NUMBER
    let finish := edge.END.NUMBER
    let pos := s!"edge {start}->{finish}"
    (if structuralEdgeAtB d edge then [] else
      [{ check := "StructuralEdge", position := pos }])
  let witnessFailures := c.edgeRoutes.zipIdx.filterMap fun (witness, index) =>
    if witnessFaithfulAtCachedB c cache witness then none
    else
      some (Failure.mk "FaithfulDecoration"
        s!"witness {index} at node {witness.edge.START.NUMBER}")
  let nodeFailures := d.NODES.flatMap fun v =>
    let pos := s!"node {v.NUMBER}"
    (if structuralNodeAtB d v then [] else
      [{ check := "StructuralNode", position := pos }]) ++
    (if hypSourceAtB d v then [] else
      [{ check := "HypothesesHaveNoIncoming", position := pos }]) ++
    (if productionAtCachedB c cache v then [] else
      [{ check := "RouteProduction", position := pos }]) ++
    (if coverageAtCachedB c cache v then [] else
      [{ check := "RouteHeadCoverage", position := pos }]) ++
    (if tailClosureAtCachedB c cache v then [] else
      [{ check := "FlowTailClosure", position := pos }]) ++
    (if rootDischargeAtCachedB c cache v then [] else
      [{ check := "RootDischarge", position := pos }])
  indexFailures ++ edgeCoverageFailuresCached c cache ++ edgeFailures ++
    witnessFailures ++ nodeFailures

@[simp] theorem edgeCoverageFailuresCached_buildCertificateCache
    (c : ExpandedCertificate) :
    edgeCoverageFailuresCached c (buildCertificateCache c) =
      edgeCoverageFailures c := by
  simp [edgeCoverageFailuresCached, edgeCoverageFailures, witnessCoverageAtB]

@[simp] theorem failuresCached_buildCertificateCache
    (c : ExpandedCertificate) :
    failuresCached c (buildCertificateCache c) = failures c := by
  simp [failuresCached, failures]

def failuresMemo (c : ExpandedCertificate) : List Failure :=
  let cache := buildCertificateCache c
  failuresCached c cache

def residualHistogram (c : ExpandedCertificate) : List (Nat × Nat) :=
  let lengths := c.edgeRoutes.map (fun witness => witness.residual.length) ++
    c.terminalOutputs.map (fun output => output.residual.length)
  (lengths.eraseDups.mergeSort (· ≤ ·)).map fun length =>
    (length, (lengths.filter (· == length)).length)

def maxResidual (c : ExpandedCertificate) : Nat :=
  (c.edgeRoutes.map (fun witness => witness.residual.length) ++
    c.terminalOutputs.map (fun output => output.residual.length)).foldl max 0

def residualCells (c : ExpandedCertificate) : Nat :=
  c.edgeRoutes.foldl (fun total witness => total + witness.residual.length) 0 +
    c.terminalOutputs.foldl (fun total output => total + output.residual.length) 0

def terminalDepCells (c : ExpandedCertificate) : Nat :=
  c.terminalOutputs.foldl
    (fun total output => total + output.dependency.length) 0

def edgeWitnessDepCells (c : ExpandedCertificate) : Nat :=
  c.edgeRoutes.foldl
    (fun total witness => total + witness.dependency.length) 0

def baseRecordCount (c : ExpandedCertificate) : Nat :=
  c.graph.NODES.length + c.graph.EDGES.length + c.graph.PATHS.length

def baseEncodedSize (c : ExpandedCertificate) : Nat :=
  baseRecordCount c +
    c.graph.EDGES.foldl (fun total edge => total + edge.DEPENDENCY.length) 0 +
    c.graph.PATHS.foldl (fun total path => total + path.COLOURS.length) 0

def witnessEncodedSize (c : ExpandedCertificate) : Nat :=
  c.edgeRoutes.length + c.terminalOutputs.length + residualCells c +
    edgeWitnessDepCells c + terminalDepCells c

def uniquenessIndexSize (c : ExpandedCertificate) : Nat :=
  c.graph.EDGES.length + c.graph.EDGES.length + c.graph.NODES.length

def positionCount (c : ExpandedCertificate) : Nat :=
  baseEncodedSize c + witnessEncodedSize c + uniquenessIndexSize c

def sizeReport (name : String) (c : ExpandedCertificate) :=
  (name, c.graph.NODES.length, c.graph.EDGES.length, c.graph.PATHS.length,
    c.edgeRoutes.length, c.terminalOutputs.length, residualHistogram c,
    maxResidual c, residualCells c, edgeWitnessDepCells c, terminalDepCells c,
    baseRecordCount c, baseEncodedSize c, witnessEncodedSize c,
    uniquenessIndexSize c, positionCount c)

def elimCostTable (c : ExpandedCertificate) : List (Nat × Nat × Nat × Nat) :=
  c.graph.NODES.filterMap fun w =>
    match elimPairsAt c.graph w with
    | [(major, minor)] =>
        let selectionCount := (outputsAt c minor).length * (outputsAt c major).length *
          (edgesBetween c.graph minor w).length *
          (edgesBetween c.graph major w).length
        some (w.NUMBER, selectionCount, (elimExpected c w major minor).length,
          (outputsAt c w).length)
    | _ => none

def alterFirstResidual (c : ExpandedCertificate) : ExpandedCertificate :=
  match c.edgeRoutes with
  | [] => c
  | witness :: rest =>
      { c with edgeRoutes := { witness with residual := witness.residual ++ [999] } :: rest }

def omitFirstWitness (c : ExpandedCertificate) : ExpandedCertificate :=
  { c with edgeRoutes := c.edgeRoutes.drop 1 }

def corruptDependency (c : ExpandedCertificate) : ExpandedCertificate :=
  match c.edgeRoutes with
  | [] => c
  | witness :: _rest =>
      let badEdge := { witness.edge with
        DEPENDENCY := normalizeDep (witness.edge.DEPENDENCY ++ [#"BAD"]) }
      let graph := { c.graph with EDGES := c.graph.EDGES.map fun edge =>
        if edge == witness.edge then badEdge else edge }
      ExpandedCertificate.mk graph
        (c.edgeRoutes.map fun selection =>
          if selection.edge == witness.edge then
            { selection with edge := badEdge, dependency := badEdge.DEPENDENCY }
          else selection)
        c.terminalOutputs

def corruptTerminal (c : ExpandedCertificate) : ExpandedCertificate :=
  match c.terminalOutputs with
  | [] => c
  | output :: rest =>
      { c with terminalOutputs := { output with dependency := [#"BAD"] } :: rest }

def failureReport (name : String) (c : ExpandedCertificate) :=
  let found := failuresMemo c
  (name, positionCount c, found.length, found)

def d11Certificate := generateCertificate d11
def eCertificate := generateCertificate Semantic.e

end CertificateModel
