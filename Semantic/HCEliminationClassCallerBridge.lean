import Semantic.HCEliminationClassOperational
import Semantic.NDTreeOccurrenceReplay










open Semantic FlowSpec
open CandidateMixedReplay OccurrenceAwareReplay
open HorizontalCompressionInput
open LocalSpecificationCandidate HCContextualMixedHE
open HCEliminationClassPreservation HCEliminationClassOperational

namespace HCEliminationClassCallerBridge

set_option linter.unusedSimpArgs false

theorem derivedOutputs_congr_of_incoming
    (left right : Graph) (vertex : Vertex)
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right)
    (premiseFlow : Vertex -> List FlowPair) :
    derivedOutputs left vertex premiseFlow =
      derivedOutputs right vertex premiseFlow := by
  have predecessors : predsOf left vertex = predsOf right vertex := by
    simp only [FlowSpec.predsOf_eq, incoming]
  have between : forall start,
      edgesBetween left start vertex = edgesBetween right start vertex := by
    intro start
    simp only [HorizontalCompressionInput.edgesBetween_eq_incoming, incoming]
  have eliminations : elimPairsAt left vertex = elimPairsAt right vertex := by
    simp only [elimPairsAt, predecessors]
  have eliminationOutputs :
      candidateEliminationOutputs left vertex premiseFlow =
        candidateEliminationOutputs right vertex premiseFlow := by
    unfold candidateEliminationOutputs
    rw [eliminations]
    simp_rw [between]
  have introductions : introductionCandidates left vertex premiseFlow =
      introductionCandidates right vertex premiseFlow := by
    unfold introductionCandidates candidateIntroductionOutputs
    rw [predecessors]
    simp_rw [between]
  have separated : separatedIntroductionOutputs left vertex premiseFlow =
      separatedIntroductionOutputs right vertex premiseFlow := by
    unfold separatedIntroductionOutputs
    rw [eliminationOutputs, introductions]
  simp only [derivedOutputs, eliminationOutputs, separated]

theorem occurrenceStep_congr_of_local_data
    (left right : Graph) (vertex : Vertex)
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right)
    (ancestors : ancestorsInto left vertex = ancestorsInto right vertex)
    (sources : candidateSourceRoutes left vertex =
      candidateSourceRoutes right vertex)
    (premiseFlow : Vertex -> List FlowPair) :
    occurrenceStep left vertex premiseFlow =
      occurrenceStep right vertex premiseFlow := by
  have predecessors : predsOf left vertex = predsOf right vertex := by
    simp only [FlowSpec.predsOf_eq, incoming]
  have derived := derivedOutputs_congr_of_incoming left right vertex incoming
    premiseFlow
  simp only [occurrenceStep, predecessors, ancestors, sources, derived]




def ReplayFrame (left right : Graph) : Nat -> Vertex -> Prop
  | 0, _ => True
  | fuel + 1, vertex =>
      get_rule.incoming vertex left = get_rule.incoming vertex right /\
      ancestorsInto left vertex = ancestorsInto right vertex /\
      candidateSourceRoutes left vertex = candidateSourceRoutes right vertex /\
      forall premise, premise ∈ predsOf left vertex ->
        ReplayFrame left right fuel premise

theorem ReplayFrame.of_leaf
    (left right : Graph) (vertex : Vertex)
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right)
    (ancestors : ancestorsInto left vertex = ancestorsInto right vertex)
    (sources : candidateSourceRoutes left vertex =
      candidateSourceRoutes right vertex)
    (noPredecessors : predsOf left vertex = []) :
    forall fuel, ReplayFrame left right fuel vertex := by
  intro fuel
  cases fuel with
  | zero => trivial
  | succ fuel =>
      exact ⟨incoming, ancestors, sources, by
        intro premise member
        rw [noPredecessors] at member
        simp at member⟩

private theorem mapM_replay_congr
    (left right : Graph) (fuel : Nat) (vertices : List Vertex)
    (calls : forall vertex, vertex ∈ vertices ->
      occurrenceReplayAt left fuel vertex =
        occurrenceReplayAt right fuel vertex) :
    vertices.mapM (fun premise => do
        let routes <- occurrenceReplayAt left fuel premise
        return (premise, routes)) =
      vertices.mapM (fun premise => do
        let routes <- occurrenceReplayAt right fuel premise
        return (premise, routes)) := by
  induction vertices with
  | nil => rfl
  | cons vertex rest ih =>
      simp only [List.mapM_cons]
      rw [calls vertex (List.Mem.head rest)]
      rw [ih]
      intro candidate member
      exact calls candidate (List.Mem.tail vertex member)

private theorem except_bind_congr
    {ε α β : Type} (result : Except ε α)
    (left right : α -> Except ε β)
    (equal : forall value, left value = right value) :
    result.bind left = result.bind right := by
  cases result with
  | error error => rfl
  | ok value => exact equal value

theorem occurrenceReplayAt_congr_of_frame
    (left right : Graph) (fuel : Nat) (vertex : Vertex)
    (frame : ReplayFrame left right fuel vertex) :
    occurrenceReplayAt left fuel vertex =
      occurrenceReplayAt right fuel vertex := by
  induction fuel generalizing vertex with
  | zero => rfl
  | succ fuel ih =>
      rcases frame with ⟨incoming, ancestors, sources, premiseFrames⟩
      have predecessors : predsOf left vertex = predsOf right vertex := by
        simp only [FlowSpec.predsOf_eq, incoming]
      have premiseCalls : forall premise, premise ∈ predsOf left vertex ->
          occurrenceReplayAt left fuel premise =
            occurrenceReplayAt right fuel premise := by
        intro premise member
        exact ih premise (premiseFrames premise member)
      have tables :
          (predsOf left vertex).mapM (fun premise => do
            let routes <- occurrenceReplayAt left fuel premise
            return (premise, routes)) =
          (predsOf right vertex).mapM (fun premise => do
            let routes <- occurrenceReplayAt right fuel premise
            return (premise, routes)) := by
        rw [← predecessors]
        exact mapM_replay_congr left right fuel (predsOf left vertex)
          premiseCalls
      rw [occurrenceReplayAt, occurrenceReplayAt]
      rw [tables]
      let tableResult := (predsOf right vertex).mapM (fun premise => do
          let routes <- occurrenceReplayAt right fuel premise
          return (premise, routes))
      exact except_bind_congr tableResult
        (fun table => occurrenceStep left vertex
          (OccurrenceAwareReplay.routesFor table))
        (fun table => occurrenceStep right vertex
          (OccurrenceAwareReplay.routesFor table))
        (fun table => occurrenceStep_congr_of_local_data left right vertex incoming
            ancestors sources (OccurrenceAwareReplay.routesFor table)
        )

def tracedRoutes (dependency : Dep) (ancestors : List Ancestral) :
    List FlowPair :=
  (ancestors.map fun ancestor => (dependency, ancestor.COLOURS)).eraseDups

@[simp] theorem reseedDerivedRoutes_singleton
    (ancestors : List Ancestral) (dependency : Dep) :
    reseedDerivedRoutes ancestors [(dependency, [])] =
      tracedRoutes dependency ancestors := by
  unfold reseedDerivedRoutes tracedRoutes
  congr 1
  induction ancestors with
  | nil => rfl
  | cons ancestor rest ih =>
      change List.flatMap (fun ancestor =>
        [(dependency, ancestor.COLOURS)]) rest =
          List.map (fun ancestor => (dependency, ancestor.COLOURS)) rest at ih
      simp [colourPathPrefixB, ih]

theorem reseedDerivedRoutes_routes_congr
    (ancestors : List Ancestral) (left right : List FlowPair)
    (routes : SelfCertifying.ListSetEq left right) :
    SelfCertifying.ListSetEq
      (reseedDerivedRoutes ancestors left)
      (reseedDerivedRoutes ancestors right) := by
  intro output
  simp only [reseedDerivedRoutes, List.mem_eraseDups, List.mem_flatMap,
    List.mem_filterMap]
  constructor
  · rintro ⟨ancestor, ancestorMember, route, routeMember, selected⟩
    exact ⟨ancestor, ancestorMember, route, (routes route).mp routeMember,
      selected⟩
  · rintro ⟨ancestor, ancestorMember, route, routeMember, selected⟩
    exact ⟨ancestor, ancestorMember, route, (routes route).mpr routeMember,
      selected⟩




theorem occurrenceReplayAt_reseed_derived_from_base
    (actual base : Graph) (fuel : Nat) (vertex : Vertex)
    (dependency : Dep) (traces : List Ancestral)
    (incoming : get_rule.incoming vertex actual =
      get_rule.incoming vertex base)
    (actualAncestors : ancestorsInto actual vertex = traces)
    (baseAncestors : ancestorsInto base vertex = [])
    (tracesNonempty : traces ≠ [])
    (derived : vertex.HYPOTHESIS = false)
    (premiseFrames : forall premise, premise ∈ predsOf actual vertex ->
      ReplayFrame actual base fuel premise)
    (baseReplay : occurrenceReplayAt base (fuel + 1) vertex =
      .ok [(dependency, [])]) :
    occurrenceReplayAt actual (fuel + 1) vertex =
      .ok (tracedRoutes dependency traces) := by
  have predecessors : predsOf actual vertex = predsOf base vertex := by
    simp only [FlowSpec.predsOf_eq, incoming]
  have premiseCalls : forall premise, premise ∈ predsOf actual vertex ->
      occurrenceReplayAt actual fuel premise =
        occurrenceReplayAt base fuel premise := by
    intro premise member
    exact occurrenceReplayAt_congr_of_frame actual base fuel premise
      (premiseFrames premise member)
  have tables :
      (predsOf actual vertex).mapM (fun premise => do
        let routes <- occurrenceReplayAt actual fuel premise
        return (premise, routes)) =
      (predsOf base vertex).mapM (fun premise => do
        let routes <- occurrenceReplayAt base fuel premise
        return (premise, routes)) := by
    rw [← predecessors]
    exact mapM_replay_congr actual base fuel (predsOf actual vertex)
      premiseCalls
  let baseTableResult := (predsOf base vertex).mapM (fun premise => do
      let routes <- occurrenceReplayAt base fuel premise
      return (premise, routes))
  let actualTableResult := (predsOf actual vertex).mapM (fun premise => do
      let routes <- occurrenceReplayAt actual fuel premise
      return (premise, routes))
  have tableResults : actualTableResult = baseTableResult := by
    exact tables
  change actualTableResult.bind (fun table =>
      occurrenceStep actual vertex
        (OccurrenceAwareReplay.routesFor table)) = _
  change baseTableResult.bind (fun table =>
      occurrenceStep base vertex
        (OccurrenceAwareReplay.routesFor table)) = _ at baseReplay
  rw [tableResults]
  cases tableResult : baseTableResult with
  | error replayError =>
      rw [tableResult] at baseReplay
      simp [Except.bind] at baseReplay
  | ok table =>
      rw [tableResult] at baseReplay
      simp only [Except.bind] at baseReplay ⊢
      have predecessorNonempty : (predsOf base vertex).isEmpty = false := by
        by_cases emptyList : predsOf base vertex = []
        · have impossible : False := by
            simp [occurrenceStep, emptyList, derived] at baseReplay
          exact False.elim impossible
        · simpa [List.isEmpty_iff] using emptyList
      have actualPredecessorNonempty :
          (predsOf actual vertex).isEmpty = false := by
        rw [predecessors]
        exact predecessorNonempty
      have derivedEq := derivedOutputs_congr_of_incoming actual base vertex
        incoming (OccurrenceAwareReplay.routesFor table)
      have baseDerived :
          derivedOutputs base vertex (OccurrenceAwareReplay.routesFor table) =
            [(dependency, [])] := by
        have nonempty : derivedOutputs base vertex
            (OccurrenceAwareReplay.routesFor table) ≠ [] := by
          intro empty
          simp [occurrenceStep, predecessorNonempty, baseAncestors, derived,
            empty] at baseReplay
        simpa [occurrenceStep, predecessorNonempty, baseAncestors, derived,
          nonempty] using baseReplay
      have actualDerived :
          derivedOutputs actual vertex (OccurrenceAwareReplay.routesFor table) =
            [(dependency, [])] := by
        rw [derivedEq, baseDerived]
      obtain ⟨ancestor, ancestorMember⟩ :=
        List.exists_mem_of_ne_nil _ tracesNonempty
      have tracedMember : (dependency, ancestor.COLOURS) ∈
          tracedRoutes dependency traces := by
        rw [tracedRoutes, List.mem_eraseDups, List.mem_map]
        exact ⟨ancestor, ancestorMember, rfl⟩
      have tracedNonempty : tracedRoutes dependency traces ≠ [] :=
        List.ne_nil_of_mem tracedMember
      have reseededNonempty :
          (reseedDerivedRoutes traces [(dependency, [])]).isEmpty = false := by
        rw [List.isEmpty_eq_false_iff]
        rw [reseedDerivedRoutes_singleton]
        exact List.ne_nil_of_mem tracedMember
      simp [Except.bind, occurrenceStep, actualPredecessorNonempty, actualAncestors,
        tracesNonempty, derived, actualDerived, reseedDerivedRoutes_singleton,
        reseededNonempty, tracedNonempty]





theorem occurrenceReplayAt_transport_derived_from_current
    (actual current : Graph) (fuel : Nat) (vertex : Vertex)
    (routes : List FlowPair) (traces : List Ancestral)
    (incoming : get_rule.incoming vertex actual =
      get_rule.incoming vertex current)
    (actualAncestors : ancestorsInto actual vertex = traces)
    (currentAncestors : ancestorsInto current vertex = [])
    (tracesNonempty : traces ≠ [])
    (derived : vertex.HYPOTHESIS = false)
    (premiseFrames : forall premise, premise ∈ predsOf actual vertex ->
      ReplayFrame actual current fuel premise)
    (currentReplay : occurrenceReplayAt current (fuel + 1) vertex =
      .ok routes)
    (transportedNonempty :
      (reseedDerivedRoutes traces routes).isEmpty = false) :
    occurrenceReplayAt actual (fuel + 1) vertex =
      .ok (reseedDerivedRoutes traces routes) := by
  have predecessors : predsOf actual vertex = predsOf current vertex := by
    simp only [FlowSpec.predsOf_eq, incoming]
  have premiseCalls : forall premise, premise ∈ predsOf actual vertex ->
      occurrenceReplayAt actual fuel premise =
        occurrenceReplayAt current fuel premise := by
    intro premise member
    exact occurrenceReplayAt_congr_of_frame actual current fuel premise
      (premiseFrames premise member)
  have tables :
      (predsOf actual vertex).mapM (fun premise => do
        let premiseRoutes <- occurrenceReplayAt actual fuel premise
        return (premise, premiseRoutes)) =
      (predsOf current vertex).mapM (fun premise => do
        let premiseRoutes <- occurrenceReplayAt current fuel premise
        return (premise, premiseRoutes)) := by
    rw [← predecessors]
    exact mapM_replay_congr actual current fuel (predsOf actual vertex)
      premiseCalls
  let currentTableResult := (predsOf current vertex).mapM (fun premise => do
      let premiseRoutes <- occurrenceReplayAt current fuel premise
      return (premise, premiseRoutes))
  let actualTableResult := (predsOf actual vertex).mapM (fun premise => do
      let premiseRoutes <- occurrenceReplayAt actual fuel premise
      return (premise, premiseRoutes))
  have tableResults : actualTableResult = currentTableResult := tables
  change actualTableResult.bind (fun table =>
      occurrenceStep actual vertex
        (OccurrenceAwareReplay.routesFor table)) = _
  change currentTableResult.bind (fun table =>
      occurrenceStep current vertex
        (OccurrenceAwareReplay.routesFor table)) = _ at currentReplay
  rw [tableResults]
  cases tableResult : currentTableResult with
  | error replayError =>
      rw [tableResult] at currentReplay
      simp [Except.bind] at currentReplay
  | ok table =>
      rw [tableResult] at currentReplay
      simp only [Except.bind] at currentReplay ⊢
      have predecessorNonempty : (predsOf current vertex).isEmpty = false := by
        by_cases emptyList : predsOf current vertex = []
        · simp [occurrenceStep, emptyList, derived] at currentReplay
        · simpa [List.isEmpty_iff] using emptyList
      have actualPredecessorNonempty :
          (predsOf actual vertex).isEmpty = false := by
        rw [predecessors]
        exact predecessorNonempty
      have currentDerivedNonempty : derivedOutputs current vertex
          (OccurrenceAwareReplay.routesFor table) ≠ [] := by
        intro empty
        simp [occurrenceStep, predecessorNonempty, currentAncestors, derived,
          empty] at currentReplay
      have currentDerived : derivedOutputs current vertex
          (OccurrenceAwareReplay.routesFor table) = routes := by
        simpa [occurrenceStep, predecessorNonempty, currentAncestors, derived,
          currentDerivedNonempty] using currentReplay
      have routesNonempty : routes ≠ [] := by
        rw [← currentDerived]
        exact currentDerivedNonempty
      have derivedEq := derivedOutputs_congr_of_incoming actual current vertex
        incoming (OccurrenceAwareReplay.routesFor table)
      have actualDerived : derivedOutputs actual vertex
          (OccurrenceAwareReplay.routesFor table) = routes := by
        rw [derivedEq, currentDerived]
      simp [Except.bind, occurrenceStep, actualPredecessorNonempty,
        actualAncestors, tracesNonempty, derived, actualDerived,
        transportedNonempty, routesNonempty]





structure DerivedRouteTransportFrame (actual current : Graph)
    (fuel : Nat) (vertex : Vertex) (routes : List FlowPair)
    (traces : List Ancestral) : Prop where
  incoming : get_rule.incoming vertex actual =
    get_rule.incoming vertex current
  actualAncestors : ancestorsInto actual vertex = traces
  currentAncestors : ancestorsInto current vertex = []
  tracesNonempty : traces ≠ []
  derived : vertex.HYPOTHESIS = false
  premiseFrames : forall premise, premise ∈ predsOf actual vertex ->
    ReplayFrame actual current fuel premise
  currentReplay : occurrenceReplayAt current (fuel + 1) vertex = .ok routes
  transportedNonempty : (reseedDerivedRoutes traces routes).isEmpty = false

theorem DerivedRouteTransportFrame.replay
    {actual current : Graph} {fuel : Nat} {vertex : Vertex}
    {routes : List FlowPair} {traces : List Ancestral}
    (frame : DerivedRouteTransportFrame actual current fuel vertex
      routes traces) :
    occurrenceReplayAt actual (fuel + 1) vertex =
      .ok (reseedDerivedRoutes traces routes) :=
  occurrenceReplayAt_transport_derived_from_current actual current fuel vertex
    routes traces frame.incoming frame.actualAncestors frame.currentAncestors
    frame.tracesNonempty frame.derived frame.premiseFrames
    frame.currentReplay frame.transportedNonempty





theorem recursivePremiseTable_of_exact_routes
    (actual : Graph) (fuel : Nat) (vertices : List Vertex)
    (expected : Vertex -> List FlowPair)
    (calls : forall vertex, vertex ∈ vertices ->
      occurrenceReplayAt actual fuel vertex = .ok (expected vertex)) :
    vertices.mapM (fun premise => do
      let routes <- occurrenceReplayAt actual fuel premise
      return (premise, routes)) =
      .ok (vertices.map fun vertex => (vertex, expected vertex)) := by
  induction vertices with
  | nil => rfl
  | cons vertex rest ih =>
      have tailCalls : forall candidate, candidate ∈ rest ->
          occurrenceReplayAt actual fuel candidate = .ok (expected candidate) := by
        intro candidate member
        exact calls candidate (List.Mem.tail vertex member)
      simp only [List.mapM_cons, calls vertex (List.Mem.head rest)]
      rw [ih tailCalls]
      rfl




structure DerivedLogicalReplayFrame (actual : Graph) (fuel : Nat)
    (vertex : Vertex) (logicalRoutes : List FlowPair)
    (traces : List Ancestral) : Prop where
  tableAndLogical : Exists fun table : List (Vertex × List FlowPair) =>
    (predsOf actual vertex).mapM (fun premise => do
      let routes <- occurrenceReplayAt actual fuel premise
      return (premise, routes)) = .ok table /\
    derivedOutputs actual vertex
      (OccurrenceAwareReplay.routesFor table) = logicalRoutes
  hasPredecessor : (predsOf actual vertex).isEmpty = false
  actualAncestors : ancestorsInto actual vertex = traces
  tracesNonempty : traces ≠ []
  derived : vertex.HYPOTHESIS = false
  logicalNonempty : logicalRoutes.isEmpty = false
  transportedNonempty :
    (reseedDerivedRoutes traces logicalRoutes).isEmpty = false

theorem DerivedLogicalReplayFrame.replay
    {actual : Graph} {fuel : Nat} {vertex : Vertex}
    {logicalRoutes : List FlowPair} {traces : List Ancestral}
    (frame : DerivedLogicalReplayFrame actual fuel vertex logicalRoutes traces) :
    occurrenceReplayAt actual (fuel + 1) vertex =
      .ok (reseedDerivedRoutes traces logicalRoutes) := by
  obtain ⟨table, tableReplay, logicalOutputs⟩ := frame.tableAndLogical
  rw [occurrenceReplayAt, tableReplay]
  change occurrenceStep actual vertex (OccurrenceAwareReplay.routesFor table) =
    .ok (reseedDerivedRoutes traces logicalRoutes)
  have step := occurrenceStep_reseeds_prefix_matched_derived actual vertex
    (OccurrenceAwareReplay.routesFor table) frame.hasPredecessor
    (by
      rw [frame.actualAncestors, List.isEmpty_eq_false_iff]
      exact frame.tracesNonempty)
    frame.derived
    (by rw [logicalOutputs]; exact frame.logicalNonempty)
    (by rw [frame.actualAncestors, logicalOutputs];
        exact frame.transportedNonempty)
  simpa only [frame.actualAncestors, logicalOutputs] using step






structure DerivedPlainReplayFrame (actual : Graph) (fuel : Nat)
    (vertex : Vertex) (logicalRoutes : List FlowPair) : Prop where
  tableAndLogical : Exists fun table : List (Vertex × List FlowPair) =>
    (predsOf actual vertex).mapM (fun premise => do
      let routes <- occurrenceReplayAt actual fuel premise
      return (premise, routes)) = .ok table /\
    derivedOutputs actual vertex
      (OccurrenceAwareReplay.routesFor table) = logicalRoutes
  hasPredecessor : (predsOf actual vertex).isEmpty = false
  noAncestors : ancestorsInto actual vertex = []
  derived : vertex.HYPOTHESIS = false
  logicalNonempty : logicalRoutes.isEmpty = false

theorem DerivedPlainReplayFrame.replay
    {actual : Graph} {fuel : Nat} {vertex : Vertex}
    {logicalRoutes : List FlowPair}
    (frame : DerivedPlainReplayFrame actual fuel vertex logicalRoutes) :
    occurrenceReplayAt actual (fuel + 1) vertex = .ok logicalRoutes := by
  obtain ⟨table, tableReplay, logicalOutputs⟩ := frame.tableAndLogical
  rw [occurrenceReplayAt, tableReplay]
  change occurrenceStep actual vertex (OccurrenceAwareReplay.routesFor table) =
    .ok logicalRoutes
  simp [occurrenceStep, frame.hasPredecessor, frame.noAncestors, frame.derived,
    logicalOutputs, frame.logicalNonempty]




theorem occurrenceReplayAt_reseed_source
    (actual : Graph) (fuel : Nat) (vertex : Vertex)
    (traces : List Ancestral)
    (noPredecessors : predsOf actual vertex = [])
    (source : vertex.HYPOTHESIS = true)
    (uncollapsed : vertex.COLLAPSED = false)
    (actualAncestors : ancestorsInto actual vertex = traces)
    (tracesNonempty : traces ≠ []) :
    occurrenceReplayAt actual (fuel + 1) vertex =
      .ok (tracedRoutes (normalizeDep [vertex.FORMULA]) traces) := by
  rw [occurrenceReplayAt]
  simp [noPredecessors, occurrenceStep, candidateSourceRoutes,
    actualAncestors, tracesNonempty, uncollapsed, tracedRoutes, source]

def SourcePremiseEvidence (actual : Graph)
    (occurrences : List ElimOccurrence) (_fuel : Nat) (vertex : Vertex) : Prop :=
  Exists fun traces : List Ancestral =>
    predsOf actual vertex = [] /\
    vertex.HYPOTHESIS = true /\
    vertex.COLLAPSED = false /\
    ancestorsInto actual vertex = traces /\
    traces ≠ [] /\
    SelfCertifying.ListSetEq
      (tracedRoutes (normalizeDep [vertex.FORMULA]) traces)
      (expectedPremiseRoutes occurrences vertex)

def DerivedPremiseEvidence (actual base : Graph)
    (occurrences : List ElimOccurrence) (fuel : Nat) (vertex : Vertex) : Prop :=
  Exists fun dependency : Dep => Exists fun traces : List Ancestral =>
    get_rule.incoming vertex actual = get_rule.incoming vertex base /\
    ancestorsInto actual vertex = traces /\
    ancestorsInto base vertex = [] /\
    traces ≠ [] /\
    vertex.HYPOTHESIS = false /\
    (forall premise, premise ∈ predsOf actual vertex ->
      ReplayFrame actual base fuel premise) /\
    occurrenceReplayAt base (fuel + 1) vertex =
      .ok [(dependency, [])] /\
    SelfCertifying.ListSetEq
      (tracedRoutes dependency traces)
      (expectedPremiseRoutes occurrences vertex)

def TransportedDerivedPremiseEvidence (actual current : Graph)
    (occurrences : List ElimOccurrence) (fuel : Nat) (vertex : Vertex) : Prop :=
  Exists fun routes : List FlowPair => Exists fun traces : List Ancestral =>
    DerivedRouteTransportFrame actual current fuel vertex routes traces /\
    SelfCertifying.ListSetEq
      (reseedDerivedRoutes traces routes)
      (expectedPremiseRoutes occurrences vertex)

def LogicalDerivedPremiseEvidence (actual : Graph)
    (occurrences : List ElimOccurrence) (fuel : Nat) (vertex : Vertex) : Prop :=
  Exists fun logicalRoutes : List FlowPair => Exists fun traces : List Ancestral =>
    DerivedLogicalReplayFrame actual fuel vertex logicalRoutes traces /\
    SelfCertifying.ListSetEq
      (reseedDerivedRoutes traces logicalRoutes)
      (expectedPremiseRoutes occurrences vertex)

inductive PremiseEvidence (actual base : Graph)
    (occurrences : List ElimOccurrence) (fuel : Nat) (vertex : Vertex) : Prop
  | source (evidence : SourcePremiseEvidence actual occurrences fuel vertex)
  | derived (evidence : DerivedPremiseEvidence actual base occurrences fuel vertex)
  | transported
      (evidence : TransportedDerivedPremiseEvidence actual base occurrences
        fuel vertex)
  | logicalDerived
      (evidence : LogicalDerivedPremiseEvidence actual occurrences fuel vertex)

theorem PremiseEvidence.replay
    {actual base : Graph} {occurrences : List ElimOccurrence}
    {fuel : Nat} {vertex : Vertex}
    (evidence : PremiseEvidence actual base occurrences fuel vertex) :
    Exists fun routes =>
      occurrenceReplayAt actual (fuel + 1) vertex = .ok routes /\
      SelfCertifying.ListSetEq routes
        (expectedPremiseRoutes occurrences vertex) := by
  cases evidence with
  | source sourceEvidence =>
      rcases sourceEvidence with
        ⟨traces, noPredecessors, source, uncollapsed, actualAncestors,
          tracesNonempty, expected⟩
      refine ⟨tracedRoutes (normalizeDep [vertex.FORMULA]) traces, ?_, expected⟩
      exact occurrenceReplayAt_reseed_source actual fuel vertex
        traces noPredecessors source uncollapsed actualAncestors tracesNonempty
  | derived derivedEvidence =>
      rcases derivedEvidence with
        ⟨dependency, traces, incoming, actualAncestors, baseAncestors,
          tracesNonempty, derived, premiseFrames, baseReplay, expected⟩
      refine ⟨tracedRoutes dependency traces, ?_, expected⟩
      exact occurrenceReplayAt_reseed_derived_from_base actual base fuel vertex
        dependency traces incoming actualAncestors baseAncestors tracesNonempty
        derived premiseFrames baseReplay
  | transported transportedEvidence =>
      rcases transportedEvidence with ⟨routes, traces, frame, expected⟩
      exact ⟨reseedDerivedRoutes traces routes, frame.replay, expected⟩
  | logicalDerived logicalEvidence =>
      rcases logicalEvidence with ⟨routes, traces, frame, expected⟩
      exact ⟨reseedDerivedRoutes traces routes, frame.replay, expected⟩

private theorem routesFor_eq_nil_of_not_mem_keys
    (table : List (Vertex × List FlowPair)) (vertex : Vertex)
    (absent : vertex ∉ table.map Prod.fst) :
    OccurrenceAwareReplay.routesFor table vertex = [] := by
  induction table with
  | nil => rfl
  | cons entry rest ih =>
      simp only [List.map_cons, List.mem_cons, not_or] at absent
      have headNe : entry.1 ≠ vertex := by
        intro equal
        exact absent.1 equal.symm
      simp only [OccurrenceAwareReplay.routesFor]
      simp [headNe]
      simpa [OccurrenceAwareReplay.routesFor] using ih absent.2





theorem recursivePremiseTable_correct
    (actual base : Graph) (occurrences : List ElimOccurrence)
    (fuel : Nat) (predecessors : List Vertex)
    (nodup : predecessors.Nodup)
    (evidence : forall vertex, vertex ∈ predecessors ->
      PremiseEvidence actual base occurrences fuel vertex)
    (support : forall vertex,
      expectedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predecessors) :
    Exists fun table =>
      predecessors.mapM (fun premise => do
        let routes <- occurrenceReplayAt actual (fuel + 1) premise
        return (premise, routes)) = .ok table /\
      table.map Prod.fst = predecessors /\
      PremiseRoutesCorrect
        (OccurrenceAwareReplay.routesFor table) occurrences := by
  have build : forall (vertices : List Vertex), vertices.Nodup ->
      (forall vertex, vertex ∈ vertices ->
        PremiseEvidence actual base occurrences fuel vertex) ->
      Exists fun table =>
        vertices.mapM (fun premise => do
          let routes <- occurrenceReplayAt actual (fuel + 1) premise
          return (premise, routes)) = .ok table /\
        table.map Prod.fst = vertices /\
        (forall vertex, vertex ∈ vertices ->
          SelfCertifying.ListSetEq
            (OccurrenceAwareReplay.routesFor table vertex)
            (expectedPremiseRoutes occurrences vertex)) := by
    intro vertices verticesNodup
    induction vertices with
    | nil =>
        intro _
        exact ⟨[], rfl, rfl, by simp⟩
    | cons vertex rest ih =>
        intro allEvidence
        have nodupParts := List.nodup_cons.mp verticesNodup
        obtain ⟨routes, replay, routesCorrect⟩ :=
          (allEvidence vertex (List.Mem.head rest)).replay
        obtain ⟨tailTable, tailReplay, tailKeys, tailCorrect⟩ :=
          ih nodupParts.2 (fun candidate member =>
            allEvidence candidate (List.Mem.tail vertex member))
        refine ⟨(vertex, routes) :: tailTable, ?_, ?_, ?_⟩
        · simp only [List.mapM_cons, replay]
          rw [tailReplay]
          rfl
        · simp [tailKeys]
        · intro candidate member
          rcases List.mem_cons.mp member with equal | tailMember
          · subst candidate
            simp [OccurrenceAwareReplay.routesFor, routesCorrect]
          · have distinct : vertex ≠ candidate := by
              intro equal
              subst candidate
              exact nodupParts.1 tailMember
            simpa [OccurrenceAwareReplay.routesFor, distinct] using
              tailCorrect candidate tailMember
  obtain ⟨table, replay, keys, onSupport⟩ :=
    build predecessors nodup evidence
  refine ⟨table, replay, keys, ?_⟩
  intro vertex
  by_cases member : vertex ∈ predecessors
  · exact onSupport vertex member
  · have expectedEmpty : expectedPremiseRoutes occurrences vertex = [] := by
      by_contra nonempty
      exact member (support vertex nonempty)
    have absent : vertex ∉ table.map Prod.fst := by
      simpa [keys] using member
    rw [routesFor_eq_nil_of_not_mem_keys table vertex absent, expectedEmpty]
    exact SelfCertifying.ListSetEq.refl []

private theorem accumulatedGraph_center_ancestors_empty
    (center : Vertex) (occurrences : List ElimOccurrence)
    (premisesAvoid : PremisesAvoidCenter center occurrences) :
    ancestorsInto (accumulatedGraph center occurrences) center = [] := by
  exact accumulated_no_center_ancestors center occurrences premisesAvoid

theorem expectedPremiseRoutes_supported
    (center : Vertex) (occurrences : List ElimOccurrence)
    (usesAvoid : UsesAvoidCenter center occurrences)
    (vertex : Vertex)
    (nonempty : expectedPremiseRoutes occurrences vertex ≠ []) :
    vertex ∈ predsOf (accumulatedGraph center occurrences) center := by
  obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ nonempty
  rw [mem_expectedPremiseRoutes_iff] at routeMember
  obtain ⟨occurrence, occurrenceMember, role⟩ := routeMember
  apply (mem_accumulated_preds_iff center vertex occurrences usesAvoid).mpr
  rcases role with ⟨major, routeEq⟩ | ⟨minor, routeEq⟩
  · exact ⟨occurrence, occurrenceMember, Or.inl major⟩
  · exact ⟨occurrence, occurrenceMember, Or.inr minor⟩




theorem AccumulatorInvariant.occurrenceReplayAt
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (actual base : Graph) (fuel : Nat)
    (centerIncoming : get_rule.incoming current.CENTER actual =
      get_rule.incoming current.CENTER
        (ActualMixedProducerBridge.neighborhoodGraph current))
    (centerAncestors : ancestorsInto actual current.CENTER = [])
    (centerDerived : current.CENTER.HYPOTHESIS = false)
    (usesAvoid : UsesAvoidCenter current.CENTER processed)
    (evidence : forall vertex,
      vertex ∈ predsOf actual current.CENTER ->
        PremiseEvidence actual base processed fuel vertex) :
    Exists fun outputs =>
      OccurrenceAwareReplay.occurrenceReplayAt actual (fuel + 2)
        current.CENTER = .ok outputs /\
      SelfCertifying.ListSetEq outputs (expectedFlowRoutes processed) := by
  have predecessorEq : predsOf actual current.CENTER =
      predsOf (ActualMixedProducerBridge.neighborhoodGraph current)
        current.CENTER := by
    simp only [FlowSpec.predsOf_eq, centerIncoming]
  have graphEqual : accumulatedGraph current.CENTER processed =
      ActualMixedProducerBridge.neighborhoodGraph current := by
    unfold accumulatedGraph
    rw [← invariant.exactShape]
  have support : forall vertex,
      expectedPremiseRoutes processed vertex ≠ [] ->
        vertex ∈ predsOf actual current.CENTER := by
    intro vertex nonempty
    rw [predecessorEq, ← graphEqual]
    exact expectedPremiseRoutes_supported current.CENTER processed usesAvoid
      vertex nonempty
  have predecessorNodup : (predsOf actual current.CENTER).Nodup := by
    unfold predsOf
    exact List.eraseDups_nodup' _
  obtain ⟨table, tableReplay, tableKeys, premiseCorrect⟩ :=
    recursivePremiseTable_correct actual base processed fuel
      (predsOf actual current.CENTER) predecessorNodup evidence support
  have localAncestors : ancestorsInto
      (ActualMixedProducerBridge.neighborhoodGraph current)
      current.CENTER = [] := by
    rw [← graphEqual]
    exact accumulatedGraph_center_ancestors_empty current.CENTER processed
      invariant.premisesAvoidCenter
  have sourceRoutes : candidateSourceRoutes actual current.CENTER =
      candidateSourceRoutes
        (ActualMixedProducerBridge.neighborhoodGraph current)
        current.CENTER := by
    simp [candidateSourceRoutes, centerAncestors, localAncestors]
  obtain ⟨outputs, localStep, outputSet⟩ :=
    HCEliminationClassOperational.AccumulatorInvariant.occurrenceStep invariant
      centerDerived usesAvoid (OccurrenceAwareReplay.routesFor table)
      premiseCorrect
  have actualStep : occurrenceStep actual current.CENTER
      (OccurrenceAwareReplay.routesFor table) = .ok outputs := by
    rw [occurrenceStep_congr_of_local_data actual
      (ActualMixedProducerBridge.neighborhoodGraph current) current.CENTER
      centerIncoming (centerAncestors.trans localAncestors.symm)
      sourceRoutes (OccurrenceAwareReplay.routesFor table)]
    exact localStep
  refine ⟨outputs, ?_, outputSet⟩
  rw [OccurrenceAwareReplay.occurrenceReplayAt]
  rw [tableReplay]
  exact actualStep

theorem AccumulatorInvariant.occurrenceReplayAt_boundary
    {partition : List HCClass} {dependencyFor : Nat -> Dep}
    {conclusion : Formula} {current : Neighborhood}
    {processed : List ElimOccurrence}
    (invariant : AccumulatorInvariant partition dependencyFor conclusion
      current processed)
    (actual base : Graph) (fuel : Nat)
    (centerIncoming : get_rule.incoming current.CENTER actual =
      get_rule.incoming current.CENTER
        (ActualMixedProducerBridge.neighborhoodGraph current))
    (centerAncestors : ancestorsInto actual current.CENTER = [])
    (centerDerived : current.CENTER.HYPOTHESIS = false)
    (usesAvoid : UsesAvoidCenter current.CENTER processed)
    (evidence : forall vertex,
      vertex ∈ predsOf actual current.CENTER ->
        PremiseEvidence actual base processed fuel vertex) :
    Exists fun outputs =>
      OccurrenceAwareReplay.occurrenceReplayAt actual (fuel + 2)
        current.CENTER = .ok outputs /\
      SelfCertifying.ListSetEq
        (boundaryDeliveries conclusion (eliminationCandidateRoutes outputs)
          current.OUTGOING)
        (observeEliminationBoundary dependencyFor current) := by
  obtain ⟨outputs, replay, outputSet⟩ :=
    HCEliminationClassCallerBridge.AccumulatorInvariant.occurrenceReplayAt
      invariant actual base fuel centerIncoming
      centerAncestors centerDerived usesAvoid evidence
  have routeSet := eliminationCandidateRoutes_congr outputSet
  rw [expectedFlowRoutes_to_occurrenceRoutes] at routeSet
  have deliveredSet := boundaryDeliveries_routes_congr conclusion
    current.OUTGOING routeSet
  refine ⟨outputs, replay, ?_⟩
  exact SelfCertifying.ListSetEq.trans deliveredSet
    (by simpa [invariant.operationalBoundary] using
      (SelfCertifying.ListSetEq.refl
        (observeEliminationBoundary dependencyFor current)))

end HCEliminationClassCallerBridge
