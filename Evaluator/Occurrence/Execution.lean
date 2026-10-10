import Evaluator.Mixed.Recursive

/-!
# Successor occurrence-aware replay kernel

The successor occurrence-aware evaluator is
`occurrenceReplayAt`; historical `Flow.flowAt` remains a separate model.

This is a successor semantics; it does not change historical `flowAt` or the
restricted candidate replay.  Its local rule kernel is shared by ordinary and
mixed vertices:

* a hypothesis mark contributes source occurrences but does not suppress
  derived occurrences;
* every structurally possible elimination is evaluated by the existing
  dependency-recomputing Cartesian kernel;
* every structurally possible introduction is evaluated independently;
* an introduction output is retained only when its residual continuation is
  not already used by an elimination output at that vertex; and
* a derived occurrence may be reseeded by an explicit ancestral trace only
  when its independently computed residual is a prefix of that trace.

The last point is the executable form of the continuation-separation invariant
which horizontal compression is expected to establish: distinct original rule
occurrences receive distinct painted continuations.  It avoids classifying an
entire predecessor vertex as "an elimination premise", which is unsound after
that predecessor has itself been compressed.

The final clause is deliberately structural.  Exhausted residuals select every
explicit occurrence trace, as before.  An active residual selects only traces
which extend it, so a later class may transport an already-painted occurrence
without cross-routing it to an unrelated use.  A nonempty derived result with
no matching trace remains an explicit error.  Stored edge dependencies are
never consulted.
-/

open DLDS

namespace Evaluator.Occurrence

open Flow Evaluator.Mixed

inductive OccurrenceReplayError where
  | fuelExhausted (vertex : Vertex)
  | nonHypothesisLeaf (vertex : Vertex)
  | noLogicalProduction (vertex : Vertex)
  | mixedProductionLost (vertex : Vertex)
  | unsupportedReseeding (vertex : Vertex)
deriving Repr, DecidableEq

/-- All formula-correct introduction computations at a vertex.  This is
route-level: it deliberately does not exclude a whole predecessor merely
because some occurrence represented by that predecessor participates in an
elimination. -/
def introductionCandidates (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
  match v.FORMULA with
  | Formula.implication _ consequent =>
      (predsOf d v).flatMap fun premise =>
        if premise.FORMULA = consequent then
          candidateIntroductionOutputs d v premise (premiseFlow premise)
        else []
  | _ => []

def residualOccursB (routes : List FlowPair) (residual : ColourPath) : Bool :=
  routes.any fun route => route.2 == residual

/-- Introduction candidates whose continuation is not already accounted for
by an elimination occurrence.  Dependencies are not used for this decision. -/
def separatedIntroductionOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
  let eliminations := candidateEliminationOutputs d v premiseFlow
  (introductionCandidates d v premiseFlow).filter fun route =>
    !residualOccursB eliminations route.2

/-- The shared logical kernel for all currently non-reseeding derived
occurrences. -/
def derivedOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
    (candidateEliminationOutputs d v premiseFlow ++
      separatedIntroductionOutputs d v premiseFlow).eraseDups

/-- Transport independently computed derived routes along explicit occurrence
traces.  The old residual is used only as occurrence identity: a trace is
eligible exactly when it extends that residual.  Dependencies are preserved
and are never read from graph labels. -/
def reseedDerivedRoutes (ancestors : List Ancestral)
    (routes : List FlowPair) : List FlowPair :=
  (ancestors.flatMap fun ancestor =>
    routes.filterMap fun route =>
      if colourPathPrefixB route.2 ancestor.COLOURS then
        some (route.1, ancestor.COLOURS)
      else none).eraseDups

def derivedRoutesCompleteB (routes : List FlowPair) : Bool :=
  routes.all fun route => route.2.isEmpty

@[simp] theorem mem_separatedIntroductionOutputs_iff
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair) :
    route ∈ separatedIntroductionOutputs d v premiseFlow ↔
      route ∈ introductionCandidates d v premiseFlow ∧
        residualOccursB (candidateEliminationOutputs d v premiseFlow)
          route.2 = false := by
  simp [separatedIntroductionOutputs]

theorem separatedIntroduction_not_elimination_residual
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair)
    (member : route ∈ separatedIntroductionOutputs d v premiseFlow) :
    residualOccursB (candidateEliminationOutputs d v premiseFlow)
      route.2 = false :=
  (mem_separatedIntroductionOutputs_iff d v premiseFlow route).mp member |>.2

theorem elimination_mem_derivedOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair)
    (member : route ∈ candidateEliminationOutputs d v premiseFlow) :
    route ∈ derivedOutputs d v premiseFlow := by
  simp [derivedOutputs, member]

theorem separatedIntroduction_mem_derivedOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair)
    (member : route ∈ separatedIntroductionOutputs d v premiseFlow) :
    route ∈ derivedOutputs d v premiseFlow := by
  simp [derivedOutputs, member]

/-- One local successor step, supplied with already-computed predecessor
routes.  Source and derived productions coexist. -/
def occurrenceStep (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) :
    Except OccurrenceReplayError (List FlowPair) :=
  let predecessors := predsOf d v
  if predecessors.isEmpty then
    if v.HYPOTHESIS then .ok (candidateSourceRoutes d v)
    else .error (.nonHypothesisLeaf v)
  else
    let derived := derivedOutputs d v premiseFlow
    if !(ancestorsInto d v).isEmpty && !v.HYPOTHESIS then
      if derived.isEmpty then .error (.noLogicalProduction v)
      else
        let reseeded := reseedDerivedRoutes (ancestorsInto d v) derived
        if reseeded.isEmpty then .error (.unsupportedReseeding v)
        else .ok reseeded
    else if v.HYPOTHESIS then
      if derived.isEmpty then .error (.mixedProductionLost v)
      else .ok ((candidateSourceRoutes d v ++ derived).eraseDups)
    else if derived.isEmpty then .error (.noLogicalProduction v)
    else .ok derived

/-- A nonempty derived result with no structurally extending occurrence trace
is rejected rather than converted to an empty successful result. -/
theorem occurrenceStep_rejects_unmatched_derived_reseeding
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (hasPredecessor : (predsOf d v).isEmpty = false)
    (hasAncestor : (ancestorsInto d v).isEmpty = false)
    (notSource : v.HYPOTHESIS = false)
    (derivedNonempty :
      (derivedOutputs d v premiseFlow).isEmpty = false)
    (unmatched : reseedDerivedRoutes (ancestorsInto d v)
      (derivedOutputs d v premiseFlow) = []) :
    occurrenceStep d v premiseFlow = .error (.unsupportedReseeding v) := by
  simp [occurrenceStep, hasPredecessor, hasAncestor, notSource,
    derivedNonempty, unmatched]

theorem occurrenceStep_reseeds_prefix_matched_derived
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (hasPredecessor : (predsOf d v).isEmpty = false)
    (hasAncestor : (ancestorsInto d v).isEmpty = false)
    (notSource : v.HYPOTHESIS = false)
    (derivedNonempty :
      (derivedOutputs d v premiseFlow).isEmpty = false)
    (matched : (reseedDerivedRoutes (ancestorsInto d v)
      (derivedOutputs d v premiseFlow)).isEmpty = false) :
    occurrenceStep d v premiseFlow =
      .ok (reseedDerivedRoutes (ancestorsInto d v)
        (derivedOutputs d v premiseFlow)) := by
  simp [occurrenceStep, hasPredecessor, hasAncestor, notSource,
    derivedNonempty, matched]

theorem occurrenceStep_reseeds_complete_derived
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (hasPredecessor : (predsOf d v).isEmpty = false)
    (hasAncestor : (ancestorsInto d v).isEmpty = false)
    (notSource : v.HYPOTHESIS = false)
    (derivedNonempty :
      (derivedOutputs d v premiseFlow).isEmpty = false)
    (complete : derivedRoutesCompleteB
      (derivedOutputs d v premiseFlow) = true) :
    occurrenceStep d v premiseFlow =
      .ok (reseedDerivedRoutes (ancestorsInto d v)
        (derivedOutputs d v premiseFlow)) := by
  have routesNonempty : derivedOutputs d v premiseFlow ≠ [] := by
    simpa [List.isEmpty_iff] using derivedNonempty
  have ancestorsNonempty : ancestorsInto d v ≠ [] := by
    simpa [List.isEmpty_iff] using hasAncestor
  obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ routesNonempty
  obtain ⟨ancestor, ancestorMember⟩ :=
    List.exists_mem_of_ne_nil _ ancestorsNonempty
  have residualEmpty : route.2 = [] := by
    have routeComplete := List.all_eq_true.mp complete route routeMember
    simpa using routeComplete
  have member : (route.1, ancestor.COLOURS) ∈
      reseedDerivedRoutes (ancestorsInto d v)
        (derivedOutputs d v premiseFlow) := by
    rw [reseedDerivedRoutes, List.mem_eraseDups, List.mem_flatMap]
    refine ⟨ancestor, ancestorMember, ?_⟩
    rw [List.mem_filterMap]
    refine ⟨route, routeMember, ?_⟩
    simp [residualEmpty, colourPathPrefixB]
  have matched : (reseedDerivedRoutes (ancestorsInto d v)
      (derivedOutputs d v premiseFlow)).isEmpty = false := by
    rw [List.isEmpty_eq_false_iff]
    exact List.ne_nil_of_mem member
  exact occurrenceStep_reseeds_prefix_matched_derived d v premiseFlow
    hasPredecessor hasAncestor notSource derivedNonempty matched

def routesFor (table : List (Vertex × List FlowPair)) (v : Vertex) :
    List FlowPair :=
  match table.find? fun entry => entry.1 == v with
  | some entry => entry.2
  | none => []

/-- Fuelled whole-graph successor evaluator.  Errors in a premise are
propagated rather than converted to an empty route set. -/
def occurrenceReplayAt (d : Graph) :
    Nat -> Vertex -> Except OccurrenceReplayError (List FlowPair)
  | 0, vertex => .error (.fuelExhausted vertex)
  | fuel + 1, vertex => do
      let predecessors := predsOf d vertex
      let premiseTable <- predecessors.mapM fun premise => do
        let routes <- occurrenceReplayAt d fuel premise
        return (premise, routes)
      occurrenceStep d vertex (routesFor premiseTable)

def occurrenceRoots (d : Graph) : List Vertex :=
  d.NODES.filter fun vertex => (get_rule.outgoing vertex d).isEmpty

def routeClosedB (route : FlowPair) : Bool :=
  route.1.isEmpty && route.2.isEmpty

/-- Strong raw-root condition: roots and route sets are nonempty, all errors
reject, and every raw route is dependency-closed and trace-complete. -/
def occurrenceRootsCompleteB (d : Graph) (fuel : Nat) : Bool :=
  let roots := occurrenceRoots d
  !roots.isEmpty && roots.all fun root =>
    match occurrenceReplayAt d fuel root with
    | .error _ => false
    | .ok routes => !routes.isEmpty && routes.all routeClosedB

end Evaluator.Occurrence
