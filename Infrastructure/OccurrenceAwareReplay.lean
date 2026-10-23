import Infrastructure.MixedRuleReplay



































open Semantic

namespace OccurrenceAwareReplay

open FlowSpec MixedRuleReplay

inductive OccurrenceReplayError where
  | fuelExhausted (vertex : Vertex)
  | nonHypothesisLeaf (vertex : Vertex)
  | noLogicalProduction (vertex : Vertex)
  | mixedProductionLost (vertex : Vertex)
  | unsupportedReseeding (vertex : Vertex)
deriving Repr, DecidableEq





def introductionSelections (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
  match v.FORMULA with
  | Formula.implication _ consequent =>
      (predsOf d v).flatMap fun premise =>
        if premise.FORMULA = consequent then
          selectionIntroductionOutputs d v premise (premiseFlow premise)
        else []
  | _ => []

def residualOccursB (routes : List FlowPair) (residual : ColourPath) : Bool :=
  routes.any fun route => route.2 == residual



def separatedIntroductionOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
  let eliminations := selectionEliminationOutputs d v premiseFlow
  (introductionSelections d v premiseFlow).filter fun route =>
    !residualOccursB eliminations route.2



def derivedOutputs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : List FlowPair :=
    (selectionEliminationOutputs d v premiseFlow ++
      separatedIntroductionOutputs d v premiseFlow).eraseDups





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
      route ∈ introductionSelections d v premiseFlow ∧
        residualOccursB (selectionEliminationOutputs d v premiseFlow)
          route.2 = false := by
  simp [separatedIntroductionOutputs]

theorem separatedIntroduction_not_elimination_residual
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair)
    (member : route ∈ separatedIntroductionOutputs d v premiseFlow) :
    residualOccursB (selectionEliminationOutputs d v premiseFlow)
      route.2 = false :=
  (mem_separatedIntroductionOutputs_iff d v premiseFlow route).mp member |>.2

theorem elimination_mem_derivedOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair)
    (member : route ∈ selectionEliminationOutputs d v premiseFlow) :
    route ∈ derivedOutputs d v premiseFlow := by
  simp [derivedOutputs, member]

theorem separatedIntroduction_mem_derivedOutputs
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair)
    (member : route ∈ separatedIntroductionOutputs d v premiseFlow) :
    route ∈ derivedOutputs d v premiseFlow := by
  simp [derivedOutputs, member]



def occurrenceStep (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) :
    Except OccurrenceReplayError (List FlowPair) :=
  let predecessors := predsOf d v
  if predecessors.isEmpty then
    if v.HYPOTHESIS then .ok (selectionSourceRoutes d v)
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
      else .ok ((selectionSourceRoutes d v ++ derived).eraseDups)
    else if derived.isEmpty then .error (.noLogicalProduction v)
    else .ok derived



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



def occurrenceRootsCompleteB (d : Graph) (fuel : Nat) : Bool :=
  let roots := occurrenceRoots d
  !roots.isEmpty && roots.all fun root =>
    match occurrenceReplayAt d fuel root with
    | .error _ => false
    | .ok routes => !routes.isEmpty && routes.all routeClosedB

end OccurrenceAwareReplay
