import Semantic.ProperHCInput














open Semantic FlowSpec

namespace LocalSpecificationCandidate




structure BranchAddress where
  choices : List Nat
deriving Repr, DecidableEq


structure EdgeColourTrace where
  colours : List Nat
deriving Repr, DecidableEq



structure CandidateResidual where
  colours : List Nat
deriving Repr, DecidableEq

def outgoing (d : Graph) (v : Vertex) : List Deduction :=
  get_rule.outgoing v d

def uniqueOutgoing? (d : Graph) (v : Vertex) : Option Deduction :=
  match outgoing d v with
  | [selected] => some selected
  | _ => none

def selectedOutgoing? (d : Graph) (v : Vertex) (colour : Nat) :
    Option Deduction :=
  match (outgoing d v).filter (fun edge => edge.COLOUR = colour) with
  | [selected] => some selected
  | _ => none

theorem selectedOutgoing_of_unique {d : Graph} {v : Vertex}
    {selected : Deduction}
    (unique : uniqueOutgoing? d v = some selected) :
    selectedOutgoing? d v selected.COLOUR = some selected := by
  unfold uniqueOutgoing? at unique
  unfold selectedOutgoing?
  generalize outgoing d v = edges at unique ⊢
  cases edges with
  | nil => simp at unique
  | cons first rest =>
      cases rest with
      | nil =>
          simp at unique
          subst first
          simp
      | cons second tail => simp at unique



def followEdgeColours (d : Graph) : Nat -> Vertex -> List Nat -> Option Vertex
  | _, current, [] => some current
  | 0, _, _ :: _ => none
  | fuel + 1, current, colour :: rest =>
      match selectedOutgoing? d current colour with
      | some selected => followEdgeColours d fuel selected.END rest
      | none => none

def decodeEdgeTrace (d : Graph) (start : Vertex) (trace : EdgeColourTrace) :
    Option Vertex :=
  followEdgeColours d (d.NODES.length + trace.colours.length + 1)
    start trace.colours









def followBranchChoicesTo (d : Graph) (finish : Vertex) :
    Nat -> Vertex -> List Nat -> Option Vertex
  | 0, _, _ => none
  | fuel + 1, current, choices =>
      if current = finish then
        if choices = [] then some current else none
      else
        match uniqueOutgoing? d current with
        | some selected =>
            followBranchChoicesTo d finish fuel selected.END choices
        | none =>
            match choices with
            | [] => none
            | colour :: rest =>
                match selectedOutgoing? d current colour with
                | some selected =>
                    followBranchChoicesTo d finish fuel selected.END rest
                | none => none

def decodeBranchAddressTo (d : Graph) (start finish : Vertex)
    (address : BranchAddress) : Option Vertex :=
  followBranchChoicesTo d finish
    (d.NODES.length + address.choices.length + 1) start address.choices

def edgeColoursOfEdges (edges : List Deduction) : List Nat :=
  edges.map (fun edge => edge.COLOUR)

def branchChoicesOfEdges (d : Graph) : List Deduction -> List Nat
  | [] => []
  | first :: rest =>
      match uniqueOutgoing? d first.START with
      | some _ => branchChoicesOfEdges d rest
      | none => first.COLOUR :: branchChoicesOfEdges d rest

def edgeTraceOfEdges (edges : List Deduction) : EdgeColourTrace :=
  ⟨edgeColoursOfEdges edges⟩

def branchAddressOfEdges (d : Graph) (edges : List Deduction) : BranchAddress :=
  ⟨branchChoicesOfEdges d edges⟩




inductive EndpointPath (d : Graph) (finish : Vertex) :
    Vertex -> List Deduction -> Prop
  | done : EndpointPath d finish finish []
  | unary {current : Vertex} {edge : Deduction} {rest : List Deduction}
      (beforeFinish : current ≠ finish)
      (edgeStart : edge.START = current)
      (edgeMember : edge ∈ d.EDGES)
      (unique : uniqueOutgoing? d current = some edge)
      (tail : EndpointPath d finish edge.END rest) :
      EndpointPath d finish current (edge :: rest)
  | branch {current : Vertex} {edge : Deduction} {rest : List Deduction}
      (beforeFinish : current ≠ finish)
      (edgeStart : edge.START = current)
      (edgeMember : edge ∈ d.EDGES)
      (notUnary : uniqueOutgoing? d current = none)
      (selected : selectedOutgoing? d current edge.COLOUR = some edge)
      (tail : EndpointPath d finish edge.END rest) :
      EndpointPath d finish current (edge :: rest)

theorem endpointPath_edgeColours_decode
    {d : Graph} {start finish : Vertex} {edges : List Deduction}
    (path : EndpointPath d finish start edges) :
    followEdgeColours d (edges.length + 1) start (edgeColoursOfEdges edges) =
      some finish := by
  induction path with
  | done => simp [followEdgeColours, edgeColoursOfEdges]
  | @unary current selected rest beforeFinish edgeStart edgeMember unique tail ih =>
      subst current
      simp only [edgeColoursOfEdges, List.map_cons, List.length_cons]
      change followEdgeColours d (Nat.succ (rest.length + 1)) selected.START
        (selected.COLOUR :: edgeColoursOfEdges rest) = some finish
      rw [followEdgeColours, selectedOutgoing_of_unique unique]
      exact ih
  | @branch current selected rest beforeFinish edgeStart edgeMember notUnary
      selectedByColour tail ih =>
      subst current
      simp only [edgeColoursOfEdges, List.map_cons, List.length_cons]
      change followEdgeColours d (Nat.succ (rest.length + 1)) selected.START
        (selected.COLOUR :: edgeColoursOfEdges rest) = some finish
      rw [followEdgeColours, selectedByColour]
      exact ih

theorem endpointPath_branchChoices_decode
    {d : Graph} {start finish : Vertex} {edges : List Deduction}
    (path : EndpointPath d finish start edges) :
    followBranchChoicesTo d finish (edges.length + 1) start
      (branchChoicesOfEdges d edges) = some finish := by
  induction path with
  | done => simp [followBranchChoicesTo, branchChoicesOfEdges]
  | @unary current selected rest beforeFinish edgeStart edgeMember unique tail ih =>
      subst current
      rw [List.length_cons, branchChoicesOfEdges, unique,
        followBranchChoicesTo.eq_def]
      simp only [beforeFinish, if_false, unique]
      exact ih
  | @branch current selected rest beforeFinish edgeStart edgeMember notUnary
      selectedByColour tail ih =>
      subst current
      rw [List.length_cons, branchChoicesOfEdges, notUnary,
        followBranchChoicesTo.eq_def]
      simp only [beforeFinish, if_false, notUnary, selectedByColour]
      exact ih




theorem endpointPath_encodings_same_endpoint
    {d : Graph} {start finish : Vertex} {edges : List Deduction}
    (path : EndpointPath d finish start edges) :
    followEdgeColours d (edges.length + 1) start
        (edgeTraceOfEdges edges).colours = some finish ∧
      followBranchChoicesTo d finish (edges.length + 1) start
        (branchAddressOfEdges d edges).choices = some finish := by
  exact ⟨endpointPath_edgeColours_decode path,
    endpointPath_branchChoices_decode path⟩







inductive MixedProduction where
  | source
  | introduction
  | elimination
deriving Repr, DecidableEq

structure CandidateRoute where
  production : MixedProduction
  dependency : List Formula
  residual : CandidateResidual
deriving Repr, DecidableEq








structure MixedRuntimeView (d : Graph) where
  mixed : Vertex
  major : Vertex
  minor : Vertex
  sourceUse : Vertex
  derivedUse : Vertex
  majorEdge : Deduction
  minorEdge : Deduction
  sourceEdge : Deduction
  derivedEdge : Deduction
  majorTrace : Ancestral
  minorTrace : Ancestral
  mixedMember : mixed ∈ d.NODES
  majorMember : major ∈ d.NODES
  minorMember : minor ∈ d.NODES
  majorIsHypothesis : major.HYPOTHESIS = true
  minorIsHypothesis : minor.HYPOTHESIS = true
  mixedHasSourceMark : mixed.HYPOTHESIS = true
  majorFormula : major.FORMULA = minor.FORMULA >> mixed.FORMULA
  incomingExactly : get_rule.incoming mixed d = [majorEdge, minorEdge]
  majorEdgeShape : majorEdge.START = major ∧ majorEdge.END = mixed ∧
    majorEdge.COLOUR = 0
  minorEdgeShape : minorEdge.START = minor ∧ minorEdge.END = mixed ∧
    minorEdge.COLOUR = 0
  outgoingExactly : get_rule.outgoing mixed d = [sourceEdge, derivedEdge]
  sourceEdgeShape : sourceEdge.START = mixed ∧ sourceEdge.END = sourceUse ∧
    sourceEdge.COLOUR = 0
  derivedEdgeShape : derivedEdge.START = mixed ∧
    derivedEdge.END = derivedUse ∧ derivedEdge.COLOUR ≠ 0
  majorTraceShape : majorTrace.START = derivedUse ∧
    majorTrace.END = major ∧
    majorTrace.COLOURS = [0, derivedEdge.COLOUR]
  minorTraceShape : minorTrace.START = derivedUse ∧
    minorTrace.END = minor ∧
    minorTrace.COLOURS = [0, derivedEdge.COLOUR]
  majorTraceMember : majorTrace ∈ d.PATHS
  minorTraceMember : minorTrace ∈ d.PATHS




structure MixedOriginWitness (before after : Graph)
    (view : MixedRuntimeView after)
    (sourceOccurrence derivedOccurrence : Vertex) : Prop where
  occurrencesDistinct : sourceOccurrence ≠ derivedOccurrence
  sourceMember : sourceOccurrence ∈ before.NODES
  derivedMember : derivedOccurrence ∈ before.NODES
  sourceWasHypothesis : sourceOccurrence.HYPOTHESIS = true
  derivedWasNotHypothesis : derivedOccurrence.HYPOTHESIS = false
  sourceFormula : sourceOccurrence.FORMULA = view.mixed.FORMULA
  derivedFormula : derivedOccurrence.FORMULA = view.mixed.FORMULA
  sameLevel : sourceOccurrence.LEVEL = derivedOccurrence.LEVEL
  sourceContinuation : ∃ edge ∈ before.EDGES,
    edge.START = sourceOccurrence ∧ edge.END = view.sourceUse ∧
      edge.COLOUR = 0
  derivedContinuation : ∃ edge ∈ before.EDGES,
    edge.START = derivedOccurrence ∧ edge.END = view.derivedUse ∧
      edge.COLOUR = 0
  majorPremise : ∃ edge ∈ before.EDGES,
    edge.START = view.major ∧ edge.END = derivedOccurrence ∧
      edge.COLOUR = 0
  minorPremise : ∃ edge ∈ before.EDGES,
    edge.START = view.minor ∧ edge.END = derivedOccurrence ∧
      edge.COLOUR = 0




def consumeCandidate (edge : Deduction) (residual : CandidateResidual) :
    Option CandidateResidual :=
  match residual.colours with
  | [] => if edge.COLOUR = 0 then some ⟨[]⟩ else none
  | colour :: rest =>
      if edge.COLOUR = colour then some ⟨rest⟩ else none

def sourceRoute {d : Graph} (view : MixedRuntimeView d) : CandidateRoute :=
  { production := .source
    dependency := normalizeDep [view.mixed.FORMULA]
    residual := ⟨[]⟩ }

def eliminationRoute {d : Graph} (view : MixedRuntimeView d) : CandidateRoute :=
  { production := .elimination
    dependency := depUnion (normalizeDep [view.minor.FORMULA])
      (normalizeDep [view.major.FORMULA])
    residual := ⟨[view.derivedEdge.COLOUR]⟩ }



def candidateMixedRoutes {d : Graph} (view : MixedRuntimeView d) :
    List CandidateRoute :=
  [sourceRoute view, eliminationRoute view]

theorem sourceRoute_uses_formula_not_stored_dependency
    {d : Graph} (view : MixedRuntimeView d) :
    (sourceRoute view).dependency = normalizeDep [view.mixed.FORMULA] := rfl

theorem eliminationRoute_uses_logical_premises_not_stored_dependency
    {d : Graph} (view : MixedRuntimeView d) :
    (eliminationRoute view).dependency =
      depUnion (normalizeDep [view.minor.FORMULA])
        (normalizeDep [view.major.FORMULA]) := rfl

theorem eliminationRoute_matches_recorded_edge_trace_tails
    {d : Graph} (view : MixedRuntimeView d) :
    view.majorTrace.COLOURS =
        0 :: (eliminationRoute view).residual.colours ∧
      view.minorTrace.COLOURS =
        0 :: (eliminationRoute view).residual.colours := by
  rcases view.majorTraceShape with ⟨_, _, majorColours⟩
  rcases view.minorTraceShape with ⟨_, _, minorColours⟩
  simpa [eliminationRoute] using And.intro majorColours minorColours

theorem sourceRoute_selects_only_source_use
    {d : Graph} (view : MixedRuntimeView d) :
    consumeCandidate view.sourceEdge (sourceRoute view).residual = some ⟨[]⟩ ∧
      consumeCandidate view.derivedEdge (sourceRoute view).residual = none := by
  rcases view.sourceEdgeShape with ⟨_, _, sourceColour⟩
  rcases view.derivedEdgeShape with ⟨_, _, derivedColour⟩
  constructor
  · simp [sourceRoute, consumeCandidate, sourceColour]
  · change (if view.derivedEdge.COLOUR = 0 then
      some ({ colours := [] } : CandidateResidual) else none) = none
    rw [if_neg derivedColour]

theorem eliminationRoute_selects_only_derived_use
    {d : Graph} (view : MixedRuntimeView d) :
    consumeCandidate view.derivedEdge (eliminationRoute view).residual =
        some ⟨[]⟩ ∧
      consumeCandidate view.sourceEdge (eliminationRoute view).residual = none := by
  rcases view.sourceEdgeShape with ⟨_, _, sourceColour⟩
  rcases view.derivedEdgeShape with ⟨_, _, derivedColour⟩
  have zeroNeDerived : 0 ≠ view.derivedEdge.COLOUR :=
    fun equal => derivedColour equal.symm
  constructor
  · simp [eliminationRoute, consumeCandidate]
  · simp [eliminationRoute, consumeCandidate, sourceColour, zeroNeDerived]

theorem candidateMixedRoutes_membership
    {d : Graph} (view : MixedRuntimeView d) (route : CandidateRoute) :
    route ∈ candidateMixedRoutes view ↔
      route = sourceRoute view ∨ route = eliminationRoute view := by
  simp [candidateMixedRoutes]

structure LocalMixedCorrespondence (before after : Graph)
    (view : MixedRuntimeView after) : Prop where
  origins : ∃ sourceOccurrence derivedOccurrence,
    MixedOriginWitness before after view sourceOccurrence derivedOccurrence
  exactRoutes : ∀ route,
    route ∈ candidateMixedRoutes view ↔
      route = sourceRoute view ∨ route = eliminationRoute view
  sourceDependency : (sourceRoute view).dependency =
    normalizeDep [view.mixed.FORMULA]
  derivedDependency : (eliminationRoute view).dependency =
    depUnion (normalizeDep [view.minor.FORMULA])
      (normalizeDep [view.major.FORMULA])
  derivedTraceTails :
    view.majorTrace.COLOURS =
        0 :: (eliminationRoute view).residual.colours ∧
      view.minorTrace.COLOURS =
        0 :: (eliminationRoute view).residual.colours
  sourceOnlyItsUse :
    consumeCandidate view.sourceEdge (sourceRoute view).residual = some ⟨[]⟩ ∧
      consumeCandidate view.derivedEdge (sourceRoute view).residual = none
  derivedOnlyItsUse :
    consumeCandidate view.derivedEdge (eliminationRoute view).residual =
        some ⟨[]⟩ ∧
      consumeCandidate view.sourceEdge (eliminationRoute view).residual = none





theorem mixedMerge_local_correspondence
    {before after : Graph} (view : MixedRuntimeView after)
    {sourceOccurrence derivedOccurrence : Vertex}
    (origins : MixedOriginWitness before after view
      sourceOccurrence derivedOccurrence) :
    LocalMixedCorrespondence before after view := by
  refine
    { origins := ⟨sourceOccurrence, derivedOccurrence, origins⟩
      exactRoutes := ?_
      sourceDependency := sourceRoute_uses_formula_not_stored_dependency view
      derivedDependency :=
        eliminationRoute_uses_logical_premises_not_stored_dependency view
      derivedTraceTails :=
        eliminationRoute_matches_recorded_edge_trace_tails view
      sourceOnlyItsUse := sourceRoute_selects_only_source_use view
      derivedOnlyItsUse := eliminationRoute_selects_only_derived_use view }
  exact candidateMixedRoutes_membership view

end LocalSpecificationCandidate
