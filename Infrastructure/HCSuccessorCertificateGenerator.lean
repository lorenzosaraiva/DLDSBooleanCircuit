import Infrastructure.HCSuccessorCertificateSoundness














open Semantic FlowSpec
open OccurrenceAwareReplay
open CertificateModel
open HCSuccessorFixedGraphReference HCSuccessorLocalCertificate
open HCSuccessorCertificatePositions HCSuccessorCertificateSoundness

namespace HCSuccessorCertificateGenerator

def PredecessorsListed (d : Graph) : Prop :=
  ∀ vertex, vertex ∈ d.NODES -> ∀ premise,
    premise ∈ predsOf d vertex -> premise ∈ d.NODES

def predecessorsListedB (d : Graph) : Bool :=
  d.NODES.all fun vertex =>
    (predsOf d vertex).all fun premise => d.NODES.contains premise

theorem predecessorsListedB_sound (d : Graph)
    (valid : predecessorsListedB d = true) : PredecessorsListed d := by
  intro vertex vertexMember premise premiseMember
  have vertexCheck := List.all_eq_true.mp valid vertex vertexMember
  have premiseCheck := List.all_eq_true.mp vertexCheck premise premiseMember
  simpa using premiseCheck

def generatedPremiseTable (d : Graph)
    (routes : Vertex -> List FlowPair) (vertex : Vertex) :
    List (Vertex × List FlowPair) :=
  (predsOf d vertex).map fun premise => (premise, routes premise)

def generatedLocalPosition (d : Graph)
    (routes : Vertex -> List FlowPair) (vertex : Vertex) :
    LocalRouteCertificate :=
  { node := vertex
    premiseTable := generatedPremiseTable d routes vertex
    claimedRoutes := routes vertex }

def generateSuccessorCertificateFromRoutes (d : Graph)
    (routes : Vertex -> List FlowPair) : SuccessorCertificate :=
  { positions := d.NODES.map (generatedLocalPosition d routes) }

def LocalRouteFixedPoint (d : Graph)
    (routes : Vertex -> List FlowPair) : Prop :=
  ∀ vertex, vertex ∈ d.NODES ->
    occurrenceStep d vertex
      (OccurrenceAwareReplay.routesFor
        (generatedPremiseTable d routes vertex)) = .ok (routes vertex)

def RootRouteAssignmentComplete (d : Graph)
    (routes : Vertex -> List FlowPair) : Prop :=
  (occurrenceRoots d).isEmpty = false ∧
    ∀ root, root ∈ occurrenceRoots d ->
      routes root ≠ [] ∧
      ∀ route, route ∈ routes root -> routeClosedB route = true

theorem claimedRoutesAt?_generated
    (d : Graph) (routes : Vertex -> List FlowPair)
    (vertex : Vertex) (member : vertex ∈ d.NODES) :
    claimedRoutesAt? (generateSuccessorCertificateFromRoutes d routes)
      vertex = some (routes vertex) := by
  unfold claimedRoutesAt? generateSuccessorCertificateFromRoutes
  generalize nodesEq : d.NODES = nodes at member ⊢
  clear nodesEq
  revert member
  induction nodes with
  | nil => simp
  | cons head tail ih =>
      intro member
      simp only [List.map_cons, List.find?_cons]
      by_cases equal : head = vertex
      · subst head
        simp [generatedLocalPosition]
      · have tailMember : vertex ∈ tail := by
          have notEqual : vertex ≠ head := Ne.symm equal
          simpa [notEqual] using member
        have beqFalse :
            ((generatedLocalPosition d routes head).node == vertex) = false := by
          simp [generatedLocalPosition, equal]
        rw [beqFalse]
        exact ih tailMember

theorem generated_node_order
    (d : Graph) (routes : Vertex -> List FlowPair) :
    (generateSuccessorCertificateFromRoutes d routes).positions.map
        LocalRouteCertificate.node = d.NODES := by
  simp [generateSuccessorCertificateFromRoutes, generatedLocalPosition,
    Function.comp_def]

theorem generateSuccessorCertificateFromRoutes_complete
    (d : Graph) (routes : Vertex -> List FlowPair)
    (predecessorsListed : PredecessorsListed d)
    (fixedPoint : LocalRouteFixedPoint d routes)
    (rootsComplete : RootRouteAssignmentComplete d routes) :
    successorFormationB d
      (generateSuccessorCertificateFromRoutes d routes) = true := by
  have nodeOrderB :
      ((generateSuccessorCertificateFromRoutes d routes).positions.map
        LocalRouteCertificate.node == d.NODES) = true := by
    simpa using generated_node_order d routes
  have localB :
      (generateSuccessorCertificateFromRoutes d routes).positions.all
        (localRouteCertificateB d) = true := by
    apply List.all_eq_true.mpr
    intro position positionMember
    obtain ⟨vertex, vertexMember, positionEq⟩ :=
      List.mem_map.mp positionMember
    subst position
    have step := fixedPoint vertex vertexMember
    change (match occurrenceStep d vertex
        (OccurrenceAwareReplay.routesFor
          (generatedPremiseTable d routes vertex)) with
      | .error _ => false
      | .ok outputs =>
          ((generatedPremiseTable d routes vertex).map Prod.fst ==
            predsOf d vertex) && setEqB outputs (routes vertex)) = true
    rw [step]
    simp [generatedPremiseTable, Function.comp_def, setEqB]
  have premiseB :
      (generateSuccessorCertificateFromRoutes d routes).positions.all
        (fun position => position.premiseTable.all
          (premiseEntryCoherentB
            (generateSuccessorCertificateFromRoutes d routes))) = true := by
    apply List.all_eq_true.mpr
    intro position positionMember
    obtain ⟨vertex, vertexMember, positionEq⟩ :=
      List.mem_map.mp positionMember
    subst position
    apply List.all_eq_true.mpr
    intro entry entryMember
    obtain ⟨premise, premiseMember, entryEq⟩ :=
      List.mem_map.mp entryMember
    subst entry
    have premiseNode := predecessorsListed vertex vertexMember premise
      premiseMember
    have lookup := claimedRoutesAt?_generated d routes premise premiseNode
    simp [premiseEntryCoherentB, lookup, setEqB]
  have rootSetB : !(occurrenceRoots d).isEmpty = true := by
    simp [rootsComplete.1]
  have rootB : (occurrenceRoots d).all
      (rootClaimsCompleteB
        (generateSuccessorCertificateFromRoutes d routes)) = true := by
    apply List.all_eq_true.mpr
    intro root rootMember
    have rootNode : root ∈ d.NODES := by
      simp [occurrenceRoots] at rootMember
      exact rootMember.1
    have lookup := claimedRoutesAt?_generated d routes root rootNode
    have complete := rootsComplete.2 root rootMember
    have nonemptyB : !(routes root).isEmpty = true := by
      have isNotEmpty : (routes root).isEmpty = false := by
        rw [List.isEmpty_eq_false_iff]
        exact complete.1
      simp [isNotEmpty]
    have closedB : (routes root).all routeClosedB = true :=
      List.all_eq_true.mpr complete.2
    simp [rootClaimsCompleteB, lookup, complete.1, closedB]
  simpa [successorFormationB, Bool.and_assoc] using
    And.intro nodeOrderB
      (And.intro localB (And.intro premiseB (And.intro rootSetB rootB)))

def replayRoutes (d : Graph) (fuel : Nat) (vertex : Vertex) :
    List FlowPair :=
  match occurrenceReplayAt d fuel vertex with
  | .error _ => []
  | .ok routes => routes

def AllNodeReplaySuccess (d : Graph) (fuel : Nat) : Prop :=
  ∀ vertex, vertex ∈ d.NODES ->
    ∃ routes, occurrenceReplayAt d fuel vertex = .ok routes

def allNodeReplaySuccessB (d : Graph) (fuel : Nat) : Bool :=
  d.NODES.all fun vertex =>
    match occurrenceReplayAt d fuel vertex with
    | .error _ => false
    | .ok _ => true

theorem allNodeReplaySuccessB_sound (d : Graph) (fuel : Nat)
    (valid : allNodeReplaySuccessB d fuel = true) :
    AllNodeReplaySuccess d fuel := by
  intro vertex vertexMember
  have checked := List.all_eq_true.mp valid vertex vertexMember
  cases replay : occurrenceReplayAt d fuel vertex with
  | error reason => simp [replay] at checked
  | ok routes => exact ⟨routes, rfl⟩

def ReplayStableOnNodes (d : Graph) (fuel : Nat) : Prop :=
  ∀ vertex, vertex ∈ d.NODES ->
    occurrenceReplayAt d (fuel + 1) vertex =
      occurrenceReplayAt d fuel vertex

def replayStableOnNodesB (d : Graph) (fuel : Nat) : Bool :=
  d.NODES.all fun vertex =>
    occurrenceReplayAt d (fuel + 1) vertex ==
      occurrenceReplayAt d fuel vertex

theorem replayStableOnNodesB_sound (d : Graph) (fuel : Nat)
    (valid : replayStableOnNodesB d fuel = true) :
    ReplayStableOnNodes d fuel := by
  intro vertex vertexMember
  have checked := List.all_eq_true.mp valid vertex vertexMember
  simpa using checked

theorem actualPremises_congr
    (left right : Vertex -> Except OccurrenceReplayError (List FlowPair)) :
    ∀ premises,
      (∀ premise, premise ∈ premises -> left premise = right premise) ->
      actualPremises left premises = actualPremises right premises := by
  intro premises pointwise
  induction premises with
  | nil => rfl
  | cons premise premises ih =>
      have head := pointwise premise (by simp)
      have tail : ∀ selection,
          selection ∈ premises -> left selection = right selection := by
        intro selection member
        exact pointwise selection (by simp [member])
      simp [actualPremises, head, ih tail]

theorem occurrenceReplayAt_stable_by_level
    (d : Graph) (wellFounded : PredecessorLevelWellFounded d) :
    ∀ level vertex,
      vertex.LEVEL = level -> vertex ∈ d.NODES ->
      ∀ fuel, level < fuel ->
        occurrenceReplayAt d (fuel + 1) vertex =
          occurrenceReplayAt d fuel vertex := by
  intro level
  induction level using Nat.strong_induction_on with
  | h level ih =>
      intro vertex vertexLevel vertexMember fuel fuelBound
      cases fuel with
      | zero => omega
      | succ predecessorFuel =>
          have premiseReplay : ∀ premise,
              premise ∈ predsOf d vertex ->
              occurrenceReplayAt d (predecessorFuel + 1) premise =
                occurrenceReplayAt d predecessorFuel premise := by
            intro premise premiseMember
            obtain ⟨premiseNode, premiseLevel⟩ :=
              wellFounded vertex vertexMember premise premiseMember
            apply ih premise.LEVEL
            · omega
            · rfl
            · exact premiseNode
            · omega
          have tablesEqual := actualPremises_congr
            (occurrenceReplayAt d (predecessorFuel + 1))
            (occurrenceReplayAt d predecessorFuel)
            (predsOf d vertex) premiseReplay
          rw [occurrenceReplayAt_succ, occurrenceReplayAt_succ, tablesEqual]

theorem replayStableOnNodes_of_level
    (d : Graph) (fuel : Nat)
    (wellFounded : PredecessorLevelWellFounded d)
    (nodeFuel : ∀ vertex, vertex ∈ d.NODES -> vertex.LEVEL < fuel) :
    ReplayStableOnNodes d fuel := by
  intro vertex vertexMember
  exact occurrenceReplayAt_stable_by_level d wellFounded vertex.LEVEL vertex
    rfl vertexMember fuel (nodeFuel vertex vertexMember)

theorem actualPremises_replayRoutes
    (d : Graph) (fuel : Nat)
    (success : AllNodeReplaySuccess d fuel) :
    ∀ premises,
      (∀ premise, premise ∈ premises -> premise ∈ d.NODES) ->
      actualPremises (occurrenceReplayAt d fuel) premises =
        .ok (premises.map fun premise =>
          (premise, replayRoutes d fuel premise)) := by
  intro premises listed
  induction premises with
  | nil => rfl
  | cons premise premises ih =>
      obtain ⟨routes, replay⟩ := success premise (listed premise (by simp))
      have tailListed : ∀ selection,
          selection ∈ premises -> selection ∈ d.NODES := by
        intro selection member
        exact listed selection (by simp [member])
      have tail := ih tailListed
      simp [actualPremises, replayRoutes, replay, tail]

theorem replayRoutes_fixedPoint
    (d : Graph) (fuel : Nat)
    (predecessorsListed : PredecessorsListed d)
    (success : AllNodeReplaySuccess d fuel)
    (stable : ReplayStableOnNodes d fuel) :
    LocalRouteFixedPoint d (replayRoutes d fuel) := by
  intro vertex vertexMember
  obtain ⟨routes, replay⟩ := success vertex vertexMember
  have premises := actualPremises_replayRoutes d fuel success
    (predsOf d vertex)
    (predecessorsListed vertex vertexMember)
  have successor := occurrenceReplayAt_succ d fuel vertex
  rw [premises] at successor
  have stability := stable vertex vertexMember
  rw [successor, replay] at stability
  have routeEq : replayRoutes d fuel vertex = routes := by
    simp [replayRoutes, replay]
  rw [routeEq]
  exact stability

theorem replayRoutes_rootsComplete
    (d : Graph) (fuel : Nat)
    (complete : occurrenceRootsCompleteB d fuel = true) :
    RootRouteAssignmentComplete d (replayRoutes d fuel) := by
  unfold occurrenceRootsCompleteB at complete
  have parts : !(occurrenceRoots d).isEmpty = true ∧
      (occurrenceRoots d).all (fun root =>
        match occurrenceReplayAt d fuel root with
        | .error _ => false
        | .ok routes => !routes.isEmpty && routes.all routeClosedB) = true := by
    simpa using complete
  constructor
  · cases emptyEq : (occurrenceRoots d).isEmpty with
    | false => rfl
    | true => simp [emptyEq] at parts
  · intro root rootMember
    have rootCheck := List.all_eq_true.mp parts.2 root rootMember
    cases replay : occurrenceReplayAt d fuel root with
    | error reason => simp [replay] at rootCheck
    | ok routes =>
        have checks : !routes.isEmpty = true ∧
            routes.all routeClosedB = true := by
          simpa [replay] using rootCheck
        have routeEq : replayRoutes d fuel root = routes := by
          simp [replayRoutes, replay]
        rw [routeEq]
        constructor
        · have isNotEmpty : routes.isEmpty = false := by
            cases emptyEq : routes.isEmpty with
            | false => rfl
            | true => simp [emptyEq] at checks
          simpa [List.isEmpty_eq_false_iff] using isNotEmpty
        · exact List.all_eq_true.mp checks.2

def generateSuccessorCertificate (d : Graph) (fuel : Nat) :
    SuccessorCertificate :=
  generateSuccessorCertificateFromRoutes d (replayRoutes d fuel)


theorem generateSuccessorCertificate_complete
    (d : Graph) (fuel : Nat)
    (predecessorsListed : PredecessorsListed d)
    (success : AllNodeReplaySuccess d fuel)
    (stable : ReplayStableOnNodes d fuel)
    (rootsComplete : occurrenceRootsCompleteB d fuel = true) :
    successorFormationB d (generateSuccessorCertificate d fuel) = true :=
  generateSuccessorCertificateFromRoutes_complete d (replayRoutes d fuel)
    predecessorsListed
    (replayRoutes_fixedPoint d fuel predecessorsListed success stable)
    (replayRoutes_rootsComplete d fuel rootsComplete)

theorem generateSuccessorCertificate_complete_of_level
    (d : Graph) (fuel : Nat)
    (wellFounded : PredecessorLevelWellFounded d)
    (nodeFuel : ∀ vertex, vertex ∈ d.NODES -> vertex.LEVEL < fuel)
    (success : AllNodeReplaySuccess d fuel)
    (rootsComplete : occurrenceRootsCompleteB d fuel = true) :
    successorFormationB d (generateSuccessorCertificate d fuel) = true := by
  have predecessorsListed : PredecessorsListed d := by
    intro vertex vertexMember premise premiseMember
    exact (wellFounded vertex vertexMember premise premiseMember).1
  exact generateSuccessorCertificate_complete d fuel predecessorsListed
    success (replayStableOnNodes_of_level d fuel wellFounded nodeFuel)
    rootsComplete

end HCSuccessorCertificateGenerator
