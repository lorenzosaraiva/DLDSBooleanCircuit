import Semantic.HCSuccessorCertificateGenerator
import Semantic.NDTreeFrontendInvariant
















open Semantic FlowSpec
open OccurrenceAwareReplay
open HCSuccessorFixedGraphReference
open HCSuccessorBranchingGridBridge
open HCSuccessorCertificatePositions
open HCSuccessorCertificateSoundness HCSuccessorCertificateGenerator

namespace HCSuccessorActualFuel



def ActualPredecessorDomain (d : Graph) : Prop :=
  ∀ vertex, vertex ∈ d.NODES -> ∀ premise,
    premise ∈ predsOf d vertex ->
      premise ∈ d.NODES ∧ premise.LEVEL = vertex.LEVEL + 1

theorem ActualPredecessorDomain.predecessorsListed
    {d : Graph} (domain : ActualPredecessorDomain d) :
    PredecessorsListed d := by
  intro vertex vertexMember premise premiseMember
  exact (domain vertex vertexMember premise premiseMember).1




theorem actualPredecessorDomain_of_check_dlds_leveled
    (d : Graph) (hygiene : check_dlds d) (leveled : LeveledColored d) :
    ActualPredecessorDomain d := by
  intro vertex _ premise premiseMember
  obtain ⟨deduction, incomingMember, startEq⟩ :=
    predsOf_has_edge premiseMember
  have edgeMember : deduction ∈ d.EDGES :=
    mem_incoming_mem_edges vertex d incomingMember
  have endEq : deduction.END = vertex :=
    mem_incoming_end_eq vertex d incomingMember
  constructor
  · rw [← startEq]
    exact (hygiene.2.1 edgeMember).1
  · have edgeLevel := leveled deduction edgeMember
    simpa [startEq, endEq] using edgeLevel

theorem ndTreeToGraph_actualPredecessorDomain
    (tree : GridCertificateSoundness.NDTree) :
    ActualPredecessorDomain (HorizontalCompressionInput.ndTreeToGraph tree) :=
  actualPredecessorDomain_of_check_dlds_leveled _
    (HorizontalCompressionInput.ndTreeToGraph_check_dlds tree)
    (HorizontalCompressionInput.ndTreeToGraph_leveledColored tree)

theorem ndTreeToGraph_allNodeReplaySuccess
    (tree : GridCertificateSoundness.NDTree)
    (wellFormed : tree.wellFormedB = true) :
    AllNodeReplaySuccess (HorizontalCompressionInput.ndTreeToGraph tree)
      (stdFuel (HorizontalCompressionInput.ndTreeToGraph tree)) := by
  intro vertex vertexMember
  obtain ⟨subtree, level, next, vertexEq, formulaEq, sourceEq, replay⟩ :=
    HorizontalCompressionInput.ndTreeToGraph_allOccurrencesReplay tree
      wellFormed vertex vertexMember
  refine ⟨[(normalizeDep subtree.dependencies, [])], ?_⟩
  simpa [stdFuel] using replay




theorem occurrenceReplayAt_succ_stable_of_actualDomain
    (d : Graph) (domain : ActualPredecessorDomain d) :
    ∀ vertex, vertex ∈ d.NODES -> ∀ fuel,
      countAbove d vertex < fuel ->
        occurrenceReplayAt d (fuel + 1) vertex =
          occurrenceReplayAt d fuel vertex := by
  have key : ∀ measure, ∀ vertex,
      vertex ∈ d.NODES -> countAbove d vertex = measure -> ∀ fuel,
      countAbove d vertex < fuel ->
        occurrenceReplayAt d (fuel + 1) vertex =
          occurrenceReplayAt d fuel vertex := by
    intro measure
    induction measure using Nat.strong_induction_on with
    | h measure ih =>
        intro vertex vertexMember measureEq fuel fuelBound
        cases fuel with
        | zero => omega
        | succ predecessorFuel =>
            have premiseReplay : ∀ premise,
                premise ∈ predsOf d vertex ->
                occurrenceReplayAt d (predecessorFuel + 1) premise =
                  occurrenceReplayAt d predecessorFuel premise := by
              intro premise premiseMember
              obtain ⟨premiseNode, premiseLevel⟩ :=
                domain vertex vertexMember premise premiseMember
              have premiseMeasure :
                  countAbove d premise < countAbove d vertex :=
                countAbove_lt_of_level_succ d premiseNode premiseLevel
              exact ih (countAbove d premise) (by omega) premise premiseNode
                rfl predecessorFuel (by omega)
            have tablesEqual := actualPremises_congr
              (occurrenceReplayAt d (predecessorFuel + 1))
              (occurrenceReplayAt d predecessorFuel)
              (predsOf d vertex) premiseReplay
            rw [occurrenceReplayAt_succ, occurrenceReplayAt_succ, tablesEqual]
  intro vertex vertexMember fuel fuelBound
  exact key (countAbove d vertex) vertex vertexMember rfl fuel fuelBound



theorem replayStableOnNodes_stdFuel_of_actualDomain
    (d : Graph) (domain : ActualPredecessorDomain d) :
    ReplayStableOnNodes d (stdFuel d) := by
  intro vertex vertexMember
  apply occurrenceReplayAt_succ_stable_of_actualDomain d domain vertex
    vertexMember (stdFuel d)
  have bound := countAbove_lt_length d vertexMember
  unfold stdFuel
  omega

def ReplaySucceeds (d : Graph) (fuel : Nat) (vertex : Vertex) : Prop :=
  ∃ routes, occurrenceReplayAt d fuel vertex = .ok routes

theorem actualPremises_success_of_mem
    (replay : Vertex -> Except OccurrenceReplayError (List FlowPair)) :
    ∀ premises table,
      actualPremises replay premises = .ok table ->
      ∀ premise, premise ∈ premises ->
        ∃ routes, replay premise = .ok routes := by
  intro premises
  induction premises with
  | nil =>
      intro table tableEq premise premiseMember
      simp at premiseMember
  | cons head tail ih =>
      intro table tableEq premise premiseMember
      simp only [List.mem_cons] at premiseMember
      cases headReplay : replay head with
      | error reason => simp [actualPremises, headReplay] at tableEq
      | ok headRoutes =>
          cases tailReplay : actualPremises replay tail with
          | error reason =>
              simp [actualPremises, headReplay, tailReplay] at tableEq
          | ok tailTable =>
              rcases premiseMember with rfl | tailMember
              · exact ⟨headRoutes, headReplay⟩
              · exact ih tailTable tailReplay premise tailMember




theorem replaySucceeds_predecessor_stdFuel
    (d : Graph) (domain : ActualPredecessorDomain d)
    {vertex premise : Vertex}
    (vertexMember : vertex ∈ d.NODES)
    (premiseMember : premise ∈ predsOf d vertex)
    (success : ReplaySucceeds d (stdFuel d) vertex) :
    ReplaySucceeds d (stdFuel d) premise := by
  obtain ⟨premiseNode, premiseLevel⟩ :=
    domain vertex vertexMember premise premiseMember
  obtain ⟨vertexRoutes, vertexReplay⟩ := success
  have positiveFuel : 0 < stdFuel d := by
    unfold stdFuel
    omega
  obtain ⟨predecessorFuel, fuelEq⟩ := Nat.exists_eq_succ_of_ne_zero
    (Nat.ne_of_gt positiveFuel)
  have unfolded := occurrenceReplayAt_succ d predecessorFuel vertex
  have vertexReplay' :
      occurrenceReplayAt d (predecessorFuel + 1) vertex =
        .ok vertexRoutes := by
    rw [show predecessorFuel + 1 = predecessorFuel.succ by omega,
      ← fuelEq]
    exact vertexReplay
  rw [vertexReplay'] at unfolded
  cases premisesEq : actualPremises (occurrenceReplayAt d predecessorFuel)
      (predsOf d vertex) with
  | error reason => simp [premisesEq] at unfolded
  | ok table =>
      obtain ⟨premiseRoutes, premiseReplay⟩ :=
        actualPremises_success_of_mem (occurrenceReplayAt d predecessorFuel)
          (predsOf d vertex) table premisesEq premise premiseMember
      have premiseMeasure : countAbove d premise < countAbove d vertex :=
        countAbove_lt_of_level_succ d premiseNode premiseLevel
      have vertexBound := countAbove_lt_length d vertexMember
      have premiseBound : countAbove d premise < predecessorFuel := by
        unfold stdFuel at fuelEq
        omega
      have stable := occurrenceReplayAt_succ_stable_of_actualDomain d domain
        premise premiseNode predecessorFuel premiseBound
      refine ⟨premiseRoutes, ?_⟩
      rw [fuelEq, stable]
      exact premiseReplay



theorem preGo_replaySucceeds_stdFuel
    (d : Graph) (domain : ActualPredecessorDomain d) :
    ∀ searchFuel frontier accumulated,
      (∀ vertex, vertex ∈ frontier ->
        vertex ∈ d.NODES ∧ ReplaySucceeds d (stdFuel d) vertex) ->
      (∀ vertex, vertex ∈ accumulated ->
        ReplaySucceeds d (stdFuel d) vertex) ->
      ∀ vertex, vertex ∈ Pre.go d searchFuel frontier accumulated ->
        ReplaySucceeds d (stdFuel d) vertex := by
  intro searchFuel
  induction searchFuel with
  | zero =>
      intro frontier accumulated frontierSuccess accumulatedSuccess
        vertex vertexMember
      simpa [Pre.go] using accumulatedSuccess vertex vertexMember
  | succ searchFuel ih =>
      intro frontier accumulated frontierSuccess accumulatedSuccess
        vertex vertexMember
      let fresh := ((frontier.flatMap (predsOf d)).eraseDups).filter
        (fun candidate => !(accumulated.any (fun seen => seen = candidate)))
      have freshSuccess : ∀ candidate, candidate ∈ fresh ->
          candidate ∈ d.NODES ∧
            ReplaySucceeds d (stdFuel d) candidate := by
        intro candidate candidateMember
        have flatMember : candidate ∈ frontier.flatMap (predsOf d) :=
          List.mem_eraseDups.mp (List.mem_filter.mp candidateMember).1
        obtain ⟨parent, parentMember, predecessor⟩ :=
          List.mem_flatMap.mp flatMember
        obtain ⟨parentNode, parentSuccess⟩ :=
          frontierSuccess parent parentMember
        exact ⟨(domain parent parentNode candidate predecessor).1,
          replaySucceeds_predecessor_stdFuel d domain parentNode predecessor
            parentSuccess⟩
      have appendedSuccess : ∀ candidate,
          candidate ∈ accumulated ++ fresh ->
          ReplaySucceeds d (stdFuel d) candidate := by
        intro candidate candidateMember
        rcases List.mem_append.mp candidateMember with old | new
        · exact accumulatedSuccess candidate old
        · exact (freshSuccess candidate new).2
      simp only [Pre.go] at vertexMember
      rw [show
          ((frontier.flatMap (predsOf d)).eraseDups).filter
              (fun candidate =>
                !(accumulated.any (fun seen => seen = candidate))) = fresh
            by rfl] at vertexMember
      split at vertexMember
      · exact accumulatedSuccess vertex vertexMember
      · exact ih fresh (accumulated ++ fresh) freshSuccess appendedSuccess
          vertex vertexMember

theorem mem_Pre_replaySucceeds_stdFuel
    (d : Graph) (domain : ActualPredecessorDomain d)
    {root vertex : Vertex} (rootMember : root ∈ d.NODES)
    (rootSuccess : ReplaySucceeds d (stdFuel d) root)
    (vertexMember : vertex ∈ Pre d root) :
    ReplaySucceeds d (stdFuel d) vertex := by
  unfold Pre at vertexMember
  apply preGo_replaySucceeds_stdFuel d domain
    (d.NODES.length + 1) [root] []
  · intro candidate candidateMember
    simp only [List.mem_singleton] at candidateMember
    subst candidate
    exact ⟨rootMember, rootSuccess⟩
  · intro candidate candidateMember
    simp at candidateMember
  · exact vertexMember

theorem rootReplaySucceeds_of_complete
    (d : Graph) (fuel : Nat)
    {root : Vertex} (rootMember : root ∈ occurrenceRoots d)
    (complete : occurrenceRootsCompleteB d fuel = true) :
    ReplaySucceeds d fuel root := by
  unfold occurrenceRootsCompleteB at complete
  have parts : !(occurrenceRoots d).isEmpty = true ∧
      (occurrenceRoots d).all (fun root =>
        match occurrenceReplayAt d fuel root with
        | .error _ => false
        | .ok routes => !routes.isEmpty && routes.all routeClosedB) = true := by
    simpa using complete
  have rootCheck := List.all_eq_true.mp parts.2 root rootMember
  cases replayEq : occurrenceReplayAt d fuel root with
  | error reason => simp [replayEq] at rootCheck
  | ok routes => exact ⟨routes, replayEq⟩



theorem allNodeReplaySuccess_of_rootsComplete
    (d : Graph) (domain : ActualPredecessorDomain d)
    (root : Vertex) (rootAt : HorizontalCompressionInput.RootAt d root)
    (reaches : HorizontalCompressionInput.EveryNodeReachesRoot d root)
    (complete : occurrenceRootsCompleteB d (stdFuel d) = true) :
    AllNodeReplaySuccess d (stdFuel d) := by
  have rootInOccurrenceRoots : root ∈ occurrenceRoots d := by
    simp [occurrenceRoots, rootAt.1, rootAt.2]
  have rootSuccess := rootReplaySucceeds_of_complete d (stdFuel d)
    rootInOccurrenceRoots complete
  intro vertex vertexMember
  rcases reaches vertex vertexMember with rfl | predecessor
  · exact rootSuccess
  · exact mem_Pre_replaySucceeds_stdFuel d domain rootAt.1 rootSuccess
      predecessor




theorem generateSuccessorCertificate_complete_of_actualDomain
    (d : Graph) (domain : ActualPredecessorDomain d)
    (success : AllNodeReplaySuccess d (stdFuel d))
    (rootsComplete : occurrenceRootsCompleteB d (stdFuel d) = true) :
    successorFormationB d
      (generateSuccessorCertificate d (stdFuel d)) = true := by
  exact generateSuccessorCertificate_complete d (stdFuel d)
    domain.predecessorsListed success
    (replayStableOnNodes_stdFuel_of_actualDomain d domain) rootsComplete



theorem generateSuccessorCertificate_complete_of_connected_actualDomain
    (d : Graph) (domain : ActualPredecessorDomain d)
    (root : Vertex) (rootAt : HorizontalCompressionInput.RootAt d root)
    (reaches : HorizontalCompressionInput.EveryNodeReachesRoot d root)
    (rootsComplete : occurrenceRootsCompleteB d (stdFuel d) = true) :
    successorFormationB d
      (generateSuccessorCertificate d (stdFuel d)) = true := by
  apply generateSuccessorCertificate_complete_of_actualDomain d domain
  · exact allNodeReplaySuccess_of_rootsComplete d domain root rootAt reaches
      rootsComplete
  · exact rootsComplete




theorem connectedActualDomain_successorPipeline
    (d : Graph) (domain : ActualPredecessorDomain d)
    (root : Vertex) (rootAt : HorizontalCompressionInput.RootAt d root)
    (reaches : HorizontalCompressionInput.EveryNodeReachesRoot d root)
    (rootsComplete : occurrenceRootsCompleteB d (stdFuel d) = true) :
    successorFormationB d
        (generateSuccessorCertificate d (stdFuel d)) = true ∧
      ReferenceCompiledGridValid d (stdFuel d) ∧
      ∀ position,
        position ∈ semanticPositions d
          (generateSuccessorCertificate d (stdFuel d)) ->
        semanticMarkedAtB d
          (generateSuccessorCertificate d (stdFuel d)) position = false := by
  have formation :=
    generateSuccessorCertificate_complete_of_connected_actualDomain d domain
      root rootAt reaches rootsComplete
  refine ⟨formation,
    (occurrenceRootsCompleteB_iff_compiledGridValid d (stdFuel d)).mp
      rootsComplete, ?_⟩
  exact (no_semantic_marked_iff_formation d
    (generateSuccessorCertificate d (stdFuel d))).mpr formation




theorem ndTreeToGraph_generatedSuccessorCertificateValid
    (tree : GridCertificateSoundness.NDTree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    successorFormationB (HorizontalCompressionInput.ndTreeToGraph tree)
      (generateSuccessorCertificate
        (HorizontalCompressionInput.ndTreeToGraph tree)
        (stdFuel (HorizontalCompressionInput.ndTreeToGraph tree))) = true := by
  apply generateSuccessorCertificate_complete_of_actualDomain
  · exact ndTreeToGraph_actualPredecessorDomain tree
  · exact ndTreeToGraph_allNodeReplaySuccess tree wellFormed
  · simpa [stdFuel] using
      HorizontalCompressionInput.ndTreeToGraph_occurrenceRootsComplete tree
        wellFormed closed

theorem ndTreeToGraph_noSemanticMarked
    (tree : GridCertificateSoundness.NDTree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    ∀ position,
      position ∈ semanticPositions
        (HorizontalCompressionInput.ndTreeToGraph tree)
        (generateSuccessorCertificate
          (HorizontalCompressionInput.ndTreeToGraph tree)
          (stdFuel (HorizontalCompressionInput.ndTreeToGraph tree))) ->
      semanticMarkedAtB
        (HorizontalCompressionInput.ndTreeToGraph tree)
        (generateSuccessorCertificate
          (HorizontalCompressionInput.ndTreeToGraph tree)
          (stdFuel (HorizontalCompressionInput.ndTreeToGraph tree)))
        position = false :=
  (no_semantic_marked_iff_formation _ _).mpr
    (ndTreeToGraph_generatedSuccessorCertificateValid tree wellFormed closed)

end HCSuccessorActualFuel
