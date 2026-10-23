import Infrastructure.NDTreeOccurrenceReplay
import Compression.HCCompressionSequence











open Semantic FlowSpec
open GridCertificateSoundness
open OccurrenceAwareReplay

namespace HorizontalCompressionInput

@[simp] theorem pre_collapse_center (neighborhood : Neighborhood) :
    (pre_collapse neighborhood).CENTER = neighborhood.CENTER := by
  unfold pre_collapse
  split <;> rfl

@[simp] theorem get_rule_center (vertex : Vertex) (graph : Graph) :
    (get_rule vertex graph).CENTER = vertex := by
  rfl

def actualHCInitialPartition (tree : NDTree) : List HCClass :=
  hcPartition (ndTreeToGraph tree)

def actualHCInitialRules (tree : NDTree) : List Neighborhood :=
  pre_compress
    (get_rules (ndTreeToGraph tree).NODES (ndTreeToGraph tree))

def actualHCInitialOccurrences (_tree : NDTree) : List Ancestral := []




structure InitialSemanticObservation where
  occurrence : Vertex
  formula : Formula
  replay : Except OccurrenceReplayError (List FlowPair)
deriving Repr

def observeInitialNeighborhood (graph : Graph) (fuel : Nat)
    (neighborhood : Neighborhood) : List InitialSemanticObservation :=
  [{ occurrence := neighborhood.CENTER
     formula := neighborhood.CENTER.FORMULA
     replay := occurrenceReplayAt graph fuel neighborhood.CENTER }]

def actualHCInitialObservations (tree : NDTree) :
    List InitialSemanticObservation :=
  (actualHCInitialRules tree).flatMap
    (observeInitialNeighborhood (ndTreeToGraph tree)
      ((ndTreeToGraph tree).NODES.length + 1))

theorem precompressed_getRules_centers
    (nodes : List Vertex) (graph : Graph) :
    (pre_compress (get_rules nodes graph)).map Neighborhood.CENTER = nodes := by
  induction nodes with
  | nil => rfl
  | cons vertex rest ih =>
      simp only [get_rules, pre_compress, List.map_cons,
        pre_collapse_center, get_rule_center, ih]

theorem actualHCInitialRules_centers (tree : NDTree) :
    (actualHCInitialRules tree).map Neighborhood.CENTER =
      (ndTreeToGraph tree).NODES := by
  exact precompressed_getRules_centers _ _

theorem precompressed_getRules_observations
    (nodes : List Vertex) (graph : Graph) (fuel : Nat) :
    (pre_compress (get_rules nodes graph)).flatMap
        (observeInitialNeighborhood graph fuel) =
      nodes.map fun vertex =>
        { occurrence := vertex
          formula := vertex.FORMULA
          replay := occurrenceReplayAt graph fuel vertex } := by
  induction nodes with
  | nil => rfl
  | cons vertex rest ih =>
      simp only [get_rules, pre_compress, List.flatMap_cons,
        observeInitialNeighborhood, pre_collapse_center, get_rule_center,
        List.singleton_append, List.map_cons, ih]

theorem actualHCInitialObservations_eq_node_map (tree : NDTree) :
    actualHCInitialObservations tree =
      (ndTreeToGraph tree).NODES.map fun vertex =>
        { occurrence := vertex
          formula := vertex.FORMULA
          replay := occurrenceReplayAt (ndTreeToGraph tree)
            ((ndTreeToGraph tree).NODES.length + 1) vertex } := by
  exact precompressed_getRules_observations _ _ _

theorem actualHCInitialObservations_replay
    (tree : NDTree) (wellFormed : tree.wellFormedB = true) :
    ∀ observation ∈ actualHCInitialObservations tree,
      InitialOccurrenceReplayAt (ndTreeToGraph tree)
        ((ndTreeToGraph tree).NODES.length + 1) observation.occurrence ∧
      observation.formula = observation.occurrence.FORMULA ∧
      observation.replay = occurrenceReplayAt (ndTreeToGraph tree)
        ((ndTreeToGraph tree).NODES.length + 1) observation.occurrence := by
  intro observation member
  rw [actualHCInitialObservations_eq_node_map] at member
  obtain ⟨vertex, vertexMember, observationEq⟩ := List.mem_map.mp member
  subst observation
  exact ⟨ndTreeToGraph_allOccurrencesReplay tree wellFormed vertex vertexMember,
    rfl, rfl⟩

def actualHCInitialOccurrenceMap (tree : NDTree) : List (Nat × Vertex) :=
  (ndTreeToGraph tree).NODES.map fun vertex => (vertex.NUMBER, vertex)

theorem actualHCInitialOccurrenceMap_numbers (tree : NDTree) :
    (actualHCInitialOccurrenceMap tree).map Prod.fst =
      List.range' 1 tree.nodeCount := by
  unfold actualHCInitialOccurrenceMap
  rw [List.map_map]
  change (encodeNDTreeAt tree 0 1).nodes.map Vertex.NUMBER = _
  exact encodeNDTreeAt_node_numbers tree 0 1



structure VerifiedNDTreeHCInitialState (tree : NDTree) : Prop where
  properInput : ProperHCInput (ndTreeToGraph tree)
  partitionIsProduction : actualHCInitialPartition tree =
    hcPartition (ndTreeToGraph tree)
  ruleCentersAreAllOccurrences :
    (actualHCInitialRules tree).map Neighborhood.CENTER =
      (ndTreeToGraph tree).NODES
  occurrenceAccumulatorEmpty : actualHCInitialOccurrences tree = []
  occurrenceMapNumbers :
    (actualHCInitialOccurrenceMap tree).map Prod.fst =
      List.range' 1 tree.nodeCount
  semanticObservationCoverage :
    ∀ observation ∈ actualHCInitialObservations tree,
      InitialOccurrenceReplayAt (ndTreeToGraph tree)
        ((ndTreeToGraph tree).NODES.length + 1) observation.occurrence ∧
      observation.formula = observation.occurrence.FORMULA ∧
      observation.replay = occurrenceReplayAt (ndTreeToGraph tree)
        ((ndTreeToGraph tree).NODES.length + 1) observation.occurrence
  rootsComplete : occurrenceRootsCompleteB (ndTreeToGraph tree)
    ((ndTreeToGraph tree).NODES.length + 1) = true

theorem ndTreeToGraph_verifiedHCInitialState
    (tree : NDTree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    VerifiedNDTreeHCInitialState tree := by
  refine
    { properInput := ndTreeToGraph_properHCInput tree wellFormed closed
      partitionIsProduction := rfl
      ruleCentersAreAllOccurrences := actualHCInitialRules_centers tree
      occurrenceAccumulatorEmpty := rfl
      occurrenceMapNumbers := actualHCInitialOccurrenceMap_numbers tree
      semanticObservationCoverage :=
        actualHCInitialObservations_replay tree wellFormed
      rootsComplete :=
        ndTreeToGraph_occurrenceRootsComplete tree wellFormed closed }

end HorizontalCompressionInput
