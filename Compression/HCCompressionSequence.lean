import Compression.HCContextualMixedHE
import Compression.HCContextualMixedHI
import Compression.ActualMixedReverseProducerBridge
















namespace Semantic
namespace HCCompressionSequence

open ActualMixedProducerBridge ActualMixedHIProducerBridge
open HCContextualMixedHE









inductive ActualCompressionTrace
    (partition : List HCClass)
    (observe : Neighborhood -> List BoundaryObservation) :
    (rules : List Neighborhood) ->
    (occurrences : List Ancestral) ->
    (final : Neighborhood) ->
    (finalOccurrences : List Ancestral) -> Prop where
  | empty
      (dummyInvisible :
        observe (rule (node 0 0 #"" false false []) [] [] [] []) = []) :
      ActualCompressionTrace partition observe [] occurrences
        (rule (node 0 0 #"" false false []) [] [] [] []) occurrences
  | single (only : Neighborhood) :
      ActualCompressionTrace partition observe [only] occurrences only occurrences
  | step
      (left right : Neighborhood)
      (rest : List Neighborhood)
      (preservesBoundary :
        observe left ++ observe right =
          observe (collapseWithOccurrences partition left right).1)
      (tail :
        ActualCompressionTrace partition observe
          ((collapseWithOccurrences partition left right).1 :: rest)
          ((occurrences.map
              (collapse.rewrite_path_endpoints
                [left.CENTER, right.CENTER]
                (collapseWithOccurrences partition left right).1.CENTER)) ++
            (collapseWithOccurrences partition left right).2)
          final finalOccurrences) :
      ActualCompressionTrace partition observe
        (left :: right :: rest) occurrences final finalOccurrences


theorem ActualCompressionTrace.refines_actual_compressor
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {rules : List Neighborhood}
    {occurrences : List Ancestral}
    {final : Neighborhood}
    {finalOccurrences : List Ancestral}
    (trace : ActualCompressionTrace partition observe
      rules occurrences final finalOccurrences) :
    compressWithOccurrences partition rules occurrences =
      (final, finalOccurrences) := by
  induction trace with
  | empty dummyInvisible => simp [compressWithOccurrences]
  | single only => simp [compressWithOccurrences]
  | step left right rest preservesBoundary tail ih =>
      simpa only [compressWithOccurrences] using ih





theorem ActualCompressionTrace.boundary_preserved
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {rules : List Neighborhood}
    {occurrences : List Ancestral}
    {final : Neighborhood}
    {finalOccurrences : List Ancestral}
    (trace : ActualCompressionTrace partition observe
      rules occurrences final finalOccurrences) :
    rules.flatMap observe = observe final := by
  induction trace with
  | empty dummyInvisible => simpa using dummyInvisible.symm
  | single only => simp
  | step left right rest preservesBoundary tail ih =>
      rw [List.flatMap_cons, List.flatMap_cons, ← List.append_assoc,
        preservesBoundary]
      simpa only [List.flatMap_cons] using ih









inductive ActualCompressionSetTrace
    (partition : List HCClass)
    (observe : Neighborhood -> List BoundaryObservation) :
    (rules : List Neighborhood) ->
    (occurrences : List Ancestral) ->
    (final : Neighborhood) ->
    (finalOccurrences : List Ancestral) -> Prop where
  | empty
      (dummyInvisible :
        observe (rule (node 0 0 #"" false false []) [] [] [] []) = []) :
      ActualCompressionSetTrace partition observe [] occurrences
        (rule (node 0 0 #"" false false []) [] [] [] []) occurrences
  | single (only : Neighborhood) :
      ActualCompressionSetTrace partition observe [only] occurrences
        only occurrences
  | step
      (left right : Neighborhood)
      (rest : List Neighborhood)
      (preservesBoundary :
        SelfCertifying.ListSetEq
          (observe left ++ observe right)
          (observe (collapseWithOccurrences partition left right).1))
      (tail :
        ActualCompressionSetTrace partition observe
          ((collapseWithOccurrences partition left right).1 :: rest)
          ((occurrences.map
              (collapse.rewrite_path_endpoints
                [left.CENTER, right.CENTER]
                (collapseWithOccurrences partition left right).1.CENTER)) ++
            (collapseWithOccurrences partition left right).2)
          final finalOccurrences) :
      ActualCompressionSetTrace partition observe
        (left :: right :: rest) occurrences final finalOccurrences

theorem ActualCompressionSetTrace.refines_actual_compressor
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {rules : List Neighborhood}
    {occurrences : List Ancestral}
    {final : Neighborhood}
    {finalOccurrences : List Ancestral}
    (trace : ActualCompressionSetTrace partition observe
      rules occurrences final finalOccurrences) :
    compressWithOccurrences partition rules occurrences =
      (final, finalOccurrences) := by
  induction trace with
  | empty dummyInvisible => simp [compressWithOccurrences]
  | single only => simp [compressWithOccurrences]
  | step left right rest preservesBoundary tail ih =>
      simpa only [compressWithOccurrences] using ih

theorem ActualCompressionSetTrace.boundary_preserved
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {rules : List Neighborhood}
    {occurrences : List Ancestral}
    {final : Neighborhood}
    {finalOccurrences : List Ancestral}
    (trace : ActualCompressionSetTrace partition observe
      rules occurrences final finalOccurrences) :
    SelfCertifying.ListSetEq (rules.flatMap observe) (observe final) := by
  induction trace with
  | empty dummyInvisible =>
      intro observation
      simp [dummyInvisible]
  | single only =>
      simpa using SelfCertifying.ListSetEq.refl (observe only)
  | step left right rest preservesBoundary tail ih =>
      apply SelfCertifying.ListSetEq.trans (ys :=
        observe (collapseWithOccurrences partition left right).1 ++
          rest.flatMap observe)
      · intro observation
        simpa only [List.flatMap_cons, List.mem_append, or_assoc] using
          or_congr (preservesBoundary observation) Iff.rfl
      · simpa only [List.flatMap_cons] using ih

theorem ActualCompressionSetTrace.refines_compress_ruleWithOccurrences
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {nodes : List Vertex}
    {d : Graph}
    {final : Neighborhood}
    {finalOccurrences : List Ancestral}
    (trace : ActualCompressionSetTrace partition observe
      (pre_compress (get_rules nodes d)) [] final finalOccurrences) :
    compress_ruleWithOccurrences partition nodes d =
      (final, finalOccurrences) := by
  unfold compress_ruleWithOccurrences
  exact trace.refines_actual_compressor

theorem ActualCompressionSetTrace.refines_compress_nodesWithPartitionCarrying
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {nodes : List Vertex}
    {d : Graph}
    {occurrences finalOccurrences : List Ancestral}
    {final : Neighborhood}
    (trace : ActualCompressionSetTrace partition observe
      (pre_compress (get_rules nodes d)) [] final finalOccurrences) :
    compress_nodesWithPartitionCarrying partition nodes d occurrences =
      let materialized := materializeCarriedOwners nodes d occurrences
      let rewritten := materialized.map
        (collapse.rewrite_path_endpoints nodes final.CENTER)
      let allOccurrences := rewritten ++ finalOccurrences
      let graph := collapse.rewrite_graph_path_endpoints nodes final.CENTER
        (set_rule (pop_rules d (get_rules nodes d)) final)
      let graph := dlds graph.NODES graph.EDGES
        (ensureOccurrenceRecords graph.PATHS allOccurrences)
      (checkAncestralPathHeads graph, allOccurrences) := by
  unfold compress_nodesWithPartitionCarrying
  rw [trace.refines_compress_ruleWithOccurrences]



theorem ActualCompressionTrace.refines_compress_ruleWithOccurrences
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {nodes : List Vertex}
    {d : Graph}
    {final : Neighborhood}
    {finalOccurrences : List Ancestral}
    (trace : ActualCompressionTrace partition observe
      (pre_compress (get_rules nodes d)) [] final finalOccurrences) :
    compress_ruleWithOccurrences partition nodes d =
      (final, finalOccurrences) := by
  unfold compress_ruleWithOccurrences
  exact trace.refines_actual_compressor






theorem ActualCompressionTrace.refines_compress_nodesWithPartitionCarrying
    {partition : List HCClass}
    {observe : Neighborhood -> List BoundaryObservation}
    {nodes : List Vertex}
    {d : Graph}
    {occurrences finalOccurrences : List Ancestral}
    {final : Neighborhood}
    (trace : ActualCompressionTrace partition observe
      (pre_compress (get_rules nodes d)) [] final finalOccurrences) :
    compress_nodesWithPartitionCarrying partition nodes d occurrences =
      let materialized := materializeCarriedOwners nodes d occurrences
      let rewritten := materialized.map
        (collapse.rewrite_path_endpoints nodes final.CENTER)
      let allOccurrences := rewritten ++ finalOccurrences
      let graph := collapse.rewrite_graph_path_endpoints nodes final.CENTER
        (set_rule (pop_rules d (get_rules nodes d)) final)
      let graph := dlds graph.NODES graph.EDGES
        (ensureOccurrenceRecords graph.PATHS allOccurrences)
      (checkAncestralPathHeads graph, allOccurrences) := by
  unfold compress_nodesWithPartitionCarrying
  rw [trace.refines_compress_ruleWithOccurrences]





structure MixedHEStageObserver
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.derived d)) ++
        observe (pre_collapse (get_rule input.source d)) =
      HCContextualMixedHE.preBoundaryDeliveries input
  after : observe (ActualMixedProducerBridge.producedNeighborhood input) =
    HCContextualMixedHE.postBoundaryDeliveries input





theorem actualMixedHE_two_rule_trace
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : MixedHEStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.derived d),
       pre_collapse (get_rule input.source d)]
      [] (ActualMixedProducerBridge.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.derived d))
          (pre_collapse (get_rule input.source d)) =
        (ActualMixedProducerBridge.producedNeighborhood input, []) := by
    simpa [ActualMixedProducerBridge.actualOrderedMixedOperation] using
      ActualMixedProducerBridge.actualOrderedMixedOperation_eq_produced input
  apply ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((HCContextualMixedHE.actualMixedHE_boundary_preserved input).symm.trans
        adapter.after.symm)
  · rw [operation]
    simpa using
      (ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (ActualMixedProducerBridge.producedNeighborhood input))


structure MixedHIStageObserver
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.derived d)) ++
        observe (pre_collapse (get_rule input.source d)) =
      HCContextualMixedHI.preBoundaryDeliveries input
  after : observe (ActualMixedHIProducerBridge.producedNeighborhood input) =
    HCContextualMixedHI.postBoundaryDeliveries input




theorem actualMixedHI_two_rule_trace
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : MixedHIStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.derived d),
       pre_collapse (get_rule input.source d)]
      [] (ActualMixedHIProducerBridge.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.derived d))
          (pre_collapse (get_rule input.source d)) =
        (ActualMixedHIProducerBridge.producedNeighborhood input, []) := by
    simpa [ActualMixedHIProducerBridge.actualOrderedMixedOperation] using
      ActualMixedHIProducerBridge.actualOrderedMixedOperation_eq_produced input
  apply ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((HCContextualMixedHI.actualMixedHI_boundary_preserved input).symm.trans
        adapter.after.symm)
  · rw [operation]
    simpa using
      (ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (ActualMixedHIProducerBridge.producedNeighborhood input))

structure ReverseMixedHEStageObserver
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.source d)) ++
        observe (pre_collapse (get_rule input.derived d)) =
      HCContextualMixedHE.preBoundaryDeliveries input
  after :
    observe
        (ActualMixedReverseProducerBridge.HE.producedNeighborhood input) =
      ActualMixedReverseProducerBridge.Contextual.hePostBoundary input

theorem actualReverseMixedHE_two_rule_trace
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : ReverseMixedHEStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.source d),
       pre_collapse (get_rule input.derived d)]
      [] (ActualMixedReverseProducerBridge.HE.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.source d))
          (pre_collapse (get_rule input.derived d)) =
        (ActualMixedReverseProducerBridge.HE.producedNeighborhood input, []) := by
    simpa [ActualMixedReverseProducerBridge.HE.actualReverseMixedOperation] using
      ActualMixedReverseProducerBridge.HE.actualReverseMixedOperation_eq_produced
        input
  apply ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((ActualMixedReverseProducerBridge.Contextual.reverseMixedHE_boundary_preserved
        input).symm.trans adapter.after.symm)
  · rw [operation]
    simpa using
      (ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (ActualMixedReverseProducerBridge.HE.producedNeighborhood input))

structure ReverseMixedHIStageObserver
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.source d)) ++
        observe (pre_collapse (get_rule input.derived d)) =
      HCContextualMixedHI.preBoundaryDeliveries input
  after :
    observe
        (ActualMixedReverseProducerBridge.HI.producedNeighborhood input) =
      ActualMixedReverseProducerBridge.Contextual.hiPostBoundary input

theorem actualReverseMixedHI_two_rule_trace
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : ReverseMixedHIStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.source d),
       pre_collapse (get_rule input.derived d)]
      [] (ActualMixedReverseProducerBridge.HI.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.source d))
          (pre_collapse (get_rule input.derived d)) =
        (ActualMixedReverseProducerBridge.HI.producedNeighborhood input, []) := by
    simpa [ActualMixedReverseProducerBridge.HI.actualReverseMixedOperation] using
      ActualMixedReverseProducerBridge.HI.actualReverseMixedOperation_eq_produced
        input
  apply ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((ActualMixedReverseProducerBridge.Contextual.reverseMixedHI_boundary_preserved
        input).symm.trans adapter.after.symm)
  · rw [operation]
    simpa using
      (ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (ActualMixedReverseProducerBridge.HI.producedNeighborhood input))

end HCCompressionSequence
end Semantic
