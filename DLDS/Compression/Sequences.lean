import DLDS.Compression.Mixed.ContextualElimination
import DLDS.Compression.Mixed.ContextualIntroduction
import DLDS.Compression.Mixed.SourceDerived

/-!
# Composition of local HC preservation along the actual compressor

This module is deliberately parametric in a boundary observation.  It does not
pretend that every collapse performed by horizontal compression has already
been proved correct.  Instead, `ActualCompressionTrace` records exactly the
remaining proof obligation at each real call to `collapseWithOccurrences` and
then proves that those obligations compose along the real
`compressWithOccurrences` recursion.

In particular, the occurrence accumulator in the trace is not an idealized
one: it is rewritten and extended by exactly the expression used by the
production compressor.
-/

namespace DLDS
namespace Compression.Sequences

open DLDS.Compression.Mixed.EliminationSource DLDS.Compression.Mixed.IntroductionSource
open DLDS.Compression.Mixed.ContextualElimination

/--
A proof trace for the production `compressWithOccurrences` loop.

`observe` is a proof-only view of the boundary behaviour of one current
neighbourhood.  A step may be constructed only after proving that the actual
production collapse preserves that observation.  The tail starts from the
actual merged neighbourhood and the actual rewritten occurrence accumulator.
-/
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

/-- A trace computes exactly the result of the production recursive loop. -/
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

/--
Per-collapse boundary preservation composes transitively through the actual
left-associated compressor sequence.
-/
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

/-!
`ActualCompressionTrace` records
literal list equality.  Actual route enumeration order is not semantic, so
the successor induction uses this parallel set-preserving trace.  It executes
the same production recursion and occurrence rewrite; only the proof-facing
observation relation is weakened from list equality to membership equality.
-/

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
        Certificate.Flow.ListSetEq
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
    Certificate.Flow.ListSetEq (rules.flatMap observe) (observe final) := by
  induction trace with
  | empty dummyInvisible =>
      intro observation
      simp [dummyInvisible]
  | single only =>
      simpa using Certificate.Flow.ListSetEq.refl (observe only)
  | step left right rest preservesBoundary tail ih =>
      apply Certificate.Flow.ListSetEq.trans (ys :=
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

/-- A trace beginning at the producer's actual `pre_compress/get_rules` list
refines `compress_ruleWithOccurrences`, not merely the inner helper. -/
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

/--
Exact executable refinement through the graph-replacement operation.  This
theorem is intentionally an equality, not a semantic preservation claim: the
semantic work remains in the trace's per-step boundary obligations.
-/
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

/--
The data required to interpret the mixed-HE local theorem as one step of an
arbitrary proof-only boundary observer.
-/
structure MixedHEStageObserver
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.derived d)) ++
        observe (pre_collapse (get_rule input.source d)) =
      DLDS.Compression.Mixed.ContextualElimination.preBoundaryDeliveries input
  after : observe (DLDS.Compression.Mixed.EliminationSource.producedNeighborhood input) =
    DLDS.Compression.Mixed.ContextualElimination.postBoundaryDeliveries input

/--
The already-proved mixed-HE operation supplies a genuine step of the actual
compressor trace for every observer satisfying the explicit stage adapter.
-/
theorem actualMixedHE_two_rule_trace
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : MixedHEStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.derived d),
       pre_collapse (get_rule input.source d)]
      [] (DLDS.Compression.Mixed.EliminationSource.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.derived d))
          (pre_collapse (get_rule input.source d)) =
        (DLDS.Compression.Mixed.EliminationSource.producedNeighborhood input, []) := by
    simpa [DLDS.Compression.Mixed.EliminationSource.actualOrderedMixedOperation] using
      DLDS.Compression.Mixed.EliminationSource.actualOrderedMixedOperation_eq_produced input
  apply DLDS.Compression.Sequences.ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((DLDS.Compression.Mixed.ContextualElimination.actualMixedHE_boundary_preserved input).symm.trans
        adapter.after.symm)
  · rw [operation]
    simpa using
      (DLDS.Compression.Sequences.ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (DLDS.Compression.Mixed.EliminationSource.producedNeighborhood input))

/-- Mixed-HI analogue of `MixedHEStageObserver`. -/
structure MixedHIStageObserver
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.derived d)) ++
        observe (pre_collapse (get_rule input.source d)) =
      DLDS.Compression.Mixed.ContextualIntroduction.preBoundaryDeliveries input
  after : observe (DLDS.Compression.Mixed.IntroductionSource.producedNeighborhood input) =
    DLDS.Compression.Mixed.ContextualIntroduction.postBoundaryDeliveries input

/--
The actual mixed-HI operation is likewise a real production-compressor step.
-/
theorem actualMixedHI_two_rule_trace
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : MixedHIStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.derived d),
       pre_collapse (get_rule input.source d)]
      [] (DLDS.Compression.Mixed.IntroductionSource.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.derived d))
          (pre_collapse (get_rule input.source d)) =
        (DLDS.Compression.Mixed.IntroductionSource.producedNeighborhood input, []) := by
    simpa [DLDS.Compression.Mixed.IntroductionSource.actualOrderedMixedOperation] using
      DLDS.Compression.Mixed.IntroductionSource.actualOrderedMixedOperation_eq_produced input
  apply DLDS.Compression.Sequences.ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((DLDS.Compression.Mixed.ContextualIntroduction.actualMixedHI_boundary_preserved input).symm.trans
        adapter.after.symm)
  · rw [operation]
    simpa using
      (DLDS.Compression.Sequences.ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (DLDS.Compression.Mixed.IntroductionSource.producedNeighborhood input))

structure ReverseMixedHEStageObserver
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.source d)) ++
        observe (pre_collapse (get_rule input.derived d)) =
      DLDS.Compression.Mixed.ContextualElimination.preBoundaryDeliveries input
  after :
    observe
        (DLDS.Compression.Mixed.SourceDerived.HE.producedNeighborhood input) =
      DLDS.Compression.Mixed.SourceDerived.Contextual.hePostBoundary input

theorem actualReverseMixedHE_two_rule_trace
    {d : Graph}
    (input : ActualMixedHEInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : ReverseMixedHEStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.source d),
       pre_collapse (get_rule input.derived d)]
      [] (DLDS.Compression.Mixed.SourceDerived.HE.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.source d))
          (pre_collapse (get_rule input.derived d)) =
        (DLDS.Compression.Mixed.SourceDerived.HE.producedNeighborhood input, []) := by
    simpa [DLDS.Compression.Mixed.SourceDerived.HE.actualReverseMixedOperation] using
      DLDS.Compression.Mixed.SourceDerived.HE.actualReverseMixedOperation_eq_produced
        input
  apply DLDS.Compression.Sequences.ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((DLDS.Compression.Mixed.SourceDerived.Contextual.reverseMixedHE_boundary_preserved
        input).symm.trans adapter.after.symm)
  · rw [operation]
    simpa using
      (DLDS.Compression.Sequences.ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (DLDS.Compression.Mixed.SourceDerived.HE.producedNeighborhood input))

structure ReverseMixedHIStageObserver
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation) : Prop where
  before :
    observe (pre_collapse (get_rule input.source d)) ++
        observe (pre_collapse (get_rule input.derived d)) =
      DLDS.Compression.Mixed.ContextualIntroduction.preBoundaryDeliveries input
  after :
    observe
        (DLDS.Compression.Mixed.SourceDerived.HI.producedNeighborhood input) =
      DLDS.Compression.Mixed.SourceDerived.Contextual.hiPostBoundary input

theorem actualReverseMixedHI_two_rule_trace
    {d : Graph}
    (input : ActualMixedHIInput d)
    (observe : Neighborhood -> List BoundaryObservation)
    (adapter : ReverseMixedHIStageObserver input observe) :
    ActualCompressionTrace (hcPartition d) observe
      [pre_collapse (get_rule input.source d),
       pre_collapse (get_rule input.derived d)]
      [] (DLDS.Compression.Mixed.SourceDerived.HI.producedNeighborhood input) [] := by
  have operation :
      collapseWithOccurrences (hcPartition d)
          (pre_collapse (get_rule input.source d))
          (pre_collapse (get_rule input.derived d)) =
        (DLDS.Compression.Mixed.SourceDerived.HI.producedNeighborhood input, []) := by
    simpa [DLDS.Compression.Mixed.SourceDerived.HI.actualReverseMixedOperation] using
      DLDS.Compression.Mixed.SourceDerived.HI.actualReverseMixedOperation_eq_produced
        input
  apply DLDS.Compression.Sequences.ActualCompressionTrace.step
  · rw [operation]
    exact adapter.before.trans
      ((DLDS.Compression.Mixed.SourceDerived.Contextual.reverseMixedHI_boundary_preserved
        input).symm.trans adapter.after.symm)
  · rw [operation]
    simpa using
      (DLDS.Compression.Sequences.ActualCompressionTrace.single
        (partition := hcPartition d)
        (observe := observe)
        (occurrences := ([] : List Ancestral))
        (DLDS.Compression.Mixed.SourceDerived.HI.producedNeighborhood input))

end Compression.Sequences
end DLDS
