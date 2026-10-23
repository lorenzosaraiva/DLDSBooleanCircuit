import Infrastructure.RouteCorrespondence
import Infrastructure.SpineNDSoundness

open Semantic FlowSpec SelfCertifying GridCertificateSoundness
namespace SingleIntroductionReference

theorem rootsB_roots_derivable (d : Graph) (accepted : rootsB d = true) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA := by
  intro root member terminal
  obtain ⟨nonempty, closed⟩ := (rootsB_iff d).mp accepted root member terminal
  obtain ⟨route, produced⟩ := List.exists_mem_of_ne_nil _ nonempty
  exact SpineND.guardFreeFlowAt_closed_derivable d (stdFuel d) root route
    (routes_subset_guardFree d (stdFuel d) root produced) (closed route produced).1

theorem ReferenceVerdict.roots_derivable (d : Graph) (valid : ReferenceVerdict d) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA := rootsB_roots_derivable d valid.2.2

theorem routes_eq_guardFree_of_labelSound (d : Graph) (structural : structuralValid d)
    (labels : Spine.LabelSound (compressorOverlay d)) : ∀ fuel v,
    routes d fuel v = LambdaFlow.guardFreeFlowAt d fuel v :=
  Spine.guards_redundant_of_labelSound_eq (compressorOverlay d) structural labels

theorem rootsB_eq_guardFree_of_labelSound (d : Graph) (structural : structuralValid d)
    (labels : Spine.LabelSound (compressorOverlay d)) :
    rootsB d = LambdaFlow.guardFreeRootsB d := by
  rw [LambdaFlow.guardFreeRootsB_eq_raw]
  unfold rootsB
  simp only [routes_eq_guardFree_of_labelSound d structural labels]

theorem ReferenceVerdict_iff_RefValid_of_labelSound (d : Graph)
    (labels : Spine.LabelSound (compressorOverlay d)) :
    ReferenceVerdict d ↔ CompleteLabelReference.RefValid (compressorOverlay d) := by
  constructor
  · intro valid
    refine ⟨valid.1, valid.2.1, labels, ?_⟩
    change LambdaFlow.guardFreeRootsB d = true
    rw [← rootsB_eq_guardFree_of_labelSound d valid.1 labels]
    exact valid.2.2
  · intro valid
    refine ⟨valid.1, valid.2.1, ?_⟩
    rw [rootsB_eq_guardFree_of_labelSound d valid.1 labels]
    exact valid.2.2.2

theorem RefValid_implies_ReferenceVerdict (d : Graph)
    (valid : CompleteLabelReference.RefValid (compressorOverlay d)) : ReferenceVerdict d :=
  (ReferenceVerdict_iff_RefValid_of_labelSound d valid.2.2.1).mpr valid

theorem historical_setEq_guardFree_of_labelSound (d : Graph)
    (structural : structuralValid d) (labels : Spine.LabelSound (compressorOverlay d))
    (fuel : Nat) (v : Vertex) :
    ListSetEq (flowAt d fuel v) (LambdaFlow.guardFreeFlowAt d fuel v) := by
  intro route
  constructor
  · intro member
    exact LambdaFlow.historical_subset_guardFree d fuel v member
  · intro member
    have tagged : route ∈ routes d fuel v := by
      rwa [routes_eq_guardFree_of_labelSound d structural labels]
    exact routes_subset_historical d fuel v tagged

#print axioms rootsB_roots_derivable
#print axioms ReferenceVerdict.roots_derivable
#print axioms ReferenceVerdict_iff_RefValid_of_labelSound
#print axioms historical_setEq_guardFree_of_labelSound
end SingleIntroductionReference
