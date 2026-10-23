import Semantic.HermannContextAdequacy
import Semantic.HCSuccessorActualCertificateSoundness





open Semantic FlowSpec GridCertificateSoundness OccurrenceAwareReplay
open HCSuccessorFixedGraphReference HCSuccessorBranchingGridBridge
open HCSuccessorCertificatePositions HCSuccessorActualFuel HCSuccessorCertificateGenerator

namespace HermannCertificateToGraph

theorem truth_tail_of_chain (value : String → Prop) (context : List Formula)
    (tail : Formula) (truth : HermannContext.Truth value (HermannContext.chain context tail))
    (assumptions : ∀ a, a ∈ context → HermannContext.Truth value a) :
    HermannContext.Truth value tail := by
  induction context with
  | nil => exact truth
  | cons a rest ih =>
      apply ih (truth (assumptions a (by simp)))
      intro f member
      exact assumptions f (by simp [member])

theorem closed_curried_derivation_rules_out_cycle
    (sourceGraph : HermannContext.Graph) (names : HermannContext.Names)
    (context : List Formula)
    (admitted : HermannContext.ContextAdmissible sourceGraph names context)
    (derivation : Derives [] (HermannContext.chain context (HermannContext.q names))) :
    ¬HermannContext.HasHamiltonianCycle sourceGraph := by
  rintro ⟨cycle⟩
  have truth := HermannContext.derives_truth (HermannContext.valuation names cycle)
    derivation (by simp)
  have falseQ := truth_tail_of_chain _ context _ truth
    (fun a member => HermannContext.admissible_true names cycle (admitted a member))
  exact (HermannContext.valuation_encode names cycle .result).mp falseQ



theorem certificate_root_ndTree
    (proofGraph : _root_.Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain proofGraph)
    (formation : successorFormationB proofGraph certificate = true)
    (root : Vertex) (rootMember : root ∈ occurrenceRoots proofGraph) :
    ∃ tree : NDTree, tree.wellFormedB = true ∧
      tree.conclusion = root.FORMULA ∧ tree.dependencies = [] := by
  have executable := HCSuccessorActualCertificateSoundness.successorFormationB_occurrenceRootsComplete_stdFuel
    proofGraph certificate domain formation
  have reference := (occurrenceRootsCompleteB_iff_reference proofGraph (stdFuel proofGraph)).mp executable
  have complete := reference.2 root rootMember
  cases replay : referenceReplayAt proofGraph (stdFuel proofGraph) root with
  | error reason => simp [ReferenceResultComplete, replay] at complete
  | success routes =>
      have parts : (∃ route, routes route) ∧
          ∀ route, routes route → routeClosedB route = true := by
        simpa [ReferenceResultComplete, replay] using complete
      obtain ⟨route, member⟩ := parts.1
      have trees := referenceReplayAt_tree_correct proofGraph (stdFuel proofGraph) root
      rw [replay] at trees
      obtain ⟨tree, wellFormed, conclusion, dependency⟩ := trees route member
      have closed : route.1 = [] ∧ route.2 = [] := by
        simpa [routeClosedB] using parts.2 route member
      have normalized : normalizeDep tree.dependencies = [] := dependency.trans closed.1
      exact ⟨tree, wellFormed, conclusion,
        (FlowBranchingGridBridge.normalizeDep_eq_nil_iff tree.dependencies).mp normalized⟩

theorem certificate_root_derivable
    (proofGraph : _root_.Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain proofGraph)
    (formation : successorFormationB proofGraph certificate = true)
    (root : Vertex) (rootMember : root ∈ occurrenceRoots proofGraph) :
    Derives [] root.FORMULA := by
  obtain ⟨tree, wellFormed, conclusion, closed⟩ :=
    certificate_root_ndTree proofGraph certificate domain formation root rootMember
  simpa [conclusion, closed] using tree.derives_dependencies wellFormed

theorem certificate_rules_out_hamiltonian_cycle
    (sourceGraph : HermannContext.Graph) (names : HermannContext.Names)
    (context : List Formula)
    (admitted : HermannContext.ContextAdmissible sourceGraph names context)
    (proofGraph : _root_.Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain proofGraph)
    (formation : successorFormationB proofGraph certificate = true)
    (root : Vertex) (rootMember : root ∈ occurrenceRoots proofGraph)
    (target : root.FORMULA = HermannContext.chain context (HermannContext.q names)) :
    ¬HermannContext.HasHamiltonianCycle sourceGraph := by
  apply closed_curried_derivation_rules_out_cycle sourceGraph names context admitted
  simpa [target] using certificate_root_derivable proofGraph certificate domain formation root rootMember

end HermannCertificateToGraph
