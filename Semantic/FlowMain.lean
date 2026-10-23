import Semantic.FlowRobustness
import Semantic.FlowProofSearch
import Semantic.MultiIntroMain

open Semantic FlowSpec GridCertificateSoundness Semantic.BranchingGrid
open FlowBranchingGridBridge RouteCertificate CompleteProfileRobustness
open SearchProblem.CertificateRobustness

namespace Flow


theorem main (d : Graph) :
    (_root_.Flow d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ MultiIntroGrid.acceptedB .general d = true) ∧
    (_root_.Flow d ↔ ∃ c, MultiIntro.StrongCertificate .general d c) ∧
    (_root_.Flow d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ MultiIntroReadings.verdict .general d) ∧
    (_root_.Flow d →
      ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
        Derives [] root.FORMULA) ∧
    (_root_.Flow d ↔ ∃ c, MultiIntro.strongCertificateB .general d c = true) ∧
    (SingleIntroductionReference.ReferenceVerdict d → _root_.Flow d) ∧
    (MultiIntro.NoMultiIntro d → (_root_.Flow d ↔ SingleIntroductionReference.ReferenceVerdict d)) :=
  ⟨(MultiIntroMain.main .general d).1,
    (MultiIntroMain.main .general d).2.1,
    (MultiIntroMain.main .general d).2.2.1,
    (MultiIntroMain.main .general d).2.2.2,
    MultiIntroMain.checked_certificate_view .general d,
    of_reference d, iff_reference_of_no_multi d⟩


theorem extras (d : Graph) (c : CompleteProfileCertificate) :
    (((_root_.Flow d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ MultiIntroGrid.acceptedB .general d = true) ∧
    (_root_.Flow d ↔ ∃ certificate, MultiIntro.StrongCertificate .general d certificate) ∧
    (_root_.Flow d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ MultiIntroReadings.verdict .general d) ∧
    (_root_.Flow d →
      ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
        Derives [] root.FORMULA)) ∧
    (MultiIntro.StrongCertificate .general d c ↔
      ∀ position ∈ MultiIntroPositions.positions d,
        ¬ MultiIntroPositions.FailedAt .general d c position) ∧
    (_root_.Flow d →
      ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
        ∃ tree : NDTree, tree.wellFormedB = true ∧
          tree.conclusion = root.FORMULA ∧ tree.dependencies = [] ∧
          occurrenceMarkedB tree.formulas root.FORMULA tree.nodeCount
            (tree.formulas.length + tree.nodeCount)
            (encodeCertifiedNDTree tree.formulas tree (treeFormulaCoverage tree)) = true) ∧
    (¬ MultiIntro.StrongCertificate .general d c →
      ∃ position ∈ MultiIntroPositions.positions d,
        Robustness.FailedAt d c position ∧
        Robustness.readBound c position ≤ (MultiIntroPositions.readVertices c position).length ∧
        (1 : ℚ) / Fintype.card (RowAlphabet c) ^ Robustness.readBound c position ≤
          Robustness.failureDensity d c position)) ∧
    (∀ slot v, Robustness.readBound c (.productionRow slot v) ≤
      1 + (predsOf c.graph v).length) ∧
    ((MultiIntroPositions.positions d).length = d.NODES.length +
      (d.NODES.filter fun v => (get_rule.outgoing v d).isEmpty).length + 3) ∧
    (¬ MultiIntro.StrongCertificate .general d c →
      ∃ position ∈ MultiIntroPositions.positions d, Robustness.FailedAt d c position) ∧
    (∀ position, Robustness.readBound c position ≤
      (MultiIntroPositions.readVertices c position).length) ∧
    (∀ slot v, Robustness.readBound c (.rootDischarge slot v) ≤ 1) ∧
    (∀ position, Robustness.check d (decodeRows c (rowAnchor c)) position =
      Robustness.check d c position) ∧
    (∀ position (left right : RowAssignment c),
      AgreeOn (Robustness.readSet c position) left right →
      Robustness.check d (decodeRows c left) position =
        Robustness.check d (decodeRows c right) position) ∧
    (∀ position, Robustness.FailedAt d c position →
      ∀ other : RowAssignment c,
        AgreeOn (Robustness.readSet c position) (rowAnchor c) other →
        Robustness.FailedAt d (decodeRows c other) position) ∧
    (∀ position, Robustness.FailedAt d c position →
      (1 : ℚ) / Fintype.card (RowAlphabet c) ^ Robustness.readBound c position ≤
        Robustness.failureDensity d c position) ∧
    (∀ position (failed : Robustness.FailedAt d c position),
      Fintype.card (FailureCylinder (Robustness.positionFailureWitness d c position failed)) =
        Fintype.card (RowAlphabet c) ^
          (Fintype.card (RowCoordinate c) - Robustness.readBound c position)) ∧
    (¬ MultiIntro.StrongCertificate .general d c →
      ∃ position ∈ MultiIntroPositions.positions d,
        ∃ witness : FailureWitness (Robustness.failureOracle d c position) (rowAnchor c),
          witness.support = Robustness.readSet c position ∧
          Fintype.card (FailureCylinder witness) =
            Fintype.card (RowAlphabet c) ^
              (Fintype.card (RowCoordinate c) - Robustness.readBound c position) ∧
          (1 : ℚ) / Fintype.card (RowAlphabet c) ^ Robustness.readBound c position ≤
            Robustness.failureDensity d c position) ∧
    (∀ slot v, Robustness.FailedAt d c (.rootDischarge slot v) ↔
      Robustness.PositiveRootDefect c v ∨ Robustness.AbsentRootDefect c v) ∧
    (∀ slot v, Robustness.FailedAt d c (.productionRow slot v) ↔
      Robustness.ExtraProductionRoute c v ∨ Robustness.MissingProductionRoute c v ∨
        Robustness.OrderedProductionDefect c v) ∧
    (∀ position, MultiIntroPositions.readVertices c position = [] →
      Robustness.FailedAt d c position → (1 : ℚ) ≤ Robustness.failureDensity d c position) ∧
    (∀ slot v, Robustness.PositiveRootDefect c v ∨ Robustness.AbsentRootDefect c v →
      ∀ other : RowAssignment c,
        AgreeOn (Robustness.readSet c (.rootDischarge slot v)) (rowAnchor c) other →
        Robustness.FailedAt d (decodeRows c other) (.rootDischarge slot v)) ∧
    (∀ (fs : List Formula) (goal : Formula) (height K : Nat)
      (assignment : OccurrenceAssignment fs.length height K),
      occurrenceMarkedB fs goal height K assignment = true → Derives [] goal) ∧
    (∀ (fs : List Formula) (goal : Formula) (height K : Nat)
      (assignment : OccurrenceAssignment fs.length height K),
      occurrenceMarkedB fs goal height K assignment = true →
        ∃ tree, extractOccurrenceGoalCone fs goal height K assignment = some tree ∧
          tree.wellFormedB = true ∧ tree.conclusion = goal ∧
          tree.dependencies = [] ∧ tree.height ≤ height) ∧
    (∀ (d₁ d₂ : Graph), buildFormulas d₁ = buildFormulas d₂ →
      SpaceA.levelBound d₁ = SpaceA.levelBound d₂ → ∀ goal : Formula,
        ¬ ∃ paths : PathInput,
          SpaceA.Marked (buildFormulas d₁) goal (SpaceA.levelBound d₁) paths ≠
            SpaceA.Marked (buildFormulas d₂) goal (SpaceA.levelBound d₂) paths) ∧
    Function.Injective (decodeRows c) ∧
    (∀ route : FlowPair,
      [route, route] ≠ [route] ∧ SelfCertifying.ListSetEq [route, route] [route] ∧
        ¬ (∃ value ∈ [route, route], value ∉ [route]) ∧
        ¬ (∃ value ∈ [route], value ∉ [route, route])) :=
  ⟨⟨MultiIntroMain.main .general d,
      MultiIntroPositions.formation_iff_no_failed_position .general d c,
      terminal_gridWitness d,
      Robustness.formation_invalid_semantic_amplification d c⟩,
    Robustness.production_read_bound c,
    MultiIntroPositions.position_count d,
    MultiIntroPositions.invalid_has_failed_position .general d c,
    Robustness.read_bound_le c, Robustness.root_read_bound c,
    Robustness.check_anchor d c, Robustness.check_agreement d c,
    Robustness.failure_persists d c, Robustness.failed_position_density d c,
    Robustness.failed_position_cylinder_card d c,
    Robustness.formation_invalid_has_badCylinder d c,
    Robustness.root_failure_positive_or_absence d c,
    Robustness.production_failure_cases d c,
    Robustness.metadata_failure_density_one d c,
    Robustness.root_defect_failure_persists d c,
    marked_sound, marked_extracts,
    fun _ _ => separation,
    Robustness.decodeRows_injective c, Robustness.duplicate_profile_boundary⟩

end Flow
