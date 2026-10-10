import Certificate.Encoding.Canonical
import Certificate.Encoding.Export
import Robustness.Certificate.Support
import Examples.FourVertex.Occurrence.Certificate
import Examples.FourVertex.Occurrence.NonHamiltonicity
import Robustness.Certificate.FailureCylinders
import DLDS.Compression.Algorithm
import Search.Certificate.Definitions
import DLDS.All
import DLDS.Compression.Mixed.IntroductionSource
import DLDS.Compression.Mixed.EliminationSource
import DLDS.Compression.Mixed.SourceDerived
import DLDS.Compression.SourcePair
import Evaluator.Boolean
import ProofSearch.BranchingGrid.Definitions
import ProofSearch.BranchingGrid.Completeness
import ProofSearch.BranchingGrid.Soundness
import Certificate.Flow.Cache
import Evaluator.Mixed.Recursive
import Certificate.Flow.GeneratorCompleteness
import Examples.FourVertex.Flow.CertificateDecision
import Evaluator.CertificateRoutes
import Flow.Compressed.DependencyBridge
import Evaluator.Compressed.Routing
import Evaluator.Definitions
import Evaluator.Grid
import Examples.Flow.ClosedImplication
import Evaluator.GridEvaluation
import Examples.Circuit
import Readings.Explicit.Composition
import Readings.Explicit.CoveredComposition
import Readings.Explicit.FanFamily
import Readings.Explicit.PremiseInteraction
import Readings.Explicit.LabelValidity
import ProofSearch.FlowCorrespondence
import Flow.UniversalDischarge
import Flow.Compressed.Elimination
import Flow.DependencyLabels
import Flow.Definitions
import Flow.TreeSoundness
import Flow.Validity
import ProofSearch.Grid.Soundness
import DLDS.Compression.Sequences
import DLDS.Compression.Mixed.ContextualElimination
import DLDS.Compression.Mixed.ContextualIntroduction
import DLDS.Compression.Elimination.PairPreservation
import DLDS.Compression.EdgeTraces
import DLDS.Compression.Elimination.StepCorrespondence
import DLDS.Compression.Elimination.Execution
import DLDS.Compression.Elimination.ClassPreservation
import DLDS.Compression.IndexedClasses
import DLDS.Compression.ClassIteration
import Certificate.Occurrence.ProductionSoundness
import Evaluator.Occurrence.Fuel
import ProofSearch.OccurrenceCorrespondence
import Certificate.Occurrence.Generator
import Certificate.Occurrence.Positions
import Certificate.Occurrence.Soundness
import Evaluator.Occurrence.Relation
import Evaluator.Occurrence.LocalRules
import Certificate.Occurrence.Definitions
import NonHamiltonicity.Certificate.Input
import NonHamiltonicity.Certificate.Soundness
import NonHamiltonicity.Context.Soundness
import NonHamiltonicity.Context.Encoding
import DLDS.Compression.FixedPoints
import DLDS.Compression.Mixed.LocalRules
import Evaluator.MultiToken.Admissibility
import Evaluator.MultiToken.Discharge
import Evaluator.MultiToken.Definitions
import Evaluator.MultiToken.CanonicalReduction
import DLDS.Compression.TreeInvariant
import DLDS.Compression.TreeInitialState
import Evaluator.Occurrence.TreeCorrespondence
import DLDS.Compression.TreeReachability
import Evaluator.CellCorrectness
import Evaluator.Occurrence.Execution
import Examples.ThreeVertex.Flow.Closed
import Examples.ThreeVertex.Flow.Acceptance
import Examples.ThreeVertex.Flow.Certificate
import Examples.ThreeVertex.Flow.Open
import Examples.FourVertex.Flow.Closed
import Examples.FourVertex.Flow.Acceptance
import Examples.FourVertex.Flow.Certificate
import Examples.FourVertex.Flow.Slots
import Examples.FourVertex.Flow.Open
import Certificate.Flow.IntroductionSuffix
import ProofSearch.Grid.Definitions
import DLDS.Compression.Domain
import Readings.Selection
import Certificate.Flow.RootDischarge
import Evaluator.RouteIndexed.Tabulation
import Evaluator.RouteIndexed.FlowCorrespondence
import Evaluator.RouteIndexed.Definitions
import Evaluator.TokenRouting
import Readings.FailurePersistence
import Certificate.Flow.Soundness
import Certificate.Flow.Definitions
import Evaluator.TreeCorrespondence
import Evaluator.UniversalAcceptance
import DLDS.ValidityDecomposition
import Evaluator.VectorLemmas
import Certificate.Flow.Positions
import Certificate.Encoding.FixedShape
import ProofSearch.Grid.Completeness
import ProofSearch.Grid.Schedulability
import ProofSearch.Grid.Counting
import Examples.FourVertex.Flow.CachedDischarge
import Examples.FourVertex.Flow.EndToEnd
import Robustness.Certificate.PositionClasses
import Robustness.AbsenceBoundary

set_option pp.fullNames true

#print axioms Certificate.Flow.GeneratorCompleteness.flowCompiledGridValid_iff_exists_strongCertificate
#print axioms Certificate.Flow.GeneratorCompleteness.generateCertificate_expanded
#print axioms Certificate.Flow.GeneratorCompleteness.generateCertificate_localChecks
#print axioms Certificate.Flow.GeneratorCompleteness.generateCertificate_strongFormation
#print axioms Certificate.Flow.GeneratorCompleteness.generateCertificate_tailClosure
#print axioms Certificate.Flow.GeneratorCompleteness.generatedCertificate_roundTrip
#print axioms Certificate.Flow.GeneratorCompleteness.generatedLocalExpected_setEq_flow
#print axioms Certificate.Flow.GeneratorCompleteness.rootsDischargedB_iff_exists_strongCertificate
#print axioms Evaluator.CertificateRoutes.certificateSkeletonAt_setEq_flowAt
#print axioms Evaluator.CertificateRoutes.certificateSkeletonCanonicalWitnessable_setEq
#print axioms Evaluator.CertificateRoutes.certificateSkeletonCanonical_setEq_flowAt
#print axioms Evaluator.CertificateRoutes.certificateSkeletonDischarged_of_strongFormation
#print axioms Evaluator.CertificateRoutes.certificateSkeletonLocalOutputs_setEq_flowStepFrom
#print axioms Evaluator.CertificateRoutes.certificateSkeletonSource_project
#print axioms Evaluator.CertificateRoutes.skeletonResidualsAt_setEq_witnessableFlow
#print axioms Evaluator.CertificateRoutes.skeletonResidualsForEdge_setEq_witnessableFlow
#print axioms Examples.FourVertex.Occurrence.NonHamiltonicity.supplied_graph_no_hamiltonian_cycle
#print axioms ProofSearch.FlowCorrespondence.completedFlowOccurrence_iff_grid
#print axioms ProofSearch.FlowCorrespondence.encodeCertifiedNDTree_marked_iff_closed
#print axioms ProofSearch.FlowCorrespondence.flowCompiledGridValid_of_strongFormation
#print axioms ProofSearch.FlowCorrespondence.rootsDischargedB_iff_flowCompiledGridValid
#print axioms ProofSearch.FlowCorrespondence.routeOccurrence_tree_dependencies
#print axioms Flow.flowRuleCorrect_collapsed
#print axioms Flow.flowRuleCorrect_of_simpleTree
#print axioms NonHamiltonicity.Certificate.Input.checked_request_sound
#print axioms DLDS.Compression.Domain.GenuineMixedSource.proper_compression_can_fail_representation
#print axioms DLDS.Compression.Domain.GenuineMixedSource.proper_inputs_can_have_mixed_source_classes
#print axioms DLDS.Compression.Domain.GenuineMixedSource.sourceSeparated_output_not_hcRepresentationWellFormed
#print axioms DLDS.Compression.Domain.GenuineMixedSource.sourceSeparated_output_not_levelFormulaUnique
#print axioms DLDS.Compression.Domain.compress_nodes_graph_sourceSeparated_eq_self_of_no_candidates
#print axioms DLDS.Compression.Domain.hcSourceClassKeyEqB_hypothesis_eq
#print axioms DLDS.Compression.Domain.source_status_split_has_at_most_two_classes
#print axioms DLDS.Compression.Domain.validDLDSB_sound_for_hc
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_end_to_end
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_flowCompiledGridValid
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_generatedStrongFormation
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_hcRepresentationWellFormed
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_levelFormulaUnique
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_rootsDischargedB
#print axioms DLDS.Compression.FixedPoints.compress_fixpoint_suitableGraph
#print axioms DLDS.Compression.FixedPoints.compress_nodes_graph_eq_self_of_no_candidates
#print axioms DLDS.Compression.FixedPoints.source_edge_has_ancestral_head_of_cursorRouteHeadCoverage
#print axioms List.Vector.ext
#print axioms List.getElem_map
#print axioms List.getElem_zipWith
#print axioms List.mem_append
#print axioms List.mem_eraseDups
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonCanonicalFast_eq_reference
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonCanonicalFast_setEq_flowAt
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonExactCanonicalAt_eq_reference
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonFastAt_setEq_reference
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonFastDischarged_of_strongFormation
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonTableAt_eq_fast
#print axioms Evaluator.RouteIndexed.Tabulation.certificateSkeletonTopologicalFastDischarged_of_strongFormation
#print axioms Evaluator.RouteIndexed.Tabulation.routeIndexedCanonicalFast_eq_canonical
#print axioms Evaluator.RouteIndexed.Tabulation.routeIndexedCanonicalFast_setEq_flowAt
#print axioms Evaluator.RouteIndexed.Tabulation.routeIndexedCanonicalTopologicalFast_eq_canonical
#print axioms Evaluator.RouteIndexed.Tabulation.routeIndexedCanonicalTopologicalFast_setEq_flowAt
#print axioms Evaluator.RouteIndexed.certificateLocalOutputs_setEq_flowStepFrom
#print axioms Evaluator.RouteIndexed.routeIndexedAt_setEq_flowAt
#print axioms Evaluator.RouteIndexed.routeIndexedCanonicalDischarged_of_rootsDischargedB
#print axioms Evaluator.RouteIndexed.routeIndexedCanonicalDischarged_of_strongFormation
#print axioms Evaluator.RouteIndexed.routeIndexedCanonicalWitnessable_setEq
#print axioms Evaluator.RouteIndexed.routeIndexedCanonical_setEq_flowAt
#print axioms Evaluator.RouteIndexed.routeIndexedLocalOutputs_setEq_flowStepFrom
#print axioms Evaluator.RouteIndexed.routeIndexedWitnessableLocal_setEq
#print axioms Search.Certificate.Robustness.AbstractRouteAbsence.absence_forces_all_slots
#print axioms Search.Certificate.Robustness.AbstractRouteAbsence.emptyRouteList_fails
#print axioms Search.Certificate.Robustness.AbstractRouteAbsence.missingRoute_minimumFailureWitnessSize
#print axioms Search.Certificate.Robustness.CanonicalGoalFlowCounterexample.canonical_goal_bits_not_supported
#print axioms Search.Certificate.Robustness.CanonicalGoalFlowCounterexample.canonical_not_discharged
#print axioms Search.Certificate.Robustness.CanonicalGoalFlowCounterexample.roots_discharged
#print axioms Search.Certificate.Robustness.canonicalAlphabet_card_le_fieldBudget_add_one
#print axioms Search.Certificate.Robustness.canonicalCoordinate_card_le_fieldBudget
#print axioms Search.Certificate.Robustness.canonicalGraphReadSet_card_le_graphFieldBudget
#print axioms Search.Certificate.Robustness.canonicalGraphSlots_length
#print axioms Search.Certificate.Robustness.canonicalMarkedAt_density_lower_bound
#print axioms Search.Certificate.Robustness.canonicalMaxTotalSupport_le_fieldBudget
#print axioms Search.Certificate.Robustness.canonicalSemanticMarked_bound_density_lower_bound
#print axioms Search.Certificate.Robustness.canonicalSemanticMarked_density_lower_bound
#print axioms Search.Certificate.Robustness.canonicalSemanticOrbit_card
#print axioms Search.Certificate.Robustness.canonicalSemanticReadSet_agreement
#print axioms Search.Certificate.Robustness.canonicalSemanticReadSet_card_le
#print axioms Search.Certificate.Robustness.canonicalSlots_length
#print axioms Search.Certificate.Robustness.canonicalSupportOfPosition_agreement
#print axioms Search.Certificate.Robustness.canonicalSupportOfPosition_card_le_fieldBudget
#print axioms Search.Certificate.Robustness.canonicalSupportOfPosition_denominator_le_fieldBudget
#print axioms Search.Certificate.Robustness.canonicalSupportOfSemanticPosition_card_le
#print axioms Search.Certificate.Robustness.canonicalSupportOfSemanticPosition_subset_readSet
#print axioms Search.Certificate.Robustness.canonicalSupportOfTotalPosition_agreement
#print axioms Search.Certificate.Robustness.canonicalSupportOfTotalPosition_card_le_fieldBudget
#print axioms Search.Certificate.Robustness.canonicalSupportOfTotalPosition_card_le_positionReadBound
#print axioms Search.Certificate.Robustness.canonicalSupportOfTotalPosition_denominator_le_fieldBudget
#print axioms Search.Certificate.Robustness.canonicalSupportOfTotalPosition_subset_syntactic
#print axioms Search.Certificate.Robustness.canonicalSyntacticReadSet_agreement
#print axioms Search.Certificate.Robustness.canonicalSyntacticReadSet_card_le_positionReadBound
#print axioms Search.Certificate.Robustness.canonicalTerminalReadSet_card_le_terminalFieldBudget
#print axioms Search.Certificate.Robustness.canonicalTerminalSlots_length
#print axioms Search.Certificate.Robustness.canonicalTotalMalformationAssignments_lower_bound
#print axioms Search.Certificate.Robustness.canonicalTotalMarkedAt_density_lower_bound
#print axioms Search.Certificate.Robustness.canonicalTotalMarkedAt_fieldBudget_density_lower_bound
#print axioms Search.Certificate.Robustness.canonicalTotalMarkedAt_positionReadBound_density_lower_bound
#print axioms Search.Certificate.Robustness.canonicalTotalOrbit_card
#print axioms Search.Certificate.Robustness.canonicalWitnessReadSet_card_le_witnessFieldBudget
#print axioms Search.Certificate.Robustness.canonicalWitnessSlots_length
#print axioms Search.Certificate.Robustness.canonical_formation_invalid_amplification
#print axioms Search.Certificate.Robustness.certificateFieldBudget_le_multiplicityExposedFieldBound
#print axioms Search.Certificate.Robustness.compressed_universally_accepted_from_strong_formation
#print axioms Search.Certificate.Robustness.decodeCertificate_edgeRoutes_congr
#print axioms Search.Certificate.Robustness.decodeCertificate_encodeCertificate
#print axioms Search.Certificate.Robustness.decodeCertificate_graph_congr
#print axioms Search.Certificate.Robustness.decodeCertificate_shapeFixedToCanonical
#print axioms Search.Certificate.Robustness.decodeCertificate_terminalOutputs_congr
#print axioms Search.Certificate.Robustness.decodeShapeFixed_edgeRoutes_length
#print axioms Search.Certificate.Robustness.decodeShapeFixed_encodeShapeFixed
#print axioms Search.Certificate.Robustness.decodeShapeFixed_graph
#print axioms Search.Certificate.Robustness.decodeShapeFixed_terminalOutputs_length
#print axioms Search.Certificate.Robustness.decodeShapeFixed_terminalOwners
#print axioms Search.Certificate.Robustness.decodeShapeFixed_witnessOwners
#print axioms Search.Certificate.Robustness.decodeTerminal_shapeFixed_dependency_length
#print axioms Search.Certificate.Robustness.decodeTerminal_shapeFixed_residual_length
#print axioms Search.Certificate.Robustness.decodeWitness_shapeFixed_dependency_length
#print axioms Search.Certificate.Robustness.decodeWitness_shapeFixed_residual_length
#print axioms Search.Certificate.Robustness.dependentFailureAssignments_lower_bound
#print axioms Search.Certificate.Robustness.dependentFailureCylinderToAssignments_injective
#print axioms Search.Certificate.Robustness.dependentFailureCylinder_card
#print axioms Search.Certificate.Robustness.dependentFailureDensity_lower_bound
#print axioms Search.Certificate.Robustness.dischargedMultiB_of_rootsDischargedB
#print axioms Search.Certificate.Robustness.dischargedMultiB_of_strongLocalRootDischarge
#print axioms Search.Certificate.Robustness.edgeWitnessCoverage_readSet_card_le_multiplicity
#print axioms Search.Certificate.Robustness.failureAssignments_lower_bound
#print axioms Search.Certificate.Robustness.failureCylinderToAssignments_injective
#print axioms Search.Certificate.Robustness.failureCylinder_card
#print axioms Search.Certificate.Robustness.failureDensity_lower_bound
#print axioms Search.Certificate.Robustness.forcesFailure_univ
#print axioms Search.Certificate.Robustness.formation_invalid_amplification
#print axioms Search.Certificate.Robustness.formation_invalid_automatic_amplification
#print axioms Search.Certificate.Robustness.formation_invalid_has_badCylinder
#print axioms Search.Certificate.Robustness.formation_invalid_implies_localMalformation
#print axioms Search.Certificate.Robustness.formation_invalid_semantic_amplification
#print axioms Search.Certificate.Robustness.formation_invalid_semantic_multiplicity_amplification
#print axioms Search.Certificate.Robustness.formation_invalid_shapeFixed_amplification
#print axioms Search.Certificate.Robustness.graphRule_failure_density
#print axioms Search.Certificate.Robustness.graphRule_is_ruleLevel
#print axioms Search.Certificate.Robustness.graphRule_readBound
#print axioms Search.Certificate.Robustness.localChecks_root_has_empty_flowRoute
#print axioms Search.Certificate.Robustness.localChecks_root_has_empty_witnessableRoute
#print axioms Search.Certificate.Robustness.localFailure_unread_invariant
#print axioms Search.Certificate.Robustness.localPayloadCoordinateCount_le_slotBound
#print axioms Search.Certificate.Robustness.localPayloadRecordBound_le_multiplicity_mul_width
#print axioms Search.Certificate.Robustness.localPayloadSlotBound_eq_recordBound
#print axioms Search.Certificate.Robustness.localPositions_append_terminals
#print axioms Search.Certificate.Robustness.malformationAssignments_lower_bound
#print axioms Search.Certificate.Robustness.malformationDensity_lower_bound
#print axioms Search.Certificate.Robustness.malformationOrbit_card
#print axioms Search.Certificate.Robustness.maxLocalSupport_le_coordinate_card
#print axioms Search.Certificate.Robustness.maxSemanticSupportBound_le_multiplicityExposed
#print axioms Search.Certificate.Robustness.minimumFailureSupport_spec
#print axioms Search.Certificate.Robustness.minimumFailureWitness_minimal
#print axioms Search.Certificate.Robustness.minimumShapeFixedFailureWitness_size_le
#print axioms Search.Certificate.Robustness.nodeProduction_readSet_card_le_multiplicity
#print axioms Search.Certificate.Robustness.nodeProduction_shapeFixedSupport_card_le_recordBound
#print axioms Search.Certificate.Robustness.position_classes_disjoint
#print axioms Search.Certificate.Robustness.position_classes_exhaustive
#print axioms Search.Certificate.Robustness.registered_witnessFaithful_readBound_unbounded
#print axioms Search.Certificate.Robustness.remoteTerminals_outputsAt_unchanged
#print axioms Search.Certificate.Robustness.rootDischargeAtB_false_of_terminal_shape
#print axioms Search.Certificate.Robustness.rootDischarge_readSet_card_le_multiplicity
#print axioms Search.Certificate.Robustness.rootDischarge_shapeFixedSupport_card_le_recordBound
#print axioms Search.Certificate.Robustness.rootTerminalShapeFailureWitness_size
#print axioms Search.Certificate.Robustness.routeCoverage_readSet_card_le_multiplicity
#print axioms Search.Certificate.Robustness.routeCoverage_shapeFixedSupport_card_le_recordBound
#print axioms Search.Certificate.Robustness.semanticMarked_has_badCylinder
#print axioms Search.Certificate.Robustness.semanticMarked_has_minimumFailureWitness
#print axioms Search.Certificate.Robustness.shapeFixedFailureCylinder_card
#print axioms Search.Certificate.Robustness.shapeFixedFailureWitness_density_lower_bound
#print axioms Search.Certificate.Robustness.shapeFixedSemanticMarked_bound_density_lower_bound
#print axioms Search.Certificate.Robustness.shapeFixedSemanticMarked_density_lower_bound
#print axioms Search.Certificate.Robustness.shapeFixedSemanticOrbit_card
#print axioms Search.Certificate.Robustness.shapeFixedSemanticReadBound_le_canonical
#print axioms Search.Certificate.Robustness.shapeFixedSemanticReadSet_agreement
#print axioms Search.Certificate.Robustness.shapeFixedStructural_support_eq_empty
#print axioms Search.Certificate.Robustness.shapeFixedSupportOfSemanticPosition_subset_readSet
#print axioms Search.Certificate.Robustness.shapeFixedToCanonical_injective
#print axioms Search.Certificate.Robustness.shapeFixedToCanonical_preserves_structural_coordinates
#print axioms Search.Certificate.Robustness.singleCoordinateFailureWitness_size
#print axioms Search.Certificate.Robustness.sourceProductionAtB_false_of_residual_length
#print axioms Search.Certificate.Robustness.sourceResidualLengthFailureWitness_size
#print axioms Search.Certificate.Robustness.strongLocalRootDischarge_rootsDischargedB
#print axioms Search.Certificate.Robustness.structuralShapeFixedFailureWitness_size
#print axioms Search.Certificate.Robustness.structuralShapeFixedFailure_density_one
#print axioms Search.Certificate.Robustness.supportOfOracle_agreement
#print axioms Search.Certificate.Robustness.supportOfOracle_minimal
#print axioms Search.Certificate.Robustness.supportOfPosition_agreement
#print axioms Search.Certificate.Robustness.supportOfPosition_card_le_maxLocalSupport
#print axioms Search.Certificate.Robustness.supportOfPosition_minimal
#print axioms Search.Certificate.Robustness.supportOfTotalPosition_agreement
#print axioms Search.Certificate.Robustness.tailClosure_readSet_card_le_multiplicity
#print axioms Search.Certificate.Robustness.tailClosure_shapeFixedSupport_card_le_recordBound
#print axioms Search.Certificate.Robustness.terminalCorruption_orbit_card
#print axioms Search.Certificate.Robustness.terminalCorruption_orbit_exposes_root
#print axioms Search.Certificate.Robustness.terminalPayloadLengths_decodeShapeFixed
#print axioms Search.Certificate.Robustness.totalCheckPositionB_decode_encode
#print axioms Search.Certificate.Robustness.total_formation_invalid_canonical_amplification
#print axioms Search.Certificate.Robustness.total_formation_invalid_syntactic_amplification
#print axioms Search.Certificate.Robustness.witnessFaithful_no_uniform_graph_bound
#print axioms Search.Certificate.Robustness.witnessFaithful_readBound_unbounded
#print axioms Search.Certificate.Robustness.witnessFaithful_readSet_card_le_multiplicity
#print axioms Search.Certificate.Robustness.witnessFaithful_shapeFixedSupport_card_le_recordBound
#print axioms Search.Certificate.Robustness.witnessFieldBudget_le_witnessCount_mul_maxWidth
#print axioms Search.Certificate.checkSemanticPositionB_eq_totalCheckPositionB
#print axioms Search.Certificate.checkSemanticPositionB_eq_totalCheckPositionB_index
#print axioms Search.Certificate.formationValid_iff_no_semanticMarked
#print axioms Search.Certificate.formation_invalid_implies_marked
#print axioms Search.Certificate.formation_invalid_implies_semanticMarked
#print axioms Search.Certificate.semanticPositionIndex_getElem
#print axioms Search.Certificate.semanticPositions_nodup
#print axioms Search.Certificate.strongFormationValid_iff_no_strongSemanticMarked
#print axioms Search.Certificate.strongFormation_invalid_implies_strongSemanticMarked
#print axioms Search.Certificate.total_formation_invalid_implies_marked
#print axioms Certificate.Flow.ListSetEq
#print axioms Certificate.Flow.compressed_universally_accepted_certificateB
#print axioms Certificate.Flow.localExpected_setEq_flow
#print axioms Certificate.Flow.productionAtB_setEq
#print axioms Certificate.Flow.rootRouteEmptyB_eq_true_iff
#print axioms Certificate.Flow.root_has_empty_flowRoute_of_localChecks
#print axioms Certificate.Flow.storedRoutes_setEq_flow
#print axioms Certificate.Flow.strongFormationValidB_sound
#print axioms Certificate.Flow.strongFormationValid_rootsDischargedB
#print axioms Certificate.Flow.strongLocalChecksB_sound
#print axioms Certificate.Flow.universalLocalRootDischargeAtB_eq_true_iff
#print axioms Certificate.Flow.universalLocalRootDischargeB_eq_true_iff
#print axioms Certificate.Flow.universalLocalRootDischarge_of_parts
#print axioms Certificate.Flow.universalLocalRootDischarge_residualClosure
#print axioms Certificate.Flow.universalLocalRootDischarge_rootsDischargedB
#print axioms Certificate.Flow.universalLocalRootDischarge_stored
#print axioms DLDS.ProofSearch.BranchingGrid.branchingGridMarkedB_embedAssignment
#print axioms DLDS.ProofSearch.BranchingGrid.branchingGridMarked_extracts
#print axioms DLDS.ProofSearch.BranchingGrid.branchingGridMarked_sound
#print axioms DLDS.Certificate.Flow.Cache.cachedDischargedMultiB_eq_dischargedMultiB
#print axioms DLDS.Certificate.Flow.Cache.dischargedMultiB_of_cached
#print axioms DLDS.Readings.Explicit.CoveredComposition.PairedCoherentExtension.covers
#print axioms DLDS.Readings.Explicit.CoveredComposition.allExplicitOutputsCovered_graph
#print axioms DLDS.Readings.Explicit.CoveredComposition.coherentB_reading
#print axioms DLDS.Readings.Explicit.CoveredComposition.coherent_reading_count
#print axioms DLDS.Readings.Explicit.CoveredComposition.control_mark_count
#print axioms DLDS.Readings.Explicit.CoveredComposition.control_no_labelDefect
#print axioms DLDS.Readings.Explicit.CoveredComposition.defective_conditioned_coherent_density
#print axioms DLDS.Readings.Explicit.CoveredComposition.defective_mark_count
#print axioms DLDS.Readings.Explicit.CoveredComposition.defective_not_explicitOutputLabelsCorrect
#print axioms DLDS.Readings.Explicit.CoveredComposition.defective_total_raw_density
#print axioms DLDS.Readings.Explicit.CoveredComposition.every_edge_strictly_descends
#print axioms DLDS.Readings.Explicit.CoveredComposition.every_node_reaches_root_within_two_edges
#print axioms DLDS.Readings.Explicit.CoveredComposition.explicitOutputLabelsCorrect_iff_no_labelDefect_graph
#print axioms DLDS.Readings.Explicit.CoveredComposition.labelDefectB_defective_iff
#print axioms DLDS.Readings.Explicit.CoveredComposition.producer_delivers_e0
#print axioms DLDS.Readings.Explicit.CoveredComposition.producer_output_reachable
#print axioms DLDS.Readings.Explicit.CoveredComposition.raw_reading_count
#print axioms DLDS.Readings.Explicit.CoveredComposition.selectedRuleActiveB_e0
#print axioms DLDS.Readings.Explicit.CoveredComposition.sourceNodes_graph
#print axioms DLDS.Readings.Explicit.CoveredComposition.sufficientSupport_derives_operational_conditions
#print axioms DLDS.Readings.Explicit.CoveredComposition.sufficientSupport_forces_rejection
#print axioms DLDS.Readings.Explicit.CoveredComposition.three_final_bad_choice_outcomes
#print axioms DLDS.Readings.Explicit.Examples.Elimination.local_bad_coordinate_suffices
#print axioms DLDS.Readings.Explicit.Examples.Elimination.two_coordinates_force_labelDefect
#print axioms DLDS.Readings.Explicit.Examples.MultiTokenMarker.defective_not_local_rule_correct
#print axioms DLDS.Readings.Explicit.InteractingPremises.allExplicitOutputsCovered_graph
#print axioms DLDS.Readings.Explicit.InteractingPremises.coherentB_reading
#print axioms DLDS.Readings.Explicit.InteractingPremises.control_no_labelDefect
#print axioms DLDS.Readings.Explicit.InteractingPremises.defective_conditioned_coherent_density
#print axioms DLDS.Readings.Explicit.InteractingPremises.defective_mark_count
#print axioms DLDS.Readings.Explicit.InteractingPremises.defective_total_raw_density
#print axioms DLDS.Readings.Explicit.InteractingPremises.labelDefectB_defective_iff
#print axioms DLDS.Readings.Explicit.InteractingPremises.local_output_alone_insufficient
#print axioms DLDS.Readings.Explicit.InteractingPremises.selectedRuleActiveB_e0
#print axioms DLDS.Readings.Explicit.InteractingPremises.selectedRuleActiveB_e1
#print axioms DLDS.Readings.Explicit.InteractingPremises.sufficientSupport_forces_labelDefect
#print axioms DLDS.Readings.Explicit.TwoStageComposition.coherentB_reading
#print axioms DLDS.Readings.Explicit.TwoStageComposition.coherent_reading_count
#print axioms DLDS.Readings.Explicit.TwoStageComposition.control_mark_count
#print axioms DLDS.Readings.Explicit.TwoStageComposition.control_no_labelDefect
#print axioms DLDS.Readings.Explicit.TwoStageComposition.defective_conditioned_coherent_density
#print axioms DLDS.Readings.Explicit.TwoStageComposition.defective_e0_output_covered
#print axioms DLDS.Readings.Explicit.TwoStageComposition.defective_mark_count
#print axioms DLDS.Readings.Explicit.TwoStageComposition.defective_total_raw_density
#print axioms DLDS.Readings.Explicit.TwoStageComposition.labelDefectB_defective_iff
#print axioms DLDS.Readings.Explicit.TwoStageComposition.not_allExplicitOutputsCovered_graph
#print axioms DLDS.Readings.Explicit.TwoStageComposition.producer_delivers_e0
#print axioms DLDS.Readings.Explicit.TwoStageComposition.producer_output_reachable
#print axioms DLDS.Readings.Explicit.TwoStageComposition.raw_reading_count
#print axioms DLDS.Readings.Explicit.TwoStageComposition.selectedRuleActiveB_e0
#print axioms DLDS.Readings.Explicit.TwoStageComposition.sourceNodes_graph
#print axioms DLDS.Readings.Explicit.TwoStageComposition.sufficientSupport_derives_operational_conditions
#print axioms DLDS.Readings.Explicit.TwoStageComposition.sufficientSupport_forces_rejection
#print axioms DLDS.Readings.Explicit.UniformElimFanFamily.agree_badSupport_forces_labelDefect
#print axioms DLDS.Readings.Explicit.UniformElimFanFamily.allExplicitOutputsCovered_graph
#print axioms DLDS.Readings.Explicit.UniformElimFanFamily.badCylinder_raw_density
#print axioms DLDS.Readings.Explicit.UniformElimFanFamily.bad_coordinate_forces_labelDefect
#print axioms DLDS.Readings.Explicit.UniformElimFanFamily.coherent_all
#print axioms DLDS.Readings.Explicit.UniformElimFanFamily.readings_nodup
#print axioms DLDS.Readings.Explicit.explicitOutputLabelsCorrectB_eq_true_iff
#print axioms DLDS.Readings.Explicit.explicitOutputLabelsCorrect_iff_localRuleCorrect
#print axioms DLDS.Readings.Explicit.explicitOutputLabelsCorrect_iff_no_labelDefect
#print axioms DLDS.Readings.Explicit.explicitOutputViolation_exposed
#print axioms DLDS.Readings.Explicit.filter_labelDefectB_eq_filter_coherentReadings
#print axioms DLDS.Readings.Explicit.labelDefectB_eq_false_of_explicitOutputLabelsCorrect
#print axioms DLDS.Readings.Explicit.labelDefectB_implies_not_localRuleCorrect
#print axioms DLDS.Readings.Explicit.labelDefectB_persists_of_coherent_active
#print axioms DLDS.Readings.Explicit.labelDefectB_sound
#print axioms DLDS.Readings.Explicit.rejection_persists_of_incoherent_or_active
#print axioms DLDS.Readings.Explicit.rejectsReadingLabelsB_eq_false_of_selected_correct
#print axioms DLDS.Readings.Explicit.rejectsReadingLabelsB_eq_true_iff
#print axioms DLDS.Readings.Explicit.selectedLabelErrorAtB_eq_outDep_comparison
#print axioms DLDS.Readings.Explicit.selectedLabelErrorAtB_eq_true_iff
#print axioms DLDS.Readings.Explicit.selectedLabelHadErrorB_eq_true_iff
#print axioms DLDS.Readings.Explicit.selectedRuleApplicationMismatchB_marks_labelDefectB
#print axioms DLDS.Readings.Explicit.selectedRuleApplicationMismatchB_rejectsReadingLabelsB
#print axioms DLDS.Examples.Readings.MultiToken.d11_count_rejects_eq_admissible
#print axioms DLDS.Examples.Readings.MultiToken.d11_swapped_rejects
#print axioms DLDS.Examples.ThreeVertex.Flow.Acceptance.bridge_invariant_package
#print axioms DLDS.Examples.ThreeVertex.Flow.Acceptance.closed_3v_self_certifying
#print axioms DLDS.Examples.FourVertex.Flow.Acceptance.closed_4v_self_certifying_of_cached
#print axioms DLDS.Examples.FourVertex.Flow.EndToEnd.closed_4v_self_certifying
#print axioms DLDS.Readings.LocalLabels.Examples.Elimination.exposed_reading_ratio
#print axioms DLDS.Readings.LocalLabels.Examples.Elimination.rule_application_mismatch_ratio
#print axioms DLDS.Readings.LocalLabels.selectedDecorationMismatchB_persists
#print axioms DLDS.Readings.LocalLabels.selectedRuleApplicationMismatchB_persists
#print axioms DLDS.admissibleMulti_accept_canonical
#print axioms DLDS.canonical_multi_accepts
#print axioms DLDS.compressed_universally_accepted'
#print axioms DLDS.compressed_universally_accepted_genuine
#print axioms DLDS.count_rejectsMultiB_eq_admissible
#print axioms DLDS.depVector_eraseDups_append_eq_zipWith_or
#print axioms DLDS.dlds_evaluation_iff
#print axioms DLDS.dlds_global_iff
#print axioms DLDS.evaluateDLDS_eq_of_nodes_eq
#print axioms DLDS.multiple_xor_cons_true
#print axioms DLDS.multiple_xor_cons_true_aux
#print axioms DLDS.node_correct
#print axioms DLDS.rejectsMultiB_persists_on_admissible
#print axioms DLDS.tree_bridge_forward
#print axioms DLDS.tree_bridge_forward_of_descent_coherent
#print axioms ProofSearch.Grid.Schedulability.exists_marked_iff_exists_closed_gridSchedulable
#print axioms ProofSearch.Grid.Counting.encodedDerivation_density
#print axioms ProofSearch.Grid.Counting.gridMarkedB_unread_invariant
#print axioms ProofSearch.Grid.Counting.readOrbit_card
#print axioms DLDS.ValidityDecomposition.Counterexamples.rawFlowRootDischarge_does_not_imply_rootDecoration
#print axioms DLDS.ValidityDecomposition.Counterexamples.rootDecoration_does_not_imply_rawFlowRootDischarge
#print axioms DLDS.ValidityDecomposition.validDLDS_and_hcRepresentation_implies_suitableGraph
#print axioms DLDS.ValidityDecomposition.validDLDS_hc_roots_iff_declarative_hc_exists_strongCertificate
#print axioms DLDS.ValidityDecomposition.validDLDS_hc_roots_iff_declarative_hc_flowGrid
#print axioms DLDS.ValidityDecomposition.validDLDS_to_flowCompiledGridValid
