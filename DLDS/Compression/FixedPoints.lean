import DLDS.ValidityDecomposition
import DLDS.Compression.IndexedClasses

open DLDS Flow Certificate.Flow.Foundation Certificate.Flow
open Certificate.Flow.GeneratorCompleteness ProofSearch.FlowCorrespondence
open DLDS.ValidityDecomposition

namespace DLDS.Compression.FixedPoints

/-!
# Fixpoints of executable horizontal compression

`compress_nodes_graph` consumes a `Graph`, not an `NDTree`.  This module records
the following construction-independent property:
when its precomputed candidate list is empty, it is definitionally a no-op.
The semantic consequences below are preservation results for such fixpoints;
they deliberately do not claim that compression establishes their hypotheses.
-/

/-- If the executable candidate search finds no repeated level/formula class,
the outer compression loop performs no step. -/
theorem compress_nodes_graph_eq_self_of_no_candidates (d : Graph)
    (noCandidates : repeated_nodes_vertex d = []) :
    compress_nodes_graph d = d := by
  unfold compress_nodes_graph
  dsimp only
  rw [DLDS.Compression.IndexedClasses.repeated_nodes_vertex_indexedWithPartition_eq]
  rw [noCandidates]
  rfl

/-! ## The source-occurrence obstruction exposed by the 5v artifact -/

/-- At a hypothesis with at least one ancestral record, raw Flow uses exactly
the ancestral colour paths; it does not additionally seed the default empty
route.  Consequently, cursor route-head coverage forces every outgoing edge
colour to occur as the head of one of those ancestral paths. -/
theorem source_edge_has_ancestral_head_of_cursorRouteHeadCoverage
    {d : Graph} {v : Vertex} {edge : Deduction}
    (coverage : CursorRouteHeadCoverage d)
    (vertexMember : v ∈ d.NODES)
    (hypothesis : v.HYPOTHESIS = true)
    (ancestorsNonempty : ancestorsInto d v ≠ [])
    (edgeMember : edge ∈ get_rule.outgoing v d) :
    ∃ ancestor ∈ ancestorsInto d v,
      headColour ancestor.COLOURS = edge.COLOUR := by
  have outgoingNonempty : get_rule.outgoing v d ≠ [] := by
    intro empty
    rw [empty] at edgeMember
    simp at edgeMember
  obtain ⟨route, routeMember, routeColour⟩ :=
    (coverage v vertexMember outgoingNonempty).2 edge edgeMember
  have routeMember' : route ∈
      (ancestorsInto d v).map fun ancestor =>
        (normalizeDep [v.FORMULA], ancestor.COLOURS) := by
    simpa [stdFuel, flowAt, hypothesis, List.isEmpty_iff,
      ancestorsNonempty] using routeMember
  rw [List.mem_map] at routeMember'
  obtain ⟨ancestor, ancestorMember, routeEq⟩ := routeMember'
  refine ⟨ancestor, ancestorMember, ?_⟩
  subst route
  exact routeColour

theorem not_cursorRouteHeadCoverage_of_source_edge_without_ancestral_head
    {d : Graph} {v : Vertex} {edge : Deduction}
    (vertexMember : v ∈ d.NODES)
    (hypothesis : v.HYPOTHESIS = true)
    (ancestorsNonempty : ancestorsInto d v ≠ [])
    (edgeMember : edge ∈ get_rule.outgoing v d)
    (headMissing : ∀ ancestor ∈ ancestorsInto d v,
      headColour ancestor.COLOURS ≠ edge.COLOUR) :
    ¬ CursorRouteHeadCoverage d := by
  intro coverage
  obtain ⟨ancestor, ancestorMember, head⟩ :=
    source_edge_has_ancestral_head_of_cursorRouteHeadCoverage coverage
      vertexMember hypothesis ancestorsNonempty edgeMember
  exact headMissing ancestor ancestorMember head

theorem compress_fixpoint_levelFormulaUnique (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (unique : LevelFormulaUnique d) :
    LevelFormulaUnique (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact unique

theorem compress_fixpoint_oneEdgePerColour (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (unique : OneEdgePerColourPerNode d) :
    OneEdgePerColourPerNode (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact unique

theorem compress_fixpoint_routeHeadCoverage (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (coverage : CursorRouteHeadCoverage d) :
    CursorRouteHeadCoverage (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact coverage

theorem compress_fixpoint_flowTailClosure (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (closure : CursorFlowTailClosure d) :
    CursorFlowTailClosure (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact closure

theorem compress_fixpoint_hcRepresentationWellFormed (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (representation : HCRepresentationWellFormed d) :
    HCRepresentationWellFormed (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact representation

theorem compress_fixpoint_suitableGraph (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (suitable : SuitableGraph d) :
    SuitableGraph (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact suitable

theorem compress_fixpoint_rootsDischargedB (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (roots : rootsDischargedB d = true) :
    rootsDischargedB (compress_nodes_graph d) = true := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact roots

theorem compress_fixpoint_flowCompiledGridValid (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (roots : rootsDischargedB d = true) :
    FlowCompiledGridValid (compress_nodes_graph d) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact (rootsDischargedB_iff_flowCompiledGridValid d).mp roots

theorem compress_fixpoint_generatedStrongFormation (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (suitable : SuitableGraph d)
    (roots : rootsDischargedB d = true) :
    StrongFormationValid (compress_nodes_graph d)
      (generateCertificate (compress_nodes_graph d)) := by
  rw [compress_nodes_graph_eq_self_of_no_candidates d noCandidates]
  exact generateCertificate_strongFormation d suitable roots

/-- End-to-end semantic triangle for an already suitable HC fixpoint.  This is
a preservation theorem, not a construction-correctness theorem. -/
theorem compress_fixpoint_end_to_end (d : Graph)
    (noCandidates : repeated_nodes_vertex d = [])
    (suitable : SuitableGraph d)
    (roots : rootsDischargedB d = true) :
    HCRepresentationWellFormed (compress_nodes_graph d) ∧
      SuitableGraph (compress_nodes_graph d) ∧
      rootsDischargedB (compress_nodes_graph d) = true ∧
      FlowCompiledGridValid (compress_nodes_graph d) ∧
      StrongFormationValid (compress_nodes_graph d)
        (generateCertificate (compress_nodes_graph d)) := by
  have identity := compress_nodes_graph_eq_self_of_no_candidates d noCandidates
  rw [identity]
  exact
    ⟨⟨suitable.oneEdgePerColour, suitable.routeHeadCoverage,
        suitable.flowTailClosure, suitable.levelFormulaUnique⟩,
      suitable,
      roots,
      (rootsDischargedB_iff_flowCompiledGridValid d).mp roots,
      generateCertificate_strongFormation d suitable roots⟩


end DLDS.Compression.FixedPoints
