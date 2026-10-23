import Semantic.RouteIndexedMultiTokenBridge

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe

namespace CertificateSkeletonRouteEvaluator

open RouteIndexedMultiToken











inductive SkeletonOutlet where
  | routeEdge (edge : Deduction)
  | routeTerminal (node : Vertex)
  deriving Repr, BEq, DecidableEq


structure CertificateRouteOccurrence where
  outlet : SkeletonOutlet
  residual : ColourPath
  deriving Repr, BEq, DecidableEq





structure CertificateRouteSkeleton where
  graph : Graph
  occurrences : List CertificateRouteOccurrence
  deriving Repr, BEq, DecidableEq

def CertificateRouteOccurrence.vertex
    (occurrence : CertificateRouteOccurrence) : Vertex :=
  match occurrence.outlet with
  | SkeletonOutlet.routeEdge route_edge => route_edge.START
  | SkeletonOutlet.routeTerminal route_node => route_node

def eraseCertificateDependencies (c : Certificate) : CertificateRouteSkeleton :=
  { graph := c.graph
    occurrences :=
      c.edgeRoutes.map (fun witness =>
        { outlet := .routeEdge witness.edge, residual := witness.residual }) ++
      c.terminalOutputs.map (fun output =>
        { outlet := .routeTerminal output.node, residual := output.residual }) }


def rewriteStoredDependencies (rewrite : Dep -> Dep)
    (c : Certificate) : Certificate :=
  { graph := c.graph
    edgeRoutes := c.edgeRoutes.map fun witness =>
      { witness with dependency := rewrite witness.dependency }
    terminalOutputs := c.terminalOutputs.map fun output =>
      { output with dependency := rewrite output.dependency } }



theorem eraseCertificateDependencies_rewriteStoredDependencies
    (rewrite : Dep -> Dep) (c : Certificate) :
    eraseCertificateDependencies (rewriteStoredDependencies rewrite c) =
      eraseCertificateDependencies c := by
  simp [eraseCertificateDependencies, rewriteStoredDependencies,
    Function.comp_def]

def occurrencesAt (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    List (CertificateRouteOccurrence × Nat) :=
  skeleton.occurrences.zipIdx.filter fun occurrence =>
    occurrence.1.vertex == v

def matchingOccurrences (skeleton : CertificateRouteSkeleton) (v : Vertex)
    (residual : ColourPath) : List (CertificateRouteOccurrence × Nat) :=
  (occurrencesAt skeleton v).filter fun occurrence =>
    occurrence.1.residual == residual


def skeletonResidualsAt (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    List ColourPath :=
  (skeleton.occurrences.filterMap fun occurrence =>
    if occurrence.vertex == v then some occurrence.residual else none).eraseDups


def skeletonResidualsForEdge (skeleton : CertificateRouteSkeleton)
    (edge : Deduction) : List ColourPath :=
  skeleton.occurrences.filterMap fun occurrence =>
    match occurrence.outlet with
    | .routeEdge candidate =>
        if candidate == edge then some occurrence.residual else none
    | .routeTerminal _ => none

theorem skeletonResidualsForEdge_erase (c : Certificate)
    (edge : Deduction) :
    ListSetEq
      (skeletonResidualsForEdge (eraseCertificateDependencies c) edge)
      ((edgeWitnessesFor c edge).map fun witness => witness.residual) := by
  intro residual
  simp [skeletonResidualsForEdge, eraseCertificateDependencies,
    edgeWitnessesFor]
  aesop

theorem skeletonResidualsAt_erase (c : Certificate) (v : Vertex) :
    ListSetEq (skeletonResidualsAt (eraseCertificateDependencies c) v)
      ((StoredRoutes c v).map Prod.snd) := by
  intro residual
  simp [skeletonResidualsAt, eraseCertificateDependencies,
    CertificateRouteOccurrence.vertex, StoredRoutes, outputsAt]
  aesop

private lemma mapResiduals_congr {left right : List FlowPair}
    (agrees : ListSetEq left right) :
    ListSetEq (left.map Prod.snd) (right.map Prod.snd) := by
  intro residual
  simp only [List.mem_map]
  constructor
  · rintro ⟨route, route_mem, rfl⟩
    exact ⟨route, (agrees route).mp route_mem, rfl⟩
  · rintro ⟨route, route_mem, rfl⟩
    exact ⟨route, (agrees route).mpr route_mem, rfl⟩

private lemma filteredResiduals_congr {left right : List FlowPair}
    (colour : Nat) (agrees : ListSetEq left right) :
    ListSetEq
      ((left.filter fun route => headColour route.2 == colour).map Prod.snd)
      ((right.filter fun route => headColour route.2 == colour).map Prod.snd) := by
  intro residual
  simp only [List.mem_map, List.mem_filter]
  constructor
  · rintro ⟨route, ⟨route_mem, colour_eq⟩, rfl⟩
    exact ⟨route, ⟨(agrees route).mp route_mem, colour_eq⟩, rfl⟩
  · rintro ⟨route, ⟨route_mem, colour_eq⟩, rfl⟩
    exact ⟨route, ⟨(agrees route).mpr route_mem, colour_eq⟩, rfl⟩




theorem skeletonResidualsAt_setEq_witnessableFlow
    (d : Graph) (c : Certificate) (expanded : ExpandedCert d c)
    (checks : LocalChecks d c)
    (v : Vertex) (v_mem : v ∈ d.NODES) :
    ListSetEq (skeletonResidualsAt (eraseCertificateDependencies c) v)
      ((WitnessableRoutes d v (flowAt d (stdFuel d) v)).map Prod.snd) :=
  (skeletonResidualsAt_erase c v).trans
    (mapResiduals_congr
      (storedRoutes_setEq_flow d c expanded checks v v_mem))

theorem skeletonResidualsAt_setEq_witnessableFlow_of_strongFormation
    (d : Graph) (c : Certificate) (valid : StrongFormationValid d c)
    (v : Vertex) (v_mem : v ∈ d.NODES) :
    ListSetEq (skeletonResidualsAt (eraseCertificateDependencies c) v)
      ((WitnessableRoutes d v (flowAt d (stdFuel d) v)).map Prod.snd) :=
  skeletonResidualsAt_setEq_witnessableFlow d c valid.1 valid.2.1 v v_mem




theorem skeletonResidualsForEdge_setEq_witnessableFlow
    (d : Graph) (c : Certificate) (expanded : ExpandedCert d c)
    (checks : LocalChecks d c)
    (v : Vertex) (v_mem : v ∈ d.NODES) (edge : Deduction)
    (edge_mem : edge ∈ get_rule.outgoing v d) :
    ListSetEq
      (skeletonResidualsForEdge (eraseCertificateDependencies c) edge)
      (((WitnessableRoutes d v (flowAt d (stdFuel d) v)).filter fun route =>
        headColour route.2 == edge.COLOUR).map Prod.snd) := by
  have graph_eq : c.graph = d := expanded.1
  have v_mem_c : v ∈ c.graph.NODES := by simpa [graph_eq] using v_mem
  have edge_mem_c : edge ∈ get_rule.outgoing v c.graph := by
    simpa [graph_eq] using edge_mem
  have coverage := checks.2.2.1 v v_mem_c edge edge_mem_c
  have certificate_edge_routes :
      ListSetEq
        ((edgeWitnessesFor c edge).map fun witness => witness.residual)
        (((StoredRoutes c v).filter fun route =>
          headColour route.2 == edge.COLOUR).map Prod.snd) := by
    have mapped := mapResiduals_congr coverage.2
    simpa [StoredRoutes, List.map_map, Function.comp_def] using mapped
  exact (skeletonResidualsForEdge_erase c edge).trans
    (certificate_edge_routes.trans
      (filteredResiduals_congr edge.COLOUR
        (storedRoutes_setEq_flow d c expanded checks v v_mem)))

theorem skeletonResidualsForEdge_setEq_witnessableFlow_of_strongFormation
    (d : Graph) (c : Certificate) (valid : StrongFormationValid d c)
    (v : Vertex) (v_mem : v ∈ d.NODES) (edge : Deduction)
    (edge_mem : edge ∈ get_rule.outgoing v d) :
    ListSetEq
      (skeletonResidualsForEdge (eraseCertificateDependencies c) edge)
      (((WitnessableRoutes d v (flowAt d (stdFuel d) v)).filter fun route =>
        headColour route.2 == edge.COLOUR).map Prod.snd) :=
  skeletonResidualsForEdge_setEq_witnessableFlow d c valid.1 valid.2.1
    v v_mem edge edge_mem





def annotateToken (skeleton : CertificateRouteSkeleton) (v : Vertex)
    (token : RouteIndexedToken) : List RouteIndexedToken :=
  match matchingOccurrences skeleton v token.residual with
  | [] => [token]
  | occurrences => occurrences.map fun occurrence =>
      { token with
        routeKey := .slot v occurrence.2 token.routeKey }

def annotateTokens (skeleton : CertificateRouteSkeleton) (v : Vertex)
    (tokens : List RouteIndexedToken) : List RouteIndexedToken :=
  tokens.flatMap (annotateToken skeleton v)



lemma annotateToken_project (skeleton : CertificateRouteSkeleton) (v : Vertex)
    (token : RouteIndexedToken) :
    ListSetEq (projectFlowPairs (annotateToken skeleton v token))
      [token.flowPair] := by
  unfold annotateToken
  cases hmatching : matchingOccurrences skeleton v token.residual with
  | nil => exact ListSetEq.refl _
  | cons occurrence occurrences =>
      intro route
      simp [projectFlowPairs, RouteIndexedToken.flowPair]
      aesop

theorem annotateTokens_project (skeleton : CertificateRouteSkeleton)
    (v : Vertex) (tokens : List RouteIndexedToken) :
    ListSetEq (projectFlowPairs (annotateTokens skeleton v tokens))
      (projectFlowPairs tokens) := by
  induction tokens with
  | nil => exact ListSetEq.refl []
  | cons token tokens ih =>
      unfold annotateTokens
      simp only [List.flatMap_cons, projectFlowPairs, List.map_append,
        List.map_cons]
      exact append_congr (annotateToken_project skeleton v token) ih




def certificateSkeletonSource (skeleton : CertificateRouteSkeleton)
    (v : Vertex) : List RouteIndexedToken :=
  annotateTokens skeleton v (sourceOutputs skeleton.graph v)

theorem certificateSkeletonSource_project
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    ListSetEq
      (projectFlowPairs (certificateSkeletonSource skeleton v))
      (let ancestors := ancestorsInto skeleton.graph v
       if ancestors.isEmpty then [(normalizeDep [v.FORMULA], [])]
       else ancestors.map fun ancestor =>
         (normalizeDep [v.FORMULA], ancestor.COLOURS)) := by
  exact (annotateTokens_project skeleton v _).trans
    (by rw [project_sourceOutputs]; exact ListSetEq.refl _)

theorem certificateSkeletonSource_dependencies
    (skeleton : CertificateRouteSkeleton) (v : Vertex)
    (token : RouteIndexedToken)
    (member : token ∈ certificateSkeletonSource skeleton v) :
    token.dependency = normalizeDep [v.FORMULA] := by
  unfold certificateSkeletonSource annotateTokens at member
  simp only [List.mem_flatMap] at member
  obtain ⟨source, source_mem, token_mem⟩ := member
  have source_dependency : source.dependency = normalizeDep [v.FORMULA] := by
    by_cases ancestors_empty : ancestorsInto skeleton.graph v = []
    · simp [sourceOutputs, ancestors_empty, mkToken] at source_mem
      subst source
      exact (normalizeDep_singleton v.FORMULA).symm
    · simp [sourceOutputs, ancestors_empty, mkToken] at source_mem
      obtain ⟨ancestor, _, rfl⟩ := source_mem
      exact (normalizeDep_singleton v.FORMULA).symm
  unfold annotateToken at token_mem
  split at token_mem
  · simp_all
  · simp only [List.mem_map] at token_mem
    obtain ⟨occurrence, _, rfl⟩ := token_mem
    exact source_dependency




def certificateSkeletonLocalOutputs
    (skeleton : CertificateRouteSkeleton)
    (premiseTokens : Vertex -> List RouteIndexedToken) (v : Vertex) :
    List RouteIndexedToken :=
  annotateTokens skeleton v
    (stepFrom skeleton.graph premiseTokens v)



theorem certificateSkeletonLocal_structuralAgreement
    (skeleton : CertificateRouteSkeleton)
    (premiseTokens : Vertex -> List RouteIndexedToken) (v : Vertex) :
    ListSetEq
      (projectFlowPairs
        (certificateSkeletonLocalOutputs skeleton premiseTokens v))
      (projectFlowPairs (stepFrom skeleton.graph premiseTokens v)) :=
  annotateTokens_project skeleton v _




theorem certificateSkeletonLocalOutputs_setEq_flowStepFrom
    (skeleton : CertificateRouteSkeleton)
    (premiseTokens : Vertex -> List RouteIndexedToken)
    (premiseFlow : Vertex -> List FlowPair) (v : Vertex)
    (agrees : forall u,
      ListSetEq (projectFlowPairs (premiseTokens u)) (premiseFlow u)) :
    ListSetEq
      (projectFlowPairs
        (certificateSkeletonLocalOutputs skeleton premiseTokens v))
      (flowStepFrom skeleton.graph premiseFlow v) :=
  (certificateSkeletonLocal_structuralAgreement skeleton premiseTokens v).trans
    (routeIndexedLocalOutputs_setEq_flowStepFrom skeleton.graph
      premiseTokens premiseFlow v agrees)




def certificateSkeletonAt (skeleton : CertificateRouteSkeleton) :
    Nat -> Vertex -> List RouteIndexedToken
  | 0, _ => []
  | fuel + 1, v =>
      certificateSkeletonLocalOutputs skeleton
        (certificateSkeletonAt skeleton fuel) v




theorem certificateSkeletonAt_setEq_flowAt
    (skeleton : CertificateRouteSkeleton) :
    forall fuel v,
      ListSetEq
        (projectFlowPairs (certificateSkeletonAt skeleton fuel v))
        (flowAt skeleton.graph fuel v) := by
  intro fuel
  induction fuel with
  | zero =>
      intro v
      exact ListSetEq.refl []
  | succ fuel ih =>
      intro v
      exact certificateSkeletonLocalOutputs_setEq_flowStepFrom skeleton
        (certificateSkeletonAt skeleton fuel)
        (flowAt skeleton.graph fuel) v ih

def certificateSkeletonCanonicalAt (c : Certificate) (v : Vertex) :
    List RouteIndexedToken :=
  let skeleton := eraseCertificateDependencies c
  certificateSkeletonAt skeleton (stdFuel skeleton.graph) v

theorem certificateSkeletonCanonical_setEq_flowAt
    (c : Certificate) (v : Vertex) :
    ListSetEq (projectFlowPairs (certificateSkeletonCanonicalAt c v))
      (flowAt c.graph (stdFuel c.graph) v) := by
  exact certificateSkeletonAt_setEq_flowAt
    (eraseCertificateDependencies c) (stdFuel c.graph) v

theorem certificateSkeletonCanonicalWitnessable_setEq
    (c : Certificate) (v : Vertex) :
    ListSetEq
      (projectFlowPairs
        (witnessableAt c.graph v (certificateSkeletonCanonicalAt c v)))
      (WitnessableRoutes c.graph v
        (flowAt c.graph (stdFuel c.graph) v)) :=
  (project_witnessableAt c.graph v _).trans
    (witnessableRoutes_congr c.graph v
      (certificateSkeletonCanonical_setEq_flowAt c v))


def CertificateSkeletonDischarged (d : Graph) (c : Certificate) : Prop :=
  forall root, root ∈ d.NODES ->
    (get_rule.outgoing root d).isEmpty = true ->
      projectFlowPairs (certificateSkeletonCanonicalAt c root) ≠ [] /\
      forall route,
        route ∈ projectFlowPairs (certificateSkeletonCanonicalAt c root) ->
          route.1 = [] /\ route.2 = []

theorem certificateSkeletonDischarged_of_rootsDischargedB
    (d : Graph) (c : Certificate) (graph_eq : c.graph = d)
    (discharged : rootsDischargedB d = true) :
    CertificateSkeletonDischarged d c := by
  subst d
  unfold rootsDischargedB at discharged
  rw [List.all_eq_true] at discharged
  intro root root_mem root_is_root
  have root_check := discharged root root_mem
  simp only [root_is_root, Bool.not_true, Bool.false_or,
    Bool.and_eq_true, List.all_eq_true] at root_check
  have correspondence := certificateSkeletonCanonical_setEq_flowAt c root
  constructor
  · intro projected_empty
    have flow_empty : flowAt c.graph (stdFuel c.graph) root = [] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro route route_mem
      have projected_mem := (correspondence route).mpr route_mem
      simp [projected_empty] at projected_mem
    have flow_nonempty : flowAt c.graph (stdFuel c.graph) root ≠ [] := by
      simpa [List.isEmpty_iff] using root_check.1
    exact flow_nonempty flow_empty
  · intro route route_mem
    have flow_mem := (correspondence route).mp route_mem
    have route_check := root_check.2 route flow_mem
    simpa [List.isEmpty_iff] using route_check




theorem certificateSkeletonDischarged_of_strongFormation
    (d : Graph) (c : Certificate) (valid : StrongFormationValid d c) :
    CertificateSkeletonDischarged d c :=
  certificateSkeletonDischarged_of_rootsDischargedB d c valid.1.1
    (strongFormationValid_rootsDischargedB d c valid)


end CertificateSkeletonRouteEvaluator
