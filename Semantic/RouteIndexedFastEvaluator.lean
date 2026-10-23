import Semantic.CertificateSkeletonRouteEvaluator

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe
open RouteIndexedMultiToken CertificateSkeletonRouteEvaluator

namespace RouteIndexedFastEvaluator










abbrev RouteIndexedTable := List (Vertex × List RouteIndexedToken)

def routeIndexedTableLookup? (table : RouteIndexedTable) (v : Vertex) :
    Option (List RouteIndexedToken) :=
  (table.find? fun entry => entry.1 == v).map Prod.snd




def routeIndexedFromTable (d : Graph) (fuel : Nat)
    (table : RouteIndexedTable) (v : Vertex) : List RouteIndexedToken :=
  match routeIndexedTableLookup? table v with
  | some tokens => tokens
  | none => evaluateAt d fuel v

def routeIndexedTableSpec (d : Graph) (fuel : Nat) : RouteIndexedTable :=
  d.NODES.map fun v => (v, evaluateAt d fuel v)

lemma routeIndexedFromTable_spec (d : Graph) (fuel : Nat) (v : Vertex) :
    routeIndexedFromTable d fuel (routeIndexedTableSpec d fuel) v =
      evaluateAt d fuel v := by
  unfold routeIndexedTableSpec
  induction d.NODES with
  | nil => simp [routeIndexedFromTable, routeIndexedTableLookup?]
  | cons node nodes ih =>
      by_cases node_eq : node = v
      · subst v
        simp [routeIndexedFromTable, routeIndexedTableLookup?]
      · have node_beq : (node == v) = false := by simp [node_eq]
        simp only [List.map_cons, routeIndexedFromTable,
          routeIndexedTableLookup?, List.find?_cons, node_beq, Option.map]
        change routeIndexedFromTable d fuel
          (nodes.map fun key => (key, evaluateAt d fuel key)) v =
            evaluateAt d fuel v
        exact ih

def routeIndexedTableInitial (d : Graph) : RouteIndexedTable :=
  d.NODES.map fun v => (v, [])

def routeIndexedTableStep (d : Graph) (fuel : Nat)
    (previous : RouteIndexedTable) : RouteIndexedTable :=
  d.NODES.map fun v =>
    (v, stepFrom d (routeIndexedFromTable d fuel previous) v)

lemma routeIndexedTableInitial_eq_spec_zero (d : Graph) :
    routeIndexedTableInitial d = routeIndexedTableSpec d 0 := by
  rfl

lemma routeIndexedTableStep_spec (d : Graph) (fuel : Nat) :
    routeIndexedTableStep d fuel (routeIndexedTableSpec d fuel) =
      routeIndexedTableSpec d (fuel + 1) := by
  simp only [routeIndexedTableStep, routeIndexedTableSpec]
  apply List.map_congr_left
  intro v v_mem
  simp only [Prod.mk.injEq, true_and]
  change stepFrom d
      (routeIndexedFromTable d fuel (routeIndexedTableSpec d fuel)) v =
    evaluateAt d (fuel + 1) v
  rw [show routeIndexedFromTable d fuel (routeIndexedTableSpec d fuel) =
      evaluateAt d fuel from funext (routeIndexedFromTable_spec d fuel)]
  rfl

def routeIndexedTableIterate (d : Graph) :
    Nat -> Nat -> RouteIndexedTable -> RouteIndexedTable
  | 0, _, previous => previous
  | remaining + 1, fuel, previous =>
      routeIndexedTableIterate d remaining (fuel + 1)
        (routeIndexedTableStep d fuel previous)

lemma routeIndexedTableIterate_spec (d : Graph) : forall remaining fuel,
    routeIndexedTableIterate d remaining fuel
        (routeIndexedTableSpec d fuel) =
      routeIndexedTableSpec d (fuel + remaining) := by
  intro remaining
  induction remaining with
  | zero => intro fuel; rfl
  | succ remaining ih =>
      intro fuel
      rw [routeIndexedTableIterate, routeIndexedTableStep_spec]
      have rest := ih (fuel + 1)
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using rest

def routeIndexedTableAt (d : Graph) (fuel : Nat) : RouteIndexedTable :=
  routeIndexedTableIterate d fuel 0 (routeIndexedTableInitial d)

def routeIndexedTable (d : Graph) : RouteIndexedTable :=
  routeIndexedTableAt d (stdFuel d)









theorem routeIndexedTableAt_eq_spec (d : Graph) (fuel : Nat) :
    routeIndexedTableAt d fuel = routeIndexedTableSpec d fuel := by
  unfold routeIndexedTableAt
  rw [routeIndexedTableInitial_eq_spec_zero]
  simpa using routeIndexedTableIterate_spec d fuel 0

theorem routeIndexedTable_eq_spec (d : Graph) :
    routeIndexedTable d = routeIndexedTableSpec d (stdFuel d) := by
  exact routeIndexedTableAt_eq_spec d (stdFuel d)

def routeIndexedCanonicalFast (d : Graph) (v : Vertex) :
    List RouteIndexedToken :=
  routeIndexedFromTable d (stdFuel d) (routeIndexedTable d) v



theorem routeIndexedCanonicalFast_eq_canonical (d : Graph) (v : Vertex) :
    routeIndexedCanonicalFast d v = canonicalAt d v := by
  unfold routeIndexedCanonicalFast canonicalAt
  rw [routeIndexedTable_eq_spec]
  exact routeIndexedFromTable_spec d (stdFuel d) v

theorem routeIndexedCanonicalFast_setEq_canonical (d : Graph) (v : Vertex) :
    ListSetEq (projectFlowPairs (routeIndexedCanonicalFast d v))
      (projectFlowPairs (canonicalAt d v)) := by
  rw [routeIndexedCanonicalFast_eq_canonical]
  exact ListSetEq.refl _

theorem routeIndexedCanonicalFast_setEq_flowAt (d : Graph) (v : Vertex) :
    ListSetEq (projectFlowPairs (routeIndexedCanonicalFast d v))
      (flowAt d (stdFuel d) v) :=
  (routeIndexedCanonicalFast_setEq_canonical d v).trans
    (routeIndexedCanonical_setEq_flowAt d v)















def RouteIndexedCanonicalFixedPoint (d : Graph) : Prop :=
  forall v, v ∈ d.NODES ->
    stepFrom d (canonicalAt d) v = canonicalAt d v

def routeIndexedCanonicalFixedPointB (d : Graph) : Bool :=
  d.NODES.all fun v => decide (stepFrom d (canonicalAt d) v = canonicalAt d v)

theorem routeIndexedCanonicalFixedPointB_sound (d : Graph)
    (valid : routeIndexedCanonicalFixedPointB d = true) :
    RouteIndexedCanonicalFixedPoint d := by
  intro v v_mem
  have checked := List.all_eq_true.mp valid v v_mem
  exact of_decide_eq_true checked

def routeIndexedTopologicalOrder (d : Graph) : List Vertex :=
  d.NODES.mergeSort fun left right => right.LEVEL <= left.LEVEL

def routeIndexedTopologicalFromTable (d : Graph)
    (table : RouteIndexedTable) (v : Vertex) : List RouteIndexedToken :=
  match routeIndexedTableLookup? table v with
  | some tokens => tokens
  | none => canonicalAt d v

def routeIndexedTopologicalFold (d : Graph) :
    List Vertex -> RouteIndexedTable -> RouteIndexedTable
  | [], table => table
  | v :: rest, table =>
      let tokens := stepFrom d (routeIndexedTopologicalFromTable d table) v
      routeIndexedTopologicalFold d rest ((v, tokens) :: table)

def routeIndexedTopologicalTable (d : Graph) : RouteIndexedTable :=
  routeIndexedTopologicalFold d (routeIndexedTopologicalOrder d) []










def routeIndexedPredecessorClosure (d : Graph) :
    Nat -> List Vertex -> List Vertex
  | 0, vertices => vertices.eraseDups
  | fuel + 1, vertices =>
      let expanded := vertices ++ vertices.flatMap (predsOf d)
      routeIndexedPredecessorClosure d fuel expanded.eraseDups

def routeIndexedTopologicalOrderFor (d : Graph)
    (targets : List Vertex) : List Vertex :=
  let needed := routeIndexedPredecessorClosure d (stdFuel d) targets
  (routeIndexedTopologicalOrder d).filter needed.contains

def routeIndexedTopologicalTableFor (d : Graph)
    (targets : List Vertex) : RouteIndexedTable :=
  routeIndexedTopologicalFold d
    (routeIndexedTopologicalOrderFor d targets) []

def routeIndexedCanonicalTopologicalFast (d : Graph) (v : Vertex) :
    List RouteIndexedToken :=
  routeIndexedTopologicalFromTable d (routeIndexedTopologicalTable d) v

def routeIndexedCanonicalTopologicalFastFor (d : Graph)
    (targets : List Vertex) (v : Vertex) : List RouteIndexedToken :=
  routeIndexedTopologicalFromTable d
    (routeIndexedTopologicalTableFor d targets) v

private lemma routeIndexedTopologicalFromTable_eq_canonical
    (d : Graph) (table : RouteIndexedTable)
    (canonicalEntries : forall entry, entry ∈ table ->
      entry.2 = canonicalAt d entry.1) (v : Vertex) :
    routeIndexedTopologicalFromTable d table v = canonicalAt d v := by
  induction table with
  | nil => rfl
  | cons entry rest ih =>
      rcases entry with ⟨node, tokens⟩
      have headCanonical : tokens = canonicalAt d node :=
        canonicalEntries (node, tokens) (by simp)
      have restCanonical : forall entry, entry ∈ rest ->
          entry.2 = canonicalAt d entry.1 := by
        intro entry entry_mem
        exact canonicalEntries entry (by simp [entry_mem])
      by_cases node_eq : node = v
      · subst v
        simp [routeIndexedTopologicalFromTable,
          routeIndexedTableLookup?, headCanonical]
      · have node_beq : (node == v) = false := by simp [node_eq]
        simp only [routeIndexedTopologicalFromTable,
          routeIndexedTableLookup?, List.find?_cons, node_beq, Option.map]
        exact ih restCanonical

private lemma routeIndexedTopologicalFold_canonicalEntries
    (d : Graph) (fixed : RouteIndexedCanonicalFixedPoint d) :
    forall order table,
      (forall v, v ∈ order -> v ∈ d.NODES) ->
      (forall entry, entry ∈ table ->
        entry.2 = canonicalAt d entry.1) ->
      forall entry,
        entry ∈ routeIndexedTopologicalFold d order table ->
          entry.2 = canonicalAt d entry.1 := by
  intro order
  induction order with
  | nil =>
      intro table order_mem tableCanonical entry entry_mem
      exact tableCanonical entry entry_mem
  | cons v rest ih =>
      intro table order_mem tableCanonical
      simp only [routeIndexedTopologicalFold]
      apply ih
      · intro u u_mem
        exact order_mem u (by simp [u_mem])
      intro entry entry_mem
      simp only [List.mem_cons] at entry_mem
      rcases entry_mem with rfl | entry_mem
      · change stepFrom d (routeIndexedTopologicalFromTable d table) v =
          canonicalAt d v
        rw [show routeIndexedTopologicalFromTable d table = canonicalAt d from
          funext (routeIndexedTopologicalFromTable_eq_canonical
            d table tableCanonical)]
        exact fixed v (order_mem v (by simp))
      · exact tableCanonical entry entry_mem

theorem routeIndexedCanonicalTopologicalFast_eq_canonical
    (d : Graph) (fixed : RouteIndexedCanonicalFixedPoint d) (v : Vertex) :
    routeIndexedCanonicalTopologicalFast d v = canonicalAt d v := by
  unfold routeIndexedCanonicalTopologicalFast routeIndexedTopologicalTable
  apply routeIndexedTopologicalFromTable_eq_canonical
  exact routeIndexedTopologicalFold_canonicalEntries d fixed
    (routeIndexedTopologicalOrder d) []
    (by
      intro u u_mem
      exact (List.mergeSort_perm d.NODES
        (fun left right => right.LEVEL <= left.LEVEL)).mem_iff.mp u_mem)
    (by simp)

theorem routeIndexedCanonicalTopologicalFastFor_eq_canonical
    (d : Graph) (fixed : RouteIndexedCanonicalFixedPoint d)
    (targets : List Vertex) (v : Vertex) :
    routeIndexedCanonicalTopologicalFastFor d targets v = canonicalAt d v := by
  unfold routeIndexedCanonicalTopologicalFastFor
    routeIndexedTopologicalTableFor
  apply routeIndexedTopologicalFromTable_eq_canonical
  exact routeIndexedTopologicalFold_canonicalEntries d fixed
    (routeIndexedTopologicalOrderFor d targets) []
    (by
      intro u u_mem
      have sorted_mem : u ∈ routeIndexedTopologicalOrder d :=
        (List.mem_filter.mp u_mem).1
      exact (List.mergeSort_perm d.NODES
        (fun left right => right.LEVEL <= left.LEVEL)).mem_iff.mp sorted_mem)
    (by simp)

theorem routeIndexedCanonicalTopologicalFast_setEq_flowAt
    (d : Graph) (fixed : RouteIndexedCanonicalFixedPoint d) (v : Vertex) :
    ListSetEq
      (projectFlowPairs (routeIndexedCanonicalTopologicalFast d v))
      (flowAt d (stdFuel d) v) := by
  rw [routeIndexedCanonicalTopologicalFast_eq_canonical d fixed v]
  exact routeIndexedCanonical_setEq_flowAt d v

theorem routeIndexedCanonicalTopologicalFastFor_setEq_flowAt
    (d : Graph) (fixed : RouteIndexedCanonicalFixedPoint d)
    (targets : List Vertex) (v : Vertex) :
    ListSetEq
      (projectFlowPairs
        (routeIndexedCanonicalTopologicalFastFor d targets v))
      (flowAt d (stdFuel d) v) := by
  rw [routeIndexedCanonicalTopologicalFastFor_eq_canonical
    d fixed targets v]
  exact routeIndexedCanonical_setEq_flowAt d v




def routeIndexedTopologicalReadyB (d : Graph) : Bool :=
  let rec go (seen : List Vertex) : List Vertex -> Bool
    | [] => true
    | v :: rest =>
        (predsOf d v).all seen.contains && go (v :: seen) rest
  go [] (routeIndexedTopologicalOrder d)

def routeIndexedTopologicalReadyForB (d : Graph)
    (targets : List Vertex) : Bool :=
  let rec go (seen : List Vertex) : List Vertex -> Bool
    | [] => true
    | v :: rest =>
        (predsOf d v).all seen.contains && go (v :: seen) rest
  go [] (routeIndexedTopologicalOrderFor d targets)





def certificateSkeletonFastAt (skeleton : CertificateRouteSkeleton)
    (v : Vertex) : List RouteIndexedToken :=
  annotateTokens skeleton v
    (routeIndexedCanonicalFast skeleton.graph v)

def certificateSkeletonTopologicalFastAt
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    List RouteIndexedToken :=
  annotateTokens skeleton v
    (routeIndexedCanonicalTopologicalFast skeleton.graph v)




def certificateSkeletonTopologicalTable
    (skeleton : CertificateRouteSkeleton) : RouteIndexedTable :=
  (routeIndexedTopologicalTable skeleton.graph).map fun entry =>
    (entry.1, annotateTokens skeleton entry.1 entry.2)

def certificateSkeletonTopologicalTableFor
    (skeleton : CertificateRouteSkeleton) (targets : List Vertex) :
    RouteIndexedTable :=
  (routeIndexedTopologicalTableFor skeleton.graph targets).map fun entry =>
    (entry.1, annotateTokens skeleton entry.1 entry.2)

theorem certificateSkeletonTopologicalFastAt_setEq_flowAt
    (skeleton : CertificateRouteSkeleton)
    (fixed : RouteIndexedCanonicalFixedPoint skeleton.graph) (v : Vertex) :
    ListSetEq
      (projectFlowPairs (certificateSkeletonTopologicalFastAt skeleton v))
      (flowAt skeleton.graph (stdFuel skeleton.graph) v) :=
  (annotateTokens_project skeleton v _).trans
    (routeIndexedCanonicalTopologicalFast_setEq_flowAt
      skeleton.graph fixed v)









def certificateSkeletonExactFromTable
    (skeleton : CertificateRouteSkeleton) (fuel : Nat)
    (table : RouteIndexedTable) (v : Vertex) : List RouteIndexedToken :=
  match routeIndexedTableLookup? table v with
  | some tokens => tokens
  | none => certificateSkeletonAt skeleton fuel v

def certificateSkeletonExactTableSpec
    (skeleton : CertificateRouteSkeleton) (fuel : Nat) :
    RouteIndexedTable :=
  skeleton.graph.NODES.map fun v =>
    (v, certificateSkeletonAt skeleton fuel v)

private lemma certificateSkeletonExactFromTable_spec
    (skeleton : CertificateRouteSkeleton) (fuel : Nat) (v : Vertex) :
    certificateSkeletonExactFromTable skeleton fuel
        (certificateSkeletonExactTableSpec skeleton fuel) v =
      certificateSkeletonAt skeleton fuel v := by
  unfold certificateSkeletonExactTableSpec
  induction skeleton.graph.NODES with
  | nil => simp [certificateSkeletonExactFromTable,
      routeIndexedTableLookup?]
  | cons node nodes ih =>
      by_cases node_eq : node = v
      · subst v
        simp [certificateSkeletonExactFromTable,
          routeIndexedTableLookup?]
      · have node_beq : (node == v) = false := by simp [node_eq]
        simp only [List.map_cons, certificateSkeletonExactFromTable,
          routeIndexedTableLookup?, List.find?_cons, node_beq, Option.map]
        change certificateSkeletonExactFromTable skeleton fuel
          (nodes.map fun key =>
            (key, certificateSkeletonAt skeleton fuel key)) v =
              certificateSkeletonAt skeleton fuel v
        exact ih

def certificateSkeletonExactTableInitial
    (skeleton : CertificateRouteSkeleton) : RouteIndexedTable :=
  skeleton.graph.NODES.map fun v => (v, [])

def certificateSkeletonExactTableStep
    (skeleton : CertificateRouteSkeleton) (fuel : Nat)
    (previous : RouteIndexedTable) : RouteIndexedTable :=
  skeleton.graph.NODES.map fun v =>
    (v, certificateSkeletonLocalOutputs skeleton
      (certificateSkeletonExactFromTable skeleton fuel previous) v)

private lemma certificateSkeletonExactTableInitial_eq_spec_zero
    (skeleton : CertificateRouteSkeleton) :
    certificateSkeletonExactTableInitial skeleton =
      certificateSkeletonExactTableSpec skeleton 0 := by
  rfl

private lemma certificateSkeletonExactTableStep_spec
    (skeleton : CertificateRouteSkeleton) (fuel : Nat) :
    certificateSkeletonExactTableStep skeleton fuel
        (certificateSkeletonExactTableSpec skeleton fuel) =
      certificateSkeletonExactTableSpec skeleton (fuel + 1) := by
  simp only [certificateSkeletonExactTableStep,
    certificateSkeletonExactTableSpec]
  apply List.map_congr_left
  intro v v_mem
  simp only [Prod.mk.injEq, true_and]
  change certificateSkeletonLocalOutputs skeleton
      (certificateSkeletonExactFromTable skeleton fuel
        (certificateSkeletonExactTableSpec skeleton fuel)) v =
    certificateSkeletonAt skeleton (fuel + 1) v
  rw [show certificateSkeletonExactFromTable skeleton fuel
      (certificateSkeletonExactTableSpec skeleton fuel) =
        certificateSkeletonAt skeleton fuel from
      funext (certificateSkeletonExactFromTable_spec skeleton fuel)]
  rfl

def certificateSkeletonExactTableIterate
    (skeleton : CertificateRouteSkeleton) :
    Nat -> Nat -> RouteIndexedTable -> RouteIndexedTable
  | 0, _, previous => previous
  | remaining + 1, fuel, previous =>
      certificateSkeletonExactTableIterate skeleton remaining (fuel + 1)
        (certificateSkeletonExactTableStep skeleton fuel previous)

private lemma certificateSkeletonExactTableIterate_spec
    (skeleton : CertificateRouteSkeleton) : forall remaining fuel,
    certificateSkeletonExactTableIterate skeleton remaining fuel
        (certificateSkeletonExactTableSpec skeleton fuel) =
      certificateSkeletonExactTableSpec skeleton (fuel + remaining) := by
  intro remaining
  induction remaining with
  | zero => intro fuel; rfl
  | succ remaining ih =>
      intro fuel
      rw [certificateSkeletonExactTableIterate,
        certificateSkeletonExactTableStep_spec]
      have rest := ih (fuel + 1)
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using rest

def certificateSkeletonExactTableAt
    (skeleton : CertificateRouteSkeleton) (fuel : Nat) :
    RouteIndexedTable :=
  certificateSkeletonExactTableIterate skeleton fuel 0
    (certificateSkeletonExactTableInitial skeleton)

def certificateSkeletonExactTable
    (skeleton : CertificateRouteSkeleton) : RouteIndexedTable :=
  certificateSkeletonExactTableAt skeleton (stdFuel skeleton.graph)

theorem certificateSkeletonExactTableAt_eq_spec
    (skeleton : CertificateRouteSkeleton) (fuel : Nat) :
    certificateSkeletonExactTableAt skeleton fuel =
      certificateSkeletonExactTableSpec skeleton fuel := by
  unfold certificateSkeletonExactTableAt
  rw [certificateSkeletonExactTableInitial_eq_spec_zero]
  simpa using certificateSkeletonExactTableIterate_spec skeleton fuel 0

theorem certificateSkeletonExactTable_eq_spec
    (skeleton : CertificateRouteSkeleton) :
    certificateSkeletonExactTable skeleton =
      certificateSkeletonExactTableSpec skeleton
        (stdFuel skeleton.graph) :=
  certificateSkeletonExactTableAt_eq_spec skeleton _

def certificateSkeletonExactCanonicalAt
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    List RouteIndexedToken :=
  certificateSkeletonExactFromTable skeleton (stdFuel skeleton.graph)
    (certificateSkeletonExactTable skeleton) v

theorem certificateSkeletonExactCanonicalAt_eq_reference
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    certificateSkeletonExactCanonicalAt skeleton v =
      certificateSkeletonAt skeleton (stdFuel skeleton.graph) v := by
  unfold certificateSkeletonExactCanonicalAt
  rw [certificateSkeletonExactTable_eq_spec]
  exact certificateSkeletonExactFromTable_spec skeleton _ v





def certificateSkeletonTable (skeleton : CertificateRouteSkeleton) :
    RouteIndexedTable :=
  (routeIndexedTable skeleton.graph).map fun entry =>
    (entry.1, annotateTokens skeleton entry.1 entry.2)

def certificateSkeletonFromTable (skeleton : CertificateRouteSkeleton)
    (table : RouteIndexedTable) (v : Vertex) : List RouteIndexedToken :=
  match routeIndexedTableLookup? table v with
  | some tokens => tokens
  | none => certificateSkeletonFastAt skeleton v

private lemma certificateSkeletonFromTable_self
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    certificateSkeletonFromTable skeleton
        (skeleton.graph.NODES.map fun node =>
          (node, certificateSkeletonFastAt skeleton node)) v =
      certificateSkeletonFastAt skeleton v := by
  induction skeleton.graph.NODES with
  | nil => rfl
  | cons node nodes ih =>
      by_cases node_eq : node = v
      · subst v
        simp [certificateSkeletonFromTable, routeIndexedTableLookup?]
      · have node_beq : (node == v) = false := by simp [node_eq]
        simp only [List.map_cons, certificateSkeletonFromTable,
          routeIndexedTableLookup?, List.find?_cons, node_beq, Option.map]
        change certificateSkeletonFromTable skeleton
          (nodes.map fun key =>
            (key, certificateSkeletonFastAt skeleton key)) v =
              certificateSkeletonFastAt skeleton v
        exact ih

theorem certificateSkeletonTable_eq_spec
    (skeleton : CertificateRouteSkeleton) :
    certificateSkeletonTable skeleton =
      skeleton.graph.NODES.map fun v =>
        (v, certificateSkeletonFastAt skeleton v) := by
  unfold certificateSkeletonTable
  rw [routeIndexedTable_eq_spec]
  unfold routeIndexedTableSpec
  rw [List.map_map]
  apply List.map_congr_left
  intro v v_mem
  unfold certificateSkeletonFastAt
  rw [routeIndexedCanonicalFast_eq_canonical]
  rfl

def certificateSkeletonTableAt (skeleton : CertificateRouteSkeleton)
    (v : Vertex) : List RouteIndexedToken :=
  certificateSkeletonFromTable skeleton (certificateSkeletonTable skeleton) v

theorem certificateSkeletonTableAt_eq_fast
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    certificateSkeletonTableAt skeleton v = certificateSkeletonFastAt skeleton v := by
  unfold certificateSkeletonTableAt
  rw [certificateSkeletonTable_eq_spec]
  exact certificateSkeletonFromTable_self skeleton v

theorem certificateSkeletonFastAt_setEq_reference
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    ListSetEq (projectFlowPairs (certificateSkeletonFastAt skeleton v))
      (projectFlowPairs
        (certificateSkeletonAt skeleton (stdFuel skeleton.graph) v)) := by
  apply ListSetEq.trans (annotateTokens_project skeleton v _)
  rw [routeIndexedCanonicalFast_eq_canonical]
  exact (routeIndexedCanonical_setEq_flowAt skeleton.graph v).trans
    (certificateSkeletonAt_setEq_flowAt skeleton
      (stdFuel skeleton.graph) v).symm

theorem certificateSkeletonFastAt_setEq_flowAt
    (skeleton : CertificateRouteSkeleton) (v : Vertex) :
    ListSetEq (projectFlowPairs (certificateSkeletonFastAt skeleton v))
      (flowAt skeleton.graph (stdFuel skeleton.graph) v) :=
  (annotateTokens_project skeleton v _).trans
    (routeIndexedCanonicalFast_setEq_flowAt skeleton.graph v)

def certificateSkeletonCanonicalFast (c : Certificate) (v : Vertex) :
    List RouteIndexedToken :=
  certificateSkeletonExactCanonicalAt (eraseCertificateDependencies c) v

theorem certificateSkeletonCanonicalFast_eq_reference
    (c : Certificate) (v : Vertex) :
    certificateSkeletonCanonicalFast c v =
      certificateSkeletonCanonicalAt c v := by
  unfold certificateSkeletonCanonicalFast certificateSkeletonCanonicalAt
  exact certificateSkeletonExactCanonicalAt_eq_reference
    (eraseCertificateDependencies c) v

def certificateSkeletonCanonicalTopologicalFast
    (c : Certificate) (v : Vertex) : List RouteIndexedToken :=
  certificateSkeletonTopologicalFastAt (eraseCertificateDependencies c) v

theorem certificateSkeletonCanonicalTopologicalFast_setEq_flowAt
    (c : Certificate) (fixed : RouteIndexedCanonicalFixedPoint c.graph)
    (v : Vertex) :
    ListSetEq
      (projectFlowPairs
        (certificateSkeletonCanonicalTopologicalFast c v))
      (flowAt c.graph (stdFuel c.graph) v) :=
  certificateSkeletonTopologicalFastAt_setEq_flowAt
    (eraseCertificateDependencies c) fixed v

theorem certificateSkeletonCanonicalTopologicalFastWitnessable_setEq
    (c : Certificate) (fixed : RouteIndexedCanonicalFixedPoint c.graph)
    (v : Vertex) :
    ListSetEq
      (projectFlowPairs
        (witnessableAt c.graph v
          (certificateSkeletonCanonicalTopologicalFast c v)))
      (WitnessableRoutes c.graph v
        (flowAt c.graph (stdFuel c.graph) v)) :=
  (project_witnessableAt c.graph v _).trans
    (witnessableRoutes_congr c.graph v
      (certificateSkeletonCanonicalTopologicalFast_setEq_flowAt c fixed v))

theorem certificateSkeletonCanonicalFast_setEq_reference
    (c : Certificate) (v : Vertex) :
    ListSetEq
      (projectFlowPairs (certificateSkeletonCanonicalFast c v))
      (projectFlowPairs (certificateSkeletonCanonicalAt c v)) := by
  rw [certificateSkeletonCanonicalFast_eq_reference]
  exact ListSetEq.refl _

theorem certificateSkeletonCanonicalFast_setEq_flowAt
    (c : Certificate) (v : Vertex) :
    ListSetEq
      (projectFlowPairs (certificateSkeletonCanonicalFast c v))
      (flowAt c.graph (stdFuel c.graph) v) := by
  rw [certificateSkeletonCanonicalFast_eq_reference]
  exact certificateSkeletonCanonical_setEq_flowAt c v

theorem certificateSkeletonCanonicalFastWitnessable_setEq
    (c : Certificate) (v : Vertex) :
    ListSetEq
      (projectFlowPairs
        (witnessableAt c.graph v (certificateSkeletonCanonicalFast c v)))
      (WitnessableRoutes c.graph v
        (flowAt c.graph (stdFuel c.graph) v)) := by
  rw [certificateSkeletonCanonicalFast_eq_reference]
  exact certificateSkeletonCanonicalWitnessable_setEq c v

def CertificateSkeletonFastDischarged (d : Graph) (c : Certificate) : Prop :=
  forall root, root ∈ d.NODES ->
    (get_rule.outgoing root d).isEmpty = true ->
      projectFlowPairs (certificateSkeletonCanonicalFast c root) ≠ [] /\
      forall route,
        route ∈ projectFlowPairs (certificateSkeletonCanonicalFast c root) ->
          route.1 = [] /\ route.2 = []

theorem certificateSkeletonFastDischarged_of_reference
    (d : Graph) (c : Certificate)
    (reference : CertificateSkeletonDischarged d c) :
    CertificateSkeletonFastDischarged d c := by
  intro root root_mem root_is_root
  have reference_root := reference root root_mem root_is_root
  have correspondence := certificateSkeletonCanonicalFast_setEq_reference c root
  constructor
  · intro fast_empty
    apply reference_root.1
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro route route_mem
    have fast_mem := (correspondence route).mpr route_mem
    simp [fast_empty] at fast_mem
  · intro route route_mem
    exact reference_root.2 route ((correspondence route).mp route_mem)

theorem certificateSkeletonFastDischarged_of_strongFormation
    (d : Graph) (c : Certificate) (valid : StrongFormationValid d c) :
    CertificateSkeletonFastDischarged d c :=
  certificateSkeletonFastDischarged_of_reference d c
    (certificateSkeletonDischarged_of_strongFormation d c valid)

def CertificateSkeletonTopologicalFastDischarged
    (d : Graph) (c : Certificate) : Prop :=
  forall root, root ∈ d.NODES ->
    (get_rule.outgoing root d).isEmpty = true ->
      projectFlowPairs
          (certificateSkeletonCanonicalTopologicalFast c root) ≠ [] /\
      forall route,
        route ∈ projectFlowPairs
          (certificateSkeletonCanonicalTopologicalFast c root) ->
            route.1 = [] /\ route.2 = []

theorem certificateSkeletonTopologicalFastDischarged_of_strongFormation
    (d : Graph) (c : Certificate) (valid : StrongFormationValid d c)
    (fixed : RouteIndexedCanonicalFixedPoint d) :
    CertificateSkeletonTopologicalFastDischarged d c := by
  have graph_eq : c.graph = d := valid.1.1
  intro root root_mem root_is_root
  have reference :=
    certificateSkeletonDischarged_of_strongFormation d c valid
      root root_mem root_is_root
  have topologicalFlow : ListSetEq
      (projectFlowPairs
        (certificateSkeletonCanonicalTopologicalFast c root))
      (flowAt d (stdFuel d) root) := by
    simpa [graph_eq] using
      certificateSkeletonCanonicalTopologicalFast_setEq_flowAt c
        (by simpa [graph_eq] using fixed) root
  have referenceFlow : ListSetEq
      (projectFlowPairs (certificateSkeletonCanonicalAt c root))
      (flowAt d (stdFuel d) root) := by
    simpa [graph_eq] using
      certificateSkeletonCanonical_setEq_flowAt c root
  have correspondence := topologicalFlow.trans referenceFlow.symm
  constructor
  · intro topological_empty
    apply reference.1
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro route route_mem
    have topological_mem := (correspondence route).mpr route_mem
    simp [topological_empty] at topological_mem
  · intro route route_mem
    exact reference.2 route ((correspondence route).mp route_mem)


end RouteIndexedFastEvaluator
