import Certificate.Flow.Cache
import ProofSearch.FlowCorrespondence

open DLDS FlowSemantics Certificate.Flow.Foundation Certificate.Flow
open ProofSearch.FlowCorrespondence
open DLDS.Certificate.Flow.Cache
open Evaluator.RouteIndexed

namespace Certificate.Flow.GeneratorCompleteness

/-!
# Completeness of the canonical Flow certificate generator

The source-side hypothesis below is deliberately stated entirely in terms of
the graph and raw Flow.  In particular, it does not mention the generated
certificate or any certificate checker.
-/

/-- Graph-side obligations which are independent of universal raw-Flow root
discharge but are required by the current local certificate format. -/
structure SuitableGraph (d : Graph) : Prop where
  structural : structuralValid d
  localShapesValid : ∀ v ∈ d.NODES, localShape d v ≠ .invalid
  edgesNodup : d.EDGES.Nodup
  oneEdgePerColour : OneEdgePerColourPerNode d
  routeHeadCoverage : CursorRouteHeadCoverage d
  flowTailClosure : CursorFlowTailClosure d
  levelFormulaUnique : LevelFormulaUnique d
  hypothesesHaveNoIncoming : HypothesesHaveNoIncoming d
  rootDecoration : RootDischarged d

/-- The cached generator is extensionally the simple specification which reads
`flowAt d (stdFuel d)` once for each edge source and root. -/
theorem generateCertificate_eq_raw (d : Graph) :
    generateCertificate d =
      { graph := d
        edgeRoutes := d.EDGES.flatMap (witnessForEdge? d)
        terminalOutputs := d.NODES.flatMap fun v =>
          if (get_rule.outgoing v d).isEmpty then
            (routesAt d v).map fun route =>
              { node := v, dependency := route.1, residual := route.2 }
          else [] } := by
  unfold generateCertificate
  dsimp only
  unfold generateCertificateFromFlowTable
  dsimp only
  have routes_eq : routesAtCached d (flowTable d) = routesAt d := by
    funext v
    exact cachedFlowAt_flowTable d v
  rw [routes_eq]
  rfl

@[simp] theorem generateCertificate_graph (d : Graph) :
    (generateCertificate d).graph = d := by
  rw [generateCertificate_eq_raw]

theorem mem_generated_edgeRoutes_iff (d : Graph)
    (witness : EdgeRouteWitness) :
    witness ∈ (generateCertificate d).edgeRoutes ↔
      ∃ edge ∈ d.EDGES,
        witness.edge = edge ∧
        (witness.dependency, witness.residual) ∈
          flowAt d (stdFuel d) edge.START ∧
        headColour witness.residual = edge.COLOUR := by
  rw [generateCertificate_eq_raw]
  simp only [witnessForEdge?, witnessForEdgeFrom, routesAt,
    List.mem_flatMap, List.mem_map, List.mem_filter]
  constructor
  · rintro ⟨edge, hedge, route, ⟨hroute, hcolour⟩, rfl⟩
    exact ⟨edge, hedge, rfl, hroute, by simpa using hcolour⟩
  · rintro ⟨candidate, hcand, hw, hroute, hcolour⟩
    subst candidate
    exact ⟨witness.edge, hcand,
      (witness.dependency, witness.residual),
      ⟨hroute, by simp [hcolour]⟩, by cases witness ; rfl⟩

theorem mem_generated_terminalOutputs_iff (d : Graph)
    (output : TerminalOutput) :
    output ∈ (generateCertificate d).terminalOutputs ↔
      output.node ∈ d.NODES ∧
      (get_rule.outgoing output.node d).isEmpty = true ∧
      (output.dependency, output.residual) ∈
        flowAt d (stdFuel d) output.node := by
  rw [generateCertificate_eq_raw]
  simp only [routesAt, List.mem_flatMap]
  constructor
  · rintro ⟨vertex, hvertex, generated⟩
    split at generated <;> rename_i hroot
    · rw [List.mem_map] at generated
      obtain ⟨route, hroute, rfl⟩ := generated
      exact ⟨hvertex, hroot, hroute⟩
    · simp at generated
  · rintro ⟨hnode, hroot, hroute⟩
    refine ⟨output.node, hnode, ?_⟩
    simp only [hroot, if_true, List.mem_map]
    exact ⟨(output.dependency, output.residual), hroute,
      by cases output ; rfl⟩

theorem mem_generated_edgeWitnessesFor_iff (d : Graph)
    (edge : Deduction) (witness : EdgeRouteWitness) :
    witness ∈ edgeWitnessesFor (generateCertificate d) edge ↔
      edge ∈ d.EDGES ∧ witness.edge = edge ∧
      (witness.dependency, witness.residual) ∈
        flowAt d (stdFuel d) edge.START ∧
      headColour witness.residual = edge.COLOUR := by
  unfold edgeWitnessesFor
  rw [List.mem_filter, mem_generated_edgeRoutes_iff]
  constructor
  · rintro ⟨⟨candidate, hcand, hw, hroute, hcolour⟩, heq⟩
    have hw' : witness.edge = edge := by simpa using heq
    have candidateEq : candidate = edge := hw.symm.trans hw'
    exact ⟨by simpa [candidateEq] using hcand, hw',
      by simpa [candidateEq] using hroute,
      by simpa [candidateEq] using hcolour⟩
  · rintro ⟨hedge, hw, hroute, hcolour⟩
    refine ⟨⟨edge, hedge, hw, ?_, ?_⟩, by simp [hw]⟩
    · simpa [hw] using hroute
    · simpa [hw] using hcolour

theorem generated_edgeWitnessesFor_setEq (d : Graph) (edge : Deduction)
    (hedge : edge ∈ d.EDGES) :
    ListSetEq
      ((edgeWitnessesFor (generateCertificate d) edge).map fun witness =>
        (witness.dependency, witness.residual))
      ((flowAt d (stdFuel d) edge.START).filter fun route =>
        headColour route.2 == edge.COLOUR) := by
  intro route
  constructor
  · intro member
    rw [List.mem_map] at member
    obtain ⟨witness, witnessMember, rfl⟩ := member
    obtain ⟨_, _, routeMember, colour⟩ :=
      (mem_generated_edgeWitnessesFor_iff d edge witness).mp witnessMember
    rw [List.mem_filter]
    exact ⟨routeMember, by simpa using colour⟩
  · intro member
    rw [List.mem_filter] at member
    refine List.mem_map.mpr ⟨
      { edge := edge, dependency := route.1, residual := route.2 }, ?_, rfl⟩
    apply (mem_generated_edgeWitnessesFor_iff d edge _).mpr
    exact ⟨hedge, rfl, member.1, by simpa using member.2⟩

theorem mem_generated_storedRoutes_iff (d : Graph) (v : Vertex)
    (route : FlowPair) :
    route ∈ StoredRoutes (generateCertificate d) v ↔
      (∃ edge ∈ d.EDGES, edge.START = v ∧
        route ∈ flowAt d (stdFuel d) v ∧
        headColour route.2 = edge.COLOUR) ∨
      (v ∈ d.NODES ∧ (get_rule.outgoing v d).isEmpty = true ∧
        route ∈ flowAt d (stdFuel d) v) := by
  unfold StoredRoutes outputsAt
  rw [List.mem_eraseDups, List.mem_append]
  constructor
  · rintro (edgeOutput | terminalOutput)
    · rw [List.mem_filterMap] at edgeOutput
      obtain ⟨witness, witnessMember, emitted⟩ := edgeOutput
      by_cases start : witness.edge.START = v
      · simp only [start, beq_self_eq_true, if_true,
          Option.some.injEq] at emitted
        subst route
        obtain ⟨edge, edgeMember, witnessEdge, routeMember, colour⟩ :=
          (mem_generated_edgeRoutes_iff d witness).mp witnessMember
        left
        have edgeStart : edge.START = v := by
          simpa [witnessEdge] using start
        refine ⟨edge, edgeMember, edgeStart, ?_, colour⟩
        simpa [edgeStart] using routeMember
      · have startB : (witness.edge.START == v) = false := by
          simpa using start
        simp [startB] at emitted
    · rw [List.mem_filterMap] at terminalOutput
      obtain ⟨output, outputMember, emitted⟩ := terminalOutput
      by_cases node : output.node = v
      · simp only [node, beq_self_eq_true, if_true,
          Option.some.injEq] at emitted
        subst route
        obtain ⟨nodeMember, root, routeMember⟩ :=
          (mem_generated_terminalOutputs_iff d output).mp outputMember
        right
        exact ⟨by simpa [node] using nodeMember,
          by simpa [node] using root, by simpa [node] using routeMember⟩
      · have nodeB : (output.node == v) = false := by simpa using node
        simp [nodeB] at emitted
  · rintro (edgeOutput | terminalOutput)
    · obtain ⟨edge, edgeMember, start, routeMember, colour⟩ := edgeOutput
      left
      rw [List.mem_filterMap]
      let witness : EdgeRouteWitness :=
        { edge, dependency := route.1, residual := route.2 }
      refine ⟨witness, ?_, ?_⟩
      · apply (mem_generated_edgeRoutes_iff d witness).mpr
        exact ⟨edge, edgeMember, rfl, by simpa [witness, start] using routeMember,
          by simpa [witness] using colour⟩
      · simp [witness, start]
    · obtain ⟨nodeMember, root, routeMember⟩ := terminalOutput
      right
      rw [List.mem_filterMap]
      let output : TerminalOutput :=
        { node := v, dependency := route.1, residual := route.2 }
      refine ⟨output, ?_, ?_⟩
      · apply (mem_generated_terminalOutputs_iff d output).mpr
        exact ⟨by simpa [output] using nodeMember,
          by simpa [output] using root,
          by simpa [output] using routeMember⟩
      · simp [output]

/-- Generation stores exactly the witnessable projection of raw Flow.  The
root hypothesis is needed because the generator stores every raw root route,
whereas `WitnessableRoutes` retains only root routes with empty residual. -/
theorem generatedStoredRoutes_setEq_flow (d : Graph)
    (roots : rootsDischargedB d = true) :
    ∀ v ∈ d.NODES,
      ListSetEq (StoredRoutes (generateCertificate d) v)
        (WitnessableRoutes d v (flowAt d (stdFuel d) v)) := by
  unfold rootsDischargedB at roots
  rw [List.all_eq_true] at roots
  intro v vMember route
  rw [mem_generated_storedRoutes_iff]
  unfold WitnessableRoutes
  rw [List.mem_filter]
  constructor
  · rintro (edgeRoute | terminalRoute)
    · obtain ⟨edge, edgeMember, edgeStart, routeMember, colour⟩ := edgeRoute
      refine ⟨routeMember, ?_⟩
      have outgoingMember : edge ∈ get_rule.outgoing v d :=
        mem_outgoing_of_mem_edges_start_eq v d edgeMember edgeStart
      exact routeContinuableAtB_of_consume_outgoing d v edge route.2
        route.2.tail outgoingMember (consume_headColour colour.symm)
    · obtain ⟨_, root, routeMember⟩ := terminalRoute
      have rootCheck := roots v vMember
      simp only [root, Bool.not_true, Bool.false_or, Bool.and_eq_true,
        List.all_eq_true] at rootCheck
      have empty := rootCheck.2 route routeMember
      refine ⟨routeMember, ?_⟩
      simp [routeContinuableAtB, root, List.isEmpty_iff.mp empty.2]
  · rintro ⟨routeMember, continuable⟩
    by_cases root : (get_rule.outgoing v d).isEmpty = true
    · right
      exact ⟨vMember, root, routeMember⟩
    · left
      have nonempty : (get_rule.outgoing v d).isEmpty = false :=
        Bool.eq_false_of_not_eq_true root
      unfold routeContinuableAtB at continuable
      simp only [nonempty, Bool.false_eq_true, if_false,
        List.any_eq_true] at continuable
      obtain ⟨edge, edgeOutgoing, consumes⟩ := continuable
      rw [Option.isSome_iff_exists] at consumes
      obtain ⟨tail, consumed⟩ := consumes
      exact ⟨edge, mem_outgoing_mem_edges v d edgeOutgoing,
        mem_outgoing_start_eq v d edgeOutgoing, routeMember,
        (consume_some_colour consumed).symm⟩

/-- Every graph edge receives a generated Flow witness. -/
theorem generateCertificate_edge_covered (d : Graph)
    (suitable : SuitableGraph d) (edge : Deduction)
    (edgeMember : edge ∈ d.EDGES) :
    edgeWitnessesFor (generateCertificate d) edge ≠ [] := by
  have startMember : edge.START ∈ d.NODES :=
    (suitable.structural.2.1 edge edgeMember).1
  have outgoingMember : edge ∈ get_rule.outgoing edge.START d :=
    mem_outgoing_of_mem_edges_start_eq edge.START d edgeMember rfl
  have outgoingNonempty : get_rule.outgoing edge.START d ≠ [] := by
    intro empty
    rw [empty] at outgoingMember
    exact List.not_mem_nil outgoingMember
  obtain ⟨_, edgeToRoute⟩ := suitable.routeHeadCoverage edge.START
    startMember outgoingNonempty
  obtain ⟨route, routeMember, colour⟩ := edgeToRoute edge outgoingMember
  intro witnessesEmpty
  have witnessMember :
      ({ edge := edge, dependency := route.1, residual := route.2 } :
        EdgeRouteWitness) ∈ edgeWitnessesFor (generateCertificate d) edge := by
    apply (mem_generated_edgeWitnessesFor_iff d edge _).mpr
    exact ⟨edgeMember, rfl, routeMember, colour⟩
  rw [witnessesEmpty] at witnessMember
  exact List.not_mem_nil witnessMember

theorem generateCertificate_terminal_covered (d : Graph)
    (roots : rootsDischargedB d = true) (v : Vertex)
    (vMember : v ∈ d.NODES)
    (root : (get_rule.outgoing v d).isEmpty = true) :
    ((generateCertificate d).terminalOutputs.filter fun output =>
      output.node == v) ≠ [] := by
  unfold rootsDischargedB at roots
  rw [List.all_eq_true] at roots
  have rootCheck := roots v vMember
  simp only [root, Bool.not_true, Bool.false_or, Bool.and_eq_true] at rootCheck
  have flowNonempty : flowAt d (stdFuel d) v ≠ [] := by
    simpa [List.isEmpty_iff] using rootCheck.1
  obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ flowNonempty
  intro outputsEmpty
  let output : TerminalOutput :=
    { node := v, dependency := route.1, residual := route.2 }
  have outputMember : output ∈ (generateCertificate d).terminalOutputs :=
    (mem_generated_terminalOutputs_iff d output).mpr
      ⟨by simpa [output] using vMember, by simpa [output] using root,
        by simpa [output] using routeMember⟩
  have filtered : output ∈
      (generateCertificate d).terminalOutputs.filter fun candidate =>
        candidate.node == v := by
    rw [List.mem_filter]
    exact ⟨outputMember, by simp [output]⟩
  rw [outputsEmpty] at filtered
  exact List.not_mem_nil filtered

theorem generateCertificate_expanded (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    ExpandedCert d (generateCertificate d) := by
  refine ⟨generateCertificate_graph d, ?_, ?_, ?_, ?_⟩
  · exact generateCertificate_edge_covered d suitable
  · intro witness witnessMember
    obtain ⟨edge, edgeMember, witnessEdge, _⟩ :=
      (mem_generated_edgeRoutes_iff d witness).mp witnessMember
    simpa [witnessEdge] using edgeMember
  · exact generateCertificate_terminal_covered d roots
  · intro output outputMember
    obtain ⟨nodeMember, root, _⟩ :=
      (mem_generated_terminalOutputs_iff d output).mp outputMember
    exact ⟨nodeMember, root⟩

private theorem localExpected_setEq_flowStepFrom_stored
    (c : Certificate) (v : Vertex) (vMember : v ∈ c.graph.NODES)
    (structural : structuralValid c.graph)
    (validShape : localShape c.graph v ≠ .invalid) :
    ListSetEq (localExpected c v)
      (flowStepFrom c.graph (StoredRoutes c) v) := by
  have _registeredVertex : v ∈ c.graph.NODES := vMember
  unfold StoredRoutes
  cases shape : localShape c.graph v with
  | source =>
      have guard := localShape_source_inv shape
      intro route
      simp [localExpected, shape, flowStepFrom, guard, sourceExpected]
  | elim major minor =>
      obtain ⟨guard, pairs⟩ := localShape_elim_raw_inv shape
      intro route
      simp [localExpected, shape, flowStepFrom, guard, pairs, elimExpected, mixedIntroExpected, reseedExpected, FlowSemantics.elimCombine, List.mem_eraseDups, List.mem_append]
      rfl
  | elimMulti pairs =>
      obtain ⟨guard, pairsEq, pairsNe⟩ :=
        localShape_elimMulti_raw_inv shape
      have pairsEmpty : pairs.isEmpty = false := by
        simpa [List.isEmpty_iff] using pairsNe
      intro route
      simp [localExpected, shape, flowStepFrom, guard, pairsEq, pairsEmpty, elimExpectedMany, mixedIntroExpected, reseedExpected, FlowSemantics.elimCombine, List.mem_eraseDups, List.mem_append]
      rfl
  | intro premise alpha beta =>
      obtain ⟨preds, formula, premiseFormula, _, _, _, noElim⟩ :=
        localShape_intro_inv structural shape
      have hypothesisFalse : v.HYPOTHESIS = false :=
        (guard_false_parts (localShape_intro_raw_inv shape).1).1
      intro route
      simp [localExpected, shape, flowStepFrom, hypothesisFalse, preds,
        formula, premiseFormula, noElim, introExpected, reseedExpected,
        List.mem_eraseDups, List.mem_append]
  | invalid => exact (validShape shape).elim

/-- A local Flow step only consumes premise routes which are witnessable at
their premise vertex.  Therefore pre-filtering every premise does not change
the produced set. -/
private theorem flowStepFrom_project_inputs (d : Graph)
    (premiseFlow : Vertex → List FlowPair) (v : Vertex) :
    ListSetEq
      (flowStepFrom d
        (fun u => WitnessableRoutes d u (premiseFlow u)) v)
      (flowStepFrom d premiseFlow v) := by
  unfold flowStepFrom
  by_cases guard :
      (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp only [guard, if_true]
    exact Certificate.Flow.ListSetEq.refl _
  · have guardFalse :
        (v.HYPOTHESIS || (predsOf d v).isEmpty) = false :=
      Bool.eq_false_of_not_eq_true guard
    simp only [guardFalse, Bool.false_eq_true, if_false]
    by_cases pairsEmpty : (elimPairsAt d v).isEmpty = true
    · simp only [pairsEmpty, if_true]
      apply eraseDups_congr
      apply append_congr
      · cases predecessors : predsOf d v with
        | nil => exact Certificate.Flow.ListSetEq.refl []
        | cons premise rest =>
          cases rest with
          | cons next tail => exact Certificate.Flow.ListSetEq.refl []
          | nil =>
            cases formula : v.FORMULA with
            | atom symbol => exact Certificate.Flow.ListSetEq.refl []
            | implication alpha beta =>
              by_cases consequent : premise.FORMULA = beta
              · simp only [consequent, if_true]
                exact introProjectInputsRaw d v premise alpha
                  (premiseFlow premise)
              · simp only [consequent, if_false]
                exact Certificate.Flow.ListSetEq.refl []
      · exact reseedProjectInputs d v premiseFlow
    · have pairsNonempty : (elimPairsAt d v).isEmpty = false :=
        Bool.eq_false_of_not_eq_true pairsEmpty
      simp only [pairsNonempty, Bool.false_eq_true, if_false]
      apply eraseDups_congr
      apply append_congr
      · apply append_congr
        · apply flatMap_congr_fun
          intro pair
          exact elimCombine_project_inputs d v pair.1 pair.2
            (premiseFlow pair.2) (premiseFlow pair.1)
        · exact mixedIntroProjectInputs d v premiseFlow
      · exact reseedProjectInputs d v premiseFlow

/-- The generator's recursively interpreted local production is raw Flow.
This is proved directly from generation and the Flow step equation, not by
invoking certificate soundness in reverse. -/
theorem generatedLocalExpected_setEq_flow (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    ∀ v ∈ d.NODES,
      ListSetEq (localExpected (generateCertificate d) v)
        (flowAt d (stdFuel d) v) := by
  intro v vMember
  have localStep := localExpected_setEq_flowStepFrom_stored
    (generateCertificate d) v (by simpa using vMember)
    (by simpa using suitable.structural)
    (by simpa using suitable.localShapesValid v vMember)
  let prior : Vertex → List FlowPair := fun u =>
    WitnessableRoutes d u (flowAt d (stdFuel d - 1) u)
  let aligned : Vertex → List FlowPair := fun u =>
    if u ∈ predsOf d v then prior u
    else StoredRoutes (generateCertificate d) u
  have storedAligned : ∀ u,
      ListSetEq (StoredRoutes (generateCertificate d) u) (aligned u) := by
    intro u
    by_cases predecessor : u ∈ predsOf d v
    · have uMember := mem_predsOf_mem_nodes suitable.structural predecessor
      obtain ⟨_level, outgoing⟩ :=
        predsOf_props suitable.structural predecessor
      have generated := generatedStoredRoutes_setEq_flow d roots u uMember
      have stable : flowAt d (stdFuel d) u =
          flowAt d (stdFuel d - 1) u :=
        flowAt_stdFuel_pred d suitable.structural uMember outgoing
      simpa [aligned, prior, predecessor, stable] using generated
    · simp only [aligned, predecessor, if_false]
      exact Certificate.Flow.ListSetEq.refl _
  have stepAligned : ListSetEq
      (flowStepFrom d (StoredRoutes (generateCertificate d)) v)
      (flowStepFrom d aligned v) :=
    flowStepFrom_congr_set d v _ _ storedAligned
  have alignedEq : flowStepFrom d aligned v = flowStepFrom d prior v := by
    apply DLDS.Certificate.Flow.Cache.flowStepFrom_congr_of_preds
    intro u predecessor
    simp [aligned, predecessor]
  have projected : ListSetEq (flowStepFrom d prior v)
      (flowStepFrom d (flowAt d (stdFuel d - 1)) v) := by
    simpa [prior] using
      flowStepFrom_project_inputs d (flowAt d (stdFuel d - 1)) v
  have fuel : stdFuel d - 1 + 1 = stdFuel d := by
    unfold stdFuel
    omega
  apply localStep.trans
  apply stepAligned.trans
  rw [alignedEq]
  apply projected.trans
  rw [flowStepFrom_flowAt, fuel]
  exact Certificate.Flow.ListSetEq.refl _

private lemma setEqB_complete {alpha : Type} [BEq alpha] [LawfulBEq alpha]
    {left right : List alpha} (same : ListSetEq left right) :
    setEqB left right = true := by
  unfold setEqB
  rw [Bool.and_eq_true]
  constructor
  · rw [List.all_eq_true]
    intro value member
    simpa using (same value).mp member
  · rw [List.all_eq_true]
    intro value member
    simpa using (same value).mpr member

theorem generateCertificate_production (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    LocalProduction (generateCertificate d) := by
  intro v vMember
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have storedFlow := generatedStoredRoutes_setEq_flow d roots v vMemberD
  have localFlow := generatedLocalExpected_setEq_flow d suitable roots v vMemberD
  have expectedFlow : ListSetEq
      (witnessableExpected (generateCertificate d) v)
      (WitnessableRoutes d v (flowAt d (stdFuel d) v)) := by
    unfold witnessableExpected
    simpa using witnessableRoutes_congr d v localFlow
  have production : ListSetEq
      (StoredRoutes (generateCertificate d) v)
      (witnessableExpected (generateCertificate d) v) :=
    storedFlow.trans expectedFlow.symm
  unfold productionAtB
  cases shape : localShape (generateCertificate d).graph v with
  | invalid =>
      have valid := suitable.localShapesValid v vMemberD
      exact (valid (by simpa using shape)).elim
  | source => exact setEqB_complete production
  | elim major minor => exact setEqB_complete production
  | elimMulti pairs => exact setEqB_complete production
  | intro premise alpha beta => exact setEqB_complete production

theorem generated_edge_projection_setEq_stored_filter (d : Graph)
    (roots : rootsDischargedB d = true) (v : Vertex)
    (vMember : v ∈ d.NODES) (edge : Deduction)
    (edgeOutgoing : edge ∈ get_rule.outgoing v d) :
    ListSetEq
      ((edgeWitnessesFor (generateCertificate d) edge).map fun witness =>
        (witness.dependency, witness.residual))
      ((StoredRoutes (generateCertificate d) v).filter fun route =>
        headColour route.2 == edge.COLOUR) := by
  have edgeMember := mem_outgoing_mem_edges v d edgeOutgoing
  have edgeStart := mem_outgoing_start_eq v d edgeOutgoing
  have generated := generated_edgeWitnessesFor_setEq d edge edgeMember
  have stored := generatedStoredRoutes_setEq_flow d roots v vMember
  intro route
  rw [generated route, List.mem_filter, List.mem_filter]
  constructor
  · rintro ⟨flowMember, colour⟩
    refine ⟨(stored route).mpr ?_, colour⟩
    unfold WitnessableRoutes
    rw [List.mem_filter]
    have flowMemberV : route ∈ flowAt d (stdFuel d) v := by
      simpa [edgeStart] using flowMember
    refine ⟨flowMemberV, ?_⟩
    have headEq : headColour route.2 = edge.COLOUR := by
      simpa only [beq_iff_eq] using colour
    exact routeContinuableAtB_of_consume_outgoing d v edge route.2
      route.2.tail edgeOutgoing (consume_headColour headEq.symm)
  · rintro ⟨storedMember, colour⟩
    have witnessable := (stored route).mp storedMember
    unfold WitnessableRoutes at witnessable
    rw [List.mem_filter] at witnessable
    exact ⟨by simpa [edgeStart] using witnessable.1, colour⟩

theorem generateCertificate_coverage (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    LocalCoverage (generateCertificate d) := by
  intro v vMember edge edgeOutgoing
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have edgeOutgoingD : edge ∈ get_rule.outgoing v d := by
    simpa using edgeOutgoing
  constructor
  · exact generateCertificate_edge_covered d suitable edge
      (mem_outgoing_mem_edges v d edgeOutgoingD)
  · simpa using generated_edge_projection_setEq_stored_filter d roots
      v vMemberD edge edgeOutgoingD

theorem generateCertificate_witnessFaithful (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    LocalWitnessFaithful (generateCertificate d) := by
  intro witness witnessMember
  obtain ⟨edge, edgeMember, witnessEdge, routeMember, colour⟩ :=
    (mem_generated_edgeRoutes_iff d witness).mp witnessMember
  have startMember : witness.edge.START ∈ d.NODES := by
    simpa [witnessEdge] using (suitable.structural.2.1 edge edgeMember).1
  have localMember : (witness.dependency, witness.residual) ∈
      localExpected (generateCertificate d) witness.edge.START :=
    (generatedLocalExpected_setEq_flow d suitable roots witness.edge.START
      startMember _).mpr (by simpa [witnessEdge] using routeMember)
  unfold witnessFaithfulAtB
  rw [Bool.or_eq_true]
  right
  rw [List.any_eq_true]
  refine ⟨(witness.dependency, witness.residual), ?_, ?_⟩
  · rw [List.mem_filter]
    exact ⟨localMember, by simp⟩
  · simp [depSetEq]

theorem generateCertificate_rootDischarge (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    LocalRootDischarge (generateCertificate d) := by
  unfold rootsDischargedB at roots
  rw [List.all_eq_true] at roots
  intro v vMember root
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have rootD : (get_rule.outgoing v d).isEmpty = true := by simpa using root
  have rootCheck := roots v vMemberD
  simp only [rootD, Bool.not_true, Bool.false_or, Bool.and_eq_true,
    List.all_eq_true] at rootCheck
  have flowNonempty : flowAt d (stdFuel d) v ≠ [] := by
    simpa [List.isEmpty_iff] using rootCheck.1
  obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ flowNonempty
  obtain ⟨dependencyEmpty, residualEmpty⟩ := rootCheck.2 route routeMember
  let output : TerminalOutput :=
    { node := v, dependency := route.1, residual := route.2 }
  have outputMember : output ∈ (generateCertificate d).terminalOutputs :=
    (mem_generated_terminalOutputs_iff d output).mpr
      ⟨by simpa [output] using vMemberD, by simpa [output] using rootD,
        by simpa [output] using routeMember⟩
  have decorated := suitable.rootDecoration v vMemberD
    (List.isEmpty_iff.mp rootD)
  refine ⟨⟨output, outputMember, by simp [output], ?_, ?_⟩,
    decorated.1, decorated.2⟩
  · simpa [output] using dependencyEmpty
  · simpa [output] using residualEmpty

theorem generateCertificate_universalRootDischarge (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    UniversalLocalRootDischarge (generateCertificate d) := by
  unfold rootsDischargedB at roots
  rw [List.all_eq_true] at roots
  intro v vMember root
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have rootD : (get_rule.outgoing v d).isEmpty = true := by simpa using root
  have rootCheck := roots v vMemberD
  simp only [rootD, Bool.not_true, Bool.false_or, Bool.and_eq_true,
    List.all_eq_true] at rootCheck
  constructor
  · intro route storedMember
    have witnessable :=
      (generatedStoredRoutes_setEq_flow d (by
        unfold rootsDischargedB
        rw [List.all_eq_true]
        exact roots) v vMemberD route).mp storedMember
    unfold WitnessableRoutes at witnessable
    rw [List.mem_filter] at witnessable
    have empty := rootCheck.2 route witnessable.1
    exact ⟨List.isEmpty_iff.mp empty.1, List.isEmpty_iff.mp empty.2⟩
  · intro route localMember
    have flowMember :=
      (generatedLocalExpected_setEq_flow d suitable (by
        unfold rootsDischargedB
        rw [List.all_eq_true]
        exact roots) v vMemberD route).mp localMember
    exact List.isEmpty_iff.mp (rootCheck.2 route flowMember).2

private lemma not_exhausted_of_continuable (d : Graph) (v : Vertex)
    (residual : ColourPath)
    (exhausted : ExhaustedConsumedTailAt d v residual)
    (continuable : RouteContinuableAt d v residual) : False := by
  obtain ⟨outgoingNonempty, residualEmpty, noZero⟩ := exhausted
  subst residual
  unfold RouteContinuableAt routeContinuableAtB at continuable
  have outgoingNotEmpty : (get_rule.outgoing v d).isEmpty = false := by
    cases empty : (get_rule.outgoing v d).isEmpty with
    | false => rfl
    | true =>
      rw [List.isEmpty_iff] at empty
      exact (outgoingNonempty empty).elim
  simp only [outgoingNotEmpty, Bool.false_eq_true, if_false,
    List.any_eq_true] at continuable
  obtain ⟨edge, edgeMember, consumes⟩ := continuable
  rw [Option.isSome_iff_exists] at consumes
  obtain ⟨tail, consumed⟩ := consumes
  have colour : edge.COLOUR = 0 := by
    simpa [headColour] using consume_some_colour consumed
  obtain ⟨found, foundAt⟩ := edgeOfColour_some_of_cover
    ⟨edge, edgeMember, colour⟩
  rw [noZero] at foundAt
  contradiction

private theorem generated_carrierAccounted (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true)
    (expanded : ExpandedCert d (generateCertificate d))
    {w u : Vertex} (wMember : w ∈ d.NODES)
    (cursor : CursorCarrierAccountedAt d w u) :
    LocalCarrierAccountedAt (generateCertificate d) w u := by
  obtain ⟨alpha, formula, accounted⟩ := cursor
  refine ⟨alpha, by simpa using formula, ?_⟩
  intro tail tailMember
  unfold continuableConsumedWitnessTails at tailMember
  rw [List.mem_filter] at tailMember
  obtain ⟨edge, edgeBetween, witness, witnessMember, witnessEdge, consumed⟩ :=
    consumedWitnessTails_exists d (generateCertificate d) expanded tailMember.1
  obtain ⟨sourceEdge, _sourceEdgeMember, generatedEdge,
      routeMember, _colour⟩ :=
    (mem_generated_edgeRoutes_iff d witness).mp witnessMember
  have edgeStart : edge.START = u := (mem_edgesBetween edgeBetween).1
  have sourceEdgeEq : sourceEdge = edge := generatedEdge.symm.trans witnessEdge
  have routeAtU : (witness.dependency, witness.residual) ∈
      flowAt d (stdFuel d) u := by
    simpa [sourceEdgeEq, edgeStart] using routeMember
  have outcome := accounted (witness.dependency, witness.residual)
    routeAtU edge edgeBetween tail (by simpa [witnessEdge] using consumed)
  rcases outcome with exhausted | result
  · exact (not_exhausted_of_continuable d w tail exhausted tailMember.2).elim
  · obtain ⟨dependency, resultMember⟩ := result
    refine ⟨dependency, ?_⟩
    unfold witnessableExpected WitnessableRoutes
    rw [List.mem_filter]
    refine ⟨?_, tailMember.2⟩
    exact (generatedLocalExpected_setEq_flow d suitable roots w wMember
      (dependency, tail)).mpr resultMember

private theorem generated_pair_direction (d : Graph) (w start partner : Vertex)
    (support : FlowPair → Deduction → FlowPair → Deduction → Bool)
    (supportPartnerConsumable : ∀ route edge partnerRoute partnerEdge,
      support route edge partnerRoute partnerEdge = true →
        ∃ partnerTail,
          consume partnerEdge partnerRoute.2 = some partnerTail)
    (closure : ∀ route ∈ flowAt d (stdFuel d) start,
      ∀ edge ∈ edgesBetween d start w, ∀ tail,
      consume edge route.2 = some tail →
      ExhaustedConsumedTailAt d w tail ∨
        ∃ partnerRoute ∈ flowAt d (stdFuel d) partner,
          ∃ partnerEdge ∈ edgesBetween d partner w,
            consume partnerEdge partnerRoute.2 = some tail ∨
              support route edge partnerRoute partnerEdge = true) :
    ((edgesBetween d start w).all fun edge =>
      (edgeWitnessesFor (generateCertificate d) edge).all fun witness =>
        match consume edge witness.residual with
        | none => true
        | some tail =>
            !routeContinuableAtB d w tail ||
              (edgesBetween d partner w).any fun partnerEdge =>
                (edgeWitnessesFor (generateCertificate d) partnerEdge).any
                  fun partnerWitness =>
                    consume partnerEdge partnerWitness.residual == some tail ||
                      support (witness.dependency, witness.residual) edge
                        (partnerWitness.dependency, partnerWitness.residual)
                        partnerEdge) = true := by
  rw [List.all_eq_true]
  intro edge edgeBetween
  rw [List.all_eq_true]
  intro witness witnessMember
  cases consumed : consume edge witness.residual with
  | none => simp
  | some tail =>
      by_cases continuable : routeContinuableAtB d w tail = true
      · simp only [continuable, Bool.not_true, Bool.false_or]
        obtain ⟨_edgeMember, _witnessEdge, routeMember, _colour⟩ :=
          (mem_generated_edgeWitnessesFor_iff d edge witness).mp witnessMember
        have edgeStart : edge.START = start := (mem_edgesBetween edgeBetween).1
        have routeAtStart : (witness.dependency, witness.residual) ∈
            flowAt d (stdFuel d) start := by
          simpa [edgeStart] using routeMember
        have outcome := closure (witness.dependency, witness.residual)
          routeAtStart edge edgeBetween tail consumed
        rcases outcome with exhausted | partnered
        · exact (not_exhausted_of_continuable d w tail exhausted continuable).elim
        · obtain ⟨partnerRoute, partnerRouteMember, partnerEdge,
              partnerEdgeBetween, agreement⟩ := partnered
          rw [List.any_eq_true]
          refine ⟨partnerEdge, partnerEdgeBetween, ?_⟩
          rw [List.any_eq_true]
          let partnerWitness : EdgeRouteWitness :=
            { edge := partnerEdge, dependency := partnerRoute.1,
              residual := partnerRoute.2 }
          have partnerEdgeOutgoing : partnerEdge ∈
              get_rule.outgoing partner d :=
            mem_outgoing_of_edgesBetween partnerEdgeBetween
          have partnerEdgeMember : partnerEdge ∈ d.EDGES :=
            mem_outgoing_mem_edges partner d partnerEdgeOutgoing
          have partnerEdgeStart : partnerEdge.START = partner :=
            (mem_edgesBetween partnerEdgeBetween).1
          have partnerWitnessMember : partnerWitness ∈
              edgeWitnessesFor (generateCertificate d) partnerEdge := by
            apply (mem_generated_edgeWitnessesFor_iff d partnerEdge _).mpr
            refine ⟨partnerEdgeMember, rfl, ?_, ?_⟩
            · simpa [partnerWitness, partnerEdgeStart] using partnerRouteMember
            · rcases agreement with same | reseeded
              · exact (consume_some_colour same).symm
              · obtain ⟨partnerTail, partnerConsumes⟩ :=
                  supportPartnerConsumable _ _ _ _ reseeded
                exact (consume_some_colour partnerConsumes).symm
          refine ⟨partnerWitness, partnerWitnessMember, ?_⟩
          rw [Bool.or_eq_true]
          rcases agreement with same | reseeded
          · left
            simp [partnerWitness, same]
          · right
            simpa [partnerWitness] using reseeded
      · have continuableFalse : routeContinuableAtB d w tail = false :=
          Bool.eq_false_of_not_eq_true continuable
        simp [continuableFalse]

/-- The generated certificate inherits occurrence-aware tail closure from the
graph's raw Flow semantics.  In particular, this proof does not recover Flow
from an already verified certificate: every generated witness is first
unfolded back to the raw route from which the generator created it. -/
theorem generateCertificate_tailClosure (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    LocalTailClosure (generateCertificate d) := by
  intro w wMember pairsNonempty
  have expanded : ExpandedCert d (generateCertificate d) :=
    generateCertificate_expanded d suitable roots
  have closure := suitable.flowTailClosure w wMember pairsNonempty
  constructor
  · intro u uMember
    rcases closure.1 u uMember with eliminationPremise | carrier
    · exact Or.inl (by simpa using eliminationPremise)
    · exact Or.inr
        (generated_carrierAccounted d suitable roots expanded wMember carrier)
  · intro major minor pairMember
    have pairClosure := closure.2 major minor pairMember
    unfold certificatePairTailClosureAtB certificatePairTailClosureFromB
    rw [Bool.and_eq_true]
    constructor
    · apply generated_pair_direction d w minor major
        (fun minorRoute minorEdge majorRoute majorEdge =>
          reseedPairSupportsB d w minorRoute minorEdge majorRoute majorEdge)
      · intro minorRoute minorEdge majorRoute majorEdge supported
        unfold reseedPairSupportsB at supported
        simp only [Bool.and_eq_true] at supported
        have consumable := supported.1.1.1.2
        unfold reseedPremiseAgreesB at consumable
        exact Option.isSome_iff_exists.mp consumable
      · exact pairClosure.1
    · apply generated_pair_direction d w major minor
        (fun majorRoute majorEdge minorRoute minorEdge =>
          reseedPairSupportsB d w minorRoute minorEdge majorRoute majorEdge)
      · intro majorRoute majorEdge minorRoute minorEdge supported
        unfold reseedPairSupportsB at supported
        simp only [Bool.and_eq_true] at supported
        have consumable := supported.1.1.1.1
        unfold reseedPremiseAgreesB at consumable
        exact Option.isSome_iff_exists.mp consumable
      · exact pairClosure.2

/-- Under the graph-side suitability invariants, the canonical generator
satisfies every current local certificate obligation. -/
theorem generateCertificate_localChecks (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    LocalChecks d (generateCertificate d) := by
  refine ⟨generateCertificate_graph d,
    generateCertificate_production d suitable roots,
    generateCertificate_coverage d suitable roots,
    generateCertificate_tailClosure d suitable roots,
    generateCertificate_rootDischarge d suitable roots,
    suitable.structural, suitable.edgesNodup, suitable.oneEdgePerColour,
    generateCertificate_witnessFaithful d suitable roots,
    suitable.levelFormulaUnique, suitable.hypothesesHaveNoIncoming⟩

/-- Completeness of the concrete canonical generator for the current strong
certificate format. -/
theorem generateCertificate_strongFormation (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    StrongFormationValid d (generateCertificate d) := by
  refine ⟨generateCertificate_expanded d suitable roots, ?_,
    generateCertificate_universalRootDischarge d suitable roots⟩
  exact generateCertificate_localChecks d suitable roots

/-- On suitable graphs, the generated certificate is strongly valid exactly
when raw Flow is discharged at every root. -/
theorem rootsDischargedB_iff_generatedStrongFormation (d : Graph)
    (suitable : SuitableGraph d) :
    rootsDischargedB d = true ↔
      StrongFormationValid d (generateCertificate d) := by
  constructor
  · exact generateCertificate_strongFormation d suitable
  · exact strongFormationValid_rootsDischargedB d (generateCertificate d)

theorem rootsDischargedB_implies_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    ∃ certificate, StrongFormationValid d certificate :=
  ⟨generateCertificate d,
    generateCertificate_strongFormation d suitable roots⟩

theorem rootsDischarged_has_strongCertificate (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    ∃ certificate, StrongFormationValid d certificate :=
  rootsDischargedB_implies_exists_strongCertificate d suitable roots

/-- Certificate existence is neither assumed nor encoded into `SuitableGraph`:
it is equivalent to the independently stated raw-Flow root semantics. -/
theorem rootsDischargedB_iff_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraph d) :
    rootsDischargedB d = true ↔
      ∃ certificate, StrongFormationValid d certificate := by
  constructor
  · exact rootsDischargedB_implies_exists_strongCertificate d suitable
  · rintro ⟨certificate, valid⟩
    exact strongFormationValid_rootsDischargedB d certificate valid

theorem rootsDischarged_iff_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraph d) :
    rootsDischargedB d = true ↔
      ∃ certificate, StrongFormationValid d certificate :=
  rootsDischargedB_iff_exists_strongCertificate d suitable

/-- The certificate-existence characterization composes with the established
raw-Flow-to-Grid bridge, without identifying either representation. -/
theorem flowCompiledGridValid_iff_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraph d) :
    FlowCompiledGridValid d ↔
      ∃ certificate, StrongFormationValid d certificate := by
  rw [← rootsDischargedB_iff_flowCompiledGridValid]
  exact rootsDischargedB_iff_exists_strongCertificate d suitable

theorem flowCompiledGridValid_iff_generatedStrongFormation (d : Graph)
    (suitable : SuitableGraph d) :
    FlowCompiledGridValid d ↔
      StrongFormationValid d (generateCertificate d) := by
  rw [← rootsDischargedB_iff_flowCompiledGridValid]
  exact rootsDischargedB_iff_generatedStrongFormation d suitable

/-- Generate-then-verify round trip at the semantic payload boundary.  The
left side is the certificate's deduplicated stored payload and the right side
is the witnessable projection of raw Flow. -/
theorem generatedCertificate_roundTrip (d : Graph)
    (suitable : SuitableGraph d) (roots : rootsDischargedB d = true) :
    ∀ v ∈ d.NODES,
      ListSetEq (StoredRoutes (generateCertificate d) v)
        (WitnessableRoutes d v (flowAt d (stdFuel d) v)) := by
  exact storedRoutes_setEq_flow d (generateCertificate d)
    (generateCertificate_expanded d suitable roots)
    (generateCertificate_localChecks d suitable roots)


end Certificate.Flow.GeneratorCompleteness
