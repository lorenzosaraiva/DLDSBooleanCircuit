import Semantic.OccurrenceCertificateGenerator
import Semantic.OccurrenceCertificateSoundness
import Semantic.FlowBranchingGridBridge

open Semantic FlowSpec SelfCertifyingProbe SelfCertifying
open FlowBranchingGridBridge RouteIndexedMultiToken

namespace OccurrenceCertificate





def CursorRouteHeadCoverageOccurrence (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] →
    (∀ r ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v,
        ExhaustedFlowArrivalAt d v r ∨
          ∃ e ∈ get_rule.outgoing v d, e.COLOUR = headColour r.2) ∧
    (∀ e ∈ get_rule.outgoing v d,
        ∃ r ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v, headColour r.2 = e.COLOUR)


def CursorPairFlowTailClosureOccurrence (d : Graph) (w M S : Vertex) : Prop :=
  (∀ bp ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) S, ∀ eS ∈ edgesBetween d S w, ∀ q,
      consume eS bp.2 = some q →
      ExhaustedConsumedTailAt d w q ∨
        ∃ bq ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) M, ∃ eM ∈ edgesBetween d M w,
          consume eM bq.2 = some q ∨
            reseedPairSupportsB d w bp eS bq eM = true) ∧
  (∀ bp ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) M, ∀ eM ∈ edgesBetween d M w, ∀ q,
      consume eM bp.2 = some q →
      ExhaustedConsumedTailAt d w q ∨
        ∃ bq ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) S, ∃ eS ∈ edgesBetween d S w,
          consume eS bq.2 = some q ∨
            reseedPairSupportsB d w bq eS bp eM = true)



def CursorCarrierAccountedAtOccurrence (d : Graph) (w u : Vertex) : Prop :=
  ∃ alpha, w.FORMULA = Formula.implication alpha u.FORMULA ∧
    ∀ bp ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) u, ∀ edge ∈ edgesBetween d u w, ∀ tail,
      consume edge bp.2 = some tail →
      ExhaustedConsumedTailAt d w tail ∨
        ∃ dependency, (dependency, tail) ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) w



def CursorFlowTailClosureOccurrence (d : Graph) : Prop :=
  ∀ w ∈ d.NODES, elimPairsAt d w ≠ [] →
    (∀ u ∈ predsOf d w,
      ElimPremiseAt d w u ∨ CursorCarrierAccountedAtOccurrence d w u) ∧
    (∀ M S, (M, S) ∈ elimPairsAt d w →
      CursorPairFlowTailClosureOccurrence d w M S)



structure SuitableGraphOccurrence (d : Graph) : Prop where
  structural : structuralValid d
  localShapesValid : ∀ v ∈ d.NODES, localShape d v ≠ .invalid
  edgesNodup : d.EDGES.Nodup
  oneEdgePerColour : OneEdgePerColourPerNode d
  routeHeadCoverage : CursorRouteHeadCoverageOccurrence d
  flowTailClosure : CursorFlowTailClosureOccurrence d
  levelFormulaUnique : LevelFormulaUnique d
  hypothesesHaveNoIncoming : HypothesesHaveNoIncoming d
  rootDecoration : RootDischarged d


theorem mem_generated_edgeRoutes_iff (d : Graph)
    (witness : EdgeRouteWitness) :
    witness ∈ (generateOccurrenceCertificate d).edgeRoutes ↔
      ∃ edge ∈ d.EDGES,
        witness.edge = edge ∧
        (witness.dependency, witness.residual) ∈
          LambdaFlow.guardFreeFlowAt d (stdFuel d) edge.START ∧
        headColour witness.residual = edge.COLOUR := by
  rw [generateOccurrenceCertificate_eq_raw]
  simp only [witnessForEdgeFrom,
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
    output ∈ (generateOccurrenceCertificate d).terminalOutputs ↔
      output.node ∈ d.NODES ∧
      (get_rule.outgoing output.node d).isEmpty = true ∧
      (output.dependency, output.residual) ∈
        LambdaFlow.guardFreeFlowAt d (stdFuel d) output.node := by
  rw [generateOccurrenceCertificate_eq_raw]
  simp only [List.mem_flatMap]
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
    witness ∈ edgeWitnessesFor (generateOccurrenceCertificate d) edge ↔
      edge ∈ d.EDGES ∧ witness.edge = edge ∧
      (witness.dependency, witness.residual) ∈
        LambdaFlow.guardFreeFlowAt d (stdFuel d) edge.START ∧
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
      ((edgeWitnessesFor (generateOccurrenceCertificate d) edge).map fun witness =>
        (witness.dependency, witness.residual))
      ((LambdaFlow.guardFreeFlowAt d (stdFuel d) edge.START).filter fun route =>
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
    route ∈ StoredRoutes (generateOccurrenceCertificate d) v ↔
      (∃ edge ∈ d.EDGES, edge.START = v ∧
        route ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v ∧
        headColour route.2 = edge.COLOUR) ∨
      (v ∈ d.NODES ∧ (get_rule.outgoing v d).isEmpty = true ∧
        route ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v) := by
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




theorem generatedStoredRoutes_setEq_guardFree (d : Graph)
    (roots : LambdaFlow.guardFreeRootsB d = true) :
    ∀ v ∈ d.NODES,
      ListSetEq (StoredRoutes (generateOccurrenceCertificate d) v)
        (WitnessableRoutes d v (LambdaFlow.guardFreeFlowAt d (stdFuel d) v)) := by
  rw [LambdaFlow.guardFreeRootsB_eq_raw] at roots
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


theorem generateOccurrenceCertificate_edge_covered (d : Graph)
    (suitable : SuitableGraphOccurrence d) (edge : Deduction)
    (edgeMember : edge ∈ d.EDGES) :
    edgeWitnessesFor (generateOccurrenceCertificate d) edge ≠ [] := by
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
        EdgeRouteWitness) ∈ edgeWitnessesFor (generateOccurrenceCertificate d) edge := by
    apply (mem_generated_edgeWitnessesFor_iff d edge _).mpr
    exact ⟨edgeMember, rfl, routeMember, colour⟩
  rw [witnessesEmpty] at witnessMember
  exact List.not_mem_nil witnessMember

theorem generateOccurrenceCertificate_terminal_covered (d : Graph)
    (roots : LambdaFlow.guardFreeRootsB d = true) (v : Vertex)
    (vMember : v ∈ d.NODES)
    (root : (get_rule.outgoing v d).isEmpty = true) :
    ((generateOccurrenceCertificate d).terminalOutputs.filter fun output =>
      output.node == v) ≠ [] := by
  rw [LambdaFlow.guardFreeRootsB_eq_raw] at roots
  rw [List.all_eq_true] at roots
  have rootCheck := roots v vMember
  simp only [root, Bool.not_true, Bool.false_or, Bool.and_eq_true] at rootCheck
  have flowNonempty : LambdaFlow.guardFreeFlowAt d (stdFuel d) v ≠ [] := by
    simpa [List.isEmpty_iff] using rootCheck.1
  obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ flowNonempty
  intro outputsEmpty
  let output : TerminalOutput :=
    { node := v, dependency := route.1, residual := route.2 }
  have outputMember : output ∈ (generateOccurrenceCertificate d).terminalOutputs :=
    (mem_generated_terminalOutputs_iff d output).mpr
      ⟨by simpa [output] using vMember, by simpa [output] using root,
        by simpa [output] using routeMember⟩
  have filtered : output ∈
      (generateOccurrenceCertificate d).terminalOutputs.filter fun candidate =>
        candidate.node == v := by
    rw [List.mem_filter]
    exact ⟨outputMember, by simp [output]⟩
  rw [outputsEmpty] at filtered
  exact List.not_mem_nil filtered

theorem generateOccurrenceCertificate_expanded (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    ExpandedCert d (generateOccurrenceCertificate d) := by
  refine ⟨generateOccurrenceCertificate_graph d, ?_, ?_, ?_, ?_⟩
  · exact generateOccurrenceCertificate_edge_covered d suitable
  · intro witness witnessMember
    obtain ⟨edge, edgeMember, witnessEdge, _⟩ :=
      (mem_generated_edgeRoutes_iff d witness).mp witnessMember
    simpa [witnessEdge] using edgeMember
  · exact generateOccurrenceCertificate_terminal_covered d roots
  · intro output outputMember
    obtain ⟨nodeMember, root, _⟩ :=
      (mem_generated_terminalOutputs_iff d output).mp outputMember
    exact ⟨nodeMember, root⟩

theorem generatedLocalExpectedOccurrence_setEq_guardFree (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    forall v, v ∈ d.NODES →
      ListSetEq (localExpectedOccurrence (generateOccurrenceCertificate d) v)
        (LambdaFlow.guardFreeFlowAt d (stdFuel d) v) := by
  intro v hv
  apply localExpectedOccurrence_setEq_guardFree_of_preds
  · simpa using suitable.structural
  · simpa using suitable.localShapesValid v hv
  · intro u hu
    exact generatedStoredRoutes_setEq_guardFree d roots u
      (mem_predsOf_mem_nodes suitable.structural hu)

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

theorem generateOccurrenceCertificate_production (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    LocalProductionOccurrence (generateOccurrenceCertificate d) := by
  intro v vMember
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have storedFlow := generatedStoredRoutes_setEq_guardFree d roots v vMemberD
  have localFlow := generatedLocalExpectedOccurrence_setEq_guardFree d suitable roots v vMemberD
  have expectedFlow : ListSetEq
      (witnessableExpectedOccurrence (generateOccurrenceCertificate d) v)
      (WitnessableRoutes d v (LambdaFlow.guardFreeFlowAt d (stdFuel d) v)) := by
    unfold witnessableExpectedOccurrence
    simpa using witnessableRoutes_congr d v localFlow
  have production : ListSetEq
      (StoredRoutes (generateOccurrenceCertificate d) v)
      (witnessableExpectedOccurrence (generateOccurrenceCertificate d) v) :=
    storedFlow.trans expectedFlow.symm
  unfold productionAtOccurrenceB
  cases shape : localShape (generateOccurrenceCertificate d).graph v with
  | invalid =>
      have valid := suitable.localShapesValid v vMemberD
      exact (valid (by simpa using shape)).elim
  | source => exact setEqB_complete production
  | elim major minor => exact setEqB_complete production
  | elimMulti pairs => exact setEqB_complete production
  | intro premise alpha beta => exact setEqB_complete production

theorem generated_edge_projection_setEq_stored_filter (d : Graph)
    (roots : LambdaFlow.guardFreeRootsB d = true) (v : Vertex)
    (vMember : v ∈ d.NODES) (edge : Deduction)
    (edgeOutgoing : edge ∈ get_rule.outgoing v d) :
    ListSetEq
      ((edgeWitnessesFor (generateOccurrenceCertificate d) edge).map fun witness =>
        (witness.dependency, witness.residual))
      ((StoredRoutes (generateOccurrenceCertificate d) v).filter fun route =>
        headColour route.2 == edge.COLOUR) := by
  have edgeMember := mem_outgoing_mem_edges v d edgeOutgoing
  have edgeStart := mem_outgoing_start_eq v d edgeOutgoing
  have generated := generated_edgeWitnessesFor_setEq d edge edgeMember
  have stored := generatedStoredRoutes_setEq_guardFree d roots v vMember
  intro route
  rw [generated route, List.mem_filter, List.mem_filter]
  constructor
  · rintro ⟨flowMember, colour⟩
    refine ⟨(stored route).mpr ?_, colour⟩
    unfold WitnessableRoutes
    rw [List.mem_filter]
    have flowMemberV : route ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v := by
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

theorem generateOccurrenceCertificate_coverage (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    LocalCoverage (generateOccurrenceCertificate d) := by
  intro v vMember edge edgeOutgoing
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have edgeOutgoingD : edge ∈ get_rule.outgoing v d := by
    simpa using edgeOutgoing
  constructor
  · exact generateOccurrenceCertificate_edge_covered d suitable edge
      (mem_outgoing_mem_edges v d edgeOutgoingD)
  · simpa using generated_edge_projection_setEq_stored_filter d roots
      v vMemberD edge edgeOutgoingD

theorem generateOccurrenceCertificate_witnessFaithful (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    LocalWitnessFaithfulOccurrence (generateOccurrenceCertificate d) := by
  intro witness witnessMember
  obtain ⟨edge, edgeMember, witnessEdge, routeMember, colour⟩ :=
    (mem_generated_edgeRoutes_iff d witness).mp witnessMember
  have startMember : witness.edge.START ∈ d.NODES := by
    simpa [witnessEdge] using (suitable.structural.2.1 edge edgeMember).1
  have localMember : (witness.dependency, witness.residual) ∈
      localExpectedOccurrence (generateOccurrenceCertificate d) witness.edge.START :=
    (generatedLocalExpectedOccurrence_setEq_guardFree d suitable roots witness.edge.START
      startMember _).mpr (by simpa [witnessEdge] using routeMember)
  unfold witnessFaithfulAtOccurrenceB
  rw [Bool.or_eq_true]
  right
  rw [List.any_eq_true]
  refine ⟨(witness.dependency, witness.residual), ?_, ?_⟩
  · rw [List.mem_filter]
    exact ⟨localMember, by simp⟩
  · simp [depSetEq]

theorem generateOccurrenceCertificate_rootDischarge (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    LocalRootDischarge (generateOccurrenceCertificate d) := by
  rw [LambdaFlow.guardFreeRootsB_eq_raw] at roots
  rw [List.all_eq_true] at roots
  intro v vMember root
  have vMemberD : v ∈ d.NODES := by simpa using vMember
  have rootD : (get_rule.outgoing v d).isEmpty = true := by simpa using root
  have rootCheck := roots v vMemberD
  simp only [rootD, Bool.not_true, Bool.false_or, Bool.and_eq_true,
    List.all_eq_true] at rootCheck
  have flowNonempty : LambdaFlow.guardFreeFlowAt d (stdFuel d) v ≠ [] := by
    simpa [List.isEmpty_iff] using rootCheck.1
  obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ flowNonempty
  obtain ⟨dependencyEmpty, residualEmpty⟩ := rootCheck.2 route routeMember
  let output : TerminalOutput :=
    { node := v, dependency := route.1, residual := route.2 }
  have outputMember : output ∈ (generateOccurrenceCertificate d).terminalOutputs :=
    (mem_generated_terminalOutputs_iff d output).mpr
      ⟨by simpa [output] using vMemberD, by simpa [output] using rootD,
        by simpa [output] using routeMember⟩
  have decorated := suitable.rootDecoration v vMemberD
    (List.isEmpty_iff.mp rootD)
  refine ⟨⟨output, outputMember, by simp [output], ?_, ?_⟩,
    decorated.1, decorated.2⟩
  · simpa [output] using dependencyEmpty
  · simpa [output] using residualEmpty

theorem generateOccurrenceCertificate_universalRootDischarge (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    UniversalLocalRootDischargeOccurrence (generateOccurrenceCertificate d) := by
  rw [LambdaFlow.guardFreeRootsB_eq_raw] at roots
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
      (generatedStoredRoutes_setEq_guardFree d (by
        rw [LambdaFlow.guardFreeRootsB_eq_raw]
        rw [List.all_eq_true]
        exact roots) v vMemberD route).mp storedMember
    unfold WitnessableRoutes at witnessable
    rw [List.mem_filter] at witnessable
    have empty := rootCheck.2 route witnessable.1
    exact ⟨List.isEmpty_iff.mp empty.1, List.isEmpty_iff.mp empty.2⟩
  · intro route localMember
    have flowMember :=
      (generatedLocalExpectedOccurrence_setEq_guardFree d suitable (by
        rw [LambdaFlow.guardFreeRootsB_eq_raw]
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
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true)
    (expanded : ExpandedCert d (generateOccurrenceCertificate d))
    {w u : Vertex} (wMember : w ∈ d.NODES)
    (cursor : CursorCarrierAccountedAtOccurrence d w u) :
    LocalCarrierAccountedAtOccurrence (generateOccurrenceCertificate d) w u := by
  obtain ⟨alpha, formula, accounted⟩ := cursor
  refine ⟨alpha, by simpa using formula, ?_⟩
  intro tail tailMember
  unfold continuableConsumedWitnessTails at tailMember
  rw [List.mem_filter] at tailMember
  obtain ⟨edge, edgeBetween, witness, witnessMember, witnessEdge, consumed⟩ :=
    consumedWitnessTails_exists d (generateOccurrenceCertificate d) expanded tailMember.1
  obtain ⟨sourceEdge, _sourceEdgeMember, generatedEdge,
      routeMember, _colour⟩ :=
    (mem_generated_edgeRoutes_iff d witness).mp witnessMember
  have edgeStart : edge.START = u := (mem_edgesBetween edgeBetween).1
  have sourceEdgeEq : sourceEdge = edge := generatedEdge.symm.trans witnessEdge
  have routeAtU : (witness.dependency, witness.residual) ∈
      LambdaFlow.guardFreeFlowAt d (stdFuel d) u := by
    simpa [sourceEdgeEq, edgeStart] using routeMember
  have outcome := accounted (witness.dependency, witness.residual)
    routeAtU edge edgeBetween tail (by simpa [witnessEdge] using consumed)
  rcases outcome with exhausted | result
  · exact (not_exhausted_of_continuable d w tail exhausted tailMember.2).elim
  · obtain ⟨dependency, resultMember⟩ := result
    refine ⟨dependency, ?_⟩
    unfold witnessableExpectedOccurrence WitnessableRoutes
    rw [List.mem_filter]
    refine ⟨?_, tailMember.2⟩
    exact (generatedLocalExpectedOccurrence_setEq_guardFree d suitable roots w wMember
      (dependency, tail)).mpr resultMember

private theorem generated_pair_direction (d : Graph) (w start partner : Vertex)
    (support : FlowPair → Deduction → FlowPair → Deduction → Bool)
    (supportPartnerConsumable : ∀ route edge partnerRoute partnerEdge,
      support route edge partnerRoute partnerEdge = true →
        ∃ partnerTail,
          consume partnerEdge partnerRoute.2 = some partnerTail)
    (closure : ∀ route ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) start,
      ∀ edge ∈ edgesBetween d start w, ∀ tail,
      consume edge route.2 = some tail →
      ExhaustedConsumedTailAt d w tail ∨
        ∃ partnerRoute ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) partner,
          ∃ partnerEdge ∈ edgesBetween d partner w,
            consume partnerEdge partnerRoute.2 = some tail ∨
              support route edge partnerRoute partnerEdge = true) :
    ((edgesBetween d start w).all fun edge =>
      (edgeWitnessesFor (generateOccurrenceCertificate d) edge).all fun witness =>
        match consume edge witness.residual with
        | none => true
        | some tail =>
            !routeContinuableAtB d w tail ||
              (edgesBetween d partner w).any fun partnerEdge =>
                (edgeWitnessesFor (generateOccurrenceCertificate d) partnerEdge).any
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
            LambdaFlow.guardFreeFlowAt d (stdFuel d) start := by
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
              edgeWitnessesFor (generateOccurrenceCertificate d) partnerEdge := by
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





theorem generateOccurrenceCertificate_tailClosure (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    LocalTailClosureOccurrence (generateOccurrenceCertificate d) := by
  intro w wMember pairsNonempty
  have expanded : ExpandedCert d (generateOccurrenceCertificate d) :=
    generateOccurrenceCertificate_expanded d suitable roots
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



theorem generateOccurrenceCertificate_localChecks (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    LocalChecksOccurrence d (generateOccurrenceCertificate d) := by
  refine ⟨generateOccurrenceCertificate_graph d,
    generateOccurrenceCertificate_production d suitable roots,
    generateOccurrenceCertificate_coverage d suitable roots,
    generateOccurrenceCertificate_tailClosure d suitable roots,
    generateOccurrenceCertificate_rootDischarge d suitable roots,
    suitable.structural, suitable.edgesNodup, suitable.oneEdgePerColour,
    generateOccurrenceCertificate_witnessFaithful d suitable roots,
    suitable.levelFormulaUnique, suitable.hypothesesHaveNoIncoming⟩



theorem generateOccurrenceCertificate_strongFormation (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    StrongFormationValidOccurrence d (generateOccurrenceCertificate d) := by
  refine ⟨generateOccurrenceCertificate_expanded d suitable roots, ?_,
    generateOccurrenceCertificate_universalRootDischarge d suitable roots⟩
  exact generateOccurrenceCertificate_localChecks d suitable roots



theorem guardFreeRootsB_iff_generatedStrongFormation (d : Graph)
    (suitable : SuitableGraphOccurrence d) :
    LambdaFlow.guardFreeRootsB d = true ↔
      StrongFormationValidOccurrence d (generateOccurrenceCertificate d) := by
  constructor
  · exact generateOccurrenceCertificate_strongFormation d suitable
  · exact strongFormationValidOccurrence_guardFreeRootsB d (generateOccurrenceCertificate d)

theorem guardFreeRootsB_implies_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    ∃ certificate, StrongFormationValidOccurrence d certificate :=
  ⟨generateOccurrenceCertificate d,
    generateOccurrenceCertificate_strongFormation d suitable roots⟩

theorem guardFreeRoots_has_strongCertificate (d : Graph)
    (suitable : SuitableGraphOccurrence d) (roots : LambdaFlow.guardFreeRootsB d = true) :
    ∃ certificate, StrongFormationValidOccurrence d certificate :=
  guardFreeRootsB_implies_exists_strongCertificate d suitable roots



theorem guardFreeRootsB_iff_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraphOccurrence d) :
    LambdaFlow.guardFreeRootsB d = true ↔
      ∃ certificate, StrongFormationValidOccurrence d certificate := by
  constructor
  · exact guardFreeRootsB_implies_exists_strongCertificate d suitable
  · rintro ⟨certificate, valid⟩
    exact strongFormationValidOccurrence_guardFreeRootsB d certificate valid

theorem guardFreeRoots_iff_exists_strongCertificate (d : Graph)
    (suitable : SuitableGraphOccurrence d) :
    LambdaFlow.guardFreeRootsB d = true ↔
      ∃ certificate, StrongFormationValidOccurrence d certificate :=
  guardFreeRootsB_iff_exists_strongCertificate d suitable


#print axioms generateOccurrenceCertificate_expanded
#print axioms generatedLocalExpectedOccurrence_setEq_guardFree
#print axioms generateOccurrenceCertificate_localChecks
#print axioms generateOccurrenceCertificate_strongFormation
#print axioms guardFreeRootsB_iff_generatedStrongFormation
#print axioms guardFreeRootsB_iff_exists_strongCertificate

end OccurrenceCertificate
