import Semantic.HCEliminationClassCallerBridge












open Semantic FlowSpec
open CandidateMixedReplay OccurrenceAwareReplay
open LocalSpecificationCandidate HCContextualMixedHE
open HCEliminationClassOperational HCEliminationClassCallerBridge

namespace HCSuccessorDeliverySemantics




def DeliveredFrom (d : Graph) (producer consumer : Vertex)
    (producerRoutes : List FlowPair) (delivery : FlowPair) : Prop :=
  Exists fun input : FlowPair => input ∈ producerRoutes /\
    Exists fun incoming : Deduction =>
      incoming ∈ edgesBetween d producer consumer /\
        consume incoming input.2 = some delivery.2 /\
        input.1 = delivery.1




def DeliveredPremise (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair)
    (formula : Formula) (delivery : FlowPair) : Prop :=
  Exists fun premise : Vertex =>
    premise ∈ predsOf d vertex /\ premise.FORMULA = formula /\
    Exists fun input : FlowPair => input ∈ premiseFlow premise /\
    Exists fun incoming : Deduction => incoming ∈ edgesBetween d premise vertex /\
      consume incoming input.2 = some delivery.2 /\ input.1 = delivery.1

theorem deliveredPremise_iff_exists_deliveredFrom
    (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair)
    (formula : Formula) (delivery : FlowPair) :
    DeliveredPremise d vertex premiseFlow formula delivery <->
      Exists fun premise : Vertex =>
        premise ∈ predsOf d vertex /\ premise.FORMULA = formula /\
          DeliveredFrom d premise vertex (premiseFlow premise) delivery := by
  rfl



def DeliveredBy (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair) (selected : Vertex -> Prop)
    (formula : Formula) (delivery : FlowPair) : Prop :=
  Exists fun premise : Vertex =>
    premise ∈ predsOf d vertex /\ selected premise /\
      premise.FORMULA = formula /\
        DeliveredFrom d premise vertex (premiseFlow premise) delivery

theorem deliveredPremise_iff_deliveredBy_all
    (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair)
    (formula : Formula) (delivery : FlowPair) :
    DeliveredPremise d vertex premiseFlow formula delivery <->
      DeliveredBy d vertex premiseFlow (fun _ => True) formula delivery := by
  constructor
  · rintro ⟨premise, predecessor, premiseFormula, delivered⟩
    exact ⟨premise, predecessor, trivial, premiseFormula, delivered⟩
  · rintro ⟨premise, predecessor, _, premiseFormula, delivered⟩
    exact ⟨premise, predecessor, premiseFormula, delivered⟩



theorem deliveredBy_all_iff_partition
    (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair) (selected : Vertex -> Prop)
    (formula : Formula) (delivery : FlowPair) :
    DeliveredBy d vertex premiseFlow (fun _ => True) formula delivery <->
      DeliveredBy d vertex premiseFlow selected formula delivery \/
        DeliveredBy d vertex premiseFlow (fun premise => ¬ selected premise)
          formula delivery := by
  classical
  constructor
  · rintro ⟨premise, predecessor, _, premiseFormula, delivered⟩
    by_cases member : selected premise
    · exact Or.inl
        ⟨premise, predecessor, member, premiseFormula, delivered⟩
    · exact Or.inr
        ⟨premise, predecessor, member, premiseFormula, delivered⟩
  · rintro (selectedDelivery | retainedDelivery)
    · obtain ⟨premise, predecessor, _, premiseFormula, delivered⟩ :=
        selectedDelivery
      exact ⟨premise, predecessor, trivial, premiseFormula, delivered⟩
    · obtain ⟨premise, predecessor, _, premiseFormula, delivered⟩ :=
        retainedDelivery
      exact ⟨premise, predecessor, trivial, premiseFormula, delivered⟩




theorem deliveredPremise_iff_of_partition
    (left right : Graph) (leftVertex rightVertex : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (leftSelected rightSelected : Vertex -> Prop)
    (selected : forall formula delivery,
      DeliveredBy left leftVertex leftFlow leftSelected formula delivery <->
        DeliveredBy right rightVertex rightFlow rightSelected formula delivery)
    (retained : forall formula delivery,
      DeliveredBy left leftVertex leftFlow
          (fun premise => ¬ leftSelected premise) formula delivery <->
        DeliveredBy right rightVertex rightFlow
          (fun premise => ¬ rightSelected premise) formula delivery) :
    forall formula delivery,
      DeliveredPremise left leftVertex leftFlow formula delivery <->
        DeliveredPremise right rightVertex rightFlow formula delivery := by
  intro formula delivery
  calc
    DeliveredPremise left leftVertex leftFlow formula delivery <->
        DeliveredBy left leftVertex leftFlow (fun _ => True)
          formula delivery :=
      deliveredPremise_iff_deliveredBy_all left leftVertex leftFlow
        formula delivery
    _ <-> DeliveredBy left leftVertex leftFlow leftSelected formula delivery \/
        DeliveredBy left leftVertex leftFlow
          (fun premise => ¬ leftSelected premise) formula delivery :=
      deliveredBy_all_iff_partition left leftVertex leftFlow leftSelected
        formula delivery
    _ <-> DeliveredBy right rightVertex rightFlow rightSelected
          formula delivery \/
        DeliveredBy right rightVertex rightFlow
          (fun premise => ¬ rightSelected premise) formula delivery :=
      or_congr (selected formula delivery) (retained formula delivery)
    _ <-> DeliveredBy right rightVertex rightFlow (fun _ => True)
          formula delivery :=
      (deliveredBy_all_iff_partition right rightVertex rightFlow
        rightSelected formula delivery).symm
    _ <-> DeliveredPremise right rightVertex rightFlow formula delivery :=
      (deliveredPremise_iff_deliveredBy_all right rightVertex rightFlow
        formula delivery).symm






def flowObservationAcross (production : MixedProduction)
    (conclusion : Formula) (route : FlowPair) (actualEdge : Deduction) :
    Option BoundaryObservation :=
  (consume actualEdge route.2).map fun tail =>
    { production := production
      conclusion := conclusion
      use := actualEdge.END
      dependency := route.1
      residual := ⟨tail⟩ }


def flowBoundaryDeliveries (production : MixedProduction)
    (conclusion : Formula) (routes : List FlowPair)
    (actualEdges : List Deduction) : List BoundaryObservation :=
  routes.flatMap fun route =>
    actualEdges.filterMap fun actualEdge =>
      flowObservationAcross production conclusion route actualEdge

theorem mem_flowBoundaryDeliveries_iff
    (production : MixedProduction) (conclusion : Formula)
    (routes : List FlowPair) (actualEdges : List Deduction)
    (observation : BoundaryObservation) :
    observation ∈
        flowBoundaryDeliveries production conclusion routes actualEdges <->
      Exists fun route : FlowPair => route ∈ routes /\
        Exists fun actualEdge : Deduction => actualEdge ∈ actualEdges /\
          flowObservationAcross production conclusion route actualEdge =
            some observation := by
  simp [flowBoundaryDeliveries]

theorem flowBoundaryDeliveries_routes_congr
    (production : MixedProduction) (conclusion : Formula)
    (actualEdges : List Deduction) {left right : List FlowPair}
    (equal : SelfCertifying.ListSetEq left right) :
    SelfCertifying.ListSetEq
      (flowBoundaryDeliveries production conclusion left actualEdges)
      (flowBoundaryDeliveries production conclusion right actualEdges) := by
  unfold flowBoundaryDeliveries
  exact SelfCertifying.flatMap_congr_list
    (fun route => actualEdges.filterMap fun actualEdge =>
      flowObservationAcross production conclusion route actualEdge) equal



theorem flowBoundaryDeliveries_elimination_eq
    (conclusion : Formula) (routes : List FlowPair)
    (actualEdges : List Deduction) :
    flowBoundaryDeliveries MixedProduction.elimination conclusion routes
        actualEdges =
      boundaryDeliveries conclusion (eliminationCandidateRoutes routes)
        actualEdges := by
  induction routes with
  | nil => rfl
  | cons route rest ih =>
      simp only [flowBoundaryDeliveries, eliminationCandidateRoutes,
        List.flatMap_cons, List.map_cons, boundaryDeliveries]
      congr 1
      · apply List.filterMap_congr
        intro actualEdge _
        cases route with
        | mk dependency residual =>
            cases residual with
            | nil =>
                by_cases colour : actualEdge.COLOUR = 0 <;>
                  simp [flowObservationAcross, observeAcross,
                    consumeCandidate, FlowSpec.consume, colour]
            | cons colour residual =>
                by_cases colourMatches : actualEdge.COLOUR = colour <;>
                  simp [flowObservationAcross, observeAcross,
                    consumeCandidate, FlowSpec.consume, colourMatches]





theorem deliveredFrom_iff_flowBoundaryDelivery
    (d : Graph) (producer consumer : Vertex)
    (producerRoutes : List FlowPair) (production : MixedProduction)
    (conclusion : Formula) (delivery : FlowPair) :
    DeliveredFrom d producer consumer producerRoutes delivery <->
      Exists fun observation : BoundaryObservation =>
        observation ∈ flowBoundaryDeliveries production conclusion
          producerRoutes (get_rule.outgoing producer d) /\
        observation.use = consumer /\
        observation.dependency = delivery.1 /\
        observation.residual.colours = delivery.2 := by
  constructor
  · rintro ⟨input, inputMember, incoming, betweenMember, consumed,
      dependency⟩
    have endpoints := FlowSpec.mem_edgesBetween betweenMember
    let observation : BoundaryObservation :=
      { production := production
        conclusion := conclusion
        use := consumer
        dependency := delivery.1
        residual := ⟨delivery.2⟩ }
    refine ⟨observation, ?_, rfl, rfl, rfl⟩
    apply (mem_flowBoundaryDeliveries_iff production conclusion producerRoutes
      (get_rule.outgoing producer d) observation).mpr
    refine ⟨input, inputMember, incoming,
      FlowSpec.mem_outgoing_of_edgesBetween betweenMember, ?_⟩
    simp [flowObservationAcross, consumed, observation, endpoints.2.1,
      dependency]
  · rintro ⟨observation, observationMember, useEq, dependencyEq,
      residualEq⟩
    obtain ⟨input, inputMember, incoming, outgoingMember, observed⟩ :=
      (mem_flowBoundaryDeliveries_iff production conclusion producerRoutes
        (get_rule.outgoing producer d) observation).mp observationMember
    cases consumed : consume incoming input.2 with
    | none => simp [flowObservationAcross, consumed] at observed
    | some tail =>
        simp only [flowObservationAcross, consumed, Option.map_some] at observed
        have observationEq :
            ({ production := production
               conclusion := conclusion
               use := incoming.END
               dependency := input.1
               residual := ⟨tail⟩ } : BoundaryObservation) = observation := by
          exact Option.some.inj observed
        have endEq : incoming.END = consumer := by
          rw [← useEq, ← observationEq]
        have inputDependency : input.1 = delivery.1 := by
          rw [← dependencyEq, ← observationEq]
        have tailEq : tail = delivery.2 := by
          simpa [← observationEq] using residualEq
        have edgeMember := mem_outgoing_mem_edges producer d outgoingMember
        have startEq := mem_outgoing_start_eq producer d outgoingMember
        have between : incoming ∈ edgesBetween d producer consumer := by
          unfold edgesBetween
          rw [List.mem_filter]
          exact ⟨edgeMember, decide_eq_true ⟨startEq, endEq⟩⟩
        refine ⟨input, inputMember, incoming, between, ?_, inputDependency⟩
        simpa [tailEq] using consumed



theorem deliveredFrom_predecessor
    (d : Graph) (producer consumer : Vertex)
    (producerRoutes : List FlowPair) (delivery : FlowPair)
    (delivered : DeliveredFrom d producer consumer producerRoutes delivery) :
    producer ∈ predsOf d consumer := by
  obtain ⟨_, _, incoming, betweenMember, _, _⟩ := delivered
  obtain ⟨startEq, endEq, edgeMember⟩ := FlowSpec.mem_edgesBetween betweenMember
  rw [FlowSpec.predsOf_eq, List.mem_eraseDups, List.mem_map]
  refine ⟨incoming, ?_, startEq⟩
  exact mem_incoming_of_mem_edges_end_eq consumer d edgeMember endEq




theorem deliveredBy_iff_exists_selected_deliveredFrom
    (d : Graph) (consumer : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (selected : Vertex -> Prop) (formula : Formula) (delivery : FlowPair) :
    DeliveredBy d consumer premiseFlow selected formula delivery <->
      Exists fun producer : Vertex => selected producer /\
        producer.FORMULA = formula /\
        DeliveredFrom d producer consumer (premiseFlow producer) delivery := by
  constructor
  · rintro ⟨producer, _, chosen, producerFormula, delivered⟩
    exact ⟨producer, chosen, producerFormula, delivered⟩
  · rintro ⟨producer, chosen, producerFormula, delivered⟩
    exact ⟨producer,
      deliveredFrom_predecessor d producer consumer (premiseFlow producer)
        delivery delivered,
      chosen, producerFormula, delivered⟩



theorem deliveredFrom_iff_of_routes_edges
    (left right : Graph) (producer consumer : Vertex)
    (leftRoutes rightRoutes : List FlowPair)
    (delivery : FlowPair)
    (routes : SelfCertifying.ListSetEq leftRoutes rightRoutes)
    (edges : forall incoming,
      incoming ∈ edgesBetween left producer consumer <->
        incoming ∈ edgesBetween right producer consumer) :
    DeliveredFrom left producer consumer leftRoutes delivery <->
      DeliveredFrom right producer consumer rightRoutes delivery := by
  constructor
  · rintro ⟨input, inputMember, incoming, incomingMember, consumed,
      dependency⟩
    exact ⟨input, (routes input).mp inputMember, incoming,
      (edges incoming).mp incomingMember, consumed, dependency⟩
  · rintro ⟨input, inputMember, incoming, incomingMember, consumed,
      dependency⟩
    exact ⟨input, (routes input).mpr inputMember, incoming,
      (edges incoming).mpr incomingMember, consumed, dependency⟩




theorem deliveredBy_iff_of_retained_routes_edges
    (left right : Graph) (consumer : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (retained : Vertex -> Prop) (formula : Formula)
    (routes : forall producer, retained producer ->
      SelfCertifying.ListSetEq (leftFlow producer) (rightFlow producer))
    (edges : forall producer, retained producer -> forall incoming,
      incoming ∈ edgesBetween left producer consumer <->
        incoming ∈ edgesBetween right producer consumer) :
    forall delivery,
      DeliveredBy left consumer leftFlow retained formula delivery <->
        DeliveredBy right consumer rightFlow retained formula delivery := by
  intro delivery
  rw [deliveredBy_iff_exists_selected_deliveredFrom,
    deliveredBy_iff_exists_selected_deliveredFrom]
  constructor
  · rintro ⟨producer, retainedProducer, producerFormula, delivered⟩
    exact ⟨producer, retainedProducer, producerFormula,
      (deliveredFrom_iff_of_routes_edges left right producer consumer
        (leftFlow producer) (rightFlow producer) delivery
        (routes producer retainedProducer)
        (edges producer retainedProducer)).mp delivered⟩
  · rintro ⟨producer, retainedProducer, producerFormula, delivered⟩
    exact ⟨producer, retainedProducer, producerFormula,
      (deliveredFrom_iff_of_routes_edges left right producer consumer
        (leftFlow producer) (rightFlow producer) delivery
        (routes producer retainedProducer)
        (edges producer retainedProducer)).mpr delivered⟩




def producerBoundaryDeliveries (d : Graph) (premiseFlow : Vertex -> List FlowPair)
    (production : MixedProduction) (conclusion : Formula)
    (producers : List Vertex) : List BoundaryObservation :=
  producers.flatMap fun producer =>
    flowBoundaryDeliveries production conclusion (premiseFlow producer)
      (get_rule.outgoing producer d)



theorem deliveredBy_iff_producerBoundaryDelivery
    (d : Graph) (consumer : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (production : MixedProduction) (conclusion : Formula)
    (producers : List Vertex)
    (producerFormula : forall producer, producer ∈ producers ->
      producer.FORMULA = conclusion)
    (delivery : FlowPair) :
    DeliveredBy d consumer premiseFlow (fun producer => producer ∈ producers)
        conclusion delivery <->
      Exists fun observation : BoundaryObservation =>
        observation ∈ producerBoundaryDeliveries d premiseFlow production
          conclusion producers /\
        observation.use = consumer /\
        observation.dependency = delivery.1 /\
        observation.residual.colours = delivery.2 := by
  constructor
  · rintro ⟨producer, predecessor, selected, formula, delivered⟩
    obtain ⟨observation, observationMember, useEq, dependencyEq, residualEq⟩ :=
      (deliveredFrom_iff_flowBoundaryDelivery d producer consumer
        (premiseFlow producer) production conclusion delivery).mp delivered
    refine ⟨observation, ?_, useEq, dependencyEq, residualEq⟩
    exact List.mem_flatMap.mpr ⟨producer, selected, observationMember⟩
  · rintro ⟨observation, observationMember, useEq, dependencyEq, residualEq⟩
    obtain ⟨producer, selected, producerMember⟩ :=
      List.mem_flatMap.mp observationMember
    have delivered :=
      (deliveredFrom_iff_flowBoundaryDelivery d producer consumer
        (premiseFlow producer) production conclusion delivery).mpr
        ⟨observation, producerMember, useEq, dependencyEq, residualEq⟩
    exact ⟨producer,
      deliveredFrom_predecessor d producer consumer (premiseFlow producer)
        delivery delivered,
      selected, producerFormula producer selected, delivered⟩




theorem deliveredBy_iff_of_producerBoundary_setEq
    (left right : Graph) (leftConsumer rightConsumer : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (leftProducers rightProducers : List Vertex)
    (production : MixedProduction) (conclusion : Formula)
    (consumerEq : leftConsumer = rightConsumer)
    (leftFormula : forall producer, producer ∈ leftProducers ->
      producer.FORMULA = conclusion)
    (rightFormula : forall producer, producer ∈ rightProducers ->
      producer.FORMULA = conclusion)
    (boundary : SelfCertifying.ListSetEq
      (producerBoundaryDeliveries left leftFlow production conclusion
        leftProducers)
      (producerBoundaryDeliveries right rightFlow production conclusion
        rightProducers)) :
    forall delivery,
      DeliveredBy left leftConsumer leftFlow
          (fun producer => producer ∈ leftProducers) conclusion delivery <->
        DeliveredBy right rightConsumer rightFlow
          (fun producer => producer ∈ rightProducers) conclusion delivery := by
  intro delivery
  rw [deliveredBy_iff_producerBoundaryDelivery left leftConsumer leftFlow
      production conclusion leftProducers leftFormula delivery,
    deliveredBy_iff_producerBoundaryDelivery right rightConsumer rightFlow
      production conclusion rightProducers rightFormula delivery]
  constructor
  · rintro ⟨observation, member, useEq, dependencyEq, residualEq⟩
    exact ⟨observation, (boundary observation).mp member,
      useEq.trans consumerEq, dependencyEq, residualEq⟩
  · rintro ⟨observation, member, useEq, dependencyEq, residualEq⟩
    exact ⟨observation, (boundary observation).mpr member,
      useEq.trans consumerEq.symm, dependencyEq, residualEq⟩





theorem deliveredPremise_of_flowBoundaryDelivery
    (d : Graph) (producer : Vertex)
    (producerRoutes : List FlowPair) (actualEdges : List Deduction)
    (production : MixedProduction) (conclusion : Formula)
    (premiseFlow : Vertex -> List FlowPair)
    (producerRoutesEq : premiseFlow producer = producerRoutes)
    (producerFormula : producer.FORMULA = conclusion)
    (edgesSound : forall actualEdge, actualEdge ∈ actualEdges ->
      actualEdge ∈ d.EDGES /\ actualEdge.START = producer)
    {observation : BoundaryObservation}
    (member : observation ∈
      flowBoundaryDeliveries production conclusion producerRoutes actualEdges) :
    DeliveredPremise d observation.use premiseFlow conclusion
      (observation.dependency, observation.residual.colours) := by
  obtain ⟨route, routeMember, actualEdge, edgeMember, observed⟩ :=
    (mem_flowBoundaryDeliveries_iff production conclusion producerRoutes
      actualEdges observation).mp member
  obtain ⟨edgeInGraph, starts⟩ := edgesSound actualEdge edgeMember
  cases consumed : consume actualEdge route.2 with
  | none => simp [flowObservationAcross, consumed] at observed
  | some tail =>
      simp [flowObservationAcross, consumed] at observed
      subst observation
      have incomingMember : actualEdge ∈
          get_rule.incoming actualEdge.END d :=
        mem_incoming_of_mem_edges_end_eq actualEdge.END d edgeInGraph rfl
      have predecessorMember : producer ∈ predsOf d actualEdge.END := by
        rw [FlowSpec.predsOf_eq, List.mem_eraseDups, List.mem_map]
        exact ⟨actualEdge, incomingMember, starts⟩
      have betweenMember : actualEdge ∈
          edgesBetween d producer actualEdge.END := by
        unfold edgesBetween
        simp [edgeInGraph, starts]
      refine ⟨producer, predecessorMember, producerFormula, route,
        ?_, actualEdge, betweenMember, consumed, rfl⟩
      simpa [producerRoutesEq] using routeMember




theorem mem_candidateEliminationOutputs_iff
    (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair) (output : FlowPair) :
    output ∈ candidateEliminationOutputs d vertex premiseFlow <->
      Exists fun major : Vertex => major ∈ predsOf d vertex /\
      Exists fun minor : Vertex => minor ∈ predsOf d vertex /\
        major.FORMULA = Formula.implication minor.FORMULA vertex.FORMULA /\
      Exists fun minorInput : FlowPair => minorInput ∈ premiseFlow minor /\
      Exists fun majorInput : FlowPair => majorInput ∈ premiseFlow major /\
      Exists fun minorEdge : Deduction =>
        minorEdge ∈ edgesBetween d minor vertex /\
      Exists fun majorEdge : Deduction =>
        majorEdge ∈ edgesBetween d major vertex /\
      Exists fun tail : ColourPath =>
        consume minorEdge minorInput.2 = some tail /\
        consume majorEdge majorInput.2 = some tail /\
        output = (depUnion minorInput.1 majorInput.1, tail) := by
  simp [candidateEliminationOutputs, elimPairsAt]
  constructor
  · rintro ⟨majorWitness, minorWitness,
      ⟨majorMember, minorMember, logical⟩,
      minorDep, minorResidual, minorRoute,
      majorDep, majorResidual, majorRoute,
      minorEdge, minorEdgeMember, majorEdge, majorEdgeMember, produced⟩
    cases minorConsumed : consume minorEdge minorResidual with
    | none => simp [minorConsumed] at produced
    | some minorTail =>
        cases majorConsumed : consume majorEdge majorResidual with
        | none => simp [minorConsumed, majorConsumed] at produced
        | some majorTail =>
            by_cases tailsEqual : minorTail = majorTail
            · subst majorTail
              simp [minorConsumed, majorConsumed] at produced
              refine ⟨majorWitness, majorMember, minorWitness, minorMember,
                logical, minorDep, minorResidual, minorRoute,
                majorDep, majorResidual, majorRoute,
                minorEdge, minorEdgeMember, majorEdge, majorEdgeMember,
                minorTail, minorConsumed, majorConsumed, ?_⟩
              exact produced.symm
            · simp [minorConsumed, majorConsumed, tailsEqual] at produced
  · rintro ⟨majorWitness, majorMember, minorWitness, minorMember, logical,
      minorDep, minorResidual, minorRoute,
      majorDep, majorResidual, majorRoute,
      minorEdge, minorEdgeMember, majorEdge, majorEdgeMember, tail,
      minorConsumed, majorConsumed, rfl⟩
    refine ⟨majorWitness, minorWitness,
      ⟨majorMember, minorMember, logical⟩,
      minorDep, minorResidual, minorRoute,
      majorDep, majorResidual, majorRoute,
      minorEdge, minorEdgeMember, majorEdge, majorEdgeMember, ?_⟩
    simp [minorConsumed, majorConsumed]



theorem candidateEliminationOutputs_setEq_of_deliveries
    (left right : Graph) (leftVertex rightVertex : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (conclusion : leftVertex.FORMULA = rightVertex.FORMULA)
    (deliveries : forall formula delivery,
      DeliveredPremise left leftVertex leftFlow formula delivery <->
        DeliveredPremise right rightVertex rightFlow formula delivery) :
    SelfCertifying.ListSetEq
      (candidateEliminationOutputs left leftVertex leftFlow)
      (candidateEliminationOutputs right rightVertex rightFlow) := by
  intro output
  rw [mem_candidateEliminationOutputs_iff,
    mem_candidateEliminationOutputs_iff]
  constructor
  · rintro ⟨major, majorMember, minor, minorMember, logical,
      minorInput, minorInputMember, majorInput, majorInputMember,
      minorEdge, minorEdgeMember, majorEdge, majorEdgeMember, tail,
      minorConsume, majorConsume, rfl⟩
    have minorDelivery : DeliveredPremise left leftVertex leftFlow
        minor.FORMULA (minorInput.1, tail) :=
      ⟨minor, minorMember, rfl, minorInput, minorInputMember,
        minorEdge, minorEdgeMember, minorConsume, rfl⟩
    have majorDelivery : DeliveredPremise left leftVertex leftFlow
        major.FORMULA (majorInput.1, tail) :=
      ⟨major, majorMember, rfl, majorInput, majorInputMember,
        majorEdge, majorEdgeMember, majorConsume, rfl⟩
    obtain ⟨rightMinor, rightMinorMember, rightMinorFormula,
      rightMinorInput, rightMinorInputMember, rightMinorEdge,
      rightMinorEdgeMember, rightMinorConsume, rightMinorDep⟩ :=
        (deliveries minor.FORMULA (minorInput.1, tail)).mp minorDelivery
    obtain ⟨rightMajor, rightMajorMember, rightMajorFormula,
      rightMajorInput, rightMajorInputMember, rightMajorEdge,
      rightMajorEdgeMember, rightMajorConsume, rightMajorDep⟩ :=
        (deliveries major.FORMULA (majorInput.1, tail)).mp majorDelivery
    refine ⟨rightMajor, rightMajorMember, rightMinor, rightMinorMember, ?_,
      rightMinorInput, rightMinorInputMember,
      rightMajorInput, rightMajorInputMember,
      rightMinorEdge, rightMinorEdgeMember,
      rightMajorEdge, rightMajorEdgeMember, tail,
      rightMinorConsume, rightMajorConsume, ?_⟩
    · rw [rightMajorFormula, rightMinorFormula, ← conclusion]
      exact logical
    · rw [rightMinorDep, rightMajorDep]
  · rintro ⟨major, majorMember, minor, minorMember, logical,
      minorInput, minorInputMember, majorInput, majorInputMember,
      minorEdge, minorEdgeMember, majorEdge, majorEdgeMember, tail,
      minorConsume, majorConsume, rfl⟩
    have minorDelivery : DeliveredPremise right rightVertex rightFlow
        minor.FORMULA (minorInput.1, tail) :=
      ⟨minor, minorMember, rfl, minorInput, minorInputMember,
        minorEdge, minorEdgeMember, minorConsume, rfl⟩
    have majorDelivery : DeliveredPremise right rightVertex rightFlow
        major.FORMULA (majorInput.1, tail) :=
      ⟨major, majorMember, rfl, majorInput, majorInputMember,
        majorEdge, majorEdgeMember, majorConsume, rfl⟩
    obtain ⟨leftMinor, leftMinorMember, leftMinorFormula,
      leftMinorInput, leftMinorInputMember, leftMinorEdge,
      leftMinorEdgeMember, leftMinorConsume, leftMinorDep⟩ :=
        (deliveries minor.FORMULA (minorInput.1, tail)).mpr minorDelivery
    obtain ⟨leftMajor, leftMajorMember, leftMajorFormula,
      leftMajorInput, leftMajorInputMember, leftMajorEdge,
      leftMajorEdgeMember, leftMajorConsume, leftMajorDep⟩ :=
        (deliveries major.FORMULA (majorInput.1, tail)).mpr majorDelivery
    refine ⟨leftMajor, leftMajorMember, leftMinor, leftMinorMember, ?_,
      leftMinorInput, leftMinorInputMember,
      leftMajorInput, leftMajorInputMember,
      leftMinorEdge, leftMinorEdgeMember,
      leftMajorEdge, leftMajorEdgeMember, tail,
      leftMinorConsume, leftMajorConsume, ?_⟩
    · rw [leftMajorFormula, leftMinorFormula, conclusion]
      exact logical
    · rw [leftMinorDep, leftMajorDep]




theorem mem_introductionCandidates_iff
    (d : Graph) (vertex : Vertex)
    (premiseFlow : Vertex -> List FlowPair) (output : FlowPair) :
    output ∈ introductionCandidates d vertex premiseFlow <->
      match vertex.FORMULA with
      | Formula.atom _ => False
      | Formula.implication antecedent consequent =>
          Exists fun delivered : FlowPair =>
            DeliveredPremise d vertex premiseFlow consequent delivered /\
              output = (depRemove delivered.1 antecedent, delivered.2) := by
  cases formulaEq : vertex.FORMULA with
  | atom symbol =>
      simp [introductionCandidates, formulaEq]
  | implication antecedent consequent =>
      simp [introductionCandidates, candidateIntroductionOutputs,
        DeliveredPremise, formulaEq]
      constructor
      · rintro ⟨premiseWitness, premiseMember, premiseFormula,
          dependency, residual, routeMember, incoming, incomingMember,
          produced⟩
        cases consumed : consume incoming residual with
        | none => simp [consumed] at produced
        | some tail =>
            simp [consumed] at produced
            refine ⟨dependency, tail, ?_, ?_⟩
            · exact ⟨premiseWitness, premiseMember, premiseFormula,
                residual, routeMember, incoming, incomingMember, consumed⟩
            · exact produced.symm
      · rintro ⟨dependency, tail, delivered, rfl⟩
        rcases delivered with ⟨premiseWitness, premiseMember, premiseFormula,
          residual, routeMember, incoming, incomingMember, consumed⟩
        refine ⟨premiseWitness, premiseMember, premiseFormula,
          dependency, residual, routeMember,
          incoming, incomingMember, ?_⟩
        simp [consumed]


theorem introductionCandidates_setEq_of_deliveries
    (left right : Graph) (leftVertex rightVertex : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (conclusion : leftVertex.FORMULA = rightVertex.FORMULA)
    (deliveries : forall formula delivery,
      DeliveredPremise left leftVertex leftFlow formula delivery <->
        DeliveredPremise right rightVertex rightFlow formula delivery) :
    SelfCertifying.ListSetEq
      (introductionCandidates left leftVertex leftFlow)
      (introductionCandidates right rightVertex rightFlow) := by
  intro output
  rw [mem_introductionCandidates_iff,
    mem_introductionCandidates_iff]
  rw [← conclusion]
  cases leftVertex.FORMULA with
  | atom symbol => rfl
  | implication antecedent consequent =>
      constructor
      · rintro ⟨delivered, deliveredMember, rfl⟩
        exact ⟨delivered,
          (deliveries consequent delivered).mp deliveredMember, rfl⟩
      · rintro ⟨delivered, deliveredMember, rfl⟩
        exact ⟨delivered,
          (deliveries consequent delivered).mpr deliveredMember, rfl⟩


theorem residualOccursB_eq_of_setEq
    (left right : List FlowPair)
    (equal : SelfCertifying.ListSetEq left right)
    (residual : ColourPath) :
    residualOccursB left residual = residualOccursB right residual := by
  unfold residualOccursB
  rw [Bool.eq_iff_iff, List.any_eq_true, List.any_eq_true]
  constructor
  · rintro ⟨route, routeMember, matching⟩
    exact ⟨route, (equal route).mp routeMember, matching⟩
  · rintro ⟨route, routeMember, matching⟩
    exact ⟨route, (equal route).mpr routeMember, matching⟩




theorem separatedIntroductionOutputs_setEq
    (leftEliminations rightEliminations : List FlowPair)
    (leftIntroductions rightIntroductions : List FlowPair)
    (eliminationsEqual : SelfCertifying.ListSetEq
      leftEliminations rightEliminations)
    (introductionsEqual : SelfCertifying.ListSetEq
      leftIntroductions rightIntroductions) :
    SelfCertifying.ListSetEq
      (leftIntroductions.filter fun route =>
        !residualOccursB leftEliminations route.2)
      (rightIntroductions.filter fun route =>
        !residualOccursB rightEliminations route.2) := by
  intro route
  simp only [List.mem_filter]
  have residualEqual := residualOccursB_eq_of_setEq
    leftEliminations rightEliminations eliminationsEqual route.2
  constructor
  · rintro ⟨member, accepted⟩
    refine ⟨(introductionsEqual route).mp member, ?_⟩
    simpa [residualEqual] using accepted
  · rintro ⟨member, accepted⟩
    refine ⟨(introductionsEqual route).mpr member, ?_⟩
    simpa [residualEqual] using accepted






theorem derivedOutputs_setEq_of_deliveries
    (left right : Graph) (leftVertex rightVertex : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (conclusion : leftVertex.FORMULA = rightVertex.FORMULA)
    (deliveries : forall formula delivery,
      DeliveredPremise left leftVertex leftFlow formula delivery <->
        DeliveredPremise right rightVertex rightFlow formula delivery) :
    SelfCertifying.ListSetEq
      (derivedOutputs left leftVertex leftFlow)
      (derivedOutputs right rightVertex rightFlow) := by
  let leftEliminations :=
    candidateEliminationOutputs left leftVertex leftFlow
  let rightEliminations :=
    candidateEliminationOutputs right rightVertex rightFlow
  let leftIntroductions := introductionCandidates left leftVertex leftFlow
  let rightIntroductions := introductionCandidates right rightVertex rightFlow
  have eliminationsEqual : SelfCertifying.ListSetEq
      leftEliminations rightEliminations :=
    candidateEliminationOutputs_setEq_of_deliveries left right
      leftVertex rightVertex leftFlow rightFlow conclusion deliveries
  have introductionsEqual : SelfCertifying.ListSetEq
      leftIntroductions rightIntroductions :=
    introductionCandidates_setEq_of_deliveries left right
      leftVertex rightVertex leftFlow rightFlow conclusion deliveries
  have separatedEqual : SelfCertifying.ListSetEq
      (leftIntroductions.filter fun route =>
        !residualOccursB leftEliminations route.2)
      (rightIntroductions.filter fun route =>
        !residualOccursB rightEliminations route.2) :=
    separatedIntroductionOutputs_setEq leftEliminations rightEliminations
      leftIntroductions rightIntroductions eliminationsEqual introductionsEqual
  intro route
  simp only [derivedOutputs, separatedIntroductionOutputs,
    List.mem_eraseDups, List.mem_append]
  change (route ∈ leftEliminations ∨
      route ∈ leftIntroductions.filter (fun candidate =>
        !residualOccursB leftEliminations candidate.2)) <->
    (route ∈ rightEliminations ∨
      route ∈ rightIntroductions.filter (fun candidate =>
        !residualOccursB rightEliminations candidate.2))
  exact or_congr (eliminationsEqual route) (separatedEqual route)





def ReplayResultSetEq
    (left right : Except OccurrenceReplayError (List FlowPair)) : Prop :=
  match left, right with
  | .error leftError, .error rightError => leftError = rightError
  | .ok leftRoutes, .ok rightRoutes =>
      SelfCertifying.ListSetEq leftRoutes rightRoutes
  | _, _ => False

theorem listIsEmpty_eq_of_setEq
    {routes routes' : List FlowPair}
    (equal : SelfCertifying.ListSetEq routes routes') :
    routes.isEmpty = routes'.isEmpty := by
  cases routes with
  | nil =>
      have rightNil : routes' = [] := by
        by_contra nonempty
        obtain ⟨route, routeMember⟩ := List.exists_mem_of_ne_nil _ nonempty
        have impossible := (equal route).mpr routeMember
        simp at impossible
      subst routes'
      rfl
  | cons route rest =>
      have rightNonempty : routes' ≠ [] := by
        intro empty
        subst routes'
        have impossible := (equal route).mp (by simp)
        simp at impossible
      cases routes' with
      | nil => exact False.elim (rightNonempty rfl)
      | cons route' rest' => rfl






theorem occurrenceStep_resultSetEq_of_deliveries_and_reseeding
    (left right : Graph) (vertex : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (predecessorsEmpty :
      (predsOf left vertex).isEmpty = (predsOf right vertex).isEmpty)
    (ancestorsEmpty :
      (ancestorsInto left vertex).isEmpty =
        (ancestorsInto right vertex).isEmpty)
    (sources : SelfCertifying.ListSetEq
      (candidateSourceRoutes left vertex)
      (candidateSourceRoutes right vertex))
    (deliveries : forall formula delivery,
      DeliveredPremise left vertex leftFlow formula delivery <->
        DeliveredPremise right vertex rightFlow formula delivery)
    (reseeded : SelfCertifying.ListSetEq
      (reseedDerivedRoutes (ancestorsInto left vertex)
        (derivedOutputs left vertex leftFlow))
      (reseedDerivedRoutes (ancestorsInto right vertex)
        (derivedOutputs right vertex rightFlow))) :
    ReplayResultSetEq
      (occurrenceStep left vertex leftFlow)
      (occurrenceStep right vertex rightFlow) := by
  have derivedEqual : SelfCertifying.ListSetEq
      (derivedOutputs left vertex leftFlow)
      (derivedOutputs right vertex rightFlow) :=
    derivedOutputs_setEq_of_deliveries left right vertex vertex
      leftFlow rightFlow rfl deliveries
  have derivedEmpty := listIsEmpty_eq_of_setEq derivedEqual
  have reseededEmpty := listIsEmpty_eq_of_setEq reseeded
  have mixedEqual : SelfCertifying.ListSetEq
      ((candidateSourceRoutes left vertex ++
        derivedOutputs left vertex leftFlow).eraseDups)
      ((candidateSourceRoutes right vertex ++
        derivedOutputs right vertex rightFlow).eraseDups) :=
    SelfCertifying.eraseDups_congr
      (SelfCertifying.append_congr sources derivedEqual)
  dsimp only [occurrenceStep]
  rw [predecessorsEmpty, ancestorsEmpty, derivedEmpty, reseededEmpty]
  cases predecessorCase : (predsOf right vertex).isEmpty <;>
    cases sourceCase : vertex.HYPOTHESIS <;>
    cases ancestorCase : (ancestorsInto right vertex).isEmpty <;>
    cases derivedCase : (derivedOutputs right vertex rightFlow).isEmpty <;>
    cases reseededCase :
      (reseedDerivedRoutes (ancestorsInto right vertex)
        (derivedOutputs right vertex rightFlow)).isEmpty <;>
    simp [predecessorCase, sourceCase, ancestorCase, derivedCase,
      reseededCase, ReplayResultSetEq, sources, derivedEqual,
      reseeded, mixedEqual]





theorem occurrenceStep_resultSetEq_of_deliveries
    (left right : Graph) (vertex : Vertex)
    (leftFlow rightFlow : Vertex -> List FlowPair)
    (predecessorsEmpty :
      (predsOf left vertex).isEmpty = (predsOf right vertex).isEmpty)
    (ancestors : ancestorsInto left vertex = ancestorsInto right vertex)
    (sources : SelfCertifying.ListSetEq
      (candidateSourceRoutes left vertex)
      (candidateSourceRoutes right vertex))
    (deliveries : forall formula delivery,
      DeliveredPremise left vertex leftFlow formula delivery <->
        DeliveredPremise right vertex rightFlow formula delivery) :
    ReplayResultSetEq
      (occurrenceStep left vertex leftFlow)
      (occurrenceStep right vertex rightFlow) := by
  have derivedEqual : SelfCertifying.ListSetEq
      (derivedOutputs left vertex leftFlow)
      (derivedOutputs right vertex rightFlow) :=
    derivedOutputs_setEq_of_deliveries left right vertex vertex
      leftFlow rightFlow rfl deliveries
  have derivedEmpty := listIsEmpty_eq_of_setEq derivedEqual
  have reseededEqual : SelfCertifying.ListSetEq
      (reseedDerivedRoutes (ancestorsInto left vertex)
        (derivedOutputs left vertex leftFlow))
      (reseedDerivedRoutes (ancestorsInto right vertex)
        (derivedOutputs right vertex rightFlow)) := by
    rw [ancestors]
    exact reseedDerivedRoutes_routes_congr
      (ancestorsInto right vertex)
      (derivedOutputs left vertex leftFlow)
      (derivedOutputs right vertex rightFlow) derivedEqual
  have reseededEmpty := listIsEmpty_eq_of_setEq reseededEqual
  rw [ancestors] at reseededEqual reseededEmpty
  have mixedEqual : SelfCertifying.ListSetEq
      ((candidateSourceRoutes left vertex ++
        derivedOutputs left vertex leftFlow).eraseDups)
      ((candidateSourceRoutes right vertex ++
        derivedOutputs right vertex rightFlow).eraseDups) :=
    SelfCertifying.eraseDups_congr
      (SelfCertifying.append_congr sources derivedEqual)
  dsimp only [occurrenceStep]
  rw [predecessorsEmpty, ancestors, derivedEmpty, reseededEmpty]
  cases predecessorCase : (predsOf right vertex).isEmpty <;>
    cases sourceCase : vertex.HYPOTHESIS <;>
    cases ancestorCase : (ancestorsInto right vertex).isEmpty <;>
    cases derivedCase : (derivedOutputs right vertex rightFlow).isEmpty <;>
    cases reseededCase :
      (reseedDerivedRoutes (ancestorsInto right vertex)
        (derivedOutputs right vertex rightFlow)).isEmpty <;>
    simp [predecessorCase, sourceCase, ancestorCase, derivedCase,
      reseededCase, ReplayResultSetEq, sources, derivedEqual,
      reseededEqual, mixedEqual]

end HCSuccessorDeliverySemantics
