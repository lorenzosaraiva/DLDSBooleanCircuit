import Semantic.OccurrenceCertificateKernel
import Semantic.OccurrenceCertificateChecks
import Semantic.DependencyGuardStability
import Semantic.FaithfulKeyedMemo

open Semantic FlowSpec SelfCertifyingProbe SelfCertifying

namespace OccurrenceCertificate

theorem localExpectedOccurrence_setEq_guardFree_of_preds (c : OccurrenceCertificate) (v : Vertex)
    (structural : structuralValid c.graph)
    (shape : localShape c.graph v ≠ .invalid)
    (agree : ∀ u ∈ predsOf c.graph v,
      ListSetEq (StoredRoutes c u) (WitnessableRoutes c.graph u
        (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph) u))) :
    ListSetEq (localExpectedOccurrence c v)
      (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph) v) := by
  have previous : ∀ u ∈ predsOf c.graph v,
      ListSetEq (outputsAt c u) (WitnessableRoutes c.graph u
        (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph - 1) u)) := by
    intro u hu
    have stable := DependencyGuard.evaluateAt_stdFuel_pred c.graph
      LambdaFlow.allowAll LambdaFlow.allowAll structural
      (mem_predsOf_mem_nodes structural hu)
    simpa only [StoredRoutes, stable] using agree u hu
  have kernel := (localExpectedOccurrence_setEq_step c v structural shape).trans
    ((DependencyGuard.stepWith_setEq_on_preds c.graph
      LambdaFlow.allowAll LambdaFlow.allowAll (outputsAt c)
      (fun u => WitnessableRoutes c.graph u
        (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph - 1) u)) v
      previous).trans (step2_project_inputs c.graph v
        (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph - 1))))
  have fuel : stdFuel c.graph - 1 + 1 = stdFuel c.graph := by
    unfold stdFuel
    omega
  change ListSetEq (localExpectedOccurrence c v)
    (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph - 1 + 1) v) at kernel
  simpa only [fuel] using kernel

theorem storedRoutes_setEq_guardFree (d : Graph) (c : OccurrenceCertificate)
    (expanded : ExpandedCertOccurrence d c) (localChecks : LocalChecksOccurrence d c) :
    ∀ v ∈ d.NODES,
      ListSetEq (StoredRoutes c v) (WitnessableRoutes d v
        (LambdaFlow.guardFreeFlowAt d (stdFuel d) v)) := by
  have graph : c.graph = d := localChecks.1
  subst d
  have structural : structuralValid c.graph := localChecks.2.2.2.2.2.1
  have production := localChecks.2.1
  have key : ∀ n, ∀ v ∈ c.graph.NODES, countAbove c.graph v = n →
      ListSetEq (StoredRoutes c v) (WitnessableRoutes c.graph v
        (LambdaFlow.guardFreeFlowAt c.graph (stdFuel c.graph) v)) := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro v hv hn
      have prod := production v hv
      have shape : localShape c.graph v ≠ .invalid := by
        intro bad
        simp [productionAtOccurrenceB, bad] at prod
      have raw := localExpectedOccurrence_setEq_guardFree_of_preds c v structural shape (by
        intro u hu
        have node := mem_predsOf_mem_nodes structural hu
        have lt := countAbove_lt_of_level_succ c.graph node (predsOf_props structural hu).1
        exact ih (countAbove c.graph u) (by omega) u node rfl)
      exact (productionAtOccurrenceB_setEq c prod).trans
        (witnessableRoutes_congr c.graph v raw)
  intro v hv
  exact key (countAbove c.graph v) v hv rfl



theorem localExpectedOccurrence_setEq_guardFree (d : Graph) (c : OccurrenceCertificate)
    (expanded : ExpandedCertOccurrence d c) (localChecks : LocalChecksOccurrence d c) :
    ∀ v ∈ d.NODES,
      ListSetEq (localExpectedOccurrence c v)
        (LambdaFlow.guardFreeFlowAt d (stdFuel d) v) := by
  have graph : c.graph = d := localChecks.1
  subst d
  have structural : structuralValid c.graph := localChecks.2.2.2.2.2.1
  intro v hv
  have shape : localShape c.graph v ≠ .invalid := by
    intro bad
    have prod := localChecks.2.1 v hv
    simp [productionAtOccurrenceB, bad] at prod
  apply localExpectedOccurrence_setEq_guardFree_of_preds c v structural shape
  intro u hu
  exact storedRoutes_setEq_guardFree c.graph c expanded localChecks u
    (mem_predsOf_mem_nodes structural hu)

theorem root_has_empty_guardFreeRoute (d : Graph) (c : OccurrenceCertificate)
    (expanded : ExpandedCertOccurrence d c) (localChecks : LocalChecksOccurrence d c)
    {v : Vertex} (node : v ∈ d.NODES)
    (root : (get_rule.outgoing v d).isEmpty = true) :
    ([], []) ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v := by
  have nodeC : v ∈ c.graph.NODES := by simpa [localChecks.1] using node
  have rootC : (get_rule.outgoing v c.graph).isEmpty = true := by
    simpa [localChecks.1] using root
  obtain ⟨output, outputMem, outputNode, outputDep, outputResidual⟩ :=
    (localChecks.2.2.2.2.1 v nodeC rootC).1
  have stored : ([], []) ∈ StoredRoutes c v := by
    unfold StoredRoutes outputsAt
    rw [List.mem_eraseDups]
    apply List.mem_append_right
    rw [List.mem_filterMap]
    exact ⟨output, outputMem, by simp [outputNode, outputDep, outputResidual]⟩
  exact (List.mem_filter.mp
    ((storedRoutes_setEq_guardFree d c expanded localChecks v node ([], [])).mp stored)).1

theorem strongFormationValidOccurrence_guardFreeRootsB (d : Graph) (c : OccurrenceCertificate)
    (valid : StrongFormationValidOccurrence d c) : LambdaFlow.guardFreeRootsB d = true := by
  obtain ⟨expanded, localChecks, universal⟩ := valid
  have raw := localExpectedOccurrence_setEq_guardFree d c expanded localChecks
  rw [LambdaFlow.guardFreeRootsB_eq_raw, List.all_eq_true]
  intro v node
  by_cases root : (get_rule.outgoing v d).isEmpty = true
  · simp only [root, Bool.not_true, Bool.false_or, Bool.and_eq_true, List.all_eq_true]
    constructor
    · have member := root_has_empty_guardFreeRoute d c expanded localChecks node root
      have nonempty : LambdaFlow.guardFreeFlowAt d (stdFuel d) v ≠ [] := by
        intro empty
        simp [empty] at member
      simpa using nonempty
    · intro route routeMem
      have nodeC : v ∈ c.graph.NODES := by simpa [localChecks.1] using node
      have rootC : (get_rule.outgoing v c.graph).isEmpty = true := by
        simpa [localChecks.1] using root
      obtain ⟨storedEmpty, expectedEmpty⟩ := universal v nodeC rootC
      have residual := expectedEmpty route ((raw v node route).mpr routeMem)
      have witnessable : route ∈ WitnessableRoutes d v
          (LambdaFlow.guardFreeFlowAt d (stdFuel d) v) := by
        simp [WitnessableRoutes, routeMem, routeContinuableAtB, root, residual]
      have stored := (storedRoutes_setEq_guardFree d c expanded localChecks
        v node route).mpr witnessable
      obtain ⟨dependency, residual'⟩ := storedEmpty route stored
      simp [dependency, residual']
  · have nonroot := Bool.eq_false_of_not_eq_true root
    simp [nonroot]

theorem strongFormationValidOccurrenceB_guardFreeRootsB (d : Graph) (c : OccurrenceCertificate)
    (checked : strongFormationValidOccurrenceB d c = true) : LambdaFlow.guardFreeRootsB d = true :=
  strongFormationValidOccurrence_guardFreeRootsB d c (strongFormationValidOccurrenceB_sound d c checked)

theorem strongFormationValidOccurrence_keyed (d : Graph) (c : OccurrenceCertificate)
    (valid : StrongFormationValidOccurrence d c) : LambdaKeyed.canonicalAcceptedB d = true := by
  rw [FaithfulKeyed.canonicalAcceptedB_eq_guardFreeRootsB]
  exact strongFormationValidOccurrence_guardFreeRootsB d c valid

theorem strongFormationValidOccurrenceB_keyed (d : Graph) (c : OccurrenceCertificate)
    (checked : strongFormationValidOccurrenceB d c = true) : LambdaKeyed.canonicalAcceptedB d = true :=
  strongFormationValidOccurrence_keyed d c (strongFormationValidOccurrenceB_sound d c checked)

#print axioms localExpectedOccurrence_setEq_guardFree_of_preds
#print axioms storedRoutes_setEq_guardFree
#print axioms localExpectedOccurrence_setEq_guardFree
#print axioms root_has_empty_guardFreeRoute
#print axioms strongFormationValidOccurrence_guardFreeRootsB
#print axioms strongFormationValidOccurrenceB_guardFreeRootsB
#print axioms strongFormationValidOccurrence_keyed
#print axioms strongFormationValidOccurrenceB_keyed

end OccurrenceCertificate
