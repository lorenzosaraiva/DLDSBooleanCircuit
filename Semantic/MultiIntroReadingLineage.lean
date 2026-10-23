import Semantic.MultiIntroReadings
import Semantic.OccurrenceReadingsLineageProjection
import Semantic.MultiIntroGridSupport
import Semantic.MultiIntroIdentity

open Semantic FlowSpec SelfCertifying SelfCertifyingProbe RouteIndexedMultiToken SpineReadings
namespace MultiIntroReadingLineage

def ordinary (d : Graph) (v : Vertex) (F : Vertex → List RouteIndexedToken)
    (edge : Deduction) : List RouteIndexedToken :=
  match v.FORMULA with
  | .implication alpha beta =>
    if edge.START.FORMULA == beta then
      (F edge.START).filterMap fun token =>
        if SingleIntroductionReference.guard d token.dependency edge then
          (consume edge token.residual).map fun tail =>
            mkToken d v (.introduction v edge token.routeKey)
              (depRemove token.dependency alpha) tail
        else none
    else []
  | _ => []

def reseed (variant : MultiIntro.Variant) (d : Graph) (v : Vertex)
    (F : Vertex → List RouteIndexedToken) (edge : Deduction) : List RouteIndexedToken :=
  match v.FORMULA with
  | .implication alpha beta =>
    if edge.START.FORMULA == beta then
      (F edge.START).flatMap fun token =>
        if SingleIntroductionReference.guard d token.dependency edge &&
            MultiIntroReadings.reseedTest variant token.flowPair edge then
          (eligibleAncestorsAtRoot d v).map fun ancestor =>
            mkToken d v (.introReseed v edge ancestor token.routeKey)
              (depRemove token.dependency alpha) ancestor.COLOURS
        else []
    else []
  | _ => []

def rawStepFrom (variant : MultiIntro.Variant) (d : Graph)
    (F : Vertex → List RouteIndexedToken) (v : Vertex) : List RouteIndexedToken :=
  if MultiIntro.multiIntroShapeB d v then
    (get_rule.incoming v d).flatMap fun edge => ordinary d v F edge ++ reseed variant d v F edge
  else TaggedLineage.rawStepFrom d F v

def stepFrom (variant : MultiIntro.Variant) (d : Graph)
    (F : Vertex → List RouteIndexedToken) (v : Vertex) : List RouteIndexedToken :=
  assignSlots v (rawStepFrom variant d F v)

def evaluateAt (variant : MultiIntro.Variant) (d : Graph) : Nat → Vertex → List RouteIndexedToken
  | 0, _ => []
  | fuel+1, v => stepFrom variant d (evaluateAt variant d fuel) v

theorem shape_facts (d : Graph) (v : Vertex)
    (shape : MultiIntro.multiIntroShapeB d v = true) :
    sourceB d v = false ∧ (elimPairsAt d v).isEmpty = true := by
  cases formula : v.FORMULA with
  | atom s => simp [MultiIntro.multiIntroShapeB, formula] at shape
  | implication alpha beta =>
    have parts : v.HYPOTHESIS = false ∧
        (((elimPairsAt d v).isEmpty = true ∧ 2 ≤ (get_rule.incoming v d).length) ∧
          (get_rule.incoming v d).all (fun e => e.START.FORMULA == beta) = true) := by
      simpa only [MultiIntro.multiIntroShapeB, formula, Bool.and_eq_true,
        Bool.not_eq_true', decide_eq_true_eq] using shape
    obtain ⟨edge, member⟩ := List.exists_mem_of_ne_nil _ (MultiIntro.shape_has_incoming d v shape)
    have pred := MultiIntro.incoming_start_pred d v edge member
    have nonempty : (predsOf d v).isEmpty = false := by
      cases h : predsOf d v with
      | nil => simp [h] at pred
      | cons head rest => rfl
    exact ⟨by simp [sourceB, parts.1, nonempty], parts.2.1.1⟩

theorem not_multi_of_source (d : Graph) (v : Vertex) (source : sourceB d v = true) :
    MultiIntro.multiIntroShapeB d v = false := by
  cases h : MultiIntro.multiIntroShapeB d v with
  | false => rfl
  | true => have contrary := (shape_facts d v h).1; simp [source] at contrary

theorem not_multi_of_pairs (d : Graph) (v : Vertex)
    (pairs : (elimPairsAt d v).isEmpty = false) : MultiIntro.multiIntroShapeB d v = false := by
  cases h : MultiIntro.multiIntroShapeB d v with
  | false => rfl
  | true => have contrary := (shape_facts d v h).2; simp [pairs] at contrary

theorem edge_incoming (d : Graph) (u v : Vertex) (edge : Deduction)
    (member : edge ∈ edgesBetween d u v) : edge ∈ get_rule.incoming v d := by
  have parts : edge ∈ d.EDGES ∧ edge.START = u ∧ edge.END = v := by
    simpa only [edgesBetween, List.mem_filter, decide_eq_true_eq] using member
  rw [incoming_eq_filter, List.mem_filter]
  exact ⟨parts.1, by simpa using parts.2.2⟩

theorem incoming_between (d : Graph) (v : Vertex) (edge : Deduction)
    (member : edge ∈ get_rule.incoming v d) : edge ∈ edgesBetween d edge.START v := by
  simp only [edgesBetween, List.mem_filter, decide_eq_true_eq]
  exact ⟨mem_incoming_mem_edges v d member, trivial, mem_incoming_end_eq v d member⟩

theorem project_ordinary (d : Graph) (v : Vertex) (F : Vertex → List RouteIndexedToken)
    (edge : Deduction) :
    projectFlowPairs (ordinary d v F edge) =
      MultiIntro.ordinaryEdge d v (fun u => projectFlowPairs (F u)) edge := by
  cases formula : v.FORMULA with
  | atom s => simp [ordinary, MultiIntro.ordinaryEdge, formula, projectFlowPairs]
  | implication alpha beta =>
    simp only [ordinary, MultiIntro.ordinaryEdge, formula]
    split
    · simp only [projectFlowPairs, List.map_filterMap, List.filterMap_map]
      congr 1
      funext token
      split <;> simp_all [mkToken, RouteIndexedToken.flowPair, Option.map_map, Function.comp_def]
    · rfl

theorem project_reseed (variant : MultiIntro.Variant) (d : Graph) (v : Vertex)
    (F : Vertex → List RouteIndexedToken) (edge : Deduction) :
    projectFlowPairs (reseed variant d v F edge) =
      MultiIntro.reseedEdge variant d v (fun u => projectFlowPairs (F u)) edge := by
  cases formula : v.FORMULA with
  | atom s => simp [reseed, MultiIntro.reseedEdge, formula, projectFlowPairs]
  | implication alpha beta =>
    simp only [reseed, MultiIntro.reseedEdge, formula]
    split
    · simp only [projectFlowPairs, List.map_flatMap, List.flatMap_map]
      congr 1
      funext token
      cases variant <;> split <;>
        simp_all [MultiIntroReadings.reseedTest, List.map_map, Function.comp_def,
          mkToken, RouteIndexedToken.flowPair]
    · rfl

theorem project_step (variant : MultiIntro.Variant) (d : Graph)
    (F : Vertex → List RouteIndexedToken) (v : Vertex) :
    ListSetEq (projectFlowPairs (stepFrom variant d F v))
      (MultiIntro.step variant d (fun u => projectFlowPairs (F u)) v) := by
  rw [stepFrom, project_assignSlots]
  unfold rawStepFrom MultiIntro.step
  split
  · simp only [projectFlowPairs, List.map_flatMap, List.map_append]
    have same : ((get_rule.incoming v d).flatMap fun e =>
        projectFlowPairs (ordinary d v F e) ++ projectFlowPairs (reseed variant d v F e)) =
        ((get_rule.incoming v d).flatMap fun e =>
        MultiIntro.ordinaryEdge d v (fun u => projectFlowPairs (F u)) e ++
        MultiIntro.reseedEdge variant d v (fun u => projectFlowPairs (F u)) e) := by
      simp only [project_ordinary, project_reseed]
    change ListSetEq ((get_rule.incoming v d).flatMap fun e =>
      projectFlowPairs (ordinary d v F e) ++ projectFlowPairs (reseed variant d v F e)) _
    rw [same]
    intro route
    simp [projectFlowPairs]
  · simpa only [TaggedLineage.stepFrom, project_assignSlots, CompleteProfileKernel.tagged] using
      TaggedLineage.project_stepFrom_self d F v

theorem evaluateAt_project (variant : MultiIntro.Variant) (d : Graph) : ∀ fuel v,
    ListSetEq (projectFlowPairs (evaluateAt variant d fuel v))
      (MultiIntro.evaluateAt variant d fuel v) := by
  intro fuel
  induction fuel with
  | zero => intro v; exact ListSetEq.refl []
  | succ fuel ih =>
    intro v
    exact (project_step variant d (evaluateAt variant d fuel) v).trans
      (MultiIntro.step_setEq variant d _ _ ih v)

end MultiIntroReadingLineage
