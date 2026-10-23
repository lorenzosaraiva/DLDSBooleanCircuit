import Semantic.Circuit.ProductionPlan
import Semantic.MultiIntroLocality

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 20000000
set_option maxRecDepth 30000

def productionBasis (d : Graph) (F : Nat) : Fin F → Formula :=
  let formulas := RefinedEval.formulas d
  fun i => formulas.getD i.val (.atom "")

def eligibleSeeds (d : Graph) (v : Vertex) : List (SeedWord d) :=
  (eligibleAncestorsAtRoot d v).attach.map fun a =>
    ⟨a.val.COLOURS, FlowSize.residual_member d _
      (FlowSize.ancestral_allowed d a.val (FlowSize.eligible_mem d v a.val a.property))⟩

theorem eligibleSeeds_map (d : Graph) (v : Vertex) {α : Type} (f : ColourPath → α) :
    (eligibleSeeds d v).map (fun w => f w.val) = (eligibleAncestorsAtRoot d v).map (fun a => f a.COLOURS) := by
  simp [eligibleSeeds, List.map_map]

def sourceRoutes (d : Graph) (v : Vertex) : List FlowPair :=
  if (ancestorsInto d v).isEmpty then [(normalizeDep [v.FORMULA], [])]
  else (ancestorsInto d v).map fun a => (normalizeDep [v.FORMULA], a.COLOURS)

theorem sourceRoutes_fit {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (supported : ∃ i, basis i = v.FORMULA) : ∀ r ∈ sourceRoutes d v, RouteFits basis (graphWords d) r := by
  intro r member
  have singletonFit : ∀ word, word ∈ [] :: graphWords d →
      RouteFits basis (graphWords d) (normalizeDep [v.FORMULA], word) := by
    intro word contained
    refine ⟨normalizeDep_canonical _, ?_, contained⟩
    intro f present
    have equal : f = v.FORMULA := by simpa using present
    simpa [equal] using supported
  unfold sourceRoutes at member
  split at member
  · have equal : r = (normalizeDep [v.FORMULA], []) := by simpa using member
    subst r
    exact singletonFit [] (by simp)
  · obtain ⟨a, contained, rfl⟩ := List.mem_map.mp member
    exact singletonFit a.COLOURS (FlowSize.residual_member d _
      (FlowSize.ancestral_allowed d a (FlowSize.ancestor_mem d v a contained)))

def introPlan {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (seeds : Option (List (SeedWord d))) (zeroOnly : Bool) : ProductionPlan d basis :=
  match predsOf d v, v.FORMULA with
  | [u], .implication alpha beta =>
      if u.FORMULA == beta then .unary (edgesBetween d u v) alpha seeds zeroOnly (.input u) else .empty
  | _, _ => .empty

def eliminationPlan {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (seeds : Option (List (SeedWord d))) : ProductionPlan d basis :=
  ProductionPlan.join ((elimPairsAt d v).map fun (major, minor) =>
    .binary (edgesBetween d minor v) (edgesBetween d major v) seeds (.input minor) (.input major))

def mixedPlan {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) : ProductionPlan d basis :=
  match v.FORMULA with
  | .implication alpha beta => ProductionPlan.join ((predsOf d v).map fun u =>
      if !elimPremiseAtB d v u && u.FORMULA == beta then
        .unary (edgesBetween d u v) alpha none false (.input u) else .empty)
  | _ => .empty

def reseedPlan {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) : ProductionPlan d basis :=
  if (eligibleAncestorsAtRoot d v).isEmpty then .empty else
    .dedup (.append (eliminationPlan basis d v (some (eligibleSeeds d v)))
      (introPlan basis d v (some (eligibleSeeds d v)) true))

def multiPlan {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) : ProductionPlan d basis :=
  .dedup (ProductionPlan.join ((get_rule.incoming v d).map fun e =>
    match v.FORMULA with
    | .implication alpha beta => if e.START.FORMULA == beta then
        .append (.unary [e] alpha none false (.input e.START))
          (.unary [e] alpha (some (eligibleSeeds d v)) false (.input e.START)) else .empty
    | _ => .empty))

def productionPlan {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) : Option (ProductionPlan d basis) :=
  if MultiIntro.multiIntroShapeB d v then some (multiPlan basis d v)
  else if v.HYPOTHESIS || (predsOf d v).isEmpty then
    if support : ∃ i, basis i = v.FORMULA then
      some (.constants (sourceRoutes d v) (sourceRoutes_fit basis d v support))
    else none
  else some (.dedup (.append
    (if (elimPairsAt d v).isEmpty then introPlan basis d v none false
     else .append (eliminationPlan basis d v none) (mixedPlan basis d v))
    (reseedPlan basis d v)))

theorem unaryMeaning_ordinary (d : Graph) (edges : List Deduction) (alpha : Formula) (r : FlowPair) :
    unaryMeaning (d := d) edges alpha none false r = edges.filterMap (fun edge =>
      if SingleIntroductionReference.guard d r.1 edge then
        (consume edge r.2).map fun tail => (depRemove r.1 alpha, tail) else none) := by
  simp [unaryMeaning, seedModes, unaryRoute, ← List.filterMap_eq_flatMap_toList]

@[simp] theorem flatMap_conditional_some {α β : Type} (values : List α) (condition : Prop)
    [Decidable condition] (f : α → β) :
    values.flatMap (fun x => (if condition then some (f x) else none).toList) =
      if condition then values.map f else [] := by
  by_cases h : condition <;> simp [h, ← List.map_eq_flatMap]

theorem unaryMeaning_reseed (d : Graph) (edges : List Deduction) (alpha : Formula)
    (seeds : List (SeedWord d)) (zero : Bool) (r : FlowPair) :
    unaryMeaning edges alpha (some seeds) zero r = edges.flatMap (fun edge =>
      if SingleIntroductionReference.guard d r.1 edge && (if zero then edge.COLOUR == 0 else true) &&
          (consume edge r.2).isSome then seeds.map (fun word => (depRemove r.1 alpha, word.val)) else []) := by
  unfold unaryMeaning
  apply List.flatMap_congr
  intro edge _
  cases consumed : consume edge r.2 <;>
    simp [seedModes, unaryRoute, List.flatMap_map, consumed, ← List.map_eq_flatMap]

theorem binaryMeaning_ordinary (d : Graph) (minors majors : List Deduction) (a b : FlowPair) :
    binaryMeaning (d := d) minors majors none a b = minors.flatMap (fun minor => majors.filterMap fun major =>
      if SingleIntroductionReference.guard d a.1 minor && SingleIntroductionReference.guard d b.1 major then
        match consume minor a.2, consume major b.2 with
        | some p, some q => if p = q then some (depUnion a.1 b.1, p) else none
        | _, _ => none
      else none) := by
  simp [binaryMeaning, seedModes, binaryRoute, ← List.filterMap_eq_flatMap_toList]
  rfl

theorem binaryMeaning_reseed (d : Graph) (minors majors : List Deduction) (seeds : List (SeedWord d))
    (a b : FlowPair) :
    binaryMeaning minors majors (some seeds) a b = minors.flatMap (fun minor => majors.flatMap fun major =>
      if reseedPremiseAgreesB a minor && reseedPremiseAgreesB b major &&
          SingleIntroductionReference.guard d a.1 minor && SingleIntroductionReference.guard d b.1 major then
        seeds.map (fun word => (depUnion a.1 b.1, word.val)) else []) := by
  unfold binaryMeaning
  apply List.flatMap_congr
  intro minor _
  apply List.flatMap_congr
  intro major _
  cases ac : consume minor a.2 <;> cases bc : consume major b.2 <;>
    simp [seedModes, binaryRoute, List.flatMap_map, ac, bc, reseedPremiseAgreesB,
      Bool.and_assoc, ← List.map_eq_flatMap]

theorem introPlan_ordinary {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (introPlan basis d v none false).meaning prior =
      match predsOf d v, v.FORMULA with
      | [u], .implication alpha beta => if u.FORMULA = beta then
          (prior u).flatMap (fun r => (edgesBetween d u v).filterMap fun e =>
            if SingleIntroductionReference.guard d r.1 e then
              (consume e r.2).map (fun tail => (depRemove r.1 alpha, tail)) else none) else []
      | _, _ => [] := by
  unfold introPlan
  split <;> simp_all [ProductionPlan.meaning, ProductionPlan.empty, unaryMeaning_ordinary]

theorem introPlan_reseed {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (introPlan basis d v (some (eligibleSeeds d v)) true).meaning prior =
      LambdaFlow.introReseedWith d (SingleIntroductionReference.guard d) v prior (eligibleAncestorsAtRoot d v) := by
  unfold introPlan LambdaFlow.introReseedWith
  split <;> simp_all [ProductionPlan.meaning, ProductionPlan.empty, unaryMeaning_reseed, eligibleSeeds_map,
    reseedZeroPremiseB, Bool.and_assoc, Bool.and_comm, Bool.and_left_comm]

theorem eliminationPlan_ordinary {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (eliminationPlan basis d v none).meaning prior = (elimPairsAt d v).flatMap (fun (major, minor) =>
      (prior minor).flatMap fun a => (prior major).flatMap fun b =>
        (edgesBetween d minor v).flatMap fun eMinor => (edgesBetween d major v).filterMap fun eMajor =>
          if SingleIntroductionReference.guard d a.1 eMinor && SingleIntroductionReference.guard d b.1 eMajor then
            match consume eMinor a.2, consume eMajor b.2 with
            | some p, some q => if p = q then some (depUnion a.1 b.1, p) else none
            | _, _ => none
          else none) := by
  simp [eliminationPlan, ProductionPlan.meaning_join, List.flatMap_map, ProductionPlan.meaning, binaryMeaning_ordinary]

theorem eliminationPlan_reseed {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (eliminationPlan basis d v (some (eligibleSeeds d v))).meaning prior =
      LambdaFlow.elimReseedWith d (SingleIntroductionReference.guard d) v prior (eligibleAncestorsAtRoot d v) := by
  simp [eliminationPlan, ProductionPlan.meaning_join, List.flatMap_map, ProductionPlan.meaning,
    binaryMeaning_reseed, eligibleSeeds_map, LambdaFlow.elimReseedWith]

theorem mixedPlan_meaning {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (mixedPlan basis d v).meaning prior = LambdaFlow.mixedWith d (SingleIntroductionReference.guard d) v prior := by
  unfold mixedPlan LambdaFlow.mixedWith
  split <;> simp_all [ProductionPlan.meaning_join, List.flatMap_map, ProductionPlan.meaning,
    ProductionPlan.empty, unaryMeaning_ordinary]

theorem reseedPlan_meaning {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (reseedPlan basis d v).meaning prior = LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v prior := by
  unfold reseedPlan LambdaFlow.reseedWith
  split <;> simp_all [ProductionPlan.meaning, ProductionPlan.empty, eliminationPlan_reseed, introPlan_reseed]

theorem multiPlan_meaning {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (prior : Vertex → List FlowPair) :
    (multiPlan basis d v).meaning prior = ((get_rule.incoming v d).flatMap fun e =>
      MultiIntro.ordinaryEdge d v prior e ++ MultiIntro.reseedEdge .general d v prior e).eraseDups := by
  unfold multiPlan
  simp only [ProductionPlan.meaning, ProductionPlan.meaning_join, List.flatMap_map]
  congr 1
  apply List.flatMap_congr
  intro e _
  cases formula : v.FORMULA <;> simp [formula, MultiIntro.ordinaryEdge, MultiIntro.reseedEdge,
    ProductionPlan.meaning, ProductionPlan.empty, unaryMeaning_ordinary, unaryMeaning_reseed,
    eligibleSeeds_map, reseedPremiseAgreesB, Bool.and_assoc, List.filterMap_eq_flatMap_toList] <;>
      split_ifs <;> rfl

theorem productionPlan_meaning {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (plan : ProductionPlan d basis) (built : productionPlan basis d v = some plan)
    (prior : Vertex → List FlowPair) : plan.meaning prior = MultiIntro.step .general d prior v := by
  unfold productionPlan at built
  split at built
  · cases built
    simpa only [MultiIntro.step, if_pos (by assumption)] using multiPlan_meaning basis d v prior
  · rename_i noMulti
    split at built
    · rename_i source
      split at built
      · cases built
        simp [ProductionPlan.meaning, MultiIntro.step, noMulti, CompleteProfileKernel.tagged,
          LambdaFlow.stepWith, source, sourceRoutes]
      · contradiction
    · rename_i notSource
      cases built
      simp [ProductionPlan.meaning, MultiIntro.step, noMulti, CompleteProfileKernel.tagged,
        LambdaFlow.stepWith, notSource, introPlan_ordinary, eliminationPlan_ordinary,
        mixedPlan_meaning, reseedPlan_meaning] <;> rfl

end Semantic.Circuit
