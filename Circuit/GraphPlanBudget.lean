import Circuit.PlanAllocation
import Circuit.GraphLoops

open FlowSpec
namespace Semantic.Circuit.Shared
set_option maxHeartbeats 50000000

theorem planBudget_mono {F : Nat} {d : Graph} {basis : Fin F → Formula}
    (plan : ProductionPlan d basis) (a b : Vertex → Nat) (le : ∀ v, a v ≤ b v) :
    (planBudget a plan).1 ≤ (planBudget b plan).1 ∧ (planBudget a plan).2 ≤ (planBudget b plan).2 := by
  induction plan with
  | input v => exact ⟨le v, le_rfl⟩
  | constants routes fits => exact ⟨le_rfl, le_rfl⟩
  | unary edges formula seeds zero child ih =>
    simp only [planBudget]
    constructor <;> gcongr <;> first | exact ih.1 | exact ih.2
  | binary minors majors seeds left right hl hr =>
    simp only [planBudget]
    constructor <;> gcongr <;> first | exact hl.1 | exact hr.1 | exact hl.2 | exact hr.2
  | append left right hl hr => exact ⟨Nat.add_le_add hl.1 hr.1, Nat.add_le_add hl.2 hr.2⟩
  | dedup child ih =>
    simp only [planBudget]
    exact ⟨ih.1, Nat.add_le_add ih.2 (Nat.mul_le_mul ih.1 ih.1)⟩

theorem budget_join {F : Nat} {d : Graph} {basis : Fin F → Formula}
    (inputs : Vertex → Nat) (plans : List (ProductionPlan d basis)) :
    planBudget inputs (ProductionPlan.join plans) =
      ((plans.map fun p => (planBudget inputs p).1).sum,
       (plans.map fun p => (planBudget inputs p).2).sum) := by
  induction plans with
  | nil => simp [ProductionPlan.join, ProductionPlan.empty, planBudget]
  | cons head tail ih =>
    change ((planBudget inputs head).1 + (planBudget inputs (ProductionPlan.join tail)).1,
      (planBudget inputs head).2 + (planBudget inputs (ProductionPlan.join tail)).2) = _
    simp [ih]

theorem labels_join {F : Nat} {d : Graph} {basis : Fin F → Formula} (g : Nat)
    (plans : List (ProductionPlan d basis)) (each : ∀ p ∈ plans, PlanLabels g p) :
    PlanLabels g (ProductionPlan.join plans) := by
  induction plans with
  | nil => trivial
  | cons head tail ih => exact ⟨each head (by simp), ih (fun p hp => each p (by simp [hp]))⟩

theorem between_labels (d : Graph) (u v : Vertex) :
    ∀ e ∈ edgesBetween d u v, e.DEPENDENCY.length ≤ graphSize d := by
  intro e he
  exact le_trans (Nat.le_add_right _ 1) (graph_label_le d e (List.mem_filter.mp he).1)

theorem seedModes_graph_le (d : Graph) (v : Vertex) :
    (seedModes (some (eligibleSeeds d v))).length ≤ graphSize d := by
  simpa [seedModes] using le_trans (Nat.le_add_right (eligibleSeeds d v).length 1) (graph_seeds_le d v)

def productionMass (R g : Nat) : Nat := (R + 1) ^ 2 * (g + 1) ^ 4

theorem mass_dominates (R g : Nat) (positive : 0 < g) :
    g ≤ productionMass R g ∧ R * g * g ≤ productionMass R g ∧
      R * R * g * g * g * g ≤ productionMass R g := by
  dsimp [productionMass]
  constructor
  · have a : 1 ≤ (R + 1) ^ 2 := Nat.one_le_pow 2 (R + 1) (by omega)
    have b : g ≤ (g + 1) ^ 4 := (Nat.le_succ g).trans (Nat.le_self_pow (by decide : 4 ≠ 0) (g + 1))
    simpa using Nat.mul_le_mul a b
  constructor
  · have a : R ≤ (R + 1) ^ 2 := (Nat.le_succ R).trans (Nat.le_self_pow (by decide : 2 ≠ 0) (R + 1))
    have b : g * g ≤ (g + 1) ^ 4 := by
      calc
        g * g ≤ (g + 1) ^ 2 := by simpa [pow_two] using Nat.mul_le_mul (Nat.le_succ g) (Nat.le_succ g)
        _ ≤ ((g + 1) ^ 2) ^ 2 := Nat.le_self_pow (by decide : 2 ≠ 0) _
        _ = (g + 1) ^ 4 := by ring
    simpa [Nat.mul_assoc] using Nat.mul_le_mul a b
  · have a : R * R ≤ (R + 1) ^ 2 := by simpa [pow_two] using Nat.mul_le_mul (Nat.le_succ R) (Nat.le_succ R)
    have b : g * g * g * g ≤ (g + 1) ^ 4 := by simpa [pow_succ, pow_two, Nat.mul_assoc] using Nat.pow_le_pow_left (Nat.le_succ g) 4
    simpa [Nat.mul_assoc] using Nat.mul_le_mul a b

theorem intro_budget {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (R : Nat) (seeds : Option (List (SeedWord d))) (zero : Bool)
    (seedBound : (seedModes seeds).length ≤ graphSize d) :
    (planBudget (fun _ => R) (introPlan basis d v seeds zero)).1 ≤ productionMass R (graphSize d) ∧
    (planBudget (fun _ => R) (introPlan basis d v seeds zero)).2 =
      (planBudget (fun _ => R) (introPlan basis d v seeds zero)).1 ∧
    PlanLabels (graphSize d) (introPlan basis d v seeds zero) := by
  have mass := (mass_dominates R (graphSize d) (graphSize_positive d)).2.1
  unfold introPlan
  split
  · rename_i u alpha beta _ _
    split
    · have edges := graph_between_le d u v
      have count : R * ((edgesBetween d u v).length * (seedModes seeds).length) ≤ R * graphSize d * graphSize d := by
        have e : (edgesBetween d u v).length ≤ graphSize d := by omega
        simpa [Nat.mul_assoc] using Nat.mul_le_mul_left R (Nat.mul_le_mul e seedBound)
      exact ⟨count.trans mass, by simp [planBudget], between_labels d u v, trivial⟩
    · simp [ProductionPlan.empty, planBudget, PlanLabels]
  · simp [ProductionPlan.empty, planBudget, PlanLabels]

theorem elimination_budget {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex)
    (R : Nat) (seeds : Option (List (SeedWord d)))
    (seedBound : (seedModes seeds).length ≤ graphSize d) :
    (planBudget (fun _ => R) (eliminationPlan basis d v seeds)).1 ≤ productionMass R (graphSize d) ∧
    (planBudget (fun _ => R) (eliminationPlan basis d v seeds)).2 =
      (planBudget (fun _ => R) (eliminationPlan basis d v seeds)).1 ∧
    PlanLabels (graphSize d) (eliminationPlan basis d v seeds) := by
  have each : ∀ pair ∈ elimPairsAt d v,
      (planBudget (fun _ => R) (.binary (edgesBetween d pair.2 v) (edgesBetween d pair.1 v) seeds
        (.input pair.2) (.input pair.1) : ProductionPlan d basis)).1 ≤
        R * R * graphSize d * graphSize d * graphSize d := by
    intro pair _
    have a : (edgesBetween d pair.2 v).length ≤ graphSize d := by have := graph_between_le d pair.2 v; omega
    have b : (edgesBetween d pair.1 v).length ≤ graphSize d := by have := graph_between_le d pair.1 v; omega
    dsimp [planBudget]
    have h := Nat.mul_le_mul_left (R * R) (Nat.mul_le_mul (Nat.mul_le_mul a b) seedBound)
    simpa [Nat.mul_assoc] using h
  simp only [eliminationPlan, budget_join, List.map_map, Function.comp_def]
  constructor
  · have h := sum_map_bound (elimPairsAt d v) _ _ each
    have e : (elimPairsAt d v).length ≤ graphSize d := by have := graph_elimination_le d v; omega
    have t := Nat.mul_le_mul_right (R * R * graphSize d * graphSize d * graphSize d) e
    have mass := (mass_dominates R (graphSize d) (graphSize_positive d)).2.2
    apply le_trans h
    apply le_trans t
    simpa [Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using mass
  constructor
  · simp [planBudget]
  · apply labels_join
    intro p hp
    obtain ⟨pair, _, rfl⟩ := List.mem_map.mp hp
    exact ⟨between_labels d pair.2 v, between_labels d pair.1 v, trivial, trivial⟩

theorem mixed_budget {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) (R : Nat) :
    (planBudget (fun _ => R) (mixedPlan basis d v)).1 ≤ productionMass R (graphSize d) ∧
    (planBudget (fun _ => R) (mixedPlan basis d v)).2 =
      (planBudget (fun _ => R) (mixedPlan basis d v)).1 ∧
    PlanLabels (graphSize d) (mixedPlan basis d v) := by
  unfold mixedPlan
  split
  · rename_i alpha beta _
    let f : Vertex → ProductionPlan d basis := fun u => if !elimPremiseAtB d v u && u.FORMULA == beta then
        ProductionPlan.unary (basis := basis) (edgesBetween d u v) alpha none false (.input u) else .empty
    have plain : ∀ u, (planBudget (fun _ => R) (f u)).2 = (planBudget (fun _ => R) (f u)).1 := by
      intro u; dsimp [f]; split <;> simp [planBudget, ProductionPlan.empty]
    have each : ∀ u ∈ predsOf d v, (planBudget (fun _ => R) (f u)).1 ≤ R * graphSize d := by
      intro u _; dsimp [f]; split
      · simp only [planBudget, seedModes, List.length_singleton, Nat.mul_one]
        exact Nat.mul_le_mul_left R (by have := graph_between_le d u v; omega)
      · simp [ProductionPlan.empty, planBudget]
    constructor
    · rw [budget_join]
      simp only [List.map_map, Function.comp_def]
      have h := sum_map_bound (predsOf d v) _ _ each
      have p : (predsOf d v).length ≤ graphSize d := by have := graph_predecessors_le d v; omega
      have t := Nat.mul_le_mul_right (R * graphSize d) p
      have m := (mass_dominates R (graphSize d) (graphSize_positive d)).2.1
      exact h.trans (t.trans (by nlinarith))
    constructor
    · rw [budget_join]
      dsimp only
      simp only [List.map_map, Function.comp_def]
      apply congrArg List.sum
      apply List.map_congr_left
      intro u _
      exact plain u
    · apply labels_join
      intro p hp
      obtain ⟨u, _, rfl⟩ := List.mem_map.mp hp
      split
      · exact ⟨between_labels d u v, trivial⟩
      · trivial
  · simp [ProductionPlan.empty, planBudget, PlanLabels]

theorem multi_budget {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) (R : Nat) :
    (planBudget (fun _ => R) (multiPlan basis d v)).1 ≤ 2 * productionMass R (graphSize d) ∧
    (planBudget (fun _ => R) (multiPlan basis d v)).2 ≤
      2 * productionMass R (graphSize d) + (2 * productionMass R (graphSize d)) ^ 2 ∧
    PlanLabels (graphSize d) (multiPlan basis d v) := by
  let plans := (get_rule.incoming v d).map fun e =>
    match v.FORMULA with
    | .implication alpha beta => if e.START.FORMULA == beta then
        ProductionPlan.append (basis := basis) (.unary [e] alpha none false (.input e.START))
          (.unary [e] alpha (some (eligibleSeeds d v)) false (.input e.START)) else .empty
    | _ => .empty
  have each : ∀ p ∈ plans, (planBudget (fun _ => R) p).1 ≤ R * (1 + graphSize d) ∧
      (planBudget (fun _ => R) p).2 = (planBudget (fun _ => R) p).1 ∧ PlanLabels (graphSize d) p := by
    intro p hp
    obtain ⟨e, he, rfl⟩ := List.mem_map.mp hp
    split
    · split
      · have s := seedModes_graph_le d v
        have label := graph_incoming_label_le d v e he
        simp only [planBudget, List.length_singleton, seedModes, List.length_map, Nat.mul_one, Nat.one_mul,
          Nat.zero_add, PlanLabels, List.mem_singleton, forall_eq] at *
        exact ⟨by nlinarith, by tauto⟩
      · simp [ProductionPlan.empty, planBudget, PlanLabels]
    · simp [ProductionPlan.empty, planBudget, PlanLabels]
  have count : (planBudget (fun _ => R) (ProductionPlan.join plans)).1 ≤ 2 * productionMass R (graphSize d) := by
    rw [budget_join]
    have h := sum_map_bound plans (fun p => (planBudget (fun _ => R) p).1) _ (fun p hp => (each p hp).1)
    have len : plans.length ≤ graphSize d := by
      simp only [plans, List.length_map]
      have := graph_incoming_le d v; omega
    have t := Nat.mul_le_mul_right (R * (1 + graphSize d)) len
    have g := graphSize_positive d
    have m := (mass_dominates R (graphSize d) g).2.1
    exact h.trans (t.trans (by nlinarith))
  have plain : (planBudget (fun _ => R) (ProductionPlan.join plans)).2 =
      (planBudget (fun _ => R) (ProductionPlan.join plans)).1 := by
    rw [budget_join]
    dsimp only
    exact congrArg List.sum (List.map_congr_left (fun p hp => (each p hp).2.1))
  change (planBudget (fun _ => R) (ProductionPlan.join plans)).1 ≤ _ ∧
    (planBudget (fun _ => R) (ProductionPlan.join plans)).2 +
      (planBudget (fun _ => R) (ProductionPlan.join plans)).1 *
      (planBudget (fun _ => R) (ProductionPlan.join plans)).1 ≤ _ ∧ _
  refine ⟨count, ?_, labels_join _ _ (fun p hp => (each p hp).2.2)⟩
  rw [plain]
  nlinarith [Nat.mul_self_le_mul_self count]

theorem production_budget {F : Nat} (basis : Fin F → Formula) (d : Graph) (v : Vertex) (R : Nat)
    (plan : ProductionPlan d basis) (built : productionPlan basis d v = some plan) :
    (planBudget (fun _ => R) plan).1 ≤ 4 * productionMass R (graphSize d) ∧
    (planBudget (fun _ => R) plan).2 ≤
      4 * productionMass R (graphSize d) + 20 * productionMass R (graphSize d) ^ 2 ∧
    PlanLabels (graphSize d) plan := by
  have positive := graphSize_positive d
  have noneSeed : (seedModes (d := d) none).length ≤ graphSize d := by simpa [seedModes] using positive
  have someSeed := seedModes_graph_le d v
  have intro := intro_budget basis d v R none false noneSeed
  have elim := elimination_budget basis d v R none noneSeed
  have mixed := mixed_budget basis d v R
  have reseedIntro := intro_budget basis d v R (some (eligibleSeeds d v)) true someSeed
  have reseedElim := elimination_budget basis d v R (some (eligibleSeeds d v)) someSeed
  have reseed : (planBudget (fun _ => R) (reseedPlan basis d v)).1 ≤ 2 * productionMass R (graphSize d) ∧
      (planBudget (fun _ => R) (reseedPlan basis d v)).2 ≤
        2 * productionMass R (graphSize d) + 4 * productionMass R (graphSize d) ^ 2 ∧
      PlanLabels (graphSize d) (reseedPlan basis d v) := by
    unfold reseedPlan
    split
    · simp [ProductionPlan.empty, planBudget, PlanLabels]
    · simp only [planBudget, PlanLabels]
      have count := Nat.add_le_add reseedElim.1 reseedIntro.1
      refine ⟨by omega, ?_, reseedElim.2.2, reseedIntro.2.2⟩
      rw [reseedElim.2.1, reseedIntro.2.1]
      nlinarith [Nat.mul_self_le_mul_self count]
  unfold productionPlan at built
  split at built
  · cases built
    have h := multi_budget basis d v R
    exact ⟨by omega, h.2.1.trans (by nlinarith), h.2.2⟩
  · split at built
    · split at built
      · cases built
        have source := graph_source_le d v
        have mass := (mass_dominates R (graphSize d) positive).1
        exact ⟨by simpa [planBudget] using source.trans (by omega),
          by simpa [planBudget] using source.trans (by nlinarith), trivial⟩
      · contradiction
    · cases built
      have ordinary : ∀ p : ProductionPlan d basis,
          p = (if (elimPairsAt d v).isEmpty then introPlan basis d v none false
            else .append (eliminationPlan basis d v none) (mixedPlan basis d v)) →
          (planBudget (fun _ => R) p).1 ≤ 2 * productionMass R (graphSize d) ∧
          (planBudget (fun _ => R) p).2 = (planBudget (fun _ => R) p).1 ∧ PlanLabels (graphSize d) p := by
        intro p hp; subst p; split
        · exact ⟨by omega, intro.2⟩
        · exact ⟨by dsimp [planBudget]; omega,
            by simp only [planBudget]; rw [elim.2.1, mixed.2.1], elim.2.2, mixed.2.2⟩
      have h := ordinary _ rfl
      simp only [planBudget, PlanLabels]
      have count := Nat.add_le_add h.1 reseed.1
      refine ⟨by omega, ?_, h.2.2, reseed.2.2⟩
      rw [h.2.1]
      nlinarith [reseed.2.1, Nat.mul_self_le_mul_self count]

end Semantic.Circuit.Shared
