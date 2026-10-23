import Semantic.SingleIntroductionReference
import Semantic.DependencyGuardStability

open Semantic FlowSpec SelfCertifyingProbe

namespace CompleteProfileKernel


structure Kernel where
  step : Graph → (Vertex → List FlowPair) → Vertex → List FlowPair
  congr : ∀ d v left right,
    (∀ u ∈ predsOf d v, left u = right u) → step d left v = step d right v

def evaluateAt (kernel : Kernel) (d : Graph) : Nat → Vertex → List FlowPair
  | 0, _ => []
  | fuel + 1, v => kernel.step d (evaluateAt kernel d fuel) v

def lookup (rows : FlowTable) (v : Vertex) : List FlowPair :=
  (flowTableLookup? rows v).getD []

def specificationTable (kernel : Kernel) (d : Graph) (fuel : Nat) : FlowTable :=
  d.NODES.map fun v => (v, evaluateAt kernel d fuel v)

def memoAt (kernel : Kernel) (d : Graph) : Nat → FlowTable
  | 0 => d.NODES.map fun v => (v, [])
  | fuel + 1 => d.NODES.map fun v => (v, kernel.step d (lookup (memoAt kernel d fuel)) v)

def rootsB (kernel : Kernel) (d : Graph) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let routes := evaluateAt kernel d (stdFuel d) v
     !routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty)

def Valid (kernel : Kernel) (d : Graph) : Prop :=
  structuralValid d ∧ LeveledColored d ∧ rootsB kernel d = true

def tagged : Kernel where
  step d := LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
  congr d v left right := LambdaFlow.stepWith_congr_of_preds d
    (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) v left right

theorem tagged_evaluate (d : Graph) : ∀ fuel v,
    evaluateAt tagged d fuel v = SingleIntroductionReference.routes d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
    intro v
    change tagged.step d (evaluateAt tagged d fuel) v =
      tagged.step d (SingleIntroductionReference.routes d fuel) v
    rw [funext ih]

theorem tagged_roots (d : Graph) : rootsB tagged d = SingleIntroductionReference.rootsB d := by
  simp only [rootsB, SingleIntroductionReference.rootsB, tagged_evaluate]

theorem tagged_valid (d : Graph) : Valid tagged d ↔ SingleIntroductionReference.ReferenceVerdict d := by
  simp only [Valid, SingleIntroductionReference.ReferenceVerdict, tagged_roots]

theorem lookup_spec (kernel : Kernel) (d : Graph) (fuel : Nat)
    {v : Vertex} (member : v ∈ d.NODES) :
    lookup (specificationTable kernel d fuel) v = evaluateAt kernel d fuel v := by
  simp [lookup, specificationTable,
    CacheCorrectness.flowTableLookup?_map_of_mem d.NODES _ member]

theorem memoAt_eq_spec (kernel : Kernel) (d : Graph)
    (structural : structuralValid d) : ∀ fuel,
    memoAt kernel d fuel = specificationTable kernel d fuel := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ fuel ih =>
    simp only [memoAt, ih, specificationTable]
    apply List.map_congr_left
    intro v hv
    apply congrArg (Prod.mk v)
    apply kernel.congr
    intro u hu
    exact lookup_spec kernel d fuel (mem_predsOf_mem_nodes structural hu)


theorem evaluateAt_succ_stable (kernel : Kernel) (d : Graph)
    (structural : structuralValid d) :
    ∀ v ∈ d.NODES, ∀ fuel, countAbove d v < fuel →
      evaluateAt kernel d (fuel + 1) v = evaluateAt kernel d fuel v := by
  have key : ∀ n, ∀ v ∈ d.NODES, countAbove d v = n →
      ∀ fuel, countAbove d v < fuel →
      evaluateAt kernel d (fuel + 1) v = evaluateAt kernel d fuel v := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro v hv hn fuel hf
      have hf1 : fuel = (fuel - 1) + 1 := by omega
      rw [evaluateAt]
      conv_rhs => rw [hf1, evaluateAt]
      apply kernel.congr
      intro u hu
      have humem := mem_predsOf_mem_nodes structural hu
      have hulvl := (predsOf_props structural hu).1
      have huca := countAbove_lt_of_level_succ d humem hulvl
      have stable := ih (countAbove d u) (by omega) u humem rfl
        (fuel - 1) (by omega)
      rw [hf1]
      exact stable
  intro v hv fuel hf
  exact key (countAbove d v) v hv rfl fuel hf

theorem standard_fixed (kernel : Kernel) (d : Graph)
    (structural : structuralValid d) {v : Vertex} (member : v ∈ d.NODES) :
    kernel.step d (evaluateAt kernel d (stdFuel d)) v =
      evaluateAt kernel d (stdFuel d) v := by
  exact evaluateAt_succ_stable kernel d structural v member (stdFuel d)
    (by have count := countAbove_le d v; unfold stdFuel; omega)


theorem fixed_eq_evaluate (kernel : Kernel) (d : Graph)
    (structural : structuralValid d) (stored : Vertex → List FlowPair)
    (fixed : ∀ v ∈ d.NODES, stored v = kernel.step d stored v) :
    ∀ fuel v, v ∈ d.NODES → countAbove d v < fuel →
      stored v = evaluateAt kernel d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v hv bound; omega
  | succ fuel ih =>
    intro v hv bound
    rw [fixed v hv, evaluateAt]
    apply kernel.congr
    intro u hu
    have humem := mem_predsOf_mem_nodes structural hu
    have hulvl := (predsOf_props structural hu).1
    have less := countAbove_lt_of_level_succ d humem hulvl
    exact ih u humem (by omega)

theorem rootsB_iff (kernel : Kernel) (d : Graph) : rootsB kernel d = true ↔
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      evaluateAt kernel d (stdFuel d) root ≠ [] ∧
      ∀ route ∈ evaluateAt kernel d (stdFuel d) root,
        route.1 = [] ∧ route.2 = [] := by
  simp only [rootsB, List.all_eq_true, Bool.or_eq_true, Bool.and_eq_true,
    Bool.not_eq_true', List.isEmpty_eq_false_iff, List.isEmpty_iff]
  constructor
  · intro checked root member terminal
    exact (checked root member).resolve_left (by simp [terminal])
  · intro checked root member
    by_cases terminal : get_rule.outgoing root d = []
    · exact Or.inr (checked root member terminal)
    · exact Or.inl (by simpa using terminal)

#print axioms fixed_eq_evaluate
#print axioms memoAt_eq_spec
#print axioms tagged_evaluate
end CompleteProfileKernel
