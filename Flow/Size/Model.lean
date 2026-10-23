import Flow.Core
import Flow.Grid.MultiIntroGridSupport
import Flow.Certificate.MultiIntro
import Mathlib.Data.List.Infix
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Finset.Prod

open Semantic FlowSpec

namespace FlowSize

set_option maxHeartbeats 3000000

def ResidualAllowed (d : Graph) (word : ColourPath) : Prop :=
  word = [] ∨ ∃ a ∈ d.PATHS, word <:+ a.COLOURS

def EdgeLabelDependent (d : Graph) (dependency : Dep) : Prop :=
  dependency = [] ∨ ∃ e ∈ d.EDGES, dependency = normalizeDep e.DEPENDENCY

theorem ancestor_mem (d : Graph) (v : Vertex) (a : Ancestral)
    (member : a ∈ ancestorsInto d v) : a ∈ d.PATHS :=
  (List.mem_filter.mp member).1

theorem eligible_mem (d : Graph) (v : Vertex) (a : Ancestral)
    (member : a ∈ eligibleAncestorsAtRoot d v) : a ∈ d.PATHS := by
  unfold eligibleAncestorsAtRoot at member
  split at member
  · contradiction
  · exact ancestor_mem d v a (List.mem_filter.mp member).1

theorem ancestral_allowed (d : Graph) (a : Ancestral) (member : a ∈ d.PATHS) :
    ResidualAllowed d a.COLOURS := Or.inr ⟨a, member, List.suffix_rfl⟩

theorem consume_suffix (e : Deduction) (word tail : ColourPath)
    (consumed : consume e word = some tail) : tail <:+ word := by
  cases word with
  | nil =>
      simp only [consume] at consumed
      split at consumed
      · cases consumed
        exact List.suffix_rfl
      · contradiction
  | cons head rest =>
      simp only [consume] at consumed
      split at consumed
      · cases consumed
        exact ⟨[head], rfl⟩
      · contradiction

theorem suffix_allowed (d : Graph) {word tail : ColourPath}
    (allowed : ResidualAllowed d word) (suffix : tail <:+ word) : ResidualAllowed d tail := by
  rcases allowed with rfl | ⟨a, member, ending⟩
  · exact Or.inl (List.suffix_nil.mp suffix)
  · exact Or.inr ⟨a, member, suffix.trans ending⟩

theorem consume_allowed (d : Graph) (e : Deduction) (word tail : ColourPath)
    (allowed : ResidualAllowed d word) (consumed : consume e word = some tail) :
    ResidualAllowed d tail := suffix_allowed d allowed (consume_suffix e word tail consumed)

private theorem elim_reseed (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (ancestors : List Ancestral) (contained : ∀ a ∈ ancestors, a ∈ d.PATHS)
    (r : FlowPair) (member : r ∈ LambdaFlow.elimReseedWith d (SingleIntroductionReference.guard d) v prior ancestors) :
    ResidualAllowed d r.2 := by
  simp only [LambdaFlow.elimReseedWith, List.mem_flatMap] at member
  obtain ⟨pair, _, left, _, right, _, eMinor, _, eMajor, _, produced⟩ := member
  split at produced
  · obtain ⟨a, ha, equal⟩ := List.mem_map.mp produced
    subst r
    exact ancestral_allowed d a (contained a ha)
  · contradiction

private theorem intro_reseed (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (ancestors : List Ancestral) (contained : ∀ a ∈ ancestors, a ∈ d.PATHS)
    (r : FlowPair) (member : r ∈ LambdaFlow.introReseedWith d (SingleIntroductionReference.guard d) v prior ancestors) :
    ResidualAllowed d r.2 := by
  unfold LambdaFlow.introReseedWith at member
  split at member
  · split at member
    · obtain ⟨input, _, produced⟩ := List.mem_flatMap.mp member
      obtain ⟨edge, _, produced⟩ := List.mem_flatMap.mp produced
      split at produced
      · obtain ⟨a, ha, equal⟩ := List.mem_map.mp produced
        subst r
        exact ancestral_allowed d a (contained a ha)
      · contradiction
    · contradiction
  · contradiction

private theorem reseed (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (r : FlowPair) (member : r ∈ LambdaFlow.reseedWith d (SingleIntroductionReference.guard d) v prior) :
    ResidualAllowed d r.2 := by
  unfold LambdaFlow.reseedWith at member
  dsimp only at member
  split at member
  · contradiction
  · rw [List.mem_eraseDups, List.mem_append] at member
    rcases member with h | h
    · exact elim_reseed d v prior _ (fun a ha => eligible_mem d v a ha) r h
    · exact intro_reseed d v prior _ (fun a ha => eligible_mem d v a ha) r h

private theorem mixed (d : Graph) (v : Vertex) (prior : Vertex → List FlowPair)
    (correct : ∀ u r, r ∈ prior u → ResidualAllowed d r.2)
    (r : FlowPair) (member : r ∈ LambdaFlow.mixedWith d (SingleIntroductionReference.guard d) v prior) :
    ResidualAllowed d r.2 := by
  unfold LambdaFlow.mixedWith at member
  split at member
  · obtain ⟨u, _, produced⟩ := List.mem_flatMap.mp member
    split at produced
    · obtain ⟨input, hi, output⟩ := List.mem_flatMap.mp produced
      obtain ⟨edge, _, output⟩ := List.mem_filterMap.mp output
      split at output
      · obtain ⟨tail, consumed, equal⟩ := Option.map_eq_some_iff.mp output
        cases equal
        exact consume_allowed d edge input.2 tail (correct u input hi) consumed
      · contradiction
    · contradiction
  · contradiction

theorem reference_step_residual (d : Graph) (prior : Vertex → List FlowPair)
    (correct : ∀ u r, r ∈ prior u → ResidualAllowed d r.2) (v : Vertex)
    (r : FlowPair) (member : r ∈ CompleteProfileKernel.tagged.step d prior v) : ResidualAllowed d r.2 := by
  change r ∈ LambdaFlow.stepWith d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d) prior v at member
  simp only [LambdaFlow.stepWith] at member
  split at member
  · split at member
    · simp only [List.mem_singleton] at member
      subst r
      exact Or.inl rfl
    · obtain ⟨a, ha, equal⟩ := List.mem_map.mp member
      subst r
      exact ancestral_allowed d a (ancestor_mem d v a ha)
  · rw [List.mem_eraseDups, List.mem_append] at member
    rcases member with core | ancestry
    · split at core
      · split at core
        · rename_i u alpha beta predecessors formula
          split at core
          · obtain ⟨input, hi, output⟩ := List.mem_flatMap.mp core
            obtain ⟨edge, _, output⟩ := List.mem_filterMap.mp output
            split at output
            · obtain ⟨tail, consumed, equal⟩ := Option.map_eq_some_iff.mp output
              cases equal
              exact consume_allowed d edge input.2 tail (correct u input hi) consumed
            · contradiction
          · contradiction
        · contradiction
      · rw [List.mem_append] at core
        rcases core with eliminated | mixedRoute
        · simp only [List.mem_flatMap] at eliminated
          obtain ⟨⟨major, minor⟩, _, left, hl, right, _, eMinor, _, output⟩ := eliminated
          obtain ⟨eMajor, _, output⟩ := List.mem_filterMap.mp output
          split at output
          · cases consumed : consume eMinor left.2 with
            | none => simp [consumed] at output
            | some tail =>
              cases other : consume eMajor right.2 with
              | none => simp [consumed, other] at output
              | some otherTail =>
                simp only [consumed, other] at output
                split at output
                · cases output
                  exact consume_allowed d eMinor left.2 tail (correct minor left hl) consumed
                · contradiction
          · contradiction
        · exact mixed d v prior correct r mixedRoute
    · exact reseed d v prior r ancestry

theorem step_residual (variant : MultiIntro.Variant) (d : Graph)
    (prior : Vertex → List FlowPair)
    (correct : ∀ u r, r ∈ prior u → ResidualAllowed d r.2) (v : Vertex)
    (r : FlowPair) (member : r ∈ MultiIntro.step variant d prior v) : ResidualAllowed d r.2 := by
  unfold MultiIntro.step at member
  split at member
  · rw [List.mem_eraseDups] at member
    obtain ⟨e, _, produced⟩ := List.mem_flatMap.mp member
    rcases List.mem_append.mp produced with ordinary | ancestry
    · unfold MultiIntro.ordinaryEdge at ordinary
      split at ordinary
      · split at ordinary
        · obtain ⟨input, hi, output⟩ := List.mem_filterMap.mp ordinary
          split at output
          · obtain ⟨tail, consumed, equal⟩ := Option.map_eq_some_iff.mp output
            cases equal
            exact consume_allowed d e input.2 tail (correct e.START input hi) consumed
          · contradiction
        · contradiction
      · contradiction
    · unfold MultiIntro.reseedEdge at ancestry
      split at ancestry
      · split at ancestry
        · obtain ⟨input, _, output⟩ := List.mem_flatMap.mp ancestry
          split_ifs at output
          · obtain ⟨a, ha, equal⟩ := List.mem_map.mp output
            subst r
            exact ancestral_allowed d a (eligible_mem d v a ha)
          · contradiction
        · contradiction
      · contradiction
  · exact reference_step_residual d prior correct v r member

theorem residual_origin (variant : MultiIntro.Variant) (d : Graph) :
    ∀ fuel v r, r ∈ MultiIntro.evaluateAt variant d fuel v → ResidualAllowed d r.2 := by
  intro fuel
  induction fuel with
  | zero => simp [MultiIntro.evaluateAt]
  | succ fuel ih => exact fun v r member => step_residual variant d _ ih v r member

theorem dependency_support (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) (r : FlowPair)
    (member : r ∈ MultiIntro.evaluateAt variant d fuel v) :
    CanonicalDep r.1 ∧ ∀ formula ∈ r.1, formula ∈ RefinedEval.formulas d :=
  ⟨MultiIntro.routes_canonical variant d fuel v r member,
    MultiIntro.evaluateAt_supported variant d fuel v
      (FaithfulKeyed.node_formula_supported d v listed) r member⟩

theorem components (d : Graph) (_valid : _root_.Flow d) (v : Vertex) (listed : v ∈ d.NODES)
    (r : FlowPair) (member : r ∈ MultiIntro.evaluateAt .general d (stdFuel d) v) :
    ResidualAllowed d r.2 ∧ CanonicalDep r.1 ∧
      ∀ formula ∈ r.1, formula ∈ RefinedEval.formulas d :=
  ⟨residual_origin .general d _ v r member, dependency_support .general d _ v listed r member⟩

end FlowSize
