import Certificate.Flow.Soundness
import Certificate.Encoding.FixedShape

/-! Historical positive production with direct fixed-slot provenance.
No coverage, reverse inclusion, root discharge, or bit-compiler correctness is assumed. -/
open DLDS FlowSemantics Certificate.Flow.Foundation
namespace Robustness

private theorem bind_mono {α β : Type} {xs ys : List α}
    {f g : α → List β} (hxy : ∀ x ∈ xs, x ∈ ys)
    (hfg : ∀ x ∈ xs, ∀ b ∈ f x, b ∈ g x) :
    ∀ b ∈ xs.flatMap f, b ∈ ys.flatMap g := by
  intro b hb
  obtain ⟨x, hx, hb⟩ := List.mem_flatMap.mp hb
  exact List.mem_flatMap.mpr ⟨x, hxy x hx, hfg x hx b hb⟩

private theorem mixed_mono (d : Graph) (v : Vertex)
    (P Q : Vertex → List FlowPair)
    (h : ∀ u r, r ∈ P u → r ∈ Q u) :
    ∀ r ∈ mixedIntroPairs d v P, r ∈ mixedIntroPairs d v Q := by
  unfold mixedIntroPairs
  cases v.FORMULA with
  | atom name => simp
  | implication alpha beta =>
    apply bind_mono (fun _ hx => hx)
    intro u _
    split
    · exact bind_mono (h u) (fun _ _ _ hr => hr)
    · exact fun _ hr => hr

private theorem elim_reseed_mono (d : Graph) (v : Vertex)
    (P Q : Vertex → List FlowPair) (ancestors : List Ancestral)
    (h : ∀ u r, r ∈ P u → r ∈ Q u) :
    ∀ r ∈ elimReseedPairs d v P ancestors,
      r ∈ elimReseedPairs d v Q ancestors := by
  unfold elimReseedPairs
  apply bind_mono (fun _ hx => hx)
  intro pair _
  apply bind_mono (h pair.2)
  intro minorRoute _
  apply bind_mono (h pair.1)
  exact fun _ _ _ hr => hr

private theorem intro_reseed_mono (d : Graph) (v : Vertex)
    (P Q : Vertex → List FlowPair) (ancestors : List Ancestral)
    (h : ∀ u r, r ∈ P u → r ∈ Q u) :
    ∀ r ∈ introReseedPairs d v P ancestors,
      r ∈ introReseedPairs d v Q ancestors := by
  unfold introReseedPairs
  split
  · split
    · exact bind_mono (h _) (fun _ _ _ hr => hr)
    · exact fun _ hr => hr
  · exact fun _ hr => hr

private theorem reseed_mono (d : Graph) (v : Vertex)
    (P Q : Vertex → List FlowPair)
    (h : ∀ u r, r ∈ P u → r ∈ Q u) :
    ∀ r ∈ reseedPairs d v P, r ∈ reseedPairs d v Q := by
  unfold reseedPairs
  dsimp only
  split
  · exact fun _ hr => hr
  · intro r hr
    simp only [List.mem_eraseDups, List.mem_append] at hr ⊢
    exact hr.elim
      (fun hr => Or.inl (elim_reseed_mono d v P Q _ h r hr))
      (fun hr => Or.inr (intro_reseed_mono d v P Q _ h r hr))

private def introFrom (d : Graph) (v : Vertex)
    (P : Vertex → List FlowPair) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [u], Formula.implication alpha beta =>
      if u.FORMULA = beta then
        (P u).flatMap fun route =>
          (edgesBetween d u v).filterMap fun edge =>
            (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)
      else []
  | _, _ => []

private theorem intro_mono (d : Graph) (v : Vertex)
    (P Q : Vertex → List FlowPair)
    (h : ∀ u r, r ∈ P u → r ∈ Q u) :
    ∀ r ∈ introFrom d v P, r ∈ introFrom d v Q := by
  unfold introFrom
  split
  · split
    · exact bind_mono (h _) (fun _ _ _ hr => hr)
    · exact fun _ hr => hr
  · exact fun _ hr => hr

/-- Existing production is monotone in its supplied premise route sets. -/
theorem flowStepFrom_mono (d : Graph) (v : Vertex)
    (P Q : Vertex → List FlowPair)
    (h : ∀ u r, r ∈ P u → r ∈ Q u) :
    ∀ r ∈ flowStepFrom d P v, r ∈ flowStepFrom d Q v := by
  intro r hr
  by_cases guard : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simpa only [flowStepFrom, guard, if_true] using hr
  · have guardFalse : (v.HYPOTHESIS || (predsOf d v).isEmpty) = false :=
      Bool.eq_false_iff.mpr guard
    unfold flowStepFrom at hr ⊢
    dsimp only at hr ⊢
    simp only [guardFalse, Bool.false_eq_true, if_false] at hr ⊢
    simp only [List.mem_eraseDups, List.mem_append] at hr ⊢
    rcases hr with hcore | hseed
    · left
      by_cases hpairs : (elimPairsAt d v).isEmpty = true
      · simp only [hpairs, if_true] at hcore ⊢
        exact intro_mono d v P Q h r hcore
      · have pairsFalse : (elimPairsAt d v).isEmpty = false := Bool.eq_false_iff.mpr hpairs
        simp only [pairsFalse, Bool.false_eq_true, if_false, List.mem_append] at hcore ⊢
        rcases hcore with helim | hintro
        · left
          apply bind_mono (fun _ hx => hx) _ r helim
          intro pair _
          apply bind_mono (h pair.2)
          intro minor _
          apply bind_mono (h pair.1)
          exact fun _ _ _ hr => hr
        · exact Or.inr (mixed_mono d v P Q h r hintro)
    · exact Or.inr (reseed_mono d v P Q h r hseed)

inductive ProducerKind where
  | source
  | step
  deriving DecidableEq, BEq, Repr

/-- Fixed route ownership and direct premise addresses.
The step kind includes all existing ordinary, mixed and reseeded Flow clauses. -/
structure JustifiedCertificate (d : Graph) (slots : Nat) where
  owner : Fin slots → Vertex
  ownerListed : ∀ slot, owner slot ∈ d.NODES
  route : Fin slots → FlowPair
  kind : Fin slots → ProducerKind
  premise : Fin slots → Fin 2 → Option (Fin slots)

def pointedSlots {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots) (slot : Fin slots) : List (Fin slots) :=
  (List.finRange 2).filterMap (c.premise slot)

def pointedRoutes {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots) (slot : Fin slots)
    (v : Vertex) : List FlowPair :=
  (pointedSlots c slot).filterMap fun pointer =>
    if c.owner pointer = v then some (c.route pointer) else none

def justificationB {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots) (slot : Fin slots) : Bool :=
  let v := c.owner slot
  let source := v.HYPOTHESIS || (predsOf d v).isEmpty
  (match c.kind slot with
    | .source => source && (pointedSlots c slot).isEmpty
    | .step => !source) &&
  (pointedSlots c slot).all (fun pointer =>
    decide (c.owner pointer ∈ predsOf d v)) &&
  decide (c.route slot ∈
    WitnessableRoutes d v (flowStepFrom d (pointedRoutes c slot) v))

/-- Every accepted stored slot denotes a witnessable recursive Flow route.
Hypotheses: structural validity and passing checks at every fixed stored slot.
No premise asserts coverage, Flow completion, certificate equality or validity. -/
theorem justification_sound {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots)
    (structural : structuralValid d)
    (checked : ∀ slot, justificationB c slot = true) :
    ∀ slot, c.route slot ∈
      WitnessableRoutes d (c.owner slot) (flowAt d (stdFuel d) (c.owner slot)) := by
  have induction : ∀ rank : Nat, ∀ slot,
      countAbove d (c.owner slot) = rank →
      c.route slot ∈ WitnessableRoutes d (c.owner slot)
        (flowAt d (stdFuel d) (c.owner slot)) := by
    intro rank
    induction rank using Nat.strong_induction_on with
    | _ rank ih =>
      intro slot rankEq
      have pass := checked slot
      simp only [justificationB, Bool.and_eq_true, List.all_eq_true,
        decide_eq_true_eq] at pass
      have produced := pass.2
      have pointedPred := pass.1.2
      rw [WitnessableRoutes, List.mem_filter] at produced ⊢
      refine ⟨?_, produced.2⟩
      have inputs : ∀ u r, r ∈ pointedRoutes c slot u →
          r ∈ flowAt d (stdFuel d - 1) u := by
        intro u r member
        obtain ⟨pointer, hp, value⟩ := List.mem_filterMap.mp member
        by_cases eq : c.owner pointer = u
        · simp only [eq, if_true, Option.some.injEq] at value
          subst r
          have pred : c.owner pointer ∈ predsOf d (c.owner slot) :=
            pointedPred pointer hp
          obtain ⟨level, outgoing⟩ := predsOf_props structural pred
          have decrease := countAbove_lt_of_level_succ d
            (c.ownerListed pointer) level
          have previous := ih (countAbove d (c.owner pointer))
            (by omega) pointer rfl
          have raw := (List.mem_filter.mp previous).1
          rw [flowAt_stdFuel_pred d structural (c.ownerListed pointer) outgoing] at raw
          simpa [eq] using raw
        · simp [eq] at value
      have step := flowStepFrom_mono d (c.owner slot) _
        (flowAt d (stdFuel d - 1)) inputs _ produced.1
      have fuel : stdFuel d - 1 + 1 = stdFuel d := by unfold stdFuel; omega
      change c.route slot ∈ flowAt d (stdFuel d - 1 + 1) (c.owner slot) at step
      simpa [fuel] using step
  intro slot
  exact induction (countAbove d (c.owner slot)) slot rfl

/-- Sets of directly addressed stored values, independent of multiplicity. -/
def Stored {d : Graph} {slots : Nat} (c : JustifiedCertificate d slots)
    (v : Vertex) (route : FlowPair) : Prop :=
  ∃ slot, c.owner slot = v ∧ c.route slot = route

theorem stored_subset_flow {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots)
    (structural : structuralValid d)
    (checked : ∀ slot, justificationB c slot = true) :
    ∀ v route, Stored c v route →
      route ∈ WitnessableRoutes d v (flowAt d (stdFuel d) v) := by
  rintro v route ⟨slot, rfl, rfl⟩
  exact justification_sound c structural checked slot
end Robustness
