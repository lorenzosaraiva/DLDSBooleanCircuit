import Infrastructure.DependencyGuardLabels

open Semantic FlowSpec SelfCertifying

namespace DependencyGuard

set_option maxHeartbeats 2000000
set_option maxRecDepth 100000



def ConsumedGuards (d : Graph) (guard : LambdaFlow.Guard)
    (prior : Vertex → List FlowPair) : Prop :=
  ∀ u v edge, edge ∈ edgesBetween d u v → ∀ route ∈ prior u,
    ∀ tail, consume edge route.2 = some tail → guard route.1 edge = true

private theorem guarded_map (guard : LambdaFlow.Guard) (route : FlowPair)
    (edge : Deduction) (f : ColourPath → FlowPair)
    (passes : ∀ tail, consume edge route.2 = some tail → guard route.1 edge = true) :
    (if guard route.1 edge then (consume edge route.2).map f else none) =
      (consume edge route.2).map f := by
  cases hc : consume edge route.2 with
  | none => simp []
  | some tail => simp [passes tail hc]

private theorem guarded_elim (guard : LambdaFlow.Guard) (left right : FlowPair)
    (minor major : Deduction)
    (passesLeft : ∀ tail, consume minor left.2 = some tail → guard left.1 minor = true)
    (passesRight : ∀ tail, consume major right.2 = some tail → guard right.1 major = true) :
    (if guard left.1 minor && guard right.1 major then
      match consume minor left.2, consume major right.2 with
      | some p, some q => if p = q then some (depUnion left.1 right.1, p) else none
      | _, _ => none
    else none) =
    (match consume minor left.2, consume major right.2 with
      | some p, some q => if p = q then some (depUnion left.1 right.1, p) else none
      | _, _ => none) := by
  cases hl : consume minor left.2 with
  | none => simp []
  | some p =>
    cases hr : consume major right.2 with
    | none => simp []
    | some q => simp [passesLeft p hl, passesRight q hr]

private theorem mixed_guard_eq (d : Graph) (guard : LambdaFlow.Guard)
    (prior : Vertex → List FlowPair) (passes : ConsumedGuards d guard prior) (v : Vertex) :
    LambdaFlow.mixedWith d guard v prior = LambdaFlow.mixedWith d LambdaFlow.allowAll v prior := by
  unfold LambdaFlow.mixedWith
  split
  · apply List.flatMap_congr
    intro u hu
    split
    · apply List.flatMap_congr
      intro route hroute
      apply List.filterMap_congr
      intro edge hedge
      simp only [LambdaFlow.allowAll, if_true]
      exact guarded_map guard route edge _ (passes u v edge hedge route hroute)
    · rfl
  · rfl

private theorem elim_reseed_guard_eq (d : Graph) (guard : LambdaFlow.Guard)
    (prior : Vertex → List FlowPair) (passes : ConsumedGuards d guard prior)
    (v : Vertex) (ancestors : List Ancestral) :
    LambdaFlow.elimReseedWith d guard v prior ancestors =
      LambdaFlow.elimReseedWith d LambdaFlow.allowAll v prior ancestors := by
  unfold LambdaFlow.elimReseedWith
  apply List.flatMap_congr
  intro pair hpair
  apply List.flatMap_congr
  intro left hleft
  apply List.flatMap_congr
  intro right hright
  apply List.flatMap_congr
  intro minor hminor
  apply List.flatMap_congr
  intro major hmajor
  cases hl : reseedPremiseAgreesB left minor with
  | false => simp [LambdaFlow.allowAll]
  | true =>
    cases hr : reseedPremiseAgreesB right major with
    | false => simp [LambdaFlow.allowAll]
    | true =>
      obtain ⟨p, hp⟩ := Option.isSome_iff_exists.mp hl
      obtain ⟨q, hq⟩ := Option.isSome_iff_exists.mp hr
      have gl := passes pair.2 v minor hminor left hleft p hp
      have gr := passes pair.1 v major hmajor right hright q hq
      simp [gl, gr, LambdaFlow.allowAll]

private theorem intro_reseed_guard_eq (d : Graph) (guard : LambdaFlow.Guard)
    (prior : Vertex → List FlowPair) (passes : ConsumedGuards d guard prior)
    (v : Vertex) (ancestors : List Ancestral) :
    LambdaFlow.introReseedWith d guard v prior ancestors =
      LambdaFlow.introReseedWith d LambdaFlow.allowAll v prior ancestors := by
  unfold LambdaFlow.introReseedWith
  split
  · split
    · apply List.flatMap_congr
      intro route hroute
      apply List.flatMap_congr
      intro edge hedge
      cases hz : reseedZeroPremiseB route edge with
      | false => simp [LambdaFlow.allowAll]
      | true =>
        have hs : (consume edge route.2).isSome = true := by
          have parts := hz
          simp only [reseedZeroPremiseB, Bool.and_eq_true] at parts
          exact parts.2
        obtain ⟨tail, consumed⟩ := Option.isSome_iff_exists.mp hs
        have guard := passes _ v edge hedge route hroute tail consumed
        simp [guard, LambdaFlow.allowAll]
    · rfl
  · rfl

private theorem reseed_guard_eq (d : Graph) (guard : LambdaFlow.Guard)
    (prior : Vertex → List FlowPair) (passes : ConsumedGuards d guard prior) (v : Vertex) :
    LambdaFlow.reseedWith d guard v prior = LambdaFlow.reseedWith d LambdaFlow.allowAll v prior := by
  unfold LambdaFlow.reseedWith
  dsimp only
  split
  · rfl
  · rw [elim_reseed_guard_eq d guard prior passes,
      intro_reseed_guard_eq d guard prior passes]



theorem step_guard_eq (d : Graph) (guard : LambdaFlow.Guard)
    (prior : Vertex → List FlowPair) (passes : ConsumedGuards d guard prior) (v : Vertex) :
    LambdaFlow.stepWith d guard guard prior v =
      LambdaFlow.stepWith d LambdaFlow.allowAll LambdaFlow.allowAll prior v := by
  by_cases source : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
  · simp [LambdaFlow.stepWith, source]
  · simp only [LambdaFlow.stepWith, source, Bool.false_eq_true, if_false]
    apply congrArg List.eraseDups
    apply congrArg₂ (· ++ ·)
    · by_cases pairs : (elimPairsAt d v).isEmpty = true
      · simp only [pairs, if_true]
        split
        · split
          · apply List.flatMap_congr
            intro route hroute
            apply List.filterMap_congr
            intro edge hedge
            simp only [LambdaFlow.allowAll, if_true]
            exact guarded_map guard route edge _ (passes _ v edge hedge route hroute)
          · rfl
        · rfl
      · simp only [pairs, Bool.false_eq_true, if_false]
        apply congrArg₂ (· ++ ·)
        · apply List.flatMap_congr
          intro pair hpair
          apply List.flatMap_congr
          intro left hleft
          apply List.flatMap_congr
          intro right hright
          apply List.flatMap_congr
          intro minor hminor
          apply List.filterMap_congr
          intro major hmajor
          simp only [LambdaFlow.allowAll, Bool.true_and, if_true]
          exact guarded_elim guard left right minor major
            (passes pair.2 v minor hminor left hleft)
            (passes pair.1 v major hmajor right hright)
        · exact mixed_guard_eq d guard prior passes v
    · exact reseed_guard_eq d guard prior passes v



theorem guards_redundant_eq (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph)
    (labels : LambdaFlow.Def23DecoratedB lg = true) :
    ∀ fuel v, LambdaFlow.lambdaFlowAt lg fuel v =
      LambdaFlow.guardFreeFlowAt lg.graph fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
    intro v
    simp only [LambdaFlow.lambdaFlowAt, LambdaFlow.guardFreeFlowAt, LambdaFlow.evaluateAt]
    rw [show LambdaFlow.evaluateAt lg.graph (LambdaFlow.labelGuard lg)
        (LambdaFlow.labelGuard lg) fuel =
        LambdaFlow.evaluateAt lg.graph LambdaFlow.allowAll LambdaFlow.allowAll fuel
      from funext ih]
    apply step_guard_eq
    intro u finish edge hedge route hroute tail consumed
    obtain ⟨start, _, edgeMember⟩ := mem_edgesBetween hedge
    apply consumed_guard_at_every_fuel lg structural labels edge edgeMember fuel route
      _ tail consumed
    simpa only [start] using hroute




theorem guards_redundant (lg : LambdaFlow.LGraph)
    (structural : structuralValid lg.graph)
    (_levels : LeveledColored lg.graph)
    (labels : LambdaFlow.Def23DecoratedB lg = true) :
    ∀ fuel v, ListSetEq (LambdaFlow.lambdaFlowAt lg fuel v)
      (LambdaFlow.guardFreeFlowAt lg.graph fuel v) := by
  intro fuel v route
  rw [guards_redundant_eq lg structural labels fuel v]

#print axioms step_guard_eq
#print axioms guards_redundant_eq
#print axioms guards_redundant

end DependencyGuard
