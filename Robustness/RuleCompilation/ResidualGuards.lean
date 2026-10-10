import Robustness.RuleCompilation.IntroSemantics

/-! Each residual guard reads one or two fixed-address colour words.
Consumption handles empty paths, incompatible lengths, and exact consumed tails.
Output witnessability is checked from its head, never assumed by the shape. -/
open DLDS Flow Robustness.BitLocality Robustness.SelectedProducers
namespace Robustness.RuleCompilation
set_option maxHeartbeats 2000000
set_option maxRecDepth 2000

def routeWord {r f : Nat} (shape : RouteShape r f) (size : Nat)
    (bits : RuleCoord shape size → Bool) (slot : Fin r)
    (position : Fin (shape.pathLength slot)) : Nat :=
  decodeWord fun bit => bits (.inr (.inr ⟨slot, position, bit⟩))

theorem route_path_cons {r f : Nat} (shape : RouteShape r f) (size : Nat)
    (bits : RuleCoord shape size → Bool) (slot : Fin r)
    (positive : 0 < shape.pathLength slot) :
    (ruleRoutes shape size bits slot).2 =
      routeWord shape size bits slot ⟨0, positive⟩ ::
        List.ofFn (fun position : Fin (shape.pathLength slot - 1) =>
          routeWord shape size bits slot ⟨position.val + 1, by omega⟩) := by
  apply List.ext_getElem
  · simp only [ruleRoutes, decodeRoutes, List.length_ofFn, List.length_cons]
    omega
  · intro index leftBound rightBound
    cases index with
    | zero => simp [ruleRoutes, decodeRoutes, routeWord]
    | succ index => simp [ruleRoutes, decodeRoutes, routeWord]

def wordPredicateAtom {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (slot : Fin r)
    (position : Fin (shape.pathLength slot)) (predicate : Nat → Bool) :
    Atom (RuleCoord shape size) k where
  address index := if bound : index.2.val < shape.colourWidth then
    .inr (.inr ⟨slot, position, ⟨index.2.val, bound⟩⟩) else .inl ⟨0, by omega⟩
  test values := predicate (decodeWord (fun bit : Fin shape.colourWidth =>
    values (0, ⟨bit.val, lt_of_lt_of_le bit.isLt width⟩)))

theorem wordPredicateAtom_eval {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (slot : Fin r)
    (position : Fin (shape.pathLength slot)) (predicate : Nat → Bool)
    (bits : RuleCoord shape size → Bool) :
    (wordPredicateAtom shape size k width slot position predicate).eval bits =
      predicate (routeWord shape size bits slot position) := by
  simp only [Atom.eval, wordPredicateAtom, routeWord]
  congr 2
  funext bit
  simp [bit.isLt]

def wordPairAtom {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (left right : Fin r)
    (lp : Fin (shape.pathLength left)) (rp : Fin (shape.pathLength right)) :
    Atom (RuleCoord shape size) k where
  address index := if bound : index.2.val < shape.colourWidth then
    if index.1 = 0 then .inr (.inr ⟨left, lp, ⟨index.2.val, bound⟩⟩)
    else .inr (.inr ⟨right, rp, ⟨index.2.val, bound⟩⟩)
    else .inl ⟨0, by omega⟩
  test values := decide
    (decodeWord (fun bit : Fin shape.colourWidth =>
      values (0, ⟨bit.val, lt_of_lt_of_le bit.isLt width⟩)) =
     decodeWord (fun bit : Fin shape.colourWidth =>
      values (1, ⟨bit.val, lt_of_lt_of_le bit.isLt width⟩)))

theorem wordPairAtom_eval {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (left right : Fin r)
    (lp : Fin (shape.pathLength left)) (rp : Fin (shape.pathLength right))
    (bits : RuleCoord shape size → Bool) :
    (wordPairAtom shape size k width left right lp rp).eval bits =
      decide (routeWord shape size bits left lp = routeWord shape size bits right rp) := by
  simp only [Atom.eval, wordPairAtom, routeWord]
  congr 2
  all_goals congr 1; funext bit; simp [bit.isLt]

theorem routeContinuable_cons (d : Graph) (v : Vertex) (head : Nat) (tail : List Nat) :
    routeContinuableAtB d v (head :: tail) = routeContinuableAtB d v [head] := by
  unfold routeContinuableAtB
  dsimp only
  split
  · rfl
  · congr 1
    funext incoming
    by_cases same : incoming.COLOUR = head <;> simp [consume, same]

def continuationGuards {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (d : Graph) (v : Vertex) (slot : Fin r) :
    List (Guard (RuleCoord shape size) k) :=
  if positive : 0 < shape.pathLength slot then
    [atomGuard (wordPredicateAtom shape size k width slot ⟨0, positive⟩
      (fun head => routeContinuableAtB d v [head]))]
  else [atomGuard (constantAtom shape size k (routeContinuableAtB d v []))]

theorem continuationGuards_check {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (d : Graph) (v : Vertex) (slot : Fin r)
    (bits : RuleCoord shape size → Bool) :
    (continuationGuards shape size k width d v slot).all (fun guard => guard.check bits) =
      routeContinuableAtB d v (ruleRoutes shape size bits slot).2 := by
  unfold continuationGuards
  split
  · rename_i positive
    simp only [List.all_cons, List.all_nil, Bool.and_true, atomGuard_check,
      wordPredicateAtom_eval]
    rw [route_path_cons shape size bits slot positive]
    exact (routeContinuable_cons _ _ _ _).symm
  · rename_i nonpositive
    have zero : shape.pathLength slot = 0 := by omega
    simp [atomGuard_check, constantAtom_eval, ruleRoutes, decodeRoutes, zero]

theorem consume_length {incoming : Deduction} {input output : ColourPath}
    (consumed : consume incoming input = some output) :
    (input.length = 0 ∧ output.length = 0) ∨ input.length = output.length + 1 := by
  cases input with
  | nil =>
    simp only [consume] at consumed
    split at consumed <;> simp_all
  | cons head tail =>
    simp only [consume] at consumed
    split at consumed <;> simp_all

def consumeGuards {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (input output : Fin r) (incoming : Deduction) :
    List (Guard (RuleCoord shape size) k) :=
  if positive : 0 < shape.pathLength input then
    if aligned : shape.pathLength input = shape.pathLength output + 1 then
      atomGuard (wordPredicateAtom shape size k width input ⟨0, positive⟩
        (fun head => decide (incoming.COLOUR = head))) ::
      (List.finRange (shape.pathLength output)).map fun position =>
        atomGuard (wordPairAtom shape size k width output input position
          ⟨position.val + 1, by omega⟩)
    else [atomGuard (constantAtom shape size k false)]
  else [atomGuard (constantAtom shape size k
    (decide (incoming.COLOUR = 0 ∧ shape.pathLength output = 0)))]

theorem consumeGuards_check {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (width : shape.colourWidth ≤ k) (input output : Fin r) (incoming : Deduction)
    (bits : RuleCoord shape size → Bool) :
    (consumeGuards shape size k width input output incoming).all
        (fun guard => guard.check bits) =
      decide (consume incoming (ruleRoutes shape size bits input).2 =
        some (ruleRoutes shape size bits output).2) := by
  unfold consumeGuards
  split
  · rename_i positive
    split
    · rename_i aligned
      apply Bool.eq_iff_iff.mpr
      simp only [List.all_cons, List.all_map, Function.comp_apply,
        atomGuard_check, wordPredicateAtom_eval, wordPairAtom_eval, Bool.and_eq_true,
        List.all_eq_true, List.mem_finRange, true_implies, decide_eq_true_eq]
      rw [route_path_cons shape size bits input positive]
      simp only [consume]
      have tailEq :
          (List.ofFn (fun position : Fin (shape.pathLength input - 1) =>
            routeWord shape size bits input ⟨position.val + 1, by omega⟩) =
              (ruleRoutes shape size bits output).2) ↔
          ∀ position : Fin (shape.pathLength output),
            routeWord shape size bits output position =
              routeWord shape size bits input ⟨position.val + 1, by omega⟩ := by
        have sameLength : shape.pathLength input - 1 = shape.pathLength output := by omega
        constructor
        · intro same position
          have value := congrArg (fun path : List Nat => path.getD position.val 0) same
          have bound : position.val < shape.pathLength input - 1 := by omega
          simpa [ruleRoutes, decodeRoutes, routeWord,
            List.getD_eq_getElem?_getD, List.getElem?_ofFn, position.isLt, bound] using value.symm
        · intro values
          apply List.ext_getElem
          · simp [ruleRoutes, decodeRoutes, sameLength]
          · intro index leftBound rightBound
            have bound : index < shape.pathLength output := by
              simpa [ruleRoutes, decodeRoutes] using rightBound
            have value := values ⟨index, bound⟩
            simpa [ruleRoutes, decodeRoutes, routeWord] using value.symm
      by_cases headMatches : incoming.COLOUR = routeWord shape size bits input ⟨0, positive⟩
      · simp only [headMatches, if_true, Option.some.injEq, true_and]
        exact tailEq.symm
      · simp [headMatches]
    · rename_i unaligned
      have impossible : consume incoming (ruleRoutes shape size bits input).2 ≠
          some (ruleRoutes shape size bits output).2 := by
        intro consumed
        have lengths := consume_length consumed
        simp only [ruleRoutes, decodeRoutes, List.length_ofFn] at lengths
        omega
      simp [atomGuard_check, constantAtom_eval, impossible]
  · rename_i nonpositive
    have zero : shape.pathLength input = 0 := by omega
    simp only [List.all_cons, List.all_nil, Bool.and_true, atomGuard_check, constantAtom_eval]
    apply Bool.eq_iff_iff.mpr
    simp only [decide_eq_true_eq]
    have inNil : (ruleRoutes shape size bits input).2 = [] := by
      simp [ruleRoutes, decodeRoutes, zero]
    rw [inNil]
    by_cases colour : incoming.COLOUR = 0
    · simp only [consume, colour, if_true, Option.some.injEq, true_and]
      constructor
      · intro length
        symm
        apply List.eq_nil_of_length_eq_zero
        simpa [ruleRoutes, decodeRoutes] using length
      · intro same
        have length := congrArg List.length same
        simpa [ruleRoutes, decodeRoutes] using length.symm
    · simp [consume, colour]

end Robustness.RuleCompilation
