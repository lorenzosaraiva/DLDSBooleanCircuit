import Semantic.LambdaFlowCache

open Semantic FlowSpec

namespace FaithfulKeyed



def DepBlind (guard : LambdaFlow.Guard) : Prop :=
  ∀ left right edge, guard left edge = guard right edge


def RouteDepBlind (guard : FlowPair → Deduction → Bool) : Prop :=
  ∀ left right residual edge, guard (left, residual) edge = guard (right, residual) edge

theorem allowAll_depBlind : DepBlind LambdaFlow.allowAll := by
  intro left right edge
  rfl

theorem depBlind_iff_edgeOnly (guard : LambdaFlow.Guard) :
    DepBlind guard ↔ ∀ deps edge, guard deps edge = guard [] edge := by
  constructor
  · intro blind deps edge
    exact blind deps [] edge
  · intro fixed left right edge
    exact (fixed left edge).trans (fixed right edge).symm

theorem edgeOnly_depBlind (test : Deduction → Bool) :
    DepBlind (fun _ edge => test edge) := by
  intro left right edge
  rfl

theorem residualOnly_routeDepBlind (test : ColourPath → Deduction → Bool) :
    RouteDepBlind (fun route edge => test route.2 edge) := by
  intro left right residual edge
  rfl

theorem consume_routeDepBlind (test : Option ColourPath → Bool) :
    RouteDepBlind (fun route edge => test (consume edge route.2)) := by
  intro left right residual edge
  rfl

theorem reseedPremiseAgrees_routeDepBlind : RouteDepBlind reseedPremiseAgreesB := by
  intro left right residual edge
  rfl

theorem reseedZeroPremise_routeDepBlind : RouteDepBlind reseedZeroPremiseB := by
  intro left right residual edge
  rfl

theorem colour_routeDepBlind (colour : Nat) :
    RouteDepBlind (fun _ edge => edge.COLOUR == colour) := by
  intro left right residual edge
  rfl

theorem residualEmpty_routeDepBlind :
    RouteDepBlind (fun route _ => route.2.isEmpty) := by
  intro left right residual edge
  rfl

theorem and_routeDepBlind {first second : FlowPair → Deduction → Bool}
    (hfirst : RouteDepBlind first) (hsecond : RouteDepBlind second) :
    RouteDepBlind (fun route edge => first route edge && second route edge) := by
  intro left right residual edge
  dsimp only
  rw [hfirst left right residual edge, hsecond left right residual edge]



theorem depSimilar_not_depBlind :
    ¬ (∀ left right stored : Dep, depSimilarB left stored = depSimilarB right stored) := by
  intro blind
  have bad := blind [.atom "a"] [.atom "b"] [.atom "a"]
  simp [depSimilarB, depSetEq] at bad

theorem historicalGuard_not_depBlind : ¬ DepBlind LambdaFlow.historicalGuard := by
  intro blind
  apply depSimilar_not_depBlind
  intro left right stored
  let v : Vertex := ⟨0, 0, .atom "a", true, false, []⟩
  let edge : Deduction := ⟨v, v, 0, stored⟩
  exact blind left right edge

#print axioms allowAll_depBlind
#print axioms depBlind_iff_edgeOnly
#print axioms residualOnly_routeDepBlind
#print axioms consume_routeDepBlind
#print axioms reseedPremiseAgrees_routeDepBlind
#print axioms reseedZeroPremise_routeDepBlind
#print axioms depSimilar_not_depBlind
#print axioms historicalGuard_not_depBlind

end FaithfulKeyed
