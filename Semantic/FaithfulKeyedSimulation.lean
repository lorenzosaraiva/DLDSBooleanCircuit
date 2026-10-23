import Semantic.FaithfulKeyedEncoding
import Semantic.FaithfulKernelCommutation
import Semantic.FaithfulFormulaSupport

open Semantic FlowSpec
namespace FaithfulKeyed



def keyedStep (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (prior : Vertex → KeyedEval.Carrier d) (v : Vertex) : KeyedEval.Carrier d :=
  KeyedEval.merge d (LambdaFlow.stepWith d ordinary reseed
    (fun u => KeyedEval.expand d (prior u)) v)

def keyedAt (d : Graph) (ordinary reseed : LambdaFlow.Guard) :
    Nat → Vertex → KeyedEval.Carrier d
  | 0, _ => []
  | fuel + 1, v => keyedStep d ordinary reseed (keyedAt d ordinary reseed fuel) v



theorem keyedAt_summary (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (ho : DepBlind ordinary) (hr : DepBlind reseed) (fuel : Nat)
    (v : Vertex) (hv : v.FORMULA ∈ RefinedEval.formulas d) :
    SummaryEq (KeyedEval.expand d (keyedAt d ordinary reseed fuel v))
      (LambdaFlow.evaluateAt d ordinary reseed fuel v) := by
  induction fuel generalizing v with
  | zero => exact SummaryEq.refl []
  | succ fuel ih =>
      have localEq := stepWith_summary_on_preds d ordinary reseed ho hr
        (fun u => KeyedEval.expand d (keyedAt d ordinary reseed fuel u))
        (LambdaFlow.evaluateAt d ordinary reseed fuel) v
        (fun u hu => ih u (predecessor_formula_supported d v u hu))
      have supported : FamilySupported d
          (LambdaFlow.stepWith d ordinary reseed
            (fun u => KeyedEval.expand d (keyedAt d ordinary reseed fuel u)) v) := by
        intro r hroute f hf
        obtain ⟨s, hs, _, hbit⟩ := (localEq.2 r.2 f).mp ⟨r, hroute, rfl, hf⟩
        exact evaluateAt_supported d ordinary reseed (fuel + 1) v hv s hs f hbit
      exact (expand_merge_summary d _ supported).trans localEq

def rootsDischargedWithB (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let routes := LambdaFlow.evaluateAt d ordinary reseed fuel v
     !routes.isEmpty && routes.all fun r => r.1.isEmpty && r.2.isEmpty)

def keyedRootsWithB (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (fuel : Nat) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let rows := keyedAt d ordinary reseed fuel v
     !rows.isEmpty && rows.all fun row => row.1.isEmpty && row.2.toList.all (!·))




theorem keyedRootsWithB_eq (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (ho : DepBlind ordinary) (hr : DepBlind reseed) (fuel : Nat) :
    keyedRootsWithB d ordinary reseed fuel = rootsDischargedWithB d ordinary reseed fuel := by
  apply Bool.eq_iff_iff.mpr
  simp only [keyedRootsWithB, rootsDischargedWithB, List.all_eq_true]
  apply forall_congr'
  intro v
  apply forall_congr'
  intro hv
  have sem := keyedAt_summary d ordinary reseed ho hr fuel v (node_formula_supported d v hv)
  have criterion :
      (!((keyedAt d ordinary reseed fuel v).isEmpty) &&
        (keyedAt d ordinary reseed fuel v).all
          (fun row => row.1.isEmpty && row.2.toList.all (!·))) =
      (!(LambdaFlow.evaluateAt d ordinary reseed fuel v).isEmpty &&
        (LambdaFlow.evaluateAt d ordinary reseed fuel v).all
          (fun r => r.1.isEmpty && r.2.isEmpty)) := by
    apply Bool.eq_iff_iff.mpr
    exact (closed_expand_iff d _).symm.trans
      (sem.closed_iff.trans (closedRoutes_iff_bool _))
  rw [criterion]

theorem keyedAt_allowAll (d : Graph) : ∀ fuel v,
    keyedAt d LambdaFlow.allowAll LambdaFlow.allowAll fuel v =
      LambdaKeyed.evaluateAt d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
      intro v
      simp only [keyedAt, keyedStep, LambdaKeyed.evaluateAt, LambdaKeyed.step]
      rw [show keyedAt d LambdaFlow.allowAll LambdaFlow.allowAll fuel =
        LambdaKeyed.evaluateAt d fuel from funext ih]

theorem keyed_guardFree_roots (d : Graph) :
    keyedRootsWithB d LambdaFlow.allowAll LambdaFlow.allowAll (stdFuel d) =
      LambdaFlow.guardFreeRootsB d := by
  rw [keyedRootsWithB_eq d _ _ allowAll_depBlind allowAll_depBlind]
  exact (LambdaFlow.guardFreeRootsB_eq_raw d).symm

#print axioms keyedAt_summary
#print axioms keyedRootsWithB_eq
#print axioms keyedAt_allowAll
#print axioms keyed_guardFree_roots
end FaithfulKeyed
