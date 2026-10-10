import Robustness.RuleCompilation.ResidualGuards

/-! Ordinary introduction compiled on the full fixed-shape Boolean product.
No consume, tail equality or witnessability premise is added. -/
open DLDS Flow Robustness.BitLocality Robustness.SelectedProducers
namespace Robustness.RuleCompilation
set_option maxHeartbeats 2000000

def removeBitAtom {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (output input : Fin r) (code : Fin f) (alpha : Formula) :
    Atom (RuleCoord shape size) k where
  address index := .inr (.inl ((if index.1 = 0 then output else input), code))
  test values := values (0, ⟨0, positive⟩) ==
    (values (1, ⟨0, positive⟩) && !decide (shape.formulaAt code = alpha))

def removeGuards {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (output input : Fin r) (alpha : Formula) :
    List (Guard (RuleCoord shape size) k) :=
  (List.finRange f).map fun code =>
    atomGuard (removeBitAtom shape size k positive output input code alpha)

theorem removeGuards_check {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (output input : Fin r) (alpha : Formula)
    (bits : RuleCoord shape size → Bool) :
    (removeGuards shape size k positive output input alpha).all (fun guard => guard.check bits) =
      decide ((ruleRoutes shape size bits output).1 =
        depRemove (ruleRoutes shape size bits input).1 alpha) := by
  apply Bool.eq_iff_iff.mpr
  simp only [removeGuards, List.all_map, Function.comp_apply,
    List.all_eq_true, List.mem_finRange, atomGuard_check, Atom.eval, removeBitAtom]
  simp only [true_implies, Fin.reduceEq, if_true, if_false, beq_iff_eq, decide_eq_true_eq]
  change (∀ code, bits (.inr (.inl (output, code))) =
    (bits (.inr (.inl (input, code))) && !decide (shape.formulaAt code = alpha))) ↔ _
  dsimp only [ruleRoutes, decodeRoutes]
  rw [← decode_remove]
  constructor
  · intro allBits
    exact congrArg (decodeDependency shape) (funext allBits)
  · intro same
    exact congrFun (decodeDependency_injective shape same)

def introProgram {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (width : shape.colourWidth ≤ k)
    {d : Graph} (frame : IntroFrame (routes := r) d) :
    Program (RuleCoord shape size) (3 + 2 * pointerWidth size) k where
  headerAddress bit := .inl bit
  guards _ :=
    removeGuards shape size k positive frame.output frame.input frame.alpha ++
    consumeGuards shape size k width frame.input frame.output frame.incoming ++
    continuationGuards shape size k width d (frame.owner frame.output) frame.output

theorem intro_compiler_check_eq {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (width : shape.colourWidth ≤ k)
    {d : Graph} (frame : IntroFrame (routes := r) d)
    (bits : RuleCoord shape size → Bool) :
    (introProgram shape size k positive width frame).check bits =
      introAcceptance shape size frame bits := by
  rw [intro_acceptance_eq]
  simp only [Program.check, introProgram, List.all_append, removeGuards_check,
    consumeGuards_check, continuationGuards_check]

theorem intro_compiler_failure_eq {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (width : shape.colourWidth ≤ k)
    {d : Graph} (frame : IntroFrame (routes := r) d)
    (bits : RuleCoord shape size → Bool) :
    (introProgram shape size k positive width frame).failure bits =
      introFailure shape size frame bits :=
  congrArg Bool.not (intro_compiler_check_eq shape size k positive width frame bits)

end Robustness.RuleCompilation
