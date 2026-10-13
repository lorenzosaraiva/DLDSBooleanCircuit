import Robustness.RuleCompilation.ElimSemantics

/-! Two consumed premises must both yield the exact stored residual.
This enforces their compatibility without an additional compatibility premise. -/
open DLDS FlowSemantics Robustness.BitLocality Robustness.SelectedProducers
namespace Robustness.RuleCompilation

def unionGuards {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (output left right : Fin r) :
    List (Guard (RuleCoord shape size) k) :=
  (List.finRange f).map fun code =>
    atomGuard (unionBitAtom shape size k positive output left right code)

theorem unionGuards_check {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (output left right : Fin r)
    (bits : RuleCoord shape size → Bool) :
    (unionGuards shape size k positive output left right).all (fun guard => guard.check bits) =
      decide ((ruleRoutes shape size bits output).1 =
        depUnion (ruleRoutes shape size bits left).1 (ruleRoutes shape size bits right).1) :=
  unionDependencyProgram_check shape size k positive output left right bits

def elimProgram {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (width : shape.colourWidth ≤ k)
    {d : Graph} (frame : ElimFrame (routes := r) d) :
    Program (RuleCoord shape size) (3 + 2 * pointerWidth size) k where
  headerAddress bit := .inl bit
  guards _ :=
    unionGuards shape size k positive frame.output frame.minor frame.major ++
    consumeGuards shape size k width frame.minor frame.output frame.minorEdge ++
    consumeGuards shape size k width frame.major frame.output frame.majorEdge ++
    continuationGuards shape size k width d (frame.owner frame.output) frame.output

theorem elim_compiler_check_eq {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (width : shape.colourWidth ≤ k)
    {d : Graph} (frame : ElimFrame (routes := r) d)
    (bits : RuleCoord shape size → Bool) :
    (elimProgram shape size k positive width frame).check bits =
      elimAcceptance shape size frame bits := by
  rw [elim_acceptance_eq]
  simp only [Program.check, elimProgram, List.all_append, unionGuards_check,
    consumeGuards_check, continuationGuards_check]

theorem elim_compiler_failure_eq {r f : Nat} (shape : RouteShape r f) (size k : Nat)
    (positive : 0 < k) (width : shape.colourWidth ≤ k)
    {d : Graph} (frame : ElimFrame (routes := r) d)
    (bits : RuleCoord shape size → Bool) :
    (elimProgram shape size k positive width frame).failure bits =
      elimFailure shape size frame bits :=
  congrArg Bool.not (elim_compiler_check_eq shape size k positive width frame bits)

end Robustness.RuleCompilation
