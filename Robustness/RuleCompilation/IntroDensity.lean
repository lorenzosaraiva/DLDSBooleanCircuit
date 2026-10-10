import Robustness.RuleCompilation.IntroCompiler

/-! Positive local failures on the explicitly fixed-shape RuleCoord product.
No pointer, producer, edge or graph corruption is quantified here. -/
open DLDS Flow Robustness.BitLocality Robustness.SelectedProducers
open Search.Certificate.Robustness
namespace Robustness.RuleCompilation

theorem intro_failure_witness {r f : Nat} (shape : RouteShape r f)
    (size colours : Nat) (width : shape.colourWidth ≤ wordWidth f colours)
    {d : Graph} (frame : IntroFrame (routes := r) d)
    (anchor : RuleCoord shape size → Bool)
    (failed : introFailure shape size frame anchor = true) :
    ∃ witness : FailureWitness (introFailure shape size frame) anchor,
      witness.support.card ≤ W size f colours := by
  let program := introProgram shape size (wordWidth f colours)
    (wordWidth_positive f colours) width frame
  have oracle : program.failure = introFailure shape size frame :=
    funext (intro_compiler_failure_eq shape size (wordWidth f colours)
      (wordWidth_positive f colours) width frame)
  rw [← oracle] at failed ⊢
  exact coded_failure_witness size f colours program anchor failed

theorem intro_failure_density {r f : Nat} (shape : RouteShape r f)
    (size colours : Nat) (width : shape.colourWidth ≤ wordWidth f colours)
    {d : Graph} (frame : IntroFrame (routes := r) d)
    (anchor : RuleCoord shape size → Bool)
    (failed : introFailure shape size frame anchor = true) :
    (1 : ℚ) / 2 ^ W size f colours ≤
      oracleFailureDensity (introFailure shape size frame) := by
  let program := introProgram shape size (wordWidth f colours)
    (wordWidth_positive f colours) width frame
  have oracle : program.failure = introFailure shape size frame :=
    funext (intro_compiler_failure_eq shape size (wordWidth f colours)
      (wordWidth_positive f colours) width frame)
  rw [← oracle] at failed ⊢
  exact coded_failure_density size f colours program anchor failed

theorem intro_failure_inverse_polynomial {r f : Nat} (shape : RouteShape r f)
    (size colours : Nat) (width : shape.colourWidth ≤ wordWidth f colours)
    {d : Graph} (frame : IntroFrame (routes := r) d)
    (anchor : RuleCoord shape size → Bool)
    (failed : introFailure shape size frame anchor = true) :
    (1 : ℚ) / (2 ^ 23 * (M size f colours : ℚ) ^ 11) ≤
      oracleFailureDensity (introFailure shape size frame) := by
  let program := introProgram shape size (wordWidth f colours)
    (wordWidth_positive f colours) width frame
  have oracle : program.failure = introFailure shape size frame :=
    funext (intro_compiler_failure_eq shape size (wordWidth f colours)
      (wordWidth_positive f colours) width frame)
  rw [← oracle] at failed ⊢
  exact coded_failure_inverse_polynomial size f colours program anchor failed

end Robustness.RuleCompilation
