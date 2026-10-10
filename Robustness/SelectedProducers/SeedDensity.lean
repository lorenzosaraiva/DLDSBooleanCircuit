import Robustness.SelectedProducers.SeedCompiler

/-! Failure cylinders and density for the actual decoded selected seed checker.
Both SeedMode.default and SeedMode.ancestor are covered. No coverage, absence
or root-discharge conclusion follows from these positive local results. -/
open DLDS Flow Robustness.BitLocality Search.Certificate.Robustness
namespace Robustness.SelectedProducers

theorem wordWidth_positive (formulas colours : Nat) : 0 < wordWidth formulas colours := by
  unfold wordWidth
  omega

theorem seed_failure_witness {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size colours : Nat) (width : shape.colourWidth ≤ wordWidth formulas colours)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (anchor : SeedCoord shape size → Bool)
    (failed : seedFailure shape d owner listed slot mode size anchor = true) :
    ∃ witness : FailureWitness (seedFailure shape d owner listed slot mode size) anchor,
      witness.support.card ≤ W size formulas colours := by
  let program := seedProgram shape size (wordWidth formulas colours)
    (wordWidth_positive formulas colours) width d (owner slot) slot mode
  have oracle : program.failure = seedFailure shape d owner listed slot mode size :=
    funext (seed_compiler_failure_eq shape size (wordWidth formulas colours)
      (wordWidth_positive formulas colours) width d owner listed slot mode)
  rw [← oracle] at failed ⊢
  exact coded_failure_witness size formulas colours program anchor failed

theorem seed_failure_density {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size colours : Nat) (width : shape.colourWidth ≤ wordWidth formulas colours)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (anchor : SeedCoord shape size → Bool)
    (failed : seedFailure shape d owner listed slot mode size anchor = true) :
    (1 : ℚ) / 2 ^ W size formulas colours ≤
      oracleFailureDensity (seedFailure shape d owner listed slot mode size) := by
  let program := seedProgram shape size (wordWidth formulas colours)
    (wordWidth_positive formulas colours) width d (owner slot) slot mode
  have oracle : program.failure = seedFailure shape d owner listed slot mode size :=
    funext (seed_compiler_failure_eq shape size (wordWidth formulas colours)
      (wordWidth_positive formulas colours) width d owner listed slot mode)
  rw [← oracle] at failed ⊢
  exact coded_failure_density size formulas colours program anchor failed

theorem seed_failure_inverse_polynomial {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size colours : Nat) (width : shape.colourWidth ≤ wordWidth formulas colours)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (anchor : SeedCoord shape size → Bool)
    (failed : seedFailure shape d owner listed slot mode size anchor = true) :
    (1 : ℚ) / (2 ^ 23 * (M size formulas colours : ℚ) ^ 11) ≤
      oracleFailureDensity (seedFailure shape d owner listed slot mode size) := by
  let program := seedProgram shape size (wordWidth formulas colours)
    (wordWidth_positive formulas colours) width d (owner slot) slot mode
  have oracle : program.failure = seedFailure shape d owner listed slot mode size :=
    funext (seed_compiler_failure_eq shape size (wordWidth formulas colours)
      (wordWidth_positive formulas colours) width d owner listed slot mode)
  rw [← oracle] at failed ⊢
  exact coded_failure_inverse_polynomial size formulas colours program anchor failed

end Robustness.SelectedProducers
