import Robustness.Density
import Robustness.ExpandedRoutes

/-! Kernel-reduced sanity for the bounded bit grammar only.
A dependency-union bit reads exactly the output bit and two premise bits.
This is not an instantiation of the unfinished certificate/bit compiler. -/
open Search.Certificate.Robustness
namespace Robustness.Sanity
open BitLocality

def unionAtom : Atom (Fin 3) 1 where
  address bit := bit.1
  test bits := bits (0, 0) == (bits (1, 0) || bits (2, 0))

def unionGuard : Guard (Fin 3) 1 where
  clauses _ := [unionAtom]

def unionProgram : Program (Fin 3) 3 1 where
  headerAddress := id
  guards _ := [unionGuard]

def valid : Fin 3 → Bool := fun _ => false
def corrupt : Fin 3 → Bool := fun bit => bit == 0

theorem valid_passes : unionProgram.check valid = true := by decide
theorem corrupt_fails : unionProgram.failure corrupt = true := by decide

theorem corrupt_has_bounded_witness :
    ∃ witness : FailureWitness unionProgram.failure corrupt,
      witness.support.card ≤ W 1 1 1 :=
  coded_failure_witness 1 1 1 unionProgram corrupt corrupt_fails
end Robustness.Sanity
