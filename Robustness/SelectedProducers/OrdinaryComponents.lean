import Robustness.SelectedProducers.SeedCompiler

/-! Ordinary-rule dependency components of the compiler.
The proved equations below cover normalized dependency production only.
These equations concern dependencies; residual constraints and premise-address
decoding are separate properties. -/
open DLDS Flow Robustness.BitLocality
namespace Robustness.SelectedProducers
set_option maxHeartbeats 2000000

theorem decodeDependency_injective {routes formulas : Nat}
    (shape : RouteShape routes formulas) :
    Function.Injective (decodeDependency shape) := by
  intro left right same
  funext code
  apply Bool.eq_iff_iff.mpr
  rw [← decoded_membership shape left code, ← decoded_membership shape right code, same]

theorem decode_union {routes formulas : Nat} (shape : RouteShape routes formulas)
    (left right : Fin formulas → Bool) :
    decodeDependency shape (fun code => left code || right code) =
      depUnion (decodeDependency shape left) (decodeDependency shape right) := by
  apply canonicalDep_eq_of_depSetEq (normalizeDep_canonical _) (depUnion_canonical _ _)
  apply (depSetEq_eq_true_iff _ _).mpr
  intro formula
  simp only [decodeDependency, depUnion, mem_normalizeDep, List.mem_append,
    List.mem_map, List.mem_filter, List.mem_finRange, true_and, Bool.or_eq_true]
  aesop

theorem decode_remove {routes formulas : Nat} (shape : RouteShape routes formulas)
    (input : Fin formulas → Bool) (discharged : Formula) :
    decodeDependency shape (fun code => input code && !decide (shape.formulaAt code = discharged)) =
      depRemove (decodeDependency shape input) discharged := by
  apply canonicalDep_eq_of_depSetEq (normalizeDep_canonical _) (depRemove_canonical _ _)
  apply (depSetEq_eq_true_iff _ _).mpr
  intro formula
  simp [decodeDependency, depRemove, List.removeAll, Bool.and_eq_true]
  aesop

def unionBitAtom {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (output left right : Fin routes)
    (code : Fin formulas) : Atom (SeedCoord shape size) k where
  address index := .inr (.inl
    ((if index.1 = 0 then output else if index.1 = 1 then left else right), code))
  test bits := bits (0, ⟨0, positive⟩) == (bits (1, ⟨0, positive⟩) || bits (2, ⟨0, positive⟩))

def unionDependencyProgram {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (output left right : Fin routes) :
    Program (SeedCoord shape size) (3 + 2 * pointerWidth size) k where
  headerAddress bit := .inl bit
  guards _ := (List.finRange formulas).map fun code =>
    atomGuard (unionBitAtom shape size k positive output left right code)

theorem unionDependencyProgram_check {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (output left right : Fin routes)
    (bits : SeedCoord shape size → Bool) :
    (unionDependencyProgram shape size k positive output left right).check bits =
      decide ((decodeRoutes shape (fun coordinate => bits (.inr coordinate)) output).1 =
        depUnion (decodeRoutes shape (fun coordinate => bits (.inr coordinate)) left).1
          (decodeRoutes shape (fun coordinate => bits (.inr coordinate)) right).1) := by
  apply Bool.eq_iff_iff.mpr
  simp only [Program.check, unionDependencyProgram, List.all_map, Function.comp_apply,
    List.all_eq_true, List.mem_finRange, atomGuard_check, Atom.eval, unionBitAtom]
  simp only [true_implies, Fin.reduceEq, if_true, if_false,
    beq_iff_eq, decide_eq_true_eq]
  change (∀ code, bits (.inr (.inl (output, code))) =
    (bits (.inr (.inl (left, code))) || bits (.inr (.inl (right, code))))) ↔ _
  dsimp only [decodeRoutes]
  rw [← decode_union]
  constructor
  · intro allBits
    exact congrArg (decodeDependency shape) (funext allBits)
  · intro same
    exact congrFun (decodeDependency_injective shape same)

end Robustness.SelectedProducers
