import Robustness.Certificate.FailureCylinders
import Mathlib.Data.Nat.Log

/-! Bounded bit-program grammar. A semantic compiler to the Flow checker is
a separate obligation; no such compiler is postulated in these theorems. -/
open CertificateSearch.Robustness
namespace Robustness.BitLocality

/-- An atom reads at most three words, each of k bits. -/
structure Atom (Coordinate : Type) (k : Nat) where
  address : Fin 3 × Fin k → Coordinate
  test : (Fin 3 × Fin k → Bool) → Bool

def Atom.eval {Coordinate : Type} {k : Nat}
    (atom : Atom Coordinate k) (input : Coordinate → Bool) : Bool :=
  atom.test (fun bit => input (atom.address bit))

/-- Three alternatives of conjunctive atoms. False padding represents absent
alternatives; an empty conjunction is true. -/
structure Guard (Coordinate : Type) (k : Nat) where
  clauses : Fin 3 → List (Atom Coordinate k)

def Guard.check {Coordinate : Type} {k : Nat}
    (guard : Guard Coordinate k) (input : Coordinate → Bool) : Bool :=
  (List.finRange 3).any fun i => (guard.clauses i).all fun atom => atom.eval input

def Guard.failure {Coordinate : Type} {k : Nat}
    (guard : Guard Coordinate k) (input : Coordinate → Bool) : Bool :=
  !guard.check input

/-- Header bits alone choose the pointed addresses and fixed local constraints.
No bound is imposed on the number of conjuncts or path/dependency positions. -/
structure Program (Coordinate : Type) (headerWidth wordWidth : Nat) where
  headerAddress : Fin headerWidth → Coordinate
  guards : (Fin headerWidth → Bool) → List (Guard Coordinate wordWidth)

def Program.check {Coordinate : Type} {h k : Nat}
    (program : Program Coordinate h k) (input : Coordinate → Bool) : Bool :=
  (program.guards (fun bit => input (program.headerAddress bit))).all
    fun guard => guard.check input

def Program.failure {Coordinate : Type} {h k : Nat}
    (program : Program Coordinate h k) (input : Coordinate → Bool) : Bool :=
  !program.check input

/-- A failed disjunctive guard fixes one failed atom in each alternative. -/
theorem guard_failure_witness {Coordinate : Type} [DecidableEq Coordinate]
    {k : Nat} (guard : Guard Coordinate k) (anchor : Coordinate → Bool)
    (failed : guard.failure anchor = true) :
    ∃ witness : FailureWitness guard.failure anchor,
      witness.support.card ≤ 9 * k := by
  classical
  have checkFalse : guard.check anchor = false := by
    simpa only [Guard.failure, Bool.not_eq_true_eq_eq_false] using failed
  have bad : ∀ i : Fin 3, ∃ atom ∈ guard.clauses i, atom.eval anchor = false := by
    intro i
    have rowFalse := List.any_eq_false.mp checkFalse i (by simp)
    obtain ⟨atom, member, badAtom⟩ := List.all_eq_false.mp (Bool.eq_false_iff.mpr rowFalse)
    exact ⟨atom, member, Bool.eq_false_iff.mpr badAtom⟩
  choose selected member failure using bad
  let support : Finset Coordinate :=
    Finset.univ.image fun index : Fin 3 × (Fin 3 × Fin k) =>
      (selected index.1).address index.2
  have bound : support.card ≤ 9 * k := by
    calc
      support.card ≤ (Finset.univ : Finset (Fin 3 × (Fin 3 × Fin k))).card :=
        Finset.card_image_le
      _ = 9 * k := by simp; omega
  refine ⟨⟨support, failed, ?_⟩, bound⟩
  intro other agree
  have same : ∀ i, (selected i).eval other = (selected i).eval anchor := by
    intro i
    apply congrArg (selected i).test
    funext bit
    exact agree _ (Finset.mem_image.mpr ⟨(i, bit), Finset.mem_univ _, rfl⟩)
  simp only [Guard.failure, Bool.not_eq_true_eq_eq_false, Guard.check, List.any_eq_false]
  intro i _
  apply Bool.eq_false_iff.mp
  exact List.all_eq_false.mpr ⟨selected i, member i,
    Bool.eq_false_iff.mp ((same i).trans (failure i))⟩

/-- Every failing program has a one-sided header-plus-nine-word witness.
The grammar, not an assumed locality property, supplies the read bound. -/
theorem program_failure_witness {Coordinate : Type} [DecidableEq Coordinate]
    {h k : Nat} (program : Program Coordinate h k) (anchor : Coordinate → Bool)
    (failed : program.failure anchor = true) :
    ∃ witness : FailureWitness program.failure anchor,
      witness.support.card ≤ h + 9 * k := by
  classical
  have checkFalse : program.check anchor = false := by
    simpa only [Program.failure, Bool.not_eq_true_eq_eq_false] using failed
  obtain ⟨guard, member, guardFalse⟩ := List.all_eq_false.mp checkFalse
  obtain ⟨localWitness, localBound⟩ := guard_failure_witness guard anchor
    (by simpa only [Guard.failure, Bool.not_eq_true_eq_eq_false] using (Bool.eq_false_iff.mpr guardFalse))
  let header : Finset Coordinate := Finset.univ.image program.headerAddress
  let support := header ∪ localWitness.support
  have headerBound : header.card ≤ h := by
    calc
      header.card ≤ (Finset.univ : Finset (Fin h)).card := Finset.card_image_le
      _ = h := by simp
  have bound : support.card ≤ h + 9 * k :=
    (Finset.card_union_le _ _).trans (Nat.add_le_add headerBound localBound)
  refine ⟨⟨support, failed, ?_⟩, bound⟩
  intro other agree
  have sameHeader : (fun bit => other (program.headerAddress bit)) =
      (fun bit => anchor (program.headerAddress bit)) := by
    funext bit
    exact agree _ (Finset.mem_union.mpr (Or.inl
      (Finset.mem_image.mpr ⟨bit, Finset.mem_univ _, rfl⟩)))
  have sameFailure : guard.failure other = true :=
    localWitness.forcesFailure other (fun coordinate hc =>
      agree coordinate (Finset.mem_union.mpr (Or.inr hc)))
  have sameCheck : guard.check other = false := by
    simpa only [Guard.failure, Bool.not_eq_true_eq_eq_false] using sameFailure
  simp only [Program.failure, Bool.not_eq_true_eq_eq_false, Program.check, sameHeader]
  exact List.all_eq_false.mpr ⟨guard, member, Bool.eq_false_iff.mp sameCheck⟩

def pointerWidth (slots : Nat) : Nat := Nat.clog 2 (max 1 slots)
def formulaWidth (formulas : Nat) : Nat := Nat.clog 2 (max 1 formulas)
def colourWidth (colours : Nat) : Nat := Nat.clog 2 (max 1 colours)
def wordWidth (formulas colours : Nat) : Nat :=
  max 1 (max (formulaWidth formulas) (colourWidth colours))
def W (slots formulas colours : Nat) : Nat :=
  3 + 2 * pointerWidth slots + 9 * wordWidth formulas colours

/-- Widths are exactly logarithmic code widths, rather than unconstrained inputs. -/
theorem coded_failure_witness {Coordinate : Type} [DecidableEq Coordinate]
    (slots formulas colours : Nat)
    (program : Program Coordinate (3 + 2 * pointerWidth slots)
      (wordWidth formulas colours))
    (anchor : Coordinate → Bool) (failed : program.failure anchor = true) :
    ∃ witness : FailureWitness program.failure anchor,
      witness.support.card ≤ W slots formulas colours :=
  program_failure_witness program anchor failed
end Robustness.BitLocality
