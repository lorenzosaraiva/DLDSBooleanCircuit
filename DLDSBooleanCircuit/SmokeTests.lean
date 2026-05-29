import DLDSBooleanCircuit

namespace SmokeTests

open Semantic

/-! Boolean-core theorem-style checks. -/

example : multiple_xor [] = false := by
  rfl

example : multiple_xor [true] = true := by
  rfl

example : multiple_xor [true, false, false] = true := by
  rfl

example : multiple_xor [true, true, false] = false := by
  rfl

/-! Constructor well-formedness checks. -/

theorem mkIntroRule_wellFormed
    {n : Nat} (rid : Nat) (encoder : List.Vector Bool n) (bit : Bool) :
    Rule.WellFormed (mkIntroRule rid encoder bit) := by
  rfl

theorem mkElimRule_wellFormed
    {n : Nat} (rid : Nat) (bit1 bit2 : Bool) :
    Rule.WellFormed (mkElimRule (n := n) rid bit1 bit2) := by
  rfl

theorem mkRepetitionRule_wellFormed
    {n : Nat} (rid : Nat) (bit : Bool) :
    Rule.WellFormed (mkRepetitionRule (n := n) rid bit) := by
  rfl

/-! Tiny intro-discharge DLDS, mirroring the main file's private example. -/

private def fA : Formula := .atom "A"

private def vA : Vertex :=
  { node := 0, LEVEL := 1, FORMULA := fA,
    HYPOTHESIS := true, COLLAPSED := false, PAST := [] }

private def vIntro : Vertex :=
  { node := 1, LEVEL := 0, FORMULA := .impl fA fA,
    HYPOTHESIS := false, COLLAPSED := false, PAST := [] }

private def eAtoIntro : Deduction :=
  { START := vA, END := vIntro, COLOUR := 0, DEPENDENCY := [fA] }

private def introBaseDLDS : DLDS :=
  { V := [vA, vIntro], E := [eAtoIntro], A := [(vIntro, vA)] }

private def introDischargeDLDS : BranchingDLDS :=
  { base := introBaseDLDS
    branchings := []
    numReading := 0
    evalOrder := [vA, vIntro] }

theorem introDischarge_semantics_expected :
    (dldsSemanticsAt introDischargeDLDS [] vIntro).toList = [false] := by
  native_decide

theorem introDischarge_root_not_rejected
    (reading : Robustness.Input 2 introDischargeDLDS.numReading) :
    Not (RejectsReading introDischargeDLDS vIntro reading) := by
  simp [RejectsReading, HasUndischargedDependency,
    RobustnessInput.toReadingInput, introDischargeDLDS]
  native_decide

/-! Tiny robustness sanity check. -/

private def prefix2_len1 : Robustness.Prefix 2 2 :=
  { len := 1
    hlen := by decide
    value := fun _ => ({ val := 0, isLt := by decide } : Fin 2) }

theorem prefix2_len1_cover_card :
    (Robustness.CoverFinset prefix2_len1).card = 2 := by
  rw [Robustness.card_coverFinset]
  norm_num [prefix2_len1]

end SmokeTests
