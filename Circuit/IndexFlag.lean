import Circuit.Controlled
import Circuit.SharedAssembly

namespace Semantic.Circuit

def indexState (width value : Nat) : Bits := fun wire =>
  if wire < width then (BitVec.ofNat width value).getLsbD wire else false

def indexFlagProgram (width key : Nat) : Shared.Program width :=
  Shared.expressionProgram (equalConstant (fun i : Fin width => i)
    (fun i => (BitVec.ofNat width key).getLsbD i.val))

theorem indexFlag_value (width key value : Nat) (keyBound : key < 2 ^ width)
    (valueBound : value < 2 ^ width) :
    (indexFlagProgram width key).value (indexState width value) = decide (value = key) := by
  rw [indexFlagProgram, Shared.expressionProgram_value, equalConstant_value]
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_finRange, true_implies, beq_iff_eq, decide_eq_true_eq]
  constructor
  · intro equal
    have bits : BitVec.ofNat width value = BitVec.ofNat width key := by
      apply BitVec.eq_of_getLsbD_eq
      intro i bound
      simpa [indexState, bound] using equal ⟨i, bound⟩
    have values := congrArg BitVec.toNat bits
    simpa [Nat.mod_eq_of_lt keyBound, Nat.mod_eq_of_lt valueBound] using values
  · rintro rfl i
    simp [indexState, i.isLt]


def placeWork (inputs base wire : Nat) : Nat :=
  if wire < inputs then wire else base + (wire - inputs)

theorem placeWork_injective (inputs base : Nat) (space : inputs ≤ base) :
    Function.Injective (placeWork inputs base) := by
  intro a b equal
  unfold placeWork at equal
  split_ifs at equal <;> omega

theorem indexFlag_placed (width key value base : Nat) (space : width < base)
    (keyBound : key < 2 ^ width) (valueBound : value < 2 ^ width)
    (state : Bits) (inputs : ∀ wire, wire < width → state wire = indexState width value wire)
    (clean : ∀ wire, base ≤ wire → state wire = false) :
    eval (rename (placeWork width base) (placeWork_injective width base (by omega))
      (indexFlagProgram width key).oracle) state =
      Function.update state (placeWork width base (indexFlagProgram width key).target)
        (decide (value = key)) := by
  have view : state ∘ placeWork width base = indexState width value := by
    funext wire
    by_cases h : wire < width
    · simp [placeWork, h, inputs wire h]
    · change state (placeWork width base wire) = indexState width value wire
      rw [placeWork, if_neg h, clean (base + (wire - width)) (by omega)]
      simp [indexState, h]
  apply rename_update
  rw [view, Shared.Program.oracle_correct _ _ (by intro wire h; simp [indexState, Nat.not_lt.mpr h]),
    indexFlag_value width key value keyBound valueBound]

end Semantic.Circuit
