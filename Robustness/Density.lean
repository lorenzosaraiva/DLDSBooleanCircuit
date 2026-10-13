import Robustness.BitLocality

/-! Density of the bounded bit-program model. These conclusions are not
transported to certificate checks without a proved bit/semantic compiler. -/
open CertificateSearch.Robustness
namespace Robustness.BitLocality

def M (slots formulas colours : Nat) : Nat :=
  max 1 (max slots (max formulas colours))

theorem W_logarithmic (slots formulas colours : Nat) :
    W slots formulas colours ≤ 12 + 11 * Nat.clog 2 (M slots formulas colours) := by
  have hs : max 1 slots ≤ M slots formulas colours := by unfold M; omega
  have hf : max 1 formulas ≤ M slots formulas colours := by unfold M; omega
  have hc : max 1 colours ≤ M slots formulas colours := by unfold M; omega
  have ps := Nat.clog_mono_right 2 hs
  have fs := Nat.clog_mono_right 2 hf
  have cs := Nat.clog_mono_right 2 hc
  unfold W pointerWidth wordWidth formulaWidth colourWidth
  omega

private theorem pow_clog_upper (n : Nat) (positive : 0 < n) :
    2 ^ Nat.clog 2 n ≤ 2 * n := by
  by_cases small : n ≤ 1
  · have eq : n = 1 := by omega
    subst n
    norm_num
  · have above : 1 < n := by omega
    have cp : 0 < Nat.clog 2 n := Nat.clog_pos (by decide) above
    have lower := Nat.pow_pred_clog_lt_self (b := 2) (by decide) above
    have shift : (Nat.clog 2 n).pred + 1 = Nat.clog 2 n := Nat.succ_pred_eq_of_pos cp
    calc
      2 ^ Nat.clog 2 n = 2 ^ ((Nat.clog 2 n).pred + 1) := by rw [shift]
      _ = 2 ^ (Nat.clog 2 n).pred * 2 := by rw [pow_succ]
      _ ≤ 2 * n := by omega

theorem polynomial_denominator (slots formulas colours : Nat) :
    2 ^ W slots formulas colours ≤ 2 ^ 23 * M slots formulas colours ^ 11 := by
  have logBound := W_logarithmic slots formulas colours
  have positive : 0 < M slots formulas colours := by unfold M; omega
  have power := pow_clog_upper (M slots formulas colours) positive
  calc
    2 ^ W slots formulas colours ≤
        2 ^ (12 + 11 * Nat.clog 2 (M slots formulas colours)) := by gcongr; norm_num
    _ = 4096 * (2 ^ Nat.clog 2 (M slots formulas colours)) ^ 11 := by
      rw [pow_add, Nat.mul_comm 11, pow_mul]
      norm_num
    _ ≤ 4096 * (2 * M slots formulas colours) ^ 11 := by gcongr
    _ = 2 ^ 23 * M slots formulas colours ^ 11 := by
      rw [mul_pow]
      norm_num
      ring

theorem coded_failure_density {Coordinate : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    (slots formulas colours : Nat)
    (program : Program Coordinate (3 + 2 * pointerWidth slots)
      (wordWidth formulas colours))
    (anchor : Coordinate → Bool) (failed : program.failure anchor = true) :
    (1 : ℚ) / 2 ^ W slots formulas colours ≤ oracleFailureDensity program.failure := by
  obtain ⟨witness, bound⟩ := coded_failure_witness slots formulas colours program anchor failed
  have base := failureDensity_lower_bound witness
  have exponent : (2 : ℚ) ^ witness.support.card ≤ 2 ^ W slots formulas colours := by
    gcongr; norm_num
  exact (one_div_le_one_div_of_le (by positivity) exponent).trans (by simpa using base)

theorem coded_failure_inverse_polynomial {Coordinate : Type}
    [Fintype Coordinate] [DecidableEq Coordinate]
    (slots formulas colours : Nat)
    (program : Program Coordinate (3 + 2 * pointerWidth slots)
      (wordWidth formulas colours))
    (anchor : Coordinate → Bool) (failed : program.failure anchor = true) :
    (1 : ℚ) / (2 ^ 23 * (M slots formulas colours : ℚ) ^ 11) ≤
      oracleFailureDensity program.failure := by
  have power : (2 : ℚ) ^ W slots formulas colours ≤
      2 ^ 23 * (M slots formulas colours : ℚ) ^ 11 := by
    exact_mod_cast polynomial_denominator slots formulas colours
  exact (one_div_le_one_div_of_le (by positivity) power).trans
    (coded_failure_density slots formulas colours program anchor failed)
end Robustness.BitLocality
