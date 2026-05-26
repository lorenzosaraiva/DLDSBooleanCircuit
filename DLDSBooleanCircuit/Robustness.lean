import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic

namespace Robustness

/--
A global input consists of r local choices,
each one selecting among m possibilities.
-/
abbrev Input (m r : Nat) := Fin r → Fin m

/--
A prefix fixes the first len choices.
-/
structure Prefix (m r : Nat) where
  len : Nat
  hlen : len ≤ r
  value : Fin len → Fin m

/--
An input x extends a prefix π
iff x agrees with π on all fixed positions.
-/
def Extends {m r : Nat}
  (x : Input m r)
  (π : Prefix m r) : Prop :=
  ∀ i : Fin π.len,
    x ⟨i.val, Nat.lt_of_lt_of_le i.isLt π.hlen⟩
      = π.value i

/--
Finite version of the cover.
-/
noncomputable def CoverFinset {m r : Nat}
  (π : Prefix m r) : Finset (Input m r) := by
  classical
  exact Finset.univ.filter fun x => Extends x π

/--
Finite version of rejected inputs.
-/
noncomputable def BadFinset {m r : Nat}
  (rejects : Input m r → Prop) : Finset (Input m r) := by
  classical
  exact Finset.univ.filter rejects

/--
A bad prefix:
every input in its finite cover is rejected.
-/
def BadPrefix {m r : Nat}
  (rejects : Input m r → Prop)
  (π : Prefix m r) : Prop :=
  ∀ x, x ∈ CoverFinset π → rejects x

/--
If π is a bad prefix,
then its cover is contained in the rejected inputs.
-/
theorem coverFinset_subset_badFinset
  {m r : Nat}
  (rejects : Input m r → Prop)
  (π : Prefix m r)
  (hπ : BadPrefix rejects π) :
  CoverFinset π ⊆ BadFinset rejects := by
  classical
  intro x hx
  simp [BadFinset, hπ x hx]

/--
Therefore the number of rejected inputs
is at least the size of the cover.
-/
theorem card_cover_le_card_bad
  {m r : Nat}
  (rejects : Input m r → Prop)
  (π : Prefix m r)
  (hπ : BadPrefix rejects π) :
  (CoverFinset π).card ≤ (BadFinset rejects).card := by
  classical
  exact Finset.card_le_card
    (coverFinset_subset_badFinset rejects π hπ)

/--
Sanity check:
the total number of global inputs is m^r.
-/
example (m r : Nat) :
  Fintype.card (Input m r) = m ^ r := by
  simp [Input]

/-- Split the positions of an input into the prefix positions and the remaining tail. -/
noncomputable def prefixIndexEquiv {m r : Nat}
  (π : Prefix m r) :
  Fin π.len ⊕ Fin (r - π.len) ≃ Fin r :=
  finSumFinEquiv.trans (finCongr (Nat.add_sub_of_le π.hlen))

/--
The subtype of inputs extending a prefix is equivalent to assignments on the
remaining positions.
-/
noncomputable def coverSubtypeEquiv
  {m r : Nat}
  (π : Prefix m r) :
  {x : Input m r // Extends x π} ≃ (Fin (r - π.len) → Fin m) where
  toFun x j := x.1 (prefixIndexEquiv π (Sum.inr j))
  invFun tail :=
    ⟨fun i =>
      match (prefixIndexEquiv π).symm i with
      | Sum.inl k => π.value k
      | Sum.inr j => tail j,
    by
      intro k
      have hk :
          (prefixIndexEquiv π).symm
              ⟨k.val, Nat.lt_of_lt_of_le k.isLt π.hlen⟩ = Sum.inl k := by
        apply (prefixIndexEquiv π).injective
        apply Fin.ext
        simp [prefixIndexEquiv]
      simp [hk]⟩
  left_inv x := by
    apply Subtype.ext
    funext i
    dsimp
    change
      (match (prefixIndexEquiv π).symm i with
      | Sum.inl k => π.value k
      | Sum.inr j => x.1 (prefixIndexEquiv π (Sum.inr j))) = x.1 i
    cases h : (prefixIndexEquiv π).symm i with
    | inl k =>
        have hi : i = prefixIndexEquiv π (Sum.inl k) := by
          calc
            i = prefixIndexEquiv π ((prefixIndexEquiv π).symm i) := by simp
            _ = prefixIndexEquiv π (Sum.inl k) := by rw [h]
        have hprefix : x.1 (prefixIndexEquiv π (Sum.inl k)) = π.value k := by
          have hx := x.2 k
          convert hx using 2
        rw [hi]
        exact hprefix.symm
    | inr j =>
        have hi : i = prefixIndexEquiv π (Sum.inr j) := by
          calc
            i = prefixIndexEquiv π ((prefixIndexEquiv π).symm i) := by simp
            _ = prefixIndexEquiv π (Sum.inr j) := by rw [h]
        rw [hi]
  right_inv tail := by
    funext j
    simp [prefixIndexEquiv]

/-- The cover of a prefix has one degree of freedom for each remaining position. -/
theorem card_coverFinset
  {m r : Nat}
  (π : Prefix m r) :
  (CoverFinset π).card = m ^ (r - π.len) := by
  classical
  calc
    (CoverFinset π).card = Fintype.card {x : Input m r // Extends x π} := by
      simp [CoverFinset, Fintype.card_subtype]
    _ = Fintype.card (Fin (r - π.len) → Fin m) := by
      exact Fintype.card_congr (coverSubtypeEquiv π)
    _ = m ^ (r - π.len) := by
      simp

/--
If π is a bad prefix, then there are at least `m ^ (r - π.len)`
rejected inputs.
-/
theorem bad_inputs_lower_bound
  {m r : Nat}
  (rejects : Input m r → Prop)
  (π : Prefix m r)
  (hπ : BadPrefix rejects π) :
  m ^ (r - π.len) ≤ (BadFinset rejects).card := by
  rw [← card_coverFinset π]
  exact card_cover_le_card_bad rejects π hπ

/--
Fixed-constant lower bound for any bad prefix whose length is at most `C`.
-/
theorem bad_inputs_lower_bound_fixed_C
  {m r C : Nat}
  (hm : 1 ≤ m)
  (rejects : Input m r → Prop)
  (π : Prefix m r)
  (hπ : BadPrefix rejects π)
  (hC : π.len ≤ C) :
  m ^ (r - C) ≤ (BadFinset rejects).card := by
  have hlower :
      m ^ (r - π.len) ≤ (BadFinset rejects).card :=
    bad_inputs_lower_bound rejects π hπ
  have hExp :
      r - C ≤ r - π.len :=
    Nat.sub_le_sub_left hC r
  have hmono :
      m ^ (r - C) ≤ m ^ (r - π.len) := by
    exact Nat.pow_le_pow_right hm hExp
  exact le_trans hmono hlower

/--
Multiplication-form density lower bound, avoiding division.
-/
theorem bad_inputs_density_fixed_C_mul
  {m r C : Nat}
  (hm : 1 ≤ m)
  (rejects : Input m r → Prop)
  (π : Prefix m r)
  (hπ : BadPrefix rejects π)
  (hC : π.len ≤ C)
  (hCr : C ≤ r) :
  m ^ r ≤ (BadFinset rejects).card * m ^ C := by
  have hlower :
      m ^ (r - C) ≤ (BadFinset rejects).card :=
    bad_inputs_lower_bound_fixed_C hm rejects π hπ hC
  have hmul :
      m ^ (r - C) * m ^ C ≤ (BadFinset rejects).card * m ^ C :=
    Nat.mul_le_mul_right (m ^ C) hlower
  simpa [← Nat.pow_add, Nat.sub_add_cancel hCr] using hmul

/--
A local obstruction:
a bounded fragment forcing rejection.
-/
structure LocalObstruction
  (m r : Nat) where
  pref : Prefix m r
  bound : Nat
  hbound : pref.len ≤ bound
  hbound_height : bound ≤ r

/--
A local obstruction is sound
if every input in the cover of its prefix is rejected.
-/
def ObstructionSound
  {m r : Nat}
  (rejects : Input m r → Prop)
  (O : LocalObstruction m r) : Prop :=
  ∀ x, x ∈ CoverFinset O.pref → rejects x

/--
Every sound local obstruction induces a bad prefix.
-/
theorem obstruction_gives_badPrefix
  {m r : Nat}
  (rejects : Input m r → Prop)
  (O : LocalObstruction m r)
  (hO : ObstructionSound rejects O) :
  BadPrefix rejects O.pref := by
  intro x hx
  exact hO x hx

/-
TODO Integration:
Instantiate `rejects` with the actual reading-level rejection predicate
once the tagged-grid ↔ reading correspondence theorem is stabilized.
-/

end Robustness
