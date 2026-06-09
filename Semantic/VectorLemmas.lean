import Semantic.Boolean

open scoped Classical

namespace Semantic

/-!
# List and vector lemmas used by the circuit proofs.
-/

/-- OR-ing a zero vector on the left is identity. -/
lemma zip_with_zero_identity :
    ∀ (N : ℕ) (v : List.Vector Bool N),
      (List.Vector.replicate N false).zipWith (· || ·) v = v := by
  intro N v
  let ⟨l, hl⟩ := v
  dsimp [List.Vector.zipWith, List.Vector.replicate]
  congr
  induction l generalizing N with
  | nil => simp
  | cons hd tl ih =>
    rw [←hl]
    have hlen : (hd :: tl).length = Nat.succ (List.length tl) := by simp [List.length]
    rw [hlen] at *
    have hrep : List.replicate (Nat.succ (List.length tl)) false =
                false :: List.replicate (List.length tl) false := by simp [List.replicate]
    rw [hrep, List.zipWith_cons_cons]
    simp
    exact ih (List.length tl) ⟨tl, rfl⟩ rfl

@[simp]
theorem List.getElem_eq_get {α : Type*} (l : List α) (i : Fin l.length) :
    l[↑i] = l.get i := rfl

/-- zipWith is commutative for commutative operations. -/
@[simp]
lemma List.Vector.zipWith_comm {n : ℕ} (f : Bool → Bool → Bool)
    (h : ∀ x y, f x y = f y x)
    (v₁ v₂ : List.Vector Bool n) :
    v₁.zipWith f v₂ = v₂.zipWith f v₁ := by
  rcases v₁ with ⟨l₁, h₁⟩
  rcases v₂ with ⟨l₂, h₂⟩
  apply List.Vector.ext
  intro i
  dsimp [List.Vector.zipWith, List.Vector.get]
  rw [List.getElem_zipWith, List.getElem_zipWith]
  apply h

/-- Folding OR with zero vectors preserves the accumulator. -/
lemma foldl_add_false {n : ℕ} (v : List.Vector Bool n) (l : List α) :
    List.foldl (fun acc (_ : α) => acc.zipWith (· || ·) (List.Vector.replicate n false)) v l = v := by
  induction l with
  | nil => rfl
  | cons _ tl ih =>
    simp only [List.foldl]
    rw [List.Vector.zipWith_comm (· || ·) Bool.or_comm]
    rw [zip_with_zero_identity]
    exact ih

lemma exists_fin_of_mem {α} {a : α} {l : List α} (h : a ∈ l) :
  ∃ i : Fin l.length, l.get i = a := by
  classical
  induction' l with x xs ih generalizing a
  · cases h
  · cases h with
    | head =>
        exact ⟨⟨0, by simp⟩, rfl⟩
    | tail =>
        have h' : a ∈ xs := by assumption
        obtain ⟨i, hi⟩ := ih h'
        refine ⟨⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩, ?_⟩
        simp only [List.get_cons_succ]
        exact hi


lemma List.length_zipWith3 {α β γ δ : Type*} (f : α → β → γ → δ)
    (as : List α) (bs : List β) (cs : List γ) :
    (List.zipWith3 f as bs cs).length = min as.length (min bs.length cs.length) := by
  induction as generalizing bs cs with
  | nil => simp [List.zipWith3]
  | cons a as' ih =>
    cases bs with
    | nil => simp [List.zipWith3]
    | cons b bs' =>
      cases cs with
      | nil => simp [List.zipWith3]
      | cons c cs' =>
        simp only [List.zipWith3, List.length_cons, ih]
        omega

lemma List.get_zipWith3 {α β γ δ : Type*} (f : α → β → γ → δ)
    (as : List α) (bs : List β) (cs : List γ) (i : Nat)
    (ha : i < as.length) (hb : i < bs.length) (hc : i < cs.length) :
    (List.zipWith3 f as bs cs).get ⟨i, by rw [List.length_zipWith3]; omega⟩ =
    f (as.get ⟨i, ha⟩) (bs.get ⟨i, hb⟩) (cs.get ⟨i, hc⟩) := by
  induction as generalizing bs cs i with
  | nil => simp at ha
  | cons a as' ih =>
    cases bs with
    | nil => simp at hb
    | cons b bs' =>
      cases cs with
      | nil => simp at hc
      | cons c cs' =>
        cases i with
        | zero => simp [List.zipWith3]
        | succ i' =>
          simp only [List.zipWith3, List.get_cons_succ]
          have ha' : i' < as'.length := Nat.lt_of_succ_lt_succ ha
          have hb' : i' < bs'.length := Nat.lt_of_succ_lt_succ hb
          have hc' : i' < cs'.length := Nat.lt_of_succ_lt_succ hc
          exact ih bs' cs' i' ha' hb' hc'

lemma Vector.zipWith_or_replicate_false_left {n : Nat} (v : List.Vector Bool n) :
    List.Vector.zipWith (· || ·) (List.Vector.replicate n false) v = v := by
  apply List.Vector.ext
  intro i
  simp [List.Vector.get_replicate]

lemma Vector.zipWith_or_replicate_false_right {n : Nat} (v : List.Vector Bool n) :
    List.Vector.zipWith (· || ·) v (List.Vector.replicate n false) = v := by
  apply List.Vector.ext
  intro i
  simp [List.Vector.get_replicate]

lemma foldl_zipWith_or_all_zeros {n : Nat} (acc : List.Vector Bool n) (vecs : List (List.Vector Bool n))
    (h_all_zero : ∀ j (hj : j < vecs.length), vecs.get ⟨j, hj⟩ = List.Vector.replicate n false) :
    List.foldl (fun a v => List.Vector.zipWith (· || ·) a v) acc vecs = acc := by
  induction vecs generalizing acc with
  | nil => rfl
  | cons v vs ih =>
    simp only [List.foldl_cons]
    have hv : v = List.Vector.replicate n false := h_all_zero 0 (by simp)
    rw [hv, Vector.zipWith_or_replicate_false_right]
    apply ih
    intro j hj
    have := h_all_zero (j + 1) (by simp; omega)
    simpa using this


lemma list_or_single_nonzero {n : Nat} (vecs : List (List.Vector Bool n))
    (i : Nat) (hi : i < vecs.length)
    (h_others : ∀ j (hj : j < vecs.length), j ≠ i →
        vecs.get ⟨j, hj⟩ = List.Vector.replicate n false) :
    List.foldl (fun acc v => List.Vector.zipWith (· || ·) acc v)
      (List.Vector.replicate n false) vecs = vecs.get ⟨i, hi⟩ := by
  induction vecs generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    simp only [List.foldl_cons, List.length_cons] at hi ⊢
    cases i with
    | zero =>
      -- x is the active element, all of xs are zeros
      have h_xs_zero : ∀ j (hj : j < xs.length),
          xs.get ⟨j, hj⟩ = List.Vector.replicate n false := by
        intro j hj
        have hj' : j + 1 < (x :: xs).length := by simp; omega
        have := h_others (j + 1) hj' (by omega)
        simpa using this
      rw [Vector.zipWith_or_replicate_false_left]
      rw [foldl_zipWith_or_all_zeros x xs h_xs_zero]
      simp
    | succ i' =>
      have hx_zero : x = List.Vector.replicate n false := by
        have h0 : 0 < (x :: xs).length := by simp
        have := h_others 0 h0 (by omega)
        simpa using this
      rw [hx_zero, Vector.zipWith_or_replicate_false_left]
      have hi' : i' < xs.length := Nat.lt_of_succ_lt_succ hi
      have h_others' : ∀ j (hj : j < xs.length), j ≠ i' →
          xs.get ⟨j, hj⟩ = List.Vector.replicate n false := by
        intro j hj hne
        have hj' : j + 1 < (x :: xs).length := by simp; omega
        have := h_others (j + 1) hj' (by omega)
        simpa using this
      have := ih i' hi' h_others'
      convert this using 1

lemma list_map_get {α β : Type*} (f : α → β) (l : List α) (i : Nat)
    (hi : i < l.length) (hi' : i < (l.map f).length) :
    (l.map f).get ⟨i, hi'⟩ = f (l.get ⟨i, hi⟩) := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    cases i with
    | zero => rfl
    | succ i' =>
      simp only [List.map_cons, List.get_cons_succ]
      have hi'_xs : i' < xs.length := Nat.lt_of_succ_lt_succ hi
      have hi'_map : i' < (xs.map f).length := by simp; exact hi'_xs
      exact ih i' hi'_xs hi'_map

lemma list_zipIdx_get_fst {α : Type*} (l : List α) (n : Nat) (i : Nat)
    (hi : i < (l.zipIdx n).length)
    (hi' : i < l.length := by simp [List.length_zipIdx] at hi; exact hi) :
    ((l.zipIdx n).get ⟨i, hi⟩).1 = l.get ⟨i, hi'⟩ := by
  induction l generalizing n i with
  | nil => simp at hi'
  | cons x xs ih =>
    cases i with
    | zero => simp
    | succ i' => simp only [List.zipIdx_cons, List.get_cons_succ]; apply ih


lemma list_range_get (n : Nat) (i : Nat) (hi : i < (List.range n).length) :
    (List.range n).get ⟨i, hi⟩ = i := by
  simp at hi ⊢


end Semantic
