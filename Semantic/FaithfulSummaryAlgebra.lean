import Semantic.LambdaFlowModel

open Semantic FlowSpec

namespace FaithfulKeyed


def HasKey (xs : List FlowPair) (p : ColourPath) : Prop :=
  ∃ r ∈ xs, r.2 = p

def HasBit (xs : List FlowPair) (p : ColourPath) (f : Formula) : Prop :=
  ∃ r ∈ xs, r.2 = p ∧ f ∈ r.1



def SummaryEq (xs ys : List FlowPair) : Prop :=
  (∀ p, HasKey xs p ↔ HasKey ys p) ∧
  (∀ p f, HasBit xs p f ↔ HasBit ys p f)

theorem SummaryEq.refl (xs : List FlowPair) : SummaryEq xs xs :=
  ⟨fun _ => Iff.rfl, fun _ _ => Iff.rfl⟩

theorem SummaryEq.symm {xs ys : List FlowPair} (h : SummaryEq xs ys) :
    SummaryEq ys xs := ⟨fun p => (h.1 p).symm, fun p f => (h.2 p f).symm⟩

theorem SummaryEq.trans {xs ys zs : List FlowPair}
    (h : SummaryEq xs ys) (k : SummaryEq ys zs) : SummaryEq xs zs :=
  ⟨fun p => (h.1 p).trans (k.1 p), fun p f => (h.2 p f).trans (k.2 p f)⟩

theorem SummaryEq.of_mem_iff {xs ys : List FlowPair}
    (h : ∀ r, r ∈ xs ↔ r ∈ ys) : SummaryEq xs ys := by
  constructor
  · intro p
    simp only [HasKey, h]
  · intro p f
    simp only [HasBit, h]

theorem SummaryEq.append {xs ys us vs : List FlowPair}
    (h : SummaryEq xs ys) (k : SummaryEq us vs) :
    SummaryEq (xs ++ us) (ys ++ vs) := by
  constructor
  · intro p
    simpa only [HasKey, List.mem_append, or_and_right, exists_or] using
      or_congr (h.1 p) (k.1 p)
  · intro p f
    simpa only [HasBit, List.mem_append, or_and_right, exists_or] using
      or_congr (h.2 p f) (k.2 p f)

theorem SummaryEq.eraseDups {xs ys : List FlowPair} (h : SummaryEq xs ys) :
    SummaryEq xs.eraseDups ys.eraseDups := by
  simpa only [SummaryEq, HasKey, HasBit, List.mem_eraseDups] using h

theorem SummaryEq.flatMap {α : Type} (as : List α)
    (f g : α → List FlowPair) (h : ∀ a ∈ as, SummaryEq (f a) (g a)) :
    SummaryEq (as.flatMap f) (as.flatMap g) := by
  induction as with
  | nil => exact SummaryEq.refl _
  | cons a as ih =>
      simpa only [List.flatMap_cons] using
        (h a (by simp)).append (ih (fun b hb => h b (by simp [hb])))

@[simp] theorem mem_depUnion (f : Formula) (xs ys : Dep) :
    f ∈ depUnion xs ys ↔ f ∈ xs ∨ f ∈ ys := by
  simp [depUnion]

@[simp] theorem mem_depRemove (f a : Formula) (xs : Dep) :
    f ∈ depRemove xs a ↔ f ∈ xs ∧ f ≠ a := by
  simp [depRemove, List.removeAll]


def unary (xs : List FlowPair) (a : Formula)
    (next : ColourPath → List ColourPath) : List FlowPair :=
  xs.flatMap fun r => (next r.2).map fun p => (depRemove r.1 a, p)


def binary (xs ys : List FlowPair)
    (next : ColourPath → ColourPath → List ColourPath) : List FlowPair :=
  xs.flatMap fun l => ys.flatMap fun r =>
    (next l.2 r.2).map fun p => (depUnion l.1 r.1, p)

theorem unary_key (xs : List FlowPair) (a : Formula)
    (next : ColourPath → List ColourPath) (q : ColourPath) :
    HasKey (unary xs a next) q ↔ ∃ p, HasKey xs p ∧ q ∈ next p := by
  simp only [HasKey, unary, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨o, ⟨r, hr, p, hp, rfl⟩, rfl⟩
    exact ⟨r.2, ⟨r, hr, rfl⟩, hp⟩
  · rintro ⟨p, ⟨r, hr, rfl⟩, hp⟩
    exact ⟨(depRemove r.1 a, q), ⟨r, hr, q, hp, rfl⟩, rfl⟩

theorem unary_bit (xs : List FlowPair) (a : Formula)
    (next : ColourPath → List ColourPath) (q : ColourPath) (f : Formula) :
    HasBit (unary xs a next) q f ↔
      f ≠ a ∧ ∃ p, HasBit xs p f ∧ q ∈ next p := by
  simp only [HasBit, unary, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨o, ⟨r, hr, p, hp, rfl⟩, rfl, hf⟩
    have hm := (mem_depRemove f a r.1).mp hf
    exact ⟨hm.2, r.2, ⟨r, hr, rfl, hm.1⟩, hp⟩
  · rintro ⟨hne, p, ⟨r, hr, rfl, hf⟩, hp⟩
    exact ⟨(depRemove r.1 a, q), ⟨r, hr, q, hp, rfl⟩, rfl,
      (mem_depRemove f a r.1).mpr ⟨hf, hne⟩⟩

theorem SummaryEq.unary {xs ys : List FlowPair} (h : SummaryEq xs ys)
    (a : Formula) (next : ColourPath → List ColourPath) :
    SummaryEq (unary xs a next) (unary ys a next) := by
  constructor
  · intro q
    simp only [unary_key, h.1]
  · intro q f
    simp only [unary_bit, h.2]

theorem binary_key (xs ys : List FlowPair)
    (next : ColourPath → ColourPath → List ColourPath) (q : ColourPath) :
    HasKey (binary xs ys next) q ↔
      ∃ p t, HasKey xs p ∧ HasKey ys t ∧ q ∈ next p t := by
  simp only [HasKey, binary, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨o, ⟨l, hl, r, hr, p, hp, rfl⟩, rfl⟩
    exact ⟨l.2, r.2, ⟨l, hl, rfl⟩, ⟨r, hr, rfl⟩, hp⟩
  · rintro ⟨p, t, ⟨l, hl, rfl⟩, ⟨r, hr, rfl⟩, hp⟩
    exact ⟨(depUnion l.1 r.1, q), ⟨l, hl, r, hr, q, hp, rfl⟩, rfl⟩

theorem binary_bit (xs ys : List FlowPair)
    (next : ColourPath → ColourPath → List ColourPath) (q : ColourPath) (f : Formula) :
    HasBit (binary xs ys next) q f ↔
      ∃ p t, ((HasBit xs p f ∧ HasKey ys t) ∨
        (HasKey xs p ∧ HasBit ys t f)) ∧ q ∈ next p t := by
  simp only [HasBit, HasKey, binary, List.mem_flatMap, List.mem_map]
  constructor
  · rintro ⟨o, ⟨l, hl, r, hr, p, hp, rfl⟩, rfl, hf⟩
    refine ⟨l.2, r.2, ?_, hp⟩
    rcases (mem_depUnion f l.1 r.1).mp hf with hf | hf
    · exact Or.inl ⟨⟨l, hl, rfl, hf⟩, ⟨r, hr, rfl⟩⟩
    · exact Or.inr ⟨⟨l, hl, rfl⟩, ⟨r, hr, rfl, hf⟩⟩
  · rintro ⟨p, t, h, hp⟩
    rcases h with h | h
    · rcases h with ⟨⟨l, hl, rfl, hf⟩, ⟨r, hr, rfl⟩⟩
      exact ⟨(depUnion l.1 r.1, q), ⟨l, hl, r, hr, q, hp, rfl⟩, rfl,
        (mem_depUnion f l.1 r.1).mpr (Or.inl hf)⟩
    · rcases h with ⟨⟨l, hl, rfl⟩, ⟨r, hr, rfl, hf⟩⟩
      exact ⟨(depUnion l.1 r.1, q), ⟨l, hl, r, hr, q, hp, rfl⟩, rfl,
        (mem_depUnion f l.1 r.1).mpr (Or.inr hf)⟩

theorem SummaryEq.binary {xs ys us vs : List FlowPair}
    (h : SummaryEq xs ys) (k : SummaryEq us vs)
    (next : ColourPath → ColourPath → List ColourPath) :
    SummaryEq (binary xs us next) (binary ys vs next) := by
  constructor
  · intro q
    simp only [binary_key, h.1, k.1]
  · intro q f
    simp only [binary_bit, h.1, k.1, h.2, k.2]

#print axioms SummaryEq.unary
#print axioms SummaryEq.binary

end FaithfulKeyed
