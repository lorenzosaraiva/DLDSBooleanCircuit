import Semantic.FaithfulSummaryAlgebra
import Semantic.LambdaKeyedEncoding

open Semantic FlowSpec
namespace FaithfulKeyed

def FamilySupported (d : Graph) (routes : List FlowPair) : Prop :=
  ∀ route ∈ routes, ∀ formula ∈ route.1, formula ∈ RefinedEval.formulas d

theorem merge_key_iff (d : Graph) (routes : List FlowPair) (p : ColourPath) :
    (∃ row ∈ KeyedEval.merge d routes, row.1 = p) ↔
      ∃ route ∈ routes, route.2 = p := by
  have h := KeyedEval.keys_merge d routes
  have hm := congrArg (fun keys => p ∈ keys) h
  simpa only [List.mem_map, List.mem_eraseDups] using (Iff.of_eq hm)




theorem expand_merge_summary (d : Graph) (routes : List FlowPair)
    (supported : FamilySupported d routes) :
    SummaryEq (KeyedEval.expand d (KeyedEval.merge d routes)) routes := by
  constructor
  · intro p
    constructor
    · rintro ⟨r, hr, hp⟩
      obtain ⟨row, hrow, rfl⟩ := List.mem_map.mp hr
      exact (merge_key_iff d routes p).mp ⟨row, hrow, hp⟩
    · intro h
      obtain ⟨row, hrow, hp⟩ := (merge_key_iff d routes p).mpr h
      exact ⟨_, List.mem_map.mpr ⟨row, hrow, rfl⟩, hp⟩
  · intro p f
    constructor
    · rintro ⟨r, hr, hp, hf⟩
      obtain ⟨row, hrow, rfl⟩ := List.mem_map.mp hr
      obtain ⟨_, source, hsource, hkey, hbit⟩ :=
        (LambdaKeyed.decode_merge_mem d routes row hrow f).mp hf
      exact ⟨source, hsource, hkey.trans hp, hbit⟩
    · rintro ⟨r, hr, hp, hf⟩
      obtain ⟨row, hrow, hkey⟩ := (merge_key_iff d routes p).mpr ⟨r, hr, hp⟩
      refine ⟨_, List.mem_map.mpr ⟨row, hrow, rfl⟩, hkey, ?_⟩
      exact (LambdaKeyed.decode_merge_mem d routes row hrow f).mpr
        ⟨supported r hr f hf, r, hr, hp.trans hkey.symm, hf⟩

def ClosedRoutes (routes : List FlowPair) : Prop :=
  (∃ route, route ∈ routes) ∧ ∀ route ∈ routes, route.1 = [] ∧ route.2 = []

theorem SummaryEq.closed_iff {left right : List FlowPair}
    (h : SummaryEq left right) : ClosedRoutes left ↔ ClosedRoutes right := by
  have transfer : ∀ xs ys : List FlowPair, SummaryEq xs ys →
      ClosedRoutes xs → ClosedRoutes ys := by
    intro xs ys hs hc
    constructor
    · obtain ⟨r, hr⟩ := hc.1
      obtain ⟨s, hs, _⟩ := (hs.1 r.2).mp ⟨r, hr, rfl⟩
      exact ⟨s, hs⟩
    · intro r hr
      constructor
      · apply List.eq_nil_iff_forall_not_mem.mpr
        intro f hf
        obtain ⟨s, hs', _, bit⟩ := (hs.2 r.2 f).mpr ⟨r, hr, rfl, hf⟩
        rw [(hc.2 s hs').1] at bit
        exact List.not_mem_nil bit
      · obtain ⟨s, hs', key⟩ := (hs.1 r.2).mpr ⟨r, hr, rfl⟩
        exact key.symm.trans (hc.2 s hs').2
  exact ⟨transfer left right h, transfer right left h.symm⟩

private theorem decoded_zip_eq_nil (formulas : List Formula) (bits : List Bool)
    (lengths : formulas.length = bits.length) :
    ((formulas.zip bits).filterMap fun (f, b) => if b then some f else none) = [] ↔
      bits.all (!·) = true := by
  induction formulas generalizing bits with
  | nil =>
      have hb : bits = [] := List.length_eq_zero_iff.mp lengths.symm
      subst bits
      simp
  | cons f fs ih =>
      cases bits with
      | nil => simp at lengths
      | cons b bs =>
          have rest : fs.length = bs.length := Nat.succ.inj lengths
          cases b <;> simp [ih bs rest]

theorem decode_eq_nil_iff (d : Graph) (vector : KeyedEval.Vector d) :
    KeyedEval.decode d vector = [] ↔ vector.toList.all (!·) = true := by
  exact decoded_zip_eq_nil (RefinedEval.formulas d) vector.toList vector.2.symm

theorem closedRoutes_iff_bool (routes : List FlowPair) :
    ClosedRoutes routes ↔
      (!routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty) = true := by
  simp only [ClosedRoutes, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true,
    List.isEmpty_iff]
  cases routes with
  | nil => simp
  | cons r rs => simp

theorem closed_expand_iff (d : Graph) (rows : KeyedEval.Carrier d) :
    ClosedRoutes (KeyedEval.expand d rows) ↔
      (!rows.isEmpty && rows.all fun row => row.1.isEmpty && row.2.toList.all (!·)) = true := by
  rw [closedRoutes_iff_bool]
  simp only [KeyedEval.expand, List.isEmpty_map, List.all_map]
  have hall : rows.all (fun row =>
      (KeyedEval.decode d row.2).isEmpty && row.1.isEmpty) =
      rows.all (fun row => row.1.isEmpty && row.2.toList.all (!·)) := by
    apply List.all_congr rfl
    intro row
    have hd := decode_eq_nil_iff d row.2
    have hb : (KeyedEval.decode d row.2).isEmpty = row.2.toList.all (!·) := by
      apply Bool.eq_iff_iff.mpr
      simpa only [List.isEmpty_iff] using hd
    simp [hb, Bool.and_comm]
  change (!rows.isEmpty && rows.all (fun row =>
      (KeyedEval.decode d row.2).isEmpty && row.1.isEmpty)) = true ↔ _
  rw [hall]

#print axioms expand_merge_summary
#print axioms SummaryEq.closed_iff
#print axioms decode_eq_nil_iff
#print axioms closedRoutes_iff_bool
#print axioms closed_expand_iff
end FaithfulKeyed
