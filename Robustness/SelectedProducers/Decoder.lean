import Robustness.BitLocality
import Flow.Definitions
import Init.Data.BitVec.Lemmas

/-! Finite Boolean coordinates for canonical dependencies and fixed-length paths.
The shape fixes route owners, formula formulaAt, path lengths and word widths.
Dependency *vector* width is fixed; decoded list cardinality can vary.
No equality with the old fixed Formula-list-entry product is asserted. -/
open DLDS FlowSemantics
set_option maxHeartbeats 2000000
namespace Robustness.SelectedProducers

def encodeWord (width value : Nat) : Fin width → Bool :=
  fun bit => (BitVec.ofNat width value).getLsbD bit.val

def decodeWord {width : Nat} (bits : Fin width → Bool) : Nat :=
  (BitVec.ofBoolListLE (List.ofFn bits)).toNat

theorem word_roundtrip (width value : Nat) (fits : value < 2 ^ width) :
    decodeWord (encodeWord width value) = value := by
  let vector : BitVec width :=
    (BitVec.ofBoolListLE (List.ofFn (encodeWord width value))).cast (by simp)
  have eq : vector = BitVec.ofNat width value := by
    apply BitVec.eq_of_getLsbD_eq
    intro bit bound
    dsimp only [vector]
    rw [BitVec.getLsbD_cast, BitVec.getLsbD_ofBoolListLE,
      List.getD_eq_getElem?_getD, List.getElem?_ofFn]
    simp only [bound, dif_pos, Option.getD_some, encodeWord]
  have natEq := congrArg BitVec.toNat eq
  simpa [vector, decodeWord, BitVec.toNat_ofNat, Nat.mod_eq_of_lt fits] using natEq

structure RouteShape (routes formulas : Nat) where
  formulaAt : Fin formulas → Formula
  injective : Function.Injective formulaAt
  pathLength : Fin routes → Nat
  colourWidth : Nat

abbrev RouteCoord {routes formulas : Nat} (shape : RouteShape routes formulas) :=
  (Fin routes × Fin formulas) ⊕
    (Σ slot : Fin routes, Fin (shape.pathLength slot) × Fin shape.colourWidth)

def decodeDependency {routes formulas : Nat} (shape : RouteShape routes formulas)
    (bits : Fin formulas → Bool) : Dep :=
  normalizeDep (((List.finRange formulas).filter bits).map shape.formulaAt)

theorem decoded_membership {routes formulas : Nat} (shape : RouteShape routes formulas)
    (bits : Fin formulas → Bool) (code : Fin formulas) :
    shape.formulaAt code ∈ decodeDependency shape bits ↔ bits code = true := by
  simp only [decodeDependency, mem_normalizeDep, List.mem_map, List.mem_filter,
    List.mem_finRange, true_and]
  constructor
  · rintro ⟨other, truth, same⟩
    have eq := shape.injective same
    simpa [eq] using truth
  · intro truth
    exact ⟨code, truth, rfl⟩

theorem dependency_roundtrip {routes formulas : Nat} (shape : RouteShape routes formulas)
    (dependency : Dep) (canonical : CanonicalDep dependency)
    (represented : ∀ formula ∈ dependency, ∃ code, shape.formulaAt code = formula) :
    decodeDependency shape (fun code => decide (shape.formulaAt code ∈ dependency)) =
      dependency := by
  apply canonicalDep_eq_of_depSetEq (normalizeDep_canonical _) canonical
  apply (depSetEq_eq_true_iff _ _).mpr
  intro formula
  simp only [mem_normalizeDep, List.mem_map, List.mem_filter,
    List.mem_finRange, true_and, decide_eq_true_eq]
  constructor
  · rintro ⟨code, member, rfl⟩
    exact member
  · intro member
    obtain ⟨code, same⟩ := represented formula member
    exact ⟨code, by simpa [same] using member, same⟩

def decodeRoutes {routes formulas : Nat} (shape : RouteShape routes formulas)
    (bits : RouteCoord shape → Bool) (slot : Fin routes) : FlowPair :=
  (decodeDependency shape (fun code => bits (.inl (slot, code))),
    List.ofFn fun position : Fin (shape.pathLength slot) =>
      decodeWord (fun bit => bits (.inr ⟨slot, position, bit⟩)))

def encodeRoutes {routes formulas : Nat} (shape : RouteShape routes formulas)
    (data : Fin routes → FlowPair) : RouteCoord shape → Bool
  | .inl (slot, code) => decide (shape.formulaAt code ∈ (data slot).1)
  | .inr ⟨slot, position, bit⟩ =>
      encodeWord shape.colourWidth ((data slot).2.getD position.val 0) bit

def ValidRoutes {routes formulas : Nat} (shape : RouteShape routes formulas)
    (data : Fin routes → FlowPair) : Prop :=
  ∀ slot, CanonicalDep (data slot).1 ∧
    (∀ formula ∈ (data slot).1, ∃ code, shape.formulaAt code = formula) ∧
    (data slot).2.length = shape.pathLength slot ∧
    ∀ colour ∈ (data slot).2, colour < 2 ^ shape.colourWidth

theorem routes_roundtrip {routes formulas : Nat} (shape : RouteShape routes formulas)
    (data : Fin routes → FlowPair) (valid : ValidRoutes shape data) :
    decodeRoutes shape (encodeRoutes shape data) = data := by
  funext slot
  apply Prod.ext
  · exact dependency_roundtrip shape _ (valid slot).1 (valid slot).2.1
  · dsimp only [decodeRoutes, encodeRoutes]
    apply List.ext_getElem
    · simpa using (valid slot).2.2.1.symm
    · intro position leftBound rightBound
      simp only [List.getElem_ofFn]
      have bound : position < (data slot).2.length := by
        simpa using rightBound
      have colourBound := (valid slot).2.2.2 _ (List.getElem_mem bound)
      have same : (data slot).2.getD position 0 = (data slot).2[position] :=
        (List.getElem_eq_getD 0).symm
      simpa only [same] using
        word_roundtrip shape.colourWidth ((data slot).2[position]) colourBound

abbrev FullCoord {routes formulas : Nat} (shape : RouteShape routes formulas)
    (headerWords headerWidth : Nat) :=
  (Fin headerWords × Fin headerWidth) ⊕ RouteCoord shape

def encodeFull {routes formulas : Nat} (shape : RouteShape routes formulas)
    (headerWords headerWidth : Nat) (header : Fin headerWords → Nat)
    (data : Fin routes → FlowPair) : FullCoord shape headerWords headerWidth → Bool
  | .inl (field, bit) => encodeWord headerWidth (header field) bit
  | .inr coordinate => encodeRoutes shape data coordinate

def decodeHeader {routes formulas : Nat} (shape : RouteShape routes formulas)
    (headerWords headerWidth : Nat) (bits : FullCoord shape headerWords headerWidth → Bool) :=
  fun field : Fin headerWords => decodeWord (fun bit => bits (.inl (field, bit)))

theorem header_roundtrip {routes formulas : Nat} (shape : RouteShape routes formulas)
    (headerWords headerWidth : Nat) (header : Fin headerWords → Nat)
    (data : Fin routes → FlowPair) (fits : ∀ field, header field < 2 ^ headerWidth) :
    decodeHeader shape headerWords headerWidth
      (encodeFull shape headerWords headerWidth header data) = header := by
  funext field
  exact word_roundtrip _ _ (fits field)

end Robustness.SelectedProducers
