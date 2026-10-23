import Semantic.Circuit.SharedAssembly
import Semantic.Circuit.ProductionPlan

namespace Semantic.Circuit.Shared

set_option maxHeartbeats 20000000
set_option maxRecDepth 100000

def reference (scope wire : Nat) : Expr scope :=
  if bound : wire < scope then .bit ⟨wire, bound⟩ else .constant false

theorem reference_value (scope wire : Nat) (state : Bits) (bound : wire < scope) :
    (reference scope wire).value state = state wire := by simp [reference, bound, Expr.value]

structure Registers (width : Nat) where
  live : Nat
  payload : Fin width → Nat

def Registers.Bounded {width : Nat} (entry : Registers width) (scope : Nat) : Prop :=
  entry.live < scope ∧ ∀ i, entry.payload i < scope

def Registers.value {width : Nat} (entry : Registers width) (state : Bits) : Bool × (Fin width → Bool) :=
  (state entry.live, fun i => state (entry.payload i))

def Registers.expressions {width : Nat} (entry : Registers width) (scope : Nat) : Candidate scope width :=
  ⟨reference scope entry.live, fun i => reference scope (entry.payload i)⟩

theorem Registers.expressions_value {width : Nat} (entry : Registers width) (scope : Nat)
    (state : Bits) (bounded : entry.Bounded scope) :
    (entry.expressions scope).value state = entry.value state := by
  apply Prod.ext
  · exact reference_value _ _ _ bounded.1
  · funext i
    exact reference_value _ _ _ (bounded.2 i)

theorem Registers.weaken {width a b : Nat} (entry : Registers width) (h : entry.Bounded a)
    (le : a ≤ b) : entry.Bounded b :=
  ⟨lt_of_lt_of_le h.1 le, fun i => lt_of_lt_of_le (h.2 i) le⟩

theorem Registers.stable {width : Nat} (entry : Registers width) {before after : Store}
    (extension : Extends before after) (base : Nat) (state : Bits)
    (bounded : entry.Bounded (base + before.size)) :
    entry.value (memory base after state) = entry.value (memory base before state) := by
  apply Prod.ext
  · exact extension.stable base state _ bounded.1
  · funext i
    exact extension.stable base state _ (bounded.2 i)

def frozen {width : Nat} (fields : Fin width → Nat) : Array Nat := Array.ofFn fields

@[simp] theorem frozen_get {width : Nat} (fields : Fin width → Nat) (i : Fin width) :
    (frozen fields)[i.val]'(by simpa only [frozen, Array.size_ofFn] using i.isLt) = fields i := by
  simp [frozen]

def vector {scope : Nat} (base : Nat) : (width : Nat) → (Fin width → Expr scope) →
    Store → (Fin width → Nat) × Store
  | 0, _, code => (Fin.elim0, code)
  | width + 1, fields, code =>
      let head := expression base (fields 0) code
      let tail := vector base width (fun i => fields i.succ) head.2
      let data := frozen (Fin.cases head.1 tail.1)
      ((fun i => data[i.val]'(by simpa only [data, frozen, Array.size_ofFn] using i.isLt)), tail.2)

theorem vector_correct {scope : Nat} (base width : Nat) (fields : Fin width → Expr scope)
    (code : Store) (space : scope ≤ base + code.size) (state : Bits) :
    let result := vector base width fields code
    Extends code result.2 ∧ (∀ i, result.1 i < base + result.2.size) ∧
      ∀ i, memory base result.2 state (result.1 i) = (fields i).value (memory base code state) := by
  induction width generalizing code with
  | zero => exact ⟨Extends.refl _, fun i => Fin.elim0 i, fun i => Fin.elim0 i⟩
  | succ width ih =>
    obtain ⟨headExtension, headBound, headValue⟩ := expression_correct base (fields 0) code space state
    have nextSpace : scope ≤ base + (expression base (fields 0) code).2.size := by
      have := headExtension.size_le
      omega
    obtain ⟨tailExtension, tailBound, tailValue⟩ := ih (fun i => fields i.succ)
      (expression base (fields 0) code).2 nextSpace
    refine ⟨headExtension.trans tailExtension, ?_, ?_⟩
    · intro i
      refine Fin.cases ?_ (fun j => ?_) i
      · have := tailExtension.size_le
        simpa only [vector, frozen_get, Fin.cases_zero] using lt_of_lt_of_le headBound (Nat.add_le_add_left this base)
      · simpa only [vector, frozen_get, Fin.cases_succ] using tailBound j
    · intro i
      refine Fin.cases ?_ (fun j => ?_) i
      · simpa only [vector, frozen_get, Fin.cases_zero] using
          (tailExtension.stable base state _ headBound).trans headValue
      · simpa only [vector, frozen_get, Fin.cases_succ] using
          (tailValue j).trans (headExtension.expression base state (fields j.succ) space)

def candidate {scope width : Nat} (base : Nat) (entry : Candidate scope width) (code : Store) :
    Registers width × Store :=
  let live := expression base entry.live code
  let fields := vector base width entry.payload live.2
  (⟨live.1, fields.1⟩, fields.2)

theorem candidate_correct {scope width : Nat} (base : Nat) (entry : Candidate scope width)
    (code : Store) (space : scope ≤ base + code.size) (state : Bits) :
    let result := candidate base entry code
    Extends code result.2 ∧ result.1.Bounded (base + result.2.size) ∧
      result.1.value (memory base result.2 state) = entry.value (memory base code state) := by
  obtain ⟨liveExtension, liveBound, liveValue⟩ := expression_correct base entry.live code space state
  have nextSpace : scope ≤ base + (expression base entry.live code).2.size := by
    have := liveExtension.size_le
    omega
  obtain ⟨fieldExtension, fieldBound, fieldValue⟩ :=
    vector_correct base width entry.payload (expression base entry.live code).2 nextSpace state
  refine ⟨liveExtension.trans fieldExtension, ?_, ?_⟩
  · exact ⟨lt_of_lt_of_le liveBound (Nat.add_le_add_left fieldExtension.size_le base), fieldBound⟩
  · apply Prod.ext
    · exact (fieldExtension.stable base state _ liveBound).trans liveValue
    · funext i
      exact (fieldValue i).trans (liveExtension.expression base state (entry.payload i) space)

def candidates {scope width : Nat} (base : Nat) :
    List (Candidate scope width) → Store → List (Registers width) × Store
  | [], code => ([], code)
  | head :: tail, code =>
      let first := candidate base head code
      let rest := candidates base tail first.2
      (first.1 :: rest.1, rest.2)

theorem candidates_correct {scope width : Nat} (base : Nat) (entries : List (Candidate scope width))
    (code : Store) (space : scope ≤ base + code.size) (state : Bits) :
    let result := candidates base entries code
    Extends code result.2 ∧ (∀ e ∈ result.1, e.Bounded (base + result.2.size)) ∧
      result.1.map (fun e => e.value (memory base result.2 state)) =
        entries.map (fun e => e.value (memory base code state)) := by
  induction entries generalizing code with
  | nil => exact ⟨Extends.refl _, fun _ h => False.elim (List.not_mem_nil h), rfl⟩
  | cons head tail ih =>
    obtain ⟨firstExtension, firstBound, firstValue⟩ := candidate_correct base head code space state
    have nextSpace : scope ≤ base + (candidate base head code).2.size := by
      have := firstExtension.size_le
      omega
    obtain ⟨restExtension, restBound, restValue⟩ := ih (candidate base head code).2 nextSpace
    refine ⟨firstExtension.trans restExtension, ?_, ?_⟩
    · intro e member
      rcases List.mem_cons.mp member with rfl | contained
      · exact Registers.weaken _ firstBound (Nat.add_le_add_left restExtension.size_le base)
      · exact restBound e contained
    · simp only [candidates, List.map_cons]
      rw [(candidate base head code).1.stable restExtension base state firstBound, firstValue, restValue]
      congr 1
      apply List.map_congr_left
      intro e _
      apply Prod.ext
      · exact firstExtension.expression base state e.live space
      · funext i
        exact firstExtension.expression base state (e.payload i) space

theorem candidates_length {scope width : Nat} (base : Nat) (entries : List (Candidate scope width)) (code : Store) :
    (candidates base entries code).1.length = entries.length := by
  induction entries generalizing code <;> simp [candidates, *]

theorem encodes_from_bits {n F I : Nat} (basis : Fin F → Formula) (words : List FlowSpec.ColourPath)
    (state : Bits) (entries : List (Candidate n (1 + (F + I)))) (routes : List FlowSpec.FlowPair)
    (capacity : ([] :: words).length ≤ 2 ^ I) (fitting : ∀ r ∈ routes, RouteFits basis words r)
    (bits : selected (entries.map (fun e => e.value state)) = routes.map (routeBits (I := I) basis words)) :
    Encodes basis words state entries routes := by
  apply Encodes.of_bits entries routes capacity _ fitting bits
  intro e contained active
  have member : (e.value state).2 ∈ selected (entries.map (fun e => e.value state)) := by
    apply List.mem_filterMap.mpr
    refine ⟨e.value state, List.mem_map.mpr ⟨e, contained, rfl⟩, ?_⟩
    simp [Candidate.value, active]
  rw [bits] at member
  obtain ⟨r, containedR, same⟩ := List.mem_map.mp member
  exact ⟨r, fitting r containedR, same.symm⟩

end Semantic.Circuit.Shared
