import Semantic.Circuit.SharedRegisters

namespace Semantic.Circuit.Shared

set_option maxHeartbeats 20000000
set_option maxRecDepth 100000

def values {width : Nat} (entries : List (Registers width)) (state : Bits) :=
  entries.map (fun e => e.value state)

def Bounded {width : Nat} (entries : List (Registers width)) (scope : Nat) : Prop :=
  ∀ e ∈ entries, e.Bounded scope

theorem Bounded.weaken {width a b : Nat} {entries : List (Registers width)}
    (h : Bounded entries a) (le : a ≤ b) : Bounded entries b :=
  fun e member => Registers.weaken e (h e member) le

theorem values_stable {width : Nat} (entries : List (Registers width)) {before after : Store}
    (extension : Extends before after) (base : Nat) (state : Bits)
    (bounded : Bounded entries (base + before.size)) :
    values entries (memory base after state) = values entries (memory base before state) := by
  apply List.map_congr_left
  intro e member
  exact e.stable extension base state (bounded e member)

theorem expressions_values {width : Nat} (entries : List (Registers width)) (scope : Nat)
    (state : Bits) (bounded : Bounded entries scope) :
    (entries.map (fun e => e.expressions scope)).map (fun e => e.value state) = values entries state := by
  simp only [List.map_map]
  apply List.map_congr_left
  intro e member
  exact e.expressions_value scope state (bounded e member)

def suppressValue {α : Type} [DecidableEq α] (pivot entry : Bool × α) : Bool × α :=
  (entry.1 && !(pivot.1 && decide (pivot.2 = entry.2)), entry.2)

theorem suppress_value {n width : Nat} (pivot entry : Candidate n width) (state : Bits) :
    (suppress pivot entry).value state = suppressValue (pivot.value state) (entry.value state) := by
  simp [suppress, Candidate.value, suppressValue, Expr.value, fieldEqual_value]
  rfl

def flagsValue {α : Type} [DecidableEq α] : Nat → List (Bool × α) → List (Bool × α)
  | 0, entries => entries
  | _ + 1, [] => []
  | fuel + 1, head :: rest => head :: flagsValue fuel (rest.map (suppressValue head))

theorem old_flags_values {n width : Nat} (entries : List (Candidate n width)) (state : Bits)
    (fuel : Nat) (enough : entries.length ≤ fuel) :
    (firstOccurrences entries).map (fun e => e.value state) =
      flagsValue fuel (entries.map (fun e => e.value state)) := by
  induction fuel generalizing entries with
  | zero =>
    have empty : entries = [] := List.length_eq_zero_iff.mp (by omega)
    subst entries
    simp [firstOccurrences, flagsValue]
  | succ fuel ih =>
    cases entries with
    | nil => simp [firstOccurrences, flagsValue]
    | cons head rest =>
      rw [firstOccurrences, List.map_cons, ih (rest.map (suppress head)) (by simpa using enough)]
      simp [flagsValue, List.map_map, Function.comp_def, suppress_value]

def suppressRegisters {width : Nat} (base : Nat) (pivot : Registers width)
    (entries : List (Registers width)) (code : Store) : List (Registers width) × Store :=
  let scope := base + code.size
  candidates base (entries.map fun entry => suppress (pivot.expressions scope) (entry.expressions scope)) code

theorem suppressRegisters_correct {width : Nat} (base : Nat) (pivot : Registers width)
    (entries : List (Registers width)) (code : Store) (state : Bits)
    (pivotBound : pivot.Bounded (base + code.size)) (bounded : Bounded entries (base + code.size)) :
    let result := suppressRegisters base pivot entries code
    Extends code result.2 ∧ Bounded result.1 (base + result.2.size) ∧
      values result.1 (memory base result.2 state) =
        (values entries (memory base code state)).map (suppressValue (pivot.value (memory base code state))) := by
  obtain ⟨extension, bounds, same⟩ := candidates_correct base
    (entries.map fun entry => suppress (pivot.expressions (base + code.size))
      (entry.expressions (base + code.size))) code (by omega) state
  refine ⟨extension, bounds, same.trans ?_⟩
  simp only [values, List.map_map]
  apply List.map_congr_left
  intro e member
  dsimp only [Function.comp_def]
  rw [suppress_value, pivot.expressions_value _ _ pivotBound, e.expressions_value _ _ (bounded e member)]

@[simp] theorem suppressRegisters_length {width : Nat} (base : Nat) (pivot : Registers width)
    (entries : List (Registers width)) (code : Store) :
    (suppressRegisters base pivot entries code).1.length = entries.length := by
  simp [suppressRegisters, candidates_length]

def firstFlags {width : Nat} (base : Nat) : Nat → List (Registers width) → Store → List (Registers width) × Store
  | 0, entries, code => (entries, code)
  | _ + 1, [], code => ([], code)
  | fuel + 1, head :: rest, code =>
      let suppressed := suppressRegisters base head rest code
      let tail := firstFlags base fuel suppressed.1 suppressed.2
      (head :: tail.1, tail.2)

theorem firstFlags_correct {width : Nat} (base fuel : Nat) (entries : List (Registers width))
    (code : Store) (state : Bits) (bounded : Bounded entries (base + code.size)) :
    let result := firstFlags base fuel entries code
    Extends code result.2 ∧ Bounded result.1 (base + result.2.size) ∧
      values result.1 (memory base result.2 state) = flagsValue fuel (values entries (memory base code state)) := by
  induction fuel generalizing entries code with
  | zero => exact ⟨Extends.refl _, bounded, rfl⟩
  | succ fuel ih =>
    cases entries with
    | nil => exact ⟨Extends.refl _, bounded, rfl⟩
    | cons head rest =>
      have headBound := bounded head (by simp)
      obtain ⟨firstExtension, firstBounds, firstValue⟩ := suppressRegisters_correct base head rest code state
        headBound (fun e member => bounded e (by simp [member]))
      obtain ⟨tailExtension, tailBounds, tailValue⟩ := ih _ _ firstBounds
      have extension := firstExtension.trans tailExtension
      refine ⟨extension, ?_, ?_⟩
      · intro e member
        rcases List.mem_cons.mp member with rfl | contained
        · exact Registers.weaken e headBound (Nat.add_le_add_left extension.size_le base)
        · exact tailBounds e contained
      · simp only [firstFlags, values, List.map_cons, flagsValue]
        rw [head.stable extension base state headBound]
        change _ :: values _ _ = _ :: _
        rw [tailValue, firstValue]
        rfl

theorem firstFlags_selected {width : Nat} (base : Nat) (entries : List (Registers width))
    (code : Store) (state : Bits) (bounded : Bounded entries (base + code.size)) :
    selected (values (firstFlags base entries.length entries code).1
      (memory base (firstFlags base entries.length entries code).2 state)) =
      (selected (values entries (memory base code state))).eraseDups := by
  rw [(firstFlags_correct base entries.length entries code state bounded).2.2]
  have old := old_flags_values (entries.map fun e => e.expressions (base + code.size))
    (memory base code state) entries.length (by simp)
  rw [expressions_values entries _ _ bounded] at old
  rw [← old, firstOccurrences_correct, expressions_values entries _ _ bounded]

theorem firstFlags_rank {width : Nat} (base : Nat) (entries : List (Registers width))
    (code : Store) (state : Bits) (bounded : Bounded entries (base + code.size)) (stored : List (Fin width → Bool)) :
    (selected (values entries (memory base code state))).eraseDups = stored ↔
      rankCharacterization (values (firstFlags base entries.length entries code).1
        (memory base (firstFlags base entries.length entries code).2 state)) stored := by
  rw [← firstFlags_selected base entries code state bounded, selected_eq_iff_rank]

end Semantic.Circuit.Shared
