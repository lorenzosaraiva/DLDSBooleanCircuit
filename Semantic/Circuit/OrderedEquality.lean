import Semantic.Circuit.RouteOperations

namespace Semantic.Circuit

set_option maxHeartbeats 6000000

def Expr.choose {n : Nat} (condition yes no : Expr n) : Expr n :=
  (condition.conj yes).disj (condition.neg.conj no)

theorem Expr.value_choose {n : Nat} (condition yes no : Expr n) (state : Bits) :
    (condition.choose yes no).value state =
      if condition.value state then yes.value state else no.value state := by
  simp only [Expr.choose, Expr.value_disj, Expr.value]
  by_cases active : condition.value state = true <;> simp_all

structure Candidate (n width : Nat) where
  live : Expr n
  payload : Fin width → Expr n

def Candidate.value {n width : Nat} (candidate : Candidate n width) (state : Bits) :
    Bool × (Fin width → Bool) :=
  (candidate.live.value state, fun i => (candidate.payload i).value state)

def selected {α : Type} (values : List (Bool × α)) : List α :=
  values.filterMap fun entry => if entry.1 then some entry.2 else none

@[simp] theorem selected_nil {α : Type} : selected ([] : List (Bool × α)) = [] := rfl

theorem selected_cons {α : Type} (head : Bool × α) (rest : List (Bool × α)) :
    selected (head :: rest) = if head.1 then head.2 :: selected rest else selected rest := by
  by_cases active : head.1 = true <;> simp [selected, active]


def suppress {n width : Nat} (pivot entry : Candidate n width) : Candidate n width :=
  ⟨entry.live.conj ((pivot.live.conj (fieldEqual pivot.payload entry.payload)).neg), entry.payload⟩


def firstOccurrences {n width : Nat} : List (Candidate n width) → List (Candidate n width)
  | [] => []
  | head :: rest => head :: firstOccurrences (rest.map (suppress head))
termination_by values => values.length

theorem firstOccurrences_length {n width : Nat} (values : List (Candidate n width)) :
    (firstOccurrences values).length = values.length := by
  induction values using firstOccurrences.induct with
  | case1 => simp [firstOccurrences]
  | case2 head rest ih => simpa [firstOccurrences] using congrArg Nat.succ ih

theorem selected_suppress {n width : Nat} (pivot : Candidate n width)
    (values : List (Candidate n width)) (state : Bits) :
    selected ((values.map (suppress pivot)).map (fun e => e.value state)) =
      (selected (values.map (fun e => e.value state))).filter
        (fun payload => !(pivot.live.value state && decide ((pivot.value state).2 = payload))) := by
  induction values with
  | nil => rfl
  | cons head rest ih =>
    simp only [List.map_cons, selected, List.filterMap_cons, suppress, Candidate.value,
      Expr.value, fieldEqual_value, List.filter_cons] at *
    by_cases headActive : head.live.value state = true <;>
      by_cases pivotActive : pivot.live.value state = true <;>
      by_cases equal : (fun i => (pivot.payload i).value state) = (fun i => (head.payload i).value state) <;>
      simp_all

theorem firstOccurrences_correct {n width : Nat} (values : List (Candidate n width)) (state : Bits) :
    selected ((firstOccurrences values).map (fun e => e.value state)) =
      (selected (values.map (fun e => e.value state))).eraseDups := by
  induction values using firstOccurrences.induct with
  | case1 => simp [firstOccurrences, selected]
  | case2 head rest ih =>
    have step : selected ((firstOccurrences (rest.map (suppress head))).map (fun e => e.value state)) =
        (selected ((rest.map (suppress head)).map (fun e => e.value state))).eraseDups := by simpa using ih
    have suppressed := selected_suppress head rest state
    dsimp only [Candidate.value] at step suppressed
    simp only [firstOccurrences, List.map_cons, selected_cons, Candidate.value]
    by_cases active : head.live.value state = true
    · simp only [active, ite_true]
      rw [step, suppressed, List.eraseDups_cons]
      simp [Candidate.value, active, Bool.beq_eq_decide_eq, eq_comm]
    · simp only [Bool.eq_false_iff.mpr active, Bool.false_eq_true, ite_false]
      rw [step, suppressed]
      simp [Candidate.value, active]




def orderedExpression {n width : Nat} :
    List (Candidate n width) → List (Candidate n width) → Expr n
  | [], right => Expr.all (right.map fun entry => entry.live.neg)
  | left, [] => Expr.all (left.map fun entry => entry.live.neg)
  | head :: rest, other :: tail =>
      head.live.choose
        (other.live.choose
          ((fieldEqual head.payload other.payload).conj (orderedExpression rest tail))
          (orderedExpression (head :: rest) tail))
        (orderedExpression rest (other :: tail))
termination_by left right => left.length + right.length

theorem selected_empty_iff {α : Type} (values : List (Bool × α)) :
    selected values = [] ↔ ∀ entry ∈ values, entry.1 = false := by
  simp only [selected, List.filterMap_eq_nil_iff]
  constructor
  · intro empty entry member
    have h := empty entry member
    exact (by simpa using h)
  · intro empty entry member
    simp [empty entry member]

theorem orderedExpression_correct {n width : Nat}
    (left right : List (Candidate n width)) (state : Bits) :
    (orderedExpression left right).value state =
      decide (selected (left.map (fun e => e.value state)) =
        selected (right.map (fun e => e.value state))) := by
  induction left, right using orderedExpression.induct with
  | case1 right =>
    apply Bool.eq_iff_iff.mpr
    simp only [orderedExpression, Expr.value_all, List.all_map, Function.comp_def,
      List.all_eq_true, Expr.value, Bool.not_eq_true', decide_eq_true_eq,
      List.map_nil, selected_nil]
    rw [eq_comm, selected_empty_iff]
    simp [Candidate.value, Expr.value]
  | case2 left nonempty =>
    apply Bool.eq_iff_iff.mpr
    simp only [orderedExpression, Expr.value_all, List.all_map, List.all_eq_true, Expr.value,
      Bool.not_eq_true', decide_eq_true_eq, List.map_nil]
    change _ ↔ selected _ = []
    rw [selected_empty_iff]
    simp [Candidate.value, Expr.value]
  | case3 head rest other tail both skipRight skipLeft =>
    simp only [orderedExpression, Expr.value_choose, Expr.value, fieldEqual_value,
      both, skipRight, skipLeft]
    by_cases headActive : head.live.value state = true <;>
      by_cases otherActive : other.live.value state = true <;>
      simp [selected, Candidate.value, headActive, otherActive]

def deduplicatedEquality {n width : Nat}
    (candidates stored : List (Candidate n width)) : Expr n :=
  orderedExpression (firstOccurrences candidates) stored

theorem deduplicatedEquality_correct {n width : Nat}
    (candidates stored : List (Candidate n width)) (state : Bits) :
    (deduplicatedEquality candidates stored).value state =
      decide ((selected (candidates.map (fun e => e.value state))).eraseDups =
        selected (stored.map (fun e => e.value state))) := by
  rw [deduplicatedEquality, orderedExpression_correct, firstOccurrences_correct]

theorem deduplicatedEquality_circuit {n width : Nat}
    (candidates stored : List (Candidate n width)) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (deduplicatedEquality candidates stored).oracle state =
      Function.update state (n + (deduplicatedEquality candidates stored).size)
        (decide ((selected (candidates.map (fun e => e.value state))).eraseDups =
          selected (stored.map (fun e => e.value state)))) := by
  rw [Expr.oracle_correct _ state clean, deduplicatedEquality_correct]

theorem selected_append {α : Type} (left right : List (Bool × α)) :
    selected (left ++ right) = selected left ++ selected right := by
  simp [selected]


theorem selected_at_rank {α : Type} (before after : List (Bool × α)) (payload : α) :
    (selected (before ++ (true, payload) :: after))[(selected before).length]? = some payload := by
  rw [selected_append, selected_cons]
  simp

def rankCharacterization {α : Type} (flags : List (Bool × α)) (stored : List α) : Prop :=
  (selected flags).length = stored.length ∧
    ∀ before payload after, flags = before ++ (true, payload) :: after →
      stored[(selected before).length]? = some payload

theorem selected_eq_iff_rank {α : Type} (flags : List (Bool × α)) (stored : List α) :
    selected flags = stored ↔ rankCharacterization flags stored := by
  constructor
  · rintro rfl
    refine ⟨rfl, ?_⟩
    intro before payload after decomposition
    rw [decomposition]
    exact selected_at_rank before after payload
  · rintro ⟨lengths, positions⟩
    induction flags generalizing stored with
    | nil => simpa [selected] using lengths.symm
    | cons entry rest ih =>
      obtain ⟨live, payload⟩ := entry
      by_cases active : live = true
      · subst live
        have head := positions [] payload rest rfl
        cases stored with
        | nil => simp [selected_cons] at lengths
        | cons other tail =>
          have equal : other = payload := by simpa [selected] using head
          subst other
          rw [selected_cons]
          simp only [ite_true, Bool.true_eq, List.cons.injEq, true_and]
          apply ih
          · simpa [selected_cons] using lengths
          · intro before value after decomposition
            have checked := positions ((true, payload) :: before) value after
              (by simp [decomposition])
            simpa [selected_cons] using checked
      · have inactive : live = false := Bool.eq_false_iff.mpr active
        subst live
        rw [selected_cons]
        simp only [Bool.false_eq_true, ite_false]
        apply ih
        · simpa [selected_cons] using lengths
        · intro before value after decomposition
          have checked := positions ((false, payload) :: before) value after
            (by simp [decomposition])
          simpa [selected_cons] using checked

theorem first_occurrence_rank_characterization {n width : Nat}
    (candidates stored : List (Candidate n width)) (state : Bits) :
    (deduplicatedEquality candidates stored).value state = true ↔
      rankCharacterization ((firstOccurrences candidates).map (fun e => e.value state))
        (selected (stored.map (fun e => e.value state))) := by
  rw [deduplicatedEquality, orderedExpression_correct, decide_eq_true_eq, selected_eq_iff_rank]

end Semantic.Circuit
