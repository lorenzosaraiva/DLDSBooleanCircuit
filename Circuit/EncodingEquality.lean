import Circuit.OrderedEquality
import Circuit.Root

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 6000000

theorem Slot.bit_injective {F I : Nat} : Function.Injective (Slot.bit (F := F) (I := I)) := by
  intro left right equal
  have occupied := congrFun equal (occupiedOffset F I)
  have masks : left.dependency = right.dependency := by
    funext i
    have h := congrFun equal (dependencyOffset I i)
    simpa [Slot.bit, dependencyOffset, Fin.addCases_left, Fin.addCases_right] using h
  have residuals : left.residual = right.residual := by
    apply BitVec.eq_of_getLsbD_eq
    intro i bound
    have h := congrFun equal (residualOffset F ⟨i, bound⟩)
    simpa only [Slot.bit, residualOffset, Fin.addCases_right] using h
  have occupancy : left.occupied = right.occupied := by
    simpa [Slot.bit, occupiedOffset, Fin.addCases_left] using occupied
  cases left
  cases right
  simp_all

def routeBits {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (route : FlowPair) : Fin (1 + (F + I)) → Bool := (encodeRoute (I := I) basis words route).bit

theorem routeBits_injective_on {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (left right : FlowPair) (leftFits : RouteFits basis words left) (rightFits : RouteFits basis words right)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (equal : routeBits (I := I) basis words left = routeBits (I := I) basis words right) : left = right := by
  have slots := Slot.bit_injective equal
  have decoded := congrArg (decodeSlot basis words) slots
  rw [decodeRoute_encode basis words left leftFits capacity,
    decodeRoute_encode basis words right rightFits capacity] at decoded
  exact Option.some.inj decoded

theorem eraseDups_map_injective_on {α β : Type}
    [DecidableEq α] [BEq α] [LawfulBEq α] [DecidableEq β] [BEq β] [LawfulBEq β]
    (f : α → β) (values : List α)
    (injective : ∀ a ∈ values, ∀ b ∈ values, f a = f b → a = b) :
    (values.map f).eraseDups = values.eraseDups.map f := by
    cases values with
    | nil => rfl
    | cons head rest =>
      have filters : (rest.map f).filter (fun value => !(value == f head)) =
          (rest.filter (fun value => !(value == head))).map f := by
        rw [List.filter_map]
        congr 1
        apply List.filter_congr
        intro value member
        have equal : f value = f head ↔ value = head :=
          ⟨injective value (by simp [member]) head (by simp), congrArg f⟩
        simp only [Function.comp_def, Bool.beq_eq_decide_eq, equal]
      rw [List.map_cons, List.eraseDups_cons, List.eraseDups_cons, List.map_cons, filters]
      congr 1
      apply eraseDups_map_injective_on f
      intro a ha b hb
      exact injective a (by simp only [List.mem_filter] at ha; simp [ha.1])
        b (by simp only [List.mem_filter] at hb; simp [hb.1])
termination_by values.length
decreasing_by
  have := List.length_filter_le (fun value => !(value == head)) rest
  simp_wf
  simp_all only [List.length_cons]
  omega

theorem routeBits_map_equal_iff {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (left right : List FlowPair) (leftFits : ∀ r ∈ left, RouteFits basis words r)
    (rightFits : ∀ r ∈ right, RouteFits basis words r) (capacity : ([] :: words).length ≤ 2 ^ I) :
    left.map (routeBits (I := I) basis words) = right.map (routeBits (I := I) basis words) ↔ left = right := by
  constructor
  · intro equal
    induction left generalizing right with
    | nil => cases right <;> simp_all
    | cons head rest ih =>
      cases right with
      | nil => simp at equal
      | cons other tail =>
        have pair := List.cons.inj equal
        have heads := routeBits_injective_on basis words head other
          (leftFits head (by simp)) (rightFits other (by simp)) capacity pair.1
        subst other
        congr 1
        exact ih tail (fun r hr => leftFits r (by simp [hr]))
          (fun r hr => rightFits r (by simp [hr])) pair.2
  · rintro rfl
    rfl



theorem route_deduplicated_equality {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (selections stored : List FlowPair) (selectionFits : ∀ r ∈ selections, RouteFits basis words r)
    (storedFits : ∀ r ∈ stored, RouteFits basis words r) (capacity : ([] :: words).length ≤ 2 ^ I) :
    ((selections.map (routeBits (I := I) basis words)).eraseDups =
      stored.map (routeBits (I := I) basis words)) ↔ selections.eraseDups = stored := by
  rw [eraseDups_map_injective_on _ selections (fun a ha b hb =>
    routeBits_injective_on basis words a b (selectionFits a ha) (selectionFits b hb) capacity)]
  exact routeBits_map_equal_iff basis words _ stored
    (fun r hr => selectionFits r (List.mem_eraseDups.mp hr)) storedFits capacity



theorem ordered_route_circuit {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (selections stored : List (Selection n (1 + (F + I)))) (produced claimed : List FlowPair)
    (state : Bits) (clean : ∀ wire, n ≤ wire → state wire = false)
    (selectionFits : ∀ r ∈ produced, RouteFits basis words r)
    (storedFits : ∀ r ∈ claimed, RouteFits basis words r)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (selectionEncoding : selected (selections.map (fun e => e.value state)) =
      produced.map (routeBits (I := I) basis words))
    (storedEncoding : selected (stored.map (fun e => e.value state)) =
      claimed.map (routeBits (I := I) basis words)) :
    eval (deduplicatedEquality selections stored).oracle state =
      Function.update state (n + (deduplicatedEquality selections stored).size)
        (decide (produced.eraseDups = claimed)) := by
  rw [deduplicatedEquality_circuit _ _ state clean, selectionEncoding, storedEncoding]
  simp only [route_deduplicated_equality basis words produced claimed selectionFits storedFits capacity]

end Semantic.Circuit
