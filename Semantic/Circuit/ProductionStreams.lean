import Semantic.Circuit.EncodingEquality

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 8000000
set_option maxRecDepth 10000



inductive Encodes {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (state : Bits) : List (Candidate n (1 + (F + I))) → List FlowPair → Prop where
  | nil : Encodes basis words state [] []
  | inactive {head tail routes} (live : head.live.value state = false)
      (rest : Encodes basis words state tail routes) :
      Encodes basis words state (head :: tail) routes
  | active {head tail route routes} (live : head.live.value state = true)
      (fits : RouteFits basis words route)
      (payload : (fun i => (head.payload i).value state) = routeBits (I := I) basis words route)
      (rest : Encodes basis words state tail routes) :
      Encodes basis words state (head :: tail) (route :: routes)

theorem Encodes.fitting {n F I : Nat} {basis : Fin F → Formula} {words state}
    {values : List (Candidate n (1 + (F + I)))} {routes}
    (encoded : Encodes basis words state values routes) : ∀ route ∈ routes, RouteFits basis words route := by
  induction encoded with
  | nil => simp
  | inactive _ _ ih => exact ih
  | active _ fits _ _ ih => simpa using And.intro fits ih

theorem Encodes.bits {n F I : Nat} {basis : Fin F → Formula} {words state}
    {values : List (Candidate n (1 + (F + I)))} {routes}
    (encoded : Encodes basis words state values routes) :
    selected (values.map (fun e => e.value state)) = routes.map (routeBits (I := I) basis words) := by
  induction encoded with
  | nil => rfl
  | inactive live _ ih => simp only [List.map_cons, selected_cons, Candidate.value, live, Bool.false_eq_true, ite_false]; exact ih
  | active live _ payload _ ih =>
    dsimp only [Candidate.value] at ih
    simp only [List.map_cons, selected_cons, Candidate.value, live, ite_true]
    rw [payload, ih]

theorem Encodes.append {n F I : Nat} {basis : Fin F → Formula} {words state}
    {left right : List (Candidate n (1 + (F + I)))} {a b}
    (ha : Encodes basis words state left a) (hb : Encodes basis words state right b) :
    Encodes basis words state (left ++ right) (a ++ b) := by
  induction ha with
  | nil => exact hb
  | inactive live _ ih => exact .inactive live ih
  | active live fits payload _ ih => exact .active live fits payload ih

theorem Encodes.inactive_all {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (state : Bits) (values : List (Candidate n (1 + (F + I))))
    (inactive : ∀ e ∈ values, e.live.value state = false) : Encodes basis words state values [] := by
  induction values with
  | nil => exact .nil
  | cons head rest ih => exact .inactive (inactive head (by simp)) (ih (fun e he => inactive e (by simp [he])))

@[simp] theorem flatMap_empty {α β : Type} (values : List α) : values.flatMap (fun _ => ([] : List β)) = [] := by
  induction values <;> simp_all

theorem Encodes.congr_values {n F I : Nat} {basis : Fin F → Formula} {words state}
    {left : List (Candidate n (1 + (F + I)))} {routes}
    (encoded : Encodes basis words state left routes) (right : List (Candidate n (1 + (F + I))))
    (same : left.map (fun e => e.value state) = right.map (fun e => e.value state)) :
    Encodes basis words state right routes := by
  induction encoded generalizing right with
  | nil => cases right <;> simp_all; exact .nil
  | inactive live rest ih =>
    cases right with
    | nil => simp at same
    | cons head tail =>
      have pair := List.cons.inj same
      apply Encodes.inactive
      · exact (congrArg Prod.fst pair.1).symm.trans live
      · exact ih tail pair.2
  | active live fits payload rest ih =>
    cases right with
    | nil => simp at same
    | cons head tail =>
      have pair := List.cons.inj same
      apply Encodes.active _ fits _ (ih tail pair.2)
      · exact (congrArg Prod.fst pair.1).symm.trans live
      · exact (congrArg Prod.snd pair.1).symm.trans payload

theorem Encodes.bind {n F I : Nat} {basis : Fin F → Formula} {words state}
    {values : List (Candidate n (1 + (F + I)))} {routes}
    (encoded : Encodes basis words state values routes)
    (operation : Candidate n (1 + (F + I)) → List (Candidate n (1 + (F + I))))
    (semantic : FlowPair → List FlowPair)
    (inactive : ∀ e, e.live.value state = false → Encodes basis words state (operation e) [])
    (active : ∀ e r, e.live.value state = true → RouteFits basis words r →
      (fun i => (e.payload i).value state) = routeBits (I := I) basis words r →
      Encodes basis words state (operation e) (semantic r)) :
    Encodes basis words state (values.flatMap operation) (routes.flatMap semantic) := by
  induction encoded with
  | nil => exact .nil
  | inactive live _ ih => simpa only [List.flatMap_cons, List.nil_append] using (inactive _ live).append ih
  | active live fits payload _ ih => exact (active _ _ live fits payload).append ih

theorem Encodes.classical_bind {α : Type} {n F I : Nat} {basis : Fin F → Formula} {words state}
    (values : List α) (operation : α → List (Candidate n (1 + (F + I)))) (semantic : α → List FlowPair)
    (correct : ∀ a ∈ values, Encodes basis words state (operation a) (semantic a)) :
    Encodes basis words state (values.flatMap operation) (values.flatMap semantic) := by
  induction values with
  | nil => exact .nil
  | cons head rest ih =>
    exact (correct head (by simp)).append (ih (fun a ha => correct a (by simp [ha])))

def constantCandidate {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (route : FlowPair) : Candidate n (1 + (F + I)) :=
  ⟨.constant true, fun i => .constant (routeBits (I := I) basis words route i)⟩

theorem constants_encode {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (routes : List FlowPair) (state : Bits) (fits : ∀ r ∈ routes, RouteFits basis words r) :
    Encodes basis words state (routes.map (constantCandidate (n := n) (I := I) basis words)) routes := by
  induction routes with
  | nil => exact .nil
  | cons head rest ih =>
    exact .active rfl (fits head (by simp)) rfl (ih (fun r hr => fits r (by simp [hr])))

def GoodStream {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (state : Bits) (values : List (Candidate n (1 + (F + I)))) : Prop :=
  ∀ e ∈ values, e.live.value state = true →
    ∃ r, RouteFits basis words r ∧ (fun i => (e.payload i).value state) = routeBits (I := I) basis words r

theorem Encodes.good {n F I : Nat} {basis : Fin F → Formula} {words state}
    {values : List (Candidate n (1 + (F + I)))} {routes}
    (encoded : Encodes basis words state values routes) : GoodStream basis words state values := by
  induction encoded with
  | nil => simp [GoodStream]
  | @inactive head tail routes live _ ih =>
    intro e member active
    rcases List.mem_cons.mp member with rfl | member
    · simp [live] at active
    · exact ih e member active
  | @active head tail route routes live fits payload _ ih =>
    intro e member active
    rcases List.mem_cons.mp member with rfl | member
    · exact ⟨route, fits, payload⟩
    · exact ih e member active

theorem Encodes.of_bits {n F I : Nat} {basis : Fin F → Formula} {words state}
    (values : List (Candidate n (1 + (F + I)))) (routes : List FlowPair)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (good : GoodStream basis words state values) (fitting : ∀ r ∈ routes, RouteFits basis words r)
    (bits : selected (values.map (fun e => e.value state)) = routes.map (routeBits (I := I) basis words)) :
    Encodes basis words state values routes := by
  induction values generalizing routes with
  | nil => cases routes <;> simp_all; exact .nil
  | cons head rest ih =>
    by_cases live : head.live.value state = true
    · obtain ⟨r, fits, payload⟩ := good head (by simp) live
      cases routes with
      | nil => simp [selected_cons, Candidate.value, live] at bits
      | cons other tail =>
        have pair : (fun i => (head.payload i).value state) = routeBits (I := I) basis words other ∧
            selected (rest.map (fun e => e.value state)) = tail.map (routeBits (I := I) basis words) := by
          simpa only [List.map_cons, selected_cons, Candidate.value, live, ite_true, List.cons.injEq] using bits
        have equal := routeBits_injective_on basis words r other fits (fitting other (by simp)) capacity
          (payload.symm.trans pair.1)
        subst other
        exact .active live fits payload (ih tail (fun e he => good e (by simp [he]))
          (fun x hx => fitting x (by simp [hx])) pair.2)
    · apply Encodes.inactive (Bool.eq_false_iff.mpr live)
      apply ih routes (fun e he => good e (by simp [he])) fitting
      simpa [selected_cons, Candidate.value, live] using bits

theorem firstOccurrences_good {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (state : Bits) (values : List (Candidate n (1 + (F + I))))
    (good : GoodStream basis words state values) : GoodStream basis words state (firstOccurrences values) := by
  induction values using firstOccurrences.induct with
  | case1 => simp [firstOccurrences, GoodStream]
  | case2 head rest ih =>
    have modified : GoodStream basis words state (rest.map (suppress head)) := by
      intro e member active
      obtain ⟨old, contained, rfl⟩ := List.mem_map.mp member
      have wasActive : old.live.value state = true := by
        have both := active
        simp only [suppress, Expr.value, Bool.and_eq_true] at both
        exact both.1
      exact good old (by simp [contained]) wasActive
    have restGood : GoodStream basis words state (firstOccurrences (rest.map (suppress head))) := by
      have adjusted : GoodStream basis words state (rest.map (suppress head)) →
          GoodStream basis words state (firstOccurrences (rest.map (suppress head))) := by simpa only [List.attach_map_val] using ih
      exact adjusted modified
    intro e member active
    simp only [firstOccurrences, List.mem_cons] at member
    rcases member with rfl | member
    · exact good e (by simp) active
    · exact restGood e member active

theorem Encodes.deduplicate {n F I : Nat} {basis : Fin F → Formula} {words state}
    {values : List (Candidate n (1 + (F + I)))} {routes}
    (encoded : Encodes basis words state values routes) (capacity : ([] :: words).length ≤ 2 ^ I) :
    Encodes basis words state (firstOccurrences values) routes.eraseDups := by
  apply Encodes.of_bits _ _ capacity (firstOccurrences_good basis words state values encoded.good)
    (fun r hr => encoded.fitting r (List.mem_eraseDups.mp hr))
  rw [firstOccurrences_correct, encoded.bits]
  exact eraseDups_map_injective_on _ routes (fun a ha b hb =>
    routeBits_injective_on basis words a b (encoded.fitting a ha) (encoded.fitting b hb) capacity)

end Semantic.Circuit
