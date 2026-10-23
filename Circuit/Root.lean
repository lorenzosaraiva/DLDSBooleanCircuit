import Circuit.Encoding
import Circuit.Gadgets
import Robustness.FlowRobustness

open FlowSpec

namespace Semantic.Circuit

set_option maxHeartbeats 5000000

def Slot.closed {F I : Nat} (slot : Slot F I) : Bool :=
  (List.finRange F).all (fun i => !slot.dependency i) &&
    (List.finRange I).all (fun i => !slot.residual.getLsbD i.val)

def Slot.safe {F I : Nat} (slot : Slot F I) : Bool := !slot.occupied || slot.closed

def codeCheck {R F I : Nat} (code : List.Vector (Slot F I) R) : Bool :=
  code.toList.any Slot.occupied && code.toList.all Slot.safe

def profileCheck (profile : List FlowPair) : Bool :=
  !profile.isEmpty && profile.all (fun route => route.1.isEmpty && route.2.isEmpty)

theorem mask_zero_encode {F : Nat} (basis : Fin F → Formula) (dependency : Dep)
    (supported : ∀ f ∈ dependency, ∃ i, basis i = f) :
    (List.finRange F).all (fun i => !decide (basis i ∈ dependency)) = dependency.isEmpty := by
  apply Bool.eq_iff_iff.mpr
  simp only [List.all_eq_true, List.mem_finRange, true_implies, Bool.not_eq_true',
    decide_eq_false_iff_not, List.isEmpty_iff]
  constructor
  · intro absent
    apply List.eq_nil_iff_forall_not_mem.mpr
    intro f member
    obtain ⟨i, equal⟩ := supported f member
    exact absent i (equal ▸ member)
  · rintro rfl
    simp

theorem bitvec_zero_test {I : Nat} (value : BitVec I) :
    (List.finRange I).all (fun i => !value.getLsbD i.val) = decide (value.toNat = 0) := by
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq]
  constructor
  · intro allZero
    have zero : value = 0 := BitVec.eq_of_getLsbD_eq (fun i hi => by
      have h := List.all_eq_true.mp allZero ⟨i, hi⟩ (List.mem_finRange _)
      simpa using h)
    simp [zero]
  · intro allZero
    have zero : value = 0 := BitVec.eq_of_toNat_eq (by simpa using allZero)
    simp [zero]

theorem residual_zero_encode {I : Nat} (words : List ColourPath) (word : ColourPath)
    (member : word ∈ [] :: words) (capacity : ([] :: words).length ≤ 2 ^ I) :
    (List.finRange I).all (fun i => !(BitVec.ofNat I (([] :: words).idxOf word)).getLsbD i.val) =
      word.isEmpty := by
  rw [bitvec_zero_test]
  have index := residual_index_bound (I := I) words word member capacity
  simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt index]
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq, List.isEmpty_iff]
  constructor
  · intro atZero
    have lookup := List.getElem?_idxOf member
    rw [atZero] at lookup
    simpa using lookup.symm
  · rintro rfl
    simp

theorem encodeRoute_closed {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (route : FlowPair) (fits : RouteFits basis words route) (capacity : ([] :: words).length ≤ 2 ^ I) :
    (encodeRoute (I := I) basis words route).closed = (route.1.isEmpty && route.2.isEmpty) := by
  simp only [Slot.closed, encodeRoute]
  rw [mask_zero_encode basis route.1 fits.2.1, residual_zero_encode words route.2 fits.2.2 capacity]

theorem encoded_check {F : Nat} (R I : Nat) (basis : Fin F → Formula) (words : List ColourPath)
    (profile : List FlowPair) (fits : ProfileFits R I basis words profile) :
    codeCheck (encodeProfile R I basis words profile fits.1) = profileCheck profile := by
  have occupied : (profile.map (encodeRoute (I := I) basis words)).any Slot.occupied = !profile.isEmpty := by
    cases profile <;> simp [encodeRoute]
  have safe : (profile.map (encodeRoute (I := I) basis words)).all Slot.safe =
      profile.all (fun r => r.1.isEmpty && r.2.isEmpty) := by
    rw [List.all_map]
    apply all_congr_mem
    intro r member
    change (!(true) || (encodeRoute (I := I) basis words r).closed) = _
    simp only [Bool.not_true, Bool.false_or]
    exact encodeRoute_closed basis words r (fits.2.2 r member) fits.2.1
  simp only [codeCheck, encodeProfile, List.Vector.toList_mk, List.any_append, List.all_append]
  rw [occupied, safe]
  simp [blankSlot, Slot.safe, profileCheck]

def occupiedOffset (F I : Nat) : Fin (1 + (F + I)) := Fin.castAdd (F + I) (0 : Fin 1)
def dependencyOffset {F : Nat} (I : Nat) (i : Fin F) : Fin (1 + (F + I)) := Fin.natAdd 1 (Fin.castAdd I i)
def residualOffset (F : Nat) {I : Nat} (i : Fin I) : Fin (1 + (F + I)) := Fin.natAdd 1 (Fin.natAdd F i)

def slotExpression (R F I : Nat) (slot : Fin R) : Expr (R * (1 + (F + I))) :=
  (Expr.neg (.bit (inputWire slot (occupiedOffset F I)))).disj
    (.conj (equalConstant (fun i : Fin F => inputWire slot (dependencyOffset I i)) (fun _ => false))
      (equalConstant (fun i : Fin I => inputWire slot (residualOffset F i)) (fun _ => false)))

def rootExpression (R F I : Nat) : Expr (R * (1 + (F + I))) :=
  .conj (Expr.any ((List.finRange R).map fun slot => .bit (inputWire slot (occupiedOffset F I))))
    (Expr.all ((List.finRange R).map (slotExpression R F I)))

theorem slotExpression_value {R F I : Nat} (code : List.Vector (Slot F I) R) (slot : Fin R) :
    (slotExpression R F I slot).value (inputState code) = (code.get slot).safe := by
  simp [slotExpression, Expr.value_disj, Expr.value, equalConstant_value, inputState_wire,
    occupiedOffset, dependencyOffset, residualOffset, Slot.bit, Slot.safe, Slot.closed,
    Fin.addCases_left, Fin.addCases_right]

theorem rootExpression_value {R F I : Nat} (code : List.Vector (Slot F I) R) :
    (rootExpression R F I).value (inputState code) = codeCheck code := by
  have slots : (List.finRange R).map code.get = code.toList := by
    obtain ⟨values, rfl⟩ := code
    exact List.map_get_finRange _
  simp only [rootExpression, Expr.value, Expr.value_any, Expr.value_all, List.any_map,
    List.all_map, Function.comp_def, slotExpression_value, inputState_wire]
  simp only [occupiedOffset, Slot.bit, Fin.addCases_left]
  rw [codeCheck, ← slots, List.any_map, List.all_map]
  rfl

def rootCircuit (R F I : Nat) : Circuit := (rootExpression R F I).oracle
def rootTarget (R F I : Nat) : Nat := R * (1 + (F + I)) + (rootExpression R F I).size
def rootWires (R F I : Nat) : Nat := rootTarget R F I + 1



theorem root_correct (d : Graph) (certificate : RouteCertificate.CompleteProfileCertificate) (vertex : Vertex) (positionSlot : Nat)
    {F : Nat} (R I : Nat) (basis : Fin F → Formula) (words : List ColourPath)
    (fits : ProfileFits R I basis words (RouteCertificate.stored certificate vertex)) :
    eval (rootCircuit R F I)
        (inputState (encodeProfile R I basis words (RouteCertificate.stored certificate vertex) fits.1)) =
      Function.update
        (inputState (encodeProfile R I basis words (RouteCertificate.stored certificate vertex) fits.1))
        (rootTarget R F I) (_root_.Flow.Robustness.check d certificate (.rootDischarge positionSlot vertex)) := by
  rw [rootCircuit, Expr.oracle_correct _ _ (inputState_clean _), rootExpression_value,
    encoded_check R I basis words _ fits]
  rfl

theorem root_ancillas_clean {R F I : Nat} (code : List.Vector (Slot F I) R) (wire : Nat)
    (workspace : R * (1 + (F + I)) ≤ wire) (notTarget : wire ≠ rootTarget R F I) :
    eval (rootCircuit R F I) (inputState code) wire = false := by
  rw [rootCircuit, (rootExpression R F I).oracle_restores _ (inputState_clean _) wire notTarget]
  exact inputState_clean code wire workspace

end Semantic.Circuit
