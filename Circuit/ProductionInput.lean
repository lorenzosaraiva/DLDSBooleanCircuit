import Circuit.ProductionAssembly

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 16000000
set_option maxRecDepth 20000

def productionReads (d : Graph) (v : Vertex) : List Vertex := v :: predsOf d v

def ProductionFits (d : Graph) (v : Vertex) (R F I : Nat) (c : RouteCertificate.CompleteProfileCertificate) : Prop :=
  ∀ u ∈ productionReads d v,
    ProfileFits R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c u)

def matrixWire {M R F I : Nat} (profile : Fin M) (slot : Fin R) (offset : Fin (1 + (F + I))) :
    Fin (M * (R * (1 + (F + I)))) := finProdFinEquiv (profile, inputWire slot offset)

def matrixState {M R F I : Nat} (codes : Fin M → List.Vector (Slot F I) R) : Bits := fun wire =>
  if bounded : wire < M * (R * (1 + (F + I))) then
    let address := finProdFinEquiv.symm ⟨wire, bounded⟩
    inputState (codes address.1) address.2.val
  else false

theorem matrixState_wire {M R F I : Nat} (codes : Fin M → List.Vector (Slot F I) R)
    (profile : Fin M) (slot : Fin R) (offset : Fin (1 + (F + I))) :
    matrixState codes (matrixWire profile slot offset).val = ((codes profile).get slot).bit offset := by
  simp only [matrixState, dif_pos (matrixWire profile slot offset).isLt]
  change inputState (codes (finProdFinEquiv.symm (finProdFinEquiv (profile, inputWire slot offset))).1)
    (finProdFinEquiv.symm (finProdFinEquiv (profile, inputWire slot offset))).2.val = _
  rw [Equiv.symm_apply_apply]
  exact inputState_wire _ _ _

theorem matrixState_clean {M R F I : Nat} (codes : Fin M → List.Vector (Slot F I) R)
    (wire : Nat) (outside : M * (R * (1 + (F + I))) ≤ wire) : matrixState codes wire = false := by
  simp only [matrixState, dif_neg (by omega : ¬wire < M * (R * (1 + (F + I))))]

def profileSelections (M R F I : Nat) (profile : Fin M) :
    List (Selection (M * (R * (1 + (F + I)))) (1 + (F + I))) :=
  (List.finRange R).map fun slot =>
    ⟨.bit (matrixWire profile slot (occupiedOffset F I)), fun offset => .bit (matrixWire profile slot offset)⟩

def slotConstant {n F I : Nat} (slot : Slot F I) : Selection n (1 + (F + I)) :=
  ⟨.constant slot.occupied, fun i => .constant (slot.bit i)⟩

theorem encoded_slots {n F I : Nat} (R : Nat) (basis : Fin F → Formula) (words : List ColourPath)
    (routes : List FlowPair) (fits : ProfileFits R I basis words routes) (state : Bits) :
    Encodes basis words state ((encodeProfile R I basis words routes fits.1).toList.map (slotConstant (n := n))) routes := by
  have front : Encodes basis words state
      (routes.map (fun r => slotConstant (n := n) (encodeRoute (I := I) basis words r))) routes :=
    constants_encode basis words routes state fits.2.2
  have back : Encodes basis words state
      ((List.replicate (R - routes.length) (blankSlot F I)).map (slotConstant (n := n))) [] := by
    apply Encodes.inactive_all
    intro e member
    simp only [List.map_replicate, List.mem_replicate] at member
    rcases member with ⟨_, rfl⟩
    rfl
  simpa only [encodeProfile, List.Vector.toList_mk, List.map_append, List.map_map,
    Function.comp_def, List.append_nil] using front.append back

theorem profileSelections_values {M R F I : Nat} (codes : Fin M → List.Vector (Slot F I) R)
    (profile : Fin M) :
    (profileSelections M R F I profile).map (fun e => e.value (matrixState codes)) =
      ((codes profile).toList.map (slotConstant (n := M * (R * (1 + (F + I)))))).map
        (fun e => e.value (matrixState codes)) := by
  have slots : (List.finRange R).map (codes profile).get = (codes profile).toList := by
    obtain ⟨values, rfl⟩ := codes profile
    exact List.map_get_finRange values
  rw [← slots]
  simp only [profileSelections, List.map_map, Function.comp_def, Selection.value, Expr.value, matrixState_wire]
  apply List.map_congr_left
  intro slot _
  simp only [slotConstant, Expr.value, Slot.bit, occupiedOffset, Fin.addCases_left]

def encodedReads (d : Graph) (v : Vertex) (R F I : Nat) (c : RouteCertificate.CompleteProfileCertificate)
    (fits : ProductionFits d v R F I c) : Fin (productionReads d v).length → List.Vector (Slot F I) R :=
  fun index => encodeProfile R I (productionBasis d F) (graphWords d)
    (RouteCertificate.stored c ((productionReads d v).get index))
    (fits _ (List.get_mem _ _)).1

def readSelections (d : Graph) (v : Vertex) (R F I : Nat) (u : Vertex) :
    List (Selection ((productionReads d v).length * (R * (1 + (F + I)))) (1 + (F + I))) :=
  if member : u ∈ productionReads d v then
    profileSelections _ R F I ⟨(productionReads d v).idxOf u, List.idxOf_lt_length_of_mem member⟩
  else []

def restrictedPrior (d : Graph) (v : Vertex) (c : RouteCertificate.CompleteProfileCertificate) (u : Vertex) : List FlowPair :=
  if u ∈ productionReads d v then RouteCertificate.stored c u else []

theorem readSelections_encode (d : Graph) (v : Vertex) (R F I : Nat) (c : RouteCertificate.CompleteProfileCertificate)
    (fits : ProductionFits d v R F I c) (u : Vertex) :
    Encodes (productionBasis d F) (graphWords d) (matrixState (encodedReads d v R F I c fits))
      (readSelections d v R F I u) (restrictedPrior d v c u) := by
  by_cases member : u ∈ productionReads d v
  · simp only [readSelections, restrictedPrior, dif_pos member, if_pos member]
    let index : Fin (productionReads d v).length := ⟨_, List.idxOf_lt_length_of_mem member⟩
    have same : (productionReads d v).get index = u := List.idxOf_get _
    have represented := encoded_slots (n := (productionReads d v).length * (R * (1 + (F + I)))) R
      (productionBasis d F) (graphWords d) (RouteCertificate.stored c u) (fits u member)
      (matrixState (encodedReads d v R F I c fits))
    apply represented.congr_values
    have equal := profileSelections_values (encodedReads d v R F I c fits) index
    simp only [encodedReads, same] at equal
    exact equal.symm
  · simp only [readSelections, restrictedPrior, dif_neg member, if_neg member]
    exact .nil

theorem restricted_step (d : Graph) (v : Vertex) (c : RouteCertificate.CompleteProfileCertificate) :
    MultiIntro.step .general d (restrictedPrior d v c) v =
      MultiIntro.step .general d (RouteCertificate.stored c) v := by
  apply MultiIntro.step_congr
  intro u member
  simp [restrictedPrior, productionReads, member]

end Semantic.Circuit
