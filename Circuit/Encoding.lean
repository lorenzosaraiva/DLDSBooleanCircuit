import Circuit.Basic
import Flow.Size.Bounds
import Mathlib.Data.List.FinRange
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Data.Nat.Log

open FlowSpec

namespace Semantic.Circuit

set_option maxHeartbeats 4000000

structure Slot (formulas indexBits : Nat) where
  occupied : Bool
  dependency : Fin formulas → Bool
  residual : BitVec indexBits

def decodeMask {F : Nat} (basis : Fin F → Formula) (mask : Fin F → Bool) : Dep :=
  normalizeDep ((List.finRange F).filterMap fun i => if mask i then some (basis i) else none)

def encodeRoute {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath) (route : FlowPair) : Slot F I :=
  ⟨true, (fun i => decide (basis i ∈ route.1)), BitVec.ofNat I (([] :: words).idxOf route.2)⟩

def blankSlot (F I : Nat) : Slot F I := ⟨false, fun _ => false, 0⟩

def decodeSlot {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (slot : Slot F I) : Option FlowPair :=
  if slot.occupied then some (decodeMask basis slot.dependency, ([] :: words).getD slot.residual.toNat []) else none

def RouteFits {F : Nat} (basis : Fin F → Formula) (words : List ColourPath) (route : FlowPair) : Prop :=
  CanonicalDep route.1 ∧ (∀ formula ∈ route.1, ∃ i, basis i = formula) ∧ route.2 ∈ [] :: words

def ProfileFits {F : Nat} (R I : Nat) (basis : Fin F → Formula) (words : List ColourPath)
    (profile : List FlowPair) : Prop :=
  profile.length ≤ R ∧ ([] :: words).length ≤ 2 ^ I ∧ ∀ route ∈ profile, RouteFits basis words route

theorem decodeMask_encode {F : Nat} (basis : Fin F → Formula) (dependency : Dep)
    (canonical : CanonicalDep dependency) (supported : ∀ f ∈ dependency, ∃ i, basis i = f) :
    decodeMask basis (fun i => decide (basis i ∈ dependency)) = dependency := by
  apply canonicalDep_eq_of_depSetEq (normalizeDep_canonical _) canonical
  apply (depSetEq_eq_true_iff _ _).mpr
  intro f
  simp only [decodeMask, mem_normalizeDep, List.mem_filterMap]
  constructor
  · rintro ⟨i, _, member⟩
    split at member
    · cases member
      exact of_decide_eq_true (by assumption)
    · contradiction
  · intro member
    obtain ⟨i, equal⟩ := supported f member
    exact ⟨i, List.mem_finRange i, by simp [equal, member]⟩

theorem residual_index_bound {I : Nat} (words : List ColourPath) (word : ColourPath)
    (contained : word ∈ [] :: words) (capacity : ([] :: words).length ≤ 2 ^ I) :
    ([] :: words).idxOf word < 2 ^ I := (List.idxOf_lt_length_of_mem contained).trans_le capacity

theorem decodeRoute_encode {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (route : FlowPair) (fits : RouteFits basis words route) (capacity : ([] :: words).length ≤ 2 ^ I) :
    decodeSlot basis words (encodeRoute (I := I) basis words route) = some route := by
  have mask := decodeMask_encode basis route.1 fits.1 fits.2.1
  have index := residual_index_bound (I := I) words route.2 fits.2.2 capacity
  simp only [decodeSlot, encodeRoute, mask, ite_true, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt index]
  have word : ([] :: words).getD (([] :: words).idxOf route.2) [] = route.2 := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_idxOf fits.2.2]
    rfl
  rw [word]

def encodeProfile {F : Nat} (R I : Nat) (basis : Fin F → Formula) (words : List ColourPath)
    (profile : List FlowPair) (fits : profile.length ≤ R) : List.Vector (Slot F I) R :=
  ⟨profile.map (encodeRoute basis words) ++ List.replicate (R - profile.length) (blankSlot F I), by
    simp only [List.length_append, List.length_map, List.length_replicate]
    omega⟩

def decodeProfile {R F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (code : List.Vector (Slot F I) R) : List FlowPair := code.toList.filterMap (decodeSlot basis words)

theorem decode_encode {F : Nat} (R I : Nat) (basis : Fin F → Formula) (words : List ColourPath)
    (profile : List FlowPair) (fits : ProfileFits R I basis words profile) :
    decodeProfile basis words (encodeProfile R I basis words profile fits.1) = profile := by
  have mapped : (profile.map (encodeRoute (I := I) basis words)).filterMap (decodeSlot basis words) = profile := by
    induction profile with
    | nil => rfl
    | cons route rest ih =>
      simp only [List.map_cons, List.filterMap_cons,
        decodeRoute_encode basis words route (fits.2.2 route (by simp)) fits.2.1]
      rw [ih ⟨by have := fits.1; simp only [List.length_cons] at this; omega,
        fits.2.1, fun r hr => fits.2.2 r (by simp [hr])⟩]
  simp only [decodeProfile, encodeProfile, List.Vector.toList_mk, List.filterMap_append]
  rw [mapped]
  simp [decodeSlot, blankSlot]

def indexWidth (W : Nat) : Nat := Nat.log 2 (W + 1) + 1

theorem index_capacity (W : Nat) : 1 + W ≤ 2 ^ indexWidth W := by
  have := Nat.lt_pow_succ_log_self (by omega : 1 < 2) (W + 1)
  unfold indexWidth
  omega

def graphBasis (d : Graph) : Fin (RefinedEval.formulas d).length → Formula :=
  (RefinedEval.formulas d).get

def graphWords (d : Graph) : List ColourPath :=
  d.PATHS.flatMap (fun a => FlowSize.nonemptySuffixes a.COLOURS)

theorem graph_words_length (d : Graph) : (graphWords d).length = FlowSize.wordTotal d := by
  have := FlowSize.residuals_length d
  change (graphWords d).length + 1 = 1 + FlowSize.wordTotal d at this
  omega

theorem flow_route_fits (d : Graph) (_valid : _root_.Flow d) (v : Vertex) (listed : v ∈ d.NODES)
    (route : FlowPair) (member : route ∈ MultiIntro.evaluateAt .general d (stdFuel d) v) :
    RouteFits (graphBasis d) (graphWords d) route := by
  obtain ⟨canonical, supported⟩ := FlowSize.dependency_support .general d _ v listed route member
  refine ⟨canonical, ?_, ?_⟩
  · intro f hf
    exact List.mem_iff_get.mp (supported f hf)
  · exact FlowSize.residual_member d route.2 (FlowSize.residual_origin .general d _ v route member)

theorem flow_profile_fits (d : Graph) (valid : _root_.Flow d) (v : Vertex) (listed : v ∈ d.NODES) :
    ProfileFits (FlowSize.profileBound d) (indexWidth (FlowSize.wordTotal d)) (graphBasis d) (graphWords d)
      (MultiIntro.evaluateAt .general d (stdFuel d) v) := by
  refine ⟨FlowSize.raw_route_bound .general d _ v listed, ?_, fun r hr => flow_route_fits d valid v listed r hr⟩
  simpa only [List.length_cons, graph_words_length, Nat.add_comm] using index_capacity (FlowSize.wordTotal d)

theorem stored_profile_fits (d : Graph) (certificate : RouteCertificate.CompleteProfileCertificate)
    (formed : MultiIntro.StrongCertificate .general d certificate) (v : Vertex) (listed : v ∈ d.NODES) :
    ProfileFits (FlowSize.profileBound d) (indexWidth (FlowSize.wordTotal d)) (graphBasis d) (graphWords d)
      (RouteCertificate.stored certificate v) := by
  rw [MultiIntro.stored_eq_routes .general d certificate formed v listed]
  exact flow_profile_fits d (MultiIntro.formation_valid .general d certificate formed) v listed

def Slot.bit {F I : Nat} (slot : Slot F I) : Fin (1 + (F + I)) → Bool :=
  Fin.addCases (fun _ => slot.occupied) (Fin.addCases slot.dependency (fun i => slot.residual.getLsbD i.val))

def inputWire {R F I : Nat} (slot : Fin R) (offset : Fin (1 + (F + I))) : Fin (R * (1 + (F + I))) :=
  finProdFinEquiv (slot, offset)

def inputState {R F I : Nat} (code : List.Vector (Slot F I) R) : Bits := fun wire =>
  if inside : wire < R * (1 + (F + I)) then
    let address := finProdFinEquiv.symm (⟨wire, inside⟩ : Fin (R * (1 + (F + I))))
    (code.get address.1).bit address.2
  else false

theorem inputState_wire {R F I : Nat} (code : List.Vector (Slot F I) R)
    (slot : Fin R) (offset : Fin (1 + (F + I))) :
    inputState code (inputWire slot offset).val = (code.get slot).bit offset := by
  unfold inputState
  rw [dif_pos (inputWire slot offset).isLt]
  change (code.get (finProdFinEquiv.symm (finProdFinEquiv (slot, offset))).1).bit
    (finProdFinEquiv.symm (finProdFinEquiv (slot, offset))).2 = _
  rw [Equiv.symm_apply_apply]

theorem inputState_clean {R F I : Nat} (code : List.Vector (Slot F I) R) (wire : Nat)
    (outside : R * (1 + (F + I)) ≤ wire) : inputState code wire = false := by
  simp [inputState, show ¬ wire < R * (1 + (F + I)) by omega]

end Semantic.Circuit
