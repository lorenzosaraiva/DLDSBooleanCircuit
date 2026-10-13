import Robustness.SelectedProducers.Selected
import Robustness.SelectedProducers.Decoder

/-! Canonical provenance words: one tag, six producer indices, and two pointers.
None pointers use zero; Some slot uses slot+1. Invalid tags decode to None;
raw invalid pointers also decode to None, so a future general checker must use
pointerValidB before interpreting them. These round trips do not compile
the non-seed checks, and make no old Formula-list-entry shape identification. -/
open DLDS FlowSemantics
namespace Robustness.SelectedProducers

def selectionWords (kind : SelectedKind) : Fin 7 → Nat :=
  match kind with
  | .seedDefault => fun _ => 0
  | .seedAncestor ancestor => fun field => if field = 0 then 1 else if field = 1 then ancestor else 0
  | .elimination pair minor major me ae => ![2, pair, minor, major, me, ae, 0]
  | .introduction route edgeIndex => ![3, route, edgeIndex, 0, 0, 0, 0]
  | .mixedIntroduction premise route edgeIndex => ![4, premise, route, edgeIndex, 0, 0, 0]
  | .eliminationReseed pair minor major me ae ancestor => ![5, pair, minor, major, me, ae, ancestor]
  | .introductionReseed route edgeIndex ancestor => ![6, route, edgeIndex, ancestor, 0, 0, 0]

def decodeSelection (words : Fin 7 → Nat) : Option SelectedKind :=
  match words 0 with
  | 0 => some .seedDefault
  | 1 => some (.seedAncestor (words 1))
  | 2 => some (.elimination (words 1) (words 2) (words 3) (words 4) (words 5))
  | 3 => some (.introduction (words 1) (words 2))
  | 4 => some (.mixedIntroduction (words 1) (words 2) (words 3))
  | 5 => some (.eliminationReseed (words 1) (words 2) (words 3) (words 4) (words 5) (words 6))
  | 6 => some (.introductionReseed (words 1) (words 2) (words 3))
  | _ => none

theorem selection_roundtrip (kind : SelectedKind) :
    decodeSelection (selectionWords kind) = some kind := by
  cases kind <;> rfl

def encodePointer {routes : Nat} : Option (Fin routes) → Nat
  | none => 0
  | some slot => slot.val + 1

def pointerValidB (routes value : Nat) : Bool := decide (value ≤ routes)

def decodePointer (routes value : Nat) : Option (Fin routes) :=
  if zero : value = 0 then none
  else if bound : value ≤ routes then some ⟨value - 1, by omega⟩ else none

theorem pointer_roundtrip {routes : Nat} (pointer : Option (Fin routes)) :
    decodePointer routes (encodePointer pointer) = pointer := by
  cases pointer with
  | none => rfl
  | some slot =>
    have bound : slot.val + 1 ≤ routes := by omega
    simp [encodePointer, decodePointer, bound]

@[ext] structure CertificatePayload {routes formulas : Nat} (shape : RouteShape routes formulas) where
  route : Fin routes → FlowPair
  selection : Fin routes → Option SelectedKind
  pointer : Fin routes → Fin 2 → Option (Fin routes)

abbrev CertificateCoord {routes formulas : Nat} (shape : RouteShape routes formulas) (width : Nat) :=
  (Fin routes × Fin 9 × Fin width) ⊕ RouteCoord shape

def payloadWords {routes formulas : Nat} {shape : RouteShape routes formulas}
    (data : CertificatePayload shape) (slot : Fin routes) (field : Fin 9) : Nat :=
  if selected : field.val < 7 then
    match data.selection slot with
    | none => 7
    | some kind => selectionWords kind ⟨field.val, selected⟩
  else encodePointer (data.pointer slot ⟨field.val - 7, by omega⟩)

def encodeCertificateBits {routes formulas : Nat} {shape : RouteShape routes formulas}
    (width : Nat) (data : CertificatePayload shape) : CertificateCoord shape width → Bool
  | .inl (slot, field, bit) => encodeWord width (payloadWords data slot field) bit
  | .inr coordinate => encodeRoutes shape data.route coordinate

def decodeCertificateBits {routes formulas : Nat} (shape : RouteShape routes formulas)
    (width : Nat) (bits : CertificateCoord shape width → Bool) : CertificatePayload shape where
  route := decodeRoutes shape (fun coordinate => bits (.inr coordinate))
  selection slot := decodeSelection fun field =>
    decodeWord fun bit => bits (.inl (slot, ⟨field.val, by omega⟩, bit))
  pointer slot field := decodePointer routes (decodeWord fun bit =>
    bits (.inl (slot, ⟨field.val + 7, by omega⟩, bit)))

/-- Every variable word is range bounded; dependencies are canonical and
represented, and each residual has exactly its shape's length and colour width.
The graph and route owners are fixed statement data, not bit coordinates. -/
def ValidPayload {routes formulas : Nat} {shape : RouteShape routes formulas}
    (width : Nat) (data : CertificatePayload shape) : Prop :=
  ValidRoutes shape data.route ∧ ∀ slot field, payloadWords data slot field < 2 ^ width

theorem certificate_bits_roundtrip {routes formulas : Nat} (shape : RouteShape routes formulas)
    (width : Nat) (data : CertificatePayload shape) (valid : ValidPayload width data) :
    decodeCertificateBits shape width (encodeCertificateBits width data) = data := by
  cases data with
  | mk route selection pointer =>
    apply Robustness.SelectedProducers.CertificatePayload.ext
    · exact routes_roundtrip shape route valid.1
    · funext slot
      change decodeSelection (fun field => decodeWord
        (encodeWord width (payloadWords (⟨route, selection, pointer⟩ : CertificatePayload shape) slot ⟨field.val, by omega⟩))) = selection slot
      have words : (fun field : Fin 7 => decodeWord
          (encodeWord width (payloadWords (⟨route, selection, pointer⟩ : CertificatePayload shape) slot ⟨field.val, by omega⟩))) =
          (fun field : Fin 7 => match selection slot with
            | none => 7 | some kind => selectionWords kind field) := by
        funext field
        rw [word_roundtrip _ _ (valid.2 slot _)]
        simp [payloadWords, field.isLt]
      rw [words]
      cases chosen : selection slot with
      | none => rfl
      | some kind => exact selection_roundtrip kind
    · funext slot field
      change decodePointer routes (decodeWord
        (encodeWord width (payloadWords (⟨route, selection, pointer⟩ : CertificatePayload shape) slot ⟨field.val + 7, by omega⟩))) = pointer slot field
      rw [word_roundtrip _ _ (valid.2 slot _)]
      have index : ¬field.val + 7 < 7 := by omega
      simp only [payloadWords, index]
      simpa using pointer_roundtrip (pointer slot field)

end Robustness.SelectedProducers
