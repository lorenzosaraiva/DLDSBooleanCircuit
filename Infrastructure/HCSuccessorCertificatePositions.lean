import Infrastructure.HCSuccessorLocalCertificate















open Semantic FlowSpec
open OccurrenceAwareReplay
open CertificateModel SelfCertifying
open HCSuccessorLocalCertificate

namespace HCSuccessorCertificatePositions

structure SuccessorCertificate where
  positions : List LocalRouteCertificate
  deriving Repr

def claimedRoutesAt? (certificate : SuccessorCertificate) (node : Vertex) :
    Option (List FlowPair) :=
  (certificate.positions.find? fun position => position.node == node).map
    LocalRouteCertificate.claimedRoutes

def premiseEntryCoherentB (certificate : SuccessorCertificate)
    (entry : Vertex × List FlowPair) : Bool :=
  match claimedRoutesAt? certificate entry.1 with
  | none => false
  | some claimed => setEqB entry.2 claimed

def rootClaimsCompleteB (certificate : SuccessorCertificate)
    (root : Vertex) : Bool :=
  match claimedRoutesAt? certificate root with
  | none => false
  | some routes => !routes.isEmpty && routes.all routeClosedB


def successorFormationB (d : Graph)
    (certificate : SuccessorCertificate) : Bool :=
  (certificate.positions.map LocalRouteCertificate.node == d.NODES) &&
  certificate.positions.all (localRouteCertificateB d) &&
  certificate.positions.all (fun position =>
    position.premiseTable.all (premiseEntryCoherentB certificate)) &&
  !(occurrenceRoots d).isEmpty &&
  (occurrenceRoots d).all (rootClaimsCompleteB certificate)

def PremiseEntryCoherent (certificate : SuccessorCertificate)
    (entry : Vertex × List FlowPair) : Prop :=
  ∃ claimed,
    claimedRoutesAt? certificate entry.1 = some claimed ∧
    ListSetEq entry.2 claimed

def RootClaimsComplete (certificate : SuccessorCertificate)
    (root : Vertex) : Prop :=
  ∃ routes,
    claimedRoutesAt? certificate root = some routes ∧
    routes ≠ [] ∧
    ∀ route, route ∈ routes -> routeClosedB route = true

structure SuccessorFormationValid (d : Graph)
    (certificate : SuccessorCertificate) : Prop where
  nodeOrder : certificate.positions.map LocalRouteCertificate.node = d.NODES
  localSound : ∀ localCertificate,
    localCertificate ∈ certificate.positions ->
      LocalRouteCertificateSound d localCertificate
  premiseCoherent : ∀ owner,
    owner ∈ certificate.positions -> ∀ entry,
      entry ∈ owner.premiseTable -> PremiseEntryCoherent certificate entry
  rootsNonempty : (occurrenceRoots d).isEmpty = false
  rootsComplete : ∀ root,
    root ∈ occurrenceRoots d -> RootClaimsComplete certificate root

theorem successorFormationB_sound
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true) :
    SuccessorFormationValid d certificate := by
  have parts :
      (certificate.positions.map LocalRouteCertificate.node == d.NODES) = true ∧
      certificate.positions.all (localRouteCertificateB d) = true ∧
      certificate.positions.all (fun position =>
        position.premiseTable.all (premiseEntryCoherentB certificate)) = true ∧
      !(occurrenceRoots d).isEmpty = true ∧
      (occurrenceRoots d).all (rootClaimsCompleteB certificate) = true := by
    simpa [successorFormationB, Bool.and_assoc] using valid
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simpa using parts.1
  · intro localCertificate member
    have checked := List.all_eq_true.mp parts.2.1 localCertificate member
    exact localRouteCertificateB_sound d localCertificate checked
  · intro owner ownerMember entry entryMember
    have ownerChecked := List.all_eq_true.mp parts.2.2.1 owner ownerMember
    have entryChecked := List.all_eq_true.mp ownerChecked entry entryMember
    unfold premiseEntryCoherentB at entryChecked
    cases lookup : claimedRoutesAt? certificate entry.1 with
    | none => simp [lookup] at entryChecked
    | some claimed =>
        exact ⟨claimed, lookup, setEqB_sound (by simpa [lookup] using entryChecked)⟩
  · cases emptyEq : (occurrenceRoots d).isEmpty with
    | false => rfl
    | true => simp [emptyEq] at parts
  · intro root rootMember
    have checked := List.all_eq_true.mp parts.2.2.2.2 root rootMember
    unfold rootClaimsCompleteB at checked
    cases lookup : claimedRoutesAt? certificate root with
    | none => simp [lookup] at checked
    | some routes =>
        have routeParts : !routes.isEmpty = true ∧
            routes.all routeClosedB = true := by
          simpa [lookup] using checked
        have nonempty : routes ≠ [] := by
          have isNotEmpty : routes.isEmpty = false := by
            cases emptyEq : routes.isEmpty with
            | false => rfl
            | true => simp [emptyEq] at routeParts
          simpa [List.isEmpty_eq_false_iff] using isNotEmpty
        exact ⟨routes, lookup, nonempty,
          List.all_eq_true.mp routeParts.2⟩

inductive SemanticVerificationPosition where
  | nodeOrder
  | rootSet
  | local (certificate : LocalRouteCertificate)
  | premise (owner : LocalRouteCertificate)
      (entry : Vertex × List FlowPair)
  | root (vertex : Vertex)
  deriving Repr

def semanticPositions (d : Graph) (certificate : SuccessorCertificate) :
    List SemanticVerificationPosition :=
  [.nodeOrder] ++
    certificate.positions.map .local ++
    certificate.positions.flatMap (fun owner =>
      owner.premiseTable.map (.premise owner)) ++
    [.rootSet] ++
    (occurrenceRoots d).map .root


def semanticMarkedAtB (d : Graph) (certificate : SuccessorCertificate) :
    SemanticVerificationPosition -> Bool
  | .nodeOrder =>
      !(certificate.positions.map LocalRouteCertificate.node == d.NODES)
  | .rootSet => (occurrenceRoots d).isEmpty
  | .local localCertificate =>
      !localRouteCertificateB d localCertificate
  | .premise _ entry =>
      !premiseEntryCoherentB certificate entry
  | .root root =>
      !rootClaimsCompleteB certificate root

def noSemanticPositionMarkedB (d : Graph)
    (certificate : SuccessorCertificate) : Bool :=
  (semanticPositions d certificate).all fun position =>
    !semanticMarkedAtB d certificate position




theorem noSemanticPositionMarkedB_eq_formation
    (d : Graph) (certificate : SuccessorCertificate) :
    noSemanticPositionMarkedB d certificate =
      successorFormationB d certificate := by
  simp [noSemanticPositionMarkedB, semanticPositions, semanticMarkedAtB, successorFormationB, List.all_append, Function.comp_def, Bool.not_not, Bool.and_assoc]

theorem no_semantic_marked_iff_formation
    (d : Graph) (certificate : SuccessorCertificate) :
    (∀ position, position ∈ semanticPositions d certificate ->
      semanticMarkedAtB d certificate position = false) ↔
      successorFormationB d certificate = true := by
  rw [← noSemanticPositionMarkedB_eq_formation]
  simp [noSemanticPositionMarkedB, List.all_eq_true]

theorem formation_invalid_implies_semantic_marked
    (d : Graph) (certificate : SuccessorCertificate)
    (invalid : successorFormationB d certificate = false) :
    ∃ position, position ∈ semanticPositions d certificate ∧
      semanticMarkedAtB d certificate position = true := by
  by_contra none
  push_neg at none
  have noMarked : ∀ position,
      position ∈ semanticPositions d certificate ->
        semanticMarkedAtB d certificate position = false := by
    intro position member
    cases value : semanticMarkedAtB d certificate position with
    | false => rfl
    | true => exact False.elim (none position member value)
  have valid := (no_semantic_marked_iff_formation d certificate).mp noMarked
  rw [invalid] at valid
  contradiction

end HCSuccessorCertificatePositions
