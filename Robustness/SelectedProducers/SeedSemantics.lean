import Robustness.SelectedProducers.Selected
import Robustness.SelectedProducers.Decoder
import Robustness.Density

/-! Exact seed-producer semantics after the Boolean route decoder.
These definitions specialize the existing selected checker, without an oracle
equality or compiler-correctness premise. -/
open DLDS FlowSemantics Robustness.BitLocality
namespace Robustness.SelectedProducers
set_option maxHeartbeats 2000000

inductive SeedMode where | default | ancestor
  deriving DecidableEq, Repr

abbrev SeedCoord {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) := Fin (3 + 2 * pointerWidth size) ⊕ RouteCoord shape

def ancestorIndex (size : Nat) (header : Fin (3 + 2 * pointerWidth size) → Bool) : Nat :=
  decodeWord fun bit : Fin (pointerWidth size) =>
    header ⟨bit.val, by omega⟩

def seedKind (mode : SeedMode) (size : Nat)
    (header : Fin (3 + 2 * pointerWidth size) → Bool) : SelectedKind :=
  match mode with
  | .default => .seedDefault
  | .ancestor => .seedAncestor (ancestorIndex size header)

def seedCertificate {routes formulas : Nat} (shape : RouteShape routes formulas)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (size : Nat)
    (bits : SeedCoord shape size → Bool) : JustifiedCertificate d routes where
  owner := owner
  ownerListed := listed
  route := decodeRoutes shape (fun coordinate => bits (.inr coordinate))
  kind _ := .source
  premise _ _ := none

def seedTarget? (d : Graph) (v : Vertex) (mode : SeedMode) (size : Nat)
    (header : Fin (3 + 2 * pointerWidth size) → Bool) : Option FlowPair :=
  if v.HYPOTHESIS || (predsOf d v).isEmpty then
    if (ancestorsInto d v).isEmpty then
      match mode with | .default => some ([v.FORMULA], []) | .ancestor => none
    else
      match mode with
      | .default => none
      | .ancestor => ((ancestorsInto d v)[ancestorIndex size header]?).map
          fun ancestor => ([v.FORMULA], ancestor.COLOURS)
  else none

def seedAcceptance {routes formulas : Nat} (shape : RouteShape routes formulas)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (size : Nat) (bits : SeedCoord shape size → Bool) : Bool :=
  selectedJustificationB (seedCertificate shape d owner listed size bits) slot
    (seedKind mode size (fun bit => bits (.inl bit)))

def seedFailure {routes formulas : Nat} (shape : RouteShape routes formulas)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (size : Nat) (bits : SeedCoord shape size → Bool) : Bool :=
  !seedAcceptance shape d owner listed slot mode size bits

private theorem guarded_route_eq (d : Graph) (v : Vertex) (actual target : FlowPair) :
    (decide (actual = target) && routeContinuableAtB d v actual.2) =
      (decide (actual = target) && routeContinuableAtB d v target.2) := by
  by_cases same : actual = target <;> simp [same]

theorem seed_acceptance_eq {routes formulas : Nat} (shape : RouteShape routes formulas)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (size : Nat) (bits : SeedCoord shape size → Bool) :
    seedAcceptance shape d owner listed slot mode size bits =
      match seedTarget? d (owner slot) mode size (fun bit => bits (.inl bit)) with
      | none => false
      | some target => decide (decodeRoutes shape (fun c => bits (.inr c)) slot = target) &&
          routeContinuableAtB d (owner slot) target.2 := by
  have pointed : pointedSlots (seedCertificate shape d owner listed size bits) slot = [] := rfl
  unfold seedAcceptance selectedJustificationB
  simp only [pointedSlots, seedCertificate,
    WitnessableRoutes, List.mem_filter, Bool.decide_and, seedKind, seedTarget?]
  by_cases source : ((owner slot).HYPOTHESIS || (predsOf d (owner slot)).isEmpty) = true
  · simp only [source, if_true, Bool.true_and]
    by_cases anc : (ancestorsInto d (owner slot)).isEmpty = true
    · cases mode <;> simp [selectedRows, source, anc]
      all_goals exact guarded_route_eq _ _ _ _
    · cases mode
      · simp [selectedRows, source, anc, Bool.false_eq_true]
      · cases choice : (ancestorsInto d (owner slot))[ancestorIndex size (fun bit => bits (.inl bit))]?
        <;> simp [selectedRows, source, anc, Bool.false_eq_true, pick, choice]
        all_goals exact guarded_route_eq _ _ _ _
  · have falseSource : ((owner slot).HYPOTHESIS || (predsOf d (owner slot)).isEmpty) = false :=
      Bool.eq_false_iff.mpr source
    simp [falseSource]

theorem decode_dependency_represented {routes formulas : Nat}
    (shape : RouteShape routes formulas) (bits : Fin formulas → Bool) :
    ∀ formula ∈ decodeDependency shape bits, ∃ code, shape.formulaAt code = formula := by
  intro formula member
  simp only [decodeDependency, mem_normalizeDep, List.mem_map] at member
  obtain ⟨code, _, same⟩ := member
  exact ⟨code, same⟩

theorem dependency_singleton_iff {routes formulas : Nat}
    (shape : RouteShape routes formulas) (bits : Fin formulas → Bool) (formula : Formula) :
    decodeDependency shape bits = [formula] ↔
      (∃ code, shape.formulaAt code = formula) ∧
      ∀ code, bits code = decide (shape.formulaAt code = formula) := by
  constructor
  · intro same
    refine ⟨decode_dependency_represented shape bits formula (by simp [same]), ?_⟩
    intro code
    apply Bool.eq_iff_iff.mpr
    simp only [decide_eq_true_eq]
    rw [← decoded_membership shape bits code, same]
    simp
  · rintro ⟨represented, equal⟩
    rw [funext equal]
    have canon : CanonicalDep [formula] := by simpa using normalizeDep_canonical [formula]
    have cover : ∀ f ∈ [formula], ∃ code, shape.formulaAt code = f := by
      intro f member
      simp only [List.mem_singleton] at member
      simpa [member] using represented
    simpa only [List.mem_singleton] using dependency_roundtrip shape [formula] canon cover

theorem decoded_path_eq_iff {routes formulas : Nat}
    (shape : RouteShape routes formulas) (bits : RouteCoord shape → Bool)
    (slot : Fin routes) (target : ColourPath) :
    (decodeRoutes shape bits slot).2 = target ↔
      target.length = shape.pathLength slot ∧
      ∀ position : Fin (shape.pathLength slot),
        decodeWord (fun bit => bits (.inr ⟨slot, position, bit⟩)) =
          target.getD position.val 0 := by
  constructor
  · intro same
    refine ⟨?_, ?_⟩
    · simpa [decodeRoutes] using congrArg List.length same.symm
    · intro position
      have value := congrArg (fun xs : List Nat => xs.getD position.val 0) same
      simpa [decodeRoutes, List.getD_eq_getElem?_getD, List.getElem?_ofFn, position.isLt] using value
  · rintro ⟨length, values⟩
    apply List.ext_getElem
    · simpa [decodeRoutes] using length.symm
    · intro index leftBound rightBound
      have bound : index < shape.pathLength slot := by simpa [decodeRoutes] using leftBound
      have value := values ⟨index, bound⟩
      change (List.ofFn (fun position : Fin (shape.pathLength slot) =>
        decodeWord (fun bit => bits (.inr ⟨slot, position, bit⟩))))[index] =
          target[index]
      rw [List.getElem_ofFn]
      have same : target.getD index 0 = target[index] := (List.getElem_eq_getD 0).symm
      simpa only [same] using value

end Robustness.SelectedProducers
