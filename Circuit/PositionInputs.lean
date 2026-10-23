import Circuit.MetadataCircuit
import Circuit.SharedProduction
import Circuit.SharedRoot
import Circuit.DataLoading

open FlowSpec
namespace Semantic.Circuit

instance profileFitsDecidable {F : Nat} (R I : Nat) (basis : Fin F → Formula)
    (words : List ColourPath) (profile : List FlowPair) : Decidable (ProfileFits R I basis words profile) := by
  unfold ProfileFits RouteFits CanonicalDep
  infer_instance

instance productionFitsDecidable (d : Graph) (v : Vertex) (R F I : Nat)
    (c : RouteCertificate.CompleteProfileCertificate) : Decidable (ProductionFits d v R F I c) := by
  unfold ProductionFits
  infer_instance

def PositionFits (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    MultiIntroPositions.Position → Prop
  | .productionRow _ v => ProductionFits d v R F I c
  | .rootDischarge _ v => ProfileFits R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c v)
  | _ => True

instance positionFitsDecidable (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) : Decidable (PositionFits d c R F I position) := by
  cases position <;> simp only [PositionFits] <;> infer_instance

def OracleFits (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) : Prop :=
  ∀ position ∈ MultiIntroPositions.positions d, PositionFits d c R F I position

def positionInputCount (d : Graph) (R F I : Nat) : MultiIntroPositions.Position → Nat
  | .productionRow _ v => productionInputCount d v R F I
  | .rootDischarge _ _ => R * (1 + (F + I))
  | _ => 0

def positionProgram (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    (position : MultiIntroPositions.Position) → Shared.Program (positionInputCount d R F I position)
  | .productionRow _ v => productionProgramShared d v R F I
  | .rootDischarge _ _ => rootProgramShared R F I
  | .graphIdentity => metadataProgram d c .graphIdentity
  | .rowKeyOrder => metadataProgram d c .rowKeyOrder
  | .structuralDomain => metadataProgram d c .structuralDomain

def positionInput (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    MultiIntroPositions.Position → Bits
  | .productionRow _ v =>
      if fits : ProductionFits d v R F I c then matrixState (encodedReads d v R F I c fits)
      else fun _ => false
  | .rootDischarge _ v =>
      if fits : ProfileFits R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c v)
      then inputState (encodeProfile R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c v) fits.1)
      else fun _ => false
  | _ => fun _ => false

theorem positionInput_clean (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) (wire : Nat)
    (outside : positionInputCount d R F I position ≤ wire) :
    positionInput d c R F I position wire = false := by
  cases position <;> simp only [positionInput, positionInputCount] at *
  all_goals first
    | rfl
    | split <;> first | rfl | exact matrixState_clean _ _ outside | exact inputState_clean _ _ outside

theorem positionProgram_correct (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) (metadata : c.graph = d)
    (fits : PositionFits d c R F I position) :
    (positionProgram d c R F I position).value (positionInput d c R F I position) =
      Flow.Robustness.check d c position := by
  cases position with
  | productionRow slot v =>
      simp only [PositionFits] at fits
      simpa only [positionProgram, positionInput, dif_pos fits] using
        productionProgramShared_correct d v R F I c slot metadata fits
  | rootDischarge slot v =>
      simp only [PositionFits] at fits
      have result := root_correct_shared d c v slot R I (productionBasis d F) (graphWords d) fits
      rw [rootCircuitShared, Shared.Program.oracle_correct _ _ (inputState_clean _)] at result
      have value := congrFun result (rootTargetShared R F I)
      simp [rootTargetShared] at value
      simpa only [positionProgram, positionInput, dif_pos fits] using value
  | graphIdentity =>
      change (Shared.expressionProgram (.constant (Flow.Robustness.check d c (.graphIdentity)) : Expr 0)).value _ = _
      rw [Shared.expressionProgram_value]
      rfl
  | rowKeyOrder =>
      change (Shared.expressionProgram (.constant (Flow.Robustness.check d c (.rowKeyOrder)) : Expr 0)).value _ = _
      rw [Shared.expressionProgram_value]
      rfl
  | structuralDomain =>
      change (Shared.expressionProgram (.constant (Flow.Robustness.check d c (.structuralDomain)) : Expr 0)).value _ = _
      rw [Shared.expressionProgram_value]
      rfl

def positionData (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) : List Bool :=
  match position with
  | .productionRow _ v =>
      if fits : ProductionFits d v R F I c then
        let data := Array.ofFn (encodedReads d v R F I c fits)
        let codes := fun i : Fin (productionReads d v).length => data[i.val]'(by simpa [data] using i.isLt)
        (List.finRange (productionInputCount d v R F I)).map fun i => matrixState codes i.val
      else (List.finRange (productionInputCount d v R F I)).map fun _ => false
  | .rootDischarge _ v =>
      if fits : ProfileFits R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c v) then
        let code := encodeProfile R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c v) fits.1
        (List.finRange (R * (1 + (F + I)))).map fun i => inputState code i.val
      else (List.finRange (R * (1 + (F + I)))).map fun _ => false
  | _ => []

theorem positionData_eq (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) :
    positionData d c R F I position =
      (List.finRange (positionInputCount d R F I position)).map fun i => positionInput d c R F I position i.val := by
  cases position with
  | productionRow slot v =>
      by_cases fits : ProductionFits d v R F I c
      · have cached : (fun i : Fin (productionReads d v).length =>
            (Array.ofFn (encodedReads d v R F I c fits))[i.val]'(by simp)) = encodedReads d v R F I c fits := by
          funext i
          simp
        simp only [positionData, positionInput, positionInputCount, dif_pos fits]
        rw [cached]
      · simp only [positionData, positionInput, positionInputCount, dif_neg fits]
  | rootDischarge slot v =>
      by_cases fits : ProfileFits R I (productionBasis d F) (graphWords d) (RouteCertificate.stored c v)
      · simp only [positionData, positionInput, positionInputCount, dif_pos fits]
      · simp only [positionData, positionInput, positionInputCount, dif_neg fits]
  | graphIdentity => rfl
  | rowKeyOrder => rfl
  | structuralDomain => rfl

theorem position_loading (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) (control base : Nat) (before : control < base)
    (state : Bits) (selected : state control = true)
    (clean : ∀ wire, base ≤ wire → state wire = false) :
    eval (loadData control base (positionData d c R F I position) before) state ∘
      (fun wire => base + wire) = positionInput d c R F I position := by
  rw [positionData_eq]
  exact loadData_view _ _ _ _ before state selected clean (positionInput_clean d c R F I position)

end Semantic.Circuit
