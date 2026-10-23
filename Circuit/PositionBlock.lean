import Circuit.OracleLayout

namespace Semantic.Circuit

def blockFlag (width key base : Nat) : Nat := base + (indexFlagProgram width key).code.length
def blockCheckBase (width key base : Nat) : Nat := blockFlag width key base + 1

def blockPrefix (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) : Circuit :=
  rename (placeWork width base) (placeWork_injective width base (by omega)) (indexFlagProgram width key).oracle ++
    loadData (blockFlag width key base) (blockCheckBase width key base)
      (positionData d c R F I position) (by simp [blockCheckBase]) ++
    controlled (blockFlag width key base)
      (rename (fun wire => blockCheckBase width key base + wire) (fun _ _ h => Nat.add_left_cancel h)
        (positionProgram d c R F I position).oracle) ++
    [Gate.x (blockCheckBase width key base + (positionProgram d c R F I position).target)]

def blockCopy (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) : Gate :=
  Gate.toffoli (blockFlag width key base)
    (blockCheckBase width key base + (positionProgram d c R F I position).target) width
    (by simp only [blockCheckBase]; omega)
    (by simp only [blockFlag]; omega)
    (by simp only [blockCheckBase, blockFlag]; omega)

def positionBlock (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) : Circuit :=
  let prepared := blockPrefix d c R F I width key base space position
  prepared ++ [blockCopy d c R F I width key base space position] ++ inverse prepared

def blockWireCount (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key : Nat)
    (position : MultiIntroPositions.Position) : Nat :=
  (indexFlagProgram width key).code.length + 1 + (positionProgram d c R F I position).wires

def blockGateCount (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key : Nat)
    (position : MultiIntroPositions.Position) : Nat :=
  2 * ((indexFlagProgram width key).oracle.length +
    (loadingTargets 0 (positionData d c R F I position)).length +
    (positionProgram d c R F I position).oracle.length + 1) + 1

theorem loadingTargets_length (left right : Nat) (data : List Bool) :
    (loadingTargets left data).length = (loadingTargets right data).length := by
  simp [loadingTargets]

theorem loadData_length (control base : Nat) (data : List Bool) (before : control < base) :
    (loadData control base data before).length = (loadingTargets 0 data).length := by
  simp only [loadData, loadWires, List.length_map, List.length_attach]
  exact loadingTargets_length _ _ _

theorem positionBlock_gates (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) :
    (positionBlock d c R F I width key base space position).length =
      blockGateCount d c R F I width key position := by
  simp only [positionBlock, blockPrefix, blockGateCount, List.length_append, List.length_singleton,
    inverse, List.length_reverse, rename, List.length_map, loadData_length, controlled_length]
  omega

theorem blockFlag_placed (width key base : Nat) :
    placeWork width base (indexFlagProgram width key).target = blockFlag width key base := by
  simp [placeWork, Shared.Program.target, blockFlag]

theorem blockPrefix_fresh (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) :
    width ∉ support (blockPrefix d c R F I width key base space position) := by
  have flagFresh : width ∉ support (rename (placeWork width base)
      (placeWork_injective width base (by omega)) (indexFlagProgram width key).oracle) := by
    rw [rename_support]
    rintro ⟨wire, _, equal⟩
    unfold placeWork at equal
    split_ifs at equal <;> omega
  have loadFresh : width ∉ support (loadData (blockFlag width key base) (blockCheckBase width key base)
      (positionData d c R F I position) (by simp [blockCheckBase])) := by
    intro member
    have h := loadData_support _ _ _ _ _ member
    simp only [blockFlag, blockCheckBase] at h
    omega
  have checkFresh : width ∉ support (controlled (blockFlag width key base)
      (rename (fun wire => blockCheckBase width key base + wire) (fun _ _ h => Nat.add_left_cancel h)
        (positionProgram d c R F I position).oracle)) := by
    intro member
    rcases controlled_support _ _ _ member with h | h
    · simp only [blockFlag] at h; omega
    · rw [rename_support] at h
      obtain ⟨wire, _, equal⟩ := h
      simp only [blockCheckBase, blockFlag] at equal
      omega
  have lastFresh : width ∉ support [Gate.x (blockCheckBase width key base +
      (positionProgram d c R F I position).target)] := by
    simp only [support, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      Gate.support, Gate.x, List.not_mem_nil, List.mem_cons, or_false]
    simp only [blockCheckBase, blockFlag]
    omega
  simpa only [blockPrefix, support, List.flatMap_append, List.mem_append, not_or] using
    And.intro (And.intro (And.intro flagFresh loadFresh) checkFresh) lastFresh

theorem blockPrefix_enabled (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base value : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) (target : Bool)
    (keyBound : key < 2 ^ width) (valueBound : value < 2 ^ width)
    (metadata : c.graph = d) (fits : PositionFits d c R F I position) :
    (blockCopy d c R F I width key base space position).enabled
      (eval (blockPrefix d c R F I width key base space position) (oracleState width value target)) =
      if value = key then !(Flow.Robustness.check d c position) else false := by
  let flag := blockFlag width key base
  let localBase := blockCheckBase width key base
  let program := positionProgram d c R F I position
  let initial := oracleState width value target
  let flagged := Function.update initial flag (decide (value = key))
  let loaded := eval (loadData flag localBase (positionData d c R F I position) (by simp [localBase, flag, blockCheckBase])) flagged
  let check := rename (fun wire => localBase + wire) (fun _ _ h => Nat.add_left_cancel h) program.oracle
  have flagResult := indexFlag_placed width key value base space keyBound valueBound initial
    (oracleState_inputs width value target) (oracleState_clean width value base target space)
  rw [blockFlag_placed] at flagResult
  have controlOutside : flag ∉ support check := by
    rw [rename_support]
    rintro ⟨wire, _, equal⟩
    dsimp [flag, localBase, blockCheckBase] at equal
    omega
  have loadFlag : loaded flag = decide (value = key) := by
    dsimp only [loaded]
    rw [loadData_outside _ _ _ _ _ _ (Or.inl (by dsimp [localBase, blockCheckBase, flag]; omega))]
    simp [flagged]
  have targetNe : flag ≠ localBase + program.target := by
    dsimp [localBase, blockCheckBase, flag]
    omega
  have prepared : eval (blockPrefix d c R F I width key base space position) initial =
      eval [Gate.x (localBase + program.target)] (eval (controlled flag check) loaded) := by
    simp only [blockPrefix, eval_append]
    rw [flagResult]
  rw [prepared]
  by_cases selected : value = key
  · have selectedFlag : loaded flag = true := by simpa [selected] using loadFlag
    have view : loaded ∘ (fun wire => localBase + wire) = positionInput d c R F I position := by
      apply position_loading _ _ _ _ _ _ flag localBase _ flagged
      · simp [flagged, selected]
      · intro wire bound
        have flagNe : wire ≠ flag := by dsimp [localBase, blockCheckBase, flag] at bound; omega
        dsimp only [flagged]
        rw [Function.update_of_ne flagNe]
        apply oracleState_clean width value base target space
        dsimp [localBase, blockCheckBase, flag, blockFlag] at bound
        omega
    have computed : eval check loaded = Function.update loaded (localBase + program.target)
        (Flow.Robustness.check d c position) := by
      apply rename_update
      rw [view, Shared.Program.oracle_correct _ _ (positionInput_clean d c R F I position),
        positionProgram_correct d c R F I position metadata fits]
    rw [controlled_true _ _ _ controlOutside selectedFlag, computed]
    simp [blockCopy, Gate.toffoli, Gate.enabled, eval, Gate.eval, Gate.x,
      targetNe, selectedFlag, selected, flag, localBase, program]
  · rw [controlled_false _ _ _ controlOutside (by simpa [selected] using loadFlag)]
    simp [blockCopy, Gate.toffoli, Gate.enabled, eval, Gate.eval, Gate.x,
      targetNe, loadFlag, selected, flag, localBase, program]

theorem positionBlock_correct (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base value : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) (target : Bool)
    (keyBound : key < 2 ^ width) (valueBound : value < 2 ^ width)
    (metadata : c.graph = d) (fits : PositionFits d c R F I position) :
    eval (positionBlock d c R F I width key base space position) (oracleState width value target) =
      oracleState width value (target ^^ (if value = key then !(Flow.Robustness.check d c position) else false)) := by
  rw [positionBlock, conjugate_gate _ _ _ (blockPrefix_fresh d c R F I width key base space position),
    blockPrefix_enabled d c R F I width key base value space position target keyBound valueBound metadata fits]
  simp [blockCopy, Gate.toffoli, oracleState]

end Semantic.Circuit
