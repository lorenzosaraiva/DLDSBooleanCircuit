import Circuit.PositionBlock

namespace Semantic.Circuit

def markedPositions (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (value : Nat) :
    List (Nat × MultiIntroPositions.Position) → Bool
  | [] => false
  | head :: tail => (if value = head.1 then !(Flow.Robustness.check d c head.2) else false) ^^
      markedPositions d c value tail

def positionBlocks (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width : Nat) :
    (base : Nat) → List (Nat × MultiIntroPositions.Position) → (width < base) → Circuit
  | _, [], _ => []
  | base, head :: tail, space =>
      positionBlock d c R F I width head.1 base space head.2 ++
        positionBlocks d c R F I width
          (base + blockWireCount d c R F I width head.1 head.2) tail (by omega)

def oracleEntries (d : Graph) : List (Nat × MultiIntroPositions.Position) :=
  (List.finRange (MultiIntroPositions.positions d).length).map fun i =>
    (i.val, (MultiIntroPositions.positions d).get i)

def oracleIndexWidth (d : Graph) : Nat := indexWidth (MultiIntroPositions.positions d).length



def oracle (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) : Circuit :=
  positionBlocks d c R F I (oracleIndexWidth d) (oracleIndexWidth d + 1) (oracleEntries d) (by omega)

def oracleTarget (d : Graph) : Nat := oracleIndexWidth d

def oracleWireCount (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) : Nat :=
  oracleIndexWidth d + 1 + ((oracleEntries d).map fun entry =>
    blockWireCount d c R F I (oracleIndexWidth d) entry.1 entry.2).sum

def oracleGateCount (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) : Nat :=
  ((oracleEntries d).map fun entry => blockGateCount d c R F I (oracleIndexWidth d) entry.1 entry.2).sum

theorem positionBlocks_correct (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width value : Nat)
    (entries : List (Nat × MultiIntroPositions.Position)) (metadata : c.graph = d)
    (fits : ∀ entry ∈ entries, PositionFits d c R F I entry.2)
    (keyBounds : ∀ entry ∈ entries, entry.1 < 2 ^ width) (valueBound : value < 2 ^ width)
    (base : Nat) (space : width < base) (target : Bool) :
    eval (positionBlocks d c R F I width base entries space) (oracleState width value target) =
      oracleState width value (target ^^ markedPositions d c value entries) := by
  induction entries generalizing base target with
  | nil => simp [positionBlocks, eval, markedPositions]
  | cons head tail ih =>
    rw [positionBlocks, eval_append,
      positionBlock_correct d c R F I width head.1 base value space head.2 target
        (keyBounds head (by simp)) valueBound metadata (fits head (by simp)),
      ih (fun e h => fits e (by simp [h])) (fun e h => keyBounds e (by simp [h]))]
    simp [markedPositions]

theorem markedPositions_absent (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (value : Nat)
    (entries : List (Nat × MultiIntroPositions.Position))
    (absent : ∀ entry ∈ entries, value ≠ entry.1) : markedPositions d c value entries = false := by
  induction entries with
  | nil => rfl
  | cons head tail ih =>
    rw [markedPositions, if_neg (absent head (by simp)),
      ih (fun e h => absent e (by simp [h]))]
    rfl

theorem markedPositions_selected (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (value : Nat)
    (position : MultiIntroPositions.Position) (entries : List (Nat × MultiIntroPositions.Position))
    (unique : (entries.map Prod.fst).Nodup) (member : (value, position) ∈ entries) :
    markedPositions d c value entries = !(Flow.Robustness.check d c position) := by
  induction entries with
  | nil => contradiction
  | cons head tail ih =>
    have nd := List.nodup_cons.mp unique
    rcases List.mem_cons.mp member with equal | later
    · subst head
      rw [markedPositions, if_pos rfl,
        markedPositions_absent d c value tail (by
          intro e h equal
          exact nd.1 (List.mem_map.mpr ⟨e, h, equal.symm⟩))]
      simp
    · have different : value ≠ head.1 := by
        intro equal
        exact nd.1 (List.mem_map.mpr ⟨(value, position), later, equal⟩)
      rw [markedPositions, if_neg different, ih nd.2 later]
      simp

theorem oracleEntries_unique (d : Graph) : ((oracleEntries d).map Prod.fst).Nodup := by
  simp only [oracleEntries, List.map_map, Function.comp_def]
  apply List.Nodup.map
  · intro a b equal
    exact Fin.ext equal
  · exact List.nodup_finRange _

theorem oracle_index_bound (d : Graph) (value : Fin (MultiIntroPositions.positions d).length) :
    value.val < 2 ^ oracleIndexWidth d := by
  have := index_capacity (MultiIntroPositions.positions d).length
  unfold oracleIndexWidth
  omega

theorem oracle_correct (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I)
    (value : Fin (MultiIntroPositions.positions d).length) :
    eval (oracle d c R F I) (indexState (oracleIndexWidth d) value.val) =
      Function.update (indexState (oracleIndexWidth d) value.val) (oracleTarget d)
        (!(Flow.Robustness.check d c ((MultiIntroPositions.positions d).get value))) := by
  have member : (value.val, (MultiIntroPositions.positions d).get value) ∈ oracleEntries d :=
    List.mem_map.mpr ⟨value, List.mem_finRange value, rfl⟩
  have allFits : ∀ entry ∈ oracleEntries d, PositionFits d c R F I entry.2 := by
    intro entry contained
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp contained
    exact fits _ (List.get_mem _ _)
  have bounds : ∀ entry ∈ oracleEntries d, entry.1 < 2 ^ oracleIndexWidth d := by
    intro entry contained
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp contained
    exact oracle_index_bound d i
  have correct := positionBlocks_correct d c R F I (oracleIndexWidth d) value.val (oracleEntries d)
    metadata allFits bounds (oracle_index_bound d value) (oracleIndexWidth d + 1) (by omega) false
  rw [markedPositions_selected d c value.val _ _ (oracleEntries_unique d) member] at correct
  have initial : oracleState (oracleIndexWidth d) value.val false = indexState (oracleIndexWidth d) value.val := by
    funext wire
    by_cases h : wire = oracleIndexWidth d
    · subst wire; simp [oracleState, indexState]
    · simp [oracleState, h]
  rw [initial] at correct
  simpa only [Bool.false_xor, oracleState, oracleTarget, oracle] using correct

theorem oracle_marks_invalid (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I) :
    (∃ value : Fin (MultiIntroPositions.positions d).length,
      eval (oracle d c R F I) (indexState (oracleIndexWidth d) value.val) (oracleTarget d) = true) ↔
      ¬ Flow.Robustness.StrongCertificate d c := by
  simp only [oracle_correct d c R F I metadata fits, Function.update_self, Bool.not_eq_true']
  constructor
  · rintro ⟨value, failed⟩ valid
    have passes := (MultiIntroPositions.positions_pass_iff .general d c).mpr valid
    have pass := passes _ (List.get_mem _ value)
    change Flow.Robustness.check d c ((MultiIntroPositions.positions d).get value) = true at pass
    rw [failed] at pass
    contradiction
  · intro invalid
    obtain ⟨position, member, failed⟩ := MultiIntroPositions.invalid_has_failed_position .general d c invalid
    obtain ⟨value, equal⟩ := List.mem_iff_get.mp member
    exact ⟨value, by simpa only [equal] using failed⟩

theorem positionBlocks_gates (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width : Nat)
    (base : Nat) (entries : List (Nat × MultiIntroPositions.Position)) (space : width < base) :
    (positionBlocks d c R F I width base entries space).length =
      (entries.map fun entry => blockGateCount d c R F I width entry.1 entry.2).sum := by
  induction entries generalizing base with
  | nil => rfl
  | cons head tail ih => simp only [positionBlocks, List.length_append, positionBlock_gates,
      ih, List.map_cons, List.sum_cons]

theorem oracle_gates_exact (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    (oracle d c R F I).length = oracleGateCount d c R F I := positionBlocks_gates _ _ _ _ _ _ _ _ _

end Semantic.Circuit
