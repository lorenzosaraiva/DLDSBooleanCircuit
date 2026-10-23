import Circuit.OracleResources

namespace Semantic.Circuit


theorem oracle_xor (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (metadata : c.graph = d) (fits : OracleFits d c R F I)
    (value : Fin (MultiIntroPositions.positions d).length) (target : Bool) :
    eval (oracle d c R F I) (oracleState (oracleIndexWidth d) value.val target) =
      oracleState (oracleIndexWidth d) value.val
        (target ^^ !(Flow.Robustness.check d c ((MultiIntroPositions.positions d).get value))) := by
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
    metadata allFits bounds (oracle_index_bound d value) (oracleIndexWidth d + 1) (by omega) target
  rw [markedPositions_selected d c value.val _ _ (oracleEntries_unique d) member] at correct
  exact correct

end Semantic.Circuit
