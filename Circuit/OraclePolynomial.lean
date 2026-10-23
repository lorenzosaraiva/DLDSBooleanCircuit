import Circuit.ProductionPolynomial
import Circuit.SharedRootSize
import Circuit.OracleResources

namespace Semantic.Circuit
open Shared
set_option maxHeartbeats 30000000

def rootAssignmentBound (R F I : Nat) : Nat := 3 + R * (2 * F + 2 * I + 13)
def rootWireBound (R F I : Nat) : Nat := R * (1 + (F + I)) + rootAssignmentBound R F I + 1
def rootGateBound (R F I : Nat) : Nat := 4 * rootAssignmentBound R F I + 1
def localWireBound (R F I g : Nat) : Nat := productionWireBound R F I g + rootWireBound R F I + 2
def localGateBound (R F I g : Nat) : Nat := productionGateBound R F I g + rootGateBound R F I + 5
def localInputBound (R F I g : Nat) : Nat := (g + 1) * (R * (1 + (F + I)))
def oracleWireBound (R F I g : Nat) : Nat := g + 2 + g * (2 * (g + 1) + 2 + localWireBound R F I g)
def oracleGateBound (R F I g : Nat) : Nat :=
  g * (2 * (8 * (g + 1) + 5 + localInputBound R F I g + localGateBound R F I g) + 3)

theorem root_resource_polynomial (R F I : Nat) :
    (rootResourceCountShared R F I).1 ≤ rootWireBound R F I ∧
    (rootResourceCountShared R F I).2 ≤ rootGateBound R F I :=
  ⟨(root_polynomial_size R F I).1.le, (root_polynomial_size R F I).2⟩

theorem indexFlag_steps_le (width key : Nat) : (indexFlagProgram width key).code.length ≤ 2 * width + 1 := by
  change (assemble width (expression width (equalConstant (fun i : Fin width => i)
    (fun i => (BitVec.ofNat width key).getLsbD i.val)) #[]).2.toList).length ≤ _
  rw [assemble_length, Array.length_toList, expression_work]
  simp only [Array.size_empty, Nat.zero_add, equalConstant, work_all, List.length_map,
    List.length_finRange, List.map_map, Function.comp_def]
  have h := sum_map_bound (List.finRange width)
    (fun i => work (if (BitVec.ofNat width key).getLsbD i.val then Expr.bit i else (Expr.bit i).neg)) 1
    (by intro i _; dsimp only; split <;> simp [work])
  simp only [List.length_finRange, Nat.mul_one] at h
  omega

theorem indexFlag_gates_le (width key : Nat) : (indexFlagProgram width key).oracle.length ≤ 8 * width + 5 := by
  have h := (indexFlagProgram width key).gate_bound
  have s := indexFlag_steps_le width key
  omega

theorem position_program_polynomial (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) :
    (positionProgram d c R F I position).wires ≤ localWireBound R F I (graphSize d) ∧
    (positionProgram d c R F I position).oracle.length ≤ localGateBound R F I (graphSize d) := by
  have production := fun v => production_polynomial_size d v R F I
  have root := root_resource_polynomial R F I
  rw [rootResourceCountShared_exact] at root
  have metadata : ∀ kind, (metadataProgram d c kind).wires = 2 ∧ (metadataProgram d c kind).oracle.length ≤ 5 := by
    intro kind
    exact ⟨metadata_wires d c kind, (metadataProgram d c kind).gate_bound⟩
  cases position with
  | productionRow slot v =>
    have h := production v
    rw [productionResourcesShared_exact] at h
    exact ⟨h.1.trans (by dsimp [localWireBound]; omega), h.2.trans (by dsimp [localGateBound]; omega)⟩
  | rootDischarge slot v =>
    exact ⟨root.1.trans (by dsimp [localWireBound]; omega), root.2.trans (by dsimp [localGateBound]; omega)⟩
  | graphIdentity =>
    have h := metadata MetadataKind.graphIdentity
    exact ⟨h.1.le.trans (by dsimp [localWireBound]; omega),
      h.2.trans (by dsimp [localGateBound]; omega)⟩
  | rowKeyOrder =>
    have h := metadata MetadataKind.rowKeyOrder
    exact ⟨h.1.le.trans (by dsimp [localWireBound]; omega),
      h.2.trans (by dsimp [localGateBound]; omega)⟩
  | structuralDomain =>
    have h := metadata MetadataKind.structuralDomain
    exact ⟨h.1.le.trans (by dsimp [localWireBound]; omega),
      h.2.trans (by dsimp [localGateBound]; omega)⟩

theorem position_loading_length_le (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (position : MultiIntroPositions.Position) :
    (loadingTargets 0 (positionData d c R F I position)).length ≤ localInputBound R F I (graphSize d) := by
  have load : (loadingTargets 0 (positionData d c R F I position)).length ≤
      (positionData d c R F I position).length := by
    simp only [loadingTargets, List.length_map]
    exact (List.length_filter_le ..).trans (by simp)
  conv_rhs at load => rw [positionData_eq]; simp only [List.length_map, List.length_finRange]
  apply load.trans
  cases position with
  | productionRow slot v =>
    have h := Nat.mul_le_mul_right (R * (1 + (F + I))) (graph_reads_le d v)
    dsimp [positionInputCount, productionInputCount, localInputBound]
    nlinarith
  | rootDischarge slot v =>
    simpa [positionInputCount, localInputBound] using
      Nat.mul_le_mul_right (R * (1 + (F + I))) (by omega : 1 ≤ graphSize d + 1)
  | graphIdentity => simp [positionInputCount]
  | rowKeyOrder => simp [positionInputCount]
  | structuralDomain => simp [positionInputCount]

theorem oracle_index_width_le (d : Graph) : oracleIndexWidth d ≤ graphSize d + 1 := by
  have h := Nat.log_le_self 2 ((MultiIntroPositions.positions d).length + 1)
  have p := graph_positions_le d
  dsimp [oracleIndexWidth, indexWidth]
  omega

theorem oracle_polynomial_size (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    oracleWireCount d c R F I ≤ oracleWireBound R F I (graphSize d) ∧
    oracleGateCount d c R F I ≤ oracleGateBound R F I (graphSize d) := by
  have width := oracle_index_width_le d
  have count : (oracleEntries d).length ≤ graphSize d := by
    simp only [oracleEntries, List.length_map, List.length_finRange]
    have h := graph_positions_le d
    omega
  have wires : ∀ e ∈ oracleEntries d, blockWireCount d c R F I (oracleIndexWidth d) e.1 e.2 ≤
      2 * (graphSize d + 1) + 2 + localWireBound R F I (graphSize d) := by
    intro e _
    have s := indexFlag_steps_le (oracleIndexWidth d) e.1
    have component := (position_program_polynomial d c R F I e.2).1
    dsimp only [blockWireCount]
    omega
  have gates : ∀ e ∈ oracleEntries d, blockGateCount d c R F I (oracleIndexWidth d) e.1 e.2 ≤
      2 * (8 * (graphSize d + 1) + 5 + localInputBound R F I (graphSize d) +
        localGateBound R F I (graphSize d)) + 3 := by
    intro e _
    have flag := indexFlag_gates_le (oracleIndexWidth d) e.1
    have load := position_loading_length_le d c R F I e.2
    have component := (position_program_polynomial d c R F I e.2).2
    dsimp only [blockGateCount]
    omega
  constructor
  · have total := sum_map_bound (oracleEntries d) _ _ wires
    have t := Nat.mul_le_mul_right (2 * (graphSize d + 1) + 2 + localWireBound R F I (graphSize d)) count
    dsimp only [oracleWireCount, oracleWireBound]
    omega
  · have total := sum_map_bound (oracleEntries d) _ _ gates
    have t := Nat.mul_le_mul_right
      (2 * (8 * (graphSize d + 1) + 5 + localInputBound R F I (graphSize d) + localGateBound R F I (graphSize d)) + 3) count
    exact total.trans t

end Semantic.Circuit
