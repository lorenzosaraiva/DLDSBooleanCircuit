import Circuit.PositionOracle

namespace Semantic.Circuit

theorem loadData_wire_bound (control base : Nat) (data : List Bool) (before : control < base)
    (wire : Nat) (member : wire ∈ support (loadData control base data before)) :
    wire < base + data.length := by
  simp only [loadData, loadWires, support, List.mem_flatMap, List.mem_map] at member
  obtain ⟨_, ⟨target, _, rfl⟩, own⟩ := member
  simp only [Gate.support, Gate.cnot, List.mem_cons, List.not_mem_nil, or_false] at own
  rcases own with equal | equal
  · obtain ⟨i, address, _⟩ := (loadingTargets_mem base data target.val).mp target.property
    have := i.isLt
    omega
  · omega

theorem blockPrefix_wire_bound (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) (wire : Nat)
    (member : wire ∈ support (blockPrefix d c R F I width key base space position)) :
    wire < base + blockWireCount d c R F I width key position := by
  let flag := indexFlagProgram width key
  let program := positionProgram d c R F I position
  have targetBound : program.target < program.wires := by simp [Shared.Program.target, Shared.Program.wires]
  have inputBound : positionInputCount d R F I position < program.wires := by
    simp only [Shared.Program.wires]
    omega
  dsimp only [program] at targetBound inputBound
  simp only [blockPrefix, support, List.flatMap_append, List.mem_append] at member
  rcases member with ((first | second) | third) | last
  · obtain ⟨source, contained, equal⟩ := (rename_support _ _ _ wire).mp first
    have bound := flag.support_bound source contained
    simp only [Shared.Program.wires] at bound
    dsimp only [flag] at bound
    simp only [placeWork] at equal
    simp only [blockWireCount]
    split_ifs at equal <;> omega
  · have bound := loadData_wire_bound _ _ _ _ wire second
    rw [positionData_eq] at bound
    simp only [List.length_map, List.length_finRange] at bound
    simp only [blockCheckBase, blockFlag, blockWireCount] at *
    omega
  · rcases controlled_support _ _ _ third with equal | contained
    · simp only [blockFlag, blockWireCount] at *
      omega
    · obtain ⟨source, contained, equal⟩ := (rename_support _ _ _ wire).mp contained
      have bound := program.support_bound source contained
      dsimp only [program] at bound
      simp only [blockCheckBase, blockFlag, blockWireCount] at *
      omega
  · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, Gate.support, Gate.x,
      List.mem_cons, List.not_mem_nil, or_false] at last
    simp only [blockCheckBase, blockFlag, blockWireCount] at *
    omega

theorem positionBlock_wire_bound (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width key base : Nat)
    (space : width < base) (position : MultiIntroPositions.Position) (wire : Nat)
    (member : wire ∈ support (positionBlock d c R F I width key base space position)) :
    wire < base + blockWireCount d c R F I width key position := by
  simp only [positionBlock, support, List.flatMap_append, List.mem_append] at member
  rcases member with (first | middle) | last
  · exact blockPrefix_wire_bound d c R F I width key base space position wire first
  · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, blockCopy, Gate.toffoli,
      Gate.support, List.mem_cons, List.not_mem_nil, or_false] at middle
    have targetBound : (positionProgram d c R F I position).target <
        (positionProgram d c R F I position).wires := by simp [Shared.Program.target, Shared.Program.wires]
    simp only [blockCheckBase, blockFlag, blockWireCount] at *
    rcases middle with equal | equal | equal <;> omega
  · have original := (support_inverse _ wire).mp last
    exact blockPrefix_wire_bound d c R F I width key base space position wire original

theorem positionBlocks_wire_bound (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I width : Nat)
    (entries : List (Nat × MultiIntroPositions.Position)) (base : Nat) (space : width < base) (wire : Nat)
    (member : wire ∈ support (positionBlocks d c R F I width base entries space)) :
    wire < base + (entries.map fun entry => blockWireCount d c R F I width entry.1 entry.2).sum := by
  induction entries generalizing base with
  | nil => simp [positionBlocks, support] at member
  | cons head tail ih =>
    simp only [positionBlocks, support, List.flatMap_append, List.mem_append] at member
    simp only [List.map_cons, List.sum_cons]
    rcases member with first | later
    · have := positionBlock_wire_bound d c R F I width head.1 base space head.2 wire first
      omega
    · have := ih (base + blockWireCount d c R F I width head.1 head.2) (by omega) later
      omega

theorem oracle_wire_bound (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat)
    (wire : Nat) (member : wire ∈ support (oracle d c R F I)) :
    wire < oracleWireCount d c R F I := positionBlocks_wire_bound _ _ _ _ _ _ _ _ _ _ member

theorem oracle_resources (d : Graph) (c : RouteCertificate.CompleteProfileCertificate) (R F I : Nat) :
    oracleWireCount d c R F I = oracleIndexWidth d + 1 +
      ((oracleEntries d).map fun entry => (indexFlagProgram (oracleIndexWidth d) entry.1).code.length + 1 +
        (positionProgram d c R F I entry.2).wires).sum ∧
    (oracle d c R F I).length = ((oracleEntries d).map fun entry =>
      2 * ((indexFlagProgram (oracleIndexWidth d) entry.1).oracle.length +
        (loadingTargets 0 (positionData d c R F I entry.2)).length +
        (positionProgram d c R F I entry.2).oracle.length + 1) + 1).sum :=
  ⟨rfl, oracle_gates_exact d c R F I⟩

end Semantic.Circuit
