import Circuit.SharedProduction
import Circuit.SharedRoot

namespace Semantic.Circuit

namespace Shared

def computationGates (base : Nat) : List Instruction → Nat
  | [] => 0
  | head :: tail => (head.resolve base).compile.length + computationGates (base + 1) tail

theorem assemble_gates (base : Nat) (instructions : List Instruction) :
    (assemble base instructions).compile.length = computationGates base instructions := by
  induction instructions generalizing base with
  | nil => rfl
  | cons head tail ih => simp [assemble, Sequence.compile, computationGates, ih]

theorem Program.oracle_gates {n : Nat} (program : Program n) :
    program.oracle.length = 2 * program.code.compile.length + 1 := by
  simp only [Program.oracle, List.length_append, List.length_singleton, inverse, List.length_reverse]
  omega

def resourceCount (base : Nat) (code : Store) : Nat × Nat :=
  (base + code.size + 1, 2 * computationGates base code.toList + 1)

theorem resourceCount_bound (base : Nat) (code : Store) :
    (resourceCount base code).2 ≤ 4 * code.size + 1 := by
  have bound := (assemble base code.toList).gate_bound
  rw [assemble_gates, assemble_length, Array.length_toList] at bound
  simp only [resourceCount]
  omega

end Shared

def productionResourcesShared (d : Graph) (v : Vertex) (R F I : Nat) : Nat × Nat :=
  Shared.resourceCount (productionInputCount d v R F I) (productionAssemblyShared d v R F I).2

theorem productionResourcesShared_exact (d : Graph) (v : Vertex) (R F I : Nat) :
    productionResourcesShared d v R F I =
      ((productionProgramShared d v R F I).wires, (productionCircuitShared d v R F I).length) := by
  apply Prod.ext
  · simp [productionResourcesShared, Shared.resourceCount, Shared.Program.wires, productionProgramShared]
  · rw [productionCircuitShared, Shared.Program.oracle_gates]
    simp only [productionResourcesShared, Shared.resourceCount, productionProgramShared, Shared.assemble_gates]

def rootResourceCountShared (R F I : Nat) : Nat × Nat :=
  Shared.resourceCount (R * (1 + (F + I))) (Shared.expression (R * (1 + (F + I))) (rootExpression R F I) #[]).2

theorem rootResourceCountShared_exact (R F I : Nat) :
    rootResourceCountShared R F I =
      ((rootProgramShared R F I).wires, (rootCircuitShared R F I).length) := by
  apply Prod.ext
  · simp [rootResourceCountShared, Shared.resourceCount, Shared.Program.wires, rootProgramShared, Shared.expressionProgram]
  · rw [rootCircuitShared, Shared.Program.oracle_gates]
    simp only [rootResourceCountShared, Shared.resourceCount, rootProgramShared, Shared.expressionProgram, Shared.assemble_gates]

end Semantic.Circuit
