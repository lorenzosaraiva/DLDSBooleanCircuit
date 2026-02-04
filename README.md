# DLDS Boolean Circuit in Lean

This repository contains the Lean 4 formalization accompanying the paper  
**"From Dag-Like Proofs to Boolean Circuits in Lean" (FoIKS 2026 submission)**.  

The development proves the correctness of a Boolean circuit construction
that encodes horizontally compressed Natural Deduction proofs
(Dag-Like Derivability Structures, DLDS) in purely implicational minimal logic.
It provides a formally certified bridge between proof compression
and circuit-level verification.

---

## Running the Code

You can explore the Lean development in two ways:

### 1. Without installation (recommended for quick use)

Open the file directly in the [Lean web editor](https://live.lean-lang.org/)  
and paste the contents of [`DLDSBooleanCircuit.lean`](DLDSBooleanCircuit.lean).  
This allows you to experiment with the definitions without setting up Lean locally.

### 2. With a local Lean installation

1. Install [Lean 4](https://lean-lang.org/) (version ≥ 4.8 recommended).  
2. Clone this repository:
   git clone https://github.com/lorenzosaraiva/DLDSBooleanCircuit.git
   cd DLDSBooleanCircuit
3. Open DLDSBooleanCircuit.lean in VS Code with the Lean4 extension, or build it with:
   lake build
   
### 3. Run in GitHub Codespaces (no local install)

Launch a ready-to-use cloud dev environment:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/<YOUR_GITHUB_USERNAME>/<YOUR_REPO>?quickstart=1)

This opens VS Code in the browser with Lean 4 and the Lean extension preinstalled.
No local setup required.

> Devcontainer setup adapted from
> [jeffsantos/lean-project-template](https://github.com/jeffsantos/lean-project-template).


## Repository Contents

**`DLDSBooleanCircuit.lean`** (~1400 lines) – Complete Lean 4 formalization with:

- **Core circuit types** (Section 1): Rules, activation bits, circuit nodes with XOR-based conflict detection
- **Boolean circuit logic** (Section 2): `multiple_xor` exactly-one-true checker, node evaluation functions
- **Correctness proofs** (Sections 3-5): 
  - `multiple_xor_bool_iff_exactlyOneActive` – XOR ↔ exactly-one-active equivalence
  - `node_correct` – Single-node correctness theorem
- **Path-based evaluation** (Section 6): Token propagation, routing-aware node logic, layer evaluation
  - `circuit_correctness` – **Main soundness theorem**: circuit acceptance implies structural error OR valid proof with discharged assumptions
- **DLDS grid construction** (Section 7): Formula universe extraction, encoder generation, grid builders
  - `dlds_evaluation_correct` – **End-to-end correctness**: combines grid construction + circuit evaluation
- **Test cases**: Identity combinator, hypothetical syllogism, incomplete proof (undischarged assumptions)

**Key theorem**:
```lean
theorem circuit_correctness : 
  evaluateCircuit = true → 
    PathStructurallyInvalid ∨ 
    (PathRepresentsValidProof ∧ AllAssumptionsDischarged)
```

### 5. Reference

This code accompanies the paper:

Lorenzo Saraiva and Hermann Haeusler
"From Dag-Like Proofs to Boolean Circuits in Lean"
submitted to FoIKS 2026.

## 6. License

This project is released under the MIT License.  
See the [LICENSE](LICENSE) file for details.
