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
   git clone https://github.com/<your-username>/DLDSBooleanCircuit.git
   cd DLDSBooleanCircuit
3. Open DLDSBooleanCircuit.lean in VS Code with the Lean4 extension, or build it with:
   lake build

### 3. Repository contents
DLDSBooleanCircuit.lean – main Lean development, including:
- Definition of circuit nodes and activation rules
- Layer-by-layer grid evaluation with error detection
- Goal node semantics
- Global soundness theorem (circuit correctness)

### 4. Reference
This code accompanies the paper:

Lorenzo Saraiva and Hermann Haeusler
"From Dag-Like Proofs to Boolean Circuits in Lean"
submitted to FoIKS 2026.



   git clone https://github.com/<your-username>/DLDSBooleanCircuit.git
   cd DLDSBooleanCircuit
