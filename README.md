# DLDS Boolean Circuit in Lean

This repository contains the Lean 4 formalization accompanying the article
"From Dag-Like Proofs to Boolean Circuits in Lean".

The development proves correctness theorems for a Boolean circuit evaluator over
DLDS path assignments. It also defines DLDS-side structural predicates and a
restricted simple-tree bridge from valid DLDS inputs to genuine circuit
acceptance. The full compressed Flow bridge for collapsed nodes, ancestor edges,
colour fan-out, and lambda-labelled edges is outside this artifact.

## Building

Install Lean through `elan`, then run:

```bash
lake build
```

The compatibility entrypoint is:

```bash
lake env lean DLDSBooleanCircuit.lean
```

## Module Layout

- `Semantic/Core.lean`: circuit data types and rule constructors.
- `Semantic/Boolean.lean`: Boolean semantics and exactly-one activation logic.
- `Semantic/VectorLemmas.lean`: list and vector support lemmas.
- `Semantic/NodeCorrectness.lean`: node-level correctness.
- `Semantic/Routing.lean`: tokens, path inputs, labels, routing, and node errors.
- `Semantic/Evaluator.lean`: layer and circuit evaluation theorems.
- `Semantic/DLDS.lean`: DLDS extraction, validity predicates, and helpers.
- `Semantic/TreeBridge.lean`: the simple-tree bridge.
- `Semantic/Examples.lean`: example DLDSs and executable witnesses.
- `Semantic.lean`: public aggregate for the core development.

`HorizontalCompressionEXEC.lean` contains the graph and horizontal compression
infrastructure used by the DLDS layer.

## Main Theorems

The article-facing theorem names are preserved, including:

- `node_correct`
- `circuit_correctness`
- `circuit_iff`
- `dlds_evaluation_correct`
- `dlds_evaluation_iff`
- `dlds_global_soundness`
- `dlds_global_completeness`
- `dlds_global_iff`
- `tree_bridge_forward`
- `tree_bridge_forward_of_descent_coherent`

The predicate `GenuinelyAccepts` rules out neutral acceptance of structurally
invalid paths and is the predicate used by the simple-tree bridge.

## License

This project is released under the MIT License. See `LICENSE`.
