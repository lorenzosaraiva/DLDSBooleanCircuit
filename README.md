# DLDS Boolean Circuits in Lean

This repository contains a Lean 4 formalization of Boolean-circuit evaluation
for dag-like derivability structures (DLDS), together with bridges from DLDS
structure to circuit acceptance. It accompanies the LSFA paper *From Dag-Like
Proofs to Boolean Circuits in Lean* and includes a post-publication extension
for horizontally compressed derivations.

## Published LSFA artifact

The development formalizes a Boolean evaluator over DLDS path assignments and
proves its pointwise and universally quantified correctness specifications. It
also proves an unconditional bridge for the uncompressed simple-tree fragment.

The simple-tree theorem
`tree_bridge_forward` takes a valid simple-tree DLDS and executable route and
discharge certificates, and concludes genuine circuit acceptance of the path
extracted by `pathsFromDLDS`. Genuine acceptance includes both the absence of a
routing error and discharge of the goal, so it cannot hold merely through the
evaluator's neutral treatment of malformed inputs.

This is the scope described by the published LSFA paper.

## Post-publication compressed bridge

The repository now also contains a multi-token, Flow-coloured model for
horizontally compressed DLDSs. The main result is:

```lean
theorem compressed_universally_accepted' (d : Graph)
    (hstruct : structuralValid d)
    (hnodup : d.EDGES.Nodup)
    (honepercol : OneEdgePerColourPerNode d)
    (hroutefan : RouteFanUnique d)
    (hfaithful : FaithfulDecoration d)
    (hcoverage : RouteHeadCoverage d)
    (hlfu : LevelFormulaUnique d)
    (hhypNoInc : HypothesesHaveNoIncoming d)
    (hreseed : ReseedFree d)
    (hftc : FlowTailClosure d)
    (hdis : dischargedMultiB d = true) :
    ∀ P, ¬ AdmissibleMultiPath d P ∨
      DischargedMulti d P (goalColumn d)
```

The proof is conditional on eight invariants attributed to the
horizontal-compression construction:

- `d.EDGES.Nodup`
- `OneEdgePerColourPerNode d`
- `RouteFanUnique d`
- `FaithfulDecoration d`
- `RouteHeadCoverage d`
- `LevelFormulaUnique d`
- `ReseedFree d`
- `FlowTailClosure d`

Their preservation by the horizontal-compression construction is not
formalized in this repository. The remaining hypotheses in the signature are
ordinary structural conditions or the executable discharge certificate.

`FlowTailClosure` expresses partner agreement at elimination kernels: a
residual contributed by either premise must have a matching contribution from
the other premise. The repository includes `exGap2`, which shows that
`FlowTailClosure` does **not** follow from the other local legality conditions.

The compressed theorem and its supporting results do not depend on `sorryAx`.
They use only Lean's standard `propext`, `Classical.choice`, and `Quot.sound`
axioms.

The LSFA paper describes the compressed case as future work. A reader arriving
here from the paper will therefore find a post-publication result beyond the
paper's stated scope. This extension is conditional in the precise sense above;
it is not an end-to-end formalization of invariant preservation by horizontal
compression.

## Open obligations

Three connections remain open:

1. Prove that the horizontal-compression construction preserves the eight
   invariants used by `compressed_universally_accepted'`.
2. Replace the executable hypothesis `dischargedMultiB d = true` with a generic
   evaluator-versus-Flow layer-coherence theorem.
3. Derive the simple-tree bridge as a formal corollary of the compressed
   multi-token theorem. The current simple-tree theorem remains independent.

## Building

Install Lean through `elan`, then run:

```bash
lake build
```

The compatibility entry point is:

```bash
lake env lean DLDSBooleanCircuit.lean
```

## Modules

Core evaluator:

- `Semantic/Core.lean` — circuit types and rule constructors.
- `Semantic/Boolean.lean` — Boolean semantics and exactly-one activation.
- `Semantic/VectorLemmas.lean` — list and fixed-length vector lemmas.
- `Semantic/NodeCorrectness.lean` — local circuit-node correctness.
- `Semantic/Routing.lean` — tokens, path inputs, routing, and structural errors.
- `Semantic/Evaluator.lean` — layer and circuit evaluation theorems.
- `Semantic/DLDS.lean` — DLDS extraction, structural predicates, and evaluator
  wrappers.

Simple-tree bridge:

- `Semantic/TreeBridge.lean` — canonical-path coherence, discharge, and the
  forward bridge.
- `Semantic/UniversalBridge.lean` — admissibility, canonical reduction, and
  universal simple-tree acceptance.
- `Semantic/Examples.lean` — executable simple-tree examples.

Flow and compressed structure:

- `Semantic/FlowModel.lean` — executable Flow semantics from Definition 22.
- `Semantic/FlowValidity.lean` — Flow-based `CorrectRuleApp` and root discharge.
- `Semantic/FlowTreeProof.lean` — Flow correctness on simple trees and general
  Flow induction support.
- `Semantic/FlowCollapsedProof.lean` — collapsed elimination, route-fan
  invariants, and `ReseedFree`.
- `Semantic/FlowEdgeDepChar.lean` — route-edge dependency characterization and
  local collapsed Flow correctness.
- `Semantic/CompressedBridge.lean` — level/formula uniqueness and compressed
  lookup primitives.
- `Semantic/CompressedRouting.lean` — Flow-coloured routing primitives,
  the one-token limitation, and compressed fixtures.

Multi-token compressed bridge:

- `Semantic/MultiTokenModel.lean` — one-token-per-route evaluator.
- `Semantic/MultiTokenAdmissible.lean` — multi-path admissibility.
- `Semantic/MultiTokenReduction.lean` — permutation congruence and reduction to
  the canonical multi-path.
- `Semantic/MultiTokenBridge.lean` — certificate-based compressed universal and
  genuine-acceptance theorems.
- `Semantic/FlowClosure.lean` — `FlowTailClosure`, `canonicalSlotOK`, and the
  strengthened compressed bridge.
- `Semantic/FlowDischarge.lean` — executable discharge-certificate boundary and
  examples.

Infrastructure and entry points:

- `HorizontalCompressionEXEC.lean` — graph and horizontal-compression
  infrastructure.
- `Semantic.lean` — aggregate library entry point.
- `DLDSBooleanCircuit.lean` — compatibility entry point.

## Main theorems and axioms

| Theorem | Axioms reported by `#print axioms` |
|---|---|
| `node_correct` | `propext`, `Quot.sound` |
| `circuit_correctness` | `propext`, `Quot.sound` |
| `circuit_completeness` | `propext`, `Quot.sound` |
| `circuit_iff` | `propext`, `Quot.sound` |
| `dlds_evaluation_correct` | `propext`, `Quot.sound` |
| `dlds_evaluation_complete` | `propext`, `Quot.sound` |
| `dlds_evaluation_iff` | `propext`, `Quot.sound` |
| `dlds_global_soundness` | `propext`, `Quot.sound` |
| `dlds_global_completeness` | `propext`, `Quot.sound` |
| `dlds_global_iff` | `propext`, `Quot.sound` |
| `tree_bridge_forward` | `propext`, `Classical.choice`, `Quot.sound` |
| `simpleTree_universally_accepted` | `propext`, `Classical.choice`, `Quot.sound` |
| `flowRuleCorrect_of_simpleTree` | `propext`, `Classical.choice`, `Quot.sound` |
| `flowRuleCorrect_collapsed` | `propext`, `Classical.choice`, `Quot.sound` |
| `admissibleMultiReducesToCanonical` | `propext`, `Classical.choice`, `Quot.sound` |
| `compressed_universally_accepted` | `propext`, `Classical.choice`, `Quot.sound` |
| `flowAt_tail_closure` | `propext`, `Classical.choice`, `Quot.sound` |
| `canonicalSlotOK` | `propext`, `Classical.choice`, `Quot.sound` |
| `compressed_universally_accepted'` | `propext`, `Classical.choice`, `Quot.sound` |

The executable example theorems in `Semantic/Examples.lean` use
`native_decide` and consequently have theorem-local native-code axioms. These
axioms do not occur in the general results listed above.

## License

This project is released under the MIT License. See `LICENSE`.
