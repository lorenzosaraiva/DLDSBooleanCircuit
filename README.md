# DLDS: verifying horizontally compressed proofs in Lean

Lean 4 formalization accompanying my PhD work at PUC-Rio, advised by Edward Hermann Haeusler, on checking compressed Natural Deduction proofs.

Natural Deduction proofs in purely implicational minimal logic can be exponentially large. Horizontal compression (Gordeev and Haeusler) merges repeated formula occurrences into a Dag-Like Derivability Structure (DLDS) of polynomial size, whose validity is decided by Flow, a recursive computation over the whole graph. This development records Flow's routes as a certificate whose correctness reduces to local checks, and studies the verification and search problems that follow: a Boolean grid for proof search, readings of a fixed DLDS, and verification positions of a certificate. The motivating application is certifying that a graph has no Hamiltonian cycle.

## Main results

The results below hold under the assumptions stated in the linked declarations.

### Boolean evaluator

| Theorem | Meaning |
| --- | --- |
| [node_correct](Evaluator/CellCorrectness.lean) | When exactly one rule is active, the Boolean node computes that rule's dependency update. |
| [dlds_evaluation_iff](Evaluator/Grid.lean) | A path evaluates to true exactly when it is malformed, or routes correctly and leaves no open assumptions at the goal. |
| [dlds_global_iff](Evaluator/Grid.lean) | All path assignments are accepted exactly when each is malformed or routes correctly with no open goal assumptions. |
| [tree_bridge_forward](Evaluator/TreeCorrespondence.lean) | A valid proof tree passes the evaluator when its routing and discharge checks pass. |

### Grid proof search

| Theorem | Meaning |
| --- | --- |
| [gridMarked_extracts](ProofSearch/Grid/Soundness.lean) | A marked assignment yields a well-formed derivation of the goal with no open assumptions, within the grid's height bound. |
| [gridMarked_sound](ProofSearch/Grid/Soundness.lean) | A marked grid assignment proves the goal without assumptions, when the formula list has no repeats and contains the goal. |
| [exists_marked_iff_exists_closed_gridSchedulable](ProofSearch/Grid/Schedulability.lean) | With a distinct formula list containing the goal, search succeeds exactly when a closed derivation can be represented within the grid's height bound. |
| [branchingGridMarked_sound](ProofSearch/BranchingGrid/Soundness.lean) | A marked branching-grid assignment proves the goal without assumptions, when the formula list has no repeats and contains the goal. |
| [branchingGridMarked_complete](ProofSearch/BranchingGrid/Completeness.lean) | Every well-formed closed derivation whose formulas are in the chosen list has a successful branching-grid encoding with sufficient space. |

### Readings

| Theorem | Meaning |
| --- | --- |
| [labelDefectB_sound](Readings/Explicit/LabelValidity.lean) | Every marked reading exposes an incorrect rule or dependency label in the graph. |
| [explicitOutputLabelsCorrect_iff_no_labelDefect](Readings/Explicit/LabelValidity.lean) | If permitted readings can select every output, correct output labels are equivalent to no reading being marked. |
| [badCylinder_raw_density](Readings/Explicit/FanFamily.lean) | A proved set of failing readings occupies exactly 1 / (width + 2) of the choices in the constructed branching family. |

### Certificates

| Theorem | Meaning |
| --- | --- |
| [storedRoutes_setEq_flow](Certificate/Flow/Soundness.lean) | The local checks guarantee equality, ignoring order and repeats, between stored routes and the usable routes computed by Flow. |
| [compressed_universally_accepted_certificateB](Certificate/Flow/Soundness.lean) | If the certificate, routing and discharge checks pass, every permitted reading ends with all goal assumptions discharged. |
| [formationValid_iff_no_semanticMarked](Certificate/Flow/Positions.lean) | A certificate satisfies its formation conditions exactly when none of its verification positions reports a failure. |
| [compressed_universally_accepted'](Flow/UniversalDischarge.lean) | Under the stated structural and discharge assumptions, every permitted reading ends with all goal assumptions discharged. |

### Non-Hamiltonicity

| Theorem | Meaning |
| --- | --- |
| [checked_request_sound](NonHamiltonicity/Certificate/Input.lean) | An input accepted by the certificate checker certifies that its source graph has no Hamiltonian cycle. |
| [supplied_graph_no_hamiltonian_cycle](Examples/FourVertex/Occurrence/NonHamiltonicity.lean) | The supplied certificate proves that the four-vertex star has no Hamiltonian cycle. |

### Examples

| Theorem | Meaning |
| --- | --- |
| [closed_3v_self_certifying](Examples/ThreeVertex/Flow/Acceptance.lean) | Every permitted reading of the closed three-vertex compressed proof ends with all goal assumptions discharged. |
| [closed_4v_self_certifying](Examples/FourVertex/Flow/EndToEnd.lean) | Every permitted reading of the closed four-vertex compressed proof ends with all goal assumptions discharged. |

### Robustness

| Theorem | Meaning |
| --- | --- |
| [readOrbit_card](ProofSearch/Grid/Counting.lean) | Counts exactly how many assignments agree on all cells read by a given evaluation. |
| [encodedDerivation_density](ProofSearch/Grid/Counting.lean) | A certified successful assignment gives a lower bound on the fraction of assignments for which grid search succeeds. |
| [formation_invalid_semantic_amplification](Certificate/Encoding/Canonical.lean) | Every invalid certificate has a failing check that persists on at least a 1 / A^r fraction of encodings, where A counts field choices and r bounds fields read. |
| [missingRoute_minimumFailureWitnessSize](Robustness/AbsenceBoundary.lean) | In the abstract list model, forcing a route to remain absent requires fixing all n list entries. |

## Building

Pinned toolchain `leanprover/lean4:v4.29.0-rc7` and Mathlib (see `lake-manifest.json`).

```sh
lake exe cache get
lake build
lake env lean Checks.lean   # prints the axioms of the main results
```

## Trust

The general theorems depend only on `propext`, `Classical.choice` and `Quot.sound`. Concrete examples and regression tests use `native_decide`, which also trusts the Lean compiler; `Checks.lean` shows exactly which results depend on it.

<details>
<summary>Modules containing native_decide</summary>

- [Certificate.Encoding.Canonical](Certificate/Encoding/Canonical.lean)
- [Certificate.Flow.Soundness](Certificate/Flow/Soundness.lean)
- [DLDS.Compression.Domain](DLDS/Compression/Domain.lean)
- [DLDS.ValidityDecomposition](DLDS/ValidityDecomposition.lean)
- [Examples.Circuit](Examples/Circuit.lean)
- [Examples.Flow.ClosedImplication](Examples/Flow/ClosedImplication.lean)
- [Examples.FourVertex.Flow.CachedDischarge](Examples/FourVertex/Flow/CachedDischarge.lean)
- [Examples.FourVertex.Flow.CertificateDecision](Examples/FourVertex/Flow/CertificateDecision.lean)
- [Examples.FourVertex.Flow.Closed](Examples/FourVertex/Flow/Closed.lean)
- [Examples.FourVertex.Flow.Slots](Examples/FourVertex/Flow/Slots.lean)
- [Examples.FourVertex.Occurrence.NonHamiltonicity](Examples/FourVertex/Occurrence/NonHamiltonicity.lean)
- [Examples.ThreeVertex.Flow.Acceptance](Examples/ThreeVertex/Flow/Acceptance.lean)
- [Examples.ThreeVertex.Flow.Certificate](Examples/ThreeVertex/Flow/Certificate.lean)
- [ProofSearch.Grid.Completeness](ProofSearch/Grid/Completeness.lean)
- [ProofSearch.Grid.Counting](ProofSearch/Grid/Counting.lean)
- [ProofSearch.Grid.Schedulability](ProofSearch/Grid/Schedulability.lean)
- [ProofSearch.Grid.Soundness](ProofSearch/Grid/Soundness.lean)
- [Readings.Explicit.LabelValidity](Readings/Explicit/LabelValidity.lean)
- [Readings.FailurePersistence](Readings/FailurePersistence.lean)
- [Robustness.Certificate.Support](Robustness/Certificate/Support.lean)
- [Search.Certificate.Definitions](Search/Certificate/Definitions.lean)

</details>

## References

- L. Gordeev and E. H. Haeusler. *Proof Compression and NP Versus PSPACE II: Addendum.* Bulletin of the Section of Logic 51(2), 2022.
- L. Saraiva and E. H. Haeusler. *From Dag-Like Proofs to Boolean Circuits in Lean.* Proceedings of LSFA 2026, EPTCS, 2026.
- L. Saraiva and E. H. Haeusler. *Quantum Verification of Compressed Proofs through DLDS-to-Circuit Compilation*, Anais do Simpósio Brasileiro de Computação e Comunicação Quântico, 2026

## License

MIT; see [LICENSE](LICENSE).