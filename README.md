# Verifying Horizontally Compressed Natural Deduction Proofs

A Lean 4 formalization of verification for horizontally compressed proofs in purely implicational minimal logic. Horizontal compression (Gordeev and Haeusler) turns a natural deduction proof into a dag-like derivability structure (DLDS), which can be exponentially smaller than the proof. This development defines when a DLDS is valid, reduces that check to local, independently checkable conditions, and applies it to certificates of non-Hamiltonicity and to quantum search oracles.

Start with [ReleaseSurface.lean](ReleaseSurface.lean): it lists every main result as a `#check`.

Author: Lorenzo Saraiva, PUC-Rio. Advisor: Edward Hermann Haeusler.

## Main results

**Validity and three local views.** `Flow d` is the validity predicate for a DLDS `d`. `Flow.main` proves, for every `d`:

- `Flow d` iff `d` is well formed and the grid built on `d` accepts;
- `Flow d` iff some certificate passes every local check, also stated for the executable checker;
- `Flow d` iff `d` is well formed, every root has a reading, and every reading of a root is a closed derivation (a reading is a proof tree unfolded from `d`);
- if `Flow d`, every root formula of `d` is derivable in natural deduction with no open assumptions.

Every closed natural deduction tree, encoded as a graph, satisfies `Flow` (`Flow.tree_accepted`).

**Local checking.** In a valid certificate, the data stored at each vertex is exactly what `Flow` computes there (`MultiIntro.stored_eq_routes`). Checking splits into one check per vertex, one per root and three global checks. A certificate is valid iff none of them fails, and each check reads only its vertex and that vertex's premises (`Flow.extras`).

**Proof search.** A formula is derivable iff some assignment of the search grid is marked, and every marked assignment yields a natural deduction proof (`Flow.search_complete`, `Flow.search_complete_dimensions`).

**Non-Hamiltonicity.** A checked certificate for the compressed proof associated with a graph G rules out a Hamiltonian cycle in G (`MultiIntro.checked_certificate_rules_out_hamiltonian_cycle`). For ten graph classes, `Flow` accepts the compressed proof produced by the pipeline and G has no Hamiltonian cycle: `Flow.Instances.<G>.pipeline_flow` and `Instances.<G>.no_hamiltonian_cycle`, for G in E3, K2_K1, P3, E4, K2_2K1, TwoK2, P3_K1, P4, Star, K3_K1.

**Quantum oracle.** The local checks compile to a reversible circuit generated in Lean. It flips its target exactly when the selected check fails, it is unitary, it acts as I - 2P with P the projection onto failing checks, it marks some check iff the certificate is invalid, and its size is polynomial (`Semantic.Circuit.oracle_correct`, `oracle_xor`, `circuitUnitary_unitary`, `oracle_reflection`, `marked_nonempty_iff_invalid`, `oracle_polynomial_size`).

**Size.** Explicit upper bounds on the number of routes at a vertex and on certificate size (`FlowSize.distinct_route_bound`, `FlowSize.flow_has_bounded_certificate`).

## Trust

General theorems use only Lean's standard axioms (`propext`, `Quot.sound`, `Classical.choice`). The graph-instance theorems also use `native_decide` for their concrete checks. The proof generator and the compressor are not trusted: every conclusion rests on checks stated in the theorems. Running `#print axioms` on any name in `ReleaseSurface.lean` shows exactly what it depends on.

## Scope

The compression theorem of Gordeev and Haeusler and the BBHT quantum search algorithm are cited, not formalized here. 

## Building

Use the Lean version in `lean-toolchain`; Mathlib is pinned in `lakefile.lean`.

```sh
lake exe cache get
lake build
lake env lean ReleaseSurface.lean
```

## Layout

- `Foundations/`: graphs, formulas and natural deduction.
- `Flow/`: the validity predicate and main theorem, with the certificate, readings, grid and size modules.
- `Search/`: proof search.
- `NonHamiltonicity/`: from certificates to non-Hamiltonicity.
- `Instances/`: the ten graph classes.
- `Circuit/`: oracle circuits.
- `Robustness/`: failure densities of defective certificates.
- `Compression/`: compression scheduling and related lemmas.
- `Infrastructure/`: supporting layers used by the proofs.

## References

- Lew Gordeev and Edward Hermann Haeusler, [Proof Compression and NP Versus PSPACE II](https://czasopisma.uni.lodz.pl/bulletin/article/view/8169), Bulletin of the Section of Logic 49 (2020), 213-230.
- Michel Boyer, Gilles Brassard, Peter Høyer and Alain Tapp, [Tight bounds on quantum searching](https://arxiv.org/abs/quant-ph/9605034), Fortschritte der Physik 46 (1998), 493-506.

## License

MIT.