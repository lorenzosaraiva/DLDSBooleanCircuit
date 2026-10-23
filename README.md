# DLDS and unified Flow

Lean 4 formalization of verification and proof search for dag-like derivability structures in purely implicational minimal logic.

## What is proved

`Flow` is `MultiIntro.ValidMI .general`: it applies dependency-label tests to ordinary inferences and handles vertices that merge several introductions. `Flow.of_reference` embeds the single-introduction reference verdict, and `Flow.iff_reference_of_no_multi` states their equivalence when multiple introductions are absent.

`Flow.main` relates Flow to the grid, complete-profile certificates, occurrence readings and closed natural deduction. `Flow.extras` packages verification positions, robustness and proof-search consequences, with the hypotheses shown in its statement. `Flow.tree_accepted` accepts the represented closed proof trees. `Flow.search_complete` and `Flow.search_complete_dimensions` establish sound and complete proof search, with explicit finite dimensions.

The ten instances are E3, K2_K1, P3, E4, K2_2K1, TwoK2, P3_K1, P4, Star and K3_K1. Each has a `Flow.Instances.<name>.pipeline_flow` theorem and an `Instances.<name>.no_hamiltonian_cycle` theorem. Their concrete checks certify these inputs; they do not establish compressor completeness.

`MultiIntro.stored_eq_routes` identifies checked stored profiles with computed routes. `MultiIntro.checked_certificate_rules_out_hamiltonian_cycle` connects a checked certificate and an admissible source context to non-Hamiltonicity. `Semantic.Circuit` proves oracle correctness, XOR behavior, unitarity, the phase action `I - 2P`, reflection and polynomial circuit size. `FlowSize.distinct_route_bound`, `raw_route_bound`, `exists_bounded_certificate` and `flow_has_bounded_certificate` give route and certificate upper bounds in the stated structural size model.

## Trust

The general headline theorems use only `propext`, `Quot.sound` and `Classical.choice`. The generator and compressor are untrusted: the logical conclusions require the checked conditions in the theorem statements. Concrete `native_decide` proofs additionally trust Lean's native computation mechanism. They appear only in the following listed modules, including the small concrete examples retained inside required mixed foundation modules:

- [Semantic.Examples](Semantic/Examples.lean)
- [Semantic.EndToEnd](Semantic/EndToEnd.lean)
- [Semantic.SelfCertifying](Semantic/SelfCertifying.lean)
- [Semantic.GridCertificateSoundness](Semantic/GridCertificateSoundness.lean)
- [SpaceACounting](SpaceACounting.lean)
- [Semantic.ValidDLDSDecomposition](Semantic/ValidDLDSDecomposition.lean)
- [Semantic.ProperHCInput](Semantic/ProperHCInput.lean)
- [SearchProblem](SearchProblem.lean)
- [CertificateRobustness](CertificateRobustness.lean)
- [CanonicalCertificateEncoding](CanonicalCertificateEncoding.lean)
- [SpaceA](SpaceA.lean)
- [Instances.E3.PipelineEquality](Instances/E3/PipelineEquality.lean)
- [Instances.E3.Formation](Instances/E3/Formation.lean)
- [Instances.E3.Proof](Instances/E3/Proof.lean)
- [Instances.K2_K1.PipelineEquality](Instances/K2_K1/PipelineEquality.lean)
- [Instances.K2_K1.Formation](Instances/K2_K1/Formation.lean)
- [Instances.K2_K1.Proof](Instances/K2_K1/Proof.lean)
- [Instances.P3.PipelineEquality](Instances/P3/PipelineEquality.lean)
- [Instances.P3.Formation](Instances/P3/Formation.lean)
- [Instances.P3.Proof](Instances/P3/Proof.lean)
- [Instances.E4.PipelineEquality](Instances/E4/PipelineEquality.lean)
- [Instances.E4.Formation](Instances/E4/Formation.lean)
- [Instances.E4.Proof](Instances/E4/Proof.lean)
- [Instances.K2_2K1.PipelineEquality](Instances/K2_2K1/PipelineEquality.lean)
- [Instances.K2_2K1.Formation](Instances/K2_2K1/Formation.lean)
- [Instances.K2_2K1.Proof](Instances/K2_2K1/Proof.lean)
- [Instances.TwoK2.PipelineEquality](Instances/TwoK2/PipelineEquality.lean)
- [Instances.TwoK2.Formation](Instances/TwoK2/Formation.lean)
- [Instances.TwoK2.Proof](Instances/TwoK2/Proof.lean)
- [Instances.P3_K1.PipelineEquality](Instances/P3_K1/PipelineEquality.lean)
- [Instances.P3_K1.Formation](Instances/P3_K1/Formation.lean)
- [Instances.P3_K1.Proof](Instances/P3_K1/Proof.lean)
- [Instances.P4.PipelineEquality](Instances/P4/PipelineEquality.lean)
- [Instances.P4.Formation](Instances/P4/Formation.lean)
- [Instances.P4.Proof](Instances/P4/Proof.lean)
- [Instances.Star.PipelineEquality](Instances/Star/PipelineEquality.lean)
- [Instances.Star.Formation](Instances/Star/Formation.lean)
- [Instances.Star.Proof](Instances/Star/Proof.lean)
- [Instances.K3_K1.Proof](Instances/K3_K1/Proof.lean)
- [Instances.K3_K1.PipelineEquality](Instances/K3_K1/PipelineEquality.lean)

Only the fixture-local native checks listed by the statement surface enter the twenty instance headlines. The general headlines do not depend on any native-check axiom. Historical reference infrastructure remains where required by imports; its propositions are separate from the unified verdict.

## Scope and references

The horizontal compression theorem of Gordeev and Haeusler is cited, not mechanized here. Completeness of the executable compressor is not proved. The BBHT search algorithm is cited; an implementation or quantum runtime bound for this development is not claimed.

References: Lew Gordeev and Edward Hermann Haeusler, [Proof Compression and NP Versus PSPACE II](https://czasopisma.uni.lodz.pl/bulletin/article/view/8169), Bulletin of the Section of Logic 49 (2020), 213-230; Michel Boyer, Gilles Brassard, Peter Høyer and Alain Tapp, [Tight bounds on quantum searching](https://arxiv.org/abs/quant-ph/9605034), Fortschritte der Physik 46 (1998), 493-506.

## Build and statement surface

Install the version in `lean-toolchain`, then run:

```sh
lake exe cache get
lake build
lake env lean Semantic/ReleaseSurface.lean
```

The default `Release` library builds exactly the import closure of `Semantic.ReleaseSurface`. [Semantic/ReleaseSurface.lean](Semantic/ReleaseSurface.lean) is the statement surface: it contains imports and explicit `#check` commands, with no definitions. Mathlib is pinned in `lakefile.lean`; Lake recreates its local dependency manifest as needed. Mathematical hypotheses remain explicit in each declaration.

## History

Earlier versions used a more permissive semantics that skipped dependency-label tests on ordinary inferences; it is superseded by Flow, which applies every label test and handles vertices that merge several introductions. The snapshot cited by the LSFA 2026 paper is commit a07f0c2065fa74017d96c0937f1b92b6324731eb.
