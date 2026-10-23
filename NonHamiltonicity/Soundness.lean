import Infrastructure.SpineNDSoundness
import NonHamiltonicity.HamiltonianCertificateToGraph

open Semantic GridCertificateSoundness

namespace NonHamiltonicity




theorem certificate_rules_out_hamiltonian_cycle
    (sourceGraph : HamiltonianContext.Graph) (names : HamiltonianContext.Names)
    (context : List Formula)
    (admitted : HamiltonianContext.ContextAdmissible sourceGraph names context)
    (proofGraph : _root_.Graph) (certificate : OccurrenceCertificate.OccurrenceCertificate)
    (formation : OccurrenceCertificate.StrongFormationValidOccurrence proofGraph certificate)
    (root : Vertex) (rootMember : root ∈ proofGraph.NODES)
    (terminal : (get_rule.outgoing root proofGraph).isEmpty = true)
    (target : root.FORMULA = HamiltonianContext.chain context (HamiltonianContext.q names)) :
    ¬HamiltonianContext.HasHamiltonianCycle sourceGraph := by
  apply HamiltonianCertificateToGraph.closed_curried_derivation_rules_out_cycle
    sourceGraph names context admitted
  simpa only [target] using
    SpineND.strongFormationValidOccurrence_roots_derivable proofGraph certificate formation
      root rootMember terminal


theorem checked_certificate_rules_out_hamiltonian_cycle
    (sourceGraph : HamiltonianContext.Graph) (names : HamiltonianContext.Names)
    (context : List Formula)
    (admitted : HamiltonianContext.ContextAdmissible sourceGraph names context)
    (proofGraph : _root_.Graph) (certificate : OccurrenceCertificate.OccurrenceCertificate)
    (formation : OccurrenceCertificate.strongFormationValidOccurrenceB proofGraph certificate = true)
    (root : Vertex) (rootMember : root ∈ proofGraph.NODES)
    (terminal : (get_rule.outgoing root proofGraph).isEmpty = true)
    (target : root.FORMULA = HamiltonianContext.chain context (HamiltonianContext.q names)) :
    ¬HamiltonianContext.HasHamiltonianCycle sourceGraph :=
  certificate_rules_out_hamiltonian_cycle sourceGraph names context admitted
    proofGraph certificate (OccurrenceCertificate.strongFormationValidOccurrenceB_sound _ _ formation)
    root rootMember terminal target


theorem chain_context_injective (tail : String) :
    Function.Injective (fun context => HamiltonianContext.chain context (Formula.atom tail)) := by
  intro left
  induction left with
  | nil => intro right same; cases right <;> simp_all [HamiltonianContext.chain]
  | cons head rest ih =>
      intro right same
      cases right with
      | nil => simp [HamiltonianContext.chain] at same
      | cons other others =>
          have parts := Formula.implication.inj same
          exact congrArg₂ List.cons parts.1 (ih parts.2)

#print axioms certificate_rules_out_hamiltonian_cycle
#print axioms checked_certificate_rules_out_hamiltonian_cycle
#print axioms chain_context_injective
end NonHamiltonicity
