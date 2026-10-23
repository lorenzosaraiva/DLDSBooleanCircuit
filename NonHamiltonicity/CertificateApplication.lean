import Infrastructure.RouteCertificateSoundness
import NonHamiltonicity.HamiltonianCertificateToGraph

open Semantic GridCertificateSoundness

namespace CertificateApplication



theorem certificate_rules_out_hamiltonian_cycle
    (sourceGraph : HamiltonianContext.Graph) (names : HamiltonianContext.Names)
    (context : List Formula)
    (admitted : HamiltonianContext.ContextAdmissible sourceGraph names context)
    (proofGraph : _root_.Graph) (certificate : RouteCertificate.CompleteProfileCertificate)
    (formation : RouteCertificate.StrongFormationValidComplete proofGraph certificate)
    (root : Vertex) (rootMember : root ∈ proofGraph.NODES)
    (terminal : (get_rule.outgoing root proofGraph).isEmpty = true)
    (target : root.FORMULA = HamiltonianContext.chain context (HamiltonianContext.q names)) :
    ¬HamiltonianContext.HasHamiltonianCycle sourceGraph := by
  apply HamiltonianCertificateToGraph.closed_curried_derivation_rules_out_cycle
    sourceGraph names context admitted
  simpa only [target] using
    RouteCertificate.strongFormationValidComplete_roots_derivable proofGraph certificate formation
      root rootMember terminal

theorem checked_certificate_rules_out_hamiltonian_cycle
    (sourceGraph : HamiltonianContext.Graph) (names : HamiltonianContext.Names)
    (context : List Formula)
    (admitted : HamiltonianContext.ContextAdmissible sourceGraph names context)
    (proofGraph : _root_.Graph) (certificate : RouteCertificate.CompleteProfileCertificate)
    (formation : RouteCertificate.strongFormationValidCompleteB proofGraph certificate = true)
    (root : Vertex) (rootMember : root ∈ proofGraph.NODES)
    (terminal : (get_rule.outgoing root proofGraph).isEmpty = true)
    (target : root.FORMULA = HamiltonianContext.chain context (HamiltonianContext.q names)) :
    ¬HamiltonianContext.HasHamiltonianCycle sourceGraph :=
  certificate_rules_out_hamiltonian_cycle sourceGraph names context admitted
    proofGraph certificate (RouteCertificate.strongFormationValidCompleteB_sound _ _ formation)
    root rootMember terminal target

#print axioms certificate_rules_out_hamiltonian_cycle
#print axioms checked_certificate_rules_out_hamiltonian_cycle
end CertificateApplication
