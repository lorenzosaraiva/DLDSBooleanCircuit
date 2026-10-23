import Flow.MultiIntroDAG
import Flow.MultiIntroSoundness
import Infrastructure.RouteCertificateCompleteness
import NonHamiltonicity.HamiltonianCertificateToGraph



open Semantic FlowSpec CertificateModel RouteCertificate GridCertificateSoundness
namespace MultiIntro
set_option maxHeartbeats 3000000

def SuitableGraph (d : Graph) : Prop := MIDomain d ∧ LeveledColored d

def LocalChecks (variant : Variant) (d : Graph) (c : CompleteProfileCertificate) : Prop :=
  SuitableGraph d ∧ ∀ v ∈ d.NODES, stored c v = localExpectedFor (kernel variant) c v

def StrongCertificate (variant : Variant) (d : Graph) (c : CompleteProfileCertificate) : Prop :=
  ExpandedCertComplete d c ∧ LocalChecks variant d c ∧ UniversalRootDischargeComplete d c

def strongCertificateB (variant : Variant) (d : Graph) (c : CompleteProfileCertificate) : Bool :=
  expandedCertCompleteB d c && miDomainB d &&
    (d.NODES.all fun v => decide (stored c v = localExpectedFor (kernel variant) c v)) &&
    universalRootDischargeCompleteB d c

def generateCertificate (variant : Variant) (d : Graph) : CompleteProfileCertificate :=
  generateFor (kernel variant) d

theorem strongCertificateB_sound (variant : Variant) (d : Graph) (c : CompleteProfileCertificate)
    (checked : strongCertificateB variant d c = true) : StrongCertificate variant d c := by
  obtain ⟨⟨⟨expanded, domain⟩, localRows⟩, roots⟩ :=
    (by simpa only [strongCertificateB, Bool.and_eq_true] using checked)
  refine ⟨(expandedCertCompleteB_iff d c).mp expanded,
    ⟨⟨domain, domain_levels d domain⟩, ?_⟩,
    (universalRootDischargeCompleteB_iff d c).mp roots⟩
  simpa only [List.all_eq_true, decide_eq_true_eq] using localRows

theorem stored_eq_routes (variant : Variant) (d : Graph) (c : CompleteProfileCertificate)
    (valid : StrongCertificate variant d c) (v : Vertex) (listed : v ∈ d.NODES) :
    stored c v = evaluateAt variant d (stdFuel d) v := by
  rw [← kernel_evaluate]
  apply kernel_fixed_eq_evaluate (kernel variant) d valid.2.1.1.1 (stored c)
  · intro u member
    simpa only [localExpectedFor, valid.1.1] using valid.2.1.2 u member
  · exact listed
  · have count := countAbove_le d v
    unfold stdFuel
    omega

theorem formation_valid (variant : Variant) (d : Graph) (c : CompleteProfileCertificate)
    (valid : StrongCertificate variant d c) : ValidMI variant d := by
  refine ⟨valid.2.1.1.1, valid.2.1.1.2, (rootsB_iff variant d).mpr ?_⟩
  intro root listed terminal
  rw [← stored_eq_routes variant d c valid root listed]
  exact valid.2.2 root listed terminal

theorem generated_stored (variant : Variant) (d : Graph) (domain : MIDomain d)
    (v : Vertex) (listed : v ∈ d.NODES) :
    stored (generateCertificate variant d) v = evaluateAt variant d (stdFuel d) v := by
  simp only [stored, generateCertificate, generateFor,
    kernel_memo_eq_spec (kernel variant) d domain]
  rw [CompleteProfileKernel.lookup_spec (kernel variant) d (stdFuel d) listed, kernel_evaluate]

theorem generated_formation (variant : Variant) (d : Graph) (valid : ValidMI variant d) :
    StrongCertificate variant d (generateCertificate variant d) := by
  refine ⟨⟨rfl, ?_⟩, ⟨⟨valid.1,valid.2.1⟩, ?_⟩, ?_⟩
  · simp [generateCertificate, generateFor,
      kernel_memo_eq_spec (kernel variant) d valid.1, CompleteProfileKernel.specificationTable,
      List.map_map, Function.comp_def]
  · intro v listed
    rw [generated_stored variant d valid.1 v listed]
    change evaluateAt variant d (stdFuel d) v =
      (kernel variant).step d (stored (generateCertificate variant d)) v
    rw [(kernel variant).congr d v (stored (generateCertificate variant d))
      (CompleteProfileKernel.evaluateAt (kernel variant) d (stdFuel d)) (by
        intro u member
        rw [generated_stored variant d valid.1 u (predecessor_props d valid.1 member).1,
          kernel_evaluate])]
    simpa only [kernel_evaluate] using
      (kernel_standard_fixed (kernel variant) d valid.1 v listed).symm
  · intro root listed terminal
    rw [generated_stored variant d valid.1 root listed]
    exact (rootsB_iff variant d).mp valid.2.2 root listed terminal

theorem valid_iff_exists_certificate (variant : Variant) (d : Graph) :
    ValidMI variant d ↔ ∃ c, StrongCertificate variant d c := by
  constructor
  · intro valid
    exact ⟨generateCertificate variant d, generated_formation variant d valid⟩
  · rintro ⟨c, formed⟩
    exact formation_valid variant d c formed

theorem valid_iff_generated_certificate (variant : Variant) (d : Graph) :
    ValidMI variant d ↔ StrongCertificate variant d (generateCertificate variant d) :=
  ⟨generated_formation variant d, formation_valid variant d (generateCertificate variant d)⟩

theorem certificate_rules_out_hamiltonian_cycle (variant : Variant)
    (sourceGraph : HamiltonianContext.Graph) (names : HamiltonianContext.Names)
    (context : List Formula)
    (admitted : HamiltonianContext.ContextAdmissible sourceGraph names context)
    (d : _root_.Graph) (c : CompleteProfileCertificate) (valid : StrongCertificate variant d c)
    (root : Vertex) (listed : root ∈ d.NODES)
    (terminal : (get_rule.outgoing root d).isEmpty = true)
    (target : root.FORMULA = HamiltonianContext.chain context (HamiltonianContext.q names)) :
    ¬ HamiltonianContext.HasHamiltonianCycle sourceGraph := by
  apply HamiltonianCertificateToGraph.closed_curried_derivation_rules_out_cycle
    sourceGraph names context admitted
  simpa only [target] using valid_roots_derivable variant d
    (formation_valid variant d c valid) root listed terminal

theorem checked_certificate_rules_out_hamiltonian_cycle (variant : Variant)
    (sourceGraph : HamiltonianContext.Graph) (names : HamiltonianContext.Names)
    (context : List Formula)
    (admitted : HamiltonianContext.ContextAdmissible sourceGraph names context)
    (d : _root_.Graph) (c : CompleteProfileCertificate) (checked : strongCertificateB variant d c = true)
    (root : Vertex) (listed : root ∈ d.NODES)
    (terminal : (get_rule.outgoing root d).isEmpty = true)
    (target : root.FORMULA = HamiltonianContext.chain context (HamiltonianContext.q names)) :
    ¬ HamiltonianContext.HasHamiltonianCycle sourceGraph :=
  certificate_rules_out_hamiltonian_cycle variant sourceGraph names context admitted d c
    (strongCertificateB_sound variant d c checked) root listed terminal target

end MultiIntro
