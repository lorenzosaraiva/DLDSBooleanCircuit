import Infrastructure.RouteCertificateModel
import Infrastructure.RouteSoundness

open Semantic FlowSpec CertificateModel GridCertificateSoundness

namespace RouteCertificate

theorem stored_eq_evaluate (kernel : CompleteProfileKernel.Kernel) (d : Graph) (c : CompleteProfileCertificate)
    (valid : StrongFormationValidFor kernel d c) (v : Vertex) (member : v ∈ d.NODES) :
    stored c v = CompleteProfileKernel.evaluateAt kernel d (stdFuel d) v := by
  apply CompleteProfileKernel.fixed_eq_evaluate kernel d valid.2.1.1 (stored c)
  · intro u hu
    simpa only [localExpectedFor, valid.1.1] using valid.2.1.2.2 u hu
  · exact member
  · have count := countAbove_le d v
    unfold stdFuel
    omega

theorem strongFormationValidFor_valid (kernel : CompleteProfileKernel.Kernel)
    (d : Graph) (c : CompleteProfileCertificate) (valid : StrongFormationValidFor kernel d c) :
    CompleteProfileKernel.Valid kernel d := by
  refine ⟨valid.2.1.1, valid.2.1.2.1, (CompleteProfileKernel.rootsB_iff kernel d).mpr ?_⟩
  intro root member terminal
  rw [← stored_eq_evaluate kernel d c valid root member]
  exact valid.2.2 root member terminal

theorem strongFormationValidComplete_ReferenceVerdict (d : Graph) (c : CompleteProfileCertificate)
    (valid : StrongFormationValidComplete d c) : SingleIntroductionReference.ReferenceVerdict d :=
  (CompleteProfileKernel.tagged_valid d).mp
    (strongFormationValidFor_valid CompleteProfileKernel.tagged d c valid)

theorem strongFormationValidCompleteB_ReferenceVerdict (d : Graph) (c : CompleteProfileCertificate)
    (checked : strongFormationValidCompleteB d c = true) : SingleIntroductionReference.ReferenceVerdict d :=
  strongFormationValidComplete_ReferenceVerdict d c (strongFormationValidCompleteB_sound d c checked)

theorem strongFormationValidComplete_roots_derivable (d : Graph) (c : CompleteProfileCertificate)
    (valid : StrongFormationValidComplete d c) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA :=
  SingleIntroductionReference.ReferenceVerdict.roots_derivable d (strongFormationValidComplete_ReferenceVerdict d c valid)

theorem strongFormationValidCompleteB_roots_derivable (d : Graph) (c : CompleteProfileCertificate)
    (checked : strongFormationValidCompleteB d c = true) :
    ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
      Derives [] root.FORMULA :=
  strongFormationValidComplete_roots_derivable d c (strongFormationValidCompleteB_sound d c checked)

#print axioms strongFormationValidFor_valid
#print axioms strongFormationValidComplete_ReferenceVerdict
#print axioms strongFormationValidCompleteB_roots_derivable
end RouteCertificate
