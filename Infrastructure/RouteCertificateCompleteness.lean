import Infrastructure.RouteCertificateSoundness

open Semantic FlowSpec CertificateModel

namespace RouteCertificate

theorem generated_stored (kernel : CompleteProfileKernel.Kernel) (d : Graph)
    (structural : structuralValid d) {v : Vertex} (member : v ∈ d.NODES) :
    stored (generateFor kernel d) v = CompleteProfileKernel.evaluateAt kernel d (stdFuel d) v := by
  simp only [stored, generateFor, CompleteProfileKernel.memoAt_eq_spec kernel d structural]
  exact CompleteProfileKernel.lookup_spec kernel d (stdFuel d) member

theorem generateFor_expanded (kernel : CompleteProfileKernel.Kernel) (d : Graph)
    (structural : structuralValid d) : ExpandedCertComplete d (generateFor kernel d) := by
  refine ⟨rfl, ?_⟩
  simp [generateFor, CompleteProfileKernel.memoAt_eq_spec kernel d structural,
    CompleteProfileKernel.specificationTable, List.map_map, Function.comp_def]

theorem generateFor_localChecks (kernel : CompleteProfileKernel.Kernel) (d : Graph)
    (suitable : SuitableGraphComplete d) : LocalChecksFor kernel d (generateFor kernel d) := by
  refine ⟨suitable.1, suitable.2, ?_⟩
  intro v member
  rw [generated_stored kernel d suitable.1 member]
  change CompleteProfileKernel.evaluateAt kernel d (stdFuel d) v =
    kernel.step d (stored (generateFor kernel d)) v
  rw [kernel.congr d v (stored (generateFor kernel d))
    (CompleteProfileKernel.evaluateAt kernel d (stdFuel d)) (fun u hu =>
      generated_stored kernel d suitable.1 (mem_predsOf_mem_nodes suitable.1 hu))]
  exact (CompleteProfileKernel.standard_fixed kernel d suitable.1 member).symm

theorem generateFor_strongFormation (kernel : CompleteProfileKernel.Kernel) (d : Graph)
    (suitable : SuitableGraphComplete d) (accepted : CompleteProfileKernel.rootsB kernel d = true) :
    StrongFormationValidFor kernel d (generateFor kernel d) := by
  refine ⟨generateFor_expanded kernel d suitable.1,
    generateFor_localChecks kernel d suitable, ?_⟩
  intro root member terminal
  rw [generated_stored kernel d suitable.1 member]
  exact (CompleteProfileKernel.rootsB_iff kernel d).mp accepted root member terminal

theorem valid_iff_generated (kernel : CompleteProfileKernel.Kernel) (d : Graph) :
    CompleteProfileKernel.Valid kernel d ↔ StrongFormationValidFor kernel d (generateFor kernel d) := by
  constructor
  · intro valid
    exact generateFor_strongFormation kernel d ⟨valid.1, valid.2.1⟩ valid.2.2
  · exact strongFormationValidFor_valid kernel d (generateFor kernel d)

theorem generateCompleteProfileCertificate_eq_generateFor (d : Graph) (structural : structuralValid d) :
    generateCompleteProfileCertificate d = generateFor CompleteProfileKernel.tagged d := by
  unfold generateCompleteProfileCertificate generateFor
  congr 1
  rw [CompleteProfileKernel.memoAt_eq_spec CompleteProfileKernel.tagged d structural]
  change LambdaFlow.tableAt d (SingleIntroductionReference.guard d) (SingleIntroductionReference.guard d)
    (stdFuel d) = CompleteProfileKernel.specificationTable CompleteProfileKernel.tagged d (stdFuel d)
  rw [LambdaFlow.tableAt_eq_spec]
  unfold LambdaFlow.specificationTable CompleteProfileKernel.specificationTable
  apply List.map_congr_left
  intro v hv
  rw [CompleteProfileKernel.tagged_evaluate]
  rfl

theorem ReferenceVerdict_generatedCompleteProfileCertificate (d : Graph) (valid : SingleIntroductionReference.ReferenceVerdict d)
    (suitable : SuitableGraphComplete d) : StrongFormationValidComplete d (generateCompleteProfileCertificate d) := by
  rw [generateCompleteProfileCertificate_eq_generateFor d suitable.1]
  exact (valid_iff_generated CompleteProfileKernel.tagged d).mp ((CompleteProfileKernel.tagged_valid d).mpr valid)



theorem ReferenceVerdict_iff_generatedCompleteProfileCertificate (d : Graph) :
    SingleIntroductionReference.ReferenceVerdict d ↔ StrongFormationValidComplete d (generateCompleteProfileCertificate d) := by
  constructor
  · intro valid
    exact ReferenceVerdict_generatedCompleteProfileCertificate d valid ⟨valid.1, valid.2.1⟩
  · exact strongFormationValidComplete_ReferenceVerdict d (generateCompleteProfileCertificate d)

theorem ReferenceVerdict_iff_exists_complete_profile_certificate (d : Graph) :
    SingleIntroductionReference.ReferenceVerdict d ↔ ∃ c, StrongFormationValidComplete d c := by
  constructor
  · intro valid
    exact ⟨generateCompleteProfileCertificate d, (ReferenceVerdict_iff_generatedCompleteProfileCertificate d).mp valid⟩
  · rintro ⟨c, valid⟩
    exact strongFormationValidComplete_ReferenceVerdict d c valid

#print axioms generateFor_strongFormation
#print axioms ReferenceVerdict_iff_generatedCompleteProfileCertificate
#print axioms ReferenceVerdict_iff_exists_complete_profile_certificate
end RouteCertificate
