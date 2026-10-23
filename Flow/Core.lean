import Flow.MultiIntroInclude


def Flow (d : Graph) : Prop := MultiIntro.ValidMI .general d

namespace Flow

theorem of_reference (d : Graph) : SingleIntroductionReference.ReferenceVerdict d → Flow d :=
  MultiIntroInclude.flowRef_implies_general d

theorem iff_reference_of_no_multi (d : Graph) (noMulti : MultiIntro.NoMultiIntro d) :
    Flow d ↔ SingleIntroductionReference.ReferenceVerdict d :=
  MultiIntro.valid_identity .general d noMulti

end Flow
