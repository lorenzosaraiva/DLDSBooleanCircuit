import Infrastructure.CompleteProfileKernel

open Semantic FlowSpec CertificateModel

namespace RouteCertificate



structure CompleteProfileCertificate where
  graph : Graph
  rows : FlowTable
  deriving Repr

abbrev stored (c : CompleteProfileCertificate) : Vertex → List FlowPair := CompleteProfileKernel.lookup c.rows


def localExpectedFor (kernel : CompleteProfileKernel.Kernel) (c : CompleteProfileCertificate)
    (v : Vertex) : List FlowPair := kernel.step c.graph (stored c) v

abbrev localExpectedComplete := localExpectedFor CompleteProfileKernel.tagged

def ExpandedCertComplete (d : Graph) (c : CompleteProfileCertificate) : Prop :=
  c.graph = d ∧ c.rows.map Prod.fst = d.NODES

def LocalChecksFor (kernel : CompleteProfileKernel.Kernel) (d : Graph) (c : CompleteProfileCertificate) : Prop :=
  structuralValid d ∧ LeveledColored d ∧
    ∀ v ∈ d.NODES, stored c v = localExpectedFor kernel c v

def UniversalRootDischargeComplete (d : Graph) (c : CompleteProfileCertificate) : Prop :=
  ∀ root ∈ d.NODES, (get_rule.outgoing root d).isEmpty = true →
    stored c root ≠ [] ∧ ∀ route ∈ stored c root, route.1 = [] ∧ route.2 = []

def StrongFormationValidFor (kernel : CompleteProfileKernel.Kernel)
    (d : Graph) (c : CompleteProfileCertificate) : Prop :=
  ExpandedCertComplete d c ∧ LocalChecksFor kernel d c ∧ UniversalRootDischargeComplete d c

abbrev LocalChecksComplete := LocalChecksFor CompleteProfileKernel.tagged
abbrev StrongFormationValidComplete := StrongFormationValidFor CompleteProfileKernel.tagged

def expandedCertCompleteB (d : Graph) (c : CompleteProfileCertificate) : Bool :=
  decide (c.graph = d) && decide (c.rows.map Prod.fst = d.NODES)

def localChecksForB (kernel : CompleteProfileKernel.Kernel) (d : Graph) (c : CompleteProfileCertificate) : Bool :=
  structuralValidB d && d.NODES.all fun v =>
    decide (stored c v = localExpectedFor kernel c v)

def universalRootDischargeCompleteB (d : Graph) (c : CompleteProfileCertificate) : Bool :=
  d.NODES.all fun v => !(get_rule.outgoing v d).isEmpty ||
    (let routes := stored c v
     !routes.isEmpty && routes.all fun route => route.1.isEmpty && route.2.isEmpty)

def strongFormationValidForB (kernel : CompleteProfileKernel.Kernel)
    (d : Graph) (c : CompleteProfileCertificate) : Bool :=
  expandedCertCompleteB d c && localChecksForB kernel d c && universalRootDischargeCompleteB d c

abbrev localChecksCompleteB := localChecksForB CompleteProfileKernel.tagged
abbrev strongFormationValidCompleteB := strongFormationValidForB CompleteProfileKernel.tagged



def SuitableGraphComplete (d : Graph) : Prop := structuralValid d ∧ LeveledColored d

def generateFor (kernel : CompleteProfileKernel.Kernel) (d : Graph) : CompleteProfileCertificate :=
  ⟨d, CompleteProfileKernel.memoAt kernel d (stdFuel d)⟩


def generateCompleteProfileCertificate (d : Graph) : CompleteProfileCertificate := ⟨d, SingleIntroductionReference.table d⟩

theorem expandedCertCompleteB_iff (d : Graph) (c : CompleteProfileCertificate) :
    expandedCertCompleteB d c = true ↔ ExpandedCertComplete d c := by
  simp [expandedCertCompleteB, ExpandedCertComplete]

theorem localChecksForB_sound (kernel : CompleteProfileKernel.Kernel)
    (d : Graph) (c : CompleteProfileCertificate) (checked : localChecksForB kernel d c = true) :
    LocalChecksFor kernel d c := by
  obtain ⟨structureB, production⟩ : structuralValidB d = true ∧
      (d.NODES.all fun v => decide (stored c v = localExpectedFor kernel c v)) = true := by
    simpa only [localChecksForB, Bool.and_eq_true] using checked
  have structural := structuralValidB_sound d structureB
  refine ⟨structural, structural.1, ?_⟩
  simpa only [List.all_eq_true, decide_eq_true_eq] using production

theorem universalRootDischargeCompleteB_iff (d : Graph) (c : CompleteProfileCertificate) :
    universalRootDischargeCompleteB d c = true ↔ UniversalRootDischargeComplete d c := by
  simp only [universalRootDischargeCompleteB, UniversalRootDischargeComplete, List.all_eq_true,
    Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true',
    List.isEmpty_eq_false_iff, List.isEmpty_iff]
  constructor
  · intro checked root member terminal
    exact (checked root member).resolve_left (by simp [terminal])
  · intro checked root member
    by_cases terminal : get_rule.outgoing root d = []
    · exact Or.inr (checked root member terminal)
    · exact Or.inl (by simpa using terminal)

theorem strongFormationValidForB_sound (kernel : CompleteProfileKernel.Kernel)
    (d : Graph) (c : CompleteProfileCertificate) (checked : strongFormationValidForB kernel d c = true) :
    StrongFormationValidFor kernel d c := by
  obtain ⟨⟨expanded, localChecks⟩, roots⟩ :=
    (by simpa only [strongFormationValidForB, Bool.and_eq_true] using checked)
  exact ⟨(expandedCertCompleteB_iff d c).mp expanded,
    localChecksForB_sound kernel d c localChecks,
    (universalRootDischargeCompleteB_iff d c).mp roots⟩

theorem strongFormationValidCompleteB_sound (d : Graph) (c : CompleteProfileCertificate)
    (checked : strongFormationValidCompleteB d c = true) : StrongFormationValidComplete d c :=
  strongFormationValidForB_sound CompleteProfileKernel.tagged d c checked

#print axioms strongFormationValidCompleteB_sound
end RouteCertificate
