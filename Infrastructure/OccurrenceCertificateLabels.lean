import Infrastructure.OccurrenceCertificateModel

open Semantic FlowSpec CertificateModel SelfCertifying

namespace OccurrenceCertificate



theorem normalized_mem_labelGroup2 (c : OccurrenceCertificate) (edge : Deduction)
    (route : FlowPair) (member : route ∈ StoredRoutes c edge.START)
    (head : headColour route.2 = edge.COLOUR) :
    (normalizeDep route.1, route.2) ∈ labelGroup2 c edge := by
  apply List.mem_filter.mpr
  constructor
  · apply List.mem_eraseDups.mpr
    exact List.mem_map.mpr ⟨route, member, rfl⟩
  · simp [head]

theorem canonicalLocalLabel_checked (c : OccurrenceCertificate) :
    LocalLabelCheck c (canonicalLocalLabel c) := by
  intro edge _
  have self (dependency : Dep) : depSetEq dependency dependency = true :=
    (depSetEq_eq_true_iff _ _).mpr fun _ => Iff.rfl
  cases group : labelGroup2 c edge with
  | nil => simp [localLabelAtOccurrenceB, canonicalLocalLabel, group]
  | cons route rest =>
      cases rest <;> simp [localLabelAtOccurrenceB, canonicalLocalLabel, group, self]




theorem localLabelAtOccurrenceB_agrees (c : OccurrenceCertificate)
    (label : Deduction → LambdaFlow.Label) (edge : Deduction)
    (checked : localLabelAtOccurrenceB c label edge = true)
    (route : FlowPair) (member : route ∈ StoredRoutes c edge.START)
    (head : headColour route.2 = edge.COLOUR) :
    LambdaFlow.labelAgreesB (label edge) route.1 = true := by
  have grouped := normalized_mem_labelGroup2 c edge route member head
  cases tagged : label edge with
  | Lambda => simp [LambdaFlow.labelAgreesB]
  | Bits dependency =>
      cases group : labelGroup2 c edge with
      | nil => simp [localLabelAtOccurrenceB, group, tagged] at checked
      | cons only rest =>
          cases rest with
          | nil =>
              have same : (normalizeDep route.1, route.2) = only := by
                simpa [group] using grouped
              have bits : depSetEq only.1 dependency = true := by
                simpa [localLabelAtOccurrenceB, group, tagged] using checked
              have sameDep := congrArg Prod.fst same
              simp only [LambdaFlow.labelAgreesB]
              apply (depSetEq_eq_true_iff _ _).mpr
              intro formula
              have equality := (depSetEq_eq_true_iff _ _).mp bits formula
              simpa [← sameDep] using equality
          | cons next tail =>
              simp [localLabelAtOccurrenceB, group, tagged] at checked

theorem LocalLabelCheck.agrees (c : OccurrenceCertificate)
    (label : Deduction → LambdaFlow.Label) (checked : LocalLabelCheck c label)
    (edge : Deduction) (edge_mem : edge ∈ c.graph.EDGES)
    (route : FlowPair) (member : route ∈ StoredRoutes c edge.START)
    (head : headColour route.2 = edge.COLOUR) :
    LambdaFlow.labelAgreesB (label edge) route.1 = true :=
  localLabelAtOccurrenceB_agrees c label edge (checked edge edge_mem) route member head

theorem canonicalLocalLabel_agrees (c : OccurrenceCertificate) (edge : Deduction)
    (edge_mem : edge ∈ c.graph.EDGES) (route : FlowPair)
    (member : route ∈ StoredRoutes c edge.START)
    (head : headColour route.2 = edge.COLOUR) :
    LambdaFlow.labelAgreesB (canonicalLocalLabel c edge) route.1 = true :=
  LocalLabelCheck.agrees c _ (canonicalLocalLabel_checked c) edge edge_mem
    route member head

end OccurrenceCertificate
