import Semantic.MultiIntroCertificate
import Semantic.CompleteProfileRobustness

open Semantic FlowSpec SelfCertifyingProbe RouteCertificate
namespace MultiIntroPositions
abbrev Position := CompleteProfilePositions.VerificationPositionComplete


def rootDischargeAt (c : CompleteProfileCertificate) (v : Vertex) : Bool :=
  !(stored c v).isEmpty &&
    (stored c v).all fun route => route.1.isEmpty && route.2.isEmpty


def check (variant : MultiIntro.Variant) (d : Graph) (c : CompleteProfileCertificate) :
    Position → Bool
  | .graphIdentity => decide (c.graph = d)
  | .rowKeyOrder => decide (c.rows.map Prod.fst = d.NODES)
  | .structuralDomain => MultiIntro.miDomainB d
  | .productionRow _ vertex => decide (stored c vertex = localExpectedFor (MultiIntro.kernel variant) c vertex)
  | .rootDischarge _ vertex => rootDischargeAt c vertex



def positions (d : Graph) : List Position :=
  [.graphIdentity, .rowKeyOrder, .structuralDomain] ++
    (d.NODES.zipIdx.map fun entry => .productionRow entry.2 entry.1) ++
    ((d.NODES.zipIdx.filter fun entry =>
      (get_rule.outgoing entry.1 d).isEmpty).map fun entry =>
        .rootDischarge entry.2 entry.1)

def FailedAt (variant : MultiIntro.Variant) (d : Graph) (c : CompleteProfileCertificate)
    (position : Position) : Prop :=
  check variant d c position = false



def readVertices (c : CompleteProfileCertificate) :
    Position → List Vertex
  | .productionRow _ vertex => vertex :: predsOf c.graph vertex
  | .rootDischarge _ vertex => [vertex]
  | _ => []

theorem domain_iff (d : Graph) :
    MultiIntro.miDomainB d = true ↔ MultiIntro.MIDomain d ∧ LeveledColored d :=
  ⟨fun h => ⟨h, MultiIntro.domain_levels d h⟩, fun h => h.1⟩

theorem checked_iff (variant : MultiIntro.Variant) (d : Graph) (c : CompleteProfileCertificate) :
    MultiIntro.strongCertificateB variant d c = true ↔ MultiIntro.StrongCertificate variant d c := by
  constructor
  · exact MultiIntro.strongCertificateB_sound variant d c
  · intro formed
    simp only [MultiIntro.strongCertificateB, Bool.and_eq_true]
    refine ⟨⟨⟨(expandedCertCompleteB_iff d c).mpr formed.1, formed.2.1.1.1⟩, ?_⟩,
      (universalRootDischargeCompleteB_iff d c).mpr formed.2.2⟩
    simpa only [List.all_eq_true, decide_eq_true_eq] using formed.2.1.2

theorem rootDischargeAt_iff (c : CompleteProfileCertificate) (v : Vertex) :
    rootDischargeAt c v = true ↔
      stored c v ≠ [] ∧ ∀ route ∈ stored c v, route.1 = [] ∧ route.2 = [] := by
  simp [rootDischargeAt]

private theorem forall_zipIdx_fst_iff {α : Type} (values : List α)
    (predicate : α → Prop) :
    (∀ entry ∈ values.zipIdx, predicate entry.1) ↔
      ∀ value ∈ values, predicate value := by
  have equal := congrArg (fun xs : List α => ∀ x ∈ xs, predicate x)
    (List.zipIdx_map_fst 0 values)
  simpa only [List.forall_mem_map] using Iff.of_eq equal

theorem positions_pass_iff (variant : MultiIntro.Variant) (d : Graph) (c : CompleteProfileCertificate) :
    (∀ position ∈ positions d, check variant d c position = true) ↔
      MultiIntro.StrongCertificate variant d c := by
  simp only [positions, List.forall_mem_append, List.forall_mem_cons,
    List.forall_mem_nil, List.forall_mem_map, List.mem_filter,
    and_imp, check, decide_eq_true_eq]
  rw [forall_zipIdx_fst_iff d.NODES (fun v => stored c v = localExpectedFor (MultiIntro.kernel variant) c v)]
  rw [forall_zipIdx_fst_iff d.NODES (fun v =>
    (get_rule.outgoing v d).isEmpty = true → rootDischargeAt c v = true)]
  simp only [rootDischargeAt_iff, domain_iff]
  unfold MultiIntro.StrongCertificate ExpandedCertComplete
    MultiIntro.LocalChecks MultiIntro.SuitableGraph UniversalRootDischargeComplete
  tauto



theorem formation_iff_no_failed_position (variant : MultiIntro.Variant) (d : Graph)
    (c : CompleteProfileCertificate) :
    MultiIntro.StrongCertificate variant d c ↔
      ∀ position ∈ positions d, ¬ FailedAt variant d c position := by
  rw [← positions_pass_iff]
  simp only [FailedAt, Bool.not_eq_false]

theorem invalid_has_failed_position (variant : MultiIntro.Variant) (d : Graph) (c : CompleteProfileCertificate)
    (invalid : ¬ MultiIntro.StrongCertificate variant d c) :
    ∃ position ∈ positions d, FailedAt variant d c position := by
  classical
  by_contra none
  apply invalid
  apply (formation_iff_no_failed_position variant d c).mpr
  intro position member failed
  exact none ⟨position, member, failed⟩

theorem checked_iff_no_failed_position (variant : MultiIntro.Variant) (d : Graph)
    (c : CompleteProfileCertificate) : MultiIntro.strongCertificateB variant d c = true ↔
      ∀ position ∈ positions d, ¬FailedAt variant d c position := by
  rw [checked_iff, formation_iff_no_failed_position]



theorem check_congr (variant : MultiIntro.Variant) (d : Graph) (left right : CompleteProfileCertificate)
    (position : Position)
    (graph : left.graph = right.graph)
    (keys : left.rows.map Prod.fst = right.rows.map Prod.fst)
    (reads : ∀ v ∈ readVertices left position,
      stored left v = stored right v) :
    check variant d left position = check variant d right position := by
  cases position with
  | graphIdentity => simp only [check, graph]
  | rowKeyOrder => simp only [check, keys]
  | structuralDomain => rfl
  | productionRow slot node =>
      have own := reads node (by simp [readVertices])
      have production : localExpectedFor (MultiIntro.kernel variant) left node = localExpectedFor (MultiIntro.kernel variant) right node := by
        unfold localExpectedFor
        rw [← graph]
        apply (MultiIntro.kernel variant).congr
        intro v member
        exact reads v (by simp [readVertices, member])
      simp only [check, own, production]
  | rootDischarge slot node =>
      have own := reads node (by simp [readVertices])
      simp only [check, rootDischargeAt, own]

theorem FailedAt_congr (variant : MultiIntro.Variant) (d : Graph) (left right : CompleteProfileCertificate)
    (position : Position)
    (graph : left.graph = right.graph)
    (keys : left.rows.map Prod.fst = right.rows.map Prod.fst)
    (reads : ∀ v ∈ readVertices left position,
      stored left v = stored right v) :
    FailedAt variant d left position ↔ FailedAt variant d right position := by
  unfold FailedAt
  rw [check_congr variant d left right position graph keys reads]


def readSet (c : CompleteProfileCertificate) (position : Position) :=
  CompleteProfileRobustness.rowReadSet c (readVertices c position)

def readBound (c : CompleteProfileCertificate) (position : Position) : Nat :=
  (readSet c position).card

theorem read_bound_le (c : CompleteProfileCertificate) (position : Position) :
    readBound c position ≤ (readVertices c position).length :=
  CompleteProfileRobustness.rowReadSet_card_le c _

theorem production_read_bound (c : CompleteProfileCertificate) (slot : Nat) (v : Vertex) :
    readBound c (.productionRow slot v) ≤ 1 + (predsOf c.graph v).length := by
  simpa [readVertices, Nat.add_comm] using read_bound_le c (.productionRow slot v)

theorem position_count (d : Graph) :
    (positions d).length = d.NODES.length +
      (d.NODES.filter fun v => (get_rule.outgoing v d).isEmpty).length + 3 := by
  have filtered : ((d.NODES.zipIdx.filter fun entry =>
      (get_rule.outgoing entry.1 d).isEmpty).map Prod.fst) =
      d.NODES.filter (fun v => (get_rule.outgoing v d).isEmpty) := by
    change List.map Prod.fst (List.filter
      ((fun v : Vertex => (get_rule.outgoing v d).isEmpty) ∘ Prod.fst) d.NODES.zipIdx) = _
    rw [← List.filter_map]
    simp only [List.zipIdx_map_fst]
  have lengths := congrArg List.length filtered
  simp only [List.length_map] at lengths
  simp only [positions, List.length_append, List.length_cons, List.length_nil,
    List.length_map, List.length_zipIdx]
  omega

#print axioms formation_iff_no_failed_position
#print axioms invalid_has_failed_position
#print axioms check_congr
#print axioms production_read_bound
#print axioms position_count
end MultiIntroPositions
