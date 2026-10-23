import Infrastructure.RouteCertificateModel

open Semantic FlowSpec CertificateModel RouteCertificate

namespace CompleteProfilePositions




inductive VerificationPositionComplete where
  | graphIdentity
  | rowKeyOrder
  | structuralDomain
  | productionRow (slot : Nat) (node : Vertex)
  | rootDischarge (slot : Nat) (node : Vertex)
  deriving Repr, DecidableEq


def rootDischargeAtCompleteB (c : CompleteProfileCertificate) (v : Vertex) : Bool :=
  !(stored c v).isEmpty &&
    (stored c v).all fun route => route.1.isEmpty && route.2.isEmpty


def checkPositionCompleteB (d : Graph) (c : CompleteProfileCertificate) :
    VerificationPositionComplete → Bool
  | .graphIdentity => decide (c.graph = d)
  | .rowKeyOrder => decide (c.rows.map Prod.fst = d.NODES)
  | .structuralDomain => structuralValidB d
  | .productionRow _ vertex => decide (stored c vertex = localExpectedComplete c vertex)
  | .rootDischarge _ vertex => rootDischargeAtCompleteB c vertex



def positionsComplete (d : Graph) : List VerificationPositionComplete :=
  [.graphIdentity, .rowKeyOrder, .structuralDomain] ++
    (d.NODES.zipIdx.map fun entry => .productionRow entry.2 entry.1) ++
    ((d.NODES.zipIdx.filter fun entry =>
      (get_rule.outgoing entry.1 d).isEmpty).map fun entry =>
        .rootDischarge entry.2 entry.1)

def FailedAtComplete (d : Graph) (c : CompleteProfileCertificate)
    (position : VerificationPositionComplete) : Prop :=
  checkPositionCompleteB d c position = false



def positionReadVertices (c : CompleteProfileCertificate) :
    VerificationPositionComplete → List Vertex
  | .productionRow _ vertex => vertex :: predsOf c.graph vertex
  | .rootDischarge _ vertex => [vertex]
  | _ => []

theorem structuralValidB_complete (d : Graph)
    (valid : structuralValid d) : structuralValidB d = true := by
  classical
  simp only [structuralValidB, Bool.and_eq_true, List.all_eq_true]
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro edge member
    simpa using valid.1 edge member
  · intro edge member
    simpa using valid.2.1 edge member
  · intro v member
    by_cases incoming : get_rule.incoming v d = []
    · simp [incoming]
    by_cases outgoing : get_rule.outgoing v d = []
    · simp [outgoing]
    obtain elimination | introduction := valid.2.2 v member incoming outgoing
    · obtain ⟨hypothesis, predecessors, pairs⟩ := elimination
      have shape : nodeElimShapeWideB d v = true := by
        simp [nodeElimShapeWideB, hypothesis, predecessors, pairs]
      simp [shape]
    · obtain ⟨premise, alpha, beta, hypothesis, edge, formula,
        premiseFormula, pairs⟩ := introduction
      have shape : nodeIntroShapeB d v = true := by
        simp [nodeIntroShapeB, hypothesis, edge, formula, premiseFormula, pairs]
      simp [shape]

theorem structuralDomain_iff (d : Graph) :
    structuralValidB d = true ↔ structuralValid d ∧ LeveledColored d := by
  constructor
  · intro checked
    have valid := structuralValidB_sound d checked
    exact ⟨valid, valid.1⟩
  · exact fun valid => structuralValidB_complete d valid.1

theorem rootDischargeAtCompleteB_iff (c : CompleteProfileCertificate) (v : Vertex) :
    rootDischargeAtCompleteB c v = true ↔
      stored c v ≠ [] ∧ ∀ route ∈ stored c v, route.1 = [] ∧ route.2 = [] := by
  simp [rootDischargeAtCompleteB]

private theorem forall_zipIdx_fst_iff {α : Type} (values : List α)
    (predicate : α → Prop) :
    (∀ entry ∈ values.zipIdx, predicate entry.1) ↔
      ∀ value ∈ values, predicate value := by
  have equal := congrArg (fun xs : List α => ∀ x ∈ xs, predicate x)
    (List.zipIdx_map_fst 0 values)
  simpa only [List.forall_mem_map] using Iff.of_eq equal

theorem positionsComplete_pass_iff (d : Graph) (c : CompleteProfileCertificate) :
    (∀ position ∈ positionsComplete d, checkPositionCompleteB d c position = true) ↔
      StrongFormationValidComplete d c := by
  simp only [positionsComplete, List.forall_mem_append, List.forall_mem_cons,
    List.forall_mem_nil, List.forall_mem_map, List.mem_filter,
    and_imp, checkPositionCompleteB, decide_eq_true_eq]
  rw [forall_zipIdx_fst_iff d.NODES (fun v => stored c v = localExpectedComplete c v)]
  rw [forall_zipIdx_fst_iff d.NODES (fun v =>
    (get_rule.outgoing v d).isEmpty = true → rootDischargeAtCompleteB c v = true)]
  simp only [rootDischargeAtCompleteB_iff, structuralDomain_iff]
  unfold StrongFormationValidComplete StrongFormationValidFor ExpandedCertComplete
    LocalChecksFor UniversalRootDischargeComplete
  tauto



theorem strongFormationValidComplete_iff_no_failed_position (d : Graph)
    (c : CompleteProfileCertificate) :
    StrongFormationValidComplete d c ↔
      ∀ position ∈ positionsComplete d, ¬ FailedAtComplete d c position := by
  rw [← positionsComplete_pass_iff]
  simp only [FailedAtComplete, Bool.not_eq_false]

theorem invalid_has_failed_position (d : Graph) (c : CompleteProfileCertificate)
    (invalid : ¬ StrongFormationValidComplete d c) :
    ∃ position ∈ positionsComplete d, FailedAtComplete d c position := by
  classical
  by_contra none
  apply invalid
  apply (strongFormationValidComplete_iff_no_failed_position d c).mpr
  intro position member failed
  exact none ⟨position, member, failed⟩



theorem checkPositionCompleteB_congr (d : Graph) (left right : CompleteProfileCertificate)
    (position : VerificationPositionComplete)
    (graph : left.graph = right.graph)
    (keys : left.rows.map Prod.fst = right.rows.map Prod.fst)
    (reads : ∀ v ∈ positionReadVertices left position,
      stored left v = stored right v) :
    checkPositionCompleteB d left position = checkPositionCompleteB d right position := by
  cases position with
  | graphIdentity => simp only [checkPositionCompleteB, graph]
  | rowKeyOrder => simp only [checkPositionCompleteB, keys]
  | structuralDomain => rfl
  | productionRow slot node =>
      have own := reads node (by simp [positionReadVertices])
      have production : localExpectedComplete left node = localExpectedComplete right node := by
        unfold localExpectedComplete localExpectedFor
        rw [← graph]
        apply CompleteProfileKernel.tagged.congr
        intro v member
        exact reads v (by simp [positionReadVertices, member])
      simp only [checkPositionCompleteB, own, production]
  | rootDischarge slot node =>
      have own := reads node (by simp [positionReadVertices])
      simp only [checkPositionCompleteB, rootDischargeAtCompleteB, own]

theorem FailedAtComplete_congr (d : Graph) (left right : CompleteProfileCertificate)
    (position : VerificationPositionComplete)
    (graph : left.graph = right.graph)
    (keys : left.rows.map Prod.fst = right.rows.map Prod.fst)
    (reads : ∀ v ∈ positionReadVertices left position,
      stored left v = stored right v) :
    FailedAtComplete d left position ↔ FailedAtComplete d right position := by
  unfold FailedAtComplete
  rw [checkPositionCompleteB_congr d left right position graph keys reads]

#print axioms strongFormationValidComplete_iff_no_failed_position
#print axioms invalid_has_failed_position
#print axioms checkPositionCompleteB_congr
end CompleteProfilePositions
