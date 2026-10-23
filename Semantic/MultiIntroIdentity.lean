import Semantic.MultiIntroLocality
import Semantic.ThesisMain

open Semantic FlowSpec
namespace MultiIntro
set_option maxHeartbeats 2000000



def noMultiIntroB (d : Graph) : Bool :=
  (d.EDGES.map Deduction.END).all fun v => !multiIntroShapeB d v

def NoMultiIntro (d : Graph) : Prop := noMultiIntroB d = true

instance (d : Graph) : Decidable (NoMultiIntro d) := inferInstanceAs (Decidable (_ = _))

theorem shape_has_incoming (d : Graph) (v : Vertex)
    (shape : multiIntroShapeB d v = true) : get_rule.incoming v d ≠ [] := by
  cases formula : v.FORMULA with
  | atom symbol => simp [multiIntroShapeB, formula] at shape
  | implication alpha beta =>
    have parts : v.HYPOTHESIS = false ∧
        (((elimPairsAt d v).isEmpty = true ∧ 2 ≤ (get_rule.incoming v d).length) ∧
          (get_rule.incoming v d).all (fun e => e.START.FORMULA == beta) = true) := by
      simpa only [multiIntroShapeB, formula, Bool.and_eq_true, Bool.not_eq_true',
        decide_eq_true_eq] using shape
    have length := parts.2.1.2
    intro empty
    simp [empty] at length

theorem shape_false (d : Graph) (noMulti : NoMultiIntro d) (v : Vertex) :
    multiIntroShapeB d v = false := by
  cases shape : multiIntroShapeB d v with
  | false => rfl
  | true =>
    obtain ⟨e, member⟩ := List.exists_mem_of_ne_nil _ (shape_has_incoming d v shape)
    have target : v ∈ d.EDGES.map Deduction.END :=
      List.mem_map.mpr ⟨e, mem_incoming_mem_edges v d member, mem_incoming_end_eq v d member⟩
    have check := List.all_eq_true.mp noMulti v target
    simp [shape] at check

theorem noMultiIntro_iff (d : Graph) :
    NoMultiIntro d ↔ ∀ v, ¬MultiIntroShape d v := by
  constructor
  · intro checked v shape
    have absent := shape_false d checked v
    simp only [MultiIntroShape, absent, Bool.false_eq_true] at shape
  · intro absent
    unfold NoMultiIntro noMultiIntroB
    rw [List.all_eq_true]
    intro v _
    cases shape : multiIntroShapeB d v with
    | false => rfl
    | true => exact False.elim (absent v shape)

theorem evaluate_identity (variant : Variant) (d : Graph) (noMulti : NoMultiIntro d) :
    ∀ fuel v, evaluateAt variant d fuel v = SingleIntroductionReference.routes d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v; rfl
  | succ fuel ih =>
    intro v
    simp only [evaluateAt, step, shape_false d noMulti, Bool.false_eq_true, if_false]
    change CompleteProfileKernel.tagged.step d (evaluateAt variant d fuel) v =
      CompleteProfileKernel.tagged.step d (SingleIntroductionReference.routes d fuel) v
    rw [funext ih]

theorem domain_identity (d : Graph) (noMulti : NoMultiIntro d) :
    MIDomain d ↔ structuralValid d := by
  have same : miDomainB d = structuralValidB d := by
    simp only [miDomainB, structuralValidB, shape_false d noMulti, Bool.or_false]
  unfold MIDomain
  rw [same]
  exact ⟨structuralValidB_sound d, CompleteProfilePositions.structuralValidB_complete d⟩

theorem valid_identity (variant : Variant) (d : Graph) (noMulti : NoMultiIntro d) :
    ValidMI variant d ↔ SingleIntroductionReference.ReferenceVerdict d := by
  have same : rootsB variant d = SingleIntroductionReference.rootsB d := by
    simp only [rootsB, SingleIntroductionReference.rootsB, evaluate_identity variant d noMulti]
  simp only [ValidMI, SingleIntroductionReference.ReferenceVerdict, domain_identity d noMulti, same]

theorem thesis_main_transfer (variant : Variant) (d : Graph) (noMulti : NoMultiIntro d) :
    (ValidMI variant d ↔ structuralValid d ∧ LeveledColored d ∧ RouteGrid.acceptedB d = true) ∧
    (ValidMI variant d ↔ ∃ c, RouteCertificate.StrongFormationValidComplete d c) ∧
    (ValidMI variant d ↔ structuralValid d ∧ LeveledColored d ∧ OccurrenceReadings.occurrenceVerdictComplete d) := by
  rw [valid_identity variant d noMulti]
  exact ⟨(ThesisMain.main d).1, (ThesisMain.main d).2.1, (ThesisMain.main d).2.2.1⟩

end MultiIntro
