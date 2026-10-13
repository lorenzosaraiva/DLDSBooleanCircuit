import Robustness.RuleCompilation.ElimDensity

/-! Literal kernel-reduced positive/negative ordinary-rule controls.
These are local route checks, with no coverage or root-discharge assertion. -/
open DLDS FlowSemantics Robustness.BitLocality Robustness.SelectedProducers
namespace Robustness.RuleCompilation.Sanity
set_option maxHeartbeats 2000000

def A : Formula := .atom "a"
def B : Formula := .atom "b"
def introInput : Vertex := node 1 1 A true false []
def introRoot : Vertex := node 0 0 (.implication A A) false false []
def introGraph (colour : Nat) : Graph :=
  dlds [introRoot, introInput] [edge introInput introRoot colour [A]] []

def introOwner (slot : Fin 2) : Vertex := if slot = 0 then introRoot else introInput

def introFrame (colour : Nat) : IntroFrame (routes := 2) (introGraph colour) where
  owner := introOwner
  listed slot := by fin_cases slot <;> simp [introOwner, introGraph]
  output := 0
  input := 1
  alpha := A
  beta := A
  edgeIndex := 0
  incoming := edge introInput introRoot colour [A]
  nonhyp := rfl
  predecessors := by
    simp [predsOf, introGraph, introOwner, introRoot, introInput, List.eraseDups_cons, Bool.beq_eq_decide_eq]
  noElimination := by
    simp [elimPairsAt, predsOf, introGraph, introOwner, introRoot, introInput, A,
      List.eraseDups_cons, Bool.beq_eq_decide_eq]
  conclusion := rfl
  premiseFormula := rfl
  edgeSelected := by
    simp [edgesBetween, introGraph, introOwner, introRoot, introInput]

def introShape (inputLength outputLength : Nat) : RouteShape 2 1 where
  formulaAt _ := A
  injective _ _ _ := Subsingleton.elim _ _
  pathLength slot := if slot = 0 then outputLength else inputLength
  colourWidth := 1

def introBits (il ol : Nat) (badDependency badColour : Bool) :
    RuleCoord (introShape il ol) 2 → Bool
  | .inl _ => false
  | .inr (.inl (slot, _)) => if slot = 0 then badDependency else true
  | .inr (.inr _) => !badColour

private theorem word_true : decodeWord (fun _ : Fin 1 => true) = 1 := by decide +kernel
private theorem word_false : decodeWord (fun _ : Fin 1 => false) = 0 := by decide +kernel
private theorem colour_true (il ol : Nat) :
    decodeWord (fun _ : Fin (introShape il ol).colourWidth => true) = 1 := word_true
private theorem colour_false (il ol : Nat) :
    decodeWord (fun _ : Fin (introShape il ol).colourWidth => false) = 0 := word_false

theorem intro_empty_passes :
    introAcceptance (introShape 0 0) 2 (introFrame 0) (introBits 0 0 false false) = true := by
  rw [intro_acceptance_eq]
  norm_num [introFrame, introBits, introShape, introOwner, introGraph, introInput,
    introRoot, A, ruleRoutes, decodeRoutes, decodeDependency, depRemove,
    List.removeAll, consume, routeContinuableAtB, get_rule.outgoing,
    get_rule.outgoing.loop, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq]

theorem intro_dependency_rejects :
    introFailure (introShape 0 0) 2 (introFrame 0) (introBits 0 0 true false) = true := by
  unfold introFailure
  rw [intro_acceptance_eq]
  norm_num [introFrame, introBits, introShape, introOwner, introGraph, introInput,
    introRoot, A, ruleRoutes, decodeRoutes, decodeDependency, depRemove,
    List.removeAll, consume, routeContinuableAtB, get_rule.outgoing,
    get_rule.outgoing.loop, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq]

theorem intro_consumed_head_passes :
    introAcceptance (introShape 1 0) 2 (introFrame 1) (introBits 1 0 false false) = true := by
  rw [intro_acceptance_eq]
  norm_num [introFrame, introBits, introShape, introOwner, introGraph, introInput,
    introRoot, A, ruleRoutes, decodeRoutes, decodeDependency, depRemove,
    List.removeAll, consume, routeContinuableAtB, get_rule.outgoing,
    get_rule.outgoing.loop, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq,
    colour_true, colour_false]

theorem intro_corrupt_head_rejects :
    introFailure (introShape 1 0) 2 (introFrame 1) (introBits 1 0 false true) = true := by
  unfold introFailure
  rw [intro_acceptance_eq]
  norm_num [introFrame, introBits, introShape, introOwner, introGraph, introInput,
    introRoot, A, ruleRoutes, decodeRoutes, decodeDependency, depRemove,
    List.removeAll, consume, routeContinuableAtB, get_rule.outgoing,
    get_rule.outgoing.loop, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq,
    colour_true, colour_false]
  all_goals decide +kernel

def minor : Vertex := node 1 1 A true false []
def major : Vertex := node 2 1 (.implication A B) true false []
def elimRoot : Vertex := node 0 0 B false false []
def elimGraph : Graph := dlds [elimRoot, minor, major]
  [edge minor elimRoot 0 [A], edge major elimRoot 0 [.implication A B]] []
def elimOwner (slot : Fin 3) : Vertex :=
  if slot = 0 then elimRoot else if slot = 1 then minor else major

def elimFrame : ElimFrame (routes := 3) elimGraph where
  owner := elimOwner
  listed slot := by fin_cases slot <;> simp [elimOwner, elimGraph]
  output := 0
  minor := 1
  major := 2
  pairIndex := 0
  minorEdgeIndex := 0
  majorEdgeIndex := 0
  minorEdge := edge minor elimRoot 0 [A]
  majorEdge := edge major elimRoot 0 [.implication A B]
  nonhyp := rfl
  pairSelected := by
    simp [elimPairsAt, predsOf, elimOwner, elimGraph, minor, major, elimRoot, A, B,
      Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq]
  minorEdgeSelected := by
    simp [edgesBetween, elimOwner, elimGraph, minor, major, elimRoot, Vertex.node.injEq]
  majorEdgeSelected := by
    simp [edgesBetween, elimOwner, elimGraph, minor, major, elimRoot, Vertex.node.injEq]

def elimShape : RouteShape 3 2 where
  formulaAt code := if code = 0 then A else .implication A B
  injective := by
    intro left right same
    fin_cases left <;> fin_cases right <;> simp_all [A]
  pathLength _ := 0
  colourWidth := 1

def elimBits (corrupt : Bool) : RuleCoord elimShape 3 → Bool
  | .inl _ => false
  | .inr (.inl (slot, code)) =>
      if slot = 0 then !corrupt else if slot = 1 then decide (code = 0) else decide (code = 1)
  | .inr (.inr coordinate) => Fin.elim0 coordinate.2.1

theorem elim_empty_passes :
    elimAcceptance elimShape 3 elimFrame (elimBits false) = true := by
  rw [← elim_compiler_check_eq elimShape 3 1 (by decide) (by decide) elimFrame]
  norm_num [Program.check, elimProgram, unionGuards, consumeGuards, continuationGuards,
    atomGuard_check, Atom.eval, unionBitAtom, constantAtom, elimFrame, elimBits, elimShape,
    elimOwner, elimGraph, minor, major, elimRoot, A, B,
    consume, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    List.finRange_succ, Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq]
  all_goals decide +kernel

theorem elim_union_rejects :
    elimFailure elimShape 3 elimFrame (elimBits true) = true := by
  unfold elimFailure
  rw [← elim_compiler_check_eq elimShape 3 1 (by decide) (by decide) elimFrame]
  norm_num [Program.check, elimProgram, unionGuards, consumeGuards, continuationGuards,
    atomGuard_check, Atom.eval, unionBitAtom, constantAtom, elimFrame, elimBits, elimShape,
    elimOwner, elimGraph, minor, major, elimRoot, A, B,
    consume, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    List.finRange_succ, Vertex.node.injEq, List.eraseDups_cons, Bool.beq_eq_decide_eq, Fin.reduceEq]


theorem intro_nonempty_root_rejects :
    introFailure (introShape 2 1) 2 (introFrame 1) (introBits 2 1 false false) = true := by
  unfold introFailure
  rw [intro_acceptance_eq]
  norm_num [introFrame, introBits, introShape, introOwner, introGraph, introInput,
    introRoot, A, ruleRoutes, decodeRoutes, decodeDependency, depRemove, List.removeAll,
    consume, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    List.finRange_succ, List.ofFn_succ, Vertex.node.injEq, List.eraseDups_cons,
    Bool.beq_eq_decide_eq, Fin.reduceEq, colour_true, colour_false]

def mismatchedTail : RuleCoord (introShape 2 1) 2 → Bool
  | .inl _ => false
  | .inr (.inl (slot, _)) => decide (slot = 1)
  | .inr (.inr coordinate) => decide (coordinate.1 = 1)

theorem consumed_tail_rejects :
    (consumeGuards (introShape 2 1) 2 1 (by decide) 1 0
      (edge introInput introRoot 1 [A])).all
        (fun guard => guard.check mismatchedTail) = false := by
  rw [consumeGuards_check]
  norm_num [ruleRoutes, decodeRoutes, introShape, mismatchedTail,
    List.ofFn_succ, colour_true, colour_false, consume, Fin.reduceEq]

theorem intro_bad_fixed_lengths_reject :
    introFailure (introShape 1 1) 2 (introFrame 1) (introBits 1 1 false false) = true := by
  unfold introFailure
  rw [intro_acceptance_eq]
  norm_num [introFrame, introBits, introShape, introOwner, introGraph, introInput,
    introRoot, A, ruleRoutes, decodeRoutes, decodeDependency, depRemove, List.removeAll,
    consume, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    List.finRange_succ, List.ofFn_succ, Vertex.node.injEq, List.eraseDups_cons,
    Bool.beq_eq_decide_eq, Fin.reduceEq, colour_true, colour_false]


theorem intro_corruption_witness :
    ∃ witness : CertificateSearch.Robustness.FailureWitness
        (introFailure (introShape 0 0) 2 (introFrame 0)) (introBits 0 0 true false),
      witness.support.card ≤ W 2 1 2 :=
  intro_failure_witness (introShape 0 0) 2 2 (by decide +kernel)
    (introFrame 0) (introBits 0 0 true false) intro_dependency_rejects

theorem elim_corruption_witness :
    ∃ witness : CertificateSearch.Robustness.FailureWitness
        (elimFailure elimShape 3 elimFrame) (elimBits true),
      witness.support.card ≤ W 3 2 2 :=
  elim_failure_witness elimShape 3 2 (by decide +kernel)
    elimFrame (elimBits true) elim_union_rejects

theorem intro_corruption_density :
    (1 : ℚ) / (2 ^ 23 * (M 2 1 2 : ℚ) ^ 11) ≤
      CertificateSearch.Robustness.oracleFailureDensity
        (introFailure (introShape 0 0) 2 (introFrame 0)) :=
  intro_failure_inverse_polynomial (introShape 0 0) 2 2 (by decide +kernel)
    (introFrame 0) (introBits 0 0 true false) intro_dependency_rejects

theorem elim_corruption_density :
    (1 : ℚ) / (2 ^ 23 * (M 3 2 2 : ℚ) ^ 11) ≤
      CertificateSearch.Robustness.oracleFailureDensity
        (elimFailure elimShape 3 elimFrame) :=
  elim_failure_inverse_polynomial elimShape 3 2 (by decide +kernel)
    elimFrame (elimBits true) elim_union_rejects

end Robustness.RuleCompilation.Sanity
