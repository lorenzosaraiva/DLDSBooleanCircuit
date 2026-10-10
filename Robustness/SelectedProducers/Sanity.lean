import Robustness.SelectedProducers.SeedDensity

/-! Small kernel-reduced controls for the actual selected seed checker and its
proved compiler. They are literal positive local fixtures, not closed proofs,
coverage certificates or a successor/Hermann application. -/
open DLDS Flow Robustness.BitLocality Search.Certificate.Robustness
namespace Robustness.SelectedProducers.Sanity

def atom : Formula := .atom "a"
def leaf : Vertex := node 1 1 atom true false []
def root : Vertex := node 2 0 atom false false []
def defaultGraph : Graph := dlds [leaf] [] []
def ancestorGraph : Graph := dlds [leaf, root] [edge leaf root 0 [atom]]
  [path root leaf [0]]

def defaultShape : RouteShape 1 1 where
  formulaAt _ := atom
  injective _ _ _ := Subsingleton.elim _ _
  pathLength _ := 0
  colourWidth := 1

def ancestorShape : RouteShape 1 1 where
  formulaAt _ := atom
  injective _ _ _ := Subsingleton.elim _ _
  pathLength _ := 1
  colourWidth := 1

def owner : Fin 1 → Vertex := fun _ => leaf
theorem default_listed : ∀ slot, owner slot ∈ defaultGraph.NODES := by intro slot; simp [owner, defaultGraph]
theorem ancestor_listed : ∀ slot, owner slot ∈ ancestorGraph.NODES := by intro slot; simp [owner, ancestorGraph]

def validDefault : SeedCoord defaultShape 1 → Bool
  | .inl _ => false
  | .inr (.inl _) => true
  | .inr (.inr coordinate) => Fin.elim0 coordinate.2.1

def corruptDependency : SeedCoord defaultShape 1 → Bool := fun _ => false

def validAncestor : SeedCoord ancestorShape 2 → Bool
  | .inl _ => false
  | .inr (.inl _) => true
  | .inr (.inr _) => false

def corruptResidual : SeedCoord ancestorShape 2 → Bool
  | .inl _ => false
  | .inr (.inl _) => true
  | .inr (.inr _) => true

def corruptIndex : SeedCoord ancestorShape 2 → Bool
  | .inl bit => decide (bit.val = 0)
  | .inr (.inl _) => true
  | .inr (.inr _) => false

private theorem pointer_one : pointerWidth 1 = 0 := by decide +kernel
private theorem pointer_two : pointerWidth 2 = 1 := by decide +kernel
private theorem word_zero : decodeWord (fun _ : Fin 1 => false) = 0 := by decide +kernel
private theorem word_one : decodeWord (fun _ : Fin 1 => true) = 1 := by decide +kernel

private theorem pointer_zero_word : decodeWord (fun _ : Fin (pointerWidth 2) => false) = 0 := by
  change decodeWord (fun _ : Fin 1 => false) = 0
  exact word_zero
private theorem pointer_one_word : decodeWord (fun _ : Fin (pointerWidth 2) => true) = 1 := by
  change decodeWord (fun _ : Fin 1 => true) = 1
  exact word_one
private theorem colour_zero_word : decodeWord (fun _ : Fin ancestorShape.colourWidth => false) = 0 := word_zero
private theorem colour_one_word : decodeWord (fun _ : Fin ancestorShape.colourWidth => true) = 1 := word_one

theorem default_passes :
    seedAcceptance defaultShape defaultGraph owner default_listed 0 .default 1 validDefault = true := by
  norm_num [seedFailure, seed_acceptance_eq, seedTarget?, ancestorIndex, pointer_one, pointer_two,
    decodeRoutes, decodeDependency, word_zero, word_one,
    pointer_zero_word, pointer_one_word, colour_zero_word, colour_one_word, defaultShape, ancestorShape,
    owner, leaf, root, atom, defaultGraph, ancestorGraph, validDefault,
    corruptDependency, validAncestor, corruptResidual, corruptIndex,
    ancestorsInto, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    consume, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq,
    BitVec.ofBoolListLE]


theorem ancestor_passes :
    seedAcceptance ancestorShape ancestorGraph owner ancestor_listed 0 .ancestor 2 validAncestor = true := by
  norm_num [seedFailure, seed_acceptance_eq, seedTarget?, ancestorIndex, pointer_one, pointer_two,
    decodeRoutes, decodeDependency, word_zero, word_one,
    pointer_zero_word, pointer_one_word, colour_zero_word, colour_one_word, defaultShape, ancestorShape,
    owner, leaf, root, atom, defaultGraph, ancestorGraph, validDefault,
    corruptDependency, validAncestor, corruptResidual, corruptIndex,
    ancestorsInto, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    consume, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq,
    BitVec.ofBoolListLE]


theorem dependency_fails :
    seedFailure defaultShape defaultGraph owner default_listed 0 .default 1 corruptDependency = true := by
  norm_num [seedFailure, seed_acceptance_eq, seedTarget?, ancestorIndex, pointer_one, pointer_two,
    decodeRoutes, decodeDependency, word_zero, word_one,
    pointer_zero_word, pointer_one_word, colour_zero_word, colour_one_word, defaultShape, ancestorShape,
    owner, leaf, root, atom, defaultGraph, ancestorGraph, validDefault,
    corruptDependency, validAncestor, corruptResidual, corruptIndex,
    ancestorsInto, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    consume, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq,
    BitVec.ofBoolListLE]


theorem residual_fails :
    seedFailure ancestorShape ancestorGraph owner ancestor_listed 0 .ancestor 2 corruptResidual = true := by
  norm_num [seedFailure, seed_acceptance_eq, seedTarget?, ancestorIndex, pointer_one, pointer_two,
    decodeRoutes, decodeDependency, word_zero, word_one,
    pointer_zero_word, pointer_one_word, colour_zero_word, colour_one_word, defaultShape, ancestorShape,
    owner, leaf, root, atom, defaultGraph, ancestorGraph, validDefault,
    corruptDependency, validAncestor, corruptResidual, corruptIndex,
    ancestorsInto, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    consume, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq,
    BitVec.ofBoolListLE]


theorem index_fails :
    seedFailure ancestorShape ancestorGraph owner ancestor_listed 0 .ancestor 2 corruptIndex = true := by
  norm_num [seedFailure, seed_acceptance_eq, seedTarget?, ancestorIndex, pointer_one, pointer_two,
    decodeRoutes, decodeDependency, word_zero, word_one,
    pointer_zero_word, pointer_one_word, colour_zero_word, colour_one_word, defaultShape, ancestorShape,
    owner, leaf, root, atom, defaultGraph, ancestorGraph, validDefault,
    corruptDependency, validAncestor, corruptResidual, corruptIndex,
    ancestorsInto, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    consume, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq,
    BitVec.ofBoolListLE]


theorem wrong_mode_fails :
    seedFailure ancestorShape ancestorGraph owner ancestor_listed 0 .default 2 validAncestor = true := by
  norm_num [seedFailure, seed_acceptance_eq, seedTarget?, ancestorIndex, pointer_one, pointer_two,
    decodeRoutes, decodeDependency, word_zero, word_one,
    pointer_zero_word, pointer_one_word, colour_zero_word, colour_one_word, defaultShape, ancestorShape,
    owner, leaf, root, atom, defaultGraph, ancestorGraph, validDefault,
    corruptDependency, validAncestor, corruptResidual, corruptIndex,
    ancestorsInto, routeContinuableAtB, get_rule.outgoing, get_rule.outgoing.loop,
    consume, List.finRange_succ, List.ofFn_succ, Vertex.node.injEq,
    BitVec.ofBoolListLE]


theorem dependency_witness :
    ∃ witness : FailureWitness
        (seedFailure defaultShape defaultGraph owner default_listed 0 .default 1) corruptDependency,
      witness.support.card ≤ W 1 1 2 :=
  seed_failure_witness defaultShape 1 2 (by decide +kernel)
    defaultGraph owner default_listed 0 .default corruptDependency dependency_fails

theorem residual_witness :
    ∃ witness : FailureWitness
        (seedFailure ancestorShape ancestorGraph owner ancestor_listed 0 .ancestor 2) corruptResidual,
      witness.support.card ≤ W 2 1 2 :=
  seed_failure_witness ancestorShape 2 2 (by decide +kernel)
    ancestorGraph owner ancestor_listed 0 .ancestor corruptResidual residual_fails

theorem actual_seed_density :
    (1 : ℚ) / (2 ^ 23 * (M 2 1 2 : ℚ) ^ 11) ≤
      oracleFailureDensity (seedFailure ancestorShape ancestorGraph owner ancestor_listed 0 .ancestor 2) :=
  seed_failure_inverse_polynomial ancestorShape 2 2 (by decide +kernel)
    ancestorGraph owner ancestor_listed 0 .ancestor corruptResidual residual_fails

end Robustness.SelectedProducers.Sanity
