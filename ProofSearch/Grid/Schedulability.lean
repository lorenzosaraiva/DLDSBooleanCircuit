import ProofSearch.Grid.Completeness
import ProofSearch.Grid.Counting

/-!
# Space A completeness within a schedulable proof-tree domain

This module records the smallest standard contraction candidate used to test
the inverse direction of `ProofSearch.Grid.gridMarked_sound`.

The formula is the implicational type of the `W` combinator,
`(A → A → B) → A → B`.  Its displayed natural-deduction tree uses the same
open `A` assumption in both eliminations before one introduction discharges
that assumption.  The current grid, by contrast, starts with one carrier per
formula column and `propagate_tokens` never forks a carrier.  The executable
facts below describe the supplied fixtures and do not establish an
all-heights non-representability theorem.
-/

open DLDS

namespace ProofSearch.Grid.Schedulability

/-- A derivation is operationally schedulable when a rectangular assignment
has a produced goal cell, no evaluator error, and the backward extractor
recovers that derivation.  Unlike `gridMarkedB`, this predicate does not assume
that the final dependency vector is discharged. -/
def GridSchedulable (fs : List Formula) (goal : Formula) (height : Nat)
    (tree : ProofSearch.Grid.NDTree) : Prop :=
  ∃ assignment : ProofSearch.Grid.Counting.Assignment fs height,
      ProofSearch.Grid.Definitions.goalProducedB
          (ProofSearch.Grid.Definitions.gridFor fs height)
          (ProofSearch.Grid.Definitions.initialVectorsFor fs) assignment.paths
          (ProofSearch.Grid.Definitions.goalColumnFor fs goal) = true ∧
      (get_eval_result
          (ProofSearch.Grid.Definitions.gridFor fs height)
          (ProofSearch.Grid.Definitions.initialVectorsFor fs) assignment.paths).2 = false ∧
      ProofSearch.Grid.extractGoalCone
        fs goal height assignment.paths = some tree

def A : Formula := #"A"
def B : Formula := #"B"
def C : Formula := A >> B
def D : Formula := A >> C
def W : Formula := D >> C

/-- The distinct subformula universe of `W`. -/
def formulas : List Formula := [A, B, C, D, W]

/-- The ordinary contraction proof `λ f a, f a a`. -/
def contractionTree : ProofSearch.Grid.NDTree :=
  .intro D C
    (.intro A B
      (.elim A B
        (.elim A C (.hyp D) (.hyp A))
        (.hyp A)))

theorem contractionTree_is_closed_proof :
    formulas.Nodup ∧
      contractionTree.wellFormedB = true ∧
      contractionTree.conclusion = W ∧
      contractionTree.dependencies = [] ∧
      contractionTree.height = 4 := by
  native_decide

/-- The smallest positive control: the one-rule identity proof needs one
transition and is represented without padding. -/
theorem direct_identity_exact_height_one :
    ProofSearch.Grid.Completeness.InPathCarrier ProofSearch.Grid.Definitions.formulas 1
        (ProofSearch.Grid.Completeness.directIdentityPath 1) ∧
      ProofSearch.Grid.extractGoalCone
          ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 1
            (ProofSearch.Grid.Completeness.directIdentityPath 1) = some ProofSearch.Grid.Completeness.directIdentityTree ∧
      ProofSearch.Grid.CertifiedMarked
          ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 1
            (ProofSearch.Grid.Completeness.directIdentityPath 1) := by
  constructor
  · unfold ProofSearch.Grid.Completeness.InPathCarrier
    native_decide
  constructor
  · native_decide
  · change ProofSearch.Grid.certifiedMarkedB
      ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 1
        (ProofSearch.Grid.Completeness.directIdentityPath 1) = true
    native_decide

theorem direct_identity_height_one_read_cells :
    (ProofSearch.Grid.Counting.readCells
      ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 1
        (ProofSearch.Grid.Completeness.directIdentityPath 1)).card = 4 := by
  native_decide

/-- Once the genuinely difficult scheduling witness exists, closed dependency
data is enough to obtain the positive proof-search mark.  This isolates the
missing inverse problem from dependency-vector correctness. -/
theorem gridMarked_of_gridSchedulable
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height : Nat)
    (tree : ProofSearch.Grid.NDTree)
    (scheduled : GridSchedulable fs goal height tree)
    (closed : tree.dependencies = []) :
    ∃ assignment : ProofSearch.Grid.Counting.Assignment fs height,
      ProofSearch.Grid.extractGoalCone
          fs goal height assignment.paths = some tree ∧
        ProofSearch.Grid.gridMarkedB
          fs goal height assignment.paths = true := by
  obtain ⟨assignment, produced, noError, extracted⟩ := scheduled
  let paths := assignment.paths
  let layers := ProofSearch.Grid.Definitions.gridFor fs height
  let initial := ProofSearch.Grid.Definitions.initialVectorsFor fs
  let goalColumn := fs.idxOf goal
  let result := get_eval_result layers initial paths
  let ss := ProofSearch.Grid.snapshots fs height paths
  have ssLength : ss.length = height + 1 := by
    simpa [ss] using ProofSearch.Grid.snapshots_length fs height paths
  have finalBound : height < ss.length := by omega
  let final := ss[height]
  have finalAt : ss[height]? = some final :=
    List.getElem?_eq_getElem finalBound
  have finalLast : ss.getLast? = some final := by
    rw [List.getLast?_eq_getElem?]
    simpa [ssLength] using finalAt
  have finalTokens : ProofSearch.Grid.Definitions.tokensAtFinal layers initial paths =
      final.tokens := by
    simpa [layers, initial, ss] using
      ProofSearch.Grid.snapshots_getLast_tokens
        fs height paths final finalLast
  have finalOutputs : result.1 = final.outputs := by
    simpa [result, layers, initial, ss] using
      ProofSearch.Grid.snapshots_getLast_outputs
        fs height paths final finalLast
  have goalFormula : fs[goalColumn]? = some goal := by
    simpa [goalColumn] using List.getElem?_idxOf hgoal
  have lastLayer : layers.getLast? =
      some (ProofSearch.Grid.Definitions.layerFor fs) := by
    rw [List.getLast?_eq_getElem?]
    simp [layers, ProofSearch.Grid.Definitions.gridFor]
  have goalNode : (ProofSearch.Grid.Definitions.layerFor fs).nodes[goalColumn]? =
      some (nodeForFormula fs goal) := by
    simp [ProofSearch.Grid.Definitions.layerFor, goalFormula]
  have goalIncoming :
      (ProofSearch.Grid.Definitions.layerFor fs).incoming[goalColumn]? =
        some (buildIncomingMapForFormula fs goal) := by
    simp [ProofSearch.Grid.Definitions.layerFor, buildIncomingMap, goalFormula]
  have goalNonempty :
      ProofSearch.Grid.tokensInColumn final goalColumn ≠ [] := by
    have produced' := produced
    change ProofSearch.Grid.Definitions.goalProducedB layers initial paths goalColumn = true
      at produced'
    unfold ProofSearch.Grid.Definitions.goalProducedB at produced'
    rw [lastLayer] at produced'
    simp only at produced'
    rw [goalNode, goalIncoming] at produced'
    simp only [Bool.and_eq_true] at produced'
    have notEmpty := produced'.1.1
    rw [finalTokens] at notEmpty
    intro empty
    have filteredEmpty :
        final.tokens.filter (·.current_column = goalColumn) = [] := by
      simpa [ProofSearch.Grid.tokensInColumn] using empty
    rw [filteredEmpty] at notEmpty
    simp at notEmpty
  have linked : ProofSearch.Grid.SnapshotsLinked
      paths (height + 1) ss := by
    simpa [ss, ProofSearch.Grid.snapshots, layers, initial,
      ProofSearch.Grid.Definitions.gridFor] using
      ProofSearch.Grid.snapshotsFrom_linked paths (height + 1) height
        (initialize_tokens initial (height + 1)) layers
  have sameLayers : ∀ snapshot ∈ ss,
      snapshot.layer = ProofSearch.Grid.Definitions.layerFor fs := by
    intro snapshot member
    have memberLayer := ProofSearch.Grid.snapshotsFrom_layers
      paths (height + 1) height
      (initialize_tokens initial (height + 1)) layers snapshot
      (by simpa [ss, ProofSearch.Grid.snapshots, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using member)
    simpa [layers, ProofSearch.Grid.Definitions.gridFor] using memberLayer
  have evaluated : ∀ snapshot ∈ ss,
      snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2 := by
    intro snapshot member
    exact ProofSearch.Grid.snapshotsFrom_evaluated
      paths (height + 1) height
      (initialize_tokens initial (height + 1)) layers snapshot
      (by simpa [ss, ProofSearch.Grid.snapshots, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using member)
  have errors : ∀ snapshot ∈ ss, snapshot.error = false := by
    intro snapshot member
    have resultNoError :
        (eval_from_level paths height
          (initialize_tokens initial (height + 1)) layers false
          (height + 1)).2 = false := by
      simpa [result, get_eval_result, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using noError
    exact ProofSearch.Grid.eval_false_implies_snapshots_no_error
      paths (height + 1) height
      (initialize_tokens initial (height + 1)) layers false resultNoError
      snapshot (by simpa [ss, ProofSearch.Grid.snapshots, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using member)
  have base : ∀ snapshot column target,
      ss[0]? = some snapshot →
      fs[column]? = some target →
      ProofSearch.Grid.tokensInColumn snapshot column ≠ [] →
      ∃ hout : column < snapshot.outputs.length,
        depVector fs [target] = snapshot.outputs.get ⟨column, hout⟩ := by
    intro snapshot column target snapshotAt formulaAt nonempty
    apply ProofSearch.Grid.snapshots_base_output
      fs hfs height paths snapshot column target
      (by simpa [ss] using snapshotAt) formulaAt nonempty
    have snapshotBound : 0 < ss.length := by omega
    have snapshotMem : snapshot ∈ ss := by
      rw [List.mem_iff_getElem]
      exact ⟨0, snapshotBound,
        List.getElem?_eq_some_iff.mp snapshotAt |>.2⟩
    exact errors snapshot snapshotMem
  obtain ⟨extractedTree, extractedTreeAt, _, _, _, _, outputEq⟩ :=
    ProofSearch.Grid.extractAt_structurally_sound
      fs paths (height + 1) ss linked sameLayers evaluated errors base
      height goalColumn final goal finalAt goalFormula goalNonempty
  have sameTree : extractedTree = tree := by
    have extracted' :
        ProofSearch.Grid.extractAt fs ss height goalColumn = some tree := by
      simpa [ProofSearch.Grid.extractGoalCone, ss, goalColumn] using extracted
    rw [extracted'] at extractedTreeAt
    exact Option.some.inj extractedTreeAt.symm
  subst extractedTree
  obtain ⟨goalOut, dependencyOutput⟩ := outputEq
  have dischargedFinal :
      (final.outputs.get ⟨goalColumn, goalOut⟩).toList.all
        (· = false) = true := by
    rw [← dependencyOutput, closed]
    simp [depVector]
  have dischargedResult :
      (match result.1[goalColumn]? with
       | none => false
       | some vector => vector.toList.all (· = false)) = true := by
    rw [finalOutputs, List.getElem?_eq_getElem goalOut]
    exact dischargedFinal
  refine ⟨assignment, extracted, ?_⟩
  unfold ProofSearch.Grid.gridMarkedB ProofSearch.Grid.Definitions.markedBFor
  change (ProofSearch.Grid.Definitions.goalProducedB layers initial paths goalColumn &&
      !result.2 &&
      (match result.1[goalColumn]? with
       | none => false
       | some vector => vector.toList.all (· = false))) = true
  have produced' :
      ProofSearch.Grid.Definitions.goalProducedB layers initial paths goalColumn = true := by
    simpa [layers, initial, goalColumn] using produced
  simp only [Bool.and_eq_true]
  refine ⟨⟨produced', ?_⟩, ?_⟩
  · have noError' : result.2 = false := by
      simpa [result, layers, initial] using noError
    simp [noError']
  · simpa [result, layers, initial] using dischargedResult

/-- Exact existential characterization of the current search space.  The
right-hand side quantifies over *schedulable* closed trees, not all closed ND
trees; this is precisely where unrestricted proof-search completeness fails
to follow from the existing development. -/
theorem exists_marked_iff_exists_closed_gridSchedulable
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height : Nat) :
    (∃ assignment : ProofSearch.Grid.Counting.Assignment fs height,
        ProofSearch.Grid.gridMarkedB
          fs goal height assignment.paths = true) ↔
      ∃ tree : ProofSearch.Grid.NDTree,
        tree.wellFormedB = true ∧
          tree.conclusion = goal ∧
          tree.dependencies = [] ∧
          tree.height ≤ height ∧
          GridSchedulable fs goal height tree := by
  constructor
  · rintro ⟨assignment, marked⟩
    let paths := assignment.paths
    obtain ⟨tree, extracted, wellFormed, conclusion, closed, treeHeight⟩ :=
      ProofSearch.Grid.gridMarked_extracts
        fs hfs goal hgoal height paths marked
    let layers := ProofSearch.Grid.Definitions.gridFor fs height
    let initial := ProofSearch.Grid.Definitions.initialVectorsFor fs
    let goalColumn := ProofSearch.Grid.Definitions.goalColumnFor fs goal
    let result := get_eval_result layers initial paths
    have marked' := marked
    change (ProofSearch.Grid.Definitions.goalProducedB layers initial paths goalColumn &&
        !result.2 &&
        (match result.1[goalColumn]? with
         | none => false
         | some vector => vector.toList.all (· = false))) = true at marked'
    simp only [Bool.and_eq_true] at marked'
    obtain ⟨⟨produced, notError⟩, _⟩ := marked'
    have noError : result.2 = false := by
      cases value : result.2 <;> simp_all
    refine ⟨tree, wellFormed, conclusion, closed, treeHeight, assignment,
      ?_, ?_, extracted⟩
    · simpa [layers, initial, goalColumn] using produced
    · simpa [result, layers, initial] using noError
  · rintro ⟨tree, _wellFormed, _conclusion, closed, _treeHeight, scheduled⟩
    obtain ⟨assignment, _extracted, marked⟩ :=
      gridMarked_of_gridSchedulable fs hfs goal hgoal height tree scheduled closed
    exact ⟨assignment, marked⟩

/-- A closed schedulable derivation supplies the marked anchor required by the
existing unread-cell density theorem.  The exponent remains the anchor's
actual `readCount`; no uniform small-read bound is claimed. -/
theorem gridSchedulable_density_lower_bound
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height : Nat)
    (tree : ProofSearch.Grid.NDTree)
    (scheduled : GridSchedulable fs goal height tree)
    (closed : tree.dependencies = []) :
    ∃ assignment : ProofSearch.Grid.Counting.Assignment fs height,
      ProofSearch.Grid.extractGoalCone
          fs goal height assignment.paths = some tree ∧
        ProofSearch.Grid.gridMarkedB
            fs goal height assignment.paths = true ∧
        (1 : ℚ) /
            (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
              (ProofSearch.Grid.Counting.readCount fs goal height assignment) ≤
          ProofSearch.Grid.Counting.markedDensity fs goal height := by
  obtain ⟨assignment, extracted, marked⟩ :=
    gridMarked_of_gridSchedulable
      fs hfs goal hgoal height tree scheduled closed
  exact ⟨assignment, extracted, marked,
    ProofSearch.Grid.Counting.markedDensity_lower_bound
      fs goal height assignment marked⟩

end ProofSearch.Grid.Schedulability
