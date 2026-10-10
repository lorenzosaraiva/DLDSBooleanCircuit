import ProofSearch.Grid.Soundness
import Evaluator.UniversalAcceptance
import Mathlib.Data.Fintype.Card

/-!
# Space A unread-cell counting

The coordinates recorded here are *path-input* coordinates
`(token.origin_column, stepIndex)`.  A live token may have moved to another
formula column, but `propagate_tokens` continues to read the row selected by
its fixed `origin_column`.

The cell containing a stop instruction is read.  No later coordinate in that
row is recorded, since the token is no longer present.  At the final grid
layer no path instruction is read.
-/

open DLDS

namespace ProofSearch.Grid.Counting

abbrev CellAddress := Nat × Nat

/-- Raw list of path-input cells dereferenced by the evaluator.  Duplicates
are removed by `readCells` below. -/
def readCellsFrom {n : Nat} (paths : PathInput) (numLevels : Nat) :
    Nat → List (Token n) → List (GridLayer n) → List CellAddress
  | _, _, [] => []
  | _, _, [_] => []
  | level, tokens, layer :: next :: rest =>
      let evaluated := evaluate_layer layer tokens
      let stepIndex := numLevels - level - 1
      let here := tokens.map fun token => (token.origin_column, stepIndex)
      let nextTokens :=
        propagate_tokens tokens paths level numLevels evaluated.1
      here ++ readCellsFrom paths numLevels (level - 1) nextTokens (next :: rest)

/-- The finite set of routing cells actually inspected while evaluating
`paths`.  `goal` is an explicit parameter so the presentation has the same
indexing as the Space A decision predicate; operationally the forward read
set does not depend on which final column is designated as the goal. -/
def readCells (fs : List Formula) (_goal : Formula) (height : Nat)
    (paths : PathInput) : Finset CellAddress :=
  let layers := ProofSearch.Grid.Definitions.gridFor fs height
  let numLevels := layers.length
  let initial := initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs) numLevels
  (readCellsFrom paths numLevels (numLevels - 1) initial layers).toFinset

/-- Equality of every effective path instruction in a finite read set.
All zero-target instructions have the same effective value (`none`). -/
def AgreeOn (cells : Finset CellAddress) (paths paths' : PathInput) : Prop :=
  ∀ cell ∈ cells,
    effStep paths cell.1 cell.2 = effStep paths' cell.1 cell.2

/-- The literal path-input entry at an address, when it exists. -/
def pathCell? (paths : PathInput) (cell : CellAddress) : Option (Nat × Nat) :=
  paths[cell.1]?.bind fun column => column[cell.2]?

/-- Literal agreement on the cells in `cells`.  This is the hypothesis stated
in the presentation theorem; `AgreeOn` above is a slightly weaker operational
notion because all zero-target instructions stop the token. -/
def CellAgreeOn (cells : Finset CellAddress) (paths paths' : PathInput) : Prop :=
  ∀ cell ∈ cells, pathCell? paths cell = pathCell? paths' cell

lemma effStep_eq_pathCell (paths : PathInput) (cell : CellAddress) :
    effStep paths cell.1 cell.2 =
      match pathCell? paths cell with
      | none => none
      | some step => if step.1 = 0 then none else some step := by
  unfold effStep pathCell?
  cases paths[cell.1]? <;> rfl

lemma CellAgreeOn.toAgreeOn {cells : Finset CellAddress} {paths paths' : PathInput}
    (agree : CellAgreeOn cells paths paths') : AgreeOn cells paths paths' := by
  intro cell member
  rw [effStep_eq_pathCell, effStep_eq_pathCell, agree cell member]

lemma propagate_tokens_eq_of_agree {n : Nat} {paths paths' : PathInput}
    (tokens : List (Token n)) (level numLevels : Nat)
    (outputs : List (List.Vector Bool n))
    (agree : ∀ token ∈ tokens,
      effStep paths token.origin_column (numLevels - level - 1) =
        effStep paths' token.origin_column (numLevels - level - 1)) :
    propagate_tokens tokens paths level numLevels outputs =
      propagate_tokens tokens paths' level numLevels outputs := by
  rw [propagate_tokens_effStep, propagate_tokens_effStep]
  apply List.filterMap_congr
  intro token member
  rw [agree token member]

/-- Local read-set congruence for the evaluator recursion. -/
theorem eval_from_level_congr_read {n : Nat} {paths paths' : PathInput}
    (numLevels : Nat) :
    ∀ (layers : List (GridLayer n)) (level : Nat) (tokens : List (Token n))
      (error : Bool),
      AgreeOn (readCellsFrom paths numLevels level tokens layers).toFinset
        paths paths' →
      eval_from_level paths level tokens layers error numLevels =
        eval_from_level paths' level tokens layers error numLevels := by
  intro layers
  induction layers with
  | nil =>
      intro level tokens error _
      rfl
  | cons layer rest ih =>
      intro level tokens error agree
      cases rest with
      | nil => rfl
      | cons next tail =>
          let outputs := (evaluate_layer layer tokens).1
          have agreeHere : ∀ token ∈ tokens,
              effStep paths token.origin_column (numLevels - level - 1) =
                effStep paths' token.origin_column (numLevels - level - 1) := by
            intro token member
            apply agree (token.origin_column, numLevels - level - 1)
            apply List.mem_toFinset.mpr
            simp only [readCellsFrom, List.mem_append, List.mem_map]
            exact Or.inl ⟨token, member, rfl⟩
          have propagation := propagate_tokens_eq_of_agree
            tokens level numLevels outputs agreeHere
          have agreeTail :
              AgreeOn
                (readCellsFrom paths numLevels (level - 1)
                  (propagate_tokens tokens paths level numLevels outputs)
                  (next :: tail)).toFinset paths paths' := by
            intro cell member
            apply agree cell
            apply List.mem_toFinset.mpr
            simp only [readCellsFrom, List.mem_append, List.mem_map]
            exact Or.inr (List.mem_toFinset.mp (by simpa [outputs] using member))
          simp only [eval_from_level]
          rw [← propagation]
          exact ih (level - 1) _ _ agreeTail

theorem get_eval_result_congr_read {n : Nat} {paths paths' : PathInput}
    (layers : List (GridLayer n)) (initialVectors : List (List.Vector Bool n))
    (agree : AgreeOn
      (readCellsFrom paths layers.length (layers.length - 1)
        (initialize_tokens initialVectors layers.length) layers).toFinset
      paths paths') :
    get_eval_result layers initialVectors paths =
      get_eval_result layers initialVectors paths' := by
  unfold get_eval_result
  exact eval_from_level_congr_read layers.length layers (layers.length - 1)
    (initialize_tokens initialVectors layers.length) false agree

/-- The final live-token state has the same local read-set invariance. -/
theorem tokensAtFinal_go_congr_read {n : Nat} {paths paths' : PathInput}
    (numLevels : Nat) :
    ∀ (layers : List (GridLayer n)) (level : Nat) (tokens : List (Token n)),
      AgreeOn (readCellsFrom paths numLevels level tokens layers).toFinset
        paths paths' →
      ProofSearch.Grid.Definitions.tokensAtFinal.go paths numLevels level tokens layers =
        ProofSearch.Grid.Definitions.tokensAtFinal.go paths' numLevels level tokens layers := by
  intro layers
  induction layers with
  | nil =>
      intro level tokens _
      rfl
  | cons layer rest ih =>
      intro level tokens agree
      cases rest with
      | nil => rfl
      | cons next tail =>
          let outputs := (evaluate_layer layer tokens).1
          have agreeHere : ∀ token ∈ tokens,
              effStep paths token.origin_column (numLevels - level - 1) =
                effStep paths' token.origin_column (numLevels - level - 1) := by
            intro token member
            apply agree (token.origin_column, numLevels - level - 1)
            apply List.mem_toFinset.mpr
            simp only [readCellsFrom, List.mem_append, List.mem_map]
            exact Or.inl ⟨token, member, rfl⟩
          have propagation := propagate_tokens_eq_of_agree
            tokens level numLevels outputs agreeHere
          have agreeTail :
              AgreeOn
                (readCellsFrom paths numLevels (level - 1)
                  (propagate_tokens tokens paths level numLevels outputs)
                  (next :: tail)).toFinset paths paths' := by
            intro cell member
            apply agree cell
            apply List.mem_toFinset.mpr
            simp only [readCellsFrom, List.mem_append, List.mem_map]
            exact Or.inr (List.mem_toFinset.mp (by simpa [outputs] using member))
          simp only [ProofSearch.Grid.Definitions.tokensAtFinal.go]
          rw [← propagation]
          exact ih (level - 1) _ agreeTail

theorem tokensAtFinal_congr_read {n : Nat} {paths paths' : PathInput}
    (layers : List (GridLayer n)) (initialVectors : List (List.Vector Bool n))
    (agree : AgreeOn
      (readCellsFrom paths layers.length (layers.length - 1)
        (initialize_tokens initialVectors layers.length) layers).toFinset
      paths paths') :
    ProofSearch.Grid.Definitions.tokensAtFinal layers initialVectors paths =
      ProofSearch.Grid.Definitions.tokensAtFinal layers initialVectors paths' := by
  unfold ProofSearch.Grid.Definitions.tokensAtFinal
  exact tokensAtFinal_go_congr_read layers.length layers (layers.length - 1)
    (initialize_tokens initialVectors layers.length) agree

/-- All extraction snapshots are invariant under changes outside the read
set. -/
theorem snapshotsFrom_congr_read {n : Nat} {paths paths' : PathInput}
    (numLevels : Nat) :
    ∀ (layers : List (GridLayer n)) (level : Nat) (tokens : List (Token n)),
      AgreeOn (readCellsFrom paths numLevels level tokens layers).toFinset
        paths paths' →
      ProofSearch.Grid.snapshotsFrom paths numLevels level tokens layers =
        ProofSearch.Grid.snapshotsFrom paths' numLevels level tokens layers := by
  intro layers
  induction layers with
  | nil =>
      intro level tokens _
      rfl
  | cons layer rest ih =>
      intro level tokens agree
      cases rest with
      | nil => rfl
      | cons next tail =>
          let outputs := (evaluate_layer layer tokens).1
          have agreeHere : ∀ token ∈ tokens,
              effStep paths token.origin_column (numLevels - level - 1) =
                effStep paths' token.origin_column (numLevels - level - 1) := by
            intro token member
            apply agree (token.origin_column, numLevels - level - 1)
            apply List.mem_toFinset.mpr
            simp only [readCellsFrom, List.mem_append, List.mem_map]
            exact Or.inl ⟨token, member, rfl⟩
          have propagation := propagate_tokens_eq_of_agree
            tokens level numLevels outputs agreeHere
          have agreeTail :
              AgreeOn
                (readCellsFrom paths numLevels (level - 1)
                  (propagate_tokens tokens paths level numLevels outputs)
                  (next :: tail)).toFinset paths paths' := by
            intro cell member
            apply agree cell
            apply List.mem_toFinset.mpr
            simp only [readCellsFrom, List.mem_append, List.mem_map]
            exact Or.inr (List.mem_toFinset.mp (by simpa [outputs] using member))
          change
            ({ level := level
               layer := layer
               tokens := tokens
               outputs := (evaluate_layer layer tokens).1
               error := (evaluate_layer layer tokens).2 } ::
              ProofSearch.Grid.snapshotsFrom paths numLevels (level - 1)
                (propagate_tokens tokens paths level numLevels outputs)
                (next :: tail)) =
            ({ level := level
               layer := layer
               tokens := tokens
               outputs := (evaluate_layer layer tokens).1
               error := (evaluate_layer layer tokens).2 } ::
              ProofSearch.Grid.snapshotsFrom paths' numLevels (level - 1)
                (propagate_tokens tokens paths' level numLevels outputs)
                (next :: tail))
          rw [← propagation]
          exact congrArg
            (fun tailSnapshots =>
              { level := level
                layer := layer
                tokens := tokens
                outputs := (evaluate_layer layer tokens).1
                error := (evaluate_layer layer tokens).2 } :: tailSnapshots)
            (ih (level - 1) _ agreeTail)

theorem snapshots_congr_read (fs : List Formula) (height : Nat)
    {paths paths' : PathInput}
    (agree : AgreeOn (readCells fs default height paths) paths paths') :
    ProofSearch.Grid.snapshots fs height paths =
      ProofSearch.Grid.snapshots fs height paths' := by
  unfold readCells at agree
  unfold ProofSearch.Grid.snapshots
  exact snapshotsFrom_congr_read _ _ _ _ agree

/-! ## Core unread-cell invariance -/

/-- Changing cells outside `readCells` cannot change the grid marker. -/
theorem gridMarkedB_unread_invariant (fs : List Formula) (goal : Formula)
    (height : Nat) {paths paths' : PathInput}
    (agree : AgreeOn (readCells fs goal height paths) paths paths') :
    ProofSearch.Grid.gridMarkedB fs goal height paths =
      ProofSearch.Grid.gridMarkedB fs goal height paths' := by
  unfold ProofSearch.Grid.gridMarkedB ProofSearch.Grid.Definitions.markedBFor
  let layers := ProofSearch.Grid.Definitions.gridFor fs height
  let initialVectors := ProofSearch.Grid.Definitions.initialVectorsFor fs
  have resultEq : get_eval_result layers initialVectors paths =
      get_eval_result layers initialVectors paths' := by
    apply get_eval_result_congr_read
    simpa [readCells, layers, initialVectors] using agree
  have finalEq : ProofSearch.Grid.Definitions.tokensAtFinal layers initialVectors paths =
      ProofSearch.Grid.Definitions.tokensAtFinal layers initialVectors paths' := by
    apply tokensAtFinal_congr_read
    simpa [readCells, layers, initialVectors] using agree
  simp only
  rw [resultEq]
  unfold ProofSearch.Grid.Definitions.goalProducedB
  rw [finalEq]

/-- Literal-cell presentation of unread-cell invariance. -/
theorem gridMarkedB_unread_cell_invariant (fs : List Formula) (goal : Formula)
    (height : Nat) {paths paths' : PathInput}
    (agree : CellAgreeOn (readCells fs goal height paths) paths paths') :
    ProofSearch.Grid.gridMarkedB fs goal height paths =
      ProofSearch.Grid.gridMarkedB fs goal height paths' :=
  gridMarkedB_unread_invariant fs goal height agree.toAgreeOn

/-- The extracted proof object is invariant for the same reason. -/
theorem extractGoalCone_unread_invariant (fs : List Formula) (goal : Formula)
    (height : Nat) {paths paths' : PathInput}
    (agree : AgreeOn (readCells fs goal height paths) paths paths') :
    ProofSearch.Grid.extractGoalCone fs goal height paths =
      ProofSearch.Grid.extractGoalCone fs goal height paths' := by
  unfold ProofSearch.Grid.extractGoalCone
  have snapshotsEq : ProofSearch.Grid.snapshots fs height paths =
      ProofSearch.Grid.snapshots fs height paths' := by
    unfold readCells at agree
    unfold ProofSearch.Grid.snapshots
    exact snapshotsFrom_congr_read _ _ _ _ agree
  rw [snapshotsEq]

/-- **Core lemma, certified form.** -/
theorem certifiedMarkedB_unread_invariant (fs : List Formula) (goal : Formula)
    (height : Nat) {paths paths' : PathInput}
    (agree : AgreeOn (readCells fs goal height paths) paths paths') :
    ProofSearch.Grid.certifiedMarkedB fs goal height paths =
      ProofSearch.Grid.certifiedMarkedB fs goal height paths' := by
  unfold ProofSearch.Grid.certifiedMarkedB
  rw [gridMarkedB_unread_invariant fs goal height agree]
  unfold ProofSearch.Grid.extractedCertificateB
  rw [extractGoalCone_unread_invariant fs goal height agree]

theorem CertifiedMarked_unread_invariant (fs : List Formula) (goal : Formula)
    (height : Nat) {paths paths' : PathInput}
    (agree : AgreeOn (readCells fs goal height paths) paths paths') :
    ProofSearch.Grid.CertifiedMarked fs goal height paths ↔
      ProofSearch.Grid.CertifiedMarked fs goal height paths' := by
  unfold ProofSearch.Grid.CertifiedMarked
  rw [certifiedMarkedB_unread_invariant fs goal height agree]

/-- Literal-cell presentation for the propositional certified marker. -/
theorem CertifiedMarked_unread_cell_invariant
    (fs : List Formula) (goal : Formula) (height : Nat)
    {paths paths' : PathInput}
    (agree : CellAgreeOn (readCells fs goal height paths) paths paths') :
    ProofSearch.Grid.CertifiedMarked fs goal height paths ↔
      ProofSearch.Grid.CertifiedMarked fs goal height paths' :=
  CertifiedMarked_unread_invariant fs goal height agree.toAgreeOn

/-! ## Finite carrier and orbit counting -/

/-- A rectangular carrier cell. -/
abbrev CarrierCell (fs : List Formula) (height : Nat) :=
  Fin fs.length × Fin height

/-- An alphabet symbol is represented by its position in the compiled step
alphabet. -/
abbrev AlphabetIndex (fs : List Formula) :=
  Fin (ProofSearch.Grid.Definitions.alphabetFor fs).length

/-- The finite, rectangular Space A carrier. -/
abbrev Assignment (fs : List Formula) (height : Nat) :=
  CarrierCell fs height → AlphabetIndex fs

/-- Decode a finite assignment into the list-of-lists representation consumed
by the evaluator. -/
def Assignment.paths {fs : List Formula} {height : Nat}
    (assignment : Assignment fs height) : PathInput :=
  List.ofFn fun column : Fin fs.length =>
    List.ofFn fun step : Fin height =>
      (ProofSearch.Grid.Definitions.alphabetFor fs).get (assignment (column, step))

/-- The read set transported to the finite rectangular carrier. -/
def assignmentReadCells (fs : List Formula) (goal : Formula) (height : Nat)
    (assignment : Assignment fs height) : Finset (CarrierCell fs height) :=
  Finset.univ.filter fun cell =>
    (cell.1.val, cell.2.val) ∈ readCells fs goal height assignment.paths

def readCount (fs : List Formula) (goal : Formula) (height : Nat)
    (assignment : Assignment fs height) : Nat :=
  (assignmentReadCells fs goal height assignment).card

/-- Assignments that agree with `assignment` on every one of its read cells.
The unread coordinates are the free coordinates of this orbit. -/
abbrev ReadOrbit (fs : List Formula) (goal : Formula) (height : Nat)
    (assignment : Assignment fs height) :=
  { other : Assignment fs height //
    ∀ cell ∈ assignmentReadCells fs goal height assignment,
      other cell = assignment cell }

/-- Marked elements of the finite rectangular carrier. -/
abbrev MarkedAssignments (fs : List Formula) (goal : Formula) (height : Nat) :=
  { assignment : Assignment fs height //
    ProofSearch.Grid.gridMarkedB fs goal height assignment.paths = true }

lemma assignment_paths_agree {fs : List Formula} {goal : Formula} {height : Nat}
    {assignment other : Assignment fs height}
    (agree : ∀ cell ∈ assignmentReadCells fs goal height assignment,
      other cell = assignment cell) :
    AgreeOn (readCells fs goal height assignment.paths)
      assignment.paths other.paths := by
  intro cell member
  by_cases columnBound : cell.1 < fs.length
  · by_cases stepBound : cell.2 < height
    · let bounded : CarrierCell fs height :=
        (⟨cell.1, columnBound⟩, ⟨cell.2, stepBound⟩)
      have boundedRead : bounded ∈
          assignmentReadCells fs goal height assignment := by
        simp [assignmentReadCells, bounded, member]
      have valueEq := agree bounded boundedRead
      simp [effStep, ProofSearch.Grid.Counting.Assignment.paths, columnBound, stepBound, bounded, valueEq]
    · simp [effStep, ProofSearch.Grid.Counting.Assignment.paths, columnBound, stepBound]
  · simp [effStep, ProofSearch.Grid.Counting.Assignment.paths, columnBound]

/-- The read orbit is equivalent to an arbitrary assignment on the complement
of the read set. -/
def readOrbitEquiv (fs : List Formula) (goal : Formula) (height : Nat)
    (assignment : Assignment fs height) :
    ReadOrbit fs goal height assignment ≃
      (↥(assignmentReadCells fs goal height assignment)ᶜ → AlphabetIndex fs) where
  toFun other cell := other.1 cell.1
  invFun free :=
    ⟨fun cell =>
        if member : cell ∈ assignmentReadCells fs goal height assignment then
          assignment cell
        else
          free ⟨cell, by simpa using member⟩,
      by
        intro cell member
        simp [member]⟩
  left_inv other := by
    apply Subtype.ext
    funext cell
    by_cases member : cell ∈ assignmentReadCells fs goal height assignment
    · simp [member, other.2 cell member]
    · simp [member]
  right_inv free := by
    funext cell
    have notMember : ¬ (cell.1 ∈
        assignmentReadCells fs goal height assignment) := by
      exact Finset.mem_compl.mp cell.2
    simp [notMember]

/-- **Exact orbit size.**  If `r` cells are read, there are exactly
`A^(N*h-r)` assignments agreeing with the marked trace on all read cells. -/
theorem readOrbit_card (fs : List Formula) (goal : Formula) (height : Nat)
    (assignment : Assignment fs height) :
    Fintype.card (ReadOrbit fs goal height assignment) =
      (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
        (fs.length * height - readCount fs goal height assignment) := by
  let read := assignmentReadCells fs goal height assignment
  calc
    Fintype.card (ReadOrbit fs goal height assignment) =
        Fintype.card (↥readᶜ → AlphabetIndex fs) :=
      Fintype.card_congr (readOrbitEquiv fs goal height assignment)
    _ = (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ readᶜ.card := by
      simp [AlphabetIndex, Finset.card_compl, CarrierCell]
    _ = (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
        (fs.length * height - readCount fs goal height assignment) := by
      have cardCarrier : Fintype.card (CarrierCell fs height) =
          fs.length * height := by simp [CarrierCell]
      rw [Finset.card_compl, cardCarrier]
      rfl

/-- Every read orbit of a marked assignment remains marked. -/
def readOrbitToMarked {fs : List Formula} {goal : Formula} {height : Nat}
    (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true) :
    ReadOrbit fs goal height assignment → MarkedAssignments fs goal height :=
  fun other =>
    ⟨other.1, by
      have agree := assignment_paths_agree other.2
      rw [← gridMarkedB_unread_invariant fs goal height agree]
      exact marked⟩

lemma readOrbitToMarked_injective {fs : List Formula} {goal : Formula}
    {height : Nat} (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true) :
    Function.Injective (readOrbitToMarked assignment marked) := by
  intro left right equal
  apply Subtype.ext
  exact congrArg
    (fun markedAssignment : MarkedAssignments fs goal height => markedAssignment.1)
    equal

/-- **Density numerator lower bound.**  A marked assignment with `r` read
cells contributes an entire orbit of at least `A^(N*h-r)` marked carrier
assignments. -/
theorem markedAssignments_lower_bound (fs : List Formula) (goal : Formula)
    (height : Nat) (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true) :
    (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
        (fs.length * height - readCount fs goal height assignment) ≤
      Fintype.card (MarkedAssignments fs goal height) := by
  rw [← readOrbit_card fs goal height assignment]
  exact Fintype.card_le_of_injective (readOrbitToMarked assignment marked)
    (readOrbitToMarked_injective assignment marked)

theorem markedAssignments_lower_bound_of_readCount_eq
    (fs : List Formula) (goal : Formula) (height r : Nat)
    (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true)
    (readCard : readCount fs goal height assignment = r) :
    (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
        (fs.length * height - r) ≤
      Fintype.card (MarkedAssignments fs goal height) := by
  simpa [readCard] using
    markedAssignments_lower_bound fs goal height assignment marked

/-- The exact finite carrier size is `A^(N*h)`. -/
theorem assignment_card (fs : List Formula) (height : Nat) :
    Fintype.card (Assignment fs height) =
      (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ (fs.length * height) := by
  simp [Assignment, CarrierCell, AlphabetIndex]

/-- Cross-multiplied density bound.  This is the natural-number form of
`density ≥ A^(-r)`. -/
theorem markedDensity_cross_bound (fs : List Formula) (goal : Formula)
    (height : Nat) (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true) :
    (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ (fs.length * height) ≤
      (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
          (readCount fs goal height assignment) *
        Fintype.card (MarkedAssignments fs goal height) := by
  let alphabetSize := (ProofSearch.Grid.Definitions.alphabetFor fs).length
  let r := readCount fs goal height assignment
  have readLe : r ≤ fs.length * height := by
    dsimp [r, readCount]
    calc
      (assignmentReadCells fs goal height assignment).card ≤
          Fintype.card (CarrierCell fs height) := Finset.card_le_univ _
      _ = fs.length * height := by simp [CarrierCell]
  have lower := markedAssignments_lower_bound fs goal height assignment marked
  change alphabetSize ^ (fs.length * height) ≤
    alphabetSize ^ r * Fintype.card (MarkedAssignments fs goal height)
  calc
    alphabetSize ^ (fs.length * height) =
        alphabetSize ^ r * alphabetSize ^ (fs.length * height - r) := by
      rw [← pow_add, Nat.add_sub_of_le readLe]
    _ ≤ alphabetSize ^ r * Fintype.card (MarkedAssignments fs goal height) :=
      Nat.mul_le_mul_left _ lower

def markedDensity (fs : List Formula) (goal : Formula) (height : Nat) : ℚ :=
  Fintype.card (MarkedAssignments fs goal height) /
    (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ (fs.length * height)

/-- Rational presentation of the same density bound. -/
theorem markedDensity_lower_bound (fs : List Formula) (goal : Formula)
    (height : Nat) (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true) :
    (1 : ℚ) /
        (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
          (readCount fs goal height assignment) ≤
      markedDensity fs goal height := by
  have alphabetPositive : 0 < (ProofSearch.Grid.Definitions.alphabetFor fs).length := by
    simp [ProofSearch.Grid.Definitions.alphabetFor]
  have leftDenominatorPositive :
      (0 : ℚ) < (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
        (readCount fs goal height assignment) := by positivity
  have rightDenominatorPositive :
      (0 : ℚ) < (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
        (fs.length * height) := by positivity
  unfold markedDensity
  apply (div_le_div_iff₀ leftDenominatorPositive rightDenominatorPositive).2
  norm_num
  have cross := markedDensity_cross_bound fs goal height assignment marked
  exact_mod_cast (show
    (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ (fs.length * height) ≤
      Fintype.card (MarkedAssignments fs goal height) *
        (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
          readCount fs goal height assignment by
    simpa [Nat.mul_comm] using cross)

theorem markedDensity_lower_bound_of_readCount_eq
    (fs : List Formula) (goal : Formula) (height r : Nat)
    (assignment : Assignment fs height)
    (marked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true)
    (readCard : readCount fs goal height assignment = r) :
    (1 : ℚ) / (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ r ≤
      markedDensity fs goal height := by
  simpa [readCard] using
    markedDensity_lower_bound fs goal height assignment marked

/-!
For a fixed derivation padded to a larger `height`, the goal-producing token
must remain live until the final layer.  Its repetition instructions are read,
so `readCount` grows with the padding height.  The lower bound therefore
decreases as `A^(-r)` even though all cells outside the live prefixes remain
free.

The following combined theorem contains only the formal density assertion.
The usual square-root-search estimate `sqrt(A^r) = A^(r/2)` is a cited query
bound; no search algorithm, BBHT, amplitudes, or quantum semantics are defined
here.
-/

/-- A derivation encoded by a marked assignment inherits the read-orbit
density bound.  `encoded` records the requested derivation but is not needed
again after certification of the assignment. -/
theorem encodedDerivation_density
    (fs : List Formula) (goal : Formula) (height : Nat)
    (tree : ProofSearch.Grid.NDTree)
    (assignment : Assignment fs height)
    (_encoded : ProofSearch.Grid.extractGoalCone
      fs goal height assignment.paths = some tree)
    (marked : ProofSearch.Grid.CertifiedMarked
      fs goal height assignment.paths) :
    (1 : ℚ) /
        (ProofSearch.Grid.Definitions.alphabetFor fs).length ^
          (readCount fs goal height assignment) ≤
      markedDensity fs goal height := by
  have gridMarked : ProofSearch.Grid.gridMarkedB
      fs goal height assignment.paths = true := by
    unfold ProofSearch.Grid.CertifiedMarked
      ProofSearch.Grid.certifiedMarkedB at marked
    have parts :
        ProofSearch.Grid.gridMarkedB fs goal height assignment.paths = true ∧
          ProofSearch.Grid.extractedCertificateB
            fs goal height assignment.paths = true := by
      simpa only [Bool.and_eq_true] using marked
    exact parts.1
  exact markedDensity_lower_bound fs goal height assignment gridMarked

theorem encodedDerivation_density_of_readCount_eq
    (fs : List Formula) (goal : Formula) (height r : Nat)
    (tree : ProofSearch.Grid.NDTree)
    (assignment : Assignment fs height)
    (encoded : ProofSearch.Grid.extractGoalCone
      fs goal height assignment.paths = some tree)
    (marked : ProofSearch.Grid.CertifiedMarked
      fs goal height assignment.paths)
    (readCard : readCount fs goal height assignment = r) :
    (1 : ℚ) / (ProofSearch.Grid.Definitions.alphabetFor fs).length ^ r ≤
      markedDensity fs goal height := by
  simpa [readCard] using
    encodedDerivation_density fs goal height tree assignment encoded marked

/-! ## Exact census validation -/

/-- Sum `number of active traces * A^unread` over a complete unread-cell
histogram. -/
def weightedTraceSum (alphabetSize : Nat) (histogram : List (Nat × Nat)) : Nat :=
  histogram.foldl (fun total entry =>
    total + entry.2 * alphabetSize ^ entry.1) 0

def censusHistogramH3 : List (Nat × Nat) :=
  [(0, 6), (1, 34), (2, 80), (3, 57), (4, 40), (5, 11), (6, 3)]

def censusHistogramH4 : List (Nat × Nat) :=
  [(0, 6), (1, 34), (2, 92), (3, 223), (4, 206),
   (5, 174), (6, 119), (7, 44), (8, 18), (9, 4)]

def censusHistogramH5 : List (Nat × Nat) :=
  [(0, 6), (1, 34), (2, 92), (3, 235), (4, 477), (5, 547),
   (6, 561), (7, 475), (8, 333), (9, 153), (10, 72), (11, 27), (12, 5)]

/-- The complete active-trace census contains 231, 920, and 3017 traces. -/
theorem census_active_trace_totals :
    (censusHistogramH3.foldl (fun n entry => n + entry.2) 0,
     censusHistogramH4.foldl (fun n entry => n + entry.2) 0,
     censusHistogramH5.foldl (fun n entry => n + entry.2) 0) =
      (231, 920, 3017) := by
  native_decide

/-- Executable validation of `sum_T A^(unread T)` against all three exact
census totals.  Here `A = 11`. -/
theorem census_weighted_sums :
    (weightedTraceSum 11 censusHistogramH3,
     weightedTraceSum 11 censusHistogramH4,
     weightedTraceSum 11 censusHistogramH5) =
      (7_757_811, 14_389_849_150, 25_705_544_720_367) := by
  native_decide

end ProofSearch.Grid.Counting
