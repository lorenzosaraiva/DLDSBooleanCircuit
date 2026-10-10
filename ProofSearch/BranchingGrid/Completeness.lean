import ProofSearch.BranchingGrid.Soundness

/-!
# A structural ND-tree compiler for Branching Grid V2

This module gives the explicit finite encoding used by the completeness
analysis.  Formula carriers `0, ..., fs.length - 1` remain permanent sources.
The compiler allocates one further carrier for every node of the ND tree and
emits the tree in postorder:

* a hypothesis forks its permanent source into its node carrier;
* an introduction consumes the premise-root carrier;
* an elimination consumes the two child-root carriers.

Consequently the generated program has exactly `tree.nodeCount` rows and uses
the safe carrier budget `fs.length + tree.nodeCount`.
-/

open DLDS

namespace ProofSearch.Grid

def NDTree.nodeCount : NDTree -> Nat
  | .hyp _ => 1
  | .intro _ _ premise => premise.nodeCount + 1
  | .elim _ _ major minor => major.nodeCount + minor.nodeCount + 1

theorem NDTree.nodeCount_pos (tree : NDTree) : 0 < tree.nodeCount := by
  induction tree <;> simp [ProofSearch.Grid.NDTree.nodeCount, *]

/-- A deliberately explicit formula-universe requirement.  It records every
formula stored in an ND constructor, including each implication assembled by
an introduction or expected by an elimination. -/
def NDTree.formulas : NDTree -> List Formula
  | .hyp formula => [formula]
  | .intro alpha beta premise =>
      alpha :: beta :: (alpha >> beta) :: premise.formulas
  | .elim alpha beta major minor =>
      alpha :: beta :: (alpha >> beta) :: (major.formulas ++ minor.formulas)

end ProofSearch.Grid

namespace DLDS
namespace ProofSearch.BranchingGrid

open ProofSearch.Grid

def FormulaCovered (fs : List Formula) (tree : NDTree) : Prop :=
  forall formula, formula ∈ tree.formulas -> formula ∈ fs

def formulaColumnOf (fs : List Formula) {formula : Formula}
    (member : formula ∈ fs) : Fin fs.length :=
  { val := fs.idxOf formula
    isLt := List.idxOf_lt_length_of_mem member }

def liftFormulaCarrier (fs : List Formula) (total : Nat)
    (column : Fin fs.length) : Fin (fs.length + total) :=
  { val := column.val, isLt := by omega }

def sourceCarrierOf (fs : List Formula) (total : Nat) {formula : Formula}
    (member : formula ∈ fs) : Fin (fs.length + total) :=
  { val := fs.idxOf formula
    isLt := by
      have bound := List.idxOf_lt_length_of_mem member
      omega }

/-- The last carrier in a nonempty allocated node interval. -/
def allocatedRoot (fs : List Formula) (total offset count : Nat)
    (positive : 0 < count) (bound : offset + count ≤ total) :
    Fin (fs.length + total) :=
  { val := fs.length + offset + count - 1
    isLt := by omega }

theorem FormulaCovered.hyp {fs : List Formula} {formula : Formula}
    (covered : FormulaCovered fs (.hyp formula)) : formula ∈ fs := by
  exact covered formula (by simp [ProofSearch.Grid.NDTree.formulas])

theorem FormulaCovered.intro_premise {fs : List Formula}
    {alpha beta : Formula} {premise : NDTree}
    (covered : FormulaCovered fs (.intro alpha beta premise)) :
    FormulaCovered fs premise := by
  intro formula member
  exact covered formula (by
    simp [ProofSearch.Grid.NDTree.formulas, member])

theorem FormulaCovered.intro_antecedent {fs : List Formula}
    {alpha beta : Formula} {premise : NDTree}
    (covered : FormulaCovered fs (.intro alpha beta premise)) : alpha ∈ fs := by
  exact covered alpha (by simp [ProofSearch.Grid.NDTree.formulas])

theorem FormulaCovered.intro_conclusion {fs : List Formula}
    {alpha beta : Formula} {premise : NDTree}
    (covered : FormulaCovered fs (.intro alpha beta premise)) :
    alpha >> beta ∈ fs := by
  exact covered (alpha >> beta) (by
    simp [ProofSearch.Grid.NDTree.formulas])

theorem FormulaCovered.elim_major {fs : List Formula}
    {alpha beta : Formula} {major minor : NDTree}
    (covered : FormulaCovered fs (.elim alpha beta major minor)) :
    FormulaCovered fs major := by
  intro formula member
  exact covered formula (by
    simp [ProofSearch.Grid.NDTree.formulas, member])

theorem FormulaCovered.elim_minor {fs : List Formula}
    {alpha beta : Formula} {major minor : NDTree}
    (covered : FormulaCovered fs (.elim alpha beta major minor)) :
    FormulaCovered fs minor := by
  intro formula member
  exact covered formula (by
    simp [ProofSearch.Grid.NDTree.formulas, member])

theorem FormulaCovered.elim_conclusion {fs : List Formula}
    {alpha beta : Formula} {major minor : NDTree}
    (covered : FormulaCovered fs (.elim alpha beta major minor)) : beta ∈ fs := by
  exact covered beta (by simp [ProofSearch.Grid.NDTree.formulas])

theorem FormulaCovered.conclusion {fs : List Formula} {tree : NDTree}
    (covered : FormulaCovered fs tree) : tree.conclusion ∈ fs := by
  cases tree with
  | hyp formula => exact covered.hyp
  | intro alpha beta premise => exact covered.intro_conclusion
  | elim alpha beta major minor => exact covered.elim_conclusion

/-- Proof-carrying version of the structural compiler.  Unlike
`compileProgram`, this function has no `idle` fallback: formula coverage and
the allocation bound construct every finite index directly. -/
def compileCertifiedFrom (fs : List Formula) (total offset : Nat)
    (tree : NDTree) (covered : FormulaCovered fs tree)
    (bound : offset + tree.nodeCount ≤ total) :
    List (BranchInstruction fs.length (fs.length + total)) :=
  match tree with
  | .hyp formula =>
      let column := formulaColumnOf fs covered.hyp
      let source := liftFormulaCarrier fs total column
      let root := allocatedRoot fs total offset 1 (by omega) (by
        simpa [ProofSearch.Grid.NDTree.nodeCount] using bound)
      [.fork source root]
  | .intro alpha beta premise =>
      let premiseProgram := compileCertifiedFrom fs total offset premise
        covered.intro_premise (by
          have full : offset + (premise.nodeCount + 1) ≤ total := by
            simpa [ProofSearch.Grid.NDTree.nodeCount] using bound
          omega)
      let premiseRoot := allocatedRoot fs total offset premise.nodeCount
        premise.nodeCount_pos (by
          have full : offset + (premise.nodeCount + 1) ≤ total := by
            simpa [ProofSearch.Grid.NDTree.nodeCount] using bound
          omega)
      let resultRoot := allocatedRoot fs total offset
        (premise.nodeCount + 1) (by omega) (by
          simpa [ProofSearch.Grid.NDTree.nodeCount] using bound)
      premiseProgram ++
        [.intro premiseRoot resultRoot
          (formulaColumnOf fs covered.intro_antecedent)
          (formulaColumnOf fs covered.intro_conclusion)]
  | .elim alpha beta major minor =>
      let majorProgram := compileCertifiedFrom fs total offset major
        covered.elim_major (by
          have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤
              total := by
            simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc]
              using bound
          have positive := minor.nodeCount_pos
          omega)
      let minorProgram := compileCertifiedFrom fs total
        (offset + major.nodeCount) minor covered.elim_minor (by
          have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤
              total := by
            simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc]
              using bound
          omega)
      let majorRoot := allocatedRoot fs total offset major.nodeCount
        major.nodeCount_pos (by
          have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤
              total := by
            simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc]
              using bound
          have positive := minor.nodeCount_pos
          omega)
      let minorRoot := allocatedRoot fs total (offset + major.nodeCount)
        minor.nodeCount minor.nodeCount_pos (by
          have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤
              total := by
            simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc]
              using bound
          omega)
      let resultRoot := allocatedRoot fs total offset
        (major.nodeCount + minor.nodeCount + 1) (by omega) (by
          simpa [ProofSearch.Grid.NDTree.nodeCount] using bound)
      majorProgram ++ minorProgram ++
        [.elim majorRoot minorRoot resultRoot
          (formulaColumnOf fs covered.elim_conclusion)]

def compileCertified (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree) :
    List (BranchInstruction fs.length (fs.length + tree.nodeCount)) :=
  compileCertifiedFrom fs tree.nodeCount 0 tree covered (by simp)

theorem compileCertifiedFrom_length (fs : List Formula) (total offset : Nat)
    (tree : NDTree) (covered : FormulaCovered fs tree)
    (bound : offset + tree.nodeCount ≤ total) :
    (compileCertifiedFrom fs total offset tree covered bound).length =
      tree.nodeCount := by
  induction tree generalizing offset with
  | hyp formula => simp [compileCertifiedFrom, ProofSearch.Grid.NDTree.nodeCount]
  | intro alpha beta premise ih =>
      simp [compileCertifiedFrom, ProofSearch.Grid.NDTree.nodeCount, ih]
  | elim alpha beta major minor ihMajor ihMinor =>
      simp [compileCertifiedFrom, ProofSearch.Grid.NDTree.nodeCount,
        ihMajor, ihMinor, Nat.add_assoc]

theorem compileCertified_length (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree) :
    (compileCertified fs tree covered).length = tree.nodeCount := by
  exact compileCertifiedFrom_length fs tree.nodeCount 0 tree covered (by simp)

def encodeCertifiedNDTree (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree) :
    OccurrenceAssignment fs.length tree.nodeCount (fs.length + tree.nodeCount) :=
  fun row => (compileCertified fs tree covered).get
    (Fin.cast (compileCertified_length fs tree covered).symm row)

@[simp] theorem occurrenceProgram_encodeCertifiedNDTree
    (fs : List Formula) (tree : NDTree) (covered : FormulaCovered fs tree) :
    occurrenceProgram (encodeCertifiedNDTree fs tree covered) =
      compileCertified fs tree covered := by
  let program := compileCertified fs tree covered
  let lengthEq : program.length = tree.nodeCount :=
    compileCertified_length fs tree covered
  calc
    occurrenceProgram (encodeCertifiedNDTree fs tree covered) =
        List.ofFn (fun row : Fin tree.nodeCount =>
          program.get (Fin.cast lengthEq.symm row)) := by
            rfl
    _ = List.ofFn program.get := (List.ofFn_congr lengthEq program.get).symm
    _ = program := List.ofFn_get program
    _ = compileCertified fs tree covered := rfl

/-! ## Operational compiler specification -/

def CarriesAt (fs : List Formula) {K : Nat}
    (state : BranchState fs.length K) (carrier : Fin K)
    (column : Fin fs.length) (tree : NDTree) : Prop :=
  exists level, state carrier = some (mkToken fs carrier column level tree)

def SourcesReady (fs : List Formula) (total : Nat)
    (state : BranchState fs.length (fs.length + total)) : Prop :=
  forall column, CarriesAt fs state (liftFormulaCarrier fs total column)
    column (.hyp (fs.get column))

def InAllocatedInterval (fs : List Formula) (offset count : Nat)
    {total : Nat} (carrier : Fin (fs.length + total)) : Prop :=
  fs.length + offset ≤ carrier.val /\
    carrier.val < fs.length + offset + count

def IntervalEmpty (fs : List Formula) (offset count : Nat)
    {total : Nat} (state : BranchState fs.length (fs.length + total)) : Prop :=
  forall carrier, InAllocatedInterval fs offset count carrier ->
    state carrier = none

def PayloadPreservedAt (fs : List Formula) {K : Nat}
    (before after : BranchState fs.length K) (carrier : Fin K) : Prop :=
  (forall column tree, CarriesAt fs before carrier column tree ->
      CarriesAt fs after carrier column tree) /\
    (before carrier = none -> after carrier = none)

def FramePreserved (fs : List Formula) (offset count : Nat) {total : Nat}
    (before after : BranchState fs.length (fs.length + total)) : Prop :=
  forall carrier, Not (InAllocatedInterval fs offset count carrier) ->
    PayloadPreservedAt fs before after carrier

structure CompilerRunSpec (fs : List Formula) (total offset : Nat)
    (tree : NDTree)
    (before : BranchState fs.length (fs.length + total))
    (rootCarrier : Fin (fs.length + total))
    (rootColumn : Fin fs.length)
    (result : EvalResult fs.length (fs.length + total)) : Prop where
  noError : result.error = false
  root : CarriesAt fs result.state rootCarrier rootColumn tree
  cleared : forall carrier,
    InAllocatedInterval fs offset tree.nodeCount carrier ->
    carrier ≠ rootCarrier ->
    result.state carrier = none
  frame : FramePreserved fs offset tree.nodeCount before result.state

theorem get_formulaColumnOf (fs : List Formula) {formula : Formula}
    (member : formula ∈ fs) :
    fs.get (formulaColumnOf fs member) = formula := by
  exact List.getElem_idxOf (List.idxOf_lt_length_of_mem member)

theorem carriesAt_ageState (fs : List Formula) {K level : Nat}
    {state : BranchState fs.length K} {carrier : Fin K}
    {column : Fin fs.length} {tree : NDTree}
    (carries : CarriesAt fs state carrier column tree) :
    CarriesAt fs (ageState level state) carrier column tree := by
  obtain ⟨oldLevel, atCarrier⟩ := carries
  refine ⟨level, ?_⟩
  simp [ageState, atCarrier, ageToken, mkToken]

theorem none_ageState (fs : List Formula) {K level : Nat}
    {state : BranchState fs.length K} {carrier : Fin K}
    (empty : state carrier = none) :
    ageState level state carrier = none := by
  simp [ageState, empty]

theorem initial_sources_ready (fs : List Formula) (total height : Nat) :
    SourcesReady fs total (initialState fs (fs.length + total) height) := by
  intro column
  refine ⟨height, ?_⟩
  simp [initialState, liftFormulaCarrier, mkToken, column.isLt]

theorem initial_interval_empty (fs : List Formula) (total offset count height : Nat) :
    IntervalEmpty fs offset count
      (initialState fs (fs.length + total) height) := by
  intro carrier inInterval
  rcases inInterval with ⟨lower, upper⟩
  have notSource : Not (carrier.val < fs.length) := by omega
  simp [initialState, notSource]

theorem evalProgramList_append (fs : List Formula) {K : Nat}
    (level : Nat) (state : BranchState fs.length K) (error : Bool)
    (first second : List (BranchInstruction fs.length K)) :
    evalProgramList fs level state error (first ++ second) =
      let middle := evalProgramList fs level state error first
      evalProgramList fs (level - first.length) middle.state middle.error second := by
  induction first generalizing level state error with
  | nil => simp [evalProgramList]
  | cons instruction rest ih =>
      simp only [List.cons_append, evalProgramList]
      rw [ih]
      simp [Nat.sub_sub, Nat.add_comm]

theorem carriesAt_set_ne_after_age (fs : List Formula) {K level : Nat}
    {state : BranchState fs.length K} {carrier target : Fin K}
    {column : Fin fs.length} {tree : NDTree} {value : Option (BranchToken fs.length K)}
    (different : carrier ≠ target)
    (carries : CarriesAt fs state carrier column tree) :
    CarriesAt fs ((ageState level state).set target value)
      carrier column tree := by
  unfold CarriesAt at carries ⊢
  obtain ⟨oldLevel, atCarrier⟩ := carries
  refine ⟨level, ?_⟩
  rw [DLDS.ProofSearch.BranchingGrid.BranchState.set_ne _ _ _ _ different]
  simp [ageState, atCarrier, ageToken, mkToken]

theorem none_set_ne_after_age (fs : List Formula) {K level : Nat}
    {state : BranchState fs.length K} {carrier target : Fin K}
    {value : Option (BranchToken fs.length K)}
    (different : carrier ≠ target) (empty : state carrier = none) :
    (ageState level state).set target value carrier = none := by
  rw [DLDS.ProofSearch.BranchingGrid.BranchState.set_ne _ _ _ _ different]
  exact none_ageState fs empty

theorem carriesAt_set_ne (fs : List Formula) {K : Nat}
    {state : BranchState fs.length K} {carrier target : Fin K}
    {column : Fin fs.length} {tree : NDTree} {value : Option (BranchToken fs.length K)}
    (different : carrier ≠ target)
    (carries : CarriesAt fs state carrier column tree) :
    CarriesAt fs (state.set target value) carrier column tree := by
  unfold CarriesAt at carries ⊢
  obtain ⟨level, atCarrier⟩ := carries
  exact ⟨level, by simpa [DLDS.ProofSearch.BranchingGrid.BranchState.set, different] using atCarrier⟩

theorem none_set_ne (fs : List Formula) {K : Nat}
    {state : BranchState fs.length K} {carrier target : Fin K}
    {value : Option (BranchToken fs.length K)}
    (different : carrier ≠ target) (empty : state carrier = none) :
    state.set target value carrier = none := by
  simpa [DLDS.ProofSearch.BranchingGrid.BranchState.set, different] using empty

theorem executeFork_spec (fs : List Formula) {K level : Nat}
    (state : BranchState fs.length K) (parent child : Fin K)
    (column : Fin fs.length) (tree : NDTree)
    (different : parent ≠ child)
    (parentCarries : CarriesAt fs state parent column tree)
    (childEmpty : state child = none) :
    let result := executeInstruction fs level state (.fork parent child)
    result.error = false /\
      CarriesAt fs result.state child column tree /\
      forall carrier, carrier ≠ child ->
        PayloadPreservedAt fs state result.state carrier := by
  obtain ⟨oldLevel, parentAt⟩ := parentCarries
  have agedParent : ageState level state parent =
      some (mkToken fs parent column level tree) := by
    simp [ageState, parentAt, ageToken, mkToken]
  have agedChild : ageState level state child = none :=
    none_ageState fs childEmpty
  simp only [executeInstruction, agedParent, agedChild]
  simp [different]
  constructor
  · refine ⟨level, ?_⟩
    simp [DLDS.ProofSearch.BranchingGrid.BranchState.set, mkToken]
  · intro carrier notChild
    constructor
    · intro otherColumn otherTree carries
      exact carriesAt_set_ne_after_age fs notChild carries
    · intro empty
      exact none_set_ne_after_age fs notChild empty

theorem executeIntro_spec (fs : List Formula) {K level : Nat}
    (state : BranchState fs.length K) (premise result : Fin K)
    (premiseColumn antecedent conclusion : Fin fs.length)
    (alpha beta : Formula) (tree : NDTree)
    (different : premise ≠ result)
    (premiseCarries : CarriesAt fs state premise premiseColumn tree)
    (resultEmpty : state result = none)
    (premiseFormula : fs.get premiseColumn = beta)
    (antecedentFormula : fs.get antecedent = alpha)
    (conclusionFormula : fs.get conclusion = alpha >> beta) :
    let step := executeInstruction fs level state
      (.intro premise result antecedent conclusion)
    step.error = false /\
      CarriesAt fs step.state result conclusion (.intro alpha beta tree) /\
      step.state premise = none /\
      forall carrier, carrier ≠ premise -> carrier ≠ result ->
        PayloadPreservedAt fs state step.state carrier := by
  obtain ⟨oldLevel, premiseAt⟩ := premiseCarries
  have agedPremise : ageState level state premise =
      some (mkToken fs premise premiseColumn level tree) := by
    simp [ageState, premiseAt, ageToken, mkToken]
  have agedResult : ageState level state result = none :=
    none_ageState fs resultEmpty
  simp only [executeInstruction, agedPremise, mkToken]
  rw [premiseFormula, antecedentFormula, conclusionFormula]
  simp [carrierAvailableAfterOne, different, agedResult]
  constructor
  · refine ⟨level, ?_⟩
    simp [DLDS.ProofSearch.BranchingGrid.BranchState.set, mkToken]
  · intro carrier notPremise notResult
    constructor
    · intro column otherTree carries
      apply carriesAt_set_ne fs notResult
      exact carriesAt_set_ne_after_age fs notPremise carries
    · intro empty
      apply none_set_ne fs notResult
      exact none_set_ne_after_age fs notPremise empty

theorem executeElim_spec (fs : List Formula) {K level : Nat}
    (state : BranchState fs.length K) (major minor result : Fin K)
    (majorColumn minorColumn conclusion : Fin fs.length)
    (alpha beta : Formula) (majorTree minorTree : NDTree)
    (majorMinor : major ≠ minor) (majorResult : major ≠ result)
    (minorResult : minor ≠ result)
    (majorCarries : CarriesAt fs state major majorColumn majorTree)
    (minorCarries : CarriesAt fs state minor minorColumn minorTree)
    (resultEmpty : state result = none)
    (majorFormula : fs.get majorColumn = alpha >> beta)
    (minorFormula : fs.get minorColumn = alpha)
    (conclusionFormula : fs.get conclusion = beta) :
    let step := executeInstruction fs level state
      (.elim major minor result conclusion)
    step.error = false /\
      CarriesAt fs step.state result conclusion
        (.elim alpha beta majorTree minorTree) /\
      step.state major = none /\
      step.state minor = none /\
      forall carrier, carrier ≠ major -> carrier ≠ minor ->
        carrier ≠ result -> PayloadPreservedAt fs state step.state carrier := by
  obtain ⟨majorLevel, majorAt⟩ := majorCarries
  obtain ⟨minorLevel, minorAt⟩ := minorCarries
  have agedMajor : ageState level state major =
      some (mkToken fs major majorColumn level majorTree) := by
    simp [ageState, majorAt, ageToken, mkToken]
  have agedMinor : ageState level state minor =
      some (mkToken fs minor minorColumn level minorTree) := by
    simp [ageState, minorAt, ageToken, mkToken]
  have agedResult : ageState level state result = none :=
    none_ageState fs resultEmpty
  simp only [executeInstruction, agedMajor, agedMinor, mkToken]
  rw [majorFormula, minorFormula, conclusionFormula]
  simp [carrierAvailableAfterTwo, majorMinor, majorResult, minorResult,
    agedResult]
  constructor
  · refine ⟨level, ?_⟩
    simp [DLDS.ProofSearch.BranchingGrid.BranchState.set, mkToken]
  · intro carrier notMajor notMinor notResult
    constructor
    · intro column otherTree carries
      apply carriesAt_set_ne fs notResult
      apply carriesAt_set_ne fs notMinor
      exact carriesAt_set_ne_after_age fs notMajor carries
    · intro empty
      apply none_set_ne fs notResult
      apply none_set_ne fs notMinor
      exact none_set_ne_after_age fs notMajor empty

theorem compileCertified_hyp_correct (fs : List Formula) (total offset level : Nat)
    (formula : Formula) (covered : FormulaCovered fs (.hyp formula))
    (bound : offset + (ProofSearch.Grid.NDTree.hyp formula).nodeCount ≤ total)
    (state : BranchState fs.length (fs.length + total))
    (sources : SourcesReady fs total state)
    (empty : IntervalEmpty fs offset (ProofSearch.Grid.NDTree.hyp formula).nodeCount state) :
    let root := allocatedRoot fs total offset (ProofSearch.Grid.NDTree.hyp formula).nodeCount
      (ProofSearch.Grid.NDTree.hyp formula).nodeCount_pos bound
    let column := formulaColumnOf fs covered.conclusion
    let result := evalProgramList fs level state false
      (compileCertifiedFrom fs total offset (.hyp formula) covered bound)
    CompilerRunSpec fs total offset (.hyp formula) state root column result := by
  let column := formulaColumnOf fs covered.hyp
  let source := liftFormulaCarrier fs total column
  let root := allocatedRoot fs total offset 1 (by omega) (by
    simpa [ProofSearch.Grid.NDTree.nodeCount] using bound)
  have sourceCarries : CarriesAt fs state source column (.hyp formula) := by
    have ready := sources column
    have formulaAt : fs.get column = formula := by
      simpa [column] using get_formulaColumnOf fs covered.hyp
    rw [formulaAt] at ready
    exact ready
  have rootIn : InAllocatedInterval fs offset 1 root := by
    constructor <;> simp [root, allocatedRoot]
  have rootEmpty : state root = none := by
    apply empty root
    simpa [ProofSearch.Grid.NDTree.nodeCount] using rootIn
  have different : source ≠ root := by
    intro equal
    have values := Fin.val_eq_of_eq equal
    simp [source, liftFormulaCarrier, column, formulaColumnOf,
      root, allocatedRoot] at values
    have idxBound := List.idxOf_lt_length_of_mem covered.hyp
    omega
  let step := executeInstruction fs (level - 1) state (.fork source root)
  let final : EvalResult fs.length (fs.length + total) :=
    { state := step.state, error := step.error }
  have stepSpec := executeFork_spec fs state source root column (.hyp formula)
    different sourceCarries rootEmpty (level := level - 1)
  have resultEq :
      evalProgramList fs level state false
        (compileCertifiedFrom fs total offset (.hyp formula) covered bound) =
      final := by
    simp [compileCertifiedFrom, evalProgramList, final, step, source, root, column]
  rw [resultEq]
  refine {
    noError := stepSpec.1
    root := ?_
    cleared := ?_
    frame := ?_ }
  · simpa [root, column, ProofSearch.Grid.NDTree.nodeCount] using
      stepSpec.2.1
  · intro carrier inInterval notRoot
    rcases inInterval with ⟨lower, upper⟩
    have sameValue : carrier.val = root.val := by
      simp [root, allocatedRoot,
        ProofSearch.Grid.NDTree.nodeCount] at lower upper ⊢
      omega
    exact False.elim (notRoot (Fin.ext sameValue))
  · intro carrier outside
    apply stepSpec.2.2 carrier
    intro equal
    apply outside
    subst carrier
    simpa [ProofSearch.Grid.NDTree.nodeCount] using rootIn

theorem compileCertified_intro_of_premise_correct
    (fs : List Formula) (total offset level : Nat)
    (alpha beta : Formula) (premise : NDTree)
    (covered : FormulaCovered fs (.intro alpha beta premise))
    (bound : offset + (ProofSearch.Grid.NDTree.intro alpha beta premise).nodeCount ≤ total)
    (wellFormed : (ProofSearch.Grid.NDTree.intro alpha beta premise).wellFormedB = true)
    (state : BranchState fs.length (fs.length + total))
    (empty : IntervalEmpty fs offset
      (ProofSearch.Grid.NDTree.intro alpha beta premise).nodeCount state)
    (premiseSpec :
      let premiseCovered := covered.intro_premise
      let premiseBound : offset + premise.nodeCount ≤ total := by
        have full : offset + (premise.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount] using bound
        omega
      let premiseRoot := allocatedRoot fs total offset premise.nodeCount
        premise.nodeCount_pos premiseBound
      let premiseColumn := formulaColumnOf fs premiseCovered.conclusion
      let premiseResult := evalProgramList fs level state false
        (compileCertifiedFrom fs total offset premise premiseCovered premiseBound)
      CompilerRunSpec fs total offset premise state premiseRoot premiseColumn
        premiseResult) :
    let root := allocatedRoot fs total offset
      (ProofSearch.Grid.NDTree.intro alpha beta premise).nodeCount
      (ProofSearch.Grid.NDTree.intro alpha beta premise).nodeCount_pos bound
    let column := formulaColumnOf fs covered.conclusion
    let result := evalProgramList fs level state false
      (compileCertifiedFrom fs total offset (.intro alpha beta premise)
        covered bound)
    CompilerRunSpec fs total offset (.intro alpha beta premise)
      state root column result := by
  simp only [ProofSearch.Grid.NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed
  obtain ⟨premiseWellFormed, premiseConclusion⟩ := wellFormed
  let premiseCovered := covered.intro_premise
  have premiseBound : offset + premise.nodeCount ≤ total := by
    have full : offset + (premise.nodeCount + 1) ≤ total := by
      simpa [ProofSearch.Grid.NDTree.nodeCount] using bound
    omega
  let premiseRoot := allocatedRoot fs total offset premise.nodeCount
    premise.nodeCount_pos premiseBound
  let root := allocatedRoot fs total offset (premise.nodeCount + 1)
    (by omega) (by
      simpa [ProofSearch.Grid.NDTree.nodeCount] using bound)
  let premiseColumn := formulaColumnOf fs premiseCovered.conclusion
  let antecedentColumn := formulaColumnOf fs covered.intro_antecedent
  let conclusionColumn := formulaColumnOf fs covered.intro_conclusion
  let premiseProgram :=
    compileCertifiedFrom fs total offset premise premiseCovered premiseBound
  let premiseResult := evalProgramList fs level state false premiseProgram
  have pSpec : CompilerRunSpec fs total offset premise state premiseRoot
      premiseColumn premiseResult := by
    simpa [premiseCovered, premiseBound, premiseRoot, premiseColumn,
      premiseProgram, premiseResult] using premiseSpec
  have rootInFull : InAllocatedInterval fs offset (premise.nodeCount + 1) root := by
    constructor <;> simp [root, allocatedRoot]
  have premiseRootInPremise :
      InAllocatedInterval fs offset premise.nodeCount premiseRoot := by
    constructor
    · change fs.length + offset ≤
        fs.length + offset + premise.nodeCount - 1
      have premisePositive := premise.nodeCount_pos
      omega
    · change fs.length + offset + premise.nodeCount - 1 <
        fs.length + offset + premise.nodeCount
      have premisePositive := premise.nodeCount_pos
      omega
  have premiseRootInFull :
      InAllocatedInterval fs offset (premise.nodeCount + 1) premiseRoot := by
    rcases premiseRootInPremise with ⟨lower, upper⟩
    exact ⟨lower, by omega⟩
  have rootOutsidePremise :
      Not (InAllocatedInterval fs offset premise.nodeCount root) := by
    intro inside
    rcases inside with ⟨lower, upper⟩
    simp [root, allocatedRoot] at upper
  have rootEmptyBefore : state root = none := by
    apply empty root
    simpa [ProofSearch.Grid.NDTree.nodeCount] using rootInFull
  have rootEmpty : premiseResult.state root = none :=
    (pSpec.frame root rootOutsidePremise).2 rootEmptyBefore
  have premiseRootNe : premiseRoot ≠ root := by
    intro equal
    have values := Fin.val_eq_of_eq equal
    simp [premiseRoot, root, allocatedRoot] at values
    have premisePositive := premise.nodeCount_pos
    omega
  have premiseFormula : fs.get premiseColumn = beta := by
    rw [get_formulaColumnOf fs premiseCovered.conclusion]
    exact premiseConclusion
  have antecedentFormula : fs.get antecedentColumn = alpha :=
    get_formulaColumnOf fs covered.intro_antecedent
  have conclusionFormula : fs.get conclusionColumn = alpha >> beta :=
    get_formulaColumnOf fs covered.intro_conclusion
  let step := executeInstruction fs
    (level - premiseProgram.length - 1) premiseResult.state
    (.intro premiseRoot root antecedentColumn conclusionColumn)
  let final : EvalResult fs.length (fs.length + total) :=
    { state := step.state, error := step.error }
  have stepSpec := executeIntro_spec fs premiseResult.state premiseRoot root
    premiseColumn antecedentColumn conclusionColumn alpha beta premise
    premiseRootNe pSpec.root rootEmpty premiseFormula antecedentFormula
    conclusionFormula (level := level - premiseProgram.length - 1)
  have resultEq :
      evalProgramList fs level state false
        (compileCertifiedFrom fs total offset (.intro alpha beta premise)
          covered bound) = final := by
    rw [show compileCertifiedFrom fs total offset (.intro alpha beta premise)
        covered bound = premiseProgram ++
          [.intro premiseRoot root antecedentColumn conclusionColumn] by
      rfl]
    rw [evalProgramList_append]
    simp [premiseResult, pSpec.noError, evalProgramList, step, final]
  rw [resultEq]
  refine {
    noError := stepSpec.1
    root := ?_
    cleared := ?_
    frame := ?_ }
  · simpa [root, conclusionColumn,
      ProofSearch.Grid.NDTree.nodeCount] using stepSpec.2.1
  · intro carrier inFull notRoot
    by_cases isPremiseRoot : carrier = premiseRoot
    · subst carrier
      exact stepSpec.2.2.1
    · have inPremise : InAllocatedInterval fs offset premise.nodeCount carrier := by
        rcases inFull with ⟨lower, upper⟩
        constructor
        · exact lower
        · have notLast : carrier.val ≠ root.val := by
            intro values
            exact notRoot (Fin.ext values)
          simp [root, allocatedRoot,
            ProofSearch.Grid.NDTree.nodeCount] at upper notLast ⊢
          have premisePositive := premise.nodeCount_pos
          omega
      have emptyAfterPremise := pSpec.cleared carrier inPremise isPremiseRoot
      exact (stepSpec.2.2.2 carrier isPremiseRoot notRoot).2
        emptyAfterPremise
  · intro carrier outsideFull
    have outsidePremise :
        Not (InAllocatedInterval fs offset premise.nodeCount carrier) := by
      intro inside
      apply outsideFull
      rcases inside with ⟨lower, upper⟩
      constructor <;> simp_all [ProofSearch.Grid.NDTree.nodeCount]
      omega
    have notPremiseRoot : carrier ≠ premiseRoot := by
      intro equal
      apply outsideFull
      subst carrier
      exact premiseRootInFull
    have notRoot : carrier ≠ root := by
      intro equal
      apply outsideFull
      subst carrier
      simpa [ProofSearch.Grid.NDTree.nodeCount] using rootInFull
    have firstPreserved := pSpec.frame carrier outsidePremise
    have secondPreserved := stepSpec.2.2.2 carrier notPremiseRoot notRoot
    constructor
    · intro column tree carries
      exact secondPreserved.1 column tree
        (firstPreserved.1 column tree carries)
    · intro wasEmpty
      exact secondPreserved.2 (firstPreserved.2 wasEmpty)

theorem compileCertified_elim_of_children_correct
    (fs : List Formula) (total offset level : Nat)
    (alpha beta : Formula) (major minor : NDTree)
    (covered : FormulaCovered fs (.elim alpha beta major minor))
    (bound : offset + (ProofSearch.Grid.NDTree.elim alpha beta major minor).nodeCount ≤ total)
    (wellFormed : (ProofSearch.Grid.NDTree.elim alpha beta major minor).wellFormedB = true)
    (state : BranchState fs.length (fs.length + total))
    (empty : IntervalEmpty fs offset
      (ProofSearch.Grid.NDTree.elim alpha beta major minor).nodeCount state)
    (majorSpec :
      let majorCovered := covered.elim_major
      let majorBound : offset + major.nodeCount ≤ total := by
        have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
        have minorPositive := minor.nodeCount_pos
        omega
      let majorRoot := allocatedRoot fs total offset major.nodeCount
        major.nodeCount_pos majorBound
      let majorColumn := formulaColumnOf fs majorCovered.conclusion
      let majorResult := evalProgramList fs level state false
        (compileCertifiedFrom fs total offset major majorCovered majorBound)
      CompilerRunSpec fs total offset major state majorRoot majorColumn majorResult)
    (minorSpec :
      let majorCovered := covered.elim_major
      let minorCovered := covered.elim_minor
      let majorBound : offset + major.nodeCount ≤ total := by
        have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
        have minorPositive := minor.nodeCount_pos
        omega
      let minorBound : offset + major.nodeCount + minor.nodeCount ≤ total := by
        have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
        omega
      let majorProgram :=
        compileCertifiedFrom fs total offset major majorCovered majorBound
      let majorResult := evalProgramList fs level state false majorProgram
      let minorRoot := allocatedRoot fs total (offset + major.nodeCount)
        minor.nodeCount minor.nodeCount_pos minorBound
      let minorColumn := formulaColumnOf fs minorCovered.conclusion
      let minorResult := evalProgramList fs (level - majorProgram.length)
        majorResult.state false
        (compileCertifiedFrom fs total (offset + major.nodeCount) minor
          minorCovered minorBound)
      CompilerRunSpec fs total (offset + major.nodeCount) minor
        majorResult.state minorRoot minorColumn minorResult) :
    let root := allocatedRoot fs total offset
      (ProofSearch.Grid.NDTree.elim alpha beta major minor).nodeCount
      (ProofSearch.Grid.NDTree.elim alpha beta major minor).nodeCount_pos bound
    let column := formulaColumnOf fs covered.conclusion
    let result := evalProgramList fs level state false
      (compileCertifiedFrom fs total offset (.elim alpha beta major minor)
        covered bound)
    CompilerRunSpec fs total offset (.elim alpha beta major minor)
      state root column result := by
  simp only [ProofSearch.Grid.NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed
  obtain ⟨⟨⟨majorWellFormed, minorWellFormed⟩, majorConclusion⟩,
    minorConclusion⟩ := wellFormed
  let majorCovered := covered.elim_major
  let minorCovered := covered.elim_minor
  have majorBound : offset + major.nodeCount ≤ total := by
    have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
      simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
    have minorPositive := minor.nodeCount_pos
    omega
  have minorBound : offset + major.nodeCount + minor.nodeCount ≤ total := by
    have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
      simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
    omega
  let majorRoot := allocatedRoot fs total offset major.nodeCount
    major.nodeCount_pos majorBound
  let minorRoot := allocatedRoot fs total (offset + major.nodeCount)
    minor.nodeCount minor.nodeCount_pos minorBound
  let root := allocatedRoot fs total offset
    (major.nodeCount + minor.nodeCount + 1) (by omega) (by
      simpa [ProofSearch.Grid.NDTree.nodeCount] using bound)
  let majorColumn := formulaColumnOf fs majorCovered.conclusion
  let minorColumn := formulaColumnOf fs minorCovered.conclusion
  let conclusionColumn := formulaColumnOf fs covered.elim_conclusion
  let majorProgram :=
    compileCertifiedFrom fs total offset major majorCovered majorBound
  let majorResult := evalProgramList fs level state false majorProgram
  let minorProgram :=
    compileCertifiedFrom fs total (offset + major.nodeCount) minor
      minorCovered minorBound
  let minorResult := evalProgramList fs (level - majorProgram.length)
    majorResult.state false minorProgram
  have mSpec : CompilerRunSpec fs total offset major state majorRoot
      majorColumn majorResult := by
    simpa [majorCovered, majorBound, majorRoot, majorColumn,
      majorProgram, majorResult] using majorSpec
  have nSpec : CompilerRunSpec fs total (offset + major.nodeCount) minor
      majorResult.state minorRoot minorColumn minorResult := by
    simpa [majorCovered, minorCovered, majorBound, minorBound, majorProgram,
      majorResult, minorRoot, minorColumn, minorProgram, minorResult] using minorSpec
  have majorRootInMajor :
      InAllocatedInterval fs offset major.nodeCount majorRoot := by
    constructor
    · change fs.length + offset ≤ fs.length + offset + major.nodeCount - 1
      have positive := major.nodeCount_pos
      omega
    · change fs.length + offset + major.nodeCount - 1 <
        fs.length + offset + major.nodeCount
      have positive := major.nodeCount_pos
      omega
  have minorRootInMinor : InAllocatedInterval fs (offset + major.nodeCount)
      minor.nodeCount minorRoot := by
    constructor
    · change fs.length + (offset + major.nodeCount) ≤
        fs.length + (offset + major.nodeCount) + minor.nodeCount - 1
      have positive := minor.nodeCount_pos
      omega
    · change fs.length + (offset + major.nodeCount) + minor.nodeCount - 1 <
        fs.length + (offset + major.nodeCount) + minor.nodeCount
      have positive := minor.nodeCount_pos
      omega
  have majorRootOutsideMinor :
      Not (InAllocatedInterval fs (offset + major.nodeCount)
        minor.nodeCount majorRoot) := by
    intro inside
    rcases inside with ⟨lower, upper⟩
    rcases majorRootInMajor with ⟨majorLower, majorUpper⟩
    omega
  have rootInFull : InAllocatedInterval fs offset
      (major.nodeCount + minor.nodeCount + 1) root := by
    constructor <;> simp [root, allocatedRoot]
  have rootOutsideMajor :
      Not (InAllocatedInterval fs offset major.nodeCount root) := by
    intro inside
    rcases inside with ⟨lower, upper⟩
    simp [root, allocatedRoot] at upper
  have rootOutsideMinor :
      Not (InAllocatedInterval fs (offset + major.nodeCount)
        minor.nodeCount root) := by
    intro inside
    rcases inside with ⟨lower, upper⟩
    simp [root, allocatedRoot] at upper
    omega
  have rootEmptyBefore : state root = none := by
    apply empty root
    simpa [ProofSearch.Grid.NDTree.nodeCount] using rootInFull
  have rootEmptyAfterMajor : majorResult.state root = none :=
    (mSpec.frame root rootOutsideMajor).2 rootEmptyBefore
  have rootEmpty : minorResult.state root = none :=
    (nSpec.frame root rootOutsideMinor).2 rootEmptyAfterMajor
  have majorRootCarries : CarriesAt fs minorResult.state majorRoot majorColumn major :=
    (nSpec.frame majorRoot majorRootOutsideMinor).1 majorColumn major mSpec.root
  have majorMinorNe : majorRoot ≠ minorRoot := by
    intro equal
    have values := Fin.val_eq_of_eq equal
    rcases majorRootInMajor with ⟨majorLower, majorUpper⟩
    rcases minorRootInMinor with ⟨minorLower, minorUpper⟩
    omega
  have majorRootNe : majorRoot ≠ root := by
    intro equal
    apply rootOutsideMajor
    rw [← equal]
    exact majorRootInMajor
  have minorRootNe : minorRoot ≠ root := by
    intro equal
    apply rootOutsideMinor
    rw [← equal]
    exact minorRootInMinor
  have majorFormula : fs.get majorColumn = alpha >> beta := by
    rw [get_formulaColumnOf fs majorCovered.conclusion]
    exact majorConclusion
  have minorFormula : fs.get minorColumn = alpha := by
    rw [get_formulaColumnOf fs minorCovered.conclusion]
    exact minorConclusion
  have conclusionFormula : fs.get conclusionColumn = beta :=
    get_formulaColumnOf fs covered.elim_conclusion
  let step := executeInstruction fs
    (level - (majorProgram ++ minorProgram).length - 1) minorResult.state
    (.elim majorRoot minorRoot root conclusionColumn)
  let final : EvalResult fs.length (fs.length + total) :=
    { state := step.state, error := step.error }
  have stepSpec := executeElim_spec fs minorResult.state majorRoot minorRoot root
    majorColumn minorColumn conclusionColumn alpha beta major minor majorMinorNe
    majorRootNe minorRootNe majorRootCarries nSpec.root rootEmpty majorFormula
    minorFormula conclusionFormula
    (level := level - (majorProgram ++ minorProgram).length - 1)
  have prefixEq : evalProgramList fs level state false
      (majorProgram ++ minorProgram) = minorResult := by
    rw [evalProgramList_append]
    simp [majorResult, mSpec.noError, minorResult]
  have resultEq :
      evalProgramList fs level state false
        (compileCertifiedFrom fs total offset (.elim alpha beta major minor)
          covered bound) = final := by
    rw [show compileCertifiedFrom fs total offset (.elim alpha beta major minor)
        covered bound = (majorProgram ++ minorProgram) ++
          [.elim majorRoot minorRoot root conclusionColumn] by
      simp only [compileCertifiedFrom]
      rfl]
    rw [evalProgramList_append, prefixEq]
    simp [nSpec.noError, evalProgramList, step, final]
  rw [resultEq]
  refine {
    noError := stepSpec.1
    root := ?_
    cleared := ?_
    frame := ?_ }
  · change CarriesAt fs step.state root conclusionColumn
      (.elim alpha beta major minor)
    exact stepSpec.2.1
  · intro carrier inFull notRoot
    by_cases isMajorRoot : carrier = majorRoot
    · subst carrier
      exact stepSpec.2.2.1
    · by_cases isMinorRoot : carrier = minorRoot
      · subst carrier
        exact stepSpec.2.2.2.1
      · have beforeStepEmpty : minorResult.state carrier = none := by
          by_cases inMajor : InAllocatedInterval fs offset major.nodeCount carrier
          · have afterMajorEmpty := mSpec.cleared carrier inMajor isMajorRoot
            have outsideMinor : Not (InAllocatedInterval fs
                (offset + major.nodeCount) minor.nodeCount carrier) := by
              intro inside
              rcases inMajor with ⟨majorLower, majorUpper⟩
              rcases inside with ⟨minorLower, minorUpper⟩
              omega
            exact (nSpec.frame carrier outsideMinor).2 afterMajorEmpty
          · have inMinor : InAllocatedInterval fs (offset + major.nodeCount)
                minor.nodeCount carrier := by
              rcases inFull with ⟨lower, upper⟩
              constructor
              · by_contra below
                push_neg at below
                apply inMajor
                exact ⟨lower, by omega⟩
              · have notLast : carrier.val ≠ root.val := by
                  intro values
                  exact notRoot (Fin.ext values)
                simp [root, allocatedRoot,
                  ProofSearch.Grid.NDTree.nodeCount] at upper notLast ⊢
                have majorPositive := major.nodeCount_pos
                have minorPositive := minor.nodeCount_pos
                omega
            exact nSpec.cleared carrier inMinor isMinorRoot
        exact (stepSpec.2.2.2.2 carrier isMajorRoot isMinorRoot notRoot).2
          beforeStepEmpty
  · intro carrier outsideFull
    have outsideMajor : Not (InAllocatedInterval fs offset major.nodeCount carrier) := by
      intro inside
      apply outsideFull
      rcases inside with ⟨lower, upper⟩
      change InAllocatedInterval fs offset
        (major.nodeCount + minor.nodeCount + 1) carrier
      exact ⟨lower, by omega⟩
    have outsideMinor : Not (InAllocatedInterval fs (offset + major.nodeCount)
        minor.nodeCount carrier) := by
      intro inside
      apply outsideFull
      rcases inside with ⟨lower, upper⟩
      change InAllocatedInterval fs offset
        (major.nodeCount + minor.nodeCount + 1) carrier
      exact ⟨by omega, by omega⟩
    have notMajorRoot : carrier ≠ majorRoot := by
      intro equal
      apply outsideFull
      subst carrier
      rcases majorRootInMajor with ⟨lower, upper⟩
      change InAllocatedInterval fs offset
        (major.nodeCount + minor.nodeCount + 1) majorRoot
      exact ⟨lower, by omega⟩
    have notMinorRoot : carrier ≠ minorRoot := by
      intro equal
      apply outsideFull
      subst carrier
      rcases minorRootInMinor with ⟨lower, upper⟩
      change InAllocatedInterval fs offset
        (major.nodeCount + minor.nodeCount + 1) minorRoot
      exact ⟨by omega, by omega⟩
    have notRoot : carrier ≠ root := by
      intro equal
      apply outsideFull
      subst carrier
      simpa [ProofSearch.Grid.NDTree.nodeCount] using rootInFull
    have firstPreserved := mSpec.frame carrier outsideMajor
    have secondPreserved := nSpec.frame carrier outsideMinor
    have thirdPreserved :=
      stepSpec.2.2.2.2 carrier notMajorRoot notMinorRoot notRoot
    constructor
    · intro column tree carries
      exact thirdPreserved.1 column tree
        (secondPreserved.1 column tree
          (firstPreserved.1 column tree carries))
    · intro wasEmpty
      exact thirdPreserved.2 (secondPreserved.2 (firstPreserved.2 wasEmpty))

theorem sourcesReady_of_frame (fs : List Formula) (total offset count : Nat)
    {before after : BranchState fs.length (fs.length + total)}
    (sources : SourcesReady fs total before)
    (frame : FramePreserved fs offset count before after) :
    SourcesReady fs total after := by
  intro column
  have outside : Not (InAllocatedInterval fs offset count
      (liftFormulaCarrier fs total column)) := by
    intro inside
    rcases inside with ⟨lower, upper⟩
    simp [liftFormulaCarrier] at lower
    omega
  exact (frame _ outside).1 column (.hyp (fs.get column)) (sources column)

/-- General operational correctness of the proof-carrying postorder compiler.
The statement is deliberately framed so recursive compilations may start in an
arbitrary state: permanent formula sources must be ready, the private carrier
interval must be empty, and everything outside that interval is preserved. -/
theorem compileCertifiedFrom_correct
    (fs : List Formula) (total offset level : Nat)
    (tree : NDTree) (covered : FormulaCovered fs tree)
    (bound : offset + tree.nodeCount ≤ total)
    (wellFormed : tree.wellFormedB = true)
    (state : BranchState fs.length (fs.length + total))
    (sources : SourcesReady fs total state)
    (empty : IntervalEmpty fs offset tree.nodeCount state) :
    let root := allocatedRoot fs total offset tree.nodeCount
      tree.nodeCount_pos bound
    let column := formulaColumnOf fs covered.conclusion
    let result := evalProgramList fs level state false
      (compileCertifiedFrom fs total offset tree covered bound)
    CompilerRunSpec fs total offset tree state root column result := by
  induction tree generalizing total offset level state with
  | hyp formula =>
      exact compileCertified_hyp_correct fs total offset level formula covered
        bound state sources empty
  | intro alpha beta premise ih =>
      have wellFormed' := wellFormed
      simp only [ProofSearch.Grid.NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed'
      obtain ⟨premiseWellFormed, premiseConclusion⟩ := wellFormed'
      let premiseCovered := covered.intro_premise
      have premiseBound : offset + premise.nodeCount ≤ total := by
        have full : offset + (premise.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount] using bound
        omega
      have premiseEmpty : IntervalEmpty fs offset premise.nodeCount state := by
        intro carrier inside
        apply empty carrier
        rcases inside with ⟨lower, upper⟩
        constructor
        · exact lower
        · simpa [ProofSearch.Grid.NDTree.nodeCount] using
            (show carrier.val < fs.length + offset + (premise.nodeCount + 1) by
              omega)
      have premiseSpec := ih total offset level premiseCovered premiseBound
        premiseWellFormed state sources premiseEmpty
      exact compileCertified_intro_of_premise_correct fs total offset level
        alpha beta premise covered bound wellFormed state empty premiseSpec
  | elim alpha beta major minor ihMajor ihMinor =>
      have wellFormed' := wellFormed
      simp only [ProofSearch.Grid.NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed'
      obtain ⟨⟨⟨majorWellFormed, minorWellFormed⟩, majorConclusion⟩,
        minorConclusion⟩ := wellFormed'
      let majorCovered := covered.elim_major
      let minorCovered := covered.elim_minor
      have majorBound : offset + major.nodeCount ≤ total := by
        have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
        have minorPositive := minor.nodeCount_pos
        omega
      have minorBound : offset + major.nodeCount + minor.nodeCount ≤ total := by
        have full : offset + (major.nodeCount + minor.nodeCount + 1) ≤ total := by
          simpa [ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc] using bound
        omega
      have majorEmpty : IntervalEmpty fs offset major.nodeCount state := by
        intro carrier inside
        apply empty carrier
        rcases inside with ⟨lower, upper⟩
        change InAllocatedInterval fs offset
          (major.nodeCount + minor.nodeCount + 1) carrier
        exact ⟨lower, by omega⟩
      have majorSpec := ihMajor total offset level majorCovered majorBound
        majorWellFormed state sources majorEmpty
      let majorProgram :=
        compileCertifiedFrom fs total offset major majorCovered majorBound
      let majorResult := evalProgramList fs level state false majorProgram
      have mSpec : CompilerRunSpec fs total offset major state
          (allocatedRoot fs total offset major.nodeCount
            major.nodeCount_pos majorBound)
          (formulaColumnOf fs majorCovered.conclusion) majorResult := by
        simpa [majorCovered, majorBound, majorProgram, majorResult] using majorSpec
      have sourcesAfterMajor : SourcesReady fs total majorResult.state :=
        sourcesReady_of_frame fs total offset major.nodeCount sources mSpec.frame
      have minorEmpty : IntervalEmpty fs (offset + major.nodeCount)
          minor.nodeCount majorResult.state := by
        intro carrier insideMinor
        have outsideMajor : Not (InAllocatedInterval fs offset
            major.nodeCount carrier) := by
          intro insideMajor
          rcases insideMajor with ⟨majorLower, majorUpper⟩
          rcases insideMinor with ⟨minorLower, minorUpper⟩
          omega
        apply (mSpec.frame carrier outsideMajor).2
        apply empty carrier
        rcases insideMinor with ⟨lower, upper⟩
        change InAllocatedInterval fs offset
          (major.nodeCount + minor.nodeCount + 1) carrier
        exact ⟨by omega, by omega⟩
      have minorSpec := ihMinor total (offset + major.nodeCount)
        (level - majorProgram.length) minorCovered minorBound minorWellFormed
        majorResult.state sourcesAfterMajor minorEmpty
      exact compileCertified_elim_of_children_correct fs total offset level
        alpha beta major minor covered bound wellFormed state empty majorSpec
        (by simpa [majorCovered, minorCovered, majorBound, minorBound,
          majorProgram, majorResult] using minorSpec)

theorem NDTree.height_le_nodeCount (tree : NDTree) :
    tree.height ≤ tree.nodeCount := by
  induction tree with
  | hyp formula => simp [ProofSearch.Grid.NDTree.height, ProofSearch.Grid.NDTree.nodeCount]
  | intro alpha beta premise ih =>
      simp [ProofSearch.Grid.NDTree.height, ProofSearch.Grid.NDTree.nodeCount, ih]
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [ProofSearch.Grid.NDTree.height, ProofSearch.Grid.NDTree.nodeCount]
      have hmax : max major.height minor.height ≤
          major.nodeCount + minor.nodeCount :=
        Nat.max_le.mpr ⟨by omega, by omega⟩
      omega

theorem FormulaCovered.dependencies {fs : List Formula} {tree : NDTree}
    (covered : FormulaCovered fs tree) :
    forall formula, formula ∈ tree.dependencies -> formula ∈ fs := by
  induction tree with
  | hyp hypothesis =>
      intro formula member
      simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_singleton] at member
      subst formula
      exact covered.hyp
  | intro alpha beta premise ih =>
      intro formula member
      simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups, List.mem_filter] at member
      exact ih covered.intro_premise formula member.1
  | elim alpha beta major minor ihMajor ihMinor =>
      intro formula member
      simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups, List.mem_append] at member
      rcases member with inMajor | inMinor
      · exact ihMajor covered.elim_major formula inMajor
      · exact ihMinor covered.elim_minor formula inMinor

theorem dependencySupportedB_of_covered (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree) :
    dependencySupportedB fs tree = true := by
  rw [dependencySupportedB, List.all_eq_true]
  intro formula member
  simpa using covered.dependencies formula member

theorem tokenAcceptsB_mkToken (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = [])
    {K : Nat} (carrier : Fin K) (column : Fin fs.length)
    (columnFormula : fs.get column = tree.conclusion)
    (level height : Nat) (heightBound : tree.height ≤ height) :
    tokenAcceptsB fs tree.conclusion height
      (mkToken fs carrier column level tree) = true := by
  have supported := dependencySupportedB_of_covered fs tree covered
  simp only [tokenAcceptsB, tokenValidB, mkToken]
  rw [wellFormed, columnFormula, supported, closed]
  simp [depVector_nil_eq_replicate, List.Vector.replicate, heightBound]

theorem mem_liveTokens_of_state_eq {n K : Nat}
    (state : BranchState n K) (carrier : Fin K) (token : BranchToken n K)
    (atCarrier : state carrier = some token) : token ∈ liveTokens state := by
  simp only [liveTokens, List.mem_filterMap]
  exact ⟨carrier, by simp, atCarrier⟩

/-- Top-level compiler correctness for the exact finite resource allocation:
`tree.nodeCount` rows and `fs.length + tree.nodeCount` carriers. -/
theorem encodeCertifiedNDTree_correct
    (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree)
    (wellFormed : tree.wellFormedB = true) :
    let root := allocatedRoot fs tree.nodeCount 0 tree.nodeCount
      tree.nodeCount_pos (by simp)
    let column := formulaColumnOf fs covered.conclusion
    let result := evalOccurrence fs tree.nodeCount
      (fs.length + tree.nodeCount) (encodeCertifiedNDTree fs tree covered)
    CompilerRunSpec fs tree.nodeCount 0 tree
      (initialState fs (fs.length + tree.nodeCount) tree.nodeCount)
      root column result := by
  have spec := compileCertifiedFrom_correct fs tree.nodeCount 0 tree.nodeCount
    tree covered (by simp) wellFormed
    (initialState fs (fs.length + tree.nodeCount) tree.nodeCount)
    (initial_sources_ready fs tree.nodeCount tree.nodeCount)
    (initial_interval_empty fs tree.nodeCount 0 tree.nodeCount tree.nodeCount)
  simpa [compileCertified, evalOccurrence] using spec

/-- Constructive completeness of occurrence mode.  No normality, greediness,
or collision-free scheduling hypothesis is needed: the ND tree is serialized
in postorder and every constructor receives its own carrier. -/
theorem occurrenceMarked_complete
    (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    occurrenceMarkedB fs tree.conclusion tree.nodeCount
      (fs.length + tree.nodeCount) (encodeCertifiedNDTree fs tree covered) = true := by
  let root := allocatedRoot fs tree.nodeCount 0 tree.nodeCount
    tree.nodeCount_pos (by simp)
  let column := formulaColumnOf fs covered.conclusion
  let result := evalOccurrence fs tree.nodeCount
    (fs.length + tree.nodeCount) (encodeCertifiedNDTree fs tree covered)
  have spec : CompilerRunSpec fs tree.nodeCount 0 tree
      (initialState fs (fs.length + tree.nodeCount) tree.nodeCount)
      root column result := by
    simpa [root, column, result] using
      encodeCertifiedNDTree_correct fs tree covered wellFormed
  obtain ⟨tokenLevel, rootAt⟩ := spec.root
  let token := mkToken fs root column tokenLevel tree
  have tokenMem : token ∈ liveTokens result.state :=
    mem_liveTokens_of_state_eq result.state root token rootAt
  have columnFormula : fs.get column = tree.conclusion := by
    exact get_formulaColumnOf fs covered.conclusion
  have accepts : tokenAcceptsB fs tree.conclusion tree.nodeCount token = true :=
    tokenAcceptsB_mkToken fs tree covered wellFormed closed root column
      columnFormula tokenLevel tree.nodeCount
        (DLDS.ProofSearch.BranchingGrid.NDTree.height_le_nodeCount tree)
  have found : (acceptedToken? fs tree.conclusion tree.nodeCount
      result.state).isSome = true := by
    unfold acceptedToken?
    rw [List.find?_isSome]
    exact ⟨token, tokenMem, accepts⟩
  simp [occurrenceMarkedB, result, spec.noError, found]

theorem branchingGridMarked_complete
    (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    branchingGridMarkedB fs tree.conclusion tree.nodeCount
      (fs.length + tree.nodeCount)
      (Sum.inr (encodeCertifiedNDTree fs tree covered)) = true := by
  exact occurrenceMarked_complete fs tree covered wellFormed closed

theorem exists_branchingGridMarked_of_closed_tree
    (fs : List Formula) (tree : NDTree)
    (covered : FormulaCovered fs tree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    ∃ assignment : BranchingAssignment fs tree.nodeCount
        (fs.length + tree.nodeCount),
      branchingGridMarkedB fs tree.conclusion tree.nodeCount
        (fs.length + tree.nodeCount) assignment = true := by
  exact ⟨Sum.inr (encodeCertifiedNDTree fs tree covered),
    branchingGridMarked_complete fs tree covered wellFormed closed⟩

/-- Every proof in the repository's ordinary implicational `Derives` system
has a finite `NDTree` whose computed dependencies are contained in its proof
context.  This closes the small representation gap between the inductive ND
judgment and the executable compiler's tree input. -/
theorem derives_has_finite_tree {context : List Formula} {goal : Formula}
    (derivation : Derives context goal) :
    ∃ tree : NDTree,
      tree.wellFormedB = true ∧
      tree.conclusion = goal ∧
      (∀ formula, formula ∈ tree.dependencies -> formula ∈ context) := by
  induction derivation with
  | @hyp context formula member =>
      refine ⟨.hyp formula, by simp [ProofSearch.Grid.NDTree.wellFormedB], rfl, ?_⟩
      intro other dependency
      simpa [ProofSearch.Grid.NDTree.dependencies] using (show other = formula from by
        simpa [ProofSearch.Grid.NDTree.dependencies] using dependency) ▸ member
  | @intro context alpha beta premise ih =>
      obtain ⟨premiseTree, premiseWellFormed, premiseConclusion,
        premiseSupport⟩ := ih
      refine ⟨.intro alpha beta premiseTree, ?_, rfl, ?_⟩
      · simp [ProofSearch.Grid.NDTree.wellFormedB, premiseWellFormed, premiseConclusion]
      · intro formula dependency
        simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups,
          List.mem_filter] at dependency
        have inExtended := premiseSupport formula dependency.1
        simp only [List.mem_cons] at inExtended
        rcases inExtended with equal | inContext
        · subst formula
          simp at dependency
        · exact inContext
  | @elim context alpha beta major minor ihMajor ihMinor =>
      obtain ⟨majorTree, majorWellFormed, majorConclusion, majorSupport⟩ :=
        ihMajor
      obtain ⟨minorTree, minorWellFormed, minorConclusion, minorSupport⟩ :=
        ihMinor
      refine ⟨.elim alpha beta majorTree minorTree, ?_, rfl, ?_⟩
      · simp [ProofSearch.Grid.NDTree.wellFormedB, majorWellFormed, minorWellFormed,
          majorConclusion, minorConclusion]
      · intro formula dependency
        simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups,
          List.mem_append] at dependency
        rcases dependency with inMajor | inMinor
        · exact majorSupport formula inMajor
        · exact minorSupport formula inMinor

/-- Unbounded proof-search equivalence.  The forward direction chooses the
finite formula universe consisting of the constructor formulas of the derived
tree and then uses the exact `nodeCount` compiler bounds. -/
theorem derivable_iff_exists_occurrenceMarked (goal : Formula) :
    Derives [] goal ↔
      ∃ (fs : List Formula) (height K : Nat)
        (assignment : OccurrenceAssignment fs.length height K),
        occurrenceMarkedB fs goal height K assignment = true := by
  constructor
  · intro derivation
    obtain ⟨tree, wellFormed, conclusion, support⟩ :=
      derives_has_finite_tree derivation
    have closed : tree.dependencies = [] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro formula member
      have impossible := support formula member
      simp at impossible
    let fs := tree.formulas
    have covered : FormulaCovered fs tree := by
      intro formula member
      exact member
    refine ⟨fs, tree.nodeCount, fs.length + tree.nodeCount,
      encodeCertifiedNDTree fs tree covered, ?_⟩
    simpa [conclusion] using
      occurrenceMarked_complete fs tree covered wellFormed closed
  · rintro ⟨fs, height, K, assignment, marked⟩
    exact occurrenceMarked_sound fs goal height K assignment marked

/-- Natural-number form of the instruction language.  It is useful for a
total executable compiler: lowering to finite indices is a separate step and
cannot silently manufacture an index. -/
inductive RawBranchInstruction where
  | fork (parent child : Nat)
  | intro (premise result antecedent conclusion : Nat)
  | elim (major minor result conclusion : Nat)
  deriving DecidableEq, Repr

structure RawCompilation where
  instructions : List RawBranchInstruction
  root : Nat
  deriving Repr

/-- Compile into the fresh interval beginning at `fs.length + offset`. -/
def compileRawFrom (fs : List Formula) (offset : Nat) : NDTree -> RawCompilation
  | .hyp formula =>
      { instructions := [.fork (fs.idxOf formula) (fs.length + offset)]
        root := fs.length + offset }
  | .intro alpha beta premise =>
      let compiled := compileRawFrom fs offset premise
      let root := fs.length + offset + premise.nodeCount
      { instructions := compiled.instructions ++
          [.intro compiled.root root (fs.idxOf alpha)
            (fs.idxOf (alpha >> beta))]
        root := root }
  | .elim _alpha beta major minor =>
      let compiledMajor := compileRawFrom fs offset major
      let compiledMinor := compileRawFrom fs (offset + major.nodeCount) minor
      let root := fs.length + offset + major.nodeCount + minor.nodeCount
      { instructions := compiledMajor.instructions ++
          compiledMinor.instructions ++
          [.elim compiledMajor.root compiledMinor.root root (fs.idxOf beta)]
        root := root }

theorem compileRawFrom_length (fs : List Formula) (offset : Nat)
    (tree : NDTree) :
    (compileRawFrom fs offset tree).instructions.length = tree.nodeCount := by
  induction tree generalizing offset with
  | hyp formula => simp [compileRawFrom, ProofSearch.Grid.NDTree.nodeCount]
  | intro alpha beta premise ih =>
      simp [compileRawFrom, ProofSearch.Grid.NDTree.nodeCount, ih]
  | elim alpha beta major minor ihMajor ihMinor =>
      simp [compileRawFrom, ProofSearch.Grid.NDTree.nodeCount, ihMajor, ihMinor, Nat.add_assoc]

theorem compileRawFrom_root (fs : List Formula) (offset : Nat)
    (tree : NDTree) :
    (compileRawFrom fs offset tree).root + 1 =
      fs.length + offset + tree.nodeCount := by
  cases tree <;> simp [compileRawFrom, ProofSearch.Grid.NDTree.nodeCount, Nat.add_assoc]

def natToFin? (bound value : Nat) : Option (Fin bound) :=
  if h : value < bound then some { val := value, isLt := h } else none

def RawBranchInstruction.lower? (n K : Nat) :
    RawBranchInstruction -> Option (BranchInstruction n K)
  | .fork parent child => do
      let parent <- natToFin? K parent
      let child <- natToFin? K child
      pure (.fork parent child)
  | .intro premise result antecedent conclusion => do
      let premise <- natToFin? K premise
      let result <- natToFin? K result
      let antecedent <- natToFin? n antecedent
      let conclusion <- natToFin? n conclusion
      pure (.intro premise result antecedent conclusion)
  | .elim major minor result conclusion => do
      let major <- natToFin? K major
      let minor <- natToFin? K minor
      let result <- natToFin? K result
      let conclusion <- natToFin? n conclusion
      pure (.elim major minor result conclusion)

def requiredBranchingHeight (tree : NDTree) : Nat := tree.nodeCount

def requiredCarrierBudget (fs : List Formula) (tree : NDTree) : Nat :=
  fs.length + tree.nodeCount

/-- Total lowering.  An out-of-range raw instruction becomes `idle`; the
coverage/bounds theorem below characterizes when that fallback is impossible.
-/
def compileProgram (fs : List Formula) (tree : NDTree) :
    List (BranchInstruction fs.length (requiredCarrierBudget fs tree)) :=
  (compileRawFrom fs 0 tree).instructions.map fun instruction =>
    (instruction.lower? fs.length (requiredCarrierBudget fs tree)).getD .idle

theorem compileProgram_length (fs : List Formula) (tree : NDTree) :
    (compileProgram fs tree).length = requiredBranchingHeight tree := by
  simp [compileProgram, requiredBranchingHeight, compileRawFrom_length]

def encodeNDTree (fs : List Formula) (tree : NDTree) :
    OccurrenceAssignment fs.length (requiredBranchingHeight tree)
      (requiredCarrierBudget fs tree) :=
  fun row =>
    (compileProgram fs tree)[row.val]?.getD .idle

def encodedResult (fs : List Formula) (tree : NDTree) :
    EvalResult fs.length (requiredCarrierBudget fs tree) :=
  evalOccurrence fs (requiredBranchingHeight tree)
    (requiredCarrierBudget fs tree) (encodeNDTree fs tree)

/-- The compiler's exact finite resource allocation. -/
theorem compiler_resource_bounds (fs : List Formula) (tree : NDTree) :
    (compileProgram fs tree).length = tree.nodeCount /\
      requiredBranchingHeight tree = tree.nodeCount /\
      requiredCarrierBudget fs tree = fs.length + tree.nodeCount := by
  constructor
  · exact compileProgram_length fs tree
  constructor <;> rfl

end ProofSearch.BranchingGrid
end DLDS
