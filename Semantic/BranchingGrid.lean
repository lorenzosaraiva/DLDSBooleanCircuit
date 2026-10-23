import SpaceACounting



















namespace Semantic
namespace BranchingGrid

open GridCertificateSoundness

abbrev CarrierId (K : Nat) := Fin K



structure BranchToken (n K : Nat) where
  carrier : CarrierId K
  formulaColumn : Fin n
  currentLevel : Nat
  dependency : List.Vector Bool n
  tree : NDTree
  deriving BEq


abbrev BranchState (n K : Nat) := CarrierId K -> Option (BranchToken n K)




inductive BranchInstruction (n K : Nat) where
  | idle
  | stop (carrier : CarrierId K)
  | move (source target : CarrierId K)
  | fork (parent child : CarrierId K)
  | intro (premise result : CarrierId K)
      (antecedent conclusion : Fin n)
  | elim (major minor result : CarrierId K) (conclusion : Fin n)
  deriving DecidableEq, Repr, Fintype



abbrev OccurrenceAssignment (n height K : Nat) :=
  Fin height -> BranchInstruction n K




abbrev BranchingAssignment (fs : List Formula) (height K : Nat) :=
  Sum (SpaceACounting.Assignment fs height)
    (OccurrenceAssignment fs.length height K)

def BranchState.set {n K : Nat} (state : BranchState n K)
    (carrier : CarrierId K) (value : Option (BranchToken n K)) :
    BranchState n K :=
  fun other => if other = carrier then value else state other

@[simp] theorem BranchState.set_eq {n K : Nat} (state : BranchState n K)
    (carrier : CarrierId K) (value : Option (BranchToken n K)) :
    state.set carrier value carrier = value := by
  simp [BranchState.set]

@[simp] theorem BranchState.set_ne {n K : Nat} (state : BranchState n K)
    (carrier other : CarrierId K) (value : Option (BranchToken n K))
    (hne : Not (other = carrier)) :
    state.set carrier value other = state other := by
  simp [BranchState.set, hne]

def mkToken (fs : List Formula) {K : Nat} (carrier : CarrierId K)
    (formulaColumn : Fin fs.length) (level : Nat) (tree : NDTree) :
    BranchToken fs.length K where
  carrier := carrier
  formulaColumn := formulaColumn
  currentLevel := level
  dependency := depVector fs tree.dependencies
  tree := tree

def ageToken {n K : Nat} (level : Nat) (token : BranchToken n K) :
    BranchToken n K :=
  { token with currentLevel := level }

def ageState {n K : Nat} (level : Nat) (state : BranchState n K) :
    BranchState n K :=
  fun carrier => (state carrier).map (ageToken level)




def initialState (fs : List Formula) (K height : Nat) :
    BranchState fs.length K :=
  fun carrier =>
    if h : carrier.val < fs.length then
      let column : Fin fs.length := { val := carrier.val, isLt := h }
      some (mkToken fs carrier column height (.hyp (fs.get column)))
    else
      none

def liveTokens {n K : Nat} (state : BranchState n K) :
    List (BranchToken n K) :=
  (List.finRange K).filterMap state

structure StepResult (n K : Nat) where
  state : BranchState n K
  error : Bool

def carrierAvailableAfterOne {n K : Nat} (state : BranchState n K)
    (consumed target : CarrierId K) : Bool :=
  decide (target = consumed) || (state target).isNone

def carrierAvailableAfterTwo {n K : Nat} (state : BranchState n K)
    (first second target : CarrierId K) : Bool :=
  decide (target = first) || decide (target = second) || (state target).isNone




def executeInstruction (fs : List Formula) {K : Nat} (level : Nat)
    (state : BranchState fs.length K)
    (instruction : BranchInstruction fs.length K) :
    StepResult fs.length K :=
  let aged := ageState level state
  match instruction with
  | .idle => { state := aged, error := false }
  | .stop carrier =>
      match aged carrier with
      | none => { state := aged, error := true }
      | some _ => { state := aged.set carrier none, error := false }
  | .move source target =>
      match aged source with
      | none => { state := aged, error := true }
      | some token =>
          if carrierAvailableAfterOne aged source target then
            let cleared := aged.set source none
            let moved := { token with carrier := target }
            { state := cleared.set target (some moved), error := false }
          else
            { state := aged, error := true }
  | .fork parent child =>
      match aged parent with
      | none => { state := aged, error := true }
      | some token =>
          if parent = child || (aged child).isSome then
            { state := aged, error := true }
          else
            let copied := { token with carrier := child }
            { state := aged.set child (some copied), error := false }
  | .intro premise result antecedent conclusion =>
      match aged premise with
      | none => { state := aged, error := true }
      | some token =>
          let alpha := fs.get antecedent
          let beta := fs.get token.formulaColumn
          if decide (fs.get conclusion = alpha >> beta) &&
              carrierAvailableAfterOne aged premise result then
            let tree := NDTree.intro alpha beta token.tree
            let cleared := aged.set premise none
            { state := cleared.set result
                (some (mkToken fs result conclusion level tree)), error := false }
          else
            { state := aged, error := true }
  | .elim major minor result conclusion =>
      match aged major, aged minor with
      | some majorToken, some minorToken =>
          if major = minor then
            { state := aged, error := true }
          else
            match fs.get majorToken.formulaColumn with
            | .atom _ => { state := aged, error := true }
            | .implication alpha beta =>
                if decide (fs.get minorToken.formulaColumn = alpha) &&
                    decide (fs.get conclusion = beta) &&
                    carrierAvailableAfterTwo aged major minor result then
                  let tree := NDTree.elim alpha beta
                    majorToken.tree minorToken.tree
                  let cleared := (aged.set major none).set minor none
                  { state := cleared.set result
                      (some (mkToken fs result conclusion level tree)),
                    error := false }
                else
                  { state := aged, error := true }
      | _, _ => { state := aged, error := true }

structure EvalResult (n K : Nat) where
  state : BranchState n K
  error : Bool

def evalProgramList (fs : List Formula) {K : Nat} :
    Nat -> BranchState fs.length K -> Bool ->
      List (BranchInstruction fs.length K) -> EvalResult fs.length K
  | _, state, error, [] => { state := state, error := error }
  | level, state, error, instruction :: rest =>
      let step := executeInstruction fs (level - 1) state instruction
      evalProgramList fs (level - 1) step.state (error || step.error) rest

def occurrenceProgram {fs : List Formula} {height K : Nat}
    (assignment : OccurrenceAssignment fs.length height K) :
    List (BranchInstruction fs.length K) :=
  List.ofFn assignment

def evalOccurrence (fs : List Formula) (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K) :
    EvalResult fs.length K :=
  evalProgramList fs height (initialState fs K height)
    (decide (K < fs.length)) (occurrenceProgram assignment)

def dependencySupportedB (fs : List Formula) (tree : NDTree) : Bool :=
  tree.dependencies.all fun formula => fs.contains formula

def tokenValidB (fs : List Formula) {K : Nat}
    (token : BranchToken fs.length K) : Bool :=
  token.tree.wellFormedB &&
    (fs.get token.formulaColumn == token.tree.conclusion) &&
    (token.dependency == depVector fs token.tree.dependencies) &&
    dependencySupportedB fs token.tree

def tokenAcceptsB (fs : List Formula) (goal : Formula) (height : Nat) {K : Nat}
    (token : BranchToken fs.length K) : Bool :=
  tokenValidB fs token && (token.tree.conclusion == goal) &&
    token.dependency.toList.all (fun bit => bit = false) &&
    decide (token.tree.height <= height)

def acceptedToken? (fs : List Formula) (goal : Formula) (height : Nat) {K : Nat}
    (state : BranchState fs.length K) : Option (BranchToken fs.length K) :=
  (liveTokens state).find? (tokenAcceptsB fs goal height)

def occurrenceMarkedB (fs : List Formula) (goal : Formula)
    (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K) : Bool :=
  let result := evalOccurrence fs height K assignment
  !result.error && (acceptedToken? fs goal height result.state).isSome



def branchingGridMarkedB (fs : List Formula) (goal : Formula)
    (height K : Nat) (assignment : BranchingAssignment fs height K) : Bool :=
  match assignment with
  | Sum.inl old => GridCertificateSoundness.gridMarkedB
      fs goal height old.paths
  | Sum.inr occurrence => occurrenceMarkedB fs goal height K occurrence

def extractOccurrenceGoalCone (fs : List Formula) (goal : Formula)
    (height K : Nat)
    (assignment : OccurrenceAssignment fs.length height K) : Option NDTree :=
  (acceptedToken? fs goal height
    (evalOccurrence fs height K assignment).state).map
    (fun token => token.tree)

def extractBranchingGoalCone (fs : List Formula) (goal : Formula)
    (height K : Nat) (assignment : BranchingAssignment fs height K) :
    Option NDTree :=
  match assignment with
  | Sum.inl old => GridCertificateSoundness.extractGoalCone
      fs goal height old.paths
  | Sum.inr occurrence => extractOccurrenceGoalCone
      fs goal height K occurrence


def embedAssignment {fs : List Formula} {height K : Nat}
    (old : SpaceACounting.Assignment fs height) :
    BranchingAssignment fs height K :=
  Sum.inl old

@[simp] theorem branchingGridMarkedB_embedAssignment
    (fs : List Formula) (goal : Formula) (height K : Nat)
    (old : SpaceACounting.Assignment fs height) :
    branchingGridMarkedB fs goal height K (embedAssignment old) =
      GridCertificateSoundness.gridMarkedB fs goal height old.paths := by
  rfl

@[simp] theorem extractBranchingGoalCone_embedAssignment
    (fs : List Formula) (goal : Formula) (height K : Nat)
    (old : SpaceACounting.Assignment fs height) :
    extractBranchingGoalCone fs goal height K (embedAssignment old) =
      GridCertificateSoundness.extractGoalCone fs goal height old.paths := by
  rfl



abbrev InstructionCode (n K : Nat) :=
  Sum Unit
    (Sum (Fin K)
      (Sum (Prod (Fin K) (Fin K))
        (Sum (Prod (Fin K) (Fin K))
          (Sum (Prod (Prod (Fin K) (Fin K)) (Prod (Fin n) (Fin n)))
            (Prod (Prod (Prod (Fin K) (Fin K)) (Fin K)) (Fin n))))))

def instructionEquiv (n K : Nat) :
    Equiv (BranchInstruction n K) (InstructionCode n K) where
  toFun
    | .idle => Sum.inl ()
    | .stop carrier => Sum.inr (Sum.inl carrier)
    | .move source target => Sum.inr (Sum.inr (Sum.inl (source, target)))
    | .fork parent child =>
        Sum.inr (Sum.inr (Sum.inr (Sum.inl (parent, child))))
    | .intro premise result antecedent conclusion =>
        Sum.inr (Sum.inr (Sum.inr (Sum.inr
          (Sum.inl ((premise, result), (antecedent, conclusion))))))
    | .elim major minor result conclusion =>
        Sum.inr (Sum.inr (Sum.inr (Sum.inr
          (Sum.inr (((major, minor), result), conclusion)))))
  invFun
    | Sum.inl () => .idle
    | Sum.inr (Sum.inl carrier) => .stop carrier
    | Sum.inr (Sum.inr (Sum.inl (source, target))) => .move source target
    | Sum.inr (Sum.inr (Sum.inr (Sum.inl (parent, child)))) =>
        .fork parent child
    | Sum.inr (Sum.inr (Sum.inr (Sum.inr
        (Sum.inl ((premise, result), (antecedent, conclusion)))))) =>
        .intro premise result antecedent conclusion
    | Sum.inr (Sum.inr (Sum.inr (Sum.inr
        (Sum.inr (((major, minor), result), conclusion))))) =>
        .elim major minor result conclusion
  left_inv instruction := by cases instruction <;> rfl
  right_inv code := by
    rcases code with value | rest
    · cases value
      rfl
    · rcases rest with stop | rest
      · rfl
      · rcases rest with move | rest
        · rfl
        · rcases rest with fork | rest
          · rfl
          · rcases rest with intro | elim <;> rfl

theorem instructionAlphabet_card (n K : Nat) :
    Fintype.card (BranchInstruction n K) =
      1 + K + K ^ 2 + K ^ 2 + K ^ 2 * n ^ 2 + K ^ 3 * n := by
  rw [Fintype.card_congr (instructionEquiv n K)]
  simp [InstructionCode]
  ring

theorem occurrenceAssignment_card (n height K : Nat) :
    Fintype.card (OccurrenceAssignment n height K) =
      (1 + K + K ^ 2 + K ^ 2 + K ^ 2 * n ^ 2 + K ^ 3 * n) ^ height := by
  rw [show Fintype.card (OccurrenceAssignment n height K) =
      Fintype.card (BranchInstruction n K) ^ Fintype.card (Fin height) by
    simp [OccurrenceAssignment]]
  rw [instructionAlphabet_card]
  simp

theorem branchingAssignment_card (fs : List Formula) (height K : Nat) :
    Fintype.card (BranchingAssignment fs height K) =
      (ProofSearchProbe.alphabetFor fs).length ^ (fs.length * height) +
      (1 + K + K ^ 2 + K ^ 2 + K ^ 2 * fs.length ^ 2 +
        K ^ 3 * fs.length) ^ height := by
  simp only [BranchingAssignment, Fintype.card_sum]
  rw [SpaceACounting.assignment_card]
  rw [occurrenceAssignment_card]

end BranchingGrid
end Semantic
