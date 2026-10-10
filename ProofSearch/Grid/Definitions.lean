import Evaluator.Grid

open DLDS

namespace ProofSearch.Grid.Definitions

def A : Formula := #"A"
def B : Formula := #"B"
def AB : Formula := A >> B
def alpha : Formula := AB >> AB

def formulas : List Formula := [A, B, AB, alpha]

def layerFor (fs : List Formula) : GridLayer fs.length :=
  { nodes := fs.map (nodeForFormula fs)
    incoming := buildIncomingMap fs }

def gridFor (fs : List Formula) (transitions : Nat) :
    List (GridLayer fs.length) :=
  List.replicate (transitions + 1) (layerFor fs)

def initialVectorsFor (fs : List Formula) :
    List (List.Vector Bool fs.length) :=
  List.range fs.length |>.map fun i =>
    ⟨List.range fs.length |>.map (fun j => decide (j = i)), by simp⟩

def goalColumnFor (fs : List Formula) (goal : Formula) : Nat :=
  fs.idxOf goal

def stepAlphabet : List (Nat × Nat) :=
  [(0, 0),
   (1, 0),
   (2, 0), (2, 1), (2, 2),
   (3, 0), (3, 1), (3, 2), (3, 3),
   (4, 0), (4, 1)]

def tokensAtFinal {n : Nat} (layers : List (GridLayer n))
    (initialVectors : List (List.Vector Bool n)) (P : PathInput) :
    List (Token n) :=
  let numLevels := layers.length
  go P numLevels (numLevels - 1)
    (initialize_tokens initialVectors numLevels) layers
where
  go {n : Nat} (P : PathInput) (numLevels : Nat) :
      Nat → List (Token n) → List (GridLayer n) → List (Token n)
    | _, tokens, [] => tokens
    | _, tokens, [_] => tokens
    | level, tokens, current :: nextLayer :: tail =>
        let outputs := (evaluate_layer current tokens).1
        let next := propagate_tokens tokens P level numLevels outputs
        go P numLevels (level - 1) next (nextLayer :: tail)

def goalProducedB {n : Nat} (layers : List (GridLayer n))
    (initialVectors : List (List.Vector Bool n)) (P : PathInput)
    (goalColumn : Nat) : Bool :=
  match layers.getLast? with
  | none => false
  | some finalLayer =>
      match finalLayer.nodes[goalColumn]?, finalLayer.incoming[goalColumn]? with
      | some goalNode, some incoming =>
          let arrivals :=
            (tokensAtFinal layers initialVectors P).filter
              (·.current_column = goalColumn)
          !arrivals.isEmpty &&
            !nodeError goalNode incoming arrivals &&
            (selectedRuleIndex? incoming arrivals).isSome
      | _, _ => false

/-- The alphabet parameter preserves the uniform proof-search interface. -/
def markedBFor (fs : List Formula) (goal : Formula) (_alphabet : List (Nat × Nat))
    (transitions : Nat) (P : PathInput) : Bool :=
  let layers := gridFor fs transitions
  let iv := initialVectorsFor fs
  let g := goalColumnFor fs goal
  let result := get_eval_result layers iv P
  let discharged :=
    match result.1[g]? with
    | none => false
    | some v => v.toList.all (· = false)
  goalProducedB layers iv P g && !result.2 && discharged

def markedB (transitions : Nat) (P : PathInput) : Bool :=
  markedBFor formulas alpha stepAlphabet transitions P

def allStop (transitions : Nat) : PathInput :=
  List.replicate formulas.length (List.replicate transitions (0, 0))

def lcg (seed : Nat) : Nat :=
  (1103515245 * seed + 12345) % 2147483648

def randomCells (alphabet : List (Nat × Nat)) :
    Nat → Nat → List (Nat × Nat) × Nat
  | 0, seed => ([], seed)
  | n + 1, seed =>
      let seed' := lcg seed
      let x := alphabet.getD (seed' % alphabet.length) (0, 0)
      let rest := randomCells alphabet n seed'
      (x :: rest.1, rest.2)

def chunk : Nat → Nat → List α → List (List α)
  | 0, _, _ => []
  | n + 1, width, xs => xs.take width :: chunk n width (xs.drop width)

def randomPath (columns transitions : Nat) (alphabet : List (Nat × Nat))
    (seed : Nat) : PathInput × Nat :=
  let cells := randomCells alphabet (columns * transitions) seed
  (chunk columns transitions cells.1, cells.2)

structure SampleReport where
  samples : Nat
  marked : Nat
  firstMarked : List PathInput
  deriving Repr

def sampleMarked (fs : List Formula) (goal : Formula)
    (alphabet : List (Nat × Nat)) (transitions samples seed0 : Nat) :
    SampleReport :=
  let result := (List.range samples).foldl (fun state _ =>
    let sample := randomPath fs.length transitions alphabet state.1
    let marked := markedBFor fs goal alphabet transitions sample.1
    let examples :=
      if marked && state.2.2.length < 3 then state.2.2 ++ [sample.1]
      else state.2.2
    (sample.2, (state.2.1 + if marked then 1 else 0, examples)))
    (seed0, (0, []))
  { samples := samples, marked := result.2.1, firstMarked := result.2.2 }

def spaceSize (alphabetSize columns transitions : Nat) : Nat :=
  alphabetSize ^ (columns * transitions)

def alphabetFor (fs : List Formula) : List (Nat × Nat) :=
  (0, 0) :: (buildIncomingMap fs).zipIdx.flatMap fun (incoming, target) =>
    (target + 1, 0) ::
      (List.range (flattenIncoming incoming).length).map fun i =>
        (target + 1, i + 1)

abbrev Histogram := List (Nat × Nat)

def histogramAdd (h : Histogram) (key amount : Nat) : Histogram :=
  match h.findIdx? fun p => p.1 = key with
  | none => h ++ [(key, amount)]
  | some i =>
      h.modify i fun p => (p.1, p.2 + amount)

def histogramMerge (a b : Histogram) : Histogram :=
  b.foldl (fun h p => histogramAdd h p.1 p.2) a

def histogramShift (h : Histogram) (amount : Nat) : Histogram :=
  h.map fun p => (p.1 + amount, p.2)

def histogramTotal (h : Histogram) : Nat :=
  h.foldl (fun n p => n + p.2) 0

def tokenEqB {n : Nat} (x y : Token n) : Bool :=
  x.origin_column = y.origin_column &&
  x.source_column = y.source_column &&
  x.current_level = y.current_level &&
  x.current_column = y.current_column &&
  x.dep_vector.toList = y.dep_vector.toList &&
  x.input_label = y.input_label

def tokenStateEqB {n : Nat} (xs ys : List (Token n)) : Bool :=
  xs.length = ys.length &&
    (xs.zip ys).all fun p => tokenEqB p.1 p.2

abbrev StateEntry (n : Nat) := List (Token n) × Histogram

def stateAdd {n : Nat} (states : List (StateEntry n))
    (tokens : List (Token n)) (hist : Histogram) : List (StateEntry n) :=
  match states.findIdx? fun s => tokenStateEqB s.1 tokens with
  | none => states ++ [(tokens, hist)]
  | some i => states.modify i fun s => (s.1, histogramMerge s.2 hist)

def choiceWords (alphabet : List α) : Nat → List (List α)
  | 0 => [[]]
  | n + 1 =>
      alphabet.flatMap fun x =>
        (choiceWords alphabet n).map fun xs => x :: xs

def propagateChosen {n : Nat} (outputs : List (List.Vector Bool n))
    (level : Nat) (tokens : List (Token n)) (choices : List (Nat × Nat)) :
    List (Token n) :=
  (tokens.zip choices).filterMap fun p =>
    let token := p.1
    let choice := p.2
    if choice.1 = 0 then none
    else
      match outputs[token.current_column]? with
      | none => none
      | some out =>
          some { token with
            source_column := token.current_column
            current_level := level - 1
            current_column := choice.1 - 1
            dep_vector := out
            input_label := choice.2 }

def advanceStates {n : Nat} (layer : GridLayer n)
    (alphabet : List (Nat × Nat)) (columns level : Nat)
    (states : List (StateEntry n)) : List (StateEntry n) :=
  states.foldl (fun next state =>
    let evaluated := evaluate_layer layer state.1
    if evaluated.2 then next
    else
      let unreadHere := columns - state.1.length
      let shifted := histogramShift state.2 unreadHere
      (choiceWords alphabet state.1.length).foldl (fun acc choices =>
        stateAdd acc (propagateChosen evaluated.1 level state.1 choices) shifted)
        next) []

structure ExactReport where
  transitions : Nat
  space : Nat
  activeTraceHistogram : Histogram
  activeTraceCount : Nat
  markedAssignments : Nat
  deriving Repr

def exactMarked (fs : List Formula) (goal : Formula) (transitions : Nat) :
    ExactReport :=
  let alphabet := alphabetFor fs
  let layer := layerFor fs
  let initial : List (StateEntry fs.length) :=
    [(initialize_tokens (initialVectorsFor fs) (transitions + 1), [(0, 1)])]
  let levels := (List.range transitions).reverse
  let finalStates := levels.foldl (fun states level =>
    advanceStates layer alphabet fs.length (level + 1) states) initial
  let goalCol := goalColumnFor fs goal
  let markedHist := finalStates.foldl (fun hist state =>
    let evaluated := evaluate_layer layer state.1
    let arrivals := state.1.filter (·.current_column = goalCol)
    let produced :=
      match layer.nodes[goalCol]?, layer.incoming[goalCol]? with
      | some cnode, some incoming =>
          !arrivals.isEmpty && !nodeError cnode incoming arrivals &&
            (selectedRuleIndex? incoming arrivals).isSome
      | _, _ => false
    let discharged :=
      match evaluated.1[goalCol]? with
      | none => false
      | some v => v.toList.all (· = false)
    if !evaluated.2 && produced && discharged then
      histogramMerge hist state.2
    else hist) []
  let markedAssignments := markedHist.foldl
    (fun total p => total + p.2 * alphabet.length ^ p.1) 0
  { transitions := transitions
    space := spaceSize alphabet.length fs.length transitions
    activeTraceHistogram := markedHist
    activeTraceCount := histogramTotal markedHist
    markedAssignments := markedAssignments }

def witness1 : PathInput :=
  [[(2, 2), (3, 1), (4, 1)],
   [(0, 0), (0, 0), (0, 0)],
   [(2, 1), (0, 0), (0, 0)],
   [(0, 0), (0, 0), (0, 0)]]

def witness2 : PathInput :=
  [[(2, 2), (0, 0), (0, 0)],
   [(0, 0), (0, 0), (0, 0)],
   [(2, 1), (3, 1), (4, 1)],
   [(0, 0), (0, 0), (0, 0)]]

def witness3 : PathInput :=
  [[(2, 2), (3, 1), (4, 1)],
   [(0, 0), (4, 1), (3, 3)],
   [(2, 1), (0, 0), (4, 0)],
   [(0, 0), (2, 2), (1, 0)]]

def unprovableFormulas : List Formula := [A, B, AB]
def unprovableAlphabet : List (Nat × Nat) :=
  [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)]

def sampleSize : Nat := 200000

def C : Formula := #"C"
def CA : Formula := C >> A
def BCA : Formula := B >> CA
def beta : Formula := A >> BCA
def largerFormulas : List Formula := [A, B, C, CA, BCA, beta]


end ProofSearch.Grid.Definitions
