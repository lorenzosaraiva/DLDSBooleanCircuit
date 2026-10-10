import ProofSearch.Grid.Definitions
import Examples.Flow.ClosedImplication

/-!
# Historical Grid proof search and shared proof-tree types

`gridMarkedB` is the historical Space A marker. This module also defines the
shared foundational `NDTree` and `Derives` types; despite the filename, it is
therefore a core chokepoint. Branching Grid V2 is the separate current Grid
family in `ProofSearch.BranchingGrid.Definitions`.
-/

open DLDS

namespace ProofSearch.Grid

structure ActiveCell where
  level : Nat
  column : Nat
  formula : Formula
  arrivals : Nat
  rule : String
  output : List Formula
  deriving Repr

structure ActiveWire where
  sourceLevel : Nat
  sourceColumn : Nat
  targetLevel : Nat
  targetColumn : Nat
  inputLabel : Nat
  dependency : List Formula
  deriving Repr

structure ActiveTrace where
  cells : List ActiveCell
  wires : List ActiveWire
  hadError : Bool
  deriving Repr

def vectorDependency {n : Nat} (fs : List Formula)
    (v : List.Vector Bool n) : List Formula :=
  (fs.zip v.toList).filterMap fun pair =>
    if pair.2 then some pair.1 else none

def selectedRuleName {n : Nat} (node : CircuitNode n)
    (incoming : NodeIncoming) (tokens : List (Token n)) : String :=
  match selectedRuleIndex? incoming tokens with
  | none => "none"
  | some ruleIndex =>
      match node.rules[ruleIndex]? with
      | none => "out-of-range"
      | some r =>
          match r.type with
          | .intro _ => "intro"
          | .elim => "elim"
          | .repetition => "repetition"

def cellsAt {n : Nat} (fs : List Formula) (level : Nat)
    (layer : GridLayer n) (tokens : List (Token n))
    (outputs : List (List.Vector Bool n)) : List ActiveCell :=
  fs.zipIdx.filterMap fun (formula, column) =>
    let arrivals := tokens.filter (fun token => token.current_column = column)
    if arrivals.isEmpty then none
    else
      match layer.nodes[column]?, layer.incoming[column]?, outputs[column]? with
      | some cnode, some incoming, some output =>
          some {
            level, column, formula, arrivals := arrivals.length
            rule := selectedRuleName cnode incoming arrivals
            output := vectorDependency fs output
          }
      | _, _, _ => none

def wireForToken {n : Nat} (fs : List Formula) (paths : PathInput)
    (currentLevel numLevels : Nat) (outputs : List (List.Vector Bool n))
    (token : Token n) : Option ActiveWire :=
  if hPath : token.origin_column < paths.length then
    let path := paths.get ⟨token.origin_column, hPath⟩
    if hLevel : currentLevel > 0 ∧ numLevels - currentLevel - 1 < path.length then
      let stepIndex := numLevels - currentLevel - 1
      let choice := path.get ⟨stepIndex, hLevel.2⟩
      if choice.1 = 0 then none
      else if hOut : token.current_column < outputs.length then
        let output := outputs.get ⟨token.current_column, hOut⟩
        some {
          sourceLevel := currentLevel
          sourceColumn := token.current_column
          targetLevel := currentLevel - 1
          targetColumn := choice.1 - 1
          inputLabel := choice.2
          dependency := vectorDependency fs output
        }
      else none
    else none
  else none

def traceFrom {n : Nat} (fs : List Formula) (paths : PathInput)
    (numLevels : Nat) : Nat -> List (Token n) -> List (GridLayer n) -> ActiveTrace
  | _, _, [] => { cells := [], wires := [], hadError := false }
  | level, tokens, layer :: rest =>
      let evaluated := evaluate_layer layer tokens
      let here := cellsAt fs level layer tokens evaluated.1
      match rest with
      | [] => { cells := here, wires := [], hadError := evaluated.2 }
      | _ =>
          let wires := tokens.filterMap (wireForToken fs paths level numLevels evaluated.1)
          let nextTokens := propagate_tokens tokens paths level numLevels evaluated.1
          let tail := traceFrom fs paths numLevels (level - 1) nextTokens rest
          { cells := here ++ tail.cells
            wires := wires ++ tail.wires
            hadError := evaluated.2 || tail.hadError }

def activeTrace (fs : List Formula) (transitions : Nat) (paths : PathInput) : ActiveTrace :=
  let layers := ProofSearch.Grid.Definitions.gridFor fs transitions
  let numLevels := layers.length
  let tokens := initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs) numLevels
  traceFrom fs paths numLevels (numLevels - 1) tokens layers

def vertexAt (fs : List Formula) (transitions level column : Nat) : Vertex :=
  Vertex.node (level * fs.length + column) level (fs.getD column default)
    (level = transitions) false []

def graphOf (fs : List Formula) (transitions : Nat) (paths : PathInput) : Graph :=
  let trace := activeTrace fs transitions paths
  let nodes := trace.cells.map fun cell =>
    vertexAt fs transitions cell.level cell.column
  let edges := trace.wires.map fun wire =>
    Deduction.edge
      (vertexAt fs transitions wire.sourceLevel wire.sourceColumn)
      (vertexAt fs transitions wire.targetLevel wire.targetColumn)
      0 wire.dependency
  Graph.dlds nodes edges []

abbrev CellAddress := Nat × Nat

def wireSource (wire : ActiveWire) : CellAddress :=
  (wire.sourceLevel, wire.sourceColumn)

def wireTarget (wire : ActiveWire) : CellAddress :=
  (wire.targetLevel, wire.targetColumn)

def backwardCone : Nat -> List ActiveWire -> List CellAddress -> List CellAddress
  | 0, _, reached => reached.eraseDups
  | fuel + 1, wires, reached =>
      let predecessors := wires.filterMap fun wire =>
        if reached.contains (wireTarget wire) then some (wireSource wire) else none
      let next := (reached ++ predecessors).eraseDups
      if next.length = reached.eraseDups.length then next
      else backwardCone fuel wires next

def goalConeTrace (fs : List Formula) (goal : Formula) (trace : ActiveTrace) : ActiveTrace :=
  let goalAddress := (0, fs.idxOf goal)
  let reached := backwardCone (trace.cells.length + 1) trace.wires [goalAddress]
  { cells := trace.cells.filter fun cell => reached.contains (cell.level, cell.column)
    wires := trace.wires.filter fun wire => reached.contains (wireTarget wire)
    hadError := trace.hadError }

def graphOfTrace (fs : List Formula) (transitions : Nat) (trace : ActiveTrace) : Graph :=
  Graph.dlds
    (trace.cells.map fun cell => vertexAt fs transitions cell.level cell.column)
    (trace.wires.map fun wire =>
      Deduction.edge
        (vertexAt fs transitions wire.sourceLevel wire.sourceColumn)
        (vertexAt fs transitions wire.targetLevel wire.targetColumn)
        0 wire.dependency)
    []

def rootDischargedB (d : Graph) : Bool :=
  d.NODES.all fun v =>
    if (get_rule.outgoing v d).isEmpty then
      ruleShapeOKB v d && decide (outDep v d = [])
    else true

def validDLDSB (d : Graph) : Bool :=
  (d.NODES.all fun n1 => d.NODES.all fun n2 =>
    decide (n1.NUMBER = n2.NUMBER -> n1 = n2)) &&
  (d.EDGES.all fun edge => decide (edge.START ∈ d.NODES ∧ edge.END ∈ d.NODES)) &&
  (d.PATHS.all fun path => decide (path.START ∈ d.NODES ∧ path.END ∈ d.NODES)) &&
  (d.EDGES.all fun edge => decide (edge.START.LEVEL = edge.END.LEVEL + 1)) &&
  (d.EDGES.all fun e1 => d.EDGES.all fun e2 =>
    decide ((e1.START = e2.START ∧ e1.END = e2.END ∧ e1.COLOUR = e2.COLOUR) -> e1 = e2)) &&
  (d.PATHS.all fun p1 => d.PATHS.all fun p2 =>
    decide ((p1.START = p2.START ∧ p1.END = p2.END ∧ p1.COLOURS = p2.COLOURS) -> p1 = p2)) &&
  (d.NODES.all fun v =>
    if v.HYPOTHESIS then decide (get_rule.incoming v d = []) else true) &&
  (d.NODES.all fun v =>
    if (get_rule.outgoing v d).isEmpty then true
    else ruleShapeOKB v d &&
      ((get_rule.outgoing v d).all fun edge => decide (edge.DEPENDENCY = outDep v d))) &&
  rootDischargedB d && colorAcyclicityB d && ancestorEdgesB d &&
  ancestorBackwayInformationB d && nonNestedAncestorEdgesB d

def packageReport (d : Graph) :=
  (d.NODES.length,
   d.EDGES.length,
   (d.NODES.filter fun v => (get_rule.outgoing v d).isEmpty).map
      (fun v => (v.NUMBER, v.FORMULA)),
   DLDS.invariantTable d,
   rootDischargedB d,
   validDLDSB d,
   DLDS.failedNodes d)

/- A compact natural-deduction proof object.  Repetition is erased by the
   extractor, so every constructor is a genuine ND rule. -/
inductive NDTree where
  | hyp (formula : Formula)
  | intro (alpha beta : Formula) (premise : NDTree)
  | elim (alpha beta : Formula) (major minor : NDTree)
  deriving Repr, BEq

def NDTree.conclusion : NDTree -> Formula
  | .hyp formula => formula
  | .intro alpha beta _ => alpha >> beta
  | .elim _ beta _ _ => beta

def NDTree.dependencies : NDTree -> List Formula
  | .hyp formula => [formula]
  | .intro alpha _ premise =>
      (premise.dependencies.filter fun formula => formula != alpha).eraseDups
  | .elim _ _ major minor => (major.dependencies ++ minor.dependencies).eraseDups

def NDTree.height : NDTree -> Nat
  | .hyp _ => 0
  | .intro _ _ premise => premise.height + 1
  | .elim _ _ major minor => max major.height minor.height + 1

def NDTree.wellFormedB : NDTree -> Bool
  | .hyp _ => true
  | .intro _ beta premise => premise.wellFormedB && premise.conclusion == beta
  | .elim alpha beta major minor =>
      major.wellFormedB && minor.wellFormedB &&
      major.conclusion == (alpha >> beta) && minor.conclusion == alpha

inductive Derives : List Formula -> Formula -> Prop where
  | hyp {context formula} : formula ∈ context -> Derives context formula
  | intro {context alpha beta} :
      Derives (alpha :: context) beta -> Derives context (alpha >> beta)
  | elim {context alpha beta} :
      Derives context (alpha >> beta) -> Derives context alpha -> Derives context beta

theorem Derives.mono {context context' formula}
    (derivation : Derives context formula)
    (subset : ∀ phi, phi ∈ context -> phi ∈ context') :
    Derives context' formula := by
  induction derivation generalizing context' with
  | hyp member => exact .hyp (subset _ member)
  | @intro context alpha beta premise ih =>
      apply ProofSearch.Grid.Derives.intro
      apply ih (context' := alpha :: context')
      intro phi member
      simp only [List.mem_cons] at member ⊢
      rcases member with equal | member
      · exact Or.inl equal
      · exact Or.inr (subset phi member)
  | elim major minor ihMajor ihMinor =>
      exact .elim (ihMajor subset) (ihMinor subset)

theorem NDTree.derives_dependencies (tree : NDTree)
    (wellFormed : tree.wellFormedB = true) :
    Derives tree.dependencies tree.conclusion := by
  induction tree with
  | hyp formula =>
      exact ProofSearch.Grid.Derives.hyp (List.Mem.head [])
  | intro alpha beta premise ih =>
      simp only [ProofSearch.Grid.NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed
      obtain ⟨premiseWellFormed, premiseConclusion⟩ := wellFormed
      have premiseDerives := ih premiseWellFormed
      rw [premiseConclusion] at premiseDerives
      apply ProofSearch.Grid.Derives.intro
      apply ProofSearch.Grid.Derives.mono premiseDerives
      intro phi member
      by_cases equal : phi = alpha
      · subst phi
        exact List.Mem.head _
      · exact List.mem_cons_of_mem _ (by
          simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups, List.mem_filter]
          exact ⟨member, by simpa using equal⟩)
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [ProofSearch.Grid.NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed
      obtain ⟨⟨⟨majorWellFormed, minorWellFormed⟩, majorConclusion⟩,
        minorConclusion⟩ := wellFormed
      have majorDerives := ihMajor majorWellFormed
      have minorDerives := ihMinor minorWellFormed
      rw [majorConclusion] at majorDerives
      rw [minorConclusion] at minorDerives
      apply ProofSearch.Grid.Derives.elim
      · apply ProofSearch.Grid.Derives.mono majorDerives
        intro phi member
        simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups, List.mem_append]
        exact Or.inl member
      · apply ProofSearch.Grid.Derives.mono minorDerives
        intro phi member
        simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_eraseDups, List.mem_append]
        exact Or.inr member

theorem extracted_certificate_sound {tree : NDTree} {goal : Formula}
    (wellFormed : tree.wellFormedB = true)
    (conclusion : tree.conclusion = goal)
    (discharged : tree.dependencies = []) :
    Derives [] goal := by
  have proof := tree.derives_dependencies wellFormed
  rw [discharged, conclusion] at proof
  exact proof

structure Snapshot (n : Nat) where
  level : Nat
  layer : GridLayer n
  tokens : List (Token n)
  outputs : List (List.Vector Bool n)
  error : Bool

def snapshotsFrom {n : Nat} (paths : PathInput) (numLevels : Nat) :
    Nat -> List (Token n) -> List (GridLayer n) -> List (Snapshot n)
  | _, _, [] => []
  | level, tokens, layer :: rest =>
      let evaluated := evaluate_layer layer tokens
      let here : Snapshot n := {
        level, layer, tokens, outputs := evaluated.1, error := evaluated.2
      }
      here :: snapshotsFrom paths numLevels (level - 1)
        (propagate_tokens tokens paths level numLevels evaluated.1) rest

def snapshots (fs : List Formula) (transitions : Nat) (paths : PathInput) :=
  let layers := ProofSearch.Grid.Definitions.gridFor fs transitions
  let numLevels := layers.length
  snapshotsFrom paths numLevels (numLevels - 1)
    (initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs) numLevels) layers

/-- Adjacent snapshots are related only by the evaluator's propagation step.
    Keeping this relation separate prevents later proofs from unfolding the
    replicated grid or normalizing all snapshots at once. -/
def SnapshotsLinked {n : Nat} (paths : PathInput) (numLevels : Nat)
    (ss : List (Snapshot n)) : Prop :=
  ∀ i current next,
    ss[i]? = some current → ss[i + 1]? = some next →
    next.tokens = propagate_tokens current.tokens paths current.level
      numLevels current.outputs

lemma snapshotsFrom_length {n : Nat} (paths : PathInput) (numLevels level : Nat)
    (tokens : List (Token n)) (layers : List (GridLayer n)) :
    (snapshotsFrom paths numLevels level tokens layers).length = layers.length := by
  induction layers generalizing level tokens with
  | nil => rfl
  | cons layer rest ih =>
      simp only [snapshotsFrom, List.length_cons]
      rw [ih]

lemma snapshotsFrom_linked {n : Nat} (paths : PathInput) (numLevels level : Nat)
    (tokens : List (Token n)) (layers : List (GridLayer n)) :
    SnapshotsLinked paths numLevels
      (snapshotsFrom paths numLevels level tokens layers) := by
  unfold SnapshotsLinked
  intro i current following hcurrent hfollowing
  induction layers generalizing level tokens i current following with
  | nil => simp [snapshotsFrom] at hcurrent
  | cons layer rest ih =>
      cases i with
      | zero =>
          simp only [snapshotsFrom, List.getElem?_cons_zero,
            Option.some.injEq] at hcurrent
          simp only [snapshotsFrom, Nat.zero_add,
            List.getElem?_cons_succ] at hfollowing
          subst current
          cases rest with
          | nil => simp [snapshotsFrom] at hfollowing
          | cons next tail =>
              simp only [snapshotsFrom, List.getElem?_cons_zero,
                Option.some.injEq] at hfollowing
              subst following
              rfl
      | succ i =>
          simp only [snapshotsFrom, List.getElem?_cons_succ] at hcurrent
          simp only [snapshotsFrom, show Nat.succ i + 1 =
            Nat.succ (i + 1) by omega,
            List.getElem?_cons_succ] at hfollowing
          exact ih (i := i) (current := current) (following := following)
            (level := level - 1)
            (tokens := propagate_tokens tokens paths level numLevels
              (evaluate_layer layer tokens).1) hcurrent hfollowing

lemma snapshotsFrom_evaluated {n : Nat} (paths : PathInput)
    (numLevels level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (snapshot : Snapshot n)
    (hmem : snapshot ∈ snapshotsFrom paths numLevels level tokens layers) :
    snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2 := by
  induction layers generalizing level tokens with
  | nil => simp [snapshotsFrom] at hmem
  | cons layer rest ih =>
      simp only [snapshotsFrom, List.mem_cons] at hmem
      cases hmem with
      | inl hhead =>
          subst snapshot
          exact ⟨rfl, rfl⟩
      | inr htail =>
          exact ih (level - 1)
            (propagate_tokens tokens paths level numLevels
              (evaluate_layer layer tokens).1) htail

lemma snapshotsFrom_layers {n : Nat} (paths : PathInput)
    (numLevels level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (snapshot : Snapshot n)
    (hmem : snapshot ∈ snapshotsFrom paths numLevels level tokens layers) :
    snapshot.layer ∈ layers := by
  induction layers generalizing level tokens with
  | nil => simp [snapshotsFrom] at hmem
  | cons layer rest ih =>
      simp only [snapshotsFrom, List.mem_cons] at hmem
      cases hmem with
      | inl hhead => subst snapshot; simp
      | inr htail =>
          exact List.mem_cons_of_mem layer
            (ih (level - 1)
              (propagate_tokens tokens paths level numLevels
                (evaluate_layer layer tokens).1) htail)

lemma snapshotsFrom_getLast_outputs {n : Nat} (paths : PathInput)
    (numLevels level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (accumulated : Bool)
    (hnonempty : layers ≠ []) :
    ∃ final,
      (snapshotsFrom paths numLevels level tokens layers).getLast? = some final ∧
      (eval_from_level paths level tokens layers accumulated numLevels).1 =
        final.outputs := by
  induction layers generalizing level tokens accumulated with
  | nil => contradiction
  | cons layer rest ih =>
      cases rest with
      | nil =>
          refine ⟨{
            level := level
            layer := layer
            tokens := tokens
            outputs := (evaluate_layer layer tokens).1
            error := (evaluate_layer layer tokens).2 }, ?_, ?_⟩
          · simp [snapshotsFrom]
          · simp [eval_from_level]
      | cons next tail =>
          let outputs := (evaluate_layer layer tokens).1
          let nextTokens := propagate_tokens tokens paths level numLevels outputs
          obtain ⟨final, hlast, heval⟩ :=
            ih (level := level - 1) (tokens := nextTokens)
              (accumulated := accumulated || (evaluate_layer layer tokens).2)
              (by simp)
          refine ⟨final, ?_, ?_⟩
          · simpa [snapshotsFrom, outputs, nextTokens] using hlast
          · simpa [eval_from_level, outputs, nextTokens] using heval

lemma tokensAtFinal_go_getLast_tokens {n : Nat} (paths : PathInput)
    (numLevels level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (hnonempty : layers ≠ []) :
    ∃ final,
      (snapshotsFrom paths numLevels level tokens layers).getLast? = some final ∧
      ProofSearch.Grid.Definitions.tokensAtFinal.go paths numLevels level tokens layers =
        final.tokens := by
  induction layers generalizing level tokens with
  | nil => contradiction
  | cons layer rest ih =>
      cases rest with
      | nil =>
          refine ⟨{
            level := level
            layer := layer
            tokens := tokens
            outputs := (evaluate_layer layer tokens).1
            error := (evaluate_layer layer tokens).2 }, ?_, ?_⟩
          · simp [snapshotsFrom]
          · simp [ProofSearch.Grid.Definitions.tokensAtFinal.go]
      | cons next tail =>
          let outputs := (evaluate_layer layer tokens).1
          let nextTokens := propagate_tokens tokens paths level numLevels outputs
          obtain ⟨final, hlast, htokens⟩ :=
            ih (level := level - 1) (tokens := nextTokens) (by simp)
          refine ⟨final, ?_, ?_⟩
          · simpa [snapshotsFrom, outputs, nextTokens] using hlast
          · simpa [ProofSearch.Grid.Definitions.tokensAtFinal.go, outputs, nextTokens]
              using htokens

lemma snapshots_length (fs : List Formula) (transitions : Nat)
    (paths : PathInput) :
    (snapshots fs transitions paths).length = transitions + 1 := by
  simp [snapshots, ProofSearch.Grid.Definitions.gridFor, snapshotsFrom_length]

lemma snapshots_zero (fs : List Formula) (transitions : Nat)
    (paths : PathInput) :
    (snapshots fs transitions paths)[0]? = some {
      level := transitions
      layer := ProofSearch.Grid.Definitions.layerFor fs
      tokens := initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs)
        (transitions + 1)
      outputs := (evaluate_layer (ProofSearch.Grid.Definitions.layerFor fs)
        (initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs)
          (transitions + 1))).1
      error := (evaluate_layer (ProofSearch.Grid.Definitions.layerFor fs)
        (initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs)
          (transitions + 1))).2 } := by
  unfold snapshots ProofSearch.Grid.Definitions.gridFor
  simp only [List.length_replicate]
  rw [show transitions + 1 = Nat.succ transitions by omega,
    List.replicate_succ]
  rfl

lemma snapshots_getLast_tokens (fs : List Formula) (transitions : Nat)
    (paths : PathInput) (final : Snapshot fs.length)
    (hlast : (snapshots fs transitions paths).getLast? = some final) :
    ProofSearch.Grid.Definitions.tokensAtFinal (ProofSearch.Grid.Definitions.gridFor fs transitions)
        (ProofSearch.Grid.Definitions.initialVectorsFor fs) paths = final.tokens := by
  obtain ⟨other, hotherLast, htokens⟩ :=
    tokensAtFinal_go_getLast_tokens paths (transitions + 1) transitions
      (initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs) (transitions + 1))
      (ProofSearch.Grid.Definitions.gridFor fs transitions) (by
        simp [ProofSearch.Grid.Definitions.gridFor])
  have hotherLast' : (snapshots fs transitions paths).getLast? = some other := by
    simpa [snapshots, ProofSearch.Grid.Definitions.gridFor] using hotherLast
  have hsame : other = final := by
    rw [hlast] at hotherLast'
    exact (Option.some.inj hotherLast').symm
  subst other
  simpa [ProofSearch.Grid.Definitions.tokensAtFinal, ProofSearch.Grid.Definitions.gridFor] using htokens

lemma snapshots_getLast_outputs (fs : List Formula) (transitions : Nat)
    (paths : PathInput) (final : Snapshot fs.length)
    (hlast : (snapshots fs transitions paths).getLast? = some final) :
    (get_eval_result (ProofSearch.Grid.Definitions.gridFor fs transitions)
      (ProofSearch.Grid.Definitions.initialVectorsFor fs) paths).1 = final.outputs := by
  obtain ⟨other, hotherLast, houtputs⟩ :=
    snapshotsFrom_getLast_outputs paths (transitions + 1) transitions
      (initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs) (transitions + 1))
      (ProofSearch.Grid.Definitions.gridFor fs transitions) false (by
        simp [ProofSearch.Grid.Definitions.gridFor])
  have hotherLast' : (snapshots fs transitions paths).getLast? = some other := by
    simpa [snapshots, ProofSearch.Grid.Definitions.gridFor] using hotherLast
  have hsame : other = final := by
    rw [hlast] at hotherLast'
    exact (Option.some.inj hotherLast').symm
  subst other
  simpa [get_eval_result, snapshots, ProofSearch.Grid.Definitions.gridFor] using houtputs

lemma eval_from_level_true_of_accumulated_error {n : Nat}
    (paths : PathInput) (level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (numLevels : Nat) :
    (eval_from_level paths level tokens layers true numLevels).2 = true := by
  induction layers generalizing level tokens with
  | nil => rfl
  | cons layer rest ih =>
      cases rest with
      | nil => simp [eval_from_level]
      | cons next tail =>
          change (eval_from_level paths (level - 1)
            (propagate_tokens tokens paths level numLevels
              (evaluate_layer layer tokens).1)
            (next :: tail) true numLevels).2 = true
          exact ih (level - 1)
            (propagate_tokens tokens paths level numLevels
              (evaluate_layer layer tokens).1)

lemma eval_false_implies_snapshots_no_error {n : Nat}
    (paths : PathInput) (numLevels level : Nat) (tokens : List (Token n))
    (layers : List (GridLayer n)) (accumulated : Bool)
    (hfalse :
      (eval_from_level paths level tokens layers accumulated numLevels).2 = false) :
    ∀ snapshot ∈ snapshotsFrom paths numLevels level tokens layers,
      snapshot.error = false := by
  induction layers generalizing level tokens accumulated with
  | nil => simp [snapshotsFrom]
  | cons layer rest ih =>
      cases rest with
      | nil =>
          simp only [eval_from_level] at hfalse
          have hlayer : (evaluate_layer layer tokens).2 = false := by
            cases accumulated <;> cases h : (evaluate_layer layer tokens).2 <;>
              simp_all
          intro snapshot hmem
          simp [snapshotsFrom] at hmem
          subst snapshot
          exact hlayer
      | cons next tail =>
          let evaluated := evaluate_layer layer tokens
          change (eval_from_level paths (level - 1)
            (propagate_tokens tokens paths level numLevels evaluated.1)
            (next :: tail) (accumulated || evaluated.2) numLevels).2 = false
            at hfalse
          have hacc : (accumulated || evaluated.2) = false := by
            cases hvalue : (accumulated || evaluated.2) with
            | false => rfl
            | true =>
                rw [hvalue] at hfalse
                have htrue := eval_from_level_true_of_accumulated_error paths
                  (level - 1)
                  (propagate_tokens tokens paths level numLevels evaluated.1)
                  (next :: tail) numLevels
                rw [htrue] at hfalse
                contradiction
          have hlayer : evaluated.2 = false := by
            have hparts : accumulated = false ∧ evaluated.2 = false := by
              simpa only [Bool.or_eq_false_eq_eq_false_and_eq_false] using hacc
            exact hparts.2
          intro snapshot hmem
          rw [snapshotsFrom] at hmem
          simp only [List.mem_cons] at hmem
          cases hmem with
          | inl hhead => subst snapshot; exact hlayer
          | inr htail =>
              exact ih (level - 1)
                (propagate_tokens tokens paths level numLevels evaluated.1)
                (accumulated || evaluated.2) hfalse snapshot htail

lemma snapshotsLinked_get_succ {n : Nat} {paths : PathInput} {numLevels : Nat}
    {ss : List (Snapshot n)} (linked : SnapshotsLinked paths numLevels ss)
    {i : Nat} {current next : Snapshot n}
    (hcurrent : ss[i]? = some current) (hnext : ss[i + 1]? = some next) :
    next.tokens = propagate_tokens current.tokens paths current.level
      numLevels current.outputs := by
  exact linked i current next hcurrent hnext

def tokensInColumn {n : Nat} (snapshot : Snapshot n) (column : Nat) : List (Token n) :=
  snapshot.tokens.filter fun token => token.current_column = column

lemma evaluate_layer_false_implies_nodeError_false {n : Nat}
    (layer : GridLayer n) (tokens : List (Token n)) (column : Nat)
    (hnode : column < layer.nodes.length)
    (hincoming : column < layer.incoming.length)
    (hnonempty : tokens.filter (fun token => token.current_column = column) ≠ [])
    (hfalse : (evaluate_layer layer tokens).2 = false) :
    nodeError (layer.nodes.get ⟨column, hnode⟩)
      (layer.incoming.get ⟨column, hincoming⟩)
      (tokens.filter fun token => token.current_column = column) = false := by
  let arrivals := tokens.filter fun token => token.current_column = column
  let cnode := layer.nodes.get ⟨column, hnode⟩
  let entry := layer.incoming.get ⟨column, hincoming⟩
  have hzip : (cnode, column) ∈ layer.nodes.zipIdx := by
    rw [List.mem_iff_getElem]
    refine ⟨column, by simpa [cnode] using hnode, ?_⟩
    simp [cnode]
  have hevaluation : evaluate_node cnode entry arrivals ∈
      (layer.nodes.zipIdx.map fun pair =>
        let tokensHere := tokens.filter (·.current_column = pair.2)
        let nodeIncoming := layer.incoming[pair.2]!
        evaluate_node pair.1 nodeIncoming tokensHere) := by
    rw [List.mem_map]
    refine ⟨(cnode, column), hzip, ?_⟩
    simp only
    have hentry : layer.incoming[column]! = entry := by
      simp [List.getElem!_eq_getElem?_getD, entry,
        List.getElem?_eq_getElem hincoming]
    rw [hentry]
  have hresult : (evaluate_node cnode entry arrivals).2 ∈
      ((layer.nodes.zipIdx.map fun pair =>
        let tokensHere := tokens.filter (·.current_column = pair.2)
        let nodeIncoming := layer.incoming[pair.2]!
        evaluate_node pair.1 nodeIncoming tokensHere).map Prod.snd) := by
    exact List.mem_map.mpr ⟨_, hevaluation, rfl⟩
  have hevalFalse : (evaluate_node cnode entry arrivals).2 = false := by
    simp only [evaluate_layer, List.any_eq_false] at hfalse
    have hnot := hfalse _ hresult
    cases hvalue : (evaluate_node cnode entry arrivals).2 with
    | false => rfl
    | true =>
        exfalso
        exact hnot (by simp [hvalue])
  unfold evaluate_node at hevalFalse
  have hisEmpty : arrivals.isEmpty = false := by
    cases h : arrivals with
    | nil => exact False.elim (hnonempty (by simpa [arrivals] using h))
    | cons head tail => rfl
  simp only [hisEmpty, Bool.false_eq_true, ↓reduceIte] at hevalFalse
  cases herror : nodeError cnode entry arrivals with
  | false => simp
  | true => simp [herror] at hevalFalse

lemma evaluate_layer_outputs_length {n : Nat} (layer : GridLayer n)
    (tokens : List (Token n)) :
    (evaluate_layer layer tokens).1.length = layer.nodes.length := by
  simp [evaluate_layer]

lemma selectedRuleIndex?_exists_of_nonempty_noError {n : Nat}
    (node : CircuitNode n) (incoming : NodeIncoming)
    (arrivals : List (Token n)) (nonempty : arrivals ≠ [])
    (noError : nodeError node incoming arrivals = false) :
    ∃ r, selectedRuleIndex? incoming arrivals = some r := by
  obtain ⟨head, tail, rfl⟩ := List.exists_cons_of_ne_nil nonempty
  unfold nodeError at noError
  cases hdecode : decodeInputLabel incoming head.input_label with
  | none => simp [hdecode] at noError
  | some decoded =>
      rcases decoded with ⟨r, slot, source⟩
      exact ⟨r, by simp [selectedRuleIndex?, hdecode]⟩

def tokenForSlot? {n : Nat} (incoming : NodeIncoming) (ruleIndex slot : Nat)
    (tokens : List (Token n)) : Option (Token n) :=
  tokens.find? fun token =>
    match decodeInputLabel incoming token.input_label with
    | some (r, s, _) => r = ruleIndex && s = slot
    | none => false

lemma tokenForSlot?_of_complete {n : Nat} (incoming : NodeIncoming)
    (ruleIndex slot : Nat) (tokens : List (Token n))
    (complete : ∃ token ∈ tokens, ∃ source,
      decodeInputLabel incoming token.input_label =
        some (ruleIndex, slot, source)) :
    ∃ token source,
      tokenForSlot? incoming ruleIndex slot tokens = some token ∧
      token ∈ tokens ∧
      decodeInputLabel incoming token.input_label =
        some (ruleIndex, slot, source) := by
  obtain ⟨witness, hwitness, source, hdecode⟩ := complete
  have hisSome : (tokenForSlot? incoming ruleIndex slot tokens).isSome := by
    unfold tokenForSlot?
    rw [List.find?_isSome]
    refine ⟨witness, hwitness, ?_⟩
    simp [hdecode]
  cases hfind : tokenForSlot? incoming ruleIndex slot tokens with
  | none => simp [hfind] at hisSome
  | some token =>
      have hmem : token ∈ tokens := by
        exact List.mem_of_find?_eq_some hfind
      have htest := List.find?_some hfind
      unfold tokenForSlot? at hfind htest
      cases htokenDecode : decodeInputLabel incoming token.input_label with
      | none => simp [htokenDecode] at htest
      | some decoded =>
          rcases decoded with ⟨r, s, src⟩
          simp only [htokenDecode, Bool.and_eq_true, decide_eq_true_eq] at htest
          rcases htest with ⟨rfl, rfl⟩
          exact ⟨token, src, rfl, hmem, htokenDecode⟩

lemma mem_propagate_tokens_predecessor {n : Nat}
    (tokens : List (Token n)) (paths : PathInput) (level numLevels : Nat)
    (outputs : List (List.Vector Bool n)) (token : Token n)
    (hmem : token ∈ propagate_tokens tokens paths level numLevels outputs) :
    ∃ previous ∈ tokens,
      ∃ hout : previous.current_column < outputs.length,
        token.source_column = previous.current_column ∧
        token.dep_vector = outputs.get ⟨previous.current_column, hout⟩ := by
  rw [propagate_tokens_effStep, List.mem_filterMap] at hmem
  obtain ⟨previous, hprevious, hstep⟩ := hmem
  by_cases hlevel : level > 0
  · rw [if_pos hlevel] at hstep
    cases heffective : effStep paths previous.origin_column
        (numLevels - level - 1) with
    | none => rw [heffective] at hstep; simp at hstep
    | some step =>
        rw [heffective] at hstep
        dsimp only at hstep
        by_cases hout : previous.current_column < outputs.length
        · rw [dif_pos hout] at hstep
          obtain rfl := Option.some.inj hstep
          exact ⟨previous, hprevious, hout, rfl, rfl⟩
        · rw [dif_neg hout] at hstep
          simp at hstep
  · rw [if_neg hlevel] at hstep
    simp at hstep

lemma mem_flattenIncomingAux_slot_source
    (incoming : NodeIncoming) (start : Nat) {r slot source : Nat}
    (hmem : (r, slot, source) ∈ flattenIncomingAux start incoming) :
    ∃ j, ∃ hj : j < incoming.length,
      ∃ hs : slot < (incoming.get ⟨j, hj⟩).length,
      r = start + j ∧
      ((incoming.get ⟨j, hj⟩).get ⟨slot, hs⟩).1 = source := by
  induction incoming generalizing start r slot source with
  | nil => simp [flattenIncomingAux] at hmem
  | cons entry rest ih =>
      cases rest with
      | nil => simp [flattenIncomingAux] at hmem
      | cons next tail =>
          simp only [flattenIncomingAux, List.mem_append] at hmem
          cases hmem with
          | inl hhead =>
              rw [List.mem_map] at hhead
              obtain ⟨⟨⟨src, edge⟩, position⟩, hzip, hequal⟩ := hhead
              simp only at hequal
              rcases hequal with ⟨rfl, rfl, rfl⟩
              obtain ⟨hposition, hpair⟩ := List.mem_zipIdx' hzip
              refine ⟨0, by simp, hposition, rfl, ?_⟩
              simpa only [List.get_cons_zero] using
                (congrArg Prod.fst hpair).symm
          | inr htail =>
              obtain ⟨j, hj, hs, hr, hsource⟩ :=
                ih (start := start + 1) htail
              refine ⟨j + 1, by simpa using Nat.succ_lt_succ hj, ?_, ?_, ?_⟩
              · simpa using hs
              · omega
              · simpa using hsource

lemma decodeInputLabel_slot_source
    (incoming : NodeIncoming) (label r slot source : Nat)
    (entry : RuleIncoming)
    (hdecode : decodeInputLabel incoming label = some (r, slot, source))
    (hentry : incoming[r]? = some entry)
    (hslot : slot < entry.length) :
    (entry.get ⟨slot, hslot⟩).1 = source := by
  cases label with
  | zero =>
      simp only [decodeInputLabel] at hdecode
      by_cases hpositive : 0 < incoming.length
      · rw [if_pos hpositive] at hdecode
        cases hlast : incoming[incoming.length - 1]! with
        | nil =>
            rw [hlast] at hdecode
            contradiction
        | cons pair rest =>
            rw [hlast] at hdecode
            have hequal := Option.some.inj hdecode
            have hr : incoming.length - 1 = r := by
              exact congrArg (fun triple => triple.1) hequal
            have hslotZero : 0 = slot := by
              exact congrArg (fun triple => triple.2.1) hequal
            have hsourcePair : pair.1 = source := by
              exact congrArg (fun triple => triple.2.2) hequal
            subst r
            subst slot
            subst source
            have hlastEntry :
                incoming[incoming.length - 1]! = entry := by
              simp [List.getElem!_eq_getElem?_getD, hentry]
            rw [hlast] at hlastEntry
            subst entry
            rfl
      · rw [if_neg hpositive] at hdecode
        contradiction
  | succ k =>
      unfold decodeInputLabel flattenIncoming at hdecode
      have hmem : (r, slot, source) ∈ flattenIncomingAux 0 incoming :=
        listGet?_mem hdecode
      obtain ⟨j, hj, hs, hr, hsource⟩ :=
        mem_flattenIncomingAux_slot_source incoming 0 hmem
      simp only [Nat.zero_add] at hr
      subst r
      have hentryExact : incoming.get ⟨j, hj⟩ = entry := by
        rw [List.getElem?_eq_getElem hj] at hentry
        exact Option.some.inj hentry
      subst entry
      simpa using hsource

theorem nodeError_false_inversion {n : Nat}
    (node : CircuitNode n) (incoming : NodeIncoming)
    (arrivals : List (Token n)) (r : Nat)
    (nonempty : arrivals ≠ [])
    (noError : nodeError node incoming arrivals = false)
    (selected : selectedRuleIndex? incoming arrivals = some r) :
    r < node.rules.length ∧
    arrivals.length = incoming[r]!.length ∧
    (∀ token ∈ arrivals, ∃ slot source,
      decodeInputLabel incoming token.input_label = some (r, slot, source) ∧
      slot < incoming[r]!.length ∧ token.source_column = source) ∧
    (∀ slot, slot < incoming[r]!.length →
      ∃ token ∈ arrivals, ∃ source,
        decodeInputLabel incoming token.input_label = some (r, slot, source) ∧
        token.source_column = source) := by
  obtain ⟨head, tail, rfl⟩ := List.exists_cons_of_ne_nil nonempty
  unfold selectedRuleIndex? at selected
  unfold nodeError at noError
  cases headDecode : decodeInputLabel incoming head.input_label with
  | none => simp [headDecode] at selected
  | some decoded =>
      rcases decoded with ⟨r0, slot0, source0⟩
      simp only [headDecode, Option.map_some, Option.some.injEq] at selected
      subst r0
      simp only [headDecode, Bool.not_eq_false', Bool.and_eq_true,
        decide_eq_true_eq, List.all_eq_true, List.any_eq_true] at noError
      rcases noError with ⟨⟨⟨ruleExists, labels⟩, count⟩, complete⟩
      refine ⟨ruleExists, count, ?_, ?_⟩
      · intro token tokenMem
        specialize labels token tokenMem
        cases tokenMem with
        | head =>
            simp only [headDecode, decide_eq_true_eq] at labels
            exact ⟨slot0, source0, headDecode, labels.2.1, labels.2.2⟩
        | tail _ tailMem =>
            cases tokenDecode : decodeInputLabel incoming token.input_label with
            | none => simp [tokenDecode] at labels
            | some data =>
                rcases data with ⟨rr, ss, src⟩
                simp only [tokenDecode, decide_eq_true_eq] at labels
                have decodedEqual : some (rr, ss, src) = some (r, ss, src) := by
                  rw [labels.1]
                exact ⟨ss, src, decodedEqual, labels.2.1, labels.2.2⟩
      · intro slot slotLt
        have slotMem : slot ∈ List.range incoming[r]!.length :=
          List.mem_range.mpr slotLt
        obtain ⟨token, tokenMem, tokenCheck⟩ := complete slot slotMem
        cases tokenDecode : decodeInputLabel incoming token.input_label with
        | none => simp [tokenDecode] at tokenCheck
        | some data =>
            rcases data with ⟨rr, ss, src⟩
            simp only [tokenDecode, decide_eq_true_eq] at tokenCheck
            exact ⟨token, tokenMem, src,
              by simpa [tokenCheck.1, tokenCheck.2.1] using tokenDecode,
              tokenCheck.2.2⟩

/-- Rule-specific specialization used by extraction.  The generic dynamic-slot
    inversion is paired with the same-index generator correspondence, exposing
    exactly the premise columns of the selected generated rule. -/
theorem nodeError_false_inversion_forFormula
    (fs : List Formula) (target : Formula) (arrivals : List (Token fs.length))
    (r : Nat) (nonempty : arrivals ≠ [])
    (noError : nodeError (nodeForFormula fs target)
      (buildIncomingMapForFormula fs target) arrivals = false)
    (selected : selectedRuleIndex?
      (buildIncomingMapForFormula fs target) arrivals = some r) :
    ∃ rule entry,
      (nodeForFormula fs target).rules[r]? = some rule ∧
      (buildIncomingMapForFormula fs target)[r]? = some entry ∧
      GeneratedRuleIncomingAligned fs target rule entry ∧
      arrivals.length = entry.length ∧
      (∀ token ∈ arrivals, ∃ slot source,
        decodeInputLabel (buildIncomingMapForFormula fs target)
            token.input_label = some (r, slot, source) ∧
        slot < entry.length ∧ token.source_column = source) ∧
      (∀ slot, slot < entry.length →
        ∃ token ∈ arrivals, ∃ source,
          decodeInputLabel (buildIncomingMapForFormula fs target)
              token.input_label = some (r, slot, source) ∧
          token.source_column = source) := by
  obtain ⟨hrule, hlength, hall, hslots⟩ :=
    nodeError_false_inversion (nodeForFormula fs target)
      (buildIncomingMapForFormula fs target) arrivals r
      nonempty noError selected
  let rule := (nodeForFormula fs target).rules[r]
  have hruleGet : (nodeForFormula fs target).rules[r]? = some rule :=
    List.getElem?_eq_getElem hrule
  obtain ⟨entry, hentryGet, haligned⟩ :=
    nodeRule_incoming_indexed fs target r rule hruleGet
  refine ⟨rule, entry, hruleGet, hentryGet, haligned, ?_, ?_, ?_⟩
  · simpa [List.getElem!_eq_getElem?_getD, hentryGet] using hlength
  · intro token htoken
    obtain ⟨slot, source, hdecode, hslot, hsource⟩ := hall token htoken
    exact ⟨slot, source, hdecode, by
      simpa [List.getElem!_eq_getElem?_getD, hentryGet] using hslot,
      hsource⟩
  · intro slot hslot
    have hslot' : slot <
        (buildIncomingMapForFormula fs target)[r]!.length := by
      simpa [List.getElem!_eq_getElem?_getD, hentryGet] using hslot
    exact hslots slot hslot'

/-- Once `nodeError` has certified a selected rule, the vector returned by the
    evaluator is exactly that rule applied to the incoming entry at the same
    index.  This is the operational half of the indexed generator
    correspondence: it prevents an existentially chosen active rule from
    drifting to another occurrence. -/
lemma evaluate_node_selected_rule {n : Nat}
    (node : CircuitNode n) (incoming : NodeIncoming)
    (arrivals : List (Token n)) (r : Nat) (rule : Rule n)
    (entry : RuleIncoming)
    (hnonempty : arrivals ≠ [])
    (herror : nodeError node incoming arrivals = false)
    (hselected : selectedRuleIndex? incoming arrivals = some r)
    (hrule : node.rules[r]? = some rule)
    (hentry : incoming[r]? = some entry)
    (haligned : incoming.length = node.rules.length) :
    (evaluate_node node incoming arrivals).1 =
      rule.combine (gather_rule_inputs entry
        (arrivals.map fun token => (token.source_column, token.dep_vector))) := by
  have hrBound : r < node.rules.length :=
    List.getElem?_eq_some_iff.mp hrule |>.1
  have hisEmpty : arrivals.isEmpty = false := by
    cases h : arrivals with
    | nil => exact False.elim (hnonempty h)
    | cons head tail => rfl
  let activated := activate_node_from_tokens node (ruleSelectorForIndex r)
  have hrActivated : r < activated.rules.length := by
    simpa [activated, activate_node_from_tokens, activateRulesAux_length] using hrBound
  have hselector :
      selectorActive node.rules.length (ruleSelectorForIndex r) r = true := by
    simp [ruleSelectorForIndex, selectorActive]
  have hrActive : is_rule_active (activated.rules.get ⟨r, hrActivated⟩) = true := by
    exact activate_node_from_tokens_selected_active node
      (ruleSelectorForIndex r) r hrBound hselector
  have hrMem : activated.rules.get ⟨r, hrActivated⟩ ∈ activated.rules :=
    List.get_mem activated.rules ⟨r, hrActivated⟩
  have hany : (extract_activations activated.rules).any (· = true) = true := by
    rw [List.any_eq_true]
    refine ⟨true, ?_, by simp⟩
    simp only [extract_activations]
    exact List.mem_map.mpr ⟨_, hrMem, hrActive⟩
  have hone : exactlyOneActive activated.rules :=
    activate_node_exactlyOne_of_any node (ruleSelectorForIndex r) hany
  have hnodup : activated.rules.Nodup :=
    List.Nodup.of_map (fun selected => selected.ruleId) activated.nodupIds
  have hactivatedLength : activated.rules.length = node.rules.length := by
    simp [activated, activate_node_from_tokens, activateRulesAux_length]
  have hincomingLength : incoming.length = activated.rules.length := by
    rw [haligned, hactivatedLength]
  obtain ⟨activeRule, i, hi, hactiveMem, hactive,
      hactiveGet, hevaluation⟩ :=
    node_logic_with_routing_correct_active activated.rules incoming
      (arrivals.map fun token => (token.source_column, token.dep_vector))
      hone hnodup hincomingLength
  have hsameRule : activeRule = activated.rules.get ⟨r, hrActivated⟩ := by
    obtain ⟨chosen, hchosenMem, hchosenActive, hunique⟩ := hone
    have hactiveEq : activeRule = chosen := hunique activeRule hactiveMem hactive
    have hrEq : activated.rules.get ⟨r, hrActivated⟩ = chosen :=
      hunique _ hrMem hrActive
    exact hactiveEq.trans hrEq.symm
  have hiEq : i = r := by
    have hgetEq : activated.rules.get ⟨i, hi⟩ =
        activated.rules.get ⟨r, hrActivated⟩ := by
      rw [hactiveGet, hsameRule]
    have hfin : (⟨i, hi⟩ : Fin activated.rules.length) = ⟨r, hrActivated⟩ :=
      (List.Nodup.get_inj_iff hnodup).mp hgetEq
    exact congrArg Fin.val hfin
  subst i
  have horiginalRule : node.rules.get ⟨r, hrBound⟩ = rule := by
    rw [List.getElem?_eq_getElem hrBound] at hrule
    exact Option.some.inj hrule
  have hactivatedRule : activeRule = setActivation rule true := by
    rw [hsameRule, activate_node_from_tokens_get node
      (ruleSelectorForIndex r) r hrBound hrActivated, hselector,
      horiginalRule]
  have hcombine : activeRule.combine = rule.combine := by
    rw [hactivatedRule]
    rfl
  unfold evaluate_node
  simp only [hisEmpty, Bool.false_eq_true, ↓reduceIte, herror,
    hselected]
  simpa [activated, hentry, List.getElem!_eq_getElem?_getD, hcombine]
    using congrArg Prod.fst hevaluation

lemma evaluate_layer_output_get {n : Nat} (layer : GridLayer n)
    (tokens : List (Token n)) (column : Nat)
    (hcolumn : column < layer.nodes.length) :
    (evaluate_layer layer tokens).1.get
        ⟨column, by simpa [evaluate_layer] using hcolumn⟩ =
      (evaluate_node (layer.nodes.get ⟨column, hcolumn⟩)
        layer.incoming[column]!
        (tokens.filter fun token => token.current_column = column)).1 := by
  simp [evaluate_layer, List.getElem_zipIdx]

lemma canonical_snapshot_selected_output
    (fs : List Formula) (snapshot : Snapshot fs.length)
    (column : Nat) (target : Formula) (r : Nat)
    (rule : Rule fs.length) (entry : RuleIncoming)
    (hlayer : snapshot.layer = ProofSearch.Grid.Definitions.layerFor fs)
    (hevaluated :
      snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2)
    (hformula : fs[column]? = some target)
    (hnonempty : tokensInColumn snapshot column ≠ [])
    (herror : nodeError (nodeForFormula fs target)
      (buildIncomingMapForFormula fs target)
      (tokensInColumn snapshot column) = false)
    (hselected : selectedRuleIndex? (buildIncomingMapForFormula fs target)
      (tokensInColumn snapshot column) = some r)
    (hrule : (nodeForFormula fs target).rules[r]? = some rule)
    (hentry : (buildIncomingMapForFormula fs target)[r]? = some entry) :
    ∃ hout : column < snapshot.outputs.length,
      snapshot.outputs.get ⟨column, hout⟩ =
        rule.combine (gather_rule_inputs entry
          ((tokensInColumn snapshot column).map fun token =>
            (token.source_column, token.dep_vector))) := by
  have hcolumn : column < fs.length :=
    List.getElem?_eq_some_iff.mp hformula |>.1
  have hnodeColumn : column < snapshot.layer.nodes.length := by
    rw [hlayer]
    simpa [ProofSearch.Grid.Definitions.layerFor] using hcolumn
  have houtputsLength : snapshot.outputs.length = fs.length := by
    calc
      snapshot.outputs.length =
          (evaluate_layer snapshot.layer snapshot.tokens).1.length := by
        rw [hevaluated.1]
      _ = snapshot.layer.nodes.length :=
        evaluate_layer_outputs_length snapshot.layer snapshot.tokens
      _ = fs.length := by rw [hlayer]; simp [ProofSearch.Grid.Definitions.layerFor]
  have hout : column < snapshot.outputs.length := by
    simpa [houtputsLength] using hcolumn
  refine ⟨hout, ?_⟩
  have hnode : snapshot.layer.nodes.get ⟨column, hnodeColumn⟩ =
      nodeForFormula fs target := by
    have hnodeGet : snapshot.layer.nodes[column]? =
        some (nodeForFormula fs target) := by
      rw [hlayer]
      simp [ProofSearch.Grid.Definitions.layerFor, hformula]
    rw [List.getElem?_eq_getElem hnodeColumn] at hnodeGet
    exact Option.some.inj hnodeGet
  have hincoming : snapshot.layer.incoming[column]! =
      buildIncomingMapForFormula fs target := by
    rw [hlayer]
    simp [ProofSearch.Grid.Definitions.layerFor, buildIncomingMap,
      List.getElem!_eq_getElem?_getD, hformula]
  calc
    snapshot.outputs.get ⟨column, hout⟩ =
        (evaluate_layer snapshot.layer snapshot.tokens).1.get
          ⟨column, by simpa [hevaluated.1] using hout⟩ := by
      have hget := congrArg (fun outputs => outputs[column]?) hevaluated.1
      change snapshot.outputs[column]? =
        (evaluate_layer snapshot.layer snapshot.tokens).1[column]? at hget
      rw [List.getElem?_eq_getElem hout,
        List.getElem?_eq_getElem (by simpa [hevaluated.1] using hout)] at hget
      exact Option.some.inj hget
    _ = (evaluate_node (snapshot.layer.nodes.get ⟨column, hnodeColumn⟩)
          snapshot.layer.incoming[column]!
          (tokensInColumn snapshot column)).1 := by
      exact evaluate_layer_output_get snapshot.layer snapshot.tokens column hnodeColumn
    _ = (evaluate_node (nodeForFormula fs target)
          (buildIncomingMapForFormula fs target)
          (tokensInColumn snapshot column)).1 := by rw [hnode, hincoming]
    _ = rule.combine (gather_rule_inputs entry
          ((tokensInColumn snapshot column).map fun token =>
            (token.source_column, token.dep_vector))) :=
      evaluate_node_selected_rule (nodeForFormula fs target)
        (buildIncomingMapForFormula fs target)
        (tokensInColumn snapshot column) r rule entry hnonempty herror
        hselected hrule hentry (incoming_rules_aligned fs target)

lemma arrived_token_has_predecessor {n : Nat} {paths : PathInput}
    {numLevels : Nat} {ss : List (Snapshot n)}
    (linked : SnapshotsLinked paths numLevels ss)
    {depth column : Nat} {previous current : Snapshot n} {token : Token n}
    (hprevious : ss[depth]? = some previous)
    (hcurrent : ss[depth + 1]? = some current)
    (htoken : token ∈ tokensInColumn current column) :
    ∃ predecessor,
      predecessor ∈ tokensInColumn previous token.source_column ∧
      ∃ hout : predecessor.current_column < previous.outputs.length,
        token.dep_vector = previous.outputs.get
          ⟨predecessor.current_column, hout⟩ := by
  have hlink := linked depth previous current hprevious hcurrent
  have htokenCurrent : token ∈ current.tokens := (List.mem_filter.mp htoken).1
  rw [hlink] at htokenCurrent
  obtain ⟨predecessor, hpredecessor, hout, hsource, hdependency⟩ :=
    mem_propagate_tokens_predecessor previous.tokens paths previous.level
      numLevels previous.outputs token htokenCurrent
  refine ⟨predecessor, ?_, hout, hdependency⟩
  unfold tokensInColumn
  rw [List.mem_filter]
  exact ⟨hpredecessor, by simp [hsource]⟩

lemma linked_arrival_dependencies_functional {n : Nat} {paths : PathInput}
    {numLevels : Nat} {ss : List (Snapshot n)}
    (linked : SnapshotsLinked paths numLevels ss)
    {depth : Nat} {previous current : Snapshot n}
    (hprevious : ss[depth]? = some previous)
    (hcurrent : ss[depth + 1]? = some current) :
    ∀ first ∈ current.tokens, ∀ second ∈ current.tokens,
      first.source_column = second.source_column →
      first.dep_vector = second.dep_vector := by
  intro first hfirst second hsecond hsources
  have hlink := linked depth previous current hprevious hcurrent
  rw [hlink] at hfirst hsecond
  obtain ⟨firstPrevious, _hfirstPrevious, hfirstOut,
      hfirstSource, hfirstDep⟩ :=
    mem_propagate_tokens_predecessor previous.tokens paths previous.level
      numLevels previous.outputs first hfirst
  obtain ⟨secondPrevious, _hsecondPrevious, hsecondOut,
      hsecondSource, hsecondDep⟩ :=
    mem_propagate_tokens_predecessor previous.tokens paths previous.level
      numLevels previous.outputs second hsecond
  have hcolumns : firstPrevious.current_column = secondPrevious.current_column := by
    omega
  rw [hfirstDep, hsecondDep]
  exact congrArg (fun index : Fin previous.outputs.length =>
    previous.outputs.get index) (Fin.ext hcolumns)

lemma arrived_token_dependency_eq
    (fs : List Formula) {paths : PathInput} {numLevels : Nat}
    {ss : List (Snapshot fs.length)}
    (linked : SnapshotsLinked paths numLevels ss)
    {depth column : Nat} {previous current : Snapshot fs.length}
    {token : Token fs.length} {deps : List Formula}
    (hprevious : ss[depth]? = some previous)
    (hcurrent : ss[depth + 1]? = some current)
    (htoken : token ∈ tokensInColumn current column)
    (hdeps : ∃ hout : token.source_column < previous.outputs.length,
      depVector fs deps = previous.outputs.get
        ⟨token.source_column, hout⟩) :
    token.dep_vector = depVector fs deps := by
  obtain ⟨predecessor, hpredecessor, hout, htokenDep⟩ :=
    arrived_token_has_predecessor linked hprevious hcurrent htoken
  obtain ⟨hsourceOut, hdeps⟩ := hdeps
  have hsource : predecessor.current_column = token.source_column :=
    of_decide_eq_true (List.mem_filter.mp hpredecessor).2
  have houtputsEqual : previous.outputs.get
        ⟨predecessor.current_column, hout⟩ =
      previous.outputs.get ⟨token.source_column, hsourceOut⟩ :=
    congrArg (fun index : Fin previous.outputs.length =>
      previous.outputs.get index) (Fin.ext hsource)
  exact htokenDep.trans (houtputsEqual.trans hdeps.symm)

lemma find?_mapped_source_of_functional {n : Nat}
    (tokens : List (Token n)) (chosen : Token n)
    (hchosen : chosen ∈ tokens)
    (hfunctional : ∀ token ∈ tokens,
      token.source_column = chosen.source_column →
      token.dep_vector = chosen.dep_vector) :
    (tokens.map fun token => (token.source_column, token.dep_vector)).find?
        (fun pair => pair.1 = chosen.source_column) =
      some (chosen.source_column, chosen.dep_vector) := by
  induction tokens with
  | nil => simp at hchosen
  | cons head tail ih =>
      simp only [List.mem_cons] at hchosen
      by_cases hsource : head.source_column = chosen.source_column
      · have hdep : head.dep_vector = chosen.dep_vector :=
          hfunctional head (by simp) hsource
        simp [hsource, hdep]
      · have htail : chosen ∈ tail := by
          cases hchosen with
          | inl hequal => subst head; contradiction
          | inr hmem => exact hmem
        have hfunctionalTail : ∀ token ∈ tail,
            token.source_column = chosen.source_column →
            token.dep_vector = chosen.dep_vector := by
          intro token htoken
          exact hfunctional token (by simp [htoken])
        simpa [hsource] using ih htail hfunctionalTail

lemma gather_rule_inputs_singleton {n : Nat}
    (available : List (Nat × List.Vector Bool n))
    (source edge : Nat) (dependency : List.Vector Bool n)
    (hfind : available.find? (fun pair => pair.1 = source) =
      some (source, dependency)) :
    gather_rule_inputs [(source, edge)] available = [dependency] := by
  simp [gather_rule_inputs, hfind]

lemma gather_rule_inputs_pair {n : Nat}
    (available : List (Nat × List.Vector Bool n))
    (source₁ edge₁ source₂ edge₂ : Nat)
    (dependency₁ dependency₂ : List.Vector Bool n)
    (hfind₁ : available.find? (fun pair => pair.1 = source₁) =
      some (source₁, dependency₁))
    (hfind₂ : available.find? (fun pair => pair.1 = source₂) =
      some (source₂, dependency₂)) :
    gather_rule_inputs [(source₁, edge₁), (source₂, edge₂)] available =
      [dependency₁, dependency₂] := by
  simp [gather_rule_inputs, hfind₁, hfind₂]

lemma depVector_intro_dependencies
    (fs : List Formula) (dependencies : List Formula) (alpha : Formula) :
    depVector fs ((dependencies.filter fun formula => formula != alpha).eraseDups) =
      (depVector fs dependencies).zipWith (fun b e => b && !e)
        (depVector fs [alpha]) := by
  simpa [List.removeAll] using
    depVector_removeAll_singleton_eq_zipWith_encoder fs dependencies alpha

lemma initialVectorsFor_get_depVector (fs : List Formula) (hfs : fs.Nodup)
    (column : Nat) (hcolumn : column < fs.length) :
    (ProofSearch.Grid.Definitions.initialVectorsFor fs).get
        ⟨column, by simpa [ProofSearch.Grid.Definitions.initialVectorsFor] using hcolumn⟩ =
      depVector fs [fs.get ⟨column, hcolumn⟩] := by
  let initial : List.Vector Bool fs.length :=
    ⟨(List.range fs.length).map (fun bit => decide (bit = column)), by simp⟩
  have hget : (ProofSearch.Grid.Definitions.initialVectorsFor fs).get
        ⟨column, by simpa [ProofSearch.Grid.Definitions.initialVectorsFor] using hcolumn⟩ =
      initial := by
    simp [ProofSearch.Grid.Definitions.initialVectorsFor, initial, List.get_eq_getElem]
  rw [hget]
  apply List.Vector.ext
  intro bit
  rw [depVector_get]
  dsimp [initial, List.Vector.get]
  simp only [List.getElem_map, List.getElem_range, List.mem_singleton]
  by_cases hindex : bit.val = column
  · have hformula : fs[bit.val] = fs[column] := by
      exact congrArg (fun index : Fin fs.length => fs.get index)
        (Fin.ext hindex)
    simp [hindex]
  · have hformula : fs[bit.val] ≠ fs[column] := by
      intro hequal
      have hfin : (bit : Fin fs.length) = ⟨column, hcolumn⟩ :=
        (List.Nodup.get_inj_iff hfs).mp hequal
      exact hindex (congrArg Fin.val hfin)
    simp [hindex, hformula]

lemma initialized_token_in_column_dependency
    (fs : List Formula) (hfs : fs.Nodup) (topLevel column : Nat)
    (target : Formula) (token : Token fs.length)
    (hformula : fs[column]? = some target)
    (htoken : token ∈
      (initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs) topLevel).filter
        (fun candidate => candidate.current_column = column)) :
    token.source_column = column ∧ token.input_label = 0 ∧
      token.dep_vector = depVector fs [target] := by
  have hcolumn : column < fs.length :=
    List.getElem?_eq_some_iff.mp hformula |>.1
  have htarget : fs.get ⟨column, hcolumn⟩ = target :=
    List.getElem?_eq_some_iff.mp hformula |>.2
  obtain ⟨hinit, hcurrent⟩ := List.mem_filter.mp htoken
  unfold initialize_tokens at hinit
  rw [List.mem_map] at hinit
  obtain ⟨⟨vector, index⟩, hzip, rfl⟩ := hinit
  rw [List.mem_iff_getElem] at hzip
  obtain ⟨position, hposition, hget⟩ := hzip
  simp only [List.getElem_zipIdx] at hget
  have hindex : position = index := by
    simpa using congrArg Prod.snd hget
  subst index
  have hvector : vector =
      (ProofSearch.Grid.Definitions.initialVectorsFor fs).get
        ⟨position, by simpa using hposition⟩ := by
    simpa using (congrArg Prod.fst hget).symm
  have hpositionColumn : position = column := by simpa using hcurrent
  subst position
  refine ⟨rfl, rfl, ?_⟩
  rw [hvector, initialVectorsFor_get_depVector fs hfs column hcolumn, htarget]

lemma snapshots_base_output
    (fs : List Formula) (hfs : fs.Nodup) (transitions : Nat)
    (paths : PathInput) (snapshot : Snapshot fs.length)
    (column : Nat) (target : Formula)
    (hsnapshot : (snapshots fs transitions paths)[0]? = some snapshot)
    (hformula : fs[column]? = some target)
    (hnonempty : tokensInColumn snapshot column ≠ [])
    (herror : snapshot.error = false) :
    ∃ hout : column < snapshot.outputs.length,
      depVector fs [target] = snapshot.outputs.get ⟨column, hout⟩ := by
  have hzero := snapshots_zero fs transitions paths
  rw [hzero] at hsnapshot
  have hsnapshotEq := Option.some.inj hsnapshot
  subst snapshot
  let topTokens := initialize_tokens (ProofSearch.Grid.Definitions.initialVectorsFor fs)
    (transitions + 1)
  let layer := ProofSearch.Grid.Definitions.layerFor fs
  let arrivals := topTokens.filter fun token => token.current_column = column
  have hnonempty' : arrivals ≠ [] := by
    simpa [tokensInColumn, arrivals, topTokens] using hnonempty
  have hcolumn : column < fs.length :=
    List.getElem?_eq_some_iff.mp hformula |>.1
  have hnodeBound : column < layer.nodes.length := by
    simpa [layer, ProofSearch.Grid.Definitions.layerFor] using hcolumn
  have hincomingBound : column < layer.incoming.length := by
    simpa [layer, ProofSearch.Grid.Definitions.layerFor, buildIncomingMap] using hcolumn
  have hlocal : nodeError (nodeForFormula fs target)
      (buildIncomingMapForFormula fs target) arrivals = false := by
    have hfalse : (evaluate_layer layer topTokens).2 = false := by
      simpa [layer, topTokens] using herror
    have hraw := evaluate_layer_false_implies_nodeError_false layer topTokens
      column hnodeBound hincomingBound (by simpa [arrivals] using hnonempty')
      hfalse
    have hnode : layer.nodes.get ⟨column, hnodeBound⟩ =
        nodeForFormula fs target := by
      have hget : layer.nodes[column]? = some (nodeForFormula fs target) := by
        simp [layer, ProofSearch.Grid.Definitions.layerFor, hformula]
      rw [List.getElem?_eq_getElem hnodeBound] at hget
      exact Option.some.inj hget
    have hincoming : layer.incoming.get ⟨column, hincomingBound⟩ =
        buildIncomingMapForFormula fs target := by
      have hget : layer.incoming[column]? =
          some (buildIncomingMapForFormula fs target) := by
        simp [layer, ProofSearch.Grid.Definitions.layerFor, buildIncomingMap, hformula]
      rw [List.getElem?_eq_getElem hincomingBound] at hget
      exact Option.some.inj hget
    rw [hnode, hincoming] at hraw
    simpa [arrivals] using hraw
  let incoming := buildIncomingMapForFormula fs target
  let node := nodeForFormula fs target
  let r := incoming.length - 1
  have hrulesLength : incoming.length = node.rules.length := by
    exact incoming_rules_aligned fs target
  obtain ⟨repId, hlastRule⟩ := nodeForFormula_getLast_rule fs target
  have hrule : node.rules[r]? = some (mkRepetitionRule repId false) := by
    have hlast := hlastRule
    rw [List.getLast?_eq_getElem?] at hlast
    simpa [incoming, node, r, hrulesLength] using hlast
  have hentry : incoming[r]? = some [(fs.idxOf target, 0)] := by
    obtain ⟨entry, hentry, haligned⟩ :=
      nodeRule_incoming_indexed fs target r (mkRepetitionRule repId false)
        (by simpa [node, incoming] using hrule)
    unfold GeneratedRuleIncomingAligned at haligned
    obtain ⟨_, haligned⟩ := haligned
    change entry = [(fs.idxOf target, 0)] at haligned
    subst entry
    simpa [incoming] using hentry
  have hpositive : 0 < incoming.length := by
    have := List.getElem?_eq_some_iff.mp hentry |>.1
    omega
  have hdecode : decodeInputLabel incoming 0 =
      some (r, 0, fs.idxOf target) := by
    unfold decodeInputLabel
    rw [if_pos hpositive]
    simp [List.getElem!_eq_getElem?_getD, hentry, r]
  obtain ⟨head, tail, harrivals⟩ := List.exists_cons_of_ne_nil
    hnonempty'
  have hheadMem : head ∈ arrivals := by rw [harrivals]; simp
  have hheadData := initialized_token_in_column_dependency fs hfs
    (transitions + 1) column target head hformula
    (by simpa [arrivals, topTokens] using hheadMem)
  have hselected : selectedRuleIndex? incoming arrivals = some r := by
    rw [harrivals]
    simp [selectedRuleIndex?, hheadData.2.1, hdecode]
  obtain ⟨hout, houtput⟩ := canonical_snapshot_selected_output fs
    { level := transitions
      layer := layer
      tokens := topTokens
      outputs := (evaluate_layer layer topTokens).1
      error := (evaluate_layer layer topTokens).2 }
    column target r (mkRepetitionRule repId false)
    [(fs.idxOf target, 0)] (by simp [layer]) (by simp) hformula
    (by simpa [tokensInColumn, arrivals, topTokens] using hnonempty')
    (by simpa [node, incoming, arrivals] using hlocal)
    (by simpa [node, incoming, arrivals] using hselected)
    (by simpa [node, incoming] using hrule)
    (by simpa [incoming] using hentry)
  have hall : ∀ token ∈ arrivals,
      token.source_column = fs.idxOf target ∧
      token.dep_vector = depVector fs [target] := by
    intro token htoken
    have hdata := initialized_token_in_column_dependency fs hfs
      (transitions + 1) column target token hformula
      (by simpa [arrivals, topTokens] using htoken)
    have htarget : fs.get ⟨column, hcolumn⟩ = target :=
      List.getElem?_eq_some_iff.mp hformula |>.2
    have hidx : fs.idxOf target = column :=
      indexOf_eq_of_get hcolumn hfs htarget
    exact ⟨hdata.1.trans hidx.symm,
      hdata.2.2⟩
  have hfind :
      (arrivals.map fun token => (token.source_column, token.dep_vector)).find?
          (fun pair => pair.1 = fs.idxOf target) =
        some (fs.idxOf target, depVector fs [target]) := by
    have hheadSource : head.source_column = fs.idxOf target := (hall head hheadMem).1
    have hheadDep : head.dep_vector = depVector fs [target] := (hall head hheadMem).2
    have hfunctional : ∀ token ∈ arrivals,
        token.source_column = head.source_column →
        token.dep_vector = head.dep_vector := by
      intro token htoken _
      rw [(hall token htoken).2, hheadDep]
    simpa [hheadSource, hheadDep] using
      find?_mapped_source_of_functional arrivals head hheadMem hfunctional
  have hgather : gather_rule_inputs [(fs.idxOf target, 0)]
      (arrivals.map fun token => (token.source_column, token.dep_vector)) =
      [depVector fs [target]] :=
    gather_rule_inputs_singleton _ _ _ _ hfind
  refine ⟨hout, ?_⟩
  rw [houtput]
  have harrivalsRecord :
      tokensInColumn
        { level := transitions
          layer := layer
          tokens := topTokens
          outputs := (evaluate_layer layer topTokens).1
          error := (evaluate_layer layer topTokens).2 }
        column = arrivals := rfl
  rw [harrivalsRecord, hgather]
  rfl

lemma encoderForIntro_eq_depVector_singleton
    (fs : List Formula) (alpha beta : Formula)
    (encoder : List.Vector Bool fs.length)
    (hencoder : encoderForIntro fs (Formula.implication alpha beta) =
      some encoder) :
    encoder = depVector fs [alpha] := by
  unfold encoderForIntro at hencoder
  have heq := Option.some.inj hencoder
  rw [← heq]
  apply List.Vector.ext
  intro i
  simp [depVector, List.Vector.get]

lemma canonical_snapshot_inversion
    (fs : List Formula) (snapshot : Snapshot fs.length)
    (column : Nat) (target : Formula)
    (hlayer : snapshot.layer = ProofSearch.Grid.Definitions.layerFor fs)
    (hevaluated :
      snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2)
    (herror : snapshot.error = false)
    (hformula : fs[column]? = some target)
    (hnonempty : tokensInColumn snapshot column ≠ []) :
    ∃ r rule entry,
      nodeError (nodeForFormula fs target)
          (buildIncomingMapForFormula fs target)
          (tokensInColumn snapshot column) = false ∧
      selectedRuleIndex? (buildIncomingMapForFormula fs target)
          (tokensInColumn snapshot column) = some r ∧
      (nodeForFormula fs target).rules[r]? = some rule ∧
      (buildIncomingMapForFormula fs target)[r]? = some entry ∧
      GeneratedRuleIncomingAligned fs target rule entry ∧
      (∀ token ∈ tokensInColumn snapshot column, ∃ slot source,
        decodeInputLabel (buildIncomingMapForFormula fs target)
            token.input_label = some (r, slot, source) ∧
        slot < entry.length ∧ token.source_column = source) ∧
      (∀ slot, slot < entry.length →
        ∃ token ∈ tokensInColumn snapshot column, ∃ source,
          decodeInputLabel (buildIncomingMapForFormula fs target)
              token.input_label = some (r, slot, source) ∧
          token.source_column = source) := by
  have hcolumn : column < fs.length := List.getElem?_eq_some_iff.mp hformula |>.1
  have hevalFalse :
      (evaluate_layer snapshot.layer snapshot.tokens).2 = false := by
    rw [← hevaluated.2]
    exact herror
  have hnode : column < snapshot.layer.nodes.length := by
    rw [hlayer]
    simp [ProofSearch.Grid.Definitions.layerFor, hcolumn]
  have hincoming : column < snapshot.layer.incoming.length := by
    rw [hlayer]
    simp [ProofSearch.Grid.Definitions.layerFor, buildIncomingMap, hcolumn]
  have hlocalRaw := evaluate_layer_false_implies_nodeError_false
    snapshot.layer snapshot.tokens column hnode hincoming hnonempty hevalFalse
  have hlocal :
      nodeError (nodeForFormula fs target)
        (buildIncomingMapForFormula fs target)
        (tokensInColumn snapshot column) = false := by
    have hformulaGet : fs[column] = target :=
      (List.getElem?_eq_some_iff.mp hformula).2
    simpa [tokensInColumn, hlayer, ProofSearch.Grid.Definitions.layerFor,
      buildIncomingMap, hformulaGet, List.get_eq_getElem] using hlocalRaw
  obtain ⟨r, hselected⟩ :=
    selectedRuleIndex?_exists_of_nonempty_noError
      (nodeForFormula fs target) (buildIncomingMapForFormula fs target)
      (tokensInColumn snapshot column) hnonempty hlocal
  obtain ⟨rule, entry, hrule, hentry, haligned, _hlength, hall, hslots⟩ :=
    nodeError_false_inversion_forFormula fs target
      (tokensInColumn snapshot column) r hnonempty hlocal hselected
  exact ⟨r, rule, entry, hlocal, hselected, hrule, hentry, haligned, hall, hslots⟩

lemma selected_slot_token {n : Nat} (incoming : NodeIncoming)
    (r : Nat) (entry : RuleIncoming) (arrivals : List (Token n))
    (hentry : incoming[r]? = some entry)
    (hall : ∀ token ∈ arrivals, ∃ slot source,
      decodeInputLabel incoming token.input_label = some (r, slot, source) ∧
      slot < entry.length ∧ token.source_column = source)
    (hslots : ∀ slot, slot < entry.length →
      ∃ token ∈ arrivals, ∃ source,
        decodeInputLabel incoming token.input_label = some (r, slot, source) ∧
        token.source_column = source)
    (slot : Nat) (hslot : slot < entry.length) :
    ∃ token,
      tokenForSlot? incoming r slot arrivals = some token ∧
      token ∈ arrivals ∧
      token.source_column = (entry.get ⟨slot, hslot⟩).1 := by
  obtain ⟨witness, hwitness, source, hwitnessDecode, _⟩ := hslots slot hslot
  obtain ⟨token, decodedSource, hfind, htoken, hdecode⟩ :=
    tokenForSlot?_of_complete incoming r slot arrivals
      ⟨witness, hwitness, source, hwitnessDecode⟩
  obtain ⟨otherSlot, otherSource, hotherDecode, _hotherSlot, hsource⟩ :=
    hall token htoken
  rw [hdecode] at hotherDecode
  have hdecodedEqual := Option.some.inj hotherDecode
  simp only [Prod.mk.injEq] at hdecodedEqual
  rcases hdecodedEqual with ⟨_, hslotEqual, hsourceEqual⟩
  subst otherSlot
  subst otherSource
  have hstatic := decodeInputLabel_slot_source incoming token.input_label
    r slot decodedSource entry hdecode hentry hslot
  exact ⟨token, hfind, htoken, by omega⟩

lemma predecessor_column_ready
    (fs : List Formula) {paths : PathInput} {numLevels : Nat}
    {ss : List (Snapshot fs.length)}
    (linked : SnapshotsLinked paths numLevels ss)
    (hlayers : ∀ snapshot ∈ ss,
      snapshot.layer = ProofSearch.Grid.Definitions.layerFor fs)
    (hevaluated : ∀ snapshot ∈ ss,
      snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2)
    {depth column : Nat} {previous current : Snapshot fs.length}
    {token : Token fs.length}
    (hprevious : ss[depth]? = some previous)
    (hcurrent : ss[depth + 1]? = some current)
    (htoken : token ∈ tokensInColumn current column) :
    tokensInColumn previous token.source_column ≠ [] ∧
      token.source_column < fs.length := by
  have hpreviousBound : depth < ss.length :=
    List.getElem?_eq_some_iff.mp hprevious |>.1
  have hpreviousMem : previous ∈ ss := by
    rw [List.mem_iff_getElem]
    exact ⟨depth, hpreviousBound,
      (List.getElem?_eq_some_iff.mp hprevious).2⟩
  obtain ⟨predecessor, hpredecessor, hout, _hdependency⟩ :=
    arrived_token_has_predecessor linked hprevious hcurrent htoken
  have hpredecessorColumn :
      predecessor.current_column = token.source_column := by
    exact of_decide_eq_true (List.mem_filter.mp hpredecessor).2
  have houtputsLength : previous.outputs.length = fs.length := by
    calc
      previous.outputs.length =
          (evaluate_layer previous.layer previous.tokens).1.length := by
        rw [(hevaluated previous hpreviousMem).1]
      _ = previous.layer.nodes.length :=
        evaluate_layer_outputs_length previous.layer previous.tokens
      _ = fs.length := by
        rw [hlayers previous hpreviousMem]
        simp [ProofSearch.Grid.Definitions.layerFor]
  constructor
  · intro hempty
    rw [hempty] at hpredecessor
    contradiction
  · have hbound : predecessor.current_column < fs.length := by
      simpa [houtputsLength] using hout
    simpa [hpredecessorColumn] using hbound

def extractAt (fs : List Formula) (ss : List (Snapshot fs.length)) :
    Nat -> Nat -> Option NDTree
  | 0, column =>
      match ss[0]?, fs[column]? with
      | some snapshot, some formula =>
          if (tokensInColumn snapshot column).isEmpty then none else some (.hyp formula)
      | _, _ => none
  | depth + 1, column => do
      let snapshot <- ss[depth + 1]?
      let node <- snapshot.layer.nodes[column]?
      let incoming <- snapshot.layer.incoming[column]?
      let arrivals := tokensInColumn snapshot column
      if arrivals.isEmpty || nodeError node incoming arrivals then none
      else
        let ruleIndex <- selectedRuleIndex? incoming arrivals
        let rule <- node.rules[ruleIndex]?
        match rule.type with
        | .repetition =>
            let token <- tokenForSlot? incoming ruleIndex 0 arrivals
            extractAt fs ss depth token.source_column
        | .intro _ =>
            let token <- tokenForSlot? incoming ruleIndex 0 arrivals
            let target <- fs[column]?
            match target with
            | .implication alpha beta =>
                let premise <- extractAt fs ss depth token.source_column
                some (.intro alpha beta premise)
            | _ => none
        | .elim =>
            let majorToken <- tokenForSlot? incoming ruleIndex 0 arrivals
            let minorToken <- tokenForSlot? incoming ruleIndex 1 arrivals
            let alpha <- fs[minorToken.source_column]?
            let beta <- fs[column]?
            let major <- extractAt fs ss depth majorToken.source_column
            let minor <- extractAt fs ss depth minorToken.source_column
            some (.elim alpha beta major minor)

theorem extractAt_structurally_sound
    (fs : List Formula) (paths : PathInput) (numLevels : Nat)
    (ss : List (Snapshot fs.length))
    (linked : SnapshotsLinked paths numLevels ss)
    (hlayers : ∀ snapshot ∈ ss,
      snapshot.layer = ProofSearch.Grid.Definitions.layerFor fs)
    (hevaluated : ∀ snapshot ∈ ss,
      snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2)
    (herrors : ∀ snapshot ∈ ss, snapshot.error = false)
    (hbase : ∀ snapshot column target,
      ss[0]? = some snapshot →
      fs[column]? = some target →
      tokensInColumn snapshot column ≠ [] →
      ∃ hout : column < snapshot.outputs.length,
        depVector fs [target] = snapshot.outputs.get ⟨column, hout⟩) :
    ∀ depth column snapshot target,
      ss[depth]? = some snapshot →
      fs[column]? = some target →
      tokensInColumn snapshot column ≠ [] →
      ∃ tree,
        extractAt fs ss depth column = some tree ∧
        tree.wellFormedB = true ∧
        tree.conclusion = target ∧
        tree.height ≤ depth ∧
        (∀ formula ∈ tree.dependencies, formula ∈ fs) ∧
        ∃ hout : column < snapshot.outputs.length,
          depVector fs tree.dependencies =
            snapshot.outputs.get ⟨column, hout⟩ := by
  intro depth
  induction depth with
  | zero =>
      intro column snapshot target hsnapshot hformula hnonempty
      have hisEmpty : (tokensInColumn snapshot column).isEmpty = false := by
        cases h : tokensInColumn snapshot column with
        | nil => exact False.elim (hnonempty h)
        | cons head tail => rfl
      refine ⟨.hyp target, ?_, rfl, rfl, by simp [ProofSearch.Grid.NDTree.height], ?_, ?_⟩
      simp [extractAt, hsnapshot, hformula, hisEmpty]
      · intro formula hmember
        simp only [ProofSearch.Grid.NDTree.dependencies, List.mem_singleton] at hmember
        subst formula
        rw [List.mem_iff_getElem]
        exact ⟨column, List.getElem?_eq_some_iff.mp hformula |>.1,
          List.getElem?_eq_some_iff.mp hformula |>.2⟩
      simpa [ProofSearch.Grid.NDTree.dependencies] using
        hbase snapshot column target hsnapshot hformula hnonempty
  | succ depth ih =>
      intro column snapshot target hsnapshot hformula hnonempty
      have hsnapshotBound : depth + 1 < ss.length :=
        List.getElem?_eq_some_iff.mp hsnapshot |>.1
      have hsnapshotMem : snapshot ∈ ss := by
        rw [List.mem_iff_getElem]
        exact ⟨depth + 1, hsnapshotBound,
          (List.getElem?_eq_some_iff.mp hsnapshot).2⟩
      let previous := ss[depth]
      have hpreviousBound : depth < ss.length := by omega
      have hprevious : ss[depth]? = some previous :=
        List.getElem?_eq_getElem hpreviousBound
      have hpreviousMem : previous ∈ ss := by
        rw [List.mem_iff_getElem]
        exact ⟨depth, hpreviousBound,
          (List.getElem?_eq_some_iff.mp hprevious).2⟩
      have hlayer := hlayers snapshot hsnapshotMem
      have heval := hevaluated snapshot hsnapshotMem
      have herr := herrors snapshot hsnapshotMem
      obtain ⟨r, rule, entry, hlocal, hselected, hrule, hentry,
          haligned, hall, hslots⟩ :=
        canonical_snapshot_inversion fs snapshot column target
          hlayer heval herr hformula hnonempty
      have hnodeGet : snapshot.layer.nodes[column]? =
          some (nodeForFormula fs target) := by
        rw [hlayer]
        simp [ProofSearch.Grid.Definitions.layerFor, hformula]
      have hincomingGet : snapshot.layer.incoming[column]? =
          some (buildIncomingMapForFormula fs target) := by
        rw [hlayer]
        simp [ProofSearch.Grid.Definitions.layerFor, buildIncomingMap, hformula]
      have hisEmpty : (tokensInColumn snapshot column).isEmpty = false := by
        cases h : tokensInColumn snapshot column with
        | nil => exact False.elim (hnonempty h)
        | cons head tail => rfl
      obtain ⟨hcurrentOut, hcurrentOutput⟩ :=
        canonical_snapshot_selected_output fs snapshot column target r rule entry
          hlayer heval hformula hnonempty hlocal hselected hrule hentry
      have hfunctional := linked_arrival_dependencies_functional linked
        hprevious hsnapshot
      cases hruleType : rule.type with
      | repetition =>
          unfold GeneratedRuleIncomingAligned at haligned
          obtain ⟨hruleWellFormed, haligned⟩ := haligned
          rw [hruleType] at haligned
          subst entry
          have hslot : 0 < ([(fs.idxOf target, 0)] : RuleIncoming).length := by simp
          obtain ⟨token, hfind, htoken, hsource⟩ :=
            selected_slot_token (buildIncomingMapForFormula fs target) r
              [(fs.idxOf target, 0)] (tokensInColumn snapshot column)
              hentry hall hslots 0 hslot
          have hready := predecessor_column_ready fs linked hlayers hevaluated
            hprevious hsnapshot htoken
          have htargetMem : target ∈ fs := by
            rw [List.mem_iff_getElem]
            exact ⟨column, List.getElem?_eq_some_iff.mp hformula |>.1,
              (List.getElem?_eq_some_iff.mp hformula).2⟩
          have hpreviousFormula : fs[token.source_column]? = some target := by
            rw [hsource]
            exact List.getElem?_idxOf htargetMem
          obtain ⟨tree, htree, hwellFormed, hconclusion, hheight,
              hsupport, hdependency⟩ :=
            ih token.source_column previous target hprevious
              hpreviousFormula hready.1
          have htokenDependency : token.dep_vector =
              depVector fs tree.dependencies :=
            arrived_token_dependency_eq fs linked hprevious hsnapshot htoken
              hdependency
          have hfindInput := find?_mapped_source_of_functional
            (tokensInColumn snapshot column) token htoken (by
              intro other hother hsources
              exact hfunctional other (List.mem_filter.mp hother).1 token
                (List.mem_filter.mp htoken).1 hsources)
          have hsource' : token.source_column = fs.idxOf target := by
            simpa using hsource
          have hgather : gather_rule_inputs [(fs.idxOf target, 0)]
              ((tokensInColumn snapshot column).map fun token =>
                (token.source_column, token.dep_vector)) = [token.dep_vector] := by
            apply gather_rule_inputs_singleton
            simpa [hsource'] using hfindInput
          unfold DLDS.Rule.WellFormed at hruleWellFormed
          rw [hruleType] at hruleWellFormed
          refine ⟨tree, ?_, hwellFormed, hconclusion, by omega, hsupport,
            hcurrentOut, ?_⟩
          simp [extractAt, hsnapshot, hnodeGet, hincomingGet, hnonempty, hlocal, hselected, hrule, hruleType, hfind, htree]
          rw [hcurrentOutput, hruleWellFormed]
          simp [hgather, htokenDependency]
      | intro encoder =>
          unfold GeneratedRuleIncomingAligned at haligned
          obtain ⟨hruleWellFormed, haligned⟩ := haligned
          rw [hruleType] at haligned
          obtain ⟨alpha, beta, rfl, hencoder, rfl⟩ := haligned
          have hslot : 0 < ([(fs.idxOf beta, 0)] : RuleIncoming).length := by simp
          obtain ⟨token, hfind, htoken, hsource⟩ :=
            selected_slot_token
              (buildIncomingMapForFormula fs (Formula.implication alpha beta)) r
              [(fs.idxOf beta, 0)] (tokensInColumn snapshot column)
              hentry hall hslots 0 hslot
          have hready := predecessor_column_ready fs linked hlayers hevaluated
            hprevious hsnapshot htoken
          have hsource' : token.source_column = fs.idxOf beta := by
            simpa using hsource
          have hbetaBound : fs.idxOf beta < fs.length := by
            rw [← hsource']
            exact hready.2
          have hbetaMem : beta ∈ fs := List.idxOf_lt_length_iff.mp hbetaBound
          have hpreviousFormula : fs[token.source_column]? = some beta := by
            rw [hsource']
            exact List.getElem?_idxOf hbetaMem
          obtain ⟨premise, hpremise, hpremiseWellFormed,
              hpremiseConclusion, hpremiseHeight, hpremiseSupport,
              hpremiseDependency⟩ :=
            ih token.source_column previous beta hprevious
              hpreviousFormula hready.1
          have htokenDependency : token.dep_vector =
              depVector fs premise.dependencies :=
            arrived_token_dependency_eq fs linked hprevious hsnapshot htoken
              hpremiseDependency
          have hfindInput := find?_mapped_source_of_functional
            (tokensInColumn snapshot column) token htoken (by
              intro other hother hsources
              exact hfunctional other (List.mem_filter.mp hother).1 token
                (List.mem_filter.mp htoken).1 hsources)
          have hgather : gather_rule_inputs [(fs.idxOf beta, 0)]
              ((tokensInColumn snapshot column).map fun token =>
                (token.source_column, token.dep_vector)) = [token.dep_vector] := by
            apply gather_rule_inputs_singleton
            simpa [hsource'] using hfindInput
          have hencoderVector : encoder = depVector fs [alpha] :=
            encoderForIntro_eq_depVector_singleton fs alpha beta encoder hencoder
          unfold DLDS.Rule.WellFormed at hruleWellFormed
          rw [hruleType] at hruleWellFormed
          refine ⟨.intro alpha beta premise, ?_, ?_, rfl, ?_, ?_,
            hcurrentOut, ?_⟩
          · simp [extractAt, hsnapshot, hnodeGet, hincomingGet, hnonempty, hlocal, hselected, hrule, hruleType, hfind, hformula, hpremise]
          · simp [ProofSearch.Grid.NDTree.wellFormedB, hpremiseWellFormed,
              hpremiseConclusion]
          · simp [ProofSearch.Grid.NDTree.height]
            omega
          · intro formula hmember
            exact hpremiseSupport formula
              (List.mem_filter.mp (List.mem_eraseDups.mp hmember) |>.1)
          · simp only [ProofSearch.Grid.NDTree.dependencies]
            rw [depVector_intro_dependencies,
              hcurrentOutput, hruleWellFormed]
            simp [hgather, htokenDependency, hencoderVector]
      | elim =>
          unfold GeneratedRuleIncomingAligned at haligned
          obtain ⟨hruleWellFormed, haligned⟩ := haligned
          rw [hruleType] at haligned
          obtain ⟨alpha, majorColumn, hmajorFormula, rfl⟩ := haligned
          have hmajorSlot :
              0 < ([(majorColumn, 0), (fs.idxOf alpha, 0)] : RuleIncoming).length :=
            by simp
          have hminorSlot :
              1 < ([(majorColumn, 0), (fs.idxOf alpha, 0)] : RuleIncoming).length :=
            by simp
          obtain ⟨majorToken, hmajorFind, hmajorToken, hmajorSource⟩ :=
            selected_slot_token (buildIncomingMapForFormula fs target) r
              [(majorColumn, 0), (fs.idxOf alpha, 0)]
              (tokensInColumn snapshot column) hentry hall hslots 0 hmajorSlot
          obtain ⟨minorToken, hminorFind, hminorToken, hminorSource⟩ :=
            selected_slot_token (buildIncomingMapForFormula fs target) r
              [(majorColumn, 0), (fs.idxOf alpha, 0)]
              (tokensInColumn snapshot column) hentry hall hslots 1 hminorSlot
          have hmajorReady := predecessor_column_ready fs linked hlayers hevaluated
            hprevious hsnapshot hmajorToken
          have hminorReady := predecessor_column_ready fs linked hlayers hevaluated
            hprevious hsnapshot hminorToken
          have hmajorSource' : majorToken.source_column = majorColumn := by
            simpa using hmajorSource
          have hminorSource' : minorToken.source_column = fs.idxOf alpha := by
            simpa using hminorSource
          have hmajorPreviousFormula :
              fs[majorToken.source_column]? =
                some (Formula.implication alpha target) := by
            rw [hmajorSource']
            exact hmajorFormula
          have halphaBound : fs.idxOf alpha < fs.length := by
            rw [← hminorSource']
            exact hminorReady.2
          have halphaMem : alpha ∈ fs := List.idxOf_lt_length_iff.mp halphaBound
          have hminorPreviousFormula :
              fs[minorToken.source_column]? = some alpha := by
            rw [hminorSource']
            exact List.getElem?_idxOf halphaMem
          obtain ⟨major, hmajor, hmajorWellFormed,
              hmajorConclusion, hmajorHeight, hmajorSupport, hmajorDependency⟩ :=
            ih majorToken.source_column previous
              (Formula.implication alpha target) hprevious
              hmajorPreviousFormula hmajorReady.1
          obtain ⟨minor, hminor, hminorWellFormed,
              hminorConclusion, hminorHeight, hminorSupport, hminorDependency⟩ :=
            ih minorToken.source_column previous alpha hprevious
              hminorPreviousFormula hminorReady.1
          have hmajorTokenDependency : majorToken.dep_vector =
              depVector fs major.dependencies :=
            arrived_token_dependency_eq fs linked hprevious hsnapshot
              hmajorToken hmajorDependency
          have hminorTokenDependency : minorToken.dep_vector =
              depVector fs minor.dependencies :=
            arrived_token_dependency_eq fs linked hprevious hsnapshot
              hminorToken hminorDependency
          have hmajorFindInput := find?_mapped_source_of_functional
            (tokensInColumn snapshot column) majorToken hmajorToken (by
              intro other hother hsources
              exact hfunctional other (List.mem_filter.mp hother).1 majorToken
                (List.mem_filter.mp hmajorToken).1 hsources)
          have hminorFindInput := find?_mapped_source_of_functional
            (tokensInColumn snapshot column) minorToken hminorToken (by
              intro other hother hsources
              exact hfunctional other (List.mem_filter.mp hother).1 minorToken
                (List.mem_filter.mp hminorToken).1 hsources)
          have hgather :
              gather_rule_inputs [(majorColumn, 0), (fs.idxOf alpha, 0)]
                ((tokensInColumn snapshot column).map fun token =>
                  (token.source_column, token.dep_vector)) =
                [majorToken.dep_vector, minorToken.dep_vector] := by
            apply gather_rule_inputs_pair
            · simpa [hmajorSource'] using hmajorFindInput
            · simpa [hminorSource'] using hminorFindInput
          unfold DLDS.Rule.WellFormed at hruleWellFormed
          rw [hruleType] at hruleWellFormed
          refine ⟨.elim alpha target major minor, ?_, ?_, rfl, ?_, ?_,
            hcurrentOut, ?_⟩
          · simp [extractAt, hsnapshot, hnodeGet, hincomingGet, hnonempty, hlocal, hselected, hrule, hruleType, hmajorFind, hminorFind, hminorPreviousFormula, hformula, hmajor, hminor]
          · simp [ProofSearch.Grid.NDTree.wellFormedB, hmajorWellFormed, hminorWellFormed,
              hmajorConclusion, hminorConclusion]
          · simp [ProofSearch.Grid.NDTree.height]
            omega
          · intro formula hmember
            have happend := List.mem_eraseDups.mp hmember
            rw [List.mem_append] at happend
            cases happend with
            | inl hmajorMem => exact hmajorSupport formula hmajorMem
            | inr hminorMem => exact hminorSupport formula hminorMem
          · simp only [ProofSearch.Grid.NDTree.dependencies]
            rw [depVector_eraseDups_append_eq_zipWith_or,
              hcurrentOutput, hruleWellFormed]
            simp [hgather, hmajorTokenDependency, hminorTokenDependency]

/- The extraction is the backward causal cone of the produced goal. -/
def extractGoalCone (fs : List Formula) (goal : Formula) (transitions : Nat)
    (paths : PathInput) : Option NDTree :=
  extractAt fs (snapshots fs transitions paths) transitions (fs.idxOf goal)

def gridMarkedB (fs : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Bool :=
  ProofSearch.Grid.Definitions.markedBFor fs goal (ProofSearch.Grid.Definitions.alphabetFor fs) height paths

lemma dependencies_empty_of_depVector_all_false
    (fs dependencies : List Formula)
    (hsupport : ∀ formula ∈ dependencies, formula ∈ fs)
    (hallFalse : (depVector fs dependencies).toList.all (· = false) = true) :
    dependencies = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro formula hmember
  have hformula := hsupport formula hmember
  rw [List.mem_iff_getElem] at hformula
  obtain ⟨index, hindex, hget⟩ := hformula
  let position : Fin fs.length := ⟨index, hindex⟩
  have hbitTrue : (depVector fs dependencies).get position = true := by
    rw [depVector_get]
    simp [position, hget, hmember]
  have hbitMem : (depVector fs dependencies).get position ∈
      (depVector fs dependencies).toList := by
    rw [List.mem_iff_getElem]
    exact ⟨index, by simpa [position] using hindex, rfl⟩
  have hbitFalse := List.all_eq_true.mp hallFalse _ hbitMem
  simp [hbitTrue] at hbitFalse

/-- Primary soundness route: the circuit mark alone supplies the extracted
    certificate.  No recomputation by `extractedCertificateB` occurs here. -/
theorem gridMarked_extracts
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height : Nat) (paths : PathInput)
    (marked : gridMarkedB fs goal height paths = true) :
    ∃ tree, extractGoalCone fs goal height paths = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = goal ∧
      tree.dependencies = [] ∧
      tree.height ≤ height := by
  let layers := ProofSearch.Grid.Definitions.gridFor fs height
  let initial := ProofSearch.Grid.Definitions.initialVectorsFor fs
  let goalColumn := fs.idxOf goal
  let result := get_eval_result layers initial paths
  change (ProofSearch.Grid.Definitions.goalProducedB layers initial paths goalColumn &&
      !result.2 &&
      (match result.1[goalColumn]? with
       | none => false
       | some vector => vector.toList.all (· = false))) = true at marked
  simp only [Bool.and_eq_true] at marked
  obtain ⟨⟨hproduced, hnoErrorNot⟩, hdischarged⟩ := marked
  have hnoError : result.2 = false := by
    cases hvalue : result.2 <;> simp_all
  let ss := snapshots fs height paths
  have hssLength : ss.length = height + 1 := by
    simpa [ss] using snapshots_length fs height paths
  have hfinalBound : height < ss.length := by omega
  let final := ss[height]
  have hfinal : ss[height]? = some final :=
    List.getElem?_eq_getElem hfinalBound
  have hlast : ss.getLast? = some final := by
    rw [List.getLast?_eq_getElem?]
    simpa [hssLength] using hfinal
  have htokens : ProofSearch.Grid.Definitions.tokensAtFinal layers initial paths =
      final.tokens := by
    simpa [layers, initial, ss] using
      snapshots_getLast_tokens fs height paths final hlast
  have houtputs : result.1 = final.outputs := by
    simpa [result, layers, initial, ss] using
      snapshots_getLast_outputs fs height paths final hlast
  have hgoalFormula : fs[goalColumn]? = some goal := by
    simpa [goalColumn] using List.getElem?_idxOf hgoal
  have hlastLayer : layers.getLast? =
      some (ProofSearch.Grid.Definitions.layerFor fs) := by
    rw [List.getLast?_eq_getElem?]
    simp [layers, ProofSearch.Grid.Definitions.gridFor]
  have hnodeGoal : (ProofSearch.Grid.Definitions.layerFor fs).nodes[goalColumn]? =
      some (nodeForFormula fs goal) := by
    simp [ProofSearch.Grid.Definitions.layerFor, hgoalFormula]
  have hincomingGoal : (ProofSearch.Grid.Definitions.layerFor fs).incoming[goalColumn]? =
      some (buildIncomingMapForFormula fs goal) := by
    simp [ProofSearch.Grid.Definitions.layerFor, buildIncomingMap, hgoalFormula]
  unfold ProofSearch.Grid.Definitions.goalProducedB at hproduced
  rw [hlastLayer] at hproduced
  simp only at hproduced
  rw [hnodeGoal, hincomingGoal] at hproduced
  simp only [Bool.and_eq_true] at hproduced
  have hgoalNonempty : tokensInColumn final goalColumn ≠ [] := by
    have hnotEmpty := hproduced.1.1
    rw [htokens] at hnotEmpty
    intro hempty
    have hfilterEmpty :
        final.tokens.filter (·.current_column = goalColumn) = [] := by
      simpa [tokensInColumn] using hempty
    rw [hfilterEmpty] at hnotEmpty
    simp at hnotEmpty
  have hlinked : SnapshotsLinked paths (height + 1) ss := by
    simpa [ss, snapshots, layers, initial, ProofSearch.Grid.Definitions.gridFor] using
      snapshotsFrom_linked paths (height + 1) height
        (initialize_tokens initial (height + 1)) layers
  have hlayers : ∀ snapshot ∈ ss,
      snapshot.layer = ProofSearch.Grid.Definitions.layerFor fs := by
    intro snapshot hmem
    have hmemLayer := snapshotsFrom_layers paths (height + 1) height
      (initialize_tokens initial (height + 1)) layers snapshot
      (by simpa [ss, snapshots, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using hmem)
    simpa [layers, ProofSearch.Grid.Definitions.gridFor] using hmemLayer
  have hevaluated : ∀ snapshot ∈ ss,
      snapshot.outputs = (evaluate_layer snapshot.layer snapshot.tokens).1 ∧
      snapshot.error = (evaluate_layer snapshot.layer snapshot.tokens).2 := by
    intro snapshot hmem
    exact snapshotsFrom_evaluated paths (height + 1) height
      (initialize_tokens initial (height + 1)) layers snapshot
      (by simpa [ss, snapshots, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using hmem)
  have herrors : ∀ snapshot ∈ ss, snapshot.error = false := by
    intro snapshot hmem
    have hresultNoError :
        (eval_from_level paths height
          (initialize_tokens initial (height + 1)) layers false
          (height + 1)).2 = false := by
      simpa [result, get_eval_result, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using hnoError
    exact eval_false_implies_snapshots_no_error paths (height + 1) height
      (initialize_tokens initial (height + 1)) layers false hresultNoError
      snapshot (by simpa [ss, snapshots, layers, initial,
        ProofSearch.Grid.Definitions.gridFor] using hmem)
  have hbase : ∀ snapshot column target,
      ss[0]? = some snapshot →
      fs[column]? = some target →
      tokensInColumn snapshot column ≠ [] →
      ∃ hout : column < snapshot.outputs.length,
        depVector fs [target] = snapshot.outputs.get ⟨column, hout⟩ := by
    intro snapshot column target hsnapshot hformula hnonempty
    apply snapshots_base_output fs hfs height paths snapshot column target
      (by simpa [ss] using hsnapshot) hformula hnonempty
    have hsnapshotBound : 0 < ss.length := by omega
    have hsnapshotMem : snapshot ∈ ss := by
      rw [List.mem_iff_getElem]
      exact ⟨0, hsnapshotBound,
        List.getElem?_eq_some_iff.mp hsnapshot |>.2⟩
    exact herrors snapshot hsnapshotMem
  obtain ⟨tree, htree, hwellFormed, hconclusion, hheight,
      hsupport, hdependency⟩ :=
    extractAt_structurally_sound fs paths (height + 1) ss hlinked
      hlayers hevaluated herrors hbase height goalColumn final goal
      hfinal hgoalFormula hgoalNonempty
  obtain ⟨hgoalOut, hdependency⟩ := hdependency
  have hdischargedFinal :
      (final.outputs.get ⟨goalColumn, hgoalOut⟩).toList.all
        (· = false) = true := by
    rw [houtputs] at hdischarged
    rw [List.getElem?_eq_getElem hgoalOut] at hdischarged
    simpa using hdischarged
  have hdependenciesEmpty : tree.dependencies = [] :=
    dependencies_empty_of_depVector_all_false fs tree.dependencies hsupport
      (by rw [hdependency]; exact hdischargedFinal)
  refine ⟨tree, ?_, hwellFormed, hconclusion, hdependenciesEmpty, hheight⟩
  simpa [extractGoalCone, ss] using htree

theorem gridMarked_sound
    (fs : List Formula) (hfs : fs.Nodup) (goal : Formula)
    (hgoal : goal ∈ fs) (height : Nat) (paths : PathInput)
    (marked : gridMarkedB fs goal height paths = true) :
    Derives [] goal := by
  obtain ⟨tree, _, hwellFormed, hconclusion, hdependencies, _⟩ :=
    gridMarked_extracts fs hfs goal hgoal height paths marked
  exact extracted_certificate_sound hwellFormed hconclusion hdependencies

theorem witness1_gridMarked_extracts :
    ∃ tree,
      extractGoalCone ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          ProofSearch.Grid.Definitions.witness1 = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = ProofSearch.Grid.Definitions.alpha ∧
      tree.dependencies = [] ∧ tree.height ≤ 3 := by
  apply gridMarked_extracts ProofSearch.Grid.Definitions.formulas (by native_decide)
    ProofSearch.Grid.Definitions.alpha (by native_decide) 3 ProofSearch.Grid.Definitions.witness1
  native_decide

theorem witness2_gridMarked_extracts :
    ∃ tree,
      extractGoalCone ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          ProofSearch.Grid.Definitions.witness2 = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = ProofSearch.Grid.Definitions.alpha ∧
      tree.dependencies = [] ∧ tree.height ≤ 3 := by
  apply gridMarked_extracts ProofSearch.Grid.Definitions.formulas (by native_decide)
    ProofSearch.Grid.Definitions.alpha (by native_decide) 3 ProofSearch.Grid.Definitions.witness2
  native_decide

theorem witness3_gridMarked_extracts :
    ∃ tree,
      extractGoalCone ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          ProofSearch.Grid.Definitions.witness3 = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = ProofSearch.Grid.Definitions.alpha ∧
      tree.dependencies = [] ∧ tree.height ≤ 3 := by
  apply gridMarked_extracts ProofSearch.Grid.Definitions.formulas (by native_decide)
    ProofSearch.Grid.Definitions.alpha (by native_decide) 3 ProofSearch.Grid.Definitions.witness3
  native_decide

def extractGoal (fs : List Formula) (goal : Formula) (transitions : Nat)
    (paths : PathInput) : Option NDTree :=
  extractGoalCone fs goal transitions paths

def extractedCertificateB (fs : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Bool :=
  match extractGoalCone fs goal height paths with
  | none => false
  | some tree => decide (
      tree.wellFormedB = true ∧
      tree.conclusion = goal ∧
      tree.dependencies = [] ∧
      tree.height ≤ height)

def certifiedMarkedB (fs : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Bool :=
  gridMarkedB fs goal height paths && extractedCertificateB fs goal height paths

def CertifiedMarked (fs : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Prop :=
  certifiedMarkedB fs goal height paths = true

theorem certifiedMarked_extracts (fs : List Formula) (goal : Formula)
    (height : Nat) (paths : PathInput)
    (marked : CertifiedMarked fs goal height paths) :
    ∃ tree, extractGoalCone fs goal height paths = some tree ∧
      tree.wellFormedB = true ∧
      tree.conclusion = goal ∧
      tree.dependencies = [] ∧
      tree.height ≤ height := by
  unfold CertifiedMarked certifiedMarkedB at marked
  have markedParts :
      gridMarkedB fs goal height paths = true ∧
      extractedCertificateB fs goal height paths = true := by
    simpa only [Bool.and_eq_true] using marked
  have certificateTrue := markedParts.2
  unfold extractedCertificateB at certificateTrue
  cases extraction : extractGoalCone fs goal height paths with
  | none => simp [extraction] at certificateTrue
  | some tree =>
      have facts :
          tree.wellFormedB = true ∧ tree.conclusion = goal ∧
          tree.dependencies = [] ∧ tree.height ≤ height := by
        simpa [extraction] using certificateTrue
      exact ⟨tree, rfl, facts⟩

theorem certifiedMarked_sound (fs : List Formula) (goal : Formula)
    (height : Nat) (paths : PathInput)
    (marked : CertifiedMarked fs goal height paths) :
    Derives [] goal := by
  obtain ⟨tree, _, wellFormed, conclusion, discharged, _⟩ :=
    certifiedMarked_extracts fs goal height paths marked
  exact extracted_certificate_sound wellFormed conclusion discharged

def extractionReport (paths : PathInput) :=
  let proof := extractGoal ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3 paths
  proof.map fun tree =>
    (tree, tree.conclusion, tree.dependencies, tree.height, tree.wellFormedB)

def cellTable (trace : ActiveTrace) :=
  trace.cells.map fun cell =>
    (cell.level, cell.column, cell.formula, cell.arrivals, cell.rule, cell.output)

def wireTable (trace : ActiveTrace) :=
  trace.wires.map fun wire =>
    (wire.sourceLevel, wire.sourceColumn, wire.targetLevel, wire.targetColumn,
      wire.inputLabel, wire.dependency)

def graphReport (name : String) (paths : PathInput) :=
  let trace := activeTrace ProofSearch.Grid.Definitions.formulas 3 paths
  let d := graphOf ProofSearch.Grid.Definitions.formulas 3 paths
  let cone := goalConeTrace ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha trace
  let dc := graphOfTrace ProofSearch.Grid.Definitions.formulas 3 cone
  (name,
   ProofSearch.Grid.Definitions.markedB 3 paths,
   trace.hadError,
   cellTable trace,
   wireTable trace,
   packageReport d,
   cellTable cone,
   wireTable cone,
   packageReport dc)

def validitySummary (paths : PathInput) :=
  let trace := activeTrace ProofSearch.Grid.Definitions.formulas 3 paths
  let full := graphOfTrace ProofSearch.Grid.Definitions.formulas 3 trace
  let cone := graphOfTrace ProofSearch.Grid.Definitions.formulas 3
    (goalConeTrace ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha trace)
  (validDLDSB full, rootDischargedB full,
   validDLDSB cone, rootDischargedB cone)

end ProofSearch.Grid
