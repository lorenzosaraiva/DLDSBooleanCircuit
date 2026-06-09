import Semantic.TreeBridge

open scoped Classical

namespace Semantic

/-!
# Examples and executable witnesses.
-/



/-- Minimal closed proof of `A ⊃ A` (one `⊃I` discharging the hypothesis `A`). -/
def exClosedAA : Graph :=
  let h : Vertex := Vertex.node 1 2 (#"A") true false []          -- hypothesis A (top, level 2)
  let c : Vertex := Vertex.node 2 1 (#"A" >> #"A") false false []  -- conclusion A⊃A (root, level 1)
  Graph.dlds [h, c] [Deduction.edge h c 0 [#"A"]] []

theorem valid_exClosedAA : ValidDLDS exClosedAA := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- check_dlds (manual: implicit-binder ∀ is not auto-decidable)
    refine ⟨?_, ?_, ?_⟩
    ·
      intro N1 N2 h1 h2
      fin_cases h1
      · fin_cases h2
        · native_decide
        · native_decide
      · fin_cases h2
        · native_decide
        · native_decide
    ·
      intro E hE
      fin_cases hE
      native_decide
    · intro P hP; simp [exClosedAA] at hP
  · unfold LeveledColored;     native_decide   -- Leveled-Colored
  · unfold Simplicity;         native_decide   -- Simplicity
  · unfold AncestorSimplicity; native_decide   -- Ancestor-Simplicity
  · unfold HypothesesHaveNoIncoming; native_decide -- hypotheses are leaves
  · unfold LocalRuleCorrect;   native_decide   -- CorrectRuleApp
  · -- closed derivation: the only root is `c`, and it is an intro node
    unfold RootDischarged
    intro r hr hroot
    fin_cases hr
    · have hne : get_rule.outgoing
          (Vertex.node 1 2 (#"A") true false []) exClosedAA ≠ [] := by
        native_decide
      exact False.elim (hne hroot)
    · constructor <;> native_decide
  · unfold ColorAcyclicity; native_decide
  · unfold AncestorEdges; native_decide
  · intro p hp; simp [exClosedAA] at hp
  · intro p₁ hp₁ p₂ hp₂; simp [exClosedAA] at hp₁

/-- Minimal closed proof of `A ⊃ (B ⊃ A)`: inner `⊃I` discharges `B` (vacuously),
    outer `⊃I` discharges the used hypothesis `A`. -/
def exClosedABA : Graph :=
  let hA   : Vertex := Vertex.node 1 3 (#"A") true false []                  -- hypothesis A (top)
  let bca  : Vertex := Vertex.node 2 2 (#"B" >> #"A") false false []          -- B⊃A (⊃I discharging B)
  let root : Vertex := Vertex.node 3 1 (#"A" >> (#"B" >> #"A")) false false [] -- A⊃(B⊃A) (root)
  Graph.dlds [hA, bca, root]
             [Deduction.edge hA bca 0 [#"A"], Deduction.edge bca root 0 [#"A"]] []

theorem valid_exClosedABA : ValidDLDS exClosedABA := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    · intro N1 N2 h1 h2; fin_cases h1 <;> fin_cases h2 <;> native_decide
    · intro E hE; fin_cases hE <;> native_decide
    · intro P hP; simp [exClosedABA] at hP
  · unfold LeveledColored;     native_decide
  · unfold Simplicity;         native_decide
  · unfold AncestorSimplicity; native_decide
  · unfold HypothesesHaveNoIncoming; native_decide
  · unfold LocalRuleCorrect;   native_decide
  · unfold RootDischarged
    intro r hr hroot
    fin_cases hr
    · have hne : get_rule.outgoing
          (Vertex.node 1 3 (#"A") true false []) exClosedABA ≠ [] := by
        native_decide
      exact False.elim (hne hroot)
    · have hne : get_rule.outgoing
          (Vertex.node 2 2 (#"B" >> #"A") false false []) exClosedABA ≠ [] := by
        native_decide
      exact False.elim (hne hroot)
    · constructor <;> native_decide
  · unfold ColorAcyclicity; native_decide
  · unfold AncestorEdges; native_decide
  · intro p hp; simp [exClosedABA] at hp
  · intro p₁ hp₁ p₂ hp₂; simp [exClosedABA] at hp₁

/-- **Genuine acceptance of `A ⊃ A`**: under its extracted path the circuit has no
    routing conflict AND discharges the goal. Derived through `dlds_evaluation_iff`:
    `evaluateDLDS = true` plus `¬ PathStructurallyInvalid` collapses the disjunction
    to `PathHasNoRoutingError ∧ AllAssumptionsDischarged` = `GenuinelyAccepts`. -/
example : GenuinelyAccepts exClosedAA (pathsFromDLDS exClosedAA) (goalColumn exClosedAA) := by
  have h1 : evaluateDLDS exClosedAA (pathsFromDLDS exClosedAA) (goalColumn exClosedAA) = true := by
    native_decide
  have hnoerr : ¬ PathStructurallyInvalid (pathsFromDLDS exClosedAA)
      (buildGridFromDLDS exClosedAA) (initialVectorsFromDLDS exClosedAA) := by
    unfold PathStructurallyInvalid; native_decide
  rcases (dlds_evaluation_iff exClosedAA _ _).mp h1 with hinv | hgood
  · exact absurd hinv hnoerr
  · exact hgood

/-- **Genuine acceptance of `A ⊃ (B ⊃ A)`**, same strong bar. -/
example : GenuinelyAccepts exClosedABA (pathsFromDLDS exClosedABA) (goalColumn exClosedABA) := by
  have h1 : evaluateDLDS exClosedABA (pathsFromDLDS exClosedABA) (goalColumn exClosedABA) = true := by
    native_decide
  have hnoerr : ¬ PathStructurallyInvalid (pathsFromDLDS exClosedABA)
      (buildGridFromDLDS exClosedABA) (initialVectorsFromDLDS exClosedABA) := by
    unfold PathStructurallyInvalid; native_decide
  rcases (dlds_evaluation_iff exClosedABA _ _).mp h1 with hinv | hgood
  · exact absurd hinv hnoerr
  · exact hgood
/-- `exClosedABA` is a simple tree DLDS (each node ≤ 1 outgoing edge; distinct
    formulas per node; no collapse; no ancestral paths). -/
theorem isSimpleTree_exClosedABA : IsSimpleTreeDLDS exClosedABA := by
  refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
  · intro v hv; fin_cases hv <;> native_decide
  · intro v hv; fin_cases hv <;> native_decide
  · rfl
  · intro u hu v hv; fin_cases hu <;> fin_cases hv <;> native_decide

/-- Non-vacuity witness: the forward bridge `tree_bridge_forward` applies to
    `exClosedABA`.
    For the closed proof `A ⊃ (B ⊃ A)`, all four hypotheses are discharged
    (`IsSimpleTreeDLDS`, `ValidDLDS`, and the two decidable certificates), yielding
    genuine circuit acceptance: no routing conflict and the goal discharged. -/
theorem genuinelyAccepts_exClosedABA_via_bridge :
    GenuinelyAccepts exClosedABA (pathsFromDLDS exClosedABA) (goalColumn exClosedABA) :=
  tree_bridge_forward exClosedABA isSimpleTree_exClosedABA valid_exClosedABA
    (by native_decide) (by native_decide)

/-!
# DLDS Circuit Evaluation Tests

This module contains test cases demonstrating the DLDS-to-circuit
translation and evaluation. Each test constructs a natural deduction
proof as a DLDS and verifies it evaluates correctly.

## Test Cases

1. **Test.Identity**: Identity proof (A⊃B)⊃(A⊃B) - valid and invalid paths
2. **Test.Syllogism**: Hypothetical syllogism (A⊃B)⊃(B⊃C)⊃(A⊃C)
3. **Test.Incomplete**: Incomplete proof - undischarged assumptions detected

## Path Encoding

Paths are `List (List (Nat × Nat))` where:
- Outer list: one entry per formula in the universe
- Inner list: one entry per level transition
- First component 0: token stops (inactive)
- First component n > 0: route to column (n-1)
- Second component: destination input label (0 = rep/self; k+1 = flattened non-rep wire)
-/

namespace Test.Identity


-- Formulas
def A : Formula := .atom "A"
def B : Formula := .atom "B"
def A_imp_B : Formula := .implication A B
def identity : Formula := .implication A_imp_B A_imp_B

-- Level 3: Assumptions
def v_A : Vertex :=
  Vertex.node 0 3 A true false []

def v_AimpB_hyp : Vertex :=
  Vertex.node 1 3 A_imp_B true false []

-- Level 2: B derived by modus ponens
def v_B : Vertex :=
  Vertex.node 2 2 B false false []

-- Level 1: A⊃B derived by intro (discharge A)
def v_AimpB : Vertex :=
  Vertex.node 3 1 A_imp_B false false []

-- Level 0: (A⊃B)⊃(A⊃B) derived by intro (discharge A⊃B)
def v_conclusion : Vertex :=
  Vertex.node 4 0 identity false false []

-- Edges
def e_A_to_B : Deduction :=
  Deduction.edge v_A v_B 0 [A]

def e_AimpB_to_B : Deduction :=
  Deduction.edge v_AimpB_hyp v_B 0 [A_imp_B]

def e_B_to_AimpB : Deduction :=
  Deduction.edge v_B v_AimpB 0 [A, B]

def e_AimpB_to_conclusion : Deduction :=
  Deduction.edge v_AimpB v_conclusion 0 [A_imp_B]

-- The DLDS
def dlds : Graph :=
  Graph.dlds
    [v_A, v_AimpB_hyp, v_B, v_AimpB, v_conclusion]
    [e_A_to_B, e_AimpB_to_B, e_B_to_AimpB, e_AimpB_to_conclusion]
    []

/-!
Formula universe (from `buildFormulas`):
- 0: A
- 1: A⊃B
- 2: B
- 3: (A⊃B)⊃(A⊃B)

Goal column: 3
-/

def validPath : Semantic.PathInput := pathsFromDLDS dlds

def invalidPath : Semantic.PathInput :=
  ([ [4, 4, 4],   -- A: wrong - tries to skip intermediate steps
     [4, 0, 0],   -- B: inactive
     [4, 3, 4],   -- A⊃B: partially correct
     [4, 0, 0]    -- conclusion: inactive
   ] : List (List Nat)).map (·.map (fun k => if k = 0 then (0, 0) else (k, 1)))

end Test.Identity


namespace Test.Syllogism


-- Formulas
def A : Formula := .atom "A"
def B : Formula := .atom "B"
def C : Formula := .atom "C"
def A_imp_B : Formula := .implication A B
def B_imp_C : Formula := .implication B C
def A_imp_C : Formula := .implication A C
def inner : Formula := .implication B_imp_C A_imp_C
def conclusion : Formula := .implication A_imp_B inner

-- Level 5: Assumptions
def v_AimpB : Vertex :=
  Vertex.node 0 5 A_imp_B true false []

def v_BimpC : Vertex :=
  Vertex.node 1 4 B_imp_C true false []

def v_A : Vertex :=
  Vertex.node 2 5 A true false []

-- Level 4: B (from A⊃B and A)
def v_B : Vertex :=
  Vertex.node 3 4 B false false []

-- Level 3: C (from B⊃C and B)
def v_C : Vertex :=
  Vertex.node 4 3 C false false []

-- Level 2: A⊃C (intro, discharge A)
def v_AimpC : Vertex :=
  Vertex.node 5 2 A_imp_C false false []

-- Level 1: (B⊃C)⊃(A⊃C) (intro, discharge B⊃C)
def v_inner : Vertex :=
  Vertex.node 6 1 inner false false []

-- Level 0: Conclusion (intro, discharge A⊃B)
def v_conclusion : Vertex :=
  Vertex.node 7 0 conclusion false false []

-- Edges
def e0 : Deduction := Deduction.edge v_AimpB v_B 0 [A_imp_B]
def e1 : Deduction := Deduction.edge v_A v_B 0 [A]
def e2 : Deduction := Deduction.edge v_BimpC v_C 0 [B_imp_C]
def e3 : Deduction := Deduction.edge v_B v_C 0 [A, A_imp_B]
def e4 : Deduction := Deduction.edge v_C v_AimpC 0 [A, A_imp_B, B_imp_C]
def e5 : Deduction := Deduction.edge v_AimpC v_inner 0 [A_imp_B, B_imp_C]
def e6 : Deduction := Deduction.edge v_inner v_conclusion 0 [A_imp_B]

def dlds : Graph :=
  Graph.dlds
    [v_AimpB, v_BimpC, v_A, v_B, v_C, v_AimpC, v_inner, v_conclusion]
    [e0, e1, e2, e3, e4, e5, e6]
    []

theorem valid_dlds : ValidDLDS dlds := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    · intro N1 N2 h1 h2; fin_cases h1 <;> fin_cases h2 <;> native_decide
    · intro E hE; fin_cases hE <;> native_decide
    · intro P hP; simp [dlds] at hP
  · unfold LeveledColored; native_decide
  · unfold Simplicity; native_decide
  · unfold AncestorSimplicity; native_decide
  · unfold HypothesesHaveNoIncoming; native_decide
  · unfold LocalRuleCorrect; native_decide
  · unfold RootDischarged; native_decide
  · unfold ColorAcyclicity; native_decide
  · unfold AncestorEdges; native_decide
  · intro p hp; simp [dlds] at hp
  · intro p₁ hp₁ p₂ hp₂; simp [dlds] at hp₁

/-!
Formula universe will include: A, B, C, A⊃B, B⊃C, A⊃C, (B⊃C)⊃(A⊃C), conclusion
Goal column: 7 (or wherever conclusion lands in eraseDups order)
-/

def validPath : Semantic.PathInput := pathsFromDLDS dlds

-- Expected: ✓ Accepted

end Test.Syllogism


namespace Test.Incomplete


-- Formulas
def A : Formula := .atom "A"
def B : Formula := .atom "B"
def A_imp_B : Formula := .implication A B

-- Level 1: Assumptions (NOT discharged)
def v_A : Vertex :=
  Vertex.node 0 1 A true false []

def v_AimpB : Vertex :=
  Vertex.node 1 1 A_imp_B true false []

-- Level 0: B derived by modus ponens
def v_B : Vertex :=
  Vertex.node 2 0 B false false []

-- Edges
def e_A : Deduction :=
  Deduction.edge v_A v_B 0 [A]

def e_AimpB : Deduction :=
  Deduction.edge v_AimpB v_B 0 [A_imp_B]

def dlds : Graph :=
  Graph.dlds
    [v_A, v_AimpB, v_B]
    [e_A, e_AimpB]
    []

/-!
Formula universe: A, A⊃B, B (3 formulas)
Goal column: 1 (B)

The path routes both assumptions to B, but since there's no
intro rule to discharge them, the final dependency vector for B
will have bits set for A and A⊃B.
-/

def path : Semantic.PathInput := pathsFromDLDS dlds

-- Expected: ✗ Rejected (undischarged assumptions)

end Test.Incomplete


namespace Test.NonRootElim

def A_f   : Formula := .atom "A_nre"
def B_f   : Formula := .atom "B_nre"
def C_f   : Formula := .atom "C_nre"
def D_f   : Formula := .atom "D_nre"
def AiB_f : Formula := .implication A_f B_f
def CiB_f : Formula := .implication C_f B_f
def DiCB_f : Formula := .implication D_f CiB_f

def v_A    : Vertex := Vertex.node 1 3 A_f   true  false []
def v_AiB  : Vertex := Vertex.node 2 3 AiB_f true  false []
def v_B    : Vertex := Vertex.node 3 2 B_f   false false []
def v_CiB  : Vertex := Vertex.node 4 1 CiB_f false false []
def v_root : Vertex := Vertex.node 5 0 DiCB_f false false []

def e0 : Deduction := Deduction.edge v_AiB v_B  0 [AiB_f]   -- major premise
def e1 : Deduction := Deduction.edge v_A   v_B  0 [A_f]     -- minor premise
def e2 : Deduction := Deduction.edge v_B   v_CiB  0 [A_f, AiB_f]
def e3 : Deduction := Deduction.edge v_CiB v_root 0 [A_f, AiB_f]

def dlds : Graph := Graph.dlds [v_A, v_AiB, v_B, v_CiB, v_root] [e0, e1, e2, e3] []

-- had_error under the (now fixed) pathsFromDLDS
def had_error_check : Bool :=
  (get_eval_result (buildGridFromDLDS dlds) (initialVectorsFromDLDS dlds) (pathsFromDLDS dlds)).snd

-- Expected: false (minor-stops fix in routeFrom makes the non-root ⊃E work)

end Test.NonRootElim

namespace Test.Stress

def graphHadError (d : Graph) : Bool :=
  (get_eval_result (buildGridFromDLDS d) (initialVectorsFromDLDS d) (pathsFromDLDS d)).snd

def simpleTreeB (d : Graph) : Bool :=
  d.NODES.all (fun v => decide ((get_rule.outgoing v d).length ≤ 1)) &&
  d.NODES.all (fun v => decide (v.COLLAPSED = false)) &&
  decide (d.PATHS = []) &&
  (d.NODES.all fun u =>
    d.NODES.all fun v =>
      decide (u.FORMULA = v.FORMULA → u = v))

def checkDLDSB (d : Graph) : Bool :=
  (d.NODES.all fun n1 =>
    d.NODES.all fun n2 =>
      decide (n1.NUMBER = n2.NUMBER → n1 = n2)) &&
  (d.EDGES.all fun e =>
    decide (e.START ∈ d.NODES ∧ e.END ∈ d.NODES)) &&
  (d.PATHS.all fun p =>
    decide (p.START ∈ d.NODES ∧ p.END ∈ d.NODES))

def validDLDSB (d : Graph) : Bool :=
  checkDLDSB d &&
  (d.EDGES.all fun e => decide (e.START.LEVEL = e.END.LEVEL + 1)) &&
  (d.EDGES.all fun e1 =>
    d.EDGES.all fun e2 =>
      decide ((e1.START = e2.START ∧ e1.END = e2.END ∧ e1.COLOUR = e2.COLOUR) → e1 = e2)) &&
  (d.PATHS.all fun p1 =>
    d.PATHS.all fun p2 =>
      decide ((p1.START = p2.START ∧ p1.END = p2.END ∧ p1.COLOURS = p2.COLOURS) → p1 = p2)) &&
  (d.NODES.all fun v =>
    if v.HYPOTHESIS then decide (get_rule.incoming v d = []) else true) &&
  (d.NODES.all fun v =>
    if (get_rule.outgoing v d).isEmpty then true
    else ruleShapeOKB v d &&
      ((get_rule.outgoing v d).all fun e => decide (e.DEPENDENCY = outDep v d))) &&
  (d.NODES.all fun r =>
    if (get_rule.outgoing r d).isEmpty then
      ruleShapeOKB r d && decide (outDep r d = [])
    else true) &&
  colorAcyclicityB d &&
  ancestorEdgesB d &&
  ancestorBackwayInformationB d &&
  nonNestedAncestorEdgesB d

def validDLDSConjunctsB (d : Graph) : List (String × Bool) :=
  [ ("check", checkDLDSB d)
  , ("leveled", d.EDGES.all fun e => decide (e.START.LEVEL = e.END.LEVEL + 1))
  , ("simplicity", d.EDGES.all fun e1 =>
      d.EDGES.all fun e2 =>
        decide ((e1.START = e2.START ∧ e1.END = e2.END ∧ e1.COLOUR = e2.COLOUR) → e1 = e2))
  , ("ancestorSimplicity", d.PATHS.all fun p1 =>
      d.PATHS.all fun p2 =>
        decide ((p1.START = p2.START ∧ p1.END = p2.END ∧ p1.COLOURS = p2.COLOURS) → p1 = p2))
  , ("hypNoIncoming", d.NODES.all fun v =>
      if v.HYPOTHESIS then decide (get_rule.incoming v d = []) else true)
  , ("localRuleCorrect", d.NODES.all fun v =>
      if (get_rule.outgoing v d).isEmpty then true
      else ruleShapeOKB v d &&
        ((get_rule.outgoing v d).all fun e => decide (e.DEPENDENCY = outDep v d)))
  , ("rootDischarged", d.NODES.all fun r =>
      if (get_rule.outgoing r d).isEmpty then
        ruleShapeOKB r d && decide (outDep r d = [])
      else true)
  , ("colorAcyclicity", colorAcyclicityB d)
  , ("ancestorEdges", ancestorEdgesB d)
  , ("ancestorBackwayInformation", ancestorBackwayInformationB d)
  , ("nonNestedAncestorEdges", nonNestedAncestorEdgesB d)
  ]

def validDLDSConjunctsString (d : Graph) : String :=
  "[" ++ ", ".intercalate ((validDLDSConjunctsB d).map fun p => s!"{p.1}={p.2}") ++ "]"

def stressLine (name : String) (d : Graph) : String :=
  s!"{name}: simpleTree={simpleTreeB d}, validDLDS={validDLDSB d}, uniqueTokenPerSlot={uniqueTokenPerSlotB d}, had_error={graphHadError d}, evaluateDLDS={evaluateDLDS d (pathsFromDLDS d) (goalColumn d)}"

def slotCountsAtElims (d : Graph) : List (Nat × Formula × Nat × Nat × List Nat) :=
  let formulas := buildFormulas d
  let traces := tokenTraceDLDS d
  List.flatten (traces.zipIdx.map fun (tokens, depth) =>
    d.NODES.filterMap fun w =>
      match classifyRule? w d, ruleIndexForNode? d formulas w with
      | some (DLDSRuleClass.elim _ _), some r =>
          let col := formulas.idxOf w.FORMULA
          let inc := buildIncomingMapForFormula formulas w.FORMULA
          let arity := (inc[r]?.getD default).length
          let here := tokens.filter (fun (t : Token _) => t.current_column = col)
          if here.isEmpty then none
          else
            let counts := (List.range arity).map fun slotIdx =>
              (here.filter fun t =>
                match decodeInputLabel inc t.input_label with
                | some (_, s, _) => s = slotIdx
                | none => false).length
            some (depth, w.FORMULA, arity, here.length, counts)
      | _, _ => none)

def slotCountsString : List (Nat × Formula × Nat × Nat × List Nat) → String
  | [] => "[]"
  | xs =>
      "[" ++ ", ".intercalate (xs.map fun row =>
        s!"(depth={row.1}, formula={row.2.1}, arity={row.2.2.1}, count={row.2.2.2.1}, slots={row.2.2.2.2})") ++ "]"

/- Stress A: three consecutive eliminations B, C, D, then introductions close
   the assumptions A, A->B, B->C, C->D. -/
namespace ThreeElimChain

def A : Formula := .atom "stressA_A"
def B : Formula := .atom "stressA_B"
def C : Formula := .atom "stressA_C"
def D : Formula := .atom "stressA_D"
def AiB : Formula := .implication A B
def BiC : Formula := .implication B C
def CiD : Formula := .implication C D
def AiD : Formula := .implication A D
def AiB_AiD : Formula := .implication AiB AiD
def BiC_AiB_AiD : Formula := .implication BiC AiB_AiD
def Root : Formula := .implication CiD BiC_AiB_AiD

def v_A : Vertex := Vertex.node 1 7 A true false []
def v_AiB : Vertex := Vertex.node 2 7 AiB true false []
def v_BiC : Vertex := Vertex.node 3 6 BiC true false []
def v_CiD : Vertex := Vertex.node 4 5 CiD true false []
def v_B : Vertex := Vertex.node 5 6 B false false []
def v_C : Vertex := Vertex.node 6 5 C false false []
def v_D : Vertex := Vertex.node 7 4 D false false []
def v_AiD : Vertex := Vertex.node 8 3 AiD false false []
def v_AiB_AiD : Vertex := Vertex.node 9 2 AiB_AiD false false []
def v_BiC_AiB_AiD : Vertex := Vertex.node 10 1 BiC_AiB_AiD false false []
def v_root : Vertex := Vertex.node 11 0 Root false false []

def e0 : Deduction := Deduction.edge v_AiB v_B 0 [AiB]
def e1 : Deduction := Deduction.edge v_A v_B 0 [A]
def e2 : Deduction := Deduction.edge v_BiC v_C 0 [BiC]
def e3 : Deduction := Deduction.edge v_B v_C 0 [A, AiB]
def e4 : Deduction := Deduction.edge v_CiD v_D 0 [CiD]
def e5 : Deduction := Deduction.edge v_C v_D 0 [A, AiB, BiC]
def e6 : Deduction := Deduction.edge v_D v_AiD 0 [A, AiB, BiC, CiD]
def e7 : Deduction := Deduction.edge v_AiD v_AiB_AiD 0 [AiB, BiC, CiD]
def e8 : Deduction := Deduction.edge v_AiB_AiD v_BiC_AiB_AiD 0 [BiC, CiD]
def e9 : Deduction := Deduction.edge v_BiC_AiB_AiD v_root 0 [CiD]

def dlds : Graph :=
  Graph.dlds [v_A, v_AiB, v_BiC, v_CiD, v_B, v_C, v_D, v_AiD,
    v_AiB_AiD, v_BiC_AiB_AiD, v_root] [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9] []

end ThreeElimChain

/- Stress B: first elim derives an implication A->C, used as the major of the
   next elimination. -/
namespace ElimConclusionAsMajor

def A : Formula := .atom "stressB_A"
def B : Formula := .atom "stressB_B"
def C : Formula := .atom "stressB_C"
def AiC : Formula := .implication A C
def B_i_AiC : Formula := .implication B AiC
def AiBC : Formula := .implication A (.implication B C)
def B_i_AiBC : Formula := .implication B AiBC
def Root : Formula := .implication B_i_AiC B_i_AiBC

def v_BiAiC : Vertex := Vertex.node 1 5 B_i_AiC true false []
def v_B : Vertex := Vertex.node 2 5 B true false []
def v_A : Vertex := Vertex.node 3 4 A true false []
def v_AiC : Vertex := Vertex.node 4 4 AiC false false []
def v_C : Vertex := Vertex.node 5 3 C false false []
def v_BC : Vertex := Vertex.node 6 2 (.implication B C) false false []
def v_AiBC : Vertex := Vertex.node 7 1 AiBC false false []
def v_root : Vertex := Vertex.node 8 0 Root false false []

def e0 : Deduction := Deduction.edge v_BiAiC v_AiC 0 [B_i_AiC]
def e1 : Deduction := Deduction.edge v_B v_AiC 0 [B]
def e2 : Deduction := Deduction.edge v_AiC v_C 0 [B, B_i_AiC]
def e3 : Deduction := Deduction.edge v_A v_C 0 [A]
def e4 : Deduction := Deduction.edge v_C v_BC 0 [A, B, B_i_AiC]
def e5 : Deduction := Deduction.edge v_BC v_AiBC 0 [A, B_i_AiC]
def e6 : Deduction := Deduction.edge v_AiBC v_root 0 [B_i_AiC]

def dlds : Graph :=
  Graph.dlds [v_BiAiC, v_B, v_A, v_AiC, v_C, v_BC, v_AiBC, v_root]
    [e0, e1, e2, e3, e4, e5, e6] []

end ElimConclusionAsMajor

/- Stress B2: first elim derives C->D, and that derived implication is used as
   the major premise of the next elimination. `ElimConclusionAsMajor` above is
   an invalid negative example. -/
namespace ElimConclusionAsMajorValid

def A : Formula := .atom "stressB2_A"
def C : Formula := .atom "stressB2_C"
def D : Formula := .atom "stressB2_D"
def E : Formula := .atom "stressB2_E"
def CiD : Formula := .implication C D
def AiCiD : Formula := .implication A CiD
def EiD : Formula := .implication E D
def CiEiD : Formula := .implication C EiD
def AiCiEiD : Formula := .implication A CiEiD
def Root : Formula := .implication AiCiD AiCiEiD

def v_AiCiD : Vertex := Vertex.node 1 6 AiCiD true false []
def v_A : Vertex := Vertex.node 2 6 A true false []
def v_C : Vertex := Vertex.node 3 5 C true false []
def v_CiD : Vertex := Vertex.node 4 5 CiD false false []
def v_D : Vertex := Vertex.node 5 4 D false false []
def v_EiD : Vertex := Vertex.node 6 3 EiD false false []
def v_CiEiD : Vertex := Vertex.node 7 2 CiEiD false false []
def v_AiCiEiD : Vertex := Vertex.node 8 1 AiCiEiD false false []
def v_root : Vertex := Vertex.node 9 0 Root false false []

def e0 : Deduction := Deduction.edge v_AiCiD v_CiD 0 [AiCiD]
def e1 : Deduction := Deduction.edge v_A v_CiD 0 [A]
def e2 : Deduction := Deduction.edge v_CiD v_D 0 [A, AiCiD]
def e3 : Deduction := Deduction.edge v_C v_D 0 [C]
def e4 : Deduction := Deduction.edge v_D v_EiD 0 [C, A, AiCiD]
def e5 : Deduction := Deduction.edge v_EiD v_CiEiD 0 [C, A, AiCiD]
def e6 : Deduction := Deduction.edge v_CiEiD v_AiCiEiD 0 [A, AiCiD]
def e7 : Deduction := Deduction.edge v_AiCiEiD v_root 0 [AiCiD]

def dlds : Graph :=
  Graph.dlds [v_AiCiD, v_A, v_C, v_CiD, v_D, v_EiD, v_CiEiD, v_AiCiEiD, v_root]
    [e0, e1, e2, e3, e4, e5, e6, e7] []

end ElimConclusionAsMajorValid

/- Stress C: elim, then intro, then elim. The introduced implication is used
   as the minor of the next elim; using it as major would duplicate the
   consequent formula and violate InjFormulas in the simple-tree fragment. -/
namespace ElimIntroElim

def A : Formula := .atom "stressC_A"
def P : Formula := .atom "stressC_P"
def Y : Formula := .atom "stressC_Y"
def Z : Formula := .atom "stressC_Z"
def AiY : Formula := .implication A Y
def PiY : Formula := .implication P Y
def K : Formula := .implication PiY Z
def AiZ : Formula := .implication A Z
def AiY_AiZ : Formula := .implication AiY AiZ
def Root : Formula := .implication K AiY_AiZ

def v_AiY : Vertex := Vertex.node 1 6 AiY true false []
def v_A : Vertex := Vertex.node 2 6 A true false []
def v_Y : Vertex := Vertex.node 3 5 Y false false []
def v_PiY : Vertex := Vertex.node 4 4 PiY false false []
def v_K : Vertex := Vertex.node 5 4 K true false []
def v_Z : Vertex := Vertex.node 6 3 Z false false []
def v_AiZ : Vertex := Vertex.node 7 2 AiZ false false []
def v_AiY_AiZ : Vertex := Vertex.node 8 1 AiY_AiZ false false []
def v_root : Vertex := Vertex.node 9 0 Root false false []

def e0 : Deduction := Deduction.edge v_AiY v_Y 0 [AiY]
def e1 : Deduction := Deduction.edge v_A v_Y 0 [A]
def e2 : Deduction := Deduction.edge v_Y v_PiY 0 [A, AiY]
def e3 : Deduction := Deduction.edge v_K v_Z 0 [K]
def e4 : Deduction := Deduction.edge v_PiY v_Z 0 [A, AiY]
def e5 : Deduction := Deduction.edge v_Z v_AiZ 0 [A, AiY, K]
def e6 : Deduction := Deduction.edge v_AiZ v_AiY_AiZ 0 [AiY, K]
def e7 : Deduction := Deduction.edge v_AiY_AiZ v_root 0 [K]

def dlds : Graph :=
  Graph.dlds [v_AiY, v_A, v_Y, v_PiY, v_K, v_Z, v_AiZ, v_AiY_AiZ, v_root]
    [e0, e1, e2, e3, e4, e5, e6, e7] []

end ElimIntroElim


end Test.Stress

namespace Test.Counterexample

/- Former counterexample to the universal structural theorem, retained as a
   regression test for the strengthened validity predicate.

   Node `v_A` is marked as a hypothesis but also has an incoming edge from `v_B`.
   `classifyRule?` treats every `HYPOTHESIS=true` node as a hypothesis and ignores
   incoming edges, while the grid still routes carriers along those edges.  The
   `HypothesesHaveNoIncoming` conjunct now excludes this malformed structure. -/
namespace HypWithIncoming

def A : Formula := .atom "ce_A"
def B : Formula := .atom "ce_B"
def AimpA : Formula := .implication A A

def v_B : Vertex := Vertex.node 1 2 B true false []
def v_A : Vertex := Vertex.node 2 1 A true false []
def v_root : Vertex := Vertex.node 3 0 AimpA false false []

def e_B_to_A : Deduction := Deduction.edge v_B v_A 0 [B]
def e_A_to_root : Deduction := Deduction.edge v_A v_root 0 [A]

def dlds : Graph := Graph.dlds [v_B, v_A, v_root] [e_B_to_A, e_A_to_root] []

def had_error : Bool :=
  (get_eval_result (buildGridFromDLDS dlds) (initialVectorsFromDLDS dlds)
    (pathsFromDLDS dlds)).snd


end HypWithIncoming

end Test.Counterexample

namespace Test.ExConflict

/-- The old availability-driven activation treated the two premises entering `B`
    as two active rules. With DLDS-derived one-hot selector, both tokens name the
    single ⊃E rule at `B`, so this must not be a structural conflict. -/
def dlds : Graph := Test.Incomplete.dlds
def paths : Semantic.PathInput := pathsFromDLDS dlds

def had_error : Bool :=
  (get_eval_result (buildGridFromDLDS dlds) (initialVectorsFromDLDS dlds) paths).snd


end Test.ExConflict

namespace Test.Summary
/-!
## Test Summary

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Identity (valid) | Correct derivation of (A⊃B)⊃(A⊃B) | ✓ Accepted |
| Identity (invalid) | Wrong routing path | ✓ Accepted (structural error) |
| Syllogism | Hypothetical syllogism | ✓ Accepted |
| Incomplete | Undischarged assumptions | ✗ Rejected |

## Interpretation

- **Accepted (true)**: Either the path has a structural error (detected by XOR
  check), or it has no routing error and all assumptions are discharged.

- **Rejected (false)**: The path is structurally valid but the proof has
  undischarged assumptions (dependency vector is non-zero at goal).

This matches the main theorem `circuit_correctness`:
```
evaluateCircuit = true →
  PathStructurallyInvalid ∨ (PathHasNoRoutingError ∧ AllAssumptionsDischarged)
```
-/
end Test.Summary


end Semantic
