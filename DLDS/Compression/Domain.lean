import DLDS.Compression.FixedPoints
import ProofSearch.BranchingGrid.Completeness

open DLDS Flow
open ProofSearch.Grid
open DLDS.ValidityDecomposition
open Certificate.Flow.GeneratorCompleteness ProofSearch.FlowCorrespondence

namespace DLDS.Compression.Domain

/-!
# The intended graph domain of horizontal compression

`IsTreeDLDS` only constrains local fan-out, colours, collapse metadata, and
ancestral paths.  `ProperHCInput` additionally chooses the unique terminal root
and requires every graph node to lie in its backwards deduction cone.
-/

def RootAt (d : Graph) (root : Vertex) : Prop :=
  root ∈ d.NODES ∧ get_rule.outgoing root d = []

def UniqueRootAt (d : Graph) (root : Vertex) : Prop :=
  RootAt d root ∧
    ∀ candidate ∈ d.NODES,
      get_rule.outgoing candidate d = [] → candidate = root

def EveryNodeReachesRoot (d : Graph) (root : Vertex) : Prop :=
  ∀ vertex ∈ d.NODES, vertex = root ∨ vertex ∈ Pre d root

def ProperHCInput (d : Graph) : Prop :=
  IsTreeDLDS d ∧ ValidDLDS d ∧
    ∃ root, UniqueRootAt d root ∧ EveryNodeReachesRoot d root

set_option maxHeartbeats 0 in
/-- Soundness of the repository's finite declarative-validity checker.  This
is kept here because it lets concrete HC inputs discharge the finite validity
package with one computation rather than one native proof per list element. -/
theorem validDLDSB_sound_for_hc {d : Graph}
    (checked : ProofSearch.Grid.validDLDSB d = true) :
    ValidDLDS d := by
  simp only [ProofSearch.Grid.validDLDSB, Bool.and_eq_true,
    List.all_eq_true, decide_eq_true_eq, colorAcyclicityB,
    ancestorEdgesB, ancestorBackwayInformationB, relativeAddressB,
    beq_iff_eq] at checked
  rcases checked with ⟨checked, nonNested⟩
  rcases checked with ⟨checked, backway⟩
  rcases checked with ⟨checked, ancestorEdges⟩
  rcases checked with ⟨checked, colorAcyclic⟩
  rcases checked with ⟨checked, rootDischarged⟩
  rcases checked with ⟨checked, localRule⟩
  rcases checked with ⟨checked, hypotheses⟩
  rcases checked with ⟨checked, ancestorSimple⟩
  rcases checked with ⟨checked, simple⟩
  rcases checked with ⟨checked, leveled⟩
  rcases checked with ⟨checked, pathsAttached⟩
  rcases checked with ⟨numbersUnique, edgesAttached⟩
  refine ⟨?_, leveled, simple, ancestorSimple, ?_, ?_, ?_,
    colorAcyclic, ancestorEdges, backway, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    intro left right leftMember rightMember
    constructor
    · exact numbersUnique left leftMember right rightMember
    · intro equal
      subst right
      rfl
    · intro edge edgeMember
      exact edgesAttached edge edgeMember
    · intro path pathMember
      exact pathsAttached path pathMember
  · intro vertex vertexMember hypothesis
    have checkedVertex := hypotheses vertex vertexMember
    simp [hypothesis] at checkedVertex
    exact checkedVertex
  · intro vertex vertexMember outgoing
    have checkedVertex := localRule vertex vertexMember
    simp [List.isEmpty_iff, outgoing] at checkedVertex
    exact checkedVertex
  · unfold ProofSearch.Grid.rootDischargedB at rootDischarged
    rw [List.all_eq_true] at rootDischarged
    intro root rootMember terminal
    have checkedRoot := rootDischarged root rootMember
    simp [terminal] at checkedRoot
    exact checkedRoot
  · intro left leftMember right rightMember
    unfold nonNestedAncestorEdgesB at nonNested
    rw [List.all_eq_true] at nonNested
    have atLeft := nonNested left leftMember
    rw [List.all_eq_true] at atLeft
    have separated := atLeft right rightMember
    rw [Bool.and_eq_true] at separated
    constructor
    · intro nested
      have impossible := separated.1
      simp [nested] at impossible
    · intro nested
      have impossible := separated.2
      simp [nested] at impossible

/-- A compression class mixes a logical source occurrence with a derived
occurrence.  The executable compressor's historical merge key identifies such
nodes because it contains only level and formula. -/
def MixedSourceHCClass (d : Graph) : Prop :=
  ∃ source ∈ d.NODES, ∃ derived ∈ d.NODES,
    source ≠ derived ∧
      source.LEVEL = derived.LEVEL ∧
      source.FORMULA = derived.FORMULA ∧
      source.HYPOTHESIS ≠ derived.HYPOTHESIS

def mixedSourceHCClassB (d : Graph) : Bool :=
  d.NODES.any fun source =>
    d.NODES.any fun derived =>
      source != derived &&
        source.LEVEL == derived.LEVEL &&
        source.FORMULA == derived.FORMULA &&
        source.HYPOTHESIS != derived.HYPOTHESIS

theorem mixedSourceHCClassB_sound {d : Graph}
    (mixed : mixedSourceHCClassB d = true) :
    MixedSourceHCClass d := by
  simpa [mixedSourceHCClassB, MixedSourceHCClass, and_assoc] using mixed

theorem hcSourceClassKeyEqB_hypothesis_eq {left right : Vertex}
    (sameClass : hcSourceClassKeyEqB left right = true) :
    left.HYPOTHESIS = right.HYPOTHESIS := by
  have fields :
      left.LEVEL = right.LEVEL ∧
        left.FORMULA = right.FORMULA ∧
        left.HYPOTHESIS = right.HYPOTHESIS := by
    simpa [hcSourceClassKeyEqB, hcClassKeyEqB, Bool.and_assoc] using sameClass
  exact fields.2.2

theorem not_hcSourceClassKeyEqB_of_hypothesis_ne {left right : Vertex}
    (different : left.HYPOTHESIS ≠ right.HYPOTHESIS) :
    hcSourceClassKeyEqB left right = false := by
  cases result : hcSourceClassKeyEqB left right with
  | false => rfl
  | true => exact False.elim (different
      (hcSourceClassKeyEqB_hypothesis_eq result))

theorem source_status_split_has_at_most_two_classes (members : List Vertex) :
    ([members.filter fun node => node.HYPOTHESIS,
      members.filter fun node => !node.HYPOTHESIS].filter
        fun subclass => 1 < subclass.length).length ≤ 2 := by
  simp only [List.filter_cons, List.filter_nil]
  split <;> split <;> simp

theorem compress_nodes_graph_sourceSeparated_eq_self_of_no_candidates
    (d : Graph)
    (noCandidates : repeated_nodes_vertex_sourceSeparated d = []) :
    compress_nodes_graph_sourceSeparated d = d := by
  unfold compress_nodes_graph_sourceSeparated
  rw [noCandidates]
  rfl

/-! ## A small verified `NDTree -> Graph` front end -/

structure EncodedNDSubtree where
  root : Vertex
  nodes : List Vertex
  edges : List Deduction
  next : Nat

def encodeNDTreeAt : NDTree → Nat → Nat → EncodedNDSubtree
  | .hyp formula, level, next =>
      let root := node next level formula true false []
      { root, nodes := [root], edges := [], next := next + 1 }
  | .intro alpha beta premise, level, next =>
      let root := node next level (alpha >> beta) false false []
      let encodedPremise := encodeNDTreeAt premise (level + 1) (next + 1)
      { root
        nodes := root :: encodedPremise.nodes
        edges := edge encodedPremise.root root 0
          (normalizeDep premise.dependencies) :: encodedPremise.edges
        next := encodedPremise.next }
  | .elim _ beta major minor, level, next =>
      let root := node next level beta false false []
      let encodedMajor := encodeNDTreeAt major (level + 1) (next + 1)
      let encodedMinor := encodeNDTreeAt minor (level + 1) encodedMajor.next
      { root
        nodes := root :: (encodedMajor.nodes ++ encodedMinor.nodes)
        edges :=
          edge encodedMajor.root root 0 (normalizeDep major.dependencies) ::
          edge encodedMinor.root root 0 (normalizeDep minor.dependencies) ::
          (encodedMajor.edges ++ encodedMinor.edges)
        next := encodedMinor.next }

def ndTreeToGraph (tree : NDTree) : Graph :=
  let encoded := encodeNDTreeAt tree 0 1
  dlds encoded.nodes encoded.edges []

@[simp] theorem ndTreeToGraph_paths (tree : NDTree) :
    (ndTreeToGraph tree).PATHS = [] := by
  rfl

/-- Standard detour-freeness check: an elimination's major premise must not
be the immediate conclusion of an implication introduction. -/
def normalNDTreeB : NDTree → Bool
  | .hyp _ => true
  | .intro _ _ premise => normalNDTreeB premise
  | .elim _ _ major minor =>
      (match major with
       | .intro _ _ _ => false
       | _ => true) && normalNDTreeB major && normalNDTreeB minor

/-! ## A connected, unique-root, closed, normal proof with a mixed class -/

namespace GenuineMixedSource

def A : Formula := #"proper_A"
def B : Formula := #"proper_B"
def Y : Formula := #"proper_Y"
def X : Formula := A >> A
def BA : Formula := B >> A
def H : Formula := A >> (X >> Y)

/--
The central normal proof is

* `A ⊢ A ⊃ A` by introduction;
* `B ⊃ A, B ⊢ A`, then `A ⊃ ((A ⊃ A) ⊃ Y) ⊢ (A ⊃ A) ⊃ Y`;
* eliminate to obtain `Y`, then greedily discharge `B`, `B ⊃ A`, and `H`.

The hypothesis `A` and the independently derived `A` occur at the same depth.
-/
def tree : NDTree :=
  let directX := ProofSearch.Grid.NDTree.intro A A (.hyp A)
  let derivedA := ProofSearch.Grid.NDTree.elim B A (.hyp BA) (.hyp B)
  let majorXY := ProofSearch.Grid.NDTree.elim A (X >> Y) (.hyp H) derivedA
  let coreY := ProofSearch.Grid.NDTree.elim X Y majorXY directX
  ProofSearch.Grid.NDTree.intro H (BA >> (B >> Y))
    (ProofSearch.Grid.NDTree.intro BA (B >> Y)
      (ProofSearch.Grid.NDTree.intro B Y coreY))

def input : Graph := ndTreeToGraph tree
def output : Graph := compress_nodes_graph input
def sourceSeparatedOutput : Graph :=
  compress_nodes_graph_sourceSeparated input

theorem tree_wellFormed : tree.wellFormedB = true := by
  native_decide

theorem tree_closed : tree.dependencies = [] := by
  native_decide

theorem tree_normal : normalNDTreeB tree = true := by
  native_decide

theorem tree_conclusion :
    tree.conclusion = H >> (BA >> (B >> Y)) := by
  native_decide

theorem input_isTreeDLDS : IsTreeDLDS input := by
  refine ⟨?_, ?_, rfl, ?_⟩
  · intro vertex vertexMember
    fin_cases vertexMember <;> native_decide
  · intro vertex vertexMember
    fin_cases vertexMember <;> native_decide
  · intro deduction deductionMember
    fin_cases deductionMember <;> native_decide

theorem input_validDLDS : ValidDLDS input := by
  apply validDLDSB_sound_for_hc
  native_decide

def root : Vertex := (encodeNDTreeAt tree 0 1).root

theorem input_uniqueRoot : UniqueRootAt input root := by
  refine ⟨⟨by native_decide, by native_decide⟩, ?_⟩
  intro candidate candidateMember terminal
  fin_cases candidateMember
  all_goals first
    | native_decide
    | (exfalso; revert terminal; native_decide)

theorem input_connected : EveryNodeReachesRoot input root := by
  intro vertex vertexMember
  fin_cases vertexMember <;> native_decide

theorem input_proper : ProperHCInput input :=
  ⟨input_isTreeDLDS, input_validDLDS, root,
    input_uniqueRoot, input_connected⟩

theorem input_mixedSourceHCClass : MixedSourceHCClass input := by
  apply mixedSourceHCClassB_sound
  native_decide

theorem proper_inputs_can_have_mixed_source_classes :
    ∃ d, ProperHCInput d ∧ MixedSourceHCClass d :=
  ⟨input, input_proper, input_mixedSourceHCClass⟩

theorem output_hypNoIncomingB_false : hypNoIncomingB output = false := by
  native_decide

theorem output_structuralValidB_false : structuralValidB output = false := by
  native_decide

theorem output_levelFormulaUniqueB_true :
    levelFormulaUniqueB output = true := by
  native_decide

theorem output_cursorRouteHeadCoverageB_false :
    cursorRouteHeadCoverageB output = false := by
  native_decide

theorem output_cursorFlowTailClosureB_false :
    cursorFlowTailClosureB output = false := by
  native_decide

theorem output_rootsDischargedB_false :
    rootsDischargedB output = false := by
  native_decide

theorem output_not_hcRepresentationWellFormed :
    ¬ HCRepresentationWellFormed output := by
  intro representation
  have checked := cursorRouteHeadCoverageB_complete
    representation.routeHeadCoverage
  rw [output_cursorRouteHeadCoverageB_false] at checked
  simp at checked

theorem proper_compression_can_fail_representation :
    ∃ d, ProperHCInput d ∧
      ¬ HCRepresentationWellFormed (compress_nodes_graph d) :=
  ⟨input, input_proper, output_not_hcRepresentationWellFormed⟩

/-!
The conservative source-separated variant removes the immediate source-role
failure, but it must retain both same-level/same-formula occurrences.  It is
therefore the identity on this smallest witness and cannot satisfy the current
compressed representation's `LevelFormulaUnique` requirement.
-/

theorem sourceSeparated_no_candidates :
    repeated_nodes_vertex_sourceSeparated input = [] := by
  native_decide

theorem sourceSeparated_output_eq_input :
    sourceSeparatedOutput = input := by
  exact compress_nodes_graph_sourceSeparated_eq_self_of_no_candidates
    input sourceSeparated_no_candidates

theorem sourceSeparated_output_hypNoIncomingB_true :
    hypNoIncomingB sourceSeparatedOutput = true := by
  native_decide

theorem sourceSeparated_output_structuralValidB_true :
    structuralValidB sourceSeparatedOutput = true := by
  native_decide

theorem sourceSeparated_output_oneEdgeB_true :
    oneEdgeB sourceSeparatedOutput = true := by
  native_decide

theorem sourceSeparated_output_routeHeadCoverageB_true :
    cursorRouteHeadCoverageB sourceSeparatedOutput = true := by
  native_decide

theorem sourceSeparated_output_flowTailClosureB_true :
    cursorFlowTailClosureB sourceSeparatedOutput = true := by
  native_decide

theorem sourceSeparated_output_rootsDischargedB_true :
    rootsDischargedB sourceSeparatedOutput = true := by
  native_decide

theorem sourceSeparated_output_levelFormulaUniqueB_false :
    levelFormulaUniqueB sourceSeparatedOutput = false := by
  native_decide

theorem sourceSeparated_output_not_levelFormulaUnique :
    ¬ LevelFormulaUnique sourceSeparatedOutput := by
  intro unique
  rw [sourceSeparated_output_eq_input] at unique
  obtain ⟨source, sourceMember, derived, derivedMember, distinct,
    sameLevel, sameFormula, _⟩ := input_mixedSourceHCClass
  exact distinct (unique source sourceMember derived derivedMember
    sameLevel sameFormula)

theorem sourceSeparated_output_not_hcRepresentationWellFormed :
    ¬ HCRepresentationWellFormed sourceSeparatedOutput := by
  intro representation
  exact sourceSeparated_output_not_levelFormulaUnique
    representation.levelFormulaUnique

theorem sourceSeparated_output_not_suitableGraph :
    ¬ SuitableGraph sourceSeparatedOutput := by
  intro suitable
  exact sourceSeparated_output_not_levelFormulaUnique
    suitable.levelFormulaUnique

theorem sourceSeparated_output_flowCompiledGridValid :
    FlowCompiledGridValid sourceSeparatedOutput :=
  (rootsDischargedB_iff_flowCompiledGridValid sourceSeparatedOutput).mp
    sourceSeparated_output_rootsDischargedB_true

end GenuineMixedSource


end DLDS.Compression.Domain
