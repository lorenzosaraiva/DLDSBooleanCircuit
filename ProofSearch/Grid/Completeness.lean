import ProofSearch.Grid.Soundness

/-!
# Space A: separation and completeness

This module isolates the proof-search grid from the DLDS verification layer.
The separation result is unconditional.  The unrestricted completeness target
is stated without adding it as an axiom; collision-free identity constructions
establish the exact-height and measured-height fragments below.
-/

open DLDS

namespace ProofSearch.Grid.Completeness

/-- The height parameter used by `buildGridFromDLDS`. -/
def levelBound (d : Graph) : Nat :=
  (d.NODES.map (·.LEVEL)).foldl max 0

/-- A graph-free presentation of the Space A grid. -/
def grid (formulas : List Formula) (height : Nat) :
    List (GridLayer formulas.length) :=
  List.replicate (height + 1)
    { nodes := formulas.map (nodeForFormula formulas)
      incoming := buildIncomingMap formulas }

/-- Transport a grid across equality of its formula-universe index. -/
def castGrid {formulas1 formulas2 : List Formula}
    (sameFormulas : formulas1 = formulas2)
    (layers : List (GridLayer formulas1.length)) :
    List (GridLayer formulas2.length) :=
  cast (congrArg (fun formulas => List (GridLayer formulas.length)) sameFormulas)
    layers

/-- `buildGridFromDLDS` forgets every graph field except the formula universe
and the maximum level.  `HEq` is necessary because `GridLayer` is indexed by
the length of the graph's formula universe. -/
theorem buildGridFromDLDS_eq_grid (d : Graph) :
    HEq (buildGridFromDLDS d) (grid (buildFormulas d) (levelBound d)) := by
  simp [buildGridFromDLDS, buildLayers, grid, levelBound]

/-- **Space A separation theorem.**  Two DLDSs with the same formula universe
and level bound compile to the same grid. -/
theorem buildGridFromDLDS_eq
    {d1 d2 : Graph}
    (formulaUniverse : buildFormulas d1 = buildFormulas d2)
    (height : levelBound d1 = levelBound d2) :
    HEq (buildGridFromDLDS d1) (buildGridFromDLDS d2) := by
  unfold buildGridFromDLDS buildLayers
  unfold levelBound at height
  rw [formulaUniverse, height]

/-- Homogeneous-equality presentation of the separation theorem. -/
theorem buildGridFromDLDS_cast_eq
    {d1 d2 : Graph}
    (formulaUniverse : buildFormulas d1 = buildFormulas d2)
    (height : levelBound d1 = levelBound d2) :
    castGrid formulaUniverse (buildGridFromDLDS d1) =
      buildGridFromDLDS d2 := by
  unfold castGrid
  apply eq_of_heq
  exact (cast_heq _ _).trans
    (buildGridFromDLDS_eq formulaUniverse height)

/-- The corrected Space A marker, presented directly as a function of the
formula universe, goal formula, and height allowance. -/
def markedB (formulas : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Bool :=
  ProofSearch.Grid.Definitions.markedBFor formulas goal
    (ProofSearch.Grid.Definitions.alphabetFor formulas) height paths

def Marked (formulas : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Prop :=
  markedB formulas goal height paths = true

/-- Acceptance of a fixed path assignment is a property of the formula
universe, goal, and height alone. -/
theorem marked_eq_of_universe_height_eq
    {formulas1 formulas2 : List Formula} {height1 height2 : Nat}
    (sameFormulas : formulas1 = formulas2)
    (sameHeight : height1 = height2)
    (goal : Formula) (paths : PathInput) :
    markedB formulas1 goal height1 paths =
      markedB formulas2 goal height2 paths := by
  subst formulas2
  subst height2
  rfl

/-- Graph-indexed form of `marked_eq_of_universe_height_eq`. -/
theorem marked_from_DLDS_iff
    {d1 d2 : Graph}
    (formulaUniverse : buildFormulas d1 = buildFormulas d2)
    (height : levelBound d1 = levelBound d2)
    (goal : Formula) (paths : PathInput) :
    Marked (buildFormulas d1) goal (levelBound d1) paths ↔
      Marked (buildFormulas d2) goal (levelBound d2) paths := by
  simp only [Marked]
  rw [marked_eq_of_universe_height_eq formulaUniverse height goal paths]

/-- Consequently, no marked path can distinguish two such DLDSs. -/
theorem no_marked_path_distinguishes
    {d1 d2 : Graph}
    (formulaUniverse : buildFormulas d1 = buildFormulas d2)
    (height : levelBound d1 = levelBound d2)
    (goal : Formula) :
    ¬ ∃ paths : PathInput,
      Marked (buildFormulas d1) goal (levelBound d1) paths ≠
        Marked (buildFormulas d2) goal (levelBound d2) paths := by
  intro distinguishes
  obtain ⟨paths, different⟩ := distinguishes
  exact different (propext
    (marked_from_DLDS_iff formulaUniverse height goal paths))

/-! ## A constructive completeness fragment -/

/-- The direct closed proof of `(A → B) → (A → B)`. -/
def directIdentityTree : ProofSearch.Grid.NDTree :=
  .intro ProofSearch.Grid.Definitions.AB ProofSearch.Grid.Definitions.AB
    (.hyp ProofSearch.Grid.Definitions.AB)

deriving instance DecidableEq for ProofSearch.Grid.NDTree

/-- Carry the `A → B` hypothesis through repetition cells and send it to the
final introduction cell.  Every unrelated origin token stops immediately. -/
def directIdentityPath (height : Nat) : PathInput :=
  [List.replicate height (0, 0),
   List.replicate height (0, 0),
   List.replicate (height - 1) (3, 0) ++ [(4, 1)],
   List.replicate height (0, 0)]

/-- The finite rectangular carrier used by the exact Space A counter. -/
def InPathCarrier (formulas : List Formula) (height : Nat)
    (paths : PathInput) : Prop :=
  paths.length = formulas.length ∧
    ∀ path ∈ paths,
      path.length = height ∧
        ∀ choice ∈ path, choice ∈ ProofSearch.Grid.Definitions.alphabetFor formulas

/-- Formula occurrences that an inverse construction must place in the grid. -/
def derivationFormulas : ProofSearch.Grid.NDTree → List Formula
  | .hyp formula => [formula]
  | .intro alpha beta premise =>
      (alpha >> beta) :: alpha :: beta :: derivationFormulas premise
  | .elim alpha beta major minor =>
      (alpha >> beta) :: alpha :: beta ::
        (derivationFormulas major ++ derivationFormulas minor)

/-- The unrestricted completeness target.  It is deliberately a proposition,
not an axiom: the current grid lacks the scheduling theorem needed to prove it
for arbitrary repeated formula occurrences and branching eliminations. -/
def CertifiedMarkedCompleteness : Prop :=
  ∀ (formulas : List Formula) (goal : Formula) (height : Nat)
      (tree : ProofSearch.Grid.NDTree),
    formulas.Nodup →
    (∀ formula ∈ derivationFormulas tree, formula ∈ formulas) →
    tree.wellFormedB = true →
    tree.conclusion = goal →
    tree.dependencies = [] →
    tree.height ≤ height →
    ∃ paths : PathInput,
      InPathCarrier formulas height paths ∧
        ProofSearch.Grid.CertifiedMarked
          formulas goal height paths

/-- Once an inverse has supplied a marked trace extracting the requested tree,
the certificate half is immediate.  The missing general step is constructing
such a trace without column/level collisions. -/
theorem certifiedMarked_of_encoded_tree
    {formulas : List Formula} {goal : Formula} {height : Nat}
    {tree : ProofSearch.Grid.NDTree} {paths : PathInput}
    (marked : ProofSearch.Grid.gridMarkedB
      formulas goal height paths = true)
    (extracted : ProofSearch.Grid.extractGoalCone
      formulas goal height paths = some tree)
    (wellFormed : tree.wellFormedB = true)
    (conclusion : tree.conclusion = goal)
    (dependencies : tree.dependencies = [])
    (heightBound : tree.height ≤ height) :
    ProofSearch.Grid.CertifiedMarked
      formulas goal height paths := by
  unfold ProofSearch.Grid.CertifiedMarked
    ProofSearch.Grid.certifiedMarkedB
    ProofSearch.Grid.extractedCertificateB
  rw [marked]
  simp [extracted, wellFormed, conclusion, dependencies, heightBound]

theorem directIdentityTree_certificate :
    directIdentityTree.wellFormedB = true ∧
    directIdentityTree.conclusion = ProofSearch.Grid.Definitions.alpha ∧
    directIdentityTree.dependencies = [] ∧
    directIdentityTree.height = 1 := by
  native_decide

theorem directIdentity_h3_certified :
    ProofSearch.Grid.CertifiedMarked
      ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
        (directIdentityPath 3) := by
  change ProofSearch.Grid.certifiedMarkedB
    ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
      (directIdentityPath 3) = true
  native_decide

theorem directIdentity_h4_certified :
    ProofSearch.Grid.CertifiedMarked
      ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 4
        (directIdentityPath 4) := by
  change ProofSearch.Grid.certifiedMarkedB
    ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 4
      (directIdentityPath 4) = true
  native_decide

theorem directIdentity_h5_certified :
    ProofSearch.Grid.CertifiedMarked
      ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 5
        (directIdentityPath 5) := by
  change ProofSearch.Grid.certifiedMarkedB
    ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 5
      (directIdentityPath 5) = true
  native_decide

theorem certifiedMarked_is_marked
    {formulas : List Formula} {goal : Formula} {height : Nat}
    {paths : PathInput}
    (certified : ProofSearch.Grid.CertifiedMarked
      formulas goal height paths) :
    ProofSearch.Grid.gridMarkedB formulas goal height paths = true := by
  unfold ProofSearch.Grid.CertifiedMarked
    ProofSearch.Grid.certifiedMarkedB at certified
  have parts :
      ProofSearch.Grid.gridMarkedB formulas goal height paths = true ∧
        ProofSearch.Grid.extractedCertificateB
          formulas goal height paths = true := by
    simpa only [Bool.and_eq_true] using certified
  exact parts.1

theorem directIdentity_marked_at_measured_heights :
    ProofSearch.Grid.gridMarkedB
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          (directIdentityPath 3) = true ∧
      ProofSearch.Grid.gridMarkedB
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 4
          (directIdentityPath 4) = true ∧
      ProofSearch.Grid.gridMarkedB
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 5
          (directIdentityPath 5) = true := by
  exact ⟨certifiedMarked_is_marked directIdentity_h3_certified,
    certifiedMarked_is_marked directIdentity_h4_certified,
    certifiedMarked_is_marked directIdentity_h5_certified⟩

/-- Exact validation of the padded inverse at all three measured heights. -/
theorem directIdentity_construction_validation :
    (InPathCarrier ProofSearch.Grid.Definitions.formulas 3 (directIdentityPath 3) ∧
      ProofSearch.Grid.extractGoalCone
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          (directIdentityPath 3) = some directIdentityTree) ∧
    (InPathCarrier ProofSearch.Grid.Definitions.formulas 4 (directIdentityPath 4) ∧
      ProofSearch.Grid.extractGoalCone
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 4
          (directIdentityPath 4) = some directIdentityTree) ∧
    (InPathCarrier ProofSearch.Grid.Definitions.formulas 5 (directIdentityPath 5) ∧
      ProofSearch.Grid.extractGoalCone
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 5
          (directIdentityPath 5) = some directIdentityTree) := by
  unfold InPathCarrier
  native_decide

/- The exact-height eta-expanded proof encoded by `witness1`: elimination,
then introduction of `A`, then introduction of `A → B`. -/
def etaIdentityTree : ProofSearch.Grid.NDTree :=
  .intro ProofSearch.Grid.Definitions.AB ProofSearch.Grid.Definitions.AB
    (.intro ProofSearch.Grid.Definitions.A ProofSearch.Grid.Definitions.B
      (.elim ProofSearch.Grid.Definitions.A ProofSearch.Grid.Definitions.B
        (.hyp ProofSearch.Grid.Definitions.AB) (.hyp ProofSearch.Grid.Definitions.A)))

theorem etaIdentityTree_certificate :
    etaIdentityTree.wellFormedB = true ∧
    etaIdentityTree.conclusion = ProofSearch.Grid.Definitions.alpha ∧
    etaIdentityTree.dependencies = [] ∧
    etaIdentityTree.height = 3 := by
  native_decide

/-- Restricted completeness at exact height: the inverse construction for the
eta-expanded identity tree has no vertical-placement freedom. -/
theorem etaIdentity_exactHeight_complete :
    InPathCarrier ProofSearch.Grid.Definitions.formulas 3 ProofSearch.Grid.Definitions.witness1 ∧
      ProofSearch.Grid.extractGoalCone
          ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
            ProofSearch.Grid.Definitions.witness1 = some etaIdentityTree ∧
      ProofSearch.Grid.CertifiedMarked
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          ProofSearch.Grid.Definitions.witness1 := by
  constructor
  · unfold InPathCarrier
    native_decide
  · constructor
    · native_decide
    · change ProofSearch.Grid.certifiedMarkedB
        ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha 3
          ProofSearch.Grid.Definitions.witness1 = true
      native_decide

/-- The constructed direct proof is complete at each height used by the exact
census.  Its logical height is one; the remaining levels are repetitions. -/
theorem directIdentity_complete_at_measured_heights
    (height : Nat) (measured : height = 3 ∨ height = 4 ∨ height = 5) :
    directIdentityTree.wellFormedB = true ∧
      directIdentityTree.conclusion = ProofSearch.Grid.Definitions.alpha ∧
      directIdentityTree.dependencies = [] ∧
      directIdentityTree.height ≤ height ∧
      ∃ paths : PathInput,
        InPathCarrier ProofSearch.Grid.Definitions.formulas height paths ∧
          ProofSearch.Grid.CertifiedMarked
            ProofSearch.Grid.Definitions.formulas ProofSearch.Grid.Definitions.alpha height paths := by
  obtain ⟨wellFormed, conclusion, dependencies, treeHeight⟩ :=
    directIdentityTree_certificate
  refine ⟨wellFormed, conclusion, dependencies, ?_, ?_⟩
  · rcases measured with rfl | rfl | rfl <;> simp [treeHeight]
  · rcases measured with rfl | rfl | rfl
    · exact ⟨directIdentityPath 3,
        directIdentity_construction_validation.1.1,
        directIdentity_h3_certified⟩
    · exact ⟨directIdentityPath 4,
        directIdentity_construction_validation.2.1.1,
        directIdentity_h4_certified⟩
    · exact ⟨directIdentityPath 5,
        directIdentity_construction_validation.2.2.1,
        directIdentity_h5_certified⟩

end ProofSearch.Grid.Completeness
