import Foundations.NaturalDeduction










open Semantic

namespace GridAcceptance


def levelBound (d : Graph) : Nat :=
  (d.NODES.map (·.LEVEL)).foldl max 0


def grid (formulas : List Formula) (height : Nat) :
    List (GridLayer formulas.length) :=
  List.replicate (height + 1)
    { nodes := formulas.map (nodeForFormula formulas)
      incoming := buildIncomingMap formulas }


def castGrid {formulas1 formulas2 : List Formula}
    (sameFormulas : formulas1 = formulas2)
    (layers : List (GridLayer formulas1.length)) :
    List (GridLayer formulas2.length) :=
  cast (congrArg (fun formulas => List (GridLayer formulas.length)) sameFormulas)
    layers




theorem buildGridFromDLDS_eq_grid (d : Graph) :
    HEq (buildGridFromDLDS d) (grid (buildFormulas d) (levelBound d)) := by
  simp [buildGridFromDLDS, buildLayers, grid, levelBound]



theorem buildGridFromDLDS_eq
    {d1 d2 : Graph}
    (formulaUniverse : buildFormulas d1 = buildFormulas d2)
    (height : levelBound d1 = levelBound d2) :
    HEq (buildGridFromDLDS d1) (buildGridFromDLDS d2) := by
  unfold buildGridFromDLDS buildLayers
  unfold levelBound at height
  rw [formulaUniverse, height]


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



def markedB (formulas : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Bool :=
  ProofSearchModel.markedBFor formulas goal
    (ProofSearchModel.alphabetFor formulas) height paths

def Marked (formulas : List Formula) (goal : Formula) (height : Nat)
    (paths : PathInput) : Prop :=
  markedB formulas goal height paths = true



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


theorem marked_from_DLDS_iff
    {d1 d2 : Graph}
    (formulaUniverse : buildFormulas d1 = buildFormulas d2)
    (height : levelBound d1 = levelBound d2)
    (goal : Formula) (paths : PathInput) :
    Marked (buildFormulas d1) goal (levelBound d1) paths ↔
      Marked (buildFormulas d2) goal (levelBound d2) paths := by
  simp only [Marked]
  rw [marked_eq_of_universe_height_eq formulaUniverse height goal paths]


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




def directIdentityTree : GridCertificateSoundness.NDTree :=
  .intro ProofSearchModel.AB ProofSearchModel.AB
    (.hyp ProofSearchModel.AB)

deriving instance DecidableEq for GridCertificateSoundness.NDTree



def directIdentityPath (height : Nat) : PathInput :=
  [List.replicate height (0, 0),
   List.replicate height (0, 0),
   List.replicate (height - 1) (3, 0) ++ [(4, 1)],
   List.replicate height (0, 0)]


def InPathCarrier (formulas : List Formula) (height : Nat)
    (paths : PathInput) : Prop :=
  paths.length = formulas.length ∧
    ∀ path ∈ paths,
      path.length = height ∧
        ∀ choice ∈ path, choice ∈ ProofSearchModel.alphabetFor formulas


def derivationFormulas : GridCertificateSoundness.NDTree → List Formula
  | .hyp formula => [formula]
  | .intro alpha beta premise =>
      (alpha >> beta) :: alpha :: beta :: derivationFormulas premise
  | .elim alpha beta major minor =>
      (alpha >> beta) :: alpha :: beta ::
        (derivationFormulas major ++ derivationFormulas minor)




def CertifiedMarkedCompleteness : Prop :=
  ∀ (formulas : List Formula) (goal : Formula) (height : Nat)
      (tree : GridCertificateSoundness.NDTree),
    formulas.Nodup →
    (∀ formula ∈ derivationFormulas tree, formula ∈ formulas) →
    tree.wellFormedB = true →
    tree.conclusion = goal →
    tree.dependencies = [] →
    tree.height ≤ height →
    ∃ paths : PathInput,
      InPathCarrier formulas height paths ∧
        GridCertificateSoundness.CertifiedMarked
          formulas goal height paths




theorem certifiedMarked_of_encoded_tree
    {formulas : List Formula} {goal : Formula} {height : Nat}
    {tree : GridCertificateSoundness.NDTree} {paths : PathInput}
    (marked : GridCertificateSoundness.gridMarkedB
      formulas goal height paths = true)
    (extracted : GridCertificateSoundness.extractGoalCone
      formulas goal height paths = some tree)
    (wellFormed : tree.wellFormedB = true)
    (conclusion : tree.conclusion = goal)
    (dependencies : tree.dependencies = [])
    (heightBound : tree.height ≤ height) :
    GridCertificateSoundness.CertifiedMarked
      formulas goal height paths := by
  unfold GridCertificateSoundness.CertifiedMarked
    GridCertificateSoundness.certifiedMarkedB
    GridCertificateSoundness.extractedCertificateB
  rw [marked]
  simp [extracted, wellFormed, conclusion, dependencies, heightBound]

theorem directIdentityTree_certificate :
    directIdentityTree.wellFormedB = true ∧
    directIdentityTree.conclusion = ProofSearchModel.alpha ∧
    directIdentityTree.dependencies = [] ∧
    directIdentityTree.height = 1 := by
  native_decide

theorem directIdentity_h3_certified :
    GridCertificateSoundness.CertifiedMarked
      ProofSearchModel.formulas ProofSearchModel.alpha 3
        (directIdentityPath 3) := by
  change GridCertificateSoundness.certifiedMarkedB
    ProofSearchModel.formulas ProofSearchModel.alpha 3
      (directIdentityPath 3) = true
  native_decide

theorem directIdentity_h4_certified :
    GridCertificateSoundness.CertifiedMarked
      ProofSearchModel.formulas ProofSearchModel.alpha 4
        (directIdentityPath 4) := by
  change GridCertificateSoundness.certifiedMarkedB
    ProofSearchModel.formulas ProofSearchModel.alpha 4
      (directIdentityPath 4) = true
  native_decide

theorem directIdentity_h5_certified :
    GridCertificateSoundness.CertifiedMarked
      ProofSearchModel.formulas ProofSearchModel.alpha 5
        (directIdentityPath 5) := by
  change GridCertificateSoundness.certifiedMarkedB
    ProofSearchModel.formulas ProofSearchModel.alpha 5
      (directIdentityPath 5) = true
  native_decide

theorem certifiedMarked_is_marked
    {formulas : List Formula} {goal : Formula} {height : Nat}
    {paths : PathInput}
    (certified : GridCertificateSoundness.CertifiedMarked
      formulas goal height paths) :
    GridCertificateSoundness.gridMarkedB formulas goal height paths = true := by
  unfold GridCertificateSoundness.CertifiedMarked
    GridCertificateSoundness.certifiedMarkedB at certified
  have parts :
      GridCertificateSoundness.gridMarkedB formulas goal height paths = true ∧
        GridCertificateSoundness.extractedCertificateB
          formulas goal height paths = true := by
    simpa only [Bool.and_eq_true] using certified
  exact parts.1

theorem directIdentity_marked_at_measured_heights :
    GridCertificateSoundness.gridMarkedB
        ProofSearchModel.formulas ProofSearchModel.alpha 3
          (directIdentityPath 3) = true ∧
      GridCertificateSoundness.gridMarkedB
        ProofSearchModel.formulas ProofSearchModel.alpha 4
          (directIdentityPath 4) = true ∧
      GridCertificateSoundness.gridMarkedB
        ProofSearchModel.formulas ProofSearchModel.alpha 5
          (directIdentityPath 5) = true := by
  exact ⟨certifiedMarked_is_marked directIdentity_h3_certified,
    certifiedMarked_is_marked directIdentity_h4_certified,
    certifiedMarked_is_marked directIdentity_h5_certified⟩


theorem directIdentity_construction_validation :
    (InPathCarrier ProofSearchModel.formulas 3 (directIdentityPath 3) ∧
      GridCertificateSoundness.extractGoalCone
        ProofSearchModel.formulas ProofSearchModel.alpha 3
          (directIdentityPath 3) = some directIdentityTree) ∧
    (InPathCarrier ProofSearchModel.formulas 4 (directIdentityPath 4) ∧
      GridCertificateSoundness.extractGoalCone
        ProofSearchModel.formulas ProofSearchModel.alpha 4
          (directIdentityPath 4) = some directIdentityTree) ∧
    (InPathCarrier ProofSearchModel.formulas 5 (directIdentityPath 5) ∧
      GridCertificateSoundness.extractGoalCone
        ProofSearchModel.formulas ProofSearchModel.alpha 5
          (directIdentityPath 5) = some directIdentityTree) := by
  unfold InPathCarrier
  native_decide



def etaIdentityTree : GridCertificateSoundness.NDTree :=
  .intro ProofSearchModel.AB ProofSearchModel.AB
    (.intro ProofSearchModel.A ProofSearchModel.B
      (.elim ProofSearchModel.A ProofSearchModel.B
        (.hyp ProofSearchModel.AB) (.hyp ProofSearchModel.A)))

theorem etaIdentityTree_certificate :
    etaIdentityTree.wellFormedB = true ∧
    etaIdentityTree.conclusion = ProofSearchModel.alpha ∧
    etaIdentityTree.dependencies = [] ∧
    etaIdentityTree.height = 3 := by
  native_decide



theorem etaIdentity_exactHeight_complete :
    InPathCarrier ProofSearchModel.formulas 3 ProofSearchModel.witness1 ∧
      GridCertificateSoundness.extractGoalCone
          ProofSearchModel.formulas ProofSearchModel.alpha 3
            ProofSearchModel.witness1 = some etaIdentityTree ∧
      GridCertificateSoundness.CertifiedMarked
        ProofSearchModel.formulas ProofSearchModel.alpha 3
          ProofSearchModel.witness1 := by
  constructor
  · unfold InPathCarrier
    native_decide
  · constructor
    · native_decide
    · change GridCertificateSoundness.certifiedMarkedB
        ProofSearchModel.formulas ProofSearchModel.alpha 3
          ProofSearchModel.witness1 = true
      native_decide



theorem directIdentity_complete_at_measured_heights
    (height : Nat) (measured : height = 3 ∨ height = 4 ∨ height = 5) :
    directIdentityTree.wellFormedB = true ∧
      directIdentityTree.conclusion = ProofSearchModel.alpha ∧
      directIdentityTree.dependencies = [] ∧
      directIdentityTree.height ≤ height ∧
      ∃ paths : PathInput,
        InPathCarrier ProofSearchModel.formulas height paths ∧
          GridCertificateSoundness.CertifiedMarked
            ProofSearchModel.formulas ProofSearchModel.alpha height paths := by
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

end GridAcceptance
