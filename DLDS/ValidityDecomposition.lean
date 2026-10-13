import Certificate.Flow.GeneratorCompleteness

open DLDS FlowSemantics Certificate.Flow.Foundation Certificate.Flow
open Certificate.Flow.GeneratorCompleteness ProofSearch.FlowCorrespondence

namespace DLDS.ValidityDecomposition

/-!
# Declarative DLDS validity versus executable Flow representation

This module keeps the declarative `ValidDLDS` conditions separate from the
additional representation invariants needed by the executable Flow and
certificate developments.
-/

/-- The independently meaningful horizontal-compression/executable-Flow
invariants which are not fields of the current declarative `ValidDLDS` record.
No certificate or certificate checker occurs in this package. -/
structure HCRepresentationWellFormed (d : Graph) : Prop where
  oneEdgePerColour : OneEdgePerColourPerNode d
  routeHeadCoverage : CursorRouteHeadCoverage d
  flowTailClosure : CursorFlowTailClosure d
  levelFormulaUnique : LevelFormulaUnique d

lemma exhaustedFlowArrivalAtB_complete {d : Graph} {v : Vertex}
    {route : FlowPair} (h : ExhaustedFlowArrivalAt d v route) :
    exhaustedFlowArrivalAtB d v route = true := by
  unfold ExhaustedFlowArrivalAt at h
  unfold exhaustedFlowArrivalAtB
  simp only [Bool.and_eq_true, List.isEmpty_iff, Option.isNone_iff_eq_none]
  exact ⟨⟨h.1, h.2.1⟩, h.2.2⟩

/-- The executable route-head checker is complete as well as sound.  The
original Flow-closure module only needed (and therefore only exported) its
soundness direction. -/
theorem cursorRouteHeadCoverageB_complete {d : Graph}
    (coverage : CursorRouteHeadCoverage d) :
    cursorRouteHeadCoverageB d = true := by
  classical
  unfold cursorRouteHeadCoverageB
  rw [List.all_eq_true]
  intro v member
  by_cases outgoing : get_rule.outgoing v d = []
  · simp [outgoing]
  · have atV := coverage v member outgoing
    have outgoingEmpty : (get_rule.outgoing v d).isEmpty = false := by
      simpa [List.isEmpty_iff] using outgoing
    simp only [outgoingEmpty, Bool.false_or]
    unfold cursorRouteHeadCoverageAtB
    rw [Bool.and_eq_true, List.all_eq_true, List.all_eq_true]
    constructor
    · intro route routeMember
      rcases atV.1 route routeMember with exhausted | covered
      · rw [Bool.or_eq_true]
        exact Or.inl (exhaustedFlowArrivalAtB_complete exhausted)
      · rw [Bool.or_eq_true]
        right
        rw [List.any_eq_true]
        obtain ⟨edge, edgeMember, colour⟩ := covered
        exact ⟨edge, edgeMember, by simpa using colour⟩
    · intro edge edgeMember
      rw [List.any_eq_true]
      obtain ⟨route, routeMember, colour⟩ := atV.2 edge edgeMember
      exact ⟨route, routeMember, by simpa using colour⟩

theorem cursorRouteHeadCoverageB_iff {d : Graph} :
    cursorRouteHeadCoverageB d = true ↔ CursorRouteHeadCoverage d :=
  ⟨cursorRouteHeadCoverageB_sound d, cursorRouteHeadCoverageB_complete⟩

lemma exhaustedConsumedTailAtB_complete {d : Graph} {w : Vertex}
    {tail : ColourPath} (h : ExhaustedConsumedTailAt d w tail) :
    exhaustedConsumedTailAtB d w tail = true := by
  unfold ExhaustedConsumedTailAt at h
  unfold exhaustedConsumedTailAtB
  simp [h.1, h.2.1, h.2.2]

lemma elimPremiseAtB_complete {d : Graph} {w u : Vertex}
    (h : ElimPremiseAt d w u) : elimPremiseAtB d w u = true := by
  unfold ElimPremiseAt at h
  unfold elimPremiseAtB
  obtain ⟨major, minor, pairMember, side⟩ := h
  rw [List.any_eq_true]
  refine ⟨(major, minor), pairMember, ?_⟩
  rw [Bool.or_eq_true]
  rcases side with rfl | rfl
  · exact Or.inl (by simp)
  · exact Or.inr (by simp)

lemma cursorCarrierAccountedAtB_complete {d : Graph} {w u : Vertex}
    (h : CursorCarrierAccountedAt d w u) :
    cursorCarrierAccountedAtB d w u = true := by
  obtain ⟨alpha, formula, accounted⟩ := h
  unfold cursorCarrierAccountedAtB
  rw [formula]
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true]
  constructor
  · trivial
  · intro route routeMember edge edgeMember
    cases consumed : consume edge route.2 with
    | none => simp
    | some tail =>
        rcases accounted route routeMember edge edgeMember tail consumed with
          exhausted | continued
        · rw [Bool.or_eq_true]
          exact Or.inl (exhaustedConsumedTailAtB_complete exhausted)
        · rw [Bool.or_eq_true]
          right
          rw [List.any_eq_true]
          obtain ⟨dependency, resultMember⟩ := continued
          exact ⟨(dependency, tail), resultMember, by simp⟩

lemma cursorPairFlowTailClosureAtB_complete {d : Graph} {w major minor : Vertex}
    (h : CursorPairFlowTailClosure d w major minor) :
    cursorPairFlowTailClosureAtB d w major minor = true := by
  unfold cursorPairFlowTailClosureAtB
  rw [Bool.and_eq_true]
  constructor
  · rw [List.all_eq_true]
    intro minorRoute minorMember
    rw [List.all_eq_true]
    intro minorEdge edgeMember
    cases consumed : consume minorEdge minorRoute.2 with
    | none => simp
    | some tail =>
        rcases h.1 minorRoute minorMember minorEdge edgeMember tail consumed with
          exhausted | partner
        · rw [Bool.or_eq_true]
          exact Or.inl (exhaustedConsumedTailAtB_complete exhausted)
        · rw [Bool.or_eq_true]
          right
          rw [List.any_eq_true]
          obtain ⟨majorRoute, majorMember, majorEdge, majorEdgeMember,
            agreement⟩ := partner
          refine ⟨majorRoute, majorMember, ?_⟩
          rw [List.any_eq_true]
          refine ⟨majorEdge, majorEdgeMember, ?_⟩
          rw [Bool.or_eq_true]
          rcases agreement with exactTail | reseed
          · exact Or.inl (by simpa using exactTail)
          · exact Or.inr reseed
  · rw [List.all_eq_true]
    intro majorRoute majorMember
    rw [List.all_eq_true]
    intro majorEdge edgeMember
    cases consumed : consume majorEdge majorRoute.2 with
    | none => simp
    | some tail =>
        rcases h.2 majorRoute majorMember majorEdge edgeMember tail consumed with
          exhausted | partner
        · rw [Bool.or_eq_true]
          exact Or.inl (exhaustedConsumedTailAtB_complete exhausted)
        · rw [Bool.or_eq_true]
          right
          rw [List.any_eq_true]
          obtain ⟨minorRoute, minorMember, minorEdge, minorEdgeMember,
            agreement⟩ := partner
          refine ⟨minorRoute, minorMember, ?_⟩
          rw [List.any_eq_true]
          refine ⟨minorEdge, minorEdgeMember, ?_⟩
          rw [Bool.or_eq_true]
          rcases agreement with exactTail | reseed
          · exact Or.inl (by simpa using exactTail)
          · exact Or.inr reseed

theorem cursorFlowTailClosureB_complete {d : Graph}
    (closure : CursorFlowTailClosure d) :
    cursorFlowTailClosureB d = true := by
  classical
  unfold cursorFlowTailClosureB
  rw [List.all_eq_true]
  intro w member
  unfold cursorFlowTailClosureAtB
  cases pairs : elimPairsAt d w with
  | nil => rfl
  | cons pair rest =>
      have nonempty : elimPairsAt d w ≠ [] := by simp [pairs]
      obtain ⟨predecessors, pairClosure⟩ := closure w member nonempty
      simp only [Bool.and_eq_true, List.all_eq_true]
      constructor
      · intro predecessor predecessorMember
        rw [Bool.or_eq_true]
        rcases predecessors predecessor predecessorMember with premise | carrier
        · exact Or.inl (elimPremiseAtB_complete premise)
        · exact Or.inr (cursorCarrierAccountedAtB_complete carrier)
      · intro current currentMember
        rcases current with ⟨major, minor⟩
        exact cursorPairFlowTailClosureAtB_complete
          (pairClosure major minor (by rw [pairs]; exact currentMember))

theorem cursorFlowTailClosureB_iff {d : Graph} :
    cursorFlowTailClosureB d = true ↔ CursorFlowTailClosure d :=
  ⟨cursorFlowTailClosureB_sound d, cursorFlowTailClosureB_complete⟩

/-- The five thematic components of the current `ValidDLDS` record.  This
record is only a regrouping of the declarative definition; in particular it
does not add executable Flow or horizontal-compression assumptions. -/
structure DeclarativeValidityComponents (d : Graph) : Prop where
  hygiene : check_dlds d
  localInference :
    HypothesesHaveNoIncoming d ∧ LocalRuleCorrect d
  deductionStructure :
    LeveledColored d ∧ Simplicity d ∧ ColorAcyclicity d
  ancestorProvenance :
    AncestorSimplicity d ∧ AncestorEdges d ∧
      AncestorBackwayInformation d ∧ NonNestedAncestorEdges d
  rootDecoration : RootDischarged d

theorem validDLDS_iff_declarativeValidityComponents {d : Graph} :
    ValidDLDS d ↔ DeclarativeValidityComponents d := by
  constructor
  · intro valid
    exact
      { hygiene := valid.hygiene
        localInference := ⟨valid.hypNoIncoming, valid.localRuleCorrect⟩
        deductionStructure :=
          ⟨valid.leveledColored, valid.simplicity, valid.colorAcyclicity⟩
        ancestorProvenance :=
          ⟨valid.ancestorSimplicity, valid.ancestorEdges,
            valid.ancestorBackwayInformation, valid.nonNestedAncestorEdges⟩
        rootDecoration := valid.rootDischarged }
  · intro components
    exact
      { hygiene := components.hygiene
        leveledColored := components.deductionStructure.1
        simplicity := components.deductionStructure.2.1
        ancestorSimplicity := components.ancestorProvenance.1
        hypNoIncoming := components.localInference.1
        localRuleCorrect := components.localInference.2
        rootDischarged := components.rootDecoration
        colorAcyclicity := components.deductionStructure.2.2
        ancestorEdges := components.ancestorProvenance.2.1
        ancestorBackwayInformation := components.ancestorProvenance.2.2.1
        nonNestedAncestorEdges := components.ancestorProvenance.2.2.2 }

private lemma classifyRule_hypothesis_hypothesis
    {d : Graph} {v : Vertex}
    (classified : classifyRule? v d = some .hypothesis) :
    v.HYPOTHESIS = true := by
  unfold classifyRule? at classified
  by_cases hypothesis : v.HYPOTHESIS = true
  · exact hypothesis
  · simp only [hypothesis] at classified
    cases incoming : get_rule.incoming v d with
    | nil => simp [incoming] at classified
    | cons edge rest =>
        cases rest with
        | nil => simp [incoming] at classified
        | cons edge' rest' =>
            cases rest' with
            | nil =>
                simp only [incoming] at classified
                split at classified
                · rename_i impossible
                  simp at impossible
                · split at classified <;> simp at classified
            | cons edge'' rest'' => simp [incoming] at classified

private lemma classifyRule_intro_shape
    {d : Graph} {v : Vertex} {premise : Deduction}
    (classified : classifyRule? v d = some (.intro premise)) :
    NodeIntroShape d v := by
  have classifiedOriginal := classified
  unfold classifyRule? at classified
  by_cases hypothesis : v.HYPOTHESIS = true
  · simp [hypothesis] at classified
  · simp only [hypothesis] at classified
    cases incoming : get_rule.incoming v d with
    | nil => simp [incoming] at classified
    | cons edge rest =>
        cases rest with
        | nil =>
            simp only [incoming, Bool.false_eq_true, ↓reduceIte] at classified
            split at classified
            next consequent =>
              have edgeEq : edge = premise := by
                simpa using classified
              subst edge
              obtain ⟨alpha, beta, formula⟩ :=
                classifyRule?_intro_formula_implication
                  (d := d) (v := v) (p := premise) classifiedOriginal
              have premiseFormula : premise.START.FORMULA = beta := by
                rw [formula] at consequent
                simpa [consequent?] using consequent.symm
              refine ⟨premise, alpha, beta, ?_, incoming, formula,
                premiseFormula, ?_⟩
              · exact Bool.eq_false_of_not_eq_true hypothesis
              · unfold elimPairsAt
                rw [predsOf_eq, incoming]
                simp [formula_ne_implication_self]
            next notConsequent => simp at classified
        | cons edge' rest' =>
            cases rest' with
            | nil =>
                simp only [incoming] at classified
                split at classified
                · simp at classified
                · split at classified <;> simp at classified
            | cons edge'' rest'' => simp [incoming] at classified

private lemma classifyRule_elim_shape
    {d : Graph} {v : Vertex} {major minor : Deduction}
    (classified : classifyRule? v d = some (.elim major minor)) :
    NodeElimShape d v := by
  have majorMember := (classifyRule?_elim_major_mem_incoming classified).1
  have minorMember : minor ∈ get_rule.incoming v d := by
    unfold classifyRule? at classified
    by_cases hypothesis : v.HYPOTHESIS = true
    · simp [hypothesis] at classified
    · simp only [hypothesis] at classified
      cases incoming : get_rule.incoming v d with
      | nil => simp [incoming] at classified
      | cons edge rest =>
          cases rest with
          | nil => simp [incoming] at classified
          | cons edge' rest' =>
              cases rest' with
              | nil =>
                  simp only [incoming] at classified
                  split at classified
                  next direct =>
                    simp_all
                  next notDirect =>
                    split at classified
                    next reverse =>
                      simp at classified
                      have minorEq := classified.2
                      subst minor
                      simp
                    next notReverse =>
                      split at classified
                      next reverse' =>
                        simp at classified
                        have minorEq := classified.2
                        subst minor
                        simp
                      next notReverse' => simp at classified
              | cons edge'' rest'' => simp [incoming] at classified
  have majorPred : major.START ∈ predsOf d v := by
    rw [predsOf_eq, List.mem_eraseDups, List.mem_map]
    exact ⟨major, majorMember, rfl⟩
  have minorPred : minor.START ∈ predsOf d v := by
    rw [predsOf_eq, List.mem_eraseDups, List.mem_map]
    exact ⟨minor, minorMember, rfl⟩
  have pairMember : (major.START, minor.START) ∈ elimPairsAt d v := by
    unfold elimPairsAt
    rw [List.mem_flatMap]
    refine ⟨major.START, majorPred, ?_⟩
    rw [List.mem_filterMap]
    refine ⟨minor.START, minorPred, ?_⟩
    simp [classifyRule?_elim_major_formula_eq_minor classified]
  refine ⟨?_, ?_, ?_⟩
  · by_contra hypothesis
    have : v.HYPOTHESIS = true := by simpa using hypothesis
    unfold classifyRule? at classified
    simp [this] at classified
  · cases preds : predsOf d v with
    | nil => simp [preds] at minorPred
    | cons first rest => rfl
  · exact List.ne_nil_of_mem pairMember

lemma ruleShapeOK_implies_localShape_valid {d : Graph} {v : Vertex}
    (shape : RuleShapeOK v d) : localShape d v ≠ .invalid := by
  unfold RuleShapeOK ruleShapeOKB at shape
  cases classified : classifyRule? v d with
  | none => simp [classified] at shape
  | some rule =>
      cases rule with
      | hypothesis =>
          have hypothesis := classifyRule_hypothesis_hypothesis classified
          simp [localShape, hypothesis]
      | intro premise =>
          obtain ⟨premise', alpha, beta, hypothesis, incoming, formula,
              premiseFormula, noElim⟩ := classifyRule_intro_shape classified
          have preds : predsOf d v = [premise'.START] := by
            rw [predsOf_eq, incoming]
            rfl
          simp [localShape, hypothesis, preds, noElim, formula,
            premiseFormula]
      | elim major minor =>
          obtain ⟨hypothesis, predsNonempty, pairsNonempty⟩ :=
            classifyRule_elim_shape classified
          simp only [localShape, hypothesis, Bool.false_eq_true, predsNonempty, Bool.or_false, if_false]
          cases pairs : elimPairsAt d v with
          | nil => exact (pairsNonempty pairs).elim
          | cons pair rest =>
              cases rest <;> simp

theorem validDLDS_localShapesValid {d : Graph} (valid : ValidDLDS d) :
    ∀ v ∈ d.NODES, localShape d v ≠ .invalid := by
  intro v member
  exact ruleShapeOK_implies_localShape_valid
    (ruleShapeOK_of_valid_node d valid member)

theorem validDLDS_structuralValid {d : Graph} (valid : ValidDLDS d) :
    structuralValid d := by
  refine ⟨valid.leveledColored, ?_, ?_⟩
  · intro edge member
    exact valid.hygiene.2.1 member
  intro v member incomingNonempty outgoingNonempty
  have rule := valid.localRuleCorrect v member outgoingNonempty
  unfold RuleShapeOK ruleShapeOKB at rule
  cases classified : classifyRule? v d with
  | none => simp [classified] at rule
  | some kind =>
      cases kind with
      | hypothesis =>
          have hypothesis := classifyRule_hypothesis_hypothesis classified
          have noIncoming := valid.hypNoIncoming v member hypothesis
          exact (incomingNonempty noIncoming).elim
      | intro premise => exact Or.inr (classifyRule_intro_shape classified)
      | elim major minor => exact Or.inl (classifyRule_elim_shape classified)

theorem validDLDS_hypothesesHaveNoIncoming {d : Graph}
    (valid : ValidDLDS d) : HypothesesHaveNoIncoming d :=
  valid.hypNoIncoming

theorem validDLDS_rootDecoration {d : Graph} (valid : ValidDLDS d) :
    RootDischarged d :=
  valid.rootDischarged

/-- Although `Simplicity` alone does not make the edge *list* duplicate-free,
full declarative validity does: a repeated edge would occur at least twice in
its endpoint's incoming list, which is incompatible with every legal rule
shape (and with the no-incoming condition in the hypothesis case). -/
theorem validDLDS_edgesNodup {d : Graph} (valid : ValidDLDS d) :
    d.EDGES.Nodup := by
  rw [List.nodup_iff_count_le_one]
  intro edge
  by_contra notAtMostOne
  have countEdges : 2 ≤ d.EDGES.count edge := by omega
  have edgeMember : edge ∈ d.EDGES :=
    List.count_pos_iff.mp (by omega)
  have endMember : edge.END ∈ d.NODES :=
    (valid.hygiene.2.1 edgeMember).2
  have countIncoming : 2 ≤ (get_rule.incoming edge.END d).count edge := by
    rw [incoming_eq_filter]
    simpa using countEdges
  have shape : RuleShapeOK edge.END d := by
    by_cases outgoing : get_rule.outgoing edge.END d = []
    · exact (valid.rootDischarged edge.END endMember outgoing).1
    · exact (valid.localRuleCorrect edge.END endMember outgoing).1
  unfold RuleShapeOK ruleShapeOKB at shape
  cases classified : classifyRule? edge.END d with
  | none => simp [classified] at shape
  | some rule =>
      unfold classifyRule? at classified
      by_cases hypothesis : edge.END.HYPOTHESIS = true
      · have noIncoming := valid.hypNoIncoming edge.END endMember hypothesis
        rw [noIncoming] at countIncoming
        simp at countIncoming
      · simp only [hypothesis, Bool.false_eq_true, if_false] at classified
        cases incoming : get_rule.incoming edge.END d with
        | nil => simp [incoming] at countIncoming
        | cons first rest =>
            cases rest with
            | nil =>
                rw [incoming] at countIncoming
                have countBound : [first].count edge ≤ [first].length :=
                  List.count_le_length
                simp only [List.length_cons, List.length_nil] at countBound
                omega
            | cons second rest' =>
                cases rest' with
                | nil =>
                    rw [incoming] at countIncoming
                    have firstEq : first = edge := by
                      by_contra different
                      rw [List.count_cons_of_ne different] at countIncoming
                      have countBound : [second].count edge ≤ [second].length :=
                        List.count_le_length
                      simp only [List.length_cons, List.length_nil] at countBound
                      omega
                    have secondEq : second = edge := by
                      by_contra different
                      subst first
                      rw [List.count_cons_self,
                        List.count_cons_of_ne different] at countIncoming
                      simp at countIncoming
                    subst first
                    subst second
                    simp only [incoming] at classified
                    split at classified
                    next direct =>
                      exact (formula_ne_implication_self
                        edge.START.FORMULA edge.END.FORMULA direct).elim
                    next notDirect =>
                      simp at classified
                | cons third rest'' => simp [incoming] at classified

/-- Local production cannot succeed at an implementation-invalid rule shape.
This is independent of certificate expansion and of raw Flow. -/
theorem localChecks_localShapesValid {d : Graph} {certificate : Certificate}
    (checked : LocalChecks d certificate) :
    ∀ vertex ∈ d.NODES, localShape d vertex ≠ .invalid := by
  obtain ⟨graph, production, _⟩ := checked
  subst d
  intro vertex member invalid
  have produced := production vertex member
  unfold productionAtB at produced
  rw [invalid] at produced
  simp at produced

/-- The certificate-local root check contains the declarative `outDep` root
condition as a conjunct.  It is still weaker than universal raw-Flow root
discharge. -/
theorem localChecks_rootDischarged {d : Graph} {certificate : Certificate}
    (checked : LocalChecks d certificate) : RootDischarged d := by
  obtain ⟨graph, _, _, _, roots, _⟩ := checked
  subst d
  intro vertex member terminal
  have atRoot := roots vertex member (by simpa [List.isEmpty_iff] using terminal)
  exact atRoot.2

/-- The precise positive bridge: declarative validity supplies rule/level/root
correctness, while the independent HC package supplies executable occurrence
and routing invariants. -/
theorem validDLDS_and_hcRepresentation_implies_suitableGraph {d : Graph}
    (valid : ValidDLDS d) (representation : HCRepresentationWellFormed d) :
    SuitableGraph d where
  structural := validDLDS_structuralValid valid
  localShapesValid := validDLDS_localShapesValid valid
  edgesNodup := validDLDS_edgesNodup valid
  oneEdgePerColour := representation.oneEdgePerColour
  routeHeadCoverage := representation.routeHeadCoverage
  flowTailClosure := representation.flowTailClosure
  levelFormulaUnique := representation.levelFormulaUnique
  hypothesesHaveNoIncoming := valid.hypNoIncoming
  rootDecoration := valid.rootDischarged

/-- Under the independent representation package, the already-proved
Flow/Grid and certificate equivalences apply.  Notice that declarative
validity is not used as a substitute for raw-Flow root discharge. -/
theorem flowCompiledGridValid_iff_exists_strongCertificate_of_validDLDS
    {d : Graph} (valid : ValidDLDS d)
    (representation : HCRepresentationWellFormed d) :
    FlowCompiledGridValid d ↔
      ∃ certificate, StrongFormationValid d certificate := by
  exact flowCompiledGridValid_iff_exists_strongCertificate d
    (validDLDS_and_hcRepresentation_implies_suitableGraph valid representation)

theorem flowCompiledGridValid_iff_generatedStrongFormation_of_validDLDS
    {d : Graph} (valid : ValidDLDS d)
    (representation : HCRepresentationWellFormed d) :
    FlowCompiledGridValid d ↔
      StrongFormationValid d (generateCertificate d) := by
  exact flowCompiledGridValid_iff_generatedStrongFormation d
    (validDLDS_and_hcRepresentation_implies_suitableGraph valid representation)

/-- This corollary is stated on the declaratively valid, representation-well-formed domain; raw root discharge supplies its conclusion. -/
theorem validDLDS_to_flowCompiledGridValid
    {d : Graph} (_valid : ValidDLDS d)
    (_representation : HCRepresentationWellFormed d)
    (roots : rootsDischargedB d = true) :
    FlowCompiledGridValid d :=
  (rootsDischargedB_iff_flowCompiledGridValid d).mp roots

theorem validDLDS_to_generatedStrongFormation
    {d : Graph} (valid : ValidDLDS d)
    (representation : HCRepresentationWellFormed d)
    (roots : rootsDischargedB d = true) :
    StrongFormationValid d (generateCertificate d) := by
  exact (rootsDischargedB_iff_generatedStrongFormation d
    (validDLDS_and_hcRepresentation_implies_suitableGraph valid representation)).mp roots

theorem validDLDS_to_exists_strongCertificate
    {d : Graph} (valid : ValidDLDS d)
    (representation : HCRepresentationWellFormed d)
    (roots : rootsDischargedB d = true) :
    ∃ certificate, StrongFormationValid d certificate := by
  exact (rootsDischargedB_iff_exists_strongCertificate d
    (validDLDS_and_hcRepresentation_implies_suitableGraph valid representation)).mp roots

/-- Exact decomposition of the *augmented* declarative/executable package.
Raw-Flow root discharge is explicit on the left because it is not a
consequence of `ValidDLDS`. -/
theorem validDLDS_hc_roots_iff_declarative_hc_flowGrid {d : Graph} :
    (ValidDLDS d ∧ HCRepresentationWellFormed d ∧
      rootsDischargedB d = true) ↔
    (DeclarativeValidityComponents d ∧ HCRepresentationWellFormed d ∧
      FlowCompiledGridValid d) := by
  constructor
  · rintro ⟨valid, representation, roots⟩
    exact ⟨validDLDS_iff_declarativeValidityComponents.mp valid,
      representation,
      (rootsDischargedB_iff_flowCompiledGridValid d).mp roots⟩
  · rintro ⟨components, representation, grid⟩
    exact ⟨validDLDS_iff_declarativeValidityComponents.mpr components,
      representation,
      (rootsDischargedB_iff_flowCompiledGridValid d).mpr grid⟩

theorem validDLDS_hc_roots_iff_declarative_hc_exists_strongCertificate
    {d : Graph} :
    (ValidDLDS d ∧ HCRepresentationWellFormed d ∧
      rootsDischargedB d = true) ↔
    (DeclarativeValidityComponents d ∧ HCRepresentationWellFormed d ∧
      ∃ certificate, StrongFormationValid d certificate) := by
  constructor
  · rintro ⟨valid, representation, roots⟩
    exact ⟨validDLDS_iff_declarativeValidityComponents.mp valid,
      representation,
      validDLDS_to_exists_strongCertificate valid representation roots⟩
  · rintro ⟨components, representation, certificate, strong⟩
    exact ⟨validDLDS_iff_declarativeValidityComponents.mpr components,
      representation,
      strongFormationValid_rootsDischargedB d certificate strong⟩

theorem validDLDS_hc_roots_iff_declarative_hc_generatedStrongFormation
    {d : Graph} :
    (ValidDLDS d ∧ HCRepresentationWellFormed d ∧
      rootsDischargedB d = true) ↔
    (DeclarativeValidityComponents d ∧ HCRepresentationWellFormed d ∧
      StrongFormationValid d (generateCertificate d)) := by
  constructor
  · rintro ⟨valid, representation, roots⟩
    exact ⟨validDLDS_iff_declarativeValidityComponents.mp valid,
      representation,
      validDLDS_to_generatedStrongFormation valid representation roots⟩
  · rintro ⟨components, representation, strong⟩
    exact ⟨validDLDS_iff_declarativeValidityComponents.mpr components,
      representation,
      strongFormationValid_rootsDischargedB d (generateCertificate d) strong⟩

namespace Counterexamples

/-! The declarative rule checker reads dependency labels but does not interpret
an edge colour as an executable residual-route demand.  Recolouring the sole
edge of the closed identity derivation therefore preserves `ValidDLDS`, while
raw Flow still starts with the default empty residual (head colour zero). -/

def routeColourMismatch : Graph :=
  let hypothesis := Vertex.node 1 1 (#"A") true false []
  let root := Vertex.node 2 0 (#"A" >> #"A") false false []
  Graph.dlds [hypothesis, root]
    [Deduction.edge hypothesis root 1 [#"A"]] []

theorem routeColourMismatch_valid : ValidDLDS routeColourMismatch := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    · intro node₁ node₂ member₁ member₂
      fin_cases member₁ <;> fin_cases member₂ <;> native_decide
    · intro edge member
      fin_cases member
      native_decide
    · intro path member
      simp [routeColourMismatch] at member
  · unfold LeveledColored
    native_decide
  · unfold Simplicity
    native_decide
  · unfold AncestorSimplicity
    native_decide
  · unfold HypothesesHaveNoIncoming
    native_decide
  · unfold LocalRuleCorrect
    native_decide
  · unfold RootDischarged
    intro root member terminal
    fin_cases member
    · have outgoing : get_rule.outgoing
          (Vertex.node 1 1 (#"A") true false []) routeColourMismatch ≠ [] := by
        native_decide
      exact (outgoing terminal).elim
    · constructor <;> native_decide
  · unfold ColorAcyclicity
    native_decide
  · unfold AncestorEdges
    native_decide
  · intro path member
    simp [routeColourMismatch] at member
  · intro path₁ member₁ path₂ member₂
    simp [routeColourMismatch] at member₁

theorem routeColourMismatch_routeHeadCoverageB_false :
    cursorRouteHeadCoverageB routeColourMismatch = false := by
  native_decide

theorem routeColourMismatch_not_routeHeadCoverage :
    ¬ CursorRouteHeadCoverage routeColourMismatch := by
  intro coverage
  have checked := cursorRouteHeadCoverageB_complete coverage
  rw [routeColourMismatch_routeHeadCoverageB_false] at checked
  simp at checked

theorem routeColourMismatch_rootsDischargedB_false :
    rootsDischargedB routeColourMismatch = false := by
  native_decide

theorem validDLDS_does_not_imply_routeHeadCoverage :
    ∃ d, ValidDLDS d ∧ ¬ CursorRouteHeadCoverage d :=
  ⟨routeColourMismatch, routeColourMismatch_valid,
    routeColourMismatch_not_routeHeadCoverage⟩

theorem routeColourMismatch_not_suitableGraph :
    ¬ SuitableGraph routeColourMismatch := by
  intro suitable
  exact routeColourMismatch_not_routeHeadCoverage suitable.routeHeadCoverage

theorem validDLDS_does_not_imply_suitableGraph :
    ∃ d, ValidDLDS d ∧ ¬ SuitableGraph d :=
  ⟨routeColourMismatch, routeColourMismatch_valid,
    routeColourMismatch_not_suitableGraph⟩

theorem routeColourMismatch_not_flowCompiledGridValid :
    ¬ FlowCompiledGridValid routeColourMismatch := by
  intro grid
  have roots := (rootsDischargedB_iff_flowCompiledGridValid
    routeColourMismatch).mpr grid
  rw [routeColourMismatch_rootsDischargedB_false] at roots
  simp at roots

theorem routeColourMismatch_no_strongCertificate :
    ¬ ∃ certificate, StrongFormationValid routeColourMismatch certificate := by
  rintro ⟨certificate, valid⟩
  have roots := strongFormationValid_rootsDischargedB
    routeColourMismatch certificate valid
  rw [routeColourMismatch_rootsDischargedB_false] at roots
  simp at roots

theorem validDLDS_does_not_imply_rawFlowRootDischarge :
    ∃ d, ValidDLDS d ∧ rootsDischargedB d = false :=
  ⟨routeColourMismatch, routeColourMismatch_valid,
    routeColourMismatch_rootsDischargedB_false⟩

theorem rootDecoration_does_not_imply_rawFlowRootDischarge :
    ∃ d, RootDischarged d ∧ rootsDischargedB d = false :=
  ⟨routeColourMismatch, routeColourMismatch_valid.rootDischarged,
    routeColourMismatch_rootsDischargedB_false⟩

/-! `Simplicity` keys an edge by source, target, and colour.  The executable
format needs the stronger source-and-colour key, independently of target. -/

def sameSourceColourFanout : Graph :=
  let h := Vertex.node 1 2 (#"A") true false []
  let root₁ := Vertex.node 2 1 (#"A" >> #"A") false false []
  let middle := Vertex.node 3 1 (#"B" >> #"A") false false []
  let root₂ := Vertex.node 4 0 (#"A" >> (#"B" >> #"A")) false false []
  Graph.dlds [h, root₁, middle, root₂]
    [Deduction.edge h root₁ 0 [#"A"],
     Deduction.edge h middle 0 [#"A"],
     Deduction.edge middle root₂ 0 [#"A"]] []

theorem sameSourceColourFanout_valid : ValidDLDS sameSourceColourFanout := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    · intro node₁ node₂ member₁ member₂
      fin_cases member₁ <;> fin_cases member₂ <;> native_decide
    · intro edge member
      fin_cases member <;> native_decide
    · intro path member
      simp [sameSourceColourFanout] at member
  · unfold LeveledColored
    native_decide
  · unfold Simplicity
    native_decide
  · unfold AncestorSimplicity
    native_decide
  · unfold HypothesesHaveNoIncoming
    native_decide
  · unfold LocalRuleCorrect
    native_decide
  · unfold RootDischarged
    intro root member terminal
    fin_cases member
    · have outgoing : get_rule.outgoing
          (Vertex.node 1 2 (#"A") true false []) sameSourceColourFanout ≠ [] := by
        native_decide
      exact (outgoing terminal).elim
    · constructor <;> native_decide
    · have outgoing : get_rule.outgoing
          (Vertex.node 3 1 (#"B" >> #"A") false false [])
            sameSourceColourFanout ≠ [] := by
        native_decide
      exact (outgoing terminal).elim
    · constructor <;> native_decide
  · unfold ColorAcyclicity
    native_decide
  · unfold AncestorEdges
    native_decide
  · intro path member
    simp [sameSourceColourFanout] at member
  · intro path₁ member₁ path₂ member₂
    simp [sameSourceColourFanout] at member₁

theorem sameSourceColourFanout_not_oneEdgePerColour :
    ¬ OneEdgePerColourPerNode sameSourceColourFanout := by
  intro unique
  let h := Vertex.node 1 2 (#"A") true false []
  let root₁ := Vertex.node 2 1 (#"A" >> #"A") false false []
  let middle := Vertex.node 3 1 (#"B" >> #"A") false false []
  let edge₁ := Deduction.edge h root₁ 0 [#"A"]
  let edge₂ := Deduction.edge h middle 0 [#"A"]
  have equal := unique edge₁ edge₂ (by native_decide) (by native_decide)
    (by exact ⟨rfl, rfl⟩)
  have endpoints : root₁ = middle := congrArg Deduction.END equal
  exact (by native_decide : root₁ ≠ middle) endpoints

theorem validDLDS_does_not_imply_oneEdgePerColour :
    ∃ d, ValidDLDS d ∧ ¬ OneEdgePerColourPerNode d :=
  ⟨sameSourceColourFanout, sameSourceColourFanout_valid,
    sameSourceColourFanout_not_oneEdgePerColour⟩

/-! Distinct derivation occurrences may legitimately share both a level and a
formula.  `LevelFormulaUnique` is therefore an HC quotient/construction
invariant, not generic semantic validity. -/

def duplicateLevelFormulaOccurrences : Graph :=
  let h₁ := Vertex.node 1 1 (#"A") true false []
  let h₂ := Vertex.node 2 1 (#"A") true false []
  let root₁ := Vertex.node 3 0 (#"A" >> #"A") false false []
  let root₂ := Vertex.node 4 0 (#"A" >> #"A") false false []
  Graph.dlds [h₁, h₂, root₁, root₂]
    [Deduction.edge h₁ root₁ 0 [#"A"],
     Deduction.edge h₂ root₂ 0 [#"A"]] []

theorem duplicateLevelFormulaOccurrences_valid :
    ValidDLDS duplicateLevelFormulaOccurrences := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    · intro node₁ node₂ member₁ member₂
      fin_cases member₁ <;> fin_cases member₂ <;> native_decide
    · intro edge member
      fin_cases member <;> native_decide
    · intro path member
      simp [duplicateLevelFormulaOccurrences] at member
  · unfold LeveledColored
    native_decide
  · unfold Simplicity
    native_decide
  · unfold AncestorSimplicity
    native_decide
  · unfold HypothesesHaveNoIncoming
    native_decide
  · unfold LocalRuleCorrect
    native_decide
  · unfold RootDischarged
    intro root member terminal
    fin_cases member
    · have outgoing : get_rule.outgoing
          (Vertex.node 1 1 (#"A") true false [])
            duplicateLevelFormulaOccurrences ≠ [] := by
        native_decide
      exact (outgoing terminal).elim
    · have outgoing : get_rule.outgoing
          (Vertex.node 2 1 (#"A") true false [])
            duplicateLevelFormulaOccurrences ≠ [] := by
        native_decide
      exact (outgoing terminal).elim
    · constructor <;> native_decide
    · constructor <;> native_decide
  · unfold ColorAcyclicity
    native_decide
  · unfold AncestorEdges
    native_decide
  · intro path member
    simp [duplicateLevelFormulaOccurrences] at member
  · intro path₁ member₁ path₂ member₂
    simp [duplicateLevelFormulaOccurrences] at member₁

theorem duplicateLevelFormulaOccurrences_not_levelFormulaUnique :
    ¬ LevelFormulaUnique duplicateLevelFormulaOccurrences := by
  intro unique
  let h₁ := Vertex.node 1 1 (#"A") true false []
  let h₂ := Vertex.node 2 1 (#"A") true false []
  have equal := unique h₁ (by native_decide) h₂ (by native_decide) rfl rfl
  exact (by native_decide : h₁ ≠ h₂) equal

theorem validDLDS_does_not_imply_levelFormulaUnique :
    ∃ d, ValidDLDS d ∧ ¬ LevelFormulaUnique d :=
  ⟨duplicateLevelFormulaOccurrences, duplicateLevelFormulaOccurrences_valid,
    duplicateLevelFormulaOccurrences_not_levelFormulaUnique⟩

namespace TailMismatch

def A : Formula := #"A"
def C : Formula := #"C"
def AC : Formula := A >> C

def minor : Vertex := Vertex.node 1 4 A true false []
def major : Vertex := Vertex.node 2 4 AC true false []
def kernel : Vertex := Vertex.node 3 3 C false false []
def left : Vertex := Vertex.node 4 2 AC false false []
def right : Vertex := Vertex.node 5 2 (AC >> C) false false []
def leftRoot : Vertex := Vertex.node 6 1 (AC >> AC) false false []
def rightRoot : Vertex := Vertex.node 7 1 (A >> (AC >> C)) false false []

/-- Both premise sources have routes with tails `[]` and `[1]`, but only the
minor source has the additional route with consumed tail `[1, 0]`. -/
def graph : Graph :=
  Graph.dlds [minor, major, kernel, left, right, leftRoot, rightRoot]
    [Deduction.edge minor kernel 0 [A],
     Deduction.edge major kernel 0 [AC],
     Deduction.edge kernel left 0 [A, AC],
     Deduction.edge kernel right 1 [A, AC],
     Deduction.edge left leftRoot 0 [AC],
     Deduction.edge right rightRoot 0 [A]]
    [Ancestral.path kernel minor [0],
     Ancestral.path right minor [0, 1],
     Ancestral.path rightRoot minor [0, 1, 0],
     Ancestral.path kernel major [0],
     Ancestral.path right major [0, 1]]

end TailMismatch

theorem tailMismatch_valid : ValidDLDS DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_⟩
    · intro node₁ node₂ member₁ member₂
      fin_cases member₁ <;> fin_cases member₂ <;> native_decide
    · intro edge member
      fin_cases member <;> native_decide
    · intro path member
      fin_cases member <;> native_decide
  · unfold LeveledColored
    native_decide
  · unfold Simplicity
    native_decide
  · unfold AncestorSimplicity
    native_decide
  · unfold HypothesesHaveNoIncoming
    native_decide
  · unfold LocalRuleCorrect
    native_decide
  · unfold RootDischarged
    intro root member
    fin_cases member <;> native_decide
  · unfold ColorAcyclicity
    native_decide
  · unfold AncestorEdges
    native_decide
  · intro path member
    fin_cases member <;> native_decide
  · intro path₁ member₁ path₂ member₂
    fin_cases member₁ <;> fin_cases member₂ <;> native_decide

theorem tailMismatch_oneEdgePerColour :
    OneEdgePerColourPerNode DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph := by
  intro edge₁ edge₂ member₁ member₂ sameKey
  fin_cases member₁ <;> fin_cases member₂ <;>
    simp_all [DLDS.ValidityDecomposition.Counterexamples.TailMismatch.minor, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.major, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.kernel, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.left, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.right, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.leftRoot, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.rightRoot]

theorem tailMismatch_routeHeadCoverage :
    CursorRouteHeadCoverage DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph :=
  cursorRouteHeadCoverageB_sound DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph (by native_decide)

theorem tailMismatch_levelFormulaUnique :
    LevelFormulaUnique DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph := by
  intro node₁ member₁ node₂ member₂ sameLevel sameFormula
  fin_cases member₁ <;> fin_cases member₂ <;>
    simp_all [DLDS.ValidityDecomposition.Counterexamples.TailMismatch.minor, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.major, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.kernel, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.left, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.right, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.leftRoot, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.rightRoot, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.A, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.C, DLDS.ValidityDecomposition.Counterexamples.TailMismatch.AC]

theorem tailMismatch_flowTailClosureB_false :
    cursorFlowTailClosureB DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph = false := by
  native_decide

theorem tailMismatch_not_flowTailClosure :
    ¬ CursorFlowTailClosure DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph := by
  intro closure
  have checked := cursorFlowTailClosureB_complete closure
  rw [tailMismatch_flowTailClosureB_false] at checked
  simp at checked

theorem tailMismatch_rootsDischargedB_true :
    rootsDischargedB DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph = true := by
  native_decide

theorem tailMismatch_flowCorrectB_true :
    flowCorrectB DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph = true := by
  native_decide

theorem validDLDS_does_not_imply_flowTailClosure :
    ∃ d, ValidDLDS d ∧ ¬ CursorFlowTailClosure d :=
  ⟨DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph, tailMismatch_valid,
    tailMismatch_not_flowTailClosure⟩

theorem flowCorrectB_does_not_imply_flowTailClosure :
    ∃ d, flowCorrectB d = true ∧ ¬ CursorFlowTailClosure d :=
  ⟨DLDS.ValidityDecomposition.Counterexamples.TailMismatch.graph, tailMismatch_flowCorrectB_true,
    tailMismatch_not_flowTailClosure⟩

/-! Raw Flow deliberately recomputes dependencies and ignores an edge's stored
dependency payload.  Thus raw root discharge can hold even when the
declarative `outDep` calculation, which reads that payload, rejects the root. -/

def wrongStoredDependency : Graph :=
  let hypothesis := Vertex.node 1 1 (#"A") true false []
  let root := Vertex.node 2 0 (#"A" >> #"A") false false []
  Graph.dlds [hypothesis, root]
    [Deduction.edge hypothesis root 0 [#"B"]] []

theorem wrongStoredDependency_rootsDischargedB_true :
    rootsDischargedB wrongStoredDependency = true := by
  native_decide

theorem wrongStoredDependency_not_rootDischarged :
    ¬ RootDischarged wrongStoredDependency := by
  intro decoration
  let root := Vertex.node 2 0 (#"A" >> #"A") false false []
  have atRoot := decoration root (by native_decide) (by native_decide)
  have wrong : outDep root wrongStoredDependency ≠ [] := by native_decide
  exact wrong atRoot.2

theorem rawFlowRootDischarge_does_not_imply_rootDecoration :
    ∃ d, rootsDischargedB d = true ∧ ¬ RootDischarged d :=
  ⟨wrongStoredDependency, wrongStoredDependency_rootsDischargedB_true,
    wrongStoredDependency_not_rootDischarged⟩

/-! The converse mismatch is structural rather than semantic.  The executable
`SuitableGraph` package deliberately ignores detached ancestral records that
cannot participate in Flow.  Adding such a record to the closed identity
graph therefore preserves every executable invariant and raw root discharge,
while violating the endpoint-hygiene component of `ValidDLDS`. -/

def suitableWithDetachedBadPath : Graph :=
  let hypothesis := Vertex.node 1 1 (#"A") true false []
  let root := Vertex.node 2 0 (#"A" >> #"A") false false []
  let ghostStart := Vertex.node 3 0 (#"X") false false []
  let ghostEnd := Vertex.node 4 1 (#"Y") false false []
  Graph.dlds [hypothesis, root]
    [Deduction.edge hypothesis root 0 [#"A"]]
    [Ancestral.path ghostStart ghostEnd []]

theorem suitableWithDetachedBadPath_suitable :
    SuitableGraph suitableWithDetachedBadPath := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact DLDS.structuralValidB_sound suitableWithDetachedBadPath
      (by native_decide)
  · intro vertex member
    fin_cases member <;>
      apply ruleShapeOK_implies_localShape_valid <;> native_decide
  · native_decide
  · exact DLDS.oneEdgeB_sound suitableWithDetachedBadPath
      (by native_decide)
  · exact cursorRouteHeadCoverageB_sound suitableWithDetachedBadPath
      (by native_decide)
  · exact cursorFlowTailClosureB_sound suitableWithDetachedBadPath
      (by native_decide)
  · exact DLDS.levelFormulaUniqueB_sound suitableWithDetachedBadPath
      (by native_decide)
  · exact DLDS.hypNoIncomingB_sound suitableWithDetachedBadPath
      (by native_decide)
  · unfold RootDischarged
    intro vertex member terminal
    fin_cases member
    · have outgoing : get_rule.outgoing
          (Vertex.node 1 1 (#"A") true false [])
          suitableWithDetachedBadPath ≠ [] := by
        native_decide
      exact (outgoing terminal).elim
    · constructor <;> native_decide

theorem suitableWithDetachedBadPath_rootsDischargedB_true :
    rootsDischargedB suitableWithDetachedBadPath = true := by
  native_decide

theorem suitableWithDetachedBadPath_not_valid :
    ¬ ValidDLDS suitableWithDetachedBadPath := by
  intro valid
  let ghostStart := Vertex.node 3 0 (#"X") false false []
  let ghostEnd := Vertex.node 4 1 (#"Y") false false []
  let badPath := Ancestral.path ghostStart ghostEnd []
  have pathMember : badPath ∈ suitableWithDetachedBadPath.PATHS := by
    native_decide
  have endpoints := valid.hygiene.2.2 pathMember
  have ghostStartMember : ghostStart ∈ suitableWithDetachedBadPath.NODES :=
    endpoints.1
  have ghostStartAbsent : ghostStart ∉ suitableWithDetachedBadPath.NODES := by
    native_decide
  exact ghostStartAbsent ghostStartMember

theorem suitableGraph_does_not_imply_validDLDS :
    ∃ d, SuitableGraph d ∧ ¬ ValidDLDS d :=
  ⟨suitableWithDetachedBadPath, suitableWithDetachedBadPath_suitable,
    suitableWithDetachedBadPath_not_valid⟩

theorem suitableGraph_and_rawRootDischarge_do_not_imply_validDLDS :
    ∃ d, SuitableGraph d ∧ rootsDischargedB d = true ∧ ¬ ValidDLDS d :=
  ⟨suitableWithDetachedBadPath, suitableWithDetachedBadPath_suitable,
    suitableWithDetachedBadPath_rootsDischargedB_true,
    suitableWithDetachedBadPath_not_valid⟩

/-! `flowCorrectB` checks Flow bullets only at non-source, non-root nodes and
then checks roots.  It does not independently require every source edge to be
the head continuation of a source route. -/

def flowCorrectWithSpuriousSourceEdge : Graph :=
  let hA := Vertex.node 1 3 (#"A") true false []
  let hBC := Vertex.node 2 3 (#"B" >> #"C") true false []
  let hB := Vertex.node 3 3 (#"B") true false []
  let rootA := Vertex.node 4 2 (#"A" >> #"A") false false []
  let kernel := Vertex.node 5 2 (#"C") false false []
  let closeB := Vertex.node 6 1 (#"B" >> #"C") false false []
  let rootC := Vertex.node 7 0
    ((#"B" >> #"C") >> (#"B" >> #"C")) false false []
  Graph.dlds [hA, hBC, hB, rootA, kernel, closeB, rootC]
    [Deduction.edge hA rootA 0 [#"A"],
     Deduction.edge hA kernel 1 [#"A"],
     Deduction.edge hBC kernel 0 [#"B" >> #"C"],
     Deduction.edge hB kernel 0 [#"B"],
     Deduction.edge kernel closeB 0 [#"B", #"B" >> #"C"],
     Deduction.edge closeB rootC 0 [#"B" >> #"C"]] []

theorem flowCorrectWithSpuriousSourceEdge_true :
    flowCorrectB flowCorrectWithSpuriousSourceEdge = true := by
  native_decide

theorem flowCorrectWithSpuriousSourceEdge_coverageB_false :
    cursorRouteHeadCoverageB flowCorrectWithSpuriousSourceEdge = false := by
  native_decide

theorem flowCorrectWithSpuriousSourceEdge_not_coverage :
    ¬ CursorRouteHeadCoverage flowCorrectWithSpuriousSourceEdge := by
  intro coverage
  have checked := cursorRouteHeadCoverageB_complete coverage
  rw [flowCorrectWithSpuriousSourceEdge_coverageB_false] at checked
  simp at checked

theorem flowCorrectB_does_not_imply_routeHeadCoverage :
    ∃ d, flowCorrectB d = true ∧ ¬ CursorRouteHeadCoverage d :=
  ⟨flowCorrectWithSpuriousSourceEdge,
    flowCorrectWithSpuriousSourceEdge_true,
    flowCorrectWithSpuriousSourceEdge_not_coverage⟩

theorem flowCorrectB_implies_rootsDischargedB {d : Graph}
    (correct : flowCorrectB d = true) : rootsDischargedB d = true := by
  unfold flowCorrectB at correct
  rw [Bool.and_eq_true] at correct
  exact correct.2

theorem validDLDS_does_not_imply_flowCorrectB :
    ∃ d, ValidDLDS d ∧ flowCorrectB d = false := by
  refine ⟨routeColourMismatch, routeColourMismatch_valid, ?_⟩
  native_decide

end Counterexamples


end DLDS.ValidityDecomposition
