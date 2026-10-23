import Infrastructure.CertificateModel









open Semantic FlowSpec
open CertificateModel

namespace SelfCertifying

abbrev Certificate := ExpandedCertificate

def StoredRoutes (c : Certificate) (v : Vertex) : List FlowPair :=
  outputsAt c v

def ListSetEq {α : Type} [BEq α] (xs ys : List α) : Prop :=
  ∀ x, x ∈ xs ↔ x ∈ ys

lemma ListSetEq.refl {α : Type} [BEq α] (xs : List α) : ListSetEq xs xs := by
  intro x
  rfl

lemma ListSetEq.symm {α : Type} [BEq α] {xs ys : List α}
    (h : ListSetEq xs ys) : ListSetEq ys xs := by
  intro x
  exact (h x).symm

lemma ListSetEq.trans {α : Type} [BEq α] {xs ys zs : List α}
    (hxy : ListSetEq xs ys) (hyz : ListSetEq ys zs) : ListSetEq xs zs := by
  intro x
  exact (hxy x).trans (hyz x)

lemma flatMap_congr_list {α β : Type} [BEq α] [BEq β]
    {l l' : List α} (f : α → List β) (h : ListSetEq l l') :
    ListSetEq (l.flatMap f) (l'.flatMap f) := by
  intro x
  simp only [List.mem_flatMap]
  constructor
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, (h a).mp ha, hx⟩
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, (h a).mpr ha, hx⟩

lemma flatMap_congr_fun {α β : Type} [BEq α] [BEq β]
    (l : List α) {f g : α → List β}
    (h : ∀ a, ListSetEq (f a) (g a)) :
    ListSetEq (l.flatMap f) (l.flatMap g) := by
  intro x
  simp only [List.mem_flatMap]
  constructor
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, ha, (h a x).mp hx⟩
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, ha, (h a x).mpr hx⟩

lemma flatMap_congr_fun_mem {α β : Type} [BEq α] [BEq β]
    (l : List α) {f g : α → List β}
    (h : ∀ a ∈ l, ListSetEq (f a) (g a)) :
    ListSetEq (l.flatMap f) (l.flatMap g) := by
  intro x
  simp only [List.mem_flatMap]
  constructor
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, ha, (h a ha x).mp hx⟩
  · rintro ⟨a, ha, hx⟩
    exact ⟨a, ha, (h a ha x).mpr hx⟩

lemma eraseDups_congr {α : Type} [BEq α] [LawfulBEq α]
    {xs ys : List α} (h : ListSetEq xs ys) :
  ListSetEq xs.eraseDups ys.eraseDups := by
  intro x
  rw [List.mem_eraseDups, List.mem_eraseDups]
  exact h x

lemma witnessableRoutes_congr (d : Graph) (v : Vertex)
    {xs ys : List FlowPair} (h : ListSetEq xs ys) :
    ListSetEq (WitnessableRoutes d v xs) (WitnessableRoutes d v ys) := by
  intro route
  unfold WitnessableRoutes
  simp only [List.mem_filter]
  exact and_congr (h route) Iff.rfl



lemma elimCombine_project_inputs (d : Graph) (v M S : Vertex)
    (FS FM : List FlowPair) :
    ListSetEq
      (elimCombine d v M S (WitnessableRoutes d S FS)
        (WitnessableRoutes d M FM))
      (elimCombine d v M S FS FM) := by
  intro result
  unfold elimCombine WitnessableRoutes
  simp only [List.mem_flatMap, List.mem_filter]
  constructor
  · rintro ⟨minor, ⟨hminor, _⟩, major, ⟨hmajor, _⟩, hrest⟩
    exact ⟨minor, hminor, major, hmajor, hrest⟩
  · rintro ⟨minor, hminor, major, hmajor, edgeMinor, hedgeMinor, hlast⟩
    rw [List.mem_filterMap] at hlast
    obtain ⟨edgeMajor, hedgeMajor, hresult⟩ := hlast
    cases hcMinor : consume edgeMinor minor.2 with
    | none => simp [hcMinor] at hresult
    | some tailMinor =>
      cases hcMajor : consume edgeMajor major.2 with
      | none => simp [hcMinor, hcMajor] at hresult
      | some tailMajor =>
        refine ⟨minor, ⟨hminor, ?_⟩, major, ⟨hmajor, ?_⟩,
          edgeMinor, hedgeMinor, ?_⟩
        · exact routeContinuableAtB_of_consume_edgesBetween d S v edgeMinor
            minor.2 tailMinor hedgeMinor hcMinor
        · exact routeContinuableAtB_of_consume_edgesBetween d M v edgeMajor
            major.2 tailMajor hedgeMajor hcMajor
        · rw [List.mem_filterMap]
          exact ⟨edgeMajor, hedgeMajor, hresult⟩

lemma introProjectInputsRaw (d : Graph) (v premise : Vertex) (alpha : Formula)
    (F : List FlowPair) :
    ListSetEq
      ((WitnessableRoutes d premise F).flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail))
      (F.flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)) := by
  intro result
  unfold WitnessableRoutes
  simp only [List.mem_flatMap, List.mem_filter]
  constructor
  · rintro ⟨route, ⟨hroute, _⟩, hresult⟩
    exact ⟨route, hroute, hresult⟩
  · rintro ⟨route, hroute, hlast⟩
    rw [List.mem_filterMap] at hlast
    obtain ⟨edge, hedge, hresult⟩ := hlast
    cases hc : consume edge route.2 with
    | none => simp [hc] at hresult
    | some tail =>
      refine ⟨route, ⟨hroute, ?_⟩, ?_⟩
      · exact routeContinuableAtB_of_consume_edgesBetween d premise v edge
          route.2 tail hedge hc
      · rw [List.mem_filterMap]
        exact ⟨edge, hedge, hresult⟩

lemma introProjectInputs (d : Graph) (v premise : Vertex) (alpha : Formula)
    (F : List FlowPair) :
    ListSetEq
      (((WitnessableRoutes d premise F).flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail =>
            (depRemove route.1 alpha, tail)).eraseDups)
      ((F.flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail =>
            (depRemove route.1 alpha, tail)).eraseDups) :=
  eraseDups_congr (introProjectInputsRaw d v premise alpha F)

lemma mixedIntroProjectInputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) :
    ListSetEq
      (mixedIntroPairs d v (fun u => WitnessableRoutes d u (F u)))
      (mixedIntroPairs d v F) := by
  unfold mixedIntroPairs
  cases hformula : v.FORMULA with
  | atom => exact ListSetEq.refl _
  | implication alpha beta =>
      apply flatMap_congr_fun_mem
      intro premise hpremise
      split
      · exact introProjectInputsRaw d v premise alpha (F premise)
      · exact ListSetEq.refl _

lemma routeContinuable_of_reseedAgree (d : Graph) (start finish : Vertex)
    (route : FlowPair) (edge : Deduction)
    (hedge : edge ∈ edgesBetween d start finish)
    (hagrees : reseedPremiseAgreesB route edge = true) :
    routeContinuableAtB d start route.2 = true := by
  unfold reseedPremiseAgreesB at hagrees
  rw [Option.isSome_iff_exists] at hagrees
  obtain ⟨tail, hconsume⟩ := hagrees
  exact routeContinuableAtB_of_consume_edgesBetween d start finish edge
    route.2 tail hedge hconsume

lemma elimReseedProjectInputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) (ancestors : List Ancestral) :
    ListSetEq
      (elimReseedPairs d v (fun u => WitnessableRoutes d u (F u)) ancestors)
      (elimReseedPairs d v F ancestors) := by
  intro result
  unfold elimReseedPairs WitnessableRoutes
  simp only [List.mem_flatMap, List.mem_filter]
  constructor
  · rintro ⟨pair, hpair, minorRoute, ⟨hminor, _⟩,
      majorRoute, ⟨hmajor, _⟩, hrest⟩
    exact ⟨pair, hpair, minorRoute, hminor, majorRoute, hmajor, hrest⟩
  · rintro ⟨pair, hpair, minorRoute, hminor, majorRoute, hmajor,
      minorEdge, hminorEdge, majorEdge, hmajorEdge, hlast⟩
    rcases pair with ⟨major, minor⟩
    split at hlast
    next hcondition =>
      simp only [Bool.and_eq_true] at hcondition
      refine ⟨(major, minor), hpair, minorRoute, ⟨hminor, ?_⟩,
        majorRoute, ⟨hmajor, ?_⟩, minorEdge, hminorEdge,
        majorEdge, hmajorEdge, ?_⟩
      · exact routeContinuable_of_reseedAgree d minor v minorRoute minorEdge
          hminorEdge hcondition.1.1.1
      · exact routeContinuable_of_reseedAgree d major v majorRoute majorEdge
          hmajorEdge hcondition.1.1.2
      · simpa [hcondition] using hlast
    next hcondition => simp at hlast

lemma introReseedProjectInputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) (ancestors : List Ancestral) :
    ListSetEq
      (introReseedPairs d v (fun u => WitnessableRoutes d u (F u)) ancestors)
      (introReseedPairs d v F ancestors) := by
  unfold introReseedPairs
  cases hp : predsOf d v with
  | nil => exact ListSetEq.refl _
  | cons premise rest =>
      cases rest with
      | cons premise' rest => exact ListSetEq.refl _
      | nil =>
        cases hformula : v.FORMULA with
        | atom => exact ListSetEq.refl _
        | implication alpha beta =>
          by_cases hbeta : (premise.FORMULA == beta) = true
          · simp only [hbeta, if_true]
            intro result
            unfold WitnessableRoutes
            simp only [List.mem_flatMap, List.mem_filter]
            constructor
            · rintro ⟨route, ⟨hroute, _⟩, hrest⟩
              exact ⟨route, hroute, hrest⟩
            · rintro ⟨route, hroute, edge, hedge, hlast⟩
              split at hlast
              next hcondition =>
                simp only [Bool.and_eq_true] at hcondition
                refine ⟨route, ⟨hroute, ?_⟩, edge, hedge, ?_⟩
                · unfold reseedZeroPremiseB at hcondition
                  simp only [Bool.and_eq_true] at hcondition
                  exact routeContinuable_of_reseedAgree d premise v route edge
                    hedge hcondition.1.2
                · simpa [hcondition] using hlast
              next hcondition => simp at hlast
          · have hbeta' : (premise.FORMULA == beta) = false :=
              Bool.eq_false_of_not_eq_true hbeta
            simp only [hbeta', Bool.false_eq_true, if_false]
            exact ListSetEq.refl []

lemma reseedProjectInputs (d : Graph) (v : Vertex)
    (F : Vertex → List FlowPair) :
    ListSetEq
      (reseedPairs d v (fun u => WitnessableRoutes d u (F u)))
      (reseedPairs d v F) := by
  unfold reseedPairs
  generalize hanc : eligibleAncestorsAtRoot d v = ancestors
  cases hempty : ancestors.isEmpty with
  | true =>
      simp only [hempty, if_true]
      exact ListSetEq.refl []
  | false =>
      simp only [hempty, Bool.false_eq_true, if_false]
      apply eraseDups_congr
      intro result
      simp only [List.mem_append]
      exact or_congr
        (elimReseedProjectInputs d v F ancestors result)
        (introReseedProjectInputs d v F ancestors result)

lemma elimCombine_congr (d : Graph) (v major minor : Vertex)
    {FS FS' FM FM' : List FlowPair}
    (hS : ListSetEq FS FS') (hM : ListSetEq FM FM') :
    ListSetEq (elimCombine d v major minor FS FM).eraseDups
      (elimCombine d v major minor FS' FM').eraseDups := by
  unfold elimCombine
  apply eraseDups_congr
  apply ListSetEq.trans (flatMap_congr_list _ hS)
  apply flatMap_congr_fun FS'
  intro minorRoute
  exact flatMap_congr_list _ hM

lemma elimCombine_congr_raw (d : Graph) (v major minor : Vertex)
    {FS FS' FM FM' : List FlowPair}
    (hS : ListSetEq FS FS') (hM : ListSetEq FM FM') :
    ListSetEq (elimCombine d v major minor FS FM)
      (elimCombine d v major minor FS' FM') := by
  unfold elimCombine
  apply ListSetEq.trans (flatMap_congr_list _ hS)
  apply flatMap_congr_fun FS'
  intro minorRoute
  exact flatMap_congr_list _ hM

lemma introCombine_congr (d : Graph) (v premise : Vertex) (alpha : Formula)
    {F F' : List FlowPair} (hF : ListSetEq F F') :
    ListSetEq
      ((F.flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail =>
            (depRemove route.1 alpha, tail)).eraseDups)
      ((F'.flatMap fun route =>
        (edgesBetween d premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail =>
            (depRemove route.1 alpha, tail)).eraseDups) := by
  exact eraseDups_congr (flatMap_congr_list _ hF)

lemma append_congr {α : Type} [BEq α] {xs xs' ys ys' : List α}
    (hx : ListSetEq xs xs') (hy : ListSetEq ys ys') :
    ListSetEq (xs ++ ys) (xs' ++ ys') := by
  intro x
  simp only [List.mem_append]
  exact or_congr (hx x) (hy x)

lemma elimReseedPairs_congr_set_of_preds (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair) (ancestors : List Ancestral)
    (hflow : ∀ u ∈ predsOf d v, ListSetEq (left u) (right u)) :
    ListSetEq (elimReseedPairs d v left ancestors)
      (elimReseedPairs d v right ancestors) := by
  unfold elimReseedPairs
  apply flatMap_congr_fun_mem
  intro pair hpair
  rcases pair with ⟨major, minor⟩
  obtain ⟨hmajor, hminor⟩ := mem_preds_of_mem_elimPairsAt hpair
  apply ListSetEq.trans (flatMap_congr_list _ (hflow minor hminor))
  apply flatMap_congr_fun (right minor)
  intro minorRoute
  exact flatMap_congr_list _ (hflow major hmajor)

lemma introReseedPairs_congr_set_of_preds (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair) (ancestors : List Ancestral)
    (hflow : ∀ u ∈ predsOf d v, ListSetEq (left u) (right u)) :
    ListSetEq (introReseedPairs d v left ancestors)
      (introReseedPairs d v right ancestors) := by
  unfold introReseedPairs
  cases hp : predsOf d v with
  | nil => exact ListSetEq.refl _
  | cons premise rest =>
      cases rest with
      | nil =>
          have hpremise : premise ∈ predsOf d v := by simp [hp]
          cases hformula : v.FORMULA <;> simp only
          · exact ListSetEq.refl _
          · split
            · exact flatMap_congr_list _ (hflow premise hpremise)
            · exact ListSetEq.refl _
      | cons premise' rest => exact ListSetEq.refl _

lemma reseedPairs_congr_set_of_preds (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hflow : ∀ u ∈ predsOf d v, ListSetEq (left u) (right u)) :
    ListSetEq (reseedPairs d v left) (reseedPairs d v right) := by
  unfold reseedPairs
  generalize hanc : eligibleAncestorsAtRoot d v = ancestors
  cases hempty : ancestors.isEmpty with
  | true =>
      simpa [hempty] using (ListSetEq.refl ([] : List FlowPair))
  | false =>
      simp only [hempty, Bool.false_eq_true, if_false]
      apply eraseDups_congr
      exact append_congr
        (elimReseedPairs_congr_set_of_preds d v left right ancestors hflow)
        (introReseedPairs_congr_set_of_preds d v left right ancestors hflow)

lemma mixedIntroPairs_congr_set_of_preds (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hflow : ∀ u ∈ predsOf d v, ListSetEq (left u) (right u)) :
    ListSetEq (mixedIntroPairs d v left) (mixedIntroPairs d v right) := by
  unfold mixedIntroPairs
  cases hformula : v.FORMULA with
  | atom => exact ListSetEq.refl _
  | implication alpha beta =>
      apply flatMap_congr_fun_mem
      intro premise hpremise
      split
      · exact flatMap_congr_list _ (hflow premise hpremise)
      · exact ListSetEq.refl _

lemma elimExpected_congr_flow (c : Certificate) {v major minor : Vertex}
    (hstruct : structuralValid c.graph)
    (hmajorNode : major ∈ c.graph.NODES)
    (hminorNode : minor ∈ c.graph.NODES)
    (hmajorOut : get_rule.outgoing major c.graph ≠ [])
    (hminorOut : get_rule.outgoing minor c.graph ≠ [])
    (hmajor : ListSetEq (StoredRoutes c major)
      (WitnessableRoutes c.graph major
        (flowAt c.graph (stdFuel c.graph) major)))
    (hminor : ListSetEq (StoredRoutes c minor)
      (WitnessableRoutes c.graph minor
        (flowAt c.graph (stdFuel c.graph) minor))) :
    ListSetEq (elimExpected c v major minor)
      (elimCombine c.graph v major minor
        (flowAt c.graph (stdFuel c.graph - 1) minor)
        (flowAt c.graph (stdFuel c.graph - 1) major)).eraseDups := by
  have hminorStable : ListSetEq (StoredRoutes c minor)
      (WitnessableRoutes c.graph minor
        (flowAt c.graph (stdFuel c.graph - 1) minor)) := by
    apply ListSetEq.trans hminor
    apply witnessableRoutes_congr
    rw [flowAt_stdFuel_pred c.graph hstruct hminorNode hminorOut]
    exact ListSetEq.refl _
  have hmajorStable : ListSetEq (StoredRoutes c major)
      (WitnessableRoutes c.graph major
        (flowAt c.graph (stdFuel c.graph - 1) major)) := by
    apply ListSetEq.trans hmajor
    apply witnessableRoutes_congr
    rw [flowAt_stdFuel_pred c.graph hstruct hmajorNode hmajorOut]
    exact ListSetEq.refl _
  unfold elimExpected
  exact (elimCombine_congr c.graph v major minor hminorStable hmajorStable).trans
    (eraseDups_congr (elimCombine_project_inputs c.graph v major minor
      (flowAt c.graph (stdFuel c.graph - 1) minor)
      (flowAt c.graph (stdFuel c.graph - 1) major)))

lemma introExpected_congr_flow (c : Certificate) {v premise : Vertex}
    (alpha : Formula) (hstruct : structuralValid c.graph)
    (hpremiseNode : premise ∈ c.graph.NODES)
    (hpremiseOut : get_rule.outgoing premise c.graph ≠ [])
    (hpremise : ListSetEq (StoredRoutes c premise)
      (WitnessableRoutes c.graph premise
        (flowAt c.graph (stdFuel c.graph) premise))) :
    ListSetEq (introExpected c v premise alpha)
      (((flowAt c.graph (stdFuel c.graph - 1) premise).flatMap fun route =>
        (edgesBetween c.graph premise v).filterMap fun edge =>
          (consume edge route.2).map fun tail =>
            (depRemove route.1 alpha, tail)).eraseDups) := by
  have hpremiseStable : ListSetEq (StoredRoutes c premise)
      (WitnessableRoutes c.graph premise
        (flowAt c.graph (stdFuel c.graph - 1) premise)) := by
    apply ListSetEq.trans hpremise
    apply witnessableRoutes_congr
    rw [flowAt_stdFuel_pred c.graph hstruct hpremiseNode hpremiseOut]
    exact ListSetEq.refl _
  unfold introExpected
  exact (introCombine_congr c.graph v premise alpha hpremiseStable).trans
    (introProjectInputs c.graph v premise alpha
      (flowAt c.graph (stdFuel c.graph - 1) premise))

def ExpandedCert (d : Graph) (c : Certificate) : Prop :=
  c.graph = d ∧
  (∀ e ∈ d.EDGES, edgeWitnessesFor c e ≠ []) ∧
  (∀ w ∈ c.edgeRoutes, w.edge ∈ d.EDGES) ∧
  (∀ v ∈ d.NODES, (get_rule.outgoing v d).isEmpty = true →
    (c.terminalOutputs.filter fun output => output.node == v) ≠ []) ∧
  (∀ output ∈ c.terminalOutputs,
    output.node ∈ d.NODES ∧
      (get_rule.outgoing output.node d).isEmpty = true)

def expandedCertB (d : Graph) (c : Certificate) : Bool :=
  decide (c.graph = d) &&
  d.EDGES.all (fun e => !(edgeWitnessesFor c e).isEmpty) &&
  c.edgeRoutes.all (fun w => d.EDGES.contains w.edge) &&
  d.NODES.all (fun v =>
    if (get_rule.outgoing v d).isEmpty then
      !(c.terminalOutputs.filter fun output => output.node == v).isEmpty
    else true) &&
  c.terminalOutputs.all (fun output =>
    d.NODES.contains output.node &&
      (get_rule.outgoing output.node d).isEmpty)

def LocalProduction (c : Certificate) : Prop :=
  ∀ v ∈ c.graph.NODES, productionAtB c v = true




def LocalWitnessFaithful (c : Certificate) : Prop :=
  ∀ witness ∈ c.edgeRoutes, witnessFaithfulAtB c witness = true

def LocalCoverage (c : Certificate) : Prop :=
  ∀ v ∈ c.graph.NODES, ∀ e ∈ get_rule.outgoing v c.graph,
    edgeWitnessesFor c e ≠ [] ∧
      ListSetEq
        ((edgeWitnessesFor c e).map fun witness =>
          (witness.dependency, witness.residual))
        ((outputsAt c v).filter fun route =>
          headColour route.2 == e.COLOUR)

def LocalCarrierAccountedAt (c : Certificate) (w u : Vertex) : Prop :=
  ∃ alpha, w.FORMULA = Formula.implication alpha u.FORMULA ∧
    ∀ tail ∈ continuableConsumedWitnessTails c u w,
      ∃ dependency, (dependency, tail) ∈ witnessableExpected c w



def LocalEqualityTailClosure (c : Certificate) : Prop :=
  ∀ w ∈ c.graph.NODES, elimPairsAt c.graph w ≠ [] →
    (∀ u ∈ predsOf c.graph w,
      ElimPremiseAt c.graph w u ∨ LocalCarrierAccountedAt c w u) ∧
    (∀ major minor, (major, minor) ∈ elimPairsAt c.graph w →
      ListSetEq (continuableConsumedWitnessTails c major w)
        (continuableConsumedWitnessTails c minor w))



def LocalTailClosure (c : Certificate) : Prop :=
  ∀ w ∈ c.graph.NODES, elimPairsAt c.graph w ≠ [] →
    (∀ u ∈ predsOf c.graph w,
      ElimPremiseAt c.graph w u ∨ LocalCarrierAccountedAt c w u) ∧
    (∀ major minor, (major, minor) ∈ elimPairsAt c.graph w →
      certificatePairTailClosureAtB c w major minor = true)

def LocalRootDischarge (c : Certificate) : Prop :=
  ∀ v ∈ c.graph.NODES, (get_rule.outgoing v c.graph).isEmpty = true →
    (∃ output, output ∈ c.terminalOutputs ∧ output.node = v ∧
      output.dependency = [] ∧ output.residual = []) ∧
    RuleShapeOK v c.graph ∧ outDep v c.graph = []

def LocalChecks (d : Graph) (c : Certificate) : Prop :=
  c.graph = d ∧
  LocalProduction c ∧ LocalCoverage c ∧
  LocalTailClosure c ∧ LocalRootDischarge c ∧
  structuralValid d ∧ d.EDGES.Nodup ∧ OneEdgePerColourPerNode d ∧
  LocalWitnessFaithful c ∧ LevelFormulaUnique d ∧
  HypothesesHaveNoIncoming d

def localChecksB (d : Graph) (c : Certificate) : Bool :=
  decide (c.graph = d) &&
  d.NODES.all (productionAtB c) &&
  d.NODES.all (coverageAtB c) &&
  d.NODES.all (tailClosureAtB c) &&
  d.NODES.all (rootDischargeAtB c) &&
  structuralValidB d && nodupB d && oneEdgeB d &&
  c.edgeRoutes.all (witnessFaithfulAtB c) &&
  levelFormulaUniqueB d && hypNoIncomingB d




def localChecksCachedB (d : Graph) (c : Certificate)
    (cache : CertificateCache) : Bool :=
  decide (c.graph = d) &&
  d.NODES.all (productionAtCachedB c cache) &&
  d.NODES.all (coverageAtCachedB c cache) &&
  d.NODES.all (tailClosureAtCachedB c cache) &&
  d.NODES.all (rootDischargeAtCachedB c cache) &&
  structuralValidB d && nodupB d && oneEdgeB d &&
  c.edgeRoutes.all (witnessFaithfulAtCachedB c cache) &&
  levelFormulaUniqueB d && hypNoIncomingB d

def localChecksMemoB (d : Graph) (c : Certificate) : Bool :=
  let cache := buildCertificateCache c
  localChecksCachedB d c cache

@[simp] theorem expandedCertCachedB_buildCertificateCache
    (d : Graph) (c : Certificate) :
    expandedCertCachedB d c (buildCertificateCache c) = expandedCertB d c := by
  simp [expandedCertCachedB, expandedCertB]

@[simp] theorem localChecksCachedB_buildCertificateCache
    (d : Graph) (c : Certificate) :
    localChecksCachedB d c (buildCertificateCache c) = localChecksB d c := by
  have hproduction : productionAtCachedB c (buildCertificateCache c) =
      productionAtB c := by
    funext v
    exact productionAtCachedB_buildCertificateCache c v
  have hcoverage : coverageAtCachedB c (buildCertificateCache c) =
      coverageAtB c := by
    funext v
    exact coverageAtCachedB_buildCertificateCache c v
  have htail : tailClosureAtCachedB c (buildCertificateCache c) =
      tailClosureAtB c := by
    funext v
    exact tailClosureAtCachedB_buildCertificateCache c v
  have hroot : rootDischargeAtCachedB c (buildCertificateCache c) =
      rootDischargeAtB c := by
    funext v
    exact rootDischargeAtCachedB_buildCertificateCache c v
  have hfaithful : witnessFaithfulAtCachedB c (buildCertificateCache c) =
      witnessFaithfulAtB c := by
    funext witness
    exact witnessFaithfulAtCachedB_buildCertificateCache c witness
  simp [localChecksCachedB, localChecksB, hproduction, hcoverage, htail,
    hroot, hfaithful]

lemma setEqB_sound {α : Type} [BEq α] [LawfulBEq α]
    {xs ys : List α} (h : setEqB xs ys = true) : ListSetEq xs ys := by
  simp only [setEqB, Bool.and_eq_true, List.all_eq_true,
    List.contains_eq_mem] at h
  intro x
  constructor
  · intro hx
    exact of_decide_eq_true (h.1 x hx)
  · intro hx
    exact of_decide_eq_true (h.2 x hx)

lemma localProduction_of_all (c : Certificate)
    (h : c.graph.NODES.all (productionAtB c) = true) : LocalProduction c := by
  rw [List.all_eq_true] at h
  exact h

lemma localWitnessFaithful_of_all (c : Certificate)
    (h : c.edgeRoutes.all (witnessFaithfulAtB c) = true) :
    LocalWitnessFaithful c := by
  rw [List.all_eq_true] at h
  exact h

lemma expandedCertB_sound (d : Graph) (c : Certificate)
    (h : expandedCertB d c = true) : ExpandedCert d c := by
  unfold expandedCertB at h
  simp only [Bool.and_eq_true, List.all_eq_true] at h
  rcases h with ⟨⟨⟨⟨hgraph, hedges⟩, hextras⟩, hterminals⟩, hterminalExtras⟩
  have hgraph' : c.graph = d := of_decide_eq_true hgraph
  refine ⟨hgraph', ?_, ?_, ?_, ?_⟩
  · intro e he
    have := hedges e he
    simpa using this
  · intro w hw
    have := hextras w hw
    simpa using this
  · intro v hv hterminal
    have := hterminals v hv
    simp [hterminal] at this
    obtain ⟨output, houtput, hnode⟩ := this
    intro hempty
    have hmem : output ∈ c.terminalOutputs.filter
        (fun selection => selection.node == v) := by
      rw [List.mem_filter]
      exact ⟨houtput, by simpa using hnode⟩
    rw [hempty] at hmem
    simp at hmem
  · intro output ho
    have houtput := hterminalExtras output ho
    simp only [] at houtput
    exact ⟨by simpa using houtput.1, houtput.2⟩

lemma productionAtB_setEq (c : Certificate) {v : Vertex}
    (h : productionAtB c v = true) :
    ListSetEq (StoredRoutes c v) (witnessableExpected c v) := by
  unfold productionAtB at h
  cases hs : localShape c.graph v <;> simp [hs] at h
  · exact setEqB_sound h
  · exact setEqB_sound h
  · exact setEqB_sound h
  · exact setEqB_sound h

lemma localShape_source_inv {d : Graph} {v : Vertex}
    (h : localShape d v = .source) :
    (v.HYPOTHESIS || (predsOf d v).isEmpty) = true := by
  unfold localShape at h
  split at h <;> simp_all ; aesop

lemma localShape_elim_raw_inv {d : Graph} {v major minor : Vertex}
    (h : localShape d v = .elim major minor) :
    (v.HYPOTHESIS || (predsOf d v).isEmpty) = false ∧
      elimPairsAt d v = [(major, minor)] := by
  unfold localShape at h
  split at h
  · simp at h
  · split at h
    · simp_all
    · split at h <;> simp_all ; aesop
    · exact False.elim (LocalRuleShape.elimMulti_ne_elim _ _ _ h)

lemma localShape_elim_inv {d : Graph} {v major minor : Vertex}
    (hstruct : structuralValid d)
    (h : localShape d v = .elim major minor) :
    elimPairsAt d v = [(major, minor)] ∧
      major ∈ d.NODES ∧ minor ∈ d.NODES ∧
      major.LEVEL = v.LEVEL + 1 ∧ minor.LEVEL = v.LEVEL + 1 ∧
      get_rule.outgoing major d ≠ [] ∧ get_rule.outgoing minor d ≠ [] := by
  obtain ⟨_, hpairs⟩ := localShape_elim_raw_inv h
  obtain ⟨hMpred, hSpred⟩ := mem_preds_of_elimPairsAt hpairs
  have hMnode := mem_predsOf_mem_nodes hstruct hMpred
  have hSnode := mem_predsOf_mem_nodes hstruct hSpred
  obtain ⟨hMlevel, hMout⟩ := predsOf_props hstruct hMpred
  obtain ⟨hSlevel, hSout⟩ := predsOf_props hstruct hSpred
  exact ⟨hpairs, hMnode, hSnode, hMlevel, hSlevel, hMout, hSout⟩

lemma localShape_elimMulti_raw_inv {d : Graph} {v : Vertex}
    {pairs : List (Vertex × Vertex)}
    (h : localShape d v = .elimMulti pairs) :
    (v.HYPOTHESIS || (predsOf d v).isEmpty) = false ∧
      elimPairsAt d v = pairs ∧ pairs ≠ [] := by
  unfold localShape at h
  split at h
  · simp at h
  · rename_i hguard
    have hguardFalse : (v.HYPOTHESIS || (predsOf d v).isEmpty) = false := by
      cases hg : (v.HYPOTHESIS || (predsOf d v).isEmpty) with
      | false => rfl
      | true => exact (hguard hg).elim
    generalize hpairsEq : elimPairsAt d v = pairList at h
    cases pairList with
    | nil =>
        cases hpreds : predsOf d v with
        | nil => simp [hpreds] at h
        | cons premise rest =>
            cases rest with
            | nil =>
                cases hform : v.FORMULA with
                | atom symbol => simp [hpreds, hform] at h
                | implication alpha beta =>
                    by_cases hpform : premise.FORMULA = beta <;>
                      simp [hpreds, hform, hpform] at h
            | cons premise' rest => simp [hpreds] at h
    | cons first rest =>
        cases rest with
        | nil => simp at h
        | cons second rest =>
            cases h
            refine ⟨hguardFalse, ?_, by simp⟩
            simp

lemma localShape_intro_raw_inv {d : Graph} {v premise : Vertex}
    {alpha beta : Formula} (h : localShape d v = .intro premise alpha beta) :
    (v.HYPOTHESIS || (predsOf d v).isEmpty) = false ∧
      predsOf d v = [premise] ∧
      v.FORMULA = Formula.implication alpha beta ∧
      premise.FORMULA = beta := by
  unfold localShape at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · split at h
      · split at h <;> simp_all
      · simp at h
    · exact False.elim (LocalRuleShape.elimMulti_ne_intro _ _ _ _ h)

lemma localShape_intro_inv {d : Graph} {v premise : Vertex}
    {alpha beta : Formula} (hstruct : structuralValid d)
    (h : localShape d v = .intro premise alpha beta) :
    predsOf d v = [premise] ∧
      v.FORMULA = Formula.implication alpha beta ∧
      premise.FORMULA = beta ∧ premise ∈ d.NODES ∧
      premise.LEVEL = v.LEVEL + 1 ∧ get_rule.outgoing premise d ≠ [] ∧
      elimPairsAt d v = [] := by
  obtain ⟨_, hpreds, hform, hpform⟩ := localShape_intro_raw_inv h
  have hpred : premise ∈ predsOf d v := by simp [hpreds]
  have hpnode := mem_predsOf_mem_nodes hstruct hpred
  obtain ⟨hlevel, hout⟩ := predsOf_props hstruct hpred
  have helim : elimPairsAt d v = [] := by
    have hneq : premise.FORMULA ≠
        Formula.implication premise.FORMULA v.FORMULA := by
      intro heq
      have hsize := congrArg sizeOf heq
      simp at hsize
      omega
    unfold elimPairsAt
    rw [hpreds]
    simp [hneq]
  exact ⟨hpreds, hform, hpform, hpnode, hlevel, hout, helim⟩

lemma guard_false_parts {d : Graph} {v : Vertex}
    (h : (v.HYPOTHESIS || (predsOf d v).isEmpty) = false) :
    v.HYPOTHESIS = false ∧ (predsOf d v).isEmpty = false := by
  cases hh : v.HYPOTHESIS <;> cases hp : (predsOf d v).isEmpty <;> simp_all

theorem storedRoutes_setEq_flow (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (hloc : LocalChecks d c) :
    ∀ v ∈ d.NODES,
      ListSetEq (StoredRoutes c v)
        (WitnessableRoutes d v (flowAt d (stdFuel d) v)) := by
  have hgraph : c.graph = d := hloc.1
  subst d
  have hprod : LocalProduction c := hloc.2.1
  have hstruct : structuralValid c.graph := hloc.2.2.2.2.2.1
  have key : ∀ n, ∀ v ∈ c.graph.NODES, countAbove c.graph v = n →
      ListSetEq (StoredRoutes c v)
        (WitnessableRoutes c.graph v
          (flowAt c.graph (stdFuel c.graph) v)) := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro v hv hn
      have hproduction := productionAtB_setEq c (hprod v hv)
      have hfuel : stdFuel c.graph - 1 + 1 = stdFuel c.graph := by
        unfold stdFuel
        omega
      cases hs : localShape c.graph v with
      | source =>
          have hguard := localShape_source_inv hs
          unfold witnessableExpected at hproduction
          simp [localExpected, hs] at hproduction
          have hflow := flowAt_top_eq c.graph (stdFuel c.graph - 1) hguard
          rw [hfuel] at hflow
          exact hproduction.trans
            (witnessableRoutes_congr c.graph v (by
              intro route
              unfold sourceExpected
              rw [hflow]))
      | elim major minor =>
          obtain ⟨hguard, hpairs⟩ := localShape_elim_raw_inv hs
          obtain ⟨hhyp, hpredsne⟩ := guard_false_parts hguard
          obtain ⟨_, hmajorNode, hminorNode, hmajorLevel, hminorLevel,
            hmajorOut, hminorOut⟩ := localShape_elim_inv hstruct hs
          have hmajorLt := countAbove_lt_of_level_succ c.graph
            hmajorNode hmajorLevel
          have hminorLt := countAbove_lt_of_level_succ c.graph
            hminorNode hminorLevel
          have hmajor := ih (countAbove c.graph major) (by omega)
            major hmajorNode rfl
          have hminor := ih (countAbove c.graph minor) (by omega)
            minor hminorNode rfl
          unfold witnessableExpected at hproduction
          simp [localExpected, hs] at hproduction
          have hcongr := elimExpected_congr_flow c (v := v) hstruct
            hmajorNode hminorNode
            hmajorOut hminorOut hmajor hminor
          have hpredRoutes : ∀ u ∈ predsOf c.graph v,
              ListSetEq (StoredRoutes c u)
                (WitnessableRoutes c.graph u
                  (flowAt c.graph (stdFuel c.graph - 1) u)) := by
            intro u hu
            have hunode := mem_predsOf_mem_nodes hstruct hu
            obtain ⟨hulevel, huout⟩ := predsOf_props hstruct hu
            have hult := countAbove_lt_of_level_succ c.graph hunode hulevel
            have huroutes := ih (countAbove c.graph u) (by omega)
              u hunode rfl
            apply huroutes.trans
            apply witnessableRoutes_congr
            intro route
            rw [flowAt_stdFuel_pred c.graph hstruct hunode huout]
          have hseed : ListSetEq (reseedExpected c v)
              (reseedPairs c.graph v
                (flowAt c.graph (stdFuel c.graph - 1))) := by
            unfold reseedExpected
            exact (reseedPairs_congr_set_of_preds c.graph v _ _
              hpredRoutes).trans
                (reseedProjectInputs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)))
          have hmixed : ListSetEq (mixedIntroExpected c v)
              (mixedIntroPairs c.graph v
                (flowAt c.graph (stdFuel c.graph - 1))).eraseDups := by
            have hpredRoutes' : ∀ u ∈ predsOf c.graph v,
                ListSetEq (outputsAt c u)
                  (WitnessableRoutes c.graph u
                    (flowAt c.graph (stdFuel c.graph - 1) u)) := by
              simpa [StoredRoutes] using hpredRoutes
            unfold mixedIntroExpected
            exact (eraseDups_congr
              (mixedIntroPairs_congr_set_of_preds c.graph v _ _
                hpredRoutes')).trans
              (eraseDups_congr
                (mixedIntroProjectInputs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1))))
          have hflow := flowAt_kernel_elim_eq c.graph
            (stdFuel c.graph - 1) hhyp hpredsne hpairs
          rw [hfuel] at hflow
          have hcombined : ListSetEq
              (elimExpected c v major minor ++ (mixedIntroExpected c v ++
                reseedExpected c v)).eraseDups
              (elimCombine c.graph v major minor
                    (flowAt c.graph (stdFuel c.graph - 1) minor)
                    (flowAt c.graph (stdFuel c.graph - 1) major) ++
                mixedIntroPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)) ++
                reseedPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1))).eraseDups := by
            intro route
            simp only [List.mem_eraseDups, List.mem_append]
            have hordinary : route ∈ elimExpected c v major minor ↔
                route ∈ elimCombine c.graph v major minor
                  (flowAt c.graph (stdFuel c.graph - 1) minor)
                  (flowAt c.graph (stdFuel c.graph - 1) major) := by
              simpa using hcongr route
            have hmixedRoute : route ∈ mixedIntroExpected c v ↔
                route ∈ mixedIntroPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)) := by
              simpa only [List.mem_eraseDups] using hmixed route
            exact (or_congr hordinary
              (or_congr hmixedRoute (hseed route))).trans (or_assoc ..).symm
          exact hproduction.trans
            (witnessableRoutes_congr c.graph v
              (hcombined.trans (by
                intro route
                rw [hflow])))
      | elimMulti pairs =>
          obtain ⟨hguard, hpairsEq, hpairsNe⟩ := localShape_elimMulti_raw_inv hs
          obtain ⟨hhyp, hpredsne⟩ := guard_false_parts hguard
          unfold witnessableExpected at hproduction
          simp [localExpected, hs] at hproduction
          have hcongr : ListSetEq (elimExpectedMany c v pairs)
              ((pairs.flatMap fun (major, minor) =>
                elimCombine c.graph v major minor
                  (flowAt c.graph (stdFuel c.graph - 1) minor)
                  (flowAt c.graph (stdFuel c.graph - 1) major)).eraseDups) := by
            unfold elimExpectedMany
            apply eraseDups_congr
            apply flatMap_congr_fun_mem
            intro pair hpair
            rcases pair with ⟨major, minor⟩
            have hpairGraph : (major, minor) ∈ elimPairsAt c.graph v := by
              rw [hpairsEq]
              exact hpair
            obtain ⟨hmajorPred, hminorPred⟩ :=
              mem_preds_of_mem_elimPairsAt hpairGraph
            have hmajorNode := mem_predsOf_mem_nodes hstruct hmajorPred
            have hminorNode := mem_predsOf_mem_nodes hstruct hminorPred
            obtain ⟨hmajorLevel, hmajorOut⟩ := predsOf_props hstruct hmajorPred
            obtain ⟨hminorLevel, hminorOut⟩ := predsOf_props hstruct hminorPred
            have hmajorLt := countAbove_lt_of_level_succ c.graph
              hmajorNode hmajorLevel
            have hminorLt := countAbove_lt_of_level_succ c.graph
              hminorNode hminorLevel
            have hmajor := ih (countAbove c.graph major) (by omega)
              major hmajorNode rfl
            have hminor := ih (countAbove c.graph minor) (by omega)
              minor hminorNode rfl
            have hmajorStable : ListSetEq (StoredRoutes c major)
                (WitnessableRoutes c.graph major
                  (flowAt c.graph (stdFuel c.graph - 1) major)) := by
              apply hmajor.trans
              apply witnessableRoutes_congr
              intro route
              rw [flowAt_stdFuel_pred c.graph hstruct hmajorNode hmajorOut]
            have hminorStable : ListSetEq (StoredRoutes c minor)
                (WitnessableRoutes c.graph minor
                  (flowAt c.graph (stdFuel c.graph - 1) minor)) := by
              apply hminor.trans
              apply witnessableRoutes_congr
              intro route
              rw [flowAt_stdFuel_pred c.graph hstruct hminorNode hminorOut]
            exact (elimCombine_congr_raw c.graph v major minor
              hminorStable hmajorStable).trans
                (elimCombine_project_inputs c.graph v major minor
                  (flowAt c.graph (stdFuel c.graph - 1) minor)
                  (flowAt c.graph (stdFuel c.graph - 1) major))
          have hpairsGraphNe : elimPairsAt c.graph v ≠ [] := by
            simpa [hpairsEq] using hpairsNe
          have hpredRoutes : ∀ u ∈ predsOf c.graph v,
              ListSetEq (StoredRoutes c u)
                (WitnessableRoutes c.graph u
                  (flowAt c.graph (stdFuel c.graph - 1) u)) := by
            intro u hu
            have hunode := mem_predsOf_mem_nodes hstruct hu
            obtain ⟨hulevel, huout⟩ := predsOf_props hstruct hu
            have hult := countAbove_lt_of_level_succ c.graph hunode hulevel
            have huroutes := ih (countAbove c.graph u) (by omega)
              u hunode rfl
            apply huroutes.trans
            apply witnessableRoutes_congr
            intro route
            rw [flowAt_stdFuel_pred c.graph hstruct hunode huout]
          have hseed : ListSetEq (reseedExpected c v)
              (reseedPairs c.graph v
                (flowAt c.graph (stdFuel c.graph - 1))) := by
            unfold reseedExpected
            exact (reseedPairs_congr_set_of_preds c.graph v _ _
              hpredRoutes).trans
                (reseedProjectInputs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)))
          have hmixed : ListSetEq (mixedIntroExpected c v)
              (mixedIntroPairs c.graph v
                (flowAt c.graph (stdFuel c.graph - 1))).eraseDups := by
            have hpredRoutes' : ∀ u ∈ predsOf c.graph v,
                ListSetEq (outputsAt c u)
                  (WitnessableRoutes c.graph u
                    (flowAt c.graph (stdFuel c.graph - 1) u)) := by
              simpa [StoredRoutes] using hpredRoutes
            unfold mixedIntroExpected
            exact (eraseDups_congr
              (mixedIntroPairs_congr_set_of_preds c.graph v _ _
                hpredRoutes')).trans
              (eraseDups_congr
                (mixedIntroProjectInputs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1))))
          have hflow := flowAt_kernel_elim_many_eq c.graph
            (stdFuel c.graph - 1) hhyp hpredsne hpairsGraphNe
          rw [hfuel, hpairsEq] at hflow
          have hcombined : ListSetEq
              (elimExpectedMany c v pairs ++ (mixedIntroExpected c v ++
                reseedExpected c v)).eraseDups
              (((pairs.flatMap fun (major, minor) =>
                  elimCombine c.graph v major minor
                    (flowAt c.graph (stdFuel c.graph - 1) minor)
                    (flowAt c.graph (stdFuel c.graph - 1) major))) ++
                mixedIntroPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)) ++
                reseedPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1))).eraseDups := by
            intro route
            simp only [List.mem_eraseDups, List.mem_append]
            have hordinary : route ∈ elimExpectedMany c v pairs ↔
                route ∈ (pairs.flatMap fun (major, minor) =>
                  elimCombine c.graph v major minor
                    (flowAt c.graph (stdFuel c.graph - 1) minor)
                    (flowAt c.graph (stdFuel c.graph - 1) major)) := by
              simpa using hcongr route
            have hmixedRoute : route ∈ mixedIntroExpected c v ↔
                route ∈ mixedIntroPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)) := by
              simpa only [List.mem_eraseDups] using hmixed route
            exact (or_congr hordinary
              (or_congr hmixedRoute (hseed route))).trans (or_assoc ..).symm
          exact hproduction.trans
            (witnessableRoutes_congr c.graph v
              (hcombined.trans (by
                intro route
                rw [hflow])))
      | intro premise alpha beta =>
          obtain ⟨hguard, _, _, _⟩ := localShape_intro_raw_inv hs
          obtain ⟨hhyp, _⟩ := guard_false_parts hguard
          obtain ⟨hpreds, hform, hpform, hpremiseNode, hpremiseLevel,
            hpremiseOut, helim⟩ := localShape_intro_inv hstruct hs
          have hpremiseLt := countAbove_lt_of_level_succ c.graph
            hpremiseNode hpremiseLevel
          have hpremise := ih (countAbove c.graph premise) (by omega)
            premise hpremiseNode rfl
          unfold witnessableExpected at hproduction
          simp [localExpected, hs] at hproduction
          have hcongr := introExpected_congr_flow c (v := v) alpha hstruct
            hpremiseNode hpremiseOut hpremise
          have hpredRoutes : ∀ u ∈ predsOf c.graph v,
              ListSetEq (StoredRoutes c u)
                (WitnessableRoutes c.graph u
                  (flowAt c.graph (stdFuel c.graph - 1) u)) := by
            intro u hu
            have hunode := mem_predsOf_mem_nodes hstruct hu
            obtain ⟨hulevel, huout⟩ := predsOf_props hstruct hu
            have hult := countAbove_lt_of_level_succ c.graph hunode hulevel
            have huroutes := ih (countAbove c.graph u) (by omega)
              u hunode rfl
            apply huroutes.trans
            apply witnessableRoutes_congr
            intro route
            rw [flowAt_stdFuel_pred c.graph hstruct hunode huout]
          have hseed : ListSetEq (reseedExpected c v)
              (reseedPairs c.graph v
                (flowAt c.graph (stdFuel c.graph - 1))) := by
            unfold reseedExpected
            exact (reseedPairs_congr_set_of_preds c.graph v _ _
              hpredRoutes).trans
                (reseedProjectInputs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1)))
          have hflow := flowAt_kernel_intro_eq c.graph
            (stdFuel c.graph - 1) hhyp hpreds hform hpform helim
          rw [hfuel] at hflow
          have hcombined : ListSetEq
              (introExpected c v premise alpha ++ reseedExpected c v).eraseDups
              (((flowAt c.graph (stdFuel c.graph - 1) premise).flatMap fun route =>
                  (edgesBetween c.graph premise v).filterMap fun edge =>
                    (consume edge route.2).map fun tail =>
                      (depRemove route.1 alpha, tail)) ++
                reseedPairs c.graph v
                  (flowAt c.graph (stdFuel c.graph - 1))).eraseDups := by
            intro route
            simp only [List.mem_eraseDups, List.mem_append]
            have hordinary : route ∈ introExpected c v premise alpha ↔
                route ∈ ((flowAt c.graph (stdFuel c.graph - 1) premise).flatMap
                  fun prior =>
                    (edgesBetween c.graph premise v).filterMap fun edge =>
                      (consume edge prior.2).map fun tail =>
                        (depRemove prior.1 alpha, tail)) := by
              simpa using hcongr route
            exact or_congr hordinary (hseed route)
          exact hproduction.trans
            (witnessableRoutes_congr c.graph v
              (hcombined.trans (by
                intro route
                rw [hflow])))
      | invalid =>
          have himpossible := hprod v hv
          unfold productionAtB at himpossible
          simp [hs] at himpossible
  intro v hv
  exact key (countAbove c.graph v) v hv rfl

lemma eq_of_mem_of_map_nodup {α β : Type} [DecidableEq β]
    (f : α → β) {xs : List α} (h : (xs.map f).Nodup)
    {a b : α} (ha : a ∈ xs) (hb : b ∈ xs) (hab : f a = f b) : a = b := by
  induction xs with
  | nil => simp at ha
  | cons x xs ih =>
      simp only [List.map_cons, List.nodup_cons] at h
      rcases h with ⟨hnot, htail⟩
      simp only [List.mem_cons] at ha hb
      rcases ha with rfl | ha
      · rcases hb with rfl | hb
        · rfl
        · exfalso
          apply hnot
          rw [hab]
          exact List.mem_map.mpr ⟨b, hb, rfl⟩
      · rcases hb with rfl | hb
        · exfalso
          apply hnot
          rw [← hab]
          exact List.mem_map.mpr ⟨a, ha, rfl⟩
        · exact ih htail ha hb

lemma coverageAtB_sound (c : Certificate) {v : Vertex}
    (h : coverageAtB c v = true) :
    ∀ e ∈ get_rule.outgoing v c.graph,
      edgeWitnessesFor c e ≠ [] ∧
        ListSetEq
          ((edgeWitnessesFor c e).map fun witness =>
            (witness.dependency, witness.residual))
          ((outputsAt c v).filter fun route =>
            headColour route.2 == e.COLOUR) := by
  unfold coverageAtB at h
  simp only [Bool.and_eq_true, List.all_eq_true] at h
  intro e he
  have hedge := h.1 e he
  have hedge' : !(edgeWitnessesFor c e).isEmpty = true ∧
      setEqB
        ((edgeWitnessesFor c e).map fun witness =>
          (witness.dependency, witness.residual))
        ((outputsAt c v).filter fun route =>
          headColour route.2 == e.COLOUR) = true := by
    simpa using hedge
  refine ⟨?_, setEqB_sound hedge'.2⟩
  cases hs : edgeWitnessesFor c e with
  | nil => simp [hs] at hedge'
  | cons witness rest => simp []

lemma tailClosureAtB_sound (c : Certificate) {v : Vertex}
    (h : tailClosureAtB c v = true) :
    elimPairsAt c.graph v ≠ [] →
      (∀ u ∈ predsOf c.graph v,
        ElimPremiseAt c.graph v u ∨ LocalCarrierAccountedAt c v u) ∧
      (∀ major minor, (major, minor) ∈ elimPairsAt c.graph v →
        certificatePairTailClosureAtB c v major minor = true) := by
  intro hpairsNe
  unfold CertificateModel.tailClosureAtB at h
  cases hpairs : elimPairsAt c.graph v with
  | nil => exact (hpairsNe hpairs).elim
  | cons pair pairs =>
    rw [hpairs] at h
    simp only [Bool.and_eq_true, List.all_eq_true] at h
    refine ⟨?_, ?_⟩
    · intro u hu
      have hu' := h.1 u hu
      rw [Bool.or_eq_true] at hu'
      rcases hu' with helim | hcarrier
      · exact Or.inl (elimPremiseAtB_sound c.graph v u helim)
      · right
        unfold certificateCarrierAccountedAtB at hcarrier
        cases hformula : v.FORMULA with
        | atom name => simp [hformula] at hcarrier
        | implication alpha beta =>
            simp only [hformula, Bool.and_eq_true, List.all_eq_true] at hcarrier
            have huFormula : u.FORMULA = beta := by simpa using hcarrier.1
            refine ⟨alpha, by simp [hformula, huFormula], ?_⟩
            intro tail htail
            have hresult := hcarrier.2 tail htail
            rw [List.any_eq_true] at hresult
            obtain ⟨result, hresultMem, hresidual⟩ := hresult
            rcases result with ⟨dependency, residual⟩
            have hresidual' : residual = tail := by simpa using hresidual
            subst tail
            exact ⟨dependency, hresultMem⟩
    · intro major minor hpair
      apply h.2 (major, minor)
      simpa [hpairs] using hpair

lemma rootDischargeAtB_sound (c : Certificate) {v : Vertex}
    (h : rootDischargeAtB c v = true)
    (hterminal : (get_rule.outgoing v c.graph).isEmpty = true) :
    (∃ output, output ∈ c.terminalOutputs ∧ output.node = v ∧
      output.dependency = [] ∧ output.residual = []) ∧
      RuleShapeOK v c.graph ∧ outDep v c.graph = [] := by
  unfold rootDischargeAtB at h
  simp only [hterminal, if_true] at h
  rw [List.any_eq_true] at h
  obtain ⟨output, houtput, hchecks⟩ := h
  rw [List.mem_filter] at houtput
  obtain ⟨houtput, hnode⟩ := houtput
  simp only [Bool.and_eq_true] at hchecks
  refine ⟨⟨output, houtput, by simpa using hnode, ?_, ?_⟩, ?_, ?_⟩
  · exact List.isEmpty_iff.mp hchecks.1.1.1.1
  · exact List.isEmpty_iff.mp hchecks.1.1.1.2
  · exact hchecks.1.2
  · exact of_decide_eq_true hchecks.2


theorem localChecksB_sound (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (h : localChecksB d c = true) :
    LocalChecks d c := by
  have _expandedDomain : ExpandedCert d c := hexp
  simp only [localChecksB, Bool.and_eq_true] at h
  rcases h with ⟨h, hhyp⟩
  rcases h with ⟨h, hlevelFormula⟩
  rcases h with ⟨h, hfaithful⟩
  rcases h with ⟨h, hone⟩
  rcases h with ⟨h, hnodup⟩
  rcases h with ⟨h, hstruct⟩
  rcases h with ⟨h, hroot⟩
  rcases h with ⟨h, htail⟩
  rcases h with ⟨h, hcoverage⟩
  rcases h with ⟨hgraph, hproduction⟩
  have hgraph' : c.graph = d := of_decide_eq_true hgraph
  have hproduction' : LocalProduction c := by
    intro v hv
    rw [hgraph'] at hv
    rw [List.all_eq_true] at hproduction
    exact hproduction v hv
  have hcoverage' : LocalCoverage c := by
    rw [List.all_eq_true] at hcoverage
    intro v hv
    exact coverageAtB_sound c
      (hcoverage v (by simpa [hgraph'] using hv))
  have htail' : LocalTailClosure c := by
    rw [List.all_eq_true] at htail
    intro v hv
    exact tailClosureAtB_sound c (htail v (by simpa [hgraph'] using hv))
  have hroot' : LocalRootDischarge c := by
    rw [List.all_eq_true] at hroot
    intro v hv hterminal
    have hcheck := hroot v (by simpa [hgraph'] using hv)
    exact rootDischargeAtB_sound c hcheck hterminal
  have hfaithful' : LocalWitnessFaithful c :=
    localWitnessFaithful_of_all c hfaithful
  exact ⟨hgraph', hproduction', hcoverage', htail', hroot',
    structuralValidB_sound d hstruct, nodupB_sound d hnodup,
    oneEdgeB_sound d hone, hfaithful',
    levelFormulaUniqueB_sound d hlevelFormula,
    hypNoIncomingB_sound d hhyp⟩

lemma edgeWitness_mem_stored (c : Certificate) {v : Vertex}
    {w : EdgeRouteWitness} (hw : w ∈ c.edgeRoutes)
    (hstart : w.edge.START = v) :
    (w.dependency, w.residual) ∈ StoredRoutes c v := by
  unfold StoredRoutes outputsAt
  rw [List.mem_eraseDups, List.mem_append]
  left
  rw [List.mem_filterMap]
  refine ⟨w, hw, ?_⟩
  simp [hstart]

lemma edgeWitness_mem_filter (c : Certificate) {e : Deduction}
    {w : EdgeRouteWitness} (hw : w ∈ c.edgeRoutes) (hwe : w.edge = e) :
    w ∈ edgeWitnessesFor c e := by
  unfold edgeWitnessesFor
  rw [List.mem_filter]
  exact ⟨hw, by simp [hwe]⟩

lemma consumedWitnessTails_mem (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) {start finish : Vertex} {e : Deduction}
    {w : EdgeRouteWitness} {tail : ColourPath}
    (he : e ∈ edgesBetween d start finish)
    (hw : w ∈ c.edgeRoutes) (hwe : w.edge = e)
    (hconsume : consume e w.residual = some tail) :
    tail ∈ consumedWitnessTails c start finish := by
  have hgraph := hexp.1
  subst d
  unfold consumedWitnessTails
  rw [List.mem_eraseDups, List.mem_flatMap]
  refine ⟨e, ?_, ?_⟩
  · exact he
  · rw [List.mem_filterMap]
    refine ⟨w, edgeWitness_mem_filter c hw hwe, ?_⟩
    exact hconsume

lemma consumedWitnessTails_exists (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) {start finish : Vertex} {tail : ColourPath}
    (htail : tail ∈ consumedWitnessTails c start finish) :
    ∃ e ∈ edgesBetween d start finish, ∃ w ∈ c.edgeRoutes,
      w.edge = e ∧ consume e w.residual = some tail := by
  have hgraph := hexp.1
  subst d
  unfold consumedWitnessTails at htail
  rw [List.mem_eraseDups, List.mem_flatMap] at htail
  obtain ⟨e, he, hresult⟩ := htail
  rw [List.mem_filterMap] at hresult
  obtain ⟨w, hwFilter, hconsume⟩ := hresult
  unfold edgeWitnessesFor at hwFilter
  rw [List.mem_filter] at hwFilter
  obtain ⟨hw, hwe⟩ := hwFilter
  have hwe' : w.edge = e := by simpa using hwe
  exact ⟨e, he, w, hw, hwe', hconsume⟩

lemma stored_mem_edgeWitness_of_outgoing (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) {v : Vertex}
    (hout : get_rule.outgoing v d ≠ []) {route : FlowPair}
    (hr : route ∈ StoredRoutes c v) :
    ∃ w ∈ c.edgeRoutes, w.edge.START = v ∧
      route = (w.dependency, w.residual) := by
  have hgraph := hexp.1
  subst d
  unfold StoredRoutes outputsAt at hr
  rw [List.mem_eraseDups, List.mem_append] at hr
  rcases hr with hedge | hterminal
  · rw [List.mem_filterMap] at hedge
    obtain ⟨w, hw, hroute⟩ := hedge
    by_cases hs : w.edge.START = v
    · simp [hs] at hroute
      exact ⟨w, hw, hs, hroute.symm⟩
    · simp [hs] at hroute
  · rw [List.mem_filterMap] at hterminal
    obtain ⟨output, ho, hroute⟩ := hterminal
    by_cases hn : output.node = v
    · have hterm := (hexp.2.2.2.2 output ho).2
      rw [hn] at hterm
      exact absurd (List.isEmpty_iff.mp hterm) hout
    · simp [hn] at hroute


def WitnessableRouteHeadCoverage (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] →
    (∀ r ∈ WitnessableRoutes d v (flowAt d (stdFuel d) v),
        ∃ e ∈ get_rule.outgoing v d, e.COLOUR = headColour r.2) ∧
    (∀ e ∈ get_rule.outgoing v d,
        ∃ r ∈ WitnessableRoutes d v (flowAt d (stdFuel d) v),
          headColour r.2 = e.COLOUR)



def WitnessablePairFlowTailClosure (d : Graph) (w M S : Vertex) : Prop :=
  (∀ bp ∈ WitnessableRoutes d S (flowAt d (stdFuel d) S),
      ∀ eS ∈ edgesBetween d S w, ∀ q,
      consume eS bp.2 = some q → RouteContinuableAt d w q →
      ∃ bq ∈ WitnessableRoutes d M (flowAt d (stdFuel d) M),
        ∃ eM ∈ edgesBetween d M w, consume eM bq.2 = some q) ∧
  (∀ bp ∈ WitnessableRoutes d M (flowAt d (stdFuel d) M),
      ∀ eM ∈ edgesBetween d M w, ∀ q,
      consume eM bp.2 = some q → RouteContinuableAt d w q →
      ∃ bq ∈ WitnessableRoutes d S (flowAt d (stdFuel d) S),
        ∃ eS ∈ edgesBetween d S w, consume eS bq.2 = some q)



def WitnessableCarrierAccountedAt (d : Graph) (w u : Vertex) : Prop :=
  ∃ alpha, w.FORMULA = Formula.implication alpha u.FORMULA ∧
    ∀ bp ∈ WitnessableRoutes d u (flowAt d (stdFuel d) u),
      ∀ edge ∈ edgesBetween d u w, ∀ tail,
      consume edge bp.2 = some tail → RouteContinuableAt d w tail →
        ∃ dependency,
          (dependency, tail) ∈
            WitnessableRoutes d w (flowAt d (stdFuel d) w)


def WitnessableFlowTailClosure (d : Graph) : Prop :=
  ∀ w ∈ d.NODES, elimPairsAt d w ≠ [] →
    (∀ u ∈ predsOf d w,
      ElimPremiseAt d w u ∨ WitnessableCarrierAccountedAt d w u) ∧
    (∀ M S, (M, S) ∈ elimPairsAt d w →
      WitnessablePairFlowTailClosure d w M S)

theorem witnessableRouteHeadCoverage_of_local (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (hloc : LocalChecks d c) :
    WitnessableRouteHeadCoverage d := by
  have hroutes := storedRoutes_setEq_flow d c hexp hloc
  have hcoverage : LocalCoverage c := hloc.2.2.1
  intro v hv hout
  constructor
  · intro route hr
    have hstored := (hroutes v hv route).mpr hr
    obtain ⟨w, hw, hstart, rfl⟩ :=
      stored_mem_edgeWitness_of_outgoing d c hexp hout hstored
    have hgraph := hloc.1
    have hedgeGraph : w.edge ∈ d.EDGES := by
      subst d
      exact hexp.2.2.1 w hw
    have hedgeOut : w.edge ∈ get_rule.outgoing v d := by
      exact mem_outgoing_of_mem_edges_start_eq v d hedgeGraph hstart
    have hwFilter := edgeWitness_mem_filter c hw rfl
    have hwMapped : (w.dependency, w.residual) ∈
        (edgeWitnessesFor c w.edge).map fun witness =>
          (witness.dependency, witness.residual) :=
      List.mem_map.mpr ⟨w, hwFilter, rfl⟩
    have hfiltered := (hcoverage v (by simpa [hgraph] using hv) w.edge
      (by simpa [hgraph] using hedgeOut)).2 _ |>.mp hwMapped
    refine ⟨w.edge, hedgeOut, ?_⟩
    rw [List.mem_filter] at hfiltered
    have hhead : headColour w.residual = w.edge.COLOUR := by
      simpa using hfiltered.2
    exact hhead.symm
  · intro edge hedge
    have hnonempty :=
      (hcoverage v (by simpa [hloc.1] using hv) edge
        (by simpa [hloc.1] using hedge)).1
    cases hs : edgeWitnessesFor c edge with
    | nil => exact (hnonempty hs).elim
    | cons w ws =>
      have hwFilter : w ∈ edgeWitnessesFor c edge := by simp [hs]
      unfold edgeWitnessesFor at hwFilter
      rw [List.mem_filter] at hwFilter
      obtain ⟨hw, hwe⟩ := hwFilter
      have hwe' : w.edge = edge := by simpa using hwe
      have hstart : w.edge.START = v := by
        rw [hwe']
        exact mem_outgoing_start_eq v d hedge
      have hstored := edgeWitness_mem_stored c hw hstart
      have hflow := (hroutes v hv _).mp hstored
      have hwMapped : (w.dependency, w.residual) ∈
          (edgeWitnessesFor c edge).map fun witness =>
            (witness.dependency, witness.residual) :=
        List.mem_map.mpr ⟨w, edgeWitness_mem_filter c hw hwe', rfl⟩
      have hfiltered := (hcoverage v (by simpa [hloc.1] using hv) edge
        (by simpa [hloc.1] using hedge)).2 _ |>.mp hwMapped
      rw [List.mem_filter] at hfiltered
      exact ⟨(w.dependency, w.residual), hflow, by simpa using hfiltered.2⟩

theorem witnessableFlowTailClosure_of_local (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (hloc : LocalChecks d c)
    (htails : LocalEqualityTailClosure c) :
    WitnessableFlowTailClosure d := by
  have hgraph := hloc.1
  subst d
  have hroutes := storedRoutes_setEq_flow c.graph c hexp hloc
  have hcoverage : LocalCoverage c := hloc.2.2.1
  have hproduction : LocalProduction c := hloc.2.1
  have hstruct : structuralValid c.graph := hloc.2.2.2.2.2.1
  intro node hnode hpairsNe
  obtain ⟨hpredCoverage, hpairTails⟩ := htails node hnode hpairsNe
  have route_to_tail : ∀ start, start ∈ c.graph.NODES →
      ∀ route ∈ WitnessableRoutes c.graph start
        (flowAt c.graph (stdFuel c.graph) start),
      ∀ edge ∈ edgesBetween c.graph start node, ∀ tail,
      consume edge route.2 = some tail →
      RouteContinuableAt c.graph node tail →
      tail ∈ continuableConsumedWitnessTails c start node := by
    intro start hstartNode route hroute edge hedge tail hconsume hcontinue
    have hedgeOut := mem_outgoing_of_edgesBetween hedge
    have hrstored := (hroutes start hstartNode route).mpr hroute
    have hrfiltered : route ∈ (StoredRoutes c start).filter fun selection =>
        headColour selection.2 == edge.COLOUR := by
      rw [List.mem_filter]
      exact ⟨hrstored, by simp [consume_some_colour hconsume]⟩
    have hrmapped := (hcoverage start hstartNode edge hedgeOut).2 route |>.mpr hrfiltered
    obtain ⟨witness, hwFilter, hsame⟩ := List.mem_map.mp hrmapped
    unfold edgeWitnessesFor at hwFilter
    rw [List.mem_filter] at hwFilter
    obtain ⟨hwitness, hwedge⟩ := hwFilter
    have hwedge' : witness.edge = edge := by simpa using hwedge
    have hresidual : witness.residual = route.2 := congrArg Prod.snd hsame
    unfold continuableConsumedWitnessTails
    rw [List.mem_filter]
    refine ⟨?_, hcontinue⟩
    apply consumedWitnessTails_mem c.graph c hexp hedge hwitness hwedge'
    simpa [hresidual] using hconsume
  have tail_to_route : ∀ start, start ∈ c.graph.NODES → ∀ tail,
      tail ∈ continuableConsumedWitnessTails c start node →
      ∃ route ∈ WitnessableRoutes c.graph start
          (flowAt c.graph (stdFuel c.graph) start),
        ∃ edge ∈ edgesBetween c.graph start node,
          consume edge route.2 = some tail := by
    intro start hstartNode tail htail
    unfold continuableConsumedWitnessTails at htail
    rw [List.mem_filter] at htail
    obtain ⟨edge, hedge, witness, hwitness, hwedge, hconsume⟩ :=
      consumedWitnessTails_exists c.graph c hexp htail.1
    have hwstart : witness.edge.START = start := by
      rw [hwedge]
      exact (mem_edgesBetween hedge).1
    have hwstored := edgeWitness_mem_stored c hwitness hwstart
    have hwflow := (hroutes start hstartNode _).mp hwstored
    exact ⟨(witness.dependency, witness.residual), hwflow,
      edge, hedge, by simpa [hwedge] using hconsume⟩
  refine ⟨?_, ?_⟩
  · intro predecessor hpredecessor
    rcases hpredCoverage predecessor hpredecessor with helim | hcarrier
    · exact Or.inl helim
    · right
      obtain ⟨alpha, hformula, haccounted⟩ := hcarrier
      refine ⟨alpha, hformula, ?_⟩
      intro route hroute edge hedge tail hconsume hcontinue
      have hpredecessorNode := mem_predsOf_mem_nodes hstruct hpredecessor
      have htail := route_to_tail predecessor hpredecessorNode route hroute
        edge hedge tail hconsume hcontinue
      obtain ⟨dependency, hlocal⟩ := haccounted tail htail
      have hstored :=
        (productionAtB_setEq c (hproduction node hnode) (dependency, tail)).mpr hlocal
      exact ⟨dependency, (hroutes node hnode _).mp hstored⟩
  intro major minor hpair
  obtain ⟨hmajorPred, hminorPred⟩ := mem_preds_of_mem_elimPairsAt hpair
  have hmajorNode := mem_predsOf_mem_nodes hstruct hmajorPred
  have hminorNode := mem_predsOf_mem_nodes hstruct hminorPred
  have htailEq := hpairTails major minor hpair
  refine ⟨?_, ?_⟩
  · intro route hroute edge hedge tail hconsume hcontinue
    have htailMinor := route_to_tail minor hminorNode route hroute edge hedge tail
      hconsume hcontinue
    have htailMajor := (htailEq tail).mpr htailMinor
    obtain ⟨majorRoute, hmajorRoute, majorEdge, hmajorEdge, hmajorConsume⟩ :=
      tail_to_route major hmajorNode tail htailMajor
    exact ⟨majorRoute, hmajorRoute, majorEdge, hmajorEdge, hmajorConsume⟩
  · intro route hroute edge hedge tail hconsume hcontinue
    have htailMajor := route_to_tail major hmajorNode route hroute edge hedge tail
      hconsume hcontinue
    have htailMinor := (htailEq tail).mp htailMajor
    obtain ⟨minorRoute, hminorRoute, minorEdge, hminorEdge, hminorConsume⟩ :=
      tail_to_route minor hminorNode tail htailMinor
    exact ⟨minorRoute, hminorRoute, minorEdge, hminorEdge, hminorConsume⟩

lemma retained_invariants_of_local (d : Graph) (c : Certificate)
    (h : LocalChecks d c) :
    structuralValid d ∧ d.EDGES.Nodup ∧ OneEdgePerColourPerNode d ∧
    LevelFormulaUnique d ∧ HypothesesHaveNoIncoming d := by
  rcases h with ⟨_, _, _, _, _, hstruct, hnodup, hone, _hfaithful,
    hlfu, hhyp⟩
  exact ⟨hstruct, hnodup, hone, hlfu, hhyp⟩

theorem invariants_of_expanded_certificate (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (hloc : LocalChecks d c) :
    structuralValid d ∧ d.EDGES.Nodup ∧ OneEdgePerColourPerNode d ∧
    WitnessableRouteHeadCoverage d ∧ LevelFormulaUnique d ∧
    HypothesesHaveNoIncoming d := by
  obtain ⟨hstruct, hnodup, hone, hlfu, hhyp⟩ :=
    retained_invariants_of_local d c hloc
  exact ⟨hstruct, hnodup, hone,
    witnessableRouteHeadCoverage_of_local d c hexp hloc, hlfu, hhyp⟩







theorem compressed_universally_accepted_certificateB (d : Graph)
    (c : Certificate) (hexp : expandedCertB d c = true)
    (hloc : localChecksB d c = true)
    (hcoverage : cursorRouteHeadCoverageB d = true)
    (hcontinuation : cursorFlowContinuationClosureB d = true)
    (hdis : dischargedMultiB d = true) :
    ∀ P, ¬ AdmissibleMultiPath d P ∨ DischargedMulti d P (goalColumn d) := by
  have hexp' := expandedCertB_sound d c hexp
  have hloc' := localChecksB_sound d c hexp' hloc
  obtain ⟨hstruct, hnodup, hone, hlfu, hhyp⟩ :=
    retained_invariants_of_local d c hloc'
  have hcontinuation' := cursorFlowContinuationClosureB_sound d hcontinuation
  exact compressed_universally_accepted' d hstruct hnodup hone
    (cursorRouteHeadCoverageB_sound d hcoverage) hlfu hhyp hcontinuation' hdis

theorem end_to_end_self_certifying :
    ∀ P, ¬ AdmissibleMultiPath Semantic.e P ∨
      DischargedMulti Semantic.e P (goalColumn Semantic.e) := by
  apply compressed_universally_accepted_certificateB Semantic.e
    CertificateModel.eCertificate
  · native_decide
  · native_decide
  · native_decide
  · first | decide | native_decide
  · native_decide

end SelfCertifying
