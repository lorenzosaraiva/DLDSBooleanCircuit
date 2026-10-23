import Infrastructure.SelfCertifying

open Semantic FlowSpec CertificateModel

namespace SelfCertifying

















def rootRouteEmptyB (route : FlowPair) : Bool :=
  route.1.isEmpty && route.2.isEmpty


def UniversalStoredRootDischarge (c : Certificate) : Prop :=
  ∀ vertex ∈ c.graph.NODES,
    (get_rule.outgoing vertex c.graph).isEmpty = true →
      ∀ route ∈ StoredRoutes c vertex,
        route.1 = [] ∧ route.2 = []



def LocalRootResidualClosure (c : Certificate) : Prop :=
  ∀ vertex ∈ c.graph.NODES,
    (get_rule.outgoing vertex c.graph).isEmpty = true →
      ∀ route ∈ localExpected c vertex, route.2 = []


def UniversalLocalRootDischargeAt (c : Certificate)
    (vertex : Vertex) : Prop :=
  (get_rule.outgoing vertex c.graph).isEmpty = true →
    (∀ route ∈ StoredRoutes c vertex,
      route.1 = [] ∧ route.2 = []) ∧
    (∀ route ∈ localExpected c vertex, route.2 = [])

def UniversalLocalRootDischarge (c : Certificate) : Prop :=
  ∀ vertex ∈ c.graph.NODES,
    UniversalLocalRootDischargeAt c vertex

def universalLocalRootDischargeAtB (c : Certificate)
    (vertex : Vertex) : Bool :=
  if (get_rule.outgoing vertex c.graph).isEmpty then
    (StoredRoutes c vertex).all rootRouteEmptyB &&
      (localExpected c vertex).all fun route => route.2.isEmpty
  else true

def universalLocalRootDischargeB (c : Certificate) : Bool :=
  c.graph.NODES.all (universalLocalRootDischargeAtB c)

lemma rootRouteEmptyB_eq_true_iff (route : FlowPair) :
    rootRouteEmptyB route = true ↔ route.1 = [] ∧ route.2 = [] := by
  rcases route with ⟨dependency, residual⟩
  simp [rootRouteEmptyB, List.isEmpty_iff]

theorem universalLocalRootDischargeAtB_eq_true_iff
    (c : Certificate) (vertex : Vertex) :
    universalLocalRootDischargeAtB c vertex = true ↔
      UniversalLocalRootDischargeAt c vertex := by
  by_cases root : (get_rule.outgoing vertex c.graph).isEmpty = true
  · rw [show universalLocalRootDischargeAtB c vertex =
        ((StoredRoutes c vertex).all rootRouteEmptyB &&
          (localExpected c vertex).all fun route => route.2.isEmpty) by
      simp [universalLocalRootDischargeAtB, root]]
    simp only [Bool.and_eq_true, List.all_eq_true]
    constructor
    · rintro ⟨stored, expected⟩
      intro _
      constructor
      · intro route member
        exact (rootRouteEmptyB_eq_true_iff route).mp (stored route member)
      · intro route member
        exact List.isEmpty_iff.mp (expected route member)
    · intro hlocal
      obtain ⟨stored, expected⟩ := hlocal root
      constructor
      · intro route member
        exact (rootRouteEmptyB_eq_true_iff route).mpr (stored route member)
      · intro route member
        exact List.isEmpty_iff.mpr (expected route member)
  · simp [universalLocalRootDischargeAtB,
      UniversalLocalRootDischargeAt, root]

theorem universalLocalRootDischargeB_eq_true_iff (c : Certificate) :
    universalLocalRootDischargeB c = true ↔
      UniversalLocalRootDischarge c := by
  unfold universalLocalRootDischargeB UniversalLocalRootDischarge
  rw [List.all_eq_true]
  constructor
  · intro checks vertex member
    exact (universalLocalRootDischargeAtB_eq_true_iff c vertex).mp
      (checks vertex member)
  · intro hlocal vertex member
    exact (universalLocalRootDischargeAtB_eq_true_iff c vertex).mpr
      (hlocal vertex member)

theorem universalLocalRootDischarge_stored
    {c : Certificate} (h : UniversalLocalRootDischarge c) :
    UniversalStoredRootDischarge c := by
  intro vertex vertex_mem root route route_mem
  exact (h vertex vertex_mem root).1 route route_mem

theorem universalLocalRootDischarge_residualClosure
    {c : Certificate} (h : UniversalLocalRootDischarge c) :
    LocalRootResidualClosure c := by
  intro vertex vertex_mem root route route_mem
  exact (h vertex vertex_mem root).2 route route_mem

theorem universalLocalRootDischarge_of_parts
    {c : Certificate} (stored : UniversalStoredRootDischarge c)
    (residual : LocalRootResidualClosure c) :
    UniversalLocalRootDischarge c := by
  intro vertex vertex_mem root
  exact ⟨stored vertex vertex_mem root,
    residual vertex vertex_mem root⟩








theorem localExpected_setEq_flow (d : Graph) (c : Certificate)
    (hexp : ExpandedCert d c) (hloc : LocalChecks d c) :
    ∀ v ∈ d.NODES,
      ListSetEq (localExpected c v) (flowAt d (stdFuel d) v) := by
  have hgraph : c.graph = d := hloc.1
  subst d
  have hroutes := storedRoutes_setEq_flow c.graph c hexp hloc
  have hprod : LocalProduction c := hloc.2.1
  have hstruct : structuralValid c.graph := hloc.2.2.2.2.2.1
  intro v hv
  have hfuel : stdFuel c.graph - 1 + 1 = stdFuel c.graph := by
    unfold stdFuel
    omega
  cases hs : localShape c.graph v with
  | source =>
      have hguard := localShape_source_inv hs
      have hflow := flowAt_top_eq c.graph (stdFuel c.graph - 1) hguard
      rw [hfuel] at hflow
      intro route
      simp only [localExpected, hs]
      unfold sourceExpected
      rw [hflow]
  | elim major minor =>
      obtain ⟨hguard, hpairs⟩ := localShape_elim_raw_inv hs
      obtain ⟨hhyp, hpredsne⟩ := guard_false_parts hguard
      obtain ⟨_, hmajorNode, hminorNode, _hmajorLevel, _hminorLevel,
        hmajorOut, hminorOut⟩ := localShape_elim_inv hstruct hs
      have hmajor := hroutes major hmajorNode
      have hminor := hroutes minor hminorNode
      have hcongr := elimExpected_congr_flow c (v := v) hstruct
        hmajorNode hminorNode hmajorOut hminorOut hmajor hminor
      have hpredRoutes : ∀ u ∈ predsOf c.graph v,
          ListSetEq (StoredRoutes c u)
            (WitnessableRoutes c.graph u
              (flowAt c.graph (stdFuel c.graph - 1) u)) := by
        intro u hu
        have hunode := mem_predsOf_mem_nodes hstruct hu
        obtain ⟨_hulevel, huout⟩ := predsOf_props hstruct hu
        have huroutes := hroutes u hunode
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
      simpa [localExpected, hs] using hcombined.trans (by
        intro route
        rw [hflow])
  | elimMulti pairs =>
      obtain ⟨hguard, hpairsEq, hpairsNe⟩ := localShape_elimMulti_raw_inv hs
      obtain ⟨hhyp, hpredsne⟩ := guard_false_parts hguard
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
        obtain ⟨_hmajorLevel, hmajorOut⟩ := predsOf_props hstruct hmajorPred
        obtain ⟨_hminorLevel, hminorOut⟩ := predsOf_props hstruct hminorPred
        have hmajor := hroutes major hmajorNode
        have hminor := hroutes minor hminorNode
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
        obtain ⟨_hulevel, huout⟩ := predsOf_props hstruct hu
        have huroutes := hroutes u hunode
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
      simpa [localExpected, hs] using hcombined.trans (by
        intro route
        rw [hflow])
  | intro premise alpha beta =>
      obtain ⟨hguard, _, _, _⟩ := localShape_intro_raw_inv hs
      obtain ⟨hhyp, _⟩ := guard_false_parts hguard
      obtain ⟨hpreds, hform, hpform, hpremiseNode, _hpremiseLevel,
        hpremiseOut, helim⟩ := localShape_intro_inv hstruct hs
      have hpremise := hroutes premise hpremiseNode
      have hcongr := introExpected_congr_flow c (v := v) alpha hstruct
        hpremiseNode hpremiseOut hpremise
      have hpredRoutes : ∀ u ∈ predsOf c.graph v,
          ListSetEq (StoredRoutes c u)
            (WitnessableRoutes c.graph u
              (flowAt c.graph (stdFuel c.graph - 1) u)) := by
        intro u hu
        have hunode := mem_predsOf_mem_nodes hstruct hu
        obtain ⟨_hulevel, huout⟩ := predsOf_props hstruct hu
        have huroutes := hroutes u hunode
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
      simpa [localExpected, hs] using hcombined.trans (by
        intro route
        rw [hflow])
  | invalid =>
      have himpossible := hprod v hv
      unfold productionAtB at himpossible
      simp [hs] at himpossible


theorem root_has_empty_flowRoute_of_localChecks (d : Graph)
    (c : Certificate) (expanded : ExpandedCert d c)
    (hlocal : LocalChecks d c)
    {vertex : Vertex} (vertex_mem : vertex ∈ d.NODES)
    (root : (get_rule.outgoing vertex d).isEmpty = true) :
    ([], []) ∈ flowAt d (stdFuel d) vertex := by
  have vertex_mem_c : vertex ∈ c.graph.NODES := by
    simpa [hlocal.1] using vertex_mem
  have root_c : (get_rule.outgoing vertex c.graph).isEmpty = true := by
    simpa [hlocal.1] using root
  obtain ⟨output, output_mem, output_node, output_dep, output_residual⟩ :=
    (hlocal.2.2.2.2.1 vertex vertex_mem_c root_c).1
  have stored : ([], []) ∈ StoredRoutes c vertex := by
    unfold StoredRoutes outputsAt
    rw [List.mem_eraseDups]
    apply List.mem_append_right
    rw [List.mem_filterMap]
    exact ⟨output, output_mem,
      by simp [output_node, output_dep, output_residual]⟩
  have witnessable := (storedRoutes_setEq_flow d c expanded hlocal
    vertex vertex_mem ([], [])).mp stored
  unfold WitnessableRoutes at witnessable
  rw [List.mem_filter] at witnessable
  exact witnessable.1




theorem universalLocalRootDischarge_rootsDischargedB (d : Graph)
    (c : Certificate) (expanded : ExpandedCert d c)
    (hlocal : LocalChecks d c)
    (huniversal : UniversalLocalRootDischarge c) :
    rootsDischargedB d = true := by
  have rawProduction := localExpected_setEq_flow d c expanded hlocal
  unfold rootsDischargedB
  rw [List.all_eq_true]
  intro vertex vertex_mem
  by_cases root : (get_rule.outgoing vertex d).isEmpty = true
  · simp only [root, Bool.not_true, Bool.false_or, Bool.and_eq_true,
      List.all_eq_true]
    constructor
    · have empty_mem := root_has_empty_flowRoute_of_localChecks d c
        expanded hlocal vertex_mem root
      have nonempty : flowAt d (stdFuel d) vertex ≠ [] := by
        intro empty
        simp [empty] at empty_mem
      simp [nonempty]
    · intro route route_mem
      have vertex_mem_c : vertex ∈ c.graph.NODES := by
        simpa [hlocal.1] using vertex_mem
      have root_c : (get_rule.outgoing vertex c.graph).isEmpty = true := by
        simpa [hlocal.1] using root
      obtain ⟨stored_empty, expected_residual_empty⟩ :=
        huniversal vertex vertex_mem_c root_c
      have expected_mem : route ∈ localExpected c vertex :=
        (rawProduction vertex vertex_mem route).mpr route_mem
      have residual_empty := expected_residual_empty route expected_mem
      have continuable : routeContinuableAtB d vertex route.2 = true := by
        simp [routeContinuableAtB, root, residual_empty]
      have witnessable : route ∈ WitnessableRoutes d vertex
          (flowAt d (stdFuel d) vertex) := by
        unfold WitnessableRoutes
        rw [List.mem_filter]
        exact ⟨route_mem, continuable⟩
      have stored_mem : route ∈ StoredRoutes c vertex :=
        (storedRoutes_setEq_flow d c expanded hlocal
          vertex vertex_mem route).mpr witnessable
      obtain ⟨dependency_empty, residual_empty'⟩ :=
        stored_empty route stored_mem
      simp [dependency_empty, residual_empty']
  · have nonroot : (get_rule.outgoing vertex d).isEmpty = false :=
      Bool.eq_false_of_not_eq_true root
    simp [nonroot]

def StrongLocalChecks (d : Graph) (c : Certificate) : Prop :=
  LocalChecks d c ∧ UniversalLocalRootDischarge c

def StrongFormationValid (d : Graph) (c : Certificate) : Prop :=
  ExpandedCert d c ∧ StrongLocalChecks d c

def strongLocalChecksB (d : Graph) (c : Certificate) : Bool :=
  localChecksB d c && universalLocalRootDischargeB c

def strongFormationValidB (d : Graph) (c : Certificate) : Bool :=
  expandedCertB d c && strongLocalChecksB d c

theorem strongLocalChecksB_sound (d : Graph) (c : Certificate)
    (expanded : ExpandedCert d c)
    (checked : strongLocalChecksB d c = true) :
    StrongLocalChecks d c := by
  rw [strongLocalChecksB, Bool.and_eq_true] at checked
  exact ⟨localChecksB_sound d c expanded checked.1,
    (universalLocalRootDischargeB_eq_true_iff c).mp checked.2⟩

theorem strongFormationValidB_sound (d : Graph) (c : Certificate)
    (checked : strongFormationValidB d c = true) :
    StrongFormationValid d c := by
  rw [strongFormationValidB, Bool.and_eq_true] at checked
  have expanded := expandedCertB_sound d c checked.1
  exact ⟨expanded, strongLocalChecksB_sound d c expanded checked.2⟩

theorem strongFormationValid_rootsDischargedB (d : Graph)
    (c : Certificate) (valid : StrongFormationValid d c) :
    rootsDischargedB d = true :=
  universalLocalRootDischarge_rootsDischargedB d c valid.1
    valid.2.1 valid.2.2


end SelfCertifying
