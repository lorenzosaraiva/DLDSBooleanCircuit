import Infrastructure.HCSuccessorActualFuel

















open Semantic FlowSpec OccurrenceAwareReplay SelfCertifying
open HCSuccessorFixedGraphReference HCSuccessorLocalCertificate
open HCSuccessorCertificatePositions HCSuccessorCertificateSoundness
open HCSuccessorCertificateGenerator HCSuccessorActualFuel
open HCSuccessorBranchingGridBridge

namespace HCSuccessorActualCertificateSoundness


theorem actualDomain_levelWellFounded_no_predecessors
    (d : Graph) (domain : ActualPredecessorDomain d)
    (opposite : PredecessorLevelWellFounded d)
    (vertex : Vertex) (member : vertex ∈ d.NODES) :
    predsOf d vertex = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro premise predecessor
  have production := (domain vertex member premise predecessor).2
  have old := (opposite vertex member premise predecessor).2
  omega


theorem countAbove_predecessor_lt
    (d : Graph) (domain : ActualPredecessorDomain d)
    {vertex premise : Vertex} (member : vertex ∈ d.NODES)
    (predecessor : premise ∈ predsOf d vertex) :
    countAbove d premise < countAbove d vertex := by
  obtain ⟨listed, level⟩ := domain vertex member premise predecessor
  exact countAbove_lt_of_level_succ d listed level


theorem countAbove_lt_stdFuel
    (d : Graph) {vertex : Vertex} (member : vertex ∈ d.NODES) :
    countAbove d vertex < stdFuel d := by
  have bound := countAbove_lt_length d member
  unfold stdFuel
  omega

theorem certificateClaims_match_reference_by_countAbove
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (domain : ActualPredecessorDomain d) :
    ∀ measure vertex,
      countAbove d vertex = measure -> vertex ∈ d.NODES ->
      ∀ fuel, measure < fuel ->
        ∃ claims routes,
          claimedRoutesAt? certificate vertex = some claims ∧
          referenceReplayAt d fuel vertex = .success routes ∧
          ClaimsMatch claims routes := by
  intro measure
  induction measure using Nat.strong_induction_on with
  | h measure ih =>
      intro vertex vertexMeasure vertexMember fuel fuelBound
      obtain ⟨localCertificate, localMember, localNode, claimsLookup⟩ :=
        localCertificateAt_of_mem d certificate formation vertex vertexMember
      have localSound := formation.localSound localCertificate localMember
      obtain ⟨localKeys, localRoutes, localStep, localClaims⟩ := localSound
      cases fuel with
      | zero => omega
      | succ predecessorFuel =>
          have buildPremises : ∀ table : List (Vertex × List FlowPair),
              (∀ entry, entry ∈ table ->
                entry ∈ localCertificate.premiseTable) ->
              ∃ referenceTable,
                referencePremises (referenceReplayAt d predecessorFuel)
                    (table.map Prod.fst) = .ok referenceTable ∧
                PremiseTableMatches table referenceTable := by
            intro table subset
            induction table with
            | nil => exact ⟨[], rfl, .nil⟩
            | cons entry tail tailIH =>
                rcases entry with ⟨premise, premiseClaims⟩
                have entryMember : (premise, premiseClaims) ∈
                    localCertificate.premiseTable :=
                  subset (premise, premiseClaims) (by simp)
                have premiseInKeys : premise ∈
                    predecessorKeys localCertificate := by
                  unfold predecessorKeys
                  exact List.mem_map.mpr
                    ⟨(premise, premiseClaims), entryMember, rfl⟩
                have premisePred : premise ∈ predsOf d vertex := by
                  rw [← localNode, ← localKeys]
                  exact premiseInKeys
                obtain ⟨premiseMember, premiseLevel⟩ :=
                  domain vertex vertexMember premise premisePred
                have premiseMeasure : countAbove d premise < countAbove d vertex :=
                  countAbove_lt_of_level_succ d premiseMember premiseLevel
                have premiseFuel : countAbove d premise < predecessorFuel := by
                  omega
                obtain ⟨recursiveClaims, premiseRoutes, recursiveLookup,
                    premiseReplay, recursiveMatch⟩ :=
                  ih (countAbove d premise) (by omega) premise rfl premiseMember
                    predecessorFuel premiseFuel
                obtain ⟨coherentClaims, coherentLookup,
                    premiseCoherent⟩ :=
                  formation.premiseCoherent localCertificate localMember
                    (premise, premiseClaims) entryMember
                have claimsEqual : coherentClaims = recursiveClaims := by
                  rw [recursiveLookup] at coherentLookup
                  exact (Option.some.inj coherentLookup).symm
                subst coherentClaims
                have headMatch : ∀ route,
                    route ∈ premiseClaims ↔ premiseRoutes route := by
                  intro route
                  exact (premiseCoherent route).trans
                    (recursiveMatch route)
                obtain ⟨tailReference, tailReplay, tailMatch⟩ :=
                  tailIH (by
                    intro tailEntry tailMember
                    exact subset tailEntry (by simp [tailMember]))
                refine ⟨(premise, premiseRoutes) :: tailReference, ?_,
                  .cons premise premiseClaims premiseRoutes tail
                    tailReference headMatch tailMatch⟩
                simp [referencePremises, premiseReplay, tailReplay]
          obtain ⟨referenceTable, premisesReplay, tablesMatch⟩ :=
            buildPremises localCertificate.premiseTable
              (fun entry member => member)
          have premisesReplayAtVertex :
              referencePremises (referenceReplayAt d predecessorFuel)
                  (predsOf d vertex) = .ok referenceTable := by
            rw [← localNode, ← localKeys]
            exact premisesReplay
          have stepCongruence :
              referenceStep d vertex
                  (fun premise route =>
                    route ∈ OccurrenceAwareReplay.routesFor
                      localCertificate.premiseTable premise) =
                referenceStep d vertex
                  (referenceRoutesFor referenceTable) :=
            referenceStep_congr d vertex _ _ tablesMatch.routesFor
          have referenceStepSuccess :
              referenceStep d vertex (referenceRoutesFor referenceTable) =
                .success localRoutes := by
            rw [← stepCongruence, ← localNode]
            exact localStep
          refine ⟨localCertificate.claimedRoutes, localRoutes,
            claimsLookup, ?_, localClaims⟩
          simp [referenceReplayAt, premisesReplayAtVertex,
            referenceStepSuccess]



theorem certificateClaims_match_reference_of_actualDomain
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (domain : ActualPredecessorDomain d)
    (vertex : Vertex) (member : vertex ∈ d.NODES)
    (fuel : Nat) (fuelBound : countAbove d vertex < fuel) :
    ∃ claims routes,
      claimedRoutesAt? certificate vertex = some claims ∧
      referenceReplayAt d fuel vertex = .success routes ∧
      ClaimsMatch claims routes :=
  certificateClaims_match_reference_by_countAbove d certificate formation
    domain (countAbove d vertex) vertex rfl member fuel fuelBound

theorem formation_referenceRootsComplete_of_actualDomain
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (domain : ActualPredecessorDomain d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> countAbove d root < fuel) :
    ReferenceRootsComplete d fuel := by
  refine ⟨formation.rootsNonempty, ?_⟩
  intro root rootMember
  have rootNode : root ∈ d.NODES := by
    simp [occurrenceRoots] at rootMember
    exact rootMember.1
  obtain ⟨claims, routes, claimsLookup, replay, claimsMatch⟩ :=
    certificateClaims_match_reference_of_actualDomain d certificate formation domain
      root rootNode fuel (rootFuel root rootMember)
  obtain ⟨rootClaims, rootLookup, rootNonempty, rootClosed⟩ :=
    formation.rootsComplete root rootMember
  have claimsEqual : rootClaims = claims := by
    rw [claimsLookup] at rootLookup
    exact (Option.some.inj rootLookup).symm
  subst rootClaims
  rw [replay]
  constructor
  · obtain ⟨route, routeMember⟩ :=
      List.exists_mem_of_ne_nil claims rootNonempty
    exact ⟨route, (claimsMatch route).mp routeMember⟩
  · intro route routeReference
    exact rootClosed route ((claimsMatch route).mpr routeReference)

theorem successorFormationB_referenceRootsComplete_of_actualDomain
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true)
    (domain : ActualPredecessorDomain d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> countAbove d root < fuel) :
    ReferenceRootsComplete d fuel :=
  formation_referenceRootsComplete_of_actualDomain d certificate
    (successorFormationB_sound d certificate valid) domain fuel rootFuel

theorem successorFormationB_occurrenceRootsComplete_of_actualDomain
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true)
    (domain : ActualPredecessorDomain d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> countAbove d root < fuel) :
    occurrenceRootsCompleteB d fuel = true :=
  (occurrenceRootsCompleteB_iff_reference d fuel).mpr
    (successorFormationB_referenceRootsComplete_of_actualDomain d certificate valid
      domain fuel rootFuel)

theorem successorFormationB_compiledGridValid_of_actualDomain
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true)
    (domain : ActualPredecessorDomain d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> countAbove d root < fuel) :
    ReferenceCompiledGridValid d fuel :=
  (occurrenceRootsCompleteB_iff_compiledGridValid d fuel).mp
    (successorFormationB_occurrenceRootsComplete_of_actualDomain d certificate valid
      domain fuel rootFuel)

theorem noSemanticMarked_compiledGridValid_of_actualDomain
    (d : Graph) (certificate : SuccessorCertificate)
    (noMarked : ∀ position,
      position ∈ semanticPositions d certificate ->
        semanticMarkedAtB d certificate position = false)
    (domain : ActualPredecessorDomain d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> countAbove d root < fuel) :
    ReferenceCompiledGridValid d fuel :=
  successorFormationB_compiledGridValid_of_actualDomain d certificate
    ((no_semantic_marked_iff_formation d certificate).mp noMarked)
    domain fuel rootFuel


theorem successorFormationB_occurrenceRootsComplete_stdFuel
    (d : Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain d)
    (valid : successorFormationB d certificate = true) :
    occurrenceRootsCompleteB d (stdFuel d) = true := by
  apply successorFormationB_occurrenceRootsComplete_of_actualDomain d certificate
    valid domain (stdFuel d)
  intro root member
  exact countAbove_lt_stdFuel d (List.mem_filter.mp member).1

theorem successorFormationB_compiledGridValid_stdFuel
    (d : Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain d)
    (valid : successorFormationB d certificate = true) :
    ReferenceCompiledGridValid d (stdFuel d) :=
  (occurrenceRootsCompleteB_iff_compiledGridValid d (stdFuel d)).mp
    (successorFormationB_occurrenceRootsComplete_stdFuel d certificate domain valid)

theorem noSemanticMarked_occurrenceRootsComplete_stdFuel
    (d : Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain d)
    (noMarked : ∀ position, position ∈ semanticPositions d certificate ->
      semanticMarkedAtB d certificate position = false) :
    occurrenceRootsCompleteB d (stdFuel d) = true :=
  successorFormationB_occurrenceRootsComplete_stdFuel d certificate domain
    ((no_semantic_marked_iff_formation d certificate).mp noMarked)

theorem noSemanticMarked_compiledGridValid_stdFuel
    (d : Graph) (certificate : SuccessorCertificate)
    (domain : ActualPredecessorDomain d)
    (noMarked : ∀ position, position ∈ semanticPositions d certificate ->
      semanticMarkedAtB d certificate position = false) :
    ReferenceCompiledGridValid d (stdFuel d) :=
  successorFormationB_compiledGridValid_stdFuel d certificate domain
    ((no_semantic_marked_iff_formation d certificate).mp noMarked)



theorem occurrenceRootsComplete_iff_exists_certificate_of_connected_actualDomain
    (d : Graph) (domain : ActualPredecessorDomain d)
    (root : Vertex) (rootAt : HorizontalCompressionInput.RootAt d root)
    (reaches : HorizontalCompressionInput.EveryNodeReachesRoot d root) :
    occurrenceRootsCompleteB d (stdFuel d) = true ↔
      ∃ certificate, successorFormationB d certificate = true := by
  constructor
  · intro complete
    exact ⟨generateSuccessorCertificate d (stdFuel d),
      generateSuccessorCertificate_complete_of_connected_actualDomain
        d domain root rootAt reaches complete⟩
  · rintro ⟨certificate, valid⟩
    exact successorFormationB_occurrenceRootsComplete_stdFuel d certificate domain valid

end HCSuccessorActualCertificateSoundness
