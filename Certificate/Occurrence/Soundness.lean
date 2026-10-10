import Certificate.Occurrence.Positions
import ProofSearch.OccurrenceCorrespondence

/-!
# Global soundness of coherent successor certificates

The local checker becomes a recursive fixed-graph soundness theorem when
listed predecessors strictly decrease a natural-valued measure. The original
specialization to smaller predecessor `LEVEL` is retained with its exact
hypotheses; it has the opposite orientation from production ND/HC edges.
Production `countAbove` corollaries are provided downstream in
`HCSuccessorProductionCertificateSoundness`, using `HCSuccessorActualFuel`.

This is fixed-graph verification correspondence. It does not prove that the
public compressor preserves structural assumptions or produces passing
certificates.
-/

open DLDS Flow
open Evaluator.Occurrence
open Evaluator.Occurrence.Relation Certificate.Occurrence
open Certificate.Occurrence.Positions
open ProofSearch.OccurrenceCorrespondence

namespace Certificate.Occurrence.Soundness

def PredecessorLevelWellFounded (d : Graph) : Prop :=
  ∀ vertex, vertex ∈ d.NODES -> ∀ premise,
    premise ∈ predsOf d vertex ->
      premise ∈ d.NODES ∧ premise.LEVEL < vertex.LEVEL

def ClaimsMatch (claims : List FlowPair) (routes : RoutePredicate) : Prop :=
  ∀ route, route ∈ claims ↔ routes route

theorem localCertificateAt_of_mem
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (vertex : Vertex) (member : vertex ∈ d.NODES) :
    ∃ localCertificate,
      localCertificate ∈ certificate.positions ∧
      localCertificate.node = vertex ∧
      claimedRoutesAt? certificate vertex =
        some localCertificate.claimedRoutes := by
  have mapped : vertex ∈
      certificate.positions.map Certificate.Occurrence.LocalRouteCertificate.node := by
    rw [formation.nodeOrder]
    exact member
  obtain ⟨candidate, candidateMember, candidateNode⟩ :=
    List.mem_map.mp mapped
  have foundSome :
      (certificate.positions.find? fun position =>
        position.node == vertex).isSome = true := by
    rw [List.find?_isSome]
    refine ⟨candidate, candidateMember, ?_⟩
    simpa using candidateNode
  cases foundEq : certificate.positions.find? (fun position =>
      position.node == vertex) with
  | none => simp [foundEq] at foundSome
  | some found =>
      have foundMember : found ∈ certificate.positions :=
        List.mem_of_find?_eq_some foundEq
      have foundNode : found.node = vertex := by
        have tested := List.find?_some foundEq
        simpa using tested
      refine ⟨found, foundMember, foundNode, ?_⟩
      simp [claimedRoutesAt?, foundEq]

/-- Orientation-independent induction: the measure supplies both recursive
descent and sufficient fuel. Local certificate semantics are unchanged. -/
theorem certificateClaims_match_reference_by_measure
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (measure : Vertex -> Nat)
    (wellFounded : ∀ vertex, vertex ∈ d.NODES -> ∀ premise,
      premise ∈ predsOf d vertex ->
        premise ∈ d.NODES ∧ measure premise < measure vertex) :
    ∀ level vertex,
      measure vertex = level -> vertex ∈ d.NODES ->
      ∀ fuel, level < fuel ->
        ∃ claims routes,
          claimedRoutesAt? certificate vertex = some claims ∧
          referenceReplayAt d fuel vertex = .success routes ∧
          ClaimsMatch claims routes := by
  intro level
  induction level using Nat.strong_induction_on with
  | h level ih =>
      intro vertex vertexLevel vertexMember fuel fuelBound
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
                  wellFounded vertex vertexMember premise premisePred
                have premiseFuel : measure premise < predecessorFuel := by
                  omega
                obtain ⟨recursiveClaims, premiseRoutes, recursiveLookup,
                    premiseReplay, recursiveMatch⟩ :=
                  ih (measure premise) (by omega) premise rfl premiseMember
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
                    route ∈ Evaluator.Occurrence.routesFor
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

/-- Original smaller-predecessor-level domain, retained unchanged. It is not
the production ND/HC edge orientation. -/
theorem certificateClaims_match_reference_by_level
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (wellFounded : PredecessorLevelWellFounded d) :
    ∀ level vertex,
      vertex.LEVEL = level -> vertex ∈ d.NODES ->
      ∀ fuel, level < fuel ->
        ∃ claims routes,
          claimedRoutesAt? certificate vertex = some claims ∧
          referenceReplayAt d fuel vertex = .success routes ∧
          ClaimsMatch claims routes :=
  certificateClaims_match_reference_by_measure d certificate formation
    Vertex.LEVEL wellFounded

theorem certificateClaims_match_reference
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (wellFounded : PredecessorLevelWellFounded d)
    (vertex : Vertex) (vertexMember : vertex ∈ d.NODES)
    (fuel : Nat) (fuelBound : vertex.LEVEL < fuel) :
    ∃ claims routes,
      claimedRoutesAt? certificate vertex = some claims ∧
      referenceReplayAt d fuel vertex = .success routes ∧
      ClaimsMatch claims routes :=
  certificateClaims_match_reference_by_level d certificate formation
    wellFounded vertex.LEVEL vertex rfl vertexMember fuel fuelBound

/-- Root closure uses only claim correspondence, independently of the
induction measure or edge orientation used to obtain it. -/
theorem formation_referenceRootsComplete_of_claimsMatch
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (fuel : Nat)
    (rootMatch : ∀ root, root ∈ occurrenceRoots d ->
      ∃ claims routes,
        claimedRoutesAt? certificate root = some claims ∧
        referenceReplayAt d fuel root = .success routes ∧
        ClaimsMatch claims routes) :
    ReferenceRootsComplete d fuel := by
  refine ⟨formation.rootsNonempty, ?_⟩
  intro root rootMember
  obtain ⟨claims, routes, claimsLookup, replay, claimsMatch⟩ :=
    rootMatch root rootMember
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

/-- Original root soundness under smaller predecessor levels. The fuel
premise is stated only for roots; strict descent supplies recursive bounds. -/
theorem formation_referenceRootsComplete
    (d : Graph) (certificate : SuccessorCertificate)
    (formation : SuccessorFormationValid d certificate)
    (wellFounded : PredecessorLevelWellFounded d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> root.LEVEL < fuel) :
    ReferenceRootsComplete d fuel := by
  apply formation_referenceRootsComplete_of_claimsMatch d certificate
    formation fuel
  intro root rootMember
  have rootNode : root ∈ d.NODES := (List.mem_filter.mp rootMember).1
  exact certificateClaims_match_reference d certificate formation wellFounded
    root rootNode fuel (rootFuel root rootMember)

theorem successorFormationB_referenceRootsComplete
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true)
    (wellFounded : PredecessorLevelWellFounded d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> root.LEVEL < fuel) :
    ReferenceRootsComplete d fuel :=
  formation_referenceRootsComplete d certificate
    (successorFormationB_sound d certificate valid) wellFounded fuel rootFuel

theorem successorFormationB_occurrenceRootsComplete
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true)
    (wellFounded : PredecessorLevelWellFounded d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> root.LEVEL < fuel) :
    occurrenceRootsCompleteB d fuel = true :=
  (occurrenceRootsCompleteB_iff_reference d fuel).mpr
    (successorFormationB_referenceRootsComplete d certificate valid
      wellFounded fuel rootFuel)

theorem successorFormationB_compiledGridValid
    (d : Graph) (certificate : SuccessorCertificate)
    (valid : successorFormationB d certificate = true)
    (wellFounded : PredecessorLevelWellFounded d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> root.LEVEL < fuel) :
    ReferenceCompiledGridValid d fuel :=
  (occurrenceRootsCompleteB_iff_compiledGridValid d fuel).mp
    (successorFormationB_occurrenceRootsComplete d certificate valid
      wellFounded fuel rootFuel)

theorem noSemanticMarked_compiledGridValid
    (d : Graph) (certificate : SuccessorCertificate)
    (noMarked : ∀ position,
      position ∈ semanticPositions d certificate ->
        semanticMarkedAtB d certificate position = false)
    (wellFounded : PredecessorLevelWellFounded d)
    (fuel : Nat)
    (rootFuel : ∀ root, root ∈ occurrenceRoots d -> root.LEVEL < fuel) :
    ReferenceCompiledGridValid d fuel :=
  successorFormationB_compiledGridValid d certificate
    ((no_semantic_marked_iff_formation d certificate).mp noMarked)
    wellFounded fuel rootFuel

end Certificate.Occurrence.Soundness
