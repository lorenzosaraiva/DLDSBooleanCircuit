import Infrastructure.HCSuccessorFixedGraphReference
import Infrastructure.SelfCertifying

















open Semantic FlowSpec
open MixedRuleReplay OccurrenceAwareReplay
open HCSuccessorFixedGraphReference
open CertificateModel SelfCertifying

namespace HCSuccessorLocalCertificate

structure LocalRouteCertificate where
  node : Vertex
  premiseTable : List (Vertex × List FlowPair)
  claimedRoutes : List FlowPair
  deriving Repr

def predecessorKeys (certificate : LocalRouteCertificate) : List Vertex :=
  certificate.premiseTable.map Prod.fst



def localRouteCertificateB (d : Graph)
    (certificate : LocalRouteCertificate) : Bool :=
  let keysOK := predecessorKeys certificate == predsOf d certificate.node
  match occurrenceStep d certificate.node
      (OccurrenceAwareReplay.routesFor certificate.premiseTable) with
  | .error _ => false
  | .ok outputs => keysOK && setEqB outputs certificate.claimedRoutes

def LocalRouteCertificateSound (d : Graph)
    (certificate : LocalRouteCertificate) : Prop :=
  predecessorKeys certificate = predsOf d certificate.node ∧
    ∃ routes : RoutePredicate,
      referenceStep d certificate.node
        (fun premise route =>
          route ∈ OccurrenceAwareReplay.routesFor
            certificate.premiseTable premise) = .success routes ∧
      ∀ route, route ∈ certificate.claimedRoutes ↔ routes route



theorem localRouteCertificateB_sound
    (d : Graph) (certificate : LocalRouteCertificate)
    (valid : localRouteCertificateB d certificate = true) :
    LocalRouteCertificateSound d certificate := by
  unfold localRouteCertificateB at valid
  cases stepEq : occurrenceStep d certificate.node
      (OccurrenceAwareReplay.routesFor certificate.premiseTable) with
  | error reason => simp [stepEq] at valid
  | ok outputs =>
      have parts :
          (predecessorKeys certificate == predsOf d certificate.node) = true ∧
          setEqB outputs certificate.claimedRoutes = true := by
        simpa [stepEq] using valid
      have keysEqual : predecessorKeys certificate =
          predsOf d certificate.node := by
        simpa using parts.1
      have outputClaims : ListSetEq outputs certificate.claimedRoutes :=
        setEqB_sound parts.2
      have refinement := occurrenceStep_matches_reference d certificate.node
        (OccurrenceAwareReplay.routesFor certificate.premiseTable)
      rw [stepEq] at refinement
      cases referenceEq : referenceStep d certificate.node
          (fun premise route =>
            route ∈ OccurrenceAwareReplay.routesFor
              certificate.premiseTable premise) with
      | error referenceError =>
          simp [MatchesReference, referenceEq] at refinement
      | success routes =>
          have outputReference : ∀ route,
              route ∈ outputs ↔ routes route := by
            simpa [MatchesReference, referenceEq] using refinement
          refine ⟨keysEqual, routes, referenceEq, ?_⟩
          intro route
          exact (outputClaims route).symm.trans (outputReference route)



def generateLocalRouteCertificate (d : Graph) (node : Vertex)
    (premiseTable : List (Vertex × List FlowPair)) : LocalRouteCertificate :=
  { node
    premiseTable
    claimedRoutes :=
      match occurrenceStep d node
          (OccurrenceAwareReplay.routesFor premiseTable) with
      | .error _ => []
      | .ok outputs => outputs }




theorem generateLocalRouteCertificate_complete
    (d : Graph) (node : Vertex)
    (premiseTable : List (Vertex × List FlowPair))
    (keys : premiseTable.map Prod.fst = predsOf d node)
    (outputs : List FlowPair)
    (success : occurrenceStep d node
      (OccurrenceAwareReplay.routesFor premiseTable) = .ok outputs) :
    localRouteCertificateB d
      (generateLocalRouteCertificate d node premiseTable) = true := by
  simp [localRouteCertificateB, generateLocalRouteCertificate,
    predecessorKeys, keys, success, setEqB]

theorem generatedLocalRouteCertificate_sound
    (d : Graph) (node : Vertex)
    (premiseTable : List (Vertex × List FlowPair))
    (keys : premiseTable.map Prod.fst = predsOf d node)
    (outputs : List FlowPair)
    (success : occurrenceStep d node
      (OccurrenceAwareReplay.routesFor premiseTable) = .ok outputs) :
    LocalRouteCertificateSound d
      (generateLocalRouteCertificate d node premiseTable) :=
  localRouteCertificateB_sound d _
    (generateLocalRouteCertificate_complete d node premiseTable
      keys outputs success)

end HCSuccessorLocalCertificate
