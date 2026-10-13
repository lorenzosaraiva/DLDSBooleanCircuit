import Evaluator.Occurrence.Relation
import Certificate.Flow.Soundness

/-!
# Local certificates for the occurrence-aware HC successor

This is a parallel successor certificate position.  It does not reuse the
historical exclusive-hypothesis `localShape`.  A position supplies the exact
ordered predecessor table and claims a set of local outputs.  The checker runs
the dependency-recomputing successor kernel and compares the result with the
claims.

Stored dependencies are therefore claims being checked.  They are not copied
into the conclusion: introduction removes an antecedent and elimination takes
`depUnion` through `occurrenceStep`, whose independent relational refinement is
used by the soundness theorem below.  Global coherence between premise claims
and recursively justified routes is a separate, still-required layer.
-/

open DLDS FlowSemantics
open Evaluator.Mixed Evaluator.Occurrence
open Evaluator.Occurrence.Relation
open Certificate.Flow.Foundation Certificate.Flow

namespace Certificate.Occurrence

structure LocalRouteCertificate where
  node : Vertex
  premiseTable : List (Vertex × List FlowPair)
  claimedRoutes : List FlowPair
  deriving Repr

def predecessorKeys (certificate : LocalRouteCertificate) : List Vertex :=
  certificate.premiseTable.map Prod.fst

/-- A local certificate position checks exact predecessor order, successful
logical evaluation, and set equality of all claimed routes. -/
def localRouteCertificateB (d : Graph)
    (certificate : LocalRouteCertificate) : Bool :=
  let keysOK := predecessorKeys certificate == predsOf d certificate.node
  match occurrenceStep d certificate.node
      (Evaluator.Occurrence.routesFor certificate.premiseTable) with
  | .error _ => false
  | .ok outputs => keysOK && setEqB outputs certificate.claimedRoutes

def LocalRouteCertificateSound (d : Graph)
    (certificate : LocalRouteCertificate) : Prop :=
  predecessorKeys certificate = predsOf d certificate.node ∧
    ∃ routes : RoutePredicate,
      referenceStep d certificate.node
        (fun premise route =>
          route ∈ Evaluator.Occurrence.routesFor
            certificate.premiseTable premise) = .success routes ∧
      ∀ route, route ∈ certificate.claimedRoutes ↔ routes route

/-- Executable local soundness.  It proves predecessor-table fidelity and both
directions of claimed-route membership in the independent reference relation. -/
theorem localRouteCertificateB_sound
    (d : Graph) (certificate : LocalRouteCertificate)
    (valid : localRouteCertificateB d certificate = true) :
    LocalRouteCertificateSound d certificate := by
  unfold localRouteCertificateB at valid
  cases stepEq : occurrenceStep d certificate.node
      (Evaluator.Occurrence.routesFor certificate.premiseTable) with
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
        (Evaluator.Occurrence.routesFor certificate.premiseTable)
      rw [stepEq] at refinement
      cases referenceEq : referenceStep d certificate.node
          (fun premise route =>
            route ∈ Evaluator.Occurrence.routesFor
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

/-- The local generator does not evaluate the whole graph.  It packages one
supplied predecessor table and the outputs independently recomputed from it. -/
def generateLocalRouteCertificate (d : Graph) (node : Vertex)
    (premiseTable : List (Vertex × List FlowPair)) : LocalRouteCertificate :=
  { node
    premiseTable
    claimedRoutes :=
      match occurrenceStep d node
          (Evaluator.Occurrence.routesFor premiseTable) with
      | .error _ => []
      | .ok outputs => outputs }

/-- Completeness of one generated position under exactly the two facts a
caller must establish: the table has the graph's predecessor order and the
local successor step succeeds. -/
theorem generateLocalRouteCertificate_complete
    (d : Graph) (node : Vertex)
    (premiseTable : List (Vertex × List FlowPair))
    (keys : premiseTable.map Prod.fst = predsOf d node)
    (outputs : List FlowPair)
    (success : occurrenceStep d node
      (Evaluator.Occurrence.routesFor premiseTable) = .ok outputs) :
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
      (Evaluator.Occurrence.routesFor premiseTable) = .ok outputs) :
    LocalRouteCertificateSound d
      (generateLocalRouteCertificate d node premiseTable) :=
  localRouteCertificateB_sound d _
    (generateLocalRouteCertificate_complete d node premiseTable
      keys outputs success)

end Certificate.Occurrence
