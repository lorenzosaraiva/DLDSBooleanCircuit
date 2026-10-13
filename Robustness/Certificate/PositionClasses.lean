import Certificate.Encoding.Canonical

/-!
Kind-based classes of the existing historical verification positions.
The one-witness decoration check stays rule-level even though its canonical
read bound is global. Collection-level includes the graph envelope, graph
uniqueness checks and the legacy aggregate, which have no separate rule kind.
-/

open DLDS Certificate.Flow.Foundation

namespace CertificateSearch.Robustness

/-- One-edge scope or decoration, one-node scope, or one rule's graph shape.
No test of certificate size or the particular failure is used. -/
def IsRuleLevel : VerificationPosition → Prop
  | .expandedWitnessScope _ _
  | .expandedTerminalScope _ _
  | .structuralEdge _ _
  | .structuralNode _ _
  | .hypothesisSource _ _
  | .witnessFaithful _ _ => True
  | _ => False

/-- Collection and aggregate checks outside the one-rule class. Besides
route collections this includes graph identity/uniqueness and the legacy
degenerate whole-check kind; these exceptions are stated explicitly. -/
def IsRouteSetLevel : VerificationPosition → Prop
  | .expandedWitnessScope _ _
  | .expandedTerminalScope _ _
  | .structuralEdge _ _
  | .structuralNode _ _
  | .hypothesisSource _ _
  | .witnessFaithful _ _ => False
  | _ => True

theorem position_classes_exhaustive (p : VerificationPosition) :
    IsRuleLevel p ∨ IsRouteSetLevel p := by
  cases p <;> simp [IsRuleLevel, IsRouteSetLevel]

theorem position_classes_disjoint (p : VerificationPosition) :
    ¬ (IsRuleLevel p ∧ IsRouteSetLevel p) := by
  cases p <;> simp [IsRuleLevel, IsRouteSetLevel]

/-- The graph/embedded-scope subcase of the rule class, excluding the
decoration check that reads routes in the mutable certificate. -/
def IsGraphRule : VerificationPosition → Prop
  | .expandedWitnessScope _ _
  | .expandedTerminalScope _ _
  | .structuralEdge _ _
  | .structuralNode _ _
  | .hypothesisSource _ _ => True
  | _ => False

theorem graphRule_is_ruleLevel (p : VerificationPosition)
    (hp : IsGraphRule p) : IsRuleLevel p := by
  cases p <;> simp_all [IsGraphRule, IsRuleLevel]

/-- This constant counts mutable canonical coordinates, not graph work. -/
def graphRuleExponent (_p : VerificationPosition) (_d : Graph)
    (_c : Certificate) : Nat := 0

theorem graphRule_readBound (d : Graph) (c : Certificate)
    (p : VerificationPosition) (_hp : p ∈ semanticPositions d c)
    (hgraph : IsGraphRule p) :
    canonicalSemanticReadBound d c p ≤ graphRuleExponent p d c := by
  cases p <;>
    simp_all [IsGraphRule, canonicalSemanticReadBound, graphRuleExponent]

theorem graphRule_failure_density (d : Graph) (c : Certificate)
    (invalid : ¬ FormationValid d c)
    (hgraph : ∀ p ∈ semanticPositions d c,
      SemanticMarkedAt d c p → IsGraphRule p) :
    ∃ p ∈ semanticPositions d c, SemanticMarkedAt d c p ∧
      (1 : ℚ) / Fintype.card (CanonicalAlphabet d c) ^
        graphRuleExponent p d c ≤
          canonicalSemanticMalformationDensity d c p := by
  obtain ⟨p, member, marked, _, _, density⟩ :=
    formation_invalid_semantic_amplification d c invalid
  have bound := graphRule_readBound d c p member (hgraph p member marked)
  have zero : canonicalSemanticReadBound d c p = 0 := by
    simpa [graphRuleExponent] using bound
  exact ⟨p, member, marked, by simpa [zero, graphRuleExponent] using density⟩

/-- Adding terminal records grows the advertised exponent of the same
one-edge decoration kind, without changing its graph or edge-witness list. -/
theorem witnessFaithful_readBound_unbounded (d : Graph) (c : Certificate)
    (slot : Nat) (w : EdgeRouteWitness) (output : TerminalOutput) (n : Nat) :
    n ≤ canonicalSemanticReadBound d
      { c with terminalOutputs := c.terminalOutputs ++ List.replicate n output }
      (.witnessFaithful slot w) := by
  have contribution : n ≤ n * (1 + terminalFieldWidth output) := by
    exact Nat.le_mul_of_pos_right n (by omega)
  simp only [canonicalSemanticReadBound, certificateFieldBudget,
    graphFieldBudget, witnessFieldBudget, terminalFieldBudget,
    List.map_append, List.sum_append, List.map_replicate, List.sum_replicate,
    nsmul_eq_mul, Nat.cast_id]
  omega

theorem localPositions_append_terminals (d : Graph) (c : Certificate)
    (extra : List TerminalOutput) :
    localSemanticPositions d { c with terminalOutputs := c.terminalOutputs ++ extra } =
      localSemanticPositions d c := rfl

theorem registered_witnessFaithful_readBound_unbounded
    (d : Graph) (c : Certificate) (slot : Nat) (w : EdgeRouteWitness)
    (output : TerminalOutput) (n : Nat)
    (member : CertificateSearch.VerificationPosition.witnessFaithful slot w ∈ localSemanticPositions d c) :
    CertificateSearch.VerificationPosition.witnessFaithful slot w ∈ semanticPositions d
        { c with terminalOutputs := c.terminalOutputs ++ List.replicate n output } ∧
      n ≤ canonicalSemanticReadBound d
        { c with terminalOutputs := c.terminalOutputs ++ List.replicate n output }
        (.witnessFaithful slot w) := by
  refine ⟨List.mem_append_right _ ?_,
    witnessFaithful_readBound_unbounded d c slot w output n⟩
  rw [localPositions_append_terminals d c _]
  exact member

theorem remoteTerminals_outputsAt_unchanged (c : Certificate) (v : Vertex)
    (output : TerminalOutput) (n : Nat) (remote : output.node ≠ v) :
    outputsAt { c with terminalOutputs := c.terminalOutputs ++ List.replicate n output } v =
      outputsAt c v := by
  simp [outputsAt, remote]

theorem witnessFaithful_no_uniform_graph_bound (d : Graph) (c : Certificate)
    (slot : Nat) (w : EdgeRouteWitness) (output : TerminalOutput) :
    ¬ ∃ bound : Nat, ∀ n : Nat,
      canonicalSemanticReadBound d
        { c with terminalOutputs := c.terminalOutputs ++ List.replicate n output }
        (.witnessFaithful slot w) ≤ bound := by
  rintro ⟨bound, bounded⟩
  have grows := witnessFaithful_readBound_unbounded d c slot w output (bound + 1)
  have upper := bounded (bound + 1)
  omega


end CertificateSearch.Robustness
