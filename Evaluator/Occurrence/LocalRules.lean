import DLDS.Compression.ClassIteration
import DLDS.Compression.EdgeTraces

/-!
# Adopted successor-Flow specification interface

This module states the local route rules used to specify the occurrence-aware
successor independently of `occurrenceStep` and `occurrenceReplayAt`.

The specification deliberately adopts edge-colour traces.  It is not a claim
that the paper's branch-address presentation and edge traces are globally
interchangeable, and it does not identify the successor with historical
`flowAt`.  In particular, source and derived productions may coexist at a
mixed vertex.

The principal refinement theorem below covers a derived, non-reseeding,
occurrence-coloured elimination centre.  Its reference result is stated as an
inductive logical occurrence judgment; successful operational evaluation and
both directions of route membership are conclusions.
-/

open DLDS Flow
open Evaluator.Mixed Evaluator.Occurrence DLDS.Compression.ClassIteration

namespace Evaluator.Occurrence.LocalRules

/-- The graph itself permits the default empty source residual either before
any occurrence trace exists or at a collapsed source with an explicit
colour-zero outgoing use. -/
def SourceDefaultPermitted (d : Graph) (v : Vertex) : Prop :=
  (ancestorsInto d v).isEmpty = true \/
    (v.COLLAPSED = true /\
      Exists fun edge => edge ∈ get_rule.outgoing v d /\ edge.COLOUR = 0)

/-- Independent source rule for the adopted edge-trace interpretation.
Dependencies come from the source formula.  A nonempty residual comes from an
explicit occurrence trace; the empty residual is kept distinct and requires
`SourceDefaultPermitted`. -/
inductive SourceProduction (d : Graph) (v : Vertex) : FlowPair -> Prop
  | traced (ancestor : Ancestral)
      (member : ancestor ∈ ancestorsInto d v) :
      SourceProduction d v
        (normalizeDep [v.FORMULA], ancestor.COLOURS)
  | default (permitted : SourceDefaultPermitted d v) :
      SourceProduction d v (normalizeDep [v.FORMULA], [])

theorem candidateSourceRoutes_iff_sourceProduction
    (d : Graph) (v : Vertex) (route : FlowPair) :
    route ∈ candidateSourceRoutes d v <-> SourceProduction d v route := by
  simp only [candidateSourceRoutes]
  split <;> rename_i permitted
  · rw [List.mem_eraseDups, List.mem_append]
    constructor
    · intro member
      rcases member with traced | default
      · obtain ⟨ancestor, ancestorMember, routeEq⟩ := List.mem_map.mp traced
        rw [← routeEq]
        exact .traced ancestor ancestorMember
      · have routeEq : route = (normalizeDep [v.FORMULA], []) := by
          simpa using default
        rw [routeEq]
        apply Evaluator.Occurrence.LocalRules.SourceProduction.default
        simpa [SourceDefaultPermitted, List.any_eq_true] using permitted
    · intro production
      cases production with
      | traced ancestor member =>
          exact Or.inl (List.mem_map.mpr ⟨ancestor, member, rfl⟩)
      | default allowed => simp
  · rw [List.mem_eraseDups]
    constructor
    · intro member
      obtain ⟨ancestor, ancestorMember, routeEq⟩ := List.mem_map.mp member
      rw [← routeEq]
      exact .traced ancestor ancestorMember
    · intro production
      cases production with
      | traced ancestor member =>
          exact List.mem_map.mpr ⟨ancestor, member, rfl⟩
      | default allowed =>
          exfalso
          apply permitted
          simpa [SourceDefaultPermitted, List.any_eq_true] using allowed

/-- A source leaf is a complete local instance of the reference rule. -/
theorem occurrenceStep_leaf_iff_sourceProduction
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (leaf : predsOf d v = []) (source : v.HYPOTHESIS = true) :
    Exists fun outputs =>
      occurrenceStep d v premiseFlow = .ok outputs /\
      forall route, route ∈ outputs <-> SourceProduction d v route := by
  refine ⟨candidateSourceRoutes d v, ?_, ?_⟩
  · simp [occurrenceStep, leaf, source]
  · exact candidateSourceRoutes_iff_sourceProduction d v

/-- Independent one-premise implication-introduction rule.  The dependency is
computed by removing the discharged antecedent, and the residual is obtained
by consuming the actual premise edge. -/
inductive IntroductionProduction (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : FlowPair -> Prop
  | intro (antecedent consequent : Formula) (premise : Vertex)
      (dependency : Dep) (residual tail : ColourPath) (edge : Deduction)
      (formula : v.FORMULA = antecedent >> consequent)
      (premiseMember : premise ∈ predsOf d v)
      (premiseFormula : premise.FORMULA = consequent)
      (routeMember : (dependency, residual) ∈ premiseFlow premise)
      (edgeMember : edge ∈ edgesBetween d premise v)
      (consumed : consume edge residual = some tail) :
      IntroductionProduction d v premiseFlow
        (depRemove dependency antecedent, tail)

theorem introductionCandidates_iff_introductionProduction
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair) :
    route ∈ introductionCandidates d v premiseFlow <->
      IntroductionProduction d v premiseFlow route := by
  constructor
  · intro member
    unfold introductionCandidates at member
    cases formulaEq : v.FORMULA with
    | atom name => simp [formulaEq] at member
    | implication antecedent consequent =>
        simp only [formulaEq] at member
        obtain ⟨premise, premiseMember, candidateMember⟩ :=
          List.mem_flatMap.mp member
        split at candidateMember <;> rename_i premiseEq
        · simp only [candidateIntroductionOutputs, premiseEq]
            at candidateMember
          rw [formulaEq] at candidateMember
          simp only [ite_true] at candidateMember
          obtain ⟨premiseRoute, premiseRouteMember, afterRoute⟩ :=
            List.mem_flatMap.mp candidateMember
          obtain ⟨edge, edgeMember, resultEq⟩ :=
            List.mem_filterMap.mp afterRoute
          rcases premiseRoute with ⟨dependency, residual⟩
          cases consumed : consume edge residual with
          | none => simp [consumed] at resultEq
          | some tail =>
              simp only [consumed, Option.map_some] at resultEq
              have routeEq : route = (depRemove dependency antecedent, tail) :=
                Option.some.inj resultEq |>.symm
              rw [routeEq]
              exact .intro antecedent consequent premise dependency residual
                tail edge formulaEq premiseMember premiseEq premiseRouteMember
                edgeMember consumed
        · simp at candidateMember
  · intro production
    cases production with
    | intro antecedent consequent premise dependency residual tail edge
        formula premiseMember premiseFormula routeMember edgeMember consumed =>
        unfold introductionCandidates
        rw [formula]
        apply List.mem_flatMap.mpr
        refine ⟨premise, premiseMember, ?_⟩
        simp only [candidateIntroductionOutputs, formula, premiseFormula,
          if_pos]
        apply List.mem_flatMap.mpr
        refine ⟨(dependency, residual), routeMember, ?_⟩
        apply List.mem_filterMap.mpr
        refine ⟨edge, edgeMember, ?_⟩
        simp [consumed]

/-- In the pure-introduction supported domain, the complete successor step is
sound and complete for `IntroductionProduction`.  The absence of elimination
outputs is an explicit continuation-separation condition, not inferred from
dependencies. -/
theorem occurrenceStep_iff_introductionProduction
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (hasPredecessor : (predsOf d v).isEmpty = false)
    (noAncestors : ancestorsInto d v = [])
    (derivedVertex : v.HYPOTHESIS = false)
    (noEliminations : candidateEliminationOutputs d v premiseFlow = [])
    (nonempty : introductionCandidates d v premiseFlow ≠ []) :
    Exists fun outputs =>
      occurrenceStep d v premiseFlow = .ok outputs /\
      forall route, route ∈ outputs <->
        IntroductionProduction d v premiseFlow route := by
  let outputs := (introductionCandidates d v premiseFlow).eraseDups
  have logicalEq : derivedOutputs d v premiseFlow = outputs := by
    simp [derivedOutputs, separatedIntroductionOutputs, noEliminations,
      residualOccursB, outputs]
  have outputsNonempty : outputs.isEmpty = false := by
    rw [List.isEmpty_eq_false_iff]
    obtain ⟨route, member⟩ := List.exists_mem_of_ne_nil _ nonempty
    exact List.ne_nil_of_mem (List.mem_eraseDups.mpr member)
  refine ⟨outputs, ?_, ?_⟩
  · simp [occurrenceStep, hasPredecessor, noAncestors, derivedVertex,
      logicalEq, outputsNonempty]
  · intro route
    change route ∈ (introductionCandidates d v premiseFlow).eraseDups <-> _
    rw [List.mem_eraseDups]
    exact introductionCandidates_iff_introductionProduction d v premiseFlow
      route

/-- Independent Cartesian implication-elimination rule.  The two premise
routes must consume their concrete incoming edges to the same tail.  The
result dependency is recomputed and no stored edge dependency is mentioned.
-/
inductive EliminationProduction (d : Graph) (v : Vertex)
    (premiseFlow : Vertex -> List FlowPair) : FlowPair -> Prop
  | elim (major minor : Vertex)
      (minorDep majorDep : Dep)
      (minorResidual majorResidual tail : ColourPath)
      (minorEdge majorEdge : Deduction)
      (pairMember : (major, minor) ∈ elimPairsAt d v)
      (minorRoute : (minorDep, minorResidual) ∈ premiseFlow minor)
      (majorRoute : (majorDep, majorResidual) ∈ premiseFlow major)
      (minorEdgeMember : minorEdge ∈ edgesBetween d minor v)
      (majorEdgeMember : majorEdge ∈ edgesBetween d major v)
      (minorConsumed : consume minorEdge minorResidual = some tail)
      (majorConsumed : consume majorEdge majorResidual = some tail) :
      EliminationProduction d v premiseFlow
        (depUnion minorDep majorDep, tail)

theorem candidateEliminationOutputs_iff_eliminationProduction
    (d : Graph) (v : Vertex) (premiseFlow : Vertex -> List FlowPair)
    (route : FlowPair) :
    route ∈ candidateEliminationOutputs d v premiseFlow <->
      EliminationProduction d v premiseFlow route := by
  constructor
  · intro member
    unfold candidateEliminationOutputs at member
    obtain ⟨pair, pairMember, afterPair⟩ := List.mem_flatMap.mp member
    rcases pair with ⟨major, minor⟩
    obtain ⟨minorRoute, minorRouteMember, afterMinor⟩ :=
      List.mem_flatMap.mp afterPair
    rcases minorRoute with ⟨minorDep, minorResidual⟩
    obtain ⟨majorRoute, majorRouteMember, afterMajor⟩ :=
      List.mem_flatMap.mp afterMinor
    rcases majorRoute with ⟨majorDep, majorResidual⟩
    obtain ⟨minorEdge, minorEdgeMember, afterMinorEdge⟩ :=
      List.mem_flatMap.mp afterMajor
    obtain ⟨majorEdge, majorEdgeMember, resultEq⟩ :=
      List.mem_filterMap.mp afterMinorEdge
    cases minorConsumed : consume minorEdge minorResidual with
    | none => simp [minorConsumed] at resultEq
    | some minorTail =>
        cases majorConsumed : consume majorEdge majorResidual with
        | none => simp [minorConsumed, majorConsumed] at resultEq
        | some majorTail =>
            by_cases tailsEq : minorTail = majorTail
            · subst majorTail
              simp only [minorConsumed, majorConsumed, ↓reduceIte,
                Option.some.injEq] at resultEq
              rw [← resultEq]
              exact .elim major minor minorDep majorDep minorResidual
                majorResidual minorTail minorEdge majorEdge pairMember
                minorRouteMember majorRouteMember minorEdgeMember
                majorEdgeMember minorConsumed majorConsumed
            · simp [minorConsumed, majorConsumed, tailsEq] at resultEq
  · intro production
    cases production with
    | elim major minor minorDep majorDep minorResidual majorResidual tail
        minorEdge majorEdge pairMember minorRoute majorRoute minorEdgeMember
        majorEdgeMember minorConsumed majorConsumed =>
        unfold candidateEliminationOutputs
        apply List.mem_flatMap.mpr
        refine ⟨(major, minor), pairMember, ?_⟩
        apply List.mem_flatMap.mpr
        refine ⟨(minorDep, minorResidual), minorRoute, ?_⟩
        apply List.mem_flatMap.mpr
        refine ⟨(majorDep, majorResidual), majorRoute, ?_⟩
        apply List.mem_flatMap.mpr
        refine ⟨minorEdge, minorEdgeMember, ?_⟩
        apply List.mem_filterMap.mpr
        refine ⟨majorEdge, majorEdgeMember, ?_⟩
        simp [minorConsumed, majorConsumed]

/-- Independent structural reseeding rule.  It preserves a logically
computed dependency and replaces the residual only with an explicit ancestor
trace extending the active residual. -/
inductive ReseededProduction (d : Graph) (v : Vertex)
    (logical : FlowPair -> Prop) : FlowPair -> Prop
  | reseed (dependency : Dep) (residual : ColourPath)
      (ancestor : Ancestral)
      (logicalRoute : logical (dependency, residual))
      (ancestorMember : ancestor ∈ ancestorsInto d v)
      (prefixes : colourPathPrefixB residual ancestor.COLOURS = true) :
      ReseededProduction d v logical (dependency, ancestor.COLOURS)

theorem reseedDerivedRoutes_iff_reseededProduction
    (d : Graph) (v : Vertex) (routes : List FlowPair)
    (output : FlowPair) :
    output ∈ reseedDerivedRoutes (ancestorsInto d v) routes <->
      ReseededProduction d v (fun route => route ∈ routes) output := by
  constructor
  · intro member
    obtain ⟨ancestor, ancestorMember, route, routeMember, prefixes, outputEq⟩ :=
      (mem_reseedDerivedRoutes_iff (ancestorsInto d v) routes output).mp member
    rw [← outputEq]
    exact .reseed route.1 route.2 ancestor routeMember ancestorMember prefixes
  · intro production
    cases production with
    | reseed dependency residual ancestor logicalRoute ancestorMember prefixes =>
        apply (mem_reseedDerivedRoutes_iff (ancestorsInto d v) routes _).mpr
        exact ⟨ancestor, ancestorMember, (dependency, residual), logicalRoute,
          prefixes, rfl⟩

/-- The routed-elimination reference used by the HC producer interface.  One
proof-side occurrence supplies both premise identities, their independently
established dependencies, their actual incoming colours, and the occurrence
tag that remains after both colours are consumed. -/
inductive RoutedEliminationProduction
    (occurrences : List RoutedElimOccurrence) : FlowPair -> Prop
  | occurrence (routed : RoutedElimOccurrence)
      (member : routed ∈ occurrences) :
      RoutedEliminationProduction occurrences
        (depUnion routed.logical.minorDep routed.logical.majorDep,
          [routed.logical.tag])

/-- A mixed hypothesis/derived centre contributes both occurrence classes.
The source and derived alternatives remain distinct proof constructors even
when their dependency vectors happen to coincide. -/
inductive MixedRoutedProduction (d : Graph) (v : Vertex)
    (occurrences : List RoutedElimOccurrence) : FlowPair -> Prop
  | source (route : FlowPair) (production : SourceProduction d v route) :
      MixedRoutedProduction d v occurrences route
  | derived (route : FlowPair)
      (production : RoutedEliminationProduction occurrences route) :
      MixedRoutedProduction d v occurrences route

theorem expectedRoutedFlowRoutes_iff_reference
    (occurrences : List RoutedElimOccurrence) (route : FlowPair) :
    route ∈ expectedRoutedFlowRoutes occurrences <->
      RoutedEliminationProduction occurrences route := by
  constructor
  · intro member
    obtain ⟨routed, routedMember, routeEq⟩ := List.mem_map.mp member
    rw [← routeEq]
    exact .occurrence routed routedMember
  · intro production
    cases production with
    | occurrence routed member =>
        exact List.mem_map.mpr ⟨routed, member, rfl⟩

/-- Mixed source/derived refinement.  A hypothesis mark does not suppress the
derived occurrence family.  The result uses the graph-defined source rule and
the routed logical reference, and proves successful execution rather than
assuming it. -/
theorem occurrenceStep_iff_mixedRoutedReference
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (nonempty : occurrences ≠ [])
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (centerSource : center.HYPOTHESIS = true)
    (hasPredecessor : (predsOf actual center).isEmpty = false)
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    Exists fun outputs =>
      occurrenceStep actual center premiseFlow = .ok outputs /\
      forall route, route ∈ outputs <->
        MixedRoutedProduction actual center occurrences route := by
  have derivedSet := derivedOutputs_setEq_expected_routed actual center
    occurrences incidence premiseFlow premiseCorrect
  obtain ⟨routed, rest, occurrencesEq⟩ := List.exists_cons_of_ne_nil nonempty
  have routedMember : routed ∈ occurrences := by
    rw [occurrencesEq]
    exact List.Mem.head rest
  have expectedMember :
      (depUnion routed.logical.minorDep routed.logical.majorDep,
        [routed.logical.tag]) ∈ expectedRoutedFlowRoutes occurrences :=
    List.mem_map.mpr ⟨routed, routedMember, rfl⟩
  have derivedMember :
      (depUnion routed.logical.minorDep routed.logical.majorDep,
        [routed.logical.tag]) ∈ derivedOutputs actual center premiseFlow :=
    (derivedSet _).mpr expectedMember
  have derivedNonempty :
      (derivedOutputs actual center premiseFlow).isEmpty = false := by
    rw [List.isEmpty_eq_false_iff]
    exact List.ne_nil_of_mem derivedMember
  let outputs :=
    (candidateSourceRoutes actual center ++
      derivedOutputs actual center premiseFlow).eraseDups
  refine ⟨outputs, ?_, ?_⟩
  · simp [occurrenceStep, hasPredecessor, centerSource, derivedNonempty,
      outputs]
  · intro route
    change route ∈ (candidateSourceRoutes actual center ++
        derivedOutputs actual center premiseFlow).eraseDups <-> _
    rw [List.mem_eraseDups, List.mem_append]
    constructor
    · intro member
      rcases member with sourceMember | derivedMember
      · exact .source route
          ((candidateSourceRoutes_iff_sourceProduction actual center route).mp
            sourceMember)
      · exact .derived route
          ((expectedRoutedFlowRoutes_iff_reference occurrences route).mp
            ((derivedSet route).mp derivedMember))
    · intro production
      cases production with
      | source sourceProduction =>
          exact Or.inl
            ((candidateSourceRoutes_iff_sourceProduction actual center route).mpr
              sourceProduction)
      | derived derivedProduction =>
          exact Or.inr ((derivedSet route).mpr
            ((expectedRoutedFlowRoutes_iff_reference occurrences route).mpr
              derivedProduction))

/-- Local soundness and completeness of the actual successor step against the
independent routed-elimination judgment.  The hypotheses say exactly that the
supplied graph has the routed incidence, its premise semantics is correct,
the centre is an ordinary derived/non-reseeding centre, and at least one
logical occurrence is present.  Error exclusion is proved, not assumed. -/
theorem occurrenceStep_iff_routedEliminationReference
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (nonempty : occurrences ≠ [])
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (centerDerived : center.HYPOTHESIS = false)
    (noAncestors : ancestorsInto actual center = [])
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    Exists fun outputs =>
      occurrenceStep actual center premiseFlow = .ok outputs /\
      forall route, route ∈ outputs <->
        RoutedEliminationProduction occurrences route := by
  obtain ⟨outputs, evaluated, setEq⟩ :=
    occurrenceStep_setEq_expected_routed actual center occurrences nonempty
      incidence centerDerived noAncestors premiseFlow premiseCorrect
  refine ⟨outputs, evaluated, ?_⟩
  intro route
  exact (setEq route).trans
    (expectedRoutedFlowRoutes_iff_reference occurrences route)

/-- Producer-facing corollary: dependency-erased exact replacement incidence
is sufficient for the same semantic refinement. -/
theorem occurrenceStep_iff_routedEliminationReference_of_exactReplacement
    (actual : Graph) (center : Vertex)
    (occurrences : List RoutedElimOccurrence)
    (nonempty : occurrences ≠ [])
    (incidence : ExactRoutedReplacementIncidence actual center occurrences)
    (centerDerived : center.HYPOTHESIS = false)
    (noAncestors : ancestorsInto actual center = [])
    (premiseFlow : Vertex -> List FlowPair)
    (premiseCorrect : RoutedPremiseRoutesCorrect premiseFlow occurrences) :
    Exists fun outputs =>
      occurrenceStep actual center premiseFlow = .ok outputs /\
      forall route, route ∈ outputs <->
        RoutedEliminationProduction occurrences route :=
  occurrenceStep_iff_routedEliminationReference actual center occurrences
    nonempty incidence.toRoutedEliminationIncidence centerDerived noAncestors
    premiseFlow premiseCorrect

/-- Graph-level recursive refinement for the same supported domain.  Premise
replay evidence is required at the predecessor boundary; the centre result
and both membership directions are derived. -/
theorem occurrenceReplayAt_iff_routedEliminationReference
    (actual : Graph) (occurrences : List RoutedElimOccurrence)
    (fuel : Nat) (center : Vertex)
    (incidence : RoutedEliminationIncidence actual center occurrences)
    (nodup : (predsOf actual center).Nodup)
    (evidence : forall vertex, vertex ∈ predsOf actual center ->
      RoutedPremiseEvidence actual occurrences fuel vertex)
    (support : forall vertex,
      expectedRoutedPremiseRoutes occurrences vertex ≠ [] ->
        vertex ∈ predsOf actual center)
    (hasPredecessor : (predsOf actual center).isEmpty = false)
    (noAncestors : ancestorsInto actual center = [])
    (derived : center.HYPOTHESIS = false)
    (expectedNonempty :
      (expectedRoutedFlowRoutes occurrences).isEmpty = false) :
    Exists fun outputs =>
      occurrenceReplayAt actual (fuel + 2) center = .ok outputs /\
      forall route, route ∈ outputs <->
        RoutedEliminationProduction occurrences route := by
  obtain ⟨outputs, replay, setEq⟩ :=
    occurrenceReplayAt_setEq_of_routed_incidence actual occurrences fuel center
      incidence nodup evidence support hasPredecessor noAncestors derived
      expectedNonempty
  refine ⟨outputs, replay, ?_⟩
  intro route
  exact (setEq route).trans
    (expectedRoutedFlowRoutes_iff_reference occurrences route)

end Evaluator.Occurrence.LocalRules
