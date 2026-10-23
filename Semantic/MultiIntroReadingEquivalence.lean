import Semantic.MultiIntroReadingForward
import Semantic.MultiIntroReadingReverse
import Semantic.MultiIntroReadingNaturalDeduction
import Semantic.MultiIntroCertificate

open Semantic FlowSpec SelfCertifying RouteIndexedMultiToken SpineReadings
open FlowBranchingGridBridge
namespace MultiIntroReadings

theorem occurrence_payload (variant : MultiIntro.Variant) (d : Graph) (v : Vertex)
    (tree : ReadingTree) (checked : occurrenceAt variant d v tree = true) :
    ∃ route, payload? variant d tree = some route := by
  simp only [occurrenceAt, Bool.and_eq_true] at checked
  cases h : payload? variant d tree with
  | none => simp [h] at checked
  | some route => exact ⟨route, rfl⟩

theorem keyed_occurrence (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (token : RouteIndexedToken)
    (member : token ∈ MultiIntroReadingLineage.evaluateAt variant d fuel v) :
    occurrenceAt variant d v token.routeKey = true ∧
      payload? variant d token.routeKey = some token.flowPair := by
  obtain ⟨atVertex, payload⟩ := MultiIntroReadingReverse.evaluateAt_payload variant d fuel v token member
  obtain ⟨nd, reconstructed, wellFormed, conclusion, _⟩ :=
    MultiIntroReadingLineage.route_tree variant d fuel v token member
  refine ⟨?_, payload⟩
  simp [occurrenceAt, ndCorrectB, payload, reconstructed, atVertex, wellFormed, conclusion]

theorem routes_eq_standard (variant : MultiIntro.Variant) (d : Graph)
    (domain : MultiIntro.MIDomain d) (v : Vertex) (listed : v ∈ d.NODES)
    (fuel : Nat) (enough : stdFuel d ≤ fuel) :
    MultiIntro.evaluateAt variant d fuel v = MultiIntro.evaluateAt variant d (stdFuel d) v := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le enough
  induction n with
  | zero => simp
  | succ n ih =>
    have stable := MultiIntro.kernel_succ_stable (MultiIntro.kernel variant) d domain v listed
      (stdFuel d + n) (by have bound := countAbove_le d v; unfold stdFuel; omega)
    calc
      MultiIntro.evaluateAt variant d (stdFuel d + (n+1)) v =
          MultiIntro.evaluateAt variant d (stdFuel d + n) v := by
        simpa only [MultiIntro.kernel_evaluate, Nat.add_assoc] using stable
      _ = MultiIntro.evaluateAt variant d (stdFuel d) v := ih (by omega)

theorem occurrence_route_iff (variant : MultiIntro.Variant) (d : Graph)
    (domain : MultiIntro.MIDomain d) (v : Vertex) (listed : v ∈ d.NODES) (route : FlowPair) :
    (∃ tree, occurrenceAt variant d v tree = true ∧ payload? variant d tree = some route) ↔
      route ∈ MultiIntro.evaluateAt variant d (stdFuel d) v := by
  constructor
  · rintro ⟨tree, checked, payload⟩
    have atVertex : vertex tree = v := by
      simp only [occurrenceAt, Bool.and_eq_true] at checked
      simpa using checked.1.1
    obtain ⟨threshold, eventual⟩ := payload_eventually_keyed variant d tree route payload
    obtain ⟨token, member, pair, _⟩ := eventual (stdFuel d + threshold) (by omega)
    rw [atVertex] at member
    have projected : route ∈ projectFlowPairs
        (MultiIntroReadingLineage.evaluateAt variant d (stdFuel d + threshold) v) :=
      List.mem_map.mpr ⟨token, member, pair⟩
    have raw := (MultiIntroReadingLineage.evaluateAt_project variant d (stdFuel d + threshold) v route).mp projected
    rwa [routes_eq_standard variant d domain v listed _ (by omega)] at raw
  · intro member
    obtain ⟨token, present, pair⟩ := List.mem_map.mp
      ((MultiIntroReadingLineage.evaluateAt_project variant d (stdFuel d) v route).mpr member)
    obtain ⟨checked, payload⟩ := keyed_occurrence variant d (stdFuel d) v token present
    exact ⟨token.routeKey, checked, pair ▸ payload⟩

theorem readings_of_valid (variant : MultiIntro.Variant) (d : Graph)
    (valid : MultiIntro.ValidMI variant d) : verdict variant d := by
  have roots := (MultiIntro.rootsB_iff variant d).mp valid.2.2
  constructor
  · intro root listed terminal
    obtain ⟨route, present⟩ := List.exists_mem_of_ne_nil _ (roots root listed terminal).1
    obtain ⟨tree, checked, _⟩ := (occurrence_route_iff variant d valid.1 root listed route).mpr present
    exact ⟨tree, checked⟩
  · intro root listed terminal tree route checked payload
    exact (roots root listed terminal).2 route
      ((occurrence_route_iff variant d valid.1 root listed route).mp ⟨tree, checked, payload⟩)

theorem verdict_roots (variant : MultiIntro.Variant) (d : Graph)
    (domain : MultiIntro.MIDomain d) (accepted : verdict variant d) :
    MultiIntro.rootsB variant d = true := by
  apply (MultiIntro.rootsB_iff variant d).mpr
  intro root listed terminal
  obtain ⟨tree, checked⟩ := accepted.1 root listed terminal
  obtain ⟨route, payload⟩ := occurrence_payload variant d root tree checked
  have present := (occurrence_route_iff variant d domain root listed route).mp ⟨tree, checked, payload⟩
  constructor
  · intro empty
    rw [empty] at present
    contradiction
  · intro raw member
    obtain ⟨reading, readingChecked, pair⟩ := (occurrence_route_iff variant d domain root listed raw).mpr member
    exact accepted.2 root listed terminal reading raw readingChecked pair

theorem valid_iff_readings (variant : MultiIntro.Variant) (d : Graph) :
    MultiIntro.ValidMI variant d ↔
      MultiIntro.MIDomain d ∧ LeveledColored d ∧ verdict variant d := by
  constructor
  · intro valid
    exact ⟨valid.1, valid.2.1, readings_of_valid variant d valid⟩
  · rintro ⟨domain, levels, accepted⟩
    exact ⟨domain, levels, verdict_roots variant d domain accepted⟩

#print axioms occurrence_route_iff
#print axioms valid_iff_readings
end MultiIntroReadings
