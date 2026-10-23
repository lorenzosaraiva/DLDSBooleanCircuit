import Semantic.Circuit.SharedFlags

open FlowSpec
namespace Semantic.Circuit.Shared

set_option maxHeartbeats 30000000
set_option maxRecDepth 100000

structure Represents {F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (state : Bits) (entries : List (Registers (1 + (F + I)))) (routes : List FlowPair) : Prop where
  fitting : ∀ r ∈ routes, RouteFits basis words r
  bits : selected (values entries state) = routes.map (routeBits (I := I) basis words)

theorem Represents.encodes {F I : Nat} {basis : Fin F → Formula} {words state entries routes}
    (represented : Represents (I := I) basis words state entries routes) (scope : Nat)
    (bounded : Bounded entries scope) (capacity : ([] :: words).length ≤ 2 ^ I) :
    Encodes basis words state (entries.map fun e => e.expressions scope) routes := by
  apply encodes_from_bits _ _ _ _ _ capacity represented.fitting
  rw [expressions_values _ _ _ bounded, represented.bits]

theorem Represents.stable {F I : Nat} {basis : Fin F → Formula} {words entries routes before after}
    (represented : Represents (I := I) basis words (memory base before state) entries routes)
    (extension : Extends before after) (bounded : Bounded entries (base + before.size)) :
    Represents basis words (memory base after state) entries routes := by
  refine ⟨represented.fitting, ?_⟩
  rw [values_stable entries extension base state bounded, represented.bits]

theorem memory_input_expression {n : Nat} (code : Store) (e : Expr n) (state : Bits) :
    e.value (memory n code state) = e.value state := by
  apply Expr.value_congr
  intro wire
  exact (assemble n code.toList).preserves_before state wire.val wire.isLt

theorem input_encodes {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (entries : List (Candidate n (1 + (F + I)))) (routes : List FlowPair) (state : Bits)
    (code : Store) (capacity : ([] :: words).length ≤ 2 ^ I)
    (encoded : Encodes basis words state entries routes) :
    Encodes basis words (memory n code state) entries routes := by
  apply encodes_from_bits _ _ _ _ _ capacity encoded.fitting
  rw [← encoded.bits]
  congr 1
  apply List.map_congr_left
  intro e _
  apply Prod.ext
  · exact memory_input_expression code e.live state
  · funext i
    exact memory_input_expression code (e.payload i) state

theorem candidates_represents {scope F I : Nat} (base : Nat) (basis : Fin F → Formula)
    (words : List ColourPath) (entries : List (Candidate scope (1 + (F + I))))
    (routes : List FlowPair) (code : Store) (space : scope ≤ base + code.size) (state : Bits)
    (encoded : Encodes basis words (memory base code state) entries routes) :
    let result := candidates base entries code
    Extends code result.2 ∧ Bounded result.1 (base + result.2.size) ∧
      Represents basis words (memory base result.2 state) result.1 routes := by
  obtain ⟨extension, bounded, same⟩ := candidates_correct base entries code space state
  exact ⟨extension, bounded, encoded.fitting, by rw [values, same, encoded.bits]⟩

def compilePlan {n F I : Nat} {d : Graph} {basis : Fin F → Formula}
    (inputs : Vertex → List (Candidate n (1 + (F + I)))) :
    ProductionPlan d basis → Store → List (Registers (1 + (F + I))) × Store
  | .input v, code => candidates n (inputs v) code
  | .constants routes _, code => candidates n (routes.map (constantCandidate (n := n) basis (graphWords d))) code
  | .unary edges formula seeds zero child, code =>
      let source := compilePlan inputs child code
      let scope := n + source.2.size
      candidates n ((source.1.map fun e => e.expressions scope).flatMap
        (unaryEmission basis d edges formula seeds zero)) source.2
  | .binary minors majors seeds left right, code =>
      let a := compilePlan inputs left code
      let b := compilePlan inputs right a.2
      let scope := n + b.2.size
      candidates n ((a.1.map fun e => e.expressions scope).flatMap fun x =>
        (b.1.map fun e => e.expressions scope).flatMap (binaryEmission basis d minors majors seeds x)) b.2
  | .append left right, code =>
      let a := compilePlan inputs left code
      let b := compilePlan inputs right a.2
      (a.1 ++ b.1, b.2)
  | .dedup child, code =>
      let source := compilePlan inputs child code
      firstFlags n source.1.length source.1 source.2

theorem compilePlan_bounds {n F I : Nat} {d : Graph} {basis : Fin F → Formula}
    (plan : ProductionPlan d basis) (inputs : Vertex → List (Candidate n (1 + (F + I)))) (code : Store) :
    Extends code (compilePlan inputs plan code).2 ∧
      Bounded (compilePlan inputs plan code).1 (n + (compilePlan inputs plan code).2.size) := by
  induction plan generalizing code with
  | input v => exact (candidates_correct n (inputs v) code (by omega) (fun _ => false)).imp_right And.left
  | constants routes fitting =>
    exact (candidates_correct n (routes.map (constantCandidate (n := n) basis (graphWords d))) code
      (by omega) (fun _ => false)).imp_right And.left
  | unary edges formula seeds zero child ih =>
    obtain ⟨firstExtension, _⟩ := ih code
    have final := candidates_correct n
      (((compilePlan inputs child code).1.map fun e => e.expressions (n + (compilePlan inputs child code).2.size)).flatMap
        (unaryEmission basis d edges formula seeds zero)) (compilePlan inputs child code).2 (by omega) (fun _ => false)
    exact ⟨firstExtension.trans final.1, final.2.1⟩
  | binary minors majors seeds left right hl hr =>
    have first := hl code
    have second := hr (compilePlan inputs left code).2
    have final := candidates_correct n
      (((compilePlan inputs left code).1.map fun e => e.expressions (n + (compilePlan inputs right (compilePlan inputs left code).2).2.size)).flatMap fun a =>
        ((compilePlan inputs right (compilePlan inputs left code).2).1.map fun e => e.expressions (n + (compilePlan inputs right (compilePlan inputs left code).2).2.size)).flatMap
          (binaryEmission basis d minors majors seeds a)) (compilePlan inputs right (compilePlan inputs left code).2).2 (by omega) (fun _ => false)
    exact ⟨(first.1.trans second.1).trans final.1, final.2.1⟩
  | append left right hl hr =>
    obtain ⟨firstExtension, firstBounds⟩ := hl code
    obtain ⟨secondExtension, secondBounds⟩ := hr (compilePlan inputs left code).2
    refine ⟨firstExtension.trans secondExtension, ?_⟩
    intro e member
    rcases List.mem_append.mp member with h | h
    · exact Registers.weaken e (firstBounds e h) (Nat.add_le_add_left secondExtension.size_le n)
    · exact secondBounds e h
  | dedup child ih =>
    obtain ⟨firstExtension, firstBounds⟩ := ih code
    have final := firstFlags_correct n (compilePlan inputs child code).1.length _ _ (fun _ => false) firstBounds
    exact ⟨firstExtension.trans final.1, final.2.1⟩

theorem compilePlan_correct {n F I : Nat} {d : Graph} {basis : Fin F → Formula}
    (plan : ProductionPlan d basis) (prior : Vertex → List FlowPair)
    (inputs : Vertex → List (Candidate n (1 + (F + I)))) (state : Bits)
    (capacity : ([] :: graphWords d).length ≤ 2 ^ I)
    (inputCorrect : ∀ v, Encodes basis (graphWords d) state (inputs v) (prior v)) (code : Store) :
    let result := compilePlan inputs plan code
    Extends code result.2 ∧ Bounded result.1 (n + result.2.size) ∧
      Represents basis (graphWords d) (memory n result.2 state) result.1 (plan.meaning prior) := by
  induction plan generalizing code with
  | input v =>
    exact candidates_represents n basis (graphWords d) (inputs v) (prior v) code (by omega) state
      (input_encodes basis (graphWords d) (inputs v) (prior v) state code capacity (inputCorrect v))
  | constants routes fitting =>
    exact candidates_represents n basis (graphWords d) (routes.map (constantCandidate (n := n) basis (graphWords d))) routes code (by omega) state
      (constants_encode basis (graphWords d) routes _ fitting)
  | unary edges formula seeds zero child ih =>
    obtain ⟨sourceExtension, sourceBounds, sourceCorrect⟩ := ih code
    have encoded := sourceCorrect.encodes _ sourceBounds capacity
    have operated := encoded.bind _ _
      (fun e inactive => unaryEmission_inactive basis d edges formula seeds zero e _ inactive)
      (fun e r live fits payload => unaryEmission_correct basis d edges formula seeds zero e r _ live fits payload capacity)
    obtain ⟨extension, bounds, correct⟩ := candidates_represents n basis (graphWords d) _ _
      (compilePlan inputs child code).2 (by omega) state operated
    exact ⟨sourceExtension.trans extension, bounds, correct⟩
  | binary minors majors seeds left right hl hr =>
    obtain ⟨leftExtension, leftBounds, leftCorrect⟩ := hl code
    obtain ⟨rightExtension, rightBounds, rightCorrect⟩ := hr (compilePlan inputs left code).2
    have leftNow := leftCorrect.stable rightExtension leftBounds
    have leftBoundsNow := Bounded.weaken leftBounds (Nat.add_le_add_left rightExtension.size_le n)
    have le := leftNow.encodes _ leftBoundsNow capacity
    have re := rightCorrect.encodes _ rightBounds capacity
    have operated : Encodes basis (graphWords d) (memory n (compilePlan inputs right (compilePlan inputs left code).2).2 state)
        (((compilePlan inputs left code).1.map fun e => e.expressions (n + (compilePlan inputs right (compilePlan inputs left code).2).2.size)).flatMap fun x =>
          ((compilePlan inputs right (compilePlan inputs left code).2).1.map fun e => e.expressions (n + (compilePlan inputs right (compilePlan inputs left code).2).2.size)).flatMap
            (binaryEmission basis d minors majors seeds x))
        ((left.meaning prior).flatMap fun a => (right.meaning prior).flatMap (binaryMeaning minors majors seeds a)) := by
      apply le.bind
      · intro a inactive
        have empty := Encodes.classical_bind
          ((compilePlan inputs right (compilePlan inputs left code).2).1.map fun e =>
            e.expressions (n + (compilePlan inputs right (compilePlan inputs left code).2).2.size))
          (binaryEmission basis d minors majors seeds a) (fun _ => [])
          (fun b _ => binaryEmission_inactive basis d minors majors seeds a b _ (Or.inl inactive))
        simpa only [flatMap_empty] using empty
      · intro a r live fits payload
        exact re.bind _ _
          (fun b inactive => binaryEmission_inactive basis d minors majors seeds a b _ (Or.inr inactive))
          (fun b s liveB fitsB payloadB => binaryEmission_correct basis d minors majors seeds a b r s _
            live liveB fits fitsB payload payloadB capacity)
    obtain ⟨extension, bounds, correct⟩ := candidates_represents n basis (graphWords d) _ _
      (compilePlan inputs right (compilePlan inputs left code).2).2 (by omega) state operated
    exact ⟨(leftExtension.trans rightExtension).trans extension, bounds, correct⟩
  | append left right hl hr =>
    dsimp only [compilePlan, ProductionPlan.meaning]
    obtain ⟨leftExtension, leftBounds, leftCorrect⟩ := hl code
    obtain ⟨rightExtension, rightBounds, rightCorrect⟩ := hr (compilePlan inputs left code).2
    have leftNow := leftCorrect.stable rightExtension leftBounds
    refine ⟨leftExtension.trans rightExtension, ?_, ?_, ?_⟩
    · intro e member
      rcases List.mem_append.mp member with h | h
      · exact Registers.weaken e (leftBounds e h) (Nat.add_le_add_left rightExtension.size_le n)
      · exact rightBounds e h
    · intro r member
      rcases List.mem_append.mp member with h | h
      · exact leftNow.fitting r h
      · exact rightCorrect.fitting r h
    · simp only [values, List.map_append, selected_append, List.map_append]
      exact congrArg₂ List.append leftNow.bits rightCorrect.bits
  | dedup child ih =>
    obtain ⟨sourceExtension, sourceBounds, sourceCorrect⟩ := ih code
    obtain ⟨extension, bounds, _⟩ := firstFlags_correct n (compilePlan inputs child code).1.length
      (compilePlan inputs child code).1 (compilePlan inputs child code).2 state sourceBounds
    have encoded := sourceCorrect.encodes _ sourceBounds capacity
    have exactBits := (encoded.deduplicate capacity).bits
    rw [firstOccurrences_correct, expressions_values _ _ _ sourceBounds] at exactBits
    exact ⟨sourceExtension.trans extension, bounds, (encoded.deduplicate capacity).fitting,
      (firstFlags_selected n _ _ state sourceBounds).trans exactBits⟩

end Semantic.Circuit.Shared
