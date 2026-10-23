import Semantic.Circuit.ProductionRoutes

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 16000000
set_option maxRecDepth 20000

abbrev SeedWord (d : Graph) := {word : ColourPath // word ∈ [] :: graphWords d}

def seedModes {d : Graph} (seeds : Option (List (SeedWord d))) : List (Option ColourPath) :=
  match seeds with
  | none => [none]
  | some words => words.map fun word => some word.val

theorem seedModes_fitting {d : Graph} (seeds : Option (List (SeedWord d)))
    (mode : Option ColourPath) (member : mode ∈ seedModes seeds)
    (word : ColourPath) (equal : mode = some word) : word ∈ [] :: graphWords d := by
  cases seeds with
  | none => simp [seedModes] at member; subst mode; contradiction
  | some words =>
    obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
    exact (Option.some.inj equal) ▸ entry.property

def unaryEmission {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (edges : List Deduction) (formula : Formula) (seeds : Option (List (SeedWord d))) (zeroOnly : Bool)
    (e : Candidate n (1 + (F + I))) : List (Candidate n (1 + (F + I))) :=
  edges.flatMap fun edge => (seedModes seeds).map fun mode =>
    unaryCandidate basis d edge formula mode (if zeroOnly then edge.COLOUR == 0 else true) e

def unaryMeaning {d : Graph} (edges : List Deduction) (formula : Formula)
    (seeds : Option (List (SeedWord d))) (zeroOnly : Bool) (r : FlowPair) : List FlowPair :=
  edges.flatMap fun edge => (seedModes seeds).flatMap fun mode =>
    (unaryRoute d edge formula mode (if zeroOnly then edge.COLOUR == 0 else true) r).toList

theorem unaryEmission_inactive {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (edges : List Deduction) (formula : Formula) (seeds : Option (List (SeedWord d))) (zeroOnly : Bool)
    (e : Candidate n (1 + (F + I))) (state : Bits) (live : e.live.value state = false) :
    Encodes basis (graphWords d) state (unaryEmission basis d edges formula seeds zeroOnly e) [] := by
  apply Encodes.inactive_all
  intro entry member
  simp only [unaryEmission, List.mem_flatMap, List.mem_map] at member
  obtain ⟨edge, _, mode, _, rfl⟩ := member
  simp [unaryCandidate, packCandidate, Expr.value, live]

theorem unaryEmission_correct {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (edges : List Deduction) (formula : Formula) (seeds : Option (List (SeedWord d))) (zeroOnly : Bool)
    (e : Candidate n (1 + (F + I))) (r : FlowPair) (state : Bits)
    (live : e.live.value state = true) (fits : RouteFits basis (graphWords d) r)
    (payload : (fun i => (e.payload i).value state) = routeBits (I := I) basis (graphWords d) r)
    (capacity : ([] :: graphWords d).length ≤ 2 ^ I) :
    Encodes basis (graphWords d) state (unaryEmission basis d edges formula seeds zeroOnly e)
      (unaryMeaning edges formula seeds zeroOnly r) := by
  unfold unaryEmission unaryMeaning
  apply Encodes.classical_bind
  intro edge _
  have correct := Encodes.classical_bind (seedModes seeds) (fun mode => [unaryCandidate basis d edge formula mode
    (if zeroOnly then edge.COLOUR == 0 else true) e]) _ (fun mode member =>
      unaryCandidate_correct basis d edge formula mode _ e r state live fits payload capacity
        (seedModes_fitting seeds mode member))
  simpa only [← List.map_eq_flatMap] using correct

def binaryEmission {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minors majors : List Deduction) (seeds : Option (List (SeedWord d)))
    (left right : Candidate n (1 + (F + I))) : List (Candidate n (1 + (F + I))) :=
  minors.flatMap fun minor => majors.flatMap fun major => (seedModes seeds).map fun mode =>
    binaryCandidate basis d minor major mode left right

def binaryMeaning {d : Graph} (minors majors : List Deduction) (seeds : Option (List (SeedWord d)))
    (left right : FlowPair) : List FlowPair :=
  minors.flatMap fun minor => majors.flatMap fun major => (seedModes seeds).flatMap fun mode =>
    (binaryRoute d minor major mode left right).toList

theorem binaryEmission_inactive {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minors majors : List Deduction) (seeds : Option (List (SeedWord d)))
    (left right : Candidate n (1 + (F + I))) (state : Bits)
    (live : left.live.value state = false ∨ right.live.value state = false) :
    Encodes basis (graphWords d) state (binaryEmission basis d minors majors seeds left right) [] := by
  apply Encodes.inactive_all
  intro entry member
  simp only [binaryEmission, List.mem_flatMap, List.mem_map] at member
  obtain ⟨minor, _, major, _, mode, _, rfl⟩ := member
  rcases live with h | h <;> simp [binaryCandidate, packCandidate, Expr.value_all, h]

theorem binaryEmission_correct {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minors majors : List Deduction) (seeds : Option (List (SeedWord d)))
    (left right : Candidate n (1 + (F + I))) (a b : FlowPair) (state : Bits)
    (al : left.live.value state = true) (bl : right.live.value state = true)
    (af : RouteFits basis (graphWords d) a) (bf : RouteFits basis (graphWords d) b)
    (ap : (fun i => (left.payload i).value state) = routeBits (I := I) basis (graphWords d) a)
    (bp : (fun i => (right.payload i).value state) = routeBits (I := I) basis (graphWords d) b)
    (capacity : ([] :: graphWords d).length ≤ 2 ^ I) :
    Encodes basis (graphWords d) state (binaryEmission basis d minors majors seeds left right)
      (binaryMeaning minors majors seeds a b) := by
  unfold binaryEmission binaryMeaning
  apply Encodes.classical_bind
  intro minor _
  apply Encodes.classical_bind
  intro major _
  have correct := Encodes.classical_bind (seedModes seeds)
    (fun mode => [binaryCandidate basis d minor major mode left right]) _ (fun mode member =>
      binaryCandidate_correct basis d minor major mode left right a b state al bl af bf ap bp capacity
        (seedModes_fitting seeds mode member))
  simpa only [← List.map_eq_flatMap] using correct



inductive ProductionPlan {F : Nat} (d : Graph) (basis : Fin F → Formula) where
  | input (vertex : Vertex)
  | constants (routes : List FlowPair) (fitting : ∀ r ∈ routes, RouteFits basis (graphWords d) r)
  | unary (edges : List Deduction) (formula : Formula) (seeds : Option (List (SeedWord d)))
      (zeroOnly : Bool) (child : ProductionPlan d basis)
  | binary (minors majors : List Deduction) (seeds : Option (List (SeedWord d)))
      (left right : ProductionPlan d basis)
  | append (left right : ProductionPlan d basis)
  | dedup (child : ProductionPlan d basis)

def ProductionPlan.meaning {F : Nat} {d : Graph} {basis : Fin F → Formula}
    (prior : Vertex → List FlowPair) : ProductionPlan d basis → List FlowPair
  | .input v => prior v
  | .constants routes _ => routes
  | .unary edges formula seeds zero child => (child.meaning prior).flatMap (fun r => unaryMeaning edges formula seeds zero r)
  | .binary minors majors seeds left right => (left.meaning prior).flatMap fun a =>
      (right.meaning prior).flatMap (fun b => binaryMeaning minors majors seeds a b)
  | .append left right => left.meaning prior ++ right.meaning prior
  | .dedup child => (child.meaning prior).eraseDups

def ProductionPlan.compile {n F I : Nat} {d : Graph} {basis : Fin F → Formula}
    (inputs : Vertex → List (Candidate n (1 + (F + I)))) : ProductionPlan d basis → List (Candidate n (1 + (F + I)))
  | .input v => inputs v
  | .constants routes _ => routes.map (constantCandidate basis (graphWords d))
  | .unary edges formula seeds zero child => (child.compile inputs).flatMap (unaryEmission basis d edges formula seeds zero)
  | .binary minors majors seeds left right => (left.compile inputs).flatMap fun a =>
      (right.compile inputs).flatMap (binaryEmission basis d minors majors seeds a)
  | .append left right => left.compile inputs ++ right.compile inputs
  | .dedup child => firstOccurrences (child.compile inputs)

theorem ProductionPlan.correct {n F I : Nat} {d : Graph} {basis : Fin F → Formula}
    (plan : ProductionPlan d basis) (prior : Vertex → List FlowPair)
    (inputs : Vertex → List (Candidate n (1 + (F + I)))) (state : Bits)
    (capacity : ([] :: graphWords d).length ≤ 2 ^ I)
    (inputCorrect : ∀ v, Encodes basis (graphWords d) state (inputs v) (prior v)) :
    Encodes basis (graphWords d) state (plan.compile inputs) (plan.meaning prior) := by
  induction plan with
  | input v => exact inputCorrect v
  | constants routes fitting => exact constants_encode basis (graphWords d) routes state fitting
  | unary edges formula seeds zero child ih =>
    exact ih.bind _ _
      (fun e inactive => unaryEmission_inactive basis d edges formula seeds zero e state inactive)
      (fun e r live fits payload => unaryEmission_correct basis d edges formula seeds zero e r state live fits payload capacity)
  | binary minors majors seeds left right hl hr =>
    apply hl.bind
    · intro a inactive
      have allEmpty := Encodes.classical_bind (right.compile inputs)
        (binaryEmission basis d minors majors seeds a) (fun _ => [])
        (fun b _ => binaryEmission_inactive basis d minors majors seeds a b state (Or.inl inactive))
      simpa only [flatMap_empty] using allEmpty
    · intro a r live fits payload
      exact hr.bind _ _
        (fun b inactive => binaryEmission_inactive basis d minors majors seeds a b state (Or.inr inactive))
        (fun b s liveB fitsB payloadB => binaryEmission_correct basis d minors majors seeds a b r s state
          live liveB fits fitsB payload payloadB capacity)
  | append left right hl hr => exact hl.append hr
  | dedup child ih => exact ih.deduplicate capacity

def ProductionPlan.empty {F : Nat} {d : Graph} {basis : Fin F → Formula} : ProductionPlan d basis :=
  .constants [] (by simp)

def ProductionPlan.join {F : Nat} {d : Graph} {basis : Fin F → Formula}
    (plans : List (ProductionPlan d basis)) : ProductionPlan d basis :=
  plans.foldr ProductionPlan.append ProductionPlan.empty

theorem ProductionPlan.meaning_join {F : Nat} {d : Graph} {basis : Fin F → Formula}
    (plans : List (ProductionPlan d basis)) (prior : Vertex → List FlowPair) :
    (ProductionPlan.join plans).meaning prior = plans.flatMap (fun plan => plan.meaning prior) := by
  induction plans with
  | nil => rfl
  | cons head rest ih =>
    change head.meaning prior ++ (ProductionPlan.join rest).meaning prior = _
    rw [ih]
    rfl

@[simp] theorem ProductionPlan.meaning_ite {F : Nat} {d : Graph} {basis : Fin F → Formula}
    (condition : Prop) [Decidable condition] (left right : ProductionPlan d basis)
    (prior : Vertex → List FlowPair) :
    (if condition then left else right).meaning prior =
      if condition then left.meaning prior else right.meaning prior := by
  by_cases h : condition <;> simp [h]

end Semantic.Circuit
