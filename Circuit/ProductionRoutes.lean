import Circuit.ProductionStreams

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 12000000
set_option maxRecDepth 10000

def selectionMask {n F I : Nat} (e : Selection n (1 + (F + I))) : Fin F → Expr n :=
  fun i => e.payload (dependencyOffset I i)
def selectionResidual {n F I : Nat} (e : Selection n (1 + (F + I))) : Fin I → Expr n :=
  fun i => e.payload (residualOffset F i)

def packSelection {n F I : Nat} (live : Expr n) (mask : Fin F → Expr n)
    (residual : Fin I → Expr n) : Selection n (1 + (F + I)) :=
  ⟨live, Fin.addCases (fun _ => .constant true) (Fin.addCases mask residual)⟩

theorem selectionMask_correct {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (e : Selection n (1 + (F + I))) (r : FlowPair) (state : Bits)
    (payload : (fun i => (e.payload i).value state) = routeBits (I := I) basis words r) :
    ∀ i, (selectionMask e i).value state = decide (basis i ∈ r.1) := by
  intro i
  simpa only [selectionMask, routeBits, encodeRoute, Slot.bit, dependencyOffset,
    Fin.addCases_right, Fin.addCases_left] using congrFun payload (dependencyOffset I i)

theorem selectionResidual_correct {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (e : Selection n (1 + (F + I))) (r : FlowPair) (state : Bits)
    (payload : (fun i => (e.payload i).value state) = routeBits (I := I) basis words r) :
    ∀ i : Fin I, (selectionResidual e i).value state =
      (BitVec.ofNat I (([] :: words).idxOf r.2)).getLsbD i.val := by
  intro i
  simpa only [selectionResidual, routeBits, encodeRoute, Slot.bit, residualOffset,
    Fin.addCases_right] using congrFun payload (residualOffset F i)

theorem packSelection_correct {n F I : Nat} (basis : Fin F → Formula) (words : List ColourPath)
    (live : Expr n) (mask : Fin F → Expr n) (residual : Fin I → Expr n) (r : FlowPair) (state : Bits)
    (maskValues : ∀ i, (mask i).value state = decide (basis i ∈ r.1))
    (residualValues : ∀ i : Fin I, (residual i).value state =
      (BitVec.ofNat I (([] :: words).idxOf r.2)).getLsbD i.val) :
    (fun i => ((packSelection live mask residual).payload i).value state) = routeBits (I := I) basis words r := by
  funext i
  refine Fin.addCases ?_ (Fin.addCases ?_ ?_) i
  · intro j
    simp only [packSelection, routeBits, encodeRoute, Slot.bit, Fin.addCases_left, Expr.value]
  · intro j
    simpa only [packSelection, routeBits, encodeRoute, Slot.bit, Fin.addCases_right, Fin.addCases_left] using maskValues j
  · intro j
    simpa only [packSelection, routeBits, encodeRoute, Slot.bit, Fin.addCases_right] using residualValues j

theorem consumeValid_word {n I : Nat} (input : Fin I → Expr n) (words : List ColourPath)
    (edge : Deduction) (word : ColourPath) (state : Bits) (member : word ∈ [] :: words)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (values : ∀ i : Fin I, (input i).value state =
      (BitVec.ofNat I (([] :: words).idxOf word)).getLsbD i.val) :
    (consumeValid input words edge).value state = (consume edge word).isSome := by
  let index : Fin ([] :: words).length := ⟨_, List.idxOf_lt_length_of_mem member⟩
  have correct := consumeValid_correct input words edge index state capacity values
  simpa only [index, List.idxOf_get] using correct

theorem consumeBit_word {n I : Nat} (input : Fin I → Expr n) (words : List ColourPath)
    (edge : Deduction) (word tail : ColourPath) (state : Bits) (member : word ∈ [] :: words)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (values : ∀ i : Fin I, (input i).value state =
      (BitVec.ofNat I (([] :: words).idxOf word)).getLsbD i.val)
    (consumed : consume edge word = some tail) :
    ∀ bit : Fin I, (consumeBit input words edge bit).value state =
      (BitVec.ofNat I (([] :: words).idxOf tail)).getLsbD bit.val := by
  intro bit
  let index : Fin ([] :: words).length := ⟨_, List.idxOf_lt_length_of_mem member⟩
  have correct := consumeBit_correct input words edge index bit state capacity values
  simpa only [index, List.idxOf_get, consumed, Option.map_some, Option.getD_some] using correct

theorem removed_fits {F : Nat} (basis : Fin F → Formula) (d : Graph) (r : FlowPair)
    (formula : Formula) (tail : ColourPath) (fits : RouteFits basis (graphWords d) r)
    (tailFits : tail ∈ [] :: graphWords d) : RouteFits basis (graphWords d) (depRemove r.1 formula, tail) := by
  refine ⟨depRemove_canonical _ _, ?_, tailFits⟩
  intro f member
  have present : f ∈ r.1 := by
    have both : f ∈ r.1 ∧ f ≠ formula := by simpa [depRemove, List.removeAll] using member
    exact both.1
  exact fits.2.1 f present

theorem union_fits {F : Nat} (basis : Fin F → Formula) (d : Graph) (left right : FlowPair)
    (tail : ColourPath) (a : RouteFits basis (graphWords d) left)
    (b : RouteFits basis (graphWords d) right) (tailFits : tail ∈ [] :: graphWords d) :
    RouteFits basis (graphWords d) (depUnion left.1 right.1, tail) := by
  refine ⟨depUnion_canonical _ _, ?_, tailFits⟩
  intro f member
  have present : f ∈ left.1 ∨ f ∈ right.1 := by simpa [depUnion] using member
  exact present.elim (a.2.1 f) (b.2.1 f)

def unaryRoute (d : Graph) (edge : Deduction) (formula : Formula) (reseed : Option ColourPath)
    (extra : Bool) (r : FlowPair) : Option FlowPair :=
  if SingleIntroductionReference.guard d r.1 edge && extra then
    (consume edge r.2).map fun tail => (depRemove r.1 formula, reseed.getD tail)
  else none

def unarySelection {n F I : Nat} (basis : Fin F → Formula) (d : Graph) (edge : Deduction)
    (formula : Formula) (reseed : Option ColourPath) (extra : Bool)
    (e : Selection n (1 + (F + I))) : Selection n (1 + (F + I)) :=
  packSelection
    (((e.live.conj (dependencyGuard basis (selectionMask e) edge)).conj (.constant extra)).conj
      (consumeValid (selectionResidual e) (graphWords d) edge))
    (maskClear basis formula (selectionMask e))
    (match reseed with
     | none => consumeBit (selectionResidual e) (graphWords d) edge
     | some word => fun i => .constant ((BitVec.ofNat I (([] :: graphWords d).idxOf word)).getLsbD i.val))

theorem unarySelection_inactive {n F I : Nat} (basis : Fin F → Formula) (d : Graph) (edge : Deduction)
    (formula : Formula) (reseed : Option ColourPath) (extra : Bool)
    (e : Selection n (1 + (F + I))) (state : Bits) (inactive : e.live.value state = false) :
    Encodes basis (graphWords d) state [unarySelection basis d edge formula reseed extra e] [] := by
  apply Encodes.inactive _ .nil
  simp [unarySelection, packSelection, Expr.value, inactive]

theorem unarySelection_correct {n F I : Nat} (basis : Fin F → Formula) (d : Graph) (edge : Deduction)
    (formula : Formula) (reseed : Option ColourPath) (extra : Bool)
    (e : Selection n (1 + (F + I))) (r : FlowPair) (state : Bits)
    (live : e.live.value state = true) (fits : RouteFits basis (graphWords d) r)
    (payload : (fun i => (e.payload i).value state) = routeBits (I := I) basis (graphWords d) r)
    (capacity : ([] :: graphWords d).length ≤ 2 ^ I)
    (reseedFits : ∀ word, reseed = some word → word ∈ [] :: graphWords d) :
    Encodes basis (graphWords d) state [unarySelection basis d edge formula reseed extra e]
      (unaryRoute d edge formula reseed extra r).toList := by
  have maskValues := selectionMask_correct basis (graphWords d) e r state payload
  have indexValues := selectionResidual_correct basis (graphWords d) e r state payload
  have guardValue := dependencyGuard_correct basis (selectionMask e) r.1 edge d state maskValues fits.2.1
  have consumeValue := consumeValid_word (selectionResidual e) (graphWords d) edge r.2 state fits.2.2 capacity indexValues
  by_cases guarded : (SingleIntroductionReference.guard d r.1 edge && extra) = true
  · cases consumed : consume edge r.2 with
    | none =>
      simp only [unaryRoute, guarded, ite_true, consumed, Option.map_none, Option.toList_none]
      apply Encodes.inactive _ .nil
      simp [unarySelection, packSelection, Expr.value, live, guardValue, consumeValue, consumed]
    | some tail =>
      have tailFits : tail ∈ [] :: graphWords d := graph_consume_closed d edge r.2 tail fits.2.2 consumed
      have outputFits : reseed.getD tail ∈ [] :: graphWords d := by
        cases reseed with
        | none => exact tailFits
        | some word => exact reseedFits word rfl
      simp only [unaryRoute, guarded, ite_true, consumed, Option.map_some, Option.toList_some]
      apply Encodes.active _ (removed_fits basis d r formula _ fits outputFits) _ .nil
      · simp [unarySelection, packSelection, Expr.value, live, guardValue, consumeValue, consumed, guarded]
      · apply packSelection_correct
        · intro i
          rw [maskClear_value, maskValues, removal_mask]
        · cases reseed with
          | none => exact consumeBit_word _ _ edge r.2 tail state fits.2.2 capacity indexValues consumed
          | some word => intro i; rfl
  · have notGuarded : (SingleIntroductionReference.guard d r.1 edge && extra) = false := Bool.eq_false_iff.mpr guarded
    simp only [unaryRoute, notGuarded, Bool.false_eq_true, ite_false, Option.toList_none]
    apply Encodes.inactive _ .nil
    simp [unarySelection, packSelection, Expr.value, live, guardValue, consumeValue, notGuarded]

def binaryRoute (d : Graph) (minor major : Deduction) (reseed : Option ColourPath)
    (left right : FlowPair) : Option FlowPair :=
  if SingleIntroductionReference.guard d left.1 minor && SingleIntroductionReference.guard d right.1 major then
    match consume minor left.2, consume major right.2 with
    | some p, some q => if reseed.isSome || decide (p = q) then
        some (depUnion left.1 right.1, reseed.getD p) else none
    | _, _ => none
  else none

def binaryAgreement {n F I : Nat} (d : Graph) (minor major : Deduction) (reseed : Option ColourPath)
    (left right : Selection n (1 + (F + I))) : Expr n :=
  match reseed with
  | some _ => .constant true
  | none => fieldEqual (consumeBit (selectionResidual left) (graphWords d) minor)
      (consumeBit (selectionResidual right) (graphWords d) major)

def binarySelection {n F I : Nat} (basis : Fin F → Formula) (d : Graph) (minor major : Deduction)
    (reseed : Option ColourPath) (left right : Selection n (1 + (F + I))) : Selection n (1 + (F + I)) :=
  packSelection
    (Expr.all [left.live, right.live, dependencyGuard basis (selectionMask left) minor,
      dependencyGuard basis (selectionMask right) major,
      consumeValid (selectionResidual left) (graphWords d) minor,
      consumeValid (selectionResidual right) (graphWords d) major,
      binaryAgreement d minor major reseed left right])
    (maskUnion (selectionMask left) (selectionMask right))
    (match reseed with
     | none => consumeBit (selectionResidual left) (graphWords d) minor
     | some word => fun i => .constant ((BitVec.ofNat I (([] :: graphWords d).idxOf word)).getLsbD i.val))

theorem binarySelection_inactive {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minor major : Deduction) (reseed : Option ColourPath) (left right : Selection n (1 + (F + I)))
    (state : Bits) (inactive : left.live.value state = false ∨ right.live.value state = false) :
    Encodes basis (graphWords d) state [binarySelection basis d minor major reseed left right] [] := by
  apply Encodes.inactive _ .nil
  rcases inactive with h | h <;>
    simp [binarySelection, packSelection, Expr.value_all, h]

theorem binarySelection_correct {n F I : Nat} (basis : Fin F → Formula) (d : Graph)
    (minor major : Deduction) (reseed : Option ColourPath) (left right : Selection n (1 + (F + I)))
    (a b : FlowPair) (state : Bits)
    (leftLive : left.live.value state = true) (rightLive : right.live.value state = true)
    (aFits : RouteFits basis (graphWords d) a) (bFits : RouteFits basis (graphWords d) b)
    (leftPayload : (fun i => (left.payload i).value state) = routeBits (I := I) basis (graphWords d) a)
    (rightPayload : (fun i => (right.payload i).value state) = routeBits (I := I) basis (graphWords d) b)
    (capacity : ([] :: graphWords d).length ≤ 2 ^ I)
    (reseedFits : ∀ word, reseed = some word → word ∈ [] :: graphWords d) :
    Encodes basis (graphWords d) state [binarySelection basis d minor major reseed left right]
      (binaryRoute d minor major reseed a b).toList := by
  have am := selectionMask_correct basis (graphWords d) left a state leftPayload
  have bm := selectionMask_correct basis (graphWords d) right b state rightPayload
  have ai := selectionResidual_correct basis (graphWords d) left a state leftPayload
  have bi := selectionResidual_correct basis (graphWords d) right b state rightPayload
  have ag := dependencyGuard_correct basis (selectionMask left) a.1 minor d state am aFits.2.1
  have bg := dependencyGuard_correct basis (selectionMask right) b.1 major d state bm bFits.2.1
  have ac := consumeValid_word (selectionResidual left) (graphWords d) minor a.2 state aFits.2.2 capacity ai
  have bc := consumeValid_word (selectionResidual right) (graphWords d) major b.2 state bFits.2.2 capacity bi
  by_cases guards : (SingleIntroductionReference.guard d a.1 minor && SingleIntroductionReference.guard d b.1 major) = true
  · cases consumedA : consume minor a.2 with
    | none =>
      simp only [binaryRoute, guards, ite_true, consumedA, Option.toList_none]
      apply Encodes.inactive _ .nil
      simp [binarySelection, packSelection, Expr.value_all, ac, consumedA]
    | some p =>
      cases consumedB : consume major b.2 with
      | none =>
        simp only [binaryRoute, guards, ite_true, consumedA, consumedB, Option.toList_none]
        apply Encodes.inactive _ .nil
        simp [binarySelection, packSelection, Expr.value_all, bc, consumedB]
      | some q =>
        have pFits := graph_consume_closed d minor a.2 p aFits.2.2 consumedA
        have qFits := graph_consume_closed d major b.2 q bFits.2.2 consumedB
        have aBits := consumeBit_word (selectionResidual left) (graphWords d) minor a.2 p state aFits.2.2 capacity ai consumedA
        have bBits := consumeBit_word (selectionResidual right) (graphWords d) major b.2 q state bFits.2.2 capacity bi consumedB
        have agreement : (binaryAgreement d minor major reseed left right).value state =
            (reseed.isSome || decide (p = q)) := by
          cases reseed with
          | some word => rfl
          | none =>
            simp only [binaryAgreement, fieldEqual_value, funext aBits, funext bBits,
              bitFields_equal_iff, residual_agreement (graphWords d) p q pFits qFits capacity,
              Option.isSome_none, Bool.false_or]
        by_cases accepted : (reseed.isSome || decide (p = q)) = true
        · have outputFits : reseed.getD p ∈ [] :: graphWords d := by
            cases reseed with
            | none => exact pFits
            | some word => exact reseedFits word rfl
          simp only [binaryRoute, guards, ite_true, consumedA, consumedB, accepted, Option.toList_some]
          apply Encodes.active _ (union_fits basis d a b _ aFits bFits outputFits) _ .nil
          · simp [binarySelection, packSelection, Expr.value_all, leftLive, rightLive, ag, bg,
              ac, bc, consumedA, consumedB, agreement, accepted, guards]
          · apply packSelection_correct
            · intro i
              rw [maskUnion_value, am, bm, union_mask]
            · cases reseed with
              | none => exact aBits
              | some word => intro i; rfl
        · simp only [binaryRoute, guards, ite_true, consumedA, consumedB, accepted, ]
          apply Encodes.inactive _ .nil
          simp [binarySelection, packSelection, Expr.value_all, agreement, accepted]
  · simp only [binaryRoute, guards, ]
    apply Encodes.inactive _ .nil
    simp [binarySelection, packSelection, Expr.value_all, ag, bg, ]
    intro _ _ leftGuard rightGuard
    exact False.elim (guards (by simp [leftGuard, rightGuard]))

end Semantic.Circuit
