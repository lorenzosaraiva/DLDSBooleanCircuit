import Circuit.Gadgets
import Circuit.Encoding

open FlowSpec
namespace Semantic.Circuit

set_option maxHeartbeats 4000000


def Expr.equivalent {n : Nat} (left right : Expr n) : Expr n :=
  (left.conj right).disj (left.neg.conj right.neg)

theorem Expr.value_equivalent {n : Nat} (left right : Expr n) (state : Bits) :
    (left.equivalent right).value state = (left.value state == right.value state) := by
  simp only [Expr.equivalent, Expr.value_disj, Expr.value]
  apply Bool.eq_iff_iff.mpr
  simp only [Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq]
  constructor
  · rintro (⟨leftTrue, rightTrue⟩ | ⟨leftFalse, rightFalse⟩) <;> simp_all
  · intro equal
    by_cases isTrue : left.value state = true
    · exact Or.inl ⟨isTrue, equal ▸ isTrue⟩
    · exact Or.inr ⟨Bool.eq_false_iff.mpr isTrue, equal ▸ Bool.eq_false_iff.mpr isTrue⟩

def maskClear {n F : Nat} (basis : Fin F → Formula) (formula : Formula)
    (mask : Fin F → Expr n) : Fin F → Expr n :=
  fun i => if basis i = formula then .constant false else mask i

def maskUnion {n F : Nat} (left right : Fin F → Expr n) : Fin F → Expr n :=
  fun i => (left i).disj (right i)

theorem maskClear_value {n F : Nat} (basis : Fin F → Formula) (formula : Formula)
    (mask : Fin F → Expr n) (state : Bits) (i : Fin F) :
    (maskClear basis formula mask i).value state =
      ((mask i).value state && decide (basis i ≠ formula)) := by
  by_cases equal : basis i = formula <;> simp [maskClear, equal, Expr.value]

theorem maskUnion_value {n F : Nat} (left right : Fin F → Expr n)
    (state : Bits) (i : Fin F) :
    (maskUnion left right i).value state = ((left i).value state || (right i).value state) :=
  Expr.value_disj _ _ _

theorem removal_mask {F : Nat} (basis : Fin F → Formula) (dependency : Dep)
    (formula : Formula) (i : Fin F) :
    decide (basis i ∈ depRemove dependency formula) =
      (decide (basis i ∈ dependency) && decide (basis i ≠ formula)) := by
  simp [depRemove, List.removeAll]

theorem union_mask {F : Nat} (basis : Fin F → Formula) (left right : Dep) (i : Fin F) :
    decide (basis i ∈ depUnion left right) =
      (decide (basis i ∈ left) || decide (basis i ∈ right)) := by
  simp [depUnion]


def fieldEqual {n k : Nat} (left right : Fin k → Expr n) : Expr n :=
  Expr.all ((List.finRange k).map fun i => (left i).equivalent (right i))

theorem fieldEqual_value {n k : Nat} (left right : Fin k → Expr n) (state : Bits) :
    (fieldEqual left right).value state =
      decide ((fun i => (left i).value state) = (fun i => (right i).value state)) := by
  apply Bool.eq_iff_iff.mpr
  simp [fieldEqual, Expr.value_all, List.all_map, Expr.value_equivalent, funext_iff]



def lookupBit {n k : Nat} (input : Fin k → Expr n) (table : List (BitVec k × Bool)) : Expr n :=
  Expr.any (table.map fun row =>
    (fieldEqual input (fun i => .constant (row.1.getLsbD i.val))).conj (.constant row.2))

theorem lookupBit_value {n k : Nat} (input : Fin k → Expr n)
    (table : List (BitVec k × Bool)) (state : Bits) :
    (lookupBit input table).value state = table.any (fun row =>
      decide ((fun i => (input i).value state) = (fun i : Fin k => row.1.getLsbD i.val)) && row.2) := by
  simp only [lookupBit, Expr.value_any, List.any_map, Function.comp_def, Expr.value,
    fieldEqual_value]
  rfl

theorem lookupBit_correct {n k : Nat} (input : Fin k → Expr n)
    (table : List (BitVec k × Bool)) (key : BitVec k) (output : Bool) (state : Bits)
    (inputValue : ∀ i : Fin k, (input i).value state = key.getLsbD i.val)
    (member : (key, output) ∈ table)
    (functional : ∀ row ∈ table, row.1 = key → row.2 = output) :
    (lookupBit input table).value state = output := by
  rw [lookupBit_value]
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨row, contained, equal, active⟩
    have sameKey : row.1 = key := by
      apply BitVec.eq_of_getLsbD_eq
      intro i bound
      exact (congrFun equal ⟨i, bound⟩).symm.trans (inputValue ⟨i, bound⟩)
    rw [← functional row contained sameKey]
    exact active
  · intro active
    exact ⟨(key, output), member, funext inputValue, active⟩

def indexedLookup {n k : Nat} (input : Fin k → Expr n) (values : List Bool) : Expr n :=
  lookupBit input ((List.finRange values.length).map fun i => (BitVec.ofNat k i.val, values.get i))

theorem indexedLookup_correct {n k : Nat} (input : Fin k → Expr n)
    (values : List Bool) (index : Fin values.length) (state : Bits)
    (capacity : values.length ≤ 2 ^ k)
    (inputValue : ∀ i : Fin k,
      (input i).value state = (BitVec.ofNat k index.val).getLsbD i.val) :
    (indexedLookup input values).value state = values.get index := by
  apply lookupBit_correct input _ _ _ state inputValue
  · exact List.mem_map.mpr ⟨index, List.mem_finRange index, rfl⟩
  · intro row member equal
    obtain ⟨other, _, rfl⟩ := List.mem_map.mp member
    have indices := congrArg BitVec.toNat equal
    simp only [BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (index.isLt.trans_le capacity),
      Nat.mod_eq_of_lt (other.isLt.trans_le capacity)] at indices
    have same : other = index := Fin.ext indices
    subst other
    rfl

def consumeTable (words : List ColourPath) (edge : Deduction) : List (Option Nat) :=
  ([] :: words).map fun word => (consume edge word).map (([] :: words).idxOf ·)



def consumeValid {n I : Nat} (input : Fin I → Expr n)
    (words : List ColourPath) (edge : Deduction) : Expr n :=
  indexedLookup input (([] :: words).map fun word => (consume edge word).isSome)

def consumeBit {n I : Nat} (input : Fin I → Expr n)
    (words : List ColourPath) (edge : Deduction) (bit : Fin I) : Expr n :=
  indexedLookup input (([] :: words).map fun word =>
    (BitVec.ofNat I (((consume edge word).map (([] :: words).idxOf ·)).getD 0)).getLsbD bit.val)

theorem indexedLookup_map_correct {α : Type} {n I : Nat} (input : Fin I → Expr n)
    (values : List α) (output : α → Bool) (index : Fin values.length) (state : Bits)
    (capacity : values.length ≤ 2 ^ I)
    (inputValue : ∀ i : Fin I,
      (input i).value state = (BitVec.ofNat I index.val).getLsbD i.val) :
    (indexedLookup input (values.map output)).value state = output (values.get index) := by
  let mappedIndex : Fin (values.map output).length := ⟨index.val, by simp⟩
  have correct := indexedLookup_correct input (values.map output) mappedIndex state
    (by simpa using capacity) inputValue
  simpa [mappedIndex, List.get_eq_getElem] using correct

theorem consumeValid_correct {n I : Nat} (input : Fin I → Expr n)
    (words : List ColourPath) (edge : Deduction) (index : Fin ([] :: words).length) (state : Bits)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (inputValue : ∀ i : Fin I,
      (input i).value state = (BitVec.ofNat I index.val).getLsbD i.val) :
    (consumeValid input words edge).value state = (consume edge (([] :: words).get index)).isSome :=
  indexedLookup_map_correct input _ _ index state capacity inputValue

theorem consumeBit_correct {n I : Nat} (input : Fin I → Expr n)
    (words : List ColourPath) (edge : Deduction) (index : Fin ([] :: words).length)
    (bit : Fin I) (state : Bits) (capacity : ([] :: words).length ≤ 2 ^ I)
    (inputValue : ∀ i : Fin I,
      (input i).value state = (BitVec.ofNat I index.val).getLsbD i.val) :
    (consumeBit input words edge bit).value state =
      (BitVec.ofNat I (((consume edge (([] :: words).get index)).map (([] :: words).idxOf ·)).getD 0)).getLsbD bit.val :=
  indexedLookup_map_correct input _ _ index state capacity inputValue

theorem consume_decode (words : List ColourPath) (edge : Deduction) (word : ColourPath)
    (closed : ∀ tail, consume edge word = some tail → tail ∈ [] :: words) :
    ((consume edge word).map (([] :: words).idxOf ·)).map
      (fun index => ([] :: words).getD index []) = consume edge word := by
  cases consumed : consume edge word with
  | none => rfl
  | some tail =>
    simp only [Option.map_some]
    congr 1
    rw [List.getD_eq_getElem?_getD, List.getElem?_idxOf (closed tail consumed)]
    rfl

theorem flow_consume_fits (variant : MultiIntro.Variant) (d : Graph)
    (fuel : Nat) (vertex : Vertex) (route : FlowPair) (edge : Deduction) (tail : ColourPath)
    (member : route ∈ MultiIntro.evaluateAt variant d fuel vertex)
    (consumed : consume edge route.2 = some tail) : tail ∈ [] :: graphWords d :=
  FlowSize.residual_member d tail
    (FlowSize.consume_allowed d edge route.2 tail
      (FlowSize.residual_origin variant d fuel vertex route member) consumed)

theorem nonemptySuffixes_suffix (word tail : ColourPath)
    (member : tail ∈ FlowSize.nonemptySuffixes word) : tail <:+ word := by
  induction word with
  | nil => simp [FlowSize.nonemptySuffixes] at member
  | cons head rest ih =>
    rcases (by simpa only [FlowSize.nonemptySuffixes, List.mem_cons] using member) with equal | contained
    · subst tail
      exact ⟨[], rfl⟩
    · exact (ih contained).trans (show rest <:+ head :: rest from ⟨[head], rfl⟩)

theorem graphWords_allowed (d : Graph) (word : ColourPath)
    (member : word ∈ [] :: graphWords d) : FlowSize.ResidualAllowed d word := by
  rcases List.mem_cons.mp member with empty | contained
  · exact Or.inl empty
  · obtain ⟨ancestor, listed, suffix⟩ := List.mem_flatMap.mp contained
    exact Or.inr ⟨ancestor, listed, nonemptySuffixes_suffix ancestor.COLOURS word suffix⟩



theorem graph_consume_closed (d : Graph) (edge : Deduction) (word tail : ColourPath)
    (member : word ∈ [] :: graphWords d) (consumed : consume edge word = some tail) :
    tail ∈ [] :: graphWords d :=
  FlowSize.residual_member d tail
    (FlowSize.consume_allowed d edge word tail (graphWords_allowed d word member) consumed)

theorem consume_decode_graph (d : Graph) (edge : Deduction) (word : ColourPath)
    (member : word ∈ [] :: graphWords d) :
    ((consume edge word).map (([] :: graphWords d).idxOf ·)).map
      (fun index => ([] :: graphWords d).getD index []) = consume edge word :=
  consume_decode (graphWords d) edge word (fun tail consumed => graph_consume_closed d edge word tail member consumed)

theorem consumeTable_correct (words : List ColourPath) (edge : Deduction) (word : ColourPath)
    (member : word ∈ [] :: words) :
    (consumeTable words edge)[([] :: words).idxOf word]? =
      some ((consume edge word).map (([] :: words).idxOf ·)) := by
  simp only [consumeTable, List.getElem?_map, List.getElem?_idxOf member, Option.map_some]



theorem residual_agreement {I : Nat} (words : List ColourPath) (left right : ColourPath)
    (leftFits : left ∈ [] :: words) (rightFits : right ∈ [] :: words)
    (capacity : ([] :: words).length ≤ 2 ^ I) :
    (BitVec.ofNat I (([] :: words).idxOf left) = BitVec.ofNat I (([] :: words).idxOf right)) ↔
      left = right := by
  constructor
  · intro equal
    have indices := congrArg BitVec.toNat equal
    simp only [BitVec.toNat_ofNat,
      Nat.mod_eq_of_lt (residual_index_bound words left leftFits capacity),
      Nat.mod_eq_of_lt (residual_index_bound words right rightFits capacity)] at indices
    exact (List.idxOf_inj leftFits).mp indices
  · rintro rfl
    rfl

def classicalGuard {n : Nat} (condition : Bool) : Expr n := .constant condition

theorem classicalGuard_value {n : Nat} (condition : Bool) (state : Bits) :
    (classicalGuard (n := n) condition).value state = condition := rfl

def formulaPresent {n F : Nat} (basis : Fin F → Formula) (mask : Fin F → Expr n)
    (formula : Formula) : Expr n :=
  Expr.any ((List.finRange F).map fun i => if basis i = formula then mask i else .constant false)

theorem formulaPresent_correct {n F : Nat} (basis : Fin F → Formula)
    (mask : Fin F → Expr n) (dependency : Dep) (state : Bits) (formula : Formula)
    (values : ∀ i, (mask i).value state = decide (basis i ∈ dependency))
    (supported : ∀ f ∈ dependency, ∃ i, basis i = f) :
    (formulaPresent basis mask formula).value state = decide (formula ∈ dependency) := by
  apply Bool.eq_iff_iff.mpr
  simp only [formulaPresent, Expr.value_any, List.any_map, Function.comp_def,
    List.any_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨i, _, active⟩
    by_cases equal : basis i = formula
    · simpa [equal, values] using active
    · simp [equal, Expr.value] at active
  · intro member
    obtain ⟨i, equal⟩ := supported formula member
    exact ⟨i, List.mem_finRange i, by simp [equal, values, member]⟩




def dependencyGuard {n F : Nat} (basis : Fin F → Formula) (mask : Fin F → Expr n)
    (edge : Deduction) : Expr n :=
  if edge.DEPENDENCY.isEmpty then .constant true else
    (Expr.all (edge.DEPENDENCY.map (formulaPresent basis mask))).conj
      (Expr.all ((List.finRange F).map fun i =>
        if basis i ∈ edge.DEPENDENCY then .constant true else (mask i).neg))

theorem dependencyGuard_correct {n F : Nat} (basis : Fin F → Formula)
    (mask : Fin F → Expr n) (dependency : Dep) (edge : Deduction) (d : Graph) (state : Bits)
    (values : ∀ i, (mask i).value state = decide (basis i ∈ dependency))
    (supported : ∀ f ∈ dependency, ∃ i, basis i = f) :
    (dependencyGuard basis mask edge).value state = SingleIntroductionReference.guard d dependency edge := by
  unfold dependencyGuard SingleIntroductionReference.guard LambdaFlow.labelGuard
    SingleIntroductionReference.compressorOverlay SingleIntroductionReference.compressorLabel
  by_cases empty : edge.DEPENDENCY.isEmpty = true
  · simp [empty, LambdaFlow.labelAgreesB, Expr.value]
  · simp only [Bool.eq_false_iff.mpr empty, Bool.false_eq_true, ite_false, LambdaFlow.labelAgreesB]
    apply Bool.eq_iff_iff.mpr
    simp only [Expr.value, Bool.and_eq_true, Expr.value_all, List.all_map, Function.comp_def,
      List.all_eq_true, formulaPresent_correct basis mask dependency state _ values supported,
      decide_eq_true_eq, depSetEq_eq_true_iff]
    constructor
    · rintro ⟨labelIncluded, maskIncluded⟩ f
      constructor
      · intro member
        obtain ⟨i, equal⟩ := supported f member
        have checked := maskIncluded i (List.mem_finRange i)
        by_contra missing
        simp [equal, missing, Expr.value, values, member] at checked
      · exact labelIncluded f
    · intro same
      refine ⟨fun f member => (same f).mpr member, ?_⟩
      intro i _
      by_cases member : basis i ∈ edge.DEPENDENCY
      · simp [member, Expr.value]
      · have absent : basis i ∉ dependency := fun h => member ((same _).mp h)
        simp [member, Expr.value, values, absent]



theorem operation_oracle {n : Nat} (operation : Expr n) (value : Bool) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false) (correct : operation.value state = value) :
    eval operation.oracle state = Function.update state (n + operation.size) value := by
  rw [operation.oracle_correct state clean, correct]

theorem maskClear_circuit {n F : Nat} (basis : Fin F → Formula) (formula : Formula)
    (mask : Fin F → Expr n) (dependency : Dep) (state : Bits) (i : Fin F)
    (clean : ∀ wire, n ≤ wire → state wire = false)
    (values : ∀ j, (mask j).value state = decide (basis j ∈ dependency)) :
    eval (maskClear basis formula mask i).oracle state =
      Function.update state (n + (maskClear basis formula mask i).size)
        (decide (basis i ∈ depRemove dependency formula)) := by
  apply operation_oracle _ _ _ clean
  rw [maskClear_value, values, removal_mask]

theorem maskUnion_circuit {n F : Nat} (basis : Fin F → Formula)
    (left right : Fin F → Expr n) (a b : Dep) (state : Bits) (i : Fin F)
    (clean : ∀ wire, n ≤ wire → state wire = false)
    (leftValues : ∀ j, (left j).value state = decide (basis j ∈ a))
    (rightValues : ∀ j, (right j).value state = decide (basis j ∈ b)) :
    eval (maskUnion left right i).oracle state =
      Function.update state (n + (maskUnion left right i).size)
        (decide (basis i ∈ depUnion a b)) := by
  apply operation_oracle _ _ _ clean
  rw [maskUnion_value, leftValues, rightValues, union_mask]

theorem dependencyGuard_circuit {n F : Nat} (basis : Fin F → Formula)
    (mask : Fin F → Expr n) (dependency : Dep) (edge : Deduction) (d : Graph) (state : Bits)
    (clean : ∀ wire, n ≤ wire → state wire = false)
    (values : ∀ i, (mask i).value state = decide (basis i ∈ dependency))
    (supported : ∀ f ∈ dependency, ∃ i, basis i = f) :
    eval (dependencyGuard basis mask edge).oracle state =
      Function.update state (n + (dependencyGuard basis mask edge).size)
        (SingleIntroductionReference.guard d dependency edge) := by
  exact operation_oracle _ _ _ clean
    (dependencyGuard_correct basis mask dependency edge d state values supported)

theorem consumeValid_circuit {n I : Nat} (input : Fin I → Expr n)
    (words : List ColourPath) (edge : Deduction) (index : Fin ([] :: words).length) (state : Bits)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (inputValue : ∀ i : Fin I,
      (input i).value state = (BitVec.ofNat I index.val).getLsbD i.val)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (consumeValid input words edge).oracle state =
      Function.update state (n + (consumeValid input words edge).size)
        (consume edge (([] :: words).get index)).isSome :=
  operation_oracle _ _ _ clean (consumeValid_correct input words edge index state capacity inputValue)

theorem consumeBit_circuit {n I : Nat} (input : Fin I → Expr n)
    (words : List ColourPath) (edge : Deduction) (index : Fin ([] :: words).length)
    (bit : Fin I) (state : Bits) (capacity : ([] :: words).length ≤ 2 ^ I)
    (inputValue : ∀ i : Fin I,
      (input i).value state = (BitVec.ofNat I index.val).getLsbD i.val)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (consumeBit input words edge bit).oracle state =
      Function.update state (n + (consumeBit input words edge bit).size)
        ((BitVec.ofNat I (((consume edge (([] :: words).get index)).map
          (([] :: words).idxOf ·)).getD 0)).getLsbD bit.val) :=
  operation_oracle _ _ _ clean (consumeBit_correct input words edge index bit state capacity inputValue)

theorem bitFields_equal_iff {I : Nat} (left right : BitVec I) :
    ((fun i : Fin I => left.getLsbD i.val) = (fun i : Fin I => right.getLsbD i.val)) ↔ left = right := by
  constructor
  · intro equal
    apply BitVec.eq_of_getLsbD_eq
    intro i bound
    exact congrFun equal ⟨i, bound⟩
  · rintro rfl
    rfl

theorem residualAgreement_circuit {n I : Nat} (left right : Fin I → Expr n)
    (words : List ColourPath) (a b : ColourPath) (state : Bits)
    (aFits : a ∈ [] :: words) (bFits : b ∈ [] :: words)
    (capacity : ([] :: words).length ≤ 2 ^ I)
    (leftValues : ∀ i : Fin I,
      (left i).value state = (BitVec.ofNat I (([] :: words).idxOf a)).getLsbD i.val)
    (rightValues : ∀ i : Fin I,
      (right i).value state = (BitVec.ofNat I (([] :: words).idxOf b)).getLsbD i.val)
    (clean : ∀ wire, n ≤ wire → state wire = false) :
    eval (fieldEqual left right).oracle state =
      Function.update state (n + (fieldEqual left right).size) (decide (a = b)) := by
  apply operation_oracle _ _ _ clean
  simp only [fieldEqual_value, funext leftValues, funext rightValues,
    bitFields_equal_iff, residual_agreement words a b aFits bFits capacity]

end Semantic.Circuit
