import Infrastructure.DLDS












open Semantic

namespace FlowSpec

abbrev Dep        := List Formula
abbrev ColourPath := List Nat

abbrev FlowPair   := Dep × ColourPath



def depUnion (a b : Dep) : Dep := normalizeDep (a ++ b)

def depRemove (a : Dep) (φ : Formula) : Dep :=
  normalizeDep (List.removeAll a [φ])

def depSetEq (a b : Dep) : Bool :=
  a.all (fun x => b.any (fun y => y = x)) &&
  b.all (fun x => a.any (fun y => y = x))



lemma Formula.lt_irrefl : ∀ formula : Formula, ¬ formula < formula := by
  intro formula
  induction formula with
  | atom symbol =>
      simp [LT.lt, Formula.lt]
  | implication antecedent consequent ihAntecedent ihConsequent =>
      simp only [LT.lt, Formula.lt]
      rintro (h | ⟨_, h⟩)
      · exact ihAntecedent h
      · exact ihConsequent h

lemma Formula.lt_trans : ∀ {left middle right : Formula},
    left < middle → middle < right → left < right := by
  intro left middle right hLeftMiddle hMiddleRight
  induction left generalizing middle right with
  | atom leftSymbol =>
      cases middle with
      | atom middleSymbol =>
          cases right with
          | atom rightSymbol =>
              exact Std.TransCmp.lt_trans hLeftMiddle hMiddleRight
          | implication _ _ => trivial
      | implication _ _ =>
          cases right with
          | atom _ => contradiction
          | implication _ _ => trivial
  | implication leftAnt leftCon ihAnt ihCon =>
      cases middle with
      | atom _ => contradiction
      | implication middleAnt middleCon =>
          cases right with
          | atom _ => contradiction
          | implication rightAnt rightCon =>
              simp only [LT.lt, Formula.lt] at hLeftMiddle hMiddleRight ⊢
              rcases hLeftMiddle with hLeftMiddle | ⟨rfl, hLeftMiddle⟩
              · rcases hMiddleRight with hMiddleRight | ⟨rfl, _⟩
                · exact Or.inl (ihAnt hLeftMiddle hMiddleRight)
                · exact Or.inl hLeftMiddle
              · rcases hMiddleRight with hMiddleRight | ⟨rfl, hMiddleRight⟩
                · exact Or.inl hMiddleRight
                · exact Or.inr ⟨rfl, ihCon hLeftMiddle hMiddleRight⟩

lemma Formula.lt_trichotomy : ∀ left right : Formula,
    left < right ∨ left = right ∨ right < left := by
  intro left
  induction left with
  | atom leftSymbol =>
      intro right
      cases right with
      | atom rightSymbol =>
          cases h : compare leftSymbol rightSymbol with
          | lt => exact Or.inl h
          | eq =>
              exact Or.inr (Or.inl (by
                apply congrArg Formula.atom
                exact Std.LawfulEqOrd.eq_of_compare h))
          | gt => exact Or.inr (Or.inr (Std.OrientedCmp.lt_of_gt h))
      | implication _ _ => exact Or.inl trivial
  | implication leftAnt leftCon ihAnt ihCon =>
      intro right
      cases right with
      | atom _ => exact Or.inr (Or.inr trivial)
      | implication rightAnt rightCon =>
          rcases ihAnt rightAnt with h | h | h
          · exact Or.inl (Or.inl h)
          · subst rightAnt
            rcases ihCon rightCon with h | h | h
            · exact Or.inl (Or.inr ⟨rfl, h⟩)
            · exact Or.inr (Or.inl (by simp [h]))
            · exact Or.inr (Or.inr (Or.inr ⟨rfl, h⟩))
          · exact Or.inr (Or.inr (Or.inl h))

lemma Formula.lt_asymm {left right : Formula} (h : left < right) :
    ¬ right < left := by
  intro reverse
  exact Formula.lt_irrefl left (Formula.lt_trans h reverse)

lemma Formula.le_total (left right : Formula) :
    (left < right ∨ left = right) ∨ (right < left ∨ right = left) := by
  rcases Formula.lt_trichotomy left right with h | h | h
  · exact Or.inl (Or.inl h)
  · exact Or.inl (Or.inr h)
  · exact Or.inr (Or.inl h)

lemma Formula.le_trans {left middle right : Formula} :
    (left < middle ∨ left = middle) →
    (middle < right ∨ middle = right) →
    (left < right ∨ left = right) := by
  rintro (hLeftMiddle | rfl) (hMiddleRight | rfl)
  · exact Or.inl (Formula.lt_trans hLeftMiddle hMiddleRight)
  · exact Or.inl hLeftMiddle
  · exact Or.inl hMiddleRight
  · exact Or.inr rfl

lemma Formula.le_antisymm {left right : Formula} :
    (left < right ∨ left = right) →
    (right < left ∨ right = left) → left = right := by
  rintro (hLeftRight | rfl) (hRightLeft | hRightLeft)
  · exact (Formula.lt_asymm hLeftRight hRightLeft).elim
  · exact hRightLeft.symm
  · rfl
  · rfl

lemma Formula.leB_total (left right : Formula) :
    (Formula.leB left right || Formula.leB right left) = true := by
  simp only [Formula.leB, decide_eq_true_eq, Bool.or_eq_true]
  exact Formula.le_total left right

lemma Formula.leB_trans (left middle right : Formula)
    (hLeftMiddle : Formula.leB left middle = true)
    (hMiddleRight : Formula.leB middle right = true) :
    Formula.leB left right = true := by
  simp only [Formula.leB, decide_eq_true_eq] at hLeftMiddle hMiddleRight ⊢
  exact Formula.le_trans hLeftMiddle hMiddleRight

lemma Formula.leB_antisymm {left right : Formula}
    (hLeftRight : Formula.leB left right = true)
    (hRightLeft : Formula.leB right left = true) : left = right := by
  simp only [Formula.leB, decide_eq_true_eq] at hLeftRight hRightLeft
  exact Formula.le_antisymm hLeftRight hRightLeft



def CanonicalDep (dependency : Dep) : Prop :=
  dependency.Nodup ∧
    dependency.Pairwise (fun left right => Formula.leB left right = true)

private lemma nodup_eraseDups_formula : ∀ dependency : Dep,
    dependency.eraseDups.Nodup
  | [] => by simp
  | formula :: rest => by
      rw [List.eraseDups_cons]
      constructor
      · intro selection hSelection hEq
        rw [List.mem_eraseDups, List.mem_filter] at hSelection
        subst selection
        simp at hSelection
      · exact nodup_eraseDups_formula
          (rest.filter fun selection => !selection == formula)
termination_by dependency => dependency.length
decreasing_by
  simpa using List.length_filter_le
    (fun selection => !selection == formula) rest


theorem normalizeDep_canonical (dependency : Dep) :
    CanonicalDep (normalizeDep dependency) := by
  constructor
  · exact List.nodup_mergeSort.mpr (nodup_eraseDups_formula dependency)
  · exact List.pairwise_mergeSort Formula.leB_trans Formula.leB_total _

@[simp] theorem normalizeDep_nil : normalizeDep [] = [] := by
  simp [normalizeDep]

@[simp] theorem normalizeDep_singleton (formula : Formula) :
    normalizeDep [formula] = [formula] := by
  unfold normalizeDep
  rw [show [formula].eraseDups = [formula] by rfl,
    List.mergeSort_singleton]

@[simp] theorem mem_normalizeDep {formula : Formula} {dependency : Dep} :
    formula ∈ normalizeDep dependency ↔ formula ∈ dependency := by
  exact (List.mergeSort_perm dependency.eraseDups Formula.leB).mem_iff.trans
    List.mem_eraseDups

theorem depUnion_canonical (left right : Dep) :
    CanonicalDep (depUnion left right) := by
  exact normalizeDep_canonical _

theorem depRemove_canonical (dependency : Dep) (formula : Formula) :
    CanonicalDep (depRemove dependency formula) := by
  exact normalizeDep_canonical _

lemma depSetEq_eq_true_iff (left right : Dep) :
    depSetEq left right = true ↔
      ∀ formula, formula ∈ left ↔ formula ∈ right := by
  simp only [depSetEq, Bool.and_eq_true, List.all_eq_true, List.any_eq_true]
  constructor
  · rintro ⟨hLeft, hRight⟩ formula
    constructor
    · intro hFormula
      obtain ⟨selection, hSelection, hEq⟩ := hLeft formula hFormula
      have hEq' : selection = formula := of_decide_eq_true hEq
      simpa [hEq'] using hSelection
    · intro hFormula
      obtain ⟨selection, hSelection, hEq⟩ := hRight formula hFormula
      have hEq' : selection = formula := of_decide_eq_true hEq
      simpa [hEq'] using hSelection
  · intro hMembership
    constructor
    · intro formula hFormula
      exact ⟨formula, (hMembership formula).mp hFormula, by simp⟩
    · intro formula hFormula
      exact ⟨formula, (hMembership formula).mpr hFormula, by simp⟩



theorem canonicalDep_eq_of_depSetEq {left right : Dep}
    (hLeft : CanonicalDep left) (hRight : CanonicalDep right)
    (hSetEq : depSetEq left right = true) : left = right := by
  have hMembership := (depSetEq_eq_true_iff left right).mp hSetEq
  have hPermutation : left.Perm right :=
    (List.perm_ext_iff_of_nodup hLeft.1 hRight.1).mpr hMembership
  exact List.Perm.eq_of_pairwise
    (fun first second _ _ hFirstSecond hSecondFirst =>
      Formula.leB_antisymm hFirstSecond hSecondFirst)
    hLeft.2 hRight.2 hPermutation




def predsOf (d : Graph) (v : Vertex) : List Vertex :=
  ((d.EDGES.filter (fun e => e.END = v)).map (·.START)).eraseDups

def edgesBetween (d : Graph) (u v : Vertex) : List Deduction :=
  d.EDGES.filter (fun e => e.START = u ∧ e.END = v)


def edgeOfColour (d : Graph) (v : Vertex) (c : Nat) : Option Deduction :=
  (d.EDGES.filter (fun e => e.START = v ∧ e.COLOUR = c)).head?

def ancestorsInto (d : Graph) (v : Vertex) : List Ancestral :=
  d.PATHS.filter (fun a => a.END = v)



def Pre (d : Graph) (w : Vertex) : List Vertex :=
  go (d.NODES.length + 1) [w] []
where
  go : Nat → List Vertex → List Vertex → List Vertex
  | 0,        _,        acc => acc
  | fuel + 1, frontier, acc =>
      let fresh := ((frontier.flatMap (predsOf d)).eraseDups).filter
        (fun u => !(acc.any (fun a => a = u)))
      if fresh.isEmpty then acc
      else go fuel fresh (acc ++ fresh)







def consume (e : Deduction) (p : ColourPath) : Option ColourPath :=
  match p with
  | []      => if e.COLOUR = 0 then some [] else none
  | o :: p' => if e.COLOUR = o then some p' else none

def routeContinuableAtB (d : Graph) (v : Vertex) (residual : ColourPath) : Bool :=
  let outgoing := get_rule.outgoing v d
  if outgoing.isEmpty then residual.isEmpty
  else outgoing.any fun edge => (consume edge residual).isSome





def RouteContinuableAt (d : Graph) (v : Vertex) (residual : ColourPath) : Prop :=
  routeContinuableAtB d v residual = true

lemma routeContinuableAtB_eq_true_iff (d : Graph) (v : Vertex)
    (residual : ColourPath) :
    routeContinuableAtB d v residual = true ↔
      RouteContinuableAt d v residual := Iff.rfl

lemma routeContinuableAtB_of_consume_outgoing (d : Graph) (v : Vertex)
    (edge : Deduction) (residual tail : ColourPath)
    (hedge : edge ∈ get_rule.outgoing v d)
    (hconsume : consume edge residual = some tail) :
    routeContinuableAtB d v residual = true := by
  unfold routeContinuableAtB
  have hout : (get_rule.outgoing v d).isEmpty = false := by
    cases hempty : (get_rule.outgoing v d).isEmpty with
    | false => rfl
    | true =>
        rw [List.isEmpty_iff] at hempty
        rw [hempty] at hedge
        exact (List.not_mem_nil hedge).elim
  simp only [hout, Bool.false_eq_true, ↓reduceIte, List.any_eq_true]
  exact ⟨edge, hedge, by simp [hconsume]⟩

lemma routeContinuableAtB_of_consume_edgesBetween (d : Graph)
    (start finish : Vertex) (edge : Deduction) (residual tail : ColourPath)
    (hedge : edge ∈ edgesBetween d start finish)
    (hconsume : consume edge residual = some tail) :
    routeContinuableAtB d start residual = true := by
  unfold edgesBetween at hedge
  rw [List.mem_filter] at hedge
  have hendpoints : edge.START = start ∧ edge.END = finish :=
    of_decide_eq_true hedge.2
  exact routeContinuableAtB_of_consume_outgoing d start edge residual tail
    (mem_outgoing_of_mem_edges_start_eq start d hedge.1 hendpoints.1) hconsume


def WitnessableRoutes (d : Graph) (v : Vertex) (routes : List FlowPair) :
    List FlowPair :=
  routes.filter fun route => routeContinuableAtB d v route.2

def leadsTo (d : Graph) : Nat → Vertex → ColourPath → Vertex → Bool
  | 0,        _, _, _ => false
  | fuel + 1, v, p, w =>
      if v = w then p.isEmpty
      else
        match edgeOfColour d v (p.headD 0) with
        | none   => false
        | some e => leadsTo d fuel e.END p.tail w


def elimPairsAt (d : Graph) (v : Vertex) : List (Vertex × Vertex) :=
  let ps := predsOf d v
  ps.flatMap (fun major =>
    ps.filterMap (fun minor =>
      if major.FORMULA = Formula.implication minor.FORMULA v.FORMULA
      then some (major, minor) else none))



def elimPremiseAtB (d : Graph) (v u : Vertex) : Bool :=
  (elimPairsAt d v).any fun (major, minor) => major == u || minor == u







def mixedIntroPairs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) : List FlowPair :=
  match v.FORMULA with
  | Formula.implication alpha beta =>
      (predsOf d v).flatMap fun premise =>
        if !elimPremiseAtB d v premise && premise.FORMULA == beta then
          (premiseFlow premise).flatMap fun (b, p) =>
            (edgesBetween d premise v).filterMap fun edge =>
              (consume edge p).map fun q => (depRemove b alpha, q)
        else []
  | _ => []

def uniqueRoot? (d : Graph) : Option Vertex :=
  match d.NODES.filter fun v => (get_rule.outgoing v d).isEmpty with
  | [root] => some root
  | _ => none

def depSimilarB (left right : Dep) : Bool :=
  depSetEq left right || left.isEmpty || right.isEmpty





def reseedPremiseAgreesB (route : FlowPair) (edge : Deduction) : Bool :=
  (consume edge route.2).isSome

def reseedZeroPremiseB (route : FlowPair) (edge : Deduction) : Bool :=
  edge.COLOUR == 0 && (consume edge route.2).isSome

def rootSeedEligibleWithPreB (d : Graph) (root : Vertex)
    (rootPre : List Vertex) (v : Vertex) (ancestor : Ancestral) : Bool :=
  match ancestor.COLOURS with
  | head :: _ =>
      d.NODES.contains ancestor.START &&
      rootPre.contains ancestor.START &&
      leadsTo d (d.NODES.length + 1) v ancestor.COLOURS ancestor.START &&
      (get_rule.outgoing v d).any fun edge =>
        edge.COLOUR == head &&
          (rootPre.contains edge.END || edge.END == root)
  | [] => false

def eligibleAncestorsAtRoot (d : Graph) (v : Vertex) : List Ancestral :=
  match uniqueRoot? d with
  | none => []
  | some root =>
      let rootPre := Pre d root
      (ancestorsInto d v).filter (rootSeedEligibleWithPreB d root rootPre v)





def colourPathPrefixB : ColourPath → ColourPath → Bool
  | [], _ => true
  | _ :: _, [] => false
  | left :: lefts, right :: rights =>
      left == right && colourPathPrefixB lefts rights




def reseedPairSupportsB (d : Graph) (w : Vertex)
    (minorRoute : FlowPair) (minorEdge : Deduction)
    (majorRoute : FlowPair) (majorEdge : Deduction) : Bool :=
  reseedPremiseAgreesB minorRoute minorEdge &&
  reseedPremiseAgreesB majorRoute majorEdge &&
  depSimilarB minorRoute.1 minorEdge.DEPENDENCY &&
  depSimilarB majorRoute.1 majorEdge.DEPENDENCY &&
  match consume minorEdge minorRoute.2, consume majorEdge majorRoute.2 with
  | some minorTail, some majorTail =>
      (eligibleAncestorsAtRoot d w).any fun ancestor =>
        colourPathPrefixB minorTail ancestor.COLOURS &&
        colourPathPrefixB majorTail ancestor.COLOURS
  | _, _ => false

theorem reseedPairSupportsB_eq_false_of_no_eligible
    (d : Graph) (w : Vertex) (minorRoute : FlowPair)
    (minorEdge : Deduction) (majorRoute : FlowPair) (majorEdge : Deduction)
    (h : eligibleAncestorsAtRoot d w = []) :
    reseedPairSupportsB d w minorRoute minorEdge majorRoute majorEdge = false := by
  cases hminor : consume minorEdge minorRoute.2 <;>
    cases hmajor : consume majorEdge majorRoute.2 <;>
      simp [reseedPairSupportsB, h, hminor, hmajor]

def elimReseedPairs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) (ancestors : List Ancestral) :
    List FlowPair :=
  (elimPairsAt d v).flatMap fun (major, minor) =>
    (premiseFlow minor).flatMap fun minorRoute =>
      (premiseFlow major).flatMap fun majorRoute =>
        (edgesBetween d minor v).flatMap fun minorEdge =>
          (edgesBetween d major v).flatMap fun majorEdge =>
            if reseedPremiseAgreesB minorRoute minorEdge &&
                reseedPremiseAgreesB majorRoute majorEdge &&
                depSimilarB minorRoute.1 minorEdge.DEPENDENCY &&
                depSimilarB majorRoute.1 majorEdge.DEPENDENCY then
              ancestors.map fun ancestor =>
                (depUnion minorRoute.1 majorRoute.1, ancestor.COLOURS)
            else []

def introReseedPairs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) (ancestors : List Ancestral) :
    List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [premise], Formula.implication alpha beta =>
      if premise.FORMULA == beta then
        (premiseFlow premise).flatMap fun route =>
          (edgesBetween d premise v).flatMap fun edge =>
            if reseedZeroPremiseB route edge &&
                depSimilarB route.1 edge.DEPENDENCY then
              ancestors.map fun ancestor =>
                (depRemove route.1 alpha, ancestor.COLOURS)
            else []
      else []
  | _, _ => []




def reseedPairs (d : Graph) (v : Vertex)
    (premiseFlow : Vertex → List FlowPair) : List FlowPair :=
  let ancestors := eligibleAncestorsAtRoot d v
  if ancestors.isEmpty then []
  else (elimReseedPairs d v premiseFlow ancestors ++
    introReseedPairs d v premiseFlow ancestors).eraseDups

lemma eligibleAncestorsAtRoot_eq_nil_of_ancestorsInto_eq_nil
    (d : Graph) (v : Vertex) (h : ancestorsInto d v = []) :
    eligibleAncestorsAtRoot d v = [] := by
  unfold eligibleAncestorsAtRoot
  split <;> simp [h]

lemma reseedPairs_eq_nil_of_ancestorsInto_eq_nil
    (d : Graph) (v : Vertex) (premiseFlow : Vertex → List FlowPair)
    (h : ancestorsInto d v = []) :
    reseedPairs d v premiseFlow = [] := by
  unfold reseedPairs
  rw [eligibleAncestorsAtRoot_eq_nil_of_ancestorsInto_eq_nil d v h]
  simp







def flowAt (d : Graph) : Nat → Vertex → List FlowPair
  | 0, _ => []
  | fuel + 1, v =>
      let preds := predsOf d v
      if v.HYPOTHESIS || preds.isEmpty then
        let anc := ancestorsInto d v
        if anc.isEmpty then [(normalizeDep [v.FORMULA], [])]
        else anc.map (fun a => (normalizeDep [v.FORMULA], a.COLOURS))
      else
        let elim : List FlowPair :=
          (elimPairsAt d v).flatMap (fun (major, minor) =>
            (flowAt d fuel minor).flatMap (fun (b₁, p₁) =>
              (flowAt d fuel major).flatMap (fun (b₂, p₂) =>
                (edgesBetween d minor v).flatMap (fun e₁ =>
                  (edgesBetween d major v).filterMap (fun e₂ =>
                    match consume e₁ p₁, consume e₂ p₂ with
                    | some q₁, some q₂ =>
                        if q₁ = q₂ then some (depUnion b₁ b₂, q₁) else none
                    | _, _ => none)))))
        let intro : List FlowPair :=
          match preds, v.FORMULA with
          | [u], Formula.implication α β =>
              if u.FORMULA = β then
                (flowAt d fuel u).flatMap (fun (b, p) =>
                  (edgesBetween d u v).filterMap (fun e =>
                    (consume e p).map (fun q => (depRemove b α, q))))
              else []
          | _, _ => []
        let core := if (elimPairsAt d v).isEmpty then intro
          else elim ++ mixedIntroPairs d v (flowAt d fuel)
        (core ++ reseedPairs d v (flowAt d fuel)).eraseDups



set_option maxHeartbeats 1000000 in
theorem flowAt_dependencies_canonical (d : Graph) :
    ∀ fuel v route, route ∈ flowAt d fuel v → CanonicalDep route.1 := by
  intro fuel v route hroute
  cases fuel with
  | zero => simp [flowAt] at hroute
  | succ fuel =>
      rw [flowAt] at hroute
      split at hroute
      · dsimp at hroute
        have hcanonical : CanonicalDep (normalizeDep [v.FORMULA]) :=
          normalizeDep_canonical [v.FORMULA]
        split at hroute
        · simp only [List.mem_singleton] at hroute
          subst route
          exact hcanonical
        · rw [List.mem_map] at hroute
          obtain ⟨ancestor, _, rfl⟩ := hroute
          exact hcanonical
      · simp_all [mixedIntroPairs, reseedPairs, elimReseedPairs,
          introReseedPairs]
        all_goals
          aesop (add safe apply depUnion_canonical)
            (add safe apply depRemove_canonical)










abbrev FlowTable := List (Vertex × List FlowPair)

def flowTableLookup? (table : FlowTable) (v : Vertex) : Option (List FlowPair) :=
  (table.find? fun entry => entry.1 == v).map (fun entry => entry.2)

def flowFromTable (d : Graph) (fuel : Nat) (table : FlowTable)
    (v : Vertex) : List FlowPair :=
  match flowTableLookup? table v with
  | some routes => routes
  | none => flowAt d fuel v

def flowStepFrom (d : Graph) (premiseFlow : Vertex → List FlowPair)
    (v : Vertex) : List FlowPair :=
  let preds := predsOf d v
  if v.HYPOTHESIS || preds.isEmpty then
    let anc := ancestorsInto d v
    if anc.isEmpty then [(normalizeDep [v.FORMULA], [])]
    else anc.map (fun a => (normalizeDep [v.FORMULA], a.COLOURS))
  else
    let elim : List FlowPair :=
      (elimPairsAt d v).flatMap (fun (major, minor) =>
        (premiseFlow minor).flatMap (fun (b₁, p₁) =>
          (premiseFlow major).flatMap (fun (b₂, p₂) =>
            (edgesBetween d minor v).flatMap (fun e₁ =>
              (edgesBetween d major v).filterMap (fun e₂ =>
                match consume e₁ p₁, consume e₂ p₂ with
                | some q₁, some q₂ =>
                    if q₁ = q₂ then some (depUnion b₁ b₂, q₁) else none
                | _, _ => none)))))
    let intro : List FlowPair :=
      match preds, v.FORMULA with
      | [u], Formula.implication α β =>
          if u.FORMULA = β then
            (premiseFlow u).flatMap (fun (b, p) =>
              (edgesBetween d u v).filterMap (fun e =>
                (consume e p).map (fun q => (depRemove b α, q))))
          else []
      | _, _ => []
    let core := if (elimPairsAt d v).isEmpty then intro
      else elim ++ mixedIntroPairs d v premiseFlow
    (core ++ reseedPairs d v premiseFlow).eraseDups

def flowTableInitial (d : Graph) : FlowTable :=
  d.NODES.map fun v => (v, [])

def flowTableStep (d : Graph) (fuel : Nat) (previous : FlowTable) :
    FlowTable :=
  d.NODES.map fun v =>
    (v, flowStepFrom d (flowFromTable d fuel previous) v)

def flowTableClosedB (d : Graph) : Bool :=
  d.EDGES.all fun edge =>
    d.NODES.contains edge.START && d.NODES.contains edge.END



def flowTableShape (table : FlowTable) : List Nat :=
  table.map fun entry => entry.2.length

def flowTableIterate (d : Graph) (mayStop : Bool) : Nat → Nat → FlowTable → FlowTable
  | 0, _, previous => previous
  | remaining + 1, fuel, previous =>
      let next := flowTableStep d fuel previous
      if mayStop && flowTableShape next == flowTableShape previous &&
          next == previous then
        next
      else
        flowTableIterate d mayStop remaining (fuel + 1) next

def flowTableAt (d : Graph) (fuel : Nat) : FlowTable :=
  flowTableIterate d (flowTableClosedB d) fuel 0 (flowTableInitial d)

def flowTable (d : Graph) : FlowTable :=
  flowTableAt d (d.NODES.length + 1)

def cachedFlowAt (d : Graph) (table : FlowTable) (v : Vertex) : List FlowPair :=
  flowFromTable d (d.NODES.length + 1) table v






def generativeRoutes (d : Graph) (w v : Vertex) : List FlowPair :=
  (flowAt d (d.NODES.length + 1) v).filter
    (fun bp => leadsTo d (d.NODES.length + 1) v bp.2 w)

def edgeFlowDeps (d : Graph) (e : Deduction) : List Dep :=
  ((d.NODES.flatMap (fun w =>
      (generativeRoutes d w e.START).filterMap (fun bp =>
        (consume e bp.2).map (fun _ => bp.1)))).eraseDups)

end FlowSpec
