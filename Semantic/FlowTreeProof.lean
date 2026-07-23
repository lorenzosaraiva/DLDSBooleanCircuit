import Semantic.FlowValidity
set_option linter.unusedSimpArgs false

/-!
# FlowTreeProof ; the Def-23 singleton bullet on the simple-tree fragment

`flowRuleCorrect_of_simpleTree` states that on a valid simple tree
(`IsSimpleTreeDLDS`, `ValidDLDS`), every node satisfies the literal Def-23
CorrectRuleApp predicate
`FlowRuleCorrect` ; tops and the root by the domain conditions, every other
kernel node through the SINGLETON bullet.

Core lemma (`flowAt_simpleTree`): in such a tree, the Flow at any node with
an outgoing edge is the singleton `[(outDep v d, [])]` ; i.e. Flow propagates
exactly the dependency the outDep-based `LocalRuleCorrect` stores, with an
exhausted residual. Induction on fuel, measured by the number of nodes at
strictly higher level (premises sit one level up by `LeveledColored`).

The collapsed-node Φᵢ case requires the coloured-route machinery developed in
the compressed modules.
-/

open Semantic

namespace FlowSpec



lemma incoming_loop_eq_filter (v : Vertex) :
    ∀ edges : List Deduction,
      get_rule.incoming.loop v edges = edges.filter (fun e => decide (e.END = v))
  | [] => by rw [get_rule.incoming.loop.eq_1]; rfl
  | e :: es => by
      rw [get_rule.incoming.loop.eq_2]
      by_cases h : e.END = v
      · simp [h, incoming_loop_eq_filter v es]
      · simp [h, incoming_loop_eq_filter v es]

lemma incoming_eq_filter (v : Vertex) (d : Graph) :
    get_rule.incoming v d = d.EDGES.filter (fun e => decide (e.END = v)) := by
  rw [get_rule.incoming.eq_1]
  exact incoming_loop_eq_filter v d.EDGES

lemma outgoing_loop_eq_filter (v : Vertex) :
    ∀ edges : List Deduction,
      get_rule.outgoing.loop v edges = edges.filter (fun e => decide (e.START = v))
  | [] => by rw [get_rule.outgoing.loop.eq_1]; rfl
  | e :: es => by
      rw [get_rule.outgoing.loop.eq_2]
      by_cases h : e.START = v
      · simp [h, outgoing_loop_eq_filter v es]
      · simp [h, outgoing_loop_eq_filter v es]

lemma outgoing_eq_filter (v : Vertex) (d : Graph) :
    get_rule.outgoing v d = d.EDGES.filter (fun e => decide (e.START = v)) := by
  rw [get_rule.outgoing.eq_1]
  exact outgoing_loop_eq_filter v d.EDGES

/--  `predsOf` through `get_rule.incoming`.  -/
lemma predsOf_eq (d : Graph) (v : Vertex) :
    predsOf d v = ((get_rule.incoming v d).map (·.START)).eraseDups := by
  unfold predsOf
  rw [incoming_eq_filter]

/--  `edgesBetween` through `get_rule.outgoing`.  -/
lemma edgesBetween_eq (d : Graph) (u v : Vertex) :
    edgesBetween d u v =
      (get_rule.outgoing u d).filter (fun e => decide (e.END = v)) := by
  unfold edgesBetween
  rw [outgoing_eq_filter, List.filter_filter]
  apply List.filter_congr
  intro e _
  simp [Bool.and_comm]



lemma formula_ne_implication_self (α β : Formula) :
    ¬ (α = Formula.implication α β) := by
  intro h
  have hs : sizeOf α = sizeOf (Formula.implication α β) := congrArg sizeOf h
  simp at hs
  omega

/--  Two formulas cannot each be an implication with the other as antecedent.  -/
lemma not_mutual_implication {a b c : Formula}
    (h1 : a = Formula.implication b c) :
    ¬ (b = Formula.implication a c) := by
  intro h2
  have s1 : sizeOf a = 1 + sizeOf b + sizeOf c := by rw [h1]; simp
  have s2 : sizeOf b = 1 + sizeOf a + sizeOf c := by rw [h2]; simp
  omega



lemma eraseDups_singleton {α : Type _} [BEq α] (a : α) :
    [a].eraseDups = [a] := rfl

lemma eraseDups_pair {a b : Vertex} (h : a ≠ b) :
    [a, b].eraseDups = [a, b] := by
  have hbeq : (b == a) = false := by
    rw [beq_eq_false_iff_ne]
    exact fun hh => h hh.symm
  simp [List.eraseDups, List.eraseDupsBy, List.eraseDupsBy.loop, hbeq]

lemma depSetEq_self (l : Dep) : depSetEq l l = true := by
  unfold depSetEq
  rw [Bool.and_eq_true]
  constructor <;>
  · rw [List.all_eq_true]
    intro x hx
    rw [List.any_eq_true]
    exact ⟨x, hx, by simp⟩

/--  `depRemove` agrees with the repo's `−` (eraseDups ∘ removeAll).  -/
lemma depRemove_eq (l : Dep) (α : Formula) :
    depRemove l α = List.eraseDups (List.removeAll l [α]) := by
  unfold depRemove
  congr 1
  unfold List.removeAll
  apply List.filter_congr
  intro x _
  simp [List.elem_cons, List.elem_nil]

/--
 Strictly fewer `p`-elements than `q`-elements when `p ⇒ q` pointwise and
    some member witnesses `q ∧ ¬p`. (Hand-rolled to avoid name drift.)
-/
lemma filter_length_lt_of_mem {α : Type _} {p q : α → Bool} :
    ∀ l : List α, (∀ x ∈ l, p x = true → q x = true) →
      ∀ a ∈ l, q a = true → p a = false →
      (l.filter p).length < (l.filter q).length := by
  intro l
  induction l with
  | nil => intro _ a ha; simp at ha
  | cons x xs ih =>
      intro hmono a ha hqa hpa
      have hmono' : ∀ y ∈ xs, p y = true → q y = true :=
        fun y hy => hmono y (List.mem_cons_of_mem x hy)
      have hle : ∀ (l' : List α), (∀ y ∈ l', p y = true → q y = true) →
          (l'.filter p).length ≤ (l'.filter q).length := by
        intro l'
        induction l' with
        | nil => intro _; simp
        | cons z zs ihz =>
            intro hm
            have hm' : ∀ y ∈ zs, p y = true → q y = true :=
              fun y hy => hm y (List.mem_cons_of_mem z hy)
            by_cases hpz : p z = true
            · have hqz := hm z (List.mem_cons_self ..) hpz
              simp [List.filter_cons, hpz, hqz]
              exact ihz hm'
            · have hpz' : p z = false := by
                cases h : p z with
                | true => exact absurd h hpz
                | false => rfl
              by_cases hqz : q z = true
              · simp [List.filter_cons, hpz', hqz]
                exact Nat.le_succ_of_le (ihz hm')
              · have hqz' : q z = false := by
                  cases h : q z with
                  | true => exact absurd h hqz
                  | false => rfl
                simp [List.filter_cons, hpz', hqz']
                exact ihz hm'
      by_cases hpx : p x = true
      · have hqx := hmono x (List.mem_cons_self ..) hpx
        have hax : a ∈ xs := by
          cases ha with
          | head => rw [hpx] at hpa; cases hpa
          | tail _ h => exact h
        have := ih hmono' a hax hqa hpa
        simp [List.filter_cons, hpx, hqx]
        omega
      · have hpx' : p x = false := by
          cases h : p x with
          | true => exact absurd h hpx
          | false => rfl
        by_cases hqx : q x = true
        · have := hle xs hmono'
          simp [List.filter_cons, hpx', hqx]
          omega
        · have hqx' : q x = false := by
            cases h : q x with
            | true => exact absurd h hqx
            | false => rfl
          have hax : a ∈ xs := by
            cases ha with
            | head => rw [hqx'] at hqa; cases hqa
            | tail _ h => exact h
          have := ih hmono' a hax hqa hpa
          simp [List.filter_cons, hpx', hqx']
          omega



/--
 Number of nodes at a strictly higher level than `v`. Premises sit one
    level up (`LeveledColored`), so this strictly decreases toward the tops.
-/
def countAbove (d : Graph) (v : Vertex) : Nat :=
  (d.NODES.filter (fun u => decide (v.LEVEL < u.LEVEL))).length

lemma countAbove_le (d : Graph) (v : Vertex) :
    countAbove d v ≤ d.NODES.length :=
  List.length_filter_le _ _

lemma countAbove_lt_of_level_succ (d : Graph) {u v : Vertex}
    (hu : u ∈ d.NODES) (hlvl : u.LEVEL = v.LEVEL + 1) :
    countAbove d u < countAbove d v := by
  apply filter_length_lt_of_mem d.NODES ?_ u hu ?_ ?_
  · intro x _ hx
    have hx' : u.LEVEL < x.LEVEL := of_decide_eq_true hx
    exact decide_eq_true (by omega)
  · exact decide_eq_true (by omega)
  · simp



lemma classifyRule?_intro_inversion {d : Graph} {v : Vertex} {p : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.intro p)) :
    v.HYPOTHESIS = false ∧ get_rule.incoming v d = [p] ∧
      consequent? v.FORMULA = some p.START.FORMULA := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · have hhypf : v.HYPOTHESIS = false := by
      cases h : v.HYPOTHESIS with
      | true => exact absurd h hhyp
      | false => rfl
    simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil => simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil =>
            simp [hinc] at hclass
            obtain ⟨hcons, hp⟩ := hclass
            subst hp
            exact ⟨hhypf, rfl, hcons⟩
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                split at hclass <;> simp at hclass
            | cons e3 es3 => simp [hinc] at hclass

lemma classifyRule?_elim_inversion {d : Graph} {v : Vertex}
    {major minor : Deduction}
    (hclass : classifyRule? v d = some (DLDSRuleClass.elim major minor)) :
    v.HYPOTHESIS = false ∧
      (get_rule.incoming v d = [major, minor] ∨
       get_rule.incoming v d = [minor, major]) := by
  unfold classifyRule? at hclass
  by_cases hhyp : v.HYPOTHESIS = true
  · simp [hhyp] at hclass
  · have hhypf : v.HYPOTHESIS = false := by
      cases h : v.HYPOTHESIS with
      | true => exact absurd h hhyp
      | false => rfl
    simp [hhyp] at hclass
    cases hinc : get_rule.incoming v d with
    | nil => simp [hinc] at hclass
    | cons e es =>
        cases es with
        | nil => simp [hinc] at hclass
        | cons e2 es2 =>
            cases es2 with
            | nil =>
                simp [hinc] at hclass
                by_cases h12 :
                    e.START.FORMULA = Formula.implication e2.START.FORMULA v.FORMULA
                · simp [h12] at hclass
                  obtain ⟨hmaj, hmin⟩ := hclass
                  subst hmaj; subst hmin
                  exact ⟨hhypf, Or.inl rfl⟩
                · simp [h12] at hclass
                  by_cases h21 :
                      e2.START.FORMULA = Formula.implication e.START.FORMULA v.FORMULA
                  · simp [h21] at hclass
                    obtain ⟨hmaj, hmin⟩ := hclass
                    subst hmaj; subst hmin
                    exact ⟨hhypf, Or.inr rfl⟩
                  · simp [h21] at hclass
            | cons e3 es3 => simp [hinc] at hclass



lemma elimPairsAt_pair (d : Graph) (v : Vertex) {M S : Vertex}
    (hpreds : predsOf d v = [M, S] ∨ predsOf d v = [S, M])
    (hMF : M.FORMULA = Formula.implication S.FORMULA v.FORMULA) :
    elimPairsAt d v = [(M, S)] := by
  have h1 : ¬ (M.FORMULA = Formula.implication M.FORMULA v.FORMULA) := by
    intro h
    rw [hMF] at h
    have := (Formula.implication.injEq _ _ _ _).mp h
    exact formula_ne_implication_self _ _ this.1
  have h2 : ¬ (S.FORMULA = Formula.implication S.FORMULA v.FORMULA) :=
    formula_ne_implication_self _ _
  have h3 : ¬ (S.FORMULA = Formula.implication M.FORMULA v.FORMULA) :=
    not_mutual_implication hMF
  cases hpreds with
  | inl h =>
      unfold elimPairsAt
      rw [h]
      simp only [List.flatMap_cons, List.flatMap_nil, List.filterMap_cons,
        List.filterMap_nil]
      rw [if_neg h1, if_pos hMF, if_neg h3, if_neg h2]
      rfl
  | inr h =>
      unfold elimPairsAt
      rw [h]
      simp only [List.flatMap_cons, List.flatMap_nil, List.filterMap_cons,
        List.filterMap_nil]
      rw [if_neg h2, if_neg h3, if_pos hMF, if_neg h1]
      rfl



/--
 In a valid simple tree with default-coloured edges, the Flow at any node
    with an outgoing edge is exactly `[(outDep v d, [])]` ; one route,
    carrying the propagated dependency, residual exhausted.
-/
lemma flowAt_simpleTree (d : Graph) (htree : IsSimpleTreeDLDS d)
    (hvalid : ValidDLDS d) :
    ∀ fuel : Nat, ∀ v : Vertex, v ∈ d.NODES →
      get_rule.outgoing v d ≠ [] → countAbove d v < fuel →
      flowAt d fuel v = [(outDep v d, [])] := by
  have hcol : ∀ e ∈ d.EDGES, e.COLOUR = 0 := htree.1.2.2.2
  intro fuel
  induction fuel with
  | zero => intro v _ _ hcount; omega
  | succ fuel ih =>
      intro v hv hout hcount
      obtain ⟨hshape, hdeps⟩ := hvalid.localRuleCorrect v hv hout
      have hsome : (classifyRule? v d).isSome = true := hshape
      obtain ⟨cls, hclass⟩ := Option.isSome_iff_exists.mp hsome
      cases cls with
      | hypothesis =>
          have hhyp : v.HYPOTHESIS = true := classifyRule?_hypothesis_hyp hclass
          have hpaths : d.PATHS = [] := htree.1.2.2.1
          rw [flowAt]
          simp [hhyp, ancestorsInto, hpaths, outDep, hclass]
      | intro p =>
          obtain ⟨hhypf, hinc, hcons⟩ := classifyRule?_intro_inversion hclass
          have hpin : p ∈ get_rule.incoming v d := by
            rw [hinc]; exact List.mem_singleton_self p
          have hpe : p ∈ d.EDGES := mem_incoming_mem_edges v d hpin
          have hpend : p.END = v := mem_incoming_end_eq v d hpin
          have hps_mem : p.START ∈ d.NODES := (hvalid.hygiene.2.1 hpe).1
          have hpout : p ∈ get_rule.outgoing p.START d :=
            mem_outgoing_of_mem_edges_start_eq p.START d hpe rfl
          have hpoutne : get_rule.outgoing p.START d ≠ [] := by
            intro h; rw [h] at hpout; exact List.not_mem_nil hpout
          have hlvl : p.START.LEVEL = v.LEVEL + 1 := by
            have := hvalid.leveledColored p hpe
            rwa [hpend] at this
          have hca : countAbove d p.START < fuel := by
            have := countAbove_lt_of_level_succ d hps_mem hlvl
            omega
          have hIH := ih p.START hps_mem hpoutne hca
          have hpdep : p.DEPENDENCY = outDep p.START d :=
            (hvalid.localRuleCorrect p.START hps_mem hpoutne).2 p hpout
          obtain ⟨α, hform⟩ : ∃ α, v.FORMULA = Formula.implication α p.START.FORMULA := by
            cases hf : v.FORMULA with
            | atom s => rw [hf] at hcons; simp [consequent?] at hcons
            | implication A B =>
                rw [hf] at hcons
                simp [consequent?] at hcons
                exact ⟨A, by rw [hcons]⟩
          have hpreds : predsOf d v = [p.START] := by
            rw [predsOf_eq, hinc]
            rfl
          have heb : edgesBetween d p.START v = [p] := by
            rw [edgesBetween_eq,
              simpleTree_outgoing_eq_singleton d htree hvalid hpout]
            simp [hpend]
          have hc0 : p.COLOUR = 0 := hcol p hpe
          have hep : elimPairsAt d v = [] := by
            unfold elimPairsAt
            rw [hpreds]
            simp only [List.flatMap_cons, List.flatMap_nil, List.filterMap_cons,
              List.filterMap_nil]
            rw [if_neg (formula_ne_implication_self p.START.FORMULA v.FORMULA)]
            rfl
          rw [flowAt]
          rw [hhypf]
          simp only [Bool.false_or, hpreds, List.isEmpty_cons, if_false,
            Bool.false_eq_true]
          rw [hep]
          simp only [List.flatMap_nil, List.isEmpty_nil, if_true]
          rw [hform]
          simp only [hIH, heb, List.flatMap_cons, List.flatMap_nil,
            List.filterMap_cons, List.filterMap_nil]
          simp [consume, hc0, reseedPairs, outDep, hclass, hform, antecedent?,
            hpdep, depRemove_eq, eraseDups_singleton]
      | elim major minor =>
          obtain ⟨hhypf, hincor⟩ := classifyRule?_elim_inversion hclass
          have hMF : major.START.FORMULA =
              Formula.implication minor.START.FORMULA v.FORMULA :=
            classifyRule?_elim_major_formula_eq_minor hclass
          have hmaj_in : major ∈ get_rule.incoming v d := by
            cases hincor with
            | inl h => rw [h]; exact List.mem_cons_self ..
            | inr h => rw [h]; exact List.mem_cons_of_mem _ (List.mem_singleton_self _)
          have hmin_in : minor ∈ get_rule.incoming v d := by
            cases hincor with
            | inl h => rw [h]; exact List.mem_cons_of_mem _ (List.mem_singleton_self _)
            | inr h => rw [h]; exact List.mem_cons_self ..
          have hMe : major ∈ d.EDGES := mem_incoming_mem_edges v d hmaj_in
          have hSe : minor ∈ d.EDGES := mem_incoming_mem_edges v d hmin_in
          have hMend : major.END = v := mem_incoming_end_eq v d hmaj_in
          have hSend : minor.END = v := mem_incoming_end_eq v d hmin_in
          have hMs_mem : major.START ∈ d.NODES := (hvalid.hygiene.2.1 hMe).1
          have hSs_mem : minor.START ∈ d.NODES := (hvalid.hygiene.2.1 hSe).1
          have hMout : major ∈ get_rule.outgoing major.START d :=
            mem_outgoing_of_mem_edges_start_eq major.START d hMe rfl
          have hSout : minor ∈ get_rule.outgoing minor.START d :=
            mem_outgoing_of_mem_edges_start_eq minor.START d hSe rfl
          have hMoutne : get_rule.outgoing major.START d ≠ [] := by
            intro h; rw [h] at hMout; exact List.not_mem_nil hMout
          have hSoutne : get_rule.outgoing minor.START d ≠ [] := by
            intro h; rw [h] at hSout; exact List.not_mem_nil hSout
          have hMlvl : major.START.LEVEL = v.LEVEL + 1 := by
            have := hvalid.leveledColored major hMe
            rwa [hMend] at this
          have hSlvl : minor.START.LEVEL = v.LEVEL + 1 := by
            have := hvalid.leveledColored minor hSe
            rwa [hSend] at this
          have hMca : countAbove d major.START < fuel := by
            have := countAbove_lt_of_level_succ d hMs_mem hMlvl
            omega
          have hSca : countAbove d minor.START < fuel := by
            have := countAbove_lt_of_level_succ d hSs_mem hSlvl
            omega
          have hIHM := ih major.START hMs_mem hMoutne hMca
          have hIHS := ih minor.START hSs_mem hSoutne hSca
          have hMdep : major.DEPENDENCY = outDep major.START d :=
            (hvalid.localRuleCorrect major.START hMs_mem hMoutne).2 major hMout
          have hSdep : minor.DEPENDENCY = outDep minor.START d :=
            (hvalid.localRuleCorrect minor.START hSs_mem hSoutne).2 minor hSout
          have hMSne : major.START ≠ minor.START := by
            intro h
            have : minor.START.FORMULA =
                Formula.implication minor.START.FORMULA v.FORMULA := by
              rw [← hMF, h]
            exact formula_ne_implication_self _ _ this
          have hpreds : predsOf d v = [major.START, minor.START] ∨
              predsOf d v = [minor.START, major.START] := by
            cases hincor with
            | inl h =>
                left
                rw [predsOf_eq, h]
                exact eraseDups_pair hMSne
            | inr h =>
                right
                rw [predsOf_eq, h]
                exact eraseDups_pair (fun hh => hMSne hh.symm)
          have hpairs : elimPairsAt d v = [(major.START, minor.START)] :=
            elimPairsAt_pair d v hpreds hMF
          have hebM : edgesBetween d major.START v = [major] := by
            rw [edgesBetween_eq,
              simpleTree_outgoing_eq_singleton d htree hvalid hMout]
            simp [hMend]
          have hebS : edgesBetween d minor.START v = [minor] := by
            rw [edgesBetween_eq,
              simpleTree_outgoing_eq_singleton d htree hvalid hSout]
            simp [hSend]
          have hMc0 : major.COLOUR = 0 := hcol major hMe
          have hSc0 : minor.COLOUR = 0 := hcol minor hSe
          cases hpreds with
          | inl hp =>
              rw [flowAt, hhypf]
              simp only [Bool.false_or, hp, List.isEmpty_cons, if_false,
                Bool.false_eq_true]
              rw [hpairs]
              simp only [List.flatMap_cons, List.flatMap_nil, hIHM, hIHS,
                hebM, hebS, List.filterMap_cons, List.filterMap_nil]
              simp [consume, hMc0, hSc0, reseedPairs, outDep, hclass, hSdep,
                hMdep, depUnion, eraseDups_singleton]
          | inr hp =>
              rw [flowAt, hhypf]
              simp only [Bool.false_or, hp, List.isEmpty_cons, if_false,
                Bool.false_eq_true]
              rw [hpairs]
              simp only [List.flatMap_cons, List.flatMap_nil, hIHM, hIHS,
                hebM, hebS, List.filterMap_cons, List.filterMap_nil]
              simp [consume, hMc0, hSc0, reseedPairs, outDep, hclass, hSdep,
                hMdep, depUnion, eraseDups_singleton]



/--
 On a valid simple tree,
    whose deduction edges carry the default colour 0 by `IsTreeDLDS`, every node satisfies
    the literal Def-23 CorrectRuleApp predicate: tops are excluded, the root
    is vacuous (no target `w` with `v ∈ Pre(w)`), and every interior kernel
    node passes the SINGLETON bullet ; its Flow is one route `(b, ε)` whose
    dependency `b` equals the stored label of its unique outgoing edge.

    A simple tree has no collapsed nodes, so only the singleton
    bullet is exercised here.
-/
theorem flowRuleCorrect_of_simpleTree (d : Graph)
    (htree : IsSimpleTreeDLDS d) (hvalid : ValidDLDS d) :
    ∀ v ∈ d.NODES, FlowRuleCorrect d v := by
  have hcol : ∀ e ∈ d.EDGES, e.COLOUR = 0 := htree.1.2.2.2
  intro v hv
  unfold FlowRuleCorrect flowRuleCorrectAtB
  by_cases hinc : (get_rule.incoming v d).isEmpty
  · rw [if_pos hinc]
  · rw [if_neg hinc]
    by_cases hout : (get_rule.outgoing v d).isEmpty
    · rw [if_pos hout]
    · rw [if_neg hout]
      have houtne : get_rule.outgoing v d ≠ [] := by
        intro h
        rw [h] at hout
        exact hout rfl
      have hca : countAbove d v < stdFuel d := by
        have := countAbove_le d v
        unfold stdFuel
        omega
      rw [flowAt_simpleTree d htree hvalid (stdFuel d) v hv houtne hca]
      show singletonBulletB d v (outDep v d) [] = true
      obtain ⟨e, he⟩ : ∃ e, e ∈ get_rule.outgoing v d := by
        cases h : get_rule.outgoing v d with
        | nil => exact absurd h houtne
        | cons a l => exact ⟨a, List.mem_cons_self ..⟩
      have hsing : get_rule.outgoing v d = [e] :=
        simpleTree_outgoing_eq_singleton d htree hvalid he
      have hce : e.COLOUR = 0 := hcol e (mem_outgoing_mem_edges v d he)
      have hdep : e.DEPENDENCY = outDep v d :=
        (hvalid.localRuleCorrect v hv houtne).2 e he
      unfold singletonBulletB
      rw [hsing]
      simp [headColour, hce, hdep, depSetEq_self]





#print axioms flowRuleCorrect_of_simpleTree
end FlowSpec
