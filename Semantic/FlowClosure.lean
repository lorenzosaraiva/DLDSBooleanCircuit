import Semantic.MultiTokenBridge

open scoped Classical

/-!
# Flow closure for compressed derivations

This module introduces `FlowTailClosure`, proves the corresponding closure
property of `flowAt`, and derives `canonicalSlotOK` for the compressed bridge.

At an elimination kernel with premises `M` and `S`, `FlowTailClosure` requires
the kernel to be binary and every consumed residual contributed by one premise
to have a matching contribution from the other. This is the pairing condition
used by `elimCombine`.

`FlowTailClosure` is an invariant of the horizontal-compression construction;
its preservation by that construction is not formalized here. The `exGap2`
fixture shows that it does not follow from the other local legality conditions.
-/

open Semantic

namespace Semantic

open FlowSpec
open ExFan2 ExFan3



/--
 **FlowTailClosure** (construction invariant #5). At every elim kernel `w`
    with pairing `(M,S)`: (a) `w` is exactly binary, and (b) partner agreement
    both directions ; a consumed residual on one premise's edge is matched by
    the co-premise. See the file header for provenance and the `exGap2`
    underivability certificate.
-/
def FlowTailClosure (d : Graph) : Prop :=
  ∀ w ∈ d.NODES, ∀ M S, elimPairsAt d w = [(M, S)] →
    (∀ u ∈ predsOf d w, u = M ∨ u = S) ∧
    (∀ bp ∈ flowAt d (stdFuel d) S, ∀ eS ∈ edgesBetween d S w, ∀ q,
        consume eS bp.2 = some q →
        ∃ bq ∈ flowAt d (stdFuel d) M, ∃ eM ∈ edgesBetween d M w,
          consume eM bq.2 = some q) ∧
    (∀ bp ∈ flowAt d (stdFuel d) M, ∀ eM ∈ edgesBetween d M w, ∀ q,
        consume eM bp.2 = some q →
        ∃ bq ∈ flowAt d (stdFuel d) S, ∃ eS ∈ edgesBetween d S w,
          consume eS bq.2 = some q)

/--
 Bool mirror: checks the partner-agreement pairing directly (inspects
    `elimPairsAt`/`consume`, not `flowAt w`-membership).
-/
def flowTailClosureAtB (d : Graph) (w : Vertex) : Bool :=
  match elimPairsAt d w with
  | [(M, S)] =>
      (predsOf d w).all (fun u => decide (u = M) || decide (u = S)) &&
      ((flowAt d (stdFuel d) S).all fun bp =>
        (edgesBetween d S w).all fun eS =>
          match consume eS bp.2 with
          | none => true
          | some q =>
              (flowAt d (stdFuel d) M).any fun bq =>
                (edgesBetween d M w).any fun eM => consume eM bq.2 == some q) &&
      ((flowAt d (stdFuel d) M).all fun bp =>
        (edgesBetween d M w).all fun eM =>
          match consume eM bp.2 with
          | none => true
          | some q =>
              (flowAt d (stdFuel d) S).any fun bq =>
                (edgesBetween d S w).any fun eS => consume eS bq.2 == some q)
  | _ => true

def flowTailClosureB (d : Graph) : Bool := d.NODES.all (flowTailClosureAtB d)



def tailClosureAtB (d : Graph) (v : Vertex) : Bool :=
  (flowAt d (stdFuel d) v).all fun bp =>
    (get_rule.outgoing v d).all fun e =>
      if e.COLOUR == headColour bp.2 then
        (flowAt d (stdFuel d) e.END).any fun bq => decide (bq.2 = bp.2.tail)
      else true

def tailClosureB (d : Graph) : Bool := d.NODES.all (tailClosureAtB d)

def tailClosureFailures (d : Graph) : List (Nat × List Nat) :=
  d.NODES.flatMap fun v =>
    (flowAt d (stdFuel d) v).filterMap fun bp =>
      if (get_rule.outgoing v d).any (fun e =>
            (e.COLOUR == headColour bp.2) &&
            !((flowAt d (stdFuel d) e.END).any fun bq =>
                decide (bq.2 = bp.2.tail)))
      then some (v.NUMBER, bp.2) else none



namespace ExGap

def fP : Formula := #"P"
def fQ : Formula := #"Q"
def fR : Formula := #"R"
def fPQ : Formula := fP >> fQ
def fRQ : Formula := fR >> fQ
def fPRQ : Formula := fP >> fRQ
def fZ : Formula := fPQ >> fPQ
def fU4 : Formula := fPQ >> fPRQ

def gv : Vertex := Vertex.node 1 4 fP true false []
def gm : Vertex := Vertex.node 2 4 fPQ true false []
def gw : Vertex := Vertex.node 3 3 fQ false false []
def gu : Vertex := Vertex.node 4 2 fPQ false false []
def gu2 : Vertex := Vertex.node 5 2 fRQ false false []
def gz : Vertex := Vertex.node 6 1 fZ false false []
def gu3 : Vertex := Vertex.node 7 1 fPRQ false false []
def gu4 : Vertex := Vertex.node 8 0 fU4 false false []

/--
 8-node fixture with ONE poisoned ancestral address `[7,8]` at hypothesis
    `P`: colour-strict realizable, head-covered, dependency-faithful, yet
    pairs with no partner at the elim kernel `Q`. Passes the whole legality
    battery identically to `exFan3C`; `flowTailClosureB`/`tailClosureB`/
    `canonicalSlotOKB` are FALSE on it and TRUE on `exFan3C`.
-/
def exGap2 : Graph :=
  Graph.dlds
    [gv, gm, gw, gu, gu2, gz, gu3, gu4]
    [Deduction.edge gv gw 5 [fP],
     Deduction.edge gv gw 6 [fP],
     Deduction.edge gv gw 7 [fP],
     Deduction.edge gm gw 3 [fPQ],
     Deduction.edge gm gw 4 [fPQ],
     Deduction.edge gw gu 9 [fP, fPQ],
     Deduction.edge gw gu2 8 [fP, fPQ],
     Deduction.edge gu gz 0 [fPQ],
     Deduction.edge gu2 gu3 13 [fP, fPQ],
     Deduction.edge gu3 gu4 0 [fPQ]]
    [Ancestral.path gu gv [5, 9],
     Ancestral.path gu3 gv [6, 8, 13],
     Ancestral.path gu2 gv [7, 8],        -- THE POISONED ADDRESS
     Ancestral.path gu gm [3, 9],
     Ancestral.path gu3 gm [4, 8, 13]]

end ExGap

open ExGap

def nodeElimShapeB (d : Graph) (v : Vertex) : Bool :=
  !v.HYPOTHESIS && !(predsOf d v).isEmpty &&
    (match elimPairsAt d v with | [_] => true | _ => false)

def nodeIntroShapeB (d : Graph) (v : Vertex) : Bool :=
  !v.HYPOTHESIS &&
    (match get_rule.incoming v d, v.FORMULA with
     | [p], Formula.implication _ β =>
         decide (p.START.FORMULA = β) && (elimPairsAt d v).isEmpty
     | _, _ => false)

def legalityBattery (d : Graph) : List (String × Bool) :=
  [("oneEdgePerColourPerNode",
     d.EDGES.all fun e1 => d.EDGES.all fun e2 =>
       if decide (e1.START = e2.START) && (e1.COLOUR == e2.COLOUR)
       then decide (e1 = e2) else true),
   ("routeFanUnique", d.NODES.all (routeFanUniqueAtB d)),
   ("faithfulDecoration", d.NODES.all (faithfulNewAtB d)),
   ("routeHeadCoverage(guarded)",
     d.NODES.all fun v =>
       (get_rule.outgoing v d).isEmpty || routeHeadCoverageAtB d v),
   ("levelFormulaUnique", levelFormulaUniqueB d),
   ("reseedFree", reseedFreeB d),
   ("flowCorrect(Def23-CorrectRuleApp+roots)", flowCorrectB d),
   ("routeEdgeDepChar", d.NODES.all (routeEdgeDepCharAtB d)),
   ("hygiene(check_dlds)",
     (d.NODES.all fun n1 => d.NODES.all fun n2 =>
        if n1.NUMBER == n2.NUMBER then decide (n1 = n2) else true) &&
     (d.EDGES.all fun e =>
        decide (e.START ∈ d.NODES) && decide (e.END ∈ d.NODES)) &&
     (d.PATHS.all fun p =>
        decide (p.START ∈ d.NODES) && decide (p.END ∈ d.NODES))),
   ("leveledColored", d.EDGES.all fun e => e.START.LEVEL == e.END.LEVEL + 1),
   ("simplicity",
     d.EDGES.all fun e1 => d.EDGES.all fun e2 =>
       if decide (e1.START = e2.START) && decide (e1.END = e2.END) &&
          (e1.COLOUR == e2.COLOUR)
       then decide (e1 = e2) else true),
   ("ancestorSimplicity",
     d.PATHS.all fun p1 => d.PATHS.all fun p2 =>
       if decide (p1.START = p2.START) && decide (p1.END = p2.END) &&
          decide (p1.COLOURS = p2.COLOURS)
       then decide (p1 = p2) else true),
   ("hypNoIncoming",
     d.NODES.all fun v => !v.HYPOTHESIS || (get_rule.incoming v d).isEmpty),
   ("localRuleCorrect(classifyRule?)",
     d.NODES.all fun v =>
       (get_rule.outgoing v d).isEmpty ||
         (ruleShapeOKB v d &&
          (get_rule.outgoing v d).all fun e =>
            decide (e.DEPENDENCY = outDep v d))),
   ("rootDischarged(Def23)",
     d.NODES.all fun v =>
       !(get_rule.outgoing v d).isEmpty ||
         (ruleShapeOKB v d && decide (outDep v d = []))),
   ("colorAcyclicity", colorAcyclicityB d),
   ("ancestorEdges", ancestorEdgesB d),
   ("ancestorBackwayInformation(Def19)", ancestorBackwayInformationB d),
   ("nonNestedAncestorEdges", nonNestedAncestorEdgesB d),
   ("structuralShape(elimPairs/intro)",
     d.NODES.all fun v =>
       (get_rule.incoming v d).isEmpty || (get_rule.outgoing v d).isEmpty ||
         (nodeElimShapeB d v || nodeIntroShapeB d v))]



/--
 `consume` on a head-colour-matched edge yields the tail, uniformly
    (nonempty residual: drop the head; empty residual with colour-0 edge:
    stays empty = `[].tail`).
-/
lemma consume_headColour {e : Deduction} {p : ColourPath}
    (hcol : e.COLOUR = headColour p) : consume e p = some p.tail := by
  unfold headColour at hcol
  unfold consume
  cases p with
  | nil => simp only [List.headD_nil] at hcol; simp [hcol]
  | cons o p' => simp only [List.headD_cons] at hcol; simp [hcol]

lemma countAbove_lt_length (d : Graph) {v : Vertex} (hv : v ∈ d.NODES) :
    countAbove d v < d.NODES.length := by
  have h := filter_length_lt_of_mem d.NODES
    (p := fun u => decide (v.LEVEL < u.LEVEL)) (q := fun _ => true)
    (fun _ _ _ => rfl) v hv rfl (by simp)
  unfold countAbove
  simpa using h

/--  Top-node flow is fuel-independent (only the ancestor addresses).  -/
lemma flowAt_top_eq (d : Graph) (fuel : Nat) {v : Vertex}
    (hguard : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true) :
    flowAt d (fuel + 1) v =
      (let anc := ancestorsInto d v
       if anc.isEmpty then [([v.FORMULA], ([] : ColourPath))]
       else anc.map (fun a => ([v.FORMULA], a.COLOURS))) := by
  rw [flowAt]
  simp only [hguard, if_true]

lemma predsOf_has_edge {d : Graph} {v u : Vertex} (h : u ∈ predsOf d v) :
    ∃ e ∈ get_rule.incoming v d, e.START = u := by
  rw [predsOf_eq, List.mem_eraseDups, List.mem_map] at h
  obtain ⟨e, he, hst⟩ := h
  exact ⟨e, he, hst⟩

lemma mem_predsOf_mem_nodes {d : Graph} (hstruct : structuralValid d)
    {v u : Vertex} (h : u ∈ predsOf d v) : u ∈ d.NODES := by
  obtain ⟨e, he, hst⟩ := predsOf_has_edge h
  have heE : e ∈ d.EDGES := mem_incoming_mem_edges v d he
  rw [← hst]; exact (hstruct.2.1 e heE).1

/--  A predecessor is one level above and has an outgoing edge (into `v`).  -/
lemma predsOf_props {d : Graph} (hstruct : structuralValid d)
    {v u : Vertex} (h : u ∈ predsOf d v) :
    u.LEVEL = v.LEVEL + 1 ∧ get_rule.outgoing u d ≠ [] := by
  obtain ⟨e, he, hst⟩ := predsOf_has_edge h
  have heE : e ∈ d.EDGES := mem_incoming_mem_edges v d he
  have hend : e.END = v := mem_incoming_end_eq v d he
  have hlvl := hstruct.1 e heE
  have hout : e ∈ get_rule.outgoing u d := by
    rw [← hst]; exact incoming_mem_outgoing_start v d he
  refine ⟨by rw [← hst, ← hend]; exact hlvl, ?_⟩
  intro hnil; rw [hnil] at hout; exact List.not_mem_nil hout

lemma mem_preds_of_elimPairsAt {d : Graph} {v M S : Vertex}
    (h : elimPairsAt d v = [(M, S)]) : M ∈ predsOf d v ∧ S ∈ predsOf d v := by
  have hmem : (M, S) ∈ elimPairsAt d v := by rw [h]; exact List.mem_singleton_self _
  unfold elimPairsAt at hmem
  rw [List.mem_flatMap] at hmem
  obtain ⟨major, hmaj, hmem⟩ := hmem
  rw [List.mem_filterMap] at hmem
  obtain ⟨minor, hmin, hcond⟩ := hmem
  by_cases hc : major.FORMULA = Formula.implication minor.FORMULA v.FORMULA
  · rw [if_pos hc] at hcond
    rw [Option.some_inj, Prod.mk.injEq] at hcond
    obtain ⟨hM, hS⟩ := hcond
    exact ⟨hM ▸ hmaj, hS ▸ hmin⟩
  · rw [if_neg hc] at hcond; exact absurd hcond (by simp)

/--
 **Fuel stability**: at a node with an outgoing edge, one extra unit of fuel
    (beyond `countAbove`) does not change `flowAt`. Strong induction on
    `countAbove`; premises stay above and keep an outgoing edge.
-/
lemma flowAt_succ_stable (d : Graph) (hstruct : structuralValid d) :
    ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] → ∀ f, countAbove d v < f →
      flowAt d (f + 1) v = flowAt d f v := by
  have key : ∀ n, ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] →
      countAbove d v = n → ∀ f, countAbove d v < f →
      flowAt d (f + 1) v = flowAt d f v := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro v hv hout hn f hf
      have hf1 : f = (f - 1) + 1 := by omega
      by_cases hguard : (v.HYPOTHESIS || (predsOf d v).isEmpty) = true
      · rw [flowAt_top_eq d f hguard]
        conv_rhs => rw [hf1, flowAt_top_eq d (f - 1) hguard]
      · have hguard' : (v.HYPOTHESIS || (predsOf d v).isEmpty) = false := by
          cases h : (v.HYPOTHESIS || (predsOf d v).isEmpty) with
          | true => exact absurd h hguard
          | false => rfl
        have hhyp : v.HYPOTHESIS = false := by
          by_contra hh
          have : v.HYPOTHESIS = true := by
            cases hb : v.HYPOTHESIS with
            | true => rfl
            | false => exact absurd hb hh
          rw [this] at hguard'; simp at hguard'
        have hpredsne : (predsOf d v).isEmpty = false := by
          cases hp : (predsOf d v).isEmpty with
          | false => rfl
          | true => rw [hhyp, hp] at hguard'; simp at hguard'
        have hinc : (get_rule.incoming v d) ≠ [] := by
          intro h
          have : (predsOf d v).isEmpty = true := by
            rw [predsOf_eq, h]; rfl
          rw [this] at hpredsne; simp at hpredsne
        have prem : ∀ u ∈ predsOf d v, flowAt d f u = flowAt d (f - 1) u := by
          intro u hu
          have humem : u ∈ d.NODES := mem_predsOf_mem_nodes hstruct hu
          obtain ⟨hulvl, huout⟩ := predsOf_props hstruct hu
          have huca : countAbove d u < countAbove d v :=
            countAbove_lt_of_level_succ d humem hulvl
          have := ih (countAbove d u) (by rw [← hn]; exact huca) u humem huout rfl
            (f - 1) (by omega)
          rw [hf1]; exact this
        rcases hstruct.2.2 v hv hinc hout with
          ⟨M, S, _, _, hpairs⟩ | ⟨p, α, β, _, hincoming, hform, hpβ, helim⟩
        · obtain ⟨hMp, hSp⟩ := mem_preds_of_elimPairsAt hpairs
          rw [flowAt_kernel_elim_eq d f hhyp hpredsne hpairs]
          conv_rhs => rw [hf1, flowAt_kernel_elim_eq d (f - 1) hhyp hpredsne hpairs]
          rw [prem S hSp, prem M hMp]
        · have hpreds : predsOf d v = [p.START] := by rw [predsOf_eq, hincoming]; rfl
          have hSp : p.START ∈ predsOf d v := by rw [hpreds]; exact List.mem_singleton_self _
          rw [flowAt_kernel_intro_eq d f hhyp hpreds hform hpβ helim]
          conv_rhs => rw [hf1, flowAt_kernel_intro_eq d (f - 1) hhyp hpreds hform hpβ helim]
          rw [prem p.START hSp]
  intro v hv hout f hf
  exact key (countAbove d v) v hv hout rfl f hf

/--
 `flowAt` at `stdFuel` equals `flowAt` at `stdFuel - 1` (feeds the
    one-level `elimCombine` unfolding).
-/
lemma flowAt_stdFuel_pred (d : Graph) (hstruct : structuralValid d)
    {v : Vertex} (hv : v ∈ d.NODES) (hout : get_rule.outgoing v d ≠ []) :
    flowAt d (stdFuel d) v = flowAt d (stdFuel d - 1) v := by
  have hlt : countAbove d v < stdFuel d - 1 := by
    have := countAbove_lt_length d hv; unfold stdFuel; omega
  have hstep := flowAt_succ_stable d hstruct v hv hout (stdFuel d - 1) hlt
  have hstd : stdFuel d - 1 + 1 = stdFuel d := by unfold stdFuel; omega
  rw [hstd] at hstep
  exact hstep



lemma mem_edgesBetween_of {d : Graph} {u w : Vertex} {e : Deduction}
    (he : e ∈ d.EDGES) (hs : e.START = u) (hw : e.END = w) :
    e ∈ edgesBetween d u w := by
  unfold edgesBetween
  rw [List.mem_filter]
  exact ⟨he, by simp [hs, hw]⟩

/--
 If a minor route and a major route consume to the SAME residual `q` over
    their edges, then `(depUnion b₁ b₂, q)` is in `elimCombine`.
-/
lemma mem_elimCombine_of {d : Graph} {w M S : Vertex}
    {FS FM : List FlowPair} {bp₁ bp₂ : FlowPair} {e₁ e₂ : Deduction} {q : ColourPath}
    (hbp₁ : bp₁ ∈ FS) (hbp₂ : bp₂ ∈ FM)
    (he₁ : e₁ ∈ edgesBetween d S w) (he₂ : e₂ ∈ edgesBetween d M w)
    (hc₁ : consume e₁ bp₁.2 = some q) (hc₂ : consume e₂ bp₂.2 = some q) :
    (depUnion bp₁.1 bp₂.1, q) ∈ elimCombine d w M S FS FM := by
  unfold elimCombine
  rw [List.mem_flatMap]
  refine ⟨bp₁, hbp₁, ?_⟩
  rw [List.mem_flatMap]
  refine ⟨bp₂, hbp₂, ?_⟩
  rw [List.mem_flatMap]
  refine ⟨e₁, he₁, ?_⟩
  rw [List.mem_filterMap]
  refine ⟨e₂, he₂, ?_⟩
  rw [hc₁, hc₂]
  simp

/--
 On a legal graph with `FlowTailClosure`,
    a route `(b,p)` at `v` crossing the outgoing edge `e` whose colour is the
    residual head continues, with SOME dependency, as residual `p.tail` at the
    target `e.END` (which is assumed to have an outgoing edge ; the non-root
    targets the canonical walk actually queries). The elim case is exactly the
    partner agreement; intro/top are direct.
-/
theorem flowAt_tail_closure (d : Graph)
    (hstruct : structuralValid d)
    (hhypNoInc : HypothesesHaveNoIncoming d)
    (hftc : FlowTailClosure d)
    {v : Vertex} (hv : v ∈ d.NODES)
    {b : Dep} {p : ColourPath} (hbp : (b, p) ∈ flowAt d (stdFuel d) v)
    {e : Deduction} (he : e ∈ get_rule.outgoing v d)
    (hcol : e.COLOUR = headColour p)
    (hwout : get_rule.outgoing e.END d ≠ []) :
    ∃ b', (b', p.tail) ∈ flowAt d (stdFuel d) e.END := by
  set w := e.END with hwdef
  have heE : e ∈ d.EDGES := mem_outgoing_mem_edges v d he
  have hestart : e.START = v := mem_outgoing_start_eq v d he
  have hwmem : w ∈ d.NODES := (hstruct.2.1 e heE).2
  have heS : e ∈ edgesBetween d v w := mem_edgesBetween_of heE hestart rfl
  have hwinc : e ∈ get_rule.incoming w d := outgoing_mem_incoming_end v d he
  have hvpred : v ∈ predsOf d w := by
    rw [predsOf_eq, List.mem_eraseDups, List.mem_map]
    exact ⟨e, hwinc, hestart⟩
  have hwincne : get_rule.incoming w d ≠ [] := by
    intro h; rw [h] at hwinc; exact List.not_mem_nil hwinc
  have hwhyp : w.HYPOTHESIS = false := by
    by_contra hh
    have : w.HYPOTHESIS = true := by
      cases hb : w.HYPOTHESIS with | true => rfl | false => exact absurd hb hh
    exact hwincne (hhypNoInc w hwmem this)
  have hwpredsne : (predsOf d w).isEmpty = false := by
    cases hp : (predsOf d w).isEmpty with
    | false => rfl
    | true => rw [List.isEmpty_iff] at hp; rw [hp] at hvpred; exact absurd hvpred (List.not_mem_nil)
  have hconsume : consume e p = some p.tail := consume_headColour hcol
  have hstd1 : stdFuel d = (stdFuel d - 1) + 1 := by unfold stdFuel; omega
  rcases hstruct.2.2 w hwmem hwincne hwout with
    ⟨M, S, _, _, hpairs⟩ | ⟨pe, α, β, _, hwincoming, hwform, hpβ, helim⟩
  ·
    obtain ⟨hbinary, hagreeS, hagreeM⟩ := hftc w hwmem M S hpairs
    obtain ⟨hMp, hSp⟩ := mem_preds_of_elimPairsAt hpairs
    have hMnode : M ∈ d.NODES := mem_predsOf_mem_nodes hstruct hMp
    have hSnode : S ∈ d.NODES := mem_predsOf_mem_nodes hstruct hSp
    have hMout : get_rule.outgoing M d ≠ [] := (predsOf_props hstruct hMp).2
    have hSout : get_rule.outgoing S d ≠ [] := (predsOf_props hstruct hSp).2
    have hwflow : flowAt d (stdFuel d) w =
        (elimCombine d w M S (flowAt d (stdFuel d - 1) S)
          (flowAt d (stdFuel d - 1) M)).eraseDups := by
      conv_lhs => rw [hstd1]
      rw [flowAt_kernel_elim_eq d (stdFuel d - 1) hwhyp hwpredsne hpairs]
    rcases hbinary v hvpred with hvM | hvS
    · -- v = M: major side supplies the route, minor supplies the partner
      have hbpM : (b, p) ∈ flowAt d (stdFuel d) M := by rw [← hvM]; exact hbp
      have heMw : e ∈ edgesBetween d M w := by rw [← hvM]; exact heS
      obtain ⟨bq, hbqS, eS, heSw, hcS⟩ :=
        hagreeM (b, p) hbpM e heMw p.tail hconsume
      have hbp' : (b, p) ∈ flowAt d (stdFuel d - 1) M := by
        rw [← flowAt_stdFuel_pred d hstruct hMnode hMout]; exact hbpM
      have hbq' : bq ∈ flowAt d (stdFuel d - 1) S := by
        rw [← flowAt_stdFuel_pred d hstruct hSnode hSout]; exact hbqS
      refine ⟨depUnion bq.1 (b, p).1, ?_⟩
      rw [hwflow, List.mem_eraseDups]
      exact mem_elimCombine_of hbq' hbp' heSw heMw hcS hconsume
    · -- v = S: minor side supplies the route, major supplies the partner
      have hbpS : (b, p) ∈ flowAt d (stdFuel d) S := by rw [← hvS]; exact hbp
      have heSw : e ∈ edgesBetween d S w := by rw [← hvS]; exact heS
      obtain ⟨bq, hbqM, eM, heMw, hcM⟩ :=
        hagreeS (b, p) hbpS e heSw p.tail hconsume
      have hbp' : (b, p) ∈ flowAt d (stdFuel d - 1) S := by
        rw [← flowAt_stdFuel_pred d hstruct hSnode hSout]; exact hbpS
      have hbq' : bq ∈ flowAt d (stdFuel d - 1) M := by
        rw [← flowAt_stdFuel_pred d hstruct hMnode hMout]; exact hbqM
      refine ⟨depUnion (b, p).1 bq.1, ?_⟩
      rw [hwflow, List.mem_eraseDups]
      exact mem_elimCombine_of hbp' hbq' heSw heMw hconsume hcM
  ·
    have hpreds : predsOf d w = [pe.START] := by rw [predsOf_eq, hwincoming]; rfl
    have hvpe : v = pe.START := by rw [hpreds] at hvpred; simpa using hvpred
    have hunode : pe.START ∈ d.NODES := by rw [← hvpe]; exact hv
    have huout : get_rule.outgoing pe.START d ≠ [] := by
      rw [← hvpe]; intro h; rw [h] at he; exact List.not_mem_nil he
    have hwflow : flowAt d (stdFuel d) w =
        ((flowAt d (stdFuel d - 1) pe.START).flatMap (fun bp =>
          (edgesBetween d pe.START w).filterMap (fun ed =>
            (consume ed bp.2).map (fun q => (depRemove bp.1 α, q))))).eraseDups := by
      conv_lhs => rw [hstd1]
      rw [flowAt_kernel_intro_eq d (stdFuel d - 1) hwhyp hpreds hwform hpβ helim]
    have hbp' : (b, p) ∈ flowAt d (stdFuel d - 1) pe.START := by
      rw [← flowAt_stdFuel_pred d hstruct hunode huout, ← hvpe]; exact hbp
    have heS' : e ∈ edgesBetween d pe.START w := by rw [← hvpe]; exact heS
    refine ⟨depRemove (b, p).1 α, ?_⟩
    rw [hwflow, List.mem_eraseDups, List.mem_flatMap]
    refine ⟨(b, p), hbp', ?_⟩
    rw [List.mem_filterMap]
    refine ⟨e, heS', ?_⟩
    rw [hconsume]; simp




/--  `nodeAtLevelFormula?` returns a node at its own (level, formula) key.  -/
lemma nodeAtLevelFormula?_self (d : Graph) (hlfu : LevelFormulaUnique d)
    {v : Vertex} (hv : v ∈ d.NODES) :
    nodeAtLevelFormula? d v.LEVEL v.FORMULA = some v := by
  unfold nodeAtLevelFormula?
  cases h : d.NODES.find?
      (fun u => decide (u.LEVEL = v.LEVEL ∧ u.FORMULA = v.FORMULA)) with
  | none =>
      exfalso
      rw [List.find?_eq_none] at h
      exact h v hv (by simp)
  | some u =>
      have humem : u ∈ d.NODES := find?_some_mem h
      have hu : u.LEVEL = v.LEVEL ∧ u.FORMULA = v.FORMULA := by
        have hh := List.find?_some h; simpa using hh
      rw [hlfu u humem v hv hu.1 hu.2]

lemma edgeOfColour_some {d : Graph} {v : Vertex} {c : Nat} {e : Deduction}
    (h : edgeOfColour d v c = some e) :
    e ∈ d.EDGES ∧ e.START = v ∧ e.COLOUR = c := by
  unfold edgeOfColour at h
  cases hf : (d.EDGES.filter (fun e => e.START = v ∧ e.COLOUR = c)) with
  | nil => rw [hf] at h; simp at h
  | cons x xs =>
      rw [hf, List.head?_cons, Option.some.injEq] at h
      subst h
      have hx : x ∈ d.EDGES.filter (fun e => e.START = v ∧ e.COLOUR = c) := by
        rw [hf]; exact List.mem_cons_self ..
      rw [List.mem_filter] at hx
      exact ⟨hx.1, (of_decide_eq_true hx.2).1, (of_decide_eq_true hx.2).2⟩

/--  A covered colour has an outgoing edge of that colour.  -/
lemma edgeOfColour_some_of_cover {d : Graph} {v : Vertex} {c : Nat}
    (h : ∃ e ∈ get_rule.outgoing v d, e.COLOUR = c) :
    ∃ e, edgeOfColour d v c = some e := by
  obtain ⟨e, hout, hcol⟩ := h
  have heE : e ∈ d.EDGES := mem_outgoing_mem_edges v d hout
  have hstart : e.START = v := mem_outgoing_start_eq v d hout
  have hmem : e ∈ d.EDGES.filter (fun e => e.START = v ∧ e.COLOUR = c) := by
    rw [List.mem_filter]; exact ⟨heE, by simp [hstart, hcol]⟩
  unfold edgeOfColour
  cases hf : d.EDGES.filter (fun e => e.START = v ∧ e.COLOUR = c) with
  | nil => rw [hf] at hmem; exact absurd hmem List.not_mem_nil
  | cons x xs => exact ⟨x, rfl⟩

lemma routeFromFlowC_stuck (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} (p : ColourPath)
    (hstuck : nodeAtLevelFormula? d lvl φ = none ∨
      ∃ v, nodeAtLevelFormula? d lvl φ = some v ∧ get_rule.outgoing v d = []) :
    ∀ fuel, routeFromFlowC d formulas fuel lvl φ p = List.replicate fuel (0, 0) := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ f ih =>
      rw [List.replicate_succ, routeFromFlowC]
      rcases hstuck with hnone | ⟨v, hsome, hterm⟩
      · rw [hnone]; simp only; rw [ih]
      · rw [hsome]; simp only [hterm]; rw [ih]

lemma admChainCB_stuck (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} (p : ColourPath)
    (hstuck : nodeAtLevelFormula? d lvl φ = none ∨
      ∃ v, nodeAtLevelFormula? d lvl φ = some v ∧ get_rule.outgoing v d = []) :
    ∀ fuel, admChainCB d formulas lvl φ p
      (routeFromFlowC d formulas fuel lvl φ p) = true := by
  intro fuel
  rw [routeFromFlowC_stuck d formulas p hstuck fuel]
  cases fuel with
  | zero =>
      rcases hstuck with hnone | ⟨v, hsome, hterm⟩
      · simp only [List.replicate, admChainCB, hnone]
      · simp only [List.replicate, admChainCB, hsome, hterm, List.isEmpty_nil]
  | succ f =>
      rw [List.replicate_succ]
      rcases hstuck with hnone | ⟨v, hsome, hterm⟩
      · simp only [admChainCB, hnone]; simp [replicate_all_stops]
      · simp only [admChainCB, hsome, hterm]; simp [replicate_all_stops]

/--  One active step of `routeFromFlowC`.  -/
lemma routeFromFlowC_step (d : Graph) (formulas : List Formula)
    {fuel lvl : Nat} {φ : Formula} {p : ColourPath} {v : Vertex}
    {e0 : Deduction} {es : List Deduction} {e : Deduction}
    (hnode : nodeAtLevelFormula? d lvl φ = some v)
    (hout : get_rule.outgoing v d = e0 :: es)
    (hedge : edgeOfColour d v (headColour p) = some e) :
    routeFromFlowC d formulas (fuel + 1) lvl φ p =
      (formulas.idxOf e.END.FORMULA + 1, inputLabelForEdgeC d formulas φ e.END) ::
      (if isMinorArrivalC d φ e.END && p.tail.isEmpty then List.replicate fuel (0, 0)
       else routeFromFlowC d formulas fuel (lvl - 1) e.END.FORMULA p.tail) := by
  rw [routeFromFlowC]; simp only [hnode, hout, hedge]

/--  One active step of `admChainCB` (mirror of `admChainB_cons_reduce`).  -/
lemma admChainCB_step (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} {p : ColourPath} {t l : Nat}
    {rest : List (Nat × Nat)} {v : Vertex} {e0 : Deduction}
    {es : List Deduction} {e : Deduction}
    (hnode : nodeAtLevelFormula? d lvl φ = some v)
    (hout : get_rule.outgoing v d = e0 :: es)
    (hedge : edgeOfColour d v (headColour p) = some e) :
    admChainCB d formulas lvl φ p ((t, l) :: rest) =
      ((t == formulas.idxOf e.END.FORMULA + 1) &&
       (l == inputLabelForEdgeC d formulas φ e.END) &&
       (if isMinorArrivalC d φ e.END && p.tail.isEmpty then rest.all (fun s => s == (0, 0))
        else admChainCB d formulas (lvl - 1) e.END.FORMULA p.tail rest)) := by
  conv_lhs => rw [admChainCB]
  simp only [hnode, hout, hedge]

/--
 **routeFromFlowC ⊨ admChainCB** ; the canonical route-following passes the
    admissibility chain check, by induction along the route. The step where
    `admChainCB` could fail (`edgeOfColour = none`) is ruled out by
    `RouteHeadCoverage`; the residual continues at the next node by
    `flowAt_tail_closure`. `lvl ≤ fuel` supplies just enough fuel (the walk
    descends one level per step and terminates at level 0 / a rootless node).
-/
lemma routeFromFlowC_admChainCB (d : Graph)
    (hstruct : structuralValid d) (hhypNoInc : HypothesesHaveNoIncoming d)
    (hlfu : LevelFormulaUnique d) (hcoverage : RouteHeadCoverage d)
    (hftc : FlowTailClosure d) (formulas : List Formula) :
    ∀ (fuel lvl : Nat) (φ : Formula) (v : Vertex) (b : Dep) (p : ColourPath),
      nodeAtLevelFormula? d lvl φ = some v → v ∈ d.NODES →
      (b, p) ∈ flowAt d (stdFuel d) v → lvl ≤ fuel →
      admChainCB d formulas lvl φ p
        (routeFromFlowC d formulas fuel lvl φ p) = true := by
  intro fuel
  induction fuel with
  | zero =>
      intro lvl φ v b p hnode hv hbp hle
      have hlvl0 : lvl = 0 := Nat.le_zero.mp hle
      have hvlvl : v.LEVEL = lvl := (nodeAtLevelFormula?_some hnode).2.1
      have hterm : get_rule.outgoing v d = [] :=
        level_zero_terminal d hstruct.1 (by rw [hvlvl, hlvl0])
      show admChainCB d formulas lvl φ p [] = true
      rw [admChainCB, hnode]; simp [hterm]
  | succ fuel ih =>
      intro lvl φ v b p hnode hv hbp hle
      have hvlvl : v.LEVEL = lvl := (nodeAtLevelFormula?_some hnode).2.1
      by_cases hout : get_rule.outgoing v d = []
      · exact admChainCB_stuck d formulas p (Or.inr ⟨v, hnode, hout⟩) (fuel + 1)
      ·
        have hcov := (hcoverage v hv hout).1 (b, p) hbp
        obtain ⟨e, hedge⟩ := edgeOfColour_some_of_cover hcov
        obtain ⟨heE, hestart, hecol⟩ := edgeOfColour_some hedge
        obtain ⟨e0, es, hcons⟩ : ∃ e0 es, get_rule.outgoing v d = e0 :: es := by
          cases h : get_rule.outgoing v d with
          | nil => exact absurd h hout
          | cons a l => exact ⟨a, l, rfl⟩
        have hEmem : e.END ∈ d.NODES := (hstruct.2.1 e heE).2
        have hElvl : e.END.LEVEL = lvl - 1 := by
          have := hstruct.1 e heE; rw [hestart, hvlvl] at this; omega
        have hEnode : nodeAtLevelFormula? d (lvl - 1) e.END.FORMULA = some e.END := by
          have := nodeAtLevelFormula?_self d hlfu hEmem; rwa [hElvl] at this
        have hein : e ∈ get_rule.outgoing v d :=
          mem_outgoing_of_mem_edges_start_eq v d heE hestart
        by_cases hmin : (isMinorArrivalC d φ e.END && p.tail.isEmpty) = true
        · rw [routeFromFlowC_step d formulas hnode hcons hedge, if_pos hmin,
              admChainCB_step d formulas hnode hcons hedge]
          simp only [beq_self_eq_true, Bool.and_self, Bool.true_and, hmin, if_true]
          exact replicate_all_stops fuel
        · rw [routeFromFlowC_step d formulas hnode hcons hedge, if_neg hmin,
              admChainCB_step d formulas hnode hcons hedge, if_neg hmin]
          simp only [beq_self_eq_true, Bool.and_self, Bool.true_and]
          by_cases hEout : get_rule.outgoing e.END d = []
          · exact admChainCB_stuck d formulas p.tail (Or.inr ⟨e.END, hEnode, hEout⟩) fuel
          · obtain ⟨b', hb'⟩ := flowAt_tail_closure d hstruct hhypNoInc hftc hv hbp
              hein (by simpa using hecol) hEout
            exact ih (lvl - 1) e.END.FORMULA e.END b' p.tail hEnode hEmem hb' (by omega)

/--
 Padding lemma for the compressed hypothesis column (mirror of
    `admHypColumnB_paddings`).
-/
lemma admHypColumnCB_paddings (d : Graph) (formulas : List Formula)
    (col lvl : Nat) (φ : Formula) (p : ColourPath) :
    ∀ (delay : Nat) (chain : List (Nat × Nat)),
      admHypColumnCB d formulas col lvl φ p delay
        (List.replicate delay (col + 1, 0) ++ chain) =
      admChainCB d formulas lvl φ p chain := by
  intro delay
  induction delay with
  | zero => intro chain; rfl
  | succ k ih =>
      intro chain
      rw [List.replicate_succ, List.cons_append, admHypColumnCB]
      simp only [beq_self_eq_true, Bool.and_self, Bool.true_and]
      exact ih chain

/--
 Under the legality conditions and `FlowTailClosure`,
    the canonical multi-path passes every slot check ; the M2c side condition,
    now a theorem.
-/
theorem canonicalSlotOK (d : Graph)
    (hstruct : structuralValid d) (hhypNoInc : HypothesesHaveNoIncoming d)
    (hlfu : LevelFormulaUnique d) (hcoverage : RouteHeadCoverage d)
    (hftc : FlowTailClosure d) :
    canonicalSlotOKB d = true := by
  unfold canonicalSlotOKB
  rw [List.all_eq_true]
  intro j hj
  have hjr : j < routeCount d := List.mem_range.mp hj
  set formulas := buildFormulas d with hf
  have hlen : (multiPathsFromFlow d).length = routeCount d := multiPathsFromFlow_length d
  have hjm : j < (multiPathsFromFlow d).length := by rw [hlen]; exact hjr
  have hjro : j < (routesOf d).length := hjr
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjm, Option.getD_some]
  show slotCheckB d j ((multiPathsFromFlow d)[j]) = true
  unfold multiPathsFromFlow slotCheckB
  simp only [List.getElem_map]
  have hgd : (routesOf d).getD j (0, 0) = (routesOf d)[j] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjro, Option.getD_some]
  rw [hgd]
  set cr := (routesOf d)[j] with hcr
  obtain ⟨col, r⟩ := cr
  simp only
  cases hfc : formulas[col]? with
  | none => exact replicate_all_stops _
  | some φ =>
      simp only [hfc]
      cases hfind : d.NODES.find?
          (fun v => v.HYPOTHESIS && decide (v.FORMULA = φ)) with
      | none => simp only [hfind]; exact replicate_all_stops _
      | some v =>
          simp only [hfind]
          have hvmem : v ∈ d.NODES := find?_some_mem hfind
          have hvform : v.FORMULA = φ := by
            have hh := List.find?_some hfind
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hh
            exact hh.2
          cases hbp : (flowAt d (stdFuel d) v)[r]? with
          | none => simp only [hbp]; exact replicate_all_stops _
          | some bp =>
              simp only [hbp]
              set maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0 with hml
              set numSteps := (buildGridFromDLDS d).length - 1 with hns
              have hvle : v.LEVEL ≤ maxLvl := level_le_maxLvl d hvmem
              have hnsml : numSteps = maxLvl := numSteps_eq_maxLvl d
              have hfuel : numSteps - (maxLvl - v.LEVEL) = v.LEVEL := by omega
              rw [admHypColumnCB_paddings]
              rw [hfuel]
              have hbpmem : bp ∈ flowAt d (stdFuel d) v := by
                have hr : r < (flowAt d (stdFuel d) v).length := by
                  by_contra hcon
                  rw [List.getElem?_eq_none (Nat.le_of_not_lt hcon)] at hbp
                  simp at hbp
                rw [List.getElem?_eq_getElem hr, Option.some.injEq] at hbp
                rw [← hbp]; exact List.getElem_mem hr
              have hnode : nodeAtLevelFormula? d v.LEVEL φ = some v := by
                rw [← hvform]; exact nodeAtLevelFormula?_self d hlfu hvmem
              exact routeFromFlowC_admChainCB d hstruct hhypNoInc hlfu hcoverage
                hftc formulas v.LEVEL v.LEVEL φ v bp.1 bp.2 hnode hvmem
                (by simpa using hbpmem) (le_refl _)



/--
 The compressed universal bridge with `canonicalSlotOKB` derived from
    `FlowTailClosure` and the legality conditions. The discharge certificate
    `dischargedMultiB` remains an explicit hypothesis.
-/
theorem compressed_universally_accepted' (d : Graph)
    (hstruct : structuralValid d)
    (hnodup : d.EDGES.Nodup)
    (honepercol : OneEdgePerColourPerNode d)
    (hroutefan : RouteFanUnique d)
    (hfaithful : FaithfulDecoration d)
    (hcoverage : RouteHeadCoverage d)
    (hlfu : LevelFormulaUnique d)
    (hhypNoInc : HypothesesHaveNoIncoming d)
    (hreseed : ReseedFree d)
    (hftc : FlowTailClosure d)
    (hdis : dischargedMultiB d = true) :
    ∀ P, ¬ AdmissibleMultiPath d P ∨ DischargedMulti d P (goalColumn d) := by
  intro P
  by_cases hadm : AdmissibleMultiPath d P
  · right
    have hslot : canonicalSlotOKB d = true :=
      canonicalSlotOK d hstruct hhypNoInc hlfu hcoverage hftc
    have heval : getEvalResultMultiDLDS d P =
        getEvalResultMultiDLDS d (multiPathsFromFlow d) :=
      admissibleMulti_eval_canonical d hslot hadm
    have hcanon : DischargedMulti d (multiPathsFromFlow d) (goalColumn d) :=
      discharge_multi d hdis
    unfold DischargedMulti at hcanon ⊢
    rw [heval]; exact hcanon
  · left; exact hadm





#print axioms compressed_universally_accepted'
end Semantic
