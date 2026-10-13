import Evaluator.MultiToken.Discharge

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

open DLDS

namespace DLDS

open FlowSemantics
open DLDS.Examples.BinaryFan DLDS.Examples.TernaryFan


/-- The cursor represented by a raw Flow entry at its current vertex.  Source
entries are continuation requests.  An empty entry produced below a source is
an exhausted arrival; nonempty entries have the same continuation state in
either case. -/
def flowCursorAt (d : Graph) (v : Vertex) (route : FlowPair) : ResidualCursor :=
  if v.HYPOTHESIS || (predsOf d v).isEmpty then
    .initial route.2
  else
    { residual := route.2, exhaustedArrival := route.2.isEmpty }

/-- A raw Flow entry has genuinely arrived and cannot continue at this
vertex.  In particular, an initial empty source route is not an arrival, and
an empty arrival is not exempt when a legitimate colour-0 edge exists. -/
def ExhaustedFlowArrivalAt (d : Graph) (v : Vertex) (route : FlowPair) : Prop :=
  let cursor := flowCursorAt d v route
  cursor.exhaustedArrival = true ∧ cursor.residual = [] ∧
    edgeOfColour d v 0 = none

def exhaustedFlowArrivalAtB (d : Graph) (v : Vertex) (route : FlowPair) : Bool :=
  let cursor := flowCursorAt d v route
  cursor.exhaustedArrival && cursor.residual.isEmpty &&
    (edgeOfColour d v 0).isNone

/-- Head coverage after making the arrival/continuation cursor explicit.
Every route that is actually going somewhere still has a matching outgoing
edge; only a genuinely exhausted arrival may stop without a head.  The
edge-to-route direction is unchanged. -/
def CursorRouteHeadCoverage (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, get_rule.outgoing v d ≠ [] →
    (∀ r ∈ flowAt d (stdFuel d) v,
        ExhaustedFlowArrivalAt d v r ∨
          ∃ e ∈ get_rule.outgoing v d, e.COLOUR = headColour r.2) ∧
    (∀ e ∈ get_rule.outgoing v d,
        ∃ r ∈ flowAt d (stdFuel d) v, headColour r.2 = e.COLOUR)

lemma cursorRouteHeadCoverage_of_raw (d : Graph) (h : RouteHeadCoverage d) :
    CursorRouteHeadCoverage d := by
  intro v hv hout
  obtain ⟨hroutes, hedges⟩ := h v hv hout
  constructor
  · intro route hroute
    exact Or.inr (hroutes route hroute)
  · exact hedges

def cursorRouteHeadCoverageAtB (d : Graph) (v : Vertex) : Bool :=
  let outgoing := get_rule.outgoing v d
  ((flowAt d (stdFuel d) v).all fun route =>
      exhaustedFlowArrivalAtB d v route ||
        outgoing.any fun edge => edge.COLOUR == headColour route.2) &&
  (outgoing.all fun edge =>
    (flowAt d (stdFuel d) v).any fun route => headColour route.2 == edge.COLOUR)


/-- A tail obtained by consuming an incoming edge is an exhausted arrival at
`w` exactly when it is empty, `w` is not terminal, and no legitimate colour-0
continuation exists. -/
def ExhaustedConsumedTailAt (d : Graph) (w : Vertex) (tail : ColourPath) : Prop :=
  get_rule.outgoing w d ≠ [] ∧ tail = [] ∧ edgeOfColour d w 0 = none

def exhaustedConsumedTailAtB (d : Graph) (w : Vertex)
    (tail : ColourPath) : Bool :=
  !(get_rule.outgoing w d).isEmpty && tail.isEmpty &&
    (edgeOfColour d w 0).isNone



/-- Partner agreement for one elimination pair at `w`. -/
def PairFlowTailClosure (d : Graph) (w M S : Vertex) : Prop :=
  (∀ bp ∈ flowAt d (stdFuel d) S, ∀ eS ∈ edgesBetween d S w, ∀ q,
      consume eS bp.2 = some q →
      ∃ bq ∈ flowAt d (stdFuel d) M, ∃ eM ∈ edgesBetween d M w,
        consume eM bq.2 = some q) ∧
  (∀ bp ∈ flowAt d (stdFuel d) M, ∀ eM ∈ edgesBetween d M w, ∀ q,
      consume eM bp.2 = some q →
      ∃ bq ∈ flowAt d (stdFuel d) S, ∃ eS ∈ edgesBetween d S w,
        consume eS bq.2 = some q)

/-- A predecessor is one of the two premises of a formula-compatible
elimination occurrence ending at `w`. -/
def ElimPremiseAt (d : Graph) (w u : Vertex) : Prop :=
  ∃ M S, (M, S) ∈ elimPairsAt d w ∧ (u = M ∨ u = S)

/-- An ordinary-introduction carrier is accounted for when its formula has the
required consequent relationship and every route it can consume across an
edge into `w` is present at `w` with the consumed residual. -/
def CarrierAccountedAt (d : Graph) (w u : Vertex) : Prop :=
  ∃ alpha, w.FORMULA = Formula.implication alpha u.FORMULA ∧
    ∀ bp ∈ flowAt d (stdFuel d) u, ∀ edge ∈ edgesBetween d u w, ∀ tail,
      consume edge bp.2 = some tail →
        ∃ dependency, (dependency, tail) ∈ flowAt d (stdFuel d) w

/-- The original tail-preserving cursor condition used by the generic
continuation theorem. -/
def CursorPairContinuationClosure (d : Graph) (w M S : Vertex) : Prop :=
  (∀ bp ∈ flowAt d (stdFuel d) S, ∀ eS ∈ edgesBetween d S w, ∀ q,
      consume eS bp.2 = some q →
      ExhaustedConsumedTailAt d w q ∨
        ∃ bq ∈ flowAt d (stdFuel d) M, ∃ eM ∈ edgesBetween d M w,
          consume eM bq.2 = some q) ∧
  (∀ bp ∈ flowAt d (stdFuel d) M, ∀ eM ∈ edgesBetween d M w, ∀ q,
      consume eM bp.2 = some q →
      ExhaustedConsumedTailAt d w q ∨
        ∃ bq ∈ flowAt d (stdFuel d) S, ∃ eS ∈ edgesBetween d S w,
          consume eS bq.2 = some q)

/-- Occurrence-aware partner agreement.  Ordinary elimination retains exact
tail agreement.  A mismatch is admitted only for a concrete clause-8
occurrence supported by one eligible stored ancestral address. -/
def CursorPairFlowTailClosure (d : Graph) (w M S : Vertex) : Prop :=
  (∀ bp ∈ flowAt d (stdFuel d) S, ∀ eS ∈ edgesBetween d S w, ∀ q,
      consume eS bp.2 = some q →
      ExhaustedConsumedTailAt d w q ∨
        ∃ bq ∈ flowAt d (stdFuel d) M, ∃ eM ∈ edgesBetween d M w,
          consume eM bq.2 = some q ∨
            reseedPairSupportsB d w bp eS bq eM = true) ∧
  (∀ bp ∈ flowAt d (stdFuel d) M, ∀ eM ∈ edgesBetween d M w, ∀ q,
      consume eM bp.2 = some q →
      ExhaustedConsumedTailAt d w q ∨
        ∃ bq ∈ flowAt d (stdFuel d) S, ∃ eS ∈ edgesBetween d S w,
          consume eS bq.2 = some q ∨
            reseedPairSupportsB d w bq eS bp eM = true)

/-- Cursor-aware carrier accounting. -/
def CursorCarrierAccountedAt (d : Graph) (w u : Vertex) : Prop :=
  ∃ alpha, w.FORMULA = Formula.implication alpha u.FORMULA ∧
    ∀ bp ∈ flowAt d (stdFuel d) u, ∀ edge ∈ edgesBetween d u w, ∀ tail,
      consume edge bp.2 = some tail →
      ExhaustedConsumedTailAt d w tail ∨
        ∃ dependency, (dependency, tail) ∈ flowAt d (stdFuel d) w

/-- Flow-tail closure with consumed arrival cursors.  The predecessor-shape
clause is unchanged. -/
def CursorFlowTailClosure (d : Graph) : Prop :=
  ∀ w ∈ d.NODES, elimPairsAt d w ≠ [] →
    (∀ u ∈ predsOf d w,
      ElimPremiseAt d w u ∨ CursorCarrierAccountedAt d w u) ∧
    (∀ M S, (M, S) ∈ elimPairsAt d w →
      CursorPairFlowTailClosure d w M S)

/-- Strong tail-preserving closure retained by the generic theorem which
recursively re-establishes exact `flowAt` membership after every edge. -/
def CursorFlowContinuationClosure (d : Graph) : Prop :=
  ∀ w ∈ d.NODES, elimPairsAt d w ≠ [] →
    (∀ u ∈ predsOf d w,
      ElimPremiseAt d w u ∨ CursorCarrierAccountedAt d w u) ∧
    (∀ M S, (M, S) ∈ elimPairsAt d w →
      CursorPairContinuationClosure d w M S)

/--
 **FlowTailClosure** (construction invariant #5). At every elimination kernel,
 every raw predecessor is either an elimination premise or an accounted-for
 ordinary-introduction carrier.  Independently, partner agreement holds in
 both directions for every elimination pair.
-/
def FlowTailClosure (d : Graph) : Prop :=
  ∀ w ∈ d.NODES, elimPairsAt d w ≠ [] →
    (∀ u ∈ predsOf d w,
      ElimPremiseAt d w u ∨ CarrierAccountedAt d w u) ∧
    (∀ M S, (M, S) ∈ elimPairsAt d w → PairFlowTailClosure d w M S)

lemma cursorFlowTailClosure_of_raw (d : Graph) (h : FlowTailClosure d) :
    CursorFlowTailClosure d := by
  intro w hw hpairs
  obtain ⟨hpred, hpairsRaw⟩ := h w hw hpairs
  constructor
  · intro u hu
    rcases hpred u hu with helim | hcarrier
    · exact Or.inl helim
    · right
      obtain ⟨alpha, hformula, haccounted⟩ := hcarrier
      refine ⟨alpha, hformula, ?_⟩
      intro bp hbp edge hedge tail hconsume
      exact Or.inr (haccounted bp hbp edge hedge tail hconsume)
  · intro M S hpair
    obtain ⟨hminor, hmajor⟩ := hpairsRaw M S hpair
    constructor
    · intro bp hbp edge hedge tail hconsume
      obtain ⟨bq, hbq, other, hother, hagree⟩ :=
        hminor bp hbp edge hedge tail hconsume
      exact Or.inr ⟨bq, hbq, other, hother, Or.inl hagree⟩
    · intro bp hbp edge hedge tail hconsume
      obtain ⟨bq, hbq, other, hother, hagree⟩ :=
        hmajor bp hbp edge hedge tail hconsume
      exact Or.inr ⟨bq, hbq, other, hother, Or.inl hagree⟩

/-- The original raw tail-closure invariant supplies exact partner tails, so it
also proves the stronger continuation form used by the generic recursive
soundness theorem. -/
lemma cursorFlowContinuationClosure_of_raw (d : Graph) (h : FlowTailClosure d) :
    CursorFlowContinuationClosure d := by
  intro w hw hpairs
  obtain ⟨hpred, hpairsRaw⟩ := h w hw hpairs
  constructor
  · intro u hu
    rcases hpred u hu with helim | hcarrier
    · exact Or.inl helim
    · right
      obtain ⟨alpha, hformula, haccounted⟩ := hcarrier
      refine ⟨alpha, hformula, ?_⟩
      intro bp hbp edge hedge tail hconsume
      exact Or.inr (haccounted bp hbp edge hedge tail hconsume)
  · intro M S hpair
    obtain ⟨hminor, hmajor⟩ := hpairsRaw M S hpair
    constructor
    · intro bp hbp edge hedge tail hconsume
      exact Or.inr (hminor bp hbp edge hedge tail hconsume)
    · intro bp hbp edge hedge tail hconsume
      exact Or.inr (hmajor bp hbp edge hedge tail hconsume)

def pairFlowTailClosureAtB (d : Graph) (w M S : Vertex) : Bool :=
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

def cursorPairFlowTailClosureAtB (d : Graph) (w M S : Vertex) : Bool :=
  ((flowAt d (stdFuel d) S).all fun bp =>
    (edgesBetween d S w).all fun eS =>
      match consume eS bp.2 with
      | none => true
      | some q =>
          exhaustedConsumedTailAtB d w q ||
            (flowAt d (stdFuel d) M).any fun bq =>
              (edgesBetween d M w).any fun eM =>
                consume eM bq.2 == some q ||
                  reseedPairSupportsB d w bp eS bq eM) &&
  ((flowAt d (stdFuel d) M).all fun bp =>
    (edgesBetween d M w).all fun eM =>
      match consume eM bp.2 with
      | none => true
      | some q =>
          exhaustedConsumedTailAtB d w q ||
            (flowAt d (stdFuel d) S).any fun bq =>
              (edgesBetween d S w).any fun eS =>
                consume eS bq.2 == some q ||
                  reseedPairSupportsB d w bq eS bp eM)

def carrierAccountedAtB (d : Graph) (w u : Vertex) : Bool :=
  match w.FORMULA with
  | Formula.implication _ beta =>
      u.FORMULA == beta &&
        (flowAt d (stdFuel d) u).all fun bp =>
          (edgesBetween d u w).all fun edge =>
            match consume edge bp.2 with
            | none => true
            | some tail =>
                (flowAt d (stdFuel d) w).any fun result => result.2 == tail
  | _ => false

def cursorCarrierAccountedAtB (d : Graph) (w u : Vertex) : Bool :=
  match w.FORMULA with
  | Formula.implication _ beta =>
      u.FORMULA == beta &&
        (flowAt d (stdFuel d) u).all fun bp =>
          (edgesBetween d u w).all fun edge =>
            match consume edge bp.2 with
            | none => true
            | some tail =>
                exhaustedConsumedTailAtB d w tail ||
                  (flowAt d (stdFuel d) w).any fun result => result.2 == tail
  | _ => false

def cursorRouteHeadCoverageB (d : Graph) : Bool :=
  d.NODES.all fun v =>
    (get_rule.outgoing v d).isEmpty || cursorRouteHeadCoverageAtB d v

def cursorFlowContinuationClosureB (d : Graph) : Bool :=
  d.NODES.all fun w =>
    match elimPairsAt d w with
    | [] => true
    | pairs =>
        (predsOf d w).all (fun u =>
          elimPremiseAtB d w u || cursorCarrierAccountedAtB d w u) &&
        pairs.all fun (major, minor) =>
          ((flowAt d (stdFuel d) minor).all fun route =>
            (edgesBetween d minor w).all fun edge =>
              match consume edge route.2 with
              | none => true
              | some tail => exhaustedConsumedTailAtB d w tail ||
                  (flowAt d (stdFuel d) major).any fun partner =>
                    (edgesBetween d major w).any fun other =>
                      consume other partner.2 == some tail) &&
          ((flowAt d (stdFuel d) major).all fun route =>
            (edgesBetween d major w).all fun edge =>
              match consume edge route.2 with
              | none => true
              | some tail => exhaustedConsumedTailAtB d w tail ||
                  (flowAt d (stdFuel d) minor).any fun partner =>
                    (edgesBetween d minor w).any fun other =>
                      consume other partner.2 == some tail)

lemma elimPremiseAtB_sound (d : Graph) (w u : Vertex)
    (h : elimPremiseAtB d w u = true) : ElimPremiseAt d w u := by
  unfold elimPremiseAtB at h
  rw [List.any_eq_true] at h
  obtain ⟨pair, hpair, hside⟩ := h
  rcases pair with ⟨major, minor⟩
  refine ⟨major, minor, hpair, ?_⟩
  rw [Bool.or_eq_true] at hside
  rcases hside with hmajor | hminor
  · have hmajor' : major = u := beq_iff_eq.mp hmajor
    exact Or.inl hmajor'.symm
  · have hminor' : minor = u := beq_iff_eq.mp hminor
    exact Or.inr hminor'.symm

lemma carrierAccountedAtB_sound (d : Graph) (w u : Vertex)
    (h : carrierAccountedAtB d w u = true) : CarrierAccountedAt d w u := by
  unfold carrierAccountedAtB at h
  cases hformula : w.FORMULA with
  | atom name => simp [hformula] at h
  | implication alpha beta =>
      simp only [hformula, Bool.and_eq_true, List.all_eq_true] at h
      have huFormula : u.FORMULA = beta := by simpa using h.1
      refine ⟨alpha, by simp [hformula, huFormula], ?_⟩
      intro bp hbp edge hedge tail hconsume
      have hresult := h.2 bp hbp edge hedge
      simp only [hconsume] at hresult
      rw [List.any_eq_true] at hresult
      obtain ⟨result, hresultMem, htail⟩ := hresult
      rcases result with ⟨dependency, residual⟩
      have htail' : residual = tail := by simpa using htail
      subst tail
      exact ⟨dependency, hresultMem⟩

/--
 Bool mirror: checks the partner-agreement pairing directly (inspects
    `elimPairsAt`/`consume`, not `flowAt w`-membership).
-/
def flowTailClosureAtB (d : Graph) (w : Vertex) : Bool :=
  match elimPairsAt d w with
  | [] => true
  | pairs =>
      (predsOf d w).all (fun u =>
        elimPremiseAtB d w u || carrierAccountedAtB d w u) &&
      pairs.all fun (M, S) => pairFlowTailClosureAtB d w M S

def flowTailClosureB (d : Graph) : Bool := d.NODES.all (flowTailClosureAtB d)

def cursorFlowTailClosureAtB (d : Graph) (w : Vertex) : Bool :=
  match elimPairsAt d w with
  | [] => true
  | pairs =>
      (predsOf d w).all (fun u =>
        elimPremiseAtB d w u || cursorCarrierAccountedAtB d w u) &&
      pairs.all fun (M, S) => cursorPairFlowTailClosureAtB d w M S

def cursorFlowTailClosureB (d : Graph) : Bool :=
  d.NODES.all (cursorFlowTailClosureAtB d)

lemma exhaustedFlowArrivalAtB_sound (d : Graph) (v : Vertex)
    (route : FlowPair) (h : exhaustedFlowArrivalAtB d v route = true) :
    ExhaustedFlowArrivalAt d v route := by
  unfold exhaustedFlowArrivalAtB at h
  simp only [Bool.and_eq_true, List.isEmpty_iff,
    Option.isNone_iff_eq_none] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

lemma exhaustedConsumedTailAtB_sound (d : Graph) (w : Vertex)
    (tail : ColourPath) (h : exhaustedConsumedTailAtB d w tail = true) :
    ExhaustedConsumedTailAt d w tail := by
  unfold exhaustedConsumedTailAtB at h
  simp only [Bool.and_eq_true,
    List.isEmpty_iff, Option.isNone_iff_eq_none] at h
  have hout : get_rule.outgoing w d ≠ [] := by
    intro hempty
    have hisEmpty : (get_rule.outgoing w d).isEmpty = true := by simp [hempty]
    rw [hisEmpty] at h
    simp at h
  exact ⟨hout, h.1.2, h.2⟩

theorem cursorRouteHeadCoverageB_sound (d : Graph) :
    cursorRouteHeadCoverageB d = true → CursorRouteHeadCoverage d := by
  classical
  intro h
  unfold cursorRouteHeadCoverageB at h
  rw [List.all_eq_true] at h
  intro v hv hout
  have hvcheck := h v hv
  have houtEmpty : (get_rule.outgoing v d).isEmpty = false := by
    simpa [List.isEmpty_iff] using hout
  simp only [houtEmpty, Bool.false_or] at hvcheck
  unfold cursorRouteHeadCoverageAtB at hvcheck
  simp only [Bool.and_eq_true, List.all_eq_true] at hvcheck
  constructor
  · intro route hroute
    have hr := hvcheck.1 route hroute
    rw [Bool.or_eq_true] at hr
    rcases hr with hexhausted | hcovered
    · exact Or.inl (exhaustedFlowArrivalAtB_sound d v route hexhausted)
    · right
      rw [List.any_eq_true] at hcovered
      obtain ⟨edge, hedge, hcolour⟩ := hcovered
      exact ⟨edge, hedge, by simpa using hcolour⟩
  · intro edge hedge
    have he := hvcheck.2 edge hedge
    rw [List.any_eq_true] at he
    obtain ⟨route, hroute, hcolour⟩ := he
    exact ⟨route, hroute, by simpa using hcolour⟩

lemma cursorPairFlowTailClosureAtB_sound (d : Graph) (w M S : Vertex)
    (h : cursorPairFlowTailClosureAtB d w M S = true) :
    CursorPairFlowTailClosure d w M S := by
  unfold cursorPairFlowTailClosureAtB at h
  unfold CursorPairFlowTailClosure
  simp only [Bool.and_eq_true, List.all_eq_true] at h
  constructor
  · intro bp hbp eS heS q hc
    have hx := h.1 bp hbp eS heS
    simp only [hc] at hx
    rw [Bool.or_eq_true] at hx
    rcases hx with hexhausted | hpartner
    · exact Or.inl (exhaustedConsumedTailAtB_sound d w q hexhausted)
    · right
      rw [List.any_eq_true] at hpartner
      obtain ⟨bq, hbq, hedges⟩ := hpartner
      rw [List.any_eq_true] at hedges
      obtain ⟨eM, heM, hconsume⟩ := hedges
      rw [Bool.or_eq_true] at hconsume
      rcases hconsume with hagree | hreseed
      · exact ⟨bq, hbq, eM, heM, Or.inl (by simpa using hagree)⟩
      · exact ⟨bq, hbq, eM, heM, Or.inr hreseed⟩
  · intro bp hbp eM heM q hc
    have hx := h.2 bp hbp eM heM
    simp only [hc] at hx
    rw [Bool.or_eq_true] at hx
    rcases hx with hexhausted | hpartner
    · exact Or.inl (exhaustedConsumedTailAtB_sound d w q hexhausted)
    · right
      rw [List.any_eq_true] at hpartner
      obtain ⟨bq, hbq, hedges⟩ := hpartner
      rw [List.any_eq_true] at hedges
      obtain ⟨eS, heS, hconsume⟩ := hedges
      rw [Bool.or_eq_true] at hconsume
      rcases hconsume with hagree | hreseed
      · exact ⟨bq, hbq, eS, heS, Or.inl (by simpa using hagree)⟩
      · exact ⟨bq, hbq, eS, heS, Or.inr hreseed⟩

lemma cursorCarrierAccountedAtB_sound (d : Graph) (w u : Vertex)
    (h : cursorCarrierAccountedAtB d w u = true) :
    CursorCarrierAccountedAt d w u := by
  unfold cursorCarrierAccountedAtB at h
  cases hformula : w.FORMULA with
  | atom name => simp [hformula] at h
  | implication alpha beta =>
      simp only [hformula, Bool.and_eq_true, List.all_eq_true] at h
      have huFormula : u.FORMULA = beta := by simpa using h.1
      refine ⟨alpha, by simp [hformula, huFormula], ?_⟩
      intro bp hbp edge hedge tail hconsume
      have hresult := h.2 bp hbp edge hedge
      simp only [hconsume] at hresult
      rw [Bool.or_eq_true] at hresult
      rcases hresult with hexhausted | hcontinued
      · exact Or.inl (exhaustedConsumedTailAtB_sound d w tail hexhausted)
      · right
        rw [List.any_eq_true] at hcontinued
        obtain ⟨result, hresultMem, htail⟩ := hcontinued
        rcases result with ⟨dependency, residual⟩
        have htail' : residual = tail := by simpa using htail
        subst tail
        exact ⟨dependency, hresultMem⟩

theorem cursorFlowContinuationClosureB_sound (d : Graph) (h : cursorFlowContinuationClosureB d = true) :
    CursorFlowContinuationClosure d := by
  classical
  unfold cursorFlowContinuationClosureB at h
  rw [List.all_eq_true] at h
  intro w hw hpairsNe
  have hw' := h w hw
  cases hpairs : elimPairsAt d w with
  | nil => exact (hpairsNe hpairs).elim
  | cons pair pairs =>
      rw [hpairs] at hw'
      simp only [Bool.and_eq_true, List.all_eq_true] at hw'
      constructor
      · intro u hu
        have hu' := hw'.1 u hu
        rw [Bool.or_eq_true] at hu'
        rcases hu' with helim | hcarrier
        · exact Or.inl (elimPremiseAtB_sound d w u helim)
        · exact Or.inr (cursorCarrierAccountedAtB_sound d w u hcarrier)
      · intro major minor hpair
        have checked := hw'.2 (major, minor) (by simpa [hpairs] using hpair)
        constructor
        · intro route hroute edge hedge tail hconsume
          have result := checked.1 route hroute edge hedge
          simp only [hconsume] at result
          rw [Bool.or_eq_true] at result
          rcases result with hexhausted | hpartner
          · exact Or.inl (exhaustedConsumedTailAtB_sound d w tail hexhausted)
          · rw [List.any_eq_true] at hpartner
            obtain ⟨partner, hpartnerMem, hother⟩ := hpartner
            rw [List.any_eq_true] at hother
            obtain ⟨other, hotherMem, hagree⟩ := hother
            exact Or.inr ⟨partner, hpartnerMem, other, hotherMem,
              by simpa using hagree⟩
        · intro route hroute edge hedge tail hconsume
          have result := checked.2 route hroute edge hedge
          simp only [hconsume] at result
          rw [Bool.or_eq_true] at result
          rcases result with hexhausted | hpartner
          · exact Or.inl (exhaustedConsumedTailAtB_sound d w tail hexhausted)
          · rw [List.any_eq_true] at hpartner
            obtain ⟨partner, hpartnerMem, hother⟩ := hpartner
            rw [List.any_eq_true] at hother
            obtain ⟨other, hotherMem, hagree⟩ := hother
            exact Or.inr ⟨partner, hpartnerMem, other, hotherMem,
              by simpa using hagree⟩

theorem cursorFlowTailClosureB_sound (d : Graph) :
    cursorFlowTailClosureB d = true → CursorFlowTailClosure d := by
  classical
  intro h
  unfold cursorFlowTailClosureB at h
  rw [List.all_eq_true] at h
  intro w hw hpairsNe
  have hw' := h w hw
  unfold cursorFlowTailClosureAtB at hw'
  cases hpairs : elimPairsAt d w with
  | nil => exact (hpairsNe hpairs).elim
  | cons pair pairs =>
      rw [hpairs] at hw'
      simp only [Bool.and_eq_true, List.all_eq_true] at hw'
      constructor
      · intro u hu
        have hu' := hw'.1 u hu
        rw [Bool.or_eq_true] at hu'
        rcases hu' with helim | hcarrier
        · exact Or.inl (elimPremiseAtB_sound d w u helim)
        · exact Or.inr (cursorCarrierAccountedAtB_sound d w u hcarrier)
      · intro M S hpair
        apply cursorPairFlowTailClosureAtB_sound
        apply hw'.2 (M, S)
        simpa [hpairs] using hpair



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



namespace Examples.RouteGap

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

end Examples.RouteGap

open DLDS.Examples.RouteGap

def nodeElimShapeB (d : Graph) (v : Vertex) : Bool :=
  !v.HYPOTHESIS && !(predsOf d v).isEmpty &&
    (match elimPairsAt d v with | [_] => true | _ => false)

def nodeElimShapeWideB (d : Graph) (v : Vertex) : Bool :=
  !v.HYPOTHESIS && !(predsOf d v).isEmpty && !(elimPairsAt d v).isEmpty

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
  simp only [hguard, if_true, normalizeDep_singleton]

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

lemma mem_preds_of_mem_elimPairsAt {d : Graph} {v M S : Vertex}
    (hmem : (M, S) ∈ elimPairsAt d v) :
    M ∈ predsOf d v ∧ S ∈ predsOf d v := by
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

lemma reseedPairs_congr_of_preds (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hflow : ∀ u ∈ predsOf d v, left u = right u) :
    reseedPairs d v left = reseedPairs d v right := by
  unfold reseedPairs
  generalize hanc : eligibleAncestorsAtRoot d v = ancestors
  cases hempty : ancestors.isEmpty with
  | true => simp [hempty]
  | false =>
    simp only [hempty, Bool.false_eq_true, if_false]
    apply congrArg List.eraseDups
    apply congrArg₂ (· ++ ·)
    · unfold elimReseedPairs
      apply List.flatMap_congr
      intro pair hpair
      rcases pair with ⟨major, minor⟩
      obtain ⟨hmajor, hminor⟩ := mem_preds_of_mem_elimPairsAt hpair
      simp only
      rw [hflow minor hminor, hflow major hmajor]
    · unfold introReseedPairs
      cases hp : predsOf d v with
      | nil => simp
      | cons premise rest =>
          cases rest with
          | nil =>
              have hpremise : premise ∈ predsOf d v := by simp [hp]
              cases hformula : v.FORMULA <;>
                simp [hflow premise hpremise]
          | cons premise' rest => simp

lemma mixedIntroPairs_congr_of_preds (d : Graph) (v : Vertex)
    (left right : Vertex → List FlowPair)
    (hflow : ∀ u ∈ predsOf d v, left u = right u) :
    mixedIntroPairs d v left = mixedIntroPairs d v right := by
  unfold mixedIntroPairs
  cases hformula : v.FORMULA with
  | atom => rfl
  | implication alpha beta =>
      apply List.flatMap_congr
      intro premise hpremise
      rw [hflow premise hpremise]

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
          ⟨_, _, hpairs⟩ | ⟨p, α, β, _, hincoming, hform, hpβ, helim⟩
        · rw [flowAt_kernel_elim_many_eq d f hhyp hpredsne hpairs]
          conv_rhs =>
            rw [hf1, flowAt_kernel_elim_many_eq d (f - 1) hhyp hpredsne hpairs]
          apply congrArg List.eraseDups
          apply congrArg₂ (· ++ ·)
          · apply congrArg₂ (· ++ ·)
            · apply List.flatMap_congr
              intro pair hpair
              rcases pair with ⟨major, minor⟩
              obtain ⟨hmajor, hminor⟩ := mem_preds_of_mem_elimPairsAt hpair
              change elimCombine d v major minor (flowAt d f minor) (flowAt d f major) =
                elimCombine d v major minor (flowAt d (f - 1) minor)
                  (flowAt d (f - 1) major)
              rw [prem minor hminor, prem major hmajor]
            · exact mixedIntroPairs_congr_of_preds d v _ _ prem
          · exact reseedPairs_congr_of_preds d v _ _ prem
        · have hpreds : predsOf d v = [p.START] := by rw [predsOf_eq, hincoming]; rfl
          have hSp : p.START ∈ predsOf d v := by rw [hpreds]; exact List.mem_singleton_self _
          rw [flowAt_kernel_intro_eq d f hhyp hpreds hform hpβ helim]
          conv_rhs => rw [hf1, flowAt_kernel_intro_eq d (f - 1) hhyp hpreds hform hpβ helim]
          apply congrArg List.eraseDups
          apply congrArg₂ (· ++ ·)
          · rw [prem p.START hSp]
          · exact reseedPairs_congr_of_preds d v _ _ prem
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
 On a legal graph with `CursorFlowContinuationClosure`,
    a route `(b,p)` at `v` crossing the outgoing edge `e` whose colour is the
    residual head continues, with SOME dependency, as residual `p.tail` at the
    target `e.END` (which is assumed to have an outgoing edge ; the non-root
    targets the canonical walk actually queries). The elim case is exactly the
    partner agreement; intro/top are direct.
-/
theorem flowAt_tail_closure (d : Graph)
    (hstruct : structuralValid d)
    (hhypNoInc : HypothesesHaveNoIncoming d)
    (hftc : CursorFlowContinuationClosure d)
    {v : Vertex} (hv : v ∈ d.NODES)
    {b : Dep} {p : ColourPath} (hbp : (b, p) ∈ flowAt d (stdFuel d) v)
    {e : Deduction} (he : e ∈ get_rule.outgoing v d)
    (hcol : e.COLOUR = headColour p)
    (hwout : get_rule.outgoing e.END d ≠ []) :
    (¬ ExhaustedConsumedTailAt d e.END p.tail) →
    ∃ b', (b', p.tail) ∈ flowAt d (stdFuel d) e.END := by
  intro hnotExhausted
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
    ⟨_, _, hpairsNe⟩ | ⟨pe, α, β, _, hwincoming, hwform, hpβ, helim⟩
  ·
    obtain ⟨hpredCoverage, hpairClosure⟩ := hftc w hwmem hpairsNe
    rcases hpredCoverage v hvpred with
      ⟨M, S, hpair, hvM | hvS⟩ | hcarrier
    ·
      obtain ⟨hagreeS, hagreeM⟩ := hpairClosure M S hpair
      obtain ⟨hMp, hSp⟩ := mem_preds_of_mem_elimPairsAt hpair
      have hMnode : M ∈ d.NODES := mem_predsOf_mem_nodes hstruct hMp
      have hSnode : S ∈ d.NODES := mem_predsOf_mem_nodes hstruct hSp
      have hMout : get_rule.outgoing M d ≠ [] := (predsOf_props hstruct hMp).2
      have hSout : get_rule.outgoing S d ≠ [] := (predsOf_props hstruct hSp).2
      have hwflow : flowAt d (stdFuel d) w =
          (((elimPairsAt d w).flatMap fun (major, minor) =>
              elimCombine d w major minor (flowAt d (stdFuel d - 1) minor)
                (flowAt d (stdFuel d - 1) major)) ++
            mixedIntroPairs d w (flowAt d (stdFuel d - 1)) ++
            reseedPairs d w (flowAt d (stdFuel d - 1))).eraseDups := by
        conv_lhs => rw [hstd1]
        rw [flowAt_kernel_elim_many_eq d (stdFuel d - 1) hwhyp hwpredsne hpairsNe]
      have hbpM : (b, p) ∈ flowAt d (stdFuel d) M := by rw [← hvM]; exact hbp
      have heMw : e ∈ edgesBetween d M w := by rw [← hvM]; exact heS
      obtain hexhausted | ⟨bq, hbqS, eS, heSw, hcS⟩ :=
        hagreeM (b, p) hbpM e heMw p.tail hconsume
      · exact (hnotExhausted hexhausted).elim
      have hbp' : (b, p) ∈ flowAt d (stdFuel d - 1) M := by
        rw [← flowAt_stdFuel_pred d hstruct hMnode hMout]; exact hbpM
      have hbq' : bq ∈ flowAt d (stdFuel d - 1) S := by
        rw [← flowAt_stdFuel_pred d hstruct hSnode hSout]; exact hbqS
      refine ⟨depUnion bq.1 (b, p).1, ?_⟩
      rw [hwflow, List.mem_eraseDups, List.mem_append]
      left
      rw [List.mem_append]
      left
      rw [List.mem_flatMap]
      exact ⟨(M, S), hpair,
        mem_elimCombine_of hbq' hbp' heSw heMw hcS hconsume⟩
    ·
      obtain ⟨hagreeS, hagreeM⟩ := hpairClosure M S hpair
      obtain ⟨hMp, hSp⟩ := mem_preds_of_mem_elimPairsAt hpair
      have hMnode : M ∈ d.NODES := mem_predsOf_mem_nodes hstruct hMp
      have hSnode : S ∈ d.NODES := mem_predsOf_mem_nodes hstruct hSp
      have hMout : get_rule.outgoing M d ≠ [] := (predsOf_props hstruct hMp).2
      have hSout : get_rule.outgoing S d ≠ [] := (predsOf_props hstruct hSp).2
      have hwflow : flowAt d (stdFuel d) w =
          (((elimPairsAt d w).flatMap fun (major, minor) =>
              elimCombine d w major minor (flowAt d (stdFuel d - 1) minor)
                (flowAt d (stdFuel d - 1) major)) ++
            mixedIntroPairs d w (flowAt d (stdFuel d - 1)) ++
            reseedPairs d w (flowAt d (stdFuel d - 1))).eraseDups := by
        conv_lhs => rw [hstd1]
        rw [flowAt_kernel_elim_many_eq d (stdFuel d - 1) hwhyp hwpredsne hpairsNe]
      have hbpS : (b, p) ∈ flowAt d (stdFuel d) S := by rw [← hvS]; exact hbp
      have heSw : e ∈ edgesBetween d S w := by rw [← hvS]; exact heS
      obtain hexhausted | ⟨bq, hbqM, eM, heMw, hcM⟩ :=
        hagreeS (b, p) hbpS e heSw p.tail hconsume
      · exact (hnotExhausted hexhausted).elim
      have hbp' : (b, p) ∈ flowAt d (stdFuel d - 1) S := by
        rw [← flowAt_stdFuel_pred d hstruct hSnode hSout]; exact hbpS
      have hbq' : bq ∈ flowAt d (stdFuel d - 1) M := by
        rw [← flowAt_stdFuel_pred d hstruct hMnode hMout]; exact hbqM
      refine ⟨depUnion (b, p).1 bq.1, ?_⟩
      rw [hwflow, List.mem_eraseDups, List.mem_append]
      left
      rw [List.mem_append]
      left
      rw [List.mem_flatMap]
      exact ⟨(M, S), hpair,
        mem_elimCombine_of hbp' hbq' heSw heMw hconsume hcM⟩
    · obtain ⟨_alpha, _hformula, haccounted⟩ := hcarrier
      rcases haccounted (b, p) hbp e heS p.tail hconsume with
        hexhausted | hcontinued
      · exact (hnotExhausted hexhausted).elim
      · exact hcontinued
  ·
    have hpreds : predsOf d w = [pe.START] := by rw [predsOf_eq, hwincoming]; rfl
    have hvpe : v = pe.START := by rw [hpreds] at hvpred; simpa using hvpred
    have hunode : pe.START ∈ d.NODES := by rw [← hvpe]; exact hv
    have huout : get_rule.outgoing pe.START d ≠ [] := by
      rw [← hvpe]; intro h; rw [h] at he; exact List.not_mem_nil he
    have hwflow : flowAt d (stdFuel d) w =
        (((flowAt d (stdFuel d - 1) pe.START).flatMap (fun bp =>
            (edgesBetween d pe.START w).filterMap (fun ed =>
              (consume ed bp.2).map (fun q => (depRemove bp.1 α, q))))) ++
          reseedPairs d w (flowAt d (stdFuel d - 1))).eraseDups := by
      conv_lhs => rw [hstd1]
      rw [flowAt_kernel_intro_eq d (stdFuel d - 1) hwhyp hpreds hwform hpβ helim]
    have hbp' : (b, p) ∈ flowAt d (stdFuel d - 1) pe.START := by
      rw [← flowAt_stdFuel_pred d hstruct hunode huout, ← hvpe]; exact hbp
    have heS' : e ∈ edgesBetween d pe.START w := by rw [← hvpe]; exact heS
    refine ⟨depRemove (b, p).1 α, ?_⟩
    rw [hwflow, List.mem_eraseDups, List.mem_append]
    left
    rw [List.mem_flatMap]
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

lemma exhaustedStopAllowedB_of_flowArrival (d : Graph)
    {lvl : Nat} {φ : Formula} {v : Vertex} {route : FlowPair}
    {cursor : ResidualCursor}
    (hnode : nodeAtLevelFormula? d lvl φ = some v)
    (hcursor : cursor = flowCursorAt d v route)
    (hexhausted : ExhaustedFlowArrivalAt d v route) :
    exhaustedStopAllowedB d lvl φ cursor = true := by
  unfold ExhaustedFlowArrivalAt at hexhausted
  dsimp only at hexhausted
  rcases hexhausted with ⟨harrival, hresidual, hedge⟩
  simp [exhaustedStopAllowedB, hnode, hcursor, harrival, hresidual, hedge]

@[simp] lemma routeFromFlowCursorC_initial (d : Graph)
    (formulas : List Formula) (fuel lvl : Nat) (φ : Formula)
    (residual : ColourPath) :
    routeFromFlowCursorC d formulas fuel lvl φ (.initial residual) =
      routeFromFlowC d formulas fuel lvl φ residual := by
  rfl

lemma routeFromFlowCursorC_eq_initial_of_not_allowed (d : Graph)
    (formulas : List Formula) :
    ∀ (fuel lvl : Nat) (φ : Formula) (cursor : ResidualCursor),
      exhaustedStopAllowedB d lvl φ cursor = false →
      routeFromFlowCursorC d formulas fuel lvl φ cursor =
        routeFromFlowC d formulas fuel lvl φ cursor.residual := by
  intro fuel
  induction fuel with
  | zero => intro lvl φ cursor _; rfl
  | succ fuel ih =>
      intro lvl φ cursor hstop
      rw [routeFromFlowCursorC]
      conv_rhs => rw [routeFromFlowC, routeFromFlowCursorC]
      simp only [hstop, exhaustedStopAllowedB_initial]
      cases hnode : nodeAtLevelFormula? d lvl φ with
      | none =>
          simp only [DLDS.ResidualCursor.initial]
          rw [ih _ _ _ hstop]
          simpa only [Bool.false_eq_true, if_false] using congrArg
            (fun rest => (0, 0) :: rest)
            (routeFromFlowCursorC_initial d formulas fuel lvl φ
              cursor.residual).symm
      | some v =>
          cases hout : get_rule.outgoing v d with
          | nil =>
              simp only [DLDS.ResidualCursor.initial, hout]
              rw [ih _ _ _ hstop]
              simpa only [Bool.false_eq_true, if_false] using congrArg
                (fun rest => (0, 0) :: rest)
                (routeFromFlowCursorC_initial d formulas fuel lvl φ
                  cursor.residual).symm
          | cons edge edges =>
              cases hedge : edgeOfColour d v (headColour cursor.residual) with
              | none =>
                  simp only [DLDS.ResidualCursor.initial, hout, hedge]
                  rw [ih _ _ _ hstop]
                  simpa only [Bool.false_eq_true, if_false] using congrArg
                    (fun rest => (0, 0) :: rest)
                    (routeFromFlowCursorC_initial d formulas fuel lvl φ
                      cursor.residual).symm
              | some selected =>
                  simp [DLDS.ResidualCursor.initial, hout, hedge, DLDS.ResidualCursor.afterEdge]

lemma routeFromFlowCursorC_stuck (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} (cursor : ResidualCursor)
    (hstuck : nodeAtLevelFormula? d lvl φ = none ∨
      ∃ v, nodeAtLevelFormula? d lvl φ = some v ∧ get_rule.outgoing v d = []) :
    ∀ fuel,
      routeFromFlowCursorC d formulas fuel lvl φ cursor =
        List.replicate fuel (0, 0) := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ f ih =>
      rw [List.replicate_succ, routeFromFlowCursorC]
      by_cases hstop : exhaustedStopAllowedB d lvl φ cursor = true
      · simp [hstop]
      · rcases hstuck with hnone | ⟨v, hsome, hterm⟩
        · simp [hstop, hnone, ih]
        · simp [hstop, hsome, hterm, ih]

lemma routeFromFlowC_stuck (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} (p : ColourPath)
    (hstuck : nodeAtLevelFormula? d lvl φ = none ∨
      ∃ v, nodeAtLevelFormula? d lvl φ = some v ∧ get_rule.outgoing v d = []) :
    ∀ fuel, routeFromFlowC d formulas fuel lvl φ p = List.replicate fuel (0, 0) := by
  intro fuel
  exact routeFromFlowCursorC_stuck d formulas (.initial p) hstuck fuel

lemma admChainCursorCB_stuck (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} (cursor : ResidualCursor)
    (hstuck : nodeAtLevelFormula? d lvl φ = none ∨
      ∃ v, nodeAtLevelFormula? d lvl φ = some v ∧ get_rule.outgoing v d = []) :
    ∀ fuel, admChainCursorCB d formulas lvl φ cursor
      (routeFromFlowCursorC d formulas fuel lvl φ cursor) = true := by
  intro fuel
  rw [routeFromFlowCursorC_stuck d formulas cursor hstuck fuel]
  cases fuel with
  | zero =>
      rcases hstuck with hnone | ⟨v, hsome, hterm⟩
      · simp [admChainCursorCB, hnone]
      · simp [admChainCursorCB, hsome, hterm]
  | succ f =>
      rw [List.replicate_succ]
      rcases hstuck with hnone | ⟨v, hsome, hterm⟩
      · simp [admChainCursorCB, hnone]
      · simp [admChainCursorCB, hsome, hterm]

lemma admChainCB_stuck (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} (p : ColourPath)
    (hstuck : nodeAtLevelFormula? d lvl φ = none ∨
      ∃ v, nodeAtLevelFormula? d lvl φ = some v ∧ get_rule.outgoing v d = []) :
    ∀ fuel, admChainCB d formulas lvl φ p
      (routeFromFlowC d formulas fuel lvl φ p) = true := by
  intro fuel
  exact admChainCursorCB_stuck d formulas (.initial p) hstuck fuel

lemma admChainCursorCB_route_of_exhaustedStop (d : Graph)
    (formulas : List Formula) {lvl : Nat} {φ : Formula}
    (cursor : ResidualCursor)
    (hstop : exhaustedStopAllowedB d lvl φ cursor = true) :
    ∀ fuel, admChainCursorCB d formulas lvl φ cursor
      (routeFromFlowCursorC d formulas fuel lvl φ cursor) = true := by
  intro fuel
  cases fuel with
  | zero => simp [routeFromFlowCursorC, admChainCursorCB, hstop]
  | succ fuel =>
      simp [routeFromFlowCursorC, admChainCursorCB, hstop]

/--  One active step of `routeFromFlowC`.  -/
lemma routeFromFlowC_step (d : Graph) (formulas : List Formula)
    {fuel lvl : Nat} {φ : Formula} {p : ColourPath} {v : Vertex}
    {e0 : Deduction} {es : List Deduction} {e : Deduction}
    (hnode : nodeAtLevelFormula? d lvl φ = some v)
    (hout : get_rule.outgoing v d = e0 :: es)
    (hedge : edgeOfColour d v (headColour p) = some e) :
    routeFromFlowC d formulas (fuel + 1) lvl φ p =
      (formulas.idxOf e.END.FORMULA + 1,
        inputLabelForArrivalC d formulas e.START e.END) ::
       (if isMinorArrivalVertexC d e.START e.END && p.tail.isEmpty then
        List.replicate fuel (0, 0)
       else routeFromFlowCursorC d formulas fuel (lvl - 1) e.END.FORMULA
         (DLDS.ResidualCursor.initial p).afterEdge) := by
  rw [routeFromFlowC, routeFromFlowCursorC]
  simp only [DLDS.ResidualCursor.initial, hnode]
  rw [hout]
  simp only
  rw [hedge]
  rfl

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
       (l == inputLabelForArrivalC d formulas e.START e.END) &&
       (if isMinorArrivalVertexC d e.START e.END && p.tail.isEmpty then
          rest.all (fun s => s == (0, 0))
        else admChainCursorCB d formulas (lvl - 1) e.END.FORMULA
          (DLDS.ResidualCursor.initial p).afterEdge rest)) := by
  rw [admChainCB, admChainCursorCB]
  simp [exhaustedStopAllowedB, DLDS.ResidualCursor.initial, hnode, hout, hedge]

lemma routeFromFlowCursorC_step (d : Graph) (formulas : List Formula)
    {fuel lvl : Nat} {φ : Formula} {cursor : ResidualCursor} {v : Vertex}
    {e0 : Deduction} {es : List Deduction} {e : Deduction}
    (hstop : exhaustedStopAllowedB d lvl φ cursor = false)
    (hnode : nodeAtLevelFormula? d lvl φ = some v)
    (hout : get_rule.outgoing v d = e0 :: es)
    (hedge : edgeOfColour d v (headColour cursor.residual) = some e) :
    routeFromFlowCursorC d formulas (fuel + 1) lvl φ cursor =
      (formulas.idxOf e.END.FORMULA + 1,
        inputLabelForArrivalC d formulas e.START e.END) ::
       (if isMinorArrivalVertexC d e.START e.END &&
            cursor.residual.tail.isEmpty then
          List.replicate fuel (0, 0)
        else routeFromFlowCursorC d formulas fuel (lvl - 1) e.END.FORMULA
          cursor.afterEdge) := by
  rw [routeFromFlowCursorC]
  simp [hstop, hnode, hout, hedge]

lemma admChainCursorCB_step (d : Graph) (formulas : List Formula)
    {lvl : Nat} {φ : Formula} {cursor : ResidualCursor}
    {t l : Nat} {rest : List (Nat × Nat)} {v : Vertex}
    {e0 : Deduction} {es : List Deduction} {e : Deduction}
    (hstop : exhaustedStopAllowedB d lvl φ cursor = false)
    (hnode : nodeAtLevelFormula? d lvl φ = some v)
    (hout : get_rule.outgoing v d = e0 :: es)
    (hedge : edgeOfColour d v (headColour cursor.residual) = some e) :
    admChainCursorCB d formulas lvl φ cursor ((t, l) :: rest) =
      ((t == formulas.idxOf e.END.FORMULA + 1) &&
       (l == inputLabelForArrivalC d formulas e.START e.END) &&
       (if isMinorArrivalVertexC d e.START e.END &&
            cursor.residual.tail.isEmpty then
          rest.all (fun s => s == (0, 0))
        else admChainCursorCB d formulas (lvl - 1) e.END.FORMULA
          cursor.afterEdge rest)) := by
  rw [admChainCursorCB.eq_def]
  simp [hstop, hnode, hout, hedge]

/--
 **routeFromFlowCursorC ⊨ admChainCursorCB** ; the canonical route-following passes the
    admissibility chain check, by induction along the route. The step where
    `admChainCB` could fail (`edgeOfColour = none`) is ruled out by
    `CursorRouteHeadCoverage`; the residual continues at the next node by
    `flowAt_tail_closure`. `lvl ≤ fuel` supplies just enough fuel (the walk
    descends one level per step and terminates at level 0 / a rootless node).
-/
lemma routeFromFlowCursorC_admChainCB (d : Graph)
    (hstruct : structuralValid d) (hhypNoInc : HypothesesHaveNoIncoming d)
    (hlfu : LevelFormulaUnique d) (hcoverage : CursorRouteHeadCoverage d)
    (hftc : CursorFlowContinuationClosure d) (formulas : List Formula) :
    ∀ (fuel lvl : Nat) (φ : Formula) (v : Vertex) (b : Dep)
      (cursor : ResidualCursor),
      nodeAtLevelFormula? d lvl φ = some v → v ∈ d.NODES →
      (b, cursor.residual) ∈ flowAt d (stdFuel d) v →
      cursor = flowCursorAt d v (b, cursor.residual) → lvl ≤ fuel →
      admChainCursorCB d formulas lvl φ cursor
        (routeFromFlowCursorC d formulas fuel lvl φ cursor) = true := by
  intro fuel
  induction fuel with
  | zero =>
      intro lvl φ v b cursor hnode hv hbp hcursor hle
      have hlvl0 : lvl = 0 := Nat.le_zero.mp hle
      have hvlvl : v.LEVEL = lvl := (nodeAtLevelFormula?_some hnode).2.1
      have hterm : get_rule.outgoing v d = [] :=
        level_zero_terminal d hstruct.1 (by rw [hvlvl, hlvl0])
      by_cases hstop : exhaustedStopAllowedB d lvl φ cursor = true
      · simp [routeFromFlowCursorC, admChainCursorCB, hstop]
      · simp [routeFromFlowCursorC, admChainCursorCB, hstop, hnode, hterm]
  | succ fuel ih =>
      intro lvl φ v b cursor hnode hv hbp hcursor hle
      have hvlvl : v.LEVEL = lvl := (nodeAtLevelFormula?_some hnode).2.1
      by_cases hstop : exhaustedStopAllowedB d lvl φ cursor = true
      · simp [routeFromFlowCursorC, admChainCursorCB, hstop]
      · have hstopFalse : exhaustedStopAllowedB d lvl φ cursor = false :=
          Bool.eq_false_of_not_eq_true hstop
        by_cases hout : get_rule.outgoing v d = []
        · exact admChainCursorCB_stuck d formulas cursor
            (Or.inr ⟨v, hnode, hout⟩) (fuel + 1)
        ·
          rcases (hcoverage v hv hout).1 (b, cursor.residual) hbp with
            hexhausted | hcov
          · exact (hstop (exhaustedStopAllowedB_of_flowArrival d hnode
              hcursor hexhausted)).elim
          · obtain ⟨e, hedge⟩ := edgeOfColour_some_of_cover hcov
            obtain ⟨heE, hestart, hecol⟩ := edgeOfColour_some hedge
            obtain ⟨e0, es, hcons⟩ :
                ∃ e0 es, get_rule.outgoing v d = e0 :: es := by
              cases h : get_rule.outgoing v d with
              | nil => exact absurd h hout
              | cons a l => exact ⟨a, l, rfl⟩
            have hEmem : e.END ∈ d.NODES := (hstruct.2.1 e heE).2
            have hElvl : e.END.LEVEL = lvl - 1 := by
              have := hstruct.1 e heE
              rw [hestart, hvlvl] at this
              omega
            have hEnode :
                nodeAtLevelFormula? d (lvl - 1) e.END.FORMULA = some e.END := by
              have := nodeAtLevelFormula?_self d hlfu hEmem
              rwa [hElvl] at this
            have hein : e ∈ get_rule.outgoing v d :=
              mem_outgoing_of_mem_edges_start_eq v d heE hestart
            by_cases hmin :
                (isMinorArrivalVertexC d e.START e.END &&
                  cursor.residual.tail.isEmpty) = true
            · rw [routeFromFlowCursorC_step d formulas hstopFalse hnode hcons
                    hedge,
                  admChainCursorCB_step d formulas hstopFalse hnode hcons hedge,
                  if_pos hmin]
              simp only [beq_self_eq_true, Bool.and_self, Bool.true_and, hmin,
                if_true]
              exact replicate_all_stops fuel
            · rw [routeFromFlowCursorC_step d formulas hstopFalse hnode hcons
                    hedge,
                  admChainCursorCB_step d formulas hstopFalse hnode hcons hedge,
                  if_neg hmin]
              simp only [beq_self_eq_true, Bool.and_self, Bool.true_and, hmin]
              let next := cursor.afterEdge
              by_cases hnextStop : exhaustedStopAllowedB d (lvl - 1)
                  e.END.FORMULA next = true
              · exact admChainCursorCB_route_of_exhaustedStop d formulas next
                  hnextStop fuel
              · by_cases hEout : get_rule.outgoing e.END d = []
                · exact admChainCursorCB_stuck d formulas next
                    (Or.inr ⟨e.END, hEnode, hEout⟩) fuel
                · have hnotExhausted :
                      ¬ ExhaustedConsumedTailAt d e.END cursor.residual.tail := by
                    rintro ⟨_hout, htail, hedge0⟩
                    have hstop' : exhaustedStopAllowedB d (lvl - 1)
                        e.END.FORMULA next = true := by
                      simp [exhaustedStopAllowedB, next,
                        DLDS.ResidualCursor.afterEdge, htail, hEnode, hedge0]
                    exact hnextStop hstop'
                  obtain ⟨b', hb'⟩ := flowAt_tail_closure d hstruct hhypNoInc
                    hftc hv hbp hein (by simpa using hecol) hEout hnotExhausted
                  have hinc : e ∈ get_rule.incoming e.END d :=
                    outgoing_mem_incoming_end v d hein
                  have hincne : get_rule.incoming e.END d ≠ [] := by
                    intro hempty
                    rw [hempty] at hinc
                    exact List.not_mem_nil hinc
                  have hEhyp : e.END.HYPOTHESIS = false := by
                    by_contra hh
                    have hhtrue : e.END.HYPOTHESIS = true := by
                      cases heq : e.END.HYPOTHESIS with
                      | true => rfl
                      | false => exact absurd heq hh
                    exact hincne (hhypNoInc e.END hEmem hhtrue)
                  have hvpred : v ∈ predsOf d e.END := by
                    rw [predsOf_eq, List.mem_eraseDups, List.mem_map]
                    exact ⟨e, hinc, hestart⟩
                  have hpreds : (predsOf d e.END).isEmpty = false := by
                    cases hempty : (predsOf d e.END).isEmpty with
                    | false => rfl
                    | true =>
                        rw [List.isEmpty_iff] at hempty
                        rw [hempty] at hvpred
                        exact absurd hvpred List.not_mem_nil
                  have hnextCursor :
                      next = flowCursorAt d e.END (b', next.residual) := by
                    simp [next, flowCursorAt, hEhyp, hpreds,
                      DLDS.ResidualCursor.afterEdge]
                  exact ih (lvl - 1) e.END.FORMULA e.END b' next hEnode hEmem
                    (by simpa [next, DLDS.ResidualCursor.afterEdge] using hb')
                    hnextCursor (by omega)

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
 Under the legality conditions and `CursorFlowContinuationClosure`,
    the canonical multi-path passes every slot check (the M2c side condition).
-/
theorem canonicalSlotOK (d : Graph)
    (hstruct : structuralValid d) (hhypNoInc : HypothesesHaveNoIncoming d)
    (hlfu : LevelFormulaUnique d) (hcoverage : CursorRouteHeadCoverage d)
    (hftc : CursorFlowContinuationClosure d) :
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
  | none =>
      have hfc' : (buildFormulas d)[col]? = none := by
        simpa [hf] using hfc
      simp []
  | some φ =>
      simp only []
      have hfc' : (buildFormulas d)[col]? = some φ := by
        simpa [hf] using hfc
      cases hsource : sourceRoute? d col r with
      | none =>
          have htable : (sourceRouteTable d)[col]?.bind
              (fun routes => routes[r]?) = none := by
            simpa [sourceRouteTable, sourceRoute?, hfc'] using hsource
          simp only
          rw [htable]
          exact replicate_all_stops _
      | some source =>
          obtain ⟨v, bp⟩ := source
          have htable : (sourceRouteTable d)[col]?.bind
              (fun routes => routes[r]?) = some (v, bp) := by
            simpa [sourceRouteTable, sourceRoute?, hfc'] using hsource
          simp only
          rw [htable]
          have hprops := sourceRoute?_some_props d hfc' hsource
          have hvmem : v ∈ d.NODES := hprops.1
          have hvform : v.FORMULA = φ := hprops.2.2.1
          have hbpmem : bp ∈ flowAt d (stdFuel d) v := hprops.2.2.2
          set maxLvl := (d.NODES.map (·.LEVEL)).foldl max 0 with hml
          set numSteps := (buildGridFromDLDS d).length - 1 with hns
          have hvle : v.LEVEL ≤ maxLvl := level_le_maxLvl d hvmem
          have hnsml : numSteps = maxLvl := numSteps_eq_maxLvl d
          have hfuel : numSteps - (maxLvl - v.LEVEL) = v.LEVEL := by omega
          rw [admHypColumnCB_paddings]
          rw [hfuel]
          have hnode : nodeAtLevelFormula? d v.LEVEL φ = some v := by
            rw [← hvform]; exact nodeAtLevelFormula?_self d hlfu hvmem
          change admChainCursorCB d formulas v.LEVEL φ
            (DLDS.ResidualCursor.initial bp.2)
            (routeFromFlowCursorC d formulas v.LEVEL v.LEVEL φ
              (DLDS.ResidualCursor.initial bp.2)) = true
          exact routeFromFlowCursorC_admChainCB d hstruct hhypNoInc hlfu
            hcoverage hftc formulas v.LEVEL v.LEVEL φ v bp.1
            (DLDS.ResidualCursor.initial bp.2) hnode hvmem
            (by simpa [DLDS.ResidualCursor.initial] using hbpmem)
            (by simp [flowCursorAt, hprops.2.1, DLDS.ResidualCursor.initial])
            (le_refl _)



/--
 The compressed universal bridge with canonicalSlotOKB derived from
 exact cursor-continuation closure and the legality conditions. The discharge
 certificate dischargedMultiB remains an explicit hypothesis.

 hnodup and honepercol are intentional representation-scope disclaimers, although
 the proof of this internal admissibility/discharge implication does not use them.
 hnodup excludes literal duplicate deduction records: raw classifyRule? counts
 incoming edge records, while Flow deduplicates predecessor vertices. Duplicating
 an introduction edge can therefore erase its raw rule classification, even when
 the concrete-arrival fallback preserves the evaluated result.
 honepercol excludes distinct same-colour outgoing edges at one source:
 edgeOfColour selects the first matching record, so without this condition the
 represented route and admissibility can depend on edge-list order.
 The duplicate-free and one-edge-per-colour hypotheses delimit the intended representation scope, although the conditional discharge proof does not consume their witnesses.
-/
theorem compressed_universally_accepted' (d : Graph)
    (hstruct : structuralValid d)
    (_hnodup : d.EDGES.Nodup)
    (_honepercol : OneEdgePerColourPerNode d)
    (hcoverage : CursorRouteHeadCoverage d)
    (hlfu : LevelFormulaUnique d)
    (hhypNoInc : HypothesesHaveNoIncoming d)
    (hftc : CursorFlowContinuationClosure d)
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





end DLDS
