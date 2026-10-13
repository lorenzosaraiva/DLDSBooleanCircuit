import Robustness.RuleCompilation.IntroDensity

/-! Ordinary elimination with fixed pair/edge indices and two fixed pointers.
Distinct owners and predecessor fidelity follow from the actual selected pair. -/
open DLDS FlowSemantics Robustness.BitLocality Robustness.SelectedProducers
namespace Robustness.RuleCompilation
set_option maxHeartbeats 2000000

structure ElimFrame {routes : Nat} (d : Graph) where
  owner : Fin routes → Vertex
  listed : ∀ slot, owner slot ∈ d.NODES
  output : Fin routes
  minor : Fin routes
  major : Fin routes
  pairIndex : Nat
  minorEdgeIndex : Nat
  majorEdgeIndex : Nat
  minorEdge : Deduction
  majorEdge : Deduction
  nonhyp : (owner output).HYPOTHESIS = false
  pairSelected : (elimPairsAt d (owner output))[pairIndex]? =
    some (owner major, owner minor)
  minorEdgeSelected : (edgesBetween d (owner minor) (owner output))[minorEdgeIndex]? =
    some minorEdge
  majorEdgeSelected : (edgesBetween d (owner major) (owner output))[majorEdgeIndex]? =
    some majorEdge

theorem ElimFrame.pair_member {r : Nat} {d : Graph} (frame : ElimFrame (routes := r) d) :
    (frame.owner frame.major, frame.owner frame.minor) ∈
      elimPairsAt d (frame.owner frame.output) :=
  List.mem_of_getElem? frame.pairSelected

theorem ElimFrame.predecessors {r : Nat} {d : Graph} (frame : ElimFrame (routes := r) d) :
    frame.owner frame.major ∈ predsOf d (frame.owner frame.output) ∧
      frame.owner frame.minor ∈ predsOf d (frame.owner frame.output) :=
  mem_preds_of_mem_elimPairsAt frame.pair_member

theorem ElimFrame.distinct {r : Nat} {d : Graph} (frame : ElimFrame (routes := r) d) :
    frame.owner frame.major ≠ frame.owner frame.minor := by
  have member := frame.pair_member
  unfold elimPairsAt at member
  obtain ⟨major, _, member⟩ := List.mem_flatMap.mp member
  obtain ⟨minor, _, value⟩ := List.mem_filterMap.mp member
  split at value
  · rename_i formula
    simp only [Option.some.injEq, Prod.mk.injEq] at value
    rcases value with ⟨majorEq, minorEq⟩
    subst major; subst minor
    intro same
    rw [same] at formula
    have sizes := congrArg sizeOf formula
    simp at sizes
    omega
  · simp at value

def elimCertificate {r f : Nat} (shape : RouteShape r f) (size : Nat)
    {d : Graph} (frame : ElimFrame (routes := r) d) (bits : RuleCoord shape size → Bool) :
    JustifiedCertificate d r where
  owner := frame.owner
  ownerListed := frame.listed
  route := ruleRoutes shape size bits
  kind _ := .step
  premise _ pointer := some (if pointer = 0 then frame.minor else frame.major)

def elimAcceptance {r f : Nat} (shape : RouteShape r f) (size : Nat)
    {d : Graph} (frame : ElimFrame (routes := r) d) (bits : RuleCoord shape size → Bool) : Bool :=
  selectedJustificationB (elimCertificate shape size frame bits) frame.output
    (.elimination frame.pairIndex 0 0 frame.minorEdgeIndex frame.majorEdgeIndex)

def elimFailure {r f : Nat} (shape : RouteShape r f) (size : Nat)
    {d : Graph} (frame : ElimFrame (routes := r) d) (bits : RuleCoord shape size → Bool) : Bool :=
  !elimAcceptance shape size frame bits

theorem elim_acceptance_eq {r f : Nat} (shape : RouteShape r f) (size : Nat)
    {d : Graph} (frame : ElimFrame (routes := r) d) (bits : RuleCoord shape size → Bool) :
    elimAcceptance shape size frame bits =
      (decide ((ruleRoutes shape size bits frame.output).1 =
        depUnion (ruleRoutes shape size bits frame.minor).1
          (ruleRoutes shape size bits frame.major).1) &&
      decide (consume frame.minorEdge (ruleRoutes shape size bits frame.minor).2 =
        some (ruleRoutes shape size bits frame.output).2) &&
      decide (consume frame.majorEdge (ruleRoutes shape size bits frame.major).2 =
        some (ruleRoutes shape size bits frame.output).2) &&
      routeContinuableAtB d (frame.owner frame.output)
        (ruleRoutes shape size bits frame.output).2) := by
  have slots : pointedSlots (elimCertificate shape size frame bits) frame.output =
      [frame.minor, frame.major] := by
    simp [pointedSlots, elimCertificate, List.finRange_succ]
  have minorRoutes : pointedRoutes (elimCertificate shape size frame bits) frame.output
      (frame.owner frame.minor) = [ruleRoutes shape size bits frame.minor] := by
    rw [pointedRoutes, slots]
    simp [elimCertificate, frame.distinct]
  have majorRoutes : pointedRoutes (elimCertificate shape size frame bits) frame.output
      (frame.owner frame.major) = [ruleRoutes shape size bits frame.major] := by
    rw [pointedRoutes, slots]
    simp [elimCertificate, Ne.symm frame.distinct]
  have pairsNotEmpty : (elimPairsAt d (frame.owner frame.output)).isEmpty = false := by
    apply List.isEmpty_eq_false_iff.mpr
    intro empty
    simpa [empty] using frame.pair_member
  have predsNotEmpty : (predsOf d (frame.owner frame.output)).isEmpty = false := by
    apply List.isEmpty_eq_false_iff.mpr
    intro empty
    simpa [empty] using frame.predecessors.1
  dsimp only [elimCertificate] at slots minorRoutes majorRoutes
  unfold elimAcceptance selectedJustificationB
  simp only [elimCertificate, slots, frame.nonhyp, predsNotEmpty, Bool.false_or,
    Bool.not_false, List.all_cons, List.all_nil, frame.predecessors.1,
    frame.predecessors.2, decide_true, Bool.and_true, Bool.true_and,
    WitnessableRoutes, List.mem_filter, Bool.decide_and]
  simp only [selectedRows, frame.nonhyp, predsNotEmpty, Bool.false_or, Bool.false_eq_true,
    if_false, pairsNotEmpty, chosenElim, pick, frame.pairSelected, Option.toList_some,
    List.flatMap_cons, List.flatMap_nil, List.append_nil, minorRoutes, majorRoutes,
    List.getElem?_cons_zero, frame.minorEdgeSelected, frame.majorEdgeSelected,
    List.filterMap_cons, List.filterMap_nil]
  cases minorConsumed : consume frame.minorEdge (ruleRoutes shape size bits frame.minor).2 with
  | none => simp
  | some minorTail =>
    cases majorConsumed : consume frame.majorEdge (ruleRoutes shape size bits frame.major).2 with
    | none => simp
    | some majorTail =>
      by_cases sameTail : minorTail = majorTail
      · subst majorTail
        simp only [if_true, List.mem_singleton, Option.some.injEq]
        apply Bool.eq_iff_iff.mpr
        simp only [Bool.and_eq_true, decide_eq_true_eq, Prod.ext_iff]
        constructor
        · rintro ⟨⟨hd, ht⟩, hw⟩
          exact ⟨⟨⟨hd, ht.symm⟩, ht.symm⟩, hw⟩
        · rintro ⟨⟨⟨hd, ht⟩, _⟩, hw⟩
          exact ⟨⟨hd, ht.symm⟩, hw⟩
      · simp only [if_neg sameTail, List.not_mem_nil, decide_false,
          Bool.false_and, Option.some.injEq]
        apply Bool.eq_iff_iff.mpr
        simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.false_eq_true]
        constructor
        · intro impossible; contradiction
        · rintro ⟨⟨⟨_, hm⟩, ha⟩, _⟩
          exact sameTail (hm.trans ha.symm)

end Robustness.RuleCompilation
