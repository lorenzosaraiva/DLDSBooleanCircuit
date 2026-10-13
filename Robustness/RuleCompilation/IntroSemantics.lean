import Robustness.SelectedProducers.OrdinaryComponents
import Robustness.SelectedProducers.SeedDensity

/-! Fixed graph, ownership and producer/premise addresses; only canonical route
bitmaps and fixed-length colour words vary. Padding coordinates are ignored. -/
open DLDS FlowSemantics Robustness.BitLocality Robustness.SelectedProducers
namespace Robustness.RuleCompilation
set_option maxHeartbeats 2000000

abbrev RuleCoord {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) := SeedCoord shape size

def ruleRoutes {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) (bits : RuleCoord shape size → Bool) : Fin routes → FlowPair :=
  decodeRoutes shape (fun coordinate => bits (.inr coordinate))

/-- Fixed non-source introduction branch and the selected incoming edge.
All fields, including lengths in RouteShape, remain fixed in the Boolean product. -/
structure IntroFrame {routes : Nat} (d : Graph) where
  owner : Fin routes → Vertex
  listed : ∀ slot, owner slot ∈ d.NODES
  output : Fin routes
  input : Fin routes
  alpha : Formula
  beta : Formula
  edgeIndex : Nat
  incoming : Deduction
  nonhyp : (owner output).HYPOTHESIS = false
  predecessors : predsOf d (owner output) = [owner input]
  noElimination : elimPairsAt d (owner output) = []
  conclusion : (owner output).FORMULA = .implication alpha beta
  premiseFormula : (owner input).FORMULA = beta
  edgeSelected : (edgesBetween d (owner input) (owner output))[edgeIndex]? = some incoming

def introCertificate {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) {d : Graph} (frame : IntroFrame (routes := routes) d)
    (bits : RuleCoord shape size → Bool) : JustifiedCertificate d routes where
  owner := frame.owner
  ownerListed := frame.listed
  route := ruleRoutes shape size bits
  kind _ := .step
  premise _ pointer := if pointer = 0 then some frame.input else none

def introAcceptance {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) {d : Graph} (frame : IntroFrame (routes := routes) d)
    (bits : RuleCoord shape size → Bool) : Bool :=
  selectedJustificationB (introCertificate shape size frame bits) frame.output
    (.introduction 0 frame.edgeIndex)

def introFailure {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) {d : Graph} (frame : IntroFrame (routes := routes) d)
    (bits : RuleCoord shape size → Bool) : Bool := !introAcceptance shape size frame bits

theorem intro_acceptance_eq {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size : Nat) {d : Graph} (frame : IntroFrame (routes := routes) d)
    (bits : RuleCoord shape size → Bool) :
    introAcceptance shape size frame bits =
      (decide ((ruleRoutes shape size bits frame.output).1 =
        depRemove (ruleRoutes shape size bits frame.input).1 frame.alpha) &&
      decide (consume frame.incoming (ruleRoutes shape size bits frame.input).2 =
        some (ruleRoutes shape size bits frame.output).2) &&
      routeContinuableAtB d (frame.owner frame.output)
        (ruleRoutes shape size bits frame.output).2) := by
  have slots : pointedSlots (introCertificate shape size frame bits) frame.output =
      [frame.input] := by
    simp [pointedSlots, introCertificate, List.finRange_succ]
  have inputRoutes : pointedRoutes (introCertificate shape size frame bits) frame.output
      (frame.owner frame.input) = [ruleRoutes shape size bits frame.input] := by
    rw [pointedRoutes, slots]
    simp [introCertificate]
  dsimp only [introCertificate] at slots inputRoutes
  unfold introAcceptance selectedJustificationB
  simp only [introCertificate, slots, frame.nonhyp, frame.predecessors,
    List.isEmpty_cons, Bool.false_or, Bool.not_false, List.all_cons, List.all_nil,
    List.mem_singleton, decide_true, Bool.and_true, Bool.true_and,
    WitnessableRoutes, List.mem_filter, Bool.decide_and]
  simp only [selectedRows, frame.nonhyp, frame.predecessors, List.isEmpty_cons,
    Bool.false_or, Bool.false_eq_true, if_false, frame.noElimination,
    List.isEmpty_nil, if_true, chosenIntro, frame.conclusion, frame.premiseFormula,
    inputRoutes, pick, frame.edgeSelected, List.getElem?_cons_zero, Option.toList_some,
    List.flatMap_cons, List.flatMap_nil, List.append_nil, List.filterMap_cons,
    List.filterMap_nil]
  cases consumed : consume frame.incoming (ruleRoutes shape size bits frame.input).2 with
  | none => simp
  | some tail =>
    simp only [Option.map_some, List.mem_singleton, Option.some.injEq]
    apply Bool.eq_iff_iff.mpr
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    simp only [Prod.ext_iff]
    constructor
    · rintro ⟨⟨hd, ht⟩, hw⟩; exact ⟨⟨hd, ht.symm⟩, hw⟩
    · rintro ⟨⟨hd, ht⟩, hw⟩; exact ⟨⟨hd, ht.symm⟩, hw⟩

end Robustness.RuleCompilation
