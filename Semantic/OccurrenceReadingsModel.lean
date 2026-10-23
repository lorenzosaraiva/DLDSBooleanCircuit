import Semantic.SingleIntroductionReference
import Semantic.SpineReadingModel





open Semantic FlowSpec RouteIndexedMultiToken SpineReadings
open SelfCertifyingProbe
namespace OccurrenceReadings

def readingPayloadComplete? (d : Graph) : ReadingTree → Option FlowPair
  | .source v seed =>
      if sourceB d v then
        match seed with
        | none => if (ancestorsInto d v).isEmpty then some ([v.FORMULA], []) else none
        | some a => if (ancestorsInto d v).contains a then some ([v.FORMULA], a.COLOURS) else none
      else none
  | .certificate _ _ | .repetition _ _ _ => none
  | .slot v _ parent =>
      if v == vertex parent then readingPayloadComplete? d parent else none
  | .introduction v e parent => do
      let p ← readingPayloadComplete? d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && (elimPairsAt d v).isEmpty &&
              predsOf d v == [vertex parent] && (vertex parent).FORMULA == beta &&
              edgeB d (vertex parent) v e && SingleIntroductionReference.guard d p.1 e then
            let tail ← consume e p.2
            some (depRemove p.1 alpha, tail)
          else none
  | .mixedIntroduction v e parent => do
      let p ← readingPayloadComplete? d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && !(elimPairsAt d v).isEmpty &&
              (predsOf d v).contains (vertex parent) &&
              !elimPremiseAtB d v (vertex parent) &&
              (vertex parent).FORMULA == beta && edgeB d (vertex parent) v e && SingleIntroductionReference.guard d p.1 e then
            let tail ← consume e p.2
            some (depRemove p.1 alpha, tail)
          else none
  | .elimination v major minor em eM km kM => do
      let pm ← readingPayloadComplete? d km
      let pM ← readingPayloadComplete? d kM
      if !sourceB d v && vertex km == minor && vertex kM == major &&
          (elimPairsAt d v).contains (major, minor) &&
          edgeB d minor v em && edgeB d major v eM &&
          SingleIntroductionReference.guard d pm.1 em && SingleIntroductionReference.guard d pM.1 eM then
        let tm ← consume em pm.2
        let tM ← consume eM pM.2
        if tm == tM then some (depUnion pm.1 pM.1, tm) else none
      else none
  | .introReseed v e a parent => do
      let p ← readingPayloadComplete? d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && predsOf d v == [vertex parent] &&
              (vertex parent).FORMULA == beta && edgeB d (vertex parent) v e &&
              reseedZeroPremiseB p e && (eligibleAncestorsAtRoot d v).contains a &&
              SingleIntroductionReference.guard d p.1 e then
            some (depRemove p.1 alpha, a.COLOURS)
          else none
  | .elimReseed v major minor em eM a km kM => do
      let pm ← readingPayloadComplete? d km
      let pM ← readingPayloadComplete? d kM
      if !sourceB d v && vertex km == minor && vertex kM == major &&
          (elimPairsAt d v).contains (major, minor) && edgeB d minor v em &&
          edgeB d major v eM && reseedPremiseAgreesB pm em &&
          reseedPremiseAgreesB pM eM && (eligibleAncestorsAtRoot d v).contains a &&
          SingleIntroductionReference.guard d pm.1 em && SingleIntroductionReference.guard d pM.1 eM then
        some (depUnion pm.1 pM.1, a.COLOURS)
      else none


def occurrenceAtCompleteB (d : Graph) (v : Vertex) (tree : ReadingTree) : Bool :=
  decide (vertex tree = v) && (readingPayloadComplete? d tree).isSome && ndCorrectB tree


def occurrenceVerdictComplete (d : Graph) : Prop :=
  (∀ v ∈ d.NODES, (get_rule.outgoing v d).isEmpty = true →
    ∃ tree, occurrenceAtCompleteB d v tree = true) ∧
  ∀ v ∈ d.NODES, (get_rule.outgoing v d).isEmpty = true →
    ∀ tree route, occurrenceAtCompleteB d v tree = true →
      readingPayloadComplete? d tree = some route → route.1 = [] ∧ route.2 = []

theorem occurrence_payload (d : Graph) (v : Vertex) (tree : ReadingTree)
    (checked : occurrenceAtCompleteB d v tree = true) :
    ∃ route, readingPayloadComplete? d tree = some route := by
  simp only [occurrenceAtCompleteB, Bool.and_eq_true] at checked
  cases hp : readingPayloadComplete? d tree with
  | none => simp [hp] at checked
  | some route => exact ⟨route, rfl⟩

#print axioms occurrence_payload
end OccurrenceReadings
