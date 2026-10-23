import Infrastructure.OccurrenceReadingsModel
import Flow.MultiIntroModel



open Semantic FlowSpec RouteIndexedMultiToken SpineReadings CertificateModel
namespace MultiIntroReadings

def reseedTest (variant : MultiIntro.Variant) (route : FlowPair) (edge : Deduction) : Bool :=
  match variant with
  | .strict => reseedZeroPremiseB route edge
  | .general => reseedPremiseAgreesB route edge

def payload? (variant : MultiIntro.Variant) (d : Graph) : ReadingTree → Option FlowPair
  | .source v seed =>
      if sourceB d v then
        match seed with
        | none => if (ancestorsInto d v).isEmpty then some ([v.FORMULA], []) else none
        | some a => if (ancestorsInto d v).contains a then some ([v.FORMULA], a.COLOURS) else none
      else none
  | .certificate _ _ | .repetition _ _ _ => none
  | .slot v _ parent =>
      if v == vertex parent then payload? variant d parent else none
  | .introduction v e parent => do
      let p ← payload? variant d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && (elimPairsAt d v).isEmpty &&
              (MultiIntro.multiIntroShapeB d v || predsOf d v == [vertex parent]) && (vertex parent).FORMULA == beta &&
              edgeB d (vertex parent) v e && SingleIntroductionReference.guard d p.1 e then
            let tail ← consume e p.2
            some (depRemove p.1 alpha, tail)
          else none
  | .mixedIntroduction v e parent => do
      let p ← payload? variant d parent
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
      let pm ← payload? variant d km
      let pM ← payload? variant d kM
      if !sourceB d v && vertex km == minor && vertex kM == major &&
          (elimPairsAt d v).contains (major, minor) &&
          edgeB d minor v em && edgeB d major v eM &&
          SingleIntroductionReference.guard d pm.1 em && SingleIntroductionReference.guard d pM.1 eM then
        let tm ← consume em pm.2
        let tM ← consume eM pM.2
        if tm == tM then some (depUnion pm.1 pM.1, tm) else none
      else none
  | .introReseed v e a parent => do
      let p ← payload? variant d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && (MultiIntro.multiIntroShapeB d v || predsOf d v == [vertex parent]) &&
              (vertex parent).FORMULA == beta && edgeB d (vertex parent) v e &&
              (if MultiIntro.multiIntroShapeB d v then reseedTest variant p e else reseedZeroPremiseB p e) && (eligibleAncestorsAtRoot d v).contains a &&
              SingleIntroductionReference.guard d p.1 e then
            some (depRemove p.1 alpha, a.COLOURS)
          else none
  | .elimReseed v major minor em eM a km kM => do
      let pm ← payload? variant d km
      let pM ← payload? variant d kM
      if !sourceB d v && vertex km == minor && vertex kM == major &&
          (elimPairsAt d v).contains (major, minor) && edgeB d minor v em &&
          edgeB d major v eM && reseedPremiseAgreesB pm em &&
          reseedPremiseAgreesB pM eM && (eligibleAncestorsAtRoot d v).contains a &&
          SingleIntroductionReference.guard d pm.1 em && SingleIntroductionReference.guard d pM.1 eM then
        some (depUnion pm.1 pM.1, a.COLOURS)
      else none


def occurrenceAt (variant : MultiIntro.Variant) (d : Graph) (v : Vertex) (tree : ReadingTree) : Bool :=
  decide (vertex tree = v) && (payload? variant d tree).isSome && ndCorrectB tree


def verdict (variant : MultiIntro.Variant) (d : Graph) : Prop :=
  (∀ v ∈ d.NODES, (get_rule.outgoing v d).isEmpty = true →
    ∃ tree, occurrenceAt variant d v tree = true) ∧
  ∀ v ∈ d.NODES, (get_rule.outgoing v d).isEmpty = true →
    ∀ tree route, occurrenceAt variant d v tree = true →
      payload? variant d tree = some route → route.1 = [] ∧ route.2 = []

end MultiIntroReadings
