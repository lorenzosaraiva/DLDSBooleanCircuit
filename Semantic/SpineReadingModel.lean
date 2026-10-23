import Semantic.FlowBranchingGridBridge
import Semantic.LambdaFlowModel








open Semantic FlowSpec RouteIndexedMultiToken FlowBranchingGridBridge
open GridCertificateSoundness SelfCertifyingProbe

namespace SpineReadings



abbrev ReadingTree := RouteKey

def vertex : ReadingTree → Vertex
  | .source v _ | .certificate v _ | .slot v _ _ | .repetition v _ _
  | .introduction v _ _ | .mixedIntroduction v _ _ | .elimination v _ _ _ _ _ _
  | .introReseed v _ _ _ | .elimReseed v _ _ _ _ _ _ _ => v



def readingDep (tree : ReadingTree) : Dep :=
  match routeKeyTree? tree with
  | some nd => normalizeDep nd.dependencies
  | none => []

def sourceB (d : Graph) (v : Vertex) : Bool :=
  v.HYPOTHESIS || (predsOf d v).isEmpty

def edgeB (d : Graph) (u v : Vertex) (e : Deduction) : Bool :=
  (edgesBetween d u v).contains e




def readingPayload? (d : Graph) : ReadingTree → Option FlowPair
  | .source v seed =>
      if sourceB d v then
        match seed with
        | none => if (ancestorsInto d v).isEmpty then some ([v.FORMULA], []) else none
        | some a => if (ancestorsInto d v).contains a then some ([v.FORMULA], a.COLOURS) else none
      else none
  | .certificate _ _ | .repetition _ _ _ => none
  | .slot v _ parent =>
      if v == vertex parent then readingPayload? d parent else none
  | .introduction v e parent => do
      let p ← readingPayload? d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && (elimPairsAt d v).isEmpty &&
              predsOf d v == [vertex parent] && (vertex parent).FORMULA == beta &&
              edgeB d (vertex parent) v e then
            let tail ← consume e p.2
            some (depRemove p.1 alpha, tail)
          else none
  | .mixedIntroduction v e parent => do
      let p ← readingPayload? d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && !(elimPairsAt d v).isEmpty &&
              (predsOf d v).contains (vertex parent) &&
              !elimPremiseAtB d v (vertex parent) &&
              (vertex parent).FORMULA == beta && edgeB d (vertex parent) v e then
            let tail ← consume e p.2
            some (depRemove p.1 alpha, tail)
          else none
  | .elimination v major minor em eM km kM => do
      let pm ← readingPayload? d km
      let pM ← readingPayload? d kM
      if !sourceB d v && vertex km == minor && vertex kM == major &&
          (elimPairsAt d v).contains (major, minor) &&
          edgeB d minor v em && edgeB d major v eM then
        let tm ← consume em pm.2
        let tM ← consume eM pM.2
        if tm == tM then some (depUnion pm.1 pM.1, tm) else none
      else none
  | .introReseed v e a parent => do
      let p ← readingPayload? d parent
      match v.FORMULA with
      | .atom _ => none
      | .implication alpha beta =>
          if !sourceB d v && predsOf d v == [vertex parent] &&
              (vertex parent).FORMULA == beta && edgeB d (vertex parent) v e &&
              reseedZeroPremiseB p e && (eligibleAncestorsAtRoot d v).contains a then
            some (depRemove p.1 alpha, a.COLOURS)
          else none
  | .elimReseed v major minor em eM a km kM => do
      let pm ← readingPayload? d km
      let pM ← readingPayload? d kM
      if !sourceB d v && vertex km == minor && vertex kM == major &&
          (elimPairsAt d v).contains (major, minor) && edgeB d minor v em &&
          edgeB d major v eM && reseedPremiseAgreesB pm em &&
          reseedPremiseAgreesB pM eM && (eligibleAncestorsAtRoot d v).contains a then
        some (depUnion pm.1 pM.1, a.COLOURS)
      else none


def activeEdges : ReadingTree → List (Deduction × ReadingTree)
  | .source _ _ | .certificate _ _ => []
  | .slot _ _ parent => activeEdges parent
  | .repetition _ e parent | .introduction _ e parent
  | .mixedIntroduction _ e parent | .introReseed _ e _ parent =>
      (e, parent) :: activeEdges parent
  | .elimination _ _ _ em eM km kM | .elimReseed _ _ _ em eM _ km kM =>
      (em, km) :: (eM, kM) :: (activeEdges km ++ activeEdges kM)


def subtrees (tree : ReadingTree) : List ReadingTree :=
  tree :: match tree with
  | .source _ _ | .certificate _ _ => []
  | .slot _ _ parent | .repetition _ _ parent | .introduction _ _ parent
  | .mixedIntroduction _ _ parent | .introReseed _ _ _ parent => subtrees parent
  | .elimination _ _ _ _ _ km kM | .elimReseed _ _ _ _ _ _ km kM =>
      subtrees km ++ subtrees kM



def fanCoherentB (tree : ReadingTree) : Bool :=
  (activeEdges tree).all fun left => (activeEdges tree).all fun right =>
    left.1.START != right.1.START || left.1 == right.1

def ndCorrectB (tree : ReadingTree) : Bool :=
  match routeKeyTree? tree with
  | none => false
  | some nd => nd.wellFormedB && decide (nd.conclusion = (vertex tree).FORMULA)




def colourCoherentB (d : Graph) (tree : ReadingTree) : Bool :=
  d.NODES.contains (vertex tree) && (get_rule.outgoing (vertex tree) d).isEmpty &&
  (readingPayload? d tree).isSome && fanCoherentB tree && ndCorrectB tree



def readingLabelDefectB (lg : LambdaFlow.LGraph) (tree : ReadingTree) : Bool :=
  (activeEdges tree).any fun (selectedEdge, child) =>
    !LambdaFlow.labelAgreesB (lg.label selectedEdge) (readingDep child)



def readingShapeDefectB (d : Graph) (tree : ReadingTree) : Bool :=
  (subtrees tree).any fun child =>
    match localShape d (vertex child) with
    | .invalid => true
    | _ => false



def readingMarkedB (lg : LambdaFlow.LGraph) (tree : ReadingTree) : Bool :=
  colourCoherentB lg.graph tree &&
    (!(readingDep tree).isEmpty || readingLabelDefectB lg tree ||
      readingShapeDefectB lg.graph tree)




def ReadingRealizable (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, ∀ route ∈ LambdaFlow.guardFreeFlowAt d (stdFuel d) v,
    ∃ tree child, colourCoherentB d tree = true ∧ child ∈ subtrees tree ∧
      vertex child = v ∧ readingPayload? d child = some route

theorem unmarked_colourCoherent_closedND (lg : LambdaFlow.LGraph) (tree : ReadingTree)
    (coherent : colourCoherentB lg.graph tree = true)
    (unmarked : readingMarkedB lg tree = false) :
    Derives [] (vertex tree).FORMULA := by
  have correct : ndCorrectB tree = true := by
    unfold colourCoherentB at coherent
    simp only [Bool.and_eq_true] at coherent
    exact coherent.2
  have empty : (readingDep tree).isEmpty = true := by
    simp only [readingMarkedB, coherent, Bool.true_and, Bool.or_eq_false_iff,
      Bool.not_eq_false'] at unmarked
    exact unmarked.1.1
  cases ht : routeKeyTree? tree with
  | none => simp [ndCorrectB, ht] at correct
  | some nd =>
      have shaped : nd.wellFormedB = true ∧ nd.conclusion = (vertex tree).FORMULA := by
        simpa [ndCorrectB, ht] using correct
      have depEmpty : normalizeDep nd.dependencies = [] := by
        simpa [readingDep, ht, List.isEmpty_iff] using empty
      have rawEmpty : nd.dependencies = [] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro f hf
        have hn := (mem_normalizeDep (formula := f) (dependency := nd.dependencies)).mpr hf
        rw [depEmpty] at hn
        exact List.not_mem_nil hn
      have derivation := NDTree.derives_dependencies nd shaped.1
      simpa [rawEmpty, shaped.2] using derivation

#print axioms unmarked_colourCoherent_closedND
end SpineReadings
