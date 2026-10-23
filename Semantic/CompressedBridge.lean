import Semantic.DLDS

open scoped Classical

namespace Semantic




def LevelFormulaUnique (d : Graph) : Prop :=
  forall indexedVertexOne, indexedVertexOne ∈ d.NODES ->
  forall indexedVertexTwo, indexedVertexTwo ∈ d.NODES ->
    indexedVertexOne.LEVEL = indexedVertexTwo.LEVEL -> indexedVertexOne.FORMULA = indexedVertexTwo.FORMULA -> indexedVertexOne = indexedVertexTwo

def levelFormulaUniqueB (d : Graph) : Bool :=
  d.NODES.all fun indexedVertexOne =>
    d.NODES.all fun indexedVertexTwo =>
      if indexedVertexOne.LEVEL == indexedVertexTwo.LEVEL then
        if indexedVertexOne.FORMULA == indexedVertexTwo.FORMULA then
          decide (indexedVertexOne = indexedVertexTwo)
        else true
      else true


def nodeAtLevelFormula? (d : Graph) (lvl : Nat) (phi : Formula) : Option Vertex :=
  d.NODES.find? fun v => decide (v.LEVEL = lvl ∧ v.FORMULA = phi)

def sourceNodeAtLevelColumn? (d : Graph) (lvl col : Nat) : Option Vertex :=
  match (buildFormulas d)[col]? with
  | none => none
  | some phi => nodeAtLevelFormula? d lvl phi

def rootNodeAtLevel? (d : Graph) (lvl : Nat) : Option Vertex :=
  d.NODES.find? fun v => decide (v.LEVEL = lvl ∧ (get_rule.outgoing v d).isEmpty)

def goalColumnAtLevel? (d : Graph) (lvl : Nat) : Option Nat :=
  match rootNodeAtLevel? d lvl with
  | none => none
  | some r => some ((buildFormulas d).idxOf r.FORMULA)

lemma nodeAtLevelFormula?_some {d : Graph} {lvl : Nat} {phi : Formula}
    {v : Vertex} (h : nodeAtLevelFormula? d lvl phi = some v) :
    v ∈ d.NODES ∧ v.LEVEL = lvl ∧ v.FORMULA = phi := by
  unfold nodeAtLevelFormula? at h
  have hmem : v ∈ d.NODES := find?_some_mem h
  have hpred := List.find?_some h
  have hmatch : v.LEVEL = lvl ∧ v.FORMULA = phi := of_decide_eq_true hpred
  exact ⟨hmem, hmatch.1, hmatch.2⟩


theorem levelFormulaUnique_eq {d : Graph} (huniq : LevelFormulaUnique d)
    {lvl : Nat} {phi : Formula} {indexedVertexOne indexedVertexTwo : Vertex}
    (hv1 : indexedVertexOne ∈ d.NODES) (hv1_level : indexedVertexOne.LEVEL = lvl)
    (hv1_formula : indexedVertexOne.FORMULA = phi)
    (hv2 : indexedVertexTwo ∈ d.NODES) (hv2_level : indexedVertexTwo.LEVEL = lvl)
    (hv2_formula : indexedVertexTwo.FORMULA = phi) :
    indexedVertexOne = indexedVertexTwo := by
  exact huniq indexedVertexOne hv1 indexedVertexTwo hv2
    (hv1_level.trans hv2_level.symm)
    (hv1_formula.trans hv2_formula.symm)

theorem nodeAtLevelFormula?_unique {d : Graph} (huniq : LevelFormulaUnique d)
    {lvl : Nat} {phi : Formula} {indexedVertexOne indexedVertexTwo : Vertex}
    (h1 : nodeAtLevelFormula? d lvl phi = some indexedVertexOne)
    (h2 : nodeAtLevelFormula? d lvl phi = some indexedVertexTwo) :
    indexedVertexOne = indexedVertexTwo := by
  have p1 := nodeAtLevelFormula?_some (d := d) (lvl := lvl) (phi := phi) h1
  have p2 := nodeAtLevelFormula?_some (d := d) (lvl := lvl) (phi := phi) h2
  exact levelFormulaUnique_eq huniq p1.1 p1.2.1 p1.2.2 p2.1 p2.2.1 p2.2.2

end Semantic
