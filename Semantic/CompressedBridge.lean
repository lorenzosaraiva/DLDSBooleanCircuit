import Semantic.DLDS

open scoped Classical

namespace Semantic

/-! Level-and-formula lookup infrastructure for compressed DLDS graphs. -/

/-- Thm 24 construction invariant: -/
def LevelFormulaUnique (d : Graph) : Prop :=
  forall v1, v1 ∈ d.NODES ->
  forall v2, v2 ∈ d.NODES ->
    v1.LEVEL = v2.LEVEL -> v1.FORMULA = v2.FORMULA -> v1 = v2

def levelFormulaUniqueB (d : Graph) : Bool :=
  d.NODES.all fun v1 =>
    d.NODES.all fun v2 =>
      if v1.LEVEL == v2.LEVEL then
        if v1.FORMULA == v2.FORMULA then
          decide (v1 = v2)
        else true
      else true

/-- Compressed-graph lookup by level and formula. -/
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

/-- Under LevelFormulaUnique, any two nodes matching the same (level, formula) key are equal. -/
theorem levelFormulaUnique_eq {d : Graph} (huniq : LevelFormulaUnique d)
    {lvl : Nat} {phi : Formula} {v1 v2 : Vertex}
    (hv1 : v1 ∈ d.NODES) (hv1_level : v1.LEVEL = lvl)
    (hv1_formula : v1.FORMULA = phi)
    (hv2 : v2 ∈ d.NODES) (hv2_level : v2.LEVEL = lvl)
    (hv2_formula : v2.FORMULA = phi) :
    v1 = v2 := by
  exact huniq v1 hv1 v2 hv2
    (hv1_level.trans hv2_level.symm)
    (hv1_formula.trans hv2_formula.symm)

theorem nodeAtLevelFormula?_unique {d : Graph} (huniq : LevelFormulaUnique d)
    {lvl : Nat} {phi : Formula} {v1 v2 : Vertex}
    (h1 : nodeAtLevelFormula? d lvl phi = some v1)
    (h2 : nodeAtLevelFormula? d lvl phi = some v2) :
    v1 = v2 := by
  have p1 := nodeAtLevelFormula?_some (d := d) (lvl := lvl) (phi := phi) h1
  have p2 := nodeAtLevelFormula?_some (d := d) (lvl := lvl) (phi := phi) h2
  exact levelFormulaUnique_eq huniq p1.1 p1.2.1 p1.2.2 p2.1 p2.2.1 p2.2.2

end Semantic
