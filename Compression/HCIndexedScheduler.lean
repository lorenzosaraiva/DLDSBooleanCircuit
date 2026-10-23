import Mathlib.Data.List.Basic
import Mathlib.Tactic
import Foundations.Graph









open scoped Classical

namespace HCIndexedScheduler

def levelFormulaEqB (level : Nat) (label : Formula) (node : Vertex) : Bool :=
  level == node.LEVEL && label == node.FORMULA

@[simp] theorem decide_eq_beq_nat (left right : Nat) :
    decide (left = right) = (left == right) := by
  by_cases equal : left = right <;> simp [equal]

@[simp] theorem decide_eq_beq_formula (left right : Formula) :
    decide (left = right) = (left == right) := by
  by_cases equal : left = right <;> simp [equal]

@[simp] theorem hcClassKeyEqB_eq_true_iff (left right : Vertex) :
    hcClassKeyEqB left right = true ↔
      left.LEVEL = right.LEVEL ∧ left.FORMULA = right.FORMULA := by
  simp [hcClassKeyEqB]

@[simp] theorem levelFormulaEqB_eq_true_iff
    (level : Nat) (label : Formula) (node : Vertex) :
    levelFormulaEqB level label node = true ↔
      level = node.LEVEL ∧ label = node.FORMULA := by
  simp [levelFormulaEqB]

@[simp] theorem loop₃_eq_any (node : Vertex) (nodes : List Vertex) :
    repeated_nodes_vertex.loop₃ node nodes =
      nodes.any (hcClassKeyEqB node) := by
  induction nodes with
  | nil => rfl
  | cons head tail ih =>
      simp [repeated_nodes_vertex.loop₃, hcClassKeyEqB, ih]

theorem loop₃_eq_any_levelFormulaEqB
    (level : Nat) (label : Formula) (node : Vertex) (nodes : List Vertex)
    (nodeMatch : levelFormulaEqB level label node = true) :
    repeated_nodes_vertex.loop₃ node nodes =
      nodes.any (levelFormulaEqB level label) := by
  obtain ⟨levelEq, formulaEq⟩ :=
    (levelFormulaEqB_eq_true_iff level label node).mp nodeMatch
  rw [loop₃_eq_any]
  induction nodes with
  | nil => rfl
  | cons other rest ih =>
      simp [hcClassKeyEqB, levelFormulaEqB, levelEq, formulaEq, ih]

theorem any_eq_true_iff_countP_pos {α : Type} (predicate : α → Bool)
    (items : List α) :
    items.any predicate = true ↔ 0 < items.countP predicate := by
  induction items with
  | nil => simp
  | cons head tail ih =>
      by_cases selected : predicate head = true <;>
        simp [selected, ih]

theorem loop₂_eq_filter_or_empty (level : Nat) (label : Formula)
    (before after : List Vertex) :
    repeated_nodes_vertex.loop₂ level label before after =
      if 1 < (before ++ after).countP (levelFormulaEqB level label) then
        after.filter (levelFormulaEqB level label)
      else
        [] := by
  induction after generalizing before with
  | nil => simp [repeated_nodes_vertex.loop₂]
  | cons head tail ih =>
      by_cases headMatch : levelFormulaEqB level label head = true
      · obtain ⟨levelEq, formulaEq⟩ :=
          (levelFormulaEqB_eq_true_iff level label head).mp headMatch
        have loop₃Eq := loop₃_eq_any_levelFormulaEqB
          level label head (before ++ tail) headMatch
        have totalCountEq :
            (before ++ head :: tail).countP
                (levelFormulaEqB level label) =
              (before ++ tail).countP
                (levelFormulaEqB level label) + 1 := by
          simp [List.countP_append, headMatch, Nat.add_assoc]
        have recursiveCountEq :
            (List.concat before head ++ tail).countP
                (levelFormulaEqB level label) =
              (before ++ head :: tail).countP
                (levelFormulaEqB level label) := by
          simp [List.concat_eq_append, List.countP_append, headMatch,
            Nat.add_left_comm]
        by_cases repeated :
            1 < (before ++ head :: tail).countP
              (levelFormulaEqB level label)
        · have restPositive :
              0 < (before ++ tail).countP
                (levelFormulaEqB level label) := by
            omega
          have otherMatch :
              (before ++ tail).any (levelFormulaEqB level label) = true :=
            (any_eq_true_iff_countP_pos _ _).mpr restPositive
          have oldLoop₃True :
              repeated_nodes_vertex.loop₃ head (before ++ tail) = true := by
            rw [loop₃Eq]
            exact otherMatch
          have guardTrue :
              (decide (head.LEVEL = level) &&
                decide (head.FORMULA = label) &&
                repeated_nodes_vertex.loop₃ head (before ++ tail)) = true := by
            rw [oldLoop₃True]
            simp [levelEq, formulaEq]
          have recursiveRepeated :
              1 < (List.concat before head ++ tail).countP
                (levelFormulaEqB level label) := by
            omega
          simp only [repeated_nodes_vertex.loop₂]
          rw [if_pos guardTrue]
          rw [ih (before := List.concat before head)]
          rw [if_pos recursiveRepeated, if_pos repeated]
          simp [headMatch]
        · have restZero :
              (before ++ tail).countP
                (levelFormulaEqB level label) = 0 := by
            omega
          have noOther :
              (before ++ tail).any (levelFormulaEqB level label) = false := by
            cases value : (before ++ tail).any
                (levelFormulaEqB level label) with
            | false => rfl
            | true =>
                have positive :=
                  (any_eq_true_iff_countP_pos _ _).mp value
                omega
          have oldLoop₃False :
              repeated_nodes_vertex.loop₃ head (before ++ tail) = false := by
            rw [loop₃Eq]
            exact noOther
          have guardFalse :
              ¬ (decide (head.LEVEL = level) &&
                decide (head.FORMULA = label) &&
                repeated_nodes_vertex.loop₃ head (before ++ tail)) = true := by
            rw [oldLoop₃False]
            simp
          have recursiveNotRepeated :
              ¬ 1 < (List.concat before head ++ tail).countP
                (levelFormulaEqB level label) := by
            omega
          simp only [repeated_nodes_vertex.loop₂]
          rw [if_neg guardFalse]
          rw [ih (before := List.concat before head)]
          rw [if_neg recursiveNotRepeated, if_neg repeated]
      · have keyMiss :
            ¬ (head.LEVEL = level ∧ head.FORMULA = label) := by
          rintro ⟨headLevel, headFormula⟩
          apply headMatch
          simp [levelFormulaEqB, headLevel, headFormula]
        have countEq :
            (before ++ head :: tail).countP
                (levelFormulaEqB level label) =
              (List.concat before head ++ tail).countP
                (levelFormulaEqB level label) := by
          simp [List.concat_eq_append, List.countP_append, headMatch]
        have guardFalse :
            ¬ (decide (head.LEVEL = level) &&
              decide (head.FORMULA = label) &&
              repeated_nodes_vertex.loop₃ head (before ++ tail)) = true := by
          intro selected
          have parts :
              (head.LEVEL = level ∧ head.FORMULA = label) ∧
                repeated_nodes_vertex.loop₃ head (before ++ tail) = true := by
            simpa only [Bool.and_eq_true, decide_eq_true_eq] using selected
          exact keyMiss parts.1
        simp only [repeated_nodes_vertex.loop₂]
        rw [if_neg guardFalse]
        rw [ih (before := List.concat before head)]
        rw [countEq]
        simp [headMatch]

def hcPartitionStep (all : List Vertex) (partition : List HCClass)
    (node : Vertex) : List HCClass :=
  if partition.any fun selection =>
      hcClassKeyEqB selection.representative node then
    partition
  else
    partition ++ [⟨node,
      all.filter (fun member => hcClassKeyEqB node member)⟩]

theorem hcPartition_eq_fold (d : Graph) :
    hcPartition d = d.NODES.foldl (hcPartitionStep d.NODES) [] := by
  rfl

def PartitionMembersCorrect (all : List Vertex)
    (partition : List HCClass) : Prop :=
  ∀ selection ∈ partition,
    selection.members =
      all.filter (hcClassKeyEqB selection.representative)

theorem hcPartitionStep_membersCorrect (all : List Vertex)
    (partition : List HCClass) (node : Vertex)
    (correct : PartitionMembersCorrect all partition) :
    PartitionMembersCorrect all (hcPartitionStep all partition node) := by
  intro selection selectionMember
  unfold hcPartitionStep at selectionMember
  split at selectionMember
  next => exact correct selection selectionMember
  next =>
    rw [List.mem_append] at selectionMember
    rcases selectionMember with oldMember | newMember
    · exact correct selection oldMember
    · simp only [List.mem_singleton] at newMember
      subst selection
      rfl

theorem hcPartitionFold_membersCorrect (all scan : List Vertex)
    (partition : List HCClass)
    (correct : PartitionMembersCorrect all partition) :
    PartitionMembersCorrect all
      (scan.foldl (hcPartitionStep all) partition) := by
  induction scan generalizing partition with
  | nil => exact correct
  | cons head tail ih =>
      exact ih (hcPartitionStep all partition head)
        (hcPartitionStep_membersCorrect all partition head correct)

theorem hcPartition_membersCorrect (d : Graph) :
    PartitionMembersCorrect d.NODES (hcPartition d) := by
  rw [hcPartition_eq_fold]
  exact hcPartitionFold_membersCorrect d.NODES d.NODES [] (by simp
    [PartitionMembersCorrect])

def PartitionCovers (partition : List HCClass)
    (seen : List Vertex) : Prop :=
  ∀ node ∈ seen, ∃ selection ∈ partition,
    hcClassKeyEqB selection.representative node = true

theorem hcPartitionStep_covers (all seen : List Vertex)
    (partition : List HCClass) (node : Vertex)
    (covers : PartitionCovers partition seen) :
    PartitionCovers (hcPartitionStep all partition node) (seen ++ [node]) := by
  intro current currentMember
  rw [List.mem_append] at currentMember
  rcases currentMember with oldMember | newMember
  · obtain ⟨selection, selectionMember, keyEq⟩ :=
      covers current oldMember
    refine ⟨selection, ?_, keyEq⟩
    simp only [hcPartitionStep]
    split
    · exact selectionMember
    · exact List.mem_append_left _ selectionMember
  · simp only [List.mem_singleton] at newMember
    subst current
    simp only [hcPartitionStep]
    split
    next found =>
      obtain ⟨selection, selectionMember, keyEq⟩ :=
        List.any_eq_true.mp found
      exact ⟨selection, selectionMember, keyEq⟩
    next missing =>
      let fresh : HCClass := ⟨node,
        all.filter (fun member => hcClassKeyEqB node member)⟩
      refine ⟨fresh, ?_, ?_⟩
      · exact List.mem_append_right _ (by simp [fresh])
      · simp [fresh, hcClassKeyEqB]

theorem hcPartitionFold_covers (all remaining seen : List Vertex)
    (partition : List HCClass)
    (covers : PartitionCovers partition seen) :
    PartitionCovers
      (remaining.foldl (hcPartitionStep all) partition)
      (seen ++ remaining) := by
  induction remaining generalizing seen partition with
  | nil => simpa using covers
  | cons head tail ih =>
      have nextCovers := hcPartitionStep_covers all seen partition head covers
      have result := ih (seen ++ [head])
        (hcPartitionStep all partition head) nextCovers
      simpa [List.append_assoc] using result

theorem hcPartition_covers (d : Graph) :
    PartitionCovers (hcPartition d) d.NODES := by
  rw [hcPartition_eq_fold]
  simpa using hcPartitionFold_covers d.NODES d.NODES [] [] (by simp
    [PartitionCovers])

def repeatedClassesAtLevel (partition : List HCClass) (level : Nat) :
    List HCClass :=
  (partition.filter fun selection => decide (1 < selection.members.length)).filter
    fun selection => selection.representative.LEVEL == level

def indexedClass? (partition : List HCClass) (level : Nat)
    (label : Formula) : Option HCClass :=
  (repeatedClassesAtLevel partition level).find? fun selection =>
    selection.representative.FORMULA == label

theorem filter_hcClassKeyEqB_eq_levelFormulaEqB
    (nodes : List Vertex) (representative : Vertex)
    (level : Nat) (label : Formula)
    (levelEq : representative.LEVEL = level)
    (formulaEq : representative.FORMULA = label) :
    nodes.filter (hcClassKeyEqB representative) =
      nodes.filter (levelFormulaEqB level label) := by
  congr 1
  funext node
  simp [hcClassKeyEqB, levelFormulaEqB, levelEq, formulaEq]

theorem indexedClass?_members_eq_referenceCell
    (d : Graph) (level : Nat) (label : Formula) :
    (match indexedClass? (hcPartition d) level label with
      | some selection => selection.members
      | none => []) =
      (if 1 < d.NODES.countP (levelFormulaEqB level label) then
        d.NODES.filter (levelFormulaEqB level label)
      else
        []) := by
  by_cases repeated :
      1 < d.NODES.countP (levelFormulaEqB level label)
  · have positive :
        0 < d.NODES.countP (levelFormulaEqB level label) := by omega
    have anyMatch :
        d.NODES.any (levelFormulaEqB level label) = true :=
      (any_eq_true_iff_countP_pos _ _).mpr positive
    obtain ⟨node, nodeMember, nodeMatch⟩ := List.any_eq_true.mp anyMatch
    obtain ⟨selection, selectionMember, selectionNodeKey⟩ :=
      hcPartition_covers d node nodeMember
    obtain ⟨nodeLevel, nodeFormula⟩ :=
      (levelFormulaEqB_eq_true_iff level label node).mp nodeMatch
    obtain ⟨selectionLevelNode, selectionFormulaNode⟩ :=
      (hcClassKeyEqB_eq_true_iff selection.representative node).mp
        selectionNodeKey
    have selectionLevel : selection.representative.LEVEL = level :=
      selectionLevelNode.trans nodeLevel.symm
    have selectionFormula : selection.representative.FORMULA = label :=
      selectionFormulaNode.trans nodeFormula.symm
    have selectionMembers :
        selection.members =
          d.NODES.filter (levelFormulaEqB level label) := by
      rw [hcPartition_membersCorrect d selection selectionMember]
      exact filter_hcClassKeyEqB_eq_levelFormulaEqB
        d.NODES selection.representative level label
        selectionLevel selectionFormula
    have selectionRepeated : 1 < selection.members.length := by
      rw [selectionMembers]
      rwa [List.countP_eq_length_filter] at repeated
    have selectionAtLevel : selection ∈
        repeatedClassesAtLevel (hcPartition d) level := by
      simp [repeatedClassesAtLevel, selectionMember, selectionRepeated,
        selectionLevel]
    have selectionLabel :
        (selection.representative.FORMULA == label) = true := by
      rw [← decide_eq_beq_formula]
      simp [selectionFormula]
    cases foundEq : indexedClass? (hcPartition d) level label with
    | none =>
        have foundEq' :
            (repeatedClassesAtLevel (hcPartition d) level).find?
              (fun current => current.representative.FORMULA == label) = none := by
          simpa [indexedClass?] using foundEq
        have absent := (List.find?_eq_none.mp foundEq')
          selection selectionAtLevel
        exact False.elim (absent selectionLabel)
    | some found =>
        have foundEq' :
            (repeatedClassesAtLevel (hcPartition d) level).find?
              (fun current => current.representative.FORMULA == label) =
                some found := by
          simpa [indexedClass?] using foundEq
        have foundMember : found ∈
            repeatedClassesAtLevel (hcPartition d) level :=
          List.mem_of_find?_eq_some foundEq'
        have foundSelected := List.find?_some foundEq'
        have foundLabel : (found.representative.FORMULA == label) = true :=
          foundSelected
        have foundPartitionMember : found ∈ hcPartition d := by
          exact (List.mem_filter.mp
            (List.mem_filter.mp foundMember).1).1
        have foundLevel : found.representative.LEVEL = level := by
          have selected := (List.mem_filter.mp foundMember).2
          simpa using selected
        have foundFormula : found.representative.FORMULA = label := by
          rw [← decide_eq_beq_formula] at foundLabel
          simpa using foundLabel
        have foundMembers :
            found.members = d.NODES.filter (levelFormulaEqB level label) := by
          rw [hcPartition_membersCorrect d found foundPartitionMember]
          exact filter_hcClassKeyEqB_eq_levelFormulaEqB d.NODES
            found.representative level label foundLevel foundFormula
        change found.members =
          (if 1 < d.NODES.countP (levelFormulaEqB level label) then
            d.NODES.filter (levelFormulaEqB level label)
          else [])
        rw [foundMembers, if_pos repeated]
  · have noClass : indexedClass? (hcPartition d) level label = none := by
      apply List.find?_eq_none.mpr
      intro selection selectionMember selectionLabel
      have repeatedMember := (List.mem_filter.mp selectionMember).1
      have selectionRepeated : 1 < selection.members.length := by
        have selected := (List.mem_filter.mp repeatedMember).2
        simpa using selected
      have selectionPartitionMember : selection ∈ hcPartition d :=
        (List.mem_filter.mp repeatedMember).1
      have selectionLevel : selection.representative.LEVEL = level := by
        have selected := (List.mem_filter.mp selectionMember).2
        simpa using selected
      have selectionFormula : selection.representative.FORMULA = label := by
        rw [← decide_eq_beq_formula] at selectionLabel
        simpa using selectionLabel
      have selectionMembers :
          selection.members =
            d.NODES.filter (levelFormulaEqB level label) := by
        rw [hcPartition_membersCorrect d selection selectionPartitionMember]
        exact filter_hcClassKeyEqB_eq_levelFormulaEqB d.NODES
          selection.representative level label selectionLevel selectionFormula
      apply repeated
      rw [List.countP_eq_length_filter, ← selectionMembers]
      exact selectionRepeated
    rw [noClass, if_neg repeated]

theorem indexedClass?_members_eq_loop₂
    (d : Graph) (level : Nat) (label : Formula) :
    (match indexedClass? (hcPartition d) level label with
      | some selection => selection.members
      | none => []) =
      repeated_nodes_vertex.loop₂ level label [] d.NODES := by
  rw [indexedClass?_members_eq_referenceCell]
  simpa using loop₂_eq_filter_or_empty level label [] d.NODES |>.symm

theorem indexed_loop₁_eq_reference (d : Graph) (level : Nat)
    (labels : List Formula) :
    repeated_nodes_vertex_indexedWithPartition.loop₁ labels
        (repeatedClassesAtLevel (hcPartition d) level) =
      repeated_nodes_vertex.loop₁ level labels d.NODES := by
  induction labels with
  | nil => rfl
  | cons label labels ih =>
      have cellEq := indexedClass?_members_eq_loop₂ d level label
      cases foundEq : indexedClass? (hcPartition d) level label with
      | none =>
          have foundEq' :
              (repeatedClassesAtLevel (hcPartition d) level).find?
                (fun current => current.representative.FORMULA == label) = none := by
            simpa [indexedClass?] using foundEq
          have referenceEmpty :
              repeated_nodes_vertex.loop₂ level label [] d.NODES = [] := by
            simpa [foundEq] using cellEq.symm
          simp only [repeated_nodes_vertex_indexedWithPartition.loop₁,
            repeated_nodes_vertex.loop₁]
          rw [foundEq', if_neg (by simp [referenceEmpty]), ih]
      | some selection =>
          have foundEq' :
              (repeatedClassesAtLevel (hcPartition d) level).find?
                (fun current => current.representative.FORMULA == label) =
                  some selection := by
            simpa [indexedClass?] using foundEq
          have selectionMember : selection ∈
              repeatedClassesAtLevel (hcPartition d) level :=
            List.mem_of_find?_eq_some foundEq'
          have repeatedMember := (List.mem_filter.mp selectionMember).1
          have selectionRepeated : 1 < selection.members.length := by
            have selected := (List.mem_filter.mp repeatedMember).2
            simpa using selected
          have selectionNonempty : selection.members ≠ [] := by
            intro empty
            rw [empty] at selectionRepeated
            simp at selectionRepeated
          have selectionEq :
              selection.members =
                repeated_nodes_vertex.loop₂ level label [] d.NODES := by
            simpa [foundEq] using cellEq
          simp only [repeated_nodes_vertex_indexedWithPartition.loop₁,
            repeated_nodes_vertex.loop₁]
          rw [foundEq', if_pos (by simpa [← selectionEq] using selectionNonempty),
            ih]
          change selection.members ::
              repeated_nodes_vertex.loop₁ level labels d.NODES =
            repeated_nodes_vertex.loop₂ level label [] d.NODES ::
              repeated_nodes_vertex.loop₁ level labels d.NODES
          rw [selectionEq]

theorem indexed_loop₀_eq_reference (d : Graph) (levels : List Nat) :
    repeated_nodes_vertex_indexedWithPartition.loop₀ levels
        (get_labels d)
        ((hcPartition d).filter fun selection =>
          decide (1 < selection.members.length)) =
      repeated_nodes_vertex.loop₀ levels (get_labels d) d.NODES := by
  induction levels with
  | nil => rfl
  | cons level levels ih =>
      have levelClassesEq := indexed_loop₁_eq_reference
        d level (get_labels d)
      simp only [repeated_nodes_vertex_indexedWithPartition.loop₀,
        repeated_nodes_vertex.loop₀]
      have atLevelEq :
          ((hcPartition d).filter fun selection =>
              decide (1 < selection.members.length)).filter
                (fun selection => selection.representative.LEVEL == level) =
            repeatedClassesAtLevel (hcPartition d) level := by
        rfl
      rw [atLevelEq, levelClassesEq, ih]
      by_cases empty :
          repeated_nodes_vertex.loop₁ level (get_labels d) d.NODES = [] <;>
        simp [empty]

theorem repeated_nodes_vertex_indexedWithPartition_eq
    (d : Graph) :
    repeated_nodes_vertex_indexedWithPartition (hcPartition d) d =
      repeated_nodes_vertex d := by
  exact indexed_loop₀_eq_reference d (get_levels d)

theorem repeated_nodes_vertex_indexed_eq (d : Graph) :
    repeated_nodes_vertex_indexed d = repeated_nodes_vertex d := by
  exact repeated_nodes_vertex_indexedWithPartition_eq d

end HCIndexedScheduler
