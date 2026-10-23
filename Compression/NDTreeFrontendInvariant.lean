import Compression.ProperHCInput
import Infrastructure.FlowBranchingGridBridge










open Semantic FlowSpec
open GridCertificateSoundness

namespace HorizontalCompressionInput

theorem encodeNDTreeAt_next (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).next = next + tree.nodeCount := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt, NDTree.nodeCount]
  | intro alpha beta premise ih =>
      simp [encodeNDTreeAt, NDTree.nodeCount, ih, Nat.add_assoc]
      omega
  | elim alpha beta major minor ihMajor ihMinor =>
      simp [encodeNDTreeAt, NDTree.nodeCount, ihMajor, ihMinor]
      omega

@[simp] theorem encodeNDTreeAt_root_number
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root.NUMBER = next := by
  cases tree <;> rfl

@[simp] theorem encodeNDTreeAt_root_level
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root.LEVEL = level := by
  cases tree <;> rfl

@[simp] theorem encodeNDTreeAt_root_formula
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root.FORMULA = tree.conclusion := by
  cases tree <;> rfl

@[simp] theorem encodeNDTreeAt_root_uncollapsed
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root.COLLAPSED = false := by
  cases tree <;> rfl

@[simp] theorem encodeNDTreeAt_root_past
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root.PAST = [] := by
  cases tree <;> rfl

@[simp] theorem encodeNDTreeAt_root_mem_nodes
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).root ∈
      (encodeNDTreeAt tree level next).nodes := by
  cases tree <;> simp [encodeNDTreeAt]

theorem encodeNDTreeAt_nodes_length
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).nodes.length = tree.nodeCount := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt, NDTree.nodeCount]
  | intro alpha beta premise ih =>
      simp [encodeNDTreeAt, NDTree.nodeCount, ih]
  | elim alpha beta major minor ihMajor ihMinor =>
      simp [encodeNDTreeAt, NDTree.nodeCount, ihMajor, ihMinor]



theorem encodeNDTreeAt_node_numbers
    (tree : NDTree) (level next : Nat) :
    (encodeNDTreeAt tree level next).nodes.map Vertex.NUMBER =
      List.range' next tree.nodeCount := by
  induction tree generalizing level next with
  | hyp formula =>
      simp [encodeNDTreeAt, NDTree.nodeCount]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.map_cons, ih, NDTree.nodeCount]
      rw [show next :: List.range' (next + 1) premise.nodeCount =
          List.range' next 1 ++ List.range' (next + 1) premise.nodeCount by
        rfl]
      rw [List.range'_append]
      congr 1 ; omega
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.map_cons, List.map_append, ihMajor, ihMinor, encodeNDTreeAt_next, NDTree.nodeCount]
      rw [show next ::
            (List.range' (next + 1) major.nodeCount ++
              List.range' (next + 1 + major.nodeCount) minor.nodeCount) =
          List.range' next 1 ++ List.range' (next + 1) major.nodeCount ++
            List.range' (next + 1 + major.nodeCount) minor.nodeCount by
        rfl]
      rw [List.range'_append]
      have hstart : next + 1 + major.nodeCount =
          next + (1 + major.nodeCount) := by omega
      rw [hstart]
      have hcount : major.nodeCount + minor.nodeCount + 1 =
          (1 + major.nodeCount) + minor.nodeCount := by omega
      rw [hcount]
      simp

theorem encodeNDTreeAt_node_numbers_nodup
    (tree : NDTree) (level next : Nat) :
    ((encodeNDTreeAt tree level next).nodes.map Vertex.NUMBER).Nodup := by
  rw [encodeNDTreeAt_node_numbers]
  exact List.nodup_range'

theorem encodeNDTreeAt_node_number_bounds
    {tree : NDTree} {level next : Nat} {vertex : Vertex}
    (member : vertex ∈ (encodeNDTreeAt tree level next).nodes) :
    next ≤ vertex.NUMBER ∧ vertex.NUMBER < next + tree.nodeCount := by
  have numberMember : vertex.NUMBER ∈
      (encodeNDTreeAt tree level next).nodes.map Vertex.NUMBER :=
    List.mem_map.mpr ⟨vertex, member, rfl⟩
  rw [encodeNDTreeAt_node_numbers, List.mem_range'] at numberMember
  obtain ⟨index, indexBound, numberEq⟩ := numberMember
  simp at numberEq
  omega



theorem encodeNDTreeAt_number_injective
    {tree : NDTree} {level next : Nat} {left right : Vertex}
    (leftMember : left ∈ (encodeNDTreeAt tree level next).nodes)
    (rightMember : right ∈ (encodeNDTreeAt tree level next).nodes)
    (sameNumber : left.NUMBER = right.NUMBER) :
    left = right := by
  have nodup := encodeNDTreeAt_node_numbers_nodup tree level next
  exact List.inj_on_of_nodup_map nodup leftMember rightMember sameNumber


theorem encodeNDTreeAt_number_iff_eq
    {tree : NDTree} {level next : Nat} {left right : Vertex}
    (leftMember : left ∈ (encodeNDTreeAt tree level next).nodes)
    (rightMember : right ∈ (encodeNDTreeAt tree level next).nodes) :
    left.NUMBER = right.NUMBER ↔ left = right := by
  constructor
  · exact encodeNDTreeAt_number_injective leftMember rightMember
  · intro equal
    subst right
    rfl



theorem encodeNDTreeAt_edges_attached
    (tree : NDTree) (level next : Nat) :
    ∀ edge ∈ (encodeNDTreeAt tree level next).edges,
      edge.START ∈ (encodeNDTreeAt tree level next).nodes ∧
        edge.END ∈ (encodeNDTreeAt tree level next).nodes := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      intro deduction deductionMember
      simp only [encodeNDTreeAt, List.mem_cons] at deductionMember ⊢
      rcases deductionMember with rfl | deductionMember
      · exact ⟨Or.inr (encodeNDTreeAt_root_mem_nodes premise _ _), Or.inl rfl⟩
      · obtain ⟨startMember, endMember⟩ := ih _ _ deduction deductionMember
        exact ⟨Or.inr startMember, Or.inr endMember⟩
  | elim alpha beta major minor ihMajor ihMinor =>
      intro deduction deductionMember
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append] at deductionMember ⊢
      rcases deductionMember with rfl | rfl | deductionMember
      · exact ⟨Or.inr (Or.inl (encodeNDTreeAt_root_mem_nodes major _ _)), Or.inl rfl⟩
      · exact ⟨Or.inr (Or.inr (encodeNDTreeAt_root_mem_nodes minor _ _)), Or.inl rfl⟩
      · rcases deductionMember with majorMember | minorMember
        · obtain ⟨startMember, endMember⟩ :=
            ihMajor _ _ deduction majorMember
          exact ⟨Or.inr (Or.inl startMember), Or.inr (Or.inl endMember)⟩
        · obtain ⟨startMember, endMember⟩ :=
            ihMinor _ _ deduction minorMember
          exact ⟨Or.inr (Or.inr startMember), Or.inr (Or.inr endMember)⟩

theorem encodeNDTreeAt_nodes_uncollapsed
    (tree : NDTree) (level next : Nat) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      vertex.COLLAPSED = false := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro vertex member
      rcases member with rfl | member
      · rfl
      · exact ih _ _ vertex member
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro vertex member
      rcases member with rfl | member
      · rfl
      · rcases member with majorMember | minorMember
        · exact ihMajor _ _ vertex majorMember
        · exact ihMinor _ _ vertex minorMember

theorem encodeNDTreeAt_nodes_past_empty
    (tree : NDTree) (level next : Nat) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      vertex.PAST = [] := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro vertex member
      rcases member with rfl | member
      · rfl
      · exact ih _ _ vertex member
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro vertex member
      rcases member with rfl | member
      · rfl
      · rcases member with majorMember | minorMember
        · exact ihMajor _ _ vertex majorMember
        · exact ihMinor _ _ vertex minorMember

theorem encodeNDTreeAt_edges_colour_zero
    (tree : NDTree) (level next : Nat) :
    ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
      deduction.COLOUR = 0 := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro deduction member
      rcases member with rfl | member
      · rfl
      · exact ih _ _ deduction member
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro deduction member
      rcases member with rfl | rfl | member
      · rfl
      · rfl
      · rcases member with majorMember | minorMember
        · exact ihMajor _ _ deduction majorMember
        · exact ihMinor _ _ deduction minorMember

theorem encodeNDTreeAt_edges_leveled
    (tree : NDTree) (level next : Nat) :
    ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
      deduction.START.LEVEL = deduction.END.LEVEL + 1 := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro deduction member
      rcases member with rfl | member
      · simp
      · exact ih _ _ deduction member
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro deduction member
      rcases member with rfl | rfl | member
      · simp
      · simp
      · rcases member with majorMember | minorMember
        · exact ihMajor _ _ deduction majorMember
        · exact ihMinor _ _ deduction minorMember

theorem encodeNDTreeAt_edge_start_number_gt
    (tree : NDTree) (level next : Nat) :
    ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
      next < deduction.START.NUMBER := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro deduction member
      rcases member with rfl | member
      · simp
      · have := ih (level + 1) (next + 1) deduction member
        omega
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro deduction member
      rcases member with rfl | rfl | member
      · simp
      · simp [encodeNDTreeAt_next]
        have := major.nodeCount_pos
        omega
      · rcases member with majorMember | minorMember
        · have := ihMajor (level + 1) (next + 1) deduction majorMember
          omega
        · have := ihMinor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next
              deduction minorMember
          rw [encodeNDTreeAt_next] at this
          omega

theorem encodeNDTreeAt_edge_start_number_bounds
    {tree : NDTree} {level next : Nat} {deduction : Deduction}
    (member : deduction ∈ (encodeNDTreeAt tree level next).edges) :
    next < deduction.START.NUMBER ∧
      deduction.START.NUMBER < next + tree.nodeCount := by
  refine ⟨encodeNDTreeAt_edge_start_number_gt tree level next deduction member, ?_⟩
  exact (encodeNDTreeAt_node_number_bounds
    (encodeNDTreeAt_edges_attached tree level next deduction member).1).2

theorem encodeNDTreeAt_edge_end_number_bounds
    {tree : NDTree} {level next : Nat} {deduction : Deduction}
    (member : deduction ∈ (encodeNDTreeAt tree level next).edges) :
    next ≤ deduction.END.NUMBER ∧
      deduction.END.NUMBER < next + tree.nodeCount := by
  exact encodeNDTreeAt_node_number_bounds
    (encodeNDTreeAt_edges_attached tree level next deduction member).2

private theorem encodeNDTreeAt_no_edge_ends_below
    (tree : NDTree) (level next : Nat) (target : Vertex)
    (below : target.NUMBER < next) :
    ((encodeNDTreeAt tree level next).edges.filter
      (fun deduction => decide (deduction.END = target))) = [] := by
  rw [List.filter_eq_nil_iff]
  intro deduction deductionMember selected
  simp only [decide_eq_true_eq] at selected
  have lower := (encodeNDTreeAt_edge_end_number_bounds
    (tree := tree) (level := level) (next := next) deductionMember).1
  rw [selected] at lower
  omega

@[simp] theorem encodeNDTreeAt_hyp_root_incoming
    (formula : Formula) (level next : Nat) :
    get_rule.incoming (encodeNDTreeAt (.hyp formula) level next).root
      (dlds (encodeNDTreeAt (.hyp formula) level next).nodes
        (encodeNDTreeAt (.hyp formula) level next).edges []) = [] := by
  rfl

@[simp] theorem encodeNDTreeAt_intro_root_incoming
    (alpha beta : Formula) (premise : NDTree) (level next : Nat) :
    get_rule.incoming (encodeNDTreeAt (.intro alpha beta premise) level next).root
      (dlds (encodeNDTreeAt (.intro alpha beta premise) level next).nodes
        (encodeNDTreeAt (.intro alpha beta premise) level next).edges []) =
      [edge (encodeNDTreeAt premise (level + 1) (next + 1)).root
        (encodeNDTreeAt (.intro alpha beta premise) level next).root 0
        (normalizeDep premise.dependencies)] := by
  rw [FlowSpec.incoming_eq_filter]
  simp [encodeNDTreeAt]
  intro deduction deductionMember endAtRoot
  have lower := (encodeNDTreeAt_edge_end_number_bounds
    (tree := premise) (level := level + 1) (next := next + 1)
    deductionMember).1
  rw [endAtRoot] at lower
  simp at lower

@[simp] theorem encodeNDTreeAt_elim_root_incoming
    (alpha beta : Formula) (major minor : NDTree) (level next : Nat) :
    get_rule.incoming (encodeNDTreeAt (.elim alpha beta major minor) level next).root
      (dlds (encodeNDTreeAt (.elim alpha beta major minor) level next).nodes
        (encodeNDTreeAt (.elim alpha beta major minor) level next).edges []) =
      [edge (encodeNDTreeAt major (level + 1) (next + 1)).root
          (encodeNDTreeAt (.elim alpha beta major minor) level next).root 0
          (normalizeDep major.dependencies),
        edge (encodeNDTreeAt minor (level + 1)
            (encodeNDTreeAt major (level + 1) (next + 1)).next).root
          (encodeNDTreeAt (.elim alpha beta major minor) level next).root 0
          (normalizeDep minor.dependencies)] := by
  rw [FlowSpec.incoming_eq_filter]
  simp [encodeNDTreeAt]
  constructor
  · intro deduction deductionMember endAtRoot
    have lower := (encodeNDTreeAt_edge_end_number_bounds
      (tree := major) (level := level + 1) (next := next + 1)
      deductionMember).1
    rw [endAtRoot] at lower
    simp at lower
  · intro deduction deductionMember endAtRoot
    have lower := (encodeNDTreeAt_edge_end_number_bounds
      (tree := minor) (level := level + 1)
      (next := (encodeNDTreeAt major (level + 1) (next + 1)).next)
      deductionMember).1
    rw [endAtRoot, encodeNDTreeAt_next] at lower
    have := major.nodeCount_pos
    simp at lower
    omega




theorem encodeNDTreeAt_edge_start_numbers_nodup
    (tree : NDTree) (level next : Nat) :
    ((encodeNDTreeAt tree level next).edges.map
      (fun deduction => deduction.START.NUMBER)).Nodup := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.map_cons, encodeNDTreeAt_root_number,
        List.nodup_cons]
      constructor
      · intro numberMember
        obtain ⟨deduction, deductionMember, numberEq⟩ :=
          List.mem_map.mp numberMember
        have greater := ih (level + 1) (next + 1)
        have lower := encodeNDTreeAt_edge_start_number_gt
          premise (level + 1) (next + 1) deduction deductionMember
        omega
      · exact ih (level + 1) (next + 1)
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.map_cons, List.map_append,
        encodeNDTreeAt_root_number, encodeNDTreeAt_next, List.nodup_cons]
      constructor
      · intro numberMember
        simp only [List.mem_cons, List.mem_append, List.mem_map] at numberMember
        rcases numberMember with numberEq | majorMember | minorMember
        · have positive := major.nodeCount_pos
          omega
        · obtain ⟨deduction, deductionMember, numberEq⟩ := majorMember
          have lower := encodeNDTreeAt_edge_start_number_gt
            major (level + 1) (next + 1) deduction deductionMember
          omega
        · obtain ⟨deduction, deductionMember, numberEq⟩ := minorMember
          have lower := encodeNDTreeAt_edge_start_number_gt minor (level + 1)
            (next + 1 + major.nodeCount) deduction deductionMember
          omega
      · constructor
        · intro numberMember
          simp only [List.mem_append, List.mem_map] at numberMember
          rcases numberMember with majorMember | minorMember
          · obtain ⟨deduction, deductionMember, numberEq⟩ := majorMember
            have upper := (encodeNDTreeAt_edge_start_number_bounds
              (tree := major) (level := level + 1) (next := next + 1)
              deductionMember).2
            omega
          · obtain ⟨deduction, deductionMember, numberEq⟩ := minorMember
            have lower := encodeNDTreeAt_edge_start_number_gt minor (level + 1)
              (next + 1 + major.nodeCount) deduction deductionMember
            omega
        · apply (ihMajor (level + 1) (next + 1)).append
            (ihMinor (level + 1) (next + 1 + major.nodeCount))
          rw [List.disjoint_left]
          intro number majorMember minorMember
          obtain ⟨majorEdge, majorEdgeMember, majorNumberEq⟩ :=
            List.mem_map.mp majorMember
          obtain ⟨minorEdge, minorEdgeMember, minorNumberEq⟩ :=
            List.mem_map.mp minorMember
          have majorUpper := (encodeNDTreeAt_edge_start_number_bounds
            (tree := major) (level := level + 1) (next := next + 1)
            majorEdgeMember).2
          have minorLower := encodeNDTreeAt_edge_start_number_gt minor
            (level + 1) (next + 1 + major.nodeCount) minorEdge minorEdgeMember
          omega

theorem encodeNDTreeAt_edge_starts_nodup
    (tree : NDTree) (level next : Nat) :
    ((encodeNDTreeAt tree level next).edges.map Deduction.START).Nodup := by
  apply List.Nodup.of_map Vertex.NUMBER
  simpa [List.map_map] using
    encodeNDTreeAt_edge_start_numbers_nodup tree level next

private theorem filter_key_eq_length_le_one
    {α β : Type} [DecidableEq β] (key : α → β) (target : β)
    (items : List α) (nodup : (items.map key).Nodup) :
    (items.filter fun item => decide (key item = target)).length ≤ 1 := by
  induction items with
  | nil => simp
  | cons item items ih =>
      simp only [List.map_cons, List.nodup_cons] at nodup
      by_cases selected : key item = target
      · have tailEmpty :
            items.filter (fun next => decide (key next = target)) = [] := by
          rw [List.filter_eq_nil_iff]
          intro next nextMember selectedNext
          simp only [decide_eq_true_eq] at selectedNext
          apply nodup.1
          exact List.mem_map.mpr ⟨next, nextMember,
            selectedNext.trans selected.symm⟩
        simp [selected, tailEmpty]
      · simp [selected, ih nodup.2]

private theorem colour_one_step_eq_filterMap
    (edges : List Deduction) (colour : Nat) (reached : List Vertex) :
    colour_one_step edges colour reached =
      edges.filterMap (fun deduction =>
        if deduction.COLOUR = colour ∧ deduction.START ∈ reached then
          some deduction.END else none) ++ reached := by
  induction edges with
  | nil => rfl
  | cons deduction edges ih =>
      unfold colour_one_step
      unfold colour_one_step at ih
      simp only [List.foldr_cons, List.filterMap_cons]
      by_cases selected :
          deduction.COLOUR = colour ∧ deduction.START ∈ reached
      · simp [selected, ih]
      · simp [selected, ih]

private theorem mem_colour_one_step
    {edges : List Deduction} {colour : Nat} {reached : List Vertex}
    {vertex : Vertex}
    (member : vertex ∈ colour_one_step edges colour reached) :
    vertex ∈ reached ∨
      ∃ deduction ∈ edges,
        deduction.COLOUR = colour ∧ deduction.START ∈ reached ∧
          vertex = deduction.END := by
  rw [colour_one_step_eq_filterMap, List.mem_append] at member
  rcases member with generated | old
  · rw [List.mem_filterMap] at generated
    obtain ⟨deduction, deductionMember, generated⟩ := generated
    by_cases selected :
        deduction.COLOUR = colour ∧ deduction.START ∈ reached
    · have endpoint : deduction.END = vertex := by
        simpa [selected.1, selected.2] using generated
      exact Or.inr ⟨deduction, deductionMember, selected.1, selected.2,
        endpoint.symm⟩
    · simp [selected] at generated
  · exact Or.inl old

private theorem colour_one_step_below
    {edges : List Deduction} {colour bound : Nat} {reached : List Vertex}
    (leveled : ∀ deduction ∈ edges,
      deduction.START.LEVEL = deduction.END.LEVEL + 1)
    (below : ∀ vertex ∈ reached, vertex.LEVEL < bound) :
    ∀ vertex ∈ colour_one_step edges colour reached,
      vertex.LEVEL < bound := by
  intro vertex member
  rcases mem_colour_one_step member with old |
      ⟨deduction, deductionMember, _, startMember, rfl⟩
  · exact below vertex old
  · have startBelow := below deduction.START startMember
    have edgeLevel := leveled deduction deductionMember
    omega

private theorem colour_closure_below
    {edges : List Deduction} {colour bound : Nat}
    (leveled : ∀ deduction ∈ edges,
      deduction.START.LEVEL = deduction.END.LEVEL + 1) :
    ∀ fuel reached,
      (∀ vertex ∈ reached, vertex.LEVEL < bound) →
      ∀ vertex ∈ colour_closure fuel edges colour reached,
        vertex.LEVEL < bound := by
  intro fuel
  induction fuel with
  | zero =>
      intro reached below vertex member
      exact below vertex member
  | succ fuel ih =>
      intro reached below vertex member
      apply ih (colour_one_step edges colour reached)
      · exact colour_one_step_below (colour := colour)
          (bound := bound) (reached := reached) (edges := edges)
          leveled below
      · exact member

private theorem mem_colour_descendant_seed
    {edges : List Deduction} {colour : Nat} {origin vertex : Vertex}
    (member : vertex ∈ edges.filterMap
      (fun deduction =>
        if deduction.COLOUR = colour ∧ deduction.START = origin then
          some deduction.END else none)) :
    ∃ deduction ∈ edges,
      deduction.COLOUR = colour ∧ deduction.START = origin ∧
        vertex = deduction.END := by
  rw [List.mem_filterMap] at member
  obtain ⟨deduction, deductionMember, generated⟩ := member
  by_cases selected :
      deduction.COLOUR = colour ∧ deduction.START = origin
  · have endpoint : deduction.END = vertex := by
      simpa [selected.1, selected.2] using generated
    exact ⟨deduction, deductionMember, selected.1, selected.2,
      endpoint.symm⟩
  · simp [selected] at generated




theorem colorAcyclicity_of_leveledColored {d : Graph}
    (leveled : LeveledColored d) : ColorAcyclicity d := by
  intro deduction deductionMember selfMember
  unfold colour_descendants at selfMember
  have seedsBelow :
      ∀ vertex ∈ d.EDGES.filterMap
        (fun edge =>
          if edge.COLOUR = deduction.COLOUR ∧ edge.START = deduction.START then
            some edge.END else none),
        vertex.LEVEL < deduction.START.LEVEL := by
    intro vertex vertexMember
    obtain ⟨edge, edgeMember, _, edgeStart, rfl⟩ :=
      mem_colour_descendant_seed vertexMember
    have edgeLevel := leveled edge edgeMember
    rw [edgeStart] at edgeLevel
    omega
  have strictlyBelow := colour_closure_below
    (edges := d.EDGES) (colour := deduction.COLOUR)
    (bound := deduction.START.LEVEL) leveled d.EDGES.length _ seedsBelow
    deduction.START selfMember
  omega

theorem encodeNDTreeAt_outgoing_length_le_one
    (tree : NDTree) (level next : Nat) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      (get_rule.outgoing vertex
        (dlds (encodeNDTreeAt tree level next).nodes
          (encodeNDTreeAt tree level next).edges [])).length ≤ 1 := by
  intro vertex _
  rw [FlowSpec.outgoing_eq_filter]
  exact filter_key_eq_length_le_one Deduction.START vertex
    (encodeNDTreeAt tree level next).edges
    (encodeNDTreeAt_edge_starts_nodup tree level next)

theorem encodeNDTreeAt_node_root_or_starts_edge
    (tree : NDTree) (level next : Nat) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      vertex = (encodeNDTreeAt tree level next).root ∨
        ∃ deduction ∈ (encodeNDTreeAt tree level next).edges,
          deduction.START = vertex := by
  induction tree generalizing level next with
  | hyp formula =>
      simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro vertex member
      rcases member with rfl | member
      · exact Or.inl rfl
      · rcases ih (level + 1) (next + 1) vertex member with
          childRoot | ⟨deduction, deductionMember, starts⟩
        · exact Or.inr ⟨edge (encodeNDTreeAt premise (level + 1) (next + 1)).root
              (node next level (alpha >> beta) false false []) 0
              (normalizeDep premise.dependencies),
            Or.inl rfl, by simp [childRoot]⟩
        · exact Or.inr ⟨deduction, Or.inr deductionMember,
            starts⟩
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro vertex member
      rcases member with rfl | member
      · exact Or.inl rfl
      · rcases member with majorMember | minorMember
        · rcases ihMajor (level + 1) (next + 1) vertex majorMember with
            majorRoot | ⟨deduction, deductionMember, starts⟩
          · exact Or.inr ⟨edge
                (encodeNDTreeAt major (level + 1) (next + 1)).root
                (node next level beta false false []) 0
                (normalizeDep major.dependencies),
              Or.inl rfl, by simp [majorRoot]⟩
          · exact Or.inr ⟨deduction,
              Or.inr (Or.inr (Or.inl deductionMember)), starts⟩
        · rcases ihMinor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next
              vertex minorMember with
            minorRoot | ⟨deduction, deductionMember, starts⟩
          · exact Or.inr ⟨edge
                (encodeNDTreeAt minor (level + 1)
                  (encodeNDTreeAt major (level + 1) (next + 1)).next).root
                (node next level beta false false []) 0
                (normalizeDep minor.dependencies),
              Or.inr (Or.inl rfl), by simp [minorRoot]⟩
          · exact Or.inr ⟨deduction,
              Or.inr (Or.inr (Or.inr deductionMember)), starts⟩

theorem encodeNDTreeAt_root_outgoing_empty
    (tree : NDTree) (level next : Nat) :
    get_rule.outgoing (encodeNDTreeAt tree level next).root
      (dlds (encodeNDTreeAt tree level next).nodes
        (encodeNDTreeAt tree level next).edges []) = [] := by
  rw [FlowSpec.outgoing_eq_filter, List.filter_eq_nil_iff]
  intro deduction deductionMember selected
  simp only [decide_eq_true_eq] at selected
  have greater := encodeNDTreeAt_edge_start_number_gt
    tree level next deduction deductionMember
  rw [selected, encodeNDTreeAt_root_number] at greater
  omega

theorem encodeNDTreeAt_uniqueRootAt
    (tree : NDTree) (level next : Nat) :
    UniqueRootAt
      (dlds (encodeNDTreeAt tree level next).nodes
        (encodeNDTreeAt tree level next).edges [])
      (encodeNDTreeAt tree level next).root := by
  constructor
  · exact ⟨encodeNDTreeAt_root_mem_nodes tree level next,
      encodeNDTreeAt_root_outgoing_empty tree level next⟩
  · intro selection selectionMember terminal
    rcases encodeNDTreeAt_node_root_or_starts_edge tree level next
        selection selectionMember with equal | ⟨deduction, deductionMember, starts⟩
    · exact equal
    · have outgoingMember : deduction ∈
          get_rule.outgoing selection
            (dlds (encodeNDTreeAt tree level next).nodes
              (encodeNDTreeAt tree level next).edges []) := by
        rw [FlowSpec.outgoing_eq_filter, List.mem_filter]
        exact ⟨deductionMember, by simp [starts]⟩
      rw [terminal] at outgoingMember
      contradiction

def DeductionStep (edges : List Deduction) (start finish : Vertex) : Prop :=
  ∃ deduction ∈ edges,
    deduction.START = start ∧ deduction.END = finish




theorem encodeNDTreeAt_reaches_root
    (tree : NDTree) (level next : Nat) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      Relation.ReflTransGen
        (DeductionStep (encodeNDTreeAt tree level next).edges)
        vertex (encodeNDTreeAt tree level next).root := by
  induction tree generalizing level next with
  | hyp formula =>
      intro vertex member
      simp [encodeNDTreeAt] at member
      subst vertex
      exact .refl
  | intro alpha beta premise ih =>
      intro vertex member
      simp only [encodeNDTreeAt, List.mem_cons] at member
      rcases member with rfl | member
      · exact .refl
      · have childReach := ih (level + 1) (next + 1) vertex member
        have lifted : Relation.ReflTransGen
            (DeductionStep
              (encodeNDTreeAt (.intro alpha beta premise) level next).edges)
            vertex (encodeNDTreeAt premise (level + 1) (next + 1)).root := by
          refine Relation.ReflTransGen.mono ?_ childReach
          intro start finish step
          obtain ⟨deduction, deductionMember, starts, ends⟩ := step
          exact ⟨deduction, by
            simp only [encodeNDTreeAt, List.mem_cons]
            exact Or.inr deductionMember, starts, ends⟩
        apply Relation.ReflTransGen.tail lifted
        exact ⟨edge (encodeNDTreeAt premise (level + 1) (next + 1)).root
            (node next level (alpha >> beta) false false []) 0
            (normalizeDep premise.dependencies), by
          simp [encodeNDTreeAt], rfl, rfl⟩
  | elim alpha beta major minor ihMajor ihMinor =>
      intro vertex member
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append] at member
      rcases member with rfl | majorMember | minorMember
      · exact .refl
      · have childReach := ihMajor (level + 1) (next + 1) vertex majorMember
        have lifted : Relation.ReflTransGen
            (DeductionStep
              (encodeNDTreeAt (.elim alpha beta major minor) level next).edges)
            vertex (encodeNDTreeAt major (level + 1) (next + 1)).root := by
          refine Relation.ReflTransGen.mono ?_ childReach
          intro start finish step
          obtain ⟨deduction, deductionMember, starts, ends⟩ := step
          exact ⟨deduction, by
            simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
            exact Or.inr (Or.inr (Or.inl deductionMember)), starts, ends⟩
        apply Relation.ReflTransGen.tail lifted
        exact ⟨edge (encodeNDTreeAt major (level + 1) (next + 1)).root
            (node next level beta false false []) 0
            (normalizeDep major.dependencies), by
          simp [encodeNDTreeAt], rfl, rfl⟩
      · have childReach := ihMinor (level + 1)
          (encodeNDTreeAt major (level + 1) (next + 1)).next
          vertex minorMember
        have lifted : Relation.ReflTransGen
            (DeductionStep
              (encodeNDTreeAt (.elim alpha beta major minor) level next).edges)
            vertex (encodeNDTreeAt minor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next).root := by
          refine Relation.ReflTransGen.mono ?_ childReach
          intro start finish step
          obtain ⟨deduction, deductionMember, starts, ends⟩ := step
          exact ⟨deduction, by
            simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
            exact Or.inr (Or.inr (Or.inr deductionMember)), starts, ends⟩
        apply Relation.ReflTransGen.tail lifted
        exact ⟨edge (encodeNDTreeAt minor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next).root
            (node next level beta false false []) 0
            (normalizeDep minor.dependencies), by
          simp [encodeNDTreeAt], rfl, rfl⟩







theorem encodeNDTreeAt_root_rule_correct
    (tree : NDTree) (level next : Nat)
    (wellFormed : tree.wellFormedB = true) :
    let graph := dlds (encodeNDTreeAt tree level next).nodes
      (encodeNDTreeAt tree level next).edges []
    RuleShapeOK (encodeNDTreeAt tree level next).root graph ∧
      outDep (encodeNDTreeAt tree level next).root graph =
        normalizeDep tree.dependencies := by
  cases tree with
  | hyp formula =>
      simp [RuleShapeOK, ruleShapeOKB, classifyRule?, outDep,
        encodeNDTreeAt, NDTree.dependencies]
  | intro alpha beta premise =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed
      rw [FlowBranchingGridBridge.normalize_dependencies_intro]
      dsimp only
      unfold RuleShapeOK ruleShapeOKB outDep classifyRule?
      rw [encodeNDTreeAt_intro_root_incoming]
      simp [encodeNDTreeAt, consequent?, antecedent?,
        FlowSpec.depRemove, wellFormed.2]
  | elim alpha beta major minor =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true, beq_iff_eq] at wellFormed
      rw [FlowBranchingGridBridge.normalize_dependencies_elim]
      have majorConclusion : major.conclusion = alpha >> beta := wellFormed.1.2
      have minorConclusion : minor.conclusion = alpha := wellFormed.2
      dsimp only
      unfold RuleShapeOK ruleShapeOKB outDep classifyRule?
      rw [encodeNDTreeAt_elim_root_incoming]
      simp [encodeNDTreeAt, FlowSpec.depUnion,
        majorConclusion, minorConclusion]

def encodedGraphAt (tree : NDTree) (level next : Nat) : Graph :=
  dlds (encodeNDTreeAt tree level next).nodes
    (encodeNDTreeAt tree level next).edges []

private theorem classifyRule_congr_incoming
    {vertex : Vertex} {left right : Graph}
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right) :
    classifyRule? vertex left = classifyRule? vertex right := by
  unfold classifyRule?
  rw [incoming]

private theorem ruleShapeOK_congr_incoming
    {vertex : Vertex} {left right : Graph}
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right) :
    RuleShapeOK vertex left ↔ RuleShapeOK vertex right := by
  unfold RuleShapeOK ruleShapeOKB
  rw [classifyRule_congr_incoming incoming]

private theorem outDep_congr_incoming
    {vertex : Vertex} {left right : Graph}
    (incoming : get_rule.incoming vertex left =
      get_rule.incoming vertex right) :
    outDep vertex left = outDep vertex right := by
  unfold outDep
  rw [classifyRule_congr_incoming incoming]

private theorem encodeNDTreeAt_no_edge_ends_at_or_above
    (tree : NDTree) (level next : Nat) (target : Vertex)
    (above : next + tree.nodeCount ≤ target.NUMBER) :
    ((encodeNDTreeAt tree level next).edges.filter
      (fun deduction => decide (deduction.END = target))) = [] := by
  rw [List.filter_eq_nil_iff]
  intro deduction deductionMember selected
  simp only [decide_eq_true_eq] at selected
  have upper := (encodeNDTreeAt_edge_end_number_bounds
    (tree := tree) (level := level) (next := next) deductionMember).2
  rw [selected] at upper
  omega

theorem encodeNDTreeAt_intro_child_incoming
    (alpha beta : Formula) (premise : NDTree) (level next : Nat)
    {vertex : Vertex}
    (member : vertex ∈ (encodeNDTreeAt premise (level + 1) (next + 1)).nodes) :
    get_rule.incoming vertex
        (encodedGraphAt (.intro alpha beta premise) level next) =
      get_rule.incoming vertex
        (encodedGraphAt premise (level + 1) (next + 1)) := by
  rw [FlowSpec.incoming_eq_filter, FlowSpec.incoming_eq_filter]
  have lower := (encodeNDTreeAt_node_number_bounds member).1
  have parentNe :
      (encodeNDTreeAt (.intro alpha beta premise) level next).root ≠ vertex := by
    intro equal
    rw [← equal, encodeNDTreeAt_root_number] at lower
    omega
  have parentNodeNe :
      (node next level (alpha >> beta) false false []) ≠ vertex := by
    simpa [encodeNDTreeAt] using parentNe
  simp [encodedGraphAt, encodeNDTreeAt, parentNodeNe]

theorem encodeNDTreeAt_elim_major_incoming
    (alpha beta : Formula) (major minor : NDTree) (level next : Nat)
    {vertex : Vertex}
    (member : vertex ∈ (encodeNDTreeAt major (level + 1) (next + 1)).nodes) :
    get_rule.incoming vertex
        (encodedGraphAt (.elim alpha beta major minor) level next) =
      get_rule.incoming vertex
        (encodedGraphAt major (level + 1) (next + 1)) := by
  rw [FlowSpec.incoming_eq_filter, FlowSpec.incoming_eq_filter]
  have bounds := encodeNDTreeAt_node_number_bounds member
  have parentNe :
      (encodeNDTreeAt (.elim alpha beta major minor) level next).root ≠ vertex := by
    intro equal
    rw [← equal, encodeNDTreeAt_root_number] at bounds
    omega
  have minorEmpty := encodeNDTreeAt_no_edge_ends_below minor (level + 1)
    (encodeNDTreeAt major (level + 1) (next + 1)).next vertex (by
      rw [encodeNDTreeAt_next]
      exact bounds.2)
  have parentNodeNe : (node next level beta false false []) ≠ vertex := by
    simpa [encodeNDTreeAt] using parentNe
  simp [encodedGraphAt, encodeNDTreeAt, parentNodeNe, minorEmpty]

theorem encodeNDTreeAt_elim_minor_incoming
    (alpha beta : Formula) (major minor : NDTree) (level next : Nat)
    {vertex : Vertex}
    (member : vertex ∈ (encodeNDTreeAt minor (level + 1)
      (encodeNDTreeAt major (level + 1) (next + 1)).next).nodes) :
    get_rule.incoming vertex
        (encodedGraphAt (.elim alpha beta major minor) level next) =
      get_rule.incoming vertex
        (encodedGraphAt minor (level + 1)
          (encodeNDTreeAt major (level + 1) (next + 1)).next) := by
  rw [FlowSpec.incoming_eq_filter, FlowSpec.incoming_eq_filter]
  have bounds := encodeNDTreeAt_node_number_bounds member
  have parentNe :
      (encodeNDTreeAt (.elim alpha beta major minor) level next).root ≠ vertex := by
    intro equal
    rw [← equal, encodeNDTreeAt_root_number] at bounds
    have majorPositive := major.nodeCount_pos
    rw [encodeNDTreeAt_next] at bounds
    omega
  have majorEmpty := encodeNDTreeAt_no_edge_ends_at_or_above major
    (level + 1) (next + 1) vertex (by
      rw [encodeNDTreeAt_next] at bounds
      exact bounds.1)
  have parentNodeNe : (node next level beta false false []) ≠ vertex := by
    simpa [encodeNDTreeAt] using parentNe
  simp [encodedGraphAt, encodeNDTreeAt, parentNodeNe, majorEmpty]




theorem encodeNDTreeAt_edges_rule_correct
    (tree : NDTree) (level next : Nat)
    (wellFormed : tree.wellFormedB = true) :
    ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
      RuleShapeOK deduction.START (encodedGraphAt tree level next) ∧
        deduction.DEPENDENCY =
          outDep deduction.START (encodedGraphAt tree level next) := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true] at wellFormed
      intro deduction deductionMember
      simp only [encodeNDTreeAt, List.mem_cons] at deductionMember
      rcases deductionMember with rfl | deductionMember
      · have rootCorrect := encodeNDTreeAt_root_rule_correct premise
          (level + 1) (next + 1) wellFormed.1
        have incoming := encodeNDTreeAt_intro_child_incoming alpha beta premise
          level next (encodeNDTreeAt_root_mem_nodes premise _ _)
        refine ⟨(ruleShapeOK_congr_incoming incoming).mpr rootCorrect.1, ?_⟩
        change normalizeDep premise.dependencies =
          outDep (encodeNDTreeAt premise (level + 1) (next + 1)).root
            (encodedGraphAt (.intro alpha beta premise) level next)
        rw [outDep_congr_incoming incoming]
        simpa [encodedGraphAt] using rootCorrect.2.symm
      · have childCorrect := ih (level + 1) (next + 1)
          wellFormed.1 deduction deductionMember
        have startMember := (encodeNDTreeAt_edges_attached premise
          (level + 1) (next + 1) deduction deductionMember).1
        have incoming := encodeNDTreeAt_intro_child_incoming alpha beta premise
          level next startMember
        exact ⟨(ruleShapeOK_congr_incoming incoming).mpr childCorrect.1,
          childCorrect.2.trans (outDep_congr_incoming incoming).symm⟩
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [NDTree.wellFormedB, Bool.and_eq_true] at wellFormed
      intro deduction deductionMember
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append] at deductionMember
      rcases deductionMember with rfl | rfl | deductionMember
      · have rootCorrect := encodeNDTreeAt_root_rule_correct major
          (level + 1) (next + 1) wellFormed.1.1.1
        have incoming := encodeNDTreeAt_elim_major_incoming alpha beta major minor
          level next (encodeNDTreeAt_root_mem_nodes major _ _)
        refine ⟨(ruleShapeOK_congr_incoming incoming).mpr rootCorrect.1, ?_⟩
        change normalizeDep major.dependencies =
          outDep (encodeNDTreeAt major (level + 1) (next + 1)).root
            (encodedGraphAt (.elim alpha beta major minor) level next)
        rw [outDep_congr_incoming incoming]
        simpa [encodedGraphAt] using rootCorrect.2.symm
      · have rootCorrect := encodeNDTreeAt_root_rule_correct minor
          (level + 1) (encodeNDTreeAt major (level + 1) (next + 1)).next
          wellFormed.1.1.2
        have incoming := encodeNDTreeAt_elim_minor_incoming alpha beta major minor
          level next (encodeNDTreeAt_root_mem_nodes minor _ _)
        refine ⟨(ruleShapeOK_congr_incoming incoming).mpr rootCorrect.1, ?_⟩
        change normalizeDep minor.dependencies =
          outDep (encodeNDTreeAt minor (level + 1)
            (encodeNDTreeAt major (level + 1) (next + 1)).next).root
            (encodedGraphAt (.elim alpha beta major minor) level next)
        rw [outDep_congr_incoming incoming]
        simpa [encodedGraphAt] using rootCorrect.2.symm
      · rcases deductionMember with majorMember | minorMember
        · have childCorrect := ihMajor (level + 1) (next + 1)
            wellFormed.1.1.1 deduction majorMember
          have startMember := (encodeNDTreeAt_edges_attached major
            (level + 1) (next + 1) deduction majorMember).1
          have incoming := encodeNDTreeAt_elim_major_incoming alpha beta
            major minor level next startMember
          exact ⟨(ruleShapeOK_congr_incoming incoming).mpr childCorrect.1,
            childCorrect.2.trans (outDep_congr_incoming incoming).symm⟩

        · have childCorrect := ihMinor (level + 1)
            (encodeNDTreeAt major (level + 1) (next + 1)).next
            wellFormed.1.1.2 deduction minorMember
          have startMember := (encodeNDTreeAt_edges_attached minor
            (level + 1) (encodeNDTreeAt major (level + 1) (next + 1)).next
            deduction minorMember).1
          have incoming := encodeNDTreeAt_elim_minor_incoming alpha beta
            major minor level next startMember
          exact ⟨(ruleShapeOK_congr_incoming incoming).mpr childCorrect.1,
            childCorrect.2.trans (outDep_congr_incoming incoming).symm⟩

theorem encodeNDTreeAt_localRuleCorrect
    (tree : NDTree) (level next : Nat)
    (wellFormed : tree.wellFormedB = true) :
  LocalRuleCorrect (encodedGraphAt tree level next) := by
  intro vertex vertexMember outgoingNonempty
  cases outgoingEq : get_rule.outgoing vertex (encodedGraphAt tree level next) with
  | nil => exact False.elim (outgoingNonempty outgoingEq)
  | cons first rest =>
      have firstOutgoing : first ∈
          get_rule.outgoing vertex (encodedGraphAt tree level next) := by
        rw [outgoingEq]
        exact List.mem_cons_self
      have firstMember := mem_outgoing_mem_edges vertex
        (encodedGraphAt tree level next) firstOutgoing
      have firstStarts := mem_outgoing_start_eq vertex
        (encodedGraphAt tree level next) firstOutgoing
      have firstCorrect := encodeNDTreeAt_edges_rule_correct tree level next
        wellFormed first firstMember
      constructor
      · simpa [firstStarts] using firstCorrect.1
      · intro deduction deductionOutgoing
        have deductionOutgoing' : deduction ∈
            get_rule.outgoing vertex (encodedGraphAt tree level next) := by
          rw [outgoingEq]
          exact deductionOutgoing
        have deductionMember := mem_outgoing_mem_edges vertex
          (encodedGraphAt tree level next) deductionOutgoing'
        have deductionStarts := mem_outgoing_start_eq vertex
          (encodedGraphAt tree level next) deductionOutgoing'
        have deductionCorrect := encodeNDTreeAt_edges_rule_correct tree level next
          wellFormed deduction deductionMember
        simpa [deductionStarts] using deductionCorrect.2

theorem encodeNDTreeAt_edge_end_not_hypothesis
    (tree : NDTree) (level next : Nat) :
    ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
      deduction.END.HYPOTHESIS = false := by
  induction tree generalizing level next with
  | hyp formula => simp [encodeNDTreeAt]
  | intro alpha beta premise ih =>
      simp only [encodeNDTreeAt, List.mem_cons]
      intro deduction member
      rcases member with rfl | member
      · rfl
      · exact ih _ _ deduction member
  | elim alpha beta major minor ihMajor ihMinor =>
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
      intro deduction member
      rcases member with rfl | rfl | member
      · rfl
      · rfl
      · rcases member with majorMember | minorMember
        · exact ihMajor _ _ deduction majorMember
        · exact ihMinor _ _ deduction minorMember




structure NDTreeStructuralEncodingInvariant
    (tree : NDTree) (level next : Nat) : Prop where
  nextExact : (encodeNDTreeAt tree level next).next = next + tree.nodeCount
  rootNumber : (encodeNDTreeAt tree level next).root.NUMBER = next
  rootLevel : (encodeNDTreeAt tree level next).root.LEVEL = level
  rootFormula : (encodeNDTreeAt tree level next).root.FORMULA = tree.conclusion
  rootMember : (encodeNDTreeAt tree level next).root ∈
    (encodeNDTreeAt tree level next).nodes
  nodeNumbersExact :
    (encodeNDTreeAt tree level next).nodes.map Vertex.NUMBER =
      List.range' next tree.nodeCount
  numberUnique : ∀ left ∈ (encodeNDTreeAt tree level next).nodes,
    ∀ right ∈ (encodeNDTreeAt tree level next).nodes,
      left.NUMBER = right.NUMBER ↔ left = right
  edgesAttached : ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
    deduction.START ∈ (encodeNDTreeAt tree level next).nodes ∧
      deduction.END ∈ (encodeNDTreeAt tree level next).nodes
  nodesUncollapsed : ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
    vertex.COLLAPSED = false
  temporaryPastEmpty : ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
    vertex.PAST = []
  edgesColourZero : ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
    deduction.COLOUR = 0
  edgesLeveled : ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
    deduction.START.LEVEL = deduction.END.LEVEL + 1
  edgeStartsUnique :
    ((encodeNDTreeAt tree level next).edges.map Deduction.START).Nodup
  edgeEndsDerived : ∀ deduction ∈ (encodeNDTreeAt tree level next).edges,
    deduction.END.HYPOTHESIS = false

theorem encodeNDTreeAt_structuralInvariant
    (tree : NDTree) (level next : Nat) :
    NDTreeStructuralEncodingInvariant tree level next := by
  constructor
  · exact encodeNDTreeAt_next tree level next
  · exact encodeNDTreeAt_root_number tree level next
  · exact encodeNDTreeAt_root_level tree level next
  · exact encodeNDTreeAt_root_formula tree level next
  · exact encodeNDTreeAt_root_mem_nodes tree level next
  · exact encodeNDTreeAt_node_numbers tree level next
  · intro left leftMember right rightMember
    exact encodeNDTreeAt_number_iff_eq leftMember rightMember
  · exact encodeNDTreeAt_edges_attached tree level next
  · exact encodeNDTreeAt_nodes_uncollapsed tree level next
  · exact encodeNDTreeAt_nodes_past_empty tree level next
  · exact encodeNDTreeAt_edges_colour_zero tree level next
  · exact encodeNDTreeAt_edges_leveled tree level next
  · exact encodeNDTreeAt_edge_starts_nodup tree level next
  · exact encodeNDTreeAt_edge_end_not_hypothesis tree level next



theorem encodeNDTreeAt_check_dlds
    (tree : NDTree) (level next : Nat) :
    check_dlds
      (dlds (encodeNDTreeAt tree level next).nodes
        (encodeNDTreeAt tree level next).edges []) := by
  refine ⟨?_, ?_, ?_⟩
  · intro left right leftMember rightMember
    exact encodeNDTreeAt_number_iff_eq leftMember rightMember
  · intro deduction deductionMember
    exact encodeNDTreeAt_edges_attached tree level next deduction deductionMember
  · simp

theorem ndTreeToGraph_check_dlds (tree : NDTree) :
    check_dlds (ndTreeToGraph tree) := by
  exact encodeNDTreeAt_check_dlds tree 0 1

theorem ndTreeToGraph_leveledColored (tree : NDTree) :
    LeveledColored (ndTreeToGraph tree) := by
  exact encodeNDTreeAt_edges_leveled tree 0 1

theorem ndTreeToGraph_nodes_uncollapsed (tree : NDTree) :
    ∀ vertex ∈ (ndTreeToGraph tree).NODES, vertex.COLLAPSED = false := by
  exact encodeNDTreeAt_nodes_uncollapsed tree 0 1

theorem ndTreeToGraph_edges_colour_zero (tree : NDTree) :
    ∀ deduction ∈ (ndTreeToGraph tree).EDGES, deduction.COLOUR = 0 := by
  exact encodeNDTreeAt_edges_colour_zero tree 0 1

theorem encodeNDTreeAt_hypothesesHaveNoIncoming
    (tree : NDTree) (level next : Nat) :
    HypothesesHaveNoIncoming
      (dlds (encodeNDTreeAt tree level next).nodes
        (encodeNDTreeAt tree level next).edges []) := by
  intro vertex vertexMember hypothesis
  rw [FlowSpec.incoming_eq_filter, List.filter_eq_nil_iff]
  intro deduction deductionMember selected
  simp only [decide_eq_true_eq] at selected
  have endDerived :=
    encodeNDTreeAt_edge_end_not_hypothesis tree level next deduction deductionMember
  rw [selected, hypothesis] at endDerived
  contradiction

theorem ndTreeToGraph_hypothesesHaveNoIncoming (tree : NDTree) :
    HypothesesHaveNoIncoming (ndTreeToGraph tree) := by
  exact encodeNDTreeAt_hypothesesHaveNoIncoming tree 0 1

theorem ndTreeToGraph_localRuleCorrect
    (tree : NDTree) (wellFormed : tree.wellFormedB = true) :
    LocalRuleCorrect (ndTreeToGraph tree) := by
  change LocalRuleCorrect (encodedGraphAt tree 0 1)
  exact encodeNDTreeAt_localRuleCorrect tree 0 1 wellFormed



theorem ndTreeToGraph_isTreeDLDS (tree : NDTree) :
    IsTreeDLDS (ndTreeToGraph tree) := by
  refine ⟨?_, ?_, rfl, ?_⟩
  · exact encodeNDTreeAt_outgoing_length_le_one tree 0 1
  · exact encodeNDTreeAt_nodes_uncollapsed tree 0 1
  · exact encodeNDTreeAt_edges_colour_zero tree 0 1

theorem ndTreeToGraph_uniqueRoot (tree : NDTree) :
    UniqueRootAt (ndTreeToGraph tree) (encodeNDTreeAt tree 0 1).root := by
  exact encodeNDTreeAt_uniqueRootAt tree 0 1

theorem ndTreeToGraph_rootDischarged
    (tree : NDTree) (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    RootDischarged (ndTreeToGraph tree) := by
  intro root rootMember terminal
  have rootEq := (ndTreeToGraph_uniqueRoot tree).2 root rootMember terminal
  subst root
  have correct := encodeNDTreeAt_root_rule_correct tree 0 1 wellFormed
  refine ⟨correct.1, ?_⟩
  change outDep (encodeNDTreeAt tree 0 1).root
    (dlds (encodeNDTreeAt tree 0 1).nodes
      (encodeNDTreeAt tree 0 1).edges []) = []
  rw [correct.2, closed]
  exact normalizeDep_nil

theorem ndTreeToGraph_simplicity (tree : NDTree) :
    Simplicity (ndTreeToGraph tree) := by
  intro left leftMember right rightMember sameKey
  exact List.inj_on_of_nodup_map
    (encodeNDTreeAt_edge_starts_nodup tree 0 1)
    leftMember rightMember sameKey.1

theorem ndTreeToGraph_colorAcyclicity (tree : NDTree) :
    ColorAcyclicity (ndTreeToGraph tree) :=
  colorAcyclicity_of_leveledColored (ndTreeToGraph_leveledColored tree)

theorem ndTreeToGraph_ancestorSimplicity (tree : NDTree) :
    AncestorSimplicity (ndTreeToGraph tree) := by
  simp [AncestorSimplicity, ndTreeToGraph]

theorem ndTreeToGraph_ancestorEdges (tree : NDTree) :
    AncestorEdges (ndTreeToGraph tree) := by
  simp [AncestorEdges, ndTreeToGraph]

theorem ndTreeToGraph_ancestorBackwayInformation (tree : NDTree) :
    AncestorBackwayInformation (ndTreeToGraph tree) := by
  simp [AncestorBackwayInformation, ndTreeToGraph]

theorem ndTreeToGraph_nonNestedAncestorEdges (tree : NDTree) :
    NonNestedAncestorEdges (ndTreeToGraph tree) := by
  simp [NonNestedAncestorEdges, ndTreeToGraph]



theorem ndTreeToGraph_validDLDS
    (tree : NDTree) (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    ValidDLDS (ndTreeToGraph tree) := by
  constructor
  · exact ndTreeToGraph_check_dlds tree
  · exact ndTreeToGraph_leveledColored tree
  · exact ndTreeToGraph_simplicity tree
  · exact ndTreeToGraph_ancestorSimplicity tree
  · exact ndTreeToGraph_hypothesesHaveNoIncoming tree
  · exact ndTreeToGraph_localRuleCorrect tree wellFormed
  · exact ndTreeToGraph_rootDischarged tree wellFormed closed
  · exact ndTreeToGraph_colorAcyclicity tree
  · exact ndTreeToGraph_ancestorEdges tree
  · exact ndTreeToGraph_ancestorBackwayInformation tree
  · exact ndTreeToGraph_nonNestedAncestorEdges tree

end HorizontalCompressionInput
