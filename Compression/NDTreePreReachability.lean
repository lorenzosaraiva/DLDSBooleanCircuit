import Compression.NDTreeFrontendInvariant









open Semantic FlowSpec
open GridCertificateSoundness ValidDLDSDecomposition

namespace HorizontalCompressionInput




inductive DeductionPath (edges : List Deduction) :
    Nat -> Vertex -> Vertex -> Prop where
  | refl (vertex : Vertex) : DeductionPath edges 0 vertex vertex
  | tail {length : Nat} {start middle finish : Vertex}
      (prior : DeductionPath edges length start middle)
      (last : DeductionStep edges middle finish) :
      DeductionPath edges (length + 1) start finish

theorem DeductionPath.mono
    {small large : List Deduction} {length : Nat} {start finish : Vertex}
    (included : ∀ deduction ∈ small, deduction ∈ large)
    (path : DeductionPath small length start finish) :
    DeductionPath large length start finish := by
  induction path with
  | refl vertex => exact .refl vertex
  | tail prior last ih =>
      apply DeductionPath.tail ih
      obtain ⟨deduction, member, starts, ends⟩ := last
      exact ⟨deduction, included deduction member, starts, ends⟩

theorem mem_pre_go_of_mem_acc (d : Graph) :
    forall fuel frontier acc vertex,
      vertex ∈ acc -> vertex ∈ Pre.go d fuel frontier acc := by
  intro fuel
  induction fuel with
  | zero =>
      intro frontier acc vertex member
      simpa [Pre.go] using member
  | succ fuel ih =>
      intro frontier acc vertex member
      simp only [Pre.go]
      split
      · exact member
      · apply ih
        exact List.mem_append_left _ member

@[simp] theorem mem_predsOf_iff (d : Graph) (start finish : Vertex) :
    start ∈ predsOf d finish ↔
      ∃ deduction ∈ d.EDGES,
        deduction.START = start ∧ deduction.END = finish := by
  simp only [predsOf, List.mem_eraseDups, List.mem_map,
    List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨deduction, ⟨member, ends⟩, starts⟩
    exact ⟨deduction, member, starts, ends⟩
  · rintro ⟨deduction, member, starts, ends⟩
    exact ⟨deduction, ⟨member, ends⟩, starts⟩









theorem deductionPath_mem_pre_go
    (d : Graph)
    (leveled : LeveledColored d)
    {length fuel : Nat} {start finish : Vertex}
    {frontier acc : List Vertex}
    (path : DeductionPath d.EDGES length start finish)
    (finishMem : finish ∈ frontier)
    (frontierLevel : ∀ vertex ∈ frontier,
      vertex.LEVEL = finish.LEVEL)
    (accLevel : ∀ vertex ∈ acc, vertex.LEVEL ≤ finish.LEVEL)
    (zeroSeen : length = 0 → finish ∈ acc)
    (enoughFuel : length ≤ fuel) :
    start ∈ Pre.go d fuel frontier acc := by
  induction path generalizing fuel frontier acc with
  | refl vertex =>
      exact mem_pre_go_of_mem_acc d fuel frontier acc vertex (zeroSeen rfl)
  | @tail length start middle finish prior last ih =>
      obtain ⟨deduction, deductionMember, starts, ends⟩ := last
      have middleLevel : middle.LEVEL = finish.LEVEL + 1 := by
        simpa [starts, ends] using leveled deduction deductionMember
      cases fuel with
      | zero => omega
      | succ fuel =>
          let fresh := ((frontier.flatMap (predsOf d)).eraseDups).filter
            (fun vertex => !(acc.any (fun seen => seen = vertex)))
          have middleFlat : middle ∈ frontier.flatMap (predsOf d) := by
            rw [List.mem_flatMap]
            exact ⟨finish, finishMem,
              (mem_predsOf_iff d middle finish).2
                ⟨deduction, deductionMember, starts, ends⟩⟩
          have middleNotAcc : middle ∉ acc := by
            intro member
            have := accLevel middle member
            omega
          have middleFresh : middle ∈ fresh := by
            simp only [fresh, List.mem_filter, List.mem_eraseDups]
            refine ⟨middleFlat, ?_⟩
            have anyFalse : acc.any (fun seen => seen = middle) = false := by
              rw [List.any_eq_false]
              intro seen seenMember selected
              have equal : seen = middle := decide_eq_true_eq.mp selected
              exact middleNotAcc (by simpa [equal] using seenMember)
            simp [anyFalse]
          have freshNonempty : fresh.isEmpty = false := by
            cases equal : fresh with
            | nil => simp [equal] at middleFresh
            | cons head tail => rfl
          have freshLevel : ∀ vertex ∈ fresh,
              vertex.LEVEL = middle.LEVEL := by
            intro vertex member
            have flatMember : vertex ∈ frontier.flatMap (predsOf d) := by
              exact (List.mem_filter.mp member).1 |> List.mem_eraseDups.mp
            rw [List.mem_flatMap] at flatMember
            obtain ⟨endpoint, endpointMember, predecessor⟩ := flatMember
            obtain ⟨edge, edgeMember, edgeStarts, edgeEnds⟩ :=
              (mem_predsOf_iff d vertex endpoint).mp predecessor
            have edgeLevel := leveled edge edgeMember
            have endpointLevel := frontierLevel endpoint endpointMember
            have vertexLevel : vertex.LEVEL = endpoint.LEVEL + 1 := by
              simpa [edgeStarts, edgeEnds] using edgeLevel
            omega
          have appendedLevel : ∀ vertex ∈ acc ++ fresh,
              vertex.LEVEL ≤ middle.LEVEL := by
            intro vertex member
            rw [List.mem_append] at member
            rcases member with old | new
            · have := accLevel vertex old
              omega
            · exact Nat.le_of_eq (freshLevel vertex new)
          simp only [Pre.go]
          rw [show
            ((frontier.flatMap (predsOf d)).eraseDups).filter
                (fun vertex => !(acc.any (fun seen => seen = vertex))) = fresh
              by rfl]
          rw [freshNonempty]
          exact ih middleFresh freshLevel appendedLevel
            (fun _ => List.mem_append_right acc middleFresh) (by omega)




theorem deductionPath_mem_Pre
    (d : Graph)
    (leveled : LeveledColored d)
    {length : Nat} {start finish : Vertex}
    (path : DeductionPath d.EDGES (length + 1) start finish)
    (enoughFuel : length + 1 ≤ d.NODES.length + 1) :
    start ∈ Pre d finish := by
  unfold Pre
  apply deductionPath_mem_pre_go d leveled path
  · simp
  · simp
  · simp
  · omega
  · exact enoughFuel





theorem encodeNDTreeAt_boundedPathToRoot
    (tree : NDTree) (level next : Nat) :
    ∀ vertex ∈ (encodeNDTreeAt tree level next).nodes,
      ∃ length, length < tree.nodeCount ∧
        DeductionPath (encodeNDTreeAt tree level next).edges length
          vertex (encodeNDTreeAt tree level next).root := by
  induction tree generalizing level next with
  | hyp formula =>
      intro vertex member
      simp [encodeNDTreeAt] at member
      subst vertex
      exact ⟨0, by simp [NDTree.nodeCount], .refl _⟩
  | intro alpha beta premise ih =>
      intro vertex member
      simp only [encodeNDTreeAt, List.mem_cons] at member
      rcases member with rfl | member
      · exact ⟨0, by simp [NDTree.nodeCount], .refl _⟩
      · obtain ⟨length, lengthBound, childPath⟩ :=
          ih (level + 1) (next + 1) vertex member
        have lifted : DeductionPath
            (encodeNDTreeAt (.intro alpha beta premise) level next).edges
            length vertex
            (encodeNDTreeAt premise (level + 1) (next + 1)).root := by
          apply childPath.mono
          intro deduction deductionMember
          simp only [encodeNDTreeAt, List.mem_cons]
          exact Or.inr deductionMember
        refine ⟨length + 1, ?_, DeductionPath.tail lifted ?_⟩
        · simp only [NDTree.nodeCount]
          omega
        · exact ⟨edge
              (encodeNDTreeAt premise (level + 1) (next + 1)).root
              (node next level (alpha >> beta) false false []) 0
              (normalizeDep premise.dependencies), by
            simp [encodeNDTreeAt], rfl, rfl⟩
  | elim alpha beta major minor ihMajor ihMinor =>
      intro vertex member
      simp only [encodeNDTreeAt, List.mem_cons, List.mem_append] at member
      rcases member with rfl | majorMember | minorMember
      · exact ⟨0, by simp [NDTree.nodeCount], .refl _⟩
      · obtain ⟨length, lengthBound, childPath⟩ :=
          ihMajor (level + 1) (next + 1) vertex majorMember
        have lifted : DeductionPath
            (encodeNDTreeAt (.elim alpha beta major minor) level next).edges
            length vertex
            (encodeNDTreeAt major (level + 1) (next + 1)).root := by
          apply childPath.mono
          intro deduction deductionMember
          simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
          exact Or.inr (Or.inr (Or.inl deductionMember))
        refine ⟨length + 1, ?_, DeductionPath.tail lifted ?_⟩
        · simp only [NDTree.nodeCount]
          omega
        · exact ⟨edge
              (encodeNDTreeAt major (level + 1) (next + 1)).root
              (node next level beta false false []) 0
              (normalizeDep major.dependencies), by
            simp [encodeNDTreeAt], rfl, rfl⟩
      · obtain ⟨length, lengthBound, childPath⟩ :=
          ihMinor (level + 1)
            (encodeNDTreeAt major (level + 1) (next + 1)).next
            vertex minorMember
        have lifted : DeductionPath
            (encodeNDTreeAt (.elim alpha beta major minor) level next).edges
            length vertex
            (encodeNDTreeAt minor (level + 1)
              (encodeNDTreeAt major (level + 1) (next + 1)).next).root := by
          apply childPath.mono
          intro deduction deductionMember
          simp only [encodeNDTreeAt, List.mem_cons, List.mem_append]
          exact Or.inr (Or.inr (Or.inr deductionMember))
        refine ⟨length + 1, ?_, DeductionPath.tail lifted ?_⟩
        · simp only [NDTree.nodeCount]
          omega
        · exact ⟨edge
              (encodeNDTreeAt minor (level + 1)
                (encodeNDTreeAt major (level + 1) (next + 1)).next).root
              (node next level beta false false []) 0
              (normalizeDep minor.dependencies), by
            simp [encodeNDTreeAt], rfl, rfl⟩




theorem encodeNDTreeAt_everyNodeReachesRoot
    (tree : NDTree) (level next : Nat) :
    EveryNodeReachesRoot (encodedGraphAt tree level next)
      (encodeNDTreeAt tree level next).root := by
  intro vertex member
  obtain ⟨length, lengthBound, path⟩ :=
    encodeNDTreeAt_boundedPathToRoot tree level next vertex member
  cases length with
  | zero =>
      cases path
      exact Or.inl rfl
  | succ length =>
      apply Or.inr
      apply deductionPath_mem_Pre (encodedGraphAt tree level next)
        (encodeNDTreeAt_edges_leveled tree level next) path
      rw [show (encodedGraphAt tree level next).NODES =
          (encodeNDTreeAt tree level next).nodes by rfl,
        encodeNDTreeAt_nodes_length]
      omega

theorem ndTreeToGraph_everyNodeReachesRoot (tree : NDTree) :
    EveryNodeReachesRoot (ndTreeToGraph tree)
      (encodeNDTreeAt tree 0 1).root := by
  exact encodeNDTreeAt_everyNodeReachesRoot tree 0 1



theorem ndTreeToGraph_properHCInput
    (tree : NDTree)
    (wellFormed : tree.wellFormedB = true)
    (closed : tree.dependencies = []) :
    ProperHCInput (ndTreeToGraph tree) := by
  refine ⟨ndTreeToGraph_isTreeDLDS tree,
    ndTreeToGraph_validDLDS tree wellFormed closed, ?_⟩
  exact ⟨(encodeNDTreeAt tree 0 1).root,
    ndTreeToGraph_uniqueRoot tree,
    ndTreeToGraph_everyNodeReachesRoot tree⟩

end HorizontalCompressionInput
