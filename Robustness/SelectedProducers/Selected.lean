import Robustness.JustifiedRoutes

/-! One explicitly selected producer for every clause of historical Flow.
Each pick is an indexed singleton (or empty on a bad index), never the final
Flow output list. Unused kinds and failed structural branches reject. -/
open DLDS Flow
namespace Robustness.SelectedProducers

inductive SelectedKind where
  | seedDefault
  | seedAncestor (ancestor : Nat)
  | elimination (pair minorRoute majorRoute minorEdge majorEdge : Nat)
  | introduction (route edge : Nat)
  | mixedIntroduction (premise route edge : Nat)
  | eliminationReseed (pair minorRoute majorRoute minorEdge majorEdge ancestor : Nat)
  | introductionReseed (route edge ancestor : Nat)
  deriving DecidableEq, Repr

def pick {α : Type} (xs : List α) (index : Nat) : List α := xs[index]?.toList

theorem pick_mem {α : Type} (xs : List α) (index : Nat) :
    ∀ x ∈ pick xs index, x ∈ xs := by
  intro x hx
  exact List.mem_of_getElem? (by simpa [pick] using hx)

private theorem bind_mono {α β : Type} {xs ys : List α}
    {f g : α → List β} (hxy : ∀ x ∈ xs, x ∈ ys)
    (hfg : ∀ x ∈ xs, ∀ b ∈ f x, b ∈ g x) :
    ∀ b ∈ xs.flatMap f, b ∈ ys.flatMap g := by
  intro b hb
  obtain ⟨x, hx, hb⟩ := List.mem_flatMap.mp hb
  exact List.mem_flatMap.mpr ⟨x, hxy x hx, hfg x hx b hb⟩

def ordinaryElim (d : Graph) (P : Vertex → List FlowPair) (v : Vertex) : List FlowPair :=
  (elimPairsAt d v).flatMap fun pair =>
    (P pair.2).flatMap fun minor =>
      (P pair.1).flatMap fun major =>
        (edgesBetween d pair.2 v).flatMap fun me =>
          (edgesBetween d pair.1 v).filterMap fun ae =>
            match consume me minor.2, consume ae major.2 with
            | some mt, some atail =>
                if mt = atail then some (depUnion minor.1 major.1, mt) else none
            | _, _ => none

def chosenElim (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (pi mi ai mei aei : Nat) : List FlowPair :=
  (pick (elimPairsAt d v) pi).flatMap fun pair =>
    (pick (P pair.2) mi).flatMap fun minor =>
      (pick (P pair.1) ai).flatMap fun major =>
        (pick (edgesBetween d pair.2 v) mei).flatMap fun me =>
          (pick (edgesBetween d pair.1 v) aei).filterMap fun ae =>
            match consume me minor.2, consume ae major.2 with
            | some mt, some atail =>
                if mt = atail then some (depUnion minor.1 major.1, mt) else none
            | _, _ => none

def ordinaryIntro (d : Graph) (P : Vertex → List FlowPair) (v : Vertex) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [u], .implication alpha beta =>
      if u.FORMULA = beta then
        (P u).flatMap fun route =>
          (edgesBetween d u v).filterMap fun edge =>
            (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)
      else []
  | _, _ => []

def chosenIntro (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (ri ei : Nat) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [u], .implication alpha beta =>
      if u.FORMULA = beta then
        (pick (P u) ri).flatMap fun route =>
          (pick (edgesBetween d u v) ei).filterMap fun edge =>
            (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)
      else []
  | _, _ => []

def chosenMixed (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (ui ri ei : Nat) : List FlowPair :=
  match v.FORMULA with
  | .implication alpha beta =>
      (pick (predsOf d v) ui).flatMap fun u =>
        if !elimPremiseAtB d v u && u.FORMULA == beta then
          (pick (P u) ri).flatMap fun route =>
            (pick (edgesBetween d u v) ei).filterMap fun edge =>
              (consume edge route.2).map fun tail => (depRemove route.1 alpha, tail)
        else []
  | _ => []

def chosenElimReseed (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (pi mi ai mei aei anci : Nat) : List FlowPair :=
  (pick (elimPairsAt d v) pi).flatMap fun pair =>
    (pick (P pair.2) mi).flatMap fun minor =>
      (pick (P pair.1) ai).flatMap fun major =>
        (pick (edgesBetween d pair.2 v) mei).flatMap fun me =>
          (pick (edgesBetween d pair.1 v) aei).flatMap fun ae =>
            if reseedPremiseAgreesB minor me && reseedPremiseAgreesB major ae &&
                depSimilarB minor.1 me.DEPENDENCY && depSimilarB major.1 ae.DEPENDENCY then
              (pick (eligibleAncestorsAtRoot d v) anci).map fun ancestor =>
                (depUnion minor.1 major.1, ancestor.COLOURS)
            else []

def chosenIntroReseed (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (ri ei anci : Nat) : List FlowPair :=
  match predsOf d v, v.FORMULA with
  | [u], .implication alpha beta =>
      if u.FORMULA == beta then
        (pick (P u) ri).flatMap fun route =>
          (pick (edgesBetween d u v) ei).flatMap fun edge =>
            if reseedZeroPremiseB route edge && depSimilarB route.1 edge.DEPENDENCY then
              (pick (eligibleAncestorsAtRoot d v) anci).map fun ancestor =>
                (depRemove route.1 alpha, ancestor.COLOURS)
            else []
      else []
  | _, _ => []

theorem chosenElim_subset (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (pi mi ai mei aei : Nat) :
    ∀ r ∈ chosenElim d P v pi mi ai mei aei, r ∈ ordinaryElim d P v := by
  unfold chosenElim ordinaryElim
  apply bind_mono (pick_mem _ _) ; intro pair _
  apply bind_mono (pick_mem _ _) ; intro minor _
  apply bind_mono (pick_mem _ _) ; intro major _
  apply bind_mono (pick_mem _ _) ; intro edge _ r hr
  obtain ⟨ae, member, value⟩ := List.mem_filterMap.mp hr
  exact List.mem_filterMap.mpr ⟨ae, pick_mem _ _ ae member, value⟩

theorem chosenIntro_subset (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (ri ei : Nat) :
    ∀ r ∈ chosenIntro d P v ri ei, r ∈ ordinaryIntro d P v := by
  unfold chosenIntro ordinaryIntro
  split
  · split
    · apply bind_mono (pick_mem _ _); intro route _ r hr
      obtain ⟨edge, member, value⟩ := List.mem_filterMap.mp hr
      exact List.mem_filterMap.mpr ⟨edge, pick_mem _ _ edge member, value⟩
    · exact fun _ hr => hr
  · exact fun _ hr => hr

theorem chosenMixed_subset (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (ui ri ei : Nat) :
    ∀ r ∈ chosenMixed d P v ui ri ei, r ∈ mixedIntroPairs d v P := by
  unfold chosenMixed mixedIntroPairs
  cases v.FORMULA with
  | atom name => simp
  | implication alpha beta =>
    apply bind_mono (pick_mem _ _); intro u _
    split
    · apply bind_mono (pick_mem _ _); intro route _ r hr
      obtain ⟨edge, member, value⟩ := List.mem_filterMap.mp hr
      exact List.mem_filterMap.mpr ⟨edge, pick_mem _ _ edge member, value⟩
    · exact fun _ hr => hr

theorem chosenElimReseed_subset (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (pi mi ai mei aei anci : Nat) :
    ∀ r ∈ chosenElimReseed d P v pi mi ai mei aei anci,
      r ∈ elimReseedPairs d v P (eligibleAncestorsAtRoot d v) := by
  unfold chosenElimReseed elimReseedPairs
  apply bind_mono (pick_mem _ _); intro pair _
  apply bind_mono (pick_mem _ _); intro minor _
  apply bind_mono (pick_mem _ _); intro major _
  apply bind_mono (pick_mem _ _); intro me _
  apply bind_mono (pick_mem _ _); intro ae _
  split
  · intro r hr
    obtain ⟨ancestor, member, value⟩ := List.mem_map.mp hr
    exact List.mem_map.mpr ⟨ancestor, pick_mem _ _ ancestor member, value⟩
  · exact fun _ hr => hr

theorem chosenIntroReseed_subset (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (ri ei anci : Nat) :
    ∀ r ∈ chosenIntroReseed d P v ri ei anci,
      r ∈ introReseedPairs d v P (eligibleAncestorsAtRoot d v) := by
  unfold chosenIntroReseed introReseedPairs
  split
  · rename_i hp hf
    simp only [hp, hf]
    split
    · apply bind_mono (pick_mem _ _); intro route _
      apply bind_mono (pick_mem _ _); intro edge _
      split
      · intro r hr
        obtain ⟨ancestor, member, value⟩ := List.mem_map.mp hr
        exact List.mem_map.mpr ⟨ancestor, pick_mem _ _ ancestor member, value⟩
      · intro r hr; simp at hr
    · intro r hr; simp at hr
  · intro r hr; simp at hr

def selectedRows (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (kind : SelectedKind) : List FlowPair :=
  if v.HYPOTHESIS || (predsOf d v).isEmpty then
    if (ancestorsInto d v).isEmpty then
      match kind with | .seedDefault => [(normalizeDep [v.FORMULA], [])] | _ => []
    else
      match kind with
      | .seedAncestor ai =>
          (pick (ancestorsInto d v) ai).map fun a => (normalizeDep [v.FORMULA], a.COLOURS)
      | _ => []
  else
    match kind with
    | .elimination pi mi ai mei aei =>
        if (elimPairsAt d v).isEmpty then [] else chosenElim d P v pi mi ai mei aei
    | .introduction ri ei =>
        if (elimPairsAt d v).isEmpty then chosenIntro d P v ri ei else []
    | .mixedIntroduction ui ri ei =>
        if (elimPairsAt d v).isEmpty then [] else chosenMixed d P v ui ri ei
    | .eliminationReseed pi mi ai mei aei anci =>
        if (eligibleAncestorsAtRoot d v).isEmpty then []
        else chosenElimReseed d P v pi mi ai mei aei anci
    | .introductionReseed ri ei anci =>
        if (eligibleAncestorsAtRoot d v).isEmpty then []
        else chosenIntroReseed d P v ri ei anci
    | _ => []

theorem selectedRows_sound (d : Graph) (P : Vertex → List FlowPair) (v : Vertex)
    (kind : SelectedKind) : ∀ route ∈ selectedRows d P v kind,
      route ∈ flowStepFrom d P v := by
  intro route member
  unfold selectedRows at member
  unfold flowStepFrom
  dsimp only
  split at member
  · rename_i source
    simp only [source, if_true]
    split at member
    · rename_i noAncestors
      simp only [noAncestors, if_true]
      cases kind <;> simp_all
    · rename_i hasAncestors
      simp only [hasAncestors, Bool.false_eq_true, if_false]
      cases kind <;> simp only at member
      all_goals try { simp at member }
      obtain ⟨ancestor, ha, value⟩ := List.mem_map.mp member
      exact List.mem_map.mpr ⟨ancestor, pick_mem _ _ ancestor ha, value⟩
  · rename_i notSource
    simp only [notSource, Bool.false_eq_true, if_false, List.mem_eraseDups, List.mem_append]
    cases kind <;> dsimp only at member
    case seedDefault => simp at member
    case seedAncestor ai => simp at member
    case elimination pi mi ai mei aei =>
      split at member
      · simp at member
      · rename_i h
        left
        simp only [h, Bool.false_eq_true, if_false, List.mem_append]
        exact Or.inl (chosenElim_subset d P v pi mi ai mei aei route member)
    case introduction ri ei =>
      split at member
      · rename_i h
        left
        simpa only [h, if_true, ordinaryIntro] using
          chosenIntro_subset d P v ri ei route member
      · simp at member
    case mixedIntroduction ui ri ei =>
      split at member
      · simp at member
      · rename_i h
        left
        simp only [h, Bool.false_eq_true, if_false, List.mem_append]
        exact Or.inr (chosenMixed_subset d P v ui ri ei route member)
    case eliminationReseed pi mi ai mei aei anci =>
      split at member
      · simp at member
      · rename_i h
        right
        simp only [reseedPairs, h, Bool.false_eq_true, if_false, List.mem_eraseDups, List.mem_append]
        exact Or.inl (chosenElimReseed_subset d P v pi mi ai mei aei anci route member)
    case introductionReseed ri ei anci =>
      split at member
      · simp at member
      · rename_i h
        right
        simp only [reseedPairs, h, Bool.false_eq_true, if_false, List.mem_eraseDups, List.mem_append]
        exact Or.inr (chosenIntroReseed_subset d P v ri ei anci route member)

def selectedJustificationB {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots) (slot : Fin slots) (kind : SelectedKind) : Bool :=
  let v := c.owner slot
  let source := v.HYPOTHESIS || (predsOf d v).isEmpty
  (match c.kind slot with
    | .source => source && (pointedSlots c slot).isEmpty
    | .step => !source) &&
  (pointedSlots c slot).all (fun pointer => decide (c.owner pointer ∈ predsOf d v)) &&
  decide (c.route slot ∈ WitnessableRoutes d v
    (selectedRows d (pointedRoutes c slot) v kind))

theorem selected_implies_old {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots) (slot : Fin slots) (kind : SelectedKind)
    (pass : selectedJustificationB c slot kind = true) :
    justificationB c slot = true := by
  simp only [selectedJustificationB, Bool.and_eq_true, List.all_eq_true,
    decide_eq_true_eq, WitnessableRoutes, List.mem_filter] at pass
  simp only [justificationB, Bool.and_eq_true, List.all_eq_true,
    decide_eq_true_eq, WitnessableRoutes, List.mem_filter]
  exact ⟨pass.1, selectedRows_sound _ _ _ _ _ pass.2.1, pass.2.2⟩

theorem selected_certificate_sound {d : Graph} {slots : Nat}
    (c : JustifiedCertificate d slots) (selected : Fin slots → SelectedKind)
    (structural : structuralValid d)
    (checked : ∀ slot, selectedJustificationB c slot (selected slot) = true) :
    ∀ slot, c.route slot ∈ WitnessableRoutes d (c.owner slot)
      (flowAt d (stdFuel d) (c.owner slot)) :=
  justification_sound c structural (fun slot => selected_implies_old c slot _ (checked slot))

end Robustness.SelectedProducers
