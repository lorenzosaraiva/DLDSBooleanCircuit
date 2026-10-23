import Infrastructure.FlowValidity
import Infrastructure.FlowTreeProof
set_option linter.unusedSimpArgs false





open Semantic

namespace FlowSpec





def ReseedFree (d : Graph) : Prop :=
  ∀ p ∈ d.PATHS, get_rule.incoming p.END d = []

def reseedFreeB (d : Graph) : Bool :=
  d.PATHS.all fun p => (get_rule.incoming p.END d).isEmpty

lemma ancestorsInto_eq_nil_of_reseedFree (d : Graph) (hreseed : ReseedFree d)
    {v : Vertex} (hincoming : get_rule.incoming v d ≠ []) :
    ancestorsInto d v = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro path hpath
  have hpathGraph : path ∈ d.PATHS := (List.mem_filter.mp hpath).1
  have hend : path.END = v := by simpa using (List.mem_filter.mp hpath).2
  have htop := hreseed path hpathGraph
  rw [hend] at htop
  exact hincoming htop

def premiseDepUnion (d : Graph) (v : Vertex) : Dep :=
  match elimPairsAt d v with
  | [(M, S)] =>
      match edgesBetween d S v, edgesBetween d M v with
      | eS :: _, eM :: _ => depUnion eS.DEPENDENCY eM.DEPENDENCY
      | _, _ => normalizeDep []
  | _ => normalizeDep []


def OneEdgePerColourPerNode (d : Graph) : Prop :=
  ∀ e1 e2, e1 ∈ d.EDGES → e2 ∈ d.EDGES →
    (e1.START = e2.START ∧ e1.COLOUR = e2.COLOUR) → e1 = e2


def RouteFanUnique (d : Graph) : Prop :=
  ∀ v ∈ d.NODES, ∀ r1 r2,
    r1 ∈ flowAt d (stdFuel d) v →
    r2 ∈ flowAt d (stdFuel d) v →
    r1.2.headD 0 = r2.2.headD 0 → r1 = r2








def elimCombine (d : Graph) (v M S : Vertex) (FS FM : List FlowPair) :
    List FlowPair :=
  FS.flatMap (fun bp₁ =>
    FM.flatMap (fun bp₂ =>
      (edgesBetween d S v).flatMap (fun e₁ =>
        (edgesBetween d M v).filterMap (fun e₂ =>
          match consume e₁ bp₁.2, consume e₂ bp₂.2 with
          | some q₁, some q₂ =>
              if q₁ = q₂ then some (depUnion bp₁.1 bp₂.1, q₁) else none
          | _, _ => none))))

lemma core_branch {α : Type _} [BEq α] (elim intr : List α)
    (extra : List α → List α) (hextra : ∀ l, extra l = [])
    (hne : elim ≠ []) :
    ((if elim.isEmpty then intr else elim) ++
        extra (if elim.isEmpty then intr else elim)).eraseDups =
      elim.eraseDups := by
  have h : elim.isEmpty = false := by
    cases h : elim.isEmpty with
    | true => exact absurd (List.isEmpty_iff.mp h) hne
    | false => rfl
  rw [h]
  simp [hextra]

lemma core_branch_with_extra {α : Type _} [BEq α]
    (elim intr extra : List α) (hne : elim ≠ []) :
    ((if elim.isEmpty then intr else elim) ++ extra).eraseDups =
      (elim ++ extra).eraseDups := by
  have h : elim.isEmpty = false := by
    cases h : elim.isEmpty with
    | true => exact absurd (List.isEmpty_iff.mp h) hne
    | false => rfl
  simp [h]










lemma flowAt_collapsed_elim (d : Graph) {v M S : Vertex} (fuel : Nat)
    (hhyp : v.HYPOTHESIS = false)
    (hpredsne : (predsOf d v).isEmpty = false)
    (hpairs : elimPairsAt d v = [(M, S)]) :
    flowAt d (fuel + 1) v =
      (elimCombine d v M S (flowAt d fuel S) (flowAt d fuel M) ++
        mixedIntroPairs d v (flowAt d fuel) ++
        reseedPairs d v (flowAt d fuel)).eraseDups := by
  rw [flowAt, hhyp]
  simp only [Bool.false_or, hpredsne, Bool.false_eq_true, if_false]
  rw [hpairs]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.isEmpty_cons, Bool.false_eq_true, if_false]
  rfl



def routeFanUniqueAtB (d : Graph) (v : Vertex) : Bool :=
  let F := flowAt d (stdFuel d) v
  F.all fun r1 =>
    F.all fun r2 =>
      if r1.2.headD 0 == r2.2.headD 0 then decide (r1 = r2) else true


end FlowSpec
