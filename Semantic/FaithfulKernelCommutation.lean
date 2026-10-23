import Semantic.FaithfulSummaryAlgebra
import Semantic.FaithfulDependencyBlind

open Semantic FlowSpec

namespace FaithfulKeyed

private def introRun (xs : List FlowPair) (a : Formula)
    (edges : List Deduction) (test : Deduction → Bool) : List FlowPair :=
  xs.flatMap fun r => edges.filterMap fun e =>
    if test e then (consume e r.2).map fun p => (depRemove r.1 a, p) else none

private theorem introRun_eq (xs : List FlowPair) (a : Formula)
    (edges : List Deduction) (test : Deduction → Bool) :
    introRun xs a edges test = unary xs a
      (fun p => edges.filterMap fun e => if test e then consume e p else none) := by
  unfold introRun unary
  apply congrArg (fun f : FlowPair → List FlowPair => xs.flatMap f)
  funext r
  rw [List.map_filterMap]
  apply congrArg (fun f : Deduction → Option FlowPair => edges.filterMap f)
  funext e
  by_cases h : test e = true <;> simp [h]

private theorem introRun_congr {xs ys : List FlowPair} (h : SummaryEq xs ys)
    (a : Formula) (edges : List Deduction) (test : Deduction → Bool) :
    SummaryEq (introRun xs a edges test) (introRun ys a edges test) := by
  rw [introRun_eq, introRun_eq]
  exact h.unary _ _

private def ordinaryNext (minor major : List Deduction) (test : Deduction → Bool)
    (p q : ColourPath) : List ColourPath :=
  minor.flatMap fun em => major.filterMap fun eM =>
    if test em && test eM then
      match consume em p, consume eM q with
      | some s, some t => if s = t then some s else none
      | _, _ => none
    else none

private def ordinaryRun (xs ys : List FlowPair) (minor major : List Deduction)
    (test : Deduction → Bool) : List FlowPair :=
  xs.flatMap fun l => ys.flatMap fun r =>
    minor.flatMap fun em => major.filterMap fun eM =>
      if test em && test eM then
        match consume em l.2, consume eM r.2 with
        | some s, some t => if s = t then some (depUnion l.1 r.1, s) else none
        | _, _ => none
      else none

private theorem ordinaryRun_eq (xs ys : List FlowPair) (minor major : List Deduction)
    (test : Deduction → Bool) :
    ordinaryRun xs ys minor major test = binary xs ys (ordinaryNext minor major test) := by
  unfold ordinaryRun binary
  apply congrArg (fun f : FlowPair → List FlowPair => xs.flatMap f)
  funext l
  apply congrArg (fun f : FlowPair → List FlowPair => ys.flatMap f)
  funext r
  simp only [ordinaryNext, List.map_flatMap]
  apply congrArg (fun f : Deduction → List FlowPair => minor.flatMap f)
  funext em
  rw [List.map_filterMap]
  apply congrArg (fun f : Deduction → Option FlowPair => major.filterMap f)
  funext eM
  by_cases h : (test em && test eM) = true
  · simp only [h, if_true]
    cases hs : consume em l.2 <;> cases ht : consume eM r.2 <;> simp [hs, ht]
    all_goals split <;> simp_all
  · simp [h]

private theorem ordinaryRun_congr {xs ys us vs : List FlowPair}
    (h : SummaryEq xs ys) (k : SummaryEq us vs)
    (minor major : List Deduction) (test : Deduction → Bool) :
    SummaryEq (ordinaryRun xs us minor major test) (ordinaryRun ys vs minor major test) := by
  rw [ordinaryRun_eq, ordinaryRun_eq]
  exact h.binary k _

private def reseedRun (xs ys : List FlowPair) (minor major : List Deduction)
    (ancestors : List Ancestral) (test : Deduction → Bool) : List FlowPair :=
  xs.flatMap fun l => ys.flatMap fun r =>
    minor.flatMap fun em => major.flatMap fun eM =>
      if (consume em l.2).isSome && (consume eM r.2).isSome && test em && test eM then
        ancestors.map fun a => (depUnion l.1 r.1, a.COLOURS)
      else []

private theorem reseedRun_eq (xs ys : List FlowPair) (minor major : List Deduction)
    (ancestors : List Ancestral) (test : Deduction → Bool) :
    reseedRun xs ys minor major ancestors test = binary xs ys (fun p q =>
      minor.flatMap fun em => major.flatMap fun eM =>
        if (consume em p).isSome && (consume eM q).isSome && test em && test eM then
          ancestors.map Ancestral.COLOURS else []) := by
  unfold reseedRun binary
  apply congrArg (fun f : FlowPair → List FlowPair => xs.flatMap f)
  funext l
  apply congrArg (fun f : FlowPair → List FlowPair => ys.flatMap f)
  funext r
  simp only [List.map_flatMap]
  apply congrArg (fun f : Deduction → List FlowPair => minor.flatMap f)
  funext em
  apply congrArg (fun f : Deduction → List FlowPair => major.flatMap f)
  funext eM
  split <;> simp_all [List.map_map]

private theorem reseedRun_congr {xs ys us vs : List FlowPair}
    (h : SummaryEq xs ys) (k : SummaryEq us vs) (minor major : List Deduction)
    (ancestors : List Ancestral) (test : Deduction → Bool) :
    SummaryEq (reseedRun xs us minor major ancestors test)
      (reseedRun ys vs minor major ancestors test) := by
  rw [reseedRun_eq, reseedRun_eq]
  exact h.binary k _

private def introReseedRun (xs : List FlowPair) (a : Formula)
    (edges : List Deduction) (ancestors : List Ancestral) (test : Deduction → Bool) : List FlowPair :=
  xs.flatMap fun r => edges.flatMap fun e =>
    if (e.COLOUR == 0 && (consume e r.2).isSome) && test e then
      ancestors.map fun anc => (depRemove r.1 a, anc.COLOURS)
    else []

private theorem introReseedRun_eq (xs : List FlowPair) (a : Formula)
    (edges : List Deduction) (ancestors : List Ancestral) (test : Deduction → Bool) :
    introReseedRun xs a edges ancestors test = unary xs a (fun p =>
      edges.flatMap fun e =>
        if (e.COLOUR == 0 && (consume e p).isSome) && test e then
          ancestors.map Ancestral.COLOURS else []) := by
  unfold introReseedRun unary
  apply congrArg (fun f : FlowPair → List FlowPair => xs.flatMap f)
  funext r
  simp only [List.map_flatMap]
  apply congrArg (fun f : Deduction → List FlowPair => edges.flatMap f)
  funext e
  split <;> simp_all [List.map_map]

private theorem introReseedRun_congr {xs ys : List FlowPair} (h : SummaryEq xs ys)
    (a : Formula) (edges : List Deduction) (ancestors : List Ancestral)
    (test : Deduction → Bool) :
    SummaryEq (introReseedRun xs a edges ancestors test)
      (introReseedRun ys a edges ancestors test) := by
  rw [introReseedRun_eq, introReseedRun_eq]
  exact h.unary _ _

theorem mixedWith_summary (d : Graph) (test : Deduction → Bool)
    (left right : Vertex → List FlowPair) (h : ∀ v, SummaryEq (left v) (right v))
    (v : Vertex) :
    SummaryEq (LambdaFlow.mixedWith d (fun _ e => test e) v left)
      (LambdaFlow.mixedWith d (fun _ e => test e) v right) := by
  unfold LambdaFlow.mixedWith
  split
  · apply SummaryEq.flatMap
    intro u hu
    split
    · exact introRun_congr (h u) _ _ _
    · exact SummaryEq.refl _
  · exact SummaryEq.refl _

theorem elimReseedWith_summary (d : Graph) (test : Deduction → Bool)
    (left right : Vertex → List FlowPair) (h : ∀ v, SummaryEq (left v) (right v))
    (v : Vertex) (ancestors : List Ancestral) :
    SummaryEq (LambdaFlow.elimReseedWith d (fun _ e => test e) v left ancestors)
      (LambdaFlow.elimReseedWith d (fun _ e => test e) v right ancestors) := by
  unfold LambdaFlow.elimReseedWith
  apply SummaryEq.flatMap
  intro pair hp
  exact reseedRun_congr (h pair.2) (h pair.1) _ _ _ _

theorem introReseedWith_summary (d : Graph) (test : Deduction → Bool)
    (left right : Vertex → List FlowPair) (h : ∀ v, SummaryEq (left v) (right v))
    (v : Vertex) (ancestors : List Ancestral) :
    SummaryEq (LambdaFlow.introReseedWith d (fun _ e => test e) v left ancestors)
      (LambdaFlow.introReseedWith d (fun _ e => test e) v right ancestors) := by
  unfold LambdaFlow.introReseedWith
  split
  · split
    · exact introReseedRun_congr (h _) _ _ _ _
    · exact SummaryEq.refl _
  · exact SummaryEq.refl _

theorem reseedWith_summary (d : Graph) (test : Deduction → Bool)
    (left right : Vertex → List FlowPair) (h : ∀ v, SummaryEq (left v) (right v))
    (v : Vertex) :
    SummaryEq (LambdaFlow.reseedWith d (fun _ e => test e) v left)
      (LambdaFlow.reseedWith d (fun _ e => test e) v right) := by
  unfold LambdaFlow.reseedWith
  dsimp only
  split
  · exact SummaryEq.refl _
  · apply SummaryEq.eraseDups
    exact (elimReseedWith_summary d test left right h v _).append
      (introReseedWith_summary d test left right h v _)

private theorem step_edgeOnly_summary (d : Graph) (ordinary reseed : Deduction → Bool)
    (left right : Vertex → List FlowPair) (h : ∀ v, SummaryEq (left v) (right v))
    (v : Vertex) :
    SummaryEq (LambdaFlow.stepWith d (fun _ e => ordinary e) (fun _ e => reseed e) left v)
      (LambdaFlow.stepWith d (fun _ e => ordinary e) (fun _ e => reseed e) right v) := by
  unfold LambdaFlow.stepWith
  dsimp only
  split
  · exact SummaryEq.refl _
  · apply SummaryEq.eraseDups
    apply SummaryEq.append
    · split
      · split
        · split
          · exact introRun_congr (h _) _ _ _
          · exact SummaryEq.refl _
        · exact SummaryEq.refl _
      · apply SummaryEq.append
        · apply SummaryEq.flatMap
          intro pair hp
          exact ordinaryRun_congr (h pair.2) (h pair.1) _ _ _
        · exact mixedWith_summary d ordinary left right h v
    · exact reseedWith_summary d reseed left right h v





theorem stepWith_summary (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (ho : DepBlind ordinary) (hr : DepBlind reseed)
    (left right : Vertex → List FlowPair) (h : ∀ v, SummaryEq (left v) (right v))
    (v : Vertex) :
    SummaryEq (LambdaFlow.stepWith d ordinary reseed left v)
      (LambdaFlow.stepWith d ordinary reseed right v) := by
  have eo : ordinary = fun _ e => ordinary [] e := by
    funext deps edge
    exact ho deps [] edge
  have er : reseed = fun _ e => reseed [] e := by
    funext deps edge
    exact hr deps [] edge
  rw [eo, er]
  exact step_edgeOnly_summary d _ _ left right h v



theorem stepWith_summary_on_preds (d : Graph) (ordinary reseed : LambdaFlow.Guard)
    (ho : DepBlind ordinary) (hr : DepBlind reseed)
    (left right : Vertex → List FlowPair) (v : Vertex)
    (h : ∀ u ∈ predsOf d v, SummaryEq (left u) (right u)) :
    SummaryEq (LambdaFlow.stepWith d ordinary reseed left v)
      (LambdaFlow.stepWith d ordinary reseed right v) := by
  let patched : Vertex → List FlowPair :=
    fun u => if u ∈ predsOf d v then right u else left u
  have localSummary : ∀ u, SummaryEq (left u) (patched u) := by
    intro u
    by_cases member : u ∈ predsOf d v
    · simpa only [patched, member, if_true] using h u member
    · simp only [patched, member, if_false]
      exact SummaryEq.refl _
  have eq := LambdaFlow.stepWith_congr_of_preds d ordinary reseed v patched right
    (fun u member => by simp only [patched, member, if_true])
  rw [← eq]
  exact stepWith_summary d ordinary reseed ho hr left patched localSummary v

#print axioms mixedWith_summary
#print axioms elimReseedWith_summary
#print axioms introReseedWith_summary
#print axioms reseedWith_summary
#print axioms stepWith_summary
#print axioms stepWith_summary_on_preds

end FaithfulKeyed
