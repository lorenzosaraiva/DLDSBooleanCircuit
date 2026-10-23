import Semantic.FlowSize
import Semantic.MultiIntroPositions
import Mathlib.Data.Finset.Card

open Semantic FlowSpec

namespace FlowSize

set_option maxHeartbeats 4000000


def nonemptySuffixes : ColourPath → List ColourPath
  | [] => []
  | head :: rest => (head :: rest) :: nonemptySuffixes rest

def residuals (d : Graph) : List ColourPath :=
  [] :: d.PATHS.flatMap (fun a => nonemptySuffixes a.COLOURS)

def wordTotal (d : Graph) : Nat := (d.PATHS.map (fun a => a.COLOURS.length)).sum
def wordMax (d : Graph) : Nat := (d.PATHS.map (fun a => a.COLOURS.length)).foldr max 0
def formulaSet (d : Graph) : Finset Formula := (RefinedEval.formulas d).toFinset
def formulaCount (d : Graph) : Nat := (formulaSet d).card
def distinctBound (d : Graph) : Nat := 2 ^ formulaCount d * (1 + wordTotal d)
def profileBound (d : Graph) : Nat := max d.PATHS.length (distinctBound d)


def routeUnits (r : FlowPair) : Nat := 1 + r.1.length + r.2.length
def profileUnits (routes : List FlowPair) : Nat := (routes.map routeUnits).sum
def computedUnits (d : Graph) : Nat :=
  (d.NODES.map (fun v => profileUnits (MultiIntro.evaluateAt .general d (stdFuel d) v))).sum
def certificateUnits (c : RouteCertificate.CompleteProfileCertificate) : Nat :=
  (c.rows.map (fun row => profileUnits row.2)).sum
def sizeBound (d : Graph) : Nat :=
  d.NODES.length * profileBound d * (1 + formulaCount d + wordMax d)

theorem nonemptySuffixes_length (word : ColourPath) :
    (nonemptySuffixes word).length = word.length := by
  induction word with
  | nil => rfl
  | cons head rest ih => simp [nonemptySuffixes, ih]

theorem suffix_member (word tail : ColourPath) (notEmpty : tail ≠ [])
    (suffix : tail <:+ word) : tail ∈ nonemptySuffixes word := by
  obtain ⟨front, rfl⟩ := suffix
  induction front with
  | nil => cases tail <;> simp_all [nonemptySuffixes]
  | cons head rest ih => simp [nonemptySuffixes, ih]

theorem residual_member (d : Graph) (word : ColourPath) (allowed : ResidualAllowed d word) :
    word ∈ residuals d := by
  rcases allowed with rfl | ⟨a, ha, suffix⟩
  · simp [residuals]
  · by_cases empty : word = []
    · simp [residuals, empty]
    · exact List.mem_cons_of_mem _ (List.mem_flatMap.mpr ⟨a, ha, suffix_member _ _ empty suffix⟩)

theorem residuals_length (d : Graph) : (residuals d).length = 1 + wordTotal d := by
  simp [residuals, List.length_flatMap, nonemptySuffixes_length, wordTotal, Nat.add_comm]

theorem member_le_foldMax (values : List Nat) (n : Nat) (member : n ∈ values) :
    n ≤ values.foldr max 0 := by
  induction values with
  | nil => contradiction
  | cons head rest ih =>
    rcases List.mem_cons.mp member with rfl | member
    · exact Nat.le_max_left _ _
    · exact (ih member).trans (Nat.le_max_right _ _)

theorem residual_length (d : Graph) (word : ColourPath) (allowed : ResidualAllowed d word) :
    word.length ≤ wordMax d := by
  rcases allowed with rfl | ⟨a, ha, suffix⟩
  · exact Nat.zero_le _
  · exact suffix.length_le.trans (member_le_foldMax _ _ (List.mem_map.mpr ⟨a, ha, rfl⟩))

theorem dependency_length (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) (r : FlowPair)
    (member : r ∈ MultiIntro.evaluateAt variant d fuel v) : r.1.length ≤ formulaCount d := by
  obtain ⟨canonical, supported⟩ := dependency_support variant d fuel v listed r member
  have sub : r.1.toFinset ⊆ formulaSet d := by
    intro f hf
    exact List.mem_toFinset.mpr (supported f (List.mem_toFinset.mp hf))
  simpa only [formulaCount, List.toFinset_card_of_nodup canonical.1] using Finset.card_le_card sub

theorem distinct_route_bound (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) :
    (MultiIntro.evaluateAt variant d fuel v).toFinset.card ≤ distinctBound d := by
  let routes := (MultiIntro.evaluateAt variant d fuel v).toFinset
  let code := fun r : FlowPair => (r.1.toFinset, r.2)
  let target : Finset (Finset Formula × ColourPath) := (formulaSet d).powerset ×ˢ (residuals d).toFinset
  have maps : Set.MapsTo code (routes : Set FlowPair)
      (target : Set (Finset Formula × ColourPath)) := by
    intro r member
    have present : r ∈ MultiIntro.evaluateAt variant d fuel v := List.mem_toFinset.mp member
    obtain ⟨_, supported⟩ := dependency_support variant d fuel v listed r present
    apply Finset.mem_product.mpr
    refine ⟨Finset.mem_powerset.mpr ?_, List.mem_toFinset.mpr ?_⟩
    · intro f hf
      exact List.mem_toFinset.mpr (supported f (List.mem_toFinset.mp hf))
    · exact residual_member d r.2 (residual_origin variant d fuel v r present)
  have injective : (routes : Set FlowPair).InjOn code := by
    intro left hl right hr same
    have hleft := (dependency_support variant d fuel v listed left (List.mem_toFinset.mp hl)).1
    have hright := (dependency_support variant d fuel v listed right (List.mem_toFinset.mp hr)).1
    apply Prod.ext
    · apply canonicalDep_eq_of_depSetEq hleft hright
      apply (depSetEq_eq_true_iff _ _).mpr
      intro f
      have sets := congrArg Prod.fst same
      change left.1.toFinset = right.1.toFinset at sets
      simpa only [List.mem_toFinset] using (show f ∈ left.1.toFinset ↔ f ∈ right.1.toFinset from sets ▸ Iff.rfl)
    · exact congrArg (fun p : Finset Formula × ColourPath => p.2) same
  have count := Finset.card_le_card_of_injOn code maps injective
  calc
    routes.card ≤ ((formulaSet d).powerset ×ˢ (residuals d).toFinset).card := count
    _ = 2 ^ formulaCount d * (residuals d).toFinset.card := by
      rw [Finset.card_product, Finset.card_powerset]; rfl
    _ ≤ 2 ^ formulaCount d * (1 + wordTotal d) :=
      Nat.mul_le_mul_left _ ((List.toFinset_card_le (residuals d)).trans_eq (residuals_length d))

private theorem nodup_eraseDups {α : Type} [BEq α] [LawfulBEq α] :
    ∀ values : List α, values.eraseDups.Nodup
  | [] => by simp
  | value :: rest => by
    rw [List.eraseDups_cons]
    constructor
    · intro candidate member equal
      rw [List.mem_eraseDups, List.mem_filter] at member
      subst candidate
      simp at member
    · exact nodup_eraseDups (rest.filter fun candidate => !candidate == value)
termination_by values => values.length
decreasing_by simpa using List.length_filter_le (fun candidate => !candidate == value) rest

theorem step_nodup_or_source (variant : MultiIntro.Variant) (d : Graph)
    (prior : Vertex → List FlowPair) (v : Vertex) :
    (MultiIntro.step variant d prior v).Nodup ∨
      (MultiIntro.step variant d prior v).length ≤ max 1 d.PATHS.length := by
  unfold MultiIntro.step
  split
  · exact Or.inl (nodup_eraseDups _)
  · change (LambdaFlow.stepWith d _ _ prior v).Nodup ∨
      (LambdaFlow.stepWith d _ _ prior v).length ≤ _
    simp only [LambdaFlow.stepWith]
    split
    · right
      split
      · simpa using Nat.le_max_left 1 d.PATHS.length
      · simpa only [List.length_map] using
          (List.length_filter_le _ d.PATHS).trans (Nat.le_max_right 1 d.PATHS.length)
    · exact Or.inl (nodup_eraseDups _)

theorem raw_route_bound (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) :
    (MultiIntro.evaluateAt variant d fuel v).length ≤ profileBound d := by
  have positive : 1 ≤ distinctBound d := by
    unfold distinctBound
    have : 0 < 2 ^ formulaCount d := pow_pos (by decide) _
    have : 0 < 2 ^ formulaCount d * (1 + wordTotal d) := Nat.mul_pos this (by omega)
    omega
  cases fuel with
  | zero => exact Nat.zero_le _
  | succ fuel =>
    rcases step_nodup_or_source variant d (MultiIntro.evaluateAt variant d fuel) v with nodup | small
    · have count := distinct_route_bound variant d (fuel+1) v listed
      change (MultiIntro.evaluateAt variant d (fuel+1) v).Nodup at nodup
      rw [List.toFinset_card_of_nodup nodup] at count
      exact count.trans (Nat.le_max_right _ _)
    · exact small.trans (max_le (positive.trans (Nat.le_max_right _ _)) (Nat.le_max_left _ _))

theorem route_unit_bound (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) (r : FlowPair)
    (member : r ∈ MultiIntro.evaluateAt variant d fuel v) :
    routeUnits r ≤ 1 + formulaCount d + wordMax d := by
  have dep := dependency_length variant d fuel v listed r member
  have res := residual_length d r.2 (residual_origin variant d fuel v r member)
  unfold routeUnits
  omega

theorem sum_map_le_length_mul {α : Type} (values : List α) (f : α → Nat) (bound : Nat)
    (bounded : ∀ x ∈ values, f x ≤ bound) : (values.map f).sum ≤ values.length * bound := by
  induction values with
  | nil => simp
  | cons head rest ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.succ_mul]
    have h := bounded head (by simp)
    have t := ih (fun x hx => bounded x (by simp [hx]))
    omega

theorem profile_unit_bound (variant : MultiIntro.Variant) (d : Graph) (fuel : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) :
    profileUnits (MultiIntro.evaluateAt variant d fuel v) ≤
      profileBound d * (1 + formulaCount d + wordMax d) := by
  exact (sum_map_le_length_mul _ routeUnits _ (route_unit_bound variant d fuel v listed)).trans
    (Nat.mul_le_mul_right _ (raw_route_bound variant d fuel v listed))

theorem computed_size_bound (d : Graph) : computedUnits d ≤ sizeBound d := by
  simpa only [computedUnits, sizeBound, Nat.mul_assoc] using
    sum_map_le_length_mul d.NODES (fun v => profileUnits (MultiIntro.evaluateAt .general d (stdFuel d) v))
      (profileBound d * (1 + formulaCount d + wordMax d))
      (fun v listed => profile_unit_bound .general d _ v listed)

theorem generated_rows (d : Graph) (domain : MultiIntro.MIDomain d) :
    (MultiIntro.generateCertificate .general d).rows =
      d.NODES.map (fun v => (v, MultiIntro.evaluateAt .general d (stdFuel d) v)) := by
  simp only [MultiIntro.generateCertificate, RouteCertificate.generateFor,
    MultiIntro.kernel_memo_eq_spec, domain, CompleteProfileKernel.specificationTable, MultiIntro.kernel_evaluate]

theorem generated_size_bound (d : Graph) (valid : _root_.Flow d) :
    certificateUnits (MultiIntro.generateCertificate .general d) ≤ sizeBound d := by
  rw [certificateUnits, generated_rows d valid.1]
  simpa only [List.map_map, Function.comp_def, computedUnits] using computed_size_bound d

theorem exists_bounded_certificate (d : Graph) (valid : _root_.Flow d) :
    ∃ c, MultiIntro.StrongCertificate .general d c ∧ certificateUnits c ≤ sizeBound d :=
  ⟨MultiIntro.generateCertificate .general d, MultiIntro.generated_formation .general d valid,
    generated_size_bound d valid⟩

theorem formulaSet_eq_nodes (d : Graph) (domain : MultiIntro.MIDomain d) :
    formulaSet d = (d.NODES.map (fun v => v.FORMULA)).toFinset := by
  ext f
  simp only [formulaSet, List.mem_toFinset, RefinedEval.formulas, mem_normalizeDep,
    List.mem_append]
  constructor
  · rintro (member | member)
    · exact member
    · obtain ⟨e, he, member⟩ := List.mem_flatMap.mp member
      have ends := MultiIntro.domain_endpoints d domain e he
      rcases (by simpa using member : f = e.START.FORMULA ∨ f = e.END.FORMULA) with rfl | rfl
      · exact List.mem_map.mpr ⟨e.START, ends.1, rfl⟩
      · exact List.mem_map.mpr ⟨e.END, ends.2, rfl⟩
  · exact Or.inl

theorem formulaCount_le_vertices (d : Graph) (domain : MultiIntro.MIDomain d) :
    formulaCount d ≤ d.NODES.length := by
  unfold formulaCount
  rw [formulaSet_eq_nodes d domain]
  exact (List.toFinset_card_le _).trans_eq (List.length_map _)

theorem foldMax_le_sum (values : List Nat) : values.foldr max 0 ≤ values.sum := by
  induction values with
  | nil => rfl
  | cons head rest ih =>
    simp only [List.foldr_cons, List.sum_cons]
    exact max_le (Nat.le_add_right _ _) (ih.trans (Nat.le_add_left _ _))

theorem wordMax_le_total (d : Graph) : wordMax d ≤ wordTotal d := foldMax_le_sum _

def coarseBound (d : Graph) : Nat :=
  d.NODES.length * max d.PATHS.length (2 ^ d.NODES.length * (1 + wordTotal d)) *
    (1 + d.NODES.length + wordTotal d)

theorem sizeBound_le_coarse (d : Graph) (domain : MultiIntro.MIDomain d) :
    sizeBound d ≤ coarseBound d := by
  have formulas := formulaCount_le_vertices d domain
  have words := wordMax_le_total d
  unfold sizeBound coarseBound profileBound distinctBound
  apply Nat.mul_le_mul
  · apply Nat.mul_le_mul_left
    exact max_le_max (Nat.le_refl _) (Nat.mul_le_mul_right _ (Nat.pow_le_pow_right (by decide) formulas))
  · omega

theorem flow_has_bounded_certificate (d : Graph) (valid : _root_.Flow d) :
    ∃ c, MultiIntro.StrongCertificate .general d c ∧ certificateUnits c ≤ coarseBound d := by
  obtain ⟨c, formed, bounded⟩ := exists_bounded_certificate d valid
  exact ⟨c, formed, bounded.trans (sizeBound_le_coarse d valid.1)⟩

theorem stored_profile_bound (d : Graph) (c : RouteCertificate.CompleteProfileCertificate)
    (formed : MultiIntro.StrongCertificate .general d c) (v : Vertex) (listed : v ∈ d.NODES) :
    profileUnits (RouteCertificate.stored c v) ≤ profileBound d * (1 + formulaCount d + wordMax d) := by
  rw [MultiIntro.stored_eq_routes .general d c formed v listed]
  exact profile_unit_bound .general d _ v listed

theorem production_read_units (d : Graph) (c : RouteCertificate.CompleteProfileCertificate)
    (formed : MultiIntro.StrongCertificate .general d c) (slot : Nat)
    (v : Vertex) (listed : v ∈ d.NODES) :
    ((MultiIntroPositions.readVertices c (.productionRow slot v)).map
      (fun u => profileUnits (RouteCertificate.stored c u))).sum ≤
        (1 + (predsOf d v).length) * profileBound d * (1 + formulaCount d + wordMax d) := by
  simp only [MultiIntroPositions.readVertices, formed.1.1]
  have readListed : ∀ u ∈ v :: predsOf d v, u ∈ d.NODES := by
    intro u member
    rcases List.mem_cons.mp member with rfl | pred
    · exact listed
    · exact (MultiIntro.predecessor_props d formed.2.1.1.1 pred).1
  simpa only [List.length_cons, Nat.add_comm 1, Nat.mul_assoc] using
    sum_map_le_length_mul (v :: predsOf d v)
      (fun u => profileUnits (RouteCertificate.stored c u))
      (profileBound d * (1 + formulaCount d + wordMax d))
      (fun u member => stored_profile_bound d c formed u (readListed u member))

end FlowSize
