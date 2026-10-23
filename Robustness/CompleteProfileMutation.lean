import Infrastructure.RouteCertificateModel
import Robustness.FailureCylinderRobustness









open Semantic FlowSpec CertificateModel RouteCertificate

namespace CompleteProfileRobustness

def rowKeys (c : CompleteProfileCertificate) : Finset Vertex := (c.rows.map Prod.fst).toFinset

abbrev RowCoordinate (c : CompleteProfileCertificate) := ↥(rowKeys c)

def rowAlphabet (c : CompleteProfileCertificate) : Finset (List FlowPair) :=
  insert [] ((c.rows.map Prod.fst).map (stored c)).toFinset

abbrev RowAlphabet (c : CompleteProfileCertificate) := ↥(rowAlphabet c)
abbrev RowAssignment (c : CompleteProfileCertificate) := RowCoordinate c → RowAlphabet c

instance rowAlphabetNonempty (c : CompleteProfileCertificate) : Nonempty (RowAlphabet c) :=
  ⟨⟨[], by simp [rowAlphabet]⟩⟩

def rowAnchor (c : CompleteProfileCertificate) : RowAssignment c := fun key =>
  ⟨stored c key, by
    apply Finset.mem_insert_of_mem
    apply List.mem_toFinset.mpr
    exact List.mem_map.mpr ⟨key, List.mem_toFinset.mp key.2, rfl⟩⟩

def assignedProfile (c : CompleteProfileCertificate) (assignment : RowAssignment c)
    (v : Vertex) : List FlowPair :=
  if member : v ∈ rowKeys c then (assignment ⟨v, member⟩).val else []

def decodeRows (c : CompleteProfileCertificate) (assignment : RowAssignment c) : CompleteProfileCertificate :=
  ⟨c.graph, (c.rows.map Prod.fst).map fun v => (v, assignedProfile c assignment v)⟩

@[simp] theorem decodeRows_graph (c : CompleteProfileCertificate) (a : RowAssignment c) :
    (decodeRows c a).graph = c.graph := rfl

@[simp] theorem decodeRows_keys (c : CompleteProfileCertificate) (a : RowAssignment c) :
    (decodeRows c a).rows.map Prod.fst = c.rows.map Prod.fst := by
  simp [decodeRows, List.map_map, Function.comp_def]

private theorem lookup_absent (rows : FlowTable) (v : Vertex)
    (absent : v ∉ rows.map Prod.fst) : CompleteProfileKernel.lookup rows v = [] := by
  unfold CompleteProfileKernel.lookup flowTableLookup?
  have missing : rows.find? (fun row => row.1 == v) = none := by
    apply List.find?_eq_none.mpr
    intro row member
    have distinct : row.1 ≠ v := by
      intro equal
      exact absent (List.mem_map.mpr ⟨row, member, equal⟩)
    simp [distinct]
  rw [missing]
  rfl

theorem stored_decodeRows (c : CompleteProfileCertificate) (a : RowAssignment c) (v : Vertex) :
    stored (decodeRows c a) v = assignedProfile c a v := by
  by_cases member : v ∈ rowKeys c
  · simp only [stored, decodeRows, CompleteProfileKernel.lookup]
    rw [CacheCorrectness.flowTableLookup?_map_of_mem _ _
      (List.mem_toFinset.mp member)]
    rfl
  · change CompleteProfileKernel.lookup (decodeRows c a).rows v = assignedProfile c a v
    rw [lookup_absent]
    · simp [assignedProfile, member]
    · simpa [rowKeys] using member

theorem assignedProfile_anchor (c : CompleteProfileCertificate) (v : Vertex) :
    assignedProfile c (rowAnchor c) v = stored c v := by
  by_cases member : v ∈ rowKeys c
  · simp [assignedProfile, member, rowAnchor]
  · simp only [assignedProfile, member, dite_false]
    exact (lookup_absent c.rows v (by simpa [rowKeys] using member)).symm

@[simp] theorem stored_decodeRows_anchor (c : CompleteProfileCertificate) (v : Vertex) :
    stored (decodeRows c (rowAnchor c)) v = stored c v := by
  rw [stored_decodeRows, assignedProfile_anchor]



def rowReadSet (c : CompleteProfileCertificate) (vertices : List Vertex) : Finset (RowCoordinate c) :=
  Finset.univ.filter fun key => key.val ∈ vertices

theorem rowReadSet_card_le (c : CompleteProfileCertificate) (vertices : List Vertex) :
    (rowReadSet c vertices).card ≤ vertices.length := by
  have bound : (rowReadSet c vertices).card ≤ vertices.toFinset.card := by
    apply Finset.card_le_card_of_injOn (fun key => key.val)
    · intro key member
      exact List.mem_toFinset.mpr (Finset.mem_filter.mp member).2
    · intro left _ right _ equality
      exact Subtype.ext equality
  exact bound.trans (List.toFinset_card_le _)

theorem stored_decodeRows_agree (c : CompleteProfileCertificate) (vertices : List Vertex)
    (left right : RowAssignment c)
    (agree : SearchProblem.CertificateRobustness.AgreeOn
      (rowReadSet c vertices) left right) :
    ∀ v ∈ vertices, stored (decodeRows c left) v = stored (decodeRows c right) v := by
  intro v member
  rw [stored_decodeRows, stored_decodeRows]
  by_cases existsKey : v ∈ rowKeys c
  · have equality := agree ⟨v, existsKey⟩ (by simp [rowReadSet, member])
    simp only [assignedProfile, existsKey, dite_true]
    exact congrArg Subtype.val equality.symm
  · simp [assignedProfile, existsKey]

#print axioms stored_decodeRows_agree
end CompleteProfileRobustness
