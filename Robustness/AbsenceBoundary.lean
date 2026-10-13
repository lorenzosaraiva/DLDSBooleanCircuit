import Robustness.Certificate.FailureCylinders

/-!
An abstract route-list model, not a DLDS-certificate instantiation.
Each of n Boolean slots can supply one distinguished required route. Failure
means that every slot is empty; any unfixed slot can repair that absence.
-/

namespace CertificateSearch.Robustness.AbstractRouteAbsence

def missingRouteB (n : Nat) (input : Assignment (Fin n) Bool) : Bool :=
  decide (∀ i, input i = false)

def emptyRouteList (n : Nat) : Assignment (Fin n) Bool := fun _ => false

theorem emptyRouteList_fails (n : Nat) :
    missingRouteB n (emptyRouteList n) = true := by
  simp [missingRouteB, emptyRouteList]

theorem absence_forces_all_slots (n : Nat) (support : Finset (Fin n))
    (forces : ForcesFailure (missingRouteB n) (emptyRouteList n) support) :
    support = Finset.univ := by
  classical
  apply Finset.eq_univ_of_forall
  intro i
  by_contra outside
  let repaired : Assignment (Fin n) Bool := fun j => if j = i then true else false
  have agree : AgreeOn support (emptyRouteList n) repaired := by
    intro j member
    have different : j ≠ i := by
      intro same
      exact outside (same ▸ member)
    simp [repaired, emptyRouteList, different]
  have failure := forces repaired agree
  have allEmpty : ∀ j, repaired j = false := by
    simpa [missingRouteB] using failure
  have impossible := allEmpty i
  simp [repaired] at impossible

theorem missingRoute_minimumFailureWitnessSize (n : Nat) :
    minimumFailureWitnessSize (missingRouteB n) (emptyRouteList n)
      (emptyRouteList_fails n) = n := by
  classical
  obtain ⟨size, forces⟩ := minimumFailureSupport_spec
    (missingRouteB n) (emptyRouteList n) (emptyRouteList_fails n)
  rw [absence_forces_all_slots n _ forces, Finset.card_univ, Fintype.card_fin] at size
  exact size.symm


end CertificateSearch.Robustness.AbstractRouteAbsence
