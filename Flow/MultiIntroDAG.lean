import Flow.MultiIntroLocality



open Semantic FlowSpec CertificateModel
namespace MultiIntro
set_option maxHeartbeats 3000000

theorem domain_levels (d : Graph) (domain : MIDomain d) : LeveledColored d := by
  simp only [MIDomain, miDomainB, Bool.and_eq_true] at domain
  have check := domain.1.1
  intro e member
  exact beq_iff_eq.mp (List.all_eq_true.mp check e member)

theorem domain_endpoints (d : Graph) (domain : MIDomain d) :
    ∀ e ∈ d.EDGES, e.START ∈ d.NODES ∧ e.END ∈ d.NODES := by
  simp only [MIDomain, miDomainB, Bool.and_eq_true] at domain
  have check := domain.1.2
  intro e member
  have h : d.NODES.contains e.START = true ∧ d.NODES.contains e.END = true := by
    simpa only [Bool.and_eq_true] using List.all_eq_true.mp check e member
  exact ⟨List.contains_iff_mem.mp h.1, List.contains_iff_mem.mp h.2⟩

theorem predecessor_props (d : Graph) (domain : MIDomain d)
    {v u : Vertex} (member : u ∈ predsOf d v) :
    u ∈ d.NODES ∧ u.LEVEL = v.LEVEL+1 := by
  obtain ⟨e, incoming, start⟩ := predsOf_has_edge member
  have edgeMember := mem_incoming_mem_edges v d incoming
  have ending := mem_incoming_end_eq v d incoming
  constructor
  · rw [← start]
    exact (domain_endpoints d domain e edgeMember).1
  · rw [← start, ← ending]
    exact domain_levels d domain e edgeMember

theorem kernel_memo_eq_spec (k : CompleteProfileKernel.Kernel) (d : Graph)
    (domain : MIDomain d) : ∀ fuel,
    CompleteProfileKernel.memoAt k d fuel = CompleteProfileKernel.specificationTable k d fuel := by
  intro fuel
  induction fuel with
  | zero => rfl
  | succ fuel ih =>
    simp only [CompleteProfileKernel.memoAt, ih, CompleteProfileKernel.specificationTable]
    apply List.map_congr_left
    intro v listed
    apply congrArg (Prod.mk v)
    apply k.congr
    intro u member
    exact CompleteProfileKernel.lookup_spec k d fuel (predecessor_props d domain member).1

theorem kernel_succ_stable (k : CompleteProfileKernel.Kernel) (d : Graph)
    (domain : MIDomain d) : ∀ v ∈ d.NODES, ∀ fuel,
    countAbove d v < fuel →
    CompleteProfileKernel.evaluateAt k d (fuel+1) v = CompleteProfileKernel.evaluateAt k d fuel v := by
  have key : ∀ n, ∀ v ∈ d.NODES, countAbove d v = n → ∀ fuel,
      countAbove d v < fuel →
      CompleteProfileKernel.evaluateAt k d (fuel+1) v = CompleteProfileKernel.evaluateAt k d fuel v := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro v listed count fuel bound
      have fuelEq : fuel = (fuel-1)+1 := by omega
      rw [CompleteProfileKernel.evaluateAt]
      conv_rhs => rw [fuelEq, CompleteProfileKernel.evaluateAt]
      apply k.congr
      intro u member
      obtain ⟨ulisted, level⟩ := predecessor_props d domain member
      have smaller := countAbove_lt_of_level_succ d ulisted level
      have stable := ih (countAbove d u) (by omega) u ulisted rfl (fuel-1) (by omega)
      rw [fuelEq]
      exact stable
  intro v listed fuel bound
  exact key (countAbove d v) v listed rfl fuel bound

theorem kernel_standard_fixed (k : CompleteProfileKernel.Kernel) (d : Graph)
    (domain : MIDomain d) (v : Vertex) (listed : v ∈ d.NODES) :
    k.step d (CompleteProfileKernel.evaluateAt k d (stdFuel d)) v =
      CompleteProfileKernel.evaluateAt k d (stdFuel d) v :=
  kernel_succ_stable k d domain v listed (stdFuel d)
    (by have bound := countAbove_le d v; unfold stdFuel; omega)

theorem kernel_fixed_eq_evaluate (k : CompleteProfileKernel.Kernel) (d : Graph)
    (domain : MIDomain d) (stored : Vertex → List FlowPair)
    (fixed : ∀ v ∈ d.NODES, stored v = k.step d stored v) :
    ∀ fuel v, v ∈ d.NODES → countAbove d v < fuel →
      stored v = CompleteProfileKernel.evaluateAt k d fuel v := by
  intro fuel
  induction fuel with
  | zero => intro v listed bound; omega
  | succ fuel ih =>
    intro v listed bound
    rw [fixed v listed, CompleteProfileKernel.evaluateAt]
    apply k.congr
    intro u member
    obtain ⟨ulisted, level⟩ := predecessor_props d domain member
    have smaller := countAbove_lt_of_level_succ d ulisted level
    exact ih u ulisted (by omega)

#print axioms kernel_succ_stable
#print axioms kernel_fixed_eq_evaluate
end MultiIntro
