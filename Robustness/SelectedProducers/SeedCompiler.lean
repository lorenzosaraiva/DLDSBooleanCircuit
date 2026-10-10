import Robustness.SelectedProducers.SeedSemantics

/-! Seed producers compiled to actual bounded programs on certificate bits.
The ancestor address is decoded from the header and may be out of range.
Static guards, canonical dependency equality, exact path equality and local
witnessability are all included. No replay or oracle-correctness premise occurs. -/
open DLDS Flow Robustness.BitLocality
namespace Robustness.SelectedProducers
set_option maxHeartbeats 2000000
set_option maxRecDepth 2000

def atomGuard {Coordinate : Type} {k : Nat} (atom : Atom Coordinate k) :
    Guard Coordinate k where
  clauses _ := [atom]

theorem atomGuard_check {Coordinate : Type} {k : Nat}
    (atom : Atom Coordinate k) (bits : Coordinate → Bool) :
    (atomGuard atom).check bits = atom.eval bits := by
  cases value : atom.eval bits <;> simp [Guard.check, atomGuard, value]

def constantAtom {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (truth : Bool) : Atom (SeedCoord shape size) k where
  address _ := .inl ⟨0, by omega⟩
  test _ := truth

def formulaAtom {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (slot : Fin routes)
    (code : Fin formulas) (formula : Formula) : Atom (SeedCoord shape size) k where
  address _ := .inr (.inl (slot, code))
  test values := values (0, ⟨0, positive⟩) == decide (shape.formulaAt code = formula)

def pathAtom {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (width : shape.colourWidth ≤ k) (slot : Fin routes)
    (position : Fin (shape.pathLength slot)) (colour : Nat) :
    Atom (SeedCoord shape size) k where
  address index :=
    if bound : index.2.val < shape.colourWidth then
      .inr (.inr ⟨slot, position, ⟨index.2.val, bound⟩⟩)
    else .inl ⟨0, by omega⟩
  test values := decide (decodeWord (fun bit : Fin shape.colourWidth =>
    values (0, ⟨bit.val, lt_of_lt_of_le bit.isLt width⟩)) = colour)

theorem constantAtom_eval {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (truth : Bool) (bits : SeedCoord shape size → Bool) :
    (constantAtom shape size k truth).eval bits = truth := rfl

theorem formulaAtom_eval {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (slot : Fin routes)
    (code : Fin formulas) (formula : Formula) (bits : SeedCoord shape size → Bool) :
    (formulaAtom shape size k positive slot code formula).eval bits =
      (bits (.inr (.inl (slot, code))) == decide (shape.formulaAt code = formula)) := rfl

theorem pathAtom_eval {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (width : shape.colourWidth ≤ k) (slot : Fin routes)
    (position : Fin (shape.pathLength slot)) (colour : Nat)
    (bits : SeedCoord shape size → Bool) :
    (pathAtom shape size k width slot position colour).eval bits =
      decide (decodeWord (fun bit => bits (.inr (.inr ⟨slot, position, bit⟩))) = colour) := by
  simp only [Atom.eval, pathAtom]
  congr 3
  funext bit
  simp [bit.isLt]

def formulaRepresentedB {routes formulas : Nat} (shape : RouteShape routes formulas)
    (formula : Formula) : Bool :=
  (List.finRange formulas).any fun code => decide (shape.formulaAt code = formula)

theorem formulaRepresentedB_iff {routes formulas : Nat} (shape : RouteShape routes formulas)
    (formula : Formula) :
    formulaRepresentedB shape formula = true ↔ ∃ code, shape.formulaAt code = formula := by
  simp [formulaRepresentedB, List.any_eq_true]

def targetGuards {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (width : shape.colourWidth ≤ k)
    (d : Graph) (v : Vertex) (slot : Fin routes) (formula : Formula)
    (path : ColourPath) : List (Guard (SeedCoord shape size) k) :=
  atomGuard (constantAtom shape size k
      (routeContinuableAtB d v path && formulaRepresentedB shape formula &&
        decide (path.length = shape.pathLength slot))) ::
    ((List.finRange formulas).map fun code =>
      atomGuard (formulaAtom shape size k positive slot code formula)) ++
    ((List.finRange (shape.pathLength slot)).map fun position =>
      atomGuard (pathAtom shape size k width slot position (path.getD position.val 0)))

theorem targetGuards_check {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (width : shape.colourWidth ≤ k)
    (d : Graph) (v : Vertex) (slot : Fin routes) (formula : Formula)
    (path : ColourPath) (bits : SeedCoord shape size → Bool) :
    (targetGuards shape size k positive width d v slot formula path).all
        (fun guard => guard.check bits) =
      (decide (decodeRoutes shape (fun coordinate => bits (.inr coordinate)) slot =
          ([formula], path)) && routeContinuableAtB d v path) := by
  apply Bool.eq_iff_iff.mpr
  simp only [targetGuards, List.all_cons, List.all_append, List.all_map, Function.comp_apply,
    atomGuard_check, constantAtom_eval, formulaAtom_eval, pathAtom_eval,
    Bool.and_eq_true, List.all_eq_true, List.mem_finRange, decide_eq_true_eq,
    beq_iff_eq]
  rw [show decodeRoutes shape (fun coordinate => bits (.inr coordinate)) slot =
      ([formula], path) ↔
      (decodeRoutes shape (fun coordinate => bits (.inr coordinate)) slot).1 = [formula] ∧
      (decodeRoutes shape (fun coordinate => bits (.inr coordinate)) slot).2 = path from Prod.ext_iff]
  rw [show (decodeRoutes shape (fun coordinate => bits (.inr coordinate)) slot).1 =
      decodeDependency shape (fun code => bits (.inr (.inl (slot, code)))) from rfl]
  rw [dependency_singleton_iff, decoded_path_eq_iff, formulaRepresentedB_iff]
  simp only [true_implies]
  tauto

def seedProgram {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (width : shape.colourWidth ≤ k)
    (d : Graph) (v : Vertex) (slot : Fin routes) (mode : SeedMode) :
    Program (SeedCoord shape size) (3 + 2 * pointerWidth size) k where
  headerAddress bit := .inl bit
  guards header :=
    match seedTarget? d v mode size header with
    | none => [atomGuard (constantAtom shape size k false)]
    | some target => targetGuards shape size k positive width d v slot v.FORMULA target.2

theorem seedTarget?_dependency (d : Graph) (v : Vertex) (mode : SeedMode) (size : Nat)
    (header : Fin (3 + 2 * pointerWidth size) → Bool) (target : FlowPair)
    (selected : seedTarget? d v mode size header = some target) :
    target.1 = [v.FORMULA] := by
  unfold seedTarget? at selected
  split at selected
  · split at selected
    · cases mode <;> simp_all
      exact congrArg Prod.fst selected.symm
    · cases mode
      · simp at selected
      · cases lookup : (ancestorsInto d v)[ancestorIndex size header]? with
        | none => simp [lookup] at selected
        | some ancestor =>
          simp only [lookup, Option.map_some, Option.some.injEq] at selected
          exact congrArg Prod.fst selected.symm
  · simp at selected

theorem seed_compiler_check_eq {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (width : shape.colourWidth ≤ k)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (bits : SeedCoord shape size → Bool) :
    (seedProgram shape size k positive width d (owner slot) slot mode).check bits =
      seedAcceptance shape d owner listed slot mode size bits := by
  rw [seed_acceptance_eq]
  unfold Program.check seedProgram
  dsimp only
  cases targetEq : seedTarget? d (owner slot) mode size (fun bit => bits (.inl bit)) with
  | none =>
    change ((atomGuard (constantAtom shape size k false)).check bits && true) = false
    rw [atomGuard_check, constantAtom_eval]
    rfl
  | some target =>
    have dep := seedTarget?_dependency d (owner slot) mode size _ target targetEq
    rcases target with ⟨dependency, path⟩
    dsimp only at dep
    subst dependency
    exact targetGuards_check shape size k positive width
      d (owner slot) slot (owner slot).FORMULA path bits

theorem seed_compiler_failure_eq {routes formulas : Nat} (shape : RouteShape routes formulas)
    (size k : Nat) (positive : 0 < k) (width : shape.colourWidth ≤ k)
    (d : Graph) (owner : Fin routes → Vertex)
    (listed : ∀ slot, owner slot ∈ d.NODES) (slot : Fin routes)
    (mode : SeedMode) (bits : SeedCoord shape size → Bool) :
    (seedProgram shape size k positive width d (owner slot) slot mode).failure bits =
      seedFailure shape d owner listed slot mode size bits :=
  congrArg Bool.not (seed_compiler_check_eq shape size k positive width d owner listed slot mode bits)

end Robustness.SelectedProducers
