import Init
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip
import Mathlib.Data.Fin.Basic
import HorizontalCompressionEXEC

open scoped Classical

namespace Semantic

/-!
# Core circuit syntax and rule constructors.
-/

instance : Inhabited Formula where
  default := #"_"

instance : ToString Formula where
  toString := Formula.repr
/-- Activation bits for each inference rule type.
    - Intro rules require one input (the premise of the implication)
    - Elim rules require two inputs (the implication and its antecedent)
    - Repetition rules require one input (identity/structural rule) -/
inductive ActivationBits
  | intro (bit : Bool)
  | elim (bit1 : Bool) (bit2 : Bool)
  | repetition (bit : Bool)
  deriving DecidableEq

/-- Rule data: the type of inference rule and its parameters.
    - `intro encoder`: Implication introduction with discharge mask
    - `elim`: Implication elimination (modus ponens)
    - `repetition`: Structural rule (identity) -/
inductive RuleData (n : Nat)
  | intro (encoder : List.Vector Bool n)
  | elim
  | repetition

/-- A single inference rule consisting of:
    - `ruleId`: Unique identifier within the node
    - `activation`: Current activation state
    - `type`: Rule type and parameters
    - `combine`: Function computing output dependency vector from inputs -/
structure Rule (n : ℕ) where
  ruleId     : Nat
  activation : ActivationBits
  type       : RuleData n
  combine    : List (List.Vector Bool n) → List.Vector Bool n

/-- Circuit node: a collection of alternative inference rules at a formula position.
    The `nodupIds` invariant ensures rule IDs are unique within the node,
    which is essential for the XOR-based exactly-one-active check. -/
structure CircuitNode (n : ℕ) where
  rules    : List (Rule n)
  nodupIds : (rules.map (·.ruleId)).Nodup

/-- Constructor for implication introduction rule (⊃I).
    This realizes Definition 3's ⊃I dependency update: discharge the antecedent
    by masking the incoming bitstring, `b' = b ∧ ¬b_α`. -/
def mkIntroRule {n : ℕ} (rid : Nat) (encoder : List.Vector Bool n) (bit : Bool) : Rule n :=
{
  ruleId     := rid,
  activation := ActivationBits.intro bit,
  type       := RuleData.intro encoder,
  combine    := fun deps =>
    match deps with
    | [d] => d.zipWith (fun b e => b && !e) encoder
    | _   => List.Vector.replicate n false
}

/-- Constructor for implication elimination rule (⊃E / modus ponens).
    This realizes Definition 3's ⊃E merge of dependency bitstrings:
    `b = b₁ ∨ b₂`. -/
def mkElimRule {n : ℕ} (rid : Nat) (bit1 bit2 : Bool) : Rule n :=
{
  ruleId     := rid,
  activation := ActivationBits.elim bit1 bit2,
  type       := RuleData.elim,
  combine    := fun deps =>
    match deps with
    | [d1, d2] => d1.zipWith (· || ·) d2
    | _        => List.Vector.replicate n false
}

/-- Constructor for repetition rule (structural).
    The combine function passes the dependency vector unchanged. -/
def mkRepetitionRule {n : ℕ} (rid : Nat) (bit : Bool) : Rule n :=
{
  ruleId     := rid,
  activation := ActivationBits.repetition bit,
  type       := RuleData.repetition,
  combine    := fun deps =>
    match deps with
    | [d] => d
    | _   => List.Vector.replicate n false
}

/-- A rule is well-formed if its combine function matches its declared type.
    This ensures the rule constructors produce consistent rules. -/
def Rule.WellFormed {n : Nat} (r : Rule n) : Prop :=
  match r.type with
  | RuleData.intro encoder =>
      r.combine = fun deps =>
        match deps with
        | [d] => d.zipWith (fun b e => b && !e) encoder
        | _ => List.Vector.replicate n false
  | RuleData.elim =>
      r.combine = fun deps =>
        match deps with
        | [d1, d2] => d1.zipWith (· || ·) d2
        | _ => List.Vector.replicate n false
  | RuleData.repetition =>
      r.combine = fun deps =>
        match deps with
        | [d] => d
        | _ => List.Vector.replicate n false


end Semantic
