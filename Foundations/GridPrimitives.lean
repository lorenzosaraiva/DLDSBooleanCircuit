import Init
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import Mathlib.Data.Vector.Mem
import Mathlib.Data.List.Duplicate
import Mathlib.Data.Vector.Defs
import Mathlib.Data.Vector.Zip
import Mathlib.Data.Fin.Basic
import Foundations.Graph

open scoped Classical

namespace Semantic





instance : Inhabited Formula where
  default := #"_"

instance : ToString Formula where
  toString := Formula.repr




inductive ActivationBits
  | intro (bit : Bool)
  | elim (bit1 : Bool) (bit2 : Bool)
  | repetition (bit : Bool)
  deriving DecidableEq





inductive RuleData (n : Nat)
  | intro (encoder : List.Vector Bool n)
  | elim
  | repetition






structure Rule (n : ℕ) where
  ruleId     : Nat
  activation : ActivationBits
  type       : RuleData n
  combine    : List (List.Vector Bool n) → List.Vector Bool n




structure CircuitNode (n : ℕ) where
  rules    : List (Rule n)
  nodupIds : (rules.map (·.ruleId)).Nodup




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
