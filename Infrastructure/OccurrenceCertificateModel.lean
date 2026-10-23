import Infrastructure.LambdaFlowCache
import Infrastructure.RootDischarge

open Semantic FlowSpec CertificateModel SelfCertifying

namespace OccurrenceCertificate


abbrev OccurrenceCertificate := Certificate
abbrev ExpandedCertOccurrence := ExpandedCert
abbrev expandedCertOccurrenceB := expandedCertB

def reseedExpectedOccurrence (c : OccurrenceCertificate) (v : Vertex) : List FlowPair :=
  LambdaFlow.reseedWith c.graph LambdaFlow.allowAll v (outputsAt c)



def localExpectedOccurrence (c : OccurrenceCertificate) (v : Vertex) : List FlowPair :=
  match localShape c.graph v with
  | .source => sourceExpected c.graph v
  | .elim major minor =>
      (elimExpected c v major minor ++ mixedIntroExpected c v ++
        reseedExpectedOccurrence c v).eraseDups
  | .elimMulti pairs =>
      (elimExpectedMany c v pairs ++ mixedIntroExpected c v ++
        reseedExpectedOccurrence c v).eraseDups
  | .intro premise alpha _ =>
      (introExpected c v premise alpha ++ reseedExpectedOccurrence c v).eraseDups
  | .invalid => []

def witnessableExpectedOccurrence (c : OccurrenceCertificate) (v : Vertex) : List FlowPair :=
  WitnessableRoutes c.graph v (localExpectedOccurrence c v)

def productionAtOccurrenceB (c : OccurrenceCertificate) (v : Vertex) : Bool :=
  match localShape c.graph v with
  | .invalid => false
  | _ => setEqB (outputsAt c v) (witnessableExpectedOccurrence c v)

def witnessFaithfulAtOccurrenceB (c : OccurrenceCertificate) (witness : EdgeRouteWitness) : Bool :=
  let matching := (localExpectedOccurrence c witness.edge.START).filter fun r =>
    r.2 == witness.residual
  matching.isEmpty || matching.any fun r => depSetEq r.1 witness.dependency

def LocalProductionOccurrence (c : OccurrenceCertificate) : Prop :=
  ∀ v ∈ c.graph.NODES, productionAtOccurrenceB c v = true

def LocalWitnessFaithfulOccurrence (c : OccurrenceCertificate) : Prop :=
  ∀ witness ∈ c.edgeRoutes, witnessFaithfulAtOccurrenceB c witness = true

def LocalCarrierAccountedAtOccurrence (c : OccurrenceCertificate) (w u : Vertex) : Prop :=
  ∃ alpha, w.FORMULA = Formula.implication alpha u.FORMULA ∧
    ∀ tail ∈ continuableConsumedWitnessTails c u w,
      ∃ dependency, (dependency, tail) ∈ witnessableExpectedOccurrence c w



def LocalTailClosureOccurrence (c : OccurrenceCertificate) : Prop :=
  ∀ w ∈ c.graph.NODES, elimPairsAt c.graph w ≠ [] →
    (∀ u ∈ predsOf c.graph w,
      ElimPremiseAt c.graph w u ∨ LocalCarrierAccountedAtOccurrence c w u) ∧
    (∀ major minor, (major, minor) ∈ elimPairsAt c.graph w →
      certificatePairTailClosureAtB c w major minor = true)

def LocalChecksOccurrence (d : Graph) (c : OccurrenceCertificate) : Prop :=
  c.graph = d ∧
  LocalProductionOccurrence c ∧ LocalCoverage c ∧
  LocalTailClosureOccurrence c ∧ LocalRootDischarge c ∧
  structuralValid d ∧ d.EDGES.Nodup ∧ OneEdgePerColourPerNode d ∧
  LocalWitnessFaithfulOccurrence c ∧ LevelFormulaUnique d ∧ HypothesesHaveNoIncoming d

def UniversalLocalRootDischargeAtOccurrence (c : OccurrenceCertificate) (v : Vertex) : Prop :=
  (get_rule.outgoing v c.graph).isEmpty = true →
    (∀ r ∈ StoredRoutes c v, r.1 = [] ∧ r.2 = []) ∧
    (∀ r ∈ localExpectedOccurrence c v, r.2 = [])

def UniversalLocalRootDischargeOccurrence (c : OccurrenceCertificate) : Prop :=
  ∀ v ∈ c.graph.NODES, UniversalLocalRootDischargeAtOccurrence c v

def StrongFormationValidOccurrence (d : Graph) (c : OccurrenceCertificate) : Prop :=
  ExpandedCertOccurrence d c ∧ LocalChecksOccurrence d c ∧ UniversalLocalRootDischargeOccurrence c

def certificateCarrierAccountedAtOccurrenceB (c : OccurrenceCertificate) (w u : Vertex) : Bool :=
  match w.FORMULA with
  | .implication _ beta =>
      u.FORMULA == beta &&
        (continuableConsumedWitnessTails c u w).all fun tail =>
          (witnessableExpectedOccurrence c w).any fun result => result.2 == tail
  | _ => false

def tailClosureAtOccurrenceB (c : OccurrenceCertificate) (w : Vertex) : Bool :=
  match elimPairsAt c.graph w with
  | [] => true
  | pairs =>
      (predsOf c.graph w).all (fun u =>
        elimPremiseAtB c.graph w u || certificateCarrierAccountedAtOccurrenceB c w u) &&
      pairs.all fun (major, minor) => certificatePairTailClosureAtB c w major minor

def rootDischargeAtOccurrenceB (c : OccurrenceCertificate) (v : Vertex) : Bool :=
  if (get_rule.outgoing v c.graph).isEmpty then
    (c.terminalOutputs.filter fun output => output.node == v).any fun output =>
      output.dependency.isEmpty && output.residual.isEmpty &&
      productionAtOccurrenceB c v && ruleShapeOKB v c.graph &&
      decide (outDep v c.graph = [])
  else true

def localChecksOccurrenceB (d : Graph) (c : OccurrenceCertificate) : Bool :=
  decide (c.graph = d) &&
  d.NODES.all (productionAtOccurrenceB c) && d.NODES.all (coverageAtB c) &&
  d.NODES.all (tailClosureAtOccurrenceB c) && d.NODES.all (rootDischargeAtOccurrenceB c) &&
  structuralValidB d && nodupB d && oneEdgeB d &&
  c.edgeRoutes.all (witnessFaithfulAtOccurrenceB c) &&
  levelFormulaUniqueB d && hypNoIncomingB d

def universalLocalRootDischargeAtOccurrenceB (c : OccurrenceCertificate) (v : Vertex) : Bool :=
  if (get_rule.outgoing v c.graph).isEmpty then
    (StoredRoutes c v).all rootRouteEmptyB &&
      (localExpectedOccurrence c v).all fun route => route.2.isEmpty
  else true

def universalLocalRootDischargeOccurrenceB (c : OccurrenceCertificate) : Bool :=
  c.graph.NODES.all (universalLocalRootDischargeAtOccurrenceB c)

def strongFormationValidOccurrenceB (d : Graph) (c : OccurrenceCertificate) : Bool :=
  expandedCertOccurrenceB d c && localChecksOccurrenceB d c && universalLocalRootDischargeOccurrenceB c


def labelGroup2 (c : OccurrenceCertificate) (edge : Deduction) : List FlowPair :=
  (((StoredRoutes c edge.START).map fun route =>
    (normalizeDep route.1, route.2)).eraseDups).filter fun route =>
      headColour route.2 == edge.COLOUR




def localLabelAtOccurrenceB (c : OccurrenceCertificate) (label : Deduction → LambdaFlow.Label)
    (edge : Deduction) : Bool :=
  match labelGroup2 c edge, label edge with
  | [route], .Bits dependency => depSetEq route.1 dependency
  | [_], .Lambda => false
  | _, .Lambda => true
  | _, .Bits _ => false

def LocalLabelCheck (c : OccurrenceCertificate) (label : Deduction → LambdaFlow.Label) : Prop :=
  ∀ edge ∈ c.graph.EDGES, localLabelAtOccurrenceB c label edge = true

def localLabelCheckB (c : OccurrenceCertificate) (label : Deduction → LambdaFlow.Label) : Bool :=
  c.graph.EDGES.all (localLabelAtOccurrenceB c label)

def canonicalLocalLabel (c : OccurrenceCertificate) (edge : Deduction) : LambdaFlow.Label :=
  match labelGroup2 c edge with
  | [route] => .Bits route.1
  | _ => .Lambda



def LabelledStrongFormationValidOccurrence (d : Graph) (c : OccurrenceCertificate)
    (label : Deduction → LambdaFlow.Label) : Prop :=
  StrongFormationValidOccurrence d c ∧ LocalLabelCheck c label

theorem expandedCertOccurrence_iff (d : Graph) (c : OccurrenceCertificate) :
    ExpandedCertOccurrence d c ↔ ExpandedCert d c := Iff.rfl

theorem localLabelCheckB_iff (c : OccurrenceCertificate) (label : Deduction → LambdaFlow.Label) :
    localLabelCheckB c label = true ↔ LocalLabelCheck c label := by
  simp [localLabelCheckB, LocalLabelCheck, List.all_eq_true]

end OccurrenceCertificate
