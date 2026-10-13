import Evaluator.Definitions
import Evaluator.Boolean
import Evaluator.VectorLemmas
import Evaluator.CellCorrectness
import Evaluator.TokenRouting
import Evaluator.GridEvaluation
import Evaluator.Grid
import Evaluator.TreeCorrespondence
import Flow.Definitions
import Flow.Validity
import Flow.TreeSoundness
import Flow.Compressed.Elimination
import Flow.DependencyLabels
import Flow.Compressed.DependencyBridge
import Evaluator.Compressed.Routing
import Evaluator.MultiToken.Definitions
import Evaluator.MultiToken.Admissibility
import Evaluator.MultiToken.CanonicalReduction
import Evaluator.MultiToken.Discharge
import Flow.UniversalDischarge

/-!
# DLDS-to-Boolean-Circuit Formalization

This library formalizes the Boolean circuit evaluator for DLDS path assignments,
DLDS-side structural predicates, the simple-tree bridge, and the FLOW proof
modules, and compressed-bridge lookup infrastructure.
-/
