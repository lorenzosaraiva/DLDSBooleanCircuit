import Semantic.Core
import Semantic.Boolean
import Semantic.VectorLemmas
import Semantic.NodeCorrectness
import Semantic.Routing
import Semantic.Evaluator
import Semantic.DLDS
import Semantic.TreeBridge
import Semantic.FlowModel
import Semantic.FlowValidity
import Semantic.FlowTreeProof
import Semantic.FlowCollapsedProof
import Semantic.FlowEdgeDepChar
import Semantic.CompressedBridge
import Semantic.CompressedRouting
import Semantic.MultiTokenModel
import Semantic.MultiTokenAdmissible
import Semantic.MultiTokenReduction
import Semantic.MultiTokenBridge
import Semantic.FlowClosure

/-!
# DLDS-to-Boolean-Circuit Formalization

This library formalizes the Boolean circuit evaluator for DLDS path assignments,
DLDS-side structural predicates, the simple-tree bridge, and the FLOW proof
modules, and compressed-bridge lookup infrastructure.
-/
