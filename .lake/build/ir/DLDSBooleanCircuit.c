// Lean compiler output
// Module: DLDSBooleanCircuit
// Imports: public import Init public import Init public import Mathlib.Data.List.Basic public import Mathlib.Tactic public import Mathlib.Data.Vector.Mem public import Mathlib.Data.List.Duplicate public import Mathlib.Data.Vector.Defs public import Mathlib.Data.Vector.Zip public import Mathlib.Data.Fin.Basic
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_Vector_zipWith___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_mkIntroRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_mkIntroRule___closed__0_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_mkElimRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_mkElimRule___closed__0_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule(lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_is__rule__active(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_is__rule__active___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_multiple__xor(lean_object*);
uint8_t l_List_or(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_multiple__xor___boxed(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_extract__activations(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_extract__activations___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_and__bool__list_spec__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_and__bool__list_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_and__bool__list(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_and__bool__list___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_list__or(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations___lam__0(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0;
lean_object* l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_CircuitNode_run(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_multiple__xor_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_multiple__xor_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___lam__0___boxed(lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___closed__0_value;
lean_object* lp_mathlib_List_Vector_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_zipIdxTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_initialize__tokens(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_initialize__tokens___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_get___redArg(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_propagate__tokens___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_propagate__tokens___closed__0;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_propagate__tokens(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_propagate__tokens___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_natToBits_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_natToBits_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_natToBits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_natToBits___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_selector_spec__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_selector_spec__0___boxed(lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_selector___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_selector___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* l_List_ofFn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_selector(lean_object*);
uint8_t l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__1(lean_object*, lean_object*);
uint8_t l_List_all___redArg(lean_object*, lean_object*);
lean_object* l_List_get_x21Internal___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_List_find_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing___lam__0(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_zipWith3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___closed__0_value;
uint8_t l_List_any___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__node(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__node___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__List_zipWith3_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__List_zipWith3_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__3_value;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4_value;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__8_value;
lean_object* l_Lean_Name_mkStr1(lean_object*);
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__10 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__10_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11_value;
lean_object* l_Lean_mkAtom(lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__12;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__13;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__14 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__14_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__18;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__19;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__20;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__21;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__22 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__22_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__23;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__24;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__25 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__25_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__27;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "List.length_zipIdx"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__28 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__28_value;
lean_object* lean_string_utf8_byte_size(lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__29;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__30;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__31 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__31_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "length_zipIdx"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__32 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__32_value;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__33_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(37, 62, 254, 99, 111, 166, 106, 120)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__33 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__33_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__34;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__35;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__36;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__37;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__38;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__39;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__40 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__40_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__41;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__42;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__43;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__44;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__45 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__45_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__45_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__47 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__47_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__48;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__49;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "locationHyp"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__50 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__50_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(229, 146, 67, 234, 45, 36, 143, 176)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "hi"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__52 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__52_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__53;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__54;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__52_value),LEAN_SCALAR_PTR_LITERAL(13, 105, 183, 27, 151, 203, 27, 169)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__55 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__55_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__57;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__58;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__59;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__60;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__61;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__62;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__63;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__64;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__65;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__66;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__67;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__68 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__68_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__69;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__70;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__71 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__71_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value_aux_0),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value_aux_1),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value_aux_2),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__71_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__73;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__74_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__74;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__75;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__76;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__77;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__78;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__79;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__80_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__80;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__81_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__81;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__82_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__82;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluate__layer___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__layer___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_evaluate__layer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_evaluate__layer___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__layer___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_evaluate__layer___closed__0_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__layer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_eval__from__level_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_eval__from__level(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_eval__from__level___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_get__eval__result(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_get__eval__result___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___lam__0(uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateCircuit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_atom_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_atom_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_impl_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_impl_elim(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula___boxed(lean_object*, lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Semantic.Formula.atom"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__2_value;
lean_object* lean_nat_to_int(lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Semantic.Formula.impl"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__6_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__7_value;
lean_object* l_String_quote(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instReprFormula___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprFormula___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instInhabitedFormula_default___closed__1_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⊃ "};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__1_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__2_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString___boxed(lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instToStringFormula___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_Formula_toString___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instToStringFormula___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instToStringFormula___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instToStringFormula = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instToStringFormula___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1_value;
static const lean_string_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4_value;
lean_object* lean_string_length(lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__5;
static lean_once_cell_t lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6;
static const lean_ctor_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__22_value)}};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__40_value)}};
static const lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8_value;
lean_object* l_Std_Format_joinSep___at___00Array_repr___at___00Lean_Elab_Structural_instReprRecArgInfo_repr_spec__1_spec__3(lean_object*, lean_object*);
lean_object* l_Std_Format_fill(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "node"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__3_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__3_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__6_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "LEVEL"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__8_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__8_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__9 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__9_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "FORMULA"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__11 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__11_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__11_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__12 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__12_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "HYPOTHESIS"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__14 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__14_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__14_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__15 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__15_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "COLLAPSED"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__17 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__17_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__17_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__18 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__18_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "PAST"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__20 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__20_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__20_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__21 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__21_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__22 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__22_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__23;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__22_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26_value;
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Bool_repr___redArg(uint8_t);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instReprVertex___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex___closed__0_value;
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
uint8_t l_instDecidableEqList___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "START"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__3_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "END"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__5_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__6;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "COLOUR"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__7_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__8_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "DEPENDENCY"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__10 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__10_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__10_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__11 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__11_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instReprDeduction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDeduction___closed__0_value;
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__0;
static lean_once_cell_t lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1;
static const lean_ctor_object lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3_value;
lean_object* lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4_spec__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1___redArg(lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "V"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__3_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__4;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "E"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__6_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "A"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__7_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instReprDLDS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS___closed__0_value;
uint8_t l_instDecidableEqProd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___closed__0;
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1___closed__0_value;
lean_object* l_List_eraseDupsBy___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildFormulas_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildFormulas(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_encoderForIntro_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_encoderForIntro_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_encoderForIntro(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_encoderForIntro___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_buildIncomingMapForFormula_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_buildIncomingMapForFormula_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___closed__0;
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildIncomingMap_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMap(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_nodeForFormula_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_nodeForFormula_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_nodeForFormula___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_nodeForFormula___closed__0;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_nodeForFormula(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_nodeForFormula___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildLayers_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildLayers_spec__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildLayers(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildGridFromDLDS(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_initialVectorsFromDLDS(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_encoderForIntro_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_encoderForIntro_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateDLDS(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateDLDS___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_readingToPath_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingToPath___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingToPath(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingToPath___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_numHyps_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_numHyps(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_or(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_HypDepVec_clearBit_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_HypDepVec_clearBit_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_hypIndex_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_hypIndex_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_hypIndex(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "source"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__3_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "readingVar"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__5_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "targets"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instReprBranching___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranching___closed__0_value;
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___closed__0_value;
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0___redArg(lean_object*);
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "base"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__3_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "branchings"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__5_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "numReading"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__7_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "evalOrder"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__8_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__8_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__9 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS___closed__0_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_incomingSources_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_incomingSources_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_incomingSources(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_envLookup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__1(lean_object*, lean_object*);
lean_object* l_List_findSome_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_findBranchTarget(lean_object*, lean_object*);
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_readingColour___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_readingColour___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_readingColour___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_readingColour___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_readingColour___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_readingColour___closed__1_value;
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingColour(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingColour___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_stepVertex_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_stepVertex_spec__1(lean_object*, lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_stepVertex(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_stepVertex___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_dldsSemantics_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_dldsSemantics_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemantics(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemantics___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemanticsAt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemanticsAt___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__7_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "B"};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value;
static const lean_string_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "C"};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eYB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_eXB___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__5_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__2_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_branchYtoB___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vB___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vY___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_vX___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_baseDLDS___closed__5_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__4_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Layer2Test_testBranchingDLDS___closed__4_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Evaluation result: "};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__0_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 55, .m_data = "✗ Rejected: Invalid routing or undischarged assumptions"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__1_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 71, .m_data = "✓ Accepted: Valid proof with discharged assumptions OR structural error"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__2_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__3_value;
static const lean_string_object lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__4_value;
lean_object* lp_mathlib_IO_println___at___00__private_Mathlib_Tactic_Linter_TextBased_0__Mathlib_Linter_TextBased_formatErrors_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_A = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_B = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity___closed__0;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB___closed__0_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion___closed__0;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__2_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB___closed__2_value;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion___closed__0;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__0;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__1;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__2;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__3;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__4;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__5;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__6;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__7;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__8;
static lean_once_cell_t lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__9;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__5_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__7_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__8_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__8_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__9 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__9_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__9_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__4_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__7_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__8_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_invalidPath___closed__8_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_C = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_conclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_conclusion___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_conclusion___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_conclusion = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_conclusion___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(7) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_conclusion___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_B__imp__C___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fC___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_A__imp__C___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_inner___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__conclusion___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__inner___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpC___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__C___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__B___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__A___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__BimpC___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_v__AimpB___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e6___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__8_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e5___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__8_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__9 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__9_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e4___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__9_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__10 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__10_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e3___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__10_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__11 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__11_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e2___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__11_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__12 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__12_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e1___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__12_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__13 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__13_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_e0___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__13_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__14 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__14_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__7_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__15 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__15_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_dlds___closed__15_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(8) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(7) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__4_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(5) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__6 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__6_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(4) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__7 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__7_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__6_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__8 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__8_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__5_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__9 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__9_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__9_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__10 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__10_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__10_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__11 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__11_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__11_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__12 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__12_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__12_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__14 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__14_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__14_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__15 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__15_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__15_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__16 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__16_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__13_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__16_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__17 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__17_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__7_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__17_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__18 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__18_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__8_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__18_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__19 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__19_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__7_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__19_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__20 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__20_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Syllogism_validPath___closed__20_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_B = (const lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_Layer2Test_fA___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_A__imp__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__1_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__B___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__AimpB___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_v__A___closed__0_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__1_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__AimpB___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_e__A___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__4_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__2_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__5 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__5_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_dlds___closed__5_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__0 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__0_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__0_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__1 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__1_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__2 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__2_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Identity_validPath___closed__4_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__2_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__3 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__3_value;
static const lean_ctor_object lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__1_value),((lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__3_value)}};
static const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__4 = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__4_value;
LEAN_EXPORT const lean_object* lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path = (const lean_object*)&lp_DLDSBooleanCircuit_Semantic_Test_Incomplete_path___closed__4_value;
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorIdx(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lean_unsigned_to_nat(2u);
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorIdx(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
uint8_t x_3; uint8_t x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get_uint8(x_1, 0);
x_4 = lean_ctor_get_uint8(x_1, 1);
x_5 = lean_box(x_3);
x_6 = lean_box(x_4);
x_7 = lean_apply_2(x_2, x_5, x_6);
return x_7;
}
else
{
uint8_t x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get_uint8(x_1, 0);
x_9 = lean_box(x_8);
x_10 = lean_apply_1(x_2, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_intro_elim(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_elim_elim(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___redArg(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_ActivationBits_repetition_elim(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = lean_ctor_get_uint8(x_1, 0);
if (x_3 == 0)
{
uint8_t x_4; 
x_4 = lean_ctor_get_uint8(x_2, 0);
if (x_4 == 0)
{
uint8_t x_5; 
x_5 = 1;
return x_5;
}
else
{
return x_3;
}
}
else
{
uint8_t x_6; 
x_6 = lean_ctor_get_uint8(x_2, 0);
return x_6;
}
}
else
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
}
case 1:
{
if (lean_obj_tag(x_2) == 1)
{
uint8_t x_8; uint8_t x_9; uint8_t x_10; uint8_t x_11; 
x_8 = lean_ctor_get_uint8(x_1, 0);
x_9 = lean_ctor_get_uint8(x_1, 1);
x_10 = lean_ctor_get_uint8(x_2, 0);
x_11 = lean_ctor_get_uint8(x_2, 1);
if (x_8 == 0)
{
if (x_10 == 0)
{
goto block_13;
}
else
{
return x_8;
}
}
else
{
if (x_10 == 0)
{
return x_10;
}
else
{
goto block_13;
}
}
block_13:
{
if (x_9 == 0)
{
if (x_11 == 0)
{
uint8_t x_12; 
x_12 = 1;
return x_12;
}
else
{
return x_9;
}
}
else
{
return x_11;
}
}
}
else
{
uint8_t x_14; 
x_14 = 0;
return x_14;
}
}
default: 
{
if (lean_obj_tag(x_2) == 2)
{
uint8_t x_15; 
x_15 = lean_ctor_get_uint8(x_1, 0);
if (x_15 == 0)
{
uint8_t x_16; 
x_16 = lean_ctor_get_uint8(x_2, 0);
if (x_16 == 0)
{
uint8_t x_17; 
x_17 = 1;
return x_17;
}
else
{
return x_15;
}
}
else
{
uint8_t x_18; 
x_18 = lean_ctor_get_uint8(x_2, 0);
return x_18;
}
}
else
{
uint8_t x_19; 
x_19 = 0;
return x_19;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits_decEq(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits_decEq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqActivationBits(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___redArg(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
case 1:
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
default: 
{
lean_object* x_4; 
x_4 = lean_unsigned_to_nat(2u);
return x_4;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorIdx(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_2, x_3);
return x_4;
}
else
{
lean_dec(x_1);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_RuleData_intro_elim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_RuleData_elim_elim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_RuleData_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_RuleData_repetition_elim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__0(uint8_t x_1, uint8_t x_2) {
_start:
{
if (x_1 == 0)
{
return x_1;
}
else
{
if (x_2 == 0)
{
return x_1;
}
else
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__0(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 1)
{
lean_object* x_9; 
x_9 = lean_ctor_get(x_4, 1);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; lean_object* x_11; 
lean_dec(x_1);
x_10 = lean_ctor_get(x_4, 0);
lean_inc(x_10);
lean_dec_ref(x_4);
x_11 = lp_mathlib_List_Vector_zipWith___redArg(x_2, x_10, x_3);
return x_11;
}
else
{
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
goto block_8;
}
}
else
{
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
goto block_8;
}
block_8:
{
uint8_t x_5; lean_object* x_6; lean_object* x_7; 
x_5 = 0;
x_6 = lean_box(x_5);
x_7 = l_List_replicateTR___redArg(x_1, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_mkIntroRule___closed__0));
lean_inc(x_3);
x_6 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_mkIntroRule___lam__1), 4, 3);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_3);
x_7 = lean_alloc_ctor(0, 0, 1);
lean_ctor_set_uint8(x_7, 0, x_4);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_3);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_8);
lean_ctor_set(x_9, 3, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkIntroRule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_4);
x_6 = lp_DLDSBooleanCircuit_Semantic_mkIntroRule(x_1, x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__0(uint8_t x_1, uint8_t x_2) {
_start:
{
if (x_1 == 0)
{
return x_2;
}
else
{
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_3 = lean_unbox(x_1);
x_4 = lean_unbox(x_2);
x_5 = lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__0(x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 1)
{
lean_object* x_8; 
x_8 = lean_ctor_get(x_3, 1);
lean_inc(x_8);
if (lean_obj_tag(x_8) == 1)
{
lean_object* x_9; 
x_9 = lean_ctor_get(x_8, 1);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_1);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec_ref(x_3);
x_11 = lean_ctor_get(x_8, 0);
lean_inc(x_11);
lean_dec_ref(x_8);
x_12 = lp_mathlib_List_Vector_zipWith___redArg(x_2, x_10, x_11);
return x_12;
}
else
{
lean_dec_ref(x_8);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
goto block_7;
}
}
else
{
lean_dec_ref(x_3);
lean_dec(x_8);
lean_dec_ref(x_2);
goto block_7;
}
}
else
{
lean_dec(x_3);
lean_dec_ref(x_2);
goto block_7;
}
block_7:
{
uint8_t x_4; lean_object* x_5; lean_object* x_6; 
x_4 = 0;
x_5 = lean_box(x_4);
x_6 = l_List_replicateTR___redArg(x_1, x_5);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule(lean_object* x_1, lean_object* x_2, uint8_t x_3, uint8_t x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_mkElimRule___closed__0));
x_6 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_mkElimRule___lam__1), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lean_alloc_ctor(1, 0, 2);
lean_ctor_set_uint8(x_7, 0, x_3);
lean_ctor_set_uint8(x_7, 1, x_4);
x_8 = lean_box(1);
x_9 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_8);
lean_ctor_set(x_9, 3, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkElimRule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; uint8_t x_6; lean_object* x_7; 
x_5 = lean_unbox(x_3);
x_6 = lean_unbox(x_4);
x_7 = lp_DLDSBooleanCircuit_Semantic_mkElimRule(x_1, x_2, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 1)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_2, 1);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; 
lean_dec(x_1);
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_8);
return x_8;
}
else
{
goto block_6;
}
}
else
{
goto block_6;
}
block_6:
{
uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_3 = 0;
x_4 = lean_box(x_3);
x_5 = l_List_replicateTR___redArg(x_1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule(lean_object* x_1, lean_object* x_2, uint8_t x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_ctor(2, 0, 1);
lean_ctor_set_uint8(x_5, 0, x_3);
x_6 = lean_box(2);
x_7 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_7, 0, x_2);
lean_ctor_set(x_7, 1, x_5);
lean_ctor_set(x_7, 2, x_6);
lean_ctor_set(x_7, 3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lean_unbox(x_3);
x_5 = lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule(x_1, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_2) == 1)
{
uint8_t x_3; 
x_3 = lean_ctor_get_uint8(x_2, 0);
if (x_3 == 0)
{
return x_3;
}
else
{
uint8_t x_4; 
x_4 = lean_ctor_get_uint8(x_2, 1);
return x_4;
}
}
else
{
uint8_t x_5; 
x_5 = lean_ctor_get_uint8(x_2, 0);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg(x_1);
lean_dec_ref(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_is__rule__active(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_is__rule__active___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_is__rule__active(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_multiple__xor(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
if (lean_obj_tag(x_4) == 0)
{
uint8_t x_9; 
x_9 = lean_unbox(x_3);
lean_dec(x_3);
return x_9;
}
else
{
uint8_t x_10; 
x_10 = lean_unbox(x_3);
if (x_10 == 0)
{
goto block_8;
}
else
{
uint8_t x_11; 
lean_inc(x_4);
x_11 = l_List_or(x_4);
if (x_11 == 0)
{
uint8_t x_12; 
lean_dec(x_4);
x_12 = lean_unbox(x_3);
lean_dec(x_3);
return x_12;
}
else
{
goto block_8;
}
}
}
block_8:
{
uint8_t x_5; 
x_5 = lean_unbox(x_3);
lean_dec(x_3);
if (x_5 == 0)
{
x_1 = x_4;
goto _start;
}
else
{
uint8_t x_7; 
lean_dec(x_4);
x_7 = 0;
return x_7;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_multiple__xor___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_DLDSBooleanCircuit_Semantic_multiple__xor(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg(x_5);
lean_dec(x_5);
x_8 = lean_box(x_7);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_8);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; uint8_t x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_1, 0);
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_1);
x_12 = lp_DLDSBooleanCircuit_Semantic_is__rule__active___redArg(x_10);
lean_dec(x_10);
x_13 = lean_box(x_12);
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_2);
x_1 = x_11;
x_2 = x_14;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_extract__activations(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_box(0);
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_extract__activations___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_extract__activations(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_extract__activations_spec__0(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_and__bool__list_spec__0(uint8_t x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_7 = x_2;
} else {
 lean_dec_ref(x_2);
 x_7 = lean_box(0);
}
if (x_1 == 0)
{
lean_dec(x_5);
x_8 = x_1;
goto block_12;
}
else
{
uint8_t x_13; 
x_13 = lean_unbox(x_5);
lean_dec(x_5);
x_8 = x_13;
goto block_12;
}
block_12:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_box(x_8);
if (lean_is_scalar(x_7)) {
 x_10 = lean_alloc_ctor(1, 2, 0);
} else {
 x_10 = x_7;
}
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_3);
x_2 = x_6;
x_3 = x_10;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_and__bool__list_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lean_unbox(x_1);
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_and__bool__list_spec__0(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_and__bool__list(uint8_t x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_box(0);
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_and__bool__list_spec__0(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_and__bool__list___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lean_unbox(x_1);
x_4 = lp_DLDSBooleanCircuit_Semantic_and__bool__list(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_mkElimRule___closed__0));
x_6 = lp_mathlib_List_Vector_zipWith___redArg(x_5, x_1, x_3);
x_1 = x_6;
x_2 = x_4;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_list__or(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = 0;
x_4 = lean_box(x_3);
x_5 = l_List_replicateTR___redArg(x_1, x_4);
x_6 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0___redArg(x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_list__or_spec__0(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4) {
_start:
{
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec_ref(x_3);
lean_dec(x_2);
x_5 = lean_box(x_4);
x_6 = l_List_replicateTR___redArg(x_1, x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; 
lean_dec(x_1);
x_7 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_7);
lean_dec_ref(x_3);
x_8 = lean_apply_1(x_7, x_2);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_4);
x_6 = lp_DLDSBooleanCircuit_Semantic_apply__activations___lam__0(x_1, x_2, x_3, x_5);
return x_6;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_apply__activations___lam__0___boxed), 4, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0, &lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0);
x_7 = l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(x_5, x_2, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_2);
x_4 = lp_DLDSBooleanCircuit_Semantic_extract__activations(x_1, x_2);
lean_inc(x_4);
x_5 = lp_DLDSBooleanCircuit_Semantic_multiple__xor(x_4);
x_6 = lp_DLDSBooleanCircuit_Semantic_and__bool__list(x_5, x_4);
lean_inc(x_1);
x_7 = lp_DLDSBooleanCircuit_Semantic_apply__activations(x_1, x_2, x_6, x_3);
x_8 = lp_DLDSBooleanCircuit_Semantic_list__or(x_1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_CircuitNode_run(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_node__logic(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_multiple__xor_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_4);
lean_dec(x_3);
x_5 = lean_box(0);
x_6 = lean_apply_1(x_2, x_5);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_2);
x_7 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; 
lean_dec(x_4);
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_apply_1(x_3, x_8);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; 
lean_inc(x_7);
lean_dec(x_3);
x_10 = lean_ctor_get(x_1, 0);
lean_inc(x_10);
lean_dec_ref(x_1);
x_11 = lean_apply_3(x_4, x_10, x_7, lean_box(0));
return x_11;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_multiple__xor_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_5);
lean_dec(x_4);
x_6 = lean_box(0);
x_7 = lean_apply_1(x_3, x_6);
return x_7;
}
else
{
lean_object* x_8; 
lean_dec(x_3);
x_8 = lean_ctor_get(x_2, 1);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; lean_object* x_10; 
lean_dec(x_5);
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_apply_1(x_4, x_9);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; 
lean_inc(x_8);
lean_dec(x_4);
x_11 = lean_ctor_get(x_2, 0);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = lean_apply_3(x_5, x_11, x_8, lean_box(0));
return x_12;
}
}
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___lam__0(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___lam__0(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___closed__0));
x_3 = lean_unsigned_to_nat(0u);
x_4 = lp_mathlib_List_Vector_ofFn___redArg(x_1, x_2);
x_5 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_3);
lean_ctor_set(x_5, 3, x_3);
lean_ctor_set(x_5, 4, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_instInhabitedToken_default(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instInhabitedToken___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_instInhabitedToken(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec(x_6);
lean_inc(x_1);
lean_inc_n(x_9, 2);
x_10 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_9);
lean_ctor_set(x_10, 2, x_1);
lean_ctor_set(x_10, 3, x_9);
lean_ctor_set(x_10, 4, x_8);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_10);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_14 = lean_ctor_get(x_12, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_12, 1);
lean_inc(x_15);
lean_dec(x_12);
lean_inc(x_1);
lean_inc_n(x_15, 2);
x_16 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_15);
lean_ctor_set(x_16, 2, x_1);
lean_ctor_set(x_16, 3, x_15);
lean_ctor_set(x_16, 4, x_14);
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_3);
x_2 = x_13;
x_3 = x_17;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_initialize__tokens(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = l_List_zipIdxTR___redArg(x_2, x_4);
x_6 = lean_box(0);
x_7 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0___redArg(x_3, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_initialize__tokens___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_initialize__tokens(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialize__tokens_spec__0(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_7; 
x_7 = lean_array_to_list(x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_8 = lean_ctor_get(x_5, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_5, 1);
lean_inc(x_9);
lean_dec_ref(x_5);
x_10 = !lean_is_exclusive(x_8);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_11 = lean_ctor_get(x_8, 0);
x_12 = lean_ctor_get(x_8, 3);
x_13 = lean_ctor_get(x_8, 4);
lean_dec(x_13);
x_14 = lean_ctor_get(x_8, 2);
lean_dec(x_14);
x_15 = lean_ctor_get(x_8, 1);
lean_dec(x_15);
x_16 = l_List_lengthTR___redArg(x_1);
x_17 = lean_nat_dec_lt(x_11, x_16);
lean_dec(x_16);
if (x_17 == 0)
{
lean_free_object(x_8);
lean_dec(x_12);
lean_dec(x_11);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_19; uint8_t x_20; 
x_19 = lean_unsigned_to_nat(0u);
x_20 = lean_nat_dec_lt(x_19, x_2);
if (x_20 == 0)
{
lean_free_object(x_8);
lean_dec(x_12);
lean_dec(x_11);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; uint8_t x_27; 
lean_inc(x_11);
x_22 = l_List_get___redArg(x_1, x_11);
x_23 = lean_nat_sub(x_3, x_2);
x_24 = lean_unsigned_to_nat(1u);
x_25 = lean_nat_sub(x_23, x_24);
lean_dec(x_23);
x_26 = l_List_lengthTR___redArg(x_22);
x_27 = lean_nat_dec_lt(x_25, x_26);
lean_dec(x_26);
if (x_27 == 0)
{
lean_dec(x_25);
lean_dec(x_22);
lean_free_object(x_8);
lean_dec(x_12);
lean_dec(x_11);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_29; uint8_t x_30; 
x_29 = l_List_get___redArg(x_22, x_25);
lean_dec(x_22);
x_30 = lean_nat_dec_eq(x_29, x_19);
if (x_30 == 0)
{
lean_object* x_31; uint8_t x_32; 
x_31 = l_List_lengthTR___redArg(x_4);
x_32 = lean_nat_dec_lt(x_12, x_31);
lean_dec(x_31);
if (x_32 == 0)
{
lean_dec(x_29);
lean_free_object(x_8);
lean_dec(x_12);
lean_dec(x_11);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_34 = lean_nat_sub(x_29, x_24);
lean_dec(x_29);
x_35 = lean_nat_sub(x_2, x_24);
lean_inc(x_12);
x_36 = l_List_get___redArg(x_4, x_12);
lean_ctor_set(x_8, 4, x_36);
lean_ctor_set(x_8, 3, x_34);
lean_ctor_set(x_8, 2, x_35);
lean_ctor_set(x_8, 1, x_12);
x_37 = lean_array_push(x_6, x_8);
x_5 = x_9;
x_6 = x_37;
goto _start;
}
}
else
{
lean_dec(x_29);
lean_free_object(x_8);
lean_dec(x_12);
lean_dec(x_11);
x_5 = x_9;
goto _start;
}
}
}
}
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; uint8_t x_43; 
x_40 = lean_ctor_get(x_8, 0);
x_41 = lean_ctor_get(x_8, 3);
lean_inc(x_41);
lean_inc(x_40);
lean_dec(x_8);
x_42 = l_List_lengthTR___redArg(x_1);
x_43 = lean_nat_dec_lt(x_40, x_42);
lean_dec(x_42);
if (x_43 == 0)
{
lean_dec(x_41);
lean_dec(x_40);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_45; uint8_t x_46; 
x_45 = lean_unsigned_to_nat(0u);
x_46 = lean_nat_dec_lt(x_45, x_2);
if (x_46 == 0)
{
lean_dec(x_41);
lean_dec(x_40);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; uint8_t x_53; 
lean_inc(x_40);
x_48 = l_List_get___redArg(x_1, x_40);
x_49 = lean_nat_sub(x_3, x_2);
x_50 = lean_unsigned_to_nat(1u);
x_51 = lean_nat_sub(x_49, x_50);
lean_dec(x_49);
x_52 = l_List_lengthTR___redArg(x_48);
x_53 = lean_nat_dec_lt(x_51, x_52);
lean_dec(x_52);
if (x_53 == 0)
{
lean_dec(x_51);
lean_dec(x_48);
lean_dec(x_41);
lean_dec(x_40);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_55; uint8_t x_56; 
x_55 = l_List_get___redArg(x_48, x_51);
lean_dec(x_48);
x_56 = lean_nat_dec_eq(x_55, x_45);
if (x_56 == 0)
{
lean_object* x_57; uint8_t x_58; 
x_57 = l_List_lengthTR___redArg(x_4);
x_58 = lean_nat_dec_lt(x_41, x_57);
lean_dec(x_57);
if (x_58 == 0)
{
lean_dec(x_55);
lean_dec(x_41);
lean_dec(x_40);
x_5 = x_9;
goto _start;
}
else
{
lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_60 = lean_nat_sub(x_55, x_50);
lean_dec(x_55);
x_61 = lean_nat_sub(x_2, x_50);
lean_inc(x_41);
x_62 = l_List_get___redArg(x_4, x_41);
x_63 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_63, 0, x_40);
lean_ctor_set(x_63, 1, x_41);
lean_ctor_set(x_63, 2, x_61);
lean_ctor_set(x_63, 3, x_60);
lean_ctor_set(x_63, 4, x_62);
x_64 = lean_array_push(x_6, x_63);
x_5 = x_9;
x_6 = x_64;
goto _start;
}
}
else
{
lean_dec(x_55);
lean_dec(x_41);
lean_dec(x_40);
x_5 = x_9;
goto _start;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_7;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_propagate__tokens___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_propagate__tokens(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_propagate__tokens___closed__0, &lp_DLDSBooleanCircuit_Semantic_propagate__tokens___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_propagate__tokens___closed__0);
x_8 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg(x_3, x_4, x_5, x_6, x_2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_propagate__tokens___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit_Semantic_propagate__tokens(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___redArg(x_1, x_2, x_3, x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_propagate__tokens_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_natToBits_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
x_5 = l_List_reverse___redArg(x_4);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_sub(x_1, x_9);
x_11 = lean_nat_sub(x_10, x_7);
lean_dec(x_7);
lean_dec(x_10);
x_12 = lean_nat_shiftr(x_2, x_11);
lean_dec(x_11);
x_13 = lean_unsigned_to_nat(2u);
x_14 = lean_nat_mod(x_12, x_13);
lean_dec(x_12);
x_15 = lean_nat_dec_eq(x_14, x_9);
lean_dec(x_14);
x_16 = lean_box(x_15);
lean_ctor_set(x_3, 1, x_4);
lean_ctor_set(x_3, 0, x_16);
{
lean_object* _tmp_2 = x_8;
lean_object* _tmp_3 = x_3;
x_3 = _tmp_2;
x_4 = _tmp_3;
}
goto _start;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; lean_object* x_27; lean_object* x_28; 
x_18 = lean_ctor_get(x_3, 0);
x_19 = lean_ctor_get(x_3, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_3);
x_20 = lean_unsigned_to_nat(1u);
x_21 = lean_nat_sub(x_1, x_20);
x_22 = lean_nat_sub(x_21, x_18);
lean_dec(x_18);
lean_dec(x_21);
x_23 = lean_nat_shiftr(x_2, x_22);
lean_dec(x_22);
x_24 = lean_unsigned_to_nat(2u);
x_25 = lean_nat_mod(x_23, x_24);
lean_dec(x_23);
x_26 = lean_nat_dec_eq(x_25, x_20);
lean_dec(x_25);
x_27 = lean_box(x_26);
x_28 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_4);
x_3 = x_19;
x_4 = x_28;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_natToBits_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_natToBits_spec__0(x_1, x_2, x_3, x_4);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_natToBits(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc(x_2);
x_3 = l_List_range(x_2);
x_4 = lean_box(0);
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_natToBits_spec__0(x_2, x_1, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_natToBits___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_natToBits(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_selector_spec__0(uint8_t x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
if (x_1 == 0)
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 1);
x_2 = x_3;
goto _start;
}
else
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_5, 1);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
lean_object* x_8; uint8_t x_9; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
lean_object* x_10; 
x_10 = lean_ctor_get(x_2, 1);
x_2 = x_10;
goto _start;
}
else
{
lean_object* x_12; uint8_t x_13; 
x_12 = lean_ctor_get(x_2, 1);
x_13 = lean_unbox(x_6);
x_1 = x_13;
x_2 = x_12;
goto _start;
}
}
else
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_2, 1);
x_16 = lean_ctor_get(x_5, 0);
x_17 = lean_unbox(x_16);
x_1 = x_17;
x_2 = x_15;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_selector_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; uint8_t x_4; lean_object* x_5; 
x_3 = lean_unbox(x_1);
x_4 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_selector_spec__0(x_3, x_2);
lean_dec(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_selector___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lp_DLDSBooleanCircuit_Semantic_natToBits(x_3, x_1);
x_5 = 1;
x_6 = l_List_zipWith___at___00List_zip_spec__0___redArg(x_2, x_4);
x_7 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_selector_spec__0(x_5, x_6);
lean_dec(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_selector___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_DLDSBooleanCircuit_Semantic_selector___lam__0(x_1, x_2, x_3);
lean_dec(x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_selector(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_List_lengthTR___redArg(x_1);
lean_inc(x_2);
x_3 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_selector___lam__0___boxed), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_unsigned_to_nat(2u);
x_5 = lean_nat_pow(x_4, x_2);
lean_dec(x_2);
x_6 = l_List_ofFn___redArg(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_box(0);
x_7 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__0(x_2, x_6);
x_8 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__1(x_3, x_6);
lean_inc(x_8);
x_9 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_9, 0, x_8);
lean_inc(x_7);
x_10 = l_List_all___redArg(x_7, x_9);
switch (lean_obj_tag(x_5)) {
case 0:
{
uint8_t x_11; 
lean_dec(x_8);
lean_dec(x_7);
x_11 = !lean_is_exclusive(x_5);
if (x_11 == 0)
{
lean_ctor_set_uint8(x_5, 0, x_10);
return x_1;
}
else
{
lean_object* x_12; 
lean_dec(x_5);
x_12 = lean_alloc_ctor(0, 0, 1);
lean_ctor_set_uint8(x_12, 0, x_10);
lean_ctor_set(x_1, 1, x_12);
return x_1;
}
}
case 1:
{
uint8_t x_13; 
x_13 = !lean_is_exclusive(x_5);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = l_List_lengthTR___redArg(x_7);
x_15 = lean_unsigned_to_nat(2u);
x_16 = lean_nat_dec_eq(x_14, x_15);
lean_dec(x_14);
if (x_16 == 0)
{
lean_dec(x_8);
lean_dec(x_7);
lean_ctor_set_uint8(x_5, 0, x_16);
lean_ctor_set_uint8(x_5, 1, x_16);
return x_1;
}
else
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; 
x_17 = lean_unsigned_to_nat(0u);
x_18 = l_List_get_x21Internal___redArg(x_17, x_7, x_17);
x_19 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_18, x_8);
lean_dec(x_18);
x_20 = lean_unsigned_to_nat(1u);
x_21 = l_List_get_x21Internal___redArg(x_17, x_7, x_20);
lean_dec(x_7);
x_22 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_21, x_8);
lean_dec(x_8);
lean_dec(x_21);
lean_ctor_set_uint8(x_5, 0, x_19);
lean_ctor_set_uint8(x_5, 1, x_22);
return x_1;
}
}
else
{
lean_object* x_23; lean_object* x_24; uint8_t x_25; 
lean_dec(x_5);
x_23 = l_List_lengthTR___redArg(x_7);
x_24 = lean_unsigned_to_nat(2u);
x_25 = lean_nat_dec_eq(x_23, x_24);
lean_dec(x_23);
if (x_25 == 0)
{
lean_object* x_26; 
lean_dec(x_8);
lean_dec(x_7);
x_26 = lean_alloc_ctor(1, 0, 2);
lean_ctor_set_uint8(x_26, 0, x_25);
lean_ctor_set_uint8(x_26, 1, x_25);
lean_ctor_set(x_1, 1, x_26);
return x_1;
}
else
{
lean_object* x_27; lean_object* x_28; uint8_t x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; lean_object* x_33; 
x_27 = lean_unsigned_to_nat(0u);
x_28 = l_List_get_x21Internal___redArg(x_27, x_7, x_27);
x_29 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_28, x_8);
lean_dec(x_28);
x_30 = lean_unsigned_to_nat(1u);
x_31 = l_List_get_x21Internal___redArg(x_27, x_7, x_30);
lean_dec(x_7);
x_32 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_31, x_8);
lean_dec(x_8);
lean_dec(x_31);
x_33 = lean_alloc_ctor(1, 0, 2);
lean_ctor_set_uint8(x_33, 0, x_29);
lean_ctor_set_uint8(x_33, 1, x_32);
lean_ctor_set(x_1, 1, x_33);
return x_1;
}
}
}
default: 
{
uint8_t x_34; 
lean_dec(x_8);
lean_dec(x_7);
x_34 = !lean_is_exclusive(x_5);
if (x_34 == 0)
{
lean_ctor_set_uint8(x_5, 0, x_10);
return x_1;
}
else
{
lean_object* x_35; 
lean_dec(x_5);
x_35 = lean_alloc_ctor(2, 0, 1);
lean_ctor_set_uint8(x_35, 0, x_10);
lean_ctor_set(x_1, 1, x_35);
return x_1;
}
}
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; uint8_t x_44; 
x_36 = lean_ctor_get(x_1, 0);
x_37 = lean_ctor_get(x_1, 1);
x_38 = lean_ctor_get(x_1, 2);
x_39 = lean_ctor_get(x_1, 3);
lean_inc(x_39);
lean_inc(x_38);
lean_inc(x_37);
lean_inc(x_36);
lean_dec(x_1);
x_40 = lean_box(0);
x_41 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__0(x_2, x_40);
x_42 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__1(x_3, x_40);
lean_inc(x_42);
x_43 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_43, 0, x_42);
lean_inc(x_41);
x_44 = l_List_all___redArg(x_41, x_43);
switch (lean_obj_tag(x_37)) {
case 0:
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; 
lean_dec(x_42);
lean_dec(x_41);
if (lean_is_exclusive(x_37)) {
 x_45 = x_37;
} else {
 lean_dec_ref(x_37);
 x_45 = lean_box(0);
}
if (lean_is_scalar(x_45)) {
 x_46 = lean_alloc_ctor(0, 0, 1);
} else {
 x_46 = x_45;
}
lean_ctor_set_uint8(x_46, 0, x_44);
x_47 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_47, 0, x_36);
lean_ctor_set(x_47, 1, x_46);
lean_ctor_set(x_47, 2, x_38);
lean_ctor_set(x_47, 3, x_39);
return x_47;
}
case 1:
{
lean_object* x_48; lean_object* x_49; lean_object* x_50; uint8_t x_51; 
if (lean_is_exclusive(x_37)) {
 x_48 = x_37;
} else {
 lean_dec_ref(x_37);
 x_48 = lean_box(0);
}
x_49 = l_List_lengthTR___redArg(x_41);
x_50 = lean_unsigned_to_nat(2u);
x_51 = lean_nat_dec_eq(x_49, x_50);
lean_dec(x_49);
if (x_51 == 0)
{
lean_object* x_52; lean_object* x_53; 
lean_dec(x_42);
lean_dec(x_41);
if (lean_is_scalar(x_48)) {
 x_52 = lean_alloc_ctor(1, 0, 2);
} else {
 x_52 = x_48;
}
lean_ctor_set_uint8(x_52, 0, x_51);
lean_ctor_set_uint8(x_52, 1, x_51);
x_53 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_53, 0, x_36);
lean_ctor_set(x_53, 1, x_52);
lean_ctor_set(x_53, 2, x_38);
lean_ctor_set(x_53, 3, x_39);
return x_53;
}
else
{
lean_object* x_54; lean_object* x_55; uint8_t x_56; lean_object* x_57; lean_object* x_58; uint8_t x_59; lean_object* x_60; lean_object* x_61; 
x_54 = lean_unsigned_to_nat(0u);
x_55 = l_List_get_x21Internal___redArg(x_54, x_41, x_54);
x_56 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_55, x_42);
lean_dec(x_55);
x_57 = lean_unsigned_to_nat(1u);
x_58 = l_List_get_x21Internal___redArg(x_54, x_41, x_57);
lean_dec(x_41);
x_59 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_58, x_42);
lean_dec(x_42);
lean_dec(x_58);
if (lean_is_scalar(x_48)) {
 x_60 = lean_alloc_ctor(1, 0, 2);
} else {
 x_60 = x_48;
}
lean_ctor_set_uint8(x_60, 0, x_56);
lean_ctor_set_uint8(x_60, 1, x_59);
x_61 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_61, 0, x_36);
lean_ctor_set(x_61, 1, x_60);
lean_ctor_set(x_61, 2, x_38);
lean_ctor_set(x_61, 3, x_39);
return x_61;
}
}
default: 
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; 
lean_dec(x_42);
lean_dec(x_41);
if (lean_is_exclusive(x_37)) {
 x_62 = x_37;
} else {
 lean_dec_ref(x_37);
 x_62 = lean_box(0);
}
if (lean_is_scalar(x_62)) {
 x_63 = lean_alloc_ctor(2, 0, 1);
} else {
 x_63 = x_62;
}
lean_ctor_set_uint8(x_63, 0, x_44);
x_64 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_64, 0, x_36);
lean_ctor_set(x_64, 1, x_63);
lean_ctor_set(x_64, 2, x_38);
lean_ctor_set(x_64, 3, x_39);
return x_64;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_set__rule__activation___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_set__rule__activation(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec(x_3);
lean_dec(x_2);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
x_8 = lean_box(0);
lean_inc(x_3);
x_9 = l_List_get_x21Internal___redArg(x_8, x_1, x_3);
lean_inc(x_2);
x_10 = lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg(x_6, x_9, x_2);
x_11 = lean_unsigned_to_nat(1u);
x_12 = lean_nat_add(x_3, x_11);
lean_dec(x_3);
x_13 = lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(x_1, x_2, x_12, x_7);
lean_ctor_set(x_4, 1, x_13);
lean_ctor_set(x_4, 0, x_10);
return x_4;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_14 = lean_ctor_get(x_4, 0);
x_15 = lean_ctor_get(x_4, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_4);
x_16 = lean_box(0);
lean_inc(x_3);
x_17 = l_List_get_x21Internal___redArg(x_16, x_1, x_3);
lean_inc(x_2);
x_18 = lp_DLDSBooleanCircuit_Semantic_set__rule__activation___redArg(x_14, x_17, x_2);
x_19 = lean_unsigned_to_nat(1u);
x_20 = lean_nat_add(x_3, x_19);
lean_dec(x_3);
x_21 = lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(x_1, x_2, x_20, x_15);
x_22 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_22, 0, x_18);
lean_ctor_set(x_22, 1, x_21);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activateRulesAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_activateRulesAux(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
x_5 = lean_apply_1(x_3, x_1);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_3(x_4, x_1, x_6, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_7; 
lean_dec(x_6);
x_7 = lean_apply_1(x_5, x_3);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_5);
x_8 = lean_ctor_get(x_4, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_9);
lean_dec_ref(x_4);
x_10 = lean_apply_3(x_6, x_3, x_8, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_activateRulesAux_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
uint8_t x_5; lean_object* x_6; lean_object* x_7; 
lean_dec(x_4);
lean_dec(x_3);
x_5 = lean_ctor_get_uint8(x_1, 0);
x_6 = lean_box(x_5);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
case 1:
{
uint8_t x_8; uint8_t x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_4);
lean_dec(x_2);
x_8 = lean_ctor_get_uint8(x_1, 0);
x_9 = lean_ctor_get_uint8(x_1, 1);
x_10 = lean_box(x_8);
x_11 = lean_box(x_9);
x_12 = lean_apply_2(x_3, x_10, x_11);
return x_12;
}
default: 
{
uint8_t x_13; lean_object* x_14; lean_object* x_15; 
lean_dec(x_3);
lean_dec(x_2);
x_13 = lean_ctor_get_uint8(x_1, 0);
x_14 = lean_box(x_13);
x_15 = lean_apply_1(x_4, x_14);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___redArg(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
switch (lean_obj_tag(x_2)) {
case 0:
{
uint8_t x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_5);
lean_dec(x_4);
x_6 = lean_ctor_get_uint8(x_2, 0);
x_7 = lean_box(x_6);
x_8 = lean_apply_1(x_3, x_7);
return x_8;
}
case 1:
{
uint8_t x_9; uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
lean_dec(x_5);
lean_dec(x_3);
x_9 = lean_ctor_get_uint8(x_2, 0);
x_10 = lean_ctor_get_uint8(x_2, 1);
x_11 = lean_box(x_9);
x_12 = lean_box(x_10);
x_13 = lean_apply_2(x_4, x_11, x_12);
return x_13;
}
default: 
{
uint8_t x_14; lean_object* x_15; lean_object* x_16; 
lean_dec(x_4);
lean_dec(x_3);
x_14 = lean_ctor_get_uint8(x_2, 0);
x_15 = lean_box(x_14);
x_16 = lean_apply_1(x_5, x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_is__rule__active_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lp_DLDSBooleanCircuit_Semantic_activateRulesAux___redArg(x_2, x_3, x_4, x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___redArg(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_activate__node__from__tokens(x_1, x_2, x_3, x_4);
lean_dec(x_3);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_nat_dec_eq(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0___lam__0(x_1, x_2);
lean_dec_ref(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = lean_array_to_list(x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec(x_5);
x_8 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0___lam__0___boxed), 2, 1);
lean_closure_set(x_8, 0, x_7);
lean_inc(x_1);
x_9 = l_List_find_x3f___redArg(x_8, x_1);
if (lean_obj_tag(x_9) == 0)
{
x_2 = x_6;
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec(x_11);
x_13 = lean_array_push(x_3, x_12);
x_2 = x_6;
x_3 = x_13;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0, &lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_apply__activations___closed__0);
x_4 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_gather__rule__inputs_spec__0(x_2, x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing___lam__0(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4) {
_start:
{
if (x_3 == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_4);
lean_dec_ref(x_2);
x_5 = lean_box(x_3);
x_6 = l_List_replicateTR___redArg(x_1, x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; 
lean_dec(x_1);
x_7 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_1(x_7, x_4);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lean_unbox(x_3);
x_6 = lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing___lam__0(x_1, x_2, x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing___lam__0___boxed), 4, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_List_zipWith3___redArg(x_5, x_2, x_3, x_4);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___lam__0(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___lam__0(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
lean_dec(x_2);
x_5 = l_List_reverse___redArg(x_4);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec(x_7);
x_10 = lean_box(0);
x_11 = l_List_get_x21Internal___redArg(x_10, x_1, x_9);
lean_inc(x_2);
x_12 = lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___redArg(x_11, x_2);
lean_ctor_set(x_3, 1, x_4);
lean_ctor_set(x_3, 0, x_12);
{
lean_object* _tmp_2 = x_8;
lean_object* _tmp_3 = x_3;
x_3 = _tmp_2;
x_4 = _tmp_3;
}
goto _start;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_14 = lean_ctor_get(x_3, 0);
x_15 = lean_ctor_get(x_3, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_3);
x_16 = lean_ctor_get(x_14, 1);
lean_inc(x_16);
lean_dec(x_14);
x_17 = lean_box(0);
x_18 = l_List_get_x21Internal___redArg(x_17, x_1, x_16);
lean_inc(x_2);
x_19 = lp_DLDSBooleanCircuit_Semantic_gather__rule__inputs___redArg(x_18, x_2);
x_20 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_4);
x_3 = x_15;
x_4 = x_20;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; uint8_t x_6; lean_object* x_7; uint8_t x_8; 
lean_inc(x_2);
x_5 = lp_DLDSBooleanCircuit_Semantic_extract__activations(x_1, x_2);
lean_inc(x_5);
x_6 = lp_DLDSBooleanCircuit_Semantic_multiple__xor(x_5);
lean_inc(x_5);
x_7 = lp_DLDSBooleanCircuit_Semantic_and__bool__list(x_6, x_5);
if (x_6 == 0)
{
lean_object* x_18; uint8_t x_19; 
x_18 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___closed__0));
x_19 = l_List_any___redArg(x_5, x_18);
x_8 = x_19;
goto block_17;
}
else
{
uint8_t x_20; 
lean_dec(x_5);
x_20 = 0;
x_8 = x_20;
goto block_17;
}
block_17:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_unsigned_to_nat(0u);
lean_inc(x_2);
x_10 = l_List_zipIdxTR___redArg(x_2, x_9);
x_11 = lean_box(0);
x_12 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg(x_3, x_4, x_10, x_11);
lean_inc(x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_apply__activations__with__routing(x_1, x_2, x_7, x_12);
x_14 = lp_DLDSBooleanCircuit_Semantic_list__or(x_1, x_13);
x_15 = lean_box(x_8);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___redArg(x_1, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_node__logic__with__routing_spec__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 4);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_9);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_1, 0);
x_12 = lean_ctor_get(x_1, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_1);
x_13 = lean_ctor_get(x_11, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_11, 4);
lean_inc(x_14);
lean_dec(x_11);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_2);
x_1 = x_12;
x_2 = x_16;
goto _start;
}
}
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg(lean_object* x_1, lean_object* x_2, uint8_t x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_6; 
lean_dec(x_2);
x_6 = l_List_reverse___redArg(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; 
x_7 = lean_ctor_get(x_4, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_4, 1);
lean_inc(x_9);
if (lean_is_exclusive(x_4)) {
 lean_ctor_release(x_4, 0);
 lean_ctor_release(x_4, 1);
 x_10 = x_4;
} else {
 lean_dec_ref(x_4);
 x_10 = lean_box(0);
}
x_11 = lean_ctor_get(x_7, 1);
lean_inc(x_11);
lean_dec(x_7);
x_12 = lean_ctor_get(x_8, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_13);
x_14 = lean_ctor_get(x_8, 2);
lean_inc(x_14);
x_15 = lean_ctor_get(x_8, 3);
lean_inc_ref(x_15);
if (lean_is_exclusive(x_8)) {
 lean_ctor_release(x_8, 0);
 lean_ctor_release(x_8, 1);
 lean_ctor_release(x_8, 2);
 lean_ctor_release(x_8, 3);
 x_16 = x_8;
} else {
 lean_dec_ref(x_8);
 x_16 = lean_box(0);
}
lean_inc(x_2);
x_22 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_22, 0, x_2);
x_23 = lean_box(0);
x_24 = l_List_get_x21Internal___redArg(x_23, x_1, x_11);
x_25 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__0(x_24, x_23);
lean_inc(x_25);
x_26 = l_List_all___redArg(x_25, x_22);
switch (lean_obj_tag(x_13)) {
case 0:
{
uint8_t x_27; 
lean_dec(x_25);
x_27 = !lean_is_exclusive(x_13);
if (x_27 == 0)
{
lean_ctor_set_uint8(x_13, 0, x_26);
x_17 = x_13;
goto block_21;
}
else
{
lean_object* x_28; 
lean_dec(x_13);
x_28 = lean_alloc_ctor(0, 0, 1);
lean_ctor_set_uint8(x_28, 0, x_26);
x_17 = x_28;
goto block_21;
}
}
case 1:
{
uint8_t x_29; 
x_29 = !lean_is_exclusive(x_13);
if (x_29 == 0)
{
lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_30 = l_List_lengthTR___redArg(x_25);
x_31 = lean_unsigned_to_nat(2u);
x_32 = lean_nat_dec_eq(x_30, x_31);
lean_dec(x_30);
if (x_32 == 0)
{
lean_dec(x_25);
lean_ctor_set_uint8(x_13, 0, x_3);
lean_ctor_set_uint8(x_13, 1, x_3);
x_17 = x_13;
goto block_21;
}
else
{
lean_object* x_33; lean_object* x_34; uint8_t x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; 
x_33 = lean_unsigned_to_nat(0u);
x_34 = l_List_get_x21Internal___redArg(x_33, x_25, x_33);
x_35 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_34, x_2);
lean_dec(x_34);
x_36 = lean_unsigned_to_nat(1u);
x_37 = l_List_get_x21Internal___redArg(x_33, x_25, x_36);
lean_dec(x_25);
x_38 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_37, x_2);
lean_dec(x_37);
lean_ctor_set_uint8(x_13, 0, x_35);
lean_ctor_set_uint8(x_13, 1, x_38);
x_17 = x_13;
goto block_21;
}
}
else
{
lean_object* x_39; lean_object* x_40; uint8_t x_41; 
lean_dec(x_13);
x_39 = l_List_lengthTR___redArg(x_25);
x_40 = lean_unsigned_to_nat(2u);
x_41 = lean_nat_dec_eq(x_39, x_40);
lean_dec(x_39);
if (x_41 == 0)
{
lean_object* x_42; 
lean_dec(x_25);
x_42 = lean_alloc_ctor(1, 0, 2);
lean_ctor_set_uint8(x_42, 0, x_3);
lean_ctor_set_uint8(x_42, 1, x_3);
x_17 = x_42;
goto block_21;
}
else
{
lean_object* x_43; lean_object* x_44; uint8_t x_45; lean_object* x_46; lean_object* x_47; uint8_t x_48; lean_object* x_49; 
x_43 = lean_unsigned_to_nat(0u);
x_44 = l_List_get_x21Internal___redArg(x_43, x_25, x_43);
x_45 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_44, x_2);
lean_dec(x_44);
x_46 = lean_unsigned_to_nat(1u);
x_47 = l_List_get_x21Internal___redArg(x_43, x_25, x_46);
lean_dec(x_25);
x_48 = l_List_elem___at___00Lean_Meta_Grind_Arith_Cutsat_checkElimEqs_spec__0(x_47, x_2);
lean_dec(x_47);
x_49 = lean_alloc_ctor(1, 0, 2);
lean_ctor_set_uint8(x_49, 0, x_45);
lean_ctor_set_uint8(x_49, 1, x_48);
x_17 = x_49;
goto block_21;
}
}
}
default: 
{
uint8_t x_50; 
lean_dec(x_25);
x_50 = !lean_is_exclusive(x_13);
if (x_50 == 0)
{
lean_ctor_set_uint8(x_13, 0, x_26);
x_17 = x_13;
goto block_21;
}
else
{
lean_object* x_51; 
lean_dec(x_13);
x_51 = lean_alloc_ctor(2, 0, 1);
lean_ctor_set_uint8(x_51, 0, x_26);
x_17 = x_51;
goto block_21;
}
}
}
block_21:
{
lean_object* x_18; lean_object* x_19; 
if (lean_is_scalar(x_16)) {
 x_18 = lean_alloc_ctor(0, 4, 0);
} else {
 x_18 = x_16;
}
lean_ctor_set(x_18, 0, x_12);
lean_ctor_set(x_18, 1, x_17);
lean_ctor_set(x_18, 2, x_14);
lean_ctor_set(x_18, 3, x_15);
if (lean_is_scalar(x_10)) {
 x_19 = lean_alloc_ctor(1, 2, 0);
} else {
 x_19 = x_10;
}
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_5);
x_4 = x_9;
x_5 = x_19;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lean_unbox(x_3);
x_7 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg(x_1, x_2, x_6, x_4, x_5);
lean_dec(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__node(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = l_List_isEmpty___redArg(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_6 = lean_box(0);
x_7 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__0(x_4, x_6);
lean_inc(x_7);
x_8 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_set__rule__activation_spec__1(x_7, x_6);
x_9 = lean_unsigned_to_nat(0u);
x_10 = l_List_zipIdxTR___redArg(x_2, x_9);
x_11 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg(x_3, x_8, x_5, x_10, x_6);
x_12 = lp_DLDSBooleanCircuit_Semantic_node__logic__with__routing(x_1, x_11, x_3, x_7);
return x_12;
}
else
{
uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_dec(x_4);
lean_dec(x_2);
x_13 = 0;
x_14 = lean_box(x_13);
x_15 = l_List_replicateTR___redArg(x_1, x_14);
x_16 = lean_box(x_13);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__node___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_evaluate__node(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, uint8_t x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lean_unbox(x_4);
x_8 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__node_spec__1(x_1, x_2, x_3, x_7, x_5, x_6);
lean_dec(x_2);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__List_zipWith3_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
if (lean_obj_tag(x_2) == 1)
{
if (lean_obj_tag(x_3) == 1)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_5);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_dec_ref(x_3);
x_12 = lean_apply_6(x_4, x_6, x_7, x_8, x_9, x_10, x_11);
return x_12;
}
else
{
lean_object* x_13; 
lean_dec(x_4);
x_13 = lean_apply_4(x_5, x_1, x_2, x_3, lean_box(0));
return x_13;
}
}
else
{
lean_object* x_14; 
lean_dec(x_4);
x_14 = lean_apply_4(x_5, x_1, x_2, x_3, lean_box(0));
return x_14;
}
}
else
{
lean_object* x_15; 
lean_dec(x_4);
x_15 = lean_apply_4(x_5, x_1, x_2, x_3, lean_box(0));
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__List_zipWith3_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
if (lean_obj_tag(x_5) == 1)
{
if (lean_obj_tag(x_6) == 1)
{
if (lean_obj_tag(x_7) == 1)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_dec(x_9);
x_10 = lean_ctor_get(x_5, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_5, 1);
lean_inc(x_11);
lean_dec_ref(x_5);
x_12 = lean_ctor_get(x_6, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_6, 1);
lean_inc(x_13);
lean_dec_ref(x_6);
x_14 = lean_ctor_get(x_7, 0);
lean_inc(x_14);
x_15 = lean_ctor_get(x_7, 1);
lean_inc(x_15);
lean_dec_ref(x_7);
x_16 = lean_apply_6(x_8, x_10, x_11, x_12, x_13, x_14, x_15);
return x_16;
}
else
{
lean_object* x_17; 
lean_dec(x_8);
x_17 = lean_apply_4(x_9, x_5, x_6, x_7, lean_box(0));
return x_17;
}
}
else
{
lean_object* x_18; 
lean_dec(x_8);
x_18 = lean_apply_4(x_9, x_5, x_6, x_7, lean_box(0));
return x_18;
}
}
else
{
lean_object* x_19; 
lean_dec(x_8);
x_19 = lean_apply_4(x_9, x_5, x_6, x_7, lean_box(0));
return x_19;
}
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__12(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__10));
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__13(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__12, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__12_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__12);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__18(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__15));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__19(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__18, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__18_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__18);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__13, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__13_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__13);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__20(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__19, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__19_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__19);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__21(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__20, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__20_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__20);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__23(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__22));
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__24(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__23, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__23_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__23);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__27(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__16);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__17);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__29(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__28));
x_2 = lean_string_utf8_byte_size(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__30(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__29, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__29_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__29);
x_2 = lean_unsigned_to_nat(0u);
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__28));
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__34(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_box(0);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__33));
x_3 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__30, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__30_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__30);
x_4 = lean_box(2);
x_5 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__35(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__34, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__34_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__34);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__27, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__27_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__27);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__36(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__35, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__35_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__35);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__26));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__37(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__36, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__36_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__36);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__38(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__37, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__37_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__37);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__39(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__38, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__38_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__38);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__24, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__24_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__24);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__41(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__40));
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__42(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__41, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__41_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__41);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__39, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__39_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__39);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__43(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__42, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__42_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__42);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__44(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__43, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__43_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__43);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__21, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__21_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__21);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__48(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__47));
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__49(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__48, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__48_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__48);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__53(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__52));
x_2 = lean_string_utf8_byte_size(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__54(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__53, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__53_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__53);
x_2 = lean_unsigned_to_nat(0u);
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__52));
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_box(0);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__55));
x_3 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__54, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__54_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__54);
x_4 = lean_box(2);
x_5 = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__57(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__58(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__57, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__57_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__57);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__59(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__58, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__58_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__58);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__60(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__59, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__59_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__59);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__51));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__61(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__60, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__60_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__60);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__49, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__49_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__49);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__62(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__61, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__61_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__61);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__46));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__63(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__62, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__62_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__62);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__64(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__63, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__63_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__63);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__65(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__64, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__64_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__64);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__44, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__44_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__44);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__66(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__65, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__65_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__65);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__11));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__67(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__66, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__66_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__66);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__69(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__68));
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__70(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__69, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__69_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__69);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__67, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__67_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__67);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__73(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__71));
x_2 = l_Lean_mkAtom(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__74(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__73, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__73_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__73);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__75(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__56);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__74, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__74_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__74);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__76(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__75, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__75_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__75);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__72));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__77(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__76, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__76_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__76);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__70, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__70_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__70);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__78(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__77, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__77_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__77);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__9));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__79(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__78, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__78_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__78);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__80(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__79, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__79_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__79);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__7));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__81(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__80, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__80_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__80);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__5);
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__82(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__81, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__81_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__81);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__4));
x_3 = lean_box(2);
x_4 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1(void) {
_start:
{
lean_object* x_1; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__82, &lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__82_once, _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__82);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
uint8_t x_7; lean_object* x_8; lean_object* x_9; 
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
x_7 = lean_ctor_get_uint8(x_1, 0);
x_8 = lean_box(x_7);
x_9 = lean_apply_2(x_3, x_8, x_2);
return x_9;
}
case 1:
{
uint8_t x_10; uint8_t x_11; lean_object* x_12; uint8_t x_13; 
lean_dec(x_6);
lean_dec(x_3);
x_10 = lean_ctor_get_uint8(x_1, 0);
x_11 = lean_ctor_get_uint8(x_1, 1);
x_12 = lean_unsigned_to_nat(2u);
x_13 = lean_nat_dec_eq(x_2, x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_dec(x_4);
x_14 = lean_box(x_10);
x_15 = lean_box(x_11);
x_16 = lean_apply_4(x_5, x_14, x_15, x_2, lean_box(0));
return x_16;
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_dec(x_5);
lean_dec(x_2);
x_17 = lean_box(x_10);
x_18 = lean_box(x_11);
x_19 = lean_apply_2(x_4, x_17, x_18);
return x_19;
}
}
default: 
{
uint8_t x_20; lean_object* x_21; lean_object* x_22; 
lean_dec(x_5);
lean_dec(x_4);
lean_dec(x_3);
x_20 = lean_ctor_get_uint8(x_1, 0);
x_21 = lean_box(x_20);
x_22 = lean_apply_2(x_6, x_21, x_2);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___redArg(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
switch (lean_obj_tag(x_2)) {
case 0:
{
uint8_t x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_7);
lean_dec(x_6);
lean_dec(x_5);
x_8 = lean_ctor_get_uint8(x_2, 0);
x_9 = lean_box(x_8);
x_10 = lean_apply_2(x_4, x_9, x_3);
return x_10;
}
case 1:
{
uint8_t x_11; uint8_t x_12; lean_object* x_13; uint8_t x_14; 
lean_dec(x_7);
lean_dec(x_4);
x_11 = lean_ctor_get_uint8(x_2, 0);
x_12 = lean_ctor_get_uint8(x_2, 1);
x_13 = lean_unsigned_to_nat(2u);
x_14 = lean_nat_dec_eq(x_3, x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_dec(x_5);
x_15 = lean_box(x_11);
x_16 = lean_box(x_12);
x_17 = lean_apply_4(x_6, x_15, x_16, x_3, lean_box(0));
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; 
lean_dec(x_6);
lean_dec(x_3);
x_18 = lean_box(x_11);
x_19 = lean_box(x_12);
x_20 = lean_apply_2(x_5, x_18, x_19);
return x_20;
}
}
default: 
{
uint8_t x_21; lean_object* x_22; lean_object* x_23; 
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_4);
x_21 = lean_ctor_get_uint8(x_2, 0);
x_22 = lean_box(x_21);
x_23 = lean_apply_2(x_7, x_22, x_3);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_evaluate__node_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluate__layer___lam__0(uint8_t x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__layer___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; uint8_t x_3; lean_object* x_4; 
x_2 = lean_unbox(x_1);
x_3 = lp_DLDSBooleanCircuit_Semantic_evaluate__layer___lam__0(x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__3(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 3);
x_9 = lean_nat_dec_eq(x_8, x_1);
if (x_9 == 0)
{
lean_free_object(x_2);
lean_dec(x_6);
x_2 = x_7;
goto _start;
}
else
{
lean_ctor_set(x_2, 1, x_3);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_14 = lean_ctor_get(x_12, 3);
x_15 = lean_nat_dec_eq(x_14, x_1);
if (x_15 == 0)
{
lean_dec(x_12);
x_2 = x_13;
goto _start;
}
else
{
lean_object* x_17; 
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_12);
lean_ctor_set(x_17, 1, x_3);
x_2 = x_13;
x_3 = x_17;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_6; 
lean_dec(x_3);
lean_dec(x_1);
x_6 = l_List_reverse___redArg(x_5);
return x_6;
}
else
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_8 = lean_ctor_get(x_4, 0);
x_9 = lean_ctor_get(x_4, 1);
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_8, 1);
lean_inc(x_11);
lean_dec(x_8);
x_12 = lean_ctor_get(x_2, 1);
x_13 = lean_box(0);
lean_inc(x_1);
x_14 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0(x_11, x_1, x_13);
x_15 = l_List_get_x21Internal___redArg(x_13, x_12, x_11);
lean_inc(x_3);
x_16 = lp_DLDSBooleanCircuit_Semantic_evaluate__node(x_3, x_10, x_15, x_14);
lean_dec(x_15);
lean_ctor_set(x_4, 1, x_5);
lean_ctor_set(x_4, 0, x_16);
{
lean_object* _tmp_3 = x_9;
lean_object* _tmp_4 = x_4;
x_4 = _tmp_3;
x_5 = _tmp_4;
}
goto _start;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_18 = lean_ctor_get(x_4, 0);
x_19 = lean_ctor_get(x_4, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_4);
x_20 = lean_ctor_get(x_18, 0);
lean_inc(x_20);
x_21 = lean_ctor_get(x_18, 1);
lean_inc(x_21);
lean_dec(x_18);
x_22 = lean_ctor_get(x_2, 1);
x_23 = lean_box(0);
lean_inc(x_1);
x_24 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_evaluate__layer_spec__0(x_21, x_1, x_23);
x_25 = l_List_get_x21Internal___redArg(x_23, x_22, x_21);
lean_inc(x_3);
x_26 = lp_DLDSBooleanCircuit_Semantic_evaluate__node(x_3, x_20, x_25, x_24);
lean_dec(x_25);
x_27 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_5);
x_4 = x_19;
x_5 = x_27;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__1(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluate__layer(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_evaluate__layer___closed__0));
x_6 = lean_unsigned_to_nat(0u);
lean_inc(x_4);
x_7 = l_List_zipIdxTR___redArg(x_4, x_6);
x_8 = lean_box(0);
x_9 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__1(x_3, x_2, x_1, x_7, x_8);
x_10 = !lean_is_exclusive(x_2);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_2, 1);
lean_dec(x_11);
x_12 = lean_ctor_get(x_2, 0);
lean_dec(x_12);
lean_inc(x_9);
x_13 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__2(x_9, x_8);
x_14 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__3(x_9, x_8);
x_15 = l_List_any___redArg(x_14, x_5);
x_16 = lean_box(x_15);
lean_ctor_set(x_2, 1, x_16);
lean_ctor_set(x_2, 0, x_13);
return x_2;
}
else
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; 
lean_dec(x_2);
lean_inc(x_9);
x_17 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__2(x_9, x_8);
x_18 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_evaluate__layer_spec__3(x_9, x_8);
x_19 = l_List_any___redArg(x_18, x_5);
x_20 = lean_box(x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_17);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_eval__from__level_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_2, 1);
x_7 = lean_ctor_get(x_2, 0);
lean_dec(x_7);
x_8 = 0;
x_9 = lean_box(x_8);
lean_inc(x_1);
x_10 = l_List_replicateTR___redArg(x_1, x_9);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_10);
{
lean_object* _tmp_1 = x_6;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_2, 1);
lean_inc(x_12);
lean_dec(x_2);
x_13 = 0;
x_14 = lean_box(x_13);
lean_inc(x_1);
x_15 = l_List_replicateTR___redArg(x_1, x_14);
x_16 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_3);
x_2 = x_12;
x_3 = x_16;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_eval__from__level(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, uint8_t x_6, lean_object* x_7) {
_start:
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_4);
lean_dec(x_3);
lean_inc(x_1);
x_8 = l_List_range(x_1);
x_9 = lean_box(0);
x_10 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_eval__from__level_spec__0(x_1, x_8, x_9);
x_11 = lean_box(x_6);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_5, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_5, 1);
lean_inc(x_14);
lean_dec_ref(x_5);
lean_inc(x_4);
lean_inc(x_1);
x_15 = lp_DLDSBooleanCircuit_Semantic_evaluate__layer(x_1, x_13, x_4);
if (lean_obj_tag(x_14) == 0)
{
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_1);
if (x_6 == 0)
{
return x_15;
}
else
{
uint8_t x_16; 
x_16 = !lean_is_exclusive(x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lean_ctor_get(x_15, 1);
lean_dec(x_17);
x_18 = lean_box(x_6);
lean_ctor_set(x_15, 1, x_18);
return x_15;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_19 = lean_ctor_get(x_15, 0);
lean_inc(x_19);
lean_dec(x_15);
x_20 = lean_box(x_6);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; 
x_22 = lean_ctor_get(x_15, 0);
lean_inc(x_22);
x_23 = lean_ctor_get(x_15, 1);
lean_inc(x_23);
lean_dec_ref(x_15);
x_24 = lp_DLDSBooleanCircuit_Semantic_propagate__tokens(x_1, x_4, x_2, x_3, x_7, x_22);
lean_dec(x_22);
x_25 = lean_unsigned_to_nat(1u);
x_26 = lean_nat_sub(x_3, x_25);
lean_dec(x_3);
if (x_6 == 0)
{
uint8_t x_27; 
x_27 = lean_unbox(x_23);
lean_dec(x_23);
x_3 = x_26;
x_4 = x_24;
x_5 = x_14;
x_6 = x_27;
goto _start;
}
else
{
lean_dec(x_23);
x_3 = x_26;
x_4 = x_24;
x_5 = x_14;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_eval__from__level___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
uint8_t x_8; lean_object* x_9; 
x_8 = lean_unbox(x_6);
x_9 = lp_DLDSBooleanCircuit_Semantic_eval__from__level(x_1, x_2, x_3, x_4, x_5, x_8, x_7);
lean_dec(x_7);
lean_dec(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_get__eval__result(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; lean_object* x_10; 
x_5 = l_List_lengthTR___redArg(x_2);
lean_inc(x_5);
x_6 = lp_DLDSBooleanCircuit_Semantic_initialize__tokens(x_1, x_3, x_5);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_5, x_7);
x_9 = 0;
x_10 = lp_DLDSBooleanCircuit_Semantic_eval__from__level(x_1, x_4, x_8, x_6, x_2, x_9, x_5);
lean_dec(x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_get__eval__result___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_get__eval__result(x_1, x_2, x_3, x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___lam__0(uint8_t x_1, uint8_t x_2, uint8_t x_3) {
_start:
{
if (x_3 == 0)
{
return x_1;
}
else
{
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; uint8_t x_6; uint8_t x_7; lean_object* x_8; 
x_4 = lean_unbox(x_1);
x_5 = lean_unbox(x_2);
x_6 = lean_unbox(x_3);
x_7 = lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___lam__0(x_4, x_5, x_6);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateCircuit(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lp_DLDSBooleanCircuit_Semantic_get__eval__result(x_1, x_2, x_3, x_4);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = l_List_lengthTR___redArg(x_7);
x_10 = lean_nat_dec_lt(x_5, x_9);
lean_dec(x_9);
if (x_10 == 0)
{
uint8_t x_11; 
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_5);
x_11 = 1;
return x_11;
}
else
{
uint8_t x_12; 
x_12 = lean_unbox(x_8);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_13 = lean_box(x_10);
x_14 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___lam__0___boxed), 3, 2);
lean_closure_set(x_14, 0, x_13);
lean_closure_set(x_14, 1, x_8);
x_15 = l_List_get___redArg(x_7, x_5);
lean_dec(x_7);
x_16 = l_List_all___redArg(x_15, x_14);
return x_16;
}
else
{
uint8_t x_17; 
lean_dec(x_7);
lean_dec(x_5);
x_17 = lean_unbox(x_8);
lean_dec(x_8);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateCircuit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_DLDSBooleanCircuit_Semantic_evaluateCircuit(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorIdx(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_unsigned_to_nat(0u);
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lean_unsigned_to_nat(1u);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorIdx___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorIdx(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_2, x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_2(x_2, x_5, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_atom_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_atom_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_impl_elim___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_impl_elim(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_Formula_ctorElim___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_string_dec_eq(x_3, x_4);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = 0;
return x_6;
}
}
else
{
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_7; 
x_7 = 0;
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
x_12 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_8, x_10);
if (x_12 == 0)
{
return x_12;
}
else
{
x_1 = x_9;
x_2 = x_11;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_15; uint8_t x_16; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 x_4 = x_1;
} else {
 lean_dec_ref(x_1);
 x_4 = lean_box(0);
}
x_15 = lean_unsigned_to_nat(1024u);
x_16 = lean_nat_dec_le(x_15, x_2);
if (x_16 == 0)
{
lean_object* x_17; 
x_17 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3, &lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3);
x_5 = x_17;
goto block_14;
}
else
{
lean_object* x_18; 
x_18 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4, &lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4);
x_5 = x_18;
goto block_14;
}
block_14:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; 
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__2));
x_7 = l_String_quote(x_3);
if (lean_is_scalar(x_4)) {
 x_8 = lean_alloc_ctor(3, 1, 0);
} else {
 x_8 = x_4;
 lean_ctor_set_tag(x_8, 3);
}
lean_ctor_set(x_8, 0, x_7);
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
x_13 = l_Repr_addAppParen(x_12, x_2);
return x_13;
}
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; uint8_t x_36; 
x_19 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_20);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 x_21 = x_1;
} else {
 lean_dec_ref(x_1);
 x_21 = lean_box(0);
}
x_22 = lean_unsigned_to_nat(1024u);
x_36 = lean_nat_dec_le(x_22, x_2);
if (x_36 == 0)
{
lean_object* x_37; 
x_37 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3, &lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__3);
x_23 = x_37;
goto block_35;
}
else
{
lean_object* x_38; 
x_38 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4, &lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__4);
x_23 = x_38;
goto block_35;
}
block_35:
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; lean_object* x_33; lean_object* x_34; 
x_24 = lean_box(1);
x_25 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___closed__7));
x_26 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_19, x_22);
if (lean_is_scalar(x_21)) {
 x_27 = lean_alloc_ctor(5, 2, 0);
} else {
 x_27 = x_21;
 lean_ctor_set_tag(x_27, 5);
}
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_24);
x_29 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_20, x_22);
x_30 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_30, 0, x_28);
lean_ctor_set(x_30, 1, x_29);
x_31 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_31, 0, x_23);
lean_ctor_set(x_31, 1, x_30);
x_32 = 0;
x_33 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_33, 0, x_31);
lean_ctor_set_uint8(x_33, sizeof(void*)*1, x_32);
x_34 = l_Repr_addAppParen(x_33, x_2);
return x_34;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__0));
x_6 = lp_DLDSBooleanCircuit_Semantic_Formula_toString(x_3);
x_7 = lean_string_append(x_5, x_6);
lean_dec_ref(x_6);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__1));
x_9 = lean_string_append(x_7, x_8);
x_10 = lp_DLDSBooleanCircuit_Semantic_Formula_toString(x_4);
x_11 = lean_string_append(x_9, x_10);
lean_dec_ref(x_10);
x_12 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__2));
x_13 = lean_string_append(x_11, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Formula_toString___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_DLDSBooleanCircuit_Semantic_Formula_toString(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__5(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1___closed__22));
x_2 = lean_string_length(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__5, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__5_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__5);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = l_Std_Format_joinSep___at___00Array_repr___at___00Lean_Elab_Structural_instReprRecArgInfo_repr_spec__1_spec__3(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = l_Std_Format_fill(x_10);
return x_11;
}
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(8u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(9u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(11u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(14u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(13u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__23(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__0));
x_2 = lean_string_length(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__23, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__23_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__23);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; lean_object* x_69; lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; lean_object* x_76; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_4);
x_5 = lean_ctor_get_uint8(x_1, sizeof(void*)*4);
x_6 = lean_ctor_get_uint8(x_1, sizeof(void*)*4 + 1);
x_7 = lean_ctor_get(x_1, 3);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5));
x_9 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__6));
x_10 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7);
x_11 = l_Nat_reprFast(x_2);
x_12 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_12, 0, x_11);
x_13 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_13, 0, x_10);
lean_ctor_set(x_13, 1, x_12);
x_14 = 0;
x_15 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set_uint8(x_15, sizeof(void*)*1, x_14);
x_16 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_16, 0, x_9);
lean_ctor_set(x_16, 1, x_15);
x_17 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3));
x_18 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_box(1);
x_20 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set(x_20, 1, x_19);
x_21 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__9));
x_22 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_21);
x_23 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_8);
x_24 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10);
x_25 = l_Nat_reprFast(x_3);
x_26 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_26, 0, x_25);
x_27 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_27, 0, x_24);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set_uint8(x_28, sizeof(void*)*1, x_14);
x_29 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_29, 0, x_23);
lean_ctor_set(x_29, 1, x_28);
x_30 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_17);
x_31 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set(x_31, 1, x_19);
x_32 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__12));
x_33 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_33, 0, x_31);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_8);
x_35 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13);
x_36 = lean_unsigned_to_nat(0u);
x_37 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_4, x_36);
x_38 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_38, 0, x_35);
lean_ctor_set(x_38, 1, x_37);
x_39 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set_uint8(x_39, sizeof(void*)*1, x_14);
x_40 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_40, 0, x_34);
lean_ctor_set(x_40, 1, x_39);
x_41 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_17);
x_42 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set(x_42, 1, x_19);
x_43 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__15));
x_44 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_44, 0, x_42);
lean_ctor_set(x_44, 1, x_43);
x_45 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_45, 0, x_44);
lean_ctor_set(x_45, 1, x_8);
x_46 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16);
x_47 = l_Bool_repr___redArg(x_5);
x_48 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_48, 0, x_46);
lean_ctor_set(x_48, 1, x_47);
x_49 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set_uint8(x_49, sizeof(void*)*1, x_14);
x_50 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_50, 0, x_45);
lean_ctor_set(x_50, 1, x_49);
x_51 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_51, 0, x_50);
lean_ctor_set(x_51, 1, x_17);
x_52 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_52, 0, x_51);
lean_ctor_set(x_52, 1, x_19);
x_53 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__18));
x_54 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_54, 0, x_52);
lean_ctor_set(x_54, 1, x_53);
x_55 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_55, 0, x_54);
lean_ctor_set(x_55, 1, x_8);
x_56 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19);
x_57 = l_Bool_repr___redArg(x_6);
x_58 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_58, 0, x_56);
lean_ctor_set(x_58, 1, x_57);
x_59 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_59, 0, x_58);
lean_ctor_set_uint8(x_59, sizeof(void*)*1, x_14);
x_60 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_60, 0, x_55);
lean_ctor_set(x_60, 1, x_59);
x_61 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_61, 0, x_60);
lean_ctor_set(x_61, 1, x_17);
x_62 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_62, 0, x_61);
lean_ctor_set(x_62, 1, x_19);
x_63 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__21));
x_64 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_64, 0, x_62);
lean_ctor_set(x_64, 1, x_63);
x_65 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_65, 0, x_64);
lean_ctor_set(x_65, 1, x_8);
x_66 = lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg(x_7);
x_67 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_67, 0, x_10);
lean_ctor_set(x_67, 1, x_66);
x_68 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_68, 0, x_67);
lean_ctor_set_uint8(x_68, sizeof(void*)*1, x_14);
x_69 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_69, 0, x_65);
lean_ctor_set(x_69, 1, x_68);
x_70 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24);
x_71 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25));
x_72 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_72, 0, x_71);
lean_ctor_set(x_72, 1, x_69);
x_73 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26));
x_74 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_74, 0, x_72);
lean_ctor_set(x_74, 1, x_73);
x_75 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_75, 0, x_70);
lean_ctor_set(x_75, 1, x_74);
x_76 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_76, 0, x_75);
lean_ctor_set_uint8(x_76, sizeof(void*)*1, x_14);
return x_76;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; uint8_t x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; uint8_t x_13; lean_object* x_14; uint8_t x_19; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_5);
x_6 = lean_ctor_get_uint8(x_1, sizeof(void*)*4);
x_7 = lean_ctor_get_uint8(x_1, sizeof(void*)*4 + 1);
x_8 = lean_ctor_get(x_1, 3);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_11);
x_12 = lean_ctor_get_uint8(x_2, sizeof(void*)*4);
x_13 = lean_ctor_get_uint8(x_2, sizeof(void*)*4 + 1);
x_14 = lean_ctor_get(x_2, 3);
lean_inc(x_14);
lean_dec_ref(x_2);
x_19 = lean_nat_dec_eq(x_3, x_9);
lean_dec(x_9);
lean_dec(x_3);
if (x_19 == 0)
{
lean_dec(x_14);
lean_dec_ref(x_11);
lean_dec(x_10);
lean_dec(x_8);
lean_dec_ref(x_5);
lean_dec(x_4);
return x_19;
}
else
{
uint8_t x_20; 
x_20 = lean_nat_dec_eq(x_4, x_10);
lean_dec(x_10);
lean_dec(x_4);
if (x_20 == 0)
{
lean_dec(x_14);
lean_dec_ref(x_11);
lean_dec(x_8);
lean_dec_ref(x_5);
return x_20;
}
else
{
uint8_t x_21; 
x_21 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_5, x_11);
lean_dec_ref(x_11);
lean_dec_ref(x_5);
if (x_21 == 0)
{
lean_dec(x_14);
lean_dec(x_8);
return x_21;
}
else
{
if (x_6 == 0)
{
if (x_12 == 0)
{
goto block_18;
}
else
{
lean_dec(x_14);
lean_dec(x_8);
return x_6;
}
}
else
{
if (x_12 == 0)
{
lean_dec(x_14);
lean_dec(x_8);
return x_12;
}
else
{
goto block_18;
}
}
}
}
}
block_17:
{
lean_object* x_15; uint8_t x_16; 
x_15 = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
x_16 = l_instDecidableEqList___redArg(x_15, x_8, x_14);
return x_16;
}
block_18:
{
if (x_7 == 0)
{
if (x_13 == 0)
{
goto block_17;
}
else
{
lean_dec(x_14);
lean_dec(x_8);
return x_7;
}
}
else
{
if (x_13 == 0)
{
lean_dec(x_14);
lean_dec(x_8);
return x_13;
}
else
{
goto block_17;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_5, x_7);
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
x_2 = x_9;
x_3 = x_6;
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_3, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_3);
lean_inc(x_1);
x_13 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_13, 0, x_2);
lean_ctor_set(x_13, 1, x_1);
x_14 = lean_unsigned_to_nat(0u);
x_15 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_11, x_14);
x_16 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_16, 0, x_13);
lean_ctor_set(x_16, 1, x_15);
x_2 = x_16;
x_3 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_5, x_7);
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_3);
lean_ctor_set(x_9, 1, x_8);
x_10 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1_spec__2(x_1, x_9, x_6);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lean_ctor_get(x_3, 0);
x_12 = lean_ctor_get(x_3, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_3);
lean_inc(x_1);
x_13 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_13, 0, x_2);
lean_ctor_set(x_13, 1, x_1);
x_14 = lean_unsigned_to_nat(0u);
x_15 = lp_DLDSBooleanCircuit_Semantic_instReprFormula_repr(x_11, x_14);
x_16 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_16, 0, x_13);
lean_ctor_set(x_16, 1, x_15);
x_17 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1_spec__2(x_1, x_16, x_12);
return x_17;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
lean_dec(x_2);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0___lam__0(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0___lam__0(x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0_spec__1(x_2, x_8, x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDeduction_repr_spec__0_spec__0(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
return x_12;
}
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__6(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(7u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(10u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5));
x_7 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__3));
x_8 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__10);
x_9 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_2);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
x_13 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_13, 0, x_7);
lean_ctor_set(x_13, 1, x_12);
x_14 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3));
x_15 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_box(1);
x_17 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__5));
x_19 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_6);
x_21 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__6, &lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__6);
x_22 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_3);
x_23 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set_uint8(x_24, sizeof(void*)*1, x_11);
x_25 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_25, 0, x_20);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_14);
x_27 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_16);
x_28 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__8));
x_29 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_29, 0, x_27);
lean_ctor_set(x_29, 1, x_28);
x_30 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_6);
x_31 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9, &lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9);
x_32 = l_Nat_reprFast(x_4);
x_33 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_33, 0, x_32);
x_34 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_34, 0, x_31);
lean_ctor_set(x_34, 1, x_33);
x_35 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_35, 0, x_34);
lean_ctor_set_uint8(x_35, sizeof(void*)*1, x_11);
x_36 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_36, 0, x_30);
lean_ctor_set(x_36, 1, x_35);
x_37 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_14);
x_38 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_16);
x_39 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__11));
x_40 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_40, 0, x_38);
lean_ctor_set(x_40, 1, x_39);
x_41 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_41, 0, x_40);
lean_ctor_set(x_41, 1, x_6);
x_42 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16);
x_43 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0___redArg(x_5);
x_44 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_44, 0, x_42);
lean_ctor_set(x_44, 1, x_43);
x_45 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_45, 0, x_44);
lean_ctor_set_uint8(x_45, sizeof(void*)*1, x_11);
x_46 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_46, 0, x_41);
lean_ctor_set(x_46, 1, x_45);
x_47 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24);
x_48 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25));
x_49 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_49, 0, x_48);
lean_ctor_set(x_49, 1, x_46);
x_50 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26));
x_51 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_51, 0, x_49);
lean_ctor_set(x_51, 1, x_50);
x_52 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_52, 0, x_47);
lean_ctor_set(x_52, 1, x_51);
x_53 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_53, 0, x_52);
lean_ctor_set_uint8(x_53, sizeof(void*)*1, x_11);
return x_53;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDeduction_repr_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 3);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_2, 2);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 3);
lean_inc(x_10);
lean_dec_ref(x_2);
x_11 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_3, x_7);
if (x_11 == 0)
{
lean_dec(x_10);
lean_dec(x_9);
lean_dec_ref(x_8);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_11;
}
else
{
uint8_t x_12; 
x_12 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_4, x_8);
if (x_12 == 0)
{
lean_dec(x_10);
lean_dec(x_9);
lean_dec(x_6);
lean_dec(x_5);
return x_12;
}
else
{
uint8_t x_13; 
x_13 = lean_nat_dec_eq(x_5, x_9);
lean_dec(x_9);
lean_dec(x_5);
if (x_13 == 0)
{
lean_dec(x_10);
lean_dec(x_6);
return x_13;
}
else
{
lean_object* x_14; uint8_t x_15; 
x_14 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula___boxed), 2, 0);
x_15 = l_instDecidableEqList___redArg(x_14, x_6, x_10);
return x_15;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction_decEq(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction_decEq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Formula_toString___closed__0));
x_2 = lean_string_length(x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__0, &lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__0_once, _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__0);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; uint8_t x_18; lean_object* x_19; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_3);
x_6 = lean_box(0);
lean_ctor_set_tag(x_1, 1);
lean_ctor_set(x_1, 1, x_6);
lean_ctor_set(x_1, 0, x_5);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_4);
x_8 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_1);
x_9 = l_List_reverse___redArg(x_8);
x_10 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_11 = lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(x_9, x_10);
x_12 = lean_obj_once(&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1, &lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1_once, _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1);
x_13 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2));
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_11);
x_15 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3));
x_16 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
x_17 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_17, 0, x_12);
lean_ctor_set(x_17, 1, x_16);
x_18 = 0;
x_19 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set_uint8(x_19, sizeof(void*)*1, x_18);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; uint8_t x_36; lean_object* x_37; 
x_20 = lean_ctor_get(x_1, 0);
x_21 = lean_ctor_get(x_1, 1);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_1);
x_22 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_20);
x_23 = lean_box(0);
x_24 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_23);
x_25 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_21);
x_26 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_24);
x_27 = l_List_reverse___redArg(x_26);
x_28 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_29 = lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(x_27, x_28);
x_30 = lean_obj_once(&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1, &lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1_once, _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1);
x_31 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2));
x_32 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_32, 0, x_31);
lean_ctor_set(x_32, 1, x_29);
x_33 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3));
x_34 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_34, 0, x_32);
lean_ctor_set(x_34, 1, x_33);
x_35 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_35, 0, x_30);
lean_ctor_set(x_35, 1, x_34);
x_36 = 0;
x_37 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set_uint8(x_37, sizeof(void*)*1, x_36);
return x_37;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8_spec__11(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_2 = x_8;
x_3 = x_6;
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_2 = x_14;
x_3 = x_11;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8_spec__11(x_1, x_8, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8_spec__11(x_1, x_14, x_11);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
lean_dec(x_2);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5_spec__8(x_2, x_8, x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__5(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_2 = x_8;
x_3 = x_6;
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_2 = x_14;
x_3 = x_11;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1_spec__4(x_1, x_8, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1_spec__4(x_1, x_14, x_11);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
lean_dec(x_2);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0_spec__1(x_2, x_8, x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__0_spec__0(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4_spec__7(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_2 = x_8;
x_3 = x_6;
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_2 = x_14;
x_3 = x_11;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4_spec__7(x_1, x_8, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4_spec__7(x_1, x_14, x_11);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
lean_dec(x_2);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg(x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2_spec__4(x_2, x_8, x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__1_spec__2(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
return x_12;
}
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__4(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(5u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5));
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__3));
x_7 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__4, &lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__4_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__4);
x_8 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___redArg(x_2);
x_9 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = 0;
x_11 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set_uint8(x_11, sizeof(void*)*1, x_10);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_6);
lean_ctor_set(x_12, 1, x_11);
x_13 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3));
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lean_box(1);
x_16 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
x_17 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__6));
x_18 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_5);
x_20 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1___redArg(x_3);
x_21 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_21, 0, x_7);
lean_ctor_set(x_21, 1, x_20);
x_22 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set_uint8(x_22, sizeof(void*)*1, x_10);
x_23 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_23, 0, x_19);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_13);
x_25 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_25, 0, x_24);
lean_ctor_set(x_25, 1, x_15);
x_26 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg___closed__8));
x_27 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
x_28 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_5);
x_29 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2___redArg(x_4);
x_30 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_30, 0, x_7);
lean_ctor_set(x_30, 1, x_29);
x_31 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_31, 0, x_30);
lean_ctor_set_uint8(x_31, sizeof(void*)*1, x_10);
x_32 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_32, 0, x_28);
lean_ctor_set(x_32, 1, x_31);
x_33 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24);
x_34 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25));
x_35 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_35, 0, x_34);
lean_ctor_set(x_35, 1, x_32);
x_36 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26));
x_37 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
x_38 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_38, 0, x_33);
lean_ctor_set(x_38, 1, x_37);
x_39 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_39, 0, x_38);
lean_ctor_set_uint8(x_39, sizeof(void*)*1, x_10);
return x_39;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__1(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__2(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
lean_inc_ref(x_1);
x_4 = l_instDecidableEqProd___redArg(x_1, x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex___boxed), 2, 0);
x_2 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___lam__0___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_2, 2);
lean_inc(x_8);
lean_dec_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex___boxed), 2, 0);
x_10 = l_instDecidableEqList___redArg(x_9, x_3, x_6);
if (x_10 == 0)
{
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec(x_4);
return x_10;
}
else
{
lean_object* x_11; uint8_t x_12; 
x_11 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqDeduction___boxed), 2, 0);
x_12 = l_instDecidableEqList___redArg(x_11, x_4, x_7);
if (x_12 == 0)
{
lean_dec(x_8);
lean_dec(x_5);
return x_12;
}
else
{
lean_object* x_13; uint8_t x_14; 
x_13 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___closed__0, &lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___closed__0);
x_14 = l_instDecidableEqList___redArg(x_13, x_5, x_8);
return x_14;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS_decEq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqDLDS(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1___closed__0));
x_3 = l_List_eraseDupsBy___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildFormulas_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 2);
lean_inc_ref(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 2);
lean_inc_ref(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildFormulas(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_box(0);
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildFormulas_spec__0(x_2, x_3);
x_5 = lp_DLDSBooleanCircuit_List_eraseDups___at___00Semantic_buildFormulas_spec__1(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_encoderForIntro_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_6, x_1);
lean_dec(x_6);
x_9 = lean_box(x_8);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_9);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_2, 0);
x_12 = lean_ctor_get(x_2, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_2);
x_13 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_11, x_1);
lean_dec(x_11);
x_14 = lean_box(x_13);
x_15 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_3);
x_2 = x_12;
x_3 = x_15;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_encoderForIntro_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_encoderForIntro_spec__0(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_encoderForIntro(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 1)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_box(0);
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_encoderForIntro_spec__0(x_3, x_1, x_4);
x_6 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_6, 0, x_5);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_1);
x_7 = lean_box(0);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_encoderForIntro___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_encoderForIntro(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_4, x_1);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_add(x_3, x_7);
lean_dec(x_3);
x_2 = x_5;
x_3 = x_8;
goto _start;
}
else
{
return x_3;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0_spec__0(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0_spec__0(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_buildIncomingMapForFormula_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
x_5 = lean_array_to_list(x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
if (lean_obj_tag(x_7) == 1)
{
uint8_t x_8; 
x_8 = !lean_is_exclusive(x_3);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_9 = lean_ctor_get(x_3, 1);
x_10 = lean_ctor_get(x_3, 0);
lean_dec(x_10);
x_11 = !lean_is_exclusive(x_6);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_12 = lean_ctor_get(x_6, 1);
x_13 = lean_ctor_get(x_6, 0);
lean_dec(x_13);
x_14 = !lean_is_exclusive(x_7);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_15 = lean_ctor_get(x_7, 0);
x_16 = lean_ctor_get(x_7, 1);
x_17 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_16, x_1);
lean_dec_ref(x_16);
if (x_17 == 0)
{
lean_free_object(x_7);
lean_dec_ref(x_15);
lean_free_object(x_6);
lean_dec(x_12);
lean_free_object(x_3);
x_3 = x_9;
goto _start;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_15, x_2);
lean_dec_ref(x_15);
x_20 = lean_unsigned_to_nat(0u);
lean_ctor_set(x_6, 1, x_20);
lean_ctor_set(x_6, 0, x_12);
lean_ctor_set_tag(x_7, 0);
lean_ctor_set(x_7, 1, x_20);
lean_ctor_set(x_7, 0, x_19);
x_21 = lean_box(0);
lean_ctor_set(x_3, 1, x_21);
lean_ctor_set(x_3, 0, x_7);
x_22 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_22, 0, x_6);
lean_ctor_set(x_22, 1, x_3);
x_23 = lean_array_push(x_4, x_22);
x_3 = x_9;
x_4 = x_23;
goto _start;
}
}
else
{
lean_object* x_25; lean_object* x_26; uint8_t x_27; 
x_25 = lean_ctor_get(x_7, 0);
x_26 = lean_ctor_get(x_7, 1);
lean_inc(x_26);
lean_inc(x_25);
lean_dec(x_7);
x_27 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_26, x_1);
lean_dec_ref(x_26);
if (x_27 == 0)
{
lean_dec_ref(x_25);
lean_free_object(x_6);
lean_dec(x_12);
lean_free_object(x_3);
x_3 = x_9;
goto _start;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_29 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_25, x_2);
lean_dec_ref(x_25);
x_30 = lean_unsigned_to_nat(0u);
lean_ctor_set(x_6, 1, x_30);
lean_ctor_set(x_6, 0, x_12);
x_31 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_31, 0, x_29);
lean_ctor_set(x_31, 1, x_30);
x_32 = lean_box(0);
lean_ctor_set(x_3, 1, x_32);
lean_ctor_set(x_3, 0, x_31);
x_33 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_33, 0, x_6);
lean_ctor_set(x_33, 1, x_3);
x_34 = lean_array_push(x_4, x_33);
x_3 = x_9;
x_4 = x_34;
goto _start;
}
}
}
else
{
lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; uint8_t x_40; 
x_36 = lean_ctor_get(x_6, 1);
lean_inc(x_36);
lean_dec(x_6);
x_37 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_37);
x_38 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_38);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 x_39 = x_7;
} else {
 lean_dec_ref(x_7);
 x_39 = lean_box(0);
}
x_40 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_38, x_1);
lean_dec_ref(x_38);
if (x_40 == 0)
{
lean_dec(x_39);
lean_dec_ref(x_37);
lean_dec(x_36);
lean_free_object(x_3);
x_3 = x_9;
goto _start;
}
else
{
lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_42 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_37, x_2);
lean_dec_ref(x_37);
x_43 = lean_unsigned_to_nat(0u);
x_44 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_44, 0, x_36);
lean_ctor_set(x_44, 1, x_43);
if (lean_is_scalar(x_39)) {
 x_45 = lean_alloc_ctor(0, 2, 0);
} else {
 x_45 = x_39;
 lean_ctor_set_tag(x_45, 0);
}
lean_ctor_set(x_45, 0, x_42);
lean_ctor_set(x_45, 1, x_43);
x_46 = lean_box(0);
lean_ctor_set(x_3, 1, x_46);
lean_ctor_set(x_3, 0, x_45);
x_47 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_47, 0, x_44);
lean_ctor_set(x_47, 1, x_3);
x_48 = lean_array_push(x_4, x_47);
x_3 = x_9;
x_4 = x_48;
goto _start;
}
}
}
else
{
lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; uint8_t x_56; 
x_50 = lean_ctor_get(x_3, 1);
lean_inc(x_50);
lean_dec(x_3);
x_51 = lean_ctor_get(x_6, 1);
lean_inc(x_51);
if (lean_is_exclusive(x_6)) {
 lean_ctor_release(x_6, 0);
 lean_ctor_release(x_6, 1);
 x_52 = x_6;
} else {
 lean_dec_ref(x_6);
 x_52 = lean_box(0);
}
x_53 = lean_ctor_get(x_7, 0);
lean_inc_ref(x_53);
x_54 = lean_ctor_get(x_7, 1);
lean_inc_ref(x_54);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 lean_ctor_release(x_7, 1);
 x_55 = x_7;
} else {
 lean_dec_ref(x_7);
 x_55 = lean_box(0);
}
x_56 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_54, x_1);
lean_dec_ref(x_54);
if (x_56 == 0)
{
lean_dec(x_55);
lean_dec_ref(x_53);
lean_dec(x_52);
lean_dec(x_51);
x_3 = x_50;
goto _start;
}
else
{
lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; 
x_58 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_53, x_2);
lean_dec_ref(x_53);
x_59 = lean_unsigned_to_nat(0u);
if (lean_is_scalar(x_52)) {
 x_60 = lean_alloc_ctor(0, 2, 0);
} else {
 x_60 = x_52;
}
lean_ctor_set(x_60, 0, x_51);
lean_ctor_set(x_60, 1, x_59);
if (lean_is_scalar(x_55)) {
 x_61 = lean_alloc_ctor(0, 2, 0);
} else {
 x_61 = x_55;
 lean_ctor_set_tag(x_61, 0);
}
lean_ctor_set(x_61, 0, x_58);
lean_ctor_set(x_61, 1, x_59);
x_62 = lean_box(0);
x_63 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_63, 0, x_61);
lean_ctor_set(x_63, 1, x_62);
x_64 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_64, 0, x_60);
lean_ctor_set(x_64, 1, x_63);
x_65 = lean_array_push(x_4, x_64);
x_3 = x_50;
x_4 = x_65;
goto _start;
}
}
}
else
{
lean_object* x_67; 
lean_dec(x_7);
lean_dec(x_6);
x_67 = lean_ctor_get(x_3, 1);
lean_inc(x_67);
lean_dec_ref(x_3);
x_3 = x_67;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_buildIncomingMapForFormula_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_buildIncomingMapForFormula_spec__1(x_1, x_2, x_3, x_4);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
if (lean_obj_tag(x_2) == 1)
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_2, 1);
x_17 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_16, x_1);
x_18 = lean_unsigned_to_nat(0u);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_box(0);
x_21 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_21, 0, x_19);
lean_ctor_set(x_21, 1, x_20);
x_22 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_20);
x_3 = x_22;
goto block_15;
}
else
{
lean_object* x_23; 
x_23 = lean_box(0);
x_3 = x_23;
goto block_15;
}
block_15:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_unsigned_to_nat(0u);
lean_inc(x_1);
x_5 = l_List_zipIdxTR___redArg(x_1, x_4);
x_6 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___closed__0, &lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___closed__0);
x_7 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_buildIncomingMapForFormula_spec__1(x_2, x_1, x_5, x_6);
x_8 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_buildIncomingMapForFormula_spec__0(x_2, x_1);
lean_dec(x_1);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_4);
x_10 = lean_box(0);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
x_13 = l_List_appendTR___redArg(x_3, x_7);
x_14 = l_List_appendTR___redArg(x_13, x_12);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildIncomingMap_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_1);
x_8 = lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula(x_1, x_6);
lean_dec(x_6);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_8);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
lean_inc(x_1);
x_12 = lp_DLDSBooleanCircuit_Semantic_buildIncomingMapForFormula(x_1, x_10);
lean_dec(x_10);
x_13 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_3);
x_2 = x_11;
x_3 = x_13;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildIncomingMap(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_box(0);
lean_inc(x_1);
x_3 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildIncomingMap_spec__0(x_1, x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_3);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_5);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_2);
x_7 = lean_apply_2(x_3, x_1, lean_box(0));
return x_7;
}
}
else
{
lean_object* x_8; 
lean_dec(x_2);
x_8 = lean_apply_2(x_3, x_1, lean_box(0));
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_3) == 1)
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_3, 1);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; lean_object* x_8; 
lean_dec(x_5);
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lean_apply_1(x_4, x_7);
return x_8;
}
else
{
lean_object* x_9; 
lean_dec(x_4);
x_9 = lean_apply_2(x_5, x_3, lean_box(0));
return x_9;
}
}
else
{
lean_object* x_10; 
lean_dec(x_4);
x_10 = lean_apply_2(x_5, x_3, lean_box(0));
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkIntroRule_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 1)
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_4, 1);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_4);
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_4, 0);
lean_inc(x_7);
lean_dec_ref(x_4);
x_8 = lean_apply_2(x_2, x_6, x_7);
return x_8;
}
else
{
lean_object* x_9; 
lean_dec(x_2);
x_9 = lean_apply_2(x_3, x_1, lean_box(0));
return x_9;
}
}
else
{
lean_object* x_10; 
lean_dec(x_2);
x_10 = lean_apply_2(x_3, x_1, lean_box(0));
return x_10;
}
}
else
{
lean_object* x_11; 
lean_dec(x_2);
x_11 = lean_apply_2(x_3, x_1, lean_box(0));
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_3) == 1)
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_3, 1);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_7; 
x_7 = lean_ctor_get(x_6, 1);
if (lean_obj_tag(x_7) == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_6);
lean_dec(x_5);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_ctor_get(x_6, 0);
lean_inc(x_9);
lean_dec_ref(x_6);
x_10 = lean_apply_2(x_4, x_8, x_9);
return x_10;
}
else
{
lean_object* x_11; 
lean_dec(x_4);
x_11 = lean_apply_2(x_5, x_3, lean_box(0));
return x_11;
}
}
else
{
lean_object* x_12; 
lean_dec(x_4);
x_12 = lean_apply_2(x_5, x_3, lean_box(0));
return x_12;
}
}
else
{
lean_object* x_13; 
lean_dec(x_4);
x_13 = lean_apply_2(x_5, x_3, lean_box(0));
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_mkElimRule_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_nodeForFormula_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = lean_array_to_list(x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_5, 0);
if (lean_obj_tag(x_6) == 1)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
lean_inc_ref(x_6);
lean_inc(x_5);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
x_9 = lean_ctor_get(x_6, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_6);
x_10 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqFormula_decEq(x_9, x_1);
lean_dec_ref(x_9);
if (x_10 == 0)
{
lean_dec(x_8);
x_2 = x_7;
goto _start;
}
else
{
lean_object* x_12; 
x_12 = lean_array_push(x_3, x_8);
x_2 = x_7;
x_3 = x_12;
goto _start;
}
}
else
{
lean_object* x_14; 
x_14 = lean_ctor_get(x_2, 1);
lean_inc(x_14);
lean_dec_ref(x_2);
x_2 = x_14;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_nodeForFormula_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_nodeForFormula_spec__0(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_5; 
x_5 = l_List_reverse___redArg(x_4);
return x_5;
}
else
{
uint8_t x_6; 
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 1);
x_9 = lean_ctor_get(x_7, 1);
lean_inc(x_9);
lean_dec(x_7);
x_10 = l_List_lengthTR___redArg(x_1);
x_11 = l_List_lengthTR___redArg(x_2);
x_12 = lean_nat_add(x_11, x_9);
lean_dec(x_9);
lean_dec(x_11);
x_13 = 0;
x_14 = lp_DLDSBooleanCircuit_Semantic_mkElimRule(x_10, x_12, x_13, x_13);
lean_ctor_set(x_3, 1, x_4);
lean_ctor_set(x_3, 0, x_14);
{
lean_object* _tmp_2 = x_8;
lean_object* _tmp_3 = x_3;
x_3 = _tmp_2;
x_4 = _tmp_3;
}
goto _start;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; uint8_t x_22; lean_object* x_23; lean_object* x_24; 
x_16 = lean_ctor_get(x_3, 0);
x_17 = lean_ctor_get(x_3, 1);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_3);
x_18 = lean_ctor_get(x_16, 1);
lean_inc(x_18);
lean_dec(x_16);
x_19 = l_List_lengthTR___redArg(x_1);
x_20 = l_List_lengthTR___redArg(x_2);
x_21 = lean_nat_add(x_20, x_18);
lean_dec(x_18);
lean_dec(x_20);
x_22 = 0;
x_23 = lp_DLDSBooleanCircuit_Semantic_mkElimRule(x_19, x_21, x_22, x_22);
x_24 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_4);
x_3 = x_17;
x_4 = x_24;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__2(x_1, x_2, x_3, x_4);
lean_dec(x_2);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec(x_6);
x_10 = l_List_lengthTR___redArg(x_1);
x_11 = 0;
x_12 = lp_DLDSBooleanCircuit_Semantic_mkIntroRule(x_10, x_9, x_8, x_11);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_12);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; lean_object* x_21; 
x_14 = lean_ctor_get(x_2, 0);
x_15 = lean_ctor_get(x_2, 1);
lean_inc(x_15);
lean_inc(x_14);
lean_dec(x_2);
x_16 = lean_ctor_get(x_14, 0);
lean_inc(x_16);
x_17 = lean_ctor_get(x_14, 1);
lean_inc(x_17);
lean_dec(x_14);
x_18 = l_List_lengthTR___redArg(x_1);
x_19 = 0;
x_20 = lp_DLDSBooleanCircuit_Semantic_mkIntroRule(x_18, x_17, x_16, x_19);
x_21 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_3);
x_2 = x_15;
x_3 = x_21;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__1(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_nodeForFormula___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_nodeForFormula(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
if (lean_obj_tag(x_2) == 1)
{
lean_object* x_23; 
lean_inc(x_1);
x_23 = lp_DLDSBooleanCircuit_Semantic_encoderForIntro(x_1, x_2);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; 
x_24 = lean_box(0);
x_3 = x_24;
goto block_22;
}
else
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; 
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_box(0);
x_27 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_27, 0, x_25);
lean_ctor_set(x_27, 1, x_26);
x_3 = x_27;
goto block_22;
}
}
else
{
lean_object* x_28; 
x_28 = lean_box(0);
x_3 = x_28;
goto block_22;
}
block_22:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_4 = lean_unsigned_to_nat(0u);
lean_inc(x_1);
x_5 = l_List_zipIdxTR___redArg(x_1, x_4);
x_6 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_nodeForFormula___closed__0, &lp_DLDSBooleanCircuit_Semantic_nodeForFormula___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_nodeForFormula___closed__0);
x_7 = lp_DLDSBooleanCircuit_List_filterMapTR_go___at___00Semantic_nodeForFormula_spec__0(x_2, x_5, x_6);
lean_inc(x_3);
x_8 = l_List_zipIdxTR___redArg(x_3, x_4);
x_9 = lean_box(0);
x_10 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__1(x_1, x_8, x_9);
lean_inc(x_7);
x_11 = l_List_zipIdxTR___redArg(x_7, x_4);
x_12 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_nodeForFormula_spec__2(x_1, x_3, x_11, x_9);
x_13 = l_List_lengthTR___redArg(x_1);
lean_dec(x_1);
x_14 = l_List_lengthTR___redArg(x_3);
lean_dec(x_3);
x_15 = l_List_lengthTR___redArg(x_7);
lean_dec(x_7);
x_16 = lean_nat_add(x_14, x_15);
lean_dec(x_15);
lean_dec(x_14);
x_17 = 0;
x_18 = lp_DLDSBooleanCircuit_Semantic_mkRepetitionRule(x_13, x_16, x_17);
x_19 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_9);
x_20 = l_List_appendTR___redArg(x_10, x_12);
x_21 = l_List_appendTR___redArg(x_20, x_19);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_nodeForFormula___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_nodeForFormula(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildLayers_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildLayers_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_1);
x_8 = lp_DLDSBooleanCircuit_Semantic_nodeForFormula(x_1, x_6);
lean_dec(x_6);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_8);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_10 = lean_ctor_get(x_2, 0);
x_11 = lean_ctor_get(x_2, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_2);
lean_inc(x_1);
x_12 = lp_DLDSBooleanCircuit_Semantic_nodeForFormula(x_1, x_10);
lean_dec(x_10);
x_13 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_3);
x_2 = x_11;
x_3 = x_13;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildLayers(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lp_DLDSBooleanCircuit_Semantic_buildFormulas(x_1);
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_box(0);
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildLayers_spec__0(x_2, x_5);
x_7 = lp_mathlib_List_foldl___at___00List_max_x3f___at___00Mathlib_CountHeartbeats_variation_spec__3_spec__3(x_4, x_6);
lean_dec(x_6);
x_8 = lean_unsigned_to_nat(1u);
x_9 = lean_nat_add(x_7, x_8);
lean_dec(x_7);
lean_inc_n(x_3, 2);
x_10 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_buildLayers_spec__1(x_3, x_3, x_5);
x_11 = lp_DLDSBooleanCircuit_Semantic_buildIncomingMap(x_3);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
x_13 = l_List_replicateTR___redArg(x_9, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_buildGridFromDLDS(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_DLDSBooleanCircuit_Semantic_buildLayers(x_1);
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_nat_dec_eq(x_6, x_1);
lean_dec(x_6);
x_9 = lean_box(x_8);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_9);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; lean_object* x_15; 
x_11 = lean_ctor_get(x_2, 0);
x_12 = lean_ctor_get(x_2, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_2);
x_13 = lean_nat_dec_eq(x_11, x_1);
lean_dec(x_11);
x_14 = lean_box(x_13);
x_15 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_3);
x_2 = x_12;
x_3 = x_15;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_1);
x_8 = l_List_range(x_1);
x_9 = lean_box(0);
x_10 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0(x_6, x_8, x_9);
lean_dec(x_6);
lean_ctor_set(x_2, 1, x_3);
lean_ctor_set(x_2, 0, x_10);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
lean_inc(x_1);
x_14 = l_List_range(x_1);
x_15 = lean_box(0);
x_16 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0(x_12, x_14, x_15);
lean_dec(x_12);
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_3);
x_2 = x_13;
x_3 = x_17;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_initialVectorsFromDLDS(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_DLDSBooleanCircuit_Semantic_buildFormulas(x_1);
x_3 = l_List_lengthTR___redArg(x_2);
lean_dec(x_2);
lean_inc(x_3);
x_4 = l_List_range(x_3);
x_5 = lean_box(0);
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__1(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_encoderForIntro_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 1)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_dec(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_2(x_2, x_4, x_5);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_2);
x_7 = lean_apply_2(x_3, x_1, lean_box(0));
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_encoderForIntro_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_2) == 1)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_dec(x_4);
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_6);
lean_dec_ref(x_2);
x_7 = lean_apply_2(x_3, x_5, x_6);
return x_7;
}
else
{
lean_object* x_8; 
lean_dec(x_3);
x_8 = lean_apply_2(x_4, x_2, lean_box(0));
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_2);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_4);
x_6 = lean_box(0);
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; 
lean_dec(x_5);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_apply_1(x_4, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_nodeForFormula__nodupIds_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateDLDS(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
lean_inc_ref(x_1);
x_4 = lp_DLDSBooleanCircuit_Semantic_buildGridFromDLDS(x_1);
lean_inc_ref(x_1);
x_5 = lp_DLDSBooleanCircuit_Semantic_initialVectorsFromDLDS(x_1);
x_6 = lp_DLDSBooleanCircuit_Semantic_buildFormulas(x_1);
x_7 = l_List_lengthTR___redArg(x_6);
lean_dec(x_6);
x_8 = lp_DLDSBooleanCircuit_Semantic_evaluateCircuit(x_7, x_4, x_5, x_2, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateDLDS___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_DLDSBooleanCircuit_Semantic_evaluateDLDS(x_1, x_2, x_3);
lean_dec(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_readingToPath_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_dec(x_6);
x_7 = lean_box(0);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_5;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_dec(x_1);
x_10 = lean_box(0);
x_11 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_2);
x_1 = x_9;
x_2 = x_11;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingToPath___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_DLDSBooleanCircuit_Semantic_buildFormulas(x_1);
x_3 = lean_box(0);
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_readingToPath_spec__0(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingToPath(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_readingToPath___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingToPath___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_readingToPath(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_3 = lp_DLDSBooleanCircuit_Semantic_readingToPath___redArg(x_1);
x_4 = lp_DLDSBooleanCircuit_Semantic_evaluateDLDS(x_1, x_3, x_2);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___redArg(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_DLDSBooleanCircuit_Semantic_evaluateDLDSReading(x_1, x_2, x_3);
lean_dec(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_numHyps_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*4);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_dec_ref(x_1);
x_1 = x_6;
goto _start;
}
else
{
uint8_t x_8; 
lean_inc(x_4);
x_8 = !lean_is_exclusive(x_1);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_1, 1);
x_10 = lean_ctor_get(x_1, 0);
lean_dec(x_10);
lean_ctor_set(x_1, 1, x_2);
{
lean_object* _tmp_0 = x_9;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_1, 1);
lean_inc(x_12);
lean_dec(x_1);
x_13 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_13, 0, x_4);
lean_ctor_set(x_13, 1, x_2);
x_1 = x_12;
x_2 = x_13;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_numHyps(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
lean_dec_ref(x_1);
x_3 = lean_box(0);
x_4 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_numHyps_spec__0(x_2, x_3);
x_5 = l_List_lengthTR___redArg(x_4);
lean_dec(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lp_DLDSBooleanCircuit_Semantic_numHyps(x_1);
x_3 = 0;
x_4 = lean_box(x_3);
x_5 = l_List_replicateTR___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_DLDSBooleanCircuit_Semantic_numHyps(x_1);
x_4 = l_List_range(x_3);
x_5 = lean_box(0);
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_initialVectorsFromDLDS_spec__0(x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_mkElimRule___closed__0));
x_4 = lp_mathlib_List_Vector_zipWith___redArg(x_3, x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_or(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_or(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_HypDepVec_clearBit_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 x_7 = x_2;
} else {
 lean_dec_ref(x_2);
 x_7 = lean_box(0);
}
x_13 = lean_ctor_get(x_5, 0);
lean_inc(x_13);
x_14 = lean_ctor_get(x_5, 1);
lean_inc(x_14);
lean_dec(x_5);
x_15 = lean_nat_dec_eq(x_14, x_1);
lean_dec(x_14);
if (x_15 == 0)
{
uint8_t x_16; 
x_16 = lean_unbox(x_13);
lean_dec(x_13);
x_8 = x_16;
goto block_12;
}
else
{
uint8_t x_17; 
lean_dec(x_13);
x_17 = 0;
x_8 = x_17;
goto block_12;
}
block_12:
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_box(x_8);
if (lean_is_scalar(x_7)) {
 x_10 = lean_alloc_ctor(1, 2, 0);
} else {
 x_10 = x_7;
}
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_3);
x_2 = x_6;
x_3 = x_10;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_HypDepVec_clearBit_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_HypDepVec_clearBit_spec__0(x_1, x_2, x_3);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = l_List_zipIdxTR___redArg(x_2, x_3);
x_5 = lean_box(0);
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_HypDepVec_clearBit_spec__0(x_1, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___redArg(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_clearBit(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_hypIndex_spec__0_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_4, x_1);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_add(x_3, x_7);
lean_dec(x_3);
x_2 = x_5;
x_3 = x_8;
goto _start;
}
else
{
lean_dec(x_5);
lean_dec_ref(x_1);
return x_3;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_hypIndex_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lp_DLDSBooleanCircuit_List_findIdx_go___at___00List_idxOf___at___00Semantic_hypIndex_spec__0_spec__0(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_hypIndex(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_box(0);
lean_inc(x_3);
x_5 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_numHyps_spec__0(x_3, x_4);
x_6 = lp_DLDSBooleanCircuit_List_idxOf___at___00Semantic_hypIndex_spec__0(x_2, x_5);
x_7 = lp_DLDSBooleanCircuit_Semantic_numHyps(x_1);
x_8 = lean_nat_dec_lt(x_6, x_7);
lean_dec(x_7);
if (x_8 == 0)
{
lean_object* x_9; 
lean_dec(x_6);
x_9 = lean_box(0);
return x_9;
}
else
{
lean_object* x_10; 
x_10 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_10, 0, x_6);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; uint8_t x_19; lean_object* x_20; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = l_Nat_reprFast(x_3);
x_6 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_6, 0, x_5);
x_7 = lean_box(0);
lean_ctor_set_tag(x_1, 1);
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_6);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_4);
x_9 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_1);
x_10 = l_List_reverse___redArg(x_9);
x_11 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_12 = lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(x_10, x_11);
x_13 = lean_obj_once(&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1, &lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1_once, _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1);
x_14 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2));
x_15 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_12);
x_16 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3));
x_17 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_18, 0, x_13);
lean_ctor_set(x_18, 1, x_17);
x_19 = 0;
x_20 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_20, 0, x_18);
lean_ctor_set_uint8(x_20, sizeof(void*)*1, x_19);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; lean_object* x_39; 
x_21 = lean_ctor_get(x_1, 0);
x_22 = lean_ctor_get(x_1, 1);
lean_inc(x_22);
lean_inc(x_21);
lean_dec(x_1);
x_23 = l_Nat_reprFast(x_21);
x_24 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_24, 0, x_23);
x_25 = lean_box(0);
x_26 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_26, 0, x_24);
lean_ctor_set(x_26, 1, x_25);
x_27 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_22);
x_28 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_26);
x_29 = l_List_reverse___redArg(x_28);
x_30 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_31 = lp_mathlib_Std_Format_joinSep___at___00Prod_repr___at___00List_repr___at___00Mathlib_Tactic_Linarith_instReprComp_repr_spec__0_spec__0_spec__2(x_29, x_30);
x_32 = lean_obj_once(&lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1, &lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1_once, _init_lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__1);
x_33 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__2));
x_34 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_31);
x_35 = ((lean_object*)(lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprDLDS_repr_spec__2_spec__4___redArg___closed__3));
x_36 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_36, 0, x_34);
lean_ctor_set(x_36, 1, x_35);
x_37 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_37, 0, x_32);
lean_ctor_set(x_37, 1, x_36);
x_38 = 0;
x_39 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_39, 0, x_37);
lean_ctor_set_uint8(x_39, sizeof(void*)*1, x_38);
return x_39;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_2 = x_8;
x_3 = x_6;
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_2 = x_14;
x_3 = x_11;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2_spec__3(x_1, x_8, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2_spec__3(x_1, x_14, x_11);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
lean_dec(x_2);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1_spec__2(x_2, x_8, x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__1(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5));
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__3));
x_7 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9, &lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprDeduction_repr___redArg___closed__9);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg(x_2);
x_9 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = 0;
x_11 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set_uint8(x_11, sizeof(void*)*1, x_10);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_6);
lean_ctor_set(x_12, 1, x_11);
x_13 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3));
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lean_box(1);
x_16 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
x_17 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__5));
x_18 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_5);
x_20 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16);
x_21 = l_Nat_reprFast(x_3);
x_22 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_22, 0, x_21);
x_23 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_23, 0, x_20);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set_uint8(x_24, sizeof(void*)*1, x_10);
x_25 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_25, 0, x_19);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_13);
x_27 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_15);
x_28 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg___closed__7));
x_29 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_29, 0, x_27);
lean_ctor_set(x_29, 1, x_28);
x_30 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_5);
x_31 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__13);
x_32 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0___redArg(x_4);
x_33 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_33, 0, x_31);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set_uint8(x_34, sizeof(void*)*1, x_10);
x_35 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_35, 0, x_30);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24);
x_37 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25));
x_38 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_35);
x_39 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26));
x_40 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_40, 0, x_38);
lean_ctor_set(x_40, 1, x_39);
x_41 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_41, 0, x_36);
lean_ctor_set(x_41, 1, x_40);
x_42 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_42, 0, x_41);
lean_ctor_set_uint8(x_42, sizeof(void*)*1, x_10);
return x_42;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranching_repr_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Prod_repr___at___00List_repr___at___00Semantic_instReprBranching_repr_spec__0_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
x_4 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex___boxed), 2, 0);
x_5 = l_instDecidableEqProd___redArg(x_3, x_4, x_1, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___lam__0(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_2, 2);
lean_inc(x_8);
lean_dec_ref(x_2);
x_9 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_3, x_6);
if (x_9 == 0)
{
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_5);
lean_dec(x_4);
return x_9;
}
else
{
uint8_t x_10; 
x_10 = lean_nat_dec_eq(x_4, x_7);
lean_dec(x_7);
lean_dec(x_4);
if (x_10 == 0)
{
lean_dec(x_8);
lean_dec(x_5);
return x_10;
}
else
{
lean_object* x_11; uint8_t x_12; 
x_11 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___closed__0));
x_12 = l_instDecidableEqList___redArg(x_11, x_5, x_8);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching_decEq(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqBranching(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1_spec__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_2 = x_8;
x_3 = x_6;
goto _start;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_2 = x_14;
x_3 = x_11;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_dec(x_1);
return x_2;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_1);
lean_ctor_set_tag(x_3, 5);
lean_ctor_set(x_3, 1, x_1);
lean_ctor_set(x_3, 0, x_2);
x_7 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_5);
x_8 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_8, 0, x_3);
lean_ctor_set(x_8, 1, x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1_spec__2(x_1, x_8, x_6);
return x_9;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_3, 0);
x_11 = lean_ctor_get(x_3, 1);
lean_inc(x_11);
lean_inc(x_10);
lean_dec(x_3);
lean_inc(x_1);
x_12 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_12, 0, x_2);
lean_ctor_set(x_12, 1, x_1);
x_13 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_10);
x_14 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_14, 0, x_12);
lean_ctor_set(x_14, 1, x_13);
x_15 = lp_DLDSBooleanCircuit_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1_spec__2(x_1, x_14, x_11);
return x_15;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
lean_dec(x_2);
x_3 = lean_box(0);
return x_3;
}
else
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_1, 1);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_4);
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Semantic_instReprBranching_repr___redArg(x_7);
x_9 = lp_DLDSBooleanCircuit_List_foldl___at___00Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0_spec__1(x_2, x_8, x_4);
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0___redArg(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_2; 
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__1));
return x_2;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; 
x_3 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__4));
x_4 = lp_DLDSBooleanCircuit_Std_Format_joinSep___at___00List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0_spec__0(x_1, x_3);
x_5 = lean_obj_once(&lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6, &lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6_once, _init_lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__6);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__7));
x_7 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_4);
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__8));
x_9 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_5);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__5));
x_7 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__3));
x_8 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__7);
x_9 = lp_DLDSBooleanCircuit_Semantic_instReprDLDS_repr___redArg(x_2);
x_10 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
x_11 = 0;
x_12 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set_uint8(x_12, sizeof(void*)*1, x_11);
x_13 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_13, 0, x_7);
lean_ctor_set(x_13, 1, x_12);
x_14 = ((lean_object*)(lp_DLDSBooleanCircuit_List_repr_x27___at___00Semantic_instReprVertex_repr_spec__0___redArg___closed__3));
x_15 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_box(1);
x_17 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_17, 0, x_15);
lean_ctor_set(x_17, 1, x_16);
x_18 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__5));
x_19 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
x_20 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_20, 0, x_19);
lean_ctor_set(x_20, 1, x_6);
x_21 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__16);
x_22 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0___redArg(x_3);
x_23 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_22);
x_24 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set_uint8(x_24, sizeof(void*)*1, x_11);
x_25 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_25, 0, x_20);
lean_ctor_set(x_25, 1, x_24);
x_26 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_14);
x_27 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_16);
x_28 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__7));
x_29 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_29, 0, x_27);
lean_ctor_set(x_29, 1, x_28);
x_30 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_30, 0, x_29);
lean_ctor_set(x_30, 1, x_6);
x_31 = l_Nat_reprFast(x_4);
x_32 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_32, 0, x_31);
x_33 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_33, 0, x_21);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set_uint8(x_34, sizeof(void*)*1, x_11);
x_35 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_35, 0, x_30);
lean_ctor_set(x_35, 1, x_34);
x_36 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_36, 0, x_35);
lean_ctor_set(x_36, 1, x_14);
x_37 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_37, 0, x_36);
lean_ctor_set(x_37, 1, x_16);
x_38 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg___closed__9));
x_39 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_39, 0, x_37);
lean_ctor_set(x_39, 1, x_38);
x_40 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_40, 0, x_39);
lean_ctor_set(x_40, 1, x_6);
x_41 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__19);
x_42 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprDLDS_repr_spec__0___redArg(x_5);
x_43 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_43, 0, x_41);
lean_ctor_set(x_43, 1, x_42);
x_44 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_44, 0, x_43);
lean_ctor_set_uint8(x_44, sizeof(void*)*1, x_11);
x_45 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_45, 0, x_40);
lean_ctor_set(x_45, 1, x_44);
x_46 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24, &lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24_once, _init_lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__24);
x_47 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__25));
x_48 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_48, 0, x_47);
lean_ctor_set(x_48, 1, x_45);
x_49 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_instReprVertex_repr___redArg___closed__26));
x_50 = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(x_50, 0, x_48);
lean_ctor_set(x_50, 1, x_49);
x_51 = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(x_51, 0, x_46);
lean_ctor_set(x_51, 1, x_50);
x_52 = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(x_52, 0, x_51);
lean_ctor_set_uint8(x_52, sizeof(void*)*1, x_11);
return x_52;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_instReprBranchingDLDS_repr(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0___redArg(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_List_repr___at___00Semantic_instReprBranchingDLDS_repr_spec__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_incomingSources_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_6, 1);
lean_inc_ref(x_1);
lean_inc_ref(x_8);
x_9 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_8, x_1);
if (x_9 == 0)
{
lean_free_object(x_2);
lean_dec(x_6);
x_2 = x_7;
goto _start;
}
else
{
lean_ctor_set(x_2, 1, x_3);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_12 = lean_ctor_get(x_2, 0);
x_13 = lean_ctor_get(x_2, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_2);
x_14 = lean_ctor_get(x_12, 1);
lean_inc_ref(x_1);
lean_inc_ref(x_14);
x_15 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_14, x_1);
if (x_15 == 0)
{
lean_dec(x_12);
x_2 = x_13;
goto _start;
}
else
{
lean_object* x_17; 
x_17 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_17, 0, x_12);
lean_ctor_set(x_17, 1, x_3);
x_2 = x_13;
x_3 = x_17;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_incomingSources_spec__1(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 0);
lean_inc_ref(x_7);
lean_dec(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = lean_ctor_get(x_9, 0);
lean_inc_ref(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_incomingSources(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_box(0);
x_5 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_incomingSources_spec__0(x_2, x_3, x_4);
x_6 = lp_DLDSBooleanCircuit_List_mapTR_loop___at___00Semantic_incomingSources_spec__1(x_5, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_envLookup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_3);
x_4 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(x_1);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_2, 1);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_dec(x_5);
lean_inc_ref(x_3);
x_9 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_7, x_3);
if (x_9 == 0)
{
lean_dec(x_8);
x_2 = x_6;
goto _start;
}
else
{
lean_dec(x_6);
lean_dec_ref(x_3);
lean_dec_ref(x_1);
return x_8;
}
}
}
}
LEAN_EXPORT uint8_t lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_dec_ref(x_2);
x_4 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__0(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 2);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = l_List_find_x3f___redArg(x_1, x_5);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; 
lean_dec(x_4);
lean_dec_ref(x_3);
x_7 = lean_box(0);
return x_7;
}
else
{
uint8_t x_8; 
x_8 = !lean_is_exclusive(x_6);
if (x_8 == 0)
{
lean_object* x_9; uint8_t x_10; 
x_9 = lean_ctor_get(x_6, 0);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_9, 0);
x_12 = lean_ctor_get(x_9, 1);
lean_dec(x_12);
lean_ctor_set(x_9, 1, x_11);
lean_ctor_set(x_9, 0, x_4);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_3);
lean_ctor_set(x_13, 1, x_9);
lean_ctor_set(x_6, 0, x_13);
return x_6;
}
else
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_14 = lean_ctor_get(x_9, 0);
lean_inc(x_14);
lean_dec(x_9);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_4);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_3);
lean_ctor_set(x_16, 1, x_15);
lean_ctor_set(x_6, 0, x_16);
return x_6;
}
}
else
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_17 = lean_ctor_get(x_6, 0);
lean_inc(x_17);
lean_dec(x_6);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
if (lean_is_exclusive(x_17)) {
 lean_ctor_release(x_17, 0);
 lean_ctor_release(x_17, 1);
 x_19 = x_17;
} else {
 lean_dec_ref(x_17);
 x_19 = lean_box(0);
}
if (lean_is_scalar(x_19)) {
 x_20 = lean_alloc_ctor(0, 2, 0);
} else {
 x_20 = x_19;
}
lean_ctor_set(x_20, 0, x_4);
lean_ctor_set(x_20, 1, x_18);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_3);
lean_ctor_set(x_21, 1, x_20);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_findBranchTarget(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__0___boxed), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lean_alloc_closure((void*)(lp_DLDSBooleanCircuit_Semantic_findBranchTarget___lam__1), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = l_List_findSome_x3f___redArg(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingColour(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_List_get_x3fInternal___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; uint8_t x_6; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_unbox(x_5);
lean_dec(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
x_7 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_readingColour___closed__0));
return x_7;
}
else
{
lean_object* x_8; 
x_8 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_readingColour___closed__1));
return x_8;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_readingColour___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_readingColour(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_stepVertex_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
lean_inc(x_2);
lean_inc_ref(x_1);
x_7 = lp_DLDSBooleanCircuit_Semantic_envLookup(x_1, x_2, x_5);
x_8 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___redArg(x_3, x_7);
x_3 = x_8;
x_4 = x_6;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_stepVertex_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_4; 
lean_dec_ref(x_1);
x_4 = l_List_reverse___redArg(x_3);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_1);
lean_inc(x_6);
x_8 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_6, x_1);
if (x_8 == 0)
{
lean_ctor_set(x_2, 1, x_3);
{
lean_object* _tmp_1 = x_7;
lean_object* _tmp_2 = x_2;
x_2 = _tmp_1;
x_3 = _tmp_2;
}
goto _start;
}
else
{
lean_free_object(x_2);
lean_dec(x_6);
x_2 = x_7;
goto _start;
}
}
else
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lean_ctor_get(x_2, 0);
x_12 = lean_ctor_get(x_2, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_2);
lean_inc_ref(x_1);
lean_inc(x_11);
x_13 = lp_DLDSBooleanCircuit_Semantic_instDecidableEqVertex_decEq(x_11, x_1);
if (x_13 == 0)
{
lean_object* x_14; 
x_14 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_3);
x_2 = x_12;
x_3 = x_14;
goto _start;
}
else
{
lean_dec(x_11);
x_2 = x_12;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_stepVertex(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_ctor_get_uint8(x_4, sizeof(void*)*4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_6);
lean_inc_ref(x_4);
lean_inc_ref(x_6);
x_7 = lp_DLDSBooleanCircuit_Semantic_incomingSources(x_6, x_4);
x_8 = lp_DLDSBooleanCircuit_Semantic_findBranchTarget(x_1, x_4);
if (lean_obj_tag(x_8) == 0)
{
lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_6);
x_9 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(x_6);
x_10 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_stepVertex_spec__0(x_6, x_3, x_9, x_7);
return x_10;
}
else
{
uint8_t x_11; 
x_11 = !lean_is_exclusive(x_8);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; 
x_12 = lean_ctor_get(x_8, 0);
x_13 = lean_ctor_get(x_12, 1);
lean_inc(x_13);
x_14 = lean_ctor_get(x_12, 0);
lean_inc(x_14);
lean_dec(x_12);
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
x_16 = lean_ctor_get(x_13, 1);
lean_inc(x_16);
lean_dec(x_13);
x_17 = lean_box(0);
lean_inc(x_14);
x_18 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_stepVertex_spec__1(x_14, x_7, x_17);
lean_inc_ref(x_6);
x_19 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(x_6);
lean_inc(x_3);
lean_inc_ref(x_6);
x_20 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_stepVertex_spec__0(x_6, x_3, x_19, x_18);
x_21 = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
x_22 = lp_DLDSBooleanCircuit_Semantic_readingColour(x_2, x_15);
lean_ctor_set(x_8, 0, x_16);
x_23 = l_Option_instDecidableEq___redArg(x_21, x_22, x_8);
if (x_23 == 0)
{
lean_dec(x_14);
lean_dec_ref(x_6);
lean_dec(x_3);
return x_20;
}
else
{
lean_object* x_24; lean_object* x_25; 
x_24 = lp_DLDSBooleanCircuit_Semantic_envLookup(x_6, x_3, x_14);
x_25 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___redArg(x_20, x_24);
return x_25;
}
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; uint8_t x_38; 
x_26 = lean_ctor_get(x_8, 0);
lean_inc(x_26);
lean_dec(x_8);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_ctor_get(x_26, 0);
lean_inc(x_28);
lean_dec(x_26);
x_29 = lean_ctor_get(x_27, 0);
lean_inc(x_29);
x_30 = lean_ctor_get(x_27, 1);
lean_inc(x_30);
lean_dec(x_27);
x_31 = lean_box(0);
lean_inc(x_28);
x_32 = lp_DLDSBooleanCircuit_List_filterTR_loop___at___00Semantic_stepVertex_spec__1(x_28, x_7, x_31);
lean_inc_ref(x_6);
x_33 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(x_6);
lean_inc(x_3);
lean_inc_ref(x_6);
x_34 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_stepVertex_spec__0(x_6, x_3, x_33, x_32);
x_35 = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
x_36 = lp_DLDSBooleanCircuit_Semantic_readingColour(x_2, x_29);
x_37 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_37, 0, x_30);
x_38 = l_Option_instDecidableEq___redArg(x_35, x_36, x_37);
if (x_38 == 0)
{
lean_dec(x_28);
lean_dec_ref(x_6);
lean_dec(x_3);
return x_34;
}
else
{
lean_object* x_39; lean_object* x_40; 
x_39 = lp_DLDSBooleanCircuit_Semantic_envLookup(x_6, x_3, x_28);
x_40 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_or___redArg(x_34, x_39);
return x_40;
}
}
}
}
else
{
lean_object* x_41; lean_object* x_42; 
lean_dec(x_3);
x_41 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_41);
lean_dec_ref(x_1);
lean_inc_ref(x_41);
x_42 = lp_DLDSBooleanCircuit_Semantic_hypIndex(x_41, x_4);
if (lean_obj_tag(x_42) == 0)
{
lean_object* x_43; 
x_43 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_zero(x_41);
return x_43;
}
else
{
lean_object* x_44; lean_object* x_45; 
x_44 = lean_ctor_get(x_42, 0);
lean_inc(x_44);
lean_dec_ref(x_42);
x_45 = lp_DLDSBooleanCircuit_Semantic_HypDepVec_oneHot___redArg(x_41, x_44);
lean_dec(x_44);
return x_45;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_stepVertex___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_stepVertex(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_dldsSemantics_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_dec_ref(x_1);
return x_3;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_inc(x_3);
lean_inc_ref(x_1);
x_8 = lp_DLDSBooleanCircuit_Semantic_stepVertex(x_1, x_2, x_3, x_6);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_6);
lean_ctor_set(x_9, 1, x_8);
x_10 = lean_box(0);
lean_ctor_set(x_4, 1, x_10);
lean_ctor_set(x_4, 0, x_9);
x_11 = l_List_appendTR___redArg(x_3, x_4);
x_3 = x_11;
x_4 = x_7;
goto _start;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_4, 0);
x_14 = lean_ctor_get(x_4, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_4);
lean_inc(x_13);
lean_inc(x_3);
lean_inc_ref(x_1);
x_15 = lp_DLDSBooleanCircuit_Semantic_stepVertex(x_1, x_2, x_3, x_13);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_13);
lean_ctor_set(x_16, 1, x_15);
x_17 = lean_box(0);
x_18 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_18, 0, x_16);
lean_ctor_set(x_18, 1, x_17);
x_19 = l_List_appendTR___redArg(x_3, x_18);
x_3 = x_19;
x_4 = x_14;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_dldsSemantics_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_dldsSemantics_spec__0(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemantics(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 3);
lean_inc(x_3);
x_4 = lean_box(0);
x_5 = lp_DLDSBooleanCircuit_List_foldl___at___00Semantic_dldsSemantics_spec__0(x_1, x_2, x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemantics___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_DLDSBooleanCircuit_Semantic_dldsSemantics(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemanticsAt(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
x_5 = lp_DLDSBooleanCircuit_Semantic_dldsSemantics(x_1, x_2);
x_6 = lp_DLDSBooleanCircuit_Semantic_envLookup(x_4, x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_dldsSemanticsAt___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_DLDSBooleanCircuit_Semantic_dldsSemanticsAt(x_1, x_2, x_3);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
x_5 = lean_apply_1(x_3, x_2);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_6, 1);
lean_inc(x_9);
lean_dec(x_6);
x_10 = lean_apply_4(x_4, x_8, x_9, x_7, x_2);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_7; 
lean_dec(x_6);
x_7 = lean_apply_1(x_5, x_4);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_5);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_3, 1);
lean_inc(x_9);
lean_dec_ref(x_3);
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_8, 1);
lean_inc(x_11);
lean_dec(x_8);
x_12 = lean_apply_4(x_6, x_10, x_11, x_9, x_4);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_envLookup_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_2);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_3, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_2(x_2, x_6, lean_box(0));
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_4);
x_6 = lean_box(0);
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; 
lean_dec(x_5);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec_ref(x_3);
x_9 = lean_apply_2(x_4, x_8, lean_box(0));
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__1_splitter(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__3_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_6, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 0);
lean_inc(x_8);
lean_dec(x_6);
x_9 = lean_ctor_get(x_7, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_7, 1);
lean_inc(x_10);
lean_dec(x_7);
x_11 = lean_apply_3(x_3, x_8, x_9, x_10);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit___private_DLDSBooleanCircuit_0__Semantic_stepVertex_match__3_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec(x_4);
x_5 = lean_box(0);
x_6 = lean_apply_1(x_3, x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
lean_dec(x_3);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
x_9 = lean_ctor_get(x_7, 0);
lean_inc(x_9);
lean_dec(x_7);
x_10 = lean_ctor_get(x_8, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_8, 1);
lean_inc(x_11);
lean_dec(x_8);
x_12 = lean_apply_3(x_4, x_9, x_10, x_11);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lp_DLDSBooleanCircuit_Semantic_evaluateDLDS(x_1, x_2, x_3);
x_6 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__0));
if (x_5 == 0)
{
lean_object* x_15; 
x_15 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__3));
x_7 = x_15;
goto block_14;
}
else
{
lean_object* x_16; 
x_16 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__4));
x_7 = x_16;
goto block_14;
}
block_14:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_string_append(x_6, x_7);
lean_dec_ref(x_7);
x_9 = lp_mathlib_IO_println___at___00__private_Mathlib_Tactic_Linter_TextBased_0__Mathlib_Linter_TextBased_formatErrors_spec__0(x_8);
if (lean_obj_tag(x_9) == 0)
{
lean_dec_ref(x_9);
if (x_5 == 0)
{
lean_object* x_10; lean_object* x_11; 
x_10 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__1));
x_11 = lp_mathlib_IO_println___at___00__private_Mathlib_Tactic_Linter_TextBased_0__Mathlib_Linter_TextBased_formatErrors_spec__0(x_10);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; 
x_12 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___closed__2));
x_13 = lp_mathlib_IO_println___at___00__private_Mathlib_Tactic_Linter_TextBased_0__Mathlib_Linter_TextBased_formatErrors_spec__0(x_12);
return x_13;
}
}
else
{
return x_9;
}
}
}
}
LEAN_EXPORT lean_object* lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_DLDSBooleanCircuit_Semantic_Testing_checkDLDSProof(x_1, x_2, x_3);
lean_dec(x_2);
return x_5;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_A__imp__B));
x_2 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity(void) {
_start:
{
lean_object* x_1; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity___closed__0, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity___closed__0);
return x_1;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion___closed__0(void) {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_1 = lean_box(0);
x_2 = 0;
x_3 = lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity;
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_unsigned_to_nat(4u);
x_6 = lean_alloc_ctor(0, 4, 2);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, x_4);
lean_ctor_set(x_6, 2, x_3);
lean_ctor_set(x_6, 3, x_1);
lean_ctor_set_uint8(x_6, sizeof(void*)*4, x_2);
lean_ctor_set_uint8(x_6, sizeof(void*)*4 + 1, x_2);
return x_6;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion(void) {
_start:
{
lean_object* x_1; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion___closed__0, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion___closed__0);
return x_1;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B___closed__0));
x_2 = lean_unsigned_to_nat(0u);
x_3 = lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion;
x_4 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB));
x_5 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
lean_ctor_set(x_5, 2, x_2);
lean_ctor_set(x_5, 3, x_1);
return x_5;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion(void) {
_start:
{
lean_object* x_1; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion___closed__0, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion___closed__0);
return x_1;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__0(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__1(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__0, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__0_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__0);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__2(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__1, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__1_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__1);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__B));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__3(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__2, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__2_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__2);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__AimpB__hyp));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__4(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__3, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__3_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__3);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__A));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__5(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__6(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__5, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__5_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__5);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__B__to__AimpB));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__7(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__6, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__6_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__6);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__B));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__8(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__7, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__7_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__7);
x_2 = ((lean_object*)(lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__A__to__B));
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__9(void) {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_box(0);
x_2 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__8, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__8_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__8);
x_3 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__4, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__4_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__4);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_1);
return x_4;
}
}
static lean_object* _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds(void) {
_start:
{
lean_object* x_1; 
x_1 = lean_obj_once(&lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__9, &lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__9_once, _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds___closed__9);
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Vector_Mem(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Duplicate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Vector_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Vector_Zip(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_DLDSBooleanCircuit_DLDSBooleanCircuit(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Mem(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Duplicate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Vector_Zip(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1 = _init_lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1();
lean_mark_persistent(lp_DLDSBooleanCircuit_Semantic_list__zipIdx__get__fst___auto__1);
lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity = _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity();
lean_mark_persistent(lp_DLDSBooleanCircuit_Semantic_Test_Identity_identity);
lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion = _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion();
lean_mark_persistent(lp_DLDSBooleanCircuit_Semantic_Test_Identity_v__conclusion);
lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion = _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion();
lean_mark_persistent(lp_DLDSBooleanCircuit_Semantic_Test_Identity_e__AimpB__to__conclusion);
lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds = _init_lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds();
lean_mark_persistent(lp_DLDSBooleanCircuit_Semantic_Test_Identity_dlds);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
