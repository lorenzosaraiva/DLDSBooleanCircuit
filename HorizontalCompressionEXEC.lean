import Init
set_option linter.unusedSimpArgs false

namespace List
  /- Set-Like Conversion -/
  prefix:max "#" => List.eraseDups
  /- Set-Like Union: l₁ ∪ l₂ = {a | a ∈ l₁ ∨ a ∈ l₂} -/
  notation:66 l₁:40 " ∪ " l₂:40 => List.eraseDups (List.append l₁ l₂)
  /- Set-Like Subtraction: l₁ − l₂ = {a | a ∈ l₁ ∧ a ∉ l₂} -/
  notation:66 l₁:40 " − " l₂:40 => List.eraseDups (List.removeAll l₁ l₂)
end List


/- Types -/
/- Dag-Like Derivability Structure -/
/- DLDS: Labels -/
inductive Formula where
| atom (SYMBOL : String) : Formula
| implication (ANTECEDENT CONSEQUENT : Formula) : Formula
export Formula (atom implication)
prefix:max "#" => Formula.atom
infixl:66 ">>" => Formula.implication
def Formula.repr (FORMULA : Formula) : String :=
    match FORMULA with
    | (atom SYMBOL) => "#" ++ SYMBOL
    | (implication ANTECEDENT CONSEQUENT) => (Formula.repr ANTECEDENT)
                                          ++ ">>"
                                          ++ (Formula.repr CONSEQUENT)
instance : Repr Formula where reprPrec formula _ := Formula.repr formula
/- DLDS: Vertices -/
structure Vertex where
node :: (NUMBER : Nat)
        (LEVEL : Nat)
        (FORMULA : Formula)
        (HYPOTHESIS : Bool)
        (COLLAPSED : Bool)
        (PAST : List Nat)   /- Temporary Collapse Metadata -/
deriving Repr
export Vertex (node)
/- DLDS: Deduction Edges -/
structure Deduction where
edge :: (START : Vertex)
        (END : Vertex)
        (COLOUR : Nat)
        (DEPENDENCY : List Formula)
deriving Repr
export Deduction (edge)
/- DLDS: Ancestral Paths -/
structure Ancestral where
path :: (START : Vertex)
        (END : Vertex)
        (COLOURS : List Nat)
deriving Repr
export Ancestral (path)
/- DLDS: Graph -/
structure Graph where
dlds :: (NODES : List Vertex)
        (EDGES : List Deduction)
        (PATHS : List Ancestral)
deriving Repr
export Graph (dlds)
/- DLDS: Neighborhoods -/
structure Neighborhood where
rule :: (CENTER : Vertex)
        (INCOMING : List Deduction)
        (OUTGOING : List Deduction)
        (DIRECT : List Ancestral)
        (INDIRECT : List Ancestral)
deriving Repr
export Neighborhood (rule)

/- Instances of Decidability: Labels -/
@[inline] def Formula.decEq (FORMULA₁ FORMULA₂ : @& Formula) : Decidable (FORMULA₁ = FORMULA₂) :=
match FORMULA₁, FORMULA₂ with
| (atom SBL₁), (atom SBL₂) => by rewrite [Formula.atom.injEq];
                                 exact String.decEq SBL₁ SBL₂;
| (atom _), (implication _ _) => by exact isFalse (Formula.noConfusion);
| (implication _ _), (atom _) => by exact isFalse (Formula.noConfusion);
| (implication ANT₁ CON₁), (implication ANT₂ CON₂) => by
  rewrite [Formula.implication.injEq];
  have DecANT : Decidable (ANT₁ = ANT₂) := Formula.decEq ANT₁ ANT₂;
  have DecCON : Decidable (CON₁ = CON₂) := Formula.decEq CON₁ CON₂;
  exact @instDecidableAnd (ANT₁ = ANT₂) (CON₁ = CON₂) DecANT DecCON;
@[inline] instance : DecidableEq Formula := Formula.decEq
/- Instances of Decidability: Vertices -/
@[inline] def Vertex.decEq (NODE₁ NODE₂ : @& Vertex) : Decidable (NODE₁ = NODE₂) :=
match NODE₁, NODE₂ with
| (node NBR₁ LVL₁ FML₁ HPT₁ COL₁ PST₁), (node NBR₂ LVL₂ FML₂ HPT₂ COL₂ PST₂) => by
  rewrite [Vertex.node.injEq];
  have DecNBR : Decidable (NBR₁ = NBR₂) := Nat.decEq NBR₁ NBR₂;
  have DecLVL : Decidable (LVL₁ = LVL₂) := Nat.decEq LVL₁ LVL₂;
  have DecFML : Decidable (FML₁ = FML₂) := Formula.decEq FML₁ FML₂;
  have DecHPT : Decidable (HPT₁ = HPT₂) := Bool.decEq HPT₁ HPT₂;
  have DecCPS : Decidable (COL₁ = COL₂) := Bool.decEq COL₁ COL₂;
  have DecPST : Decidable (PST₁ = PST₂) := List.hasDecEq PST₁ PST₂;
  have DecAND₁ := @instDecidableAnd (COL₁ = COL₂) (PST₁ = PST₂) DecCPS DecPST;
  have DecAND₂ := @instDecidableAnd (HPT₁ = HPT₂) ( COL₁ = COL₂
                                                  ∧ PST₁ = PST₂ ) DecHPT DecAND₁;
  have DecAND₃ := @instDecidableAnd (FML₁ = FML₂) ( HPT₁ = HPT₂
                                                  ∧ COL₁ = COL₂
                                                  ∧ PST₁ = PST₂ ) DecFML DecAND₂;
  have DecAND₄ := @instDecidableAnd (LVL₁ = LVL₂) ( FML₁ = FML₂
                                                  ∧ HPT₁ = HPT₂
                                                  ∧ COL₁ = COL₂
                                                  ∧ PST₁ = PST₂ ) DecLVL DecAND₃;
  exact @instDecidableAnd (NBR₁ = NBR₂) ( LVL₁ = LVL₂
                                        ∧ FML₁ = FML₂
                                        ∧ HPT₁ = HPT₂
                                        ∧ COL₁ = COL₂
                                        ∧ PST₁ = PST₂ ) DecNBR DecAND₄;
@[inline] instance : DecidableEq Vertex := Vertex.decEq
/- Instances of Decidability: Deduction Edges -/
@[inline] def Deduction.decEq (EDGE₁ EDGE₂ : @& Deduction) : Decidable (EDGE₁ = EDGE₂) :=
match EDGE₁, EDGE₂ with
| (edge STT₁ END₁ CLR₁ DEP₁), (edge STT₂ END₂ CLR₂ DEP₂) => by
  rewrite [Deduction.edge.injEq];
  have DecSTT : Decidable (STT₁ = STT₂) := Vertex.decEq STT₁ STT₂;
  have DecEND : Decidable (END₁ = END₂) := Vertex.decEq END₁ END₂;
  have DecCLR : Decidable (CLR₁ = CLR₂) := Nat.decEq CLR₁ CLR₂;
  have DecDEP : Decidable (DEP₁ = DEP₂) := List.hasDecEq DEP₁ DEP₂;
  have DecAND₁ := @instDecidableAnd (CLR₁ = CLR₂) (DEP₁ = DEP₂) DecCLR DecDEP;
  have DecAND₂ := @instDecidableAnd (END₁ = END₂) (CLR₁ = CLR₂ ∧ DEP₁ = DEP₂) DecEND DecAND₁;
  exact @instDecidableAnd (STT₁ = STT₂) (END₁ = END₂ ∧ CLR₁ = CLR₂ ∧ DEP₁ = DEP₂) DecSTT DecAND₂;
@[inline] instance : DecidableEq Deduction := Deduction.decEq
/- Instances of Decidability: Ancestral Paths -/
@[inline] def Ancestral.decEq (PATH₁ PATH₂ : @& Ancestral) : Decidable (PATH₁ = PATH₂) :=
match PATH₁, PATH₂ with
| (path STT₁ END₁ CLRS₁), (path STT₂ END₂ CLRS₂) => by
  rewrite [Ancestral.path.injEq];
  have DecSTT : Decidable (STT₁ = STT₂) := Vertex.decEq STT₁ STT₂;
  have DecEND : Decidable (END₁ = END₂) := Vertex.decEq END₁ END₂;
  have DecCLRS : Decidable (CLRS₁ = CLRS₂) := List.hasDecEq CLRS₁ CLRS₂;
  have DecAND := @instDecidableAnd (END₁ = END₂) (CLRS₁ = CLRS₂) DecEND DecCLRS;
  exact @instDecidableAnd (STT₁ = STT₂) (END₁ = END₂ ∧ CLRS₁ = CLRS₂) DecSTT DecAND;
@[inline] instance : DecidableEq Ancestral := Ancestral.decEq
/- Instances of Decidability: Graph -/
@[inline] def Graph.decEq (DLDS₁ DLDS₂ : @& Graph) : Decidable (DLDS₁ = DLDS₂) := by
match DLDS₁, DLDS₂ with
| (dlds NODES₁ EDGES₁ PATHS₁), (dlds NODES₂ EDGES₂ PATHS₂) =>
  rewrite [Graph.dlds.injEq];
  have DecNODES : Decidable (NODES₁ = NODES₂) := List.hasDecEq NODES₁ NODES₂;
  have DecEDGES : Decidable (EDGES₁ = EDGES₂) := List.hasDecEq EDGES₁ EDGES₂;
  have DecPATHS : Decidable (PATHS₁ = PATHS₂) := List.hasDecEq PATHS₁ PATHS₂;
  have DecAND := @instDecidableAnd (EDGES₁ = EDGES₂) (PATHS₁ = PATHS₂) DecEDGES DecPATHS;
  exact @instDecidableAnd (NODES₁ = NODES₂) (EDGES₁ = EDGES₂ ∧ PATHS₁ = PATHS₂) DecNODES DecAND;
@[inline] instance : DecidableEq Graph := Graph.decEq
/- Instances of Decidability: Neighborhoods -/
@[inline] def Neighborhood.decEq (RULE₁ RULE₂ : @& Neighborhood) : Decidable (RULE₁ = RULE₂) :=
match RULE₁, RULE₂ with
| (rule CTR₁ INC₁ OUT₁ DIR₁ IND₁), (rule CTR₂ INC₂ OUT₂ DIR₂ IND₂) => by
  rewrite [Neighborhood.rule.injEq];
  have DecCTR : Decidable (CTR₁ = CTR₂) := Vertex.decEq CTR₁ CTR₂;
  have DecINC : Decidable (INC₁ = INC₂) := List.hasDecEq INC₁ INC₂;
  have DecOUT : Decidable (OUT₁ = OUT₂) := List.hasDecEq OUT₁ OUT₂;
  have DecDIR : Decidable (DIR₁ = DIR₂) := List.hasDecEq DIR₁ DIR₂;
  have DecIND : Decidable (IND₁ = IND₂) := List.hasDecEq IND₁ IND₂;
  have DecAND₁ := @instDecidableAnd (DIR₁ = DIR₂) (IND₁ = IND₂) DecDIR DecIND;
  have DecAND₂ := @instDecidableAnd (OUT₁ = OUT₂) ( DIR₁ = DIR₂
                                                  ∧ IND₁ = IND₂ ) DecOUT DecAND₁;
  have DecAND₃ := @instDecidableAnd (INC₁ = INC₂) ( OUT₁ = OUT₂
                                                  ∧ DIR₁ = DIR₂
                                                  ∧ IND₁ = IND₂ ) DecINC DecAND₂;
  exact @instDecidableAnd (CTR₁ = CTR₂) ( INC₁ = INC₂
                                        ∧ OUT₁ = OUT₂
                                        ∧ DIR₁ = DIR₂
                                        ∧ IND₁ = IND₂ ) DecCTR DecAND₃;
@[inline] instance : DecidableEq Neighborhood := Neighborhood.decEq

/- Unfold Equality: Vertex -/
theorem Vertex.node.injEq' {NODE₁ NODE₂ : Vertex} :
  -----------------------------------------------------------------------------------
  ( (NODE₁ = NODE₂) ↔ ( NODE₁.NUMBER = NODE₂.NUMBER
                      ∧ NODE₁.LEVEL = NODE₂.LEVEL
                      ∧ NODE₁.FORMULA = NODE₂.FORMULA
                      ∧ NODE₁.HYPOTHESIS = NODE₂.HYPOTHESIS
                      ∧ NODE₁.COLLAPSED = NODE₂.COLLAPSED
                      ∧ NODE₁.PAST = NODE₂.PAST ) ) := by
match NODE₁, NODE₂ with
| (node NBR₁ LVL₁ FML₁ HPT₁ COL₁ PST₁),
  (node NBR₂ LVL₂ FML₂ HPT₂ COL₂ PST₂) => simp only [Vertex.node.injEq];
/- Unfold Equality: Deduction -/
theorem Deduction.edge.injEq' {EDGE₁ EDGE₂ : Deduction} :
  -----------------------------------------------------------------------------------
  ( (EDGE₁ = EDGE₂) ↔ ( EDGE₁.START = EDGE₂.START
                      ∧ EDGE₁.END = EDGE₂.END
                      ∧ EDGE₁.COLOUR = EDGE₂.COLOUR
                      ∧ EDGE₁.DEPENDENCY = EDGE₂.DEPENDENCY ) ) := by
match EDGE₁, EDGE₂ with
| (edge STT₁ END₁ CLR₁ DEP₁), (edge STT₂ END₂ CLR₂ DEP₂) => simp only [Deduction.edge.injEq];
/- Unfold Equality: Ancestral -/
theorem Ancestral.path.injEq' {PATH₁ PATH₂ : Ancestral} :
  -----------------------------------------------------------------------------------
  ( (PATH₁ = PATH₂) ↔ ( PATH₁.START = PATH₂.START
                      ∧ PATH₁.END = PATH₂.END
                      ∧ PATH₁.COLOURS = PATH₂.COLOURS ) ) := by
match PATH₁, PATH₂ with
| (path STT₁ END₁ CLRS₁), (path STT₂ END₂ CLRS₂) => simp only [Ancestral.path.injEq];


/- Methods & Definitions -/
/- Get: Incoming Deductions -/--------------------------------------------------------------------------------------------------
def get_rule.incoming (NODE : Vertex) (DLDS : Graph) : List Deduction :=
    loop NODE DLDS.EDGES
    where loop (NODE : Vertex) (EDGES : List Deduction) : List Deduction :=
          match EDGES with
          | [] => []
          | (EDGE::EDGES) => if   ( EDGE.END = NODE )
                             then ( EDGE :: loop NODE EDGES )
                             else ( loop NODE EDGES )
    ----------------------------------------------------------------------------------------------------------------------------
/- Get: Outgoing Deductions -/--------------------------------------------------------------------------------------------------
def get_rule.outgoing (NODE : Vertex) (DLDS : Graph) : List Deduction :=
    loop NODE DLDS.EDGES
    where loop (NODE : Vertex) (EDGES : List Deduction) : List Deduction :=
          match EDGES with
          | [] => []
          | (EDGE::EDGES) => if   ( EDGE.START = NODE )
                             then ( EDGE :: loop NODE EDGES )
                             else ( loop NODE EDGES )
    ----------------------------------------------------------------------------------------------------------------------------
/- Get: Direct Ancestrals -/----------------------------------------------------------------------------------------------------
def get_rule.direct (NODE : Vertex) (DLDS : Graph) : List Ancestral :=
    loop NODE DLDS.PATHS
    where loop (NODE : Vertex) (PATHS : List Ancestral) : List Ancestral :=
          match PATHS with
          | [] => []
          | (PATH::PATHS) => if   ( PATH.END = NODE )
                             then ( PATH :: loop NODE PATHS )
                             else ( loop NODE PATHS )
    ----------------------------------------------------------------------------------------------------------------------------
/- Get: Indirect Ancestrals -/--------------------------------------------------------------------------------------------------
def get_rule.indirect (NODE : Vertex) (DLDS : Graph) : List Ancestral :=
    loop (get_rule.incoming NODE DLDS) DLDS.PATHS
    where loop (EDGES : List Deduction) (PATHS : List Ancestral) : List Ancestral :=
          match EDGES with
          | [] => []
          | (EDGE::EDGES) => get_rule.direct.loop EDGE.START PATHS ++ loop EDGES PATHS
    ----------------------------------------------------------------------------------------------------------------------------
/- Collapse: NODE × DLDS → Neighborhood -/--------------------------------------------------------------------------------------
def get_rule (NODE : Vertex) (DLDS : Graph) : Neighborhood :=
    rule ( NODE )
         ( get_rule.incoming NODE DLDS )
         ( get_rule.outgoing NODE DLDS )
         ( get_rule.direct NODE DLDS )
         ( get_rule.indirect NODE DLDS )
    ----------------------------------------------------------------------------------------------------------------------------

/- Collapse & Type Conditionals -/
/- Check: Past Collapses (Vertex) & Path Colours (Ancestral) -/-------------------------------------------------------------
def check_numbers (NUMBERS : List Nat) : Prop :=
    ( NUMBERS ≠ [] )
  ∧ ( ∀{NUMBER : Nat}, ( NUMBER ∈ NUMBERS ) → ( NUMBER > 0 ) )
    ------------------------------------------------------------------------------------------------------------------------
/- Check: Nodes Set (Graph) -/----------------------------------------------------------------------------------------------
def check_dlds (DLDS : Graph) : Prop :=
    ( ∀{NODE₁ NODE₂ : Vertex}, ( NODE₁ ∈ DLDS.NODES ) →
                               ( NODE₂ ∈ DLDS.NODES ) →
      --------------------------------------
      ( ( NODE₁.NUMBER = NODE₂.NUMBER ) ↔ ( NODE₁ = NODE₂ ) ) )
  ∧ ( ∀{EDGE : Deduction}, ( EDGE ∈ DLDS.EDGES ) →
      --------------------------------------
      ( EDGE.START ∈ DLDS.NODES ∧ EDGE.END ∈ DLDS.NODES ) )
  ∧ ( ∀{PATH : Ancestral}, ( PATH ∈ DLDS.PATHS ) →
      --------------------------------------
      ( PATH.START ∈ DLDS.NODES ∧ PATH.END ∈ DLDS.NODES ) )
    ------------------------------------------------------------------------------------------------------------------------
/- Check: Collapse Nodes (Vertexes) & Incoming Edges (Deductions) -/--------------------------------------------------------
def check_collapse_nodes (RULEᵤ RULEᵥ : Neighborhood) : Prop :=
    ( RULEᵤ.CENTER.NUMBER > RULEᵥ.CENTER.NUMBER )
  ∧ ( RULEᵥ.CENTER.NUMBER ∉ RULEᵤ.CENTER.PAST )
  ∧ ( RULEᵤ.CENTER.LEVEL = RULEᵥ.CENTER.LEVEL )
  ∧ ( RULEᵤ.CENTER.FORMULA = RULEᵥ.CENTER.FORMULA )
  ∧ ( ∀{INCᵤ INCᵥ : Deduction}, ( INCᵤ ∈ RULEᵤ.INCOMING ) →
                                ( INCᵥ ∈ RULEᵥ.INCOMING ) →
                                ( INCᵤ.START ≠ INCᵥ.START ) )
    ------------------------------------------------------------------------------------------------------------------------
/- Check: Outgoing Edges (Deductions) & Collapse Nodes (Vertexes) & Incoming Edges (Deductions) -/-------------------------
def check_collapse_edges (RULEᵤ RULEᵥ : Neighborhood) : Prop :=
    ( ∃(OUTᵤ OUTᵥ : Deduction), ( OUTᵤ ∈ RULEᵤ.OUTGOING )
                              ∧ ( OUTᵥ ∈ RULEᵥ.OUTGOING )
                              ∧ ( OUTᵤ.COLOUR > 0 )
                              ∧ ( OUTᵥ.END = OUTᵤ.END )
                              ∧ ( OUTᵥ.COLOUR = OUTᵤ.COLOUR )
                              ∧ ( OUTᵥ.DEPENDENCY = OUTᵤ.DEPENDENCY ) )
  ∧ ( RULEᵤ.CENTER.LEVEL = RULEᵥ.CENTER.LEVEL )
  ∧ ( RULEᵤ.CENTER.FORMULA = RULEᵥ.CENTER.FORMULA )
  ∧ ( ∀{INCᵤ INCᵥ : Deduction}, ( INCᵤ ∈ RULEᵤ.INCOMING ) →
                                ( INCᵥ ∈ RULEᵥ.INCOMING ) →
                                ( INCᵤ.START ≠ INCᵥ.START ) )
    ------------------------------------------------------------------------------------------------------------------------

/- Neighborhood Type Hierarchy -/
/- Neighborhood: Type 0 (Non-Collapsed Node Without Incoming Ancestral Paths) ⊇-Elimination -/
def type0_elimination (RULE : Neighborhood) : Prop :=
    ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 ) ∧ ( RULE.CENTER.HYPOTHESIS = false )
  ∧ ( RULE.CENTER.COLLAPSED = false ) ∧ ( RULE.CENTER.PAST = [] )
  ∧ ( ∃(inc_nbr out_nbr : Nat),
      ∃(antecedent out_fml : Formula),
      ∃(major_hpt minor_hpt : Bool),
      ∃(major_dep minor_dep : List Formula),
      ------------------------------------------------------
      ( inc_nbr > 0 ) ∧ ( out_nbr > 0 )
    ∧ RULE.INCOMING = [ edge (node (inc_nbr+1)
                                   (RULE.CENTER.LEVEL+1)
                                   (antecedent>>RULE.CENTER.FORMULA)
                                   (major_hpt)
                                   (false)
                                   []) /- Left Child & Major Premise -/
                             RULE.CENTER
                             0
                             #major_dep,
                        edge (node inc_nbr (RULE.CENTER.LEVEL+1) antecedent minor_hpt false []) /- Right Child & Minor Premise -/
                             RULE.CENTER
                             0
                             #minor_dep ]
    ∧ RULE.OUTGOING = [ edge RULE.CENTER
                             (node out_nbr (RULE.CENTER.LEVEL-1) out_fml false false [])
                             0
                             (minor_dep ∪ major_dep) ]
    ∧ RULE.DIRECT   = []
    ∧ RULE.INDIRECT = [] )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Type 0 (Non-Collapsed Node Without Incoming Ancestral Paths) ⊇-Introduction -/
def type0_introduction (RULE : Neighborhood) : Prop :=
    ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 ) ∧ ( RULE.CENTER.HYPOTHESIS = false )
  ∧ ( RULE.CENTER.COLLAPSED = false ) ∧ ( RULE.CENTER.PAST = [] )
  ∧ ( ∃(inc_nbr out_nbr : Nat),
      ∃(antecedent consequent out_fml : Formula),
      ∃(inc_dep : List Formula),
    ------------------------------------------------------
      ( RULE.CENTER.FORMULA = antecedent>>consequent )
    ∧ ( inc_nbr > 0 ) ∧ ( out_nbr > 0 )
    ∧ RULE.INCOMING = [ edge (node inc_nbr (RULE.CENTER.LEVEL+1) consequent false false [])  /- Unique Child & Sole Premise -/
                             RULE.CENTER
                             0
                             #inc_dep ]
    ∧ RULE.OUTGOING = [ edge RULE.CENTER
                             (node out_nbr (RULE.CENTER.LEVEL-1) out_fml false false [])
                             0
                             (inc_dep − [antecedent]) ]
    ∧ RULE.DIRECT   = []
    ∧ RULE.INDIRECT = [] )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Type 0 (Non-Collapsed Node Without Incoming Ancestral Paths) Hypothesis (Top Formula) -/
def type0_hypothesis (RULE : Neighborhood) : Prop :=
    ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 ) ∧ ( RULE.CENTER.HYPOTHESIS = true )
  ∧ ( RULE.CENTER.COLLAPSED = false ) ∧ ( RULE.CENTER.PAST = [] )
  ∧ ( ∃(out_nbr : Nat),
      ∃(out_fml : Formula),
    ------------------------------------------------------
      ( out_nbr > 0 )
    ∧ RULE.INCOMING = []
    ∧ RULE.OUTGOING = [ edge RULE.CENTER
                             (node out_nbr (RULE.CENTER.LEVEL-1) out_fml false false [])
                             0
                             [RULE.CENTER.FORMULA] ]
    ∧ RULE.DIRECT   = []
    ∧ RULE.INDIRECT = [] )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Type 2 (Non-Collapsed Node With Incoming Ancestral Paths) ⊇-Elimination -/
def type2_elimination (RULE : Neighborhood) : Prop :=
    ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 ) ∧ ( RULE.CENTER.HYPOTHESIS = false )
  ∧ ( RULE.CENTER.COLLAPSED = false ) ∧ ( RULE.CENTER.PAST = [] )
  ∧ ( ∃(inc_nbr out_nbr anc_nbr anc_lvl : Nat),
      ∃(antecedent out_fml anc_fml : Formula),
      ∃(major_hpt minor_hpt out_hpt : Bool),
      ∃(major_dep minor_dep : List Formula),
      ∃(past colour : Nat)(pasts colours : List Nat),
      ------------------------------------------------------
      ( inc_nbr > 0 ) ∧ ( out_nbr > 0 )
    ∧ ( anc_nbr > 0 ) ∧ ( anc_lvl + List.length (0::colour::colours) = RULE.CENTER.LEVEL )
    ∧ ( colour ∈ (out_nbr::past::pasts) ) ∧ ( check_numbers (past::pasts) ) ∧ ( check_numbers (colour::colours) )
    ∧ RULE.INCOMING = [ edge (node (inc_nbr+1) (RULE.CENTER.LEVEL+1) (antecedent>>RULE.CENTER.FORMULA) major_hpt false []) /- Right Child & Major Premise -/
                             RULE.CENTER
                             0
                             #major_dep,
                        edge (node inc_nbr (RULE.CENTER.LEVEL+1) antecedent minor_hpt false [])                            /- Left Child & Minor Premise -/
                             RULE.CENTER
                             0
                             #minor_dep ]
    ∧ RULE.OUTGOING = [ edge RULE.CENTER
                             (node out_nbr (RULE.CENTER.LEVEL-1) out_fml out_hpt true (past::pasts))
                             0
                             (minor_dep ∪ major_dep) ]
    ∧ RULE.DIRECT   = [ path (node anc_nbr anc_lvl anc_fml false false [])
                             RULE.CENTER
                             (0::colour::colours) ]
    ∧ RULE.INDIRECT = [] )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Type 2 (Non-Collapsed Node With Incoming Ancestral Paths) ⊇-Introduction -/
def type2_introduction (RULE : Neighborhood) : Prop :=
    ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 ) ∧ ( RULE.CENTER.HYPOTHESIS = false )
  ∧ ( RULE.CENTER.COLLAPSED = false ) ∧ ( RULE.CENTER.PAST = [] )
  ∧ ( ∃(inc_nbr out_nbr anc_nbr anc_lvl : Nat),
      ∃(antecedent consequent out_fml anc_fml : Formula),
      ∃(out_hpt : Bool),
      ∃(inc_dep : List Formula),
      ∃(past colour : Nat)(pasts colours : List Nat),
    ------------------------------------------------------
      ( RULE.CENTER.FORMULA = antecedent>>consequent )
    ∧ ( inc_nbr > 0 ) ∧ ( out_nbr > 0 )
    ∧ ( anc_nbr > 0 ) ∧ ( anc_lvl + List.length (0::colour::colours) = RULE.CENTER.LEVEL )
    ∧ ( colour ∈ (out_nbr::past::pasts) ) ∧ ( check_numbers (past::pasts) ) ∧ ( check_numbers (colour::colours) )
    ∧ RULE.INCOMING = [ edge (node inc_nbr (RULE.CENTER.LEVEL+1) consequent false false [])  /- Unique Child & Sole Premise -/
                             RULE.CENTER
                             0
                             #inc_dep ]
    ∧ RULE.OUTGOING = [ edge RULE.CENTER
                             (node out_nbr (RULE.CENTER.LEVEL-1) out_fml out_hpt true (past::pasts))
                             0
                             (inc_dep − [antecedent]) ]
    ∧ RULE.DIRECT   = [ path (node anc_nbr anc_lvl anc_fml false false [])
                             RULE.CENTER
                             (0::colour::colours) ]
    ∧ RULE.INDIRECT = [] )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Type 2 (Non-Collapsed Node With Incoming Ancestral Paths) Hypothesis (Top Formula) -/
def type2_hypothesis (RULE : Neighborhood) : Prop :=
    ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 ) ∧ ( RULE.CENTER.HYPOTHESIS = true )
  ∧ ( RULE.CENTER.COLLAPSED = false ) ∧ ( RULE.CENTER.PAST = [] )
  ∧ ( ∃(out_nbr anc_nbr anc_lvl : Nat),
      ∃(out_fml anc_fml : Formula),
      ∃(out_hpt : Bool),
      ∃(past colour : Nat)(pasts colours : List Nat),
    ------------------------------------------------------
      ( out_nbr > 0 )
    ∧ ( anc_nbr > 0 ) ∧ ( anc_lvl + List.length (0::colour::colours) = RULE.CENTER.LEVEL )
    ∧ ( colour ∈ (out_nbr::past::pasts) ) ∧ ( check_numbers (past::pasts) ) ∧ ( check_numbers (colour::colours) )
    ∧ RULE.INCOMING = []
    ∧ RULE.OUTGOING = [ edge RULE.CENTER
                             (node out_nbr (RULE.CENTER.LEVEL-1) out_fml out_hpt true (past::pasts))
                             0
                             [RULE.CENTER.FORMULA] ]
    ∧ RULE.DIRECT   = [ path (node anc_nbr anc_lvl anc_fml false false [])
                             RULE.CENTER
                             (0::colour::colours) ]
    ∧ RULE.INDIRECT = [] )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Check Incoming Edges (Type 1 & 3) -/--------------------------------------------------------------------------------------------------------------------------
def type_incoming (RULE : Neighborhood) : Prop := ∀{INC : Deduction}, ( INC ∈ RULE.INCOMING ) → ( check INC RULE.CENTER RULE.INDIRECT )
  where check (INC : Deduction) (CENTER : Vertex) (INDIRECT : List Ancestral) : Prop :=
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
        ( ( INC.START.NUMBER > 0 ) ∧ ( INC.START.LEVEL = CENTER.LEVEL + 1 )
        ∧ ( INC.START.COLLAPSED = false ) ∧ ( INC.START.PAST = [] ) )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( INC.END = CENTER )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( INC.COLOUR = 0 )                                                                                                /- := Incoming Edge => -/
        /- Deduction-Ancestral Duo: -/-----------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ∃(colour : Nat)(colours : List Nat)(anc : Vertex), ( path anc INC.START (0::colour::colours) ∈ INDIRECT ) )     /- => Indirect Path => -/
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Check Outgoing Edges (Type 1) -/------------------------------------------------------------------------------------------------------------------------------
def type_outgoing₁ (RULE : Neighborhood) : Prop := ∀{OUT : Deduction}, ( OUT ∈ RULE.OUTGOING ) → ( type_outgoing₁.check_h₁ OUT RULE.CENTER
                                                                                                 ∨ type_outgoing₁.check_ie₁ OUT RULE.CENTER RULE.INDIRECT )
  where check_h₁ (OUT : Deduction) (CENTER : Vertex) : Prop :=
        /- Type 1 Hypothesis -/------------------------------------------------------------------------------------------------------------------------------------------------
        ( CENTER.HYPOTHESIS = true )
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.START = CENTER )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ( OUT.END.NUMBER > 0 ) ∧ ( OUT.END.LEVEL = CENTER.LEVEL - 1 )
        ∧ ( OUT.END.COLLAPSED = false ) ∧ ( OUT.END.PAST = [] ) )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.COLOUR = 0 )
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_ie₁ (OUT : Deduction) (CENTER : Vertex) (INDIRECT : List Ancestral) : Prop :=
        /- Type 1 Introduction & Elimination -/--------------------------------------------------------------------------------------------------------------------------------
        ( ( CENTER.HYPOTHESIS = false ) ∨ ( CENTER.COLLAPSED = true ) )
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.START = CENTER )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ( OUT.END.NUMBER > 0 ) ∧ ( OUT.END.LEVEL = CENTER.LEVEL - 1 )
        ∧ ( OUT.END.COLLAPSED = false ) ∧ ( OUT.END.PAST = [] ) )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.COLOUR ∈ (CENTER.NUMBER::CENTER.PAST) )                                                                             /- := Outgoing Edge => -/
        /- Deduction-Ancestral Duo: -/-----------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ∃(inc : Vertex), ( path OUT.END inc [0, OUT.COLOUR] ∈ INDIRECT ) )                                                      /- => Indirect Path => -/
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Check Outgoing Edges (Type 3) -/------------------------------------------------------------------------------------------------------------------------------
def type_outgoing₃ (RULE : Neighborhood) : Prop := ∀{OUT : Deduction}, ( OUT ∈ RULE.OUTGOING ) → ( ( type_outgoing₁.check_h₁ OUT RULE.CENTER
                                                                                                   ∨ type_outgoing₁.check_ie₁ OUT RULE.CENTER RULE.INDIRECT )
                                                                                                 ∨ ( type_outgoing₃.check_h₃ OUT RULE.CENTER RULE.DIRECT
                                                                                                   ∨ type_outgoing₃.check_ie₃ OUT RULE.CENTER RULE.INDIRECT ) )
  where check_h₃ (OUT : Deduction) (CENTER : Vertex) (DIRECT : List Ancestral) : Prop :=
        /- Type 3 Hypothesis -/------------------------------------------------------------------------------------------------------------------------------------------------
        ( CENTER.HYPOTHESIS = true )
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.START = CENTER )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ( OUT.END.NUMBER > 0 ) ∧ ( OUT.END.LEVEL = CENTER.LEVEL - 1 )
        ∧ ( OUT.END.COLLAPSED = true ) ∧ ( ∃(past : Nat)(pasts : List Nat), ( check_numbers (past::pasts) )
                                                                          ∧ ( OUT.END.PAST = (past::pasts) ) ) )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.COLOUR ∈ (CENTER.NUMBER::CENTER.PAST) )                                                                             /- := Outgoing Edge => -/
        /- Deduction-Ancestral Duo: -/-----------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ∃(colours : List Nat)(anc : Vertex), ( path anc CENTER (OUT.COLOUR::colours) ∈ DIRECT ) )                               /- => Direct Path . -/
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        check_ie₃ (OUT : Deduction) (CENTER : Vertex) (INDIRECT : List Ancestral) : Prop :=
        /- Type 3 Introduction & Elimination -/--------------------------------------------------------------------------------------------------------------------------------
        ( ( CENTER.HYPOTHESIS = false ) ∨ ( CENTER.COLLAPSED = true ) )
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.START = CENTER )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ( OUT.END.NUMBER > 0 ) ∧ ( OUT.END.LEVEL = CENTER.LEVEL - 1 )
        ∧ ( OUT.END.COLLAPSED = true ) ∧ ( ∃(past : Nat)(pasts : List Nat), ( check_numbers (past::pasts) )
                                                                          ∧ ( OUT.END.PAST = (past::pasts) ) ) )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( OUT.COLOUR ∈ (CENTER.NUMBER::CENTER.PAST) )                                                                             /- := Outgoing Edge => -/
        /- Deduction-Ancestral Duo: -/-----------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ∃(colours : List Nat)(inc anc : Vertex), ( path anc inc (0::OUT.COLOUR::colours) ∈ INDIRECT ) )                         /- => Indirect Path => -/
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Check Direct Paths (Type 1 & 3) -/----------------------------------------------------------------------------------------------------------------------------
def type_direct (RULE : Neighborhood) : Prop := ∀{DIR : Ancestral}, ( DIR ∈ RULE.DIRECT ) → ( check DIR RULE.CENTER RULE.OUTGOING )
  where check (DIR : Ancestral) (CENTER : Vertex) (OUTGOING : List Deduction) : Prop :=
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
        ( ( DIR.START.NUMBER > 0 ) ∧ ( DIR.START.LEVEL ≤ CENTER.LEVEL - 1 ) ∧ ( DIR.START.HYPOTHESIS = false )
        ∧ ( DIR.START.COLLAPSED = false ) ∧ ( DIR.START.PAST = [] ) )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( DIR.END = CENTER )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( DIR.START.LEVEL + List.length (DIR.COLOURS) = CENTER.LEVEL )
      ∧ ( ∃(colour₁ colour₂ : Nat),
          ∃(colours : List Nat), ( check_numbers (colour₁::colour₂::colours) )
                               ∧ ( colour₁ ∈ (CENTER.NUMBER::CENTER.PAST) )
                               ∧ ( DIR.COLOURS = (colour₁::colour₂::colours) )                                                /- := Direct Path => -/
                                 /- Deduction-Ancestral Duo: -/----------------------------------------------------------------------------------------------------------------
                               ∧ ( ∃(out : Vertex),                                                                           /- => Outgoing Edge . -/
                                   ∃(dep_out : List Formula), ( out.COLLAPSED = true )
                                                            ∧ ( colour₂ ∈ (out.NUMBER::out.PAST) )
                                                            ∧ ( edge CENTER out colour₁ dep_out ∈ OUTGOING )
                                                            ∧ ( ∀{all_out : Deduction}, ( all_out ∈ OUTGOING ) →
                                                                                        ( ( all_out.COLOUR = colour₁ ) ↔ ( all_out = edge CENTER out colour₁ dep_out ) ) ) ) )
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Check Indirect Paths (Type 1 & 3) -/--------------------------------------------------------------------------------------------------------------------------
def type_indirect (RULE : Neighborhood) : Prop := ∀{IND : Ancestral}, ( IND ∈ RULE.INDIRECT ) → ( check IND RULE.CENTER RULE.INCOMING RULE.OUTGOING )
  where check (IND : Ancestral) (CENTER : Vertex) (INCOMING OUTGOING : List Deduction) : Prop :=
        /- Start Node: -/------------------------------------------------------------------------------------------------------------------------------------------------------
        ( ( IND.START.NUMBER > 0 ) ∧ ( IND.START.LEVEL ≤ CENTER.LEVEL - 1 ) ∧ ( IND.START.HYPOTHESIS = false )
        ∧ ( IND.START.COLLAPSED = false ) ∧ ( IND.START.PAST = [] ) )
        /- End Node: -/--------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( ( IND.END.NUMBER > 0 ) ∧ ( IND.END.LEVEL = CENTER.LEVEL + 1 )
        ∧ ( IND.END.COLLAPSED = false ) ∧ ( IND.END.PAST = [] ) )
        /- Colours: -/---------------------------------------------------------------------------------------------------------------------------------------------------------
      ∧ ( IND.START.LEVEL + List.length (IND.COLOURS) = CENTER.LEVEL + 1 )
      ∧ ( ∃(colour : Nat),
          ∃(colours : List Nat), ( check_numbers (colour::colours) )
                               ∧ ( colour ∈ (CENTER.NUMBER::CENTER.PAST) )
                               ∧ ( IND.COLOURS = (0::colour::colours) )                                                         /- := Indirect Path => -/
                                 /- Deduction-Ancestral Trio: -/---------------------------------------------------------------------------------------------------------------
                               ∧ ( ∃(dep_inc : List Formula), ( edge IND.END CENTER 0 dep_inc ∈ INCOMING )                      /- => Incoming Edge => -/
                                                            ∧ ( ∀{all_inc : Deduction}, ( all_inc ∈ INCOMING ) →
                                                                                        ( ( all_inc.START = IND.END ) ↔ ( all_inc = edge IND.END CENTER 0 dep_inc ) ) ) )
                               ∧ ( ∃(out : Vertex),                                                                             /- => Outgoing Edge . -/
                                   ∃(dep_out : List Formula), ( ( colours = [] ) ↔ ( out = IND.START ) )
                                                            ∧ ( edge CENTER out colour dep_out ∈ OUTGOING )
                                                            ∧ ( ∀{all_out : Deduction}, ( all_out ∈ OUTGOING ) →
                                                                                        ( ( all_out.COLOUR = colour ) ↔ ( all_out = edge CENTER out colour dep_out ) ) ) ) )
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Pre-Type 1 (Collapsed Nodes With Short Neighboring Ancestral Paths) Collapsed Node -/
def type1_pre_collapse (RULE : Neighborhood) : Prop :=
    /- Check Center -/-----------------------------------------------------------------------------------------------------------------------
    ( ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 )
    ∧ ( RULE.CENTER.COLLAPSED = false )
    ∧ ( RULE.CENTER.PAST = [] )
    /- Check Deduction Edges -/--------------------------------------------------------------------------------------------------------------
    ∧ ( ( RULE.INCOMING = [] ) ↔ ( RULE.CENTER.HYPOTHESIS = true ) )
    ∧ ( List.length (RULE.INCOMING) ≤ 2 )
    ∧ ( ∃(out : Deduction), ( RULE.OUTGOING = [out] ) )
    ∧ ( ∀{OUT₁ OUT₂ : Deduction}, ( OUT₁ ∈ RULE.OUTGOING ) →
                                  ( OUT₂ ∈ RULE.OUTGOING ) →
                                  ( OUT₁.COLOUR > 0 ∨ OUT₂.COLOUR > 0 ) →
                                  ( ( OUT₁.COLOUR = OUT₂.COLOUR ) ↔ ( OUT₁ = OUT₂ ) ) )
    /- Check Ancestral Paths -/--------------------------------------------------------------------------------------------------------------
    ∧ ( RULE.DIRECT = [] )
    ∧ ( ∀{ind₁ ind₂ : Ancestral}, ( ind₁ ∈ RULE.INDIRECT ) →
                                  ( ind₂ ∈ RULE.INDIRECT ) → ( ( ind₁.COLOURS = ind₂.COLOURS ) ↔ ( ind₁.START = ind₂.START ) ) )
    ∧ ( List.length (RULE.INDIRECT) = List.length (RULE.INCOMING) )
    ∧ ( ∀{ind : Ancestral}, ( ind ∈ RULE.INDIRECT ) → ( ind.COLOURS = [0, RULE.CENTER.NUMBER] ) )
    /- Generic Properties -/-----------------------------------------------------------------------------------------------------------------
    ∧ ( type_incoming RULE ) ∧ ( type_outgoing₁ RULE )
    ∧ ( type_indirect RULE ) )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Type 1 (Collapsed Nodes With Short Neighboring Ancestral Paths) Collapsed Node -/
def type1_collapse (RULE : Neighborhood) : Prop :=
    /- Check Center -/-----------------------------------------------------------------------------------------------------------------------
    ( ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 )
    ∧ ( RULE.CENTER.COLLAPSED = true )
    ∧ ( ∃(past : Nat)(pasts : List Nat), ( check_numbers (past::pasts) )
                                       ∧ ( RULE.CENTER.PAST = (past::pasts) ) )
    /- Check Deduction Edges -/--------------------------------------------------------------------------------------------------------------
    ∧ ( ( RULE.INCOMING = [] ) → ( RULE.CENTER.HYPOTHESIS = true ) )
    ∧ ( ∃(out : Deduction)(outs : List Deduction), ( RULE.OUTGOING = (out::outs) ) )
    ∧ ( ∀{OUT₁ OUT₂ : Deduction}, ( OUT₁ ∈ RULE.OUTGOING ) →
                                  ( OUT₂ ∈ RULE.OUTGOING ) →
                                  ( OUT₁.COLOUR > 0 ∨ OUT₂.COLOUR > 0 ) →
                                  ( ( OUT₁.COLOUR = OUT₂.COLOUR ) ↔ ( OUT₁ = OUT₂ ) ) )
    /- Check Ancestral Paths -/--------------------------------------------------------------------------------------------------------------
    ∧ ( RULE.DIRECT = [] )
    ∧ ( List.length (RULE.INDIRECT) = List.length (RULE.INCOMING) )
    ∧ ( ∀{ind : Ancestral}, ( ind ∈ RULE.INDIRECT ) → ( ∃(colour : Nat), ( ind.COLOURS = [0, colour] ) ) )
    /- Generic Properties -/-----------------------------------------------------------------------------------------------------------------
    ∧ ( type_incoming RULE ) ∧ ( type_outgoing₁ RULE )
    ∧ ( type_indirect RULE ) )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Neighborhood: Pre-Type 3 (Collapsed Nodes With Long Neighboring Ancestral Paths) Collapsed Node -/
def type3_pre_collapse (RULE : Neighborhood) : Prop :=
    /- Check Center -/-----------------------------------------------------------------------------------------------------------------------
    ( ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 )
    ∧ ( RULE.CENTER.COLLAPSED = false )
    ∧ ( RULE.CENTER.PAST = [] )
    /- Check Deduction Edges -/--------------------------------------------------------------------------------------------------------------
    ∧ ( ( RULE.INCOMING = [] ) ↔ ( RULE.CENTER.HYPOTHESIS = true ) )
    ∧ ( List.length (RULE.INCOMING) ≤ 2 )
    ∧ ( ∃(out : Deduction), ( RULE.OUTGOING = [out] ) )
    ∧ ( ∀{OUT₁ OUT₂ : Deduction}, ( OUT₁ ∈ RULE.OUTGOING ) →
                                  ( OUT₂ ∈ RULE.OUTGOING ) →
                                  ( OUT₁.COLOUR > 0 ∨ OUT₂.COLOUR > 0 ) →
                                  ( ( OUT₁.COLOUR = OUT₂.COLOUR ) ↔ ( OUT₁ = OUT₂ ) ) )
    /- Check Ancestral Paths -/--------------------------------------------------------------------------------------------------------------
    ∧ ( ( RULE.CENTER.HYPOTHESIS = false ) → ( RULE.DIRECT = [] ) )
    ∧ ( ( RULE.DIRECT ≠ [] ) → ( RULE.CENTER.HYPOTHESIS = true ) )
    ∧ ( ( RULE.DIRECT = [] ) ∨ ( ∃(dir : Ancestral), ( RULE.DIRECT = [dir] ) ) )
    ∧ ( ∀{ind₁ ind₂ : Ancestral}, ( ind₁ ∈ RULE.INDIRECT ) →
                                  ( ind₂ ∈ RULE.INDIRECT ) → ( ( ind₁.COLOURS = ind₂.COLOURS ) ↔ ( ind₁.START = ind₂.START ) ) )
    ∧ ( List.length (RULE.INDIRECT) = List.length (RULE.INCOMING) )
    /- Generic Properties -/-----------------------------------------------------------------------------------------------------------------
    ∧ ( type_incoming RULE ) ∧ ( type_outgoing₃ RULE )
    ∧ ( type_direct RULE ) ∧ ( type_indirect RULE ) )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Neighborhood: Type 3 (Collapsed Nodes With Long Neighboring Ancestral Paths) Collapsed Node -/
def type3_collapse (RULE : Neighborhood) : Prop :=
    /- Check Center -/-----------------------------------------------------------------------------------------------------------------------
    ( ( RULE.CENTER.NUMBER > 0 ) ∧ ( RULE.CENTER.LEVEL > 0 )
    ∧ ( RULE.CENTER.COLLAPSED = true )
    ∧ ( ∃(past : Nat)(pasts : List Nat), ( check_numbers (past::pasts) )
                                       ∧ ( RULE.CENTER.PAST = (past::pasts) ) )
    /- Check Deduction Edges -/--------------------------------------------------------------------------------------------------------------
    ∧ ( ( RULE.INCOMING = [] ) → ( RULE.CENTER.HYPOTHESIS = true ) )
    ∧ ( ∃(out : Deduction)(outs : List Deduction), ( RULE.OUTGOING = (out::outs) ) )
    ∧ ( ∀{OUT₁ OUT₂ : Deduction}, ( OUT₁ ∈ RULE.OUTGOING ) →
                                  ( OUT₂ ∈ RULE.OUTGOING ) →
                                  ( OUT₁.COLOUR > 0 ∨ OUT₂.COLOUR > 0 ) →
                                  ( ( OUT₁.COLOUR = OUT₂.COLOUR ) ↔ ( OUT₁ = OUT₂ ) ) )
    /- Check Ancestral Paths -/--------------------------------------------------------------------------------------------------------------
    ∧ ( ( RULE.CENTER.HYPOTHESIS = false ) → ( RULE.DIRECT = [] ) )
    ∧ ( ( RULE.DIRECT ≠ [] ) → ( RULE.CENTER.HYPOTHESIS = true ) )
    ∧ ( List.length (RULE.INDIRECT) = List.length (RULE.INCOMING) )
    /- Generic Properties -/-----------------------------------------------------------------------------------------------------------------
    ∧ ( type_incoming RULE ) ∧ ( type_outgoing₃ RULE )
    ∧ ( type_direct RULE ) ∧ ( type_indirect RULE ) )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Pre-Collapse Methods -/
/- Paint: Deduction Edge -/------------------------------------------------------------------------------------------------------------------
def pre_collapse.outgoing (COLOUR : Nat) (HYPOTHESIS : Bool) (OUTGOING : List Deduction) (DIRECT : List Ancestral) : List Deduction :=
    match HYPOTHESIS, OUTGOING, DIRECT with
    | _, [], _ => panic! "Zero Outgoing Edges!!!"
    | _, (_::_::_), _ => panic! "Multiple Outgoing Edges!!!"
    | _, _, (_::_::_) => panic! "Multiple Direct Paths!!!"
    -- Hypothesis ∧ Single Outgoing Edge ∧ Zero Direct Paths => Return Outgoing Edge (Unpainted)
    | true, [_], [] => OUTGOING
    -- Hypothesis ∧ Single Outgoing Edge ∧ Single Direct Path => Return Outgoing Edge (Painted)
    | true, [OUT], [_] => [ edge OUT.START OUT.END COLOUR OUT.DEPENDENCY ]
    -- Non-Hypothesis ∧ Single Outgoing Edge ∧ Zero Direct Paths => Return Outgoing Edge (Painted)
    | false, [OUT], [] => [ edge OUT.START OUT.END COLOUR OUT.DEPENDENCY ]
    -- Non-Hypothesis ∧ Single Outgoing Edge ∧ Single Direct Path => Return Outgoing Edge (Painted)
    | false, [OUT], [_] => [ edge OUT.START OUT.END COLOUR OUT.DEPENDENCY ]
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Rewrite: Ancestral Paths -/---------------------------------------------------------------------------------------------------------------
def pre_collapse.direct (COLOUR : Nat) (HYPOTHESIS : Bool) (DIRECT : List Ancestral) : List Ancestral :=
    match HYPOTHESIS, DIRECT with
    | _, (_::_::_) => panic! "Multiple Direct Paths!!!"
    -- Hypothesis ∧ Zero Direct Paths => Return Nothing
    | true, [] => []
    -- Hypothesis ∧ Single Direct Path => Paint Direct Path
    | true, [PATH] => paint COLOUR PATH
    -- Non-Hypothesis ∧ Zero Direct Paths => Return Nothing
    | false, [] => []
    -- Non-Hypothesis ∧ Single Direct Path => Return Nothing
    | false, [_] => []
  where paint (COLOUR : Nat) (PATH : Ancestral) : List Ancestral :=
        match PATH.COLOURS with
        | [] => panic! "Blank Path!!!"
        | ((_+1)::_) => panic! "Broken Path!!!"
        -- Correctly Colored Path => Return Indirect Path(s)
        | (0::COLOURS) => [ path PATH.START PATH.END (COLOUR::COLOURS) ]
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Create: Ancestral Paths -/----------------------------------------------------------------------------------------------------------------
def pre_collapse.indirect (COLOUR : Nat) (HYPOTHESIS : Bool) (INCOMING OUTGOING : List Deduction) (DIRECT : List Ancestral) : List Ancestral :=
    match HYPOTHESIS, INCOMING, OUTGOING, DIRECT with
    | true, (_::_), _, _ => panic! "Hypothesis With Incoming Edge(s)!!!"
    | false, [], _, _ => panic! "Non-Hypothesis Without Incoming Edge(s)"
    | _, _, [], _ => panic! "Zero Outgoing Edges!!!"
    | _, _, (_::_::_), _ => panic! "Multiple Outgoing Edges!!!"
    | _, _, _, (_::_::_) => panic! "Multiple Direct Paths!!!"
    -- Hypothesis ∧ Single Outgoing Edge ∧ Zero Direct Paths => Return Nothing
    | true, _, [_], [] => []
    -- Hypothesis ∧ Single Outgoing Edge ∧ Single Direct Path => Return Nothing
    | true, _, [_], [_] => []
    -- Non-Hypothesis ∧ Single Outgoing Edge ∧ Zero Direct Paths => Create Indirect Path(s)
    | false, (_::_), [OUT], [] => create COLOUR INCOMING OUT
    -- Non-Hypothesis ∧ Single Outgoing Edge ∧ Single Direct Path => Move-Up Direct Path(s)
    | false, (_::_), [_], [PATH] => move_up COLOUR INCOMING PATH
  where create (COLOUR : Nat) (INCOMING : List Deduction) (OUT : Deduction) : List Ancestral :=
        match INCOMING with
        | [] => []
        | (IN::INS) => ( path OUT.END IN.START [0, COLOUR] )
                    :: ( create COLOUR INS OUT )
        move_up (COLOUR : Nat) (INCOMING : List Deduction) (PATH : Ancestral) : List Ancestral :=
        match INCOMING, PATH with
        | _, (path _ _ []) => panic! "Blank Path!!!"
        -- Colored Path => Return Indirect Path(s)
        | [], (path _ _ (_::_)) => []
        | (IN::INS), (path _ _ (ZERO::COLOURS)) => ( path PATH.START IN.START (ZERO::COLOUR::COLOURS) )
                                                :: ( move_up COLOUR INS PATH )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Pre-Collapse Definitions -/
/- Pre-Collapse: Neighborhood → Neighborhood -/----------------------------------------------------------------------------------------------
def pre_collapse (RULE : Neighborhood) : Neighborhood :=
    match RULE.CENTER.COLLAPSED with
    | true => RULE
    | false => rule ( RULE.CENTER )
                    ( RULE.INCOMING )
                    ( pre_collapse.outgoing RULE.CENTER.NUMBER RULE.CENTER.HYPOTHESIS RULE.OUTGOING RULE.DIRECT )
                    ( pre_collapse.direct RULE.CENTER.NUMBER RULE.CENTER.HYPOTHESIS RULE.DIRECT )
                    ( pre_collapse.indirect RULE.CENTER.NUMBER RULE.CENTER.HYPOTHESIS RULE.INCOMING RULE.OUTGOING RULE.DIRECT )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Collapse Methods -/
/- Collapse: NODE × NODE → NODE -/-----------------------------------------------------------------------------------------------------------
def collapse.center (LEFT RIGHT : Vertex) : Vertex :=
    node ( LEFT.NUMBER )
         ( LEFT.LEVEL )
         ( LEFT.FORMULA )
         ( LEFT.HYPOTHESIS || RIGHT.HYPOTHESIS )
         ( true )
         ( RIGHT.NUMBER :: LEFT.PAST )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Rewrite: Deduction Edge End -/------------------------------------------------------------------------------------------------------------
def collapse.rewrite_incoming (COLLAPSE : Vertex) (EDGES : List Deduction) : List Deduction :=
    match EDGES with
    | [] => []
    | (EDGE::EDGES) => ( edge EDGE.START COLLAPSE EDGE.COLOUR EDGE.DEPENDENCY ) :: ( rewrite_incoming COLLAPSE EDGES )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Rewrite: Deduction Edge Start -/----------------------------------------------------------------------------------------------------------
def collapse.rewrite_outgoing (COLLAPSE : Vertex) (EDGES : List Deduction) : List Deduction :=
    match EDGES with
    | [] => []
    | (EDGE::EDGES) => ( edge COLLAPSE EDGE.END EDGE.COLOUR EDGE.DEPENDENCY ) :: ( rewrite_outgoing COLLAPSE EDGES )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Rewrite: Ancestral Edge End -/------------------------------------------------------------------------------------------------------------
def collapse.rewrite_direct (COLLAPSE : Vertex) (PATHS : List Ancestral) : List Ancestral :=
    match PATHS with
    | [] => []
    | (PATH::PATHS) => ( path PATH.START COLLAPSE PATH.COLOURS ) :: ( rewrite_direct COLLAPSE PATHS )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Collapse Definitions (Collapses a Single Pair of Nodes) -/
/- Collapse: RULE × RULE → Neighborhood -/---------------------------------------------------------------------------------------------------
def collapse (RULEᵤ RULEᵥ : Neighborhood) : Neighborhood :=
    rule ( collapse.center RULEᵤ.CENTER RULEᵥ.CENTER )
         ( collapse.rewrite_incoming (collapse.center RULEᵤ.CENTER RULEᵥ.CENTER) RULEᵥ.INCOMING
        ++ collapse.rewrite_incoming (collapse.center RULEᵤ.CENTER RULEᵥ.CENTER) RULEᵤ.INCOMING )
         ( collapse.rewrite_outgoing (collapse.center RULEᵤ.CENTER RULEᵥ.CENTER) RULEᵥ.OUTGOING
        ++ collapse.rewrite_outgoing (collapse.center RULEᵤ.CENTER RULEᵥ.CENTER) RULEᵤ.OUTGOING )
         ( collapse.rewrite_direct (collapse.center RULEᵤ.CENTER RULEᵥ.CENTER) RULEᵥ.DIRECT
        ++ collapse.rewrite_direct (collapse.center RULEᵤ.CENTER RULEᵥ.CENTER) RULEᵤ.DIRECT )
         ( RULEᵥ.INDIRECT
        ++ RULEᵤ.INDIRECT )
    -----------------------------------------------------------------------------------------------------------------------------------------
/- Collapse: NODE × NODE × DLDS → Neighborhood -/--------------------------------------------------------------------------------------------
def collapse_rule (U V : Vertex) (DLDS : Graph) : Neighborhood := collapse ( pre_collapse (get_rule U DLDS) )
                                                                           ( pre_collapse (get_rule V DLDS) )
    -----------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

/- Compress Methods -/
/- Convert: NODES × DLDS → Neighborhood -/----------------------------------------------------------
def get_rules (NODES : List Vertex) (DLDS : Graph) : List Neighborhood :=
    match NODES with
    | [] => []
    | (NODE::NODES) => ( get_rule NODE DLDS :: get_rules NODES DLDS )
    ------------------------------------------------------------------------------------------------
/- Compress: RULES → List Neighborhood -/----------------------------------------------------------------------------------------------------
def pre_compress (RULES : List Neighborhood) : List Neighborhood :=
    match RULES with
    | [] => []
    | (RULE::RULES) => ( pre_collapse RULE ) :: ( pre_compress RULES )
/- Compress: RULES → Neighborhood -/---------------------------------------------------------------------------------------------------------
def compress (RULES : List Neighborhood) : Neighborhood :=
    match RULES with
    | [] => rule (node 0 0 #"" false false []) [] [] [] []
    | [RULE] => RULE
    | (RULE₁::RULE₂::RULES) => compress ((collapse ( RULE₁ ) ( RULE₂ ))::RULES)
    termination_by RULES.length
    decreasing_by
      simp only [List.length];
      simp +arith;
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Compression Definitions (Collapses Multiple Pairs of Nodes) -/
/- Compress: NODES × DLDS → Neighborhood -/--------------------------------------------------------------------------------------------------
def compress_rule (NODES : List Vertex) (DLDS : Graph) : Neighborhood := compress ( pre_compress (get_rules NODES DLDS) )
    -----------------------------------------------------------------------------------------------------------------------------------------

/- Convert: RULE × DLDS → Graph -/------------------------------------------------------------------
def pop_rules (DLDS : Graph) (RULES : List Neighborhood) : Graph :=
    dlds ( List.removeAll DLDS.NODES ( center RULES ) )
         ( List.removeAll DLDS.EDGES ( incoming RULES ++ outgoing RULES ) )
         ( List.removeAll DLDS.PATHS ( direct RULES ++ indirect RULES ) )
    where center (RULES : List Neighborhood) : List Vertex :=
          match RULES with
          | [] => []
          | (RULE::RULES) => ( RULE.CENTER :: center RULES )
          incoming (RULES : List Neighborhood) : List Deduction :=
          match RULES with
          | [] => []
          | (RULE::RULES) => ( RULE.INCOMING ++ incoming RULES )
          outgoing (RULES : List Neighborhood) : List Deduction :=
          match RULES with
          | [] => []
          | (RULE::RULES) => ( RULE.OUTGOING ++ outgoing RULES )
          direct (RULES : List Neighborhood) : List Ancestral :=
          match RULES with
          | [] => []
          | (RULE::RULES) => ( RULE.DIRECT ++ direct RULES )
          indirect (RULES : List Neighborhood) : List Ancestral :=
          match RULES with
          | [] => []
          | (RULE::RULES) => ( RULE.INDIRECT ++ indirect RULES )
    ------------------------------------------------------------------------------------------------
/- Convert: RULE × DLDS → Graph -/------------------------------------------------------------------
def set_rule (DLDS : Graph) (RULE : Neighborhood) : Graph :=
    dlds ( RULE.CENTER :: DLDS.NODES )
         ( RULE.INCOMING ++ RULE.OUTGOING ++ DLDS.EDGES )
         ( RULE.DIRECT ++ RULE.INDIRECT ++ DLDS.PATHS )
    ------------------------------------------------------------------------------------------------

/- Collapse: NODE × NODE × DLDS → Graph -/----------------------------------------------------------
def collapse_nodes (U V : Vertex) (DLDS : Graph) : Graph :=
    set_rule ( pop_rules DLDS ( get_rules [U,V] DLDS ) )
             ( collapse_rule U V DLDS )
    ------------------------------------------------------------------------------------------------
/- Compress: NODE × DLDS → Graph -/-----------------------------------------------------------------
def compress_nodes (NODES : List Vertex) (DLDS : Graph) : Graph :=
    set_rule ( pop_rules DLDS ( get_rules NODES DLDS ) )
             ( compress_rule NODES DLDS )
    ------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

/- DLDS-Graph Methods -/
/- Get: DLDS → Nat -/-------------------------------------------------------------------------------
def get_levels (DLDS : Graph) : List Nat := loop [] DLDS.NODES
    where loop (LEVELS : List Nat) (NODES : List Vertex) : List Nat :=
          match NODES with
          | [] => LEVELS
          | (NODE::NODES) => if   ( NODE.LEVEL ∈ LEVELS )
                             then ( loop LEVELS NODES )
                             else ( loop (NODE.LEVEL :: LEVELS) NODES )
/- Get: DLDS → Formula -/---------------------------------------------------------------------------
def get_labels (DLDS : Graph) : List Formula := loop [] DLDS.NODES
    where loop (LABELS : List Formula) (NODES : List Vertex) : List Formula :=
          match NODES with
          | [] => LABELS
          | (NODE::NODES) => if   ( NODE.FORMULA ∈ LABELS )
                             then ( loop LABELS NODES )
                             else ( loop (NODE.FORMULA :: LABELS) NODES )
    ------------------------------------------------------------------------------------------------

/- Convert: DLDS → Vertex -/------------------------------------------------------------------------
def repeated_nodes_vertex (DLDS : Graph) : List (List (List Vertex)) := loop₀ ( get_levels DLDS )
                                                                              ( get_labels DLDS )
                                                                              ( DLDS.NODES )
        -- Separates the DLDS by level:
  where loop₀ (LEVELS : List Nat) (LABELS : List Formula) (NODES : List Vertex) : List (List (List Vertex)) :=
        match LEVELS with
        | [] => []
        | (LEVEL::LEVELS) => if   ( loop₁ LEVEL LABELS NODES ≠ [] )
                             then ( loop₁ LEVEL LABELS NODES ) :: ( loop₀ LEVELS LABELS NODES )
                             else ( loop₀ LEVELS LABELS NODES )
        -- Separates levels by label:
        loop₁ (LEVEL : Nat) (LABELS : List Formula) (NODES : List Vertex) : List (List Vertex) :=
        match LABELS with
        | [] => []
        | (LABEL::LABELS) => if   ( loop₂ LEVEL LABEL [] NODES ≠ [] )
                             then ( loop₂ LEVEL LABEL [] NODES ) :: ( loop₁ LEVEL LABELS NODES )
                             else ( loop₁ LEVEL LABELS NODES )
        -- Gives the collapsable nodes in LEVEL of DLDS:
        loop₂ (LEVEL : Nat) (LABEL : Formula) (NODES₁ NODES₂ : List Vertex) : List Vertex :=
        match NODES₁, NODES₂ with
        | _, [] => []
        | NODES₁, (NODE::NODES₂) => if   ( NODE.LEVEL = LEVEL )
                                      && ( NODE.FORMULA = LABEL )
                                      && ( loop₃ NODE (NODES₁ ++ NODES₂) )
                                    then ( NODE ) :: ( loop₂ LEVEL LABEL (List.concat NODES₁ NODE) NODES₂ )
                                    else ( loop₂ LEVEL LABEL (List.concat NODES₁ NODE) NODES₂ )
        -- Checks if NODE is collapsable:
        loop₃ (NODE : Vertex) (NODESₓ : List Vertex) : Bool :=
        match NODESₓ with
        | [] => false
        | (NODEₓ::NODESₓ) => ( NODE.LEVEL = NODEₓ.LEVEL
                            && NODE.FORMULA = NODEₓ.FORMULA )
                          || ( loop₃ NODE NODESₓ )
    ------------------------------------------------------------------------------------------------

/- DLDS-Graph Definitions -/
/- HC Algorithm: Convert: DLDS → DLDS -/------------------------------------------------------------
def compress_nodes_graph (DLDS : Graph) : Graph := loop₀ ( DLDS )
                                                         ( repeated_nodes_vertex DLDS )
  where loop₀ (DLDS : Graph) (LEVELS : List (List (List Vertex))) : Graph :=
        match LEVELS with
        | [] => DLDS
        | (LEVEL::LEVELS) => loop₀ ( loop₁ DLDS LEVEL ) LEVELS
        loop₁ (DLDS : Graph) (FORMULAS : List (List Vertex)) : Graph :=
        match FORMULAS with
        | [] => DLDS
        | (FORMULA::FORMULAS) => loop₁ ( compress_nodes FORMULA DLDS ) FORMULAS
    ------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

/- Misc. Definitions -/
/- Collapse: NUMBER × DLDS → Vertex -/--------------------------------------------------------------
def get_first_vertex (NUMBER : Nat) (DLDS : Graph) : Vertex :=
    loop NUMBER DLDS.NODES
    where loop (NUMBER : Nat) (NODES : List Vertex) : Vertex :=
          match NODES with
          | [] => node 0 0 #"" false false []
          | (NODE::NODES) => if   ( NODE.NUMBER = NUMBER )
                             then ( NODE )
                             else ( loop NUMBER NODES )
/- Collapse: Nat × Nat × Graph → Graph -/-----------------------------------------------------------
def collapse_nodes_nat (U V : Nat) (DLDS : Graph) : Graph := collapse_nodes ( get_first_vertex U DLDS )
                                                                            ( get_first_vertex V DLDS )
                                                                            ( DLDS )
        --------------------------------------------------------------------------------------------

/- Printable Definitions -/
/- Print: DLDS → String -/--------------------------------------------------------------------------
def repeated_nodes_string (DLDS : Graph) : List (List (List String)) := loop₀ ( repeated_nodes_vertex DLDS )
  where loop₀ (NODES : List (List (List Vertex))) : List (List (List String)) :=
        match NODES with
        | [] => []
        | (NODE::NODES) => ( loop₁ NODE )
                        :: ( loop₀ NODES )
        loop₁ (NODES : List (List Vertex)) : List (List String) :=
        match NODES with
        | [] => []
        | (NODE::NODES) => ( loop₂ NODE )
                        :: ( loop₁ NODES )
        loop₂ (NODES : List Vertex) : List String :=
        match NODES with
        | [] => []
        | (NODE::NODES) => ( loop₃ NODE )
                        :: ( loop₂ NODES )
        loop₃ (NODE : Vertex) : String := NODE.LEVEL.repr ++ "-"
                                       ++ NODE.NUMBER.repr ++ "-"
                                       ++ NODE.FORMULA.repr
/- Print: DLDS → String -/--------------------------------------------------------------------------
def check_sub_graph (DLDS₁ DLDS₂ : Graph) : List Bool := [ nodes DLDS₁.NODES DLDS₂.NODES ,
                                                           edges DLDS₁.EDGES DLDS₂.EDGES ,
                                                           paths DLDS₁.PATHS DLDS₂.PATHS ]
  where nodes (NODES₁ NODES₂ : List Vertex) : Bool :=
        match NODES₁ with
        | [] => True
        | (NODE₁::NODES₁) => ( NODE₁ ∈ NODES₂ )
                           ∧ ( nodes NODES₁ NODES₂ )
        edges (EDGES₁ EDGES₂ : List Deduction) : Bool :=
        match EDGES₁ with
        | [] => True
        | (EDGE₁::EDGES₁) => ( EDGE₁ ∈ EDGES₂ )
                           ∧ ( edges EDGES₁ EDGES₂ )
        paths (PATHS₁ PATHS₂ : List Ancestral) : Bool :=
        match PATHS₁ with
        | [] => True
        | (PATH₁::PATHS₁) => ( PATH₁ ∈ PATHS₂ )
                           ∧ ( paths PATHS₁ PATHS₂ )
        --------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------

/- Example -/
--Lvl.5:
def n0 := ( node 0 5 #"A1" true false [] )
def n1 := ( node 1 5 (#"A1" >> #"A2") true false [] )
def n2 := ( node 2 5 #"A1" true false [] )
def n3 := ( node 3 5 (#"A1" >> (#"A2" >> #"A3")) true false [] )
def n4 := ( node 4 5 #"A1" true false [] )
def n5 := ( node 5 5 (#"A1" >> #"A2") true false [] )
def n6 := ( node 6 5 #"A1" true false [] )
def n7 := ( node 7 5 (#"A1" >> #"A2") true false [] )
def n8 := ( node 8 5 #"A1" true false [] )
def n9 := ( node 9 5 (#"A1" >> (#"A2" >> #"A3")) true false [] )
--Lvl.4:
def n10 := ( node 10 4 #"A2" false false [] )
def n11 := ( node 11 4 (#"A2" >> #"A3") false false [] )
def n12 := ( node 12 4 #"A2" false false [] )
def n13 := ( node 13 4 (#"A2" >> (#"A3" >> #"A4")) true false [] )
def n14 := ( node 14 4 #"A2" false false [] )
def n15 := ( node 15 4 (#"A2" >> #"A3") false false [] )
--Lvl.3:
def n16 := ( node 16 3 #"A3" false false [] )
def n17 := ( node 17 3 (#"A3" >> #"A4") false false [] )
def n18 := ( node 18 3 #"A3" false false [] )
def n19 := ( node 19 3 (#"A3" >> (#"A4" >> #"A5")) true false [] )
--Lvl.2:
def n20 := ( node 20 2 #"A4" false false [] )
def n21 := ( node 21 2 (#"A4" >> #"A5") false false [] )
--Lvl.1:
def n22 := ( node 22 1 #"A5" false false [] )
--Lvl.0:
def n23 := ( node 23 0 (#"A4" >> #"A5") false false [] )
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
def e0 := ( edge (n0) (n10) 0 [#"A1"] )
def e1 := ( edge (n1) (n10) 0 [(#"A1" >> #"A2")] )
def e2 := ( edge (n2) (n11) 0 [#"A1"] )
def e3 := ( edge (n3) (n11) 0 [(#"A1" >> (#"A2" >> #"A3"))] )
def e4 := ( edge (n4) (n12) 0 [#"A1"] )
def e5 := ( edge (n5) (n12) 0 [(#"A1" >> #"A2")] )
def e6 := ( edge (n6) (n14) 0 [#"A1"] )
def e7 := ( edge (n7) (n14) 0 [(#"A1" >> #"A2")] )
def e8 := ( edge (n8) (n15) 0 [#"A1"] )
def e9 := ( edge (n9) (n15) 0 [(#"A1" >> (#"A2" >> #"A3"))] )
-------------------------------------------------------------------------------------------------------------------------------
def e10 := ( edge (n10) (n16) 0 [#"A1", (#"A1" >> #"A2")] )
def e11 := ( edge (n11) (n16) 0 [#"A1", (#"A1" >> (#"A2" >> #"A3"))] )
def e12 := ( edge (n12) (n17) 0 [#"A1", (#"A1" >> #"A2")] )
def e13 := ( edge (n13) (n17) 0 [(#"A2" >> (#"A3" >> #"A4"))] )
def e14 := ( edge (n14) (n18) 0 [#"A1", (#"A1" >> #"A2")] )
def e15 := ( edge (n15) (n18) 0 [#"A1", (#"A1" >> (#"A2" >> #"A3"))] )
-------------------------------------------------------------------------------------------------------------------------------
def e16 := ( edge (n16) (n20) 0 [#"A1", (#"A1" >> #"A2"), (#"A1" >> (#"A2" >> #"A3"))] )
def e17 := ( edge (n17) (n20) 0 [#"A1", (#"A1" >> #"A2"), (#"A2" >> (#"A3" >> #"A4"))] )
def e18 := ( edge (n18) (n21) 0 [#"A1", (#"A1" >> #"A2"), (#"A1" >> (#"A2" >> #"A3"))] )
def e19 := ( edge (n19) (n21) 0 [(#"A3" >> (#"A4" >> #"A5"))] )
-------------------------------------------------------------------------------------------------------------------------------
def e20 := ( edge (n20) (n22) 0 [#"A1", (#"A1" >> #"A2"), (#"A1" >> (#"A2" >> #"A3")), (#"A2" >> (#"A3" >> #"A4"))] )
def e21 := ( edge (n21) (n22) 0 [#"A1", (#"A1" >> #"A2"), (#"A1" >> (#"A2" >> #"A3")), (#"A3" >> (#"A4" >> #"A5"))] )
-------------------------------------------------------------------------------------------------------------------------------
def e22 := ( edge (n22) (n23) 0 [#"A1", (#"A1" >> #"A2"), (#"A1" >> (#"A2" >> #"A3")), (#"A2" >> (#"A3" >> #"A4")), (#"A3" >> (#"A4" >> #"A5"))] )
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
def d0 := ( dlds [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23]
                 [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22]
                 [] )

-- Lvl. 3: Nodes 16 & 18
def d1 := collapse_nodes_nat 16 18 d0

-- Lvl. 4: Nodes 11 & 15
def d2 := collapse_nodes_nat 11 15 d1

-- Lvl. 4: Nodes 10 & 12
def d3 := collapse_nodes_nat 10 12 d2

-- Lvl. 4: Nodes 10 & 14
def d4 := collapse_nodes_nat 10 14 d3

-- Lvl. 5: Nodes 3 & 9
def d5 := collapse_nodes_nat 3 9 d4

-- Lvl. 5: Nodes 1 & 5
def d6 := collapse_nodes_nat 1 5 d5

-- Lvl. 5: Nodes 1 & 7
def d7 := collapse_nodes_nat 1 7 d6

-- Lvl. 5: Nodes 0 & 2
def d8 := collapse_nodes_nat 0 2 d7

-- Lvl. 5: Nodes 0 & 4
def d9 := collapse_nodes_nat 0 4 d8

-- Lvl. 5: Nodes 0 & 6
def d10 := collapse_nodes_nat 0 6 d9

-- Lvl. 5: Nodes 0 & 8
def d11 := collapse_nodes_nat 0 8 d10

-- Result of HC:
def d11' := compress_nodes_graph d0
