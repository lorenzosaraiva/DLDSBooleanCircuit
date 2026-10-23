import Circuit.GraphSize

open FlowSpec
namespace Semantic.Circuit

theorem incoming_eq_filter (d : Graph) (v : Vertex) :
    get_rule.incoming v d = d.EDGES.filter (fun e => e.END = v) := by
  have loop : ∀ es : List Deduction, get_rule.incoming.loop v es = es.filter (fun e => e.END = v) := by
    intro es
    induction es with
    | nil => rfl
    | cons head tail ih => simp [get_rule.incoming.loop, List.filter_cons, ih]
  exact loop d.EDGES

theorem outgoing_eq_filter (d : Graph) (v : Vertex) :
    get_rule.outgoing v d = d.EDGES.filter (fun e => e.START = v) := by
  have loop : ∀ es : List Deduction, get_rule.outgoing.loop v es = es.filter (fun e => e.START = v) := by
    intro es
    induction es with
    | nil => rfl
    | cons head tail ih => simp [get_rule.outgoing.loop, List.filter_cons, ih]
  exact loop d.EDGES

theorem graph_incoming_le (d : Graph) (v : Vertex) :
    (get_rule.incoming v d).length + 1 ≤ graphSize d := by
  rw [incoming_eq_filter]
  have h := List.length_filter_le (l := d.EDGES) (fun e => e.END = v)
  have t := graph_edges_le d
  omega

theorem graph_outgoing_le (d : Graph) (v : Vertex) :
    (get_rule.outgoing v d).length + 1 ≤ graphSize d := by
  rw [outgoing_eq_filter]
  have h := List.length_filter_le (l := d.EDGES) (fun e => e.START = v)
  have t := graph_edges_le d
  omega

theorem graph_incoming_label_le (d : Graph) (v : Vertex) (e : Deduction)
    (member : e ∈ get_rule.incoming v d) : e.DEPENDENCY.length ≤ graphSize d := by
  rw [incoming_eq_filter] at member
  have h := graph_label_le d e (List.mem_filter.mp member).1
  omega

end Semantic.Circuit
