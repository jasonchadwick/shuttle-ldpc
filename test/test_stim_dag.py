import networkx as nx
import stim
import src.stim_dag as stim_dag

# def test_dag():
#     # Test that circ_to_dag preserves circuit. Note that
#     # dag_to_circ(circ_to_dag(C)) will not necessarily give an identical circuit
#     # back if the original circuit has multiple valid orderings.
    
#     # TODO
#     raise NotImplementedError

# def test_dag_eq():
#     # TODO
#     raise NotImplementedError

def test_dag_expand():
    # Test that dag_expand maintains correct circuit
    circ = stim.Circuit.generated('surface_code:rotated_memory_x', distance=3, rounds=2, after_clifford_depolarization=0.01, before_round_data_depolarization=0.02, before_measure_flip_probability=0.03)
    circdag = stim_dag.circ_to_dag(circ)
    circdag_expanded = stim_dag.expand_dag(circdag)
    circdag_expanded1 = stim_dag.expand_dag(stim_dag.circ_to_dag(stim_dag.dag_to_circ(circdag)))
    assert nx.utils.graphs_equal(circdag_expanded.dag, circdag_expanded1.dag)
    assert circdag_expanded.instrs == circdag_expanded1.instrs