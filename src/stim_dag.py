import stim
import networkx as nx
import heapq
from typing import Any

def instr_target_qubits(instr, qubit_meas_history: list[int]) -> list[int]:
    tgts = set()
    for tgt in instr.targets_copy():
        if tgt.is_qubit_target:
            tgts.add(tgt.qubit_value)
        else:
            # TODO: use qubit_meas_history
            assert tgt.value < 0
            tgts.add(qubit_meas_history[tgt.value])
    return list(sorted(tgts))

class InstrNode:
    name: str
    targets_stim: list[stim.GateTarget]
    qubits: list[int]
    args: Any | None

    def __init__(
            self,
            name: str | None = None,
            targets_stim: list[stim.GateTarget] = [],
            qubits: list[int] = [],
            args: Any = None,
            instr: stim.CircuitInstruction | None = None,
            qubit_meas_history: list[int] | None = None,
        ):
        if (not name or not targets_stim or not qubits) and (not instr or qubit_meas_history is None):
            raise ValueError('Either provide `name`, `targets_stim`, and `qubits`, or provide `instr` and `qubit_meas_history')
        if (not name or not targets_stim or not qubits):
            assert instr is not None
            self.name = instr.name
            self.targets_stim = instr.targets_copy()
            self.qubits = instr_target_qubits(instr, qubit_meas_history)
            self.args = instr.gate_args_copy()
        else:
            assert name is not None
            self.name = name
            self.targets_stim = targets_stim
            self.qubits = qubits
            self.args = args

    def to_stim_instr(self):
        return stim.CircuitInstruction(self.name, self.targets_stim, ())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name and tuple(sorted(self.targets_stim, key=lambda t: t.value)) == tuple(sorted(other.targets_stim, key=lambda t: t.value)) and self.args == other.args
        else:
            return False
        
    def __hash__(self):
        return hash((self.name,) + tuple(t.value for t in sorted(self.targets_stim, key=lambda t: t.value)) + tuple(self.args))

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.name != other.name:
                raise ValueError(f'Cannot combine instructions with different names. Names {self.name}, {other.name}')
            if self.args != other.args:
                raise ValueError(f'Cannot combine instructions with different args. Args {self.args}, {other.args}')
            if set(self.qubits).intersection(set(other.qubits)) or set(self.targets_stim).intersection(set(other.targets_stim)):
                raise ValueError(f'Instruction targets intersect! {self.qubits, self.targets_stim, other.qubits, other.targets_stim}')
            return InstrNode(
                name=self.name,
                targets_stim=self.targets_stim + other.targets_stim,
                qubits=self.qubits + other.qubits,
                args=self.args,
            )
        else:
            raise ValueError('Tried to add two different objects!')
    
    def __str__(self):
        if self.args:
            return f'{self.name}({", ".join(str(a) for a in self.args)}) {" ".join(str(q) for q in self.qubits)}'
        return f'{self.name} {" ".join(str(q) for q in self.qubits)}'
    
    def __repr__(self):
        return self.__str__()

class CircDAG:
    dag: nx.DiGraph
    instrs: list[InstrNode]

    def __init__(self, dag, instrs):
        self.dag = dag
        self.instrs = instrs

    def __eq__(self, other):
        if isinstance(other, CircDAG):
            return nx.is_isomorphic(self.dag, other.dag, node_match=lambda n1,n2: self.instrs[n1] == other.instrs[n2])
        else:
            return False

def is_meas_related(instr: stim.CircuitInstruction):
    return instr.name in ['DETECTOR', 'OBSERVABLE_INCLUDE', 'M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ']

def circ_to_dag(circ) -> CircDAG:
    dag = nx.DiGraph()
    instrs = []
    meas_history: list[int] = []
    last_instr_by_qubit: dict[int, int] = dict()
    last_meas_or_det_instr: int | None = None
    for instr in circ:
        if not instr.targets_copy():
            continue
        instrnode = InstrNode(instr=instr, qubit_meas_history=meas_history)

        instrs.append(instrnode)
        idx = len(instrs)-1
        dag.add_node(idx)
        for qubit in instrnode.qubits:
            if qubit in last_instr_by_qubit:
                dag.add_edge(last_instr_by_qubit[qubit], idx)
            last_instr_by_qubit[qubit] = idx
        if is_meas_related(instr):
            if last_meas_or_det_instr is not None:
                dag.add_edge(last_meas_or_det_instr, idx)
            last_meas_or_det_instr = idx

        if instr.name in ['M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ']:
            meas_history += [t.qubit_value for t in instr.targets_copy()]
        
    return CircDAG(dag, instrs)

def dag_to_circ(dag: CircDAG):
    circ = stim.Circuit()
    for i in nx.topological_sort(dag.dag):
        instrnode = dag.instrs[i]
        circ.append(instrnode.name, instrnode.targets_stim, instrnode.args)
    return circ

def expand_dag(circdag: CircDAG):
    dag_expanded = nx.DiGraph()
    instrs_expanded = []
    frontier_heap = []
    top_gens = [list(gen) for gen in nx.topological_generations(circdag.dag)]
    gen_by_idx = {i: g for g,ii in enumerate(top_gens) for i in ii}
    for n in top_gens[0]:
        heapq.heappush(frontier_heap, (0, n))
    frontier_set = set(next(nx.topological_generations(circdag.dag)))
    processed_set = set()
    last_instr_by_qubit: dict[int, int] = dict()
    meas_history = []
    while frontier_heap:
        _,idx_old = heapq.heappop(frontier_heap)
        frontier_set.remove(idx_old)
        node = circdag.instrs[idx_old]
        if node.name in ['QUBIT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE']:
            instrs_expanded.append(node)
            idx = len(instrs_expanded)-1
            dag_expanded.add_node(idx)
            for qubit in node.qubits:
                if qubit in last_instr_by_qubit:
                    dag_expanded.add_edge(last_instr_by_qubit[qubit], idx)
                last_instr_by_qubit[qubit] = idx
        else:
            if node.name in ['CX', 'CZ', 'CY', 'DEPOLARIZE2']:
                qubits_per_instr = 2
            elif node.name in ['X', 'Y', 'Z', 'H', 'S', 'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'M', 'MX', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ', 'R', 'RX', 'RZ']:
                qubits_per_instr = 1
            else:
                raise ValueError('Unknown gate name', node.name)
            assert len(node.targets_stim) % qubits_per_instr == 0
            assert len(node.targets_stim) == len(node.qubits)
            covered_qs = []
            for i in range(len(node.targets_stim) // qubits_per_instr):
                new_node = InstrNode(
                    name=node.name,
                    targets_stim=node.targets_stim[i*qubits_per_instr:(i+1)*qubits_per_instr],
                    qubits=node.qubits[i*qubits_per_instr:(i+1)*qubits_per_instr],
                    args=node.args,
                )
                covered_qs += node.qubits[i*qubits_per_instr:(i+1)*qubits_per_instr]
                instrs_expanded.append(new_node)
                idx = len(instrs_expanded)-1
                dag_expanded.add_node(idx)
                for qubit in new_node.qubits:
                    if qubit in last_instr_by_qubit:
                        dag_expanded.add_edge(last_instr_by_qubit[qubit], idx)
                    last_instr_by_qubit[qubit] = idx

                if node.name in ['M', 'MZ', 'MX']:
                    meas_history += new_node.qubits
            assert covered_qs == node.qubits

        processed_set.add(idx_old)
        for succ in circdag.dag.successors(idx_old):
            if succ not in processed_set and succ not in frontier_set:
                heapq.heappush(frontier_heap, (gen_by_idx[succ], succ))
                frontier_set.add(succ)
    return CircDAG(dag_expanded, instrs_expanded)

def _simplify_dag_helper(circdag: CircDAG, combinable_instrs: list[str], search_depth=10) -> CircDAG:
    # Useful helper if we want to run multiple passes. E.g. first combine CX
    # gates, then single-qubit gates, then Ms and Rs, then errors.
    dag_simplified = circdag.dag.copy()
    instrs_simplified = circdag.instrs.copy()
    frontier_heap = []
    top_gens = [list(gen) for gen in nx.topological_generations(dag_simplified)]
    gen_by_idx = {i: g for g,ii in enumerate(top_gens) for i in ii}
    for n in top_gens[0]:
        heapq.heappush(frontier_heap, (0, n))
    frontier_set = set(top_gens[0])
    processed_set = set()
    unprocessed_set = set(dag_simplified.nodes)
    while frontier_heap:
        _,idx = heapq.heappop(frontier_heap)
        assert idx in frontier_set
        frontier_set.remove(idx)
        node = instrs_simplified[idx]

        # If we make a new node, add it back onto the frontier instead of
        # calling it done. Its generation is the maximum of the generations of
        # its constituents. Recompute all topological generations of its
        # descendants.
        if node.name in combinable_instrs:
            # Look through all possible siblings. We only look into future nodes
            # because any possible past sibling would have already triggered
            # this and would have picked this up.
            combined_indices = [idx]

            print(idx, set(nx.descendants(dag_simplified, idx)))
            possible_siblings = unprocessed_set - set(nx.descendants(dag_simplified, idx))
            print(idx, possible_siblings)
            possible_siblings.remove(idx)
            while possible_siblings:
                sib = min(possible_siblings, key=lambda s: gen_by_idx[s])
                possible_siblings.remove(sib)
                sib_node = instrs_simplified[sib]
                assert not set(sib_node.qubits).intersection(node.qubits), (idx, node, sib, sib_node)
                assert not set(sib_node.qubits).intersection(node.targets_stim)
                if sib_node.name == node.name and sib_node.args == node.args:
                    combined_indices.append(sib)

                    # TODO: this feels expensive...
                    possible_siblings -= set(nx.descendants(dag_simplified, sib))
            
            if len(combined_indices) > 1:
                new_node = InstrNode(
                    name=node.name,
                    targets_stim=[t for ci in combined_indices for t in instrs_simplified[ci].targets_stim],
                    qubits=[q for ci in combined_indices for q in instrs_simplified[ci].qubits],
                    args=node.args,
                )
                new_idx = len(instrs_simplified)
                instrs_simplified.append(new_node)
                dag_simplified.add_node(new_idx)
                for ci in combined_indices:
                    for pred in dag_simplified.predecessors(ci):
                        dag_simplified.add_edge(pred, new_idx)
                    for succ in dag_simplified.successors(ci):
                        dag_simplified.add_edge(new_idx, succ)
                        frontier_set.add(ci)

                    if ci in frontier_set:
                        frontier_set.remove(ci)
                    processed_set.add(ci)
                    if ci in unprocessed_set:
                        unprocessed_set.remove(ci)
                    dag_simplified.remove_node(ci)
                    instrs_simplified[ci] = None
            
                # TODO: may be a more efficient way to recalculate generations here
                top_gens = [list(gen) for gen in nx.topological_generations(dag_simplified)]
                gen_by_idx = {i: g for g,ii in enumerate(top_gens) for i in ii}
                frontier_heap = [(gen_by_idx[node], node) for node in frontier_set]
                heapq.heapify(frontier_heap)
            else:
                for succ in dag_simplified.successors(idx):
                    if succ not in processed_set:
                        assert succ in unprocessed_set
                        frontier_set.add(succ)
                        heapq.heappush(frontier_heap, (gen_by_idx[succ], succ))
                processed_set.add(idx)
                unprocessed_set.remove(idx)
        else:
            for succ in dag_simplified.successors(idx):
                if succ not in processed_set:
                    assert succ in unprocessed_set
                    frontier_set.add(succ)
                    heapq.heappush(frontier_heap, (gen_by_idx[succ], succ))
            processed_set.add(idx)
            unprocessed_set.remove(idx)
    
    # Relabel instructions to remove the empty ones that we merged in
    relabel_dict = {}
    offset = 0
    instrs_simplified_clean = []
    for i,instr in enumerate(instrs_simplified):
        if instr is None:
            assert i not in dag_simplified.nodes
            offset += 1
        else:
            relabel_dict[i] = i - offset
            assert i - offset == len(instrs_simplified_clean)
            instrs_simplified_clean.append(instr)
    dag_simplified_clean = nx.relabel_nodes(dag_simplified, relabel_dict)
    return CircDAG(dag_simplified_clean, instrs_simplified_clean)

def simplify_dag(circdag: CircDAG):
    # TODO: merge identical error instructions with same qubits but different
    # args

    error_names = ['X_ERROR', 'Y_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2']
    gate_names_2q = ['CX', 'CY', 'CZ']
    gate_names_1q = ['X', 'Y', 'Z', 'H', 'S', 'Sdg']
    mr_names = ['M', 'MR', 'MX', 'MY', 'MZ', 'MRX', 'MRY', 'MRZ', 'R', 'RX', 'RY', 'RZ']
    # TODO: merging M-type instructions is a bit more complicated, so might wait
    # on that one for now...
    circdag_new = _simplify_dag_helper(circdag, gate_names_2q)
    # circdag_new = _simplify_dag_helper(circdag_new, gate_names_1q)
    # # circdag_new = _simplify_dag_helper(circdag_new, gate_names_2q)
    # circdag_new = _simplify_dag_helper(circdag_new, error_names)

    return circdag_new

def simplify_circ(circ: stim.Circuit):
    return dag_to_circ(simplify_dag(circ_to_dag(circ)))

def dag_eq(dag):
    raise NotImplementedError