from dataclasses import dataclass
from numpy.typing import NDArray
from src.HGPCode import HGPCode
import numpy as np
import math
import networkx as nx
import sys
import itertools
import stim

# utility functions from open source python package
from ldpc.code_util import construct_generator_matrix, estimate_code_distance
from ldpc.mod2 import rank, reduced_row_echelon, row_echelon

class SHYPSCode(HGPCode):
    """
    Represents an SHYPS code, including specifying all gauge operators and parity checks
    refs: [1] arXiv:2502.07150 Computing Efficiently in QLDPC Codes 
    
    Assumes data qubits are laid out in one square grid, e.g.:
    ```
    00 03 06
    01 04 07
    02 05 08
    ```
    In this structure, each X guague operator is contained in a single row
    each Z gaugue operator is contained in a single column.
    """
    def __init__(self, r, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)

        self.check_order_matters = False

        self.H = self._simplex_code(r)
        self.G = construct_generator_matrix(self.H).toarray().astype(int)
        self.G_string = [''.join([str(x) for x in row]) for row in self.G.T]
        self.r = r

        # qubit layout
        self.n = 2**r-1
        self.data = np.full((self.n, self.n), -1, dtype=int)
        self.num_data = self.n**2
        for i in range(self.num_data):
            coords = self.data_idx_to_coords(i)
            self.data[*coords] = i
    
        # gauge checks (checks we need to physically measure)
        self.Gx = np.kron(self.H, np.eye(self.n)).astype(bool)
        self.Gz = np.kron(np.eye(self.n), self.H).astype(bool)
        self.X_checks = [
            list(np.where(self.Gx[i])[0]) for i in range(self.Gx.shape[0])
        ] # each entry is a list of data qubits to check
        self.Z_checks = [
            list(np.where(self.Gz[i])[0]) for i in range(self.Gz.shape[0])
        ] # each entry is a list of data qubits to check

        self.data_indices = list(range(self.num_data))
        self.X_ancilla_indices = list(range(self.num_data, self.num_data + len(self.X_checks)))
        self.Z_ancilla_indices = list(range(self.num_data + len(self.X_checks), self.num_data + len(self.X_checks) + len(self.Z_checks)))

        self.qubit_coords = [self.data_idx_to_coords(i) for i in range(self.num_data + len(self.X_checks) + len(self.Z_checks))]

        # Stabilizer checks (checks we can infer from the gauge checks)
        # Each entry is a list of gauge X checks s.t. when combined, gives the stabilizer check
        self.Sx =  np.kron(self.H, self.G).astype(bool)
        self.Sz =  np.kron(self.G, self.H).astype(bool)
        self.X_gauge_stabilizers = [[] for _ in range(self.n * self.r)]
        for i in range(self.Sx.shape[1]):
            if i%self.n<self.r:
                self.X_gauge_stabilizers[i%self.n + i//self.n*self.r] = (np.where(self.G[i%self.n])[0]+(i//self.n)*self.n).tolist()
        assert not any(len(stabs) == 0 for stabs in self.X_gauge_stabilizers)

        self.Z_gauge_stabilizers = [[] for _ in range(self.n * self.r)]
        for i in range(self.Sz.shape[1]):
            if i//self.n<self.r:
                self.Z_gauge_stabilizers[i] = (np.where(self.G[i//self.n])[0]*self.n + i%self.n).tolist()
        assert not any(len(stabs) == 0 for stabs in self.Z_gauge_stabilizers)

        # code parameters
        self.N = self.num_data = self.n**2
        self.K = self.r**2
        self.D = 2**(self.r-1)
        assert rank(self.H) + rank(self.G) == self.n, "Rank-nullity is not satisfied"

        # logical qubits
        emt_pvts = np.hstack([np.zeros((r,self.n-self.r)), np.eye(self.r)]).astype(int)
        self.Lx = np.kron(emt_pvts, self.G).T
        self.Lz = np.kron(self.G, emt_pvts).T

        assert (self.Sx@self.Lx%2).max() == 0, "Logical Z is not in ker(Sx)"
        assert (self.Sz@self.Lz%2).max() == 0, "Logical X is not in ker(Sz)"

        self.logical_qubits = np.array([*zip(self.Lx, self.Lz)])

        # assign checks into layers
        self.X_layers = []
        self.Z_layers = []
        available_X_checks_block1 = [(self.X_ancilla_indices[a],d) for a,data in enumerate(self.X_checks) for d in data]
        available_Z_checks_block1 = [(d,self.Z_ancilla_indices[a]) for a,data in enumerate(self.Z_checks) for d in data]
        for i,checks in enumerate([available_X_checks_block1, available_Z_checks_block1]):
            graph = nx.Graph(checks)
            # compute edge coloring to determine CX layers
            coloring = nx.coloring.greedy_color(nx.line_graph(graph), strategy='random_sequential')
            num_colors = max(coloring.values()) + 1
            target_num_colors = 2*[self.n, self.n][i]
            layers = [[] for _ in range(target_num_colors)]

            counters = [0 for _ in range(num_colors)]
            for edge, color in coloring.items():
                q0,q1 = edge
                layers[color + counters[color]].append((q0,q1) if edge in checks else (q1,q0))
                counters[color] += num_colors
                if color + counters[color] >= target_num_colors:
                    counters[color] = 0

            if any(len(layer) == 0 for layer in layers):
                print(f"WARNING: {sum(len(layer)==0 for layer in layers)}/{len(layers)} layers are empty")

            if i == 0:
                self.X_layers = layers
            else:
                self.Z_layers = layers

    def get_stim(self, basis: str, rounds: int, p_g: float, p_m: float, idle_per_round: float = 0.0, midpoint_data_err: float = 0.0) -> stim.Circuit:
        circ = stim.Circuit()
        meas_rec = {}
        circ.append('R' if basis == 'Z' else 'RX', self.data_indices, [])

        idle_per_CX_layer = idle_per_round / len(self.X_layers + self.Z_layers)

        for i in range(rounds):
            if i == rounds//2 and midpoint_data_err:
                circ.append('DEPOLARIZE1', self.data_indices, midpoint_data_err)

            circ.append('RX', self.X_ancilla_indices, [])
            circ.append('R', self.Z_ancilla_indices, [])

            for layer in self.X_layers + self.Z_layers:
                if idle_per_CX_layer:
                    circ.append('DEPOLARIZE1', self.data_indices + self.X_ancilla_indices + self.Z_ancilla_indices, idle_per_CX_layer)
                circ.append('CX', [q for pair in layer for q in pair], [])
                circ.append('DEPOLARIZE2', [q for pair in layer for q in pair], p_g)

            circ.append('MX', self.X_ancilla_indices, p_m)
            for q in self.X_ancilla_indices:
                for qb in meas_rec.keys():
                    meas_rec[qb] = [m-1 for m in meas_rec[qb]]
                meas_rec.setdefault(q, []).append(-1)
            
            circ.append('M', self.Z_ancilla_indices, p_m)
            for q in self.Z_ancilla_indices:
                for qb in meas_rec.keys():
                    meas_rec[qb] = [m-1 for m in meas_rec[qb]]
                meas_rec.setdefault(q, []).append(-1)

            for stab_idx,stab in enumerate(self.X_gauge_stabilizers if basis == 'X' else self.Z_gauge_stabilizers):
                ancillae = [(self.X_ancilla_indices[anc] if basis == 'X' else self.Z_ancilla_indices[anc]) for anc in stab]
                if i == 0:
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[q][-1]) for q in ancillae], (stab_idx if basis == 'X' else stab_idx + len(self.X_gauge_stabilizers), i))
                else:
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[q][-1]) for q in ancillae] + [stim.target_rec(meas_rec[q][-2]) for q in ancillae], (stab_idx if basis == 'X' else stab_idx + len(self.X_gauge_stabilizers), i))
            if i > 0:
                for stab_idx,stab in enumerate(self.Z_gauge_stabilizers if basis == 'X' else self.X_gauge_stabilizers):
                    ancillae = [(self.X_ancilla_indices[anc] if basis == 'X' else self.Z_ancilla_indices[anc]) for anc in stab]
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[q][-1]) for q in ancillae] + [stim.target_rec(meas_rec[q][-2]) for q in ancillae], (stab_idx if basis == 'Z' else stab_idx + len(self.X_gauge_stabilizers), i))

        circ.append('MX' if basis == 'X' else 'M', self.data_indices, p_m)
        for q in self.data_indices:
            for qb in meas_rec.keys():
                meas_rec[qb] = [m-1 for m in meas_rec[qb]]
            meas_rec.setdefault(q, []).append(-1)

        for stab_idx,stab in enumerate(self.X_gauge_stabilizers if basis == 'X' else self.Z_gauge_stabilizers):
            ancillae = [(self.X_ancilla_indices[anc] if basis == 'X' else self.Z_ancilla_indices[anc]) for anc in stab]
            data = [data_q for anc in stab for data_q in (self.X_checks[anc] if basis == 'X' else self.Z_checks[anc])]
            circ.append('DETECTOR', [stim.target_rec(meas_rec[q][-1]) for q in ancillae + data], (stab_idx if basis == 'X' else stab_idx + len(self.X_gauge_stabilizers), i))
        
        X_obs, Z_obs = self.Lx.T, self.Lz.T
        for i,obs in enumerate(X_obs if basis == 'X' else Z_obs):
            circ.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_rec[q][-1]) for q in self.data_indices if obs[q]], i)

        return circ

    def compute_code_parameters(self):
        return (2**self.r-1)**2, self.r**2, 2**(self.r-1)

    def compute_logical_operators(self):
        return self.Lx, self.Lz

    def _simplex_code(self, r: int):
        """
        Generate the [2^r-1, r, 2^(r-1)] (classical) binary simplex code
        """
        primative_polynomials = {
            3: '1101',
            4: '11001',
            5: '101001',
            6: '1100001',
            7: '11000001',
            8: '100011101'
        }
        x =  self._make_cyclic_matrix(2**r-1)
        H = sum([self._matrix_power(x, i) for i in [j for j,exponent in enumerate(primative_polynomials[r]) if exponent == '1']]) 
        return H.T
    

    def data_idx_to_coords(self, idx, sq_ln = None, transposed = False):
        sq_ln = sq_ln or self.n
        r, c = idx // sq_ln, idx % sq_ln
        return (r,c) if not transposed else (c,r)
    
    def logical_idx_to_coords(self, idx, transposed = False):
        return self.data_idx_to_coords(idx, self.r, transposed)
    
    def coords_to_data_idx(self, coords, sq_ln = None, transposed = False):
        sq_ln = sq_ln or self.n
        r,c = coords
        return r*sq_ln+ c if not transposed else c*sq_ln + r
    
    def coords_to_logical_idx(self, coords, transposed = False):
        return self.coords_to_data_idx(coords, self.r, transposed)
    