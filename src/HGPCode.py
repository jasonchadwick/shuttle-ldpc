
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import math
import stim
import networkx as nx
import sys
import json

# utility functions from open source python package
from ldpc.code_util import construct_generator_matrix, estimate_code_distance
from ldpc.mod2 import rank, reduced_row_echelon, row_echelon
from src.QECCode import QECCode

class HGPCode(QECCode):
    """Represents an HGP code, including specifying all parity checks. 
    
    Assumes data qubits are laid out in two adjacent square grids, e.g.:
    ```
    00 03 06 | 09 10 11
    01 04 07 | 12 13 14
    02 05 08 | 15 16 17
    ```
    In this structure, each X check looks at qubits in a single row of each
    block and each Z check looks at qubits in a single column of each block.
    """
    def __init__(self, H1: NDArray[np.bool_], H2: NDArray[np.bool_], compute_code_params: bool = False, compute_logical_ops: bool = False, pretty_print_logical_ops: bool = False):        
        self.H1 = H1
        self.H2 = H2
        self.r1, self.c1 = H1.shape
        self.r2, self.c2 = H2.shape
        #if (self.r1 > self.c1 or self.r2 > self.c2):
        #    raise NotImplementedError(f'Must have r1 <= c1 and r2 <= c2')
        self.num_X_checks = self.c1*self.r2
        self.num_Z_checks = self.r1*self.c2
        self.num_data = self.r1*self.r2 + self.c1*self.c2
        self.check_order_matters = False

        self.block0_rows = max(self.c1, self.c2)
        self.block0_cols = min(self.c1, self.c2)
        self.block0_size = self.c1 * self.c2
        self.block1_rows = max(self.r1, self.r2)
        self.block1_cols = min(self.r1, self.r2)
        self.block1_size = self.r1 * self.r2

        self.data = np.full((self.block0_rows, self.block0_cols + self.block1_cols), -1, dtype=int)
        for i in range(self.num_data):
            coords = self.data_idx_to_coords(i)
            self.data[*coords] = i

        self.Hx = np.zeros((self.num_X_checks, self.num_data), dtype=bool)
        self.Hx[:, :self.block1_size] = np.kron(H1.T, np.eye(self.r2, self.r2, dtype=bool))
        self.Hx[:, self.block1_size:] = np.kron(np.eye(self.c1, self.c1, dtype=bool), H2)
        self.Hz = np.zeros((self.num_Z_checks, self.num_data), dtype=bool)
        self.Hz[:, :self.block1_size] = np.kron(np.eye(self.r1, self.r1, dtype=bool), H2.T)
        self.Hz[:, self.block1_size:] = np.kron(H1, np.eye(self.c2, self.c2, dtype=bool))

        self.X_checks: list[list[int]] = [[] for _ in range(self.num_X_checks)] # each entry is a list of data qubits to check
        self.Z_checks: list[list[int]] = [[] for _ in range(self.num_Z_checks)]
        self.X_check_rows = np.full((self.num_X_checks, 2), -1, dtype=int)
        self.Z_check_cols = np.full((self.num_Z_checks, 2), -1, dtype=int)
        
        for check_idx in range(self.num_X_checks):
            for q_idx in range(self.num_data):
                if self.Hx[check_idx, q_idx]:
                    data_q = q_idx
                    check_idx = check_idx
                    data_coords = self.data_idx_to_coords(data_q)
                    self.X_checks[check_idx].append(data_q)
                    if self.X_check_rows[check_idx, self.block_idx(data_q)] != -1:
                        if self.X_check_rows[check_idx, self.block_idx(data_q)] != data_coords[0]:
                            raise ValueError(f'X check {check_idx} has data qubits in different rows (q{data_q}, prev row {self.X_check_rows[check_idx, self.block_idx(data_q)]})')
                    self.X_check_rows[check_idx, self.block_idx(data_q)] = data_coords[0]
                if self.Hz[check_idx, q_idx]:
                    data_q = q_idx
                    check_idx = check_idx
                    data_coords = self.data_idx_to_coords(data_q)
                    self.Z_checks[check_idx].append(data_q)
                    if self.Z_check_cols[check_idx, self.block_idx(data_q)] != -1:
                        if self.Z_check_cols[check_idx, self.block_idx(data_q)] != data_coords[1]:
                            raise ValueError(f'Z check {check_idx} has data qubits in different columns (q{data_q}, prev col {self.Z_check_cols[check_idx, self.block_idx(data_q)]})')
                    self.Z_check_cols[check_idx, self.block_idx(data_q)] = data_coords[1]
        
        self.data_indices = list(range(self.num_data))
        self.X_ancilla_indices = list(range(self.num_data, self.num_data + self.num_X_checks))
        self.Z_ancilla_indices = list(range(self.num_data + self.num_X_checks, self.num_data + self.num_X_checks + self.num_Z_checks))

        # Compute code parameters
        self.code_params = self.compute_code_parameters() if compute_code_params else None

        ### TODO: REMOVE BEFORE RELEASING THE CODE
        self.pretty_print_logical_ops = pretty_print_logical_ops
        
        # Compute logical operators
        if compute_logical_ops:
            self.logical_X, self.logical_Z = self.compute_logical_operators()

        # assign checks into layers
        self.X_layers = []
        self.Z_layers = []
        available_X_checks_block1 = [(self.X_ancilla_indices[a],d) for a,data in enumerate(self.X_checks) for d in data]
        available_Z_checks_block1 = [(d,self.Z_ancilla_indices[a]) for a,data in enumerate(self.Z_checks) for d in data]
        for i,checks in enumerate([available_X_checks_block1, available_Z_checks_block1]):
            graph = nx.Graph(checks)
            # compute edge coloring to determine CX layers
            coloring = nx.coloring.greedy_color(nx.line_graph(graph), strategy='largest_first')
            num_colors = max(coloring.values()) + 1
            target_num_colors = 2*[self.num_data, self.num_data][i]
            layers = [[] for _ in range(num_colors)]

            # counters = [0 for _ in range(num_colors)]
            for edge, color in coloring.items():
                q0,q1 = edge
                layers[color].append((q0,q1) if edge in checks else (q1,q0))
                # counters[color] += num_colors
                # if color + counters[color] >= target_num_colors:
                    # counters[color] = 0

            if any(len(layer) == 0 for layer in layers):
                print(f"WARNING: {sum(len(layer)==0 for layer in layers)}/{len(layers)} layers are empty")

            if i == 0:
                self.X_layers = layers
            else:
                self.Z_layers = layers

    def compute_code_parameters(self):
        """Compute the code parameters N, K, and D for this code."""
        self.N = self.num_data

        self.k1 = self.r1 - rank(self.H1.astype(int))
        self.k2 = self.r2 - rank(self.H2.astype(int))
        self.k1t = self.c1 - rank(self.H1.T.astype(int))
        self.k2t = self.c2 - rank(self.H2.T.astype(int))
        self.K = self.k1*self.k2 + self.k1t*self.k2t

        
        d_1 = find_classical_code_distance(self.H1)
        d_2 = find_classical_code_distance(self.H2)

        self.D = min(d_1,d_2)
    
        return (self.N,self.K,self.D)
        
    def compute_logical_operators(self) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Compute the logical X and Z operators
        """
        ## TODO: need to cope with non-square blocks

        H_1 = self.H1.astype(int)
        H_2 = self.H2.astype(int)

        generator_basis = []
        image_complement_basis = []
        for i,pcm in enumerate([H_1,H_1.T,H_2,H_2.T]):
            # Find basis for the code words
            G = construct_generator_matrix(pcm).toarray()
            basis = G
            generator_basis.append(basis)
  
            # Find basis for the code words not in the image of the parity check matrix
            _,_,_, pvts = row_echelon(pcm)
            missing_pvts = [i for i in range(pcm.shape[1]) if i not in pvts]
            _, c = pcm.shape  
            pvts_ordered = []
            pvts_back_up = missing_pvts.copy()
            while missing_pvts:
                pvt = missing_pvts.pop(0)
                row_to_check = len(pvts_ordered)
                if row_to_check >= len(basis):
                    pvts_ordered = [[i==pvt for i in range(c)] for pvt in pvts_back_up]
                    break
                if basis[row_to_check,pvt] == 1:
                    pvts_ordered.append([i==pvt for i in range(c)])
                else:
                    missing_pvts.append(pvt)
            
            pvts = np.array(pvts_ordered).astype(int)
            image_complement_basis.append(pvts)

        ## Computing Z logcal operators
        logical_Z = []

        for z1 in generator_basis[1]:
            for z2 in image_complement_basis[3]:
                zero_length = self.num_data - len(z1)*len(z2)
                logical_Z.append(np.hstack((np.kron(z1,z2), np.zeros(zero_length, dtype=int))))
                if self.pretty_print_logical_ops:
                    print(f"Z: {z1}(x){z2}(+)[0]^{zero_length}")

        for z1 in image_complement_basis[0]:
            for z2 in generator_basis[2]:
                zero_length = self.num_data - len(z1)*len(z2)
                logical_Z.append(np.hstack((np.zeros(zero_length, dtype=int), np.kron(z1,z2))))
                if self.pretty_print_logical_ops:
                    print(f'Z: [0]^{zero_length}(+){z1}(x){z2}')

        # Computing X logcal operators
        logical_X = []

        for x1 in image_complement_basis[1]:
            for x2 in generator_basis[3]:
                zero_length = self.num_data - len(x1)*len(x2)
                logical_X.append(np.hstack((np.kron(x1,x2), np.zeros(zero_length, dtype=int))))
                
                if self.pretty_print_logical_ops:
                    print(f"X: {x1}(x){x2}(+)[0]^{zero_length}")


        for x1 in generator_basis[0]:
            for x2 in image_complement_basis[2]:
                zero_length = self.num_data - len(x1)*len(x2)
                logical_X.append(np.hstack((np.zeros(zero_length, dtype=int), np.kron(x1,x2))))
                if self.pretty_print_logical_ops:
                    print(f'X: [0]^{zero_length}(+){x1}(x){x2}')

        logical_X = np.array(logical_X)
        logical_Z = np.array(logical_Z)

        assert (self.Hx@logical_Z.T%2).max() == 0, "Logical Z is not in ker(Hx)"
        assert (self.Hz@logical_X.T%2).max() == 0, "Logical X is not in ker(Hz)"
        
        return logical_X, logical_Z

    @classmethod
    def random(cls, n: int, m: int, seed: int | None = None):
        """Sample two random m-regular, 2n-node bipartite graphs, construct
        classical parity check matrices from them, and return an HGPCode object
        created from those matrices.

        We do this using a Micro-Canonical Configuration Model.
        """
        rng = np.random.default_rng(seed)
        valid = False
        while not valid:
            valid = True
            edges_flat_1 = np.vstack((rng.permutation(np.arange(m*n)), rng.permutation(np.arange(m*n)))).T
            edges_1 = np.floor_divide(edges_flat_1, m)
            _, count = np.unique(np.hstack((edges_1, edges_1[:,::-1])), axis=0, return_counts=True)
            if not all(count == 1):
                valid = False
                continue
            edges_flat_2 = np.vstack((rng.permutation(np.arange(m*n)), rng.permutation(np.arange(m*n)))).T
            edges_2 = np.floor_divide(edges_flat_2, m)
            _, count = np.unique(np.hstack((edges_1, edges_1[:,::-1])), axis=0, return_counts=True)
            if not all(count == 1):
                valid = False
                continue
            H1 = np.zeros((n, n), dtype=bool)
            H2 = np.zeros((n, n), dtype=bool)
            for i,j in edges_1:
                H1[i,j] = True
            for i,j in edges_2:
                H2[i,j] = True
            valid = all(np.sum(H1, axis=0) > 0) and all(np.sum(H1, axis=1) > 0) and all(np.sum(H2, axis=0) > 0) and all(np.sum(H2, axis=1) > 0)
        return cls(H1, H2)
    
    @classmethod
    def cyclic(cls, n: int):
        """
            Create a [[18n^2, 8, 2n]] HGP code with a cyclic checks
            Source: arXiv:2411.03302
        """
        x_perm = cls._make_cyclic_matrix(cls, 3*n)
        p_x = sum([cls._matrix_power(cls, x_perm,i) for i in range(3)])
        return cls(p_x, p_x.T)

    @classmethod
    def small_example(cls):
        """
            Creates a HGP code based on the symmetric Hamming code
        """
        H_hamming = np.array([
            [1, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ])
        H = (H_hamming.T@H_hamming % 2).astype(bool)
        return cls(H,H)

    @classmethod
    def good_HGP_code(cls, n_data : int, d_data : int, d_check : int = 0, min_k :  int = 1, min_d : int = 0, min_girth : int = 6, seed : int | None = None, max_iter : int = 1000, progress : bool = False):
        """
        Generate a good HGP code with parameters n and d using rejection sampling

        Args:
            n_data (int): Number of data bits
            d_data (int): #checks each data bit belongs to
            d_check (int, optional):  weight of checks (default to d_data)
            min_k (int, optional): Minimum number of logical qubits
            min_d (int, optional): Minimum code distance
            min_girth (int, optional): Minimum girth of the Tanner graph
            seed (int, optional): Random seed
            max_iter (int, optional): Maximum number of attempts
            progeress (bool, optional): Print parameters of randomly generated codes
        """
        d_check = d_data if d_check == 0 else d_check
        if progress:
            print(f"Generating first classical code")
        H1, _ = generate_classical_code(n_data, d_data, d_check, min_k, min_d, min_girth, seed, max_iter, progress)
        if progress:
            print(f"Generating second classical code")
        H2, _ = generate_classical_code(n_data, d_data, d_check, min_k, min_d, min_girth, seed, max_iter, progress)
        return cls(H1, H2)

    @classmethod
    def toric_code(cls, d: int):
        """
        Generate a toric code with parameters [[2d^2, 2, d]]
        """
        H1 = np.eye(d, dtype=bool) + np.eye(d, k=1, dtype=bool)
        H2 = np.eye(d, dtype=bool) + np.eye(d, k=1, dtype=bool)
        H1[-1,0] = 1
        H2[-1,0] = 1
        return cls(H1, H2)
    
    @classmethod
    def surface_code(cls, d: int):
        """
        Generate an unrotated surface code with parameters [[2d^2, 2, d]]
        """
        H1 = (np.eye(d, dtype=bool) + np.eye(d, k=1, dtype=bool))[:-1]
        return cls(H1, H1)
    
    @classmethod
    def read_json(cls, filename: str):
        """
        Read an HGP code from a JSON file
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            H1 = np.array(data['H1'], dtype=bool)
            H2 = np.array(data['H2'], dtype=bool)
            return cls(H1, H2)


    @classmethod
    def simplex_code(cls, r: int):
        """
        Generate [[2(2^r-1)^2, 2r^2, 2^(r-1)]] HGP simplex code 
        arXiv:2502.07150, arXiv:2210.03537
        """
        primative_polynomials = {
            3: '1101',
            4: '11001',
            5: '101001',
            6: '1100001',
            7: '11000001',
            8: '100011101'
        }
        x =  cls._make_cyclic_matrix(cls, 2**r-1)
        H = sum([cls._matrix_power(cls, x, i) for i in [j for j,exponent in enumerate(primative_polynomials[r]) if exponent == '1']]) 
        return cls(H,H)


    def _make_cyclic_matrix(self, n: int) -> NDArray:
        """
        Util function for code construction
        Make a cyclic matrix of size n x n
        """
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            matrix[i, (i+1) % n] = 1
        return matrix


    def _matrix_power(self, matrix: NDArray, power: int) -> NDArray:
        """
        Util function for code construction
        Raise a matrix to a power
        """
        result = np.eye(matrix.shape[0], dtype=int)
        for _ in range(power):
            result = result @ matrix
        return result

    def data_idx_to_coords(self, idx: int, transposed: bool = False) -> tuple[int, int]:
        if idx >= self.block1_size: # block 1
            idx -= self.block1_size
            r,c = idx // self.block0_cols, idx % self.block0_cols
            if transposed:
                return c,r
            else:
                return r,c
        else:
            r,c = idx % self.block1_cols, self.block0_cols + idx // self.block1_cols
            if transposed:
                return c-self.block1_cols, r+self.block1_cols
            else:
                return r,c
        
    def coords_to_data_idx(self, coords: tuple[int, int], transposed: bool = False) -> int:
        x, y = coords
        q = 0
        if y >= self.block1_cols: # left block
            y -= self.block0_cols
            if transposed:
                x,y = y,x
            q = y*self.block0_cols + x
        else: # right block
            if transposed:
                x,y = y,x
            q = x*self.block1_cols + y + self.block0_size
        if q >= self.num_data:
            raise ValueError(f'Invalid coordinates {coords}')
        return q
    
    def block_idx(self, qubit_idx: int) -> int:
        return 1 if qubit_idx < self.block1_size else 0
    
    def get_stim(self, basis: str, rounds: int, p_g: float, p_m: float, idle_per_round: float = 0.0, midpoint_data_err: float = 0.0) -> stim.Circuit:
        circ = stim.Circuit()
        
        for q in self.data_indices:
            circ.append('QUBIT_COORDS', [q], tuple(self.data_idx_to_coords(q)))
        
        meas_rec = {}
        circ.append('R' if basis == 'Z' else 'RX', self.data_indices, [])

        # assume that we can perform the movements in 
        idle_per_CX_layer = idle_per_round / (max([len(checks) for checks in self.X_checks] + [len(checks) for checks in self.Z_checks]))

        for i in range(rounds):
            # if i == rounds//2 and midpoint_data_err:
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

            for stab_idx,stab in enumerate(self.X_checks if basis == 'X' else self.Z_checks):
                ancilla_idx = (self.X_ancilla_indices[stab_idx] if basis == 'X' else self.Z_ancilla_indices[stab_idx])
                if i == 0:
                    circ.append('DETECTOR', stim.target_rec(meas_rec[ancilla_idx][-1]), (ancilla_idx, i))
                else:
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[ancilla_idx][-1]), stim.target_rec(meas_rec[ancilla_idx][-2])], (ancilla_idx, i))
            if i > 0:
                for stab_idx,stab in enumerate(self.Z_checks if basis == 'X' else self.X_checks):
                    ancilla_idx = (self.X_ancilla_indices[stab_idx] if basis == 'X' else self.Z_ancilla_indices[stab_idx])
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[ancilla_idx][-1]), stim.target_rec(meas_rec[ancilla_idx][-2])], (ancilla_idx, i))

        circ.append('MX' if basis == 'X' else 'M', self.data_indices, p_m)
        for q in self.data_indices:
            for qb in meas_rec.keys():
                meas_rec[qb] = [m-1 for m in meas_rec[qb]]
            meas_rec.setdefault(q, []).append(-1)

        for stab_idx,stab in enumerate(self.X_checks if basis == 'X' else self.Z_checks):
            ancilla_idx = (self.X_ancilla_indices[stab_idx] if basis == 'X' else self.Z_ancilla_indices[stab_idx])
            circ.append('DETECTOR', [stim.target_rec(meas_rec[q][-1]) for q in [ancilla_idx] + stab], (ancilla_idx, i))
        
        X_obs, Z_obs = self.compute_logical_operators()
        X_obs = X_obs.T
        Z_obs = Z_obs.T
        for i,obs in enumerate(X_obs if basis == 'X' else Z_obs):
            circ.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_rec[q][-1]) for q in self.data_indices if obs[q]], i)

        return circ
        
## Helper functions for generating good HGP codes


def find_classical_code_distance(pcm:NDArray, rk = 0) -> int:
    """
    Find the distance of a classical code given its parity check matrix
    """
    H = pcm.astype(int)

    n_check, n_data = H.shape
    
    rk = rank(H) if rk == 0 else rk

    if rk == n_data:
    # 'This is a trivial code (no logical bits)'
        return -1

    if rk == n_check and n_check < n_data:
        return  estimate_code_distance(H)[0]
    
    d_H =  estimate_code_distance(H)[0]    
    d_Ht =  estimate_code_distance(H.T)[0]
    return min(d_H,d_Ht)

def generate_bipartite_graph(n, m, p, q, seed=None):
    """ 
    Generate a (p,q)-regular bipartite graph with (n+m) nodes at random
    """

    rng = np.random.default_rng(seed)
    
    # Create stubs for both sets of nodes
    left_stubs = np.repeat(np.arange(n), p)
    right_stubs = np.repeat(np.arange(m), q)
    
    # Shuffle the stubs
    rng.shuffle(left_stubs)
    rng.shuffle(right_stubs)
    
    # Create edges by pairing the stubs
    edges = np.vstack((left_stubs, right_stubs)).T
    H = np.zeros((m, n), dtype=int)
    G = nx.Graph()

    for u, v in edges:
        H[v, u] = 1
        G.add_edge(u, v+n)
    return H, G

def _generate_classical_code(n_data, n_check, d_data, d_check, min_k, min_d, min_girth, seed, progress):
    """
    Generate a classical LDPC code with the given parameters
    Args:
        n_data (int): Number of data bits
        n_check (int): Number of checks
        d_data (int): #checks each data bit belongs to
        d_check (int): weight of checks
        min_k (int): Minimum number of logical bits
        min_d (int): Minimum code distance
        min_girth (int): Minimum girth of the Tanner graph
    Returns:
        H (ndarray): Parity check matrix
        G (Graph): Tanner graph (factor graph) of the code
    """

    H, G = generate_bipartite_graph(n_data,n_check,d_data, d_check, seed)

    # Reject graphs with girth less than min_girth 
    girth = nx.algorithms.girth(G)
    while (girth < min_girth):
        return _generate_classical_code(n_data,n_check,d_data, d_check, min_k, min_d, min_girth, seed, progress)
    
    # Reject check matrices with full rank
    k = n_data - rank(H)
    if k < min_k:
        if progress:
            print(f"\t\t...Rejecting [{n_data},{k},*] code with girth {girth}")
        return _generate_classical_code(n_data,n_check,d_data, d_check, min_k, min_d, min_girth, seed, progress)
    

    # Reject distance is small
    d = find_classical_code_distance(H, n_data-k)

    if d < min_d:
        if progress:
            print(f"\t\t...Rejecting [{n_data},{k},{d}] code with girth {girth}")
        return _generate_classical_code(n_data,n_check,d_data, d_check, min_k, min_d, min_girth, seed, progress)
    
    if progress:
        print(f"\tGenerated [{n_data},{k},{d}] code with girth {girth}")
    return H, G

def generate_classical_code(n_data : int, d_data : int, d_check : int = 0, min_k :  int = 1, min_d : int  = 1, min_girth : int = 6, seed : int | None = None, max_iter : int = 1000, progress : bool = False):
    """
    Generate a classical LDPC code with specified girth via rejection sampling

    Args:
        n_data (int): Number of data bits
        d_data (int): #checks each data bit belongs to
        d_check (int, optional):  weight of checks (default to d_data)
        min_k (int, optional): Minimum number of logical qubits
        min_d (int, optional): Minimum code distance
        min_girth (int, optional): Minimum girth of the Tanner graph
        seed (int, optional): Random seed
        max_iter (int, optional): Maximum number of iterations
        progress (bool, optional): Print progress
    Returns:
        H (ndarray): Parity check matrix
        G (Graph): Tanner graph (factor graph) of the code
    """
    # Calculate the number of check nodes
    d_check = d_data if d_check == 0 else d_check
    n_check = int(d_data * n_data / d_check)
    if d_data* n_data != d_check* n_check:
        raise ValueError("The degree sequences are not valid")
    if progress:
        print(f"\t...To generate [{n_data},>={min_k},>={min_d}] code with girth >={min_girth}")
    sys.setrecursionlimit(max_iter)
    H, G = _generate_classical_code(n_data, n_check, d_data, d_check, min_k, min_d, min_girth, seed, progress)
    sys.setrecursionlimit(1000)

    return H, G



####### Will be useful later for finding single-shot decoable codes
def _cheeger_bounds(graph):
    """
    Compute bounds on the Cheeger constant of a regular bipartite graph using its spectrum.
    
    :param graph: A NetworkX graph
    :return: A tuple (lower_bound, upper_bound) for the Cheeger constant
    """
    # Compute the adjacency matrix
    A = nx.adjacency_matrix(graph).toarray()
    
    # Compute eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.sort(np.abs(eigenvalues))  # Sort eigenvalues by absolute value
    
    d = eigenvalues[-1]  # Largest eigenvalue (regular degree)
    lambda_2 = eigenvalues[-2]  # Second-largest eigenvalue
    
    # Cheeger's inequality bounds
    lower_bound = (d - lambda_2) / 2
    upper_bound = np.sqrt(2 * d * (d - lambda_2))
    
    return lower_bound, upper_bound
