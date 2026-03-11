"""Parent class for quantum error correcting codes."""
import numpy as np
from numpy.typing import NDArray

class QECCode:
    num_data: int
    data_indices: list[int]
    X_ancilla_indices: list[int]
    Z_ancilla_indices: list[int]

    # Each entry is a list of data qubit indices that are checked by the i-th
    # ancilla qubit.
    X_checks: list[list[int]]
    Z_checks: list[list[int]]

    # For a subsystem code, specifies the X and Z check results (indices into
    # X_checks and Z_checks) to multiply together to get the actual stabilizers.
    # If empty, X_checks and Z_checks are used directly as stabilizers.
    X_gauge_stabilizers: list[list[int]] = []
    Z_gauge_stabilizers: list[list[int]] = []

    # Can be defined for easier visualization of the code layout, but does not
    # change simulation results.
    qubit_coords: list[tuple[int, int]] = []

    check_cx_layers: list[list[tuple[int, int]]] = []

    def compute_code_parameters(self):
        raise NotImplementedError
    
    def compute_logical_operators(self):
        raise NotImplementedError
    
    def get_detector_coords(self) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
        """Get coordinates of X and Z stabilizer detectors in the generated Stim
        circuit. Needed for windowed decoding.
        
        This must remain consistent with the detector coordinate standard
        used in simulate.py."""
        x_coords, z_coords = {}, {}
        if self.X_gauge_stabilizers:
            assert self.Z_gauge_stabilizers
            x_coords = {i:(i,) for i in range(len(self.X_gauge_stabilizers))}
            z_coords = {i:(i+len(self.X_gauge_stabilizers),) for i in range(len(self.Z_gauge_stabilizers))}
        else:
            x_coords = {i:self.qubit_coords[ancilla_idx] if self.qubit_coords else (ancilla_idx,) for i,ancilla_idx in enumerate(self.X_ancilla_indices)}
            z_coords = {i:self.qubit_coords[ancilla_idx] if self.qubit_coords else (ancilla_idx,) for i,ancilla_idx in enumerate(self.Z_ancilla_indices)}
        return x_coords, z_coords
    
    def get_Hz(self) -> NDArray[np.bool_]:
        """Returns the Z stabilizer matrix for the code."""
        Hz = np.zeros((len(self.Z_ancilla_indices), len(self.data_indices)), dtype=bool)
        for i, ancilla_idx in enumerate(self.Z_ancilla_indices):
            for j, data_idx in enumerate(self.Z_checks[i]):
                Hz[i, self.data_indices.index(data_idx)] = 1
        assert not any(np.sum(Hz, axis=1) == 0), "Stabilizer matrix has empty rows"
        assert not any(np.sum(Hz, axis=0) == 0), "Stabilizer matrix has empty columns"
        return Hz

    def get_Hx(self) -> NDArray[np.bool_]:
        """Returns the X stabilizer matrix for the code."""
        Hx = np.zeros((len(self.X_ancilla_indices), len(self.data_indices)), dtype=bool)
        for i, ancilla_idx in enumerate(self.X_ancilla_indices):
            for j, data_idx in enumerate(self.X_checks[i]):
                Hx[i, self.data_indices.index(data_idx)] = 1
        assert not any(np.sum(Hx, axis=1) == 0), "Stabilizer matrix has empty rows"
        assert not any(np.sum(Hx, axis=0) == 0), "Stabilizer matrix has empty columns"
        return Hx


    def _rref_gf2(self, matrix):
        """Compute the Reduced Row Echelon Form of a matrix over GF(2)."""
        A = matrix.copy() % 2
        rows, cols = A.shape
        r = 0
        pivots = []
        for c in range(cols):
            if r >= rows: break
            
            pivot_row = r
            while pivot_row < rows and A[pivot_row, c] == 0:
                pivot_row += 1
                
            if pivot_row == rows: continue
            
            # Swap current row with pivot row
            A[[r, pivot_row]] = A[[pivot_row, r]]
            pivots.append(c)
            
            # Eliminate other 1s in the current column
            for i in range(rows):
                if i != r and A[i, c] == 1:
                    A[i] = (A[i] + A[r]) % 2
            r += 1
            
        return A, pivots

    def _nullspace_gf2(self, matrix):
        """Compute the null space of a binary matrix over GF(2)."""
        A, pivots = self._rref_gf2(matrix)
        rows, cols = A.shape
        rank = len(pivots)
        nullity = cols - rank
        
        if nullity == 0:
            return np.zeros((0, cols), dtype=int)
        
        free_vars = [c for c in range(cols) if c not in pivots]
        ns = np.zeros((nullity, cols), dtype=int)
        
        for i, fv in enumerate(free_vars):
            ns[i, fv] = 1
            for j, pv in enumerate(pivots):
                ns[i, pv] = A[j, fv]
                
        return ns

    def compute_logicals(self, hx, hz):
        """
        Compute CSS logical operators Lx and Lz.
        Lx in ker(Hz) and Lz in ker(Hx) such that Lx @ Lz.T = I_k mod 2.
        """
        # 1. Lx candidates commute with Hz; Lz candidates commute with Hx
        Kx = self._nullspace_gf2(hz)
        Kz = self._nullspace_gf2(hx)
        
        # 2. Compute the symplectic intersection matrix M
        M = (Kx @ Kz.T) % 2
        
        # 3. Diagonalize M over GF(2) using row/col operations to find canonical pairs
        rows, cols = M.shape
        r = 0
        for _ in range(min(rows, cols)):
            pivot_found = False
            # Find a 1 in the remaining M[r:, r:] submatrix
            for i in range(r, rows):
                for j in range(r, cols):
                    if M[i, j] == 1:
                        # Swap rows r and i (updates M and Kx)
                        if i != r:
                            M[[r, i]] = M[[i, r]]
                            Kx[[r, i]] = Kx[[i, r]]
                        # Swap cols r and j (updates M and Kz)
                        if j != r:
                            M[:, [r, j]] = M[:, [j, r]]
                            Kz[[r, j]] = Kz[[j, r]]
                        pivot_found = True
                        break
                if pivot_found:
                    break
            
            if not pivot_found:
                break # Reached the rank of M (this rank is exactly 'k')
                
            # Eliminate all other 1s in column r
            for i in range(rows):
                if i != r and M[i, r] == 1:
                    M[i] = (M[i] + M[r]) % 2
                    Kx[i] = (Kx[i] + Kx[r]) % 2
                    
            # Eliminate all other 1s in row r
            for j in range(cols):
                if j != r and M[r, j] == 1:
                    M[:, j] = (M[:, j] + M[:, r]) % 2
                    Kz[j] = (Kz[j] + Kz[r]) % 2
                    
            r += 1
            
        # The first r rows now form our conjugate logical operator pairs
        return Kx[:r], Kz[:r], r

class TestCode(QECCode):
    # Data qubits are in a grid, X checks are along cols, Z checks are along
    # rows. Doesn't correspond to any actual QEC code, just for testing
    # scheduling
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.num_data = w*h
        self.data_indices = list(range(self.num_data))
        self.data_coords = [(i%w, i//w) for i in self.data_indices]
        self.data_coord_to_idx = {c:i for i,c in zip(self.data_indices, self.data_coords)}
        self.X_ancilla_indices = []
        self.X_checks = []
        self.Z_ancilla_indices = []
        self.Z_checks = []

    def add_all_checks(self):
        self.X_ancilla_indices = list(range(self.num_data, self.num_data + self.w))
        self.X_checks = [[self.data_coord_to_idx[(x,y)] for y in range(self.h)] for x in range(self.w)]
        self.Z_ancilla_indices = list(range(self.num_data + self.w, self.num_data + self.w + self.h))
        self.Z_checks = [[self.data_coord_to_idx[(x,y)] for x in range(self.w)] for y in range(self.h)]

    def add_row_check(self, y: int):
        self.Z_ancilla_indices.append(self.num_data + len(self.Z_ancilla_indices) + len(self.X_ancilla_indices))
        self.Z_checks.append([self.data_coord_to_idx[(x,y)] for x in range(self.w)])
    
    def add_col_check(self, x: int):
        self.X_ancilla_indices.append(self.num_data + len(self.Z_ancilla_indices) + len(self.X_ancilla_indices))
        self.X_checks.append([self.data_coord_to_idx[(x,y)] for y in range(self.h)])