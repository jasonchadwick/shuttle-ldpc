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