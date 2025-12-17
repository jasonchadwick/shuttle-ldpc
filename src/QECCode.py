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