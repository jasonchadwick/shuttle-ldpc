from qldpc import codes
from qldpc.objects import Pauli
from sympy.abc import x, y
import numpy as np
from src.QECCode import QECCode

class HypercubeCode(QECCode):
    def __init__(
            self,
            r: int
        ):
        self.r = r
        self.qldpc_code = codes.ManyHypercubeCode(r)
        self.num_data = self.qldpc_code.num_qubits
        self.data_indices = list(range(self.num_data))
        Hx = self.qldpc_code.get_matrix(Pauli.X)
        Hz = self.qldpc_code.get_matrix(Pauli.Z)
        num_X = Hx.shape[0]
        num_Z = Hz.shape[0]
        self.X_ancilla_indices = list(range(self.num_data, self.num_data + num_X))
        self.Z_ancilla_indices = list(range(self.num_data + num_X, self.num_data + num_X + num_Z))
        self.X_checks = [[int(x) for x in np.nonzero(Hx[a,:])[0]] for a in range(Hx.shape[0])]
        self.Z_checks = [[int(x) for x in np.nonzero(Hz[a,:])[0]] for a in range(Hz.shape[0])]

        self.qubit_coords = []
        m = int(np.sqrt(self.num_data))
        for i in range(self.num_data):
            self.qubit_coords.append((i % m, i // m))

        assert self.num_data == 6**self.r
        assert self.qldpc_code.dimension == 4**self.r

    def compute_code_parameters(self, distance_bound_num_trials: int = 100):
        return (self.num_data, self.qldpc_code.dimension, 2**self.r)

    def compute_logical_operators(self):
        Lx = np.array(self.qldpc_code.get_logical_ops(Pauli.X), dtype=bool)
        Lz = np.array(self.qldpc_code.get_logical_ops(Pauli.Z), dtype=bool)
        return Lx, Lz
        
        