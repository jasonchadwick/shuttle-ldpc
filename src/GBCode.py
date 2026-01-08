from qldpc import codes
from qldpc.objects import Pauli
from sympy.abc import x, y
import numpy as np
from src.QECCode import QECCode

class GBCode(QECCode):
    def __init__(
            self,
            l: int,
            m: int,
            a_orders: tuple[list[int], list[int]],
            b_orders: tuple[list[int], list[int]],
        ):
        orders = {x: l, y: m}
        poly_a = 0
        for a_ord in a_orders[0]:
            poly_a += x**a_ord
        for a_ord in a_orders[1]:
            poly_a += y**a_ord
        poly_b = 0
        for b_ord in b_orders[0]:
            poly_b += x**b_ord
        for b_ord in b_orders[1]:
            poly_b += y**b_ord
        self.qldpc_code = codes.BBCode(orders, poly_a, poly_b)
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
        for i in range(self.num_data):
            self.qubit_coords.append((i % m, i // m))

    def get_distance_bound(self, num_trials: int):
        return self.qldpc_code.get_distance_bound(num_trials=num_trials)
    
    def compute_code_parameters(self, distance_bound_num_trials: int = 100):
        return (self.num_data, self.qldpc_code.dimension, self.get_distance_bound(distance_bound_num_trials))

    def compute_logical_operators(self):
        Lx = np.array(self.qldpc_code.get_logical_ops(Pauli.X), dtype=bool)
        Lz = np.array(self.qldpc_code.get_logical_ops(Pauli.Z), dtype=bool)
        return Lx, Lz