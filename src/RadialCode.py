from qldpc import codes
from qldpc.objects import Pauli
from sympy.abc import x, y
import numpy as np
from src.QECCode import QECCode

import sys
from pathlib import Path
external_path = Path("src/extern/radial/decoding").resolve()
if str(external_path) not in sys.path:
    sys.path.insert(0, str(external_path))
from src.extern.radial.decoding.ckt_noise import SinterDecoder_BPOSD_OWD
import src.extern.radial.decoding.circuit_stuff as cs

CODES_PATH = "notebooks/data/radial_mats/"

def load_code(code: str, r: int, s: int):
    path = CODES_PATH + code + "/"
    hx = np.loadtxt(path + "hx.csv", delimiter=",", dtype=int)
    hz = np.loadtxt(path + "hz.csv", delimiter=",", dtype=int)
    lx = np.loadtxt(path + "lx.csv", delimiter=",", dtype=int)
    lz = np.loadtxt(path + "lz.csv", delimiter=",", dtype=int)
    if len(lx.shape) == 0 or len(lz.shape) == 0:
        print(code, 'failed')
    return cs.Code(r, s, hx, hz, lx, lz)

class RadialCode(QECCode):
    def __init__(
            self,
            r, s, d
        ):
        code = load_code(f'r{r}_s{s}_d{d}', r, s)
        self.d = d
        self.r = code.r
        self.s = code.s
        self.num_data = code.N
        self.data_indices = list(range(self.num_data))
        num_X = code.nX
        num_Z = code.nZ
        self.X_ancilla_indices = list(range(self.num_data, self.num_data + num_X))
        self.Z_ancilla_indices = list(range(self.num_data + num_X, self.num_data + num_X + num_Z))
        self.X_checks = code.checkToBitsX
        self.Z_checks = code.checkToBitsZ

        self.Lx = code.logicalsX
        self.Lz = code.logicalsZ

        self.qubit_coords = []
        m = int(np.sqrt(self.num_data))
        for i in range(self.num_data):
            self.qubit_coords.append((i % m, i // m))

    
    def compute_code_parameters(self):
        return (self.num_data, 2*(self.r-1)**2, self.d)

    def compute_logical_operators(self):
        Lx = np.zeros((len(self.Lx), self.num_data), bool)
        for i,row in enumerate(self.Lx):
            Lx[i,row] = 1
        # Lx[self.Lx] = 1
        Lz = np.zeros((len(self.Lx), self.num_data), bool)
        # Lz[self.Lz] = 1
        for i,row in enumerate(self.Lx):
            Lz[i,row] = 1
        return Lx, Lz
        
        