import networkx as nx
import numpy as np
from extern.spin_qec.ldpc_code.QECCode import QECCode
from extern.spin_qec.ldpc_code.HGPCode import HGPCode
from extern.spin_qec.ldpc_code.SHYPSCode import SHYPSCode
from extern.spin_qec.ldpc_code.GBCode import GBCode
from extern.spin_qec.ldpc_code.RotatedSurfaceCode import RotatedSurfaceCode

# TODO: is it enough to just make a list of objectives? Will this automatically
# translate to the desired shuttling schedules for e.g. GB codes and surface
# codes?

# A schedule is a dict mapping ancilla qubit indices to ordered lists of data
# qubits they must interact with. 

class ObjectiveSchedule:
    """Represents the schedule of CXs that each ancilla qubit must perform to
    measure its stabilizer.
    """
    schedule: dict[int, list[int]]
    data_coords: dict[int, tuple[int, int]]
    order_fixed: bool
    def __init__(
            self,
            schedule: dict[int, list[int]],
            data_coords: dict[int, tuple[int, int]],
            order_fixed: bool = True,
        ):
        self.schedule = schedule
        self.data_coords = data_coords
        self.order_fixed = order_fixed

class ShuttleMapper:
    """Builds an abstract shuttling schedule from a QEC code."""
    def __init__(self):
        pass

    def map_code(
            self,
            code: QECCode,
        ) -> ObjectiveSchedule:

        schedule = {}
        order_fixed = False
        if hasattr(code, 'get_cx_schedule'):
            cx_schedule = code.get_cx_schedule()
            for layer in cx_schedule:
                for ctrl,tgt in layer:
                    if ctrl in code.X_ancilla_indices + code.Z_ancilla_indices:
                        schedule.setdefault(ctrl, []).append(tgt)
                    else:
                        schedule.setdefault(tgt, []).append(ctrl)
            order_fixed = True
        else:
            for X_i,X_check in enumerate(code.X_checks):
                schedule[code.X_ancilla_indices[X_i]] = [code.data_indices[d] for d in X_check]
            for Z_i,Z_check in enumerate(code.Z_checks):
                schedule[code.Z_ancilla_indices[Z_i]] = [code.data_indices[d] for d in Z_check]

        return ObjectiveSchedule(
            schedule=schedule,
            data_coords=self._place_data(code),
            order_fixed=order_fixed,
        )
    
    def _place_data(self, code: QECCode) -> dict[int, tuple[int, int]]:
        if isinstance(code, HGPCode):
            raise NotImplementedError
        elif isinstance(code, RotatedSurfaceCode):
            return {q: code.qubit_coords[q] for q in code.data_indices}
        elif isinstance(code, GBCode):
            raise NotImplementedError
        else:
            # place on a square grid
            rows = int(np.sqrt(code.num_data))
            cols = int(np.ceil(code.num_data / rows))
            r = 0
            c = 0
            coords = {}
            for q in code.data_indices:
                coords[q] = (r,c)
                c += 1
                if c >= cols:
                    c = 0
                    r += 1
            return coords