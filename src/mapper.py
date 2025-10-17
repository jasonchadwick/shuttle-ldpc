import networkx as nx
import numpy as np
from extern.spin_qec.ldpc_code.QECCode import QECCode
from extern.spin_qec.ldpc_code.HGPCode import HGPCode
from extern.spin_qec.ldpc_code.SHYPSCode import SHYPSCode
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
    order_fixed: bool
    coordinated_shuttle: bool # TODO: do we need this?
    def __init__(
            self,
            schedule: dict[int, list[int]],
            order_fixed: bool = True,
            coordinated_shuttle: bool = False,
        ):
        self.schedule = schedule
        self.order_fixed = order_fixed
        self.coordinated_shuttle = coordinated_shuttle

class ShuttleMapper:
    """Builds an abstract shuttling schedule from a QEC code."""
    def __init__(self):
        pass

    def map_code(
            self,
            code: QECCode,
        ) -> ObjectiveSchedule:

        if hasattr(code, 'get_cx_schedule'):
            cx_schedule = code.get_cx_schedule()
            
        else:
            # TODO
            # assume distance-preserving; order checks to minimize shuttle
            # distance (TSP with open boundary)
            raise NotImplementedError

        raise NotImplementedError