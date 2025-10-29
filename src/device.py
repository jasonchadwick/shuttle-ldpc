from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

@dataclass
class HardwareParams:
    T2: float # in seconds
    cx_err: float
    cx_duration: float # in ns
    h_err: float
    h_duration: float # in ns
    shuttle_err: float # per dot-length shuttle
    shuttle_duration: float # per shuttle, in ns
    init_err: float
    init_duration: float
    measure_err: float
    measure_duration: float

def hardware_params(
        p: float,
        T2: float | None = None,
        shuttle_mul: int = 100,
    ) -> HardwareParams:
    return HardwareParams(
        T2=T2 if T2 is not None else shuttle_mul / np.log(1 - p),
        cx_err=p,
        cx_duration=0,
        h_err=p,
        h_duration=0,
        shuttle_err=p/shuttle_mul,
        shuttle_duration=1,
        init_err=p,
        init_duration=0,
        measure_err=p,
        measure_duration=0,
    )

default_params = hardware_params(1e-3)

class InstrName(Enum):
    CX = 0
    SHUTTLE = 1
    MEASURE = 2
    H = 4
    CZ = 5
    INSTANTIATE = 6
    REMOVE_QUBIT = 7

@dataclass
class Instruction:
    name: InstrName
    qubits: list[int]
    spins: list[list[int]]
    readouts: list[int]
    dots: list[tuple[int, int]]
    duration: int
    error: float
    kwargs: dict[str, Any] = field(default_factory=dict)

class SpinBusDevice:
    def __init__(self) -> None:
        raise NotImplementedError

################################################################################
# below is old code
################################################################################

class EdgeType(Enum):
    DOT = 0
    QUBIT = 1
    READOUT = 2

class EODevice:
    def __init__(
            self,
            hardware_params: HardwareParams,
            layout: tuple[bool, bool, bool, bool],
            nrows: int,
            ncols: int,
            qubits: list[tuple[tuple[int, int], ...] | None],
            rows_to_skip_meas: list[int] = [],
            cols_to_skip_meas: list[int] = [],
            save_animation_frames: bool = False,
            data_indices: list[int] = [],
            X_check_indices: list[int] = [],
            Z_check_indices: list[int] = [],
            qubits_to_enforce_dag_order: list[int] = [],
        ):
        """Initialize the device, defining the locations of the dots and
        readouts, as well as initial placement of the qubits.
        
        Args:
            hardware_params: Hardware parameters.
            layout: (readout_top, readout_right, readout_bottom, readout_left).
            nrows: Number of rows of qubits.
            ncols: Number of columns of qubits.
            qubits: List of 3-tuples of EO qubit coordinates.
            rows_to_skip_meas: List of row indices where we do not need a
                measurement unit.
            cols_to_skip_meas: List of column indices where we do not need a
                measurement unit.
        """
        self.hardware_params = hardware_params
        self.layout = layout
        self.save_animation_frames = save_animation_frames
        self.data_indices = data_indices
        self.X_check_indices = X_check_indices
        self.Z_check_indices = Z_check_indices
        self._frame_idx = 0

        # Define qubits
        self.nrows = nrows
        self.ncols = ncols
        self._qubit_array = np.full((nrows, ncols), -1, dtype=int)
        self._qubits_init = qubits
        for i,qubit in enumerate(qubits):
            if qubit:
                self._qubit_array[qubit[0][0], qubit[0][1]] = i
                self._qubit_array[qubit[1][0], qubit[1][1]] = i
                self._qubit_array[qubit[2][0], qubit[2][1]] = i
        self._qubits = qubits
        
        # Define readout devices
        readout_size = 1
        self.readout_devices = []
        self._readout_array = np.full((nrows, ncols), -1, dtype=int)
        self._rows_to_skip_meas = rows_to_skip_meas
        self._cols_to_skip_meas = cols_to_skip_meas
        readout_idx = 0
        if layout[0]:
            # assuming readout capabilities as shown in
            # https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.011023
            # appendix X
            col_top = 0
            while col_top < ncols:
                if col_top in cols_to_skip_meas:
                    col_top += 1
                    continue
                self._readout_array[:2, col_top:col_top+readout_size] = readout_idx
                self.readout_devices.append([(0, col_top+i) for i in range(readout_size)] + [(1, col_top+i) for i in range(readout_size)])
                readout_idx += 1
                col_top += readout_size
        if layout[1]:
            row_right = 0
            while row_right < nrows:
                if row_right in rows_to_skip_meas:
                    row_right += 1
                    continue
                self._readout_array[row_right:row_right+readout_size, -2:] = readout_idx
                self.readout_devices.append([(row_right+i, ncols-2) for i in range(readout_size)] + [(row_right+i, ncols-1) for i in range(readout_size)])
                readout_idx += 1
                row_right += readout_size
        if layout[2]:
            col_bottom = 0
            while col_bottom < ncols:
                if col_bottom in cols_to_skip_meas:
                    col_bottom += 1
                    continue
                self._readout_array[-2:, col_bottom:col_bottom+readout_size] = readout_idx
                self.readout_devices.append([(nrows-2, col_bottom+i) for i in range(readout_size)] + [(nrows-1, col_bottom+i) for i in range(readout_size)])
                readout_idx += 1
                col_bottom += readout_size
        if layout[3]:
            row_left = 0
            while row_left < nrows:
                if row_left in rows_to_skip_meas:
                    row_left += 1
                    continue
                self._readout_array[row_left:row_left+readout_size, :2] = readout_idx
                self.readout_devices.append([(row_left+i, 0) for i in range(readout_size)] + [(row_left+i, 1) for i in range(readout_size)])
                readout_idx += 1
                row_left += readout_size

        self.schedule = []
        self.schedule_dag = nx.DiGraph()
        self._instructions_by_target = dict() # qubit, readout, or dot -> most recent instruction index
    
    def fresh_copy(self):
        """Return a fresh copy of the device (a new instance with the same
        state as the current one had upon initialization)."""
        new_device = EODevice(
            self.hardware_params,
            self.layout,
            self._qubit_array.shape[0],
            self._qubit_array.shape[1],
            self._qubits_init,
            rows_to_skip_meas=self._rows_to_skip_meas,
            cols_to_skip_meas=self._cols_to_skip_meas,
            save_animation_frames=self.save_animation_frames,
            data_indices=self.data_indices,
            X_check_indices=self.X_check_indices,
            Z_check_indices=self.Z_check_indices,
        )
        return new_device

    def is_horizontal(self, qubit_idx: int) -> bool:
        """Check if a qubit is horizontally oriented."""
        dots = self.qubit_coords(qubit_idx)
        return dots[0][0] == dots[1][0] == dots[2][0]

    def is_vertical(self, qubit_idx: int) -> bool:
        """Check if a qubit is vertically oriented."""
        dots = self.qubit_coords(qubit_idx)
        return dots[0][1] == dots[1][1] == dots[2][1]

    def qubit_coords(self, qubit_idx: int) -> tuple[tuple[int, int], ...]:
        """Get the coordinates of the dots of a qubit."""
        dots = self._qubits[qubit_idx]
        if not dots:
            raise ValueError(f'Qubit {qubit_idx} does not exist.')
        return dots
    
    def _check_contiguous_qubit(self, qubit_idx: int, silent: bool = False) -> bool:
        """Check if a qubit is contiguous."""
        dots = self.qubit_coords(qubit_idx)
        connected_01 = int(np.linalg.norm(np.array(dots[0]) - np.array(dots[1])) == 1)
        connected_12 = int(np.linalg.norm(np.array(dots[1]) - np.array(dots[2])) == 1)
        connected_02 = int(np.linalg.norm(np.array(dots[0]) - np.array(dots[2])) == 1)
        if (connected_01 + connected_12 + connected_02) < 2:
            if silent:
                return False
            else:
                raise ValueError(f'Qubit {qubit_idx} is not contiguous Dots: {dots}.')
        else:
            return True
    
    def _check_connected_qubits(self, qubit_idx0: int, qubit_idx1: int, silent: bool = False) -> bool:
        dots0 = self.qubit_coords(qubit_idx0)
        dots1 = self.qubit_coords(qubit_idx1)
        connected = False
        for dot0 in dots0:
            for dot1 in dots1:
                if np.linalg.norm(np.array(dot0) - np.array(dot1)) == 1:
                    connected = True
                    break
            if connected:
                break
        if not connected:
            if silent:
                return False
            else:
                raise ValueError(f'Qubits {qubit_idx0} and {qubit_idx1} are not connected.')
        else:
            return True
        
    def _check_shuttle_dot(self, source_dot: tuple[int, int], destination_dot: tuple[int, int], silent: bool = False) -> bool:
        if self._qubit_array[source_dot] == -1 or self._qubit_array[destination_dot] != -1:
            if silent:
                return False
            else:
                raise ValueError(f'Attempted to shuttle {source_dot} to {destination_dot}, but source is empty or destination is already occupied (by qubit {self._qubit_array[destination_dot]}?).')
        else:
            return True
        
    def _get_readout_idx(self, dots: list[tuple[int, int]]) -> int:
        measure_indices = []
        for dot in dots:
            if self._readout_array[dot[0], dot[1]] > -1:
                measure_indices.append(self._readout_array[dot[0], dot[1]])
        chosen_readout = max(set(measure_indices), key=measure_indices.count, default=-1)
        if measure_indices.count(chosen_readout) < 2:
            return -1
        else:
            return chosen_readout
        
    def _check_readout_valid(self, dots: list[tuple[int, int]], silent: bool = False) -> bool:
        if self._get_readout_idx(list(dots)) == -1:
            if silent:
                return False
            else:
                raise ValueError(f'Dots {dots} does not have a valid readout.')
        else:
            return True
    
    def _check_init(self, dots: list[tuple[int, int]], silent: bool = False) -> bool:
        if self._qubit_array[dots[0]] != -1 or self._qubit_array[dots[1]] != -1 or self._qubit_array[dots[2]] != -1:
            if silent:
                return False
            else:
                raise ValueError('Dots are already occupied.', dots)
        else:
            return True

    def _execute_instruction(self, instr_idx: int):
        instr = self.schedule[instr_idx]
        if instr.name == InstrName.CX:
            self.cx(instr.qubits[0], instr.qubits[1], instr_idx)
        elif instr.name == InstrName.CZ:
            self.cz(instr.qubits[0], instr.qubits[1], instr_idx)
        elif instr.name == InstrName.H:
            self.h(instr.qubits[0], instr_idx)
        elif instr.name == InstrName.MEASURE:
            self.measure(instr.qubits[0], instr_idx)
        elif instr.name == InstrName.SWAP:
            self.swap(instr.qubits[0], instr.qubits[1], instr_idx)
        elif instr.name == InstrName.SHUTTLE_DOT:
            self.shuttle_single_dot(instr.dots[0], instr.dots[1], instr_idx)
        elif instr.name == InstrName.INSTANTIATE:
            self.instantiate(instr.qubits[0], tuple(instr.dots), instr_idx)
        elif instr.name == InstrName.REMOVE_QUBIT:
            self.remove_qubit(instr.qubits[0], instr_idx)
        else:
            raise ValueError(f'Invalid instruction name: {instr.name}')

    def _check_instr_execution_valid(self, instr: Instruction, silent: bool = False) -> bool:
        if instr.name in [InstrName.CX, InstrName.CZ, InstrName.SWAP]:
            return self._check_connected_qubits(instr.qubits[0], instr.qubits[1], silent=silent) and self._check_contiguous_qubit(instr.qubits[0], silent=silent) and self._check_contiguous_qubit(instr.qubits[1], silent=silent)
        elif instr.name == InstrName.INSTANTIATE:
            return self._check_readout_valid(instr.dots, silent=silent) and self._check_init(instr.dots, silent=silent)
        elif instr.name == InstrName.MEASURE:
            return self._check_readout_valid(instr.dots, silent=silent) and self._check_contiguous_qubit(instr.qubits[0], silent=silent)
        elif instr.name == InstrName.H:
            return self._check_contiguous_qubit(instr.qubits[0], silent=silent)
        elif instr.name == InstrName.SHUTTLE_DOT:
            return self._check_shuttle_dot(instr.dots[0], instr.dots[1], silent=silent)
        elif instr.name == InstrName.REMOVE_QUBIT:
            return True
        else:
            raise ValueError(f'Invalid instruction name: {instr.name}')
 
    def _append_instruction(
            self,
            instr: Instruction,
            qubits_relax_dag_order: list[int] = [],
            readout_enforce_ordering: bool = False,
            dots_enforce_ordering: bool = False,
        ):
        """Append an instruction to the schedule.

        Args:
            instr: Instruction to append.
            qubits_relax_dag_order: List of qubit indices for which the
                instruction can be executed before the previous instruction that
                uses the qubit.
            dots_enforce_ordering: If True, instruction has dependencies on the
                previous instructions that use the same dots.
        """
        self._check_instr_execution_valid(instr)
        self.schedule.append(instr)
        instr_idx = len(self.schedule)-1
        self.schedule_dag.add_node(instr_idx)
        for i,qubit in enumerate(instr.qubits):
            for spin in instr.spins[i]:
                key = (0, qubit, spin)
                if qubit not in qubits_relax_dag_order and key in self._instructions_by_target:
                    self.schedule_dag.add_edge(self._instructions_by_target[key][-1], instr_idx)
                self._instructions_by_target.setdefault(key, []).append(instr_idx)
            # if qubit not in qubits_relax_dag_order and qubit in self._instructions_by_target:
            #     self.schedule_dag.add_edge(self._instructions_by_target[qubit][-1], instr_idx)
            # self._instructions_by_target.setdefault(qubit, []).append(instr_idx)
        for readout in instr.readouts:
            frontier_idx = readout + len(self._qubits)
            if readout_enforce_ordering and frontier_idx in self._instructions_by_target:
                self.schedule_dag.add_edge(self._instructions_by_target[frontier_idx][-1], instr_idx)
            self._instructions_by_target.setdefault(frontier_idx, []).append(instr_idx)
        for dot in instr.dots:
            key = (1, *dot)
            if dots_enforce_ordering and key in self._instructions_by_target:
                for prev_instr_idx in self._instructions_by_target[key]:
                    self.schedule_dag.add_edge(prev_instr_idx, instr_idx)
            self._instructions_by_target.setdefault(key, []).append(instr_idx)
        if self.save_animation_frames:
            self.save_animation_frame()

    def h(self, qubit: int, instr_idx: int | None = None):
        """Add a Hadamard gate to the schedule."""
        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.H,
                    [qubit],
                    [[0,1,2]],
                    [],
                    list(self.qubit_coords(qubit)),
                    self.hardware_params.single_qubit_duration,
                    self.hardware_params.single_qubit_err
                )
            )

    def cx(self, qubit0: int, qubit1: int, instr_idx: int | None = None, qubit0_relax_scheduling: bool = False, qubit1_relax_scheduling: bool = False):
        """Add a CX gate to the schedule."""
        # Must be either parallel or linear
        is_parallel = np.all(np.linalg.norm(np.array(self.qubit_coords(qubit0)) - np.array(self.qubit_coords(qubit1)), axis=1) == 1)
        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.CX,
                    [qubit0, qubit1],
                    [[0,1,2], [0,1,2]],
                    [],
                    list(self.qubit_coords(qubit0))+list(self.qubit_coords(qubit1)),
                    self.hardware_params.cx_duration_parallel if is_parallel else self.hardware_params.cx_duration_linear,
                    self.hardware_params.cx_err
                ),
                [qubit0] if qubit0_relax_scheduling else [] + [qubit1] if qubit1_relax_scheduling else [],
            )

    def cz(self, qubit0: int, qubit1: int, instr_idx: int | None = None, qubit0_relax_scheduling: bool = False, qubit1_relax_scheduling: bool = False):
        """Add a CZ gate to the schedule."""
        # Must be either parallel or linear
        is_parallel = np.all(np.linalg.norm(np.array(self.qubit_coords(qubit0)) - np.array(self.qubit_coords(qubit1)), axis=1) == 1)
        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.CZ,
                    [qubit0, qubit1],
                    [[0,1,2], [0,1,2]],
                    [],
                    list(self.qubit_coords(qubit0))+list(self.qubit_coords(qubit1)),
                    self.hardware_params.cx_duration_parallel if is_parallel else self.hardware_params.cx_duration_linear,
                    self.hardware_params.cx_err
                ),
                [qubit0] if qubit0_relax_scheduling else [] + [qubit1] if qubit1_relax_scheduling else [],
            )

    def swap(self, qubit0: int, qubit1: int, instr_idx: int | None = None):
        """Add a SWAP gate to the schedule."""
        is_parallel = np.all(np.linalg.norm(np.array(self.qubit_coords(qubit0)) - np.array(self.qubit_coords(qubit1)), axis=1) == 1)
        
        for dot0,dot1 in zip(self.qubit_coords(qubit0), self.qubit_coords(qubit1)):
            self._qubit_array[dot0] = qubit1
            self._qubit_array[dot1] = qubit0
        qubit0_coords = self.qubit_coords(qubit0)
        self._qubits[qubit0] = self.qubit_coords(qubit1)
        self._qubits[qubit1] = qubit0_coords
        
        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.SWAP,
                    [qubit0, qubit1],
                    [[0,1,2], [0,1,2]],
                    [],
                    list(self.qubit_coords(qubit0))+list(self.qubit_coords(qubit1)),
                    self.hardware_params.swap_duration_parallel if is_parallel else self.hardware_params.swap_duration_linear,
                    self.hardware_params.swap_err
                )
            )

    # TODO: intermediate dot shuttles should *not* enforce qubit instruction
    # dependency (but should enforce *spin* instruction dependency). Need to
    # re-do the dependency graph to account for this.
    def shuttle_follow_path(self, qubit: int, path: list[tuple[int, int]], enforce_contiguous: bool = False, dots_enforce_ordering: bool = False):
        """Shuttle a qubit along a path. Assumes qubit is being shuttled
        on-axis.

        Args:
            qubit: Qubit index.
            path: List of dot coordinates to shuttle along.
            enforce_contiguous: Whether to enforce that the qubit is contiguous
                along the path, to prevent unexpected errors. Disable at your
                own risk.
        """
        self._check_contiguous_qubit(qubit)
        for dot in path:
            if dot[0] < 0 or dot[0] >= self._qubit_array.shape[0] or dot[1] < 0 or dot[1] >= self._qubit_array.shape[1]:
                raise ValueError(f'Shuttle path goes out of bounds. Path: {path}')
            if not (self._qubit_array[dot[0], dot[1]] == -1 or self._qubit_array[dot[0], dot[1]] == qubit):
                raise ValueError(f'Shuttle path intersects with another qubit. Path: {path}')
        if not all (dot in path for dot in self.qubit_coords(qubit)):
            raise ValueError(f'The qubit must be in the path. Path: {path}, qubit: {self.qubit_coords(qubit)}')
         
        for idx in range(1, len(path)-2):
            dots = path[idx-1:idx+3]
            if set(dots[:3]) != set(self.qubit_coords(qubit)):
                raise ValueError(f'Qubit must be contiguous along the path. Path: {path}, qubit: {self.qubit_coords(qubit)}')
            self.shuttle_single_dot(dots[2], dots[3], dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot(dots[1], dots[2], dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot(dots[0], dots[1], dots_enforce_ordering=dots_enforce_ordering)
            self._check_contiguous_qubit(qubit)

        self._check_contiguous_qubit(qubit)

    def shuttle_on_axis(self, qubit: int, num_dots: int, instr_idx: int | None = None, dots_enforce_ordering: bool = False):
        """
        
        Args:
            qubit: Qubit index.
            num_dots: Number of dots to shuttle. If positive, shuttle down or to
                the right (depending on the qubit's orientation). If negative,
                shuttle up or to the left.
        """
        self._check_contiguous_qubit(qubit)
        row_oriented = self.qubit_coords(qubit)[0][0] == self.qubit_coords(qubit)[1][0]
        positive_dir = num_dots > 0
        init_dots = sorted(list(self.qubit_coords(qubit)))
        dots_to_use = []
        if not positive_dir:
            dots_to_use = dots_to_use[::-1]
        for d in range(abs(num_dots)+3):
            if row_oriented:
                if positive_dir:
                    dots_to_use.append((init_dots[0][0], init_dots[0][1]+d))
                else:
                    dots_to_use.append((init_dots[2][0], init_dots[2][1]-d))
            else:
                if positive_dir:
                    dots_to_use.append((init_dots[0][0]+d, init_dots[0][1]))
                else:
                    dots_to_use.append((init_dots[2][0]-d, init_dots[2][1]))
        self.shuttle_follow_path(qubit, dots_to_use, dots_enforce_ordering=dots_enforce_ordering)

    def shuttle_off_axis(self, qubit: int, num_dots: int, dots_enforce_ordering: bool = False):
        """Shuttle a qubit 

        Args:
            qubit: Qubit index.
            num_dots: Number of dots to shuttle. If positive, shuttle down or to
                the right (depending on the qubit's orientation). If negative,
                shuttle up or to the left.
        """
        self._check_contiguous_qubit(qubit)
        init_dots = list(self.qubit_coords(qubit))
        row_oriented = init_dots[0][0] == init_dots[1][0]
        positive_dir = num_dots > 0
        for i in range(3):
            for d in range(1, abs(num_dots)+1):
                if row_oriented:
                    if positive_dir:
                        start_dot = (init_dots[i][0]+d-1, init_dots[i][1])
                        end_dot = (init_dots[i][0]+d, init_dots[i][1])
                    else:
                        start_dot = (init_dots[i][0]-d+1, init_dots[i][1])
                        end_dot = (init_dots[i][0]-d, init_dots[i][1])
                    self.shuttle_single_dot(start_dot, end_dot, dots_enforce_ordering=dots_enforce_ordering)
                else:
                    if positive_dir:
                        start_dot = (init_dots[i][0], init_dots[i][1]+d-1)
                        end_dot = (init_dots[i][0], init_dots[i][1]+d)
                    else:
                        start_dot = (init_dots[i][0], init_dots[i][1]-d+1)
                        end_dot = (init_dots[i][0], init_dots[i][1]-d)
                    self.shuttle_single_dot(start_dot, end_dot, dots_enforce_ordering=dots_enforce_ordering)

    def shuttle_single_dot(self, dot: tuple[int, int], final_dot: tuple[int, int], instr_idx: int | None = None, dots_enforce_ordering: bool = False):
        """Shuttle a single dot by one unit.
        
        Args:
            dot: Dot to shuttle.
            final_dot: Destination dot.
            instr_idx: If provided, do not append the instruction to the
                schedule.
            dots_enforce_ordering: If True, instruction has dependencies on the
                previous instructions that use the same dots.
        """
        q_idx = self._qubit_array[dot]
        if q_idx == -1:
            raise ValueError(f'Attempted to move {dot} to {final_dot}, but start dot is empty.' + (f' Instr. {instr_idx}.' if instr_idx is not None else ''))
        if self._qubit_array[final_dot] != -1:
            raise ValueError(f'Attempted to move {dot} (qubit {q_idx}) to {final_dot}, but final dot is already occupied by qubit {self._qubit_array[final_dot]}.' + (f' Instr. {instr_idx}.' if instr_idx is not None else ''))
        if np.linalg.norm(np.array(dot) - np.array(final_dot)) != 1:
            raise ValueError(f'Attempted to move {dot} to {final_dot}, but they are not adjacent.' + (f' Instr. {instr_idx}.' if instr_idx is not None else ''))

        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.SHUTTLE_DOT,
                    [q_idx],
                    [[self._qubits[q_idx].index(dot)]],
                    [],
                    [dot, final_dot],
                    self.hardware_params.shuttle_duration,
                    self.hardware_params.shuttle_err
                ),
                dots_enforce_ordering=dots_enforce_ordering,
            )
        
        self._qubit_array[final_dot] = self._qubit_array[dot]
        self._qubit_array[dot] = -1
        self._qubits[q_idx] = tuple([final_dot if d == dot else d for d in self.qubit_coords(q_idx)])

    def measure(self, qubit: int, instr_idx: int | None = None, readout_enforce_ordering: bool = False):
        """Add a measurement instruction to the schedule."""
        chosen_readout = self._get_readout_idx(list(self.qubit_coords(qubit)))

        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.MEASURE,
                    [qubit],
                    [[0,1,2]],
                    [chosen_readout],
                    list(self.qubit_coords(qubit)),
                    self.hardware_params.measure_duration,
                    self.hardware_params.measure_err,
                ),
                readout_enforce_ordering=readout_enforce_ordering,
            )

    def remove_qubit(self, qubit: int, instr_idx: int | None = None):
        """Remove a qubit from the device."""
        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.REMOVE_QUBIT,
                    [qubit],
                    [[0,1,2]],
                    [],
                    list(self.qubit_coords(qubit)),
                    0,
                    0,
                )
            )
        
        self._qubit_array[self.qubit_coords(qubit)[0]] = -1
        self._qubit_array[self.qubit_coords(qubit)[1]] = -1
        self._qubit_array[self.qubit_coords(qubit)[2]] = -1
        self._qubits[qubit] = None

    def instantiate(self, qubit: int, dots: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], instr_idx: int | None = None, readout_enforce_ordering: bool = False):
        """Place a qubit on the device.
        
        Args:
            qubit: Qubit index. Must not already exist on the device.
            dots: 3-tuple of empty dot coordinates.
        """
        if self._qubits[qubit]:
            raise ValueError(f'Tried to instantiate qubit {qubit} on dots {dots}, but qubit already exists on the device.' + (f' Instr. {instr_idx}.' if instr_idx is not None else ''))
        if self._qubit_array[dots[0]] != -1 or self._qubit_array[dots[1]] != -1 or self._qubit_array[dots[2]] != -1:
            raise ValueError(f'Tried to instantiate qubit {qubit} on dots {dots}, but one or more dots are already occupied.' + (f' Instr. {instr_idx}.' if instr_idx is not None else ''))
        
        chosen_readout = self._get_readout_idx(list(dots))

        if instr_idx is None:
            self._append_instruction(Instruction(
                    InstrName.INSTANTIATE,
                    [qubit],
                    [[0,1,2]],
                    [chosen_readout],
                    list(dots),
                    self.hardware_params.measure_duration,
                    self.hardware_params.measure_err,
                ),
                readout_enforce_ordering=readout_enforce_ordering,
            )

        self._qubit_array[dots[0]] = qubit
        self._qubit_array[dots[1]] = qubit
        self._qubit_array[dots[2]] = qubit
        self._qubits[qubit] = dots

    def simple_rotation(self, qubit: int, dots_enforce_ordering: bool = False):
        """Rotate a qubit from horizontal to vertical or vice versa."""
        self._check_contiguous_qubit(qubit)
        dots = self.qubit_coords(qubit)
        is_horizontal = dots[0][0] == dots[1][0] == dots[2][0]
        is_vertical = dots[0][1] == dots[1][1] == dots[2][1]
        if not (is_horizontal or is_vertical):
            raise ValueError('Qubit is not oriented correctly.')
        if is_horizontal:
            self._qubits[qubit] = tuple(sorted(dots))
            self.shuttle_single_dot(dots[0], (dots[0][0]-1, dots[0][1]), dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot((dots[0][0]-1, dots[0][1]), (dots[0][0]-1, dots[0][1]+1), dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot(dots[2], (dots[2][0]+1, dots[2][1]), dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot((dots[2][0]+1, dots[2][1]), (dots[2][0]+1, dots[2][1]-1), dots_enforce_ordering=dots_enforce_ordering)
        else:
            self._qubits[qubit] = tuple(sorted(dots, key=lambda x: x[1]))
            self.shuttle_single_dot(dots[0], (dots[0][0], dots[0][1]-1), dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot((dots[0][0], dots[0][1]-1), (dots[0][0]+1, dots[0][1]-1), dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot(dots[2], (dots[2][0], dots[2][1]+1), dots_enforce_ordering=dots_enforce_ordering)
            self.shuttle_single_dot((dots[2][0], dots[2][1]+1), (dots[2][0]-1, dots[2][1]+1), dots_enforce_ordering=dots_enforce_ordering)

    def plot_snapshot(
            self,
            ax: plt.Axes | None = None,
            label_qubits: bool = True,
            label_readouts: bool = True,
            markersize=18,
            linewidth=5,
            fig_width=10,
            readout_label_h_offset=1.5,
            readout_label_v_offset=1,
            instruction_indices_to_show: list[int] = [],
            frame_idx: int | None = None,
        ):
        """Plot a snapshot of the device."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        
        for i in range(self._qubit_array.shape[0]):
            for j in range(self._qubit_array.shape[1]):
                q_idx = self._qubit_array[i, j]
                if q_idx == -1:
                    ax.plot(j, i, 'o', markersize=markersize, markerfacecolor='none', markeredgecolor='gray', zorder=2)
                else:
                    if q_idx in self.data_indices:
                        color = 'k'
                    elif q_idx in self.X_check_indices:
                        color = 'cornflowerblue'
                    elif q_idx in self.Z_check_indices:
                        color = 'firebrick'
                    else:
                        color = 'yellow'
                    ax.plot(j, i, 'o', markersize=markersize, color=color, zorder=4)
                    if label_qubits and (i == 0 or self._qubit_array[i-1, j] != q_idx) and (j == 0 or self._qubit_array[i, j-1] != q_idx):
                        # r,g,b = cmap(norm(q_idx))[:3]
                        # textcolor = 'k' if (r*0.299 + g*0.587 + b*0.114)*256 > 186 else 'w'
                        ax.text(j, i, f'q{q_idx}', ha='center', va='center', color='w', zorder=5)
                    if i > 0 and self._qubit_array[i-1, j] == q_idx:
                        ax.plot([j, j], [i-1, i], '-', zorder=3.9, color=color, linewidth=linewidth)
                    if j > 0 and self._qubit_array[i, j-1] == q_idx:
                        ax.plot([j-1, j], [i, i], '-', zorder=3.9, color=color, linewidth=linewidth)
        cmap = mpl.colormaps['plasma']
        norm = mpl.colors.Normalize(vmin=0, vmax=len(self.readout_devices))
        for readout_idx, coords in enumerate(self.readout_devices):
            bounding_box = (min([dot[1] for dot in coords])-0.5, max([dot[1] for dot in coords])+0.5, min([dot[0] for dot in coords])-0.5, max([dot[0] for dot in coords])+0.5)
            for dot in coords:
                ax.add_patch(mpl.patches.Rectangle((dot[1]-0.5, dot[0]-0.5), 1, 1, color=cmap(norm(readout_idx)), alpha=0.2, zorder=3))#, edgecolor='none'))
            ax.add_patch(mpl.patches.Rectangle((bounding_box[0], bounding_box[2]), bounding_box[1]-bounding_box[0], bounding_box[3]-bounding_box[2], facecolor='none', edgecolor='black', zorder=3))
            if label_readouts:
                center_coords = (np.mean([dot[0] for dot in coords]), np.mean([dot[1] for dot in coords]))
                # one of the coords is 0.5 away from the last dot
                if center_coords[1] == 0.5:
                    nearest_coords_off_array = (center_coords[0], -readout_label_h_offset)
                    bracket_box = (-0.8, -0.6, bounding_box[2]+0.2, bounding_box[3]-0.2)
                elif center_coords[1] == self._qubit_array.shape[1]-1.5:
                    nearest_coords_off_array = (center_coords[0], self._qubit_array.shape[1]-1+readout_label_h_offset)
                    bracket_box = (bounding_box[1]+0.6-0.5, bounding_box[1]+0.8-0.5, bounding_box[2]+0.2, bounding_box[3]-0.2)
                elif center_coords[0] == 0.5:
                    nearest_coords_off_array = (-readout_label_h_offset, center_coords[1])
                    bracket_box = (bounding_box[0]+0.2, bounding_box[1]-0.2, -0.8, -0.6)
                elif center_coords[0] == self._qubit_array.shape[0]-1.5:
                    nearest_coords_off_array = (self._qubit_array.shape[0]-1+readout_label_h_offset, center_coords[1])
                    bracket_box = (bounding_box[0]+0.2, bounding_box[1]-0.2, bounding_box[3]+0.6-0.5, bounding_box[3]+0.8-0.5)
                else:
                    raise ValueError('Readout device is not on the edge of the array??')
                ax.text(nearest_coords_off_array[1], nearest_coords_off_array[0], f'r{readout_idx}', ha='center', va='center', zorder=5)
                ax.add_patch(mpl.patches.Rectangle((bracket_box[0], bracket_box[2]), bracket_box[1]-bracket_box[0], bracket_box[3]-bracket_box[2], color=cmap(norm(readout_idx)), zorder=3))

        for instr_idx in instruction_indices_to_show:
            instr = self.schedule[instr_idx]
            if instr.name == InstrName.CX or instr.name == InstrName.CZ:
                assert len(instr.qubits) == 2
                for dot in instr.dots:
                    if self._qubit_array[*dot] == -1:
                        continue
                    ax.plot(dot[1], dot[0], 'o', markersize=markersize*1.1, markerfacecolor='none', markeredgecolor='orange', markeredgewidth=linewidth, zorder=6.1)
                    if dot[0] > 0 and self._qubit_array[dot[0]-1, dot[1]] == instr.qubits[0] or self._qubit_array[dot[0]-1, dot[1]] == instr.qubits[1]:
                        ax.plot([dot[1], dot[1]], [dot[0]-1, dot[0]], '-', color='orange', linewidth=linewidth, zorder=3.95)
                    if dot[1] > 0 and self._qubit_array[dot[0], dot[1]-1] == instr.qubits[0] or self._qubit_array[dot[0], dot[1]-1] == instr.qubits[1]:
                        ax.plot([dot[1]-1, dot[1]], [dot[0], dot[0]], '-', color='orange', linewidth=linewidth, zorder=3.95)
            elif instr.name == InstrName.SWAP:
                assert len(instr.qubits) == 2
                for dot in instr.dots:
                    if self._qubit_array[*dot] == -1:
                        continue
                    ax.plot(dot[1], dot[0], 'o', markersize=markersize*1.1, markerfacecolor='none', markeredgecolor='gold', markeredgewidth=linewidth, zorder=6.1)
                    if dot[0] > 0 and self._qubit_array[dot[0]-1, dot[1]] == instr.qubits[0] or self._qubit_array[dot[0]-1, dot[1]] == instr.qubits[1]:
                        ax.plot([dot[1], dot[1]], [dot[0]-1, dot[0]], '-', color='orange', linewidth=linewidth, zorder=3.95)
                    if dot[1] > 0 and self._qubit_array[dot[0], dot[1]-1] == instr.qubits[0] or self._qubit_array[dot[0], dot[1]-1] == instr.qubits[1]:
                        ax.plot([dot[1]-1, dot[1]], [dot[0], dot[0]], '-', color='orange', linewidth=linewidth, zorder=3.95)
            elif instr.name == InstrName.H:
                assert len(instr.qubits) == 1
                for dot in instr.dots:
                    if self._qubit_array[*dot] == -1:
                        continue
                    ax.plot(dot[1], dot[0], 'o', markersize=markersize*1.1, markerfacecolor='none', markeredgecolor='cyan', markeredgewidth=linewidth, zorder=6.1)
                    if dot[0] > 0 and self._qubit_array[dot[0]-1, dot[1]] == instr.qubits[0]:
                        ax.plot([dot[1], dot[1]], [dot[0]-1, dot[0]], '-', color='cyan', linewidth=linewidth, zorder=3.95)
                    if dot[1] > 0 and self._qubit_array[dot[0], dot[1]-1] == instr.qubits[0]:
                        ax.plot([dot[1]-1, dot[1]], [dot[0], dot[0]], '-', color='cyan', linewidth=linewidth, zorder=3.95)
            elif instr.name == InstrName.SHUTTLE_DOT:
                assert len(instr.qubits) == 1
                init_dot, final_dot = instr.dots
                ax.plot(final_dot[1], final_dot[0], 'o', markersize=markersize*1.1, markerfacecolor='none', markeredgecolor='forestgreen', markeredgewidth=linewidth, zorder=6)
                ax.plot([init_dot[1], final_dot[1]], [init_dot[0], final_dot[0]], '-', color='forestgreen', linewidth=linewidth, zorder=2)
            elif instr.name == InstrName.MEASURE or instr.name == InstrName.INSTANTIATE:
                assert len(instr.qubits) == 1 and len(instr.readouts) == 1
                readout_idx, = instr.readouts
                coords = self.readout_devices[readout_idx]
                bounding_box = (min([dot[1] for dot in coords])-0.5, max([dot[1] for dot in coords])+0.5, min([dot[0] for dot in coords])-0.5, max([dot[0] for dot in coords])+0.5)
                ax.add_patch(mpl.patches.Rectangle((bounding_box[0], bounding_box[2]), bounding_box[1]-bounding_box[0], bounding_box[3]-bounding_box[2], facecolor='none', edgecolor='m', linewidth=linewidth, zorder=3))
                for dot in instr.dots:
                    ax.plot(dot[1], dot[0], 'o', markersize=markersize*1.1, markerfacecolor='none', markeredgecolor='m', markeredgewidth=linewidth, zorder=6.2)

        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.invert_yaxis()

        if frame_idx is not None:
            ax.text(0, -1, f'{frame_idx:03d}', ha='center', va='top', fontsize=12, zorder=5)

        return ax
    
    def save_animation_frame(self, instruction_indices_to_show: list[int] = [], frame_num: int = None, **plot_kwargs):
        frame_idx = frame_num if frame_num is not None else self._frame_idx
        if instruction_indices_to_show == []:
            instruction_indices_to_show = [-1] if len(self.schedule) > 0 else []
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = self.plot_snapshot(ax, instruction_indices_to_show=instruction_indices_to_show, frame_idx=frame_idx, **plot_kwargs)
        plt.savefig(f'figures/anim/frame{frame_idx:06d}.png', bbox_inches='tight')
        plt.close()
        self._frame_idx += 1

    def compile_schedule(self, schedule, schedule_dag, animation_frame_interval: int = 0, col_conflict_policies: dict[int, str] = {}, rng: np.random.Generator = np.random.default_rng(), **animation_frame_kwargs):
        self.schedule = schedule
        self.schedule_dag = schedule_dag
        dt = math.gcd(*[instr.duration for instr in self.schedule])
        instr_frontier = set(next(nx.topological_generations(self.schedule_dag)))
        completed_instructions = set()
        completed_instruction_count = 0
        active_instructions = dict()
        active_qubits = set()
        active_dots = set()
        active_readouts = set()
        tlist = []
        instructions_by_ns = []
        most_recently_occupied = np.full(self._qubit_array.shape, -1, dtype=int)
        most_recently_occupied_delay = np.full(self._qubit_array.shape, -1, dtype=int)
        while completed_instruction_count < len(self.schedule):
            # tick down occupation delay
            most_recently_occupied_delay -= (most_recently_occupied_delay > 0).astype(int)

            completed_instructions_this_round = set()
            for instr_idx, remaining_ns in active_instructions.items():
                if remaining_ns <= dt:
                    completed_instructions_this_round.add(instr_idx)
                else:
                    active_instructions[instr_idx] -= dt
            for instr_idx in completed_instructions_this_round:
                active_instructions.pop(instr_idx)
                active_qubits -= set(self.schedule[instr_idx].qubits)
                active_dots -= set(self.schedule[instr_idx].dots)
                active_readouts -= set(self.schedule[instr_idx].readouts)
                assert instr_idx not in completed_instructions
                completed_instructions.add(instr_idx)
                completed_instruction_count += 1

            competing_for_dot = dict()
            for instr_idx in instr_frontier:
                if all(pred_idx in completed_instructions for pred_idx in self.schedule_dag.predecessors(instr_idx)):
                    instr = self.schedule[instr_idx]
                    if instr.name == InstrName.SHUTTLE_DOT:
                        competing_for_dot.setdefault(instr.dots[1], set()).add(instr_idx)

            # when shuttling, if multiple qubits want to shuttle into the
            # same dot, we give priority to the qubit that most recently
            # occupied that dot. This priority lasts for one round after the
            # qubit has left.
            prohibited_instrs = set()
            for dot, competing_instrs in competing_for_dot.items():
                if len(competing_instrs) > 1:
                    qubits = [self.schedule[instr_idx].qubits[0] for instr_idx in competing_instrs]
                    if most_recently_occupied_delay[dot] > 0 or most_recently_occupied[dot] in qubits:
                        # reserved for most_recently_occupied
                        others = {instr_idx for instr_idx in competing_instrs if self.schedule[instr_idx].qubits[0] != most_recently_occupied[dot]}
                        prohibited_instrs |= others
                    elif dot[1] in col_conflict_policies:
                        # use col_conflict_policies
                        policy = col_conflict_policies[dot[1]]
                        chosen_instr = None
                        for instr_idx in competing_instrs:
                            instr = self.schedule[instr_idx]
                            if instr.dots[0][0 if policy == 'row' else 1] == dot[0 if policy == 'row' else 1]:
                                chosen_instr = instr_idx
                                break
                        if chosen_instr is not None:
                            others = competing_instrs - {chosen_instr}
                            prohibited_instrs |= others

            if tlist and tlist[-1] >= 500:
                pass

            activated_instrs = []
            for instr_idx in rng.permutation(list(instr_frontier)): # TODO: randomly sort?
                if instr_idx < 7:
                    pass
                if instr_idx not in prohibited_instrs and all(pred_idx in completed_instructions for pred_idx in self.schedule_dag.predecessors(instr_idx)) and self._check_instr_execution_valid(self.schedule[instr_idx], silent=True):
                    instr = self.schedule[instr_idx]
                    if (#any(q in active_qubits for q in instr.qubits) or
                        any(d in active_dots for d in instr.dots) or
                        any(r in active_readouts for r in instr.readouts)
                       ):
                        continue
                    if self.schedule[instr_idx].name == InstrName.SHUTTLE_DOT:
                        most_recently_occupied[self.schedule[instr_idx].dots[1]] = self.schedule[instr_idx].qubits[0]
                        # most_recently_occupied_delay[self.schedule[instr_idx].dots[1]] = 1
                    active_instructions[instr_idx] = self.schedule[instr_idx].duration
                    active_qubits |= set(instr.qubits)
                    active_dots |= set(instr.dots)
                    active_readouts |= set(instr.readouts)
                    activated_instrs.append(instr_idx)
            for instr_idx in activated_instrs:
                self._execute_instruction(instr_idx)
                instr_frontier.remove(instr_idx)
                instr_frontier |= (set(self.schedule_dag.successors(instr_idx)) - completed_instructions - active_instructions.keys())
            assert instr_frontier.intersection(completed_instructions) == set()
            assert instr_frontier.intersection(active_instructions.keys()) == set()

            if len(tlist) > 0:
                tlist.append(tlist[-1]+dt)
            else:
                tlist.append(0)
            instructions_by_ns.append(list(active_instructions.keys()))

            # if tlist[-1] < 600:
            # # if tlist[-1] > 4000:
            # #     print('Schedule took too long to compile. Aborting.')
            #     pending = list(sorted(list(instr_frontier)))
            #     print(f'{tlist[-1]}. Pending instructions:', pending)
            #     print(prohibited_instrs)
            #     print([all(pred_idx in completed_instructions for pred_idx in self.schedule_dag.predecessors(instr_idx)) for instr_idx in pending])
            #     print([self._check_instr_execution_valid(self.schedule[instr_idx], silent=True) for instr_idx in pending])
            #     # raise ValueError('Schedule took too long to compile.
            #     # Aborting.')
            # else:
            #     raise ValueError()

            if animation_frame_interval > 0 and tlist[-1] % animation_frame_interval == 0:
                self.save_animation_frame(list(active_instructions.keys()), tlist[-1], **animation_frame_kwargs)
 
        if animation_frame_interval > 0:
            self.save_animation_frame([], tlist[-1], **animation_frame_kwargs)

        return tlist, instructions_by_ns