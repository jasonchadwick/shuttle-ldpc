from dataclasses import dataclass, field, asdict
from typing import Any
from enum import Enum
import numpy as np
import hashlib
import math
import pickle
from pathlib import Path

import stim
import heapq
import itertools
from copy import copy, deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from python_tsp.exact import solve_tsp_dynamic_programming, solve_tsp_brute_force, solve_tsp_branch_and_bound
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
from src.QECCode import QECCode

MAX_T = 10**10

@dataclass
class HardwareParams:
    cx_duration: int # in ns
    h_duration: int # in ns
    shuttle_duration: int # per unit cell shuttle, in ns
    emplace_duration: int
    init_duration: int
    measure_duration: int

@dataclass
class ErrorParams:
    T2: float # in seconds
    cx_err: float
    h_err: float
    shuttle_err: float # per unit cell shuttle
    emplace_err: float
    init_err: float
    measure_err: float

default_hwp = HardwareParams(
    cx_duration=100,
    h_duration=100,
    shuttle_duration=1000, # 10 m/s
    emplace_duration=100,
    init_duration=500,
    measure_duration=500,
)

def error_params(
        p: float = 1e-3,
        T2: float = 100e-6,
        p_sh: float = 1e-5,
    ) -> ErrorParams:
    return ErrorParams(
        T2=T2,
        cx_err=p,
        h_err=p,
        shuttle_err=p_sh,
        emplace_err=p_sh/10,
        init_err=p,
        measure_err=p,
    )

@dataclass
class Instruction:
    duration: int

    def qubits_list(self) -> list[int]:
        raise NotImplementedError

    def __str__(self):
        string = self.__class__.__name__ + '('
        for key,val in self.__dict__.items():
            if isinstance(val, float):
                string += f'{key}={val:3g}, '
            else:
                string += f'{key}={val}, '
        string = string[:-2] + ')'
        return string

@dataclass
class MultiQubitInstruction(Instruction):
    qubits: list[int]

    def qubits_list(self) -> list[int]:
        return self.qubits

@dataclass
class SingleQubitInstruction(Instruction):
    qubit: int

    def qubits_list(self) -> list[int]:
        return [self.qubit]

class GateName(Enum):
    CX = 'CX'
    CZ = 'CZ'
    H = 'H'

@dataclass
class Gate(MultiQubitInstruction):
    name: GateName

@dataclass
class Instantiate(MultiQubitInstruction):
    cell_coords: list[tuple[int, int]]

@dataclass
class Measure(MultiQubitInstruction):
    cell_coords: list[tuple[int, int]]

@dataclass
class Shuttle(SingleQubitInstruction):
    start_coords: tuple[int, int]
    end_coords: tuple[int, int]

class DeviceComponent(Enum):
    READOUT = 0
    INTERACTION_ZONE = 1
    SHUTTLE_CHANNEL = 2
    SHUTTLE_INTERSECTION = 3
    NONEXISTENT = 4

    def __str__(self):
        return self.name
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __gt__(self, other):
        return self.value > other.value

@dataclass
class DeviceLocation:
    component: DeviceComponent
    coords: tuple[float, float] | tuple[int, int]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

@dataclass
class EmplaceDisplace(SingleQubitInstruction):
    # Move a qubit in or out of a readout zone or an interaction zone.
    coords: tuple[int, int]
    start_component: DeviceComponent
    end_component: DeviceComponent

class Direction(Enum):
    UP = 'UP'
    RIGHT = 'RIGHT'
    DOWN = 'DOWN'
    LEFT = 'LEFT'

    @classmethod
    def from_shuttle(cls, instr: Shuttle):
        if instr.end_coords[1] > instr.start_coords[1]:
            return cls.UP
        elif instr.end_coords[1] < instr.start_coords[1]:
            return cls.DOWN
        elif instr.end_coords[0] > instr.start_coords[0]:
            return cls.RIGHT
        else:
            assert instr.end_coords[0] < instr.start_coords[0]
            return cls.LEFT

@dataclass
class Frame:
    t: int
    qubit_positions: dict[int, tuple[float, float]]
    twoq_gates: list[tuple[int, int]]
    oneq_gates: list[int]
    init_qs: list[int]
    meas_qs: list[int]
    active_shuttles: list[tuple[int, Direction]] # (qubit, direction). 0=up, 1=right, 2=down, 3=left

def sort(edge: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    u,v = sorted(edge)
    return (u,v)

class CompiledShuttlingSchedule:
    qubits: list[int]
    data_qubits: list[int]
    instructions: list[Instruction]
    instruction_start_times: list[int]
    instructions_by_qubit: dict[int, list[int]]

    def __init__(self, qubits: list[int], data_qubits: list[int]):
        self.qubits = qubits
        self.data_qubits = data_qubits
        self.instructions = []
        self.instruction_start_times = []
        self.instructions_by_qubit = {q:[] for q in qubits}

    def append_instr(self, instr: Instruction, start_time: int):
        for q in instr.qubits_list():
            self.instructions_by_qubit[q].append(len(self.instructions))
        self.instructions.append(instr)
        self.instruction_start_times.append(start_time)

    def total_duration(self) -> int:
        t_tot = 0
        for instr, t in zip(self.instructions, self.instruction_start_times):
            t_tot = max(t_tot, t + instr.duration)
        return t_tot

    def __add__(self, other):
        offset = self.total_duration()
        if isinstance(other, self.__class__):
            new_sched = CompiledShuttlingSchedule(list(sorted(set(self.qubits) | set(other.qubits))), list(sorted(set(self.data_qubits) | set(other.data_qubits))))
            for instr, t in zip(self.instructions, self.instruction_start_times):
                new_sched.append_instr(instr, t)
            for instr, t in zip(other.instructions, other.instruction_start_times):
                new_sched.append_instr(instr, t + offset)
            return new_sched
        else:
            raise TypeError
    
    def __mul__(self, other):
        if isinstance(other, int):
            new_sched = CompiledShuttlingSchedule(self.qubits, self.data_qubits)
            for _ in range(other):
                new_sched += self
            return new_sched
        else:
            raise TypeError
    
    def __rmul__(self, other):
        if isinstance(other, int):
            new_sched = CompiledShuttlingSchedule(self.qubits, self.data_qubits)
            for _ in range(other):
                new_sched += self
            return new_sched
        else:
            raise TypeError

    def get_frames(self, frame_timing: int = 100) -> list[Frame]:
        """Convert per-qubit schedules to per-frame view, where each qubit"""
        frames = []
        qubit_positions: dict[int, tuple[float, float]] = dict()
        qubit_positions_planned: dict[int, tuple[float, float]] = dict()
        qubits_to_remove: set[int] = set()
        for t in range(-frame_timing, self.total_duration() + 2*frame_timing, frame_timing):
            qubits_moved_in_frame: set[int] = set()
            qubits_measured_in_frame: set[int] = set()
            active_instrs = []
            for i,(t0,instr) in enumerate(zip(self.instruction_start_times, self.instructions)):
                if t0 <= t < t0 + instr.duration:
                    active_instrs.append(i)
            twoq_gates = []
            oneq_gates = []
            init_qs = []
            meas_qs = []
            active_shuttles = []
            for instr_idx in active_instrs:
                instr = self.instructions[instr_idx]
                t0 = self.instruction_start_times[instr_idx]
                if isinstance(instr, SingleQubitInstruction):
                    if isinstance(instr, Shuttle):
                        a = (t - t0) / instr.duration
                        qubit_positions[instr.qubit] = ((1-a)*instr.start_coords[0] + a*instr.end_coords[0], (1-a)*instr.start_coords[1] + a*instr.end_coords[1])
                        qubit_positions_planned[instr.qubit] = instr.end_coords
                        active_shuttles.append((instr.qubit, Direction.from_shuttle(instr)))
                        qubits_moved_in_frame.add(instr.qubit)
                    elif isinstance(instr, EmplaceDisplace):
                        x0,y0 = qubit_positions[instr.qubit]
                        x1,y1 = instr.coords
                        if instr.end_component == DeviceComponent.READOUT:
                            x1 -= 0.4
                            y1 -= 0.2
                        elif instr.end_component == DeviceComponent.INTERACTION_ZONE:
                            x1 -= 0.15
                            y1 -= 0.4
                            if (x1, y1) in qubit_positions.values():
                                # Already a qubit in the interaction zone
                                y1 += 0.2
                        a = (t-t0) / instr.duration
                        qubit_positions[instr.qubit] = ((1-a)*x0 + a*x1, (1-a)*y0 + a*y1)
                        qubit_positions_planned[instr.qubit] = (x1, y1)
                        qubits_moved_in_frame.add(instr.qubit)
                else:
                    assert isinstance(instr, MultiQubitInstruction)
                    if isinstance(instr, Gate):
                        if instr.name == GateName.H:
                            oneq_gates += instr.qubits
                        else:
                            twoq_gates += list(zip(instr.qubits, instr.qubits[1:]))[::2]
                    elif isinstance(instr, Instantiate):
                        init_qs += instr.qubits
                        for q,coords in zip(instr.qubits, instr.cell_coords):
                            qubit_positions[q] = (coords[0] - 0.4, coords[1] - 0.2)
                            qubits_moved_in_frame.add(q)
                    elif isinstance(instr, Measure):
                        meas_qs += instr.qubits
                        qubits_to_remove |= set(instr.qubits)
                        qubits_measured_in_frame |= set(instr.qubits)

            frames.append(Frame(
                t,
                copy(qubit_positions),
                twoq_gates,
                oneq_gates,
                init_qs,
                meas_qs,
                active_shuttles,
            ))

            for q in qubits_to_remove - qubits_measured_in_frame:
                qubit_positions.pop(q)
                qubits_to_remove.remove(q)

            for q,coords in list(qubit_positions_planned.items()):
                if q not in qubit_positions:
                    qubit_positions_planned.pop(q)
                elif q not in qubits_moved_in_frame:
                    qubit_positions[q] = coords
                    qubit_positions_planned.pop(q)
                elif coords == qubit_positions[q]:
                    qubit_positions_planned.pop(q)
            
            if len(set(qubit_positions.values())) != len(qubit_positions):
                print(f'Issue at t={t}')
                for qubit, pos in qubit_positions.items():
                    if pos in [p for q,p in qubit_positions.items() if q != qubit]:
                        print(f'\tOverlapping: {[q for q,p in qubit_positions.items() if p == pos]} at pos {pos}')

        return frames

    def to_stim_circuit(self, code: QECCode, basis: str, error_params: ErrorParams, XYZ_decode: bool = False) -> stim.Circuit:
        """Convert the compiled shuttling schedule back into a Stim circuit."""
        circ = stim.Circuit()

        if hasattr(code, 'qubit_coords'):
            for q,coords in enumerate(code.qubit_coords):
                circ.append('QUBIT_COORDS', q, coords)

        qubit_time_last_used: dict[int, int] = dict()
        meas_counter: dict[int, list[int]] = dict() # meas index of each measurement
        meas_count: int = 0
        instrs_sort = np.argsort(self.instruction_start_times)
        instructions = [self.instructions[i] for i in instrs_sort]
        instruction_start_times = [self.instruction_start_times[i] for i in instrs_sort]
        last_start_time = 0
        init_basis_change = False
        meas_basis_change = False
        for instr_idx, (instr, start_time) in enumerate(zip(instructions, instruction_start_times)):
            # apply idle errors
            if not isinstance(instr, Instantiate) and start_time > last_start_time:
                idle_ops: dict[float, list[int]] = dict()
                for q in instr.qubits_list():
                    if qubit_time_last_used[q] < start_time:
                        idle_time = start_time - qubit_time_last_used[q]
                        idle_err = 1 - np.exp(-idle_time*1e-9 / error_params.T2)
                        idle_ops.setdefault(idle_err, []).append(q)
                for idle_err, qs in idle_ops.items():
                    circ.append('Z_ERROR', qs, min(0.75, idle_err))
                last_start_time = start_time

            # apply instruction
            if isinstance(instr, Gate):
                if instr.name == GateName.CX:
                    circ.append('CX', instr.qubits, ())
                    circ.append('DEPOLARIZE2', instr.qubits, min(15/16, error_params.cx_err))
                elif instr.name == GateName.H:
                    circ.append('H', instr.qubits, ())
                    circ.append('DEPOLARIZE1', instr.qubits, min(0.75, error_params.h_err))
                else:
                    raise NotImplementedError
            elif isinstance(instr, Instantiate):
                circ.append('R', instr.qubits, ())
                circ.append('X_ERROR', instr.qubits, error_params.init_err)
                if instr_idx == 0:
                    if basis == 'X' and set(instr.qubits) == set(code.data_indices):
                        instr0 = self.instructions[self.instructions_by_qubit[code.data_indices[0]][0]]
                        instr1 = self.instructions[self.instructions_by_qubit[code.data_indices[0]][1]]
                        assert instr0 == instr
                        if not (isinstance(instr1, Gate) and instr1.name == 'H'):
                            init_basis_change = True
                    if init_basis_change:
                        # print('Changing data qubit initialization to X basis...')
                        circ.append('H', instr.qubits, ())
                        circ.append('DEPOLARIZE1', instr.qubits, error_params.h_err)
            elif isinstance(instr, Measure):
                if instr_idx == len(self.instructions)-1:
                    if basis == 'X' and set(instr.qubits) == set(code.data_indices):
                        instr2 = self.instructions[self.instructions_by_qubit[code.data_indices[0]][-2]]
                        instr1 = self.instructions[self.instructions_by_qubit[code.data_indices[0]][-1]]
                        assert instr1 == instr
                        if not (isinstance(instr2, Gate) and instr2.name == 'H'):
                            meas_basis_change = True
                    if meas_basis_change:
                        # print('Changing data qubit measurement to X basis...')
                        circ.append('H', instr.qubits, ())
                        circ.append('DEPOLARIZE1', instr.qubits, error_params.h_err)

                circ.append('X_ERROR', instr.qubits, error_params.measure_err)
                circ.append('M', instr.qubits, ())
                for q in instr.qubits:
                    meas_counter.setdefault(q, []).append(meas_count)
                    meas_count += 1
                # Add detectors if ancilla qubit
                for q in instr.qubits:
                    if q in code.X_ancilla_indices + code.Z_ancilla_indices:
                        if code.X_gauge_stabilizers:
                            assert code.Z_gauge_stabilizers
                            
                        else:
                            if len(meas_counter[q]) > 1 and (XYZ_decode or (basis == 'X' and q in code.X_ancilla_indices) or (basis == 'Z' and q in code.Z_ancilla_indices)):
                                circ.append('DETECTOR', [stim.target_rec(meas_counter[q][-1] - meas_count), stim.target_rec(meas_counter[q][-2] - meas_count)], (q, len(meas_counter[q])-1))
                            else:
                                if (basis == 'X' and q in code.X_ancilla_indices) or (basis == 'Z' and q in code.Z_ancilla_indices):
                                    circ.append('DETECTOR', [stim.target_rec(meas_counter[q][-1] - meas_count)], (q, len(meas_counter[q])-1))
            elif isinstance(instr, Shuttle):
                circ.append('Z_ERROR', instr.qubit, error_params.shuttle_err)
            elif isinstance(instr, EmplaceDisplace):
                circ.append('Z_ERROR', instr.qubit, min(0.75, error_params.emplace_err))
            else:
                raise ValueError('Unsupported instruction:', instr)
            
            # update state
            for q in instr.qubits_list():
                qubit_time_last_used[q] = start_time + instr.duration
                            
        # Add final detectors
        for q in meas_counter:
            if basis == 'X' and q in code.X_ancilla_indices:
                data_indices = code.X_checks[code.X_ancilla_indices.index(q)]
            elif basis == 'Z' and q in code.Z_ancilla_indices:
                data_indices = code.Z_checks[code.Z_ancilla_indices.index(q)]
            else:
                continue
            circ.append('DETECTOR', [stim.target_rec(meas_counter[d][-1] - meas_count) for d in data_indices] + [stim.target_rec(meas_counter[q][-1] - meas_count)], (q, len(meas_counter[q])))
        
        # Add logical observables
        Lx,Lz = code.compute_logical_operators()
        # assert Lx.shape[0] == code.compute_code_parameters()[1]
        for i in range(Lx.shape[0]):
            logicals = np.nonzero(Lx[i,:])[0] if basis == 'X' else np.nonzero(Lz[i,:])[0]
            circ.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_counter[q][-1] - meas_count) for q in logicals], i)

        assert (init_basis_change and meas_basis_change) or (not init_basis_change and not meas_basis_change)

        return circ
    
    def to_stim_circuit_simple(self, code: QECCode, basis: str, error_params: ErrorParams) -> stim.Circuit:
        """Convert the compiled shuttling schedule back into a Stim circuit."""
        circ = stim.Circuit()

        if hasattr(code, 'qubit_coords'):
            for q,coords in enumerate(code.qubit_coords):
                circ.append('QUBIT_COORDS', q, coords)

        qubit_time_last_used: dict[int, int] = dict()
        meas_counter: dict[int, list[int]] = dict() # meas index of each measurement
        meas_count: int = 0
        instrs_sort = np.argsort(self.instruction_start_times)
        instructions = [self.instructions[i] for i in instrs_sort]
        instruction_start_times = [self.instruction_start_times[i] for i in instrs_sort]
        instr_dag = nx.DiGraph()
        for i,instr in enumerate(instructions):
            instr_dag.add_node(i)
        for q,indices in self.instructions_by_qubit.items():
            for i,ii in zip(indices[:-1], indices[1:]):
                instr_dag.add_edge(i, ii)
        last_start_time = 0
        for instr, start_time in zip(instructions, instruction_start_times):
            # apply idle errors
            if not isinstance(instr, Instantiate) and start_time > last_start_time:
                idle_ops: dict[float, list[int]] = dict()
                for q in instr.qubits_list():
                    if qubit_time_last_used[q] < start_time:
                        idle_time = start_time - qubit_time_last_used[q]
                        idle_err = 1 - np.exp(-idle_time*1e-9 / error_params.T2)
                        idle_ops.setdefault(idle_err, []).append(q)
                for idle_err, qs in idle_ops.items():
                    circ.append('Z_ERROR', qs, min(0.75, idle_err))
                last_start_time = start_time

            # apply instruction
            if isinstance(instr, Gate):
                if instr.name == GateName.CX:
                    circ.append('CX', instr.qubits, ())
                    circ.append('DEPOLARIZE2', instr.qubits, min(15/16, error_params.cx_err))
                elif instr.name == GateName.H:
                    circ.append('H', instr.qubits, ())
                    circ.append('DEPOLARIZE1', instr.qubits, min(0.75, error_params.h_err))
                else:
                    raise NotImplementedError
            elif isinstance(instr, Instantiate):
                circ.append('R', instr.qubits, ())
                circ.append('X_ERROR', instr.qubits, error_params.init_err)
            elif isinstance(instr, Measure):
                circ.append('X_ERROR', instr.qubits, error_params.measure_err)
                circ.append('M', instr.qubits, ())
                for q in instr.qubits:
                    meas_counter.setdefault(q, []).append(meas_count)
                    meas_count += 1
                # Add detectors if ancilla qubit
                for q in instr.qubits:
                    if q in code.X_ancilla_indices + code.Z_ancilla_indices:
                        if len(meas_counter[q]) > 1:
                            circ.append('DETECTOR', [stim.target_rec(meas_counter[q][-1] - meas_count), stim.target_rec(meas_counter[q][-2] - meas_count)], (q, len(meas_counter[q])-1))
                        else:
                            if (basis == 'X' and q in code.X_ancilla_indices) or (basis == 'Z' and q in code.Z_ancilla_indices):
                                circ.append('DETECTOR', [stim.target_rec(meas_counter[q][-1] - meas_count)], (q, len(meas_counter[q])-1))
            elif isinstance(instr, Shuttle):
                circ.append('Z_ERROR', instr.qubit, error_params.shuttle_err)
            elif isinstance(instr, EmplaceDisplace):
                circ.append('Z_ERROR', instr.qubit, min(0.75, error_params.emplace_err))
            else:
                raise ValueError('Unsupported instruction:', instr)
            
            # update state
            for q in instr.qubits_list():
                qubit_time_last_used[q] = start_time + instr.duration
                            
        # Add final detectors
        for q in meas_counter:
            if basis == 'X' and q in code.X_ancilla_indices:
                data_indices = code.X_checks[code.X_ancilla_indices.index(q)]
            elif basis == 'Z' and q in code.Z_ancilla_indices:
                data_indices = code.Z_checks[code.Z_ancilla_indices.index(q)]
            else:
                continue
            circ.append('DETECTOR', [stim.target_rec(meas_counter[d][-1] - meas_count) for d in data_indices] + [stim.target_rec(meas_counter[q][-1] - meas_count)], (q, len(meas_counter[q])))
        
        # Add logical observables
        Lx,Lz = code.compute_logical_operators()
        # assert Lx.shape[0] == code.compute_code_parameters()[1]
        for i in range(Lx.shape[0]):
            logicals = np.nonzero(Lx[i,:])[0] if basis == 'X' else np.nonzero(Lz[i,:])[0]
            circ.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_counter[q][-1] - meas_count) for q in logicals], i)

        return circ

    def update_safe_intervals(
            self,
            node_safe_intervals: dict[DeviceComponent, dict[tuple[int, int], list[tuple[int, int]]]],
            edge_safe_intervals: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]],
            qubit: int,
            buffer_time: int,
            allow_interleaving: bool = True,
        ) -> tuple[dict[DeviceComponent, dict[tuple[int, int], list[tuple[int, int]]]], dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        """The key function that updates the safe "intervals" to account
        for a newly-scheduled qubit, ensuring that future schedules do
        not collide with this one."""
        
        def get_new_intervals(intervals_old: list[tuple[int, int]], t_start: int, instr_duration: int):
            """Update the intervals for one component."""
            intervals_new = []
            for (s,e) in intervals_old:
                if t_start - buffer_time < e and s < t_start + instr_duration + buffer_time:
                    interval0 = (max(s, t_start + instr_duration + buffer_time), e)
                    if interval0[0] < interval0[1]:
                        intervals_new.append(interval0)
                    interval1 = (s, min(t_start - buffer_time, e))
                    if interval1[0] < interval1[1]:
                        intervals_new.append(interval1)
                else:
                    intervals_new.append((s,e))
            if allow_interleaving:
                return intervals_new
            else:
                intervals_new = [(s,e) for (s,e) in intervals_new if e == MAX_T]
                assert len(intervals_new) == 1
                return intervals_new

        # Identify times when each qubit is idling, figure out
        # what component it is in, and update intervals
        qubit_idle_times_and_components = [] # (start, end, coords, component)
        cur_coords = (0, 0)
        cur_component = DeviceComponent.READOUT
        for i,(instr, t_start) in list(enumerate(zip(self.instructions, self.instruction_start_times)))[:-1]:
            if qubit in instr.qubits_list():
                if isinstance(instr, Instantiate):
                    cur_coords = instr.cell_coords[instr.qubits.index(qubit)]
                    cur_component = DeviceComponent.READOUT
                elif isinstance(instr, Shuttle):
                    cur_coords = instr.end_coords
                elif isinstance(instr, EmplaceDisplace):
                    cur_component = instr.end_component
                if t_start + instr.duration < self.instruction_start_times[i+1]:
                    # Idle before next instruction
                    qubit_idle_times_and_components.append((t_start + instr.duration, self.instruction_start_times[i+1], cur_coords, cur_component))
        for (t_start, t_end, coords, comp) in qubit_idle_times_and_components:
            node_safe_intervals[comp][coords] = get_new_intervals(node_safe_intervals[comp][coords], t_start, t_end - t_start)

        for instr_idx in self.instructions_by_qubit[qubit]:
            instr = self.instructions[instr_idx]
            t_start = self.instruction_start_times[instr_idx]
            if isinstance(instr, Shuttle):
                edge = sort((instr.start_coords, instr.end_coords))
                edge_safe_intervals[edge] = get_new_intervals(edge_safe_intervals[edge], t_start, instr.duration)
                node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.start_coords] = get_new_intervals(node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.start_coords], t_start, 0)
                node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.end_coords] = get_new_intervals(node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.end_coords], t_start + instr.duration, 0)
            elif isinstance(instr, EmplaceDisplace):
                # Qubit changing locations
                qubit_arrival = t_start + instr.duration
                qubit_departure = None
                for instr_future, t_start_future in zip(self.instructions[instr_idx+1:], self.instruction_start_times[instr_idx+1:]):
                    if instr.qubit in instr_future.qubits_list() and (isinstance(instr_future, EmplaceDisplace) or isinstance(instr_future, Measure)):
                        qubit_departure = t_start_future
                        break
                assert qubit_departure is not None
                node_safe_intervals[instr.start_component][instr.coords] = get_new_intervals(node_safe_intervals[instr.start_component][instr.coords], qubit_arrival, 0)
                node_safe_intervals[instr.end_component][instr.coords] = get_new_intervals(node_safe_intervals[instr.end_component][instr.coords], qubit_arrival, 0)
            elif isinstance(instr, Instantiate):
                coords = instr.cell_coords
                for coords in instr.cell_coords:
                    node_safe_intervals[DeviceComponent.READOUT][coords] = get_new_intervals(node_safe_intervals[DeviceComponent.READOUT][coords], t_start, instr.duration)
        
        if not allow_interleaving:
            # only one interval per location, starting once it becomes fully free
            for node in node_safe_intervals:
                for coord in node_safe_intervals[node]:
                    node_safe_intervals[node][coord] = [(start,end) for start,end in node_safe_intervals[node][coord] if end == MAX_T]
                    assert len(node_safe_intervals[node][coord]) == 1
            for edge in edge_safe_intervals:
                edge_safe_intervals[edge] = [(start,end) for start,end in edge_safe_intervals[edge] if end == MAX_T]
                assert len(edge_safe_intervals[edge]) == 1

        return node_safe_intervals, edge_safe_intervals

    def get_safe_intervals(
            self,
            all_coords: list[tuple[int, int]],
            all_edges: list[tuple[tuple[int, int], tuple[int, int]]],
            allow_interleaving: bool,
            buffer_time: int,
        ) -> tuple[dict[DeviceComponent, dict[tuple[int, int], list[tuple[int, int]]]], dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]]:
        node_safe_intervals: dict[DeviceComponent, dict[tuple[int, int], list[tuple[int, int]]]] = {
            DeviceComponent.SHUTTLE_INTERSECTION: {c:[(0, MAX_T)] for c in all_coords},
            DeviceComponent.INTERACTION_ZONE: {c:[(0, MAX_T)] for c in all_coords},
            DeviceComponent.READOUT: {c:[(0, MAX_T)] for c in all_coords},
        }
        edge_safe_intervals: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {e:[(0, MAX_T)] for e in all_edges}

        for qubit in self.qubits:
            node_safe_intervals, edge_safe_intervals = self.update_safe_intervals(
                node_safe_intervals=node_safe_intervals,
                edge_safe_intervals=edge_safe_intervals,
                qubit=qubit,
                buffer_time=buffer_time,
                allow_interleaving=allow_interleaving,
            )

        return node_safe_intervals, edge_safe_intervals

class ScheduleMetrics:
    qubit_active_durations: dict[int, int]
    qubit_idle_durations: dict[int, int]
    qubit_distance_traveled: dict[int, int]
    avg_qubit_idle_frac: float
    # avg_distance_traveled: float

    def __init__(self, schedule: CompiledShuttlingSchedule):
        self.qubit_active_durations = dict()
        qubit_init_times = dict()
        qubit_last_active = dict()
        self.qubit_idle_durations = dict()
        self.qubit_distance_traveled = dict()

        for instr, start_time in zip(schedule.instructions, schedule.instruction_start_times):
            qubits = instr.qubits_list()
            if isinstance(instr, Shuttle):
                for q in qubits:
                    self.qubit_distance_traveled[q] = self.qubit_distance_traveled.get(q, 0) + 1
            elif isinstance(instr, Instantiate):
                for q in qubits:
                    if q in qubit_init_times:
                        # Starting new round; exit
                        print('Multiple rounds detected; exiting early')
                        return
                    qubit_init_times[q] = start_time
            elif isinstance(instr, Measure):
                for q in qubits:
                    self.qubit_active_durations[q] = start_time + instr.duration - qubit_init_times[q]
            for q in qubits:
                if not isinstance(instr, Instantiate) and qubit_last_active[q] < start_time:
                    self.qubit_idle_durations[q] = self.qubit_idle_durations.get(q, 0) + start_time - qubit_last_active[q]
                qubit_last_active[q] = start_time + instr.duration
        self.avg_qubit_idle_frac = float(np.mean([d/self.qubit_active_durations[q] for q,d in self.qubit_idle_durations.items() if q in self.qubit_distance_traveled]))
        self.avg_distance_traveled = float(np.mean([d for d in self.qubit_distance_traveled.items()]))

class SchedulingMethod(Enum):
    GREEDY = 'GREEDY' # no lookahead, schedule a shuttle one edge at a time
    TILED = 'TILED' # every ancilla qubit of the same basis has the same movement pattern

@dataclass
class State:
    x: int
    y: int
    component: DeviceComponent
    t_start: int
    t_end: int
    cur_task: int | tuple[bool, ...]
    __slots__ = ('x', 'y', 'component', 't_start', 't_end', 'cur_task')

    def tuple(self):
        return (self.x, self.y, self.component, self.t_start, self.t_end, self.cur_task)

    def __hash__(self):
        return hash(self.tuple())

    def __eq__(self, other):
        return self.tuple() == other.tuple()

    def __str__(self):
        return f'State(x={self.x}, y={self.y}, component={self.component}, t_start={self.t_start}, t_end={self.t_end}, cur_task={self.cur_task})'
    
    def __lt__(self, other):
        return self.tuple() < other.tuple()
    
    def __gt__(self, other):
        return self.tuple() > other.tuple()

class UnitCellDevice:
    """Represents a shuttling-enabled unit cell architecture. Qubits can be
    shuttled along a square grid of channels. Each intersection has integer
    coordinates and can hold two qubits without blocking the shuttling. Readout
    and two-qubit gates can be performed at any intersection."""
    w: int
    h: int
    hardware_params: HardwareParams
    debug: bool

    def __init__(
            self,
            w: int,
            h: int,
            hardware_params: HardwareParams,
            debug: bool = False
        ) -> None:
        self.w = w
        self.h = h
        self.hardware_params = hardware_params
        self.debug = debug
        self.all_coords = [(x,y) for x in range(self.w) for y in range(self.h)]
        self.all_edges = [sort((c0, c1)) for c0 in self.all_coords for c1 in self.all_coords if c0 != c1]

    def _schedule_cache_dir(self, cache_dir: str | Path | None = None) -> Path:
        if cache_dir is None:
            # Default to a repo-local cache folder.
            return Path(__file__).resolve().parents[1] / ".cache" / "unit_cell_device"
        return Path(cache_dir)

    def _schedule_cache_path(self, key: str, cache_dir: str | Path | None = None) -> Path:
        dirpath = self._schedule_cache_dir(cache_dir)
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath / f"schedule_{key}.pkl"

    def _schedule_cache_key(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod,
            separate_X_Z: bool,
            buffer_time: int,
            debug_qubit: int | None,
        ) -> str:
        # Create a deterministic hash based on the inputs that affect the schedule.
        key_data = (
            self.w,
            self.h,
            asdict(self.hardware_params),
            self.debug,
            code.__class__.__name__,
            tuple(code.data_indices),
            tuple(code.X_ancilla_indices),
            tuple(code.Z_ancilla_indices),
            tuple(tuple(checks) for checks in code.X_checks),
            tuple(tuple(checks) for checks in code.Z_checks),
            tuple(code.qubit_coords) if hasattr(code, 'qubit_coords') else (),
            tuple(sorted(static_data_coords.items())),
            tuple(tuple(layer) for layer in cx_layers),
            use_highways,
            refocus_shuttle_noise,
            scheduling_method,
            separate_X_Z,
            buffer_time,
            debug_qubit,
        )
        digest = hashlib.sha256(pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
        return digest

    def compile_QEC_schedule(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            rounds: int,
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod = SchedulingMethod.GREEDY,
            separate_X_Z: bool = False,
            buffer_time: int = 100,
            debug_qubit: int | None = None,
            use_cache: bool = True,
            force_overwrite_cache: bool = False,
            cache_dir: str | Path | None = '.scheduler_cache',
        ):
        schedule = CompiledShuttlingSchedule(sorted(list(static_data_coords.keys())), sorted(list(static_data_coords.keys())))
        schedule.append_instr(
            Instantiate(
                self.hardware_params.init_duration,
                code.data_indices,
                [static_data_coords[data] for data in code.data_indices],
            ), 0
        )
        t = schedule.total_duration()
        for q in code.data_indices:
            schedule.append_instr(
                EmplaceDisplace(
                    self.hardware_params.emplace_duration,
                    q,
                    static_data_coords[q],
                    DeviceComponent.READOUT,
                    DeviceComponent.INTERACTION_ZONE,
                ), t,
            )
        SE_sched = self.compile_SE_schedule(
            code,
            static_data_coords,
            cx_layers,
            use_highways,
            refocus_shuttle_noise,
            scheduling_method=scheduling_method,
            separate_X_Z=separate_X_Z,
            buffer_time=buffer_time,
            debug_qubit=debug_qubit,
            use_cache=use_cache,
            force_overwrite_cache=force_overwrite_cache,
            cache_dir=cache_dir,
        )
        schedule += rounds * SE_sched
        t = schedule.total_duration()
        for q in code.data_indices:
            schedule.append_instr(
                EmplaceDisplace(
                    self.hardware_params.emplace_duration,
                    q,
                    static_data_coords[q],
                    DeviceComponent.READOUT,
                    DeviceComponent.INTERACTION_ZONE,
                ), t,
            )
        schedule.append_instr(
            Measure(
                self.hardware_params.init_duration,
                code.data_indices,
                [static_data_coords[data] for data in code.data_indices],
            ), schedule.total_duration()
        )
        return schedule

    def compile_SE_schedule(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod = SchedulingMethod.GREEDY,
            separate_X_Z: bool = False,
            buffer_time: int = 0,
            debug_qubit: int | None = None,
            use_cache: bool = False,
            force_overwrite_cache: bool = False,
            cache_dir: str | Path | None = None,
        ) -> CompiledShuttlingSchedule:
        schedule = None
        cache_path = None
        if use_cache:
            cache_key = self._schedule_cache_key(
                code=code,
                static_data_coords=static_data_coords,
                cx_layers=cx_layers,
                use_highways=use_highways,
                refocus_shuttle_noise=refocus_shuttle_noise,
                scheduling_method=scheduling_method,
                separate_X_Z=separate_X_Z,
                buffer_time=buffer_time,
                debug_qubit=debug_qubit,
            )
            cache_path = self._schedule_cache_path(cache_key, cache_dir)
            if cache_path.exists() and not force_overwrite_cache:
                try:
                    with open(cache_path, "rb") as f:
                        schedule = pickle.load(f)
                    if self.debug:
                        print(f"Loaded cached schedule from {cache_path}")
                    return schedule
                except Exception as e:
                    if self.debug:
                        print(f"Failed to load schedule cache ({cache_path}): {e}")

        if scheduling_method == SchedulingMethod.GREEDY:
            schedule_fn = self._compile_SE_schedule_greedy
        elif scheduling_method == SchedulingMethod.TILED:
            schedule_fn = self._compile_SE_schedule_tiled
        else:
            raise ValueError('Unknown scheduling method')

        if separate_X_Z:
            X_sched = schedule_fn(
                code=code,
                static_data_coords=static_data_coords,
                cx_layers=cx_layers,
                use_highways=use_highways,
                refocus_shuttle_noise=refocus_shuttle_noise,
                anc_basis_filter='X',
                buffer_time=buffer_time,
                debug_qubit=debug_qubit,
            )
            schedule = schedule_fn(
                code=code,
                static_data_coords=static_data_coords,
                cx_layers=cx_layers,
                use_highways=use_highways,
                refocus_shuttle_noise=refocus_shuttle_noise,
                anc_basis_filter='Z',
                buffer_time=buffer_time,
                prior_schedule=X_sched,
                allow_interleave_with_prior=False,
                debug_qubit=debug_qubit,
            )
        else:
            schedule = schedule_fn(
                code=code,
                static_data_coords=static_data_coords,
                cx_layers=cx_layers,
                use_highways=use_highways,
                refocus_shuttle_noise=refocus_shuttle_noise,
                buffer_time=buffer_time,
                debug_qubit=debug_qubit,
            )

        if use_cache and cache_path is not None:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(schedule, f, protocol=pickle.HIGHEST_PROTOCOL)
                if self.debug:
                    print(f"Saved schedule cache to {cache_path}")
            except Exception as e:
                if self.debug:
                    print(f"Failed to write schedule cache ({cache_path}): {e}")

        return schedule

    def _compile_SE_schedule_greedy(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            anc_basis_filter: str | None = None,
            buffer_time: int = 0,
            prior_schedule: CompiledShuttlingSchedule | None = None,
            allow_interleave_with_prior: bool = False,
            debug_qubit: int | None = None,
        ) -> CompiledShuttlingSchedule:
        """Compile a shuttling schedule for a syndrome extraction round. Assumes
        data qubits are already initialized and fixed in place within each unit
        cell.
        """
        ########################################################################
        # Checking inputs and setting up variables
        ########################################################################
        if len(set(static_data_coords.values())) < len(static_data_coords):
            raise ValueError("Static data qubit positions must be unique.")
        if any(coord < 0 for coords in static_data_coords.values() for coord in coords):
            raise ValueError("Negative data coordinates not supported")

        if anc_basis_filter == 'X':
            anc_qubits = code.X_ancilla_indices
        elif anc_basis_filter == 'Z':
            anc_qubits = code.Z_ancilla_indices
        else:
            anc_qubits = list(sorted(code.X_ancilla_indices + code.Z_ancilla_indices))

        all_qubits = list(sorted(code.data_indices + anc_qubits))

        if len(static_data_coords) > 0 and set(static_data_coords) != set(code.data_indices):
            raise ValueError('Must supply all or no data indices')
        if len(static_data_coords) == 0:
            raise NotImplementedError('Auto computing data positions not supported yet')

        if cx_layers:
            init_ancilla_coords, cx_layers = self._map_init_ancilla_positions(static_data_coords, code, cx_layers)
            init_ancilla_coords = {a:c for a,c in init_ancilla_coords.items() if a in anc_qubits}
            assert set(init_ancilla_coords.keys()) == set(anc_qubits)
            ancilla_data_to_visit = dict()
            for layer in cx_layers:
                for qa,qb in layer:
                    if qa in anc_qubits:
                        ancilla_data_to_visit.setdefault(qa, []).append(qb)
                    elif qb in anc_qubits:
                        ancilla_data_to_visit.setdefault(qb, []).append(qa)
                    else:
                        continue
        else:
            init_ancilla_coords = None
            ancilla_data_to_visit = dict()
            for anc,checks in zip(code.X_ancilla_indices + code.Z_ancilla_indices, code.X_checks + code.Z_checks):
                ancilla_data_to_visit[anc] = checks

        # Variables used in compilation
        if prior_schedule:
            schedule = prior_schedule
        else:
            schedule = CompiledShuttlingSchedule(sorted(code.data_indices + code.X_ancilla_indices + code.Z_ancilla_indices), code.data_indices)
        node_safe_intervals, edge_safe_intervals = schedule.get_safe_intervals(
            self.all_coords,
            self.all_edges,
            allow_interleaving=allow_interleave_with_prior,
            buffer_time=buffer_time,
        )
        if init_ancilla_coords:
            schedule.append_instr(
                Instantiate(
                    self.hardware_params.init_duration,
                    anc_qubits,
                    [init_ancilla_coords[anc] for anc in anc_qubits],
                ), 0
            )
            schedule.append_instr(
                Gate(
                    self.hardware_params.h_duration,
                    code.X_ancilla_indices,
                    GateName.H,
                ), schedule.total_duration()
            )

        ########################################################################
        # Helper functions
        ########################################################################
        def distance(coordsA, coordsB):
            return abs(coordsA[0] - coordsB[0])*0.999 + abs(coordsA[1] - coordsB[1])

        tsp_results = dict()
        def solve_tsp_memoized(cur_coords, coords_to_reach):
            # Canonicalize coordinates b/c relative distances are the important
            # part
            xc,yc = cur_coords
            coords_canonicalized = [(0,0)]
            for x,y in coords_to_reach:
                coords_canonicalized.append((x-xc, y-yc))
            key = tuple(coords_canonicalized)
            if key in tsp_results:
                return tsp_results[key]
            else:
                distance_matrix = np.array([[distance(a, b) for a in [cur_coords] + coords_to_reach] for b in [cur_coords] + coords_to_reach])
                distance_matrix[:, 0] = 0
                if distance_matrix.shape[0] < 10:
                    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
                else:
                    permutation, _ = solve_tsp_local_search(distance_matrix, max_processing_time=0.001)
                tsp_results[key] = permutation
                return permutation

        def get_cx_duration(anc: int, task_idx: int, num_tasks: int) -> int:
            duration = self.hardware_params.cx_duration
            if refocus_shuttle_noise and anc in code.Z_ancilla_indices:
                if task_idx == num_tasks:
                    raise ValueError('All tasks complete - don\'t ask for CX duration!')
                if task_idx == 0 or task_idx == num_tasks-1:
                    duration += self.hardware_params.h_duration
                else:
                    duration += 2*self.hardware_params.h_duration
            return duration

        def get_cx_ops(anc: int, data: int, data_idx_in_stab: int, stab_weight: int) -> list[Gate]:
            ops = []
            if refocus_shuttle_noise and anc in code.Z_ancilla_indices:
                if data_idx_in_stab != 0:
                    ops.append(Gate(self.hardware_params.h_duration, [anc], GateName.H))
                ops.append(Gate(self.hardware_params.cx_duration, [data, anc], GateName.CX))
                if data_idx_in_stab != stab_weight-1:
                    ops.append(Gate(self.hardware_params.h_duration, [anc], GateName.H))
            else:
                if anc in code.Z_ancilla_indices:
                    ops.append(Gate(self.hardware_params.cx_duration, [data, anc], GateName.CX))
                else:
                    assert anc in code.X_ancilla_indices
                    ops.append(Gate(self.hardware_params.cx_duration, [anc, data], GateName.CX))
            
            # Debug check
            if sum(op.duration for op in ops) != get_cx_duration(anc, data_idx_in_stab, stab_weight):
                pass
            assert sum(op.duration for op in ops) == get_cx_duration(anc, data_idx_in_stab, stab_weight)

            return ops

        def get_heuristic(anc, target_indices, target_coords):
            # Heuristic roughly approximates total remaining time to complete
            # data checks and readout if there are no obstacles.
            def heuristic(state: State):
                cur_task = state.cur_task
                if isinstance(cur_task, int):
                    # Fixed order of tasks
                    if cur_task == len(target_coords):
                        if state.component == DeviceComponent.READOUT:
                            if state.t_end - cost_to_node[state] >= self.hardware_params.measure_duration:
                                return 0
                            else:
                                return 2*self.hardware_params.emplace_duration
                        else:
                            return self.hardware_params.emplace_duration
                    else:
                        cost = 0
                        if (state.component == DeviceComponent.INTERACTION_ZONE or state.component == DeviceComponent.READOUT) and (state.x, state.y) != target_coords[cur_task]:
                            cost += self.hardware_params.emplace_duration
                        if state.component == DeviceComponent.INTERACTION_ZONE and (state.x, state.y) == target_coords[cur_task]:
                            cost += get_cx_duration(anc, cur_task, len(target_indices))
                            cur_task += 1
                            cost += self.hardware_params.emplace_duration
                        # Time to shuttle to remaining tasks and do CXs
                        cur = state.x, state.y
                        for tgt_idx,tgt in list(enumerate(target_coords))[cur_task:]:
                            cost += distance(tgt, cur)*self.hardware_params.shuttle_duration + get_cx_duration(anc, tgt_idx, len(target_indices)) + 2*self.hardware_params.emplace_duration
                            cur = tgt
                        return cost
                else:
                    # Can choose order of tasks
                    if all(cur_task):
                        if state.component == DeviceComponent.READOUT:
                            if state.t_end - cost_to_node[state] >= self.hardware_params.measure_duration:
                                return 0
                            else:
                                return 2*self.hardware_params.emplace_duration
                        else:
                            return self.hardware_params.emplace_duration
                    else:
                        assert not isinstance(state.cur_task, int)
                        tasks_completed = sum(cur_task)
                        cost = 0
                        coords_to_reach = [c for i,c in enumerate(target_coords) if not state.cur_task[i]]
                        if (state.component == DeviceComponent.INTERACTION_ZONE or state.component == DeviceComponent.READOUT) and (state.x, state.y) not in coords_to_reach:
                            cost += self.hardware_params.emplace_duration
                        if state.component == DeviceComponent.INTERACTION_ZONE and (state.x, state.y) in coords_to_reach:
                            cost += get_cx_duration(anc, tasks_completed, len(target_indices))
                            cur_task = tuple(True if c == (state.x, state.y) else b for b,c in zip(cur_task, target_coords))
                            cost += self.hardware_params.emplace_duration
                        
                        # Time to shuttle to remaining tasks and do CXs
                        cur = state.x, state.y
                        permutation = solve_tsp_memoized(cur, coords_to_reach)
                        for i in permutation[1:]:
                            tgt = coords_to_reach[i-1]
                            cost += distance(tgt, cur)*self.hardware_params.shuttle_duration + get_cx_duration(anc, tasks_completed+i-1, len(target_indices)) + 2*self.hardware_params.emplace_duration
                            cur = tgt
                        return cost
            return heuristic

        ########################################################################
        # Optimization loop - for each ancilla
        ########################################################################
        if not self.debug:
            print(f'Optimizing {len(anc_qubits)} ancilla schedules', end='')
        for anc in anc_qubits:
            data_qubits = ancilla_data_to_visit[anc]
            data_coords = [static_data_coords[data] for data in data_qubits]
            if self.debug:
                self.printd('Optimizing ancilla qubit', anc, 'on data qubits', data_qubits, 'at coordinates', data_coords)
            heuristic = get_heuristic(anc, data_qubits, data_coords)
            frontier = set() # TODO: can use min-heap or pqueue
            frontier_heap = []
            if init_ancilla_coords:
                x,y = init_ancilla_coords[anc]
                for (int_start, int_end) in node_safe_intervals[DeviceComponent.READOUT][(x,y)]:
                    if int_end - int_start < self.hardware_params.init_duration:
                        continue
                    state_new = State(
                        x=x,
                        y=y,
                        component=DeviceComponent.READOUT,
                        t_start=int_start,
                        t_end=int_end,
                        cur_task=(0 if cx_layers else (False,)*len(data_qubits)),
                    )
                    frontier.add(state_new)
                    heapq.heappush(frontier_heap, (heuristic(state_new), state_new))
            else:
                # Free to choose where and when to start
                for coords in [(x,y) for x in range(self.w) for y in range(self.h)]:
                    for (int_start, int_end) in node_safe_intervals[DeviceComponent.READOUT][coords]:
                        if int_end - int_start < self.hardware_params.init_duration:
                            continue
                        state_new = State(
                            x=coords[0],
                            y=coords[1],
                            component=DeviceComponent.READOUT,
                            t_start=int_start,
                            t_end=int_end,
                            cur_task=(0 if cx_layers else (False,)*len(data_qubits)),
                        )
                        frontier.add(state_new)
                        heapq.heappush(frontier_heap, (heuristic(state_new), state_new))
            start_states = deepcopy(frontier)
            came_from = {}
            cost_to_node = {
                s:
                self.hardware_params.init_duration+(self.hardware_params.h_duration if anc in code.X_ancilla_indices else 0)+s.t_start
                for s in frontier
            }
            node_heuristic_vals = {s: heuristic(s) for s in frontier}

            # self.printd(anc, (x,y), data_coords)

            def add_state(state: State, state_new: State, transition_cost: int):
                assert transition_cost > 0
                self.printd(f'\tADD {(state_new.x, state_new.y, str(state_new.component), state_new.cur_task)} FROM {(state.x, state.y, str(state.component), state.cur_task)} WITH COST', transition_cost)
                new_cost_to_node = max(cost_to_node[state] + transition_cost, state_new.t_start)
                if state_new in cost_to_node:
                    self.printd(f'\t\ttalready seen, cost {cost_to_node[state_new]}, new cost {new_cost_to_node}')
                if cost_to_node.get(state_new, MAX_T) > new_cost_to_node:
                    cost_to_node[state_new] = new_cost_to_node
                    if state not in start_states and came_from[state] == state_new:
                        raise RuntimeError
                    came_from[state_new] = state
                    node_heuristic_vals[state_new] = cost_to_node[state_new] + heuristic(state_new)
                    if state_new not in frontier:
                        frontier.add(state_new)
                        heapq.heappush(frontier_heap, (node_heuristic_vals[state_new], state_new))
                    self.printd('\t\testimated cost:', node_heuristic_vals[state_new])
                else:
                    self.printd('\t\tpass')

            def trace_path(state) -> list[State]:
                state_path = [state]
                prev_state = state
                while prev_state in came_from:
                    prev_state = came_from[prev_state]
                    state_path.append(copy(prev_state))
                state_path = list(reversed(state_path))
                return state_path

            found_solution = False
            while frontier:
                _,state = heapq.heappop(frontier_heap)
                frontier.remove(state)

                self.printd(f'POP {(state.x, state.y, str(state.component), state.cur_task)} with cost {node_heuristic_vals[state]} ({cost_to_node[state]} + {heuristic(state)})')
                if node_heuristic_vals[state] - cost_to_node[state] == 0:
                    # DONE

                    # TODO: recalculate cost_to_node so that we push the idling
                    # as early as possible rather than the current approach of
                    # as late as possible

                    # Trace optimized path to build schedule
                    state_path = trace_path(state)

                    if not init_ancilla_coords:
                        schedule.append_instr(Instantiate(
                            self.hardware_params.init_duration,
                            [anc],
                            [(state_path[0].x, state_path[0].y)],
                        ), state_path[0].t_start)
                        if anc in code.X_ancilla_indices:
                            schedule.append_instr(Gate(
                                self.hardware_params.h_duration,
                                [anc],
                                GateName.H,
                            ), state_path[0].t_start + self.hardware_params.init_duration)

                    for si,s in enumerate(state_path):
                        self.printd(cost_to_node[s], s)
                        if si == len(state_path)-1:
                            continue
                        ss = state_path[si+1]
                        if (ss.x, ss.y) != (s.x, s.y):
                            assert s.component == DeviceComponent.SHUTTLE_INTERSECTION and ss.component == DeviceComponent.SHUTTLE_INTERSECTION, (s, ss)
                            schedule.append_instr(Shuttle(
                                self.hardware_params.shuttle_duration,
                                anc,
                                (s.x, s.y),
                                (ss.x, ss.y),
                            ), cost_to_node[ss] - self.hardware_params.shuttle_duration)
                        elif s.component != ss.component:
                            assert (s.x, s.y) == (ss.x, ss.y)
                            schedule.append_instr(EmplaceDisplace(
                                self.hardware_params.emplace_duration,
                                anc,
                                (s.x, s.y),
                                s.component,
                                ss.component,
                            ), cost_to_node[ss] - self.hardware_params.emplace_duration)
                        elif s.cur_task != ss.cur_task:
                            if isinstance(s.cur_task, int):
                                idx = s.cur_task
                                task_number = idx
                            else:
                                assert not isinstance(ss.cur_task, int)
                                idxs = [i for i,(b,bb) in enumerate(zip(s.cur_task, ss.cur_task)) if b != bb]
                                assert len(idxs) == 1
                                idx = idxs[0]
                                task_number = sum(s.cur_task)
                            cx_ops = get_cx_ops(anc, data_qubits[idx], task_number, len(data_qubits))
                            cx_ops_dur = get_cx_duration(anc, task_number, len(data_qubits))
                            assert cx_ops_dur == sum(op.duration for op in cx_ops)
                            if anc == debug_qubit:
                                print(anc, idx, task_number, len(cx_ops))
                                for op in cx_ops:
                                    print('\t', op)
                            t = cost_to_node[ss] - cx_ops_dur
                            for op in cx_ops:
                                schedule.append_instr(op, t)
                                t += op.duration
                    assert state_path[-1].component == DeviceComponent.READOUT
                    if anc in code.X_ancilla_indices:
                        schedule.append_instr(Gate(
                            self.hardware_params.h_duration,
                            [anc],
                            GateName.H,
                        ), cost_to_node[state])
                    schedule.append_instr(Measure(
                        self.hardware_params.measure_duration,
                        [anc],
                        [(state_path[-1].x, state_path[-1].y)],
                    ), cost_to_node[state] + self.hardware_params.h_duration)

                    node_safe_intervals, edge_safe_intervals = schedule.update_safe_intervals(
                        node_safe_intervals=node_safe_intervals,
                        edge_safe_intervals=edge_safe_intervals,
                        qubit=anc,
                        buffer_time=buffer_time,
                        allow_interleaving=True,
                    )

                    found_solution = True
                    break

                # Shuttling steps
                if state.component == DeviceComponent.SHUTTLE_INTERSECTION:
                    for dx,dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_x,new_y = (state.x + dx, state.y + dy)
                        if not (0 <= new_x < self.w and 0 <= new_y < self.h):
                            continue
                        shuttle_edge = sort(((state.x, state.y), (new_x,new_y)))
                        for (edge_start, edge_end) in edge_safe_intervals[shuttle_edge]:
                            earliest_start = max(edge_start, cost_to_node[state])
                            latest_end = min(edge_end, state.t_end + self.hardware_params.shuttle_duration)
                            if latest_end - earliest_start >= self.hardware_params.shuttle_duration:
                                # We can shuttle through the intersection in
                                # this interval. Now we need to see if the
                                # intersection at the end is available.
                                for (int_start, int_end) in node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][(new_x, new_y)]:
                                    if int_start <= latest_end and earliest_start+self.hardware_params.shuttle_duration <= int_end:
                                        earliest_arrival = max(int_start, earliest_start + self.hardware_params.shuttle_duration)
                                        state_new = State(
                                            x=new_x,
                                            y=new_y,
                                            component=DeviceComponent.SHUTTLE_INTERSECTION,
                                            t_start=int_start,
                                            t_end=int_end,
                                            cur_task=state.cur_task,
                                        )
                                        transition_cost = earliest_arrival - cost_to_node[state]
                                        add_state(state, state_new, transition_cost)
                # Possible emplacement
                for component in DeviceComponent:
                    if DeviceComponent(component.value) != state.component and DeviceComponent(component.value) in node_safe_intervals:
                        for (start, end) in node_safe_intervals[DeviceComponent(component.value)][(state.x, state.y)]:
                            earliest_arrival = max(start, cost_to_node[state] + self.hardware_params.emplace_duration)
                            latest_arrival = min(end, state.t_end + self.hardware_params.emplace_duration)
                            if latest_arrival >= earliest_arrival:
                                state_new = State(
                                    x=state.x,
                                    y=state.y,
                                    component=DeviceComponent(component.value),
                                    t_start=start,
                                    t_end=end,
                                    cur_task=state.cur_task,
                                )
                                transition_cost = earliest_arrival - cost_to_node[state]
                                add_state(state, state_new, transition_cost)
                # Do task (CX)
                if isinstance(state.cur_task, int):
                    if (state.cur_task < len(data_coords)
                        and state.component == DeviceComponent.INTERACTION_ZONE
                        and (state.x, state.y) == data_coords[state.cur_task]
                        and state.t_end >= cost_to_node[state] + get_cx_duration(anc, state.cur_task, len(data_coords))
                    ):
                        state_new = State(
                            x=state.x,
                            y=state.y,
                            component=DeviceComponent.INTERACTION_ZONE,
                            t_start=state.t_start,
                            t_end=state.t_end,
                            cur_task=state.cur_task + 1,
                        )
                        add_state(state, state_new, transition_cost=get_cx_duration(anc, state.cur_task, len(data_coords)))
                else:
                    # cur_task is tuple of bools, each indicating whether that
                    # data qubit has been checked
                    for i,(data_q, coords) in enumerate(zip(data_qubits, data_coords)):
                        if (not state.cur_task[i]
                            and state.component == DeviceComponent.INTERACTION_ZONE
                            and (state.x, state.y) == coords
                            and state.t_end >= cost_to_node[state] + get_cx_duration(anc, state.cur_task, len(data_coords))
                        ):
                            new_cur_task = tuple(s if j != i else True for j,s in enumerate(state.cur_task))
                            state_new = State(
                                x=state.x,
                                y=state.y,
                                component=DeviceComponent.INTERACTION_ZONE,
                                t_start=state.t_start,
                                t_end=state.t_end,
                                cur_task=new_cur_task,
                            )
                            add_state(state, state_new, transition_cost=get_cx_duration(anc, state.cur_task, len(data_coords)))
            if not found_solution:
                raise RuntimeError(f'Unable to find solution for qubit {anc}')
            if not self.debug:
                print('.', end='')
        if not self.debug:
            print()

        return schedule

    def _compile_SE_schedule_tiled(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            anc_basis_filter: str | None = None,
            buffer_time: int = 0,
            prior_schedule: CompiledShuttlingSchedule | None = None,
            allow_interleave_with_prior: bool = False,
            debug_qubit: int | None = None,
        ) -> CompiledShuttlingSchedule:
        """Compile a shuttling schedule for a syndrome extraction round. Assumes
        data qubits are already initialized and fixed in place within each unit
        cell.
        """
        ########################################################################
        # Checking inputs and setting up variables
        ########################################################################
        if len(set(static_data_coords.values())) < len(static_data_coords):
            raise ValueError("Static data qubit positions must be unique.")
        if any(coord < 0 for coords in static_data_coords.values() for coord in coords):
            raise ValueError("Negative data coordinates not supported")

        if not code.ancilla_reference_positions:
            raise ValueError('Code must have ancilla_reference_positions defined to use _compile_SE_schedule_tiled!')

        if anc_basis_filter == 'X':
            anc_qubits = code.X_ancilla_indices
        elif anc_basis_filter == 'Z':
            anc_qubits = code.Z_ancilla_indices
        else:
            raise NotImplementedError('Scheduling X and Z ancillae together is not yet implemented')
            anc_qubits = list(sorted(code.X_ancilla_indices + code.Z_ancilla_indices))

        # TODO: check that each ancilla's set of data qubits is tileable

        all_qubits = list(sorted(code.data_indices + anc_qubits))

        if len(static_data_coords) > 0 and set(static_data_coords) != set(code.data_indices):
            raise ValueError('Must supply all or no data indices')
        if len(static_data_coords) == 0:
            raise NotImplementedError('Auto computing data positions not supported yet')

        if cx_layers:
            init_ancilla_coords, cx_layers = self._map_init_ancilla_positions(static_data_coords, code, cx_layers)
            init_ancilla_coords = {a:c for a,c in init_ancilla_coords.items() if a in anc_qubits}
            assert set(init_ancilla_coords.keys()) == set(anc_qubits)
            ancilla_data_to_visit = dict()
            for layer in cx_layers:
                for qa,qb in layer:
                    if qa in anc_qubits:
                        ancilla_data_to_visit.setdefault(qa, []).append(qb)
                    elif qb in anc_qubits:
                        ancilla_data_to_visit.setdefault(qb, []).append(qa)
                    else:
                        continue
        else:
            init_ancilla_coords = None
            ancilla_data_to_visit = dict()
            for hero_anc,checks in zip(code.X_ancilla_indices + code.Z_ancilla_indices, code.X_checks + code.Z_checks):
                ancilla_data_to_visit[hero_anc] = checks

        # Pick hero ancilla
        max_deg = 0
        hero_anc = -1
        for anc in anc_qubits:
            deg = len(ancilla_data_to_visit[anc])
            if deg > max_deg:
                hero_anc = anc
                max_deg = deg
        assert hero_anc > 1

        # Variables used in compilation
        if prior_schedule:
            schedule = prior_schedule
        else:
            schedule = CompiledShuttlingSchedule(sorted(code.data_indices + code.X_ancilla_indices + code.Z_ancilla_indices), code.data_indices)
        node_safe_intervals, edge_safe_intervals = schedule.get_safe_intervals(
            self.all_coords,
            self.all_edges,
            allow_interleaving=allow_interleave_with_prior,
            buffer_time=buffer_time,
        )
        if init_ancilla_coords:
            schedule.append_instr(
                Instantiate(
                    self.hardware_params.init_duration,
                    anc_qubits,
                    [init_ancilla_coords[anc] for anc in anc_qubits],
                ), 0
            )
            schedule.append_instr(
                Gate(
                    self.hardware_params.h_duration,
                    code.X_ancilla_indices,
                    GateName.H,
                ), schedule.total_duration()
            )

        ########################################################################
        # Helper functions
        ########################################################################
        def distance(coordsA, coordsB):
            return abs(coordsA[0] - coordsB[0]) + abs(coordsA[1] - coordsB[1])

        tsp_results = dict()
        def solve_tsp_memoized(cur_coords, coords_to_reach):
            # Canonicalize coordinates b/c relative distances are the important
            # part
            xc,yc = cur_coords
            coords_canonicalized = [(0,0)]
            for x,y in coords_to_reach:
                coords_canonicalized.append((x-xc, y-yc))
            key = tuple(coords_canonicalized)
            if key in tsp_results:
                return tsp_results[key]
            else:
                distance_matrix = np.array([[distance(a, b) for a in [cur_coords] + coords_to_reach] for b in [cur_coords] + coords_to_reach])
                distance_matrix[:, 0] = 0
                permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
                tsp_results[key] = permutation
                return permutation

        def get_cx_duration(anc: int, task_idx: int, num_tasks: int) -> int:
            duration = self.hardware_params.cx_duration
            if refocus_shuttle_noise and anc in code.Z_ancilla_indices:
                if task_idx == num_tasks:
                    raise ValueError('All tasks complete - don\'t ask for CX duration!')
                if task_idx == 0 or task_idx == num_tasks-1:
                    duration += self.hardware_params.h_duration
                else:
                    duration += 2*self.hardware_params.h_duration
            return duration

        def get_cx_ops(anc: int, data: int, data_idx_in_stab: int, stab_weight: int) -> list[Gate]:
            ops = []
            if refocus_shuttle_noise and anc in code.Z_ancilla_indices:
                if data_idx_in_stab != 0:
                    ops.append(Gate(self.hardware_params.h_duration, [anc], GateName.H))
                ops.append(Gate(self.hardware_params.cx_duration, [data, anc], GateName.CX))
                if data_idx_in_stab != stab_weight-1:
                    ops.append(Gate(self.hardware_params.h_duration, [anc], GateName.H))
            else:
                if anc in code.Z_ancilla_indices:
                    ops.append(Gate(self.hardware_params.cx_duration, [data, anc], GateName.CX))
                else:
                    assert anc in code.X_ancilla_indices
                    ops.append(Gate(self.hardware_params.cx_duration, [anc, data], GateName.CX))
            
            # Debug check
            if sum(op.duration for op in ops) != get_cx_duration(anc, data_idx_in_stab, stab_weight):
                pass
            assert sum(op.duration for op in ops) == get_cx_duration(anc, data_idx_in_stab, stab_weight)

            return ops

        def get_heuristic(anc, target_indices, target_coords):
            # Heuristic roughly approximates total remaining time to complete
            # data checks and readout if there are no obstacles.
            def heuristic(state: State):
                cur_task = state.cur_task
                if isinstance(cur_task, int):
                    # Fixed order of tasks
                    if cur_task == len(target_coords):
                        if state.component == DeviceComponent.READOUT:
                            if state.t_end - cost_to_node[state] >= self.hardware_params.measure_duration:
                                return 0
                            else:
                                return 2*self.hardware_params.emplace_duration
                        else:
                            return self.hardware_params.emplace_duration
                    else:
                        cost = 0
                        if (state.component == DeviceComponent.INTERACTION_ZONE or state.component == DeviceComponent.READOUT) and (state.x, state.y) != target_coords[cur_task]:
                            cost += self.hardware_params.emplace_duration
                        if state.component == DeviceComponent.INTERACTION_ZONE and (state.x, state.y) == target_coords[cur_task]:
                            cost += get_cx_duration(anc, cur_task, len(target_indices))
                            cur_task += 1
                            cost += self.hardware_params.emplace_duration
                        # Time to shuttle to remaining tasks and do CXs
                        cur = state.x, state.y
                        for tgt_idx,tgt in list(enumerate(target_coords))[cur_task:]:
                            cost += distance(tgt, cur)*self.hardware_params.shuttle_duration + get_cx_duration(anc, tgt_idx, len(target_indices)) + 2*self.hardware_params.emplace_duration
                            cur = tgt
                        return cost
                else:
                    # Can choose order of tasks
                    if all(cur_task):
                        if state.component == DeviceComponent.READOUT:
                            if state.t_end - cost_to_node[state] >= self.hardware_params.measure_duration:
                                return 0
                            else:
                                return 2*self.hardware_params.emplace_duration
                        else:
                            return self.hardware_params.emplace_duration
                    else:
                        assert not isinstance(state.cur_task, int)
                        tasks_completed = sum(cur_task)
                        cost = 0
                        coords_to_reach = [c for i,c in enumerate(target_coords) if not state.cur_task[i]]
                        if (state.component == DeviceComponent.INTERACTION_ZONE or state.component == DeviceComponent.READOUT) and (state.x, state.y) not in coords_to_reach:
                            cost += self.hardware_params.emplace_duration
                        if state.component == DeviceComponent.INTERACTION_ZONE and (state.x, state.y) in coords_to_reach:
                            cost += get_cx_duration(anc, tasks_completed, len(target_indices))
                            cur_task = tuple(True if c == (state.x, state.y) else b for b,c in zip(cur_task, target_coords))
                            cost += self.hardware_params.emplace_duration
                        
                        # Time to shuttle to remaining tasks and do CXs
                        cur = state.x, state.y
                        permutation = solve_tsp_memoized(cur, coords_to_reach)
                        for i in permutation[1:]:
                            tgt = coords_to_reach[i-1]
                            cost += distance(tgt, cur)*self.hardware_params.shuttle_duration + get_cx_duration(anc, tasks_completed+i-1, len(target_indices)) + 2*self.hardware_params.emplace_duration
                            cur = tgt
                        return cost
            return heuristic

        ########################################################################
        # Optimization
        # TODO: can make this much simpler for this case. Currently just copied
        # from greedy method, but we can just use a single TSP call to decide
        # the shuttling steps. Don't need to bother with A* search because we
        # dont expect any obstacles.
        ########################################################################
        data_qubits = ancilla_data_to_visit[hero_anc]
        data_coords = [static_data_coords[data] for data in data_qubits]
        if self.debug:
            self.printd('Optimizing ancilla qubit', hero_anc, 'on data qubits', data_qubits, 'at coordinates', data_coords)
        heuristic = get_heuristic(hero_anc, data_qubits, data_coords)
        frontier = set() # TODO: can use min-heap or pqueue
        frontier_heap = []
        if init_ancilla_coords:
            x,y = init_ancilla_coords[hero_anc]
            for (int_start, int_end) in node_safe_intervals[DeviceComponent.READOUT][(x,y)]:
                if int_end - int_start < self.hardware_params.init_duration:
                    continue
                state_new = State(
                    x=x,
                    y=y,
                    component=DeviceComponent.READOUT,
                    t_start=int_start,
                    t_end=int_end,
                    cur_task=(0 if cx_layers else (False,)*len(data_qubits)),
                )
                frontier.add(state_new)
                heapq.heappush(frontier_heap, (heuristic(state_new), state_new))
        else:
            # Free to choose where and when to start
            for coords in [(x,y) for x in range(self.w) for y in range(self.h)]:
                for (int_start, int_end) in node_safe_intervals[DeviceComponent.READOUT][coords]:
                    if int_end - int_start < self.hardware_params.init_duration:
                        continue
                    state_new = State(
                        x=coords[0],
                        y=coords[1],
                        component=DeviceComponent.READOUT,
                        t_start=int_start,
                        t_end=int_end,
                        cur_task=(0 if cx_layers else (False,)*len(data_qubits)),
                    )
                    frontier.add(state_new)
                    heapq.heappush(frontier_heap, (heuristic(state_new), state_new))
        start_states = deepcopy(frontier)
        came_from = {}
        cost_to_node = {
            s:
            self.hardware_params.init_duration+(self.hardware_params.h_duration if hero_anc in code.X_ancilla_indices else 0)+s.t_start
            for s in frontier
        }
        node_heuristic_vals = {s: heuristic(s) for s in frontier}

        # self.printd(anc, (x,y), data_coords)

        def add_state(state: State, state_new: State, transition_cost: int):
            assert transition_cost > 0
            self.printd(f'\tADD {(state_new.x, state_new.y, str(state_new.component), state_new.cur_task)} FROM {(state.x, state.y, str(state.component), state.cur_task)} WITH COST', transition_cost)
            new_cost_to_node = max(cost_to_node[state] + transition_cost, state_new.t_start)
            if state_new in cost_to_node:
                self.printd(f'\t\ttalready seen, cost {cost_to_node[state_new]}, new cost {new_cost_to_node}')
            if cost_to_node.get(state_new, MAX_T) > new_cost_to_node:
                cost_to_node[state_new] = new_cost_to_node
                if state not in start_states and came_from[state] == state_new:
                    raise RuntimeError
                came_from[state_new] = state
                node_heuristic_vals[state_new] = cost_to_node[state_new] + heuristic(state_new)
                if state_new not in frontier:
                    frontier.add(state_new)
                    heapq.heappush(frontier_heap, (node_heuristic_vals[state_new], state_new))
                self.printd('\t\testimated cost:', node_heuristic_vals[state_new])
            else:
                self.printd('\t\tpass')

        def trace_path(state) -> list[State]:
            state_path = [state]
            prev_state = state
            while prev_state in came_from:
                prev_state = came_from[prev_state]
                state_path.append(copy(prev_state))
            state_path = list(reversed(state_path))
            return state_path

        data_by_visit_order = []
        hero_anc_start_loc = init_ancilla_coords[hero_anc] if init_ancilla_coords else None
        found_solution = False
        while frontier:
            _,state = heapq.heappop(frontier_heap)
            frontier.remove(state)

            self.printd(f'POP {(state.x, state.y, str(state.component), state.cur_task)} with cost {node_heuristic_vals[state]} ({cost_to_node[state]} + {heuristic(state)})')
            if node_heuristic_vals[state] - cost_to_node[state] == 0:
                # DONE

                # TODO: recalculate cost_to_node so that we push the idling
                # as early as possible rather than the current approach of
                # as late as possible

                # Trace optimized path to build schedule
                state_path = trace_path(state)

                if not init_ancilla_coords:
                    schedule.append_instr(Instantiate(
                        self.hardware_params.init_duration,
                        [hero_anc],
                        [(state_path[0].x, state_path[0].y)],
                    ), state_path[0].t_start)
                    if hero_anc in code.X_ancilla_indices:
                        schedule.append_instr(Gate(
                            self.hardware_params.h_duration,
                            [hero_anc],
                            GateName.H,
                        ), state_path[0].t_start + self.hardware_params.init_duration)
                    hero_anc_start_loc = (state_path[0].x, state_path[0].y)

                for si,s in enumerate(state_path):
                    self.printd(cost_to_node[s], s)
                    if si == len(state_path)-1:
                        continue
                    ss = state_path[si+1]
                    if (ss.x, ss.y) != (s.x, s.y):
                        assert s.component == DeviceComponent.SHUTTLE_INTERSECTION and ss.component == DeviceComponent.SHUTTLE_INTERSECTION, (s, ss)
                        schedule.append_instr(Shuttle(
                            self.hardware_params.shuttle_duration,
                            hero_anc,
                            (s.x, s.y),
                            (ss.x, ss.y),
                        ), cost_to_node[ss] - self.hardware_params.shuttle_duration)
                    elif s.component != ss.component:
                        assert (s.x, s.y) == (ss.x, ss.y)
                        schedule.append_instr(EmplaceDisplace(
                            self.hardware_params.emplace_duration,
                            hero_anc,
                            (s.x, s.y),
                            s.component,
                            ss.component,
                        ), cost_to_node[ss] - self.hardware_params.emplace_duration)
                    elif s.cur_task != ss.cur_task:
                        if isinstance(s.cur_task, int):
                            idx = s.cur_task
                            task_number = idx
                        else:
                            assert not isinstance(ss.cur_task, int)
                            idxs = [i for i,(b,bb) in enumerate(zip(s.cur_task, ss.cur_task)) if b != bb]
                            assert len(idxs) == 1
                            idx = idxs[0]
                            task_number = sum(s.cur_task)
                        cx_ops = get_cx_ops(hero_anc, data_qubits[idx], task_number, len(data_qubits))
                        cx_ops_dur = get_cx_duration(hero_anc, task_number, len(data_qubits))
                        assert cx_ops_dur == sum(op.duration for op in cx_ops)
                        if hero_anc == debug_qubit:
                            print(hero_anc, idx, task_number, len(cx_ops))
                            for op in cx_ops:
                                print('\t', op)
                        t = cost_to_node[ss] - cx_ops_dur
                        for op in cx_ops:
                            schedule.append_instr(op, t)
                            t += op.duration
                        data_by_visit_order.append(data_qubits[idx])
                assert state_path[-1].component == DeviceComponent.READOUT
                if hero_anc in code.X_ancilla_indices:
                    schedule.append_instr(Gate(
                        self.hardware_params.h_duration,
                        [hero_anc],
                        GateName.H,
                    ), cost_to_node[state])
                schedule.append_instr(Measure(
                    self.hardware_params.measure_duration,
                    [hero_anc],
                    [(state_path[-1].x, state_path[-1].y)],
                ), cost_to_node[state] + self.hardware_params.h_duration)

                node_safe_intervals, edge_safe_intervals = schedule.update_safe_intervals(
                    node_safe_intervals=node_safe_intervals,
                    edge_safe_intervals=edge_safe_intervals,
                    qubit=hero_anc,
                    buffer_time=buffer_time,
                    allow_interleaving=True,
                )

                found_solution = True
                break

            # Shuttling steps
            if state.component == DeviceComponent.SHUTTLE_INTERSECTION:
                for dx,dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    new_x,new_y = (state.x + dx, state.y + dy)
                    if not (0 <= new_x < self.w and 0 <= new_y < self.h):
                        continue
                    shuttle_edge = sort(((state.x, state.y), (new_x,new_y)))
                    for (edge_start, edge_end) in edge_safe_intervals[shuttle_edge]:
                        earliest_start = max(edge_start, cost_to_node[state])
                        latest_end = min(edge_end, state.t_end + self.hardware_params.shuttle_duration)
                        if latest_end - earliest_start >= self.hardware_params.shuttle_duration:
                            # We can shuttle through the intersection in
                            # this interval. Now we need to see if the
                            # intersection at the end is available.
                            for (int_start, int_end) in node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][(new_x, new_y)]:
                                if int_start <= latest_end and earliest_start+self.hardware_params.shuttle_duration <= int_end:
                                    earliest_arrival = max(int_start, earliest_start + self.hardware_params.shuttle_duration)
                                    state_new = State(
                                        x=new_x,
                                        y=new_y,
                                        component=DeviceComponent.SHUTTLE_INTERSECTION,
                                        t_start=int_start,
                                        t_end=int_end,
                                        cur_task=state.cur_task,
                                    )
                                    transition_cost = earliest_arrival - cost_to_node[state]
                                    add_state(state, state_new, transition_cost)
            # Possible emplacement
            for component in DeviceComponent:
                if DeviceComponent(component.value) != state.component and DeviceComponent(component.value) in node_safe_intervals:
                    for (start, end) in node_safe_intervals[DeviceComponent(component.value)][(state.x, state.y)]:
                        earliest_arrival = max(start, cost_to_node[state] + self.hardware_params.emplace_duration)
                        latest_arrival = min(end, state.t_end + self.hardware_params.emplace_duration)
                        if latest_arrival >= earliest_arrival:
                            state_new = State(
                                x=state.x,
                                y=state.y,
                                component=DeviceComponent(component.value),
                                t_start=start,
                                t_end=end,
                                cur_task=state.cur_task,
                            )
                            transition_cost = earliest_arrival - cost_to_node[state]
                            add_state(state, state_new, transition_cost)
            # Do task (CX)
            if isinstance(state.cur_task, int):
                if (state.cur_task < len(data_coords)
                    and state.component == DeviceComponent.INTERACTION_ZONE
                    and (state.x, state.y) == data_coords[state.cur_task]
                    and state.t_end >= cost_to_node[state] + get_cx_duration(hero_anc, state.cur_task, len(data_coords))
                ):
                    state_new = State(
                        x=state.x,
                        y=state.y,
                        component=DeviceComponent.INTERACTION_ZONE,
                        t_start=state.t_start,
                        t_end=state.t_end,
                        cur_task=state.cur_task + 1,
                    )
                    add_state(state, state_new, transition_cost=get_cx_duration(hero_anc, state.cur_task, len(data_coords)))
            else:
                # cur_task is tuple of bools, each indicating whether that
                # data qubit has been checked
                for i,(data_q, coords) in enumerate(zip(data_qubits, data_coords)):
                    if (not state.cur_task[i]
                        and state.component == DeviceComponent.INTERACTION_ZONE
                        and (state.x, state.y) == coords
                        and state.t_end >= cost_to_node[state] + get_cx_duration(hero_anc, state.cur_task, len(data_coords))
                    ):
                        new_cur_task = tuple(s if j != i else True for j,s in enumerate(state.cur_task))
                        state_new = State(
                            x=state.x,
                            y=state.y,
                            component=DeviceComponent.INTERACTION_ZONE,
                            t_start=state.t_start,
                            t_end=state.t_end,
                            cur_task=new_cur_task,
                        )
                        add_state(state, state_new, transition_cost=get_cx_duration(hero_anc, state.cur_task, len(data_coords)))
        if not found_solution:
            raise RuntimeError(f'Unable to find solution for qubit {hero_anc}')
        
        print('Solved tiled solution!')

        ########################################################################
        # Copying schedule to the rest of the ancilla qubits
        ########################################################################
        assert hero_anc_start_loc
        hero_x, hero_y = code.ancilla_reference_positions[hero_anc]
        hero_data_coords_normalized = [(static_data_coords[data][0]-hero_x, static_data_coords[data][1]-hero_y) for data in data_by_visit_order]
        hero_instructions = []
        hero_instr_times = []
        for i,instr in enumerate(schedule.instructions):
            if hero_anc in instr.qubits_list():
                hero_instructions.append(instr)
                hero_instr_times.append(schedule.instruction_start_times[i])
        for anc in anc_qubits:
            if anc == hero_anc:
                continue
            # Order the data qubits in the same way as hero_anc
            # TODO: this normalization method isn't valid for edge ancillae; not
            # guaranteed that min is the same. Need a better way of mapping the
            # data to each other. Should also be resilient - for surface code,
            # each layout is a square, so both edges would be "normalized" to
            # the same thing
            anc_x, anc_y = code.ancilla_reference_positions[anc]
            anc_data_coords_normalized = [(static_data_coords[data][0]-anc_x, static_data_coords[data][1]-anc_y) for data in ancilla_data_to_visit[anc]]
            assert set(anc_data_coords_normalized).issubset(set(hero_data_coords_normalized))
            anc_data_ordered = []
            hero_q_to_anc_q = {hero_anc: anc}
            data_anc_offset = None
            for hero_data,hero_data_coords in zip(data_by_visit_order, hero_data_coords_normalized):
                found = False
                for data,data_coords in zip(ancilla_data_to_visit[anc], anc_data_coords_normalized):
                    if hero_data_coords == data_coords:
                        anc_data_ordered.append(data)
                        hero_q_to_anc_q[hero_data] = data
                        true_data_coords = static_data_coords[data]
                        # offset_from_anc = # TODO

                        found = True
                        break
                if not found:
                    anc_data_ordered.append(None)
            assert len([d for d in anc_data_ordered if d is not None]) == len(ancilla_data_to_visit[anc])
                
            def coord_transform(hero_coords):
                x,y = hero_coords
                anc_coords = (x - hero_x + anc_x, y - hero_y + anc_y)
                return anc_coords

            for instr,start_time in zip(hero_instructions, hero_instr_times):
                if isinstance(instr, Instantiate):
                    schedule.append_instr(
                        Instantiate(
                            instr.duration,
                            [hero_q_to_anc_q[q] for q in instr.qubits],
                            [coord_transform(c) for c in instr.cell_coords],
                        ),
                        start_time
                    )
                elif isinstance(instr, Gate):
                    if not all(q in hero_q_to_anc_q for q in instr.qubits):
                        continue
                    schedule.append_instr(
                        Gate(
                            instr.duration,
                            [hero_q_to_anc_q[q] for q in instr.qubits],
                            instr.name,
                        ),
                        start_time
                    )
                elif isinstance(instr, Shuttle):
                    schedule.append_instr(
                        Shuttle(
                            instr.duration,
                            hero_q_to_anc_q[instr.qubit],
                            coord_transform(instr.start_coords),
                            coord_transform(instr.end_coords),
                        ),
                        start_time
                    )
                elif isinstance(instr, EmplaceDisplace):
                    schedule.append_instr(
                        EmplaceDisplace(
                            instr.duration,
                            hero_q_to_anc_q[instr.qubit],
                            coord_transform(instr.coords),
                            instr.start_component,
                            instr.end_component,
                        ),
                        start_time
                    )
                elif isinstance(instr, Measure):
                    schedule.append_instr(
                        Measure(
                            instr.duration,
                            [hero_q_to_anc_q[q] for q in instr.qubits],
                            [coord_transform(c) for c in instr.cell_coords],
                        ),
                        start_time
                    )
                else:
                    raise RuntimeError('Unexpected instruction', instr, 'of type', type(instr))

        # TODO: re-optimizing boundary qubits to reduce unnecessary shuttling
        # TODO: BB code does not cleanly tile - need a different approach

        return schedule

    def _map_init_ancilla_positions(
            self,
            data_positions: dict[int, tuple[int, int]],
            code: QECCode,
            cx_layers: list[list[tuple[int, int]]],
        ) -> tuple[dict[int, tuple[int, int]], list[list[tuple[int, int]]]]:
        """Given static positions of data qubits, determine where to initially
        place ancilla qubits at the start of a syndrome extraction round.
        """
        if not cx_layers:
            # TODO: could also do this dynamically while scheduling (route to
            # the best next data qubit at the time, taking into account all
            # scheduled ancilla paths)
            checks = []
            for check_idx, datas in enumerate(code.X_checks):
                checks += [(code.X_ancilla_indices[check_idx], d) for d in datas]
            for check_idx, datas in enumerate(code.Z_checks):
                checks += [(d, code.Z_ancilla_indices[check_idx]) for d in datas]
            graph = nx.Graph(checks)

            # compute edge coloring to determine CX layers
            coloring = nx.coloring.greedy_color(nx.line_graph(graph), strategy='largest_first')
            num_colors = max(coloring.values()) + 1

            cx_layers = [[] for _ in range(num_colors)]
            for edge, color in coloring.items():
                q0,q1 = edge
                cx_layers[color].append((q0,q1) if edge in checks else (q1,q0))
        
        # Put each ancilla with the data qubit it first interacts with.
        ancilla_first_data: dict[int, int] = dict()
        for layer in cx_layers:
            for qa,qb in layer:
                if qa in data_positions and qb not in ancilla_first_data:
                    ancilla_first_data[qb] = qa
                if qb in data_positions and qa not in ancilla_first_data:
                    ancilla_first_data[qa] = qb

        ancilla_positions: dict[int, tuple[int, int]] = dict()
        blocked_anc = []
        for anc, data in ancilla_first_data.items():
            if data_positions[data] not in ancilla_positions.values():
                ancilla_positions[anc] = data_positions[data]
            else:
                blocked_anc.append(anc)
        available_coords = [(x,y) for x in range(self.w) for y in range(self.h) if (x,y) not in ancilla_positions.values()]
        for anc in blocked_anc:
            data = ancilla_first_data[anc]
            d_coords = data_positions[data]
            # find nearest unmapped coordinate
            best = min(available_coords, key=(lambda c: abs(c[0]-d_coords[0]) + abs(c[1]-d_coords[1])))
            ancilla_positions[anc] = best
            available_coords.remove(best)

        return ancilla_positions, cx_layers
    
    def printd(self, *args):
        if self.debug:
            print(*args)

def circ_dag(circ) -> nx.DiGraph:
    dag = nx.DiGraph()
    meas_history = []
    last_instr_by_qubit: dict[int, int] = dict()
    for i,instr in enumerate(circ):
        dag.add_node(i)
        if instr.name == 'DETECTOR' or instr.name == 'OBSERVABLE_INCLUDE':
            for target in set(instr.targets_copy()):
                assert target.value < 0
                qubit = meas_history[target.value]
                if qubit in last_instr_by_qubit:
                    if last_instr_by_qubit[qubit] != i:
                        dag.add_edge(last_instr_by_qubit[qubit], i)
            last_instr_by_qubit[qubit] = i
        else:
            if instr.name in ['M', 'MX']:
                for t in instr.targets_copy():
                    qubit = t.qubit_value
                    assert qubit is not None
                    meas_history.append(qubit)
            for target in set(instr.targets_copy()):
                assert target.is_qubit_target
                qubit = target.qubit_value
                if qubit in last_instr_by_qubit:
                    if last_instr_by_qubit[qubit] == i:
                        raise RuntimeError
                    dag.add_edge(last_instr_by_qubit[qubit], i)
                last_instr_by_qubit[qubit] = i
    return dag

def instr_targets(instr, ignore_non_qubits = True) -> list[int]:
    tgts = []
    for tgt in instr.targets_copy():
        if ignore_non_qubits and not tgt.is_qubit_target:
            continue
        assert tgt.is_qubit_target
        tgts.append(tgt.qubit_value)
    return tgts

def simplify_stim_circuit(circ) -> stim.Circuit:
    # 1. Combine errors on the same qubit
    last_instr_per_qubit = dict()
    instr_accumulated_arg = dict()
    new_circ = stim.Circuit()
    for instr in circ:
        if len(instr.targets_copy()) == 1 and ('ERROR' in instr.name or 'DEPOLARIZE' in instr.name):
            target = instr.targets_copy()[0]
            assert len(instr.gate_args_copy()) == 1
            gate_arg = instr.gate_args_copy()[0]
            assert target.is_qubit_target
            qubit = target.qubit_value
            if instr.name == last_instr_per_qubit.get(qubit, None):
                instr_accumulated_arg[qubit].append(gate_arg)
            else:
                # complete old instruction
                if qubit in last_instr_per_qubit:
                    reduced_arg = 1
                    for arg in instr_accumulated_arg[qubit]:
                        reduced_arg *= (1 - arg)
                    reduced_arg = 1 - reduced_arg
                    new_circ.append(last_instr_per_qubit[qubit], qubit, reduced_arg)

                # start new instruction
                last_instr_per_qubit[qubit] = instr.name
                instr_accumulated_arg[qubit] = [gate_arg]
        else:
            for target in instr.targets_copy():
                if target.is_qubit_target:
                    qubit = target.qubit_value
                    if qubit in last_instr_per_qubit:
                        reduced_arg = 1
                        for arg in instr_accumulated_arg[qubit]:
                            reduced_arg *= (1 - arg)
                        reduced_arg = 1 - reduced_arg
                        new_circ.append(last_instr_per_qubit[qubit], qubit, reduced_arg)
                        last_instr_per_qubit.pop(qubit)
                        instr_accumulated_arg.pop(qubit)
            new_circ.append(instr)
    # new_circ = circ
    print(f'Reduced circuit length from {len(circ)} to {len(new_circ)}')

    # 2. Combine instructions that are the same on multiple qubits
    # First, combine gates
    change_made = True
    search_depth = 3
    new_new_circ = stim.Circuit()
    last_circ = new_circ
    while change_made:
        print(f'\nLooping. Current circuit length {len(last_circ)}...')
        dag = circ_dag(last_circ)
        new_circ = stim.Circuit()
        generations = list(nx.topological_generations(dag))
        processed_indices = set()
        unprocessed_indices = set(range(len(last_circ)))
        frontier = list(generations[0])
        change_made = False

        # Iterate through the circuit instructions. Upon reaching a
        # gate/reset/measure, remember it and then continue to iterate through
        # the circuit instructions looking for matching instructions on other
        # qubits.

        while frontier:
            if change_made:
                break
            combined_instrs = []
            combined_qubits = set()
            instr_idx = frontier.pop(0)
            instr = last_circ[instr_idx]
            if instr.name in ['CX', 'CZ', 'R', 'M', 'MX', 'RX', 'H']:
                # Find matching instructions on other qubits
                combined_qubits |= set(instr_targets(instr))
                combined_instrs.append(instr_idx)
                generation_idx = [i for i,gen in enumerate(generations) if instr_idx in gen][0]
                potential_siblings = [i for gen in generations[max(0, generation_idx - search_depth):min(len(generations)-1,  generation_idx + search_depth)] for i in gen if i in unprocessed_indices]
                potential_siblings = [i for i in potential_siblings if not nx.has_path(dag, i, instr_idx) and not nx.has_path(dag, instr_idx, i)]
                while potential_siblings:
                    sib = potential_siblings.pop(0)
                    sib_instr = last_circ[sib]
                    if combined_qubits.intersection(set(instr_targets(sib_instr))):
                        continue
                    if sib_instr.name == instr.name:
                        qs = set(instr_targets(sib_instr))
                        assert not combined_qubits.intersection(qs)
                        combined_qubits |= qs
                        combined_instrs.append(sib)

                        # TODO: need to append all instructions that are topologically
                        # in front of this instruction
                        for anc in nx.ancestors(dag, sib):
                            if anc in unprocessed_indices:
                                new_circ.append(last_circ[anc])
                                processed_indices.add(anc)
                                unprocessed_indices.remove(anc)

                        potential_siblings = [i for i in potential_siblings if not nx.has_path(dag, i, sib) and not nx.has_path(dag, sib, i)]
            if len(combined_instrs) > 1:
                print(f'MERGING instructions {combined_instrs}')
                for i in combined_instrs:
                    print(i, last_circ[i])

                # TODO: check that all instructions topologically in front of
                # this one are already processed
                new_circ.append(instr.name, combined_qubits, ())
                for i in combined_instrs:
                    if i in frontier:
                        frontier.remove(i)
                processed_indices |= set(combined_instrs)
                unprocessed_indices -= set(combined_instrs)
                change_made = True
            else:
                new_circ.append(instr)
                processed_indices.add(instr_idx)
                unprocessed_indices.remove(instr_idx)
            
            for succ in dag.successors(instr_idx):
                if succ not in processed_indices and succ not in frontier:
                    frontier.append(succ)
        
        if change_made:
            # Finish up adding remaining instructions
            for i in [i for gen in list(nx.topological_generations(dag)) for i in gen]:
                if i in unprocessed_indices:
                    new_circ.append(last_circ[i])
        
        last_circ = new_circ.copy()

    return new_circ