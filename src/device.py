from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np
import math
import stim
from copy import copy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.QECCode import QECCode

@dataclass
class HardwareParams:
    T2: float # in seconds
    cx_err: float
    cx_duration: int # in ns
    h_err: float
    h_duration: int # in ns
    shuttle_err: float # per unit cell shuttle
    shuttle_duration: int # per unit cell shuttle, in ns
    emplace_err: float
    emplace_duration: int
    init_err: float
    init_duration: int
    measure_err: float
    measure_duration: int

def hardware_params(
        p: float,
        T2: float = 100e-6,
        p_sh: int = 1e-5,
    ) -> HardwareParams:
    return HardwareParams(
        T2=T2,
        cx_err=p,
        cx_duration=100,
        h_err=p,
        h_duration=100,
        shuttle_err=p_sh,
        shuttle_duration=1000, # 10 m/s
        emplace_err=p_sh/10,
        emplace_duration=100,
        init_err=p,
        init_duration=500,
        measure_err=p,
        measure_duration=500,
    )

default_params = hardware_params(1e-3)

@dataclass
class Instruction:
    duration: int
    error: float

    def qubits_list(self):
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

    def qubits_list(self):
        return self.qubits

@dataclass
class SingleQubitInstruction(Instruction):
    qubit: int

    def qubits_list(self):
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
    READOUT = 'READOUT'
    INTERACTION_ZONE = 'INTERACTION_ZONE'
    SHUTTLE_CHANNEL = 'SHUTTLE_CHANNEL'
    SHUTTLE_INTERSECTION = 'SHUTTLE_INTERSECTION'
    NONEXISTENT = 'NONEXISTENT'

    def __str__(self):
        return self.value

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
class ShuttlingLocation(DeviceLocation):
    edge: tuple[tuple[int, int], tuple[int, int]] # order indicates shuttling direction

@dataclass
class EmplaceDisplace(SingleQubitInstruction):
    # Move a qubit in or out of a readout zone or an interaction zone.
    coords: tuple[int, int]
    start_component: DeviceComponent
    end_component: DeviceComponent
    
@dataclass
class Idle(SingleQubitInstruction):
    coords: tuple[int, int]
    component: DeviceComponent

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
        if isinstance(other, self.__class__):
            new_sched = CompiledShuttlingSchedule(list(sorted(set(self.qubits) | set(other.qubits))), list(sorted(set(self.data_qubits) | set(other.data_qubits))))
            for instr, t in zip(self.instructions, self.instruction_start_times):
                new_sched.append_instr(instr, t)
            for instr, t in zip(other.instructions, other.instruction_start_times):
                offset = self.total_duration()
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

        return frames

    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the compiled shuttling schedule back into a Stim circuit."""
        raise NotImplementedError()


class SchedulingMethod(Enum):
    GREEDY = 'GREEDY' # no lookahead, schedule a shuttle one edge at a time

@dataclass
class State:
    x: int
    y: int
    component: DeviceComponent
    t_start: int
    t_end: int
    cur_task: int

    def __hash__(self):
        return hash((self.x, self.y, self.component, self.t_start, self.t_end, self.cur_task))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __str__(self):
        string = self.__class__.__name__ + '('
        for key,val in self.__dict__.items():
            if isinstance(val, float):
                string += f'{key}={val:3g}, '
            else:
                string += f'{key}={val}, '
        string = string[:-2] + ')'
        return string

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

    def compile_QEC_schedule(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            rounds: int,
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod = SchedulingMethod.GREEDY,
        ):
        schedule = CompiledShuttlingSchedule(sorted(list(static_data_coords.keys())), sorted(list(static_data_coords.keys())))
        schedule.append_instr(
            Instantiate(
                self.hardware_params.init_duration,
                self.hardware_params.init_err,
                code.data_indices,
                [static_data_coords[data] for data in code.data_indices],
            ), 0
        )
        for q in code.data_indices:
            schedule.append_instr(
                EmplaceDisplace(
                    self.hardware_params.emplace_duration,
                    self.hardware_params.emplace_err,
                    q,
                    static_data_coords[q],
                    DeviceComponent.READOUT,
                    DeviceComponent.INTERACTION_ZONE,
                ), self.hardware_params.init_duration,
            )
        SE_sched = self.compile_SE_schedule_greedy(
            code,
            static_data_coords,
            cx_layers,
            use_highways,
            refocus_shuttle_noise,
            scheduling_method,
        )
        schedule += rounds * SE_sched
        t = schedule.total_duration()
        for q in code.data_indices:
            schedule.append_instr(
                EmplaceDisplace(
                    self.hardware_params.emplace_duration,
                    self.hardware_params.emplace_err,
                    q,
                    static_data_coords[q],
                    DeviceComponent.READOUT,
                    DeviceComponent.INTERACTION_ZONE,
                ), t,
            )
        schedule.append_instr(
            Measure(
                self.hardware_params.init_duration,
                self.hardware_params.init_err,
                code.data_indices,
                [static_data_coords[data] for data in code.data_indices],
            ), schedule.total_duration()
        )
        return schedule

    def compile_SE_schedule_greedy(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod = SchedulingMethod.GREEDY,
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

        all_qubits = set(code.data_indices + code.X_ancilla_indices + code.Z_ancilla_indices)
        all_qubits = list(sorted(all_qubits))
        anc_qubits = list(sorted(code.X_ancilla_indices + code.Z_ancilla_indices))
        print(anc_qubits)

        if len(static_data_coords) > 0 and set(static_data_coords) != set(code.data_indices):
            raise ValueError('Must supply all or no data indices')
        if len(static_data_coords) == 0:
            raise NotImplementedError('Auto computing data positions not supported yet')

        init_ancilla_coords, cx_layers = self._map_init_ancilla_positions(static_data_coords, code, cx_layers)
        assert set(init_ancilla_coords.keys()) == set(anc_qubits)
        ancilla_data_to_visit = dict()
        for layer in cx_layers:
            for qa,qb in layer:
                if qa in anc_qubits:
                    ancilla_data_to_visit.setdefault(qa, []).append(qb)
                elif qb in anc_qubits:
                    ancilla_data_to_visit.setdefault(qb, []).append(qa)
                else:
                    raise ValueError

        def sort(edge: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
            u,v = sorted(edge)
            return (u,v)

        # Variables used in compilation
        remaining_ancilla = set(anc_qubits)
        all_coords = [(x,y) for x in range(self.w) for y in range(self.h)]
        all_edges = [sort((c0, c1)) for c0 in all_coords for c1 in all_coords if c0 != c1]
        MAX_T = 10**10
        edge_safe_intervals: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {e:[(0, MAX_T)] for e in all_edges}
        node_safe_intervals: dict[DeviceComponent, dict[tuple[int, int], list[tuple[int, int]]]] = {
            DeviceComponent.SHUTTLE_INTERSECTION: {c:[(0, MAX_T)] for c in all_coords},
            DeviceComponent.INTERACTION_ZONE: {c:[(0, MAX_T)] for c in all_coords},
            DeviceComponent.READOUT: {c:[(0, MAX_T)] for c in all_coords},
        }
        schedule = CompiledShuttlingSchedule(sorted(list(static_data_coords.keys()) + anc_qubits), sorted(list(static_data_coords.keys())))
        schedule.append_instr(
            Instantiate(
                self.hardware_params.init_duration,
                self.hardware_params.init_err,
                anc_qubits,
                [init_ancilla_coords[anc] for anc in anc_qubits],
            ), 0
        )

        def get_heuristic(target_coords):
            # Heuristic roughly approximates total remaining time to complete
            # data checks and readout if there are no obstacles.
            def heuristic(state: State):
                cur_task = state.cur_task
                if cur_task == len(target_coords):
                    if state.component == DeviceComponent.READOUT:
                        if state.t_end - state.t_start >= self.hardware_params.measure_duration:
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
                        cost += self.hardware_params.cx_duration
                        cur_task += 1
                        cost += self.hardware_params.emplace_duration
                    # Time to shuttle to remaining tasks and do CXs
                    cur_x, cur_y = state.x, state.y
                    for x_target,y_target in target_coords[cur_task:]:
                        # TODO: account for H time if refocusing shuttles
                        cost += (abs(x_target - cur_x)*0.999 + abs(y_target - cur_y))*self.hardware_params.shuttle_duration + self.hardware_params.cx_duration + 2*self.hardware_params.emplace_duration
                        cur_x, cur_y = x_target, y_target
                    return cost
            return heuristic

        if not self.debug:
            print(f'Optimizing {len(remaining_ancilla)} ancilla schedules', end='')
        while remaining_ancilla:
            anc = remaining_ancilla.pop()
            data_qubits = ancilla_data_to_visit[anc]
            data_coords = [static_data_coords[data] for data in data_qubits]
            heuristic = get_heuristic(data_coords)
            x,y = init_ancilla_coords[anc]
            state_start = State(x=x, y=y, component=DeviceComponent.READOUT, t_start=0, t_end=MAX_T, cur_task=0)
            frontier = {state_start} # TODO use min-heap or pqueue with precomputed heuristic values
            came_from = {}
            cost_to_node = {state_start: 500}
            node_heuristic_vals = {state_start: heuristic(state_start)}

            self.printd(anc, (x,y), data_coords)

            def update_safe_intervals(
                    schedule: CompiledShuttlingSchedule,
                ):
                def get_new_intervals(intervals_old: list[tuple[int, int]], t_start: int, instr_duration: int, buffer: int = 0):
                    intervals_new = []
                    for (s,e) in intervals_old:
                        if t_start - buffer < e and s < t_start + instr_duration + buffer:
                            interval0 = (max(s, t_start + instr_duration + buffer), e)
                            if interval0[0] < interval0[1]:
                                intervals_new.append(interval0)
                            interval1 = (s, min(t_start - buffer, e))
                            if interval1[0] < interval1[1]:
                                intervals_new.append(interval1)
                        else:
                            intervals_new.append((s,e))
                    return intervals_new

                # TODO: identify times when each qubit is idling, figure out
                # what component it is in, and update intervals

                for i,(instr, t_start) in enumerate(zip(schedule.instructions, schedule.instruction_start_times)):
                    if isinstance(instr, Shuttle):
                        edge = sort((instr.start_coords, instr.end_coords))
                        edge_safe_intervals[edge] = get_new_intervals(edge_safe_intervals[edge], t_start, instr.duration, buffer=100)
                        node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.start_coords] = get_new_intervals(node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.start_coords], t_start, 0, buffer=100)
                        node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.end_coords] = get_new_intervals(node_safe_intervals[DeviceComponent.SHUTTLE_INTERSECTION][instr.end_coords], t_start + instr.duration, 0, buffer=100)
                    elif isinstance(instr, EmplaceDisplace):
                        # Qubit changing locations
                        qubit_arrival = t_start + instr.duration
                        qubit_departure = None
                        for instr_future, t_start_future in zip(schedule.instructions[i+1:], schedule.instruction_start_times[i+1:]):
                            if instr.qubit in instr_future.qubits_list() and (isinstance(instr_future, EmplaceDisplace) or isinstance(instr_future, Measure)):
                                qubit_departure = t_start_future
                                break
                        assert qubit_departure is not None
                        node_safe_intervals[instr.end_component][instr.coords] = get_new_intervals(node_safe_intervals[instr.end_component][instr.coords], qubit_arrival, qubit_departure-qubit_arrival, buffer=50)
                    elif isinstance(instr, Instantiate):
                        coords = instr.cell_coords
                        for coords in instr.cell_coords:
                            node_safe_intervals[DeviceComponent.READOUT][coords] = get_new_intervals(node_safe_intervals[DeviceComponent.READOUT][coords], t_start, instr.duration, buffer=50)

            def add_state(state, state_new, transition_cost):
                assert transition_cost > 0
                self.printd(f'\tADD {(state_new.x, state_new.y, str(state_new.component), state_new.cur_task)} FROM {(state.x, state.y, str(state.component), state.cur_task)} WITH COST', transition_cost)
                if state_new in cost_to_node:
                    self.printd(f'\t\ttalready seen, cost {cost_to_node[state_new]}, new cost {cost_to_node[state] + transition_cost}')
                if cost_to_node.get(state_new, MAX_T) > cost_to_node[state] + transition_cost:
                    cost_to_node[state_new] = cost_to_node[state] + transition_cost
                    if state != state_start and came_from[state] == state_new:
                        raise RuntimeError
                    came_from[state_new] = state
                    node_heuristic_vals[state_new] = cost_to_node[state_new] + heuristic(state_new)
                    if state_new not in frontier:
                        frontier.add(state_new)
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
                state = min(frontier, key=lambda state: node_heuristic_vals[state])
                self.printd(f'POP {(state.x, state.y, str(state.component), state.cur_task)} with cost {node_heuristic_vals[state]} ({cost_to_node[state]} + {heuristic(state)})')
                if node_heuristic_vals[state] - cost_to_node[state] == 0:
                    # DONE
                    # Trace optimized path to build schedule
                    state_path = trace_path(state)
                    for si,s in enumerate(state_path):
                        self.printd(cost_to_node[s], s)
                        if si == len(state_path)-1:
                            continue
                        ss = state_path[si+1]
                        if (ss.x, ss.y) != (s.x, s.y):
                            assert s.component == DeviceComponent.SHUTTLE_INTERSECTION and ss.component == DeviceComponent.SHUTTLE_INTERSECTION
                            schedule.append_instr(Shuttle(
                                self.hardware_params.shuttle_duration,
                                self.hardware_params.shuttle_err,
                                anc,
                                (s.x, s.y),
                                (ss.x, ss.y),
                            ), cost_to_node[ss] - self.hardware_params.shuttle_duration)
                        elif s.component != ss.component:
                            schedule.append_instr(EmplaceDisplace(
                                self.hardware_params.emplace_duration,
                                self.hardware_params.emplace_err,
                                anc,
                                (s.x, s.y),
                                s.component,
                                ss.component,
                            ), cost_to_node[ss] - self.hardware_params.emplace_duration)
                        elif s.cur_task != ss.cur_task:
                            schedule.append_instr(Gate(
                                self.hardware_params.cx_duration,
                                self.hardware_params.cx_err,
                                [anc, data_qubits[s.cur_task]] if anc in code.X_ancilla_indices else [data_qubits[s.cur_task], anc],
                                GateName.CX,
                            ), cost_to_node[ss] - self.hardware_params.cx_duration)
                    assert state_path[-1].component == DeviceComponent.READOUT
                    # schedule.append_instr(EmplaceDisplace(
                    #     self.hardware_params.emplace_duration,
                    #     self.hardware_params.emplace_err,
                    #     anc,
                    #     (state_path[-1].x, state_path[-1].y),
                    #     state_path[-2].component,
                    #     state_path[-1].component,
                    # ), cost_to_node[state_path[-1]] - self.hardware_params.emplace_duration)
                    schedule.append_instr(Measure(
                        self.hardware_params.measure_duration,
                        self.hardware_params.measure_err,
                        [anc],
                        [(state_path[-1].x, state_path[-1].y)],
                    ), cost_to_node[state])

                    update_safe_intervals(schedule)

                    found_solution = True
                    break
                
                frontier.remove(state)

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
                if (state.cur_task < len(data_coords)
                    and state.component == DeviceComponent.INTERACTION_ZONE
                    and (state.x, state.y) == data_coords[state.cur_task]
                    and state.t_end >= cost_to_node[state] + self.hardware_params.cx_duration):
                    state_new = State(
                        x=state.x,
                        y=state.y,
                        component=DeviceComponent.INTERACTION_ZONE,
                        t_start=state.t_start,
                        t_end=state.t_end,
                        cur_task=state.cur_task + 1,
                    )
                    add_state(state, state_new, transition_cost=self.hardware_params.cx_duration)
            if not found_solution:
                raise RuntimeError(f'Unable to find solution for qubit {anc}')
            if not self.debug:
                print('.', end='')

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