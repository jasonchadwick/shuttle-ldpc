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
class Frame:
    qubit_positions: dict[int, tuple[float, float]]
    twoq_gates: list[tuple[int, int]]
    oneq_gates: list[int]
    active_shuttles: list[tuple[int, int]] # (qubit, direction). 0=up, 1=right, 2=down, 3=left

class CompiledShuttlingSchedule:
    qubits: list[int]
    instructions: list[Instruction]
    instructions_by_qubit: dict[int, list[int]]

    def __init__(self, qubits: list[int]):
        self.qubits = qubits
        self.instructions = []
        self.instructions_by_qubit = {q:[] for q in qubits}

    def append_instr(self, instr: Instruction):
        for q in instr.qubits_list():
            self.instructions_by_qubit[q].append(len(self.instructions))
        self.instructions.append(instr)

    def get_frames(self, frames_per_segment: int = 10) -> list[Frame]:
        """Convert per-qubit schedules to per-frame view, where each qubit"""
        raise NotImplementedError()

    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the compiled shuttling schedule back into a Stim circuit."""
        raise NotImplementedError()

class Direction(Enum):
    UP = 'UP'
    RIGHT = 'RIGHT'
    DOWN = 'DOWN'
    LEFT = 'LEFT'

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

    def __init__(
            self,
            w: int,
            h: int,
            hardware_params: HardwareParams
        ) -> None:
        self.w = w
        self.h = h
        self.hardware_params = hardware_params
    
    def compile_data_init_schedule(
            self,
            code: QECCode,
            static_data_positions: dict[int, tuple[int, int]],
        ):
        raise NotImplementedError

    def compile_data_meas_schedule(
            self,
            code: QECCode,
            static_data_positions: dict[int, tuple[int, int]],
        ):
        raise NotImplementedError

    def compile_SE_schedule_greedy(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod = SchedulingMethod.GREEDY,
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

        all_qubits = set(code.data_indices + code.X_ancilla_indices + code.Z_ancilla_indices)
        all_qubits = list(sorted(all_qubits))
        anc_qubits = list(sorted(code.X_ancilla_indices + code.Z_ancilla_indices))

        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(all_qubits)
        for check_idx, check_qs in enumerate(code.X_checks):
            for data_q in check_qs:
                interaction_graph.add_edge(code.X_ancilla_indices[check_idx], data_q)
        for check_idx, check_qs in enumerate(code.Z_checks):
            for data_q in check_qs:
                interaction_graph.add_edge(code.Z_ancilla_indices[check_idx], data_q)

        if len(static_data_coords) > 0 and set(static_data_coords) != set(code.data_indices):
            raise ValueError('Must supply all or no data indices')
        if len(static_data_coords) == 0:
            raise NotImplementedError('Auto computing data positions not supported yet')

        init_ancilla_coords, cx_layers = self._map_init_ancilla_positions(static_data_coords, code, interaction_graph, cx_layers)
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
        ancilla_schedules: dict[int, list[Instruction]] = {}

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
                    if (state.component == DeviceComponent.INTERACTION_ZONE and (state.x, state.y) != target_coords[cur_task]) or state.component == DeviceComponent.READOUT:
                        cost += self.hardware_params.emplace_duration
                    if state.component == DeviceComponent.INTERACTION_ZONE and (state.x, state.y) == target_coords[cur_task]:
                        cost += self.hardware_params.cx_duration
                        if cur_task == len(target_coords)-1:
                            cur_task += 1
                        else:
                            cost += self.hardware_params.emplace_duration
                    # Time to shuttle to remaining tasks and do CXs
                    cur_x, cur_y = state.x, state.y
                    for x_target,y_target in target_coords[cur_task:]:
                        # TODO: account for H time if refocusing shuttles
                        cost += (abs(x_target - cur_x) + abs(y_target - cur_y))*self.hardware_params.shuttle_duration + self.hardware_params.cx_duration + 2*self.hardware_params.emplace_duration
                        cur_x, cur_y = x_target, y_target
                    cost += self.hardware_params.emplace_duration # emplace into readout
                    return cost
            return heuristic

        while remaining_ancilla:
            anc = remaining_ancilla.pop()
            data_qubits = ancilla_data_to_visit[anc]
            data_coords = [static_data_coords[data] for data in data_qubits]
            heuristic = get_heuristic(data_coords)
            x,y = init_ancilla_coords[anc]
            state_start = State(x=x, y=y, component=DeviceComponent.READOUT, t_start=0, t_end=MAX_T, cur_task=0)
            frontier = {state_start} # TODO use min-heap or pqueue with precomputed heuristic values
            came_from = {}
            cost_to_node = {state_start: 0}
            node_heuristic_vals = {state_start: heuristic(state_start)}

            def add_state(state, state_new, transition_cost):
                assert transition_cost > 0
                print(f'ADD {(state_new.x, state_new.y, str(state_new.component), state_new.cur_task)} FROM {(state.x, state.y, str(state.component), state.cur_task)} WITH COST', transition_cost)
                if state_new in cost_to_node:
                    print(f'\talready seen, cost {cost_to_node[state_new]}')
                if cost_to_node.get(state_new, MAX_T) > cost_to_node[state] + transition_cost:
                    cost_to_node[state_new] = cost_to_node[state] + transition_cost
                    if state != state_start and came_from[state] == state_new:
                        raise RuntimeError
                    came_from[state_new] = state
                    node_heuristic_vals[state_new] = cost_to_node[state_new] + heuristic(state_new)
                    if state_new not in frontier:
                        frontier.add(state_new)

            def trace_path(state):
                state_path = []
                prev_state = state
                while prev_state in came_from:
                    prev_state = came_from[state]
                    state_path.append(copy(prev_state))
                state_path = list(reversed(state_path))
                return state_path

            # Each node is a device intersection and a time
            while frontier:
                state = min(frontier, key=lambda state: cost_to_node[state] + node_heuristic_vals[state])
                if node_heuristic_vals[state] == 0:
                    # DONE
                    for s in trace_path(state):
                        print((s.x, s.y, s.component, cost_to_node[s]))
                    raise NotImplementedError
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
                                    if int_start <= latest_end and earliest_start <= int_end:
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

            pass

        raise NotImplementedError


    def compile_SE_schedule(
            self,
            code: QECCode,
            static_data_coords: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
            scheduling_method: SchedulingMethod = SchedulingMethod.GREEDY,
            debug_qubit: int | None = None,
        ) -> tuple[dict[int, tuple[int, int]], CompiledShuttlingSchedule]:
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

        interaction_graph = nx.Graph()
        interaction_graph.add_nodes_from(all_qubits)
        for check_idx, check_qs in enumerate(code.X_checks):
            for data_q in check_qs:
                interaction_graph.add_edge(code.X_ancilla_indices[check_idx], data_q)
        for check_idx, check_qs in enumerate(code.Z_checks):
            for data_q in check_qs:
                interaction_graph.add_edge(code.Z_ancilla_indices[check_idx], data_q)

        if len(static_data_coords) > 0 and set(static_data_coords) != set(code.data_indices):
            raise ValueError('Must supply all or no data indices')
        if len(static_data_coords) == 0:
            raise NotImplementedError('Auto computing data positions not supported yet')

        init_ancilla_coords, cx_layers = self._map_init_ancilla_positions(static_data_coords, code, interaction_graph, cx_layers)
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

        # Variables used in compilation
        qubit_locations: dict[int, DeviceLocation] = {q:DeviceLocation(DeviceComponent.NONEXISTENT, (-1,-1)) for q in all_qubits}
        remaining_ancilla = set(anc_qubits)
        instr_time_remaining: dict[int, int] = dict()
        instruction_queue: dict[int, Instruction] = dict()
        pending_ancilla_qubits: set[int] = set()

        def append_instr(instr: Instruction):
            # print(instr)
            if debug_qubit in instr.qubits_list():
                print(f'\tScheduling instruction {instr}')
            compiled_schedule.append_instr(instr)
            instr_time_remaining[len(compiled_schedule.instructions)-1] = instr.duration
            for q in instr.qubits_list():
                if q in anc_qubits:
                    if q == debug_qubit:
                        print(f'\t\tRemoving {debug_qubit} from pending_ancilla_qubits')
                    pending_ancilla_qubits.discard(q)

        ########################################################################
        # Scheduling
        ########################################################################

        compiled_schedule = CompiledShuttlingSchedule(all_qubits)
        time = 0
        qubit_locations.update(
            {qubit: DeviceLocation(DeviceComponent.INTERACTION_ZONE, coords)
             for qubit, coords in static_data_coords.items()}
        )
        append_instr(
            Instantiate(
                qubits=anc_qubits,
                duration=self.hardware_params.init_duration,
                error=self.hardware_params.init_err,
                cell_coords=[init_ancilla_coords[anc] for anc in anc_qubits],
            )
        )
        qubit_locations.update(
            {qubit: DeviceLocation(DeviceComponent.INTERACTION_ZONE, coords)
             for qubit, coords in init_ancilla_coords.items()}
        )
        time += self.hardware_params.init_duration

        # Loop until all ancillas are done measuring their data qubits
        while remaining_ancilla:
            completed_instrs = []
            for instr_idx, time_remaining in instr_time_remaining.items():
                assert time_remaining >= 0
                if time_remaining == 0:
                    instr = compiled_schedule.instructions[instr_idx]
                    if debug_qubit in instr.qubits_list():
                        print('Completed instruction:', instr)
                    completed_instrs.append(instr_idx)
                    if isinstance(instr, Measure):
                        pending_ancilla_qubits -= set(instr.qubits_list())
                        for q in instr.qubits_list():
                            if q in remaining_ancilla:
                                remaining_ancilla.remove(q)
                                if q == debug_qubit:
                                    print(f'\tDone with qubit {q}')
                            qubit_locations.pop(q)
                    else:
                        if debug_qubit in instr.qubits_list() and debug_qubit in remaining_ancilla:
                            print(f'\tAdding {debug_qubit} to pending_ancilla_qubits')
                        pending_ancilla_qubits.update([q for q in instr.qubits_list() if q in remaining_ancilla])

                    # Instruction-specific state changes
                    if isinstance(instr, Shuttle):
                        qubit_locations[instr.qubit] = DeviceLocation(
                            component=DeviceComponent.SHUTTLE_CHANNEL,
                            coords=instr.end_coords
                        )
                        if instr.qubit == debug_qubit:
                            print(f'\tSHUTTLE: updated {instr.qubit} location to {qubit_locations[instr.qubit]}')
                    elif isinstance(instr, EmplaceDisplace):
                        assert qubit_locations[instr.qubit] and qubit_locations[instr.qubit].component == instr.start_component, (qubit_locations[instr.qubit], instr)
                        qubit_locations[instr.qubit] = DeviceLocation(
                            component=instr.end_component,
                            coords=instr.coords,
                        )
                        if instr.qubit == debug_qubit:
                            print(f'\tEMPLACE: updated {instr.qubit} location to {qubit_locations[instr.qubit]}')
                    elif isinstance(instr, Gate):
                        if instr.name == GateName.CX:
                            qa, qb = instr.qubits
                            q_anc = None
                            if qa in remaining_ancilla:
                                q_anc = qa
                            else:
                                assert qb in remaining_ancilla
                                q_anc = qb
                            ancilla_data_to_visit[q_anc].pop(0)
                        elif instr.name == GateName.CZ:
                            raise NotImplementedError
                        if debug_qubit in instr.qubits:
                            print(f'\tGATE: applied {instr.name} to {instr.qubits}')
            for instr_idx in completed_instrs:
                instr_time_remaining.pop(instr_idx)

            # Start new instructions. When qubits are freed up, they either just
            # finished a shuttling operation (so the ancilla is in the same cell
            # as the data it wants to talk to) or just finished a gate (so a
            # shuttle or another gate is needed).
            for qubit in [int(q) for q in np.random.permutation(list(pending_ancilla_qubits))]:
                qubit_loc = qubit_locations[qubit]
                qubit_coords = qubit_loc.coords
                if qubit == debug_qubit:
                    print(f'Starting new instruction for qubit {qubit} at location {qubit_coords}...')
                assert isinstance(qubit_coords[0], int) and isinstance(qubit_coords[1], int), (qubit, qubit_coords)
                qubit_coords = (int(qubit_coords[0]), int(qubit_coords[1]))
                if qubit in remaining_ancilla:
                    # Determine which data qubit to go to next
                    if ancilla_data_to_visit and ancilla_data_to_visit[qubit]:
                        target_data = ancilla_data_to_visit[qubit][0]
                        data_coords = static_data_coords[target_data]
                        if qubit == debug_qubit:
                            print(f'\tTargeting next data qubit {target_data} at location {data_coords}')
                        if data_coords == qubit_coords:
                            # Done shuttling, try to emplace into interaction
                            # zone and do the CX
                            if qubit_loc.component == DeviceComponent.SHUTTLE_CHANNEL:
                                # Need to emplace into interaction zone. First,
                                # have to determine whether the space is
                                # available

                                interaction_zone_count = 0
                                occupants = []
                                for q, loc in qubit_locations.items():
                                    if loc and loc.component == DeviceComponent.INTERACTION_ZONE and loc.coords == data_coords:
                                        interaction_zone_count += 1
                                        occupants.append(q)
                                if interaction_zone_count > 1:
                                    if qubit == debug_qubit:
                                        print(f'\t\tAttempted emplace, but interaction zone already occupied (by {occupants})! Waiting...')
                                    continue

                                if qubit == debug_qubit:
                                    print(f'\t\tEmplacing into interaction zone...', data_coords, qubit_coords)
                                append_instr(
                                    EmplaceDisplace(
                                        self.hardware_params.emplace_duration,
                                        self.hardware_params.emplace_err,
                                        qubit,
                                        data_coords,
                                        DeviceComponent.SHUTTLE_CHANNEL,
                                        DeviceComponent.INTERACTION_ZONE,
                                    )
                                )
                            elif qubit_loc.component == DeviceComponent.INTERACTION_ZONE:
                                # Can do the CX
                                if qubit == debug_qubit:
                                    print(f'\t\tApplying CX between qubits {[target_data, qubit] if qubit in code.Z_ancilla_indices else [qubit, target_data]}')
                                append_instr(
                                    Gate(
                                        self.hardware_params.cx_duration,
                                        self.hardware_params.cx_err,
                                        [target_data, qubit] if qubit in code.Z_ancilla_indices else [qubit, target_data],
                                        GateName.CX,
                                    )
                                )
                            else:
                                raise RuntimeError('Unexpected qubit position')
                        else:
                            # Continue shuttling

                            def attempt_schedule(shuttle_edge: tuple[tuple[int, int], tuple[int, int]]) -> bool:
                                changing_idx = 0 if shuttle_edge[0][0] != shuttle_edge[1][0] else 1
                                positive_direction = shuttle_edge[0][changing_idx] < shuttle_edge[1][changing_idx]
                                can_do_shuttle = not (shuttle_edge in qubit_locations.values() or (shuttle_edge[1], shuttle_edge[0]) in qubit_locations.values())
                                if use_highways:
                                    # channels alternate allowed directions
                                    can_do_shuttle = can_do_shuttle and ((data_coords[changing_idx] % 2 == 1) ^ positive_direction)
                                if can_do_shuttle:
                                    append_instr(
                                        Shuttle(
                                            self.hardware_params.shuttle_duration,
                                            self.hardware_params.shuttle_err,
                                            qubit=qubit,
                                            start_coords=(int(qubit_coords[0]), int(qubit_coords[1])),
                                            end_coords=shuttle_edge[1],
                                        )
                                    )
                                    if qubit == debug_qubit:
                                        print(f'\t\tSuccessfully scheduled shuttle of qubit {qubit} from {shuttle_edge[0]} to {shuttle_edge[1]}!')
                                    return True
                                if qubit == debug_qubit:
                                    print(f'\t\tAttempted shuttle of qubit {qubit} from {shuttle_edge[0]} to {shuttle_edge[1]}, but failed.')
                                return False

                            x_move_needed = data_coords[0] - qubit_coords[0]
                            y_move_needed = data_coords[1] - qubit_coords[1]

                            scheduled_shuttle = False
                            # First try to move horizontally
                            # Can shuttle = edge is free and direction aligns
                            # with highways
                            if x_move_needed != 0:
                                x_shuttle_edge = (qubit_coords, (int(qubit_coords[0] + np.sign(x_move_needed)), qubit_coords[1]))
                                scheduled_shuttle = attempt_schedule(x_shuttle_edge)
                            if not scheduled_shuttle and y_move_needed != 0:
                                y_shuttle_edge = (qubit_coords, (qubit_coords[0], int(qubit_coords[1] + np.sign(y_move_needed))))
                                scheduled_shuttle = attempt_schedule(y_shuttle_edge)
                            if not scheduled_shuttle:
                                # If still didn't schedule shuttle, try moving
                                # opposite direction in x just to do something.
                                shuttle_edge = (qubit_coords, (int(qubit_coords[0] - np.sign(x_move_needed)), qubit_coords[1]))
                                scheduled_shuttle = attempt_schedule(shuttle_edge)
                            if not scheduled_shuttle:
                                # Try last remaining shuttle option
                                shuttle_edge = (qubit_coords, (qubit_coords[0], int(qubit_coords[1] - np.sign(y_move_needed))))
                                scheduled_shuttle = attempt_schedule(shuttle_edge)
                            if not scheduled_shuttle:
                                # Finally, if no edges are available, we just
                                # wait. With highways, there will eventually be
                                # a valid path to take. Without highways, we
                                # might just idle forever.
                                print(f'Qubit {qubit} stuck, idling...')
                                # TODO: could do this a lot smarter if we plan
                                # out the shuttling paths ahead of time.
                                # Shouldn't move an ancilla qubit into the
                                # shuttling channels until we know it can do the
                                # whole path without getting stuck.
                    else:
                        if qubit_loc.component == DeviceComponent.INTERACTION_ZONE:
                            # Need to measure out the ancilla qubit. Default to the
                            # nearest readout port and see whether it's
                            # available.
                            if qubit == debug_qubit:
                                print(f'Ancilla qubit {qubit} done with all data qubits, moving to measurement...')
                            for q, loc in qubit_locations.items():
                                if loc and loc.component == DeviceComponent.READOUT and loc.coords == qubit_coords:
                                    # Wait until this one is gone
                                    if qubit == debug_qubit:
                                        print(f'Adjacent readout occupied, waiting...')
                                    continue

                            append_instr(
                                EmplaceDisplace(
                                    self.hardware_params.emplace_duration,
                                    self.hardware_params.emplace_err,
                                    qubit,
                                    qubit_coords,
                                    DeviceComponent.INTERACTION_ZONE,
                                    DeviceComponent.READOUT,
                                )
                            )
                        else:
                            assert qubit_loc.component == DeviceComponent.READOUT
                            if qubit == debug_qubit:
                                print(f'Measuring out ancilla qubit {qubit}')
                            append_instr(
                                Measure(
                                    self.hardware_params.measure_duration,
                                    self.hardware_params.measure_err,
                                    [qubit],
                                    [qubit_coords],
                                )
                            )
                else:
                    raise RuntimeError('Qubit in idling_qubits but not in remaining_ancilla')
            # If two ancillae compete for the same data qubit, we give priority
            # to the one with more remaining data. We break ties by prioritizing
            # the nearest ancilla to the data qubit.
            # NOTE: this might actually not matter, since we expect the
            # shuttling time per-edge to be longer than the CX time...
            # TODO: need specific instructions for shuttling into "interaction
            # zone" and "readout zone". Need to keep track of which ancillae are
            # actively on the shuttling tracks and which are off the tracks.

            # TODO: if cx_layers is empty, each ancilla can decide dynamically
            # which data qubit to visit next. Need to keep track of which data
            # qubits each ancilla qubit is heading towards - then we can make an
            # informed decision when we schedule a new ancilla to decide which
            # data it should go to.

            # Step forward in time
            time_incr = min(instr_time_remaining.values())
            instr_time_remaining = {instr: time - time_incr for instr,time in instr_time_remaining.items()}

        raise NotImplementedError

    def _map_init_ancilla_positions(
            self,
            data_positions: dict[int, tuple[int, int]],
            code: QECCode,
            interaction_graph: nx.Graph,
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