from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np
import math
import stim
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from extern.spin_qec.ldpc_code.QECCode import QECCode

@dataclass
class HardwareParams:
    T2: float # in seconds
    cx_err: float
    cx_duration: int # in ns
    h_err: float
    h_duration: int # in ns
    shuttle_err: float # per unit cell shuttle
    shuttle_duration: int # per unit cell shuttle, in ns
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
        init_err=p,
        init_duration=500,
        measure_err=p,
        measure_duration=500,
    )

default_params = hardware_params(1e-3)

@dataclass
class Instruction:
    qubits: list[int]
    duration: int
    error: float

class GateName(Enum):
    CX = 0
    CZ = 1
    H = 2

@dataclass
class Gate(Instruction):
    name: GateName

@dataclass
class Instantiate(Instruction):
    cell_coords: list[tuple[int, int]]

@dataclass
class Measure(Instruction):
    cell_coords: list[tuple[int, int]]

@dataclass
class Shuttle(Instruction):
    start_coords: tuple[int, int]
    end_coords: tuple[int, int]

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
        for q in instr.qubits:
            self.instructions_by_qubit[q].append(len(self.instructions))
        self.instructions.append(instr)

    def get_frames(self, frames_per_segment: int = 10) -> list[Frame]:
        """Convert per-qubit schedules to per-frame view, where each qubit"""
        raise NotImplementedError()

    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the compiled shuttling schedule back into a Stim circuit."""
        raise NotImplementedError()

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

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

    def compile_SE_schedule(
            self,
            code: QECCode,
            static_data_positions: dict[int, tuple[int, int]],
            cx_layers: list[list[tuple[int, int]]],
            use_highways: bool,
            refocus_shuttle_noise: bool,
        ) -> tuple[dict[int, tuple[int, int]], CompiledShuttlingSchedule]:
        """Compile a shuttling schedule for a syndrome extraction round. Assumes
        data qubits are already initialized and fixed in place within each unit
        cell.
        """
        ########################################################################
        # Checking inputs and setting up variables
        ########################################################################
        if len(set(static_data_positions.values())) < len(static_data_positions):
            raise ValueError("Static data qubit positions must be unique.")

        all_qubits = set(code.data_indices + code.X_ancilla_indices + code.Z_ancilla_indices)
        all_qubits = list(sorted(all_qubits))
        anc_qubits = list(sorted(code.X_ancilla_indices + code.Z_ancilla_indices))

        interaction_graph = nx.Graph(all_qubits)
        for check_idx, check_qs in enumerate(code.X_checks):
            for data_q in check_qs:
                interaction_graph.add_edge(code.X_ancilla_indices[check_idx], data_q)
        for check_idx, check_qs in enumerate(code.Z_checks):
            for data_q in check_qs:
                interaction_graph.add_edge(code.Z_ancilla_indices[check_idx], data_q)

        if len(static_data_positions) > 0 and set(static_data_positions) != set(code.data_indices):
            raise ValueError('Must supply all or no data indices')
        if len(static_data_positions) == 0:
            raise NotImplementedError('Auto computing data positions not supported yet')
        
        ancilla_data_to_visit = dict()
        if cx_layers:
            for layer in cx_layers:
                for qa,qb in layer:
                    if qa in anc_qubits:
                        ancilla_data_to_visit.setdefault(qa, []).append(qb)
                    elif qb in anc_qubits:
                        ancilla_data_to_visit.setdefault(qb, []).append(qa)
                    else:
                        raise ValueError
        ########################################################################
        # Scheduling
        ########################################################################
        init_ancilla_positions = self._map_init_ancilla_positions(static_data_positions, code, interaction_graph)
        assert set(init_ancilla_positions.keys()) == set(anc_qubits)
        current_positions: dict[int, tuple[int, int] | None] = {q:None for q in all_qubits}
        current_positions.update(static_data_positions)

        compiled_schedule = CompiledShuttlingSchedule(all_qubits)
        time = 0
        compiled_schedule.append_instr(
            Instantiate(
                qubits=anc_qubits,
                duration=self.hardware_params.init_duration,
                error=self.hardware_params.init_err,
                cell_coords=[init_ancilla_positions[anc] for anc in anc_qubits],
            )
        )
        current_positions.update(init_ancilla_positions)
        time += self.hardware_params.init_duration

        # Loop until all ancillas are done measuring their data qubits
        remaining_ancilla = set(anc_qubits)
        instr_time_remaining: dict[int, int] = dict()

        # Each unit cell can hold one ancilla, one readout, and one
        # measuring/initializing qubit at a time.
        static_ancilla_coords: dict[int, tuple[int, int]] = init_ancilla_positions.copy()
        shuttling_ancilla_coords: dict[int, tuple[float, float]] = dict()
        occupied_edges: dict[int, tuple[tuple[float, float], tuple[float, float]]] = dict()
        active_readout_ports: set[tuple[int, int]] = set()
        
        idling_qubits = set(init_ancilla_positions.keys())
        while remaining_ancilla:
            completed_instrs = []
            for instr_idx, time_remaining in instr_time_remaining.items():
                assert time_remaining >= 0
                if time_remaining == 0:
                    instr = compiled_schedule.instructions[instr_idx]
                    completed_instrs.append(instr_idx)
                    if isinstance(instr, Measure):
                        idling_qubits -= set(instr.qubits)
                    else:
                        idling_qubits.update(instr.qubits)
                    if isinstance(instr, Shuttle):
                        assert len(instr.qubits) == 1
                        qubit = instr.qubits[0]
                        shuttling_ancilla_coords[qubit] = instr.end_coords
                        occupied_edges.pop(qubit)
            for instr_idx in completed_instrs:
                instr_time_remaining.pop(instr_idx)

            # Start new instructions. When qubits are freed up, they either just
            # finished a shuttling operation (so the ancilla is in the same cell
            # as the data it wants to talk to) or just finished a gate (so a
            # shuttle or another gate is needed).
            # TODO
            for qubit in idling_qubits:
                if qubit in remaining_ancilla:
                    # Determine which data qubit to go to next
                    if ancilla_data_to_visit and ancilla_data_to_visit[qubit]:
                        target_data = ancilla_data_to_visit[qubit][0]
                        data_coords = static_data_positions[target_data]
                        anc_coords = shuttling_ancilla_coords[qubit]
                        if data_coords == anc_coords:
                            # Done shuttling, do the CX
                            # TODO: check static_ancilla_coords first to make
                            # sure another ancilla isn't in the way
                            # TODO: update static_ancilla_coords and
                            # shuttling_ancilla_coords
                        else:
                            # Continue shuttling
                            # TODO: decide on route (looking at shuttling_ancilla_coords)
                            # TODO: append instruction
                            # TODO: update occupied_edges
                else:
                    pass

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
        ) -> dict[int, tuple[int, int]]:
        """Given static positions of data qubits, determine where to initially
        place ancilla qubits at the start of a syndrome extraction round.
        """
        raise NotImplementedError