
import numpy as np
import copy
from math import ceil, floor
from numpy.typing import NDArray
import stim
from src.QECCode import QECCode
import src.device as device

class Qubit():
    """A single physical qubit on a device.
    """
    def __init__(self, idx: int, coords: tuple[int, int]) -> None:
        """Initialize.
        
        Args:
            idx: Index of the qubit.
            coords: Coordinates of the qubit on the device.
        """
        self.idx: int = idx
        self.coords: tuple[int, int] = coords

    def __repr__(self) -> str:
        return f'{self.idx}, Coords: {self.coords}'

class DataQubit(Qubit):
    """Data qubit used to store logical information.
    """
    pass

class MeasureQubit(Qubit):
    """Ancilla qubit used to perform stabilizer measurements.
    """
    def __init__(self, idx: int, coords: tuple[int, int], data_qubits: list[DataQubit | None], basis: str) -> None:
        """Initialize.
        
        Args:
            idx: Index of the qubit.
            coords: Coordinates of the qubit on the device.
            data_qubits: List of data qubits that this qubit measures.
        """
        super().__init__(idx, coords)
        self.data_qubits = data_qubits
        self.basis = basis

    def __repr__(self):
        return f'{self.idx}, Coords: {self.coords}, Basis: {self.basis}, Data Qubits: {self.data_qubits}'

class RotatedSurfaceCode(QECCode):
    def __init__(
            self,
            d: int | None = None,
            dz: int | None = None,
            dx: int | None = None,
            id_offset: int = 0,
        ):
        if not d and (not dz or not dx):
            raise ValueError("Either d or both dx and dz must be provided.")
        if d:
            self.dz = self.dx = d
        else:
            assert dz and dx
            self.dz = dz
            self.dx = dx
        self.num_data: int = self.dz * self.dx

        self.device: list[list[Qubit | None]] = [
            [None for _ in range(2*self.dx+1)] for _ in range(2*self.dz+1)]
        
        assert len(self.device) == 2*self.dz+1
        assert len(self.device[0]) == 2*self.dx+1

        self.data: list[DataQubit] = self._place_data(id_offset)
        self._place_ancilla(id_offset)

        self.ancilla = self.x_ancilla + self.z_ancilla
        self.all_qubits: list[DataQubit | MeasureQubit] = sorted(self.ancilla + self.data, key=lambda q: q.idx)

        self.data_indices = [q.idx for q in self.data]
        self.X_ancilla_indices = [q.idx for q in self.x_ancilla]
        self.Z_ancilla_indices = [q.idx for q in self.z_ancilla]
        self.X_checks = [[d.idx for d in ancilla.data_qubits if d] for ancilla in self.x_ancilla]
        self.Z_checks = [[d.idx for d in ancilla.data_qubits if d] for ancilla in self.z_ancilla]

        self.check_cx_layers = []
        for i in range(4):
            self.check_cx_layers.append([])
            for measure in self.x_ancilla:
                dqi = measure.data_qubits[i]
                if dqi != None:
                    self.check_cx_layers[-1].append((measure.idx, dqi.idx))
            for measure in self.z_ancilla:
                dqi = measure.data_qubits[i]
                if dqi != None:
                    self.check_cx_layers[-1].append((dqi.idx, measure.idx))

        self.qubit_coords = [(q.coords[0], q.coords[1]) for q in self.all_qubits]
        self.ancilla_reference_positions = [self.qubit_coords[a] for a in [q.idx for q in self.ancilla]]

        self.hz = np.array([[1 if q_idx in checks else 0 for q_idx in self.data_indices] for checks in self.Z_checks])
        self.hx = np.array([[1 if q_idx in checks else 0 for q_idx in self.data_indices] for checks in self.X_checks])
        # lx, lz = self.compute_logical_operators()
        # self.lz = lz.T
        # self.lx = lx.T

    def _place_data(
            self,
            id_offset: int = 0,
        ) -> list[DataQubit]:
        data: list[DataQubit] = [
            DataQubit(id_offset+(self.dx*row + col), (2*row+1, 2*col+1)) 
            for col in range(self.dx) for row in range(self.dz)]
        data = list(sorted(data, key=lambda q: q.idx))
        
        for data_qubit in data:
            self.device[data_qubit.coords[0]][data_qubit.coords[1]] = data_qubit
        return data

    def _get_neighboring_data_qubits(self, coords: tuple[int, int], basis: str) -> list[DataQubit | None]:
        if basis == 'Z':
            offsets = np.array([[1, 1], [+1, -1], [-1, +1], [-1, -1]])
        else:
            offsets = np.array([[1, 1], [-1, +1], [+1, -1], [-1, -1]])
        qubits = []
        for offset in offsets:
            new_coords = (coords[0] + offset[0], coords[1] + offset[1])
            if (new_coords[0] > 0 and new_coords[0] < len(self.device)
                and new_coords[1] > 0 and new_coords[1] < len(self.device[0])):
                q = self.device[new_coords[0]][new_coords[1]]
                if isinstance(q, DataQubit):
                    qubits.append(q)
                else:
                    qubits.append(None)
            else:
                qubits.append(None)
            
        return qubits

    def _place_ancilla(self, id_offset: int = 0) -> None:
        # number of qubits already placed (= index of next qubit)
        q_count = len(self.data) + id_offset

        self.x_ancilla: list[MeasureQubit] = []
        self.z_ancilla: list[MeasureQubit] = []
        for row in range(self.dz+1):
            for col in range(self.dx+1):
                if (row + col) % 2 == 0 and col != 0 and col != self.dx: # Z basis
                    coords = (2*row, 2*col)
                    data_qubits = self._get_neighboring_data_qubits(coords, 'Z')
                    if all(q is None for q in data_qubits):
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, 'Z')
                    self.device[coords[0]][coords[1]] = measure_q
                    self.z_ancilla.append(measure_q)
                    q_count += 1
                elif (row + col) % 2 == 1 and row != 0 and row != self.dz: # Z basis
                    coords = (2*row, 2*col)
                    data_qubits = self._get_neighboring_data_qubits(coords, 'X')
                    if all(q is None for q in data_qubits):
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, 'X')
                    self.device[coords[0]][coords[1]] = measure_q
                    self.x_ancilla.append(measure_q)
                    q_count += 1

    def compute_logical_operators(self) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        logical_X = np.zeros((1, self.num_data), dtype=np.int_)
        logical_Z = np.zeros((1, self.num_data), dtype=np.int_)
        for q in self.data:
            if q.coords[0] == min(q1.coords[0] for q1 in self.data):
                logical_X[0, q.idx] = 1
            if q.coords[1] ==  min(q1.coords[1] for q1 in self.data):
                logical_Z[0, q.idx] = 1
        return logical_X, logical_Z

    def compute_code_parameters(self):
        return self.dz*self.dx, 1, min(self.dz, self.dx)
    
    def draw(self):
        """Draw the surface code.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(self.dx+1, self.dz+1))
        ax.set_xlim(-1, 2*self.dx+1)
        ax.set_ylim(-1, 2*self.dz+1)
        ax.set_aspect('equal')

        for q in self.all_qubits:
            # draw rectangle for stabilizer, labeled with ancilla qubit index.
            # Color is red for X basis, blue for Z basis.
            # Draw white circle for data qubit, labeled with data qubit index.
            # All rects should be behind the circles.
            if isinstance(q, MeasureQubit):
                color = 'red' if q.basis == 'X' else 'blue'
                rect = patches.Rectangle((q.coords[1]-0.7, q.coords[0]-0.7), 1.4, 1.4, linewidth=1, edgecolor=color, facecolor='none', zorder=1)
                ax.add_patch(rect)
                ax.text(q.coords[1], q.coords[0], str(q.idx), fontsize=8, ha='center', va='center', color=color, zorder=2)
            else:
                circle = patches.Circle((q.coords[1], q.coords[0]), radius=0.5, linewidth=1, edgecolor='black', facecolor='white', zorder=2)
                ax.add_patch(circle)
                ax.text(q.coords[1], q.coords[0], str(q.idx), fontsize=8, ha='center', va='center', zorder=3)

        plt.show()

    def get_stim(self, basis: str, rounds: int, p_g: float, p_m: float, idle_per_round: float = 0.0, midpoint_data_err: float = 0.0) -> stim.Circuit:
        circ = stim.Circuit()
        
        for q in self.all_qubits:
            circ.append('QUBIT_COORDS', [q.idx], tuple(q.coords))
        
        meas_rec = {}
        circ.append('R' if basis == 'Z' else 'RX', self.data_indices, [])

        idle_per_CX_layer = idle_per_round / 4

        for i in range(rounds):
            # if i == rounds//2 and midpoint_data_err:
            circ.append('DEPOLARIZE1', self.data_indices, midpoint_data_err)

            circ.append('RX', self.X_ancilla_indices, [])
            circ.append('R', self.Z_ancilla_indices, [])

            for layer in self.check_cx_layers:
                if idle_per_CX_layer:
                    circ.append('DEPOLARIZE1', self.data_indices + self.X_ancilla_indices + self.Z_ancilla_indices, idle_per_CX_layer)
                circ.append('CX', [q for pair in layer for q in pair], [])
                circ.append('DEPOLARIZE2', [q for pair in layer for q in pair], p_g)
                circ.append('TICK')

            circ.append('MX', self.X_ancilla_indices, p_m)
            for q in self.X_ancilla_indices:
                for qb in meas_rec.keys():
                    meas_rec[qb] = [m-1 for m in meas_rec[qb]]
                meas_rec.setdefault(q, []).append(-1)
            
            circ.append('M', self.Z_ancilla_indices, p_m)
            for q in self.Z_ancilla_indices:
                for qb in meas_rec.keys():
                    meas_rec[qb] = [m-1 for m in meas_rec[qb]]
                meas_rec.setdefault(q, []).append(-1)

            for stab_idx,stab in enumerate(self.X_checks if basis == 'X' else self.Z_checks):
                ancilla_idx = (self.X_ancilla_indices[stab_idx] if basis == 'X' else self.Z_ancilla_indices[stab_idx])
                if i == 0:
                    circ.append('DETECTOR', stim.target_rec(meas_rec[ancilla_idx][-1]), (ancilla_idx, i))
                else:
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[ancilla_idx][-1]), stim.target_rec(meas_rec[ancilla_idx][-2])], (ancilla_idx, i))
            if i > 0:
                for stab_idx,stab in enumerate(self.Z_checks if basis == 'X' else self.X_checks):
                    ancilla_idx = (self.Z_ancilla_indices[stab_idx] if basis == 'X' else self.X_ancilla_indices[stab_idx])
                    circ.append('DETECTOR', [stim.target_rec(meas_rec[ancilla_idx][-1]), stim.target_rec(meas_rec[ancilla_idx][-2])], (ancilla_idx, i))

        circ.append('MX' if basis == 'X' else 'M', self.data_indices, p_m)
        for q in self.data_indices:
            for qb in meas_rec.keys():
                meas_rec[qb] = [m-1 for m in meas_rec[qb]]
            meas_rec.setdefault(q, []).append(-1)

        for stab_idx,stab in enumerate(self.X_checks if basis == 'X' else self.Z_checks):
            ancilla_idx = (self.X_ancilla_indices[stab_idx] if basis == 'X' else self.Z_ancilla_indices[stab_idx])
            circ.append('DETECTOR', [stim.target_rec(meas_rec[q][-1]) for q in [ancilla_idx] + stab], (ancilla_idx, i))
        
        X_obs, Z_obs = self.compute_logical_operators()
        X_obs = X_obs.T
        Z_obs = Z_obs.T
        for i,obs in enumerate(X_obs if basis == 'X' else Z_obs):
            circ.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_rec[q][-1]) for q in self.data_indices if obs[q]], i)

        return circ
    
    def get_schedule(
            self,
            hwp: device.HardwareParams,
            basis: str,
            rounds: int = 1,
            refocus_shuttling_noise: bool = False,
        ) -> device.CompiledShuttlingSchedule:
        assert basis in ['X', 'Z']

        schedule = device.CompiledShuttlingSchedule([q.idx for q in self.all_qubits], self.data_indices)

        # Initialize
        data_init_coords = {q:((x+1)//2, (y+1)//2) for q,(x,y) in [(q,self.qubit_coords[q]) for q in self.data_indices]}
        anc_init_coords = {q:((x+2)//2, (y+2)//2) for q,(x,y) in [(q,self.qubit_coords[q]) for q in self.X_ancilla_indices + self.Z_ancilla_indices]}
        schedule.append_instr(
            device.Instantiate(
                hwp.init_duration,
                schedule.data_qubits,
                [data_init_coords[q] for q in schedule.data_qubits],
            ),
            0,
        )
        if basis == 'X':
            schedule.append_instr(
                device.Gate(
                    hwp.h_duration,
                    schedule.data_qubits,
                    device.GateName.H,
                ), schedule.total_duration()
            )
        t = schedule.total_duration()
        for data in schedule.data_qubits:
            schedule.append_instr(
                device.EmplaceDisplace(
                    hwp.emplace_duration,
                    data,
                    data_init_coords[data],
                    device.DeviceComponent.READOUT,
                    device.DeviceComponent.INTERACTION_ZONE,
                ),
                t,
            )

        def do_check(
                anc_idx: int,
                data: DataQubit,
                cur_anc_coords: tuple[int, int],
                t_start: int,
                start_anc_component: device.DeviceComponent = device.DeviceComponent.SHUTTLE_INTERSECTION,
                end_anc_component: device.DeviceComponent = device.DeviceComponent.SHUTTLE_INTERSECTION,
                first_check: bool = False,
                last_check: bool = False,
            ):
            data_idx = data.idx
            data_coords = data_init_coords[data_idx]
            assert cur_anc_coords == data_coords
            schedule.append_instr(device.EmplaceDisplace(
                hwp.emplace_duration,
                anc_idx,
                data_coords,
                start_anc_component,
                device.DeviceComponent.INTERACTION_ZONE,
            ), t_start)
            t_cx = t_start + hwp.emplace_duration
            if refocus_shuttling_noise and not first_check:
                if anc_idx in self.Z_ancilla_indices:
                    schedule.append_instr(device.Gate(
                        hwp.h_duration,
                        [anc_idx],
                        device.GateName.H,
                    ), t_cx)
                t_cx += hwp.h_duration
            schedule.append_instr(device.Gate(
                hwp.cx_duration,
                [anc_idx, data_idx] if anc_idx in self.X_ancilla_indices else [data_idx, anc_idx],
                device.GateName.CX,
            ), t_cx)
            t_leave = t_cx + hwp.cx_duration
            if refocus_shuttling_noise and not last_check:
                if anc_idx in self.Z_ancilla_indices:
                    schedule.append_instr(device.Gate(
                        hwp.h_duration,
                        [anc_idx],
                        device.GateName.H,
                    ), t_leave)
                t_leave += hwp.h_duration
            schedule.append_instr(device.EmplaceDisplace(
                hwp.emplace_duration,
                anc_idx,
                data_coords,
                device.DeviceComponent.INTERACTION_ZONE,
                end_anc_component,
            ), t_leave)
            t_end = t_leave + hwp.emplace_duration
            return t_end

        anc_indices = self.X_ancilla_indices + self.Z_ancilla_indices
        for round_idx in range(rounds):
            schedule.append_instr(
                device.Instantiate(
                    hwp.init_duration,
                    anc_indices,
                    [anc_init_coords[anc] for anc in anc_indices],
                ),
                schedule.total_duration()
            )
            schedule.append_instr(device.Gate(
                hwp.h_duration,
                self.X_ancilla_indices,
                device.GateName.H, 
            ), schedule.total_duration())

            # ti: time we start check i
            t1 = schedule.total_duration()
            if round_idx == 0:
                assert t1 == 2*hwp.init_duration + hwp.emplace_duration + hwp.h_duration + (0 if basis == 'Z' else hwp.h_duration), (t1, 2*hwp.init_duration + hwp.emplace_duration + hwp.h_duration)
            t2 = t1 + 2*hwp.emplace_duration + hwp.cx_duration + hwp.shuttle_duration
            t3 = t2 + 2*hwp.emplace_duration + hwp.cx_duration + 2*hwp.shuttle_duration
            t4 = t3 + 2*hwp.emplace_duration + hwp.cx_duration + hwp.shuttle_duration
            t_end = t4 + 2*hwp.emplace_duration + hwp.cx_duration
            if refocus_shuttling_noise:
                t2 += hwp.h_duration
                t3 += 3*hwp.h_duration
                t4 += 5*hwp.h_duration
                t_end += 6*hwp.h_duration

            # Syndrome measurement
            final_anc_coords: dict[int, tuple[int, int]] = dict()
            for ai,anc in enumerate(anc_indices):
                cur_anc_coords = anc_init_coords[anc]
                if anc in self.X_ancilla_indices:
                    checked_data = self.x_ancilla[ai].data_qubits
                else:
                    checked_data = self.z_ancilla[ai - len(self.X_ancilla_indices)].data_qubits

                # First check
                if checked_data[0]:
                    t = do_check(
                        anc,
                        checked_data[0],
                        cur_anc_coords,
                        t1,
                        start_anc_component=device.DeviceComponent.READOUT,
                        first_check=True,
                    )
                    assert t == t2 - hwp.shuttle_duration, (t, t2)
                else:
                    schedule.append_instr(device.EmplaceDisplace(
                        hwp.emplace_duration,
                        anc,
                        cur_anc_coords,
                        device.DeviceComponent.READOUT,
                        device.DeviceComponent.SHUTTLE_INTERSECTION,
                    ), t1)
                
                # Shuttle
                if anc in self.X_ancilla_indices:
                    dx,dy = -1,0
                else:
                    dx,dy = 0,-1
                schedule.append_instr(device.Shuttle(
                    hwp.shuttle_duration,
                    anc,
                    cur_anc_coords,
                    (cur_anc_coords[0]+dx, cur_anc_coords[1]+dy),
                ), t2 - hwp.shuttle_duration)
                cur_anc_coords = (cur_anc_coords[0]+dx, cur_anc_coords[1]+dy)

                # Second check
                if checked_data[1]:
                    t = do_check(
                        anc,
                        checked_data[1],
                        cur_anc_coords,
                        t2,
                        first_check=not checked_data[0],
                        last_check=(not checked_data[2] and not checked_data[3]),
                    )
                
                # Shuttle
                if anc in self.X_ancilla_indices:
                    shuttles = [(0,-1), (1,0)]
                else:
                    shuttles = [(-1,0), (0,1)]
                for i,(dx,dy) in enumerate(shuttles):
                    schedule.append_instr(device.Shuttle(
                        hwp.shuttle_duration,
                        anc,
                        cur_anc_coords,
                        (cur_anc_coords[0]+dx, cur_anc_coords[1]+dy),
                    ), t3 - (len(shuttles)-i)*hwp.shuttle_duration)
                    cur_anc_coords = (cur_anc_coords[0]+dx, cur_anc_coords[1]+dy)

                # Third check
                if checked_data[2]:
                    t = do_check(
                        anc,
                        checked_data[2],
                        cur_anc_coords,
                        t3,
                        first_check=(not checked_data[0] and not checked_data[1]),
                        last_check=(not checked_data[3]),
                    )
                
                # Shuttle
                if anc in self.X_ancilla_indices:
                    dx,dy = -1,0
                else:
                    dx,dy = 0,-1
                schedule.append_instr(device.Shuttle(
                    hwp.shuttle_duration,
                    anc,
                    cur_anc_coords,
                    (cur_anc_coords[0]+dx, cur_anc_coords[1]+dy),
                ), t4 - hwp.shuttle_duration)
                cur_anc_coords = (cur_anc_coords[0]+dx, cur_anc_coords[1]+dy)

                # Fourth check
                if checked_data[3]:
                    t = do_check(
                        anc,
                        checked_data[3],
                        cur_anc_coords,
                        t4,
                        end_anc_component=device.DeviceComponent.READOUT,
                        last_check=True,
                    )
                    assert t == t_end
                final_anc_coords[anc] = cur_anc_coords
            schedule.append_instr(device.Gate(
                hwp.h_duration,
                self.X_ancilla_indices,
                device.GateName.H, 
            ), t_end)
            schedule.append_instr(device.Measure(
                hwp.measure_duration,
                anc_indices,
                [final_anc_coords[anc] for anc in anc_indices],
            ), t_end + hwp.h_duration)

        # Measure data
        t = schedule.total_duration()
        for data in schedule.data_qubits:
            schedule.append_instr(
                device.EmplaceDisplace(
                    hwp.emplace_duration,
                    data,
                    data_init_coords[data],
                    device.DeviceComponent.INTERACTION_ZONE,
                    device.DeviceComponent.READOUT,
                ),
                t
            )
        if basis == 'X':
            schedule.append_instr(
                device.Gate(
                    hwp.h_duration,
                    schedule.data_qubits,
                    device.GateName.H,
                ), schedule.total_duration()
            )
        schedule.append_instr(
            device.Measure(
                hwp.init_duration,
                schedule.data_qubits,
                [data_init_coords[q] for q in schedule.data_qubits],
            ),
            schedule.total_duration()
        )
        
        return schedule