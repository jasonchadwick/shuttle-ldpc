import stim
from src.device import SpinBusDevice
from src.mapper import ObjectiveSchedule

def get_stim(
        schedule: ObjectiveSchedule,
        device: 
        basis: str = 'Z',
        xyz_decoding: bool = True,
    ):
    circ = stim.Circuit()

    data_indices = schedule.data_indices
    X_anc_indices = schedule.X_ancilla_indices
    Z_anc_indices = schedule.Z_ancilla_indices
    observable = schedule.logical_observable_qubits[basis]

    for q,coords in schedule.data_coords.items():
        circ.append('QUBIT_COORDS', q, coords)
    
    circ.append('R' if basis == 'Z' else 'RX', data_indices, ())
    circ.append()