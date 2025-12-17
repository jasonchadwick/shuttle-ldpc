# The QEC mapping problem

We seek to implement a syndrome extraction round of an arbitrary QEC code on a unit-cell architecture, where shuttling tracks lie on a square grid with integer coordinates marking each intersection. Data qubits are assumed to have user-specified fixed locations, with at most one data qubit living at each intersection. Each intersection also contains an initialization/measurement port that can be used to initialize or measure one ancilla qubit at a time. When a qubit shuttles to an intersection, moving it off the shuttling track and into the readout zone or the interaction zone (where the data qubit lives), or moving between the two, incurs a time cost `emplace_duration`.

Assuming we define the initialization locations of all the ancilla qubits, mapping the syndrome extraction round becomes a multi-agent path planning (MAPP) problem. Each ancilla qubit has a list of data qubits (coordinates) it must visit, where each visit involves an emplace into the interaction zone, a two-qubit gate (duration `cx_duration`), and an emplace back to shuttling track. Crucially, only two qubits (the data qubit and one ancilla) can be in the interaction zone at any time. Similarly, only one qubit can be in a readout zone at any time. Qubits on the shuttling tracks cannot pass each other in opposite directions. We can assume that it is fine for two qubits to both momentarily occupy an intersection as long as they are not both idling (neglecting the minor shuttling overhead for one qubit to briefly move out of the way of the other).

For an ancilla qubit at an intersection, it has the following available actions:
1. `SHUTTLE` to an adjacent intersection with time cost `shuttle_duration`.
2. `EMPLACE` into either the local interaction zone or the readout zone, if there is space.
3. `IDLE` at current intersection for arbitrary integer time `t`.

An ancilla qubit in an interaction zone has the following available actions:
1. `CX` gate with the data qubit (this is required with each data qubit in the target list of the ancilla).
2. `EMPLACE` to either the intersection or readout zone, if there is space.
3. `IDLE` at current intersection for arbitrary integer time `t`.

An ancilla qubit in a readout zone has the following available actions:
1. `MEASURE`, which removes the qubit from the device after `measure_duration` time (freeing up the readout).
2. `EMPLACE` to either the intersection or interaction zone, if there is space.
3. `IDLE` at current intersection for arbitrary integer time `t`.

## Initial attempt at scheduling

I tried to write a simple greedy A* scheduling algorithm for this problem, but I got tripped up because there is a lot of flexibility in the system. It is simple to find a path for the ancilla qubit to get to a target data qubit, accounting for the existing paths that have already been scheduled, but if the interaction zone is occupied when it arrives then it needs to wait. But this does not seem ideal, because I could easily see this creating a traffic jam where the qubit currently in the interaction zone cannot go anywhere.