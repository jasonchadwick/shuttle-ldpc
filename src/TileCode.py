import numpy as np
import matplotlib.pyplot as plt
from src.QECCode import QECCode

class TileCode(QECCode):
    # Construction of arXiv:2504.09171.
    def __init__(
            self,
            X_pattern: list[tuple[tuple[int, int], tuple[int, int]]],
            Z_pattern: list[tuple[tuple[int, int], tuple[int, int]]],
            rotated: bool,
            size: int,
        ):
        self._X_pattern = X_pattern
        self._Z_pattern = Z_pattern
        self.B = max(c for edge in X_pattern + Z_pattern for node in edge for c in node)

        # Condition (T1): no support on top or right edges of BxB box
        for edge in X_pattern + Z_pattern:
            ((x1,y1),(x2,y2)) = edge
            assert not (y1 == y2 == self.B or x1 == x2 == self.B)
        
        # Condition (T2): 
        for edge in X_pattern + Z_pattern:
            ((x1,y1),(x2,y2)) = edge
            # TODO

        self._ancilla_coords = dict()

        self._bulk_nodes = set()
        self._Z_boundary_nodes = set()
        self._X_boundary_nodes = set()

        if rotated:
            raise NotImplementedError
        else:
            self._bulk_nodes = set((x,y) for x in range(size) for y in range(size))
            for (offset_x, offset_y, basis) in [(0,self.B-1,'X'), (self.B-1,0,'Z'), (0,-self.B+1,'X'), (-self.B+1,0,'Z')]:
                offset_nodes = set((x+offset_x, y+offset_y) for (x,y) in self._bulk_nodes)
                new_nodes = offset_nodes.difference(self._bulk_nodes)
                if basis == 'X':
                    self._X_boundary_nodes |= new_nodes
                elif basis == 'Z':
                    self._Z_boundary_nodes |= new_nodes
            assert set(self._bulk_nodes).isdisjoint(self._X_boundary_nodes)
            assert set(self._bulk_nodes).isdisjoint(self._Z_boundary_nodes)
            self._bulk_nodes = list(sorted(self._bulk_nodes))
            self._X_boundary_nodes = list(sorted(self._X_boundary_nodes))
            self._Z_boundary_nodes = list(sorted(self._Z_boundary_nodes))

        def sort_edge(edge):
            return tuple(sorted(edge))

        X_check_edges = []
        Z_check_edges = []
        self._qubit_edges = set()
        for x in range(size + self.B-1):
            for y in range(size + self.B-1):
                self._qubit_edges.add(((x,y), (x,y+1)))
                self._qubit_edges.add(((x,y), (x+1,y)))
        for x,y in self._bulk_nodes:
            check = []
            for edge_offset in X_pattern:
                ((dx1,dy1), (dx2,dy2)) = edge_offset
                true_edge = sort_edge(((x+dx1, y+dy1), (x+dx2, y+dy2)))
                check.append(true_edge)
            X_check_edges.append(check)

            check = []
            for edge_offset in Z_pattern:
                ((dx1,dy1), (dx2,dy2)) = edge_offset
                true_edge = sort_edge(((x+dx1, y+dy1), (x+dx2, y+dy2)))
                check.append(true_edge)
            Z_check_edges.append(check)

        for x,y in self._X_boundary_nodes:
            check = []
            for edge_offset in X_pattern:
                ((dx1,dy1), (dx2,dy2)) = edge_offset
                true_edge = sort_edge(((x+dx1, y+dy1), (x+dx2, y+dy2)))
                if true_edge in self._qubit_edges:
                    check.append(true_edge)
            X_check_edges.append(check)
        for x,y in self._Z_boundary_nodes:
            check = []
            for edge_offset in Z_pattern:
                ((dx1,dy1), (dx2,dy2)) = edge_offset
                true_edge = sort_edge(((x+dx1, y+dy1), (x+dx2, y+dy2)))
                if true_edge in self._qubit_edges:
                    check.append(true_edge)
            Z_check_edges.append(check)
        
        self._edge_coords = dict()
        for edge in self._qubit_edges:
            ((x1,y1),(x2,y2)) = edge
            x_mid, y_mid = (x1+x2)/2, (y1+y2)/2
            x = int(np.floor(x_mid + y_mid))
            y = int(np.floor(x_mid - y_mid))
            self._edge_coords[edge] = (x,y)
        self.num_data = len(self._edge_coords)
        self.data_indices = list(range(self.num_data))
        self.qubit_coords = list(sorted(self._edge_coords.values()))
        edge_to_qubit_index = {edge: self.qubit_coords.index(coords) for edge,coords in self._edge_coords.items()}

        self.X_checks = [[edge_to_qubit_index[edge] for edge in check] for check in X_check_edges]
        self.Z_checks = [[edge_to_qubit_index[edge] for edge in check] for check in Z_check_edges]

        self.X_ancilla_indices = list(range(self.num_data, self.num_data + len(self.X_checks)))
        self.Z_ancilla_indices = list(range(self.num_data + len(self.X_checks), self.num_data + len(self.Z_checks) + len(self.Z_checks)))

        self.Lx, self.Lz, self.k = self.compute_logicals(self.get_Hx(), self.get_Hz())

        # Rotate and compress into a rectangle
        coords_transformed = []
        for coord in self.qubit_coords:
            x,y = coord
            u = x-y
            v = (x+y+1)//2
            coords_transformed.append((u,v))
        self.qubit_coords = coords_transformed

        # Adjust to be all positive and starting at 0
        min_x = min(x for x,_ in self.qubit_coords)
        min_y = min(y for _,y in self.qubit_coords)
        self.qubit_coords = [(x-min_x,y-min_y) for (x,y) in self.qubit_coords]
        self.ancilla_reference_positions = dict()
        for i, X_node in enumerate(self._bulk_nodes + self._X_boundary_nodes):
            assert len(X_check_edges[i]) == len(self.X_checks[i])
            x0,y0 = X_node
            x = x0+y0
            y = x0-y0
            u = x-y
            v = (x+y+1)//2
            self.ancilla_reference_positions[self.X_ancilla_indices[i]] = (u+min_x,v+min_y)
        for i, Z_node in enumerate(self._bulk_nodes + self._Z_boundary_nodes):
            assert len(Z_check_edges[i]) == len(self.Z_checks[i])
            x0,y0 = Z_node
            x = x0+y0
            y = x0-y0
            u = x-y
            v = (x+y+1)//2
            self.ancilla_reference_positions[self.Z_ancilla_indices[i]] = (u+min_x,v+min_y)

    def compute_code_parameters(self):
        return (self.num_data, self.k, -1)

    def compute_logical_operators(self):
        return self.Lx, self.Lz
    
    def plot_construction(self, X_pattern_origin = (4,0), Z_pattern_origin = (0,4)):
        fig,ax = plt.subplots()
        for edge in self._qubit_edges:
            ((x1,y1),(x2,y2)) = edge
            plt.plot([x1,x2], [y1,y2], '-', color='gray', linewidth=1)
        Xx,Xy = X_pattern_origin
        for edge in self._X_pattern:
            ((x1,y1),(x2,y2)) = edge
            plt.plot([Xx+x1,Xx+x2], [Xy+y1,Xy+y2], '-', color='r', linewidth=2)
        plt.plot(Xx, Xy, 's', color='r', markersize=10)
        ax.add_patch(plt.Rectangle(X_pattern_origin,self.B,self.B,color='r',alpha=0.2))
        Zx,Zy = Z_pattern_origin
        for edge in self._Z_pattern:
            ((x1,y1),(x2,y2)) = edge
            plt.plot([Zx+x1,Zx+x2], [Zy+y1,Zy+y2], '-', color='b', linewidth=2)
        plt.plot(Zx, Zy, 'o', color='b', markersize=10)
        ax.add_patch(plt.Rectangle(Z_pattern_origin,self.B,self.B,color='b',alpha=0.2))
        for node in self._bulk_nodes:
            plt.plot(node[0], node[1], 'o', color='k')
        for node in self._X_boundary_nodes:
            plt.plot(node[0], node[1], 'o', color='r')
        for node in self._Z_boundary_nodes:
            plt.plot(node[0], node[1], 'o', color='b')
        plt.show()

    def plot_layout(self, X_check_idx: int = 40, Z_check_idx: int = 60):
        fig,ax = plt.subplots()
        for i in self.X_checks[X_check_idx]:
            plt.plot(*self.qubit_coords[i], 's', color='r', markersize=10, alpha=0.4)
        for i in self.Z_checks[Z_check_idx]:
            plt.plot(*self.qubit_coords[i], 'o', color='b', markersize=10, alpha=0.6)
        for x,y in self.qubit_coords:
            plt.plot(x, y, 'o', color='k')
        plt.show()
    
    def _optimize_SE_circuit(self, X_anc_idx=0, Z_anc_idx=0):
        X_checks = self.X_checks[X_anc_idx]
        Z_checks = self.Z_checks[Z_anc_idx]
        # Key principle: single fault on ancilla qubit should not propagate to
        # two data qubit faults along the same logical operator. Unless the
        # correction is easy and would complete a stabilizer...
    
    @classmethod
    def known_code(cls, n, k, d):
        # From Table I of arXiv:2504.09171
        if (n,k,d) == (288,8,12):
            return cls(
                X_pattern = [
                    ((0,0), (1,0)),
                    ((2,0), (2,1)),
                    ((2,1), (3,1)),
                    ((2,2), (3,2)),
                    ((0,2), (0,3)),
                    ((1,2), (1,3)),
                ],
                Z_pattern = [
                    ((0,0), (0,1)),
                    ((0,1), (0,2)),
                    ((0,2), (1,2)),
                    ((1,0), (2,0)),
                    ((2,0), (3,0)),
                    ((2,2), (2,3)),
                ],
                rotated=False,
                size=10,
            )
        elif (n,k,d) == (288, 8, 14):
            return cls(
                X_pattern = [
                    ((0,0), (1,0)),
                    ((0,0), (0,1)),
                    ((0,1), (1,1)),
                    ((1,1), (1,2)),
                    ((1,2), (0,2)),
                    ((0,2), (0,3)),
                    ((2,0), (3,0)),
                    ((2,2), (2,3)),
                ],
                Z_pattern = [
                    ((0,0), (1,0)),
                    ((0,2), (0,3)),
                    ((1,1), (2,1)),
                    ((2,0), (3,0)),
                    ((2,0), (2,1)),
                    ((2,1), (2,2)),
                    ((2,2), (2,3)),
                    ((2,2), (3,2)),
                ],
                rotated=False,
                size=10,
            )
        elif (n,k,d) == (288, 18, 13):
            return cls(
                X_pattern = [
                    ((0,0), (1,0)),
                    ((1,0), (1,1)),
                    ((1,1), (1,2)),
                    ((0,1), (0,2)),
                    ((0,3), (1,3)),
                    ((2,2), (3,2)),
                    ((3,0), (4,0)),
                    ((3,3), (3,4)),
                ],
                Z_pattern = [
                    ((0,0), (1,0)),
                    ((1,1), (1,2)),
                    ((0,3), (0,4)),
                    ((3,0), (3,1)),
                    ((2,2), (3,2)),
                    ((3,2), (4,2)),
                    ((2,3), (3,3)),
                    ((3,3), (3,4)),
                ],
                rotated=False,
                size=9,
            )
        elif (n,k,d) == (512, 18, 19):
            return cls(
                X_pattern = [
                    ((0,0), (1,0)),
                    ((1,0), (1,1)),
                    ((1,1), (1,2)),
                    ((0,1), (0,2)),
                    ((0,3), (1,3)),
                    ((2,2), (3,2)),
                    ((3,0), (4,0)),
                    ((3,3), (3,4)),
                ],
                Z_pattern = [
                    ((0,0), (1,0)),
                    ((1,1), (1,2)),
                    ((0,3), (0,4)),
                    ((3,0), (3,1)),
                    ((2,2), (3,2)),
                    ((3,2), (4,2)),
                    ((2,3), (3,3)),
                    ((3,3), (3,4)),
                ],
                rotated=False,
                size=13,
            )
        elif (n,k,d) == (512, 18, 23):
            raise NotImplementedError