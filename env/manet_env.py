import numpy as np
import networkx as nx

class MANETEnv:
    def __init__(self, num_nodes=50, area_size=1000, comm_range=250, init_energy=200.0):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.comm_range = comm_range

        self.positions = np.random.rand(num_nodes, 2) * area_size
        self.destinations = np.random.rand(num_nodes, 2) * area_size
        self.speeds = np.random.uniform(2.0, 10.0, num_nodes)

        self.energies = np.full(num_nodes, init_energy)
        self.dead_nodes = np.zeros(num_nodes, dtype=bool)

        self.e_tx = 0.05
        self.e_rx = 0.02
        self.e_idle = 0.001

        self.G = nx.Graph()
        self.update_topology()

    def update_topology(self):
        self.G.clear()
        active_nodes = np.where(~self.dead_nodes)[0]
        self.G.add_nodes_from(active_nodes)

        if len(active_nodes) < 2: return

        active_positions = self.positions[active_nodes]
        diff = active_positions[:, np.newaxis, :] - active_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        adj_matrix = (distances <= self.comm_range) & (distances > 0)
        edges = np.argwhere(adj_matrix)

        for u_idx, v_idx in edges:
            if u_idx < v_idx:
                self.G.add_edge(active_nodes[u_idx], active_nodes[v_idx], weight=distances[u_idx, v_idx])

    def step_mobility(self):
        directions = self.destinations - self.positions
        distances = np.linalg.norm(directions, axis=1)

        reached = distances < self.speeds
        self.positions[reached] = self.destinations[reached]
        self.destinations[reached] = np.random.rand(np.sum(reached), 2) * self.area_size

        move_vectors = (directions / (distances[:, None] + 1e-9)) * self.speeds[:, None]
        self.positions[~reached] += move_vectors[~reached]
        self.positions = np.clip(self.positions, 0, self.area_size)

        self.consume_energy(np.arange(self.num_nodes), amount=self.e_idle)
        self.update_topology()

    def consume_energy(self, nodes, amount):
        valid_nodes = [n for n in nodes if not self.dead_nodes[n]]
        self.energies[valid_nodes] -= amount
        just_died = (self.energies <= 0) & (~self.dead_nodes)
        self.dead_nodes[just_died] = True
        self.energies[self.dead_nodes] = 0.0

    def transmit_packet(self, sender, receiver):
        if self.dead_nodes[sender] or self.dead_nodes[receiver]: return False
        if self.G.has_edge(sender, receiver):
            self.consume_energy([sender], self.e_tx)
            self.consume_energy([receiver], self.e_rx)
            return True
        return False

    def get_neighbors(self, node):
        return list(self.G.neighbors(node)) if node in self.G else []
