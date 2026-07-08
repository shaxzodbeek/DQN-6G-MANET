import networkx as nx

class ShortestPathRouting:
    def get_next_hop(self, G, current, dest):
        try:
            path = nx.shortest_path(G, source=current, target=dest, weight='weight')
            if len(path) > 1: return path[1]
        except nx.NetworkXNoPath: return None
        return None
