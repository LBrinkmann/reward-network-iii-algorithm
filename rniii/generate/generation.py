import hashlib
import json
import random
import string
import yaml
import os
from collections import Counter

import networkx as nx
import numpy as np

from network import Network, Node, Edge
from environment import Environment

# from .utils import parse_network, calculate_q_value, calculate_trace

seed = 42
random.seed(seed)
np.random.seed(seed)


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


class NetworkGenerator:
    """
    Network Generator class
    """

    def __init__(self, environment: Environment):
        """
        Initializes a network generator object with parameters
        obtained from streamlit form
        Args:
            params (dict): parameter dictionary
        """

        self.network_objects = []
        self.networks = []
        self.env = environment

        # parameters for visualization
        self.node_size = 2200
        self.arc_rad = 0.1

        self.from_to = {
            (e_def.from_level, tl): e_def.rewards
            for e_def in self.env.edges
            for tl in e_def.to_levels
        }
        self.start_node = None

    def generate(self, n_networks):
        """
        Using the functions defined above this method generates networks with
        visualization info included.
        The generated network(s) are also saved in a json file at location
        specified by save_path
        """

        # sample and store training networks
        self.networks = []
        for _ in range(n_networks):
            g = self.sample_network()
            net = nx.json_graph.node_link_data(g)

            # NEW: shuffle randomly the order of the nodes in circular layout
            pos = nx.circular_layout(g)
            node_order = list(g.nodes(data=True))
            random.shuffle(node_order)
            random_pos = {}
            for a, b in zip(node_order, [pos[node] for node in g]):
                random_pos[a[0]] = b

            pos_map = {
                k: {"x": v[0] * 100, "y": v[1] * -1 * 100}
                for k, v in random_pos.items()
            }

            # NEW: add vertices for visualization purposes
            for ii, e in enumerate(g.edges()):
                if reversed(e) in g.edges():
                    net["links"][ii]["arc_type"] = "curved"
                    arc = nx.draw_networkx_edges(
                        g,
                        random_pos,
                        edgelist=[e],
                        node_size=self.node_size,
                        connectionstyle=f"arc3, rad = {self.arc_rad}",
                    )
                else:
                    net["links"][ii]["arc_type"] = "straight"
                    arc = nx.draw_networkx_edges(
                        g, random_pos, edgelist=[e], node_size=self.node_size
                    )

                vert = arc[0].get_path().vertices.T[:, :3] * 100

                net["links"][ii]["source_x"] = vert[0, 0]
                net["links"][ii]["source_y"] = -1 * vert[1, 0]
                net["links"][ii]["arc_x"] = vert[0, 1]
                net["links"][ii]["arc_y"] = -1 * vert[1, 1]
                net["links"][ii]["target_x"] = vert[0, 2]
                net["links"][ii]["target_y"] = -1 * vert[1, 2]

            network_id = hashlib.md5(
                json.dumps(net, sort_keys=True).encode("utf-8")
            ).hexdigest()

            c = Counter([e["source"] for e in net["links"]])

            if (
                    all(value == self.env.n_edges_per_node for value in c.values())
                    and len(list(c.keys())) == self.env.n_nodes
            ):
                create_network = self.create_network_object(
                    pos_map=pos_map,
                    n_steps=self.env.n_steps,
                    network_id=network_id,
                    **net,
                )
                self.networks.append(create_network)
                print(f"Network {len(self.networks)} created")
            else:
                print(
                    f"counter {c}, nodes are {list(c.keys())} "
                    f"(n={len(list(c.keys()))})"
                )

        return self.networks

    # individual network building functions
    #######################################
    def add_link(self, G, source_node, target_node):
        from_level = G.nodes[source_node]["level"]
        to_level = G.nodes[target_node]["level"]
        reward_idx = random.choice(self.from_to[(from_level, to_level)])
        reward = self.env.rewards[reward_idx]
        G.add_edge(
            source_node,
            target_node,
            reward=reward.reward,
            reward_idx=reward_idx,
            color=reward.color,
        )

    @staticmethod
    def add_new_node(G, level):
        idx = len(G)
        name = string.ascii_uppercase[idx % len(string.ascii_lowercase)]
        G.add_node(idx, name=name, level=level)
        return idx

    @staticmethod
    def nodes_random_sorted_by_in_degree(G, nodes):
        return sorted(
            nodes, key=lambda n: G.in_degree(n) + random.random() * 0.1, reverse=False
        )

    @staticmethod
    def nodes_random_sorted_by_out_degree(G, nodes):
        return sorted(
            nodes,
            key=lambda n: G.nodes[n]["level"]
                          + G.out_degree(n) * 0.1
                          + random.random() * 0.01,
            reverse=False,
        )

    def edge_is_allowed(self, G, source_node, target_node):
        if source_node == target_node:
            return False
        if target_node in G[source_node]:
            return False
        from_level = G.nodes[source_node]["level"]
        to_level = G.nodes[target_node]["level"]
        return (from_level, to_level) in self.from_to

    def allowed_target_nodes(self, G, nodes, source_node):
        return [node for node in nodes if self.edge_is_allowed(G, source_node, node)]

    def assign_levels(self, graph):
        levels = self.env.levels.copy()
        # min number of nodes to each level
        for level in levels:
            level.n_nodes = level.min_n_nodes

        # total number of nodes per levels
        n_nodes = sum([level.n_nodes for level in levels])
        assert n_nodes <= self.env.n_nodes

        # spread missing nodes over levels
        for i in range(self.env.n_nodes - n_nodes):
            # possible levels
            possible_levels = [
                level
                for level in levels
                if (level.max_n_nodes is None) or (level.n_nodes < level.max_n_nodes)
            ]
            # choose level
            level = random.choice(possible_levels)
            # add node to level
            level.n_nodes += 1

        # add nodes to graph
        for level in levels:
            for _ in range(level.n_nodes):
                node_idx = self.add_new_node(graph, level.idx)
                if level.is_start and self.start_node is None:
                    self.start_node = node_idx

    def sample_network(self):
        graph = nx.DiGraph()

        self.assign_levels(graph)
        for i in range(self.env.n_edges_per_node * self.env.n_nodes):
            allowed_source_nodes = [
                n
                for n in graph.nodes
                if graph.out_degree(n) < self.env.n_edges_per_node
            ]
            if len(allowed_source_nodes) == 0:
                raise ValueError("No allowed nodes to connect from.")
            source_node = self.nodes_random_sorted_by_out_degree(
                graph, allowed_source_nodes
            )[0]
            allowed_target_nodes = self.allowed_target_nodes(
                graph, graph.nodes, source_node
            )
            if len(allowed_target_nodes) == 0:
                raise ValueError("No allowed nodes to connect to.")
            target_node = self.nodes_random_sorted_by_in_degree(
                graph, allowed_target_nodes
            )[0]
            self.add_link(graph, source_node, target_node)
        return graph

    @staticmethod
    def parse_node(name, pos_map, level, id, **__):
        return Node(
            node_num=id,
            display_name=name,
            node_size=3,
            level=level,
            **pos_map[id],
        )

    @staticmethod
    def parse_link(source, target, **props):
        return Edge(source_num=source, target_num=target, **props)

    def create_network_object(
            self, pos_map, starting_node=0, *, nodes, links, network_id, **kwargs
    ):
        return Network(
            nodes=[self.parse_node(**n, pos_map=pos_map) for n in nodes],
            edges=[self.parse_link(**l) for l in links],
            starting_node=starting_node,
            network_id=network_id,
        )

    def save_as_json(self):
        """
        filename: path to save networks file to
        """
        #return json.dumps(self.networks)
        return json.dumps([ob.__dict__ for ob in self.networks])



if __name__ == "__main__":

    # --------Specify paths--------------------------
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    root_dir = os.sep.join(current_dir.split(os.sep)[:2])

    # Specify directories depending on system
    if root_dir == "/mnt":  # (cluster)
        user_name = os.sep.join(current_dir.split(os.sep)[4:5])
        home_dir = f"/mnt/beegfs/home/{user_name}"
        project_dir = os.path.join(home_dir, "CHM", "reward_networks_III", "reward-network-iii-algorithm")
        code_dir = os.path.join(project_dir, "generate")
        params_dir = os.path.join(project_dir, "params", "generate")
        data_dir = os.path.join(project_dir, "data")

    elif root_dir == "/Users":  # (local)
        project_dir = os.path.split(os.getcwd())[0]
        data_dir = os.path.join(project_dir, "data")
        params_dir = os.path.join(project_dir, "params", "generate")

    environment = load_yaml("../params/generate/default_environment.yml")
    generate_params = Environment(**environment)

    net_generator = NetworkGenerator(generate_params)
    networks = net_generator.generate(100)
    net_generator.save_as_json()
