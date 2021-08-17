"""Usage: generate.py PARAMETER_YAML OUTPUT_FOLDER

....

Arguments:
    PARAMETER_YAML        A yaml parameter file.
    OUTPUT_FOLDER         A folder used for the outputs.

Outputs:
    ....
"""


from docopt import docopt
import networkx as nx
import numpy as np
import uuid
import random
from scripts.utils.utils import load_yaml
from scripts.utils import store
import ipdb
# nodes from 0 to 5
# actions are 0 and 1


def generate_networks(n_networks, weights, n_nodes, seed=None):
    """
    This well create a list of networks in json style format.
    """
    # ToDo : fix the generation process. it generates dead nodes (nodes without any incoming paths)
    # ToDo: seperation between node id (0 to 5), and node label (e.g. city names)
    random.seed(seed)
    np.random.seed(seed)
    weights = np.array(weights)
    nodes = np.arange(n_nodes)
    actions = [0, 1]
    graph_list = []
    while len(graph_list) < n_networks:
        digraph = nx.DiGraph(directed=True)
        weight_count = np.zeros(4, dtype='int32')
        for i in range(0, n_nodes):
            for j in range(2):
                if j == 0:
                    target = int(np.random.choice(np.delete(nodes, [i])))
                    weight = int(np.random.choice(weights[np.where(weight_count < 3)[0]]))
                    weight_count[np.where(weights == weight)[0][0]] += 1
                    action = np.random.choice(actions)
                    digraph.add_edge(i, target, weight=weight, action=action)
                else:
                    target = int(np.random.choice(np.delete(nodes, [i, target])))
                    weight = int(np.random.choice(weights[np.where(weight_count < 3)[0]]))
                    weight_count[np.where(weights == weight)[0][0]] += 1
                    action = [set(actions)-set([action])][0].pop()
                    digraph.add_edge(i, target, weight=weight, action=action)
        if nx.algorithms.components.strongly_connected.is_strongly_connected(digraph) == True:
            digraph_json = nx.json_graph.node_link_data(digraph)
            digraph_json['network_id'] = f'{seed}_{len(graph_list)}'
            graph_list.append(digraph_json)
    return graph_list


def main(parameter_yaml, output_folder):
    params = load_yaml(parameter_yaml)
    networks = generate_networks(**params)
    store.store_json(networks, output_folder, 'networks')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    arguments_low = {k.lower(): v for k, v in arguments.items()}
    main(**arguments_low)
