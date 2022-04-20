"""Usage: get_solutions.py PARAMETER_YAML NETWORKS_JSON OUTPUT_FOLDER

....

Arguments:
    PARAMETER_YAML        A yaml parameter file.
    NETWORKS_JSON         A json of generated networks.

Outputs:
    DF_SOLUTIONS        A dataframe with all solutions.
"""


from docopt import docopt
import numpy as np
import pandas as pd
from scripts.utils.utils import load_yaml
from scripts.utils import store
from itertools import product
from scripts.pruning_models.model import calculate_reward_transition_matrices_new, calculate_q_matrix_avpruning


def get_actions(n_steps):
    # creates all possible action combinations in a network
    for actions in product((0, 1), repeat=n_steps):
        yield actions


def get_all_solutions(network, params):
    # T: 1 for link - source node, destination node, action
    # L: destination node - source node, action
    # R: reward - source node, action 
    T, R, L = calculate_reward_transition_matrices_new(network, params['n_nodes'])
    # Q: action value - remaining steps, source node, action
    Q = calculate_q_matrix_avpruning(R, T, params['n_steps'], gamma_g=0.0, gamma_s=0.0) 
    rows = []
    for i, actions in enumerate(get_actions(params['n_steps'])):
        action_str = ''.join(str(a) for a in actions)
        source_node = network['starting_node']
        cum_reward = 0
        solution_rows = []
        for step, a in enumerate(actions):
            remaining_steps = params['n_steps'] - step - 1
            target_node = L[source_node,a]
            reward = R[source_node, a]
            cum_reward += reward
            solution_rows.append(
                {
                    'network_id': network['network_id'],
                    'solution_id': i,
                    'actions': action_str,
                    'action': a,
                    'step': step,
                    'source_node': source_node,
                    'target_node': target_node,
                    'reward': reward,
                    'other_reward': R[source_node, (1-a)],
                    'cum_reward': cum_reward,
                    '2step_lookahead': Q[1,source_node,a],
                    'other_2step_lookahead': Q[1,source_node,(1-a)],
                    'lookahead': Q[remaining_steps,source_node,a],
                    'other_lookahead': Q[remaining_steps,source_node,(1-a)],
                }
            )
            source_node = target_node
        rows.extend({**r, 'total_reward': cum_reward} for r in solution_rows)
    max_total_reward = max(r['total_reward'] for r in rows)
    rows = [
        {**r, 'total_regret': max_total_reward - r['total_reward']}
        for r in rows
        if (max_total_reward - r['total_reward']) < params['regret_cap']
    ]
    return rows


def create_df(networks, params):
    all_sols = []
    for network in networks:
        sol_list = get_all_solutions(network, params)
        all_sols.extend(sol_list)
    df = pd.DataFrame(data=all_sols)
    return df

def main(parameter_yaml, networks_json, output_folder):
    params = load_yaml(parameter_yaml)
    networks = store.load_json(networks_json)
    df = create_df(networks, params)
    store.store_df(df, output_folder, 'solutions')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    arguments_low = {k.lower(): v for k, v in arguments.items()}
    main(**arguments_low)
