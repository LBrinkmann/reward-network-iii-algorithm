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
from scripts.pruning_models.classification import apply_models
from scripts.pruning_models.model import calculate_reward_transition_matrices_new, calculate_q_matrix_avpruning, calculate_traces

# nodes from 0 to 5


def get_actions(n_steps):
    # creates all possible action combinations in a network
    all_acts = []
    for j in product((0, 1), repeat=n_steps):
        all_acts.append(j)
    return all_acts


def make_row(network, solution_idx, RT, AT, NT, ORT, max_tot_reward):
    """
        Make dataframe rows for each solution for storage.

    """

    return {
        'Environment_ID': network['network_id'], 'Solution_IDx': solution_idx,
        'Starting_Node': network['starting_node'], 'Total_Reward': sum(RT), 'Lookahead_Reward': max_tot_reward, 'Action_Trace': AT,
        'Node_Trace': NT, 'Reward_Trace': RT, 'Other_Reward_Trace': ORT, 'Action_IDx': np.arange(8)}


def get_all_solutions(network, params):
    all_acts = get_actions(params['n_steps'])
    solution_list = []
    T, R = calculate_reward_transition_matrices_new(network, params['n_nodes'])
    Q = calculate_q_matrix_avpruning(R, T, params['n_steps'], gamma_g=0.0, gamma_s=0.0)
    max_tot_reward, aaa, bbb, ccc, ddd, eee = calculate_traces(Q, T, R)
    for i in range(2**params['n_steps']):
        solution_idx = i
        NT = np.zeros((params['n_steps'] + 1), dtype=np.int64)  # node trace: step
        AT = np.array(all_acts[i])  # action trace: step
        RT = np.zeros((params['n_steps']), dtype=np.int64)  # reward trace: step
        ORT = np.zeros((params['n_steps']), dtype=np.int64)  # other reward trace: step
        OAT = 1-AT
        starting_node = network['starting_node']
        NT[0] = starting_node
        for j in range(params['n_steps']):
            if j < 1:
                target_node = np.where(T[starting_node] > 0.5)[0][AT[j]]
                reward = R[starting_node][AT[j]]
                other_reward = R[starting_node][OAT[j]]
            else:
                reward = R[target_node][AT[j]]
                other_reward = R[target_node][OAT[j]]
                target_node = np.where(T[target_node] > 0.5)[0][AT[j]]
            NT[j+1] = target_node
            RT[j] = reward
            ORT[j] = other_reward
        solution_list.extend([make_row(network, solution_idx, RT, AT,
                             NT, ORT, max_tot_reward[starting_node])])
    return solution_list


def create_df(networks, params):
    all_sols = []
    for network in networks:
        sol_list = get_all_solutions(network, params)
        all_sols.extend(sol_list)
    df = pd.DataFrame(data=all_sols)
    return df


def get_df_metrics(df, params):
    df['Total_Regret'] = df['Lookahead_Reward'] - df['Total_Reward']
    df['N_Nodes'] = df.apply(add_n_unique, axis=1)
    df['N_Myopic'] = df.apply(add_n_myopic, axis=1)
    df['N_Strict_Myopic'] = df.apply(add_n_strict_myopic, axis=1)
    df['N_Large_Cost'] = df.apply(add_n_large, large_cost=params['large_cost'], axis=1)
    df['N_Large_Cost_Beginning'] = df.apply(
        add_n_large_start, n_steps=params['n_steps'], large_cost=params['large_cost'], axis=1)
    df = df.drop(['Action_Trace', 'Node_Trace', 'Reward_Trace',
                  'Other_Reward_Trace', 'Lookahead_Reward'], axis=1)
    return df


def expand_actions(df_walk, lst_col):
    '''expand lst_col (actions) column to rows. 1 row -> 8 rows '''
    df_walk_actions = pd.DataFrame({col: np.repeat(df_walk[col].values, df_walk[lst_col].str.len())
                                    for col in df_walk.columns.difference([lst_col])}).assign(**{lst_col: np.concatenate(df_walk[lst_col].values)})[df_walk.columns.tolist()]
    return df_walk_actions


def add_n_unique(row):
    return len(np.unique(row['Node_Trace']))


def add_n_myopic(row):
    return sum(row['Reward_Trace'] >= row['Other_Reward_Trace'])


def add_n_strict_myopic(row):
    return sum(row['Reward_Trace'] > row['Other_Reward_Trace'])


def add_n_large(row, large_cost):
    return sum(row['Reward_Trace'] < large_cost+5)


def add_n_large_start(row, n_steps, large_cost):
    return sum(row['Reward_Trace'][:int(n_steps/2)] < large_cost+5)


def add_source(row):
    return row['Node_Trace'][row['Action_IDx']]


def add_target(row):
    return row['Node_Trace'][row['Action_IDx']+1]


def add_action_reward(row):
    return row['Reward_Trace'][row['Action_IDx']]


def add_other_reward(row):
    return row['Other_Reward_Trace'][row['Action_IDx']]


def manipulate_df(df, params):
    df = expand_actions(df, 'Action_IDx')
    df['Source_Node'] = df.apply(add_source, axis=1)
    df['Target_Node'] = df.apply(add_target, axis=1)
    df['Action_Reward'] = df.apply(add_action_reward, axis=1)
    df['Other_Reward'] = df.apply(add_other_reward, axis=1)
    df['Total_Regret'] = df['Total_Reward'] - df['Lookahead_Reward']
    df = df[df['Total_Regret'] < params['regret_cap']]
    df = df.drop(['Action_Trace', 'Node_Trace', 'Reward_Trace',
                  'Other_Reward_Trace'], axis=1)
    return df


def main(parameter_yaml, networks_json, output_folder):
    params = load_yaml(parameter_yaml)
    networks = store.load_json(networks_json)
    df = create_df(networks, params)
    df_metrics = df.copy()
    df_metrics = get_df_metrics(df_metrics, params)
    df = manipulate_df(df, params)
    store.store_df(df, output_folder, 'solutions')
    store.store_df(df_metrics, output_folder, 'metrics')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    arguments_low = {k.lower(): v for k, v in arguments.items()}
    main(**arguments_low)
