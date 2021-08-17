"""
Terminology:
    node: nodes of the network
    action: possible actions from each node (typically 2)
    reward: each action from each node has a related reward
    transition: (node, action) tuple
    remaining: the number of remaining actions left before the game ends
    state: (node, remaining) tuple
"""

import numpy as np
import hashlib
import uuid


def calculate_reward_transition_matrices_old(network, n_nodes):
    """
        Calculate the reward and transition matrices.
        R (original node : action):
            reward on transition from original node by doing the action
        T (original node : desitination node : action):
            one if action from original node leads to destination node, 0 otherwise

    """
    T = np.zeros((n_nodes, n_nodes, 2))  # original node, destination node, action
    R = np.zeros((n_nodes, 2))  # original node, action

    for j in range(len(network['links'])):
        link = network['links'][j]
        origin = link['source']
        destination = link['target']
        action = 0 if link['action'] == 'L' else 1
        reward = link['weight']
        R[origin-1, action] = reward
        T[origin-1, destination-1, action] = 1
    return T, R


def calculate_reward_transition_matrices_new(network, n_nodes):
    """
        Calculate the reward and transition matrices.
        R (original node : action):
            reward on transition from original node by doing the action
        T (original node : desitination node : action):
            one if action from original node leads to destination node, 0 otherwise

    """
    T = np.zeros((n_nodes, n_nodes, 2))  # original node, destination node, action
    R = np.zeros((n_nodes, 2))  # original node, action

    actions = [a for a in network['actions']]
    actions.sort(key=lambda d: d['targetId'])
    counter = {}

    for action in actions:
        action_idx = counter.get(action['sourceId'], 0)
        counter[action['sourceId']] = action_idx + 1
        R[action['sourceId'], action_idx] = action['reward']
        T[action['sourceId'], action['targetId'], action_idx] = 1
    return T, R


def calculate_reward_transition_matrices(network, n_nodes):
    if 'links' in network:
        print('links')
        return calculate_reward_transition_matrices_old(network, n_nodes)
    else:
        print('nolinks')
        return calculate_reward_transition_matrices_new(network, n_nodes)


def calculate_q_matrix(R, T, n_steps, gamma):
    """
        Calculate the total reward matrix with pruning.
        Q (remaining steps:original node:action): total reward until that step
        from original node with actions

    """
    n_nodes = T.shape[0]
    Q = np.zeros((n_steps, n_nodes, 2))  # remaining, original node, action
    Q[0] = R
    for k in range(n_steps - 1):
        Q1 = np.einsum('jb,ija->iab', Q[k], T)
        Q2 = np.max(Q1, axis=2)
        Q[k+1] = R + (1-gamma) * Q2
    return Q


def calculate_q_matrix_avpruning(R, T, n_steps, gamma_g, gamma_s):
    """
        Calculate the total reward matrix with aversive pruning.
        Q (remaining steps:original node:action): total reward until that step from original
        node with actions

    """
    n_nodes = T.shape[0]
    Q = np.zeros((n_steps, n_nodes, 2))  # remaining, original node, action
    Q[0] = R
    for k in range(n_steps - 1):
        # Q value of all reachable states (with a single action) from a given state
        Q1 = np.einsum('jb,ija->iab', Q[k], T)
        # for each reachable state, the largest Q value
        Q2 = np.max(Q1, axis=2)

        # Q value of current (state, actions) is discounted largest Q value of reachable state
        # plus the reward of the corresponding action
        for s in range(6):
            for g in range(2):
                if R[s, g] <= -100:
                    Q[k+1, s, g] = R[s, g] + (1-gamma_s) * Q2[s, g]
                else:
                    Q[k+1, s, g] = R[s, g] + (1-gamma_g) * Q2[s, g]
    return Q


def check_reachability(T, n_steps):
    """
        Check node reachability.

    """
    T_any = np.max(T, axis=2)
    reachable = np.zeros(T.shape)
    np.fill_diagonal(reachable, 1)
    for k in range(n_steps - 1):
        reachable = np.einsum('ij,jk->ik', reachable, T_any)
    return reachable


def calculate_traces_no_planing(T, R, n_steps, worst=False):
    """
        Calculate action, node, reward, reward difference, total reward difference, and
        total reward matrices with no planing.

        AT(step:starting node): action trace from each starting node
        NT(step:starting node): node trace from each starting node
        RT(step:starting node): reward trace from each starting node
        RD(step:starting node): reward difference trace from each starting node
        RT(starting node): total rewards from each starting node

    """
    n_nodes = T.shape[0]
    NT = np.zeros((n_steps + 1, n_nodes), dtype='int64')  # node trace: step, starting node
    AT = np.zeros((n_steps, n_nodes), dtype='int64')  # action trace: step, starting node
    RT = np.zeros((n_steps, n_nodes), dtype='int64')  # reward trace: step, starting node
    RD = np.zeros((n_steps, n_nodes), dtype='int64')  # reward diff trace: step, starting node

    NT[0] = np.arange(0, n_nodes)
    for l in range(0, n_steps):
        if worst:
            AT[l] = np.argmin(R[NT[l]], axis=1)
        else:
            AT[l] = np.argmax(R[NT[l]], axis=1)
        NT[l+1] = np.argmax(T[NT[l], :, AT[l]], axis=1)
        RT[l] = R[NT[l], AT[l]]
        RD[l] = R[NT[l], AT[l]] - R[NT[l], 1 - AT[l]]
    RT_tot = np.sum(RT, axis=0)
    QD = RD
    return RT_tot, AT, NT, RT, RD, QD


def calculate_traces(Q, T, R):
    """
        Calculate action, node, reward, reward difference, total reward difference, and
        total reward matrices with planing.

        AT(step:starting node): action trace from each starting node
        NT(step:starting node): node trace from each starting node
        RT(step:starting node): reward trace from each starting node
        RD(step:starting node): reward difference trace from each starting node
        RT(starting node): total rewards from each starting node

    """
    n_nodes = T.shape[0]
    n_steps = Q.shape[0]
    NT = np.zeros((n_steps + 1, n_nodes), dtype='int64')  # node trace: step, starting node
    AT = np.zeros((n_steps, n_nodes), dtype='int64')  # action trace: step, starting node
    RT = np.zeros((n_steps, n_nodes), dtype='int64')  # reward trace: step, starting node
    RD = np.zeros((n_steps, n_nodes), dtype='int64')  # reward diff trace: step, starting node
    QD = np.zeros((n_steps, n_nodes), dtype='int64')  # reward diff trace: step, starting node
    NT[0] = np.arange(0, n_nodes)
    for l in range(0, n_steps):
        AT[l] = np.argmax(Q[-l-1, NT[l]], axis=1)
        NT[l+1] = np.argmax(T[NT[l], :, AT[l]], axis=1)
        RT[l] = R[NT[l], AT[l]]
        RD[l] = R[NT[l], AT[l]] - R[NT[l], 1 - AT[l]]
        QD[l] = Q[-l-1, NT[l], AT[l]] - Q[-l-1, NT[l], 1 - AT[l]]
    RT_tot = np.sum(RT, axis=0)
    return RT_tot, AT, NT, RT, RD, QD


def make_row(network, RT_tot, AT, NT, RT, RD, QD, n, n_steps, model_name, model_parameter):
    """
        Make dataframe rows for each environment for storage.

    """

    new_id = hashlib.md5((network['network_id'] + str(n)).encode()).hexdigest()
    solution_id = str(uuid.uuid4())
    model_parameter_capitalize = capitalize_keys(model_parameter)

    return {
        'Environment_ID': new_id, 'Solution_ID': solution_id,  'Network_ID': network['network_id'],
        'Starting_Node': n, 'Depth': n_steps, 'Total_Reward': RT_tot[n], 'Action_Trace': AT[:, n],
        'Node_Trace': NT[:, n], 'Reward_Trace': RT[:, n], 'Reward_Diff': RD[:, n],
        'Q_Diff': QD[:, n], 'Model_Name': model_name, **model_parameter_capitalize}


def capitalize_name(name):
    return '_'.join(s.capitalize() for s in name.split('_'))


def capitalize_keys(d):
    return {capitalize_name(k): v for k, v in d.items()}


def eval_network(network, n_steps, n_nodes, model='pruning', model_parameter={}, model_name=None):
    """
        Evaluate each network.
    """
    T, R = calculate_reward_transition_matrices(network, n_nodes=n_nodes)

    if model == 'pruning':
        Q = calculate_q_matrix(R, T, n_steps=n_steps, **model_parameter)
        RT_tot, AT, NT, RT, RD, QD = calculate_traces(Q, T, R)
    elif model == 'avpruning':
        Q = calculate_q_matrix_avpruning(R, T, n_steps=n_steps, **model_parameter)
        RT_tot, AT, NT, RT, RD, QD = calculate_traces(Q, T, R)
    elif model == 'no_planing':
        RT_tot, AT, NT, RT, RD, QD = calculate_traces_no_planing(T, R, n_steps, **model_parameter)

    return [make_row(network, RT_tot, AT, NT, RT, RD, QD, n, n_steps, model_name, model_parameter) for n in range(n_nodes)]
