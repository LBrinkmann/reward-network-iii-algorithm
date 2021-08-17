import torch as th


def map_source_target_action_map(networks, id_name='environmentId'):
    """
    Creates a map for the action idx [0,1].
    The action with the smaller targetId is always the 0 action, the one with the larger
    targetId is the 1 action.

    Returns:
        dict (environmentId, sourceId, targetId): [0,1]

    """
    # dict with key (envid,source,target) and value action
    stam = {}
    for net in networks:
        actions = [a for a in net['actions']]
        actions.sort(key=lambda d: d['targetId'])
        counter = {}
        for action in actions:
            action_idx = counter.get(action['sourceId'], 0)
            counter[action['sourceId']] = action_idx + 1
            stam[(net[id_name], action['sourceId'], action['targetId'])] = action_idx
    return stam


def calc_T_R(networks, device=th.device('cpu')):
    T, RI, RL = calc_T_RI_RL(networks, device)
    R = RL[RI]
    return T, R


def calc_T_R_numpy(networks):
    T, R = calc_T_R(networks, device=th.device('cpu'))
    return T.numpy(), R.numpy()


def calc_T_RI_RL(networks, device=th.device('cpu')):
    """
        Calculate the reward and transition matrices.
        R (original node : action):
            reward on transition from original node by doing the action
        T (original node : desitination node : action):
            one if action from original node leads to destination node, 0 otherwise

    """
    n_nodes = len(networks[0]['nodes'])
    n_networks = len(networks)

    # create map from reward to rewardIdx
    rewards = list(set(a['reward'] for n in networks for a in n['actions']))
    rewards = sorted(rewards)
    reward_idx_map = {r: i for i, r in enumerate(rewards)}

    # create map form source, target to action
    stam = map_source_target_action_map(networks, id_name='networkId')
    n_actions = max(stam.values()) + 1

    # original node, destination node, action
    T = th.zeros((n_networks, n_nodes, n_nodes, n_actions), dtype=th.float, device=device)
    RI = th.zeros((n_networks, n_nodes, n_actions), dtype=th.int64,
                  device=device)  # original node, action
    RL = th.tensor(rewards, dtype=th.float, device=device)
    for n_idx, network in enumerate(networks):
        n_id = network['networkId']
        for action in network['actions']:
            s_idx = action['sourceId']
            t_idx = action['targetId']
            r_idx = reward_idx_map[action['reward']]
            a_idx = stam[(n_id, s_idx, t_idx)]
            RI[n_idx, s_idx, a_idx] = r_idx
            T[n_idx, s_idx, t_idx, a_idx] = 1
    return T, RI, RL
