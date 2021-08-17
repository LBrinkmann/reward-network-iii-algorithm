
import numpy as np
from mc.pruning_models.model import calculate_reward_transition_matrices
import torch as th
from scipy.special import softmax


def calculate_q_all(R, T, n_steps, G, device=th.device('cpu')):
    n_networks = T.shape[0]
    n_participants = G.shape[0]

    n_idx = n_networks * n_participants

    E = th.arange(n_networks).unsqueeze(0).repeat(n_participants, 1).reshape(n_idx)  # [i]-> e
    P = th.arange(n_participants).unsqueeze(1).repeat(1, n_networks).reshape(n_idx)  # [i]-> p

    Q = calculate_q(R, T, n_steps, G, P, E, device)
    Q = Q.reshape(n_participants, n_networks, 8, 6, 2)  # [p,n,m,s,a]
    return Q


def calculate_q(R, T, n_steps, G, P, N, device=th.device('cpu'), **_):
    r"""Calculates the Q matrix under aversive pruning. (Same then above, but different interface)

    The Q matrix indicates the expected reward (under aversive pruning)
    from a given node, at a given step for a given action. This method
    is calculating simultaneously the Q matrix for multiple networks and
    participants.

    In the output, each row is for a tuple of participant and environment (as indicated by P and E)

    Args:
        T: The transition matrix. A 4-D `Tensor` of type `th.float32` and shape
            `[e,s,t,a]`. A unit value indicates a possible
            transition from a source node to a target node.
        R: The reward matrix. A 3-D `Tensor` of type `th.float32` and shape
            `[e,s,a]`. The reward for a given action.
        n_steps: A integer. The number of moves/steps to plan ahead.
        G: The pruning factor. A 2-D `Tensor` of type `th.float32` and shape
            `[p,g]`. Each participants has two pruning factors.
            The first value in the second dimension is the general pruning factor,
            the second value  is the aversive pruning factor.
        P: Participant idx for each output index. A 1-D `Tensor` of type `th.int64` and shape
            `[i]`.
        N: Network idx for each output index. A 1-D `Tensor` of type `th.int64` and shape
            `[i]`.
        device: A torch device.

    Indices:
        i: general index
        n: network [0..n_network]
        p: participant [0..n_participants]
        s: source node of action [0..n_nodes]
        t: target node of action [0..n_nodes]
        a: action [0,1]
        b: action of target node [0,1]
        g: action type [general, specific]
        m: move / step within path [0..n_steps]


    Returns:
        Q matrix
            A 5-D `Tensor` of type `th.float32` and shape
            `[i, m, s, a]`
    """

    assert len(P) == len(N)

    n_index = P.shape[0]
    n_networks = T.shape[0]
    n_nodes = T.shape[1]
    n_participants = G.shape[0]

    idx = th.arange(n_index).unsqueeze(-1).unsqueeze(-1)  # [i,*,*]
    R_av = (R <= -100).type(th.int64)[N]  # [i,s,a]->g
    G_inv = (1 - G[P])  # [i,g,s,a]
    GT = G_inv[idx, R_av]  # [i,s,a]

    Q = th.zeros((len(P), n_steps, n_nodes, 2), device=device)  # [i,m,s,a]
    Q[:, n_steps-1] = R[N]  # [i,m,s,a]

    _T = T[N]  # [i,s,t,a]
    _R = R[N]  # [i,s,a]

    for m in range(n_steps-2, -1, -1):  # m: move / step
        # Q values for the next move
        Q_next = Q[:, m + 1]  # [i,t,b], because we are in the next move s->t and a->b

        # Possible Q values for each next move (b) for each of this moves (a)
        Q_possible_next = th.einsum('itb,ista->isab', Q_next, _T)  # [i,s,a,b]

        # for each action, the max of the possible next actions
        Q_next_max = th.max(Q_possible_next, axis=-1)[0]  # [i,s,a]

        Q[:, m, :, :] = _R + GT * Q_next_max  # [i,s,a]
    return Q  # [i,m,s,a]


# used by: derivative_classification.py
def calc_T_R_torch(networks, n_nodes=None, device=th.device('cpu')):
    T, R = calc_T_R(networks, n_nodes=n_nodes)
    T = th.tensor(T, dtype=th.float32, device=device)
    R = th.tensor(R, dtype=th.float32, device=device)
    return T, R


def calc_T_R(networks, n_nodes=None):
    r"""Calculates T and R matrices of given network list.

    Args:
        networks: Networks list.

    Returns:
        R (original node:action): Reward matrix.
            reward on transition from original node by doing the action
        T (original node:destination node:action): Transition matrix. one if
            action from original node leads to destination node, 0 otherwise
    """
    if n_nodes is None:
        n_nodes = len(networks[0]['nodes'])
    T_list = []
    R_list = []
    for network in networks:
        _T, _R = calculate_reward_transition_matrices(network, n_nodes)
        T_list.append(_T)
        R_list.append(_R)
    T = np.stack(T_list)
    R = np.stack(R_list)
    return T, R


def calculate_traces_stochastic(Q, T, R, starting_node):
    n_steps = Q.shape[0]  # Q: step,node,action
    NT = np.zeros((n_steps + 1), dtype='int64')  # node trace: step, starting node
    AT = np.zeros((n_steps), dtype='int64')  # action trace: step, starting node
    RT = np.zeros((n_steps), dtype='int64')  # reward trace: step, starting node
    NT[0] = starting_node
    PT = softmax(Q, axis=-1)
    for l in range(0, n_steps):
        AT[l] = np.random.choice(a=[0, 1], size=1, p=PT[l, NT[l]])
        NT[l+1] = np.argmax(T[NT[l], :, AT[l]], axis=0)
        RT[l] = R[NT[l], AT[l]]
    RT_tot = np.sum(RT, axis=0)
    return RT_tot, AT, NT, RT


def calculate_expected_reward(
        Q, T, R, ST=None, beta=None, stochastic=False, device=th.device('cpu')):
    r"""Calculates the expected reward.

    This method takes a Q matrix and a temperature `beta` and
    calculates the expected reward for a set of networks.

    Args:
        Q: The Q matrix. A 5-D `Tensor` of type `th.float32` and shape
            `[p, n, m, s, a]`. The expected
            reward of a given action.
        T: The transition matrix. A 4-D `Tensor` of type `th.float32` and shape
            `[n, s, t, a]`. A unit value indicates a possible
            transition from a source node to a target node.
        R: The reward matrix. A 3-D `Tensor` of type `th.float32` and shape
            `[n, s, a]`. The reward for a given action.
        ST: A 1-D `Tensor` of type `th.int` and size `[n]->f` .
            Starting nodes to be used for each network. If None, reward is calculated
            for all starting nodes.
        beta: A float or None. The inverse temperature. Greedy, if None.
        device: A torch device.

    Indices:
        n: network [0..n_network]
        p: participant [0..n_participants]
        s: source node of action [0..n_nodes]
        t: target node of action [0..n_nodes]
        a: action [0,1]
        m: move / step within path [0..n_steps]
        f: starting node of the network [0..n_nodes]

    Returns:
        Expected reward
            A 3-D `Tensor` of type `th.float32` and shape
            `[participant, environment, startingNode]` or `[participant, environment]`
            if ST is defined.
    """
    n_user = Q.shape[0]
    n_envs = Q.shape[1]
    n_steps = Q.shape[2]
    n_nodes = Q.shape[3]

    if beta is None:
        P = th.zeros_like(Q, device=device)  # [p,n,m,s,a]
        P = th.where(Q[:, :, :, :, [0]] == Q[:, :, :, :, [1]],
                     th.tensor(0.5, device=device), P)  # [p,n,m,s,a]
        P = th.where(Q[:, :, :, :, [0]] > Q[:, :, :, :, [1]],
                     th.tensor([[[[[1., 0.]]]]], device=device), P)  # [p,n,m,s,a]
        P = th.where(Q[:, :, :, :, [0]] < Q[:, :, :, :, [1]],
                     th.tensor([[[[[0., 1.]]]]], device=device), P)  # [p,n,m,s,a]
    else:
        P = th.nn.functional.softmax(
            Q*beta[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis], dim=-1)  # [p,n,m,s,a]

    if stochastic:
        m = th.distributions.OneHotCategorical(P.reshape(-1, 2))  # [p*n*m*s,a]
        P = m.sample().reshape(Q.shape)  # [p,n,m,s,a]

    if ST is None:
        p_source = th.zeros((n_user, n_envs, n_nodes, n_nodes), device=device)  # [p,n,f,s]
        for i in range(n_nodes):
            p_source[:, :, i, i] = 1
        e_reward = th.zeros((n_user, n_envs, n_nodes), device=device)  # [p,n,f]
    else:
        p_source = th.zeros((1, n_envs, 1, n_nodes), device=device)  # [*,n,*,s]
        p_source[0, np.arange(n_envs), 0, ST] = 1  # [*,n,*,s]
        p_source = p_source.repeat(n_user, 1, 1, 1)  # [p,n,*,s]

        e_reward = th.zeros((n_user, n_envs, 1), device=device)  # [p,n,*]

    for m in range(0, n_steps):
        # P(s, a | m) = P(s | m) * P(a | s, m)
        p_source_action = th.einsum('pnfs,pnsa->pnfsa', p_source, P[:, :, m])  # [p,n,f,s,a]

        # P(s | m + 1) = P(t | m) = P(s, a | m) * T(s, t, a)
        p_source = th.einsum('pnfsa,nsta->pnft', p_source_action, T)  # [p,n,f,t]

        # E(reward | m) = P(a, s | m) * R(s, a)
        e_reward += th.einsum('pnfsa,nsa->pnf', p_source_action, R)  # [p,n,f]

    if ST is None:
        return e_reward  # [p,n,f]
    else:
        return e_reward[:, :, 0]  # [p,n]


# legacy


def calculate_traces_meanfield(Q, T, R, ST, device=th.device('cpu')):
    beta = th.ones(1, device=device)
    return calculate_traces(Q, T, R, ST=ST, beta=beta, device=device)


# testing


def G_fake(n_participants, seed=None):
    np.random.seed(seed)
    gamma_g = np.random.uniform(low=0.1, high=0.3, size=n_participants)
    gamma_s = np.random.uniform(low=0.3, high=0.5, size=n_participants)
    G = np.column_stack((gamma_g, gamma_s))
    return G


def calc_S_fake(networks, n_nodes, n_participants, n_steps, seed=None):
    T, R = calc_T_R(networks, n_nodes)
    G = G_fake(n_participants, seed)
    Q = calc_all_Q(T, R, G, n_steps)
    Q = Q + np.random.normal(100, 30, Q.shape)
    S_fake = calc_Q_correct(Q)
    return G, S_fake


# archive

# legacy names

# used by: derivative_classification.py
calculate_q_matrix_avpruning = calculate_q_all

# used by: derivative_classification.py
calculate_q_matrix_avpruning2 = calculate_q

calculate_traces = calculate_expected_reward


# legacy method


def calc_all_Q(T, R, G, n_steps):
    T = th.tensor(T, dtype=th.float32)
    R = th.tensor(R, dtype=th.float32)
    G = th.tensor(G, dtype=th.float32)
    Q = calculate_q_matrix_avpruning(R, T, n_steps, G)
    return Q.numpy()


def calc_Q_correct(Q):
    Q_corr = (Q.max(axis=-1, keepdims=1) == Q).astype(int)
    return Q_corr


def calc_Q_correct_torch(Q):
    Q_corr = th.eq(th.max(Q, dim=-1, keepdim=True)[0], Q).type(th.float32)
    return Q_corr
