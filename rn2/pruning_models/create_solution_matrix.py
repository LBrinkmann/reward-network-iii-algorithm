from mc.utils.df_utils import selector
import pandas as pd
import numpy as np
import torch as th
import mc.pruning_models.torch_model as tm
from mc.pruning_models.transition_matrix import map_source_target_action_map


def create_solution_meta_information(
        solutions, batches, valid_cutoff, fix_environment_id=True, filter_treatment=None,
        selectors={}):
    # filter solutions
    if filter_treatment is not None:
        for k, v in filter_treatment.items():
            solutions = [s for s in solutions if s['treatment'][k] == v]

    keys = [
        '_id', 'batchId', 'playerId', 'isValid', 'createdAt',
        'previousSolutionId', 'chainId', 'networkId', 'environmentId'
    ]

    df_meta = pd.DataFrame.from_records(
        [{k: {**s, **s['treatment']}.get(k) for k in keys} for s in solutions])

    # the environment id, is in fact the network id
    if fix_environment_id:
        raise NotImplementedError("Not supported currently")
        df_meta['environmentId'] = df_meta.rename(columns={'networkId': 'environmentId'})
    # idx for the local usage
    df_meta['solutionIdx'] = np.arange(len(df_meta))

    solution_idx_map = pd.Series(
        df_meta['solutionIdx'].values, index=df_meta['_id'], dtype=pd.Int64Dtype())
    df_meta['previousSolutionIdx'] = df_meta['previousSolutionId'].map(solution_idx_map)

    df_meta = df_meta[df_meta['batchId'].isin(batches) & df_meta['isValid']]

    df_meta['createdAt'] = pd.to_datetime(df_meta['createdAt'])
    df_meta['chainPos'] = df_meta.groupby('chainId')['createdAt'].rank().astype(int) - 1

    mainly_valid_player = df_meta.groupby(['playerId'])['isValid'].transform('sum') > valid_cutoff

    # filter out machine solutions
    is_actual_player = ~df_meta['playerId'].isnull()

    # filter for relevants batches, valid player and valid solutions
    df_meta = df_meta[mainly_valid_player & is_actual_player]

    w = selector(df_meta, selectors)
    df_meta = df_meta[w]

    # add participants idx
    df_meta = df_meta.set_index(['playerId', 'environmentId'], verify_integrity=True)

    return solutions, df_meta


def process(environments, solutions, solutions_meta):
    data = _process(
        environments, solutions, solutions_meta, create_old=True)
    check_new_s(**data)
    return data['SO'].numpy(), data['RO'].numpy(), data['VO'].numpy()


def check_new_s(participant, network, move, source, action, SO, **_):
    S_test = SO[participant, network, move, source]
    S_test_bool = th.argmax(S_test, dim=-1)
    assert (S_test_bool == action).all()


def _process(
        networks, solutions, solutions_meta, create_old=True, debug=True, device=th.device('cpu')):
    solutions_meta = solutions_meta.reset_index()
    first_solution = solutions[solutions_meta.iloc[0]['solutionIdx']]
    n_steps = len(first_solution['actions'])
    n_nodes = len(networks[0]['nodes'])
    n_solutions = len(solutions_meta)

    # create participant idx
    participant_ids = solutions_meta['playerId'].unique()
    participants_idx = {p: i for i, p in enumerate(participant_ids)}
    solutions_meta['participantsIdx'] = solutions_meta['playerId'].map(participants_idx)
    n_participants = len(participant_ids)

    # create network idx
    network_ids = [e['networkId'] for e in networks]
    network_idx = {n: i for i, n in enumerate(network_ids)}
    solutions_meta['networkIdx'] = solutions_meta['networkId'].map(network_idx)
    n_networks = len(network_ids)

    # create mapping between source, target onto action [0,1]
    netst = map_source_target_action_map(networks)

    if debug:
        T, R = tm.calc_T_R(networks)

    reward_map = {
        -100: 0,
        -20: 1,
        20: 2,
        140: 3
    }

    if create_old:
        SO = th.zeros((n_participants, n_networks, n_steps, n_nodes, 2))
        RO = th.zeros((n_participants, n_networks))
        VO = th.zeros((n_participants, n_networks))

    values = []
    for idx, sol in solutions_meta.iterrows():
        solution = solutions[sol['solutionIdx']]
        n_id = sol['networkId']
        e_id = sol['environmentId']
        p_id = sol['playerId']

        n_idx = network_idx[n_id]
        p_idx = participants_idx[p_id]

        actions = solution['actions']
        assert len(actions) == n_steps
        prev_solution_idx = sol['previousSolutionIdx']
        if not pd.isnull(prev_solution_idx):
            prev_solution = solutions[prev_solution_idx]
            # TODO: handle machine player
            p_p_idx = -1
            # p_p_id = prev_solution['playerId'][p_id]
            # p_p_idx = participants_idx[p_p_id]
        else:
            p_p_idx = -1
        for s_idx, step in enumerate(actions):
            source = step['sourceId']
            target = step['targetId']
            reward = reward_map[step['reward']]
            action = netst[(e_id, source, target)]
            if debug:
                assert step['reward'] == int(R[n_idx, source, action])

            if create_old:
                SO[p_idx, n_idx, s_idx, source, action] = 1

            if not pd.isnull(prev_solution_idx):
                p_step = prev_solution['actions'][s_idx]
                p_source = p_step['sourceId']
                p_target = p_step['targetId']
                p_reward = reward_map[p_step['reward']]
                p_action = netst[(e_id, p_source, p_target)]
                if debug:
                    assert p_step['reward'] == R[n_idx, p_source, p_action]
            else:
                p_source, p_target, p_reward, p_action = -1, -1, -1, -1

            values.append([
                n_idx, action, s_idx, source, p_idx, reward,
                p_action, s_idx, p_source, p_p_idx, p_reward])

        if create_old:
            RO[p_idx, n_idx] = solution['totalReward']
            VO[p_idx, n_idx] = 1

    names = ['network', 'action', 'move', 'source', 'participant', 'reward',
             'pAction', 'pMove', 'pSource', 'pParticipant', 'pReward']

    values = th.tensor(values, dtype=th.int64, device=device)

    rd = {n: values[:, i].clone() for i, n in enumerate(names)}

    if create_old:
        rd = {'VO': VO, 'SO': SO, 'RO': RO, **rd}

    return rd
