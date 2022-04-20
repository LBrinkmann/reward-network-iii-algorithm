import pandas as pd
from scripts.utils.array_to_df import using_multiindex
from scripts.pruning_models import torch_model as tm
import torch as th


# used by: apply_models.py, derivative_classification.py
def calc_regret(
        networks, settings_df, n_nodes, n_steps, device_name,
        samples=None, all_starting_nodes=True):
    device = th.device(device_name)
    T, R = tm.calc_T_R_torch(networks, n_nodes, device)
    G = th.tensor([[0, 0]], dtype=th.float32, device=device)
    if not all_starting_nodes:
        ST = [n['startingNodeId'] for n in networks]
        ST = th.tensor(ST, dtype=th.long, device=device)
    else:
        ST = None
    Q = tm.calculate_q_matrix_avpruning(R, T, n_steps, G, device=device)
    lookahead = tm.calculate_traces(Q, T, R, ST=ST, device=device)

    settings_df = settings_df.reset_index()
    G = th.tensor(
        settings_df[['gamma_g', 'gamma_s']].values,
        dtype=th.float32, device=device)
    beta = th.tensor(
        settings_df['beta'].values,
        dtype=th.float32, device=device)

    Q = tm.calculate_q_matrix_avpruning(R, T, n_steps, G, device=device)

    if samples is None:
        samples = 1
        stochastic = False
    else:
        stochastic = True

    all_regret_df = []
    for s in range(samples):
        rewards = tm.calculate_traces(
            Q, T, R, ST=ST, beta=beta, device=device, stochastic=stochastic)
        regred = lookahead - rewards

        columns = ['Setting_IDX', 'Network_IDX'] + \
            (['Starting_Node'] if all_starting_nodes else [])
        regret_df = using_multiindex(regred.cpu().numpy(), columns)
        if all_starting_nodes:
            regret_df['Environment_IDX'] =  \
                regret_df['Network_IDX'] * 6 + regret_df['Starting_Node']
        else:
            regret_df['Environment_IDX'] = regret_df['Network_IDX']
        if stochastic:
            regret_df['sample'] = s
        all_regret_df.append(regret_df)

    regret_df = pd.concat(all_regret_df)
    regret_df = regret_df.rename(columns={'value': 'Regret'})
    regret_df = regret_df.merge(settings_df, left_on='Setting_IDX', right_index=True)
    regret_df['Network_ID'] = regret_df['Network_IDX'].apply(
        lambda i: networks[i].get('network_id', networks[i].get('networkId')))
    return regret_df
