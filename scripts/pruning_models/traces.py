import numpy as np
import mc.pruning_models.torch_model as tm
import torch


def calculate_traces_deterministic(Q, T, R, ST):
    #used in solution_manipulation
    n_envs = Q.shape[0]
    n_steps = Q.shape[1]  # Q: env,step,node,action
    n_node = Q.shape[2]
    NT = np.zeros((n_envs, n_steps + 1), dtype='int64')  # node trace: step, starting node
    AT = np.zeros((n_envs, n_steps), dtype='int64')  # action trace: step, starting node
    RT = np.zeros((n_envs, n_steps), dtype='int64')  # reward trace: step, starting node
    RT_tot = np.zeros((n_envs), dtype='int64')
    NT[:, 0] = ST
    PT = np.argmax(Q, axis=-1)
    for i in range(n_envs):
        PT_env = PT[i, :, :]
        AT_env = AT[i, :]
        NT_env = NT[i, :]
        T_env = T[i, :, :, :]
        RT_env = RT[i, :]
        R_env = R[i, :, :]
        for l in range(0, n_steps):
            AT_env[l] = PT_env[l, NT_env[l]]
            NT_env[l+1] = np.argmax(T_env[NT_env[l], :, AT_env[l]], axis=0)
            RT_env[l] = R_env[NT_env[l], AT_env[l]]
        PT[i, :, :] = PT_env
        AT[i, :] = AT_env
        NT[i, :] = NT_env
        T[i, :, :, :] = T_env
        RT[i, :] = RT_env
        R[i, :, :] = R_env
        RT_tot[i] = np.sum(RT_env, axis=0)
    return RT_tot, AT, NT, RT


def calculate_all_traces(networks_selected, G, n_nodes, n_steps, sourceId_list, device_name='cpu'):
    #used in solution_manipulation
    T, R = tm.calc_T_R(networks_selected, n_nodes)
    device = torch.device(device_name)
    T = torch.tensor(T, dtype=torch.float32, device=device)
    R = torch.tensor(R, dtype=torch.float32, device=device)
    G = torch.tensor(G, dtype=torch.float32, device=device)
    Q = tm.calculate_q_matrix_avpruning(R, T, n_steps, G, device=device)
    T = np.array(T)
    R = np.array(R)
    Q = np.array(Q).reshape(R.shape[0], n_steps, R.shape[1], R.shape[2])
    ST = np.array(sourceId_list)
    RT_tot, AT, NT, RT = calculate_traces_deterministic(Q, T, R, ST)
    return RT_tot, AT, NT, RT
