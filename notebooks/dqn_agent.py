# This file specifices the Deep Q Learning AI agent model to solve a Reward Network DAG
# See also: https://pyth.org/tutorials/intermediate/reinforcement_q_learning.html
# and: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
#
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################

# import modules
import numpy as np
import pandas as pd
from collections import namedtuple, deque, Counter
from itertools import count
from tqdm import tqdm
import math
import random
import glob
import logging
import re
import sys
import os
import json
import time,datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
# TODO: specify custom types for state and action of env

# import Pytorch modules
import torch as th
import torch.nn as nn               # layers
import torch.optim as optim         # optimizers
import torch.nn.functional as F     #

# import the custom reward network environment class
from environment_vect import Reward_Network

#######################################
## Initialize NN
#######################################

class DQN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, obs):
        # evaluate q values
        x = F.relu(self.linear1(obs['obs']))
        q = F.relu(self.linear2(x))

        # --- QUESTION ---
        # In calculating the Q values we should make the calculations for all actions,
        # and then the invalid actions will be set to a very low Q value e.g. -1000
        # Is the forward pass the correct point at which to set the invalid actions to very low Q values?
        # E.g. code snippet below
        #
        # The selection of the valid actions' Q values should then be performed in the act method using obs['mask'] on the network output;
        # I would also guess that this is also necessary because we need to maintain the same dimension
        # of output_size for the network (different networks at different step numbers may have different
        # number of valid actions that can be performed),
        # --- Answer ---
        # 1. I think this is person taste. I personally would NOT do the masking
        # with the DQN model. Instead, I would write a method that is masking
        # the q values and then do the masking after calculating the q values
        # within the Agent. Example:
        # q_values = self.target_net(...)
        # q_values_masked = apply_mask(q_values, mask)
        #
        # 2. For random actions (i.e. for epsilon greedy) you also need to apply
        #    the mask. So handling all of this in the act method (see 1.) makes
        #    sense. Generally speaking (this related to the act method): I would calculate for each network at
        #    each step, both, a greedy and a random action. Then I would decide
        #    indepently randomly for each network and at each step, which of the
        #    two actions (random or greedy) to use. The principal:
        #    better a bit to large matrix calculation, but simple logic
        #
        #    greedy_action = ...
        #    random_action = ...
        #    select_random = (th.rand(size=actions_shape, device=self.device) < eps).long()
        #    action = select_random * random_actions + (1 - select_random) * greedy_actions
        #
        # 3. It seems to be good practice to set the masked value to the minimum
        #    possible one. torch.finfo(logits.dtype).min
        #    See here: https://boring-guy.sh/posts/masking-rl/
        q[~obs['mask']] = -1000
        return q


#######################################
## Initialize agent
#######################################

class Agent:
    def __init__(self, obs_dim: int, action_dim: tuple, save_dir: str):

        # assert tests
        # TODO

        # specify environment parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # specify DNNs used by the agent in training and learning Q(s,a) from experience
        # to predict the most optimal action - we implement this in the Learn section
        # two DNNs - policy net with Q_{online} and target net with Q_{target}- that
        # independently approximate the optimal action-value function.
        self.policy_net = DQN(1,1,1)
        self.target_net = DQN(1,1,1)

        # specify \epsilon greedy policy exploration parameters (relevant in exploration)
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # specify \gamma parameter (how far-sighted our agent is)
        self.gamma = 0.9

        # specify training loop parameters
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.save_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # specify which loss function and which optimizer to use (and their respective params)
        self.lr = 0.00025
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = th.nn.SmoothL1Loss()

        # specify output directory
        self.save_dir = save_dir


    def act(self, obs):
        """
        Given a observation, choose an epsilon-greedy action (explore) or use DNN to
        select the action which, given $S=s$, is associated to highest $Q(s,a)$
        Inputs:
        - obs (dict with values of th.tensor): Observation of the current state - current nodes -
                                               and boolean mask of next possible nodes, shape is (state_dim)
        Outputs:
        - action_idx (th.tensor): An integer representing which action the AI agent will perform
        """
        # assert tests
        assert len(obs) == self.obs_dim and isinstance(obs, dict), \
            f"Wrong length of state representation: expected dict of size {self.obs_dim}, got: {len(obs)}"

        # for the moment keep a random action selection strategy to mock agent choosing action
        action_idx = th.squeeze(th.multinomial(obs['mask'].type(th.float),1))

        # # EXPLORE (select random action from the action space)
        # if np.random.rand() < self.exploration_rate:
        #     #action_idx = np.random.randint(self.action_dim)
        #     action_idx = th.squeeze(th.multinomial(obs['mask'].type(th.float),1))

        # # or EXPLOIT
        # else:
        #     if self.use_cuda:
        #         obs = obs.cuda()

        #     # return Q values for each VALID action in the action space A | S=s
        #     ---QUESTION---
        #     see question in line 57, would this be the step at which we select the valid actions' Q values?
        #     action_values = self.policy_net(obs)[obs['mask']]

        #     # select action with highest Q value
        #     action_idx = th.argmax(action_values, axis=1).item()

        # # decrease exploration_rate
        # self.exploration_rate *= self.exploration_rate_decay
        # self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action_idx


    def td_estimate(self, state, action):
        """
        This function returns the temporal difference estimate for a (state,action) pair.
        In other words, it returns the predicted optimal $Q^*$ for a given state s - action a pair
        """
        # we use the online model here we get Q_online(s,a)
        current_Q = self.policy_net(state)#[np.arange(0, self.batch_size), action]
        return current_Q

    # note that we don't want to update target net parameters by backprop (hence the th.no_grad),
    # instead the online net parameters will take the place of the target net parameters periodically
    @th.no_grad()
    def td_target(self, reward, next_state):
        """
        This function returns the expected Q values, that is
        Q <- E_{policy} ()
        """

        # Because we don’t know what next action a' will be,
        # we use the action a' that maximizes Q_{online} in the next state s'
        next_state_Q = self.policy_net(next_state)
        best_action = th.argmax(next_state_Q, axis=1)

        next_Q = th.zeros_like(reward, device=self.device)
        # we skip the first observation and set the future value for the terminal
        # state to 0
        next_Q[:, :-1] = self.target_net(next_state)[:, 1:].max(-1)[0].detach()

        return reward + (self.gamma * next_Q)


    def update_Q_online(self, td_estimate, td_target):
        """
        This function updates the parameters of the "online" DQN by means of backpropagation.
        The loss value is given by the (td_estimate - td_target)

        \theta_{online} <- \theta_{online} + \alpha((TD_estimate - TD_target))
        """

        # calculate loss, defined as TD_estimate - TD_target,
        # then do gradient descent step to try to minimize loss
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()

        # truncate large gradients as in original DQN paper TODO: finish
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def sync_Q_target(self):
        """
        This function periodically copies \theta_online parameters
        to be the \theta_target parameters
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        """
        This function saves model checkpoints
        """
        save_path = os.path.join(self.save_dir,f"Reward_network_iii_dqn_model_{int(self.curr_step // self.save_every)}.chkpt")
        th.save(dict(model=self.policy_net.state_dict(),
                     exploration_rate=self.exploration_rate),
                     save_path)
        print(f"Reward_network_iii_dqn_model checkpoint saved to {save_path} at step {self.curr_step}")

    def learn(self,sample):
        """
        Update online action value (Q) function with a batch of experiences
        Inputs:
        - sample (dict with values as th.tensors): sample from Memory buffer object
        """

        # if applicable update target net parameters
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # if applicable save model checkpoints
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Break down Memory buffer sample TODO: finish
        state = sample['obs']
        next_state = sample['obs']
        action = sample['action']
        reward = sample['reward']

        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        # Get TD Target
        td_tgt = self.td_target(reward, next_state)#, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


#######################################
## Initialize Memory buffer class
# a data structure which temporarily saves the agent’s observations,
# allowing our learning procedure to update on them multiple times
#######################################

class Memory():
    """Storage for observation of a DQN agent.

    Observations are stored large continuous tensor.
    The tensor are automatically initialized upon the first call of store().
    Important: all tensors to be stored need to be passed at the first call of
    the store. Also the shape of tensors to be stored needs to be consistent.


    Typical usage:
        mem = Memory(...)
        for episode in range(n_episodes):
            obs = env.init()
            for round in range(n_rounds):
                action = agent(obs)
                next_obs, reward = env.step()
                mem.store(**obs, reward=reward, action=action)
                obs = next_obs
            mem.finish_episode()

            sample = mem.sample()
            update_agents(sample)
    """

    def __init__(self, device, size, n_rounds):
        """
            Args:
                device: device for the memory
                size: number of episodes to store
                n_rounds; number of rounds to store per episode
        """
        self.memory = None
        self.size = size
        self.n_rounds = n_rounds
        self.device = device
        self.current_row = 0
        self.episodes_stored = 0

    def init_store(self, obs):
        """
        Initialize the memory tensor.
        """
        self.memory = {k: th.zeros((self.size, self.n_rounds, *t.shape),
                                   dtype=t.dtype, device=self.device)
                       for k, t in obs.items() if t is not None}

    def finish_episode(self):
        """Moves the currently active slice in memory to the next episode.
        """
        self.episodes_stored += 1
        self.current_row = (self.current_row + 1) % self.size

    def store(self, round, **state):
        """Stores multiple tensor in the memory.
        """
        # if empty initialize tensors
        if self.memory is None:
            self.init_store(state)

        for k, t in state.items():
            if t is not None:
                self.memory[k][self.current_row, round] = t.to(self.device)

    def sample(self, batch_size, device, **kwargs):
        """Samples form the memory.

        Returns:
            dict | None: Dict being stored. If the batch size is larger than the number
            of episodes stored 'None' is returned.
        """
        if len(self) < batch_size:
            return None
        random_memory_idx = th.randperm(len(self))[:batch_size]
        print(f'random_memory_idx', random_memory_idx)

        sample = {k: v[random_memory_idx].to(device) for k, v in self.memory.items()}
        return sample

    def __len__(self):
        """The current memory usage, i.e. the number of valid episodes in
        the memory.This increases as episodes are added to the memory until the
        maximum size of the memory is reached.
        """
        return min(self.episodes_stored, self.size)


#######################################
## Initialize Logger
# ---- QUESTION -----
# This logger is adapted from https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# In the tutorial there's a distinction between logging each step within an episode,
# logging an episode and logging the average of episodes; I guess that in our case only the
# logging an episode and logging the average of episodes would be needed right?
# Because q values and loss values would be obtained from the Agent's Learn method, which requires a sample from memory buffer
# (and this in turn requires completing an episode)

# ---- ANSWER -----
# You are right, that q values can only be recorded at the end to the episode.
# However, you still have one q value for each network and each step. What i did
# in the past is:
# Calculating q.min(),q.max() and q.mean() over all environments (i.e.
# networks), but for each move seperate. It will be good to see, how the
# q-values evolve over the (in your case) 8 moves. If you record a list of
# vectors, you can later concatenate them into a single multidimensional metrix.

# I have a method turning a multidimension numpy array into a nice dataframe
# with useful columns.
# Example Usage:
# q_max = numpy array shape episodes x steps
# df = numpy_to_df(A, ['episodes, 'steps'], value_name='q_max')

# def numpy_to_df(A, columns, value_columns=None, value_name='value'):
#     """
#     Turns a multi-dimension numpy array into a single dataframe.

#     Args:
#         A: multi-dimensional numpy array.
#         columns: the column name for each dimension.
#         value_columns: the last dimension can be turned into separate columns
#             filled with the individual values;
#             if not None: needs to be list matching in length the size of the
#             last dimension of A.
#             if 'None': all values will be stored in a single column
#         value_name: name for column storing the matrix values (only relevant if
#         value_columns == None)
#     """
#     shape = A.shape
#     if value_columns is not None:
#         assert len(columns) == len(shape) - 1
#         new_shape = (-1, len(value_columns))
#     else:
#         assert len(columns) == len(shape)
#         new_shape = (-1,)
#         value_columns = [value_name]

#     index = pd.MultiIndex.from_product(
#         [range(s) for s, c in zip(shape, columns)], names=columns)
#     df = pd.DataFrame(A.reshape(*new_shape), columns=value_columns, index=index)
#     df = df.reset_index()
#     return df


#######################################

class MetricLogger:
    def __init__(self, save_dir):

        # History metrics
        self.history_metrics = {'ep_rewards':[],
                                'ep_avg_losses':[],
                                'ep_avg_qs':[]}
        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg = {'ep_rewards':[],
                           'ep_avg_losses':[],
                           'ep_avg_qs':[]}
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # number of episodes to consider to calculate mean episode {current_metric}
        self.take_n_episodes = 5

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        """
        To be called at every transition within an episode
        """
        self.curr_ep_reward += reward
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """
        Mark end of episode
        """

        self.history_metrics['ep_rewards'].append(self.curr_ep_reward)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.history_metrics['ep_avg_losses'].append(ep_avg_loss)
        self.history_metrics['ep_avg_qs'].append(ep_avg_q)

        # reset values to zero
        self.init_episode()

    def init_episode(self):
        """
        Initialize current metrics values
        """
        self.curr_ep_reward = 0.0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        """
        Return info on metric trends over average of multiple episodes
        """

        print(self.history_metrics['ep_rewards'])
        mean_ep_reward = np.round(np.mean(self.history_metrics['ep_rewards'][-self.take_n_episodes:]), 3)
        mean_ep_loss = np.round(np.mean(self.history_metrics['ep_avg_losses'][-self.take_n_episodes:]), 3)
        mean_ep_q = np.round(np.mean(self.history_metrics['ep_avg_qs'][-self.take_n_episodes:]), 3)
        self.moving_avg['ep_rewards'].append(mean_ep_reward)
        self.moving_avg['ep_avg_losses'].append(mean_ep_loss)
        self.moving_avg['ep_avg_qs'].append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(f"Episode {episode} - "
              f"Step {step} - "
              f"Epsilon {epsilon} - "
              f"Mean Reward {mean_ep_reward} - "
              f"Mean Loss {mean_ep_loss} - "
              f"Mean Q Value {mean_ep_q} - "
              f"Time Delta {time_since_last_record} - "
              f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    def save_metrics(self):
        """
        Saves moving average metrics as csv file
        """
        metrics_df = pd.DataFrame.from_dict(self.moving_avg,
                                            orient='index',
                                            columns=list(self.moving_avg.keys()))
        metrics_df.to_csv(os.path.join(save_dir,'moving_average_metrics.csv'),sep='\t')

    def plot_metric(self):

        self.plot_attr = {"ep_rewards":save_dir / "reward_plot.pdf",
                          "ep_avg_losses":save_dir / "loss_plot.pdf",
                          "ep_avg_qs":save_dir / "q_plot.pdf"}

        for metric_name, metric_plot_path in self.plot_attr.items():
            plt.plot(self.moving_avg[metric_name])
            plt.savefig(metric_plot_path,format='pdf',dpi=300)
            plt.clf()



#######################################
## MAIN
#######################################

if __name__ == "__main__":


    # --------Specify paths--------------------------
    # Specify directories (cluster)
    #home_dir = f"/home/mpib/bonati"
    #project_dir = os.path.join(home_dir,'CHM','reward_networks')
    #data_dir = os.path.join(project_dir, 'data','rawdata')
    #logs_dir = os.path.join(project_dir, 'logs')

    # Specify directories (local)
    project_folder = os.getcwd()
    data_dir = os.path.join(project_folder,'data')
    save_dir = os.path.join("../..",'data','solutions')
    log_dir = os.path.join("../..",'data','log')

    # Load networks to test
    with open(os.path.join(data_dir,'train.json')) as json_file:
        train = json.load(json_file)
    test = train[10:13]

    # ---------Parameters----------------------------------
    N_EPISODES = 5
    N_ROUNDS = 3
    N_NETWORKS = len(test)
    N_NODES = 10
    OBS_SHAPE = 7 # OR 8?
    BATCH_SIZE = 4
    # specify if cuda is available
    use_cuda = th.cuda.is_available()
    print(f"Using CUDA: {use_cuda} \n")
    if not use_cuda:
        DEVICE = th.device('cpu')
    else:
        DEVICE=th.device('cuda')

    # ---------Start analysis------------------------------
    # initialize environment(s)
    env = Reward_Network(test)

    # initialize Agent
    AI_agent = Agent(obs_dim=2, action_dim=env.action_space_idx.shape, save_dir=log_dir)

    # initialize Memory buffer
    Mem = Memory(device=DEVICE, size=5, n_rounds=N_ROUNDS)

    # initialize Logger
    logger = MetricLogger(log_dir)


    for e in range(N_EPISODES):
        # ----QUESTION-----
        # I understood the concept of episode in RL as 1 episode = a sequence of states, actions and rewards,
        # which ends with terminal state. We also said last time that we want to sample from memory within each episode.
        # Just to make sure I understood correctly, the role of rounds here then is to make sure that we have a varied
        # (varied as in, different actions taken so different reward outcomes) and large number of transitions to sample
        # from the memory buffer within each episode?

        # ---- Answer---
        # Sorry, there has been a confusion with the terminology. We only need
        # episode (aka update steps) and moves (aka rounds or steps). So there are only
        # two loops needed.

        for round in range(N_ROUNDS):

            # reset env(s)
            env.reset()
            # obtain first observation of the env(s)
            obs = env.observe()

            # Solve the reward networks!
            while True:

                # choose action to perform in environment(s)
                action = AI_agent.act(obs)
                # agent performs action
                next_obs, reward = env.step(action)
                # remember transitions in memory
                Mem.store(**obs,round=round,reward=reward, action=action)
                obs = next_obs

                # ---QUESTION---- -> place this code snippet after all rounds or inside rounds?
                # This question is also linked to line 587-588
                # In the tutorials I saw so far (e.g. https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
                # within each episode the agent stores in memory each environment transition,
                # and then learns from given a replay memory sample immediately after.
                # Our training structure is different, with multiple rounds per each episode; would it be correct then to place the
                # Learn method of the agent not in the code snippet below, but
                # rather outside in line 588?
                # ---Answer----
                # I would do the learning step only once for each episode.
                # You might want to log reward at each step (aggregated over all
                # networks).
                # You might want to log q and loss at the end of the episode
                # after the update step.

                #q, loss = AI_agent.learn()

                # Logging (see also question in Logger, the code snippet might be changed to logger.log_rounds)
                #logger.log_step(reward, loss, q)

                if env.is_done:
                    break

        Mem.finish_episode()
        sample = Mem.sample(BATCH_SIZE,device=DEVICE)
        if sample is not None:
            for k,v in sample.items():
                print(k, v.shape)
        else:
            print(f"Skip episode {e}")

        # ---QUESTION---- is this the correct placement? Se also question above
        #q, loss = AI_agent.learn(sample)
        #logger.log_episode()

        #if e % 2 == 0:
        #    logger.record(episode=e, epsilon=AI_agent.exploration_rate,
        #    step=AI_agent.curr_step)

        # ---Answer---
        # yes, would learn once per episode.

    #logger.save_metrics()
