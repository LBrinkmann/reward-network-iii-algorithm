# This file specifices the Deep Q Learning AI agent model to solve a Reward Network DAG
# See also: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
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
import re
import json
import time,datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional
import argparse

# filesystem and log file specific imports
import logging
import glob
import os
import sys

# import Pytorch + hyperparameter tuning modules
import torch as th
import torch.nn as nn               # layers
import torch.optim as optim         # optimizers
import torch.nn.functional as F     #
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

# import the custom reward network environment class
from environment_vect import Reward_Network

#######################################
## Initialize NN
#######################################

class DQN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(input_size[-1], hidden_size[-1])
        self.linear2 = nn.Linear(hidden_size[-1], output_size[-1])

    def forward(self, obs):
        # evaluate q values
        #i = obs['obs'][:,:,:].float()
        i = obs.float()
        x = F.relu(self.linear1(i))
        #q = F.relu(self.linear2(x))
        q = self.linear2(x)
        return q


#######################################
## Initialize agent
#######################################

class Agent:
    def __init__(self, obs_dim: int, config_params:dict, action_dim: tuple, save_dir: str, device):
        """
        Initializes an object of class Agent 

        Args:
            obs_dim (int): number of elements present in the observation (2, action space observation + valid action mask)
            config_params (dict): a dict of all parameters and constants of the reward network problem (e.g. number of nodes, number of networks..)
            action_dim (tuple): shape of action space of one environment
            save_dir (str): path to folder where to save model checkpoints into
            device: torhc device (cpu or cuda)
        """        

        # assert tests
        assert len(config_params)==13, f'expected 13 key-value pairs in config_params, got {len(config_params)} instead'
        assert os.path.exists(save_dir), f'{save_dir} is not a valid path (does not exist)'

        # specify environment parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.network_params =  {'N_NETWORKS':config['n_networks'],
                                'N_NODES':config['n_nodes'],
                                'N_ROUNDS':config['n_rounds']}

        # specify DNNs used by the agent in training and learning Q(s,a) from experience
        # to predict the most optimal action - we implement this in the Learn section
        # two DNNs - policy net with Q_{online} and target net with Q_{target}- that
        # independently approximate the optimal action-value function.
        input_size = (self.network_params['N_NETWORKS'],self.network_params['N_NODES'],20)
        hidden_size = (self.network_params['N_NETWORKS'],self.network_params['N_NODES'],config['nn_hidden_size'])
        # one q value for each action
        output_size = (self.network_params['N_NETWORKS'],self.network_params['N_NODES'],1)
        self.policy_net = DQN(input_size, output_size, hidden_size)
        self.target_net = DQN(input_size, output_size, hidden_size)

        # specify \epsilon greedy policy exploration parameters (relevant in exploration)
        self.exploration_rate = 1
        self.exploration_rate_decay = config['exploration_rate_decay']#0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # specify \gamma parameter (how far-sighted our agent is)
        self.gamma = 0.9

        # specify training loop parameters
        self.burnin = 10  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = config['nn_update_frequency']#1e4  # no. of experiences between Q_target & Q_online sync
        self.save_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # specify which loss function and which optimizer to use (and their respective params)
        self.lr = config['lr']#0.00025
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = th.nn.SmoothL1Loss(reduction='none')

        # specify output directory
        self.save_dir = save_dir

        # torch device
        self.device = device

    def apply_mask(self,q_values,mask):
        """
        This method assigns a very low q value to the invalid actions in a network,
        as indicated by a mask provided with the env observation

        Args:
            q_values (th.tensor): estimated q values for all actions in the envs
            mask (th.tensor): boolean mask indicating with True the valid actions in all networks

        Returns:
            q_values (th.tensor): _description_
        """        

        q_values[~mask] = th.finfo(q_values.dtype).min
        return q_values

    def act(self, obs):
        """
        Given a observation, choose an epsilon-greedy action (explore) or use DNN to
        select the action which, given $S=s$, is associated to highest $Q(s,a)$

        Args:
            obs (dict with values of th.tensor): observation from the env(s) comprising of one hot encoded reward+step counter+ big loss counter 
                                                 and a valid action mask

        Returns:
            action (th.tensor): node index representing next nodes to move to for all envs
            action_values (th.tensor): estimated q values for action
        """        

        # assert tests
        assert len(obs) == self.obs_dim and isinstance(obs, dict), \
            f"Wrong length of state representation: expected dict of size {self.obs_dim}, got: {len(obs)}"

        # for the moment keep a random action selection strategy to mock agent choosing action
        #action_idx = th.squeeze(th.multinomial(obs['mask'].type(th.float),1))

        # EXPLORE (select random action from the action space)
        #if np.random.rand() < self.exploration_rate:
        #random_actions = th.squeeze(th.multinomial(obs['mask'].type(th.float),1))
        random_actions = th.multinomial(obs['mask'].type(th.float),1)
        #print(f'random actions {th.squeeze(random_actions,dim=-1)}')

        # EXPLOIT (select greedy action)
        # return Q values for each action in the action space A | S=s
        action_q_values = self.policy_net(obs['obs'])
        # apply masking to obtain Q values for each VALID action (invalid actions set to very low Q value)
        action_q_values = self.apply_mask(action_q_values,obs['mask'])
        # select action with highest Q value
        greedy_actions = th.argmax(action_q_values, axis=1)#.item()
        #print(f'greedy actions {th.squeeze(greedy_actions,dim=-1)}')

        # select between random or greedy action in each env
        select_random = (th.rand(self.network_params['N_NETWORKS'], device=self.device) < self.exploration_rate).long()
        #print(f'action selection -> {select_random}')
        action = select_random * random_actions + (1 - select_random) * greedy_actions
        #print(f'action -> {action}')
        print('\n')

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action[:,0],action_q_values


    def td_estimate(self, state, state_mask):
        """
        This function returns the TD estimate for a (state,action) pair - the predicted optimal Q∗ for a given state s
        
        Args:
            state (dict of th.tensor): observation
            state_mask (th.tensor): boolean mask to the observation matrix

        Returns:
            td_est: Q∗_online(s,a)
        """
        
        # we use the online model here we get Q_online(s,a)
        td_est = self.policy_net(state)#[np.arange(0, self.batch_size), action]
        # apply masking (invalid actions set to very low Q value)
        td_est = self.apply_mask(td_est,state_mask)
        return td_est

    # note that we don't want to update target net parameters by backprop (hence the th.no_grad),
    # instead the online net parameters will take the place of the target net parameters periodically
    @th.no_grad()
    def td_target(self, reward, state, state_mask):
        """
        This method returns TD target - aggregation of current reward and the estimated Q∗ in the next state s'

        Args:
            reward (_type_): reward obtained at current observation
            state (_type_): observation corresponding to applying next action a'
            state_mask (_type_): boolean mask to the observation matrix

        Returns:
            td_tgt: estimated q values from target net
        """        

        # Because we don’t know what next action a' will be,
        # we use the action a' that maximizes Q_{online} in the next state s'
        #next_state_Q_values = self.policy_net(state)
        # apply masking (invalid actions set to very low Q value)
        #next_state_Q_values = self.apply_mask(next_state_Q_values,state['mask'])
        #best_action = th.argmax(next_state_Q_values, axis=1)
        
        # or size
        next_max_Q2 = th.zeros(state.shape[:3],device=self.device)
        #next_max_Q = th.zeros_like(next_state_Q_values, device=self.device)
        # we skip the first observation and set the future value for the terminal
        # state to 0
        target_Q = self.target_net(state)
        target_Q = self.apply_mask(target_Q,state_mask)
        # batch,steps,netowkrs,nodes
        next_Q = target_Q[:,1:]
        # batch,steps,networks
        next_max_Q = next_Q.max(-1)[0].detach()
        next_max_Q2[:,:-1,:] = next_max_Q
        #TODO: check dimensions, remmebr batch dimension - check gradients
        #next_Q[:, :-1] = self.target_net(state)[:, 1:].max(-1)[0].detach()

        return th.squeeze(reward) + (self.gamma * next_max_Q2)


    def update_Q_online(self, td_estimate, td_target):
        """
        This function updates the parameters of the "online" DQN by means of backpropagation.
        The loss value is given by the (td_estimate - td_target)

        \theta_{online} <- \theta_{online} + \alpha((TD_estimate - TD_target))

        Args:
            td_estimate (_type_): q values as estimated from policy net
            td_target (_type_): q values as estimated from target net

        Returns:
            loss: loss value
        """        

        # calculate loss, defined as SmoothL1Loss on (TD_estimate,TD_target),
        # then do gradient descent step to try to minimize loss
        loss = self.loss_fn(td_estimate, td_target)
        print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()

        # truncate large gradients as in original DQN paper
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

    def learn(self,memory_sample):
        """
        Update online action value (Q) function with a batch of experiences.
        As we sample inputs from memory, we compute loss using TD estimate and TD target, then backpropagate this loss down 
        Q_online to update its parameters θ_online

        Args:
            memory_sample (dict with values as th.tensors): sample from Memory buffer object

        Returns:
            (th.tensor,float): estimated Q values + loss value
        """        

        # if applicable update target net parameters
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # if applicable save model checkpoints
        #if self.curr_step % self.save_every == 0:
        #    self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Break down Memory buffer sample 
        state = memory_sample['obs']
        print(f'memory sample state shape {state.shape}')
        state_mask = memory_sample['mask']
        print(f'memory sample state mask shape {state_mask.shape}')
        action = memory_sample['action']
        print(f'memory sample action shape {action.shape}')
        reward = memory_sample['reward']
        print(f'memory sample reward shape {reward.shape}')

        # Get TD Estimate (mask alreadzy applied in function)
        td_est = self.td_estimate(state, state_mask)
        print(f'Calculated td_est of shape {td_est.shape}')
        # Get TD Target
        td_tgt = self.td_target(reward, state, state_mask)
        print(f'td_tgt {td_tgt}')

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss


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
        """
        Stores multiple tensor in the memory.
        In **state we have:
        - observation mask for valid actions
        - observation matrix with one hot encoded info on reward index, level, step counter, loss counter
        - obtained reward tensor
        - action tensor
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
    def __init__(self, save_dir, n_networks, n_episodes, n_nodes, n_steps=8):
        """
        Initialize logger object

        Args:
            save_dir (str): path to folder where logged metrics will be saved
            n_networks (int): number of networks
            n_episodes (int): number of episodes
            n_nodes (int): number of nodes in one network
            n_steps (int, optional): number of steps needed to complete a network. Defaults to 8.
        """     

        self.save_dir = save_dir

        # params
        self.n_networks = n_networks
        self.n_episodes = n_episodes
        self.n_nodes = n_nodes
        self.n_steps = n_steps

        # Q values and reward stats
        self.q_step_log = th.zeros(n_steps,n_networks,n_nodes)
        self.reward_step_log = th.zeros(n_steps,n_networks)

        # Episode metrics
        self.episode_metrics = {'reward_steps':[],
                                'reward_episode':[],
                                'reward_episode_all_envs':[],
                                'loss':[],
                                'q_mean_steps':[], 
                                'q_min_steps':[],
                                'q_max_steps':[],
                                'q_mean':[], 
                                'q_min':[],
                                'q_max':[]}
        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.record_metrics = {'rewards':[],
                                'loss':[],
                                'q_mean':[], 
                                'q_min':[],
                                'q_max':[]}

        # number of episodes to consider to calculate mean episode {current_metric}
        self.take_n_episodes = 5

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def init_episode(self):
        """
        Reset current metrics values
        """
        self.curr_ep_reward = 0.0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.q_step_log = th.zeros(self.n_steps,self.n_networks,self.n_nodes)
        self.reward_step_log = th.zeros(self.n_steps,self.n_networks)

    def log_step(self, reward, q_step, step_number):
        """
        To be called at every transition within an episode, saves reward of the step
        and the aggregate functions of q values for each network-step 

        Args:
            reward (th.tensor): reward obtained in current step in the env(s) (for all networks)
            q_step (th.tensor): q values of actions in current step in the env(s) (for all networks)
            step_number (int): current step in the env(s)
        """        

        self.curr_ep_reward += reward
        self.reward_step_log[step_number,:] = reward[:,0]
        self.q_step_log[step_number,:,:] = q_step[:,:,0].detach()


    def log_episode(self):
        """
        Store metrics'values at end of a single episode

        Args:
            q (th.tensor): q values for each env  
            loss (float): loss value
        """        

        # log the total reward obtained in the episode for each of the networks
        #self.episode_metrics['rewards'].append(self.curr_ep_reward)
        self.episode_metrics['reward_steps'].append(self.reward_step_log)
        self.episode_metrics['reward_episode'].append(th.squeeze(th.sum(self.reward_step_log,dim=0)))
        self.episode_metrics['reward_episode_all_envs'].append(th.mean(th.squeeze(th.sum(self.reward_step_log,dim=0))).item())
        # log the loss value in the episode for each of the networks TODO: adapt to store when Learn method is called
        #self.episode_metrics['loss'].append(loss)

        # log the mean, min and max q value in the episode over all envs but FOR EACH STEP SEPARATELY
        # (apply mask to self.q_step_log ? we are mainly interested in the mean min and max of valid actions)
        self.episode_metrics['q_mean_steps'].append(th.mean(self.q_step_log,dim=0))
        #self.episode_metrics['q_min_steps'].append(th.amin(self.q_step_log,dim=(1,2)))
        self.episode_metrics['q_max_steps'].append(th.amax(self.q_step_log,dim=(1,2)))
        # log the average of mean, min and max q value in the episode ACROSS ALL STEPS
        self.episode_metrics['q_mean'].append(th.mean(self.episode_metrics['q_mean_steps'][-1]))
        #self.episode_metrics['q_min'].append(th.mean(self.episode_metrics['q_min_steps'][-1]))
        self.episode_metrics['q_max'].append(th.mean(self.episode_metrics['q_max_steps'][-1]))


        # reset values to zero
        self.init_episode()

    def log_episode_learn(self,q,loss):
        """
        Store metrics'values at the call of Learn method TODO: finish

        Args:
            q (th.tensor): q values for each env  
            loss (float): loss value
        """        
        # log the q values from learn method
        self.episode_metrics['q_learn'].append(q)
        # log the loss value from learn method
        self.episode_metrics['loss'].append(loss)


    def record(self, episode, epsilon): #, step):
        """
        This method prints out during training the average trend of different metrics recorded for each episode.
        The avergae trend is calculated counting the last self.take_n_episodes completed
        TODO: finish for loss and minimum q value
        Args:
            episode (int): the current episode number
            epsilon (float): the current exploration rate for the greedy policy
        """        
        mean_ep_reward = np.round(th.mean(self.episode_metrics['reward_episode'][-self.take_n_episodes:][0]), 3)
        #mean_ep_loss = np.round(th.mean(self.episode_metrics['loss'][-self.take_n_episodes:][0]), 3)
        mean_ep_q_mean = np.round(th.mean(self.episode_metrics['q_mean'][-self.take_n_episodes:][0]), 3)
        #mean_ep_q_min = np.round(np.mean(self.episode_metrics['q_min'][-self.take_n_episodes:][0]), 3)
        mean_ep_q_max = np.round(th.mean(self.episode_metrics['q_max'][-self.take_n_episodes:][0]), 3)
        self.record_metrics['reward_episode'].append(mean_ep_reward)
        #self.record_metrics['loss'].append(mean_ep_loss)
        self.record_metrics['q_mean'].append(mean_ep_q_mean)
        #self.record_metrics['q_min'].append(mean_ep_q_min)
        self.record_metrics['q_max'].append(mean_ep_q_max)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(f"We are at Episode {episode} - "
              f"Epsilon {epsilon} - "
              f"Mean Reward over last {self.take_n_episodes} episodes: {mean_ep_reward} - "
              #f"Mean Loss over last {self.take_n_episodes} episodes: {mean_ep_loss} - "
              f"Mean Q Value over last {self.take_n_episodes} episodes: {mean_ep_q_mean} - "
              #f"Min Q Value over last {self.take_n_episodes} episodes: {mean_ep_q_min} - "
              f"Max Q Value over last {self.take_n_episodes} episodes: {mean_ep_q_max} - "
              f"Time Delta {time_since_last_record} - "
              f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

    def save_metrics(self):
        """
        Saves moving average metrics as csv file
        """
        metrics_df = pd.DataFrame.from_dict(self.record_metrics,
                                            orient='index',
                                            columns=list(self.record_metrics.keys()))
        metrics_df.to_csv(os.path.join(self.save_dir,'moving_average_metrics.csv'),sep='\t')

    def plot_metric(self):

        self.plot_attr = {#"reward_episode":save_dir / "reward_plot.pdf",
                          #"reward_step":save_dir / "reward_step_plot.pdf"
                          "reward_episode_all_envs": os.path.join(self.save_dir,"reward_all_envs_plot.pdf"),
                          #"ep_loss":save_dir / "loss_plot.pdf",
                          "q_mean":os.path.join(self.save_dir,"reward_all_envs_mean_q.pdf"),
                          #"ep_min_q":save_dir / "min_q_plot.pdf",
                          "q_max":os.path.join(self.save_dir,"reward_all_envs_max_q.pdf")}


        for metric_name, metric_plot_path in self.plot_attr.items():
            plt.plot(self.episode_metrics[metric_name])
            plt.title(f'{metric_name}',fontsize=20)
            plt.xlabel('Episode', fontsize=17)
            #plt.ylim(-200, 400)
            plt.savefig(metric_plot_path,format='pdf',dpi=300)
            plt.clf()



#######################################
## TRAINING FUNCTION
#######################################
def train_agent(config):
    """
    Train AI agent to solve reward networks

    Args:
        config (dict): dict containing parameter values, data paths and
                       flag to run or not run hyperparameter tuning
    """      

    # ---------Loading of the networks---------------------
        # Load networks to test
    with open(config['data_path']) as json_file:
        train = json.load(json_file)
    test = train[:10]
    # add number of netowkrs to config
    config['n_networks'] = len(test)
   

    # ---------Specify device (cpu or cuda)----------------
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
    AI_agent = Agent(obs_dim=2,
                     config_params = config,
                     action_dim=env.action_space_idx.shape, 
                     save_dir=config['log_path'],
                     device=DEVICE)

    # initialize Memory buffer
    Mem = Memory(device=DEVICE, size=config['memory_size'], n_rounds=config['n_rounds'])

    # initialize Logger
    logger = MetricLogger(config['log_path'],config['n_networks'],config['n_episodes'],config['n_nodes'])


    for e in range(config['n_episodes']):
        print(f'----EPISODE {e+1}---- \n')

        # reset env(s)
        env.reset()
        # obtain first observation of the env(s)
        obs = env.observe()

        for round in range(config['n_rounds']):

            # Solve the reward networks!
            #while True:
            print('\n')
            print(f'ROUND/STEP {round} \n')

            # choose action to perform in environment(s)
            action,step_q_values = AI_agent.act(obs)
            #print(f'q values for step {round} -> \n {step_q_values[:,:,0].detach()}')
            # agent performs action
            # if we are in the last step we only need reward, else output also the next state
            if round!=7:
                next_obs, reward = env.step(action,round)
            else:
                reward = env.step(action,round)
            #print(f'reward -> {reward}')
            # remember transitions in memory
            Mem.store(**obs,round=round,reward=reward, action=action)
            if round!=7:
                obs = next_obs
            # Logging (step)
            logger.log_step(reward,step_q_values,round)

            if env.is_done:
                break
        
        #--END OF EPISODE--
        Mem.finish_episode()

        logger.log_episode()
        
        print('\n')
        print('\n')
        print(f'EPISODE {e+1} MEMORY SAMPLE!')
        sample = Mem.sample(config['batch_size'],device=DEVICE)
        if sample is not None:
            for k,v in sample.items():
                print(k, v.shape)

            # Learning step
            q, loss = AI_agent.learn(sample)
            print(loss)

            if config['tune']:
                # Send the current training result back to Tune
                tune.report(loss=loss)
            # Logging (mimick values as if obtained form leanr method to test the logger)
            #q,loss = random.randint(0,5),random.randint(0,1)
            #logger.log_episode_learn(q,loss)
        
        else:
            print(f"Skip episode {e+1}")
        print('\n')
        #if e % 5 == 0:
        #    logger.record(episode=e, epsilon=AI_agent.exploration_rate)

    # final logging
    #print(logger.episode_metrics['reward_episode_all_envs'])
    logger.plot_metric()
    #logger.save_metrics()

#######################################
## MAIN
#######################################

if __name__ == "__main__":

    # get start time of the script:
    start = time.time()

    # --------Specify arguments--------------------------
    parser = argparse.ArgumentParser(description="DQN Argument Parser (Project: Reward Networks III)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--tune", action="store_true", help="run hyperparameter tuning")
    parser.add_argument("data_src", type=str, help="Data source location (path to JSON networks file)")
    parser.add_argument("results_dest", type=str, help="Results location (checkpoints + figures + logs)")
    args = parser.parse_args()

    # --------Specify paths--------------------------
    current_dir = os.getcwd()
    root_dir = os.sep.join(current_dir.split(os.sep)[:2])
    # Specify directories depending on system (local vs cluster)
    if root_dir == '/mnt':
        user_name = os.sep.join(current_dir.split(os.sep)[4:5])
        home_dir = f"/mnt/beegfs/home/{user_name}"
        project_dir = os.path.join(home_dir,'CHM','reward_networks_III')
        data_dir = os.path.join(project_dir, 'data')
        out_dir = os.path.join(data_dir, 'results')

    elif root_dir == '/Users':
        # Specify directories (local)
        project_folder = os.getcwd()
        data_dir = os.path.join(project_folder,'data')
        save_dir = os.path.join("../..",'data','solutions')
        log_dir = os.path.join("../..",'data','log')


    # ---------Parameters----------------------------------
    config = {'data_path':os.path.join(args.data_src,'train_viz.json'),
              'log_path':os.path.join(os.path.split(project_folder)[0],'data/log'),
              'n_episodes':500,
              'n_rounds':8,
              'n_nodes':10,
              'batch_size':10,
              'memory_size':50,
              'lr':0.0001,
              'nn_hidden_size':10,
              'exploration_rate_decay':0.8,
              'nn_update_frequency':500,
              'tune':False
              }

    # if we want to run hyperparameter tuning then define search space and 
    # Ray Tune object TODO: check if Ray can be used on SLURM cluster
    if (args.tune==True):
        
        # set tune option to true
        config['tune']=True

        # Define a search space and initialize the search algorithm.
        search_space = {"lr": tune.grid_search([1e-5, 1e-4]),
                        "batch_size":tune.qrandint(5,20),
                        "memory_size":tune.qrandint(50,500),
                        "nn_hidden_size":tune.qrandint(5,20),
                        'exploration_rate_decay':tune.grid_search([0.6, 0.95]),
                        'nn_update_frequency':tune.qrandint(50,500)}
        algo = OptunaSearch()

        # Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
        tuner = tune.Tuner(train_agent,
                           tune_config=tune.TuneConfig(metric="loss",
                                                       mode="min",
                                                       search_alg=algo),
                           run_config=air.RunConfig(stop={"training_iteration": 5}),
                           param_space=search_space)

        results = tuner.fit()
        print("Best config is:", results.get_best_result().config)
    
    else:

        # train the agent
        train_agent(config)

    # N_EPISODES = 50
    # N_ROUNDS = 8 # or number of steps
    # N_NETWORKS = len(test)
    # N_NODES = 10
    # #OBS_SHAPE = 7 # OR 8?
    # BATCH_SIZE = 4
    # network_params = {'N_NETWORKS':N_NETWORKS,
    #                  'N_NODES':N_NODES,
    #                  'N_ROUNDS':N_ROUNDS}

    # # specify if cuda is available
    # use_cuda = th.cuda.is_available()
    # print(f"Using CUDA: {use_cuda} \n")
    # if not use_cuda:
    #     DEVICE = th.device('cpu')
    # else:
    #     DEVICE=th.device('cuda')

    # # ---------Start analysis------------------------------
    # # initialize environment(s)
    # env = Reward_Network(test)

    # # initialize Agent
    # AI_agent = Agent(obs_dim=2,
    #                  network_params = network_params,
    #                  action_dim=env.action_space_idx.shape, 
    #                  save_dir=log_dir,
    #                  device=DEVICE)

    # # initialize Memory buffer
    # Mem = Memory(device=DEVICE, size=5, n_rounds=N_ROUNDS)

    # # initialize Logger
    # logger = MetricLogger(log_dir,N_NETWORKS,N_EPISODES,N_NODES)


    # for e in range(N_EPISODES):
    #     print(f'----EPISODE {e+1}---- \n')

    #     # reset env(s)
    #     env.reset()
    #     # obtain first observation of the env(s)
    #     obs = env.observe()

    #     for round in range(N_ROUNDS):

    #         # Solve the reward networks!
    #         #while True:
    #         print('\n')
    #         print(f'ROUND/STEP {round} \n')

    #         # choose action to perform in environment(s)
    #         action,step_q_values = AI_agent.act(obs)
    #         print(f'q values for step {round} -> \n {step_q_values[:,:,0].detach()}')
    #         # agent performs action
    #         # if we are in the last step we only need reward, else output also the next state
    #         if round!=7:
    #             next_obs, reward = env.step(action,round)
    #         else:
    #             reward = env.step(action,round)
    #         print(f'reward -> {reward}')
    #         # remember transitions in memory
    #         Mem.store(**obs,round=round,reward=reward, action=action)
    #         if round!=7:
    #             obs = next_obs

    #         # Logging (step)
    #         logger.log_step(reward,step_q_values,round)

    #         if env.is_done:
    #             break
        
    #     #--END OF EPISODE--
    #     Mem.finish_episode()

    #     logger.log_episode()
        
    #     print('\n')
    #     print('\n')
    #     print('MEMORY SAMPLE?')
    #     sample = Mem.sample(BATCH_SIZE,device=DEVICE)

    #     if sample is not None:
    #         for k,v in sample.items():
    #             print(k, v.shape)

    #         # Learning step
    #         q, loss = AI_agent.learn(sample)
    #         # Logging (mimick values as if obtainedform leanr method to test the logger)
    #         #q,loss = random.randint(0,5),random.randint(0,1)
    #         #logger.log_episode_learn(q,loss)
        
    #     else:
    #         print(f"Skip episode {e}")

    #     #if e % 5 == 0:
    #     #    logger.record(episode=e, epsilon=AI_agent.exploration_rate)

    # # final logging
    # #logger.save_metrics()
    