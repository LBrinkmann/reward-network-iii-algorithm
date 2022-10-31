# This file specifices the Reward Network Environment class in OpenAI Gym style.
# A Reward Network object can store and step in multiple networks at a time.
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################
import gym
from gym import spaces
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
import logging
import json
import random
import time
import torch
import torch.nn.functional as F
from collections import Counter
from typing import Optional


def restructure_edges(network):
    """
    This function restructures the edges from list of dicts
    to one dict, to improve construction of edges matrix and 
    env vectorization

    Args:
        network (list): list of dicts, where each dict is a Reward Network with nodes and edges' info

    Returns:
        new_edges (dict): dict with list for source id, target id and reward
    """    

    new_edges= {'source_id':[],'target_id':[],'reward':[]}
    for e in network['edges']:
        new_edges['source_id'].append(e['source_id'])
        new_edges['target_id'].append(e['target_id'])
        new_edges['reward'].append(e['reward'])
    return new_edges 

def extract_level(network):
    """
    This function extracts the level for each node in a network

    Args:
        network (_type_): _description_

    Returns:
        _type_: _description_
    """    
    level= {}
    for e in network['nodes']:
        level[e['node_num']]=e['level']+1
    return level


class Reward_Network(gym.Env):
    
    def __init__(self, network):
        """
        Initializes a reward network object given ntowkr(s) info in JSON format

        Args:
            network (list of dict): list of network information, where each network in list is a dict
                                    with keys nodes-edges-starting_node-total_reward
        """        
        
        #-------------
        # assert tests TODO
        #-------------

        # reward network information from json file (can be just one network or multiple networks)
        self.network = network
       
        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.MAX_STEP = 8
        self.N_REWARD_IDX = 6 #(5 valid rewards + one to indicate no reward possible)
        self.N_NODES = 10
        self.N_LEVELS = 5 #(4 valid levels + one to indicate no level possible)
        self.N_NETWORKS = len(self.network)

        # define node numbers (from 0 to 9)
        self.nodes = torch.stack([torch.arange(10)]*self.N_NETWORKS,dim = 0)
        # define starting nodes
        self.starting_nodes = torch.tensor(list(map(lambda n: n['starting_node'], self.network)), dtype=torch.long)
        # define possible rewards along with corresponding reward index
        self.possible_rewards = {-100:1, -20:2, 0:3, 20:4, 140:5}

        # initialize action space ("reward index adjacency matrix")
        # 0 here means that no edge is present, all other indices from 1 to 5 indicate a reward
        # (the higher the index number, the higher the reward)
        self.action_space_idx = torch.full((self.N_NETWORKS,self.N_NODES, self.N_NODES), 1).long()  
        new_edges = list(map(restructure_edges,network))
        self.network_idx = torch.arange(self.N_NETWORKS, dtype=torch.long)

        # initialize level information for all networks (organized in a n_networks x n_nodes x n_nodes matrix)
        # 4 possible levels (of current node in edge) + 0 value to indicate no edge possible
        levels = list(map(extract_level,network))
        self.level_space = torch.full((self.N_NETWORKS,self.N_NODES, self.N_NODES), 0).long() 
        
        for n in range(self.N_NETWORKS):
            buffer_action_space = torch.full((self.N_NODES, self.N_NODES), 0).long()
            source = torch.tensor(new_edges[n]['source_id']).long()
            target = torch.tensor(new_edges[n]['target_id']).long()
            reward = torch.tensor(new_edges[n]['reward']).long()
            reward.apply_(lambda val: self.possible_rewards.get(val, 0))
            buffer_action_space[source,target]=reward
            self.action_space_idx[n,:,:] = buffer_action_space
            
            buffer_level = torch.full((self.N_NODES, self.N_NODES), 0).long()
            where_edges_present = self.action_space_idx[n,:,:]!=0
            for node in range(self.N_NODES):
                buffer_level[node,where_edges_present[node,:]] = levels[n][node]
            self.level_space[n,:,:] = buffer_level


        # define reward map
        self.reward_map = torch.zeros(max(self.possible_rewards.values()) + 1, dtype=torch.long)
        self.reward_map[list(self.possible_rewards.values())] = torch.tensor(list(self.possible_rewards.keys()), dtype=torch.long)

        # boolean adjacency matrix 
        self.edge_is_present = torch.squeeze(torch.unsqueeze(self.action_space_idx!=0,dim=-1))
        

    def reset(self):
        """
        Resets variables that keep track of env interaction metrics e.g. reward,step counter, loss counter,..
        at the end of each episode
        """        
        # Reset the state of the environment to an initial state
        self.reward_balance = torch.full((self.N_NETWORKS,1),self.INIT_REWARD)
        self.step_counter = torch.full((self.N_NETWORKS,1),self.INIT_STEP)
        self.big_loss_counter = torch.zeros((self.N_NETWORKS,1),dtype=torch.long)
        self.is_done = False 
        self.current_node = self.starting_nodes.clone()

        print('ENVIRONMENT INITIALIZED:')
        print(f'- set of nodes of shape {self.nodes.shape}')
        print(f'- action space of shape {self.action_space_idx.shape}')
        print(f'- reward balance of shape {self.reward_balance.shape}')
        print(f'- big loss counter of shape {self.big_loss_counter.shape}')
        print(f'- step counter of shape {self.step_counter.shape}')
        print(f'- current node of shape {self.current_node.shape}')
        print('\n')

    
    def step(self, action,round):
        """
        Take a step in all environments given an action for each env;
        here action is given in the form of node index for each env
        action_i \in [0,1,2,3,4,5,6,7,8,9]

        Args:
            action (th.tensor): tensor of size n_networks x 1 
            round (int): current round number at which the step is applied. 
                            Relevant to decide if the next observation of envs after action 
                            also needs to be returned or not

        Returns:
            rewards (th.tensor): tensor of size n_networks x 1 with the correspinding reward obtained
                                 in each env for a specific action a
            
            (for DQN, if not at last round) 
            next_obs (dict of th.tensor): observation of env(s) following action a
        """

        self.source_node = self.current_node
        
        # add to big loss counter if 1 present in rewards_idx
        rewards_idx = torch.unsqueeze(self.action_space_idx[self.network_idx,self.current_node,action], dim=-1)
        # add to big loss counter if 1 present in rewards_idx
        self.big_loss_counter= torch.add(self.big_loss_counter,(rewards_idx==1).int())
        # obtain numerical reward value correspongint o reward indices
        rewards = self.reward_map[rewards_idx]
        # add rewards to reward balance
        self.reward_balance = torch.add(self.reward_balance,rewards)

        # update the current node for all envs
        self.current_node = action
        # update step counter
        self.step_counter = torch.add(self.step_counter,1)
        if torch.all(self.step_counter == 8):
            self.is_done = True

        # (relevant for DQN) if we are in the last step return only rewards,
        # else return also observation after action has been taken
        if round!=7:
            next_obs = self.observe()
            return next_obs,rewards
        else:
            return rewards
        
        
    def get_state(self):
        """
        Returns the current state of the environment.
        State information given by this funciton is less detailed compared
        to the observation. 
        """
        return {'current_node':self.current_node,
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}


    def observe(self):
        """
        Returns observation from the environment. The observation is made of a boolean mask indicating which 
        actions are valid in each env + a main observation matrix.
        For each node in each environment the main observation matrix contains contatenated one hot encoded info on:
        - reward index 
        - step counter
        - loss counter (has an edge with associated reward of -100 been taken yet)
        - level (what is the level of the current/starting node of an edge)

        was max_step + 1 before, now only max step because we cre about step 0 1 2 3 4 5 6 7 (in total 8 steps)

        Returns:
            obs (dict of th.tensor): main observation matrix (key=obs) + boolean mask (key=mask)
        """
        self.observation_matrix = torch.zeros((self.N_NETWORKS,self.N_NODES,(self.N_REWARD_IDX+self.MAX_STEP+1+self.N_LEVELS)),dtype=torch.long)
        self.next_rewards_idx = torch.squeeze(torch.unsqueeze(self.action_space_idx[self.network_idx,self.current_node,:],dim=-1))
        self.next_edges_levels_idx = torch.squeeze(torch.unsqueeze(self.level_space[self.network_idx,self.current_node,:],dim=-1))

        # one hot encoding of reward_idx
        #print(f'one hot encoding of reward idx shape \n')
        #print(F.one_hot(self.next_rewards_idx,num_classes=self.N_REWARD_IDX).shape)
        #print(f'example one hot encoding of reward idx for node 3 of all networks \n')
        #print(self.next_rewards_idx )
        #print(F.one_hot(self.next_rewards_idx,num_classes=self.N_REWARD_IDX)[:,3,:])
        self.observation_matrix[:,:,:self.N_REWARD_IDX] = F.one_hot(self.next_rewards_idx,num_classes=self.N_REWARD_IDX)
        
        # one hot encoding of step counter TODO: n_steps or (n_steps+1)
        #print(f'one hot encoding of step counter \n')
        print(self.step_counter)
        print(F.one_hot(self.step_counter,num_classes=self.MAX_STEP))
        self.observation_matrix[:,:,self.N_REWARD_IDX:(self.N_REWARD_IDX+self.MAX_STEP)] = torch.repeat_interleave(F.one_hot(self.step_counter,num_classes=self.MAX_STEP), self.N_NODES,dim=1)

        # big loss counter
        #print(f'big loss counter \n')
        #print(self.big_loss_counter.shape)
        #print(f'big loss counter repeated to fit 10 nodes \n')
        #print(torch.repeat_interleave(self.big_loss_counter, self.N_NODES,dim=1))
        #print(f'big loss counter repeated to fit 10 nodes shape \n')
        #print(torch.repeat_interleave(self.big_loss_counter, self.N_NODES,dim=1).shape)

        # :,:,-1
        self.observation_matrix[:,:,(self.N_REWARD_IDX+self.MAX_STEP):(self.N_REWARD_IDX+self.MAX_STEP+1)] = torch.unsqueeze(torch.repeat_interleave(self.big_loss_counter, self.N_NODES,dim=1),dim=-1)
        #print(f'shape of main observation matrix: {self.observation_matrix.shape}')

        # one hot encoding of current levle node for an edge
        self.observation_matrix[:,:,-5:] = F.one_hot(self.next_edges_levels_idx,num_classes=self.N_LEVELS)
        
        # the second observation matrix (boolean mask indicating valid actions)
        self.next_nodes = torch.squeeze(torch.unsqueeze(self.edge_is_present[self.network_idx,self.current_node,:],dim=-1))
        
        return {'mask':self.next_nodes,
                'obs':self.observation_matrix}



# For quick testing purposes, comment out if not needed

#with open(os.path.join(os.getcwd(),'data','train.json')) as json_file:
#      test = json.load(json_file)
#env_test = Reward_Network(test[10:13])


#env_test.reset()
#obs = env_test.observe()

#next_obs,rewards= env_test.step(torch.tensor([3,6,2]))
# print(rewards,rewards.shape)
# print('\n')
# #print(env_test.observe())
