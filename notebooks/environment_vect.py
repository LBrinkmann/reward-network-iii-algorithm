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
from collections import Counter
from typing import Optional


def restructure_edges(network):
    """
    This function restructures the edges from list of dicts
    to one dict, to improve construction of edges matrix and 
    env vectorization
    """

    new_edges= {'source_id':[],'target_id':[],'reward':[]}
    for e in network['edges']:
        new_edges['source_id'].append(e['source_id'])
        new_edges['target_id'].append(e['target_id'])
        new_edges['reward'].append(e['reward'])
    return new_edges 


class Reward_Network(gym.Env):
    
    def __init__(self, network, to_log=False):
        
        #-------------
        # assert tests TODO
        #-------------

        # reward network information from json file (can be just one network or multiple networks)
        self.network = network
       
        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.MAX_STEP = 8
        self.N_NODES = 10
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
        for n in range(self.N_NETWORKS):
            buffer_action_space = torch.full((self.N_NODES, self.N_NODES), 0).long()
            source = torch.tensor(new_edges[n]['source_id']).long()
            target = torch.tensor(new_edges[n]['target_id']).long()
            reward = torch.tensor(new_edges[n]['reward']).long()
            reward.apply_(lambda val: self.possible_rewards.get(val, 0))
            buffer_action_space[source,target]=reward
            self.action_space_idx[n,:,:] = buffer_action_space

        # define reward map
        self.reward_map = torch.zeros(max(self.possible_rewards.values()) + 1, dtype=torch.long)
        self.reward_map[list(self.possible_rewards.values())] = torch.tensor(list(self.possible_rewards.keys()), dtype=torch.long)

        # boolean adjacency matrix 
        self.edge_is_present = torch.squeeze(torch.unsqueeze(self.action_space_idx!=0,dim=-1))
        

    def reset(self):
        # Reset the state of the environment to an initial state
        self.reward_balance = torch.full((self.N_NETWORKS,1),self.INIT_REWARD)
        self.step_counter = torch.full((self.N_NETWORKS,1),self.INIT_STEP)
        self.is_done = False 
        self.current_node = self.starting_nodes.clone()

        print('Environment initialized: \n')
        print(f'- set of nodes of shape {self.nodes.shape}')
        print(f'- action space of shape {self.action_space_idx.shape}')
        print(f'- reward balance of shape {self.reward_balance.shape}')
        print(f'- step counter of shape {self.step_counter.shape}')
        print(f'- current node of shape {self.current_node.shape}')

    
    def step(self, action):
        '''
        Take a step in all environments; here action corresponds to the target nodes for each env
        action_i \in [0,1,2,3,4,5,6,7,8,9]
        '''

        self.source_node = self.current_node
        print(f'Source nodes are: {self.current_node}, we are going to new nodes {action}')
        
        rewards_idx = torch.unsqueeze(self.action_space_idx[self.network_idx,self.current_node,action], dim=-1)
        rewards = self.reward_map[rewards_idx]
        self.reward_balance = torch.add(self.reward_balance,rewards)
        print(f'We get rewards : {rewards[:,0]} and the new reward balance is: {self.reward_balance[:,0]}')
        self.current_node = action
        print(f'Now we are in nodes: {self.current_node}')
        self.step_counter = torch.add(self.step_counter,1)
        print(f'Step counter for all networks is: {self.step_counter[:,0]}')
        print('\n')

        if torch.all(self.step_counter == 8):
            self.is_done = True

        return rewards
        
        
    def get_state(self):
        """
        this function returns the current state of the environment.
        State information given by this funciton is less detailed compared
        to the observation. 
        """
        return {'current_node':self.current_node,
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}


    def get_possible_rewards(self,obs):
        """
        this function returns the next possible rewards given an observation;
        the rewards are selected using boolean masking, and the resulting array is split
        into sub-tensors whose size is given by how many valid edges are present in each network
        in the current observation.
        """
        self.n_rewards_per_network = torch.count_nonzero(obs['next_possible_nodes'],dim=1).tolist()
        self.next_rewards_all = torch.masked_select(obs['next_possible_rewards_idx'][self.network_idx],obs['next_possible_nodes'][self.network_idx])
        self.next_rewards_per_network = torch.split(self.next_rewards_all,self.n_rewards_per_network)
        
        return self.next_rewards_per_network

    def observe(self):
        """
        this function returns observation from the environment
        """

        self.next_nodes = torch.squeeze(torch.unsqueeze(self.edge_is_present[self.network_idx,self.current_node,:],dim=-1))
        self.next_rewards_idx = torch.squeeze(torch.unsqueeze(self.action_space_idx[self.network_idx,self.current_node,:],dim=-1))

        return {'current_node':self.current_node,
                'next_possible_nodes':self.next_nodes,
                'next_possible_rewards_idx':self.next_rewards_idx,
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}