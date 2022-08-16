import gym
from gym import spaces
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import os
import glob
import logging
import json
import random
sns.set_theme()

class Reward_Network(gym.Env):
    
    def __init__(self, network):
        
        #-------------
        # assert tests
        #-------------
        assert len(network)>0, f'No reward networks were passed ot the environment class'

        # reward network information from json file
        self.network = network
        
        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.MAX_STEP = 8

        # network info
        self.id = self.network['network_id']
        self.nodes = [n['id'] for n in self.network['nodes']]
        self.action_space = self.network['actions']
        self.possible_rewards = [-100, -20, 0, 20, 140]
        self.reward_range = (min(self.possible_rewards)*self.MAX_STEP,self.network['max_reward'])
    

    def reset(self):
        # Reset the state of the environment to an initial state
        self.reward_balance = self.INIT_REWARD
        self.step_counter = self.INIT_STEP
        self.is_done = False
        
        # Set the current step to the starting node of the graph
        self.current_node = self.network['starting_node'] #self.G[0]['starting_node']



    def step(self, action):
        # Execute one time step within the environment
        #self._take_action(action)
        self.source_node = action['sourceId']
        self.reward_balance += action['reward']
        self.current_node = action['targetId']
        self.step_counter += 1

        if self.step_counter == 8:
            self.is_done = True
        
        return {'source_node':self.source_node,
                'current_node':self.current_node,
                'reward':action['reward'],
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}


    def observe(self):
        """
        this function returns observation from the environment
        """
        return {'current_node':self.current_node,
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}

    def render(self):
        # Render the environment to the screen
        print('TODO')


if __name__ == 'main':

    with open(r'C:\Users\Sara Bonati\Desktop\MPI work\Machines\Reward_network_task\data\dev\train.json') as json_file:
        networks = json.load(json_file)

    env = Reward_Network(networks[0])
    env.reset()

    print(f'START \n We are in node {env.current_node}')

    while env.is_done==False:
        print(env.observe())
        print('\n')
        random_action = random.choice([a for a in env.action_space if a['sourceId']==env.current_node])
        obs = env.step(random_action)
        print(obs)