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
import time
sns.set_theme()



class Reward_Network(gym.Env):
    
    def __init__(self, network, to_log=False):
        
        #-------------
        # assert tests
        #-------------

        # reward network information from json file
        self.network = network
       
        # initial reward and step values
        self.INIT_REWARD = 0
        self.INIT_STEP = 0
        self.MAX_STEP = 8

        # network info
        self.id = self.network['network_id']
        self.nodes = [n['node_num'] for n in self.network['nodes']]
        self.action_space = self.network['edges']
        self.possible_rewards = [-100, -20, 0, 20, 140]
        self.reward_range = (min(self.possible_rewards)*self.MAX_STEP,self.network['max_reward'])

        if to_log:
            # logging info
            logging_fn = os.path.join('../../logs',f'{self.id}_{time.strftime("%Y_%m-%d_%H-%M-%S")}.log')
            # start logging:
            logging.basicConfig(filename=logging_fn, 
                                level=logging.DEBUG, 
                                format='%(asctime)s %(message)s',
                                datefmt='%d/%m/%Y %H:%M:%S')

    

    def reset(self):
        # Reset the state of the environment to an initial state
        self.reward_balance = self.INIT_REWARD
        self.step_counter = self.INIT_STEP
        self.is_done = False
        
        # Set the current step to the starting node of the graph
        self.current_node = self.network['starting_node'] #self.G[0]['starting_node']
        logging.info(f'NETWORK {self.id} \n')
        logging.info(f'INIT: Reward balance {self.reward_balance}, n_steps done {self.step_counter}')


    def step(self, action):
        # Execute one time step within the environment
        #self._take_action(action)
        self.source_node = action['source_num'] # OR with _id alternatively
        self.reward_balance += action['reward']
        self.current_node = action['target_num']
        self.step_counter += 1

        if self.step_counter == 8:
            self.is_done = True
           

        logging.info(f'Step {self.step_counter} : we go from node {self.source_node} to node {self.current_node}')
        logging.info(f'Reward for this edge: {action["reward"]} (in total {self.reward_balance})')
        logging.info(f'DONE? {self.is_done})')

        return {'source_node':self.source_node,
                'current_node':self.current_node,
                'reward':action['reward'],
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}


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

    def observe(self):
        """
        this function returns observation from the environment
        """
        return {'current_node':self.current_node,
                'actions_available':[n for n in self.action_space if n['source_num'] == self.current_node],
                'next_possible_nodes':np.asarray([n['target_num'] for n in self.action_space if n['source_num'] == self.current_node]),
                'next_possible_rewards':np.asarray([n['reward'] for n in self.action_space if n['source_num'] == self.current_node]),
                'total_reward':self.reward_balance,
                'n_steps':self.step_counter,
                'done':self.is_done}

    def render(self):
        # Render the environment to the screen
        print('TODO')


# For quick testing purposes, comment out if not needed

#with open(os.path.join('../data','test.json')) as json_file:
#    test = json.load(json_file)
#env_test = Reward_Network(test[0])
#check_env(env_test)
#env_test.reset()
#print(env_test.get_state())
#print(env_test.observe())