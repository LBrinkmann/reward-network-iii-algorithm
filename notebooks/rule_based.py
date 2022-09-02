#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Parameters
project_folder = "../.."
solution_columns = ["network_id", "strategy", "step", "source_node", "current_node", "reward", "total_reward"]


# # Rule
# 
# * in general always take the edge with the larger payoff
# * however: take the first large loss (only the first)

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import os
import glob
import json
import random
import logging
sns.set_theme()

from environment import Reward_Network


# In[2]:


project_folder= 'c:\\Users\\Sara Bonati\\Desktop\\MPI_work\\Machines\\Reward_network_task'
solution_columns= ['network_id','strategy','step','source_node','current_node','reward','total_reward']


# In[3]:


# directory management
code_dir = os.path.join(project_folder,'reward-network-iii-algorithm')
data_dir = os.path.join(project_folder,'data','rawdata')
solutions_dir = os.path.join(project_folder,'data','solutions')
to_log=False


# ### define agent class

# In[11]:


class Rule_Agent:

    def __init__(self,strategy):

        # assert tests
        assert strategy in ['highest_payoff','take_first_loss','random'], f'Strategy name must be one of {["highest_payoff","take_first_loss","random"]}, got {strategy}'
        
        self.strategy =  strategy        
        self.colors = {'highest_payoff':'skyblue','take_first_loss':'orangered','random':'springgreen'}


    def select_action(self,possible_actions,possible_actions_rewards):
        """
        This method selects an action to take in the environment based on current strategy
        """

        if self.strategy == 'take_first_loss':
            print(self.strategy,self.loss_counter,possible_actions_rewards)

        if self.strategy=='random':
            return random.choice(possible_actions)

        # take first loss -> select among possible actions the one that gives best reward BUT make sure to take a first big loss
        if self.strategy == 'take_first_loss' and self.loss_counter<1 and -100 in possible_actions_rewards:
            self.loss_counter +=1
            print(np.argwhere(possible_actions_rewards==-100)[0][0])

            if len(np.argwhere(possible_actions_rewards==-100)[0])!=2: # that is, we have only one big loss in the possible actions
                print('only one big loss')
                return possible_actions[np.argwhere(possible_actions_rewards==-100)[0][0]]
            else: # else if both actions lead to big loss pick a random one
                print('two big losses, select one action at random')
                return possible_actions[random.choice(np.argwhere(possible_actions_rewards==-100)[0])]
        else:
            # highest payoff -> select among possible actions the one that gives best reward
            #print(f'take highest payoff {self.possible_actions[np.argmax(self.possible_actions_rewards)]}')
            #print('\n')
            try:
                print(f'rewards: {possible_actions_rewards}')
                if not np.all(possible_actions_rewards == possible_actions_rewards[0]):
                    print(f'choose: {np.argmax(possible_actions_rewards)}')
                    return possible_actions[np.argmax(possible_actions_rewards)]
                else:
                    print(f'choose random')
                    return random.choice(possible_actions)
            except:
                print('Error')
                print(self.environment.id)
                print(self.environment.action_space)

    def solve(self,network):
        """ 
        the agent solves the task, with different constraints depending on the strategy.
        Returns solved reward network in tabular form
        """
        if self.strategy == 'take_first_loss':
            self.loss_counter = 0 # to reset!

        self.solution = []
        self.solution_filename = os.path.join(solutions_dir,f'{network["network_id"]}_Solution_log_{self.strategy}.csv')    
        
        self.environment = Reward_Network(network)
        self.environment.reset()
        print(self.environment.get_state())
        print(self.environment.observe())
        
        while self.environment.is_done==False:
            s = []
            obs = self.environment.observe()
            print(obs.keys())
            #a = self.select_action()
            a = self.select_action(obs['actions_available'],obs['next_possible_rewards'])
            step = self.environment.step(a)
            #obs = self.environment.step(a)
            s.append(self.environment.id)
            s.append(self.strategy)
            s.append(step['n_steps'])
            s.append(step['source_node'])
            s.append(step['current_node'])
            s.append(step['reward'])
            s.append(step['total_reward'])
            self.solution.append(s)
        print('\n')
        self.solution_df = pd.DataFrame(self.solution, columns = solution_columns)
        self.solution_df.to_csv(self.solution_filename,sep='\t')

    def save_solutions(self):
        self.all_solutions_filename = os.path.join(solutions_dir,f'{self.strategy}_train.csv')
        self.solutions_fn = glob.glob(solutions_dir+f'/*{self.strategy}*.csv')
        self.df = pd.concat([pd.read_csv(s,sep='\t') for s in self.solutions_fn],ignore_index=True)
        self.df[solution_columns].to_csv(self.all_solutions_filename,sep='\t')

    def inspect_solutions(self):
        self.df = pd.read_csv(os.path.join(solutions_dir,f'{self.strategy}_train.csv'),sep='\t')
        g = sns.displot(data=self.df[self.df['step']==8], x="total_reward", kde=True, color=self.colors[self.strategy])
        g.set(xlim=(-400,400),xlabel='Final reward',ylabel='Count')
        plt.show()


# ## Solve networks + compare strategies' results

# In[12]:


with open(os.path.join(data_dir,'train.json')) as json_file:
    train = json.load(json_file)
print(f'NUMBER OF NETWORKS FOUND IN TRAIN.JSON: {len(train)}')


# In[13]:


print("-------A (highest payoff)-------")
A = Rule_Agent('highest_payoff')

for network in train:
    A.solve(network)
print('\n')


# In[14]:


print("-------B (take_first_loss)-------")
B = Rule_Agent('take_first_loss')

for network in train:
    B.solve(network)
print('\n')


# In[15]:


print("-------C (random)-------")
C = Rule_Agent('random')

for network in train:
    C.solve(network)
print('\n')


# In[16]:


A.save_solutions()
A.inspect_solutions()


# In[17]:


B.save_solutions()
B.inspect_solutions()


# In[18]:


C.save_solutions()
C.inspect_solutions()


# ## analyze solution final rewards and compare the different strategies

# In[19]:


if os.path.exists(A.all_solutions_filename) and os.path.exists(B.all_solutions_filename) and os.path.exists(C.all_solutions_filename):
    
    # load solution data 
    strategy_A = pd.read_csv(A.all_solutions_filename,sep='\t')
    strategy_B = pd.read_csv(B.all_solutions_filename,sep='\t')
    strategy_C = pd.read_csv(C.all_solutions_filename,sep='\t')
    # create solution data file with all strategies in one file 
    strategy_data = pd.concat([strategy_A,strategy_B,strategy_C],ignore_index=True)[solution_columns]
    strategy_data_final=strategy_data[strategy_data['step']==8]
    strategy_data.to_csv(os.path.join(solutions_dir,'ALL_SOLUTIONS.csv'),sep='\t')

    # hist plot
    g=sns.displot(data=strategy_data_final, x="total_reward", hue="strategy", kind="hist")
    g.set(xlabel='Final total reward',ylabel='Count',title=f'Strategy final total reward comparison')
    plt.show()


# In[20]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# perform Tukey's test
tukey = pairwise_tukeyhsd(endog=strategy_data_final['total_reward'],
                          groups=strategy_data_final['strategy'],
                          alpha=0.05)

#display results
print(tukey)


# In[21]:


sns.boxplot(x="strategy", y="total_reward", data=strategy_data_final)
plt.show()


# In[33]:


g = sns.relplot(
    data=strategy_data,
    x="step", y="reward", col='strategy',hue='strategy',
    height=4, aspect=.9, kind="line",palette={'highest_payoff':'skyblue','take_first_loss':'orangered','random':'springgreen'}
)
for ax in g.axes.flat:
    labels = ax.get_xticklabels() # get x labels
    ax.set_xticks(ticks=[1,2,3,4,5,6,7,8]) # set new labels
    ax.set_xticklabels(fontsize=10,labels=[str(i) for i in range(1,9)])
plt.show()


# ## inspect solutions more closely

# In[23]:


strategy_A = pd.read_csv(A.all_solutions_filename,sep='\t')[solution_columns]
strategy_B = pd.read_csv(B.all_solutions_filename,sep='\t')[solution_columns]
strategy_C = pd.read_csv(C.all_solutions_filename,sep='\t')[solution_columns]

strategy_A_wide = strategy_A.pivot_table(index="network_id", columns="step", values="reward")
strategy_A_wide.head(30)


# In[24]:


strategy_B_wide = strategy_B.pivot_table(index="network_id", columns="step", values="reward")
strategy_B_wide.head(30)


# In[25]:


strategy_C_wide = strategy_C.pivot_table(index="network_id", columns="step", values="reward")
strategy_C_wide.head(30)


# In[29]:


strategy_A_wide.mean(axis=0)


# In[30]:


strategy_B_wide.mean(axis=0)


# In[31]:


strategy_C_wide.mean(axis=0)

