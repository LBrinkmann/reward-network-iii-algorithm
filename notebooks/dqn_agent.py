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
import glob
import logging
import re
import sys
import os
import time,datetime
import matplotlib.pyplot as plt
import seaborn as sns

# import Pytorch modules
import torch
import torch.nn as nn               # layers 
import torch.optim as optim         # optimizers
import torch.nn.functional as F     # 

# import the custom reward network environment class
from environment import Reward_Network

#######################################
## Initialize agent
#######################################

class DQN_agent:
    def __init__(self, state_dim: int, action_dim: int, save_dir: str):
        
        # assert tests
        # TODO

        # specify environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        # specify DNN used by the agent in training and learning Q(s,a) from experience 
        # to predict the most optimal action - we implement this in the Learn section
        # two DNNs - Q_{online} and Q_{target}- that independently approximate the 
        # optimal action-value function.
        self.net = DQN(self.state_dim, self.action_dim).float()

        # specify \epsilon greedy policy exploration parameters 
        # relevant in exploration
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        
        # specify experience replay and memory buffer parameters
        self.memory = deque(maxlen=100000)
        self.save_every = 5e5  # no. of experiences between saving 
        self.batch_size = 32

        # specify \gamma parameter (how far-sighted our agent is)
        self.gamma = 0.9

        # specify training loop parameters
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # specify which loss function and which optimizer to use (and their respective params)
        self.lr = 0.00025
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # specify output directory
        self.save_dir = save_dir

        # specify if cuda is available
        self.use_cuda = torch.cuda.is_available()

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action (explore) or use DNN to
        select the action which, given $S=s$, is associated to highest $Q(s,a)$
        Inputs:
        - state (tuple or list??): A single observation of the current state, shape is (state_dim)
        Outputs:
        - action_idx (int): An integer representing which action the AI agent will perform
        """
        # assert tests
        assert len(state) == self.state_dim and isinstance(state, list), \
            f"Wrong length of state representation: expected {self.state_dim}, got: {len(state)}"


        # EXPLORE (select random action from the action space)
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # or EXPLOIT
        else:
            self.state = state.__array__()
            if self.use_cuda:
                self.state = torch.tensor(self.state).cuda()
            else:
                self.state = torch.tensor(self.state)
            self.state = self.state.unsqueeze(0)
            # return Q values for each action in the action space | S=s
            self.action_values = self.net(self.state, model="online")
            # select action with highest Q value
            self.action_idx = torch.argmax(self.action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        
        return action_idx


    def cache(self, state, next_state, action, reward, done):
        """
        Add the experience/ transition from one state to another,
        along with action and reward info, to memory
        Inputs:
        - state: 
        - next_state: 
        - action: 
        - reward: (int) r \in {-100,-20,0,20,140}
        - done: bool indicator of whether step_number>8
        """
        # assert tests
        assert len(state)==2, f'expected a tuple of (state,step number), got {len(state)}'
        assert self.node_ids.count(state(0))>0, f'expected a node id between A and F, got node {state(0)}'
        assert self.action_space.count(action)>0, f'expected an action between 0 (left) and 1 (right), got {state(1)}'


        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of (batch_size) experiences/transitions 
        from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        This function returns the temporal difference estimate for a (state,action) pair.
        In other words, it returns the predicted optimal $Q^*$ for a given state s
        """
        # we use the online model here
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    # note that we don't want to update target net parameters by backprop (hence the torch.no_grad),
    # instead the online net parameters will take the place of the target net parameters periodically
    @torch.no_grad()
    def td_target(self, reward, next_state, done):

        # Because we don’t know what next action a' will be, 
        # we use the action a' that maximizes Q_{online} in the next state s'
 
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target):
        """
        This function updates the patameters of the "online" DQN by means of backpropagation.
        The loss value is given by the (td_estimate - td_target)

        \theta_{online} <- \theta_{online} + \alpha((TD_estimate - TD_target))
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """
        This function periodically copies \theta_online parameters 
        to be the \theta_target parameters
        """
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def save(self):
        """
        This function saves model checkpoints
        """
        save_path = (self.save_dir / f"Reward_network_iii_dqn_model_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(dict(model=self.net.state_dict(),
                        exploration_rate=self.exploration_rate),
                   save_path)
        print(f"Reward_network_iii_dqn_model checkpoint saved to {save_path} at step {self.curr_step}")

    def learn(self):
        """
        Update online action value (Q) function with a batch of experiences
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


#######################################
## Initialize DQN TODO: adapt to our task
#######################################
# history as additional input for the model
# observation!!
# state is (node,step), 
# action space only L,R or more actions? action space -> all possible edges, Q value 


#edge -> (reward, step i am in, how many large losses yet), output Q value for each outgoing edge
#input dims (batch, edges, feature vector)


class DQN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def reset(self):
        self.hidden = None

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        h, self.hidden = self.rnn(x, self.hidden)
        q = self.linear2(h)
        return q


#######################################
## Initialize Logger
#######################################

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()



#######################################
## MAIN - START DQN AGENT
#######################################

if __name__ == "__main__":

    # get arguments from shell script
    if len(sys.argv) > 1:
        method_ = sys.argv[1]
        if method_ =='dqn':
            select_data = sys.argv[2]
    
    # --------Specify paths--------------------------
    # Specify directories (cluster)
    home_dir = f"/home/mpib/bonati"
    project_dir = os.path.join(home_dir,'CHM','reward_networks')
    data_dir = os.path.join(project_dir, 'data','rawdata') 
    logs_dir = os.path.join(project_dir, 'logs')

    if method_=='dqn':
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir.mkdir(parents=True)
        if not os.path.exists(os.path.join(logs_dir,method_)):
            os.makedirs(os.path.join(logs_dir,method_))
        logging_fn = os.path.join(logs_dir,method_,f'{method_}_{time.strftime("%Y_%m-%d_%H-%M-%S")}.log')
    
    # -------Set-up logging---------------------
    # Get current data and time as a string:
    timestr = time.strftime("%Y_%m-%d_%H-%M-%S")
    # start logging:
    logging.basicConfig(
        filename=logging_fn, level=logging.DEBUG, format='%(asctime)s %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')

    # Add basic script information to the logger
    logging.info("------Start Running dqn_agent.py------")
    logging.info(f"Operating system: {sys.platform}\n")

    # ---------Start analysis------------------------------
    if method_=='dqn':
        logging.info(f"Starting method DQN...\n")
        
        # specify if cuda is available
        use_cuda = torch.cuda.is_available()
        logging.info(f"Using CUDA: {use_cuda} \n")
        
        # initialize environment
        env = Reward_Network(network)
        
        # initialize Agent + Logger
        AI_agent = DQN_agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
        logger = MetricLogger(save_dir)

        EPISODES = 10
        for e in range(EPISODES):

            state = env.reset()
            # Solve the reward networks!
            while True:

                # Run agent on the state
                action = AI_agent.act(state)
                # Agent performs action
                next_state, reward, done, info = env.step(action)
                # Remember
                AI_agent.cache(state, next_state, action, reward, done)
                # Learn
                q, loss = AI_agent.learn()
                # Logging
                logger.log_step(reward, loss, q)
                # Update state
                state = next_state
                # Check if end of game
                if done or info["flag_get"]:
                    break

            logger.log_episode()
            if e % 20 == 0:
                logger.record(episode=e, epsilon=AI_agent.exploration_rate, step=AI_agent.curr_step)