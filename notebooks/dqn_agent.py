# This file specifices the Deep Q Learning AI agent model to solve a Reward Network DAG
# See also: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# and: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
#
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################

import argparse
import datetime
import json
# filesystem and log file specific imports
import os
import pickle
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
# import modules
import numpy as np
# import Pytorch + hyperparameter tuning modules
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

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
        # i = obs['obs'][:,:,:].float()
        i = obs.float()
        x = F.relu(self.linear1(i))
        # q = F.relu(self.linear2(x))
        q = self.linear2(x)
        return q


#######################################
## Initialize agent
#######################################


class Agent:
    def __init__(
            self, obs_dim: int, config: dict, action_dim: tuple, save_dir: str, device
    ):
        """
        Initializes an object of class Agent

        Args:
        obs_dim (int): number of elements present in the observation (2, action space observation + valid
        action mask)
        config (dict): a dict of all parameters and constants of the reward network problem (e.g. number
        of nodes, number of networks..)
        action_dim (tuple): shape of action space of one environment
        save_dir (str): path to folder where to save model checkpoints into device: torch device
        (cpu or cuda)
        """

        # assert tests
        assert os.path.exists(
            save_dir
        ), f"{save_dir} is not a valid path (does not exist)"

        # specify environment parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_networks = config.n_networks

        # specify DNNs used by the agent in training and learning Q(s,a) from experience
        # to predict the most optimal action - we implement this in the Learn section
        # two DNNs - policy net with Q_{online} and target net with Q_{target}- that
        # independently approximate the optimal action-value function.
        input_size = (
            config.n_networks,
            config.n_nodes,
            20,
        )
        hidden_size = (
            config.n_networks,
            config.n_nodes,
            config.nn_hidden_size,
        )
        # one q value for each action
        output_size = (
            config.n_networks,
            config.n_nodes,
            1,
        )
        self.policy_net = DQN(input_size, output_size, hidden_size)
        self.target_net = DQN(input_size, output_size, hidden_size)

        # specify \epsilon greedy policy exploration parameters (relevant in exploration)
        self.exploration_rate = 1
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # specify \gamma parameter (how far-sighted our agent is)
        self.gamma = 0.9

        # specify training loop parameters
        self.burnin = 10  # min. experiences before training
        self.learn_every = 5  # no. of experiences between updates to Q_online
        self.sync_every = (
            config.nn_update_frequency
        )  # 1e4  # no. of experiences between Q_target & Q_online sync
        self.save_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # specify which loss function and which optimizer to use (and their respective params)
        self.lr = config.lr
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = th.nn.SmoothL1Loss(reduction="none")

        # specify output directory
        self.save_dir = save_dir

        # torch device
        self.device = device

    @staticmethod
    def apply_mask(q_values, mask):
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

        Args: obs (dict with values of th.tensor): observation from the env(s) comprising of one hot encoded
        reward+step counter+ big loss counter and a valid action mask

        Returns:
            action (th.tensor): node index representing next nodes to move to for all envs
            action_values (th.tensor): estimated q values for action
        """

        # assert tests
        assert len(obs) == self.obs_dim and isinstance(
            obs, dict
        ), f"Wrong length of state representation: expected dict of size {self.obs_dim}, got: {len(obs)}"

        # for the moment keep a random action selection strategy to mock agent choosing action
        # action_idx = th.squeeze(th.multinomial(obs['mask'].type(th.float),1))

        # EXPLORE (select random action from the action space)
        # if np.random.rand() < self.exploration_rate:
        # random_actions = th.squeeze(th.multinomial(obs['mask'].type(th.float),1))
        random_actions = th.multinomial(obs["mask"].type(th.float), 1)
        # print(f'random actions {th.squeeze(random_actions,dim=-1)}')

        # EXPLOIT (select greedy action)
        # return Q values for each action in the action space A | S=s
        action_q_values = self.policy_net(obs["obs"])
        # apply masking to obtain Q values for each VALID action (invalid actions set to very low Q value)
        action_q_values = self.apply_mask(action_q_values, obs["mask"])
        # select action with highest Q value
        greedy_actions = th.argmax(action_q_values, dim=1)  # .item()
        # print(f'greedy actions {th.squeeze(greedy_actions,dim=-1)}')

        # select between random or greedy action in each env
        select_random = (
                th.rand(self.n_networks, device=self.device)
                < self.exploration_rate
        ).long()

        action = select_random * random_actions + (1 - select_random) * greedy_actions

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return action[:, 0], action_q_values

    def td_estimate(self, state, state_mask, action):
        """
        This function returns the TD estimate for a (state,action) pair

        Args:
            state (dict of th.tensor): observation
            state_mask (th.tensor): boolean mask to the observation matrix
            action (th.tensor): actions taken in all envs from memory sample

        Returns:
            td_est: Q∗_online(s,a)
        """

        # we use the online model here we get Q_online(s,a)
        td_est = self.policy_net(state)
        # apply masking (invalid actions set to very low Q value)
        td_est = self.apply_mask(td_est, state_mask)
        # select Q values for the respective actions from memory sample
        td_est_actions = (
            th.squeeze(td_est).gather(-1, th.unsqueeze(action, -1)).squeeze(-1)
        )
        return td_est_actions

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

        # state has dimensions batch_size,n_steps,n_networks,n_nodes,
        # length of one hot encoded observation info - in our case 20
        print(f"state shape -> {state.shape}")
        # reward has dimensions batch_size,n_steps,n_networks,1
        print(f"reward shape -> {reward.shape}")

        next_max_Q2 = th.zeros(state.shape[:3], device=self.device)
        print(f"next_max_Q2 shape -> {next_max_Q2.shape}")

        # target q has dimensions batch_size,n_steps,n_networks,n_nodes,1
        target_Q = self.target_net(state)
        target_Q = self.apply_mask(target_Q, state_mask)
        print(f"target_Q shape -> {target_Q.shape}")
        # next_Q has dimensions batch_size,(n_steps -1),n_networks,n_nodes,1
        # (we skip the first observation and set the future value for the terminal state to 0)
        next_Q = target_Q[:, 1:]
        print(f"next_Q shape -> {next_Q.shape}")

        # next_max_Q has dimension batch,steps,networks
        next_max_Q = th.squeeze(next_Q).max(-1)[0].detach()
        print(f"next_max_Q shape -> {next_max_Q.shape}")

        next_max_Q2[:, :-1, :] = next_max_Q

        return th.squeeze(reward) + (self.gamma * next_max_Q2)

    def update_Q_online(self, td_estimate, td_target):
        """
        This function updates the parameters of the "online" DQN by means of backpropagation.
        The loss value is given by F.smooth_l1_loss(td_estimate - td_target)

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
        self.optimizer.zero_grad()

        # we apply mean to get from dimension (batch_size,1) to 1 (scalar)
        loss.mean().backward()

        # truncate large gradients as in original DQN paper
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.mean().item()

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
        save_path = os.path.join(
            self.save_dir,
            f"Reward_network_iii_dqn_model_{int(self.curr_step // self.save_every)}.chkpt",
        )
        th.save(
            dict(
                model=self.policy_net.state_dict(),
                exploration_rate=self.exploration_rate,
            ),
            save_path,
        )
        print(
            f"Reward_network_iii_dqn_model checkpoint saved to {save_path} at step {self.curr_step}"
        )

    def learn(self, memory_sample):
        """
        Update online action value (Q) function with a batch of experiences.
        As we sample inputs from memory, we compute loss using TD estimate and TD target, then backpropagate this loss down
        Q_online to update its parameters θ_online

        Args:
            memory_sample (dict with values as th.tensors): sample from Memory buffer object, includes
            as keys 'obs','mask','action','reward'

        Returns:
            (th.tensor,float): estimated Q values + loss value
        """

        # if applicable update target net parameters
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # if applicable save model checkpoints
        # if self.curr_step % self.save_every == 0:
        #    self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Get TD Estimate (mask already applied in function)
        td_est = self.td_estimate(memory_sample["obs"], memory_sample["mask"], memory_sample["action"])
        print(f"Calculated td_est of shape {td_est.shape}")
        # Get TD Target
        td_tgt = self.td_target(memory_sample["reward"], memory_sample["obs"], memory_sample["mask"])
        print(f"td_tgt shape {td_tgt.shape}")

        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss


#######################################
## Initialize Memory buffer class
# a data structure which temporarily saves the agent’s observations,
# allowing our learning procedure to update on them multiple times
#######################################


class Memory:
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
        self.memory = {
            k: th.zeros(
                (self.size, self.n_rounds, *t.shape), dtype=t.dtype, device=self.device
            )
            for k, t in obs.items()
            if t is not None
        }

    def finish_episode(self):
        """Moves the currently active slice in memory to the next episode."""
        self.episodes_stored += 1
        self.current_row = (self.current_row + 1) % self.size

    def store(self, round_num, **state):
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
                self.memory[k][self.current_row, round_num] = t.to(self.device)

    def sample(self, batch_size, device, **kwargs):
        """Samples form the memory.

        Returns:
            dict | None: Dict being stored. If the batch size is larger than the number
            of episodes stored 'None' is returned.
        """
        if len(self) < batch_size:
            return None
        random_memory_idx = th.randperm(len(self))[:batch_size]
        print(f"random_memory_idx", random_memory_idx)

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
        self.q_step_log = th.zeros(n_steps, n_networks, n_nodes)
        self.reward_step_log = th.zeros(n_steps, n_networks)

        # Episode metrics
        self.episode_metrics = {
            "reward_steps": [],
            "reward_episode": [],
            "reward_episode_all_envs": [],
            "loss": [],
            "q_mean_steps": [],
            "q_min_steps": [],
            "q_max_steps": [],
            "q_mean": [],
            "q_min": [],
            "q_max": [],
            "q_learn": [],
        }
        self.ep_rewards = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        self.record_metrics = {
            "rewards": [],
            "loss": [],
            "q_mean": [],
            "q_min": [],
            "q_max": [],
        }

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
        self.q_step_log = th.zeros(self.n_steps, self.n_networks, self.n_nodes)
        self.reward_step_log = th.zeros(self.n_steps, self.n_networks)

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
        self.reward_step_log[step_number, :] = reward[:, 0]
        self.q_step_log[step_number, :, :] = q_step[:, :, 0].detach()

    def log_episode(self):
        """
        Store metrics'values at end of a single episode
        """

        # log the total reward obtained in the episode for each of the networks
        # self.episode_metrics['rewards'].append(self.curr_ep_reward)
        self.episode_metrics["reward_steps"].append(self.reward_step_log)
        self.episode_metrics["reward_episode"].append(
            th.squeeze(th.sum(self.reward_step_log, dim=0))
        )
        self.episode_metrics["reward_episode_all_envs"].append(
            th.mean(th.squeeze(th.sum(self.reward_step_log, dim=0))).item()
        )
        # log the loss value in the episode for each of the networks TODO: adapt to store when Learn method is called
        # self.episode_metrics['loss'].append(loss)

        # log the mean, min and max q value in the episode over all envs but FOR EACH STEP SEPARATELY
        # (apply mask to self.q_step_log ? we are mainly interested in the mean min and max of valid actions)
        self.episode_metrics["q_mean_steps"].append(th.mean(self.q_step_log, dim=0))
        self.episode_metrics["q_min_steps"].append(th.amin(self.q_step_log, dim=(1, 2)))
        self.episode_metrics["q_max_steps"].append(th.amax(self.q_step_log, dim=(1, 2)))
        # log the average of mean, min and max q value in the episode ACROSS ALL STEPS
        self.episode_metrics["q_mean"].append(
            th.mean(self.episode_metrics["q_mean_steps"][-1])
        )
        self.episode_metrics["q_min"].append(
            th.mean(self.episode_metrics["q_min_steps"][-1])
        )
        self.episode_metrics["q_max"].append(
            th.mean(self.episode_metrics["q_max_steps"][-1])
        )

        # reset values to zero
        self.init_episode()

    def log_episode_learn(self, q, loss):
        """
        Store metrics values at the call of Learn method TODO: finish

        Args:
            q (th.tensor): q values for each env
            loss (float): loss value
        """
        # log the q values from learn method
        self.episode_metrics["q_learn"].append(q)
        # log the loss value from learn method
        self.episode_metrics["loss"].append(loss)

    def record(self, episode, epsilon):
        """
        This method prints out during training the average trend of different metrics recorded for each episode.
        The avergae trend is calculated counting the last self.take_n_episodes completed
        TODO: finish for loss and minimum q value
        Args:
            episode (int): the current episode number
            epsilon (float): the current exploration rate for the greedy policy
        """
        mean_ep_reward = np.round(
            th.mean(self.episode_metrics["reward_episode"][-self.take_n_episodes:][0]),
            3,
        )
        mean_ep_loss = np.round(
            th.mean(self.episode_metrics["loss"][-self.take_n_episodes:][0]), 3
        )
        mean_ep_q_mean = np.round(
            th.mean(self.episode_metrics["q_mean"][-self.take_n_episodes:][0]), 3
        )
        mean_ep_q_min = np.round(
            np.mean(self.episode_metrics["q_min"][-self.take_n_episodes:][0]), 3
        )
        mean_ep_q_max = np.round(
            th.mean(self.episode_metrics["q_max"][-self.take_n_episodes:][0]), 3
        )
        self.record_metrics["reward_episode"].append(mean_ep_reward)
        # self.record_metrics['loss'].append(mean_ep_loss)
        self.record_metrics["q_mean"].append(mean_ep_q_mean)
        # self.record_metrics['q_min'].append(mean_ep_q_min)
        self.record_metrics["q_max"].append(mean_ep_q_max)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"We are at Episode {episode} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward over last {self.take_n_episodes} episodes: {mean_ep_reward} - "
            # f"Mean Loss over last {self.take_n_episodes} episodes: {mean_ep_loss} - "
            f"Mean Q Value over last {self.take_n_episodes} episodes: {mean_ep_q_mean} - "
            # f"Min Q Value over last {self.take_n_episodes} episodes: {mean_ep_q_min} - "
            f"Max Q Value over last {self.take_n_episodes} episodes: {mean_ep_q_max} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

    def save_metrics(self):
        """
        Saves moving average metrics as csv file
        """
        with open(os.path.join(self.save_dir, "metrics.pickle"), "wb") as handle:
            pickle.dump(self.episode_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_metric(self):
        plot_attr = {
            "reward_episode_all_envs": os.path.join(
                self.save_dir, "reward_all_envs_plot.pdf"
            ),
            "q_mean": os.path.join(self.save_dir, "reward_all_envs_mean_q.pdf"),
            "q_max": os.path.join(self.save_dir, "reward_all_envs_max_q.pdf"),
            "loss": os.path.join(self.save_dir, "loss_all_envs.pdf"),
        }

        for metric_name, metric_plot_path in plot_attr.items():
            plt.plot(self.episode_metrics[metric_name])
            plt.title(f"{metric_name}", fontsize=20)
            plt.xlabel("Episode", fontsize=17)
            # plt.ylim(-200, 400)
            plt.savefig(metric_plot_path, format="pdf", dpi=300)
            plt.clf()


#######################################
# TRAINING FUNCTION(S)
#######################################
def train_agent(config=None):
    """
    Train AI agent to solve reward networks (using wandb)

    Args:
        config (dict): dict containing parameter values, data paths and
                       flag to run or not run hyperparameter tuning
    """

    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        # ---------Loading of the networks---------------------
        print(
            f"Loading networks from file: {os.path.join(data_dir, config.data_name)}"
        )
        # Load networks to test
        with open(os.path.join(data_dir, config.data_name)) as json_file:
            train = json.load(json_file)
        test = train[:]
        print(f"Number of networks loaded: {len(test)}")

        # ---------Specify device (cpu or cuda)----------------
        if not th.cuda.is_available():
            DEVICE = th.device("cpu")
        else:
            DEVICE = th.device("cuda")

        # ---------Start analysis------------------------------
        # initialize environment(s)
        env = Reward_Network(test)

        # initialize Agent
        AI_agent = Agent(
            obs_dim=2,
            config=config,
            action_dim=env.action_space_idx.shape,
            save_dir=out_dir,
            device=DEVICE,
        )

        # initialize Memory buffer
        Mem = Memory(
            device=DEVICE, size=config.memory_size, n_rounds=config.n_rounds
        )

        # initialize Logger
        logger = MetricLogger(
            out_dir, config.n_networks, config.n_episodes, config.n_nodes
        )

        for e in range(config.n_episodes):
            print(f"----EPISODE {e + 1}---- \n")

            # reset env(s)
            env.reset()
            # obtain first observation of the env(s)
            obs = env.observe()

            for round_num in range(config.n_rounds):

                # Solve the reward networks!
                # while True:
                print("\n")
                print(f"ROUND/STEP {round_num} \n")

                # choose action to perform in environment(s)
                action, step_q_values = AI_agent.act(obs)
                # print(f'q values for step {round} -> \n {step_q_values[:,:,0].detach()}')
                # agent performs action
                # if we are in the last step we only need reward, else output also the next state
                if round_num != 7:
                    next_obs, reward = env.step(action, round_num)
                else:
                    reward = env.step(action, round_num)
                # remember transitions in memory
                Mem.store(round_num, reward=reward, action=action, **obs)
                if round_num != 7:
                    obs = next_obs
                # Logging (step)
                logger.log_step(reward, step_q_values, round_num)

                if env.is_done:
                    break

            # --END OF EPISODE--
            Mem.finish_episode()
            logger.log_episode()

            print("\n")
            print("\n")
            print(f"EPISODE {e + 1} MEMORY SAMPLE!")
            sample = Mem.sample(config.batch_size, device=DEVICE)
            if sample is not None:
                # for k,v in sample.items():
                #    print(k, v.shape)
                print("\n")
                print("\n")
                # Learning step
                q, loss = AI_agent.learn(sample)

                # Send the current training result back to Wandb
                wandb.log({"batch_loss": loss})
            else:
                print(f"Skip episode {e + 1}")
            print("\n")


def train_agent_local(config):
    """
    Train dqn agent locally, with a default config parameter dictionary
    """
    # ---------Loading of the networks---------------------
    print(f"Loading networks from file: {os.path.join(data_dir, config.data_name)}")
    # Load networks to test
    with open(os.path.join(data_dir, config.data_name)) as json_file:
        train = json.load(json_file)
    test = train[:]
    print(f"Number of networks loaded: {len(test)}")

    # ---------Specify device (cpu or cuda)----------------
    if not th.cuda.is_available():
        DEVICE = th.device("cpu")
    else:
        DEVICE = th.device("cuda")

    # ---------Start analysis------------------------------
    # initialize environment(s)
    env = Reward_Network(test)

    # initialize Agent
    AI_agent = Agent(
        obs_dim=2,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=out_dir,
        device=DEVICE,
    )

    # initialize Memory buffer
    Mem = Memory(device=DEVICE, size=config.memory_size, n_rounds=config.n_rounds)

    # initialize Logger
    logger = MetricLogger(
        out_dir, config.n_networks, config.n_episodes, config.n_nodes
    )

    for e in range(config.n_episodes):
        print(f"----EPISODE {e + 1}---- \n")

        # reset env(s)
        env.reset()
        # obtain first observation of the env(s)
        obs = env.observe()

        for round_num in tqdm(range(config.n_rounds)):

            # Solve the reward networks!
            # while True:
            print("\n")
            print(f"ROUND/STEP {round_num} \n")

            # choose action to perform in environment(s)
            action, step_q_values = AI_agent.act(obs)
            # print(f'q values for step {round} -> \n {step_q_values[:,:,0].detach()}')
            # agent performs action
            # if we are in the last step we only need reward, else output also the next state
            if round_num != 7:
                next_obs, reward = env.step(action, round_num)
            else:
                reward = env.step(action, round_num)
            # remember transitions in memory
            Mem.store(round_num, reward=reward, action=action, **obs)
            if round_num != 7:
                obs = next_obs
            # Logging (step)
            logger.log_step(reward, step_q_values, round_num)

            if env.is_done:
                break

        # --END OF EPISODE--
        Mem.finish_episode()
        logger.log_episode()

        print(f"EPISODE {e + 1} MEMORY SAMPLE!")
        sample = Mem.sample(config.batch_size, device=DEVICE)
        if sample is not None:
            print("\n")
            # Learning step
            q, loss = AI_agent.learn(sample)
            logger.log_episode_learn(q, loss)

        else:
            print(f"Skip episode {e + 1}")
        print("\n")

    # final logging
    logger.save_metrics()


#######################################
## MAIN
#######################################

if __name__ == "__main__":

    # --------Specify arguments--------------------------
    parser = argparse.ArgumentParser(
        description="DQN Argument Parser (Project: Reward Networks III)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--local", action="store_true", help="run locally and do not use wandb"
    )
    args = parser.parse_args()

    # --------Specify paths--------------------------
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    root_dir = os.sep.join(current_dir.split(os.sep)[:2])

    # Specify directories depending on system (local vs cluster)
    if root_dir == "/mnt":
        user_name = os.sep.join(current_dir.split(os.sep)[4:5])
        home_dir = f"/mnt/beegfs/home/{user_name}"
        project_dir = os.path.join(home_dir, "CHM", "reward_networks_III")
        code_dir = os.path.join(project_dir, "reward-network-iii-algorithm")
        data_dir = os.path.join(code_dir, "data")
        out_dir = os.path.join(project_dir, "results")

    elif root_dir == "/Users":
        # Specify directories (local)
        project_folder = os.getcwd()
        data_dir = os.path.join(project_folder, "data")
        out_dir = os.path.join(os.path.split(project_folder)[0], "data", "log")

    if args.local:
        # ---------Default Parameters for local testing -------------------
        config_default_dict = {
            "data_name": "train_viz_test.json",
            "n_episodes": 100,
            "n_networks": 954,
            "n_rounds": 8,
            "n_nodes": 10,
            "batch_size": 10,
            "memory_size": 50,
            "lr": 0.0001,
            "nn_hidden_size": 10,
            "exploration_rate_decay": 0.8,
            "nn_update_frequency": 200,
        }
        config_default = SimpleNamespace(**config_default_dict)
        # train agent!
        train_agent_local(config=config_default)

    else:

        # train agent!
        train_agent()
