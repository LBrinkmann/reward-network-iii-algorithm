# This file specifies the Deep Q Learning AI agent model to solve a Reward Network DAG
# See also: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# and: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
#
#
# Author @ Sara Bonati
# Project: Reward Network III
# Center for Humans and Machines, MPIB Berlin
###############################################

import datetime
import json
# filesystem and log file specific imports
import os
import pickle
import time

import matplotlib.pyplot as plt
# import modules
import numpy as np
# import Pytorch + hyperparameter tuning modules
import torch as th
import wandb
from pydantic import BaseModel

# import the custom reward network environment class
from environment_vect import Reward_Network
from memory import Memory
from nn import DQN
from logger import MetricLogger


class Config(BaseModel):
    data_name: str = "train_viz_test.json"
    n_episodes: int = 50
    n_networks: int = 954
    n_rounds: int = 8
    learning_rate: float = 1.e-4
    batch_size: int = 8
    nn_hidden_layer_size: int = 10
    memory_size: int = 100
    exploration_rate_decay: float = 0.9
    nn_update_frequency: int = 100


WANDB_ENABLED = os.environ.get("WANDB_MODE", "enabled") == "enabled"


def train():
    if WANDB_ENABLED:
        with wandb.init():
            config = Config(**wandb.config)
            train_agent(wandb.config)
    else:
        config = Config()
        train_agent(config)


def log(data):
    if WANDB_ENABLED:
        wandb.log(data)
    else:
        print(" | ".join(f"{k}: {v}" for k, v in data.items()))


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
        save_dir (str): path to folder where to save model checkpoints into
        device: torch device (cpu or cuda)
        """

        # assert tests
        assert os.path.exists(
            save_dir
        ), f"{save_dir} is not a valid path (does not exist)"

        # specify environment parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_networks = config.n_networks

        # torch device
        self.device = device

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
            config.nn_hidden_layer_size,
        )
        # one q value for each action
        output_size = (
            config.n_networks,
            config.n_nodes,
            1,
        )
        self.policy_net = DQN(input_size, output_size, hidden_size)
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = DQN(input_size, output_size, hidden_size)
        self.target_net = self.target_net.to(self.device)

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
        self.lr = config.learning_rate
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = th.nn.SmoothL1Loss(reduction="none")

        # specify output directory
        self.save_dir = save_dir

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

        obs['mask'] = obs['mask'].to(self.device)
        obs["obs"] = obs["obs"].to(self.device)

        # EXPLORE (select random action from the action space)
        random_actions = th.multinomial(obs["mask"].type(th.float), 1)
        # print(f'random actions {th.squeeze(random_actions,dim=-1)}')

        # EXPLOIT (select greedy action)
        # return Q values for each action in the action space A | S=s
        action_q_values = self.policy_net(obs["obs"])
        # apply masking to obtain Q values for each VALID action (invalid actions set to very low Q value)
        action_q_values = self.apply_mask(action_q_values, obs["mask"])
        # select action with highest Q value
        greedy_actions = th.argmax(action_q_values, dim=1).to(self.device)
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
        state = state.to(self.device)
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

        state = state.to(self.device)
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
        As we sample inputs from memory, we compute loss using TD estimate and TD target,
        then backpropagate this loss down Q_online to update its parameters θ_online

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
    # with wandb.init(config=config):
    #    config = wandb.config

    # ---------Loading of the networks---------------------
    print(
        f"Loading networks from file: {os.path.join(data_dir, config.data_name)}"
    )
    # Load networks to test
    with open(os.path.join(data_dir, config.data_name)) as json_file:
        networks = json.load(json_file)
    print(f"Number of networks loaded: {len(networks)}")

    # ---------Specify device (cpu or cuda)----------------
    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

    # ---------Start analysis------------------------------
    # initialize environment(s)
    env = Reward_Network(networks)

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
        log({"avg_reward_all_envs": logger.episode_metrics['reward_episode_all_envs'][-1],
             "mean_q_all_envs": logger.episode_metrics['mean_q'][-1],
             "max_q_all_envs": logger.episode_metrics['max_q'][-1]})

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

            # Send the current training result back to Wandb (if wandb enabled),
            # else print metrics
            log({"batch_loss": loss})
        else:
            print(f"Skip episode {e + 1}")
        print("\n")


if __name__ == "__main__":

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
        project_folder = os.path.split(os.getcwd())[0]
        data_dir = os.path.join(project_folder, "data")
        out_dir = os.path.join(data_dir, "log")

    # train
    train()
