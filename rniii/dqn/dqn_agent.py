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
import torch
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
    n_nodes: int = 10
    learning_rate: float = 1.e-4
    batch_size: int = 8
    nn_hidden_layer_size: int = 10
    memory_size: int = 100
    exploration_rate_decay: float = 0.9
    nn_update_frequency: int = 100


# change string to compare os.environ with to enable ("enabled") or disable wandb
WANDB_ENABLED = os.environ.get("WANDB_MODE", "enabled") == "enabled"


def train():
    if WANDB_ENABLED:
        with wandb.init():
            config = Config(**wandb.config)
            train_agent(wandb.config)
    else:
        config = Config()
        train_agent(config)


def log(data, table=None):
    if WANDB_ENABLED:
        wandb.log(data)
        if table:
            table.add_data(v for k, v in data.items())
        wandb.log({"metrics_table": table})
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

        # specify parameters for rule based
        self.loss_counter = th.zeros(config.n_networks).int()
        self.n_losses = th.full((config.n_networks,), 2).int()

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

    def reset_loss_counter(self):
        """
        this method resets the loss counter at the end of each episode for the "take_loss" strategy
        """
        self.loss_counter = th.zeros(self.n_networks).int()

    def act_rule_based(self, obs, strategy: str):
        """
        Given a observation, choose an action (explore) according to a solving strategy with no
        DQN involved

        Args: obs (dict with values of th.tensor): observation from the env(s) comprising of one hot encoded
        reward+step counter+ big loss counter and a valid action mask

        Returns:
            action (th.tensor): node index representing next nodes to move to for all envs
            strategy (string): name of rule based strategy to use, one between ["myopic","take_loss","random"]
        """

        obs['mask'] = obs['mask'].to(self.device)
        obs["obs"] = obs["obs"].to(self.device)

        # get the reward indices for each of the 10 nodes within each environment
        current_possible_reward_idx = th.zeros((self.n_networks, 10)).type(th.long)
        splitted = th.split(th.nonzero(obs["obs"][:, :, :6]), 10)
        for i in range(len(splitted)):
            current_possible_reward_idx[i, :] = splitted[i][:, 2]

        if strategy == "myopic":
            action = th.unsqueeze(th.argmax(current_possible_reward_idx, dim=1), dim=-1)

        elif strategy == "take_loss":
            action = th.unsqueeze(th.argmax(current_possible_reward_idx, dim=1), dim=-1)

            if not th.equal(self.loss_counter, self.n_losses):  # that is, if there are still envs where loss counter <2

                loss_envs = (self.loss_counter != self.n_losses).nonzero()
                # print(f"environments where loss counter is still <2", loss_envs.shape)
                if loss_envs is not None:
                    envs_where_loss_present = th.unique(
                        (current_possible_reward_idx[loss_envs[:, 0], :] == 1).nonzero()[:, 0])
                    # print("environment with loss counter <2 where there is a loss", envs_where_loss_present.shape)
                    indices_selected_losses = th.multinomial(
                        (current_possible_reward_idx[loss_envs[:, 0], :] == 1)[envs_where_loss_present, :].float(), 1)
                    # print("indices of selected loss actions", indices_selected_losses.shape)
                    loss_actions = current_possible_reward_idx[loss_envs[:, 0], :].gather(1, indices_selected_losses)
                    # print("actual actions", loss_actions.shape)
                    action[envs_where_loss_present, 0] = indices_selected_losses[:, 0]

                    indices_loss_counter = loss_envs[:, 0][th.isin(loss_envs[:, 0], envs_where_loss_present)]
                    indices_loss_counter2 = th.arange(self.n_networks)[
                        th.isin(th.arange(self.n_networks), indices_loss_counter)]
                    # print("indices of envs to make +1 on loss counter", indices_loss_counter2.shape)
                    self.loss_counter[indices_loss_counter2] += 1

        elif strategy == "random":
            action= th.multinomial(obs["mask"].type(th.float), 1)

        return action[:, 0]

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
    # new!
    env_myopic = Reward_Network(networks)
    env_loss = Reward_Network(networks)
    env_random = Reward_Network(networks)

    # initialize Agent
    AI_agent = Agent(
        obs_dim=2,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=out_dir,
        device=DEVICE,
    )

    AI_agent_myopic = Agent(
        obs_dim=2,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=out_dir,
        device=DEVICE,
    )

    AI_agent_take_loss = Agent(
        obs_dim=2,
        config=config,
        action_dim=env.action_space_idx.shape,
        save_dir=out_dir,
        device=DEVICE,
    )

    AI_agent_random = Agent(
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

    # initialize wandb table
    log_table = wandb.Table(columns=[
        "episode",
        "avg_reward_all_envs",
        "avg_reward_rule_based_myopic_all_envs",
        "avg_reward_rule_based_2losses_all_envs",
        "avg_reward_rule_based_random_all_envs",
        "mean_q_all_envs",
        "max_q_all_envs",
        "batch_loss"
    ])

    for e in range(config.n_episodes):
        print(f"----EPISODE {e + 1}---- \n")

        # reset env(s)
        env.reset()
        # obtain first observation of the env(s)
        obs = env.observe()

        # new! double env to keep track of rule based steps
        env_myopic.reset()
        # obtain first observation of the env(s)
        obs_myopic = env_myopic.observe()
        env_random.reset()
        # obtain first observation of the env(s)
        obs_random = env_random.observe()
        env_loss.reset()
        # obtain first observation of the env(s)
        obs_loss = env_loss.observe()

        for round_num in range(config.n_rounds):

            # Solve the reward networks!
            # while True:
            print("\n")
            print(f"ROUND/STEP {round_num} \n")

            # choose action to perform in environment(s)
            action, step_q_values = AI_agent.act(obs)

            action_myopic = AI_agent_myopic.act_rule_based(obs_myopic, "myopic")
            action_take_loss = AI_agent_take_loss.act_rule_based(obs_loss, "take_loss")
            action_random = AI_agent_random.act_rule_based(obs_random, "random")

            # print(f'q values for step {round} -> \n {step_q_values[:,:,0].detach()}')
            # agent performs action
            # if we are in the last step we only need reward, else output also the next state
            if round_num != 7:
                next_obs, reward = env.step(action, round_num)

                next_obs_myopic, reward_myopic = env_myopic.step(action_myopic, round_num)
                next_obs_take_loss, reward_take_loss = env_loss.step(action_take_loss, round_num)
                next_obs_random, reward_random = env_random.step(action_random, round_num)
                reward2 = {"myopic": reward_myopic, "take_loss": reward_take_loss, "random": reward_random}

            else:
                reward = env.step(action, round_num)

                reward_myopic = env_myopic.step(action_myopic, round_num)
                reward_take_loss = env_loss.step(action_take_loss, round_num)
                reward_random = env_random.step(action_random, round_num)
                reward2 = {"myopic": reward_myopic, "take_loss": reward_take_loss, "random": reward_random}
                AI_agent_take_loss.reset_loss_counter()

            # remember transitions in memory
            Mem.store(round_num, reward=reward, action=action, **obs)
            if round_num != 7:
                obs = next_obs

                obs_myopic = next_obs_myopic
                obs_loss = next_obs_take_loss
                obs_random = next_obs_random

            # Logging (step)
            logger.log_step(reward, reward2, step_q_values, round_num)

            if env.is_done:
                break

        # --END OF EPISODE--
        Mem.finish_episode()
        logger.log_episode()
        # log({"episode": e + 1,
        #     "avg_reward_all_envs": logger.episode_metrics['reward_episode_all_envs'][-1],
        #     "mean_q_all_envs": logger.episode_metrics['q_mean'][-1],
        #     "max_q_all_envs": logger.episode_metrics['q_max'][-1]
        #     })

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
            # new! log everything , also in a table
            log({"episode": e + 1,
                 "avg_reward_all_envs": logger.episode_metrics['reward_episode_all_envs'][-1],
                 "avg_reward_rule_based_myopic_all_envs":
                     logger.episode_metrics['rule_based_reward_episode_all_envs']["myopic"][-1],
                 "avg_reward_rule_based_2losses_all_envs":
                     logger.episode_metrics['rule_based_reward_episode_all_envs']["take_loss"][-1],
                 "avg_reward_rule_based_random_all_envs":
                     logger.episode_metrics['rule_based_reward_episode_all_envs']["random"][-1],
                 "mean_q_all_envs": logger.episode_metrics['q_mean'][-1],
                 "max_q_all_envs": logger.episode_metrics['q_max'][-1],
                 "batch_loss": loss
                 })

            # log({"batch_loss": loss, "learn_episode": e + 1})
        else:
            # new! log everything , also in a table
            log({"episode": e + 1,
                 "avg_reward_all_envs": logger.episode_metrics['reward_episode_all_envs'][-1],
                 "avg_reward_rule_based_myopic_all_envs":
                     logger.episode_metrics['rule_based_reward_episode_all_envs']["myopic"][-1],
                 "avg_reward_rule_based_2losses_all_envs":
                     logger.episode_metrics['rule_based_reward_episode_all_envs']["take_loss"][-1],
                 "avg_reward_rule_based_random_all_envs":
                     logger.episode_metrics['rule_based_reward_episode_all_envs']["random"][-1],
                 "mean_q_all_envs": logger.episode_metrics['q_mean'][-1],
                 "max_q_all_envs": logger.episode_metrics['q_max'][-1],
                 "batch_loss": float("nan")
                 })

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
        print(os.getcwd())
        data_dir = os.path.join(os.getcwd(), "..", "..", "data")
        out_dir = os.path.join(data_dir, "log")

    # train
    train()
