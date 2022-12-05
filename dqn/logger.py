import datetime
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th


class MetricLogger:
    def __init__(self, save_dir: str, n_networks: int, n_episodes: int, n_nodes: int, n_steps=8):
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
