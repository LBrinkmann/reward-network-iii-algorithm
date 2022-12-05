import torch as th


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
