import torch.nn as nn
import torch.nn.functional as F


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
