import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, pred_state_size, state_size, action_size, seed, fcs1_units=600,
                 fc2_units=300):  # , fc3_units=200):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(pred_state_size + state_size, fcs1_units)
        self.bn0 = nn.LayerNorm(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.do1 = nn.Dropout(.2)
        self.fc5 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, prev_state, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        pred_state = state - prev_state
        if len(state.shape) == 2:
            dual_states = torch.cat((pred_state, state), dim=1)
        else:
            dual_states = torch.cat((pred_state, state))

        xs = self.fcs1(dual_states)
        xs = self.bn0(xs)
        xs = F.relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc5(x)

        return x
