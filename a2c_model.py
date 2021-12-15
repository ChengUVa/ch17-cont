import ptan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 200

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.base = nn.Sequential(nn.Linear(obs_size, HID_SIZE), nn.ReLU(),)

        # policy head
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size), nn.Tanh(),  # in range (-1,1)
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size), nn.Softplus(),  # smoothed RELU, positive
        )

        # value function head
        self.value = nn.Linear(HID_SIZE, 1)  # no activation

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        """convert observation to actins"""
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)  # sample action from normal dist.
        actions = np.clip(actions, -1, 1)  # action value between -1 and 1
        # agent_states is not used for sampling actions here
        return actions, agent_states
