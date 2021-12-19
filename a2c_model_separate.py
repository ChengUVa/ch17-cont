import ptan
import numpy as np
import torch
import torch.nn as nn

HID_SIZE = 200

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(), 
            #nn.Linear(HID_SIZE, HID_SIZE),
            #nn.ReLU(), 
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(), # in range (-1, 1)
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_size, val_scale=1.0):
        super(ModelCritic, self).__init__()
        self.val_scale = val_scale

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            #nn.Linear(HID_SIZE, HID_SIZE),
            #nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x) * self.val_scale


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, act_net, device="cpu"):
        self.net = act_net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        #rnd = np.random.normal(size=logstd.shape)
        #actions = mu + np.exp(logstd) * rnd
        actions = np.random.normal(mu, np.exp(logstd))
        actions = np.clip(actions, -1, 1)
        return actions, agent_states

