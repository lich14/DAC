import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class lowPolicy(nn.Module):

    def __init__(self, feature_dim, action_dim, num_options, hidden_dim=64):
        super(lowPolicy, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_options = num_options

        self.body_actor = layer_init(nn.Linear(self.feature_dim, self.hidden_dim))
        self.body_critic = layer_init(nn.Linear(self.feature_dim, self.hidden_dim))

        self.a = layer_init(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.mean = layer_init(nn.Linear(self.hidden_dim, self.action_dim))
        self.logstd = layer_init(nn.Linear(self.hidden_dim, self.action_dim))

        self.v1 = layer_init(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.v2 = layer_init(nn.Linear(self.hidden_dim, self.num_options))

    def forward(self, x):
        body_actor = F.tanh(self.body_actor(x))
        y = F.tanh(self.a(body_actor))
        mean = self.mean(y)
        logstd = self.logstd(y)
        std = logstd.exp()

        dist = Normal(mean, std)
        action = dist.sample()
        a_logp = dist.log_prob(action)
        entropy = dist.entropy()

        body_critic = F.relu(self.body_critic(x))
        z = F.relu(self.v1(body_critic))
        value = self.v2(z)

        return {
            'action': action,
            'a_logp': a_logp,
            'value': value,
            'entropy': entropy,
            'mean': mean,
            'logstd': logstd,
        }


def layer_init(layer, w_scale=0.1):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class OptionNet(nn.Module):

    def __init__(self, num_options, feature_dim, hidden_dim=64):
        super(OptionNet, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_options = num_options
        self.fc_body1 = layer_init(nn.Linear(self.feature_dim, self.hidden_dim))
        self.fc_body2 = layer_init(nn.Linear(self.feature_dim, self.hidden_dim))

        self.fc_beta = layer_init(nn.Linear(self.hidden_dim, self.num_options))
        self.fc_option = layer_init(nn.Linear(self.hidden_dim, self.num_options))
        self.fc_value = layer_init(nn.Linear(self.hidden_dim, self.num_options))

    def forward(self, x):
        body1 = F.tanh(self.fc_body1(x))
        beta = F.sigmoid(self.fc_beta(body1))
        q = F.softmax(self.fc_option(body1))

        body2 = F.relu(self.fc_body2(x))
        value = self.fc_value(body2)
        return {
            'q': q,
            'beta': beta,
            'value': value,
        }


class Store():

    def __init__(self, transition, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.data = transition

    def store(self, add):
        self.buffer[self.counter] = add
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def empty(self):
        self.buffer = np.empty(self.buffer_capacity, dtype=self.data)

    def show(self):
        return self.buffer, self.buffer_capacity, self.batch_size
