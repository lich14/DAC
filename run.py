import gym
import mujoco_py
import torch
import numpy as np
from DAC_0 import to_np
import DAC_divide
import DAC
import DAC_cross
import argparse
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--method', type=str, default='DAC_0', help='method for training')
    parser.add_argument('--env', type=str, default='Walker2d-v2', help='environment')

    args = parser.parse_args()
    return args


class task():

    def __init__(self, config):

        self.env = gym.make(config.get('env_name'))
        config['action_dim'] = self.env.action_space.shape[0] * config.get('num_options')
        config['feature_dim'] = self.env.observation_space.shape[0]
        config['start_list'] = [i * self.env.action_space.shape[0] for i in range(config.get('num_options'))]
        config['end_list'] = [(i + 1) * self.env.action_space.shape[0] for i in range(config.get('num_options'))]

        self.lowtrans = np.dtype([
            ('s', np.float64, (config.get('feature_dim'),)),
            ('s_', np.float64, (config.get('feature_dim'),)),
            ('a', np.float64, (config.get('action_dim'),)),
            ('option', np.float64),
            ('option_', np.float64),
            ('r', np.float64),
            ('a_logp', np.float64),
            ('done', np.float64),
            ('pre_option', np.float64),
        ])

        self.hightrans = np.dtype([
            ('s', np.float64, (config.get('feature_dim'),)),
            ('s_', np.float64, (config.get('feature_dim'),)),
            ('option', np.float64),
            ('pre_option', np.float64),
            ('r', np.float64),
            ('option_logp', np.float64),
            ('done', np.float64),
        ])

        if config.get('method') == 'DAC_divide':
            self.nets = DAC_divide.DACAgent(config, self.lowtrans, self.hightrans, device)
        elif config.get('method') == 'DAC':
            self.nets = DAC.DACAgent(config, self.lowtrans, self.hightrans, device)
        else:
            self.nets = DAC_cross.DACAgent(config, self.lowtrans, self.hightrans, device)

        self.is_initial_states = 1
        self.prev_options = torch.tensor(100)
        self.states = self.env.reset()
        self.record = None
        self.reward = 0
        self.loop = 0
        self.innerstep = 0
        self.lowtrainloop = 0
        self.hightrainloop = 0

    def step(self):
        self.innerstep += 1
        self.states = torch.tensor(self.states, dtype=torch.double, device=device)
        highoutput = self.nets.highnet(self.states)
        options, options_logp = self.nets.sample_option(highoutput, self.prev_options, self.is_initial_states)

        lowoutput = self.nets.choose_action(self.states, options)
        input_action, actions, a_logp = lowoutput['input_action'], lowoutput['action'], lowoutput['a_logp']
        next_states, rewards, terminals, info = self.env.step(to_np(input_action))
        self.reward += rewards

        self.is_initial_states = torch.tensor(terminals).double()
        high_iftrain = self.nets.highmemory.store(
            (self.states.to('cpu'), next_states, options.to('cpu'), self.prev_options.to('cpu'), rewards,
             options_logp.to('cpu'), self.is_initial_states.to('cpu')))

        if self.record is not None:
            low_iftrain = self.nets.lowmemory.store(
                (self.record[0].to('cpu'), self.record[1], self.record[2].to('cpu'), self.record[3].to('cpu'),
                 options.to('cpu'), self.record[4], self.record[5].to('cpu').detach(), self.record[6],
                 self.record[7].to('cpu')))
            self.train(low_iftrain, high_iftrain)

        self.record = [
            self.states, next_states, actions, options, rewards, a_logp, self.is_initial_states, self.prev_options
        ]
        self.prev_options = options
        if terminals:
            self.prev_options = torch.tensor(100)
            writer.add_scalar('reward', self.reward, self.loop)
            writer.add_scalar('step', self.innerstep, self.loop)
            self.innerstep = 0
            self.reward = 0
            self.loop += 1
            self.states = self.env.reset()

        self.states = next_states

    def train(self, low_iftrain, high_iftrain):
        if low_iftrain is True:
            record = self.nets.lowtrain()
            writer.add_scalar('low/actor_loss', record['actionloss'], self.lowtrainloop)
            writer.add_scalar('low/critic_loss', record['valueloss'], self.lowtrainloop)
            writer.add_scalar('low/entropy', record['entropy'], self.lowtrainloop)
            self.lowtrainloop += 1
        if high_iftrain is True:
            record = self.nets.hightrain()
            writer.add_scalar('high/actor_loss', record['actionloss'], self.hightrainloop)
            writer.add_scalar('high/critic_loss', record['valueloss'], self.hightrainloop)
            writer.add_scalar('high/entropy', record['entropy'], self.hightrainloop)
            self.hightrainloop += 1


if __name__ == "__main__":
    args = get_args()

    config = {
        'env_name': args.env,
        'num_options': 4,
        'buffer_cap': 2048,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.95,
        'clip_param': 0.2,
        'entropy_para_high': 0.01,
        'entropy_para_low': 0,
        'low_lr': 0.0003,
        'high_lr': 0.0003,
        'totalstep': 4000000,
        'ppoepoch': 10,
        'with_repara': False,
        'hidden_dim': 64,
        'method': args.method,
        'soft_tau': 0.01,
        'max_grad_norm': 0.5,
    }
    name = 'env_name_' + config.get('env_name') + '_method_' + config.get('method')
    board_path = f"runs/{name}"
    writer = SummaryWriter(board_path)

    agent = task(config)
    terminal = False
    for _ in range(config.get('totalstep')):
        agent.step()

    writer.close()
