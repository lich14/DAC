'''
coded by lch
consider single value function, no frozen
target value function and current value function are all low value function
'''

from model import lowPolicy, OptionNet, Store
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import numpy as np
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import torch.nn as nn


class DACAgent():

    def __init__(self, config, lowtran, hightran, device):
        self.config = config
        self.device = device
        self.lownet = lowPolicy(config.get('feature_dim'), config.get('action_dim'), config.get('num_options'),
                                config.get('hidden_dim')).double().to(self.device)
        self.highnet = OptionNet(config.get('num_options'), config.get('feature_dim'),
                                 config.get('hidden_dim')).double().to(self.device)
        self.lowmemory = Store(lowtran, config.get('buffer_cap'), config.get('batch_size'))
        self.highmemory = Store(hightran, config.get('buffer_cap'), config.get('batch_size'))
        self.lowoptimizition = torch.optim.Adam(self.lownet.parameters(), lr=config.get('low_lr'))
        self.highoptimizition = torch.optim.Adam(self.highnet.parameters(), lr=config.get('high_lr'))
        self.start_list = config.get('start_list')
        self.end_list = config.get('end_list')

    def sample_option(self, prediction, prev_option, is_intial_states):
        with torch.no_grad():
            q_option = prediction['q']
            mask = torch.zeros_like(q_option)
            beta = 1
            if is_intial_states == 0:
                mask[prev_option] = 1
                beta = prediction['beta'][prev_option]

            pi_hat_option = (1 - beta) * mask + beta * q_option

            dist = torch.distributions.Categorical(probs=q_option)
            options = dist.sample()
            options_logp = dist.log_prob(options)

            dist = torch.distributions.Categorical(probs=pi_hat_option)
            options_hat = dist.sample()
            options_hat_logp = dist.log_prob(options_hat)

            if is_intial_states:
                options = options
                options_logp = options_logp
            else:
                options = options_hat
                options_logp = options_hat_logp

        return options, options_logp

    def choose_action(self, state, option):
        state = state.to(self.device)
        low_action_total = self.lownet(state)
        start_index = self.start_list[option]
        end_index = self.end_list[option]

        action = low_action_total['action']
        a_logp = low_action_total['a_logp'][start_index:end_index]
        low_value = low_action_total['value'][option]
        input_action = action[start_index:end_index].to('cpu')
        input_action = input_action * 2 - 1

        return {
            'action': action,
            'a_logp': a_logp.sum(),
            'low_value': low_value,
            'input_action': input_action,
        }

    def lowtrain(self):
        buffer, buffer_capacity, batch_size = self.lowmemory.show()
        s = torch.tensor(buffer['s'], dtype=torch.double).to(self.device)
        option = torch.tensor(buffer['option'], dtype=torch.double).view(-1, 1).to(self.device)

        s_ = torch.tensor(buffer['s_'], dtype=torch.double).to(self.device)
        option_ = torch.tensor(buffer['option_'], dtype=torch.double).view(-1, 1).to(self.device)

        a = torch.tensor(buffer['a'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(buffer['a_logp'], dtype=torch.double).view(-1, 1).to(self.device)
        r = torch.tensor(buffer['r'], dtype=torch.double).view(-1, 1).to(self.device)
        done = torch.tensor(buffer['done'], dtype=torch.double).view(-1, 1).to(self.device)

        action_loss_record, value_loss_record, entropy_record, loop_record = 0, 0, 0, 0

        with torch.no_grad():
            value_next = self.lownet(s_)['value']
            option_change_next = torch.where(option_ > 5, torch.zeros_like(option_), option_)
            value_next_zeros = torch.gather(value_next, 1, option_change_next.long())
            value_next = torch.where(option_ > 5,
                                     value_next.sum(dim=1, keepdim=True) / self.config.get('num_options'),
                                     value_next_zeros)

            value_now = self.lownet(s)['value']
            option_change_now = torch.where(option > 5, torch.zeros_like(option), option)
            value_now_zeros = torch.gather(value_now, 1, option_change_now.long())
            value_now = torch.where(option > 5,
                                    value_now.sum(dim=1, keepdim=True) / self.config.get('num_options'),
                                    value_now_zeros)

            delta = r + (1 - done) * self.config.get('gamma') * value_next - value_now
            adv = torch.zeros_like(delta)
            adv[-1] = delta[-1]
            # GAE
            for i in reversed(range(buffer_capacity - 1)):
                adv[i] = delta[i] + self.config.get('tau') * (1 - done[i]) * adv[i + 1]

            target_v = value_now + adv
            adv = (adv - adv.mean()) / (adv.std() + np.finfo(np.float).eps)  # Normalize advantage

        for _ in range(self.config.get('ppoepoch')):
            for index in BatchSampler(SubsetRandomSampler(range(buffer_capacity)), batch_size, False):
                mean, logstd = self.lownet(s[index])['mean'], self.lownet(s[index])['logstd']
                std = logstd.exp()
                dist = Normal(mean, std)
                a_logp = dist.log_prob(a[index])
                option_short = option[index]
                mask = torch.zeros_like(a_logp).double()
                index_list = [torch.where(option_short == i)[0] for i in range(self.config.get('num_options'))]
                input_list = torch.zeros(self.config.get('num_options'), self.config.get('action_dim'))
                start_list = self.config.get('start_list')
                end_list = self.config.get('end_list')
                for i in range(self.config.get('num_options')):
                    input_list[i][start_list[i]:end_list[i]] = 1

                for i in range(self.config.get('num_options')):
                    if torch.tensor(index_list[i].shape) != 0:
                        mask[index_list[i]] = torch.ones(torch.tensor(index_list[i].shape),
                                                         self.config.get('action_dim')).double().to(
                                                             self.device) * input_list[i].double().to(self.device)

                a_logp = a_logp * mask
                a_p_1 = a_logp.sum(dim=1, keepdim=True)

                ratio = torch.exp((a_p_1 - old_a_logp[index]))
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.config.get('clip_param'),
                                    1.0 + self.config.get('clip_param')) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy() * mask

                value_now = self.lownet(s[index])['value']
                option_change_now = torch.where(option[index] > 5, torch.zeros_like(option[index]), option[index])
                value_now_zeros = torch.gather(value_now, 1, option_change_now.long())
                value_now = torch.where(option[index] > 5,
                                        value_now.sum(dim=1, keepdim=True) / self.config.get('num_options'),
                                        value_now_zeros)

                value_loss = F.smooth_l1_loss(value_now, target_v[index])
                self.lowoptimizition.zero_grad()
                loss = action_loss + value_loss - self.config.get('entropy_para_low') * entropy.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.lownet.parameters(), self.config.get('max_grad_norm'))
                self.lowoptimizition.step()

                action_loss_record += action_loss.cpu().detach()
                value_loss_record += value_loss.cpu().detach()
                entropy_record += entropy.mean().cpu().detach()
                loop_record += 1

        return {
            'actionloss': action_loss_record / loop_record,
            'valueloss': value_loss_record / loop_record,
            'entropy': entropy_record / loop_record,
        }

    def hightrain(self):
        buffer, buffer_capacity, batch_size = self.highmemory.show()
        s = torch.tensor(buffer['s'], dtype=torch.double).to(self.device)
        pre_option = torch.tensor(buffer['pre_option'], dtype=torch.double).view(-1, 1).to(self.device)

        s_ = torch.tensor(buffer['s_'], dtype=torch.double).to(self.device)
        option = torch.tensor(buffer['option'], dtype=torch.double).view(-1, 1).to(self.device)

        option_logp = torch.tensor(buffer['option_logp'], dtype=torch.double).view(-1, 1).to(self.device)
        r = torch.tensor(buffer['r'], dtype=torch.double).view(-1, 1).to(self.device)
        done = torch.tensor(buffer['done'], dtype=torch.double).view(-1, 1).to(self.device)
        action_loss_record, value_loss_record, entropy_record, loop_record = 0, 0, 0, 0

        with torch.no_grad():
            value_low_now = self.lownet(s)['value']
            value_low_next = self.lownet(s_)['value']

            q_option_now = self.highnet(s)['q']
            q_option_next = self.highnet(s_)['q']

            beta_now = self.highnet(s)['beta']
            beta_next = self.highnet(s_)['beta']

            option_now = self.sample_option_multi(q_option_now, beta_now, pre_option)
            option_next = self.sample_option_multi(q_option_next, beta_next, option)

            value_high_now = value_low_now * option_now
            value_high_next = value_low_next * option_next
            value_high_now = value_high_now.sum(dim=1, keepdim=True)
            value_high_next = value_high_next.sum(dim=1, keepdim=True)

            delta = r + (1 - done) * self.config.get('gamma') * value_high_next - value_high_now
            adv = torch.zeros_like(delta)
            adv[-1] = delta[-1]
            # GAE
            for i in reversed(range(buffer_capacity - 1)):
                adv[i] = delta[i] + self.config.get('tau') * (1 - done[i]) * adv[i + 1]

            target_v = value_high_now + adv
            adv = (adv - adv.mean()) / (adv.std() + np.finfo(np.float).eps)  # Normalize advantage

        for _ in range(self.config.get('ppoepoch')):
            for index in BatchSampler(SubsetRandomSampler(range(buffer_capacity)), batch_size, False):
                q_short, beta_short = self.highnet(s[index])['q'], self.highnet(s[index])['beta']
                pre_option_short = pre_option[index]
                pi_hat_option = self.sample_option_multi(q_short, beta_short, pre_option_short)
                pi_hat_p = torch.gather(pi_hat_option, 1, option[index].long())
                ratio = pi_hat_p / torch.exp(option_logp[index])
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.config.get('clip_param'),
                                    1.0 + self.config.get('clip_param')) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                m = Categorical(pi_hat_option)
                entropy = m.entropy()

                value_low_now_short = self.lownet(s[index])['value']
                value_high_now_short = value_low_now_short * pre_option_short
                value_high_now_short = value_high_now_short.sum(dim=1, keepdim=True)

                value_loss = F.smooth_l1_loss(value_high_now_short, target_v[index])
                self.highoptimizition.zero_grad()
                loss = action_loss + value_loss - self.config.get('entropy_para_high') * entropy.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.highnet.parameters(), self.config.get('max_grad_norm'))
                self.highoptimizition.step()
                action_loss_record += action_loss.cpu().detach()
                value_loss_record += value_loss.cpu().detach()
                entropy_record += entropy.mean().cpu().detach()
                loop_record += 1

        return {
            'actionloss': action_loss_record / loop_record,
            'valueloss': value_loss_record / loop_record,
            'entropy': entropy_record / loop_record,
        }

    def sample_option_multi(self, q, beta, pre_option):
        index_init = torch.where(pre_option > 80)[0]
        index_run = torch.where(pre_option < 81)[0]
        mask = torch.zeros_like(q)
        add_ones = torch.ones_like(q)
        mask[index_run, :] = mask[index_run, :].scatter_(1, pre_option[index_run, :].long(), add_ones[index_run, :])
        beta_change = beta * mask
        beta_change = beta_change.sum(dim=1, keepdim=True)
        beta_change[index_init, :] = 1

        pi_hat_option = (1 - beta_change) * mask + beta_change * q
        pi_hat_option[index_init, :] = q[index_init, :]

        return pi_hat_option


class LinearSchedule:

    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


def to_np(t):
    return t.cpu().detach().numpy()
