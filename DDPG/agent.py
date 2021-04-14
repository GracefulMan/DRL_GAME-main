import os
from copy import deepcopy  # deepcopy target_network

import torch
import numpy as np
import numpy.random as rd
from net import QNet, QNetDuel, QNetTwin, QNetTwinDuel
from net import Actor, ActorRNN
from net import Critic, CriticAdv, CriticTwin
from net import InterDPG
from typing import Tuple


class AgentBase:
    def __init__(self):
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None

    def init(self, net_dim, state_dim, action_dim):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :int net_dim: the dimension of networks (the width of neural networks)
        :int state_dim: the dimension of state (the number of state vector)
        :int action_dim: the dimension of action (the number of discrete action)
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def select_actions(self, states) -> np.ndarray:
        """Select actions for exploration

        :array states: (state, ) or (state, state, ...) or state.shape==(n, *state_dim)
        :return array action: action.shape==(-1, action_dim), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device).detach_()
        actions = self.act(states)
        return actions.cpu().numpy()  # -1 < action < +1

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale,  0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        :buffer: Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        :int batch_size: sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


class AgentBaseRNN:
    def __init__(self):
        self.learning_rate = 1e-4
        self.soft_update_tau = 2 ** -8  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None

    def init(self, net_dim, state_dim, action_dim):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :int net_dim: the dimension of networks (the width of neural networks)
        :int state_dim: the dimension of state (the number of state vector)
        :int action_dim: the dimension of action (the number of discrete action)
        """

    def select_action(self, state, hidden) -> Tuple[np.ndarray, np.ndarray]:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action, hidden = self.act(states, hidden)

        return action[0].cpu().numpy(), hidden.cpu().numpy()

    def select_actions(self, states, hidden) -> Tuple[np.ndarray, Tuple]:
        """Select actions for exploration

        :array states: (state, ) or (state, state, ...) or state.shape==(n, *state_dim)
        :return array action: action.shape==(-1, action_dim), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device).detach_()
        actions = self.act(states, hidden)
        return actions.cpu().numpy(), hidden  # -1 < action < +1

    def init_hidden(self, batch_size):
        pass


    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        hidden = self.init_hidden(batch_size=1)
        for _ in range(target_step):
            action, hidden = self.select_action(self.state, hidden)
            h, c = hidden
            h = h.cpu()
            c = c.cpu()
            hidden = (h, c)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale,  0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other, hidden)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        :buffer: Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        :int batch_size: sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


'''Value-based Methods (DQN variances)'''


class AgentDQN(AgentBase):
    def __init__(self):
        super(AgentDQN, self).__init__()
        self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
        self.action_dim = None  # chose discrete action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri  # to keep the same from Actor-Critic framework

        self.criterion = torch.torch.nn.MSELoss()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
            action = self.act(states)[0]
            a_int = action.argmax(dim=0).cpu().numpy()
        return a_int

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        next_q = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)  # next_state
                next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            q_eval = self.cri(state).gather(1, action.type(torch.long))
            obj_critic = self.criterion(q_eval, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return next_q.mean().item(), obj_critic.item()


class AgentDuelingDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

    def init(self, net_dim, state_dim, action_dim):
        """Contribution of Dueling DQN

        Advantage function --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetDuel(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.MSELoss()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.softmax = torch.nn.Softmax(dim=1)

    def init(self, net_dim, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = self.softmax(actions)[0]
            a_prob = action.detach().cpu().numpy()  # choose action according to Q value
            a_int = rd.choice(self.action_dim, p=a_prob)
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).cpu().numpy()
        return a_int

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """Contribution of DDQN (Double DQN)

        Twin Q-Network. Use min(q1, q2) to reduce over-estimation.
        """
        buffer.update_now_len_before_sample()

        next_q = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s))
                next_q = next_q.max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            act_int = action.type(torch.long)
            q1, q2 = [qs.gather(1, act_int) for qs in self.act.get_q1_q2(state)]
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return next_q.mean().item(), obj_critic.item() / 2


class AgentD3QN(AgentDoubleDQN):  # D3QN: Dueling Double DQN
    def __init__(self):
        super().__init__()

    def init(self, net_dim, state_dim, action_dim):
        """Contribution of D3QN (Dueling Double DQN)

        There are not contribution of D3QN.
        Obviously, DoubleDQN is compatible with DuelingDQN.
        Any beginner can come up with this idea (D3QN) independently.
        """
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cri = QNetTwinDuel(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act = self.cri

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)


'''Actor-Critic Methods (Policy Gradient)'''


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.ou_explore_noise = 0.3  # explore noise of action
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim):
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.ou_explore_noise)
        # I don't recommend to use OU-Noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.MSELoss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def select_action(self, state) ->Tuple[np.ndarray, np.ndarray]:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0].cpu().numpy()
        return (action + self.ou_noise()).clip(-1, 1)


    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = self.cri_target(next_s, self.act_target(next_s))
                q_label = reward + mask * next_q
            q_value = self.cri(state, action)
            obj_critic = self.criterion(q_value, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_actor.item(), obj_critic.item()


def get_full_feature(feature, hidden):
    h, c = hidden
    h, c = h.squeeze(), c.squeeze()
    new_feature = torch.cat([feature, h, c], dim=1)
    return new_feature


class AgentDDPGWithRNN(AgentBaseRNN):
    def __init__(self):
        super(AgentDDPGWithRNN, self).__init__()
        self.ou_explore_noise = 0.3  # explore noise of action
        self.ou_noise = None

    def init(self, net_dim, state_dim, action_dim):
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.ou_explore_noise)
        # I don't recommend to use OU-Noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act = ActorRNN(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = Critic(net_dim, net_dim * 2 + state_dim, net_dim * 2 + action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.MSELoss()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

    def init_hidden(self, batch_size):
        return self.act.init_hidden(batch_size)

    def select_action(self, state, hidden) ->Tuple[np.ndarray, np.ndarray]:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action, hidden = self.act(states, hidden)
        action = action[0].cpu().numpy()
        return (action + self.ou_noise()).clip(-1, 1), hidden


    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        def add_dim(hidden):
            return hidden.unsqueeze(dim=0)
        buffer.update_now_len_before_sample()
        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s, hidden_1, hidden_next_1, hidden_2, hidden_next_2 = buffer.sample_batch(batch_size)
                hidden_1, hidden_2, hidden_next_1, hidden_next_2 = add_dim(hidden_1), add_dim(hidden_2), add_dim(hidden_next_1), add_dim(hidden_next_2)
                hidden = (hidden_1, hidden_2)
                hidden_next = (hidden_next_1, hidden_next_2)
                newstate = get_full_feature(state, hidden)
                newaction = get_full_feature(action, hidden)
                newstate_next = get_full_feature(next_s, hidden_next)
                tmp_action, tmp_hidden = self.act_target(next_s, hidden_next)
                newaction_next = get_full_feature(tmp_action, tmp_hidden)
                next_q = self.cri_target(newstate_next,  newaction_next)
                q_label = reward + mask * next_q
            q_value = self.cri(newstate, newaction)
            obj_critic = self.criterion(q_value, q_label)

            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            #q_value_pg = self.act(state, hidden)[0]  # policy gradient
            tmp_action, tmp_hidden = self.act(state, hidden)
            newaction = get_full_feature(tmp_action, tmp_hidden)
            obj_actor = -self.cri_target(newstate, newaction).mean()
            self.act_optimizer.zero_grad()
            obj_actor.backward()
            self.act_optimizer.step()
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_actor.item(), obj_critic.item()


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy, if_gpu):
        """Experience Replay Buffer
        # 采用的是当满了的时候清除最早记录的方法。如果用cpu实现，可以用quene感觉。

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :bool if_on_policy: on-policy or off-policy
        :bool if_gpu: create buffer space on CPU RAM or GPU
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_all(
        self.if_on_policy = if_on_policy
        self.if_gpu = if_gpu

        if if_on_policy: # SARSA的更新方式
            self.if_gpu = False
            other_dim = 1 + 1 + action_dim * 2
        else:
            other_dim = 1 + 1 + action_dim

        if self.if_gpu:
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        else:
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)

    def append_buffer(self, state, other):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, device=self.device)
            other = torch.as_tensor(other, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            other = torch.as_tensor(other, dtype=torch.float32, device=self.device)

        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                # 表示buffer没填满但是又不够size这么多，我就把剩下的用新的前面的填满。
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device) if self.if_gpu \
            else rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def sample_all(self) -> tuple:
        """sample all the data in ReplayBuffer (for on-policy)

        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor noise:  noise.shape ==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        """
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],
                all_other[:, 1],
                all_other[:, 2:2 + self.action_dim],
                all_other[:, 2 + self.action_dim:],
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """we empty the buffer by set now_len=0. On-policy need to empty buffer before exploration
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        """print the state norm information: state_avg, state_std

        We don't suggest to use running stat state.
        We directly do normalization on state using the historical avg and std
        eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
        neg_avg = -states.mean()
        div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

        :array neg_avg: neg_avg.shape=(state_dim)
        :array div_std: div_std.shape=(state_dim)
        """
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| print_norm: state_avg, state_fix_std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")


class ReplayBufferRNN:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy, if_gpu, hidden_dim):
        """Experience Replay Buffer
        # 采用的是当满了的时候清除最早记录的方法。如果用cpu实现，可以用quene感觉。

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :bool if_on_policy: on-policy or off-policy
        :bool if_gpu: create buffer space on CPU RAM or GPU
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_all(
        self.if_on_policy = if_on_policy
        self.if_gpu = if_gpu

        if if_on_policy: # SARSA的更新方式
            self.if_gpu = False
            other_dim = 1 + 1 + action_dim * 2
        else:
            other_dim = 1 + 1 + action_dim

        if self.if_gpu:
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
            self.buf_hidden_1 = torch.empty((max_len, hidden_dim), dtype=torch.float32, device=self.device)
            self.buf_hidden_2 = torch.empty((max_len, hidden_dim), dtype=torch.float32, device=self.device)

        else:
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
            self.buf_hidden_1 = np.empty((max_len, hidden_dim), dtype=np.float32)
            self.buf_hidden_2 = np.empty((max_len, hidden_dim), dtype=np.float32)


    def append_buffer(self, state, other, hidden):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, device=self.device)
            other = torch.as_tensor(other, device=self.device)
            hidden = torch.as_tensor(hidden, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other
        self.buf_hidden_1[self.next_idx], self.buf_hidden_2[self.next_idx] = hidden

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other, hidden):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            other = torch.as_tensor(other, dtype=torch.float32, device=self.device)
            h1, h2 = hidden
            hidden_1 = torch.as_tensor(h1, dtype=torch.float32, device=self.device)
            hidden_2 = torch.as_tensor(h2, dtype=torch.float32, device=self.device)


        size = len(other)
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                # 表示buffer没填满但是又不够size这么多，我就把剩下的用新的前面的填满。
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
                self.buf_hidden_1[self.next_idx: self.max_len] = hidden_1[:self.max_len - self.next_idx]
                self.buf_hidden_2[self.next_idx: self.max_len] = hidden_2[:self.max_len - self.next_idx]

            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
            self.buf_hidden_1[0:next_idx] = hidden_1[-next_idx:]
            self.buf_hidden_2[0:next_idx] = hidden_2[-next_idx:]

        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
            self.buf_hidden_1[self.next_idx:next_idx] = hidden_1
            self.buf_hidden_2[self.next_idx:next_idx] = hidden_2

        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device) if self.if_gpu \
            else rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1],
                self.buf_hidden_1[indices],
                self.buf_hidden_1[indices + 1],
                self.buf_hidden_2[indices],
                self.buf_hidden_2[indices + 1]
                )

    def sample_all(self) -> tuple:
        """sample all the data in ReplayBuffer (for on-policy)

        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor noise:  noise.shape ==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        """
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0],
                all_other[:, 1],
                all_other[:, 2:2 + self.action_dim],
                all_other[:, 2 + self.action_dim:],
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device),
                torch.as_tensor(self.buf_hidden_1[:self.now_len], device=self.device),
                torch.as_tensor(self.buf_hidden_2[:self.now_len], device=self.device)

                )

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """we empty the buffer by set now_len=0. On-policy need to empty buffer before exploration
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        """print the state norm information: state_avg, state_std

        We don't suggest to use running stat state.
        We directly do normalization on state using the historical avg and std
        eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
        neg_avg = -states.mean()
        div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

        :array neg_avg: neg_avg.shape=(state_dim)
        :array div_std: div_std.shape=(state_dim)
        """
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| print_norm: state_avg, state_fix_std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")



class ReplayBufferMP:
    def __init__(self, max_len, state_dim, action_dim, rollout_num, if_on_policy, if_gpu):
        """Experience Replay Buffer for Multiple Processing

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :int rollout_num: the rollout workers number
        :bool if_on_policy: on-policy or off-policy
        :bool if_gpu: create buffer space on CPU RAM or GPU
        """
        self.now_len = 0
        self.max_len = max_len
        self.rollout_num = rollout_num

        self.if_gpu = if_gpu
        if if_on_policy:
            self.if_gpu = False

        _max_len = max_len // rollout_num
        self.buffers = [ReplayBuffer(_max_len, state_dim, action_dim, if_on_policy, if_gpu=if_gpu)
                        for _ in range(rollout_num)]

    def extend_buffer(self, state, other, i):
        self.buffers[i].extend_buffer(state, other)

    def sample_batch(self, batch_size) -> tuple:
        rd_batch_sizes = rd.rand(self.rollout_num)
        rd_batch_sizes = (rd_batch_sizes * (batch_size / rd_batch_sizes.sum())).astype(np.int)
        # 这个rd_batch_size和batch_size的大小一致，从不同的buffer按概率采样。
        l__r_m_a_s_ns = [self.buffers[i].sample_batch(rd_batch_sizes[i])
                         for i in range(self.rollout_num) if rd_batch_sizes[i] > 2]
        return (torch.cat([item[0] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[1] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[2] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[3] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[4] for item in l__r_m_a_s_ns], dim=0))

    def sample_all(self) -> tuple:
        l__r_m_a_n_s = [self.buffers[i].sample_all()
                        for i in range(self.rollout_num)]
        return (torch.cat([item[0] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[1] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[2] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[3] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[4] for item in l__r_m_a_n_s], dim=0))

    def update_now_len_before_sample(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len_before_sample()
            self.now_len += buffer.now_len

    def empty_buffer_before_explore(self):
        for buffer in self.buffers:
            buffer.empty_buffer_before_explore()

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)

class ReplayBufferMPRNN:
    def __init__(self, max_len, state_dim, action_dim, rollout_num, if_on_policy, if_gpu, hidden_dim):
        """Experience Replay Buffer for Multiple Processing

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :int rollout_num: the rollout workers number
        :bool if_on_policy: on-policy or off-policy
        :bool if_gpu: create buffer space on CPU RAM or GPU
        """
        self.now_len = 0
        self.max_len = max_len
        self.rollout_num = rollout_num

        self.if_gpu = if_gpu
        if if_on_policy:
            self.if_gpu = False

        _max_len = max_len // rollout_num
        self.buffers = [ReplayBufferRNN(_max_len, state_dim, action_dim, if_on_policy, if_gpu=if_gpu,hidden_dim=hidden_dim)
                        for _ in range(rollout_num)]

    def extend_buffer(self, state, other, hidden, i):
        self.buffers[i].extend_buffer(state, other, hidden)

    def sample_batch(self, batch_size) -> tuple:
        rd_batch_sizes = rd.rand(self.rollout_num)
        rd_batch_sizes = (rd_batch_sizes * (batch_size / rd_batch_sizes.sum())).astype(np.int)
        # 这个rd_batch_size和batch_size的大小一致，从不同的buffer按概率采样。
        l__r_m_a_s_ns = [self.buffers[i].sample_batch(rd_batch_sizes[i])
                         for i in range(self.rollout_num) if rd_batch_sizes[i] > 2]
        return (torch.cat([item[0] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[1] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[2] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[3] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[4] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[5] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[6] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[7] for item in l__r_m_a_s_ns], dim=0),
                torch.cat([item[8] for item in l__r_m_a_s_ns], dim=0)

                )

    def sample_all(self) -> tuple:
        l__r_m_a_n_s = [self.buffers[i].sample_all()
                        for i in range(self.rollout_num)]
        return (torch.cat([item[0] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[1] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[2] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[3] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[4] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[5] for item in l__r_m_a_n_s], dim=0),
                torch.cat([item[6] for item in l__r_m_a_n_s], dim=0)

                )

    def update_now_len_before_sample(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len_before_sample()
            self.now_len += buffer.now_len

    def empty_buffer_before_explore(self):
        for buffer in self.buffers:
            buffer.empty_buffer_before_explore()

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
