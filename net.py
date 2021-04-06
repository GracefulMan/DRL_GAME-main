import torch
import torch.nn as nn
import numpy as np

class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(ActorPPO, self).__init__()
        if isinstance(state_dim, int):
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, action_dim), )
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def compute_logprob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return logprob.sum(1)

class ActorPPOWithRNN(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(ActorPPO, self).__init__()
        if isinstance(state_dim, int):
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, mid_dim))
            self.rnn = nn.RNN(input_size=mid_dim, hidden_size=mid_dim) # (seq, batch, feature)
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim), )

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def compute_logprob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return logprob.sum(1)

class CriticAdv(nn.Module):
    def __init__(self, state_dim, mid_dim):
        super(CriticAdv, self).__init__()
        if isinstance(state_dim, int):
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, 1))
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                     nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                     nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                     nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                     nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                     NnnReshape(-1),
                                     nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))

        layer_norm(self.net[-1], std=0.5)  # output layer for Q value

    def forward(self, state):
        return self.net(state)  # Q value


class InterPPO(nn.Module):  # Pixel-level state version
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def set_dim(i):
            return int(12 * 1.5 ** i)

        if isinstance(state_dim, int):
            self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim))  # the only difference.
        else:
            self.enc_s = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                                       nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                                       nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                                       nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                                       nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                                       nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                                       nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                                       NnnReshape(-1),
                                       nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim))
        out_dim = mid_dim

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))
        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)

        self.dec_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))
        self.dec_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))

        layer_norm(self.dec_a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg.tanh()

    def get_action_noise(self, state):
        s_ = self.enc_s(state)
        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()

        # a_noise = torch.normal(a_avg, a_std) # same as below
        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def get_q_logprob(self, state, noise):
        s_ = self.enc_s(state)

        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))
        logprob = -(noise.pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q, logprob

    def get_q1_q2_logprob(self, state, action):
        s_ = self.enc_s(state)

        q1 = self.dec_q1(s_)
        q2 = self.dec_q2(s_)

        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()
        logprob = -(((a_avg - action) / a_std).pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q1, q2, logprob


"""utils"""


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        assert (mid_dim / (2 ** 3)) % 1 == 0

        def set_dim(i):
            return int((3 / 2) ** i * mid_dim)

        self.dense1 = nn.Sequential(nn.Linear(set_dim(0), set_dim(0) // 2), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(set_dim(1), set_dim(1) // 2), nn.Hardswish())
        self.out_dim = set_dim(2)

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


if __name__ == '__main__':
    pass
