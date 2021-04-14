import torch
import torch.nn as nn
import numpy as np

'''Q Network'''


class ImageConv(nn.Module):
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def __init__(self, image_shape: tuple) -> None:
        '''
        image_shape:(channel, w, h), such as: (3, 200, 240).
        '''
        assert len(image_shape) == 3, ValueError("wrong image shape.")
        super(ImageConv, self).__init__()
        channels, w, h = image_shape
        filters = 32
        self.net = nn.Sequential(
            nn.Conv2d(channels, filters, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            tmp = torch.rand((1, channels, w, h))
            self.out_dim = self.net(tmp).shape[1]
        self._initialize_weights()

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state)  # Q value


class QNetDuel(nn.Module):  # Dueling DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super(QNetDuel, self).__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))  # Q value
        self.net_adv = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val(t_tmp)
        q_adv = self.net_adv(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling Q value


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super(QNetTwin, self).__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        q1 = self.net_q1(tmp)
        q2 = self.net_q2(tmp)
        return q1, q2  # two Q values


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super(QNetTwinDuel, self).__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q1 value
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q2 value
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


'''Policy Network (Actor)'''


class Actor(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state): # [batch_size, state_dim],eg:[254, 3]
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

### Hongying: add RNN into Reinforcement.


class ActorRNN(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super(ActorRNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU()
                                 )
        self.hidden_dim = mid_dim
        self.num_layers = 1
        self.rnn = nn.LSTM(input_size=mid_dim, hidden_size=mid_dim, num_layers=self.num_layers)

        self.action = nn.Linear(mid_dim, action_dim)

    def init_hidden(self, batch_size):
        #TODO: 将lstm怎么加入到网络中考虑清楚。
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        )

    def forward(self, state, hidden):
        out = self.net(state).detach_()
        out = out.view(1, state.shape[0], -1) #(seq_len, batch, input_size)
        # Initialize hidden state with zeros
        out, hidden = self.rnn(out, hidden)
        out = out.squeeze()
        return self.action(out).tanh(), hidden  # action.tanh()

    def get_action(self, state, hidden, action_std):
        with torch.no_grad():
            action, hidden = self.forward(state, hidden)
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0), hidden

    def get_full_state_and_action(self, state, hidden, action_std):
        action, hidden = self.get_action(state, hidden, action_std)
        h, c = hidden
        h, c = h.squeeze(), c.squeeze()
        new_state = torch.cat([state, h, c], dim=1)
        new_action = torch.cat([action, h, c], dim=1)
        return new_state, new_action






class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):

        return self.net(torch.cat((state, action), dim=1))  # Q value


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


class CriticTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        if if_use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense = DenseNet(mid_dim)
            lay_dim = nn_dense.out_dim
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn_dense, )  # state-action value function
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            lay_dim = mid_dim
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, lay_dim), nn.ReLU())

        self.net_q1 = nn.Linear(lay_dim, 1)
        self.net_q2 = nn.Linear(lay_dim, 1)
        layer_norm(self.net_q1, std=0.1)
        layer_norm(self.net_q2, std=0.1)

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


'''Integrated Network (Parameter sharing)'''


class InterDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim))
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim))

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.Hardswish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh())
        self.dec_q = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.Hardswish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)))

    @staticmethod
    def add_noise(a, noise_std):
        a_temp = torch.normal(a, noise_std)
        mask = torch.tensor((a_temp < -1.0) + (a_temp > 1.0), dtype=torch.float32).cuda()

        noise_uniform = torch.rand_like(a)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q = self.dec_q(q_)
        return q

    def next_q_action(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        '''q_target (without noise)'''
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        '''q_target (with noise)'''
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a



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


def demo_conv2d_state():
    state_dim = (3, 96, 96)
    batch_size = 3

    def set_dim(i):
        return int(12 * 1.5 ** i)

    mid_dim = 128
    net = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                        nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                        nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                        nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                        nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                        nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                        nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                        NnnReshape(-1),
                        nn.Linear(set_dim(5), mid_dim), nn.ReLU())

    inp_shape = list(state_dim)
    inp_shape.insert(0, batch_size)
    inp = torch.ones(inp_shape, dtype=torch.float32)
    inp = inp.view(3, -1)
    print(inp.shape)
    out = net(inp)
    print(out.shape)
    exit()


if __name__ == '__main__':
    state_dim = 16
    mid_dim = 128
    action_dim = 2
    batch_size = 64


    actor = ActorRNN(mid_dim=mid_dim, state_dim=state_dim, action_dim=action_dim)
    observation = torch.FloatTensor(torch.rand(size=(batch_size, state_dim)))
    ho = actor.init_hidden(batch_size)
    act, hidden = actor(observation, ho)
    act, hidden = actor.get_action(observation, hidden, 1)
    h_0, c_0 = hidden
    print(f'state dim:{(batch_size, state_dim)}')
    print(f'action shape:{act.shape}')
    print(f'hidden state:{h_0.shape}')
    newstate, newaction = actor.get_full_state_and_action(observation, hidden, 1)
    critic = Critic(mid_dim=mid_dim, state_dim=mid_dim * 2 + state_dim, action_dim=mid_dim * 2 + action_dim)
    print(critic)
    print("newstate:", newstate.shape)
    print("newaction:",newaction.shape)
    critic.forward(newstate, newaction)




