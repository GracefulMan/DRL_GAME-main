import os
import numpy as np
import numpy.random as rd
import gym
import cv2


class PreprocessEnv(gym.Wrapper):  # environment wrapper #
    def __init__(self, env, if_print=True, data_type=np.float32, is_image=False, is_gray=False, resize=None):
        """Preprocess a standard OpenAI gym environment for RL training.

        :param env: a standard OpenAI gym environment, it has env.reset() and env.step()
        :param if_print: print the information of environment. Such as env_name, state_dim ...
        :param data_type: convert state (sometimes float64) to data_type (float32).
        """
        super(PreprocessEnv, self).__init__(env)
        self.env = env
        self.data_type = data_type
        self.if_print = if_print
        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_reward
         ) = get_gym_env_info(env)
        # 图像输入
        self.is_gray = is_gray
        self.is_image = is_image
        self.resize = resize
        if self.is_image:
            w, h, c = self.state_dim
            if self.is_gray:c = 1
            if self.resize is not None:
                w, h = self.resize, self.resize
            self.state_dim = [c, w, h]
            print('new state_dim:', self.state_dim)


        state_avg, state_std = get_avg_std__for_state_norm(self.env_name)
        if state_avg is not None:
            self.neg_state_avg = -state_avg
            self.div_state_std = 1 / (state_std + 1e-4)

            self.reset = self.reset_norm
            self.step = self.step_norm
        else:
            self.reset = self.reset_type
            self.step = self.step_type


    def print_info(self):
        print(f"\n| env_name:  {self.env_name}, action space if_discrete: {self.if_discrete}"
              f"\n| state_dim: {self.state_dim}, action_dim: {self.action_dim}, action_max: {self.action_max}"
              f"\n| max_step:  {self.max_step}, target_reward: {self.target_reward}")

    def reset_type(self) -> np.ndarray:
        """ state = env.reset()

        convert the data type of state from float64 to float32

        :return array state: state.shape==(state_dim, )
        """
        state = self.env.reset()
        if self.is_image:
            if self.is_gray:
                state = process_frame(state, self.resize)
            state = state.transpose([2, 0, 1])
            state = state / 255.
        return state.astype(self.data_type)

    def reset_norm(self) -> np.ndarray:
        """ state = env.reset()

        convert the data type of state from float64 to float32
        do normalization on state

        :return array state: state.shape==(state_dim, )
        """
        state = self.env.reset()
        (state + self.neg_state_avg) * self.div_state_std
        return state.astype(self.data_type)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        """ next_state, reward, done = env.step(action)

        convert the data type of state from float64 to float32,
        adjust action range to (-action_max, +action_max)

        :return array state:  state.shape==(state_dim, )
        :return float reward: reward of one step
        :return bool  done  : the terminal of an training episode
        :return dict  info  : the information save in a dict. OpenAI gym standard. Send a `None` is OK
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        if self.is_image:
            if self.is_gray:
                state = process_frame(state, self.resize)
            state = state.transpose([2, 0, 1])
            state = state / 255.

        return state.astype(self.data_type), reward, done, info

    def step_norm(self, action) -> (np.ndarray, float, bool, dict):
        """ next_state, reward, done = env.step(action)

        convert the data type of state from float64 to float32,
        adjust action range to (-action_max, +action_max)
        do normalization on state

        :return array state:  state.shape==(state_dim, )
        :return float reward: reward of one step
        :return bool  done  : the terminal of an training episode
        :return dict  info  : the information save in a dict. OpenAI gym standard. Send a `None` is OK
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(self.data_type), reward, done, info


class AtariGameEnv(PreprocessEnv):
    def __init__(self,*args ,**kwargs):
        kwargs['is_image'] = True
        kwargs['is_gray'] = True
        kwargs['resize'] = 80
        self.layer_nums = 3
        self.gap = 3
        self.current = 0
        super(AtariGameEnv, self).__init__(*args, **kwargs)
        self.state_dim[0] = self.layer_nums
        self.state = np.zeros((self.state_dim))

        self.lives = 0
        self.was_real_done = True
        self.print_info()


    def reset_type(self, **kwargs) -> np.ndarray:

        if self.was_real_done:
            state = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            state, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()

        if self.is_image:
            if self.is_gray:
                state = process_frame(state, self.resize)
            state = state.transpose([2, 0, 1])
            state = state / 255.
        self.state = np.concatenate((self.state[1:], state))
        return self.state.astype(self.data_type)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        state, reward, done, info = self.env.step(action * self.action_max)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives

        if self.is_image:
            if self.is_gray:
                state = process_frame(state, self.resize)
            state = state.transpose([2, 0, 1])
            state = state / 255.
        if self.current % self.gap == 0:
            self.state = np.concatenate((self.state[1:], state))
        else:
            self.current += 1
            if self.current == self.gap:
                self.current = 0
        return self.state.astype(self.data_type), reward, done, info

    def print_info(self):
        PreprocessEnv.print_info(self)



def process_frame(frame, resize):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if resize is not None:
        frame = cv2.resize(frame, (resize, resize))
    return frame[:, :, np.newaxis]

def get_avg_std__for_state_norm(env_name) -> (np.ndarray, np.ndarray):
    """return the state normalization data: neg_avg and div_std

    ReplayBuffer.print_state_norm() will print `neg_avg` and `div_std`
    You can save these array to here. And PreprocessEnv will load them automatically.
    eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
    neg_avg = -states.mean()
    div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())


    :str env_name: the name of environment that helps to find neg_avg and div_std
    :return array avg: neg_avg.shape=(state_dim)
    :return array std: div_std.shape=(state_dim)
    """
    avg = None
    std = None
    if env_name == 'LunarLanderContinuous-v2':
        avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
                        -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
        std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
                        0.25886878, 0.277233, 0.27771219])
    elif env_name == "BipedalWalker-v3":
        avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
                        -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
                        4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
                        -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
                        3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
                        5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
        std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
                        0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
                        0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
                        0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
                        0.11752805, 0.14116005, 0.13839757, 0.07760469])
    elif env_name == 'ReacherBulletEnv-v0':
        avg = np.array([0.03149641, 0.0485873, -0.04949671, -0.06938662, -0.14157104,
                        0.02433294, -0.09097818, 0.4405931, 0.10299437], dtype=np.float32)
        std = np.array([0.12277275, 0.1347579, 0.14567468, 0.14747661, 0.51311225,
                        0.5199606, 0.2710207, 0.48395795, 0.40876198], dtype=np.float32)
    elif env_name == 'AntBulletEnv-v0':
        avg = np.array([-2.2785307e-01, -4.1971792e-02, 9.2752278e-01, 8.3731368e-02,
                        1.2131270e-03, -5.7878396e-03, 1.8127944e-02, -1.1823924e-02,
                        1.5717462e-01, 1.2224792e-03, -1.9672018e-01, 6.4919023e-03,
                        -2.0346987e-01, 5.1609759e-04, 1.6572942e-01, -6.0344036e-03,
                        -1.6024958e-02, -1.3426526e-03, 3.8138664e-01, -5.6816568e-03,
                        -1.8004493e-01, -3.2685725e-03, -1.5989083e-01, 7.0396746e-03,
                        7.2912598e-01, 8.3666992e-01, 8.2824707e-01, 7.6196289e-01],
                       dtype=np.float32)
        std = np.array([0.09652393, 0.33918667, 0.23290202, 0.13423778, 0.10426794,
                        0.11678293, 0.39058578, 0.28871638, 0.5447721, 0.36814892,
                        0.73530555, 0.29377502, 0.5031936, 0.36130348, 0.71889997,
                        0.2496559, 0.5484764, 0.39613277, 0.7103549, 0.25976712,
                        0.56372136, 0.36917716, 0.7030704, 0.26312646, 0.30555955,
                        0.2681793, 0.27192947, 0.29626447], dtype=np.float32)
    return avg, std


def get_gym_env_info(env) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.
    env_name: the environment name, such as XxxXxx-v0
    state_dim: the dimension of state
    action_dim: the dimension of continuous action; Or the number of discrete action
    action_max: the max action of continuous action; action_max == 1 when it is discrete action space
    if_discrete: Is this env a discrete action space?
    target_reward: the target episode return, if agent reach this score, then it pass this game (env).
    max_step: the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step

    :env: a standard OpenAI gym environment, it has env.reset() and env.step()
    :bool if_print: print the information of environment. Such as env_name, state_dim ...
    """
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id
    state_shape = env.observation_space.shape
    state_dim = [state_shape[0]] if len(state_shape) == 1 else list(state_shape)  # sometimes state_dim is a list
    target_reward = getattr(env, 'target_reward', None)
    target_reward_default = getattr(env.spec, 'reward_threshold', None)
    if target_reward is None:
        target_reward = target_reward_default
    if target_reward is None:
        target_reward = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_reward




"""Custom environment: Fix Env CarRacing-v0 - Box2D"""


def fix_car_racing_env(env, frame_num=3, action_num=3) -> gym.Wrapper:  # 2020-12-12
    setattr(env, 'old_step', env.step)  # env.old_step = env.step
    setattr(env, 'env_name', 'CarRacing-Fix')
    setattr(env, 'state_dim', (frame_num, 96, 96))
    setattr(env, 'action_dim', 3)
    setattr(env, 'if_discrete', False)
    setattr(env, 'target_reward', 700)  # 900 in default

    setattr(env, 'state_stack', None)  # env.state_stack = None
    setattr(env, 'avg_reward', 0)  # env.avg_reward = 0
    """ cancel the print() in environment
    comment 'car_racing.py' line 233-234: print('Track generation ...
    comment 'car_racing.py' line 308-309: print("retry to generate track ...
    """

    def rgb2gray(rgb):
        # # rgb image -> gray [0, 1]
        # gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114]).astype(np.float32)
        # if norm:
        #     # normalize
        #     gray = gray / 128. - 1.
        # return gray

        state = rgb[:, :, 1]  # show green
        state[86:, 24:36] = rgb[86:, 24:36, 2]  # show red
        state[86:, 72:] = rgb[86:, 72:, 0]  # show blue
        state = (state - 128).astype(np.float32) / 128.
        return state

    def decorator_step(env_step):
        def new_env_step(action):
            action = action.copy()
            action[1:] = (action[1:] + 1) / 2  # fix action_space.low

            reward_sum = 0
            done = state = None
            try:
                for _ in range(action_num):
                    state, reward, done, info = env_step(action)
                    state = rgb2gray(state)

                    if done:
                        reward += 100  # don't penalize "die state"
                    if state.mean() > 192:  # 185.0:  # penalize when outside of road
                        reward -= 0.05

                    env.avg_reward = env.avg_reward * 0.95 + reward * 0.05
                    if env.avg_reward <= -0.1:  # done if car don't move
                        done = True

                    reward_sum += reward

                    if done:
                        break
            except Exception as error:
                print(f"| CarRacing-v0 Error 'stack underflow'? {error}")
                reward_sum = 0
                done = True
            env.state_stack.pop(0)
            env.state_stack.append(state)

            return np.array(env.state_stack).flatten(), reward_sum, done, {}

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            state = rgb2gray(env_reset())
            env.state_stack = [state, ] * frame_num
            return np.array(env.state_stack).flatten()

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


def render__car_racing():
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    import time
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)

    state_dim = env.state_dim

    _state = env.reset()
    import cv2
    action = np.array((0, 1.0, -1.0))
    for i in range(321):
        # action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        show = state.reshape(state_dim)
        show = ((show[0] + 1.0) * 128).astype(np.uint8)
        #cv2.imshow('', show)
        #cv2.waitKey(1)
        if done:
            break
        # env.render()


"""Utils"""


def get_video_to_watch_gym_render():
    import cv2  # pip3 install opencv-python
    import gym  # pip3 install gym==0.17 pyglet==1.5.0  # env.render() bug in gym==0.18, pyglet==1.6
    import torch
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    '''choose env'''
    # from elegantrl.env import PreprocessEnv
    env = PreprocessEnv(env=gym.make('CarRacing-v0'))
    #env = gym.make('CarRacing-v0')
    '''choose algorithm'''
    from agent import AgentModSAC
    agent = AgentModSAC()
    net_dim = 2 ** 7
    cwd = 'AgentModSAC/AntBulletEnv-v0_0/'
    # from elegantrl.agent import AgentModSAC
    # agent = AgentModSAC()
    # net_dim = 2 ** 7
    # cwd = 'AgentModSAC/BipedalWalker-v3_2/'

    '''initialize agent'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    #action_dim = env.action_space.shape
    agent.init(net_dim, state_dim, action_dim)
    agent.save_load_model(cwd=cwd, if_save=False)

    '''initialize evaluete and env.render()'''
    device = agent.device
    save_frame_dir = 'frames'
    save_video = 'gym_render.mp4'

    os.makedirs(save_frame_dir, exist_ok=True)

    state = env.reset()
    for i in range(1024):
        frame = env.render('rgb_array')
        cv2.imwrite(f'{save_frame_dir}/{i:06}.png', frame)
        # cv2.imshow('', frame)
        # cv2.waitKey(1)

        s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
        a_tensor = agent.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # if use 'with torch.no_grad()', then '.detach()' not need.
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        if done:
            state = env.reset()
        else:
            state = next_state
    env.close()

    '''convert frames png/jpg to video mp4/avi using ffmpeg'''
    os.system(f"| Convert frames to video using ffmpeg. Save in {save_video}")
    os.system(f'ffmpeg -r 60 -f image2 -s 600x400 -i {save_frame_dir}/%06d.png '
              f'-crf 25 -vb 20M -pix_fmt yuv420p {save_video}')


if __name__ == '__main__':
    import os
    import pybullet_envs
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    get_video_to_watch_gym_render()
    render__car_racing()