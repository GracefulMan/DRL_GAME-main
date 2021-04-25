from atari_env import AtariGameEnv
import numpy as np
import gym


class Dataset:
    # 用前面三帧预测出后续一帧
    def __init__(self,env,  size=10000):
        self.env = AtariGameEnv(env, frame_stack=False)
        self.size = size
        shp = self.env.observation_space.shape
        self._train_data = np.empty((0, ) + shp, dtype=np.float32)
        self._reward = np.empty((0,), dtype=np.float)

    def update_buffer(self):
        obs = self.env.reset()
        index = 0
        while index < self.size:
            self._train_data = np.append(self._train_data, obs)
            obs, reward, terminal, _ = self.env.step(self.env.action_space.sample())
            self._reward = np.append(self._reward, reward)
            if terminal:
                self.env.reset()
            index += 1
            if index % 100 == 0:
                print(f'index:{index}')


dataset = Dataset(env=gym.make('Breakout-v4'))
dataset.update_buffer()