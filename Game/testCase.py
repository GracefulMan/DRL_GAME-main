import gym
import time
import numpy as np


def env_test():
    from Game.env import AtariGameEnv, PreprocessEnv
    env = gym.make('Breakout-v4')
    env = PreprocessEnv(env, is_image=True,is_gray=False)
    print(env.__dict__)
    print('id:', env.unwrapped.spec.id)
    print('target reward:', env.target_reward)
    print('observation space:', env.observation_space.shape)
    print('action space:', env.action_space)
    epochs = 1000
    env.reset()
    for _ in range(epochs):
        env.render()
        action = env.action_space.sample()
        s_, reward, done, _ = env.step(action)
        print(s_.shape)
        if reward!=0: print(reward)
        time.sleep(0.01)



def env_test2():
    from Game.env import AtariGameEnv, PreprocessEnv
    env = gym.make('Breakout-v4')
    env = AtariGameEnv(env)
    print('id:', env.unwrapped.spec.id)
    print('target reward:', env.target_reward)
    print('observation space:', env.observation_space.shape)
    print('action space:', env.action_space)
    epochs = 1000
    env.reset()
    for _ in range(epochs):
        env.render()
        action = env.action_space.sample()
        s_, reward, done, _ = env.step(action)
        print(s_.shape)
        if reward!=0: print(reward)
        time.sleep(0.01)



def env_test3():
    from atari_env import AtariGameEnv
    env = gym.make('CarRacing-v0')
    env = AtariGameEnv(env, episode_life=False)
    epochs = 1000
    env.reset()
    for _ in range(epochs):
        env.render()
        action = env.action_space.sample()
        s_, reward, done, _ = env.step(action)
        if reward!=0: print(reward)
        time.sleep(0.01)


def numpy_test():
    state_dim = [3]
    tmp= np.empty((123, *state_dim))
    print(tmp.shape)


def image_test():
    import cv2
    def process_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame[:, :, np.newaxis]
    env = gym.make('Breakout-v4')
    obs = env.reset()
    obs1 = process_frame(obs)
    print(obs1.shape)
    cv2.imshow('', obs1)


if __name__ == '__main__':
    # numpy_test()
    # env_test()
    #image_test()
    # env_test3()