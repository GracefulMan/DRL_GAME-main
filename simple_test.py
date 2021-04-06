import gym
from nes_py.wrappers import JoypadSpace
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

def contra_game_render():
    env = gym.make('Contra-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print("actions", env.action_space)
    print("observation_space ", env.observation_space.shape)
    done = False
    env.reset()
    for step in range(5000):
        if done:
            print("Over")
            break
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()

    env.close()


if __name__ == '__main__':
    contra_game_render()