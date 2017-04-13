from coop_env import CoopEnv
import gym
import time
import interface
from keras.layers import Dense, Activation
from keras.models import Sequential


def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

    return total_reward, num_steps

def run_nn_policy(env, model):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    state = env.reset()
    env.render()

    total_reward = 0
    num_steps = 0
    while True:
        state, reward, is_terminal, debug_info = env.step(
            interface.build_nn_output(model.predict_on_batch(interface.build_nn_input(state, 2))))
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break
    return total_reward, num_steps


def create_model(k):
    model = Sequential()
    model.add(Dense(units=512, input_dim = k * 4 + 4))
    model.add(Activation('relu'))
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dense(units=4))
    model.compile(optimizer='rmsprop',
        loss='mse')
    return model

def main():
    env = gym.make('coop-v0')
    total_reward, num_steps = run_nn_policy(env, create_model(2))
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
