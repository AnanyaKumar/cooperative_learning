from coop_env import CoopEnv
import gym
import time

from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np


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

def run_nn_policy(env, model, stddev=1.0):
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
    old_state = env.reset()
    env.render()

    episode = []
    total_reward = 0
    num_steps = 0
    while True:
        state_rep = interface.build_nn_input(old_state, 2)
        pred = model.predict_on_batch(state_rep)
        action = interface.build_nn_output(pred, env.get_max_accel(), std_x=stddev, std_y=stddev)
        new_state, reward, is_terminal, debug_info = env.step(action)
        episode.append((state_rep, pred, action, reward))
        env.render()
        old_state = new_state

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break
    return total_reward, num_steps, episode

def reinforce(env, model, episode, total_reward, stddev=1.0):
    gt = total_reward
    var = stddev ** 2
    for state_rep, pred, action, reward in episode:
        action_t = np.array(action).transpose()
        target = (action_t - pred)/var * gt + pred
        model.train_on_batch(state_rep, target)
        gt -= reward

def create_model(k):
    model = Sequential()
    model.add(Dense(units=56, input_dim = k * 4 + 4))
    model.add(Activation('relu'))
    model.add(Dense(units=56))
    model.add(Activation('relu'))
    model.add(Dense(units=56))
    model.add(Activation('relu'))
    model.add(Dense(units=56))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.compile(optimizer='rmsprop',
        loss='mse')
    return model

def main():
    env = gym.make('coop-v0')
    model = create_model(2)
    stddev = 100.0
    stddev_delta = 0.1
    stddev_min = 1.0
    while (1):
        total_reward, num_steps, episode = run_nn_policy(env, model, stddev)
        reinforce(env, model, episode, total_reward, stddev)
        if stddev > stddev_min:
            stddev -= stddev_delta
        print total_reward, num_steps
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
