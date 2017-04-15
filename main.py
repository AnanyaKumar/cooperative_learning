from coop_env import CoopEnv
import gym
import time
import interface
from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
from rolling_stats import RollingStats

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

def run_nn_policy(env, model, k, stddev=1.0, render=True):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.
    model: NN model
    k: number of neighboring cars we use as NN input

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    old_state = env.reset()
    if render:
        env.render()

    episode = []
    total_reward = 0
    num_steps = 0
    while True:
        state_rep = interface.build_nn_input(old_state, k)
        pred = model.predict_on_batch(state_rep)
        action = interface.build_nn_output(pred, std_x=stddev, std_y=stddev)
        clipped_action = interface.clip_output(action, env.get_max_accel())
        # It's important that we send the clipped action to the environment, but
        # use the unclipped action for reinforce. Otherwise reinforce won't
        # get the correct updates.
        new_state, reward, is_terminal, debug_info = env.step(clipped_action)
        episode.append((state_rep, pred, action, reward))
        if render:
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
    model.add(Dense(units=2, input_dim = k * 4 + 4, kernel_initializer='zeros',
        bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # model.add(Dense(units=56))
    # model.add(Activation('relu'))
    # model.add(Dense(units=56))
    # model.add(Activation('relu'))
    # model.add(Dense(units=56))
    # model.add(Activation('relu'))
    # model.add(Dense(units=2))
    model.compile(optimizer='rmsprop',
        loss='mse')
    return model

def main():
    env = gym.make('coop-v0')
    smooth_average_reward = RollingStats(30)
    num_training_iterations = 2000
    num_testing_iterations = 50
    k = 0
    model = create_model(k)
    # Anneal the standard deviation down.
    test_std_dev = 0.01
    stddev = 10.0
    stddev_delta = 0.01
    stddev_min = 0.2
    for i in range(num_training_iterations):
        total_reward, num_steps, episode = run_nn_policy(env, model, k, stddev, False)
        reinforce(env, model, episode, total_reward, stddev)
        if stddev > stddev_min:
            stddev -= stddev_delta
        smooth_average_reward.add_num(total_reward)
        print smooth_average_reward.get_average(), total_reward, num_steps

    ave_reward = 0.0
    ave_steps = 0.0
    for i in range(num_testing_iterations):
        total_reward, num_steps, _ = run_nn_policy(env, model, k, test_std_dev, False)
        ave_reward += total_reward
        ave_steps += num_steps
    ave_reward /= num_testing_iterations
    ave_steps /= num_testing_iterations
    print('Agent received total reward of: %f' % ave_reward)
    print('Agent took %d steps' % ave_steps)


if __name__ == '__main__':
    main()
