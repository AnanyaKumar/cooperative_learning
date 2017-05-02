from coop_env import CoopEnv
import gym
import math
import time
import interface
from keras.layers import Dense, Activation, Lambda

from keras.models import Sequential
from keras import optimizers
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

def run_nn_policy(env, nn, k, l, stddev=1.0, render=False):
    total_reward = 0
    num_steps = 0
    state = env.reset()
    while True:
        state_rep = interface.build_nn_input(state, k, l)
        pred = nn.predict_on_batch(state_rep)
        action = interface.build_nn_output(pred, std_x=stddev, std_y=stddev)
        clipped_action = interface.clip_output(action, env.get_max_accel())
        state, reward, is_terminal, debug_info = env.step(clipped_action)
        if render:
            env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

    return total_reward, num_steps

def run_actor_critic_episode(env, actor, critic, k, l, stddev=1.0, render=True):
    """Run one episode of actor-critic with baseline. See page 272 of
        Sutton and Barto for algorithm."""
    old_state = env.reset()
    if render:
        env.render()

    total_reward = 0
    num_steps = 0
    num_cars = len(env._cars)
    while True:
        old_state_rep = interface.build_nn_input(old_state, k, l)
        pred = actor.predict_on_batch(old_state_rep)
        action = interface.build_nn_output(pred, std_x=stddev, std_y=stddev)

        clipped_action = interface.clip_output(action, env.get_max_accel())
        # It's important that we send the clipped action to the environment, but
        # use the unclipped action for reinforce. Otherwise reinforce won't
        # get the correct updates.
        new_state, reward, is_terminal, debug_info = env.step(clipped_action)

        # Train critic
        if is_terminal:
            next_reward = np.array([[0.0]] * num_cars)
        else:
            new_state_rep = interface.build_nn_input(new_state, k, l)
            next_reward = critic.predict_on_batch(new_state_rep)
        for i in range(num_cars):
            next_reward[i][0] += reward
        cur_reward = critic.predict_on_batch(old_state_rep)
        delta = next_reward - cur_reward
        critic.train_on_batch(old_state_rep, next_reward)
        # print num_steps, critic.predict_on_batch(old_state_rep)[0][0]

        # Train actor

        action_t = np.array(action).transpose()
        target = (action_t - pred) / (stddev ** 2) * delta + pred
        actor.train_on_batch(old_state_rep, target)

        # Render if requested.
        if render:
            env.render()

        # Bookkeeping.
        old_state = new_state
        total_reward += reward
        num_steps += 1

        if is_terminal:
            break
    return total_reward, num_steps


def run_monte_carlo_episode(env, actor, critic, k, l, stddev=1.0, render=True):
    """Run one episode of monte carlo reinforce with baseline. See page 271 of
    Sutton and Barto for algorithm."""
    old_state = env.reset()
    if render:
        env.render()

    episode = []
    total_reward = 0
    num_steps = 0
    while True:
        old_state_rep = interface.build_nn_input(old_state, k, l)
        pred = actor.predict_on_batch(old_state_rep)
        action = interface.build_nn_output(pred, std_x=stddev, std_y=stddev)
        clipped_action = interface.clip_output(action, env.get_max_accel())
        # It's important that we send the clipped action to the environment, but
        # use the unclipped action for reinforce. Otherwise reinforce won't
        # get the correct updates.
        new_state, reward, is_terminal, debug_info = env.step(clipped_action)
        episode.append((old_state_rep, pred, action, reward))
        num_cars = len(env._cars)
        # Train critic
        if is_terminal:
            next_reward = np.array([[0.0]] * num_cars)
        else:
            new_state_rep = interface.build_nn_input(new_state, k, l)
            next_reward = critic.predict_on_batch(new_state_rep)
        for i in range(num_cars):
            next_reward[i][0] += reward
        critic.train_on_batch(old_state_rep, next_reward)

        # Render if requested.
        if render:
            env.render()

        # Bookkeeping.
        old_state = new_state
        total_reward += reward
        num_steps += 1

        if is_terminal:
            break
    
    Gt = total_reward
    var = stddev ** 2
    state_batch = []
    target_batch = []
    for state_rep, pred, action, reward in episode:
        action_t = np.array(action).transpose()
        critic_reward = critic.predict_on_batch(state_rep)[0][0]
        target = (action_t - pred)/var * (Gt - critic_reward) + pred
        state_batch.append(state_rep)
        target_batch.append(target)
        Gt -= reward
    actor.train_on_batch(np.concatenate(state_batch), np.concatenate(target_batch))
    return total_reward, num_steps
    
def create_policy_model(k, l, max_acc):
    model = Sequential()
    model.add(Dense(units=8, input_dim = interface.get_nn_input_dim(k,l)))
    model.add(Activation('relu'))
    model.add(Dense(units=8))
    model.add(Activation('relu'))
    model.add(Dense(units=8))
    model.add(Activation('relu'))
    # model.add(Dense(units=2, kernel_initializer='zeros',
    #     bias_initializer='zeros'))
    # model.add(Activation('relu'))
    # model.add(Dense(units=56))
    # model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('tanh'))
    layer = Dense(2, trainable=False)
    model.add(layer)
    l = layer.get_weights()
    print l
    l[0][0] = np.array([0.0, max_acc])
    l[0][1] = np.array([max_acc, 0.0])
    l[1] = np.array([max_acc/2.0,0.0]) # Bias the network to go forward.
    layer.set_weights(l)
    print layer.get_weights()
    #rmsprop = optimizers.RMSprop(lr=0.01, clipnorm=10.)
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def create_critic_model(k,l):
    model = Sequential()
    model.add(Dense(units=12, input_dim = interface.get_nn_input_dim(k,l)))
    model.add(Activation('relu'))
    model.add(Dense(units=12))
    model.add(Activation('relu'))
    model.add(Dense(units=12))
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    #rmsprop = optimizers.RMSprop(lr=0.01, clipnorm=10.)
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def get_test_reward(env, actor, critic, k, l, test_std_dev, num_testing_iterations=50):
    ave_reward = 0.0
    ave_steps = 0.0
    for i in range(num_testing_iterations):
        total_reward, num_steps = run_nn_policy(env, actor, k, l, test_std_dev, False)
        # print total_reward
        ave_reward += total_reward
        ave_steps += num_steps
    ave_reward /= num_testing_iterations
    ave_steps /= num_testing_iterations
    return ave_reward, ave_steps

def main():
    env = gym.make('coop4cars-v0')
    num_training_iterations = 20000
    testing_frequency = 200 # After how many training iterations do we check the testing error?
    num_testing_iterations = 50
    k = 0 # Number of closest cars the neural net stores
    l = 0 # Number of closest obstacles the neural net stores
    # Check that k < total number of cars, l <= total number of obstacles
    
    # Initialize actor and critics.
    actor = create_policy_model(k, l, env.get_max_accel())
    critic = create_critic_model(k, l)

    # Anneal the standard deviation down.
    test_std_dev = 0.00001
    stddev = 0.03
    stddev_delta = 0.000000
    stddev_min = 0.0001

    for i in range(num_training_iterations):
        total_reward, num_steps = run_monte_carlo_episode(env, actor, critic, k, l, stddev, False)
        if stddev > stddev_min:
            stddev -= stddev_delta
        # Get test error every so often
        if i % testing_frequency == 0:
            ave_reward, ave_steps = get_test_reward(env, actor, critic, k, l, test_std_dev, 5)
            print ave_reward, ave_steps, stddev
            # print model.get_weights()
    ave_reward, ave_steps = get_test_reward(env, actor, critic, k, l, test_std_dev, 50)
    print ave_reward, ave_steps, stddev

if __name__ == '__main__':
    main()
