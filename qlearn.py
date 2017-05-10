import numpy as np
import gym
import random
import interface

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation

from coop_env import CoopEnv
from core import *

from hyperparameters import *

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(units=12, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(units=12))
    model.add(Activation('relu'))
    # model.add(Dense(units=12))
    # model.add(Activation('relu'))
    model.add(Dense(units=NUM_ACTIONS))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def run_episode(env, mk_state_repr, policy, render_opt=False):
    state_repr = mk_state_repr(env.reset())
    if render_opt:
        env.render()
    episode = []
    total_reward = 0
    num_steps = 0
    while True:
        action = policy(state_repr)
        next_state, reward, is_terminal, _ = env.step(action)
        if render_opt:
            env.render()
        next_state_repr = mk_state_repr(next_state)
        episode.append(Sample(state_repr, action, reward, next_state_repr, is_terminal))
        state_repr = next_state_repr
        total_reward += reward[0]
        num_steps += 1
        if is_terminal:
            break

    return num_steps, total_reward, episode

def optimal_action_value(action_values):
    return max(enumerate(action_values), key=lambda p: p[1])

def qlearn_train(online_model, target_model, experience):
    old_action_values = online_model.predict_on_batch(experience.state)
    new_action_values = target_model.predict_on_batch(experience.next_state)
    #print new_action_values
    
    target = []
    for old_values, new_values, reward in zip(old_action_values.tolist(), new_action_values.tolist(), experience.reward):
        action, value = optimal_action_value(new_values)
        if experience.is_terminal:
            old_values[action] = reward
        else:
            old_values[action] = reward + GAMMA * value
        target.append(list(old_values))

    #print "target=", target
    online_model.train_on_batch(experience.state, np.array(target))

def main():
    env = gym.make('coop1car1obs-v0')
    k = 0
    l = 0
    online_model = create_model(interface.get_nn_input_dim(k, l))
    target_model = create_model(interface.get_nn_input_dim(k, l))

    def mk_state_repr(state):
        return interface.build_nn_input(state, k, l)

    def policy(state_repr):
        pred = online_model.predict_on_batch(state_repr)
        actions = []
        for output in pred:
            if random.uniform(0., 1.) <= EPS:
                actions.append(random.randint(0, NUM_ACTIONS-1))
            else:
                action, _ = optimal_action_value(output)
                actions.append(action)
        
        return actions

    replay_memory = ReplayMemory(REPLAY_MEMORY_MAX_SIZE)
    updates = 0
    
    for i in range(NUM_TRAINING_ITERS):
        num_steps, total_reward, episode = run_episode(env, mk_state_repr, policy, render_opt=False)
        for experience in episode:
            replay_memory.append(experience)
        for experience in replay_memory.sample(REPLAY_MEMORY_BATCH_SIZE):
            qlearn_train(online_model, target_model, experience)
        updates += REPLAY_MEMORY_BATCH_SIZE
        if updates >= TARGET_FIXING_THRESHOLD:
            online_model.set_weights(target_model.get_weights())
            updates = 0
        
        #print "total reward=",total_reward, "num steps=", num_steps
                                                 
        if i % TESTING_FREQUENCY == 0:
            total_reward = 0
            for j in range(NUM_TESTING_ITERS):
                _, reward, _ = run_episode(env, mk_state_repr, policy, render_opt=True)
                total_reward += reward
            print "total_reward=", total_reward

        
    
    
    
main()
