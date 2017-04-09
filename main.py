from coop_env import CoopEnv
import gym
import time

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

        time.sleep(0.1)

    return total_reward, num_steps


def main():
    env = gym.make('coop-v0')
    total_reward, num_steps = run_random_policy(env)
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
