"""Core classes."""

import random

class Sample:
    """Represents a reinforcement learning sample.
    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.
    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, is_terminal):
      self.state = state
      self.action = action
      self.reward = reward
      self.next_state = next_state
      self.is_terminal = is_terminal
      self.validate()

    def validate(self):
#      assert(type(self.state) == list)
#      assert(type(self.next_state) == list)
#      if self.is_terminal:
#        assert self.next_state[0][0][0] == 0
        return True

    def get_type(self):
      return "Sample"


class ReplayMemory:
    """Uses a ring buffer to store replay memory"""
    def __init__(self, max_size=10**7):
        # The items will be stored in the range [self.low, self.high),
        # so it is inclusive of the lower index but not the upper index.
        assert(max_size >= 1)
        self.low = 0
        self.high = 0
        self.size = 0
        self.max_size = max_size
        # TODO: if desired make this into a numpy array for efficiency, if required.
        self.storage = [None for _ in range(max_size+1)]

    def validate_indices(self):
        """Check if the indices are valid"""
        assert 0 <= self.low and self.low <= self.max_size
        assert 0 <= self.high and self.high <= self.max_size
        assert self.make_valid(self.low + self.size) == self.high
        assert 0 <= self.size and self.size <= self.max_size

    def make_valid(self,index):
        """Take a non-negative index and mod it to be in the right range"""
        return index % (len(self.storage))

    def append(self, value):
        """Add a value to memory"""
        self.validate_indices()
        self.storage[self.high] = value
        self.high = self.make_valid(self.high + 1)
        if self.size == self.max_size:
          self.storage[self.low] = None
          self.low = self.make_valid(self.low + 1)
        else:
          self.size += 1
        self.validate_indices()

    def sample(self, batch_size, indexes=None):
        """Sample batch_size > 0 samples from memory. The first sample will be the
           most recently appended, the other samples are uniformly random"""
        assert self.size > 0
        self.validate_indices()
        batch_size = min(self.size, batch_size)
        sample_list = [None] * batch_size
        # We deviate from the DeepMind paper, and make sure the newest sample is given at least once
        # to the neural net.
        sample_list[0] = self.storage[self.make_valid(self.low + self.size - 1)]
        assert sample_list[0] != None
        for i in range(1, batch_size):
          r = random.randint(0, self.size - 1)
          sample_list[i] = self.storage[self.make_valid(self.low + r)]
          assert sample_list[i] != None
        return sample_list
