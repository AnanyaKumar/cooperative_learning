
from collections import deque

class RollingStats:
  """Outputs rolling averages of numbers inserted. Useful for collecting smooth statistics."""
  
  def __init__(self, size):
    """The maximum number of previous data points we should consider"""
    self.data = deque()
    self.max_size = size
    self.len = 0
    self.sum = 0

  def add_num(self, num):
    """Insert a number into the stream"""
    self.data.appendleft(num)
    self.sum += num
    self.len += 1
    if self.len > self.max_size:
      self.len -= 1
      self.sum -= self.data.pop()
      assert self.len == self.max_size

  def length(self):
    """Get the number of elements currently in the stream"""
    return self.len

  def get_average(self):
    """Print the rolling average of data points"""
    if self.len == 0:
      return 0
    return self.sum / self.len
