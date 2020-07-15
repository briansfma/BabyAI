import numpy as np
from numba import jit

# Activation Functions
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
  return np.tanh(x)

# Neuron types (classified by activation for now)
class SigmoidNeuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

class ReLuNeuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return relu(total)

class TanhNeuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return tanh(total)

class SimpleNN:
  def __init__(self):
    # He initialization for hidden ReLus, Xavier init for output Tanh
    weights_h1 = np.random.randn(1, 3) * np.sqrt(2/3)
    weights_h2 = np.random.randn(1, 3) * np.sqrt(2/3)
    weights_o = np.random.randn(1, 2) * np.sqrt(1/2)
    bias_h1 = np.random.randn() * 0.1
    bias_h2 = np.random.randn() * 0.1
    bias_o = np.random.randn() * 0.1

    # The Neuron class here is from the previous section
    self.h1 = ReLuNeuron(weights_h1, bias_h1)
    self.h2 = ReLuNeuron(weights_h2, bias_h2)
    self.o = TanhNeuron(weights_o, bias_o)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)
    out_o = self.o.feedforward([out_h1, out_h2])

    # print("Out_h1: {}, Out_h2: {}, Out_o: {}".format(out_h1, out_h2, out_o))

    return out_o

  def train(self):
    pass

# Generate a reward value for given steering/throttle input.
# Rough approximation now for early training
class SimEnviron:
  def __init__(self):
    self.maxspeed = 0.
    self.speed = 0.
    self.heading = 0.
    self.dist = 0.
    self.s = 0.

  def reward(self, steering, throttle):

    # Handle throttle/brake effect on speed
    self.speed += throttle - 0.1 * (self.speed)**2  # Approximate drag/friction
    self.speed = np.maximum(self.speed, 0)              # Clamp min value to 0 (stopped)
    self.maxspeed += 1 - 0.1 * (self.speed)**2

    # Handle steering effect on heading
    self.heading += self.speed * steering   # Discrete approximation

    # Handle net effect on car's position
    interval = self.speed * np.cos(self.heading)
    self.dist += interval
    self.s += self.speed * np.sin(self.heading)

    # Report Progress
    print("  Speed: {:06.4f}, Heading: {:06.4f}, D: {:06.4f}, S: {:06.4f}"
          .format(self.speed, self.heading, self.dist, self.s))

    # Assign a reward value
    # interval: the closer to theoretical max, the more forward progress we're making.
    # self.s: the closer to 0, the closer we are sticking to the path.
    d_loss = np.exp(-2 * (self.maxspeed - interval))
    s_loss = np.exp(-2 * self.s)
    reward = d_loss + s_loss

    print(reward)
    return reward


# Let's play!
environment = SimEnviron()
calcSteer = SimpleNN()
calcThrottle = SimpleNN()

# Initial conditions
x = np.array([0., 0., 0.])   # x1 = Dist, x2 = last Steering, x3 = last Throttle

# Do the feedforward loops
for i in range(3):
  steer_out = np.asscalar(calcSteer.feedforward(x))
  throt_out = np.asscalar(calcThrottle.feedforward(x))

  print("Steering = {:06.4f}, Throttle = {:06.4f}".format(steer_out, throt_out))

  environment.reward(steer_out, throt_out)
