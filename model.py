import numpy as np
import matplotlib.pyplot as plt
from numba import jit


class SimEnviron:
  def __init__(self):
    self.heading = 0.
    self.s = 0.
    self.old_s = 0.

    # self.maxspeed = 0.
    # self.speed = 0.
    # self.heading = 0.
    # self.dist = 0.
    # self.s = 0.

  def reward(self, steering):
    # Easy base case test
    reward = 0 - steering

    '''
    TODO: The more fun model will include steering as an effector of heading and thus side displacement s
    '''
    # # Handle throttle/brake effect on speed
    # self.speed += throttle - 0.1 * (self.speed)**2  # Approximate drag/friction
    # self.speed = np.maximum(self.speed, 0)              # Clamp min value to 0 (stopped)
    # self.maxspeed += 1 - 0.1 * (self.speed)**2
    #
    # # Handle steering effect on heading
    # self.heading += self.speed * steering   # Discrete approximation
    #
    # # Handle net effect on car's position
    # interval = self.speed * np.cos(self.heading)
    # self.dist += interval
    # self.s += self.speed * np.sin(self.heading)

    return reward


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class SigmoidNeuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return np.asscalar(sigmoid(total))


class SimpleNN:
  def __init__(self):
    # He initialization for hidden ReLus, Xavier init for output Tanh
    # Case X = [X1 X2 X3]: weights_h1 = np.random.randn(1, 3) * np.sqrt(2/3)
    w_h1 = np.random.randn(1, 2)[0] * np.sqrt(1 / 2)
    w_h2 = np.random.randn(1, 2)[0] * np.sqrt(1 / 2)
    w_out = np.random.randn(1, 2)[0] * np.sqrt(1 / 2)
    b_h1 = np.random.randn() * 0.1
    b_h2 = np.random.randn() * 0.1
    b_out = np.random.randn() * 0.1

    # The Neuron class here is from the previous section
    self.h1 = SigmoidNeuron(w_h1, b_h1)
    self.h2 = SigmoidNeuron(w_h2, b_h2)
    self.out = SigmoidNeuron(w_out, b_out)

    self.result_h1 = 0
    self.result_h2 = 0
    self.result_out = 0

  def feedforward(self, x):
    self.result_h1 = self.h1.feedforward(x)
    self.result_h2 = self.h2.feedforward(x)
    self.result_out = self.out.feedforward([self.result_h1, self.result_h2])

    # print("H1 = {}, H2 = {}, Out = {}".format(self.result_h1, self.result_h2, self.result_out))

    return self.result_out

  def train(self, x, error, learnrate, learnrate_bias=0):
    # Set bias learn rate == learn rate if not explicitly provided
    if learnrate_bias == 0:
      learnrate_bias = learnrate

    del_wout = error * (self.result_out * (1 - self.result_out))
    self.out.weights += learnrate * del_wout * np.array([self.result_h1, self.result_h2])
    self.out.bias += learnrate_bias * del_wout

    del_wh1 = del_wout * self.out.weights[0] * (self.result_h1 * (1 - self.result_h1))
    self.h1.weights += learnrate * del_wh1 * x
    self.h1.bias += learnrate_bias * del_wh1

    # print(self.out.weights)

    del_wh2 = del_wout * self.out.weights[1] * (self.result_h2 * (1 - self.result_h2))
    self.h2.weights += learnrate * del_wh2 * x
    self.h2.bias += learnrate_bias * del_wh2


'''
Simple test script for Neural Network functionality
'''

# Target Value
y = np.array(0.5)

# Constant Inputs (Only for testing, for now)
x = np.array([1, 2])

# Initialize network
network = SimpleNN()

# Initialize environment
sim = SimEnviron()

point_xs = []
point_ys = []

for i in range(100):
  # Calculate one gradient descent step for each weight
  steering = network.feedforward([sim.s, sim.heading]) - 0.5  # Sigmoid output centers at y == 0.5
  reward = sim.reward(steering)
  error = reward

  print('N{}: Steering: {}, \tReward: {}, \tError: {}'.format(i + 1, steering, reward, error))
  print('\tS: {},  \t\tHeading: {}'.format(sim.s, sim.heading))

  point_xs.append(i + 1)
  point_ys.append(steering)

  network.train(x, error, learnrate=1)

plt.plot(point_xs, point_ys)
plt.show()