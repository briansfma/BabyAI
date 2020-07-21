import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time


class SimEnviron:
  def __init__(self):
    self.heading = 0.
    self.s = 0.

    # self.maxspeed = 0.
    self.speed = 1
    self.dist = 0.

  def reward(self, steering):

    # Handle steering effect on heading
    self.heading += self.speed * steering   # Discrete approximation

    # Handle net effect on car's position
    self.s += self.speed * np.sin(self.heading)
    self.dist += self.speed * np.cos(self.heading)

    reward = self.heading*2*np.pi + self.s

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

# Initialize network
network = SimpleNN()

# Initialize environment
sim = SimEnviron()

# Time steps to run
STEP_COUNT = 200

point_t = []
point_steer = []
point_h = []
point_s = []
point_d = []

start = time.time()

old_reward = 1
for i in range(STEP_COUNT):
  input = np.array([sim.s, sim.heading])

  # Calculate one gradient descent step for each weight
  steering = network.feedforward(input) - 0.5  # Sigmoid output centers at y == 0.5
  reward = sim.reward(steering)
  error = -reward + 0.05*(reward - old_reward)

  old_reward = reward

  print('N{}: Steering: {}, \tReward: {}, \tError: {}'.format(i + 1, steering, reward, error))
  print('\tS: {},  \t\tHeading: {}'.format(sim.s, sim.heading))

  point_t.append(i + 1)
  point_steer.append(steering)
  point_h.append(sim.heading)
  point_s.append(sim.s)
  point_d.append(sim.dist)

  network.train(input, error, learnrate=1)


end = time.time()
print("Elapsed time for calculation: {}".format(end - start))


# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(point_t, point_steer)
ax1.set_xlabel('Time Step t')
ax1.set_ylabel('Steering (rad)')
ax2.plot(point_t, point_h)
ax2.set_xlabel('Time Step t')
ax2.set_ylabel('Heading (rad)')
ax3.plot(point_d, point_s)
ax3.set_xlabel('Distance d')
ax3.set_ylabel('Side displacement s')
plt.show()

# After plots show good, print/save weights and biases
print("\nFinal Network Weights and Biases")
print("H1: \n\tWeights: {}  \tBias: {}".format(network.h1.weights, network.h1.bias))
print("H2: \n\tWeights: {}  \tBias: {}".format(network.h2.weights, network.h2.bias))
print("Output: \n\tWeights: {}  \tBias: {}".format(network.out.weights, network.out.bias))
