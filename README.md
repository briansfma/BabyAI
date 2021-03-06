# BabyAI

Most recent AI developments have centered on generating inferences from vast amounts of data. But this actually isn't how humans learn! From birth, we start inferring from the briefest experiences, so even if we are not 100% correct, we move on, and we learn complex behaviors with relatively low investment. BabyAI attempts to recreate this phenomenon in the simplified context of learning to drive.

BabyAI won't train on batches of data -- each moment it spends in its environment is the data fed back to the neural network, which then must update and serve the next output in real-time. Performant code is a must, and the network architecture itself will have to be extremely fast. This is a heavy work in progress, if the project interests you I'd be honored to receive your contact or even contribution!

[//]: # (Image References)
[image1]: vJoystick1.png "vJoy working"
[image2]: FrameCapture1.png "Frame capture"
[image3]: Figure_1.png "Training (test)"

## Components (Dependencies)

* Python 3.5+
* Numpy
* Numba
* vJoy
* PyvJoy
* D3DShot
* Assetto Corsa

## Progress log

* Virtual joystick controller implemented (translates discrete neural network output to continuous X/Y-axis inputs)
![alt text][image1]
* Fast video frame capture implemented - resizes frames down to 160x90 for now as the CNN does not need much detail
![alt text][image2]
* Simple NN has been implemented that learns how to drive straight in real-time, via a simple reinforcement learning
scheme. The network makes distinct progress learning in three "stages":
    * Initial uncontrolled oscillation (approx t=0 to t=20 in this example)
    * Realizing allowed limits to steering angle (approx t=20 to t=50)
    * Slowly getting the hang of maintaining a straight line (t>50).
![alt text][image3]
While this behavior likely isn't robust against perturbations or curved target paths, we will add complexity to the
network as it becomes necessary.