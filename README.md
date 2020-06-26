# BabyAI

Most recent AI developments have centered on generating inferences from vast amounts of data. But this actually isn't how humans learn! From birth, we start inferring from the briefest experiences, so even if we are not 100% correct, we move on, and we learn complex behaviors with relatively low investment. BabyAI attempts to recreate this phenomenon in the simplified context of learning to drive.

BabyAI won't train on batches of data -- each moment it spends in its environment is the data fed back to the neural network, which then must update and serve the next output in real-time. Performant code is a must, and the network architecture itself will have to be extremely fast. This is a heavy work in progress, if the project interests you I'd be honored to receive your contact or even contribution!

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
* Fast video frame capture implemented - resizes frames down to 160x90 for now as the CNN does not need much detail
* Experimenting with the NN model now