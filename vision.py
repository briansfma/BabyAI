import numpy as np
import matplotlib.pyplot as plt
import time
import d3dshot
from numba import jit


TARGET_INPUT_WIDTH = 160    # pixels


class FrameGrabber:
    @jit
    def __init__(self):
        # Init frame grabber
        self.f = d3dshot.create(capture_output="numpy")
        print("Found primary display. Resolution: {} x {}"
                .format(self.f.display.resolution[0],
                        self.f.display.resolution[1]))

        # Calc scale factor and offset for image resize
        self.factor = self.f.display.resolution[0] // TARGET_INPUT_WIDTH
        self.offset = self.factor // 2

        self.f.capture()
        time.sleep(1)   # Pause for init time

    # Gets RGB screen capture and resizes it to serve to neural net.
    @jit
    def get_processed_frame(self):
        raw = self.f.get_latest_frame()

        return raw[self.offset::self.factor, self.offset::self.factor, :]

    @jit
    def __del__(self):
        self.f.stop()
        print("Framegrabber stopped")


@jit
def testfunction():
    d = FrameGrabber()

    i = 0
    while True:
        i+=1
        frame = d.get_processed_frame()
        print("Frame #{}".format(i))

        # # Visualization
        # plt.imshow(frame)
        # plt.show()

testfunction()

