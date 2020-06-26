import sys
import os
import platform
import numpy as np
from numba import jit

from control import Controller


# @jit
def main():
    # Start controller helper
    veh_controller = Controller()

    # # Write/read file pipe
    # file = open("test_pipe", "r")
    #
    # for i in range(1000000):
    #     file.write("01_000001.png,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,True,27.77,0.0,0.79601,0.0,0.0")
    #     response = file.readlines(1)


if __name__ == "__main__":
    print("Starting BabyAI")
    main()