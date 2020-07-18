import sys
import pyvjoy

# Parameters: # of steps for steering, throttle and brake, response speed
HALF_AXIS = 16384
FULL_AXIS = 32768
BUTTON_1 = 1
BUTTON_2 = 2

STEER_RES = 40
THROT_RES = 4
STEER_SPEED = 100
THROT_SPEED = 2000


class Controller:
    def __init__(self):
        # Raw input counters
        self.x_key = 0      # Range: -40 to +40
        self.y_key = 0      # Range: -4 to +4
        self.w_key = None   # Range: None, True or False

        # History values for x and y axes
        self.last_x = HALF_AXIS  # center
        self.last_y = 0

        # Attempt to capture vJoy device (must be intsalled to Windows already)
        self.j = 0
        try:
            self.j = pyvjoy.VJoyDevice(1)
            print("vJoy device connected")
        except:
            sys.exit("Could not create virtual joystick")

    def simple_controller_output(self, target, old_val, speed):
        if target > old_val:
            output = old_val + speed
            if output > target: return target
            else: return output
        elif target < old_val:
            output = old_val - speed
            if output < target: return target
            else: return output
        else:
            return target

    def update(self, x_command, y_command):
        # Clamp x and y commanded values
        x_command += self.x_key
        self.x_key = x_command if x_command < STEER_RES else STEER_RES
        self.x_key = x_command if x_command > -STEER_RES else -STEER_RES
        y_command += self.y_key
        self.y_key = y_command if y_command < THROT_RES else THROT_RES
        self.y_key = y_command if y_command > -THROT_RES else -THROT_RES

        # print("Var status: x: {}, y: {}, w: {}".format(x_key, y_key, w_key))

        # calculate target output
        x_target = HALF_AXIS + HALF_AXIS // STEER_RES * self.x_key
        y_target = FULL_AXIS // THROT_RES * self.y_key

        # calculate output to pass to joystick (avoid large instantaneous jumps)
        x_output = self.simple_controller_output(x_target, self.last_x, STEER_SPEED)
        self.last_x = x_output
        y_output = self.simple_controller_output(y_target, self.last_y, THROT_SPEED)
        self.last_y = y_output

        y_split_y = y_output if y_output > 0 else 0
        y_split_z = -y_output if y_output < 0 else 0

        # Update the vJoy device with new axis and button values
        try:
            # The 'efficient' method from vJoy's docs - set multiple values at once
            # j.data
            # >>> < pyvjoy._sdk._JOYSTICK_POSITION_V2 at 0 x.... >

            # Fill X and Y data to send
            self.j.data.wAxisX = x_output
            self.j.data.wAxisY = y_split_y
            self.j.data.wAxisZ = y_split_z

            # Set button presses
            # How to use: buttons are binary position, so
            # 19 = 00010011 in binary, giving buttons 1, 2, and 5
            # j.data.lButtons = 19
            if self.w_key is None:
                self.j.data.lButtons = 0            # no buttons pressed
            elif self.w_key:
                self.j.data.lButtons = BUTTON_1     # only button 1 pressed
                self.w_key = None
            else:
                self.j.data.lButtons = BUTTON_2     # only button 2 pressed
                self.w_key = None

            # send data to vJoy device
            self.j.update()

            # print("Axes: x: {}, y: {}, z: {}, w: {}".format(
            #         j.data.wAxisX, j.data.wAxisY, j.data.wAxisZ, w_key))
        except:
            sys.exit("Virtual joystick lost, or not found")