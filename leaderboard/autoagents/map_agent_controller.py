# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the low level control a vehicle
    """

    def __init__(self, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        """Constructor method."""

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self.past_steering = 0
        self._lon_controller = PIDLongitudinalController(**args_longitudinal)
        self._lat_controller = PIDLateralController(offset, **args_lateral)

    def run_step(self, target_speed, current_speed, target_location, current_location, current_heading):
        """Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint at a given target_speed."""

        acceleration = self._lon_controller.run_step(target_speed, current_speed)
        current_steering = self._lat_controller.run_step(target_location, current_location, current_heading)

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PIDLongitudinalController():
    """PIDLongitudinalController implements longitudinal control using a PID."""

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.05):
        """Constructor method."""
        self._kp = K_P
        self._kd = K_D
        self._ki = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, current_speed):
        """Execute one step of longitudinal control to reach a given target speed."""

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._kp * error) + (self._kd * _de) + (self._ki * _ie), -1.0, 1.0)


class PIDLateralController():
    """PIDLateralController implements lateral control using a PID."""

    def __init__(self, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.05):
        """Constructor method."""
        self._kp = K_P
        self._kd = K_D
        self._ki = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, target_location, current_location, current_heading):
        """Execute one step of lateral control to steer the vehicle towards a certain waypoint."""

        # Get the ego's location and forward vector
        ego_loc = current_location
        v_vec = current_heading
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        w_loc = target_location
        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._kp * _dot) + (self._kd * _de) + (self._ki * _ie), -1.0, 1.0)
