# Copyright (c) 2021 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
    from pygame.locals import K_e
    from pygame.locals import K_f
    from pygame.locals import K_r
    from pygame.locals import K_t
    from pygame.locals import K_g
except ImportError:
    raise RuntimeError(
        "cannot import pygame, make sure pygame package is installed")

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.human_agent import HumanAgent as HumanAgent_
from leaderboard.autoagents.human_agent import KeyboardControl as KeyboardControl_
from leaderboard.autoagents.human_agent import HumanInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
    return "HumanAgent"


class LogHumanInterface(HumanInterface):
    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # Process sensor data
        image_center = input_data["Center"][1][:, :, -2::-1]
        self._surface = pygame.surfarray.make_surface(
            image_center.swapaxes(0, 1))

        # Add the left mirror
        if self._left_mirror:
            image_left = input_data["Left"][1][:, :, -2::-1]
            left_surface = pygame.surfarray.make_surface(
                image_left.swapaxes(0, 1))
            self._surface.blit(left_surface, (0, 0))

        # Add the right mirror
        if self._right_mirror:
            image_right = input_data["Right"][1][:, :, -2::-1]
            right_surface = pygame.surfarray.make_surface(
                image_right.swapaxes(0, 1))
            self._surface.blit(
                right_surface, ((1 - self._scale) * self._width, 0))

        # Show scenario name
        scenario_name = CarlaDataProvider.get_latest_scenario()
        text = "{}".format(scenario_name)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_texture = font.render(text, True, (255, 255, 255))
        self._surface.blit(
            text_texture, (self._width // 2 - 80, self._height - 20))

        # Show ttc warning
        text = input_data["TooCloseText"]
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_texture = font.render(text, True, (255, 255, 255))
        self._surface.blit(
            text_texture, (self._width // 2 - 50, self._height - 85))

        # Show passive lane change warning
        text = input_data["LCWarningText"]
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_texture = font.render(text, True, (255, 255, 255))
        self._surface.blit(
            text_texture, (self._width // 2 - 82, self._height - 55))

        # Display image
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()


class HumanAgent(HumanAgent_):

    TTC_THRESHOLD = 5  # seconds
    WAYPOINT_MIN_GAP = 10  # meters
    ROUTE_LC_WARNING_DISTANCE = 100  # meters

    def setup(self, path_to_conf_file):

        self.track = Track.SENSORS

        # Get the ego instance
        self._player = None

        for vehicle in CarlaDataProvider.get_world().get_actors().filter("vehicle.*"):
            if vehicle.attributes["role_name"] == "hero":
                self._player = vehicle
                break

        if self._player is None:
            raise ValueError("Couldn't find the ego vehicle")

        self._clock = pygame.time.Clock()

        self.agent_engaged = False
        self.camera_width = 1280
        self.camera_height = 720
        self._side_scale = 0.3
        self._left_mirror = True
        self._right_mirror = True

        self._hic = LogHumanInterface(
            self.camera_width,
            self.camera_height,
            self._side_scale,
            self._left_mirror,
            self._right_mirror,
        )
        self._controller = KeyboardControl(
            self._player, self._global_plan_world_coord, path_to_conf_file
        )
        self._prev_timestamp = 0

        self._clock = pygame.time.Clock()

        # Attach obstacle sensor, calculate ttc (time to collision)
        world = CarlaDataProvider.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.obstacle")
        blueprint.set_attribute("distance", "30")
        self._obstacle_sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self._player
        )
        self._obstacle_sensor.listen(
            lambda event: self._update_obstacle(event))
        self._ttc = self.TTC_THRESHOLD * 2  # Time to collision, default to safe value

        # Get planned route
        wps_queue = self._controller._agent._local_planner._waypoints_queue
        self._lc_wps = [
            x
            for x in wps_queue
            if x[1] in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]
        ]

        # Deduplicate waypoints
        i = 0
        while i < len(self._lc_wps) - 1:
            if (
                self._lc_wps[i][0].transform.location.distance(
                    self._lc_wps[i + 1][0].transform.location
                )
                < self.WAYPOINT_MIN_GAP
            ):
                del self._lc_wps[i + 1]
            else:
                i += 1

    def _update_obstacle(self, event):

        if (
            "vehicle" not in event.other_actor.type_id
            and "walker" not in event.other_actor.type_id
        ):
            return

        v_obstacle = CarlaDataProvider.get_velocity(event.other_actor)
        v_ego = CarlaDataProvider.get_velocity(self._player)
        # Avoid division by zero
        ttc = event.distance / max((v_ego - v_obstacle), 0.01)
        self._ttc = ttc

    def run_step(self, input_data, timestamp):

        if self._ttc < self.TTC_THRESHOLD:
            self._ttc = self.TTC_THRESHOLD * 2
            input_data["TooCloseText"] = "Too Close!"
        else:
            input_data["TooCloseText"] = ""

        # If a passiv lane change is near, display warning
        loc_ego = CarlaDataProvider.get_location(self._player)
        curr_lc_wps = []
        i = 0
        while i < len(self._lc_wps):
            dist = loc_ego.distance(self._lc_wps[i][0].transform.location)
            if (
                loc_ego.distance(self._lc_wps[i][0].transform.location)
                < self.WAYPOINT_MIN_GAP
            ):
                del self._lc_wps[i]  # Passed, remove it
            elif dist < self.ROUTE_LC_WARNING_DISTANCE:
                curr_lc_wps.append(self._lc_wps[i])
                i += 1
            else:
                i += 1

        left_lc_dists = [
            loc_ego.distance(wp[0].transform.location)
            for wp in curr_lc_wps
            if wp[1] == RoadOption.CHANGELANELEFT
        ]
        right_lc_dists = [
            loc_ego.distance(wp[0].transform.location)
            for wp in curr_lc_wps
            if wp[1] == RoadOption.CHANGELANERIGHT
        ]
        text = ""
        left_lc_string = ", ".join([f"{int(x)}m" for x in left_lc_dists])
        right_lc_string = ", ".join([f"{int(x)}m" for x in right_lc_dists])
        if len(left_lc_dists) > 0:
            text += f"Left: {left_lc_string}"
        if len(right_lc_dists) > 0:
            text += f" Right: {right_lc_string}"
        input_data["LCWarningText"] = text

        return super().run_step(input_data, timestamp)


class KeyboardControl(KeyboardControl_):

    def __init__(self, player, route, path_to_conf_file):
        self._player = player
        self._route = route
        self._agent_active = False
        self._controller_tick = 0
        self._offset_tick = 0
        self._left_lane_tick = 0
        self._right_lane_tick = 0

        # Add an agent that follows the route to the ego
        self._agent = BasicAgent(
            player, 30, {"distance_ratio": 0.3, "base_min_distance": 2}
        )
        self._agent.follow_speed_limits()
        self._agent.ignore_traffic_lights()
        self._agent.ignore_stop_signs()
        self._agent.ignore_vehicles()

        route = CarlaDataProvider.get_ego_route()
        tmap = CarlaDataProvider.get_map()
        waypoint_route = []
        for transform, road_option in route:
            wp = tmap.get_waypoint(transform.location)
            waypoint_route.append([wp, road_option])
        self._agent.set_global_plan(waypoint_route)

        super().__init__(path_to_conf_file)

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        agent_control = self._agent.run_step()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_DOWN] or keys[K_s]:  # Braking takes priority
            self._control.throttle = 0.0
            self._control.brake = 1.0
        elif keys[K_UP] or keys[K_w]:  # Accelerate
            self._control.throttle = 1.0
            self._control.brake = 0.0
        else:  # Idle
            self._control.throttle = 0.0
            self._control.brake = 0.0

        # Controller
        if keys[K_e] and self._controller_tick == 0:
            if self._agent_active:
                print("Deactivating the controller")
            else:
                print("Activating the controller")
            self._controller_tick = 20
            self._agent_active = not self._agent_active
        self._controller_tick = max(self._controller_tick - 1, 0)

        # LaneInvadingTurn
        if keys[K_f] and self._offset_tick == 0:
            self._agent_active = True
            self._agent.set_offset(0.7)
            self._offset_tick = 20
        self._offset_tick = max(self._offset_tick - 1, 0)

        # Left Lane change
        if keys[K_r] and self._offset_tick == 0:
            self._agent_active = True
            self._agent.set_offset(-3.2)
            self._offset_tick = 20
        self._offset_tick = max(self._offset_tick - 1, 0)

        # Right Lane change
        if keys[K_t] and self._offset_tick == 0:
            self._agent_active = True
            self._agent.set_offset(3.2)
            self._offset_tick = 20
        self._offset_tick = max(self._offset_tick - 1, 0)

        # Remove offset
        if keys[K_g] and self._offset_tick == 0:
            self._agent_active = True
            self._agent.set_offset(0)
            self._offset_tick = 20
        self._offset_tick = max(self._offset_tick - 1, 0)

        if not self._agent_active:
            steer_increment = 3e-4 * milliseconds
            if keys[K_LEFT] or keys[K_a]:
                self._steer_cache -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                self._steer_cache += steer_increment
            else:
                self._steer_cache = 0.0
        else:
            self._steer_cache = agent_control.steer

        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _record_control(self):

        velocity = self._player.get_velocity()
        angular_velocity = self._player.get_angular_velocity()
        acceleration = self._player.get_acceleration()
        transform = self._player.get_transform()

        new_record = {
            "control": {
                "throttle": round(self._control.throttle, 2),
                "steer": round(self._control.steer, 2),
                "brake": round(self._control.brake, 2),
                "hand_brake": self._control.hand_brake,
                "reverse": self._control.reverse,
                "manual_gear_shift": self._control.manual_gear_shift,
                "gear": self._control.gear,
            },
            "state": {
                "velocity": {
                    "x": round(velocity.x, 1),
                    "y": round(velocity.y, 1),
                    "z": round(velocity.z, 1),
                    "value": round(velocity.length(), 1),
                },
                "angular_velocity": {
                    "x": round(angular_velocity.x, 1),
                    "y": round(angular_velocity.y, 1),
                    "z": round(angular_velocity.z, 1),
                    "value": round(angular_velocity.length(), 1),
                },
                "acceleration": {
                    "x": round(acceleration.x, 1),
                    "y": round(acceleration.y, 1),
                    "z": round(acceleration.z, 1),
                    "value": round(acceleration.length(), 1),
                },
                "transform": {
                    "x": round(transform.location.x, 1),
                    "y": round(transform.location.y, 1),
                    "z": round(transform.location.z, 1),
                    "roll": round(transform.rotation.roll, 1),
                    "pitch": round(transform.rotation.pitch, 1),
                    "yaw": round(transform.rotation.yaw, 1),
                },
            },
        }

        self._log_data["records"].append(new_record)
