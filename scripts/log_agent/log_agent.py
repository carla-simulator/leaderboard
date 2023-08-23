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
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from agents.navigation.basic_agent import BasicAgent

from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.autoagents.human_agent import HumanAgent as HumanAgent_
from leaderboard.autoagents.human_agent import KeyboardControl as KeyboardControl_
from leaderboard.autoagents.human_agent import HumanInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
    return "HumanAgent"


class HumanAgent(HumanAgent_):

    TTC_THRESHOLD = 5

    def setup(self, path_to_conf_file):

        self.track = Track.SENSORS

        # Get the ego instance
        self._player = None

        for vehicle in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
            if vehicle.attributes['role_name'] == 'hero':
                self._player = vehicle
                break

        if self._player is None:
            raise ValueError("Couldn't find the ego vehicle")

        self._clock = pygame.time.Clock()

        self.agent_engaged = False
        self.camera_width = 1280
        self.camera_height = 720
        self._side_scale = 0.3
        self._left_mirror = False
        self._right_mirror = False

        self._hic = HumanInterface(
            self.camera_width,
            self.camera_height,
            self._side_scale,
            self._left_mirror,
            self._right_mirror
        )
        self._controller = KeyboardControl(self._player, self._global_plan_world_coord, path_to_conf_file)
        self._prev_timestamp = 0

        self._clock = pygame.time.Clock()

        # Attach obstacle sensor, calculate ttc
        world = CarlaDataProvider.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.obstacle')
        blueprint.set_attribute('distance', '30')
        self._obstacle_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._player)
        self._obstacle_sensor.listen(lambda event: self._update_obstacle(event))
        self._ttc = self.TTC_THRESHOLD*2 # Time to collision, default to safe value

    
    def _update_obstacle(self, event):

        if "vehicle" not in event.other_actor.type_id and "walker" not in event.other_actor.type_id:
            return

        v_obstacle = CarlaDataProvider.get_velocity(event.other_actor)
        v_ego = CarlaDataProvider.get_velocity(self._player)
        ttc = event.distance / max((v_ego - v_obstacle), 0.01) # Avoid division by zero
        self._ttc = ttc

    
    def run_step(self, input_data, timestamp):

        # If ttc is small enough, display warning
        if self._ttc < self.TTC_THRESHOLD:
        
            text = "Too Close!"
            pos = (self.camera_width // 2, self.camera_height // 1.5)
            font = pygame.font.Font(pygame.font.get_default_font(), 16)
            surface = pygame.Surface((120, 50)) # width and height
            surface.fill((0, 0, 0, 0))
            text_texture = font.render(text, True, (255, 255, 255))
            surface.blit(text_texture, (22, 18))
            surface.set_alpha(220)
            self._hic._display.blit(surface, pos)
            pygame.display.flip()

            self._ttc = self.TTC_THRESHOLD*2

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
        self._agent = BasicAgent(player, 30, {'distance_ratio': 0.3, 'base_min_distance': 2})
        self._agent.follow_speed_limits()
        self._agent.ignore_traffic_lights()
        self._agent.ignore_stop_signs()
        self._agent.ignore_vehicles()

        # route_keypoints = CarlaDataProvider.get_ego_route()
        # grp = CarlaDataProvider.get_global_route_planner()
        # route = []
        # for i in range(len(route_keypoints) - 1):
        #     waypoint = route_keypoints[i][0].location
        #     waypoint_next = route_keypoints[i + 1][0]
        #     interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        #     for wp, connection in interpolated_trace:
        #         route.append((wp, connection))

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

        max_speed = self._player.get_speed_limit()
        current_speed = self._player.get_velocity().length()
        if keys[K_DOWN] or keys[K_s]:  # Braking takes priority
            self._control.throttle = 0.0
            self._control.brake = 1.0
        elif 3.6 * current_speed > max_speed:  # Ensure the max speed is never surpassed
            self._control.throttle = agent_control.throttle
            self._control.brake = agent_control.brake
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
            self._agent.set_offset(0.5)
            self._offset_tick = 20
        self._offset_tick = max(self._offset_tick - 1, 0)

        # Left Lane change
        if keys[K_r] and self._offset_tick == 0:
            self._agent_active = True
            self._agent.set_offset(-3.0)
            self._offset_tick = 20
        self._offset_tick = max(self._offset_tick - 1, 0)

        # Right Lane change
        if keys[K_t] and self._offset_tick == 0:
            self._agent_active = True
            self._agent.set_offset(3.0)
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
            'control': {
                'throttle': round(self._control.throttle, 2),
                'steer': round(self._control.steer, 2),
                'brake': round(self._control.brake, 2),
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            },
            'state': {
                'velocity': {
                    'x': round(velocity.x, 1),
                    'y': round(velocity.y, 1),
                    'z': round(velocity.z, 1),
                    'value': round(velocity.length(), 1)
                },
                'angular_velocity': {
                    'x': round(angular_velocity.x, 1),
                    'y': round(angular_velocity.y, 1),
                    'z': round(angular_velocity.z, 1),
                    'value': round(angular_velocity.length(), 1)
                },
                'acceleration': {
                    'x': round(acceleration.x, 1),
                    'y': round(acceleration.y, 1),
                    'z': round(acceleration.z, 1),
                    'value': round(acceleration.length(), 1)
                },
                'transform': {
                    'x': round(transform.location.x, 1),
                    'y': round(transform.location.y, 1),
                    'z': round(transform.location.z, 1),
                    'roll': round(transform.rotation.roll, 1),
                    'pitch': round(transform.rotation.pitch, 1),
                    'yaw': round(transform.rotation.yaw, 1)
                }
            }
        }

        self._log_data['records'].append(new_record)
