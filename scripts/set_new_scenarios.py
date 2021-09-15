import argparse
from argparse import RawTextHelpFormatter
import math
import os
import sys

import carla

from leaderboard.utils.checkpoint_tools import fetch_dict

SCENARIO_COLOR = {
    "Scenario1": [carla.Color(255, 0, 0), "Red"],
    "Scenario2": [carla.Color(0, 255, 0), "Green"],
    "Scenario3": [carla.Color(0, 0, 255), "Blue"],
    "Scenario4": [carla.Color(255, 100, 0), "Orange"],
    "Scenario5": [carla.Color(0, 255, 100), "Blueish green"],
    "Scenario6": [carla.Color(100, 0, 255), "Purple"],
    "Scenario7": [carla.Color(255, 100, 255), "Pink"],
    "Scenario8": [carla.Color(255, 255, 100), "Yellow"],
    "Scenario9": [carla.Color(100, 255, 255), "Light Blue"], 
    "Scenario10": [carla.Color(100, 100, 100), "Gray"]
}

def apart_enough(world, _waypoint, scenario_waypoint):
    """
    Uses the same condition as in route_scenario to see if they will
    be differentiated
    """
    TRIGGER_THRESHOLD = 4.0
    TRIGGER_ANGLE_THRESHOLD = 10

    dx = float(_waypoint["x"]) - scenario_waypoint.transform.location.x
    dy = float(_waypoint["y"]) - scenario_waypoint.transform.location.y
    distance = math.sqrt(dx * dx + dy * dy)

    dyaw = float(_waypoint["yaw"]) - scenario_waypoint.transform.rotation.yaw
    dist_angle = math.sqrt(dyaw * dyaw)

    if distance < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
        world.debug.draw_point(scenario_waypoint.transform.location + carla.Location(z=1),
                               size=float(0.15), color=carla.Color(255, 0, 0))
    else:
        world.debug.draw_point(scenario_waypoint.transform.location + carla.Location(z=1),
                               size=float(0.15), color=carla.Color(0, 255, 0))

def draw_scenarios(world, scenarios, args):
    """
    Draws all the points related to args.scenarios
    """
    z = 3

    tmap = world.get_map()

    if scenarios["scenario_type"] in args.scenarios:
        number = float(scenarios["scenario_type"][8:])
        color = SCENARIO_COLOR[scenarios["scenario_type"]][0]

        event_list = scenarios["available_event_configurations"]
        for i in range(len(event_list)):
            event = event_list[i]
            _waypoint = event['transform']  # trigger point of this scenario
            location = carla.Location(float(_waypoint["x"]), float(_waypoint["y"]), float(_waypoint["z"]))
            yaw = float(_waypoint["yaw"])

            scenario_location = location + carla.Location(z=number / z)
            world.debug.draw_point(scenario_location, size=float(0.15), color=color)
            world.debug.draw_string(scenario_location + carla.Location(z=0.1), text=str(i+1), color=carla.Color(0, 0, 0), life_time=1000)

            if args.debug:
                spectator = world.get_spectator()
                spectator.set_transform(carla.Transform(location + carla.Location(z=50),
                                                            carla.Rotation(pitch=-90)))

                if args.modify:
                    wp = tmap.get_waypoint(location)
                    new_transform = wp.previous(5)[0].transform
                    new_location = new_transform.location
                    new_yaw = new_transform.rotation.yaw

                    input(" Scenario [{}/{}] at (x={}, y={}, z={}, yaw={}). Press Enter to continue".format(
                        i+1, len(event_list), round(new_location.x,1), round(new_location.y,1), round(new_location.z,1), round(new_yaw,1)))
                else:
                    input(" Scenario [{}/{}] at (x={}, y={}, z={}, yaw={}). Press Enter to continue".format(
                        i+1, len(event_list), round(location.x,1), round(location.y,1), round(location.z,1), round(yaw,1)))
        world.wait_for_tick()

def print_final_message(args):
    """Prints the final message about the scenario colors"""
    print("\n ---------------------------- ")
    end_color= "\x1b[0m"
    for ar_sc in args.scenarios:
        color = SCENARIO_COLOR[ar_sc][0]
        true_color = "\x1b[38;2;" + str(color.r) +";" + str(color.g) + ";" + str(color.b) + "m"
        print(" {}{} is colored as {}{}".format(true_color, ar_sc, SCENARIO_COLOR[ar_sc][1], end_color))
    print(" (Colors shown are just orientative, as they might not correspond with the CARLA view)")
    print(" ---------------------------- \n")

def main():
    """Used to help with the visualization of the scenario trigger points"""
    # general parameters
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')

    parser.add_argument('--file-path', required=True,
                        help='path to the .json file containing the scenarios')
    parser.add_argument('--scenarios', nargs='+', required=True,
                        help='scenarios to be checked. To check multiple scenarios separate them by spaces (example "1 3 6 9")')

    parser.add_argument('--debug', action='store_true',
                        help='Scenarios are printed one by one, and additional information is given')
    parser.add_argument('--modify', action='store_true',
                        help='Create new scenarios behind the given ones')
    parser.add_argument('--load-town',
                        help='Loads a specific town on which to check the scenarios (example "Town01")')
    parser.add_argument('--reload', action='store_true',
                        help='Loads a specific town on which to check the scenarios (example "Town01")')
    args = parser.parse_args()

    if args.modify:
        args.debug = True

    if args.load_town and args.reload:
        raise ValueError("'load_town' and 'reload' can't be active at the same time")

    try:
        # Set the world
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(20)
        if args.load_town:
            world = client.load_world(args.load_town)
        elif args.reload:
            world = client.reload_world()
        else:
            world = client.get_world()
        args.town = world.get_map().name.split("/")[-1]

        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # Read the json file
        data = fetch_dict(args.file_path)
        data = data["available_scenarios"][0]
        args.scenarios = ["Scenario" + ar_sc for ar_sc in args.scenarios]

        for scenarios in data[args.town]:
            draw_scenarios(world, scenarios, args)
        print_final_message(args)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
