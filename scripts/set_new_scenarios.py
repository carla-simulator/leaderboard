import argparse
import json
import math
from argparse import RawTextHelpFormatter
from leaderboard.utils.checkpoint_tools import fetch_dict
import carla
import os

SCENARIO_COLOR = {
    "Scenario1": [carla.Color(255, 0, 0), "Red"],        # Red
    "Scenario2": [carla.Color(0, 255, 0), "Green"],        # Green
    "Scenario3": [carla.Color(0, 0, 255), "Blue"],        # Blue
    "Scenario4": [carla.Color(255, 100, 0), "Orange"],      # Orange
    "Scenario5": [carla.Color(0, 255, 100), "Blueish green"],      # Blueish green
    "Scenario6": [carla.Color(100, 0, 255), "Purple"],      # Purple
    "Scenario7": [carla.Color(255, 100, 255), "Pink"],    # Pink
    "Scenario8": [carla.Color(255, 255, 100), "Yellow"],    # Yellow
    "Scenario9": [carla.Color(100, 255, 255), "Light Blue"],    # Light Blue
    "Scenario10": [carla.Color(100, 100, 100), "Gray"]   # Gray
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
    # if distance < TRIGGER_THRESHOLD:
        world.debug.draw_point(scenario_waypoint.transform.location + carla.Location(z=1),
                               size=float(0.15), color=carla.Color(255, 0, 0))
    else:
        world.debug.draw_point(scenario_waypoint.transform.location + carla.Location(z=1),
                               size=float(0.15), color=carla.Color(0, 255, 0))

def save_from_wp(endpoint, wp):
    """
    Creates a mini json with the data from the scenario location.
    used to copy paste it to the .json
    """
    with open(endpoint, mode='w') as fd:

        entry = {}
        transform = {
            "x": str(round(wp.transform.location.x, 2)),
            "y": str(round(wp.transform.location.y, 2)),
            "z": "1.0",
            "yaw": str(round(wp.transform.rotation.yaw, 0)),
            "pitch": str(round(wp.transform.rotation.pitch, 0)),
        }
        entry["transform"] = transform
        entry["other_actors"] = {}
        json.dump(entry, fd, indent=4)

def save_from_dict(endpoint, wp):
    """
    Creates a mini json with the data from the scenario waypoint.
    used to copy paste it to the .json
    """
    with open(endpoint, mode='w') as fd:

        entry = {}
        transform = {
            "x": str(round(float(wp["x"]), 2)),
            "y": str(round(float(wp["y"]), 2)),
            "z": "1.0",
            "yaw": str(round(float(wp["yaw"]), 0)),
            "pitch": str(round(float(wp["pitch"]), 0)),
        }
        entry["transform"] = transform
        entry["other_actors"] = {}
        json.dump(entry, fd, indent=4)

def draw_scenarios(world, scenarios, args):
    """
    Draws all the points related to args.scenarios
    """
    z = 3

    if scenarios["scenario_type"] in args.scenarios:
        number = float(scenarios["scenario_type"][8:])
        color = SCENARIO_COLOR[scenarios["scenario_type"]][0]

        event_list = scenarios["available_event_configurations"]
        # number_of_scenarios = len(event_list)
        # for event in event_list:
        for i in range(len(event_list)):
            event = event_list[i]
            _waypoint = event['transform']  # trigger point of this scenario
            location = carla.Location(float(_waypoint["x"]), float(_waypoint["y"]), float(_waypoint["z"]))

            scenario_location = location + carla.Location(z=number / z)
            world.debug.draw_point(scenario_location, size=float(0.15), color=color)

            if args.debug:
                save_from_dict(args.endpoint, _waypoint)
                spectator = world.get_spectator()
                spectator.set_transform(carla.Transform(location + carla.Location(z=50),
                                                            carla.Rotation(pitch=-90)))
                print(" Scenario [{}/{}]. Press Enter for the next scenario".format(i+1, len(event_list)))
                input()

def modify_junction_scenarios(world, scenarios, args):
    """
    Used to move scenario trigger points a certain distance to the front
    """
    DISTANCE = 2

    if scenarios["scenario_type"] in args.scenarios:
        event_list = scenarios["available_event_configurations"]

        for event in event_list:
            _waypoint = event['transform']  # trigger point of this scenario
            location = carla.Location(float(_waypoint["x"]), float(_waypoint["y"]), float(_waypoint["z"]))
            rotation = carla.Rotation(float(0), float(_waypoint["pitch"]), float(_waypoint["yaw"]))
            print("Init location: {}. Init rotation: {}".format(location, rotation))
            world.debug.draw_point(location, size=float(0.15), color=carla.Color(0, 255, 255))

            # Used to get the S7-10 points
            new_waypoint = world.get_map().get_waypoint(location)
            scenario_waypoint = new_waypoint.next(DISTANCE)[0]

            apart_enough(world, _waypoint, scenario_waypoint)

            print("Last location: {}. Last rotation: {}".format(
                scenario_waypoint.transform.location,
                scenario_waypoint.transform.rotation))

            save_from_wp(args.endpoint, scenario_waypoint)

            spectator = world.get_spectator()
            spectator.set_transform(carla.Transform(scenario_waypoint.transform.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))

            input("Press Enter to continue...\n")

def main():
    """
    Used to help with the visualization of the scenario trigger points, as well as its
    modifications.
        --town: Selects the town
        --scenario: The scenario that will be printed. Use the number of the scenarios 1 2 3 ...
        --modify: Used to modify the trigger_points of the given scenario in args.scenarios.
          debug is auto-enabled here. It will be shown red if they aren't apart enough or green, if they are
        --debug: If debug is selected, the points will be shown one by one, and stored at --endpoint,
    in case some copy-pasting is required.
    """

    # general parameters
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--town', default='Town09')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inipoint', default="../data/Tests.json")
    parser.add_argument('--endpoint', default="set_new_scenarios.json")
    parser.add_argument('--scenarios', nargs='+', default='Scenario7')
    parser.add_argument('--modify', action='store_true')
    parser.add_argument('--host', default='localhost', help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')

    args = parser.parse_args()

    # 0) Set the world
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(20)
    world = client.load_world(args.town)

    settings = world.get_settings()
    settings.fixed_delta_seconds = None
    settings.synchronous_mode = False
    world.apply_settings(settings)

    # 1) Read the json file
    data = fetch_dict(args.inipoint)
    data = data["available_scenarios"][0]

    town_data = data[args.town]

    new_args_scenario = []
    for ar_sc in args.scenarios:
        new_args_scenario.append("Scenario" + ar_sc)
    args.scenarios = new_args_scenario

    for scenarios in town_data:

        if args.modify:
            modify_junction_scenarios(world, scenarios, args)
        else:
            draw_scenarios(world, scenarios, args)

    print(" ---------------------------- ")
    for ar_sc in args.scenarios:
        print(" {} is colored as {}".format(ar_sc, SCENARIO_COLOR[ar_sc][1]))
    print(" ---------------------------- ")


if __name__ == '__main__':
    main()
