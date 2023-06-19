#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import time
import os
import carla
import argparse

from queue import Queue, Empty

FPS = 20

################### User simulation configuration ####################
# 1) Choose the sensors
SENSORS = [
    [
        'CameraTest1',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 720, 'image_size_y': 1080, 'fov': 100,
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
    ],
    [
        'CameraTest2',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 720, 'image_size_y': 1080, 'fov': 100,
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    ]
]

# 2) Choose a weather
WEATHER = carla.WeatherParameters(
    sun_azimuth_angle=0.0, sun_altitude_angle=60.0,
    cloudiness=30.0, precipitation=0.0, precipitation_deposits=10.0, wetness=15.0,
    wind_intensity=10.0,
    fog_density=2.0, fog_distance=0.0, fog_falloff=0.0)

# 3) Choose a recorder file
RECORDER_PATH = "/home/glopez/Downloads/logs/test_Town10_2.log"
################# End user simulation configuration ##################

def create_folders(endpoint, sensor_ids):
    for sensor_id in sensor_ids:
        sensor_endpoint = f"{endpoint}/{sensor_id}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)

def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))

def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return

def get_ego_id(recorder_file):
    for line in recorder_file.split("\n"):
        if line.startswith(" Create ") and 'vehicle.lincoln.mkz_2017' in line:
            return int(line.split(" ")[2][:-1])

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-e', '--endpoint', required=True, help='Endpoint folder path')
    args = argparser.parse_args()

    active_sensors = []

    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        settings = world.get_settings()
        settings.fixed_delta_seconds = 1 / FPS
        settings.synchronous_mode = True
        world.apply_settings(settings)

        world.tick()

        # 1) Change the weather
        world.set_weather(WEATHER)

        world.tick()

        # 2) Start the recorder
        recorder_str = client.show_recorder_file_info(RECORDER_PATH, True)
        recording_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
        client.replay_file(RECORDER_PATH, 0, 0, get_ego_id(recorder_str), False)
        # with open("/home/glopez/Downloads/logs/test_Town10_2.txt", 'w') as fd:
        #     fd.write(recorder_str)

        world.tick()

        # 3) Link onto the ego vehicle
        hero = None
        while hero is None:
            print("Waiting for the ego vehicle...")
            possible_vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    hero = vehicle
                    break
            time.sleep(5)

        # 4) Create the sensors, and save their data into a queue
        create_folders(args.endpoint, [s[0] for s in SENSORS])
        sensor_queue = Queue()
        for sensor in SENSORS:

            # Extract the data from the sesor configuration
            sensor_id = sensor[0]
            attributes = sensor[1]
            blueprint_name = attributes.pop('bp', None)
            sensor_transform = carla.Transform(
                carla.Location(x=attributes.pop('x'), y=attributes.pop('y'), z=attributes.pop('z')),
                carla.Rotation(pitch=attributes.pop('pitch'), roll=attributes.pop('roll'), yaw=attributes.pop('yaw'))
            )

            # Get the blueprint and add the atrtibutes
            blueprint = blueprint_library.find(blueprint_name)
            for key, value in attributes.items():
                blueprint.set_attribute(str(key), str(value))

            # Create the sensors and its callback
            sensor = world.spawn_actor(blueprint, sensor_transform, hero)
            add_listener(sensor, sensor_queue, sensor_id)
            active_sensors.append(sensor)

        world.tick()

        # 5) Start getting the data
        start_time = world.get_snapshot().timestamp.elapsed_seconds
        sensor_amount = len(SENSORS)

        while True:
            current_time = world.get_snapshot().timestamp.elapsed_seconds
            current_duration = current_time - start_time
            if current_duration >= recording_duration:
                break

            # print(f">>>>>  Time: {round(current_duration, 3)} / {round(recording_duration, 3)}  <<<<<")
            completion =  format(round(current_duration / recording_duration * 100, 2), '3.2f')
            print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")

            # Get all the sensors data of that frame (and wait if needed)
            try:
                sensors_frame_data = {}
                frame = world.get_snapshot().frame
                while len(sensors_frame_data.keys()) < sensor_amount:
                    sensor_data = sensor_queue.get(True, 10.0)
                    if sensor_data[1] != frame:
                        continue  # Ignore previous frame data
                    sensors_frame_data[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))
            except Empty:
                raise ValueError("A sensor took too long to send their data")

            # Do something with the data
            for sensor_id, (frame, data) in sensors_frame_data.items():
                data.save_to_disk(f"{args.endpoint}/{sensor_id}/{frame}.png")

            world.tick()

    # End the simulation
    finally:
        # stop and remove cameras
        for sensor in active_sensors:
            sensor.stop()
            sensor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
