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
        'CameraTest',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 720, 'image_size_y': 1080, 'fov': 100,
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
    ],
    [
        'LidarTest',
        {
            'bp': 'sensor.lidar.ray_cast',
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    ],
    [
        'SemanticLidarTest',
        {
            'bp': 'sensor.lidar.ray_cast_semantic',
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    ],
    [
        'RADARTest',
        {
            'bp': 'sensor.other.radar',
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    ],
    [
        'GnssTest',
        {
            'bp': 'sensor.other.gnss',
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    ],
    [
        'IMUTest',
        {
            'bp': 'sensor.other.imu',
            'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        }
    ]
]

# 2) Choose a weather
WEATHER = carla.WeatherParameters(
    sun_azimuth_angle=60.0, sun_altitude_angle=40.0,
    cloudiness=50.0, precipitation=0.0, precipitation_deposits=40.0, wetness=15.0,
    wind_intensity=10.0,
    fog_density=5.0, fog_distance=0.0, fog_falloff=0.0)

# 3) Choose a recorder file
RECORDER_INFO = [
    [f"{os.getcwd()}/records/Town10_1.log", 80, 5],
    [f"{os.getcwd()}/records/Town10_2.log", 55, 2],
    [f"{os.getcwd()}/records/Town10_4.log", 30, 15],
    [f"{os.getcwd()}/records/Town10_3.log", 0, 40],
    [f"{os.getcwd()}/records/Town10_2.log", 80, 20]
]
################# End user simulation configuration ##################

def create_folders(endpoint, sensors):
    for sensor_id, sensor_bp in sensors:
        sensor_endpoint = f"{endpoint}/{sensor_id}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)

        if 'gnss' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Altitude,Latitude,Longitude\n"
                data_file.write(data_txt)

        if 'imu' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,Gyroscope X,Gyroscope Y,Gyroscope Z\n"
                data_file.write(data_txt)

def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))

def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return

def get_ego_id(recorder_file):
    for line in recorder_file.split("\n"):
        if line.startswith(" Create ") and 'vehicle.lincoln.mkz_2017' in line:
            return int(line.split(" ")[2][:-1])

def save_data_to_disk(sensor_id, frame, data, endpoint):
    """
    Saves the sensor data into file:
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'
    """
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.png"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.LidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.SemanticLidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.RadarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.csv"
        data_txt = f"Altitude,Azimuth,Depth,Velocity\n"
        for point_data in data:
            data_txt += f"{point_data.altitude},{point_data.azimuth},{point_data.depth},{point_data.velocity}\n"
        with open(sensor_endpoint, 'w') as data_file:
            data_file.write(data_txt)

    elif isinstance(data, carla.GnssMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
            data_file.write(data_txt)

    elif isinstance(data, carla.IMUMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{data.accelerometer.x},{data.accelerometer.y},{data.accelerometer.z},{data.compass},{data.gyroscope.x},{data.gyroscope.y},{data.gyroscope.z}\n"
            data_file.write(data_txt)

    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

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
        client.set_timeout(30.0)
        world = client.get_world()

        for recorder_path, recorder_start, recorder_duration in RECORDER_INFO:
            # 0) 
            print(f"Running: {recorder_path}")
            endpoint = f"{args.endpoint}/{recorder_path.split('/')[-1][:-4]}"

            # 1) Get the recorder map and load the world
            recorder_str = client.show_recorder_file_info(recorder_path, True)
            recorder_map = recorder_str.split("\n")[1][5:]
            world = client.load_world(recorder_map)
            world.tick()

            # 2) Change the weather and synchronous mode
            world.set_weather(WEATHER)
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / FPS
            settings.synchronous_mode = True
            world.apply_settings(settings)

            for _ in range(100):
                world.tick()

            # 3) Replay the recorder
            max_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
            if recorder_duration == 0:
                recorder_duration = max_duration
            elif recorder_start + recorder_duration > max_duration:
                print("Found a duration that exceeds the recorder length. Reducing it...")
                recorder_duration = max_duration - recorder_start
            print(f"Duration: {round(recorder_duration, 2)} - Frames: {round(20*recorder_duration, 0)}")

            client.replay_file(recorder_path, recorder_start, recorder_duration, get_ego_id(recorder_str), False)
            with open(f"{recorder_path[:-4]}.txt", 'w') as fd:
                fd.write(recorder_str)
            world.tick()

            # 4) Link onto the ego vehicle
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

            # 5) Create the sensors, and save their data into a queue
            create_folders(endpoint, [[s[0], s[1].get('bp')] for s in SENSORS])
            blueprint_library = world.get_blueprint_library()
            sensor_queue = Queue()
            for sensor in SENSORS:

                # Extract the data from the sesor configuration
                sensor_id = sensor[0]
                attributes = sensor[1]
                blueprint_name = attributes.get('bp')
                sensor_transform = carla.Transform(
                    carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
                    carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
                )

                # Get the blueprint and add the attributes
                blueprint = blueprint_library.find(blueprint_name)
                for key, value in attributes.items():
                    if key in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        continue
                    blueprint.set_attribute(str(key), str(value))

                # Create the sensors and its callback
                sensor = world.spawn_actor(blueprint, sensor_transform, hero)
                add_listener(sensor, sensor_queue, sensor_id)
                active_sensors.append(sensor)

            world.tick()

            # 6) Run the simulation
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            sensor_amount = len(SENSORS)

            while True:
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                # print(f">>>>>  Time: {round(current_duration, 3)} / {round(recorder_duration, 3)}  <<<<<")
                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
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
                    save_data_to_disk(sensor_id, frame, data, endpoint)

                world.tick()

            for sensor in active_sensors:
                sensor.stop()
                sensor.destroy()
            active_sensors = []

            for _ in range(50):
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
