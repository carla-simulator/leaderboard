#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import time
import os
import sys
import threading
import numpy as np
import cv2
import struct

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
from queue import Queue
from queue import Empty

# convert to 16bit PNG (R=8bit for preview, G+B=float32)
def Buffer2PNG16(filename, buffer, w, h):

    # float32 x 4 buffer
    a = np.frombuffer(buffer, dtype = np.float32)

    # int16 * 3 with mask
    b = np.frombuffer(a.data, dtype = np.uint16)
    b = b.reshape([-1, 4])

    # result is b[::2,:3]
    # masking colum of 4th int16
    # masking every second row
    rgb16 = b[::2,:3].reshape(h, w, 3)
    cv2.imwrite(filename, rgb16)

# we use a function for the listener so all paramerters keep as local copy
# for the lambda when it is called
def add_listener(camera, sensor_queue, id, side):
    camera.listen(lambda data: sensor_callback(data, sensor_queue, id, side))

def do_nothing():
    pass

# we use a function for the listener so all paramerters keep as local copy
# for the lambda when it is called
def add_listener_gbuffer(camera, sensor_queue, side):
    camera.listen_to_gbuffer(carla.GBufferTextureID.SceneColor,    lambda data: sensor_callback(data, sensor_queue, "SceneColor", side))
    # camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth,    lambda data: sensor_callback(data, sensor_queue, "SceneDepth", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.SceneDepth,    lambda data: do_nothing())
    camera.listen_to_gbuffer(carla.GBufferTextureID.SceneStencil,  lambda data: sensor_callback(data, sensor_queue, "SceneStencil", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferA,      lambda data: sensor_callback(data, sensor_queue, "GBufferA", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferB,      lambda data: sensor_callback(data, sensor_queue, "GBufferB", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferC,      lambda data: sensor_callback(data, sensor_queue, "GBufferC", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferD,      lambda data: sensor_callback(data, sensor_queue, "GBufferD", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferE,      lambda data: sensor_callback(data, sensor_queue, "GBufferE", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.GBufferF,      lambda data: sensor_callback(data, sensor_queue, "GBufferF", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.Velocity,      lambda data: sensor_callback(data, sensor_queue, "Velocity", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.SSAO,          lambda data: sensor_callback(data, sensor_queue, "SSAO", side))
    # camera.listen_to_gbuffer(carla.GBufferTextureID.CustomDepth,   lambda data: sensor_callback(data, sensor_queue, "CustomDepth", side))
    camera.listen_to_gbuffer(carla.GBufferTextureID.CustomDepth,   lambda data: do_nothing())
    camera.listen_to_gbuffer(carla.GBufferTextureID.CustomStencil, lambda data: sensor_callback(data, sensor_queue, "CustomStencil", side))

# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name, sensor_side):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data, sensor_side))

def write_image(buffer, frame, id, side):
    # print(frame, id, buffer)
    if id == "SceneDepth" or id == "CustomDepth":
        # f = open("buffer.bin", "wb")
        # f.write(buffer.raw_data)
        # f.close()
        Buffer2PNG16('_out/capture/%s_%d/%06d_%s_%d.png' % (id, side, frame, id, side),
                buffer.raw_data,
                buffer.width,
                buffer.height)
    else:
        buffer.save_to_disk('_out/capture/%s_%d/%06d_%s_%d.png' % (id, side, frame, id, side), carla.ColorConverter.Raw)
    print('file %06d_%s_%d.png' % (frame, id, side))

def create_folders(endpoint):
    if not os.path.exists(endpoint):
        os.makedirs(endpoint)

def createCameras(world, trans, hero, camera_list, actor_list, sensor_queue, id):
    # size
    w = 1920
    h = 1080
    fov = 90

    # RGB camera
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    # spawn
    camera = world.spawn_actor(bp, trans, hero)
    add_listener(camera, sensor_queue, "RGB", id)
    add_listener_gbuffer(camera, sensor_queue, id)
    actor_list.append(camera)
    camera_list[id-1] = camera
    print("Created RGB Camera %d" % id)

    # semantic camera
    bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    # spawn
    camera = world.spawn_actor(bp, trans, hero)
    add_listener(camera, sensor_queue, "Semantic", id)
    actor_list.append(camera)
    print("Created Semantic Camera %d" % id)

    # instance semantic camera
    bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    # spawn
    camera = world.spawn_actor(bp, trans, hero)
    add_listener(camera, sensor_queue, "InstanceSemantic", id)
    actor_list.append(camera)
    print("Created Instance Semantic Camera %d" % id)


def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-s', '--skip', default=0, type=int, help='Amount of initial frames to skip before writting (default 0)')
    argparser.add_argument('-e', '--endpoint', required=True, help='Endpoint folder path')
    args = argparser.parse_args()

    create_folders(args.endpoint)

    actor_list = []

    try:

        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        world = client.get_world()
        settings = world.get_settings()
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # Get the ego vehicle
        while hero is None:
            print("Waiting for the ego vehicle...")
            possible_vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    hero = vehicle
                    break
            time.sleep(5)

        # Read Yaml


        # Create sensor

        trans = hero.get_transform()
        trans.location.x = 0.5
        trans.location.y = 0
        trans.location.z = 1.7
        trans.rotation.pitch = 0
        trans.rotation.yaw = 0
        trans.rotation.roll = 0
        sensor_queue = Queue()
        total_cameras = 8
        camera_list = [None] * (total_cameras + 1)
        # create all cameras
        for i in range(total_cameras):
            trans.rotation.yaw = i * (360 / total_cameras)
            createCameras(world, trans, hero, camera_list, actor_list, sensor_queue, i+1)
        # add another camera for stereo front vision
        trans.location.y = -0.5
        trans.rotation.yaw = 0
        createCameras(world, trans, hero, camera_list, actor_list, sensor_queue, total_cameras+1)
        total_cameras += 1

        # Sensor listen

        # On tick

        skip_frames = args.skip
        total_frames = 0
        max_frames = 700
        while (total_frames < max_frames):
            results = []
            start = sensor_queue.qsize()
            frame = world.tick()
            print("%08d (%06d/%06d %1.2f)" % (frame, total_frames, max_frames, total_frames / max_frames))

            # wait to get all data for this frame
            while (sensor_queue.qsize() < start + total_cameras + (total_cameras * 11)):
            # while (sensor_queue.qsize() < start + total_cameras + (total_cameras * 0)):
                time.sleep(0.1)

            # check to skip this frame
            if skip_frames > 0:
                skip_frames -= 1
                s_frame = sensor_queue.get(True, 1.0)
                while s_frame:
                    try:
                        s_frame = sensor_queue.get(True, 1.0)
                    except:
                        break
                continue

            # capture camera transform
            for i, cam in enumerate(camera_list):
                if (not os.path.exists("_out/capture/%s_%d/" % ("Camera", i+1))):
                    os.makedirs("_out/capture/%s_%d/" % ("Camera", i+1))
                f = open('_out/capture/%s_%d/%06d_%s_%d.txt' % ("Camera", i+1, frame, "Camera", i+1), "wt")
                trans = cam.get_transform()
                trans2 = hero.get_transform()
                f.write("%1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f\n" %
                        (round(trans.location.x, 3), round(trans.location.y, 3), round(trans.location.z, 3),
                            round(trans.rotation.pitch, 3), round(trans.rotation.yaw, 3), round(trans.rotation.roll, 3),
                            round(trans.location.x - trans2.location.x, 3), round(trans.location.y - trans2.location.y, 3), round(trans.location.z - trans2.location.z, 3),
                            round(trans.rotation.pitch - trans2.rotation.pitch, 3), round(trans.rotation.yaw - trans2.rotation.yaw, 3), round(trans.rotation.roll - trans2.rotation.roll, 3)
                        ))
                f.close()

            # write frames saved in memory
            total_threads = 0
            max_threads = 5
            s_frame = sensor_queue.get(True, 1.0)
            while s_frame:
                # print("Frame: %d   Sensor: %s   Side: %d" % (s_frame[0], s_frame[1], s_frame[3]))
                res = threading.Thread(target=write_image, args=(s_frame[2], s_frame[0], s_frame[1], s_frame[3]))
                results.append(res)
                res.start()

                # limit the simultaneos threads
                total_threads += 1
                if total_threads > max_threads:
                    for res in results:
                        res.join()
                    results = []
                    total_threads = 0

                try:
                    s_frame = sensor_queue.get(True, 1.0)
                except:
                    break

            for res in results:
                res.join()

            # check number of frames to capture
            total_frames += 1
            if total_frames >= max_frames:
                break



    # Final checks

    finally:
        # stop and remove cameras
        for actor in actor_list:
            print("Stopping camera %d" % actor.id)
            actor.stop()
            actor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("Asynchronous mode setup")



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
