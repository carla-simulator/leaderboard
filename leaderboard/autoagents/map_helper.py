#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This file has several useful functions related to the AD Map library
"""

from __future__ import print_function

import carla
import numpy as np
import ad_map_access as ad

def carla_loc_to_enu(carla_location):
    """Transform a CARLA location into an ENU point"""
    return ad.map.point.createENUPoint(carla_location.x, -carla_location.y, carla_location.z)

def enu_to_carla_loc(enu_point):
    """Transform an ENU point into a CARLA location"""
    return carla.Location(float(enu_point.x), float(-enu_point.y), float(enu_point.z))

def get_route_segment(start_location, end_location, distance=1, probability=0):
    """Gets the shortest route between two points"""
    route_segment = None
    start_lane_id = None
    distance_ = ad.physics.Distance(distance)
    probability_ = ad.physics.Probability(probability)

    # As the AD map library projects everything to ground level, remove the z component
    start_z = start_location.z
    start_location.z = 0

    end_z = end_location.z
    end_location.z = 0

    start_matches = ad.map.match.AdMapMatching().getMapMatchedPositions(
        carla_loc_to_enu(start_location), distance_, probability_
    )

    if not start_matches:
        start_location.z = start_z
        print("WARNING: Couldn't find a paraPoint for location '{}'.".format(start_location))
        return route_segment, start_lane_id

    end_matches = ad.map.match.AdMapMatching().getMapMatchedPositions(
        carla_loc_to_enu(end_location), distance_, probability_
    )

    if not end_matches:
        end_location.z = end_z
        print("WARNING: Couldn't find a paraPoint for location '{}'.".format(end_location))
        return route_segment, start_lane_id

    min_length = float('inf')

    for start_match in start_matches:
        start_point = start_match.lanePoint.paraPoint
        for end_match in end_matches:
            end_point = end_match.lanePoint.paraPoint

            # Get the route
            new_route_segment = ad.map.route.planRoute(start_point, end_point)
            if len(new_route_segment.roadSegments) == 0:
                continue  # The route doesn't exist, ignore it

            # Calculate route length (sum of the mean of the road's lanes length)
            length = 0
            for road_segment in new_route_segment.roadSegments:
                road_length = 0
                number_lanes = 0
                for lane_segment in road_segment.drivableLaneSegments:
                    seg_start = float(lane_segment.laneInterval.start)
                    seg_end = float(lane_segment.laneInterval.end)
                    seg_length = float(ad.map.lane.calcLength(lane_segment.laneInterval.laneId))

                    road_length += seg_length * abs(seg_end - seg_start)
                    number_lanes += 1

                if number_lanes != 0:
                    length += road_length / number_lanes

            # Save the shortest route
            if length < min_length:
                min_length = length
                route_segment = new_route_segment
                start_lane_id = start_point.laneId

    if not route_segment:
        start_location.z = start_z
        end_location.z = end_z
        print("WARNING: Couldn't find a viable route between locations "
              "'{}' and '{}'.".format(start_location, end_location))

    return route_segment, start_lane_id

def get_route_lane_list(route, start_lane_id, prev_lane_id=None):
    """Given a route and its starting lane, returns the lane segments corresponding
    to the route. This supposes that no lane changes occur during the route"""
    segments = []

    for road_segment in route.roadSegments:
        for lane_segment in road_segment.drivableLaneSegments:

            if prev_lane_id and prev_lane_id not in lane_segment.predecessors:
                continue  # Lane doesn't connect to the previous one
            elif not prev_lane_id and lane_segment.laneInterval.laneId != start_lane_id:
                continue  # Lane is different than the starting one
            prev_lane_id = lane_segment.laneInterval.laneId

            segments.append(lane_segment)

    return segments

def to_ad_paraPoint(location, distance=1, probability=0):
    """Transforms a carla.Location into an ad.map.point.ParaPoint()"""
    location_z = location.z

    # As the AD map library projects everything to ground level, remove the z component
    location.z = 0

    mapMatching = ad.map.match.AdMapMatching()
    match_results = mapMatching.getMapMatchedPositions(
        carla_loc_to_enu(location),
        ad.physics.Distance(distance),
        ad.physics.Probability(probability)
    )

    if not match_results:
        location.z = location_z
        print("WARNING: Couldn't find a para point for CARLA location {}. Consider "
              "increasing the distance or reducing the probability".format(location))
        return None

    # Filter the closest one to the given location
    distance = [float(mmap.matchedPointDistance) for mmap in match_results]
    return match_results[distance.index(min(distance))].lanePoint.paraPoint

def get_lane_interval_list(lane_interval, distance=1):
    """Separates a given lane interval smaller intervals of length equal to 'distance'"""
    start = float(lane_interval.start)
    end = float(lane_interval.end)
    length = float(ad.map.lane.calcLength(lane_interval.laneId))
    if start == end:
        return []
    return np.arange(start, end, np.sign(end - start) * distance / length)