#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Traffic-Manager agent: drives the ego with CARLA's TrafficManager (autopilot)
instead of an AD stack, so leaderboard / scenario_runner scenarios can be used
to stress-test TM behaviour. The route's global plan is handed to the TM via
set_path so the ego follows the prescribed route, and the experimental TM
capabilities under development (adaptive lateral control + junction
gap-acceptance) are enabled on the ego.

A forward LIDAR provides static-obstacle sensing that the TM cannot do itself on
the leaderboard maps: TM's in-engine ray query (world.CastRay) returns nothing on
the tiled Town12/13 LargeMaps, so cones, barriers and open doors are invisible to
it there. A real sensor actor renders in the server pipeline and does see them.
We detect obstacles intruding the ego's forward corridor and steer around them by
commanding a TM lane offset (tm.vehicle_lane_offset), which the TM applies on top
of its own control -- so the ego keeps following the route while giving the
obstacle clearance.

Because the scenario manager applies whatever VehicleControl the agent returns
after the world tick, run_step returns the ego's *current* control (the one the
TM last applied), re-applying it rather than fighting the TM with an empty one.
"""

from __future__ import print_function

import math
import os

import numpy as np

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'TmAgent'


# CARLA lidar points are in the sensor frame: +x forward, +z up. Flip if the
# offset is seen steering *into* the obstacle (sign of the lateral axis).
_Y_TO_RIGHT = 1.0

# Forward corridor the lidar watches for intruding obstacles.
_MIN_AHEAD = 3.0        # m: ignore points on/under the bumper
_MAX_AHEAD = 14.0       # m: only react to near obstacles
_NEAR_AHEAD = 10.0      # m: window used to locate the imminent obstacle edges
_Z_LOW = -1.3           # m (sensor-relative): above the ground return
_Z_HIGH = 0.4           # m: below overhead structures
_LANE_BAND = 2.6        # m: half-width of the watched corridor
_MIN_POINTS = 12        # minimum points to treat as a real obstacle
_CENTER_BLOCK = 0.9     # m: only dodge when points intrude this close to the
                        #     driving centerline (a real in-lane blockage); parked
                        #     cars lining the road sit well outside this and are
                        #     ignored, so the ego does not veer into roadside clutter
_MIN_BLOCK_POINTS = 10  # min centerline-intruding points to count as a blockage
_EGO_HALF = 1.05        # m: ego half width
_CLEAR_MARGIN = 0.4     # m: extra clearance to leave
_MAX_OFFSET = 1.9       # m: cap the lateral offset (into the verified-clear
                        #     oncoming lane when passing a lane-blocking obstacle)
_PASS_DECISIVE = 1.8    # m: fixed, decisive dodge committed once a pass scenario's
                        #     gap is accepted. We KNOW the obstacle blocks the lane
                        #     (VehicleOpensDoor/ParkedObstacle/... def), so we do not
                        #     size the dodge from the noisy instantaneous lidar edge
                        #     (which under-shoots and grazes the obstacle) -- we
                        #     commit a decisive shift into the already-cleared lane.
_OFFSET_RATE = 0.20     # m/tick: normal smoothing rate toward the target offset
_OFFSET_RATE_FAST = 0.40  # m/tick: once a gap is accepted, dive out fast
_RETURN_RATE = 0.30     # m/tick: return to lane promptly once past the obstacle

# --- Target-lane gap acceptance (before invading the adjacent lane) ----------
_GAP_LANE_WIDTH = 3.5   # m: nominal lane width for locating the target lane
_GAP_HORIZON = 60.0     # m: how far up/down the target lane to look for traffic
_GAP_TTC = 5.0          # s: min time-to-conflict of any target-lane vehicle
_GAP_MIN_DIST = 14.0    # m: a vehicle this close ahead in the target lane blocks
                        #     regardless of closing speed (slow/stopped lead)
_HOLD_SLOWDOWN = 92.0   # % speed reduction while waiting for a gap (crawl, so the
                        #     ego stops short of the obstacle instead of hitting it)

# --- Scenario dispatch (from the TrafficManager's own situation estimate) -----
# The lidar dodge only makes sense when a static/parked obstacle blocks part of
# the ego's lane and must be passed. It must NOT run in parking lots, junctions
# or open road -- there it would veer into roadside clutter. Rather than
# memorising trigger coordinates from the route file, we ask the (now
# situation-aware) TrafficManager what it thinks the situation is, inferred each
# tick from world state, and gate the dodge on a pass-type estimate. The integer
# values mirror SituationLabel in DataStructures.h.
_SIT_CLEAR = 0
_SIT_STATIC_BLOCKER = 1
_SIT_ONCOMING_BLOCK = 2
_SIT_PEDESTRIAN_CROSSING = 3
_SIT_JUNCTION_YIELD = 4
_SIT_ONCOMING_INTRUSION = 5
_SIT_LEAD_BRAKING = 6
_SIT_NAMES = {
    _SIT_CLEAR: 'CLEAR', _SIT_STATIC_BLOCKER: 'STATIC_BLOCKER',
    _SIT_ONCOMING_BLOCK: 'ONCOMING_BLOCK', _SIT_PEDESTRIAN_CROSSING: 'PEDESTRIAN_CROSSING',
    _SIT_JUNCTION_YIELD: 'JUNCTION_YIELD', _SIT_ONCOMING_INTRUSION: 'ONCOMING_INTRUSION',
    _SIT_LEAD_BRAKING: 'LEAD_BRAKING',
}
# Situations in which the lidar dodge is the right tool: a lane-blocking obstacle
# to pass. A pedestrian/oncoming/junction estimate must NOT trigger a dodge.
_PASS_SITUATIONS = {_SIT_STATIC_BLOCKER, _SIT_ONCOMING_BLOCK}
# The lidar spins at 10 Hz while the sim runs at ~20 fps, so each frame is only
# half a rotation: frames alternate between seeing an obstacle and missing it.
# Latch the last obstacle target for this many frames so the "miss" frames do not
# reset the offset before it has built up (otherwise the ego never shifts enough).
_HOLD_TICKS = 26        # ~1.3 s: hold the pass offset well past the obstacle, so
                        #     the ego does not swing back before physically clearing
                        #     it (the body leaves the narrow corridor before the ego
                        #     has passed the intruding door/edge)


class TmAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._configured = False
        self._hero = None
        self._tm = None
        self._offset = 0.0
        self._hold_target = 0.0
        self._hold_ticks = 0
        self._pt_buffer = []
        self._committed = False
        self._dodge_sign = 0.0
        self._pass_offset = 0.0
        self._waiting_dir = 0.0
        self._collision_sensor = None
        self._world = None
        self._dbg = None
        self._sit = _SIT_CLEAR       # last situation estimate read from the TM
        self._debug = os.getenv("TM_AGENT_DEBUG") is not None

    def _situation(self):
        """The TrafficManager's current situation estimate for the ego, inferred
        server-side from world state (no route coordinates). Returns a
        SituationLabel int; CLEAR if the API/estimate is unavailable."""
        if self._tm is None or self._hero is None:
            return _SIT_CLEAR
        try:
            return self._tm.get_vehicle_situation(self._hero)
        except Exception:
            return _SIT_CLEAR   # older wheel without the binding -> never gate on it

    def _in_pass_scenario(self):
        """True when the TM estimates a lane-blocking obstacle to pass. This is
        the inferred replacement for the old route-coordinate gate: the TM tells
        us it is a pass situation, and that is when the lidar dodge is the tool."""
        self._sit = self._situation()
        return self._sit in _PASS_SITUATIONS

    def sensors(self):
        # Speedometer (plumbing) + a forward LIDAR for static-obstacle sensing.
        # The leaderboard fixes the lidar intrinsics (range 85, 64ch, 10Hz); we
        # only choose the mount pose and id.
        return [
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.6,
             'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'id': 'LIDAR'},
        ]

    def _configure(self):
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            for actor in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
                if actor.attributes.get('role_name') == 'hero':
                    hero = actor
                    break
        if hero is None:
            return False

        tm_port = CarlaDataProvider.get_traffic_manager_port()
        tm = CarlaDataProvider.get_client().get_trafficmanager(tm_port)

        hero.set_autopilot(True, tm_port)

        # Follow the prescribed route: feed the global plan's locations to the TM.
        if self._global_plan_world_coord:
            path = [transform.location for (transform, _road_option)
                    in self._global_plan_world_coord]
            try:
                tm.set_path(hero, path)
            except Exception as exc:  # pragma: no cover - diagnostic only
                print("[tm_agent] set_path failed:", exc)

        # Enable the experimental TM capabilities on the ego.
        for method, arg in (("vehicle_lateral_avoidance", True),
                            ("vehicle_lateral_avoidance_raycast", True),
                            ("vehicle_junction_gap_acceptance", True)):
            fn = getattr(tm, method, None)
            if fn is not None:
                try:
                    fn(hero, arg)
                except Exception as exc:  # pragma: no cover
                    print("[tm_agent] %s failed: %s" % (method, exc))

        # Diagnostic collision sensor: prints contacts live so a run can be
        # judged from the log without waiting for the end-of-route checkpoint.
        try:
            world = CarlaDataProvider.get_world()
            bp = world.get_blueprint_library().find('sensor.other.collision')
            self._collision_sensor = world.spawn_actor(
                bp, carla.Transform(), attach_to=hero)

            self._hit_ids = set()

            def _on_collision(event):
                oid = getattr(event.other_actor, 'id', -1)
                if oid in self._hit_ids:
                    return                      # one line per struck actor
                self._hit_ids.add(oid)
                other = getattr(event.other_actor, 'type_id', '?')
                try:
                    eloc = self._hero.get_location()
                except Exception:
                    eloc = event.transform.location
                print("[tm_agent] COLLISION with %s (id=%d) near ego (%.0f,%.0f)" %
                      (other, oid, eloc.x, eloc.y))
            self._collision_sensor.listen(_on_collision)
        except Exception as exc:  # pragma: no cover
            print("[tm_agent] collision sensor failed:", exc)

        self._hero = hero
        self._tm = tm
        self._world = CarlaDataProvider.get_world()
        self._configured = True
        print("[tm_agent] ego on TM autopilot (port %d), route fed, features on"
              % tm_port)
        return True

    def _target_offset(self, points):
        """Lateral offset (m; + = right) that centres the ego in the free gap
        between the nearest left/right intrusions of a near obstacle, or 0.0 if
        the corridor is clear. Stashes diagnostics in self._dbg."""
        self._dbg = None
        if points is None or len(points) < _MIN_POINTS:
            return 0.0
        x = points[:, 0]
        y = points[:, 1] * _Y_TO_RIGHT
        z = points[:, 2]
        mask = ((x > _MIN_AHEAD) & (x < _MAX_AHEAD) &
                (z > _Z_LOW) & (z < _Z_HIGH) & (np.abs(y) < _LANE_BAND))
        ny = y[mask]
        if ny.shape[0] < _MIN_POINTS:
            return 0.0

        # Only react to a genuine in-lane blockage: something intruding close to
        # the driving centerline. Parked cars lining the road never do, so this
        # rejects roadside clutter (parking lots) that would otherwise make the
        # ego veer into other parked vehicles.
        if np.count_nonzero(np.abs(ny) < _CENTER_BLOCK) < _MIN_BLOCK_POINTS:
            return 0.0

        right = ny[ny > 0.0]
        left = ny[ny < 0.0]
        # Nearest-to-centre edge of each side (robust to a few stray points).
        r_edge = float(np.percentile(right, 4)) if right.size >= 5 else float('inf')
        l_edge = float(np.percentile(left, 96)) if left.size >= 5 else float('-inf')
        self._dbg = (ny.shape[0], l_edge, r_edge)

        # Corridor clear on both sides -> no offset.
        if r_edge > _EGO_HALF and l_edge < -_EGO_HALF:
            return 0.0

        need = 2.0 * _EGO_HALF + _CLEAR_MARGIN
        if math.isinf(r_edge):          # only a left intrusion -> shift right to clear it
            center = l_edge + need / 2.0
        elif math.isinf(l_edge):        # only a right intrusion (the door) -> shift left
            center = r_edge - need / 2.0
        else:
            # Intrusions on both sides: aim for the middle of the gap between them.
            # If the gap is too narrow to fit, this still biases toward the middle,
            # leaving as much clearance as possible while the ego slows.
            center = (l_edge + r_edge) / 2.0

        return max(-_MAX_OFFSET, min(_MAX_OFFSET, center))

    def _lane_clear(self, side_sign):
        """True if the adjacent lane on `side_sign` (+1 right, -1 left) is free of
        conflicting traffic over the commit horizon -- checking BOTH oncoming
        (ahead, closing) and overtaking (behind, catching up) vehicles."""
        if self._world is None or self._hero is None:
            return True
        tf = self._hero.get_transform()
        f = tf.get_forward_vector()
        r = tf.get_right_vector()
        ev = self._hero.get_velocity()
        target_lat = side_sign * _GAP_LANE_WIDTH
        try:
            vehicles = self._world.get_actors().filter('vehicle.*')
        except Exception:
            return True
        for v in vehicles:
            if v.id == self._hero.id:
                continue
            loc = v.get_location()
            dx = loc.x - tf.location.x
            dy = loc.y - tf.location.y
            lon = dx * f.x + dy * f.y          # + ahead
            lat = dx * r.x + dy * r.y          # + right
            if abs(lat - target_lat) > _GAP_LANE_WIDTH * 0.6:
                continue                        # not in the target lane
            if abs(lon) > _GAP_HORIZON:
                continue
            # A vehicle close ahead in the target lane blocks outright (slow lead).
            if 0.0 < lon < _GAP_MIN_DIST:
                return False
            vv = v.get_velocity()
            rel_v = (vv.x - ev.x) * f.x + (vv.y - ev.y) * f.y   # closing rate (fwd)
            if lon > 0.0 and rel_v < -0.5:      # ahead and approaching (oncoming/slower)
                if lon / (-rel_v) < _GAP_TTC:
                    return False
            elif lon < 0.0 and rel_v > 0.5:     # behind and catching up (overtaker)
                if (-lon) / rel_v < _GAP_TTC:
                    return False
        return True

    def _process_lidar(self, input_data):
        if self._tm is None or self._hero is None:
            return

        # Scenario dispatch: only run the obstacle-pass dodge when the TM's own
        # situation estimate says a lane-blocking obstacle is ahead. Elsewhere
        # (parking, junctions, open road, pedestrian, oncoming) keep the lane
        # offset at zero and hand driving back to the TrafficManager, so nearby
        # parked/roadside cars never make the ego veer into them.
        if not self._in_pass_scenario():
            self._pt_buffer = []
            self._committed = False
            self._dodge_sign = 0.0
            self._waiting_dir = 0.0
            if self._offset != 0.0:
                self._offset += max(-_RETURN_RATE,
                                    min(_RETURN_RATE, 0.0 - self._offset))
                if abs(self._offset) < 0.03:
                    self._offset = 0.0
            try:
                self._tm.vehicle_lane_offset(self._hero, self._offset)
                self._tm.vehicle_percentage_speed_difference(self._hero, 0.0)
            except Exception:
                pass
            return

        data = input_data.get('LIDAR')
        points = data[1] if data is not None else None
        # Accumulate the last 2 frames (~one full 10 Hz rotation at 20 fps) so each
        # decision sees the whole scene, not a flip-flopping half-rotation.
        if points is not None and len(points) > 0:
            self._pt_buffer.append(points)
            self._pt_buffer = self._pt_buffer[-2:]
        combined = np.concatenate(self._pt_buffer, axis=0) if self._pt_buffer else None
        target = self._target_offset(combined)

        seen = self._dbg is not None and abs(target) > 0.05

        # Latch "obstacle present" across the lidar's half-rotation miss-frames.
        if seen:
            self._hold_ticks = _HOLD_TICKS
        elif self._hold_ticks > 0:
            self._hold_ticks -= 1
        present = seen or self._hold_ticks > 0

        # State machine. COMMITTED: hold a FIXED pass offset + direction (no
        # oscillation, never flips toward the obstacle) until it is behind us.
        # Else: on a fresh obstacle, decide a direction and only dive out if the
        # target lane is clear (oncoming ahead + overtaking behind); otherwise
        # wait -- holding in lane and slowing so the ego stops short of it.
        blocked = False
        if self._committed:
            target = self._pass_offset if present else 0.0
            if not present and abs(self._offset) < 0.08:
                self._committed = False
                self._dodge_sign = 0.0
        elif present:
            want_dir = self._dodge_sign
            if seen:
                want_dir = 1.0 if target > 0 else -1.0
            elif self._waiting_dir != 0.0:
                want_dir = self._waiting_dir
            if want_dir == 0.0:
                target = 0.0
            elif self._lane_clear(want_dir):
                self._committed = True
                self._dodge_sign = want_dir
                self._waiting_dir = 0.0
                self._pass_offset = want_dir * min(_MAX_OFFSET,
                                                   max(_PASS_DECISIVE, abs(target) + 0.3))
                target = self._pass_offset
            else:
                target = 0.0
                blocked = True
                self._waiting_dir = want_dir
        else:
            target = 0.0
            self._waiting_dir = 0.0

        # Longitudinal: crawl while waiting for a gap so the ego stops short of the
        # obstacle instead of running into it; resume normal speed otherwise.
        try:
            self._tm.vehicle_percentage_speed_difference(
                self._hero, _HOLD_SLOWDOWN if blocked else 0.0)
        except Exception:
            pass

        # Fast out when committing to the pass, prompt return when coming back.
        if abs(target) > abs(self._offset):
            rate = _OFFSET_RATE_FAST if self._committed else _OFFSET_RATE
        else:
            rate = _RETURN_RATE
        self._offset += max(-rate, min(rate, target - self._offset))
        if abs(self._offset) < 0.03:
            self._offset = 0.0
        try:
            self._tm.vehicle_lane_offset(self._hero, self._offset)
        except Exception:
            pass
        if self._debug and (abs(self._offset) > 0.05 or self._dbg is not None):
            try:
                loc = self._hero.get_location()
                pos = "(%.0f,%.0f)" % (loc.x, loc.y)
            except Exception:
                pos = "(?)"
            dbg = self._dbg
            info = ("obst n=%d edges=[%.2f,%.2f]" % dbg) if dbg else "no-obst"
            print("[tm_agent] ego%s sit=%s %s target=%.2f offset=%.2f blocked=%s commit=%s" %
                  (pos, _SIT_NAMES.get(self._sit, '?'), info, target, self._offset,
                   blocked, self._committed))

    def _diag_perception(self):
        """Diagnostic: dump every nearby vehicle's on-path geometry relative to
        the ego -- the same world-state view the TrafficManager perceives. Tells
        us whether actor perception can see an obstacle (e.g. the open-door car)
        as a stopped, on-path blocker. Env-gated (TM_PERC_DEBUG)."""
        if self._world is None or self._hero is None:
            return
        try:
            et = self._hero.get_transform()
            eloc = et.location
            fwd = et.get_forward_vector()
        except Exception:
            return
        rx, ry = -fwd.y, fwd.x            # right of heading (TM convention)
        rows = []
        for v in self._world.get_actors().filter('vehicle.*'):
            if v.id == self._hero.id:
                continue
            vl = v.get_location()
            dx, dy = vl.x - eloc.x, vl.y - eloc.y
            lon = dx * fwd.x + dy * fwd.y          # ahead > 0
            lat = dx * rx + dy * ry                # right > 0
            if lon < -2.0 or lon > 25.0 or abs(lat) > 6.0:
                continue
            vv = v.get_velocity()
            spd = (vv.x ** 2 + vv.y ** 2) ** 0.5
            ext = v.bounding_box.extent
            rows.append((lon, lat, spd, max(ext.x, ext.y),
                         v.type_id.split('.')[-1]))
        rows.sort(key=lambda r: r[0])
        loc = "(%.0f,%.0f)" % (eloc.x, eloc.y)
        if rows:
            desc = " | ".join("%s lon=%.1f lat=%.2f spd=%.1f half=%.2f" %
                              (r[4], r[0], r[1], r[2], r[3]) for r in rows[:4])
            print("[perc] ego%s : %s" % (loc, desc))

    def run_step(self, input_data, timestamp):
        if not self._configured:
            self._configure()
        if os.environ.get('TM_PERC_DEBUG') is not None:
            self._diag_perception()
        # TM_AGENT_NO_LIDAR: hand all obstacle handling to the (now
        # situation-aware) TrafficManager C++ stage and disable the python
        # lidar dodge, so a run measures what the TM decides from world state
        # alone. The lidar dodge otherwise commands vehicle_lane_offset and
        # would override the TM's own lateral command.
        if os.environ.get('TM_AGENT_NO_LIDAR') is None:
            self._process_lidar(input_data)
        if self._hero is not None:
            try:
                return self._hero.get_control()
            except Exception:
                pass
        return carla.VehicleControl()

    def destroy(self):
        self._hero = None
