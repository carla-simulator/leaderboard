"""
The sensor list that gets validated has to be the one that gets spawned (#199).

Runs without a CARLA server, carla and py_trees are stubbed:

    python -m pytest tests/test_sensor_suite.py
"""

import importlib.abc
import importlib.machinery
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STUBBED = {"carla", "py_trees", "agents", "shapely", "dictor", "requests", "simple_watchdog_timer",
           "networkx", "tabulate"}


class _StubMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub


class _Stub(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()

    def __call__(self, *args, **kwargs):
        return _Stub()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _StubMeta(name, (_Stub,), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

from leaderboard import leaderboard_evaluator  # noqa: E402
from leaderboard.autoagents import agent_wrapper  # noqa: E402
from leaderboard.autoagents.autonomous_agent import Track  # noqa: E402

LEGAL = [{"type": "sensor.camera.rgb", "id": "Center", "x": 0.7, "y": 0.0, "z": 1.6,
          "roll": 0, "pitch": 0, "yaw": 0, "width": 800, "height": 600, "fov": 100}]
# The 50 m high camera breaks the 3 m radius rule of the SENSORS track
ILLEGAL = LEGAL + [{"type": "sensor.camera.rgb", "id": "BEV", "x": 0.0, "y": 0.0, "z": 50.0,
                    "roll": 0, "pitch": -90, "yaw": 0, "width": 800, "height": 800, "fov": 90}]


class SensorSuiteTest(unittest.TestCase):

    def test_every_route_is_validated(self):
        suites = iter([LEGAL, ILLEGAL])

        class Agent(object):
            track = Track.SENSORS

            def __init__(self, *args):
                self.suite = next(suites)

            @staticmethod
            def get_ros_version():
                return 2

            def set_global_plan(self, *args):
                pass

            def setup(self, *args):
                pass

            def sensors(self):
                return self.suite

        evaluator = object.__new__(leaderboard_evaluator.LeaderboardEvaluator)
        evaluator.sensors = None
        evaluator.sensors_initialized = False
        evaluator._ros1_server = None
        evaluator.module_agent = types.SimpleNamespace(get_entry_point=lambda: "Agent", Agent=Agent)
        for name in ("statistics_manager", "manager", "_load_and_wait_for_world",
                     "_register_statistics", "_cleanup"):
            setattr(evaluator, name, mock.MagicMock())

        args = types.SimpleNamespace(timeout=10, host="localhost", port=2000, debug=0,
                                     agent_config="", track="SENSORS", record=None)
        with mock.patch.object(leaderboard_evaluator, "RouteScenario"), \
                mock.patch.object(leaderboard_evaluator, "Watchdog"):
            for index in (0, 1):
                config = types.SimpleNamespace(name="route", repetition_index=0, index=index, town="Town01")
                evaluator._load_and_run_scenario(args, config)

        self.assertEqual(evaluator.manager.load_scenario.call_count, 1,
                         "route 2 was loaded with a sensor suite that was never validated")
        self.assertIs(evaluator.manager.load_scenario.call_args.args[-1], LEGAL)
        status = evaluator._register_statistics.call_args.args[1]
        self.assertEqual(status, leaderboard_evaluator.FAILURE_MESSAGES["Sensors"][0])

    def test_wrapper_spawns_the_given_list(self):
        agent = mock.MagicMock()
        agent.sensors.return_value = ILLEGAL
        world = mock.MagicMock()

        with mock.patch.object(agent_wrapper.SimulationData, "get_world", return_value=world), \
                mock.patch.object(agent_wrapper.AgentWrapper, "_sensors_list", []):
            agent_wrapper.AgentWrapper(agent).setup_sensors(mock.MagicMock(), LEGAL)

        self.assertEqual(world.spawn_actor.call_count, len(LEGAL))
        agent.sensors.assert_not_called()


if __name__ == "__main__":
    unittest.main()
