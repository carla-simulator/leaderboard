#!/bin/bash

CARLA_ROOT=${CARLA_ROOT}
SCENARIO_RUNNER_ROOT=${SCENARIO_RUNNER_ROOT}

CHALLENGE_TRACK_CODENAME=SENSORS
CHALLENGE_TIME_AVAILABLE=1000
ALPHADRIVE_CHECKPOINT_ENDPOINT=my_results.json



PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}":${PYTHONPATH} python leaderboard/leaderboard_evaluator.py \
--challenge-mode=True \
--scenarios=/home/grossanc/Projects/leaderboard/data/all_towns_traffic_scenarios_private.json \
--routes=/home/grossanc/Projects/leaderboard/data/routes_testleaderboard_debug.xml \
--repetitions=1 \
--track=${CHALLENGE_PHASE_CODENAME} \
--time-available=${CHALLENGE_TIME_AVAILABLE} \
--checkpoint=${ALPHADRIVE_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
#--record
