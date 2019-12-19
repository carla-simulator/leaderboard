#!/bin/bash

PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/:${SCENARIO_RUNNER_ROOT}":${PYTHONPATH} python leaderboard/leaderboard_evaluator.py \
--challenge-mode \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_PHASE_CODENAME} \
--time-available=${CHALLENGE_TIME_AVAILABLE} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}

