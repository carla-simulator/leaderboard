#!/bin/bash

# The only required inputs are '--routes' and '--agent', the rest can be left empty
python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--host=${CARLA_HOST} \
--port=${CARLA_PORT} \
--traffic-manager-port=${CARLA_TM_PORT} \
--traffic-manager-seed=${CARLA_TM_SEED} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--timeout=${CARLA_TIMEOUT} \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--track=${CHALLENGE_TRACK_CODENAME} \
--resume=${RESUME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
