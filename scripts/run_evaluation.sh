#!/bin/bash

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
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

