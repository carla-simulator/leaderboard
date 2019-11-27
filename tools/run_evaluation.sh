#!/bin/bash

# !Make sure you set $CHALLENGE_PHASE_CODENAME (e.g. dev_track_3)
# !Make sure you set $CHALLENGE_TIME_AVAILABLE (e.g. 10000)

CHALLENGE_PHASE_CODENAME=dev_track_3 python leaderboard/leaderboard_evaluator.py \
--scenarios=data/all_towns_traffic_scenarios_private.json \
--routes=data/routes_testchallenge.xml \
--debug=0 \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG}