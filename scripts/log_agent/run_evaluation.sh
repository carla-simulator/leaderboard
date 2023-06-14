#!/bin/bash

DOC_STRING="Run log agent."

USAGE_STRING=$(cat <<- END
Usage: $0 [-h|--help] [-m|--mode MODE] [-id|--test-id ID]
The following modes are supported:
    * log
    * playback
END
)

usage() { echo "$DOC_STRING"; echo "$USAGE_STRING"; exit 1; }

MODE="log"
ID="test"
ROUTES_SUBSET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m |--mode )
      MODE=$2
      if [ "$MODE" != "log" ] && [ "$MODE" != "playback" ]; then
        usage
      fi
      shift 2 ;;
    -id | --test-id )
      ID=$2
      shift 2 ;;
    -rs | --route-subset )
      ROUTES_SUBSET=$2
      shift 2 ;;
    -h | --help )
      usage
      ;;
    * )
      shift ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_FOLDER="$SCRIPT_DIR/results/$ID"
EGO_LOG_FILE="$LOG_FOLDER/log.json"

if [ $MODE = "log" ]; then
  mkdir -m 777 -p $LOG_FOLDER
  touch $EGO_LOG_FILE
  export RECORD_PATH=$LOG_FOLDER
  export CHECKPOINT_ENDPOINT="$LOG_FOLDER/results_log.json"
elif [ -f "$EGO_LOG_FILE" ]; then
  export CHECKPOINT_ENDPOINT="$LOG_FOLDER/results_playback.json"
else
  echo "Could not playback the simulation from a non-existing file"
  exit 1
fi

# Create configuration file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "mode: $MODE\nfile: $EGO_LOG_FILE" > $SCRIPT_DIR/log_agent_config.txt

export TEAM_AGENT=$SCRIPT_DIR/log_agent.py
export TEAM_CONFIG=$SCRIPT_DIR/log_agent_config.txt
export ROUTES=$SCRIPT_DIR/routes_logging.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0

export DEBUG_CHECKPOINT_ENDPOINT="$LOG_FOLDER/live_results.txt"

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}
