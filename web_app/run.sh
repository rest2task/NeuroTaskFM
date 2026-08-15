#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCREEN_NAME="${NEUROTASKFM_SCREEN_NAME:-neurotaskfm}"

if [[ "${1:-}" == "--screen-child" ]]; then
  export OMP_NUM_THREADS=32

  cd "$SCRIPT_DIR"
  if [[ ! -f dist/index.html ]]; then
    npm run build
  fi

  exec "$SCRIPT_DIR/../build/bin/neurotask-web" --root "$SCRIPT_DIR" --address 0.0.0.0 --port "${NEUROTASKFM_PORT:-8000}"
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "NeuroTaskFM requires GNU Screen, but 'screen' was not found." >&2
  exit 1
fi

screen -wipe "$SCREEN_NAME" >/dev/null 2>&1 || :

if screen -S "$SCREEN_NAME" -Q windows >/dev/null 2>&1; then
  echo "NeuroTaskFM is already running in Screen session '$SCREEN_NAME'." >&2
  echo "Attach with: screen -r $SCREEN_NAME" >&2
  exit 1
fi

if ! screen -DmS "$SCREEN_NAME" "$SCRIPT_DIR/run.sh" --screen-child \
  </dev/null >/dev/null 2>&1; then
  echo "Could not create Screen session '$SCREEN_NAME'." >&2
  exit 1
fi

for _attempt in 1 2 3 4 5; do
  if ! screen -S "$SCREEN_NAME" -Q windows >/dev/null 2>&1; then
    echo "NeuroTaskFM failed to stay running in Screen session '$SCREEN_NAME'." >&2
    exit 1
  fi
  sleep 0.2
done

echo "NeuroTaskFM started in detached Screen session '$SCREEN_NAME'."
echo "Attach: screen -r $SCREEN_NAME"
echo "Stop:   screen -S $SCREEN_NAME -X quit"
