#!/usr/bin/env bash
set -euo pipefail

CODE=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/jinxulin/CFA-envs/e3c-torch260/bin/python
DATA=/home/jinxulin/CFA/_Data
ROOT="$DATA/results/snr_lds_20260925/b1"
R16="$DATA/results/repeatability_r16"
PILOT=/home/jinxulin/BA-LDS/Reports/evidence/r16_pilots.json
LAUNCHER="$ROOT/launcher_b"
export PYTHONPATH="$CODE/src"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

run() {
  "$PY" "$CODE/tools/aggregation_controls.py" snr-transforms \
    --data-root "$DATA" --output "$ROOT/controls"
  "$PY" -m balds.cli.repeatability summarize --data-root "$DATA" \
    --output "$R16" --pilot-manifest "$PILOT" --analysis-output "$ROOT/repeatability"
}

status() {
  "$PY" -m balds.cli.repeatability status --data-root "$DATA" --output "$R16"
  test ! -f "$ROOT/controls/snr_completion.json" || echo 'controls=complete'
  if [[ -f "$LAUNCHER/pid" ]] && kill -0 "$(<"$LAUNCHER/pid")" 2>/dev/null; then
    echo 'worker=running'
  else
    echo 'worker=not_running'
  fi
  [[ ! -f "$LAUNCHER/exit_code" ]] || echo "exit_code=$(<"$LAUNCHER/exit_code")"
}

verify() {
  test "$(<"$LAUNCHER/exit_code")" = 0
  "$PY" "$CODE/tools/verify_snr_appendix.py" \
    --controls "$ROOT/controls" --repeatability "$ROOT/repeatability"
}

case "${1:-}" in
  launch)
    mkdir -p "$LAUNCHER"
    if [[ -f "$LAUNCHER/pid" ]] && kill -0 "$(<"$LAUNCHER/pid")" 2>/dev/null; then
      echo 'B worker already running' >&2; exit 2
    fi
    nohup "$0" run > "$LAUNCHER/run.log" 2>&1 &
    printf '%s\n' "$!" > "$LAUNCHER/pid.tmp"; mv "$LAUNCHER/pid.tmp" "$LAUNCHER/pid"
    ;;
  run)
    mkdir -p "$LAUNCHER"
    trap 'rc=$?; printf "%s\n" "$rc" > "$LAUNCHER/exit_code.tmp"; mv "$LAUNCHER/exit_code.tmp" "$LAUNCHER/exit_code"; exit "$rc"' EXIT
    run
    ;;
  resume) exec "$0" launch ;;
  status) status ;;
  verify) verify ;;
  *) echo "usage: $0 {launch|run|resume|status|verify}" >&2; exit 2 ;;
esac
