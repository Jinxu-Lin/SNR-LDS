#!/usr/bin/env bash
set -euo pipefail

CODE=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/jinxulin/CFA-envs/e3c-torch260/bin/python
DATA=/home/jinxulin/CFA/_Data
ROOT="$DATA/results/snr_lds_20260925/a3"
ATTEMPT=j3-snr-a3
MANIFEST="$ROOT/inputs/benchmark_manifest.json"
PANELS="$ROOT/panels"
LAUNCHER="$ROOT/launcher_a"
export PYTHONPATH="$CODE/src"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

run() {
  "$PY" "$CODE/tools/prepare_benchmarks.py" \
    --data-root "$DATA" --output "$MANIFEST" \
    --derived-input-root results/snr_lds_20260925/a3/inputs \
    --restore-ab2-das \
    --source-manifest "$DATA/results/ba_c2_c10_das_presquare_20260922/inputs/bench_manifest.json" \
    --source-manifest "$DATA/results/c10_cpu_closeout_20260923/inputs/bench_manifest.json" \
    --source-manifest "$DATA/results/ddpm_noncore_val_repair_20260924/cpu/ba_manifest.json" \
    --source-manifest "$DATA/results/ddpm_noncore_val_repair_20260924/shared_gpu/ba_manifest.json"
  "$PY" -m balds.cli.ba batch --rule snr_zero_mean_gaussian_v1 \
    --zetas 1,2,3,4 --primary-zeta 3 --save-fits \
    --manifest "$MANIFEST" --data-root "$DATA" --output "$PANELS"
  "$PY" "$CODE/tools/summarize_benchmarks.py" \
    --manifest "$MANIFEST" --results "$PANELS" --output "$ROOT/tables"
  "$PY" -m balds.cli.ba deletion --evaluations "$PANELS" \
    --utilities "$DATA/results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv" \
    --output "$ROOT/edel"
  "$PY" "$CODE/tools/plot_snr_benchmarks.py" --manifest "$MANIFEST" \
    --data-root "$DATA" --results "$PANELS" --output "$ROOT/figures"
  "$PY" -m balds.cli.ba verify --manifest "$MANIFEST" \
    --data-root "$DATA" --output "$PANELS" > "$ROOT/verify_a.json.tmp"
  mv "$ROOT/verify_a.json.tmp" "$ROOT/verify_a.json"
}

status() {
  echo "attempt=$ATTEMPT"
  "$PY" -m balds.cli.ba status --manifest "$MANIFEST" --output "$PANELS"
  if [[ -f "$LAUNCHER/pid" ]] && kill -0 "$(<"$LAUNCHER/pid")" 2>/dev/null; then
    echo 'worker=running'
  else
    echo 'worker=not_running'
  fi
  [[ ! -f "$LAUNCHER/exit_code" ]] || echo "exit_code=$(<"$LAUNCHER/exit_code")"
}

verify() {
  "$PY" -m balds.cli.ba verify --manifest "$MANIFEST" --data-root "$DATA" --output "$PANELS"
  test -s "$ROOT/tables/summary.json"
  test -s "$ROOT/edel/edel_selection.json"
  test -s "$ROOT/figures/fixed_c2_s42_gen_q0_q1.png"
  test "$(<"$LAUNCHER/exit_code")" = 0
}

case "${1:-}" in
  launch)
    mkdir -p "$LAUNCHER"
    if [[ -f "$LAUNCHER/pid" ]] && kill -0 "$(<"$LAUNCHER/pid")" 2>/dev/null; then
      echo 'A worker already running' >&2; exit 2
    fi
    printf '%s\n' "$ATTEMPT" > "$LAUNCHER/attempt_id.tmp"; mv "$LAUNCHER/attempt_id.tmp" "$LAUNCHER/attempt_id"
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
