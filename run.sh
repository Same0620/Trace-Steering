#!/bin/bash
# run.sh <script.py> [args...] -- run one pipeline script on a GPU node through Slurm.
# stdout+stderr are streamed to the terminal AND saved to results/log_<script>.txt.
#
#   ./run.sh selftest_steer.py
#   ./run.sh cache.py --mode random --n-mean 2000 --n-persample 200
#   SLURM_TIME=04:00:00 ./run.sh sweep.py
#
# Overridable via environment: SLURM_PARTITION (dev), SLURM_ACCOUNT (MST115329: the only project with SU balance on 2026-09-12),
# SLURM_TIME (03:00:00), SLURM_GPUS (1), SLURM_CPUS (8), SLURM_MEM (64G).
# The Python interpreter and HF cache are fixed: no torch on the system python; /home is nearly full.
set -uo pipefail
cd "$(dirname "$0")"
SCRIPT="${1:?usage: run.sh <script.py> [args...]}"; shift
PY=/work/u4161854/.conda/envs/self_improve/bin/python
export HF_HOME=/work/u4161854/.cache/huggingface
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results
LOG="results/log_${SCRIPT%.py}$( [ -n "${1:-}" ] && printf "_%s" "$(echo "$1" | tr -c "A-Za-z0-9" "_" | sed "s/_*$//")" ).txt"   # mode flag appended (e.g. _v2, _resume_panel)
ACCT=(--account="${SLURM_ACCOUNT:-MST115329}")   # the cluster refuses jobs without --account
echo "[run.sh] $(date '+%F %T')  srun -p ${SLURM_PARTITION:-dev} ${ACCT[*]} --gres=gpu:${SLURM_GPUS:-1} --time=${SLURM_TIME:-03:00:00}  $PY $SCRIPT $*" | tee "$LOG"
echo "[run.sh] HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE  log=$LOG" | tee -a "$LOG"
srun --partition="${SLURM_PARTITION:-dev}" "${ACCT[@]}" --nodes=1 --ntasks=1 \
     --gres=gpu:"${SLURM_GPUS:-1}" --cpus-per-task="${SLURM_CPUS:-8}" --mem="${SLURM_MEM:-64G}" \
     --time="${SLURM_TIME:-03:00:00}" --job-name="ft-${SCRIPT%.py}" \
     bash -c 'echo "[run.sh] node=$(hostname) job=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"; exec "$@"' _ "$PY" "$SCRIPT" "$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo "[run.sh] $(date '+%F %T')  exit $STATUS" | tee -a "$LOG"
exit "$STATUS"
