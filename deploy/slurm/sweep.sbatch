#!/bin/bash
# Sharded sweep on a Slurm cluster (see docs/deploy.md), as an array job: task i takes every N-th task / environment.
#   apptainer build s1arena-core.sif docker-daemon://s1arena:core      (or docker://REGISTRY/s1arena:core)
#   sbatch --array=0-7  deploy/slurm/sweep.sbatch static
#   sbatch --array=0-15 --dependency=afterok:<static job id> deploy/slurm/sweep.sbatch envs
#   IMAGE=s1arena-balrog.sif EXTRA="--envs game/" sbatch --array=0-7 deploy/slurm/sweep.sbatch envs
# The repository checkout on the shared filesystem holds results/, data_local/ and .cache/; the keys come from its .env.
#SBATCH --job-name=s1arena
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/s1arena-%A_%a.out
set -euo pipefail
STAGE=${1:?usage: sbatch --array=0-(N-1) sweep.sbatch static|envs}
PLAN=${PLAN:-sweeps/full.toml}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
IMAGE=${IMAGE:-$REPO/s1arena-core.sif}
COUNT=$((SLURM_ARRAY_TASK_MAX + 1))
cd "$REPO"
apptainer exec --cleanenv --env S1ARENA_ENV_FILE=/app/.env \
  --bind "$REPO/results:/app/results,$REPO/data_local:/app/data_local,$REPO/.cache:/app/.cache,$REPO/.env:/app/.env" \
  --pwd /app "$IMAGE" \
  s1arena sweep "$PLAN" --only "$STAGE" --shard "$SLURM_ARRAY_TASK_ID/$COUNT" ${EXTRA:-}
