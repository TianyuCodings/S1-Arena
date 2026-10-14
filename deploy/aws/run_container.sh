#!/bin/bash
# run.sh <container name> <image> [--docker] <s1arena args...>
# One s1arena process in its own container; results, local data and caches are the host's ~/s1arena dirs, and the
# code and plans are mounted read-only from ~/s1arena so a fix needs no rebuild. --docker also hands it the host's
# Docker socket (sweep-docker starts the SWE-bench containers next to it); --gateway also passes
# ~/s1arena-extra/gateway.env (AI_GATEWAY_API_KEY, for jev) and --bedrock ~/s1arena-extra/bedrock.env (AWS_API_KEY,
# for bedrock: models) to this container only.
name=$1 image=$2; shift 2
extra=()
while [ "${1:0:2}" = -- ]; do
  case $1 in
    --docker) extra+=(-v /var/run/docker.sock:/var/run/docker.sock) ;;
    --gateway) extra+=(--env-file "$HOME/s1arena-extra/gateway.env") ;;
    --bedrock) extra+=(--env-file "$HOME/s1arena-extra/bedrock.env") ;;
  esac; shift
done
cd ~/s1arena && mkdir -p results data_local .cache
docker rm -f "$name" >/dev/null 2>&1
docker run -d --name "$name" --network host --env-file ~/s1arena-extra/copilot.env -e PYTHONUNBUFFERED=1 --shm-size=2g \
  -v ~/s1arena/results:/app/results -v ~/s1arena/data_local:/app/data_local -v ~/s1arena/.cache:/app/.cache \
  -v ~/s1arena/s1arena:/app/s1arena:ro -v ~/s1arena/sweeps:/app/sweeps:ro -v ~/s1arena-extra:/extra:ro \
  "${extra[@]}" "$image" "$@"
