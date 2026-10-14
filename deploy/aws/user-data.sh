#!/bin/bash
# EC2 user data (Amazon Linux 2023, x86_64) that runs a whole sweep on one instance and leaves the results in S3
# (see docs/deploy.md). Set the variables below, or pass them in the instance's user data before this script.
# Needs an instance role that may read the SSM parameters and write to the bucket. Suggested: m7i.8xlarge
# (32 vCPU, 128 GB) with a 1 TB gp3 root volume (the SWE-bench images take 2.5-7 GB each while in use).
set -euxo pipefail
: "${BUCKET:?s3 bucket for the results, e.g. s3://my-bucket/s1arena}"
REPO_URL=${REPO_URL:-https://github.com/TianyuCodings/S1-Arena.git}
BRANCH=${BRANCH:-main}
PLAN=${PLAN:-sweeps/full.toml}
SSM_PREFIX=${SSM_PREFIX:-/s1arena}   # SecureString parameters: $SSM_PREFIX/AI_GATEWAY_API_KEY, /HF_TOKEN, /GITHUB_TOKEN
STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}

dnf install -y docker git awscli
systemctl enable --now docker
param() { aws ssm get-parameter --with-decryption --name "$SSM_PREFIX/$1" --query Parameter.Value --output text 2>/dev/null || true; }

W=/opt/s1arena && mkdir -p $W && cd $W
TOKEN=$(param GITHUB_TOKEN)  # optional: a read-only token, needed only while the repository is private
URL=$REPO_URL && [ -n "$TOKEN" ] && URL=${REPO_URL/https:\/\//https://x-access-token:$TOKEN@}
[ -d repo ] || git clone --branch "$BRANCH" "$URL" repo
cd repo && git remote set-url origin "$REPO_URL"
umask 077 && { echo "AI_GATEWAY_API_KEY=$(param AI_GATEWAY_API_KEY)"; echo "HF_TOKEN=$(param HF_TOKEN)"; } > .env && umask 022
mkdir -p results data_local .cache
# the built requests of the redistributable sources, from the Hugging Face dataset (the images copy data/)
docker run --rm --env-file .env -v "$PWD:/w" -w /w python:3.12-slim sh -c "pip install -q . && s1arena download"
aws s3 sync "$BUCKET/results" results --only-show-errors || true      # resume from an earlier instance
aws s3 sync "$BUCKET/data_local" data_local --only-show-errors || true

docker build -f deploy/Dockerfile --target core -t s1arena:core .
docker build -f deploy/Dockerfile --target balrog -t s1arena:balrog .

# keep S3 up to date while the sweep runs
( while true; do aws s3 sync results "$BUCKET/results" --only-show-errors; sleep 600; done ) &
SYNC=$!

run() {  # run <image> <s1arena args...>
  docker run --rm --env-file .env -v "$PWD/results:/app/results" -v "$PWD/data_local:/app/data_local" \
    -v "$PWD/.cache:/app/.cache" "$@"
}
run s1arena:core   sweep "$PLAN" --only static
run s1arena:core   sweep "$PLAN" --only envs & CORE=$!
run s1arena:balrog sweep "$PLAN" --only envs --envs game/ & GAMES=$!
wait $CORE || true; wait $GAMES || true
# the coding controller starts its SWE-bench containers next to this one, through the host's Docker socket
run -v /var/run/docker.sock:/var/run/docker.sock s1arena:core sweep-docker "$PLAN" || true
run s1arena:core sweep-status "$PLAN"

kill $SYNC || true
aws s3 sync results "$BUCKET/results" --only-show-errors
[ "$STOP_WHEN_DONE" = 1 ] && shutdown -h now
