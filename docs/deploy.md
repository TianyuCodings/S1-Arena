# Running sweeps on a cluster or on AWS

Step-by-step launch sized for maximum CPU: [cluster_guide.md](cluster_guide.md).

Everything goes through `s1arena sweep` / `sweep-docker` (README, "Sweeps"); results are cached files under
`results/`, so any number of workers can share one `results/` directory as long as they work on different tasks and
environments (`--shard I/N`), and every run resumes where the last one stopped.

## What runs where

| Part | Image / host | Notes |
|---|---|---|
| Static tasks (105 leaves) | `s1arena:core` | API calls only; CPU is enough for gateway models |
| Most environments (embodied, realtime, loop, text games, ScienceWorld, ALFWorld) | `s1arena:core` | Jericho with spaCy, ScienceWorld with Java 17 are in the image |
| BALROG games (`game/`: Crafter, Baba Is AI, MiniHack, NetHack) | `s1arena:balrog` | NLE built from source with the clock-pinning patch |
| WebShop (`loop/web_shopping`) | its own server | Python 3.8 + JDK 11, 17 GB RAM: start it with `scripts/setup/webshop/` and point `S1ARENA_WEBSHOP_URL` at it; without it the environment is skipped |
| SWE-bench coding controller | a host with Docker | `s1arena sweep-docker`; x86_64, ~1 TB disk for 40 images in turn, 8 GB per container |
| Local models (`lm:`, `dgemma:`) | GPU nodes with vLLM | not in these images |

Environments that cannot start in an image are skipped (not cached), so running the `core` and `balrog` stages
over the same plan is safe. The two stages must run in order: static first, then environments, because realtime
environments charge each model its median latency from its static runs.

## Build the images

Download the built requests first (`uv run s1arena download`): the images copy `data/`.

```bash
docker build -f deploy/Dockerfile --target core   -t REGISTRY/s1arena:core .
docker build -f deploy/Dockerfile --target balrog -t REGISTRY/s1arena:balrog .
docker push REGISTRY/s1arena:core && docker push REGISTRY/s1arena:balrog
```

The images hold the code, `data/` and `sweeps/`, no keys. Mount `results/`, `data_local/` (items built locally that
are not redistributed) and `.cache/` (downloaded game files, datasets) from shared storage, and pass
`AI_GATEWAY_API_KEY` / `HF_TOKEN` as environment variables.

## 1) On a cluster

**Kubernetes** ([deploy/k8s/sweep-job.yaml](../deploy/k8s/sweep-job.yaml)): a ReadWriteMany volume (EFS, NFS) for
`results/`, and indexed Jobs whose pods take `--shard $JOB_COMPLETION_INDEX/N`.

```bash
kubectl create secret generic s1arena-keys --from-literal=AI_GATEWAY_API_KEY=... --from-literal=HF_TOKEN=...
# copy data_local/ onto the volume once (e.g. kubectl cp through a pod that mounts it)
kubectl apply -f deploy/k8s/sweep-job.yaml           # creates the volume and the three jobs
```

Apply the `s1arena-static` job first and the two environment jobs once it has completed (split the file, or
`kubectl wait --for=condition=complete job/s1arena-static --timeout=24h` in between). Rerunning a job only fills in
what is missing. For the SWE-bench controller, run `sweep-docker` on a node with Docker (see AWS below); it needs
the Docker socket, which most clusters do not hand to pods.

**Slurm** ([deploy/slurm/sweep.sbatch](../deploy/slurm/sweep.sbatch)): build Apptainer images from the Docker
ones, keep the repository checkout (with `.env`, `data_local/`, `results/`) on the shared filesystem, and submit
array jobs:

```bash
apptainer build s1arena-core.sif docker://REGISTRY/s1arena:core
apptainer build s1arena-balrog.sif docker://REGISTRY/s1arena:balrog
mkdir -p logs results .cache
S=$(sbatch --parsable --array=0-7 deploy/slurm/sweep.sbatch static)
sbatch --array=0-15 --dependency=afterok:$S deploy/slurm/sweep.sbatch envs
IMAGE=$PWD/s1arena-balrog.sif EXTRA="--envs game/" sbatch --array=0-7 --dependency=afterok:$S deploy/slurm/sweep.sbatch envs
```

When all shards are done: `s1arena sweep-status sweeps/full.toml` (shards do not write the status file themselves).

## 2) On AWS

[deploy/aws/user-data.sh](../deploy/aws/user-data.sh) turns one EC2 instance into the whole pipeline: it installs
Docker, clones the repository, builds both images, runs static → environments (core and BALROG in parallel) →
`sweep-docker` → `sweep-status`, syncs `results/` to S3 every 10 minutes and at the end, and stops the instance.
A new instance first pulls `results/` back from S3, so a stopped or interrupted sweep (e.g. a Spot reclaim) resumes.

1. Store the secrets in SSM Parameter Store (SecureString): `/s1arena/AI_GATEWAY_API_KEY`, `/s1arena/HF_TOKEN`,
   and, only while the repository is private, `/s1arena/GITHUB_TOKEN` (a read-only token).
2. Upload `data_local/` (the sources that do not allow redistribution, built with `s1arena build --local`) to
   `s3://BUCKET/s1arena/data_local/`. The script downloads the other requests from the Hugging Face dataset.
3. An instance role allowing `ssm:GetParameter` on `/s1arena/*` (with `kms:Decrypt`) and `s3:GetObject`,
   `s3:PutObject`, `s3:ListBucket` on the bucket.
4. Launch (Amazon Linux 2023, x86_64, m7i.8xlarge or similar, 1 TB gp3 root volume) with user data made of these
   lines followed by the contents of `deploy/aws/user-data.sh`:

   ```bash
   #!/bin/bash
   export BUCKET=s3://BUCKET/s1arena BRANCH=main PLAN=sweeps/full.toml
   ```

5. Follow it with `sudo tail -f /var/log/cloud-init-output.log`; results land in `s3://BUCKET/s1arena/results/`.
   Score anywhere after `aws s3 sync s3://BUCKET/s1arena/results results`.

To go wider than one instance, run the Kubernetes jobs on EKS (EFS as the RWX volume) and keep one EC2 instance
with this script for `sweep-docker` only (`docker run ... sweep-docker`), and WebShop's server on a 32 GB instance.

## Notes

- Budgets (`budget_usd`) are tracked per worker process: with N shards a model can spend up to N × its budget.
  Divide the plan's budgets by the number of shards.
- A gateway refusal (empty balance, HTTP 402) stops that model in every shard; top up and rerun the same command.
- Shards split tasks and environments round-robin, not by cost; large environments may leave one shard last.
