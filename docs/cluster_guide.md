# Launching the full sweep on a cluster (max CPU)

Reference: [deploy.md](deploy.md). This page is the step-by-step version, sized for a cluster.
`sweeps/full.toml` has 268 static task leaves and 136 environments (embodied, game, loop, realtime) per model.

## Which plan

Check coverage before you launch anything: `s1arena sweep-status <plan>`. Only missing work runs, so a plan that is
mostly done is cheap to launch again. `sweeps/smoke.toml` checks every task and environment with 3 requests and one
short episode; `sweeps/full.toml` runs everything. Copy `full.toml` and edit its `[[models]]` list for your own models.
WebShop (`loop/web_shopping`) needs a WebShop server and the coding controller (`loop/coding_controller`) needs a Docker
host (step 6); give them their own plans if the other machines do not have these.

Model kinds and the keys they need (put them in `.env` or the k8s secret):

| Kind | Needs |
|---|---|
| gateway (`llm:`, `jev`) | `AI_GATEWAY_API_KEY` |
| `bedrock:` | `AWS_API_KEY`, optional `BEDROCK_REGION` (default us-east-1) |
| `copilot:` | the `copilot` CLI on PATH (or `COPILOT_CLI_PATH` / `COPILOT_RUNTIME_URL`, `COPILOT_CONNECTION_TOKEN`); not in the images |
| `lm:` (open weights) | GPU nodes with vLLM; not in the images |

## 0. How the work splits, and where CPU helps

| Stage | Bound by | Unit of sharding | Max useful shards | CPU per shard |
|---|---|---|---|---|
| static (`--only static`) | gateway API latency and rate limits | task leaf (268) | 268 | 1–2 (threads wait on HTTP) |
| envs, `core` image | simulators (MuJoCo, Java ScienceWorld, Jericho, …) plus API | environment (136) | 136 | 4–8, `env_workers` = cores |
| envs, `balrog` image (`--envs game/`) | NetHack/MiniHack/Crafter plus API | game environment | the `game/` count | 4–8 |
| WebShop server | one pod, RAM | – | 1 | 2 CPU, 20 GB RAM |
| `sweep-docker` (SWE-bench) | Docker host disk | – | 1 host | 8+ CPU, 1 TB disk |

- `--shard I/N` gives shard I every N-th task / environment (round-robin, not by cost). Shards beyond the number of
  units sit idle, so use at most 268 static shards and 136 env shards.
- One shard is one process. Inside it, `workers` threads run static calls and `env_workers` threads play environment
  seeds. For CPU-heavy simulators, prefer more shards (processes) to more threads per process, because of the GIL.
- The static stage does not need many cores; it needs many concurrent requests. Spending CPU there buys nothing
  once the gateway rate limit is reached.

## 1. Make a cluster plan

Copy the plan so the per-process settings match the hardware, and divide budgets by the shard count
(budgets are per process: with N shards a model can spend up to N × `budget_usd`).

```bash
cp sweeps/full.toml sweeps/cluster.toml
```

Edit `sweeps/cluster.toml`:

```toml
name = "full"          # keep "full": same results/ files, so sweep-status sweeps/full.toml sees everything

[defaults]
k = 1
workers = 16           # concurrent API calls per static shard
env_workers = 8        # = CPUs requested per env shard

[[models]]
spec = "jev"
budget_usd = 5         # e.g. 300 total / 64 shards
# ...same for the other models
```

Commit it, or copy it onto the shared filesystem, so the images or the checkout see it. For images, rebuild after
adding it, because `sweeps/` is baked in.

## 2. Find the CPU you can get

```bash
# Slurm
sinfo -o "%P %a %D %c %m %G" ; squeue -u $USER ; sshare -U   # partitions, cores/node, mem, your fair share
# Kubernetes
kubectl describe nodes | grep -A6 "Allocatable" ; kubectl describe quota -n $NS
```

Pick the shard counts from the free cores:

- `ENV_SHARDS = min(136, free_cores / CPUS_PER_ENV_SHARD)`, e.g. 512 cores / 8 = 64 shards.
- `STATIC_SHARDS` = 32–64 (rate-limit bound; more only helps if the gateway allows it).
- BALROG shards: 8–16.

## 3. Shared state (once)

The results volume must be shared and writable by every shard:

- `results/`
- `data_local/` (copied from a workstation that built it)
- `.cache/`

Keys:

- **Slurm:** `.env` in the checkout with `AI_GATEWAY_API_KEY=...` and `HF_TOKEN=...`.
- **Kubernetes:**

  ```bash
  kubectl create secret generic s1arena-keys --from-literal=AI_GATEWAY_API_KEY=... --from-literal=HF_TOKEN=...
  ```

Check the gateway balance first. A refusal (HTTP 402) stops a model in every shard.

## 4. Images

```bash
docker build -f deploy/Dockerfile --target core    -t $REG/s1arena:core .
docker build -f deploy/Dockerfile --target balrog  -t $REG/s1arena:balrog .
docker build -f deploy/Dockerfile --target webshop -t $REG/s1arena:webshop .   # PR #2, untested so far
docker push $REG/s1arena:core $REG/s1arena:balrog $REG/s1arena:webshop      # or push one by one
```

On Slurm, convert the images:

```bash
apptainer build s1arena-core.sif docker://$REG/s1arena:core
apptainer build s1arena-balrog.sif docker://$REG/s1arena:balrog
```

## 5a. Launch on Slurm

```bash
cd /shared/s1arena && mkdir -p logs results .cache
export PLAN=sweeps/cluster.toml
# stage 1: static, 64 shards × 2 CPU
S=$(sbatch --parsable --array=0-63 --cpus-per-task=2 --mem=4G deploy/slurm/sweep.sbatch static)
# stage 2 (after stage 1): core envs, 64 shards × 8 CPU, and BALROG games, 16 shards × 8 CPU
sbatch --array=0-63 --cpus-per-task=8 --mem=32G --dependency=afterok:$S deploy/slurm/sweep.sbatch envs
IMAGE=$PWD/s1arena-balrog.sif EXTRA="--envs game/" \
  sbatch --array=0-15 --cpus-per-task=8 --mem=32G --dependency=afterok:$S deploy/slurm/sweep.sbatch envs
```

- Command-line `--cpus-per-task/--mem` override the `#SBATCH` defaults in the script.
- Limit concurrency on a busy partition with `--array=0-63%32`.
- Add `-p <partition>` and `--time` as needed.
- WebShop: one server resets connections when many sessions start at once (seen at 28). Do not shard the WebShop
  plans widely. Run 1 shard per plan with `env_workers` 2–4 (as in the `*_shop.toml` plans), and keep the total
  concurrent episodes across all jobs hitting one server under about 16. Start more servers to go wider.
- WebShop on Slurm: run the server on one 32 GB node and export `S1ARENA_WEBSHOP_URL=http://<node>:3000` before the
  env submission. Either use `scripts/setup/webshop/` (with `--host 0.0.0.0` on the server), or run the image:
  `apptainer run -B /shared/webshop:/webshop s1arena-webshop.sif`.

## 5b. Launch on Kubernetes

In [deploy/k8s/sweep-job.yaml](../deploy/k8s/sweep-job.yaml):

- Set `REGISTRY` and the RWX `storageClassName`.
- Set the plan arg to `sweeps/cluster.toml`.
- Set `completions`/`parallelism` and the `/N` in `--shard` to your shard counts (they must agree).
- Set `requests.cpu` to `env_workers`.

Then:

```bash
kubectl apply -f deploy/k8s/webshop.yaml                    # optional; wait until Ready (first start ~15 min)
kubectl apply -f <static Job + PVC part>
kubectl wait --for=condition=complete job/s1arena-static --timeout=48h
kubectl apply -f <s1arena-envs and s1arena-envs-balrog part>
```

## 6. SWE-bench coding controller

This needs a Docker host (not a pod). Run it on one big VM or EC2 (see [deploy.md](deploy.md) "On AWS"):

```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v $PWD/results:/app/results \
  --env-file .env $REG/s1arena:core sweep-docker sweeps/full.toml
# or from anywhere, against a remote Docker host that holds the 40 evaluation images:
DOCKER_HOST=ssh://ec2-user@<host> s1arena sweep-docker sweeps/full.toml
```

Its concurrency is `env_workers` (containers at once, 8 GB each) plus the plan's optional `[docker]` table
(`parallel_seeds`, `prefetch`, `keep_images`, `min_free_gb`). Size it to the host's RAM, not the cluster's cores.

## 7. Monitor, resume, finish

```bash
tail -f logs/s1arena-*.out          # Slurm   | kubectl logs -f job/s1arena-envs  (K8s)
s1arena sweep-status sweeps/full.toml   # any time, from a node that mounts results/
```

- **Resume:** resubmit the same command. Cached results are skipped, so only missing work runs.
- **Failure handling:** a failed shard fails only its own units; rerun the array or the indices that failed (e.g. `--array=5,17`).
- **Skips:** environments skipped because they could not start (no WebShop server, BALROG games in `core`) are not
  cached. They run when you rerun in the right image.
- **Done:** run `s1arena sweep-status sweeps/full.toml` and `s1arena sweep-report sweeps/full.toml`, then copy
  `results/` back.

## Known limits

- Budgets are per shard (see step 1); there is no global cap across shards yet.
- Round-robin sharding: the slowest shard sets the wall-clock time. Oversharding (more shards than cores, e.g. via
  `%` throttling on Slurm) evens this out.
- The deploy assets are syntax-checked, not yet run end to end. Watch the first shard's log before scaling out.
