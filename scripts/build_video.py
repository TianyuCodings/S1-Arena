"""Make the closed-loop video of the leaderboard website: one recorded episode, replayed and rendered.

Usage (from the repository root; the environment's packages are not in the base install):

    uv run --with highway-env==1.12.1 --with gymnasium==1.3.0 --with imageio==2.37.0 --with imageio-ffmpeg==0.6.0 \
        --with pillow python scripts/build_video.py --episodes path/to/results/episodes

The script reads one episode that the runner recorded (seed, outcome and the trace of chosen options with their
probabilities), plays the trace again in the same environment exactly as the runner did (reset(seed); per decision
observe(), the idle steps while the model decides, act()), and renders every simulation frame. It writes the video
only when the replay reproduces the recorded outcome and a second replay gives the same frames; otherwise it stops.
Each frame carries a caption with the model, the decision counter and the chosen option with its probability.

Output in site/media/: the MP4 (H.264, yuv420p, no audio), a poster image, and closed_loop.json (environment, model,
seed, outcome, the trace and the check), which scripts/build_examples.py and scripts/build_site.py read. The raw
episode files stay out of the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from s1arena.env import ROOT
from s1arena.registry import discover, envs_under

MEDIA = ROOT / "site" / "media"
ENV = "realtime/highway/intersection"
MODEL_ID = "typesafe-ai/jev"  # a System-1 model; its successful episode with the most decisions on this environment
SEED = 9
FPS = 15  # highway-env simulates 15 frames per second, so the video runs in real time
END_HOLD_S = 2.0  # the last frame stays this long with the outcome
WIDTH = 480  # the 600 px render, scaled down (even size for yuv420p)


def load_episode(episodes: Path, model_id: str, env: str, seed: int) -> dict:
    path = episodes / model_id.replace("/", "__") / f"{env.replace('/', '__')}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["seed"] == seed:
            return row
    raise SystemExit(f"no episode with seed {seed} in {path}")


class FrameTap:
    """Stands in for highway-env's video-recorder hook, so _simulate hands over every intermediate frame."""

    def __init__(self, env) -> None:
        self.env, self.frames, self.label = env, [], ""

    def _capture_frame(self) -> None:
        self.frames.append((self.env.render(), self.label))


def replay(env_cls, row: dict, render: bool) -> tuple[dict, list[str], list]:
    """The runner's loop (s1arena.envs.runner.play_episode) with the recorded choices in place of the model's."""
    env = env_cls()
    env.reset(row["seed"])
    tap = None
    if render:
        env.u.render_mode = "rgb_array"  # offscreen: the env was built without a render mode
        tap = FrameTap(env.u)
        env.u._record_video_wrapper = tap
        tap.frames.append((env.u.render(), "start"))
    hashes = []
    try:
        for k, step in enumerate(row["trace"], 1):
            if env.done():
                break
            request = env.observe()
            hashes.append(hashlib.sha1(request.model_dump_json().encode()).hexdigest())
            for _ in range(row.get("hold_steps") or 0):  # the world moves on while the model decides
                if env.done():
                    break
                env.idle()
            if env.done():
                break
            if tap:
                tap.label = f"{k}"
            env.act(step["c"])
            if tap:
                tap.frames.append((env.u.render(), tap.label))
        outcome = json.loads(json.dumps(env.outcome()))
    finally:
        env.close()
    return outcome, hashes, (tap.frames if tap else [])


def caption(frame: np.ndarray, top: str, bottom: str):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")
    font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    for y0, text, f in ((0, top, font), (img.height - 30, bottom, small)):
        draw.rectangle([0, y0, img.width, y0 + 30], fill=(20, 23, 28, 210))
        draw.text((10, y0 + 6), text, fill=(240, 242, 245), font=f)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", required=True, help="folder of recorded episodes (results/episodes)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--env", default=ENV)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(MEDIA))
    args = parser.parse_args()
    discover()
    env_cls = next(c for c in envs_under(args.env) if c.name == args.env)
    row = load_episode(Path(args.episodes), args.model, args.env, args.seed)
    results = json.loads((ROOT / "leaderboard" / "results.json").read_text(encoding="utf-8"))
    model_name = next((m["name"] for m in results["models"] if args.model in (m["ids"].get("agentic"), m["ids"].get("static"))), args.model)

    plain, plain_hashes, _ = replay(env_cls, row, render=False)
    rendered, hashes, frames = replay(env_cls, row, render=True)
    _, _, again = replay(env_cls, row, render=True)
    same_frames = len(again) == len(frames) and all(np.array_equal(a[0], b[0]) for a, b in zip(frames, again))
    checks = {"outcome_matches_record": plain == row["outcome"] and rendered == row["outcome"],
              "rendering_changes_nothing": hashes == plain_hashes, "decisions_replayed": len(hashes) == len(row["trace"]),
              "second_replay_same_frames": same_frames}
    if not all(checks.values()):
        raise SystemExit(f"replay does not match the recorded episode: {checks}; outcome {plain} vs {row['outcome']}")

    trace = row["trace"]
    n = len(trace)
    out = row["outcome"]
    result = ("arrived, no collision" if out.get("success") else "collision" if out.get("crashed") else "did not arrive")
    images = []
    for frame, label in frames:
        if label == "start":
            top = f"{model_name} · start"
        else:
            k = int(label)
            step = trace[k - 1]
            top = f"{model_name} · decision {k} of {n}: {step['c']['action']} (p {step['p']:.2f})"
        images.append(caption(frame, top, "green car: driven by the model · speed only; the route turns left"))
    final = caption(frames[-1][0], f"{model_name} · {result} · return {out['score']:.2f}",
                    f"seed {row['seed']} · {n} decisions in {out.get('sim_time_s', n):.0f} s of simulated time")
    images += [final] * round(END_HOLD_S * FPS)

    import imageio.v2 as imageio

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{args.env.split('/')[-1]}_{args.model.split('/')[-1]}_seed{row['seed']}"
    video, poster = dest / f"{stem}.mp4", dest / f"{stem}.jpg"
    writer = imageio.get_writer(video, fps=FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=16,
                                output_params=["-an", "-crf", "28", "-preset", "slow", "-movflags", "+faststart",
                                               "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact"])
    for img in images:
        writer.append_data(np.asarray(img))
    writer.close()
    images[min(len(images) - 1, 5 * FPS)].save(poster, quality=82, optimize=True)
    meta = {
        "env": args.env, "env_source": env_cls.source, "model": model_name, "model_id": args.model, "seed": row["seed"],
        "mode": row.get("mode"), "assumed_latency_ms": row.get("assumed_latency_ms"), "hold_steps": row.get("hold_steps"),
        "outcome": out, "trace": [{"action": s["c"]["action"], "p": s["p"]} for s in trace],
        "replay": checks, "video": f"media/{video.name}", "poster": f"media/{poster.name}", "fps": FPS,
        "frames": len(images), "duration_s": round(len(images) / FPS, 1), "bytes": video.stat().st_size,
    }
    (dest / "closed_loop.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{video} ({video.stat().st_size // 1024} KB, {meta['duration_s']} s), {poster.name}; replay checks: {checks}")


if __name__ == "__main__":
    main()
