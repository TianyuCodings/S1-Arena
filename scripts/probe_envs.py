"""Which S1-Arena environments start in this image: the sweep's own check (construct, reset seed 0, close) plus one
observe(), each environment in its own subprocess. No model is called.

    python probe_envs.py [prefix ...]      e.g. embodied/ loop/ realtime/
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from s1arena import registry

CHILD = """
import sys
from s1arena import registry
registry.discover()
cls = registry.ENVS[sys.argv[1]]
if hasattr(cls, "image"):
    print("DOCKER"); sys.exit(0)
env = cls()
try:
    env.reset(0)
    env.observe_agent(env.active_agents()[0]) if getattr(env, "agents", None) else env.observe()
finally:
    env.close()
print("OK")
"""


def probe(name: str) -> str:
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, name], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return f"TIMEOUT {name}"
    if p.returncode == 0:
        return f"{(p.stdout.strip().splitlines() or ['?'])[-1]:7s} {name}"
    return f"FAIL    {name}: {(p.stderr.strip().splitlines() or ['?'])[-1][:220]}"


if __name__ == "__main__":
    registry.discover()
    names = sorted({cls.name for prefix in (sys.argv[1:] or [""]) for cls in registry.envs_under(prefix)})
    with ThreadPoolExecutor(8) as pool:
        for line in pool.map(probe, names):
            print(line, flush=True)
