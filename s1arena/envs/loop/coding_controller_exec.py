"""Containers and grading for loop/coding_controller (see coding_controller.py).

Every episode runs in its own container, started from the SWE-bench evaluation image the dataset names for the
instance (swebench/sweb.eval.x86_64.<id>:latest) with no network, a fixed host name and PYTHONHASHSEED=0 for every
command, so the same actions print the same text. Commands run as root in /testbed inside the image's conda
environment, under an in-container `timeout`.

The graded patch is exactly what changed since the episode started: at the start the working tree (tracked and
untracked, not ignored) is written as a git tree object through a private index file, and at the end the same is done
again and the two trees are diffed. Nothing touches the repository's own index, HEAD or refs, so commits, stashes or
index edits made by the actions do not change what is graded; only the files do.

Grading follows swebench.harness.run_evaluation.run_instance of swebench 5.0.2: a fresh container from the same
image (default network, as there), the patch applied with the harness's own four apply commands in order (resetting
the tree between attempts, then the reverse check), the instance's eval script from the dataset run with a time
limit, and swebench's get_eval_report on the output; an empty patch is unresolved without running tests, as there.
Two small differences: the container does not get CAP_SYS_ADMIN (the harness adds it for browser sandboxes of
JavaScript repositories; the Python repositories of SWE-bench Verified do not need it), and the patch and script are
written through stdin instead of `docker cp`.

Two ideas come from the user's Escalate harness (harness/container.py): wrapping commands in the conda environment
with an in-container timeout, and grading against a harness-side baseline of the initial tree rather than the
repository's own HEAD. The code here is new.

Containers carry the label s1arena.env=coding_controller plus the owning process id and host; they are removed when
the episode closes, at interpreter exit, and (for a process that died) by the next process that starts an episode.
"""

from __future__ import annotations

import atexit
import os
import secrets
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

LABEL = "s1arena.env=coding_controller"
WORKDIR = "/testbed"
SCRATCH = "/tmp/s1arena"  # harness files inside the container, outside the repository
CONDA = "source /opt/miniconda3/bin/activate && conda activate testbed"
COMMAND_ENV = {"PYTHONHASHSEED": "0", "PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb", "PIP_NO_INPUT": "1"}
MEMORY = "8g"  # per container; protects the host from a runaway command
# kept out of the graded patch (usually ignored by the repositories' .gitignore anyway)
EXCLUDES = ("':(exclude,glob)**/__pycache__/**'", "':(exclude,glob)**/*.pyc'", "':(exclude,glob)**/.pytest_cache/**'",
            "':(exclude,glob)**/*.egg-info/**'")

_LIVE: set[str] = set()  # containers this process started and has not removed yet
_LIVE_LOCK = threading.Lock()
_SWEPT = False


class ContainerError(RuntimeError):
    """Docker failed (not the command inside it): the episode says nothing about the model."""


class DockerTimeout(ContainerError):
    pass


def _docker(args: list[str], input: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], input=input, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise DockerTimeout(f"docker {args[0]} timed out after {timeout:.0f} s") from error
    except OSError as error:
        raise ContainerError(f"docker unavailable: {error}") from error


def _remove(name: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)
    except Exception:
        pass
    with _LIVE_LOCK:
        _LIVE.discard(name)


@atexit.register
def _remove_all() -> None:
    with _LIVE_LOCK:
        names = list(_LIVE)
    for name in names:
        _remove(name)


def sweep_orphans() -> list[str]:
    """Remove this env's containers on this host whose owning process is gone (a crashed or killed run)."""
    out = _docker(["ps", "-a", "--filter", f"label={LABEL}", "--format", '{{.Names}}\t{{.Label "s1arena.pid"}}\t{{.Label "s1arena.host"}}'])
    removed = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[2] != socket.gethostname() or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
            continue  # still running
        except ProcessLookupError:
            pass
        except PermissionError:
            continue  # another user's live process
        _remove(parts[0])
        removed.append(parts[0])
    return removed


def _sweep_once() -> None:
    global _SWEPT
    with _LIVE_LOCK:
        if _SWEPT:
            return
        _SWEPT = True
    sweep_orphans()


def _start(image: str, prefix: str, extra: list[str], command: list[str]) -> str:
    """docker run -d a labelled container; returns its name."""
    _sweep_once()
    name = f"{prefix}-{os.getpid()}-{secrets.token_hex(3)}"
    labels = ["--label", LABEL, "--label", f"s1arena.pid={os.getpid()}", "--label", f"s1arena.host={socket.gethostname()}"]
    with _LIVE_LOCK:
        _LIVE.add(name)
    out = _docker(["run", "-d", "--name", name, *labels, "--memory", MEMORY, *extra, image, *command], timeout=300)
    if out.returncode != 0:
        _remove(name)
        raise ContainerError(f"could not start {image}: {out.stderr.strip()[-300:]}")
    return name


@dataclass
class Result:
    exit_code: int
    output: str  # stdout and stderr interleaved, as the shell printed them
    seconds: float
    timed_out: bool = False


class Sandbox:
    """One episode's container."""

    def __init__(self, image: str, instance_id: str) -> None:
        self.image = image
        self.prefix = "jevcc-" + instance_id.replace("__", "-").replace("/", "-")[:40].lower()
        self.name: str | None = None

    def start(self) -> None:
        self.name = _start(self.image, self.prefix, ["--network", "none", "--hostname", "testbed"], ["sleep", "infinity"])
        setup = self._exec(f"mkdir -p {SCRATCH} && git config --global --add safe.directory {WORKDIR}")
        if setup.returncode != 0:
            raise ContainerError(f"container setup failed: {setup.stdout[-300:]}")

    def _exec(self, script: str, input: str | None = None, timeout: float = 300, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        """Harness-side shell in /testbed (root), stdout and stderr merged."""
        if self.name is None:
            raise ContainerError("the container is not running")
        flags = ["-i"] if input is not None else []
        for key, value in (env or {}).items():
            flags += ["-e", f"{key}={value}"]
        out = _docker(["exec", *flags, "-w", WORKDIR, self.name, "bash", "-c", f"{{ {script} ; }} 2>&1"], input=input, timeout=timeout)
        if out.returncode in (125, 126, 127) and "Error response from daemon" in (out.stderr or ""):
            raise ContainerError(out.stderr.strip()[-300:])
        return out

    def _write(self, path: str, content: str) -> None:
        out = self._exec(f"cat > {shlex.quote(path)}", input=content)
        if out.returncode != 0:
            raise ContainerError(f"could not write {path}: {out.stdout[-300:]}")

    def run(self, command: str, timeout: int) -> Result:
        """Run a command as the agent does: bash in /testbed, conda environment active, time limit, no stdin."""
        self._write(f"{SCRATCH}/command.sh", command if command.endswith("\n") else command + "\n")
        start = time.perf_counter()
        try:
            out = self._exec(f"{CONDA} && cd {WORKDIR} && timeout -k 5 {int(timeout)} bash {SCRATCH}/command.sh < /dev/null",
                             timeout=timeout + 60, env=COMMAND_ENV)
        except DockerTimeout:  # e.g. a background process kept the output open after the time limit
            return Result(124, "(no output: the command did not return)", time.perf_counter() - start, timed_out=True)
        seconds = time.perf_counter() - start
        return Result(out.returncode, out.stdout, seconds, timed_out=out.returncode in (124, 137) and seconds >= timeout - 1)

    def apply(self, patch: str) -> Result:
        """git apply (all or nothing), as an agent applying a unified diff would."""
        self._write(f"{SCRATCH}/candidate.diff", patch)
        start = time.perf_counter()
        out = self._exec(f"git apply --verbose {SCRATCH}/candidate.diff")
        return Result(out.returncode, out.stdout, time.perf_counter() - start)

    def snapshot(self) -> str:
        """The working tree (tracked and untracked files, not ignored ones) as a git tree id, via a private index."""
        index = f"GIT_INDEX_FILE={SCRATCH}/index"
        out = self._exec(f"{index} git -c core.fileMode=false add -A -- . {' '.join(EXCLUDES)} && {index} git write-tree")
        tree = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if out.returncode != 0 or len(tree) not in (40, 64):
            raise SnapshotError(f"could not read the working tree: {out.stdout.strip()[-300:]}")
        return tree

    def diff(self, base: str, tree: str | None = None) -> str:
        """The patch from the initial tree to the current one (exact, binary-safe)."""
        tree = tree or self.snapshot()
        if tree == base:
            return ""
        out = _docker(["exec", "-w", WORKDIR, self.name, "git", "diff", "--binary", "--no-color", base, tree], timeout=300)
        if out.returncode != 0:
            raise SnapshotError(f"git diff failed: {out.stderr.strip()[-300:]}")
        return out.stdout

    def revert(self, base: str) -> Result:
        """Restore every file to the initial tree: changed and deleted files rewritten, added files removed."""
        start = time.perf_counter()
        tree = self.snapshot()
        if tree == base:
            return Result(0, "Nothing to revert: the repository is unchanged.", time.perf_counter() - start)
        names = _docker(["exec", "-w", WORKDIR, self.name, "git", "diff", "--name-status", "--no-renames", "-z", base, tree], timeout=300)
        if names.returncode != 0:
            raise SnapshotError(f"git diff failed: {names.stderr.strip()[-300:]}")
        fields = names.stdout.split("\0")
        added, restore = [], []
        for status, path in zip(fields[0::2], fields[1::2]):
            (added if status.startswith("A") else restore).append(path)
        index = f"GIT_INDEX_FILE={SCRATCH}/revert_index"
        script = [f"{index} git read-tree {base}"]
        if restore:
            script.append(f"{index} git checkout-index -f -- " + " ".join(shlex.quote(p) for p in restore))
        if added:
            script.append("rm -f -- " + " ".join(shlex.quote(p) for p in added))
        out = self._exec(" && ".join(script))
        if out.returncode != 0 or self.snapshot() != base:
            raise SnapshotError(f"revert failed: {out.stdout.strip()[-300:]}")
        files = len(added) + len(restore)
        return Result(0, f"Reverted {files} file{'s' if files != 1 else ''}; the repository is back to its initial state.", time.perf_counter() - start)

    def stop(self) -> None:
        if self.name:
            _remove(self.name)
            self.name = None


class SnapshotError(RuntimeError):
    """The repository could not be read (e.g. an action deleted .git): the change cannot be extracted."""


# ---------------------------------------------------------------------------------------------------
# Grading: SWE-bench's own evaluation of one patch.

class GradingError(RuntimeError):
    """Infrastructure failure while grading (docker, or an environment fault swebench flags): not the model's doing."""


def grade(instance: dict[str, Any], patch: str, timeout: int = 1800, run_without_patch: bool = False) -> dict[str, Any]:
    """Grade a patch as swebench 5.0.2's run_instance does. run_without_patch runs the tests on the untouched
    repository (a validation check: the instance's FAIL_TO_PASS tests must then fail)."""
    from swebench.harness.constants import CONTAINER_PATCH_FILE
    from swebench.harness.grading import get_eval_report
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS
    from swebench.harness.utils import make_test_spec

    started = time.perf_counter()
    iid = instance["instance_id"]
    if not run_without_patch and not patch.strip():
        return {"resolved": False, "status": "empty_patch", "grade_s": 0.0}
    spec = make_test_spec(instance)
    name = _start(spec.image, "jevcc-grade-" + iid.replace("__", "-")[:34].lower(), [], ["tail", "-f", "/dev/null"])
    try:
        def sh(script: str, input: str | None = None, limit: float = 600) -> subprocess.CompletedProcess:
            return _docker(["exec", *(["-i"] if input is not None else []), "-w", WORKDIR, name, "bash", "-c", script], input=input, timeout=limit)

        apply_used = None
        if not run_without_patch:
            if sh(f"cat > {CONTAINER_PATCH_FILE}", input=patch).returncode != 0:
                raise GradingError("could not copy the patch into the grading container")
            for attempt, command in enumerate(GIT_APPLY_CMDS):
                if attempt:  # a failed attempt can leave partial state; restart from a pristine tree, as the harness does
                    sh("git checkout -- . ; git clean -fd")
                if sh(f"{command} {CONTAINER_PATCH_FILE}").returncode == 0:
                    apply_used = command
                    break
            if apply_used is None and sh(f"git apply --check --reverse {CONTAINER_PATCH_FILE}").returncode == 0:
                apply_used = "already applied"
            if apply_used is None:
                return {"resolved": False, "status": "patch_apply_failed", "grade_s": round(time.perf_counter() - started, 1)}
        if sh("cat > /eval.sh", input=spec.eval_script).returncode != 0:
            raise GradingError("could not copy the eval script into the grading container")
        test_start = time.perf_counter()
        try:  # one stream, in the order written: the harness reads docker's combined stream (set -x markers are on stderr)
            out = _docker(["exec", name, "bash", "-c", "/bin/bash /eval.sh 2>&1"], timeout=timeout)
            log, timed_out = (out.stdout or "") + (out.stderr or ""), False
        except DockerTimeout:
            log, timed_out = "", True
        test_s = time.perf_counter() - test_start
        if timed_out:
            return {"resolved": False, "status": "timeout", "grade_s": round(time.perf_counter() - started, 1), "test_s": round(test_s, 1)}
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(log)
            log_path = f.name
        try:
            prediction = {"instance_id": iid, "model_name_or_path": "s1arena", "model_patch": patch}
            report = get_eval_report(test_spec=spec, prediction=prediction, test_log_path=log_path, include_tests_status=True)[iid]
        finally:
            os.unlink(log_path)
        if report.get("infra_failure"):
            raise GradingError(f"{iid}: environment fault while testing ({report.get('infra_failure_reason')})")
        result = {"resolved": bool(report["resolved"]), "status": "graded" if report.get("patch_successfully_applied") else "tests_not_parsed",
                  "grade_s": round(time.perf_counter() - started, 1), "test_s": round(test_s, 1)}
        if apply_used and apply_used != GIT_APPLY_CMDS[0]:
            result["apply_command"] = apply_used
        status = report.get("tests_status") or {}
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if key in status:
                passed, failed = len(status[key].get("success", [])), len(status[key].get("failure", []))
                result[key.lower()] = f"{passed}/{passed + failed}"
        if not result["resolved"]:
            result["log_tail"] = log[-1500:]
        return result
    finally:
        _remove(name)
