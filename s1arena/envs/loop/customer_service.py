"""Customer service with τ²-bench: the telecom domain in its no-user ("solo") mode.

τ²-bench (Barres et al., 2025; sierra-research/tau2-bench, MIT) simulates customer-service conversations. In the
telecom domain's no-user ablation (`tau2 run --domain telecom --agent llm_agent_solo --user dummy_user`) the agent
gets a ticket instead of a customer and solves it with tool calls only: the 13 agent tools (customer, line, bill and
roaming operations) plus the 30 tools of the customer's phone, which in solo mode the agent calls itself, and a
`done` tool that ends the task. Nothing in this harness calls a language model: the tools and the grader are
deterministic code.

The episode runs inside τ²'s own objects: its Environment (telecom domain, solo mode), its Orchestrator (with
DummyUser, max_steps=200 and max_errors=10, τ²'s defaults) and its evaluator. The harness plays the agent: a
subclass of τ²'s LLMSoloAgent whose next message is the tool call the model chose instead of an LLM completion.
The orchestrator is stepped exactly as its run() loop would step it. DummyUser is constructed directly, as τ²'s gym
does (at this commit τ²'s runner passes it arguments its constructor does not take, so the CLI command above fails).

The state is what τ² gives the solo agent: its instructions (τ²'s solo-agent instructions with the two sentences
about JSON output and the stop message rewritten for the option interface), the domain policy verbatim (the solo
main policy and the technical-support manual, 22,819 characters), the ticket verbatim, and every tool call made so
far with its result as the agent receives it (τ²'s function notation; the result verbatim, errors included). Results
older than the newest HISTORY_CHARS characters of results are left out (the calls stay listed).

Each tool call takes one or two decisions:
1. "tool": a choice over the tools, in τ²'s order (agent tools, phone tools, done); each option is the tool's name,
   its argument names and the description from its schema. Choosing done ends the episode.
2. only for a tool with arguments, a second request with one question per argument (customer_service_args.py):
   a choice over candidate values collected from the tool's schema (enum values, defaults), the ticket, the policy's
   code literals and numbers, and earlier tool results (IDs, phone numbers, names, dates, amounts...); true/false for
   a boolean; object arguments field by field; list arguments as one boolean per candidate.
The call then runs through τ²'s environment and its result joins the history. A decision without a usable answer
calls nothing (it still counts); in the second request it also drops the chosen tool.

Outcome: τ²'s evaluator reward (EvaluationType.ALL; for telecom the reward basis is the environment assertions,
plus the expected actions for the 20 tasks whose expected action is a transfer to a human agent) as the score,
success = reward 1. As in τ², an episode that did not end with `done` scores 0: τ²'s step limit (200 orchestrator steps, i.e.
about 100 tool calls), 10 tool errors, or this harness's decision budget (max_decisions; reported as
"decision_limit"). The outcome also reports the termination reason, tool calls, tool errors and the calls made.

Tasks: τ²'s default "base" split of the telecom domain, 114 tasks (the split τ²'s README says to evaluate agents on;
its 40-task "test" split is a subset of it). All 114 are valid solo tasks (a ticket and expected actions). Seed i is
the i-th task after a fixed shuffle (so any prefix of seeds mixes the three issue types); the outcome names the τ²
task id.

Determinism: the telecom tools are deterministic except for one call to uuid4 (the ID of a new draft bill when data is
refuelled for a customer without one); the harness makes that call draw from a generator seeded by the episode (and
reseeds it before τ²'s evaluator replays the episode, so the replay creates the same IDs). Tool-call ids are
numbered. The same seed and the same decisions give the same episode.

Data: tau2 is installed from git without its data directory. On first use the repository tarball at the pinned
commit is streamed and only the telecom files the domain reads are kept (checksums verified) under .cache/tau2/;
TAU2_DATA_DIR points there before tau2 is imported (tau2 reads it at import time).
"""

from __future__ import annotations

import hashlib
import os
import random
import tarfile
import threading
import time
import urllib.request
import uuid
from typing import Any

from ...env import ROOT
from ...registry import register_env
from ...schema import DecisionRequest, Question
from ..base import Environment
from .customer_service_args import ArgForm, Values, arg_form, arg_specs, output_values, policy_values, ticket_values

COMMIT = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"  # the tau2 commit pinned in pyproject.toml (tau2 1.0.1)
TARBALL = f"https://codeload.github.com/sierra-research/tau2-bench/tar.gz/{COMMIT}"
CACHE = ROOT / ".cache" / "tau2"
DATA_DIR = CACHE / "data"  # TAU2_DATA_DIR; tau2 reads <TAU2_DATA_DIR>/tau2/domains/telecom/...
FILES = {  # every file the telecom domain reads (solo and normal mode, manual policy), sha256 at COMMIT
    "data/tau2/domains/telecom/db.toml": "562d647ef9d7df8df91eafd8ee76036e707c8f9e32aaedc4a7fde06975aea2c0",
    "data/tau2/domains/telecom/user_db.toml": "4886107ea0c16f8e16d74bef91cfb557de61822afe9361f5a9b241806aea2e4c",
    "data/tau2/domains/telecom/main_policy.md": "95943844a0cf11fdf0e2842b483b81d9f9338aa53235712ed131db075d086ed7",
    "data/tau2/domains/telecom/main_policy_solo.md": "781296ae5419169cb64aac0c4fa2aa6129c78f3c1592f64abd80df6fbabb7042",
    "data/tau2/domains/telecom/tech_support_manual.md": "015a3ee49ec8c199c5e1a059c922081937b92cf3a5ea74fedbc4357816f389ba",
    "data/tau2/domains/telecom/tasks.json": "37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4",
    "data/tau2/domains/telecom/split_tasks.json": "605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec",
}
DOMAIN = "telecom"  # τ²'s registry name: telecom with the manual (non-workflow) technical-support policy
TASK_SPLIT = "base"
TASK_ORDER_SEED = "s1arena/loop/customer_service/tau2_telecom_solo"
MAX_STEPS, MAX_ERRORS = 200, 10  # τ²'s defaults (tau2.config DEFAULT_MAX_STEPS, DEFAULT_MAX_ERRORS)
HISTORY_CHARS = 60_000  # results shown in the history, newest first; older results are left out
STOP = "done"  # the stop tool τ²'s LLMSoloAgent adds
NO_OP = "none"  # a decision without a usable answer

# τ²'s AGENT_SOLO_INSTRUCTION (tau2/agent/llm_agent.py). The stop sentence and the JSON sentence are rewritten for the
# option interface: "To do so, send a message containing a single tool call to the `done` tool. Do not include any
# other tool calls in this last message." and "Always make sure you generate valid JSON only."
INSTRUCTIONS = """You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, call the `done` tool.

Always follow the policy."""

_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------------
# τ² data and imports.


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download() -> None:
    """Stream the repository tarball at COMMIT and keep only FILES (checked against their sha256)."""
    prefix = f"tau2-bench-{COMMIT}/"
    found: dict[str, bytes] = {}
    request = urllib.request.Request(TARBALL, headers={"User-Agent": "s1arena"})
    with urllib.request.urlopen(request, timeout=600) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            name = member.name[len(prefix):] if member.name.startswith(prefix) else ""
            if name in FILES and member.isfile():
                found[name] = tar.extractfile(member).read()
    for name, digest in FILES.items():
        if name not in found:
            raise RuntimeError(f"tau2-bench tarball at {COMMIT[:7]} has no {name}")
        if _sha256(found[name]) != digest:
            raise RuntimeError(f"tau2-bench {name}: checksum mismatch")
    for name, data in found.items():
        path = CACHE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)


def _ensure_data() -> None:
    if not all((CACHE / name).exists() for name in FILES):
        _download()
    os.environ["TAU2_DATA_DIR"] = str(DATA_DIR)


class _SeededUUID:
    """Stands in for the uuid module inside tau2's telecom tools: uuid4() draws from the generator the running
    episode set for this thread (the tools create draft-bill IDs with it)."""

    def __init__(self) -> None:
        self.local = threading.local()

    def uuid4(self) -> uuid.UUID:
        rng = getattr(self.local, "rng", None)
        return uuid.uuid4() if rng is None else uuid.UUID(int=rng.getrandbits(128), version=4)

    def __getattr__(self, name: str) -> Any:
        return getattr(uuid, name)


_UUIDS = _SeededUUID()


def _tau2() -> dict[str, Any]:
    """Import tau2 once (data in place, logging off) and keep what the harness uses."""
    with _LOCK:
        if "tau2" in _CACHE:
            return _CACHE["tau2"]
        _ensure_data()
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no network lookup when litellm is imported
        from loguru import logger

        logger.disable("tau2")
        import tau2.domains.telecom.tools as telecom_tools
        import tau2.utils.utils as tau2_utils
        from tau2.agent.llm_agent import LLMSoloAgent
        from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, UserMessage
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.registry import registry
        from tau2.user.user_simulator import DummyUser
        from tau2.utils.tools import to_functional_format
        from tau2.utils.utils import get_now

        if os.path.realpath(tau2_utils.DATA_DIR) != os.path.realpath(DATA_DIR):
            raise RuntimeError(f"tau2 was imported with TAU2_DATA_DIR={tau2_utils.DATA_DIR}, expected {DATA_DIR}")
        telecom_tools.uuid = _UUIDS

        class HarnessSoloAgent(LLMSoloAgent):
            """τ²'s solo agent with the LLM replaced by the model's decision: generate_next_message returns the
            message the harness set (a done call becomes τ²'s stop message, as in LLMSoloAgent)."""

            def __init__(self, tools, domain_policy, task):
                super().__init__(tools=tools, domain_policy=domain_policy, task=task, llm="s1arena-harness")
                self.next_message = None

            def generate_next_message(self, message, state):
                if isinstance(message, UserMessage):
                    raise ValueError("the solo agent does not take user messages")
                if isinstance(message, MultiToolMessage):
                    state.messages.extend(message.tool_messages)
                elif message is not None:
                    state.messages.append(message)
                reply, self.next_message = self.next_message, None
                if reply is None:
                    raise RuntimeError("no decision to send")
                reply = self._check_if_stop_toolcall(reply)
                state.messages.append(reply)
                return reply, state

        tasks = registry.get_tasks_loader(DOMAIN)(TASK_SPLIT)
        base_index = {task.id: i for i, task in enumerate(tasks)}
        order = list(tasks)
        random.Random(TASK_ORDER_SEED).shuffle(order)
        _CACHE["tau2"] = {
            "registry": registry, "Orchestrator": Orchestrator, "DummyUser": DummyUser, "Agent": HarnessSoloAgent,
            "AssistantMessage": AssistantMessage, "ToolCall": ToolCall, "evaluate": evaluate_simulation,
            "ALL": EvaluationType.ALL, "functional": to_functional_format, "now": get_now,
            "tasks": order, "base_index": base_index,
        }
        return _CACHE["tau2"]


def _policy_values(policy: str) -> Values:
    with _LOCK:
        if _CACHE.get("policy") != policy:
            _CACHE["policy"], _CACHE["policy_values"] = policy, policy_values(policy)
        return _CACHE["policy_values"]


# ---------------------------------------------------------------------------------------------------
# Environment.


@register_env
class Tau2TelecomSolo(Environment):
    name = "loop/customer_service/tau2_telecom_solo"
    leaf = "loop/customer_service"
    source = "τ²-bench (Barres et al., 2025), telecom domain, no-user (solo) mode; tau2 1.0.1 at commit b7ea907"
    url = "https://github.com/sierra-research/tau2-bench"
    license = "MIT"
    tier = "B"
    criteria_source = ("Ticket and domain policy verbatim from τ² (solo policy: main_policy_solo.md + tech_support_manual.md); τ²'s "
                       "solo-agent instructions with the stop and JSON sentences rewritten for the options; tools, descriptions and "
                       "argument types from τ²'s tool schemas; reward from τ²'s own evaluator")
    notes = ("τ² telecom 'base' split (114 tasks: 49 MMS, 36 mobile-data and 29 no-service tickets), fixed shuffle. Solo mode: the agent "
             "calls its 13 tools and the 30 phone tools itself, no user and no LLM anywhere. One decision picks a tool (43 tools + done); "
             "a tool with arguments takes a second decision with one question per argument over values found in the schema, ticket, "
             "policy and earlier tool results. Score = τ²'s reward (environment assertions; plus expected actions for the 20 "
             "transfer tasks), success = reward 1; ends at done, after τ²'s 200 steps (~100 tool calls) or 10 tool errors "
             "(reward 0), or at 200 decisions (reward 0).")
    mode = "paused"
    episodes = 114
    max_decisions = 200  # τ²'s max_steps is 200; a tool call takes one or two decisions here, two orchestrator steps there
    score_name = "reward"

    def reset(self, seed: int) -> None:
        t = _tau2()
        if not 0 <= seed < len(t["tasks"]):
            raise ValueError(f"seed {seed} out of range: {len(t['tasks'])} tasks")
        self.t = t
        self.seed = seed
        self.task = t["tasks"][seed]
        self.environment = t["registry"].get_env_constructor(DOMAIN)(solo_mode=True)
        tools = self.environment.get_tools() + self.environment.get_user_tools(include=self.task.user_tools)
        self.agent = t["Agent"](tools=tools, domain_policy=self.environment.get_policy(), task=self.task)  # adds done
        self.orchestrator = t["Orchestrator"](domain=DOMAIN, agent=self.agent, user=t["DummyUser"](), environment=self.environment,
                                              task=self.task, max_steps=MAX_STEPS, max_errors=MAX_ERRORS, seed=None, solo_mode=True)
        self.policy = self.environment.get_policy()
        self.tools = {tool.name: tool for tool in self.agent.tools}
        self.specs = {name: arg_specs(tool.openai_schema["function"]["parameters"]) for name, tool in self.tools.items()}
        self.tool_options = {name: self._describe(name) for name in self.tools}
        self.pool = ticket_values(self.task.ticket)
        self.pool.extend(_policy_values(self.policy))
        self.entries: list[tuple[Any, Any] | None] = []  # (ToolCall, ToolMessage) per call; None for a decision that called nothing
        self.pending: str | None = None  # a tool with arguments waiting for its argument decision
        self.form: ArgForm | None = None
        self.started = False
        self.uuid_seed = f"{TASK_ORDER_SEED}/{seed}"
        self.uuid_rng = random.Random(self.uuid_seed)
        self._outcome: dict[str, Any] | None = None

    def _describe(self, name: str) -> str:
        function = self.tools[name].openai_schema["function"]
        params = ", ".join(function["parameters"].get("properties", {}))
        return f"{name}({params}): {function['description'].strip()}"

    # --- requests ---

    def _history(self) -> str:
        if not self.entries:
            return "none yet"
        lines, budget, n = [], HISTORY_CHARS, sum(entry is not None for entry in self.entries)
        for entry in reversed(self.entries):
            if entry is None:
                lines.append("(no usable answer: nothing was called)")
                continue
            call, result = entry
            content = str(result.content)
            if len(content) <= budget:
                budget -= len(content)
                shown = content.replace("\n", "\n   ")
            else:
                budget = 0
                shown = f"(result left out: only the newest {HISTORY_CHARS:,} characters of results are shown)"
            lines.append(f"{n}. {self.t['functional'](call)}\n   → {shown}")
            n -= 1
        return "\n".join(reversed(lines))

    def _form(self, tool: str) -> ArgForm:
        if self.form is None or self.form.tool != tool:
            self.form = arg_form(tool, self.specs[tool], self.pool)
        return self.form

    def observe(self) -> DecisionRequest:
        state = {"instructions": INSTRUCTIONS, "policy": self.policy, "ticket": self.task.ticket, "tool calls so far": self._history()}
        if self.pending is None:
            question = Question(type="choice", criteria=dict(self.tool_options), instructions=(
                "Which tool do you call next? Each option is a tool: its name, its arguments and the description from its schema. "
                "The arguments of the chosen tool are asked next; done ends the task."))
            return DecisionRequest(state=state, questions={"tool": question})
        state["tool being called"] = self.tool_options[self.pending]
        return DecisionRequest(state=state, questions=dict(self._form(self.pending).questions))

    def default_choice(self, request: DecisionRequest) -> dict[str, str]:
        """No usable answer: call nothing."""
        return {name: NO_OP for name in request.questions}

    # --- actions ---

    def act(self, choice: dict[str, str]) -> None:
        if self.pending is None:
            tool = choice.get("tool", NO_OP)
            if tool not in self.tools:
                self.entries.append(None)
                return
            form = self._form(tool)
            if form.questions:
                self.pending = tool
                return
            self._call(tool, form.arguments({}) or {})
            return
        tool, self.pending = self.pending, None
        arguments = self._form(tool).arguments(choice)
        if arguments is None:
            self.entries.append(None)
            return
        self._call(tool, arguments)

    def _call(self, tool: str, arguments: dict[str, Any]) -> None:
        """Send one tool call through τ²'s orchestrator, stepping it as Orchestrator.run() does."""
        t, orchestrator = self.t, self.orchestrator
        number = sum(entry is not None for entry in self.entries) + 1
        call = t["ToolCall"](id=f"call_{number}", name=tool, arguments=arguments, requestor="assistant")
        self.agent.next_message = t["AssistantMessage"](role="assistant", content=None, tool_calls=[call])
        _UUIDS.local.rng = self.uuid_rng
        try:
            if not self.started:
                self.started = True
                orchestrator._run_start_time = t["now"]()
                orchestrator._run_start_perf = time.perf_counter()
                orchestrator.initialize()  # sets up the task's initial state, then takes the first message
            else:
                orchestrator.step()  # the agent's turn
                orchestrator._check_termination()
            if not orchestrator.done:
                orchestrator.step()  # the environment runs the call
                orchestrator._check_termination()
                self.entries.append((call, orchestrator.message))
                self.pool.extend(output_values(orchestrator.message.content))
        finally:
            _UUIDS.local.rng = None
        self.form = None

    def done(self) -> bool:
        return self.started and self.orchestrator.done

    # --- outcome ---

    def outcome(self) -> dict[str, Any]:
        if self._outcome is None:
            self._outcome = self._evaluate()
        return self._outcome

    def _evaluate(self) -> dict[str, Any]:
        calls = [entry for entry in self.entries if entry is not None]
        info = {"task_id": self.task.id, "tau2_base_index": self.t["base_index"][self.task.id], "tool_calls": len(calls),
                "tool_errors": sum(bool(result.error) for _, result in calls), "calls": [self.t["functional"](call) for call, _ in calls]}
        if not self.done():  # the decision budget ran out before the agent called done: τ² scores an unfinished run 0
            return {"success": False, "score": 0.0, "termination": "decision_limit", **info}
        simulation = self.orchestrator._finalize()
        _UUIDS.local.rng = random.Random(self.uuid_seed)  # τ²'s replay of the episode creates the same draft-bill IDs
        try:
            reward = self.t["evaluate"](simulation=simulation, task=self.task, evaluation_type=self.t["ALL"], solo_mode=True, domain=DOMAIN)
        finally:
            _UUIDS.local.rng = None
        termination = getattr(simulation.termination_reason, "value", simulation.termination_reason)
        score = float(reward.reward)
        out = {"success": score == 1.0, "score": score, "termination": termination, **info}
        if reward.reward_breakdown:
            out["reward_breakdown"] = {getattr(k, "value", k): v for k, v in reward.reward_breakdown.items()}
        if reward.env_assertions:
            out["assertions_met"] = [bool(check.met) for check in reward.env_assertions]
        if reward.action_checks and "reward_breakdown" in out and "ACTION" in out["reward_breakdown"]:
            out["actions_matched"] = [bool(check.action_match) for check in reward.action_checks]
        return out

    def close(self) -> None:
        for attribute in ("orchestrator", "environment", "agent"):
            if hasattr(self, attribute):
                setattr(self, attribute, None)

    def card(self) -> dict[str, Any]:
        return {**super().card(), "tau2_commit": COMMIT, "task_split": TASK_SPLIT, "tau2_max_steps": MAX_STEPS, "tau2_max_errors": MAX_ERRORS}
