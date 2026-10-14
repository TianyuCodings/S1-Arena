"""Domain: agent single-step decisions, static (Agent 单步决策（静态）).

Single decisions cut from existing agent trajectories: the next GUI action, the value of a tool
argument, whether a proposed step is right, whether a run will succeed, and which agent and step
caused a failure. Gold answers come from expert demonstrations, recorded environment outcomes,
execution-checked ground truth or human annotations of the source datasets, never from our own
labeling.
"""
