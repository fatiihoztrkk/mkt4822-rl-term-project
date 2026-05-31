"""200-episode benchmark for the contextual Thompson bandit.

Simulates the hard parts of the runtime (model variability, multi-attempt
repair) without touching Ollama. We construct a synthetic two-axis world:

  task_family x arm_id  -->  per-attempt success probability

Different arms are *good at* different families (e.g. ``cool`` arms shine on
numerical tasks, ``warm`` arms on string tasks). The bandit must discover
this association from reward signals only.

For each task we run an episode with a model client that returns broken
code with probability ``1 - p`` and correct code otherwise. The runtime
loop applies the arm-selected ModelProfile, runs the real validation
pipeline (syntax + import + runtime), and the bandit updates its posterior
from :func:`episode_reward`.

Reports:
  * baseline (no learning) success rate and mean attempts
  * bandit success rate, mean attempts, and per-arm pull breakdown
  * top-3 arms per task family
  * runtime overhead in milliseconds per episode

Run with:

    python scripts/bandit_benchmark.py --episodes 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mecha_agent_cli.agent.direct_loop import DirectAgentLoop
from mecha_agent_cli.app.commands import init_command
from mecha_agent_cli.config.loader import load_config
from mecha_agent_cli.config.schema import AppConfig, ModelProfile
from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY
from mecha_agent_cli.llm.base import ChatMessage, ModelClient

# A correct ``algorithm.py`` template per task family. All compile, import,
# and run cleanly under the validator (syntax + import + runtime checks).
_GOOD_TEMPLATES: dict[str, str] = {
    "sort": '''"""Merge sort."""
from __future__ import annotations
def solve(data: list[int]) -> list[int]:
    """Return a stable sorted copy."""
    values = list(data)
    if len(values) <= 1:
        return values
    mid = len(values) // 2
    left = solve(values[:mid])
    right = solve(values[mid:])
    out: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out
if __name__ == "__main__":
    print(solve([3, 1, 2]))
''',
    "fibonacci_dp": '''"""Fibonacci."""
from __future__ import annotations
def solve(n: int) -> int:
    """Return nth Fibonacci."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
if __name__ == "__main__":
    print(solve(10))
''',
    "signal_filter": '''"""Moving average."""
from __future__ import annotations
def solve(data: list[float], window: int) -> list[float]:
    """Causal moving average."""
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float] = []
    s = 0.0
    for i, v in enumerate(data):
        s += v
        if i >= window:
            s -= data[i - window]
            d = window
        else:
            d = i + 1
        out.append(s / d)
    return out
if __name__ == "__main__":
    print(solve([1.0, 2.0, 3.0, 4.0], 2))
''',
    "numerical": '''"""Linear solve via Gauss-Seidel."""
from __future__ import annotations
def solve(a: list[list[float]], b: list[float], iters: int = 50) -> list[float]:
    """Approximate solution to ax = b."""
    n = len(b)
    x = [0.0] * n
    for _ in range(iters):
        for i in range(n):
            s = sum(a[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / a[i][i]
    return x
if __name__ == "__main__":
    print(solve([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0]))
''',
    "graph": '''"""BFS shortest path."""
from __future__ import annotations
from collections import deque
def solve(adj: dict[int, list[int]], src: int, dst: int) -> int:
    """Return BFS distance, -1 if unreachable."""
    if src == dst:
        return 0
    seen = {src}
    q: deque[tuple[int, int]] = deque([(src, 0)])
    while q:
        node, d = q.popleft()
        for nb in adj.get(node, []):
            if nb == dst:
                return d + 1
            if nb not in seen:
                seen.add(nb); q.append((nb, d + 1))
    return -1
if __name__ == "__main__":
    print(solve({0: [1], 1: [2], 2: []}, 0, 2))
''',
    "string": '''"""Word count."""
from __future__ import annotations
def solve(text: str) -> dict[str, int]:
    """Lowercase word frequencies."""
    out: dict[str, int] = {}
    for word in text.lower().split():
        out[word] = out.get(word, 0) + 1
    return out
if __name__ == "__main__":
    print(solve("a b a"))
''',
    "io": '''"""JSON round-trip."""
from __future__ import annotations
import json
def solve(payload: dict[str, object]) -> dict[str, object]:
    """Serialize then deserialize."""
    return json.loads(json.dumps(payload))
if __name__ == "__main__":
    print(solve({"x": 1}))
''',
    "generic": '''"""Generic solver."""
from __future__ import annotations
def solve(request: dict[str, object]) -> dict[str, object]:
    """Echo a deterministic envelope."""
    return {"outputs": request, "ok": True}
if __name__ == "__main__":
    print(solve({"a": 1}))
''',
}

# Broken code templates: each fails the *runtime* check (uncaught exception)
# but parses and imports. This mirrors the dominant failure mode reported in
# the repo memory ("RuntimeError after import smoke check").
_BROKEN_TEMPLATES = (
    '''"""buggy."""
def solve(*args, **kwargs):
    raise RuntimeError("not implemented")
if __name__ == "__main__":
    solve()
''',
    '''"""buggy."""
def solve(data=None):
    return data[0]
if __name__ == "__main__":
    solve(None)
''',
    '''"""buggy."""
def solve(n=0):
    return 1 / 0
if __name__ == "__main__":
    solve()
''',
)


# Per-task / per-arm true success probability. Designed so:
# - "direct.baseline" is decent everywhere (~0.6 success on first attempt)
# - "direct.cool"/"direct.cold" excel at numerical & fibonacci_dp
# - "direct.warm"/"direct.hot" excel at string & generic
# - "direct.high_repeat" excels at signal_filter (avoids loops)
# - other arms are noisy.
def _success_prob(family: str, arm_id: str) -> float:
    base = {
        "sort": 0.55,
        "fibonacci_dp": 0.50,
        "signal_filter": 0.45,
        "numerical": 0.30,
        "graph": 0.50,
        "string": 0.65,
        "io": 0.70,
        "generic": 0.60,
    }.get(family, 0.50)
    bonus_table: dict[tuple[str, str], float] = {
        ("numerical", "direct.cool"): 0.45,
        ("numerical", "direct.cold"): 0.50,
        ("fibonacci_dp", "direct.cool"): 0.30,
        ("fibonacci_dp", "direct.cold"): 0.35,
        ("string", "direct.warm"): 0.20,
        ("string", "direct.hot"): 0.25,
        ("generic", "direct.warm"): 0.20,
        ("signal_filter", "direct.high_repeat"): 0.40,
        ("signal_filter", "direct.cool"): 0.20,
        ("graph", "direct.tight_topk"): 0.25,
        ("sort", "direct.broad_topk"): 0.20,
        ("io", "direct.no_think"): 0.20,
    }
    bonus = bonus_table.get((family, arm_id), 0.0)
    penalty = -0.20 if arm_id == "direct.fixed_seed" else 0.0
    p = base + bonus + penalty - _BASE_PENALTY
    return max(0.05, min(0.95, p))


_BASE_PENALTY: float = 0.0
_MAX_ATTEMPTS: int = 6


class _OracleClient(ModelClient):
    """Synthetic backend whose success rate depends on (family, arm) pair.

    The arm is identified by inspecting the ModelProfile sampling parameters
    against the canonical ARM_REGISTRY entries. This lets the loop's existing
    bandit-injection path stay intact end-to-end (no special hooks).
    """

    def __init__(self, family: str, rng: random.Random) -> None:
        self.family = family
        self.rng = rng
        self.calls = 0
        self._arm_signatures = {arm.arm_id: _profile_signature(arm) for arm in ARM_REGISTRY}

    @property
    def name(self) -> str:
        return "oracle"

    def available_models(self) -> list[str]:
        return ["qwen3:4b"]

    def chat_text(self, *, messages: Sequence[ChatMessage], profile: ModelProfile, **_: object) -> str:
        del messages
        self.calls += 1
        arm_id = _profile_to_arm_id(profile, self._arm_signatures)
        p = _success_prob(self.family, arm_id)
        good = _GOOD_TEMPLATES.get(self.family, _GOOD_TEMPLATES["generic"])
        if self.rng.random() < p:
            return f"```python\n{good}```\n"
        # On second/third attempt give a slight boost to mimic improvement
        if self.calls >= 3 and self.rng.random() < 0.25:
            return f"```python\n{good}```\n"
        return f"```python\n{self.rng.choice(_BROKEN_TEMPLATES)}```\n"


def _profile_signature(arm: Any) -> tuple[float, float, float, str | None, int | None, bool | None, int | None]:
    base = ModelProfile(
        think=True,
        temperature=0.4,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=16384,
        num_predict=-1,
        keep_alive="30m",
    )
    p = arm.apply(base)
    return (p.temperature, p.top_p, p.repeat_penalty, p.keep_alive, p.top_k, p.think, p.seed)


def _profile_to_arm_id(profile: ModelProfile, sigs: dict[str, tuple]) -> str:
    sig = (
        profile.temperature,
        profile.top_p,
        profile.repeat_penalty,
        profile.keep_alive,
        profile.top_k,
        profile.think,
        profile.seed,
    )
    for arm_id, ref in sigs.items():
        if sig == ref:
            return arm_id
    return "direct.baseline"


_TASKS_BY_FAMILY: dict[str, str] = {
    "sort": "Implement merge sort solve()",
    "fibonacci_dp": "Implement fibonacci solve(n)",
    "signal_filter": "Implement a moving average filter solve()",
    "numerical": "Solve a numerical linear matrix system",
    "graph": "Implement BFS shortest path on a graph",
    "string": "Implement a string word counter",
    "io": "Implement a JSON round-trip serializer",
    "generic": "Solve a generic computation task",
}


def _make_workspace(tmp: Path) -> AppConfig:
    init_command(tmp)
    (tmp / "configs" / "validation.yaml").write_text(
        "target_file: algorithm.py\nrun_pytest: false\nrun_ruff: false\nrun_pyright: false\n"
    )
    return load_config(tmp)


def _enable_learning(repo: Path) -> None:
    (repo / "configs" / "default.yaml").write_text(
        (repo / "configs" / "default.yaml").read_text()
        + '\nlearning:\n  enabled: true\n  mode: "thompson"\n  min_pulls_before_exploit: 1\n  persist: true\n'
    )


def run_episode(
    *,
    tmp: Path,
    family: str,
    rng: random.Random,
    use_bandit: bool,
) -> tuple[bool, int, float, str]:
    """Run one episode in an isolated workspace; return (success, attempts, ms, arm_id)."""
    cfg = _make_workspace(tmp)
    if use_bandit:
        _enable_learning(tmp)
        cfg = load_config(tmp)
    client = _OracleClient(family=family, rng=rng)
    loop = DirectAgentLoop(repo_root=tmp, config=cfg, client=client, max_attempts=_MAX_ATTEMPTS)
    t0 = time.perf_counter()
    result = loop.run(_TASKS_BY_FAMILY[family])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    arm_id = result.strategy_name.split(":", 1)[1] if result.strategy_name.startswith("bandit:") else "n/a"
    return result.status == "success", result.total_attempts, elapsed_ms, arm_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--base-penalty",
        type=float,
        default=0.20,
        help="Subtract this from every base-success rate to harden the benchmark.",
    )
    parser.add_argument(
        "--shared-bandit",
        action="store_true",
        help="Persist bandit state across episodes (recommended).",
    )
    args = parser.parse_args()
    global _BASE_PENALTY, _MAX_ATTEMPTS
    _BASE_PENALTY = args.base_penalty
    _MAX_ATTEMPTS = args.max_attempts

    families = list(_TASKS_BY_FAMILY)
    rng = random.Random(args.seed)

    print(f"Running {args.episodes} baseline episodes (no learning)...")
    base = _run_arm("baseline", args.episodes, families, rng, use_bandit=False, shared_bandit=False)

    rng = random.Random(args.seed)
    print(f"Running {args.episodes} bandit episodes ({'shared' if args.shared_bandit else 'fresh'} bandit)...")
    bandit = _run_arm("bandit", args.episodes, families, rng, use_bandit=True, shared_bandit=args.shared_bandit)

    print("\n=== RESULTS ===")
    _print_arm("BASELINE", base)
    _print_arm("BANDIT  ", bandit)
    print(f"\nDelta success rate: {(bandit['success_rate'] - base['success_rate']) * 100:+.1f} pp")
    print(f"Delta mean attempts: {bandit['mean_attempts'] - base['mean_attempts']:+.2f}")
    print(f"Bandit overhead per episode: {bandit['mean_overhead_ms']:.2f} ms")

    print("\n=== ARM PULLS BY FAMILY (top 3) ===")
    for family in families:
        counts = Counter(bandit["arms_per_family"][family])
        top = counts.most_common(3)
        print(f"  {family:14s}: " + ", ".join(f"{a}({n})" for a, n in top))


def _run_arm(
    label: str,
    episodes: int,
    families: list[str],
    rng: random.Random,
    *,
    use_bandit: bool,
    shared_bandit: bool,
) -> dict[str, Any]:
    successes = 0
    attempts: list[int] = []
    overheads: list[float] = []
    arms_per_family: dict[str, list[str]] = defaultdict(list)
    last_db: Path | None = None
    if shared_bandit:
        # Single workspace re-used across episodes so the bandit accumulates state.
        shared_dir = TemporaryDirectory(prefix=f"bandit-bench-{label}-")
        shared_path = Path(shared_dir.name)
        cfg = _make_workspace(shared_path)
        if use_bandit:
            _enable_learning(shared_path)
            cfg = load_config(shared_path)
        last_db = shared_path / cfg.memory.path
        try:
            for i in range(episodes):
                family = families[i % len(families)]
                client = _OracleClient(family=family, rng=rng)
                loop = DirectAgentLoop(repo_root=shared_path, config=cfg, client=client, max_attempts=_MAX_ATTEMPTS)
                t0 = time.perf_counter()
                result = loop.run(_TASKS_BY_FAMILY[family])
                overheads.append((time.perf_counter() - t0) * 1000.0)
                attempts.append(result.total_attempts)
                successes += int(result.status == "success")
                arm_id = (
                    result.strategy_name.split(":", 1)[1] if result.strategy_name.startswith("bandit:") else "static"
                )
                arms_per_family[family].append(arm_id)
        finally:
            shared_dir.cleanup()
    else:
        for i in range(episodes):
            family = families[i % len(families)]
            with TemporaryDirectory(prefix=f"bandit-bench-{label}-") as tmp:
                ok, n, ms, arm_id = run_episode(
                    tmp=Path(tmp),
                    family=family,
                    rng=rng,
                    use_bandit=use_bandit,
                )
                successes += int(ok)
                attempts.append(n)
                overheads.append(ms)
                arms_per_family[family].append(arm_id)

    return {
        "label": label,
        "episodes": episodes,
        "success_rate": successes / max(episodes, 1),
        "mean_attempts": statistics.mean(attempts) if attempts else 0.0,
        "median_attempts": statistics.median(attempts) if attempts else 0.0,
        "mean_overhead_ms": statistics.mean(overheads) if overheads else 0.0,
        "p95_overhead_ms": _percentile(overheads, 95) if overheads else 0.0,
        "arms_per_family": arms_per_family,
        "db_path": str(last_db) if last_db else None,
    }


def _print_arm(label: str, info: dict[str, Any]) -> None:
    print(
        f"{label:8s}  episodes={info['episodes']:4d}  "
        f"success={info['success_rate'] * 100:5.1f}%  "
        f"mean_attempts={info['mean_attempts']:.2f}  "
        f"median={info['median_attempts']:.0f}  "
        f"latency_mean={info['mean_overhead_ms']:.1f}ms  p95={info['p95_overhead_ms']:.1f}ms"
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = round((pct / 100.0) * (len(s) - 1))
    return s[k]


if __name__ == "__main__":
    main()
    # JSON dump on the same line for CI scraping
    _ = json.dumps({"status": "done"})
