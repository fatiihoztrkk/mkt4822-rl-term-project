"""Contextual bandit with SQLite-backed posterior persistence."""

from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from mecha_agent_cli.config.schema import LearningConfig
from mecha_agent_cli.learning.arm_registry import ARM_REGISTRY, Arm, get_arm
from mecha_agent_cli.learning.q_learning import QLearningPolicy, QValue
from mecha_agent_cli.learning.reward import reward_to_beta_update

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bandit_arms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  context_key TEXT NOT NULL,
  arm_id TEXT NOT NULL,
  alpha REAL NOT NULL DEFAULT 1.0,
  beta REAL NOT NULL DEFAULT 1.0,
  pulls INTEGER NOT NULL DEFAULT 0,
  cumulative_reward REAL NOT NULL DEFAULT 0.0,
  last_reward REAL NOT NULL DEFAULT 0.0,
  last_success INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  UNIQUE(context_key, arm_id)
);
CREATE INDEX IF NOT EXISTS idx_bandit_arms_ctx ON bandit_arms(context_key);
CREATE TABLE IF NOT EXISTS q_learning_values (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  context_key TEXT NOT NULL,
  arm_id TEXT NOT NULL,
  q_value REAL NOT NULL DEFAULT 0.0,
  updates INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  UNIQUE(context_key, arm_id)
);
CREATE INDEX IF NOT EXISTS idx_q_learning_values_ctx ON q_learning_values(context_key);
"""


@dataclass(frozen=True)
class ArmStat:
    """Posterior statistics for one context and model-profile arm."""

    context_key: str
    arm_id: str
    alpha: float
    beta: float
    pulls: int
    cumulative_reward: float
    last_reward: float
    last_success: bool

    @property
    def mean(self) -> float:
        """Return the Beta posterior mean."""
        return self.alpha / (self.alpha + self.beta)


class BanditStore:
    """Persist arm posteriors in SQLite or memory."""

    def __init__(self, db_path: Path | None) -> None:
        self.db_path = db_path
        self._rows: dict[tuple[str, str], ArmStat] = {}
        self._q_rows: dict[tuple[str, str], QValue] = {}
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        if self.db_path is None:
            raise RuntimeError("in-memory store has no SQLite connection")
        return sqlite3.connect(self.db_path)

    def fetch(self, context_key: str) -> dict[str, ArmStat]:
        """Return statistics for ``context_key`` keyed by arm identifier."""
        if self.db_path is None:
            return {arm_id: stat for (ctx, arm_id), stat in self._rows.items() if ctx == context_key}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT arm_id, alpha, beta, pulls, cumulative_reward, last_reward, last_success
                FROM bandit_arms WHERE context_key = ?
                """,
                (context_key,),
            ).fetchall()
        return {str(row[0]): _row_to_stat(context_key, row) for row in rows}

    def upsert(self, stat: ArmStat) -> None:
        """Insert or replace one arm posterior."""
        if self.db_path is None:
            self._rows[(stat.context_key, stat.arm_id)] = stat
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bandit_arms(
                  context_key, arm_id, alpha, beta, pulls, cumulative_reward,
                  last_reward, last_success, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(context_key, arm_id) DO UPDATE SET
                  alpha = excluded.alpha,
                  beta = excluded.beta,
                  pulls = excluded.pulls,
                  cumulative_reward = excluded.cumulative_reward,
                  last_reward = excluded.last_reward,
                  last_success = excluded.last_success,
                  updated_at = excluded.updated_at
                """,
                (
                    stat.context_key,
                    stat.arm_id,
                    stat.alpha,
                    stat.beta,
                    stat.pulls,
                    stat.cumulative_reward,
                    stat.last_reward,
                    int(stat.last_success),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def all_rows(self) -> list[ArmStat]:
        """Return every stored posterior."""
        if self.db_path is None:
            return list(self._rows.values())
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT context_key, arm_id, alpha, beta, pulls, cumulative_reward, last_reward, last_success
                FROM bandit_arms
                """
            ).fetchall()
        return [
            ArmStat(
                context_key=str(row[0]),
                arm_id=str(row[1]),
                alpha=float(row[2]),
                beta=float(row[3]),
                pulls=int(row[4]),
                cumulative_reward=float(row[5]),
                last_reward=float(row[6]),
                last_success=bool(row[7]),
            )
            for row in rows
        ]

    def fetch_q_values(self, context_key: str) -> dict[str, QValue]:
        """Return Q estimates for ``context_key`` keyed by arm identifier."""
        if self.db_path is None:
            return {arm_id: value for (ctx, arm_id), value in self._q_rows.items() if ctx == context_key}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT arm_id, q_value, updates
                FROM q_learning_values WHERE context_key = ?
                """,
                (context_key,),
            ).fetchall()
        return {
            str(row[0]): QValue(
                context_key=context_key,
                arm_id=str(row[0]),
                value=float(row[1]),
                updates=int(row[2]),
            )
            for row in rows
        }

    def upsert_q_value(self, value: QValue) -> None:
        """Insert or replace one tabular Q estimate."""
        if self.db_path is None:
            self._q_rows[(value.context_key, value.arm_id)] = value
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO q_learning_values(context_key, arm_id, q_value, updates, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(context_key, arm_id) DO UPDATE SET
                  q_value = excluded.q_value,
                  updates = excluded.updates,
                  updated_at = excluded.updated_at
                """,
                (
                    value.context_key,
                    value.arm_id,
                    value.value,
                    value.updates,
                    datetime.now(UTC).isoformat(),
                ),
            )


class ThompsonBandit:
    """Select profiles with contextual bandit or tabular Q-learning policies."""

    BASELINE_ARM_ID = "direct.baseline"

    def __init__(
        self,
        store: BanditStore,
        cfg: LearningConfig,
        *,
        arms: tuple[Arm, ...] = ARM_REGISTRY,
        rng: random.Random | None = None,
    ) -> None:
        self.store = store
        self.cfg = cfg
        self.arms = arms
        self.rng = rng or random.Random()
        self.q_learning = QLearningPolicy(epsilon=cfg.epsilon, rng=self.rng)

    def select(self, context_key: str) -> Arm:
        """Select an arm for ``context_key`` according to configured mode."""
        if self.cfg.mode == "off":
            return get_arm(self.BASELINE_ARM_ID)
        stats = self.store.fetch(context_key)
        under_pulled = [
            arm
            for arm in self.arms
            if stats.get(arm.arm_id, self._prior_stat(context_key, arm)).pulls < self.cfg.min_pulls_before_exploit
        ]
        if under_pulled:
            # During exploration, sample from arm priors so high-quality arms
            # are tried before low-quality ones.
            return max(
                under_pulled,
                key=lambda arm: self.rng.betavariate(arm.prior_alpha, arm.prior_beta),
            )
        if self.cfg.mode == "q_learning":
            arm_id = self.q_learning.select([arm.arm_id for arm in self.arms], self.store.fetch_q_values(context_key))
            return next(arm for arm in self.arms if arm.arm_id == arm_id)
        if self.cfg.mode == "ucb1":
            return max(self.arms, key=lambda arm: self._ucb1_score(stats, context_key, arm))
        if self.cfg.mode == "greedy":
            if self.rng.random() < self.cfg.epsilon:
                return self.rng.choice(self.arms)
            return max(self.arms, key=lambda arm: stats.get(arm.arm_id, self._prior_stat(context_key, arm)).mean)
        if self.cfg.mode == "softmax":
            return self._softmax_select(stats, context_key)
        # Thompson (default): pure Bayesian — prior for unseen arms, posterior for observed.
        return max(
            self.arms,
            key=lambda arm: self.rng.betavariate(
                stats.get(arm.arm_id, self._prior_stat(context_key, arm)).alpha,
                stats.get(arm.arm_id, self._prior_stat(context_key, arm)).beta,
            ),
        )

    # Arms that work well as repair arms after a first-attempt failure.
    # direct.cold is excluded: excellent at a0 but poor at repair (0/3 repair success).
    _REPAIR_BOOST_ARMS = frozenset({"direct.warm", "direct.baseline", "direct.cool", "direct.high_repeat"})
    # Arms to penalise in repair context — observed consistently in failure chains.
    # direct.tight_topk and direct.broad_topk added after appearing in all 3 failures
    # of the 8-repeat benchmark run.
    _REPAIR_PENALTY_ARMS = frozenset(
        {"direct.fixed_seed", "direct.hot", "direct.no_think", "direct.cold", "direct.tight_topk", "direct.broad_topk"}
    )

    def _prior_stat(self, context_key: str, arm: Arm) -> ArmStat:
        """Return a context-adapted informative prior when no stored data exists.

        After a runtime or import failure, repair-proven arms (warm, baseline,
        cool, high_repeat) receive a strong alpha boost. Arms observed in failure
        chains receive a beta penalty. Arms not in either set get a smaller beta
        nudge to widen the gap toward the boost set.
        """
        alpha, beta = arm.prior_alpha, arm.prior_beta
        if "|pf:runtime|" in context_key or "|pf:import|" in context_key:
            if arm.arm_id in self._REPAIR_BOOST_ARMS:
                alpha += 1.0
            elif arm.arm_id in self._REPAIR_PENALTY_ARMS:
                beta += 1.2
            else:
                beta += 0.4
        return ArmStat(context_key, arm.arm_id, alpha, beta, 0, 0.0, 0.0, False)

    def _ucb1_score(self, stats: dict[str, ArmStat], context_key: str, arm: Arm) -> float:
        stat = stats.get(arm.arm_id, self._prior_stat(context_key, arm))
        if stat.pulls == 0:
            return math.inf
        total_pulls = max(1, sum(item.pulls for item in stats.values()))
        return stat.mean + self.cfg.ucb_c * math.sqrt(math.log(total_pulls) / stat.pulls)

    def _softmax_select(self, stats: dict[str, ArmStat], context_key: str) -> Arm:
        """Boltzmann/softmax arm selection using ucb_c as inverse temperature."""
        temperature = max(0.01, self.cfg.ucb_c)
        means = [stats.get(arm.arm_id, self._prior_stat(context_key, arm)).mean for arm in self.arms]
        scaled = [m / temperature for m in means]
        max_s = max(scaled)
        exp_s = [math.exp(s - max_s) for s in scaled]
        total = sum(exp_s)
        weights = [e / total for e in exp_s]
        return self.rng.choices(list(self.arms), weights=weights, k=1)[0]

    def update(self, *, context_key: str, arm_id: str, reward: float, success: bool) -> ArmStat:
        """Apply one reward observation to an arm posterior."""
        prior = self.store.fetch(context_key).get(arm_id, _zero_stat(context_key, arm_id))
        pseudo_success = reward_to_beta_update(reward)
        stat = ArmStat(
            context_key=context_key,
            arm_id=arm_id,
            alpha=1.0 + (prior.alpha - 1.0) * self.cfg.decay_factor + pseudo_success,
            beta=1.0 + (prior.beta - 1.0) * self.cfg.decay_factor + (1.0 - pseudo_success),
            pulls=prior.pulls + 1,
            cumulative_reward=prior.cumulative_reward + reward,
            last_reward=reward,
            last_success=success,
        )
        self.store.upsert(stat)
        q_values = self.store.fetch_q_values(context_key)
        q_value = self.q_learning.update(
            context_key=context_key,
            arm_id=arm_id,
            reward=reward,
            prior=q_values.get(arm_id),
        )
        self.store.upsert_q_value(q_value)
        return stat


def _row_to_stat(context_key: str, row: tuple[object, ...]) -> ArmStat:
    arm_id, alpha, beta, pulls, cumulative_reward, last_reward, last_success = cast(
        tuple[str, float, float, int, float, float, int],
        row,
    )
    return ArmStat(
        context_key=context_key,
        arm_id=arm_id,
        alpha=alpha,
        beta=beta,
        pulls=pulls,
        cumulative_reward=cumulative_reward,
        last_reward=last_reward,
        last_success=bool(last_success),
    )


def _zero_stat(context_key: str, arm_id: str) -> ArmStat:
    return ArmStat(context_key, arm_id, 1.0, 1.0, 0, 0.0, 0.0, False)


__all__ = ["ArmStat", "BanditStore", "ThompsonBandit"]
