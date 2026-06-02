"""Actor-Critic Reinforcement Learning — Python standard library only.

Implements a full Actor-Critic training loop on a discrete grid environment
where an agent learns to reach a goal state. Produces training visualisation,
a separate evaluation stage, and writes generation_report.csv.
"""

import csv
import random
import math

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SimpleEnvironment:
    """1-D grid: agent moves left/right, reward +10 at goal, -1 per step."""

    def __init__(self, goal: int = 5, seed: int | None = 42) -> None:
        self.goal = goal
        self.state = 0
        self._rng = random.Random(seed)

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.goal, self.state + 1)
        reward = 10.0 if self.state == self.goal else -1.0
        done = self.state == self.goal
        return self.state, reward, done


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class SimpleActor:
    """Tabular stochastic policy with decaying exploration."""

    def __init__(self, n_states: int, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random(42)
        # Start with uniform policy (0.5 = random)
        self.policy = {s: 0.5 for s in range(n_states + 1)}

    def select_action(self, state: int, epsilon: float = 0.0) -> int:
        if self._rng.random() < epsilon:
            return self._rng.randint(0, 1)
        return 1 if self._rng.random() < self.policy[state] else 0

    def update(self, state: int, advantage: float, lr: float = 0.02) -> None:
        if advantage > 0:
            self.policy[state] = min(0.95, self.policy[state] + lr * advantage)
        else:
            self.policy[state] = max(0.05, self.policy[state] + lr * advantage)


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

class SimpleCritic:
    """Tabular value function V(s) updated via TD(0)."""

    def __init__(self, lr: float = 0.1) -> None:
        self.values: dict[int, float] = {}
        self.lr = lr

    def value(self, state: int) -> float:
        return self.values.get(state, 0.0)

    def update(self, state: int, reward: float, next_state: int,
               done: bool, gamma: float = 0.95) -> float:
        v_next = 0.0 if done else self.value(next_state)
        td_target = reward + gamma * v_next
        td_error = td_target - self.value(state)
        self.values[state] = self.value(state) + self.lr * td_error
        return td_error


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(num_episodes: int = 100, max_steps: int = 100,
          seed: int = 42) -> tuple[list[dict], "SimpleActor"]:
    rng = random.Random(seed)
    env = SimpleEnvironment(goal=5, seed=seed)
    actor = SimpleActor(n_states=5, rng=rng)
    critic = SimpleCritic(lr=0.1)
    records: list[dict] = []

    for ep in range(1, num_episodes + 1):
        # Decaying exploration: high early, low later
        epsilon = max(0.05, 0.5 * math.exp(-ep / 30))
        state = env.reset()
        total_reward, steps = 0.0, 0
        done = False

        while not done and steps < max_steps:
            action = actor.select_action(state, epsilon=epsilon)
            next_state, reward, done = env.step(action)
            advantage = critic.update(state, reward, next_state, done)
            actor.update(state, advantage)
            total_reward += reward
            state = next_state
            steps += 1

        records.append({"episode": ep, "steps": steps,
                        "total_reward": round(total_reward, 2)})
    return records, actor


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(actor: "SimpleActor", num_trials: int = 20, seed: int = 99) -> dict:
    """Evaluate the trained actor greedily (epsilon=0)."""
    rng = random.Random(seed)
    env = SimpleEnvironment(goal=5, seed=seed)
    total, successes = 0.0, 0
    for _ in range(num_trials):
        state = env.reset()
        ep_reward, steps, done = 0.0, 0, False
        while not done and steps < 50:
            action = actor.select_action(state, epsilon=0.0)
            state, reward, done = env.step(action)
            ep_reward += reward
            steps += 1
        total += ep_reward
        if done:
            successes += 1
    return {"mean_reward": round(total / num_trials, 2),
            "success_rate": round(successes / num_trials, 2)}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def print_progress(records: list[dict], window: int = 10) -> None:
    print("\nTraining Progress (10-episode moving average):")
    print(f"{'Episode':>8} | {'Avg Reward':>10} | {'Avg Steps':>9} | Learning Curve")
    print("-" * 60)
    for i in range(0, len(records), window):
        chunk = records[i: i + window]
        avg_r = sum(r["total_reward"] for r in chunk) / len(chunk)
        avg_s = sum(r["steps"] for r in chunk) / len(chunk)
        bar_len = max(0, min(30, int((avg_r + 50) / 3)))
        bar = "█" * bar_len
        ep = chunk[-1]["episode"]
        print(f"{ep:>8} | {avg_r:>10.1f} | {avg_s:>9.1f} | {bar}")


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], eval_result: dict,
              path: str = "generation_report.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "steps", "total_reward"])
        writer.writeheader()
        writer.writerows(records)
        writer.writerow({"episode": "evaluation",
                         "steps": eval_result["success_rate"],
                         "total_reward": eval_result["mean_reward"]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Actor-Critic Reinforcement Learning ===")
    records, trained_actor = train(num_episodes=100, max_steps=100, seed=42)
    print_progress(records)

    print("\n=== Evaluation Stage (trained actor, epsilon=0) ===")
    eval_result = evaluate(trained_actor, num_trials=20, seed=99)
    print(f"Mean reward : {eval_result['mean_reward']:.2f}")
    print(f"Success rate: {eval_result['success_rate']:.0%}")

    write_csv(records, eval_result)
    print("\ngeneration_report.csv written to current directory.")
