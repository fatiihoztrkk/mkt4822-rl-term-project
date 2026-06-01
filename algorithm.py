"""Small, reproducible Actor-Critic reinforcement-learning application."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

Policy = list[list[float]]


def _softmax(logits: list[float]) -> list[float]:
    """Return stable softmax probabilities for a short list of logits."""
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def _sample_action(probabilities: list[float], rng: random.Random) -> int:
    """Draw an action index from a probability distribution."""
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def _transition(state: int, action: int, goal: int) -> tuple[int, float, bool]:
    """Move in a tiny line world where action 0 is left and action 1 is right."""
    next_state = max(0, min(goal, state + (-1 if action == 0 else 1)))
    done = next_state == goal
    reward = 10.0 if done else -0.05
    return next_state, reward, done


def train_actor_critic(episodes: int, max_steps: int, seed: int) -> dict[str, object]:
    """Train a tabular softmax actor and state-value critic in a line world."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    rng = random.Random(seed)
    goal = 5
    actor: Policy = [[0.0, 0.0] for _ in range(goal + 1)]
    critic = [0.0 for _ in range(goal + 1)]
    history: list[float] = []
    successes = 0
    gamma = 0.95
    actor_rate = 0.10
    critic_rate = 0.20

    for _ in range(episodes):
        state = 0
        episode_reward = 0.0
        for _ in range(max_steps):
            probabilities = _softmax(actor[state])
            action = _sample_action(probabilities, rng)
            next_state, reward, done = _transition(state, action, goal)
            target = reward if done else reward + gamma * critic[next_state]
            advantage = target - critic[state]
            critic[state] += critic_rate * advantage
            for candidate in range(2):
                gradient = (1.0 if candidate == action else 0.0) - probabilities[candidate]
                actor[state][candidate] += actor_rate * advantage * gradient
            episode_reward += reward
            state = next_state
            if done:
                successes += 1
                break
        history.append(episode_reward)

    return {
        "policy": actor,
        "values": critic,
        "history": history,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_reward": sum(history) / episodes,
    }


def evaluate_policy(policy: Policy, episodes: int, max_steps: int, seed: int) -> dict[str, float]:
    """Evaluate a learned policy greedily with deterministic tie-breaking."""
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    random.Random(seed)  # Keep the evaluation API explicitly seed-controlled.
    goal = 5
    successes = 0
    rewards: list[float] = []
    steps_taken: list[int] = []
    for _ in range(episodes):
        state = 0
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            action = max(range(2), key=lambda index: (policy[state][index], index))
            state, reward, done = _transition(state, action, goal)
            total_reward += reward
            if done:
                successes += 1
                steps_taken.append(step)
                break
        else:
            steps_taken.append(max_steps)
        rewards.append(total_reward)
    return {
        "episodes": float(episodes),
        "success_rate": successes / episodes,
        "mean_reward": sum(rewards) / episodes,
        "mean_steps": sum(steps_taken) / episodes,
    }


def render_visualization(training_history: list[float], policy: Policy) -> str:
    """Write a text visualization of learning progress and policy direction."""
    bucket_size = max(1, len(training_history) // 10)
    lines = ["Actor-Critic training progress", ""]
    for start in range(0, len(training_history), bucket_size):
        bucket = training_history[start : start + bucket_size]
        mean_reward = sum(bucket) / len(bucket)
        bar = "#" * max(1, int((mean_reward + 1.0) * 2))
        lines.append(f"Episodes {start + 1:03d}-{start + len(bucket):03d}: {mean_reward:6.2f} {bar}")
    lines.extend(["", "Greedy policy by state:"])
    for state, logits in enumerate(policy[:-1]):
        direction = "RIGHT" if logits[1] >= logits[0] else "LEFT"
        lines.append(f"State {state}: {direction} logits={logits}")
    output = "\n".join(lines) + "\n"
    Path("training_progress.txt").write_text(output, encoding="utf-8")
    return output


def write_generation_report(
    training_summary: dict[str, object],
    evaluation_summary: dict[str, float],
    path: str,
) -> None:
    """Write compact, reproducible training and evaluation metrics to CSV."""
    rows = [
        ("training", "episodes", training_summary["episodes"]),
        ("training", "success_rate", f"{float(training_summary['success_rate']):.4f}"),
        ("training", "mean_reward", f"{float(training_summary['mean_reward']):.4f}"),
        ("evaluation", "episodes", int(evaluation_summary["episodes"])),
        ("evaluation", "success_rate", f"{evaluation_summary['success_rate']:.4f}"),
        ("evaluation", "mean_reward", f"{evaluation_summary['mean_reward']:.4f}"),
        ("evaluation", "mean_steps", f"{evaluation_summary['mean_steps']:.4f}"),
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "metric", "value"])
        writer.writerows(rows)


def main() -> None:
    """Train, evaluate, visualize, and report the Actor-Critic experiment."""
    training = train_actor_critic(episodes=180, max_steps=20, seed=7)
    policy = training["policy"]
    history = training["history"]
    if not isinstance(policy, list) or not isinstance(history, list):
        raise TypeError("training summary contains invalid policy or history")
    evaluation = evaluate_policy(policy, episodes=40, max_steps=20, seed=17)
    visualization = render_visualization(history, policy)
    write_generation_report(training, evaluation, "generation_report.csv")
    print(visualization)
    print(f"Training success rate: {float(training['success_rate']):.2%}")
    print(f"Evaluation success rate: {evaluation['success_rate']:.2%}")


if __name__ == "__main__":
    main()
