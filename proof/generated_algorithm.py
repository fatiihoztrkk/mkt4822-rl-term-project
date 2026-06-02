import random
import csv
import os
from typing import List, Dict, Tuple, Optional

class SimpleEnvironment:
    def __init__(self, num_states: int = 3):
        self.num_states = num_states
        self.start_state = 0
        self.goal_state = num_states - 1

    def get_actions(self, state: int) -> List[int]:
        return [0, 1]

    def step(self, state: int, action: int) -> Tuple[int, int]:
        if action == 0:
            next_state = state
        else:
            next_state = state + 1
        reward = 1 if next_state == self.goal_state else -1
        return next_state, reward

def train_actor_critic(
    env: SimpleEnvironment,
    num_episodes: int,
    max_steps_per_episode: int,
    learning_rate: float = 0.1
) -> Tuple[Dict[int, int], Dict[int, float], List[Dict[str, float]]]:
    """
    Train the Actor-Critic model using a simple tabular approach.

    Args:
        env: The environment instance.
        num_episodes: Number of training episodes.
        max_steps_per_episode: Maximum steps per episode.
        learning_rate: Learning rate for value function updates.

    Returns:
        Tuple of policy, value function, and training data.
    """
    policy = {state: 0 for state in range(env.num_states)}
    value = {state: 0.0 for state in range(env.num_states)}
    training_data = []

    for episode in range(num_episodes):
        state = env.start_state
        total_reward = 0
        steps = 0
        for step in range(max_steps_per_episode):
            action = policy[state]
            next_state, reward = env.step(state, action)
            total_reward += reward
            steps += 1

            # Update value function using TD(0)
            delta = reward + value[next_state] - value[state]
            value[state] += learning_rate * delta

            if next_state == env.goal_state:
                break

            state = next_state

        # Update policy after each episode
        for state in range(env.num_states):
            possible_actions = env.get_actions(state)
            best_action = max(possible_actions, key=lambda a: value[a])
            policy[state] = best_action

        training_data.append({
            'episode': episode,
            'steps': steps,
            'total_reward': total_reward
        })

    return policy, value, training_data

def evaluate_policy(
    env: SimpleEnvironment,
    policy: Dict[int, int],
    num_evaluation_episodes: int = 3
) -> List[Dict[str, float]]:
    """
    Evaluate the trained policy.

    Args:
        env: Environment instance.
        policy: Policy dictionary (state -> action).
        num_evaluation_episodes: Number of evaluation episodes.

    Returns:
        List of evaluation results per episode.
    """
    results = []
    for _ in range(num_evaluation_episodes):
        state = env.start_state
        total_reward = 0
        steps = 0
        while state != env.goal_state:
            action = policy[state]
            next_state, reward = env.step(state, action)
            total_reward += reward
            steps += 1
            state = next_state
        results.append({
            'episode': len(results) + 1,
            'steps': steps,
            'total_reward': total_reward
        })
    return results

def generate_csv_report(
    results: List[Dict[str, float]],
    output_path: str = 'generation_report.csv'
):
    """
    Generate a CSV report of training and evaluation results.

    Args:
        results: List of results dictionaries.
        output_path: Path to the CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'steps', 'total_reward'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def visualize_progress(
    training_data: List[Dict[str, float]]
):
    """
    Print a textual visualization of the training progress.

    Args:
        training_data: List of training results.
    """
    print("\nTraining Progress:")
    for i, data in enumerate(training_data):
        print(f"Episode {i+1}: {data['steps']} steps, Total Reward: {data['total_reward']:.2f}")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    env = SimpleEnvironment(num_states=3)
    num_episodes = 5
    max_steps_per_episode = 10
    learning_rate = 0.1

    # Train
    policy, value, training_data = train_actor_critic(
        env,
        num_episodes,
        max_steps_per_episode,
        learning_rate
    )
    print(f"Training completed. Policy: {policy}")
    print(f"Value function: {value}")

    # Evaluate
    evaluation_results = evaluate_policy(env, policy)

    # Generate CSV report
    generate_csv_report(training_data + evaluation_results)

    # Visualize progress
    visualize_progress(training_data)
