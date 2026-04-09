from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class BanditResult:
    total_reward: int
    counts: List[int]
    estimated_values: List[float]
    rewards_by_arm: List[int]
    chosen_arms: List[int] = field(default_factory=list)

    @property
    def best_arm(self) -> int:
        return max(range(len(self.estimated_values)), key=self.estimated_values.__getitem__)


class FourArmedBandit:
    def __init__(self, probabilities: Sequence[float], seed: int | None = None) -> None:
        if len(probabilities) != 4:
            raise ValueError("Exactly four arm probabilities are required.")
        if any(prob < 0.0 or prob > 1.0 for prob in probabilities):
            raise ValueError("Probabilities must be between 0.0 and 1.0.")
        self.probabilities = list(probabilities)
        self._rng = random.Random(seed)

    def pull(self, arm_index: int) -> int:
        if arm_index < 0 or arm_index >= len(self.probabilities):
            raise IndexError("Arm index out of range.")
        return int(self._rng.random() < self.probabilities[arm_index])


class UCB1Solver:
    def __init__(self, n_arms: int = 4, seed: int |None= None) -> None:
        if n_arms <= 0:
            raise ValueError("Number of arms must be positive.")
        self.n_arms = n_arms
        self.test_log_var = "a"
        self._rng = random.Random(seed)

    def argmax_random(self, values, rng, tol=1e-12):
        max_val = max(values)
        candidates = [i for i, v in enumerate(values) if abs(v - max_val) <= tol]
        return rng.choice(candidates)

    def select_arm(self, counts: Sequence[int], values: Sequence[float], total_pulls: int) -> int:
        # UCB1 forces an initial visit to every arm before using the confidence bound formula.
        for arm_index, count in enumerate(counts):
            if count == 0:
                return arm_index
            
        confidence_bounds = []
        for arm_index in range(self.n_arms):
            # Higher uncertainty yields a larger bonus, so rarely sampled arms stay attractive.
            bonus = math.sqrt((2.0 * math.log(total_pulls)) / counts[arm_index])
            confidence_bounds.append(values[arm_index] + bonus)

        return self.argmax_random(confidence_bounds, self._rng)

    def run(self, bandit: FourArmedBandit, rounds: int) -> BanditResult:
        if rounds <= 0:
            raise ValueError("Rounds must be positive.")

        counts = [0] * self.n_arms
        rewards_by_arm = [0] * self.n_arms
        values = [0.0] * self.n_arms
        chosen_arms: List[int] = []
        total_reward = 0

        for step in range(rounds):
            total_pulls = step + 1
            arm = self.select_arm(counts, values, total_pulls)
            reward = bandit.pull(arm)

            counts[arm] += 1
            rewards_by_arm[arm] += reward
            total_reward += reward
            chosen_arms.append(arm)

            values[arm] += (reward - values[arm]) / counts[arm]

        return BanditResult(
            total_reward=total_reward,
            counts=counts,
            estimated_values=values,
            rewards_by_arm=rewards_by_arm,
            chosen_arms=chosen_arms,
        )


class UCBDecisionEngine:
    def __init__(
        self,
        n_arms: int = 4,
        warm_start_each_arm: bool = True,
        warm_start_value: float = 0.5,
        seed: int | None = None
    ) -> None:
        if not 0.0 <= warm_start_value <= 1.0:
            raise ValueError("warm_start_value must be between 0.0 and 1.0.")
        self._solver = UCB1Solver(n_arms=n_arms, seed = seed)
        self._rng = random.Random(seed)
        self.n_arms = n_arms
        self.warm_start_each_arm = warm_start_each_arm
        self.warm_start_value = warm_start_value
        self.counts = [1] * n_arms if warm_start_each_arm else [0] * n_arms
        self.values = [warm_start_value + self._rng.uniform (-0.05, 0.05) for _ in range(n_arms)] if warm_start_each_arm else [0.0] * n_arms
        self.rewards_by_arm = [0] * n_arms
        self.total_reward = 0
        self.total_pulls = n_arms if warm_start_each_arm else 0
        self.last_selected_arm: int | None = None
        self.awaiting_reward = False
        self.selection_history: List[int] = []

    def choose_next_arm(self) -> int:
        # The robot should not choose again until the previous trial has reported its reward.
        if self.awaiting_reward:
            raise ValueError("Reward for the previous selection is still pending.")
        arm = self.ucb_arm()
        self.last_selected_arm = arm
        self.awaiting_reward = True
        self.selection_history.append(arm)
        return arm

    def ucb_arm(self) -> int:
        return self._solver.select_arm(
            counts=self.counts,
            values=self.values,
            total_pulls=max(1, self.total_pulls),
        )
    
    def argmax_random(self, values, rng, tol=1e-12):
        max_val = max(values)
        candidates = [i for i, v in enumerate(values) if abs(v - max_val) <= tol]
        return rng.choice(candidates)

    def greedy_arm(self) -> int:
        for arm_index, count in enumerate(self.counts):
            if count == 0:
                return arm_index
        return self.argmax_random(self.values, self._rng)

    def record_reward(self, reward: int, arm_index: int | None = None) -> int:
        # In the ROS flow, reward feedback normally applies to the most recently selected arm.
        arm = self.last_selected_arm if arm_index is None else arm_index
        if arm is None:
            raise ValueError("No arm has been selected yet.")
        if arm < 0 or arm >= self.n_arms:
            raise IndexError("Arm index out of range.")
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1.")

        self.total_pulls += 1
        self.counts[arm] += 1
        self.rewards_by_arm[arm] += reward
        self.total_reward += reward
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.awaiting_reward = False
        return arm


class ThompsonSamplingTracker:
    def __init__(self, n_arms: int = 4, forgetfulness: float = 0.05, seed: int | None = None) -> None:
        if not 0.0 <= forgetfulness < 1.0:
            raise ValueError("Forgetfulness must be in the range [0.0, 1.0).")
        self.n_arms = n_arms
        self.forgetfulness = forgetfulness
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms
        self._rng = random.Random(seed)

    def update(self, arm: int, reward: int) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError("Arm index out of range.")
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1.")

        # Forgetfulness pulls the posterior back toward its uninformed prior over time.
        for idx in range(self.n_arms):
            self.alpha[idx] = 1.0 + (self.alpha[idx] - 1.0) * (1.0 - self.forgetfulness)
            self.beta[idx] = 1.0 + (self.beta[idx] - 1.0) * (1.0 - self.forgetfulness)

        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def sample_scores(self) -> List[float]:
        return [self._rng.betavariate(self.alpha[idx], self.beta[idx]) for idx in range(self.n_arms)]

    def best_arm(self) -> int:
        samples = self.sample_scores()
        return max(range(self.n_arms), key=samples.__getitem__)


class SoftmaxPolicyTracker:
    def __init__(self, n_arms: int = 4, temperature: float = 0.2, learning_rate: float = 0.1) -> None:
        if temperature <= 0.0:
            raise ValueError("Temperature must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive.")
        self.n_arms = n_arms
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.preferences = [0.0] * n_arms

    def update(self, arm: int, reward: int) -> None:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError("Arm index out of range.")
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1.")

        # The rewarded arm's preference moves up or down relative to its current softmax probability.
        expected_reward = self.probabilities()[arm]
        self.preferences[arm] += self.learning_rate * (reward - expected_reward)

    def probabilities(self) -> List[float]:
        scaled = [pref / self.temperature for pref in self.preferences]
        max_scaled = max(scaled)
        exp_values = [math.exp(value - max_scaled) for value in scaled]
        total = sum(exp_values)
        return [value / total for value in exp_values]

    def best_arm(self) -> int:
        probabilities = self.probabilities()
        return max(range(self.n_arms), key=probabilities.__getitem__)


@dataclass
class PolicySnapshot:
    ucb_best_arm: int
    greedy_best_arm: int
    ucb_counts: List[int]
    ucb_values: List[float]
    thompson_best_arm: int
    thompson_alpha: List[float]
    thompson_beta: List[float]
    softmax_best_arm: int
    softmax_probabilities: List[float]


class PolicyComparisonEngine:
    def __init__(
        self,
        n_arms: int = 4,
        thompson_forgetfulness: float = 0.05,
        thompson_seed: int | None = None,
        softmax_temperature: float = 0.2,
        softmax_learning_rate: float = 0.1,
        action_policy: str = "ucb",
        warm_start_each_arm: bool = False,
        warm_start_value: float = 0.5,
    ) -> None:
        self.ucb = UCBDecisionEngine(
            n_arms=n_arms,
            warm_start_each_arm=warm_start_each_arm,
            warm_start_value=warm_start_value,
        )
        self.action_policy = action_policy.lower()
        if self.action_policy not in {"ucb", "greedy"}:
            raise ValueError("action_policy must be either 'ucb' or 'greedy'.")
        self.n_arms = n_arms
        self.thompson_forgetfulness = thompson_forgetfulness
        self.thompson_seed = thompson_seed
        self.softmax_temperature = softmax_temperature
        self.softmax_learning_rate = softmax_learning_rate
        self.warm_start_each_arm = warm_start_each_arm
        self.warm_start_value = warm_start_value
        self.experiment_active = False
        self.thompson = ThompsonSamplingTracker(
            n_arms=n_arms,
            forgetfulness=thompson_forgetfulness,
            seed=thompson_seed,
        )
        self.softmax = SoftmaxPolicyTracker(
            n_arms=n_arms,
            temperature=softmax_temperature,
            learning_rate=softmax_learning_rate,
        )

    def start_experiment(self) -> None:
        self.experiment_active = True

    def stop_experiment(self) -> None:
        self.experiment_active = False

    def reset_experiment(self) -> None:
        self.ucb = UCBDecisionEngine(
            n_arms=self.n_arms,
            warm_start_each_arm=self.warm_start_each_arm,
            warm_start_value=self.warm_start_value,
        )
        self.thompson = ThompsonSamplingTracker(
            n_arms=self.n_arms,
            forgetfulness=self.thompson_forgetfulness,
            seed=self.thompson_seed,
        )
        self.softmax = SoftmaxPolicyTracker(
            n_arms=self.n_arms,
            temperature=self.softmax_temperature,
            learning_rate=self.softmax_learning_rate,
        )
        self.experiment_active = False

    def choose_next_arm(self) -> int:
        if not self.experiment_active:
            raise ValueError("Experiment is not active. Start the experiment before requesting a bucket.")
        if self.ucb.awaiting_reward:
            raise ValueError("Reward for the previous selection is still pending.")

        if self.action_policy == "greedy":
            arm = self.ucb.greedy_arm()
            self.ucb.last_selected_arm = arm
            self.ucb.awaiting_reward = True
            self.ucb.selection_history.append(arm)
            return arm

        return self.ucb.choose_next_arm()

    def record_reward(self, reward: int, arm_index: int | None = None) -> int:
        if not self.experiment_active:
            raise ValueError("Experiment is not active. Start the experiment before reporting rewards.")
        # UCB remains the controller; the other policies observe the same trial only for comparison.
        arm = self.ucb.record_reward(reward=reward, arm_index=arm_index)
        self.thompson.update(arm=arm, reward=reward)
        self.softmax.update(arm=arm, reward=reward)
        return arm

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(
            ucb_best_arm=self.ucb.ucb_arm(),
            greedy_best_arm=self.ucb.greedy_arm(),
            ucb_counts=list(self.ucb.counts),
            ucb_values=list(self.ucb.values),
            thompson_best_arm=self.thompson.best_arm(),
            thompson_alpha=list(self.thompson.alpha),
            thompson_beta=list(self.thompson.beta),
            softmax_best_arm=self.softmax.best_arm(),
            softmax_probabilities=self.softmax.probabilities(),
        )
