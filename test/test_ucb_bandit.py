import unittest

from mab_ucb_bandit.ucb_bandit import (
    FourArmedBandit,
    PolicyComparisonEngine,
    SoftmaxPolicyTracker,
    ThompsonSamplingTracker,
    UCB1Solver,
    UCBDecisionEngine,
)


def _echo_like_message(step: int, arm: int, reward: int, counts: list[int], values: list[float]) -> str:
    return "\n".join(
        [
            f"step: {step}",
            f"selected_bucket: {arm}",
            f"reward: {reward}",
            f"counts: {counts}",
            f"estimated_values: {[round(value, 3) for value in values]}",
            "---",
        ]
    )


def _policy_echo_like_message(
    step: int,
    selected_arm: int,
    reward: int,
    snapshot,
) -> str:
    return "\n".join(
        [
            f"step: {step}",
            f"selected_bucket: {selected_arm}",
            f"reward: {reward}",
            f"ucb_best_next: {snapshot.ucb_best_arm}",
            f"greedy_best_next: {snapshot.greedy_best_arm}",
            f"thompson_best_next: {snapshot.thompson_best_arm}",
            f"softmax_best_next: {snapshot.softmax_best_arm}",
            f"counts: {snapshot.ucb_counts}",
            f"estimated_values: {[round(value, 3) for value in snapshot.ucb_values]}",
            f"softmax_probabilities: {[round(value, 3) for value in snapshot.softmax_probabilities]}",
            "---",
        ]
    )


class TestUCBBandit(unittest.TestCase):
    def test_solver_pulls_each_arm_once_before_exploitation(self) -> None:
        bandit = FourArmedBandit([0.1, 0.2, 0.3, 0.9], seed=1)
        solver = UCB1Solver(n_arms=4)

        result = solver.run(bandit, rounds=4)

        self.assertEqual(result.counts, [1, 1, 1, 1])
        self.assertEqual(result.chosen_arms, [0, 1, 2, 3])

    def test_solver_learns_best_arm_over_time(self) -> None:
        bandit = FourArmedBandit([0.05, 0.15, 0.35, 0.8], seed=7)
        solver = UCB1Solver(n_arms=4)

        result = solver.run(bandit, rounds=400)

        self.assertEqual(result.best_arm, 3)
        self.assertEqual(result.counts[3], max(result.counts))
        self.assertGreater(result.counts[3], 200)

    def test_invalid_probability_count_raises(self) -> None:
        with self.assertRaises(ValueError) as context:
            FourArmedBandit([0.1, 0.2, 0.3], seed=0)

        self.assertIn("four", str(context.exception).lower())

    def test_decision_engine_selects_each_arm_before_updating(self) -> None:
        engine = UCBDecisionEngine(n_arms=4)

        selections = []
        for reward in [0, 0, 0, 0]:
            selections.append(engine.choose_next_arm())
            engine.record_reward(reward)

        self.assertEqual(selections, [0, 1, 2, 3])

    def test_decision_engine_updates_ucb_state_from_rewards(self) -> None:
        engine = UCBDecisionEngine(n_arms=4)

        self.assertEqual(engine.choose_next_arm(), 0)
        self.assertEqual(engine.record_reward(1), 0)
        self.assertEqual(engine.choose_next_arm(), 1)
        self.assertEqual(engine.record_reward(0), 1)
        self.assertEqual(engine.choose_next_arm(), 2)
        self.assertEqual(engine.record_reward(1), 2)
        self.assertEqual(engine.choose_next_arm(), 3)
        self.assertEqual(engine.record_reward(1), 3)

        self.assertEqual(engine.counts, [1, 1, 1, 1])
        self.assertEqual(engine.rewards_by_arm, [1, 0, 1, 1])
        self.assertEqual(engine.total_reward, 3)

    def test_ucb_sequence_echo_trace(self) -> None:
        engine = UCBDecisionEngine(n_arms=4)
        rewards = [0, 1, 0, 1, 1, 1]
        selected_arms = []

        for step, reward in enumerate(rewards, start=1):
            arm = engine.choose_next_arm()
            engine.record_reward(reward)
            selected_arms.append(arm)
            print(_echo_like_message(step, arm, reward, engine.counts, engine.values))

        self.assertEqual(selected_arms[:4], [0, 1, 2, 3])
        self.assertEqual(engine.total_pulls, len(rewards))
        self.assertEqual(engine.selection_history, selected_arms)

    def test_policy_sequence_echo_trace(self) -> None:
        engine = PolicyComparisonEngine(
            n_arms=4,
            action_policy="ucb",
            thompson_seed=5,
        )
        rewards = [0, 1, 0, 1, 1, 1]

        engine.start_experiment()
        for step, reward in enumerate(rewards, start=1):
            arm = engine.choose_next_arm()
            engine.record_reward(reward)
            snapshot = engine.snapshot()
            print(_policy_echo_like_message(step, arm, reward, snapshot))

        self.assertEqual(engine.ucb.total_pulls, len(rewards))

    def test_decision_engine_rejects_reward_before_selection(self) -> None:
        engine = UCBDecisionEngine(n_arms=4)

        with self.assertRaises(ValueError):
            engine.record_reward(1)

    def test_decision_engine_requires_reward_before_next_selection(self) -> None:
        engine = UCBDecisionEngine(n_arms=4)

        self.assertEqual(engine.choose_next_arm(), 0)

        with self.assertRaises(ValueError):
            engine.choose_next_arm()

    def test_thompson_tracker_applies_forgetfulness(self) -> None:
        tracker = ThompsonSamplingTracker(n_arms=4, forgetfulness=0.5, seed=3)

        tracker.update(arm=0, reward=1)
        tracker.update(arm=0, reward=0)

        self.assertGreater(tracker.alpha[0], 1.0)
        self.assertGreater(tracker.beta[0], 1.0)
        self.assertLess(tracker.alpha[0], 3.0)

    def test_softmax_tracker_biases_toward_rewarded_arm(self) -> None:
        tracker = SoftmaxPolicyTracker(n_arms=4, temperature=0.5, learning_rate=0.4)

        tracker.update(arm=2, reward=1)
        probabilities = tracker.probabilities()

        self.assertEqual(tracker.best_arm(), 2)
        self.assertGreater(probabilities[2], 0.25)

    def test_policy_comparison_engine_updates_all_policies(self) -> None:
        engine = PolicyComparisonEngine(
            n_arms=4,
            thompson_forgetfulness=0.1,
            thompson_seed=5,
            softmax_temperature=0.5,
            softmax_learning_rate=0.3,
        )

        engine.start_experiment()
        self.assertEqual(engine.choose_next_arm(), 0)
        self.assertEqual(engine.record_reward(1), 0)
        snapshot = engine.snapshot()

        self.assertEqual(snapshot.ucb_counts, [1, 0, 0, 0])
        self.assertEqual(snapshot.ucb_best_arm, 1)
        self.assertEqual(snapshot.greedy_best_arm, 1)
        self.assertGreater(snapshot.thompson_alpha[0], 1.0)
        self.assertEqual(snapshot.softmax_best_arm, 0)
        self.assertGreater(snapshot.softmax_probabilities[0], 0.25)

    def test_policy_comparison_engine_can_use_greedy_action_policy(self) -> None:
        engine = PolicyComparisonEngine(
            n_arms=4,
            action_policy="greedy",
        )

        engine.start_experiment()
        self.assertEqual(engine.choose_next_arm(), 0)
        engine.record_reward(0)
        self.assertEqual(engine.choose_next_arm(), 1)
        engine.record_reward(1)
        self.assertEqual(engine.choose_next_arm(), 2)

    def test_policy_comparison_engine_rejects_unknown_action_policy(self) -> None:
        with self.assertRaises(ValueError):
            PolicyComparisonEngine(n_arms=4, action_policy="random")

    def test_policy_comparison_engine_requires_experiment_start(self) -> None:
        engine = PolicyComparisonEngine(n_arms=4)

        with self.assertRaises(ValueError):
            engine.choose_next_arm()

        with self.assertRaises(ValueError):
            engine.record_reward(1)

    def test_policy_comparison_engine_stop_blocks_future_actions(self) -> None:
        engine = PolicyComparisonEngine(n_arms=4)
        engine.start_experiment()
        self.assertEqual(engine.choose_next_arm(), 0)
        engine.record_reward(1)

        engine.stop_experiment()

        with self.assertRaises(ValueError):
            engine.choose_next_arm()

    def test_policy_comparison_engine_reset_clears_state(self) -> None:
        engine = PolicyComparisonEngine(n_arms=4)
        engine.start_experiment()
        self.assertEqual(engine.choose_next_arm(), 0)
        engine.record_reward(1)

        engine.reset_experiment()

        self.assertFalse(engine.experiment_active)
        self.assertEqual(engine.ucb.counts, [0, 0, 0, 0])
        self.assertEqual(engine.ucb.values, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(engine.thompson.alpha, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(engine.softmax.preferences, [0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
