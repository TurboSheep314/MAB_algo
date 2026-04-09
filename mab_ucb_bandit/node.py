from __future__ import annotations

import rclpy
import random
from rclpy.node import Node
from std_msgs.msg import Empty, Int32

from .ucb_bandit import PolicyComparisonEngine


class UCBBanditNode(Node):
    def __init__(self) -> None:
        super().__init__("ucb_bandit_node")
        # Topic names and tracker settings stay configurable so this node can drop into a larger ROS graph.
        self.declare_parameter("request_topic", "move_to_next_ucb_bucket")
        self.declare_parameter("reward_topic", "bandit_reward")
        self.declare_parameter("bucket_topic", "ucb_selected_bucket")
        self.declare_parameter("start_topic", "start_experiment")
        self.declare_parameter("stop_topic", "stop_experiment")
        self.declare_parameter("reset_topic", "reset_experiment")
        self.declare_parameter("action_policy", "ucb")
        self.declare_parameter("warm_start_each_arm", True)
        self.declare_parameter("warm_start_value", 0.5)
        self.declare_parameter("thompson_forgetfulness", 0.05)
        self.declare_parameter("thompson_seed", 11)
        self.declare_parameter("softmax_temperature", 0.2)
        self.declare_parameter("softmax_learning_rate", 0.1)

        self.request_topic = str(self.get_parameter("request_topic").value)
        self.reward_topic = str(self.get_parameter("reward_topic").value)
        self.bucket_topic = str(self.get_parameter("bucket_topic").value)
        self.start_topic = str(self.get_parameter("start_topic").value)
        self.stop_topic = str(self.get_parameter("stop_topic").value)
        self.reset_topic = str(self.get_parameter("reset_topic").value)
        self.action_policy = str(self.get_parameter("action_policy").value)
        self.warm_start_each_arm = bool(self.get_parameter("warm_start_each_arm").value)
        self.warm_start_value = float(self.get_parameter("warm_start_value").value)
        self.thompson_forgetfulness = float(self.get_parameter("thompson_forgetfulness").value)
        self.thompson_seed = int(self.get_parameter("thompson_seed").value)
        self.softmax_temperature = float(self.get_parameter("softmax_temperature").value)
        self.softmax_learning_rate = float(self.get_parameter("softmax_learning_rate").value)

        self.create_timer(4, self._timer_func)

        # UCB selects the bucket to publish; Thompson and softmax are tracked in parallel for logging only.
        self.engine = PolicyComparisonEngine(
            n_arms=4,
            thompson_forgetfulness=self.thompson_forgetfulness,
            thompson_seed=self.thompson_seed,
            softmax_temperature=self.softmax_temperature,
            softmax_learning_rate=self.softmax_learning_rate,
            action_policy=self.action_policy,
            warm_start_each_arm=self.warm_start_each_arm,
            warm_start_value=self.warm_start_value,
        )
        
        self.bucket_publisher = self.create_publisher(Int32, self.bucket_topic, 10)
        self.request_subscriber = self.create_subscription(
            Empty,
            self.request_topic,
            self._handle_selection_request,
            10,
        )
        self.reward_subscriber = self.create_subscription(
            Int32,
            self.reward_topic,
            self._handle_reward,
            10,
        )
        self.start_subscriber = self.create_subscription(
            Empty,
            self.start_topic,
            self._handle_start_experiment,
            10,
        )
        self.stop_subscriber = self.create_subscription(
            Empty,
            self.stop_topic,
            self._handle_stop_experiment,
            10,
        )
        self.reset_subscriber = self.create_subscription(
            Empty,
            self.reset_topic,
            self._handle_reset_experiment,
            10,
        )

        self.get_logger().info(
            f"Listening for start on '{self.start_topic}', stop on '{self.stop_topic}', reset on '{self.reset_topic}', "
            f"move requests on '{self.request_topic}', rewards on '{self.reward_topic}', "
            f"publishing bucket choices on '{self.bucket_topic}', using '{self.action_policy}' as the action policy, "
            f"and warm_start_each_arm={self.warm_start_each_arm}, "
            f"and values = {[round(v,3) for v in self.engine.ucb.values]}, "
            f"and total pulls = {self.engine.ucb.total_pulls}. "  
        )

    def _timer_func(self) -> None:
        self.get_logger().info(
            f"Listening for start on '{self.start_topic}', stop on '{self.stop_topic}', reset on '{self.reset_topic}', "
            f"move requests on '{self.request_topic}', rewards on '{self.reward_topic}', "
            f"publishing bucket choices on '{self.bucket_topic}', using '{self.action_policy}' as the action policy, "
            f"and warm_start_each_arm={self.warm_start_each_arm}, "
            f"and values = {[round(v,3) for v in self.engine.ucb.values]}, "
            f"and total pulls = {self.engine.ucb.total_pulls}. "  
        )

    def _handle_start_experiment(self, _: Empty) -> None:
        self.engine.start_experiment()
        self.get_logger().info("Experiment started. Bucket requests will now be accepted.")

    def _handle_stop_experiment(self, _: Empty) -> None:
        self.engine.stop_experiment()
        self.get_logger().info("Experiment stopped. Bucket requests and reward updates are now ignored.")

    def _handle_reset_experiment(self, _: Empty) -> None:
        self.engine.reset_experiment()
        self.get_logger().info("Experiment reset. All learned policy state has been cleared.")

    def _handle_selection_request(self, _: Empty) -> None:
        try:
            arm = self.engine.choose_next_arm()
        except ValueError as exc:
            self.get_logger().error(f"Rejected selection request: {exc}")
            return
        # The published bucket index is the contract this package exposes to the motion controller.
        message = Int32()
        message.data = arm
        self.bucket_publisher.publish(message)
        self.get_logger().error(f"TO BE WARMED?: {self.engine.warm_start_each_arm}")
        self.get_logger().error(f"LOG: {self.engine.ucb._solver.test_log_var}")
        self.get_logger().info(f"Selected bucket {arm} and published command to motion package.")

    def _handle_reward(self, msg: Int32) -> None:
        try:
            # Reward messages close the current trial and let every tracker update from the observed outcome.
            arm = self.engine.record_reward(msg.data)
        except (ValueError, IndexError) as exc:
            self.get_logger().error(f"Rejected reward update: {exc}")
            return

        self.get_logger().info(
            f"Recorded reward {msg.data} for bucket {arm}. "
            f"UCB counts={self.engine.ucb.counts}, "
            f"UCB values={[round(value, 3) for value in self.engine.ucb.values]}"
        )
        snapshot = self.engine.snapshot()
        self.get_logger().info(
            f"Comparison policies: UCB best={snapshot.ucb_best_arm}, "
            f"greedy best={snapshot.greedy_best_arm}, "
            f"Thompson best={snapshot.thompson_best_arm}, "
            f"alpha={[round(value, 3) for value in snapshot.thompson_alpha]}, "
            f"beta={[round(value, 3) for value in snapshot.thompson_beta]}, "
            f"Softmax best={snapshot.softmax_best_arm}, "
            f"probs={[round(value, 3) for value in snapshot.softmax_probabilities]}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UCBBanditNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
