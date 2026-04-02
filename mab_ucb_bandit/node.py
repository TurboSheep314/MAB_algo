from __future__ import annotations

import rclpy
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
        self.declare_parameter("thompson_forgetfulness", 0.05)
        self.declare_parameter("thompson_seed", 11)
        self.declare_parameter("softmax_temperature", 0.2)
        self.declare_parameter("softmax_learning_rate", 0.1)
        self.declare_parameter("arm_probabilities", [0.9, 0.7, 0.5, 0.3])

        # this is later for use in a launch file
        #probabilities = [float(value) for value in self.get_parameter("arm_probabilities").value]
        #if len(probabilities) != 4:
        #    raise ValueError("arm_probabilities must contain exactly 4 values.")
        #

        request_topic = str(self.get_parameter("request_topic").value)
        reward_topic = str(self.get_parameter("reward_topic").value)
        bucket_topic = str(self.get_parameter("bucket_topic").value)
        start_topic = str(self.get_parameter("start_topic").value)
        stop_topic = str(self.get_parameter("stop_topic").value)
        reset_topic = str(self.get_parameter("reset_topic").value)
        action_policy = str(self.get_parameter("action_policy").value)
        thompson_forgetfulness = float(self.get_parameter("thompson_forgetfulness").value)
        thompson_seed = int(self.get_parameter("thompson_seed").value)
        softmax_temperature = float(self.get_parameter("softmax_temperature").value)
        softmax_learning_rate = float(self.get_parameter("softmax_learning_rate").value)

        # UCB selects the bucket to publish; Thompson and softmax are tracked in parallel for logging only.
        self.engine = PolicyComparisonEngine(
            n_arms=4,
            thompson_forgetfulness=thompson_forgetfulness,
            thompson_seed=thompson_seed,
            softmax_temperature=softmax_temperature,
            softmax_learning_rate=softmax_learning_rate,
            action_policy=action_policy,
        )
        self.bucket_publisher = self.create_publisher(Int32, bucket_topic, 10)
        self.request_subscriber = self.create_subscription(
            Empty,
            request_topic,
            self._handle_selection_request,
            10,
        )
        self.reward_subscriber = self.create_subscription(
            Int32,
            reward_topic,
            self._handle_reward,
            10,
        )
        self.start_subscriber = self.create_subscription(
            Empty,
            start_topic,
            self._handle_start_experiment,
            10,
        )
        self.stop_subscriber = self.create_subscription(
            Empty,
            stop_topic,
            self._handle_stop_experiment,
            10,
        )
        self.reset_subscriber = self.create_subscription(
            Empty,
            reset_topic,
            self._handle_reset_experiment,
            10,
        )

        self.get_logger().info(
            f"Listening for start on '{start_topic}', stop on '{stop_topic}', reset on '{reset_topic}', "
            f"move requests on '{request_topic}', rewards on '{reward_topic}', "
            f"publishing bucket choices on '{bucket_topic}', and using '{action_policy}' as the action policy."
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
