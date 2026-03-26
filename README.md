# MAB UCB Bandit

This repository contains a ROS 2 Python package for a 4-armed bandit solved with the UCB1 algorithm, while also tracking Thompson sampling with forgetfulness and a softmax policy for comparison.

## Package contents

- `mab_ucb_bandit/ucb_bandit.py`: core UCB solver, simulation helpers, topic-driven decision engine, and comparison trackers
- `mab_ucb_bandit/node.py`: ROS 2 node that listens for button-trigger topics, publishes the chosen bucket, and logs comparison-policy state
- `test/test_ucb_bandit.py`: unit tests for the solver, decision state, and comparison trackers

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/dev.txt
python -m pip install -e .
```

## Run tests

```bash
python -m unittest discover -s test
```

## ROS 2 usage

Place the MAB_algo package into the ROS 2 workspace `src/` directory and build with:

```bash
colcon build --packages-select mab_ucb_bandit
source install/setup.bash
ros2 run mab_ucb_bandit ucb_bandit_node
```

YOu have two options for implementation, right now all of the topics in this package are set to be run as ros parameters so you can set them in the launch file like this:

```bash
parameters=[{
    "start_topic": "/gui/start",
    "stop_topic": "/gui/stop",
    "reset_topic": "/gui/reset",
    "request_topic": "/gui/next_arm",
    "reward_topic": "/experiment/reward",
    "bucket_topic": "/robot/target_bucket",
    "action_policy": "ucb",
}
```


OR you can replace the paramaters with your esiting topics form the work flow e.g.

```bash
    self.request_subscriber = self.create_subscription(
    Empty,
    request_topic,
    self._handle_selection_request,
    10,
)
```


Default topic flow:

- Publish `std_msgs/Empty` on `start_experiment` to arm the package and allow bucket selection.
- Publish `std_msgs/Empty` on `stop_experiment` to pause the experiment and ignore future selection/reward messages.
- Publish `std_msgs/Empty` on `reset_experiment` to clear all learned UCB, Thompson, and softmax state.
- Publish `std_msgs/Empty` on `move_to_next_ucb_bucket` when the button is pressed to request the next bucket choice.
- This node publishes `std_msgs/Int32` on `ucb_selected_bucket` with the chosen bucket index `0..3`.
- After the robot finishes the trial, publish `std_msgs/Int32` on `bandit_reward` with `0` or `1` to update the UCB estimate for the last chosen bucket.
- Thompson sampling with forgetfulness and a softmax policy update on the same observed trials, but they do not affect the published bucket choice.
- Use the `action_policy` parameter to switch the published choice between `ucb` and `greedy` conditions.  Set this to greedy for now we can change this later if we need.

Optional ROS parameters:

- `request_topic`: topic name for the selection trigger
- `reward_topic`: topic name for reward feedback
- `bucket_topic`: topic name for the chosen bucket output
- `start_topic`: topic name for the experiment start trigger
- `stop_topic`: topic name for the experiment stop trigger
- `reset_topic`: topic name for the experiment reset trigger
- `action_policy`: published action mode, either `ucb` or `greedy`
- `thompson_forgetfulness`: decay factor for Thompson sampling memory
- `thompson_seed`: RNG seed for repeatable Thompson samples in logs
- `softmax_temperature`: exploration temperature for the comparison softmax policy
- `softmax_learning_rate`: update step size for the comparison softmax policy

## Requirements

- `requirements/base.txt`: local Python packaging dependencies
- `requirements/dev.txt`: local dev and test dependencies
- `requirements/ros.txt`: ROS 2 dependencies that come from your ROS install, not from `pip`

To run ROS 2 package tests inside the workspace:

```bash
colcon test --packages-select mab_ucb_bandit
```

Example manual test with ROS topics:

```bash
ros2 run mab_ucb_bandit ucb_bandit_node
ros2 topic pub --once /start_experiment std_msgs/msg/Empty "{}"
ros2 topic pub --once /move_to_next_ucb_bucket std_msgs/msg/Empty "{}"
ros2 topic echo /ucb_selected_bucket
ros2 topic pub --once /bandit_reward std_msgs/msg/Int32 "{data: 1}"
ros2 topic pub --once /stop_experiment std_msgs/msg/Empty "{}"
ros2 topic pub --once /reset_experiment std_msgs/msg/Empty "{}"
```
