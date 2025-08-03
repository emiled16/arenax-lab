# Franka Golf - ArenaX Labs ML Hiring Challenge

## HIRING OVERVIEW

This competition is part of ArenaX Labs' ML Hiring Campaign. If you're an engineer specializing in Reinforcement Learning, your Franka Golf submission is your application.

### How it works:

1. Connect your LinkedIn account to SAI
2. Submit your trained model for the Franka Golf challenge
3. Apply to our posting on LinkedIn
4. At the end of the competition, we'll review all submissions
5. The top 10 performing submissions will be invited to a technical interview to join our Machine Learning team

If you're up for the challenge, apply for the opportunity to help shape SAI's RL environments and contribute to cutting-edge research. Good luck!

## Description

Franka Golf is a precision manipulation environment where a 7-DOF Franka robotic arm must play mini-golf by grasping a club and striking a ball toward a hole. The agent operates in a continuous action space, controlling the arm's end-effector position, orientation, and gripper state through inverse kinematics. Success requires fine motor control for grasping, dynamic coordination for swinging, and spatial reasoning to align shots accurately. The task integrates complex sub-skills like object interaction, tool use, and trajectory planning into a unified challenge.

### Competition Timeline

- **Start**: 3 weeks ago (Tue Jul 08 2025)
- **Finish**: 2 weeks to go (Fri Aug 22 2025)

## Create Environment

To instantiate this competition environment, you can use:

```python
from sai_rl import SAIClient

sai = SAIClient(comp_id="franka-ml-hiring")
env = sai.make_env()
```

To learn more about this environment, please click here.

## Evaluation

**Note**: Models submitted prior to July 30, 2025 will be fully re-evaluated with an updated evaluation function. For more details on the changes made, see the "Evaluation" tab.

Agents are evaluated based on their ability to complete the full golf task sequence: grasping the club, maintaining grip, and guiding the ball into the hole. The reward function is shaped by intermediate sub-goals such as proper alignment, secure grasping, and progressive ball movement, with heavy penalties for failure conditions like dropping the club or overshooting the hole. Final performance is determined by successful completion of the task with smooth, efficient, and precise motion.

### Evaluation Function

```python
_FLOAT_EPS = np.finfo(np.float64).eps

def distance_between_objects(pos1, pos2):
    return abs(np.linalg.norm(pos1 - pos2))

def ball_in_hole(ball_pos, hole_pos):
    return (distance_between_objects(ball_pos, hole_pos) < 0.06).astype(np.float32)

def check_grasp(env):
    club_body_id = env.golf_club_id
    left_finger_body_id = env.left_finger_body_id
    right_finger_body_id = env.right_finger_body_id
    ncon = env.robot_model.data.ncon
    
    if ncon == 0:
        return False

    contact = env.robot_model.data.contact
    club_left_contact = False
    club_right_contact = False
    
    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2
        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]
        
        if body1 == club_body_id or body2 == club_body_id:
            other_body = body2 if body1 == club_body_id else body1
            
            if other_body == left_finger_body_id:
                club_left_contact = True
            elif other_body == right_finger_body_id:
                club_right_contact = True
    
    return club_left_contact and club_right_contact

def check_ball_club_contact(env):
    club_body_id = env.club_head_id
    ball_body_id = env.golf_ball_id
    ncon = env.robot_model.data.ncon
    
    if ncon == 0:
        return False

    contact = env.robot_model.data.contact
    
    for i in range(ncon):
        geom1 = contact[i].geom1
        geom2 = contact[i].geom2
        body1 = env.robot_model.model.geom_bodyid[geom1]
        body2 = env.robot_model.model.geom_bodyid[geom2]
        
        if (body1 == club_body_id and body2 == ball_body_id) or (
            body1 == ball_body_id and body2 == club_body_id
        ):
            return True
    
    return False

def evaluation_fn(env, eval_state):
    if not eval_state.get("timestep", False):
        eval_state["timestep"] = 0
    
    if not eval_state.get("closest_distance_to_club", False):
        eval_state["closest_distance_to_club"] = 10000

    ee_pos = env.robot_model.data.site(env.ee_site_id).xpos
    club_grip_pos = env.robot_model.data.xpos[env.golf_club_id]
    ball_pos = env.robot_model.data.xpos[env.golf_ball_id]
    hole_pos = env.robot_model.data.xpos[env.golf_hole_id]

    ee_club_dist = distance_between_objects(ee_pos, club_grip_pos)
    robot_grasped_club = check_grasp(env)
    robot_hit_ball_with_club = check_ball_club_contact(env)

    if robot_hit_ball_with_club and not eval_state.get(
        "robot_hit_ball_with_club", False
    ):
        eval_state["robot_hit_ball_with_club"] = eval_state["timestep"]
    
    if robot_grasped_club and not eval_state.get("robot_grasped_club", False):
        eval_state["robot_grasped_club"] = eval_state["timestep"]

    eval_state["closest_distance_to_club"] = min(
        eval_state["closest_distance_to_club"], ee_club_dist
    )
    eval_state["timestep"] += 1

    if eval_state.get("terminated", False) or eval_state.get("truncated", False):
        reward = 0.0
        
        if ball_in_hole(ball_pos, hole_pos):
            reward += (
                10.0 - eval_state["timestep"] * 0.01
            )  # Min 4.5 reward for getting the ball in the hole
        else:
            reward += (
                1.1867 - distance_between_objects(ball_pos, hole_pos)
            ) * 2  # Max ~2.25 reward for being close to the hole
        
        if eval_state.get("robot_hit_ball_with_club", False):
            reward += (
                3.5 - eval_state["robot_hit_ball_with_club"] * 0.005
            )  # Min 0.25 reward for hitting the ball with club head
        
        if eval_state.get("robot_grasped_club", False):
            reward += (
                1.65 - eval_state["robot_grasped_club"] * 0.001
            )  # Min 1.0 reward for grasping
        else:
            reward += max(
                1 - eval_state["closest_distance_to_club"], 0
            )  # Max 1.0 reward for being close to the club grip
        
        return (reward, eval_state)
    
    return (0.0, eval_state)
```

---

## Environment

### Description

Franka Golf tasks a Franka 7-DOF robotic arm with playing a simplified golf game by grasping a golf club, striking a golf ball, and maneuvering it toward a designated hole. The robot must precisely control its end-effector to approach the club, align and close its gripper to secure it, lift the club, and swing to hit the ball, ultimately guiding the ball closer to the hole.

The primary challenges of Franka Golf are: precise manipulation for grasping, dynamic control for hitting the ball, and spatial reasoning to approach the hole, all within a continuous and high-dimensional action and observation space.

| Property | Value |
|----------|-------|
| **Action Space** | Box(shape=(7,), low=-1, high=1) |
| **Observation Space** | Box(shape=(31,), low=-inf, high=inf) |
| **Import** | `gym.make("FrankaIkGolfCourseEnv-v0", render_mode="human")` |

### Actions

The action space is a continuous vector of shape (7,), where each dimension corresponds to a component of the inverse kinematics command for the robotic arm's end-effector pose and gripper control. The table below describes each dimension, interpreted by the IK solver to compute joint commands.

| Index | Action |
|-------|--------|
| 0 | End-Effector X (Δx) |
| 1 | End-Effector Y (Δy) |
| 2 | End-Effector Z (Δz) |
| 3 | End-Effector Roll (Δroll) |
| 4 | End-Effector Pitch (Δpitch) |
| 5 | End-Effector Yaw (Δyaw) |
| 6 | Gripper Open/Close |

- **End-Effector X, Y, Z (Indices 0-2)**: Specifies the displacement (delta) of the end-effector relative to its current position, given as [Δx, Δy, Δz] in meters.
- **End-Effector Roll, Pitch, Yaw (Indices 3-5)**: Specifies the angular displacement (delta) of the end-effector orientation relative to its current orientation, given as [Δroll, Δpitch, Δyaw] in radians, applied as incremental rotations.
- **Gripper Open/Close (Index 6)**: Adjusts the gripper state, where -1 closes and 1 fully opens. This behavior is identical in both environments.

### Observations

The observation is a vector containing 6 key components, representing the state of the robotic arm, golf club, ball, and their relationships in the world frame, with a total dimension of (31,) for a single environment. Below, each observation is described with its purpose and dimension.

#### Joint Positions
- **Description**: The absolute positions of the robot's joints, showing the current arm configuration.
- **Dimension**: (9,) (7-DOF arm + 2 gripper joints).

#### Joint Velocities
- **Description**: The velocities of the robot's joints, indicating how fast each joint is moving.
- **Dimension**: (9,) (7-DOF arm + 2 gripper joints).

#### Ball Position
- **Description**: The golf ball's position in the world frame, allowing the robot to locate it.
- **Dimension**: (3,) ([x, y, z] coordinates).

#### Club Position
- **Description**: The golf club's position in the world frame, helping the robot track its location.
- **Dimension**: (3,) ([x, y, z] coordinates).

#### Club Orientation
- **Description**: The quaternion orientation of the golf club in the world frame, detailing its alignment.
- **Dimension**: (4,) ([w, x, y, z] quaternion components).

#### Hole Position
- **Description**: The golf hole's position in the world frame.
- **Dimension**: (3,) ([x, y, z] coordinates).

### Rewards

The reward function is designed to guide the robotic arm in the Golf Course environments toward successfully grasping a golf club and hitting a ball into a hole. It combines weighted sub-task rewards, each incentivizing a specific behavior critical to the task, into a total reward. Below, each reward component is described with its purpose and weight, followed by the total reward equation.

#### Reward Components

**End-Effector Positioning:**
- **r_ee_club_dist**: Encourages the end-effector to move close to the golf club grip for grasping. Weight: 1.0
- **r_align_ee_handle**: Promotes alignment of the end-effector's orientation with the club handle's orientation to ensure a proper grasp. Weight: 2.0

**Gripper Control:**
- **r_fingers_club_grasp**: Rewards proper grasping of the club grip with the fingers. Weight: 5.0

**Task Objective:**
- **r_ball_hole_dist**: Incentivizes moving the golf ball closer to the hole, the primary task goal. Weight: 10.0
- **r_ball_in_hole**: Provides a large positive reward when the ball enters the hole, marking task success. Weight: 20.0

**Penalties:**
- **r_joint_vel**: Penalizes excessive joint velocities to promote smooth and efficient arm movements. Weight: -0.0001
- **r_club_dropped**: Applies a strong penalty if the club is dropped below a minimum height of 0.25, preventing task failure. Weight: -2.0
- **r_ball_passed_hole**: Penalizes overshooting the hole after hitting the ball, encouraging precision. Weight: -4.0

#### Total Reward Equation

The total reward (R) at each timestep is computed as a weighted sum of the following components:

R = 1.0 · r_ee_club_dist + 2.0 · r_align_ee_handle + 5.0 · r_fingers_club_grasp + 10.0 · r_ball_hole_dist + 20.0 · r_ball_in_hole - 0.0001 · r_joint_vel - 2.0 · r_club_dropped - 4.0 · r_ball_passed_hole

### Additional Info

The info dictionary returned by the environment contains:
- **success** (bool): True if the task is completed (e.g., ball in hole), otherwise False.

### Starting State

At the beginning of each episode, when the environment is reset, the Franka 7-DOF robotic arm is initialized at a fixed position in the top-right quadrant of the golf course, with its base anchored and joints set to a default configuration. The golf club is randomly spawned near the robot's gripper, within a small radius (e.g., 0.1-0.3 meters) of the end-effector's initial position, ensuring graspability while introducing slight variability. The golf ball is placed at a fixed location along the hole's axis.

### Episode Termination

An episode concludes when one of the following conditions is met, after which the environment resets to the starting state for the next episode:

1. **Ball in Hole**: The golf ball enters the hole, defined as the ball's center being within 0.05 meters of the hole's center.
2. **Club Dropped**: The golf club drops to the ground, detected when the club's grip is no longer held by the gripper and contacts the ground.
3. **Ball Passes Hole**: The golf ball overshoots the hole, defined as the ball's center moving beyond the hole's center by more than 0.1 meters after being struck.
4. **Time Limit**: The maximum time limit is reached, corresponding to 650 time steps.

## Create Environment

The preferred method for instantiating a SAI environment is through the SAI Client (internet connection required):

```python
from sai_rl import SAIClient

sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env()
```

However, if you do not have an internet connection, then you can also use:

```python
import sai_mujoco
import gymnasium as gym

env = gym.make("FrankaIkGolfCourseEnv-v0")
```

### Arguments

The environment accepts the following optional keyword arguments:

- **hide_overlay** (bool, default=True): If True, hides the on-screen overlay displaying additional stats during rendering.
- **render_mode** (str): "human" for interactive rendering; "rgb_array" to return RGB frames at each timestep.
- **deterministic_reset** (bool, default=True): If True, disables random noise in the environment and robot state during reset.

```python
env = sai.make_env(
    hide_overlay=False,
    deterministic_reset=False
)
```












