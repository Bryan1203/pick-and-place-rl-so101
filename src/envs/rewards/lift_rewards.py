"""Working reward functions for lift cube task.

These are the reward functions that achieve 100% success rate:
- v11: State-based (SAC) - 100% success at 1M steps
- v19: Image-based (DrQ-v2) - 100% success at 2M steps

For historical/experimental reward versions, see _legacy_rewards.py.
"""

import numpy as np


def reward_v11(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V11: Dense reward for state-based training.

    Structure:
    - Reach reward (tanh distance)
    - Push-down penalty
    - Drop penalty
    - Grasp bonus + continuous lift reward
    - Binary lift bonus
    - Target height bonus
    - Action rate penalty (only when lifted)
    - Success bonus

    Achieved 100% success at 1M steps with SAC.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]

    # Reach reward
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 0.25

        # Continuous lift reward when grasping
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 2.0

    # Binary lift bonus
    if cube_z > 0.02:
        reward += 1.0

    # Target height bonus (aligned with success: z > lift_height)
    if cube_z > env.lift_height:
        reward += 1.0

    # Action rate penalty for smoothness (only when lifted, to not hinder lifting)
    if action is not None and cube_z > 0.06:
        action_delta = action - env._prev_action
        action_penalty = 0.01 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v19(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V19: Dense reward for image-based training.

    Key innovations over v11:
    - Per-finger reach reward (moving finger gets own reach gradient)
    - Stronger grasp bonus (1.5 vs 0.25)
    - Doubled lift coefficient (4.0 vs 2.0)
    - Threshold ramp from 0.04m to 0.08m
    - Hold count bonus (escalating reward for sustained height)

    Achieved 100% success at 2M steps with DrQ-v2.
    """
    reward = 0.0
    cube_pos = info["cube_pos"]
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    gripper_state = info["gripper_state"]
    is_grasping = info["is_grasping"]
    hold_count = info["hold_count"]
    is_closed = gripper_state < 0.25

    # Standard gripper reach (static finger is part of gripper frame)
    gripper_reach = 1.0 - np.tanh(10.0 * gripper_to_cube)

    # Moving finger reach - only applies when gripper is close to cube
    reach_threshold = 0.7  # ~3cm from cube
    if gripper_reach < reach_threshold:
        reach_reward = gripper_reach
    else:
        if is_closed:
            moving_reach = 1.0
        else:
            moving_finger_pos = env._get_moving_finger_pos()
            moving_to_cube = np.linalg.norm(moving_finger_pos - cube_pos)
            moving_reach = 1.0 - np.tanh(10.0 * moving_to_cube)

        reach_reward = (gripper_reach + moving_reach) * 0.5

    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        push_penalty = (0.01 - cube_z) * 50.0
        reward -= push_penalty

    # Drop penalty
    if was_grasping and not is_grasping:
        reward -= 2.0

    # Grasp bonus
    if is_grasping:
        reward += 1.5

        # Continuous lift reward (4.0x coefficient)
        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        reward += lift_progress * 4.0

        # Binary lift bonus at 0.02m
        if cube_z > 0.02:
            reward += 1.0

        # Linear threshold ramp from 0.04m to 0.08m
        if cube_z > 0.04:
            threshold_progress = min(1.0, (cube_z - 0.04) / (env.lift_height - 0.04))
            reward += threshold_progress * 2.0

    # Target height bonus
    if cube_z > env.lift_height:
        reward += 1.0

        # Hold count bonus - escalating reward for sustained height
        reward += 0.5 * hold_count

    # Action rate penalty during hold phase
    if action is not None and cube_z > env.lift_height and hold_count > 0:
        action_delta = action - env._prev_action
        action_penalty = 0.02 * np.sum(action_delta**2)
        reward -= action_penalty

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward


def reward_v20(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V20: Dense reward for pick-and-place to a container.

    Structure:
    - Phase 1 Reach: tanh gripper-to-cube distance reward
    - Push-down penalty
    - Drop penalty (disabled when cube is over target — releasing is the goal)
    - Phase 2 Grasp + Lift: grasp bonus + lift progress to clear container rim
    - Rim clearance bonus
    - Phase 3 Transport: tanh cube-to-target reward (gated on being lifted)
    - Phase 4 Place: bonus for releasing cube in target zone
    - Success bonus
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    cube_to_target = info.get("cube_to_target", None)

    # === Phase 1: Reach ===
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        reward -= (0.01 - cube_z) * 50.0

    # Drop penalty — suppressed when cube is already over target (releasing is desired)
    over_target = cube_to_target is not None and cube_to_target < 0.05
    if was_grasping and not is_grasping and not over_target:
        reward -= 2.0

    # === Phase 2: Grasp + Lift ===
    clear_height = 0.04  # must clear 3cm container walls with margin
    if is_grasping:
        reward += 0.25

        lift_progress = np.clip((cube_z - 0.015) / (clear_height - 0.015), 0.0, 1.0)
        reward += lift_progress * 2.0

    # Rim clearance bonus
    if cube_z > clear_height:
        reward += 1.0

    # === Phase 3: Transport ===
    # Only reward horizontal progress once cube clears the container rim
    if cube_z > clear_height and cube_to_target is not None:
        transport_reward = 1.0 - np.tanh(10.0 * cube_to_target)
        reward += transport_reward * 2.0

    # === Phase 4: Place ===
    # Reward releasing cube in the target zone
    if over_target and not is_grasping and cube_z < 0.05:
        reward += 2.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0

    return reward
