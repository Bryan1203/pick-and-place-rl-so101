"""Working reward functions for lift cube task.

These are the reward functions that achieve 100% success rate:
- v11: State-based (SAC) - 100% success at 1M steps
- v19: Image-based (DrQ-v2) - 100% success at 2M steps

For historical/experimental reward versions, see _legacy_rewards.py.
"""

import numpy as np


def reward_sparse_success(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """Sparse success-only reward for grasp/lift/hold.

    Returns 1 only when the environment reports success. For the lift task,
    success means the cube is grasped, lifted above ``env.lift_height``, and
    held there for ``env.hold_steps``. Non-success transitions receive 0.
    """
    return 1.0 if info.get("is_success", False) else 0.0


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
    - Phase 2 Grasp + Lift: grasp bonus + lift progress (mirrors v11)
    - Binary lift bonus + target height bonus
    - Phase 3 Transport: tanh(3x) cube-to-target reward (gated on being lifted)
    - Phase 3.5 Lower: reward lowering the cube when over target while grasping
      (lift reward suppressed in this phase to avoid conflicting gradients)
    - Phase 4 Place: bonus for releasing cube in target zone
    - Success bonus

    Populates info["reward_components"] with per-phase breakdown for debugging.
    """
    reward = 0.0
    cube_z = info["cube_z"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    cube_to_target = info.get("cube_to_target", None)

    # Reward component tracking
    components = {
        "reach": 0.0,
        "push_down_penalty": 0.0,
        "drop_penalty": 0.0,
        "grasp_bonus": 0.0,
        "lift_progress": 0.0,
        "lift_bonus": 0.0,
        "transport": 0.0,
        "lower": 0.0,
        "place": 0.0,
        "success": 0.0,
    }

    # === Phase 1: Reach ===
    reach_reward = 1.0 - np.tanh(10.0 * gripper_to_cube)
    reward += reach_reward
    components["reach"] = reach_reward

    # Push-down penalty
    if cube_z < 0.01:
        penalty = (0.01 - cube_z) * 50.0
        reward -= penalty
        components["push_down_penalty"] = -penalty

    # Drop penalty — suppressed when cube is already over target (releasing is desired)
    over_target = cube_to_target is not None and cube_to_target < 0.05
    if was_grasping and not is_grasping and not over_target:
        reward -= 2.0
        components["drop_penalty"] = -2.0

    # === Phase 2: Grasp + Lift (mirrors v11, suppressed when over target) ===
    # env.lift_height acts as clear_height — set to container wall height in config (e.g. 0.04)
    if is_grasping and not over_target:
        reward += 0.25
        components["grasp_bonus"] = 0.25

        lift_progress = max(0, cube_z - 0.015) / (env.lift_height - 0.015)
        lift_val = lift_progress * 2.0
        reward += lift_val
        components["lift_progress"] = lift_val

    # Binary lift bonus — fires when cube clears container walls (suppressed when over target)
    if not over_target and cube_z > env.lift_height:
        reward += 1.0
        components["lift_bonus"] = 1.0

    # === Phase 3: Transport ===
    # Only reward horizontal progress once cube clears the container rim and is NOT yet over target.
    # Suppressed when over_target so the agent isn't rewarded for staying high above the container.
    if cube_z > env.lift_height and cube_to_target is not None and not over_target:
        transport_reward = 1.0 - np.tanh(3.0 * cube_to_target)
        transport_val = transport_reward * 2.0
        reward += transport_val
        components["transport"] = transport_val

    # === Phase 3.5: Lower ===
    # Once over the container and still grasping, reward descending into the container.
    # Uses tanh so the gradient is alive at any height above 0.03 (container wall).
    if over_target and is_grasping and cube_z > 0.03:
        lower_progress = 1.0 - np.tanh(10.0 * cube_z)
        lower_val = lower_progress * 2.0
        reward += lower_val
        components["lower"] = lower_val

    # === Phase 4: Place ===
    # Reward releasing cube in the target zone
    if over_target and not is_grasping and cube_z < 0.05:
        reward += 2.0
        components["place"] = 2.0

    # Success bonus
    if info["is_success"]:
        reward += 10.0
        components["success"] = 10.0

    # Store components in info for logging
    info["reward_components"] = components

    return reward


# ---------------------------------------------------------------------------
# Reward utilities (Meta-World inspired)
# ---------------------------------------------------------------------------

def _tolerance(x: float, bounds: tuple[float, float] = (0.0, 0.0),
               margin: float = 0.0, value_at_margin: float = 0.1) -> float:
    """Returns 1 inside bounds, decays smoothly to 0 outside using a long-tail sigmoid.

    Based on Meta-World's reward_utils.tolerance() with sigmoid="long_tail".
    The long-tail sigmoid is 1/(1 + (d*scale)^2) where scale is chosen so that
    the value at distance=margin equals value_at_margin.
    """
    lower, upper = bounds
    if lower <= x <= upper:
        return 1.0
    if margin == 0:
        return 0.0
    d = (lower - x if x < lower else x - upper) / margin
    scale = np.sqrt(1.0 / value_at_margin - 1.0)
    return 1.0 / ((d * scale) ** 2 + 1.0)


def _hamacher(a: float, b: float) -> float:
    """Hamacher (t-norm) product: soft AND for values in [0, 1].

    h(a, b) = (a*b) / (a + b - a*b)
    Returns 0 when either input is 0, 1 when both are 1.
    """
    denom = a + b - a * b
    return (a * b / denom) if denom > 1e-8 else 0.0


def reward_v21(env, info: dict, was_grasping: bool = False, action: np.ndarray | None = None) -> float:
    """V21: Pick-and-place reward inspired by Meta-World v2 design.

    Key differences from v20:
    - No hard phase switching. All reward terms are continuously active.
    - tolerance() replaces binary thresholds: smooth gradient from episode start.
    - hamacher_product gates grasping quality × placement quality.
    - Transport and place are unified: lifted bonus scales with in_place (5× at goal).
    - 3D in_place: measures distance to the resting position, not just XY.
    - Lift bonus persists when over target regardless of height (no cliff on descent).
    - Release reward: in_place × gripper_openness, only valuable near target.
    - Discrete stage label logged to info["stage"] for debugging.

    Stages (logged but not used for gating):
        0 = Reaching  (not grasping, cube not lifted)
        1 = Grasping  (grasping, cube not lifted)
        2 = Transporting (lifted, not over target)
        3 = Lowering  (lifted or grasping, over target)
        4 = Placed    (success)
    """
    cube_z = info["cube_z"]
    cube_pos = info["cube_pos"]
    gripper_to_cube = info["gripper_to_cube"]
    is_grasping = info["is_grasping"]
    cube_to_target_xy = info.get("cube_to_target", 0.3)

    # Compute the tolerance margin from the env's actual place target and the
    # cube's approximate start position (curriculum stage 3: cube near (0.25, 0.0)).
    # Adding 0.10 as buffer for path curvature and position randomization.
    # This ensures in_place ≈ value_at_margin (0.1) at episode start, giving a
    # meaningful gradient without inflating the reward too early.
    if env._place_target_pos is not None:
        cube_start_approx = np.array([0.25, 0.0, 0.015])
        INIT_TO_TARGET = float(np.linalg.norm(cube_start_approx - env._place_target_pos)) + 0.10
    else:
        INIT_TO_TARGET = 0.35

    # --- Grasp quality (0→1) ---
    # Tolerance: 1.0 when gripper is within 2cm of cube, decays over 12cm.
    object_grasped = _tolerance(gripper_to_cube, bounds=(0, 0.02), margin=0.12)
    # Hard contact overrides distance estimate
    if is_grasping:
        object_grasped = max(object_grasped, 0.9)

    # --- Placement quality (0→1) using 3D distance to resting position ---
    # Measures XY + Z distance to the target, so lowering the cube into the
    # container increases in_place (unlike the XY-only version in earlier drafts).
    if env._place_target_pos is not None:
        cube_to_target_3d = float(np.linalg.norm(cube_pos - env._place_target_pos))
    else:
        cube_to_target_3d = cube_to_target_xy
    in_place = _tolerance(cube_to_target_3d, bounds=(0, 0.05), margin=INIT_TO_TARGET,
                          value_at_margin=0.1)

    # --- Base reward: soft AND (both must be partially satisfied) ---
    base = _hamacher(object_grasped, in_place)

    # --- Lift + transport bonus (Meta-World key idea) ---
    # +1 for being lifted, +5*in_place so transport and place are one smooth signal.
    # Also fires when cube is over the container (XY within 5cm) regardless of height,
    # so the policy keeps its bonus while lowering into the container.
    is_lifted = cube_z > env.lift_height
    over_target_xy = cube_to_target_xy < 0.05
    lift_bonus = (1.0 + 5.0 * in_place) if (is_lifted or over_target_xy) else 0.0

    # --- Release reward: encourage opening gripper when cube is near target ---
    # Scales with in_place so it's only meaningful when the cube is actually close
    # to the resting position (3D). No incentive to drop early during transport.
    gripper_openness = np.clip((info["gripper_state"] - 0.25) / 0.30, 0.0, 1.0)
    release_reward = in_place * gripper_openness * 3.0

    # --- Push-down penalty (keep from burying the cube) ---
    push_penalty = max(0.0, (0.01 - cube_z) * 50.0) if cube_z < 0.01 else 0.0

    # --- Success ---
    success_bonus = 10.0 if info.get("is_success", False) else 0.0

    reward = base + lift_bonus + release_reward - push_penalty + success_bonus

    # --- Stage classification (for W&B logging) ---
    if info.get("is_success", False):
        stage = 4
    elif over_target_xy and (is_lifted or is_grasping):
        stage = 3
    elif is_lifted:
        stage = 2
    elif is_grasping:
        stage = 1
    else:
        stage = 0

    # --- Populate info for callback logging ---
    info["reward_components"] = {
        "base_hamacher": base,
        "lift_bonus": lift_bonus,
        "release": release_reward,
        "push_penalty": -push_penalty,
        "success": success_bonus,
    }
    info["reward_stage"] = stage

    return reward
