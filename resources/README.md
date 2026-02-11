# Booster T1 Description (Playground)

> [!IMPORTANT]
> Requires MuJoCo 2.3.4 or later.

## Overview

This package contains a simplified robot description of the [T1 Humanoid
Robot](https://www.boosterobotics.com/robots/) developed by [Google DeepMind](https://deepmind.google/). It is derived from the [publicly available
URDF
description](https://github.com/google-deepmind/mujoco_menagerie/tree/main/booster_t1).

<p float="left">
  <img src="sim_scene.png" width="400">
</p>


## Project-specific adaptations

This repository adds an extra layer on top of the adapted T1 model from the
MuJoCo Menagerie flow:

1. Increased `kp` and `kv` for tighter position-actuator tracking in task
   execution.
2. Added a task scene (`scene_t1.xml`) that composes the robot model with
   environment elements.
3. Added a box object (`box.xml`) for interaction tasks (e.g., reach, contact,
   and manipulation experiments).

These additions are focused on task execution and controller validation, while
preserving the original base model structure.

## License

This model is released under an [Apache-2.0 License](LICENSE).
