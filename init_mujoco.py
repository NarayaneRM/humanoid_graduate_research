import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "resources/scene.xml"

# =========================
# LOAD MODEL
# =========================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

dt = model.opt.timestep

# =========================
# RESET + INITIAL POSE
# =========================

mujoco.mj_resetData(model, data)



data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

# =========================
# STANDING TARGET (LIKE BOOSTER)
# =========================

# Target = initial joint configuration
dof_targets = data.qpos.copy()[7:]

# PD gains (safe values)
kp = np.ones(model.nu) * 200.0
kd = np.ones(model.nu) * 6.0

print("Standing controller ready")
print("Number of actuators:", model.nu)

# =========================
# MAIN LOOP
# =========================

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.elevation = -20

    while viewer.is_running():

        # Joint states (skip floating base)
        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:]

        # ===== BOOSTER STYLE PD =====
        ctrl = kp * (dof_targets - dof_pos) - kd * dof_vel

        # Respect actuator limits
        ctrl = np.clip(
            ctrl,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1]
        )

        data.ctrl[:] = ctrl

        mujoco.mj_step(model, data)

        viewer.cam.lookat[:] = data.qpos[0:3]
        viewer.sync()

        time.sleep(dt)
