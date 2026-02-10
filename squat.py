import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "resources/T1_locomotion.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
dt = model.opt.timestep

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Targets and PD controller
dof_targets = data.qpos.copy()[7:]
kp = np.ones(model.nu) * 200
kd = np.ones(model.nu) * 6

# Step timing
t = 0.0
step_time = 2.0

# Lateral amplitude (COM shift)
sway_amp = -0.12

# Step height
step_height = 0.8

# Which foot is currently the support foot
left_support = True

# Variable to detect cycle restart
prev_phase = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.elevation = -20

    while viewer.is_running():

        t += dt
        phase = (t % step_time) / step_time

        # ==================================================
        # LEG SWITCH 
        # Switch only when the cycle restarts
        # ==================================================
        if phase < prev_phase:
            left_support = not left_support
        prev_phase = phase

        # Clear targets
        dof_targets[:] = 0.0

        # ==================================================
        # SWING LEG
        # ==================================================
        knee = step_height * np.sin(np.pi * phase)

        dof_targets[6] = -0.5*knee   # Right hip pitch
        dof_targets[9] = knee        # Right knee
        dof_targets[10] = -0.5*knee  # Right ankle
        dof_targets[0] = -0.5*knee   # Left hip pitch
        dof_targets[3] = knee        # Left knee
        dof_targets[4] = -0.5*knee   # Left ankle

        # ==================================================
        # PD CONTROL
        # ==================================================
        dof_pos = data.qpos[7:]
        dof_vel = data.qvel[6:]

        ctrl = kp*(dof_targets - dof_pos) - kd*dof_vel
        ctrl = np.clip(ctrl,
                       model.actuator_ctrlrange[:,0],
                       model.actuator_ctrlrange[:,1])

        data.ctrl[:] = ctrl

        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.qpos[0:3]
        viewer.sync()
        time.sleep(dt)
