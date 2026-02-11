import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "resources/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
dt = model.opt.timestep

# inicia no keyframe home do t1.xml
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}

LHIP, LKNEE, LANK = act["Left_Hip_Pitch"], act["Left_Knee_Pitch"], act["Left_Ankle_Pitch"]
RHIP, RKNEE, RANK = act["Right_Hip_Pitch"], act["Right_Knee_Pitch"], act["Right_Ankle_Pitch"]

base_ctrl = data.ctrl.copy()
ctrl_cmd = base_ctrl.copy()

t = 0.0
freeze_time = 10.0      # fase A: parado
squat_time = 2.0       # lento para reduzir tremor
hip_amp = 2.0
knee_amp = 1.5
ank_amp = 0.35
alpha = 0.92           # filtro (mais alto = mais suave)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.elevation = -20

    while viewer.is_running():
        t += dt
        target = base_ctrl.copy()

        if t > freeze_time:
            phase = ((t - freeze_time) % squat_time) / squat_time
            s = 0.5 * (1 - np.cos(2 * np.pi * phase))

            target[LHIP] = base_ctrl[LHIP] - hip_amp * s
            target[LKNEE] = base_ctrl[LKNEE] + knee_amp * s
            target[LANK] = base_ctrl[LANK] - ank_amp * s

            target[RHIP] = base_ctrl[RHIP] - hip_amp * s
            target[RKNEE] = base_ctrl[RKNEE] + knee_amp * s
            target[RANK] = base_ctrl[RANK] - ank_amp * s

        # filtro para reduzir vibração nos pés
        ctrl_cmd = alpha * ctrl_cmd + (1 - alpha) * target
        data.ctrl[:] = np.clip(ctrl_cmd, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.qpos[0:3]
        viewer.sync()
        time.sleep(dt)
