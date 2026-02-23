import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "resources/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
dt = model.opt.timestep

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}

LHIP, LKNEE, LANK = act["Left_Hip_Pitch"], act["Left_Knee_Pitch"], act["Left_Ankle_Pitch"]
RHIP, RKNEE, RANK = act["Right_Hip_Pitch"], act["Right_Knee_Pitch"], act["Right_Ankle_Pitch"]
LSP, LSR = act["Left_Shoulder_Pitch"], act["Left_Shoulder_Roll"]
LEP, LEY = act["Left_Elbow_Pitch"], act["Left_Elbow_Yaw"]
RSP, RSR = act["Right_Shoulder_Pitch"], act["Right_Shoulder_Roll"]
REP, REY = act["Right_Elbow_Pitch"], act["Right_Elbow_Yaw"]

base_ctrl = data.ctrl.copy()
ctrl_cmd = base_ctrl.copy()

t = 0.0
freeze_time = 0.4
reach_time = 2.8
lift_time = 5.0
hold_time = 3.0
alpha = 0.94

settle_time = 1.8


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_pose(target, pose, w):
    for k, v in pose.items():
        target[k] = base_ctrl[k] + w * v


def blend_poses(target, pose_a, pose_b, p):
    keys = set(pose_a.keys()) | set(pose_b.keys())
    for k in keys:
        va = pose_a.get(k, 0.0)
        vb = pose_b.get(k, 0.0)
        target[k] = base_ctrl[k] + (1.0 - p) * va + p * vb


squat_reach_pose = {
    LHIP: -1.85, LKNEE: 1.15, LANK: -0.22,
    RHIP: -1.85, RKNEE: 1.15, RANK: -0.22,
    LSP: -0.95, LSR: 0.05, LEP: 0.0, LEY: -0.0,
    RSP: -0.95, RSR: -0.05, REP: 0.0, REY: 0.0,
}

squat_squeeze_pose = {
    LHIP: -1.85, LKNEE: 1.15, LANK: -0.22,
    RHIP: -1.85, RKNEE: 1.15, RANK: -0.22,
    LSP: -0.98, LSR: -0.1, LEP: 0.0, LEY: -0.0,
    RSP: -0.98, RSR: 0.1, REP: 0.0, REY: 0.0,
}

squat_lift_pose = {
    LHIP: -0.70, LKNEE: 0.90, LANK: -0.28,
    RHIP: -0.70, RKNEE: 0.90, RANK: -0.28,
    LSP: -0.92, LSR: -0.42, LEP: 0.45, LEY: -0.28,
    RSP: -0.92, RSR: 0.42, REP: 0.45, REY: 0.28,
}

squat_carry_pose = {
    LHIP: -0.70, LKNEE: 0.90, LANK: -0.28,
    RHIP: -0.70, RKNEE: 0.90, RANK: -0.28,
    LSP: -0.98, LSR: -0.18, LEP: 0.50, LEY: -0.20,
    RSP: -0.98, RSR: 0.18, REP: 0.50, REY: 0.20,
}

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.elevation = -20

    while viewer.is_running():
        t += dt
        target = base_ctrl.copy()

        if t <= freeze_time:
            pass
        elif t <= (freeze_time + reach_time):
            p = smooth01((t - freeze_time) / reach_time)
            apply_pose(target, squat_reach_pose, p)
        elif t <= (freeze_time + reach_time + settle_time):
            apply_pose(target, squat_squeeze_pose, 1.0)
        else:
            tau = t - (freeze_time + reach_time + settle_time)
            if tau < lift_time:
                p = smooth01(tau / lift_time)
                blend_poses(target, squat_squeeze_pose, squat_lift_pose, p)
            elif tau < (lift_time + hold_time):
                apply_pose(target, squat_carry_pose, 1.0)
            else:
                apply_pose(target, squat_carry_pose, 1.0)

        ctrl_cmd = alpha * ctrl_cmd + (1 - alpha) * target
        data.ctrl[:] = np.clip(ctrl_cmd, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.qpos[0:3]
        viewer.sync()
        time.sleep(dt)
