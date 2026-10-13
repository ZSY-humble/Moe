"""在平地上用配置文件中的 PD 参数保持 Go2 初始关节姿态。"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collision", action="store_true", help="仅显示碰撞几何体"
    )
    args = parser.parse_args()
    config = Path(__file__).with_name("configs") / "go2.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/go2/flat.xml"))
    model.opt.timestep = float(cfg["simulation_dt"])
    data = mujoco.MjData(model)

    # 按关节名称匹配配置，避免依赖模型中的关节和电机排列顺序。
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in cfg["mujoco_joint_names"]
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise ValueError("配置中存在模型未定义的关节")
    qpos_ids = model.jnt_qposadr[joint_ids]
    qvel_ids = model.jnt_dofadr[joint_ids]
    actuator_ids = []
    for joint_id in joint_ids:
        matches = np.flatnonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == joint_id)
        )
        if len(matches) != 1:
            raise ValueError("每个受控关节必须对应一个电机")
        actuator_ids.append(int(matches[0]))

    target = np.asarray(cfg["default_angles"], dtype=float)
    kp = np.asarray(cfg["kps"], dtype=float)
    kd = np.asarray(cfg["kds"], dtype=float)
    data.qpos[:3] = cfg["base_init_pos"]
    data.qpos[3:7] = cfg["base_init_quat"]
    data.qpos[qpos_ids] = target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"模型：宇树 Go2；场景：平地；配置：{config}", flush=True)
    print(f"Kp：{kp.tolist()}\nKd：{kd.tolist()}", flush=True)
    print(f"目标关节角（rad）：{target.tolist()}", flush=True)
    print("控制律：τ = Kp × (目标角度 − 当前角度) − Kd × 关节速度")
    print("浮动基座、重力和地面接触均启用，关闭窗口结束。", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        if args.collision:
            viewer.opt.geomgroup[2] = 0
            viewer.opt.geomgroup[0] = 1
        else:
            # group 0 还包含地面；只隐藏机器人碰撞体，保留地面。
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] > 0:
                    if model.geom_contype[geom_id] != 0:
                        model.geom_group[geom_id] = 3
            viewer.opt.geomgroup[3] = 0
        viewer.sync()
        frame_deadline = time.monotonic()
        while viewer.is_running():
            # 不加载策略，不固定基座，不添加重力补偿。
            for _ in range(max(1, round(1.0 / (60 * model.opt.timestep)))):
                data.ctrl[actuator_ids] = (
                    kp * (target - data.qpos[qpos_ids])
                    - kd * data.qvel[qvel_ids]
                )
                mujoco.mj_step(model, data)
            viewer.sync()
            frame_deadline += model.opt.timestep * max(
                1, round(1.0 / (60 * model.opt.timestep))
            )
            wait = frame_deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.5:
                frame_deadline = time.monotonic()


if __name__ == "__main__":
    main()
