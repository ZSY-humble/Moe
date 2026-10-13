"""使用当前默认初始姿态与 SLR 原始 PD 参数，查看 Race 站立。"""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_robot(config=None):
    config = Path(config) if config else (
        Path(__file__).with_name("configs") / "race.yaml"
    )
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    model_path = Path(cfg["model_path"].replace("{ROOT_DIR}", str(ROOT)))
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = float(cfg["physics_dt"])
    data = mujoco.MjData(model)
    joints = [model.joint(name).id for name in cfg["joint_names"]]
    qpos = model.jnt_qposadr[joints]
    qvel = model.jnt_dofadr[joints]
    actuators = []
    for joint in joints:
        matches = np.flatnonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == joint)
        )
        if len(matches) != 1:
            raise ValueError("每个 Race 主动关节必须唯一对应一个力矩电机")
        actuators.append(int(matches[0]))
    if len(set(joints)) != 12 or model.nu != 12:
        raise ValueError("Race 必须包含 12 个不同的主动关节")
    target = np.asarray(cfg["default_angles"], dtype=float)
    if np.any(target < model.jnt_range[joints, 0]) or np.any(
        target > model.jnt_range[joints, 1]
    ):
        raise ValueError("Race 的默认关节角超出 URDF 限位")
    if np.any(model.dof_damping[qvel]) or np.any(model.dof_frictionloss[qvel]):
        raise ValueError("被动项在控制器中单独施加，模型不得重复添加")
    data.qpos[:3] = cfg["base_init_pos"]
    data.qpos[3:7] = cfg["base_init_quat"]
    data.qpos[qpos] = target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data, cfg, qpos, qvel, actuators


def step_pd(model, data, cfg, qpos, qvel, actuators):
    """沿用来源的先电机限幅、再叠加被动阻尼和摩擦的次序。"""
    velocity = data.qvel[qvel]
    torque = np.asarray(cfg["kps"]) * (
        np.asarray(cfg["default_angles"]) - data.qpos[qpos]
    ) - np.asarray(cfg["kds"]) * velocity
    limits = np.asarray(cfg["torque_limits"])
    torque = np.clip(torque, -limits, limits)
    moving_sign = np.where(np.abs(velocity) > 1e-6, np.sign(velocity), 0.0)
    torque -= np.asarray(cfg["passive_damping"]) * velocity
    torque -= np.asarray(cfg["passive_friction"]) * moving_sign
    if not np.isfinite(torque).all():
        raise RuntimeError("PD 力矩出现非有限数值")
    data.ctrl[actuators] = torque
    ratio = float(cfg["pd_dt"]) / model.opt.timestep
    steps = int(round(ratio))
    if steps < 1 or not np.isclose(ratio, steps):
        raise ValueError("MuJoCo 积分步长必须整除原始 5 ms PD 周期")
    for _ in range(steps):
        mujoco.mj_step(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collision", action="store_true", help="显示碰撞体，隐藏外观网格"
    )
    parser.add_argument("--config", type=Path, help="指定姿态配置文件")
    args = parser.parse_args()
    robot = load_robot(args.config)
    model, data, cfg, *_ = robot
    print("Race 0710：平地、自由基座、重力开启、固定配置中的目标角。")
    print(f"每腿 Kp={cfg['kps'][:3]}，Kd={cfg['kds'][:3]}。")
    print(f"前腿目标角={cfg['default_angles'][:3]}，"
          f"后腿={cfg['default_angles'][6:9]} rad。")
    print("电机限幅及被动项沿用 SLR 参数；关闭窗口结束。", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body(cfg["base_name"]).id
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        # 地面组 0，碰撞组 1，外观组 2；显示开关不改物理碰撞。
        viewer.opt.geomgroup[0] = 1
        viewer.opt.geomgroup[1] = int(args.collision)
        viewer.opt.geomgroup[2] = int(not args.collision)
        viewer.sync()
        deadline = time.monotonic()
        frame_steps = max(1, round(1 / (60 * float(cfg["pd_dt"]))))
        while viewer.is_running():
            for _ in range(frame_steps):
                step_pd(*robot)
            viewer.sync()
            deadline += frame_steps * float(cfg["pd_dt"])
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                deadline = time.monotonic()


if __name__ == "__main__":
    main()
