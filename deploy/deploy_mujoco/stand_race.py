"""使用当前默认初始姿态与 SLR 原始 PD 参数，查看 Race 站立。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_robot(config=None, overrides=None):
    config = Path(config) if config else (
        Path(__file__).with_name("configs") / "race.yaml"
    )
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if overrides:
        cfg.update(overrides)
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
    parser.add_argument("--collision", action="store_true", help="显示碰撞体")
    parser.add_argument("--config", type=Path, help="姿态及控制配置文件")
    parser.add_argument("--zero", action="store_true", help="12 个目标角全部为零")
    parser.add_argument("--angles", type=float, nargs=12,
                        help="按左前、右前、左后、右后，每腿三个角，单位 rad")
    for name in ("kps", "kds", "torque_limits"):
        parser.add_argument("--" + name, type=float, nargs="+",
                            help="每腿三个值，或按关节顺序提供十二个值")
    parser.add_argument("--height", type=float, help="释放初始高度 m，不锁定机身")
    parser.add_argument("--auto_height", action="store_true",
                        help="按当前姿态把最低足底放到地面上方 5 mm 后释放")
    parser.add_argument("--duration", type=float, default=0,
                        help="测试秒数，窗口模式 0 表示持续运行")
    parser.add_argument("--headless", action="store_true", help="无窗口测试")
    parser.add_argument("--report", type=Path, help="保存测试结果 JSON")
    args = parser.parse_args()
    if args.zero and args.angles is not None:
        parser.error("--zero 和 --angles 只能选择一个")
    if args.height is not None and args.auto_height:
        parser.error("--height 和 --auto_height 只能选择一个")
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("测试时长必须为非负有限值")
    if args.headless and args.duration == 0:
        parser.error("无窗口测试必须指定 --duration")
    overrides = {}
    if args.zero:
        overrides["default_angles"] = [0.] * 12
    if args.angles is not None:
        if not np.isfinite(args.angles).all():
            parser.error("目标角必须为有限数值")
        overrides["default_angles"] = args.angles
    for name in ("kps", "kds", "torque_limits"):
        values = getattr(args, name)
        if values is not None:
            if len(values) not in (3, 12):
                parser.error(f"{name} 必须提供 3 或 12 个值")
            if not np.isfinite(values).all() or min(values) < 0:
                parser.error(f"{name} 必须为非负有限值")
            overrides[name] = values * 4 if len(values) == 3 else values
    if args.height is not None:
        if not np.isfinite(args.height) or args.height <= 0:
            parser.error("初始高度必须为正有限值")
        cfg_path = args.config or Path(__file__).with_name("configs") / "race.yaml"
        base = yaml.safe_load(cfg_path.read_text())["base_init_pos"]
        overrides["base_init_pos"] = [base[0], base[1], args.height]
    robot = load_robot(args.config, overrides)
    model, data, cfg, *_ = robot
    feet = [model.body(f"pitch03_leg_{leg}_link").id
            for leg in ("f_l", "f_r", "b_l", "b_r")]
    if args.auto_height:
        gaps = [data.geom_xpos[i, 2] - model.geom_size[i, 0]
                for i in range(model.ngeom)
                if model.geom_bodyid[i] in feet
                and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE]
        data.qpos[2] += .005 - min(gaps)
        cfg["base_init_pos"] = data.qpos[:3].tolist()
        mujoco.mj_forward(model, data)
    print("纯 PD：自由基座、重力开启，机身高度由物理仿真决定。")
    print(f"目标角：{cfg['default_angles']}")
    print(f"Kp：{cfg['kps']}；Kd：{cfg['kds']}")
    print(f"释放初始高度：{data.qpos[2]:.4f} m", flush=True)
    samples = []
    nonfoot_steps = 0
    context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    with context as viewer:
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body(cfg["base_name"]).id
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135.
            viewer.cam.elevation = -20.
            viewer.opt.geomgroup[1] = int(args.collision)
            viewer.opt.geomgroup[2] = int(not args.collision)
        deadline = time.monotonic()
        frame_steps = max(1, round(1 / (60 * float(cfg["pd_dt"]))))
        while (viewer is None or viewer.is_running()) and (
                args.duration == 0 or data.time < args.duration):
            for _ in range(frame_steps):
                step_pd(*robot)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("仿真状态出现非有限值")
            loads = np.zeros(4)
            other = False
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                bodies = [model.geom_bodyid[g] for g in contact.geom]
                if 0 not in bodies:
                    continue
                body = bodies[1] if bodies[0] == 0 else bodies[0]
                if body in feet:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_id, force)
                    loads[feet.index(body)] += max(0., force[0])
                else:
                    other = True
            nonfoot_steps += int(other)
            rot = data.xmat[model.body(cfg["base_name"]).id].reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(rot[2, 2], -1., 1.)))
            samples.append([float(data.time), float(data.qpos[2]), float(tilt),
                            *loads.tolist()])
            # 长时间窗口只保留末两秒统计。
            samples = [row for row in samples if row[0] >= data.time - 2.]
            if viewer is not None:
                viewer.sync()
                deadline += frame_steps * float(cfg["pd_dt"])
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -.5:
                    deadline = time.monotonic()
    rows = np.asarray(samples)
    report = {
        "duration_s": float(data.time), "config": cfg,
        "final_height_m": float(data.qpos[2]),
        "final_tilt_deg": float(rows[-1, 2]),
        "last_2s_mean_height_m": float(rows[:, 1].mean()),
        "last_2s_mean_foot_loads_n": rows[:, 3:].mean(axis=0).tolist(),
        "last_2s_four_feet_contact_fraction": float(
            (rows[:, 3:] > 1.).all(axis=1).mean()),
        "nonfoot_ground_contact_samples": nonfoot_steps,
        "final_angles_rad": data.qpos[robot[3]].tolist(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
