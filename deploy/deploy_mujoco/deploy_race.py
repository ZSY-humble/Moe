"""在 MuJoCo 中加载本项目训练并导出的 Race CTS 策略。"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

# 无窗口截图使用 EGL；必须在导入 MuJoCo 前选择图形后端。
if "--headless" in sys.argv and "--render-output" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import torch

from stand_race import ROOT, load_robot, step_pd


OBSERVATION_ORDER = [
    "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos", "joint_vel", "actions",
]
OBSERVATION_SCALES = {
    "base_ang_vel": 0.25, "projected_gravity": 1.0,
    "velocity_commands": 1.0, "joint_pos": 1.0,
    "joint_vel": 0.05, "actions": 1.0,
}


def sha256(path):
    """读取资产摘要，拒绝把其他机器人的策略误用于 Race。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(policy_path, cfg):
    """核对导出合同与本机模型、姿态及控制参数。"""
    manifest_path = policy_path.with_name("deployment.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"缺少导出合同 {manifest_path}；请用 Race 的 play.py "
            "重新导出 policy.pt 与 deployment.json。"
        )
    contract = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1, "robot": "race", "num_single_obs": 45,
        "history_length": 10, "action_scale": 0.25,
        "history_initialization": "repeat_first_frame",
        "action_clip": [-100.0, 100.0],
        "observation_clip": [-100.0, 100.0], "policy_dt": 0.02,
        "pd_dt": float(cfg["pd_dt"]),
        "physics_dt": float(cfg["physics_dt"]),
        "base_init_pos": cfg["base_init_pos"],
        "base_init_quat": cfg["base_init_quat"],
        "joint_names": cfg["joint_names"],
        "observation_order": OBSERVATION_ORDER,
        "observation_scales": OBSERVATION_SCALES,
    }
    for name in (
        "default_angles", "kps", "kds", "torque_limits",
        "passive_damping", "passive_friction", "armature",
    ):
        required[name] = cfg[name]
    urdf_path = Path(cfg["urdf_path"].replace("{ROOT_DIR}", str(ROOT)))
    model_path = Path(cfg["model_path"].replace("{ROOT_DIR}", str(ROOT)))
    required["urdf_sha256"] = sha256(urdf_path)
    required["mujoco_sha256"] = sha256(model_path)
    for name, expected in required.items():
        actual = contract.get(name)
        if isinstance(expected, (int, float)):
            matches = isinstance(actual, (int, float)) and np.isclose(
                actual, expected, rtol=1e-7, atol=1e-9
            )
        elif isinstance(expected, list) and expected and isinstance(
            expected[0], (int, float)
        ):
            try:
                matches = np.asarray(actual).shape == (len(expected),) and (
                    np.allclose(actual, expected, rtol=1e-7, atol=1e-9)
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"导出合同不一致：{name}，导出值={actual}，本机值={expected}"
            )
    if "policy_sha256" in contract:
        if contract["policy_sha256"] != sha256(policy_path):
            raise ValueError("policy.pt 摘要与导出合同不一致")
    return contract


def build_observation(model, data, base_id, qpos, qvel, nominal, action,
                      command):
    """按训练顺序构建 45 维观测，角速度与重力都使用机身坐标系。"""
    rotation = data.xmat[base_id].reshape(3, 3)
    world_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_id,
        world_velocity, 0,
    )
    # 先显式读取世界角速度，再转到 URDF 的机身轴，避免惯性主轴混淆。
    body_angular_velocity = rotation.T @ world_velocity[:3]
    projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    terms = [
        body_angular_velocity, projected_gravity, command,
        data.qpos[qpos] - nominal, data.qvel[qvel], action,
    ]
    # 与 IsaacLab 一致：各项先裁剪原始观测，再乘缩放系数。
    observation = np.concatenate([
        np.clip(term, -100.0, 100.0) * OBSERVATION_SCALES[name]
        for name, term in zip(OBSERVATION_ORDER, terms)
    ]).astype(np.float32)
    if observation.shape != (45,) or not np.isfinite(observation).all():
        raise RuntimeError("Race 策略观测维度错误或出现非有限数值")
    return observation


def ground_contacts(model, data, foot_ids):
    """统计四脚与地面的接触载荷，脚序为前左、前右、后左、后右。"""
    ground = model.geom("ground").id
    loads = np.zeros(4)
    other_contacts = 0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if contact.geom1 == ground:
            robot_geom = contact.geom2
        elif contact.geom2 == ground:
            robot_geom = contact.geom1
        else:
            continue
        body = int(model.geom_bodyid[robot_geom])
        if body in foot_ids:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, force)
            loads[foot_ids.index(body)] += max(0.0, force[0])
        else:
            other_contacts += 1
    return loads, other_contacts


def save_render(model, data, base_id, output):
    """生成闭环结束时的真实仿真截图。"""
    import imageio.v3 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = base_id
    camera.distance = 1.8
    camera.azimuth = 135.0
    camera.elevation = -20.0
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0
    option.geomgroup[2] = 1
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        imageio.imwrite(output, renderer.render())


def run(args):
    """运行有界策略闭环；短训练检查点只用于验证整个部署通路。"""
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"找不到 TorchScript 策略：{policy_path}")
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("--duration 必须为有限正数")
    command = np.asarray(args.command, dtype=np.float32)
    if not np.isfinite(command).all():
        raise ValueError("速度命令必须包含三个有限数值")
    robot = load_robot(args.config)
    model, data, cfg, qpos, qvel, actuators = robot
    contract = verify_manifest(policy_path, cfg)
    nominal = np.asarray(cfg["default_angles"], dtype=np.float64).copy()
    joints = [model.joint(name).id for name in cfg["joint_names"]]
    limits = model.jnt_range[joints]
    exported_limits = np.asarray(contract.get("target_joint_limits"))
    if exported_limits.shape != (12, 2) or not np.allclose(
        exported_limits, limits, rtol=1e-7, atol=1e-8
    ):
        raise ValueError("导出的目标角限位与 Race URDF 限位不一致")
    base_id = model.body(cfg["base_name"]).id
    foot_names = [
        "pitch03_" + name.removeprefix("roll_").replace("_joint", "_link")
        for name in cfg["joint_names"] if name.startswith("roll_")
    ]
    foot_ids = [model.body(name).id for name in foot_names]
    # 单机器人推理限制 CPU 线程，避免小批量网络的线程调度开销。
    torch.set_num_threads(1)
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    if not hasattr(policy, "reset"):
        raise ValueError("导出策略缺少 reset()，需要 Race CTS 单帧导出模型")
    policy.reset()
    for name, expected in (
        ("num_actions", 12), ("num_single_obs", 45),
        ("num_actor_obs", 450), ("history_len", 10),
    ):
        if not hasattr(policy, name) or int(getattr(policy, name)) != expected:
            raise ValueError(f"TorchScript 策略属性 {name} 必须为 {expected}")
    # IsaacLab 的历史缓冲在首次采样时重复填入首帧；按同样规则初始化。
    initialization = contract.get("history_initialization", "repeat_first_frame")
    if initialization != "repeat_first_frame":
        raise ValueError("Race 历史初始化方式必须与 IsaacLab 首帧重复规则一致")
    initial_observation = build_observation(
        model, data, base_id, qpos, qvel, nominal,
        np.zeros(12, dtype=np.float32), command,
    )
    with torch.inference_mode():
        initial_tensor = torch.from_numpy(initial_observation).unsqueeze(0)
        for _ in range(int(contract["history_length"])):
            policy(initial_tensor)
    policy_dt = float(contract["policy_dt"])
    pd_dt = float(cfg["pd_dt"])
    ratio = policy_dt / pd_dt
    decimation = round(ratio)
    if decimation < 1 or not np.isclose(ratio, decimation):
        raise ValueError("PD 周期必须整除策略周期")
    target_steps = math.ceil(args.duration / policy_dt)
    action = np.zeros(12, dtype=np.float32)
    max_action = 0.0
    max_observation = 0.0
    max_tilt = 0.0
    minimum_height = float(data.xpos[base_id, 2])
    load_samples = []
    other_contact_steps = 0
    policy_steps = 0
    clipped_target_steps = 0
    start = time.monotonic()
    viewer_context = nullcontext(None) if args.headless else (
        mujoco.viewer.launch_passive(model, data)
    )
    print(f"Race 策略：{policy_path}")
    print("策略 20 ms / PD 5 ms / 积分 1 ms；12 动作，45 维当前帧。")
    with viewer_context as viewer, torch.inference_mode():
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[2] = 1
        for index in range(target_steps):
            if viewer is not None and not viewer.is_running():
                break
            observation = build_observation(
                model, data, base_id, qpos, qvel, nominal, action, command,
            )
            result = policy(torch.from_numpy(observation).unsqueeze(0))
            raw_action = result.detach().cpu().numpy()
            if raw_action.shape != (1, 12) or not np.isfinite(raw_action).all():
                raise RuntimeError("策略必须输出有限的 [1, 12] 动作张量")
            action = np.clip(raw_action[0], -100.0, 100.0)
            target = nominal + float(contract["action_scale"]) * action
            clipped_target = np.clip(target, limits[:, 0], limits[:, 1])
            clipped_target_steps += int(not np.array_equal(target, clipped_target))
            cfg["default_angles"] = clipped_target
            max_action = max(max_action, float(np.max(np.abs(action))))
            max_observation = max(
                max_observation, float(np.max(np.abs(observation)))
            )
            for _ in range(decimation):
                step_pd(model, data, cfg, qpos, qvel, actuators)
            if not all(np.isfinite(value).all() for value in (
                data.qpos, data.qvel, data.ctrl,
            )):
                raise RuntimeError("MuJoCo 状态或 PD 力矩出现非有限数值")
            # 更新所有派生量，确保截图、接触及观测来自当前状态。
            mujoco.mj_forward(model, data)
            rotation = data.xmat[base_id].reshape(3, 3)
            tilt = math.degrees(math.acos(np.clip(rotation[2, 2], -1.0, 1.0)))
            max_tilt = max(max_tilt, tilt)
            minimum_height = min(minimum_height, float(data.xpos[base_id, 2]))
            loads, other_contacts = ground_contacts(model, data, foot_ids)
            load_samples.append(loads)
            other_contact_steps += int(other_contacts > 0)
            policy_steps += 1
            if viewer is not None:
                viewer.sync()
                delay = start + (index + 1) * policy_dt - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    final_loads, final_other_contacts = ground_contacts(model, data, foot_ids)
    report = {
        "robot": "race", "task": contract.get("task"),
        "policy_path": str(policy_path), "policy_sha256": sha256(policy_path),
        "simulation_seconds": float(data.time), "policy_steps": policy_steps,
        "completed_requested_duration": policy_steps == target_steps,
        "wall_seconds": time.monotonic() - start,
        "command": command.tolist(), "joint_names": cfg["joint_names"],
        "history_initialization": initialization,
        "all_finite": True, "max_abs_observation": max_observation,
        "max_abs_action": max_action, "clipped_target_steps": clipped_target_steps,
        "minimum_base_height_m": minimum_height,
        "final_base_height_m": float(data.xpos[base_id, 2]),
        "max_base_tilt_deg": max_tilt,
        "final_base_tilt_deg": tilt,
        "final_foot_loads_n": final_loads.tolist(),
        "mean_foot_loads_n": np.mean(load_samples, axis=0).tolist(),
        "final_contact_foot_count": int(np.count_nonzero(final_loads > 1.0)),
        "nonfoot_ground_contact_steps": other_contact_steps,
        "final_nonfoot_ground_contacts": final_other_contacts,
        "note": "本报告验证策略闭环与数值有效性，不评定步态收敛。",
    }
    if args.render_output:
        save_render(model, data, base_id, args.render_output)
        report["render_output"] = str(args.render_output.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True,
                        help="Race 导出的 policy.pt，旁边须有 deployment.json")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="仿真持续秒数，默认 30 秒")
    parser.add_argument("--headless", action="store_true", help="不打开窗口")
    parser.add_argument("--command", type=float, nargs=3,
                        default=[0.2, 0.0, 0.0], metavar=("VX", "VY", "YAW"),
                        help="机身前向速度、横向速度和偏航角速度命令")
    parser.add_argument("--config", type=Path,
                        help="Race 控制配置，默认 configs/race.yaml")
    parser.add_argument("--report", type=Path, help="保存闭环测试 JSON 报告")
    parser.add_argument("--render-output", type=Path, help="保存结束状态 PNG 截图")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
