"""在 MuJoCo 中加载本项目训练并导出的 Race CTS 策略。"""

import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
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


class KeyboardCommands:
    """按住方向键运动，松开停止；Z/X 调整平移速度档位。"""

    def __init__(self, initial, key_source=None):
        from keyboard_state import WindowKeys

        self.keys = key_source if key_source is not None else WindowKeys()
        self.speed = 1.0
        self.command = np.zeros(3, dtype=np.float32)
        self.events = deque(maxlen=128)
        self.lock = Lock()
        self.change_count = 0

    def key_callback(self, keycode):
        """窗口线程只排队速度档位事件。"""
        if ord("a") <= keycode <= ord("z"):
            keycode -= ord("a") - ord("A")
        if keycode in (ord("Z"), ord("X")):
            with self.lock:
                self.events.append(keycode)

    def read(self):
        with self.lock:
            pending = list(self.events)
            self.events.clear()
        for keycode in pending:
            delta = -0.1 if keycode == ord("Z") else 0.1
            self.speed = round(max(0.0, self.speed + delta), 1)
            print(f"\n平移速度档位：{self.speed:.1f} m/s", flush=True)
        pressed = self.keys.read()
        before = self.command.copy()
        self.command[:] = 0.0
        if " " not in pressed:
            direction = np.array([
                int("W" in pressed) - int("S" in pressed),
                int("A" in pressed) - int("D" in pressed),
            ], dtype=np.float32)
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.command[:2] = direction * self.speed / norm
            self.command[2] = 0.6 * (
                int("Q" in pressed) - int("E" in pressed)
            )
        self.change_count += int(not np.array_equal(before, self.command))
        return self.command.copy()


def show_keyboard_status(viewer, model, data, base_id, command):
    """在窗口及终端显示命令和实际机身速度，不能把命令当成测量值。"""
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, velocity, 0,
    )
    rotation = data.xmat[base_id].reshape(3, 3)
    body_velocity = rotation.T @ velocity[3:]
    angular_velocity = rotation.T @ velocity[:3]
    actual = np.array([body_velocity[0], body_velocity[1], angular_velocity[2]])
    if hasattr(viewer, "set_texts"):
        viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Race policy + PD\nW/S: vx  A/D: vy  Q/E: yaw\n"
            "Z/X: speed -/+ 0.1 m/s; SPACE: stop\nCommand vx / vy / yaw\n"
            "Measured vx / vy / yaw\nPosition x / y / z",
            "\nHold WASD to move; release to stop\n"
            "Click viewport to focus\n"
            + " / ".join(f"{value:+.2f}" for value in command) + "\n"
            + " / ".join(f"{value:+.2f}" for value in actual) + "\n"
            + " / ".join(f"{value:+.2f}" for value in data.xpos[base_id]),
        ))
    print(
        f"\r速度命令 vx={command[0]:+.2f} vy={command[1]:+.2f} "
        f"yaw={command[2]:+.2f}；实际 vx={actual[0]:+.2f} "
        f"vy={actual[1]:+.2f} yaw={actual[2]:+.2f}；位置 "
        f"x={data.xpos[base_id, 0]:+.2f} y={data.xpos[base_id, 1]:+.2f} "
        f"z={data.xpos[base_id, 2]:+.2f}",
        end="", flush=True,
    )


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
    """运行固定命令或键盘策略闭环；不改动模型和原始 PD 控制。"""
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"找不到 TorchScript 策略：{policy_path}")
    keyboard = bool(getattr(args, "keyboard", False))
    if keyboard and args.headless:
        raise ValueError("--keyboard 需要 MuJoCo 窗口，不能与 --headless 同用")
    unlimited = keyboard and not args.headless and args.duration == 0
    if not np.isfinite(args.duration) or (args.duration <= 0 and not unlimited):
        raise ValueError("--duration 必须为正数；键盘窗口可设为 0 运行至关闭")
    initial_command = args.command
    if initial_command is None:
        initial_command = [0.0, 0.0, 0.0] if keyboard else [0.2, 0.0, 0.0]
    command = np.asarray(initial_command, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError("速度命令必须是 vx、vy、yaw 三个数值")
    if not np.isfinite(command).all():
        raise ValueError("速度命令必须包含三个有限数值")
    controls = KeyboardCommands(command) if keyboard else None
    if controls is not None:
        command = controls.read()
    initial_command = command.copy()
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
    initial_position = data.xpos[base_id].copy()
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
    target_steps = None if unlimited else math.ceil(args.duration / policy_dt)
    action = np.zeros(12, dtype=np.float32)
    max_action = 0.0
    max_observation = 0.0
    max_tilt = 0.0
    tilt = math.degrees(math.acos(np.clip(data.xmat[base_id, 8], -1.0, 1.0)))
    minimum_height = float(data.xpos[base_id, 2])
    # 累计求和而非逐帧留存，长时间键盘运行保持恒定内存。
    load_sum = np.zeros(4)
    other_contact_steps = 0
    policy_steps = 0
    clipped_target_steps = 0
    start = time.monotonic()
    viewer_context = nullcontext(None) if args.headless else (
        mujoco.viewer.launch_passive(
            model, data,
            key_callback=controls.key_callback if controls else None,
        )
    )
    print(f"Race 策略：{policy_path}")
    print("策略 20 ms / PD 5 ms / 积分 1 ms；12 动作，45 维当前帧。")
    if keyboard:
        print("先点击 MuJoCo 画面获得键盘焦点。")
        print("按住 W/S 前后移动，A/D 左右移动；默认平移速度 1.0 m/s。")
        print("Z/X 每次减速/加速 0.1 m/s；按住 Q/E 转向 0.6 rad/s。")
        print("松开方向键停止；空格停止；切换窗口停止；关闭窗口退出。", flush=True)
    with viewer_context as viewer, torch.inference_mode():
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = base_id
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            viewer.opt.geomgroup[1] = 0
            viewer.opt.geomgroup[2] = 1
        deadline = time.monotonic()
        next_status = deadline
        while target_steps is None or policy_steps < target_steps:
            if viewer is not None and not viewer.is_running():
                break
            if controls is not None:
                command = controls.read()
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
            load_sum += loads
            other_contact_steps += int(other_contacts > 0)
            policy_steps += 1
            if viewer is not None:
                if keyboard and time.monotonic() >= next_status:
                    show_keyboard_status(viewer, model, data, base_id, command)
                    next_status = time.monotonic() + 1.0
                viewer.sync()
                deadline += policy_dt
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:
                    # 窗口阻塞后重置实时基准，避免持续高速追赶造成操控失真。
                    deadline = time.monotonic()
    if keyboard:
        print()
    final_loads, final_other_contacts = ground_contacts(model, data, foot_ids)
    report = {
        "robot": "race", "task": contract.get("task"),
        "policy_path": str(policy_path), "policy_sha256": sha256(policy_path),
        "simulation_seconds": float(data.time), "policy_steps": policy_steps,
        "requested_duration_seconds": float(args.duration),
        "completed_requested_duration": (
            target_steps is not None and policy_steps == target_steps
        ),
        "wall_seconds": time.monotonic() - start,
        "command": command.tolist(), "initial_command": initial_command.tolist(),
        "command_mode": "keyboard" if keyboard else "fixed",
        "keyboard_command_changes": controls.change_count if controls else 0,
        "joint_names": cfg["joint_names"],
        "initial_base_position_m": initial_position.tolist(),
        "final_base_position_m": data.xpos[base_id].tolist(),
        "displacement_m": (data.xpos[base_id] - initial_position).tolist(),
        "history_initialization": initialization,
        "all_finite": True, "max_abs_observation": max_observation,
        "max_abs_action": max_action, "clipped_target_steps": clipped_target_steps,
        "minimum_base_height_m": minimum_height,
        "final_base_height_m": float(data.xpos[base_id, 2]),
        "max_base_tilt_deg": max_tilt,
        "final_base_tilt_deg": tilt,
        "final_foot_loads_n": final_loads.tolist(),
        "mean_foot_loads_n": (load_sum / max(policy_steps, 1)).tolist(),
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
                        help="仿真秒数，默认 30；键盘窗口设 0 可持续运行")
    parser.add_argument("--headless", action="store_true", help="不打开窗口")
    parser.add_argument("--keyboard", action="store_true",
                        help="按住 WASD 移动，默认 1 m/s；Z/X 调速；松键停止")
    parser.add_argument("--command", type=float, nargs=3,
                        default=None, metavar=("VX", "VY", "YAW"),
                        help="机身速度命令；键盘默认全零，固定默认 0.2 0 0")
    parser.add_argument("--config", type=Path,
                        help="Race 控制配置，默认 configs/race.yaml")
    parser.add_argument("--report", type=Path, help="保存闭环测试 JSON 报告")
    parser.add_argument("--render-output", type=Path, help="保存结束状态 PNG 截图")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
