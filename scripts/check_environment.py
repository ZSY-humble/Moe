"""检查依赖、CUDA 和预训练 Go2 策略的 MuJoCo 闭环运行。"""

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
# 避免 scripts/rsl_rl 被识别为同名包，遮蔽可编辑安装的定制库。
if Path(sys.path[0]).resolve() == ROOT / "scripts":
    sys.path[0] = str(ROOT)


def check_packages():
    """检查 README 的关键版本及可编辑安装的源码路径。"""
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("本项目需要 Python 3.11")
    expected = {
        "isaaclab": "2.3.2.post1",
        "isaacsim": "5.1.0.0",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "tensordict": "0.8.3",
        "robot_lab": "2.3.0",
        "rsl-rl-lib": "3.3.0",
        "mujoco": "3.4.0",
        "pygame": "2.6.1",
    }
    versions = {}
    for package, version in expected.items():
        actual = metadata.version(package)
        if actual.split("+")[0] != version:
            raise RuntimeError(f"{package}: 期望 {version}，实际 {actual}")
        versions[package] = actual

    import h5py
    import tensordict
    import torchvision
    from rsl_rl.runners import OnPolicyRunnerCTS
    import rsl_rl

    source = Path(rsl_rl.__file__).resolve()
    if not source.is_relative_to(ROOT / "source" / "rsl_rl"):
        raise RuntimeError(f"RSL-RL 未使用仓库定制源码: {source}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stdout.strip() or result.stderr.strip())
    versions["pip_check"] = result.stdout.strip()
    return versions


def check_cuda():
    """实际运行 GPU 前向、反向计算，确认 CUDA 可用。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查主机驱动和 GPU 访问权限")
    model = torch.nn.Sequential(
        torch.nn.Linear(45, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 12),
    ).cuda()
    inputs = torch.randn(64, 45, device="cuda")
    loss = model(inputs).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("GPU 计算出现非有限数值")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "loss": loss.item(),
    }


def check_mujoco(steps, render):
    """复用部署代码，运行模型加载、策略推理和 PD 控制闭环。"""
    if render:
        # 只有显式请求截图才加载图形后端，允许调用方选择 OSMesa 等后端。
        os.environ.setdefault("MUJOCO_GL", "egl")
        backend = os.environ["MUJOCO_GL"].strip().lower()
        if backend in {"disable", "disabled", "off", "false", "0"}:
            raise RuntimeError(
                "--render 需要图形后端。请设置 MUJOCO_GL=egl 或 osmesa；"
                "仅验证物理仿真时请移除 --render。"
            )
    else:
        # 纯物理检查不需要 EGL；也覆盖 shell 中遗留的 EGL 后端设置。
        os.environ["MUJOCO_GL"] = "disable"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    sys.path.insert(0, str(ROOT / "deploy" / "deploy_mujoco"))
    import mujoco
    import numpy as np
    import torch
    from deploy_go2 import (
        ACTUATOR_GROUPS,
        action_to_target,
        build_features,
        build_single_obs,
    )
    from utils import (
        build_delay_buffers,
        load_config,
        pd_control,
        sample_delayed_targets,
        set_initial_state,
    )

    cfg = load_config("go2.yaml")
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.dt
    set_initial_state(
        data, cfg.base_init_pos, cfg.base_init_quat, cfg.default_angles
    )
    mujoco.mj_forward(model, data)
    policy = torch.jit.load(str(cfg.policy_path), map_location="cpu")
    policy.eval()
    layout = [
        ("ang_vel", 3), ("gravity", 3), ("cmd", 3),
        ("joint_pos", 12), ("joint_vel", 12), ("last_action", 12),
    ]
    action = np.zeros(cfg.num_actions, dtype=np.float32)
    target = cfg.default_angles.copy()
    history = build_delay_buffers(target, delay_max=cfg.delay_max)
    rng = np.random.default_rng(cfg.delay_seed)
    start_position = data.qpos[:3].copy()
    min_height = float(data.qpos[2])
    with torch.inference_mode():
        for step in range(1, steps + 1):
            data.ctrl[:] = pd_control(
                target, data.qpos[7:], cfg.kps,
                np.zeros(cfg.num_actions), data.qvel[6:], cfg.kds,
            )
            mujoco.mj_step(model, data)
            if step % cfg.decimation == 0:
                features = build_features(data, action, cfg.cmd_init, cfg)
                observation = build_single_obs(features, layout)
                output = policy(torch.from_numpy(observation).unsqueeze(0))
                if output.shape != (1, cfg.num_actions):
                    raise RuntimeError(f"策略输出维度错误: {output.shape}")
                action = output.numpy().squeeze()[cfg.idx_model2mj]
                history.append(action_to_target(action, cfg).copy())
                target = sample_delayed_targets(
                    history, ACTUATOR_GROUPS,
                    cfg.delay_min, cfg.delay_max, rng,
                )
                if not np.isfinite(action).all():
                    raise RuntimeError(f"第 {step} 步策略输出非有限数值")
            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"第 {step} 步仿真状态非有限")
            min_height = min(min_height, float(data.qpos[2]))
    result = {
        "steps": steps,
        "simulation_seconds": float(data.time),
        "policy": str(cfg.policy_path),
        "scene": str(cfg.xml_path),
        "displacement_m": (data.qpos[:3] - start_position).tolist(),
        "minimum_base_height_m": min_height,
        "render_requested": render,
        "graphics_backend": os.environ["MUJOCO_GL"],
    }
    if render:
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.qpos[:3]
            camera.distance = 3.0
            camera.azimuth = 135.0
            camera.elevation = -25.0
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            if frame.shape != (240, 320, 3) or frame.std() < 1:
                raise RuntimeError("MuJoCo 渲染图像异常")
            import imageio.v3 as imageio

            screenshot = ROOT / "logs/environment/mujoco-smoke.png"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(screenshot, frame)
            result["render"] = str(screenshot)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks", nargs="+", choices=["packages", "cuda", "mujoco"],
        default=["packages", "cuda", "mujoco"],
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--render", action="store_true",
        help="额外验证离屏截图；默认只验证不依赖 EGL 的物理与策略闭环。",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "logs/environment/smoke-results.json",
    )
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("--steps 必须至少为 10")
    checks = {
        "packages": check_packages,
        "cuda": check_cuda,
        "mujoco": lambda: check_mujoco(args.steps, args.render),
    }
    report = {"python": sys.version, "executable": sys.executable, "checks": {}}
    failed = False
    for name in args.checks:
        try:
            details = checks[name]()
            report["checks"][name] = {"passed": True, "details": details}
            print(f"[通过] {name}: {details}", flush=True)
        except Exception as error:
            failed = True
            report["checks"][name] = {"passed": False, "error": str(error)}
            print(f"[失败] {name}: {error}", flush=True)
            traceback.print_exc()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
