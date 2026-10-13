# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys

# Preload native extensions before isaacsim/Kit modules are imported to avoid
# Windows DLL loader conflicts when Isaac Lab/RSL-RL import them later.
import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from utils import export_cts_policy_as_jit, export_cts_policy_as_onnx

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=int(1e9), help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="RobotLab-Race-v0", help="播放任务，默认 Race。"
)
parser.add_argument(
    "--steps", type=int, default=0, help="播放步数，0 表示运行到关闭窗口。"
)
parser.add_argument(
    "--check_export", action="store_true",
    help="逐步核对导出 TorchScript 与 IsaacLab 策略输出，仅用于 Race。",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--fix_commands", action="store_true", default=False, help="Fix the velocity commands.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.keyboard and args_cli.headless:
    parser.error("键盘回放需要本地窗口，请去掉 --headless。")
if args_cli.keyboard and args_cli.fix_commands:
    parser.error("--keyboard 与 --fix_commands 不能同时使用。")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
# from scripts.reinforcement_learning.utils import camera_follow
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerCTS

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import robot_lab.tasks  # noqa: F401

def fix_commands(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Fix commanded velocity to a constant target.

    Prefer locking the command generator to keep policy observations consistent
    with environment internal command state.
    """
    fixed_lin_x, fixed_lin_y, fixed_ang_z = 1.0, 0.0, 0.0

    base_velocity_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity_cfg is None:
        return

    fixed_cfg = UniformVelocityCommandCfg(
        asset_name=getattr(base_velocity_cfg, "asset_name", "robot"),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        resampling_time_range=(5.0, 5.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(fixed_lin_x, fixed_lin_x),
            lin_vel_y=(fixed_lin_y, fixed_lin_y),
            ang_vel_z=(fixed_ang_z, fixed_ang_z),
            heading=None,
        ),
        debug_vis=True,
    )
    env_cfg.commands.base_velocity = fixed_cfg

    # terrain_levels_vel_gym expects custom Go2RLGymCommand fields.
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.task.startswith("RobotLab-Race"):
        # 验证 Race 时使用名义机器人参数与默认姿态，便于对照部署。
        env_cfg.observations.single_obs.enable_corruption = False
        for name in vars(env_cfg.events):
            if not name.startswith("_") and name.startswith("randomize_"):
                setattr(env_cfg.events, name, None)
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        env_cfg.events.reset_base.params["pose_range"] = {}
        env_cfg.events.reset_base.params["velocity_range"] = {}
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.curriculum.terrain_levels = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        # resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        # if not resume_path:
        #     print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
        #     return
        raise NotImplementedError("Pre-trained checkpoint retrieval is disabled temporarily.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.task.startswith("RobotLab-Race"):
        from race_checkpoint import verify_race_checkpoint
        verify_race_checkpoint(resume_path, env_cfg)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # fix velocity commands if specified
    if args_cli.fix_commands:
        fix_commands(env_cfg)

    keyboard_ranges = None
    if args_cli.keyboard:
        import copy

        keyboard_ranges = copy.deepcopy(env_cfg.commands.base_velocity.ranges)
        # 启动和复位均从零命令开始，随后由键盘接管。
        env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot", heading_command=False,
            rel_standing_envs=0.0, rel_heading_envs=0.0,
            resampling_time_range=(1.0e9, 1.0e9),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0), heading=None,
            ),
            debug_vis=True,
        )
        env_cfg.curriculum.terrain_levels = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        import imageio
        video_path = os.path.join(log_dir, "videos", "play", time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")
        writer = imageio.get_writer(video_path, fps=int(1/env.unwrapped.step_dt))
        # video_kwargs = {
        #     "video_folder": os.path.join(log_dir, "videos", "play"),
        #     "step_trigger": lambda step: step == 0,
        #     "video_length": args_cli.video_length,
        #     "disable_logger": True,
        # }
        # print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        # env = gym.wrappers.RecordVideo(env, **video_kwargs)  # Store all frames into list is slow and memory-consuming, use imageio

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    keyboard = None
    if args_cli.keyboard:
        from keyboard_control import attach_keyboard

        keyboard = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=0.5, v_y_sensitivity=0.25,
            omega_z_sensitivity=0.5, sim_device=env.unwrapped.device,
        ))
        attach_keyboard(env.unwrapped, keyboard, keyboard_ranges)
        print("[INFO] 点击仿真窗口后按住方向键移动，Z/X 转向；"
              "松键停止，L 清零。速度限制沿用训练配置。")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerCTS":
        runner = OnPolicyRunnerCTS(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if agent_cfg.class_name == "OnPolicyRunnerCTS":
        export_cts_policy_as_jit(policy_nn, actor_obs_normalizer=policy_nn.actor_obs_normalizer, single_obs_normalizer=policy_nn.single_obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_cts_policy_as_onnx(policy_nn, actor_obs_normalizer=policy_nn.actor_obs_normalizer, single_obs_normalizer=policy_nn.single_obs_normalizer, path=export_model_dir, filename="policy.onnx")
        if args_cli.task.startswith("RobotLab-Race"):
            from race_export import write_deployment_manifest
            write_deployment_manifest(
                env.unwrapped, policy_nn, export_model_dir,
                task=args_cli.task, checkpoint=resume_path,
            )
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    exported_policy = None
    export_needs_reset = True
    export_max_error = 0.0
    if args_cli.check_export:
        if not args_cli.task.startswith("RobotLab-Race"):
            raise ValueError("--check_export 当前用于 Race CTS 的完整流程核对")
        torch.set_num_threads(1)
        exported_policy = torch.jit.load(
            os.path.join(export_model_dir, "policy.pt"), map_location="cpu"
        ).eval()

    # env.unwrapped.eye = (1.1, 3.3, 0.9)
    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running() and (
        args_cli.steps <= 0 or timestep < args_cli.steps
    ):
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if exported_policy is not None:
                single = obs["single_obs"][0:1].cpu()
                if export_needs_reset:
                    exported_policy.reset()
                    for _ in range(exported_policy.history_len - 1):
                        exported_policy(single)
                exported_action = exported_policy(single)
                error = (exported_action - actions[0:1].cpu()).abs().max().item()
                export_max_error = max(export_max_error, error)
                if not torch.isfinite(exported_action).all() or error > 1e-5:
                    raise RuntimeError(f"导出策略与训练推理不一致，最大差值 {error}")
            # env stepping
            obs, _, dones, _ = env.step(actions)
            export_needs_reset = bool(dones[0].item())
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            writer.append_data(env.env.render())
        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        # camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    print(f"[INFO] 已完成 {timestep} 步策略播放；策略导出至 {export_model_dir}")
    if exported_policy is not None:
        result_path = os.path.join(export_model_dir, "export-check.json")
        with open(result_path, "w", encoding="utf-8") as stream:
            json.dump({"passed": True, "steps": timestep,
                       "maximum_action_difference": export_max_error}, stream,
                      ensure_ascii=False, indent=2)
        print(f"[INFO] {timestep} 步导出一致性核对通过，最大差值 {export_max_error}")

    # close the video writer
    if args_cli.video:
        writer.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
