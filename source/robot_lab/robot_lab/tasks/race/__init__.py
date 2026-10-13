"""注册 Race 四足机器人的粗糙地形和平地训练任务。"""

import gymnasium as gym


for task_id, environment_config, runner_config in (
    ("RobotLab-Race-v0", "RaceEnvCfg", "RaceMoECTSRunnerCfg"),
    ("RobotLab-Race-Flat-v0", "RaceFlatEnvCfg", "RaceFlatMoECTSRunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="robot_lab.tasks.race.env:RaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{environment_config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:{runner_config}",
        },
    )
