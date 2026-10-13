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

# 独立对照版本：环境与动作管理器直接复用原 Go2。
gym.register(
    id="RobotLab-Race-Go2Reference-v0",
    entry_point="robot_lab.tasks.go2.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_reference_cfg:RaceGo2ReferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_reference_cfg:RaceGo2ReferenceRunnerCfg",
    },
)
