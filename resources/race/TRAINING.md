# Race 训练、回放与 MuJoCo 部署

当前项目使用 Race 0710 模型训练，算法沿用原框架的 MoE CTS。
`train.py` 和 `play.py` 默认任务均为 `RobotLab-Race-v0`；原 Go2 任务
保留，可通过 `--task RobotLab-Go2-v0` 显式使用。

## 开始训练

在项目根目录运行，使用现有 Conda 环境：

```bash
# 在服务器克隆后的仓库根目录运行。
conda activate go2_rl_robotlab
export OMNI_KIT_ACCEPT_EULA=YES

python scripts/rsl_rl/train.py \
    --task RobotLab-Race-v0 --headless --device cuda:0 \
    --num_envs 256 --logger tensorboard --run_name race_train
```

默认训练 10,000 轮，每 250 轮和结束时保存检查点。256 个并行环境已
在本机 RTX 4060 Laptop 8 GB 上验证；需要为其他图形程序保留更多
显存时可使用 `--num_envs 128`。日志在 `logs/rsl_rl/race_moe_cts/`。

服务器需先核对显卡型号。当前 Isaac Sim 5.1 官方不支持没有 RT 核心的
A100/H100；PyTorch CUDA 或 MuJoCo 无渲染检查通过，不代表 IsaacLab
训练后端已验证可用。无窗口 `--headless` 也不能作为硬件兼容性的保证。
参见 [Isaac Sim 硬件要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。

先训练平地时改用 `--task RobotLab-Race-Flat-v0`，日志在
`logs/rsl_rl/race_flat_moe_cts/`。平地和粗糙地形使用相同网络接口，
可以先学平地再继续训练粗糙地形。

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

## 恢复训练

继续最近一次同任务训练：

```bash
python scripts/rsl_rl/train.py --task RobotLab-Race-v0 \
    --headless --device cuda:0 --num_envs 256 --resume
```

显式指定检查点，或从平地继续训练粗糙地形：

```bash
python scripts/rsl_rl/train.py --task RobotLab-Race-v0 \
    --headless --device cuda:0 --num_envs 256 --resume \
    --checkpoint /绝对路径/训练目录/model_1000.pt
```

保留检查点同目录的 `params/env.yaml`，加载前会核对机器人接口。
同为 12 个动作的 Go2 权重不能直接当作 Race 策略使用。
恢复时 `--max_iterations` 表示本次继续执行的轮数。

## 回放与导出

### 本地键盘回放服务器权重

将服务器本次运行目录中的 `model_600.pt` 和完整 `params/` 目录下载到
本地同一个目录，例如 `logs/server/race_600/`。其中 `params/env.yaml`
用于验证模型来源、关节顺序和控制参数，不能只复制权重。

在本地 RTX 机器上运行（需要图形桌面，不加 `--headless`）：

```bash
python scripts/rsl_rl/play.py \
    --task RobotLab-Race-Flat-v0 --num_envs 1 --device cuda:0 \
    --checkpoint logs/server/race_600/model_600.pt \
    --keyboard --real-time
```

点击仿真窗口让其获得焦点，按住 `↑/↓` 前进／后退，`←/→` 左移／
右移，`Z/X` 左转／右转；松键停止，`L` 清零。键盘速度为前后
0.5 m/s、横向 0.25 m/s、转向 0.5 rad/s，并按训练的速度范围限幅。
该功能只作用于回放实例，不修改训练奖励、控制参数或碰撞体。

回放最近一次粗糙地形训练：

```bash
python scripts/rsl_rl/play.py --task RobotLab-Race-v0 --num_envs 4
```

回放指定检查点，并进行有界的导出一致性检查：

```bash
python scripts/rsl_rl/play.py --task RobotLab-Race-v0 \
    --checkpoint /绝对路径/训练目录/model_1000.pt \
    --num_envs 4 --headless --device cuda:0 --steps 500 --check_export
```

平地检查点可用平地任务回放，也可在粗糙任务中加载进行测试。
回放关闭质量、增益、摩擦、推搡和初始状态扰动，使用名义参数。
每次回放都会在检查点目录下生成：

- `exported/policy.pt`：MuJoCo 使用的 TorchScript，输入 45 维当前帧，
  内部维护 10 帧历史，提供 `reset()`。
- `exported/policy.onnx`：输入 450 维历史，历史由调用方按观测项维护。
- `exported/deployment.json`：本次导出的关节顺序、初角、控制周期、PD、
  动作／观测缩放及模型摘要。
- `exported/export-check.json`：使用 `--check_export` 时保存逐步输出核对结果。

`--steps 0` 表示持续回放直到关闭窗口；`--video` 可录制回放，
`--video_length 500` 表示录制 500 步后结束。

## MuJoCo 策略闭环

```bash
python deploy/deploy_mujoco/deploy_race.py \
    --policy /绝对路径/训练目录/exported/policy.pt \
    --command 0.2 0 0 --duration 30
```

无窗口短时验证并保存 JSON：

```bash
python deploy/deploy_mujoco/deploy_race.py \
    --policy /绝对路径/训练目录/exported/policy.pt \
    --headless --duration 10 --command 0.2 0 0 \
    --report logs/race-training/mujoco-deployment.json
```

`--command` 依次是机身前向速度、侧向速度与偏航角速度。
部署只读取当前导出的 `deployment.json`，不会读取来源项目的旧
`resources/race/control_contract.json`。保留 `policy.pt` 和合同在同目录。

## 机器人与任务参数

- 名义参数：`deploy/deploy_mujoco/configs/race.yaml`。
- 默认前腿角：`[0, -0.17, 0.54] rad`；后腿角：`[0, -0.08, 0.40] rad`。
- 每腿依次为 `roll / pitch01 / pitch02`，策略顺序为前左、前右、后左、后右。
- 每腿 Kp：`[19, 30, 32]`；Kd：`[0.80, 0.85, 0.70]`。
- 执行器先限制电机 PD 力矩，再叠加原被动阻尼和摩擦，保留 armature。
- 策略 50 Hz，训练物理积分与 PD 更新均为 200 Hz；MuJoCo 采用
  1,000 Hz 细积分、200 Hz PD，同样以 50 Hz 更新策略。
- 动作为默认角周围 `0.25 rad × action` 的偏移，目标按原 URDF 硬限位裁剪。
- 三组观测：策略历史 450 维、价值网络 275 维、当前帧 45 维。
- 奖励保留原 Go2 已开启的 15 项，包括速度跟踪、高度、关节能耗、
  动作平滑、关节姿态与足部调节 `feet_regulation`（权重 `-0.05`）。
  已撤除迁移时额外加入的机身倾斜、足端滑动、腾空时间、静止左右对称四项。
- 初始关节角采用小幅加法扰动；训练仍启用温和的质量、增益、质心、
  接触材质和推搡随机化，名义 PD 配置不变。

Race 模型导入保留原 26 个碰撞体、4 个固定足端连杆、完整 CAD 惯量。
不合并固定足端，是为了用脚名分别获取四脚接触信息；Go2 同样有
足端接触检测和简化碰撞模型。USD 位于 `resources/race/usd/`，由 URDF
自动生成；该目录为缓存，不需要手动编辑。

环境和网络配置分别位于：

- `source/robot_lab/robot_lab/tasks/race/env_cfg.py`
- `source/robot_lab/robot_lab/tasks/race/rsl_rl_cfg.py`
- `source/robot_lab/robot_lab/assets/race.py`
- `source/robot_lab/robot_lab/assets/race_actuator.py`

## 验证范围

结构与短时零动作测试可重复运行：

```bash
python scripts/tools/check_race_training.py --headless --device cuda:0
```

结果和训练测试日志保存在 `logs/race-training/`。已验证模型导入、
短时 PD 支撑、平地／粗糙地形训练、恢复、检查点保存、策略回放、
TorchScript／ONNX 导出与 MuJoCo 闭环。短训练检查点用于证明流程
完整，不作为已收敛步态发布。当前 RoboGauge 接口仍绑定 Go2 评估模型，
Race 任务使用本地回放和 MuJoCo 验证。

## 零位与用户 PD 参数实验

当前默认控制配置改为零位实验：12 个目标角均为 0 rad，
每腿 Kp 为 `[24, 40, 32]`、Kd 为 `[0.6, 0.8, 0.7]`。
初始名义高度和高度奖励目标共同读取 `race.yaml` 的高度 `0.321 m`，
这是按零位最低足球距地 5 mm 计算后取整的释放高度，不固定浮动基座。
复位时仍保留原有高度、角度和速度随机化；目标高度不会随随机扰动改变。
原默认配置保存在 `deploy/deploy_mujoco/configs/race_pre_zero.yaml`。
模型碰撞体、惯量、电机限幅、被动项和奖励项数量保持原样。

在服务器的项目根目录、原 Conda 环境中分别从头启动：

```bash
bash scripts/tools/train_race_zero.sh flat
# 停止平地任务后启动复杂地形，两个任务避免同时占用 GPU 0。
bash scripts/tools/train_race_zero.sh rough
```

两版均为 GPU 0、4096 环境、10000 轮，使用 nohup 和无缓冲输出，
每 250 轮保存；平地权重在 `logs/rsl_rl/race_flat_moe_cts/`，
复杂地形权重在 `logs/rsl_rl/race_moe_cts/`，运行名称包含 `race_zero`。
不使用旧姿态权重续训。旧权重 MuJoCo 回放需显式指定旧配置
`--config deploy/deploy_mujoco/configs/race_pre_zero.yaml`，避免控制参数混用。

这组零位参数的 MuJoCo 60 秒测试：最终高度约 0.2335 m，
倾斜约 14.8 度；末两秒四脚接触，承重约 15.4/15.6/55.7/55.2 N，
存在非足端触地。它是用户指定的训练对比版本，不能视为稳定站姿认证。
