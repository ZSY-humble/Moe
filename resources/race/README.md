# Race 0710 模型与纯 PD 查看

模型复制自 `/home/zsy/project/SLR-master/resources/race`，复制时逐文件
核对 SHA-256，记录在 `import_manifest.json`。现已接入本项目训练任务，
训练、恢复、回放与策略部署见 [TRAINING.md](TRAINING.md)。

- `urdf/race.urdf`：来源项目当前使用的 26 基本碰撞体模型。
- `meshes/`：原始外观网格，保留大小写文件名与相对引用。
- `urdf/22nm_dog_0710.urdf`、`mjcf/mjmodel_train.xml`：原始对照模型。
- `mjcf/race_pd.xml`：从当前 `race.urdf` 转换的平地 MuJoCo 查看模型。
  保留质量、完整惯量、关节坐标、转轴、限位、碰撞体及自碰撞设置。
  地面棋盘纹理、背景、灯光和显示设置直接复用 `resources/go2/flat.xml`。
- `SOURCE_README.md`：来源项目的说明和历史记录快照。
- `control_contract.json`：旧发布合同的来源快照，当前 PD 不采用其中旧值。

PD、力矩限幅、被动项与转子惯量来自来源项目的
`configs/race_config.py`。本地 `deploy/deploy_mujoco/configs/race.yaml`
现已使用优化的初始目标角；原参数快照保存为 `race_baseline.yaml`：

- 每腿 Kp：`[19, 30, 32]`；Kd：`[0.80, 0.85, 0.70]`。
- 推荐前腿目标角：`[0, -0.17, 0.54] rad`；后腿：`[0, -0.08, 0.40] rad`。
  每腿顺序为 `roll / pitch01 / pitch02`，左右相同。
- 原始前腿：`[0, -0.1, 0.3] rad`；后腿：`[0, 0, 0.3] rad`。
- 初始基座：`[0, 0, 0.36] m`，正立，初速度为零。
- 电机限幅：`[22, 22, 33] N·m`，在 PD 计算后、被动项叠加前执行。
- 保留原被动阻尼、被动摩擦与转子惯量，避免重复添加关节阻尼。
- PD 周期 5 ms；MuJoCo 内部积分 1 ms，每个 PD 周期内保持力矩。

在项目根目录运行：

```bash
conda activate go2_rl_robotlab
python deploy/deploy_mujoco/stand_race.py
python deploy/deploy_mujoco/stand_race.py --collision
xdg-open resources/race/race_collision.html
```

对照原始姿态：

```bash
python deploy/deploy_mujoco/stand_race.py \
  --config deploy/deploy_mujoco/configs/race_baseline.yaml
```

窗口关闭后退出。纯 PD 使用固定目标角、自由基座和原重力，不依赖策略权重。
HTML 可切换原始与推荐姿态、初始目标姿态与 PD 落地 20 秒后的实际状态，
以及原 URDF 与转换后的 MuJoCo 模型。切换姿态只改变刚体位置和旋转。
HTML 用于检查几何、位置与尺寸，实际接触由 MuJoCo 物理引擎计算。
MuJoCo 与来源项目 PhysX 的接触求解器不同，运行结果不保证逐帧一致。

## 初始姿态的短时实测

按训练初始化用途，在原模型、原 PD 和同一平地中测试 20 秒，统计末
2 秒的平均载荷。672 组全局搜索候选中，筛出四脚承重接近、后脚俯仰
小于 15 度、每个足底几何带都不缩小的候选，再复测 16 组取整关节角。
推荐姿态及三组 ±0.02 rad 初始角扰动均通过四脚支撑检查。

- 左前 / 右前 / 左后 / 右后：原始 `44.8 / 43.8 / 30.6 / 29.6 N`，
  推荐 `39.1 / 38.1 / 37.2 / 36.3 N`。
- 四脚平均载荷的变异系数：`19.1% → 2.7%`，越小表示承重越均衡。
- 落地后后脚俯仰：约 `18° → 8°`。
- 四脚 CAD 底部 1 mm 带投影面积均增加，后脚分别增加约 `4.5% / 17.4%`。
- 原有 26 个碰撞体、足端 23.5 mm 球半径、质量惯量和全部控制参数保留。

足端原 CAD 是曲面，因此用“最低点以上 1 mm 带的水平投影凸包面积”
比较姿态下的足底几何范围。该指标不依赖穿地深度，不是物理压力面积；
现有球形碰撞仍采用点接触。

完整结果：`logs/race-stance-selected/selection.json`；对照图：
`logs/race-stance-selected/stance-comparison.png`。
模型查看与 Race 训练均按同一份 YAML 中的 `joint_names` 与
`default_angles` 设置初始化和动作参考姿态。

需要重新生成查看模型和页面时：

```bash
python scripts/tools/build_race_pd_model.py
python scripts/tools/export_race_collision.py
```

重做姿态搜索时，使用新的日志目录，保持原控制参数作为基线：

```bash
python scripts/tools/optimize_race_stance.py \
  --output logs/race-new-search --iterations 20
python scripts/tools/select_race_stance.py \
  --search logs/race-new-search --output logs/race-new-selection
```

筛选脚本输出 `race_optimized.yaml` 和 `selection.json`，供复核后同步默认角。
