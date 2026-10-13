"""将 Race 原始 URDF 和由它生成的 MuJoCo 碰撞体导出为离线 HTML。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import mujoco
import yaml

from export_go2_collision import mjcf_geometry, urdf_geometry


ROOT = Path(__file__).resolve().parents[2]
LEG_NAMES = dict(zip(("FL", "FR", "RL", "RR"),
                     ("f_l", "f_r", "b_l", "b_r")))


def pose_data(cfg, config_path, label, row=None, settled=False):
    """初始姿态与落地状态分别从真实关节角计算碰撞几何。"""
    cfg = deepcopy(cfg)
    cfg["mujoco_joint_names"] = cfg["joint_names"]
    if settled:
        model = mujoco.MjModel.from_xml_path(
            str(ROOT / "resources/race/mjcf/race_pd.xml")
        )
        qpos = row["final_qpos"]
        cfg["base_init_pos"] = qpos[:3]
        cfg["base_init_quat"] = qpos[3:7]
        cfg["default_angles"] = [
            qpos[model.joint(name).qposadr[0]] for name in cfg["joint_names"]
        ]
    sources = {
        "mjcf": {
            "label": "MuJoCo / 原 URDF 转换",
            "path": "resources/race/mjcf/race_pd.xml",
            "geometries": mjcf_geometry(
                cfg, ROOT / "resources/race/mjcf/race_pd.xml"
            ),
        },
        "urdf": {
            "label": "SLR 训练 / URDF",
            "path": "resources/race/urdf/race.urdf",
            "geometries": urdf_geometry(
                cfg, ROOT / "resources/race/urdf/race.urdf"
            ),
        },
    }
    for entry in sources.values():
        if len(entry["geometries"]) != 26:
            raise ValueError("当前 Race 应有 26 个基本碰撞体")
        for geom in entry["geometries"]:
            geom["part"] = next(
                (prefix for prefix, leg in LEG_NAMES.items()
                 if f"_leg_{leg}_" in geom["link"]), "body"
            )
    result = {
        "label": label, "sources": sources, "configPath": config_path,
        "height": cfg["base_init_pos"][2], "anglePrecision": 2,
        "stateLabel": "PD 20 秒后" if settled else "初始",
        "jointCaption": "PD 20 秒后实际关节角 · rad" if settled
                        else "初始目标关节角 · rad",
        "joints": dict(zip(cfg["joint_names"], cfg["default_angles"])),
        "jointOrder": {
            prefix: [f"{kind}_leg_{leg}_joint"
                     for kind in ("roll", "pitch01", "pitch02")]
            for prefix, leg in LEG_NAMES.items()
        },
    }
    if row:
        result["metrics"] = {
            "loads": row["normal_load_mean_N"], "cv": row["load_cv"],
            "patch": row["reference_patch_mm2"],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "resources/race/race_collision.html"
    )
    parser.add_argument("--report", type=Path, default=(
        ROOT / "logs/race-stance-selected/selection.json"
    ), help="可选的 20 秒姿态实测报告，用于增加落地状态对照")
    args = parser.parse_args()
    cfg = yaml.safe_load(
        (ROOT / "deploy/deploy_mujoco/configs/race.yaml").read_text(
            encoding="utf-8"
        )
    )
    config_path = "deploy/deploy_mujoco/configs/race.yaml"
    data = pose_data(cfg, config_path, "推荐 · 初始目标姿态")
    if args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        expected = [cfg["default_angles"][i] for i in (1, 2, 7, 8)]
        if expected != report["best"]["parameters"]:
            raise ValueError("实测报告与当前默认目标角不匹配，请重新复测")
        baseline_path = "deploy/deploy_mujoco/configs/race_baseline.yaml"
        baseline_cfg = yaml.safe_load((ROOT / baseline_path).read_text())
        poses = {}
        for key, label, config, path, row in (
            ("recommended", "推荐", cfg, config_path, report["best"]),
            ("baseline", "原始", baseline_cfg, baseline_path, report["baseline"]),
        ):
            for settled in (False, True):
                suffix = "settled" if settled else "initial"
                state = "PD 20 秒后" if settled else "初始目标姿态"
                poses[f"{key}_{suffix}"] = pose_data(
                    config, path, f"{label} · {state}", row, settled
                )
        data = deepcopy(poses["recommended_initial"])
        data["poses"] = poses
    template = (ROOT / "resources/go2/collision_viewer.template.html").read_text(
        encoding="utf-8"
    )
    template = template.replace("Go2 碰撞模型", "Race 0710 碰撞模型")
    template = template.replace("宇树 Go2 ·", "Race 0710 ·")
    template = template.replace("Go2 三维碰撞模型", "Race 三维碰撞模型")
    template = template.replace("IsaacLab / URDF", "SLR 训练 / URDF")
    template = template.replace(
        "deploy/deploy_mujoco/configs/go2.yaml",
        "deploy/deploy_mujoco/configs/race.yaml",
    )
    template = template.replace("髋关节、大腿关节、小腿关节",
                                "roll、pitch01、pitch02")
    template = template.replace(
        "初始关节姿态下的真实碰撞几何。",
        "保留原有 26 个碰撞体，可对照原始与推荐姿态。",
    )
    template = template.replace(
        "这是初始姿态的几何展示；站立动力学请运行纯 PD 查看指令。",
        "可切换初始目标姿态和 PD 落地 20 秒后的实测几何；"
        "动态站立请运行纯 PD 查看指令。",
    )
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        template.replace("__GO2_MODEL_DATA__", encoded.replace("<", "\\u003c")),
        encoding="utf-8",
    )
    print(f"已生成：{args.output}；每套模型包含 26 个碰撞体")


if __name__ == "__main__":
    main()
