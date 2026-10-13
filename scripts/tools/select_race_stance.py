"""按四脚承重和足底几何约束，筛选并复测便于使用的 Race 初始姿态。"""

import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import yaml

from optimize_race_stance import Experiment, ROOT


def suitable(row, baseline):
    """四脚承重接近，后脚更平，四个足底几何带均不缩小。"""
    return (
        row["stable"] and row["load_cv"] < .05
        and max(abs(x) for x in row["foot_pitch_deg"][2:]) < 15.
        and np.all(np.asarray(row["reference_patch_mm2"])
                   >= baseline["reference_patch_mm2"])
        and np.all(np.asarray(row["near_ground_patch_mm2"])
                   >= baseline["near_ground_patch_mm2"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=(
        ROOT / "logs/race-stance-search-refined"
    ))
    parser.add_argument("--output", type=Path, default=(
        ROOT / "logs/race-stance-selected"
    ))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "deploy/deploy_mujoco/configs/race_baseline.yaml"
    config = baseline_path if baseline_path.exists() else None
    experiment = Experiment(args.output, resume=True, config=config)
    cfg = experiment.cfg
    baseline_params = [cfg["default_angles"][i] for i in (1, 2, 7, 8)]
    baseline = experiment.evaluate(baseline_params, label="baseline", full=True)
    candidates = [json.loads(line) for line in
                  (args.search / "candidates.jsonl").read_text().splitlines()]
    feasible = sorted((row for row in candidates if suitable(row, baseline)),
                      key=lambda row: row["score"])
    if not feasible:
        raise RuntimeError("已有候选中未找到同时满足足底改善和承重约束的姿态")
    rounded = set()
    # 最好的三个可行候选，各取相邻的百分之一弧度网格值。
    for row in feasible[:3]:
        options = [(np.floor(x * 100) / 100, np.ceil(x * 100) / 100)
                   for x in row["parameters"]]
        rounded.update(product(*options))
    trials = [experiment.evaluate(p, label="rounded") for p in sorted(rounded)]
    accepted = sorted((row for row in trials if suitable(row, baseline)),
                      key=lambda row: row["score"])
    if not accepted:
        raise RuntimeError("取整后的候选未通过约束，请扩大局部搜索")
    best = experiment.evaluate(accepted[0]["parameters"],
                               label="selected", full=True)
    print("选定姿态：", {key: best[key] for key in (
        "parameters", "normal_load_mean_N", "load_cv", "foot_pitch_deg",
        "reference_patch_mm2", "near_ground_patch_mm2", "maximum_tilt_deg"
    )}, flush=True)
    # 验证小幅初始角偏差；不改变碰撞体、控制目标或任何动力学参数。
    rng = np.random.default_rng(31)
    perturbations = []
    for index in range(3):
        offset = rng.uniform(-.02, .02, 12)
        row = experiment.evaluate(best["parameters"], full=True,
                                  perturb=offset, label=f"perturb_{index}")
        row["initial_joint_offset_rad"] = offset.tolist()
        perturbations.append(row)
        print(f"初始角扰动 {index}：", {key: row.get(key) for key in (
            "stable", "load_cv", "normal_load_mean_N", "maximum_tilt_deg",
            "visual_min_z_mm", "failure"
        )}, flush=True)
    (args.output / "perturbations.json").write_text(
        json.dumps(perturbations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(row["stable"] and row["load_cv"] < baseline["load_cv"]
               for row in perturbations):
        raise RuntimeError("初始角扰动后未保持支撑或承重均衡不及原始基线")
    optimized = deepcopy(cfg)
    optimized["default_angles"] = experiment.angles(
        best["parameters"]
    ).round(2).tolist()
    config_path = ROOT / "deploy/deploy_mujoco/configs/race_optimized.yaml"
    config_path.write_text(
        "# 20 秒平地初始姿态复测通过；仅调整默认角，保留原 PD 与碰撞体。\n"
        + yaml.safe_dump(optimized, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    model_path = ROOT / "resources/race/mjcf/race_pd.xml"
    report = {
        "baseline": baseline, "best": best, "perturbations": perturbations,
        "search_candidate_count": len(candidates),
        "feasible_search_count": len(feasible), "rounded_count": len(trials),
        "feasible_rounded_count": len(accepted), "config": str(config_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "criteria": "20 秒，末 2 秒统计；四脚持续支撑；载荷变异系数 <5%；"
                    "后脚俯仰 <15 度；四脚底部 1 mm 带和近地 1 mm 带均不缩小。",
        "geometry_note": "面积为 CAD 网格底部带投影凸包面积，"
                         "只用于比较姿态；球形碰撞保持原样。",
    }
    (args.output / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("短时复测及三组初始角扰动通过：", config_path, flush=True)


if __name__ == "__main__":
    main()
