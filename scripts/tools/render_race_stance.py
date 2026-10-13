"""从实测状态生成 Race 姿态、四脚承重与足底几何对照图。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mujoco
import numpy as np

from optimize_race_stance import Experiment, ROOT, clipped_patch_polygon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=(
        ROOT / "logs/race-stance-selected/selection.json"
    ))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = args.report.parent
    experiment = Experiment(output)
    model = experiment.model
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 540
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0., 0., .18]
    camera.distance = 1.18
    camera.azimuth = 95.
    camera.elevation = -12.
    option = mujoco.MjvOption()
    option.geomgroup[:3] = [1, 0, 1]
    rows = [report["baseline"], report["best"]]
    renders, footprints = [], []
    with mujoco.Renderer(model, height=540, width=960) as renderer:
        for row in rows:
            data = mujoco.MjData(model)
            data.qpos[:] = row["final_qpos"]
            data.qvel[:] = row["final_qvel"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            renders.append(renderer.render().copy())
            polygons = []
            for index, body in enumerate(experiment.bodies):
                rotation = data.xmat[body].reshape(3, 3)
                triangles = (experiment.triangles[index] @ rotation.T
                             + data.xpos[body])
                minimum = triangles[:, :, 2].min()
                hull = clipped_patch_polygon(triangles, minimum)
                # 相对脚球中心投影绘图，显示原足底与碰撞接触位置的关系。
                center = data.geom_xpos[experiment.geoms[index], :2]
                polygons.append((hull - center) * 1000)
            footprints.append(polygons)

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": font_name, "axes.unicode_minus": False,
        "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(16, 12), facecolor="#f7f9fb")
    grid = figure.add_gridspec(3, 4, height_ratios=[1.25, .8, .8],
                             hspace=.45, wspace=.4)
    colors = ["#929cab", "#147d92"]
    labels = ["原始", "推荐"]
    legs = ["左前", "右前", "左后", "右后"]
    titles = ["原始：前腿 [0, −0.10, 0.30]，后腿 [0, 0, 0.30] rad",
              "推荐：前腿 [0, −0.17, 0.54]，后腿 [0, −0.08, 0.40] rad"]
    for index, image in enumerate(renders):
        axis = figure.add_subplot(grid[0, index * 2:index * 2 + 2])
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(titles[index], fontsize=12, pad=12)
    axis = figure.add_subplot(grid[1, :2])
    x = np.arange(4)
    for index, row in enumerate(rows):
        bars = axis.bar(x + (index - .5) * .34,
                        row["normal_load_mean_N"], .34,
                        color=colors[index], label=labels[index])
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    axis.axhline(model.body_mass.sum() * 9.81 / 4, color="#bbb",
                 linestyle="--", linewidth=1)
    axis.set(xticks=x, xticklabels=legs, ylim=(0, 52), ylabel="法向载荷 / N")
    axis.set_title("四脚平均承重（18～20 秒）：变异系数 19.1% → 2.7%")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis = figure.add_subplot(grid[1, 2:])
    for index, row in enumerate(rows):
        bars = axis.bar(x + (index - .5) * .34, row["foot_pitch_deg"], .34,
                        color=colors[index], label=labels[index])
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    axis.set(xticks=x, xticklabels=legs, ylim=(0, 22), ylabel="俯仰角 / 度")
    axis.set_title("落地后脚端俯仰角：后脚约 18° → 8°")
    for index, leg in enumerate(legs):
        axis = figure.add_subplot(grid[2, index])
        for j, polygons in enumerate(footprints):
            polygon = polygons[index]
            axis.fill(polygon[:, 0], polygon[:, 1], alpha=.26,
                      facecolor=colors[j], edgecolor=colors[j], linewidth=1.6)
        axis.plot(0, 0, "+", color="#252b35", markersize=10)
        old = rows[0]["reference_patch_mm2"][index]
        new = rows[1]["reference_patch_mm2"][index]
        axis.set_title(f"{leg}底部带：{old:.1f} → {new:.1f} mm²", fontsize=11)
        axis.set(xlabel="相对脚球中心 X / mm", ylabel="Y / mm",
                 xlim=(-16, 16), ylim=(-16, 16), aspect="equal")
        axis.grid(alpha=.16)
    figure.suptitle("Race 初始姿态优化 · 同一平地场景 · 原 PD 与碰撞体",
                    fontsize=21, y=.985)
    figure.text(.5, .045,
                "底部带：原 CAD 最低点以上 1 mm 范围的水平投影凸包；"
                "“+”是球形碰撞的地面接触投影。\n"
                "球半径保持 23.5 mm；图中面积是姿态几何指标，"
                "不代表实际压力接触面积。测试时长 20 秒。",
                ha="center", va="center", fontsize=11, color="#58616e")
    figure.subplots_adjust(top=.93, bottom=.12, left=.055, right=.98)
    path = output / "stance-comparison.png"
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    print("已生成：", path)


if __name__ == "__main__":
    main()
