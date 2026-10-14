"""汇总实际接触归因并绘制中央盒下沿的接触位置，不改碰撞资产。"""

import json
from pathlib import Path
import shutil
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
import trimesh

from audit_race_cad_sections import clip_segments_z
from check_race_bodyfit_v2 import ROOT, ASSETS, OUT, clip_triangles


def main():
    source = OUT/'shape_contacts_seed42.json'
    report = json.loads(source.read_text())
    baseline = json.loads((OUT/'v2_seed42.json').read_text())
    assert report['completed_episodes'] == baseline['completed_episodes'] == 489
    assert report['termination_counts'] == baseline['termination_counts']
    assert report['candidate_metadata'] == baseline['candidate_metadata']
    trace = report['shape_contact_trace']
    assert trace['unmatched_episodes'] == 0
    assert sum(trace['dominant_pair_episode_counts'].values()) == 489
    log = Path('/tmp/race_v2_shape_contacts.log')
    assert not re.search(r'CUDA error|failed to allocate|Traceback|PhysX error',
                         log.read_text(errors='replace'))
    shutil.copyfile(log, OUT/'physx_logs/race_v2_shape_contacts.log')
    points = {'fr_mjx_thigh':[], 'fl_mjx_thigh':[]}
    for episode in trace['episodes']:
        name = episode['dominant_pair'].split(' <-> ')[-1]
        if name not in points:
            continue
        record = max(episode['contacts_in_termination_window'],
                     key=lambda x:x['normal_contact_force_n'])
        point = max(record['contact_points'],
                    key=lambda x:abs(x['normal_force_n']))
        points[name].append(point['position_base_frame_m'])
    points = {name:np.asarray(values) for name, values in points.items()}
    mesh = trimesh.load_mesh(ROOT/'resources/race/meshes/base_link.stl')
    all_points = np.concatenate(list(points.values()))
    closest, distances, _ = trimesh.proximity.closest_point(mesh, all_points)
    spec = json.loads((ASSETS/'mjx_bodyfit_v2.json').read_text())
    base = [g for g in spec['geoms'] if g['body'] == 'base_link']
    center_box = next(g for g in base if g['name'] == 'bodyfit_base_3')
    center, half = np.asarray(center_box['pos']), np.asarray(center_box['size'])
    cross = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [.06, 0, 0])
    cross = clip_segments_z(cross, center[2]-half[2], center[2]+half[2])
    cross = cross.reshape(-1, 2, 3)
    lower_band = clip_segments_z(cross, -.097, -.087)
    lower_width = (np.ptp(lower_band[:, 1])*1000 if len(lower_band) else None)
    summary = {
        'matched_episodes':489, 'seed':42, 'num_envs':32, 'control_steps':200,
        'action':'每20ms独立N(0,1)动作，目标角缩放0.25rad；非策略回放。',
        'baseline_termination_count_unchanged':True,
        'dominant_contact_pairs':trace['dominant_pair_episode_counts'],
        'any_contact_pairs':trace['any_pair_episode_counts'],
        'central_box_front_thigh_fraction':487/489,
        'position_method':trace['segment_assignment'],
        'segment_ambiguities':trace['ambiguous_segment_points'],
        'front_thigh_contact_points_base_frame_m':{
            name:values.tolist() for name, values in points.items()},
        'median_contact_point_distance_to_base_cad_surface_m':float(
            np.median(distances)),
        'cad_distance_p05_p95_m':np.quantile(distances, [.05, .95]).tolist(),
        'central_x_006m_cad_cross_section_bounds_m':[
            cross.reshape(-1, 3).min(axis=0).tolist(),
            cross.reshape(-1, 3).max(axis=0).tolist()],
        'limitations':['仅统计种子42的一组大幅随机动作，不代表训练策略的碰撞频率。',
                       '刚体接触对来自GPU实测；机身分段由实际接触点与盒面距离推定。',
                       '接触点转入求解后基座坐标，CAD距离含单物理步运动误差，不是全腿机械间隙。'],
    }
    if lower_width is not None:
        summary['central_lower_band_width_mm'] = lower_width
    (OUT/'contact_attribution.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)+'\n')

    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(), 'font.size':10,
                         'axes.unicode_minus':False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={'width_ratios':[1.6, 1]})
    ax = axes[0]
    triangles = clip_triangles(mesh.triangles, 2, .06, True)
    ax.add_collection(PolyCollection(triangles[:, :, :2]*1000,
                                     facecolor='#d0d3d6', edgecolors='none',
                                     rasterized=True))
    for geom in base:
        c, h = np.asarray(geom['pos']), np.asarray(geom['size'])
        ax.add_patch(Rectangle((c[:2]-h[:2])*1000, h[0]*2000, h[1]*2000,
                               facecolor='none', edgecolor='#737c82'))
    colors = {'fr_mjx_thigh':'#b94e42', 'fl_mjx_thigh':'#246b9c'}
    labels = {'fr_mjx_thigh':'右前大腿（250次主接触）',
              'fl_mjx_thigh':'左前大腿（237次主接触）'}
    for name, values in points.items():
        ax.scatter(values[:, 0]*1000, values[:, 1]*1000, s=12,
                   alpha=.5, color=colors[name], label=labels[name])
    ax.set(xlim=(-270, 280), ylim=(-125, 125), aspect='equal',
           xlabel='纵向 X（mm，正方向为前）', ylabel='横向 Y（mm）',
           title='实际接触点集中在中央盒两侧')
    ax.legend(loc='upper left', frameon=False)
    ax = axes[1]
    ax.add_patch(Rectangle(((center[1]-half[1])*1000,
                            (center[2]-half[2])*1000),
                           half[1]*2000, half[2]*2000,
                           facecolor='#f1f1f1', edgecolor='#737c82'))
    ax.add_collection(LineCollection(cross[:, :, 1:3]*1000,
                                     colors='#373f44', linewidths=1.))
    for name, values in points.items():
        chosen = values[np.abs(values[:, 0]-.06)<.005]
        ax.scatter(chosen[:, 1]*1000, chosen[:, 2]*1000, s=16,
                   alpha=.6, color=colors[name])
    ax.set(xlim=(-125, 125), ylim=(-120, 75), aspect='equal',
           xlabel='横向 Y（mm）', ylabel='高度 Z（mm）',
           title='中央截面 X=60 mm：接触位于下角')
    fig.suptitle('新版七段机身：剩余早停主要来自前腿大腿与中央盒',
                 fontsize=14)
    fig.text(.02, .02,
             '来源：32环境×200控制步，种子42；487/489次最大接触对为前腿大腿。'
             '右图叠加X=60±5 mm的接触点，黑线为X=60 mm的CAD截面。', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .93))
    fig.savefig(OUT/'contact_locations.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['dominant_contact_pairs',
          'median_contact_point_distance_to_base_cad_surface_m',
          'central_x_006m_cad_cross_section_bounds_m']},
          ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
