"""独立核验七段机身 CAD 覆盖及同姿态碰撞间隙，不改训练资产。"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from compare_race_collision_geometry import (
    ROOT, LEGS, DEFAULT, CURRENT_XML, build_candidate,
    mechanical_assertions, distances, set_pose,
)

ASSETS = Path('/home/zsy/project/mujoco_playground/mujoco_playground/'
              '_src/locomotion/race/assets')
OUT = ROOT / 'logs/race_bodyfit_v2'


def clip_triangles(triangles, axis, bound, keep_lower):
    """精确切割跨界三角面；完整保留内部面，丢弃外部面。"""
    signs = triangles[:, :, axis] - bound
    if not keep_lower:
        signs = -signs
    inside = signs <= 0
    result = list(triangles[inside.all(axis=1)])
    for triangle in triangles[inside.any(axis=1) & ~inside.all(axis=1)]:
        polygon = []
        for first, second in zip(triangle, np.roll(triangle, -1, axis=0)):
            da, db = first[axis] - bound, second[axis] - bound
            if not keep_lower:
                da, db = -da, -db
            if da <= 0:
                polygon.append(first)
            if (da <= 0) != (db <= 0):
                polygon.append(first + da / (da - db) * (second - first))
        for i in range(1, len(polygon) - 1):
            result.append([polygon[0], polygon[i], polygon[i + 1]])
    return np.asarray(result).reshape(-1, 3, 3)


def inspect_cad(spec, parent):
    """每个纵向区间的完整三角面均被对应凸盒覆盖，非稀疏采样。"""
    boxes = [g for g in spec['geoms'] if g['body'] == 'base_link']
    assert len(boxes) == 7 and len(spec['geoms']) == 32
    assert [g for g in spec['geoms'] if g['body'] != 'base_link'] == [
        g for g in parent['geoms'] if g['body'] != 'base_link']
    hashes = {}
    for relative, expected in spec['source_mesh_sha256'].items():
        path = ROOT / 'resources/race' / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, relative
        hashes[relative] = actual
    mesh = trimesh.load_mesh(ROOT / 'resources/race/meshes/base_link.stl')
    triangles = clip_triangles(mesh.triangles, 2, .06, True)
    intervals = spec['body_fit']['segments']
    assert np.isclose(intervals[0]['x_interval_m'][0], triangles[:, :, 0].min())
    assert np.isclose(intervals[-1]['x_interval_m'][1], triangles[:, :, 0].max())
    rows = []
    for i, (geom, segment) in enumerate(zip(boxes, intervals, strict=True)):
        assert geom['type'] == 'box' and geom['quat'] == [1., 0., 0., 0.]
        lower, upper = segment['x_interval_m']
        if i:
            assert abs(lower - intervals[i-1]['x_interval_m'][1]) < 1e-12
        clipped = clip_triangles(triangles, 0, lower, False)
        clipped = clip_triangles(clipped, 0, upper, True)
        error = np.max(np.abs(clipped.reshape(-1, 3) - geom['pos'])
                       - geom['size'])
        assert error <= -.000499999, (geom['name'], error)
        rows.append({'name':geom['name'], 'x_interval_m':[lower, upper],
                     'dimensions_m':(2*np.asarray(geom['size'])).tolist(),
                     'clipped_triangle_count':len(clipped),
                     'max_outside_m':float(error)})
    current_box = [.518, .206, .157]
    volume = sum(np.prod(2*np.asarray(g['size'])) for g in boxes)
    return {'passed':True, 'source_mesh_hashes_verified':hashes,
            'non_base_geometries_identical_to_parent':True,
            'coverage_method':'纵向区间覆盖完整CAD范围；每段裁切三角面的全部顶点在对应凸盒内，因此整片表面均被覆盖。',
            'top_clip_m':.06,
            'limitations':'维持原配方忽略Z>0.06m的顶部附件；CAD非封闭，未验证实体体积。',
            'segments':rows, 'sum_box_volumes_m3':float(volume),
            'box_volume_reduction_vs_current':float(1-volume/np.prod(current_box)),
            'box_volume_reduction_vs_parent':float(1-volume/spec['body_fit']['parent_base_box_volume_m3'])}


def indices(model):
    joints = [model.joint(f'{kind}_leg_{leg}_joint').id
              for leg in LEGS for kind in ('roll', 'pitch01', 'pitch02')]
    colliders = np.flatnonzero(model.geom_group == 1)
    base = [g for g in colliders
            if model.geom_bodyid[g] == model.body('base_link').id]
    legs = []
    for leg in LEGS:
        bodies = [model.body(f'pitch0{i}_leg_{leg}_link').id for i in (1, 2, 3)]
        for geom in colliders:
            name = model.geom(geom).name
            # 同当前训练，只跳过机身与髋壳的四对检测。
            if model.geom_bodyid[geom] in bodies and not (
                name.endswith('_hip_housing') or name.endswith('_mjx_hip')):
                legs.append(geom)
    return model.jnt_qposadr[joints], base, legs


def min_gap(model, data, base, legs):
    return float(min(distances(model, data, b, legs).min() for b in base))


def render_envelopes(spec, parent):
    """直观看到 CAD 投影与分段机身盒，所有坐标均为毫米。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    font = FontProperties(
        fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(), 'font.size':10,
                         'axes.unicode_minus':False})
    mesh = trimesh.load_mesh(ROOT / 'resources/race/meshes/base_link.stl')
    triangles = clip_triangles(mesh.triangles, 2, .06, True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, payload, title, color in zip(
            axes, [parent, spec], ['上一版：单盒机身', '新版：七段机身'],
            ['#b53b32', '#267882'], strict=True):
        for geom in payload['geoms']:
            if geom['body'] != 'base_link':
                continue
            center, half = np.asarray(geom['pos']), np.asarray(geom['size'])
            ax.add_patch(Rectangle((center[:2]-half[:2])*1000,
                                   half[0]*2000, half[1]*2000,
                                   facecolor=color, alpha=.12, linewidth=0))
        ax.add_collection(PolyCollection(
            triangles[:, :, :2]*1000, facecolor='#6c747b',
            edgecolors='none', rasterized=True))
        for geom in payload['geoms']:
            if geom['body'] != 'base_link':
                continue
            center, half = np.asarray(geom['pos']), np.asarray(geom['size'])
            ax.add_patch(Rectangle((center[:2]-half[:2])*1000,
                                   half[0]*2000, half[1]*2000,
                                   facecolor='none', edgecolor=color,
                                   linewidth=1.3))
        ax.set(xlim=(-275, 285), ylim=(-125, 125), aspect='equal',
               xlabel='纵向 X（mm）', title=title)
        ax.set_ylabel('横向 Y（mm）')
        for x in [-182.5, 182.5]:
            ax.axvline(x, color='#40464b', linewidth=.7, linestyle='--')
    fig.suptitle('相同 CAD、相同腿部几何；只改变机身包络', fontsize=14)
    fig.text(.02, .025,
             '灰色：Z≤60 mm 的 CAD 表面投影；彩色：碰撞盒投影。'
             '虚线：前后髋位置。投影不表示完整三维实体覆盖。', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .93))
    fig.savefig(OUT/'body_envelopes.png', dpi=170)
    plt.close(fig)


def scan(model, poses):
    data = mujoco.MjData(model)
    qpos, base, legs = indices(model)
    gaps = []
    for pose in poses:
        set_pose(model, data, qpos, pose)
        gaps.append(min_gap(model, data, base, legs))
    gaps = np.asarray(gaps)
    return {'samples':len(poses),
            'overlap_fraction_below_minus_1um':float((gaps < -1e-6).mean()),
            'minimum_gap_m':float(gaps.min()),
            'median_gap_m':float(np.median(gaps))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parent_path = ASSETS / 'mjx26_v1.json'
    candidate_path = ASSETS / 'mjx_bodyfit_v2.json'
    spec = json.loads(candidate_path.read_text())
    parent = json.loads(parent_path.read_text())
    assert spec['parent_geometry_sha256'] == hashlib.sha256(
        parent_path.read_bytes()).hexdigest()
    report = {'candidate_sha256':hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
              'cad_coverage':inspect_cad(spec, parent)}
    paths = {'current':CURRENT_XML}
    for name, source in [('parent', parent_path), ('v2', candidate_path)]:
        paths[name] = OUT / f'same_mechanics_{name}.xml'
        build_candidate(source, paths[name])
    models = {name:mujoco.MjModel.from_xml_path(str(path))
              for name, path in paths.items()}
    report['mechanical_checks'] = {
        name:mechanical_assertions(models['current'], model)
        for name, model in models.items() if name != 'current'}
    rng = np.random.default_rng(20261014)
    m = models['current']
    js = [m.joint(f'{kind}_leg_{leg}_joint').id
          for leg in LEGS for kind in ('roll', 'pitch01', 'pitch02')]
    broad = np.clip(DEFAULT+rng.normal(0, .25, (1024, 12)),
                    m.jnt_range[js, 0], m.jnt_range[js, 1])
    small = DEFAULT+rng.uniform(-.03, .03, (1024, 12))
    report['geometry_pressure_scan'] = {
        'note':'同1024组姿态，标准差0.25rad；仅几何压力扫描，不代表训练状态分布或跌倒率。四对髋壳已排除。',
        **{name:scan(model, broad) for name, model in models.items()}}
    report['small_geometry_scan'] = {
        name:scan(model, small) for name, model in models.items()}
    sample = json.loads((ROOT / 'logs/race_collision_fix/filtered_seed42.json')
                       .read_text())['first_ten_reset_samples'][2]
    angles = dict(zip(sample['joint_names'], sample['joint_pos_rad']))
    pose = np.asarray([angles[f'{kind}_leg_{leg}_joint']
                       for leg in LEGS for kind in ('roll', 'pitch01', 'pitch02')])
    replay = {}
    for name, model in models.items():
        data = mujoco.MjData(model)
        qpos, base, legs = indices(model)
        set_pose(model, data, qpos, pose)
        thigh = [g for g in legs if model.geom_bodyid[g] ==
                 model.body('pitch01_leg_b_r_link').id]
        replay[name] = {'right_rear_thigh_base_gap_m':min_gap(
            model, data, base, thigh)}
    report['previous_227n_sample_geometry'] = replay
    report['formal_training_geometry_replaced'] = False
    render_envelopes(spec, parent)
    (OUT / 'geometry.json').write_text(json.dumps(report, ensure_ascii=False,
                                                 indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['cad_coverage',
          'geometry_pressure_scan','small_geometry_scan',
          'previous_227n_sample_geometry']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
