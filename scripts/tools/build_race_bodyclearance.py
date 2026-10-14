"""基于已核验 v2，仅收紧中央盒下角，生成独立对照候选。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from check_race_bodyfit_v2 import ASSETS, indices, min_gap
from compare_race_collision_geometry import (
    ROOT, CURRENT_XML, DEFAULT, build_candidate, mechanical_assertions, set_pose,
)

OUT = ROOT / 'logs/race_bodyclearance_v3'
TARGET = ROOT / 'resources/race/collision_candidates/mjx_bodyclearance_v3.json'


def main():
    parent_path = ASSETS / 'mjx_bodyfit_v2.json'
    original = json.loads(parent_path.read_text())
    spec = deepcopy(original)
    old = next(g for g in spec['geoms'] if g['name'] == 'bodyfit_base_3')
    # 横截面遵循圆底趋势，局部有意欠包络；不全局缩窄或修改腿部。
    bands = [('upper', -.025, .0605, .195),
             ('middle', -.060, -.025, .170),
             ('lower', -.090, -.060, .100)]
    replacements = []
    for name, lo, hi, width in bands:
        geom = deepcopy(old)
        geom['name'] = f'clearance_base_3_{name}'
        geom['pos'][2] = (lo+hi)/2
        geom['size'][1:] = [width/2, (hi-lo)/2]
        assert all(abs(np.asarray(geom['pos'])-old['pos'])
                   + np.asarray(geom['size']) <= np.asarray(old['size'])+1e-12)
        replacements.append(geom)
    spec['geoms'] = [g for g in spec['geoms'] if g['name'] != old['name']]
    spec['geoms'][3:3] = replacements
    assert [g for g in original['geoms'] if g['name'] != old['name']] == [
        g for g in spec['geoms'] if not g['name'].startswith('clearance_base_3_')]
    spec['mode'] = 'mjx_bodyclearance_v3'
    spec['source'] = '用户授权收紧中央盒下角的独立候选，非完整 CAD 外包络'
    spec['parent_geometry_sha256'] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    spec['central_box_before'] = old
    spec['body_fit'] = {
        'method': '保留六段，中央盒替换为上宽下窄三个子盒，均为旧盒子集。',
        'full_cad_coverage_required': False,
        'bands': [{'name': n, 'z_interval_m': [lo, hi], 'width_m': w}
                  for n, lo, hi, w in bands],
        'limitation': '局部外壳会先于简化盒接触障碍，需要策略回放验证。',
    }
    spec.pop('fit_checks', None)
    spec['recipe']['contact_filter'] = '自碰撞开启，仅过滤机身与四个髋壳；保持原终止规则'
    spec['recipe']['margin_m'] = None
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n')
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_path = OUT / 'same_mechanics_v3.xml'
    build_candidate(TARGET, candidate_path)
    reference = mujoco.MjModel.from_xml_path(str(CURRENT_XML))
    model = mujoco.MjModel.from_xml_path(str(candidate_path))
    checks = mechanical_assertions(reference, model)
    data = mujoco.MjData(model)
    qpos, bases, legs = indices(model)
    set_pose(model, data, qpos, DEFAULT)
    old_volume = np.prod(2*np.asarray(old['size']))
    volume = sum(np.prod(2*np.asarray(g['size'])) for g in replacements)
    result = {
        'candidate_path': str(TARGET),
        'sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        'mechanics': checks, 'collider_count': len(spec['geoms']),
        'only_central_base_box_changed': True,
        'new_boxes_are_subsets_of_original': True,
        'central_box_volume_reduction': float(1-volume/old_volume),
        'default_base_to_leg_minimum_geometry_gap_m': min_gap(
            model, data, bases, legs),
        'formal_training_geometry_replaced': False,
        'cad_full_coverage': False,
    }
    (OUT / 'geometry.json').write_text(json.dumps(
        result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
