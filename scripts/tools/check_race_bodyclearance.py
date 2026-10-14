"""汇总独立收紧模型的几何、导入、随机压力与接触回归证据。"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/race_bodyclearance_v3'


def main():
    geometry = json.loads((OUT/'geometry.json').read_text())
    target = Path(geometry['candidate_path'])
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == geometry['sha256']
    payload = json.loads(target.read_text())
    for relative, expected in payload['source_mesh_sha256'].items():
        assert hashlib.sha256((ROOT/'resources/race'/relative).read_bytes()
                              ).hexdigest() == expected
    logs_dir = OUT/'physx_logs'
    logs_dir.mkdir(exist_ok=True)
    reports = {}
    for name in ['v3_seed42', 'v3_seed43', 'v3_seed44',
                 'v3_uniform_003', 'v3_zero_10s', 'v3_probes']:
        path = OUT/(name+'.json')
        report = json.loads(path.read_text())
        log = Path('/tmp')/('race_'+name+'.log')
        content = log.read_text(errors='replace')
        assert '[诊断完成]' in content
        assert not re.search(r'Traceback|CUDA error|PhysX error|'
                             r'failed to allocate|GPU error|eOUT_OF_MEMORY',
                             content), name
        assert log.stat().st_mtime >= path.stat().st_mtime
        shutil.copyfile(log, logs_dir/log.name)
        metadata = report['candidate_metadata']
        assert metadata['source_sha256'] == digest
        assert metadata['actual_usd_shape_and_dimensions_verified']
        assert metadata['inertials_and_joints_unchanged']
        assert metadata['count'] == 34
        assert report['base_hip_housing_filter_enabled']
        filters = report['collision_filter_relationships']
        assert len(filters) == 1 and len(filters[0]['targets']) == 4
        reports[name] = report
    rows = []
    for seed in [42, 43, 44]:
        previous = json.loads((ROOT/'logs/race_bodyfit_v2'/
                               f'v2_seed{seed}.json').read_text())
        current = reports[f'v3_seed{seed}']
        for key in ['seed', 'num_envs', 'steps', 'action_std', 'action_seed',
                    'action_distribution', 'total_environment_steps',
                    'training_randomization_enabled', 'self_collisions']:
            assert previous[key] == current[key], (seed, key)
        assert previous['candidate_metadata']['source_sha256'] == (
            payload['parent_geometry_sha256'])
        rows.append({'seed': seed,
                     'v2_illegal_contact': previous['termination_counts'][
                         'illegal_contact'],
                     'v3_illegal_contact': current['termination_counts'][
                         'illegal_contact']})
    small = reports['v3_uniform_003']
    assert small['joint_delta_bound_rad'] == .03
    assert small['termination_counts']['illegal_contact'] == 0
    zero = reports['v3_zero_10s']
    assert zero['self_collisions']
    assert zero['first_base_contact_over_1n'] is None
    assert zero['last_step']['time_s'] == 10.
    assert all(.33 < v < .35 for v in zero['last_step']['base_height_m'])
    probes = reports['v3_probes']
    for label, contact, terminated in zip(
            probes['env_case_order'], probes['max_base_contact_n_per_env'],
            probes['any_illegal_contact_per_env']):
        should_contact = label in ['retained_thigh_contact', 'base_ground']
        assert terminated == should_contact
        assert (contact > 1.) == should_contact
    previous_total = sum(row['v2_illegal_contact'] for row in rows)
    total = sum(row['v3_illegal_contact'] for row in rows)
    formal = ROOT/'resources/race/urdf/race.urdf'
    assert hashlib.sha256(formal.read_bytes()).hexdigest() == (
        '2dba749b393968e2ccdedd5aed99ad69fba7efb924931faf017c0f51946863be')
    result = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'local_candidate_checks_passed': True,
        'geometry': geometry,
        'stress_test': {'rows': rows, 'v2_total': previous_total,
                        'v3_total': total,
                        'reduction_fraction': 1-total/previous_total,
                        'action': '每20ms独立高斯动作N(0,1)，目标角缩放0.25rad；不是策略学习。',
                        'comparison': '动作序列独立于复位随机数；终止不同后，复位状态不逐回合配对。'},
        'uniform_003_terminations': small['termination_counts'],
        'zero_action_10s': zero['last_step'],
        'controlled_contact_probes': {
            'labels': probes['env_case_order'],
            'max_contact_n': probes['max_base_contact_n_per_env'],
            'terminated': probes['any_illegal_contact_per_env'],
            'scope': '故意重叠探针只验证碰撞检测与终止仍生效，不代表正常载荷或实机安全。'},
        'source_mesh_hashes_verified': True,
        'actual_usd_shape_dimensions_verified': True,
        'formal_training_geometry_replaced': False,
        'ppo_or_stair_success_tested': False,
        'limitations': ['欠包络是用户授权的简化，不是精确 CAD 接触模型。',
                        'Go2的4941组扫描仅涉及左前大腿盒与机身盒，不能证明所有碰撞对永不接触。',
                        '未将剩余25次终止全部归因为触地或真实机械碰撞。'],
    }
    (OUT/'verification.json').write_text(json.dumps(
        result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': True, 'stress_test': result['stress_test'],
                      'stand_height_range_m': [
                          float(np.min(zero['last_step']['base_height_m'])),
                          float(np.max(zero['last_step']['base_height_m']))]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
