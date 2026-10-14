#!/usr/bin/env bash
# 默认启动平地模型；窗口中按 1/2 切换模型，按 T 切换地形。
set -euo pipefail

RACE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RACE_PYTHON="${RACE_PYTHON:-/home/zsy/anaconda3/envs/go2_rl_robotlab/bin/python}"
RACE_MODEL="${1:-3400}"
RACE_TERRAIN="${2:-flat}"

case "$RACE_MODEL" in
    3400|9400) ;;
    *) echo "模型参数必须为 3400 或 9400" >&2; exit 2 ;;
esac

cd -- "$RACE_ROOT"
exec env -u PYTHONPATH MUJOCO_GL=glfw OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$RACE_PYTHON" -u scripts/tools/evaluate_race_checkpoints.py \
    --checkpoint "$RACE_ROOT/model_${RACE_MODEL}.pt" \
    --keyboard --terrain "$RACE_TERRAIN"
