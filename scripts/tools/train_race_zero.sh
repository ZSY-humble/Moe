#!/usr/bin/env bash
# Race 默认 16384 环境、300000 轮，按用户指定每 1000 轮保存。
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  flat) task=RobotLab-Race-Flat-v0 ;;
  rough) task=RobotLab-Race-v0 ;;
  *) echo '用法：bash scripts/tools/train_race_zero.sh flat|rough'; exit 2 ;;
esac
run_name="race_original_${1}"
mkdir -p logs/nohup
log_path="logs/nohup/${run_name}_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  OMNI_KIT_ACCEPT_EULA=YES \
  python -u scripts/rsl_rl/train.py \
  --task "$task" --headless --device cuda:0 \
  --logger tensorboard --run_name "$run_name" \
  > "$log_path" 2>&1 < /dev/null &
train_pid=$!
echo "训练任务：$task；PID：$train_pid"
echo "日志：$log_path"
printf '查看日志：tail -n 80 -f %q\n' "$log_path"
