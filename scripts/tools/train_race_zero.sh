#!/usr/bin/env bash
# 在已激活的 Conda 环境中启动一版零位 Race 训练。
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  flat) task=RobotLab-Race-Flat-v0 ;;
  rough) task=RobotLab-Race-v0 ;;
  *) echo '用法：bash scripts/tools/train_race_zero.sh flat|rough'; exit 2 ;;
esac
run_name="race_zero_${1}_4096"
mkdir -p logs/nohup
log_path="logs/nohup/${run_name}_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  OMNI_KIT_ACCEPT_EULA=YES \
  python -u scripts/rsl_rl/train.py \
  --task "$task" --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 10000 \
  --logger tensorboard --run_name "$run_name" \
  > "$log_path" 2>&1 < /dev/null &
train_pid=$!
echo "训练任务：$task；PID：$train_pid"
echo "日志：$log_path"
printf '查看日志：tail -n 80 -f %q\n' "$log_path"
