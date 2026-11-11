#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
export MUJOCO_GL=egl
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
mkdir -p eval_out

echo "=== 1. Running Baseline (delta=0) ==="
python3 eval/eval_policy.py \
    --policy ./lora_champion \
    --dataset-root data/cube500_bin \
    --dataset-repo-id local/cube500_bin \
    --seeds 5000-5250 \
    --min-reach 0.23 --max-reach 0.27 --count 30 \
    --max-pol 360 \
    --fault-deg 0.0 \
    --tag baseline_0deg 2>&1 | tee eval_out/baseline_0deg.log

echo "=== 2. Running Fault Injection Sweeps ==="
for joint in shoulder_lift elbow_flex wrist_flex; do
    for deg in 1.0 2.0 4.0 6.0 8.0; do
        echo ">>> Evaluating $joint at delta=${deg} deg <<<"
        python3 eval/eval_policy.py \
            --policy ./lora_champion \
            --dataset-root data/cube500_bin \
            --dataset-repo-id local/cube500_bin \
            --seeds 5000-5250 \
            --min-reach 0.23 --max-reach 0.27 --count 30 \
            --max-pol 360 \
            --fault-joint "$joint" \
            --fault-deg "$deg" \
            --tag "sweep_${joint}_${deg}deg" 2>&1 | tee "eval_out/sweep_${joint}_${deg}deg.log"
    done
done

echo "=== All sweeps complete! ==="
