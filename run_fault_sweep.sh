#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
export MUJOCO_GL=egl
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
mkdir -p eval_out

echo "=== 1. Checking Baseline (delta=0) ==="
if [ ! -f "eval_out/summary_baseline_0deg.json" ]; then
    echo "Running baseline evaluation..."
    python3 eval/eval_policy.py \
        --policy ./lora_champion \
        --dataset-root data/cube500_bin \
        --dataset-repo-id local/cube500_bin \
        --seeds 5000-5250 \
        --min-reach 0.23 --max-reach 0.27 --count 30 \
        --max-pol 360 \
        --fault-deg 0.0 \
        --tag baseline_0deg 2>&1 | tee eval_out/baseline_0deg.log
else
    echo "Baseline eval_out/summary_baseline_0deg.json already exists, skipping."
fi

echo "=== 2. Running Fault Injection Sweeps ==="
for joint in shoulder_lift elbow_flex wrist_flex; do
    for deg in 1.0 2.0 4.0 6.0 8.0; do
        TAG="sweep_${joint}_${deg}deg"
        if [ -f "eval_out/summary_${TAG}.json" ]; then
            echo ">>> Result for $TAG already exists, skipping. <<<"
            continue
        fi
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
            --tag "$TAG" 2>&1 | tee "eval_out/${TAG}.log"
    done
done

echo "=== 3. Aggregating Results ==="
python3 -c "
import glob, json, os

results = []
for p in sorted(glob.glob('eval_out/summary_*.json')):
    d = json.load(open(p))['summary']
    results.append(d)

print('\n' + '='*75)
print(f'| {\"Tag\":<24} | {\"N\":>3} | {\"Grasp\":>14} | {\"Lift\":>14} | {\"Succ\":>14} |')
print('|' + '-'*26 + '|' + '-'*5 + '|' + '-'*16 + '|' + '-'*16 + '|' + '-'*16 + '|')
for r in results:
    g_str = f\"{r['grasp']}/{r['n']} ({r['grasp']/r['n']*100:.1f}%)\"
    l_str = f\"{r['lift']}/{r['n']} ({r['lift']/r['n']*100:.1f}%)\"
    s_str = f\"{r['succ']}/{r['n']} ({r['succ']/r['n']*100:.1f}%)\"
    print(f'| {r[\"tag\"]:24} | {r[\"n\"]:3} | {g_str:>14} | {l_str:>14} | {s_str:>14} |')
print('='*75 + '\n')
"

echo "=== All sweeps complete! ==="
