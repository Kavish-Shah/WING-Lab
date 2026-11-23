#!/usr/bin/env bash
# ==============================================================================
# Weeks 5-6 Automated Evaluation Pipeline:
# Internal Signal Detectors & 300-Episode Fault Ramp Experiment
# ==============================================================================
set -euo pipefail

export PYTHONPATH=.
export HF_TOKEN=${HF_TOKEN:-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}

OUT_DIR="eval_out"
EPISODES=${1:-50}
JOINT=${2:-shoulder_lift}

mkdir -p "$OUT_DIR"

echo "=========================================================="
echo ">>> WEEKS 5-6 EVALUATION PIPELINE <<<"
echo "Target Episodes per Ramp: $EPISODES"
echo "Fault Joint: $JOINT (0 deg -> 8 deg)"
echo "=========================================================="

run_ramp() {
  local POLICY=$1
  local TAG=$2
  if [ -f "$OUT_DIR/ramp_${TAG}_summary.json" ]; then
    echo ">>> [$TAG] $OUT_DIR/ramp_${TAG}_summary.json already exists! Skipping to next... <<<"
  else
    echo ">>> Running $EPISODES-Episode Ramp on $TAG ($POLICY)... <<<"
    python3 eval/eval_ramp.py \
      --policy "$POLICY" \
      --episodes "$EPISODES" \
      --fault-joint "$JOINT" \
      --start-deg 0.0 \
      --max-deg 8.0 \
      --tag "$TAG" \
      --out-dir "$OUT_DIR"
  fi
}

# 1. Baseline Policy Ramp (Unadapted Champion)
run_ramp "ataghof/molmoact2-so101nexus-lora-champion" "baseline"

# 2. Weak Masker (2deg) Ramp
run_ramp "kvshah/molmoact2-so101-masker-2deg" "masker_2deg"

# 3. Medium Masker (5deg) Ramp
run_ramp "kvshah/molmoact2-so101-masker-5deg" "masker_5deg"

# 4. Strong Masker (10deg) Ramp
run_ramp "kvshah/molmoact2-so101-masker-10deg" "masker_10deg"

# 5. Train Activation Probe on Strong Masker
if [ -f "$OUT_DIR/activation_probe_summary.json" ]; then
  echo ">>> [probe_masker_10deg] Summary already exists! Skipping... <<<"
else
  echo ">>> [5/5] Training Latent Activation Probe on Strong Masker... <<<"
  python3 eval/train_activation_probe.py \
    --policy kvshah/molmoact2-so101-masker-10deg \
    --episodes-per-fault 10 \
    --fault-sweep "0,1,2,4,6,8" \
    --tag probe_masker_10deg \
    --out-dir "$OUT_DIR"
fi

echo "=========================================================="
echo ">>> ALL WEEKS 5-6 EVALUATIONS COMPLETED SUCCESSFULLY! <<<"
echo "Summaries and telemetry CSVs saved to $OUT_DIR/"
echo "=========================================================="
