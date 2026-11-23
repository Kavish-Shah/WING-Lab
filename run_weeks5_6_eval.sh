#!/usr/bin/env bash
# ==============================================================================
# Weeks 5-6 Automated Evaluation Pipeline:
# Internal Signal Detectors & 300-Episode Fault Ramp Experiment
# ==============================================================================
set -euo pipefail

OUT_DIR="eval_out"
EPISODES=${1:-300}
JOINT=${2:-shoulder_lift}

mkdir -p "$OUT_DIR"

echo "=========================================================="
echo ">>> WEEKS 5-6 EVALUATION PIPELINE <<<"
echo "Target Episodes per Ramp: $EPISODES"
echo "Fault Joint: $JOINT (0 deg -> 8 deg)"
echo "=========================================================="

# 1. Baseline Policy Ramp (Unadapted Champion)
echo ">>> [1/4] Running 300-Episode Ramp on Baseline Policy (ataghof/molmoact2-so101nexus-lora-champion)..."
python3 eval/eval_ramp.py \
  --policy ataghof/molmoact2-so101nexus-lora-champion \
  --episodes "$EPISODES" \
  --fault-joint "$JOINT" \
  --start-deg 0.0 \
  --max-deg 8.0 \
  --tag baseline \
  --out-dir "$OUT_DIR"

# 2. Weak Masker (2deg) Ramp
echo ">>> [2/4] Running 300-Episode Ramp on Weak Masker (2deg)..."
python3 eval/eval_ramp.py \
  --policy kvshah/molmoact2-so101-masker-2deg \
  --episodes "$EPISODES" \
  --fault-joint "$JOINT" \
  --start-deg 0.0 \
  --max-deg 8.0 \
  --tag masker_2deg \
  --out-dir "$OUT_DIR"

# 3. Medium Masker (5deg) Ramp
echo ">>> [3/4] Running 300-Episode Ramp on Medium Masker (5deg)..."
python3 eval/eval_ramp.py \
  --policy kvshah/molmoact2-so101-masker-5deg \
  --episodes "$EPISODES" \
  --fault-joint "$JOINT" \
  --start-deg 0.0 \
  --max-deg 8.0 \
  --tag masker_5deg \
  --out-dir "$OUT_DIR"

# 4. Strong Masker (10deg) Ramp
echo ">>> [4/4] Running 300-Episode Ramp on Strong Masker (10deg)..."
python3 eval/eval_ramp.py \
  --policy kvshah/molmoact2-so101-masker-10deg \
  --episodes "$EPISODES" \
  --fault-joint "$JOINT" \
  --start-deg 0.0 \
  --max-deg 8.0 \
  --tag masker_10deg \
  --out-dir "$OUT_DIR"

# 5. Train Activation Probe on Strong Masker
echo ">>> [5/5] Training Latent Activation Probe on Strong Masker..."
python3 eval/train_activation_probe.py \
  --policy kvshah/molmoact2-so101-masker-10deg \
  --episodes-per-fault 10 \
  --fault-sweep "0,1,2,4,6,8" \
  --tag probe_masker_10deg \
  --out-dir "$OUT_DIR"

echo "=========================================================="
echo ">>> ALL WEEKS 5-6 EVALUATIONS COMPLETED SUCCESSFULLY! <<<"
echo "Summaries and telemetry CSVs saved to $OUT_DIR/"
echo "=========================================================="
