#!/usr/bin/env bash
# Train the Self-Adaptive VLA Masker policies (Week 3-5).
# Trains LoRA on MolmoAct2 VLM vision backbone + full action expert on relabeled data.
# Fits a 24 GB GPU (RTX 4090 or RTX 3090).
#
# Usage:
#   bash configs/train_masker.sh 2deg   # Weak Masker (+/- 2 deg)
#   bash configs/train_masker.sh 5deg   # Medium Masker (+/- 5 deg)
#   bash configs/train_masker.sh 10deg  # Strong Masker (+/- 10 deg)
#   bash configs/train_masker.sh all    # Train all 3 sequentially
set -euo pipefail

TIER=${1:-2deg}
STEPS=${STEPS:-4000}

train_tier() {
    local t=$1
    echo "=========================================================="
    echo ">>> Starting training for Masker tier: ${t} <<<"
    echo "=========================================================="

    local DATA_ROOT="data/cube500_masker_${t}"
    local REPO_ID="local/cube500_masker_${t}"
    local OUT_DIR="outputs/masker_${t}"

    if [ ! -d "$DATA_ROOT" ]; then
        echo "Error: Dataset directory $DATA_ROOT does not exist!"
        exit 1
    fi

    lerobot-train \
      --dataset.repo_id="$REPO_ID" \
      --dataset.root="$DATA_ROOT" \
      --dataset.video_backend=pyav \
      --dataset.image_transforms.enable=true \
      --policy.type=molmoact2 \
      --policy.checkpoint_path=allenai/MolmoAct2-SO100_101 \
      --policy.device=cuda \
      --policy.action_mode=both \
      --policy.enable_lora_vlm=true \
      --policy.chunk_size=10 \
      --policy.n_action_steps=10 \
      --policy.setup_type='single so100/so101 robotic arm in molmoact2' \
      --policy.control_mode='absolute joint pose' \
      --policy.image_keys='["observation.images.cam0","observation.images.cam1"]' \
      --policy.model_dtype=bfloat16 \
      --policy.num_flow_timesteps=8 \
      --policy.gradient_checkpointing=true \
      --policy.normalize_gripper=true \
      --policy.optimizer_action_expert_lr=5e-5 \
      --policy.push_to_hub=false \
      --wandb.enable=false \
      --job_name="masker_${t}" \
      --output_dir="$OUT_DIR" \
      --steps="$STEPS" \
      --batch_size=8 \
      --num_workers=4 \
      --log_freq=100 \
      --save_checkpoint=true \
      --save_freq="$STEPS"

    echo "Checkpoint ready: $OUT_DIR/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model"
}

if [ "$TIER" = "all" ]; then
    for t in 2deg 5deg 10deg; do
        train_tier "$t"
    done
else
    train_tier "$TIER"
fi

echo "All requested masker training runs completed successfully!"
