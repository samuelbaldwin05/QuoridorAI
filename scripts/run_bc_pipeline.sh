#!/usr/bin/env bash
# run_bc_pipeline.sh — End-to-end BC pretraining pipeline for PPO BFS models.
#
# Chains three steps into a single reproducible command:
#   1. Generate 6-channel BC data (HeuristicBot self-play, ~10min for 1000 games)
#   2. Pretrain BC (supervised action imitation, ~5min for 20 epochs)
#   3. PPO fine-tuning with the BC warm-start checkpoint
#
# The steps can also be run independently — this script is a convenience wrapper.
#
# Usage:
#   bash scripts/run_bc_pipeline.sh                      # bfs_resnet, 1000 games
#   bash scripts/run_bc_pipeline.sh bfs_resnet 1000      # explicit args
#   bash scripts/run_bc_pipeline.sh bfs 500              # faster smoke test
#
# Args:
#   $1  MODEL   — PPO model variant: baseline, resnet, bfs, bfs_resnet (default: bfs_resnet)
#   $2  GAMES   — Number of self-play games to generate (default: 1000)
#
# Prerequisites:
#   - wandb login  (run before leaving — W&B is used for BC and PPO logging)
#   - Plug in power — data generation is CPU-bound and takes ~10 minutes

set -euo pipefail

MODEL="${1:-bfs_resnet}"
GAMES="${2:-1000}"

# Validate model arg
case "$MODEL" in
    baseline|resnet|bfs|bfs_resnet) ;;
    *) echo "ERROR: Unknown model '$MODEL'. Choose: baseline resnet bfs bfs_resnet" >&2; exit 1 ;;
esac

# 6-channel models need BFS data; 4-channel models use the standard dataset.
case "$MODEL" in
    bfs|bfs_resnet) DATA_FLAG="--bfs"; DATA_PATH="data/bc_data_bfs.npz" ;;
    *)              DATA_FLAG="";      DATA_PATH="data/bc_data.npz" ;;
esac

CHECKPOINT="checkpoints/bc_pretrained_${MODEL}.pt"

echo "======================================================"
echo " BC Pipeline: model=${MODEL}, games=${GAMES}"
echo " Dataset:     ${DATA_PATH}"
echo " Checkpoint:  ${CHECKPOINT}"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Generate expert data
# ---------------------------------------------------------------------------
echo "=== [Step 1/3] Generating ${GAMES} HeuristicBot vs HeuristicBot games ==="
# shellcheck disable=SC2086  # DATA_FLAG is intentionally unquoted (may be empty)
python -m scripts.generate_bc_data $DATA_FLAG --games "$GAMES"
echo ""

# ---------------------------------------------------------------------------
# Step 2: BC pretraining
# ---------------------------------------------------------------------------
echo "=== [Step 2/3] BC pretraining: model=${MODEL} ==="
python -m scripts.pretrain_bc \
    --model "$MODEL" \
    --data  "$DATA_PATH"
echo ""

# ---------------------------------------------------------------------------
# Step 3: PPO fine-tuning with BC warm-start
# ---------------------------------------------------------------------------
echo "=== [Step 3/3] PPO fine-tuning with checkpoint: ${CHECKPOINT} ==="
python -m scripts.train_ppo \
    --model      "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --run-name   "${MODEL}_bc"
echo ""

echo "======================================================"
echo " Pipeline complete."
echo " Checkpoint: ${CHECKPOINT}"
echo " W&B run:    ${MODEL}_bc"
echo "======================================================"
