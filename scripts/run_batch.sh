#!/usr/bin/env bash
# run_batch.sh — Sequential PPO experiment runner.
#
# Runs five experiments back-to-back, each for 150k steps. Logs are written to
# logs/batch_<timestamp>/ so you can review what happened after returning.
# Each experiment is a separate W&B run with a descriptive name.
#
# Experiments:
#   1. baseline_bc       — 4-ch CNN, BC warmstart (control)
#   2. resnet_cold       — 4-ch ResNet + LayerNorm, cold start
#   3. bfs_resnet_cold   — 6-ch ResNet + BFS, cold start
#   4. baseline_long_hz  — baseline + γ=0.999/λ=0.97 (longer horizon)
#   5. baseline_clip05   — baseline + grad_clip=0.5 (tighter clipping)
#
# Usage (IMPORTANT — use caffeinate to prevent macOS sleep):
#   caffeinate -i bash scripts/run_batch.sh   ← recommended
#   bash scripts/run_batch.sh                 ← works but computer may sleep mid-batch
#
# Prerequisites:
#   - wandb login  (run this before leaving — the batch uses W&B)
#   - checkpoints/bc_pretrained.pt must exist (needed by experiments 1, 4, 5)
#   - Plug in power — battery mode throttles CPU and may still allow sleep

set -euo pipefail

# Warn if caffeinate is not active — macOS will sleep and pause training.
if ! pgrep -x caffeinate > /dev/null 2>&1; then
    echo "WARNING: caffeinate is not running. Your computer may sleep mid-batch."
    echo "   Recommended: caffeinate -i bash scripts/run_batch.sh"
    echo "   Continuing in 5 seconds — Ctrl+C to cancel and restart with caffeinate."
    sleep 5
fi

STEPS=150000
LOG_DIR="logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"

echo "Batch started at $(date)" | tee "$SUMMARY"
echo "Logs: $LOG_DIR" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ---------------------------------------------------------------------------
# run_exp NAME ARGS...
#   Runs one experiment. On non-zero exit, logs the failure and continues.
# ---------------------------------------------------------------------------
run_exp() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"

    echo "=== [$name] Starting at $(date) ===" | tee -a "$SUMMARY"
    echo "    Args: $*" | tee -a "$SUMMARY"

    # Run the experiment; capture exit code without triggering set -e
    python -m scripts.train_ppo "$@" \
        --run-name "$name" \
        --max-steps "$STEPS" \
        2>&1 | tee "$log"
    local exit_code="${PIPESTATUS[0]}"

    if [ "$exit_code" -eq 0 ]; then
        echo "=== [$name] COMPLETED at $(date) ===" | tee -a "$SUMMARY"
    else
        echo "=== [$name] FAILED (exit $exit_code) at $(date) ===" | tee -a "$SUMMARY"
        echo "    See $log for details." | tee -a "$SUMMARY"
    fi
    echo "" | tee -a "$SUMMARY"
}

# ---------------------------------------------------------------------------
# Experiment 1: Baseline + BC warmstart (control)
# Why: establishes the reference win rate for the current architecture and
#      hyperparameters. Everything else is compared against this.
# ---------------------------------------------------------------------------
run_exp "baseline_bc" \
    --model baseline \
    --checkpoint checkpoints/bc_pretrained.pt

# ---------------------------------------------------------------------------
# Experiment 2: ResNet + LayerNorm, cold start
# Why: LayerNorm removes the BatchNorm train/eval inconsistency that may be
#      artificially capping the baseline. No BC warmstart (key mismatch).
# ---------------------------------------------------------------------------
run_exp "resnet_cold" \
    --model resnet

# ---------------------------------------------------------------------------
# Experiment 3: BFS + ResNet, cold start
# Why: adds explicit shortest-path signal (ch4/ch5) so the model can directly
#      see the path-length effect of wall placements. The hardest thing for the
#      baseline to learn from wall positions alone.
# ---------------------------------------------------------------------------
run_exp "bfs_resnet_cold" \
    --model bfs_resnet

# ---------------------------------------------------------------------------
# Experiment 4: Baseline + longer horizon (γ=0.999, λ=0.97)
# Why: effective horizon ≈ 333 steps vs current ≈ 100. Quoridor games are
#      20-40 moves; the default γ/λ only reaches ~17 moves back reliably.
#      Keeping baseline architecture isolates the γ/λ effect.
# ---------------------------------------------------------------------------
run_exp "baseline_long_hz" \
    --model baseline \
    --checkpoint checkpoints/bc_pretrained.pt \
    --gamma 0.999 \
    --lam 0.97

# ---------------------------------------------------------------------------
# Experiment 5: Baseline + tighter gradient clipping (0.5 vs 10.0)
# Why: 0.5 is the more standard PPO grad clip; 10.0 may allow policy updates
#      that are too large, destabilizing the BC warmstart. Direct comparison
#      against experiment 1 (same everything, only grad_clip changes).
# ---------------------------------------------------------------------------
run_exp "baseline_clip05" \
    --model baseline \
    --checkpoint checkpoints/bc_pretrained.pt \
    --grad-clip 0.5

# ---------------------------------------------------------------------------
echo "" | tee -a "$SUMMARY"
echo "=== Batch complete at $(date) ===" | tee -a "$SUMMARY"
echo "Review W&B project 'quoridor-ppo' for full training curves." | tee -a "$SUMMARY"
echo "Individual logs: $LOG_DIR/" | tee -a "$SUMMARY"
