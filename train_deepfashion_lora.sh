#!/usr/bin/env bash
# ======================================================================================
#
# Unified Training Script for CatVTON-Flux Fine-Tuning
#
# This script serves as the central entry point for all training scenarios,
# including quick validation tests, full LoRA fine-tuning, and optimized
# runs on high-performance hardware like the H100.
#
# ======================================================================================

# --- Configuration ---
# Set default parameters for a full training run.
# These can be overridden by command-line flags.
ACCELERATE_CONFIG_FILE="accelerate_config_fixed.yaml"  # Use fixed config to avoid GLIBCXX issues
TRAIN_SCRIPT="deepfashion_training/train_deepfashion_lora.py"
OUTPUT_DIR_BASE="/home/sheldon/cvton/catvton-flux/runs"

# Default training parameters
MAX_TRAIN_STEPS=15000
LEARNING_RATE=1e-4
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
VALIDATION_STEPS=500
CHECKPOINTING_STEPS=500
MIXED_PRECISION="bf16" # Use "bf16" for modern GPUs (like H100), "fp16" for others.
MAX_TRAIN_SAMPLES="None" # Use "None" to train on the full dataset

# --- Command-Line Flag Parsing ---
# Parse arguments to adjust the script's behavior for different scenarios.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test)
            echo "🚀 RUN MODE: Quick Test"
            RUN_TYPE="test"
            MAX_TRAIN_STEPS=50
            MAX_TRAIN_SAMPLES=100 # Use a small subset of data for a fast run
            VALIDATION_STEPS=25
            CHECKPOINTING_STEPS=50
            ACCELERATE_CONFIG_FILE="accelerate_config_no_deepspeed.yaml"  # Use non-DeepSpeed config to avoid GLIBCXX issues
            shift
            ;;
        --h100)
            echo "🚀 RUN MODE: H100 Optimized"
            RUN_TYPE="h100_full"
            ACCELERATE_CONFIG_FILE="accelerate_config_h100.yaml"  # Use H100-optimized config
            MIXED_PRECISION="bf16"
            # H100s can handle larger batch sizes, which can speed up training.
            TRAIN_BATCH_SIZE=2
            GRADIENT_ACCUMULATION_STEPS=2
            shift
            ;;
        *)
            echo "❌ Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# Set default run type if not specified
if [ -z "$RUN_TYPE" ]; then
    echo "🚀 RUN MODE: Full LoRA Fine-Tuning"
    RUN_TYPE="full"
fi

# --- Dynamic Output Directory ---
# Create a descriptive output directory based on the run type and timestamp.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR_BASE}/deepfashion_lora_${RUN_TYPE}_${TIMESTAMP}"
echo "💾 Output will be saved to: ${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

# --- Build Accelerate Command ---
# Assemble the final command to be executed.

# Set environment variables to avoid DeepSpeed CPU Adam issues
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

CMD="accelerate launch --config_file \"$ACCELERATE_CONFIG_FILE\" \"$TRAIN_SCRIPT\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --max_train_steps=\"$MAX_TRAIN_STEPS\" \
    --learning_rate=\"$LEARNING_RATE\" \
    --train_batch_size=\"$TRAIN_BATCH_SIZE\" \
    --gradient_accumulation_steps=\"$GRADIENT_ACCUMULATION_STEPS\" \
    --validation_steps=\"$VALIDATION_STEPS\" \
    --checkpointing_steps=\"$CHECKPOINTING_STEPS\" \
    --mixed_precision=\"$MIXED_PRECISION\""

# Only add the max_train_samples argument if it's set to a number.
if [ "$MAX_TRAIN_SAMPLES" != "None" ]; then
    CMD="$CMD --max_train_samples=\"$MAX_TRAIN_SAMPLES\""
fi

# --- Execution ---
# Print the command for verification and then execute it.
echo "--------------------------------------------------------------------"
echo "Executing command:"
echo "$CMD"
echo "--------------------------------------------------------------------"
eval "$CMD"

# --- Completion ---
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training finished successfully. Results are in: $OUTPUT_DIR"
else
    echo "❌ Training failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE
