#!/usr/bin/env bash
# Unified training script for DeepFashion LoRA on CatVTON-Flux

# Default parameters for a full training run
ACCELERATE_CONFIG_FILE="accelerate_config.yaml"
TRAIN_SCRIPT="deepfashion_training/train_deepfashion_lora.py"
OUTPUT_DIR="/home/sheldon/cvton/catvton-flux/deepfashion_training/runs/deepfashion_lora_full_$(date +%Y%m%d_%H%M%S)"
MAX_TRAIN_STEPS=10000
LEARNING_RATE=1e-4
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
VALIDATION_STEPS=500
CHECKPOINTING_STEPS=500
MIXED_PRECISION="bf16" # Use "fp16" or "bf16"
MAX_TRAIN_SAMPLES=None

# --- Command Line Argument Parsing ---
# Allows for overriding defaults for different scenarios (e.g., quick test, h100)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --quick-test)
            echo "🚀 Running in QUICK TEST mode!"
            MAX_TRAIN_STEPS=50
            MAX_TRAIN_SAMPLES=100
            VALIDATION_STEPS=25
            CHECKPOINTING_STEPS=50
            OUTPUT_DIR="/home/sheldon/cvton/catvton-flux/deepfashion_training/runs/deepfashion_lora_test_$(date +%Y%m%d_%H%M%S)"
            shift
            ;;
        --h100)
            echo "🚀 Configuring for H100!"
            MIXED_PRECISION="bf16"
            # H100 can handle larger batches
            TRAIN_BATCH_SIZE=2 
            GRADIENT_ACCUMULATION_STEPS=2
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

# --- Build Command ---
CMD="accelerate launch --config_file \"$ACCELERATE_CONFIG_FILE\" \"$TRAIN_SCRIPT\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --max_train_steps=\"$MAX_TRAIN_STEPS\" \
    --learning_rate=\"$LEARNING_RATE\" \
    --train_batch_size=\"$TRAIN_BATCH_SIZE\" \
    --gradient_accumulation_steps=\"$GRADIENT_ACCUMULATION_STEPS\" \
    --validation_steps=\"$VALIDATION_STEPS\" \
    --checkpointing_steps=\"$CHECKPOINTING_STEPS\" \
    --mixed_precision=\"$MIXED_PRECISION\""

if [ "$MAX_TRAIN_SAMPLES" != "None" ]; then
    CMD="$CMD --max_train_samples=\"$MAX_TRAIN_SAMPLES\""
fi

# --- Execute Command ---
echo "Executing command:"
echo "$CMD"
eval "$CMD"

echo "✅ Training finished. Results are in: $OUTPUT_DIR"