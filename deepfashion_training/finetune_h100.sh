#!/bin/bash

# =================================================================
# Full Fine-Tuning Script for H100 GPU
# =================================================================
# This script is designed for maximum performance and quality on a
# high-end GPU like the H100. It performs full fine-tuning of the
# transformer model, which yields better results than LoRA but
# requires significantly more VRAM and compute.
#
# Estimated Time: 4-8 hours on an H100
# =================================================================

set -e  # Exit immediately if a command exits with a non-zero status.

echo "🚀 [H100] Starting FULL Fine-Tuning for CatVTON-Flux with DeepFashion"
echo

# --- Configuration ---
# Use the local beta model as the base for fine-tuning
MODEL_PATH="/home/sheldon/cvton/catvton-flux-beta"
FALLBACK_MODEL="xiaozaa/catvton-flux-alpha" # Fallback if local model not found
OUTPUT_DIR="/home/sheldon/cvton/catvton-flux/finetune_h100_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_DIR}/logs"

# Check if the primary local model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  Local model not found at $MODEL_PATH. Using online fallback: $FALLBACK_MODEL"
    MODEL_PATH="$FALLBACK_MODEL"
fi

echo "🏷️  Base Model: $MODEL_PATH"
echo "📁 Output Directory: $OUTPUT_DIR"
echo

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# --- Environment Setup ---
echo "🔧 Activating conda environment 'flux'..."
# This assumes miniconda is in the home directory. Adjust if necessary.
# Use 'conda init' once on the new machine if you have issues with sourcing.
source ~/miniconda3/etc/profile.d/conda.sh || source ~/miniconda/etc/profile.d/conda.sh
conda activate flux

# Verify the environment
echo "🔍 Environment Check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
echo

# --- Training Execution ---
# We use `accelerate launch` for multi-GPU and DeepSpeed support, which is ideal for H100.
# The configuration is loaded from the `accelerate_config.yaml` file.

echo "🔥 Launching full fine-tuning process with Accelerate..."

accelerate launch --config_file /home/sheldon/cvton/catvton-flux/accelerate_config.yaml /home/sheldon/cvton/catvton-flux/train_flux_inpaint.py \
    --pretrained_model_name_or_path="$MODEL_PATH" \
    --pretrained_inpaint_model_name_or_path="$MODEL_PATH/transformer" \
    --output_dir="$OUTPUT_DIR" \
    --logging_dir="$LOG_DIR" \
    --dataset_type="deepfashion" \
    --train_base_model \
    --height=512 \
    --width=384 \
    --learning_rate=5e-6 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=10000 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --mixed_precision="bf16" \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=3 \
    --validation_steps=500 \
    --seed=42 \
    --report_to="tensorboard" \
    --gradient_checkpointing

# --- Completion ---
echo
echo "✅ [H100] Full Fine-Tuning Completed!"
echo "📁 Trained model and logs are saved in: $OUTPUT_DIR"
echo
echo "📋 NEXT STEPS:"
echo "1. Evaluate the generated sample images in the output directory."
echo "2. Test the fully fine-tuned model using the inference scripts."
echo "3. To use the new model, point your application to the '$OUTPUT_DIR' directory."
echo