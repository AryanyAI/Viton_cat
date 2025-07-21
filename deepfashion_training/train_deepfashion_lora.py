#!/usr/bin/env python3
"""
Unified Training Script for DeepFashion + CatVTON-Flux LoRA
Supports both quick tests and full training runs.
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfashion_training.deepfashion_dataset import DeepFashionDataset
from train_flux_inpaint import main as train_main

def create_training_args():
    """Create arguments for training."""
    parser = argparse.ArgumentParser(description="Train CatVTON-Flux with DeepFashion using LoRA.")

    # Model paths
    parser.add_argument('--pretrained_model_name_or_path', type=str, default="black-forest-labs/FLUX.1-dev", help="Path to the base pretrained model (contains tokenizers).")
    parser.add_argument('--pretrained_inpaint_model_name_or_path', type=str, default="xiaozaa/catvton-flux-alpha", help="Path to the pretrained inpaint model (transformer).")

    # Dataset settings
    parser.add_argument('--dataroot', type=str, default=None, help="Not used with DeepFashion dataset.")
    parser.add_argument('--train_data_list', type=str, default=None)
    parser.add_argument('--train_verification_list', type=str, default=None)
    parser.add_argument('--validation_data_list', type=str, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None, help="Max training samples for quick testing.")
    parser.add_argument('--max_validation_samples', type=int, default=20, help="Max validation samples.")

    # Training parameters
    parser.add_argument('--max_train_steps', type=int, default=10000, help="Total training steps.")
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=1, help="Batch size per GPU.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Effective batch size = train_batch_size * gradient_accumulation_steps.")

    # LoRA parameters
    parser.add_argument('--train_base_model', action='store_true', default=True, help="Enable LoRA training.")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default="constant")
    parser.add_argument('--lr_warmup_steps', type=int, default=100)
    parser.add_argument('--lr_num_cycles', type=int, default=1)
    parser.add_argument('--lr_power', type=float, default=1.0)

    # Memory and performance
    parser.add_argument('--mixed_precision', type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision training.")
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    parser.add_argument('--allow_tf32', action='store_true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Validation and Checkpointing
    parser.add_argument('--validation_steps', type=int, default=500)
    parser.add_argument('--checkpointing_steps', type=int, default=500)
    parser.add_argument('--checkpoints_total_limit', type=int, default=5)

    # Output
    parser.add_argument('--output_dir', type=str, default=f"/home/sheldon/cvton/catvton-flux/deepfashion_training/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--logging_dir', type=str, default="logs")

    # Image settings
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--max_sequence_length', type=int, default=512)

    # Training details
    parser.add_argument('--weighting_scheme', type=str, default="logit_normal")
    parser.add_argument('--logit_mean', type=float, default=0.0)
    parser.add_argument('--logit_std', type=float, default=1.0)
    parser.add_argument('--mode_scale', type=float, default=1.29)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--dropout_prob', type=float, default=0.0)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--use_8bit_adam', action='store_true', default=False)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--scale_lr', action='store_true', default=False)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--variant', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--report_to', type=str, default="tensorboard") # Changed from None to tensorboard

    # Hub settings
    parser.add_argument('--push_to_hub', action='store_true', default=False)
    parser.add_argument('--hub_token', type=str, default=None)
    parser.add_argument('--hub_model_id', type=str, default=None)

    # Add a new argument to point to the local beta model if it exists
    parser.add_argument('--local_beta_model_path', type=str, default="/home/sheldon/cvton/catvton-flux-beta", help="Path to the local beta model.")


    return parser.parse_args()

def modify_train_script_for_deepfashion(args):
    """Monkey patch the train script to use DeepFashion dataset."""
    import train_flux_inpaint
    original_main = train_flux_inpaint.main

    def modified_main(args):
        print("🔄 Modifying training script for DeepFashion dataset...")
        original_dataset_class = getattr(train_flux_inpaint, 'VitonHDTestDataset', None)

        def create_deepfashion_dataset(*dataset_args, **dataset_kwargs):
            phase = dataset_kwargs.get('phase', 'train')
            size = (args.height, args.width)
            max_samples = args.max_train_samples if phase == 'train' else args.max_validation_samples
            print(f"🔄 Creating DeepFashion dataset: phase={phase}, size={size}, max_samples={max_samples or 'all'}")
            return DeepFashionDataset(phase=phase, size=size, max_samples=max_samples)

        train_flux_inpaint.VitonHDTestDataset = create_deepfashion_dataset

        try:
            return original_main(args)
        finally:
            if original_dataset_class:
                train_flux_inpaint.VitonHDTestDataset = original_dataset_class

    train_flux_inpaint.main = modified_main
    return train_flux_inpaint.main

def main():
    args = create_training_args()

    print("🚀 Starting DeepFashion LoRA Training...")
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU.")
        return

    print(f"✅ GPU available: {torch.cuda.get_device_name()}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # --- Model Path Logic ---
    # The base model path for tokenizers, VAE, etc.
    base_model_path = "black-forest-labs/FLUX.1-dev"
    
    # The transformer model path, starting with the public alpha model.
    inpaint_model_path = "xiaozaa/catvton-flux-alpha"

    # If a local beta model exists, use it as the starting point for the transformer.
    if os.path.exists(args.local_beta_model_path):
        print(f"✅ Found local beta model at {args.local_beta_model_path}. Using it for the transformer.")
        # The transformer weights are in a 'transformer' subdirectory of the beta model.
        local_transformer_path = os.path.join(args.local_beta_model_path, "transformer")
        if os.path.exists(local_transformer_path):
            inpaint_model_path = local_transformer_path
        else:
            # If the 'transformer' subdirectory doesn't exist, use the beta path directly.
            inpaint_model_path = args.local_beta_model_path
    else:
        print(f"Could not find local beta model. Using public alpha model '{inpaint_model_path}' for the transformer.")

    # Set the arguments correctly before passing them to the training function.
    # This is the crucial fix: ensuring the tokenizer is loaded from the base model,
    # and the transformer is loaded from the separate fine-tuned model path.
    args.pretrained_model_name_or_path = base_model_path
    args.pretrained_inpaint_model_name_or_path = inpaint_model_path


    print(f"📁 Output directory: {args.output_dir}")
    print(f"🏷️  Base Model (for tokenizers): {args.pretrained_model_name_or_path}")
    print(f"🎯 Transformer Model (for fine-tuning): {args.pretrained_inpaint_model_name_or_path}")
    print(f"⚙️  Training steps: {args.max_train_steps}")
    print(f"📏 Image size: {args.height}x{args.width*2} (concatenated)")
    print(f"📦 Effective batch size: {args.train_batch_size * args.gradient_accumulation_steps}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    modified_main_func = modify_train_script_for_deepfashion(args)

    try:
        print("🔥 Starting training...")
        modified_main_func(args)
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
