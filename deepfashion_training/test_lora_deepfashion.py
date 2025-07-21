#!/usr/bin/env python3
"""
Quick LoRA Test Script for DeepFashion + CatVTON-Flux
20-30 minute validation before full training investment
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfashion_dataset import DeepFashionDataset
from train_flux_inpaint import main


def create_test_args():
    """Create arguments for quick LoRA test"""
    
    args = argparse.Namespace()
    
    # Model paths
    args.pretrained_model_name_or_path = "/home/sheldon/cvton/catvton-flux-beta"
    args.pretrained_inpaint_model_name_or_path = "/home/sheldon/cvton/catvton-flux-beta/transformer"
    
    # Fallback to online model if local not found
    if not os.path.exists(args.pretrained_model_name_or_path):
        print("Local model not found, using online model...")
        args.pretrained_model_name_or_path = "xiaozaa/catvton-flux-alpha"
        args.pretrained_inpaint_model_name_or_path = "xiaozaa/catvton-flux-alpha"
    
    # Dataset settings
    args.dataroot = None  # Not used with DeepFashion
    args.train_data_list = None
    args.train_verification_list = None  
    args.validation_data_list = None
    
    # Quick test parameters
    args.max_train_steps = 50  # Very short for quick test
    args.num_train_epochs = 1
    args.train_batch_size = 1  # Small batch for memory efficiency
    args.gradient_accumulation_steps = 4  # Effective batch size = 4
    
    # LoRA parameters
    args.train_base_model = True  # Enable training
    args.learning_rate = 1e-4  # Conservative learning rate
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 5
    args.lr_num_cycles = 1
    args.lr_power = 1.0
    
    # Memory and performance
    args.mixed_precision = "bf16"  # Use bf16 for L40S
    args.gradient_checkpointing = True
    args.allow_tf32 = True
    args.max_grad_norm = 1.0
    
    # Validation
    args.validation_steps = 25  # Validate halfway through
    args.checkpointing_steps = 50  # Save at end
    args.checkpoints_total_limit = 2
    
    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"/home/sheldon/cvton/catvton-flux/deepfashion_training/test_run_{timestamp}"
    args.logging_dir = "logs"
    
    # Image settings
    args.height = 512
    args.width = 384  # Will be doubled to 768 for concatenation
    args.max_sequence_length = 512
    
    # Training parameters
    args.weighting_scheme = "logit_normal"
    args.logit_mean = 0.0
    args.logit_std = 1.0
    args.mode_scale = 1.29
    args.guidance_scale = 3.5
    args.dropout_prob = 0.0
    
    # Optimizer
    args.optimizer = "adamw"
    args.use_8bit_adam = False
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_weight_decay = 1e-2
    args.adam_epsilon = 1e-8
    args.scale_lr = False
    
    # Prodigy (not used)
    args.prodigy_beta3 = None
    args.prodigy_decouple = None
    args.prodigy_use_bias_correction = None
    args.prodigy_safeguard_warmup = None
    
    # Misc
    args.seed = 42
    args.revision = None
    args.variant = None
    args.resume_from_checkpoint = None
    
    # Hub settings
    args.push_to_hub = False
    args.hub_token = None
    args.hub_model_id = None
    args.report_to = None  # Disable wandb for quick test
    
    return args


def modify_train_script_for_deepfashion():
    """
    Monkey patch the train script to use DeepFashion dataset
    """
    import train_flux_inpaint
    original_main = train_flux_inpaint.main
    
    def modified_main(args):
        # Replace dataset creation with DeepFashion
        print("🔄 Modifying training script for DeepFashion dataset...")
        
        # Store original dataset creation
        original_dataset_class = train_flux_inpaint.VitonHDTestDataset
        
        # Replace with DeepFashion dataset
        def create_deepfashion_dataset(*dataset_args, **dataset_kwargs):
            phase = dataset_kwargs.get('phase', 'train')
            size = (args.height, args.width)
            max_samples = 100 if phase == 'train' else 20  # Limit for quick test
            
            print(f"🔄 Creating DeepFashion dataset: phase={phase}, size={size}, max_samples={max_samples}")
            return DeepFashionDataset(phase=phase, size=size, max_samples=max_samples)
        
        # Monkey patch
        train_flux_inpaint.VitonHDTestDataset = create_deepfashion_dataset
        
        try:
            # Run original main with our modifications
            return original_main(args)
        finally:
            # Restore original
            train_flux_inpaint.VitonHDTestDataset = original_dataset_class
    
    train_flux_inpaint.main = modified_main


def main():
    print("🚀 Starting DeepFashion LoRA Test Training...")
    print("📊 This will take approximately 20-30 minutes")
    print("💾 Testing with 100 training samples, 50 steps total")
    print()
    
    # Check environment
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires GPU.")
        return
        
    print(f"✅ GPU available: {torch.cuda.get_device_name()}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Create test arguments
    args = create_test_args()
    
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🏷️  Model: {args.pretrained_model_name_or_path}")
    print(f"⚙️  Training steps: {args.max_train_steps}")
    print(f"📏 Image size: {args.height}x{args.width*2} (concatenated)")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Modify training script for DeepFashion
    modify_train_script_for_deepfashion()
    
    try:
        print("🔥 Starting training...")
        # FIX: Call the main function from the imported training script, not the local main function
        main_result = train_flux_inpaint.main(args)

        print()
        print("✅ Test training completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print()
        print("📋 NEXT STEPS:")
        print("1. Check the generated images in the output directory")
        print("2. If quality looks good, run full training:")
        print("   ./train_deepfashion_lora.sh")
        print("3. If issues found, adjust parameters and re-test")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)