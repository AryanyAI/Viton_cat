# CatVTON-Flux: Complete Training Guide

## 1. Project Overview and Goal

**Objective:** To extend the capabilities of the CatVTON-Flux model beyond its pre-trained domain of upper-body garments. The strategic goal is to create a state-of-the-art virtual try-on solution that accurately handles **lower-body** (e.g., pants, skirts) and **full-body** (e.g., dresses, jumpsuits) clothing.

**Core Strategy:** We will achieve this by fine-tuning the existing, high-performance model using LoRA (Low-Rank Adaptation) on a diverse dataset that includes all garment types. This approach is cost-effective and leverages the power of the pre-trained model.

---

## 2. The Training Pipeline: How It Works

Our training setup is designed to be robust and safe, preserving the original author's proven code while adapting it for our specific needs.

1.  **Base Model (`black-forest-labs/FLUX.1-dev`):** This is the foundational model. It contains all the core components, including the essential **tokenizers**, VAE, and text encoders.
2.  **Fine-Tuned Transformer (`xiaozaa/catvton-flux-alpha`):** This is the specialized transformer from the original CatVTON-Flux author, which has been pre-trained for virtual try-on. We use this as the starting point for our own fine-tuning.
3.  **Dataset (`SaffalPoosh/deepFashion-with-masks`):** This is our new, diverse dataset. It is loaded automatically from Hugging Face and contains the required images (`image`, `cloth`), `mask`, and `caption` for training on all garment types.
4.  **Training Scripts:**
    *   `train_deepfashion_lora.sh`: The main entry point. A shell script that sets up the training environment and parameters.
    *   `deepfashion_training/train_deepfashion_lora.py`: The primary Python script that orchestrates the training process. It correctly loads the base model and the fine-tuned transformer.
    *   `deepfashion_training/deepfashion_dataset.py`: A Python module that loads and prepares our custom dataset.
    *   `train_flux_inpaint.py`: The **original, unmodified** training script from the author. We use its powerful, proven logic to run the actual training steps.

---

## 3. How to Run Training

All training operations are managed through the `train_deepfashion_lora.sh` script.

### 3.1. Quick Test Run (Recommended First Step)

This is a 20-30 minute, low-cost run to verify that the entire pipeline is working correctly before committing to a full, expensive training session.

**Command:**
```bash
bash ./train_deepfashion_lora.sh --quick-test
```

### 3.2. Full LoRA Fine-Tuning

This is the main training run to produce the final, high-quality LoRA weights for all garment types. This will be a long and resource-intensive process.

**Command:**
```bash
bash ./train_deepfashion_lora.sh
```

### 3.3. H100 Optimized Full Run

This command uses settings optimized for a high-performance H100 GPU, such as a larger batch size and `bfloat16` precision, to maximize hardware utilization.

**Command:**
```bash
bash ./train_deepfashion_lora.sh --h100
```

---

## 4. Output and Trained Weights

*   All training outputs, including checkpoints and validation images, will be saved in a timestamped directory inside `/home/sheldon/cvton/catvton-flux/runs/`.
*   The final trained LoRA weights will be located within the `.../transformer` subdirectory of the last checkpoint. These weights can be loaded by the inference scripts (`app_lora.py`, `tryon_inference_lora.py`) for testing the final model.
