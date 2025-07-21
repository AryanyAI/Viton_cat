# CatVTON-Flux: Lower-Body and Full-Body Training Strategy

## 1. Objective

The primary strategic objective is to extend the capabilities of the existing CatVTON-Flux model beyond its pre-trained domain of upper-body garments. The goal is to create a comprehensive, state-of-the-art virtual try-on solution that can accurately handle **lower-body** (e.g., pants, skirts) and **full-body** (e.g., dresses, jumpsuits) clothing items. This will position the model to compete with and surpass leading industry solutions.

## 2. Core Problem

The base model, `catvton-flux-alpha`, was pre-trained on the VITON-HD dataset. While this dataset provides high-quality data, it is exclusively focused on upper-body garments. As a result, the model currently lacks the knowledge to process and render other types of clothing accurately.

## 3. Solution: Fine-Tuning with a Diverse Dataset

The solution is to fine-tune the pre-trained model using a more diverse dataset that includes the target garment types.

### 3.1. Selected Dataset

**Dataset:** `SaffalPoosh/deepFashion-with-masks` on Hugging Face.

**Rationale:** This dataset is ideal for our objective for the following reasons:
- **Variety:** It contains a wide range of garment types, explicitly labeled as `cloth_type: 1` (upper-body), `2` (lower-body), and `3` (full-body). This provides the necessary examples to teach the model new concepts.
- **Compatibility:** The dataset provides all the required inputs for our training pipeline:
    - `image`: The model photograph.
    - `cloth`: The isolated garment image.
    - `mask`: The segmentation mask for the garment area.
    - `caption`: A descriptive text caption for the garment.
- **Accessibility:** It can be loaded directly and automatically within the training script via the `datasets` library, requiring no manual download or setup.

### 3.2. Implementation Strategy

The implementation will follow best practices to ensure stability and maintain the integrity of the original, working code.

- **No Modification to Core Training Script:** The original `train_flux_inpaint.py` script will **not** be modified. Its logic is proven and robust.
- **Custom Dataset Adapter:** A new script, `deepfashion_training/deepfashion_dataset.py`, acts as an adapter. It loads the `deepFashion-with-masks` dataset and formats it into the exact dictionary structure that `train_flux_inpaint.py` expects.
- **Dynamic Loading:** The training is initiated by a wrapper script (`deepfashion_training/train_deepfashion_lora.py`) that dynamically "monkey-patches" the training pipeline, telling it to use our `DeepFashionDataset` instead of the default `VitonHDTestDataset`.

This approach is clean, safe, and allows us to leverage the power of the original training code without introducing risk.

## 4. Training Plan

The training will be conducted using the unified shell script `deepfashion_training/train_deepfashion_lora.sh`, which supports multiple scenarios.

1.  **Quick Test Run (`--quick-test` flag):**
    - **Purpose:** A 20-30 minute, low-cost run to verify the end-to-end pipeline is working correctly before committing to a full training session.
    - **Action:** Trains on a small subset of the data for a limited number of steps.

2.  **Full LoRA Fine-Tuning:**
    - **Purpose:** The main training run to produce the final, high-quality LoRA weights.
    - **Action:** Trains on the complete `deepFashion-with-masks` dataset for a significant number of steps.

3.  **H100 Optimized Run (`--h100` flag):**
    - **Purpose:** A configuration optimized for a high-performance H100 GPU.
    - **Action:** Uses `bfloat16` precision and a larger batch size to maximize hardware utilization.

By following this documented strategy, we will systematically and safely extend the model's capabilities to achieve our project goals.
