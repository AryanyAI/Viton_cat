#!/usr/bin/env python3
"""
DeepFashion Dataset Adapter for CatVTON-Flux Training
Compatible with existing VitonHDTestDataset infrastructure
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset


class DeepFashionDataset(Dataset):
    """
    DeepFashion dataset adapter that mimics VitonHDTestDataset interface
    Works with SaffalPoosh/deepFashion-with-masks dataset
    """
    
    def __init__(self, dataroot_path=None, phase="train", order="paired", 
                 size=(512, 384), data_list=None, max_samples=None):
        """
        Initialize DeepFashion dataset
        
        Args:
            dataroot_path: Not used, kept for compatibility
            phase: train/test phase
            order: paired/unpaired order
            size: (height, width) tuple
            data_list: Not used, kept for compatibility  
            max_samples: Limit dataset size for testing
        """
        self.phase = phase
        self.order = order
        self.size = size
        self.height, self.width = size
        
        print(f"Loading DeepFashion dataset, phase: {phase}")
        
        # Load dataset from HuggingFace
        try:
            split_name = "train" if phase == "train" else "validation"
            self.dataset = load_dataset("SaffalPoosh/deepFashion-with-masks", split=split_name)

            # Limit samples for testing
            if max_samples:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

            print(f"Loaded {len(self.dataset)} samples from DeepFashion dataset for training on ALL garment types.")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
            
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        """
        Returns data in VitonHDTestDataset compatible format
        """
        try:
            item = self.dataset[index]
            
            # Extract images
            person_image = item['image']  # Person wearing clothes
            garment_image = item['cloth']  # Garment only
            mask_image = item['mask']  # Inpainting mask
            caption = item.get('caption', 'a photo of a piece of clothing') # Use caption from dataset if available

            # Convert PIL to RGB if needed
            if person_image.mode != 'RGB':
                person_image = person_image.convert('RGB')
            if garment_image.mode != 'RGB':
                garment_image = garment_image.convert('RGB')
            if mask_image.mode != 'L':
                mask_image = mask_image.convert('L')
                
            # Apply transforms
            person_tensor = self.transform(person_image)  # [-1, 1]
            garment_tensor = self.transform(garment_image)  # [-1, 1] 
            mask_tensor = self.mask_transform(mask_image)  # [0, 1]
            
            # Create concatenated image (person + garment side by side)
            # This matches the CatVTON training format
            concat_image = torch.cat([garment_tensor, person_tensor], dim=2)  # Concat on width
            
            # Create masked person image for inpainting
            # Mask out the region where garment should be placed
            mask_3d = mask_tensor.repeat(3, 1, 1)  # Convert mask to 3-channel
            masked_person = person_tensor * (1.0 - mask_3d)  # Mask out garment area
            
            # Return in VitonHDTestDataset format
            return {
                'image': concat_image,  # [3, H, 2*W] - garment + person concatenated
                'inpaint_mask': mask_tensor,  # [1, H, W] - where to inpaint
                'im_mask': masked_person,  # [3, H, W] - person with garment area masked
                'cloth_pure': garment_tensor,  # [3, H, W] - pure garment image
                'caption_cloth': caption, # Use the more descriptive caption
                'file_name': f"deepfashion_{index}"
            }

        except Exception as e:
            print(f"Error loading item {index}: {e}")
            # Return a dummy item to prevent training crash
            return self._get_dummy_item()
            
    def _get_garment_type(self):
        """Determine garment type - could be enhanced with actual classification"""
        return "clothing"  # Generic for now
        
    def _get_dummy_item(self):
        """Return dummy item in case of loading error"""
        dummy_image = torch.zeros(3, self.height, self.width)
        dummy_mask = torch.ones(1, self.height, self.width)
        
        return {
            'image': torch.cat([dummy_image, dummy_image], dim=2),
            'inpaint_mask': dummy_mask,
            'im_mask': dummy_image,
            'cloth_pure': dummy_image,
            'caption_cloth': "dummy garment",
            'file_name': "dummy"
        }


# Test the dataset
if __name__ == "__main__":
    print("Testing DeepFashion dataset...")
    
    # Test dataset loading
    dataset = DeepFashionDataset(phase="train", size=(512, 384), max_samples=5)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test item loading
    item = dataset[0]
    print("Sample item keys:", item.keys())
    print("Image shape:", item['image'].shape)
    print("Mask shape:", item['inpaint_mask'].shape)
    
    print("DeepFashion dataset test completed successfully!")