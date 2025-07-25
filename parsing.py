#!/usr/bin/env python3
"""
Human Parsing Model using GaitParsing U²-Net implementation
Based on: https://github.com/wzb-bupt/GaitParsing
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path

# --- Refactoring Note ---
# The U2NET model architecture has been moved to a separate file `u2net_model.py`
# to avoid code duplication. We now import the model from there.
from models.u2net import U2NET

class HumanParsingModel:
    """Human Parsing Model using GaitParsing U²-Net implementation"""
    
    def __init__(self, model_path='../weights/human_parsing.pth', device=None):
        self.input_height = 144
        self.input_width = 96
        self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.model_path = model_path
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # GaitParsing class definitions (7 classes for gait parsing)
        self.get_class_names = [
            'background', 'head', 'body', 'r_arm', 'l_arm', 'r_leg', 'l_leg'
        ]
        
        self.get_class_colors = [
            [0, 0, 0],       # background
            [255, 0, 0],     # head
            [255, 255, 0],   # body  
            [0, 0, 255],     # r_arm
            [255, 0, 255],   # l_arm
            [0, 255, 0],     # r_leg
            [0, 255, 255]    # l_leg
        ]
        
        self.num_classes = len(self.get_class_names)
        self.palette_idx = np.array(self.get_class_colors)
        
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the U²-Net model with pretrained weights"""
        try:
            # Initialize U²-Net with 3 input channels (RGB) and num_classes output channels
            self.model = U2NET(3, self.num_classes)
            
            # Load pretrained weights
            if Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded GaitParsing weights from {self.model_path}")
            else:
                print(f"⚠️  Model weights not found at {self.model_path}")
                print("   Using randomly initialized weights")
            
            self.model.eval()
            
            # Move to device
            self.model.to(self.device)
                
        except Exception as e:
            print(f"❌ Failed to load parsing model: {e}")
            self.model = None

    def is_model_loaded(self):
        """Check if model is properly loaded"""
        return self.model is not None

    def extract_parsing(self, input_images):
        """Extract human parsing masks from input images"""
        if self.model is None:
            print("❌ Model not loaded")
            return []

        try:
            batch_imgs = []
            batch_parsing = []
            
            # Prepare batch
            for i, img in enumerate(input_images):
                # Resize to model input size
                crop_img = cv2.resize(img, (self.input_width, self.input_height), 
                                    interpolation=cv2.INTER_LINEAR)
                
                # Convert to tensor and normalize
                crop_img = torch.from_numpy(crop_img.transpose((2, 0, 1)))
                crop_img = self.transform(crop_img.float().div(255.0))
                crop_img = crop_img.view(1, crop_img.shape[0], crop_img.shape[1], crop_img.shape[2])
                
                if i == 0:
                    batch_imgs = crop_img
                else:
                    batch_imgs = torch.cat((batch_imgs, crop_img), 0)
            
            # Move to device
            batch_imgs = batch_imgs.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs, _, _, _, _, _, _ = self.model(batch_imgs)
                outputs = outputs.cpu()
            
            # Process outputs
            for i, img in enumerate(input_images):
                img_height, img_width = img.shape[:2]
                prediction = outputs[i, :, :, :].data.numpy()
                
                # Convert prediction to colored parsing mask
                result = np.zeros((self.input_height, self.input_width, 3))
                for h in range(prediction.shape[1]):
                    for w in range(prediction.shape[2]):
                        result[h][w] = self.palette_idx[np.argmax(prediction[:, h, w])]
                
                # Resize back to original size
                parsing_im = cv2.resize(result, (img_width, img_height), 
                                      interpolation=cv2.INTER_NEAREST)
                
                # Convert to grayscale class labels for compatibility
                parsing_gray = np.zeros((img_height, img_width), dtype=np.uint8)
                for class_id, color in enumerate(self.get_class_colors):
                    mask = np.all(parsing_im == color, axis=2)
                    parsing_gray[mask] = class_id
                
                batch_parsing.append(parsing_gray)
            
            return batch_parsing
            
        except Exception as e:
            print(f"❌ Parsing extraction failed: {e}")
            # Return empty parsing masks as fallback
            return [np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) for img in input_images]

    def get_class_names_list(self):
        """Get list of class names"""
        return self.get_class_names.copy()

    def get_num_classes(self):
        """Get number of classes"""
        return self.num_classes

if __name__ == "__main__":
    # Simple test
    parser = HumanParsingModel()
    print(f"Model loaded: {parser.is_model_loaded()}")
    print(f"Number of classes: {parser.get_num_classes()}")
    print(f"Class names: {parser.get_class_names_list()}")