# I want to check if the current model architecture matches the pretrained weights

# --- Refactoring Note ---
# Assuming 'u2net.py' contains the U2NET model architecture from our previous steps.
# If your file is named 'u2net_model.py', change the import accordingly.
from models.u2net import U2NET 

import torch
import cv2
import numpy as np
import logging
import torchvision.transforms as transforms
from PIL import Image
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_silhouette_extractor(weights_path: str | None = None):
    """
    Creates and prepares a U²-Net model for silhouette extraction.
    This function now also handles moving the model to the correct device.
    """
    try:
        # 1. Determine the appropriate device from central config
        device = config.DEVICE
        logger.info(f"Using device: {device.upper()}")

        # 2. Load the U²-Net model architecture
        model = U2NET(in_ch=3, out_ch=1) # Standard U2NET for segmentation has 1 output channel
        logger.info("U²-Net model architecture loaded successfully.")
        
        # 3. Load the pretrained weights
        # We load to the CPU first, then move the whole model to the target device.
        model.load_state_dict(torch.load(weights_path or config.U2NET_WEIGHTS, map_location='cpu'))
        logger.info(f"Pretrained weights loaded from {weights_path}.")
        
        # --- FIX: Move the entire model to the selected device ---
        model.to(device)
        
        # 4. Set the model to evaluation mode
        model.eval()
        logger.info("Model set to evaluation mode and moved to device.")
        
        # Return both the model and the device it's on
        return model, device

    except Exception as e:
        logger.error(f"Failed to create silhouette extractor: {e}")
        raise e

def extract_silhouette(model, device, input_image):
    """Extract silhouette from the input image using the prepared U²-Net model."""
    try:
        # --- IMPROVEMENT: Use proper preprocessing as expected by the model ---
        # The model was trained on images normalized with these specific values.
        # Not using them will lead to poor results.
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        original_height, original_width = input_image.shape[:2]
        
        # Apply the preprocessing
        input_tensor = preprocess(input_image)
        
        # Add a batch dimension (B, C, H, W) and move the tensor to the correct device
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Forward pass through the model
        with torch.no_grad():
            # The model returns multiple outputs, we use the first one (d0)
            d0, _, _, _, _, _, _ = model(input_tensor)
        
        # Post-process the output
        pred = d0[:,0,:,:] # Get the primary segmentation map
        pred = torch.sigmoid(pred) # Apply sigmoid to get probabilities
        
        # Normalize the prediction to a 0-255 range
        pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
        
        # Convert to a NumPy array and resize
        silhouette_raw = pred.squeeze().cpu().numpy()
        silhouette_norm = (silhouette_raw * 255).astype(np.uint8)
        
        # Resize back to original image size
        silhouette = cv2.resize(silhouette_norm, (original_width, original_height))
        
        return silhouette
    except Exception as e:
        logger.error(f"Error during silhouette extraction: {e}")
        raise e
    
if __name__ == "__main__":
    try:
        # Create the silhouette extractor, which now returns the model and device
        extractor, device = create_silhouette_extractor()
        
        # Load a test image
        # Ensure the image is converted to RGB format, as OpenCV loads in BGR
        bgr_image = cv2.imread('./imgs/person_crop_2.jpg')
        if bgr_image is None:
            raise FileNotFoundError("Test image not found. Please check the path './person_crop_2.jpg'")
        input_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Extract silhouette by passing the model, device, and image
        silhouette_mask = extract_silhouette(extractor, device, input_image)
        
        # Save the output mask to a file to verify it
        output_filename = "test_silhouette_output.png"
        cv2.imwrite(output_filename, silhouette_mask)
        print(f"\n✅ Silhouette mask saved to '{output_filename}'")
        
        # Display the input and output images using OpenCV
        cv2.imshow("Input Image", bgr_image) # Show the original BGR image
        cv2.imshow("Silhouette Mask", silhouette_mask)
        print("\nDisplaying images. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")