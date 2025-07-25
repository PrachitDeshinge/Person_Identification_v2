import torch
from u2netcopy import U2NET

def check_compatibility(model_path, weights_path):
    try:
        # Load the model architecture
        net = U2NET(3, 1)
        model_state_dict = net.state_dict()
        
        # Load the pre-trained weights
        weights_state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        
        # Add missing 'num_batches_tracked' keys
        for key in model_state_dict:
            if key.endswith('num_batches_tracked') and key not in weights_state_dict:
                weights_state_dict[key] = model_state_dict[key]
        
        # Now, try to load the state dict
        net.load_state_dict(weights_state_dict)
        
        # If we are here, it worked.
        print("The model and weights are compatible.")
        print("Successfully loaded weights into the model.")

        # For verbosity, let's still print the key differences after fixing them.
        model_keys = set(net.state_dict().keys())
        weights_keys = set(weights_state_dict.keys())
        
        missing_keys = model_keys - weights_keys
        unexpected_keys = weights_keys - model_keys
        
        print(f"Missing keys (after fix): {missing_keys}")
        print(f"Unexpected keys (after fix): {unexpected_keys}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    model_path = 'u2netcopy.py'
    weights_path = '/Users/prachit/Projects/Computer_Vision/Person_Identification/weights/u2net.pth'
    check_compatibility(model_path, weights_path)