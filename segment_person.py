import argparse
from models.hrnet import segment_person

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person Segmentation using HRNet')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights.')
    parser.add_argument('--output', type=str, default='segmentation_mask.png', help='Path to save the output segmentation mask.')

    args = parser.parse_args()

    # Perform segmentation
    segmentation_mask = segment_person(args.image, args.weights)

    # Save the output
    segmentation_mask.save(args.output)

    print(f'Segmentation mask saved to {args.output}')
