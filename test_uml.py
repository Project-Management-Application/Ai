import os
import sys
import json
from OCR import UmlTextRecognizer
import torch

def test_uml_extraction(image_paths, output_dir):
    """Test UML extraction on multiple images"""
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'yolo_model_path': 'yolov5/runs/train/uml_yolo2/weights/best.pt',
        'use_gpu_ocr': torch.cuda.is_available(),
        'save_debug_images': True,
        'debug_output_dir': os.path.join(output_dir, 'debug_images'),
        'detection': {
            'img_size': 640,
            'conf_threshold': 0.3,
            'iou_threshold': 0.3
        }
    }

    # Create output directories
    os.makedirs(config['debug_output_dir'], exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the recognizer
    print("Initializing UML Text Recognizer...")
    recognizer = UmlTextRecognizer(config)

    # Process each image
    results = {}
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i + 1}/{len(image_paths)}: {image_path}")

        try:
            # Extract UML data
            uml_data = recognizer.process_image(image_path)

            # Save individual result
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_result.json")

            with open(output_path, 'w') as f:
                json.dump(uml_data, f, indent=4)

            print(f"Result saved to {output_path}")
            results[base_name] = uml_data

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    # Save combined results
    combined_path = os.path.join(output_dir, "all_results.json")
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nAll results saved to {combined_path}")
    return results

if __name__ == "__main__":
    # Get image paths from command line or use defaults
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
        output_dir = "uml_results"
    else:
        # Default test images
        test_dir = "test"
        image_paths = [
            os.path.join(test_dir, "uml_test.png"),
            os.path.join(test_dir, "uml_test2.jpg"),
        ]
        output_dir = "uml_results"

    # Run test
    test_uml_extraction(image_paths, output_dir)