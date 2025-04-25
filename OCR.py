import os
import sys

# Add YOLOv5 directory to path
sys.path.insert(0, 'yolov5')

# Now import the other modules
import json
import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR

# Import YOLOv5 utilities with explicit relative imports
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.models.experimental import attempt_load


class UmlTextRecognizer:
    def __init__(self, config):
        self.config = config

        # Initialize YOLO detector
        print("Initializing YOLOv5 model...")
        self.device = select_device(config['device'])
        self.yolo_model = attempt_load(config['yolo_model_path'], device=self.device)
        self.yolo_model.eval()

        # Initialize PaddleOCR
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=config['use_gpu_ocr'])

        print("All models initialized successfully")

    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1['x_min'], box1['y_min'], box1['x_max'], box1['y_max']
        x2_min, y2_min, x2_max, y2_max = box2['x_min'], box2['y_min'], box2['x_max'], box2['y_max']

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def merge_overlapping_boxes(self, elements, iou_threshold=0.3):
        merged = []
        while elements:
            box1 = elements.pop(0)
            i = 0
            while i < len(elements):
                box2 = elements[i]
                if self.compute_iou(box1, box2) > iou_threshold:
                    if box2['conf'] > box1['conf']:
                        box1 = box2
                    elements.pop(i)
                else:
                    i += 1
            merged.append(box1)
        return merged

    def process_image(self, image_path):
        """Main function to process a UML diagram image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Run YOLO detection to find actors and use cases
        actors, use_cases = self._detect_elements(img)

        # Enhanced text extraction for each element
        actors_with_text = self._process_elements(img, actors, "Actor")
        use_cases_with_text = self._process_elements(img, use_cases, "UseCase")

        # Merge overlapping use cases
        use_cases_with_text = self.merge_overlapping_boxes(use_cases_with_text, iou_threshold=0.3)

        # Structure the output
        result = {
            'actors': [actor['text'] for actor in actors_with_text if actor['text']],
            'use_cases': [use_case['text'] for use_case in use_cases_with_text if use_case['text']],
            'relationships': []
        }

        # Save debug visualization if enabled
        if self.config['save_debug_images']:
            self._save_debug_visualization(img, actors_with_text, use_cases_with_text, image_path)

        return result

    def _detect_elements(self, img):
        """Detect actors and use cases using YOLOv5"""
        # Prepare image for YOLOv5 inference
        img0 = img.copy()
        img_size = self.config['detection']['img_size']
        img_processed = letterbox(img, img_size, stride=32, auto=True)[0]
        img_processed = img_processed.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_processed = np.ascontiguousarray(img_processed)
        img_processed = torch.from_numpy(img_processed).to(self.device).float()
        img_processed /= 255.0

        if img_processed.ndimension() == 3:
            img_processed = img_processed.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            pred = self.yolo_model(img_processed)[0]

        # Apply Non-Maximum Suppression
        conf_thres = self.config['detection']['conf_threshold']
        iou_thres = self.config['detection']['iou_threshold']
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

        # Process detections
        actors = []
        use_cases = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img_processed.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in det:
                    x_min, y_min, x_max, y_max = map(int, xyxy)

                    detection = {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'conf': float(conf),
                        'class': int(cls)
                    }

                    if int(cls) == 0:  # Actor
                        actors.append(detection)
                    elif int(cls) == 1:  # Use case
                        use_cases.append(detection)

        return actors, use_cases

    def _process_elements(self, img, elements, element_type):
        """Process detected elements with enhanced text extraction"""
        elements_with_text = []

        for i, elem in enumerate(elements):
            # Extract coordinates
            x_min, y_min, x_max, y_max = elem['x_min'], elem['y_min'], elem['x_max'], elem['y_max']

            # Determine text region based on element type
            if element_type == "Actor":
                # For actors, look below the figure
                text_y_min = y_max
                text_y_max = min(y_max + int((y_max - y_min) * 0.7), img.shape[0])
                text_x_min = max(0, x_min - 10)
                text_x_max = min(x_max + 10, img.shape[1])
            else:  # UseCase
                # For use cases, look inside the oval with padding
                padding_x = int((x_max - x_min) * 0.15)
                padding_y = int((y_max - y_min) * 0.15)
                text_x_min = x_min + padding_x
                text_y_min = y_min + padding_y
                text_x_max = x_max - padding_x
                text_y_max = y_max - padding_y

            # Safety checks for boundaries
            text_x_min = max(0, text_x_min)
            text_y_min = max(0, text_y_min)
            text_x_max = min(img.shape[1], text_x_max)
            text_y_max = min(img.shape[0], text_y_max)

            # Skip if region is too small
            if text_x_max - text_x_min < 5 or text_y_max - text_y_min < 5:
                continue

            # Extract text region
            text_region = img[text_y_min:text_y_max, text_x_min:text_x_max]

            # Extract text with PaddleOCR
            ocr_result = self._extract_text_with_ocr(text_region)

            # Add to results
            elem_with_text = elem.copy()
            elem_with_text['text'] = ocr_result
            elem_with_text['text_region'] = [text_x_min, text_y_min, text_x_max, text_y_max]
            elem_with_text['element_type'] = element_type
            elements_with_text.append(elem_with_text)

        return elements_with_text

    def _extract_text_with_ocr(self, image_region):
        """Extract text using PaddleOCR with enhanced preprocessing"""
        # Skip if region is too small
        if image_region.shape[0] < 5 or image_region.shape[1] < 5:
            return ""

        # Apply preprocessing to enhance text visibility
        scale_factor = 2.0
        resized = cv2.resize(image_region, None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        white_count = np.sum(binary == 255)
        black_count = np.sum(binary == 0)
        if white_count < black_count:
            binary = cv2.bitwise_not(binary)
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Run OCR
        try:
            binary_result = self.ocr.ocr(binary, cls=True)
            original_result = self.ocr.ocr(image_region, cls=True)

            all_texts = []
            if binary_result and len(binary_result) > 0 and binary_result[0]:
                for line in binary_result[0]:
                    if line[1][0]:
                        all_texts.append((line[1][0], line[1][1]))
            if original_result and len(original_result) > 0 and original_result[0]:
                for line in original_result[0]:
                    if line[1][0]:
                        all_texts.append((line[1][0], line[1][1]))

            if all_texts:
                all_texts.sort(key=lambda x: x[1], reverse=True)
                return all_texts[0][0].strip()
            else:
                return ""

        except Exception as e:
            print(f"OCR error: {str(e)}")
            return ""

    def _save_debug_visualization(self, img, actors, use_cases, original_img_path):
        """Save debug visualization of detected elements and extracted text"""
        debug_img = img.copy()

        # Draw actors
        for actor in actors:
            x_min, y_min, x_max, y_max = actor['x_min'], actor['y_min'], actor['x_max'], actor['y_max']
            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            tx_min, ty_min, tx_max, ty_max = actor['text_region']
            cv2.rectangle(debug_img, (tx_min, ty_min), (tx_max, ty_max), (0, 255, 255), 2)
            cv2.putText(debug_img, f"Actor: {actor['text']}",
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw use cases
        for use_case in use_cases:
            x_min, y_min, x_max, y_max = use_case['x_min'], use_case['y_min'], use_case['x_max'], use_case['y_max']
            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            tx_min, ty_min, tx_max, ty_max = use_case['text_region']
            cv2.rectangle(debug_img, (tx_min, ty_min), (tx_max, ty_max), (255, 255, 0), 2)
            cv2.putText(debug_img, f"Use Case: {use_case['text']}",
                        (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save debug image
        base_name = os.path.basename(original_img_path)
        debug_path = os.path.join(self.config['debug_output_dir'], f"debug_{base_name}")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug visualization saved to {debug_path}")


def save_json_results(data, output_path):
    """Save structured UML data as JSON"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"UML data saved to {output_path}")


def main():
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'yolo_model_path': 'yolov5/runs/train/uml_yolo2/weights/best.pt',
        'use_gpu_ocr': torch.cuda.is_available(),
        'save_debug_images': True,
        'debug_output_dir': 'debug_output',
        'detection': {
            'img_size': 640,
            'conf_threshold': 0.3,
            'iou_threshold': 0.3
        }
    }

    # Create output directory if it doesn't exist
    os.makedirs(config['debug_output_dir'], exist_ok=True)

    try:
        # Parse command line arguments
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if len(sys.argv) > 2:
                output_path = sys.argv[2]
            else:
                output_path = 'uml_data.json'
        else:
            # Default paths
            image_path = r'C:\Users\mimou\PycharmProjects\RAG_Ai_Py\test\uml_test3.png'
            output_path = r'C:\Users\mimou\PycharmProjects\RAG_Ai_Py\uml_data.json'

        print(f"Processing UML diagram: {image_path}")

        # Initialize the UML text recognizer
        recognizer = UmlTextRecognizer(config)

        # Process the image
        uml_data = recognizer.process_image(image_path)

        # Save results
        save_json_results(uml_data, output_path)

        # Print summary
        print("\nExtracted UML Data:")
        print(json.dumps(uml_data, indent=4))

        print(f"\nSummary:")
        print(f"- Detected {len(uml_data['actors'])} actors")
        print(f"- Detected {len(uml_data['use_cases'])} use cases")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()