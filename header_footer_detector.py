# Header Footer Detection System
# Complete setup for beginners

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class HeaderFooterDetector:
    def __init__(self, model_path=None):
        """Initialize the detector"""
        self.model = None
        self.model_path = model_path
        self.class_names = ['Banner', 'Footer', 'Header']
        
    def setup_dataset_structure(self, base_path="dataset"):
        """Create the required folder structure for training"""
        folders = [
            f"{base_path}/images/train",
            f"{base_path}/images/val", 
            f"{base_path}/images/test",
            f"{base_path}/labels/train",
            f"{base_path}/labels/val",
            f"{base_path}/labels/test"
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        data_yaml = {
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 3,
            'names': ['Banner', 'Footer', 'Header']
        }
        
        with open(f'{base_path}/data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
            
        print(f"Dataset structure created at: {base_path}")
        return base_path
    
    def convert_coco_to_yolo(self, coco_json_path, images_dir, output_dir):
        """Convert COCO format to YOLO format"""
        import json
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create directories
        Path(f"{output_dir}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/labels").mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for image_info in coco_data['images']:
            image_id = image_info['id']
            image_name = image_info['file_name']
            image_width = image_info['width']
            image_height = image_info['height']
            
            # Copy image file
            src_image = os.path.join(images_dir, image_name)
            dst_image = os.path.join(f"{output_dir}/images", image_name)
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            
            # Convert annotations
            yolo_annotations = []
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    # Convert COCO bbox to YOLO format
                    x, y, w, h = annotation['bbox']
                    x_center = (x + w/2) / image_width
                    y_center = (y + h/2) / image_height
                    width = w / image_width
                    height = h / image_height
                    
                    class_id = annotation['category_id'] - 1  # COCO is 1-indexed
                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            # Save YOLO format label file
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(f"{output_dir}/labels", label_name)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"COCO to YOLO conversion completed. Output saved to: {output_dir}")
    
    def prepare_training_data(self, images_dir, labels_dir, dataset_path="dataset"):
        """Prepare training data by splitting into train/val/test"""
        import random
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(images_dir).glob(ext))
        
        # Shuffle and split
        random.shuffle(image_files)
        total_images = len(image_files)
        
        train_split = int(0.7 * total_images)  # 70% for training
        val_split = int(0.2 * total_images)    # 20% for validation
        # Remaining 10% for test
        
        splits = {
            'train': image_files[:train_split],
            'val': image_files[train_split:train_split + val_split],
            'test': image_files[train_split + val_split:]
        }
        
        # Copy files to respective directories
        for split_name, files in splits.items():
            for img_file in files:
                # Copy image
                dst_img = f"{dataset_path}/images/{split_name}/{img_file.name}"
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label file
                label_file = Path(labels_dir) / f"{img_file.stem}.txt"
                if label_file.exists():
                    dst_label = f"{dataset_path}/labels/{split_name}/{label_file.name}"
                    shutil.copy2(label_file, dst_label)
        
        print(f"Data split completed:")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val: {len(splits['val'])} images") 
        print(f"  Test: {len(splits['test'])} images")
    
    def train_model(self, dataset_path="dataset", epochs=100, img_size=640):
        """Train the YOLO model"""
        print("Starting model training...")
        
        # Load a pre-trained YOLOv8 model
        model = YOLO('yolov8n.pt')  # nano version for faster training
        
        # Train the model
        results = model.train(
            data=f'{dataset_path}/data.yaml',
            epochs=epochs,
            imgsz=img_size,
            patience=50,
            batch=16,
            workers=4
        )
        
        # Save the trained model
        model.save('header_footer_model.pt')
        self.model = model
        print("Training completed! Model saved as 'header_footer_model.pt'")
        
        return results
    
    def load_model(self, model_path="header_footer_model.pt"):
        """Load a trained model"""
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Model loaded from: {model_path}")
        else:
            print(f"Model file not found: {model_path}")
    
    def detect_header_footer(self, image_path, confidence_threshold=0.5):
        """Detect headers and footers in an image"""
        if self.model is None:
            print("No model loaded. Please train or load a model first.")
            return None
        
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return detections
    
    def visualize_detections(self, image_path, detections, save_path=None):
        """Visualize detections on the image"""
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_name = det['class']
            confidence = det['confidence']
            
            # Choose color based on class
            colors = {'Banner': (255, 0, 0), 'Footer': (0, 255, 0), 'Header': (0, 0, 255)}  # Red, Green, Blue
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display or save
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to: {save_path}")
        
        return image_rgb
    
    def batch_process(self, images_folder, output_folder=None):
        """Process multiple images"""
        if output_folder:
            Path(output_folder).mkdir(exist_ok=True)
        
        results = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for image_file in Path(images_folder).iterdir():
            if image_file.suffix.lower() in image_extensions:
                print(f"Processing: {image_file.name}")
                
                # Detect headers and footers
                detections = self.detect_header_footer(str(image_file))
                results[image_file.name] = detections
                
                # Save visualization if output folder specified
                if output_folder and detections:
                    output_path = Path(output_folder) / f"detected_{image_file.name}"
                    self.visualize_detections(str(image_file), detections, str(output_path))
        
        return results

# Example usage and setup instructions
def main():
    """Main function demonstrating how to use the system"""
    detector = HeaderFooterDetector()
    
    print("=== Header Footer Detection System ===")
    print("Follow these steps:")
    print("\n1. Setup dataset structure")
    dataset_path = detector.setup_dataset_structure()
    
    print("\n2. If you have COCO format data, convert it:")
    print("   detector.convert_coco_to_yolo('path/to/coco.json', 'path/to/images', 'converted_data')")
    
    print("\n3. Prepare your training data:")
    print("   detector.prepare_training_data('path/to/images', 'path/to/labels', dataset_path)")
    
    print("\n4. Train the model:")
    print("   detector.train_model(dataset_path, epochs=100)")
    
    print("\n5. Load trained model and detect:")
    print("   detector.load_model('header_footer_model.pt')")
    print("   detections = detector.detect_header_footer('path/to/image.jpg')")
    
    print("\n6. Process multiple images:")
    print("   results = detector.batch_process('input_images/', 'output_results/')")
    
    print("\nFor your 32 images, you can start by running:")
    print("python this_script.py")

if __name__ == "__main__":
    main()