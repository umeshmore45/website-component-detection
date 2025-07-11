 # Simple script to run with your data
# Save this as 'run_my_detection.py'

import sys
import os
from pathlib import Path
import cv2

# Add the detector to path
sys.path.append('.')

# Import our detector
from header_footer_detector import HeaderFooterDetector

def main():
    print("🚀 Starting Header/Footer/Banner Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = HeaderFooterDetector()
    
    # Check if folders exist
    if not os.path.exists('original_images'):
        print("❌ Error: 'original_images' folder not found!")
        print("Please create 'original_images' folder and put your 32 images there.")
        return
    
    if not os.path.exists('original_labels'):
        print("❌ Error: 'original_labels' folder not found!")
        print("Please create 'original_labels' folder and put your .txt label files there.")
        return
    
    # Count files
    image_count = len(list(Path('original_images').glob('*')))
    label_count = len(list(Path('original_labels').glob('*.txt')))
    
    print(f"📁 Found {image_count} images and {label_count} label files")
    
    if image_count == 0:
        print("❌ No images found in 'original_images' folder!")
        return
    
    if label_count == 0:
        print("❌ No label files found in 'original_labels' folder!")
        return
    
    # Step 1: Setup dataset structure
    print("\n📋 Step 1: Setting up dataset structure...")
    dataset_path = detector.setup_dataset_structure("my_dataset")
    
    # Step 2: Prepare training data
    print("\n📋 Step 2: Preparing training data...")
    detector.prepare_training_data(
        'original_images',
        'original_labels',
        dataset_path
    )
    
    # Step 3: Train the model
    print("\n📋 Step 3: Training the model...")
    print("This may take 10-30 minutes depending on your computer...")
    
    try:
        detector.train_model(dataset_path, epochs=100)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 4: Test the model
    print("\n📋 Step 4: Testing the trained model...")
    
    # Load the trained model
    detector.load_model('header_footer_model.pt')
    
    # Test on one image first
    test_images = list(Path('original_images').glob('*'))
    if test_images:
        test_image = str(test_images[0])
        print(f"Testing on: {test_image}")
        
        detections = detector.detect_header_footer(test_image)
        
        if detections:
            print(f"✅ Found {len(detections)} detections:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class']} (confidence: {det['confidence']:.2f})")
            
            # Create visualization
            detector.visualize_detections(test_image, detections, 'test_result.jpg')
            print("🖼️  Test result saved as 'test_result.jpg'")
        else:
            print("⚠️  No detections found in test image")
    
    # Step 5: Process all images
    print("\n📋 Step 5: Processing all images...")
    
    # Create results folder
    Path('results').mkdir(exist_ok=True)
    
    results = detector.batch_process('original_images', 'results')
    
    # Print summary
    total_detections = sum(len(dets) for dets in results.values())
    print(f"\n🎉 Processing completed!")
    print(f"📊 Summary:")
    print(f"   - Images processed: {len(results)}")
    print(f"   - Total detections: {total_detections}")
    print(f"   - Results saved in: 'results/' folder")
    
    # Count by class
    class_counts = {'Banner': 0, 'Footer': 0, 'Header': 0}
    for dets in results.values():
        for det in dets:
            class_counts[det['class']] += 1
    
    print(f"   - Banners found: {class_counts['Banner']}")
    print(f"   - Footers found: {class_counts['Footer']}")
    print(f"   - Headers found: {class_counts['Header']}")
    
    print(f"\n✅ All done! Check the 'results' folder for your detected images.")

def crop_by_class(image_path, detections, output_dir="crops"):
    image_name = Path(image_path).stem
    image_folder = Path(output_dir) / image_name
    image_folder.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(image_path)
    for i, det in enumerate(detections):
        class_name = det['class']
        if class_name in ['Banner', 'Header']:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            crop = image[y1:y2, x1:x2]
            crop_path = image_folder / f"{class_name}_{i+1}.jpg"
            cv2.imwrite(str(crop_path), crop)
            print(f"Cropped {class_name} saved to: {crop_path}")

def detect_on_new_image(image_path):
    detector = HeaderFooterDetector()
    detector.load_model('header_footer_model.pt')
    detections = detector.detect_header_footer(image_path, confidence_threshold=0.5)
    if detections:
        print(f"✅ Found {len(detections)} detections in {image_path}:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class']} (confidence: {det['confidence']:.2f})")
        detector.visualize_detections(image_path, detections, 'new_image_result.jpg')
        print("🖼️  Result saved as 'new_image_result.jpg'")
        crop_by_class(image_path, detections)
    else:
        print("⚠️  No detections found in the new image.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detect_on_new_image(image_path)
    else:
        main()