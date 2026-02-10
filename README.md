# Website Component Detection

A YOLO-based machine learning system for automatically detecting and classifying website components (Headers, Footers, and Banners) in screenshots.

## Overview

This project uses YOLOv8 to detect and classify common website components in screenshots or images. It can identify:
- **Headers** - Top navigation bars and header sections
- **Footers** - Bottom footer sections with links and information
- **Banners** - Promotional banners and call-to-action sections

## Features

- 🤖 Train custom YOLO models on your own dataset
- 🔍 Detect components in single images or batch process multiple images
- 📊 Visualize detections with bounding boxes and labels
- ✂️ Automatically crop detected components
- 📈 Comprehensive training metrics and validation

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install ultralytics opencv-python numpy pyyaml matplotlib pillow
```

## Project Structure

```
website-component-detection/
├── run_my_detection.py           # Main script for training and detection
├── header_footer_detector.py     # Core detector class
├── requirements.txt              # Python dependencies
├── original_images/              # Training images (you provide)
├── original_labels/              # Training labels in YOLO format (you provide)
├── results/                      # Detection results output
├── crops/                        # Cropped detected components
└── header_footer_model.pt        # Trained model (generated)
```

## Usage

### Mode 1: Training a New Model

If you have training data (images and YOLO format labels):

1. **Prepare your data:**
   - Create `original_images/` folder and add your training images
   - Create `original_labels/` folder and add corresponding `.txt` label files

2. **Run training:**
   ```bash
   python3 run_my_detection.py
   ```

This will:
- Set up the dataset structure
- Train a YOLO model (takes 10-30 minutes)
- Validate the model
- Process all images and save results to `results/` folder

### Mode 2: Detect Components in a New Image

If you already have a trained model (`header_footer_model.pt`):

```bash
python3 run_my_detection.py path/to/your/screenshot.png
```

**Example:**
```bash
python3 run_my_detection.py "Screenshot 2026-02-10 at 11.36.55 AM.png"
```

This will:
- Load the pre-trained model
- Detect headers, footers, and banners
- Save visualization as `new_image_result.jpg`
- Create cropped images of detected components in `crops/` folder

## Label Format

Labels should be in YOLO format (`.txt` files):

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0 = Banner, 1 = Footer, 2 = Header
- All coordinates are normalized (0-1)

**Example:**
```
2 0.5 0.05 0.9 0.08    # Header at top
1 0.5 0.95 0.9 0.10    # Footer at bottom
0 0.5 0.30 0.8 0.15    # Banner in middle
```

## Output

### Detection Results

- **Visualized images** with bounding boxes saved in `results/` folder
- **Individual crops** of detected components saved in `crops/` folder
- **Console output** with detection statistics and confidence scores

### Example Output

```
✅ Found 3 detections in screenshot.png:
  1. Header (confidence: 0.92)
  2. Banner (confidence: 0.87)
  3. Footer (confidence: 0.89)
🖼️  Result saved as 'new_image_result.jpg'
```

## Training Configuration

Default training parameters (can be modified in `run_my_detection.py`):

- **Epochs**: 100
- **Image size**: 640x640
- **Batch size**: Auto
- **Model**: YOLOv8n (nano - fastest)

## Troubleshooting

### Python Version Issues

If you get `ModuleNotFoundError`, ensure you're using the correct Python version:

```bash
# Check Python version
python3 --version

# Install dependencies for Python 3
python3 -m pip install -r requirements.txt

# Run with Python 3
python3 run_my_detection.py
```

### No Detections Found

- Ensure your model is trained on similar data
- Try lowering the confidence threshold in the script
- Check that your input image is clear and properly formatted

### Training Data Requirements

- Minimum recommended: 30+ images with labels
- More diverse data = better model performance
- Ensure labels are correctly formatted in YOLO format

## Advanced Usage

### Adjust Confidence Threshold

Edit `run_my_detection.py` and modify:

```python
detections = detector.detect_header_footer(image_path, confidence_threshold=0.5)
```

Lower values (0.3-0.4) = more detections but more false positives
Higher values (0.6-0.8) = fewer but more confident detections

### Change Training Parameters

Edit the training call in `run_my_detection.py`:

```python
detector.train_model(dataset_path, epochs=150, img_size=800)
```

## Dependencies

- **ultralytics**: YOLOv8 implementation
- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **pyyaml**: Configuration file handling
- **matplotlib**: Visualization
- **pillow**: Image manipulation

## License

This project is provided as-is for educational and commercial use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
