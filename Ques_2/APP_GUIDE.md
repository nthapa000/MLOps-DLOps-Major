# CityScape Segmentation Streamlit App - Deployment Guide

## Overview
This is a **2-page Streamlit web application** that deploys the trained UNet semantic segmentation model with the following features:

### ✨ Features

#### **Page 1: Training Metrics & Model Performance**
- 📊 **Test Set Metrics Display**: Shows the model's mIOU and mDICE scores on the held-out test set
- 📈 **Training Curves**: Displays three comprehensive plots:
  - Training Loss Curve (Cross-Entropy Loss)
  - Training mIOU Curve (Mean Intersection over Union)
  - Training mDICE Curve (Mean Dice Coefficient)
- 📋 **Training Summary**: Provides statistics including:
  - Final training metrics
  - Best metrics achieved during training
  - Loss reduction percentage

#### **Page 2: Semantic Segmentation Demonstration**
- 🎨 **Image Upload Interface**: Upload up to 4 PNG/JPG test images
- 🔄 **Real-time Prediction**: Model generates segmentation masks for uploaded images
- 🖼️ **Side-by-side Comparison**:
  - Original input image
  - Ground-truth segmentation mask
  - Model's predicted segmentation mask

---

## Installation & Setup

### Step 1: Install Dependencies
```bash
cd /home/m25csa019/MLOps-DLOps-Major/Ques_2
pip install streamlit torch torchvision opencv-python numpy matplotlib
```

### Step 2: Verify Required Files
Ensure the following files exist in the `Ques_2/` directory:
- ✅ `model.pth` - Trained model weights
- ✅ `training_metrics.json` - Training history (loss, mIOU, mDICE)
- ✅ `test_metrics.json` - Test set metrics
- ✅ `test_split.json` - Test image paths
- ✅ `Question2/loss_curve.png` - Training loss plot
- ✅ `Question2/miou_curve.png` - Training mIOU plot
- ✅ `Question2/mdice_curve.png` - Training mDICE plot
- ✅ `data/CameraRGB/` - Test images (RGB)
- ✅ `data/CameraMask/` - Ground-truth masks

---

## Running the Application

### Launch the Streamlit App
```bash
cd /home/m25csa019/MLOps-DLOps-Major/Ques_2
streamlit run app.py
```

The app will start on `http://localhost:8501/`

### Example Output
```
  You can now view your Streamlit app in your browser.
  
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## Using the Application

### Page 1: Training Metrics
1. **Navigate** to "📈 Page 1: Training Metrics" using the sidebar
2. **View Test Set Performance**:
   - Current test mIOU: 0.2698
   - Current test mDICE: 0.3100
   - Test set size: 80 images
3. **Analyze Training Curves**:
   - Loss decreased from initial value
   - mIOU and mDICE improvements over epochs
4. **Review Training Summary**:
   - Best metrics achieved
   - Loss reduction statistics

### Page 2: Segmentation Demo
1. **Navigate** to "🎨 Page 2: Segmentation Demo"
2. **Upload Test Images**:
   - Click "📁 Upload up to 4 test images"
   - Select PNG or JPG files from `data/CameraRGB/`
   - Image filenames should match those in the dataset (e.g., `010578.png`)
3. **View Results**:
   - Input image displayed on the left
   - Ground-truth mask in the middle
   - Model prediction on the right
4. **Compare Predictions**:
   - Analyze how well the model's segmentation matches ground truth
   - Identify prediction strengths and weaknesses

---

## Application Architecture

### Backend Components
```
├── UNet Model
│   ├── 4 Encoder blocks (Conv layers + BatchNorm + ReLU)
│   ├── Bottleneck layer
│   └── 4 Decoder blocks (Transposed Conv + Skip connections)
│
├── Configuration
│   ├── 23 semantic classes
│   ├── Input size: 96×128 pixels
│   ├── Device: GPU (CUDA) or CPU (auto-detected)
│
└── Utilities
    ├── Image preprocessing & resizing
    ├── Segmentation mask conversion to RGB
    ├── Ground-truth mask loading
    └── Model inference (@torch.no_grad())
```

### Frontend Pages
- **Page 1**: Displays pre-generated plots & metrics
- **Page 2**: Interactive demo with file upload & real-time inference

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Test mIOU** | 0.2698 |
| **Test mDICE** | 0.3100 |
| **Test Set Size** | 80 images |
| **Model Architecture** | UNet (Encoder-Decoder) |
| **Number of Classes** | 23 (semantic categories) |
| **Input Resolution** | 96×128 px |

---

## Troubleshooting

### App won't start
```bash
# Ensure all dependencies are installed
pip install --upgrade streamlit torch torchvision opencv-python

# Check if model file exists
ls -la model.pth
```

### Ground-truth masks not found
```bash
# Verify mask directory structure
ls data/CameraMask/
```

### Slow inference
- Ensure GPU is available: Run `python -c "import torch; print(torch.cuda.is_available())"`
- If using CPU, inference will be slower

### Port 8501 already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

---

## File Structure
```
Ques_2/
├── app.py                      # Main Streamlit application
├── model.pth                   # Trained model weights
├── train.py                    # Training script (reference)
├── training_metrics.json       # Training history
├── test_metrics.json          # Test set metrics
├── test_split.json            # Test image paths
├── Question2/
│   ├── loss_curve.png         # Training loss plot
│   ├── miou_curve.png         # Training mIOU plot
│   └── mdice_curve.png        # Training mDICE plot
└── data/
    ├── CameraRGB/             # Test images (original)
    └── CameraMask/            # Ground-truth masks
```

---

## API/Function Reference

### Key Functions
- `load_model()` - Loads trained UNet weights (cached)
- `predict(model, img)` - Runs inference on image
- `mask_to_rgb(mask_arr)` - Converts class indices to RGB visualization
- `load_gt_mask(filename)` - Loads ground-truth mask for comparison

### Streamlit Features Used
- `st.file_uploader()` - Image upload widget
- `st.metric()` - Display KPI cards
- `st.image()` - Display images/masks
- `st.columns()` - Layout management
- `st.cache_resource` - Model caching for performance

---

## Notes
- Model is loaded only once and cached for performance
- GPU/CPU auto-detection ensures compatibility
- All plots are pre-generated by training script
- Supports up to 4 simultaneous image uploads

