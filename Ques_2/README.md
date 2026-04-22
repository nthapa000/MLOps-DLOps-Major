# CityScape Semantic Segmentation - Question 2

## 📊 Model Performance

**Test Results:**
- **mIOU (Mean Intersection over Union)**: 0.2698
- **mDICE (Mean Dice Coefficient)**: 0.3100

---

## 🎯 Project Overview

This project implements a **UNet-based semantic segmentation model** for the CityScapes dataset with a complete ML pipeline including a professional 2-page Streamlit web application.

### ✅ Completed Components

1. **Model Training** (`train.py`)
   - UNet architecture with 23 semantic classes
   - 20 epochs of training
   - Adam optimizer with CosineAnnealingLR
   - Automatic model checkpoint saving

2. **Streamlit Web App** (`app.py`) - **2-Page Application**
   - **Page 1**: Training metrics & performance visualization
   - **Page 2**: Interactive segmentation demo

3. **Test Evaluation**
   - 80 test images (20% of dataset)
   - Comprehensive metrics calculation
   - Visual mask generation

---

## 🚀 Quick Start

### Installation
```bash
cd /home/m25csa019/MLOps-DLOps-Major/Ques_2
python3 -m pip install -r requirements.txt
```

### Run the Web App
```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## 📱 App Features

### Page 1: 📈 Training Metrics
- Test set mIOU and mDICE scores
- Three training curve plots:
  - Loss curve (Cross-Entropy)
  - mIOU curve
  - mDICE curve
- Training summary statistics

### Page 2: 🎨 Segmentation Demo
- Upload up to 4 test images
- View side-by-side comparison:
  - Original input image
  - Ground-truth mask
  - Model prediction
- Real-time inference with GPU support

---

## 📁 Project Structure

```
Ques_2/
├── app.py                    # Streamlit web application (2-page)
├── train.py                  # Training script
├── model.pth                 # Trained weights
├── training_metrics.json     # Training history
├── test_metrics.json         # Test set metrics
├── test_split.json          # Test image paths
├── requirements.txt          # Python dependencies
├── APP_GUIDE.md             # Full documentation
├── QUICK_START.md           # Quick start guide
├── IMPLEMENTATION_SUMMARY.md # Implementation details
├── Question2/               # Training plots
│   ├── loss_curve.png
│   ├── miou_curve.png
│   └── mdice_curve.png
└── data/
    ├── CameraRGB/           # Test images
    └── CameraMask/          # Ground-truth masks
```

---

## 🏗️ Model Architecture

**UNet Encoder-Decoder:**
- **Input**: 3-channel RGB image (96×128 px)
- **Encoder**: 4 convolutional blocks (3→64→128→256→512)
- **Bottleneck**: 512→1024 channels
- **Decoder**: 4 transposed conv blocks with skip connections
- **Output**: 23 class logits → semantic segmentation

---

## 📖 Documentation

For comprehensive details, see:
- [APP_GUIDE.md](APP_GUIDE.md) - Full application documentation
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

---

## ✨ Key Features

✅ Professional 2-page Streamlit app  
✅ Real-time model inference  
✅ GPU acceleration support  
✅ Beautiful data visualization  
✅ Interactive demo interface  
✅ Comprehensive error handling  
✅ Complete documentation  

---

**Status**: ✅ Complete & Ready for Deployment
