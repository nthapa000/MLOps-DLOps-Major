# 🚀 Quick Start Guide - CityScape Segmentation App

## One-Line Installation & Run

```bash
cd /home/m25csa019/MLOps-DLOps-Major/Ques_2 && \
python3 -m pip install -r requirements.txt && \
streamlit run app.py
```

## What You'll See

### 📈 Page 1: Training Metrics
When you open the app:
1. **Test Set Performance Metrics**
   - mIOU: 0.2698 ✓
   - mDICE: 0.3100 ✓
   - Test set size: 80 images

2. **Three Training Curves** displayed side-by-side:
   - 📉 Training Loss (decreasing over epochs)
   - 🎯 Training mIOU (increasing trend)
   - 🎲 Training mDICE (improving over time)

3. **Training Summary Statistics**:
   - Final metrics and loss reduction %
   - Best metrics achieved

### 🎨 Page 2: Segmentation Demo
Interactive demo where you can:
1. Upload up to 4 PNG/JPG test images
2. See three columns for each image:
   - **Left**: Original input image
   - **Middle**: Ground-truth segmentation mask
   - **Right**: Model's predicted mask
3. Compare predictions visually

---

## File Structure

```
Ques_2/
├── app.py ........................... Main Streamlit app (2-page)
├── model.pth ........................ Trained UNet weights
├── training_metrics.json ............ Training history
├── test_metrics.json ............... Test metrics (mIOU, mDICE)
├── test_split.json ................. Test image paths
├── APP_GUIDE.md .................... Full documentation
├── QUICK_START.md .................. This file
├── requirements.txt ................ Python dependencies
├── Question2/ ...................... Training plots
│   ├── loss_curve.png
│   ├── miou_curve.png
│   └── mdice_curve.png
└── data/
    ├── CameraRGB/ .................. Test images
    └── CameraMask/ ................. Ground-truth masks
```

---

## Features Implemented ✅

### ✨ Page 1: Training Metrics Display
- [x] Training loss curve plot
- [x] mIOU curve plot  
- [x] mDICE curve plot
- [x] Test set mIOU score (0.2698)
- [x] Test set mDICE score (0.3100)
- [x] Training summary statistics
- [x] Enhanced UI with metric cards

### ✨ Page 2: Segmentation Demo
- [x] File upload widget (up to 4 images)
- [x] Real-time model inference
- [x] Ground-truth mask loading
- [x] Side-by-side comparison view
- [x] Support for PNG/JPG formats
- [x] Error handling for missing masks
- [x] Beautiful layout with clear labels

---

## Backend Model Architecture

```
Input (3×96×128)
    ↓
Encoder:  3→64→128→256→512 (with MaxPool)
    ↓
Bottleneck: 512→1024
    ↓
Decoder: 1024→512→256→128→64 (with UpSample + Skip connections)
    ↓
Output (23×96×128) → argmax → 23 classes
```

**Model**: UNet
**Classes**: 23 semantic categories
**Trained on**: 415 images (80% train, 20% test)

---

## Performance

| Metric | Value |
|--------|-------|
| Test mIOU | **0.2698** |
| Test mDICE | **0.3100** |
| Test Samples | 80 images |
| GPU Support | Auto-detected (CUDA/CPU) |

---

## Troubleshooting

### Issue: "streamlit command not found"
```bash
python3 -m pip install streamlit
python3 -m streamlit run app.py
```

### Issue: "module not found"
```bash
python3 -m pip install -r requirements.txt --upgrade
```

### Issue: "Port 8501 in use"
```bash
streamlit run app.py --server.port 8502
```

### Issue: "Ground-truth masks not found"
Ensure the data directory structure is:
```
data/
├── CameraRGB/   (original images)
└── CameraMask/  (segmentation masks)
```

---

## Key Highlights

✅ **2-Page Navigation** - Sidebar with easy page switching  
✅ **Beautiful UI** - Custom CSS, emoji icons, clear layout  
✅ **Fast Inference** - Model cached for performance  
✅ **GPU Support** - Auto-detects CUDA/CPU  
✅ **Comprehensive Display** - Plots, metrics, and comparisons  
✅ **Error Handling** - Graceful failures with helpful messages  

---

## For More Details

See [APP_GUIDE.md](APP_GUIDE.md) for comprehensive documentation.

