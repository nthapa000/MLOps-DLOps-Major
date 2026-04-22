# CityScape Segmentation — Run Guide

## Step 1: Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install streamlit opencv-python scikit-learn matplotlib tqdm numpy pillow
```

> Skip PyTorch if already installed. Run `python -c "import torch; print(torch.cuda.is_available())"` to verify GPU.

---

## Step 2: Train the Model

```bash
cd Ques_2
python train.py
```

- Trains UNet for **25 epochs** on single GPU (`cuda:0`)
- Saves `model.pth`, `training_metrics.json`, `test_metrics.json`
- Saves plots to `Question2/` (loss_curve.png, miou_curve.png, mdice_curve.png)
- Auto-updates `README.md` with test mIOU and mDice

> Training takes ~15–30 min on GPU depending on hardware.  
> 25 epochs + LR 5e-4 with CosineAnnealingLR is tuned to safely exceed the 0.48 threshold faster.

---

## Step 3: Commit Plots to GitHub

```bash
cd Ques_2
git add Question2/ model.pth training_metrics.json test_metrics.json README.md
git commit -m "Question2: Add UNet training plots and test metrics"
git push
```

---

## Step 4: Launch the Streamlit App

```bash
cd Ques_2
streamlit run app.py
```

- **Page 1**: Training loss, mIOU, mDice curves + test set metrics
- **Page 2**: Upload 4 PNG images from the test set → see Ground Truth + Predicted masks

> For Page 2, upload files directly from `data/CameraRGB/` (e.g. `000026.png`).  
> The app auto-matches the ground-truth mask by filename from `data/CameraMask/`.
