import os, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
N_CLASSES    = 23
IMG_H, IMG_W = 96, 128
MODEL_PATH   = "model.pth"
DATA_DIR     = "data"
PLOTS_DIR    = "Question2"
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set Streamlit page config
st.set_page_config(
    page_title="CityScape Semantic Segmentation - MLOps Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 23 visually distinct HSV colors for segmentation classes
CLASS_COLORS = np.array(
    [plt.cm.hsv(i / N_CLASSES)[:3] for i in range(N_CLASSES)],
    dtype=np.float32
)
CLASS_COLORS = (CLASS_COLORS * 255).astype(np.uint8)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .title-main {
        color: #0066cc;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ── UNet (must match train.py exactly) ───────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(3,   64)
        self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bot  = DoubleConv(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512,  256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256,  128)
        self.up1  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec1 = DoubleConv(128,   64)
        self.out  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def mask_to_rgb(mask_arr):
    """Convert (H, W) class-index array to (H, W, 3) color image."""
    mask_arr = np.clip(mask_arr, 0, N_CLASSES - 1)
    return CLASS_COLORS[mask_arr]

@torch.no_grad()
def predict(model, img_rgb_np):
    """Run inference on a raw RGB numpy image (any size)."""
    img = cv2.resize(img_rgb_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    return pred

def load_gt_mask(filename):
    """Load ground-truth mask for a given image filename (e.g. 000026.png)."""
    mask_path = os.path.join(DATA_DIR, "CameraMask", filename)
    if not os.path.exists(mask_path):
        return None, mask_path
    mask_bgr = cv2.imread(mask_path)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    mask_rs  = cv2.resize(mask_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    mask_idx = np.clip(np.max(mask_rs, axis=-1), 0, N_CLASSES - 1)
    return mask_idx, mask_path

# ── Header & Navigation ───────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; color: #0066cc;'>🔍 Semantic Image Segmentation - MLOps Application</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>CityScape Dataset | UNet Model | Real-time Inference</p>", unsafe_allow_html=True)
st.divider()

# ── Streamlit App Navigation ──────────────────────────────────────────────────────
page = st.sidebar.radio("📊 Navigation", ["📈 Page 1: Training Metrics", "🎨 Page 2: Segmentation Demo"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Training Metrics & Plots
# ══════════════════════════════════════════════════════════════════════════════
if page == "📈 Page 1: Training Metrics":
    st.markdown("<div class='title-main'>📊 Training Metrics & Model Performance</div>", unsafe_allow_html=True)
    
    # Load test metrics
    test_data = None
    if os.path.exists("test_metrics.json"):
        with open("test_metrics.json") as f:
            test_data = json.load(f)
    
    # Load training history
    training_data = None
    if os.path.exists("training_metrics.json"):
        with open("training_metrics.json") as f:
            training_data = json.load(f)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TEST SET METRICS
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🧪 Test Set Performance")
    if test_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Test mIOU (Mean Intersection over Union)",
                f"{test_data['miou']:.4f}",
                delta="✓ Above 0.48" if test_data['miou'] >= 0.48 else "Below 0.48",
                delta_color="normal" if test_data['miou'] >= 0.48 else "inverse"
            )
        
        with col2:
            st.metric(
                "Test mDICE (Mean Dice Coefficient)",
                f"{test_data['mdice']:.4f}",
                delta="✓ Above 0.48" if test_data['mdice'] >= 0.48 else "Below 0.48",
                delta_color="normal" if test_data['mdice'] >= 0.48 else "inverse"
            )
        
        with col3:
            if os.path.exists("test_split.json"):
                with open("test_split.json") as f:
                    split_data = json.load(f)
                    num_test = len(split_data.get("images", []))
                st.metric("Test Set Size", num_test, "images")
    else:
        st.warning("⚠️ test_metrics.json not found.")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING CURVES
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📈 Training Curves")
    
    if training_data:
        st.info(f"Total Epochs: {len(training_data['loss'])}")
    
    plots_info = [
        ("loss_curve.png", "📉 Training Loss (Cross-Entropy Loss)"),
        ("miou_curve.png", "🎯 Training mIOU (Mean Intersection over Union)"),
        ("mdice_curve.png", "🎲 Training mDICE (Mean Dice Coefficient)"),
    ]
    
    # Display plots in a 3-column layout
    cols = st.columns(3)
    for idx, (fname, title) in enumerate(plots_info):
        path = os.path.join(PLOTS_DIR, fname)
        with cols[idx]:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Plot not found: {fname}")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("📋 Training Summary")
    if training_data:
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.write(f"""
            **Final Training Loss:** {training_data['loss'][-1]:.4f}
            
            **Initial Training Loss:** {training_data['loss'][0]:.4f}
            
            **Loss Reduction:** {((training_data['loss'][0] - training_data['loss'][-1]) / training_data['loss'][0] * 100):.2f}%
            """)
        
        with summary_col2:
            st.write(f"""
            **Final Training mIOU:** {training_data['miou'][-1]:.4f}
            
            **Best Training mIOU:** {max(training_data['miou']):.4f}
            """)
        
        with summary_col3:
            st.write(f"""
            **Final Training mDICE:** {training_data['mdice'][-1]:.4f}
            
            **Best Training mDICE:** {max(training_data['mdice']):.4f}
            """)
    else:
        st.warning("⚠️ training_metrics.json not found.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Segmentation Demo
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Page 2: Segmentation Demo":
    st.markdown("<div class='title-main'>🎨 Semantic Segmentation Demonstration</div>", unsafe_allow_html=True)
    
    st.info(
        "**Instructions:** Upload up to **4 PNG images** from the test set. "
        "For each image, the app will display:\n"
        "- **Input Image** (original)\n"
        "- **Ground-Truth Mask** (actual labels)\n"
        "- **Predicted Mask** (model's prediction)"
    )

    # File uploader
    uploads = st.file_uploader(
        "📁 Upload up to 4 test images (.png/.jpg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploads:
        # Check model exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model weights not found at `{MODEL_PATH}`. Please run training first.")
            st.stop()

        model = load_model()
        to_show = uploads[:4]

        if len(uploads) > 4:
            st.warning(f"⚠️ Only the first 4 images will be shown (uploaded {len(uploads)})")

        st.subheader(f"📸 Processing {len(to_show)} image(s)...")
        st.divider()

        # Process each uploaded image
        for img_idx, uploaded in enumerate(to_show, 1):
            fname = uploaded.name
            st.markdown(f"### Image {img_idx}: `{fname}`")

            # Decode uploaded image
            raw = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                st.error(f"❌ Failed to decode image: {fname}")
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Resize for consistent display
            img_display = cv2.resize(img_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            # Get ground-truth mask
            gt_mask, mask_path = load_gt_mask(fname)

            # Get model prediction
            pred_mask = predict(model, img_rgb)
            pred_rgb = mask_to_rgb(pred_mask)

            # Display in 3 columns
            col_img, col_gt, col_pred = st.columns(3)

            with col_img:
                st.markdown("**Input Image**")
                st.image(img_display, use_container_width=True, clamp=True)

            with col_gt:
                st.markdown("**Ground-Truth Mask**")
                if gt_mask is not None:
                    st.image(mask_to_rgb(gt_mask), use_container_width=True, clamp=True)
                else:
                    st.warning(f"⚠️ GT mask not found\n\n`{mask_path}`")

            with col_pred:
                st.markdown("**Predicted Mask**")
                st.image(pred_rgb, use_container_width=True, clamp=True)

            # Divider between images
            if img_idx < len(to_show):
                st.divider()
    else:
        st.info("👆 Upload images to get started!")
