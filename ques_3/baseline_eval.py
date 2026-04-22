import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset
from thop import profile
import numpy as np
from tqdm import tqdm

def get_gflops(classifier, input_seconds=2.0):
    # ECAPA-TDNN expects (Batch, Time, Feat)
    # 2 seconds = approx 200 frames of 80 fbank bins
    num_frames = int(input_seconds * 100)
    inputs = torch.randn(1, num_frames, 80)
    macs, params = profile(classifier.mods.embedding_model, inputs=(inputs, ))
    gflops = (macs * 2) / 1e9 
    return gflops

def evaluate_accuracy_sid(classifier, test_ds, val_ds, num_test=100, num_gallery_spks=500):
    print(f"Building gallery from validation set (up to {num_gallery_spks} speakers)...")
    gallery_embeddings = {} # label -> embedding
    
    # Use streaming to get one sample per speaker from validation set
    pbar = tqdm(desc="Building Gallery")
    for sample in val_ds:
        spk_label = sample["label"]
        if spk_label not in gallery_embeddings:
            audio = torch.tensor(sample["audio"]["array"]).unsqueeze(0).float()
            with torch.no_grad():
                emb = classifier.encode_batch(audio)
                gallery_embeddings[spk_label] = emb.squeeze(0).cpu()
            pbar.update(1)
        if len(gallery_embeddings) >= num_gallery_spks:
            break
    pbar.close()
            
    ref_labels = list(gallery_embeddings.keys())
    ref_embs = torch.stack([gallery_embeddings[l] for l in ref_labels]) # (N, Emb)
    
    print(f"Evaluating {num_test} samples from test set...")
    correct = 0
    total = 0
    
    it = iter(test_ds)
    for _ in tqdm(range(num_test), desc="Testing"):
        try:
            sample = next(it)
        except StopIteration:
            break
            
        audio = torch.tensor(sample["audio"]["array"]).unsqueeze(0).float()
        true_label = sample["label"]
        
        with torch.no_grad():
            emb = classifier.encode_batch(audio).squeeze(0).cpu()
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), ref_embs)
            pred_idx = torch.argmax(cos_sim).item()
            pred_label = ref_labels[pred_idx]
            
            if pred_label == true_label:
                correct += 1
            total += 1
            
    if total == 0: return 0.0
    return (correct / total) * 100

if __name__ == "__main__":
    print("Loading model...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    
    print("Calculating GFLOPs...")
    gflops = get_gflops(classifier)
    print(f"Baseline GFLOPs: {gflops:.6f}")
    
    print("Loading dataset (streaming)...")
    # Using small portions for speed as requested
    ds_test = load_dataset("s3prl/superb", "si", split="test", streaming=True)
    ds_val = load_dataset("s3prl/superb", "si", split="validation", streaming=True)
    
    print("Evaluating baseline accuracy (Top-1 SID)...")
    # Top-1 identification accuracy on 100 test samples with 200 possible speakers in gallery
    acc = evaluate_accuracy_sid(classifier, ds_test, ds_val, num_test=100, num_gallery_spks=200)
    print(f"Baseline Accuracy: {acc:.2f}%")
