import torch
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset
import os

# Load a few samples from the dataset to check structure
print("Loading dataset samples...")
ds = load_dataset("s3prl/superb", "si", split="test", streaming=True)
sample = next(iter(ds))
print(f"Sample keys: {sample.keys()}")
print(f"Sample label: {sample['label']}")
# print(f"Sample speaker_id: {sample['speaker_id']}") # Often exists in SI

# Load the model
print("Loading model...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})

# Check the classifier head
print(f"Model keys: {classifier.mods.keys()}")
if 'classifier' in classifier.mods:
    class_head = classifier.mods.classifier
    print(f"Classifier head: {class_head}")
    for name, param in class_head.named_parameters():
        if 'weight' in name:
            print(f"Weight {name} shape: {param.shape}")
            break

if hasattr(classifier.hparams, 'label_encoder'):
    le = classifier.hparams.label_encoder
    # Check what attributes le has
    print(f"Label encoder attributes: {dir(le)}")
    if hasattr(le, 'ind2lab'):
        print(f"Label encoder ind2lab size: {len(le.ind2lab)}")
        print(f"First 5 labels: {[le.ind2lab[i] for i in range(5)]}")
else:
    print("No label_encoder in hparams.")

# Try to get labels from the dataset
ds = load_dataset("s3prl/superb", "si", split="test", streaming=True)
sample = next(iter(ds))
print(f"Sample speaker_id: {sample['speaker_id']}")
# Load a few samples to see speaker IDs
it = iter(ds)
for _ in range(5):
    s = next(it)
    print(f"Label: {s['label']}, Speaker ID: {s['speaker_id']}")
