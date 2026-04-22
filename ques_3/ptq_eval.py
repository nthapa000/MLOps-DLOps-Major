import torch
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset
from thop import profile
from tqdm import tqdm
from baseline_eval import get_gflops, evaluate_accuracy_sid

if __name__ == "__main__":
    print("Loading model...")
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    
    print("Applying PTQ (Dynamic INT8)...")
    # Dynamic quantization for Linear modules
    # ECAPA-TDNN has many Linear layers in the embedding head and some other parts.
    # Conv1d dynamic quantization is not widely supported, but let's try.
    quantized_model = torch.quantization.quantize_dynamic(
        classifier.mods.embedding_model, 
        {torch.nn.Linear}, # Linear is well supported
        dtype=torch.qint8
    )
    classifier.mods.embedding_model = quantized_model
    
    # Theoretical GFLOPs for PTQ:
    # Usually, the number of ops remains the same, but they are INT8.
    # If the tool doesn't show a difference, I'll calculate based on the fact that
    # the number of operations is identical, but the precision is INT8.
    # HOWEVER, some tools might report 0 for quantized layers.
    print("Calculating PTQ GFLOPs...")
    gflops = get_gflops(classifier)
    print(f"PTQ GFLOPs: {gflops:.6f}")
    
    print("Loading dataset (streaming)...")
    ds_test = load_dataset("s3prl/superb", "si", split="test", streaming=True)
    ds_val = load_dataset("s3prl/superb", "si", split="validation", streaming=True)
    
    print("Evaluating PTQ accuracy (Top-1 SID)...")
    acc = evaluate_accuracy_sid(classifier, ds_test, ds_val, num_test=100, num_gallery_spks=200)
    print(f"PTQ Accuracy: {acc:.2f}%")
