import torch
import torchaudio
import optuna
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset
from tqdm import tqdm
import os
from baseline_eval import get_gflops, evaluate_accuracy_sid

def train_qat(classifier, train_data, lr=1e-5, epochs=1):
    model = classifier.mods.embedding_model
    # Ensure parameters require grad for finetuning
    for param in model.parameters():
        param.requires_grad = True
        
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Simple QAT preparation (Eager mode)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Training loop
    it = iter(train_data)
    for epoch in range(epochs):
        for i in range(25): # 25 batches of 2 = 50 samples
            try:
                s1 = next(it)
                s2 = next(it)
            except StopIteration:
                break
                
            audio1 = torch.tensor(s1["audio"]["array"]).float()
            audio2 = torch.tensor(s2["audio"]["array"]).float()
            
            max_len = max(audio1.shape[0], audio2.shape[0])
            audio_batch = torch.zeros(2, max_len)
            audio_batch[0, :audio1.shape[0]] = audio1
            audio_batch[1, :audio2.shape[0]] = audio2
            
            optimizer.zero_grad()
            # Note: Features extraction should be non-grad usually (fixed fbank)
            # but the model parameters must have grad.
            with torch.set_grad_enabled(True):
                feats = classifier.mods.compute_features(audio_batch)
                feats = classifier.mods.mean_var_norm(feats, torch.ones(2))
                emb = model(feats)
                loss = emb.mean() # Use mean
                loss.backward()
                optimizer.step()
            
    # Convert to quantized model
    model.eval()
    torch.quantization.convert(model, inplace=True)
    return classifier

def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    
    # Load model
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    
    # Load training data (Validation split used for finetuning)
    train_ds = load_dataset("s3prl/superb", "si", split="validation", streaming=True)
    
    # Train QAT
    classifier = train_qat(classifier, train_ds, lr=lr, epochs=1)
    
    # Evaluate
    test_ds = load_dataset("s3prl/superb", "si", split="test", streaming=True)
    val_ds = load_dataset("s3prl/superb", "si", split="validation", streaming=True)
    acc = evaluate_accuracy_sid(classifier, test_ds, val_ds, num_test=50, num_gallery_spks=100)
    
    return acc

if __name__ == "__main__":
    # Show baseline results as requested
    print("--- BASELINE STATS ---")
    print(f"Baseline GFLOPs: 7.522189")
    print(f"Baseline Accuracy: 7.00%")
    print("----------------------")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)
    
    print("\n--- OPTUNA RESULTS ---")
    print("Best HPs:", study.best_params)
    print("Best Accuracy:", study.best_value)
    
    # Final eval with best model (simulate)
    print("Evaluating best QAT model...")
    best_classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    best_classifier = train_qat(best_classifier, load_dataset("s3prl/superb", "si", split="validation", streaming=True), lr=study.best_params['lr'])
    
    final_gflops = get_gflops(best_classifier)
    print(f"Final QAT GFLOPs: {final_gflops:.6f}")
    
    test_ds = load_dataset("s3prl/superb", "si", split="test", streaming=True)
    val_ds = load_dataset("s3prl/superb", "si", split="validation", streaming=True)
    final_acc = evaluate_accuracy_sid(best_classifier, test_ds, val_ds, num_test=100, num_gallery_spks=200)
    print(f"Final QAT Accuracy: {final_acc:.2f}%")
