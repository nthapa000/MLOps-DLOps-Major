import json
import os

# Load test metrics
test_metrics_path = "test_metrics.json"
test_split_path = "test_split.json"

if os.path.exists(test_metrics_path) and os.path.exists(test_split_path):
    with open(test_metrics_path, "r") as f:
        test_metrics = json.load(f)
    
    with open(test_split_path, "r") as f:
        test_split = json.load(f)
    
    num_test_samples = len(test_split["images"])
    
    print("=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest Samples: {num_test_samples}")
    print(f"\nMetrics:")
    print(f"  • Mean IoU (mIOU):  {test_metrics['miou']:.4f}")
    print(f"  • Mean Dice (mDICE): {test_metrics['mdice']:.4f}")
    print("\n" + "=" * 60)
    print("\nTest Set Sample Images:")
    print("-" * 60)
    for i, img_path in enumerate(test_split["images"][:10], 1):
        filename = os.path.basename(img_path)
        print(f"  {i:2d}. {filename}")
    if num_test_samples > 10:
        print(f"  ... and {num_test_samples - 10} more images")
    print("=" * 60 + "\n")
else:
    print("Error: Test metrics or test split files not found!")
