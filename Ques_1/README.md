# Bengali to English Translation

Uses [Helsinki-NLP/opus-mt-bn-en](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en) from HuggingFace to translate Bengali sentences to English and evaluate using BLEU score.

---

## Project Structure

```
Ques_1/
├── Dockerfile        # Docker image definition
├── translate.py      # Translation script (Bengali → English)
├── evaluate.py       # BLEU score evaluation using sacrebleu
├── command.md        # Quick Docker commands reference
└── dataset/
    ├── input.txt     # Bengali input sentences
    ├── reference.txt # Reference English translations
    └── output.txt    # Generated English translations (created after run)
```

---

## Step 1 — Build the Docker Image

```bash
docker build -t bengali-translator .
```

---

## Step 2 — Run Translation

Translates `dataset/input.txt` (Bengali) → `dataset/output.txt` (English).

```bash
docker run --rm \
    -v $(pwd):/app \
    bengali-translator
```

---

## Step 3 — View Output

```bash
cat dataset/output.txt
```

---

## Step 4 — Evaluate BLEU Score

Computes corpus-level and sentence-level BLEU score between `dataset/output.txt` and `dataset/reference.txt`.

```bash
docker run --rm \
    -v $(pwd):/app \
    bengali-translator python evaluate.py
```

### Sample Results

| Metric             | Value  |
|--------------------|--------|
| Corpus BLEU        | 38.55  |
| Brevity Penalty    | 0.9083 |
| 1-gram precision   | 73.08% |
| 2-gram precision   | 48.77% |
| 3-gram precision   | 35.45% |
| 4-gram precision   | 25.68% |

---

## Step 5 — Run with GPU (optional)

```bash
docker run --rm --gpus all \
    -v $(pwd):/app \
    bengali-translator
```

---

## Step 6 — Clean Up

```bash
docker rmi bengali-translator
```

---

## Notes

- Model downloads automatically on first run (~300 MB)
- To persist the HuggingFace cache across runs:
  ```bash
  docker run --rm \
      -v $(pwd):/app \
      -v hf-cache:/root/.cache/huggingface \
      bengali-translator
  ```
