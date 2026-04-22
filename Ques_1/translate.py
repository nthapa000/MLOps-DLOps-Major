import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
INPUT_FILE = os.path.join(DATASET_DIR, "input.txt")
OUTPUT_FILE = os.path.join(DATASET_DIR, "output.txt")
REFERENCE_FILE = os.path.join(DATASET_DIR, "reference.txt")


def load_model_and_tokenizer(model_name: str):
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(DEVICE)
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def translate_sentence(text: str, model, tokenizer, max_length: int = 512) -> str:
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def read_sentences(filepath: str) -> list:
    """Read non-empty, non-comment lines from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]


def translate_file(input_file: str, output_file: str, model, tokenizer) -> list:
    logger.info(f"Reading input from: {input_file}")
    sentences = read_sentences(input_file)
    logger.info(f"Found {len(sentences)} sentences to translate")

    translations = []
    for i, sentence in enumerate(sentences, 1):
        logger.info(f"Translating sentence {i}/{len(sentences)}: {sentence[:60]}...")
        translated = translate_sentence(sentence, model, tokenizer)
        translations.append(translated)
        logger.info(f"  -> {translated}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(translations) + '\n')

    logger.info(f"Translations saved to: {output_file}")
    return translations


def calculate_bleu_score(reference_file: str, translations: list) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        references = read_sentences(reference_file)

        if len(references) != len(translations):
            logger.warning(
                f"Reference count ({len(references)}) != translation count ({len(translations)}). "
                "Truncating to the shorter list."
            )
            n = min(len(references), len(translations))
            references = references[:n]
            translations = translations[:n]

        # corpus_bleu expects: list-of-reference-lists, list-of-hypotheses
        ref_corpus = [[word_tokenize(ref.lower())] for ref in references]
        hyp_corpus = [word_tokenize(hyp.lower()) for hyp in translations]

        smoothing = SmoothingFunction().method1
        bleu = corpus_bleu(ref_corpus, hyp_corpus, smoothing_function=smoothing)

        logger.info(f"BLEU Score: {bleu:.4f}")
        return bleu

    except ImportError:
        logger.warning("NLTK not available. Install with: pip install nltk")
        return None
    except Exception as e:
        logger.warning(f"Error calculating BLEU score: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("Bengali to English Translation Evaluation")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info("=" * 60)

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    translations = translate_file(INPUT_FILE, OUTPUT_FILE, model, tokenizer)

    if os.path.exists(REFERENCE_FILE):
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Metrics")
        logger.info("=" * 60)
        calculate_bleu_score(REFERENCE_FILE, translations)
    else:
        logger.info(f"Reference file not found: {REFERENCE_FILE} — skipping BLEU")

    logger.info("\n" + "=" * 60)
    logger.info("Translation completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
