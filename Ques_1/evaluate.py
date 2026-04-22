import sacrebleu

HYPOTHESIS_FILE = "dataset/output.txt"
REFERENCE_FILE  = "dataset/reference.txt"


def read_sentences(filepath: str) -> list:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def main():
    hypotheses = read_sentences(HYPOTHESIS_FILE)
    references  = read_sentences(REFERENCE_FILE)

    if len(hypotheses) != len(references):
        print(f"Warning: hypothesis count ({len(hypotheses)}) != reference count ({len(references)})")
        n = min(len(hypotheses), len(references))
        hypotheses = hypotheses[:n]
        references  = references[:n]

    # Corpus-level BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print("=" * 50)
    print("SacreBLEU Evaluation")
    print("=" * 50)
    print(f"Corpus BLEU Score : {bleu.score:.2f}")
    print(f"Brevity Penalty   : {bleu.bp:.4f}")
    print(f"N-gram precisions : {[round(p, 2) for p in bleu.precisions]}")
    print(f"Hypothesis length : {bleu.sys_len}")
    print(f"Reference length  : {bleu.ref_len}")
    print("=" * 50)

    # Sentence-level BLEU for each pair
    print("\nSentence-level BLEU:")
    print("-" * 50)
    for i, (hyp, ref) in enumerate(zip(hypotheses, references), 1):
        score = sacrebleu.sentence_bleu(hyp, [ref])
        print(f"[{i:02d}] BLEU={score.score:.2f}")
        print(f"      HYP: {hyp}")
        print(f"      REF: {ref}")


if __name__ == "__main__":
    main()
