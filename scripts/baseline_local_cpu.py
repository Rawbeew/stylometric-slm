"""
baseline_local_cpu.py
Runs the feature-engineering baseline on the held-out eval set, locally,
no GPU. Uses your Zenodo feature set: TTR + syllables/word + sentence length
+ function-word ratio + punctuation density. Trained as logistic regression.

This is the "before" picture — what accuracy do you get without any neural net?
Useful even if you stop here.

Usage:
  python scripts/baseline_local_cpu.py
"""

import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "splits"
RESULTS_DIR = REPO_ROOT / "results"


# ---------- Feature extraction (from the Zenodo paper) ----------

STOPWORDS_EN = set("""a an the and or but if then else when while of in on at to from for with by as is are was were be been being have has had do does did this that these those it its their there here i you he she we they me him her us them my your his hers our their not no nor so than too very can will just dont don't""".split())
STOPWORDS_FR = set("""le la les un une des de du d' et ou mais si alors quand de dans sur à pour par avec comme est sont était étaient être été avoir avait ont fait fait ce cette ces celui celle ceux celles il elle on nous vous je tu me te se lui leur y ne pas non plus très peut vont juste""".split())
STOPWORDS_ES = set("""el la los un una unos unas de del y o pero si entonces cuando de en a para por con como es son era eran estar sido tener tiene tuvieron hace hacen este esta estos estas ese esa esos esas eso eso es no ni muy puede pueden solo""".split())
STOPWORDS_IT = set("""il lo la i gli un una di del dello della dei degli delle e o ma se allora quando di in a per da con come è sono era erano essere stato avere aveva hanno ha questo questa questi queste quello quella quelli quelle non né molto può possono solo""".split())

STOPWORDS = {
    "en": STOPWORDS_EN,
    "fr": STOPWORDS_FR,
    "es": STOPWORDS_ES,
    "it": STOPWORDS_IT,
}


def count_syllables(word: str) -> int:
    """Approximate syllable count for English/European words."""
    word = word.lower().strip(".,;:!?")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def extract_features(text: str, lang: str) -> dict:
    """Compute the 5-feature stylometric vector."""
    text = text.strip()
    # Tokenize (whitespace for European; character-level fallback for CJK)
    tokens = text.split()
    n_tokens = len(tokens)
    types = Counter(t.lower() for t in tokens)
    n_types = len(types)

    # 1. TTR (type-token ratio)
    ttr = n_types / n_tokens if n_tokens else 0.0

    # 2. Mean syllables per word
    total_syl = sum(count_syllables(t) for t in tokens)
    syl_per_word = total_syl / n_tokens if n_tokens else 0.0

    # 3. Mean sentence length (split on . ? !)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    mean_sent_len = n_tokens / len(sentences) if sentences else 0.0

    # 4. Function-word ratio (using lang-specific stopwords)
    sw = STOPWORDS.get(lang, STOPWORDS_EN)
    n_sw = sum(1 for t in tokens if t.lower() in sw)
    fn_ratio = n_sw / n_tokens if n_tokens else 0.0

    # 5. Punctuation density (per 100 tokens)
    punct = sum(1 for ch in text if ch in ",.;:!?")
    punct_density = (punct / n_tokens * 100) if n_tokens else 0.0

    return {
        "ttr": ttr,
        "syl_per_word": syl_per_word,
        "mean_sent_len": mean_sent_len,
        "fn_ratio": fn_ratio,
        "punct_density": punct_density,
    }


def featurize(rows: list) -> tuple:
    """Turn rows into (X, y) for logistic regression."""
    X = []
    y = []
    for r in rows:
        feats = extract_features(r["text"], r["language"])
        X.append([
            feats["ttr"],
            feats["syl_per_word"],
            feats["mean_sent_len"],
            feats["fn_ratio"],
            feats["punct_density"],
        ])
        y.append(r["author"])
    return X, y


def main():
    train_rows = []
    with open(SPLITS_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            train_rows.append(json.loads(line))

    eval_rows = []
    with open(SPLITS_DIR / "eval.jsonl", encoding="utf-8") as f:
        for line in f:
            eval_rows.append(json.loads(line))

    print(f"train: {len(train_rows)}, eval: {len(eval_rows)}")
    print("featurizing...")

    X_train, y_train = featurize(train_rows)
    X_eval, y_eval = featurize(eval_rows)

    # Logistic regression (no GPU needed, sklearn)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    print("training logistic regression...")
    clf.fit(X_train_s, y_train)

    print("predicting...")
    preds = clf.predict(X_eval_s)

    # Per-author, per-language accuracy
    from collections import defaultdict
    correct = defaultdict(lambda: defaultdict(int))
    total = defaultdict(lambda: defaultdict(int))
    for gold, pred, lang in zip(y_eval, preds, [r["language"] for r in eval_rows]):
        total[lang][gold] += 1
        if pred == gold:
            correct[lang][gold] += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline_logreg.json"
    out.write_text(json.dumps({
        "per_author_per_lang": {
            f"{lang}/{author}": {
                "correct": correct[lang][author],
                "total": total[lang][author],
                "accuracy": correct[lang][author] / total[lang][author] if total[lang][author] else 0.0
            }
            for lang in sorted(total.keys())
            for author in sorted(total[lang].keys())
        },
        "overall": {
            lang: {
                "correct": sum(correct[lang].values()),
                "total": sum(total[lang].values()),
                "accuracy": sum(correct[lang].values()) / sum(total[lang].values()) if sum(total[lang].values()) else 0.0,
            }
            for lang in sorted(total.keys())
        }
    }, indent=2))

    print(f"wrote {out}")
    print()
    print("baseline accuracy per language:")
    for lang, s in json.loads(out.read_text())["overall"].items():
        print(f"  {lang}: {s['accuracy']:.3f}  ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
