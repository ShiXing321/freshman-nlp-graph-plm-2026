"""Compare the downloaded CoNLL parquet test split with a public IOB1 copy."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
source = ROOT / ".cache/conll-reference-test.txt"
if not source.exists():
    import sys
    sys.path.insert(0, str(ROOT))
    from common.experiment import download
    download("https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb", source)
tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
reference, words, labels = [], [], []
for line in source.read_text(encoding="utf-8").splitlines() + [""]:
    if line.startswith("-DOCSTART-"):
        continue
    if not line.strip():
        if words:
            reference.append({"tokens": words, "tags": labels})
        words, labels = [], []
        continue
    fields = line.split()
    label = fields[-1]
    previous = labels[-1] if labels else "O"
    if label.startswith("I-") and previous[2:] != label[2:]:
        label = "B-" + label[2:]
    words.append(fields[0])
    labels.append(label)
actual = [json.loads(line) for line in (ROOT / "nlp-beginner/legacy/data/processed/ner/test.jsonl").read_text(encoding="utf-8").splitlines()]
mismatches = []
for i, (a, b) in enumerate(zip(actual, reference)):
    labels = [tags[t] for t in a["tags"]]
    if a["tokens"] != b["tokens"] or labels != b["tags"]:
        mismatches.append(i)
result = {"source": "https://github.com/synalp/NER/blob/master/corpus/CoNLL-2003/eng.testb",
    "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "actual_sentences": len(actual), "reference_sentences": len(reference),
    "conversion": "IOB1 to IOB2 at entity starts", "mismatching_sentences": mismatches,
    "first_sentence": reference[0],
    "status": "passed" if not mismatches and len(actual) == len(reference) else "mismatch"}
(ROOT / "运行记录/诊断/conll_reference_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k != "first_sentence"}, ensure_ascii=False))
