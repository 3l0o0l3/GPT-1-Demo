"""Prepare the public-data training lab. Downloads bounded TinyStories prefixes.

No model training. Run `.venv/Scripts/python.exe training_data.py` from this folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[CLS]", "[UNK]"]
HF_REPO = "roneneldan/TinyStories"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}"
UCI_URL = "https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def text_hash(text: str) -> str:
    return digest_bytes(normalized(text).encode("utf-8"))


def get_small(url: str, maximum: int = 4 * 1024 * 1024) -> bytes:
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GPT1-seminar-training-data/1.0"})
            with OPENER.open(request, timeout=40) as response:
                raw = response.read(maximum + 1)
            if len(raw) > maximum:
                raise ValueError(f"Small-resource size limit exceeded: {url}")
            return raw
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stories_prefix(revision: str, filename: str, count: int, excluded: set[str], cache_dir: Path):
    """Stop reading after count distinct documents; never download the whole corpus."""
    name = "train" if "train" in filename else "validation"
    cache = cache_dir / f"tinystories-{name}-{count}-{revision[:12]}.jsonl"
    info_file = cache.with_suffix(".provenance.json")
    excluded_fingerprint = digest_bytes("\n".join(sorted(excluded)).encode("utf-8"))
    if cache.exists() and info_file.exists():
        records = load_jsonl(cache)
        info = json.loads(info_file.read_text(encoding="utf-8"))
        hashes = [text_hash(record["text"]) for record in records]
        if len(records) == count and len(set(hashes)) == count and not excluded.intersection(hashes) and info.get("selectedSha256") == file_hash(cache) and info.get("excludedSetSha256") == excluded_fingerprint:
            print(f"Reused verified {name} selection: {len(records)} stories", flush=True)
            return records, info
    url = f"{HF_URL}/resolve/{revision}/{filename}"
    maximum = 160 * 1024 * 1024 if name == "train" else 24 * 1024 * 1024
    records, seen, consumed, documents, duplicates = [], set(excluded), 0, 0, 0
    content_hash = hashlib.sha256()
    delimiter = b"<|endoftext|>"
    buffer = b""
    # A failed partial read is retried from the start; never concatenate broken streams.
    for attempt in range(3):
        records, seen, consumed, documents, duplicates = [], set(excluded), 0, 0, 0
        content_hash = hashlib.sha256()
        buffer = b""
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GPT1-seminar-training-data/1.0"})
            with OPENER.open(request, timeout=60) as response:
                while len(records) < count:
                    block = response.read(65536)
                    if not block:
                        raise ValueError(f"Source ended before {count} distinct {name} stories")
                    consumed += len(block)
                    if consumed > maximum:
                        raise ValueError(f"Bounded {name} download exceeded {maximum} bytes")
                    content_hash.update(block)
                    buffer += block
                    while delimiter in buffer and len(records) < count:
                        document, buffer = buffer.split(delimiter, 1)
                        documents += 1
                        text = document.decode("utf-8").strip()
                        if not text:
                            continue
                        fingerprint = text_hash(text)
                        if fingerprint in seen:
                            duplicates += 1
                            continue
                        seen.add(fingerprint)
                        records.append({"id": f"tinystories-{name}-{documents:08d}-{fingerprint[:12]}", "text": text})
                        if len(records) % 10000 == 0:
                            print(f"Selected {len(records)} {name} stories; read {consumed / 1e6:.1f} MB", flush=True)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == 2:
                raise
            print(f"Retrying bounded {name} download after {type(error).__name__}", flush=True)
            time.sleep(attempt + 1)
    write_jsonl(cache, records)
    info = {"url": url, "revision": revision, "filename": filename, "requestedCount": count,
            "selectedCount": len(records), "sourceDocumentsExamined": documents,
            "normalizedDuplicatesSkipped": duplicates, "downloadBytesConsumed": consumed,
            "excludedSetSha256": excluded_fingerprint,
            "consumedPrefixSha256": content_hash.hexdigest(), "selectedSha256": file_hash(cache),
            "selection": "First N distinct non-empty documents in source order, not a random sample. Stop HTTP stream after enough documents."}
    write_json(info_file, info)
    return records, info


def sentiment_data(raw: bytes, seed: int):
    filenames = {"amazon_cells_labelled.txt": "amazon", "imdb_labelled.txt": "imdb", "yelp_labelled.txt": "yelp"}
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for filename, source in filenames.items():
            matches = [name for name in archive.namelist() if Path(name).name == filename and "__MACOSX" not in name]
            if len(matches) != 1:
                raise ValueError(f"Expected one UCI source file {filename}")
            content = archive.read(matches[0]).decode("utf-8-sig")
            # The source uses LF records; splitlines() would also split a U+0085
            # character inside an IMDb sentence and corrupt its record boundary.
            for line_number, line in enumerate(content.split("\n"), 1):
                if not line.strip():
                    continue
                text, raw_label = line.rsplit("\t", 1)
                if raw_label.strip() not in ("0", "1") or not text.strip():
                    raise ValueError(f"Invalid sentiment row {source}:{line_number}")
                rows.append({"text": text.strip(), "label": int(raw_label), "source": source,
                             "sourceLine": line_number})
    groups = defaultdict(list)
    for row in rows:
        groups[text_hash(row["text"])].append(row)
    conflicts, deduplicated, duplicates = [], [], []
    for fingerprint, group in sorted(groups.items()):
        if len({row["label"] for row in group}) != 1:
            conflicts.append({"normalizedSha256": fingerprint, "occurrences": group})
            continue
        row = group[0]
        deduplicated.append({"id": f"uci331-{fingerprint[:20]}", "text": row["text"],
                             "label": row["label"], "source": row["source"], "sourceLine": row["sourceLine"]})
        if len(group) > 1:
            duplicates.append({"id": f"uci331-{fingerprint[:20]}", "occurrences": group})
    strata = defaultdict(list)
    for row in deduplicated:
        strata[(row["source"], row["label"])].append(row)
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda row: row["id"])
        rng.shuffle(group)
        train_end, val_count = int(len(group) * .6), int(len(group) * .2)
        splits["train"].extend(group[:train_end])
        splits["validation"].extend(group[train_end:train_end + val_count])
        splits["test"].extend(group[train_end + val_count:])
    for split in splits.values():
        rng.shuffle(split)
    report = {"rawCount": len(rows), "rawBySource": dict(Counter(row["source"] for row in rows)),
              "retainedCount": len(deduplicated), "duplicatesRemoved": sum(len(item["occurrences"]) - 1 for item in duplicates),
              "conflictingGroupsRemoved": len(conflicts), "conflictingRowsRemoved": sum(len(item["occurrences"]) for item in conflicts),
              "duplicateGroups": duplicates, "conflicts": conflicts,
              "splitStrata": {name: dict(Counter(f"{row['source']}:{row['label']}" for row in split)) for name, split in splits.items()},
              "rule": f"NFKC + collapsed whitespace + casefold exact-text grouping before split; conflicting labels excluded; same-label groups retain first source in amazon/imdb/yelp order. Fixed seed {seed}, floor60/floor20/remainder per source+label."}
    return splits, report


def fit_tokenizer(stories, sentiment_train, destination: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=2, special_tokens=SPECIALS,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    texts = (record["text"] for collection in (stories, sentiment_train) for record in collection)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(stories) + len(sentiment_train))
    if [tokenizer.token_to_id(token) for token in SPECIALS] != list(range(5)):
        raise AssertionError("Special token IDs do not match the training contract")
    tokenizer.save(str(destination))
    return tokenizer


def encode_records(tokenizer, records):
    for start in range(0, len(records), 512):
        chunk = records[start:start + 512]
        encodings = tokenizer.encode_batch([row["text"] for row in chunk], add_special_tokens=False)
        for row, encoding in zip(chunk, encodings):
            if not encoding.ids or any(value < 5 for value in encoding.ids):
                raise AssertionError(f"Empty or special-containing content token sequence: {row['id']}")
            row["ids"] = encoding.ids


def validate_outputs(stories_train, stories_val, sentiment):
    named = {"stories-train": stories_train, "stories-validation": stories_val,
             **{f"sentiment-{key}": value for key, value in sentiment.items()}}
    identifiers = [row["id"] for collection in named.values() for row in collection]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Duplicate sample ID")
    sets = {key: {text_hash(row["text"]) for row in value} for key, value in named.items()}
    for key, rows in named.items():
        if len(rows) != len(sets[key]):
            raise AssertionError(f"Duplicate normalized text in {key}")
    fit = sets["stories-train"] | sets["sentiment-train"]
    held_out = sets["stories-validation"] | sets["sentiment-validation"] | sets["sentiment-test"]
    if fit & held_out:
        raise AssertionError("Tokenizer/LM train texts intersect held-out texts")
    if sets["sentiment-validation"] & sets["sentiment-test"]:
        raise AssertionError("Sentiment validation/test overlap")
    return {"uniqueIds": len(identifiers), "normalizedDuplicatesWithinSplits": 0,
            "tokenizerFitHeldOutExactOverlap": 0, "sentimentValidationTestOverlap": 0,
            "normalization": "Unicode NFKC, whitespace collapse, strip, casefold"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--train-stories", type=int, default=50000)
    parser.add_argument("--validation-stories", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", help="Pinned TinyStories 40-character source revision; defaults to existing manifest revision, otherwise current main.")
    args = parser.parse_args()
    if not 1 <= args.train_stories <= 50000 or not 1 <= args.validation_stories <= 2000:
        parser.error("Bounded demo selection supports 1–50000 train and 1–2000 validation stories")
    data = args.data_dir.resolve()
    raw_dir, licenses = data / "raw", data / "licenses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    licenses.mkdir(parents=True, exist_ok=True)
    previous_manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8")) if (data / "manifest.json").exists() else None
    previous_revision = next((source.get("revision") for source in previous_manifest["sources"] if source["name"] == "TinyStories"), None) if previous_manifest else None
    revision = args.revision or previous_revision
    if not revision:
        metadata = json.loads(get_small(f"https://huggingface.co/api/datasets/{HF_REPO}"))
        revision = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid source revision")
    card = get_small(f"{HF_URL}/raw/{revision}/README.md")
    (licenses / "TinyStories-dataset-card.md").write_bytes(card)
    # Preserve original license statements alongside the source snapshots.
    (licenses / "TinyStories-CDLA-Sharing-1.0.html").write_bytes(get_small("https://cdla.dev/sharing-1-0/"))
    (licenses / "UCI-Sentiment-publisher-page.html").write_bytes(get_small(UCI_URL))
    (licenses / "UCI-CC-BY-4.0-legalcode.txt").write_bytes(get_small("https://creativecommons.org/licenses/by/4.0/legalcode.txt"))
    train, train_info = stories_prefix(revision, "TinyStories-train.txt", args.train_stories, set(), raw_dir)
    validation, val_info = stories_prefix(revision, "TinyStories-valid.txt", args.validation_stories,
                                         {text_hash(row["text"]) for row in train}, raw_dir)
    archive_path = raw_dir / "uci331-sentiment-labelled-sentences.zip"
    if not archive_path.exists():
        archive_path.write_bytes(get_small(UCI_ZIP, maximum=2 * 1024 * 1024))
    if previous_manifest:
        previous_archive_hash = next((source.get("sha256") for source in previous_manifest["sources"] if source["name"] == "UCI Sentiment Labelled Sentences"), None)
        if previous_archive_hash and previous_archive_hash != file_hash(archive_path):
            raise ValueError("Cached UCI archive differs from the pinned manifest checksum")
    sentiment, dedup_report = sentiment_data(archive_path.read_bytes(), args.seed)
    write_json(data / "sentiment-deduplication.json", dedup_report)
    print(f"Sentiment: {dedup_report['rawCount']} raw → {dedup_report['retainedCount']} distinct, non-conflicting", flush=True)
    validation_report = validate_outputs(train, validation, sentiment)
    print("Fitting byte-level BPE on training text only…", flush=True)
    tokenizer = fit_tokenizer(train, sentiment["train"], data / "tokenizer.json")
    for collection in (train, validation, *sentiment.values()):
        encode_records(tokenizer, collection)
    audit_rng = random.Random(args.seed)
    audited = [row for collection in (train, validation, *sentiment.values()) for row in audit_rng.sample(collection, min(64, len(collection)))]
    if any(tokenizer.decode(row["ids"], skip_special_tokens=False) != row["text"] for row in audited):
        raise AssertionError("Byte-level BPE round-trip mismatch")
    validation_report["tokenizerRoundTripChecks"] = len(audited)
    validation_report["tokenizerSpecialIds"] = [tokenizer.token_to_id(token) for token in SPECIALS]
    write_jsonl(data / "stories-train.jsonl", train)
    write_jsonl(data / "stories-val.jsonl", validation)
    write_json(data / "sentiment.json", sentiment)
    write_json(data / "validation-report.json", validation_report)
    outputs = ["tokenizer.json", "stories-train.jsonl", "stories-val.jsonl", "sentiment.json", "sentiment-deduplication.json", "validation-report.json"]
    manifest = {
        "version": 1,
        "sources": [
            {"name": "TinyStories", "url": HF_URL, "license": "CDLA-Sharing-1.0", "revision": revision,
             "description": "Public synthetic short English stories generated by GPT-3.5/GPT-4; original train/validation files, not V2."},
            {"name": "UCI Sentiment Labelled Sentences", "url": UCI_URL, "license": "CC-BY-4.0",
             "sha256": file_hash(archive_path), "downloadUrl": UCI_ZIP,
             "description": "Human review sentences from Amazon, IMDb and Yelp; positive/negative labels. DOI 10.24432/C57604."}],
        "counts": {"storiesTrain": len(train), "storiesValidation": len(validation),
                   "sentimentTrain": len(sentiment["train"]), "sentimentValidation": len(sentiment["validation"]), "sentimentTest": len(sentiment["test"])},
        "tokenizer": {"vocabSize": tokenizer.get_vocab_size(), "targetVocabSize": 8192,
                      "specialTokens": {token: tokenizer.token_to_id(token) for token in SPECIALS},
                      "fitSplits": ["stories-train", "sentiment.train"], "fitSampleCount": len(train) + len(sentiment["train"]),
                      "fitSampleIdsSha256": digest_bytes("\n".join(sorted(row["id"] for collection in (train, sentiment["train"]) for row in collection)).encode()),
                      "algorithm": "Byte-level BPE, min_frequency=2, full byte alphabet, no post-processor or automatic special tokens"},
        "splitSeed": args.seed,
        "selection": {"stories": "Deterministic source-order prefix of distinct documents; NOT random sampling. Original official validation source remains held out.",
                      "sentiment": "Deduplicate normalized text first; exclude conflicting-label groups; seeded 60/20/20 within source+label strata.",
                      "train": train_info, "validation": val_info},
        "checksums": {name: file_hash(data / name) for name in outputs},
        "validation": validation_report,
        "tokenCounts": {"storiesTrain": sum(len(row["ids"]) for row in train), "storiesValidation": sum(len(row["ids"]) for row in validation),
                        **{f"sentiment-{name}": sum(len(row["ids"]) for row in rows) for name, rows in sentiment.items()}},
        "sentimentCleaning": {key: value for key, value in dedup_report.items() if key not in ("duplicateGroups", "conflicts")},
        "licenses": {path.name: file_hash(path) for path in sorted(licenses.iterdir()) if path.is_file()},
    }
    write_json(data / "manifest.json", manifest)
    print(json.dumps({"data": str(data), "counts": manifest["counts"], "vocabSize": tokenizer.get_vocab_size(), "checks": validation_report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
