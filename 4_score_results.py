"""
Scores raw model output (JSONL from the inference harness) against reference
transcripts: computes WER, CER, a lexical-overlap proxy for semantic
similarity (replace with real LaBSE scores using 5_rescore_labse.py before
treating any hallucination-rate numbers as final), and flags decoding-level
repetition collapse.

Handles a known Kathbath-specific issue: 100 utterance identifiers collide
between the train and valid splits, referring to different recordings. This
script resolves the collision using two facts about how run_benchmark
processes samples: (1) within a single uninterrupted run, samples are
iterated in list order, train entries first, then valid entries, so for any
colliding id, the first occurrence in the output file is the train version
and the second is the valid version; (2) if a run was resumed after an
interruption, the resume logic can only ever skip the *later* occurrence of
a repeated id (since it checks against already-written lines), so a
colliding id appearing only once in a resumed file's output must be the
train version.
"""

import json
import difflib
import csv
import jiwer


def load_meta(path, split):
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["utt_id"]] = {
                "text": rec["text"], "duration": rec.get("duration"),
                "gender": rec.get("gender"), "split": split,
            }
    return out


def assign_splits_positional(results_path, collided_ids, train_ids, valid_ids):
    """Resolve which split each row belongs to for a Kathbath results file."""
    seen_count = {}
    resolved = []
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            if "error" in rec:
                continue
            if sid in collided_ids:
                seen_count[sid] = seen_count.get(sid, 0) + 1
                split = "train" if seen_count[sid] == 1 else "valid"
            else:
                split = "train" if sid in train_ids else "valid"
            resolved.append((sid, split, rec))
    return resolved


def compute_wer_cer(ref, hyp):
    if not ref.strip() or not hyp.strip():
        return None, None
    try:
        return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)
    except Exception:
        return None, None


def lexical_similarity(a, b):
    """Cheap proxy only -- rescore with LaBSE (script 5) before reporting
    any hallucination-rate numbers. This proxy under-counts hallucination
    relative to real semantic similarity; see paper Section IV."""
    if not a.strip() or not b.strip():
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def detect_repetition(text, ngram_size=3, max_repeats=4):
    """A prediction is flagged as repetition collapse if any exact n-gram
    (default: 3 consecutive words) occurs 4+ times anywhere in the output,
    not necessarily consecutively. Outputs shorter than ngram_size *
    max_repeats words cannot trigger this flag."""
    words = text.split()
    if len(words) < ngram_size * max_repeats:
        return False
    seen = {}
    for i in range(len(words) - ngram_size + 1):
        ngram = tuple(words[i:i + ngram_size])
        seen[ngram] = seen.get(ngram, 0) + 1
        if seen[ngram] >= max_repeats:
            return True
    return False


def classify(wer, sim, wer_threshold=0.5, sim_threshold=0.45):
    if wer is None:
        return "EMPTY_OUTPUT"
    if wer <= wer_threshold:
        return "FAITHFUL"
    elif sim >= sim_threshold:
        return "PARAPHRASE_OR_RECOGNITION_ERROR"
    else:
        return "HALLUCINATION_CANDIDATE_LEXICAL"


def score_kathbath(train_meta_path, valid_meta_path, file_configs, out_csv):
    train = load_meta(train_meta_path, "train")
    valid = load_meta(valid_meta_path, "valid")
    collided_ids = set(train.keys()) & set(valid.keys())
    print(f"Collided ids between train/valid: {len(collided_ids)}")

    def get_ref(sid, split):
        return (train if split == "train" else valid)[sid]

    all_rows = []
    for fname, model_name, prompt_version in file_configs:
        resolved = assign_splits_positional(fname, collided_ids, train.keys(), valid.keys())
        for sid, split, rec in resolved:
            ref_entry = get_ref(sid, split)
            ref = ref_entry["text"]
            hyp = rec.get("cleaned_response", "")
            wer, cer = compute_wer_cer(ref, hyp)
            sim = lexical_similarity(ref, hyp)
            label = classify(wer, sim)
            rep = detect_repetition(hyp)
            all_rows.append({
                "sample_id": sid, "split": split, "source_corpus": "kathbath",
                "model_name": model_name, "prompt_version": prompt_version,
                "reference": ref, "prediction": hyp,
                "wer": wer, "cer": cer, "lexical_similarity": round(sim, 4),
                "label": label, "repetition_flag": rep,
                "duration": ref_entry["duration"], "gender": ref_entry["gender"],
                "has_unintelligible_tag": "<unintelligible>" in ref,
                "inference_time_sec": rec.get("inference_time_sec"),
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {out_csv}")


def score_indicvoices(meta_path, file_configs, out_csv):
    """IndicVoices has globally unique ids, no split-collision handling needed."""
    manifest = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            manifest[rec["utt_id"]] = {"text": rec["text"], "duration": rec.get("duration")}

    all_rows = []
    for fname, model_name, prompt_version in file_configs:
        seen = {}
        with open(fname, "rb") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip corrupted lines (e.g. from an interrupted write)
                if "error" in rec:
                    continue
                sid = rec["sample_id"]
                if sid not in seen:  # dedupe exact-duplicate lines from resumed batches
                    seen[sid] = rec

        for sid, rec in seen.items():
            if sid not in manifest:
                continue
            ref_entry = manifest[sid]
            ref = ref_entry["text"]
            hyp = rec.get("cleaned_response", "")
            wer, cer = compute_wer_cer(ref, hyp)
            sim = lexical_similarity(ref, hyp)
            label = classify(wer, sim)
            rep = detect_repetition(hyp)
            all_rows.append({
                "sample_id": sid, "source_corpus": "indicvoices",
                "model_name": model_name, "prompt_version": prompt_version,
                "reference": ref, "prediction": hyp,
                "wer": wer, "cer": cer, "lexical_similarity": round(sim, 4),
                "label": label, "repetition_flag": rep,
                "duration": ref_entry["duration"],
                "has_unintelligible_tag": "<unintelligible>" in ref,
                "inference_time_sec": rec.get("inference_time_sec"),
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows to {out_csv}")


if __name__ == "__main__":
    kathbath_configs = [
        ("whisper_results.jsonl", "Whisper-large-v3-turbo", "dedicated_asr"),
        ("indicconformer_results.jsonl", "IndicConformer-600M-multilingual", "dedicated_asr"),
        ("qwen_verbatim_results.jsonl", "Qwen2.5-Omni-7B-bnb4bit", "verbatim_v1"),
        ("qwen_normalized_results.jsonl", "Qwen2.5-Omni-7B-bnb4bit", "normalized_v1"),
    ]
    score_kathbath("meta.jsonl", "meta_valid.jsonl", kathbath_configs, "kathbath_scored_results.csv")

    indicvoices_configs = [
        ("whisper_indicvoices_results.jsonl", "Whisper-large-v3-turbo", "dedicated_asr"),
        ("indicconformer_indicvoices_results.jsonl", "IndicConformer-600M-multilingual", "dedicated_asr"),
        ("qwen_indicvoices_verbatim_results.jsonl", "Qwen2.5-Omni-7B-bnb4bit", "verbatim_v1"),
        ("qwen_indicvoices_normalized_results.jsonl", "Qwen2.5-Omni-7B-bnb4bit", "normalized_v1"),
    ]
    score_indicvoices("meta_IndicVoices.jsonl", indicvoices_configs, "indicvoices_scored_results.csv")
