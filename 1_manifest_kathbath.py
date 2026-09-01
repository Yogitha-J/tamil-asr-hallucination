"""
Builds the Kathbath manifest from meta.jsonl (train + valid splits).

Handles a known data-quality issue: utterance identifiers are not globally
unique across the train and valid splits (100 identifiers collide while
referring to acoustically and lexically distinct recordings). This script
does not attempt to fix that at manifest time; see 4_score_results.py for
the positional disambiguation used downstream when scoring model outputs.
"""

import os
import json


def load_manifest(meta_path, source_corpus, split_name):
    base_dir = os.path.dirname(meta_path)
    audio_dir = os.path.join(base_dir, "audio")
    samples = []
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            utt_id = rec["utt_id"]
            samples.append({
                "sample_id": utt_id,
                "source_corpus": source_corpus,
                "split": split_name,
                "speaker_id": str(rec["speaker_id"]),
                "gender": rec.get("gender"),
                "duration": rec.get("duration"),
                # NOTE: do not trust rec["audio_path"] verbatim -- it may contain
                # a stale absolute path from the original uploader's environment.
                # Reconstruct from utt_id and the known folder structure instead.
                "audio_path": os.path.join(audio_dir, f"{utt_id}.wav"),
                "original_transcript": rec["text"],
            })
    return samples


if __name__ == "__main__":
    meta_paths = [
        ("path/to/kathbath_subset/meta.jsonl", "train"),
        ("path/to/Kathbath_subset_valid/meta.jsonl", "valid"),
    ]

    all_samples = []
    for path, split in meta_paths:
        all_samples.extend(load_manifest(path, "kathbath", split))

    print(f"Total samples: {len(all_samples)}")

    missing = [s for s in all_samples if not os.path.exists(s["audio_path"])]
    print(f"Missing audio files: {len(missing)} / {len(all_samples)}")

    with open("kathbath_manifest.jsonl", "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
