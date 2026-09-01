"""
Builds the IndicVoices manifest from meta.jsonl.

Unlike Kathbath, IndicVoices ships as a single file with globally unique
utterance identifiers, and its audio_path field is already correct and
absolute, so no path reconstruction is needed here.
"""

import os
import json


def load_indicvoices_manifest(meta_path, source_corpus="indicvoices"):
    samples = []
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            samples.append({
                "sample_id": rec["utt_id"],
                "source_corpus": source_corpus,
                "speaker_id": rec["speaker_id"],
                "duration": rec.get("duration"),
                "audio_path": rec["audio_path"],
                "original_transcript": rec["text"],
                "normalized_transcript": rec.get("normalized"),
                "verbatim_transcript": rec.get("verbatim"),
            })
    return samples


if __name__ == "__main__":
    meta_path = "path/to/meta_IndicVoices.jsonl"
    indicvoices_samples = load_indicvoices_manifest(meta_path)

    print(f"Total IndicVoices samples: {len(indicvoices_samples)}")

    missing = [s for s in indicvoices_samples if not os.path.exists(s["audio_path"])]
    print(f"Missing audio files: {len(missing)} / {len(indicvoices_samples)}")

    with open("indicvoices_manifest.jsonl", "w") as f:
        for s in indicvoices_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
