from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_samples"

import pandas as pd
import numpy as np

kb = pd.read_csv(DATA / "kathbath_scored_results_labse.csv")
iv = pd.read_csv(DATA / "indicvoices_scored_results_COMPLETE_labse.csv")
iv["split"] = None

common_cols = ["sample_id","split","source_corpus","model_name","prompt_version","wer",
               "semantic_similarity"]
combined = pd.concat([kb[common_cols], iv[common_cols]], ignore_index=True)

MODEL_ORDER = [
    ("IndicConformer-600M-multilingual", "dedicated_asr"),
    ("Whisper-large-v3-turbo", "dedicated_asr"),
    ("Qwen2.5-Omni-7B-bnb4bit", "verbatim_v1"),
    ("Qwen2.5-Omni-7B-bnb4bit", "normalized_v1"),
]

def classify(wer, sim, wer_thresh, sim_thresh):
    if pd.isna(wer):
        return "EMPTY_OUTPUT"
    if wer <= wer_thresh:
        return "FAITHFUL"
    elif sim >= sim_thresh:
        return "PARAPHRASE_OR_RECOGNITION_ERROR"
    else:
        return "HALLUCINATION_CANDIDATE"

# ============================================================
# PART 1: Threshold sensitivity — does model ordering hold across similarity thresholds?
# ============================================================
print("="*100)
print("THRESHOLD SENSITIVITY ANALYSIS (WER threshold fixed at 0.5)")
print("="*100)

sim_thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

for corpus in ["kathbath", "indicvoices"]:
    print(f"\n--- {corpus} ---")
    print(f"{'Model':<35} " + "  ".join([f"sim={t:.2f}" for t in sim_thresholds]))
    for model, prompt in MODEL_ORDER:
        sub = combined[(combined["model_name"]==model) & (combined["prompt_version"]==prompt) & (combined["source_corpus"]==corpus)]
        rates = []
        for st in sim_thresholds:
            labels = sub.apply(lambda r: classify(r["wer"], r["semantic_similarity"], 0.5, st), axis=1)
            hallu_rate = (labels == "HALLUCINATION_CANDIDATE").mean()
            rates.append(f"{hallu_rate:.3f}")
        print(f"{model[:33]:<35} " + "     ".join(rates))

# ============================================================
# PART 2: Kathbath IndicConformer — train split vs validation split, separately
# ============================================================
print("\n" + "="*100)
print("KATHBATH: INDICCONFORMER — TRAIN SPLIT vs VALIDATION SPLIT (potential contamination check)")
print("="*100)

ic_kb = kb[(kb["model_name"]=="IndicConformer-600M-multilingual")]
for split_name in ["train", "valid"]:
    sub = ic_kb[ic_kb["split"] == split_name]
    if len(sub) == 0:
        continue
    n = len(sub)
    mean_wer = sub["wer"].mean()
    hallu_rate = (sub["label_semantic"]=="HALLUCINATION_CANDIDATE").mean()
    faithful_rate = (sub["label_semantic"]=="FAITHFUL").mean()
    print(f"\n{split_name} split (n={n}):")
    print(f"  Mean WER: {mean_wer:.3f}")
    print(f"  Faithful rate: {faithful_rate:.3f}")
    print(f"  Hallucination rate: {hallu_rate:.3f}")
